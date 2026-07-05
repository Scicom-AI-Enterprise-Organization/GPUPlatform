"""Context parallelism (sequence sharding) for Qwen3.6 — the GatedDeltaNet HYBRID.

Qwen3.6 is mostly LINEAR attention (stateful GatedDeltaNet) with periodic softmax full-attention, so
the gemma-4 zigzag ring does NOT apply. This shards ONE long packed sequence into CONTIGUOUS chunks
across a CP group and makes each layer type correct:

  * GatedDeltaNet layers  → relay the two stateful pieces (conv-state + delta-rule recurrent-state)
    across ranks via ONE differentiable pair-P2P per layer (recv at layer start, send at layer end).
    FlashQLA / causal_conv1d expose native initial_state/final_state args that propagate gradient.
  * Full-attention layers  → a position-aware ring (pure torch, so no flash-attn dependency here).

Verified: the distributed GDN state relay reproduces the full (non-CP) layer's output AND gradient
(test_gdn_cp_dist.py, rel ~1e-4). See CLAUDE.md "Context parallelism" for the design + the deadlock
fix (one pair-op per layer + a grad-requiring anchor on the recv op).

Enabled by qwen3_5.py `--cp_size N` (>1). Single-doc long-context is the target (drop seq_idx so the
conv relay works). CP is orthogonal to FSDP (params shard over all ranks; sequence shards over the CP
group; data parallelism is across CP groups).
"""
import os
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# CP context (filled by setup_cp) + per-batch state the monkeypatched kernels/attention read.
# "group" = the ring/collective comm; "relay_group" = a SEPARATE comm for the GDN state relay (see
# setup_cp — keeping the asymmetric relay P2P off the ring comm is what avoids the backward wedge).
_CP = {"group": None, "relay_group": None, "rank": 0, "world": 1, "prev": None, "next": None,
       "anchor": None, "active": False, "pos": None, "doc": None, "op": 0}

_DBG = os.environ.get("CP_DEBUG") == "1"


def _dbg(kind, peer, shape):
    """Trace every cross-rank P2P (env CP_DEBUG=1). On a hang the LAST line printed per rank pinpoints
    the op the two ranks disagreed on — cheap and decisive vs. attaching a debugger to a wedged NCCL."""
    if _DBG:
        _CP["op"] += 1
        print(f"[cp r{_CP['rank']} op{_CP['op']:04d}] {kind} peer={peer} shape={tuple(shape)}",
              file=sys.stderr, flush=True)


def cp_active():
    return _CP["active"]


def setup_cp(world_size, cp_size, global_rank, device):
    """Partition ranks into world_size/cp_size CP groups of cp_size CONSECUTIVE ranks. Returns
    (cp_group, dp_size, dp_rank, cp_rank). Every rank must call this (new_group is collective).

    ⚠ When dp==1 (cp_size == world_size) the CP group spans ALL ranks — the SAME device set FSDP's
    default process group uses. Creating a SEPARATE communicator here deadlocks against FSDP: two NCCL
    comms over the same devices, and the sequential GDN relay (rank r sends at layer-end, rank r+1
    recvs at layer-start) makes each rank enqueue the relay P2P and FSDP's all-gather/reduce-scatter in
    a DIFFERENT relative order → a cross-communicator circular wait (both GPUs spin at 100%, forward
    never returns). Reusing the DEFAULT PG (group=None) gives ONE global comm ordering → deadlock-free.
    (The single-GDN-layer test never hit this: no FSDP, hence no second communicator.)"""
    assert world_size % cp_size == 0, f"world_size {world_size} not divisible by cp_size {cp_size}"
    dp = world_size // cp_size
    # The GDN state RELAY rides a dedicated process group (relay_group), separate from the ring's group.
    # The full-attn RING keeps the CP group (default PG when dp==1). Empirically bracketed on 2×H20
    # (2026-07-05, cp_fsdp_repro.py + gateway):
    #   - relay on relay_group + ring on default PG (dp==1): step 0 fwd+bwd COMPLETES, step 1 WEDGES
    #     (FSDP's iter-1 implicit all-gather prefetch reorders vs the ring on the shared default PG).
    #   - ring ALSO on its own new_group (dp==1): the FORWARD deadlocks (a 2nd collective-comm over the
    #     same GPUs co-schedules with FSDP's forward all-gather → cross-comm circular wait). So the ring
    #     CANNOT leave the default PG at dp==1.
    # ⇒ dp==1 CP+FSDP is the OPEN hard corner (see qwen-cp-deadlock-diagnosis). The robust path is to
    #   NOT shard a FROZEN base with FSDP at all (replicate it, shard only the sequence) — no FSDP
    #   collectives to reorder against. dp>1 keeps each CP group's ring+relay on their own subgroup comms.
    if dp == 1:
        cp_group, base = None, 0                      # ring on the default PG (moving it off deadlocks fwd)
        relay_group = dist.new_group(list(range(world_size)))
    else:
        cp_group = relay_group = None; base = 0
        for d in range(dp):
            ranks = list(range(d * cp_size, (d + 1) * cp_size))
            g = dist.new_group(ranks)                 # ring comm for this CP group
            rg = dist.new_group(ranks)                # SEPARATE relay comm for this CP group
            if global_rank in ranks:
                cp_group, relay_group, base = g, rg, d * cp_size
    cp_rank = global_rank % cp_size
    _CP.update(group=cp_group, relay_group=relay_group, rank=cp_rank, world=cp_size,
               prev=(base + cp_rank - 1) if cp_rank > 0 else None,
               next=(base + cp_rank + 1) if cp_rank < cp_size - 1 else None,
               anchor=torch.zeros((), device=device, requires_grad=True),
               active=True, device=device)
    return cp_group, dp, global_rank // cp_size, cp_rank


# ========================================================================================
# differentiable cross-rank state relay — ONE pair-op per GDN layer (deadlock-free)
# ========================================================================================
# The GDN relay MUST use the same P2P mechanism as the full-attn ring (`batch_isend_irecv`), NOT plain
# `dist.isend/irecv`. Unbatched isend/irecv lazily spins up a SEPARATE 2-rank NCCL communicator; the
# ring's batched P2P uses the group's coalesced-P2P communicator. Two different comms over the same two
# devices → in BACKWARD one rank sits in a ring step (ring comm) while the other sits in a GDN send
# (P2P comm) → cross-communicator circular wait → deadlock (both GPUs spin at 100%, backward never
# returns). Routing the GDN pair through `batch_isend_irecv` on `_CP["group"]` puts GDN + ring on ONE
# comm, so their ops match by issue order (both ranks walk layers in the same reverse order in backward)
# → deadlock-free. (Forward tolerated the two comms by luck of a benign order; backward does not.)
_BARRIER = os.environ.get("CP_BARRIER") == "1"

def _drain():
    """Barrier on the CP group before a P2P so no FSDP all-gather/reduce-scatter (on the default comm,
    a SEPARATE NCCL communicator over the same GPUs) is in flight when the P2P kernel launches. Without
    this the P2P and an FSDP collective co-schedule on the GPU and circular-wait (the step-1+ deadlock;
    step 0 is masked by FSDP's lazy-init sync)."""
    if _BARRIER:
        dist.barrier(group=_CP["group"])

def _send_pair(a, b, dst):
    _dbg("SEND2", dst, a.shape)
    _drain()
    g = _CP["relay_group"]                            # dedicated relay comm (NOT the ring's group)
    for r in dist.batch_isend_irecv([dist.P2POp(dist.isend, a.contiguous(), dst, group=g),
                                     dist.P2POp(dist.isend, b.contiguous(), dst, group=g)]):
        r.wait()

def _recv_pair(sa, da, sb, db, src):
    _dbg("RECV2", src, sa)
    _drain()
    g = _CP["relay_group"]                            # dedicated relay comm (NOT the ring's group)
    a = torch.empty(sa, dtype=da, device=_CP["device"])
    b = torch.empty(sb, dtype=db, device=_CP["device"])
    for r in dist.batch_isend_irecv([dist.P2POp(dist.irecv, a, src, group=g),
                                     dist.P2POp(dist.irecv, b, src, group=g)]):
        r.wait()
    return a, b

class _SendPairToNext(torch.autograd.Function):
    # `anchor` (the rank-local grad-requiring scalar) is an input so this Function's backward ALWAYS
    # fires — exactly like _RecvPairFromPrev below. Without it, autograd PRUNES the send-backward on
    # layers whose upstream is entirely FROZEN (LoRA trains only the full-attn q/k/v/o, so every GDN
    # layer BELOW the first LoRA layer on cp_rank 0 has no grad-requiring input): the sender then never
    # posts its grad-RECV while the receiver's grad-SEND (anchored) still fires → the rank pair's relay
    # op COUNTS diverge → NCCL (which matches P2P purely by order, no tags) pairs later ops with the
    # WRONG payloads and eventually wedges. THE root cause of the qwen CP training deadlock: step 0's
    # leftover unmatched sends silently corrupt step 1's grads, then the queues jam (found via the
    # CP_DEBUG op-count mismatch: r0=50 ops vs r1=56). The anchor guarantees every rank runs the same
    # relay backward ops in the same reverse-layer order. (The received grad flows into frozen-only
    # paths and is dropped by autograd — numerically a no-op, but the COMM must still happen.)
    @staticmethod
    def forward(ctx, a, b, anchor, dst):
        ctx.dst = dst; ctx.sa, ctx.da, ctx.sb, ctx.db = a.shape, a.dtype, b.shape, b.dtype
        _send_pair(a, b, dst)
        # tie the output to the anchor so it requires grad even when a/b are frozen-derived; cast to
        # a's dtype so `o + <this>` does NOT type-promote the layer output (anchor is fp32).
        return (anchor * 0).to(a.dtype)
    @staticmethod
    def backward(ctx, _g):
        ga, gb = _recv_pair(ctx.sa, ctx.da, ctx.sb, ctx.db, ctx.dst)
        return ga, gb, None, None

class _RecvPairFromPrev(torch.autograd.Function):
    # `anchor` (a grad-requiring scalar) forces the outputs to require grad, else this Function's
    # backward is never called (all other inputs are non-tensors) and the sender hangs on recv.
    @staticmethod
    def forward(ctx, anchor, sa, da, sb, db, src):
        ctx.src = src; ctx.sa, ctx.da, ctx.sb, ctx.db = sa, da, sb, db
        return _recv_pair(sa, da, sb, db, src)
    @staticmethod
    def backward(ctx, ga, gb):
        _send_pair(ga, gb, ctx.src)
        return None, None, None, None, None, None


def install_gdn_cp(model, gdn_class):
    """Monkeypatch every GatedDeltaNet layer so its conv + delta-rule kernels relay state across the
    CP group. One combined pair-relay per layer: conv_wrap (start) RECVs both states, gdr_wrap (end)
    SENDs both — a fixed, per-layer, deadlock-free comm order."""
    for layer in model.modules():
        if isinstance(layer, gdn_class):
            _patch_gdn_layer(layer)


def _patch_gdn_layer(layer):
    _conv, _gdr = layer.causal_conv1d_fn, layer.chunk_gated_delta_rule
    st = {"rec_init": None, "conv_final": None}
    rank, W = _CP["rank"], _CP["world"]

    def conv_wrap(x, weight=None, bias=None, seq_idx=None, initial_states=None,
                  return_final_states=False, final_states_out=None, activation=None):
        B, D, _ = x.shape
        k1 = layer.conv_kernel_size - 1
        conv_init = st["rec_init"] = None
        if _CP["prev"] is not None:
            conv_buf, st["rec_init"] = _RecvPairFromPrev.apply(
                _CP["anchor"], (B, k1, D), x.dtype, layer._cp_rec_shape, layer._cp_rec_dtype, _CP["prev"])
            conv_init = conv_buf.transpose(1, 2)          # (B,D,k1), stride(1)==1 for the kernel
        out, fin = _conv(x, weight, bias, activation=activation,
                         initial_states=conv_init, return_final_states=True)
        st["conv_final"] = fin.transpose(1, 2).contiguous()
        return out

    def gdr_wrap(q, k, v, g=None, beta=None, initial_state=None, output_final_state=False,
                 use_qk_l2norm_in_kernel=False, cu_seqlens=None):
        Hv, Kd, Vd = v.shape[2], q.shape[3], v.shape[3]
        layer._cp_rec_shape = (1, Hv, Kd, Vd)                 # the relayed unit is ONE doc's state
        n_docs = (cu_seqlens.numel() - 1) if cu_seqlens is not None else 1
        # Recurrent state is PER-DOC ([n_docs, Hv, K, V], fp32). ALWAYS pass a REAL tensor (never None):
        # the kernel's backward returns dh0 for `initial_state`, and torch>=2.10 raises if that forward
        # input was None (not a Variable). Seed doc 0 with the state relayed from the previous rank IFF
        # this chunk's first doc continues across the boundary; every other doc (and the whole tensor
        # when not a continuation, e.g. rank 0) starts at ZERO — numerically identical to None but a
        # valid Variable for the dh0 grad. (The relayed unit is always one doc's state, [1, ...].)
        def _zeros(n):
            return torch.zeros(n, Hv, Kd, Vd, dtype=torch.float32, device=q.device)
        if _CP.get("first_cont") and st["rec_init"] is not None:
            rec0 = st["rec_init"].to(torch.float32)          # [1, Hv, K, V], carries grad -> prev rank
            init = rec0 if n_docs == 1 else torch.cat([rec0, _zeros(n_docs - 1)], dim=0)
        else:
            init = _zeros(n_docs)                            # fresh start; dh0 grad is unused/ignored
        o, fs = _gdr(q, k, v, g=g, beta=beta, initial_state=init, output_final_state=True,
                     use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens)
        layer._cp_rec_dtype = fs.dtype
        if _CP["next"] is not None:
            last = fs[-1:].contiguous()                       # last doc's final state -> next rank
            o = o + _SendPairToNext.apply(st["conv_final"], last, _CP["anchor"], _CP["next"])
        return o, fs

    layer._cp_rec_shape = (1, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim)
    layer._cp_rec_dtype = torch.float32
    layer.causal_conv1d_fn = conv_wrap
    layer.chunk_gated_delta_rule = gdr_wrap


# ========================================================================================
# full-attention ring (pure torch, position-aware) for the softmax layers
# ========================================================================================
class RingComm:
    def __init__(self, group):
        self._pg = group; self._ops = []; self._reqs = None
        self.rank = dist.get_rank(group); self.world_size = dist.get_world_size(group)
        # group=None is the default PG (dp==1): group-rank == global-rank (identity), so the neighbour's
        # global rank is just (rank±1)%W. get_global_rank() rejects None, so compute it directly there.
        if group is None:
            self.send_rank = (self.rank + 1) % self.world_size
            self.recv_rank = (self.rank - 1) % self.world_size
        else:
            self.send_rank = dist.get_global_rank(group, (self.rank + 1) % self.world_size)
            self.recv_rank = dist.get_global_rank(group, (self.rank - 1) % self.world_size)

    def send_recv(self, to_send, recv=None):
        to_send = to_send.contiguous()
        res = torch.empty_like(to_send) if recv is None else recv
        _dbg("RING", f"{self.send_rank}/{self.recv_rank}", to_send.shape)
        self._ops += [dist.P2POp(dist.isend, to_send, self.send_rank, group=self._pg),
                      dist.P2POp(dist.irecv, res, self.recv_rank, group=self._pg)]
        return res

    def commit(self):
        _drain()
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        for r in self._reqs:
            r.wait()
        self._reqs = None; self._ops = []


def _upd(out, lse, blk_out, blk_lse):
    blk_out = blk_out.to(torch.float32)
    blk_lse = blk_lse.transpose(-2, -1).unsqueeze(-1)
    if out is None:
        return blk_out, blk_lse
    out = out - F.sigmoid(blk_lse - lse) * (out - blk_out)
    lse = lse - F.logsigmoid(lse - blk_lse)
    return out, lse


def _mask(qp, qd, kp, kd, window):
    keep = (qd[:, None] == kd[None, :]) & (kp[None, :] <= qp[:, None])
    if window is not None:
        keep &= (qp[:, None] - kp[None, :]) <= window
    return keep


# Long-context knobs (the fp32/qb=1024 defaults are fine to ~16k tokens/rank; at 128k+/rank the fp32
# score matrices are both the memory (24×1024×131k×4B ≈ 13GB/block) and the speed (no tensor cores)
# wall). bf16 matmuls (cuBLAS accumulates fp32 internally; only the score ROUNDING is bf16 — the same
# trade fused flash kernels make) are ~4× faster on H20/H100 and halve transients. SGPU_RING_FP32=1
# restores full-fp32 scores; SGPU_RING_QB tunes the q-block (smaller = smaller transients).
_RING_QB = int(os.environ.get("SGPU_RING_QB", "1024") or 1024)
_RING_FP32 = os.environ.get("SGPU_RING_FP32") == "1"
# SGPU_RING_FUSED=1: route the full-attn ring blocks through the FUSED FA3 kernel
# (kernels-community/flash-attn3 — the same kernel the non-CP path uses) instead of the pure-torch
# blocks: O(S) memory, tensor-core speed. The ring/comm structure is unchanged; only the per-step
# block math is swapped. Requires the kernel to be resolvable (hub-cached) in the venv.
_RING_FUSED = os.environ.get("SGPU_RING_FUSED") == "1"
_FA3I = None


def _fa3():
    """The FA3 low-level interface (lazy; cached by the `kernels` hub the first time)."""
    global _FA3I
    if _FA3I is None:
        from kernels import get_kernel
        _FA3I = get_kernel("kernels-community/flash-attn3").flash_attn_interface
    return _FA3I


def _cols(qp_blk, qd_blk, kp, kd, window):
    """Per-q-block K-column pre-selection: causal (kp ≤ max qp), inside the window (if any), and —
    the long-context lever on MULTIPACKED bins — belonging to one of the q-block's DOCS (attention
    never crosses docs, so a ~27k-doc column slice replaces the full 131k chunk). Conservative
    superset; the exact _mask still applies. Returns a LongTensor index (possibly empty)."""
    colm = kp <= int(qp_blk.max())
    if window is not None:
        colm &= kp >= (int(qp_blk.min()) - window)
    colm &= torch.isin(kd, torch.unique(qd_blk))
    return colm.nonzero(as_tuple=False).squeeze(1)


def _fwd_blk(q, k, v, qp, qd, kp, kd, scale, window, qb=None):
    qb = qb or _RING_QB
    Tq, hq, d = q.shape; Tk, hkv, dv = v.shape; g = hq // hkv
    cd = torch.float32 if _RING_FP32 else q.dtype     # score-matmul compute dtype
    kt = k.transpose(0, 1); vt = v.transpose(0, 1)    # (hkv,Tk,*) — GQA-expand AFTER column select
    out = torch.empty(Tq, hq, dv, dtype=torch.float32, device=q.device)
    lse = torch.empty(hq, Tq, dtype=torch.float32, device=q.device)
    for s0 in range(0, Tq, qb):
        s1 = min(s0 + qb, Tq)
        idx = _cols(qp[s0:s1], qd[s0:s1], kp, kd, window)
        if idx.numel() == 0:
            out[s0:s1] = 0.0                          # no reachable columns: identity contribution
            lse[:, s0:s1] = float("-inf")
            continue
        kb = kt.index_select(1, idx); vb = vt.index_select(1, idx)
        kf = (kb.repeat_interleave(g, 0) if g != 1 else kb).to(cd)   # (hq,|idx|,d)
        vf = (vb.repeat_interleave(g, 0) if g != 1 else vb).to(cd)
        keep = _mask(qp[s0:s1], qd[s0:s1], kp[idx], kd[idx], window)
        sc = torch.matmul(q[s0:s1].transpose(0, 1).to(cd), kf.transpose(-1, -2)).float() * scale
        sc = sc.masked_fill(~keep.unsqueeze(0), float("-inf"))
        m = sc.amax(-1, keepdim=True); m = torch.where(torch.isneginf(m), torch.zeros_like(m), m)
        p = torch.exp(sc - m); den = p.sum(-1, keepdim=True)
        out[s0:s1] = (torch.matmul(p.to(cd), vf).float() / den.clamp(min=1e-20)).transpose(0, 1)
        lb = (m + torch.log(den.clamp(min=1e-20))).squeeze(-1)
        lse[:, s0:s1] = torch.where(den.squeeze(-1) == 0, torch.full_like(lb, float("-inf")), lb)
    return out.to(q.dtype), lse


def _bwd_blk(dout, q, k, v, out, lse, qp, qd, kp, kd, scale, window, qb=None):
    qb = qb or _RING_QB
    Tq, hq, d = q.shape; Tk, hkv, dv = v.shape; g = hq // hkv
    cd = torch.float32 if _RING_FP32 else q.dtype
    kt = k.transpose(0, 1); vt = v.transpose(0, 1)    # (hkv,Tk,*) — GQA-expand AFTER column select
    of = out.float().transpose(0, 1); dof = dout.float().transpose(0, 1)
    delta = (dof * of).sum(-1)
    dq = torch.zeros(hq, Tq, d, dtype=torch.float32, device=q.device)
    # dk/dv accumulate at KV-HEAD granularity (hkv, Tk, ·) — the per-block g-sum happens before the
    # index_add_, so the full-width fp32 buffers are g× smaller (0.5GB vs 3.2GB at 131k tokens).
    dk_e = torch.zeros(hkv, Tk, d, dtype=torch.float32, device=q.device)
    dv_e = torch.zeros(hkv, Tk, dv, dtype=torch.float32, device=q.device)
    for s0 in range(0, Tq, qb):
        s1 = min(s0 + qb, Tq)
        idx = _cols(qp[s0:s1], qd[s0:s1], kp, kd, window)
        if idx.numel() == 0:
            continue                                   # no reachable columns: zero grads
        kb = kt.index_select(1, idx); vb = vt.index_select(1, idx)
        kf = (kb.repeat_interleave(g, 0) if g != 1 else kb).to(cd)   # (hq,|idx|,d)
        vf = (vb.repeat_interleave(g, 0) if g != 1 else vb).to(cd)
        keep = _mask(qp[s0:s1], qd[s0:s1], kp[idx], kd[idx], window)
        qbl = q[s0:s1].transpose(0, 1).to(cd)
        sc = torch.matmul(qbl, kf.transpose(-1, -2)).float() * scale
        sc = sc.masked_fill(~keep.unsqueeze(0), float("-inf"))
        p = torch.exp(sc - lse[:, s0:s1].unsqueeze(-1))
        dob = dof[:, s0:s1]
        ds = (p * (torch.matmul(dob.to(cd), vf.transpose(-1, -2)).float()
                   - delta[:, s0:s1].unsqueeze(-1)) * scale)
        bdv = torch.matmul(p.to(cd).transpose(-1, -2), dob.to(cd)).float()   # (hq,|idx|,dv)
        bdk = torch.matmul(ds.to(cd).transpose(-1, -2), qbl).float()         # (hq,|idx|,d)
        if g != 1:
            bdv = bdv.view(hkv, g, -1, dv).sum(1); bdk = bdk.view(hkv, g, -1, d).sum(1)
        dv_e.index_add_(1, idx, bdv)
        dk_e.index_add_(1, idx, bdk)
        dq[:, s0:s1] = torch.matmul(ds.to(cd), kf).float()
    return (dq.transpose(0, 1).to(q.dtype), dk_e.transpose(0, 1).to(q.dtype), dv_e.transpose(0, 1).to(q.dtype))


def _gather_posdoc(pos, doc, W, group):
    pl = [torch.empty_like(pos) for _ in range(W)]; dl = [torch.empty_like(doc) for _ in range(W)]
    dist.all_gather(pl, pos.contiguous(), group=group); dist.all_gather(dl, doc.contiguous(), group=group)
    return pl, dl


def _skip_step(pos, all_pos_src, window):
    """True when the (local-q, src-KV) pair can contribute NOTHING: every src key is causally after
    every local query (contiguous sharding → whole later chunks), or — with a sliding window — every
    src key is entirely outside the window. Compute-only decision (each rank skips its own blocks);
    the ring COMM still rotates every step on every rank, so the comm pattern stays symmetric."""
    if int(all_pos_src.min()) > int(pos.max()):
        return True                                    # all keys after all queries → fully non-causal
    if window is not None and int(pos.min()) - int(all_pos_src.max()) > window:
        return True                                    # all keys before the window
    return False


def _ring_fwd(group, q, k, v, pos, doc, scale, window):
    comm = RingComm(group); W, rank = comm.world_size, comm.rank
    all_pos, all_doc = _gather_posdoc(pos, doc, W, group)
    out = lse = None; nk = nv = None
    for step in range(W):
        if step + 1 != W:
            nk = comm.send_recv(k); nv = comm.send_recv(v); comm.commit()
        src = (rank - step) % W
        # skip compute for chunks that can't contribute (step 0 = the diagonal, never skipped, so
        # `out` is always initialized before any skippable step).
        if not _skip_step(pos, all_pos[src], window):
            bo, bl = _fwd_blk(q, k, v, pos, doc, all_pos[src], all_doc[src], scale, window)
            out, lse = _upd(out, lse, bo, bl)
        if step + 1 != W:
            comm.wait(); k, v = nk, nv
    return out.to(q.dtype), lse.squeeze(-1).transpose(0, 1).contiguous()


def _ring_bwd(group, dout, q, k, v, out, lse, pos, doc, scale, window):
    kv, dkv = RingComm(group), RingComm(group); W, rank = kv.world_size, kv.rank
    all_pos, all_doc = _gather_posdoc(pos, doc, W, group)
    dq = dk = dv = None; ndk = ndv = None; nk = nv = None
    for step in range(W):
        if step + 1 != W:
            nk = kv.send_recv(k); nv = kv.send_recv(v); kv.commit()
        src = (rank - step) % W
        if step > 0 and _skip_step(pos, all_pos[src], window):
            # non-contributing chunk: this KV's grad is untouched — pass the accumulated dk/dv
            # straight through the dkv ring (dk = 0 + ndk). Step 0 (diagonal) is never skipped.
            dkv.wait(); dk, dv = ndk, ndv
        else:
            bdq, bdk, bdv = _bwd_blk(dout, q, k, v, out, lse, pos, doc, all_pos[src], all_doc[src], scale, window)
            if dq is None:
                dq, dk, dv = bdq.float(), bdk.float(), bdv.float()
            else:
                dq += bdq; dkv.wait(); dk = bdk.float() + ndk; dv = bdv.float() + ndv
        if step + 1 != W:
            kv.wait(); k, v = nk, nv
        ndk = dkv.send_recv(dk); ndv = dkv.send_recv(dv); dkv.commit()
    dkv.wait()
    return dq.to(q.dtype), ndk.to(q.dtype), ndv.to(q.dtype)


# ---- FUSED FA3 ring blocks (SGPU_RING_FUSED=1) --------------------------------------------------
# Contiguous causal sharding makes the fused decomposition tiny: per ring step at most ONE kernel
# call — the diagonal (src == rank) is a plain varlen-causal call over the local doc segments; an
# EARLIER chunk (src < rank) can only contribute through the single doc that straddles from src's
# chunk into this rank's chunk start (docs are contiguous), i.e. one causal=False block of
# (my-first-doc rows × that doc's cols in src); LATER chunks contribute nothing. The backward feeds
# the GLOBAL lse into FA3's `_flash_attn_backward` — the standard ring-flash-attention recipe (the
# hd≤256 backward consumes the passed lse; only hd>256 paths recompute, see the gemma notes).

def _cu2(n, dev):
    return torch.tensor([0, n], dtype=torch.int32, device=dev)


def _doc_cu(doc):
    """Local cu_seqlens (int32) from the chunk's contiguous per-token doc ids."""
    d = doc
    bounds = [0] + (torch.nonzero(d[1:] != d[:-1]).squeeze(1) + 1).tolist() + [d.numel()]
    return torch.tensor(bounds, dtype=torch.int32, device=d.device)


def _fa3_fwd(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal):
    fi = _fa3()
    out, lse, *_ = fi._flash_attn_forward(
        q, k, v, None, None, None, None, cu_q, cu_k, None, None, None, max_q, max_k,
        None, None, None, None, None, None, None, None, None, scale,
        causal=causal, window_size_left=-1, window_size_right=-1,
        attention_chunk=0, softcap=0.0, num_splits=1, pack_gqa=None, sm_margin=0)
    return out, lse


def _fa3_bwd(dout, q, k, v, out, lse, cu_q, cu_k, max_q, max_k, scale, causal):
    fi = _fa3()
    dq = torch.empty_like(q); dk = torch.empty_like(k); dv = torch.empty_like(v)
    fi._flash_attn_backward(
        dout, q, k, v, out, lse, cu_q, cu_k, None, None, max_q, max_k, dq, dk, dv,
        scale, causal=causal, window_size_left=-1, window_size_right=-1,
        softcap=0.0, deterministic=False, sm_margin=0)
    return dq, dk, dv


def _prev_doc_span(doc, all_doc_src):
    """(qn, ks, ke): rows of MY chunk's first doc + that doc's contiguous span in the src chunk —
    the only possible contribution from an earlier chunk. None when the doc doesn't reach back."""
    d0 = doc[0]
    hit = torch.nonzero(all_doc_src == d0)
    if hit.numel() == 0:
        return None
    qn = int((doc == d0).sum())
    return qn, int(hit[0]), int(hit[-1]) + 1


def _ring_fwd_fused(group, q, k, v, pos, doc, scale):
    comm = RingComm(group); W, rank = comm.world_size, comm.rank
    all_pos, all_doc = _gather_posdoc(pos, doc, W, group)
    cu_loc = _doc_cu(doc)
    max_loc = int((cu_loc[1:] - cu_loc[:-1]).max())
    Tq, hq, _ = q.shape; dv_dim = v.shape[2]
    out = lse = None; nk = nv = None
    for step in range(W):
        if step + 1 != W:
            nk = comm.send_recv(k); nv = comm.send_recv(v); comm.commit()
        src = (rank - step) % W
        bo = bl = None
        if src == rank:
            o_, l_ = _fa3_fwd(q, k, v, cu_loc, cu_loc, max_loc, max_loc, scale, True)
            bo, bl = o_, l_
        elif src < rank:
            span = _prev_doc_span(doc, all_doc[src])
            if span is not None:
                qn, ks, ke = span
                o_, l_ = _fa3_fwd(q[:qn], k[ks:ke].contiguous(), v[ks:ke].contiguous(),
                                  _cu2(qn, q.device), _cu2(ke - ks, q.device), qn, ke - ks, scale, False)
                bo = q.new_zeros(Tq, hq, dv_dim); bo[:qn] = o_
                bl = torch.full((hq, Tq), float("-inf"), dtype=torch.float32, device=q.device)
                bl[:, :qn] = l_
        if bo is not None:
            out, lse = _upd(out, lse, bo, bl)
        if step + 1 != W:
            comm.wait(); k, v = nk, nv
    return out.to(q.dtype), lse.squeeze(-1).transpose(0, 1).contiguous()


def _ring_bwd_fused(group, dout, q, k, v, out, lse, pos, doc, scale):
    kv, dkv = RingComm(group), RingComm(group); W, rank = kv.world_size, kv.rank
    all_pos, all_doc = _gather_posdoc(pos, doc, W, group)
    cu_loc = _doc_cu(doc)
    max_loc = int((cu_loc[1:] - cu_loc[:-1]).max())
    dq = dk = dv = None; ndk = ndv = None; nk = nv = None
    lse_c = lse.contiguous()
    for step in range(W):
        if step + 1 != W:
            nk = kv.send_recv(k); nv = kv.send_recv(v); kv.commit()
        src = (rank - step) % W
        if src == rank:
            bdq, bdk, bdv = _fa3_bwd(dout, q, k, v, out, lse_c, cu_loc, cu_loc,
                                     max_loc, max_loc, scale, True)
            dq, dk, dv = bdq.float(), bdk.float(), bdv.float()   # step 0 = diagonal: initialize
        else:
            span = _prev_doc_span(doc, all_doc[src]) if src < rank else None
            if span is None:
                dkv.wait(); dk, dv = ndk, ndv                     # nothing to add: pass through
            else:
                qn, ks, ke = span
                bdq, bdk, bdv = _fa3_bwd(
                    dout[:qn].contiguous(), q[:qn].contiguous(),
                    k[ks:ke].contiguous(), v[ks:ke].contiguous(),
                    out[:qn].contiguous(), lse_c[:, :qn].contiguous(),
                    _cu2(qn, q.device), _cu2(ke - ks, q.device), qn, ke - ks, scale, False)
                dq[:qn] += bdq.float()
                dkv.wait()
                dk, dv = ndk.clone(), ndv.clone()
                dk[ks:ke] += bdk.float(); dv[ks:ke] += bdv.float()
        if step + 1 != W:
            kv.wait(); k, v = nk, nv
        ndk = dkv.send_recv(dk); ndv = dkv.send_recv(dv); dkv.commit()
    dkv.wait()
    return dq.to(q.dtype), ndk.to(q.dtype), ndv.to(q.dtype)


class _RingAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos, doc, scale, window, group):
        k, v = k.contiguous(), v.contiguous()
        pos, doc = pos.long(), doc.long()
        # fused FA3 path: full-causal only (qwen full-attn has no sliding window)
        fused = _RING_FUSED and window is None
        if fused:
            out, lse = _ring_fwd_fused(group, q, k, v, pos, doc, scale)
        else:
            out, lse = _ring_fwd(group, q, k, v, pos, doc, scale, window)
        ctx.save_for_backward(q, k, v, out, lse, pos, doc)
        ctx.scale, ctx.window, ctx.group, ctx.fused = scale, window, group, fused
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, pos, doc = ctx.saved_tensors
        if ctx.fused:
            # saved lse is (hq, Tq) — already FA3's varlen softmax_lse layout
            dq, dk, dv = _ring_bwd_fused(ctx.group, dout.contiguous(), q, k, v, out,
                                         lse, pos, doc, ctx.scale)
        else:
            dq, dk, dv = _ring_bwd(ctx.group, dout, q, k, v, out, lse, pos, doc, ctx.scale, ctx.window)
        return dq, dk, dv, None, None, None, None, None


def cp_full_attention(module, query, key, value, attention_mask,
                      cu_seq_lens_q=None, cu_seq_lens_k=None, max_length_q=None, max_length_k=None,
                      sliding_window=None, scaling=None, **kwargs):
    """AttentionInterface backend for Qwen full-attention layers under CP: a contiguous ring over the
    CP group. q/k/v: (B=1, H, S_local, D) -> packed (S_local, H, D)."""
    q = query.permute(0, 2, 1, 3).squeeze(0)
    k = key.permute(0, 2, 1, 3).squeeze(0)
    v = value.permute(0, 2, 1, 3).squeeze(0)
    if scaling is None:
        scaling = q.shape[-1] ** -0.5
    window = int(sliding_window) if sliding_window is not None else None
    out = _RingAttn.apply(q, k, v, _CP["pos"], _CP["doc"], scaling, window, _CP["group"])
    return out.unsqueeze(0), None


# ========================================================================================
# contiguous sequence sharding + causal-LM target pre-shift
# ========================================================================================
def shard_batch(batch, cp_size, cp_rank, pad_id=0):
    """This CP rank's CONTIGUOUS chunk of a collated full batch, + LOCAL cu_seqlens, + per-token
    (position_id, doc_id) for the full-attn ring (set on _CP). Labels are pre-shifted to per-token
    NEXT-token targets (the loss then aligns hidden↔labels 1:1) — the usual global hidden[:-1]/
    labels[1:] shift is invalid across rank boundaries under sharding. The packed sequence total is
    padded to a multiple of cp_size so chunks are equal."""
    ids = batch["input_ids"][0]; pos = batch["position_ids"][0]; lab = batch["labels"][0]
    cu = batch["cu_seq_lens_q"].tolist()
    dev = ids.device
    # per-token doc id from cu_seqlens
    doc = torch.cat([torch.full((cu[i + 1] - cu[i],), i, dtype=torch.long, device=dev)
                     for i in range(len(cu) - 1)]) if len(cu) > 1 else torch.zeros_like(ids)
    # causal-LM target = next label per doc (doc-end -> -100)
    tgt = lab.clone()
    tgt[:-1] = lab[1:]
    for i in range(1, len(cu) - 1):
        tgt[cu[i] - 1] = -100                       # last token of each doc has no in-doc next
    tgt[-1] = -100
    S = ids.shape[0]
    pad = (-S) % cp_size
    if pad:
        ids = torch.cat([ids, torch.full((pad,), pad_id, dtype=ids.dtype, device=dev)])
        pos = torch.cat([pos, torch.arange(int(pos[-1]) + 1, int(pos[-1]) + 1 + pad, dtype=pos.dtype, device=dev)])
        tgt = torch.cat([tgt, torch.full((pad,), -100, dtype=tgt.dtype, device=dev)])
        doc = torch.cat([doc, torch.full((pad,), int(doc[-1]) + 1, dtype=torch.long, device=dev)])
    c = (S + pad) // cp_size
    sl = slice(cp_rank * c, (cp_rank + 1) * c)
    li, lp, lt, ld = ids[sl], pos[sl], tgt[sl], doc[sl]
    _CP["pos"], _CP["doc"] = lp.long(), ld.long()
    # Does this rank's first doc CONTINUE the previous rank's last doc (chunk boundary mid-document)?
    # If so the GDN recurrent state relayed from rank-1 seeds this chunk's first doc; else it starts
    # fresh. (The conv is whole-chunk either way — matches the non-CP trainer, which passes no seq_idx.)
    _CP["first_cont"] = bool(cp_rank > 0 and int(doc[cp_rank * c]) == int(doc[cp_rank * c - 1]))
    # local cu_seqlens for this contiguous chunk (split doc-segments at the chunk's own boundaries)
    seg = []
    d0 = int(ld[0]); start = 0
    for j in range(1, c):
        if int(ld[j]) != d0:
            seg.append(j - start); start = j; d0 = int(ld[j])
    seg.append(c - start)
    cu_loc = torch.tensor([0] + np.cumsum(seg).tolist(), dtype=torch.int32, device=dev)
    return {
        "input_ids": li.unsqueeze(0),
        "position_ids": lp.unsqueeze(0),
        "labels": lt.unsqueeze(0),
        "attention_mask": None,
        "mm_token_type_ids": torch.zeros_like(li).unsqueeze(0),
        "cu_seq_lens_q": cu_loc,
        "cu_seq_lens_k": cu_loc,
        "max_length_q": int(max(seg)),
        "max_length_k": int(max(seg)),
    }
