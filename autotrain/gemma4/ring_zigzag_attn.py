"""Zigzag ring (context-parallel) FlashAttention for the head_dim-512 FA4 cute fork.

Context parallelism shards ONE long packed sequence across `world_size` GPUs and computes
exact attention by ringing the K/V blocks around the ranks, combining the per-block partial
outputs with the online-softmax LSE recurrence. "Zigzag" is the causal load-balancing layout:
the sequence is split into `2*world_size` equal chunks and rank r holds chunk r AND chunk
(2*world_size-1-r), so every rank does the same amount of causal work.

This is the FA4-cute analogue of
  https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/zigzag_ring_flash_attn_varlen.py
The ring control-flow (which chunks attend which, the dK/dV ring reduction) is reproduced
verbatim from that reference — only the per-block forward/backward primitives are swapped:

  * forward  → `flash_attn.cute.interface._flash_attn_fwd(..., return_lse=True)`, which returns
    (out, lse) with lse = ln(sum_j exp(scale * S_ij)) (natural log, scale folded in — verified
    against the fork's softmax.py epilogue). That is exactly the convention the reference's
    `update_out_and_lse` online-softmax combine assumes.

  * backward → a recompute block (`_ring_bwd`) that consumes the GLOBAL lse produced by the ring
    forward: p_ij = exp(scale*S_ij - LSE_i). The fork's own head_dim>256 backward
    (`_flash_attn_bwd_large_headdim`) re-softmaxes each block LOCALLY and refuses an external
    lse, so it CANNOT be used for the ring's partial (cross-rank) key blocks — those need the
    global normalisation. `_ring_bwd` is `_bwd_large_headdim_block`'s math with softmax(s)
    replaced by exp(s - lse), which reduces to the fork's backward when the block is the whole
    key axis, and is the correct per-block gradient contribution otherwise.

Correctness contract (see test_ring_attn.py): for the same q/k/v/dout, the ring out and the
dq/dk/dv gradients must equal the single-GPU (non-ring) FA4 result up to bf16 noise.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F

from flash_attn.cute.interface import _flash_attn_fwd


# ----------------------------------------------------------------------------------------
# online-softmax combine + ring comm (adapted from ring_flash_attn/utils.py)
# ----------------------------------------------------------------------------------------
def _update_out_and_lse(out, lse, block_out, block_lse):
    # out: (T, H, D) fp32 ; lse: (T, H, 1) fp32
    # block_out: (T, H, D) ; block_lse: (H, T)  (the cute-kernel varlen lse layout)
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)  # (H,T) -> (T,H,1)
    # O <- O - sigmoid(l_b - l) * (O - O_b) ; l <- l - logsigmoid(l - l_b)
    # (numerically-stable rewrite of the two-way flash-attention rescale; see
    #  ring-flash-attention PR #34)
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse


def update_out_and_lse(out, lse, block_out, block_lse, slice_=None):
    if out is None:
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        so, sl = out[slice_], lse[slice_]
        so, sl = _update_out_and_lse(so, sl, block_out, block_lse)
        out[slice_], lse[slice_] = so, sl
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


class RingComm:
    """Send to (rank+1), recv from (rank-1) — a single ring rotation of K/V per step."""

    def __init__(self, process_group):
        self._pg = process_group
        self._ops = []
        self._reqs = None
        self.rank = dist.get_rank(self._pg)
        self.world_size = dist.get_world_size(self._pg)
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size
        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._pg, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._pg, self.recv_rank)

    def send_recv(self, to_send, recv_tensor=None):
        # CONTIGUIFY FIRST: the sender flattens to C-order (isend below), so the recv buffer must
        # also be C-order-contiguous. torch.empty_like PRESERVES the input's strides, so
        # empty_like(a_non_contiguous_tensor) is itself non-contiguous → NCCL would write the flat
        # payload into a strided buffer and silently PERMUTE it. (This bit dk/dv, which arrive here
        # as .transpose(0,1) views; out/dq are k-contractions so a k-permutation was invisible there
        # — the bug only showed in the k-indexed gradients.)
        to_send = to_send.contiguous()
        res = torch.empty_like(to_send) if recv_tensor is None else recv_tensor
        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._pg)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._pg)
        self._ops += [send_op, recv_op]
        return res

    def commit(self):
        assert self._reqs is None, "commit called twice"
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        assert self._reqs is not None, "wait before commit"
        for r in self._reqs:
            r.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(self, k, v, k_buffer=None, v_buffer=None):
        next_k = self.send_recv(k, k_buffer)
        next_v = self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v


# ----------------------------------------------------------------------------------------
# zigzag half-index helpers (adapted from the reference; front = first half per document)
# ----------------------------------------------------------------------------------------
def get_half_index(cu_seqlens, *, front: bool):
    if len(cu_seqlens) == 2:  # single document: a plain slice, no gather
        half = int(cu_seqlens[-1]) // 2
        return slice(None, half) if front else slice(half, None)
    index = torch.zeros((int(cu_seqlens[-1]),), dtype=torch.bool)
    for i in range(len(cu_seqlens) - 1):
        start, end = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        if front:
            end = (start + end) // 2
        else:
            start = (start + end) // 2
        index[start:end] = True
    return index


def get_half_lse(lse, cu_seqlens, *, front: bool):
    """Extract the front/back half of each document from a (H, total_q) lse."""
    new_lse = torch.empty(
        (lse.shape[0], lse.shape[1] // 2), dtype=lse.dtype, device=lse.device
    )
    for i in range(len(cu_seqlens) - 1):
        start, end = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        new_start, new_end = start // 2, end // 2
        if front:
            end -= (end - start) // 2
        else:
            start += (end - start) // 2
        new_lse[:, new_start:new_end] = lse[:, start:end]
    return new_lse


# ----------------------------------------------------------------------------------------
# per-block forward / backward primitives (FA4 cute fwd ; torch recompute bwd w/ external lse)
# ----------------------------------------------------------------------------------------
def _fa4_fwd(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal, window_size):
    """One attention block via the FA4 cute kernel. Returns (out (Tq,H,Dv), lse (H,Tq))."""
    out, lse = _flash_attn_fwd(
        q, k, v,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=max_q, max_seqlen_k=max_k,
        softmax_scale=scale, causal=causal,
        window_size_left=window_size[0], window_size_right=window_size[1],
        return_lse=True,
    )
    return out, lse


def _ring_bwd(dout, q, k, v, out, lse, scale, causal, window_size, cu_q, cu_k,
              dq_buf, dk_buf, dv_buf, q_block=1024):
    """Exact backward for ONE ring key-block, consuming the GLOBAL lse.

    Mirrors flash_attn.cute.interface._bwd_large_headdim_block but with the block-local
    `softmax(s)` replaced by `exp(scale*s - lse_i)` so each partial key-block contributes
    the gradient it would under the GLOBAL softmax normalisation (the whole point of ring
    attention). Packed varlen only (q/k/v/out/dout are (total, h, d); lse is (h, total_q)).
    Writes dq into dq_buf[:Tq], dk into dk_buf[:Tk], dv into dv_buf[:Tk] (matching the
    reference's pre-allocated-buffer convention) and returns those views.
    """
    Tq, hq, d = q.shape
    Tk, hkv, _ = k.shape
    dv_dim = v.shape[-1]
    g = hq // hkv
    dev = q.device

    wl, wr = window_size
    is_varlen_mask = (len(cu_q) > 2)  # >1 document -> need per-doc block-diagonal masking
    doc_q = doc_k = None
    if is_varlen_mask:
        cuq = cu_q.to(device=dev).long()
        cuk = cu_k.to(device=dev).long()
        doc_q = torch.repeat_interleave(torch.arange(cuq.numel() - 1, device=dev), cuq[1:] - cuq[:-1])
        doc_k = torch.repeat_interleave(torch.arange(cuk.numel() - 1, device=dev), cuk[1:] - cuk[:-1])

    # (b=1, h, s, d)
    qb_all = q.unsqueeze(0).transpose(1, 2)                 # (1,hq,Tq,d)
    of = out.float().unsqueeze(0).transpose(1, 2)           # (1,hq,Tq,dv)
    dob_all = dout.unsqueeze(0).transpose(1, 2)             # (1,hq,Tq,dv)
    kt = k.unsqueeze(0).transpose(1, 2)                     # (1,hkv,Tk,d)
    vt = v.unsqueeze(0).transpose(1, 2)                     # (1,hkv,Tk,dv)
    kf_e = kt.repeat_interleave(g, dim=1) if g != 1 else kt  # (1,hq,Tk,d)
    vf_e = vt.repeat_interleave(g, dim=1) if g != 1 else vt
    lse_all = lse.unsqueeze(0).unsqueeze(-1)                # (1,hq,Tq,1)

    delta = (dob_all.float() * of).sum(-1)                  # (1,hq,Tq)  = rowsum(dO . O)
    dq = torch.empty((1, hq, Tq, d), dtype=q.dtype, device=dev)
    dk_e = torch.zeros((1, hq, Tk, d), dtype=torch.float32, device=dev)
    dv_e = torch.zeros((1, hq, Tk, dv_dim), dtype=torch.float32, device=dev)

    offset = Tk - Tq                                        # bottom-right causal alignment
    pure_causal = causal and wl is None and wr is None
    masked = causal or wl is not None or wr is not None or is_varlen_mask
    j_full = torch.arange(Tk, device=dev).view(1, Tk)
    for q0 in range(0, Tq, q_block):
        q1 = min(q0 + q_block, Tq)
        kmax = min(Tk, q1 + offset) if pure_causal else Tk
        qb = qb_all[:, :, q0:q1]                            # (1,hq,bm,d)
        dob = dob_all[:, :, q0:q1]
        deltab = delta[:, :, q0:q1].unsqueeze(-1)           # (1,hq,bm,1)
        lseb = lse_all[:, :, q0:q1]                         # (1,hq,bm,1) global lse
        kf_b, vf_b = kf_e[:, :, :kmax], vf_e[:, :, :kmax]

        mask = None
        if masked:
            i_idx = torch.arange(q0, q1, device=dev).view(-1, 1) + offset
            j_idx = j_full[:, :kmax]
            mask = torch.zeros((q1 - q0, kmax), dtype=torch.bool, device=dev)
            if causal or wr is not None:
                wr_eff = wr if (wr is not None and not causal) else 0
                if causal and wr is None:
                    mask |= j_idx > i_idx
                else:
                    mask |= j_idx > i_idx + wr_eff
            if wl is not None:
                mask |= j_idx < i_idx - wl
            if is_varlen_mask:
                mask |= doc_q[q0:q1].view(-1, 1) != doc_k[:kmax].view(1, -1)
            mask = mask.view(1, 1, q1 - q0, kmax)

        s = torch.matmul(qb, kf_b.transpose(-1, -2)).float() * scale   # (1,hq,bm,kmax)
        if mask is not None:
            s = s.masked_fill(mask, float("-inf"))
        p = torch.exp(s - lseb)                             # GLOBAL-normalised probabilities
        p_cast = p.to(q.dtype)
        dp = torch.matmul(dob, vf_b.transpose(-1, -2)).float()
        ds = (p * (dp - deltab) * scale).to(q.dtype)
        dv_e[:, :, :kmax] += torch.matmul(p_cast.transpose(-1, -2), dob).float()
        dq[:, :, q0:q1] = torch.matmul(ds, kf_b)
        dk_e[:, :, :kmax] += torch.matmul(ds.transpose(-1, -2), qb).float()

    if g != 1:  # reduce expanded kv-head grads back to hkv heads
        dk = dk_e.view(1, hkv, g, Tk, d).sum(2)
        dvv = dv_e.view(1, hkv, g, Tk, dv_dim).sum(2)
    else:
        dk, dvv = dk_e, dv_e

    dq_buf[:Tq] = dq.transpose(1, 2).squeeze(0).to(q.dtype)
    dk_buf[:Tk] = dk.transpose(1, 2).squeeze(0).to(q.dtype)
    dv_buf[:Tk] = dvv.transpose(1, 2).squeeze(0).to(q.dtype)
    return dq_buf[:Tq], dk_buf[:Tk], dv_buf[:Tk]


# ----------------------------------------------------------------------------------------
# forward / backward over the ring (structure verbatim from the reference varlen impl)
# ----------------------------------------------------------------------------------------
def zigzag_ring_flash_attn_varlen_forward(
    process_group, q, k, v, cu_seqlens, max_seqlen, half_index0, half_index1,
    softmax_scale, causal=True, window_size=(None, None),
):
    assert causal, "zigzag ring is meaningless for causal=False"
    comm = RingComm(process_group)
    block_seq_len = q.shape[0] // 2
    q1 = q[half_index1]
    out = lse = None
    next_k = next_v = None
    half_cu = cu_seqlens // 2
    half_max = max_seqlen // 2

    def fwd(q_, k_, v_, causal_):
        sq, sk = q_.shape[0], k_.shape[0]
        cu_q = half_cu if sq == block_seq_len else cu_seqlens
        mq = half_max if sq == block_seq_len else max_seqlen
        cu_k = half_cu if sk == block_seq_len else cu_seqlens
        mk = half_max if sk == block_seq_len else max_seqlen
        return _fa4_fwd(q_, k_, v_, cu_q, cu_k, mq, mk, softmax_scale, causal_, window_size)

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        if step == 0:
            block_out, block_lse = fwd(q, k, v, True)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0, v0 = k[half_index0], v[half_index0]
            block_out, block_lse = fwd(q, k0, v0, False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = fwd(q1, k, v, False)
            out, lse = update_out_and_lse(
                out, lse, block_out, block_lse, slice_=(half_index1,)
            )

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(0, 1).contiguous()   # (T,H,1) -> (H,T)
    return out, lse


def zigzag_ring_flash_attn_varlen_backward(
    process_group, dout, q, k, v, out, softmax_lse, cu_seqlens, max_seqlen,
    half_index0, half_index1, softmax_scale, causal=True, window_size=(None, None),
):
    assert causal, "zigzag ring is meaningless for causal=False"
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq = dk = dv = None
    next_dk = next_dv = None
    next_k = next_v = None
    dk_comm_buffer = dv_comm_buffer = None

    dout1 = dout[half_index1]
    q1 = q[half_index1]
    out1 = out[half_index1]
    softmax_lse1 = get_half_lse(softmax_lse, cu_seqlens, front=False).contiguous()
    block_seq_len = q.shape[0] // 2
    half_cu = cu_seqlens // 2
    half_max = max_seqlen // 2

    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    def bwd(dout_, q_, k_, v_, out_, lse_, causal_):
        sq, sk = q_.shape[0], k_.shape[0]
        cu_q = half_cu if sq == block_seq_len else cu_seqlens
        cu_k = half_cu if sk == block_seq_len else cu_seqlens
        _ring_bwd(dout_, q_, k_, v_, out_, lse_, softmax_scale, causal_, window_size,
                  cu_q, cu_k, dq_buffer, dk_buffer, dv_buffer)

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step == 0:
            bwd(dout, q, k, v, out, softmax_lse, True)
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
        else:
            if step <= kv_comm.rank:
                k0, v0 = k[half_index0], v[half_index0]
                bwd(dout, q, k0, v0, out, softmax_lse, False)
                dq += dq_buffer
            else:
                bwd(dout1, q1, k, v, out1, softmax_lse1, False)
                dq[half_index1] += dq_buffer[:block_seq_len]

            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                dk[half_index0] += dk_buffer[:block_seq_len]
                dv[half_index0] += dv_buffer[:block_seq_len]
            else:
                dk += dk_buffer
                dv += dv_buffer

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv, dk_comm_buffer, dv_comm_buffer)

    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class ZigZagRingFlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, softmax_scale, causal,
                window_size, group):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        k, v = k.contiguous(), v.contiguous()
        half_index0 = get_half_index(cu_seqlens, front=True)
        half_index1 = get_half_index(cu_seqlens, front=False)
        out, softmax_lse = zigzag_ring_flash_attn_varlen_forward(
            group, q, k, v, cu_seqlens, max_seqlen, half_index0, half_index1,
            softmax_scale=softmax_scale, causal=causal, window_size=window_size,
        )
        is_tensor = isinstance(half_index0, torch.Tensor)
        ctx.is_half_index_tensor = is_tensor
        if is_tensor:
            ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens, half_index0, half_index1)
        else:
            ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens)
            ctx.half_index0, ctx.half_index1 = half_index0, half_index1
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        if ctx.is_half_index_tensor:
            q, k, v, out, softmax_lse, cu_seqlens, half_index0, half_index1 = ctx.saved_tensors
        else:
            q, k, v, out, softmax_lse, cu_seqlens = ctx.saved_tensors
            half_index0, half_index1 = ctx.half_index0, ctx.half_index1
        dq, dk, dv = zigzag_ring_flash_attn_varlen_backward(
            ctx.group, dout, q, k, v, out, softmax_lse, cu_seqlens, ctx.max_seqlen,
            half_index0, half_index1, softmax_scale=ctx.softmax_scale,
            causal=ctx.causal, window_size=ctx.window_size,
        )
        return dq, dk, dv, None, None, None, None, None, None


def zigzag_ring_flash_attn_varlen_func(
    q, k, v, cu_seqlens, max_seqlen, softmax_scale=None, causal=True,
    window_size=(None, None), group=None,
):
    """Context-parallel (zigzag ring) varlen attention.

    q/k/v: packed (total_local_tokens, num_head, head_dim) for THIS rank's zigzag shard.
    cu_seqlens/max_seqlen: LOCAL packing metadata (each doc contributes its 2 zigzag halves).
    Returns the local attention output (total_local_tokens, num_head, head_dim); dq/dk/dv flow
    back through autograd. `group` is the context-parallel process group (default: WORLD).
    """
    return ZigZagRingFlashAttnVarlenFunc.apply(
        q, k, v, cu_seqlens, max_seqlen, softmax_scale, causal, window_size, group
    )


# ========================================================================================
# Position-aware ring (for SLIDING-WINDOW layers) — the general, mask-exact path.
# ========================================================================================
# The fused zigzag ring above is full-causal only: a kernel's `window_size` is bottom-right
# aligned WITHIN each block call, but zigzag's q1/k0 blocks are non-contiguous global chunks,
# so a fused window would mask the wrong keys. gemma-4's 256-dim sliding layers therefore ring
# through this path instead: a plain (vanilla) ring where each (local-q, ring-kv) block builds
# its keep-mask from GLOBAL per-token (position_id, doc_id) — same doc AND causal (k_pos<=q_pos)
# AND within the left window (q_pos-k_pos <= window). That mask is correct for ANY sharding
# (zigzag or not), the online-softmax LSE combine gives the global normalisation, and the
# backward recomputes each block with that global LSE. Slower than a fused kernel (torch
# matmuls), but the sliding window is small so most blocks mask to nothing.
#
# position_ids/doc_ids are per-token int tensors for THIS rank's shard (the collator has them;
# doc_ids are GLOBAL doc indices so a query only ever attends keys of its own document, on any
# rank). window = the left window size (gemma sliding_window); None = full causal (also valid
# here, just slower than the fused path).

def _posaware_mask(q_pos, q_doc, k_pos, k_doc, window):
    """(Tq, Tk) bool KEEP mask: same doc & causal & within left window. True = attend."""
    keep = (q_doc[:, None] == k_doc[None, :]) & (k_pos[None, :] <= q_pos[:, None])
    if window is not None:
        keep &= (q_pos[:, None] - k_pos[None, :]) <= window
    return keep


def _posaware_fwd_block(q, k, v, q_pos, q_doc, k_pos, k_doc, scale, window, q_block=1024):
    """out (Tq,H,Dv) + lse (H,Tq) for local q vs one ring kv block, global-position masked.
    Blocked over q so the (H,bm,Tk) score is bounded. Fully-masked rows get lse=-inf (contribute
    nothing to the online-softmax combine)."""
    Tq, hq, d = q.shape
    Tk, hkv, dv = v.shape
    g = hq // hkv
    dev = q.device
    kt = k.transpose(0, 1); vt = v.transpose(0, 1)                     # (hkv,Tk,*)
    kf = kt.repeat_interleave(g, dim=0) if g != 1 else kt              # (hq,Tk,d)
    vf = vt.repeat_interleave(g, dim=0) if g != 1 else vt
    out = torch.empty(Tq, hq, dv, dtype=torch.float32, device=dev)
    lse = torch.empty(hq, Tq, dtype=torch.float32, device=dev)
    for s0 in range(0, Tq, q_block):
        s1 = min(s0 + q_block, Tq)
        qb = q[s0:s1].transpose(0, 1)                                 # (hq,bm,d)
        sc = torch.matmul(qb.float(), kf.float().transpose(-1, -2)) * scale   # (hq,bm,Tk)
        keep = _posaware_mask(q_pos[s0:s1], q_doc[s0:s1], k_pos, k_doc, window)  # (bm,Tk)
        sc = sc.masked_fill(~keep.unsqueeze(0), float("-inf"))
        m = sc.amax(dim=-1, keepdim=True)
        m = torch.where(torch.isneginf(m), torch.zeros_like(m), m)
        p = torch.exp(sc - m)
        denom = p.sum(dim=-1, keepdim=True)                           # (hq,bm,1)
        ob = torch.matmul(p, vf.float()) / denom.clamp(min=1e-20)     # (hq,bm,dv)
        lb = (m + torch.log(denom.clamp(min=1e-20))).squeeze(-1)      # (hq,bm) natural-log lse
        lb = torch.where(denom.squeeze(-1) == 0, torch.full_like(lb, float("-inf")), lb)
        out[s0:s1] = ob.transpose(0, 1)
        lse[:, s0:s1] = lb
    return out.to(q.dtype), lse


def _posaware_bwd_block(dout, q, k, v, out, lse, q_pos, q_doc, k_pos, k_doc, scale, window,
                        q_block=1024):
    """dq (Tq,H,D), dk (Tk,Hkv,D), dv (Tk,Hkv,Dv) for local q vs one ring kv block, using the
    GLOBAL lse (p = exp(scale*S - lse)). Same math as _ring_bwd, masked by global positions."""
    Tq, hq, d = q.shape
    Tk, hkv, dv = v.shape
    g = hq // hkv
    dev = q.device
    kt = k.transpose(0, 1); vt = v.transpose(0, 1)
    kf = (kt.repeat_interleave(g, dim=0) if g != 1 else kt).float()   # (hq,Tk,d)
    vf = (vt.repeat_interleave(g, dim=0) if g != 1 else vt).float()
    of = out.float().transpose(0, 1)                                  # (hq,Tq,dv)
    dof = dout.float().transpose(0, 1)                                # (hq,Tq,dv)
    delta = (dof * of).sum(-1)                                        # (hq,Tq)
    dq = torch.zeros(hq, Tq, d, dtype=torch.float32, device=dev)
    dk_e = torch.zeros(hq, Tk, d, dtype=torch.float32, device=dev)
    dv_e = torch.zeros(hq, Tk, dv, dtype=torch.float32, device=dev)
    for s0 in range(0, Tq, q_block):
        s1 = min(s0 + q_block, Tq)
        qb = q[s0:s1].transpose(0, 1).float()                         # (hq,bm,d)
        sc = torch.matmul(qb, kf.transpose(-1, -2)) * scale           # (hq,bm,Tk)
        keep = _posaware_mask(q_pos[s0:s1], q_doc[s0:s1], k_pos, k_doc, window)
        sc = sc.masked_fill(~keep.unsqueeze(0), float("-inf"))
        p = torch.exp(sc - lse[:, s0:s1].unsqueeze(-1))               # global-normalised
        dob = dof[:, s0:s1]                                           # (hq,bm,dv)
        dp = torch.matmul(dob, vf.transpose(-1, -2))                  # (hq,bm,Tk)
        ds = p * (dp - delta[:, s0:s1].unsqueeze(-1)) * scale
        dv_e += torch.matmul(p.transpose(-1, -2), dob)
        dq[:, s0:s1] = torch.matmul(ds, kf)
        dk_e += torch.matmul(ds.transpose(-1, -2), qb)
    if g != 1:
        dk = dk_e.view(hkv, g, Tk, d).sum(1)
        dvv = dv_e.view(hkv, g, Tk, dv).sum(1)
    else:
        dk, dvv = dk_e, dv_e
    return (dq.transpose(0, 1).to(q.dtype),
            dk.transpose(0, 1).to(q.dtype),
            dvv.transpose(0, 1).to(q.dtype))


def _gather_posdoc(group, pos, doc, world_size):
    """All-gather the STATIC per-token (position_id, doc_id) of every rank ONCE, up front.

    position/doc never change, so we gather them instead of ringing them alongside k/v. This is
    load-bearing for the backward: ringing 4 tensors (k,v,pos,doc) on the kv comm while dk/dv ring
    2 on a second comm over the SAME NCCL group interleaves asymmetric P2P and mis-matches the
    dk/dv recvs (silently corrupts the gradients). Ring only k,v (2) + dk,dv (2) — the symmetric
    pattern the fused zigzag path is verified with. Returns lists indexed by GROUP rank; the block
    at ring step `s` uses the shard from src rank (rank - s) mod world_size.
    """
    pos = pos.contiguous(); doc = doc.contiguous()
    pl = [torch.empty_like(pos) for _ in range(world_size)]
    dl = [torch.empty_like(doc) for _ in range(world_size)]
    dist.all_gather(pl, pos, group=group)
    dist.all_gather(dl, doc, group=group)
    return pl, dl


def posaware_ring_forward(group, q, k, v, q_pos, q_doc, softmax_scale, window):
    comm = RingComm(group)
    W, rank = comm.world_size, comm.rank
    all_pos, all_doc = _gather_posdoc(group, q_pos, q_doc, W)
    out = lse = None
    nk = nv = None
    for step in range(W):
        if step + 1 != W:
            nk, nv = comm.send_recv_kv(k, v)
        src = (rank - step) % W                       # which rank's k/v this block holds
        block_out, block_lse = _posaware_fwd_block(
            q, k, v, q_pos, q_doc, all_pos[src], all_doc[src], softmax_scale, window)
        out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        if step + 1 != W:
            comm.wait()
            k, v = nk, nv
    return out.to(q.dtype), lse.squeeze(-1).transpose(0, 1).contiguous()  # (H,T)


def posaware_ring_backward(group, dout, q, k, v, out, softmax_lse, q_pos, q_doc,
                           softmax_scale, window):
    kv_comm = RingComm(group)
    d_kv_comm = RingComm(group)
    W, rank = kv_comm.world_size, kv_comm.rank
    all_pos, all_doc = _gather_posdoc(group, q_pos, q_doc, W)
    dq = dk = dv = None
    next_dk = next_dv = None
    nk = nv = None
    for step in range(W):
        if step + 1 != W:
            nk, nv = kv_comm.send_recv_kv(k, v)
        src = (rank - step) % W
        bdq, bdk, bdv = _posaware_bwd_block(
            dout, q, k, v, out, softmax_lse, q_pos, q_doc, all_pos[src], all_doc[src],
            softmax_scale, window)
        if dq is None:
            dq = bdq.to(torch.float32)
            dk = bdk.to(torch.float32)
            dv = bdv.to(torch.float32)
        else:
            dq += bdq
            d_kv_comm.wait()
            dk = bdk.to(torch.float32) + next_dk
            dv = bdv.to(torch.float32) + next_dv
        if step + 1 != W:
            kv_comm.wait()
            k, v = nk, nv
        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)
    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class PosAwareRingAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, position_ids, doc_ids, softmax_scale, window, group):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        k, v = k.contiguous(), v.contiguous()
        position_ids = position_ids.to(torch.long)
        doc_ids = doc_ids.to(torch.long)
        out, lse = posaware_ring_forward(
            group, q, k, v, position_ids, doc_ids, softmax_scale, window)
        ctx.save_for_backward(q, k, v, out, lse, position_ids, doc_ids)
        ctx.softmax_scale = softmax_scale
        ctx.window = window
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse, position_ids, doc_ids = ctx.saved_tensors
        dq, dk, dv = posaware_ring_backward(
            ctx.group, dout, q, k, v, out, lse, position_ids, doc_ids,
            ctx.softmax_scale, ctx.window)
        return dq, dk, dv, None, None, None, None, None


def posaware_ring_attn_func(q, k, v, position_ids, doc_ids, softmax_scale=None,
                            window=None, group=None):
    """Position-aware (sliding-window-capable) context-parallel ring attention.

    q/k/v: packed (total_local_tokens, num_head, head_dim) for THIS rank's shard.
    position_ids/doc_ids: (total_local_tokens,) per-token GLOBAL per-doc position + global doc id.
    window: left sliding-window size (gemma sliding_window); None = full causal.
    Correct for any sharding (mask is built from global positions). group = CP process group.
    """
    return PosAwareRingAttnFunc.apply(
        q, k, v, position_ids, doc_ids, softmax_scale, window, group)
