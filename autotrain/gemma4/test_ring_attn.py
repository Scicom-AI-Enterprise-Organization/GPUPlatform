"""Distributed correctness test for zigzag ring (context-parallel) attention.

The invariant the user asked for: **the output AND the q/k/v gradients from the ring
(sequence sharded across GPUs, zigzag-balanced) must equal the output+grads WITHOUT the
ring (the whole sequence on one GPU).**

Run on N GPUs:
    torchrun --nproc_per_node=2 test_ring_attn.py

For each config it:
  1. builds identical full q/k/v/dout on every rank (rank-0 seeded + broadcast),
  2. NON-RING reference: flash_attn_varlen_func(full) fwd+bwd  -> out_full, dq/dk/dv_full,
     plus an fp32 naive-attention ground truth (exact grads via autograd),
  3. RING: extract THIS rank's zigzag shard (chunk r + chunk 2W-1-r of each doc), run
     zigzag_ring_flash_attn_varlen_func fwd+bwd -> out_loc, dq/dk/dv_loc,
  4. compares the ring shard against the same zigzag slice of the non-ring reference
     (and the fp32 ground truth). max-abs diffs are all-reduced (MAX) across ranks.

Geometries mirror Gemma-4: head_dim-512 global (32 q / 4 kv heads) and head_dim-256
sliding-shaped (32 q / 16 kv), plus a multi-doc varlen pack and a tiny fp32-tol config.
"""
import os
import sys

import torch
import torch.distributed as dist
from flash_attn.cute.interface import flash_attn_varlen_func

from ring_zigzag_attn import zigzag_ring_flash_attn_varlen_func, posaware_ring_attn_func


# ----------------------------------------------------------------------------------------
def zigzag_shard(x, cu_full, world_size, rank):
    """This rank's zigzag shard of a packed (S, ...) tensor + its local cu_seqlens.

    Each document is split into 2*world_size equal chunks; rank r takes chunk r and chunk
    (2*world_size-1-r), concatenated [front, back] per document. Returns (local, local_cu).
    Requires every document length divisible by 2*world_size.
    """
    twoW = 2 * world_size
    parts, local_lens = [], []
    for d in range(len(cu_full) - 1):
        s, e = int(cu_full[d]), int(cu_full[d + 1])
        Ld = e - s
        assert Ld % twoW == 0, f"doc len {Ld} not divisible by 2*world_size={twoW}"
        c = Ld // twoW
        doc = x[s:e]
        front = doc[rank * c:(rank + 1) * c]
        back = doc[(twoW - 1 - rank) * c:(twoW - rank) * c]
        parts.append(torch.cat([front, back], dim=0))
        local_lens.append(2 * c)
    local = torch.cat(parts, dim=0)
    local_cu = torch.tensor([0] + torch.cumsum(torch.tensor(local_lens), 0).tolist(),
                            dtype=torch.int32, device=x.device)
    return local, local_cu


def naive_attn(q, k, v, cu, scale, causal, window):
    """fp32 per-document attention ground truth (autograd-differentiable). q (S,hq,d)."""
    hq, hkv = q.shape[1], k.shape[1]
    g = hq // hkv
    kk = k.repeat_interleave(g, dim=1) if g != 1 else k
    vv = v.repeat_interleave(g, dim=1) if g != 1 else v
    outs = []
    wl, wr = window
    for d in range(len(cu) - 1):
        a, b = int(cu[d]), int(cu[d + 1])
        L = b - a
        qd = q[a:b].transpose(0, 1)                      # (hq, L, d)
        kd = kk[a:b].transpose(0, 1)
        vd = vv[a:b].transpose(0, 1)
        s = torch.matmul(qd, kd.transpose(-1, -2)) * scale   # (hq, L, L)
        i = torch.arange(L, device=q.device).view(-1, 1)
        j = torch.arange(L, device=q.device).view(1, -1)
        mask = torch.zeros((L, L), dtype=torch.bool, device=q.device)
        if causal or wr is not None:
            mask |= j > i + (wr or 0)
        if wl is not None:
            mask |= j < i - wl
        s = s.masked_fill(mask.view(1, L, L), float("-inf"))
        p = torch.softmax(s, dim=-1)
        outs.append(torch.matmul(p, vd).transpose(0, 1))     # (L, hq, dv)
    return torch.cat(outs, dim=0)


def diff(a, b):
    a, b = a.float(), b.float()
    denom = b.norm().item() or 1.0
    return (a - b).abs().max().item(), (a - b).norm().item() / denom


# ----------------------------------------------------------------------------------------
def run_config(cfg, rank, world_size, device):
    hq, hkv, d = cfg["hq"], cfg["hkv"], cfg["d"]
    doc_lens = cfg["doc_lens"]
    causal = cfg.get("causal", True)
    window = cfg.get("window", (None, None))
    scale = d ** -0.5
    S = sum(doc_lens)
    cu_full = torch.tensor([0] + torch.cumsum(torch.tensor(doc_lens), 0).tolist(),
                           dtype=torch.int32, device=device)
    max_full = max(doc_lens)

    # ---- identical full tensors on every rank (rank 0 generates, broadcast) ----
    if rank == 0:
        gen = torch.Generator(device="cpu").manual_seed(1234)
        q = torch.randn(S, hq, d, generator=gen, dtype=torch.float32) * 0.5
        k = torch.randn(S, hkv, d, generator=gen, dtype=torch.float32) * 0.5
        v = torch.randn(S, hkv, d, generator=gen, dtype=torch.float32) * 0.5
        dout = torch.randn(S, hq, d, generator=gen, dtype=torch.float32) * 0.5
        blob = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1), dout.reshape(-1)]).to(device)
    else:
        n = S * hq * d + S * hkv * d + S * hkv * d + S * hq * d
        blob = torch.empty(n, device=device)
    dist.broadcast(blob, src=0)
    o1, o2, o3 = S * hq * d, S * hq * d + S * hkv * d, S * hq * d + 2 * S * hkv * d
    q = blob[:o1].reshape(S, hq, d).to(torch.bfloat16)
    k = blob[o1:o2].reshape(S, hkv, d).to(torch.bfloat16)
    v = blob[o2:o3].reshape(S, hkv, d).to(torch.bfloat16)
    dout = blob[o3:].reshape(S, hq, d).to(torch.bfloat16)

    # ---- NON-RING reference (fork, full sequence, autograd) ----
    qf, kf, vf = (t.clone().detach().requires_grad_(True) for t in (q, k, v))
    out_full, _ = flash_attn_varlen_func(   # always returns (out, lse)
        qf, kf, vf, cu_seqlens_q=cu_full, cu_seqlens_k=cu_full,
        max_seqlen_q=max_full, max_seqlen_k=max_full,
        softmax_scale=scale, causal=causal, window_size=window,
    )
    out_full.backward(dout)
    dqf, dkf, dvf = qf.grad, kf.grad, vf.grad

    # ---- fp32 naive ground truth ----
    q32, k32, v32 = (t.float().clone().detach().requires_grad_(True) for t in (q, k, v))
    out_naive = naive_attn(q32, k32, v32, cu_full, scale, causal, window)
    out_naive.backward(dout.float())
    dqn, dkn, dvn = q32.grad, k32.grad, v32.grad

    # ---- RING: this rank's zigzag shard ----
    q_loc, cu_loc = zigzag_shard(q, cu_full, world_size, rank)
    k_loc, _ = zigzag_shard(k, cu_full, world_size, rank)
    v_loc, _ = zigzag_shard(v, cu_full, world_size, rank)
    dout_loc, _ = zigzag_shard(dout, cu_full, world_size, rank)
    max_loc = int((cu_loc[1:] - cu_loc[:-1]).max())

    q_loc = q_loc.clone().detach().requires_grad_(True)
    k_loc = k_loc.clone().detach().requires_grad_(True)
    v_loc = v_loc.clone().detach().requires_grad_(True)
    out_ring = zigzag_ring_flash_attn_varlen_func(
        q_loc, k_loc, v_loc, cu_loc, max_loc,
        softmax_scale=scale, causal=causal, window_size=window, group=None,
    )
    out_ring.backward(dout_loc)
    dq_ring, dk_ring, dv_ring = q_loc.grad, k_loc.grad, v_loc.grad

    # ---- reference sharded the same zigzag way, for a shard-vs-shard compare ----
    ref = {}
    for name, full_t in [("out", out_full), ("dq", dqf), ("dk", dkf), ("dv", dvf)]:
        ref[name], _ = zigzag_shard(full_t.detach(), cu_full, world_size, rank)
    refn = {}
    for name, full_t in [("out", out_naive), ("dq", dqn), ("dk", dkn), ("dv", dvn)]:
        refn[name], _ = zigzag_shard(full_t.detach(), cu_full, world_size, rank)

    got = {"out": out_ring.detach(), "dq": dq_ring, "dk": dk_ring, "dv": dv_ring}

    results = {}
    for name in ["out", "dq", "dk", "dv"]:
        ma_r, rl_r = diff(got[name], ref[name])       # ring vs non-ring fork
        ma_n, rl_n = diff(got[name], refn[name])       # ring vs fp32 ground truth
        # all-reduce MAX across ranks
        t = torch.tensor([ma_r, rl_r, ma_n, rl_n], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        results[name] = t.tolist()
    return results


def run_posaware_config(cfg, rank, world_size, device):
    """Sliding-window (position-aware) ring vs non-ring FA-with-window + fp32 naive-with-window."""
    hq, hkv, d = cfg["hq"], cfg["hkv"], cfg["d"]
    doc_lens = cfg["doc_lens"]
    w = cfg["window"]                       # left sliding-window size
    scale = d ** -0.5
    S = sum(doc_lens)
    cu_full = torch.tensor([0] + torch.cumsum(torch.tensor(doc_lens), 0).tolist(),
                           dtype=torch.int32, device=device)
    max_full = max(doc_lens)
    # per-doc position ids (0..L-1) + global doc ids
    pos = torch.cat([torch.arange(L, device=device) for L in doc_lens])
    docid = torch.cat([torch.full((L,), i, device=device) for i, L in enumerate(doc_lens)])

    if rank == 0:
        gen = torch.Generator(device="cpu").manual_seed(7)
        q = torch.randn(S, hq, d, generator=gen) * 0.5
        k = torch.randn(S, hkv, d, generator=gen) * 0.5
        v = torch.randn(S, hkv, d, generator=gen) * 0.5
        dout = torch.randn(S, hq, d, generator=gen) * 0.5
        blob = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1), dout.reshape(-1)]).to(device)
    else:
        blob = torch.empty(2 * S * hq * d + 2 * S * hkv * d, device=device)
    dist.broadcast(blob, src=0)
    o1, o2, o3 = S * hq * d, S * hq * d + S * hkv * d, S * hq * d + 2 * S * hkv * d
    q = blob[:o1].reshape(S, hq, d).to(torch.bfloat16)
    k = blob[o1:o2].reshape(S, hkv, d).to(torch.bfloat16)
    v = blob[o2:o3].reshape(S, hkv, d).to(torch.bfloat16)
    dout = blob[o3:].reshape(S, hq, d).to(torch.bfloat16)

    # non-ring reference: fused FA with left window (bottom-right aligned per-doc = correct contiguous)
    qf, kf, vf = (t.clone().detach().requires_grad_(True) for t in (q, k, v))
    out_full, _ = flash_attn_varlen_func(
        qf, kf, vf, cu_seqlens_q=cu_full, cu_seqlens_k=cu_full,
        max_seqlen_q=max_full, max_seqlen_k=max_full,
        softmax_scale=scale, causal=True, window_size=(w, 0))
    out_full.backward(dout)
    dqf, dkf, dvf = qf.grad, kf.grad, vf.grad
    # fp32 ground truth
    q32, k32, v32 = (t.float().clone().detach().requires_grad_(True) for t in (q, k, v))
    out_naive = naive_attn(q32, k32, v32, cu_full, scale, True, (w, 0))
    out_naive.backward(dout.float())
    dqn, dkn, dvn = q32.grad, k32.grad, v32.grad

    # RING (position-aware): this rank's zigzag shard of q/k/v AND pos/doc
    q_loc, _ = zigzag_shard(q, cu_full, world_size, rank)
    k_loc, _ = zigzag_shard(k, cu_full, world_size, rank)
    v_loc, _ = zigzag_shard(v, cu_full, world_size, rank)
    dout_loc, _ = zigzag_shard(dout, cu_full, world_size, rank)
    pos_loc, _ = zigzag_shard(pos, cu_full, world_size, rank)
    doc_loc, _ = zigzag_shard(docid, cu_full, world_size, rank)
    q_loc = q_loc.clone().detach().requires_grad_(True)
    k_loc = k_loc.clone().detach().requires_grad_(True)
    v_loc = v_loc.clone().detach().requires_grad_(True)
    out_ring = posaware_ring_attn_func(
        q_loc, k_loc, v_loc, pos_loc, doc_loc, softmax_scale=scale, window=w, group=None)
    out_ring.backward(dout_loc)
    got = {"out": out_ring.detach(), "dq": q_loc.grad, "dk": k_loc.grad, "dv": v_loc.grad}

    ref, refn = {}, {}
    for name, ft in [("out", out_full), ("dq", dqf), ("dk", dkf), ("dv", dvf)]:
        ref[name], _ = zigzag_shard(ft.detach(), cu_full, world_size, rank)
    for name, ft in [("out", out_naive), ("dq", dqn), ("dk", dkn), ("dv", dvn)]:
        refn[name], _ = zigzag_shard(ft.detach(), cu_full, world_size, rank)

    results = {}
    for name in ["out", "dq", "dk", "dv"]:
        ma_r, rl_r = diff(got[name], ref[name])
        ma_n, rl_n = diff(got[name], refn[name])
        t = torch.tensor([ma_r, rl_r, ma_n, rl_n], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        results[name] = t.tolist()
    return results


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl")

    twoW = 2 * world_size
    configs = [
        {"name": "gemma4-global hd512 (32q/4kv) single-doc causal",
         "hq": 32, "hkv": 4, "d": 512, "doc_lens": [twoW * 512]},   # S=2048 @ W=2
        {"name": "gemma4-sliding-geom hd256 (32q/16kv) single-doc causal",
         "hq": 32, "hkv": 16, "d": 256, "doc_lens": [twoW * 512]},
        {"name": "multi-doc varlen hd512 (8q/2kv) causal",
         "hq": 8, "hkv": 2, "d": 512, "doc_lens": [twoW * 256, twoW * 128, twoW * 192]},
        {"name": "tiny hd512 (8q/2kv) single-doc causal (fp32-tight)",
         "hq": 8, "hkv": 2, "d": 512, "doc_lens": [twoW * 64]},
    ]
    # Sliding-window (position-aware ring) — gemma4's 256-dim sliding layers.
    posaware_configs = [
        {"name": "gemma4-sliding hd256 (32q/16kv) single-doc window=300",
         "hq": 32, "hkv": 16, "d": 256, "doc_lens": [twoW * 512], "window": 300},
        {"name": "sliding hd512 (8q/2kv) multi-doc window=200",
         "hq": 8, "hkv": 2, "d": 512, "doc_lens": [twoW * 256, twoW * 128], "window": 200},
    ]

    if rank == 0:
        print(f"world_size={world_size}  torch={torch.__version__}", flush=True)
    all_ok = True

    def report(title, res):
        nonlocal all_ok
        print(f"\n=== {title}  (2W={twoW} chunks) ===", flush=True)
        for name in ["out", "dq", "dk", "dv"]:
            ma_r, rl_r, ma_n, rl_n = res[name]
            ok_r = rl_r < 2e-2      # ring vs non-ring fork (same kernel, bf16) — tight
            ok_n = rl_n < 3e-2      # ring vs fp32 ground truth — bf16 noise
            ok = ok_r and ok_n
            all_ok &= ok
            flag = "PASS" if ok else "FAIL"
            print(f"  [{flag}] {name:3s}  ring-vs-nonring: max_abs={ma_r:.3e} rel={rl_r:.3e}"
                  f"   ring-vs-fp32: max_abs={ma_n:.3e} rel={rl_n:.3e}", flush=True)

    for cfg in configs:
        res = run_config(cfg, rank, world_size, device)
        if rank == 0:
            report(cfg["name"], res)
    for cfg in posaware_configs:
        res = run_posaware_config(cfg, rank, world_size, device)
        if rank == 0:
            report(cfg["name"] + " [posaware]", res)
    ok_t = torch.tensor([1 if all_ok else 0], device=device)
    dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("\n" + ("ALL RING CONFIGS PASS ✅" if ok_t.item() else "SOME CONFIGS FAILED ❌"), flush=True)
    dist.destroy_process_group()
    sys.exit(0 if ok_t.item() else 1)


if __name__ == "__main__":
    main()
