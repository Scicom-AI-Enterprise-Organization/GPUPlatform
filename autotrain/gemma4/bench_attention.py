"""Benchmark: FlashAttention-4 (head_dim-512 cute fork) vs vLLM's Triton unified attention.

Compares the two attention kernels gemma-4 can use at inference time, on the two
attention geometries gemma-4-31B actually has:

  * "full"    — GLOBAL layers: head_dim 512, 32 q-heads / 4 kv-heads (GQA 8), full causal.
                This is the FA4 raison d'etre: FA2/FA3 cap at head_dim 256, so vLLM's
                FlashAttention backend can't serve it — only the Triton backend (and FA4) can.
  * "sliding" — SLIDING layers: head_dim 256, 32 q-heads / 16 kv-heads (GQA 2), window 1024.

For each geometry it times two regimes:
  * prefill — q_len == kv_len == L  (the whole packed prompt attends causally)
  * decode  — q_len == 1, kv_len == L  (one new token attends L cached keys)
at context lengths L in {1024, 2048, 4096, 8192, 16384}.

Both kernels run on the SAME q/k/v values (the Triton side repacks k/v into a paged KV
cache), so the per-row output is compared head-to-head (max-abs / cosine). An SDPA fp32
reference anchors absolute correctness at the smaller L where the O(S^2) score still fits.

  FA4   : flash_attn.cute.interface.flash_attn_varlen_func  (contiguous varlen, no paging)
  Triton: vllm.v1.attention.ops.triton_unified_attention.unified_attention  (paged KV)
          — exact kernel source copied from a vLLM checkout into bench_vllm_shim/ (no full
            vLLM build); production decode uses the 3D split-KV path, replicated here.

  CUDA_VISIBLE_DEVICES=7 python bench_attention.py            # full run, JSON to stdout/--out
  CUDA_VISIBLE_DEVICES=7 python bench_attention.py --quick    # smaller iters, smoke
"""
import argparse
import json
import os
import sys

import torch

# The vLLM Triton kernel imports `vllm.*` internally; point those imports at the local
# shim package under bench_vllm_shim/ (which carries the exact kernel source copied from
# a vLLM checkout by bench_setup.sh, plus tiny stand-ins for envs/logger/platforms/...).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_vllm_shim"))

# ---- vLLM Triton kernel (exact source from a vLLM checkout via the bench_vllm_shim) ----
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

# ---- FlashAttention-4 head_dim-512 cute fork ----
from flash_attn.cute.interface import flash_attn_varlen_func as fa4_varlen

DEVICE = "cuda"
DTYPE = torch.bfloat16
BLOCK_SIZE = 16  # vLLM paged-KV block size (default)
MIN_LAUNCH_GRID_SIZE_2D = 128  # vLLM triton_attn.py constant (drives seq_threshold_3D)

# gemma-4-31B attention geometries (from the model config: global_head_dim=512,
# num_global_key_value_heads=4; sliding head_dim=256, num_key_value_heads=16; 32 q-heads).
CONFIGS = {
    "full":    dict(num_q_heads=32, num_kv_heads=4,  head_dim=512, window=None),
    "sliding": dict(num_q_heads=32, num_kv_heads=16, head_dim=256, window=1024),
}
CTX_LENS = [1024, 2048, 4096, 8192, 16384, 32768]
MODES = ["prefill", "decode"]


def make_qkv(mode, L, cfg, seed=0):
    """q/k/v in the FA4 layout (total_tokens, num_heads, head_dim)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    Hq, Hk, D = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    q_len = 1 if mode == "decode" else L
    kv_len = L
    q = torch.randn(q_len, Hq, D, dtype=DTYPE, device=DEVICE, generator=g)
    k = torch.randn(kv_len, Hk, D, dtype=DTYPE, device=DEVICE, generator=g)
    v = torch.randn(kv_len, Hk, D, dtype=DTYPE, device=DEVICE, generator=g)
    return q, k, v, q_len, kv_len


def fa4_window(cfg):
    # flash-attn (left, right) inclusive convention: a window of W => left = W-1.
    return (cfg["window"] - 1, 0) if cfg["window"] is not None else (None, None)


def triton_window(cfg):
    return (cfg["window"] - 1, 0) if cfg["window"] is not None else (-1, -1)


def run_fa4(q, k, v, q_len, kv_len, cfg, scale, num_splits=1):
    """num_splits=1 -> no split-KV (one CTA walks the whole KV per head: starves the GPU at
    decode). num_splits=0 -> the fork's FlashDecoding heuristic picks a split count
    (min(num_SMs // total_mblocks, 128, num_n_blocks)) so the long KV is processed by many CTAs
    in parallel and combined. For prefill there are already enough M-blocks, so the heuristic
    returns 1 (no overhead); for decode (q_len=1) it returns many -> the decode speedup.
    pack_gqa auto-enables for GQA (qhead>kvhead) in BOTH cases, so split-KV is the only lever."""
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=DEVICE)
    cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device=DEVICE)
    out = fa4_varlen(
        q, k, v,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=q_len, max_seqlen_k=kv_len,
        softmax_scale=scale, causal=True, window_size=fa4_window(cfg),
        num_splits=num_splits,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out  # (q_len, Hq, D)


# Manual-FlashDecoding split candidates (the fork's native SplitKV asserts out on SM90).
DECODE_SPLIT_CANDIDATES = [2, 4, 8, 16, 32, 64]


def run_fa4_decode_split(q, k, v, cfg, scale, n_splits):
    """FlashDecoding for FA4 on SM90, done around the kernel (its native SplitKV is
    Blackwell-only — `assert not is_split_kv` on SM 9.0). Split the KV into `n_splits`
    contiguous chunks and run them as ONE varlen batch — the decode query repeated once per
    chunk (cu_seqlens_q = 0,1,2,…,n) against per-chunk key ranges (cu_seqlens_k) — with
    `return_lse=True`, then combine the n partial outputs with a log-sum-exp reduction in
    PyTorch. This launches n× the CTAs so the long KV is processed in parallel instead of one
    CTA walking it serially. Exact: for decode the single query sits at the end and attends
    EVERY key with no causal mask, so each chunk is a plain full attention over its key range
    (causal=False); a sliding layer attends only the last `window` keys, so we split just those.
        combine:  O = Σ_s exp(lse_s − M)·O_s / Σ_s exp(lse_s − M),  M = max_s lse_s  (per head)."""
    Hq, Hk, D = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    W = cfg["window"]
    kv_len = k.shape[0]
    if W is not None and W < kv_len:            # sliding: only the last-window keys are attended
        k, v, kv_len = k[kv_len - W:].contiguous(), v[kv_len - W:].contiguous(), W
    n = max(1, min(n_splits, kv_len))
    base, rem = divmod(kv_len, n)               # near-equal, strictly-positive chunks
    bounds, acc = [0], 0
    for i in range(n):
        acc += base + (1 if i < rem else 0)
        bounds.append(acc)
    cu_k = torch.tensor(bounds, dtype=torch.int32, device=DEVICE)
    cu_q = torch.arange(n + 1, dtype=torch.int32, device=DEVICE)
    max_chunk = base + (1 if rem else 0)
    q_rep = q.expand(n, Hq, D).contiguous()     # same decode query, one row per chunk
    out, lse = fa4_varlen(
        q_rep, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=1, max_seqlen_k=max_chunk,
        softmax_scale=scale, causal=False, window_size=(None, None),
        num_splits=1, return_lse=True,
    )
    # out: (n, Hq, D); lse: (num_head, total_q) = (Hq, n)
    lse = lse.transpose(0, 1).float()           # (n, Hq)
    m = lse.max(dim=0, keepdim=True).values     # (1, Hq)
    w = torch.exp(lse - m)                       # (n, Hq)
    denom = w.sum(dim=0).clamp_min(1e-20)        # (Hq,)
    o = (w.unsqueeze(-1) * out.float()).sum(dim=0) / denom.unsqueeze(-1)  # (Hq, D)
    return o.unsqueeze(0).to(q.dtype)            # (1, Hq, D)


def build_paged_cache(k, v, kv_len, cfg):
    """Repack contiguous (kv_len, Hk, D) k/v into vLLM paged caches + block table."""
    Hk, D = cfg["num_kv_heads"], cfg["head_dim"]
    num_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    key_cache = torch.zeros(num_blocks, BLOCK_SIZE, Hk, D, dtype=DTYPE, device=DEVICE)
    value_cache = torch.zeros_like(key_cache)
    key_cache.view(-1, Hk, D)[:kv_len] = k
    value_cache.view(-1, Hk, D)[:kv_len] = v
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).unsqueeze(0)
    return key_cache, value_cache, block_table


def make_triton_runner(q, key_cache, value_cache, block_table, q_len, kv_len, cfg, scale):
    """Return (callable, out_tensor). For decode (q_len==1) wire up the 3D split-KV path
    exactly as vLLM's TritonAttention backend does, so we benchmark the fast decode kernel."""
    Hq, D = cfg["num_q_heads"], cfg["head_dim"]
    out = torch.empty(q_len, Hq, D, dtype=DTYPE, device=DEVICE)
    cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=DEVICE)
    seqused_k = torch.tensor([kv_len], dtype=torch.int32, device=DEVICE)

    # 3D split-KV decode path (vLLM backend defaults): seq_threshold_3D = 128 // num_kv_heads,
    # 16 parallel softmax segments. use_3d fires only when max_seqlen_q == 1 and num_seqs <= thr.
    num_par_softmax_segments = 16
    seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // cfg["num_kv_heads"]
    hs_padded = 1 << (D - 1).bit_length()
    segm_out = torch.empty((seq_threshold_3D, Hq, num_par_softmax_segments, hs_padded),
                           dtype=torch.float32, device=DEVICE)
    segm_max = torch.empty((seq_threshold_3D, Hq, num_par_softmax_segments),
                           dtype=torch.float32, device=DEVICE)
    segm_expsum = torch.empty_like(segm_max)

    win = triton_window(cfg)

    def call():
        unified_attention(
            q=q, k=key_cache, v=value_cache, out=out,
            cu_seqlens_q=cu_q, max_seqlen_q=q_len,
            seqused_k=seqused_k, max_seqlen_k=kv_len,
            softmax_scale=scale, causal=True, window_size=win,
            block_table=block_table, softcap=0,
            q_descale=None, k_descale=None, v_descale=None,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=segm_out, softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_expsum,
            kv_quant_mode=KVQuantMode.NONE,
        )
        return out

    return call, out


def sdpa_reference(q, k, v, q_len, kv_len, cfg, scale):
    """fp32 SDPA per-document causal reference (GQA expanded). Only called for small L."""
    Hq, Hk, D = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qf = q.float().transpose(0, 1)  # (Hq, q_len, D)
    kf = k.float().transpose(0, 1)  # (Hk, kv_len, D)
    vf = v.float().transpose(0, 1)
    rep = Hq // Hk
    kf = kf.repeat_interleave(rep, dim=0)
    vf = vf.repeat_interleave(rep, dim=0)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale  # (Hq, q_len, kv_len)
    # causal: query global row = kv_len - q_len + i attends keys j <= that row
    qpos = torch.arange(kv_len - q_len, kv_len, device=DEVICE).unsqueeze(1)
    kpos = torch.arange(kv_len, device=DEVICE).unsqueeze(0)
    mask = kpos > qpos
    if cfg["window"] is not None:
        mask |= (kpos <= qpos - cfg["window"])  # window of W => attend (qpos-W, qpos]
    scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf)  # (Hq, q_len, D)
    return out.transpose(0, 1).to(DTYPE)  # (q_len, Hq, D)


def attn_flops(mode, L, cfg):
    """Causal attention FLOPs: 2 matmuls (QK^T, PV), each 2*Hq*D MACs per (query,key) pair.
    prefill attends ~L^2/2 pairs (causal); decode attends L pairs. Sliding caps keys at window."""
    Hq, D = cfg["num_q_heads"], cfg["head_dim"]
    W = cfg["window"]
    if mode == "decode":
        keys = min(L, W) if W else L
        pairs = keys
    else:
        if W and W < L:
            pairs = (W * (W + 1) // 2) + (L - W) * W  # ramp then steady window
        else:
            pairs = L * (L + 1) // 2
    return 2 * 2 * Hq * D * pairs  # 2 matmuls * 2 flop/MAC * heads * dim * pairs


def time_call(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))  # ms
    return times[len(times) // 2]  # median


def diff_stats(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    max_abs = (a - b).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return max_abs, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--sdpa-ref-max-l", type=int, default=4096,
                    help="anchor against fp32 SDPA for L <= this (O(S^2) score must fit)")
    args = ap.parse_args()
    if args.quick:
        args.iters, args.warmup = 5, 2

    dev_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device={dev_name}  sm_{cap[0]}{cap[1]}  torch={torch.__version__}  "
          f"triton={__import__('triton').__version__}", flush=True)
    print(f"iters={args.iters} warmup={args.warmup}\n", flush=True)

    rows = []
    # fa4_ms = FA4 best (prefill: plain; decode: manual FlashDecoding = run_fa4_decode_split,
    # best of DECODE_SPLIT_CANDIDATES). fa4_base = FA4 no-split (num_splits=1) — what the first
    # benchmark measured. `split` = chosen #KV-splits for decode (1 for prefill).
    header = (f"{'cfg':8} {'mode':7} {'L':>6} {'hd':>4} {'fa4_ms':>9} {'fa4_base':>9} {'triton_ms':>10} "
              f"{'fa4/tri':>8} {'base/tri':>9} {'split':>6} {'fa4_TF/s':>9} {'tri_TF/s':>9} {'cos':>8} {'vs_sdpa':>16}")
    print(header)
    print("-" * len(header))

    for cfg_name, cfg in CONFIGS.items():
        D = cfg["head_dim"]
        scale = D ** -0.5
        for mode in MODES:
            for L in CTX_LENS:
                q, k, v, q_len, kv_len = make_qkv(mode, L, cfg)
                # --- FA4 baseline (no split) ---
                try:
                    base_fn = lambda: run_fa4(q, k, v, q_len, kv_len, cfg, scale, num_splits=1)
                    base_out = base_fn()
                except Exception as e:
                    print(f"{cfg_name:8} {mode:7} {L:>6} {D:>4}  FA4 ERROR: {type(e).__name__}: {str(e)[:80]}")
                    rows.append(dict(cfg=cfg_name, mode=mode, L=L, head_dim=D, error_fa4=repr(e)))
                    continue

                # --- FA4 "best": prefill uses the plain kernel (already parallel); decode uses
                #     manual FlashDecoding, sweeping the split count for the fastest. ---
                split = 1
                if mode == "decode":
                    # seed "best" with the no-split path so a split only wins if it's faster
                    # (at small L the split overhead can exceed the naive cost).
                    best_ms = time_call(base_fn, max(8, args.iters // 4), args.warmup)
                    best_n, fa4_fn = 1, base_fn
                    for n in DECODE_SPLIT_CANDIDATES:
                        cand = (lambda nn: (lambda: run_fa4_decode_split(q, k, v, cfg, scale, nn)))(n)
                        try:
                            ms = time_call(cand, max(8, args.iters // 4), args.warmup)
                        except Exception:
                            continue
                        if ms < best_ms:
                            best_ms, best_n, fa4_fn = ms, n, cand
                    split = best_n
                    fa4_out = run_fa4_decode_split(q, k, v, cfg, scale, best_n)
                else:
                    fa4_fn, fa4_out = base_fn, base_out

                kc, vc, bt = build_paged_cache(k, v, kv_len, cfg)
                tri_call, tri_out = make_triton_runner(q, kc, vc, bt, q_len, kv_len, cfg, scale)
                try:
                    tri_call()
                except Exception as e:
                    print(f"{cfg_name:8} {mode:7} {L:>6} {D:>4}  TRITON ERROR: {type(e).__name__}: {str(e)[:80]}")
                    rows.append(dict(cfg=cfg_name, mode=mode, L=L, head_dim=D, error_triton=repr(e)))
                    continue

                max_abs, cos = diff_stats(fa4_out, tri_out)

                # --- correctness anchor vs fp32 SDPA. Always for decode (q_len=1 -> tiny
                # (Hq,1,L) score); for prefill only where the O(Hq*L^2) score still fits. ---
                vs_sdpa = ""
                if mode == "decode" or L <= args.sdpa_ref_max_l:
                    ref = sdpa_reference(q, k, v, q_len, kv_len, cfg, scale)
                    fa_ma, fa_cos = diff_stats(fa4_out, ref)
                    tr_ma, tr_cos = diff_stats(tri_out, ref)
                    vs_sdpa = f"fa{fa_cos:.4f}/tr{tr_cos:.4f}"

                # --- time ---
                fa4_ms = time_call(fa4_fn, args.iters, args.warmup)
                base_ms = fa4_ms if mode != "decode" else time_call(base_fn, args.iters, args.warmup)
                tri_ms = time_call(tri_call, args.iters, args.warmup)
                flops = attn_flops(mode, L, cfg)
                fa4_tf = flops / (fa4_ms * 1e-3) / 1e12
                tri_tf = flops / (tri_ms * 1e-3) / 1e12
                ratio = fa4_ms / tri_ms
                base_ratio = base_ms / tri_ms

                print(f"{cfg_name:8} {mode:7} {L:>6} {D:>4} {fa4_ms:>9.4f} {base_ms:>9.4f} {tri_ms:>10.4f} "
                      f"{ratio:>8.2f} {base_ratio:>9.2f} {split:>6} {fa4_tf:>9.1f} {tri_tf:>9.1f} {cos:>8.5f} {vs_sdpa:>16}",
                      flush=True)
                rows.append(dict(cfg=cfg_name, mode=mode, L=L, head_dim=D,
                                 num_q_heads=cfg["num_q_heads"], num_kv_heads=cfg["num_kv_heads"],
                                 window=cfg["window"], fa4_ms=fa4_ms, fa4_base_ms=base_ms,
                                 triton_ms=tri_ms, fa4_over_triton=ratio, fa4_base_over_triton=base_ratio,
                                 decode_split=split, fa4_decode_speedup=base_ms / fa4_ms,
                                 fa4_tflops=fa4_tf, triton_tflops=tri_tf,
                                 max_abs_diff=max_abs, cosine=cos, vs_sdpa=vs_sdpa))
                del q, k, v, kc, vc, bt, fa4_out, tri_out, base_out
                torch.cuda.empty_cache()

    out = dict(device=dev_name, sm=f"{cap[0]}{cap[1]}", torch=torch.__version__,
               triton=__import__("triton").__version__, iters=args.iters,
               warmup=args.warmup, block_size=BLOCK_SIZE, rows=rows)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}", flush=True)
    else:
        print("\n" + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
