"""CUDA-graph evidence for the short-context decode gap (see CLAUDE.md "Why short-context decode
can't be won"). Measures FA4 FlashDecoding vs vLLM Triton decode BOTH eager and CUDA-graphed, full
hd512 (32 q / 4 kv). Graphs strip per-call launch/dispatch overhead — if the short-L gap were launch
overhead, graphing would close it; instead Triton pulls *ahead* at short L (its decode kernel does
less GPU work for tiny KV), proving the gap is kernel work, not launch.

    CUDA_VISIBLE_DEVICES=7 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 python bench_decode_cudagraph.py
"""
import torch

from bench_attention import (CONFIGS, CTX_LENS, DEVICE, DTYPE, BLOCK_SIZE,
                             MIN_LAUNCH_GRID_SIZE_2D, fused_combine, fa4_varlen,
                             unified_attention, KVQuantMode)


def t(fn, it=200, wu=50):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(True) for _ in range(it)]
    e = [torch.cuda.Event(True) for _ in range(it)]
    for i in range(it):
        s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    return sorted(s[i].elapsed_time(e[i]) for i in range(it))[it // 2]


def try_graph(call):
    """Capture call() into a CUDA graph (warm up on a side stream first); return replay fn."""
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            call()
    torch.cuda.current_stream().wait_stream(s)
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call()
        return g.replay
    except Exception as ex:
        print("   graph capture FAILED:", type(ex).__name__, str(ex)[:90])
        return None


def main():
    cfg = CONFIGS["full"]
    Hq, Hk, D = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    scale = D ** -0.5
    print(f"device={torch.cuda.get_device_name(0)}  full hd{D} {Hq}q/{Hk}kv\n")
    print(f"{'L':>6} {'N':>3} | {'eager fa4':>9} {'tri':>7} {'r':>5} | {'graph fa4':>9} {'tri':>7} {'r':>5}")
    for L in CTX_LENS:
        q = torch.randn(1, Hq, D, dtype=DTYPE, device=DEVICE)
        k = torch.randn(L, Hk, D, dtype=DTYPE, device=DEVICE); v = torch.randn_like(k)
        N = 8 if L <= 2048 else 16
        step = L // N
        cu_q = torch.arange(N + 1, dtype=torch.int32, device=DEVICE)
        cu_k = torch.arange(0, L + 1, step, dtype=torch.int32, device=DEVICE)

        def fa4_call():
            out, lse = fa4_varlen(q.expand(N, Hq, D), k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                  max_seqlen_q=1, max_seqlen_k=step, softmax_scale=scale,
                                  causal=False, window_size=(None, None), num_splits=1, return_lse=True)
            return fused_combine(out, lse, Hq, D)

        nb = (L + BLOCK_SIZE - 1) // BLOCK_SIZE
        kc = torch.zeros(nb, BLOCK_SIZE, Hk, D, dtype=DTYPE, device=DEVICE); vc = torch.zeros_like(kc)
        kc.view(-1, Hk, D)[:L] = k; vc.view(-1, Hk, D)[:L] = v
        bt = torch.arange(nb, dtype=torch.int32, device=DEVICE).unsqueeze(0)
        sk = torch.tensor([L], dtype=torch.int32, device=DEVICE)
        cu1 = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
        nseg, thr = 16, MIN_LAUNCH_GRID_SIZE_2D // Hk
        so = torch.empty((thr, Hq, nseg, 512), dtype=torch.float32, device=DEVICE)
        sm = torch.empty((thr, Hq, nseg), dtype=torch.float32, device=DEVICE); se = torch.empty_like(sm)
        ot = torch.empty(1, Hq, D, dtype=DTYPE, device=DEVICE)

        def tri_call():
            unified_attention(q=q, k=kc, v=vc, out=ot, cu_seqlens_q=cu1, max_seqlen_q=1, seqused_k=sk,
                              max_seqlen_k=L, softmax_scale=scale, causal=True, window_size=(-1, -1),
                              block_table=bt, softcap=0, q_descale=None, k_descale=None, v_descale=None,
                              seq_threshold_3D=thr, num_par_softmax_segments=nseg, softmax_segm_output=so,
                              softmax_segm_max=sm, softmax_segm_expsum=se, kv_quant_mode=KVQuantMode.NONE)
            return ot

        fe, te = t(fa4_call), t(tri_call)
        fg, tg = try_graph(fa4_call), try_graph(tri_call)
        fgm = t(fg) if fg else float("nan")
        tgm = t(tg) if tg else float("nan")
        print(f"{L:>6} {N:>3} | {fe:>9.3f} {te:>7.3f} {fe/te:>5.2f} | {fgm:>9.3f} {tgm:>7.3f} {fgm/tgm:>5.2f}")


if __name__ == "__main__":
    main()
