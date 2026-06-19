"""Correctness + speed of the Triton blockwise FP8 dequant (`dequant_triton.py`) vs the PyTorch
reference (`lora.dequantize_fp8_blockwise`), forward AND backward.

No model needed — random FP8 weights + random positive block scales of representative MiniMax-M2
shapes (a q/k/v/o-style 2D matrix, plus the stacked MoE `gate_up` / `down` expert tensors).

  python bench_dequant.py                 # default shapes
  python bench_dequant.py --experts 256 --hidden 3072 --inter 1536

Checks, per shape:
  * forward:  max_abs(triton, pytorch)  (both do float32(w)*scale -> bf16, so ~0)
  * backward: grad wrt scale_inv — triton analytic vs PyTorch autograd (max_abs)
  * timing:   forward ms + backward ms, triton vs pytorch, with speedups
"""
import argparse
import torch

from dequant_triton import dequantize_fp8_blockwise_triton

BLOCK = (128, 128)


# ---- PyTorch reference (same chunked math as lora.dequantize_fp8_blockwise) ----
def pytorch_dequant(weight, scale_inv, block_size, out_dtype=torch.bfloat16, expert_chunk=16):
    bm, bn = block_size
    *lead, out_f, in_f = weight.shape
    so, si = out_f // bm, in_f // bn
    if not lead:
        w = weight.to(torch.float32).reshape(so, bm, si, bn)
        s = scale_inv.to(torch.float32).reshape(so, 1, si, 1)
        return (w * s).reshape(out_f, in_f).to(out_dtype)
    E = 1
    for d in lead:
        E *= d
    w3 = weight.reshape(E, out_f, in_f)
    s3 = scale_inv.reshape(E, so, si)
    out = torch.empty(E, out_f, in_f, dtype=out_dtype, device=weight.device)
    for st in range(0, E, expert_chunk):
        en = min(st + expert_chunk, E)
        wc = w3[st:en].to(torch.float32).reshape(en - st, so, bm, si, bn)
        sc = s3[st:en].to(torch.float32).reshape(en - st, so, 1, si, 1)
        out[st:en] = (wc * sc).reshape(en - st, out_f, in_f).to(out_dtype)
    return out.reshape(*lead, out_f, in_f)


def rand_fp8(shape, device):
    return (torch.randn(shape, device=device, dtype=torch.float32) * 8.0).to(torch.float8_e4m3fn)


def rand_scale(out_f, in_f, lead, device):
    bm, bn = BLOCK
    shape = (*lead, out_f // bm, in_f // bn)
    return (torch.rand(shape, device=device, dtype=torch.float32) * 0.02 + 1e-3)


def bench_one(name, weight, scale_inv, iters=50, warmup=10):
    dev = weight.device
    g = torch.randn(tuple(weight.shape), device=dev, dtype=torch.bfloat16)  # upstream grad

    # ---- correctness: forward ----
    out_t = dequantize_fp8_blockwise_triton(weight, scale_inv, BLOCK)
    out_p = pytorch_dequant(weight, scale_inv, BLOCK)
    fwd_max = (out_t.float() - out_p.float()).abs().max().item()

    # ---- correctness: backward (grad wrt scale_inv) ----
    s_t = scale_inv.clone().requires_grad_(True)
    o_t = dequantize_fp8_blockwise_triton(weight, s_t, BLOCK)
    (o_t.float() * g.float()).sum().backward()
    gs_t = s_t.grad.detach().clone()

    s_p = scale_inv.clone().requires_grad_(True)
    o_p = pytorch_dequant(weight, s_p, BLOCK)
    (o_p.float() * g.float()).sum().backward()
    gs_p = s_p.grad.detach().clone()
    bwd_max = (gs_t - gs_p).abs().max().item()
    bwd_rel = bwd_max / (gs_p.abs().max().item() + 1e-12)

    def timed(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            fn()
        en.record(); torch.cuda.synchronize()
        return st.elapsed_time(en) / iters

    # forward timing (no autograd)
    with torch.no_grad():
        t_fwd_tri = timed(lambda: dequantize_fp8_blockwise_triton(weight, scale_inv, BLOCK))
        t_fwd_pt = timed(lambda: pytorch_dequant(weight, scale_inv, BLOCK))

    # backward timing: forward+backward for grad wrt scale (the differentiable input)
    def fb_tri():
        s = scale_inv.detach().requires_grad_(True)
        o = dequantize_fp8_blockwise_triton(weight, s, BLOCK)
        (o.float() * g.float()).sum().backward()

    def fb_pt():
        s = scale_inv.detach().requires_grad_(True)
        o = pytorch_dequant(weight, s, BLOCK)
        (o.float() * g.float()).sum().backward()

    t_fb_tri = timed(fb_tri)
    t_fb_pt = timed(fb_pt)

    print(f"\n[{name}]  weight {tuple(weight.shape)}  {weight.numel()/1e6:.1f}M elems")
    print(f"  correctness  fwd max_abs={fwd_max:.3e}   bwd(scale) max_abs={bwd_max:.3e} rel={bwd_rel:.3e}")
    print(f"  forward      triton={t_fwd_tri*1e3:8.1f}us  pytorch={t_fwd_pt*1e3:8.1f}us   speedup={t_fwd_pt/t_fwd_tri:5.2f}x")
    print(f"  fwd+bwd      triton={t_fb_tri*1e3:8.1f}us  pytorch={t_fb_pt*1e3:8.1f}us   speedup={t_fb_pt/t_fb_tri:5.2f}x")
    ok = fwd_max < 1e-2 and bwd_rel < 1e-3
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=3072)
    ap.add_argument("--inter", type=int, default=1536)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"
    dev = "cuda"
    torch.manual_seed(0)
    E, H, I = args.experts, args.hidden, args.inter
    print(f"device={torch.cuda.get_device_name(0)}  E={E} H={H} I={I}  block={BLOCK}")

    cases = []
    # 2D attention projection (out, in)
    cases.append(("attn_proj_2d", (H, H)))
    # stacked MoE experts: gate_up (E, 2I, H), down (E, H, I)
    cases.append(("moe_gate_up", (E, 2 * I, H)))
    cases.append(("moe_down", (E, H, I)))

    all_ok = True
    for name, shape in cases:
        *lead, out_f, in_f = shape
        w = rand_fp8(shape, dev)
        s = rand_scale(out_f, in_f, tuple(lead), dev)
        all_ok = bench_one(name, w, s) and all_ok

    print("\n================ summary ================")
    print("PASS - Triton matches PyTorch (fwd + bwd) " if all_ok else "FAIL - mismatch beyond tolerance")
    assert all_ok


if __name__ == "__main__":
    main()
