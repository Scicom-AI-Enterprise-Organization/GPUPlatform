"""Compare the Triton FP8 dequant (`dequant_triton.py`) vs the torch reference (`lora.dequantize_fp8`).

Three checks, on the REAL Mistral-Small-4 per-tensor weight shapes (CUDA only):

  [FWD] numerical parity of the dequantized weight (Triton vs torch), 2D (FP8Linear) + 3D (FP8Experts).
  [BWD] full forward+backward through the actual `LinearLoRA` training op (frozen FP8 base + bf16
        LoRA): same inputs/adapters, torch-dequant vs Triton-dequant, compare the forward output AND
        every gradient (d/dx, d/d lora_a, d/d lora_b). The frozen base needs no grad, so this proves
        swapping in the Triton dequant does not change what training sees.
  [SPEED] wall-clock (CUDA events) + peak memory of the dequant op alone, 2D + 3D. Triton fuses
        fp8->bf16*scale into one pass (no fp32 transient) — expect a speed + peak-memory win.

Run on a GPU:  python bench_dequant.py
"""
import time

import torch
import torch.nn.functional as F

import lora as L
import dequant_triton as DT


# ---------------------------------------------------------------------------
def quantize_per_tensor(w):
    """bf16/fp32 weight -> (fp8 weight, inverse scale). 2D -> scalar scale; 3D -> (E,1,1)."""
    lead = w.shape[:-2]
    flat = w.float().reshape(*lead, -1) if lead else w.float().reshape(-1)
    max_abs = flat.abs().amax(dim=-1)
    max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    fmax = torch.finfo(torch.float8_e4m3fn).max
    scale = fmax / max_abs                                   # (*lead,) or scalar
    s_b = scale.reshape(*lead, 1, 1) if lead else scale.reshape(1, 1)
    q = torch.clamp(w.float() * s_b, min=torch.finfo(torch.float8_e4m3fn).min, max=fmax).to(torch.float8_e4m3fn)
    scale_inv = (1.0 / scale)
    scale_inv = scale_inv.reshape(*lead, 1, 1) if lead else scale_inv.reshape(())
    return q, scale_inv


def diff(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().max().item(), ((a - b).norm() / (b.norm() + 1e-12)).item()


# ---------------------------------------------------------------------------
class FakeFP8Linear(torch.nn.Module):
    """Per-tensor FP8Linear stand-in: fp8 weight + scalar weight_scale_inv, block_size=None."""

    def __init__(self, out_f, in_f, device):
        super().__init__()
        base = torch.randn(out_f, in_f, device=device) * 0.02
        q, s = quantize_per_tensor(base)
        self.weight = torch.nn.Parameter(q, requires_grad=False)
        self.weight_scale_inv = torch.nn.Parameter(s.to(device), requires_grad=False)
        self.block_size = None
        self.in_features, self.out_features = in_f, out_f
        self.bias = None


def fwd_parity(device):
    print("\n================ [FWD] dequant parity (Triton vs torch) ================")
    cases = [("2D o_proj  (4096x4096)", (4096, 4096)),
             ("2D kv_b    (6144x256) ", (6144, 256)),
             ("3D experts (128,4096,4096)", (128, 4096, 4096))]
    ok = True
    for name, shape in cases:
        w = torch.randn(*shape, device=device) * 0.02
        q, s = quantize_per_tensor(w)
        d_torch = L.dequantize_fp8(q, s, None, out_dtype=torch.bfloat16)
        d_tri = DT.dequantize_fp8_triton(q, s, None, out_dtype=torch.bfloat16)
        ma, rel = diff(d_tri, d_torch)
        passed = ma < 1e-2 and rel < 1e-3   # both go fp8->fp32*scale->bf16; expect ~bit-identical
        ok &= passed
        print(f"  [{'OK ' if passed else 'FAIL'}] {name}: max_abs={ma:.3e} rel_l2={rel:.3e}")
    return ok


def bwd_parity(device):
    print("\n================ [BWD] LinearLoRA fwd+bwd parity (torch vs Triton dequant) ================")
    torch.manual_seed(0)
    base = FakeFP8Linear(4096, 4096, device)
    m = L.LinearLoRA(base, r=16, alpha=16.0, lora_dtype=torch.bfloat16).to(device)
    with torch.no_grad():
        m.lora_b.weight.normal_(std=0.02)   # non-zero adapter so there are real LoRA grads
    x0 = torch.randn(512, 4096, device=device, dtype=torch.bfloat16)
    g = torch.randn(512, 4096, device=device, dtype=torch.bfloat16)

    def run():
        x = x0.clone().requires_grad_(True)
        for p in (m.lora_a.weight, m.lora_b.weight):
            p.grad = None
        y = m(x)
        (y.float() * g.float()).sum().backward()
        return (y.detach().clone(), x.grad.detach().clone(),
                m.lora_a.weight.grad.detach().clone(), m.lora_b.weight.grad.detach().clone())

    L._TRITON_DEQUANT = None                 # torch dequant path
    y_t, gx_t, ga_t, gb_t = run()
    L._TRITON_DEQUANT = DT.dequantize_fp8_triton   # Triton dequant path
    y_r, gx_r, ga_r, gb_r = run()
    L._TRITON_DEQUANT = None

    ok = True
    for name, a, b in [("forward y", y_r, y_t), ("grad d/dx", gx_r, gx_t),
                       ("grad d/d lora_a", ga_r, ga_t), ("grad d/d lora_b", gb_r, gb_t)]:
        ma, rel = diff(a, b)
        passed = ma < 1e-2 and rel < 5e-3
        ok &= passed
        print(f"  [{'OK ' if passed else 'FAIL'}] {name}: max_abs={ma:.3e} rel_l2={rel:.3e}")
    return ok


def _time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters, torch.cuda.max_memory_allocated() / 1e9


def speed(device):
    print("\n================ [SPEED] dequant op (Triton vs torch) ================")
    cases = [("2D o_proj  (4096x4096)", (4096, 4096)),
             ("3D experts (128,4096,4096)", (128, 4096, 4096)),
             ("3D down    (128,4096,2048)", (128, 4096, 2048))]
    for name, shape in cases:
        w = torch.randn(*shape, device=device) * 0.02
        q, s = quantize_per_tensor(w)
        t_torch, m_torch = _time(lambda: L.dequantize_fp8(q, s, None, out_dtype=torch.bfloat16))
        t_tri, m_tri = _time(lambda: DT.dequantize_fp8_triton(q, s, None, out_dtype=torch.bfloat16))
        elems = 1
        for d in shape:
            elems *= d
        print(f"  {name}: torch={t_torch:.3f}ms (peak {m_torch:.2f}GB)  "
              f"triton={t_tri:.3f}ms (peak {m_tri:.2f}GB)  speedup={t_torch/t_tri:.2f}x  "
              f"mem={m_torch/max(m_tri,1e-9):.2f}x  ({elems/1e6:.0f}M elems)")


def main():
    assert torch.cuda.is_available(), "bench_dequant needs a CUDA GPU"
    dev = "cuda"
    print(f"torch {torch.__version__}  device={torch.cuda.get_device_name(0)}  triton_available={DT._HAVE_TRITON}")
    f = fwd_parity(dev)
    b = bwd_parity(dev)
    speed(dev)
    print("\n================ summary ================")
    print(f"[FWD] parity ........ {'PASS' if f else 'FAIL'}")
    print(f"[BWD] parity ........ {'PASS' if b else 'FAIL'}")
    assert f and b, "Triton dequant diverged from the torch reference"
    print("PASS - Triton dequant matches torch on forward + backward; see speed table above.")


if __name__ == "__main__":
    main()
