"""Correctness tests for the MiniMax-M2 LoRA (`lora.py`) — runnable on CPU, no GPU/Triton.

These check the parts the training run lives or dies on:

  1. `dequantize_fp8_blockwise` reconstructs a block-scaled FP8 weight (2D attn + 3D experts).
  2. `LinearLoRA`: zero-B init is a no-op; with a non-zero B it equals base + scaling*B(A(x)).
  3. `fused_lora_experts_forward`: the fused grouped-MoE forward (base + per-expert LoRA,
     SwiGLU gate applied to base+LoRA before the down projection) matches an independent
     per-token/per-expert loop reference, on BOTH the forward output AND the LoRA gradients.
     A zero-B adapter must also reduce to the pure base MoE.

Run: python test_lora.py
"""
import os
import sys

# Force the autograd-registered grouped_mm loop fallback (torch._grouped_mm is CUDA-only).
os.environ.setdefault("MINIMAX_GROUPED_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

import lora as L


def report(name, got, ref, atol, rtol):
    got, ref = got.float(), ref.float()
    max_abs = (got - ref).abs().max().item()
    denom = ref.norm().item() or 1.0
    rel = (got - ref).norm().item() / denom
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max_abs={max_abs:.3e} rel_l2={rel:.3e}")
    return ok


# ---------------------------------------------------------------------------
def test_dequant():
    """Blockwise FP8 dequant round-trips a quantized weight (2D and 3D), within fp8 error."""
    torch.manual_seed(0)
    bm = bn = 4  # tiny blocks for the test
    block = (bm, bn)

    def quantize(w):  # mirror transformers' Fp8Quantize per-block (max-abs) scaling
        out_f, in_f = w.shape[-2], w.shape[-1]
        so, si = out_f // bm, in_f // bn
        lead = w.shape[:-2]
        wr = w.float().reshape(*lead, so, bm, si, bn)
        max_abs = wr.abs().amax(dim=(-3, -1))
        max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
        fmax = torch.finfo(torch.float8_e4m3fn).max
        scales = fmax / max_abs
        q = torch.clamp(wr * scales.unsqueeze(-1).unsqueeze(-3),
                        min=torch.finfo(torch.float8_e4m3fn).min, max=fmax)
        q = q.to(torch.float8_e4m3fn).reshape(*lead, out_f, in_f)
        return q, (1.0 / scales).float()

    ok = True
    for shape in [(8, 12), (3, 8, 12)]:
        w = torch.randn(*shape)
        q, scale_inv = quantize(w)
        deq = L.dequantize_fp8_blockwise(q, scale_inv, block, out_dtype=torch.float32)
        assert deq.shape == w.shape, f"{deq.shape} != {w.shape}"
        # fp8 e4m3 has ~3 mantissa bits, so per-element error is large but rel_l2 should be small.
        rel = (deq - w).norm().item() / (w.norm().item() or 1.0)
        passed = rel < 0.05  # block round-trip should track the original to a few %
        print(f"[{'PASS' if passed else 'FAIL'}] fp8 dequant round-trip {tuple(shape)}: "
              f"rel_l2={rel:.3e} (fp8-lossy, < 0.05 expected)")
        ok &= passed
    return ok


# ---------------------------------------------------------------------------
def test_linear_lora():
    """Zero-B = no-op; random-B = base + scaling * B(A(x))."""
    torch.manual_seed(1)
    base = nn.Linear(16, 24, bias=False)
    x = torch.randn(5, 16)

    m = L.LinearLoRA(base, r=4, alpha=8.0, lora_dtype=torch.float32)
    ok = report("LinearLoRA zero-B == base", m(x), base(x), atol=1e-6, rtol=1e-5)

    with torch.no_grad():
        m.lora_b.weight.normal_()  # make the adapter non-trivial
    expect = base(x) + m.scaling * m.lora_b(m.lora_a(x))
    ok &= report("LinearLoRA base + scaling*B(A(x))", m(x), expect, atol=1e-5, rtol=1e-4)
    return ok


# ---------------------------------------------------------------------------
class DummyExperts(nn.Module):
    """A bf16/fp32 stand-in for FP8Experts: same param names + _apply_gate + act_fn."""

    def __init__(self, E, H, I, dtype=torch.float32):
        super().__init__()
        self.num_experts = E
        self.block_size = None
        self.act_fn = nn.SiLU()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, H, dtype=dtype) * 0.05, requires_grad=False)
        self.down_proj = nn.Parameter(torch.randn(E, H, I, dtype=dtype) * 0.05, requires_grad=False)

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate) * up


def reference_experts(experts, x, top_k_index, top_k_weights):
    """Independent per-token/per-expert reference for the LoRA-MoE forward."""
    E = experts.num_experts
    sc = experts.lora_scaling
    out = torch.zeros_like(x)
    for t in range(x.size(0)):
        acc = torch.zeros(x.size(1))
        for j in range(top_k_index.size(1)):
            e = int(top_k_index[t, j])
            w = top_k_weights[t, j]
            xt = x[t].float()
            gu = xt @ experts.gate_up_proj[e].float().T
            d_gu = (experts.gate_up_lora_b[e].float() @ experts.gate_up_lora_a[e].float())  # (2I,H)
            gu = gu + sc * (xt @ d_gu.T)
            half = gu.size(-1) // 2
            inter = F.silu(gu[:half]) * gu[half:]
            dn = inter @ experts.down_proj[e].float().T
            d_dn = (experts.down_lora_b[e].float() @ experts.down_lora_a[e].float())  # (H,I)
            dn = dn + sc * (inter @ d_dn.T)
            acc = acc + w.float() * dn
        out[t] = acc
    return out


def test_fused_experts():
    torch.manual_seed(2)
    E, H, I, top_k, T = 6, 8, 5, 2, 13
    experts = DummyExperts(E, H, I)
    L.add_expert_lora(experts, r=4, alpha=8.0, lora_dtype=torch.float32)

    x = torch.randn(T, H)
    top_k_index = torch.stack([torch.randperm(E)[:top_k] for _ in range(T)])
    raw = torch.rand(T, top_k)
    top_k_weights = raw / raw.sum(-1, keepdim=True)

    # (a) zero-B adapter must reduce to the pure base MoE.
    base_only = reference_experts(experts, x, top_k_index, top_k_weights)  # B=0 here
    fused0 = experts(x, top_k_index, top_k_weights)
    ok = report("fused MoE (B=0) == base reference", fused0, base_only, atol=1e-4, rtol=1e-3)

    # (b) non-trivial adapter: fused == reference (fwd).
    with torch.no_grad():
        experts.gate_up_lora_b.normal_(std=0.5)
        experts.down_lora_b.normal_(std=0.5)
    ref = reference_experts(experts, x, top_k_index, top_k_weights)
    fused = experts(x, top_k_index, top_k_weights)
    ok &= report("fused MoE + LoRA == reference (fwd)", fused, ref, atol=1e-4, rtol=1e-3)

    # (c) gradients: same upstream grad through fused vs reference -> same LoRA .grad,
    #     and the frozen base params get NO grad.
    xg = x.clone().requires_grad_(True)
    g = torch.randn(T, H)

    for p in (experts.gate_up_lora_a, experts.gate_up_lora_b, experts.down_lora_a, experts.down_lora_b):
        p.grad = None
    (experts(xg, top_k_index, top_k_weights) * g).sum().backward()
    grads_fused = {k: getattr(experts, k).grad.clone() for k in
                   ("gate_up_lora_a", "gate_up_lora_b", "down_lora_a", "down_lora_b")}
    assert experts.gate_up_proj.grad is None and experts.down_proj.grad is None, "frozen base got a grad!"

    for p in (experts.gate_up_lora_a, experts.gate_up_lora_b, experts.down_lora_a, experts.down_lora_b):
        p.grad = None
    (reference_experts(experts, x, top_k_index, top_k_weights) * g).sum().backward()
    grads_ref = {k: getattr(experts, k).grad.clone() for k in grads_fused}

    for k in grads_fused:
        ok &= report(f"  d/d {k}", grads_fused[k], grads_ref[k], atol=1e-4, rtol=1e-3)
    return ok


if __name__ == "__main__":
    print(f"torch {torch.__version__}, MINIMAX_GROUPED_FALLBACK={os.environ.get('MINIMAX_GROUPED_FALLBACK')}")
    results = [
        test_dequant(),
        test_linear_lora(),
        test_fused_experts(),
    ]
    if all(results):
        print(f"\nALL {len(results)} TEST GROUPS PASSED")
        sys.exit(0)
    print(f"\n{sum(1 for r in results if not r)} TEST GROUP(S) FAILED")
    sys.exit(1)
