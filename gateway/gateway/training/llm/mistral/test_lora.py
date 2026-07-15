"""Correctness tests for the Mistral-Small-4 LoRA (`lora.py`) — runnable on CPU, no GPU/Triton.

These check the parts the training run lives or dies on:

  1. `dequantize_fp8` reconstructs an FP8 weight in BOTH the layouts the helper supports:
     per-tensor (Mistral-Small-4: `block_size=None`, scalar / (E,1,1) scale) and 128x128
     block-scaled (sibling models).
  2. `LinearLoRA`: zero-B init is a no-op; with a non-zero B it equals base + scaling*B(A(x)).
     Also exercised against a per-tensor FP8 base (the MLA / shared-expert case).
  3. `fused_lora_experts_forward`: the fused grouped-MoE forward (base + per-expert LoRA,
     SwiGLU gate applied to base+LoRA before the down projection) matches an independent
     per-token/per-expert loop reference, on BOTH the forward output AND the LoRA gradients.
     A zero-B adapter must also reduce to the pure base MoE.

Run: python test_lora.py
"""
import os
import sys

# Force the autograd-registered grouped_mm loop fallback (torch._grouped_mm is CUDA-only).
os.environ.setdefault("MISTRAL_GROUPED_FALLBACK", "1")

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


def _quantize_blocks(w, bm, bn):
    """Mirror transformers' Fp8Quantize per-block (max-abs) scaling -> (fp8 weight, scale_inv)."""
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


def _quantize_per_tensor(w):
    """Per-tensor max-abs FP8 quant (Mistral-Small-4 layout). Returns (fp8 weight, scalar/(E,1,1) scale)."""
    lead = w.shape[:-2]
    flat = w.float().reshape(*lead, -1)
    max_abs = flat.abs().amax(dim=-1)
    max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    fmax = torch.finfo(torch.float8_e4m3fn).max
    scales = fmax / max_abs                      # (*lead,)
    s_b = scales.reshape(*lead, 1, 1)
    q = torch.clamp(w.float() * s_b, min=torch.finfo(torch.float8_e4m3fn).min, max=fmax).to(torch.float8_e4m3fn)
    scale_inv = (1.0 / scales)                    # (*lead,)
    if not lead:
        return q, scale_inv.reshape(())           # scalar (FP8Linear)
    return q, scale_inv.reshape(*lead, 1, 1)      # (E,1,1) (FP8Experts)


# ---------------------------------------------------------------------------
def test_dequant():
    """FP8 dequant round-trips, within fp8 error, for per-tensor (2D + 3D) and block layouts."""
    torch.manual_seed(0)
    ok = True

    # ---- per-tensor (Mistral-Small-4) ----
    for shape in [(8, 12), (3, 8, 12)]:
        w = torch.randn(*shape)
        q, scale_inv = _quantize_per_tensor(w)
        deq = L.dequantize_fp8(q, scale_inv, block_size=None, out_dtype=torch.float32)
        assert deq.shape == w.shape, f"{deq.shape} != {w.shape}"
        rel = (deq - w).norm().item() / (w.norm().item() or 1.0)
        passed = rel < 0.10  # per-tensor fp8 (single scale over the whole matrix) is coarser
        print(f"[{'PASS' if passed else 'FAIL'}] per-tensor fp8 round-trip {tuple(shape)}: "
              f"rel_l2={rel:.3e} (fp8-lossy, < 0.10 expected)")
        ok &= passed

    # ---- block-scaled (128x128 sibling layout; tiny blocks here) ----
    bm = bn = 4
    for shape in [(8, 12), (3, 8, 12)]:
        w = torch.randn(*shape)
        q, scale_inv = _quantize_blocks(w, bm, bn)
        deq = L.dequantize_fp8(q, scale_inv, block_size=(bm, bn), out_dtype=torch.float32)
        assert deq.shape == w.shape
        rel = (deq - w).norm().item() / (w.norm().item() or 1.0)
        passed = rel < 0.05  # per-block scaling tracks the original to a few %
        print(f"[{'PASS' if passed else 'FAIL'}] block fp8 round-trip {tuple(shape)}: "
              f"rel_l2={rel:.3e} (fp8-lossy, < 0.05 expected)")
        ok &= passed
    return ok


# ---------------------------------------------------------------------------
class FakeFP8Linear(nn.Module):
    """Minimal per-tensor FP8Linear stand-in: fp8 weight + scalar weight_scale_inv, block_size=None."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        q, s = _quantize_per_tensor(base.weight.data)
        self.weight = nn.Parameter(q, requires_grad=False)
        self.weight_scale_inv = nn.Parameter(s, requires_grad=False)
        self.block_size = None
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = None


def test_linear_lora():
    """Zero-B = no-op; random-B = base + scaling * B(A(x)). Tested on bf16 and per-tensor FP8 bases."""
    torch.manual_seed(1)
    ok = True

    # bf16 base
    base = nn.Linear(16, 24, bias=False)
    x = torch.randn(5, 16)
    m = L.LinearLoRA(base, r=4, alpha=8.0, lora_dtype=torch.float32)
    ok &= report("LinearLoRA (bf16 base) zero-B == base", m(x), base(x), atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        m.lora_b.weight.normal_()
    expect = base(x) + m.scaling * m.lora_b(m.lora_a(x))
    ok &= report("LinearLoRA (bf16 base) base + scaling*B(A(x))", m(x), expect, atol=1e-5, rtol=1e-4)

    # per-tensor FP8 base: zero-B must reproduce the dequantized base linear.
    fp8 = FakeFP8Linear(base)
    deq_w = L.dequantize_fp8(fp8.weight, fp8.weight_scale_inv, None, out_dtype=torch.float32)
    ref = F.linear(x, deq_w)
    mf = L.LinearLoRA(fp8, r=4, alpha=8.0, lora_dtype=torch.float32)
    assert mf._fp8, "FP8 base not detected by LinearLoRA"
    ok &= report("LinearLoRA (fp8 base) zero-B == dequant(base)", mf(x), ref, atol=1e-4, rtol=1e-3)
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
    """Independent per-token/per-expert reference for the routed LoRA-MoE forward."""
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
    g = torch.randn(T, H)

    for p in (experts.gate_up_lora_a, experts.gate_up_lora_b, experts.down_lora_a, experts.down_lora_b):
        p.grad = None
    (experts(x, top_k_index, top_k_weights) * g).sum().backward()
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


# ---------------------------------------------------------------------------
# DoRA (weight-decomposed LoRA): W_eff = magnitude * normalize(W + scaling*B@A)
# ---------------------------------------------------------------------------
def test_linear_dora():
    """DoRA is identity at init (B=0, mag=‖W‖); with a non-zero B it equals the composed weight."""
    torch.manual_seed(1)
    base = nn.Linear(16, 24, bias=False)
    x = torch.randn(5, 16)

    m = L.LinearLoRA(base, r=4, alpha=8.0, lora_dtype=torch.float32, use_dora=True)
    ok = report("LinearDoRA init (B=0, mag=||W||) == base", m(x), base(x), atol=1e-5, rtol=1e-4)

    with torch.no_grad():
        m.lora_b.weight.normal_()
    W = base.weight.float()
    adapted = W + m.scaling * (m.lora_b.weight.float() @ m.lora_a.weight.float())
    w_eff = m.magnitude.float().unsqueeze(1) * (adapted / adapted.norm(dim=1, keepdim=True))
    ok &= report("LinearDoRA == magnitude*normalize(W+sBA)", m(x), F.linear(x, w_eff), atol=1e-5, rtol=1e-4)
    return ok


def reference_experts_dora(experts, x, top_k_index, top_k_weights):
    """Per-token/per-expert reference for the DoRA-MoE forward (compose W_eff, then apply)."""
    sc = experts.lora_scaling
    out = torch.zeros_like(x)
    for t in range(x.size(0)):
        acc = torch.zeros(x.size(1))
        for j in range(top_k_index.size(1)):
            e = int(top_k_index[t, j]); w = top_k_weights[t, j].float(); xt = x[t].float()
            gu_w = experts.gate_up_proj[e].float() + sc * (experts.gate_up_lora_b[e].float() @ experts.gate_up_lora_a[e].float())
            gu_eff = experts.gate_up_mag[e].float().unsqueeze(1) * (gu_w / gu_w.norm(dim=1, keepdim=True))
            gu = xt @ gu_eff.T
            half = gu.size(-1) // 2
            inter = F.silu(gu[:half]) * gu[half:]
            dn_w = experts.down_proj[e].float() + sc * (experts.down_lora_b[e].float() @ experts.down_lora_a[e].float())
            dn_eff = experts.down_mag[e].float().unsqueeze(1) * (dn_w / dn_w.norm(dim=1, keepdim=True))
            acc = acc + w * (inter @ dn_eff.T)
        out[t] = acc
    return out


def test_fused_experts_dora():
    torch.manual_seed(2)
    E, H, I, top_k, T = 6, 8, 5, 2, 13
    experts = DummyExperts(E, H, I)
    L.add_expert_lora(experts, r=4, alpha=8.0, lora_dtype=torch.float32, use_dora=True)

    x = torch.randn(T, H)
    top_k_index = torch.stack([torch.randperm(E)[:top_k] for _ in range(T)])
    raw = torch.rand(T, top_k)
    top_k_weights = raw / raw.sum(-1, keepdim=True)

    base_ref = reference_experts(experts, x, top_k_index, top_k_weights)  # B=0 => base
    ok = report("fused MoE DoRA (init) == base reference", experts(x, top_k_index, top_k_weights), base_ref, atol=1e-4, rtol=1e-3)

    with torch.no_grad():
        experts.gate_up_lora_b.normal_(std=0.5)
        experts.down_lora_b.normal_(std=0.5)
    ref = reference_experts_dora(experts, x, top_k_index, top_k_weights)
    ok &= report("fused MoE DoRA == reference (fwd)", experts(x, top_k_index, top_k_weights), ref, atol=1e-4, rtol=1e-3)

    g = torch.randn(T, H)
    trainables = ("gate_up_lora_a", "gate_up_lora_b", "down_lora_a", "down_lora_b", "gate_up_mag", "down_mag")
    for k in trainables:
        getattr(experts, k).grad = None
    (experts(x, top_k_index, top_k_weights) * g).sum().backward()
    assert experts.gate_up_proj.grad is None and experts.down_proj.grad is None, "frozen base got a grad!"
    for k in trainables:
        gr = getattr(experts, k).grad
        good = gr is not None and torch.isfinite(gr).all() and gr.abs().sum() > 0
        print(f"[{'PASS' if good else 'FAIL'}]   d/d {k}: finite+nonzero grad")
        ok &= bool(good)
    return ok


if __name__ == "__main__":
    print(f"torch {torch.__version__}, MISTRAL_GROUPED_FALLBACK={os.environ.get('MISTRAL_GROUPED_FALLBACK')}")
    results = [
        test_dequant(),
        test_linear_lora(),
        test_fused_experts(),
        test_linear_dora(),
        test_fused_experts_dora(),
    ]
    if all(results):
        print(f"\nALL {len(results)} TEST GROUPS PASSED")
        sys.exit(0)
    print(f"\n{sum(1 for r in results if not r)} TEST GROUP(S) FAILED")
    sys.exit(1)
