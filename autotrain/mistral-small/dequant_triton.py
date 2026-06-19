"""Fused Triton FP8 dequant for Mistral-Small-4 (per-tensor) — the fast path for `lora.py`.

The minimax-m2 sibling shipped a Triton block-dequant (`dequant_triton.py`, ~9x). Mistral-Small-4
is **per-tensor** FP8 (`weight_block_size=null`): each 2D `FP8Linear` weight has a single scalar
`weight_scale_inv`, and each 3D `FP8Experts` weight has a per-expert `(E,1,1)` scalar. So the
dequant is `bf16(w) = float8_to_float32(w) * scale_inv`.

The torch reference (`lora.dequantize_fp8`) does `w.to(fp32) * scale -> .to(bf16)`, which
materialises a full **fp32 transient** (4 bytes/elem) on top of the fp8 input (1 byte) and the
bf16 output (2 bytes) — ~3 separate passes over memory. This kernel fuses it into a **single
pass**: load fp8, multiply by the (per-expert) scale in registers, store bf16 — no fp32 transient.
For the 3D routed-expert weight (E=128, 2I=4096, H=4096 ≈ 2.1B elems) that is a large bandwidth +
peak-memory win; see `bench_dequant.py`.

The frozen base weight needs no gradient (QLoRA: grads flow through `F.linear` to the activations
/ the LoRA adapters, not the frozen weight), so this is a plain forward op — no custom autograd.
`lora.py` routes its per-tensor dequant here when `MISTRAL_DEQUANT_TRITON=1` (or when
`lora._TRITON_DEQUANT` is assigned) and the weight is on CUDA; everything else falls back to torch.

    from dequant_triton import dequantize_fp8_triton
    w_bf16 = dequantize_fp8_triton(fp8_weight, scale_inv)        # per-tensor
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _dequant_per_tensor_kernel(
        w_ptr,          # *fp8e4nv   (N,)  flattened weight
        s_ptr,          # *fp32      (E,)  per-expert inverse scales
        o_ptr,          # *out_dtype (N,)  flattened output
        n_elements,     # int        total elements N
        rows_per_expert,  # int      out*in  (== N for a 2D weight, so expert is always 0)
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        w = tl.load(w_ptr + offs, mask=mask).to(tl.float32)   # fp8 -> fp32 in-register
        expert = offs // rows_per_expert                       # which expert this element belongs to
        s = tl.load(s_ptr + expert, mask=mask, other=1.0)      # gather per-expert scale (scalar for 2D)
        o = w * s
        tl.store(o_ptr + offs, o.to(o_ptr.dtype.element_ty), mask=mask)


def _torch_per_tensor(weight, scale_inv, out_dtype):
    """Reference per-tensor dequant (also the CPU fallback). Materialises an fp32 transient."""
    return (weight.to(torch.float32) * scale_inv.to(torch.float32)).to(out_dtype)


def _torch_block(weight, scale_inv, block_size, out_dtype):
    """Minimal block-scaled dequant fallback (Mistral-Small-4 is per-tensor; here for completeness)."""
    bm, bn = block_size
    *lead, out_f, in_f = weight.shape
    so, si = out_f // bm, in_f // bn
    w = weight.to(torch.float32).reshape(*lead, so, bm, si, bn)
    s = scale_inv.to(torch.float32).reshape(*lead, so, 1, si, 1)
    return (w * s).reshape(*lead, out_f, in_f).to(out_dtype)


def dequantize_fp8_triton(weight, scale_inv, block_size=None, out_dtype=torch.bfloat16):
    """Fused single-pass FP8->`out_dtype` dequant.

    Args mirror `lora.dequantize_fp8`:
        weight:    (..., out, in) float8_e4m3fn   (2D FP8Linear or 3D FP8Experts)
        scale_inv: per-tensor scalar / (E,1,1) per-expert   (block grid if block_size given)
        block_size: None for per-tensor (the Triton path); a (bm,bn) tuple falls back to torch.
    """
    if block_size is not None:
        return _torch_block(weight, scale_inv, block_size, out_dtype)
    if not (_HAVE_TRITON and weight.is_cuda):
        return _torch_per_tensor(weight, scale_inv, out_dtype)

    *lead, out_f, in_f = weight.shape
    rows_per_expert = out_f * in_f
    E = 1
    for d in lead:
        E *= d
    n = weight.numel()

    w_flat = weight.reshape(-1)                                  # contiguous fp8 view/copy
    s_flat = scale_inv.reshape(-1).to(torch.float32).contiguous()  # (E,)  (1, for 2D)
    if s_flat.numel() != E:  # a true scalar (0-dim) reshapes to (1,); broadcast defensively
        s_flat = s_flat.reshape(()).expand(E).contiguous() if s_flat.numel() == 1 else s_flat.contiguous()

    out = torch.empty(n, dtype=out_dtype, device=weight.device)
    BLOCK = 2048
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _dequant_per_tensor_kernel[grid](w_flat, s_flat, out, n, rows_per_expert, BLOCK=BLOCK)
    return out.reshape(weight.shape)
