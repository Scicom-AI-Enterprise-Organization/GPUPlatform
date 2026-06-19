"""Triton blockwise FP8 dequant for `lora.py` — a faster, autograd-capable drop-in for
`dequantize_fp8_blockwise`.

The op: a block-scaled FP8 weight (`float8_e4m3fn`, block `bm x bn`, default 128x128) ->
`out_dtype` (bf16):

    out[..., o, i] = float32(weight[..., o, i]) * scale_inv[..., o // bm, i // bn]

The PyTorch reference (`lora.dequantize_fp8_blockwise`) upcasts the whole tensor to fp32 and
forms an fp32 product (~4x the bf16 size transiently); even chunked over experts it reads/writes
several large temporaries. This Triton kernel fuses upcast * scale * downcast into a single pass
with **one scale load per 128x128 block** and no fp32 temporary in HBM.

Autograd ("support backward"):
  * The FP8 `weight` is the output of a quantizer and is **not differentiable** — there is no
    meaningful gradient through fp8 rounding — so its grad is `None` (matches how it's used:
    a frozen base weight).
  * `scale_inv` IS differentiable. d out / d scale_inv[b] = sum over the block of
    float32(weight), so grad_scale[b] = sum_{o,i in block} grad_out[o,i] * float32(weight[o,i]).
    Implemented as a second Triton kernel (one reduction per block). Verified against PyTorch
    autograd in `bench_dequant.py`.

3D stacked expert weights `(E, out, in)` (scale `(E, out/bm, in/bn)`) are handled by viewing them
as 2D `(E*out, in)` / `(E*out/bm, in/bn)` — block index `r//bm` is expert-consistent because
`out % bm == 0`, so the same 2D kernel covers both.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_fwd_kernel(w_ptr, s_ptr, o_ptr, M, N, S_STRIDE, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    # int64 offsets: row*N overflows int32 for the big MoE tensors (E*out*in > 2^31).
    offs = rm[:, None].to(tl.int64) * N + rn[None, :].to(tl.int64)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + pm * S_STRIDE + pn).to(tl.float32)   # one scale per block
    tl.store(o_ptr + offs, (w * s).to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _dequant_bwd_scale_kernel(g_ptr, w_ptr, gs_ptr, M, N, S_STRIDE, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    offs = rm[:, None].to(tl.int64) * N + rn[None, :].to(tl.int64)   # int64: avoid int32 overflow
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(gs_ptr + pm * S_STRIDE + pn, tl.sum(g * w))      # grad wrt this block's scale


def _check_tiling(out_f, in_f, bm, bn):
    if out_f % bm != 0 or in_f % bn != 0:
        raise ValueError(f"FP8 dequant: ({out_f},{in_f}) not divisible by block ({bm},{bn}).")


def _triton_dequant_2d(weight2d, scale2d, bm, bn, out_dtype):
    M, N = weight2d.shape
    out = torch.empty((M, N), dtype=out_dtype, device=weight2d.device)
    grid = (M // bm, N // bn)
    _dequant_fwd_kernel[grid](weight2d, scale2d, out, M, N, N // bn, BM=bm, BN=bn)
    return out


def _triton_grad_scale_2d(grad2d, weight2d, bm, bn):
    M, N = weight2d.shape
    gs = torch.empty((M // bm, N // bn), dtype=torch.float32, device=weight2d.device)
    grid = (M // bm, N // bn)
    _dequant_bwd_scale_kernel[grid](grad2d.contiguous(), weight2d, gs, M, N, N // bn, BM=bm, BN=bn)
    return gs


class _BlockFP8Dequant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale_inv, bm, bn, out_dtype):
        *lead, out_f, in_f = weight.shape
        _check_tiling(out_f, in_f, bm, bn)
        E = 1
        for d in lead:
            E *= d
        w2 = weight.reshape(E * out_f, in_f) if lead else weight
        s2 = scale_inv.reshape((E * out_f) // bm, in_f // bn) if lead else scale_inv
        out = _triton_dequant_2d(w2, s2.contiguous(), bm, bn, out_dtype)
        ctx.save_for_backward(w2)
        ctx.bm, ctx.bn = bm, bn
        ctx.lead, ctx.out_f, ctx.in_f = tuple(lead), out_f, in_f
        ctx.scale_shape = scale_inv.shape
        ctx.scale_needs_grad = scale_inv.requires_grad
        return out.reshape(*lead, out_f, in_f) if lead else out

    @staticmethod
    def backward(ctx, grad_out):
        grad_scale = None
        if ctx.scale_needs_grad:
            (w2,) = ctx.saved_tensors
            g2 = grad_out.reshape(w2.shape)
            grad_scale = _triton_grad_scale_2d(g2, w2, ctx.bm, ctx.bn).reshape(ctx.scale_shape)
        # weight (fp8) is non-differentiable -> None; bm, bn, out_dtype are non-tensor -> None.
        return None, grad_scale, None, None, None


def dequantize_fp8_blockwise_triton(weight, scale_inv, block_size, out_dtype=torch.bfloat16):
    """Triton drop-in for `lora.dequantize_fp8_blockwise` (same signature + semantics)."""
    bm, bn = block_size
    return _BlockFP8Dequant.apply(weight, scale_inv, bm, bn, out_dtype)
