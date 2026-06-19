"""Custom LoRA for MiniMax-M2 (FP8 MoE) packed LoRA finetuning.

This is the *interesting part* of the MiniMax-M2 run, the analogue of gemma4's
`attention.py`. Where gemma-4 needed a custom per-layer attention (dual head_dim
512/256), MiniMax-M2 has a *uniform* head_dim 128 (every layer fits FlashAttention),
so there is **no custom attention here** — the model runs with stock
`attn_implementation="flash_attention_2"` and FlashAttention's native varlen packing.

The hard part for MiniMax-M2 is instead the **FP8 MoE**:

  * The released weights are FP8 block-quantized (`float8_e4m3fn`, block 128x128).
    Transformers loads them as `FP8Linear` (q/k/v/o) and `FP8Experts`
    (3D stacked expert params `gate_up_proj` (E,2I,H) + `down_proj` (E,H,I) with
    per-block fp32 `*_scale_inv`).
  * Transformers' FP8 forward kernels (Triton `w8a8` / DeepGEMM, and the per-expert
    activation quant) are **inference-only: they are NOT autograd-differentiable**.
    LoRA training needs gradients to flow *through* the frozen base of every layer to
    reach LoRA params in earlier layers, so a non-differentiable frozen forward breaks
    training for all LoRA below it.

So for TRAINING we replace the frozen base forward with a **differentiable** path
(exactly QLoRA's trick): keep the weight stored in FP8 (cheap), and inside the forward
dequantize the block-scaled weight to bf16 on the fly and run a normal (differentiable)
matmul. The dequantized bf16 weight is transient — activation checkpointing on each
decoder layer recomputes it in the backward instead of retaining it, so peak memory
stays ~ one layer's bf16 weights, not the whole model's.

  * Attention LoRA   ->  `LinearLoRA` wraps the frozen `FP8Linear` (q/k/v/o):
                         y = dequant_linear(x) + scaling * B(A(x)).
  * MoE expert LoRA  ->  per-expert low-rank adapters folded into a **fused grouped_mm**
                         experts forward: gate_up and down each compute
                         (frozen dequant grouped_mm) + (bf16 LoRA grouped_mm), with the
                         SwiGLU gate applied to the *sum* (base+LoRA) before the down
                         projection (the gate is non-linear, so base and LoRA cannot be
                         split after the block — they must be combined at each projection).

Everything here is base-agnostic (fp8 OR bf16 weights) so the grouped LoRA math is unit
tested on CPU in bf16 with no GPU / Triton — see `test_lora.py`.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# transformers ships the grouped-GEMM helpers + the autograd-registered CPU fallback op.
# Importing the module registers `torch.ops.transformers.grouped_mm_fallback`.
from transformers.integrations import moe as _moe


# Set MINIMAX_GROUPED_FALLBACK=1 to force the autograd-registered loop fallback for
# torch._grouped_mm (used by the CPU unit test; torch._grouped_mm has no CPU kernel).
_FORCE_FALLBACK = os.environ.get("MINIMAX_GROUPED_FALLBACK", "0") == "1"

# Triton blockwise FP8 dequant (bit-exact fwd, ~3.8x-9x faster than the PyTorch path on H100;
# verified in bench_dequant.py). Used on CUDA only; CPU (the unit test) keeps the torch path.
# Disable with MINIMAX_DEQUANT_TRITON=0. Import is guarded so a triton-less box (laptop) still works.
_USE_TRITON_DEQUANT = os.environ.get("MINIMAX_DEQUANT_TRITON", "1") != "0"
try:
    from dequant_triton import dequantize_fp8_blockwise_triton as _triton_dequant
except Exception:
    _triton_dequant = None


# ---------------------------------------------------------------------------
# FP8 block dequant (differentiable w.r.t. the matmul input; the weight is frozen)
# ---------------------------------------------------------------------------
def dequantize_fp8_blockwise(
    weight: torch.Tensor, scale_inv: torch.Tensor, block_size, out_dtype=torch.bfloat16
) -> torch.Tensor:
    """Dequantize a block-scaled FP8 weight to `out_dtype`.

    Mirrors transformers' `Fp8Dequantize`:  w_deq = fp8_to_fp32(w) * scale_inv (per block).

    Args:
        weight:    (..., out, in) float8_e4m3fn  (leading dims optional, e.g. num_experts)
        scale_inv: (..., ceil(out/bm), ceil(in/bn)) float32
        block_size: (bm, bn)
    Returns:
        (..., out, in) in `out_dtype`.

    NOTE: MiniMax-M2's projection dims (3072, 1536) are all multiples of 128, so no
    padding is needed; we assert exact tiling rather than silently mis-scaling.
    """
    # CUDA fast path: the Triton kernel (bit-exact fwd, autograd-capable, ~9x on MoE tensors).
    if _USE_TRITON_DEQUANT and _triton_dequant is not None and weight.is_cuda:
        return _triton_dequant(weight, scale_inv, block_size, out_dtype)

    bm, bn = block_size
    *lead, out_f, in_f = weight.shape
    if out_f % bm != 0 or in_f % bn != 0:
        raise ValueError(
            f"FP8 weight {tuple(weight.shape)} not divisible by block {block_size}; "
            "blockwise dequant assumes exact tiling (MiniMax-M2 dims are multiples of 128)."
        )
    so, si = out_f // bm, in_f // bn

    # 2D (attention q/k/v/o) weights are small — dequant in one shot.
    if not lead:
        w = weight.to(torch.float32).reshape(so, bm, si, bn)
        s = scale_inv.to(torch.float32).reshape(so, 1, si, 1)
        return (w * s).reshape(out_f, in_f).to(out_dtype)

    # Stacked (E, out, in) expert weights: a single-shot dequant would upcast ALL experts to
    # fp32 (~2x the bf16 size) AND build an fp32 product on top — ~4x the bf16 weight transiently
    # (e.g. ~36GB for a 256-expert MoE layer), which OOMs a full per-layer (non-FSDP) forward.
    # Dequant in expert CHUNKS into a preallocated bf16 buffer instead: bit-identical math (each
    # expert dequants independently), but the fp32 transient is bounded to `chunk` experts.
    # (Under FSDP each rank holds 1/world_size of the experts, so this only matters for the
    # full-model paths — compare_logits.py / merge_infer.py.)
    E = 1
    for d in lead:
        E *= d
    w3 = weight.reshape(E, out_f, in_f)
    s3 = scale_inv.reshape(E, so, si)
    out = torch.empty(E, out_f, in_f, dtype=out_dtype, device=weight.device)
    chunk = max(1, int(os.environ.get("MINIMAX_DEQUANT_EXPERT_CHUNK", "16")))
    for st in range(0, E, chunk):
        en = min(st + chunk, E)
        wc = w3[st:en].to(torch.float32).reshape(en - st, so, bm, si, bn)
        sc = s3[st:en].to(torch.float32).reshape(en - st, so, 1, si, 1)
        out[st:en] = (wc * sc).reshape(en - st, out_f, in_f).to(out_dtype)
    return out.reshape(*lead, out_f, in_f)


def _is_fp8(module_or_tensor) -> bool:
    t = module_or_tensor.weight if hasattr(module_or_tensor, "weight") else module_or_tensor
    return isinstance(t, torch.Tensor) and t.element_size() == 1


# ---------------------------------------------------------------------------
# grouped matmul: x (S, in) sorted-by-expert @ weight (E, out, in).T  -> (S, out)
# ---------------------------------------------------------------------------
def grouped_linear(x: torch.Tensor, weight_eoi: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Per-expert linear over a token batch already sorted by expert id.

    Equivalent to concatenating ``x[start_g:end_g] @ weight_eoi[g].T`` over expert
    groups g, where group boundaries are given by the cumulative `offsets`.

    Args:
        x:          (S, in) — tokens sorted by expert.
        weight_eoi: (E, out, in) — stacked expert weights (F.linear layout).
        offsets:    (E,) int32 — cumulative token counts per expert (== torch._grouped_mm offs).
    Returns:
        (S, out)
    """
    if _FORCE_FALLBACK or x.device.type == "cpu":
        # The autograd-registered fallback op wants weight (E, in, out); it loops with
        # torch.mm so it is correct + differentiable on CPU (torch._grouped_mm is CUDA-only).
        w = weight_eoi.transpose(-2, -1).contiguous()
        return torch.ops.transformers.grouped_mm_fallback(
            x.to(w.dtype), w, offs=offsets.to(torch.int32)
        )
    # GPU fast path: transformers' dispatcher picks torch.nn.functional.grouped_mm /
    # torch._grouped_mm (bf16) and casts the input to the weight dtype.
    return _moe._grouped_linear(x, weight_eoi, offsets, is_transposed=False)


# ---------------------------------------------------------------------------
# Attention LoRA — wraps a frozen (FP8)Linear
# ---------------------------------------------------------------------------
class LinearLoRA(nn.Module):
    """y = base(x) + scaling * B(A(x)).

    `base` is the original frozen Linear. If it is an FP8Linear (fp8 weight + block
    `weight_scale_inv`), the base path dequantizes to bf16 and runs a differentiable
    `F.linear` instead of FP8Linear's inference-only Triton forward. `lora_b` is
    zero-initialised so the adapter is a no-op at step 0 (a non-zero B corrupts the
    frozen model from the first step — the bug that produced garbage in the gemma4 run).
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 16.0,
                 lora_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.r = r

        in_features = base.in_features
        out_features = base.out_features
        # Create the adapters on the SAME device as the frozen base weight. nn.Linear defaults to
        # CPU, which is wrong whenever the base is already placed on a GPU before LoRA is applied
        # (e.g. device_map="auto" in compare_logits.py / merge_infer.py) -> "mat2 is on cpu" at the
        # first lora matmul. Under FSDP the base is on CPU at apply time, so this is still correct
        # (fully_shard moves the adapters onto the mesh device afterwards).
        dev = base.weight.device
        self.lora_a = nn.Linear(in_features, r, bias=False, dtype=lora_dtype, device=dev)
        self.lora_b = nn.Linear(r, out_features, bias=False, dtype=lora_dtype, device=dev)
        # kaiming_uniform_ needs an RNG fill, which fails on a meta tensor (low-CPU meta-init
        # path in minimax_m2.py). Skip here when meta — minimax_m2._reinit_lora_ re-inits the
        # adapters after `to_empty` materializes them on GPU. Non-meta path is unchanged.
        if not self.lora_a.weight.is_meta:
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.lora_a.weight)
                nn.init.zeros_(self.lora_b.weight)  # CRITICAL: adapter = identity at init

        # Cache FP8 metadata once (the base weight stays fp8 + frozen).
        self._fp8 = _is_fp8(base) and getattr(base, "weight_scale_inv", None) is not None
        self._block_size = getattr(base, "block_size", None)
        self._bias = getattr(base, "bias", None)

    def _base_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fp8:
            w = dequantize_fp8_blockwise(
                self.base.weight, self.base.weight_scale_inv, self._block_size, out_dtype=x.dtype
            )
            return F.linear(x, w, self._bias)
        return self.base(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._base_forward(x)
        lora = self.lora_b(self.lora_a(x.to(self.lora_a.weight.dtype)))
        return base + self.scaling * lora.to(base.dtype)


# ---------------------------------------------------------------------------
# MoE expert LoRA — folded into a fused grouped_mm experts forward
# ---------------------------------------------------------------------------
def _expert_base_weight(experts: nn.Module, which: str, out_dtype: torch.dtype) -> torch.Tensor:
    """Return the frozen expert weight (E, out, in) in `out_dtype`, dequantizing if fp8."""
    if which == "gate_up":
        w = experts.gate_up_proj
        s = getattr(experts, "gate_up_proj_scale_inv", None)
    else:
        w = experts.down_proj
        s = getattr(experts, "down_proj_scale_inv", None)
    if w.element_size() == 1 and s is not None:
        return dequantize_fp8_blockwise(w, s, experts.block_size, out_dtype=out_dtype)
    return w.to(out_dtype)


def fused_lora_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Fused grouped-MoE forward with per-expert LoRA folded into each projection.

    Bound onto an experts module (FP8Experts in production, a bf16 stand-in in tests).
    Structure mirrors transformers' `(fp8_)grouped_mm_experts_forward`: sort tokens by
    expert, grouped matmul, restore order, weighted sum. The LoRA delta is added to each
    projection BEFORE the SwiGLU gate (the gate is non-linear, so base+LoRA must be
    combined at the projection, not after the block).
    """
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)
    compute_dtype = self.gate_up_lora_a.dtype  # bf16

    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)

    # Sort token-expert pairs by expert so each expert's tokens are contiguous.
    perm = torch.argsort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)

    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    x_g = hidden_states[token_idx[perm]].to(compute_dtype)  # (S, H)

    # Cumulative tokens-per-expert (== grouped_mm offsets). histc avoids cuda-graph issues.
    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    # ---- gate_up projection: frozen base (dequant) + bf16 LoRA, both grouped ----
    gu_w = _expert_base_weight(self, "gate_up", compute_dtype)             # (E, 2I, H)
    gate_up = grouped_linear(x_g, gu_w, offsets)                          # (S, 2I)
    gu_lora = grouped_linear(x_g, self.gate_up_lora_a, offsets)           # (S, r)
    gu_lora = grouped_linear(gu_lora, self.gate_up_lora_b, offsets)       # (S, 2I)
    gate_up = gate_up + self.lora_scaling * gu_lora

    inter = self._apply_gate(gate_up)                                    # (S, I)  SwiGLU

    # ---- down projection: frozen base (dequant) + bf16 LoRA, both grouped ----
    dn_w = _expert_base_weight(self, "down", compute_dtype)               # (E, H, I)
    down = grouped_linear(inter, dn_w, offsets)                          # (S, H)
    dn_lora = grouped_linear(inter, self.down_lora_a, offsets)           # (S, r)
    dn_lora = grouped_linear(dn_lora, self.down_lora_b, offsets)         # (S, H)
    down = down + self.lora_scaling * dn_lora

    # Apply routing weights, restore original token order, sum the top-k per token.
    weighted = down * sample_weights_g.to(down.dtype).unsqueeze(-1)
    weighted = weighted[inv_perm]
    final = weighted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    return final.to(hidden_states.dtype)


def add_expert_lora(experts: nn.Module, r: int = 16, alpha: float = 16.0,
                    lora_dtype: torch.dtype = torch.bfloat16) -> int:
    """Attach per-expert LoRA params to an experts module and bind the fused forward.

    Stores adapters in F.linear/grouped layout (E, out, in):
        gate_up_lora_a (E, r, H)   gate_up_lora_b (E, 2I, r)
        down_lora_a    (E, r, I)   down_lora_b    (E, H, r)
    `*_lora_b` are zero so the adapter is a no-op at init. Returns the #trainable params.
    """
    E = experts.num_experts
    # Derive dims from the stored expert weights (works for FP8Experts and bf16 stand-ins).
    two_I, H = experts.gate_up_proj.shape[-2], experts.gate_up_proj.shape[-1]
    I = experts.down_proj.shape[-1]
    dev = experts.gate_up_proj.device

    def _param(shape, zero):
        t = torch.zeros(*shape, dtype=lora_dtype, device=dev)
        if not zero and not t.is_meta:  # (E, r, in): init the A factors (skip on meta — see _reinit_lora_)
            nn.init.kaiming_uniform_(t)
        return nn.Parameter(t, requires_grad=True)

    experts.gate_up_lora_a = _param((E, r, H), zero=False)
    experts.gate_up_lora_b = _param((E, two_I, r), zero=True)
    experts.down_lora_a = _param((E, r, I), zero=False)
    experts.down_lora_b = _param((E, H, r), zero=True)
    experts.lora_scaling = alpha / r
    experts.lora_r = r

    # Bind the fused LoRA forward to THIS instance (overrides the dispatched class forward).
    import types
    experts.forward = types.MethodType(fused_lora_experts_forward, experts)

    return sum(p.numel() for p in (
        experts.gate_up_lora_a, experts.gate_up_lora_b, experts.down_lora_a, experts.down_lora_b
    ))


# ---------------------------------------------------------------------------
# Top-level: freeze base, wrap attention q/k/v/o, add MoE expert LoRA
# ---------------------------------------------------------------------------
ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def apply_minimax_lora(
    model: nn.Module,
    attn_r: int = 16,
    attn_alpha: float = 16.0,
    moe_r: int = 16,
    moe_alpha: float = 16.0,
    include_moe: bool = True,
    lora_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Freeze every base param, wrap attention q/k/v/o with LinearLoRA, and (optionally)
    add per-expert LoRA to every MoE block. Returns a stats dict.

    The router `gate` (kept bf16, not fp8) and the layernorms stay frozen and un-adapted.
    """
    for p in model.parameters():
        p.requires_grad = False

    n_attn, attn_params = 0, 0
    # Wrap attention projections. They may be nn.Linear or FP8Linear; both expose
    # in_features/out_features/weight.
    for name, module in list(model.named_modules()):
        if not name.endswith("self_attn"):
            continue
        for proj in ATTN_TARGETS:
            child = getattr(module, proj, None)
            if child is None:
                continue
            lora = LinearLoRA(child, r=attn_r, alpha=attn_alpha, lora_dtype=lora_dtype)
            setattr(module, proj, lora)
            n_attn += 1
            attn_params += lora.lora_a.weight.numel() + lora.lora_b.weight.numel()

    n_moe, moe_params = 0, 0
    if include_moe:
        # MiniMaxM2SparseMoeBlock.experts is the (FP8)Experts module holding the 3D params.
        for name, module in list(model.named_modules()):
            if name.endswith(".experts") and hasattr(module, "gate_up_proj"):
                moe_params += add_expert_lora(module, r=moe_r, alpha=moe_alpha, lora_dtype=lora_dtype)
                n_moe += 1

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "attn_modules_wrapped": n_attn,
        "moe_blocks_adapted": n_moe,
        "attn_lora_params": attn_params,
        "moe_lora_params": moe_params,
        "trainable_params": total,
    }


def lora_state_dict(model: nn.Module) -> dict:
    """Collect just the trainable LoRA tensors (FSDP-safe full_tensor gather happens upstream)."""
    return {n: p for n, p in model.named_parameters() if p.requires_grad}
