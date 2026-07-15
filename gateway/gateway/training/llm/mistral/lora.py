"""Custom LoRA for Mistral-Small-4-119B (FP8 MoE + MLA attention) packed LoRA finetuning.

This is the *interesting part* of the Mistral-Small-4 run, the analogue of the
minimax-m2 sibling's `lora.py`. Like minimax-m2, Mistral-Small-4's text model has a
**uniform attention head_dim of 128** (qk_head_dim == v_head_dim == 128), so every
layer fits FlashAttention and there is **no custom attention** — the model runs with
stock `attn_implementation="flash_attention_2"/"_3"` and FlashAttention's native varlen
packing. The two things that DO differ from minimax-m2:

  1. **Attention is MLA (DeepSeek-style), not plain q/k/v/o.** The projections we adapt
     are `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj` (the
     compressed-latent query/kv up/down projections + the output projection). They are
     all `nn.Linear` -> `FP8Linear`, so `LinearLoRA` wraps each one unchanged.
  2. **FP8 is PER-TENSOR, not 128x128 block-scaled.** Mistral-Small-4's
     `quantization_config` has `weight_block_size=null` + `activation_scheme="static"`,
     so transformers loads `FP8Linear`/`FP8Experts` with a *scalar* `weight_scale_inv`
     (per-tensor; per-expert `(E,1,1)` for the 3D expert weights). `dequantize_fp8`
     below handles both per-tensor (block_size=None) and block-scaled layouts.

The MoE FP8 story is otherwise identical to minimax-m2:

  * The released weights are FP8 (`float8_e4m3fn`). transformers loads them as
    `FP8Linear` (MLA q/kv/o + shared-expert MLP) and `FP8Experts` (3D stacked routed
    expert params `gate_up_proj` (E,2I,H) + `down_proj` (E,H,I) with per-tensor fp32
    `*_scale_inv`).
  * transformers' FP8 forward kernels (Triton `w8a8` / DeepGEMM, the per-expert
    activation quant) are **inference-only: they are NOT autograd-differentiable**. For
    Mistral-Small-4 specifically, the fused grouped/batched/deepgemm experts dispatches
    even *refuse* `activation_scheme="static"` (they `raise NotImplementedError`), so
    only the eager per-expert FP8 loop runs at inference. None of that path can train.

So for TRAINING we replace the frozen base forward with a **differentiable** path
(exactly QLoRA's trick): keep the weight stored in FP8 (cheap), and inside the forward
dequantize the (per-tensor) scaled weight to bf16 on the fly and run a normal
(differentiable) matmul. The dequantized bf16 weight is transient — activation
checkpointing on each decoder layer recomputes it in the backward instead of retaining
it, so peak memory stays ~ one layer's bf16 weights, not the whole model's.

  * Attention LoRA     ->  `LinearLoRA` wraps each frozen MLA `FP8Linear`:
                           y = dequant_linear(x) + scaling * B(A(x)).
  * Shared-expert LoRA ->  `LinearLoRA` wraps the shared MLP's `gate_proj`/`up_proj`/
                           `down_proj` (also FP8Linear).
  * Routed-expert LoRA ->  per-expert low-rank adapters folded into a **fused grouped_mm**
                           experts forward: gate_up and down each compute
                           (frozen dequant grouped_mm) + (bf16 LoRA grouped_mm), with the
                           SwiGLU gate applied to the *sum* (base+LoRA) before the down
                           projection (the gate is non-linear, so base and LoRA cannot be
                           split after the block — they must be combined at each projection).

Everything here is base-agnostic (fp8 OR bf16 weights) so the grouped LoRA math is unit
tested on CPU in bf16 with no GPU / Triton — see `test_lora.py`.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# transformers ships the grouped-GEMM helpers + the autograd-registered CPU fallback op.
# Importing the module registers `torch.ops.transformers.grouped_mm_fallback`.
from transformers.integrations import moe as _moe


# Set MISTRAL_GROUPED_FALLBACK=1 to force the autograd-registered loop fallback for
# torch._grouped_mm (used by the CPU unit test; torch._grouped_mm has no CPU kernel).
_FORCE_FALLBACK = os.environ.get("MISTRAL_GROUPED_FALLBACK", "0") == "1"

# Optional fused Triton dequant fast path (single-pass fp8->bf16*scale, no fp32 transient).
# Enable with MISTRAL_DEQUANT_TRITON=1, or assign `lora._TRITON_DEQUANT` at runtime (see
# dequant_triton.py + bench_dequant.py). It only kicks in for CUDA tensors; CPU/meta stay on the
# torch path. The frozen weight needs no grad, so this is a plain (non-autograd) op — correct for
# the QLoRA case where the base is frozen and gradients flow through F.linear to x / the adapters.
_TRITON_DEQUANT = None
if os.environ.get("MISTRAL_DEQUANT_TRITON", "0") == "1":
    try:
        from dequant_triton import dequantize_fp8_triton as _TRITON_DEQUANT
    except Exception as _e:  # noqa: BLE001
        _TRITON_DEQUANT = None


# ---------------------------------------------------------------------------
# FP8 dequant (differentiable w.r.t. the matmul input; the weight is frozen)
# ---------------------------------------------------------------------------
def dequantize_fp8(
    weight: torch.Tensor, scale_inv: torch.Tensor, block_size=None, out_dtype=torch.bfloat16
) -> torch.Tensor:
    """Dequantize an FP8 weight to `out_dtype`.

    Mistral-Small-4 is **per-tensor** quantized (`weight_block_size=null`): `scale_inv`
    is a scalar for a 2D `FP8Linear` weight and `(E,1,1)` per-expert for the 3D
    `FP8Experts` weights. We also support 128x128 block-scaled layouts (`block_size` not
    None) so the same helper works for block-quantized siblings and the unit tests.

        w_deq = fp8_to_fp32(w) * scale_inv   (broadcast per-tensor, or per-block)

    Args:
        weight:    (..., out, in) float8_e4m3fn  (leading dims optional, e.g. num_experts)
        scale_inv: per-tensor -> scalar or (..., 1, 1); block -> (..., ceil(out/bm), ceil(in/bn))
        block_size: (bm, bn) for block-scaled, or None for per-tensor.
    Returns:
        (..., out, in) in `out_dtype`.
    """
    # Fused Triton fast path (CUDA only) — single pass, no fp32 transient. See dequant_triton.py.
    if _TRITON_DEQUANT is not None and weight.is_cuda:
        return _TRITON_DEQUANT(weight, scale_inv, block_size, out_dtype)

    *lead, out_f, in_f = weight.shape

    # ---- per-tensor (Mistral-Small-4): scale is a single value (or per-expert scalar) ----
    if block_size is None:
        if not lead:  # 2D (MLA / shared-expert Linear): one scalar, dequant in one shot.
            return (weight.to(torch.float32) * scale_inv.to(torch.float32)).to(out_dtype)
        # Stacked (E, out, in) routed-expert weights: chunk over experts so the fp32
        # transient is bounded to `chunk` experts (a full-model dequant of all 128 experts
        # at once would build a ~GBs fp32 tensor and OOM the compare/merge paths). Under
        # FSDP each rank holds 1/world_size of the experts, so this only matters for the
        # full-model (compare_logits / merge_infer) paths.
        E = math.prod(lead)
        w3 = weight.reshape(E, out_f, in_f)
        s3 = scale_inv.reshape(E, 1, 1)
        out = torch.empty(E, out_f, in_f, dtype=out_dtype, device=weight.device)
        chunk = max(1, int(os.environ.get("MISTRAL_DEQUANT_EXPERT_CHUNK", "16")))
        for st in range(0, E, chunk):
            en = min(st + chunk, E)
            out[st:en] = (w3[st:en].to(torch.float32) * s3[st:en].to(torch.float32)).to(out_dtype)
        return out.reshape(*lead, out_f, in_f)

    # ---- block-scaled (sibling models / tests): per-(bm,bn)-tile scale ----
    bm, bn = block_size
    if out_f % bm != 0 or in_f % bn != 0:
        raise ValueError(
            f"FP8 weight {tuple(weight.shape)} not divisible by block {block_size}; "
            "blockwise dequant assumes exact tiling."
        )
    so, si = out_f // bm, in_f // bn
    if not lead:
        w = weight.to(torch.float32).reshape(so, bm, si, bn)
        s = scale_inv.to(torch.float32).reshape(so, 1, si, 1)
        return (w * s).reshape(out_f, in_f).to(out_dtype)
    E = math.prod(lead)
    w3 = weight.reshape(E, out_f, in_f)
    s3 = scale_inv.reshape(E, so, si)
    out = torch.empty(E, out_f, in_f, dtype=out_dtype, device=weight.device)
    chunk = max(1, int(os.environ.get("MISTRAL_DEQUANT_EXPERT_CHUNK", "16")))
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
# Linear LoRA — wraps a frozen (FP8)Linear (MLA q/kv/o + shared-expert MLP)
# ---------------------------------------------------------------------------
class LinearLoRA(nn.Module):
    """LoRA:  y = base(x) + scaling * B(A(x)).
    DoRA:  y = F.linear(x, magnitude * normalize(W + scaling*B@A))   (weight-decomposed LoRA).

    `base` is the original frozen Linear. If it is an FP8Linear (fp8 weight +
    `weight_scale_inv`), the base path dequantizes to bf16 and runs a differentiable
    `F.linear` instead of FP8Linear's inference-only kernel, plus `scaling * B(A(x))`.
    `lora_b` is zero-initialised so the adapter is a no-op at step 0 (a non-zero B
    corrupts the frozen model from the first step).

    DoRA (`use_dora=True`) adds a per-output-row `magnitude` initialised to the base weight's
    row-norms; since B=0 and magnitude=‖W_row‖ at init, `magnitude*normalize(W) == W`, so DoRA
    is also the identity at step 0.
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 16.0,
                 lora_dtype: torch.dtype = torch.bfloat16, use_dora: bool = False):
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.r = r
        self.use_dora = use_dora

        in_features = base.in_features
        out_features = base.out_features

        # Cache FP8 metadata once (the base weight stays fp8 + frozen). Set BEFORE the magnitude
        # init below, which dequantizes the base via _base_weight().
        self._fp8 = _is_fp8(base) and getattr(base, "weight_scale_inv", None) is not None
        self._block_size = getattr(base, "block_size", None)  # None for Mistral-Small-4 (per-tensor)
        self._bias = getattr(base, "bias", None)

        # Create the adapters on the SAME device as the frozen base weight. nn.Linear defaults
        # to CPU, which is wrong whenever the base is already on a GPU before LoRA is applied
        # (e.g. device_map="auto" in compare_logits.py / merge_infer.py) -> "mat2 is on cpu".
        # Under FSDP the base is on CPU at apply time, so this is still correct (fully_shard
        # moves the adapters onto the mesh device afterwards).
        dev = base.weight.device
        self.lora_a = nn.Linear(in_features, r, bias=False, dtype=lora_dtype, device=dev)
        self.lora_b = nn.Linear(r, out_features, bias=False, dtype=lora_dtype, device=dev)
        # kaiming_uniform_ needs an RNG fill, which fails on a meta tensor (low-CPU meta-init
        # path in mistral_small.py). Skip here when meta — `_reinit_lora_` re-inits the adapters
        # (and magnitude) after `to_empty` materialises them on GPU. Non-meta path is unchanged.
        if not self.lora_a.weight.is_meta:
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.lora_a.weight)
                nn.init.zeros_(self.lora_b.weight)  # CRITICAL: adapter = identity at init

        if use_dora:
            if base.weight.is_meta:
                mag = torch.ones(out_features, dtype=lora_dtype, device=dev)
            else:
                with torch.no_grad():
                    mag = self._base_weight(torch.float32).norm(dim=1).to(lora_dtype)
            self.magnitude = nn.Parameter(mag, requires_grad=True)

    def _base_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """The frozen base weight (out, in) in `dtype`, dequantizing if fp8."""
        if self._fp8:
            return dequantize_fp8(
                self.base.weight, self.base.weight_scale_inv, self._block_size, out_dtype=dtype
            )
        return self.base.weight.to(dtype)

    def _base_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fp8:
            w = dequantize_fp8(
                self.base.weight, self.base.weight_scale_inv, self._block_size, out_dtype=x.dtype
            )
            return F.linear(x, w, self._bias)
        return self.base(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dora:
            w = self._base_weight(self.lora_a.weight.dtype)                     # (out, in)
            delta = self.lora_b.weight @ self.lora_a.weight                     # (out, in)
            adapted = w + self.scaling * delta
            direction = adapted / adapted.norm(dim=1, keepdim=True).clamp_min(1e-8)
            w_eff = self.magnitude.unsqueeze(1) * direction
            return F.linear(x.to(w_eff.dtype), w_eff, self._bias).to(x.dtype)
        base = self._base_forward(x)
        lora = self.lora_b(self.lora_a(x.to(self.lora_a.weight.dtype)))
        return base + self.scaling * lora.to(base.dtype)


# ---------------------------------------------------------------------------
# Routed-expert LoRA — folded into a fused grouped_mm experts forward
# ---------------------------------------------------------------------------
def _expert_base_weight(experts: nn.Module, which: str, out_dtype: torch.dtype) -> torch.Tensor:
    """Return the frozen routed-expert weight (E, out, in) in `out_dtype`, dequantizing if fp8."""
    if which == "gate_up":
        w = experts.gate_up_proj
        s = getattr(experts, "gate_up_proj_scale_inv", None)
    else:
        w = experts.down_proj
        s = getattr(experts, "down_proj_scale_inv", None)
    if w.element_size() == 1 and s is not None:
        return dequantize_fp8(w, s, getattr(experts, "block_size", None), out_dtype=out_dtype)
    return w.to(out_dtype)


def _expert_project(experts, x_g, which, compute_dtype, offsets):
    """One routed-expert projection with the adapter folded in (LoRA or DoRA).

    LoRA:  out = grouped(dequant base) + scaling * grouped(grouped(x, A), B)
    DoRA:  W_eff = magnitude * normalize(dequant base + scaling*(B@A));  out = grouped(x, W_eff)
    """
    base = _expert_base_weight(experts, which, compute_dtype)   # (E, out, in)
    a = getattr(experts, f"{which}_lora_a")                     # (E, r, in)
    b = getattr(experts, f"{which}_lora_b")                     # (E, out, r)
    if getattr(experts, "use_dora", False):
        mag = getattr(experts, f"{which}_mag")                  # (E, out)
        delta = torch.bmm(b.to(a.dtype), a)                     # (E, out, in)
        adapted = base.to(a.dtype) + experts.lora_scaling * delta
        direction = adapted / adapted.norm(dim=2, keepdim=True).clamp_min(1e-8)
        w_eff = mag.unsqueeze(-1) * direction                   # (E, out, in)
        return grouped_linear(x_g, w_eff, offsets)
    out = grouped_linear(x_g, base, offsets)                    # (S, out)
    lo = grouped_linear(x_g, a, offsets)                        # (S, r)
    lo = grouped_linear(lo, b, offsets)                         # (S, out)
    return out + experts.lora_scaling * lo


def fused_lora_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Fused grouped-MoE forward with per-expert LoRA folded into each projection.

    Bound onto a routed-experts module (FP8Experts in production, a bf16 stand-in in
    tests). Structure mirrors transformers' `(fp8_)grouped_mm_experts_forward`: sort
    tokens by expert, grouped matmul, restore order, weighted sum. The LoRA delta is
    added to each projection BEFORE the SwiGLU gate (the gate is non-linear, so base+LoRA
    must be combined at the projection, not after the block).

    NOTE: this covers ONLY the routed experts. The shared expert is a separate dense MLP
    (wrapped with `LinearLoRA`), added to the routed output by `Mistral4MoE.forward`.
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

    sample_weights_g = sample_weights[perm]
    x_g = hidden_states[token_idx[perm]].to(compute_dtype)  # (S, H)

    # Cumulative tokens-per-expert (== grouped_mm offsets). histc avoids cuda-graph issues.
    expert_ids_g = expert_ids[perm]
    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    # ---- gate_up projection (LoRA or DoRA) -> SwiGLU -> down projection ----
    gate_up = _expert_project(self, x_g, "gate_up", compute_dtype, offsets)   # (S, 2I)
    inter = self._apply_gate(gate_up)                                         # (S, I)  SwiGLU
    down = _expert_project(self, inter, "down", compute_dtype, offsets)       # (S, H)

    # Apply routing weights, restore original token order, sum the top-k per token.
    weighted = down * sample_weights_g.to(down.dtype).unsqueeze(-1)
    weighted = weighted[inv_perm]
    final = weighted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    return final.to(hidden_states.dtype)


def add_expert_lora(experts: nn.Module, r: int = 16, alpha: float = 16.0,
                    lora_dtype: torch.dtype = torch.bfloat16, use_dora: bool = False) -> int:
    """Attach per-expert LoRA/DoRA params to a routed-experts module and bind the fused forward.

    Stores adapters in F.linear/grouped layout (E, out, in):
        gate_up_lora_a (E, r, H)   gate_up_lora_b (E, 2I, r)
        down_lora_a    (E, r, I)   down_lora_b    (E, H, r)
    `*_lora_b` are zero so the adapter is a no-op at init. For DoRA add per-expert per-output-row
    magnitudes `gate_up_mag (E, 2I)` / `down_mag (E, H)` from the base row-norms. Returns #params.
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
    experts.use_dora = bool(use_dora)

    params = [experts.gate_up_lora_a, experts.gate_up_lora_b, experts.down_lora_a, experts.down_lora_b]

    if use_dora:
        def _mag(which, out_dim):
            if experts.gate_up_proj.is_meta:
                return nn.Parameter(torch.ones(E, out_dim, dtype=lora_dtype, device=dev), requires_grad=True)
            with torch.no_grad():
                w = _expert_base_weight(experts, which, torch.float32)  # (E, out, in)
                return nn.Parameter(w.norm(dim=2).to(lora_dtype), requires_grad=True)
        experts.gate_up_mag = _mag("gate_up", two_I)
        experts.down_mag = _mag("down", H)
        params += [experts.gate_up_mag, experts.down_mag]

    # Bind the fused LoRA forward to THIS instance (overrides the dispatched class forward).
    import types
    experts.forward = types.MethodType(fused_lora_experts_forward, experts)

    return sum(p.numel() for p in params)


# ---------------------------------------------------------------------------
# Top-level: freeze base, wrap MLA attention, add routed + shared expert LoRA
# ---------------------------------------------------------------------------
# Mistral-Small-4's attention is MLA (DeepSeek-style): compressed-latent query/kv
# projections instead of plain q/k/v. These five Linears are the analogue of q/k/v/o.
ATTN_TARGETS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")
SHARED_TARGETS = ("gate_proj", "up_proj", "down_proj")


def apply_mistral_lora(
    model: nn.Module,
    attn_r: int = 16,
    attn_alpha: float = 16.0,
    moe_r: int = 16,
    moe_alpha: float = 16.0,
    include_moe: bool = True,
    include_shared: bool = True,
    lora_dtype: torch.dtype = torch.bfloat16,
    use_dora: bool = False,
) -> dict:
    """Freeze every base param, wrap MLA attention projections with LinearLoRA, and
    (optionally) add per-expert LoRA to every routed-experts block + LinearLoRA to the
    shared-expert MLP. Returns a stats dict.

    The router `gate` (kept bf16, not fp8), the q/kv layernorms, the embeddings, the
    vision tower / projector, and the lm_head stay frozen and un-adapted.
    """
    for p in model.parameters():
        p.requires_grad = False

    # ---- attention: MLA q_a/q_b/kv_a/kv_b/o (FP8Linear or nn.Linear) ----
    n_attn, attn_params = 0, 0
    for name, module in list(model.named_modules()):
        if not name.endswith("self_attn"):
            continue
        for proj in ATTN_TARGETS:
            child = getattr(module, proj, None)
            if child is None or not isinstance(child, nn.Linear):
                continue
            lora = LinearLoRA(child, r=attn_r, alpha=attn_alpha, lora_dtype=lora_dtype, use_dora=use_dora)
            setattr(module, proj, lora)
            n_attn += 1
            attn_params += sum(p.numel() for p in lora.parameters() if p.requires_grad)

    n_moe, moe_params = 0, 0
    n_shared, shared_params = 0, 0
    if include_moe:
        # Routed experts: the (FP8)Experts module holding the 3D params (fused LoRA/DoRA).
        for name, module in list(model.named_modules()):
            if name.endswith(".experts") and hasattr(module, "gate_up_proj"):
                moe_params += add_expert_lora(module, r=moe_r, alpha=moe_alpha,
                                              lora_dtype=lora_dtype, use_dora=use_dora)
                n_moe += 1
        # Shared expert: a dense SwiGLU MLP (gate/up/down FP8Linear) -> LinearLoRA/DoRA each.
        if include_shared:
            for name, module in list(model.named_modules()):
                if not name.endswith(".shared_experts"):
                    continue
                for proj in SHARED_TARGETS:
                    child = getattr(module, proj, None)
                    if child is None or not isinstance(child, nn.Linear):
                        continue
                    lora = LinearLoRA(child, r=moe_r, alpha=moe_alpha, lora_dtype=lora_dtype, use_dora=use_dora)
                    setattr(module, proj, lora)
                    n_shared += 1
                    shared_params += sum(p.numel() for p in lora.parameters() if p.requires_grad)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "attn_modules_wrapped": n_attn,
        "moe_blocks_adapted": n_moe,
        "shared_modules_wrapped": n_shared,
        "attn_lora_params": attn_params,
        "moe_lora_params": moe_params,
        "shared_lora_params": shared_params,
        "trainable_params": total,
        "use_dora": bool(use_dora),
    }


def lora_state_dict(model: nn.Module) -> dict:
    """Collect just the trainable LoRA tensors (FSDP-safe full_tensor gather happens upstream)."""
    return {n: p for n, p in model.named_parameters() if p.requires_grad}
