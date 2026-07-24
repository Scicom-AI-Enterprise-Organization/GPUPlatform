"""Shared bf16 fused-MoE expert adapter (LoRA + DoRA) for the qwen / gemma trainers.

The Qwen3.5-MoE (`Qwen3_5MoeExperts`) and Gemma-4-MoE (`Gemma4TextExperts`) expert modules
store their experts as 3D `nn.Parameter`s — `gate_up_proj (E, 2I, H)` + `down_proj (E, H, I)` —
and expose the *identical* `forward(hidden_states, top_k_index, top_k_weights)` doing
`gate, up = gate_up.chunk(2); act_fn(gate)*up; down`. That is byte-for-byte the shape the
MiniMax-M2 trainer already adapts with a fused `grouped_mm` LoRA forward (`minimax/lora.py`).

The MiniMax path is entangled with FP8 dequant; qwen/gemma bases are plain bf16, so this module
is the FP8-free distillation of that same machinery, shared by both bf16 MoE trainers:

  * `add_expert_adapter(experts, r, alpha, use_dora)` — attaches per-expert adapter params to an
    experts module and rebinds its `forward` to the fused grouped version below.
  * `fused_adapter_experts_forward` — sort tokens by expert, grouped matmul, restore order,
    weighted top-k sum. Two modes per projection:
      - **LoRA** : W_eff = base + scaling·(B@A) per expert, then ONE grouped matmul (folded delta —
                   avoids a rank-r grouped_mm operand that trips the kernel's 16-byte stride check)
      - **DoRA** : compose W_eff = magnitude · normalize(base + scaling·(B@A)) per expert per
                   output row, then ONE grouped matmul with the composed weight (the direction
                   normalization is non-linear, so base+delta cannot be split after the matmul).

Both are the identity at init (`*_lora_b = 0`, and for DoRA `magnitude = ‖base row‖`), so step 0
leaves the frozen model untouched. Everything is base-agnostic and unit-tested on CPU in fp32
(no GPU / grouped-mm kernel) — run `python moe_adapter.py`.
"""
from __future__ import annotations

import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

# Force the CPU loop fallback for the grouped matmul (torch._grouped_mm is CUDA-only). Set by the
# CPU unit test; on a GPU box the fast transformers dispatcher path is used instead.
_FORCE_FALLBACK = os.environ.get("MOE_ADAPTER_GROUPED_FALLBACK", "0") == "1"


# ---------------------------------------------------------------------------
# MoE expert rank derivation (Thinking Machines LoRA recipe)
# ---------------------------------------------------------------------------
# HF config attrs that name the MoE top-k (active experts per token), most-specific first.
_TOPK_ATTRS = ("num_experts_per_tok", "top_k_experts", "num_experts_per_token", "moe_topk", "moe_top_k")


def num_active_experts(config) -> int | None:
    """Best-effort read of the MoE top-k (active experts/token) from a HF config, checking the
    text sub-config too (multimodal models like gemma-4). Returns None if not found."""
    for obj in (config, getattr(config, "text_config", None), getattr(config, "get_text_config", lambda: None)()):
        if obj is None:
            continue
        for a in _TOPK_ATTRS:
            v = getattr(obj, a, None)
            if isinstance(v, int) and v > 0:
                return v
    return None


def derive_moe_rank(attn_r: int, attn_alpha: float, top_k: int) -> tuple[int, float]:
    """MoE expert rank = attention rank / active-experts (top-k), with the scaling (alpha/r) held
    constant. https://thinkingmachines.ai/blog/lora/ — a token routes to only top-k of the experts,
    so dividing the rank by top-k matches the dense per-token update magnitude. Returns (moe_r,
    moe_alpha); moe_alpha = moe_r · (attn_alpha/attn_r) keeps the same scaling as the dense LoRA."""
    k = max(1, int(top_k))
    moe_r = max(1, round(attn_r / k))
    scaling = (attn_alpha / attn_r) if attn_r else 2.0
    return moe_r, moe_r * scaling


# ---------------------------------------------------------------------------
# grouped matmul: x (S, in) sorted-by-expert @ weight (E, out, in).T -> (S, out)
# ---------------------------------------------------------------------------
def grouped_linear(x: torch.Tensor, weight_eoi: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Per-expert linear over a token batch already sorted by expert id.

    Equivalent to concatenating ``x[start_g:end_g] @ weight_eoi[g].T`` over expert groups g,
    where group boundaries are the cumulative `offsets` (== ``torch._grouped_mm`` offs).

    Args:
        x:          (S, in) — tokens sorted by expert.
        weight_eoi: (E, out, in) — stacked expert weights (F.linear layout).
        offsets:    (E,) int32 — cumulative token counts per expert.
    """
    if _FORCE_FALLBACK or x.device.type == "cpu":
        # Loop fallback (correct + differentiable; torch._grouped_mm has no CPU kernel).
        outs = []
        start = 0
        for e in range(weight_eoi.shape[0]):
            end = int(offsets[e])
            outs.append(x[start:end] @ weight_eoi[e].to(x.dtype).transpose(-2, -1))
            start = end
        return torch.cat(outs, dim=0)
    # GPU fast path: transformers' dispatcher picks torch._grouped_mm (bf16) + casts input to
    # the weight dtype. Same call the MiniMax trainer uses in production.
    from transformers.integrations import moe as _moe  # registers the grouped-mm helpers
    return _moe._grouped_linear(x, weight_eoi, offsets, is_transposed=False)


# ---------------------------------------------------------------------------
# per-projection adapter apply (LoRA or DoRA)
# ---------------------------------------------------------------------------
def _adapter_project(
    experts: nn.Module,
    x_g: torch.Tensor,
    base_weight: torch.Tensor,  # (E, out, in) F.linear layout
    which: str,                 # "gate_up" | "down"
    offsets: torch.Tensor,
) -> torch.Tensor:
    a = getattr(experts, f"{which}_lora_a")   # (E, r, in)
    b = getattr(experts, f"{which}_lora_b")   # (E, out, r)
    if getattr(experts, "use_dora", False):
        mag = getattr(experts, f"{which}_mag")            # (E, out)
        delta = torch.bmm(b.to(a.dtype), a)               # (E, out, in)
        adapted = base_weight.to(a.dtype) + experts.lora_scaling * delta
        direction = adapted / adapted.norm(dim=2, keepdim=True).clamp_min(1e-8)
        w_eff = mag.unsqueeze(-1) * direction             # (E, out, in)
        return grouped_linear(x_g, w_eff, offsets)
    # LoRA: fold the low-rank delta INTO the base weight, then ONE grouped matmul (same
    # single-matmul structure as the DoRA branch and the `grouped_linear(x_g, base_weight)`
    # path above). The old two-step form (grouped(x, A) -> grouped(_, B)) fed the rank-r
    # intermediate straight into torch._grouped_mm, whose Hopper kernel rejects a rank-32
    # operand with "strides should be multiple of 16 bytes" (r64/128 slipped through; r32 did
    # not). Folding keeps every matmul full-width (H / 2I / out), so every rank works — and it's
    # strictly cheaper than the DoRA path (no norm/magnitude), which already runs fine at r32,
    # so memory/throughput are non-issues.
    delta = torch.bmm(b.to(a.dtype), a)                   # (E, out, in) = B·A per expert
    w_eff = base_weight.to(a.dtype) + experts.lora_scaling * delta
    return grouped_linear(x_g, w_eff, offsets)            # (S, out)


# ---------------------------------------------------------------------------
# fused grouped-MoE forward with per-expert adapter folded in
# ---------------------------------------------------------------------------
def fused_adapter_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Bound onto a `*Experts` module. Mirrors the stock experts forward
    (sort tokens by expert, grouped matmul, weighted top-k sum) with the adapter delta folded
    into each projection BEFORE the SwiGLU gate (the gate is non-linear)."""
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)
    compute_dtype = self.gate_up_lora_a.dtype  # bf16 (fp32 in the CPU test)

    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)

    # Sort token-expert pairs by expert so each expert's tokens are contiguous.
    perm = torch.argsort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)

    sample_weights_g = sample_weights[perm]
    expert_ids_g = expert_ids[perm]
    x_g = hidden_states[token_idx[perm]].to(compute_dtype)  # (S, H)

    # Cumulative tokens-per-expert (== grouped_mm offsets). histc avoids cuda-graph issues.
    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    # ---- gate_up projection -> SwiGLU -> down projection ----
    gate_up = _adapter_project(self, x_g, self.gate_up_proj, "gate_up", offsets)  # (S, 2I)
    gate, up = gate_up.chunk(2, dim=-1)
    inter = self._moe_act_fn(gate) * up                                           # (S, I)
    down = _adapter_project(self, inter, self.down_proj, "down", offsets)         # (S, H)

    # Apply routing weights, restore original token order, sum the top-k per token.
    weighted = down * sample_weights_g.to(down.dtype).unsqueeze(-1)
    weighted = weighted[inv_perm]
    final = weighted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    return final.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# attach adapters + rebind forward
# ---------------------------------------------------------------------------
def add_expert_adapter(
    experts: nn.Module,
    r: int = 16,
    alpha: float = 16.0,
    use_dora: bool = False,
    lora_dtype: torch.dtype = torch.bfloat16,
    act_fn=None,
) -> int:
    """Attach per-expert LoRA/DoRA params to a fused-experts module and bind the fused forward.

    Adapters in F.linear/grouped layout (E, out, in):
        gate_up_lora_a (E, r, H)   gate_up_lora_b (E, 2I, r)
        down_lora_a    (E, r, I)   down_lora_b    (E, H, r)
    `*_lora_b` are zero so the adapter is a no-op at init. For DoRA add per-expert per-output-row
    magnitudes (E, 2I) / (E, H) initialised from the base weight row-norms — so W_eff == base at
    init too. Returns the number of trainable params added.

    `act_fn` is the SwiGLU activation. Pass it explicitly when the experts module may not carry
    `.act_fn` — e.g. a Liger-patched `LigerExperts` (qwen's `apply_liger_kernel_to_qwen3_5_moe`
    replaces the module and drops `act_fn`). Falls back to `experts.act_fn` when present (gemma).
    """
    # Resolve the activation BEFORE we override forward. Fallback to the module's own act_fn
    # (present on stock Gemma4TextExperts / Qwen3_5MoeExperts, absent on LigerExperts).
    experts._moe_act_fn = act_fn if act_fn is not None else getattr(experts, "act_fn", None)
    if experts._moe_act_fn is None:
        raise ValueError(
            "add_expert_adapter: no SwiGLU activation available — the experts module has no "
            "`.act_fn` (e.g. a Liger-patched LigerExperts); pass act_fn=ACT2FN[config.hidden_act]."
        )
    E = experts.num_experts
    two_I, H = experts.gate_up_proj.shape[-2], experts.gate_up_proj.shape[-1]
    Hd, I = experts.down_proj.shape[-2], experts.down_proj.shape[-1]
    dev = experts.gate_up_proj.device

    def _param(shape, zero):
        t = torch.zeros(*shape, dtype=lora_dtype, device=dev)
        # (E, r, in): kaiming-init the A factors (skip on meta — no RNG fill there).
        if not zero and not t.is_meta:
            nn.init.kaiming_uniform_(t)
        return nn.Parameter(t, requires_grad=True)

    experts.gate_up_lora_a = _param((E, r, H), zero=False)
    experts.gate_up_lora_b = _param((E, two_I, r), zero=True)
    experts.down_lora_a = _param((E, r, I), zero=False)
    experts.down_lora_b = _param((E, Hd, r), zero=True)
    experts.lora_scaling = alpha / r
    experts.lora_r = r
    experts.use_dora = bool(use_dora)

    n = sum(p.numel() for p in (
        experts.gate_up_lora_a, experts.gate_up_lora_b, experts.down_lora_a, experts.down_lora_b
    ))

    if use_dora:
        experts.gate_up_mag = _init_magnitude(experts.gate_up_proj, lora_dtype)  # (E, 2I)
        experts.down_mag = _init_magnitude(experts.down_proj, lora_dtype)        # (E, H)
        n += experts.gate_up_mag.numel() + experts.down_mag.numel()

    experts.forward = types.MethodType(fused_adapter_experts_forward, experts)
    return n


def _init_magnitude(weight_eoi: torch.Tensor, dtype: torch.dtype) -> nn.Parameter:
    """Per-expert per-output-row L2 norm (over the input dim) of a (E, out, in) base weight.

    On a meta tensor the norm can't be computed — fall back to ones; the trainer re-initialises
    from the materialised weight (`reinit_expert_magnitudes`) after the base is loaded.
    """
    if weight_eoi.is_meta:
        E, out, _ = weight_eoi.shape
        return nn.Parameter(torch.ones(E, out, dtype=dtype, device=weight_eoi.device), requires_grad=True)
    with torch.no_grad():
        norms = weight_eoi.detach().to(torch.float32).norm(dim=2)  # (E, out)
    return nn.Parameter(norms.to(dtype), requires_grad=True)


@torch.no_grad()
def reinit_expert_magnitudes(experts: nn.Module) -> None:
    """Re-init DoRA magnitudes from the (now materialised) base weights. For trainers that build
    the model on `meta` and load the base afterwards; a no-op for LoRA / real-weight applies."""
    if not getattr(experts, "use_dora", False):
        return
    experts.gate_up_mag.copy_(experts.gate_up_proj.detach().to(torch.float32).norm(dim=2).to(experts.gate_up_mag.dtype))
    experts.down_mag.copy_(experts.down_proj.detach().to(torch.float32).norm(dim=2).to(experts.down_mag.dtype))


# ---------------------------------------------------------------------------
# merge fold (used by the qwen/gemma merge scripts) — compose the adapter into the base weight
# ---------------------------------------------------------------------------
@torch.no_grad()
def fold_expert_adapter(
    gate_up: torch.Tensor,   # (E, 2I, H) fp32 base, modified in place / returned
    down: torch.Tensor,      # (E, H, I)  fp32 base
    adapters: dict,          # {"gate_up_lora_a": ..., "gate_up_lora_b": ..., "down_lora_a": ...,
                             #  "down_lora_b": ..., optionally "gate_up_mag"/"down_mag"}
    scaling: float,
    use_dora: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold LoRA/DoRA expert adapters into fp32 base weights (for serving a merged bf16 model).

    LoRA:  W += scaling·(B[e] @ A[e]) per expert.
    DoRA:  W  = magnitude · normalize(W + scaling·(B[e] @ A[e])) per expert per output row.
    """
    def _fold(base, which):
        # Align adapters to the base weight's device — the base may be on cuda (a device_map="auto"
        # merge, e.g. gemma) while the adapters come from torch.load(map_location="cpu").
        dev = base.device
        a = adapters[f"{which}_lora_a"].to(device=dev, dtype=torch.float32)   # (E, r, in)
        b = adapters[f"{which}_lora_b"].to(device=dev, dtype=torch.float32)   # (E, out, r)
        delta = scaling * torch.bmm(b, a)                   # (E, out, in)
        if use_dora:
            adapted = base + delta
            direction = adapted / adapted.norm(dim=2, keepdim=True).clamp_min(1e-8)
            mag = adapters[f"{which}_mag"].to(device=dev, dtype=torch.float32)  # (E, out)
            return mag.unsqueeze(-1) * direction
        return base + delta

    return _fold(gate_up, "gate_up"), _fold(down, "down")


# ---------------------------------------------------------------------------
# CPU unit test (fp32, no GPU / grouped-mm kernel): fused == dense reference
# ---------------------------------------------------------------------------
def _reference_experts_forward(experts, hidden_states, top_k_index, top_k_weights, gate_up_eff, down_eff):
    """Stock per-expert loop (as in Qwen3_5MoeExperts.forward) but with pre-composed effective
    weights — the ground truth the fused grouped path must match."""
    final = torch.zeros_like(hidden_states)
    E = experts.num_experts
    for e in range(E):
        # tokens routed to expert e (any of the top-k slots)
        pos = (top_k_index == e).nonzero(as_tuple=False)  # (n, 2) -> (token, slot)
        if pos.numel() == 0:
            continue
        tok = pos[:, 0]
        slot = pos[:, 1]
        x = hidden_states[tok]
        gate, up = (x @ gate_up_eff[e].T).chunk(2, dim=-1)
        h = experts.act_fn(gate) * up
        out = h @ down_eff[e].T
        out = out * top_k_weights[tok, slot, None]
        final.index_add_(0, tok, out.to(final.dtype))
    return final


def _make_fake_experts(E, H, I, act="silu"):
    from transformers.activations import ACT2FN
    m = nn.Module()
    m.num_experts = E
    m.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, H, dtype=torch.float32) * 0.02)
    m.down_proj = nn.Parameter(torch.randn(E, H, I, dtype=torch.float32) * 0.02)
    m.act_fn = ACT2FN[act]
    return m


def _test():
    os.environ["MOE_ADAPTER_GROUPED_FALLBACK"] = "1"
    global _FORCE_FALLBACK
    _FORCE_FALLBACK = True
    torch.manual_seed(0)

    E, H, I, T, K = 8, 64, 32, 40, 2
    for use_dora in (False, True):
        experts = _make_fake_experts(E, H, I)
        base_gu = experts.gate_up_proj.detach().clone()
        base_dn = experts.down_proj.detach().clone()

        n = add_expert_adapter(experts, r=8, alpha=16.0, use_dora=use_dora, lora_dtype=torch.float32)

        hidden = torch.randn(T, H, dtype=torch.float32)
        logits = torch.randn(T, E)
        w, idx = torch.topk(torch.softmax(logits, dim=-1), K, dim=-1)
        w = w / w.sum(-1, keepdim=True)

        # ---- (1) identity at init: B=0 (+ mag=‖W‖) => fused == stock base experts forward ----
        fused0 = experts(hidden, idx, w)
        ref_base = _reference_experts_forward(experts, hidden, idx, w, base_gu, base_dn)
        err0 = (fused0 - ref_base).abs().max().item()
        assert err0 < 1e-4, f"[use_dora={use_dora}] init identity broken: max|Δ|={err0}"

        # ---- (2) with a non-zero B, fused grouped == dense composed reference ----
        with torch.no_grad():
            experts.gate_up_lora_b.normal_(0, 0.02)
            experts.down_lora_b.normal_(0, 0.02)
        s = experts.lora_scaling
        if use_dora:
            gu_adapt = base_gu + s * torch.bmm(experts.gate_up_lora_b, experts.gate_up_lora_a)
            dn_adapt = base_dn + s * torch.bmm(experts.down_lora_b, experts.down_lora_a)
            gu_eff = experts.gate_up_mag.unsqueeze(-1) * (gu_adapt / gu_adapt.norm(dim=2, keepdim=True))
            dn_eff = experts.down_mag.unsqueeze(-1) * (dn_adapt / dn_adapt.norm(dim=2, keepdim=True))
        else:
            gu_eff = base_gu + s * torch.bmm(experts.gate_up_lora_b, experts.gate_up_lora_a)
            dn_eff = base_dn + s * torch.bmm(experts.down_lora_b, experts.down_lora_a)

        fused = experts(hidden, idx, w)
        ref = _reference_experts_forward(experts, hidden, idx, w, gu_eff, dn_eff)
        err = (fused - ref).abs().max().item()
        assert err < 1e-4, f"[use_dora={use_dora}] fused != reference: max|Δ|={err}"

        # ---- (3) merge fold matches the composed effective weight ----
        adapters = {
            "gate_up_lora_a": experts.gate_up_lora_a, "gate_up_lora_b": experts.gate_up_lora_b,
            "down_lora_a": experts.down_lora_a, "down_lora_b": experts.down_lora_b,
        }
        if use_dora:
            adapters["gate_up_mag"] = experts.gate_up_mag
            adapters["down_mag"] = experts.down_mag
        gu_fold, dn_fold = fold_expert_adapter(base_gu.float(), base_dn.float(), adapters, s, use_dora)
        assert (gu_fold - gu_eff).abs().max().item() < 1e-4, f"[use_dora={use_dora}] gate_up fold mismatch"
        assert (dn_fold - dn_eff).abs().max().item() < 1e-4, f"[use_dora={use_dora}] down fold mismatch"

        print(f"[moe_adapter] use_dora={use_dora}: OK  (added {n} params, "
              f"init_err={err0:.2e}, fused_err={err:.2e})")

    # ---- (4) MoE rank derivation: rank/top_k, scaling held constant ----
    import types as _t
    assert derive_moe_rank(256, 512, 8) == (32, 64.0), derive_moe_rank(256, 512, 8)   # scaling 2.0
    assert derive_moe_rank(256, 512, 4) == (64, 128.0)
    r16, a16 = derive_moe_rank(16, 16.0, 8)   # scaling 1.0
    assert (r16, a16) == (2, 2.0), (r16, a16)
    assert derive_moe_rank(256, 512, 1) == (256, 512.0)  # dense-equivalent when top_k=1
    # num_active_experts: direct attr, text_config nesting, and absence.
    assert num_active_experts(_t.SimpleNamespace(num_experts_per_tok=8)) == 8
    assert num_active_experts(_t.SimpleNamespace(text_config=_t.SimpleNamespace(top_k_experts=4))) == 4
    assert num_active_experts(_t.SimpleNamespace(hidden_size=1)) is None
    print("[moe_adapter] rank derivation: OK (256/8→32 α64, 16/8→2, top_k lookup incl. text_config)")

    # ---- (5) act_fn: explicit override works; missing .act_fn + no override raises (LigerExperts) ----
    import torch.nn as _nn
    ex = _make_fake_experts(4, 32, 16)
    del ex.act_fn  # simulate a Liger-patched LigerExperts (no .act_fn attribute)
    try:
        add_expert_adapter(ex, r=4, alpha=8.0, lora_dtype=torch.float32)
        raise SystemExit("[moe_adapter] FAIL: expected ValueError when act_fn missing")
    except ValueError:
        pass
    ex2 = _make_fake_experts(4, 32, 16)
    del ex2.act_fn
    add_expert_adapter(ex2, r=4, alpha=8.0, lora_dtype=torch.float32, act_fn=_nn.SiLU())  # explicit
    _h = torch.randn(6, 32); _w, _i = torch.topk(torch.softmax(torch.randn(6, 4), -1), 2, -1)
    ex2(_h, _i, _w / _w.sum(-1, keepdim=True))  # forward must run using the passed act_fn
    print("[moe_adapter] act_fn: OK (explicit override runs; missing+no-override raises)")

    print("[moe_adapter] all CPU tests passed")


if __name__ == "__main__":
    _test()
