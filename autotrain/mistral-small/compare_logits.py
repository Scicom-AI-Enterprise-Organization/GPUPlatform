"""Assert that our Mistral-Small-4 LoRA implementation gives the SAME logits as a trusted reference.

The Mistral-Small-4 analogue of minimax-m2's `compare_logits.py`. There is no custom attention
here (uniform head_dim 128 -> stock flash attention), so the thing under test is `lora.py`:

  * `LinearLoRA` wraps the frozen MLA q_a/q_b/kv_a/kv_b/o + the shared-expert MLP (all FP8Linear)
    and runs a DIFFERENTIABLE on-the-fly bf16 dequant of the PER-TENSOR FP8 weight (instead of
    transformers' inference-only FP8 kernel), plus `scaling * B(A(x))`.
  * `fused_lora_experts_forward` overrides the routed MoE experts forward with a dequant
    grouped_mm base + bf16 grouped LoRA.

Because `lora_b` (and every expert `*_lora_b`) is ZERO-initialised, the adapter contributes
nothing at init, so our model's logits should equal the frozen base's logits.

THE REFERENCE — like the minimax-m2 sibling, transformers' native FP8 *inference* path may be an
unreliable reference (per-tensor static w8a8 / the eager FP8Experts loop on this stack), so we
build an INDEPENDENT bf16 reference: a naive per-token / per-expert loop over the SAME dequantized
weights (completely separate code from `lora.py`'s fused grouped path). If `lora.py`'s fused
forward is correct, the two must agree. We also run the native FP8 forward for information.

Checks (on the real model, across a few prompts):
  (0) [informational] native FP8 forward — reported (may NaN/diverge on this stack; not asserted).
  (1) every `*_lora_b` is exactly zero at init  (the no-op-at-init invariant).
  (2) independent bf16 reference  vs  our LoRA(B=0):  next-token argmax MUST match,
      cosine > threshold (both are bf16 dequant; the only diff is fused-grouped vs naive-loop order).
  (3) wiring sanity: poke a non-zero B into the adapters and confirm the logits CHANGE
      (proves (2) matched because B=0, not because the LoRA path is silently disconnected).

GPU companion to `test_lora.py` (which proves the B=0 no-op + grads bit-exactly on CPU). Run on
the pod before any expensive training run.

    python compare_logits.py
    MISTRAL_PER_GPU_GIB=30 python compare_logits.py   # tune device_map headroom
"""
import argparse
import gc
import os

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, Mistral3ForConditionalGeneration

from lora import apply_mistral_lora, dequantize_fp8

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-Small-4-119B-2603")
ATTN_IMPL = os.environ.get("MISTRAL_ATTN_IMPL", "flash_attention_3")
DEVICE_MAP = os.environ.get("MISTRAL_DEVICE_MAP", "auto")  # spread the FP8 base across all GPUs
# Cap per-GPU weights so device_map="auto" leaves headroom for the bf16 dequant transient.
PER_GPU_GIB = int(os.environ.get("MISTRAL_PER_GPU_GIB", "30"))
RUN_NATIVE = os.environ.get("MISTRAL_RUN_NATIVE", "1") == "1"

PROMPTS = [
    "Hello world, how are you today?",
    "The capital of France is",
    "def fibonacci(n):",
]
TOPK = 5


def load_patched_config():
    """AutoConfig + the transformers 5.5.0 FP8 workaround (see mistral_small.py.load_patched_config)."""
    config = AutoConfig.from_pretrained(MODEL_ID)
    tc = config.get_text_config()
    if not hasattr(tc, "num_experts"):
        tc.num_experts = tc.num_local_experts
    return config


# --------------------------------------------------------------------------------------------
# Independent bf16 reference: a separate, naive dequant forward (NOT lora.py's fused path).
# --------------------------------------------------------------------------------------------
def _ref_dequant(weight, scale_inv, block_size):
    """Independent FP8->bf16 dequant (its own code; cross-checks lora.py's)."""
    if block_size is None:  # per-tensor (Mistral-Small-4): scalar / per-expert (.,1,1) scale
        return (weight.to(torch.float32) * scale_inv.to(torch.float32)).to(torch.bfloat16)
    bm, bn = block_size
    *lead, out_f, in_f = weight.shape
    so, si = out_f // bm, in_f // bn
    w = weight.to(torch.float32).reshape(*lead, so, bm, si, bn)
    s = scale_inv.to(torch.float32).reshape(*lead, so, 1, si, 1)
    return (w * s).reshape(*lead, out_f, in_f).to(torch.bfloat16)


def _is_fp8(mod):
    return getattr(mod, "weight", None) is not None and mod.weight.element_size() == 1 \
        and getattr(mod, "weight_scale_inv", None) is not None


def patch_independent_reference(model):
    """Replace every FP8Linear / FP8Experts forward with a naive bf16 dequant implementation."""
    import types
    n_lin = n_exp = 0
    for mod in model.modules():
        if _is_fp8(mod) and not hasattr(mod, "num_experts"):
            def lin_fwd(self, x):
                w = _ref_dequant(self.weight, self.weight_scale_inv, getattr(self, "block_size", None))
                return F.linear(x, w, getattr(self, "bias", None))
            mod.forward = types.MethodType(lin_fwd, mod)
            n_lin += 1
        elif hasattr(mod, "num_experts") and hasattr(mod, "gate_up_proj"):
            def exp_fwd(self, hidden_states, top_k_index, top_k_weights):
                # Dequant ONE expert at a time inside the loop (tiny (2I,H)/(H,I) slices) — keeps
                # the reference memory-frugal AND independent of lora.py's fused/chunked path.
                bs = getattr(self, "block_size", None)
                T, H = hidden_states.shape
                out = torch.zeros(T, H, dtype=hidden_states.dtype, device=hidden_states.device)
                for t in range(T):
                    x = hidden_states[t]
                    for j in range(top_k_index.size(1)):
                        e = int(top_k_index[t, j])
                        gu_e = _ref_dequant(self.gate_up_proj[e], self.gate_up_proj_scale_inv[e], bs)
                        dn_e = _ref_dequant(self.down_proj[e], self.down_proj_scale_inv[e], bs)
                        g = F.linear(x, gu_e)                        # (2I,)
                        inter = self._apply_gate(g.unsqueeze(0)).squeeze(0)  # (I,) SwiGLU
                        y = F.linear(inter, dn_e)                    # (H,)
                        out[t] += top_k_weights[t, j].to(y.dtype) * y
                return out.to(hidden_states.dtype)
            mod.forward = types.MethodType(exp_fwd, mod)
            n_exp += 1
    print(f">> independent reference: patched {n_lin} FP8Linear + {n_exp} FP8Experts", flush=True)
    return model


def load_base():
    kw = dict(config=load_patched_config(), dtype=torch.bfloat16, attn_implementation=ATTN_IMPL,
              device_map=DEVICE_MAP, low_cpu_mem_usage=True)
    if DEVICE_MAP == "auto" and torch.cuda.is_available():
        n = torch.cuda.device_count()
        kw["max_memory"] = {i: f"{PER_GPU_GIB}GiB" for i in range(n)}
        print(f">> max_memory cap: {PER_GPU_GIB}GiB x {n} GPUs (headroom for dequant transient)", flush=True)
    print(f">> loading {MODEL_ID} (FP8, attn={ATTN_IMPL}, device_map={DEVICE_MAP})", flush=True)
    return Mistral3ForConditionalGeneration.from_pretrained(MODEL_ID, **kw).eval()


def free(m):
    del m
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def last_token_logits(model, input_ids):
    dev = getattr(model, "device", "cuda")
    out = model(input_ids=input_ids.to(dev), use_cache=False)
    return out.logits[0, -1].float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn_r", type=int, default=16)
    ap.add_argument("--moe_r", type=int, default=16)
    ap.add_argument("--no_moe_lora", action="store_true")
    ap.add_argument("--no_shared_lora", action="store_true")
    ap.add_argument("--cos_threshold", type=float, default=0.997)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    enc = [torch.tensor(tok(p)["input_ids"], dtype=torch.long).unsqueeze(0) for p in PROMPTS]
    for p, e in zip(PROMPTS, enc):
        print(f"   prompt={p!r}  L={e.shape[-1]} tokens", flush=True)

    # ---- (0) native FP8 transformers — informational only ------------------------------------
    native = None
    if RUN_NATIVE:
        print("\n>> [0] native FP8 transformers (informational; per-tensor static w8a8)", flush=True)
        m = load_base()
        native = [last_token_logits(m, e) for e in enc]
        free(m)
        for p, ln in zip(PROMPTS, native):
            nanf = torch.isnan(ln).float().mean().item()
            print(f"   {p!r}: nan_frac={nanf:.3f} argmax={ln.argmax().item()}", flush=True)

    # ---- (C) independent bf16 reference ------------------------------------------------------
    print("\n>> [C] independent bf16 reference (naive per-expert dequant loop)", flush=True)
    m = load_base()
    patch_independent_reference(m)
    ref = [last_token_logits(m, e) for e in enc]
    free(m)

    # ---- (B) our LoRA implementation, B=0 ----------------------------------------------------
    print("\n>> [B] our LoRA(B=0) implementation (differentiable dequant base + zero adapter)", flush=True)
    m = load_base()
    stats = apply_mistral_lora(m, attn_r=args.attn_r, moe_r=args.moe_r,
                               include_moe=not args.no_moe_lora, include_shared=not args.no_shared_lora)
    print(f">> LoRA: {stats['attn_modules_wrapped']} attn + {stats['moe_blocks_adapted']} routed-MoE + "
          f"{stats['shared_modules_wrapped']} shared blocks, {stats['trainable_params']/1e6:.1f}M params", flush=True)

    nonzero_b = [n for n, p in m.named_parameters()
                 if (n.endswith("lora_b") or n.endswith("lora_b.weight")) and p.abs().sum().item() != 0.0]
    assert not nonzero_b, f"lora_b NOT zero at init (no-op broken): {nonzero_b[:3]}"
    print(">> [1] OK: all *_lora_b are exactly zero at init", flush=True)

    ours = [last_token_logits(m, e) for e in enc]

    # ---- (2) compare independent reference vs our LoRA(B=0) ----------------------------------
    print("\n================ [2] independent bf16 ref  vs  our LoRA(B=0) ================", flush=True)
    all_pass = True
    for p, lr, lo in zip(PROMPTS, ref, ours):
        am_r, am_o = lr.argmax().item(), lo.argmax().item()
        top_r, top_o = lr.topk(TOPK).indices.tolist(), lo.topk(TOPK).indices.tolist()
        cos = F.cosine_similarity(lr, lo, dim=0).item()
        max_abs = (lr - lo).abs().max().item()
        overlap = len(set(top_r) & set(top_o))
        # Headline correctness = the next-token (top-1 argmax) must match AND the logit vectors must
        # be ~equal (cosine). Near-TIED ranks (4th/5th) may swap in the top-5 at bf16 precision
        # (fused grouped_mm vs naive per-expert loop accumulate in different order) — expected noise.
        ok = (am_r == am_o) and (cos > args.cos_threshold)
        all_pass = all_pass and ok
        print(f"  {'OK ' if ok else 'FAIL'} {p!r}", flush=True)
        print(f"       argmax ref={am_r} ({tok.decode([am_r])!r})  ours={am_o} ({tok.decode([am_o])!r})  [top-1 {'match' if am_r==am_o else 'MISMATCH'}]")
        print(f"       top{TOPK} ref ={top_r}")
        print(f"       top{TOPK} ours={top_o}  ({overlap}/{TOPK} overlap)")
        print(f"       cosine={cos:.6f}  max_abs_diff={max_abs:.4f}")

    # ---- (3) wiring sanity: non-zero B must CHANGE the logits --------------------------------
    print("\n================ [3] wiring sanity: B != 0 must change logits ================", flush=True)
    with torch.no_grad():
        for n_, p_ in m.named_parameters():
            if n_.endswith("lora_b") or n_.endswith("lora_b.weight"):
                p_.normal_(mean=0.0, std=0.02)
    ours_bx = [last_token_logits(m, e) for e in enc]
    changed = True
    for p, lo, lx in zip(PROMPTS, ours, ours_bx):
        diff = (lo - lx).abs().max().item()
        did = diff > 1e-3
        changed = changed and did
        print(f"  {'OK ' if did else 'FAIL'} {p!r}  max_abs(B0,BX)={diff:.4f}", flush=True)
    free(m)

    print("\n================ summary ================", flush=True)
    print(f"[1] lora_b zero-init ................. PASS")
    print(f"[2] independent ref == LoRA(B=0) ..... {'PASS' if all_pass else 'FAIL'}")
    print(f"[3] B!=0 changes logits .............. {'PASS' if changed else 'FAIL'}")
    assert all_pass, "independent bf16 ref and LoRA(B=0) top-1/cosine diverged"
    assert changed, "non-zero B did NOT change logits — LoRA path is disconnected"
    print("\nPASS - LoRA(B=0) reproduces the independent bf16 reference (argmax+cosine); adapter wired.")


if __name__ == "__main__":
    main()
