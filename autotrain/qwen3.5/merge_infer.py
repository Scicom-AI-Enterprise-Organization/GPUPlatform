"""Merge the trained Qwen3.6 LoRA adapters back into the base model and run inference.

The trainer uses a *custom* LinearLoRA (not PEFT): each wrapped Linear computes

    y = W x + (alpha/r) * B (A x)

so merging is just  W <- W + (alpha/r) * (B @ A)  on the base weight. The checkpoint
(checkpointing-<model>/lora.pt) stores only the trainable adapter tensors, keyed

    <prefix>.lora_a.weight   shape (r, in)
    <prefix>.lora_b.weight   shape (out, r)

where <prefix> is e.g.
    model.language_model.layers.3._checkpoint_wrapped_module.self_attn.q_proj
The clean base model has the Linear at model.language_model.layers.3.self_attn.q_proj.weight
(no ._checkpoint_wrapped_module wrapper), so we strip that segment before folding.

Works for BOTH the dense (Qwen3.6-27B) and MoE (Qwen3.6-35B-A3B) checkpoints — the base class is
resolved from the config, and the GatedDeltaNet chunk kernel is patched (contiguous v) for the
prefill step of generate(), exactly as in training.

    python merge_infer.py --lora checkpointing-qwen3.6-27b/lora.pt --prompt "..."
    python merge_infer.py --lora checkpointing-qwen3.6-35b-a3b/lora.pt --no-merge   # base (A/B ref)
"""
import argparse
import json
import os

import torch
from transformers import AutoConfig, AutoTokenizer


def load_meta(lora_path):
    meta_path = os.path.join(os.path.dirname(lora_path) or ".", "lora_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def resolve_base_cls(model_id):
    """Return (ForConditionalGeneration class, GatedDeltaNet class) for dense or MoE."""
    cfg = AutoConfig.from_pretrained(model_id)
    is_moe = "Moe" in (cfg.architectures or [""])[0]
    if is_moe:
        from transformers import Qwen3_5MoeForConditionalGeneration as Base
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
        return Base, M.Qwen3_5MoeGatedDeltaNet, is_moe
    from transformers import Qwen3_5ForConditionalGeneration as Base
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    return Base, M.Qwen3_5GatedDeltaNet, is_moe


def patch_gated_delta_rule(model, gdn_cls):
    """Patch every GatedDeltaNet's chunk kernel to .contiguous() v (TileLang needs stride[-1]==1).
    Needed for the prefill step of generate() on the packed/varlen path."""
    from flash_qla import chunk_gated_delta_rule

    def _patched(q, k, v, *args, **kwargs):
        return chunk_gated_delta_rule(q, k, v.contiguous(), *args, **kwargs)

    n = 0
    for module in model.modules():
        if isinstance(module, gdn_cls):
            module.chunk_gated_delta_rule = _patched
            n += 1
    print(f">> patched {n} GatedDeltaNet chunk kernels (contiguous v)")


@torch.no_grad()
def merge_lora_(model, lora_path, scaling):
    """In-place: add scaling * (B @ A) to each base Linear that has an adapter."""
    lora = torch.load(lora_path, map_location="cpu")
    state = dict(model.named_parameters())

    prefixes = sorted({
        k[: -len(".lora_a.weight")] for k in lora if k.endswith(".lora_a.weight")
    })
    if not prefixes:
        raise ValueError(f"No '*.lora_a.weight' keys in {lora_path}; keys look like: {list(lora)[:4]}")

    def base_name(prefix):
        for w in ("._checkpoint_wrapped_module", "._fsdp_wrapped_module", "._orig_mod"):
            prefix = prefix.replace(w, "")
        return prefix

    merged, missing = 0, []
    for p in prefixes:
        A = lora[f"{p}.lora_a.weight"].float()   # (r, in)
        B = lora[f"{p}.lora_b.weight"].float()   # (out, r)
        delta = scaling * (B @ A)                # (out, in)

        base = base_name(p)
        target = next((c for c in (f"{base}.weight", f"{base}.linear.weight") if c in state), None)
        if target is None:
            missing.append(base)
            continue
        w = state[target]
        w.add_(delta.to(dtype=w.dtype, device=w.device))
        merged += 1

    print(f">> merged {merged}/{len(prefixes)} adapters (scaling={scaling})")
    if missing:
        print(f">> WARNING: no base weight found for {len(missing)} prefixes, e.g. {missing[:3]}")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="checkpointing-qwen3.6-27b/lora.pt")
    ap.add_argument("--model-id", default=None, help="overrides lora_meta.json / default")
    ap.add_argument("--scaling", type=float, default=None, help="overrides lora_meta.json (alpha/r)")
    ap.add_argument("--attn", default="sdpa", help="attn_implementation for generation "
                    "(sdpa/eager/kernels-community/flash-attn3). GatedDeltaNet layers use FlashQLA regardless.")
    ap.add_argument("--prompt", default="Explain what a large language model is, in two sentences.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--no-merge", action="store_true", help="skip the merge (generate from the base) for an A/B check")
    ap.add_argument("--merged-out", default=None, help="dir to save_pretrained the merged model")
    args = ap.parse_args()

    meta = load_meta(args.lora)
    model_id = args.model_id or meta.get("model_id", "Qwen/Qwen3.6-27B")
    scaling = args.scaling if args.scaling is not None else meta.get("scaling")
    if scaling is None and not args.no_merge:
        raise SystemExit("scaling unknown: pass --scaling alpha/r (no lora_meta.json found)")

    Base, gdn_cls, is_moe = resolve_base_cls(model_id)
    print(f">> loading base {model_id} (is_moe={is_moe}, bf16, attn={args.attn}, device_map=auto)")
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        model = Base.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation=args.attn, device_map="auto",
        )
    except Exception as e:
        print(f">> attn={args.attn} failed ({type(e).__name__}: {e}); retrying attn=eager")
        model = Base.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="eager", device_map="auto",
        )
    model.eval()
    patch_gated_delta_rule(model, gdn_cls)

    if not args.no_merge:
        merge_lora_(model, args.lora, scaling)
    else:
        print(">> --no-merge: generating from the BASE model (no adapters folded)")

    if args.merged_out:
        print(f">> saving merged model to {args.merged_out}")
        model.save_pretrained(args.merged_out, safe_serialization=True)
        tok.save_pretrained(args.merged_out)

    try:
        msgs = [{"role": "user", "content": args.prompt}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        input_ids = enc["input_ids"]
    except Exception:
        input_ids = tok(args.prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    print(f">> generating ({args.max_new_tokens} new tokens, greedy)")
    out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print("\n===== PROMPT =====\n" + args.prompt)
    print("\n===== OUTPUT =====\n" + text)


if __name__ == "__main__":
    main()
