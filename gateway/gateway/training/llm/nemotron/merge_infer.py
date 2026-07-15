"""Merge a trained Nemotron-H LoRA into the base model and (optionally) save a merged
checkpoint for vLLM serving.

The trainer uses a custom LinearLoRA (not PEFT): each wrapped Linear computes
    y = W x + (alpha/r)·B(A x)
so merging is  W <- W + (alpha/r)·(B@A)  on the base weight. The checkpoint (lora.pt) stores

    <prefix>.lora_a.weight   (r, in)
    <prefix>.lora_b.weight   (out, r)

plus, with --train_embeddings, the FULL embed_tokens/lm_head weights as non-`.lora_*` keys
(copied straight into the base, not folded). We load a clean base (sdpa attention, torch
Mamba path), fold/copy, optionally save_pretrained, then generate.

    python merge_infer.py --prompt "Hello" --max-new-tokens 64
    python merge_infer.py --merged-out ./nemotron-merged --no-generate
"""
import argparse
import json
import os

import torch
from transformers import AutoConfig, AutoTokenizer, NemotronHForCausalLM


def load_meta(lora_path):
    meta_path = os.path.join(os.path.dirname(lora_path) or ".", "lora_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def _base_name(prefix):
    for w in ("._checkpoint_wrapped_module", "._fsdp_wrapped_module", "._orig_mod"):
        prefix = prefix.replace(w, "")
    return prefix


@torch.no_grad()
def merge_lora_(model, lora_path, scaling, use_dora=False):
    """In-place: fold the adapter into each adapted Linear, then REPLACE any full-weight
    tensors (embed_tokens/lm_head from --train_embeddings). Nemotron adapts only Linears
    (attention q/k/v/o + Mamba in/out_proj); the MoE experts stay frozen (no 3D fold).

    LoRA:  W += scaling·(B@A)     DoRA:  W = magnitude·normalize(W + scaling·B@A)."""
    lora = torch.load(lora_path, map_location="cpu")
    state = dict(model.named_parameters())

    prefixes = sorted({k[: -len(".lora_a.weight")] for k in lora if k.endswith(".lora_a.weight")})
    full_keys = [k for k in lora if not (k.endswith(".lora_a.weight") or k.endswith(".lora_b.weight")
                                         or k.endswith(".magnitude"))]
    if not prefixes and not full_keys:
        raise ValueError(f"No '*.lora_a.weight' or full-weight keys in {lora_path}; keys: {list(lora)[:4]}")

    merged, missing = 0, []
    for p in prefixes:
        A = lora[f"{p}.lora_a.weight"].float()
        B = lora[f"{p}.lora_b.weight"].float()
        base = _base_name(p)
        target = next((c for c in (f"{base}.weight", f"{base}.linear.weight") if c in state), None)
        if target is None:
            missing.append(base)
            continue
        w = state[target]
        adapted = w.float() + scaling * (B @ A)
        if use_dora:
            mag = lora[f"{p}.magnitude"].float()
            direction = adapted / adapted.norm(dim=1, keepdim=True).clamp_min(1e-8)
            adapted = mag.unsqueeze(1) * direction
        w.copy_(adapted.to(dtype=w.dtype, device=w.device))
        merged += 1
    print(f">> merged {merged}/{len(prefixes)} adapters (scaling={scaling}, dora={use_dora})")
    if missing:
        print(f">> WARNING: no base weight for {len(missing)} prefixes, e.g. {missing[:3]}")

    n_full = 0
    for k in full_keys:
        base = _base_name(k)
        target = base if base in state else (base.replace(".linear.", ".")
                                             if base.replace(".linear.", ".") in state else None)
        if target is None:
            print(f">> WARNING: full-weight key {k} → no matching base param; skipped")
            continue
        w = state[target]
        src = lora[k]
        if tuple(src.shape) != tuple(w.shape):
            print(f">> WARNING: full-weight {k} shape {tuple(src.shape)} != base {tuple(w.shape)}; skipped")
            continue
        w.copy_(src.to(dtype=w.dtype, device=w.device))
        n_full += 1
    if full_keys:
        print(f">> replaced {n_full}/{len(full_keys)} full weight(s) (embed_tokens/lm_head)")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="checkpointing/lora.pt")
    ap.add_argument("--model-id", default=None, help="overrides lora_meta.json / MODEL_ID")
    ap.add_argument("--scaling", type=float, default=None, help="overrides lora_meta.json (alpha/r)")
    ap.add_argument("--prompt", default="Explain mixture-of-experts in one sentence.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--merged-out", default=None, help="dir to save_pretrained the merged model")
    ap.add_argument("--no-generate", action="store_true", help="merge + save only")
    args = ap.parse_args()

    meta = load_meta(args.lora)
    model_id = args.model_id or os.environ.get("MODEL_ID") or meta.get("model_id",
                                                                       "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    scaling = args.scaling if args.scaling is not None else meta.get("scaling")
    if scaling is None:
        r = float(meta.get("r") or meta.get("lora_r") or 32)
        alpha = float(meta.get("alpha") or meta.get("lora_alpha") or 64)
        scaling = alpha / r
    use_dora = bool(meta.get("use_dora", False))

    print(f">> loading base {model_id} (bf16, sdpa, torch Mamba path, device_map=auto)")
    config = AutoConfig.from_pretrained(model_id)
    config.use_mamba_kernels = False
    # Tokenizer is only needed to save alongside the merged model (serving) + for generation.
    # Best-effort: a merge-to-disk must not abort just because the tokenizer can't instantiate.
    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:  # noqa: BLE001
        print(f">> WARN: tokenizer load failed ({type(e).__name__}: {e}); merging weights only")
    try:
        model = NemotronHForCausalLM.from_pretrained(
            model_id, config=config, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
        )
    except (ImportError, ValueError) as e:  # accelerate missing → load without device_map
        print(f">> device_map=auto unavailable ({type(e).__name__}); loading on default device")
        model = NemotronHForCausalLM.from_pretrained(
            model_id, config=config, dtype=torch.bfloat16, attn_implementation="sdpa",
        )
    model.eval()

    merge_lora_(model, args.lora, float(scaling), use_dora=use_dora)

    if args.merged_out:
        print(f">> saving merged model to {args.merged_out}")
        model.save_pretrained(args.merged_out, safe_serialization=True)
        if tok is not None:
            tok.save_pretrained(args.merged_out)

    if args.no_generate:
        print(">> --no-generate: merge + save done")
        return
    if tok is None:
        print(">> no tokenizer available — skipping the sanity generation")
        return

    try:
        msgs = [{"role": "user", "content": args.prompt}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        input_ids = enc["input_ids"]
    except Exception:
        input_ids = tok(args.prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    print(f">> generating ({args.max_new_tokens} new tokens)")
    out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    print("\n===== OUTPUT =====\n" + tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
