"""Merge the trained LoRA adapters back into Gemma-4 and run inference.

The training script uses a *custom* LinearLoRA (not PEFT): each wrapped Linear computes

    y = W x + (alpha/r) * B (A x)

so merging is just  W <- W + (alpha/r) * (B @ A)  applied to the base weight. The checkpoint
(checkpointing/lora.pt) stores only the trainable adapter tensors, keyed

    <prefix>.lora_a.weight   shape (r, in)
    <prefix>.lora_b.weight   shape (out, r)

where <prefix> is the module path that, in the *base* model, holds the Linear at
<prefix>.weight (e.g. model.layers.0.self_attn.q_proj). We load a clean base model with a
standard attention impl (sdpa — generation uses a normal causal mask, NOT the packed
dynamic_attention), fold the deltas in, optionally save_pretrained, then generate.

    python merge_infer.py --prompt "Hello" --max-new-tokens 64
    python merge_infer.py --merged-out ./gemma4-merged   # also persist the merged model
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration


def load_meta(lora_path):
    meta_path = os.path.join(os.path.dirname(lora_path) or ".", "lora_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


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
        # Strip FSDP / activation-checkpoint wrapper segments the trained model inserts
        # (e.g. layers.0._checkpoint_wrapped_module.self_attn.q_proj) — the clean base model
        # has no wrappers (layers.0.self_attn.q_proj).
        for w in ("._checkpoint_wrapped_module", "._fsdp_wrapped_module", "._orig_mod"):
            prefix = prefix.replace(w, "")
        return prefix

    merged, missing = 0, []
    diffs: dict[str, dict] = {}   # per-layer weight-change report (base W vs merged W+Δ)
    for p in prefixes:
        A = lora[f"{p}.lora_a.weight"].float()   # (r, in)
        B = lora[f"{p}.lora_b.weight"].float()   # (out, r)
        delta = scaling * (B @ A)                # (out, in)

        # The base weight lives at <prefix>.weight. LinearLoRA wraps the original Linear as
        # <prefix>.linear, so a checkpoint kept unwrapped may instead carry <prefix>.linear.weight.
        base = base_name(p)
        target = next((c for c in (f"{base}.weight", f"{base}.linear.weight") if c in state), None)
        if target is None:
            missing.append(base)
            continue
        w = state[target]
        d = delta.to(device=w.device, dtype=torch.float32)   # the exact fold Δ, for reporting
        # How much the LoRA moved this layer: relative Frobenius change |Δ|/|W| (+ raw
        # norms + max element). Cheap (Δ is already computed); logged per layer + summarized.
        wn = w.float().norm().item()
        dn = d.norm().item()
        diffs[base] = {"rel": (dn / wn) if wn else float("inf"), "abs": dn, "w": wn,
                       "max": d.abs().max().item()}
        print(f"[merge-diff] {base}: |Δ|/|W|={diffs[base]['rel']:.4f} "
              f"|Δ|={dn:.3e} |W|={wn:.3e} max|Δ|={diffs[base]['max']:.3e}")
        w.add_(d.to(dtype=w.dtype))
        merged += 1

    print(f">> merged {merged}/{len(prefixes)} adapters (scaling={scaling})")
    if missing:
        print(f">> WARNING: no base weight found for {len(missing)} prefixes, e.g. {missing[:3]}")
    if diffs:
        import json as _json
        ranked = sorted(((v["rel"], k) for k, v in diffs.items()), reverse=True)
        mean_rel = sum(r for r, _ in ranked) / len(ranked)
        print(f">> [merge-diff] summary: {len(diffs)} layers · mean |Δ|/|W|={mean_rel:.4f} · "
              f"max={ranked[0][0]:.4f} ({ranked[0][1]}) · min={ranked[-1][0]:.4f} ({ranked[-1][1]})")
        # Structured one-liner the gateway can capture into the report (not an @@HF marker).
        print("@@MERGE_DIFF " + _json.dumps({
            "layers": len(diffs),
            "mean_rel": round(mean_rel, 6),
            "max_rel": round(ranked[0][0], 6), "max_layer": ranked[0][1],
            "min_rel": round(ranked[-1][0], 6), "min_layer": ranked[-1][1],
            "top": [{"layer": k, "rel": round(r, 6)} for r, k in ranked[:8]],
        }))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="checkpointing/lora.pt")
    ap.add_argument("--model-id", default=None, help="overrides lora_meta.json / default")
    ap.add_argument("--scaling", type=float, default=None, help="overrides lora_meta.json (alpha/r)")
    ap.add_argument("--prompt", default="Terangkan konsep AI dalam satu ayat.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--merged-out", default=None, help="dir to save_pretrained the merged model")
    ap.add_argument("--no-generate", action="store_true",
                    help="merge + save only, skip the sanity-check generation (used by the try-it serve path)")
    args = ap.parse_args()

    meta = load_meta(args.lora)
    model_id = args.model_id or meta.get("model_id", "google/gemma-4-31B-it")
    scaling = args.scaling if args.scaling is not None else meta.get("scaling")
    if scaling is None:
        raise SystemExit("scaling unknown: pass --scaling alpha/r (no lora_meta.json found)")

    print(f">> loading base {model_id} (bf16, sdpa, device_map=auto)")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
    )
    model.eval()

    merge_lora_(model, args.lora, scaling)

    if args.merged_out:
        print(f">> saving merged model to {args.merged_out}")
        model.save_pretrained(args.merged_out, safe_serialization=True)
        tok.save_pretrained(args.merged_out)
        # gemma-4 is multimodal: vLLM refuses to load the served dir without
        # processor_config.json + preprocessor_config.json, which model.save_pretrained
        # writes NEITHER. Fetch processor_config.json from the base and derive
        # preprocessor_config.json from its embedded image_processor block.
        try:
            import shutil
            from huggingface_hub import hf_hub_download
            dst = os.path.join(args.merged_out, "processor_config.json")
            if not os.path.exists(dst):
                shutil.copy(hf_hub_download(model_id, "processor_config.json"), dst)
            with open(dst) as f:
                _pj = json.load(f)
            _img = _pj.get("image_processor")
            if isinstance(_img, dict):
                with open(os.path.join(args.merged_out, "preprocessor_config.json"), "w") as f:
                    json.dump(_img, f, indent=2)
                print(">> wrote processor_config.json + preprocessor_config.json (gemma-4 multimodal)")
        except Exception as e:  # noqa: BLE001
            print(f">> WARN: could not write processor configs (vLLM multimodal load may fail): {e}")

    if args.no_generate:
        print(">> --no-generate: merge + save done, skipping inference")
        return

    # Chat-format the prompt when a template exists; otherwise feed it raw. apply_chat_template
    # returns a BatchEncoding (dict) in transformers 5.x, so pull out the input_ids tensor.
    try:
        msgs = [{"role": "user", "content": args.prompt}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        input_ids = enc["input_ids"]
    except Exception:
        input_ids = tok(args.prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    print(f">> generating ({args.max_new_tokens} new tokens)")
    out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print("\n===== PROMPT =====\n" + args.prompt)
    print("\n===== OUTPUT =====\n" + text)


if __name__ == "__main__":
    main()
