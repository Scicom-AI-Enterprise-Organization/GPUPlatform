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


# Adapter tensor suffixes (LinearLoRA + fused-expert LoRA/DoRA) — everything else is a full weight.
_ADAPTER_SUFFIXES = (".lora_a.weight", ".lora_b.weight", ".magnitude",
                     ".gate_up_lora_a", ".gate_up_lora_b", ".down_lora_a", ".down_lora_b",
                     ".gate_up_mag", ".down_mag")


@torch.no_grad()
def merge_lora_(model, lora_path, scaling, moe_scaling=None, use_dora=False):
    """In-place fold of the trained adapters into the bf16 base, then REPLACE any full-weight
    tensors (embed_tokens/lm_head from --train_embeddings).

    LinearLoRA (attention/MLP):  LoRA W += scaling·(B@A);  DoRA W = magnitude·normalize(W+scaling·B@A).
    Fused routed experts (gemma-4-MoE `...experts.{gate_up,down}_proj`, 3D): per-expert same fold at
    moe_scaling via moe_adapter.fold_expert_adapter."""
    import moe_adapter as MA
    if moe_scaling is None:
        moe_scaling = scaling
    lora = torch.load(lora_path, map_location="cpu")
    state = dict(model.named_parameters())

    def base_name(prefix):
        # Strip FSDP / activation-checkpoint wrapper segments the trained model inserts
        # (e.g. layers.0._checkpoint_wrapped_module.self_attn.q_proj) — the clean base model
        # has no wrappers (layers.0.self_attn.q_proj).
        for w in ("._checkpoint_wrapped_module", "._fsdp_wrapped_module", "._orig_mod"):
            prefix = prefix.replace(w, "")
        return prefix

    prefixes = sorted({
        k[: -len(".lora_a.weight")] for k in lora if k.endswith(".lora_a.weight")
    })
    expert_prefixes = sorted({k[: -len(".gate_up_lora_a")] for k in lora if k.endswith(".gate_up_lora_a")})
    # Full-weight (non-adapter) keys: embed_tokens/lm_head saved whole by --train_embeddings.
    full_keys = [k for k in lora if not any(k.endswith(s) for s in _ADAPTER_SUFFIXES)]
    if not prefixes and not expert_prefixes and not full_keys:
        raise ValueError(f"No adapter or full-weight keys in {lora_path}; keys look like: {list(lora)[:4]}")

    merged, missing = 0, []
    diffs: dict[str, dict] = {}   # per-layer weight-change report (base W vs merged W+Δ)
    for p in prefixes:
        A = lora[f"{p}.lora_a.weight"].float()   # (r, in)
        B = lora[f"{p}.lora_b.weight"].float()   # (out, r)

        # The base weight lives at <prefix>.weight. LinearLoRA wraps the original Linear as
        # <prefix>.linear, so a checkpoint kept unwrapped may instead carry <prefix>.linear.weight.
        base = base_name(p)
        target = next((c for c in (f"{base}.weight", f"{base}.linear.weight") if c in state), None)
        if target is None:
            missing.append(base)
            continue
        w = state[target]
        adapted = w.float() + scaling * (B @ A)
        if use_dora:
            mag = lora[f"{p}.magnitude"].float()  # (out,)
            direction = adapted / adapted.norm(dim=1, keepdim=True).clamp_min(1e-8)
            new_w = mag.unsqueeze(1) * direction
        else:
            new_w = adapted
        d = (new_w - w.float())   # net change, for reporting
        wn = w.float().norm().item()
        dn = d.norm().item()
        diffs[base] = {"rel": (dn / wn) if wn else float("inf"), "abs": dn, "w": wn,
                       "max": d.abs().max().item()}
        print(f"[merge-diff] {base}: |Δ|/|W|={diffs[base]['rel']:.4f} "
              f"|Δ|={dn:.3e} |W|={wn:.3e} max|Δ|={diffs[base]['max']:.3e}")
        w.copy_(new_w.to(dtype=w.dtype))
        merged += 1

    print(f">> merged {merged}/{len(prefixes)} Linear adapters (scaling={scaling}, dora={use_dora})")
    if missing:
        print(f">> WARNING: no base weight found for {len(missing)} prefixes, e.g. {missing[:3]}")

    # ---- fused routed experts (gemma-4-MoE 3D gate_up_proj/down_proj) ----
    n_exp = 0
    for p in expert_prefixes:
        base = base_name(p)
        if f"{base}.gate_up_proj" not in state or f"{base}.down_proj" not in state:
            missing.append(base)
            continue
        adapters = {
            "gate_up_lora_a": lora[f"{p}.gate_up_lora_a"], "gate_up_lora_b": lora[f"{p}.gate_up_lora_b"],
            "down_lora_a": lora[f"{p}.down_lora_a"], "down_lora_b": lora[f"{p}.down_lora_b"],
        }
        if use_dora:
            adapters["gate_up_mag"] = lora[f"{p}.gate_up_mag"]
            adapters["down_mag"] = lora[f"{p}.down_mag"]
        gu = state[f"{base}.gate_up_proj"]; dn = state[f"{base}.down_proj"]
        gu_f, dn_f = MA.fold_expert_adapter(gu.float(), dn.float(), adapters, moe_scaling, use_dora)
        gu.copy_(gu_f.to(dtype=gu.dtype)); dn.copy_(dn_f.to(dtype=dn.dtype))
        n_exp += 1
    if expert_prefixes:
        print(f">> merged {n_exp}/{len(expert_prefixes)} routed-expert blocks (moe_scaling={moe_scaling})")

    # ---- full-weight replacement (embed_tokens / lm_head from --train_embeddings) ----
    def find_full_target(key):
        base = base_name(key)
        if base in state:
            return base
        stripped = base.replace(".linear.", ".")
        if stripped in state:
            return stripped
        # gemma-4 ties embed_tokens <-> lm_head (one tensor), so the checkpoint may key it
        # under either name; find whichever the clean base exposes (copy propagates the tie).
        leaf = base.rsplit(".", 1)[0].rsplit(".", 1)[-1] if base.endswith(".weight") else None
        aliases = {"embed_tokens": ("embed_tokens", "lm_head"),
                   "lm_head": ("lm_head", "embed_tokens")}.get(leaf, (leaf,) if leaf else ())
        for name in state:
            if any(name.endswith(f"{al}.weight") for al in aliases):
                return name
        return None

    n_full = 0
    for k in full_keys:
        target = find_full_target(k)
        if target is None:
            print(f">> WARNING: full-weight key {k} → no matching base param; skipped")
            continue
        w = state[target]
        src = lora[k]
        if tuple(src.shape) != tuple(w.shape):
            print(f">> WARNING: full-weight {k} shape {tuple(src.shape)} != base {target} "
                  f"{tuple(w.shape)}; skipped")
            continue
        w.copy_(src.to(device=w.device, dtype=w.dtype))
        n_full += 1
        print(f"[merge-full] {k} → {target}: replaced full weight {tuple(w.shape)}")
    if full_keys:
        print(f">> replaced {n_full}/{len(full_keys)} full weight(s) "
              f"(embed_tokens/lm_head from --train_embeddings)")
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
    # gemma-4-MoE routed experts were adapted with a separate (smaller) moe_r/moe_alpha.
    moe_scaling = float(meta.get("moe_alpha", 32)) / float(meta.get("moe_r", 16))
    use_dora = bool(meta.get("use_dora", False))

    print(f">> loading base {model_id} (bf16, sdpa, device_map=auto)")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
    )
    model.eval()

    merge_lora_(model, args.lora, scaling, moe_scaling=moe_scaling, use_dora=use_dora)

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
