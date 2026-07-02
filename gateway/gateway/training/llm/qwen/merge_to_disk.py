"""Fold a trained Qwen3.6 (dense 27B / MoE 35B-A3B, bf16) LoRA into the base and save a merged HF
checkpoint for vLLM serving.

Qwen uses the same simple LinearLoRA (`<prefix>.lora_a.weight` (r,in) + `.lora_b.weight` (out,r)) as
gemma, and its base is **bf16** (not FP8), so folding is just  W <- W + scaling·(B@A)  — no dequant.
We load the model class the config declares (`Qwen3_5ForConditionalGeneration` /
`Qwen3_5MoeForConditionalGeneration`, present in the qwen training venv's transformers), fold, and
`save_pretrained` at the requested dtype. The result is a standard HF checkpoint vLLM loads.

    python merge_to_disk.py --lora checkpointing/lora.pt --out /share/merged-qwen --dtype fp16
"""
import argparse
import json
import os

import torch

MODEL_ID = os.environ.get("MODEL_ID")
_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load_meta(lora_path: str) -> dict:
    meta_path = os.path.join(os.path.dirname(lora_path) or ".", "lora_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


@torch.no_grad()
def fold_lora(model, lora_path: str, scaling: float) -> int:
    """In-place: add scaling·(B@A) to each base Linear that has an adapter. Same fold as gemma's
    merge_infer.merge_lora_ (wrapper-strip + <prefix>.weight or <prefix>.linear.weight)."""
    lora = torch.load(lora_path, map_location="cpu")
    state = dict(model.named_parameters())
    prefixes = sorted({k[: -len(".lora_a.weight")] for k in lora if k.endswith(".lora_a.weight")})
    if not prefixes:
        raise ValueError(f"no '*.lora_a.weight' keys in {lora_path}; keys look like {list(lora)[:4]}")

    def base_name(prefix: str) -> str:
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
    print(f">> merged {merged}/{len(prefixes)} adapters (scaling={scaling})", flush=True)
    if missing:
        print(f">> WARNING: no base weight for {len(missing)} prefixes, e.g. {missing[:3]}", flush=True)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="checkpointing/lora.pt")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--out", default="/share/merged-qwen")
    ap.add_argument("--dtype", default="fp16", choices=list(_DTYPES), help="merged model dtype (default fp16)")
    ap.add_argument("--scaling", type=float, default=None, help="override lora_meta.json (alpha/r)")
    args = ap.parse_args()

    out_dtype = _DTYPES[args.dtype]
    from transformers import AutoConfig, AutoTokenizer
    import transformers as _tf

    meta = load_meta(args.lora)
    model_id = args.model_id or MODEL_ID or meta.get("model_id")
    if not model_id:
        raise SystemExit("no model id (pass --model-id or set MODEL_ID, or ensure lora_meta.json has model_id)")

    scaling = args.scaling if args.scaling is not None else meta.get("scaling")
    if scaling is None:
        r = float(meta.get("lora_r") or meta.get("r") or 16)
        alpha = float(meta.get("lora_alpha") or meta.get("alpha") or 32)
        scaling = alpha / r

    config = AutoConfig.from_pretrained(model_id)
    cls_name = (getattr(config, "architectures", None) or [None])[0]
    ModelCls = getattr(_tf, cls_name, None) if cls_name else None
    if ModelCls is None:
        from transformers import AutoModelForCausalLM as ModelCls  # type: ignore

    print(f">> loading base {model_id} ({args.dtype}) via {cls_name or 'AutoModelForCausalLM'}", flush=True)
    model = ModelCls.from_pretrained(model_id, dtype=out_dtype, low_cpu_mem_usage=True).eval()
    fold_lora(model, args.lora, float(scaling))

    print(f">> saving merged {args.dtype} model -> {args.out}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(args.out)
    print(">> done.", flush=True)


if __name__ == "__main__":
    main()
