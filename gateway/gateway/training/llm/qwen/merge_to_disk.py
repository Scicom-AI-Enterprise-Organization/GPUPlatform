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
import sys

import torch

# This script runs from llm/qwen/, but `moe_adapter` (the shared fused-expert fold) lives in the
# parent llm/ dir — put it on the path so the routed-expert merge can import it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_ID = os.environ.get("MODEL_ID")
_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load_meta(lora_path: str) -> dict:
    meta_path = os.path.join(os.path.dirname(lora_path) or ".", "lora_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


# Adapter tensor suffixes (LinearLoRA + fused-expert LoRA/DoRA) — everything else in the
# checkpoint is a full weight (embed_tokens/lm_head from --train_embeddings).
_ADAPTER_SUFFIXES = (".lora_a.weight", ".lora_b.weight", ".magnitude",
                     ".gate_up_lora_a", ".gate_up_lora_b", ".down_lora_a", ".down_lora_b",
                     ".gate_up_mag", ".down_mag")


@torch.no_grad()
def fold_lora(model, lora_path: str, scaling: float, moe_scaling: float = None, use_dora: bool = False) -> int:
    """In-place fold of the trained adapters into the bf16 base.

    LinearLoRA (attention q/k/v/o + shared-expert MLP):
        LoRA:  W += scaling·(B@A)          DoRA:  W  = magnitude·normalize(W + scaling·B@A)
    Fused routed experts (`...experts.{gate_up,down}_proj`, 3D):  per-expert same fold via
    moe_adapter.fold_expert_adapter. Shared-expert LinearLoRA uses `moe_scaling` (the experts were
    wrapped with moe_r/moe_alpha), attention uses `scaling`."""
    import moe_adapter as MA
    if moe_scaling is None:
        moe_scaling = scaling
    lora = torch.load(lora_path, map_location="cpu")
    state = dict(model.named_parameters())
    prefixes = sorted({k[: -len(".lora_a.weight")] for k in lora if k.endswith(".lora_a.weight")})
    expert_prefixes = sorted({k[: -len(".gate_up_lora_a")] for k in lora if k.endswith(".gate_up_lora_a")})
    full_keys = [k for k in lora if not any(k.endswith(s) for s in _ADAPTER_SUFFIXES)]
    if not prefixes and not expert_prefixes and not full_keys:
        raise ValueError(f"no adapter or full-weight keys in {lora_path}; keys look like {list(lora)[:4]}")

    def base_name(prefix: str) -> str:
        for w in ("._checkpoint_wrapped_module", "._fsdp_wrapped_module", "._orig_mod"):
            prefix = prefix.replace(w, "")
        return prefix

    merged, missing = 0, []
    for p in prefixes:
        A = lora[f"{p}.lora_a.weight"].float()   # (r, in)
        B = lora[f"{p}.lora_b.weight"].float()   # (out, r)
        base = base_name(p)
        target = next((c for c in (f"{base}.weight", f"{base}.linear.weight") if c in state), None)
        if target is None:
            missing.append(base)
            continue
        w = state[target]
        # shared-expert MLP was adapted with moe_r/moe_alpha; attention with r/alpha.
        sc = moe_scaling if ".shared_expert." in base else scaling
        adapted = w.float() + sc * (B @ A)
        if use_dora:
            mag = lora[f"{p}.magnitude"].float()  # (out,)
            direction = adapted / adapted.norm(dim=1, keepdim=True).clamp_min(1e-8)
            adapted = mag.unsqueeze(1) * direction
            w.copy_(adapted.to(dtype=w.dtype, device=w.device))
        else:
            w.add_((sc * (B @ A)).to(dtype=w.dtype, device=w.device))
        merged += 1
    print(f">> merged {merged}/{len(prefixes)} Linear adapters (scaling={scaling}, dora={use_dora})", flush=True)
    if missing:
        print(f">> WARNING: no base weight for {len(missing)} prefixes, e.g. {missing[:3]}", flush=True)

    # ---- fused routed experts (3D gate_up_proj/down_proj) ----
    n_exp = 0
    for p in expert_prefixes:
        base = base_name(p)
        gu_key = next((c for c in (f"{base}.gate_up_proj",) if c in state), None)
        dn_key = next((c for c in (f"{base}.down_proj",) if c in state), None)
        if gu_key is None or dn_key is None:
            missing.append(base)
            continue
        adapters = {
            "gate_up_lora_a": lora[f"{p}.gate_up_lora_a"], "gate_up_lora_b": lora[f"{p}.gate_up_lora_b"],
            "down_lora_a": lora[f"{p}.down_lora_a"], "down_lora_b": lora[f"{p}.down_lora_b"],
        }
        if use_dora:
            adapters["gate_up_mag"] = lora[f"{p}.gate_up_mag"]
            adapters["down_mag"] = lora[f"{p}.down_mag"]
        gu = state[gu_key]; dn = state[dn_key]
        gu_f, dn_f = MA.fold_expert_adapter(gu.float(), dn.float(), adapters, moe_scaling, use_dora)
        gu.copy_(gu_f.to(dtype=gu.dtype, device=gu.device))
        dn.copy_(dn_f.to(dtype=dn.dtype, device=dn.device))
        n_exp += 1
    if expert_prefixes:
        print(f">> merged {n_exp}/{len(expert_prefixes)} routed-expert blocks (moe_scaling={moe_scaling})", flush=True)

    # Full-weight tensors (embed_tokens/lm_head from --train_embeddings): COPY into the base
    # (replace, not add-delta). Qwen3.6-27B is untied so both may be present; matched by name.
    n_full = 0
    for k in full_keys:
        base = base_name(k)
        target = base if base in state else (base.replace(".linear.", ".")
                                             if base.replace(".linear.", ".") in state else None)
        if target is None:
            print(f">> WARNING: full-weight key {k} → no matching base param; skipped", flush=True)
            continue
        w = state[target]
        src = lora[k]
        if tuple(src.shape) != tuple(w.shape):
            print(f">> WARNING: full-weight {k} shape {tuple(src.shape)} != base {tuple(w.shape)}; skipped", flush=True)
            continue
        w.copy_(src.to(dtype=w.dtype, device=w.device))
        n_full += 1
    if full_keys:
        print(f">> replaced {n_full}/{len(full_keys)} full weight(s) (embed/lm_head)", flush=True)
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
    # MoE routed + shared experts were adapted with moe_r/moe_alpha (a separate, smaller rank).
    moe_scaling = float(meta.get("moe_alpha", 32)) / float(meta.get("moe_r", 16))
    use_dora = bool(meta.get("use_dora", False))

    config = AutoConfig.from_pretrained(model_id)
    cls_name = (getattr(config, "architectures", None) or [None])[0]
    ModelCls = getattr(_tf, cls_name, None) if cls_name else None
    if ModelCls is None:
        from transformers import AutoModelForCausalLM as ModelCls  # type: ignore

    print(f">> loading base {model_id} ({args.dtype}) via {cls_name or 'AutoModelForCausalLM'}", flush=True)
    model = ModelCls.from_pretrained(model_id, dtype=out_dtype, low_cpu_mem_usage=True).eval()
    fold_lora(model, args.lora, float(scaling), moe_scaling=moe_scaling, use_dora=use_dora)

    print(f">> saving merged {args.dtype} model -> {args.out}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(args.out)
    print(">> done.", flush=True)


if __name__ == "__main__":
    main()
