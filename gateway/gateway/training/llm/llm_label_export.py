#!/usr/bin/env python3
"""Post-training Label-platform export for autotrain LLM (all arches).

Shipped to the run's VM as the whole `llm/` dir and run with the LLM trainer
venv (/share/autotrain-llm-{arch}/bin/python). It:

  1. downloads the LoRA checkpoint (lora.pt + lora_meta.json) from S3,
  2. loads the finetuned model for the run's arch with its LoRA applied:
       • gemma / qwen — dense: fold the LinearLoRA delta into the base weights
         (the simple `y = W x + (alpha/r) B(A x)` merge),
       • minimax / mistral — FP8 MoE: attach the adapters to the FP8 base via the
         arch's own `merge_infer.attach_lora` (no bf16 merge — the base is too big),
  3. generates one assistant response per eval row (user-message prompts),
  4. emits a structured line the gateway parses:

     @@LABEL {"items": [...], "count": N}
     @@LABEL {"error": "..."}

Config (JSON via --config):
  {
    "model_s3":         "s3://bucket/training-runs/<id>/checkpoint/",
    "base_model":       "google/gemma-4-31B-it",   # fallback if lora_meta absent
    "arch":             "gemma",
    "eval_rows":        [{"messages":[{"role":"user","content":"…"}],
                          "lov":"…", "language":"…"}, …],
    "n_samples":        110,
    "max_new_tokens":   512,
    "lora_r":           16,    # fallback when lora_meta.json is absent
    "lora_alpha":       32,    # fallback when lora_meta.json is absent
    "run_id":           "train-xxxxxxxx",
    "work_dir":         "/share",
    "hf_home":          "/share/huggingface",  # optional; reuse training cache
    "s3_creds":         {"bucket","region","endpoint","access_key","secret_key"}
  }
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

# Both merge_infer.py and this file are shipped to /tmp/sgpu_llm_label/ on the VM.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def emit(obj: dict) -> None:
    print("@@LABEL " + json.dumps(obj), flush=True)


def log(m: str) -> None:
    print(f"[llm-label-export] {m}", flush=True)


def _s3_client(creds: dict):
    import boto3
    from botocore.client import Config as BotoConfig
    return boto3.client(
        "s3",
        region_name=creds.get("region") or "us-east-1",
        endpoint_url=creds.get("endpoint") or None,
        aws_access_key_id=creds.get("access_key") or None,
        aws_secret_access_key=creds.get("secret_key") or None,
        config=BotoConfig(signature_version="s3v4"),
    )


def _download_lora(model_s3: str, creds: dict, dest_dir: str) -> str:
    """Download lora.pt (+ optional lora_meta.json) from the S3 artifact prefix.
    Returns the local path to lora.pt."""
    os.makedirs(dest_dir, exist_ok=True)
    cli = _s3_client(creds)
    m = re.match(r"s3://([^/]+)/?(.*)", model_s3.rstrip("/"))
    if not m:
        raise ValueError(f"invalid model_s3 URI: {model_s3!r}")
    bucket, prefix = m.group(1), m.group(2).strip("/")
    for fname in ("lora.pt", "lora_meta.json"):
        key = f"{prefix}/{fname}" if prefix else fname
        dest = os.path.join(dest_dir, fname)
        log(f"downloading s3://{bucket}/{key} …")
        try:
            cli.download_file(bucket, key, dest)
        except Exception as e:  # noqa: BLE001
            if fname == "lora.pt":
                raise RuntimeError(f"lora.pt not found at s3://{bucket}/{key}: {e}") from e
            log(f"note: {fname} not present (older checkpoint) — using config fallbacks")
    return os.path.join(dest_dir, "lora.pt")


def load_meta(lora_path: str) -> dict:
    meta_path = os.path.join(os.path.dirname(lora_path) or ".", "lora_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def _fold_linear_lora(model, lora_path: str, scaling: float) -> int:
    """Dense archs (gemma, qwen): fold the LinearLoRA delta scaling*(B@A) into each
    base Linear weight. The checkpoint stores <prefix>.lora_a.weight (r,in) +
    <prefix>.lora_b.weight (out,r); the base weight is <prefix>.weight (or, if the
    checkpoint kept the wrapper, <prefix>.linear.weight). Same fold as gemma's
    merge_infer.merge_lora_ — inlined so it has no arch-specific import."""
    import torch

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
    with torch.no_grad():
        for p in prefixes:
            A = lora[f"{p}.lora_a.weight"].float()   # (r, in)
            B = lora[f"{p}.lora_b.weight"].float()   # (out, r)
            delta = scaling * (B @ A)
            base = base_name(p)
            target = next((c for c in (f"{base}.weight", f"{base}.linear.weight") if c in state), None)
            if target is None:
                missing.append(base)
                continue
            w = state[target]
            w.add_(delta.to(dtype=w.dtype, device=w.device))
            merged += 1
    log(f"merged {merged}/{len(prefixes)} adapters (scaling={scaling:.4f})")
    if missing:
        log(f"WARNING: no base weight for {len(missing)} prefixes, e.g. {missing[:3]}")
    return merged


def _load_arch_module(arch: str):
    """Import the FP8 arch's own merge_infer.py by file (its subdir on sys.path so the
    flat `from lora import …` / dequant imports resolve). Avoids clashing with gemma's
    top-level merge_infer module name."""
    import importlib.util

    arch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), arch)
    if arch_dir not in sys.path:
        sys.path.insert(0, arch_dir)
    spec = importlib.util.spec_from_file_location(f"{arch}_merge_infer", os.path.join(arch_dir, "merge_infer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_model_and_tok(arch: str, base_model: str, lora_path: str, meta: dict, cfg: dict):
    """Return (model, tokenizer) for `arch` with the trained LoRA applied. Generation
    uses a normal causal mask (attn_implementation left to each arch's inference default),
    NOT the packed training attention."""
    import time as _time

    import torch
    from transformers import AutoTokenizer

    t0 = _time.time()

    if arch in ("gemma", "qwen"):
        # Dense: fold the simple lora_a/lora_b delta into the base weights.
        scaling = meta.get("scaling")
        if scaling is None:
            r = float(cfg.get("lora_r") or 16)
            alpha = float(cfg.get("lora_alpha") or 32)
            scaling = alpha / r
        tok = AutoTokenizer.from_pretrained(base_model)
        log(f"loading {base_model} (bf16, sdpa, device_map=auto) …")
        if arch == "gemma":
            from transformers import Gemma4ForConditionalGeneration
            model = Gemma4ForConditionalGeneration.from_pretrained(
                base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
            )
        else:
            # qwen3.6 dense (Qwen3_5ForConditionalGeneration) / MoE
            # (Qwen3_5MoeForConditionalGeneration) — load the exact class the config
            # declares (present in the qwen venv's transformers); fall back to Auto.
            from transformers import AutoConfig
            import transformers as _tf
            cfg_obj = AutoConfig.from_pretrained(base_model)
            cls_name = (getattr(cfg_obj, "architectures", None) or [None])[0]
            ModelCls = getattr(_tf, cls_name, None) if cls_name else None
            if ModelCls is None:
                from transformers import AutoModelForCausalLM as ModelCls  # type: ignore
            model = ModelCls.from_pretrained(
                base_model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
            )
        model.eval()
        log(f"loaded in {_time.time() - t0:.1f}s; merging LoRA (scaling={scaling:.4f}) …")
        _fold_linear_lora(model, lora_path, scaling)
        log("LoRA merged — ready for inference")
        return model, tok

    if arch in ("minimax", "mistral"):
        # FP8 MoE: attach adapters to the FP8 base (no bf16 merge) via the arch's
        # own merge_infer. Slow dequant inference path, but correct.
        mi = _load_arch_module(arch)
        log(f"loading FP8 base {base_model} (device_map=auto) …")
        if arch == "minimax":
            from transformers import MiniMaxM2ForCausalLM
            tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            model = MiniMaxM2ForCausalLM.from_pretrained(
                base_model, dtype=torch.bfloat16,
                attn_implementation=os.environ.get("MINIMAX_ATTN_IMPL", "flash_attention_3"),
                device_map="auto",
            )
        else:
            from transformers import Mistral3ForConditionalGeneration
            tok = AutoTokenizer.from_pretrained(base_model)
            config = mi.load_patched_config(base_model)
            model = Mistral3ForConditionalGeneration.from_pretrained(
                base_model, config=config, dtype=torch.bfloat16,
                attn_implementation=os.environ.get("MISTRAL_ATTN_IMPL", "flash_attention_3"),
                device_map="auto",
            )
        model.eval()
        log(f"loaded FP8 base in {_time.time() - t0:.1f}s; attaching LoRA adapters …")
        mi.attach_lora(model, lora_path, meta)
        log("LoRA attached — ready for inference")
        return model, tok

    raise RuntimeError(
        f"unsupported arch {arch!r} for LLM label export (supported: gemma, qwen, minimax, mistral)"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)

    arch = (cfg.get("arch") or "gemma").lower()

    import torch

    # Pin HF model cache to the shared volume used during training so the base model
    # download is skipped when it was already fetched for the training run.
    hf_home = cfg.get("hf_home") or os.environ.get("HF_HOME")
    if hf_home:
        os.environ["HF_HOME"] = hf_home

    work_dir = (cfg.get("work_dir") or "/share").rstrip("/")
    run_id = cfg.get("run_id") or "unknown"
    lora_dir = f"{work_dir}/sgpu-llm-label/{run_id}/lora"

    creds = cfg.get("s3_creds") or {}
    lora_path = _download_lora(cfg["model_s3"], creds, lora_dir)

    meta = load_meta(lora_path)
    base_model = cfg.get("base_model") or meta.get("model_id") or "google/gemma-4-31B-it"

    model, tok = _load_model_and_tok(arch, base_model, lora_path, meta, cfg)

    eval_rows = cfg.get("eval_rows") or []
    n = int(cfg.get("n_samples") or 0) or len(eval_rows)
    eval_rows = eval_rows[:n]
    max_new = int(cfg.get("max_new_tokens") or 512)

    if not eval_rows:
        emit({"error": "no eval rows in config — check that the eval dataset is non-empty"})
        return 1

    items = []
    log(f"generating {len(eval_rows)} response(s) (max_new_tokens={max_new}) …")
    for i, row in enumerate(eval_rows):
        msgs = list(row.get("messages") or [])
        if not msgs:
            log(f"row {i}: empty messages, skipping")
            continue
        try:
            # apply_chat_template returns a BatchEncoding dict in transformers 5.x.
            enc = tok.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = enc["input_ids"].to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_new,
                    do_sample=False,
                )
            response = tok.decode(
                out[0][input_ids.shape[-1]:], skip_special_tokens=True
            ).strip()
            items.append({
                "messages": msgs + [{"role": "assistant", "content": response}],
                "lov": row.get("lov", ""),
                "language": row.get("language", ""),
            })
            log(f"[{i + 1}/{len(eval_rows)}] {len(response)} chars")
        except Exception as e:  # noqa: BLE001
            log(f"row {i} failed: {e}")

    if not items:
        emit({"error": "generation produced no items (all rows failed or were empty)"})
        return 1

    emit({"items": items, "count": len(items)})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        emit({"error": str(e)})
        sys.exit(1)
