#!/usr/bin/env python3
"""Post-training Label-platform export for autotrain LLM (Gemma-4 LoRA).

Shipped to the run's VM as the whole `llm/` dir and run with the LLM trainer
venv (/share/autotrain-llm-{arch}/bin/python). It:

  1. downloads the LoRA checkpoint (lora.pt + lora_meta.json) from S3,
  2. loads the base Gemma-4 model and merges the LoRA adapters in-place
     (reusing merge_infer.merge_lora_, shipped as a sibling),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)

    arch = (cfg.get("arch") or "gemma").lower()
    if arch != "gemma":
        emit({"error": f"LLM label export supports only gemma arch in this version (got {arch!r}). "
                       f"MiniMax and Mistral support is planned."})
        return 1

    import torch
    from merge_infer import load_meta, merge_lora_
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

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
    scaling = meta.get("scaling")
    if scaling is None:
        r = float(cfg.get("lora_r") or 16)
        alpha = float(cfg.get("lora_alpha") or 32)
        scaling = alpha / r

    log(f"loading {base_model} (bf16, sdpa, device_map=auto) …")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(base_model)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    log(f"loaded in {time.time() - t0:.1f}s; merging LoRA (scaling={scaling:.4f}) …")
    merge_lora_(model, lora_path, scaling)
    log("LoRA merged — ready for inference")

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
