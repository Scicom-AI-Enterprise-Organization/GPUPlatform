#!/usr/bin/env python3
"""On-demand push of a finished autotrain model from S3 to a Hugging Face repo.

Shipped to the run's VM and run with the run's training venv (huggingface_hub +
boto3 are already installed there). It downloads the model artifact from S3 (the
`s3://…/best-model/` or `…/model/` prefix — a standard HF model folder) and
uploads it to a HF model repo, then prints ONE structured line the gateway parses:

  @@HF {"repo": "org/name", "url": "https://huggingface.co/org/name"}   on success
  @@HF {"error": "..."}                                                  on failure

Config (JSON via --config):
  {model_s3, region, endpoint, access_key, secret_key, model_dir, repo, token,
   private, hf_endpoint}
`endpoint` is the S3 endpoint; `hf_endpoint` (optional) is a custom Hugging Face
Hub (HF_ENDPOINT) — None/"" → huggingface.co.

LLM merge mode (merge=true): the S3 artifact is a raw LoRA checkpoint, not a loadable
model. We download it to `ckpt_dir`, merge it into `base_model` on GPU (per-arch, via
the shipped llm/ merge scripts run with `train_py`), write the merged HF model to
`model_dir`, then upload that. Extra keys: {merge, base_model, arch, merge_dtype,
llm_dir, train_py, ckpt_dir, visible_devices}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def emit(obj: dict) -> None:
    print("@@HF " + json.dumps(obj), flush=True)


def log(m: str) -> None:
    print(f"[hf-export] {m}", flush=True)


def _download_model(cfg: dict) -> str:
    import boto3
    from botocore.client import Config as BotoConfig

    s3 = cfg["model_s3"]
    assert s3.startswith("s3://"), f"bad model_s3: {s3}"
    bucket, _, prefix = s3[len("s3://"):].partition("/")
    prefix = prefix.rstrip("/") + "/"
    cli = boto3.client(
        "s3", region_name=cfg.get("region") or "us-east-1",
        endpoint_url=cfg.get("endpoint") or None,
        aws_access_key_id=cfg.get("access_key") or None,
        aws_secret_access_key=cfg.get("secret_key") or None,
        config=BotoConfig(signature_version="s3v4"),
    )
    dest = cfg["model_dir"]
    os.makedirs(dest, exist_ok=True)
    log(f"downloading {s3} → {dest} (cached on the VM after the first call) …")
    n = fetched = total = 0

    def _progress(rel: str, size: int):
        # boto3 calls this with the bytes transferred per chunk; log every ~10% so a
        # multi-GB safetensors shard isn't a single silent line for minutes.
        state = {"seen": 0, "last": -1}

        def cb(amount: int) -> None:
            state["seen"] += amount
            pct = int(state["seen"] * 100 / size) if size else 100
            if pct >= state["last"] + 10:
                state["last"] = pct
                log(f"    {rel}: {state['seen'] / 1e6:.0f}/{size / 1e6:.0f} MB ({pct}%)")
        return cb

    for page in cli.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel:
                continue
            fp = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(fp) or dest, exist_ok=True)
            if not (os.path.exists(fp) and os.path.getsize(fp) == obj["Size"]):
                log(f"  ↓ {rel} ({obj['Size'] / 1e6:.0f} MB)")
                cb = _progress(rel, obj["Size"]) if obj["Size"] > 100 * 1e6 else None
                cli.download_file(bucket, obj["Key"], fp, Callback=cb)
                fetched += 1
            else:
                log(f"  = {rel} (cached)")
            n += 1
            total += obj["Size"]
    if n == 0:
        raise RuntimeError(f"no model files found under {s3}")
    log(f"model ready: {n} files · {total / 1e6:.0f} MB · {fetched} fetched / {n - fetched} cached")
    return dest


def _merge_lora(cfg: dict) -> str:
    """LLM merge mode: download the raw LoRA checkpoint from S3, merge it into the base
    model on GPU (per-arch, via the shipped llm/ merge scripts), and return the merged
    dir — a loadable HF model folder. Reuses llm_playground.merge_lora_to_dir +
    ensure_processor_configs (imported from the shipped `llm_dir`)."""
    import sys as _sys

    # 1) Download the checkpoint (lora.pt + lora_meta.json) into ckpt_dir.
    ckpt_dir = cfg["ckpt_dir"]
    _download_model({**cfg, "model_dir": ckpt_dir})
    lora = os.path.join(ckpt_dir, "lora.pt")
    if not os.path.exists(lora):
        pts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if not pts:
            raise RuntimeError(f"no LoRA checkpoint (.pt) found under {cfg['model_s3']}")
        lora = os.path.join(ckpt_dir, pts[0])

    # 2) Merge LoRA → base on GPU. merge_lora_to_dir dispatches per-arch (gemma folds bf16
    #    via merge_infer.py; qwen/minimax/mistral dequant+fold via their merge scripts).
    merged = cfg["model_dir"]
    llm_dir = cfg["llm_dir"]
    _lp = os.path.join(llm_dir, "llm_playground.py")
    if not os.path.exists(_lp):
        listing = os.listdir(llm_dir) if os.path.isdir(llm_dir) else "MISSING DIR"
        raise RuntimeError(f"llm_playground.py not shipped to {llm_dir} (contents: {listing})")
    _sys.path.insert(0, llm_dir)
    from llm_playground import merge_lora_to_dir, ensure_processor_configs

    base_model = cfg.get("base_model") or ""
    env = dict(os.environ)
    if cfg.get("visible_devices"):
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["visible_devices"])
    # The base model is usually gated (e.g. google/gemma-*) → the merge's from_pretrained
    # needs an HF token WITH access to the BASE model (a different account than the push
    # target may own). Use base_hf_token (set by the caller; falls back to the push token).
    tok = (cfg.get("base_hf_token") or cfg.get("token") or "").strip()
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
        # Do NOT force hf_transfer on here: it's known to STALL on the tm/tm-2 boxes
        # (see training/llm/CLAUDE.md — the same reason training's own env_vars sets
        # HF_HUB_ENABLE_HF_TRANSFER=0 there). This used to setdefault it to "1",
        # silently re-enabling the exact accelerator training explicitly avoids —
        # `env` already started as a copy of this process's inherited os.environ, so
        # whatever the run's own env_vars decided (e.g. explicitly "0") carries through
        # unmodified; only opt in if the caller's config says so.
    train_py = cfg.get("train_py") or _sys.executable
    log(f"merging LoRA into {base_model} → {merged} (dtype={cfg.get('merge_dtype') or 'fp16'}) …")
    rc = merge_lora_to_dir(base_model, lora, merged, train_py, llm_dir,
                           dtype=(cfg.get("merge_dtype") or "fp16"), env=env)
    if rc != 0:
        raise RuntimeError(f"LoRA merge failed (rc={rc})")
    # Multimodal archs (gemma/mistral) need their processor/preprocessor configs alongside
    # the merged weights for the repo to load + serve downstream.
    try:
        ensure_processor_configs(base_model, merged)
    except Exception as e:  # noqa: BLE001
        log(f"processor-config fixup skipped: {e}")
    log(f"merged model ready at {merged}")
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)

    repo = (cfg.get("repo") or "").strip()
    if not repo:
        emit({"error": "no target repo"})
        return 1
    token = (cfg.get("token") or "").strip() or None
    private = bool(cfg.get("private"))
    # Custom HF_ENDPOINT (a self-hosted, HF-compatible Hub) — None → huggingface.co.
    hf_endpoint = (cfg.get("hf_endpoint") or "").strip().rstrip("/") or None
    base_url = hf_endpoint or "https://huggingface.co"
    # The self-hosted mirror speaks LFS but NOT Xet — a modern huggingface_hub (with
    # hf_xet) otherwise probes `{endpoint}/api/models/{repo}/xet-write-token/{rev}`,
    # gets a 404, and aborts the push. Force the LFS path for mirror targets. Must be
    # set before huggingface_hub is imported (the constant is read at import time).
    if hf_endpoint:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    model_dir = _merge_lora(cfg) if cfg.get("merge") else _download_model(cfg)

    from huggingface_hub import HfApi
    api = HfApi(token=token, endpoint=hf_endpoint)
    log(f"create_repo {repo} (private={private}) at {base_url} …")
    api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)
    # Count + size the payload so the log shows what's about to upload (the per-file
    # progress bars stream too — the gateway runs this with stderr merged into stdout).
    files = [os.path.join(r, fn) for r, _d, fns in os.walk(model_dir) for fn in fns]
    nbytes = sum(os.path.getsize(p) for p in files)
    log(f"uploading {len(files)} file(s) · {nbytes / 1e6:.0f} MB → {repo} (this can take a few minutes) …")
    api.upload_folder(folder_path=model_dir, repo_id=repo, repo_type="model")
    log(f"upload complete → {base_url}/{repo}")
    emit({"repo": repo, "url": f"{base_url}/{repo}"})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        emit({"error": str(e)})
        sys.exit(1)
