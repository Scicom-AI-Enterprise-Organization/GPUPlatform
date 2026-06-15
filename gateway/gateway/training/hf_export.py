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

    model_dir = _download_model(cfg)

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
