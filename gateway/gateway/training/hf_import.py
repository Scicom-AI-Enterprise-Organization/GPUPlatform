#!/usr/bin/env python3
"""On-demand restore of a model FROM a Hugging Face repo INTO a run's S3 model prefix.

The exact inverse of hf_export.py. Used when a run's trained model artifact has been
deleted from S3 (GC'd, bucket wiped, …) so the "Try it" playground and the label
export both fail with `no model files found under s3://…/model/`. This downloads a
standard HF model folder from a repo and uploads every file back under that prefix,
so the downstream S3-download paths (tts_infer.py / transcribe.py / hf_export.py)
find the model again.

Runs on the GATEWAY (in its venv — huggingface_hub + boto3 are already there); no
GPU and no training box needed (the artifact is already a complete model). Prints ONE
structured line the gateway parses:

  @@HFIMPORT {"repo": "org/name", "s3": "s3://…/model/", "files": N, "bytes": B}  on success
  @@HFIMPORT {"error": "..."}                                                      on failure

Config (JSON via --config):
  {repo, revision, token, hf_endpoint, model_s3, region, endpoint, access_key,
   secret_key, model_dir}
`endpoint` is the S3 endpoint; `hf_endpoint` (optional) is a custom Hugging Face Hub
(HF_ENDPOINT) — None/"" → huggingface.co.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# hf_hub's local_dir keeps a bookkeeping cache here — never upload it to S3.
_SKIP_DIRS = {".cache"}


def emit(obj: dict) -> None:
    print("@@HFIMPORT " + json.dumps(obj), flush=True)


def log(m: str) -> None:
    print(f"[hf-import] {m}", flush=True)


def _download_repo(cfg: dict) -> str:
    """snapshot_download the whole repo into model_dir (real files, no symlinks) and
    return the dir. A custom hf_endpoint (self-hosted mirror) speaks LFS but NOT Xet, so
    disable Xet before huggingface_hub is imported (the constant is read at import time)."""
    repo = (cfg.get("repo") or "").strip()
    assert repo, "no source repo"
    hf_endpoint = (cfg.get("hf_endpoint") or "").strip().rstrip("/") or None
    if hf_endpoint:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    from huggingface_hub import snapshot_download

    dest = cfg["model_dir"]
    os.makedirs(dest, exist_ok=True)
    revision = (cfg.get("revision") or "").strip() or None
    base = hf_endpoint or "https://huggingface.co"
    log(f"downloading {base}/{repo}"
        + (f"@{revision}" if revision else "")
        + f" → {dest} …")
    t0 = time.time()
    snapshot_download(
        repo_id=repo, repo_type="model", revision=revision,
        local_dir=dest, token=(cfg.get("token") or None),
        endpoint=hf_endpoint,
    )
    log(f"snapshot downloaded in {time.time() - t0:.1f}s")
    return dest


def _upload_to_s3(cfg: dict, model_dir: str) -> tuple[int, int]:
    """Upload every file under model_dir to s3://bucket/prefix/<relpath>, mirroring
    hf_export.py's download loop in reverse (per-shard % progress for big safetensors).
    Returns (n_files, total_bytes)."""
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

    # Collect the files first (skip hf_hub's .cache bookkeeping dir) so the log can show
    # the payload up front, like the export side.
    files: list[tuple[str, str]] = []  # (abs_path, rel_key)
    for root, dnames, fnames in os.walk(model_dir):
        dnames[:] = [d for d in dnames if d not in _SKIP_DIRS]
        for fn in fnames:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, model_dir)
            files.append((fp, rel.replace(os.sep, "/")))
    nbytes = sum(os.path.getsize(fp) for fp, _ in files)
    log(f"uploading {len(files)} file(s) · {nbytes / 1e6:.0f} MB → {s3} …")

    def _progress(rel: str, size: int):
        # boto3 calls this with bytes-per-chunk; log every ~10% so a multi-GB safetensors
        # shard isn't a single silent line for minutes (mirrors hf_export._progress).
        state = {"seen": 0, "last": -1}

        def cb(amount: int) -> None:
            state["seen"] += amount
            pct = int(state["seen"] * 100 / size) if size else 100
            if pct >= state["last"] + 10:
                state["last"] = pct
                log(f"    {rel}: {state['seen'] / 1e6:.0f}/{size / 1e6:.0f} MB ({pct}%)")
        return cb

    for fp, rel in files:
        size = os.path.getsize(fp)
        log(f"  ↑ {rel} ({size / 1e6:.0f} MB)")
        cb = _progress(rel, size) if size > 100 * 1e6 else None
        cli.upload_file(fp, bucket, prefix + rel, Callback=cb)
    return len(files), nbytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)

    repo = (cfg.get("repo") or "").strip()
    if not repo:
        emit({"error": "no source repo"})
        return 1

    model_dir = _download_repo(cfg)
    n, nbytes = _upload_to_s3(cfg, model_dir)
    if n == 0:
        emit({"error": f"the repo {repo} downloaded no files — nothing to upload"})
        return 1
    log(f"restore complete: {n} file(s) · {nbytes / 1e6:.0f} MB → {cfg['model_s3']}")
    emit({"repo": repo, "s3": cfg["model_s3"], "files": n, "bytes": nbytes})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        emit({"error": str(e)})
        sys.exit(1)
