#!/usr/bin/env python3
"""Standalone transcription for the Autotrain "Try it" playground. Shipped to the
run's VM over SSH by the gateway and run with the VM's existing trainer venv
(torch/transformers/boto3/librosa from training). It downloads the finetuned
model from S3, transcribes one uploaded clip, and prints a single structured line:

  @@TEXT {"text": "...", "device": "cuda"|"cpu"}   on success
  @@TEXT {"error": "..."}                            on failure

Config (JSON, path via --config): {model_s3, region, endpoint, access_key,
secret_key, audio_path, model_dir, language?, task?, gpu?}. `gpu` is an explicit
device choice: a GPU index ("6"), "cpu", or "auto"/empty (most-free GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def emit(obj: dict) -> None:
    print("@@TEXT " + json.dumps(obj), flush=True)


def _pick_gpu() -> str | None:
    """Index of the GPU with the most free memory (>5 GiB), else None (CPU) — so
    a try-it doesn't OOM or disturb a training already running on the box."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
    except Exception:
        return None
    best, best_free = None, 0
    for line in out.strip().splitlines():
        try:
            idx, free = (x.strip() for x in line.split(","))
            free_i = int(free)
        except ValueError:
            continue
        if free_i > best_free:
            best, best_free = idx, free_i
    return best if (best is not None and best_free > 5000) else None


def _download_model(cfg: dict) -> str:
    import boto3
    from botocore.client import Config as BotoConfig

    s3 = cfg["model_s3"]
    assert s3.startswith("s3://"), f"bad model_s3: {s3}"
    bucket, _, prefix = s3[len("s3://"):].partition("/")
    prefix = prefix.rstrip("/") + "/"
    cli = boto3.client(
        "s3",
        region_name=cfg.get("region") or "us-east-1",
        endpoint_url=cfg.get("endpoint") or None,
        aws_access_key_id=cfg.get("access_key") or None,
        aws_secret_access_key=cfg.get("secret_key") or None,
        config=BotoConfig(signature_version="s3v4"),
    )
    dest = cfg.get("model_dir") or "/tmp/sgpu-tryit-model"
    os.makedirs(dest, exist_ok=True)
    n = 0
    for page in cli.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel:
                continue
            fp = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(fp) or dest, exist_ok=True)
            # Skip re-download when the cached file already matches (size).
            if not (os.path.exists(fp) and os.path.getsize(fp) == obj["Size"]):
                cli.download_file(bucket, key, fp)
            n += 1
    if n == 0:
        raise RuntimeError(f"no model files found under {s3}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)

    # Device: explicit GPU index ("6"), "cpu", or auto (most-free GPU). Pin via
    # CUDA_VISIBLE_DEVICES BEFORE importing torch so the chosen GPU is device 0.
    sel = str(cfg.get("gpu") or "auto").strip().lower()
    if sel == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        want_cuda = False
    elif sel.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = sel
        want_cuda = True
    else:
        g = _pick_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = g if g is not None else ""
        want_cuda = g is not None

    import librosa
    import torch
    from transformers import pipeline

    model_dir = _download_model(cfg)
    use_cuda = want_cuda and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        device=0 if use_cuda else -1,
        torch_dtype=dtype,
        chunk_length_s=30,  # transcribe arbitrarily long clips by chunking
    )
    gen_kwargs = {"task": cfg.get("task") or "transcribe"}
    if cfg.get("language"):
        gen_kwargs["language"] = cfg["language"]
    # Decode via librosa (the same path training uses) and hand the pipeline a raw
    # array — passing a file path makes the pipeline shell out to ffmpeg, which the
    # VM doesn't have. librosa (soundfile/audioread) already handles these clips.
    audio, sr = librosa.load(cfg["audio_path"], sr=16000, mono=True)
    out = asr({"raw": audio, "sampling_rate": sr}, generate_kwargs=gen_kwargs, return_timestamps=False)
    text = out["text"] if isinstance(out, dict) else str(out)
    emit({"text": (text or "").strip(), "device": device})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        emit({"error": str(e)})
        sys.exit(1)
