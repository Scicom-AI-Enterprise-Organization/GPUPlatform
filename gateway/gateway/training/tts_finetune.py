#!/usr/bin/env python3
"""Standalone Qwen3 + NeuCodec TTS finetune orchestrator — shipped to a RunPod
pod / VM by the gateway's Autotrain runner (task_type=tts) and run over SSH.

It vendors three scripts from the `finetune-multilingual-tts` reference (sibling
`tts/` dir, SFTP'd alongside this file) and runs the full pipeline:
  1. resolve the registered {audio, transcription[, speaker]} dataset → write
     audio files + `audio_paths.json` + `meta.jsonl`,
  2. `convert_neucodec.py` — encode audio → NeuCodec speech tokens,
  3. `pack_stage1.py` — pack tokens+text into an MDS streaming dataset,
  4. `torchrun qwen3_tts_flash.py` — finetune Qwen3 as a causal LM over the pack
     (loss-only; metrics go to W&B/MLflow via HF Trainer report_to),
then upload the checkpoint to S3 (+ optional HF push).

No gateway imports — config arrives as a single JSON file (--config). The
sub-scripts emit `[AUTOTRAIN_PROGRESS] step=… percent=…`; the gateway parses
those from the stream. This file emits @@ARTIFACT / @@DONE / @@ERROR.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TTS_DIR = os.path.join(THIS_DIR, "tts")
_LOSS_RE = re.compile(r"'loss':\s*([0-9.eE+-]+)")


def log(msg: str) -> None:
    print(msg, flush=True)


def emit(tag: str, obj: dict) -> None:
    print(f"@@{tag} {json.dumps(obj)}", flush=True)


def parse_precision(p):
    """'<load>-<amp>' → (torch_dtype_name, amp): weight load dtype + the
    mixed-precision (AMP) train dtype. Back-compat: bare 'bf16'/'fp16' = load
    fp32 + that AMP; 'fp32' = full fp32 (no AMP)."""
    p = (p or "fp32-bf16").lower()
    if "-" in p:
        load, amp = p.split("-", 1)
    elif p == "fp32":
        load, amp = "fp32", ""
    else:
        load, amp = "fp32", p
    load_dt = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}.get(load, "float32")
    return load_dt, (amp if amp in ("bf16", "fp16") else "")


def _run_loss(cmd: list[str], cwd: str, env: dict) -> float | None:
    """Run a command, tee its stdout, and return the last HF-logged train loss
    (so a sweep can rank TTS trials, which are loss-only)."""
    log(f"[gateway] $ {' '.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = None
    for line in p.stdout:  # type: ignore[union-attr]
        print(line, end="", flush=True)
        m = _LOSS_RE.search(line)
        if m:
            try:
                last = float(m.group(1))
            except ValueError:
                pass
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)
    return last


# install.sh from the reference, pip form (the pod image already has CUDA torch;
# we pin the stack the trainer needs).
DEPS = [
    "torch==2.9.1", "transformers==4.57.3", "accelerate",
    "mosaicml-streaming", "librosa", "soundfile", "datasets", "wandb",
    "neucodec", "pandas", "pyarrow", "multiprocess", "liger-kernel",
    "git+https://github.com/apple/ml-cross-entropy",
]


def ensure_deps(report_to: list[str]) -> None:
    try:
        import transformers, datasets, streaming, neucodec  # noqa: F401
        log("[deps] core TTS stack present")
    except Exception:
        log("[deps] installing TTS stack (torch/transformers/neucodec/streaming/…)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *DEPS])
    if "mlflow" in report_to:
        try:
            import mlflow  # noqa: F401
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mlflow"])


# --------------------------------------------------------------------------
# dataset → audio files + meta.jsonl + audio_paths.json
# --------------------------------------------------------------------------
def _s3_client(ds: dict):
    import boto3
    from botocore.client import Config as BotoConfig
    return boto3.client(
        "s3", region_name=ds.get("region") or "us-east-1",
        endpoint_url=ds.get("endpoint") or None,
        aws_access_key_id=ds.get("access_key") or None,
        aws_secret_access_key=ds.get("secret_key") or None,
        config=BotoConfig(signature_version="s3v4"),
    )


def _read_metadata_rows(ds: dict) -> list[dict]:
    import csv, io
    cli = _s3_client(ds)
    body = cli.get_object(Bucket=ds["bucket"], Key=ds["metadata_key"])["Body"].read()
    text = body.decode("utf-8", errors="replace")
    fmt = (ds.get("format") or "").lower() or (
        "jsonl" if ds["metadata_key"].endswith(".jsonl")
        else "json" if ds["metadata_key"].endswith(".json") else "csv"
    )
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(text)))
    if fmt == "jsonl":
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("data", data.get("rows", []))


def build_dataset(cfg: dict, work: str) -> tuple[str, str]:
    """Materialise audio under work/audio/, write audio_paths.json + meta.jsonl.
    Returns (audio_paths_json, meta_jsonl)."""
    import soundfile as sf

    ds = cfg["dataset"]
    audio_field = ds.get("audio_field") or "audio"
    text_field = ds.get("transcription_field") or "transcription"
    speaker_field = cfg.get("speaker_field") or "speaker"
    default_speaker = cfg.get("default_speaker") or "speaker"

    audio_dir = os.path.join(work, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    rows: list[dict] = []

    if ds.get("kind") == "hf":
        from datasets import load_dataset
        dd = load_dataset(ds["hf_repo"], token=ds.get("hf_token") or None)
        idx = 0
        for split_name, split in dd.items():
            tf = (ds.get("split_fields") or {}).get(split_name, text_field)
            for r in split:
                a = r[audio_field]
                rel = f"audio/{idx}.wav"
                sf.write(os.path.join(work, rel), a["array"], a["sampling_rate"])
                rows.append({
                    "filename_audio": rel, "text": str(r.get(tf, "")),
                    "speaker": str(r.get(speaker_field) or default_speaker),
                })
                idx += 1
    else:
        cli = _s3_client(ds)
        prefix = (ds.get("audio_prefix") or "").strip("/")
        for r in _read_metadata_rows(ds):
            ref = r.get(audio_field)
            txt = r.get(text_field)
            if not ref or txt is None:
                continue
            base = os.path.basename(str(ref).split("?")[0]) or "a.wav"
            rel = f"audio/{base}"
            dest = os.path.join(work, rel)
            try:
                if str(ref).startswith(("http://", "https://")):
                    import urllib.request
                    urllib.request.urlretrieve(str(ref), dest)
                else:
                    key = "/".join(p for p in [prefix, str(ref).lstrip("/")] if p)
                    cli.download_file(ds["bucket"], key, dest)
            except Exception as e:  # noqa: BLE001
                log(f"[data] skip {ref!r}: {e}")
                continue
            rows.append({
                "filename_audio": rel, "text": str(txt),
                "speaker": str(r.get(speaker_field) or default_speaker),
            })

    if not rows:
        raise RuntimeError("no usable {audio, transcription} rows resolved from the dataset")

    audio_paths = os.path.join(work, "audio_paths.json")
    with open(audio_paths, "w") as f:
        json.dump([r["filename_audio"] for r in rows], f)
    meta = os.path.join(work, "meta.jsonl")
    with open(meta, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"[data] {len(rows)} rows · audio under {audio_dir}")
    return audio_paths, meta


# --------------------------------------------------------------------------
def run(cfg: dict) -> None:
    work = os.path.abspath(cfg.get("work_dir") or "/workspace/autotrain-tts")
    os.makedirs(work, exist_ok=True)

    # tracking env (gateway injected the resolved secrets in cfg["tracking"]["env"])
    tracking = cfg.get("tracking") or {}
    for k, v in (tracking.get("env") or {}).items():
        if v not in (None, ""):
            os.environ[k] = str(v)
    report_to = list(tracking.get("report_to") or [])

    ensure_deps(report_to)

    model = cfg.get("base_model") or "Qwen/Qwen3-1.7B-Base"
    tokenizer = cfg.get("tokenizer") or "Scicom-intl/Multilingual-Expressive-TTS-1.7B"
    block_size = int(cfg.get("block_size", 10240))
    seq_len = int(cfg.get("pack_sequence_length", 4096))
    epochs = int(cfg.get("max_epochs", 3))
    batch = int(cfg.get("batch_size", 8))
    grad_accum = int(cfg.get("grad_accum", 4))
    lr = float(cfg.get("learning_rate", 2e-5))
    load_dt, amp = parse_precision(cfg.get("precision"))
    gpus = max(1, int(cfg.get("gpu_count", 1)))

    audio_paths, meta = build_dataset(cfg, work)
    packed = os.path.join(work, "packed")
    out_dir = os.path.join(work, "out")
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if report_to and cfg.get("run_name"):
        env.setdefault("WANDB_NAME", cfg["run_name"])

    def step(cmd: list[str], cwd: str) -> None:
        log(f"[gateway] $ {' '.join(cmd)}")
        subprocess.check_call(cmd, cwd=cwd, env=env)

    # 1. NeuCodec tokenization
    step([sys.executable, "-u", os.path.join(TTS_DIR, "convert_neucodec.py"),
          "--file", audio_paths], cwd=work)
    # 2. pack into MDS
    step([sys.executable, "-u", os.path.join(TTS_DIR, "pack_stage1.py"),
          "--dataset", meta, "--output_dir", packed,
          "--tokenizer", tokenizer, "--sequence_length", str(seq_len)], cwd=work)
    # 3. finetune via torchrun
    log(f"[train] precision: load={load_dt} amp={amp or 'none'}")
    dtype_args = ["--torch_dtype", load_dt]
    if amp == "bf16":
        dtype_args += ["--bf16"]
    elif amp == "fp16":
        dtype_args += ["--fp16"]
    report_flag = ",".join(report_to) if report_to else "none"
    last_loss = _run_loss([
        "torchrun", f"--nproc_per_node={gpus}", os.path.join(TTS_DIR, "qwen3_tts_flash.py"),
        "--model_name_or_path", model,
        "--train_file", packed,
        "--output_dir", out_dir,
        "--do_train", "--do_eval", "false",
        "--num_train_epochs", str(epochs),
        "--per_device_train_batch_size", str(batch),
        "--gradient_accumulation_steps", str(grad_accum),
        "--learning_rate", str(lr),
        "--warmup_steps", str(int(cfg.get("warmup_steps", 0))),
        "--block_size", str(block_size),
        "--logging_steps", "1",
        "--save_strategy", "epoch",
        "--save_total_limit", "3",
        "--gradient_checkpointing", "true",
        "--ddp_find_unused_parameters", "false",
        "--dataloader_num_workers", "5",
        "--remove_unused_columns", "false",
        "--report_to", report_flag,
        *dtype_args,
    ], work, env)

    # ---- upload checkpoint to S3 + optional HF ----
    art = cfg.get("artifacts") or {}
    s3_uri = None
    if art.get("bucket") and os.path.isdir(out_dir):
        cli = _s3_client(art)
        base_key = art["prefix"].rstrip("/") + "/model"
        for root, _dirs, files in os.walk(out_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, out_dir)
                cli.upload_file(fp, art["bucket"], f"{base_key}/{rel}")
        s3_uri = f"s3://{art['bucket']}/{base_key}/"
        log(f"[upload] checkpoint → {s3_uri}")

    hf_repo = None
    if cfg.get("hf_push_repo") and cfg.get("hf_token"):
        try:
            from huggingface_hub import HfApi
            HfApi().upload_folder(
                folder_path=out_dir, repo_id=cfg["hf_push_repo"],
                repo_type="model", token=cfg["hf_token"],
            )
            hf_repo = cfg["hf_push_repo"]
            log(f"[upload] pushed → https://huggingface.co/{hf_repo}")
        except Exception as e:  # noqa: BLE001
            log(f"[upload] HF push failed: {e}")

    emit("ARTIFACT", {"s3_uri": s3_uri, "hf_repo": hf_repo})
    emit("DONE", {"best": ({"loss": last_loss} if last_loss is not None else None),
                  "epochs": epochs, "stopped_early": False})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--deps-only", action="store_true",
                    help="install dependencies then exit (used by the sweep orchestrator)")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    try:
        if a.deps_only:
            ensure_deps((cfg.get("tracking") or {}).get("report_to") or [])
            log("[deps] ready (deps-only)")
            return 0
        run(cfg)
        return 0
    except Exception as e:  # noqa: BLE001
        emit("ERROR", {"message": str(e)})
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
