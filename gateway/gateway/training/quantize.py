#!/usr/bin/env python3
"""Standalone LLM quantization worker — shipped to a RunPod pod / VM by the
gateway's Quantization runner (quantization_api.py) and executed over SSH. It has
NO gateway imports; everything it needs arrives in a single JSON config file (path
passed via --config). It installs its own deps into an isolated uv venv, pulls a
base model from HuggingFace, quantizes it with llm-compressor (compressed-tensors),
uploads the compressed model to S3, and optionally pushes it to the HF Hub.

Contract with the gateway (parsed from stdout):
  @@PROGRESS {json}  free-form stage markers: {"stage": str, "percent": float}
  @@SIZES {json}     {"source_gb": float, "quantized_gb": float}
  @@ARTIFACT {json}  after upload: {"s3_uri": str, "hf_repo": str|null}
  @@DONE {json}      final: {"scheme": str}
  @@ERROR {json}     fatal: {"message": str}
Every other line is free-form progress and streamed to the job's log.

Two phases (mirrors the autotrain trainers):
  --deps-only : system python, build/reuse the uv venv, then exit.
  (run)       : gateway launches {venv}/bin/python quantize.py --config …; the heavy
                imports resolve from the venv (done lazily inside run()).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback


def log(msg: str) -> None:
    print(msg, flush=True)


def emit(tag: str, obj: dict) -> None:
    print(f"@@{tag} {json.dumps(obj)}", flush=True)


# --------------------------------------------------------------------------
# Quantization schemes — keep in sync with quantization_api._SCHEMES.
# Each entry: (needs_calibration, recipe-builder name). The builder is resolved in
# _build_recipe() lazily (llm-compressor lives in the venv, not system python).
# --------------------------------------------------------------------------
QUANT_SCHEMES = {
    "fp8-dynamic": False,
    "w4a16": True,
    "w8a8-int8": True,
    "fp8": True,
    "nvfp4": True,
    "awq": True,
}


# --------------------------------------------------------------------------
# Dependency bootstrap — an isolated uv venv with the llm-compressor stack.
# --------------------------------------------------------------------------
DEFAULT_VENV = "/share/quant-llmcompressor"


def _ensure_venv(cfg: dict) -> str:
    """Create/reuse an isolated uv venv with the llm-compressor stack; return its
    python. Idempotent — fast when the venv is ready."""
    import shutil

    venv = (cfg.get("venv_path") or DEFAULT_VENV).rstrip("/")
    py = os.path.join(venv, "bin", "python")
    env = {**os.environ, "PIP_CONSTRAINT": "", "PIP_REQUIRE_HASHES": "0"}
    # llmcompressor pulls compressed-tensors + a compatible transformers/torch. We
    # don't pin torch to a CUDA index here: the pod image already carries a matching
    # CUDA torch (cu1300 default), and llmcompressor's torch dep is broad.
    pkgs = [
        "llmcompressor>=0.6", "compressed-tensors", "transformers>=4.48",
        "datasets>=2.20", "accelerate>=0.30", "boto3", "huggingface_hub",
    ]
    check_mods = ["torch", "transformers", "datasets", "llmcompressor", "compressed_tensors", "boto3"]

    def _present() -> bool:
        probe = "import " + ", ".join(check_mods) + "\n"
        try:
            subprocess.check_call([py, "-c", probe],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    if os.path.exists(py) and _present():
        log(f"[deps] llm-compressor venv ready: {py}")
        return py
    have_uv = shutil.which("uv") is not None

    def _pip(*args):
        if have_uv:
            subprocess.check_call(["uv", "pip", "install", "--python", py, *args], env=env)
        else:
            subprocess.check_call([py, "-m", "pip", "install", "-q", *args], env=env)

    if not os.path.exists(py):
        log(f"[deps] creating venv {venv} …")
        if have_uv:
            subprocess.check_call(["uv", "venv", venv, "--python", "3.12"], env=env)
        else:
            subprocess.check_call([sys.executable, "-m", "venv", venv], env=env)
            subprocess.check_call([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], env=env)
    log(f"[deps] installing llm-compressor stack into {venv} …")
    _pip(*pkgs)
    log(f"[deps] llm-compressor venv ready: {py}")
    return py


# --------------------------------------------------------------------------
# Calibration dataset → list[str] of text samples.
# --------------------------------------------------------------------------
def _s3_client(ds: dict):
    import boto3
    from botocore.client import Config as BotoConfig

    return boto3.client(
        "s3",
        region_name=ds.get("region") or "us-east-1",
        endpoint_url=ds.get("endpoint") or None,
        aws_access_key_id=ds.get("access_key") or None,
        aws_secret_access_key=ds.get("secret_key") or None,
        config=BotoConfig(signature_version="s3v4"),
    )


def _read_s3_metadata_rows(ds: dict) -> list[dict]:
    import csv
    import io

    cli = _s3_client(ds)
    body = cli.get_object(Bucket=ds["bucket"], Key=ds["metadata_key"])["Body"].read()
    text = body.decode("utf-8", errors="replace")
    fmt = (ds.get("format") or "").lower()
    if not fmt:
        key = ds["metadata_key"]
        fmt = "jsonl" if key.endswith(".jsonl") else ("json" if key.endswith(".json") else "csv")
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(text)))
    if fmt == "jsonl":
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("data", data.get("rows", []))


def _guess_text_field(row: dict) -> str | None:
    for cand in ("text", "content", "prompt", "instruction", "question", "input", "sentence"):
        if cand in row and isinstance(row[cand], str):
            return cand
    for k, v in row.items():
        if isinstance(v, str) and v.strip():
            return k
    return None


def _messages_to_text(tokenizer, messages) -> str | None:
    """Render an OpenAI-style [{role,content}] list to a training string via the
    tokenizer's chat template (falls back to concatenation)."""
    if not isinstance(messages, list) or not messages:
        return None
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))


def _load_calibration_texts(cfg: dict, tokenizer) -> list[str]:
    """Return up to num_calibration_samples raw text strings from the calibration
    dataset spec (resolved by the gateway from a Datasets resource)."""
    ds = cfg.get("dataset") or {}
    n = int(cfg.get("num_calibration_samples") or 512)
    text_field = cfg.get("calib_text_field")
    msg_field = cfg.get("calib_messages_field")
    kind = ds.get("kind")
    texts: list[str] = []

    if kind in ("hf", "llm"):
        from datasets import load_dataset

        repo = ds.get("hf_repo")
        token = ds.get("hf_token") or None
        log(f"[calib] loading HF dataset {repo} …")
        # Stream so we don't pull a huge dataset to grab a few hundred rows.
        try:
            data = load_dataset(repo, split="train", streaming=True, token=token)
        except Exception:
            data = load_dataset(repo, split="train", token=token)
        for i, row in enumerate(data):
            if len(texts) >= n:
                break
            t = None
            mf = msg_field or ds.get("messages_field") or ("messages" if "messages" in row else None)
            if mf and mf in row:
                t = _messages_to_text(tokenizer, row[mf])
            if t is None:
                # The dataset's transcription_field is a platform default ("transcription")
                # that often doesn't exist on a plain text repo — fall back to guessing
                # rather than silently skipping every row.
                tf = text_field or ds.get("transcription_field")
                if not tf or tf not in row:
                    tf = _guess_text_field(row)
                if tf and tf in row and isinstance(row[tf], str):
                    t = row[tf]
            if t and t.strip():
                texts.append(t)
    elif kind in ("s3", "upload"):
        log("[calib] loading S3 metadata rows …")
        rows = _read_s3_metadata_rows(ds)
        for row in rows:
            if len(texts) >= n:
                break
            tf = text_field or ds.get("transcription_field")
            if not tf or tf not in row:
                tf = _guess_text_field(row)
            if tf and tf in row and isinstance(row[tf], str) and row[tf].strip():
                texts.append(row[tf])
    else:
        raise RuntimeError(
            f"calibration dataset kind '{kind}' is not supported — use a text dataset "
            f"(HuggingFace, LLM chat, or an uploaded/S3 text file)"
        )

    if not texts:
        raise RuntimeError("no calibration text found — check the dataset's text/messages field")
    log(f"[calib] collected {len(texts)} calibration samples")
    return texts


def _build_calibration_dataset(cfg: dict, tokenizer):
    """Tokenize calibration texts into a HF Dataset llm-compressor's oneshot() can
    consume (columns: input_ids, attention_mask)."""
    from datasets import Dataset as HFDataset

    texts = _load_calibration_texts(cfg, tokenizer)
    max_len = int(cfg.get("max_seq_length") or 2048)

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    ds = HFDataset.from_dict({"text": texts})
    ds = ds.map(_tok, batched=True, remove_columns=["text"])
    return ds


# --------------------------------------------------------------------------
# Recipe builder — scheme id → llm-compressor modifier(s).
# --------------------------------------------------------------------------
def _build_recipe(cfg: dict):
    scheme = cfg["scheme"]
    ignore = list(cfg.get("ignore_layers") or ["lm_head"])
    from llmcompressor.modifiers.quantization import QuantizationModifier

    if scheme == "fp8-dynamic":
        return QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=ignore)
    if scheme == "fp8":
        # Static FP8 (per-tensor activation scales) — calibration fills the scales.
        return QuantizationModifier(targets="Linear", scheme="FP8", ignore=ignore)
    if scheme == "nvfp4":
        return QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=ignore)
    if scheme == "w4a16":
        from llmcompressor.modifiers.quantization import GPTQModifier
        return GPTQModifier(
            targets="Linear", scheme="W4A16", ignore=ignore,
            dampening_frac=float(cfg.get("dampening_frac") or 0.01),
        )
    if scheme == "w8a8-int8":
        from llmcompressor.modifiers.quantization import GPTQModifier
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        return [
            SmoothQuantModifier(smoothing_strength=float(cfg.get("smoothing_strength") or 0.8)),
            GPTQModifier(
                targets="Linear", scheme="W8A8", ignore=ignore,
                dampening_frac=float(cfg.get("dampening_frac") or 0.01),
            ),
        ]
    if scheme == "awq":
        from llmcompressor.modifiers.awq import AWQModifier
        return AWQModifier(targets="Linear", scheme="W4A16", ignore=ignore)
    raise RuntimeError(f"unknown scheme '{scheme}'")


def _dir_size_gb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return round(total / (1024 ** 3), 3)


# --------------------------------------------------------------------------
def run(cfg: dict) -> None:
    import tempfile
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot

    scheme = cfg["scheme"]
    needs_calib = QUANT_SCHEMES.get(scheme, True)
    source = cfg["source_model"]
    hf_token = cfg.get("hf_token") or None
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)

    emit("PROGRESS", {"stage": "loading-model", "percent": 5})
    log(f"[quant] loading {source} (scheme={scheme}) …")
    model = AutoModelForCausalLM.from_pretrained(
        source, torch_dtype="auto", device_map="auto", token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(source, token=hf_token)

    recipe = _build_recipe(cfg)
    oneshot_kwargs: dict = {"model": model, "recipe": recipe}
    if needs_calib:
        emit("PROGRESS", {"stage": "calibrating", "percent": 25})
        calib = _build_calibration_dataset(cfg, tokenizer)
        oneshot_kwargs.update({
            "dataset": calib,
            "num_calibration_samples": min(int(cfg.get("num_calibration_samples") or 512), len(calib)),
            "max_seq_length": int(cfg.get("max_seq_length") or 2048),
        })

    emit("PROGRESS", {"stage": "quantizing", "percent": 45})
    log("[quant] running llm-compressor oneshot …")
    oneshot(**oneshot_kwargs)

    out_dir = os.path.join(tempfile.mkdtemp(prefix="sgpu-quant-"), source.split("/")[-1] + f"-{scheme}")
    os.makedirs(out_dir, exist_ok=True)
    emit("PROGRESS", {"stage": "saving", "percent": 75})
    log(f"[quant] saving compressed model → {out_dir}")
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)

    try:
        emit("SIZES", {"quantized_gb": _dir_size_gb(out_dir)})
    except Exception:
        pass

    # ---- upload compressed model to S3 ----
    art = cfg.get("artifacts") or {}
    s3_uri = None
    if art.get("bucket"):
        emit("PROGRESS", {"stage": "uploading", "percent": 88})
        cli = _s3_client(art)
        base_key = art["prefix"].rstrip("/") + "/model"
        for root, _dirs, files in os.walk(out_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, out_dir)
                cli.upload_file(fp, art["bucket"], f"{base_key}/{rel}")
        s3_uri = f"s3://{art['bucket']}/{base_key}/"
        log(f"[upload] quantized model → {s3_uri}")

    # ---- optional HF push ----
    hf_repo = None
    if cfg.get("hf_push_repo") and hf_token:
        try:
            emit("PROGRESS", {"stage": "hf-push", "percent": 95})
            from huggingface_hub import HfApi

            repo = cfg["hf_push_repo"]
            api = HfApi(token=hf_token)
            api.create_repo(repo, private=bool(cfg.get("hf_push_private", True)),
                            exist_ok=True, repo_type="model")
            api.upload_folder(folder_path=out_dir, repo_id=repo, repo_type="model")
            hf_repo = repo
            log(f"[upload] pushed quantized model → https://huggingface.co/{repo}")
        except Exception as e:  # noqa: BLE001
            log(f"[upload] HF push failed: {e}")

    emit("ARTIFACT", {"s3_uri": s3_uri, "hf_repo": hf_repo})
    emit("PROGRESS", {"stage": "done", "percent": 100})
    emit("DONE", {"scheme": scheme})


def _redirect_tmp(base: str) -> None:
    """Move TMPDIR / HF cache off the small local /tmp onto a roomy dir (default
    /share) — a big model + its quantized copy easily overflow a pod's /tmp."""
    try:
        os.makedirs(base, exist_ok=True)
        for var in ("TMPDIR", "HF_HOME", "XDG_CACHE_HOME"):
            if not os.environ.get(var):
                d = os.path.join(base, {"TMPDIR": "tmp", "HF_HOME": "huggingface", "XDG_CACHE_HOME": "cache"}[var])
                os.makedirs(d, exist_ok=True)
                os.environ[var] = d
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to JSON config")
    ap.add_argument("--deps-only", action="store_true", help="install dependencies then exit")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    _redirect_tmp(cfg.get("work_dir") or "/share")
    try:
        _ensure_venv(cfg)
        if a.deps_only:
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
