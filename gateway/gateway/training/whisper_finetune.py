#!/usr/bin/env python3
"""Standalone Whisper finetuning script — shipped to a RunPod pod / VM by the
gateway's Autotrain runner and executed over SSH. It has NO gateway imports;
everything it needs arrives in a single JSON config file (path passed via
--config). It installs its own deps, resolves the dataset (S3 or HuggingFace),
finetunes a Whisper model with HF Seq2SeqTrainer, evaluates WER + CER every
epoch, early-stops on patience, then uploads the best model + metrics to S3
(and optionally pushes to the HF Hub).

Contract with the gateway (parsed from stdout):
  @@METRIC {json}   one per epoch: {epoch, wer, cer, eval_loss, train_loss}
  @@ARTIFACT {json} after upload: {s3_uri, hf_repo?}
  @@DONE {json}     final: {best:{epoch,wer,cer}, epochs:int, stopped_early:bool}
  @@ERROR {json}    fatal: {message}
Every other line is free-form progress and streamed to the run's log.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback


def log(msg: str) -> None:
    print(msg, flush=True)


def emit(tag: str, obj: dict) -> None:
    """Structured line the gateway parses out of the stream."""
    print(f"@@{tag} {json.dumps(obj)}", flush=True)


# --------------------------------------------------------------------------
# Dependency bootstrap — runs before the heavy imports so a fresh pod works.
# --------------------------------------------------------------------------
def ensure_deps() -> None:
    # datasets>=4.0 decodes the Audio feature via torchcodec (needs ffmpeg + a
    # torch-matched wheel) and dies with "please install 'torchcodec'" on a box
    # without it. Pin to the 3.x line so the soundfile/librosa decoder is used —
    # that's what we install below and it yields the {array,sampling_rate} dicts
    # the trainer expects. This must run BEFORE any `import datasets`: a pip
    # downgrade can't swap a module that's already imported in this process, so
    # we detect the installed version via metadata (no import) and downgrade.
    import importlib.metadata as _md
    try:
        _ds = _md.version("datasets")
        if int(_ds.split(".")[0]) >= 4:
            log(f"[deps] datasets {_ds} requires torchcodec for audio; pinning <4.0 …")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "datasets>=2.20,<4.0"]
            )
    except Exception:
        pass

    try:
        import transformers  # noqa: F401
        import datasets  # noqa: F401
        import evaluate  # noqa: F401
        import jiwer  # noqa: F401
        import soundfile  # noqa: F401
        import boto3  # noqa: F401
        log("[deps] already present")
        return
    except Exception:
        pass
    log("[deps] installing transformers/datasets/evaluate/jiwer/audio/boto3 …")
    pkgs = [
        "transformers>=4.44",
        "datasets>=2.20,<4.0",
        "evaluate",
        "jiwer",
        "accelerate>=0.30",
        "soundfile",
        "librosa",
        "boto3",
        "huggingface_hub",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *pkgs]
    )
    log("[deps] done")


def ensure_tracker_deps(report_to: list[str]) -> None:
    """Install experiment-tracker clients on demand (only what's enabled)."""
    want: list[str] = []
    if "wandb" in report_to:
        try:
            import wandb  # noqa: F401
        except Exception:
            want.append("wandb")
    if "mlflow" in report_to:
        try:
            import mlflow  # noqa: F401
        except Exception:
            want.append("mlflow")
    if want:
        log(f"[deps] installing trackers: {', '.join(want)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *want]
        )


# --------------------------------------------------------------------------
# Dataset resolution → list[{"audio": <local path>, "text": str, "split": str?}]
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


def _read_metadata_rows(ds: dict) -> list[dict]:
    """Read the dataset's metadata file (csv/json/jsonl) from S3 into dicts."""
    import csv
    import io

    cli = _s3_client(ds)
    body = cli.get_object(Bucket=ds["bucket"], Key=ds["metadata_key"])["Body"].read()
    text = body.decode("utf-8", errors="replace")
    fmt = (ds.get("format") or "").lower()
    if not fmt:
        fmt = "jsonl" if ds["metadata_key"].endswith(".jsonl") else (
            "json" if ds["metadata_key"].endswith(".json") else "csv"
        )
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(text)))
    if fmt == "jsonl":
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("data", data.get("rows", []))


def _download_audio_s3(ds: dict, ref: str, dest_dir: str) -> str | None:
    """Resolve a metadata audio reference to a local file. `ref` is either a
    full http(s) presigned URL (post-transform datasets) or a key relative to
    the storage prefix + audio_prefix."""
    import urllib.request

    fname = os.path.basename(ref.split("?")[0]) or "audio.wav"
    local = os.path.join(dest_dir, fname)
    if os.path.exists(local):
        return local
    try:
        if ref.startswith("http://") or ref.startswith("https://"):
            urllib.request.urlretrieve(ref, local)
        else:
            cli = _s3_client(ds)
            prefix = (ds.get("audio_prefix") or "").strip("/")
            key = "/".join(p for p in [prefix, ref.lstrip("/")] if p)
            cli.download_file(ds["bucket"], key, local)
        return local
    except Exception as e:  # noqa: BLE001
        log(f"[data] skip audio {ref!r}: {e}")
        return None


def load_pairs(ds: dict, work: str) -> list[dict]:
    """Return [{audio, text, split?}] for a dataset spec (S3 or HF)."""
    kind = ds.get("kind")
    audio_field = ds.get("audio_field") or "audio"
    text_field = ds.get("transcription_field") or "transcription"

    if kind == "hf":
        from datasets import load_dataset

        token = ds.get("hf_token") or None
        repo = ds["hf_repo"]
        log(f"[data] loading HF dataset {repo}")
        dd = load_dataset(repo, token=token)
        out: list[dict] = []
        for split_name, split in dd.items():
            tf = (ds.get("split_fields") or {}).get(split_name, text_field)
            for row in split:
                out.append({"audio": row[audio_field], "text": row.get(tf, ""),
                            "split": split_name})
        return out

    # S3 / upload metadata
    audio_dir = os.path.join(work, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    rows = _read_metadata_rows(ds)
    log(f"[data] {len(rows)} metadata rows from s3://{ds['bucket']}/{ds['metadata_key']}")
    pairs: list[dict] = []
    for r in rows:
        ref = r.get(audio_field)
        text = r.get(text_field)
        if not ref or text is None:
            continue
        local = _download_audio_s3(ds, str(ref), audio_dir)
        if local is None:
            continue
        item = {"audio": local, "text": str(text)}
        if r.get("split"):
            item["split"] = str(r["split"])
        pairs.append(item)
    return pairs


# --------------------------------------------------------------------------
# Train / eval split — `split` column wins, else seeded hold-out.
# --------------------------------------------------------------------------
EVAL_SPLITS = {"test", "validation", "valid", "eval", "dev"}


def split_pairs(pairs: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    labelled = [p for p in pairs if p.get("split")]
    if labelled and any(p["split"].lower() in EVAL_SPLITS for p in labelled):
        train = [p for p in pairs if (p.get("split") or "train").lower() not in EVAL_SPLITS]
        ev = [p for p in pairs if (p.get("split") or "").lower() in EVAL_SPLITS]
        log(f"[split] using dataset split column: {len(train)} train / {len(ev)} eval")
        return train, ev
    import random

    pct = float(cfg.get("eval_split_pct", 10)) / 100.0
    rng = random.Random(int(cfg.get("split_seed", 42)))
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_eval = max(1, int(len(pairs) * pct)) if len(pairs) > 1 else 0
    eval_idx = set(idx[:n_eval])
    train = [pairs[i] for i in idx if i not in eval_idx]
    ev = [pairs[i] for i in idx if i in eval_idx]
    log(f"[split] seeded hold-out {cfg.get('eval_split_pct', 10)}%: "
        f"{len(train)} train / {len(ev)} eval (seed={cfg.get('split_seed', 42)})")
    return train, ev


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def run(cfg: dict) -> None:
    import numpy as np
    import torch
    from datasets import Audio, Dataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
    import evaluate as hf_evaluate

    work = tempfile.mkdtemp(prefix="autotrain-")
    base_model = cfg["base_model"]
    language = cfg.get("language") or None
    task = cfg.get("task") or "transcribe"
    metric_name = (cfg.get("eval_metric") or "wer").lower()

    log(f"[train] base_model={base_model} metric={metric_name} "
        f"max_epochs={cfg['max_epochs']} patience={cfg.get('patience', 0)}")

    pairs = load_pairs(cfg["dataset"], work)
    if cfg.get("test_dataset"):
        train_pairs = pairs
        eval_pairs = load_pairs(cfg["test_dataset"], work)
        log(f"[split] separate test dataset: {len(train_pairs)} train / {len(eval_pairs)} eval")
    else:
        train_pairs, eval_pairs = split_pairs(pairs, cfg)
    if not train_pairs or not eval_pairs:
        raise RuntimeError(
            f"need both train and eval examples (got {len(train_pairs)}/{len(eval_pairs)})"
        )

    processor = WhisperProcessor.from_pretrained(base_model, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model.generation_config.language = language
    model.generation_config.task = task
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    def _to_ds(rows: list[dict]) -> "Dataset":
        d = Dataset.from_dict({
            "audio": [r["audio"] for r in rows],
            "text": [r["text"] for r in rows],
        })
        return d.cast_column("audio", Audio(sampling_rate=16000))

    train_ds, eval_ds = _to_ds(train_pairs), _to_ds(eval_pairs)

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(prepare, remove_columns=eval_ds.column_names)

    class Collator:
        def __call__(self, features):
            inp = [{"input_features": f["input_features"]} for f in features]
            batch = processor.feature_extractor.pad(inp, return_tensors="pt")
            labels = processor.tokenizer.pad(
                [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
            )
            lab = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
            if (lab[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
                lab = lab[:, 1:]
            batch["labels"] = lab
            return batch

    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    # ---- experiment tracking (W&B / MLflow via HF Trainer's report_to) ----
    tracking = cfg.get("tracking") or {}
    report_to = list(tracking.get("report_to") or [])
    for k, v in (tracking.get("env") or {}).items():
        if v not in (None, ""):
            os.environ[k] = str(v)
    if "wandb" in report_to and not os.environ.get("WANDB_API_KEY"):
        log("[track] W&B requested but no WANDB_API_KEY — disabling W&B")
        report_to = [r for r in report_to if r != "wandb"]
    if "mlflow" in report_to and not os.environ.get("MLFLOW_TRACKING_URI"):
        log("[track] MLflow requested but no MLFLOW_TRACKING_URI — disabling MLflow")
        report_to = [r for r in report_to if r != "mlflow"]
    if report_to:
        log(f"[track] reporting metrics to: {', '.join(report_to)}")

    out_dir = os.path.join(work, "out")
    precision = (cfg.get("precision") or "fp16").lower()
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=int(cfg.get("batch_size", 8)),
        per_device_eval_batch_size=int(cfg.get("batch_size", 8)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 1)),
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        num_train_epochs=float(cfg["max_epochs"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=int(cfg.get("generation_max_length", 225)),
        fp16=(precision == "fp16"),
        bf16=(precision == "bf16"),
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=False,
        save_total_limit=1,
        logging_steps=max(1, int(cfg.get("logging_steps", 10))),
        report_to=report_to,
        run_name=cfg.get("run_name") or None,
    )

    class MetricEmitter(TrainerCallback):
        """Stream per-epoch results so the gateway can chart WER/CER live."""
        def on_log(self, a, state, control, logs=None, **kw):
            # Training-step logs carry "loss" (every logging_steps); eval logs
            # carry "eval_loss". Emit a STEP point per training log so the
            # platform can draw a live loss curve.
            logs = logs or {}
            if "loss" in logs:
                emit("STEP", {
                    "step": int(state.global_step),
                    "loss": logs.get("loss"),
                    "lr": logs.get("learning_rate"),
                    "epoch": round(float(logs.get("epoch") or state.epoch or 0), 3),
                })

        def on_evaluate(self, a, state, control, metrics=None, **kw):
            m = metrics or {}
            train_loss = None
            for h in reversed(state.log_history):
                if "loss" in h:
                    train_loss = h["loss"]
                    break
            emit("METRIC", {
                "epoch": round(float(state.epoch or 0), 3),
                "wer": m.get("eval_wer"),
                "cer": m.get("eval_cer"),
                "eval_loss": m.get("eval_loss"),
                "train_loss": train_loss,
            })

    callbacks: list = [MetricEmitter()]
    patience = int(cfg.get("patience", 0) or 0)
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=Collator(),
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=callbacks,
    )

    result = trainer.train()
    epochs_ran = int(round(result.metrics.get("epoch", 0)))
    stopped_early = epochs_ran < int(cfg["max_epochs"])

    best_dir = os.path.join(work, "best-model")
    trainer.save_model(best_dir)
    processor.save_pretrained(best_dir)
    final = trainer.evaluate()
    best = {
        "epoch": epochs_ran,
        "wer": final.get("eval_wer"),
        "cer": final.get("eval_cer"),
        "eval_loss": final.get("eval_loss"),
    }

    # ---- upload artifacts to S3 (best model + metrics) ----
    art = cfg.get("artifacts") or {}
    s3_uri = None
    if art.get("bucket"):
        cli = _s3_client(art)
        base_key = art["prefix"].rstrip("/") + "/best-model"
        for root, _dirs, files in os.walk(best_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, best_dir)
                cli.upload_file(fp, art["bucket"], f"{base_key}/{rel}")
        metrics_key = art["prefix"].rstrip("/") + "/metrics.json"
        cli.put_object(
            Bucket=art["bucket"], Key=metrics_key,
            Body=json.dumps({"best": best, "epochs": epochs_ran}).encode(),
        )
        s3_uri = f"s3://{art['bucket']}/{base_key}/"
        log(f"[upload] best model → {s3_uri}")

    # ---- optional HF push ----
    hf_repo = None
    if cfg.get("hf_push_repo") and cfg.get("hf_token"):
        try:
            model.push_to_hub(cfg["hf_push_repo"], token=cfg["hf_token"])
            processor.push_to_hub(cfg["hf_push_repo"], token=cfg["hf_token"])
            hf_repo = cfg["hf_push_repo"]
            log(f"[upload] pushed best model → https://huggingface.co/{hf_repo}")
        except Exception as e:  # noqa: BLE001
            log(f"[upload] HF push failed: {e}")

    emit("ARTIFACT", {"s3_uri": s3_uri, "hf_repo": hf_repo})
    emit("DONE", {"best": best, "epochs": epochs_ran, "stopped_early": stopped_early})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to JSON config")
    ap.add_argument("--deps-only", action="store_true",
                    help="install dependencies then exit (used by the sweep orchestrator)")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    try:
        ensure_deps()
        ensure_tracker_deps((cfg.get("tracking") or {}).get("report_to") or [])
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
