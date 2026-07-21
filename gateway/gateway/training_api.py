"""Autotrain feature — Whisper finetuning, SSH-orchestrated by the gateway.

Mirrors the benchmark feature (gateway/gateway/bench.py) but, because there is
no external training CLI, the gateway owns the pod lifecycle itself: provision a
RunPod pod (or use a registered VM), SFTP a self-contained trainer
(`training/whisper_finetune.py`) onto it, run it over SSH, stream stdout to a
Redis list (capped) for live SSE replay, parse `@@METRIC/@@DONE` lines into
`result_json`, then upload logs to S3 and tear the pod down.

Subprocess-free: the run lives as an asyncio task in the gateway process. If the
gateway dies mid-run the run is orphaned (the pod is alive on RunPod but nobody
is collecting); `cleanup_orphaned_running()` marks such rows failed on startup.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import math
import os
import re
import shlex
import subprocess
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, field_validator
from .pathsafe import validate_path_field
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    cast,
    func,
    or_,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from . import crypto
from .auth import require_section
from .db import Base, Dataset, Provider, Storage, User, get_session, session_factory
# Reuse the benchmark S3 plumbing (target-parameterised) and compute's RunPod helpers.
from .bench import (
    S3Target,
    _s3_client,
    s3_put_text,
    s3_get_bytes,
    s3_put_bytes,
    s3_list,
    s3_presign_get,
)
from . import compute

logger = logging.getLogger("gateway.training")

LOG_LIST_CAP = 5000
LOG_LIST_TTL_S = 12_960_000  # ~5 months
POLL_INTERVAL_S = 5.0
POLL_TIMEOUT_S = 900.0
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
# LLM cloud try-it serves via vLLM ≥0.23, which needs a CUDA-13 host (a cu12 image →
# EngineCore "driver too old" crash — see gateway/gateway/CLAUDE.md). Provision LLM
# try-it/merge pods on a cu1300 image so allowedCudaVersions pins a ≥580-driver host.
LLM_VLLM_IMAGE = "runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404"
# Ensure `uv` exists + is on PATH before the trainer's --deps-only runs, so its
# venv is built with `uv venv` / `uv pip install` (fast, hash-constraint-free)
# rather than the slow `python -m venv` + pip fallback. Mirrors the benchmark VM
# bootstrap (pyremote_shim). Prepended AFTER the user env so $HOME is set first;
# non-fatal (|| true) → if curl/network is unavailable the trainer still falls
# back to venv+pip. Applies to ASR, TTS train, and NeuCodec (pack-only) runs.
_UV_BOOTSTRAP = (
    'export PATH="$HOME/.local/bin:$PATH"\n'
    'if ! command -v uv >/dev/null 2>&1; then '
    'curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true; fi\n'
    'export PATH="$HOME/.local/bin:$PATH"\n'
)
# Valid training-audio augmentation techniques (mirror whisper_finetune._AUG_FUNCS).
_AUG_TECHNIQUES = {"telephone", "noise", "dropout", "gain", "pitch", "speed", "reverb", "bandpass"}
# TTS audio-eval methods (run on the test set): char-error-rate, UTMOSv2 MOS,
# TitaNet speaker similarity. The eval runner is a follow-up; the selection +
# config plumbing land here so runs record what to evaluate.
_TTS_EVAL_METHODS = {"cer", "mos", "similarity"}
# LLM LoRA target linear projections the user may select (gemma4 trainer). The
# attention q/k/v/o are the default; gate/up/down are the MLP/dense layers. gemma4.py
# warns for any target that wraps no nn.Linear on the loaded arch.
_LORA_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                        "in_proj", "out_proj"}  # in/out_proj = Nemotron-H Mamba2 mixer projections
_LORA_TARGET_DEFAULT = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Strong refs to in-flight runner tasks (else asyncio may GC them) + per-run
# teardown state (RunPod id + api key) so terminate can delete the pod.
_active_runners: dict[str, asyncio.Task] = {}
_RUN_STATE: dict[str, dict] = {}
# On-demand label exports in flight (the retry endpoint), keyed by run id → its
# background task. Re-clicking "Export to Label" cancels a stuck attempt and starts
# fresh (the cancelled task's project-creation step never runs, so no duplicate).
_active_label_exports: dict[str, asyncio.Task] = {}
# On-demand "Export to Hugging Face" pushes in flight (per run id → task), so a
# re-click supersedes a stuck attempt rather than running two pushes at once.
_active_hf_exports: dict[str, asyncio.Task] = {}
# Cached SSH client per run for the live GPU-util poll (reused across polls).
_GPU_SSH: dict[str, Any] = {}


# ---------- DB model (mirror Benchmark) ---------------------------------


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # train-<hex8>
    name: Mapped[str] = mapped_column(String(128))
    dataset_id: Mapped[str] = mapped_column(String(64), index=True)
    test_dataset_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    base_model: Mapped[str] = mapped_column(String(255))
    # "asr" (Whisper finetune) | "tts" (Qwen3 + NeuCodec finetune).
    task_type: Mapped[str] = mapped_column(String(16), default="asr", server_default="asr", nullable=False)
    # Full training config (hyperparams + split settings). Credentials are NEVER
    # stored here — they're resolved + injected into the pod at run time.
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    s3_prefix: Mapped[str] = mapped_column(String(255))
    exit_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_text: Mapped[Optional[str]] = mapped_column(String(4096), nullable=True)
    # {"epochs": [{epoch,wer,cer,eval_loss,train_loss}], "best": {...},
    #  "artifact": {s3_uri,hf_repo}, "stopped_early": bool}
    result_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    cost_per_hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    runpod_pod_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    provider_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    storage_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    gpu_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    gpu_count: Mapped[int] = mapped_column(Integer, default=1, server_default="1", nullable=False)
    visible_devices: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)


# ---------- S3 + filesystem helpers -------------------------------------


def _gen_id() -> str:
    import uuid
    return "train-" + uuid.uuid4().hex[:8]


def _work_dir(run_id: str) -> Path:
    d = Path(os.environ.get("AUTOTRAIN_WORK_DIR", "/tmp/sgpu-train")) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _full_log_path(run_id: str) -> Path:
    return _work_dir(run_id) / "_full.log"


def _trainer_script_path() -> Path:
    return Path(__file__).parent / "training" / "whisper_finetune.py"


def _s3_creds_from_storage(row: Optional[Storage]) -> dict:
    """{bucket, region, endpoint, access_key, secret_key, prefix} for a kind=s3
    Storage row, decrypting credentials. Falls back to env."""
    if row is None:
        return {
            "bucket": os.environ.get("BENCHMARK_S3_BUCKET", "").strip(),
            "region": os.environ.get("AWS_REGION", "ap-southeast-5"),
            "endpoint": None,
            "access_key": os.environ.get("AWS_ACCESS_KEY_ID") or None,
            "secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
            "prefix": "",
        }
    cfg = row.config or {}
    enc = cfg.get("credentials_enc")
    if enc:
        creds = json.loads(crypto.decrypt(enc))
        access_key = creds.get("accessKeyId")
        secret_key = creds.get("secretAccessKey")
    else:
        access_key = os.environ.get("AWS_ACCESS_KEY_ID") or None
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or None
    return {
        "bucket": (cfg.get("bucket") or "").strip(),
        "region": (cfg.get("region") or os.environ.get("AWS_REGION", "ap-southeast-5")),
        "endpoint": (cfg.get("endpoint") or None),
        "access_key": access_key,
        "secret_key": secret_key,
        "prefix": (cfg.get("prefix") or "").strip().strip("/"),
    }


async def _training_s3_target(storage_id: Optional[str]) -> S3Target:
    """Resolve the S3 destination for a run's logs/artifacts. prefix_root ends
    with 'training-runs/'."""
    creds = {
        "bucket": os.environ.get("BENCHMARK_S3_BUCKET", "").strip(),
        "region": os.environ.get("AWS_REGION", "ap-southeast-5"),
        "endpoint": None,
        "access_key": os.environ.get("AWS_ACCESS_KEY_ID") or None,
        "secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
        "prefix": "",
    }
    if storage_id:
        async with session_factory()() as s:
            row = await s.get(Storage, storage_id)
        creds = _s3_creds_from_storage(row)
    prefix = creds["prefix"]
    prefix_root = f"{prefix}/training-runs/" if prefix else "training-runs/"
    return S3Target(
        bucket=creds["bucket"], region=creds["region"], endpoint=creds["endpoint"],
        access_key=creds["access_key"], secret_key=creds["secret_key"], prefix_root=prefix_root,
    )


async def _resolve_global_env() -> dict:
    """Admin-managed org-wide env/secrets (the Secrets page) as {key: value}.
    Same source benchmarks use for HF_TOKEN etc."""
    try:
        from .global_env_api import load_global_env
        async with session_factory()() as s:
            return await load_global_env(s)
    except Exception:
        return {}


def _llm_arch(model_id: Optional[str]) -> str:
    """gemma | minimax | mistral | qwen | nemotron from an LLM base model id (mirrors
    llm_finetune.detect_arch + llm_pack.detect_arch). Drives the per-arch venv +
    trainer choice. Unknown → gemma (the default LLM base) so the venv path is still
    well-formed; the trainer raises a clear error on a truly unsupported model."""
    n = (model_id or "").lower()
    if "minimax" in n:
        return "minimax"
    if "mistral" in n:
        return "mistral"
    if "qwen" in n:
        return "qwen"  # Qwen3.5/3.6 dense + MoE (auto-detected by the trainer from config)
    if "nemotron" in n:
        return "nemotron"  # NVIDIA Nemotron-H hybrid Mamba2/attention MoE (one-doc-per-bin pack)
    return "gemma"


# Per-arch LLM venv dir on the shared box (/share). Bump an arch's entry here to roll its
# venv forward WITHOUT touching the arch string (which drives trainer dispatch). gemma →
# gemma-v2 (2026-07-12): forces a FRESH venv on the next gemma run so the deps installer
# re-clones the FA4 cute fork and picks up the head_dim-512 forward retile (m64n80→m64n64,
# +15-20%) — a reused venv keeps the old kernel (llm_finetune's "already imports → skip"
# fast path). ⚠ KEEP IN SYNC with training/llm_finetune.py::_LLM_VENV_VERSION (box side).
_LLM_VENV_VERSION = {"gemma": "gemma-v2"}


def _llm_venv(arch: str) -> str:
    """Canonical /share venv dir for an LLM arch (gateway side; single source of truth).
    Applies the per-arch version override so training AND every post-train op (all of which
    re-derive from arch — venv_path is NOT persisted to config_json) resolve to the SAME dir."""
    return f"/share/autotrain-llm-{_LLM_VENV_VERSION.get(arch, arch)}"


def _tts_arch(model_id: Optional[str]) -> str:
    """omnivoice | qwen3 from a TTS base model id. OmniVoice (k2-fsa/OmniVoice or
    a Scicom omnivoice repo) uses a different trainer + codec + packed format than
    the default Qwen3+NeuCodec path. Drives the per-arch venv, trainer ship, and
    packed-dataset kind. Unknown → qwen3 (the historical default)."""
    n = (model_id or "").lower()
    if "omnivoice" in n:
        return "omnivoice"
    return "qwen3"


def _train_venv_default(task_type: Optional[str], base_model: Optional[str]) -> str:
    """The shared box venv a run's trainer creates/reuses, keyed by task_type + arch.
    Single source of truth for run_training AND every box-side op that runs AFTER a
    run trained (HF export, …). The resolved venv is only shipped to the box in the
    remote config — it is NOT written back to the run's config_json — so anything that
    later re-derives it MUST use this same mapping. A hardcoded /share/autotrain-tts
    (as the export path used to) silently misses OmniVoice's own /share/autotrain-omnivoice
    and ASR's /share/autotrain-whisper → "venv python not found" on a run that trained fine."""
    tt = (task_type or "asr").lower()
    if tt == "llm":
        # gemma vs minimax need different `kernels` pins (transformers 5.5.0 import
        # crashes outside each range) → SEPARATE venvs per arch.
        return _llm_venv(_llm_arch(base_model))
    if tt == "tts":
        # OmniVoice (torch 2.8/cu128 + its own repo) gets a SEPARATE venv from the
        # Qwen3+NeuCodec TTS stack (cu13/2.9). Selected by base model.
        return ("/share/autotrain-omnivoice"
                if _tts_arch(base_model) == "omnivoice" else "/share/autotrain-tts")
    return "/share/autotrain-whisper"


async def _resolve_dataset_spec(dataset_id: str, hf_token_fallback: Optional[str] = None) -> dict:
    """Turn a Dataset row into the trainer's dataset spec, with creds inlined."""
    label_token = None
    async with session_factory()() as s:
        ds = await s.get(Dataset, dataset_id)
        if ds is None:
            raise RuntimeError(f"dataset {dataset_id} not found")
        storage = await s.get(Storage, ds.storage_id) if ds.storage_id else None
        if ds.kind == "label":
            from .datasets_api import _label_token
            label_token = await _label_token(ds, s)

    if ds.kind == "label":
        # Resolve the labeling-platform export server-side (the gateway has the
        # API helpers) into {audio_url, id, text, split} rows the trainer
        # downloads. The token + base URL are inlined so the GPU box can fetch
        # platform-hosted clips (those need the bearer token); presigned audio_urls
        # download directly.
        from .datasets_api import _label_export_rows
        rows, _total = await asyncio.to_thread(
            _label_export_rows, ds.label_base_url, ds.label_project_id, label_token,
            ds.label_status or "approved", 10**9, 0,
        )
        tf = ds.transcription_field or "transcription"
        spec_rows = [
            {"audio_url": str(r.get("audio_url") or ""), "id": r.get("id"),
             "text": "" if r.get(tf) is None else str(r.get(tf)), "split": str(r.get("split") or "train")}
            for r in rows
        ]
        return {
            "kind": "label",
            "label_base_url": (ds.label_base_url or "").rstrip("/"),
            "label_project_id": ds.label_project_id,
            "label_token": label_token,
            "rows": spec_rows,
        }

    if ds.kind in ("hf", "llm"):
        # kind=llm is an HF chat dataset (hf_repo + messages_field) — same spec as
        # kind=hf, just with the chat column carried along. Autotrain never resolves
        # kind=llm here (it trains on llm_packed), but quantization calibrates on it.
        hf_token = None
        if storage is not None and storage.kind == "huggingface":
            enc = (storage.config or {}).get("credentials_enc")
            if enc:
                hf_token = json.loads(crypto.decrypt(enc)).get("token")
        hf_token = hf_token or os.environ.get("HF_TOKEN") or None
        return {
            "kind": "hf",
            "hf_repo": ds.hf_repo,
            "hf_token": hf_token,
            "audio_field": ds.audio_field,
            "transcription_field": ds.transcription_field,
            "messages_field": ds.messages_field,
            "split_fields": ds.split_fields or {},
        }

    creds = _s3_creds_from_storage(storage)
    if ds.kind == "tts_packed":
        # Pre-packed (NeuCodec + multipack) ChiniDataset: s3_metadata_uri is the
        # s3:// prefix of the shards. The TTS trainer streams it (skips convert+pack).
        return {
            "kind": "tts_packed",
            "packed_uri": ds.s3_metadata_uri,
            "region": creds["region"], "endpoint": creds["endpoint"],
            "access_key": creds["access_key"], "secret_key": creds["secret_key"],
        }
    if ds.kind == "omnivoice_packed":
        # Pre-packed OmniVoice tokens (Higgs WebDataset shards + manifests):
        # s3_metadata_uri is the shards prefix. omnivoice_finetune downloads it
        # → ./tokens and trains directly (skips manifests + Higgs tokenization).
        return {
            "kind": "omnivoice_packed",
            "packed_uri": ds.s3_metadata_uri,
            # The Higgs audio codec recorded at pack time — the orchestrator bundles
            # THIS (not the LM base model) into the clean checkpoint's audio_tokenizer/.
            "higgs_tokenizer": ((ds.split_fields or {}).get("_omnivoice_pack") or {}).get("tokenizer"),
            "region": creds["region"], "endpoint": creds["endpoint"],
            "access_key": creds["access_key"], "secret_key": creds["secret_key"],
        }
    if ds.kind in ("llm_packed", "llm_dpo_packed"):
        # Pre-packed (chat multipack) ChiniDataset: s3_metadata_uri is the s3://
        # prefix of the shards (input_ids/labels/position_ids/attention_mask). The
        # LLM trainer downloads it → ./packed_data and runs gemma4.py directly.
        # llm_dpo_packed = same columns, whole preference pairs per bin (chosen-first
        # layout + pre-aligned targets) for training_type=dpo.
        pack = (ds.split_fields or {}).get("_llm_pack") or {}
        return {
            "kind": ds.kind,
            "packed_uri": ds.s3_metadata_uri,
            "tokenizer": pack.get("tokenizer"),
            "sequence_length": pack.get("sequence_length"),
            "region": creds["region"], "endpoint": creds["endpoint"],
            "access_key": creds["access_key"], "secret_key": creds["secret_key"],
        }
    if ds.kind == "s3" and ds.s3_metadata_uri:
        # s3_metadata_uri is either a full "s3://bucket/key" URI or a key
        # relative to the storage's bucket (used as-is). Mirror datasets_api's
        # preview resolution so the trainer reads the exact object the UI does
        # — the old code blindly chopped 5 chars assuming an "s3://" prefix,
        # turning "datasets/…" into bucket="sets" → GetObject AccessDenied.
        u = urlparse(ds.s3_metadata_uri)
        if u.scheme == "s3":
            bucket = u.netloc or creds["bucket"]
            metadata_key = u.path.lstrip("/")
        else:
            bucket = creds["bucket"]
            metadata_key = ds.s3_metadata_uri.lstrip("/")
    else:  # upload
        bucket = creds["bucket"]
        parts = [creds["prefix"], f"datasets/{ds.id}", ds.metadata_filename or "metadata.csv"]
        metadata_key = "/".join(p.strip("/") for p in parts if p)
    audio_prefix = "/".join(p.strip("/") for p in [creds["prefix"], ds.audio_prefix or ""] if p)
    return {
        "kind": "s3",
        "bucket": bucket,
        "region": creds["region"],
        "endpoint": creds["endpoint"],
        "access_key": creds["access_key"],
        "secret_key": creds["secret_key"],
        "metadata_key": metadata_key,
        "audio_prefix": audio_prefix,
        "format": ds.format,
        "audio_field": ds.audio_field,
        "transcription_field": ds.transcription_field,
        # Rows manually un-ticked in the row browser → skipped by the reader.
        "excluded_rows": sorted(int(x) for x in (ds.excluded_rows or [])),
    }


# ---------- log streaming ------------------------------------------------


async def _push_redis(redis, run_id: str, line: str) -> None:
    """Append to the capped Redis list only (for live SSE replay)."""
    key = f"train:logs:{run_id}"
    try:
        await redis.rpush(key, line)
        await redis.ltrim(key, -LOG_LIST_CAP, -1)
    except Exception:
        pass


async def _push_log(redis, run_id: str, line: str) -> None:
    """Gateway-side line: Redis (live) + on-disk (canonical → S3 logs.txt)."""
    await _push_redis(redis, run_id, line)
    try:
        with open(_full_log_path(run_id), "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# HuggingFace/tqdm download progress lines, e.g.
#   "model-00001-of-00002.safetensors:  26%|██▌       | 13.1G/49.9G [09:00<12:14, 50.2MB/s]"
# Over a pipe (no TTY) tqdm can't rewrite a line in place, so every refresh is captured as a
# NEW line — one model pull floods the log with tens of thousands of them. The /logs/trim
# endpoint strips them. Signature = "<pct>%|<bar>| <n>/<total> [" — tqdm-specific enough that
# it won't hit real trainer/gateway log lines.
_PROGRESS_RE = re.compile(r"\d{1,3}%\|.*?\|\s*\S+/\S+\s*\[")


def _is_progress_line(line: str) -> bool:
    # Never drop a line carrying a chart payload — a tqdm bar can be \r-prefixed onto the same
    # captured line as an @@STEP/@@METRIC (see stream_training_logs' \r note).
    if "@@" in line:
        return False
    # A single stored entry may pack several \r-separated refreshes; drop it if ANY is a bar.
    return any(_PROGRESS_RE.search(seg) for seg in line.split("\r"))


# ---------- SSH (paramiko) ----------------------------------------------


def _gen_ssh_key(work: Path) -> tuple[str, str]:
    """Mint an ephemeral keypair for pod access. Returns (priv_path, pub_text)."""
    priv = work / "id_ed25519"
    if not priv.exists():
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", str(priv)],
            check=True,
        )
    pub = (work / "id_ed25519.pub").read_text().strip()
    return str(priv), pub


def _ssh_connect(host: str, port: int, user: str, key_filename: str):
    import paramiko

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    last = None
    for _ in range(30):  # pod sshd can lag the API "ready" signal
        try:
            cli.connect(
                hostname=host, port=port, username=user,
                key_filename=key_filename, timeout=20, banner_timeout=30, auth_timeout=30,
            )
            # SSH-level keepalive (every 30s). Without it, a long-idle exec channel —
            # e.g. the multi-GB HF `upload_folder` streams VM→huggingface.co directly,
            # so NOTHING flows over this control channel for many minutes — gets its TCP
            # connection silently dropped by a NAT/firewall on the path. paramiko's
            # blocking chan.recv() in _ssh_run_stream then never sees EOF and hangs
            # FOREVER: the VM-side push finishes (model lands on HF) and prints @@HF, but
            # that line vanishes into the dead socket → the export task never resolves and
            # the run sticks on "uploading…" (real bug: train-96dc20e1). Keepalive both
            # keeps the NAT mapping warm AND lets paramiko tear down a genuinely dead peer
            # so recv() raises instead of blocking (→ the export is marked failed, not stuck).
            try:
                tr = cli.get_transport()
                if tr is not None:
                    tr.set_keepalive(30)
            except Exception:  # noqa: BLE001 — keepalive is best-effort; never fail a connect over it
                pass
            return cli
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5)
    raise RuntimeError(f"SSH connect failed after retries: {last}")


def _ssh_put(cli, local: str, remote: str) -> None:
    """Upload a local file via a base64-pipe over an exec channel — NOT SFTP.
    The managed GPU VMs we target (PAI DSW, TM) front sshd with a proxy that
    doesn't expose the SFTP subsystem, so `cli.open_sftp()` dies with
    "EOF during negotiation". exec is the only channel type these proxies allow;
    base64 keeps the payload to a single safe argv (no quoting/heredoc issues).
    Mirrors pyremote_shim's config upload. Files shipped here (trainer scripts,
    config json) are tens of KB — well under ARG_MAX for a printf argv."""
    import base64 as _b64
    import shlex as _shlex

    with open(local, "rb") as f:
        b64 = _b64.b64encode(f.read()).decode("ascii")
    write_cmd = (
        f"mkdir -p \"$(dirname {_shlex.quote(remote)})\" && "
        f"printf %s {_shlex.quote(b64)} | base64 -d > {_shlex.quote(remote)}"
    )
    chan = cli.get_transport().open_session()
    chan.set_combine_stderr(True)
    chan.exec_command(f"bash -c {_shlex.quote(write_cmd)}")
    out = b""
    while not chan.exit_status_ready() or chan.recv_ready():
        if chan.recv_ready():
            out += chan.recv(8192)
        else:
            time.sleep(0.05)
    rc = chan.recv_exit_status()
    if rc != 0:
        raise RuntimeError(
            f"remote write of {remote} failed (rc={rc}): {out.decode('utf-8', 'replace').strip()}"
        )


def _ssh_put_bytes(cli, data: bytes, remote: str) -> None:
    """Write arbitrary bytes to a remote file — NOT SFTP, NOT raw stdin. The
    proxied GPU SSH endpoints we target (PAI DSW, TM, RunPod-behind-a-proxy) only
    reliably pass a single safe argv — the same reason _ssh_put base64s its
    payload. Raw binary on an exec channel's *stdin* (the old `cat > file`) gets
    mangled / short-written on some of them: `cat` still exits 0, leaving a corrupt
    file, so a downstream `tar -xzf` died with rc=2 (the bug this fixes). Instead we
    base64 the payload and write it in argv-sized chunks (`printf %s … | base64 -d`,
    first truncates then appends) — no stdin, no ~128 KB single-argv cap — then
    verify the landed byte count so a short write is loud, not silent."""
    import base64 as _b64
    import shlex as _shlex

    b64 = _b64.b64encode(data).decode("ascii")
    rq = _shlex.quote(remote)
    # 60 KB of base64 per write keeps each argv well under Linux MAX_ARG_STRLEN
    # (128 KB per single argument), with margin for the rest of the command.
    CHUNK = 60_000
    chunks = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)] or [""]
    for idx, chunk in enumerate(chunks):
        inner = (
            (f"mkdir -p \"$(dirname {rq})\" && " if idx == 0 else "")
            + f"printf %s {_shlex.quote(chunk)} | base64 -d {'>' if idx == 0 else '>>'} {rq}"
        )
        rc, out = _ssh_exec_out(cli, f"bash -c {_shlex.quote(inner)}")
        if rc != 0:
            raise RuntimeError(
                f"remote write of {remote} failed (chunk {idx + 1}/{len(chunks)}, "
                f"rc={rc}): {out.strip()}"
            )
    # Verify the whole payload landed — a silent short write is exactly what made
    # the old raw-stdin path fail untar with rc=2.
    rc, out = _ssh_exec_out(cli, f"wc -c < {rq}")
    got = out.strip()
    if rc == 0 and got.isdigit() and int(got) != len(data):
        raise RuntimeError(
            f"remote write of {remote} truncated: {got} of {len(data)} bytes landed"
        )


def _ssh_put_dir_tar(cli, local_dir: str, remote_dir: str) -> None:
    """Tar a local directory, stream it over SSH, and untar it on the remote —
    ships a whole package tree (e.g. the vendored chinidataset) in one shot,
    skipping __pycache__. No SFTP / no per-file round-trips."""
    import io
    import shlex as _shlex
    import tarfile

    def _filter(ti: "tarfile.TarInfo"):
        base = os.path.basename(ti.name)
        if base == "__pycache__" or base.endswith((".pyc", ".pyo")):
            return None
        # Neutralize ownership: the local (macOS) uid/gid don't exist on the pod,
        # and GNU tar extracting as root tries to chown the files to them →
        # "Cannot change ownership … Invalid argument" → rc=2. Owned-by-root (0:0)
        # archive + --no-same-owner below = portable extraction anywhere.
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        return ti

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(local_dir, arcname=".", filter=_filter)
    # Unique per-call remote path — a fixed name races when two runs ship to the SAME
    # box concurrently (e.g. two finetunes on different GPUs of one VM), corrupting
    # each other's tarball ("remote write … truncated").
    import secrets as _secrets
    remote_tar = f"/tmp/_sgpu_ship_{_secrets.token_hex(6)}.tar.gz"
    _ssh_put_bytes(cli, buf.getvalue(), remote_tar)
    lines: list[str] = []
    rc = _ssh_run_stream(
        cli,
        # --no-same-owner: don't restore archived ownership (we run as root on the
        # pod; a chown to a nonexistent uid is a hard rc=2 even though the file
        # content extracts fine).
        f"mkdir -p {_shlex.quote(remote_dir)} && "
        f"tar --no-same-owner -xzf {_shlex.quote(remote_tar)} -C {_shlex.quote(remote_dir)} 2>&1; "
        f"_rc=$?; rm -f {_shlex.quote(remote_tar)}; exit $_rc",
        lines.append,
    )
    if rc != 0:
        detail = "; ".join(lines[-5:]) if lines else "(no output)"
        raise RuntimeError(f"failed to ship/untar {local_dir} → {remote_dir} (rc={rc}): {detail}")


def _render_env_exports(env: dict) -> str:
    """Render `export K=V` lines (+ `mkdir -p` for absolute-path values) for the
    user's OS env, to be prepended to the trainer command so it's exported on the
    remote before the process starts. Mirrors the bench/pyremote VM env handling:
    HOME / HF_HOME / XDG_CACHE_HOME / TRITON_CACHE_DIR / … must live in the
    process env (libs read them at import time), not the trainer config. Keys are
    re-validated as shell-safe names here; values are shell-quoted."""
    lines: list[str] = []
    for k, v in (env or {}).items():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(k)):
            continue
        vs = str(v)
        lines.append(f"export {k}={shlex.quote(vs)}")
        if vs.startswith("/"):
            lines.append(f"mkdir -p {shlex.quote(vs)}")
    return ("\n".join(lines) + "\n") if lines else ""


def _ssh_exec(cli, command: str) -> int:
    """Run a one-shot command (no streaming), return its exit status."""
    chan = cli.get_transport().open_session()
    chan.exec_command(command)
    return chan.recv_exit_status()


def _ssh_exec_out(cli, command: str) -> tuple[int, str]:
    """Run a one-shot command; return (exit_status, combined stdout+stderr).
    Drains the channel to EOF *before* reading the exit status — the safe pattern.
    (The recv_ready()/exit_status_ready() poll used elsewhere can stop before a
    large buffered reply is fully read; see _ssh_run_stream's docstring.) Lets
    callers surface a remote command's error text instead of just an opaque rc."""
    chan = cli.get_transport().open_session()
    chan.set_combine_stderr(True)
    chan.exec_command(command)
    buf = b""
    while True:
        chunk = chan.recv(65536)
        if not chunk:
            break
        buf += chunk
    return chan.recv_exit_status(), buf.decode("utf-8", "replace")


def _ssh_run_stream(cli, command: str, on_line) -> int:
    """Run `command`, calling on_line(str) per stdout/stderr line. Blocking —
    call via asyncio.to_thread. Returns the remote exit status.

    Drains the channel with a blocking recv() until EOF. The earlier
    recv_ready()/exit_status_ready() poll could break *before* a large buffered
    reply was fully drained — a race that truncated big single-line responses
    (e.g. a TTS wav_b64 ~160 KB), leaving the caller a partial, unparseable line
    while small replies (ASR text) slipped through in one recv. recv() returns
    b"" on channel EOF, so the whole reply is delivered before we stop."""
    chan = cli.get_transport().open_session()
    chan.set_combine_stderr(True)
    chan.exec_command(command)
    buf = b""
    while True:
        data = chan.recv(65536)
        if not data:
            break
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            on_line(line.decode("utf-8", "replace"))
    if buf:
        on_line(buf.decode("utf-8", "replace"))
    return chan.recv_exit_status()


# ---------- RunPod provisioning -----------------------------------------


async def _provision_pod(
    api_key: str, name: str, image: str, gpu_type: str, gpu_count: int,
    secure_cloud: bool, disk_gb: int, volume_gb: int, pub_key: str,
    data_center_id: Optional[str] = None,
) -> tuple[str, str, int, Optional[float]]:
    """Create a RunPod pod and poll until SSH lands. Returns
    (runpod_id, ip, port, cost_per_hr). Mirrors compute._provision_runpod."""
    body: dict[str, Any] = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [compute._map_gpu(gpu_type)],
        "gpuCount": max(1, int(gpu_count)),
        "cloudType": "SECURE" if secure_cloud else "COMMUNITY",
        "containerDiskInGb": int(disk_gb),
        "volumeInGb": int(volume_gb),
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": pub_key} if pub_key else {},
    }
    cuda_v = compute._extract_cuda_version(image or "")
    if cuda_v:
        body["allowedCudaVersions"] = [cuda_v]
    dc = compute._data_center_ids(data_center_id)
    if dc:
        body["dataCenterIds"] = dc

    async with compute._client(api_key=api_key) as cli:
        r = await cli.post("/pods", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"RunPod refused create: HTTP {r.status_code} {r.text}"[:500])
        data = r.json()
        runpod_id = data.get("id")
        if not runpod_id:
            raise RuntimeError(f"RunPod response missing id: {data}"[:500])
        cost = data.get("costPerHr") or data.get("cost_per_hr")
        try:
            cost_f = float(cost) if cost is not None else None
        except (TypeError, ValueError):
            cost_f = None

        # The pod now EXISTS and is billing. If anything below fails (SSH never lands,
        # an API hiccup, cancellation), the caller never receives `runpod_id` — the
        # `runpod_id, … = await _provision_pod(…)` assignment doesn't run when the call
        # raises — so it can't terminate it. Clean up our own pod here, then re-raise.
        try:
            deadline = time.time() + POLL_TIMEOUT_S
            while time.time() < deadline:
                await asyncio.sleep(POLL_INTERVAL_S)
                pr = await cli.get(f"/pods/{runpod_id}")
                if pr.status_code >= 400:
                    continue
                ip, port = compute._extract_ssh(pr.json() or {})
                if ip and port:
                    return runpod_id, ip, int(port), cost_f
            raise RuntimeError(f"pod {runpod_id} SSH not ready after {POLL_TIMEOUT_S}s")
        except BaseException:
            await _terminate_pod(api_key, runpod_id)
            raise


async def _terminate_pod(api_key: str, runpod_id: str) -> None:
    try:
        async with compute._client(api_key=api_key) as cli:
            await cli.delete(f"/pods/{runpod_id}")
    except Exception as e:  # noqa: BLE001
        logger.warning("training: pod %s teardown failed: %s", runpod_id, e)


async def _teardown_run_pod_if_any(row, redis=None) -> None:
    """Best-effort tear down a finalized run's RunPod pod so it stops billing.

    The live path (`_safe_run`) terminates the pod inline after finalize, but the
    restart finalize-from-log paths (`_resume_orphan_stream` / `_reconcile_orphan`)
    reach the box over SSH and historically only swept temp files — leaving the
    RunPod pod to bill indefinitely. `runpod_pod_id` is only ever set for RunPod
    runs (VM runs leave it NULL), so its presence is the signal to tear down."""
    pod = getattr(row, "runpod_pod_id", None)
    prov_id = getattr(row, "provider_id", None)
    if not (pod and prov_id):
        return
    try:
        api_key = await compute._resolve_api_key(prov_id)
        await _terminate_pod(api_key, pod)
        if redis is not None:
            await _push_log(redis, row.id, f"[gateway] pod {pod} torn down (billing stopped)")
    except Exception as e:  # noqa: BLE001 — teardown is best-effort
        logger.warning("training %s: pod %s teardown failed: %s", getattr(row, "id", "?"), pod, e)


async def _reap_label_pod(run_id: str, provider_id: Optional[str]) -> bool:
    """Best-effort: tear down an orphaned cloud label-export pod (named
    sgpu-label-<run_id>) on `provider_id` (a RunPod account) — the one
    _create_label_project_for_run spawns. Used by cancel + startup reconcile so a
    pod whose task died (cancel / gateway restart) can't bill on. True if one was
    torn down."""
    if not provider_id:
        return False
    try:
        api_key = await compute._resolve_api_key(provider_id)
        async with compute._client(api_key=api_key) as cli:
            data = (await cli.get("/pods")).json()
        pods = data if isinstance(data, list) else (data.get("pods") or data.get("data") or [])
        for p in pods:
            if p.get("name") == f"sgpu-label-{run_id}" and p.get("id"):
                await _terminate_pod(api_key, p["id"])
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


async def _reap_hf_export_pod(run_id: str, provider_id: Optional[str]) -> bool:
    """Best-effort: tear down an orphaned cloud HF-export pod (named sgpu-hfexport-<run_id>)
    on `provider_id` (a RunPod account, or None → env key) — the one export_to_huggingface
    spawns for a merged export. Used by cancel + startup reconcile so a pod whose task died
    (cancel / gateway restart) can't bill on. True if one was torn down."""
    try:
        api_key = await compute._resolve_api_key(provider_id)
        async with compute._client(api_key=api_key) as cli:
            data = (await cli.get("/pods")).json()
        pods = data if isinstance(data, list) else (data.get("pods") or data.get("data") or [])
        for p in pods:
            if p.get("name") == f"sgpu-hfexport-{run_id}" and p.get("id"):
                await _terminate_pod(api_key, p["id"])
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


# ---------- the runner ---------------------------------------------------


async def _safe_run(redis, run_id: str) -> None:
    try:
        await run_training(redis, run_id)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("training run %s crashed", run_id)
        await _finalize(run_id, "failed", None, "internal error — see gateway logs")


async def _flush_result(run_id: str, result: dict) -> None:
    """Persist the in-flight result_json (steps, trials, gpu_samples, …) WITHOUT
    touching status, so the UI shows live loss / per-trial status while running.
    JSON-copy so SQLAlchemy sees a new value and the writer can't race the dict."""
    snap = json.loads(json.dumps(result))
    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
        if row is None or row.status not in ("running",):
            return
        row.result_json = snap
        await s.commit()


async def _finalize(run_id: str, status: str, exit_code: Optional[int],
                    error_text: Optional[str], result_json: Optional[dict] = None) -> None:
    prev_status: Optional[str] = None
    task = ""
    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
        if row is None or row.status in ("cancelled",):
            return
        prev_status = row.status
        task = row.task_type or ""
        row.status = status
        row.exit_code = exit_code
        if error_text:
            row.error_text = error_text[:4000]
        if result_json is not None:
            row.result_json = result_json
        row.ended_at = datetime.now(timezone.utc)
        await s.commit()
    # Count the terminal transition once (guards against _finalize being called
    # twice for the same run, e.g. process-exit + janitor). This monotonic counter
    # is what Grafana alerts on for autotrain failures — see metrics.py.
    if status in ("done", "failed") and prev_status != status:
        try:
            from . import metrics
            metrics.AUTOTRAIN_RUNS_FINISHED.labels(status=status, task=task).inc()
        except Exception:
            pass


async def _finish_tts_pack(run_id: str, cfg: dict, packed: dict) -> str:
    """Create the packed (NeuCodec + multipack) dataset row from a finished
    pack-only run — ONE split-aware tts_packed dataset whose s3_metadata_uri
    prefix holds train/ + test/ subdirs. `_tts_pack.splits` records the per-split
    record counts so the UI shows a split picker and the trainer can hold out
    the test split. Returns the new dataset id."""
    src_id = cfg.get("pack_source_dataset_id")
    new_id = "ds-" + os.urandom(4).hex()
    splits = packed.get("splits") or {}  # {split: record_count}
    # OmniVoice packs to Higgs-codec WebDataset shards (format="omnivoice"), a
    # different kind than the NeuCodec ChiniDataset path.
    is_omni = (packed.get("format") == "omnivoice") or _tts_arch(cfg.get("base_model")) == "omnivoice"
    async with session_factory()() as s:
        src = await s.get(Dataset, src_id) if src_id else None
        run = await s.get(TrainingRun, run_id)
        base = (src.name if src else (run_id if run else src_id)) or run_id
        if is_omni:
            ds = Dataset(
                id=new_id,
                owner_id=(src.owner_id if src else run.owner_id),
                name=(f"{base}-omnivoice-packed")[:255],
                description=(f"OmniVoice Higgs-codec WebDataset shards (tokenizer "
                             f"{packed.get('tokenizer')}, splits {list(splits) or ['(flat)']}) "
                             f"of {base}")[:2048],
                kind="omnivoice_packed",
                format="webdataset",
                storage_id=(run.storage_id if run else (src.storage_id if src else None)),
                s3_metadata_uri=packed.get("s3_uri"),
                num_rows=packed.get("samples"),
                audio_field="audio", transcription_field="text",
                split_fields={"_omnivoice_pack": {"tokenizer": packed.get("tokenizer"),
                                                   "splits": splits}},
            )
        else:
            ds = Dataset(
                id=new_id,
                owner_id=(src.owner_id if src else run.owner_id),
                name=(f"{base}-tts-packed")[:255],
                description=(f"NeuCodec + multipack (seq_len {packed.get('sequence_length')}, "
                             f"tokenizer {packed.get('tokenizer')}, splits {list(splits) or ['(flat)']}) "
                             f"of {base}")[:2048],
                kind="tts_packed",
                format="chinidataset",  # multipacked NeuCodec layout (ChiniDataset parquet shards)
                storage_id=(run.storage_id if run else (src.storage_id if src else None)),
                s3_metadata_uri=packed.get("s3_uri"),
                num_rows=packed.get("samples"),
                audio_field="audio", transcription_field="text",
                split_fields={"_tts_pack": {"tokenizer": packed.get("tokenizer"),
                                            "sequence_length": packed.get("sequence_length"),
                                            "splits": splits}},
            )
        s.add(ds)
        await s.commit()
    return new_id


async def _set_dataset_transform(dataset_id: Optional[str], status: str, log_line: str) -> None:
    """Stamp a dataset's transform_status/transform_log (polled by the datasets UI)."""
    if not dataset_id:
        return
    async with session_factory()() as s:
        d = await s.get(Dataset, dataset_id)
        if d is None:
            return
        d.transform_status = status
        prev = d.transform_log or ""
        d.transform_log = (prev + ("\n" if prev else "") + log_line)[-8000:]
        await s.commit()


def _tts_pack_splits(ds: Optional[Dataset]) -> dict:
    """Per-split record counts of a tts_packed dataset (split_fields._tts_pack.splits)."""
    if ds is None:
        return {}
    return ((ds.split_fields or {}).get("_tts_pack") or {}).get("splits") or {}


async def _set_label_export_state(run_id: str, state: dict) -> None:
    """Merge a label-export status object into result_json.label_export so the UI can
    show 'exporting to Label' (instead of 'done') while a post-train export runs, and
    revert once it finishes. Unlike _flush_result this updates an already-done run."""
    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
        if row is None or row.status == "cancelled":
            return
        rj = dict(row.result_json or {})
        cur = dict(rj.get("label_export") or {})
        cur.update(state)
        rj["label_export"] = cur
        row.result_json = rj
        await s.commit()


async def _resolve_label_creds(cfg: dict) -> tuple[Optional[str], Optional[str]]:
    """(base_url, token) for the Label platform from a run cfg — resolving a
    Secrets-page (GlobalEnv) key when one was picked, else the pasted value (token =
    Fernet-encrypted label_token_enc). Either may be None when unconfigured."""
    genv = await _resolve_global_env()
    url_secret = (cfg.get("label_base_url_secret") or "").strip()
    base_url = ((genv.get(url_secret) or "").strip().rstrip("/") if url_secret
                else (cfg.get("label_base_url") or "").strip().rstrip("/"))
    tok_secret = (cfg.get("label_token_secret") or "").strip()
    if tok_secret:
        token = (genv.get(tok_secret) or "").strip() or None
    else:
        token = None
        enc = cfg.get("label_token_enc")
        if enc:
            try:
                token = (json.loads(crypto.decrypt(enc)) or {}).get("token")
            except Exception:  # noqa: BLE001
                token = None
    return (base_url or None), token


async def _verify_label_platform(base_url: str, token: str) -> Optional[str]:
    """Pre-flight before spending a pod / touching the VM: confirm the Label platform
    is reachable and the token isn't rejected. Returns None on success, else a short
    error string. Tolerant of route shape — only a connection failure, an explicit
    401/403, or a 5xx is fatal (a 404/405 still proves the server is up + the token
    passed). Hits the read-only project list (any authenticated member can list)."""
    import httpx

    url = f"{base_url.rstrip('/')}/api/projects"
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as cli:
            r = await cli.get(url, headers={"Authorization": f"Bearer {token}"})
    except Exception as e:  # noqa: BLE001
        return f"can't reach the Label platform at {base_url} ({e.__class__.__name__})"
    if r.status_code in (401, 403):
        return f"Label platform rejected the token (HTTP {r.status_code}) — needs an admin PAT"
    if r.status_code >= 500:
        return f"Label platform unhealthy (HTTP {r.status_code}) at {base_url}"
    return None


async def _fetch_eval_rows_from_upload(
    storage, dataset_id: str, filename: str, messages_field: str
) -> list[dict]:
    """Read eval prompt rows from a kind=upload dataset's OWN uploaded S3 file
    (json / jsonl / parquet). Remaps a custom messages column to `messages` and
    parses any JSON-string messages back to a list — what the on-VM generator and
    the human_mos import both expect."""
    from . import bench, dataset_metadata
    from .datasets_api import _metadata_key, _s3_target_and_prefix

    target, _ = _s3_target_and_prefix(storage)
    key = _metadata_key(storage, dataset_id, filename)
    body = await asyncio.to_thread(bench.s3_get_bytes, key, target)
    if not body:
        raise RuntimeError(f"eval file {filename!r} not found in storage")
    rows = await asyncio.to_thread(dataset_metadata.parse_rows_any, filename, body, 10 ** 9)
    for r in rows:
        if messages_field != "messages" and messages_field in r and "messages" not in r:
            r["messages"] = r[messages_field]
        v = r.get("messages")
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    r["messages"] = parsed
            except (json.JSONDecodeError, ValueError):
                pass
    return rows


async def _fetch_eval_rows_from_catalog(dataset_id: str) -> list[dict]:
    """Read JSONL rows from a kind=hf Dataset that was pushed to the platform catalog
    (or a standalone catalog repo), OR the uploaded file of a kind=upload dataset.
    Returns a list of parsed row dicts. Raises RuntimeError on any failure."""
    from sqlalchemy import select as _sa_select
    from .db import CatalogRepo as _CatalogRepo

    # kind=upload eval dataset: rows live in the dataset's own uploaded S3 file
    # (not a catalog repo). Resolve + read it directly.
    async with session_factory()() as s:
        ds_up = await s.get(Dataset, dataset_id)
        if ds_up is not None and ds_up.kind == "upload":
            up_storage = await s.get(Storage, ds_up.storage_id) if ds_up.storage_id else None
            if up_storage is None or up_storage.kind != "s3":
                raise RuntimeError("uploaded eval dataset has no S3 storage")
            if not ds_up.metadata_filename:
                raise RuntimeError("uploaded eval dataset has no file — upload one first")
            up_mf = getattr(ds_up, "messages_field", None) or "messages"
            up_name = ds_up.metadata_filename
            return await _fetch_eval_rows_from_upload(up_storage, dataset_id, up_name, up_mf)

    async with session_factory()() as s:
        ds = await s.get(Dataset, dataset_id)
        repo = None

        if ds is not None:
            if ds.kind not in ("hf", "hosted"):
                raise RuntimeError(
                    f"eval dataset must be kind=hf or kind=hosted (got {ds.kind!r})"
                )
            # Prefer the explicit catalog_repo_id FK; fall back to full_id lookup.
            repo_id_attr = getattr(ds, "catalog_repo_id", None)
            if repo_id_attr:
                repo = await s.get(_CatalogRepo, repo_id_attr)
            if repo is None and ds.hf_repo:
                res = await s.execute(
                    _sa_select(_CatalogRepo).where(_CatalogRepo.full_id == ds.hf_repo).limit(1)
                )
                repo = res.scalar_one_or_none()
        else:
            # No Dataset record — treat dataset_id as a CatalogRepo ID directly.
            # This happens when a kind=hosted catalog repo is picked from the Datasets page.
            repo = await s.get(_CatalogRepo, dataset_id)

        if repo is None:
            raise RuntimeError(
                f"no platform catalog repo found for {dataset_id!r}. "
                "Upload the JSONL via HF push first."
            )
        storage = await s.get(Storage, repo.storage_id) if repo.storage_id else None

    creds = _s3_creds_from_storage(storage)
    if not creds.get("bucket"):
        raise RuntimeError("catalog repo storage has no S3 bucket")

    manifest = repo.manifest or []
    jsonl_entry = next(
        (e for e in manifest if isinstance(e, dict) and (e.get("path") or "").endswith(".jsonl")),
        None,
    )
    if jsonl_entry is None:
        paths = [e.get("path") for e in manifest[:10] if isinstance(e, dict)]
        raise RuntimeError(
            f"no .jsonl file in catalog repo {repo.full_id!r} manifest "
            f"(found: {paths})"
        )

    prefix = (repo.prefix or "").rstrip("/")
    oid = jsonl_entry.get("oid") or ""
    if repo.versioned and oid:
        s3_key = f"{prefix}/blobs/{oid}" if prefix else f"blobs/{oid}"
    else:
        path = jsonl_entry.get("path", "")
        s3_key = f"{prefix}/{path}" if prefix else path

    import boto3
    from botocore.client import Config as _BotoConfig

    cli = boto3.client(
        "s3", region_name=creds.get("region") or "us-east-1",
        endpoint_url=creds.get("endpoint") or None,
        aws_access_key_id=creds.get("access_key") or None,
        aws_secret_access_key=creds.get("secret_key") or None,
        config=_BotoConfig(signature_version="s3v4"),
    )
    try:
        resp = await asyncio.to_thread(
            lambda: cli.get_object(Bucket=creds["bucket"], Key=s3_key)
        )
        content = resp["Body"].read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(
            f"failed to download eval JSONL from s3://{creds['bucket']}/{s3_key}: {e}"
        ) from e

    rows = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


async def _create_llm_label_project_for_run(run_id: str, cfg: dict, result: dict, redis) -> None:
    """Post-train: generate responses for eval rows using the finetuned LLM on the
    run's VM, then create a Label-platform *human_mos* project seeded with them.
    Records the project under result_json.label_project. VM-provider runs only."""
    # Resolve Label platform URL + token (same logic as _create_label_project_for_run).
    genv = await _resolve_global_env()
    url_secret = (cfg.get("label_base_url_secret") or "").strip()
    base_url = (genv.get(url_secret) or "").strip().rstrip("/") if url_secret else (
        (cfg.get("label_base_url") or "").strip().rstrip("/")
    )
    tok_secret = (cfg.get("label_token_secret") or "").strip()
    token = None
    if tok_secret:
        token = (genv.get(tok_secret) or "").strip() or None
    else:
        enc = cfg.get("label_token_enc")
        if enc:
            try:
                token = (json.loads(crypto.decrypt(enc)) or {}).get("token")
            except Exception:  # noqa: BLE001
                token = None
    if not base_url:
        await _push_log(redis, run_id, "[gateway] llm label export: base_url missing — skipped")
        return
    if not token:
        await _push_log(redis, run_id, "[gateway] llm label export: token missing — skipped")
        return

    model_s3 = ((result or {}).get("artifact") or {}).get("s3_uri")
    if not model_s3:
        await _push_log(redis, run_id, "[gateway] llm label export: no model artifact — skipped")
        return

    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
        if row is None:
            return
        prov = await s.get(Provider, row.provider_id) if row.provider_id else None
        storage = await s.get(Storage, row.storage_id) if row.storage_id else None
        owner_id = row.owner_id
        run_name = row.name or run_id
        storage_id = row.storage_id
    if prov is None or prov.kind != "vm":
        await _push_log(redis, run_id, "[gateway] llm label export needs a VM provider (skipped)")
        return

    eval_dataset_id = cfg.get("llm_label_eval_dataset_id")
    if not eval_dataset_id:
        await _push_log(redis, run_id, "[gateway] llm label export: no eval dataset configured — skipped")
        return

    await _push_log(redis, run_id, f"[gateway] llm label export: fetching eval rows from {eval_dataset_id} …")
    try:
        all_rows = await _fetch_eval_rows_from_catalog(eval_dataset_id)
    except Exception as e:
        await _push_log(redis, run_id, f"[gateway] llm label export: failed to fetch eval rows: {e}")
        await _set_label_export_state(run_id, {"status": "failed", "error": str(e)[:300]})
        return

    n = int(cfg.get("llm_label_samples") or 0) or len(all_rows)
    eval_rows = all_rows[:n]
    await _push_log(redis, run_id,
                    f"[gateway] llm label export: {len(eval_rows)} rows ({len(all_rows)} total in dataset)")

    creds = _s3_creds_from_storage(storage)
    # The LLM label export merges the LoRA into the (usually gated) base on the box, so the
    # merge's from_pretrained needs an HF token. Inject one as HF_TOKEN — env_vars is
    # otherwise empty and the base download would 401 (matches the HF-export merge fix).
    # Precedence: an explicit base-model token (secret key / pasted, from the export tab) >
    # the run's own token (its hf_token_secret) > the platform HF_TOKEN secret.
    _hf_tok = None
    if cfg.get("base_hf_token_secret"):
        _hf_tok = genv.get(cfg["base_hf_token_secret"])
    elif cfg.get("base_hf_token_enc"):
        try:
            _hf_tok = (json.loads(crypto.decrypt(cfg["base_hf_token_enc"])) or {}).get("token")
        except Exception:  # noqa: BLE001
            _hf_tok = None
    _hf_tok = _hf_tok or (genv.get(cfg["hf_token_secret"]) if cfg.get("hf_token_secret") else None) or genv.get("HF_TOKEN")
    cfg = _cfg_with_hf_token(cfg, _hf_tok)
    ssh = await _resolve_run_ssh(row)
    if ssh is None:
        await _push_log(redis, run_id, "[gateway] llm label export: can't reach the run's VM (skipped)")
        await _set_label_export_state(run_id, {"status": "failed", "error": "can't reach the run's VM"})
        return

    await _set_label_export_state(run_id, {"status": "running"})
    export_lines: list[str] = []
    _sent = {"n": 0}

    async def _export_pump() -> None:
        while True:
            await asyncio.sleep(0.5)
            while _sent["n"] < len(export_lines):
                await _push_log(redis, run_id, export_lines[_sent["n"]])
                _sent["n"] += 1

    pump_task = asyncio.create_task(_export_pump())
    try:
        await _push_log(redis, run_id,
                        f"[gateway] llm label export: generating {len(eval_rows)} response(s) on VM …")
        manifest = await asyncio.to_thread(
            _run_llm_label_export_ssh, *ssh, run_id, model_s3, cfg, creds,
            eval_rows, export_lines.append,
        )
        items = manifest.get("items") or []
        if not items:
            await _push_log(redis, run_id, "[gateway] llm label export: no items generated")
            await _set_label_export_state(run_id, {"status": "failed", "error": "no items generated"})
            return

        # ---- Label platform: create human_mos project → set MOS axes → import tasks ----
        import httpx

        proj_name = (cfg.get("label_project_name") or f"{run_name}-eval")[:200]
        axes = [a for a in (cfg.get("llm_label_mos_axes") or []) if a] or [
            "Relevance", "Accuracy", "Helpfulness", "Tone"
        ]
        base_model_name = (cfg.get("base_model") or "").split("/")[-1] or "llm"
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as cli:
            r = await cli.post(f"{base_url}/api/projects", headers=headers, json={
                "name": proj_name, "type": "human_mos",
                "description": f"Autotrain LLM eval — {run_name} ({run_id})",
            })
            r.raise_for_status()
            pid = ((r.json() or {}).get("project") or {}).get("id")
            if not pid:
                raise RuntimeError(f"create project returned no id: {r.text[:200]}")
            # human_mos supports mos_axes via PATCH
            r = await cli.patch(f"{base_url}/api/projects/{pid}", headers=headers,
                                json={"mos_enabled": True, "mos_axes": axes})
            r.raise_for_status()
            # Each task: human_mos_data.messages (last = assistant) + model field.
            # Encode lov/language in the model field as JSON so labellers can filter.
            tasks = []
            for it in items:
                meta_str = json.dumps({
                    "model": base_model_name,
                    "lov": it.get("lov", ""),
                    "language": it.get("language", ""),
                })
                tasks.append({
                    "human_mos_data": {"messages": it.get("messages") or []},
                    "model": meta_str,
                })
            r = await cli.post(f"{base_url}/api/projects/{pid}/tasks",
                               headers=headers, json={"tasks": tasks})
            r.raise_for_status()

        # Round-trip kind=label Dataset pointing at the new project.
        label_ds_id = "ds-" + os.urandom(4).hex()
        async with session_factory()() as s:
            s.add(Dataset(
                id=label_ds_id, owner_id=owner_id,
                name=(f"{run_name}-llm-eval-labels")[:255],
                description=(f"Human MOS labels for autotrain LLM run {run_id}")[:2048],
                kind="label", storage_id=storage_id,
                label_base_url=base_url, label_project_id=pid,
                label_token_secret=(tok_secret or None),
                label_token_enc=(None if tok_secret else crypto.encrypt(json.dumps({"token": token}))),
                label_status="approved",
                audio_field="audio_url", transcription_field="transcription",
            ))
            run2 = await s.get(TrainingRun, run_id)
            if run2 is not None:
                rj = dict(run2.result_json or {})
                rj["label_project"] = {
                    "id": pid, "url": f"{base_url}/dashboard/projects/{pid}",
                    "count": len(tasks), "dataset_id": label_ds_id,
                    "project_name": proj_name, "project_type": "human_mos",
                }
                rj["label_export"] = {"status": "done"}
                run2.result_json = rj
            await s.commit()
        await _push_log(redis, run_id,
                        f"[gateway] llm label project created: {base_url}/dashboard/projects/{pid} "
                        f"({len(tasks)} conversations) + dataset {label_ds_id}")
    except Exception as e:  # noqa: BLE001
        await _set_label_export_state(run_id, {"status": "failed", "error": str(e)[:300]})
        raise
    finally:
        pump_task.cancel()
        try:
            await pump_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        while _sent["n"] < len(export_lines):
            try:
                await _push_log(redis, run_id, export_lines[_sent["n"]])
            except Exception:  # noqa: BLE001
                pass
            _sent["n"] += 1




async def _create_label_project_for_run(run_id: str, cfg: dict, result: dict, redis) -> None:
    """Post-train: synthesize N clips from the finished TTS model on the run's VM,
    upload them to the run's S3 storage, then create a Label-platform *recording*
    project (MOS rating enabled), configure its storage, import the clips as tasks,
    and auto-create a round-trip kind=label Dataset. Records the project under
    result_json.label_project. VM-provider runs only (synthesis needs the box)."""
    # URL + token each resolve from a Secrets-page (GlobalEnv) key when one was
    # picked, else from the pasted value (token = Fernet-encrypted label_token_enc).
    base_url, token = await _resolve_label_creds(cfg)
    tok_secret = (cfg.get("label_token_secret") or "").strip()
    if not base_url:
        await _push_log(redis, run_id, "[gateway] label export: base_url missing/unresolved — skipped")
        return
    if not token:
        await _push_log(redis, run_id, "[gateway] label export: token missing/unresolved — skipped")
        return
    # Verify connectivity + auth to the Label platform BEFORE spending a pod / touching
    # the VM — a bad URL or token fails here, not after a multi-minute synth.
    verr = await _verify_label_platform(base_url, token)
    if verr:
        await _push_log(redis, run_id, f"[gateway] label export: {verr} — skipped (no pod/VM used)")
        await _set_label_export_state(run_id, {"status": "failed", "error": verr})
        return
    model_s3 = ((result or {}).get("artifact") or {}).get("s3_uri")
    if not model_s3:
        await _push_log(redis, run_id, "[gateway] label export: no model artifact — skipped")
        return
    model_s3 = ((result or {}).get("artifact") or {}).get("s3_uri")
    if not model_s3:
        await _push_log(redis, run_id, "[gateway] label export: no model artifact — skipped")
        return

    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
        if row is None:
            return
        prov = await s.get(Provider, row.provider_id) if row.provider_id else None
        storage = await s.get(Storage, row.storage_id) if row.storage_id else None
        main_ds = await s.get(Dataset, row.dataset_id) if row.dataset_id else None
        test_ds = await s.get(Dataset, row.test_dataset_id) if row.test_dataset_id else None
        owner_id = row.owner_id
        run_name = row.name or run_id
        storage_id = row.storage_id
    # Where to synthesize: an explicit Run-on target chosen in the export tab —
    # "vm" (a registered VM provider) or "cloud" (a fresh RunPod pod) — else the
    # run's own box. For the AUTO post-train export the training box is still alive
    # (its teardown happens after this), so a cloud run reuses its own training pod
    # via _resolve_run_ssh below — no new pod. Only a later manual retry, once that
    # pod is gone, needs an explicit Run-on target.
    run_on = (cfg.get("label_run_on") or "").strip().lower()
    exp_provider_id = (cfg.get("label_provider_id") or "").strip() or None

    creds = _s3_creds_from_storage(storage)
    if not creds.get("bucket"):
        await _push_log(redis, run_id, "[gateway] label export: storage has no S3 bucket — skipped")
        return

    # Pick the text source: a separate test dataset → else the main dataset's
    # held-out test split → else a random sample of the train split (or flat root).
    if test_ds is not None and test_ds.s3_metadata_uri:
        base = test_ds.s3_metadata_uri.rstrip("/")
        tsp = _tts_pack_splits(test_ds)
        packed_uri = f"{base}/test" if "test" in tsp else base
        is_random, source_desc = False, "test dataset"
    else:
        base = (main_ds.s3_metadata_uri or "").rstrip("/") if main_ds else ""
        if not base:
            await _push_log(redis, run_id, "[gateway] label export: no packed dataset uri — skipped")
            return
        msp = _tts_pack_splits(main_ds)
        if "test" in msp:
            packed_uri, is_random, source_desc = f"{base}/test", False, "held-out test split"
        elif "train" in msp:
            packed_uri, is_random, source_desc = f"{base}/train", True, "random train sample"
        else:
            packed_uri, is_random, source_desc = base, True, "random sample"

    n = max(1, int(cfg.get("label_samples") or 32))
    speaker = (cfg.get("default_speaker") or "").strip()
    # Object key under the run's storage prefix → put the FULL key in audio_filename
    # and set the Label storage prefix to "" so BOTH the proxy playback path and the
    # bare-key S3 export presign resolve to the same object.
    sp = (creds.get("prefix") or "").strip("/")
    upload_prefix = (f"{sp}/training-runs/{run_id}/tts-label" if sp
                     else f"training-runs/{run_id}/tts-label")

    # Mark in-flight up front (cloud provisioning takes minutes) so the UI shows
    # "exporting to Label" instead of the run's terminal "done" the whole time.
    # Clear any prior error so a retry doesn't carry a stale failure message (state
    # is merged, so an explicit None resets it).
    await _set_label_export_state(run_id, {"status": "running", "error": None})
    # CUDA pin for the synthesis (Run-on → CUDA_VISIBLE_DEVICES), else all GPUs.
    gpu_for_synth = (cfg.get("label_visible_devices") or "").strip() or "auto"
    # A pod we spawn for the "cloud" target → (api_key, runpod_id) to tear down after.
    spawned: Optional[tuple[str, str]] = None
    try:
        if run_on == "cloud":
            api_key = await compute._resolve_api_key(exp_provider_id)
            key_filename, pub = _gen_ssh_key(_work_dir(run_id))
            gpu_type = cfg.get("label_gpu_type") or "NVIDIA L40S"
            gpu_count = max(1, int(cfg.get("label_gpu_count") or 1))
            await _push_log(redis, run_id,
                            f"[gateway] label export: provisioning RunPod pod ({gpu_type} x{gpu_count}) …")
            runpod_id, host, port, _cost = await _provision_pod(
                api_key, f"sgpu-label-{run_id}", cfg.get("image") or DEFAULT_IMAGE,
                gpu_type, gpu_count, bool(cfg.get("label_secure_cloud", True)),
                int(cfg.get("label_disk_gb", 60)), int(cfg.get("label_volume_gb", 80)), pub,
                data_center_id=cfg.get("label_data_center_id"))
            spawned = (api_key, runpod_id)
            await _push_log(redis, run_id,
                            "[gateway] label export: installing the TTS stack on the pod (first build is slow) …")
            # Stream the (slow) install to the run's logs via a short-lived pump, so it
            # isn't silent and any pip/uv error is visible live (not just a final rc).
            build_lines: list[str] = []
            _bn = {"n": 0}

            async def _build_pump() -> None:
                while True:
                    await asyncio.sleep(0.5)
                    while _bn["n"] < len(build_lines):
                        await _push_log(redis, run_id, build_lines[_bn["n"]])
                        _bn["n"] += 1

            bpump = asyncio.create_task(_build_pump())
            try:
                await asyncio.to_thread(_label_build_tts_venv_ssh, host, int(port), "root",
                                        key_filename, run_id, dict(cfg), build_lines.append)
            finally:
                bpump.cancel()
                try:
                    await bpump
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
                while _bn["n"] < len(build_lines):  # flush any tail the pump missed
                    try:
                        await _push_log(redis, run_id, build_lines[_bn["n"]])
                    except Exception:  # noqa: BLE001
                        pass
                    _bn["n"] += 1
            ssh = (host, int(port), "root", key_filename)
        elif run_on == "vm":
            vm_prov = prov
            if exp_provider_id and (prov is None or prov.id != exp_provider_id):
                async with session_factory()() as s:
                    vm_prov = await s.get(Provider, exp_provider_id)
            if vm_prov is None or vm_prov.kind != "vm":
                await _set_label_export_state(run_id, {"status": "failed", "error": "chosen VM provider not found"})
                return
            ssh = await _resolve_provider_ssh(vm_prov, run_id)
            if ssh is None:
                await _push_log(redis, run_id, "[gateway] label export: can't reach the chosen VM (skipped)")
                await _set_label_export_state(run_id, {"status": "failed", "error": "can't reach the chosen VM"})
                return
        else:
            ssh = await _resolve_run_ssh(row)
            if ssh is None:
                await _push_log(redis, run_id,
                                "[gateway] label export: the run's box is gone — open the Export to Label "
                                "tab and pick a Run-on target (VM or cloud) to retry (skipped)")
                await _set_label_export_state(run_id, {"status": "failed",
                                                       "error": "the training box is gone — pick a Run-on target (VM or cloud) and retry"})
                return
    except asyncio.CancelledError:
        if spawned:
            await _terminate_pod(*spawned)
        raise
    except Exception as e:  # noqa: BLE001
        if spawned:
            try:
                await _terminate_pod(*spawned)
            except Exception:  # noqa: BLE001
                pass
        await _push_log(redis, run_id, f"[gateway] label export: provisioning failed: {e}")
        await _set_label_export_state(run_id, {"status": "failed", "error": str(e)[:1200]})
        return
    # Stream the VM-side synthesis/upload log to the run's logs (the script runs in a
    # thread; it appends lines to this buffer and an async pump mirrors them to Redis,
    # the same decoupling run_training uses) — so a multi-minute synth isn't silent.
    export_lines: list[str] = []
    _sent = {"n": 0}

    async def _export_pump() -> None:
        while True:
            await asyncio.sleep(0.5)
            while _sent["n"] < len(export_lines):
                await _push_log(redis, run_id, export_lines[_sent["n"]])
                _sent["n"] += 1

    pump_task = asyncio.create_task(_export_pump())
    try:
        await _push_log(redis, run_id, f"[gateway] label export: synthesizing {n} clips ({source_desc}) …")
        manifest = await asyncio.to_thread(
            _run_tts_label_export_ssh, *ssh, run_id, model_s3, creds, packed_uri, "",
            is_random, n, speaker, creds["bucket"], upload_prefix, dict(cfg), gpu_for_synth,
            export_lines.append,
        )
        items = manifest.get("items") or []
        if not items:
            await _push_log(redis, run_id, "[gateway] label export: no clips synthesized")
            await _set_label_export_state(run_id, {"status": "failed", "error": "no clips synthesized"})
            return

        # ---- Label platform: create project → storage → MOS → tasks ----
        import httpx

        base_name = (cfg.get("label_project_name") or f"{run_name}-eval")
        axes = [a for a in (cfg.get("label_mos_axes") or []) if a] or ["Naturalness", "Intelligibility", "Noise"]
        headers = {"Authorization": f"Bearer {token}"}

        async def _make_one(cli, group: list[dict], proj_name: str, speaker: str) -> dict:
            """Create a recording+MOS project, configure its S3 storage, seed it with
            `group`'s clips as tasks, and round-trip a kind=label dataset. Returns the
            result_json card. `speaker` ("" for the combined project) tags both."""
            r = await cli.post(f"{base_url}/api/projects", headers=headers, json={
                "name": proj_name, "type": "recording",
                "description": (f"Autotrain TTS eval — {run_name} ({run_id})"
                                + (f" · speaker {speaker}" if speaker else "")),
            })
            r.raise_for_status()
            pid = ((r.json() or {}).get("project") or {}).get("id")
            if not pid:
                raise RuntimeError(f"create project returned no id: {r.text[:200]}")
            r = await cli.put(f"{base_url}/api/projects/{pid}/storage", headers=headers, json={
                "provider": "s3", "bucket": creds["bucket"], "region": creds["region"],
                "prefix": "", "endpoint": creds.get("endpoint") or "",
                "access_key": creds.get("access_key") or "", "secret_key": creds.get("secret_key") or "",
            })
            r.raise_for_status()
            r = await cli.patch(f"{base_url}/api/projects/{pid}", headers=headers,
                                json={"mos_enabled": True, "mos_axes": axes})
            r.raise_for_status()
            tasks = [{"transcription": it.get("text") or "", "audio_filename": it.get("key") or ""}
                     for it in group if it.get("key")]
            r = await cli.post(f"{base_url}/api/projects/{pid}/tasks", headers=headers, json={"tasks": tasks})
            r.raise_for_status()

            # Round-trip: a kind=label dataset pointing at the new project.
            label_ds_id = "ds-" + os.urandom(4).hex()
            async with session_factory()() as s:
                s.add(Dataset(
                    id=label_ds_id, owner_id=owner_id,
                    name=(f"{run_name}-tts-eval-labels" + (f" ({speaker})" if speaker else ""))[:255],
                    description=(f"Human MOS / recording labels for autotrain TTS run {run_id}"
                                 + (f" — speaker {speaker}" if speaker else ""))[:2048],
                    kind="label", storage_id=storage_id,
                    label_base_url=base_url, label_project_id=pid,
                    # Reference the Secrets key when one was used (no token copy); else store
                    # the resolved token Fernet-encrypted, like the datasets create path.
                    label_token_secret=(tok_secret or None),
                    label_token_enc=(None if tok_secret else crypto.encrypt(json.dumps({"token": token}))),
                    label_status="approved",
                    audio_field="audio_url", transcription_field="transcription",
                ))
                await s.commit()
            await _push_log(
                redis, run_id,
                f"[gateway] label project created{f' [{speaker}]' if speaker else ''}: "
                f"{base_url}/dashboard/projects/{pid} ({len(tasks)} clips) + dataset {label_ds_id}")
            return {
                "id": pid, "url": f"{base_url}/dashboard/projects/{pid}",
                "count": len(tasks), "dataset_id": label_ds_id, "project_name": proj_name,
                **({"speaker": speaker} if speaker else {}),
            }

        # One project per speaker (each seeded from that speaker's own clips) when
        # per-speaker is on + speakers are named; otherwise a single combined project.
        speakers_cfg = [str(s).strip() for s in (cfg.get("label_speakers") or []) if str(s).strip()]
        per_speaker = bool(cfg.get("label_per_speaker")) and bool(speakers_cfg)
        cards: list[dict] = []
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as cli:
            if per_speaker:
                from collections import defaultdict
                by_spk: dict[str, list[dict]] = defaultdict(list)
                for it in items:
                    by_spk[(it.get("speaker") or "").strip()].append(it)
                for spk in speakers_cfg:
                    grp = by_spk.get(spk) or []
                    if not grp:
                        await _push_log(redis, run_id,
                                        f"[gateway] label export: no clips for speaker {spk!r} — skipped")
                        continue
                    cards.append(await _make_one(cli, grp, f"{base_name} — {spk}"[:200], spk))
                if not cards:
                    await _set_label_export_state(run_id, {"status": "failed",
                                                           "error": "no clips matched the named speakers"})
                    return
            else:
                cards.append(await _make_one(cli, items, base_name[:200], ""))

        # Stamp the run: label_projects (all) + label_project (first, for older UIs).
        async with session_factory()() as s:
            run2 = await s.get(TrainingRun, run_id)
            if run2 is not None:
                rj = dict(run2.result_json or {})
                rj["label_projects"] = cards
                rj["label_project"] = cards[0] if cards else None
                rj["label_export"] = {"status": "done"}
                run2.result_json = rj
            await s.commit()
    except Exception as e:  # noqa: BLE001 — record the failure, then re-raise so the caller logs it
        await _set_label_export_state(run_id, {"status": "failed", "error": str(e)[:300]})
        raise
    finally:
        # Tear down a pod we spawned for the "cloud" target so it can't bill on
        # past the synthesis (best-effort; mirrors _provision_pod's self-cleanup).
        if spawned:
            try:
                await _terminate_pod(*spawned)
                await _push_log(redis, run_id, "[gateway] label export: cloud pod torn down")
            except Exception:  # noqa: BLE001
                logger.warning("label export %s: pod teardown failed", run_id)
        pump_task.cancel()
        try:
            await pump_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        # Flush any lines the pump hadn't mirrored yet.
        while _sent["n"] < len(export_lines):
            try:
                await _push_log(redis, run_id, export_lines[_sent["n"]])
            except Exception:  # noqa: BLE001
                pass
            _sent["n"] += 1


async def _auto_hf_push(run_id: str, cfg: dict, result: dict, redis) -> None:
    """Post-train: push the run's best/final model to the Hugging Face repo named in
    cfg['hf_push_repo'], from the run's still-alive training box (no new pod). The
    gateway-side twin of the manual /hf-export — used for the auto push after a
    successful TTS/ASR run. (LLM already self-pushes from its trainer.) Best-effort:
    a failure here never marks the run failed. Token: the run's HF token, else the
    org HF_TOKEN secret; pushes to huggingface.co."""
    repo = (cfg.get("hf_push_repo") or "").strip()
    model_s3 = ((result or {}).get("artifact") or {}).get("s3_uri")
    if not repo or not model_s3:
        return
    token = (cfg.get("hf_token") or "").strip() or (await _resolve_global_env()).get("HF_TOKEN")
    if not token:
        await _push_log(redis, run_id,
                        "[gateway] HF export: no token (set an HF token on the run or HF_TOKEN in Secrets) — skipped")
        return
    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
        storage = await s.get(Storage, row.storage_id) if row and row.storage_id else None
    if row is None:
        return
    creds = _s3_creds_from_storage(storage)
    ssh = await _resolve_run_ssh(row)
    if ssh is None:
        await _push_log(redis, run_id,
                        "[gateway] HF export: the run's box is gone — use the Export to Hugging Face button later (skipped)")
        return
    private = bool(cfg.get("hf_push_private", True))

    # Stream the VM-side download/upload to the run log (the script runs in a thread;
    # an async pump mirrors its buffered lines to Redis), like the manual /hf-export.
    export_lines: list[str] = []
    _sent = {"n": 0}

    async def _pump() -> None:
        while True:
            await asyncio.sleep(0.5)
            while _sent["n"] < len(export_lines):
                await _push_log(redis, run_id, export_lines[_sent["n"]])
                _sent["n"] += 1

    pump_task = asyncio.create_task(_pump())
    await _set_hf_export_state(run_id, {"status": "running", "repo": repo, "url": None, "error": None})
    try:
        await _push_log(redis, run_id,
                        f"[gateway] exporting best model to Hugging Face → {repo} (from the run's box) …")
        res = await asyncio.to_thread(
            _run_hf_export_ssh, *ssh, run_id, model_s3, creds, repo, token, private, dict(cfg),
            None, export_lines.append)
        async with session_factory()() as s:
            r2 = await s.get(TrainingRun, run_id)
            if r2 is not None:
                rj = dict(r2.result_json or {})
                rj["hf_export"] = {"status": "done", "repo": res.get("repo", repo), "url": res.get("url")}
                art = dict(rj.get("artifact") or {})
                art["hf_repo"] = res.get("repo", repo)
                rj["artifact"] = art
                r2.result_json = rj
                await s.commit()
        await _push_log(redis, run_id, f"[gateway] pushed to Hugging Face: {res.get('url')}")
    except Exception as e:  # noqa: BLE001
        await _set_hf_export_state(run_id, {"status": "failed", "error": str(e)[:300]})
        await _push_log(redis, run_id, f"[gateway] HF export failed: {e}")
    finally:
        pump_task.cancel()
        try:
            await pump_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        while _sent["n"] < len(export_lines):
            try:
                await _push_log(redis, run_id, export_lines[_sent["n"]])
            except Exception:  # noqa: BLE001
                pass
            _sent["n"] += 1


async def _mirror_pack_log(redis, run_id: str, dataset_id: Optional[str]) -> None:
    """Mirror a pack run's live log tail into the source dataset's transform_log
    (replace, not append) every few seconds, so the datasets UI shows the real
    deps/NeuCodec/pack progress instead of a static 'queued' line. Permission-safe
    (the datasets card already reads transform_log; no autotrain-section needed).
    Only mirrors while the dataset is still 'running' — never clobbers the final
    done/failed summary."""
    if not dataset_id:
        return
    while True:
        await asyncio.sleep(3)
        try:
            raw = await redis.lrange(f"train:logs:{run_id}", -150, -1)
            txt = "\n".join(
                (b.decode("utf-8", "replace") if isinstance(b, bytes) else str(b)) for b in raw
            )
            async with session_factory()() as s:
                d = await s.get(Dataset, dataset_id)
                if d is None or d.transform_status != "running":
                    return
                d.transform_log = txt[-8000:]
                await s.commit()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass


async def cancel_pack_run_for_dataset(dataset_id: str) -> bool:
    """Terminate an active pack-only training run sourced from this dataset (used
    by the datasets 'Cancel' button). Returns True if one was found + cancelled."""
    async with session_factory()() as s:
        rows = (
            await s.execute(
                select(TrainingRun).where(TrainingRun.status.in_(("queued", "running")))
            )
        ).scalars().all()
        target = next(
            (r for r in rows
             if (r.config_json or {}).get("pack_only")
             and (r.config_json or {}).get("pack_source_dataset_id") == dataset_id),
            None,
        )
        if target is None:
            return False
        t = _active_runners.get(target.id)
        if t:
            t.cancel()
        st = _RUN_STATE.pop(target.id, None)
        if st:
            await _terminate_pod(st.get("api_key"), st.get("runpod_id"))
        target.status = "cancelled"
        target.ended_at = datetime.now(timezone.utc)
        await s.commit()
        return True


async def run_training(redis, run_id: str) -> None:
    work = _work_dir(run_id)
    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
        if row is None:
            return
        row.status = "running"
        row.started_at = datetime.now(timezone.utc)
        await s.commit()
        cfg = dict(row.config_json or {})
        provider_id = row.provider_id
        storage_id = row.storage_id
        dataset_id = row.dataset_id
        test_dataset_id = row.test_dataset_id
        s3_prefix = row.s3_prefix
        gpu_type = row.gpu_type or "NVIDIA L40S"
        gpu_count = row.gpu_count or 1
        visible_devices = row.visible_devices

    await _push_log(redis, run_id, f"[gateway] starting autotrain run {run_id}")

    # ---- build the trainer config (creds injected, never persisted) ----
    try:
        # Org-wide secrets (the Secrets page) are the source of truth for creds;
        # os.environ is the last-resort fallback.
        genv = await _resolve_global_env()

        def g(k: str) -> Optional[str]:
            v = genv.get(k)
            return v if v not in (None, "") else (os.environ.get(k) or None)

        # Per-run HF token (gated/private datasets + push to Hub). A user-supplied
        # token wins — a referenced Secrets-page key (hf_token_secret) or the pasted
        # token (Fernet-encrypted hf_token_enc). None here → the org HF_TOKEN fallback
        # is applied below where needed.
        _run_hf_token: Optional[str] = None
        _hf_sec = (cfg.get("hf_token_secret") or "").strip()
        if _hf_sec:
            _run_hf_token = g(_hf_sec)
        elif cfg.get("hf_token_enc"):
            try:
                _run_hf_token = json.loads(crypto.decrypt(cfg["hf_token_enc"])).get("token")
            except Exception:  # noqa: BLE001
                _run_hf_token = None
        # Token used to read the dataset(s): the user's run token, else the org HF_TOKEN.
        _ds_hf_token = _run_hf_token or g("HF_TOKEN")

        cfg["dataset"] = await _resolve_dataset_spec(dataset_id, _ds_hf_token)
        if cfg.get("no_eval"):
            # "No test set" — train on everything, no eval. Never resolve a test
            # dataset or set test_from_split (the trainers force eval off).
            await _push_log(redis, run_id, "[gateway] no_eval: training with no test set / no eval")
        elif test_dataset_id and test_dataset_id != dataset_id:
            cfg["test_dataset"] = await _resolve_dataset_spec(test_dataset_id, _ds_hf_token)
        elif test_dataset_id and test_dataset_id == dataset_id:
            # Train + eval on the SAME dataset → evaluate on its built-in
            # test/validation rows (the trainer's split_pairs prefers a `split`
            # column, falling back to a seeded hold-out only if absent). Do NOT
            # set a separate test_dataset (that would eval on the full set).
            cfg["test_from_split"] = True
            await _push_log(
                redis, run_id,
                "[gateway] test dataset == training dataset → evaluating on its "
                "own test/validation split (split column).",
            )
        target = await _training_s3_target(storage_id)
        cfg["artifacts"] = {
            "bucket": target.bucket, "region": target.region, "endpoint": target.endpoint,
            "access_key": target.access_key, "secret_key": target.secret_key,
            "prefix": s3_prefix.rstrip("/"),
        }
        # HF token for the trainer itself: push to Hub + gated base-model download.
        # The user's run token (resolved above) wins; else the org HF_TOKEN when the run
        # pushes to HF or trains a gated base (llm — gemma-4 downloads on the box). The
        # fastText lang model used for ASR language detection is public (no token).
        _hf_tok = _run_hf_token or (
            g("HF_TOKEN")
            if (cfg.get("hf_push_repo") or (cfg.get("task_type") or "").lower() == "llm")
            else None
        )
        if _hf_tok:
            cfg["hf_token"] = _hf_tok
        # Experiment tracking: non-secret per-run fields from config_json override
        # the global-env value; secrets (WANDB_API_KEY, MLFLOW_TRACKING_USERNAME/
        # PASSWORD) come from the Secrets page. Built here at run time, never
        # persisted on the row.
        # Named tracking credentials (Secrets page) win; global-env is the
        # fallback. A selected credential id implies that tracker is enabled.
        from .tracking_creds_api import resolve_tracking_env
        wandb_cred = await resolve_tracking_env(cfg.get("wandb_credential_id"))
        mlflow_cred = await resolve_tracking_env(cfg.get("mlflow_credential_id"))
        report_to = list(cfg.get("report_to") or [])
        if cfg.get("wandb_credential_id") and "wandb" not in report_to:
            report_to.append("wandb")
        if cfg.get("mlflow_credential_id") and "mlflow" not in report_to:
            report_to.append("mlflow")
        report_to = [t for t in report_to if t in ("mlflow", "wandb")]
        if report_to:
            env: dict[str, Optional[str]] = {}
            if "wandb" in report_to:
                env["WANDB_API_KEY"] = wandb_cred.get("WANDB_API_KEY") or g("WANDB_API_KEY")
                env["WANDB_PROJECT"] = cfg.get("wandb_project") or g("WANDB_PROJECT")
                env["WANDB_ENTITY"] = cfg.get("wandb_entity") or g("WANDB_ENTITY")
                env["WANDB_NAME"] = cfg.get("run_name")
            if "mlflow" in report_to:
                env["MLFLOW_TRACKING_URI"] = (cfg.get("mlflow_tracking_uri")
                                              or mlflow_cred.get("MLFLOW_TRACKING_URI") or g("MLFLOW_TRACKING_URI"))
                env["MLFLOW_EXPERIMENT_NAME"] = cfg.get("mlflow_experiment") or g("MLFLOW_EXPERIMENT_NAME")
                env["MLFLOW_TRACKING_USERNAME"] = mlflow_cred.get("MLFLOW_TRACKING_USERNAME") or g("MLFLOW_TRACKING_USERNAME")
                env["MLFLOW_TRACKING_PASSWORD"] = mlflow_cred.get("MLFLOW_TRACKING_PASSWORD") or g("MLFLOW_TRACKING_PASSWORD")
                env["HF_MLFLOW_LOG_ARTIFACTS"] = "0"
            cfg["tracking"] = {"report_to": report_to, "env": {k: v for k, v in env.items() if v}}
    except Exception as e:  # noqa: BLE001
        await _push_log(redis, run_id, f"[gateway] config resolve failed: {e}")
        await _finalize(run_id, "failed", None, f"config resolve failed: {e}")
        return

    # ---- resolve where to run ----
    is_vm = False
    api_key = None
    host = port = user = key_filename = None
    runpod_id = None
    try:
        if provider_id:
            async with session_factory()() as s:
                prov = await s.get(Provider, provider_id)
            if prov is not None and prov.kind == "vm":
                is_vm = True
                pcfg = prov.config or {}
                host = pcfg.get("host")
                port = int(pcfg.get("port") or 22)
                user = pcfg.get("user") or "root"
                enc = pcfg.get("private_key_enc")
                if not enc:
                    raise RuntimeError("VM provider has no stored private key")
                key_filename = str(work / "vm_key")
                Path(key_filename).write_text(crypto.decrypt(enc))
                os.chmod(key_filename, 0o600)
                # Empty visible_devices means "all visible GPUs" (the form's hint). On a
                # VM the box's GPU inventory is authoritative — without this, gpu_count
                # keeps the create-time default (1) and the trainer launches nproc=1,
                # silently using a single GPU on a multi-GPU box. (RunPod pods keep
                # body.gpu_count, which IS the pod's GPU count.)
                if not (visible_devices or "").strip():
                    vm_n = len(pcfg.get("gpus") or []) or int(pcfg.get("gpu_count") or 0)
                    if vm_n > 0:
                        gpu_count = vm_n
                        await _push_log(
                            redis, run_id,
                            f"[gateway] no GPU pin → using all {vm_n} GPU(s) on {prov.name}",
                        )

        if not is_vm:
            api_key = await compute._resolve_api_key(provider_id)
            key_filename, pub = _gen_ssh_key(work)
            await _push_log(redis, run_id,
                            f"[gateway] provisioning RunPod pod ({gpu_type} x{gpu_count}) …")
            runpod_id, host, port, cost = await _provision_pod(
                api_key, f"sgpu-train-{run_id}", cfg.get("image") or DEFAULT_IMAGE,
                gpu_type, gpu_count, bool(cfg.get("secure_cloud", True)),
                int(cfg.get("disk_gb", 60)), int(cfg.get("volume_gb", 80)), pub,
                data_center_id=cfg.get("data_center_id"),
            )
            user = "root"
            _RUN_STATE[run_id] = {"runpod_id": runpod_id, "api_key": api_key}
            async with session_factory()() as s:
                r2 = await s.get(TrainingRun, run_id)
                if r2 is not None:
                    r2.runpod_pod_id = runpod_id
                    r2.cost_per_hr = cost
                    await s.commit()
            await _push_log(redis, run_id, f"[gateway] pod {runpod_id} ready at {host}:{port}")
    except Exception as e:  # noqa: BLE001
        await _push_log(redis, run_id, f"[gateway] provisioning failed: {e}")
        if runpod_id and api_key:
            await _terminate_pod(api_key, runpod_id)
        await _finalize(run_id, "failed", None, f"provisioning failed: {e}")
        _RUN_STATE.pop(run_id, None)
        return

    # ---- ship + run the trainer over SSH, streaming stdout ----
    result: dict = {"epochs": [], "steps": [], "gpu_samples": [], "best": None,
                    "artifact": None, "stopped_early": False, "progress": None}
    line_buf: list[str] = []

    def on_line(line: str) -> None:
        line_buf.append(line)
        try:
            with open(_full_log_path(run_id), "a") as f:
                f.write(line + "\n")
        except Exception:
            pass
        if line.startswith("[AUTOTRAIN_PROGRESS]"):
            try:
                kv = dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)
                result["progress"] = {"step": kv.get("step"), "percent": float(kv.get("percent") or 0)}
            except Exception:
                pass
        # Sweep: the orchestrator emits "[sweep] trial N START …" when a slot
        # picks up a trial, and prefixes that trial's worker output with
        # "[trial N]". Mark the trial running, and tag its step/epoch metrics.
        ms = re.search(r"\[sweep\] trial (\d+) START", line)
        if ms:
            t = int(ms.group(1))
            trials = result.get("trials") or []
            if 0 <= t < len(trials) and trials[t].get("status") == "pending":
                trials[t]["status"] = "running"
        tm = re.match(r"\s*\[trial (\d+)\]", line)
        trial_idx = int(tm.group(1)) if tm else None
        for tag, key in (("@@METRIC ", "metric"), ("@@STEP ", "step"),
                         ("@@DONE ", "done"),
                         ("@@ARTIFACT ", "artifact"), ("@@ERROR ", "error"),
                         ("@@TRIAL ", "trial"), ("@@PACKED ", "packed")):
            # The tag can be prefixed by a tqdm progress bar (\r, no newline) on
            # the same captured line, so find it anywhere — not just at the start.
            ti = line.find(tag)
            if ti < 0:
                continue
            # A trial-prefixed @@DONE/@@ARTIFACT belongs to ONE trial's worker —
            # don't let it clobber the sweep-level best/artifact (the sweep
            # orchestrator emits its own un-prefixed @@DONE / @@TRIAL).
            if trial_idx is not None and key in ("done", "artifact"):
                break
            try:
                obj = json.loads(line[ti + len(tag):])
            except Exception:
                break
            if key == "metric":
                if "tts_eval" in obj:  # post-training TTS audio eval (CER/MOS/similarity)
                    result["tts_eval"] = obj["tts_eval"]
                    break
                if trial_idx is not None:
                    obj["trial"] = trial_idx
                result["epochs"].append(obj)
            elif key == "step":  # per-N-step training loss for the live curve
                if trial_idx is not None:
                    obj["trial"] = trial_idx
                result["steps"].append(obj)
            elif key == "done":
                result["best"] = obj.get("best")
                result["stopped_early"] = bool(obj.get("stopped_early"))
                if obj.get("trials") is not None:  # sweep summary (final)
                    result["trials"] = obj["trials"]
            elif key == "trial":  # a finished sweep trial → update its plan entry
                t = obj.get("trial")
                trials = result.get("trials") or []
                if isinstance(t, int) and 0 <= t < len(trials):
                    trials[t].update(obj)
                else:
                    result.setdefault("trials", []).append(obj)
            elif key == "artifact":
                result["artifact"] = obj
            elif key == "packed":  # TTS pack-only transform → packed dataset spec
                result["packed"] = obj
            elif key == "error":
                result["error"] = obj.get("message")
            break

    sent = {"n": 0}

    async def pump() -> None:
        # on_line already wrote each line to the on-disk log; the pump only
        # mirrors buffered lines into Redis for live SSE replay.
        while True:
            await asyncio.sleep(0.5)
            while sent["n"] < len(line_buf):
                await _push_redis(redis, run_id, line_buf[sent["n"]])
                sent["n"] += 1

    pump_task = asyncio.create_task(pump())

    # Periodically sample the run's GPUs into result["gpu_samples"] so the
    # metrics tab can show the util/mem/temp graph for a *finished* run (the
    # live /gpu poll only works while running). Uses its own SSH (distinct
    # _GPU_SSH key) so it doesn't fight the live /gpu endpoint's connection.
    _gpu_t0 = time.time()

    async def gpu_sampler() -> None:
        if not host or not key_filename:
            return
        await asyncio.sleep(8)  # let the trainer get onto the GPU first
        while True:
            try:
                gpus = await asyncio.to_thread(
                    _gpu_query_sync, f"{run_id}:sampler", host, int(port), user,
                    key_filename, visible_devices,
                )
                if gpus:
                    result["gpu_samples"].append(
                        {"t": int(time.time() - _gpu_t0), "gpus": gpus}
                    )
                    if len(result["gpu_samples"]) > 600:
                        result["gpu_samples"] = result["gpu_samples"][-600:]
            except Exception:
                pass
            await asyncio.sleep(6)

    gpu_task = asyncio.create_task(gpu_sampler())

    async def result_flusher() -> None:
        # Persist the growing result (steps, per-trial status, gpu_samples) every
        # few seconds so the metrics tab + trials list are live, not just at end.
        while True:
            await asyncio.sleep(8)
            try:
                await _flush_result(run_id, result)
            except Exception:
                pass

    flush_task = asyncio.create_task(result_flusher())
    # Pack-only runs: mirror the live log tail into the source dataset so the
    # datasets "Transformation" card shows real progress (not just "queued").
    _pack_src = cfg.get("pack_source_dataset_id") if cfg.get("pack_only") else None
    mirror_task = asyncio.create_task(_mirror_pack_log(redis, run_id, _pack_src))
    rc = 1
    try:
        cfg["gpu_count"] = gpu_count
        task_type = (cfg.get("task_type") or "asr").lower()
        # Isolated uv venv per task (mirrors serverless's vLLM venv_path) so the
        # heavy stack never clobbers the box's system python or another task's
        # deps. The trainer's --deps-only creates/installs it; the run phase
        # launches from {venv}/bin/python. Arch/task → venv lives in one helper so
        # post-train box ops (HF export) resolve the SAME venv (venv_path isn't
        # persisted to config_json — see _train_venv_default).
        venv_path = (cfg.get("venv_path")
                     or _train_venv_default(task_type, cfg.get("base_model"))).rstrip("/")
        cfg["venv_path"] = venv_path
        venv_py = shlex.quote(f"{venv_path}/bin/python")
        # Sweep mode: a sweep orchestrator schedules GPU-pinned trials. Resolve
        # the GPU id list to schedule across + the rank metric, baked into cfg.
        sweep_on = any(isinstance(v, list) and v for v in (cfg.get("sweep") or {}).values())
        if sweep_on:
            if visible_devices:
                cfg["sweep_gpus"] = [x.strip() for x in visible_devices.split(",") if x.strip()]
            elif not is_vm:
                cfg["sweep_gpus"] = [str(i) for i in range(gpu_count)]
            else:
                cfg["sweep_gpus"] = []  # VM, no pin → single slot (all GPUs)
            cfg["sweep_metric"] = "loss" if task_type in ("tts", "llm") else (cfg.get("eval_metric") or "wer")
            # Seed the full trial plan (cross-product) as "pending" so the UI can
            # list every trial up front; on_line flips each to running/done/failed.
            import itertools
            _sk = [k for k, v in (cfg.get("sweep") or {}).items() if isinstance(v, list) and v]
            _grid = [dict(zip(_sk, combo)) for combo in itertools.product(*[cfg["sweep"][k] for k in _sk])]
            result["trials"] = [
                {"trial": i, "params": p, "status": "pending", "metric": None}
                for i, p in enumerate(_grid)
            ]
        cfg_path = work / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        cli = await asyncio.to_thread(_ssh_connect, host, int(port), user, key_filename)
        try:
            # Per-run staging dir so two runs on the SAME VM (e.g. an ASR + a TTS on
            # disjoint GPUs, or two of the same kind) never clobber each other's
            # config or worker code at a shared /tmp path — the second ship would
            # otherwise win and the first trainer reads the wrong config/script
            # (wrong model/dataset → crash). The TTS worker resolves TTS_DIR as a
            # sibling (THIS_DIR/tts) and the sweep orchestrator resolves its worker
            # as a sibling too, so everything must be co-located in `stage`.
            stage = f"/tmp/sgpu_run_{run_id}"
            remote_cfg = f"{stage}/config.json"
            await asyncio.to_thread(_ssh_put, cli, str(cfg_path), remote_cfg)
            base = _trainer_script_path().parent  # gateway/gateway/training/
            if task_type == "tts" and _tts_arch(cfg.get("base_model")) == "omnivoice":
                # OmniVoice: its own orchestrator + vendored omnivoice/ scripts. It
                # imports build_dataset + S3 helpers from tts_finetune.py, so ship
                # that sibling too (its module imports are light — no NeuCodec at
                # import time; the tts/ dir isn't needed for omnivoice).
                await asyncio.to_thread(_ssh_put, cli, str(base / "omnivoice_finetune.py"), f"{stage}/omnivoice_finetune.py")
                await asyncio.to_thread(_ssh_put, cli, str(base / "tts_finetune.py"), f"{stage}/tts_finetune.py")
                await asyncio.to_thread(_ssh_put_dir_tar, cli, str(base / "omnivoice"), f"{stage}/omnivoice")
                worker_remote = f"{stage}/omnivoice_finetune.py"
            elif task_type == "tts":
                await asyncio.to_thread(_ssh_put, cli, str(base / "tts_finetune.py"), f"{stage}/tts_finetune.py")
                # The tts/ dir carries the vendored chinidataset package + the
                # convert/pack/train scripts; ship it as a sibling of the worker.
                await asyncio.to_thread(_ssh_put_dir_tar, cli, str(base / "tts"), f"{stage}/tts")
                worker_remote = f"{stage}/tts_finetune.py"
            elif task_type == "llm":
                await asyncio.to_thread(_ssh_put, cli, str(base / "llm_finetune.py"), f"{stage}/llm_finetune.py")
                # The llm/ dir carries the vendored gemma4 trainer (gemma4.py +
                # attention.py) + the chinidataset package; ship it as a sibling.
                await asyncio.to_thread(_ssh_put_dir_tar, cli, str(base / "llm"), f"{stage}/llm")
                worker_remote = f"{stage}/llm_finetune.py"
            else:
                await asyncio.to_thread(_ssh_put, cli, str(_trainer_script_path()), f"{stage}/whisper_finetune.py")
                worker_remote = f"{stage}/whisper_finetune.py"
            # … then either the worker directly, or the sweep orchestrator over it.
            # User OS env (HOME, cache dirs, …) exported before the trainer in
            # both sweep and single runs — absolute values mkdir'd. GPU pinning
            # stays per-branch (the sweep orchestrator pins each trial itself).
            user_env = _render_env_exports(cfg.get("env_vars") or {})
            # --- deps phase: create/reuse the isolated uv venv (system python) ---
            # uv bootstrap goes AFTER user_env (needs $HOME) so the child python's
            # shutil.which("uv") finds it and the trainer uses uv, not pip.
            deps_cmd = f"{user_env}{_UV_BOOTSTRAP}python -u {worker_remote} --deps-only --config {remote_cfg}"
            await _push_log(redis, run_id, f"[gateway] $ {deps_cmd}")
            drc = await asyncio.to_thread(_ssh_run_stream, cli, deps_cmd, on_line)
            if drc != 0:
                raise RuntimeError(f"dependency setup (uv venv {venv_path}) failed (rc={drc})")
            # --- run phase: launch the trainer from the venv python ---
            if sweep_on:
                await asyncio.to_thread(_ssh_put, cli, str(base / "sweep_runner.py"), f"{stage}/sweep_runner.py")
                remote_script = f"{stage}/sweep_runner.py"
                env_prefix = ""  # the orchestrator pins each trial itself
                launch = f"{venv_py} -u"
            else:
                remote_script = worker_remote
                env_prefix = f"CUDA_VISIBLE_DEVICES={visible_devices} " if visible_devices else ""
                # Multi-GPU single run → DistributedDataParallel via torch's
                # launcher (one process per GPU), unless disabled — falls back to
                # plain `python` (HF's implicit DataParallel). torch.distributed.run
                # is used over the `torchrun` console script so it doesn't depend
                # on PATH. The trainer rank-guards its log/upload (see _IS_MAIN).
                n_gpus = (len([x for x in visible_devices.split(",") if x.strip()])
                          if visible_devices else int(gpu_count or 1))
                if cfg.get("use_ddp", True) and n_gpus > 1 and task_type == "asr":
                    # Pick a FREE port on the VM at launch (the old deterministic
                    # 29500+hash collided with a prior run's TIME_WAIT / concurrent
                    # runs → EADDRINUSE). MP is set first; the CVD pin stays in the
                    # `VAR=val python` export position so torchrun still sees the GPUs.
                    _fp = (venv_py + " -c \"import socket,sys; s=socket.socket(); s.bind(('',0)); "
                           "sys.stdout.write(str(s.getsockname()[1])); s.close()\"")
                    env_prefix = (f"MP=$({_fp}); "
                                  + (f"CUDA_VISIBLE_DEVICES={visible_devices} " if visible_devices else ""))
                    launch = f"{venv_py} -m torch.distributed.run --nproc_per_node={n_gpus} --master_port=$MP"
                    await _push_log(redis, run_id, f"[gateway] DDP via torch.distributed.run · {n_gpus} GPUs (free port)")
                else:
                    launch = f"{venv_py} -u"
            # Graceful early-stop signal: the trainer polls this flag file each step
            # (a TrainerCallback) → stops cleanly + saves/uploads the partial model.
            # `export` so it propagates to torchrun's worker processes; the
            # /stop-early endpoint just `touch`es this path over SSH.
            stop_flag = f"{stage}/STOP"
            cmd = f"{user_env}export SGPU_STOP_FLAG={stop_flag};{env_prefix}{launch} {remote_script} --config {remote_cfg}"
            await _push_log(redis, run_id, f"[gateway] $ {cmd}")
            # Detach the trainer into its own session with output → a log file, so a
            # gateway restart / SSH drop can't SIGHUP-kill it. Stream by tailing that
            # log behind a watcher that stops once the training pid exits; the
            # trainer's exit code rides back on an `@@RC:` line. cleanup_orphaned_running
            # + the janitor finalize a surviving run from the same log + pidfile.
            rlog, rpid, rsh = _remote_run_paths(run_id)
            # The runscript records ITS OWN pid ($$ — the bash that also writes the
            # trailing @@RC), so a dead pidfile always implies @@RC is already in the
            # log. (Recording $! of `setsid` raced: setsid can exit before @@RC, so a
            # liveness check would see "dead" mid-run and mis-finalize the run.)
            # @@RC captures the trainer's exit code BEFORE the per-run stage dir is
            # removed (rlog/rpid live outside `stage`, so finalize-from-log still
            # works). A killed process skips this — `stage` is tiny, so a leftover is
            # harmless.
            run_sh = (f"#!/bin/bash\necho $$ > {rpid}\n{cmd}\nRC=$?\n"
                      f"echo \"@@RC:$RC\"\nrm -rf {stage} 2>/dev/null || true\n")
            await asyncio.to_thread(_ssh_put_bytes, cli, run_sh.encode(), rsh)
            await asyncio.to_thread(
                _ssh_exec, cli,
                f"rm -f {rlog} {rpid}; setsid bash {rsh} > {rlog} 2>&1 </dev/null &")
            await _push_log(redis, run_id, "[gateway] trainer detached (survives gateway restart) — tailing log")
            _rc_box = {"rc": None}

            def _cap(l: str) -> None:
                _m = re.match(r"\s*@@RC:(-?\d+)", l)
                if _m:
                    _rc_box["rc"] = int(_m.group(1))
                    return  # internal marker — don't surface in the run log
                on_line(l)

            _stream = (f"tail -n +1 -F {rlog} & T=$!; "
                       f'while :; do P=$(cat {rpid} 2>/dev/null); '
                       f'if [ -n "$P" ] && ! kill -0 "$P" 2>/dev/null; then break; fi; sleep 2; done; '
                       f"sleep 2; kill $T 2>/dev/null")
            await asyncio.to_thread(_ssh_run_stream, cli, _stream, _cap)
            rc = _rc_box["rc"] if _rc_box["rc"] is not None else 1
        finally:
            try:
                cli.close()
            except Exception:
                pass
    except Exception as e:  # noqa: BLE001
        await _push_log(redis, run_id, f"[gateway] run failed: {e}")
        result.setdefault("error", str(e))
    finally:
        gpu_task.cancel()
        try:
            await gpu_task
        except BaseException:
            pass
        _GPU_SSH.pop(f"{run_id}:sampler", None)
        flush_task.cancel()
        try:
            await flush_task
        except BaseException:
            pass
        pump_task.cancel()
        try:
            await pump_task
        except BaseException:
            pass
        mirror_task.cancel()
        try:
            await mirror_task
        except BaseException:
            pass
        # final flush of any lines the pump didn't reach before cancellation
        while sent["n"] < len(line_buf):
            await _push_redis(redis, run_id, line_buf[sent["n"]])
            sent["n"] += 1

    # ---- upload logs, finalize, teardown ----
    try:
        full = _full_log_path(run_id)
        if full.exists():
            s3_put_text(s3_prefix + "logs.txt", full.read_text(errors="replace"), target=target)
    except Exception as e:  # noqa: BLE001
        logger.warning("training %s: logs upload failed: %s", run_id, e)

    # @@DONE (best) or @@ARTIFACT means the trainer finished + uploaded → treat as
    # done even if the exit code wasn't captured (e.g. a tail race on a detached run).
    _ok = (rc == 0) or (result.get("best") is not None) or bool(result.get("artifact"))
    status = "done" if (_ok and not result.get("error")) else "failed"
    err = result.get("error") if status == "failed" else None
    if status == "failed" and not err:
        err = f"trainer exited with code {rc}"
    await _finalize(run_id, status, rc, err, result_json=result)
    await _push_log(redis, run_id, f"[gateway] run {status} (rc={rc})")

    # "Pack for TTS" transform: on success create a packed dataset row + stamp
    # the source dataset's transform_status (the datasets UI polls it).
    if cfg.get("pack_only"):
        src_id = cfg.get("pack_source_dataset_id")
        packed = result.get("packed") or {}
        if status == "done" and packed.get("s3_uri"):
            try:
                new_ds_id = await _finish_tts_pack(run_id, cfg, packed)
                await _set_dataset_transform(
                    src_id, "done",
                    f"packed → {packed['s3_uri']} (splits {packed.get('splits') or {}}); "
                    f"created dataset {new_ds_id}")
            except Exception as e:  # noqa: BLE001
                await _set_dataset_transform(src_id, "failed", f"pack ok but dataset create failed: {e}")
        else:
            await _set_dataset_transform(src_id, "failed", err or f"pack failed (rc={rc})")

    # After a successful run, optionally create a Label-platform project seeded with
    # model outputs for human evaluation. TTS → recording+MOS project (synthesized
    # clips); LLM → human_mos project (generated text responses). Best-effort: a
    # failure here never marks the run failed.
    if status == "done" and cfg.get("label_export"):
        _tt = cfg.get("task_type")
        try:
            if _tt == "tts":
                await _create_label_project_for_run(run_id, cfg, result, redis)
            elif _tt == "llm":
                await _create_llm_label_project_for_run(run_id, cfg, result, redis)
        except Exception as e:  # noqa: BLE001
            await _push_log(redis, run_id, f"[gateway] label export failed: {e}")

    # After a successful run with an HF repo configured, push the best model to
    # Hugging Face from the run's still-alive box (no new pod). LLM self-pushes from
    # its trainer, so skip it here. Best-effort: never fails the run.
    if (status == "done" and (cfg.get("hf_push_repo") or "").strip()
            and (cfg.get("task_type") or "").lower() != "llm"):
        try:
            await _auto_hf_push(run_id, cfg, result, redis)
        except Exception as e:  # noqa: BLE001
            await _push_log(redis, run_id, f"[gateway] HF export failed: {e}")

    if runpod_id and api_key:
        await _terminate_pod(api_key, runpod_id)
        await _push_log(redis, run_id, f"[gateway] pod {runpod_id} torn down")
    elif is_vm and host and key_filename:
        # VM box persists between runs, so sweep the detached run's temp files
        # (runscript/log/pid + the per-run code stage) — else /tmp accrues one set
        # per run. Safe here: the run is finalized and its full log is already
        # uploaded to S3 above. (The runscript removes `stage` itself on a clean
        # exit; this also covers an abnormal exit that left it behind.)
        await _cleanup_remote_run_files((host, port, user, key_filename), run_id)
    _RUN_STATE.pop(run_id, None)
    try:
        await redis.expire(f"train:logs:{run_id}", LOG_LIST_TTL_S)
    except Exception:
        pass


# ---------- lifecycle hooks (mirror bench) ------------------------------


def _remote_run_paths(run_id: str) -> tuple[str, str, str]:
    """(log, pidfile, runscript) on the VM/pod for a detached training run."""
    return (f"/tmp/sgpu_train_{run_id}.log", f"/tmp/sgpu_train_{run_id}.pid", f"/tmp/sgpu_train_{run_id}.sh")


async def _cleanup_remote_run_files(ssh: Optional[tuple], run_id: str) -> None:
    """Best-effort sweep of a finished run's detached temp files on the VM: the
    runscript / log / pidfile (`/tmp/sgpu_train_{id}.{log,pid,sh}`) plus the per-run
    code-staging dir (`/tmp/sgpu_run_{id}`). The runscript removes its own `stage`
    on a clean exit, but the log/pid/runscript deliberately outlive it — the log
    must survive a gateway-down window so a restart can finalize-from-log — and
    would otherwise pile up in /tmp on a long-lived box, one set per run. So this
    runs LATE: only AFTER the run is finalized (terminal) and its log persisted to
    S3. A terminal row is never reconciled, so the remote log is no longer needed.
    No-op for RunPod runs (the pod teardown takes /tmp with it)."""
    if not ssh:
        return
    rlog, rpid, rsh = _remote_run_paths(run_id)
    stage = f"/tmp/sgpu_run_{run_id}"

    def _do() -> None:
        cli = _ssh_connect(ssh[0], int(ssh[1]), ssh[2], ssh[3])
        try:
            _ssh_capture(cli, "rm -rf " + " ".join(shlex.quote(t) for t in (rlog, rpid, rsh, stage)))
        finally:
            try:
                cli.close()
            except Exception:  # noqa: BLE001
                pass

    try:
        await asyncio.to_thread(_do)
    except Exception as e:  # noqa: BLE001
        logger.warning("training %s: remote temp cleanup failed: %s", run_id, e)


def _parse_remote_log(text: str) -> tuple[Optional[int], dict]:
    """Reconstruct (rc, result_json) from a detached run's log file — used to
    finalize a run the gateway stopped streaming (after a restart). Mirrors the
    subset of on_line marker handling that matters for the final record."""
    result: dict = {"epochs": [], "steps": [], "trials": [], "gpu_samples": []}
    rc: Optional[int] = None
    for line in text.splitlines():
        m = re.match(r"\s*@@RC:(-?\d+)", line)
        if m:
            rc = int(m.group(1)); continue
        for tag, key in (("@@METRIC ", "metric"), ("@@STEP ", "step"), ("@@DONE ", "done"),
                         ("@@ARTIFACT ", "artifact"), ("@@ERROR ", "error"),
                         ("@@TRIAL ", "trial"), ("@@PACKED ", "packed")):
            ti = line.find(tag)
            if ti < 0:
                continue
            try:
                obj = json.loads(line[ti + len(tag):])
            except Exception:  # noqa: BLE001
                break
            if key == "metric":
                (result.__setitem__("tts_eval", obj["tts_eval"]) if "tts_eval" in obj
                 else result["epochs"].append(obj))
            elif key == "step":
                result["steps"].append(obj)
            elif key == "done":
                result["best"] = obj.get("best"); result["stopped_early"] = bool(obj.get("stopped_early"))
                if obj.get("trials") is not None:
                    result["trials"] = obj["trials"]
            elif key == "trial":
                t = obj.get("trial")
                if isinstance(t, int) and 0 <= t < len(result["trials"]):
                    result["trials"][t].update(obj)
                else:
                    result["trials"].append(obj)
            elif key == "artifact":
                result["artifact"] = obj
            elif key == "packed":
                result["packed"] = obj
            elif key == "error":
                result["error"] = obj.get("message")
            break
    return rc, result


async def _redispatch_run(redis, run_id: str, reason: str) -> str:
    """A VM run whose gateway-side launch was cut short by a restart — queued, or
    'running' with no remote log (i.e. interrupted during the config-ship / uv-deps /
    launch phases, which run over a live SSH channel BEFORE the trainer detaches) —
    is re-queued + re-run rather than failed, so a gateway restart is transparent.
    Bounded (≤3) to avoid a crash-loop. Non-VM (RunPod) runs are failed instead:
    re-running would silently spawn a new billed pod and the old pod's state is gone."""
    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
        if row is None or row.status in ("done", "cancelled"):
            return row.status if row else "gone"
        prov = await s.get(Provider, row.provider_id) if row.provider_id else None
        rj = dict(row.result_json or {})
        attempts = int(rj.get("restart_relaunches") or 0)
        if prov is None or prov.kind != "vm" or attempts >= 3:
            row.status = "failed"
            row.error_text = (f"orphaned by gateway restart ({reason})"
                              + (f" — gave up after {attempts} relaunch(es)" if attempts else ""))[:4000]
            row.ended_at = datetime.now(timezone.utc)
            orphan_pod = row.runpod_pod_id
            orphan_prov = row.provider_id
            prov_is_runpod = prov is not None and prov.kind == "runpod"
            await s.commit()
            # The old RunPod pod survived the restart and we're NOT re-running it,
            # so it would bill indefinitely with no DB record driving a teardown.
            # Best-effort terminate it now (outside the session — the HTTP call
            # mustn't hold a pooled connection).
            if orphan_pod and prov_is_runpod:
                try:
                    api_key = await compute._resolve_api_key(orphan_prov)
                    await _terminate_pod(api_key, orphan_pod)
                    await _push_log(redis, run_id, f"[gateway] terminated orphaned pod {orphan_pod} (billing stopped)")
                except Exception as e:  # noqa: BLE001 — teardown is best-effort
                    logger.warning("training %s: orphan pod %s teardown failed: %s", run_id, orphan_pod, e)
            return "failed"
        rj["restart_relaunches"] = attempts + 1
        row.result_json = rj
        row.status = "queued"
        row.ended_at = None
        await s.commit()
    await _push_log(redis, run_id, f"[gateway] re-queued after gateway restart ({reason}); relaunch {attempts + 1}/3")
    task = asyncio.create_task(_safe_run(redis, run_id))
    _active_runners[run_id] = task
    task.add_done_callback(lambda _t: _active_runners.pop(run_id, None))
    return "requeued"


async def _apply_log_to_result(run_id: str, full_log: str) -> None:
    """Rebuild result_json (steps/epochs/best/…) from the full remote log and persist
    it, preserving gpu_samples/progress already recorded. Idempotent — the log is the
    source of truth, so re-parsing it each poll never double-counts (unlike appending)."""
    _rc, result = _parse_remote_log(full_log or "")
    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
        if row is None or row.status != "running":
            return
        existing = row.result_json or {}
        # gpu_samples come from a separate sampler (not the log) → keep what we had.
        if existing.get("gpu_samples") and not result.get("gpu_samples"):
            result["gpu_samples"] = existing["gpu_samples"]
        if existing.get("progress") and not result.get("progress"):
            result["progress"] = existing["progress"]
        row.result_json = result
        await s.commit()


async def _resume_orphan_stream(redis, run_id: str) -> None:
    """Re-attach LIVE streaming to a detached run that survived a gateway restart.
    The trainer keeps running (setsid on the box), but the coroutine that tailed its
    log died with the old gateway — so the Logs tab + loss chart FREEZE until the run
    exits (they'd only backfill at finalize). This re-adopts the run: it polls the
    remote log, pushes NEW lines to Redis (live log resumes) and re-parses it into
    result_json (loss/epoch charts resume), then finalizes when the trainer exits.
    Idempotent: only one streamer per run (the caller guards on _active_runners), and
    result_json is REBUILT from the full log each poll so it never double-counts. A few
    log lines emitted during the brief gateway-down window may be absent from the LIVE
    view but are in the saved S3 log. Registered in _active_runners by the caller."""
    rlog, rpid, _sh = _remote_run_paths(run_id)
    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
    if row is None or row.status != "running":
        return
    try:
        ssh = await _resolve_run_ssh(row)
    except Exception:  # noqa: BLE001
        ssh = None
    if ssh is None:
        return  # box unreachable right now — the janitor re-adopts on a later tick
    host, port, suser, key = ssh

    def _probe() -> tuple[bool, str]:
        cli = _ssh_connect(host, int(port), suser, key)
        try:
            alive = "@@A" in _ssh_capture(
                cli, f'kill -0 "$(cat {rpid} 2>/dev/null)" 2>/dev/null && echo @@A || echo @@D')
            return alive, _ssh_capture(cli, f"cat {rlog} 2>/dev/null")
        finally:
            try:
                cli.close()
            except Exception:  # noqa: BLE001
                pass

    try:
        alive, full = await asyncio.to_thread(_probe)
    except Exception:  # noqa: BLE001
        return
    # Anchor the live cursor at the CURRENT end of the log so we don't re-push the
    # (capped) prefix already in Redis; refresh metrics immediately (gap-free).
    seen = len((full or "").splitlines())
    await _push_log(redis, run_id,
                    "[gateway] re-attached live stream after gateway restart "
                    "(a few lines during the restart window may be missing here — the saved log is complete)")
    await _apply_log_to_result(run_id, full)

    while alive:
        await asyncio.sleep(8)
        try:
            alive, full = await asyncio.to_thread(_probe)
        except Exception:  # noqa: BLE001 — transient SSH blip; retry next tick
            continue
        cur = (full or "").splitlines()
        new = cur[seen:]
        if new:
            for ln in new:
                await _push_redis(redis, run_id, ln)  # live log; S3 copy = full log at finalize
            seen = len(cur)
            await _apply_log_to_result(run_id, full)

    # Trainer exited → finalize from the complete log (same logic as _reconcile_orphan).
    rc, result = _parse_remote_log(full or "")
    async with session_factory()() as s:
        r0 = await s.get(TrainingRun, run_id)
        existing = (r0.result_json or {}) if r0 else {}
    if existing.get("gpu_samples") and not result.get("gpu_samples"):
        result["gpu_samples"] = existing["gpu_samples"]
    _ok = (rc == 0) or (result.get("best") is not None) or bool(result.get("artifact"))
    status = "done" if (_ok and not result.get("error")) else "failed"
    err = result.get("error") if status == "failed" else None
    if status == "failed" and not err:
        err = f"trainer exited with code {rc}" if rc is not None else "trainer process gone (no exit code captured)"
    await _finalize(run_id, status, rc, err, result_json=result)
    await _push_log(redis, run_id, f"[gateway] finalized from log after restart: {status} (rc={rc})")
    try:
        s3_target = await _training_s3_target(row.storage_id)
        s3_put_text((row.s3_prefix or "") + "logs.txt", full or "", target=s3_target)
    except Exception as e:  # noqa: BLE001
        logger.warning("training %s: resume-finalize log upload failed: %s", run_id, e)
    await _cleanup_remote_run_files((host, port, suser, key), run_id)
    # RunPod pod (if this was a cloud run) keeps billing until torn down — the
    # live path does this inline; the restart-finalize path must too.
    await _teardown_run_pod_if_any(row, redis)


async def _reconcile_orphan(row, redis) -> str:
    """Reconcile one orphaned (gateway-untracked) training run over SSH: the
    detached trainer survives a gateway restart, so check its pid — still alive →
    leave it running (finalized later when it exits); exited with a log → finalize
    from it; interrupted before it detached (no log) → re-queue + relaunch; box
    transiently unreachable → leave running for the janitor to re-probe."""
    rlog, rpid, _sh = _remote_run_paths(row.id)
    try:
        ssh = await _resolve_run_ssh(row)
    except Exception:  # noqa: BLE001
        ssh = None
    if ssh is None:
        # Can't reach the box right now (e.g. SSH/tunnel not up yet just after a
        # restart). Don't fail — leave it running so the janitor re-probes; a truly
        # dead box stays 'running' until it returns or the user terminates.
        return "running"
    host, port, user, key = ssh

    def _probe() -> tuple[bool, str]:
        cli = _ssh_connect(host, int(port), user, key)
        try:
            alive = "@@A" in _ssh_capture(cli, f'kill -0 "$(cat {rpid} 2>/dev/null)" 2>/dev/null && echo @@A || echo @@D')
            return alive, _ssh_capture(cli, f"cat {rlog} 2>/dev/null")
        finally:
            try:
                cli.close()
            except Exception:  # noqa: BLE001
                pass
    try:
        alive, log = await asyncio.to_thread(_probe)
    except Exception:  # noqa: BLE001 — transient SSH error → retry next janitor tick
        return "running"
    if alive:
        # Detached trainer survived the restart — re-attach LIVE streaming so the Logs
        # tab + loss chart resume (they'd otherwise freeze until the run exits). One
        # streamer per run: register the task in _active_runners here (synchronously,
        # before the coroutine first runs) so a concurrent reconcile / janitor tick
        # can't double-adopt; the streamer finalizes the run when the trainer exits.
        if row.id not in _active_runners:
            t = asyncio.create_task(_resume_orphan_stream(redis, row.id))
            _active_runners[row.id] = t
            t.add_done_callback(
                lambda _t, rid=row.id: (_active_runners.pop(rid, None)
                                        if _active_runners.get(rid) is _t else None))
        return "running"  # detached trainer survived; the streamer finalizes it on exit
    if not (log or "").strip():
        # No process AND no log → the run was cut short before the trainer detached
        # (during config/deps/launch). Re-run it instead of losing it.
        return await _redispatch_run(redis, row.id, "process gone, no log")
    rc, result = _parse_remote_log(log)
    # @@DONE (best) or @@ARTIFACT means the trainer finished + uploaded → treat as
    # done even if the exit code wasn't captured (e.g. a tail race on a detached run).
    _ok = (rc == 0) or (result.get("best") is not None) or bool(result.get("artifact"))
    status = "done" if (_ok and not result.get("error")) else "failed"
    err = result.get("error") if status == "failed" else None
    if status == "failed" and not err:
        err = f"trainer exited with code {rc}" if rc is not None else "trainer process gone (no exit code captured)"
    await _finalize(row.id, status, rc, err, result_json=result)
    await _push_log(redis, row.id, f"[gateway] finalized from log after restart: {status} (rc={rc})")
    # The live path uploads logs.txt from its local copy, but a restart-finalized
    # run never streamed the tail — the remote log is the ONLY complete copy. Push
    # it to S3 before sweeping the run's temp files (best-effort: never block the
    # finalize/cleanup on it).
    try:
        s3_target = await _training_s3_target(row.storage_id)
        s3_put_text((row.s3_prefix or "") + "logs.txt", log, target=s3_target)
    except Exception as e:  # noqa: BLE001
        logger.warning("training %s: restart-finalize log upload failed: %s", row.id, e)
    await _cleanup_remote_run_files((host, port, user, key), row.id)
    # Tear down the RunPod pod (cloud run) so it stops billing after this
    # finalize-from-log — the live path does it inline; this path must too.
    await _teardown_run_pod_if_any(row, redis)
    return status


async def cleanup_orphaned_running(redis) -> int:
    """On startup, reconcile running/queued rows. Detached trainers survive a
    gateway restart, so each is SSH-probed: alive → kept running (the janitor
    finalizes it when it exits); exited → finalized from its log; unreachable →
    failed. Returns the count moved to a terminal state."""
    async with session_factory()() as s:
        rows = (await s.execute(
            select(TrainingRun).where(TrainingRun.status.in_(["running", "queued"]))
        )).scalars().all()
    n = 0
    for row in rows:
        # A multi-node sweep PARENT has no provider_id of its own (each trial has
        # its own) — _reconcile_orphan would just no-op "can't reach the box" on it
        # forever. Its dispatcher is a plain in-process asyncio task (unlike a
        # single run, which the generic SSH-reconcile below can finalize-from-log),
        # so a restart kills it outright while the already-dispatched trials keep
        # training untouched (each IS a normal single run, reconciled below like
        # any other). Route it to run_multi_node_sweep(resume=True) instead, which
        # re-adopts result_json["trials"] rather than resetting/double-dispatching.
        if row.status == "running" and (row.config_json or {}).get("nodes"):
            if row.id not in _active_runners:
                t = asyncio.create_task(_safe_run_multi_node(redis, row.id, resume=True))
                _active_runners[row.id] = t
                t.add_done_callback(
                    lambda _t, rid=row.id: (_active_runners.pop(rid, None)
                                            if _active_runners.get(rid) is _t else None))
            continue
        # Queued rows never launched a detached process → re-run them (a restart
        # mid-queue shouldn't drop the run). _redispatch_run is bounded + VM-only.
        if row.status == "queued":
            await _redispatch_run(redis, row.id, "never started")
            n += 1
            continue
        if (await _reconcile_orphan(row, redis)) not in ("running", "requeued"):
            n += 1
    # A gateway restart loses the in-memory try-it setup tasks, so any surviving
    # cloud try-it pod is orphaned — terminate them all (the session is over).
    try:
        await _reap_idle_tryit_pods(force_all=True)
    except Exception as e:  # noqa: BLE001
        logger.warning("try-it startup reaper: %s", e)
    # Un-stick label/HF exports left "running" by the restart (their task is gone).
    try:
        await _reconcile_orphaned_exports(redis)
    except Exception as e:  # noqa: BLE001
        logger.warning("orphaned-export reconcile: %s", e)
    return n


async def _reconcile_orphaned_exports(redis) -> None:
    """On startup, un-stick a label/HF export left 'running' by a gateway restart:
    its in-memory task is gone, but result_json.{label,hf}_export.status stays
    'running' → the UI sticks on 'exporting to Label' / 'pushing to HF' forever. Mark
    such orphans failed and tear down any orphaned cloud label pod so it can't bill."""
    try:
        async with session_factory()() as s:
            rows = (await s.execute(select(TrainingRun).where(or_(
                TrainingRun.result_json["label_export"]["status"].as_string() == "running",
                TrainingRun.result_json["hf_export"]["status"].as_string() == "running",
            )))).scalars().all()
    except Exception as e:  # noqa: BLE001 — JSON-path query unsupported? skip (never break boot)
        logger.warning("training: orphaned-export reconcile query failed: %s", e)
        return
    for row in rows:
        rj = dict(row.result_json or {})
        changed: list[str] = []
        for k in ("label_export", "hf_export"):
            st = rj.get(k)
            if isinstance(st, dict) and st.get("status") == "running":
                rj[k] = {**st, "status": "failed", "error": "interrupted by a gateway restart"}
                changed.append(k)
        if not changed:
            continue
        async with session_factory()() as s:
            r2 = await s.get(TrainingRun, row.id)
            if r2 is not None:
                r2.result_json = rj
                await s.commit()
        if "label_export" in changed:
            cfg = row.config_json or {}
            prov_id = (cfg.get("label_provider_id") or "").strip() or row.provider_id
            if await _reap_label_pod(row.id, prov_id):
                try:
                    await _push_log(redis, row.id, "[gateway] label export: orphaned pod torn down on restart")
                except Exception:  # noqa: BLE001
                    pass
        if "hf_export" in changed:
            he = rj.get("hf_export") or {}
            if he.get("pod_id") or he.get("run_on") == "cloud":
                if await _reap_hf_export_pod(row.id, he.get("provider_id")):
                    try:
                        await _push_log(redis, row.id, "[gateway] HF export: orphaned pod torn down on restart")
                    except Exception:  # noqa: BLE001
                        pass
            # Close a leaked reverse tunnel (autossh survives the gateway restart, retrying the
            # dead pod forever) — the export pod host is recorded in the export state.
            if he.get("tunnel_host"):
                try:
                    from . import vm_tunnel
                    await asyncio.to_thread(vm_tunnel.close, he["tunnel_host"])
                except Exception:  # noqa: BLE001
                    pass
        try:
            await _push_log(redis, row.id, "[gateway] export was interrupted by a gateway restart — marked failed")
        except Exception:  # noqa: BLE001
            pass


async def training_janitor_loop(redis, interval: int = 90) -> None:
    """Finalize runs that survived a gateway restart (left 'running' by cleanup)
    once their detached trainer exits. Skips runs this gateway is actively
    managing (they finalize themselves)."""
    while True:
        try:
            await asyncio.sleep(interval)
            async with session_factory()() as s:
                rows = (await s.execute(
                    select(TrainingRun).where(TrainingRun.status == "running")
                )).scalars().all()
            for row in rows:
                if row.id in _active_runners:
                    continue  # actively run by THIS gateway (VM or pod) → it finalizes
                await _reconcile_orphan(row, redis)
            # Reap idle / orphaned cloud try-it pods so they can't bill forever.
            try:
                n = await _reap_idle_tryit_pods()
                if n:
                    logger.info("training janitor: reaped %d idle try-it pod(s)", n)
            except Exception as e:  # noqa: BLE001
                logger.warning("try-it reaper: %s", e)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("training janitor: %s", e)


# ---------- schemas ------------------------------------------------------


class NodeSpec(BaseModel):
    """One (provider, GPU-range) target for a multi-node sweep. `visible_devices`
    empty/None -> resolve to ALL of that provider's GPUs at create time."""
    provider_id: str
    visible_devices: Optional[str] = None


class CreateTrainingRunRequest(BaseModel):
    name: str
    dataset_id: str
    base_model: str
    task_type: str = "asr"             # "asr" | "tts" | "llm"
    test_dataset_id: Optional[str] = None
    # ---- TTS-only (Qwen3 + NeuCodec) ----
    tokenizer: Optional[str] = None    # pack tokenizer (speech tokens); default set in runner
    block_size: int = 10240            # training context length
    pack_sequence_length: int = 4096   # per-utterance pack length
    default_speaker: Optional[str] = None
    speaker_field: Optional[str] = None
    # Pack-only TTS dataset transform: run convert_neucodec + pack_stage1 and
    # upload the ChiniDataset shards (no training) — the gateway then creates a
    # packed dataset. Set internally by the datasets "Pack for TTS" endpoint.
    pack_only: bool = False
    pack_source_dataset_id: Optional[str] = None
    # ---- hyperparameter sweep ----
    # {param: [values]} → cross-product = trials, run in a GPU-pinned pool on one
    # box (concurrency = #gpus / gpus_per_trial). Empty = single run.
    sweep: dict = {}
    gpus_per_trial: int = 1
    # Multi-node sweep: several (provider, GPU-range) targets instead of one
    # provider_id/visible_devices pool. Each node contributes floor(its GPU
    # count / gpus_per_trial) concurrent trial slots; trials are drawn from a
    # shared queue as slots free up. Each trial is still a normal single-node
    # run (NO cross-host distributed training) — this only fans a sweep's
    # grid out across boxes. None/empty -> the existing single-box sweep path.
    nodes: Optional[list[NodeSpec]] = None
    # Internal: set by run_multi_node_sweep on each per-trial child body it creates
    # (never sent by a real client) — marks this run as a child of a multi-node sweep
    # so the list endpoints can hide it by default (the parent row is what users browse;
    # children stay individually reachable via GET /{run_id} and the parent's trials[]).
    sweep_parent_id: Optional[str] = None
    # Hyperparams + split + eval settings (all optional; trainer has defaults).
    eval_metric: str = "wer"           # "wer" | "cer" (ASR only)
    # Normalize text (case/punctuation/numbers) before WER/CER, Whisper-style.
    # Off = score raw text (much higher, but exact-match strict).
    normalize_text: bool = True
    max_epochs: int = 3
    # Hard cap on optimizer steps; >0 overrides max_epochs (HF `max_steps`). 0 =
    # run the full epochs. Handy for quick debug runs (e.g. max_steps=50).
    max_steps: int = 0
    # Eval + checkpoint cadence: "epoch" (default) or "steps". In "steps" mode,
    # evaluate + checkpoint every eval_steps / save_steps optimizer steps. Kept
    # equal by the form so HF's load_best_model_at_end constraint (Whisper) holds.
    eval_strategy: str = "epoch"
    eval_steps: int = 500
    save_strategy: str = "epoch"
    save_steps: int = 500
    patience: int = 0                  # 0 = no early stop
    # Disable evaluation entirely: train on ALL data with no held-out test set, no
    # eval loss / WER-CER, no best-checkpoint selection, no early stopping. The
    # form's "No test set" option. test_dataset_id should be null when set.
    no_eval: bool = False
    eval_split_pct: float = 10.0
    split_seed: int = 42
    batch_size: int = 8
    grad_accum: int = 1
    # LLM FSDP CPU offload (params/optimizer in host RAM: big VRAM saver, PCIe-slow).
    # None = per-arch default (gemma/qwen ON, minimax/mistral OFF); the form sets it.
    cpu_offload: Optional[bool] = None
    # LLM (gemma4 + qwen3.5/3.6): context parallelism — shards one packed sequence across the CP
    # group so context longer than a single GPU's VRAM can train. Needs >=2 GPUs.
    context_parallel: Optional[bool] = None
    # CP group size (GPUs that jointly shard ONE sequence). None/0 → all run GPUs (dp=1). When set,
    # the run's GPU count must be a multiple of it; data parallelism runs across the CP groups
    # (dp_size = world / cp_size). Only meaningful when context_parallel is on.
    cp_size: Optional[int] = None
    learning_rate: float = 1e-5
    warmup_steps: int = 0
    # HF LR schedule: linear (warmup→linear decay, the default), cosine,
    # constant_with_warmup (warmup→hold), or constant (no warmup/decay).
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.0
    # LoRA / PEFT — train low-rank adapters on the attention projections instead
    # of the full model (less VRAM, faster). Merged into the base at save time
    # so the artifact is a drop-in Whisper checkpoint (no peft to load/serve).
    use_lora: bool = False
    lora_r: int = 16
    # alpha is conventionally a ratio of r (2× common). When set, alpha is derived
    # as round(r × ratio) so a LoRA-r sweep carries alpha along (no permutation).
    # `lora_alpha` is the absolute fallback when no ratio is given.
    lora_alpha_ratio: Optional[float] = 2.0
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # task_type=llm + gemma only: True = FA4 head_dim-512 cute fork (default, faster,
    # needs a CUDA-13 host); False = FA3 wheel + dynamic_attention (SDPA-tiled, runs on
    # the standard cu12.x RunPod host). Ignored by minimax/mistral (always FA3).
    gemma_fa4: bool = True
    # LLM-only training objective: "sft" (default — supervised finetune over a
    # kind=llm_packed dataset) or "dpo" (Direct Preference Optimization over a
    # kind=llm_dpo_packed preference-pair dataset; fused multipacked DPO loss, the
    # frozen reference = base with LoRA disabled). DPO is wired for the qwen trainer
    # only and is incompatible with context_parallel.
    training_type: str = "sft"
    dpo_beta: float = 0.1          # DPO temperature β on the log-ratio (training_type=dpo)
    # LLM-only (gemma4): which linear projections to apply LoRA to. Default is the
    # attention projections (q/k/v/o); add MLP/dense layers (gate_proj, up_proj,
    # down_proj) to adapt those too. LLM finetune is always LoRA. Unknown names are
    # dropped; an empty result falls back to the q/k/v/o default.
    lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # LLM-only (gemma4): also FULL-train the token embeddings + LM head (gemma-4 ties them,
    # so this is one ~1.4B weight), not just LoRA. Helps the finetune reliably learn to emit
    # special tokens (e.g. <|tool_call>) that attention-only LoRA can only nudge via hidden
    # states. Costs extra optimizer state (gemma cpu_offload default is ON). Pairs well with
    # adding the MLP (gate_proj,up_proj,down_proj) to lora_target_modules.
    train_embeddings: bool = False
    # LLM-only: use DoRA (weight-decomposed LoRA) instead of plain LoRA for every adapted module
    # (attention + the MoE experts on minimax/mistral/qwen-MoE/gemma-MoE). DoRA adds a trainable
    # per-output-row magnitude and takes the direction from W+ΔW. Incompatible with DPO (the
    # LoRA-disabled reference assumes the base is untouched). All LLM archs.
    use_dora: bool = False
    # LLM MoE-only: skip the fused routed-expert adapter (adapt attention only). The experts are
    # 3D tensors (not nn.Linear), so they're NOT in lora_target_modules — they're adapted by
    # default on minimax/mistral/qwen-MoE/gemma-MoE; set True to opt out. No effect on dense
    # models or nemotron (whose MoE experts are always frozen).
    no_moe_lora: bool = False
    # Freeze the encoder; train the decoder only (faster, less overfit on small data).
    freeze_encoder: bool = False
    # Multi-GPU strategy for a single (non-sweep) run: DDP via torchrun (one
    # process/GPU, default) vs HF's implicit DataParallel. Ignored on 1 GPU and
    # for sweeps (those pin gpus_per_trial per trial).
    use_ddp: bool = True
    # Emit a training-loss point every N optimizer steps (@@STEP) for the live
    # loss curve. Smaller N = smoother graph, more log lines.
    logging_steps: int = 10
    # Audio augmentation on TRAINING audio only — names from AUG_TECHNIQUES
    # (telephone/noise/dropout/gain/pitch/speed/reverb/bandpass). Empty = off.
    # One enabled technique is picked at random per augmented sample.
    augment_techniques: list[str] = []
    augment_prob: float = 0.5
    # ASR-only: SpecAugment — time/feature masking on the mel input features during
    # training (HF apply_spec_augment). Complements the waveform augmentations above;
    # the standard Whisper-finetune regularizer for small corpora.
    spec_augment: bool = False
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_feature_prob: float = 0.0
    mask_feature_length: int = 10
    # TTS-only: audio eval methods to run on the test set (cer | mos | similarity).
    eval_methods: list[str] = []
    # TTS-only: how many generated clips the heavy eval scores (per method). The
    # gen + NeuCodec-decode + scorer pass dominates a short run, so a small count
    # keeps debug runs fast. Default 64.
    eval_max_samples: int = 64
    # OmniVoice-only knobs (base model = k2-fsa/OmniVoice). language_id tagging for
    # the Higgs manifests + the held-out eval split + trainer attn/batch.
    default_language: Optional[str] = "en"     # language_id when no per-row field
    language_field: Optional[str] = None       # dataset column holding language_id
    eval_test_per_speaker: int = 25            # held-out test clips/speaker (no explicit test split)
    attn_implementation: Optional[str] = "flex_attention"  # set "sdpa" if flex_attention won't build
    batch_tokens: Optional[int] = None         # OmniVoice token-packing batch budget (default 8192)
    # TTS-only: after a successful run, synthesize N clips from the trained model
    # and auto-create a Label-platform *recording* project (with MOS rating) seeded
    # with them, for human listening / MOS. Texts come from the held-out test split
    # if present, else a random sample of the train split. VM-provider runs only
    # (synthesis needs the box). The token is Fernet-encrypted into the run config —
    # never stored or returned in plaintext.
    label_export: bool = False
    label_base_url: str = "http://localhost:3002"
    label_base_url_secret: Optional[str] = None  # GlobalEnv key holding the URL (wins over label_base_url)
    label_token: Optional[str] = None            # lpat_… PAT (admin); never persisted raw
    label_token_secret: Optional[str] = None     # GlobalEnv key holding the lpat (wins over label_token)
    label_project_name: Optional[str] = None     # defaults to "<run name>-eval"
    label_samples: int = 32
    label_mos_axes: list[str] = ["Naturalness", "Intelligibility", "Noise"]
    # Balance the synthesized eval clips across these speaker names (round-robin),
    # e.g. ["A","B"] + 32 samples → 16 each. Empty → original packed voices.
    label_speakers: list[str] = []
    # Prefix each Label task's transcription with the speaker name ("spk: text").
    label_speaker_prefix: bool = False
    # Drop text samples whose transcript contains any of these phrases (case-
    # insensitive, whitespace-collapsed — so "E M G S" matches any spacing).
    label_reject_keywords: list[str] = []
    # With label_speakers: make a SEPARATE Label project per speaker, each seeded
    # only from that speaker's own clips (label_samples is then per speaker). Else
    # the speakers are round-robin'd into one combined project.
    label_per_speaker: bool = False
    # ---- LLM label export (task_type=llm only) ----
    # kind=hf benchmark dataset (uploaded via HF push to the platform catalog) whose
    # user-turn messages are used as prompts for generation after training completes.
    llm_label_eval_dataset_id: Optional[str] = None
    llm_label_samples: Optional[int] = None          # default: all rows in the dataset
    llm_label_max_new_tokens: Optional[int] = None   # default: 512
    llm_label_mos_axes: Optional[list[str]] = None   # default: Relevance/Accuracy/Helpfulness/Tone
    precision: str = "fp32-bf16"        # "<load>-<amp>", e.g. fp32-bf16
    language: Optional[str] = None
    task: str = "transcribe"
    # Where to run.
    provider_id: Optional[str] = None  # kind=vm → bare metal; else RunPod
    gpu_type: str = "NVIDIA L40S"
    gpu_count: int = 1
    secure_cloud: bool = True
    data_center_id: Optional[str] = None  # RunPod region pin; blank/None → auto
    disk_gb: int = 60
    volume_gb: int = 80
    # RunPod pod image. Sets the CUDA host (the allowedCudaVersions filter is parsed
    # from the image tag): the FA4 gemma stack needs a CUDA-13 image (e.g.
    # runpod/pytorch:1.0.6-cu1300-torch291-ubuntu2404); blank → DEFAULT_IMAGE (cu1281).
    image: Optional[str] = None
    visible_devices: Optional[str] = None
    storage_id: Optional[str] = None
    hf_push_repo: Optional[str] = None
    # HF token for the run (gated/private datasets + push to Hub). A Secrets-page key
    # reference (hf_token_secret) wins; else the pasted token, stored Fernet-encrypted
    # — never raw. Resolved into cfg["hf_token"] at run time. Mirrors serverless/new.
    hf_token: Optional[str] = None
    hf_token_secret: Optional[str] = None
    # Roomy dir on the remote for checkpoints + temp (TMPDIR). Defaults to
    # /share (the VM's big volume); /tmp is a small disk that overflows on big
    # models. The best model is uploaded to S3 regardless.
    work_dir: str = "/share"
    # Isolated uv venv for the trainer's deps (like serverless's vLLM venv_path) —
    # keeps the heavy stack off the box's system python. Default per task:
    # /share/autotrain-whisper (asr) or /share/autotrain-tts (tts).
    venv_path: Optional[str] = None

    @field_validator("work_dir", "venv_path")
    @classmethod
    def _safe_paths(cls, v, info):  # noqa: N805
        # These land in remote shell commands ({venv}/bin/python, rm -rf {work_dir}).
        return validate_path_field(v, info.field_name)
    # rm the run's checkpoint/work dir on the remote once the run ends (the best
    # model is already on S3). Keeps the volume from filling across runs.
    cleanup_checkpoints: bool = True
    # Experiment tracking. report_to is a subset of ["mlflow", "wandb"]. Only
    # non-secret per-run knobs live here; the creds (WANDB_API_KEY,
    # MLFLOW_TRACKING_URI/USERNAME/PASSWORD) come from the global Secrets page.
    report_to: list[str] = []
    # Named tracking credentials (Secrets page → Tracking credentials card).
    # Selecting one enables that tracker for the run.
    wandb_credential_id: Optional[str] = None
    mlflow_credential_id: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: Optional[str] = None
    # OS env vars exported on the remote before the trainer runs (HOME, cache
    # dirs, …). Absolute-path values are auto-mkdir'd. Must be process env —
    # libs read HOME/HF_HOME/XDG_CACHE_HOME at import time, so a config field
    # alone wouldn't take. Keys validated as shell-safe env names on create.
    env_vars: dict[str, str] = {}


class TrainingRunRecord(BaseModel):
    id: str
    name: str
    status: str
    dataset_id: str
    test_dataset_id: Optional[str] = None
    base_model: str
    task_type: str = "asr"
    s3_prefix: str
    config_json: dict
    exit_code: Optional[int] = None
    error_text: Optional[str] = None
    result_json: Optional[dict] = None
    created_by: str
    created_at: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    cost_per_hr: Optional[float] = None
    provider_id: Optional[str] = None
    provider_name: Optional[str] = None
    provider_kind: Optional[str] = None  # "vm" | "runpod" | …
    storage_id: Optional[str] = None
    storage_name: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_count: int = 1
    visible_devices: Optional[str] = None


class TrainingFile(BaseModel):
    name: str
    size: int
    modified: str
    download_url: str


# ---------- HTTP API -----------------------------------------------------


router = APIRouter(prefix="/v1/training-runs", tags=["autotrain"])


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _best_from_epochs(result: Optional[dict], config: Optional[dict]) -> Optional[dict]:
    """The genuinely-best epoch, chosen by the run's selection metric — always
    LOWEST wins (WER/CER for ASR, eval_loss for TTS/LLM). Derived from the same
    per-epoch points the metrics table shows, so the headline "best" and the table
    can never disagree.

    Why recompute on read: the trainer's own `best` came from a final
    trainer.evaluate() of whatever weights load_best_model_at_end left loaded, so a
    run trained before greater_is_better was pinned False (HF's default treated a
    higher WER as "better") reported the WORST epoch as "best". Recomputing here
    fixes those finished runs without a re-run. Returns None for sweeps (best is
    per-trial, not per-epoch) or when no epoch carries the metric — caller falls
    back to the stored value."""
    result = result or {}
    if result.get("trials"):            # sweep: keep the sweep-level best trial
        return None
    epochs = result.get("epochs") or []
    if not epochs:
        return None
    task = str((config or {}).get("task_type") or "").lower()
    metric = "eval_loss" if task in ("tts", "llm") \
        else str((config or {}).get("eval_metric") or "wer").lower()
    scored = [e for e in epochs if isinstance(e.get(metric), (int, float))]
    if not scored:
        return None
    b = min(scored, key=lambda e: e[metric])
    return {
        "epoch": int(round(float(b.get("epoch") or 0))),
        "wer": b.get("wer"),
        "cer": b.get("cer"),
        "eval_loss": b.get("eval_loss"),
    }


def _to_record(
    row: TrainingRun, owner_username: str,
    provider_name: Optional[str] = None, provider_kind: Optional[str] = None,
    storage_name: Optional[str] = None,
) -> TrainingRunRecord:
    result_json = row.result_json
    if result_json:
        corrected = _best_from_epochs(result_json, row.config_json)
        if corrected is not None and corrected != result_json.get("best"):
            result_json = {**result_json, "best": corrected}
    return TrainingRunRecord(
        id=row.id, name=row.name, status=row.status, dataset_id=row.dataset_id,
        test_dataset_id=row.test_dataset_id, base_model=row.base_model,
        task_type=row.task_type,
        s3_prefix=row.s3_prefix, config_json=row.config_json or {},
        exit_code=row.exit_code, error_text=row.error_text, result_json=result_json,
        created_by=owner_username, created_at=_iso(row.created_at) or "",
        started_at=_iso(row.started_at), ended_at=_iso(row.ended_at),
        cost_per_hr=row.cost_per_hr, provider_id=row.provider_id,
        provider_name=provider_name, provider_kind=provider_kind,
        storage_id=row.storage_id, storage_name=storage_name,
        gpu_type=row.gpu_type, gpu_count=row.gpu_count, visible_devices=row.visible_devices,
    )


def _parse_gpu_indices(visible_devices: Optional[str]) -> list[int]:
    """Parse a CUDA_VISIBLE_DEVICES string into GPU indices, rejecting anything
    that isn't a non-negative integer or that repeats — with a clear 400."""
    toks = [t.strip() for t in (visible_devices or "").split(",") if t.strip()]
    ids: list[int] = []
    for t in toks:
        if not re.fullmatch(r"\d+", t):
            raise HTTPException(
                status_code=400,
                detail=f"visible_devices: '{t}' is not a valid GPU index — use non-negative integers like 0,1",
            )
        ids.append(int(t))
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise HTTPException(status_code=400, detail=f"visible_devices has duplicate GPU indices: {dup}")
    return ids


def _validate_sweep(sweep: dict) -> None:
    """Reject non-numeric sweep values up front (learning_rate must be positive
    numbers; the count-like knobs positive integers) so a bad cell can't silently
    drop a trial or blow up mid-run."""
    for v in (sweep.get("learning_rate") or []):
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"sweep learning_rate: {v!r} is not a number like 1e-4")
        if not math.isfinite(f) or f <= 0:
            raise HTTPException(status_code=400, detail=f"sweep learning_rate: {v!r} must be a positive number like 1e-4")
    for v in (sweep.get("weight_decay") or []):  # 0 is valid for weight decay
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"sweep weight_decay: {v!r} is not a number")
        if not math.isfinite(f) or f < 0:
            raise HTTPException(status_code=400, detail=f"sweep weight_decay: {v!r} must be a non-negative number")
    for key in ("batch_size", "grad_accum", "max_epochs", "max_steps", "block_size", "lora_r", "lora_alpha"):
        for v in (sweep.get(key) or []):
            try:
                n = int(v)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"sweep {key}: {v!r} must be a positive integer")
            if n <= 0:
                raise HTTPException(status_code=400, detail=f"sweep {key}: {v!r} must be a positive integer")
    for v in (sweep.get("freeze_encoder") or []):
        if str(v).lower() not in ("on", "off", "true", "false", "1", "0", "yes", "no"):
            raise HTTPException(status_code=400, detail=f"sweep freeze_encoder: {v!r} must be on/off")
    for v in (sweep.get("use_lora") or []):
        if str(v).lower() not in ("on", "off", "true", "false", "1", "0", "yes", "no"):
            raise HTTPException(status_code=400, detail=f"sweep use_lora: {v!r} must be on/off")


async def _create_and_launch_run(
    body: CreateTrainingRunRequest, user: User, session: AsyncSession, redis,
) -> TrainingRun:
    """Validate + insert + launch ONE training run. Shared by the HTTP endpoint
    (create_training_run) and the multi-node sweep dispatcher (_run_multi_node_sweep),
    which calls this once per trial with a node-specific provider_id/visible_devices
    and that trial's swept hyperparams overlaid — each trial is a normal single
    (non-sweep) run under the hood, just launched programmatically instead of over HTTP."""
    ds = await session.get(Dataset, body.dataset_id)
    if ds is None or (ds.owner_id != user.id and not user.is_admin):
        raise HTTPException(status_code=400, detail="unknown dataset_id")
    tds = None
    if body.test_dataset_id:
        tds = await session.get(Dataset, body.test_dataset_id)
        if tds is None or (tds.owner_id != user.id and not user.is_admin):
            raise HTTPException(status_code=400, detail="unknown test_dataset_id")
    # TTS trains directly on pre-packed (tts_packed) ChiniDataset shards — convert
    # + multipack already happened on the Datasets page. Block size + tokenizer
    # are derived from the packed dataset / base model, not asked in the form.
    # A pack-only run is the EXCEPTION: it consumes a non-packed {audio,
    # transcription, speaker} source to PRODUCE the packed dataset, so it must
    # not require kind=tts_packed (that would make 'Pack for TTS' reject itself).
    if body.task_type == "tts" and not body.pack_only:
        # OmniVoice consumes its own Higgs WebDataset shards (kind=omnivoice_packed);
        # the Qwen3+NeuCodec path consumes kind=tts_packed.
        if _tts_arch(body.base_model) == "omnivoice":
            if ds.kind != "omnivoice_packed":
                raise HTTPException(
                    status_code=400,
                    detail="OmniVoice training needs a kind=omnivoice_packed dataset — pack it first via 'Pack for OmniVoice'",
                )
            if tds is not None and tds.kind != "omnivoice_packed":
                raise HTTPException(status_code=400, detail="OmniVoice test dataset must also be kind=omnivoice_packed")
        else:
            if ds.kind != "tts_packed":
                raise HTTPException(
                    status_code=400,
                    detail="TTS training needs a packed dataset (kind=tts_packed) — pack it first via 'Pack for TTS'",
                )
            if tds is not None and tds.kind != "tts_packed":
                raise HTTPException(status_code=400, detail="TTS test dataset must also be packed (kind=tts_packed)")
            # block size follows the packed dataset's sequence_length.
            _pack_meta = (ds.split_fields or {}).get("_tts_pack") or {}
            _seq = _pack_meta.get("sequence_length")
            if _seq:
                body.block_size = int(_seq)
        # the trainer tokenizes with the base model's tokenizer; record it.
        body.tokenizer = body.base_model
    if body.task_type == "llm":
        # LLM finetune consumes a pre-packed chat ChiniDataset — kind=llm_packed for
        # SFT, kind=llm_dpo_packed (preference pairs, chosen-first bins) for DPO. The
        # trainer (gemma4.py / minimax_m2.py / qwen3_5.py, chosen by base-model arch)
        # reads the packed ids directly (no re-tokenization).
        _dpo = (body.training_type or "sft") == "dpo"
        if body.training_type not in ("sft", "dpo"):
            raise HTTPException(status_code=400, detail="training_type must be 'sft' or 'dpo'")
        _want_kind = "llm_dpo_packed" if _dpo else "llm_packed"
        if ds.kind != _want_kind:
            raise HTTPException(
                status_code=400,
                detail=(f"{'DPO' if _dpo else 'LLM'} training needs a packed dataset (kind={_want_kind}) — "
                        f"pack it first via 'Pack for LLM'{' with the DPO objective' if _dpo else ''}"),
            )
        if tds is not None and tds.kind != _want_kind:
            raise HTTPException(status_code=400, detail=f"LLM test dataset must also be packed (kind={_want_kind})")
        if _dpo:
            if _llm_arch(body.base_model) not in ("qwen", "gemma"):
                raise HTTPException(
                    status_code=400,
                    detail="DPO training is currently wired for qwen (Qwen3.5/3.6) and gemma-4 base models only",
                )
            if body.context_parallel:
                raise HTTPException(
                    status_code=400,
                    detail="DPO is incompatible with context parallelism — per-sequence log-probs "
                           "would need cross-rank reduction; turn one of them off",
                )
            if not math.isfinite(body.dpo_beta) or body.dpo_beta <= 0:
                raise HTTPException(status_code=400, detail=f"dpo_beta must be a positive number (got {body.dpo_beta!r})")
            if body.train_embeddings:
                raise HTTPException(
                    status_code=400,
                    detail="train_embeddings is incompatible with DPO — the LoRA-disabled "
                           "reference assumes the base weights (incl. embeddings/lm_head) stay "
                           "frozen; training them makes the reference drift. Turn one off.",
                )
            if body.use_dora:
                raise HTTPException(
                    status_code=400,
                    detail="DoRA is incompatible with DPO (v1) — the LoRA-disabled reference "
                           "assumes the base weights stay frozen, but DoRA retrains a per-row "
                           "magnitude on top of the base. Turn one off.",
                )
        _pm = (ds.split_fields or {}).get("_llm_pack") or {}
        if _pm.get("sequence_length"):
            body.block_size = int(_pm["sequence_length"])
        # The dataset was tokenized at pack time with a specific tokenizer/arch; the
        # trainer reads those ids verbatim, so the base model's arch MUST match the
        # pack arch (a gemma-packed dataset trained as minimax = garbage ids).
        _pack_arch = _pm.get("arch")
        _model_arch = _llm_arch(body.base_model)
        if _pack_arch and _pack_arch in ("gemma", "minimax", "mistral", "qwen", "nemotron") and _pack_arch != _model_arch:
            raise HTTPException(
                status_code=400,
                detail=(f"base model arch '{_model_arch}' ≠ dataset pack arch '{_pack_arch}'. "
                        f"This dataset was packed with a {_pack_arch} tokenizer — pick a {_pack_arch} "
                        f"base model, or re-pack with a {_model_arch} tokenizer."),
            )
        # tokenization already happened at pack time; record the base model.
        body.tokenizer = body.base_model
    if body.storage_id:
        st = await session.get(Storage, body.storage_id)
        if st is None or st.kind != "s3" or not st.enabled:
            raise HTTPException(status_code=400, detail="storage must be an enabled s3 backend")
    # On a VM the hardware is fixed by the box, so the RunPod-pod knobs
    # (gpu_type / secure_cloud / disk_gb / volume_gb) don't apply — reflect the
    # VM's actual GPU and drop the cloud-only fields so the config tab isn't
    # misleading (e.g. "L40S" shown while it trains on the VM's H20s).
    eff_gpu_type = body.gpu_type
    is_vm_run = False
    if body.provider_id:
        prov = await session.get(Provider, body.provider_id)
        if prov is None:
            raise HTTPException(status_code=400, detail="unknown provider_id")
        # Providers are per-user — a run bills the owner's cloud account or runs on
        # their VM. Enforce ownership (admins exempt), matching the try-it path.
        if prov.owner_id != user.id and not user.is_admin:
            raise HTTPException(status_code=403, detail="that provider isn't yours")
        if prov.kind == "vm":
            is_vm_run = True
            vm_gpus = (prov.config or {}).get("gpus") or []
            if vm_gpus:
                eff_gpu_type = vm_gpus[0]
    elif body.nodes:
        # Multi-node sweep: no single provider_id — reflect the first node's VM
        # hardware instead of falling through to the RunPod gpu_type default.
        is_vm_run = True
        first_prov = await session.get(Provider, body.nodes[0].provider_id)
        if first_prov is not None:
            first_gpus = (first_prov.config or {}).get("gpus") or []
            if first_gpus:
                eff_gpu_type = first_gpus[0]

    # ---- validate GPU pin, learning rate, and sweep values (clear 400s) ----
    # A multi-node sweep's top-level provider_id/visible_devices/gpu_count are unused
    # (each node carries its own) — validate + resolve the node list instead. Each
    # CHILD trial is created by recursing into this same function with a concrete
    # single provider_id/visible_devices (nodes=None), so it goes through the normal
    # single-pool checks below on its own — no duplicated logic needed here.
    if not math.isfinite(body.learning_rate) or body.learning_rate <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"learning_rate must be a positive number like 1e-4 (got {body.learning_rate!r})",
        )
    if body.nodes:
        resolved_nodes = await _validate_and_resolve_nodes(body.nodes, user, session)
        body = body.model_copy(update={"nodes": resolved_nodes})
        _validate_sweep(body.sweep or {})
        if not any(isinstance(v, list) and v for v in (body.sweep or {}).values()):
            raise HTTPException(status_code=400, detail="nodes requires a non-empty sweep grid")
        if body.gpus_per_trial < 1:
            raise HTTPException(status_code=400, detail="gpus_per_trial must be at least 1")
        pinned_ids = []
        gpu_bound = sum(len(_parse_gpu_indices(n.visible_devices)) for n in body.nodes)
    else:
        if is_vm_run:
            gpu_bound = len(vm_gpus) or int((prov.config or {}).get("gpu_count") or 0)
            target_label = "this VM"
        else:
            gpu_bound = body.gpu_count
            target_label = "the pod"
        pinned_ids = _parse_gpu_indices(body.visible_devices)
        if gpu_bound and pinned_ids:
            oob = sorted({i for i in pinned_ids if i >= gpu_bound})
            if oob:
                raise HTTPException(
                    status_code=400,
                    detail=(f"visible_devices out of range: {oob} — {target_label} has {gpu_bound} "
                            f"GPU(s), valid indices are 0–{gpu_bound - 1}"),
                )
        # Context-parallel group size: must evenly divide the run's GPU count (world = cp_size × dp_size).
        if body.context_parallel and body.cp_size:
            eff_world = len(pinned_ids) if pinned_ids else gpu_bound
            if body.cp_size < 2:
                raise HTTPException(status_code=400, detail="cp_size must be >= 2 when context parallelism is on")
            if eff_world and (body.cp_size > eff_world or eff_world % body.cp_size != 0):
                raise HTTPException(
                    status_code=400,
                    detail=(f"cp_size ({body.cp_size}) must evenly divide the run's GPU count "
                            f"({eff_world}); data parallelism runs across the {eff_world // body.cp_size if body.cp_size else 0} CP groups"),
                )
        _validate_sweep(body.sweep or {})
        sweep_on = any(isinstance(v, list) and v for v in (body.sweep or {}).values())
        if sweep_on:
            slots = len(pinned_ids) if pinned_ids else gpu_bound
            if slots and body.gpus_per_trial > slots:
                raise HTTPException(
                    status_code=400,
                    detail=(f"gpus_per_trial ({body.gpus_per_trial}) exceeds the {slots} GPU(s) "
                            f"available to the sweep on {target_label}"),
                )

    run_id = _gen_id()
    target = await _training_s3_target(body.storage_id)
    s3_prefix = f"{target.prefix_root}{run_id}/"
    config = {
        "task_type": body.task_type,
        "tokenizer": body.tokenizer, "block_size": body.block_size,
        "pack_sequence_length": body.pack_sequence_length,
        "default_speaker": body.default_speaker, "speaker_field": body.speaker_field,
        "pack_only": body.pack_only, "pack_source_dataset_id": body.pack_source_dataset_id,
        "sweep": body.sweep or {}, "gpus_per_trial": body.gpus_per_trial,
        "eval_metric": body.eval_metric, "normalize_text": body.normalize_text,
        "max_epochs": body.max_epochs, "max_steps": body.max_steps,
        "eval_strategy": body.eval_strategy, "eval_steps": body.eval_steps,
        "save_strategy": body.save_strategy, "save_steps": body.save_steps,
        "patience": body.patience, "eval_split_pct": body.eval_split_pct,
        "no_eval": bool(body.no_eval),
        "split_seed": body.split_seed, "batch_size": body.batch_size,
        "grad_accum": body.grad_accum, "cpu_offload": body.cpu_offload,
        "context_parallel": body.context_parallel,
        "cp_size": body.cp_size,
        "learning_rate": body.learning_rate,
        "warmup_steps": body.warmup_steps, "lr_scheduler_type": body.lr_scheduler_type,
        "weight_decay": body.weight_decay,
        # LLM finetune (gemma4) is always LoRA — force it on regardless of the toggle.
        "use_lora": True if body.task_type == "llm" else body.use_lora,
        "lora_r": body.lora_r,
        "lora_alpha_ratio": body.lora_alpha_ratio,
        "gemma_fa4": body.gemma_fa4,
        # LLM objective: "sft" | "dpo" (+ β). llm_finetune routes on training_type.
        "training_type": (body.training_type or "sft") if body.task_type == "llm" else "sft",
        "dpo_beta": body.dpo_beta,
        "lora_alpha": body.lora_alpha, "lora_dropout": body.lora_dropout,
        "lora_target_modules": ([m for m in (body.lora_target_modules or [])
                                 if m in _LORA_TARGET_MODULES] or list(_LORA_TARGET_DEFAULT)),
        "train_embeddings": bool(body.train_embeddings),  # gemma4: full-train embed+lm_head
        "use_dora": bool(body.use_dora),  # LLM: DoRA instead of LoRA (attention + MoE experts)
        "no_moe_lora": bool(body.no_moe_lora),  # LLM MoE: opt out of the routed-expert adapter
        "freeze_encoder": body.freeze_encoder, "use_ddp": body.use_ddp,
        "logging_steps": body.logging_steps,
        "augment_techniques": [t for t in (body.augment_techniques or []) if t in _AUG_TECHNIQUES],
        "augment_prob": body.augment_prob,
        "eval_methods": [m for m in (body.eval_methods or []) if m in _TTS_EVAL_METHODS],
        "eval_max_samples": body.eval_max_samples,
        # OmniVoice (kind=omnivoice_packed) knobs — language tagging + eval split +
        # trainer attn/batch. No-ops for the Qwen3+NeuCodec TTS path.
        "default_language": (body.default_language or "en"),
        "language_field": (body.language_field or None),
        "eval_test_per_speaker": int(body.eval_test_per_speaker or 25),
        "attn_implementation": (body.attn_implementation or "flex_attention"),
        "batch_tokens": body.batch_tokens,
        # Post-train Label-platform export (TTS + LLM). The token is stored Fernet-
        # encrypted (label_token_enc) like the kind=label dataset's — never raw.
        "label_export": bool(body.label_export and body.task_type in ("tts", "llm")),
        "label_base_url": (body.label_base_url or "").strip().rstrip("/") or "http://localhost:3002",
        "label_base_url_secret": (body.label_base_url_secret or "").strip() or None,
        # Token: a Secrets-page key reference wins; else the pasted token Fernet-encrypted.
        "label_token_secret": (body.label_token_secret or "").strip() or None,
        "label_token_enc": (
            crypto.encrypt(json.dumps({"token": body.label_token.strip()}))
            if (body.label_export and body.label_token and body.label_token.strip()
                and not (body.label_token_secret or "").strip()) else None
        ),
        "label_project_name": (body.label_project_name or "").strip() or None,
        "label_samples": int(body.label_samples or 32),
        "label_mos_axes": [a.strip() for a in (body.label_mos_axes or []) if str(a).strip()],
        "label_speakers": [str(s).strip() for s in (body.label_speakers or []) if str(s).strip()],
        "label_speaker_prefix": bool(body.label_speaker_prefix),
        "label_reject_keywords": [str(k).strip() for k in (body.label_reject_keywords or []) if str(k).strip()],
        "label_per_speaker": bool(body.label_per_speaker),
        # LLM-only label export fields (ignored for TTS/ASR).
        "llm_label_eval_dataset_id": body.llm_label_eval_dataset_id if body.task_type == "llm" else None,
        "llm_label_samples": int(body.llm_label_samples or 0) or None,
        "llm_label_max_new_tokens": int(body.llm_label_max_new_tokens or 512),
        "llm_label_mos_axes": (
            [a.strip() for a in (body.llm_label_mos_axes or []) if str(a).strip()]
            or ["Relevance", "Accuracy", "Helpfulness", "Tone"]
        ) if body.task_type == "llm" else [],
        "precision": body.precision, "language": body.language, "task": body.task,
        "base_model": body.base_model,
        # Cloud-pod knobs are irrelevant on a VM — omit them so the config tab
        # matches reality (VM hardware is fixed; gpu_type reflects the VM).
        **({} if is_vm_run else {
            "secure_cloud": body.secure_cloud,
            "data_center_id": (body.data_center_id or "").strip() or None,
            "disk_gb": body.disk_gb, "volume_gb": body.volume_gb,
            "image": (body.image or "").strip() or None,
        }),
        "hf_push_repo": body.hf_push_repo,
        # HF token: a Secrets-page key reference wins; else the pasted token Fernet-
        # encrypted (never raw). Resolved into cfg["hf_token"] at run time.
        "hf_token_secret": (body.hf_token_secret or "").strip() or None,
        "hf_token_enc": (
            crypto.encrypt(json.dumps({"token": body.hf_token.strip()}))
            if (body.hf_token and body.hf_token.strip()
                and not (body.hf_token_secret or "").strip()) else None
        ),
        "work_dir": (body.work_dir or "/share").strip() or "/share",
        "venv_path": (body.venv_path or "").strip() or None,
        "cleanup_checkpoints": body.cleanup_checkpoints,
        # experiment tracking — non-secret knobs + the chosen credential ids.
        "report_to": [t for t in body.report_to if t in ("mlflow", "wandb")],
        "wandb_credential_id": body.wandb_credential_id,
        "mlflow_credential_id": body.mlflow_credential_id,
        "wandb_project": body.wandb_project, "wandb_entity": body.wandb_entity,
        "mlflow_tracking_uri": body.mlflow_tracking_uri,
        "mlflow_experiment": body.mlflow_experiment,
        "run_name": body.name.strip() or run_id,
        # OS env exported on the remote before the trainer; keep only shell-safe
        # names, coerce values to str. Applied (with mkdir for abs paths) by the
        # runner — see _render_env_exports.
        "env_vars": {
            k: str(v) for k, v in (body.env_vars or {}).items()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(k))
        },
        # Multi-node sweep bookkeeping: the resolved node list (concrete visible_devices,
        # empty already expanded to "all") + a full copy of this request so
        # run_multi_node_sweep can reconstruct per-trial child bodies later (the
        # transformed `config` dict above isn't 1:1 invertible back to the request shape).
        "nodes": [n.model_dump() for n in body.nodes] if body.nodes else None,
        "_multi_node_request": body.model_dump() if body.nodes else None,
        "sweep_parent_id": body.sweep_parent_id,
    }
    row = TrainingRun(
        id=run_id, name=body.name.strip() or run_id, dataset_id=body.dataset_id,
        test_dataset_id=body.test_dataset_id, base_model=body.base_model,
        task_type=body.task_type,
        config_json=config, status="queued", s3_prefix=s3_prefix, owner_id=user.id,
        provider_id=body.provider_id, storage_id=body.storage_id,
        # A VM's hardware is fixed by the box: record its full GPU count when the run
        # isn't pinned to a subset (so the row/display reflect what it'll actually use),
        # else the pinned count. RunPod pods use body.gpu_count (the pod's GPU count).
        # A multi-node sweep's "GPU count" is the sum across all its nodes.
        gpu_type=eff_gpu_type,
        gpu_count=(gpu_bound if body.nodes else
                   ((len(pinned_ids) or gpu_bound or body.gpu_count) if is_vm_run else body.gpu_count)),
        visible_devices=(None if body.nodes else body.visible_devices),
    )
    session.add(row)
    await session.commit()

    if body.nodes:
        task = asyncio.create_task(_safe_run_multi_node(redis, run_id))
    else:
        task = asyncio.create_task(_safe_run(redis, run_id))
    _active_runners[run_id] = task
    task.add_done_callback(lambda _t: _active_runners.pop(run_id, None))

    return row


@router.post("", response_model=TrainingRunRecord)
async def create_training_run(
    body: CreateTrainingRunRequest,
    request: Request,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    row = await _create_and_launch_run(body, user, session, request.app.state.redis)
    return _to_record(row, user.username)


def _expand_sweep(sweep: dict) -> list[dict]:
    """{param: [v1, v2]} -> [{param: v1}, {param: v2}, …] (cross-product). Mirrors
    training/sweep_runner.py's expand() — kept as its own tiny copy here rather than
    importing the box-shipped script as a gateway module."""
    keys = [k for k, v in (sweep or {}).items() if isinstance(v, list) and v]
    if not keys:
        return [{}]
    grids = [sweep[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*grids)]


async def _validate_and_resolve_nodes(nodes: list[NodeSpec], user: User,
                                       session: AsyncSession) -> list[NodeSpec]:
    """Check ownership + GPU bounds for each node, and expand an empty/None
    visible_devices into ALL of that provider's GPUs ("0,1,…,N-1")."""
    resolved: list[NodeSpec] = []
    for n in nodes:
        prov = await session.get(Provider, n.provider_id)
        if prov is None:
            raise HTTPException(status_code=400, detail=f"unknown provider_id: {n.provider_id}")
        if prov.owner_id != user.id and not user.is_admin:
            raise HTTPException(status_code=403, detail=f"provider {n.provider_id} isn't yours")
        gpu_bound = (len((prov.config or {}).get("gpus") or [])
                     if prov.kind == "vm" else int((prov.config or {}).get("gpu_count") or 0))
        vis = (n.visible_devices or "").strip()
        if not vis:
            if not gpu_bound:
                raise HTTPException(status_code=400,
                                     detail=f"node {n.provider_id}: no visible_devices given and the "
                                            f"provider's GPU count is unknown — pass an explicit list")
            vis = ",".join(str(i) for i in range(gpu_bound))
        else:
            pinned = _parse_gpu_indices(vis)
            if gpu_bound:
                oob = sorted({i for i in pinned if i >= gpu_bound})
                if oob:
                    raise HTTPException(status_code=400,
                                         detail=f"node {n.provider_id}: visible_devices out of range "
                                                f"{oob} (has {gpu_bound} GPUs)")
        resolved.append(NodeSpec(provider_id=n.provider_id, visible_devices=vis))
    return resolved


async def _safe_run_multi_node(redis, run_id: str, *, resume: bool = False) -> None:
    try:
        await run_multi_node_sweep(redis, run_id, resume=resume)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("multi-node sweep %s crashed", run_id)
        await _finalize(run_id, "failed", None, "internal error — see gateway logs")


async def run_multi_node_sweep(redis, run_id: str, *, resume: bool = False) -> None:
    """Dispatcher for a multi-node sweep TrainingRun: fans the sweep grid out across
    `cfg["nodes"]`, one real independent TrainingRun (via _create_and_launch_run) per
    trial, written into THIS row's result_json in the same {"trials": […], "best": …}
    shape sweep_runner.py produces (so hf-export?source_trial=N, listings, etc. all
    treat this exactly like an ASR/TTS/single-box-LLM sweep). NOT cross-host
    distributed training — each trial is single-node, on whichever node picks it up.

    `resume=True` (set by `_resume_orphaned_sweeps` on gateway startup): unlike a
    single-run TrainingRun, this dispatcher lives ONLY as an in-process asyncio task
    — a gateway restart kills it outright while the already-dispatched trials (each
    a normal detached TrainingRun) keep training untouched, and the generic
    `_reconcile_orphan` re-attaches THEIR live log streaming fine (they have a real
    provider_id). But nothing was left polling/merging their metrics into the
    parent, dispatching any still-pending trial once a slot frees, or ever finalizing
    the sweep — it just sits at status=running with a permanently frozen loss curve.
    Resuming re-adopts whatever's already in `result_json["trials"]` instead of
    resetting every trial back to "pending" and double-dispatching fresh children."""
    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
        if row is None:
            return
        cfg = row.config_json or {}
        owner_id = row.owner_id
        existing_trials = list((row.result_json or {}).get("trials") or []) if resume else []
        if not resume:
            row.status = "running"
            row.started_at = datetime.now(timezone.utc)
        await s.commit()

    req_dict = cfg.get("_multi_node_request") or {}
    base_body = CreateTrainingRunRequest(**req_dict)
    nodes = [NodeSpec(**n) for n in (cfg.get("nodes") or [])]
    sweep_metric = "loss" if cfg.get("task_type") in ("tts", "llm") else (cfg.get("eval_metric") or "wer")
    combos = _expand_sweep(cfg.get("sweep") or {})
    gpus_per_trial = max(1, int(cfg.get("gpus_per_trial") or 1))

    # Each node contributes floor(its GPU count / gpus_per_trial) concurrent slots,
    # sliced the same way sweep_runner.py slices a single box's pool.
    slots: list[tuple[NodeSpec, list[str]]] = []
    for node in nodes:
        gpus = [x for x in node.visible_devices.split(",") if x.strip()]
        chunks = [gpus[i:i + gpus_per_trial] for i in range(0, len(gpus), gpus_per_trial)]
        chunks = [c for c in chunks if len(c) == gpus_per_trial]
        for c in chunks:
            slots.append((node, c))
    if not slots:
        await _finalize(run_id, "failed", None,
                         f"no node offers >= {gpus_per_trial} GPU(s) per trial (gpus_per_trial)")
        return

    if resume and len(existing_trials) == len(combos):
        trials = existing_trials
        await _push_log(redis, run_id,
                         f"[sweep] resuming after a gateway restart · "
                         f"{sum(1 for t in trials if t.get('status') == 'running')} trial(s) still in flight, "
                         f"{sum(1 for t in trials if t.get('status') == 'pending')} pending")
    else:
        trials = [{"trial": i, "params": c, "node": None, "run_id": None,
                   "status": "pending", "metric": None, "artifact": None}
                  for i, c in enumerate(combos)]
        await _push_log(redis, run_id,
                         f"[sweep] {len(combos)} trial(s) · {len(slots)} slot(s) across {len(nodes)} node(s) "
                         f"· {gpus_per_trial} GPU/trial · rank by {sweep_metric} (asc)")
    # Each child already runs its own gpu_sampler + @@STEP log parser (it's a normal
    # single run under the hood) — no separate SSH polling needed here. We just pull
    # its result_json.steps/gpu_samples on the same poll cadence, tag them by trial
    # (steps) / offset GPU indices by trial (gpu_samples, so an H20 "GPU 0" on trial 1
    # doesn't overplot trial 0's "GPU 0"), and merge into THIS row's result_json so
    # the run-detail page's LossCurve/GpuCard render the whole sweep like a single-box one.
    trial_steps: dict[int, list] = {}
    trial_gpu: dict[int, list] = {}

    def _merged_steps() -> list:
        return [pt for i in sorted(trial_steps) for pt in trial_steps[i]]

    def _merged_gpu_samples() -> list:
        return [s for i in sorted(trial_gpu) for s in trial_gpu[i]]

    async def flush() -> None:
        await _flush_result(run_id, {
            "trials": trials, "best": None,
            "steps": _merged_steps(), "gpu_samples": _merged_gpu_samples(),
            # Directly observable staleness: if this stops advancing while the row
            # still says status=running, the dispatcher died (e.g. a gateway restart
            # with no resume) — see run_multi_node_sweep's `resume` path.
            "metrics_updated_at": _iso(datetime.now(timezone.utc)),
        })

    def _pull_child_metrics(i: int, res: dict) -> None:
        trial_steps[i] = [{**st, "trial": i} for st in (res.get("steps") or [])]
        trial_gpu[i] = [
            {"t": gs.get("t"), "gpus": [{**g, "index": (g.get("index") or 0) + i * 100} for g in (gs.get("gpus") or [])]}
            for gs in (res.get("gpu_samples") or [])
        ]

    # Trials already dispatched (have a run_id) and still running when this
    # dispatcher started — only possible on `resume` (a fresh start never has
    # any). Queue only the genuinely not-yet-dispatched trials; each already-
    # running one is handed to whichever slot it originally occupied (matched by
    # node+GPU list) so that slot's worker adopts it — monitor first, THEN fall
    # into the normal queue-draining loop once it completes, exactly like a
    # freshly dispatched trial finishing and picking up the next pending one.
    queue: asyncio.Queue = asyncio.Queue()
    adopt_by_slot: dict[tuple[str, str], tuple[int, str]] = {}
    for i, t in enumerate(trials):
        if t.get("status") == "running" and t.get("run_id"):
            node_desc = t.get("node") or {}
            key = (node_desc.get("provider_id"), node_desc.get("visible_devices"))
            adopt_by_slot[key] = (i, t["run_id"])
        elif t.get("status") == "pending":
            queue.put_nowait((i, t["params"]))
        # else: already done/failed/cancelled from before the restart — nothing to do.

    # Relay each child's meaningful log lines into the parent's own log stream
    # (tagged [trial N]) — mirrors sweep_runner.py's `[trial i] {line}` prefixing
    # for a single-box sweep, so the parent's Logs tab visibly moves instead of
    # only updating on trial start/finish. Filtered (not every line) since 3+
    # concurrent children's raw stdout (pip/deps/HF download noise) would flood it.
    # On resume this replays each adopted trial's FULL history once (last_idx starts
    # at 0) — a one-time catch-up burst, not incorrect, just noisier than usual.
    _RELAY_MARKERS = ("@@STEP", "@@DONE", "@@ERROR", "[train]", "[trainer]",
                      "[deps]", "[model]", "[preflight]", "error", "Error", "Traceback")
    last_idx: dict[int, int] = {}

    async def _relay_child_log(i: int, child_id: str) -> None:
        try:
            raw = await redis.lrange(f"train:logs:{child_id}", 0, -1)
        except Exception:
            return
        start = last_idx.get(i, 0)
        for line in raw[start:]:
            text = line.decode() if isinstance(line, bytes) else line
            if any(m in text for m in _RELAY_MARKERS):
                await _push_log(redis, run_id, f"[trial {i}] {text}")
        last_idx[i] = len(raw)

    async def _monitor_trial(i: int, child_id: str) -> None:
        trial = trials[i]
        try:
            tick = 0
            while True:
                await asyncio.sleep(5)
                await _relay_child_log(i, child_id)
                tick += 1
                if tick % 2:  # metrics/status every ~10s, log relay every ~5s
                    continue
                async with session_factory()() as s:
                    child = await s.get(TrainingRun, child_id)
                    if child is None:
                        trial["status"] = "failed"
                        break
                    res = child.result_json or {}
                    _pull_child_metrics(i, res)
                    if child.status in ("done", "failed", "cancelled"):
                        await _relay_child_log(i, child_id)  # catch any tail lines
                        trial["status"] = child.status
                        trial["metric"] = ((res.get("best") or {}).get(
                            "loss" if sweep_metric == "loss" else sweep_metric))
                        trial["artifact"] = res.get("artifact")
                        break
                await flush()  # persist the just-pulled steps/gpu_samples live, not only at trial end
            await _push_log(redis, run_id,
                             f"[sweep] trial {i} {trial['status'].upper()} "
                             f"({sweep_metric}={trial['metric']}) -> {child_id}")
            await flush()
        except Exception as e:  # noqa: BLE001
            logger.exception("multi-node sweep %s trial %d crashed", run_id, i)
            trial["status"] = "failed"
            trial["error"] = str(e)
            await _push_log(redis, run_id, f"[sweep] trial {i} FAILED (dispatcher error): {e}")
            await flush()

    async def worker(node: NodeSpec, gpu_slice: list[str]) -> None:
        vis = ",".join(gpu_slice)
        adopted = adopt_by_slot.get((node.provider_id, vis))
        if adopted is not None:
            await _monitor_trial(*adopted)
        while True:
            try:
                i, combo = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            trial = trials[i]
            trial["node"] = {"provider_id": node.provider_id, "visible_devices": vis}
            trial_body = base_body.model_copy(update={
                "sweep": {}, "gpus_per_trial": 1, "nodes": None,
                "provider_id": node.provider_id, "visible_devices": vis,
                "gpu_count": len(gpu_slice),
                "name": f"{base_body.name}-t{i}",
                "sweep_parent_id": run_id,
                **combo,
            })
            try:
                async with session_factory()() as s:
                    user = await s.get(User, owner_id)
                    child = await _create_and_launch_run(trial_body, user, s, redis)
                    await s.commit()
                    child_id = child.id
                trial["run_id"] = child_id
                trial["status"] = "running"
                await _push_log(redis, run_id,
                                 f"[sweep] trial {i} START params={json.dumps(combo)} "
                                 f"node={node.provider_id}:{vis} -> {child_id}")
                await flush()
            except Exception as e:  # noqa: BLE001
                logger.exception("multi-node sweep %s trial %d (node %s) crashed", run_id, i, node.provider_id)
                trial["status"] = "failed"
                trial["error"] = str(e)
                await _push_log(redis, run_id, f"[sweep] trial {i} FAILED (dispatcher error): {e}")
                await flush()
                continue
            await _monitor_trial(i, child_id)

    await asyncio.gather(*[worker(node, gpu_slice) for node, gpu_slice in slots])

    done = [t for t in trials if t["status"] != "pending"]
    ranked = sorted((t for t in done if t.get("metric") is not None), key=lambda t: t["metric"])
    best = ranked[0] if ranked else None
    if best:
        await _push_log(redis, run_id,
                         f"[sweep] best: trial {best['trial']} · {sweep_metric}={best['metric']} "
                         f"· {json.dumps(best['params'])}")
    else:
        await _push_log(redis, run_id, "[sweep] done — no trial reported a metric")
    await _finalize(run_id, "done", 0, None, {
        "trials": done, "best": best,
        "steps": _merged_steps(), "gpu_samples": _merged_gpu_samples(),
    })


@router.post("/{run_id}/restart", response_model=TrainingRunRecord)
async def restart_training_run(
    run_id: str,
    request: Request,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Clone a run's full config (dataset, model, hyperparams, env vars, …) into
    a fresh queued run and launch it. Handy to re-run a finished/failed run
    unchanged — the source run is left as-is."""
    src = await _owned(run_id, user, session)
    new_id = _gen_id()
    target = await _training_s3_target(src.storage_id)
    config = dict(src.config_json or {})
    config["run_name"] = src.name  # run_name drives W&B/MLflow naming
    row = TrainingRun(
        id=new_id, name=src.name, dataset_id=src.dataset_id,
        test_dataset_id=src.test_dataset_id, base_model=src.base_model,
        task_type=src.task_type, config_json=config, status="queued",
        s3_prefix=f"{target.prefix_root}{new_id}/", owner_id=user.id,
        provider_id=src.provider_id, storage_id=src.storage_id,
        gpu_type=src.gpu_type, gpu_count=src.gpu_count, visible_devices=src.visible_devices,
    )
    session.add(row)
    await session.commit()

    redis = request.app.state.redis
    task = asyncio.create_task(_safe_run(redis, new_id))
    _active_runners[new_id] = task
    task.add_done_callback(lambda _t: _active_runners.pop(new_id, None))
    return _to_record(row, user.username)


@router.get("", response_model=list[TrainingRunRecord])
async def list_training_runs(
    scope: str = "mine",
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Back-compat full list. result_json is slimmed to `best` exactly like
    /_page (each run's steps/epochs/gpu_samples can be thousands of points —
    shipping them for EVERY run made this the slowest list endpoint in the API,
    ~146ms p50 at 60 runs and growing linearly). The full record still comes
    from GET /{run_id}."""
    q = select(TrainingRun).order_by(TrainingRun.created_at.desc())
    if not (scope == "all" and user.is_admin):
        q = q.where(TrainingRun.owner_id == user.id)
    rows = (await session.execute(q)).scalars().all()
    # owner username map — one IN query, not one session.get per owner
    names: dict[int, str] = {}
    if rows:
        urows = await session.execute(
            select(User.id, User.username).where(User.id.in_({r.owner_id for r in rows}))
        )
        names = {uid: uname for uid, uname in urows.all()}
    out: list[TrainingRunRecord] = []
    for r in rows:
        rec = _to_record(r, names.get(r.owner_id, "?"))
        rec.result_json = {"best": rec.result_json.get("best")} if rec.result_json else None
        out.append(rec)
    return out


class TrainingRunPageResponse(BaseModel):
    total: int
    items: list[TrainingRunRecord]


@router.get("/_page", response_model=TrainingRunPageResponse)
async def list_training_runs_page(
    scope: str = "mine",
    q: str = "",
    status: str = "",
    sort: str = "newest",
    limit: int = Query(12, ge=1, le=100),
    offset: int = Query(0, ge=0),
    include_children: bool = False,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Paged run list — the web list fetches page-by-page so hundreds of runs
    (each with a steps-heavy result_json) don't ship up front. Search/filter/sort
    run server-side so they cover ALL runs, not just the loaded page. result_json
    is slimmed to `best` (the only key the list cards read); the full record
    comes from GET /{run_id}. Declared before /{run_id} (declaration-order
    matching); the plain list endpoint above stays for back-compat."""
    stmt = select(TrainingRun)
    if not (scope == "all" and user.is_admin):
        stmt = stmt.where(TrainingRun.owner_id == user.id)
    if status:
        stmt = stmt.where(TrainingRun.status == status)
    if not include_children:
        # Hide multi-node sweep child trials by default — they're real TrainingRun
        # rows (so logs/hf-export/etc. work unmodified) but only the parent sweep
        # row is meant to be browsed; children stay reachable via GET /{run_id} or
        # the parent's own result_json.trials[].run_id.
        stmt = stmt.where(TrainingRun.config_json.op("->>")("sweep_parent_id").is_(None))
    # Multi-token search (every token must match), mirroring the old client-side
    # filter: name/id/status/model/dataset/gpu + owner username + the config JSON
    # (sweep params, task type, …) as text.
    for tok in (q or "").lower().split():
        like = f"%{tok}%"
        stmt = stmt.where(
            or_(
                TrainingRun.id.ilike(like),
                TrainingRun.name.ilike(like),
                TrainingRun.status.ilike(like),
                TrainingRun.base_model.ilike(like),
                TrainingRun.dataset_id.ilike(like),
                TrainingRun.task_type.ilike(like),
                TrainingRun.gpu_type.ilike(like),
                cast(TrainingRun.config_json, Text).ilike(like),
                TrainingRun.owner_id.in_(select(User.id).where(User.username.ilike(like))),
            )
        )
    total = (
        await session.execute(select(func.count()).select_from(stmt.subquery()))
    ).scalar_one()
    order = (
        TrainingRun.created_at.asc() if sort == "oldest" else TrainingRun.created_at.desc()
    )
    rows = (await session.execute(stmt.order_by(order).limit(limit).offset(offset))).scalars().all()
    names: dict[int, str] = {}
    if rows:
        urows = await session.execute(
            select(User.id, User.username).where(User.id.in_({r.owner_id for r in rows}))
        )
        names = {uid: uname for uid, uname in urows.all()}
    items: list[TrainingRunRecord] = []
    for r in rows:
        rec = _to_record(r, names.get(r.owner_id, "?"))
        # steps/epochs/gpu_samples can be thousands of points per run — the list
        # only shows the headline metric.
        rec.result_json = {"best": rec.result_json.get("best")} if rec.result_json else None
        items.append(rec)
    return TrainingRunPageResponse(total=total, items=items)


async def _owned(run_id: str, user: User, session: AsyncSession) -> TrainingRun:
    row = await session.get(TrainingRun, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    if row.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="not yours")
    return row


@router.get("/{run_id}", response_model=TrainingRunRecord)
async def get_training_run(
    run_id: str,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    row = await _owned(run_id, user, session)
    u = await session.get(User, row.owner_id)
    prov = await session.get(Provider, row.provider_id) if row.provider_id else None
    store = await session.get(Storage, row.storage_id) if row.storage_id else None
    rec = _to_record(
        row, u.username if u else "?",
        provider_name=prov.name if prov else None,
        provider_kind=prov.kind if prov else None,
        storage_name=store.name if store else None,
    )
    # VM hardware is fixed by the box, so the provider's live GPU inventory is
    # authoritative — override the value stored at create time, which can be stale
    # (a run created before the VM's `gpus` were probed shows the RunPod default
    # "NVIDIA L40S"×1). Reflects the GPUs the run actually used: the pin, else all.
    if prov is not None and prov.kind == "vm":
        vm_gpus = (prov.config or {}).get("gpus") or []
        if vm_gpus:
            rec.gpu_type = vm_gpus[0]
            pinned = [x for x in (row.visible_devices or "").split(",") if x.strip()]
            rec.gpu_count = len(pinned) or len(vm_gpus)
    return rec


# ---- Portable export/import (move a finished run between deployments) --------
# Self-contained JSON: the DB row's config + result metrics/loss curves, plus the
# run's small S3 artifacts (logs, lora_meta, sample outputs) inlined as base64. Big
# checkpoints (lora.pt / *.safetensors) exceed the cap and only appear in
# files_omitted — peers import to inspect metrics/loss/config/logs, NOT to re-serve
# the weights. Import re-creates the row + writes files into THIS deployment's bucket.
# Mirrors the benchmark export/import in bench.py.

TRAIN_EXPORT_KIND = "gpuplatform.autotrain.export"
# Never embed control/secret artifacts (SSH keys, the raw config with resolved paths).
_TRAIN_EXPORT_SKIP = {
    "vm_key", "vm_key.pub", "rp_key", "rp_key.pub",
    "id_ed25519", "id_ed25519.pub", "config.yaml", "config.json",
}
_TRAIN_PER_FILE_CAP = 25 * 1024 * 1024   # 25 MiB per file
_TRAIN_TOTAL_CAP = 50 * 1024 * 1024      # 50 MiB total embedded


class ImportTrainingData(BaseModel):
    name: str
    dataset_id: str = ""
    test_dataset_id: Optional[str] = None
    base_model: str = ""
    task_type: str = "asr"
    config_json: Optional[dict] = None
    status: str = "done"
    exit_code: Optional[int] = None
    error_text: Optional[str] = None
    result_json: Optional[dict] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    cost_per_hr: Optional[float] = None
    gpu_type: Optional[str] = None
    gpu_count: int = 1
    visible_devices: Optional[str] = None


class ImportTrainingFile(BaseModel):
    name: str          # path relative to the run's S3 prefix
    content_b64: str


class ImportTrainingBody(BaseModel):
    kind: str
    version: int = 1
    source_run_id: Optional[str] = None
    run: ImportTrainingData
    files: list[ImportTrainingFile] = []


@router.get("/{run_id}/export")
async def export_training_run(
    run_id: str,
    include_files: bool = True,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Download a finished run as a self-contained JSON: the DB row's config +
    result metrics/loss, plus the run's small S3 artifacts inlined as base64 (so the
    destination needs no access to this instance's bucket). Big checkpoints exceed
    the size cap and are listed in files_omitted. Pair with /v1/training-runs/import
    on another deployment."""
    import base64

    row = await _owned(run_id, user, session)

    # Bake the RESOLVED GPU into the export (not the raw stored value): a VM run's
    # box hardware is authoritative, and the importer drops the provider so it can't
    # self-heal — without this, a VM run exported before its `gpus` were probed would
    # import as the stale "NVIDIA L40S"×1 default instead of the box's real GPUs.
    eff_gpu_type, eff_gpu_count = row.gpu_type, row.gpu_count
    if row.provider_id:
        prov = await session.get(Provider, row.provider_id)
        if prov is not None and prov.kind == "vm":
            vm_gpus = (prov.config or {}).get("gpus") or []
            if vm_gpus:
                eff_gpu_type = vm_gpus[0]
                pinned = [x for x in (row.visible_devices or "").split(",") if x.strip()]
                eff_gpu_count = len(pinned) or len(vm_gpus)

    files: list[dict] = []
    omitted: list[dict] = []
    # No bucket resolvable (null/deleted storage + no env bucket) → export metadata
    # without inlined files instead of 500ing.
    target = None
    if include_files:
        try:
            target = await _training_s3_target(row.storage_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("export %s: no s3 target (%s) — metadata only", run_id, e)
    if include_files and target and target.bucket and row.s3_prefix:
        total = 0
        try:
            listing = await asyncio.to_thread(s3_list, row.s3_prefix, target)
        except Exception as e:  # noqa: BLE001
            logger.warning("export %s: file list failed: %s", run_id, e)
            listing = []
        for it in listing:
            key = it["key"]
            rel = key[len(row.s3_prefix):] if key.startswith(row.s3_prefix) else key
            base = rel.rsplit("/", 1)[-1]
            if not rel or base in _TRAIN_EXPORT_SKIP:
                continue
            size = int(it.get("size") or 0)
            if size > _TRAIN_PER_FILE_CAP or total + size > _TRAIN_TOTAL_CAP:
                omitted.append({"name": rel, "size": size, "reason": "exceeds export size cap"})
                continue
            data = await asyncio.to_thread(s3_get_bytes, key, target)
            if data is None:
                omitted.append({"name": rel, "size": size, "reason": "unreadable"})
                continue
            total += len(data)
            files.append({"name": rel, "content_b64": base64.b64encode(data).decode("ascii")})

    export = {
        "kind": TRAIN_EXPORT_KIND,
        "version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": row.id,
        "run": {
            "name": row.name,
            "dataset_id": row.dataset_id,
            "test_dataset_id": row.test_dataset_id,
            "base_model": row.base_model,
            "task_type": row.task_type,
            "config_json": row.config_json,
            "status": row.status,
            "exit_code": row.exit_code,
            "error_text": row.error_text,
            "result_json": row.result_json,
            "created_at": _iso(row.created_at),
            "started_at": _iso(row.started_at),
            "ended_at": _iso(row.ended_at),
            "cost_per_hr": row.cost_per_hr,
            "gpu_type": eff_gpu_type,
            "gpu_count": eff_gpu_count,
            "visible_devices": row.visible_devices,
        },
        "files": files,
        "files_omitted": omitted,
    }
    return Response(
        content=json.dumps(export),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{row.id}.autotrain.json"'},
    )


@router.post("/import", response_model=TrainingRunRecord)
async def import_training_run(
    body: ImportTrainingBody,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Re-create a run exported from another deployment. Mints a fresh id, owns it
    as the importer, writes any embedded files into THIS deployment's bucket, and
    stores config/results so the dashboard renders fully. The import is metadata-only
    — it can't be resumed/served (provider/storage are intentionally dropped); its
    status reflects the source run."""
    import base64

    if body.kind != TRAIN_EXPORT_KIND:
        raise HTTPException(
            status_code=400,
            detail={"error": f"not an autotrain export (kind={body.kind!r})"},
        )
    data = body.run
    new_id = _gen_id()
    target = await _training_s3_target(None)  # write into this deployment's own bucket
    prefix = f"{target.prefix_root}{new_id}/"

    written = 0
    if target.bucket:
        for f in body.files:
            rel = (f.name or "").lstrip("/")
            if not rel or ".." in rel:
                continue
            try:
                raw = base64.b64decode(f.content_b64)
            except Exception:
                continue
            try:
                await asyncio.to_thread(s3_put_bytes, f"{prefix}{rel}", raw, target)
                written += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("import %s: failed to write %s: %s", new_id, rel, e)

    def _parse(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

    row = TrainingRun(
        id=new_id,
        name=(data.name or "imported")[:128],
        dataset_id=data.dataset_id or "",
        test_dataset_id=data.test_dataset_id,
        base_model=data.base_model or "",
        task_type=data.task_type or "asr",
        config_json=data.config_json or {},
        status=data.status or "done",
        s3_prefix=prefix,
        exit_code=data.exit_code,
        error_text=data.error_text,
        result_json=data.result_json,
        owner_id=user.id,
        created_at=_parse(data.created_at) or datetime.now(timezone.utc),
        started_at=_parse(data.started_at),
        ended_at=_parse(data.ended_at),
        cost_per_hr=data.cost_per_hr,
        runpod_pod_id=None,
        provider_id=None,
        storage_id=None,
        gpu_type=data.gpu_type,
        gpu_count=data.gpu_count or 1,
        visible_devices=data.visible_devices,
    )
    session.add(row)
    await session.commit()
    u = await session.get(User, user.id)
    return _to_record(row, u.username if u else user.username)


class LabelExportRequest(BaseModel):
    # All optional — fall back to the run's config_json. A pasted token wins over a
    # secret-key ref; both are stored on the run's config so a later retry reuses them.
    base_url: Optional[str] = None
    base_url_secret: Optional[str] = None
    token: Optional[str] = None
    token_secret: Optional[str] = None
    project_name: Optional[str] = None
    samples: Optional[int] = None
    mos_axes: Optional[list[str]] = None
    speakers: Optional[list[str]] = None  # balance clips across these speaker names
    speaker_prefix: Optional[bool] = None  # prefix transcription with the speaker name
    reject_keywords: Optional[list[str]] = None  # drop text samples containing these phrases
    per_speaker: Optional[bool] = None    # one project per speaker (from each speaker's own clips)
    tts_codec: Optional[str] = None       # "neucodec" (upstream 24 kHz) | "neucodec-44k" (Scicom 44.1 kHz)
    # Run-on target for the synthesis (mirrors serverless/new). "vm" → a registered
    # VM provider; "cloud" → spawn a fresh RunPod pod (provision → synth → teardown).
    # Absent → the run's own VM (back-compat).
    run_on: Optional[str] = None
    provider_id: Optional[str] = None     # vm provider (run_on=vm) or RunPod account (run_on=cloud)
    gpu_type: Optional[str] = None         # run_on=cloud
    gpu_count: Optional[int] = None        # run_on=cloud
    secure_cloud: Optional[bool] = None    # run_on=cloud
    data_center_id: Optional[str] = None   # run_on=cloud — RunPod region pin; blank → auto
    disk_gb: Optional[int] = None          # run_on=cloud
    volume_gb: Optional[int] = None        # run_on=cloud
    visible_devices: Optional[str] = None  # CUDA_VISIBLE_DEVICES pin for the synth
    venv_path: Optional[str] = None        # TTS uv venv on the box (default /share/autotrain-tts)
    # LLM-only override fields
    llm_eval_dataset_id: Optional[str] = None   # override eval dataset for LLM
    llm_samples: Optional[int] = None           # override response count
    llm_mos_axes: Optional[list[str]] = None    # override rating axes
    llm_max_new_tokens: Optional[int] = None    # override generation length
    vllm_version: Optional[str] = None          # LLM: vLLM version for the merge→serve venv (default 0.23.0)
    # LLM: HF token to download the (usually gated) base model for the merge — a pasted
    # token or a global-secret key. Else the run's own token, else platform HF_TOKEN.
    base_hf_token: Optional[str] = None
    base_hf_token_secret: Optional[str] = None


@router.post("/{run_id}/label-export")
async def retry_label_export(
    run_id: str,
    body: LabelExportRequest,
    request: Request,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """(Re)run the Label-platform export for a finished TTS run: synthesize N clips
    from the trained model + create/seed a recording+MOS project. Used when the run
    finished without label creds, or to re-export. Runs in the background (synthesis
    over SSH takes minutes) — progress streams to the run's logs and the
    result_json.label_project card appears when done. Runs on the chosen Run-on
    target: a registered VM provider, or a fresh RunPod pod (spawned + torn down)."""
    row = await _owned(run_id, user, session)
    if (row.task_type or "asr") not in ("tts", "llm"):
        raise HTTPException(status_code=400, detail="label export is for TTS and LLM runs")
    if row.status != "done":
        raise HTTPException(status_code=400, detail="the run must finish successfully first")
    if not (((row.result_json or {}).get("artifact") or {}).get("s3_uri")):
        raise HTTPException(status_code=400, detail="no trained model artifact for this run")
    # Validate the chosen Run-on target (mirrors serverless/new). Absent → fall back
    # to the run's own VM (back-compat), which must then be a live VM provider.
    run_on = (body.run_on or "").strip().lower()
    exp_provider_id = (body.provider_id or "").strip() or None
    if run_on == "cloud":
        if exp_provider_id:
            p = await session.get(Provider, exp_provider_id)
            if p is None or p.kind != "runpod":
                raise HTTPException(status_code=400, detail="provider_id must be a RunPod account for cloud export")
    elif run_on == "vm":
        vm_id = exp_provider_id or row.provider_id
        p = await session.get(Provider, vm_id) if vm_id else None
        if p is None or p.kind != "vm":
            raise HTTPException(status_code=400, detail="pick a registered VM provider for a bare-metal export")
        exp_provider_id = vm_id
    else:
        prov = await session.get(Provider, row.provider_id) if row.provider_id else None
        if prov is None or prov.kind != "vm":
            raise HTTPException(status_code=400, detail=("label export runs on a VM provider; this run used a "
                                "cloud pod (gone after training). Pick a Run-on target (VM or cloud)."))
    # A previous export still in flight (possibly stuck on a slow synth)? Cancel it
    # and start fresh — re-clicking "Export to Label" supersedes it. The cancelled
    # task's project-creation step never runs, so no duplicate project is created
    # (the orphaned VM synth just finishes + is discarded).
    prev = _active_label_exports.pop(run_id, None)
    if prev is not None and not prev.done():
        prev.cancel()
        await _push_log(request.app.state.redis, run_id,
                        "[gateway] label export: superseded by a new request — restarting")

    cfg = dict(row.config_json or {})
    if body.base_url is not None or body.base_url_secret is not None:
        cfg["label_base_url"] = ((body.base_url or "").strip().rstrip("/")
                                 or cfg.get("label_base_url") or "http://localhost:3002")
        cfg["label_base_url_secret"] = (body.base_url_secret or "").strip() or None
    if (body.token_secret or "").strip():
        cfg["label_token_secret"] = body.token_secret.strip()
        cfg["label_token_enc"] = None
    elif (body.token or "").strip():
        cfg["label_token_enc"] = crypto.encrypt(json.dumps({"token": body.token.strip()}))
        cfg["label_token_secret"] = None
    if body.project_name is not None:
        cfg["label_project_name"] = body.project_name.strip() or None
    if body.samples is not None:
        cfg["label_samples"] = max(1, int(body.samples))
    if body.mos_axes is not None:
        cfg["label_mos_axes"] = [a.strip() for a in body.mos_axes if str(a).strip()]
    if body.speakers is not None:
        cfg["label_speakers"] = [s.strip() for s in body.speakers if str(s).strip()]
    if body.speaker_prefix is not None:
        cfg["label_speaker_prefix"] = bool(body.speaker_prefix)
    if body.reject_keywords is not None:
        cfg["label_reject_keywords"] = [k.strip() for k in body.reject_keywords if str(k).strip()]
    if body.per_speaker is not None:
        cfg["label_per_speaker"] = bool(body.per_speaker)
    if body.tts_codec is not None:
        cfg["tts_codec"] = (body.tts_codec.strip() or "neucodec")
    # Run-on target (where to synthesize) + the cloud pod spec.
    cfg["label_run_on"] = run_on or None
    cfg["label_provider_id"] = exp_provider_id
    if body.gpu_type is not None:
        cfg["label_gpu_type"] = body.gpu_type.strip() or None
    if body.gpu_count is not None:
        cfg["label_gpu_count"] = max(1, int(body.gpu_count))
    if body.secure_cloud is not None:
        cfg["label_secure_cloud"] = bool(body.secure_cloud)
    if body.data_center_id is not None:
        cfg["label_data_center_id"] = body.data_center_id.strip() or None
    if body.disk_gb is not None:
        cfg["label_disk_gb"] = max(20, int(body.disk_gb))
    if body.volume_gb is not None:
        cfg["label_volume_gb"] = max(0, int(body.volume_gb))
    if body.visible_devices is not None:
        cfg["label_visible_devices"] = body.visible_devices.strip() or None
    if body.venv_path is not None:
        cfg["venv_path"] = body.venv_path.strip() or cfg.get("venv_path")
    # LLM-only overrides
    if body.llm_eval_dataset_id is not None:
        cfg["llm_label_eval_dataset_id"] = body.llm_eval_dataset_id.strip() or None
    if body.llm_samples is not None:
        cfg["llm_label_samples"] = max(1, int(body.llm_samples))
    if body.llm_mos_axes is not None:
        cfg["llm_label_mos_axes"] = [a.strip() for a in body.llm_mos_axes if str(a).strip()]
    if body.llm_max_new_tokens is not None:
        cfg["llm_label_max_new_tokens"] = max(1, int(body.llm_max_new_tokens))
    if body.vllm_version is not None:
        cfg["label_vllm_version"] = body.vllm_version.strip() or None
    # Base-model (gated) download token for the LLM merge — a referenced global secret
    # (key stored plainly) or a pasted token (Fernet-encrypted, never raw), mirroring the
    # Label-platform token handling above.
    if (body.base_hf_token_secret or "").strip():
        cfg["base_hf_token_secret"] = body.base_hf_token_secret.strip()
        cfg["base_hf_token_enc"] = None
    elif (body.base_hf_token or "").strip():
        cfg["base_hf_token_enc"] = crypto.encrypt(json.dumps({"token": body.base_hf_token.strip()}))
        cfg["base_hf_token_secret"] = None
    cfg["label_export"] = True

    has_url = bool((cfg.get("label_base_url_secret") or "").strip() or (cfg.get("label_base_url") or "").strip())
    has_tok = bool((cfg.get("label_token_secret") or "").strip() or cfg.get("label_token_enc"))
    if not has_url:
        raise HTTPException(status_code=400, detail="provide the Label platform URL")
    if not has_tok:
        raise HTTPException(status_code=400, detail="provide a Label platform API token (or pick a secret)")
    # For LLM runs, an eval dataset is required.
    if (row.task_type or "asr") == "llm" and not cfg.get("llm_label_eval_dataset_id"):
        raise HTTPException(status_code=400, detail="provide an eval dataset ID for LLM label export")

    # Pre-flight: confirm the Label platform is reachable + the token is accepted
    # BEFORE we spawn a pod / touch the VM / run synthesis. Fails the request fast so
    # the error shows in the export dialog, not after a multi-minute synth.
    pf_base_url, pf_token = await _resolve_label_creds(cfg)
    verr = (await _verify_label_platform(pf_base_url, pf_token)
            if (pf_base_url and pf_token) else "Label platform URL/token unresolved")
    if verr:
        raise HTTPException(status_code=502, detail=f"Label platform check failed — {verr}")

    # Persist the (merged) label_* fields so the run remembers them for next time.
    result = dict(row.result_json or {})
    async with session_factory()() as s:
        r2 = await s.get(TrainingRun, run_id)
        if r2 is not None:
            merged = dict(r2.config_json or {})
            for k in ("label_export", "label_base_url", "label_base_url_secret",
                      "label_token_secret", "label_token_enc", "label_project_name",
                      "label_samples", "label_mos_axes", "label_speakers", "label_speaker_prefix",
                      "label_reject_keywords", "label_per_speaker", "label_run_on", "label_provider_id",
                      "label_gpu_type", "label_gpu_count", "label_secure_cloud", "label_data_center_id",
                      "label_disk_gb", "label_volume_gb", "label_visible_devices", "venv_path", "tts_codec",
                      "llm_label_eval_dataset_id", "llm_label_samples",
                      "llm_label_max_new_tokens", "llm_label_mos_axes",
                      "base_hf_token_secret", "base_hf_token_enc"):
                merged[k] = cfg.get(k)
            r2.config_json = merged
            await s.commit()

    redis = request.app.state.redis
    task_type = row.task_type or "asr"

    async def _bg() -> None:
        try:
            await _push_log(redis, run_id, "[gateway] label export: retrying on request …")
            if task_type == "llm":
                await _create_llm_label_project_for_run(run_id, cfg, result, redis)
            else:
                await _create_label_project_for_run(run_id, cfg, result, redis)
        except asyncio.CancelledError:
            await _push_log(redis, run_id, "[gateway] label export: cancelled (superseded)")
            raise
        except Exception as e:  # noqa: BLE001
            await _push_log(redis, run_id, f"[gateway] label export failed: {e}")
        finally:
            # Only clear the entry if it's still ours (a superseding request may have
            # already replaced it with a newer task).
            if _active_label_exports.get(run_id) is asyncio.current_task():
                _active_label_exports.pop(run_id, None)

    task = asyncio.create_task(_bg())
    _active_label_exports[run_id] = task
    return {"status": "started"}


@router.post("/{run_id}/label-export/cancel")
async def cancel_label_export(
    run_id: str,
    request: Request,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Stop a running Label export. Cancels the in-memory task (its finally tears down
    any pod it spawned + stops the synthesis), kills the box-side synth process (it
    runs over SSH, outliving the task — and a gateway restart leaves it orphaned with
    status stuck on 'running'), and best-effort tears down an orphaned cloud
    sgpu-label-<run_id> pod so it can't bill on."""
    row = await _owned(run_id, user, session)
    redis = request.app.state.redis

    # 1. Cancel the in-memory task — its finally tears down a spawned cloud pod and
    #    aborts the project-creation step. (No-op after a gateway restart.)
    t = _active_label_exports.pop(run_id, None)
    if t is not None and not t.done():
        t.cancel()

    # 2. Kill the box-side synth by script name (handles a VM run, where there's no
    #    pod to tear down, and the orphaned case). The config path is per-export so
    #    this can't outlive into another run's synth on a shared box.
    ssh = await _resolve_run_ssh(row)
    if ssh is not None:
        host, port, suser, key = ssh

        def _kill() -> None:
            cli = _ssh_connect(host, int(port), suser, key)
            try:
                _ssh_exec(cli, "pkill -9 -f tts_label_export.py 2>/dev/null; "
                               "rm -f /tmp/sgpu_tts_label_cfg.json 2>/dev/null || true")
            finally:
                cli.close()

        try:
            await asyncio.to_thread(_kill)
        except Exception as e:  # noqa: BLE001
            await _push_log(redis, run_id, f"[gateway] label export: box kill attempt failed: {e}")

    # 3. Orphan cleanup: tear down a spawned cloud pod (sgpu-label-<run_id>) the
    #    cancelled task couldn't (gateway restarted mid-export) — else it bills on.
    cfg = dict(row.config_json or {})
    label_prov_id = (cfg.get("label_provider_id") or "").strip() or row.provider_id
    if await _reap_label_pod(run_id, label_prov_id):
        await _push_log(redis, run_id, "[gateway] label export: cloud pod torn down")

    await _set_label_export_state(run_id, {"status": "cancelled", "error": "stopped by user"})
    await _push_log(redis, run_id, "[gateway] label export stopped by user.")
    return {"status": "cancelled"}


async def _set_hf_export_state(run_id: str, state: dict) -> None:
    """Merge an hf-export status object into result_json.hf_export so the UI can
    show 'pushing to Hugging Face' + the resulting link even on an already-done run."""
    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
        if row is None or row.status == "cancelled":
            return
        rj = dict(row.result_json or {})
        cur = dict(rj.get("hf_export") or {})
        cur.update(state)
        rj["hf_export"] = cur
        row.result_json = rj
        await s.commit()


async def _hf_token_for_storage(storage_id: Optional[str], session: AsyncSession) -> Optional[str]:
    """HF token from a kind=huggingface Storage: a referenced global secret
    (config.hf_token_secret) wins, else the storage's own encrypted token."""
    st = await session.get(Storage, storage_id) if storage_id else None
    if st is None:
        return None
    cfg = st.config or {}
    ref = cfg.get("hf_token_secret")
    if ref:
        from .global_env_api import load_global_env
        tok = (await load_global_env(session)).get(ref)
        if tok:
            return tok
    enc = cfg.get("credentials_enc")
    if enc:
        try:
            return json.loads(crypto.decrypt(enc)).get("token")
        except Exception:  # noqa: BLE001
            return None
    return None


async def _hf_endpoint_for_storage(storage_id: Optional[str], session: AsyncSession) -> Optional[str]:
    """Custom HF Hub endpoint (HF_ENDPOINT) from a kind=huggingface Storage, or None
    for huggingface.co. Precedence: a referenced global secret (config.endpoint_secret)
    > the storage's literal `endpoint`. Mirrors datasets_api._hf_endpoint."""
    st = await session.get(Storage, storage_id) if storage_id else None
    if st is None or st.kind != "huggingface":
        return None
    cfg = st.config or {}
    ref = cfg.get("endpoint_secret")
    if ref:
        from .global_env_api import load_global_env
        val = (await load_global_env(session)).get(ref)
        if val and val.strip():
            return val.strip().rstrip("/")
    ep = (cfg.get("endpoint") or "").strip()
    return ep.rstrip("/") or None


def _loopback_endpoint(endpoint: Optional[str]) -> Optional[str]:
    """If `endpoint` points at THIS gateway's OWN /hf mirror (its public host, or
    localhost), rewrite it to a loopback URL (127.0.0.1:<GATEWAY_BIND port>) keeping
    the path. A gateway-side export then hits the mirror directly — bypassing the
    nginx ingress (whose `client_max_body_size` caps a single-part LFS PUT, killing
    multi-GB shards) and TLS. External mirrors (a different host) are left untouched."""
    if not endpoint:
        return endpoint
    from urllib.parse import urlparse
    ep = urlparse(endpoint)
    if not ep.hostname:
        return endpoint
    pub_host = urlparse(os.environ.get("GATEWAY_PUBLIC_URL", "").strip()).hostname
    is_own = ep.hostname in ("localhost", "127.0.0.1") or (pub_host and ep.hostname == pub_host)
    if not is_own:
        return endpoint  # a different host → external mirror, don't redirect to ourselves
    port = (os.environ.get("GATEWAY_BIND", "0.0.0.0:8080").rsplit(":", 1)[-1] or "8080").strip()
    return f"http://127.0.0.1:{port}{ep.path}".rstrip("/")


def _s3_uri_has_objects(model_s3: str, creds: dict) -> bool:
    """True if the s3://bucket/prefix URI lists ≥1 object with these creds. Used to
    fast-fail a merge whose checkpoint isn't reachable here (e.g. an imported run whose
    artifact points at another deployment's storage). Blocking — call via to_thread."""
    try:
        import boto3
        from botocore.client import Config as BotoConfig
        if not (model_s3 or "").startswith("s3://"):
            return False
        bucket, _, prefix = model_s3[len("s3://"):].partition("/")
        cli = boto3.client(
            "s3", region_name=creds.get("region") or "us-east-1",
            endpoint_url=creds.get("endpoint") or None,
            aws_access_key_id=creds.get("access_key") or None,
            aws_secret_access_key=creds.get("secret_key") or None,
            config=BotoConfig(signature_version="s3v4"),
        )
        r = cli.list_objects_v2(Bucket=bucket, Prefix=prefix.rstrip("/") + "/", MaxKeys=1)
        return int(r.get("KeyCount") or 0) > 0
    except Exception:  # noqa: BLE001 — unreachable / no creds / bad bucket → treat as not reachable
        return False


class HfExportRequest(BaseModel):
    repo: str
    storage_id: Optional[str] = None   # a kind=huggingface Storage (provides token + custom endpoint)
    private: bool = False
    # Push a SPECIFIC sweep trial instead of the run's stored "best" artifact. The sweep
    # orchestrator ranks trials by sweep_metric (loss for TTS/LLM) — for TTS that doesn't
    # track the metrics that actually matter (CER/MOS/speaker similarity), so the
    # loss-ranked "best" is often not the trial you actually want to publish. Looked up
    # from this run's own result_json.trials; 400s if the trial doesn't exist or didn't finish.
    source_trial: Optional[int] = None
    # HF token for downloading a GATED BASE MODEL during an LLM merge (e.g. google/gemma-*),
    # mirrors serverless/new. This is NOT the destination/push token — that comes from the
    # storage (or platform HF_TOKEN). A pasted token or a referenced global-secret key;
    # falls back to the push token when unset. Only used for a merge.
    base_hf_token: Optional[str] = None         # pasted token (hf_…)
    base_hf_token_secret: Optional[str] = None  # a global-secret key (e.g. "HF_TOKEN")
    # LLM only: merge the raw LoRA checkpoint into the base model on a GPU and upload a
    # loadable HF model (safetensors) instead of the adapter. No-op for ASR/TTS (their
    # artifact is already a merged model). Needs a compute target below.
    merge: bool = False
    merge_dtype: Optional[str] = None      # fp16 (default) | bf16; only affects FP8 MoE archs
    vllm_version: Optional[str] = None     # accepted for symmetry with try-it (unused by export)
    # Run-on target for the merge (mirrors LabelExportRequest). "cloud" → spin up a fresh
    # RunPod pod (build the arch venv, merge, upload, tear down); "vm" → a chosen VM;
    # None → the run's own VM (back-compat).
    run_on: Optional[str] = None
    provider_id: Optional[str] = None      # vm provider (run_on=vm) or RunPod account (run_on=cloud)
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None
    secure_cloud: Optional[bool] = None
    data_center_id: Optional[str] = None
    disk_gb: Optional[int] = None
    volume_gb: Optional[int] = None
    visible_devices: Optional[str] = None  # CUDA pin for the merge
    venv_path: Optional[str] = None        # override the merge venv path
    image: Optional[str] = None            # cloud pod image; blank → DEFAULT_IMAGE

    @field_validator("venv_path")
    @classmethod
    def _safe_venv_path(cls, v, info):  # noqa: N805
        return validate_path_field(v, info.field_name)


@router.post("/{run_id}/hf-export")
async def export_to_huggingface(
    run_id: str,
    body: HfExportRequest,
    request: Request,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Push this run's BEST/final model to a Hugging Face repo on demand.

    ASR/TTS artifacts are already merged models at train-end → uploaded as-is, either from
    the GATEWAY (run_on="gateway", or a custom-endpoint mirror — box-independent, the model
    is fetched from S3 here) or the run's VM. LLM artifacts are a raw
    LoRA checkpoint → must be MERGED into the base on a GPU first; that runs on the run's
    VM, a chosen VM, or a fresh RunPod pod (Run-on picker), then the merged safetensors
    are uploaded. The HF token comes from the selected kind=huggingface Storage (else the
    platform HF_TOKEN secret). Runs in the background; status + link land in
    result_json.hf_export."""
    row = await _owned(run_id, user, session)
    if row.status != "done":
        raise HTTPException(status_code=400, detail="the run must finish successfully first")
    if body.source_trial is not None:
        trials = ((row.result_json or {}).get("trials")) or []
        trial = next((t for t in trials if t.get("trial") == body.source_trial), None)
        if trial is None:
            raise HTTPException(status_code=400, detail=f"trial {body.source_trial} not found on this run")
        if trial.get("status") != "done":
            raise HTTPException(status_code=400,
                                 detail=f"trial {body.source_trial} status is {trial.get('status')!r}, not done")
        model_s3 = ((trial.get("artifact") or {}).get("s3_uri"))
        if not model_s3:
            raise HTTPException(status_code=400, detail=f"trial {body.source_trial} has no model artifact")
    else:
        model_s3 = (((row.result_json or {}).get("artifact") or {}).get("s3_uri"))
    if not model_s3:
        raise HTTPException(status_code=400, detail="no trained model artifact for this run")
    repo = (body.repo or "").strip()
    if not repo:
        raise HTTPException(status_code=400, detail="a target repo (org/name) is required")
    # Destination/push token: from the selected HuggingFace storage, else platform HF_TOKEN.
    token = await _hf_token_for_storage(body.storage_id, session)
    if not token:
        token = (await _resolve_global_env()).get("HF_TOKEN")
    if not token:
        raise HTTPException(status_code=400, detail=("no Hugging Face token — pick a HuggingFace storage "
                            "or set HF_TOKEN in Secrets"))
    # Base-model token: for downloading a GATED base model during a merge (may be a
    # different HF account than the push target). Pasted > referenced global secret >
    # (fallback) the push token above.
    base_token = (body.base_hf_token or "").strip() or None
    if not base_token and body.base_hf_token_secret:
        from .global_env_api import load_global_env
        base_token = (await load_global_env(session)).get(body.base_hf_token_secret)
    if not base_token:
        base_token = token
    # Custom HF_ENDPOINT from the same storage (self-hosted mirror), or None → huggingface.co.
    hf_endpoint = await _hf_endpoint_for_storage(body.storage_id, session)
    task_type = (row.task_type or "asr").lower()
    base_model = row.base_model or (row.config_json or {}).get("base_model") or ""
    arch = _llm_arch(base_model)

    # Merge only applies to LLM: its S3 artifact is a raw custom-LoRA checkpoint (lora.pt
    # + meta), not a loadable HF model. ASR/TTS are already merged at train-end → no-op.
    merge = bool(body.merge) and task_type == "llm"
    if task_type == "llm" and not merge:
        raise HTTPException(status_code=400, detail=(
            "LLM runs are stored as a raw LoRA checkpoint (not loadable by transformers). "
            "Enable 'Merge LoRA into base model' to upload a loadable model."))
    merge_dtype = (body.merge_dtype or "fp16").strip().lower()
    if merge_dtype not in ("fp16", "bf16"):
        merge_dtype = "fp16"

    creds = _s3_creds_from_storage(await session.get(Storage, row.storage_id) if row.storage_id else None)
    cfg = dict(row.config_json or {})
    private = bool(body.private)
    redis = request.app.state.redis

    # A merge re-downloads the checkpoint on a GPU box → fast-fail (before spending a pod)
    # if it isn't reachable from this deployment. The usual cause is an imported run whose
    # checkpoint lives on another deployment's storage (dropped by the import size cap).
    if merge and not await asyncio.to_thread(_s3_uri_has_objects, model_s3, creds):
        raise HTTPException(status_code=400, detail=(
            f"the trained checkpoint isn't reachable from this deployment ({model_s3}) — "
            "likely an imported run whose checkpoint lives elsewhere. Nothing to merge."))

    # ---- compute target -------------------------------------------------------------
    # Non-merge (ASR/TTS): push from the GATEWAY when asked (run_on="gateway") or when the
    # storage has a custom endpoint (its modern huggingface_hub handles the `…/hf` path the
    # VM's older client mis-parses) — the model is fetched from S3 here + pushed directly, so
    # it does NOT need the run's training box (whose shared uv venv is often gone weeks after
    # the run). Otherwise the run's VM. Merge (LLM): a GPU is required → the run's VM, a chosen
    # VM, or a fresh RunPod pod.
    mirror_merge = merge and bool(hf_endpoint)   # merge → a custom / self-hosted HF mirror
    run_on = (body.run_on or "").strip().lower()
    if run_on == "gateway" and merge:
        raise HTTPException(status_code=400, detail=(
            "gateway export can't merge — an LLM merge needs a GPU (Run-on: a VM or a fresh cloud pod)."))
    local_export = (bool(hf_endpoint) or run_on == "gateway") and not merge
    # A gateway push re-downloads the checkpoint here (like merge) → fast-fail if the artifact
    # isn't reachable from this deployment (an imported run whose checkpoint lives elsewhere)
    # before streaming a doomed download. (The custom-endpoint local_export path is unchanged.)
    if run_on == "gateway" and not await asyncio.to_thread(_s3_uri_has_objects, model_s3, creds):
        raise HTTPException(status_code=400, detail=(
            f"the trained model isn't reachable from this deployment ({model_s3}) — likely an "
            "imported run whose checkpoint lives elsewhere. Push it from its VM instead."))
    # Talk to our OWN mirror over loopback (bypass the nginx body cap + TLS). A gateway-side
    # push reaches it directly; a merge on a GPU box reaches that loopback through a reverse
    # SSH tunnel (needs_tunnel). Keep the public endpoint for the displayed URL. An external
    # mirror (different host) is left untouched and pushed to directly from the box.
    push_endpoint = _loopback_endpoint(hf_endpoint) if (local_export or mirror_merge) else hf_endpoint
    needs_tunnel = mirror_merge and bool(push_endpoint) and push_endpoint != hf_endpoint
    exp_provider_id = (body.provider_id or "").strip() or None
    ssh = None            # (host, port, user, key) when we run on an existing box
    cloud_spec = None     # pod spec dict when we must provision a fresh pod in _bg
    if not local_export:
        if run_on == "cloud":
            if not merge:
                raise HTTPException(status_code=400, detail=(
                    "cloud export is only for LLM merge — ASR/TTS models export from a VM or "
                    "a custom-endpoint HuggingFace storage."))
            prov = await session.get(Provider, exp_provider_id) if exp_provider_id else None
            if exp_provider_id and (prov is None or prov.kind != "runpod"):
                raise HTTPException(status_code=400, detail="the chosen cloud provider must be a RunPod account")
            if prov is not None and prov.owner_id != user.id and not user.is_admin:
                raise HTTPException(status_code=403, detail="that provider isn't yours")
            cloud_spec = {
                "provider_id": exp_provider_id,
                "gpu_type": (body.gpu_type or "").strip() or (row.gpu_type or "NVIDIA L40S"),
                "gpu_count": max(1, int(body.gpu_count or 1)),
                "secure_cloud": True if body.secure_cloud is None else bool(body.secure_cloud),
                "data_center_id": (body.data_center_id or "").strip() or None,
                "disk_gb": int(body.disk_gb or 80),
                "volume_gb": int(body.volume_gb or 80),
                "image": (body.image or "").strip() or None,
            }
        elif run_on == "vm":
            vm_id = exp_provider_id or row.provider_id
            prov = await session.get(Provider, vm_id) if vm_id else None
            if prov is None or prov.kind != "vm":
                raise HTTPException(status_code=400, detail="the chosen VM provider was not found")
            # A caller-chosen VM (exp_provider_id) must be the caller's own; the run's
            # own box (row.provider_id) already passed _owned() above.
            if exp_provider_id and prov.owner_id != user.id and not user.is_admin:
                raise HTTPException(status_code=403, detail="that VM provider isn't yours")
            ssh = await _resolve_provider_ssh(prov, run_id)
            if ssh is None:
                raise HTTPException(status_code=400, detail="can't reach the chosen VM (SSH coords unavailable)")
        else:
            prov = await session.get(Provider, row.provider_id) if row.provider_id else None
            if prov is None or prov.kind != "vm":
                if merge:
                    raise HTTPException(status_code=400, detail=(
                        "this run's training box is gone — pick a Run-on target (a VM or a fresh "
                        "cloud pod) to run the merge."))
                raise HTTPException(status_code=400, detail=(
                    "HF export runs on a VM provider; this run used a cloud pod (gone after "
                    "training). Pick a Run-on target, or use a HuggingFace storage with a custom "
                    "endpoint (pushes from the gateway)."))
            ssh = await _resolve_run_ssh(row)
            if ssh is None:
                raise HTTPException(status_code=400, detail="can't reach the run's VM (SSH coords unavailable)")

    # The merge venv: on a VM it's the arch venv training already built; on a fresh pod
    # _build_llm_venv_ssh builds it. None for non-merge (uses the run's train venv).
    venv_path = (body.venv_path or "").strip() or (_llm_venv(arch) if merge else None)
    visible_devices = (body.visible_devices or "").strip() or None

    # Re-click supersedes a stuck push (the orphaned upload just finishes + is discarded).
    prev = _active_hf_exports.pop(run_id, None)
    if prev is not None and not prev.done():
        prev.cancel()
        await _push_log(redis, run_id, "[gateway] HF export: superseded by a new request — restarting")
    await _set_hf_export_state(run_id, {
        "status": "running", "repo": repo, "url": None, "error": None, "merge": merge,
        "run_on": (run_on or ("vm" if ssh else None)),
        "provider_id": exp_provider_id if cloud_spec else None, "pod_id": None,
    })

    async def _bg() -> None:
        # Stream the box-side build/download/merge/upload lines to the run's logs (the
        # script runs in a thread; it appends here and an async pump mirrors to Redis)
        # so a multi-minute push isn't silent.
        export_lines: list[str] = []
        _sent = {"n": 0}

        async def _pump() -> None:
            while True:
                await asyncio.sleep(0.5)
                while _sent["n"] < len(export_lines):
                    await _push_log(redis, run_id, export_lines[_sent["n"]])
                    _sent["n"] += 1

        pump_task = asyncio.create_task(_pump())
        spawned: Optional[tuple[str, str]] = None   # (api_key, pod_id) for cloud teardown
        tunnel_host: Optional[str] = None           # reverse-tunnel host to close on exit
        run_ssh = ssh
        try:
            # Run-on = cloud: provision a fresh pod + build the arch merge venv on it.
            if cloud_spec is not None:
                api_key = await compute._resolve_api_key(cloud_spec["provider_id"])
                key_filename, pub = _gen_ssh_key(_work_dir(run_id))
                await _push_log(redis, run_id,
                                f"[gateway] HF export: provisioning RunPod pod "
                                f"({cloud_spec['gpu_type']} x{cloud_spec['gpu_count']}) …")
                runpod_id, phost, pport, _cost = await _provision_pod(
                    api_key, f"sgpu-hfexport-{run_id}", cloud_spec["image"] or DEFAULT_IMAGE,
                    cloud_spec["gpu_type"], cloud_spec["gpu_count"], cloud_spec["secure_cloud"],
                    cloud_spec["disk_gb"], cloud_spec["volume_gb"], pub,
                    data_center_id=cloud_spec["data_center_id"])
                spawned = (api_key, runpod_id)
                await _set_hf_export_state(run_id, {"pod_id": runpod_id})
                await _push_log(redis, run_id,
                                "[gateway] HF export: building the LLM stack on the pod (first build is slow) …")
                await asyncio.to_thread(_build_llm_venv_ssh, phost, int(pport), "root",
                                        key_filename, run_id, dict(cfg), base_model,
                                        venv_path, export_lines.append)
                run_ssh = (phost, int(pport), "root", key_filename)

            where = "the gateway" if local_export else ("a fresh cloud pod" if spawned else "the run's VM")
            await _push_log(redis, run_id,
                            f"[gateway] exporting {'merged model' if merge else 'best model'} to Hugging Face "
                            f"→ {repo} (from {where}) …")
            if local_export:
                if push_endpoint != hf_endpoint:
                    await _push_log(redis, run_id, f"[gateway] pushing via {push_endpoint} (loopback, bypassing the ingress) …")
                res = await asyncio.to_thread(
                    _run_hf_export_local, run_id, model_s3, creds, repo, token, private,
                    push_endpoint, export_lines.append)
            elif merge:
                # Push a merged model to a self-hosted mirror? A gateway-local mirror is only
                # reachable from the gateway (loopback), so open a reverse SSH tunnel on the
                # merge box (mirrors the RunPod-worker tunnel) — the box then pushes to the
                # loopback endpoint, which routes back through the tunnel to the gateway mirror.
                box_endpoint = push_endpoint if mirror_merge else None
                if needs_tunnel:
                    from . import vm_tunnel
                    h, p, u, kf = run_ssh
                    gw_host, gw_port = vm_tunnel.parse_host_port(os.environ.get("GATEWAY_BIND", "127.0.0.1:8080"), 8080)
                    await asyncio.to_thread(
                        vm_tunnel.ensure, h, int(p), u, Path(kf).read_text(),
                        [vm_tunnel.Forward(gw_port, gw_host, gw_port)])
                    tunnel_host = h
                    # Record the host so cancel / restart-reconcile can close the tunnel even
                    # if this task's finally never runs (gateway restart) or races cancellation.
                    await _set_hf_export_state(run_id, {"tunnel_host": h})
                    await _push_log(redis, run_id,
                                    "[gateway] HF export: reverse SSH tunnel up (box → gateway mirror) …")
                res = await asyncio.to_thread(
                    _run_hf_merge_export_ssh, *run_ssh, run_id, model_s3, creds, repo, token,
                    private, dict(cfg), base_model, arch, merge_dtype, venv_path,
                    visible_devices, base_token, box_endpoint, export_lines.append)
            else:
                res = await asyncio.to_thread(
                    _run_hf_export_ssh, *run_ssh, run_id, model_s3, creds, repo, token, private, cfg,
                    hf_endpoint, export_lines.append)
            # The script builds the URL from the endpoint it pushed to; show the
            # public endpoint instead of the loopback one.
            res_url = res.get("url")
            if (local_export or needs_tunnel) and push_endpoint and push_endpoint != hf_endpoint and res_url:
                res_url = res_url.replace(push_endpoint, hf_endpoint, 1)
            async with session_factory()() as s:
                r2 = await s.get(TrainingRun, run_id)
                if r2 is not None:
                    rj = dict(r2.result_json or {})
                    rj["hf_export"] = {"status": "done", "repo": res.get("repo", repo),
                                       "url": res_url, "merge": merge}
                    art = dict(rj.get("artifact") or {})
                    art["hf_repo"] = res.get("repo", repo)
                    rj["artifact"] = art
                    r2.result_json = rj
                    await s.commit()
            await _push_log(redis, run_id, f"[gateway] pushed to Hugging Face: {res_url}")
        except asyncio.CancelledError:
            await _push_log(redis, run_id, "[gateway] HF export: cancelled (superseded)")
            raise
        except Exception as e:  # noqa: BLE001
            await _set_hf_export_state(run_id, {"status": "failed", "error": str(e)[:1200]})
            await _push_log(redis, run_id, f"[gateway] HF export failed: {e}")
        finally:
            # Tear down a pod we spawned so it can't bill past the export (best-effort).
            if spawned:
                try:
                    await _terminate_pod(*spawned)
                    await _set_hf_export_state(run_id, {"pod_id": None})
                    await _push_log(redis, run_id, "[gateway] HF export: cloud pod torn down")
                except Exception:  # noqa: BLE001
                    logger.warning("HF export %s: pod teardown failed", run_id)
            # Close the reverse SSH tunnel (autossh subprocess) so it can't linger.
            if tunnel_host:
                try:
                    from . import vm_tunnel
                    vm_tunnel.close(tunnel_host)
                except Exception:  # noqa: BLE001
                    pass
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            while _sent["n"] < len(export_lines):
                try:
                    await _push_log(redis, run_id, export_lines[_sent["n"]])
                except Exception:  # noqa: BLE001
                    pass
                _sent["n"] += 1
            if _active_hf_exports.get(run_id) is asyncio.current_task():
                _active_hf_exports.pop(run_id, None)

    task = asyncio.create_task(_bg())
    _active_hf_exports[run_id] = task
    return {"status": "started"}


@router.post("/{run_id}/hf-export/cancel")
async def cancel_huggingface_export(
    run_id: str,
    request: Request,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Stop a stuck/running HF export. Cancels the in-memory task AND kills the VM-side
    download/upload process (it runs over SSH, outliving the task — and a gateway restart
    leaves it orphaned with status stuck on "running"). Killing by the uniquely-named
    per-run script is precise: it can't touch another run's export."""
    row = await _owned(run_id, user, session)
    redis = request.app.state.redis

    # 1. Cancel the in-memory background task, if this gateway still owns one.
    t = _active_hf_exports.pop(run_id, None)
    if t is not None and not t.done():
        t.cancel()

    # 2. Kill the VM-side process by its unique script name (handles the orphaned case
    #    where no in-memory task exists because the gateway restarted mid-push).
    killed = False
    ssh = await _resolve_run_ssh(row)
    if ssh is not None:
        host, port, suser, key = ssh
        work_dir = ((row.config_json or {}).get("work_dir") or "/share").rstrip("/")

        def _kill() -> bool:
            cli = _ssh_connect(host, int(port), suser, key)
            try:
                out = _ssh_capture(
                    cli,
                    # Match every per-request tag (sgpu_hf_export_<run_id>-<token>…).
                    f"pkill -9 -f sgpu_hf_export_{run_id} 2>/dev/null; sleep 1; "
                    f"if pgrep -f sgpu_hf_export_{run_id} >/dev/null; then echo ALIVE; else echo DEAD; fi; "
                    f"rm -rf /tmp/sgpu_hf_export_{run_id}* /tmp/sgpu_llm_hfexport_{run_id}* "
                    f"{work_dir}/sgpu-hf-export/{run_id}* 2>/dev/null || true",
                )
                return "DEAD" in out
            finally:
                cli.close()

        try:
            killed = await asyncio.to_thread(_kill)
        except Exception as e:  # noqa: BLE001
            await _push_log(redis, run_id, f"[gateway] HF export: VM kill attempt failed: {e}")

    # Tear down a spawned cloud pod (sgpu-hfexport-<run_id>) the cancelled task couldn't
    # (gateway restarted mid-export), else it bills on. Also close its reverse SSH tunnel —
    # the autossh is detached (survives a gateway restart) and would otherwise retry the
    # dead pod forever (leaked process). Both are best-effort + idempotent.
    he = (row.result_json or {}).get("hf_export") or {}
    if he.get("pod_id") or he.get("run_on") == "cloud":
        if await _reap_hf_export_pod(run_id, he.get("provider_id")):
            await _push_log(redis, run_id, "[gateway] HF export: cloud pod torn down")
    if he.get("tunnel_host"):
        try:
            from . import vm_tunnel
            await asyncio.to_thread(vm_tunnel.close, he["tunnel_host"])
        except Exception:  # noqa: BLE001
            pass

    await _set_hf_export_state(run_id, {"status": "cancelled", "error": "stopped by user"})
    await _push_log(redis, run_id, "[gateway] HF export stopped by user.")
    return {"status": "cancelled", "vm_process_killed": killed}


@router.get("/{run_id}/metrics")
async def training_metrics(
    run_id: str,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """All metrics for a run in one call — for dashboards + programmatic pulls.
    Per-step training loss, per-epoch eval (WER/CER/loss), best checkpoint, and
    the GPU util/mem/temp time series. Read straight from the persisted
    result_json, so it works for finished runs too."""
    row = await _owned(run_id, user, session)
    r = row.result_json or {}
    return {
        "id": row.id,
        "status": row.status,
        "steps": r.get("steps") or [],
        "epochs": r.get("epochs") or [],
        "gpu_samples": r.get("gpu_samples") or [],
        "trials": r.get("trials"),  # sweep plan + per-trial status; None for single runs
        # best = lowest-metric epoch (recomputed from `epochs` so it's correct even
        # for runs whose stored `best` predates the greater_is_better fix).
        "best": _best_from_epochs(r, row.config_json) or r.get("best"),
        "artifact": r.get("artifact"),
        "stopped_early": bool(r.get("stopped_early")),
        "error": r.get("error") or row.error_text,
    }


class RenameTrainingRunRequest(BaseModel):
    name: str


@router.patch("/{run_id}", response_model=TrainingRunRecord)
async def rename_training_run(
    run_id: str,
    body: RenameTrainingRunRequest,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Rename a run (display label only — does not touch the W&B/MLflow run_name
    baked into config_json at create time)."""
    row = await _owned(run_id, user, session)
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    row.name = name[:128]
    await session.commit()
    u = await session.get(User, row.owner_id)
    return _to_record(row, u.username if u else "?")


async def _kill_and_clean_vm_run(row: TrainingRun) -> None:
    """SSH to the run's VM, group-kill the detached trainer, and delete its
    checkpoints/work dir + per-run staging + run files. The trainer is launched
    with `setsid` so it SURVIVES the gateway task cancel — without this it keeps
    training (holding the GPU) and its checkpoints pile up on /share. No-op for a
    RunPod run (the pod teardown takes everything with it) or if SSH is unresolvable."""
    prov = None
    if row.provider_id:
        async with session_factory()() as s:
            prov = await s.get(Provider, row.provider_id)
    if prov is None or prov.kind != "vm":
        return
    ssh = await _resolve_run_ssh(row)
    if ssh is None:
        return
    rlog, rpid, rsh = _remote_run_paths(row.id)
    stage = f"/tmp/sgpu_run_{row.id}"
    wd = ((row.result_json or {}).get("work_dir_remote") or "").strip()

    def _do() -> None:
        cli = _ssh_connect(*ssh)
        try:
            # The trainer logs "[trainer] work dir: <path>" early; recover it from the
            # remote log if it wasn't persisted to result_json yet, so we delete the
            # exact dir (never a glob that could hit a concurrent run).
            workdir = wd
            if not workdir:
                try:
                    m = re.search(r"\[trainer\] work dir: (\S+)", _ssh_capture(cli, f"cat {rlog} 2>/dev/null"))
                    if m:
                        workdir = m.group(1)
                except Exception:  # noqa: BLE001
                    pass
            # Group-kill the detached run-script's session (it's a setsid session
            # leader, so pid == pgid → -P reaches torchrun + its workers).
            _ssh_capture(cli, (
                f'P="$(cat {rpid} 2>/dev/null)"; '
                f'if [ -n "$P" ]; then kill -TERM -"$P" 2>/dev/null || kill -TERM "$P" 2>/dev/null; '
                f'sleep 2; kill -KILL -"$P" 2>/dev/null || kill -KILL "$P" 2>/dev/null || true; fi'))
            targets = [t for t in [workdir, stage, rlog, rpid, rsh] if t]
            if targets:
                _ssh_capture(cli, "rm -rf " + " ".join(shlex.quote(t) for t in targets))
        finally:
            try:
                cli.close()
            except Exception:  # noqa: BLE001
                pass

    await asyncio.to_thread(_do)


@router.delete("/{run_id}")
async def delete_training_run(
    run_id: str,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    row = await _owned(run_id, user, session)
    t = _active_runners.get(run_id)
    if t:
        t.cancel()
    st = _RUN_STATE.pop(run_id, None)
    if st:
        await _terminate_pod(st["api_key"], st["runpod_id"])
    else:
        # VM run: kill the (detached) trainer + delete its checkpoints before the row goes.
        try:
            await _kill_and_clean_vm_run(row)
        except Exception as e:  # noqa: BLE001
            logger.warning("delete %s: VM kill/cleanup failed: %s", run_id, e)
    await session.delete(row)
    await session.commit()
    return {"ok": True, "id": run_id}


@router.post("/{run_id}/terminate", response_model=TrainingRunRecord)
async def terminate_training_run(
    run_id: str,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    row = await _owned(run_id, user, session)
    if row.status in ("done", "failed", "cancelled"):
        raise HTTPException(status_code=409, detail=f"already {row.status}")
    t = _active_runners.get(run_id)
    if t:
        t.cancel()
    st = _RUN_STATE.pop(run_id, None)
    if st:
        await _terminate_pod(st["api_key"], st["runpod_id"])
    else:
        # VM run: the detached trainer survives the task cancel — SSH-kill it and
        # delete its checkpoints/work dir so it stops training + frees /share.
        try:
            await _kill_and_clean_vm_run(row)
        except Exception as e:  # noqa: BLE001
            logger.warning("terminate %s: VM kill/cleanup failed: %s", run_id, e)
    row.status = "cancelled"
    row.ended_at = datetime.now(timezone.utc)
    await session.commit()
    u = await session.get(User, row.owner_id)
    return _to_record(row, u.username if u else "?")


@router.post("/{run_id}/stop-early", response_model=TrainingRunRecord)
async def stop_training_early(
    run_id: str,
    request: Request,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Gracefully stop a RUNNING training run: signal the trainer to finish at the
    next step and SAVE + upload the partial model (then finalize as done, run any
    label/HF export). Contrast /terminate, which hard-kills and discards. The
    trainer polls $SGPU_STOP_FLAG every step; we just `touch` that file over SSH."""
    row = await _owned(run_id, user, session)
    if row.status != "running":
        raise HTTPException(status_code=409, detail=f"run is {row.status}, not running")
    ssh = await _resolve_run_ssh(row)
    if ssh is None:
        raise HTTPException(status_code=400, detail="can't reach the run's box to signal early-stop")
    stop_flag = f"/tmp/sgpu_run_{run_id}/STOP"

    def _touch() -> None:
        cli = _ssh_connect(ssh[0], int(ssh[1]), ssh[2], ssh[3])
        try:
            _ssh_capture(cli, f"touch {stop_flag} 2>/dev/null || true")
        finally:
            cli.close()

    await asyncio.to_thread(_touch)
    redis = request.app.state.redis
    await _push_log(redis, run_id, "[gateway] early-stop requested — the trainer will finish the current "
                                   "step, save + upload the model, then finalize.")
    # UI hint so the status badge can show "stopping early" until the run finalizes.
    async with session_factory()() as s:
        r2 = await s.get(TrainingRun, run_id)
        if r2 is not None:
            rj = dict(r2.result_json or {})
            rj["stopping_early"] = True
            r2.result_json = rj
            await s.commit()
    await session.refresh(row)
    u = await session.get(User, row.owner_id)
    return _to_record(row, u.username if u else "?")


@router.get("/{run_id}/logs")
async def get_training_logs(
    run_id: str,
    tail: int = 400,
    request: Request = None,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    row = await _owned(run_id, user, session)
    redis = request.app.state.redis
    lines: list[str] = []
    try:
        raw = await redis.lrange(f"train:logs:{run_id}", -int(tail), -1)
        lines = [b.decode("utf-8", "replace") if isinstance(b, bytes) else str(b) for b in raw]
    except Exception:
        pass
    if not lines:
        full = _full_log_path(run_id)
        if full.exists():
            lines = full.read_text(errors="replace").splitlines()[-int(tail):]
    return {"status": row.status, "error_text": row.error_text, "lines": lines}


@router.get("/{run_id}/logs/stream")
async def stream_training_logs(
    run_id: str,
    request: Request,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    row = await _owned(run_id, user, session)
    redis = request.app.state.redis

    async def gen() -> AsyncIterator[str]:
        key = f"train:logs:{run_id}"
        sent = 0
        while True:
            try:
                raw = await redis.lrange(key, sent, -1)
            except Exception:
                raw = []
            for b in raw:
                line = b.decode("utf-8", "replace") if isinstance(b, bytes) else str(b)
                # Strip carriage returns: tqdm progress bars use \r, which SSE
                # treats as a line terminator — it would split the line and drop
                # the trailing @@STEP/@@METRIC payload the live charts parse.
                line = line.replace("\r", "")
                yield f"data: {line}\n\n"
                sent += 1
            async with session_factory()() as s:
                cur = await s.get(TrainingRun, run_id)
            # Keep streaming a finished run while a post-train export is still in
            # flight (label synthesis / HF push run AFTER the run is "done" — else the
            # stream would end immediately and their progress never reaches the UI).
            rj = (cur.result_json or {}) if cur is not None else {}
            exporting = (((rj.get("label_export") or {}).get("status") == "running")
                         or ((rj.get("hf_export") or {}).get("status") == "running"))
            if cur is None or (cur.status in ("done", "failed", "cancelled") and not exporting):
                yield "event: end\ndata: end\n\n"
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/{run_id}/logs/trim")
async def trim_training_logs(
    run_id: str,
    request: Request,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Strip tqdm/HF download progress-bar lines from a run's stored logs — the live Redis
    list (what the Logs tab replays), the on-disk canonical log, and, for a finalized run,
    the S3 logs.txt. One model pull can add tens of thousands of these useless lines. Returns
    how many were removed; the client should reconnect its log stream to see the trimmed view."""
    row = await _owned(run_id, user, session)
    redis = request.app.state.redis
    terminal = row.status in ("done", "failed", "cancelled")
    redis_removed = disk_removed = s3_removed = 0

    # 1) Redis live list — rewrite in place (delete + rpush the survivors). This is what the
    #    Logs tab shows; the SSE cursor is per-connection, so the client reconnects afterwards.
    key = f"train:logs:{run_id}"
    try:
        raw = await redis.lrange(key, 0, -1)
        kept: list[str] = []
        for b in raw:
            s = b.decode("utf-8", "replace") if isinstance(b, bytes) else str(b)
            if _is_progress_line(s):
                redis_removed += 1
            else:
                kept.append(s)
        if redis_removed:
            async with redis.pipeline(transaction=True) as pipe:
                pipe.delete(key)
                if kept:
                    pipe.rpush(key, *kept)
                if terminal:
                    pipe.expire(key, LOG_LIST_TTL_S)  # keep the finalized-run retention
                await pipe.execute()
    except Exception as e:  # noqa: BLE001
        logger.warning("trim %s: redis rewrite failed: %s", run_id, e)

    # 2) On-disk canonical log (the source uploaded to S3 for live / restart finalizes).
    full = _full_log_path(run_id)
    try:
        if full.exists():
            src = full.read_text(errors="replace").splitlines()
            keep = [ln for ln in src if not _is_progress_line(ln)]
            disk_removed = len(src) - len(keep)
            if disk_removed:
                full.write_text(("\n".join(keep) + "\n") if keep else "")
    except Exception as e:  # noqa: BLE001
        logger.warning("trim %s: disk rewrite failed: %s", run_id, e)

    # 3) S3 logs.txt for a finalized run (its local disk copy may be gone after teardown).
    if terminal and row.s3_prefix:
        try:
            target = await _training_s3_target(row.storage_id)
            s3_key = row.s3_prefix + "logs.txt"
            data = await asyncio.to_thread(s3_get_bytes, s3_key, target)
            if data:
                src = data.decode("utf-8", "replace").splitlines()
                keep = [ln for ln in src if not _is_progress_line(ln)]
                s3_removed = len(src) - len(keep)
                if s3_removed:
                    await asyncio.to_thread(
                        s3_put_text, s3_key, ("\n".join(keep) + "\n") if keep else "", target)
        except Exception as e:  # noqa: BLE001
            logger.warning("trim %s: S3 logs.txt rewrite failed: %s", run_id, e)

    removed = redis_removed or disk_removed or s3_removed
    return {"ok": True, "removed": removed,
            "redis_removed": redis_removed, "disk_removed": disk_removed, "s3_removed": s3_removed}


@router.get("/{run_id}/files", response_model=list[TrainingFile])
async def list_training_files(
    run_id: str,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    row = await _owned(run_id, user, session)
    target = await _training_s3_target(row.storage_id)
    out: list[TrainingFile] = []
    try:
        for obj in s3_list(row.s3_prefix, target=target):
            name = obj["key"][len(row.s3_prefix):] if obj["key"].startswith(row.s3_prefix) else obj["key"]
            out.append(TrainingFile(
                name=name, size=obj.get("size", 0),
                modified=obj.get("modified", ""),
                download_url=s3_presign_get(obj["key"], target=target),
            ))
    except Exception as e:  # noqa: BLE001
        logger.warning("training %s: file list failed: %s", run_id, e)
    return out


# ---------- live GPU utilisation (only the run's GPUs) ------------------


def _run_gpu_id_csv(row: TrainingRun) -> Optional[str]:
    """The physical GPU ids this run actually uses (for `nvidia-smi -i`):
    the visible_devices pin, else the sweep's GPU set, else None (= all)."""
    if row.visible_devices and row.visible_devices.strip():
        return ",".join(x.strip() for x in row.visible_devices.split(",") if x.strip())
    sg = (row.config_json or {}).get("sweep_gpus")
    if sg:
        return ",".join(str(x) for x in sg)
    return None


async def _runpod_pod_ssh(
    run_id: str, provider_id: Optional[str], pod_id: Optional[str], key_name: str = "id_ed25519",
) -> Optional[tuple[str, int, str, str]]:
    """(host, port, "root", key_filename) for a live RunPod pod, reusing the
    ephemeral key minted at provision. None if pod / SSH / key is unresolvable.
    Shared by the run's training pod (_resolve_run_ssh) and a fresh try-it pod."""
    if not pod_id:
        return None
    try:
        api_key = await compute._resolve_api_key(provider_id)
        async with compute._client(api_key=api_key) as cli:
            pr = await cli.get(f"/pods/{pod_id}")
        ip, port = compute._extract_ssh(pr.json() or {})
    except Exception:
        return None
    key = str(_work_dir(run_id) / key_name)
    if not ip or not port or not Path(key).exists():
        return None
    return ip, int(port), "root", key


async def _resolve_run_ssh(row: TrainingRun) -> Optional[tuple[str, int, str, str]]:
    """(host, port, user, key_filename) for the run's box — VM provider or the
    RunPod pod — reusing the ephemeral key minted at launch. None if unresolvable."""
    work = _work_dir(row.id)
    prov = None
    if row.provider_id:
        async with session_factory()() as s:
            prov = await s.get(Provider, row.provider_id)
    if prov is not None and prov.kind == "vm":
        pc = prov.config or {}
        key = str(work / "vm_key")
        if not Path(key).exists():
            enc = pc.get("private_key_enc")
            if not enc:
                return None
            Path(key).write_text(crypto.decrypt(enc))
            os.chmod(key, 0o600)
        return pc.get("host"), int(pc.get("port") or 22), pc.get("user") or "root", key
    # RunPod pod (the run's training pod)
    return await _runpod_pod_ssh(row.id, row.provider_id, row.runpod_pod_id)


async def _resolve_tryit_ssh(row: TrainingRun) -> Optional[tuple[str, int, str, str]]:
    """SSH coords for the run's *try-it* compute, which (unlike _resolve_run_ssh) may
    be a target the user chose at load time — decoupled from where the run trained:
    a fresh RunPod pod (cloud) or any registered VM provider (vm). Falls back to the
    run's own box when no try-it target is set (legacy cloud-ASR / VM-trained runs)."""
    ti = _tryit_state(row)
    target = ti.get("target")
    if not target:
        return await _resolve_run_ssh(row)
    if target == "vm":
        async with session_factory()() as s:
            prov = await s.get(Provider, ti.get("provider_id"))
        if prov is None:
            return None
        return await _resolve_provider_ssh(prov, row.id)
    if target == "cloud":
        # pod_id + the RunPod account live in try-it state — NOT row.runpod_pod_id.
        return await _runpod_pod_ssh(row.id, ti.get("provider_id"), ti.get("pod_id"))
    return None


async def _resolve_provider_ssh(prov: Provider, run_id: str) -> Optional[tuple[str, int, str, str]]:
    """(host, port, user, key_filename) for an ARBITRARY registered VM provider — the
    same shape `_resolve_run_ssh` returns, but for a VM the export was explicitly
    routed to (Run-on → Bare metal) rather than the run's own box. Decrypts the
    provider's stored private key into the run's scratch dir. None if unconfigured."""
    if prov.kind != "vm":
        return None
    pc = prov.config or {}
    enc = pc.get("private_key_enc")
    if not enc:
        return None
    key = str(_work_dir(run_id) / "label_vm_key")
    Path(key).write_text(crypto.decrypt(enc))
    os.chmod(key, 0o600)
    return pc.get("host"), int(pc.get("port") or 22), pc.get("user") or "root", key


def _label_build_tts_venv_ssh(host: str, port: int, user: str, key_filename: str,
                              run_id: str, cfg: dict, line_sink=None) -> None:
    """On a freshly-spawned pod: ship tts_finetune.py + run `--deps-only` (reusing the
    training path's uv bootstrap) to build the TTS venv the export's synthesis needs.
    Blocking — call via to_thread. Mirrors `_tryit_build_venv_ssh` (the ASR twin).
    `line_sink(str)` (if given) gets every install line so the caller can stream the
    slow build; on failure the RuntimeError carries the output tail (the real
    pip/uv/build error — not just an opaque rc)."""
    import tempfile
    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        base = _trainer_script_path().parent  # gateway/gateway/training/
        _ssh_put(cli, str(base / "tts_finetune.py"), "/tmp/sgpu_label_tts.py")
        dconf = {"venv_path": (cfg.get("venv_path") or "/share/autotrain-tts").rstrip("/"),
                 "work_dir": (cfg.get("work_dir") or "/share").rstrip("/"),
                 # codec variant drives which neucodec pkg the venv installs.
                 "tts_codec": (cfg.get("tts_codec") or "neucodec")}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps(dconf)); local = f.name
        try:
            _ssh_put(cli, local, "/tmp/sgpu_label_deps.json")
        finally:
            try:
                os.unlink(local)
            except OSError:
                pass
        user_env = _render_env_exports(cfg.get("env_vars") or {})
        deps_cmd = (f"{user_env}{_UV_BOOTSTRAP}"
                    "python -u /tmp/sgpu_label_tts.py --deps-only --config /tmp/sgpu_label_deps.json")
        tail: list[str] = []

        def on_line(line: str) -> None:
            if line_sink is not None:
                try:
                    line_sink(line)
                except Exception:  # noqa: BLE001
                    pass
            tail.append(line)            # keep a rolling tail (the error is at the END)
            if len(tail) > 80:
                del tail[0]

        rc = _ssh_run_stream(cli, deps_cmd, on_line)
        if rc != 0:
            detail = "\n".join(tail[-25:]).strip()
            raise RuntimeError(f"TTS deps install failed on the label-export pod (rc={rc})"
                               + (f":\n{detail}" if detail else ""))
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


def _build_llm_venv_ssh(host: str, port: int, user: str, key_filename: str,
                        run_id: str, cfg: dict, base_model: str,
                        venv_path: Optional[str] = None, line_sink=None) -> None:
    """On a freshly-spawned pod: ship llm_finetune.py + the llm/ dir and run `--deps-only`
    (the exact cloud-training deps phase) to build the per-arch training venv the merge /
    vLLM-serve needs (torch + transformers + peft + boto3 + huggingface_hub). FA4 is
    skipped (gemma_fa4=False): the merge folds base bf16 without the flash-attn cute fork,
    so we avoid its slow git build. On a VM the venv already exists, so this is
    cloud-only. Blocking — call via to_thread; on failure the RuntimeError carries the
    install tail (the real pip/uv error, not just an opaque rc). Mirrors
    `_label_build_tts_venv_ssh` (the TTS twin)."""
    arch = _llm_arch(base_model)
    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        base = _trainer_script_path().parent  # gateway/gateway/training/
        stage = f"/tmp/sgpu_llm_deps_{run_id}"
        _ssh_exec(cli, f"mkdir -p {stage}")
        _ssh_put(cli, str(base / "llm_finetune.py"), f"{stage}/llm_finetune.py")
        _ssh_put_dir_tar(cli, str(base / "llm"), f"{stage}/llm")
        dconf = {
            "base_model": base_model,
            "venv_path": (venv_path or _llm_venv(arch)).rstrip("/"),
            "work_dir": (cfg.get("work_dir") or "/share").rstrip("/"),
            "gemma_fa4": False,   # merge needs no FA4 kernel → skip the slow cute-fork build
        }
        _ssh_put_bytes(cli, json.dumps(dconf).encode("utf-8"), f"{stage}/deps.json")
        user_env = _render_env_exports(cfg.get("env_vars") or {})
        deps_cmd = (f"{user_env}{_UV_BOOTSTRAP}"
                    f"python -u {stage}/llm_finetune.py --deps-only --config {stage}/deps.json")
        tail: list[str] = []

        def on_line(line: str) -> None:
            if line_sink is not None:
                try:
                    line_sink(line)
                except Exception:  # noqa: BLE001
                    pass
            tail.append(line)            # keep a rolling tail (the error is at the END)
            if len(tail) > 80:
                del tail[0]

        rc = _ssh_run_stream(cli, deps_cmd, on_line)
        if rc != 0:
            detail = "\n".join(tail[-25:]).strip()
            raise RuntimeError(f"LLM deps install failed on the pod (rc={rc})"
                               + (f":\n{detail}" if detail else ""))
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


# ---------- cloud try-it pod (ASR): a finished RunPod-cloud run's training pod is
# torn down, so "Try it" provisions a fresh, short-lived pod on demand, builds the
# Whisper venv (--deps-only), and keeps it warm for transcribes. It auto-tears-down
# after _TRYIT_IDLE_S idle (bumped on each transcribe) or on Unload — the janitor +
# startup cleanup reap it (restart-safe), so it can't bill forever. State lives on
# the run's result_json["tryit"] = {pod_id, provider_id, phase, message, expires_at};
# phase ∈ provisioning|installing|ready|error. VM runs keep using the SSH path above.
_TRYIT_IDLE_S = 15 * 60          # auto-teardown after this many idle seconds (user-chosen)
_tryit_tasks: dict[str, "asyncio.Task"] = {}   # in-flight setup tasks (dedup + cancel)


def _tryit_state(row) -> dict:
    return dict((getattr(row, "result_json", None) or {}).get("tryit") or {})


def _tryit_exp() -> str:
    return _iso(datetime.now(timezone.utc) + timedelta(seconds=_TRYIT_IDLE_S)) or ""


async def _tryit_save(run_id: str, **fields) -> None:
    """Merge fields into the run's result_json['tryit'] (JSON; no migration)."""
    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
        if row is None:
            return
        rj = dict(row.result_json or {})
        ti = dict(rj.get("tryit") or {})
        ti.update(fields)
        rj["tryit"] = ti
        row.result_json = rj
        # NB: the try-it pod id lives ONLY in tryit state (resolved by
        # _resolve_tryit_ssh) — we do NOT touch row.runpod_pod_id, which is the
        # run's training-pod metadata (overwriting it conflated two lifecycles).
        await s.commit()


async def _tryit_teardown(run_id: str) -> None:
    """Terminate the try-it pod (if any) and clear the state. Idempotent + best-effort.
    Cloud target → terminate the RunPod pod; VM target → SIGTERM the persistent worker
    on the chosen VM (frees its GPU). Leaves row.runpod_pod_id (training metadata) alone."""
    task = _tryit_tasks.pop(run_id, None)
    if task is not None and not task.done():
        task.cancel()
    vm_provider_id = None
    cfg: dict = {}
    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
        if row is None:
            return
        ti = dict((row.result_json or {}).get("tryit") or {})
        pod_id, prov = ti.get("pod_id"), ti.get("provider_id")
        target, kind = ti.get("target"), (ti.get("kind") or row.task_type or "asr")
        if target == "vm":
            vm_provider_id = prov
            cfg = dict(row.config_json or {})
        rj = dict(row.result_json or {})
        rj.pop("tryit", None)
        row.result_json = rj
        await s.commit()
    if pod_id:
        try:
            api_key = await compute._resolve_api_key(prov)
            await _terminate_pod(api_key, pod_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("try-it pod %s teardown failed: %s", pod_id, e)
    elif vm_provider_id:
        # VM target: stop the resident worker on the chosen VM so it doesn't hold the GPU.
        try:
            async with session_factory()() as s:
                vprov = await s.get(Provider, vm_provider_id)
            ssh = await _resolve_provider_ssh(vprov, run_id) if vprov else None
            if ssh:
                await asyncio.to_thread(_playground_stop_ssh, *ssh, run_id, cfg, kind)
        except Exception as e:  # noqa: BLE001
            logger.warning("try-it VM worker teardown failed for %s: %s", run_id, e)


def _tryit_build_venv_ssh(host: str, port: int, user: str, key_filename: str,
                          run_id: str, cfg: dict) -> None:
    """On the fresh pod: ship whisper_finetune.py + run `--deps-only` (reusing the
    training path's uv bootstrap) to build the Whisper venv. Blocking — to_thread."""
    import tempfile
    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        _ssh_put(cli, str(_trainer_script_path()), "/tmp/sgpu_tryit_whisper.py")
        dconf = {"venv_path": cfg.get("venv_path") or "/share/autotrain-whisper",
                 "work_dir": cfg.get("work_dir") or "/share"}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps(dconf)); local = f.name
        try:
            _ssh_put(cli, local, "/tmp/sgpu_tryit_deps.json")
        finally:
            try:
                os.unlink(local)
            except OSError:
                pass
        user_env = _render_env_exports(cfg.get("env_vars") or {})
        deps_cmd = (f"{user_env}{_UV_BOOTSTRAP}"
                    "python -u /tmp/sgpu_tryit_whisper.py --deps-only --config /tmp/sgpu_tryit_deps.json")
        rc = _ssh_run_stream(cli, deps_cmd, lambda _l: None)
        if rc != 0:
            raise RuntimeError(f"Whisper deps install failed on the try-it pod (rc={rc})")
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


def _cfg_with_hf_token(cfg: dict, token: Optional[str]) -> dict:
    """Return a shallow copy of `cfg` with HF_TOKEN merged into env_vars (so the box job's
    `_render_env_exports` prefix exports it — and its subprocesses inherit it — letting a
    gated base-model download work). No-op when `token` is empty. NOT persisted: only the
    local cfg handed to an SSH runner is touched, never written back to the DB."""
    if not token:
        return cfg
    return {**cfg, "env_vars": {**(cfg.get("env_vars") or {}), "HF_TOKEN": token}}


async def _tryit_cloud_setup(run_id: str, *, gpu_type: str, gpu_count: int, cloud_type: str,
                             provider_id: Optional[str], kind: str,
                             base_hf_token: Optional[str] = None) -> None:
    """Provision a fresh pod for a cloud try-it (ASR / TTS / LLM) on the chosen GPU /
    account, build the right venv, and keep it warm. ASR/TTS load per-request after the
    venv build → ready immediately. LLM launches a detached vLLM orchestrator (download
    LoRA → merge → serve) and polls its ready-file, so it stays 'installing' until vLLM
    is healthy. Updates result_json['tryit'].phase as it goes; tears the pod down on
    failure so a half-provisioned pod can't leak. The pod id lives in tryit state
    (resolved by _resolve_tryit_ssh)."""
    pod_id = None
    try:
        async with session_factory()() as s:
            row = await s.get(TrainingRun, run_id)
            cfg = dict(row.config_json or {})
            storage = await s.get(Storage, row.storage_id) if row.storage_id else None
            model_s3 = ((row.result_json or {}).get("artifact") or {}).get("s3_uri")
        creds = _s3_creds_from_storage(storage)
        await _tryit_save(run_id, target="cloud", provider_id=provider_id, kind=kind,
                          gpu_type=gpu_type, gpu_count=gpu_count, cloud_type=cloud_type,
                          phase="provisioning", message="starting a GPU pod …", expires_at=_tryit_exp())
        api_key = await compute._resolve_api_key(provider_id)
        key_filename, pub = _gen_ssh_key(_work_dir(run_id))
        # LLM serves via vLLM ≥0.23 → cu1300 image; it also holds the base model for the
        # merge, so give it generous disk. ASR/TTS keep the small cu1281 default.
        image = cfg.get("image") or (LLM_VLLM_IMAGE if kind == "llm" else DEFAULT_IMAGE)
        disk_gb = int(cfg.get("disk_gb") or (250 if kind == "llm" else 60))
        volume_gb = int(cfg.get("volume_gb") or (250 if kind == "llm" else 80))
        runpod_id, host, port, _cost = await _provision_pod(
            api_key, f"sgpu-tryit-{run_id}", image,
            gpu_type, max(1, int(gpu_count)), (cloud_type or "SECURE") == "SECURE",
            disk_gb, volume_gb, pub,
        )
        pod_id = runpod_id
        if kind == "llm":
            await _tryit_save(run_id, pod_id=runpod_id, phase="installing",
                              message="installing the LLM stack (torch + transformers + vLLM) — first build is slow …",
                              expires_at=_tryit_exp())
            async with session_factory()() as s:
                r = await s.get(TrainingRun, run_id)
            ti = _tryit_state(r)   # gpus / vllm knobs stashed by playground_start
            base_model = ti.get("base_model") or cfg.get("base_model") or "google/gemma-4-31B-it"
            gpus = ti.get("gpus") or ",".join(str(i) for i in range(max(1, int(gpu_count))))
            # The merge downloads the (usually gated) base model → inject HF_TOKEN into the
            # orchestrator's env. Fresh pod = no HF cache, so this is required for gated models.
            cfg = _cfg_with_hf_token(cfg, base_hf_token)
            # Build the per-arch venv on the fresh pod (the /share venv doesn't exist here),
            # then launch the same detached orchestrator the VM path uses.
            await asyncio.to_thread(_build_llm_venv_ssh, host, int(port), "root", key_filename,
                                    run_id, cfg, base_model, cfg.get("venv_path"))
            await _tryit_save(run_id, phase="installing",
                              message="merging LoRA + starting vLLM — first load is slow …",
                              expires_at=_tryit_exp())
            await asyncio.to_thread(
                _llm_playground_start_ssh, host, int(port), "root", key_filename, run_id,
                model_s3, creds, cfg, base_model, gpus, run_id,
                int(ti.get("max_model_len") or 16384), ti.get("vllm_args") or "",
                ti.get("vllm_version") or "0.23.0")
            # vLLM readiness is async: poll the pod's ready-file, bumping expiry so the idle
            # reaper can't kill a still-loading pod. This task stays alive until ready.
            ssh_pod = (host, int(port), "root", key_filename)
            deadline = time.time() + 45 * 60
            while time.time() < deadline:
                await _tryit_save(run_id, expires_at=_tryit_exp())
                await asyncio.sleep(20)
                try:
                    stt = await asyncio.to_thread(_playground_status_ssh, *ssh_pod, run_id, cfg, "llm")
                except Exception:  # noqa: BLE001
                    stt = {}
                if stt.get("ready"):
                    await _tryit_save(run_id, phase="ready",
                                      message="ready — type a prompt and chat", expires_at=_tryit_exp())
                    return
            raise RuntimeError("vLLM didn't become ready within 45 min")
        stack = "TTS stack (torch + neucodec)" if kind == "tts" else "Whisper stack (torch + fasttext)"
        await _tryit_save(run_id, pod_id=runpod_id, phase="installing",
                          message=f"installing the {stack} — first load ~10 min …",
                          expires_at=_tryit_exp())
        # TTS reuses the label-export TTS venv bootstrap (ships tts_finetune.py +
        # --deps-only); ASR builds the Whisper venv. Same (host,port,user,key,run_id,cfg) shape.
        build = _label_build_tts_venv_ssh if kind == "tts" else _tryit_build_venv_ssh
        await asyncio.to_thread(build, host, int(port), "root", key_filename, run_id, cfg)
        await _tryit_save(run_id, phase="ready",
                          message=("ready — type text and synthesize" if kind == "tts"
                                   else "ready — upload a clip and transcribe"),
                          expires_at=_tryit_exp())
    except asyncio.CancelledError:
        await _tryit_teardown(run_id)
        raise
    except Exception as e:  # noqa: BLE001
        # Surface the error AND make sure the pod (if created) is gone.
        if pod_id:
            try:
                await _terminate_pod(await compute._resolve_api_key(provider_id), pod_id)
            except Exception:  # noqa: BLE001
                pass
        await _tryit_save(run_id, phase="error", pod_id=None,
                          message=f"try-it setup failed: {str(e)[:280]}")
    finally:
        _tryit_tasks.pop(run_id, None)


def _tryit_status(row) -> "PlaygroundStatus":
    """Map a cloud try-it pod's state → the PlaygroundStatus the UI polls."""
    ti = _tryit_state(row)
    phase = ti.get("phase")
    msg = ti.get("message") or ""
    return PlaygroundStatus(
        running=phase in ("provisioning", "installing", "ready"),
        ready=(phase == "ready"),
        kind=ti.get("kind") or (getattr(row, "task_type", None) or "asr"),
        logs=([f"[try-it] {msg}"] if msg else []),
    )


async def _reap_idle_tryit_pods(force_all: bool = False) -> int:
    """Terminate try-it pods whose idle window has elapsed (periodic janitor). With
    force_all=True, terminate every try-it pod regardless of expiry — used on startup,
    where the in-memory setup tasks are gone, so any surviving pod is orphaned."""
    now = datetime.now(timezone.utc)
    async with session_factory()() as s:
        rows = (await s.execute(
            select(TrainingRun).where(TrainingRun.status == "done")
        )).scalars().all()
    reaped = 0
    for row in rows:
        ti = dict((row.result_json or {}).get("tryit") or {})
        if not ti.get("pod_id"):
            continue
        if row.id in _tryit_tasks:
            continue  # setup still in flight (this gateway) → leave it
        try:
            exp = datetime.fromisoformat(str(ti.get("expires_at") or "").replace("Z", "+00:00"))
        except ValueError:
            exp = now  # malformed → reap
        if force_all or exp <= now:
            await _tryit_teardown(row.id)
            reaped += 1
    return reaped


# ---------- persistent "Try it" worker (loaded model, served over a Unix socket,
# lifecycle managed over SSH — the small sibling of the serverless worker) --------


def _llm_tryit_port(run_id: str) -> int:
    """Deterministic localhost port for a run's vLLM try-it server (stable across
    gateway processes, so status/chat reach the same port the loader bound)."""
    import hashlib
    return 18000 + int(hashlib.md5(run_id.encode()).hexdigest()[:4], 16) % 2000


def _playground_paths(cfg: dict, run_id: str, kind: str) -> dict:
    """VM-side paths for a run's persistent Try-it server. model_dir matches the
    one-shot cache dir so the persistent + one-shot paths share the weights."""
    work = (cfg.get("work_dir") or "/share").rstrip("/")
    base = {
        "ready": f"/tmp/sgpu_tryit_{run_id}.sock.ready",
        "pid": f"/tmp/sgpu_tryit_{run_id}.pid",
        "log": f"/tmp/sgpu_tryit_{run_id}.server.log",
        "cfg": f"/tmp/sgpu_tryit_{run_id}.server.json",
    }
    if kind == "llm":
        # LLM: merge LoRA → save → serve with vLLM. The training venv runs the merge;
        # vLLM serves from its own dedicated venv. Merged model is cached. Default to
        # the run's ARCH venv (/share/autotrain-llm-<arch>, where training built its
        # deps) — the generic /share/autotrain-llm doesn't exist for arch runs.
        tdir = f"{work}/sgpu-llm-tryit/{run_id}"
        _arch = _llm_arch(cfg.get("base_model"))
        venv = (cfg.get("venv_path") or _llm_venv(_arch)).rstrip("/")
        return {
            **base,
            "tryit_dir": tdir,                   # the whole per-run tree (merged + ckpt) — swept on unload
            "model_dir": f"{tdir}/merged",      # vLLM serves this
            "ckpt_dir": f"{tdir}/ckpt",          # downloaded lora.pt + meta
            "vllm_venv": "/share/autotrain-llm-vllm",
            "port": _llm_tryit_port(run_id),
            "venv": venv,
            "py": f"{venv}/bin/python",
        }
    md = f"{work}/sgpu-tts-tryit/{run_id}" if kind == "tts" else f"{work}/sgpu-tryit/{run_id}"
    # Reuse the run's training uv venv (persisted in config_json by run_training);
    # fall back to the per-task default the trainer also uses. Never machine python.
    venv = (cfg.get("venv_path")
            or ("/share/autotrain-tts" if kind == "tts" else "/share/autotrain-whisper")).rstrip("/")
    return {
        **base,
        "sock": f"/tmp/sgpu_tryit_{run_id}.sock",
        "model_dir": md,
        "venv": venv,
        "py": f"{venv}/bin/python",
    }


def _ssh_capture(cli, command: str) -> str:
    """Run a command over SSH, return its combined stdout+stderr as one string.
    _ssh_run_stream hands us newline-stripped lines, so rejoin WITH newlines — else
    multi-line output (e.g. a run log) collapses and line-anchored parsing breaks."""
    chunks: list[str] = []
    _ssh_run_stream(cli, command, lambda l: chunks.append(l))
    return "\n".join(chunks)


def _persistent_status(cli, paths: dict) -> dict:
    """{running, ready, device?, kind?, logs[]} in one SSH probe — the log tail lets
    the UI show the worker's load progress (download / shard load) while polling."""
    cmd = (f'P="$(cat {paths["pid"]} 2>/dev/null)"; '
           f'if [ -n "$P" ] && kill -0 "$P" 2>/dev/null; then '
           f'if [ -f {paths["ready"]} ]; then echo "READY $(cat {paths["ready"]})"; '
           f'else echo LOADING; fi; else echo DOWN; fi; '
           f'echo "@@STATUSLOG@@"; tail -n 300 {paths["log"]} 2>/dev/null')
    head, _, logpart = _ssh_capture(cli, cmd).partition("@@STATUSLOG@@")
    line = next((x for x in head.strip().splitlines() if x.startswith(("READY", "LOADING", "DOWN"))), "DOWN")
    logs = [x for x in logpart.splitlines() if x.strip()][-300:]
    base = {"logs": logs}
    if line.startswith("READY"):
        try:
            meta = json.loads(line[len("READY "):])
        except Exception:  # noqa: BLE001
            meta = {}
        return {**base, "running": True, "ready": True, "device": meta.get("device"), "kind": meta.get("kind")}
    return {**base, "running": line.startswith("LOADING"), "ready": False}


def _persistent_request(cli, paths: dict, req: dict) -> tuple[dict, list[str]]:
    """Forward one request to the loaded server over its socket; return (resp, log_tail)."""
    import shlex
    _ssh_put_bytes(cli, json.dumps(req).encode(), "/tmp/sgpu_tryit_req.json")
    sock = shlex.quote(paths["sock"])
    # Same uv venv as the run (the client is stdlib-only, but stay off machine python).
    cmd = f'{shlex.quote(paths["py"])} /tmp/sgpu_tryit_client.py --sock {sock} --req /tmp/sgpu_tryit_req.json'
    got: dict = {"resp": None}

    def on_line(l: str) -> None:
        s = l.strip()
        if s.startswith("{") and got["resp"] is None:
            try:
                got["resp"] = json.loads(s)
            except Exception:  # noqa: BLE001
                pass
    _ssh_run_stream(cli, cmd, on_line)
    tail = [x for x in _ssh_capture(cli, f"tail -n 6 {paths['log']} 2>/dev/null").splitlines() if x.strip()]
    if got["resp"] is None:
        raise RuntimeError("persistent worker did not respond (it may have crashed — check Restart)")
    return got["resp"], tail


def _playground_start_ssh(host, port, user, key_filename, run_id, kind, model_s3, creds, cfg, gpu=None,
                          idle_timeout=300):
    """Launch the persistent server on the VM (nohup, own session) and record its
    pid. Returns the status after launch (loading). idle_timeout=0 disables the
    server's idle auto-unload."""
    import shlex
    import tempfile
    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        paths = _playground_paths(cfg, run_id, kind)
        # already up? just report.
        st = _persistent_status(cli, paths)
        if st["running"]:
            return st
        base = _trainer_script_path().parent
        _ssh_put(cli, str(base / "tryit_server.py"), "/tmp/sgpu_tryit_server.py")
        _ssh_put(cli, str(base / "tryit_client.py"), "/tmp/sgpu_tryit_client.py")
        sconf = {
            "kind": kind, "model_s3": model_s3,
            "region": creds.get("region"), "endpoint": creds.get("endpoint"),
            "access_key": creds.get("access_key"), "secret_key": creds.get("secret_key"),
            "model_dir": paths["model_dir"], "sock": paths["sock"], "gpu": gpu or "auto",
            "language": cfg.get("language") or None, "task": cfg.get("task") or "transcribe",
            "idle_timeout": idle_timeout, "pid": paths["pid"],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps(sconf)); local_cfg = f.name
        try:
            _ssh_put(cli, local_cfg, paths["cfg"])
        finally:
            try:
                os.unlink(local_cfg)
            except OSError:
                pass
        py = paths["py"]  # the run's training uv venv (asr→whisper, tts→tts); never machine python
        user_env = _render_env_exports(cfg.get("env_vars") or {})
        # setsid → its own session/pgid so stop can kill the whole group cleanly.
        cmd = (f'{user_env}rm -f {paths["ready"]}; PY="{py}"; '
               f'if [ ! -x "$PY" ]; then echo "venv python not found at $PY — train this run first (it creates the uv venv)" > {paths["log"]}; exit 1; fi; '
               f'setsid nohup "$PY" -u /tmp/sgpu_tryit_server.py --config {shlex.quote(paths["cfg"])} '
               f'> {paths["log"]} 2>&1 & echo $! > {paths["pid"]}; sleep 0.3; cat {paths["pid"]}')
        _ssh_capture(cli, cmd)
        return _persistent_status(cli, paths)
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


def _playground_stop_ssh(host, port, user, key_filename, run_id, cfg, kind):
    """Stop the run's persistent server — SIGTERM its process group (own session),
    then clean up the socket/pid/ready files. Precise: only the recorded pid."""
    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        paths = _playground_paths(cfg, run_id, kind)
        sock = paths.get("sock", "")  # LLM (vLLM) has no unix socket
        # LLM unload also reclaims the per-run merged model (~59GB) + downloaded ckpt
        # from /share — KEEPS the shared vLLM venv (/share/autotrain-llm-vllm, reused by
        # every LLM run). Done AFTER the kill so vLLM's mmap'd safetensors are released.
        # run_id is `train-<hex>` (no shell metachars), so the controlled path is safe.
        extra = (f' rm -rf {paths["tryit_dir"]} 2>/dev/null;'
                 if (kind == "llm" and paths.get("tryit_dir")) else "")
        cmd = (f'P="$(cat {paths["pid"]} 2>/dev/null)"; '
               f'if [ -n "$P" ]; then kill -TERM -"$P" 2>/dev/null || kill -TERM "$P" 2>/dev/null; sleep 1; '
               f'kill -KILL -"$P" 2>/dev/null || true; fi; sleep 1;'
               f'rm -f {sock} {paths["ready"]} {paths["pid"]};{extra} echo stopped')
        _ssh_capture(cli, cmd)
        return {"running": False, "ready": False}
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


def _playground_status_ssh(host, port, user, key_filename, run_id, cfg, kind):
    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        return _persistent_status(cli, _playground_paths(cfg, run_id, kind))
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


def _llm_playground_start_ssh(host, port, user, key_filename, run_id, model_s3, creds, cfg,
                              base_model, gpus, served_name, max_model_len=16384, vllm_args="",
                              vllm_version="0.23.0", merge_dtype="fp16"):
    """LLM try-it: launch the detached orchestrator (download LoRA → merge → save →
    vLLM serve eager) on the VM and record its pid. The orchestrator logs every step
    to paths['log'] (the status tab tails it) and writes paths['ready'] once vLLM is
    healthy. The merged model is cached so a re-load skips straight to serving."""
    import shlex
    import tempfile
    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        paths = _playground_paths(cfg, run_id, "llm")
        st = _persistent_status(cli, paths)
        if st["running"]:
            return st
        # Ship the llm/ trainer dir (merge_infer.py + llm_playground.py + vendored code)
        # to a stable per-run location the orchestrator runs from.
        code_dir = f"/tmp/sgpu_llm_code_{run_id}"
        base = _trainer_script_path().parent  # gateway/gateway/training/
        _ssh_put_dir_tar(cli, str(base / "llm"), code_dir)
        gpu_list = [g for g in str(gpus or "").split(",") if g.strip()]
        sconf = {
            "model_s3": model_s3,
            "region": creds.get("region"), "endpoint": creds.get("endpoint"),
            "access_key": creds.get("access_key"), "secret_key": creds.get("secret_key"),
            "base_model": base_model,
            "merged_dir": paths["model_dir"], "work_dir": paths["ckpt_dir"],
            "llm_dir": code_dir, "train_py": paths["py"], "vllm_venv": paths["vllm_venv"],
            "port": paths["port"], "tp": max(1, len(gpu_list)), "gpus": gpus or "",
            "max_model_len": int(max_model_len), "gpu_mem_util": 0.90,
            "served_model_name": served_name, "ready_file": paths["ready"],
            "vllm_args": vllm_args or "",
            "vllm_version": (vllm_version or "0.23.0"),
            "merge_dtype": (merge_dtype or "fp16"),
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps(sconf)); local_cfg = f.name
        try:
            _ssh_put(cli, local_cfg, paths["cfg"])
        finally:
            try:
                os.unlink(local_cfg)
            except OSError:
                pass
        py = paths["py"]  # training venv (transformers 5.5.0 + boto3) runs the merge
        user_env = _render_env_exports(cfg.get("env_vars") or {})
        cmd = (f'{user_env}rm -f {paths["ready"]}; PY="{py}"; '
               f'if [ ! -x "$PY" ]; then echo "training venv python not found at $PY — train this run first" > {paths["log"]}; exit 1; fi; '
               f'setsid nohup "$PY" -u {code_dir}/llm_playground.py --config {shlex.quote(paths["cfg"])} '
               f'> {paths["log"]} 2>&1 & echo $! > {paths["pid"]}; sleep 0.3; cat {paths["pid"]}')
        _ssh_capture(cli, cmd)
        return _persistent_status(cli, paths)
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


# ---------- persistent NeuCodec decoder for a tts_packed DATASET ---------------
# A small sibling of the run try-it server: keeps NeuCodec resident on a chosen VM
# so each "play utt N" click on a packed dataset decodes that utterance's speech
# codes straight to audio (no LM, no S3 model — NeuCodec is pulled from HF). Reuses
# the same persistent-server machinery; only the cfg (kind=tts_decode, no model_s3)
# and the dataset-scoped paths differ.


def _dataset_decoder_paths(dataset_id: str, venv: str) -> dict:
    v = (venv or "/share/autotrain-tts").rstrip("/")
    return {
        "sock": f"/tmp/sgpu_dsdec_{dataset_id}.sock",
        "ready": f"/tmp/sgpu_dsdec_{dataset_id}.sock.ready",
        "pid": f"/tmp/sgpu_dsdec_{dataset_id}.pid",
        "log": f"/tmp/sgpu_dsdec_{dataset_id}.server.log",
        "cfg": f"/tmp/sgpu_dsdec_{dataset_id}.server.json",
        "venv": v,
        "py": f"{v}/bin/python",
    }


def dataset_decoder_start_ssh(host, port, user, key_filename, dataset_id, venv,
                              gpu=None, idle_timeout=600, env_vars=None):
    """Launch the persistent NeuCodec decoder on the VM (nohup, own session); idle
    auto-unload after idle_timeout s. Returns the status after launch (loading)."""
    import shlex
    import tempfile
    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        paths = _dataset_decoder_paths(dataset_id, venv)
        st = _persistent_status(cli, paths)
        if st["running"]:
            return st
        base = _trainer_script_path().parent
        _ssh_put(cli, str(base / "tryit_server.py"), "/tmp/sgpu_tryit_server.py")
        _ssh_put(cli, str(base / "tryit_client.py"), "/tmp/sgpu_tryit_client.py")
        sconf = {"kind": "tts_decode", "sock": paths["sock"], "gpu": gpu or "auto",
                 "idle_timeout": idle_timeout, "pid": paths["pid"]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps(sconf)); local_cfg = f.name
        try:
            _ssh_put(cli, local_cfg, paths["cfg"])
        finally:
            try:
                os.unlink(local_cfg)
            except OSError:
                pass
        user_env = _render_env_exports(env_vars or {})
        cmd = (f'{user_env}rm -f {paths["ready"]}; PY="{paths["py"]}"; '
               f'if [ ! -x "$PY" ]; then echo "venv python not found at $PY — the TTS venv ({venv}) must exist on this box" > {paths["log"]}; exit 1; fi; '
               f'setsid nohup "$PY" -u /tmp/sgpu_tryit_server.py --config {shlex.quote(paths["cfg"])} '
               f'> {paths["log"]} 2>&1 & echo $! > {paths["pid"]}; sleep 0.3; cat {paths["pid"]}')
        _ssh_capture(cli, cmd)
        return _persistent_status(cli, paths)
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


def dataset_decoder_status_ssh(host, port, user, key_filename, dataset_id, venv):
    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        return _persistent_status(cli, _dataset_decoder_paths(dataset_id, venv))
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


def dataset_decoder_decode_ssh(host, port, user, key_filename, dataset_id, venv, text):
    """Send one utterance's text (with its <|s_N|> codes) to the resident decoder
    and return its {wav_b64, sample_rate, …} response."""
    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        resp, _tail = _persistent_request(cli, _dataset_decoder_paths(dataset_id, venv), {"text": text})
        return resp
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


def dataset_decoder_stop_ssh(host, port, user, key_filename, dataset_id, venv):
    import shlex
    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        paths = _dataset_decoder_paths(dataset_id, venv)
        cmd = (f'P="$(cat {paths["pid"]} 2>/dev/null)"; '
               f'if [ -n "$P" ]; then kill -TERM -"$P" 2>/dev/null || kill -TERM "$P" 2>/dev/null; sleep 1; '
               f'kill -KILL -"$P" 2>/dev/null || true; fi; '
               f'rm -f {paths["sock"]} {paths["ready"]} {paths["pid"]}; echo stopped')
        _ssh_capture(cli, cmd)
        return {"running": False, "ready": False}
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


def _run_transcribe_ssh(host: str, port: int, user: str, key_filename: str,
                        run_id: str, model_s3: str, creds: dict, audio: bytes,
                        filename: str, cfg: dict, gpu: Optional[str] = None) -> tuple[str, Optional[str]]:
    """SSH to the run's VM, ship the transcribe script + the audio clip, run it
    against the finetuned model (downloaded from S3 there), and return the text.
    Blocking — call via to_thread."""
    import tempfile

    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        ext = os.path.splitext(filename)[1] or ".wav"
        remote_audio = f"/tmp/sgpu_tryit_audio{ext}"
        _ssh_put_bytes(cli, audio, remote_audio)
        # Persistent server loaded? route to it (no per-request model load).
        _paths = _playground_paths(cfg, run_id, "asr")
        if _persistent_status(cli, _paths).get("ready"):
            resp, tail = _persistent_request(cli, _paths, {"audio_path": remote_audio})
            if resp.get("error"):
                raise RuntimeError(resp["error"])
            return resp.get("text", ""), resp.get("device"), (tail or ["served by the persistent worker"])
        base = _trainer_script_path().parent  # gateway/gateway/training/
        _ssh_put(cli, str(base / "transcribe.py"), "/tmp/sgpu_transcribe.py")
        work_dir = (cfg.get("work_dir") or "/share").rstrip("/")
        tconf = {
            "model_s3": model_s3,
            "region": creds.get("region"), "endpoint": creds.get("endpoint"),
            "access_key": creds.get("access_key"), "secret_key": creds.get("secret_key"),
            "audio_path": remote_audio,
            "model_dir": f"{work_dir}/sgpu-tryit/{run_id}",
            "language": cfg.get("language") or None,
            "task": cfg.get("task") or "transcribe",
            "gpu": gpu or "auto",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps(tconf))
            local_cfg = f.name
        try:
            _ssh_put(cli, local_cfg, "/tmp/sgpu_transcribe.json")
        finally:
            try:
                os.unlink(local_cfg)
            except OSError:
                pass
        user_env = _render_env_exports(cfg.get("env_vars") or {})
        out: dict = {"text": None, "raw": None, "device": None, "error": None, "lines": []}

        def on_line(line: str) -> None:
            out["lines"].append(line)
            j = line.find("@@TEXT ")
            if j >= 0:
                try:
                    obj = json.loads(line[j + len("@@TEXT "):])
                except Exception:  # noqa: BLE001
                    return
                if obj.get("error"):
                    out["error"] = obj["error"]
                else:
                    out["text"] = obj.get("text", "")
                    out["raw"] = obj.get("raw")
                    out["device"] = obj.get("device")

        # Run with the ASR trainer venv (torch/transformers/librosa/boto3); the
        # box's bare `python` lacks them → "Could not import module 'pipeline'".
        # Fall back to system python3/python if the venv is somehow absent.
        # Reuse the STT run's training uv venv (config_json); never machine python.
        venv = (cfg.get("venv_path") or "/share/autotrain-whisper").rstrip("/")
        py = f"{venv}/bin/python"
        cmd = (f'{user_env}PY="{py}"; '
               f'if [ ! -x "$PY" ]; then echo "venv python not found at $PY — train this run first (it creates the uv venv)"; exit 1; fi; '
               f'"$PY" -u /tmp/sgpu_transcribe.py --config /tmp/sgpu_transcribe.json')
        rc = _ssh_run_stream(cli, cmd, on_line)
        if out["error"]:
            raise RuntimeError(out["error"])
        if out["text"] is None:
            tail = "\n".join(out["lines"][-15:])
            raise RuntimeError(f"transcription produced no output (rc={rc}):\n{tail}")
        return out["text"], out.get("device"), out.get("raw"), out["lines"][-120:]
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


def _gpu_query_sync(run_id: str, host: str, port: int, user: str,
                    key_filename: str, gpu_csv: Optional[str]) -> list[dict]:
    """`nvidia-smi` over a cached SSH connection → per-GPU util + memory. Only
    the run's GPUs when `gpu_csv` is set. Blocking — call via to_thread."""
    import paramiko

    cli = _GPU_SSH.get(run_id)
    tr = cli.get_transport() if cli is not None else None
    if cli is None or tr is None or not tr.is_active():
        cli = paramiko.SSHClient()
        cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        cli.connect(hostname=host, port=port, username=user, key_filename=key_filename,
                    timeout=8, banner_timeout=10, auth_timeout=10)
        # This client is cached in _GPU_SSH and reused across polling gaps, so keep the
        # connection warm (same keepalive rationale as _ssh_connect) — else an idle-dropped
        # TCP wedges the next poll's recv() or forces a needless reconnect.
        try:
            tr2 = cli.get_transport()
            if tr2 is not None:
                tr2.set_keepalive(30)
        except Exception:  # noqa: BLE001
            pass
        _GPU_SSH[run_id] = cli
    q = ("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,"
         "temperature.gpu,name --format=csv,noheader,nounits")
    if gpu_csv:
        q += f" -i {gpu_csv}"
    try:
        chan = cli.get_transport().open_session()
        chan.exec_command(q)
        out = b""
        while True:
            d = chan.recv(4096)
            if not d:
                break
            out += d
        chan.recv_exit_status()
    except Exception:
        try:
            cli.close()
        finally:
            _GPU_SSH.pop(run_id, None)
        raise
    gpus: list[dict] = []
    for line in out.decode("utf-8", "replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append({
                "index": int(parts[0]), "util": float(parts[1]),
                "mem_used": float(parts[2]), "mem_total": float(parts[3]),
                "temp": float(parts[4]), "name": parts[5],
            })
        except ValueError:
            continue
    return gpus


@router.get("/{run_id}/gpu")
async def training_gpu(
    run_id: str,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Live per-GPU utilisation for the run's GPUs only. Polled by the UI while
    the run is active; returns an empty list once it isn't running."""
    row = await _owned(run_id, user, session)
    if row.status != "running":
        _GPU_SSH.pop(run_id, None)
        return {"status": row.status, "gpus": []}
    if (row.config_json or {}).get("nodes"):
        # Multi-node sweep parent: no single SSH target of its own — poll each
        # currently-running child trial's own GPUs and merge (same trial-offset
        # scheme as the persisted gpu_samples in run_multi_node_sweep, so the live
        # view and the historical chart use consistent GPU indices).
        trials = (row.result_json or {}).get("trials") or []
        gpus: list[dict] = []
        for t in trials:
            if t.get("status") != "running" or not t.get("run_id"):
                continue
            child = await session.get(TrainingRun, t["run_id"])
            if child is None:
                continue
            ssh = await _resolve_run_ssh(child)
            if ssh is None:
                continue
            try:
                cgpus = await asyncio.to_thread(_gpu_query_sync, t["run_id"], *ssh, _run_gpu_id_csv(child))
            except Exception:  # noqa: BLE001
                continue
            trial_i = t.get("trial") or 0
            gpus.extend({**g, "index": (g.get("index") or 0) + trial_i * 100} for g in cgpus)
        return {"status": row.status, "gpus": gpus}
    ssh = await _resolve_run_ssh(row)
    if ssh is None:
        return {"status": row.status, "gpus": [], "error": "ssh coords unavailable"}
    try:
        gpus = await asyncio.to_thread(_gpu_query_sync, run_id, *ssh, _run_gpu_id_csv(row))
    except Exception as e:  # noqa: BLE001
        return {"status": row.status, "gpus": [], "error": str(e)[:200]}
    return {"status": row.status, "gpus": gpus}


class TranscribeResponse(BaseModel):
    text: str
    # Raw decode with special tokens KEPT (<|startoftranscript|><|lang|><|transcribe|>
    # <|notimestamps|> … <|endoftext|>) — lets the playground verify the finetuned
    # model emits the Whisper prompt + EOS. None if the raw decode was skipped.
    raw: Optional[str] = None
    device: Optional[str] = None
    logs: list[str] = []          # VM-side progress (model download, inference) for the playground


@router.post("/{run_id}/transcribe", response_model=TranscribeResponse)
async def transcribe_with_run(
    run_id: str,
    request: Request,
    filename: str = Query("audio.wav", description="original file name (for the audio extension)"),
    gpu: Optional[str] = Query(None, description="GPU index to run on (e.g. '6'), 'cpu', or 'auto'"),
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Try-it playground: transcribe one uploaded clip with this run's finetuned
    model. Runs on the run's VM over SSH (the control plane has no GPU/ML deps),
    so it needs a finished ASR run on a kind=vm provider — cloud-pod runs are
    gone after training. The raw audio bytes are the request body."""
    row = await _owned(run_id, user, session)
    if (row.task_type or "asr") != "asr":
        raise HTTPException(status_code=400, detail="try-it is for ASR (Whisper) runs")
    if row.status != "done":
        raise HTTPException(status_code=400, detail="the run must finish training first")
    model_s3 = ((row.result_json or {}).get("artifact") or {}).get("s3_uri")
    if not model_s3:
        raise HTTPException(status_code=400, detail="no trained model artifact for this run")
    prov = await session.get(Provider, row.provider_id) if row.provider_id else None
    is_cloud = prov is None or prov.kind != "vm"
    if is_cloud:
        # Cloud run — the training pod is gone. Transcribe runs on the on-demand
        # try-it pod (Load it first via …/playground/start). Bump its idle window.
        ti = _tryit_state(row)
        if ti.get("phase") != "ready":
            raise HTTPException(
                status_code=409,
                detail=("load the try-it pod first (it spins up a temporary GPU pod for this "
                        "cloud run) — POST …/playground/start, then transcribe once it's ready."),
            )
        await _tryit_save(run_id, expires_at=_tryit_exp())
    # Validate the GPU choice against the run's pinned GPUs (if any).
    sel = (gpu or "").strip().lower()
    if sel and sel not in ("cpu", "auto"):
        if not sel.isdigit():
            raise HTTPException(status_code=400, detail="gpu must be a GPU index, 'cpu', or 'auto'")
        allowed = [x.strip() for x in (row.visible_devices or "").split(",") if x.strip()]
        if allowed and sel not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"GPU {sel} not in this run's GPUs ({', '.join(allowed)})",
            )
    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="empty audio upload")
    if len(audio) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="audio too large (max 25 MB) — try-it is for short clips")
    storage = await session.get(Storage, row.storage_id) if row.storage_id else None
    creds = _s3_creds_from_storage(storage)
    ssh = await _resolve_run_ssh(row)
    if ssh is None:
        raise HTTPException(status_code=400, detail="can't reach the run's VM (SSH coords unavailable)")
    try:
        text, device, raw, logs = await asyncio.to_thread(
            _run_transcribe_ssh, *ssh, run_id, model_s3, creds, audio,
            filename or "audio.wav", dict(row.config_json or {}), sel or "auto",
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"transcription failed: {e}")
    return TranscribeResponse(text=text, device=device, raw=raw, logs=logs)


def _run_synthesize_ssh(host: str, port: int, user: str, key_filename: str,
                        run_id: str, model_s3: str, creds: dict, text: str,
                        speaker: str, cfg: dict, gpu: Optional[str] = None) -> tuple[bytes, int, Optional[str]]:
    """SSH to the run's VM, ship tts_infer.py, synthesize `text` with the finetuned
    TTS model (downloaded from S3 there), and return (wav_bytes, sample_rate,
    device). The TTS twin of _run_transcribe_ssh. Blocking — call via to_thread."""
    import tempfile

    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        # Persistent server loaded? route to it (no per-request model load).
        _paths = _playground_paths(cfg, run_id, "tts")
        if _persistent_status(cli, _paths).get("ready"):
            req = {"text": text, "speaker": speaker or ""}
            resp, tail = _persistent_request(cli, _paths, req)
            if resp.get("error"):
                raise RuntimeError(resp["error"])
            if not resp.get("wav_b64"):
                raise RuntimeError("persistent worker returned no audio")
            return (resp["wav_b64"], int(resp.get("sample_rate") or 24000), resp.get("device"),
                    (tail or ["served by the persistent worker"]), resp.get("prompt"), resp.get("gen_text"))
        base = _trainer_script_path().parent  # gateway/gateway/training/
        _ssh_put(cli, str(base / "tts" / "tts_infer.py"), "/tmp/sgpu_tts_infer.py")
        work_dir = (cfg.get("work_dir") or "/share").rstrip("/")
        tconf = {
            "model_s3": model_s3,
            "region": creds.get("region"), "endpoint": creds.get("endpoint"),
            "access_key": creds.get("access_key"), "secret_key": creds.get("secret_key"),
            "model_dir": f"{work_dir}/sgpu-tts-tryit/{run_id}",
            "text": text, "speaker": speaker or "", "gpu": gpu or "auto",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps(tconf))
            local_cfg = f.name
        try:
            _ssh_put(cli, local_cfg, "/tmp/sgpu_tts_infer.json")
        finally:
            try:
                os.unlink(local_cfg)
            except OSError:
                pass
        user_env = _render_env_exports(cfg.get("env_vars") or {})
        # Reuse the TTS run's training uv venv (config_json; torch/transformers/
        # neucodec/soundfile/boto3); never machine python.
        venv = (cfg.get("venv_path") or "/share/autotrain-tts").rstrip("/")
        py = f"{venv}/bin/python"
        out: dict = {"wav": None, "sr": None, "device": None, "error": None, "lines": [],
                     "prompt": None, "gen_text": None}

        def on_line(line: str) -> None:
            j = line.find("@@AUDIO ")
            if j < 0:
                if len(out["lines"]) < 400:  # keep a tail for errors; skip the huge b64 line
                    out["lines"].append(line)
                return
            try:
                obj = json.loads(line[j + len("@@AUDIO "):])
            except Exception:  # noqa: BLE001
                return
            out["prompt"] = obj.get("prompt"); out["gen_text"] = obj.get("gen_text")
            if obj.get("error"):
                out["error"] = obj["error"]
            else:
                out["wav"] = obj.get("wav_b64"); out["sr"] = obj.get("sample_rate"); out["device"] = obj.get("device")

        cmd = (f'{user_env}PY="{py}"; '
               f'if [ ! -x "$PY" ]; then echo "venv python not found at $PY — train this run first (it creates the uv venv)"; exit 1; fi; '
               f'"$PY" -u /tmp/sgpu_tts_infer.py --config /tmp/sgpu_tts_infer.json')
        rc = _ssh_run_stream(cli, cmd, on_line)
        if out["error"]:
            raise RuntimeError(out["error"])
        if not out["wav"]:
            tail = "\n".join(out["lines"][-15:])
            raise RuntimeError(f"synthesis produced no audio (rc={rc}):\n{tail}")
        return (out["wav"], int(out["sr"] or 24000), out.get("device"), out["lines"][-120:],
                out.get("prompt"), out.get("gen_text"))  # wav is base64
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


def _run_tts_label_export_ssh(host: str, port: int, user: str, key_filename: str,
                              run_id: str, model_s3: str, creds: dict, packed_uri: str,
                              split_subdir: str, is_random: bool, n_samples: int,
                              speaker: str, upload_bucket: str, upload_prefix: str,
                              cfg: dict, gpu: str = "auto", line_sink=None) -> dict:
    """SSH to the run's VM, ship the whole tts/ dir, synthesize n_samples clips from
    the finetuned model + upload them to S3, and return the parsed @@LABEL manifest
    {bucket, items:[{key,text}], count}. Blocking — call via to_thread. The TTS
    batch twin of _run_synthesize_ssh. `line_sink(str)` (if given) receives every
    VM-side log line so the caller can stream synthesis progress to the run's logs
    (otherwise a multi-minute synth/upload is silent and the run looks stuck)."""
    import tempfile

    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        base = _trainer_script_path().parent  # gateway/gateway/training/
        # Ship the whole tts/ dir (chinidataset + tts_eval + tts_infer + the export
        # script import as siblings) — the export reads packed shards + synthesizes.
        _ssh_put_dir_tar(cli, str(base / "tts"), "/tmp/sgpu_tts_label")
        work_dir = (cfg.get("work_dir") or "/share").rstrip("/")
        tconf = {
            "model_s3": model_s3,
            "region": creds.get("region"), "endpoint": creds.get("endpoint"),
            "access_key": creds.get("access_key"), "secret_key": creds.get("secret_key"),
            "model_dir": f"{work_dir}/sgpu-tts-label/{run_id}/model",
            "packed_dir": f"{work_dir}/sgpu-tts-label/{run_id}/packed",
            "packed_uri": packed_uri, "split_subdir": split_subdir or "",
            "random": bool(is_random), "n_samples": int(n_samples), "seed": 42,
            "speaker": speaker or "", "gpu": gpu or "auto", "max_new_tokens": 1024,
            "upload_bucket": upload_bucket, "upload_prefix": upload_prefix,
            # Balance the synthesized clips round-robin across these speaker names
            # (e.g. 2 speakers + 32 samples → 16 each). Empty → original packed voices.
            "label_speakers": [str(s).strip() for s in (cfg.get("label_speakers") or []) if str(s).strip()],
            # Prefix each task's transcription with the speaker name ("spk: text").
            "label_speaker_prefix": bool(cfg.get("label_speaker_prefix")),
            # Drop text samples containing any of these phrases (case-insensitive).
            "reject_keywords": [str(k).strip() for k in (cfg.get("label_reject_keywords") or []) if str(k).strip()],
            # true → N clips from EACH speaker's own utterances (gateway then makes
            # one project per speaker); false → round-robin re-voicing into one project.
            "per_speaker": bool(cfg.get("label_per_speaker")),
            # NeuCodec decoder variant: "neucodec" (upstream 24 kHz) | "neucodec-44k".
            "tts_codec": (cfg.get("tts_codec") or "neucodec"),
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps(tconf))
            local_cfg = f.name
        try:
            _ssh_put(cli, local_cfg, "/tmp/sgpu_tts_label_cfg.json")
        finally:
            try:
                os.unlink(local_cfg)
            except OSError:
                pass
        user_env = _render_env_exports(cfg.get("env_vars") or {})
        venv = (cfg.get("venv_path") or "/share/autotrain-tts").rstrip("/")
        py = f"{venv}/bin/python"
        out: dict = {"manifest": None, "error": None, "lines": []}

        def on_line(line: str) -> None:
            j = line.find("@@LABEL ")
            if j < 0:
                if line_sink is not None:
                    try:
                        line_sink(line)  # stream synth/upload progress to the run's logs
                    except Exception:  # noqa: BLE001
                        pass
                if len(out["lines"]) < 400:  # keep a tail for errors
                    out["lines"].append(line)
                return
            try:
                obj = json.loads(line[j + len("@@LABEL "):])
            except Exception:  # noqa: BLE001
                return
            if obj.get("error"):
                out["error"] = obj["error"]
            else:
                out["manifest"] = obj

        cmd = (f'{user_env}PY="{py}"; '
               f'if [ ! -x "$PY" ]; then echo "venv python not found at $PY — train this run first"; exit 1; fi; '
               f'"$PY" -u /tmp/sgpu_tts_label/tts_label_export.py --config /tmp/sgpu_tts_label_cfg.json')
        rc = _ssh_run_stream(cli, cmd, on_line)
        if out["error"]:
            raise RuntimeError(out["error"])
        if not out["manifest"]:
            tail = "\n".join(out["lines"][-15:])
            raise RuntimeError(f"label export produced no manifest (rc={rc}):\n{tail}")
        return out["manifest"]
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


def _run_llm_label_export_ssh(host: str, port: int, user: str, key_filename: str,
                              run_id: str, model_s3: str, cfg: dict, s3_creds: dict,
                              eval_rows: list, line_sink=None) -> dict:
    """SSH to the run's VM, ship the llm/ dir, generate responses for eval_rows
    from the finetuned LLM, and return the parsed @@LABEL manifest {items, count}.
    Blocking — call via to_thread. `line_sink(str)` receives every VM-side log line."""
    import tempfile

    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        base = _trainer_script_path().parent  # gateway/gateway/training/
        _ssh_put_dir_tar(cli, str(base / "llm"), "/tmp/sgpu_llm_label")
        work_dir = (cfg.get("work_dir") or "/share").rstrip("/")
        arch = _llm_arch(cfg.get("base_model"))
        tconf = {
            "model_s3": model_s3,
            "base_model": cfg.get("base_model") or "",
            "arch": arch,
            "eval_rows": eval_rows,
            "n_samples": int(cfg.get("llm_label_samples") or 0) or len(eval_rows),
            "max_new_tokens": int(cfg.get("llm_label_max_new_tokens") or 512),
            "lora_r": cfg.get("lora_r") or 16,
            "lora_alpha": cfg.get("lora_alpha") or 32,
            "run_id": run_id,
            "work_dir": work_dir,
            "hf_home": f"{work_dir}/huggingface",
            "s3_creds": {
                "bucket": s3_creds.get("bucket"), "region": s3_creds.get("region"),
                "endpoint": s3_creds.get("endpoint"), "access_key": s3_creds.get("access_key"),
                "secret_key": s3_creds.get("secret_key"),
            },
            # vLLM-offline generation (fast path; transformers is the fallback): merge the
            # LoRA → serve the merged model with vLLM in its own venv, batched.
            "use_vllm": True,
            "llm_dir": "/tmp/sgpu_llm_label",       # where the llm/ dir ships (below)
            "merged_dir": f"{work_dir}/sgpu-llm-label/{run_id}/merged",
            "vllm_venv": "/share/autotrain-llm-vllm",   # shared with the try-it playground
            "vllm_version": (cfg.get("label_vllm_version") or "0.23.0"),
            "merge_dtype": "fp16",
            "visible_devices": (cfg.get("label_visible_devices") or cfg.get("visible_devices") or ""),
            "gpu_mem_util": 0.85,
            "max_model_len": int(cfg.get("label_max_model_len") or 32768),
        }
        # Stream over stdin (no argv-length limit): the cfg embeds the eval rows,
        # so it scales with row count and easily exceeds _ssh_put's ~128 KB
        # MAX_ARG_STRLEN base64-argv cap (→ "/bin/bash: Argument list too long").
        _ssh_put_bytes(cli, json.dumps(tconf).encode("utf-8"), "/tmp/sgpu_llm_label_cfg.json")
        user_env = _render_env_exports(cfg.get("env_vars") or {})
        # Use the same arch venv that training used — it already has torch + transformers.
        venv_path = (cfg.get("venv_path") or _llm_venv(arch)).rstrip("/")
        py = f"{venv_path}/bin/python"
        out: dict = {"manifest": None, "error": None, "lines": []}

        def on_line(line: str) -> None:
            j = line.find("@@LABEL ")
            if j < 0:
                if line_sink is not None:
                    try:
                        line_sink(line)
                    except Exception:  # noqa: BLE001
                        pass
                if len(out["lines"]) < 400:
                    out["lines"].append(line)
                return
            try:
                obj = json.loads(line[j + len("@@LABEL "):])
            except Exception:  # noqa: BLE001
                return
            if obj.get("error"):
                out["error"] = obj["error"]
            else:
                out["manifest"] = obj

        cmd = (f'{user_env}PY="{py}"; '
               f'if [ ! -x "$PY" ]; then echo "venv python not found at $PY — train this run first"; exit 1; fi; '
               f'"$PY" -u /tmp/sgpu_llm_label/llm_label_export.py --config /tmp/sgpu_llm_label_cfg.json')
        rc = _ssh_run_stream(cli, cmd, on_line)
        if out["error"]:
            raise RuntimeError(out["error"])
        if not out["manifest"]:
            tail = "\n".join(out["lines"][-15:])
            raise RuntimeError(f"llm label export produced no manifest (rc={rc}):\n{tail}")
        return out["manifest"]
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


def _run_hf_export_local(run_id: str, model_s3: str, creds: dict, repo: str,
                         token: str, private: bool, hf_endpoint: Optional[str] = None,
                         line_sink=None) -> dict:
    """Run the SAME hf_export.py on the GATEWAY (not the VM) as a subprocess in the
    gateway venv, streaming its lines to `line_sink`. Used when the target HF storage
    has a custom endpoint (a self-hosted mirror): the gateway's modern huggingface_hub
    handles a path-prefixed endpoint (`…/hf`) that the VM's older client mis-parses,
    and the gateway reaches the mirror directly (no VM tunnel needed). Blocking — call
    via to_thread. The model is fetched from S3 to a local temp dir."""
    import subprocess
    import sys as _sys
    import tempfile

    script = str(_trainer_script_path().parent / "hf_export.py")
    model_dir = os.path.join(tempfile.gettempdir(), "sgpu-hf-export", run_id, "model")
    tconf = {
        "model_s3": model_s3,
        "region": creds.get("region"), "endpoint": creds.get("endpoint"),
        "access_key": creds.get("access_key"), "secret_key": creds.get("secret_key"),
        "model_dir": model_dir,
        "repo": repo, "token": token, "private": bool(private),
        "hf_endpoint": hf_endpoint or None,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(tconf, f)
        cfg_path = f.name
    out: dict = {"result": None, "error": None, "lines": []}
    rc = -1
    try:
        # stderr→stdout so HF's upload progress bars stream alongside the script's lines.
        proc = subprocess.Popen(
            [_sys.executable, "-u", script, "--config", cfg_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            j = line.find("@@HF ")
            if j < 0:
                if line_sink is not None:
                    try:
                        line_sink(line)
                    except Exception:  # noqa: BLE001
                        pass
                if len(out["lines"]) < 400:
                    out["lines"].append(line)
                continue
            try:
                obj = json.loads(line[j + len("@@HF "):])
            except Exception:  # noqa: BLE001
                continue
            if obj.get("error"):
                out["error"] = obj["error"]
            else:
                out["result"] = obj
        rc = proc.wait()
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass
    if out["error"]:
        raise RuntimeError(out["error"])
    if not out["result"]:
        tail = "\n".join(out["lines"][-15:])
        raise RuntimeError(f"HF export produced no result (rc={rc}):\n{tail}")
    return out["result"]


def _run_hf_export_ssh(host: str, port: int, user: str, key_filename: str,
                       run_id: str, model_s3: str, creds: dict, repo: str,
                       token: str, private: bool, cfg: dict,
                       hf_endpoint: Optional[str] = None, line_sink=None) -> dict:
    """SSH to the run's VM, ship hf_export.py, download the model from S3 + push it
    to a Hugging Face repo, and return the parsed @@HF {repo, url}. Blocking — call
    via to_thread. `line_sink(str)` (if given) gets every VM-side line so the caller
    can stream the multi-minute download/upload to the run's logs (else it looks
    stuck)."""
    import tempfile

    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        base = _trainer_script_path().parent  # gateway/gateway/training/
        remote_py = f"/tmp/sgpu_hf_export_{run_id}.py"
        remote_cfg = f"/tmp/sgpu_hf_export_{run_id}.json"
        _ssh_put(cli, str(base / "hf_export.py"), remote_py)
        work_dir = (cfg.get("work_dir") or "/share").rstrip("/")
        tconf = {
            "model_s3": model_s3,
            "region": creds.get("region"), "endpoint": creds.get("endpoint"),
            "access_key": creds.get("access_key"), "secret_key": creds.get("secret_key"),
            "model_dir": f"{work_dir}/sgpu-hf-export/{run_id}/model",
            "repo": repo, "token": token, "private": bool(private),
            "hf_endpoint": hf_endpoint or None,  # custom Hub (HF_ENDPOINT); None → huggingface.co
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps(tconf))
            local_cfg = f.name
        try:
            _ssh_put(cli, local_cfg, remote_cfg)
        finally:
            try:
                os.unlink(local_cfg)
            except OSError:
                pass
        user_env = _render_env_exports(cfg.get("env_vars") or {})
        # hf_export.py is arch-agnostic (S3 download → HF push), but it still needs a
        # venv on the box with huggingface_hub + boto3 — namely the one this run TRAINED
        # in. venv_path isn't persisted to config_json, so re-derive it by task/arch
        # (OmniVoice → autotrain-omnivoice, ASR → autotrain-whisper); a hardcoded
        # autotrain-tts default made OmniVoice/ASR pushes fail with "venv not found".
        venv = (cfg.get("venv_path")
                or _train_venv_default(cfg.get("task_type"), cfg.get("base_model"))).rstrip("/")
        py = f"{venv}/bin/python"
        out: dict = {"result": None, "error": None, "lines": []}

        def on_line(line: str) -> None:
            j = line.find("@@HF ")
            if j < 0:
                if line_sink is not None:
                    try:
                        line_sink(line)  # stream download/upload progress to the run's logs
                    except Exception:  # noqa: BLE001
                        pass
                if len(out["lines"]) < 400:
                    out["lines"].append(line)
                return
            try:
                obj = json.loads(line[j + len("@@HF "):])
            except Exception:  # noqa: BLE001
                return
            if obj.get("error"):
                out["error"] = obj["error"]
            else:
                out["result"] = obj

        # 2>&1 merges the HF upload's progress bars (stderr) into the stream so they
        # show in the logs too, alongside the script's own download/upload lines.
        cmd = (f'{user_env}PY="{py}"; '
               f'if [ ! -x "$PY" ]; then echo "venv python not found at $PY — train this run first"; exit 1; fi; '
               f'"$PY" -u {remote_py} --config {remote_cfg} 2>&1; rm -f {remote_py} {remote_cfg}')
        rc = _ssh_run_stream(cli, cmd, on_line)
        if out["error"]:
            raise RuntimeError(out["error"])
        if not out["result"]:
            tail = "\n".join(out["lines"][-15:])
            raise RuntimeError(f"HF export produced no result (rc={rc}):\n{tail}")
        return out["result"]
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


def _run_hf_merge_export_ssh(host: str, port: int, user: str, key_filename: str,
                             run_id: str, model_s3: str, creds: dict, repo: str,
                             token: str, private: bool, cfg: dict, base_model: str,
                             arch: str, merge_dtype: str, venv_path: Optional[str],
                             visible_devices: Optional[str], base_token: Optional[str] = None,
                             hf_endpoint: Optional[str] = None, line_sink=None) -> dict:
    """SSH to a box (the run's VM or a fresh pod), ship hf_export.py + the llm/ dir, then
    download the LoRA checkpoint from S3 → MERGE it into the base model on GPU (per-arch,
    via the shipped merge scripts) → upload the merged safetensors to a Hugging Face repo,
    and return the parsed @@HF {repo, url}. Blocking — call via to_thread. `line_sink(str)`
    streams every box-side line (download / merge / upload) to the run's logs. The whole
    script runs under the arch venv python (transformers + peft + boto3 + huggingface_hub).
    `token` pushes to the destination repo; `base_token` (if given) downloads a gated base
    model for the merge (else the push token is reused). huggingface.co only (v1)."""
    cli = _ssh_connect(host, int(port), user, key_filename)
    try:
        import secrets as _secrets
        base = _trainer_script_path().parent  # gateway/gateway/training/
        # Per-REQUEST scratch (not per-run): re-clicking "Export to HF" supersedes the
        # in-flight task, but its box-side merge keeps running in a thread and its trailing
        # `rm -rf` would otherwise delete the NEW request's shipped llm/ dir mid-merge
        # (→ "No module named 'llm_playground'"). A unique tag isolates concurrent attempts.
        # `sgpu_hf_export_<run_id>` stays a prefix so cancel_huggingface_export's pkill matches.
        tag = f"{run_id}-{_secrets.token_hex(4)}"
        remote_py = f"/tmp/sgpu_hf_export_{tag}.py"
        remote_cfg = f"/tmp/sgpu_hf_export_{tag}.json"
        code_dir = f"/tmp/sgpu_llm_hfexport_{tag}"
        _ssh_put(cli, str(base / "hf_export.py"), remote_py)
        _ssh_put_dir_tar(cli, str(base / "llm"), code_dir)
        work_dir = (cfg.get("work_dir") or "/share").rstrip("/")
        scratch = f"{work_dir}/sgpu-hf-export/{tag}"
        venv = (venv_path or _llm_venv(arch)).rstrip("/")
        py = f"{venv}/bin/python"
        tconf = {
            "model_s3": model_s3,
            "region": creds.get("region"), "endpoint": creds.get("endpoint"),
            "access_key": creds.get("access_key"), "secret_key": creds.get("secret_key"),
            "model_dir": f"{scratch}/merged",  # merged model → uploaded
            "ckpt_dir": f"{scratch}/ckpt",     # raw LoRA download
            "repo": repo, "token": token, "private": bool(private),
            "base_hf_token": base_token or token,    # gated base-model download token (merge)
            # None → huggingface.co; a loopback URL (reached via a reverse tunnel) or an
            # external URL → push the merged model to that self-hosted HF mirror.
            "hf_endpoint": hf_endpoint or None,
            "merge": True,
            "base_model": base_model, "arch": arch,
            "merge_dtype": merge_dtype or "fp16",
            "llm_dir": code_dir, "train_py": py,     # merge subprocess runs from the arch venv
            "visible_devices": visible_devices or "",
        }
        # Stream the cfg over stdin (token stays out of argv; consistent with the label path).
        _ssh_put_bytes(cli, json.dumps(tconf).encode("utf-8"), remote_cfg)
        user_env = _render_env_exports(cfg.get("env_vars") or {})
        out: dict = {"result": None, "error": None, "lines": []}

        def on_line(line: str) -> None:
            j = line.find("@@HF ")
            if j < 0:
                if line_sink is not None:
                    try:
                        line_sink(line)
                    except Exception:  # noqa: BLE001
                        pass
                if len(out["lines"]) < 400:
                    out["lines"].append(line)
                return
            try:
                obj = json.loads(line[j + len("@@HF "):])
            except Exception:  # noqa: BLE001
                return
            if obj.get("error"):
                out["error"] = obj["error"]
            else:
                out["result"] = obj

        cmd = (f'{user_env}PY="{py}"; '
               f'if [ ! -x "$PY" ]; then echo "arch venv python not found at $PY — build the venv first"; exit 1; fi; '
               f'"$PY" -u {remote_py} --config {remote_cfg} 2>&1; '
               # Clean up only THIS request's tagged scratch (incl. the merged model, so a VM
               # isn't left holding tens of GB) — never a sibling request's dir.
               f'rm -rf {remote_py} {remote_cfg} {code_dir} {scratch}')
        rc = _ssh_run_stream(cli, cmd, on_line)
        if out["error"]:
            raise RuntimeError(out["error"])
        if not out["result"]:
            tail = "\n".join(out["lines"][-15:])
            raise RuntimeError(f"HF merged export produced no result (rc={rc}):\n{tail}")
        return out["result"]
    finally:
        try:
            cli.close()
        except Exception:  # noqa: BLE001
            pass


class SynthesizeResponse(BaseModel):
    audio_b64: str          # base64-encoded WAV (PCM_16) the browser can play
    sample_rate: int
    device: Optional[str] = None
    logs: list[str] = []    # VM-side progress (model download, generation) for the playground
    prompt: Optional[str] = None    # the exact prompt fed to the model
    gen_text: Optional[str] = None  # the model's raw generation (speech tokens) before NeuCodec


@router.post("/{run_id}/synthesize", response_model=SynthesizeResponse)
async def synthesize_with_run(
    run_id: str,
    text: str = Query(..., description="text to synthesize"),
    speaker: str = Query("", description="optional speaker name (matches how the data was packed)"),
    gpu: Optional[str] = Query(None, description="GPU index (e.g. '6'), 'cpu', or 'auto'"),
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Try-it playground (TTS): synthesize speech for `text` with this run's
    finetuned TTS model. Runs on the run's VM over SSH (the control plane has no
    GPU/ML deps) → needs a finished TTS run on a kind=vm provider. Returns a WAV."""
    row = await _owned(run_id, user, session)
    if (row.task_type or "asr") != "tts":
        raise HTTPException(status_code=400, detail="synthesize is for TTS runs")
    if row.status != "done":
        raise HTTPException(status_code=400, detail="the run must finish training first")
    model_s3 = ((row.result_json or {}).get("artifact") or {}).get("s3_uri")
    if not model_s3:
        raise HTTPException(status_code=400, detail="no trained model artifact for this run")
    prov = await session.get(Provider, row.provider_id) if row.provider_id else None
    if prov is None or prov.kind != "vm":
        raise HTTPException(status_code=400, detail=("try-it runs on a VM provider; this run used a "
                            "cloud pod (gone after training). Push to HF and try there, or re-run on a VM."))
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="text too long (max 2000 chars) — try-it is for short clips")
    sel = (gpu or "").strip().lower()
    if sel and sel not in ("cpu", "auto"):
        if not sel.isdigit():
            raise HTTPException(status_code=400, detail="gpu must be a GPU index, 'cpu', or 'auto'")
        allowed = [x.strip() for x in (row.visible_devices or "").split(",") if x.strip()]
        if allowed and sel not in allowed:
            raise HTTPException(status_code=400, detail=f"GPU {sel} not in this run's GPUs ({', '.join(allowed)})")
    storage = await session.get(Storage, row.storage_id) if row.storage_id else None
    creds = _s3_creds_from_storage(storage)
    ssh = await _resolve_run_ssh(row)
    if ssh is None:
        raise HTTPException(status_code=400, detail="can't reach the run's VM (SSH coords unavailable)")
    try:
        wav_b64, sr, device, logs, prompt, gen_text = await asyncio.to_thread(
            _run_synthesize_ssh, *ssh, run_id, model_s3, creds, text, speaker,
            dict(row.config_json or {}), sel or "auto",
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"synthesis failed: {e}")
    return SynthesizeResponse(audio_b64=wav_b64, sample_rate=sr, device=device, logs=logs,
                              prompt=prompt, gen_text=gen_text)


class PlaygroundStatus(BaseModel):
    running: bool = False
    ready: bool = False
    device: Optional[str] = None
    kind: Optional[str] = None
    logs: list[str] = []    # tail of the worker's load/serve log (for live load progress)


async def _playground_validate(run_id: str, user: User, session: AsyncSession):
    """Shared try-it validation: a finished ASR/TTS/LLM run with a model artifact.
    Returns (row, kind, model_s3, creds, cfg) — provider/SSH resolution is the
    caller's (the run's own box vs a chosen VM)."""
    row = await _owned(run_id, user, session)
    kind = (row.task_type or "asr").lower()
    if kind not in ("asr", "tts", "llm"):
        raise HTTPException(status_code=400, detail="try-it is for ASR/TTS/LLM runs")
    if row.status != "done":
        raise HTTPException(status_code=400, detail="the run must finish training first")
    model_s3 = ((row.result_json or {}).get("artifact") or {}).get("s3_uri")
    if not model_s3:
        raise HTTPException(status_code=400, detail="no trained model artifact for this run")
    storage = await session.get(Storage, row.storage_id) if row.storage_id else None
    creds = _s3_creds_from_storage(storage)
    return row, kind, model_s3, creds, dict(row.config_json or {})


async def _playground_ctx(run_id: str, user: User, session: AsyncSession):
    """Persistent try-it on the run's try-it compute — the target chosen at load time
    (a fresh cloud pod, a chosen VM, or the run's own box), resolved by _resolve_tryit_ssh.
    Returns (row, kind, model_s3, creds, ssh, cfg)."""
    row, kind, model_s3, creds, cfg = await _playground_validate(run_id, user, session)
    ssh = await _resolve_tryit_ssh(row)
    if ssh is None:
        raise HTTPException(status_code=400, detail="can't reach the try-it compute — load it via 'Try it' first")
    return row, kind, model_s3, creds, ssh, cfg


async def _playground_ctx_for_target(run_id: str, user: User, session: AsyncSession,
                                     *, vm_provider_id: Optional[str]):
    """Persistent try-it on a CHOSEN VM provider (Run-on → Bare metal), which may
    differ from where the run trained. Validates the provider is the caller's own
    (or admin) and kind=vm. Returns (row, kind, model_s3, creds, ssh, cfg)."""
    row, kind, model_s3, creds, cfg = await _playground_validate(run_id, user, session)
    prov = await session.get(Provider, vm_provider_id) if vm_provider_id else None
    if prov is None or prov.kind != "vm":
        raise HTTPException(status_code=400, detail="pick a registered VM provider to run try-it on")
    if prov.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="that VM provider isn't yours")
    ssh = await _resolve_provider_ssh(prov, run_id)
    if ssh is None:
        raise HTTPException(status_code=400, detail="can't reach the chosen VM (SSH coords / key unavailable)")
    return row, kind, model_s3, creds, ssh, cfg


@router.post("/{run_id}/playground/start", response_model=PlaygroundStatus)
async def playground_start(
    run_id: str,
    target: Optional[str] = Query(None,
        description="'cloud' (spin up a fresh RunPod pod) or 'vm' (a registered VM provider). "
                    "Omitted → derived from where the run trained (back-compat)."),
    gpu: Optional[str] = Query(None,
        description="VM target: GPU index, 'cpu', or 'auto' (LLM: a comma-list like '6,7'). Ignored for cloud."),
    gpu_type: Optional[str] = Query(None, description="cloud target: GPU type to provision (e.g. 'L40S')."),
    gpu_count: Optional[int] = Query(None, ge=1, le=8, description="cloud target: number of GPUs."),
    cloud_type: Optional[str] = Query(None, description="cloud target: 'SECURE' or 'COMMUNITY'."),
    provider_id: Optional[str] = Query(None, description="cloud: RunPod account; vm: the chosen VM provider."),
    idle_minutes: float = Query(5, ge=0, le=120,
                                description="VM target: auto-unload after this many idle minutes (0 = never)"),
    vllm_args: Optional[str] = Query(None,
        description="LLM only: extra vLLM serve CLI args, verbatim (e.g. '--enable-auto-tool-choice "
                    "--tool-call-parser hermes --max-model-len 32768'). Appended last so they override "
                    "the defaults; the platform-set flags (model/port/served-name/tp) are rejected."),
    vllm_version: Optional[str] = Query(None,
        description="LLM only: vLLM version to install in the serve venv (default 0.23.0), like serverless/new."),
    hf_token: Optional[str] = Query(None,
        description="LLM only: HF token to download the (usually gated) base model for the merge. "
                    "Needed on a fresh cloud pod / a VM without it cached."),
    hf_token_secret: Optional[str] = Query(None,
        description="LLM only: a global-secret key holding the HF base-model token (else platform HF_TOKEN)."),
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Load the run's model into a persistent worker so try-it requests skip the
    per-call model load. The COMPUTE target is chosen here, decoupled from where the
    run trained:
      • cloud → spin up a fresh RunPod pod (gpu_type / gpu_count / cloud_type / account),
        build the venv, keep it warm. ASR/TTS go phase=provisioning→ready; LLM builds
        the arch venv + serves vLLM (provisioning→installing→ready once vLLM is healthy).
      • vm    → load a persistent worker on the chosen (or the run's) VM provider.
    Switching target/spec while one is loaded tears the old down first. Poll
    …/playground/status until ready. LLM serves via vLLM on either target."""
    row0 = await _owned(run_id, user, session)
    prov0 = await session.get(Provider, row0.provider_id) if row0.provider_id else None
    kind0 = (row0.task_type or "asr").lower()
    explicit = bool((target or "").strip())
    eff_target = (target or "").strip().lower()
    if eff_target not in ("cloud", "vm"):
        eff_target = "vm" if (prov0 is not None and prov0.kind == "vm") else "cloud"
    ti = _tryit_state(row0)
    cur_target = ti.get("target")

    # LLM merges a (usually gated) base model on the box → the merge's from_pretrained
    # needs an HF token. Resolve one: explicit paste > referenced global secret > the
    # platform HF_TOKEN secret. Injected as HF_TOKEN into the box job env (below).
    base_hf_tok: Optional[str] = None
    if kind0 == "llm":
        base_hf_tok = (hf_token or "").strip() or None
        if not base_hf_tok and (hf_token_secret or "").strip():
            from .global_env_api import load_global_env
            base_hf_tok = (await load_global_env(session)).get(hf_token_secret.strip())
        if not base_hf_tok:
            base_hf_tok = (await _resolve_global_env()).get("HF_TOKEN")

    if eff_target == "cloud":
        if row0.status != "done":
            raise HTTPException(status_code=400, detail="the run must finish training first")
        if not ((row0.result_json or {}).get("artifact") or {}).get("s3_uri"):
            raise HTTPException(status_code=400, detail="no trained model artifact for this run")
        # Resolve the pod spec. Explicit requests carry the picker's values; the
        # legacy (no-target) cloud-ASR path defaults to the run's training compute.
        acct = (provider_id or "").strip() or (None if explicit else row0.provider_id)
        gtype = (gpu_type or "").strip() or (row0.gpu_type or "NVIDIA L40S")
        gcount = max(1, int(gpu_count or 1))
        ctype = (cloud_type or "").strip().upper()
        if ctype not in ("SECURE", "COMMUNITY"):
            ctype = "SECURE"
        # LLM: a fresh dedicated pod has exactly `gcount` GPUs → serve tensor-parallel
        # across all of them; validate the optional vLLM args like the VM path. The knobs
        # are stashed in try-it state for _tryit_cloud_setup's LLM branch to consume.
        llm_extra: dict = {}
        if kind0 == "llm":
            va = (vllm_args or "").strip()
            if va:
                from .main import _validate_vllm_args, _VLLM_RESERVED_MULTI
                _validate_vllm_args(va, label="vLLM args", reserved=_VLLM_RESERVED_MULTI)
            llm_extra = {
                "gpus": ",".join(str(i) for i in range(gcount)),
                "vllm_args": va,
                "vllm_version": (vllm_version or "").strip() or "0.23.0",
                "base_model": row0.base_model or (row0.config_json or {}).get("base_model") or "",
                "max_model_len": 16384,
            }
        # Idempotent: same cloud spec already provisioning/ready → just report it.
        same_spec = (cur_target == "cloud" and ti.get("provider_id") == acct
                     and ti.get("gpu_type") == gtype and int(ti.get("gpu_count") or 1) == gcount
                     and (ti.get("cloud_type") or "SECURE") == ctype)
        if same_spec and ti.get("phase") in ("provisioning", "installing", "ready"):
            return _tryit_status(row0)
        # Different target/spec (or a prior errored attempt) → tear the old one down first.
        if cur_target:
            await _tryit_teardown(run_id)
        await _tryit_save(run_id, target="cloud", provider_id=acct, kind=kind0,
                          gpu_type=gtype, gpu_count=gcount, cloud_type=ctype,
                          phase="provisioning", message="starting a GPU pod …",
                          expires_at=_tryit_exp(), **llm_extra)
        _tryit_tasks[run_id] = asyncio.create_task(_tryit_cloud_setup(
            run_id, gpu_type=gtype, gpu_count=gcount, cloud_type=ctype, provider_id=acct,
            kind=kind0, base_hf_token=base_hf_tok))
        async with session_factory()() as s:
            row0 = await s.get(TrainingRun, run_id)
        return _tryit_status(row0)

    # ---- vm target: persistent worker on the chosen (or the run's) VM provider ----
    vm_pid = (provider_id or "").strip() or (row0.provider_id if (prov0 and prov0.kind == "vm") else None)
    # Switching away from a cloud pod, or to a different VM, tears the old down first.
    if cur_target == "cloud" or (cur_target == "vm" and ti.get("provider_id") != vm_pid):
        await _tryit_teardown(run_id)
    row, kind, model_s3, creds, ssh, cfg = await _playground_ctx_for_target(
        run_id, user, session, vm_provider_id=vm_pid)
    # Record the target so status / stop / transcribe / synthesize resolve the SAME box.
    await _tryit_save(run_id, target="vm", provider_id=vm_pid, kind=kind)
    if kind == "llm":
        # Serve GPUs: an explicit `gpu` (comma-list like "6,7") overrides the run's
        # training pin; tensor-parallel size = the number of GPUs chosen.
        sel = (gpu or "").strip().lower()
        gpus = sel if (sel and sel not in ("auto", "cpu")) else (row.visible_devices or "").strip()
        gpu_ids = [g.strip() for g in gpus.split(",") if g.strip()]
        if not gpu_ids or not all(g.isdigit() for g in gpu_ids):
            raise HTTPException(status_code=400,
                detail="LLM try-it needs a comma-separated GPU list (e.g. '6,7') — pass ?gpu=6,7 or set the run's visible_devices")
        gpus = ",".join(gpu_ids)
        va = (vllm_args or "").strip()
        if va:
            # Reuse the serverless validator (reject pasted backslashes / bad quotes /
            # the flags the platform sets itself: model/port/served-name/tp/pp/sleep).
            from .main import _validate_vllm_args, _VLLM_RESERVED_MULTI
            _validate_vllm_args(va, label="vLLM args", reserved=_VLLM_RESERVED_MULTI)
        vver = (vllm_version or "").strip() or "0.23.0"
        cfg = _cfg_with_hf_token(cfg, base_hf_tok)  # merge downloads the gated base
        try:
            st = await asyncio.to_thread(_llm_playground_start_ssh, *ssh, run_id, model_s3, creds, cfg,
                                         (row.base_model or cfg.get("base_model") or "google/gemma-4-31B-it"),
                                         gpus, run_id, 16384, va, vver)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"could not start the vLLM server: {e}")
        return PlaygroundStatus(**st)
    sel = (gpu or "").strip().lower()
    if sel and sel not in ("cpu", "auto") and not sel.isdigit():
        raise HTTPException(status_code=400, detail="gpu must be a GPU index, 'cpu', or 'auto'")
    try:
        st = await asyncio.to_thread(_playground_start_ssh, *ssh, run_id, kind, model_s3, creds, cfg,
                                     sel or "auto", int(idle_minutes * 60))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"could not start the worker: {e}")
    return PlaygroundStatus(**st)


@router.get("/{run_id}/playground/status", response_model=PlaygroundStatus)
async def playground_status(
    run_id: str,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    row0 = await _owned(run_id, user, session)
    prov0 = await session.get(Provider, row0.provider_id) if row0.provider_id else None
    ti = _tryit_state(row0)
    # Cloud try-it (any task type, incl. LLM): the setup task owns the phase
    # (provisioning→installing→ready — including the LLM merge + vLLM readiness), so
    # report it without SSHing (the pod may not be up yet). Also covers legacy cloud-ASR
    # runs with no try-it target recorded (never loaded → phase None → not running).
    if ti.get("target") == "cloud" or (
            (row0.task_type or "asr").lower() == "asr" and (prov0 is None or prov0.kind != "vm")):
        return _tryit_status(row0)   # cloud try-it pod phase (provisioning/installing/ready/error)
    row, kind, _m, _c, ssh, cfg = await _playground_ctx(run_id, user, session)
    try:
        st = await asyncio.to_thread(_playground_status_ssh, *ssh, run_id, cfg, kind)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"status check failed: {e}")
    return PlaygroundStatus(**st)


@router.post("/{run_id}/playground/stop", response_model=PlaygroundStatus)
async def playground_stop(
    run_id: str,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Unload the persistent worker and free the GPU (SIGTERM its process group).
    For a cloud try-it run this terminates the on-demand pod entirely."""
    row0 = await _owned(run_id, user, session)
    prov0 = await session.get(Provider, row0.provider_id) if row0.provider_id else None
    ti = _tryit_state(row0)
    # Cloud try-it (any task type): tearing down = terminating the on-demand pod.
    if ti.get("target") == "cloud" or (
            (row0.task_type or "asr").lower() == "asr" and (prov0 is None or prov0.kind != "vm")):
        await _tryit_teardown(run_id)
        return PlaygroundStatus(running=False, ready=False,
                                kind=ti.get("kind") or (row0.task_type or "asr"),
                                logs=["[try-it] pod torn down"])
    row, kind, _m, _c, ssh, cfg = await _playground_ctx(run_id, user, session)
    try:
        st = await asyncio.to_thread(_playground_stop_ssh, *ssh, run_id, cfg, kind)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"stop failed: {e}")
    return PlaygroundStatus(**st)


@router.post("/{run_id}/playground/chat")
async def playground_chat(run_id: str, request: Request):
    """LLM try-it: stream a chat completion from the run's vLLM server (loaded via
    …/playground/start). Proxies to the VM's vLLM /v1/chat/completions over SSH
    (`curl -N`) and relays the OpenAI SSE straight to the browser.

    Auth + run/SSH resolution happen in a short-lived DB session that is RELEASED
    before the stream starts — a StreamingResponse must not hold a pooled DB
    connection for the whole stream (see the SSE-pool-exhaustion gotcha)."""
    from . import auth as _auth
    body = await request.json()
    async with session_factory()() as s:
        user = await _auth.current_user(request, s)
        if not await _auth.has_section(user, "autotrain", s):
            raise HTTPException(status_code=403, detail="no autotrain access")
        row, kind, _m, _c, ssh, cfg = await _playground_ctx(run_id, user, s)
        if kind != "llm":
            raise HTTPException(status_code=400, detail="chat is for LLM runs")
        paths = _playground_paths(cfg, run_id, "llm")
        vport = paths["port"]
        # Keep a cloud try-it pod alive across chats (else the idle reaper could reap it
        # mid-conversation — the setup task only bumps expiry until it goes ready).
        if _tryit_state(row).get("target") == "cloud":
            await _tryit_save(run_id, expires_at=_tryit_exp())
    host, p, suser, key = ssh

    msgs = body.get("messages") or []
    if not isinstance(msgs, list) or not msgs:
        raise HTTPException(status_code=400, detail="messages[] required")
    payload: dict = {"model": run_id, "messages": msgs, "stream": True,
                     "stream_options": {"include_usage": True}}
    for k in ("temperature", "top_p", "max_tokens", "seed", "presence_penalty",
              "frequency_penalty", "stop"):
        if body.get(k) is not None:
            payload[k] = body[k]
    # Tool calling — forwarded to vLLM verbatim (needs the server launched with
    # --enable-auto-tool-choice --tool-call-parser <p> via the custom vLLM args).
    # Accept either the OpenAI-wrapped shape or a bare {name, description, parameters}
    # (auto-wrapped), like the SyntheticGen playground, so pasted specs Just Work.
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        norm = []
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("function"), dict):
                norm.append(t)
            elif isinstance(t, dict) and isinstance(t.get("name"), str):
                norm.append({"type": "function", "function": {
                    "name": t["name"], "description": t.get("description") or "",
                    "parameters": t["parameters"] if isinstance(t.get("parameters"), dict)
                    else {"type": "object", "properties": {}},
                }})
            else:
                norm.append(t)
        payload["tools"] = norm
        payload["tool_choice"] = body.get("tool_choice") or "auto"
    req_bytes = json.dumps(payload).encode()
    remote_req = f"/tmp/sgpu_llm_chat_{run_id}.json"
    import shlex
    curl = (f"curl -sN --no-buffer -X POST http://127.0.0.1:{vport}/v1/chat/completions "
            f"-H 'Content-Type: application/json' --data @{shlex.quote(remote_req)}")

    async def gen():
        try:
            cli = await asyncio.to_thread(_ssh_connect, host, int(p), suser, key)
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'error': f'ssh failed: {e}'})}\n\n".encode()
            yield b"data: [DONE]\n\n"
            return
        try:
            await asyncio.to_thread(_ssh_put_bytes, cli, req_bytes, remote_req)
            chan = cli.get_transport().open_session()
            chan.settimeout(None)
            chan.set_combine_stderr(True)
            await asyncio.to_thread(chan.exec_command, curl)
            while True:
                data = await asyncio.to_thread(chan.recv, 8192)
                if not data:
                    break
                yield data
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode()
        finally:
            try:
                cli.close()
            except Exception:  # noqa: BLE001
                pass

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache, no-transform",
                                      "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})
