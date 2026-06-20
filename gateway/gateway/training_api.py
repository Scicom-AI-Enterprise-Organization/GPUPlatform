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
import json
import logging
import math
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
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
_LORA_TARGET_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
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
    """gemma | minimax | mistral from an LLM base model id (mirrors
    llm_finetune.detect_arch + llm_pack.detect_arch). Drives the per-arch venv +
    trainer choice. Unknown → gemma (the default LLM base) so the venv path is still
    well-formed; the trainer raises a clear error on a truly unsupported model."""
    n = (model_id or "").lower()
    if "minimax" in n:
        return "minimax"
    if "mistral" in n:
        return "mistral"
    return "gemma"


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

    if ds.kind == "hf":
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
    if ds.kind == "llm_packed":
        # Pre-packed (chat multipack) ChiniDataset: s3_metadata_uri is the s3://
        # prefix of the shards (input_ids/labels/position_ids/attention_mask). The
        # LLM trainer downloads it → ./packed_data and runs gemma4.py directly.
        pack = (ds.split_fields or {}).get("_llm_pack") or {}
        return {
            "kind": "llm_packed",
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
    """Stream raw bytes to a remote file over an exec channel's stdin (no SFTP,
    no argv-length limit) — for payloads too big for _ssh_put's base64 argv
    (capped ~128 KB by MAX_ARG_STRLEN), e.g. an uploaded audio clip."""
    import shlex as _shlex

    write_cmd = f"mkdir -p \"$(dirname {_shlex.quote(remote)})\" && cat > {_shlex.quote(remote)}"
    chan = cli.get_transport().open_session()
    chan.set_combine_stderr(True)
    chan.exec_command(f"bash -c {_shlex.quote(write_cmd)}")
    chan.sendall(data)
    chan.shutdown_write()
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
        return ti

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(local_dir, arcname=".", filter=_filter)
    _ssh_put_bytes(cli, buf.getvalue(), "/tmp/_sgpu_ship.tar.gz")
    rc = _ssh_exec(
        cli,
        f"mkdir -p {_shlex.quote(remote_dir)} && "
        f"tar -xzf /tmp/_sgpu_ship.tar.gz -C {_shlex.quote(remote_dir)} && rm -f /tmp/_sgpu_ship.tar.gz",
    )
    if rc != 0:
        raise RuntimeError(f"failed to ship/untar {local_dir} → {remote_dir} (rc={rc})")


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


async def _terminate_pod(api_key: str, runpod_id: str) -> None:
    try:
        async with compute._client(api_key=api_key) as cli:
            await cli.delete(f"/pods/{runpod_id}")
    except Exception as e:  # noqa: BLE001
        logger.warning("training: pod %s teardown failed: %s", runpod_id, e)


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
    async with session_factory()() as s:
        src = await s.get(Dataset, src_id) if src_id else None
        run = await s.get(TrainingRun, run_id)
        base = (src.name if src else (run_id if run else src_id)) or run_id
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


async def _create_label_project_for_run(run_id: str, cfg: dict, result: dict, redis) -> None:
    """Post-train: synthesize N clips from the finished TTS model on the run's VM,
    upload them to the run's S3 storage, then create a Label-platform *recording*
    project (MOS rating enabled), configure its storage, import the clips as tasks,
    and auto-create a round-trip kind=label Dataset. Records the project under
    result_json.label_project. VM-provider runs only (synthesis needs the box)."""
    # URL + token each resolve from a Secrets-page (GlobalEnv) key when one was
    # picked, else from the pasted value (token = Fernet-encrypted label_token_enc).
    genv = await _resolve_global_env()
    url_secret = (cfg.get("label_base_url_secret") or "").strip()
    if url_secret:
        base_url = (genv.get(url_secret) or "").strip().rstrip("/")
    else:
        base_url = (cfg.get("label_base_url") or "").strip().rstrip("/")
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
        await _push_log(redis, run_id, "[gateway] label export: base_url missing/unresolved — skipped")
        return
    if not token:
        await _push_log(redis, run_id, "[gateway] label export: token missing/unresolved — skipped")
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
    if prov is None or prov.kind != "vm":
        await _push_log(redis, run_id, "[gateway] label export needs a VM provider (skipped)")
        return

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

    ssh = await _resolve_run_ssh(row)
    if ssh is None:
        await _push_log(redis, run_id, "[gateway] label export: can't reach the run's VM (skipped)")
        await _set_label_export_state(run_id, {"status": "failed", "error": "can't reach the run's VM"})
        return

    # Mark the export in-flight so the UI shows "exporting to Label" instead of the
    # run's terminal "done" status until it finishes (or fails).
    await _set_label_export_state(run_id, {"status": "running"})
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
            is_random, n, speaker, creds["bucket"], upload_prefix, dict(cfg), "auto",
            export_lines.append,
        )
        items = manifest.get("items") or []
        if not items:
            await _push_log(redis, run_id, "[gateway] label export: no clips synthesized")
            await _set_label_export_state(run_id, {"status": "failed", "error": "no clips synthesized"})
            return

        # ---- Label platform: create project → storage → MOS → tasks ----
        import httpx

        proj_name = (cfg.get("label_project_name") or f"{run_name}-eval")[:200]
        axes = [a for a in (cfg.get("label_mos_axes") or []) if a] or ["Naturalness", "Intelligibility", "Noise"]
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as cli:
            r = await cli.post(f"{base_url}/api/projects", headers=headers, json={
                "name": proj_name, "type": "recording",
                "description": f"Autotrain TTS eval — {run_name} ({run_id})",
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
                     for it in items if it.get("key")]
            r = await cli.post(f"{base_url}/api/projects/{pid}/tasks", headers=headers, json={"tasks": tasks})
            r.raise_for_status()

        # Round-trip: a kind=label dataset pointing at the new project + stamp the run.
        label_ds_id = "ds-" + os.urandom(4).hex()
        async with session_factory()() as s:
            s.add(Dataset(
                id=label_ds_id, owner_id=owner_id,
                name=(f"{run_name}-tts-eval-labels")[:255],
                description=(f"Human MOS / recording labels for autotrain TTS run {run_id}")[:2048],
                kind="label", storage_id=storage_id,
                label_base_url=base_url, label_project_id=pid,
                # Reference the Secrets key when one was used (no token copy); else store
                # the resolved token Fernet-encrypted, like the datasets create path.
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
                    "count": len(tasks), "dataset_id": label_ds_id, "project_name": proj_name,
                }
                rj["label_export"] = {"status": "done"}
                run2.result_json = rj
            await s.commit()
        await _push_log(
            redis, run_id,
            f"[gateway] label project created: {base_url}/dashboard/projects/{pid} "
            f"({len(tasks)} clips) + dataset {label_ds_id}")
    except Exception as e:  # noqa: BLE001 — record the failure, then re-raise so the caller logs it
        await _set_label_export_state(run_id, {"status": "failed", "error": str(e)[:300]})
        raise
    finally:
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

        cfg["dataset"] = await _resolve_dataset_spec(dataset_id, g("HF_TOKEN"))
        if cfg.get("no_eval"):
            # "No test set" — train on everything, no eval. Never resolve a test
            # dataset or set test_from_split (the trainers force eval off).
            await _push_log(redis, run_id, "[gateway] no_eval: training with no test set / no eval")
        elif test_dataset_id and test_dataset_id != dataset_id:
            cfg["test_dataset"] = await _resolve_dataset_spec(test_dataset_id, g("HF_TOKEN"))
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
        # LLM training downloads the gated gemma-4 base model on the box, so it
        # always needs the HF token (not just when pushing the result to HF).
        if cfg.get("hf_push_repo") or (cfg.get("task_type") or "").lower() == "llm":
            cfg["hf_token"] = g("HF_TOKEN")
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

        if not is_vm:
            api_key = await compute._resolve_api_key(provider_id)
            key_filename, pub = _gen_ssh_key(work)
            await _push_log(redis, run_id,
                            f"[gateway] provisioning RunPod pod ({gpu_type} x{gpu_count}) …")
            runpod_id, host, port, cost = await _provision_pod(
                api_key, f"sgpu-train-{run_id}", cfg.get("image", DEFAULT_IMAGE),
                gpu_type, gpu_count, bool(cfg.get("secure_cloud", True)),
                int(cfg.get("disk_gb", 60)), int(cfg.get("volume_gb", 80)), pub,
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
        # launches from {venv}/bin/python.
        if task_type == "llm":
            # gemma vs minimax need different `kernels` pins (transformers 5.5.0
            # import crashes outside each range) → SEPARATE venvs per arch.
            _default_venv = f"/share/autotrain-llm-{_llm_arch(cfg.get('base_model'))}"
        elif task_type == "tts":
            _default_venv = "/share/autotrain-tts"
        else:
            _default_venv = "/share/autotrain-whisper"
        venv_path = (cfg.get("venv_path") or _default_venv).rstrip("/")
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
            if task_type == "tts":
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

    # After a successful TTS run, optionally synthesize a handful of clips from the
    # trained model and seed a Label-platform recording+MOS project with them.
    # Best-effort: a failure here never marks the run failed.
    if status == "done" and (cfg.get("task_type") == "tts") and cfg.get("label_export"):
        try:
            await _create_label_project_for_run(run_id, cfg, result, redis)
        except Exception as e:  # noqa: BLE001
            await _push_log(redis, run_id, f"[gateway] label export failed: {e}")

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
            await s.commit()
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
        return "running"  # detached trainer survived; finalize when it exits
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
        # Queued rows never launched a detached process → re-run them (a restart
        # mid-queue shouldn't drop the run). _redispatch_run is bounded + VM-only.
        if row.status == "queued":
            await _redispatch_run(redis, row.id, "never started")
            n += 1
            continue
        if (await _reconcile_orphan(row, redis)) not in ("running", "requeued"):
            n += 1
    return n


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
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("training janitor: %s", e)


# ---------- schemas ------------------------------------------------------


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
    # LLM-only (gemma4): which linear projections to apply LoRA to. Default is the
    # attention projections (q/k/v/o); add MLP/dense layers (gate_proj, up_proj,
    # down_proj) to adapt those too. LLM finetune is always LoRA. Unknown names are
    # dropped; an empty result falls back to the q/k/v/o default.
    lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
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
    # TTS-only: audio eval methods to run on the test set (cer | mos | similarity).
    eval_methods: list[str] = []
    # TTS-only: how many generated clips the heavy eval scores (per method). The
    # gen + NeuCodec-decode + scorer pass dominates a short run, so a small count
    # keeps debug runs fast. Default 64.
    eval_max_samples: int = 64
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
    precision: str = "fp32-bf16"        # "<load>-<amp>", e.g. fp32-bf16
    language: Optional[str] = None
    task: str = "transcribe"
    # Where to run.
    provider_id: Optional[str] = None  # kind=vm → bare metal; else RunPod
    gpu_type: str = "NVIDIA L40S"
    gpu_count: int = 1
    secure_cloud: bool = True
    disk_gb: int = 60
    volume_gb: int = 80
    visible_devices: Optional[str] = None
    storage_id: Optional[str] = None
    hf_push_repo: Optional[str] = None
    # Roomy dir on the remote for checkpoints + temp (TMPDIR). Defaults to
    # /share (the VM's big volume); /tmp is a small disk that overflows on big
    # models. The best model is uploaded to S3 regardless.
    work_dir: str = "/share"
    # Isolated uv venv for the trainer's deps (like serverless's vLLM venv_path) —
    # keeps the heavy stack off the box's system python. Default per task:
    # /share/autotrain-whisper (asr) or /share/autotrain-tts (tts).
    venv_path: Optional[str] = None
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


def _to_record(
    row: TrainingRun, owner_username: str,
    provider_name: Optional[str] = None, provider_kind: Optional[str] = None,
    storage_name: Optional[str] = None,
) -> TrainingRunRecord:
    return TrainingRunRecord(
        id=row.id, name=row.name, status=row.status, dataset_id=row.dataset_id,
        test_dataset_id=row.test_dataset_id, base_model=row.base_model,
        task_type=row.task_type,
        s3_prefix=row.s3_prefix, config_json=row.config_json or {},
        exit_code=row.exit_code, error_text=row.error_text, result_json=row.result_json,
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


@router.post("", response_model=TrainingRunRecord)
async def create_training_run(
    body: CreateTrainingRunRequest,
    request: Request,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
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
        # LLM finetune consumes a pre-packed chat ChiniDataset (kind=llm_packed) —
        # the trainer (gemma4.py / minimax_m2.py, chosen by base-model arch) reads
        # the packed ids directly (no re-tokenization).
        if ds.kind != "llm_packed":
            raise HTTPException(
                status_code=400,
                detail="LLM training needs a packed dataset (kind=llm_packed) — pack it first via 'Pack for LLM'",
            )
        if tds is not None and tds.kind != "llm_packed":
            raise HTTPException(status_code=400, detail="LLM test dataset must also be packed (kind=llm_packed)")
        _pm = (ds.split_fields or {}).get("_llm_pack") or {}
        if _pm.get("sequence_length"):
            body.block_size = int(_pm["sequence_length"])
        # The dataset was tokenized at pack time with a specific tokenizer/arch; the
        # trainer reads those ids verbatim, so the base model's arch MUST match the
        # pack arch (a gemma-packed dataset trained as minimax = garbage ids).
        _pack_arch = _pm.get("arch")
        _model_arch = _llm_arch(body.base_model)
        if _pack_arch and _pack_arch in ("gemma", "minimax", "mistral") and _pack_arch != _model_arch:
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
        if prov.kind == "vm":
            is_vm_run = True
            vm_gpus = (prov.config or {}).get("gpus") or []
            if vm_gpus:
                eff_gpu_type = vm_gpus[0]

    # ---- validate GPU pin, learning rate, and sweep values (clear 400s) ----
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
    if not math.isfinite(body.learning_rate) or body.learning_rate <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"learning_rate must be a positive number like 1e-4 (got {body.learning_rate!r})",
        )
    _validate_sweep(body.sweep or {})
    sweep_on = any(isinstance(v, list) and v for v in (body.sweep or {}).values())
    if sweep_on:
        if body.gpus_per_trial < 1:
            raise HTTPException(status_code=400, detail="gpus_per_trial must be at least 1")
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
        "grad_accum": body.grad_accum, "learning_rate": body.learning_rate,
        "warmup_steps": body.warmup_steps, "lr_scheduler_type": body.lr_scheduler_type,
        "weight_decay": body.weight_decay,
        # LLM finetune (gemma4) is always LoRA — force it on regardless of the toggle.
        "use_lora": True if body.task_type == "llm" else body.use_lora,
        "lora_r": body.lora_r,
        "lora_alpha_ratio": body.lora_alpha_ratio,
        "lora_alpha": body.lora_alpha, "lora_dropout": body.lora_dropout,
        "lora_target_modules": ([m for m in (body.lora_target_modules or [])
                                 if m in _LORA_TARGET_MODULES] or list(_LORA_TARGET_DEFAULT)),
        "freeze_encoder": body.freeze_encoder, "use_ddp": body.use_ddp,
        "logging_steps": body.logging_steps,
        "augment_techniques": [t for t in (body.augment_techniques or []) if t in _AUG_TECHNIQUES],
        "augment_prob": body.augment_prob,
        "eval_methods": [m for m in (body.eval_methods or []) if m in _TTS_EVAL_METHODS],
        "eval_max_samples": body.eval_max_samples,
        # Post-train Label-platform export (TTS only). The token is stored Fernet-
        # encrypted (label_token_enc) like the kind=label dataset's — never raw.
        "label_export": bool(body.label_export and body.task_type == "tts"),
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
        "precision": body.precision, "language": body.language, "task": body.task,
        "base_model": body.base_model,
        # Cloud-pod knobs are irrelevant on a VM — omit them so the config tab
        # matches reality (VM hardware is fixed; gpu_type reflects the VM).
        **({} if is_vm_run else {
            "secure_cloud": body.secure_cloud,
            "disk_gb": body.disk_gb, "volume_gb": body.volume_gb,
        }),
        "hf_push_repo": body.hf_push_repo,
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
    }
    row = TrainingRun(
        id=run_id, name=body.name.strip() or run_id, dataset_id=body.dataset_id,
        test_dataset_id=body.test_dataset_id, base_model=body.base_model,
        task_type=body.task_type,
        config_json=config, status="queued", s3_prefix=s3_prefix, owner_id=user.id,
        provider_id=body.provider_id, storage_id=body.storage_id,
        gpu_type=eff_gpu_type, gpu_count=body.gpu_count, visible_devices=body.visible_devices,
    )
    session.add(row)
    await session.commit()

    redis = request.app.state.redis
    task = asyncio.create_task(_safe_run(redis, run_id))
    _active_runners[run_id] = task
    task.add_done_callback(lambda _t: _active_runners.pop(run_id, None))

    return _to_record(row, user.username)


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
    q = select(TrainingRun).order_by(TrainingRun.created_at.desc())
    if not (scope == "all" and user.is_admin):
        q = q.where(TrainingRun.owner_id == user.id)
    rows = (await session.execute(q)).scalars().all()
    # owner username map
    names: dict[int, str] = {}
    for row in rows:
        if row.owner_id not in names:
            u = await session.get(User, row.owner_id)
            names[row.owner_id] = u.username if u else "?"
    return [_to_record(r, names.get(r.owner_id, "?")) for r in rows]


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
    return _to_record(
        row, u.username if u else "?",
        provider_name=prov.name if prov else None,
        provider_kind=prov.kind if prov else None,
        storage_name=store.name if store else None,
    )


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
    result_json.label_project card appears when done. VM-provider runs only."""
    row = await _owned(run_id, user, session)
    if (row.task_type or "asr") != "tts":
        raise HTTPException(status_code=400, detail="label export is for TTS runs")
    if row.status != "done":
        raise HTTPException(status_code=400, detail="the run must finish successfully first")
    if not (((row.result_json or {}).get("artifact") or {}).get("s3_uri")):
        raise HTTPException(status_code=400, detail="no trained model artifact for this run")
    prov = await session.get(Provider, row.provider_id) if row.provider_id else None
    if prov is None or prov.kind != "vm":
        raise HTTPException(status_code=400, detail=("label export runs on a VM provider; this run "
                            "used a cloud pod (gone after training). Re-run on a VM."))
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
    cfg["label_export"] = True

    has_url = bool((cfg.get("label_base_url_secret") or "").strip() or (cfg.get("label_base_url") or "").strip())
    has_tok = bool((cfg.get("label_token_secret") or "").strip() or cfg.get("label_token_enc"))
    if not has_url:
        raise HTTPException(status_code=400, detail="provide the Label platform URL")
    if not has_tok:
        raise HTTPException(status_code=400, detail="provide a Label platform API token (or pick a secret)")

    # Persist the (merged) label_* fields so the run remembers them for next time.
    result = dict(row.result_json or {})
    async with session_factory()() as s:
        r2 = await s.get(TrainingRun, run_id)
        if r2 is not None:
            merged = dict(r2.config_json or {})
            for k in ("label_export", "label_base_url", "label_base_url_secret",
                      "label_token_secret", "label_token_enc", "label_project_name",
                      "label_samples", "label_mos_axes", "label_speakers", "label_speaker_prefix"):
                merged[k] = cfg.get(k)
            r2.config_json = merged
            await s.commit()

    redis = request.app.state.redis

    async def _bg() -> None:
        try:
            await _push_log(redis, run_id, "[gateway] label export: retrying on request …")
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


class HfExportRequest(BaseModel):
    repo: str
    storage_id: Optional[str] = None   # a kind=huggingface Storage (provides the token)
    private: bool = False


@router.post("/{run_id}/hf-export")
async def export_to_huggingface(
    run_id: str,
    body: HfExportRequest,
    request: Request,
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Push this run's BEST/final model to a Hugging Face repo on demand. The model
    artifact (result_json.artifact.s3_uri) is the best checkpoint — the trainer only
    uploads the final/best model to S3, never the intermediate checkpoint-N dirs — so
    this pushes exactly that. The HF token comes from the selected kind=huggingface
    Storage (else the platform HF_TOKEN secret). Runs in the background; status +
    link land in result_json.hf_export. VM-provider runs only."""
    row = await _owned(run_id, user, session)
    if row.status != "done":
        raise HTTPException(status_code=400, detail="the run must finish successfully first")
    model_s3 = (((row.result_json or {}).get("artifact") or {}).get("s3_uri"))
    if not model_s3:
        raise HTTPException(status_code=400, detail="no trained model artifact for this run")
    repo = (body.repo or "").strip()
    if not repo:
        raise HTTPException(status_code=400, detail="a target repo (org/name) is required")
    token = await _hf_token_for_storage(body.storage_id, session)
    if not token:
        token = (await _resolve_global_env()).get("HF_TOKEN")
    if not token:
        raise HTTPException(status_code=400, detail=("no Hugging Face token — pick a HuggingFace storage "
                            "or set HF_TOKEN in Secrets"))
    # Custom HF_ENDPOINT from the same storage (self-hosted mirror), or None → huggingface.co.
    hf_endpoint = await _hf_endpoint_for_storage(body.storage_id, session)
    # A custom endpoint pushes from the GATEWAY (the control plane reaches the mirror
    # directly + its huggingface_hub handles the `…/hf` path the VM's older client
    # mis-parses); the model artifact lives on S3 so no VM is involved. The default
    # huggingface.co path still runs on the VM (avoids large downloads on the laptop).
    local_export = bool(hf_endpoint)
    # For a gateway-side push to our OWN mirror, talk to it over loopback so a
    # multi-GB LFS PUT bypasses the nginx ingress body-size cap (+ TLS). Keep the
    # original endpoint for the displayed/stored repo URL.
    push_endpoint = _loopback_endpoint(hf_endpoint) if local_export else hf_endpoint
    creds = _s3_creds_from_storage(await session.get(Storage, row.storage_id) if row.storage_id else None)
    ssh = None
    if not local_export:
        prov = await session.get(Provider, row.provider_id) if row.provider_id else None
        if prov is None or prov.kind != "vm":
            raise HTTPException(status_code=400, detail=("HF export runs on a VM provider; this run used a "
                                "cloud pod (gone after training). Re-run on a VM, or use a HuggingFace "
                                "storage with a custom endpoint (pushes from the gateway)."))
        ssh = await _resolve_run_ssh(row)
        if ssh is None:
            raise HTTPException(status_code=400, detail="can't reach the run's VM (SSH coords unavailable)")
    cfg = dict(row.config_json or {})
    private = bool(body.private)
    redis = request.app.state.redis

    # Re-click supersedes a stuck push (the orphaned upload just finishes + is discarded).
    prev = _active_hf_exports.pop(run_id, None)
    if prev is not None and not prev.done():
        prev.cancel()
        await _push_log(redis, run_id, "[gateway] HF export: superseded by a new request — restarting")
    await _set_hf_export_state(run_id, {"status": "running", "repo": repo, "url": None, "error": None})

    async def _bg() -> None:
        # Stream the VM-side download/upload lines to the run's logs (the script runs
        # in a thread; it appends to this buffer and an async pump mirrors to Redis)
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
        try:
            where = "the gateway" if local_export else "the run's VM"
            await _push_log(redis, run_id, f"[gateway] exporting best model to Hugging Face → {repo} (from {where}) …")
            if local_export:
                if push_endpoint != hf_endpoint:
                    await _push_log(redis, run_id, f"[gateway] pushing via {push_endpoint} (loopback, bypassing the ingress) …")
                res = await asyncio.to_thread(
                    _run_hf_export_local, run_id, model_s3, creds, repo, token, private,
                    push_endpoint, export_lines.append)
            else:
                res = await asyncio.to_thread(
                    _run_hf_export_ssh, *ssh, run_id, model_s3, creds, repo, token, private, cfg,
                    hf_endpoint, export_lines.append)
            # The script builds the URL from the endpoint it pushed to; show the
            # public endpoint instead of the loopback one.
            res_url = res.get("url")
            if local_export and push_endpoint and push_endpoint != hf_endpoint and res_url:
                res_url = res_url.replace(push_endpoint, hf_endpoint, 1)
            async with session_factory()() as s:
                r2 = await s.get(TrainingRun, run_id)
                if r2 is not None:
                    rj = dict(r2.result_json or {})
                    rj["hf_export"] = {"status": "done", "repo": res.get("repo", repo), "url": res_url}
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
                    f"pkill -9 -f sgpu_hf_export_{run_id} 2>/dev/null; sleep 1; "
                    f"if pgrep -f sgpu_hf_export_{run_id}.py >/dev/null; then echo ALIVE; else echo DEAD; fi; "
                    f"rm -rf /tmp/sgpu_hf_export_{run_id}.* {work_dir}/sgpu-hf-export/{run_id} 2>/dev/null || true",
                )
                return "DEAD" in out
            finally:
                cli.close()

        try:
            killed = await asyncio.to_thread(_kill)
        except Exception as e:  # noqa: BLE001
            await _push_log(redis, run_id, f"[gateway] HF export: VM kill attempt failed: {e}")

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
        "best": r.get("best"),
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
            if cur is None or cur.status in ("done", "failed", "cancelled"):
                yield "event: end\ndata: end\n\n"
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream")


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
    # RunPod pod
    if not row.runpod_pod_id:
        return None
    try:
        api_key = await compute._resolve_api_key(row.provider_id)
        async with compute._client(api_key=api_key) as cli:
            pr = await cli.get(f"/pods/{row.runpod_pod_id}")
        ip, port = compute._extract_ssh(pr.json() or {})
    except Exception:
        return None
    key = str(work / "id_ed25519")
    if not ip or not port or not Path(key).exists():
        return None
    return ip, int(port), "root", key


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
        # LLM (gemma-4): merge LoRA → save → serve with vLLM. The training venv runs
        # the merge; vLLM serves from its own dedicated venv. Merged model is cached.
        tdir = f"{work}/sgpu-llm-tryit/{run_id}"
        venv = (cfg.get("venv_path") or "/share/autotrain-llm").rstrip("/")
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
           f'echo "@@STATUSLOG@@"; tail -n 14 {paths["log"]} 2>/dev/null')
    head, _, logpart = _ssh_capture(cli, cmd).partition("@@STATUSLOG@@")
    line = next((x for x in head.strip().splitlines() if x.startswith(("READY", "LOADING", "DOWN"))), "DOWN")
    logs = [x for x in logpart.splitlines() if x.strip()][-14:]
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
                              base_model, gpus, served_name, max_model_len=16384, vllm_args=""):
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
        out: dict = {"text": None, "device": None, "error": None, "lines": []}

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
        return out["text"], out.get("device"), out["lines"][-120:]
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
    if prov is None or prov.kind != "vm":
        raise HTTPException(
            status_code=400,
            detail=("try-it runs on a VM provider; this run used a cloud pod "
                    "(gone after training). Push to HF and try there, or re-run on a VM."),
        )
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
        text, device, logs = await asyncio.to_thread(
            _run_transcribe_ssh, *ssh, run_id, model_s3, creds, audio,
            filename or "audio.wav", dict(row.config_json or {}), sel or "auto",
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"transcription failed: {e}")
    return TranscribeResponse(text=text, device=device, logs=logs)


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
        venv = (cfg.get("venv_path") or "/share/autotrain-tts").rstrip("/")
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


async def _playground_ctx(run_id: str, user: User, session: AsyncSession):
    """Validate + resolve everything the playground lifecycle needs: the run must
    be a finished ASR/TTS run on a VM with a model artifact. Returns
    (row, kind, model_s3, creds, ssh, cfg)."""
    row = await _owned(run_id, user, session)
    kind = (row.task_type or "asr").lower()
    if kind not in ("asr", "tts", "llm"):
        raise HTTPException(status_code=400, detail="try-it is for ASR/TTS/LLM runs")
    if row.status != "done":
        raise HTTPException(status_code=400, detail="the run must finish training first")
    model_s3 = ((row.result_json or {}).get("artifact") or {}).get("s3_uri")
    if not model_s3:
        raise HTTPException(status_code=400, detail="no trained model artifact for this run")
    prov = await session.get(Provider, row.provider_id) if row.provider_id else None
    if prov is None or prov.kind != "vm":
        raise HTTPException(status_code=400, detail="persistent try-it needs a VM provider")
    storage = await session.get(Storage, row.storage_id) if row.storage_id else None
    creds = _s3_creds_from_storage(storage)
    ssh = await _resolve_run_ssh(row)
    if ssh is None:
        raise HTTPException(status_code=400, detail="can't reach the run's VM (SSH coords unavailable)")
    return row, kind, model_s3, creds, ssh, dict(row.config_json or {})


@router.post("/{run_id}/playground/start", response_model=PlaygroundStatus)
async def playground_start(
    run_id: str,
    gpu: Optional[str] = Query(None, description="GPU index, 'cpu', or 'auto'"),
    idle_minutes: float = Query(5, ge=0, le=120,
                                description="auto-unload after this many idle minutes (0 = never)"),
    vllm_args: Optional[str] = Query(None,
        description="LLM only: extra vLLM serve CLI args, verbatim (e.g. '--enable-auto-tool-choice "
                    "--tool-call-parser hermes --max-model-len 32768'). Appended last so they override "
                    "the defaults; the platform-set flags (model/port/served-name/tp) are rejected."),
    user: User = Depends(require_section("autotrain")),
    session: AsyncSession = Depends(get_session),
):
    """Load the run's model into a persistent worker on its VM (served over a Unix
    socket) so try-it requests skip the per-call model load. Returns immediately;
    poll …/playground/status until ready. Auto-unloads after idle_minutes idle.

    LLM (gemma-4) runs serve with vLLM (eager); `gpu` may be a comma-list (e.g. "6,7")
    to pick the serve GPUs (TP = count), defaulting to the run's training pin:
    download LoRA → merge → save → vLLM serve."""
    row, kind, model_s3, creds, ssh, cfg = await _playground_ctx(run_id, user, session)
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
        try:
            st = await asyncio.to_thread(_llm_playground_start_ssh, *ssh, run_id, model_s3, creds, cfg,
                                         (row.base_model or cfg.get("base_model") or "google/gemma-4-31B-it"),
                                         gpus, run_id, 16384, va)
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
    """Unload the persistent worker and free the GPU (SIGTERM its process group)."""
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
