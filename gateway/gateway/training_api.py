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
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
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
# Valid training-audio augmentation techniques (mirror whisper_finetune._AUG_FUNCS).
_AUG_TECHNIQUES = {"telephone", "noise", "dropout", "gain", "pitch", "speed", "reverb", "bandpass"}

# Strong refs to in-flight runner tasks (else asyncio may GC them) + per-run
# teardown state (RunPod id + api key) so terminate can delete the pod.
_active_runners: dict[str, asyncio.Task] = {}
_RUN_STATE: dict[str, dict] = {}
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


async def _resolve_dataset_spec(dataset_id: str, hf_token_fallback: Optional[str] = None) -> dict:
    """Turn a Dataset row into the trainer's dataset spec, with creds inlined."""
    async with session_factory()() as s:
        ds = await s.get(Dataset, dataset_id)
        if ds is None:
            raise RuntimeError(f"dataset {dataset_id} not found")
        storage = await s.get(Storage, ds.storage_id) if ds.storage_id else None

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
    if ds.kind == "s3" and ds.s3_metadata_uri:
        # s3://bucket/key
        without = ds.s3_metadata_uri[len("s3://"):]
        bucket, _, key = without.partition("/")
        metadata_key = key
        bucket = bucket or creds["bucket"]
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
    call via asyncio.to_thread. Returns the remote exit status."""
    chan = cli.get_transport().open_session()
    chan.set_combine_stderr(True)
    chan.exec_command(command)
    buf = b""
    while True:
        if chan.recv_ready():
            data = chan.recv(8192)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                on_line(line.decode("utf-8", "replace"))
        elif chan.exit_status_ready() and not chan.recv_ready():
            break
        else:
            time.sleep(0.2)
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
    async with session_factory()() as s:
        row = await s.get(TrainingRun, run_id)
        if row is None or row.status in ("cancelled",):
            return
        row.status = status
        row.exit_code = exit_code
        if error_text:
            row.error_text = error_text[:4000]
        if result_json is not None:
            row.result_json = result_json
        row.ended_at = datetime.now(timezone.utc)
        await s.commit()


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
        if test_dataset_id and test_dataset_id != dataset_id:
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
        if cfg.get("hf_push_repo"):
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
                         ("@@TRIAL ", "trial")):
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
    rc = 1
    try:
        cfg["gpu_count"] = gpu_count
        task_type = (cfg.get("task_type") or "asr").lower()
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
            cfg["sweep_metric"] = "loss" if task_type == "tts" else (cfg.get("eval_metric") or "wer")
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
            remote_cfg = "/tmp/autotrain_config.json"
            await asyncio.to_thread(_ssh_put, cli, str(cfg_path), remote_cfg)
            base = _trainer_script_path().parent  # gateway/gateway/training/
            # Ship the worker for the task type …
            if task_type == "tts":
                await asyncio.to_thread(_ssh_exec, cli, "mkdir -p /tmp/tts")
                await asyncio.to_thread(_ssh_put, cli, str(base / "tts_finetune.py"), "/tmp/tts_finetune.py")
                for fn in ("convert_neucodec.py", "pack_stage1.py", "qwen3_tts_flash.py"):
                    await asyncio.to_thread(_ssh_put, cli, str(base / "tts" / fn), f"/tmp/tts/{fn}")
                worker_remote = "/tmp/tts_finetune.py"
            else:
                await asyncio.to_thread(_ssh_put, cli, str(_trainer_script_path()), "/tmp/whisper_finetune.py")
                worker_remote = "/tmp/whisper_finetune.py"
            # … then either the worker directly, or the sweep orchestrator over it.
            # User OS env (HOME, cache dirs, …) exported before the trainer in
            # both sweep and single runs — absolute values mkdir'd. GPU pinning
            # stays per-branch (the sweep orchestrator pins each trial itself).
            user_env = _render_env_exports(cfg.get("env_vars") or {})
            if sweep_on:
                await asyncio.to_thread(_ssh_put, cli, str(base / "sweep_runner.py"), "/tmp/sweep_runner.py")
                remote_script = "/tmp/sweep_runner.py"
                env_prefix = ""  # the orchestrator pins each trial itself
            else:
                remote_script = worker_remote
                env_prefix = f"CUDA_VISIBLE_DEVICES={visible_devices} " if visible_devices else ""
            cmd = f"{user_env}{env_prefix}python -u {remote_script} --config {remote_cfg}"
            await _push_log(redis, run_id, f"[gateway] $ {cmd}")
            rc = await asyncio.to_thread(_ssh_run_stream, cli, cmd, on_line)
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

    status = "done" if (rc == 0 and not result.get("error")) else "failed"
    err = result.get("error") if status == "failed" else None
    if status == "failed" and not err:
        err = f"trainer exited with code {rc}"
    await _finalize(run_id, status, rc, err, result_json=result)
    await _push_log(redis, run_id, f"[gateway] run {status} (rc={rc})")

    if runpod_id and api_key:
        await _terminate_pod(api_key, runpod_id)
        await _push_log(redis, run_id, f"[gateway] pod {runpod_id} torn down")
    _RUN_STATE.pop(run_id, None)
    try:
        await redis.expire(f"train:logs:{run_id}", LOG_LIST_TTL_S)
    except Exception:
        pass


# ---------- lifecycle hooks (mirror bench) ------------------------------


async def cleanup_orphaned_running(redis) -> int:
    """On startup, mark running/queued rows failed — the gateway restarted and
    their pods (if any) are dangling on RunPod."""
    n = 0
    async with session_factory()() as s:
        rows = (await s.execute(
            select(TrainingRun).where(TrainingRun.status.in_(["running", "queued"]))
        )).scalars().all()
        for row in rows:
            row.status = "failed"
            row.error_text = (
                "orphaned by gateway restart — pod (if any) is still on RunPod, "
                "terminate it manually"
            )
            row.ended_at = datetime.now(timezone.utc)
            n += 1
        if n:
            await s.commit()
    return n


# ---------- schemas ------------------------------------------------------


class CreateTrainingRunRequest(BaseModel):
    name: str
    dataset_id: str
    base_model: str
    task_type: str = "asr"             # "asr" | "tts"
    test_dataset_id: Optional[str] = None
    # ---- TTS-only (Qwen3 + NeuCodec) ----
    tokenizer: Optional[str] = None    # pack tokenizer (speech tokens); default set in runner
    block_size: int = 10240            # training context length
    pack_sequence_length: int = 4096   # per-utterance pack length
    default_speaker: Optional[str] = None
    speaker_field: Optional[str] = None
    # ---- hyperparameter sweep ----
    # {param: [values]} → cross-product = trials, run in a GPU-pinned pool on one
    # box (concurrency = #gpus / gpus_per_trial). Empty = single run.
    sweep: dict = {}
    gpus_per_trial: int = 1
    # Hyperparams + split + eval settings (all optional; trainer has defaults).
    eval_metric: str = "wer"           # "wer" | "cer" (ASR only)
    max_epochs: int = 3
    patience: int = 0                  # 0 = no early stop
    eval_split_pct: float = 10.0
    split_seed: int = 42
    batch_size: int = 8
    grad_accum: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 0
    weight_decay: float = 0.0
    # Emit a training-loss point every N optimizer steps (@@STEP) for the live
    # loss curve. Smaller N = smoother graph, more log lines.
    logging_steps: int = 10
    # Audio augmentation on TRAINING audio only — names from AUG_TECHNIQUES
    # (telephone/noise/dropout/gain/pitch/speed/reverb/bandpass). Empty = off.
    # One enabled technique is picked at random per augmented sample.
    augment_techniques: list[str] = []
    augment_prob: float = 0.5
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
    storage_id: Optional[str] = None
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


def _to_record(row: TrainingRun, owner_username: str) -> TrainingRunRecord:
    return TrainingRunRecord(
        id=row.id, name=row.name, status=row.status, dataset_id=row.dataset_id,
        test_dataset_id=row.test_dataset_id, base_model=row.base_model,
        task_type=row.task_type,
        s3_prefix=row.s3_prefix, config_json=row.config_json or {},
        exit_code=row.exit_code, error_text=row.error_text, result_json=row.result_json,
        created_by=owner_username, created_at=_iso(row.created_at) or "",
        started_at=_iso(row.started_at), ended_at=_iso(row.ended_at),
        cost_per_hr=row.cost_per_hr, provider_id=row.provider_id, storage_id=row.storage_id,
        gpu_type=row.gpu_type, gpu_count=row.gpu_count, visible_devices=row.visible_devices,
    )


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
    if body.test_dataset_id:
        tds = await session.get(Dataset, body.test_dataset_id)
        if tds is None or (tds.owner_id != user.id and not user.is_admin):
            raise HTTPException(status_code=400, detail="unknown test_dataset_id")
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

    run_id = _gen_id()
    target = await _training_s3_target(body.storage_id)
    s3_prefix = f"{target.prefix_root}{run_id}/"
    config = {
        "task_type": body.task_type,
        "tokenizer": body.tokenizer, "block_size": body.block_size,
        "pack_sequence_length": body.pack_sequence_length,
        "default_speaker": body.default_speaker, "speaker_field": body.speaker_field,
        "sweep": body.sweep or {}, "gpus_per_trial": body.gpus_per_trial,
        "eval_metric": body.eval_metric, "max_epochs": body.max_epochs,
        "patience": body.patience, "eval_split_pct": body.eval_split_pct,
        "split_seed": body.split_seed, "batch_size": body.batch_size,
        "grad_accum": body.grad_accum, "learning_rate": body.learning_rate,
        "warmup_steps": body.warmup_steps, "weight_decay": body.weight_decay,
        "logging_steps": body.logging_steps,
        "augment_techniques": [t for t in (body.augment_techniques or []) if t in _AUG_TECHNIQUES],
        "augment_prob": body.augment_prob,
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
    return _to_record(row, u.username if u else "?")


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
    row.status = "cancelled"
    row.ended_at = datetime.now(timezone.utc)
    await session.commit()
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
