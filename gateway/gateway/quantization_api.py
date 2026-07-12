"""Quantization feature — llm-compressor (compressed-tensors) post-training
quantization, SSH-orchestrated by the gateway.

A sibling of the Autotrain feature (training_api.py): the gateway owns the box
lifecycle — provision a RunPod pod (or use a registered VM), SFTP a self-contained
worker (`training/quantize.py`) onto it, run it over SSH, stream stdout to a Redis
list for live SSE replay, parse `@@ARTIFACT/@@DONE` lines into `result_json`, then
upload logs to S3 and tear the pod down. The heavy SSH / pod / dataset plumbing is
REUSED from training_api (imported as `ta`) rather than re-implemented — this module
owns only the quantization-specific bits (the DB row, the recipe schemes, and the
worker contract).

What it does: pull a base model from HuggingFace, quantize it with llm-compressor
(one of a curated set of schemes — some data-free, some needing dataset calibration
drawn from a Datasets resource), save the compressed-tensors model to S3, and
optionally push it back to the HF Hub.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
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
from . import compute
from . import training_api as ta
from .auth import require_section
from .bench import S3Target, s3_list, s3_presign_get, s3_put_text
from .db import Base, Dataset, Provider, Storage, User, get_session, session_factory

logger = logging.getLogger("gateway.quantization")

LOG_LIST_CAP = ta.LOG_LIST_CAP
LOG_LIST_TTL_S = ta.LOG_LIST_TTL_S
# llm-compressor needs a recent torch (compressed-tensors ≥0.7). Default RunPod pods
# to the cu1300 image (≥580 driver) so the worker's venv torch (whatever it resolves)
# always has a compatible host — same reasoning as the LLM try-it/merge path.
DEFAULT_IMAGE = ta.LLM_VLLM_IMAGE

# ---------- quantization schemes -----------------------------------------
# id → (label, needs_calibration). The worker (training/quantize.py) maps each id to
# an llm-compressor recipe; keep this set in sync with QUANT_SCHEMES there and with
# the web form's scheme dropdown. Calibrated schemes require a calibration dataset.
_SCHEMES: dict[str, tuple[str, bool]] = {
    "fp8-dynamic": ("FP8 dynamic (W8A8, data-free)", False),
    "w4a16": ("W4A16 (GPTQ, 4-bit weights)", True),
    "w8a8-int8": ("W8A8 INT8 (SmoothQuant + GPTQ)", True),
    "fp8": ("FP8 static (W8A8, calibrated)", True),
    "nvfp4": ("NVFP4 (4-bit microscale)", True),
    "awq": ("AWQ (W4A16)", True),
}
# Calibration datasets must carry text — reuse the Datasets resource but only the
# text-bearing kinds (an audio/packed dataset has nothing to calibrate an LLM on).
_CALIB_DATASET_KINDS = ("hf", "llm", "upload", "s3")


# ---------- DB model (mirror TrainingRun) --------------------------------


class QuantizationJob(Base):
    __tablename__ = "quantization_jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # quant-<hex8>
    name: Mapped[str] = mapped_column(String(128))
    # HF repo id of the base model to pull + quantize (e.g. "Qwen/Qwen3-8B").
    source_model: Mapped[str] = mapped_column(String(255))
    # One of _SCHEMES keys.
    scheme: Mapped[str] = mapped_column(String(32), default="fp8-dynamic", server_default="fp8-dynamic", nullable=False)
    # Calibration dataset (Datasets resource id) — only for calibrated schemes.
    calibration_dataset_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    # Full job config (recipe knobs + push target). Credentials are NEVER stored
    # here — they're resolved + injected into the box at run time.
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    s3_prefix: Mapped[str] = mapped_column(String(255))
    exit_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_text: Mapped[Optional[str]] = mapped_column(String(4096), nullable=True)
    # {"artifact": {s3_uri, hf_repo}, "hf_export": {...}, "sizes": {...}, "progress": …}
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


# Strong refs to in-flight runner tasks + per-job teardown state (RunPod id + api key).
_active_runners: dict[str, asyncio.Task] = {}
_RUN_STATE: dict[str, dict] = {}
_active_hf_exports: dict[str, asyncio.Task] = {}


# ---------- helpers ------------------------------------------------------


def _gen_id() -> str:
    import uuid
    return "quant-" + uuid.uuid4().hex[:8]


def _work_dir(job_id: str) -> Path:
    d = Path(os.environ.get("QUANT_WORK_DIR", "/tmp/sgpu-quant")) / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _full_log_path(job_id: str) -> Path:
    return _work_dir(job_id) / "_full.log"


def _worker_script_path() -> Path:
    return Path(__file__).parent / "training" / "quantize.py"


def _remote_run_paths(job_id: str) -> tuple[str, str, str]:
    """(log, pidfile, runscript) on the VM/pod for a detached quantization run."""
    return (f"/tmp/sgpu_quant_{job_id}.log", f"/tmp/sgpu_quant_{job_id}.pid", f"/tmp/sgpu_quant_{job_id}.sh")


async def _quant_s3_target(storage_id: Optional[str]) -> S3Target:
    """S3 destination for a job's logs + quantized model. prefix_root ends with
    'quantization-jobs/'. Mirrors training_api._training_s3_target."""
    creds = ta._s3_creds_from_storage(None)
    if storage_id:
        async with session_factory()() as s:
            row = await s.get(Storage, storage_id)
        creds = ta._s3_creds_from_storage(row)
    prefix = creds["prefix"]
    prefix_root = f"{prefix}/quantization-jobs/" if prefix else "quantization-jobs/"
    return S3Target(
        bucket=creds["bucket"], region=creds["region"], endpoint=creds["endpoint"],
        access_key=creds["access_key"], secret_key=creds["secret_key"], prefix_root=prefix_root,
    )


async def _push_redis(redis, job_id: str, line: str) -> None:
    key = f"quant:logs:{job_id}"
    try:
        await redis.rpush(key, line)
        await redis.ltrim(key, -LOG_LIST_CAP, -1)
    except Exception:
        pass


async def _push_log(redis, job_id: str, line: str) -> None:
    await _push_redis(redis, job_id, line)
    try:
        with open(_full_log_path(job_id), "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


async def _flush_result(job_id: str, result: dict) -> None:
    snap = json.loads(json.dumps(result))
    async with session_factory()() as s:
        row = await s.get(QuantizationJob, job_id)
        if row is None or row.status not in ("running",):
            return
        row.result_json = snap
        await s.commit()


async def _finalize(job_id: str, status: str, exit_code: Optional[int],
                    error_text: Optional[str], result_json: Optional[dict] = None) -> None:
    async with session_factory()() as s:
        row = await s.get(QuantizationJob, job_id)
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


# ---------- the runner ---------------------------------------------------


async def _safe_run(redis, job_id: str) -> None:
    try:
        await run_quantization(redis, job_id)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("quantization job %s crashed", job_id)
        await _finalize(job_id, "failed", None, "internal error — see gateway logs")


async def run_quantization(redis, job_id: str) -> None:
    work = _work_dir(job_id)
    async with session_factory()() as s:
        row = await s.get(QuantizationJob, job_id)
        if row is None:
            return
        row.status = "running"
        row.started_at = datetime.now(timezone.utc)
        await s.commit()
        cfg = dict(row.config_json or {})
        provider_id = row.provider_id
        storage_id = row.storage_id
        calibration_dataset_id = row.calibration_dataset_id
        s3_prefix = row.s3_prefix
        gpu_type = row.gpu_type or "NVIDIA H100 80GB HBM3"
        gpu_count = row.gpu_count or 1
        visible_devices = row.visible_devices

    await _push_log(redis, job_id, f"[gateway] starting quantization job {job_id}")

    # ---- build the worker config (creds injected, never persisted) ----
    try:
        genv = await ta._resolve_global_env()

        def g(k: str) -> Optional[str]:
            v = genv.get(k)
            return v if v not in (None, "") else (os.environ.get(k) or None)

        # Per-job HF token (gated/private source model + push to Hub).
        _job_hf_token: Optional[str] = None
        _hf_sec = (cfg.get("hf_token_secret") or "").strip()
        if _hf_sec:
            _job_hf_token = g(_hf_sec)
        elif cfg.get("hf_token_enc"):
            try:
                _job_hf_token = json.loads(crypto.decrypt(cfg["hf_token_enc"])).get("token")
            except Exception:  # noqa: BLE001
                _job_hf_token = None
        _hf_tok = _job_hf_token or g("HF_TOKEN")
        if _hf_tok:
            cfg["hf_token"] = _hf_tok

        # Calibration dataset (calibrated schemes only). Reuse the training dataset
        # resolver — it inlines creds + HF token so the box can load rows itself.
        if calibration_dataset_id:
            cfg["dataset"] = await ta._resolve_dataset_spec(calibration_dataset_id, _hf_tok)

        target = await _quant_s3_target(storage_id)
        cfg["artifacts"] = {
            "bucket": target.bucket, "region": target.region, "endpoint": target.endpoint,
            "access_key": target.access_key, "secret_key": target.secret_key,
            "prefix": s3_prefix.rstrip("/"),
        }
    except Exception as e:  # noqa: BLE001
        await _push_log(redis, job_id, f"[gateway] config resolve failed: {e}")
        await _finalize(job_id, "failed", None, f"config resolve failed: {e}")
        return

    # ---- resolve where to run (VM provider or a fresh RunPod pod) ----
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
                if not (visible_devices or "").strip():
                    vm_n = len(pcfg.get("gpus") or []) or int(pcfg.get("gpu_count") or 0)
                    if vm_n > 0:
                        gpu_count = vm_n

        if not is_vm:
            api_key = await compute._resolve_api_key(provider_id)
            key_filename, pub = ta._gen_ssh_key(work)
            await _push_log(redis, job_id,
                            f"[gateway] provisioning RunPod pod ({gpu_type} x{gpu_count}) …")
            runpod_id, host, port, cost = await ta._provision_pod(
                api_key, f"sgpu-quant-{job_id}", cfg.get("image") or DEFAULT_IMAGE,
                gpu_type, gpu_count, bool(cfg.get("secure_cloud", True)),
                int(cfg.get("disk_gb", 60)), int(cfg.get("volume_gb", 120)), pub,
                data_center_id=cfg.get("data_center_id"),
            )
            user = "root"
            _RUN_STATE[job_id] = {"runpod_id": runpod_id, "api_key": api_key}
            async with session_factory()() as s:
                r2 = await s.get(QuantizationJob, job_id)
                if r2 is not None:
                    r2.runpod_pod_id = runpod_id
                    r2.cost_per_hr = cost
                    await s.commit()
            await _push_log(redis, job_id, f"[gateway] pod {runpod_id} ready at {host}:{port}")
    except Exception as e:  # noqa: BLE001
        await _push_log(redis, job_id, f"[gateway] provisioning failed: {e}")
        if runpod_id and api_key:
            await ta._terminate_pod(api_key, runpod_id)
        await _finalize(job_id, "failed", None, f"provisioning failed: {e}")
        _RUN_STATE.pop(job_id, None)
        return

    # ---- ship + run the worker over SSH, streaming stdout ----
    result: dict = {"artifact": None, "hf_repo": None, "sizes": None, "progress": None}
    line_buf: list[str] = []

    def on_line(line: str) -> None:
        line_buf.append(line)
        try:
            with open(_full_log_path(job_id), "a") as f:
                f.write(line + "\n")
        except Exception:
            pass
        for tag, key in (("@@ARTIFACT ", "artifact"), ("@@SIZES ", "sizes"),
                         ("@@PROGRESS ", "progress"), ("@@DONE ", "_done"), ("@@ERROR ", "_error")):
            if line.startswith(tag):
                try:
                    payload = json.loads(line[len(tag):])
                except Exception:
                    return
                if key == "artifact":
                    result["artifact"] = payload.get("s3_uri")
                    result["hf_repo"] = payload.get("hf_repo")
                elif key == "sizes":
                    result["sizes"] = payload
                elif key == "progress":
                    result["progress"] = payload
                elif key == "_error":
                    result["error"] = payload.get("message") or "quantization failed"
                return

    sent = {"n": 0}

    async def pump() -> None:
        while True:
            while sent["n"] < len(line_buf):
                await _push_redis(redis, job_id, line_buf[sent["n"]])
                sent["n"] += 1
            await asyncio.sleep(0.5)

    async def result_flusher() -> None:
        while True:
            await asyncio.sleep(8)
            await _flush_result(job_id, result)

    pump_task = asyncio.create_task(pump())
    flush_task = asyncio.create_task(result_flusher())
    rc = 1
    try:
        cfg["gpu_count"] = gpu_count
        venv_path = (cfg.get("venv_path") or "/share/quant-llmcompressor").rstrip("/")
        cfg["venv_path"] = venv_path
        venv_py = shlex.quote(f"{venv_path}/bin/python")
        cfg_path = work / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        cli = await asyncio.to_thread(ta._ssh_connect, host, int(port), user, key_filename)
        try:
            stage = f"/tmp/sgpu_qrun_{job_id}"
            remote_cfg = f"{stage}/config.json"
            await asyncio.to_thread(ta._ssh_put, cli, str(cfg_path), remote_cfg)
            await asyncio.to_thread(ta._ssh_put, cli, str(_worker_script_path()), f"{stage}/quantize.py")
            worker_remote = f"{stage}/quantize.py"
            user_env = ta._render_env_exports(cfg.get("env_vars") or {})
            # --- deps phase: create/reuse the isolated uv venv (system python) ---
            deps_cmd = f"{user_env}{ta._UV_BOOTSTRAP}python -u {worker_remote} --deps-only --config {remote_cfg}"
            await _push_log(redis, job_id, f"[gateway] $ {deps_cmd}")
            drc = await asyncio.to_thread(ta._ssh_run_stream, cli, deps_cmd, on_line)
            if drc != 0:
                raise RuntimeError(f"dependency setup (uv venv {venv_path}) failed (rc={drc})")
            # --- run phase: launch the worker from the venv python, detached ---
            env_prefix = f"CUDA_VISIBLE_DEVICES={visible_devices} " if visible_devices else ""
            cmd = f"{user_env}{env_prefix}{venv_py} -u {worker_remote} --config {remote_cfg}"
            await _push_log(redis, job_id, f"[gateway] $ {cmd}")
            rlog, rpid, rsh = _remote_run_paths(job_id)
            run_sh = (f"#!/bin/bash\necho $$ > {rpid}\n{cmd}\nRC=$?\n"
                      f"echo \"@@RC:$RC\"\nrm -rf {stage} 2>/dev/null || true\n")
            await asyncio.to_thread(ta._ssh_put_bytes, cli, run_sh.encode(), rsh)
            await asyncio.to_thread(
                ta._ssh_exec, cli,
                f"rm -f {rlog} {rpid}; setsid bash {rsh} > {rlog} 2>&1 </dev/null &")
            await _push_log(redis, job_id, "[gateway] worker detached (survives gateway restart) — tailing log")
            _rc_box = {"rc": None}

            def _cap(l: str) -> None:
                m = re.match(r"\s*@@RC:(-?\d+)", l)
                if m:
                    _rc_box["rc"] = int(m.group(1))
                    return
                on_line(l)

            _stream = (f"tail -n +1 -F {rlog} & T=$!; "
                       f'while :; do P=$(cat {rpid} 2>/dev/null); '
                       f'if [ -n "$P" ] && ! kill -0 "$P" 2>/dev/null; then break; fi; sleep 2; done; '
                       f"sleep 2; kill $T 2>/dev/null")
            await asyncio.to_thread(ta._ssh_run_stream, cli, _stream, _cap)
            rc = _rc_box["rc"] if _rc_box["rc"] is not None else 1
        finally:
            try:
                cli.close()
            except Exception:
                pass
    except Exception as e:  # noqa: BLE001
        await _push_log(redis, job_id, f"[gateway] run failed: {e}")
        result.setdefault("error", str(e))
    finally:
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
        while sent["n"] < len(line_buf):
            await _push_redis(redis, job_id, line_buf[sent["n"]])
            sent["n"] += 1

    # ---- upload logs, finalize, teardown ----
    try:
        full = _full_log_path(job_id)
        if full.exists():
            s3_put_text(s3_prefix + "logs.txt", full.read_text(errors="replace"), target=target)
    except Exception as e:  # noqa: BLE001
        logger.warning("quantization %s: logs upload failed: %s", job_id, e)

    _ok = (rc == 0) or bool(result.get("artifact"))
    status = "done" if (_ok and not result.get("error")) else "failed"
    err = result.get("error") if status == "failed" else None
    if status == "failed" and not err:
        err = f"worker exited with code {rc}"
    await _finalize(job_id, status, rc, err, result_json=result)
    await _push_log(redis, job_id, f"[gateway] job {status} (rc={rc})")

    if runpod_id and api_key:
        await ta._terminate_pod(api_key, runpod_id)
        await _push_log(redis, job_id, f"[gateway] pod {runpod_id} torn down")
    elif is_vm and host and key_filename:
        try:
            await ta._cleanup_remote_run_files((host, port, user, key_filename), job_id)
        except Exception:
            pass
        # The autotrain cleanup only sweeps sgpu_train_* / sgpu_run_* — sweep our own
        # detached files + stage too.
        try:
            def _sweep() -> None:
                c = ta._ssh_connect(host, int(port), user, key_filename)
                try:
                    rlog, rpid, rsh = _remote_run_paths(job_id)
                    ta._ssh_exec(c, f"rm -rf {rlog} {rpid} {rsh} /tmp/sgpu_qrun_{job_id} 2>/dev/null || true")
                finally:
                    c.close()
            await asyncio.to_thread(_sweep)
        except Exception:
            pass
    _RUN_STATE.pop(job_id, None)
    try:
        await redis.expire(f"quant:logs:{job_id}", LOG_LIST_TTL_S)
    except Exception:
        pass


async def cleanup_orphaned_running(redis) -> int:
    """On gateway startup, fail any job left 'queued'/'running' by a previous
    process. Quant jobs are short (minutes) and detached workers aren't reconciled
    from their log (unlike autotrain) — simplest correct behavior is to mark them
    failed so the UI isn't stuck; the user can re-run. Returns the count."""
    n = 0
    orphan_pods: list[tuple[str, str]] = []  # (provider_id, runpod_pod_id)
    async with session_factory()() as s:
        rows = (await s.execute(
            select(QuantizationJob).where(QuantizationJob.status.in_(("queued", "running")))
        )).scalars().all()
        for row in rows:
            row.status = "failed"
            row.error_text = "gateway restarted while this job was in flight — re-run it"
            row.ended_at = datetime.now(timezone.utc)
            if row.runpod_pod_id and row.provider_id:
                orphan_pods.append((row.provider_id, row.runpod_pod_id))
            n += 1
        if n:
            await s.commit()
    # These jobs are failed but their cloud pods (if any) keep billing until torn
    # down. Best-effort terminate, off the session (the HTTP call mustn't hold a
    # pooled connection during startup reconcile).
    for prov_id, pod_id in orphan_pods:
        try:
            api_key = await compute._resolve_api_key(prov_id)
            await ta._terminate_pod(api_key, pod_id)
            logger.info("quantization: torn down orphaned pod %s (billing stopped)", pod_id)
        except Exception as e:  # noqa: BLE001 — teardown is best-effort
            logger.warning("quantization: orphaned pod %s teardown failed: %s", pod_id, e)
    return n


# ---------- Pydantic models ----------------------------------------------


class CreateQuantizationJobRequest(BaseModel):
    name: str
    source_model: str
    scheme: str = "fp8-dynamic"
    calibration_dataset_id: Optional[str] = None
    # Calibration knobs (calibrated schemes).
    num_calibration_samples: int = 512
    max_seq_length: int = 2048
    # Text column (kind=hf/upload/s3) or chat-messages column (kind=llm) in the
    # calibration dataset. Left blank → the worker auto-detects a sensible default.
    calib_text_field: Optional[str] = None
    calib_messages_field: Optional[str] = None
    # Recipe knobs.
    ignore_layers: list[str] = ["lm_head"]
    smoothing_strength: float = 0.8      # SmoothQuant (w8a8-int8)
    dampening_frac: float = 0.01         # GPTQ
    # Push target.
    hf_push_repo: Optional[str] = None
    hf_push_private: bool = True
    hf_token: Optional[str] = None
    hf_token_secret: Optional[str] = None
    # Compute.
    provider_id: Optional[str] = None
    storage_id: Optional[str] = None
    gpu_type: str = "NVIDIA H100 80GB HBM3"
    gpu_count: int = 1
    visible_devices: Optional[str] = None
    # RunPod-only knobs.
    secure_cloud: bool = True
    data_center_id: Optional[str] = None
    disk_gb: int = 60
    volume_gb: int = 120
    image: Optional[str] = None
    work_dir: Optional[str] = None
    venv_path: Optional[str] = None
    env_vars: Optional[dict] = None

    @field_validator("work_dir", "venv_path")
    @classmethod
    def _safe_paths(cls, v, info):  # noqa: N805
        # These reach remote shell commands on the quant box/pod.
        return validate_path_field(v, info.field_name)


class QuantizationJobRecord(BaseModel):
    id: str
    name: str
    status: str
    source_model: str
    scheme: str
    calibration_dataset_id: Optional[str] = None
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
    provider_kind: Optional[str] = None
    storage_id: Optional[str] = None
    storage_name: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_count: int = 1
    visible_devices: Optional[str] = None


class QuantizationJobPageResponse(BaseModel):
    total: int
    items: list[QuantizationJobRecord]


class QuantizationSchemesResponse(BaseModel):
    # id → {label, needs_calibration}; drives the form dropdown.
    schemes: dict[str, dict]
    calib_dataset_kinds: list[str]


# ---------- HTTP API -----------------------------------------------------


router = APIRouter(prefix="/v1/quantization-jobs", tags=["quantization"])


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _to_record(
    row: QuantizationJob, owner_username: str,
    provider_name: Optional[str] = None, provider_kind: Optional[str] = None,
    storage_name: Optional[str] = None,
) -> QuantizationJobRecord:
    return QuantizationJobRecord(
        id=row.id, name=row.name, status=row.status, source_model=row.source_model,
        scheme=row.scheme, calibration_dataset_id=row.calibration_dataset_id,
        s3_prefix=row.s3_prefix, config_json=row.config_json or {},
        exit_code=row.exit_code, error_text=row.error_text, result_json=row.result_json,
        created_by=owner_username, created_at=_iso(row.created_at) or "",
        started_at=_iso(row.started_at), ended_at=_iso(row.ended_at),
        cost_per_hr=row.cost_per_hr, provider_id=row.provider_id,
        provider_name=provider_name, provider_kind=provider_kind,
        storage_id=row.storage_id, storage_name=storage_name,
        gpu_type=row.gpu_type, gpu_count=row.gpu_count, visible_devices=row.visible_devices,
    )


@router.get("/schemes", response_model=QuantizationSchemesResponse)
async def list_schemes(user: User = Depends(require_section("quantization"))):
    return QuantizationSchemesResponse(
        schemes={k: {"label": lbl, "needs_calibration": calib} for k, (lbl, calib) in _SCHEMES.items()},
        calib_dataset_kinds=list(_CALIB_DATASET_KINDS),
    )


@router.post("", response_model=QuantizationJobRecord)
async def create_quantization_job(
    body: CreateQuantizationJobRequest,
    request: Request,
    user: User = Depends(require_section("quantization")),
    session: AsyncSession = Depends(get_session),
):
    if body.scheme not in _SCHEMES:
        raise HTTPException(status_code=400, detail=f"unknown scheme '{body.scheme}'")
    if not (body.source_model or "").strip():
        raise HTTPException(status_code=400, detail="source_model is required")
    needs_calib = _SCHEMES[body.scheme][1]
    ds = None
    if body.calibration_dataset_id:
        ds = await session.get(Dataset, body.calibration_dataset_id)
        if ds is None or (ds.owner_id != user.id and not user.is_admin):
            raise HTTPException(status_code=400, detail="unknown calibration_dataset_id")
        if ds.kind not in _CALIB_DATASET_KINDS:
            raise HTTPException(
                status_code=400,
                detail=(f"calibration dataset must be a text dataset "
                        f"({', '.join(_CALIB_DATASET_KINDS)}); got kind={ds.kind}"),
            )
    if needs_calib and not body.calibration_dataset_id:
        raise HTTPException(
            status_code=400,
            detail=f"scheme '{body.scheme}' needs a calibration dataset — pick one under Datasets",
        )
    if body.storage_id:
        st = await session.get(Storage, body.storage_id)
        if st is None or st.kind != "s3" or not st.enabled:
            raise HTTPException(status_code=400, detail="storage must be an enabled s3 backend")

    # Compute target: VM hardware is fixed by the box, so drop cloud-only knobs.
    eff_gpu_type = body.gpu_type
    is_vm_run = False
    vm_gpus: list = []
    if body.provider_id:
        prov = await session.get(Provider, body.provider_id)
        if prov is None:
            raise HTTPException(status_code=400, detail="unknown provider_id")
        # Providers are per-user — enforce ownership (admins exempt) so a job can't
        # bill another tenant's cloud account or run on their VM.
        if prov.owner_id != user.id and not user.is_admin:
            raise HTTPException(status_code=403, detail="that provider isn't yours")
        if prov.kind == "vm":
            is_vm_run = True
            vm_gpus = (prov.config or {}).get("gpus") or []
            if vm_gpus:
                eff_gpu_type = vm_gpus[0]
    gpu_bound = (len(vm_gpus) or int((prov.config or {}).get("gpu_count") or 0)) if is_vm_run else body.gpu_count
    pinned_ids = ta._parse_gpu_indices(body.visible_devices)
    if gpu_bound and pinned_ids:
        oob = sorted({i for i in pinned_ids if i >= gpu_bound})
        if oob:
            raise HTTPException(
                status_code=400,
                detail=f"visible_devices out of range: {oob} — target has {gpu_bound} GPU(s)",
            )

    job_id = _gen_id()
    target = await _quant_s3_target(body.storage_id)
    s3_prefix = f"{target.prefix_root}{job_id}/"
    config = {
        "source_model": body.source_model.strip(),
        "scheme": body.scheme,
        "num_calibration_samples": max(1, int(body.num_calibration_samples)),
        "max_seq_length": max(128, int(body.max_seq_length)),
        "calib_text_field": (body.calib_text_field or "").strip() or None,
        "calib_messages_field": (body.calib_messages_field or "").strip() or None,
        "ignore_layers": [str(x).strip() for x in (body.ignore_layers or []) if str(x).strip()] or ["lm_head"],
        "smoothing_strength": float(body.smoothing_strength),
        "dampening_frac": float(body.dampening_frac),
        "hf_push_repo": (body.hf_push_repo or "").strip() or None,
        "hf_push_private": bool(body.hf_push_private),
        "hf_token_secret": (body.hf_token_secret or "").strip() or None,
        "hf_token_enc": (
            crypto.encrypt(json.dumps({"token": body.hf_token.strip()}))
            if (body.hf_token and body.hf_token.strip() and not (body.hf_token_secret or "").strip())
            else None
        ),
        # Cloud-pod knobs are irrelevant on a VM.
        **({} if is_vm_run else {
            "secure_cloud": body.secure_cloud,
            "data_center_id": (body.data_center_id or "").strip() or None,
            "disk_gb": body.disk_gb, "volume_gb": body.volume_gb,
            "image": (body.image or "").strip() or None,
        }),
        "work_dir": (body.work_dir or "/share").strip() or "/share",
        "venv_path": (body.venv_path or "").strip() or None,
        "env_vars": {
            k: str(v) for k, v in (body.env_vars or {}).items()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(k))
        },
    }
    row = QuantizationJob(
        id=job_id, name=body.name.strip() or job_id, source_model=body.source_model.strip(),
        scheme=body.scheme, calibration_dataset_id=body.calibration_dataset_id,
        config_json=config, status="queued", s3_prefix=s3_prefix, owner_id=user.id,
        provider_id=body.provider_id, storage_id=body.storage_id,
        gpu_type=eff_gpu_type,
        gpu_count=((len(pinned_ids) or gpu_bound or body.gpu_count) if is_vm_run else body.gpu_count),
        visible_devices=body.visible_devices,
    )
    session.add(row)
    await session.commit()

    redis = request.app.state.redis
    task = asyncio.create_task(_safe_run(redis, job_id))
    _active_runners[job_id] = task
    task.add_done_callback(lambda _t: _active_runners.pop(job_id, None))
    return _to_record(row, user.username)


@router.get("", response_model=list[QuantizationJobRecord])
async def list_quantization_jobs(
    scope: str = "mine",
    user: User = Depends(require_section("quantization")),
    session: AsyncSession = Depends(get_session),
):
    q = select(QuantizationJob).order_by(QuantizationJob.created_at.desc())
    if not (scope == "all" and user.is_admin):
        q = q.where(QuantizationJob.owner_id == user.id)
    rows = (await session.execute(q)).scalars().all()
    names: dict[int, str] = {}
    for row in rows:
        if row.owner_id not in names:
            u = await session.get(User, row.owner_id)
            names[row.owner_id] = u.username if u else "?"
    return [_to_record(r, names.get(r.owner_id, "?")) for r in rows]


@router.get("/_page", response_model=QuantizationJobPageResponse)
async def list_quantization_jobs_page(
    scope: str = "mine",
    q: str = "",
    status: str = "",
    sort: str = "newest",
    limit: int = Query(12, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(require_section("quantization")),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(QuantizationJob)
    if not (scope == "all" and user.is_admin):
        stmt = stmt.where(QuantizationJob.owner_id == user.id)
    if status:
        stmt = stmt.where(QuantizationJob.status == status)
    for tok in (q or "").lower().split():
        like = f"%{tok}%"
        stmt = stmt.where(
            or_(
                QuantizationJob.id.ilike(like),
                QuantizationJob.name.ilike(like),
                QuantizationJob.status.ilike(like),
                QuantizationJob.source_model.ilike(like),
                QuantizationJob.scheme.ilike(like),
                cast(QuantizationJob.config_json, Text).ilike(like),
                QuantizationJob.owner_id.in_(select(User.id).where(User.username.ilike(like))),
            )
        )
    total = (await session.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    order = QuantizationJob.created_at.asc() if sort == "oldest" else QuantizationJob.created_at.desc()
    rows = (await session.execute(stmt.order_by(order).limit(limit).offset(offset))).scalars().all()
    names: dict[int, str] = {}
    if rows:
        urows = await session.execute(
            select(User.id, User.username).where(User.id.in_({r.owner_id for r in rows}))
        )
        names = {uid: uname for uid, uname in urows.all()}
    items = [_to_record(r, names.get(r.owner_id, "?")) for r in rows]
    return QuantizationJobPageResponse(total=total, items=items)


async def _owned(job_id: str, user: User, session: AsyncSession) -> QuantizationJob:
    row = await session.get(QuantizationJob, job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    if row.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="not yours")
    return row


@router.get("/{job_id}", response_model=QuantizationJobRecord)
async def get_quantization_job(
    job_id: str,
    user: User = Depends(require_section("quantization")),
    session: AsyncSession = Depends(get_session),
):
    row = await _owned(job_id, user, session)
    u = await session.get(User, row.owner_id)
    prov = await session.get(Provider, row.provider_id) if row.provider_id else None
    store = await session.get(Storage, row.storage_id) if row.storage_id else None
    return _to_record(
        row, u.username if u else "?",
        provider_name=prov.name if prov else None,
        provider_kind=prov.kind if prov else None,
        storage_name=store.name if store else None,
    )


class RenameJobRequest(BaseModel):
    name: str


@router.patch("/{job_id}", response_model=QuantizationJobRecord)
async def rename_quantization_job(
    job_id: str,
    body: RenameJobRequest,
    user: User = Depends(require_section("quantization")),
    session: AsyncSession = Depends(get_session),
):
    row = await _owned(job_id, user, session)
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")
    row.name = name[:128]
    await session.commit()
    u = await session.get(User, row.owner_id)
    return _to_record(row, u.username if u else "?")


@router.delete("/{job_id}")
async def delete_quantization_job(
    job_id: str,
    user: User = Depends(require_section("quantization")),
    session: AsyncSession = Depends(get_session),
):
    row = await _owned(job_id, user, session)
    t = _active_runners.get(job_id)
    if t:
        t.cancel()
    st = _RUN_STATE.pop(job_id, None)
    if st:
        await ta._terminate_pod(st["api_key"], st["runpod_id"])
    await session.delete(row)
    await session.commit()
    return {"ok": True, "id": job_id}


@router.post("/{job_id}/terminate", response_model=QuantizationJobRecord)
async def terminate_quantization_job(
    job_id: str,
    user: User = Depends(require_section("quantization")),
    session: AsyncSession = Depends(get_session),
):
    row = await _owned(job_id, user, session)
    if row.status in ("done", "failed", "cancelled"):
        raise HTTPException(status_code=409, detail=f"already {row.status}")
    t = _active_runners.get(job_id)
    if t:
        t.cancel()
    st = _RUN_STATE.pop(job_id, None)
    if st:
        await ta._terminate_pod(st["api_key"], st["runpod_id"])
    row.status = "cancelled"
    row.ended_at = datetime.now(timezone.utc)
    await session.commit()
    u = await session.get(User, row.owner_id)
    return _to_record(row, u.username if u else "?")


@router.get("/{job_id}/files", response_model=list[ta.TrainingFile])
async def list_quantization_files(
    job_id: str,
    user: User = Depends(require_section("quantization")),
    session: AsyncSession = Depends(get_session),
):
    """List the job's S3 files (compressed model shards, logs.txt) with presigned
    download URLs. Mirrors autotrain's /files."""
    row = await _owned(job_id, user, session)
    target = await _quant_s3_target(row.storage_id)
    out: list[ta.TrainingFile] = []
    try:
        for obj in s3_list(row.s3_prefix, target=target):
            name = obj["key"][len(row.s3_prefix):] if obj["key"].startswith(row.s3_prefix) else obj["key"]
            out.append(ta.TrainingFile(
                name=name, size=obj.get("size", 0),
                modified=obj.get("modified", ""),
                download_url=s3_presign_get(obj["key"], target=target),
            ))
    except Exception as e:  # noqa: BLE001
        logger.warning("quantization %s: file list failed: %s", job_id, e)
    return out


@router.get("/{job_id}/logs")
async def get_quantization_logs(
    job_id: str,
    tail: int = 400,
    request: Request = None,
    user: User = Depends(require_section("quantization")),
    session: AsyncSession = Depends(get_session),
):
    row = await _owned(job_id, user, session)
    redis = request.app.state.redis
    lines: list[str] = []
    try:
        raw = await redis.lrange(f"quant:logs:{job_id}", -int(tail), -1)
        lines = [b.decode("utf-8", "replace") if isinstance(b, bytes) else str(b) for b in raw]
    except Exception:
        pass
    if not lines:
        full = _full_log_path(job_id)
        if full.exists():
            lines = full.read_text(errors="replace").splitlines()[-int(tail):]
    return {"status": row.status, "error_text": row.error_text, "lines": lines}


@router.get("/{job_id}/logs/stream")
async def stream_quantization_logs(
    job_id: str,
    request: Request,
    user: User = Depends(require_section("quantization")),
    session: AsyncSession = Depends(get_session),
):
    await _owned(job_id, user, session)
    redis = request.app.state.redis

    async def gen() -> AsyncIterator[str]:
        key = f"quant:logs:{job_id}"
        sent = 0
        while True:
            try:
                raw = await redis.lrange(key, sent, -1)
            except Exception:
                raw = []
            for b in raw:
                line = b.decode("utf-8", "replace") if isinstance(b, bytes) else str(b)
                line = line.replace("\r", "")
                yield f"data: {line}\n\n"
                sent += 1
            async with session_factory()() as s:
                cur = await s.get(QuantizationJob, job_id)
            rj = (cur.result_json or {}) if cur is not None else {}
            exporting = (rj.get("hf_export") or {}).get("status") == "running"
            if cur is None or (cur.status in ("done", "failed", "cancelled") and not exporting):
                yield "event: end\ndata: end\n\n"
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------- on-demand HF export (gateway-local, no GPU) ------------------


async def _set_hf_export_state(job_id: str, state: dict) -> None:
    async with session_factory()() as s:
        row = await s.get(QuantizationJob, job_id)
        if row is None:
            return
        rj = dict(row.result_json or {})
        rj["hf_export"] = state
        if state.get("status") == "done" and state.get("repo"):
            rj["hf_repo"] = state["repo"]
        row.result_json = rj
        await s.commit()


class HfExportRequest(BaseModel):
    repo: str
    private: bool = True
    storage_id: Optional[str] = None
    token: Optional[str] = None
    # Where the push runs: "gateway" (default — fetch from S3 here, box-independent)
    # or "vm" (push from the job's VM; needs that box + the quant venv on it).
    run_on: str = "gateway"


def _hf_push_local(model_s3: str, s3creds: dict, repo: str, token: str, private: bool,
                   hf_endpoint: Optional[str] = None) -> str:
    """Download the quantized model dir from S3 to a temp dir and push it to HF
    (or a custom Hub — the self-hosted mirror — via `hf_endpoint`). Runs on the
    gateway (no GPU) — the quantized model is a plain compressed-tensors folder.
    Returns the repo URL. Blocking → call via asyncio.to_thread."""
    import tempfile
    import boto3
    from botocore.client import Config as BotoConfig
    from huggingface_hub import HfApi

    # The self-hosted mirror speaks LFS but NOT Xet — a modern huggingface_hub
    # otherwise probes `{endpoint}/api/models/{repo}/xet-write-token/{rev}`, 404s,
    # and aborts the push. huggingface_hub may already be imported in this process
    # (constants fixed at import time), so patch the live module too — the same
    # belt-and-suspenders as dataset_transform (see its HF_HUB_DISABLE_XET note).
    if hf_endpoint:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        try:
            import huggingface_hub.constants as _hfc
            _hfc.HF_HUB_DISABLE_XET = True
        except Exception:  # noqa: BLE001 — best-effort; the env var still applies
            pass

    assert model_s3.startswith("s3://"), f"bad model_s3: {model_s3}"
    bucket, _, prefix = model_s3[len("s3://"):].partition("/")
    prefix = prefix.rstrip("/") + "/"
    cli = boto3.client(
        "s3", region_name=s3creds.get("region") or "us-east-1",
        endpoint_url=s3creds.get("endpoint") or None,
        aws_access_key_id=s3creds.get("access_key") or None,
        aws_secret_access_key=s3creds.get("secret_key") or None,
        config=BotoConfig(signature_version="s3v4"),
    )
    dest = tempfile.mkdtemp(prefix="sgpu-quant-hf-")
    paginator = cli.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel:
                continue
            fp = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            cli.download_file(bucket, key, fp)
    api = HfApi(token=token, endpoint=hf_endpoint or None)
    api.create_repo(repo, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=dest, repo_id=repo, repo_type="model")
    return f"{hf_endpoint}/{repo}" if hf_endpoint else f"https://huggingface.co/{repo}"


@router.post("/{job_id}/hf-export", response_model=QuantizationJobRecord)
async def export_to_huggingface(
    job_id: str,
    body: HfExportRequest,
    request: Request,
    user: User = Depends(require_section("quantization")),
    session: AsyncSession = Depends(get_session),
):
    row = await _owned(job_id, user, session)
    if row.status != "done":
        raise HTTPException(status_code=409, detail=f"job is {row.status}, not done")
    model_s3 = (row.result_json or {}).get("artifact")
    if not model_s3:
        raise HTTPException(status_code=400, detail="no quantized model artifact to export")
    repo = (body.repo or "").strip()
    if "/" not in repo:
        raise HTTPException(status_code=400, detail="repo must be 'owner/name'")

    # HF token: explicit body → storage (secret reference or encrypted token, via the
    # shared autotrain helper) → global HF_TOKEN. The storage may also carry a custom
    # Hub endpoint (the self-hosted mirror).
    token = (body.token or "").strip() or None
    if not token and body.storage_id:
        token = await ta._hf_token_for_storage(body.storage_id, session)
    hf_endpoint = await ta._hf_endpoint_for_storage(body.storage_id, session) if body.storage_id else None
    if not token:
        genv = await ta._resolve_global_env()
        token = genv.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise HTTPException(status_code=400, detail="no HF token — pass one, pick an HF storage, or set HF_TOKEN")

    s3creds = ta._s3_creds_from_storage(await session.get(Storage, row.storage_id) if row.storage_id else None)
    redis = request.app.state.redis

    # "vm": push from the job's VM via the shared hf_export.py shipper (needs the box
    # + the quant venv still there). Resolve SSH coords up front for a clear 400.
    run_on = (body.run_on or "gateway").strip().lower()
    vm_ssh: Optional[tuple] = None
    if run_on == "vm":
        prov = await session.get(Provider, row.provider_id) if row.provider_id else None
        if prov is None or prov.kind != "vm":
            raise HTTPException(status_code=400, detail="this job didn't run on a VM — push from the gateway instead")
        # A custom endpoint pointing at THIS gateway's own /hf mirror is unreachable
        # from the box without a reverse tunnel — push from the gateway instead.
        if hf_endpoint and ta._loopback_endpoint(hf_endpoint) != hf_endpoint:
            raise HTTPException(
                status_code=400,
                detail="this storage pushes to the gateway's own HF mirror — use Run on: Gateway",
            )
        pcfg = prov.config or {}
        enc = pcfg.get("private_key_enc")
        if not enc:
            raise HTTPException(status_code=400, detail="VM provider has no stored private key")
        key_filename = str(_work_dir(job_id) / "vm_key")
        Path(key_filename).write_text(crypto.decrypt(enc))
        os.chmod(key_filename, 0o600)
        vm_ssh = (pcfg.get("host"), int(pcfg.get("port") or 22), pcfg.get("user") or "root", key_filename)

    prev = _active_hf_exports.get(job_id)
    if prev:
        prev.cancel()

    # cfg for the VM shipper: the quant venv has boto3 + huggingface_hub. venv_path
    # isn't persisted to config_json (mirrors autotrain), so re-derive the default.
    job_cfg = dict(row.config_json or {})
    job_cfg["venv_path"] = (job_cfg.get("venv_path") or "/share/quant-llmcompressor")
    loop = asyncio.get_running_loop()

    def _sink(line: str) -> None:
        # Stream the VM-side download/upload lines into the job log (thread → loop).
        asyncio.run_coroutine_threadsafe(_push_log(redis, job_id, line), loop)

    async def _bg() -> None:
        await _set_hf_export_state(job_id, {"status": "running", "repo": repo})
        where = "from the job's VM" if vm_ssh else "from the gateway"
        await _push_log(redis, job_id, f"[gateway] exporting quantized model {where} → https://huggingface.co/{repo} …")
        try:
            if vm_ssh:
                res = await asyncio.to_thread(
                    ta._run_hf_export_ssh, *vm_ssh, job_id, model_s3, s3creds,
                    repo, token, body.private, job_cfg, hf_endpoint, _sink,
                )
                url = res.get("url") or f"https://huggingface.co/{repo}"
            else:
                # Own-mirror endpoints are rewritten to loopback so the upload
                # bypasses the ingress (mirrors autotrain's local export). The
                # stored URL keeps the ORIGINAL endpoint (loopback is useless to
                # a browser).
                url = await asyncio.to_thread(
                    _hf_push_local, model_s3, s3creds, repo, token, body.private,
                    ta._loopback_endpoint(hf_endpoint),
                )
                if hf_endpoint:
                    url = f"{hf_endpoint}/{repo}"
            await _set_hf_export_state(job_id, {"status": "done", "repo": repo, "url": url})
            await _push_log(redis, job_id, f"[gateway] pushed → {url}")
        except Exception as e:  # noqa: BLE001
            await _set_hf_export_state(job_id, {"status": "failed", "repo": repo, "error": str(e)})
            await _push_log(redis, job_id, f"[gateway] HF export failed: {e}")
        finally:
            _active_hf_exports.pop(job_id, None)

    task = asyncio.create_task(_bg())
    _active_hf_exports[job_id] = task
    await session.refresh(row)
    u = await session.get(User, row.owner_id)
    return _to_record(row, u.username if u else "?")


@router.post("/{job_id}/hf-export/cancel", response_model=QuantizationJobRecord)
async def cancel_huggingface_export(
    job_id: str,
    request: Request,
    user: User = Depends(require_section("quantization")),
    session: AsyncSession = Depends(get_session),
):
    """Stop a running HF export. Gateway pushes run in-process (cancel the task,
    done); VM pushes leave a box-side hf_export.py process — best-effort pkill it
    by its uniquely-named script (mirrors autotrain's cancel)."""
    row = await _owned(job_id, user, session)
    t = _active_hf_exports.pop(job_id, None)
    if t is not None and not t.done():
        t.cancel()
    # Best-effort: kill a VM-side push (run_on="vm") so it can't keep uploading.
    prov = await session.get(Provider, row.provider_id) if row.provider_id else None
    if prov is not None and prov.kind == "vm":
        pcfg = prov.config or {}
        enc = pcfg.get("private_key_enc")
        if enc:
            try:
                key_filename = str(_work_dir(job_id) / "vm_key")
                Path(key_filename).write_text(crypto.decrypt(enc))
                os.chmod(key_filename, 0o600)
                host, port_, suser = pcfg.get("host"), int(pcfg.get("port") or 22), pcfg.get("user") or "root"

                def _kill() -> None:
                    cli = ta._ssh_connect(host, port_, suser, key_filename)
                    try:
                        ta._ssh_exec(cli, f"pkill -9 -f sgpu_hf_export_{job_id} 2>/dev/null; "
                                          f"rm -f /tmp/sgpu_hf_export_{job_id}* 2>/dev/null || true")
                    finally:
                        cli.close()

                await asyncio.to_thread(_kill)
            except Exception:  # noqa: BLE001
                pass
    rj = dict(row.result_json or {})
    hf = dict(rj.get("hf_export") or {})
    if hf.get("status") == "running":
        hf["status"] = "cancelled"
        hf.setdefault("error", "cancelled by user")
        rj["hf_export"] = hf
        row.result_json = rj
        await session.commit()
        await _push_log(request.app.state.redis, job_id, "[gateway] HF export cancelled")
    u = await session.get(User, row.owner_id)
    return _to_record(row, u.username if u else "?")
