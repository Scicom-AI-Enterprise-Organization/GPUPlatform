"""Benchmark feature — approach B (SSH-orchestrated via llm-benchmaq).

Gateway shells out to `benchmaq runpod bench config.yaml` as a subprocess,
streams stdout/stderr to a redis list (capped) for live SSE replay, and on
exit syncs any result files into S3 + parses them for the Metrics tab.

Subprocess lives only in the gateway process. If gateway dies mid-run the
bench is orphaned (pod is alive on RunPod but nobody's collecting). On
startup we mark all `running` rows as `failed` with a clear message — the
user can re-submit or terminate the dangling pod from RunPod's dashboard.
"""
from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import boto3
import httpx
import yaml
from botocore.client import Config as BotoConfig
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from . import audit
from .auth import require_section
from .db import Base, User, get_session, session_factory

logger = logging.getLogger("gateway.bench")

LOG_LIST_CAP = 5000          # max lines retained in redis per bench
LOG_LIST_TTL_S = 12_960_000  # ~5 months after benchmark completes


# ---------- DB model ----------------------------------------------------


class BenchmarkTemplate(Base):
    __tablename__ = "benchmark_templates"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    config_yaml: Mapped[str] = mapped_column(String(65535))
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class Benchmark(Base):
    __tablename__ = "benchmarks"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    config_yaml: Mapped[str] = mapped_column(String(65535))
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    s3_prefix: Mapped[str] = mapped_column(String(255))
    exit_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_text: Mapped[Optional[str]] = mapped_column(String(4096), nullable=True)
    result_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # RunPod $/hour quote at spawn time, captured by scraping the pod_id out of
    # benchmaq stdout then querying RunPod /pods/{id}. NULL while the pod isn't
    # up yet, and stays at the original quoted rate for the life of the run.
    cost_per_hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    runpod_pod_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # User-selected cloud provider. NULL = use platform default (RunPod via env).
    # FK omitted to keep this column nullable without cascade headaches; we
    # validate ownership at create time.
    provider_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    # VM runs only: SSH back in after benchmaq exits to rm -rf the model's
    # local_dir + HF hub cache. Default true so users don't fill the VM disk.
    cleanup_model: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)


# ---------- S3 ----------------------------------------------------------


def _bucket() -> str:
    b = os.environ.get("BENCHMARK_S3_BUCKET", "").strip()
    if not b:
        raise RuntimeError("BENCHMARK_S3_BUCKET not set")
    return b


def _s3_prefix_root() -> str:
    p = os.environ.get("BENCHMARK_S3_PREFIX", "benchmarks/").strip().lstrip("/")
    if not p.endswith("/"):
        p += "/"
    return p


def _aws_region() -> str:
    return os.environ.get("AWS_REGION", "ap-southeast-5")


def _s3_client():
    region = _aws_region()
    return boto3.client(
        "s3",
        region_name=region,
        # Pin to the regional endpoint. Default `s3.amazonaws.com` redirects
        # to the bucket's region, but presigned URLs signed with a non-default
        # region get a 400 on the global host before the redirect can happen.
        endpoint_url=f"https://s3.{region}.amazonaws.com",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


def s3_put_text(key: str, body: str) -> None:
    _s3_client().put_object(Bucket=_bucket(), Key=key, Body=body.encode("utf-8"))


def s3_put_file(key: str, path: str) -> None:
    with open(path, "rb") as f:
        _s3_client().put_object(Bucket=_bucket(), Key=key, Body=f.read())


def s3_get_text(key: str) -> Optional[str]:
    """Read an S3 object as utf-8 text. Returns None if the key is missing."""
    try:
        obj = _s3_client().get_object(Bucket=_bucket(), Key=key)
        return obj["Body"].read().decode("utf-8", "replace")
    except Exception:
        return None


def s3_list(prefix: str) -> list[dict]:
    cli = _s3_client()
    out: list[dict] = []
    token = None
    while True:
        kwargs = {"Bucket": _bucket(), "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        r = cli.list_objects_v2(**kwargs)
        for obj in r.get("Contents", []):
            out.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "modified": obj["LastModified"].isoformat() if obj.get("LastModified") else "",
            })
        if not r.get("IsTruncated"):
            break
        token = r.get("NextContinuationToken")
    return out


def s3_presign_get(key: str, expires: int = 3600) -> str:
    return _s3_client().generate_presigned_url(
        "get_object", Params={"Bucket": _bucket(), "Key": key}, ExpiresIn=expires
    )


# ---------- Helpers -----------------------------------------------------


def benchmark_s3_prefix(bench_id: str) -> str:
    return f"{_s3_prefix_root()}{bench_id}/"


def _gen_id() -> str:
    import uuid
    return f"bench-{uuid.uuid4().hex[:8]}"


def _work_dir(bench_id: str) -> Path:
    base = Path(os.environ.get("BENCHMARK_WORK_DIR", "/tmp/sgpu-bench"))
    p = base / bench_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ssh_key_path() -> str:
    p = os.environ.get("BENCHMARK_SSH_KEY_PATH", "").strip()
    if not p:
        # benchmaq's own default
        p = str(Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go")
    return os.path.expanduser(p)


def _resolve_config(
    raw_yaml: str,
    vm_target: Optional[dict] = None,
) -> str:
    """Inject runtime values (SSH key path, RunPod API key) into the user's YAML.

    Users paste a config that may have `ssh_private_key: "path/to/your/private/key"`
    or empty `runpod_api_key: ""`. We replace those with real values from env so
    they don't have to know about the runpodctl-managed key location.

    When `vm_target` is provided (bare-metal mode), we rewrite the `remote:`
    block to point at the user-registered VM and drop the `runpod:` block so
    benchmaq's vllm runner doesn't accidentally pick it up. Expected shape:
        {"host": str, "port": int, "user": str, "key_filename": str}
    """
    cfg = yaml.safe_load(raw_yaml) or {}
    if not isinstance(cfg, dict):
        return raw_yaml

    if vm_target is None:
        rp = cfg.setdefault("runpod", {})
        if not rp.get("ssh_private_key") or "path/to/your" in str(rp.get("ssh_private_key")):
            rp["ssh_private_key"] = _ssh_key_path()
        if not rp.get("runpod_api_key"):
            rp["runpod_api_key"] = os.environ.get("RUNPOD_API_KEY", "")

        rem = cfg.setdefault("remote", {})
        if not rem.get("key_filename") or "path/to/your" in str(rem.get("key_filename")):
            rem["key_filename"] = _ssh_key_path()
    else:
        # Bare-metal VM: drop runpod block (irrelevant + would confuse benchmaq)
        # and rewrite remote to use benchmaq's `backend: ssh` runner — a
        # paramiko-based path with proper live-streaming, idempotent
        # uv+benchmaq install on the VM, and zero dependency on pyremote.
        # benchmaq[vllm] pulls vLLM transitively via the extra; we pin the
        # vllm version explicitly when the user picked one in the form.
        cfg.pop("runpod", None)
        rem = cfg.setdefault("remote", {})
        rem["backend"] = "ssh"
        rem["host"] = vm_target["host"]
        rem["port"] = int(vm_target.get("port") or 22)
        rem["username"] = vm_target.get("user", "root")
        rem["key_filename"] = vm_target["key_filename"]
        uv = rem.setdefault("uv", {})
        uv.setdefault("path", "~/.bench-venv")
        uv.setdefault("python_version", "3.11")
        uv.setdefault(
            "benchmaq_ref",
            "git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq.git@75d1353",
        )
        # If the form rendered a vLLM pin under remote.dependencies (legacy
        # path), surface it as uv.vllm_version so the new ssh backend picks
        # it up. Otherwise leave unset = latest.
        if "vllm_version" not in uv:
            for dep in (rem.get("dependencies") or []):
                if isinstance(dep, str) and dep.startswith("vllm==") and len(dep) > 6:
                    uv["vllm_version"] = dep.split("==", 1)[1].strip()
                    break
        # The new backend installs benchmaq[vllm] + vllm itself; the legacy
        # `dependencies` field is unused here.
        rem.pop("dependencies", None)

        # `/workspace/...` is RunPod's per-pod mount and doesn't exist on
        # bare-metal VMs (where the SSH user is typically `ubuntu` and only
        # has write access under $HOME). Rewrite any model.local_dir or
        # results.result_dir that starts with `/workspace/` to live under
        # the user's home instead.
        items = cfg.get("benchmark") or []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                model = item.get("model")
                if isinstance(model, dict):
                    ld = str(model.get("local_dir") or "")
                    if ld.startswith("/workspace/"):
                        model["local_dir"] = "~/" + ld[len("/workspace/"):]
                results = item.get("results")
                if isinstance(results, dict):
                    rd = str(results.get("result_dir") or "")
                    if rd.startswith("/workspace/"):
                        results["result_dir"] = "~/" + rd[len("/workspace/"):]

    return yaml.safe_dump(cfg, sort_keys=False)


def _pick_engine_subcommand(raw_yaml: str) -> list[str]:
    """Read engine from the first benchmark item; default to vllm.

    benchmaq separates engines into top-level subcommands (`vllm bench`,
    `sglang bench`) and only those honour the `remote:` block. The `runpod`
    subcommand is only for the spawn-a-pod-then-bench flow we use for the
    cloud target.
    """
    try:
        cfg = yaml.safe_load(raw_yaml) or {}
        items = cfg.get("benchmark") or []
        if items and isinstance(items, list):
            engine = str(items[0].get("engine") or "vllm").lower()
            if engine == "sglang":
                return ["sglang", "bench"]
    except Exception:
        pass
    return ["vllm", "bench"]


async def _materialise_vm_key(work_dir: Path, provider_id: str) -> dict:
    """Look up the provider, decrypt its private key, write to `work/vm_key`
    with 0600, and return the dict `_resolve_config` expects as `vm_target`.

    Raises if the provider is missing or the key can't be decrypted.
    """
    from . import crypto
    from .db import Provider
    async with session_factory()() as s:
        prov = await s.get(Provider, provider_id)
    if prov is None:
        raise RuntimeError(f"provider {provider_id} not found")
    if prov.kind != "vm":
        raise RuntimeError(f"provider {provider_id} is kind={prov.kind}, expected vm")
    cfg = prov.config or {}
    enc = cfg.get("private_key_enc")
    if not enc:
        raise RuntimeError(f"provider {provider_id} has no stored key")
    pk_text = crypto.decrypt(enc)
    key_path = work_dir / "vm_key"
    key_path.write_text(pk_text + ("\n" if not pk_text.endswith("\n") else ""))
    os.chmod(key_path, 0o600)
    return {
        "host": cfg.get("host", ""),
        "port": int(cfg.get("port") or 22),
        "user": cfg.get("user", "root"),
        "key_filename": str(key_path),
    }


def _hf_cache_dir(repo_id: str) -> str:
    """HuggingFace's on-disk cache layout: `~/.cache/huggingface/hub/models--<org>--<name>`.
    Slashes in the repo id become double-dashes. Used to clean up after a VM run."""
    sanitised = repo_id.replace("/", "--")
    return f"~/.cache/huggingface/hub/models--{sanitised}"


def _ssh_cleanup_paths_sync(vm_target: dict, paths: list[str]) -> tuple[bool, str]:
    """Open SSH and `rm -rf` each path. Returns (ok, message)."""
    import paramiko
    try:
        pkey = paramiko.PKey.from_path(vm_target["key_filename"])
    except Exception:
        # Older paramiko / non-standard key — fall back to type-probing.
        from .vm_probe import _load_pkey
        with open(vm_target["key_filename"], "r") as f:
            pkey = _load_pkey(f.read())
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=vm_target["host"],
            port=int(vm_target["port"]),
            username=vm_target["user"],
            pkey=pkey,
            timeout=15,
            banner_timeout=15,
            auth_timeout=15,
            look_for_keys=False,
            allow_agent=False,
        )
    except Exception as e:
        return False, f"SSH connect failed: {e}"

    try:
        # Quote each path with single quotes; reject any with embedded quotes
        # (we generated them ourselves, but belt-and-braces).
        safe = [p for p in paths if "'" not in p]
        if not safe:
            return False, "no safe paths to clean"
        # `rm -rf` is fine for missing paths; we use bash -lc so ~ expands.
        cmd = "; ".join(f"rm -rf '{p}'" for p in safe)
        full = f"bash -lc \"{cmd}\""
        stdin, stdout, stderr = client.exec_command(full, timeout=60)
        rc = stdout.channel.recv_exit_status()
        err = stderr.read().decode(errors="replace").strip()
        if rc != 0:
            return False, f"rm exited {rc}: {err[:200]}"
        return True, f"removed {len(safe)} path{'s' if len(safe) != 1 else ''}"
    finally:
        try:
            client.close()
        except Exception:
            pass


async def _cleanup_vm_model(
    redis,
    bench_id: str,
    vm_target: dict,
    raw_yaml: str,
) -> None:
    """After a bare-metal run ends, remove the model from the VM so the disk
    doesn't fill up with stale downloads. Targets both the user's `local_dir`
    (if set) and the standard HF hub cache. Best-effort — failures log but
    never bubble up since the benchmark itself already finished."""
    try:
        cfg = yaml.safe_load(raw_yaml) or {}
        items = cfg.get("benchmark") or []
        first = items[0] if items and isinstance(items, list) else {}
        model = first.get("model") or {}
        repo_id = str(model.get("repo_id") or "").strip()
        local_dir = str(model.get("local_dir") or "").strip()
    except Exception as e:
        await _push_log(redis, bench_id, f"[gateway] vm cleanup: could not parse YAML: {e}")
        return

    paths: list[str] = []
    if local_dir:
        paths.append(local_dir)
    if repo_id:
        paths.append(_hf_cache_dir(repo_id))
    if not paths:
        return

    await _push_log(redis, bench_id, f"[gateway] vm cleanup: removing {', '.join(paths)}")
    try:
        ok, msg = await asyncio.to_thread(_ssh_cleanup_paths_sync, vm_target, paths)
    except Exception as e:
        await _push_log(redis, bench_id, f"[gateway] vm cleanup failed: {e}")
        return
    level = "info" if ok else "warning"
    await _push_log(redis, bench_id, f"[gateway] vm cleanup [{level}]: {msg}")


# ---------- Subprocess runner ------------------------------------------


# Tracks live runs so DELETE can kill the subprocess. {bench_id: asyncio.subprocess.Process}
_LIVE: dict[str, asyncio.subprocess.Process] = {}


def _full_log_path(bench_id: str) -> Path:
    """On-disk file that captures *every* log line for a run, uncapped.
    Uploaded to S3 as `{prefix}logs.txt` on completion so the UI can replay
    the full log even after the redis list has been TTL'd or LRU-trimmed."""
    return _work_dir(bench_id) / "_full.log"


async def _push_log(redis, bench_id: str, line: str) -> None:
    if not line:
        return
    # Append to the full on-disk log (best-effort). This is the canonical
    # record — redis is just the live-tail cache.
    try:
        with _full_log_path(bench_id).open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    key = f"bench:logs:{bench_id}"
    try:
        await redis.rpush(key, line)
        await redis.ltrim(key, -LOG_LIST_CAP, -1)
    except Exception:
        # Logs are best-effort — never let log-pipe failures kill the runner.
        pass


# benchmaq prints `Pod created: <runpod-id>` exactly once when the pod comes
# up. We use that to capture the RunPod $/hour rate so the UI can display
# a live cost ticker. Captured once per bench (set keeps us idempotent).
_POD_CREATED_RE = re.compile(r"Pod created:\s*(\S+)")
_COST_CAPTURED: set[str] = set()


async def _fetch_runpod_cost(pod_id: str) -> Optional[float]:
    """Return RunPod's costPerHr for a pod by id, or None if anything goes
    sideways (no API key, pod not found, transient network error). Best-effort
    — never raises."""
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        return None
    base = os.environ.get("RUNPOD_API_BASE", "https://rest.runpod.io/v1")
    try:
        async with httpx.AsyncClient(
            base_url=base,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15.0,
        ) as cli:
            r = await cli.get(f"/pods/{pod_id}")
            if r.status_code >= 400:
                return None
            data = r.json()
            cost = data.get("costPerHr") or data.get("cost_per_hr")
            return float(cost) if cost is not None else None
    except Exception:
        return None


async def _capture_runpod_cost(bench_id: str, pod_id: str) -> None:
    """Look up the hourly rate for the pod benchmaq just spawned and store it
    on the row. Logged but otherwise best-effort — a cost-tracking failure must
    never break the run."""
    cost = await _fetch_runpod_cost(pod_id)
    try:
        async with session_factory()() as s:
            row = await s.get(Benchmark, bench_id)
            if row is None:
                return
            row.runpod_pod_id = pod_id
            row.cost_per_hr = cost
            await s.commit()
    except Exception:
        logger.warning("bench %s: failed to persist cost for pod %s", bench_id, pod_id)
        return
    logger.info("bench %s: pod=%s cost=%s/hr", bench_id, pod_id, cost)


async def _drain(stream: asyncio.StreamReader, prefix: str, redis, bench_id: str) -> None:
    """Read lines from a subprocess pipe and fan them out to redis + python log."""
    while True:
        line = await stream.readline()
        if not line:
            return
        text = line.decode("utf-8", "replace").rstrip()
        await _push_log(redis, bench_id, f"{prefix}{text}")
        # First-seen `Pod created: <id>` → kick off a cost lookup. Only watch
        # stdout (prefix == "") since stderr can echo unrelated text.
        if not prefix and bench_id not in _COST_CAPTURED:
            m = _POD_CREATED_RE.search(text)
            if m:
                _COST_CAPTURED.add(bench_id)
                asyncio.create_task(_capture_runpod_cost(bench_id, m.group(1)))


async def run_benchmark(redis, bench_id: str, raw_yaml: str) -> None:
    """End-to-end runner for one benchmark. Owns the subprocess from spawn → S3 sync."""
    work = _work_dir(bench_id)

    # Mark running + start time, and read provider_id so we know whether to
    # take the RunPod path or the VM/SSH path.
    async with session_factory()() as s:
        b = await s.get(Benchmark, bench_id)
        if b is None:
            return
        b.status = "running"
        b.started_at = datetime.now(timezone.utc)
        provider_id = b.provider_id
        cleanup_model = bool(getattr(b, "cleanup_model", True))
        await s.commit()

    vm_target: Optional[dict] = None
    if provider_id:
        try:
            vm_target = await _materialise_vm_key(work, provider_id)
            await _push_log(redis, bench_id, f"[gateway] bare-metal target: {vm_target['user']}@{vm_target['host']}:{vm_target['port']}")
        except Exception as e:
            await _push_log(redis, bench_id, f"[gateway] could not prepare VM target: {e}")
            async with session_factory()() as s:
                b2 = await s.get(Benchmark, bench_id)
                if b2 is not None:
                    b2.status = "failed"
                    b2.error_text = f"VM target setup failed: {e}"[:4000]
                    b2.ended_at = datetime.now(timezone.utc)
                    await s.commit()
            return

    cfg_path = work / "config.yaml"
    cfg_path.write_text(_resolve_config(raw_yaml, vm_target=vm_target))

    if vm_target:
        sub_cmd = _pick_engine_subcommand(raw_yaml)
        await _push_log(redis, bench_id, f"[gateway] starting benchmaq {' '.join(sub_cmd)} (cwd={work})")
    else:
        sub_cmd = ["runpod", "bench"]
        await _push_log(redis, bench_id, f"[gateway] starting benchmaq runpod bench (cwd={work})")

    env = dict(os.environ)
    env["RUNPOD_API_KEY"] = os.environ.get("RUNPOD_API_KEY", "")
    env["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    # benchmaq writes results into the cwd by default unless config says otherwise.

    # Prefer the venv-local `benchmaq` (sibling of the running python) since
    # the gateway process inherits PATH from however it was launched, which
    # may not include .venv/bin. Fall back to PATH lookup, then bare name.
    sibling = Path(sys.executable).parent / "benchmaq"
    if sibling.exists():
        benchmaq_bin = str(sibling)
    else:
        benchmaq_bin = shutil.which("benchmaq") or "benchmaq"

    # Make sure the venv's bin is on PATH for the subprocess too — benchmaq
    # itself shells out to runpodctl, uv, etc., and may need them.
    env_path = env.get("PATH", "")
    venv_bin = str(Path(sys.executable).parent)
    if venv_bin not in env_path.split(":"):
        env["PATH"] = f"{venv_bin}:{env_path}" if env_path else venv_bin

    # Force unbuffered stdout/stderr — without this, benchmaq's print()s sit
    # in the pipe buffer and the UI sees nothing until the run finishes.
    env["PYTHONUNBUFFERED"] = "1"

    await _push_log(redis, bench_id, f"[gateway] benchmaq binary: {benchmaq_bin}")

    # Invoke through python -u so even C-level stdio is line-buffered, in case
    # benchmaq spawns subprocesses (runpodctl) whose output also needs to flow.
    # For VM (bare-metal) runs, we route through a thin wrapper that installs
    # the pyremote reconnect-per-command shim before benchmaq's CLI runs.
    # This sidesteps Go-based SSH proxies (e.g. TM's `ssh.*.gpu.tm.com.my`)
    # that enforce one exec channel per TCP connection.
    if vm_target is not None:
        cmd_argv = [sys.executable, "-u", "-m", "gateway.bench_remote_wrapper", *sub_cmd, str(cfg_path)]
    else:
        cmd_argv = [sys.executable, "-u", benchmaq_bin, *sub_cmd, str(cfg_path)]
    proc = await asyncio.create_subprocess_exec(
        *cmd_argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(work),
        env=env,
    )
    _LIVE[bench_id] = proc

    try:
        await asyncio.gather(
            _drain(proc.stdout, "", redis, bench_id),
            _drain(proc.stderr, "[stderr] ", redis, bench_id),
        )
        rc = await proc.wait()
    except asyncio.CancelledError:
        await _push_log(redis, bench_id, "[gateway] cancelled — killing subprocess")
        try:
            proc.kill()
        except Exception:
            pass
        rc = -1
        raise
    finally:
        _LIVE.pop(bench_id, None)

    await _push_log(redis, bench_id, f"[gateway] benchmaq exited rc={rc}")

    # Bare-metal runs leave the model in the VM's HF cache + local_dir. Clean
    # both up so a series of benchmarks on different models doesn't fill the
    # VM's disk. Best-effort: failures are logged but don't change the run
    # outcome (the benchmark already finished).
    if vm_target is not None and cleanup_model:
        try:
            await _cleanup_vm_model(redis, bench_id, vm_target, raw_yaml)
        except Exception as e:
            await _push_log(redis, bench_id, f"[gateway] vm cleanup crashed: {e}")
    elif vm_target is not None:
        await _push_log(redis, bench_id, "[gateway] vm cleanup: skipped (cleanup_model=false)")

    # Sync any result files dropped under work/ into S3.
    prefix = benchmark_s3_prefix(bench_id)
    s3_put_text(f"{prefix}config.yaml", _resolve_config(raw_yaml, vm_target=vm_target))
    result_json: Optional[dict] = None
    error_excerpt: Optional[str] = None

    # Upload the complete, uncapped log to S3 as logs.txt. The stream endpoint
    # falls back to this for terminal benches so the UI can replay the full
    # log forever, even after the redis list has been trimmed or TTL'd.
    full_log = _full_log_path(bench_id)
    if full_log.exists():
        try:
            s3_put_file(f"{prefix}logs.txt", str(full_log))
        except Exception as e:
            await _push_log(redis, bench_id, f"[gateway] s3 upload failed for logs.txt: {e}")

    for path in sorted(work.rglob("*")):
        if not path.is_file() or path.name in ("config.yaml", "_full.log"):
            continue
        rel = path.relative_to(work).as_posix()
        try:
            s3_put_file(f"{prefix}{rel}", str(path))
        except Exception as e:
            await _push_log(redis, bench_id, f"[gateway] s3 upload failed for {rel}: {e}")
        # First plausible result.json wins for the Metrics tab.
        if result_json is None and path.suffix == ".json":
            try:
                with path.open() as f:
                    candidate = json.load(f)
                if isinstance(candidate, dict):
                    result_json = candidate
            except Exception:
                pass

    if rc != 0:
        # Tail the full on-disk log for the error_text card on the list page.
        try:
            if full_log.exists():
                with full_log.open("r", encoding="utf-8", errors="replace") as f:
                    tail_lines = f.readlines()[-50:]
                error_excerpt = "".join(tail_lines)[-4000:]
                # Surface a clean one-liner for known failure patterns so the
                # list page doesn't show a raw log wall.
                _cuda_m = re.search(
                    r"CUDA mismatch[^\n]{0,200}", error_excerpt, re.IGNORECASE
                )
                if _cuda_m:
                    error_excerpt = _cuda_m.group(0).strip()
        except Exception:
            error_excerpt = None

    async with session_factory()() as s:
        b = await s.get(Benchmark, bench_id)
        if b is None:
            return
        b.status = "done" if rc == 0 else "failed"
        b.exit_code = rc
        b.error_text = error_excerpt
        b.result_json = result_json
        b.ended_at = datetime.now(timezone.utc)
        await s.commit()

    # TTL on the log list so old runs eventually drop out of redis.
    try:
        await redis.expire(f"bench:logs:{bench_id}", LOG_LIST_TTL_S)
    except Exception:
        pass


# ---------- Startup hooks -----------------------------------------------


def bootstrap_ssh_key_from_env() -> None:
    """Prod delivers the RunPod SSH private key via env var
    (BENCHMARK_SSH_PRIVATE_KEY) from a SealedSecret — pods can't read the
    developer's ~/.runpod/ssh/RunPod-Key-Go file. We materialize the env
    value to disk at BENCHMARK_SSH_KEY_PATH (chmod 0600) on startup so the
    rest of the code keeps using a normal file path.

    Idempotent: skips if the file already exists, so local dev keeps using
    the runpodctl-managed key without disturbance.
    """
    key = os.environ.get("BENCHMARK_SSH_PRIVATE_KEY", "")
    if not key.strip():
        return
    path = Path(_ssh_key_path())
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    text = key if key.endswith("\n") else key + "\n"
    path.write_text(text)
    try:
        path.chmod(0o600)
    except Exception:
        pass
    logger.info("bench: wrote SSH key from env to %s", path)


async def cleanup_orphaned_running(redis) -> int:
    """Called from main.py lifespan. Marks any rows still 'running' (left over
    from a previous gateway process) as 'failed' with a recovery message."""
    async with session_factory()() as s:
        rows = await s.execute(
            update(Benchmark)
            .where(Benchmark.status.in_(["running", "queued"]))
            .values(
                status="failed",
                error_text="orphaned by gateway restart — pod (if any) is still on RunPod, terminate it manually",
                ended_at=datetime.now(timezone.utc),
            )
            .returning(Benchmark.id)
        )
        ids = [row[0] for row in rows.all()]
        await s.commit()
    for bid in ids:
        # Push the marker line first so it lands in _full.log before we
        # upload that file as the canonical logs.txt in S3.
        try:
            await _push_log(redis, bid, "[gateway] orphaned by gateway restart — marking failed")
        except Exception:
            pass
        # Upload whatever we managed to capture on disk so the Logs tab can
        # replay it. Without this, the stream falls back to the trimmed redis
        # tail and the user loses the head of the run.
        full_log = _full_log_path(bid)
        if full_log.exists():
            try:
                s3_put_file(f"{benchmark_s3_prefix(bid)}logs.txt", str(full_log))
            except Exception as e:
                logger.warning("orphan %s: failed to upload _full.log: %s", bid, e)
    return len(ids)


# ---------- Pydantic schemas -------------------------------------------


class CreateBenchmarkRequest(BaseModel):
    name: str
    config_yaml: str
    # NULL/absent means use the platform default cloud (RunPod). Set to a
    # provider id (from /v1/providers) to bind this run to a user-registered
    # VM. Phase 2 just persists the choice; phase 3 will route execution.
    provider_id: Optional[str] = None
    # VM runs only: remove the model from the VM after the run. Ignored for
    # cloud runs since the RunPod pod is torn down anyway.
    cleanup_model: Optional[bool] = None


class BenchmarkRecord(BaseModel):
    id: str
    name: str
    status: str
    s3_prefix: str
    config_yaml: str
    exit_code: Optional[int] = None
    error_text: Optional[str] = None
    result_json: Optional[dict] = None
    created_by: str
    created_at: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    cost_per_hr: Optional[float] = None
    provider_id: Optional[str] = None


class FileRecord(BaseModel):
    name: str
    size: int
    modified: str
    download_url: str


class TemplateRecord(BaseModel):
    id: str
    name: str
    config_yaml: str
    created_at: str


class CreateTemplateRequest(BaseModel):
    name: str
    config_yaml: str


# ---------- HTTP API ----------------------------------------------------


router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


def _to_record(b: Benchmark, owner_username: str) -> BenchmarkRecord:
    return BenchmarkRecord(
        id=b.id,
        name=b.name,
        status=b.status,
        s3_prefix=b.s3_prefix,
        config_yaml=b.config_yaml,
        exit_code=b.exit_code,
        error_text=b.error_text,
        result_json=b.result_json,
        created_by=owner_username,
        created_at=b.created_at.isoformat() if b.created_at else "",
        started_at=b.started_at.isoformat() if b.started_at else None,
        ended_at=b.ended_at.isoformat() if b.ended_at else None,
        cost_per_hr=b.cost_per_hr,
        provider_id=b.provider_id,
    )


# ---------- Templates --------------------------------------------------
# These come BEFORE the /benchmarks/{id}/* routes so /benchmarks/templates
# isn't captured by the {bench_id} path parameter.


@router.get("/templates", response_model=list[TemplateRecord])
async def list_templates(
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    rows = await session.execute(
        select(BenchmarkTemplate)
        .where(BenchmarkTemplate.owner_id == user.id)
        .order_by(BenchmarkTemplate.created_at.desc())
    )
    return [
        TemplateRecord(
            id=t.id,
            name=t.name,
            config_yaml=t.config_yaml,
            created_at=t.created_at.isoformat() if t.created_at else "",
        )
        for t in rows.scalars().all()
    ]


@router.post("/templates", response_model=TemplateRecord)
async def create_template(
    body: CreateTemplateRequest,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    # Validate the YAML at least parses — saving garbage helps no one.
    try:
        cfg = yaml.safe_load(body.config_yaml) or {}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail={"error": f"invalid YAML: {e}"})
    if not isinstance(cfg, dict):
        raise HTTPException(status_code=400, detail={"error": "top-level YAML must be a mapping"})

    import uuid
    t = BenchmarkTemplate(
        id=f"tpl-{uuid.uuid4().hex[:8]}",
        name=body.name.strip()[:128] or "untitled",
        config_yaml=body.config_yaml,
        owner_id=user.id,
    )
    session.add(t)
    await session.commit()
    return TemplateRecord(
        id=t.id, name=t.name, config_yaml=t.config_yaml,
        created_at=t.created_at.isoformat() if t.created_at else "",
    )


@router.delete("/templates/{template_id}")
async def delete_template(
    template_id: str,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    t = await session.get(BenchmarkTemplate, template_id)
    if not t:
        raise HTTPException(status_code=404, detail={"error": "template not found"})
    if not user.is_admin and t.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    await session.delete(t)
    await session.commit()
    return {"ok": True, "id": template_id}


# ---------- Benchmarks -------------------------------------------------


@router.post("", response_model=BenchmarkRecord)
async def create_benchmark(
    body: CreateBenchmarkRequest,
    request: Request,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    try:
        cfg = yaml.safe_load(body.config_yaml) or {}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail={"error": f"invalid YAML: {e}"})
    if not isinstance(cfg, dict):
        raise HTTPException(status_code=400, detail={"error": "top-level YAML must be a mapping"})

    if body.provider_id:
        from .db import Provider
        prov = await session.get(Provider, body.provider_id)
        if prov is None:
            raise HTTPException(status_code=400, detail={"error": "unknown provider_id"})

    bench_id = _gen_id()
    s3_prefix = benchmark_s3_prefix(bench_id)

    bench = Benchmark(
        id=bench_id,
        name=body.name,
        config_yaml=body.config_yaml,
        status="queued",
        s3_prefix=s3_prefix,
        owner_id=user.id,
        provider_id=body.provider_id,
        # Only honoured when provider_id is set (VM path). Default True.
        cleanup_model=True if body.cleanup_model is None else bool(body.cleanup_model),
    )
    session.add(bench)
    await session.commit()

    # Kick off the runner. asyncio.create_task is fire-and-forget — the runner
    # owns its own DB session + error handling.
    redis = request.app.state.redis
    asyncio.create_task(_safe_run(redis, bench_id, body.config_yaml))

    await audit.record(user, "benchmark.create", "benchmark", bench_id, body.name)

    bench = await session.get(Benchmark, bench_id)
    return _to_record(bench, user.username)


async def _safe_run(redis, bench_id: str, raw_yaml: str) -> None:
    try:
        await run_benchmark(redis, bench_id, raw_yaml)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("benchmark %s crashed: %s", bench_id, e)
        async with session_factory()() as s:
            b = await s.get(Benchmark, bench_id)
            if b and b.status not in ("done", "failed", "cancelled"):
                b.status = "failed"
                b.error_text = f"runner crashed: {e}"[:4000]
                b.ended_at = datetime.now(timezone.utc)
                await s.commit()


@router.get("", response_model=list[BenchmarkRecord])
async def list_benchmarks(
    scope: str = "mine",
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    # Admins default to their own runs; pass ?scope=all to see everyone's.
    # Non-admins are always scoped to own regardless of the param.
    show_all = user.is_admin and scope == "all"
    if show_all:
        rows = await session.execute(select(Benchmark).order_by(Benchmark.created_at.desc()))
    else:
        rows = await session.execute(
            select(Benchmark).where(Benchmark.owner_id == user.id).order_by(Benchmark.created_at.desc())
        )
    out: list[BenchmarkRecord] = []
    for b in rows.scalars().all():
        owner = await session.get(User, b.owner_id)
        out.append(_to_record(b, owner.username if owner else ""))
    return out


# ---------- Aggregate (cross-benchmark dashboard) ----------------------
# Defined BEFORE /{bench_id} so the literal path "/_aggregate" isn't captured
# by the path-param matcher. (FastAPI/Starlette match in declaration order.)


class AggregatePoint(BaseModel):
    benchmark_id: str
    benchmark_name: str
    model: str | None = None
    gpu_type: str | None = None
    gpu_count: int = 1
    engine: str = "vllm"
    tp: int = 1
    dp: int = 1
    context_len: int = 0
    output_len: int = 0
    concurrency: int = 0
    num_prompts: int = 0
    duration_s: float | None = None
    output_throughput: float | None = None
    output_throughput_per_gpu: float | None = None
    request_throughput: float | None = None
    median_ttft_ms: float | None = None
    p99_ttft_ms: float | None = None
    median_tpot_ms: float | None = None
    p99_tpot_ms: float | None = None
    median_itl_ms: float | None = None
    median_e2el_ms: float | None = None
    p99_e2el_ms: float | None = None


_AGG_CACHE: dict[str, tuple[float, list[AggregatePoint]]] = {}
_AGG_TTL_S = 60.0


def _safe_num(d: dict, k: str) -> float | None:
    v = d.get(k)
    if isinstance(v, (int, float)) and v == v:
        return float(v)
    return None


def _parse_dims_from_filename(name: str) -> dict:
    import re
    base = name.split("/")[-1]
    out = {"context_len": 0, "output_len": 0, "num_prompts": 0, "concurrency": 0}
    m = re.search(r"_in(\d+)_out(\d+)_p(\d+)_c(\d+)", base)
    if m:
        out["context_len"] = int(m.group(1))
        out["output_len"] = int(m.group(2))
        out["num_prompts"] = int(m.group(3))
        out["concurrency"] = int(m.group(4))
    return out


def _parse_config(yaml_text: str) -> dict:
    try:
        cfg = yaml.safe_load(yaml_text) or {}
    except Exception:
        return {}
    if not isinstance(cfg, dict):
        return {}
    pod = ((cfg.get("runpod") or {}).get("pod") or {})
    benches = cfg.get("benchmark") or []
    first = benches[0] if benches else {}
    serve = (first.get("serve") or {})
    return {
        "gpu_type": pod.get("gpu_type"),
        "gpu_count": int(pod.get("gpu_count") or 1),
        "engine": first.get("engine") or "vllm",
        "model": ((first.get("model") or {}).get("repo_id")),
        "tp": int(serve.get("tensor_parallel_size") or 1),
        "dp": int(serve.get("data_parallel_size") or 1),
    }


@router.get("/_aggregate", response_model=list[AggregatePoint])
async def aggregate(
    scope: str = "mine",
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    show_all = user.is_admin and scope == "all"
    cache_key = "admin-all" if show_all else f"u{user.id}"
    now = time.time()
    cached = _AGG_CACHE.get(cache_key)
    if cached and cached[0] > now:
        return cached[1]

    if show_all:
        rows = await session.execute(
            select(Benchmark).where(Benchmark.status.in_(["done", "running", "failed"]))
        )
    else:
        rows = await session.execute(
            select(Benchmark)
            .where(Benchmark.owner_id == user.id)
            .where(Benchmark.status.in_(["done", "running", "failed"]))
        )
    benches = list(rows.scalars().all())

    cli = _s3_client()
    bucket = _bucket()

    async def fetch_one(b: Benchmark) -> list[AggregatePoint]:
        cfg_meta = _parse_config(b.config_yaml or "")
        # VM (bare-metal) benches don't have a runpod.pod block in the YAML,
        # so gpu_type/gpu_count come back empty. Fall back to the provider's
        # stored GPU info (populated by the last Test/availability probe) so
        # the Performance explorer can label the series with e.g. "L40S
        # (TM-VM1)" instead of "—".
        if b.provider_id and not cfg_meta.get("gpu_type"):
            try:
                async with session_factory()() as _s:
                    from .db import Provider as _Provider
                    prov = await _s.get(_Provider, b.provider_id)
                if prov is not None:
                    pcfg = prov.config or {}
                    gpus_list = pcfg.get("gpus") or []
                    if isinstance(gpus_list, list) and gpus_list:
                        short = str(gpus_list[0]).replace("NVIDIA ", "").strip()
                        cfg_meta["gpu_type"] = f"{short} ({prov.name})"
                    cfg_meta["gpu_count"] = int(pcfg.get("gpu_count") or cfg_meta.get("gpu_count") or 1)
            except Exception as e:
                logger.warning("aggregate: provider lookup for %s failed: %s", b.id, e)
        try:
            keys = []
            token = None
            while True:
                kwargs = {"Bucket": bucket, "Prefix": b.s3_prefix}
                if token: kwargs["ContinuationToken"] = token
                r = await asyncio.to_thread(cli.list_objects_v2, **kwargs)
                for obj in r.get("Contents", []):
                    k = obj["Key"]
                    if k.lower().endswith(".json") and not k.endswith("_DONE"):
                        keys.append(k)
                if not r.get("IsTruncated"): break
                token = r.get("NextContinuationToken")
        except Exception as e:
            logger.warning("aggregate: list %s failed: %s", b.id, e)
            return []

        async def fetch_json(key: str) -> AggregatePoint | None:
            try:
                obj = await asyncio.to_thread(cli.get_object, Bucket=bucket, Key=key)
                body = obj["Body"].read()
                data = json.loads(body)
            except Exception:
                return None
            if not isinstance(data, dict):
                return None
            dims = _parse_dims_from_filename(key)
            gpu_count = cfg_meta.get("gpu_count") or 1
            tput = _safe_num(data, "output_throughput")
            return AggregatePoint(
                benchmark_id=b.id,
                benchmark_name=b.name,
                model=cfg_meta.get("model"),
                gpu_type=cfg_meta.get("gpu_type"),
                gpu_count=gpu_count,
                engine=cfg_meta.get("engine") or "vllm",
                tp=cfg_meta.get("tp") or 1,
                dp=cfg_meta.get("dp") or 1,
                context_len=dims["context_len"] or int(_safe_num(data, "random_input_len") or 0),
                output_len=dims["output_len"] or int(_safe_num(data, "random_output_len") or 0),
                concurrency=dims["concurrency"] or int(_safe_num(data, "max_concurrency") or 0),
                num_prompts=dims["num_prompts"] or int(_safe_num(data, "num_prompts") or 0),
                duration_s=_safe_num(data, "duration"),
                output_throughput=tput,
                output_throughput_per_gpu=(tput / gpu_count) if (tput and gpu_count) else None,
                request_throughput=_safe_num(data, "request_throughput"),
                median_ttft_ms=_safe_num(data, "median_ttft_ms"),
                p99_ttft_ms=_safe_num(data, "p99_ttft_ms"),
                median_tpot_ms=_safe_num(data, "median_tpot_ms"),
                p99_tpot_ms=_safe_num(data, "p99_tpot_ms"),
                median_itl_ms=_safe_num(data, "median_itl_ms"),
                median_e2el_ms=_safe_num(data, "median_e2el_ms"),
                p99_e2el_ms=_safe_num(data, "p99_e2el_ms"),
            )

        results = await asyncio.gather(*[fetch_json(k) for k in keys])
        return [p for p in results if p is not None]

    nested = await asyncio.gather(*[fetch_one(b) for b in benches])
    flat: list[AggregatePoint] = [p for sub in nested for p in sub]
    _AGG_CACHE[cache_key] = (now + _AGG_TTL_S, flat)
    return flat


@router.get("/{bench_id}", response_model=BenchmarkRecord)
async def get_benchmark(
    bench_id: str,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    b = await session.get(Benchmark, bench_id)
    if not b:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    if not user.is_admin and b.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    owner = await session.get(User, b.owner_id)
    return _to_record(b, owner.username if owner else "")


@router.delete("/{bench_id}")
async def delete_benchmark(
    bench_id: str,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    b = await session.get(Benchmark, bench_id)
    if not b:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    if not user.is_admin and b.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    proc = _LIVE.pop(bench_id, None)
    if proc and proc.returncode is None:
        try:
            proc.kill()
        except Exception:
            pass
    bench_name = b.name
    # Snapshot billing inputs before the row is gone. If the user deletes a
    # bench that's still running, ended_at will be None and the audit helper
    # treats "now" as the end — giving us a "spent so far at deletion" total.
    cost = audit.cost_breakdown(b.started_at, b.ended_at, b.cost_per_hr)
    await session.delete(b)
    await session.commit()
    await audit.record(
        user, "benchmark.delete", "benchmark", bench_id, bench_name,
        details=cost,
    )
    return {"ok": True, "id": bench_id}


@router.get("/{bench_id}/logs/stream")
async def stream_logs(
    bench_id: str,
    request: Request,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    b = await session.get(Benchmark, bench_id)
    if not b:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    if not user.is_admin and b.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"error": "forbidden"})

    redis = request.app.state.redis
    initial_status = b.status
    # Source-of-truth for logs is the on-disk _full.log (uncapped) while the
    # bench is live, and S3 logs.txt once it's been uploaded. Redis is only a
    # last-resort fallback for benches that ran before the on-disk tee landed.
    s3_full_log: Optional[str] = None
    if initial_status in ("done", "failed", "cancelled"):
        s3_full_log = s3_get_text(f"{benchmark_s3_prefix(bench_id)}logs.txt")
    full_log = _full_log_path(bench_id)

    async def gen() -> AsyncIterator[bytes]:
        # 1) Terminal + S3 has it → stream the canonical copy and close.
        if s3_full_log is not None:
            for line in s3_full_log.splitlines():
                yield f"data: {line}\n\n".encode("utf-8")
            yield f"event: end\ndata: {initial_status}\n\n".encode("utf-8")
            return

        # 2) Terminal but no S3 copy and no on-disk file → legacy bench, fall
        # back to whatever redis still has (will be trimmed, but it's all we've
        # got). Newer terminal benches always have one of the above.
        if initial_status in ("done", "failed", "cancelled") and not full_log.exists():
            key = f"bench:logs:{bench_id}"
            try:
                lines = await redis.lrange(key, 0, -1)
            except Exception:
                lines = []
            for line in lines:
                yield f"data: {line}\n\n".encode("utf-8")
            yield f"event: end\ndata: {initial_status}\n\n".encode("utf-8")
            return

        # 3) Live or recently-terminal: tail _full.log from disk. This is the
        # uncapped, canonical record — the file is appended to on every
        # _push_log call, so we just keep reading from where we left off.
        pos = 0
        buf = b""
        while True:
            chunk = b""
            if full_log.exists():
                try:
                    with full_log.open("rb") as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos += len(chunk)
                except Exception:
                    chunk = b""
            if chunk:
                buf += chunk
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line = buf[:nl].decode("utf-8", "replace")
                    buf = buf[nl + 1:]
                    yield f"data: {line}\n\n".encode("utf-8")
                continue
            # No new bytes — check whether the run finished.
            async with session_factory()() as s:
                cur = await s.get(Benchmark, bench_id)
            if cur and cur.status in ("done", "failed", "cancelled"):
                # Flush any trailing partial line (run ended mid-write).
                if buf:
                    yield f"data: {buf.decode('utf-8', 'replace')}\n\n".encode("utf-8")
                    buf = b""
                yield f"event: end\ndata: {cur.status}\n\n".encode("utf-8")
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )



@router.get("/{bench_id}/files", response_model=list[FileRecord])
async def list_files(
    bench_id: str,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    b = await session.get(Benchmark, bench_id)
    if not b:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    if not user.is_admin and b.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    items = s3_list(b.s3_prefix)
    out: list[FileRecord] = []
    for it in items:
        rel = it["key"][len(b.s3_prefix):] if it["key"].startswith(b.s3_prefix) else it["key"]
        out.append(FileRecord(
            name=rel or it["key"],
            size=it["size"],
            modified=it["modified"],
            download_url=s3_presign_get(it["key"]),
        ))
    return out
