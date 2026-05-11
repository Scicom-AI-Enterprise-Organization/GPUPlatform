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
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import boto3
import yaml
from botocore.client import Config as BotoConfig
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    DateTime,
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


def _resolve_config(raw_yaml: str) -> str:
    """Inject runtime values (SSH key path, RunPod API key) into the user's YAML.

    Users paste a config that may have `ssh_private_key: "path/to/your/private/key"`
    or empty `runpod_api_key: ""`. We replace those with real values from env so
    they don't have to know about the runpodctl-managed key location.
    """
    cfg = yaml.safe_load(raw_yaml) or {}
    if not isinstance(cfg, dict):
        return raw_yaml

    rp = cfg.setdefault("runpod", {})
    if not rp.get("ssh_private_key") or "path/to/your" in str(rp.get("ssh_private_key")):
        rp["ssh_private_key"] = _ssh_key_path()
    if not rp.get("runpod_api_key"):
        rp["runpod_api_key"] = os.environ.get("RUNPOD_API_KEY", "")

    rem = cfg.setdefault("remote", {})
    if not rem.get("key_filename") or "path/to/your" in str(rem.get("key_filename")):
        rem["key_filename"] = _ssh_key_path()

    return yaml.safe_dump(cfg, sort_keys=False)


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


async def _drain(stream: asyncio.StreamReader, prefix: str, redis, bench_id: str) -> None:
    """Read lines from a subprocess pipe and fan them out to redis + python log."""
    while True:
        line = await stream.readline()
        if not line:
            return
        text = line.decode("utf-8", "replace").rstrip()
        await _push_log(redis, bench_id, f"{prefix}{text}")


async def run_benchmark(redis, bench_id: str, raw_yaml: str) -> None:
    """End-to-end runner for one benchmark. Owns the subprocess from spawn → S3 sync."""
    work = _work_dir(bench_id)
    cfg_path = work / "config.yaml"
    cfg_path.write_text(_resolve_config(raw_yaml))

    # Mark running + start time.
    async with session_factory()() as s:
        b = await s.get(Benchmark, bench_id)
        if b is None:
            return
        b.status = "running"
        b.started_at = datetime.now(timezone.utc)
        await s.commit()

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
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", benchmaq_bin, "runpod", "bench", str(cfg_path),
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

    # Sync any result files dropped under work/ into S3.
    prefix = benchmark_s3_prefix(bench_id)
    s3_put_text(f"{prefix}config.yaml", _resolve_config(raw_yaml))
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
        try:
            await _push_log(redis, bid, "[gateway] orphaned by gateway restart — marking failed")
        except Exception:
            pass
    return len(ids)


# ---------- Pydantic schemas -------------------------------------------


class CreateBenchmarkRequest(BaseModel):
    name: str
    config_yaml: str


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

    bench_id = _gen_id()
    s3_prefix = benchmark_s3_prefix(bench_id)

    bench = Benchmark(
        id=bench_id,
        name=body.name,
        config_yaml=body.config_yaml,
        status="queued",
        s3_prefix=s3_prefix,
        owner_id=user.id,
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
    await session.delete(b)
    await session.commit()
    await audit.record(user, "benchmark.delete", "benchmark", bench_id, bench_name)
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
    # If the bench is already terminal and we've uploaded logs.txt to S3,
    # stream the full log from there — this is the canonical, uncapped copy.
    # Redis may have been trimmed or TTL'd by now, so it would otherwise show
    # a partial / empty log for older runs.
    initial_status = b.status
    s3_full_log: Optional[str] = None
    if initial_status in ("done", "failed", "cancelled"):
        s3_full_log = s3_get_text(f"{benchmark_s3_prefix(bench_id)}logs.txt")

    async def gen() -> AsyncIterator[bytes]:
        if s3_full_log is not None:
            for line in s3_full_log.splitlines():
                yield f"data: {line}\n\n".encode("utf-8")
            yield f"event: end\ndata: {initial_status}\n\n".encode("utf-8")
            return

        key = f"bench:logs:{bench_id}"
        # Replay everything we already have, then poll for new lines until terminal.
        cursor = 0
        while True:
            try:
                lines = await redis.lrange(key, cursor, cursor + 199)
            except Exception:
                lines = []
            if lines:
                for line in lines:
                    yield f"data: {line}\n\n".encode("utf-8")
                cursor += len(lines)
            else:
                # Check terminal status — if so, send a final marker and close.
                async with session_factory()() as s:
                    cur = await s.get(Benchmark, bench_id)
                if cur and cur.status in ("done", "failed", "cancelled"):
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
