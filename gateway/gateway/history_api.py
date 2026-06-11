"""Job-history API — read-only, admin-gated endpoints that expose the durable
history of every job-like record on the platform so an external tool (SlurmUI)
can poll and ingest it.

One endpoint per kind (the records have different native shapes), each returning
a common `JobRecord` envelope plus a kind-specific `detail` blob:

    GET /v1/history/benchmarks   — benchmark runs        (benchmarks table)
    GET /v1/history/training     — autotrain runs        (training_runs table)
    GET /v1/history/compute      — compute pods          (compute_pods table)
    GET /v1/history/inference    — serverless requests   (requests table)
    GET /v1/history/proxy        — LLM-proxy requests    (proxy_requests table)

Common query params (all optional): `since`/`until` (ISO-8601, filter on
created_at), `status` (csv), `user` (username) or `owner_id` (int), `limit`
(default 200, max 1000), `offset`, `order` (asc|desc, default desc).

Auth: **admin only** — this returns every user's records platform-wide, so
SlurmUI should authenticate with an admin `sgpu_` API key.

Volume note: the `requests`/`proxy_requests` tables are high-volume. Those two
endpoints default to the last 7 days when `since` is omitted, and only project
small columns (the request payload / vLLM output JSON is never loaded — loading
it over a wide window OOMs the gateway). benchmark/training/compute are
low-volume and have no default window.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import require_admin
from .db import App, Provider, Request as ReqRow, User, get_session, get_user_by_username

router = APIRouter(prefix="/v1/history", tags=["history"])

DEFAULT_LIMIT = 200
MAX_LIMIT = 1000
DEFAULT_WINDOW_DAYS = 7  # look-back for the high-volume tables when `since` omitted


# ── helpers ──────────────────────────────────────────────────────────────────

def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _parse_dt(s: Optional[str], field: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid {field} (expected ISO-8601): {s!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _status_list(status: Optional[str]) -> Optional[list[str]]:
    if not status:
        return None
    wanted = [s.strip() for s in status.split(",") if s.strip()]
    return wanted or None


def _duration_s(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    if start is None or end is None:
        return None
    d = (end - start).total_seconds()
    return round(d, 3) if d >= 0 else None


def _int(x: Any) -> Optional[int]:
    try:
        return int(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _bench_meta(config_yaml: Optional[str]) -> dict[str, Any]:
    """Best-effort GPU + serve topology from a benchmark's submitted config YAML.
    The `benchmarks` table has no gpu column — the GPU (and TP/DP/EP) live in the
    config the user submitted (`runpod.pod.*` + the first `benchmark[].serve.*`).
    All keys are None when absent (e.g. a manual/`base_url` run names no pod)."""
    meta: dict[str, Any] = {
        "gpu_type": None, "gpu_count": None, "engine": None, "base_url": None,
        "tensor_parallel_size": None, "data_parallel_size": None,
        "expert_parallel": None, "max_model_len": None,
    }
    if not config_yaml:
        return meta
    try:
        cfg = yaml.safe_load(config_yaml)
    except yaml.YAMLError:
        return meta
    if not isinstance(cfg, dict):
        return meta
    runpod = cfg.get("runpod") if isinstance(cfg.get("runpod"), dict) else {}
    pod = runpod.get("pod") if isinstance(runpod.get("pod"), dict) else {}
    meta["gpu_type"] = pod.get("gpu_type")
    meta["gpu_count"] = pod.get("gpu_count")
    benches = cfg.get("benchmark")
    first = benches[0] if isinstance(benches, list) and benches and isinstance(benches[0], dict) else {}
    meta["engine"] = first.get("engine")
    meta["base_url"] = first.get("base_url")
    serve = first.get("serve") if isinstance(first.get("serve"), dict) else {}
    meta["tensor_parallel_size"] = serve.get("tensor_parallel_size")
    meta["data_parallel_size"] = serve.get("data_parallel_size")
    meta["expert_parallel"] = serve.get("enable_expert_parallel")
    meta["max_model_len"] = serve.get("max_model_len")
    return meta


async def _prov_map(session: AsyncSession, prov_ids: set[Optional[str]]) -> dict[str, Provider]:
    """Resolve provider_id → Provider row for one page of records, so every kind
    can report which backend (vm | runpod | pi) and which registered account a
    job ran on instead of an opaque provider_id."""
    ids = {p for p in prov_ids if p}
    if not ids:
        return {}
    rows = (await session.execute(select(Provider).where(Provider.id.in_(ids)))).scalars().all()
    return {p.id: p for p in rows}


def _prov_fields(prov: Optional[Provider]) -> dict[str, Any]:
    """{provider_kind, provider_name} for a detail blob. provider_kind is None
    (not "runpod") when no provider row resolves — the caller decides the
    platform-default backend for its own kind."""
    return {
        "provider_kind": prov.kind if prov else None,
        "provider_name": prov.name if prov else None,
    }


async def _owner_filter(session: AsyncSession, user: Optional[str], owner_id: Optional[int]) -> Optional[int]:
    """Resolve the requested owner. `user` (username) wins over `owner_id`.
    Returns None for "no filter", or -1 (matches nothing) if the username is unknown."""
    if user:
        u = await get_user_by_username(session, user)
        return u.id if u else -1
    return owner_id


async def _user_map(session: AsyncSession, owner_ids: list[Optional[int]]) -> dict[int, str]:
    ids = {i for i in owner_ids if i is not None}
    if not ids:
        return {}
    rows = (await session.execute(select(User).where(User.id.in_(ids)))).scalars().all()
    return {u.id: u.username for u in rows}


def _username(umap: dict[int, str], owner_id: Optional[int]) -> str:
    if owner_id is None:
        return "(anonymous)"
    return umap.get(owner_id, str(owner_id))


# ── response models ──────────────────────────────────────────────────────────

class JobRecord(BaseModel):
    kind: str
    id: str
    name: Optional[str] = None
    user: str                       # username (or "(anonymous)" / numeric id fallback)
    owner_id: Optional[int] = None
    status: str
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_s: Optional[float] = None
    error_text: Optional[str] = None
    detail: dict[str, Any] = {}     # kind-specific extras


class HistoryResponse(BaseModel):
    kind: str
    count: int
    has_more: bool                  # true if more rows exist past this page
    window: dict[str, Optional[str]]  # {since, until} actually applied
    jobs: list[JobRecord]
    note: str


# ── shared query for the low-volume "discrete job" tables ────────────────────

async def _query_discrete(session, model_cls, *, since, until, status, owner, limit, offset, order):
    st = select(model_cls)
    if since is not None:
        st = st.where(model_cls.created_at >= since)
    if until is not None:
        st = st.where(model_cls.created_at < until)
    wanted = _status_list(status)
    if wanted:
        st = st.where(model_cls.status.in_(wanted))
    if owner is not None:
        st = st.where(model_cls.owner_id == owner)
    st = st.order_by(model_cls.created_at.asc() if order == "asc" else model_cls.created_at.desc())
    rows = (await session.execute(st.offset(offset).limit(limit + 1))).scalars().all()
    return rows[:limit], len(rows) > limit


# Common Query() declarations reused across endpoints.
_Q_SINCE = Query(None, description="ISO-8601; keep rows with created_at >= since")
_Q_UNTIL = Query(None, description="ISO-8601; keep rows with created_at < until")
_Q_STATUS = Query(None, description="comma-separated statuses to include")
_Q_USER = Query(None, description="filter by username (wins over owner_id)")
_Q_OWNER = Query(None, description="filter by numeric owner id")
_Q_LIMIT = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)
_Q_OFFSET = Query(0, ge=0)
_Q_ORDER = Query("desc", pattern="^(asc|desc)$", description="order by created_at")


@router.get("", summary="List the available history kinds")
async def history_index(_: User = Depends(require_admin)):
    return {
        "kinds": ["benchmark", "training", "compute", "inference", "proxy"],
        "endpoints": {
            "benchmark": "/v1/history/benchmarks",
            "training": "/v1/history/training",
            "compute": "/v1/history/compute",
            "inference": "/v1/history/inference",
            "proxy": "/v1/history/proxy",
        },
        "params": ["since", "until", "status", "user", "owner_id", "limit", "offset", "order"],
        "note": "Admin only. inference/proxy default to the last "
                f"{DEFAULT_WINDOW_DAYS}d when `since` is omitted.",
        # Self-describing schema: every key a consumer can expect, per kind.
        "fields": {
            "envelope": {
                "kind": "record type: benchmark | training | compute | inference | proxy",
                "id": "platform id (bench-*/train-*/cmp-*/req-*/prx-*)",
                "name": "human name; for inference/proxy this is the served model",
                "user": "username of the owner ('(anonymous)' if none)",
                "owner_id": "numeric user id",
                "status": "lifecycle status (see each endpoint's note for its values)",
                "created_at": "ISO-8601 submission time",
                "started_at": "ISO-8601 execution start (null if never started)",
                "ended_at": "ISO-8601 terminal time (null if still running)",
                "duration_s": "ended_at - started_at (or created_at) in seconds",
                "error_text": "failure reason when status is failed/error",
                "detail": "kind-specific extras, see below",
            },
            "common_detail": {
                "provider_kind": "backend the job ran on: vm | runpod | pi; null = platform-default cloud",
                "provider_name": "display name of the registered provider account; null = platform default",
                "provider_id": "id of the registered provider account",
                "gpu_type": "GPU model requested/used (e.g. 'NVIDIA H100 80GB HBM3')",
                "gpu_count": "number of GPUs",
                "cost_per_hr": "hourly $ rate captured at spawn (null when unknown, e.g. own VM)",
                "visible_devices": "CUDA_VISIBLE_DEVICES pin requested for the job (null = all GPUs)",
            },
            "benchmark_detail": {
                "engine": "serving engine benchmarked (vllm | ...)",
                "base_url": "external OpenAI-compatible URL when benchmarking an already-running server",
                "backend": "vm | runpod | pi | external — where the run executed",
                "tensor_parallel_size": "vLLM TP degree from the serve config",
                "data_parallel_size": "vLLM DP degree from the serve config",
                "expert_parallel": "whether expert parallelism was enabled",
                "max_model_len": "context length the server was launched with",
                "exit_code": "benchmaq process exit code",
                "runpod_pod_id": "cloud pod id when spawned on RunPod",
                "storage_id": "storage backend the results were synced to",
                "result_json": "throughput/TTFT/TPOT/E2EL summary per concurrency",
            },
            "training_detail": {
                "base_model": "HF model the finetune started from",
                "task_type": "asr (Whisper) | tts (Qwen3+NeuCodec)",
                "dataset_id": "training dataset",
                "storage_id": "storage backend for logs/artifacts",
                "runpod_pod_id": "cloud pod id when spawned on RunPod",
                "exit_code": "trainer process exit code",
                "result_json": "per-epoch metrics (wer/cer/loss), best epoch, artifact URIs",
            },
            "compute_detail": {
                "image": "container image the pod booted",
                "cloud_type": "SECURE | COMMUNITY (RunPod tier)",
                "container_disk_gb": "ephemeral container disk size",
                "volume_gb": "persistent volume size",
                "public_ip": "node address SSH was exposed on (null until ready)",
                "ssh_port": "SSH port on public_ip",
                "runpod_pod_id": "cloud pod id when spawned on RunPod",
            },
            "inference_detail": {
                "app_id": "endpoint the request was served by",
                "model": "model name from the request payload",
                "endpoint": "API path used (/v1/chat/completions, /v1/audio/transcriptions, ...)",
                "is_stream": "whether the request was SSE streaming",
                "prompt_tokens": "input tokens (best-effort from vLLM usage; null for most streams)",
                "completion_tokens": "output tokens (best-effort; null for most streams)",
                "requested_gpu_type": "GPU type the endpoint is configured to run on",
                "requested_gpu_count": "GPUs per worker the endpoint requests",
                "worker": ("node that actually served the request (null for pre-upgrade rows or "
                           "requests that never reached a worker): {machine_id, hostname, gpu_name, "
                           "gpu_count, gpu_memory, driver_version, visible_devices (CUDA_VISIBLE_DEVICES "
                           "as seen by the worker), runpod_pod_id}"),
            },
            "proxy_detail": {
                "endpoint_id": "proxy endpoint id",
                "model": "model name requested",
                "upstream": "upstream base URL the proxy forwarded to",
                "status_code": "upstream HTTP status",
                "latency_ms": "upstream latency",
                "is_stream": "whether the request was streaming",
                "prompt_tokens": "input tokens reported by the upstream",
                "completion_tokens": "output tokens reported by the upstream",
            },
        },
    }


@router.get("/benchmarks", response_model=HistoryResponse)
async def history_benchmarks(
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
    since: Optional[str] = _Q_SINCE, until: Optional[str] = _Q_UNTIL,
    status: Optional[str] = _Q_STATUS, user: Optional[str] = _Q_USER,
    owner_id: Optional[int] = _Q_OWNER, limit: int = _Q_LIMIT,
    offset: int = _Q_OFFSET, order: str = _Q_ORDER,
):
    from .bench import Benchmark
    s_dt, u_dt = _parse_dt(since, "since"), _parse_dt(until, "until")
    owner = await _owner_filter(session, user, owner_id)
    rows, has_more = await _query_discrete(
        session, Benchmark, since=s_dt, until=u_dt, status=status, owner=owner,
        limit=limit, offset=offset, order=order)
    umap = await _user_map(session, [r.owner_id for r in rows])
    # VM benchmarks keep their GPU on the registered provider (set by its last
    # Test / nvidia-smi), NOT in config_yaml — resolve providers for this page so
    # VM runs report a gpu_type too, not just the RunPod path.
    provmap = await _prov_map(session, {r.provider_id for r in rows})

    jobs = []
    for r in rows:
        meta = _bench_meta(r.config_yaml)       # gpu_type, gpu_count, engine, TP/DP/EP, base_url, ...
        prov = provmap.get(r.provider_id) if r.provider_id else None
        if prov is not None and prov.kind == "vm":
            gpus = (prov.config or {}).get("gpus") or []
            if not meta["gpu_type"] and gpus:
                meta["gpu_type"] = gpus[0]
            if meta["gpu_count"] is None:
                meta["gpu_count"] = (prov.config or {}).get("gpu_count") or (len(gpus) or None)
        # backend the run executed on: vm | runpod | pi | external (base_url) | runpod (platform default)
        backend = prov.kind if prov is not None else ("external" if meta["base_url"] else "runpod")
        jobs.append(JobRecord(
            kind="benchmark", id=r.id, name=r.name, owner_id=r.owner_id,
            user=_username(umap, r.owner_id), status=r.status,
            created_at=_iso(r.created_at), started_at=_iso(r.started_at), ended_at=_iso(r.ended_at),
            duration_s=_duration_s(r.started_at or r.created_at, r.ended_at), error_text=r.error_text,
            detail={
                **meta,
                **_prov_fields(prov),
                "backend": backend,
                "visible_devices": r.visible_devices,
                "exit_code": r.exit_code, "cost_per_hr": r.cost_per_hr, "provider_id": r.provider_id,
                "storage_id": r.storage_id, "runpod_pod_id": r.runpod_pod_id, "result_json": r.result_json,
            },
        ))
    return HistoryResponse(
        kind="benchmark", count=len(jobs), has_more=has_more,
        window={"since": _iso(s_dt), "until": _iso(u_dt)}, jobs=jobs,
        note="Benchmark runs. status: queued|running|done|failed|cancelled.")


@router.get("/training", response_model=HistoryResponse)
async def history_training(
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
    since: Optional[str] = _Q_SINCE, until: Optional[str] = _Q_UNTIL,
    status: Optional[str] = _Q_STATUS, user: Optional[str] = _Q_USER,
    owner_id: Optional[int] = _Q_OWNER, limit: int = _Q_LIMIT,
    offset: int = _Q_OFFSET, order: str = _Q_ORDER,
):
    from .training_api import TrainingRun
    s_dt, u_dt = _parse_dt(since, "since"), _parse_dt(until, "until")
    owner = await _owner_filter(session, user, owner_id)
    rows, has_more = await _query_discrete(
        session, TrainingRun, since=s_dt, until=u_dt, status=status, owner=owner,
        limit=limit, offset=offset, order=order)
    umap = await _user_map(session, [r.owner_id for r in rows])
    provmap = await _prov_map(session, {r.provider_id for r in rows})
    jobs = [JobRecord(
        kind="training", id=r.id, name=r.name, owner_id=r.owner_id,
        user=_username(umap, r.owner_id), status=r.status,
        created_at=_iso(r.created_at), started_at=_iso(r.started_at), ended_at=_iso(r.ended_at),
        duration_s=_duration_s(r.started_at or r.created_at, r.ended_at), error_text=r.error_text,
        detail={
            "base_model": r.base_model, "task_type": r.task_type, "dataset_id": r.dataset_id,
            "gpu_type": r.gpu_type, "gpu_count": r.gpu_count, "cost_per_hr": r.cost_per_hr,
            **_prov_fields(provmap.get(r.provider_id)),
            "visible_devices": r.visible_devices, "storage_id": r.storage_id,
            "runpod_pod_id": r.runpod_pod_id,
            "exit_code": r.exit_code, "provider_id": r.provider_id, "result_json": r.result_json,
        },
    ) for r in rows]
    return HistoryResponse(
        kind="training", count=len(jobs), has_more=has_more,
        window={"since": _iso(s_dt), "until": _iso(u_dt)}, jobs=jobs,
        note="Autotrain runs. status: queued|running|done|failed|cancelled|paused.")


@router.get("/compute", response_model=HistoryResponse)
async def history_compute(
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
    since: Optional[str] = _Q_SINCE, until: Optional[str] = _Q_UNTIL,
    status: Optional[str] = _Q_STATUS, user: Optional[str] = _Q_USER,
    owner_id: Optional[int] = _Q_OWNER, limit: int = _Q_LIMIT,
    offset: int = _Q_OFFSET, order: str = _Q_ORDER,
):
    from .compute import ComputePod
    s_dt, u_dt = _parse_dt(since, "since"), _parse_dt(until, "until")
    owner = await _owner_filter(session, user, owner_id)
    rows, has_more = await _query_discrete(
        session, ComputePod, since=s_dt, until=u_dt, status=status, owner=owner,
        limit=limit, offset=offset, order=order)
    umap = await _user_map(session, [r.owner_id for r in rows])
    provmap = await _prov_map(session, {r.provider_id for r in rows})
    jobs = [JobRecord(
        kind="compute", id=r.id, name=r.name, owner_id=r.owner_id,
        user=_username(umap, r.owner_id), status=r.status,
        created_at=_iso(r.created_at), started_at=_iso(r.ready_at), ended_at=_iso(r.terminated_at),
        duration_s=_duration_s(r.ready_at, r.terminated_at), error_text=r.error_text,
        detail={
            "gpu_type": r.gpu_type, "gpu_count": r.gpu_count, "image": r.image,
            "cloud_type": r.cloud_type, "cost_per_hr": r.cost_per_hr, "provider_id": r.provider_id,
            **_prov_fields(provmap.get(r.provider_id)),
            "container_disk_gb": r.container_disk_gb, "volume_gb": r.volume_gb,
            "public_ip": r.public_ip, "ssh_port": r.ssh_port,
            "runpod_pod_id": r.runpod_pod_id,
        },
    ) for r in rows]
    return HistoryResponse(
        kind="compute", count=len(jobs), has_more=has_more,
        window={"since": _iso(s_dt), "until": _iso(u_dt)}, jobs=jobs,
        note="Compute pods. status: creating|running|failed|terminated|pending_approval|rejected.")


@router.get("/inference", response_model=HistoryResponse)
async def history_inference(
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
    since: Optional[str] = _Q_SINCE, until: Optional[str] = _Q_UNTIL,
    status: Optional[str] = _Q_STATUS, user: Optional[str] = _Q_USER,
    owner_id: Optional[int] = _Q_OWNER, limit: int = _Q_LIMIT,
    offset: int = _Q_OFFSET, order: str = _Q_ORDER,
):
    s_dt, u_dt = _parse_dt(since, "since"), _parse_dt(until, "until")
    if s_dt is None:  # bound the scan over the high-volume requests table
        s_dt = datetime.now(timezone.utc) - timedelta(days=DEFAULT_WINDOW_DAYS)
    owner = await _owner_filter(session, user, owner_id)

    # Project to small columns only; pull `model` + token counts out of the JSON
    # in SQL so the multi-KB payload/output blobs never enter memory.
    model_col = func.json_extract_path_text(ReqRow.payload, "model").label("model")
    tin_col = func.json_extract_path_text(ReqRow.output, "usage", "prompt_tokens").label("tin")
    tout_col = func.json_extract_path_text(ReqRow.output, "usage", "completion_tokens").label("tout")
    st = select(
        ReqRow.request_id, ReqRow.app_id, ReqRow.owner_id, ReqRow.endpoint,
        ReqRow.status, ReqRow.is_stream, ReqRow.created_at, ReqRow.completed_at,
        ReqRow.worker_meta,
        model_col, tin_col, tout_col,
    ).where(ReqRow.created_at >= s_dt)
    if u_dt is not None:
        st = st.where(ReqRow.created_at < u_dt)
    wanted = _status_list(status)
    if wanted:
        st = st.where(ReqRow.status.in_(wanted))
    if owner is not None:
        st = st.where(ReqRow.owner_id == owner)
    st = st.order_by(ReqRow.created_at.asc() if order == "asc" else ReqRow.created_at.desc())
    rows = (await session.execute(st.offset(offset).limit(limit + 1))).all()
    has_more = len(rows) > limit
    rows = rows[:limit]
    umap = await _user_map(session, [r.owner_id for r in rows])
    # The endpoint's configured GPU shape — what was *requested*; the worker_meta
    # blob (stamped by the worker that served the request) is what *actually ran*.
    app_ids = {r.app_id for r in rows if r.app_id}
    appmap: dict[str, App] = {}
    if app_ids:
        for a in (await session.execute(select(App).where(App.app_id.in_(app_ids)))).scalars().all():
            appmap[a.app_id] = a
    # Provider attribution comes from the serving app — same provider_kind /
    # provider_name fields the discrete job kinds carry.
    provmap = await _prov_map(session, {a.provider_id for a in appmap.values()})
    jobs = []
    for r in rows:
        a = appmap.get(r.app_id)
        jobs.append(JobRecord(
            kind="inference", id=r.request_id, name=r.model, owner_id=r.owner_id,
            user=_username(umap, r.owner_id), status=r.status,
            created_at=_iso(r.created_at), started_at=None, ended_at=_iso(r.completed_at),
            duration_s=_duration_s(r.created_at, r.completed_at), error_text=None,
            detail={
                "app_id": r.app_id, "model": r.model, "endpoint": r.endpoint, "is_stream": r.is_stream,
                "prompt_tokens": _int(r.tin), "completion_tokens": _int(r.tout),
                "requested_gpu_type": a.gpu if a else None,
                "requested_gpu_count": a.gpu_count if a else None,
                **_prov_fields(provmap.get(a.provider_id) if a else None),
                "worker": r.worker_meta,
            },
        ))
    return HistoryResponse(
        kind="inference", count=len(jobs), has_more=has_more,
        window={"since": _iso(s_dt), "until": _iso(u_dt)}, jobs=jobs,
        note=(f"Serverless inference requests (defaults to the last {DEFAULT_WINDOW_DAYS}d when "
              "`since` is omitted; request/response bodies are not included). "
              "status: pending|queued|running|completed|failed|cancelled."))


@router.get("/proxy", response_model=HistoryResponse)
async def history_proxy(
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
    since: Optional[str] = _Q_SINCE, until: Optional[str] = _Q_UNTIL,
    status: Optional[str] = _Q_STATUS, user: Optional[str] = _Q_USER,
    owner_id: Optional[int] = _Q_OWNER, limit: int = _Q_LIMIT,
    offset: int = _Q_OFFSET, order: str = _Q_ORDER,
):
    from .proxy_api import ProxyRequest
    s_dt, u_dt = _parse_dt(since, "since"), _parse_dt(until, "until")
    if s_dt is None:  # high-volume table — bound the scan
        s_dt = datetime.now(timezone.utc) - timedelta(days=DEFAULT_WINDOW_DAYS)
    owner = await _owner_filter(session, user, owner_id)
    st = select(ProxyRequest).where(ProxyRequest.created_at >= s_dt)
    if u_dt is not None:
        st = st.where(ProxyRequest.created_at < u_dt)
    wanted = _status_list(status)
    if wanted:
        st = st.where(ProxyRequest.status.in_(wanted))
    if owner is not None:
        st = st.where(ProxyRequest.owner_id == owner)
    st = st.order_by(ProxyRequest.created_at.asc() if order == "asc" else ProxyRequest.created_at.desc())
    rows = (await session.execute(st.offset(offset).limit(limit + 1))).scalars().all()
    has_more = len(rows) > limit
    rows = rows[:limit]
    umap = await _user_map(session, [r.owner_id for r in rows])
    jobs = [JobRecord(
        kind="proxy", id=r.id, name=r.model, owner_id=r.owner_id,
        user=_username(umap, r.owner_id), status=r.status,
        created_at=_iso(r.created_at), started_at=_iso(r.started_at), ended_at=_iso(r.completed_at),
        duration_s=_duration_s(r.started_at, r.completed_at), error_text=r.error_text,
        detail={
            "endpoint_id": r.endpoint_id, "model": r.model, "upstream": r.upstream,
            "status_code": r.status_code, "latency_ms": r.latency_ms, "is_stream": r.is_stream,
            "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
        },
    ) for r in rows]
    return HistoryResponse(
        kind="proxy", count=len(jobs), has_more=has_more,
        window={"since": _iso(s_dt), "until": _iso(u_dt)}, jobs=jobs,
        note=(f"LLM-proxy requests (defaults to the last {DEFAULT_WINDOW_DAYS}d when `since` is "
              "omitted). status: queued|running|completed|cancelled|failed."))


# ── inference summary ────────────────────────────────────────────────────────
# The requests table is far too large to ship row-by-row to a dashboard
# (raw endpoints cap out and undercount). For serverless, analytics only needs
# *creation counts*, so this aggregates in SQL — exact over any window, a few
# hundred rows out.

class InferenceSummaryRow(BaseModel):
    date: str                       # YYYY-MM-DD in the requested tz
    app_id: Optional[str]
    user: Optional[str]
    status: str
    provider_kind: Optional[str]
    provider_name: Optional[str]
    count: int


class InferenceSummaryResponse(BaseModel):
    kind: str = "inference_summary"
    window: dict[str, Any]
    rows: list[InferenceSummaryRow]
    note: str


@router.get("/summary", response_model=InferenceSummaryResponse)
async def history_inference_summary(
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
    since: Optional[str] = _Q_SINCE, until: Optional[str] = _Q_UNTIL,
    tz: str = Query("UTC", description="IANA timezone for day bucketing"),
):
    from zoneinfo import ZoneInfo
    try:
        ZoneInfo(tz)
    except Exception:
        raise HTTPException(status_code=400, detail=f"unknown timezone: {tz}")
    s_dt, u_dt = _parse_dt(since, "since"), _parse_dt(until, "until")
    if s_dt is None:
        s_dt = datetime.now(timezone.utc) - timedelta(days=DEFAULT_WINDOW_DAYS)
    day = func.to_char(func.timezone(tz, ReqRow.created_at), "YYYY-MM-DD").label("day")
    st = (
        select(day, ReqRow.app_id, ReqRow.owner_id, ReqRow.status, func.count().label("n"))
        .where(ReqRow.created_at >= s_dt)
    )
    if u_dt is not None:
        st = st.where(ReqRow.created_at < u_dt)
    st = st.group_by(day, ReqRow.app_id, ReqRow.owner_id, ReqRow.status)
    rows = (await session.execute(st)).all()

    umap = await _user_map(session, [r.owner_id for r in rows])
    app_ids = {r.app_id for r in rows if r.app_id}
    appmap: dict[str, App] = {}
    if app_ids:
        for a in (await session.execute(select(App).where(App.app_id.in_(app_ids)))).scalars().all():
            appmap[a.app_id] = a
    provmap = await _prov_map(session, {a.provider_id for a in appmap.values()})

    out = []
    for r in rows:
        a = appmap.get(r.app_id)
        prov = provmap.get(a.provider_id) if a else None
        out.append(InferenceSummaryRow(
            date=r.day, app_id=r.app_id, user=_username(umap, r.owner_id),
            status=r.status, count=r.n, **_prov_fields(prov),
        ))
    out.sort(key=lambda x: (x.date, x.app_id or "", x.status))
    return InferenceSummaryResponse(
        window={"since": _iso(s_dt), "until": _iso(u_dt)}, rows=out,
        note=("Exact creation counts for serverless inference requests, grouped by "
              "day/app/user/status with provider attribution — use instead of paging "
              f"/inference for analytics (defaults to the last {DEFAULT_WINDOW_DAYS}d)."))
