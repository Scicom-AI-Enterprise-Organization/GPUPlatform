"""Prometheus metrics for the gateway.

Exposed at GET /metrics (auth-exempt — scrapers don't have keys; protect via
network/ingress allowlist if needed).

Counters/gauges defined:
  - gateway_requests_total{route, status}
  - gateway_inflight_requests          (point-in-time, gauge)
  - gateway_queue_length{app_id}       (sampled per-scrape from Redis)
  - gateway_workers{app_id, status}    (sampled per-scrape from Redis)
  - gateway_provision_total{provider, ok}
  - gateway_terminate_total{provider, ok}

Serverless-API HTTP instrumentation (the gateway's OWN request layer — NOT the
vLLM worker metrics, which are shipped separately by the workers):
  - serverless_http_requests_total{method, route, http_status, app_id}
  - serverless_http_request_duration_seconds{method, route, http_status, app_id}  (histogram)
`route` is the matched ROUTE TEMPLATE (e.g. "/{app_id}/v1/chat/completions"),
never the raw URL, so app_ids / request bodies can't blow up label cardinality.
`app_id` is the path param for scoped routes (empty for global/control-plane ones),
so `GET /{app_id}/metrics` can serve a per-fleet view (see render_app) for Grafana
to scrape and alert on, e.g., non-2xx responses.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import Metric

if TYPE_CHECKING:
    import redis.asyncio as redis_async


_registry = CollectorRegistry()


REQUESTS_TOTAL = Counter(
    "gateway_requests_total",
    "Total HTTP requests handled by the gateway",
    ["route", "status"],
    registry=_registry,
)

INFLIGHT = Gauge(
    "gateway_inflight_requests",
    "Number of requests currently being handled by the gateway",
    registry=_registry,
)

QUEUE_LENGTH = Gauge(
    "gateway_queue_length",
    "Length of the per-app job queue at scrape time",
    ["app_id"],
    registry=_registry,
)

WORKERS_TOTAL = Gauge(
    "gateway_workers",
    "Number of live workers per app at scrape time",
    ["app_id"],
    registry=_registry,
)

PROVISION_TOTAL = Counter(
    "gateway_provision_total",
    "Worker provision attempts",
    ["provider", "ok"],
    registry=_registry,
)

TERMINATE_TOTAL = Counter(
    "gateway_terminate_total",
    "Worker terminate attempts",
    ["provider", "ok"],
    registry=_registry,
)


# ---- Serverless-API HTTP metrics (the gateway's own request layer) ----------
# Buckets are extended past prometheus_client's defaults (which top out at 10s)
# because the inference proxy endpoints block on a worker result and can take
# tens of seconds; without the longer buckets every slow inference call lands in
# +Inf and you lose all latency resolution above 10s.
_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
    10.0, 30.0, 60.0, 120.0, float("inf"),
)

# Label is `route` (not `endpoint`): the kube scrape attaches its own target
# label `endpoint` = the Service port name ("http"), which would shadow an app
# label of the same name (it lands as `exported_endpoint`). `route` avoids that.
_HTTP_LABELS = ["method", "route", "http_status", "app_id"]

# `serverless_*` namespace so these don't collide with the vLLM workers' own
# `http_requests_total` / `http_request_duration_seconds` (both scraped into the
# same VictoriaMetrics).
HTTP_REQUESTS_TOTAL = Counter(
    "serverless_http_requests_total",
    "Total serverless gateway HTTP requests",
    _HTTP_LABELS,
    registry=_registry,
)

HTTP_REQUEST_DURATION = Histogram(
    "serverless_http_request_duration_seconds",
    "Serverless gateway HTTP request latency",
    _HTTP_LABELS,
    buckets=_LATENCY_BUCKETS,
    registry=_registry,
)

# Job OUTCOMES, as opposed to HTTP statuses: async requests (/run/{app_id})
# return HTTP 200 at enqueue, so a job that later fails/times out (worker died,
# queue eaten, model wake OOM) never shows as a 5xx in the HTTP counters. This
# counts terminal job-status transitions instead — bumped wherever a Redis
# result is mirrored into the requests table. Lazy mirroring means a failed job
# is counted when first observed (result poll / queue page / orphan sweep), not
# at the instant it failed.
JOBS_TOTAL = Counter(
    "serverless_jobs_total",
    "Terminal inference job outcomes (completed / failed / timeout / cancelled)",
    ["app_id", "status"],
    registry=_registry,
)


def observe_job_outcome(app_id: str, status: str) -> None:
    """Record one job reaching a terminal status. Caller guarantees this is a
    real transition (old status != new status) so outcomes aren't double-counted."""
    if status and status != "pending":
        JOBS_TOTAL.labels(app_id=app_id or "", status=status).inc()


# Route TEMPLATES excluded from the HTTP instrumentation — health/readiness probes
# and the metrics scrapes themselves (global + per-app), which would otherwise
# drown out real API traffic. Override with METRICS_IGNORE_PATHS (comma-separated
# route templates). The serverless API surface (/v1/*, /{app_id}/v1/*) is exactly
# what we DO want to measure.
IGNORE_PATHS = {
    p.strip()
    for p in os.environ.get(
        "METRICS_IGNORE_PATHS",
        "/metrics,/metrics/resources,/{app_id}/metrics,/proxy/{endpoint}/metrics,/health,/ready,/version,/",
    ).split(",")
    if p.strip()
}


def observe_http(
    method: str, endpoint: str, status: int, app_id: str, duration_s: float
) -> None:
    """Record one gateway HTTP request: bump the count and observe its latency.
    `endpoint` must be the matched route TEMPLATE (caller collapses unmatched
    paths) so label cardinality stays bounded; `app_id` is the route's path param
    (empty string for global/control-plane routes)."""
    s = str(status)
    HTTP_REQUESTS_TOTAL.labels(
        method=method, route=endpoint, http_status=s, app_id=app_id
    ).inc()
    HTTP_REQUEST_DURATION.labels(
        method=method, route=endpoint, http_status=s, app_id=app_id
    ).observe(duration_s)


def render_app(app_id: str) -> bytes:
    """Prometheus exposition of ONE app's gateway HTTP metrics — appended to
    `GET /{app_id}/metrics` so Grafana/Prometheus can scrape per-fleet (e.g.
    /tm-fleet/metrics) and alert on non-2xx. Re-emits the http_* series filtered
    to this app_id through a throwaway registry (so HELP/TYPE/formatting are
    handled by generate_latest)."""

    class _AppFilter:
        def collect(self):
            for metric in (HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION):
                for mf in metric.collect():
                    samples = [s for s in mf.samples if s.labels.get("app_id") == app_id]
                    if not samples:
                        continue
                    m = Metric(mf.name, mf.documentation, mf.type)
                    m.samples = samples
                    yield m

    reg = CollectorRegistry()
    reg.register(_AppFilter())
    return generate_latest(reg)


# ---- LLM-proxy metrics (the /proxy/{name} router) ---------------------------
# The proxy funnels every request through _handle/_finish, not the OpenAI app
# routes, so it gets its own series (labelled by the endpoint id, the model alias
# the client sent, the upstream that actually served it, and the outcome).
_PROXY_LABELS = ["proxy", "model", "upstream", "status"]
PROXY_REQUESTS_TOTAL = Counter(
    "proxy_requests_total",
    "LLM-proxy requests routed, by endpoint / model alias / chosen upstream / outcome",
    _PROXY_LABELS,
    registry=_registry,
)
PROXY_REQUEST_DURATION = Histogram(
    "proxy_request_duration_seconds",
    "LLM-proxy request latency (gateway-observed, end to end)",
    ["proxy", "model"],
    buckets=_LATENCY_BUCKETS,
    registry=_registry,
)


def observe_proxy(
    proxy: str, model: str, upstream: str, status: str, duration_s: float | None = None
) -> None:
    """Record one proxied request: bump the outcome counter and, when the request
    actually ran (latency known), observe its latency. `proxy` is the endpoint id;
    `status` is the terminal state (completed / failed / cancelled / timeout / …)."""
    PROXY_REQUESTS_TOTAL.labels(
        proxy=proxy, model=model or "", upstream=upstream or "", status=status
    ).inc()
    if duration_s is not None:
        PROXY_REQUEST_DURATION.labels(proxy=proxy, model=model or "").observe(duration_s)


def render_proxy(proxy: str) -> bytes:
    """Prometheus exposition of ONE proxy endpoint's metrics, filtered to its id —
    served at `GET /proxy/{name}/metrics`, the sibling of `/{app_id}/metrics`."""

    class _ProxyFilter:
        def collect(self):
            for metric in (PROXY_REQUESTS_TOTAL, PROXY_REQUEST_DURATION):
                for mf in metric.collect():
                    samples = [s for s in mf.samples if s.labels.get("proxy") == proxy]
                    if not samples:
                        continue
                    m = Metric(mf.name, mf.documentation, mf.type)
                    m.samples = samples
                    yield m

    reg = CollectorRegistry()
    reg.register(_ProxyFilter())
    return generate_latest(reg)


# ---- Platform resource metrics (sampled from Postgres per scrape) -----------
# Served at GET /api/metrics as a business/ops exporter — a SEPARATE registry so
# it stays independent of the infra `/metrics` target (which is Redis + HTTP).
# The autotrain series are the priority: alert in Grafana on a failed train via
#   platform_autotrain_runs{status="failed"} > 0            (simple, always present)
#   platform_autotrain_run_info{status="failed"} == 1       (per-run, labelled name/owner)
_resource_registry = CollectorRegistry()

RES_APPS = Gauge(
    "platform_serverless_apps", "Serverless inference endpoints", registry=_resource_registry
)
RES_APP_INFO = Gauge(
    "platform_serverless_app_info",
    "One series per serverless endpoint (value 1); model/gpu/mode as labels",
    ["app_id", "model", "gpu", "mode"],
    registry=_resource_registry,
)
RES_DATASETS = Gauge(
    "platform_datasets", "Autotrain datasets by kind", ["kind"], registry=_resource_registry
)
RES_DATASET_ROWS = Gauge(
    "platform_dataset_rows", "Total rows across all datasets (sum of num_rows)", registry=_resource_registry
)
RES_DATASET_BYTES = Gauge(
    "platform_dataset_bytes", "Total dataset size in bytes (sum of size_bytes)", registry=_resource_registry
)
RES_STORAGE = Gauge(
    "platform_storage_backends", "Storage backends by kind", ["kind"], registry=_resource_registry
)
RES_PROVIDERS = Gauge(
    "platform_gpu_providers", "GPU providers by kind", ["kind"], registry=_resource_registry
)
RES_COMPUTE = Gauge(
    "platform_compute_pods", "Compute (notebook) pods by status", ["status"], registry=_resource_registry
)
RES_USERS = Gauge(
    "platform_users", "Platform users by role", ["role"], registry=_resource_registry
)
RES_BENCH = Gauge(
    "platform_benchmarks", "Benchmarks by status", ["status"], registry=_resource_registry
)
RES_RUNS = Gauge(
    "platform_autotrain_runs", "Autotrain training runs by status and task", ["status", "task"], registry=_resource_registry
)
RES_RUN_INFO = Gauge(
    "platform_autotrain_run_info",
    "One series per autotrain run (value 1); status carried as a label for per-run alerting",
    ["run_id", "name", "task", "status", "owner"],
    registry=_resource_registry,
)
RES_RUN_RUNNING_SECONDS = Gauge(
    "platform_autotrain_run_running_seconds",
    "Seconds a currently-RUNNING autotrain run has been going — alert on stuck runs, "
    "e.g. platform_autotrain_run_running_seconds > 14400 (4h)",
    ["run_id", "name", "task", "owner"],
    registry=_resource_registry,
)
RES_GITOPS = Gauge(
    "platform_gitops_repos", "GitOps repos by last sync status", ["sync_status"], registry=_resource_registry
)
RES_GITOPS_RESOURCES = Gauge(
    "platform_gitops_resources", "GitOps-managed resources by kind and status", ["kind", "status"], registry=_resource_registry
)

# Monotonic counter bumped ONCE when a training run reaches a terminal state
# (done/failed) — incremented at the finalize transition, NOT sampled. This is
# the alert-friendly autotrain failure signal: unlike the cumulative
# `platform_autotrain_runs{status="failed"}` gauge (which is >0 forever once any
# run has ever failed), this fires only on NEW failures and auto-resolves:
#   increase(platform_autotrain_runs_finished_total{status="failed"}[10m]) > 0
AUTOTRAIN_RUNS_FINISHED = Counter(
    "platform_autotrain_runs_finished_total",
    "Autotrain runs that reached a terminal state, counted once at the transition",
    ["status", "task"],
    registry=_resource_registry,
)
# Pre-create the common failure series at 0 so dashboards/alerts have a baseline
# before the first post-restart failure (instead of evaluating to no-data).
for _t in ("asr", "tts"):
    AUTOTRAIN_RUNS_FINISHED.labels(status="failed", task=_t)

# Always emit these statuses (0 when none) so alert expressions like
# `platform_autotrain_runs{status="failed"} > 0` always have a series to evaluate
# instead of going to no-data when the platform has never had a failure.
_RUN_STATUSES = ("queued", "running", "done", "failed", "cancelled")
_BENCH_STATUSES = ("queued", "running", "done", "failed", "cancelled")
# Bound the per-run info cardinality to the most recent runs.
_RUN_INFO_LIMIT = 500


async def sample_resources(session) -> None:
    """Sample platform resource counts + per-run autotrain state from Postgres
    into the resource registry. Cheap COUNT/GROUP BY queries; called per scrape."""
    from sqlalchemy import func, select

    from .bench import Benchmark
    from .compute import ComputePod
    from .db import App, Dataset, Provider, Storage, User
    from .training_api import TrainingRun
    try:
        from .db import GitopsRepo, GitopsResource
    except Exception:  # pragma: no cover — gitops tables optional
        GitopsRepo = GitopsResource = None

    async def _count(model) -> int:
        return int((await session.execute(select(func.count()).select_from(model))).scalar() or 0)

    async def _group(gauge, col, label: str, *, default_label: str = "unknown") -> None:
        gauge.clear()
        for key, n in (await session.execute(select(col, func.count()).group_by(col))).all():
            gauge.labels(**{label: key or default_label}).set(int(n))

    # ---- serverless apps (count + per-app info) ----
    RES_APPS.set(await _count(App))
    RES_APP_INFO.clear()
    for a in (await session.execute(select(App))).scalars().all():
        RES_APP_INFO.labels(
            app_id=a.app_id, model=(a.model or ""), gpu=(a.gpu or ""),
            mode=(getattr(a, "mode", "single") or "single"),
        ).set(1)

    # ---- datasets (by kind + total rows/bytes) ----
    await _group(RES_DATASETS, Dataset.kind, "kind")
    RES_DATASET_ROWS.set(int((await session.execute(
        select(func.coalesce(func.sum(Dataset.num_rows), 0))
    )).scalar() or 0))
    RES_DATASET_BYTES.set(int((await session.execute(
        select(func.coalesce(func.sum(Dataset.size_bytes), 0))
    )).scalar() or 0))

    # ---- storage / providers / compute pods / users ----
    await _group(RES_STORAGE, Storage.kind, "kind")
    await _group(RES_PROVIDERS, Provider.kind, "kind")
    await _group(RES_COMPUTE, ComputePod.status, "status")
    await _group(RES_USERS, User.role, "role")

    # ---- benchmarks (always emit the known statuses, 0 when none) ----
    bench_counts = {s: 0 for s in _BENCH_STATUSES}
    for status, n in (await session.execute(
        select(Benchmark.status, func.count()).group_by(Benchmark.status)
    )).all():
        bench_counts[status or "unknown"] = int(n)
    RES_BENCH.clear()
    for status, n in bench_counts.items():
        RES_BENCH.labels(status=status).set(n)

    # ---- autotrain run counts by (status, task) ----
    run_counts: dict[tuple[str, str], int] = {}
    for status, task, n in (await session.execute(
        select(TrainingRun.status, TrainingRun.task_type, func.count())
        .group_by(TrainingRun.status, TrainingRun.task_type)
    )).all():
        run_counts[(status or "unknown", task or "")] = int(n)
    for t in ("asr", "tts"):  # keep failed/{asr,tts} present for clean alerting
        run_counts.setdefault(("failed", t), 0)
    RES_RUNS.clear()
    for (status, task), n in run_counts.items():
        RES_RUNS.labels(status=status, task=task).set(n)

    # ---- per-run info + running-duration (most recent runs) ----
    now = datetime.now(timezone.utc)
    rows = (await session.execute(
        select(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(_RUN_INFO_LIMIT)
    )).scalars().all()
    owners: dict[int, str] = {}
    owner_ids = {r.owner_id for r in rows}
    if owner_ids:
        for u in (await session.execute(select(User).where(User.id.in_(owner_ids)))).scalars().all():
            owners[u.id] = u.username
    RES_RUN_INFO.clear()
    RES_RUN_RUNNING_SECONDS.clear()
    for r in rows:
        owner = owners.get(r.owner_id, "?")
        RES_RUN_INFO.labels(
            run_id=r.id, name=r.name or r.id, task=r.task_type or "",
            status=r.status or "unknown", owner=owner,
        ).set(1)
        if r.status == "running":
            start = r.started_at or r.created_at
            if start is not None:
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                RES_RUN_RUNNING_SECONDS.labels(
                    run_id=r.id, name=r.name or r.id, task=r.task_type or "", owner=owner,
                ).set(max(0.0, (now - start).total_seconds()))

    # ---- gitops (repos by sync status + managed resources by kind/status) ----
    if GitopsRepo is not None:
        await _group(RES_GITOPS, GitopsRepo.last_sync_status, "sync_status", default_label="never")
        RES_GITOPS_RESOURCES.clear()
        for kind, status, n in (await session.execute(
            select(GitopsResource.kind, GitopsResource.status, func.count())
            .group_by(GitopsResource.kind, GitopsResource.status)
        )).all():
            RES_GITOPS_RESOURCES.labels(kind=kind or "unknown", status=status or "unknown").set(int(n))


async def render_resources(session) -> tuple[bytes, str]:
    """Sample Postgres resources, then serialize the resource registry. Backs
    GET /api/metrics."""
    await sample_resources(session)
    return generate_latest(_resource_registry), CONTENT_TYPE_LATEST


async def render(rdb: "redis_async.Redis") -> tuple[bytes, str]:
    """Sample point-in-time gauges from Redis, then serialize the registry."""
    # Discover app_ids from existing worker_index/queue keys (Postgres is the
    # source of truth for app metadata, but for metrics we just want anything
    # with active state in Redis).
    app_ids: set[str] = set()
    async for key in rdb.scan_iter(match="worker_index:*"):
        app_ids.add(key.split(":", 1)[1])
    async for key in rdb.scan_iter(match="queue:*"):
        app_ids.add(key.split(":", 1)[1])

    QUEUE_LENGTH.clear()
    WORKERS_TOTAL.clear()
    for app_id in app_ids:
        QUEUE_LENGTH.labels(app_id=app_id).set(await rdb.llen(f"queue:{app_id}"))
        members = await rdb.smembers(f"worker_index:{app_id}")
        live = 0
        for mid in members:
            if await rdb.exists(f"worker:{mid}"):
                live += 1
        WORKERS_TOTAL.labels(app_id=app_id).set(live)

    return generate_latest(_registry), CONTENT_TYPE_LATEST
