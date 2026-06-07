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

# Route TEMPLATES excluded from the HTTP instrumentation — health/readiness probes
# and the metrics scrapes themselves (global + per-app), which would otherwise
# drown out real API traffic. Override with METRICS_IGNORE_PATHS (comma-separated
# route templates). The serverless API surface (/v1/*, /{app_id}/v1/*) is exactly
# what we DO want to measure.
IGNORE_PATHS = {
    p.strip()
    for p in os.environ.get(
        "METRICS_IGNORE_PATHS",
        "/metrics,/{app_id}/metrics,/health,/ready,/version,/",
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
