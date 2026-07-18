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
import logging
import os
import time
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


logger = logging.getLogger("gateway.metrics")

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


# ---- Gateway runtime health -------------------------------------------------
# Self-observability for the control plane itself: is Redis reachable, is the DB
# pool saturating, are the background loops actually ticking, is the stats writer
# shedding. Sampled per scrape (Redis/pool/queue-depth) or bumped by the runtime
# (heartbeats, exception counter). Alert rules over these live in
# deploy/monitoring/prometheus/alerts.yml + the Helm PrometheusRule.

# Unix timestamp of each background loop's last completed tick. Loops call
# metrics.loop_heartbeat(<name>) at the END of a successful tick, so a loop that
# is wedged (or died without cancelling) goes stale and
#   time() - max by (loop) (gateway_loop_last_tick_timestamp_seconds) > <threshold>
# catches it. In HA mode only the leader ticks the leader-only loops — aggregate
# with max() across replicas so a clean failover doesn't page.
LOOP_LAST_TICK = Gauge(
    "gateway_loop_last_tick_timestamp_seconds",
    "Unix time of each background loop's last completed tick",
    ["loop"],
    registry=_registry,
)


def loop_heartbeat(loop: str) -> None:
    """Stamp a background loop's liveness. Call at the end of each successful
    tick (never from an except branch — a permanently-failing loop should read
    as stalled, not alive)."""
    LOOP_LAST_TICK.labels(loop=loop).set_to_current_time()


# Bumped by the global exception handler in main.py — every 500 that came from an
# UNHANDLED exception (as opposed to a deliberate HTTPException). Any sustained
# increase is a bug, not load.
UNHANDLED_EXCEPTIONS = Counter(
    "gateway_unhandled_exceptions_total",
    "Unhandled exceptions that reached the global handler (returned as 500)",
    ["route"],
    registry=_registry,
)

# Redis reachability, sampled at scrape time by render(). When Redis is down the
# scrape still succeeds (render degrades instead of raising) with this at 0 —
# monitoring must not go blind exactly when the hot path's store dies.
REDIS_UP = Gauge(
    "gateway_redis_up",
    "1 if Redis answered PING during the last /metrics sample, else 0",
    registry=_registry,
)
REDIS_PING_SECONDS = Gauge(
    "gateway_redis_ping_seconds",
    "Redis PING round-trip observed during the last /metrics sample",
    registry=_registry,
)

# SQLAlchemy pool pressure (per replica). checked_out at/near capacity for
# minutes = the pool-exhaustion incident shape — alert before handlers block on
# pool_timeout.
DB_POOL_CHECKED_OUT = Gauge(
    "gateway_db_pool_checked_out",
    "DB connections currently checked out of the SQLAlchemy pool",
    registry=_registry,
)
DB_POOL_CAPACITY = Gauge(
    "gateway_db_pool_capacity",
    "Max DB connections the pool can hand out (pool_size + max_overflow)",
    registry=_registry,
)

# Stats-writer backpressure: queue depth (sampled) + total dropped intents
# (bumped by stats_writer._enqueue on overflow). Sustained drops mean the
# Activity dashboard is silently losing rows — raise STATS_FLUSH_MAX_BATCH.
STATS_WRITER_QUEUE_DEPTH = Gauge(
    "gateway_stats_writer_queue_depth",
    "Pending stats-writer intents awaiting the batch flush",
    registry=_registry,
)
STATS_WRITER_DROPPED = Counter(
    "gateway_stats_writer_dropped_total",
    "Stats-writer intents dropped because the bounded queue was full",
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
# Shared by the serverless + proxy TTFT/TPS histograms. TTFT (seconds) spans
# sub-100ms to cold-start tens-of-seconds; TPS is output throughput (completion
# tokens / generation time).
_TTFT_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0, float("inf"),
)
_TPS_BUCKETS = (
    1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, float("inf"),
)
# Audio (whisper) mean token NEGATIVE log-likelihood = -avg_logprob (≥ 0), from
# the upstream's `verbose_json` segments. We store the NEGATED logprob on purpose:
# prometheus_client omits a histogram's `_sum` sample once it observes a negative
# value (the spec bans rate() over a sum that can decrease), which would break the
# windowed-mean drift query. NLL is non-negative, so `_sum` is emitted and the mean
# is rate(_sum)/rate(_count). LOWER NLL = more confident; a persistent RISE is the
# model-drift / degradation signal. Buckets (`le` upper bounds, ascending): a
# healthy whisper turbo sits around 0.15..0.30, so resolution is densest there and
# coarsens out toward hallucination (> 1).
_NLL_BUCKETS = (
    0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, float("inf"),
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

# Streamed-inference quality metrics, gateway-observed per request (the stream
# relay measures time-to-first-token and the generation window). Labelled by app
# + the model alias the client sent, so a multi-model fleet splits per member.
_STREAM_LABELS = ["app_id", "model"]
SERVERLESS_TTFT = Histogram(
    "serverless_ttft_seconds",
    "Serverless time-to-first-token (streamed requests)",
    _STREAM_LABELS,
    buckets=_TTFT_BUCKETS,
    registry=_registry,
)
SERVERLESS_TPS = Histogram(
    "serverless_tokens_per_second",
    "Serverless output throughput (completion tokens / generation time)",
    _STREAM_LABELS,
    buckets=_TPS_BUCKETS,
    registry=_registry,
)


def observe_serverless_stream(
    app_id: str, model: str, ttft_s: float | None = None, tps: float | None = None
) -> None:
    """Record a completed streamed serverless request's TTFT / output throughput.
    Both are optional (None → not observed); only meaningful for completed streams."""
    if ttft_s is not None:
        SERVERLESS_TTFT.labels(app_id=app_id, model=model or "").observe(ttft_s)
    if tps is not None:
        SERVERLESS_TPS.labels(app_id=app_id, model=model or "").observe(tps)

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
        "/metrics,/metrics/resources,/{app_id}/metrics,/proxy/{endpoint}/metrics,"
        "/proxy/{endpoint}/health,/proxy/{endpoint}/healthz,/health,/ready,/version,/",
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
            for metric in (HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION, SERVERLESS_TTFT, SERVERLESS_TPS):
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
PROXY_TTFT = Histogram(
    "proxy_ttft_seconds",
    "LLM-proxy time-to-first-token (streamed requests)",
    ["proxy", "model"],
    buckets=_TTFT_BUCKETS,
    registry=_registry,
)
PROXY_TPS = Histogram(
    "proxy_tokens_per_second",
    "LLM-proxy output throughput (completion tokens / generation time)",
    ["proxy", "model"],
    buckets=_TPS_BUCKETS,
    registry=_registry,
)
# Audio (whisper) transcription/translation confidence: the duration-weighted mean
# of -avg_logprob (NLL) over the upstream's per-segment `avg_logprob` (only present
# when the caller asks for response_format=verbose_json). Watch its windowed mean
# for MODEL DRIFT:
#   rate(proxy_audio_nll_sum[30m]) / rate(proxy_audio_nll_count[30m])
# alert if it RISES above a per-model baseline (e.g. > 0.45 for whisper turbo, which
# baselines around 0.19). p95 via histogram_quantile(0.95, ...) catches tail decay.
PROXY_AUDIO_NLL = Histogram(
    "proxy_audio_nll",
    "LLM-proxy audio (whisper) mean token negative log-likelihood (=-avg_logprob), "
    "duration-weighted over verbose_json segments (drift signal; >= 0, LOWER = more confident)",
    ["proxy", "model"],
    buckets=_NLL_BUCKETS,
    registry=_registry,
)


# TTS round-trip quality: the proxy transcribes its own generated audio through a
# whisper STT (async, off the response path) and scores it vs the input text. CER is
# the meaningful signal for CJK (word-based WER ≈ 1.0 there); WER for spaced scripts.
# Watch the windowed mean for TTS DRIFT:
#   rate(proxy_tts_cer_sum[30m]) / rate(proxy_tts_cer_count[30m])   (rises = worse)
# Labelled by voice/speaker too, so per-speaker regressions surface.
_ERR_RATE_BUCKETS = (
    0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, float("inf"),
)
_TTS_LABELS = ["proxy", "model", "voice"]
PROXY_TTS_CER = Histogram(
    "proxy_tts_cer",
    "TTS proxy character error rate (generated audio → STT → vs input text; drift signal)",
    _TTS_LABELS, buckets=_ERR_RATE_BUCKETS, registry=_registry,
)
PROXY_TTS_WER = Histogram(
    "proxy_tts_wer",
    "TTS proxy word error rate (generated audio → STT → vs input text; drift signal)",
    _TTS_LABELS, buckets=_ERR_RATE_BUCKETS, registry=_registry,
)
PROXY_TTS_EVAL_TOTAL = Counter(
    "proxy_tts_eval_total",
    "TTS CER/WER round-trip evaluations by outcome (scored / stt_error / skipped)",
    ["proxy", "result"], registry=_registry,
)


def observe_tts_eval(proxy: str, model: str, voice: str, cer: float, wer: float) -> None:
    """Record one completed TTS→STT round-trip: CER + WER (both ≥ 0). Best-effort;
    caller guarantees numeric inputs from a successful STT transcription."""
    lbl = {"proxy": proxy, "model": model or "", "voice": voice or ""}
    PROXY_TTS_CER.labels(**lbl).observe(max(0.0, float(cer)))
    PROXY_TTS_WER.labels(**lbl).observe(max(0.0, float(wer)))
    PROXY_TTS_EVAL_TOTAL.labels(proxy=proxy, result="scored").inc()


def observe_tts_eval_outcome(proxy: str, result: str) -> None:
    """Bump the eval outcome counter for a non-scored case (stt_error / skipped)."""
    PROXY_TTS_EVAL_TOTAL.labels(proxy=proxy, result=result).inc()


# Drift-sample capture: audio (+ JSON sidecar) persisted to storage when a request
# crosses a threshold (STT low avg_logprob / TTS high CER-WER). Watch the rate of
# "saved" as a leading indicator of how much bad output is flowing.
PROXY_CAPTURE_TOTAL = Counter(
    "proxy_capture_total",
    "Drift audio samples captured to storage, by kind (stt/tts) and outcome (saved/error/skipped)",
    ["proxy", "kind", "result"], registry=_registry,
)


def observe_capture(proxy: str, kind: str, result: str) -> None:
    PROXY_CAPTURE_TOTAL.labels(proxy=proxy, kind=kind or "", result=result).inc()


# Depth of the shared off-path background-job queue (TTS CER/WER evals + drift-sample
# uploads). A persistently-high value means the worker pool is saturated — raise
# PROXY_BG_WORKERS. Process-wide (not per-proxy), so it lives on the infra /metrics.
PROXY_BG_QUEUE = Gauge(
    "proxy_bg_queue_depth",
    "Pending off-path background jobs (TTS eval + drift capture) awaiting a worker",
    registry=_registry,
)


def set_bg_queue_depth(n: int) -> None:
    PROXY_BG_QUEUE.set(n)


def observe_proxy(
    proxy: str, model: str, upstream: str, status: str, duration_s: float | None = None,
    ttft_s: float | None = None, tps: float | None = None, avg_logprob: float | None = None,
) -> None:
    """Record one proxied request: bump the outcome counter and, when known, observe
    its latency / time-to-first-token / output throughput. `proxy` is the endpoint id;
    `status` is the terminal state (completed / failed / cancelled / timeout / …).
    ttft_s/tps are only meaningful for completed requests (None otherwise). avg_logprob
    is the audio-drift signal (whisper verbose_json segments; None for non-audio or
    when the caller didn't request verbose_json)."""
    PROXY_REQUESTS_TOTAL.labels(
        proxy=proxy, model=model or "", upstream=upstream or "", status=status
    ).inc()
    if duration_s is not None:
        PROXY_REQUEST_DURATION.labels(proxy=proxy, model=model or "").observe(duration_s)
    if ttft_s is not None:
        PROXY_TTFT.labels(proxy=proxy, model=model or "").observe(ttft_s)
    if tps is not None:
        PROXY_TPS.labels(proxy=proxy, model=model or "").observe(tps)
    if avg_logprob is not None:
        # Store NLL (-logprob, >= 0) so the histogram keeps its _sum sample; see
        # the PROXY_AUDIO_NLL / _NLL_BUCKETS notes for why.
        PROXY_AUDIO_NLL.labels(proxy=proxy, model=model or "").observe(-avg_logprob)


def render_proxy(proxy: str) -> bytes:
    """Prometheus exposition of ONE proxy endpoint's metrics, filtered to its id —
    served at `GET /proxy/{name}/metrics`, the sibling of `/{app_id}/metrics`."""

    class _ProxyFilter:
        def collect(self):
            for metric in (PROXY_REQUESTS_TOTAL, PROXY_REQUEST_DURATION, PROXY_TTFT, PROXY_TPS,
                           PROXY_AUDIO_NLL, PROXY_TTS_CER, PROXY_TTS_WER, PROXY_TTS_EVAL_TOTAL,
                           PROXY_CAPTURE_TOTAL):
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
# Live GPU occupancy: one series per (node, GPU) currently busy (value 1), with
# the workload type + name as labels. Lets Grafana render a per-GPU "what's
# running" heatmap from VictoriaMetrics alongside the DCGM utilization, and
# `sum by (node) (platform_gpu_busy)` gives busy-GPU counts per node. `gpu="all"`
# means the workload didn't pin specific devices (no CUDA_VISIBLE_DEVICES).
RES_GPU_BUSY = Gauge(
    "platform_gpu_busy",
    "One series per occupied (node, GPU) at scrape time (value 1); workload + name as labels",
    ["node", "gpu", "workload", "name"],
    registry=_resource_registry,
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


def _parse_gpu_ids(vd: "str | None") -> list[int]:
    vd = (vd or "").strip()
    if not vd:
        return []
    try:
        return [int(x.strip()) for x in vd.split(",") if x.strip() != ""]
    except ValueError:
        return []


async def sample_gpu_busy(session, rdb) -> None:
    """Sample live per-GPU occupancy into `platform_gpu_busy`. Inference: any
    endpoint with ≥1 live worker (Redis) marks its GPU ids busy. Benchmark: any
    run with status=running marks its pinned GPUs. Node is the bound provider's
    name (falls back to the GPU type / 'runpod' / 'shared')."""
    from sqlalchemy import select

    from .bench import Benchmark
    from .db import App, Provider

    RES_GPU_BUSY.clear()
    provs = {p.id: p.name for p in (await session.execute(select(Provider))).scalars().all()}

    # Benchmarks currently running.
    for b in (await session.execute(
        select(Benchmark).where(Benchmark.status == "running")
    )).scalars().all():
        node = (provs.get(b.provider_id) if b.provider_id else None) or (
            "runpod" if b.runpod_pod_id else "shared"
        )
        gpus = _parse_gpu_ids(b.visible_devices) or [-1]
        for g in gpus:
            RES_GPU_BUSY.labels(
                node=node, gpu=("all" if g < 0 else str(g)),
                workload="benchmark", name=(b.name or b.id),
            ).set(1)

    # Inference endpoints with at least one live worker registered in Redis.
    # Pipelined: one round-trip for all apps' member sets, one for all liveness
    # checks (was 1 + workers awaits per app).
    if rdb is not None:
        apps = (await session.execute(select(App))).scalars().all()
        pipe = rdb.pipeline(transaction=False)
        for a in apps:
            pipe.smembers(f"worker_index:{a.app_id}")
        member_sets = await pipe.execute() if apps else []
        flat: list[tuple[int, str]] = [
            (i, mid) for i, members in enumerate(member_sets) for mid in (members or ())
        ]
        live_apps: set[int] = set()
        if flat:
            pipe = rdb.pipeline(transaction=False)
            for _, mid in flat:
                pipe.exists(f"worker:{mid}")
            flags = await pipe.execute()
            live_apps = {i for (i, _), alive in zip(flat, flags) if alive}
        for i, a in enumerate(apps):
            if i not in live_apps:
                continue
            node = (provs.get(a.provider_id) if a.provider_id else None) or (a.gpu or "shared")
            gpus = _parse_gpu_ids(getattr(a, "visible_devices", None)) or list(range(int(a.gpu_count or 1)))
            for g in gpus:
                RES_GPU_BUSY.labels(
                    node=node, gpu=str(g), workload="inference", name=(a.name or a.app_id),
                ).set(1)


async def render_resources(session, rdb=None) -> tuple[bytes, str]:
    """Sample Postgres resources (+ live GPU occupancy when a Redis handle is
    given), then serialize the resource registry. Backs GET /metrics/resources."""
    await sample_resources(session)
    if rdb is not None:
        try:
            await sample_gpu_busy(session, rdb)
        except Exception:  # best-effort — never break the scrape over occupancy
            pass
    return generate_latest(_resource_registry), CONTENT_TYPE_LATEST


def _sample_runtime_health() -> None:
    """Sample the DB-pool and stats-writer gauges. Never raises — a scrape must
    not fail because an internal sample did."""
    try:
        from . import db as _db
        st = _db.pool_status()
        if st is not None:
            DB_POOL_CHECKED_OUT.set(st["checked_out"])
            DB_POOL_CAPACITY.set(st["capacity"])
    except Exception:  # noqa: BLE001 — best-effort self-observation
        logger.debug("db pool sample failed", exc_info=True)
    try:
        from . import stats_writer as _sw
        STATS_WRITER_QUEUE_DEPTH.set(_sw.queue_depth())
    except Exception:  # noqa: BLE001
        logger.debug("stats-writer depth sample failed", exc_info=True)


async def _sample_redis(rdb: "redis_async.Redis") -> None:
    """Sample the per-app queue/worker gauges from Redis, pipelined (the old
    per-app awaits were 2 + 2·apps + workers round-trips per scrape). Raises on
    Redis failure — the caller converts that into gateway_redis_up=0."""
    t0 = time.perf_counter()
    await rdb.ping()
    REDIS_PING_SECONDS.set(time.perf_counter() - t0)

    # Discover app_ids from existing worker_index/queue keys (Postgres is the
    # source of truth for app metadata, but for metrics we just want anything
    # with active state in Redis).
    app_ids: set[str] = set()
    async for key in rdb.scan_iter(match="worker_index:*"):
        app_ids.add(key.split(":", 1)[1])
    async for key in rdb.scan_iter(match="queue:*"):
        app_ids.add(key.split(":", 1)[1])
    ids = sorted(app_ids)

    if ids:
        pipe = rdb.pipeline(transaction=False)
        for app_id in ids:
            pipe.llen(f"queue:{app_id}")
            pipe.smembers(f"worker_index:{app_id}")
        res = await pipe.execute()
    else:
        res = []

    QUEUE_LENGTH.clear()
    WORKERS_TOTAL.clear()
    flat_members: list[tuple[str, str]] = []
    for i, app_id in enumerate(ids):
        QUEUE_LENGTH.labels(app_id=app_id).set(int(res[2 * i] or 0))
        flat_members.extend((app_id, mid) for mid in (res[2 * i + 1] or ()))

    live_by_app: dict[str, int] = {app_id: 0 for app_id in ids}
    if flat_members:
        pipe = rdb.pipeline(transaction=False)
        for _, mid in flat_members:
            pipe.exists(f"worker:{mid}")
        flags = await pipe.execute()
        for (app_id, _), alive in zip(flat_members, flags):
            if alive:
                live_by_app[app_id] += 1
    for app_id in ids:
        WORKERS_TOTAL.labels(app_id=app_id).set(live_by_app[app_id])


async def render(rdb: "redis_async.Redis") -> tuple[bytes, str]:
    """Sample point-in-time gauges (Redis queue/worker state, DB pool, stats
    writer), then serialize the registry. Degrades instead of raising when Redis
    is unreachable — the scrape then serves gateway_redis_up=0 plus every
    in-process counter/histogram, rather than 500ing and leaving monitoring
    blind during exactly the outage it should be reporting."""
    try:
        await _sample_redis(rdb)
        REDIS_UP.set(1)
    except Exception:  # noqa: BLE001 — degrade, don't fail the scrape
        REDIS_UP.set(0)
        logger.warning("metrics: redis sample failed — serving degraded scrape", exc_info=True)
    _sample_runtime_health()
    return generate_latest(_registry), CONTENT_TYPE_LATEST
