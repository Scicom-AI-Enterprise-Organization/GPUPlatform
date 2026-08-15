"""OTLP tracing for LLM-proxy requests — Tempo/Jaeger as the request record.

**Why this exists.** Every proxied request writes a `proxy_requests` row (INSERT at
admission, UPDATE at start, UPDATE at terminal). That is a fine audit trail at dev
volume and the wrong storage engine at prod volume: Postgres is a transactional store
being asked to absorb an append-only, write-once, read-rarely event stream that grows
without bound, and the Queue tab reads it back with `ORDER BY created_at DESC` over a
table nobody prunes (``PROXY_REQUEST_RETENTION_DAYS`` defaults to 0 = keep forever).

Distributed-tracing backends are built for exactly this shape — columnar/object-store
blocks, TTL by design, attribute indexes, and a UI for one request's life. So each
proxied request also becomes ONE span, exported over OTLP. With
``PROXY_REQUEST_STORE=off`` the span becomes the *only* record and Postgres carries
none of it (see ``proxy_api._store_mode``).

**Rules this module holds itself to:**

- **It can never break the request path.** Every public function is wrapped: a dead
  collector, a missing wheel, a malformed attribute — all degrade to a no-op plus (at
  most) one log line. The exporter's own queue is bounded and drops on overflow, the
  same best-effort contract ``stats_writer`` makes.
- **Import is lazy and optional.** The opentelemetry wheels are declared in
  `pyproject.toml`, but an already-built venv won't have them until it is re-synced —
  and a gateway that refuses to boot for a *telemetry* dependency is a worse outage
  than no telemetry. Missing import → ``enabled()`` stays False, forever, quietly.
- **Sampling decides at the END, not the start.** A head sampler (the OTel default)
  chooses before the request runs, so a 5% sample keeps 5% of the failures — exactly
  the 95% you needed. ``_PolicyProcessor`` instead keeps every error / slow / blocked
  request and ratio-samples only the boring successes. Cost: a sampled-out span is
  built and thrown away (cheap — it is never serialized), which is the trade a real
  tail sampler in the collector would make on a bigger scale.

Config (all optional; tracing is OFF unless ``PROXY_TRACING=1``):

    PROXY_TRACING=1                       # master switch
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318   # Tempo/Jaeger/collector
    OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf           # or `grpc`
    OTEL_EXPORTER_OTLP_HEADERS=authorization=Basic%20…  # e.g. Grafana Cloud
    OTEL_SERVICE_NAME=gateway
    PROXY_TRACE_SAMPLE_RATIO=1.0          # ratio for SUCCESSFUL requests only
    PROXY_TRACE_SLOW_MS=0                 # 0=off; else always keep >= this latency
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

logger = logging.getLogger("gateway.tracing")

# ---------- config -----------------------------------------------------------

SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME") or "gateway"
_SAMPLE_RATIO = float(os.environ.get("PROXY_TRACE_SAMPLE_RATIO", "1.0") or "1.0")
_SLOW_MS = int(os.environ.get("PROXY_TRACE_SLOW_MS", "0") or "0")
# A span whose request never reached a terminal state (handler died between admission
# and _finish) would sit in _PENDING forever. The sweeper ends those as `unknown`.
_STALE_S = float(os.environ.get("PROXY_TRACE_STALE_S", "7200") or "7200")

# Statuses that are ALWAYS exported regardless of the sample ratio — the whole point
# of sampling at the end rather than the start.
_ALWAYS_KEEP = frozenset({"failed", "cancelled", "blocked", "unknown"})

_enabled = False
_tracer: Any = None
_provider: Any = None
_init_lock = threading.Lock()
_init_done = False

# request_id -> {"span": Span, "created": float, "attrs": dict}
_PENDING: dict[str, dict] = {}
# Bound the in-flight map: a pathological leak must not become the OOM. Well above
# any real concurrent-request count (the gate caps that far lower).
_PENDING_MAX = int(os.environ.get("PROXY_TRACE_PENDING_MAX", "50000") or "50000")


def enabled() -> bool:
    """True when spans are being produced. Cheap — safe on the hot path."""
    return _enabled


# ---------- sampling ---------------------------------------------------------

def _keep(status: str, status_code: Optional[int], latency_ms: Optional[int],
          trace_id: int) -> bool:
    """Export decision, made once the outcome is known.

    Order matters: the interesting-request rules come FIRST so a 1% ratio still
    yields 100% of the failures. Deterministic in `trace_id` so the same trace is
    kept or dropped identically by every replica (they never see the same trace
    twice today, but a future multi-span trace must not be half-exported)."""
    if status in _ALWAYS_KEEP:
        return True
    if status_code is not None and status_code >= 400:
        return True
    if _SLOW_MS > 0 and latency_ms is not None and latency_ms >= _SLOW_MS:
        return True
    if _SAMPLE_RATIO >= 1.0:
        return True
    if _SAMPLE_RATIO <= 0.0:
        return False
    # Bottom 64 bits of the trace id, compared against the ratio — the same
    # construction TraceIdRatioBased uses, applied after the fact.
    return (trace_id & 0xFFFFFFFFFFFFFFFF) < int(_SAMPLE_RATIO * (1 << 64))


def _make_policy_processor(inner):
    """Wrap a BatchSpanProcessor so `on_end` can drop a span before it is queued."""
    from opentelemetry.sdk.trace import SpanProcessor

    class _PolicyProcessor(SpanProcessor):
        def on_start(self, span, parent_context=None):  # noqa: D102
            inner.on_start(span, parent_context)

        def on_end(self, span):  # noqa: D102
            try:
                attrs = span.attributes or {}
                if not _keep(str(attrs.get("sgpu.status") or ""),
                             attrs.get("http.response.status_code"),
                             attrs.get("sgpu.latency_ms"),
                             span.context.trace_id if span.context else 0):
                    return
            except Exception:  # noqa: BLE001 — never drop telemetry on a policy bug
                pass
            inner.on_end(span)

        def shutdown(self):  # noqa: D102
            inner.shutdown()

        def force_flush(self, timeout_millis: int = 30000):  # noqa: D102
            return inner.force_flush(timeout_millis)

    return _PolicyProcessor()


# ---------- lifecycle --------------------------------------------------------

def _build_exporter():
    """OTLP span exporter for the configured protocol. `http/protobuf` (port 4318)
    is the default because it needs no grpcio wheel; `grpc` (4317) is imported only
    if asked for, so a deployment that wants it installs the extra package."""
    proto = (os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL") or "http/protobuf").strip().lower()
    if proto in ("grpc", "otlp_grpc"):
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        return OTLPSpanExporter()
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    return OTLPSpanExporter()


def start() -> bool:
    """Initialise the tracer provider. Idempotent; safe to call from lifespan.
    Returns whether tracing came up. Never raises."""
    global _enabled, _tracer, _provider, _init_done
    with _init_lock:
        if _init_done:
            return _enabled
        _init_done = True
        if os.environ.get("PROXY_TRACING", "0") != "1":
            return False
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as e:
            logger.warning(
                "PROXY_TRACING=1 but the opentelemetry SDK is not installed (%s) — "
                "proxy tracing stays OFF. Install with: uv pip install -e './gateway'", e)
            return False
        try:
            resource = Resource.create({
                "service.name": SERVICE_NAME,
                "service.namespace": os.environ.get("OTEL_SERVICE_NAMESPACE") or "serverless-gpu",
            })
            _provider = TracerProvider(resource=resource)
            _provider.add_span_processor(_make_policy_processor(BatchSpanProcessor(_build_exporter())))
            # Deliberately NOT trace.set_tracer_provider(): this provider serves the
            # proxy's own spans. Claiming the global would silently adopt every
            # library that auto-instruments itself, which is a different (and much
            # larger) traffic decision than the one being made here.
            _tracer = _provider.get_tracer("gateway.proxy")
            _enabled = True
            logger.info("proxy tracing ON (service=%s, endpoint=%s, protocol=%s, ratio=%.3g)",
                        SERVICE_NAME, os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or "<default>",
                        os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL") or "http/protobuf",
                        _SAMPLE_RATIO)
            return True
        except Exception:  # noqa: BLE001 — telemetry must never block startup
            logger.exception("proxy tracing failed to start — continuing without it")
            _enabled = False
            return False


def shutdown() -> None:
    """Flush + stop the exporter (called from lifespan shutdown). Never raises."""
    global _enabled
    _enabled = False
    p = _provider
    if p is None:
        return
    try:
        for rid in list(_PENDING):
            end_request(rid, "unknown", error="gateway shutting down")
        p.shutdown()
    except Exception:  # noqa: BLE001
        logger.debug("tracer shutdown failed", exc_info=True)


# ---------- span lifecycle ---------------------------------------------------

def _set(span, key: str, value: Any) -> None:
    if value is not None and value != "":
        span.set_attribute(key, value)


def start_request(request_id: str, *, endpoint_id: str, endpoint_name: Optional[str],
                  model: Optional[str], owner: Optional[str], owner_id: Optional[int],
                  is_stream: bool, headers: Optional[dict] = None,
                  created_at: Optional[float] = None) -> None:
    """Open the span for a proxied request, at admission.

    `headers` are the CALLER's request headers: a client already tracing its own work
    sends `traceparent`, and extracting it makes this span a child of theirs rather
    than an orphan — the difference between "the gateway was slow" and "here is the
    gateway's leg of my slow request". Never raises."""
    if not _enabled:
        return
    try:
        if len(_PENDING) >= _PENDING_MAX:
            logger.warning("proxy tracing: %d spans in flight (max) — dropping new spans "
                           "until the sweeper runs", len(_PENDING))
            return
        ctx = None
        if headers:
            try:
                from opentelemetry.propagate import extract
                ctx = extract(dict(headers))
            except Exception:  # noqa: BLE001 — a malformed traceparent is the client's bug
                ctx = None
        from opentelemetry.trace import SpanKind
        span = _tracer.start_span(
            f"proxy {endpoint_name or endpoint_id}", context=ctx, kind=SpanKind.SERVER,
            start_time=int((created_at or time.time()) * 1_000_000_000),
        )
        _set(span, "sgpu.request.id", request_id)
        _set(span, "sgpu.request.kind", "proxy")
        _set(span, "sgpu.proxy.id", endpoint_id)
        _set(span, "sgpu.proxy.name", endpoint_name)
        _set(span, "sgpu.model", model)
        _set(span, "gen_ai.request.model", model)
        _set(span, "sgpu.owner", owner)
        _set(span, "sgpu.owner.id", owner_id)
        span.set_attribute("sgpu.stream", bool(is_stream))
        span.set_attribute("sgpu.status", "queued")
        _PENDING[request_id] = {"span": span, "created": time.time()}
    except Exception:  # noqa: BLE001
        logger.debug("start_request span failed", exc_info=True)


def mark_started(request_id: str, upstream: Optional[str] = None) -> None:
    """The request left the queue and is being forwarded. Recorded as a span EVENT so
    queue-wait is readable inside the span (its timestamp minus the span start) without
    a second span per request."""
    if not _enabled:
        return
    try:
        rec = _PENDING.get(request_id)
        if rec is None:
            return
        span = rec["span"]
        span.add_event("running", {"sgpu.upstream": upstream} if upstream else None)
        span.set_attribute("sgpu.status", "running")
        _set(span, "sgpu.upstream", upstream)
    except Exception:  # noqa: BLE001
        logger.debug("mark_started span failed", exc_info=True)


def end_request(request_id: str, status: str, *, status_code: Optional[int] = None,
                latency_ms: Optional[int] = None, ttft_ms: Optional[int] = None,
                pt: Optional[int] = None, ct: Optional[int] = None,
                error: Optional[str] = None, upstream: Optional[str] = None,
                avg_logprob: Optional[float] = None) -> None:
    """Close the span with the terminal outcome. Mirrors `_finish`'s arguments 1:1 so
    the trace and the (optional) Postgres row can never describe different requests."""
    if not _enabled:
        return
    rec = _PENDING.pop(request_id, None)
    if rec is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode
        span = rec["span"]
        span.set_attribute("sgpu.status", status)
        _set(span, "http.response.status_code", status_code)
        _set(span, "sgpu.latency_ms", latency_ms)
        _set(span, "sgpu.ttft_ms", ttft_ms)
        _set(span, "gen_ai.usage.input_tokens", pt)
        _set(span, "gen_ai.usage.output_tokens", ct)
        _set(span, "sgpu.upstream", upstream)
        _set(span, "sgpu.audio.avg_logprob", avg_logprob)
        if error:
            _set(span, "sgpu.error", str(error)[:2048])
        # `blocked` is a red-team guard doing its job, and `cancelled` is the caller
        # leaving — neither is a gateway error, and marking them ERROR would put both
        # into every error-rate panel built on span status.
        if status == "failed":
            span.set_status(Status(StatusCode.ERROR, str(error)[:256] if error else None))
            _set(span, "error.type", (str(error).split(":")[0][:64] if error else "failed"))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()
    except Exception:  # noqa: BLE001
        logger.debug("end_request span failed", exc_info=True)


def trace_id(request_id: str) -> Optional[str]:
    """Hex trace id of an in-flight request — for `X-Trace-Id` on the response, which
    is what lets a caller (or the Queue tab) jump straight to the trace."""
    if not _enabled:
        return None
    try:
        rec = _PENDING.get(request_id)
        if rec is None:
            return None
        return format(rec["span"].get_span_context().trace_id, "032x")
    except Exception:  # noqa: BLE001
        return None


def sweep(max_age_s: Optional[float] = None) -> int:
    """End spans whose request never reached `_finish` (handler killed, gateway
    restarted mid-stream). Returns how many were reaped. Called from the proxy health
    loop — without it a leak is unbounded AND the trace never lands, so the request
    would be invisible in BOTH stores."""
    if not _enabled:
        return 0
    cutoff = time.time() - (max_age_s if max_age_s is not None else _STALE_S)
    stale = [rid for rid, rec in list(_PENDING.items()) if rec.get("created", 0) < cutoff]
    for rid in stale:
        end_request(rid, "unknown", error="no terminal state recorded (handler died or gateway restarted)")
    if stale:
        logger.warning("proxy tracing: reaped %d stale span(s)", len(stale))
    return len(stale)


def pending_count() -> int:
    """In-flight spans — sampled by /metrics so a leak is visible before it is an OOM."""
    return len(_PENDING)
