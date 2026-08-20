"""Correlation-ID observability via `wan`, layered UNDER the gateway's own telemetry.

`wan` (github.com/Scicom-AI-Enterprise-Organization/wan) is the org's FastAPI
observability boilerplate. The gateway already has three purpose-built pieces that
overlap it, so this module takes only the part that was genuinely missing — a
**correlation id** that survives across service boundaries — and disables the rest.

What we take from wan:
  * a request-scoped correlation id, reusing an inbound `X-Correlation-ID` when a
    caller already has one, minting one otherwise;
  * `propagation.install()`, which COMPOSES a correlation-id propagator onto the
    global OTel propagator, so every OTel-instrumented HTTP client injects
    `X-Correlation-ID` alongside `traceparent`/`baggage` on outbound calls.

What we deliberately keep ours, and why:
  * **`/metrics`** — `metrics.py` (1000+ lines of gateway-specific series) already
    serves it from `main.py`. `enable_prometheus_metrics=False`.
  * **health endpoints / Scalar docs** — the gateway has its own, and the web UI ships
    `/api-docs`. Both off.
  * **the access log** — `accesslog.log_request` emits a field schema
    (`durationMs`/`app_id`/`requestId`) that SlurmUI's Grafana dashboards query by.
    wan's `type=request` line would duplicate it in a different shape, and `app_id` is
    per-request so wan's *static* `log_static_fields` could not reproduce it anyway.
    `enable_request_log=False`.
  * **the proxy span + its tail sampler** — see `tracing.py`. No conflict exists:
    `tracing.start()` deliberately does NOT call `set_tracer_provider()`, keeping a
    private provider, so wan is free to own the global one. Do not "fix" either side
    to share a provider — the private provider is what keeps `_PolicyProcessor`
    (keep every failure, ratio-sample successes) off wan's head sampler.

⚠ **`wan.patch()` always calls `setup_logging()` and always reconfigures the root
logger — there is no flag to skip it.** So `main.py` re-asserts
`accesslog.init_root_logging()` *after* `start()`. Call them the other way round and
every gateway log line silently changes shape, breaking the Loki queries above.

Contract, identical to `tracing.py`'s: never break the request path, never break boot.
A missing wheel or a bad config degrades to a no-op plus one log line.

Config:
    GATEWAY_CORRELATION=1            # master switch (default OFF)
    CORRELATION_ID_HEADER=X-Correlation-ID
    SERVICE_NAME=gateway
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any, Optional

logger = logging.getLogger("gateway.observability")

HEADER = os.environ.get("CORRELATION_ID_HEADER") or "X-Correlation-ID"
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME") or os.environ.get("SERVICE_NAME") or "gateway"

_enabled = False
_init_done = False
_init_lock = threading.Lock()
_wan: Any = None


def enabled() -> bool:
    """True when wan came up. Cheap — safe on the hot path."""
    return _enabled


def start(app) -> bool:
    """Install wan's correlation id + outbound propagation. Idempotent, never raises.

    Returns whether it came up. Must be called BEFORE
    `accesslog.init_root_logging()` — see the module docstring.
    """
    global _enabled, _init_done, _wan
    with _init_lock:
        if _init_done:
            return _enabled
        _init_done = True
        if os.environ.get("GATEWAY_CORRELATION", "0") != "1":
            return False
        try:
            import wan
        except ImportError as e:
            logger.warning(
                "GATEWAY_CORRELATION=1 but `wan` is not installed (%s) — correlation "
                "ids stay OFF. Install with: uv pip install -e './gateway'", e)
            return False
        try:
            wan.patch(
                app=app,
                service_name=SERVICE_NAME,
                # --- ours, not wan's (see module docstring) ---
                enable_prometheus_metrics=False,
                enable_health_endpoints=False,
                enable_scalar_doc=False,
                enable_request_log=False,
                # --- the reason we're here ---
                enable_correlation_id_propagation=True,
                correlation_id_header=HEADER,
                # Injects traceparent + X-Correlation-ID on outbound httpx calls. The
                # proxy's upstream calls additionally go through an explicit hook (see
                # proxy_api._http) so propagation does not depend on this wheel being
                # present.
                enable_httpx_instrumentation=True,
            )
            _wan = wan
            _enabled = True
            logger.info("correlation ids on (header=%s, service=%s)", HEADER, SERVICE_NAME)
        except Exception:  # noqa: BLE001 — telemetry must never break boot
            logger.warning("wan.patch() failed — correlation ids stay OFF", exc_info=True)
            return False
        return _enabled


def correlation_id() -> Optional[str]:
    """The current request's correlation id, or None outside a request.

    Checks BOTH sources, because they are populated at different times and either can
    be the only one present:

      * `accesslog.correlation_id_var` — set by `metrics_mw` on every request, whether
        or not wan is installed. This is the gateway's source of truth and the value
        the log lines and the proxy span carry.
      * wan's own contextvar — set by wan's middleware, and the only source during the
        window before `metrics_mw` has run.

    ⚠ Do not reduce this to the wan lookup alone. `GATEWAY_CORRELATION` defaults to
    OFF, and in that mode `metrics_mw` still mints and sets an id — reading only wan
    would return None and upstream propagation would silently stop happening in the
    default configuration.

    wan returns the sentinel `'-'` when unset; normalised to None so callers can treat
    "no id" uniformly rather than forwarding a bare dash.
    """
    try:
        from .accesslog import correlation_id_var
        cid = correlation_id_var.get()
        if cid:
            return cid
    except Exception:  # noqa: BLE001
        pass
    if not _enabled or _wan is None:
        return None
    try:
        cid = _wan.get_correlation_id()
    except Exception:  # noqa: BLE001
        return None
    return cid if cid and cid != "-" else None


def new_id() -> str:
    """Mint a correlation id for a request that arrived without one."""
    return f"cid-{uuid.uuid4().hex[:16]}"


def inject(headers: dict) -> dict:
    """Add `traceparent`/`baggage`/`X-Correlation-ID` to an outbound header dict.

    Mutates and returns `headers`. Uses the GLOBAL OTel propagator, which
    `wan.propagation.install()` has composed the correlation-id propagator onto — so
    this stays correct if wan changes which headers it carries.

    Used by the proxy's shared httpx client so upstream workers log under the same
    correlation id. Never raises: a telemetry failure must not fail the proxied call.
    """
    try:
        from opentelemetry import propagate
        propagate.inject(headers)
    except Exception:  # noqa: BLE001
        pass
    # Not belt-and-braces — this is the ONLY path when wan is absent (the default),
    # because `propagate.inject` carries the correlation id only once wan has composed
    # its propagator onto the global one. Without this the id would be set on the log
    # lines and the span but never reach the upstream worker.
    try:
        cid = correlation_id()
        if cid and not any(k.lower() == HEADER.lower() for k in headers):
            headers[HEADER] = cid
    except Exception:  # noqa: BLE001
        pass
    return headers
