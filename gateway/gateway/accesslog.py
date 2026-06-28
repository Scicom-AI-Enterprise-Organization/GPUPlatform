"""Structured HTTP access logging for the gateway.

One line per request. With ``LOG_JSON=1`` each line is a JSON object whose fields
(``status`` / ``route`` / ``method`` / ``durationMs`` / ``app_id`` / ``requestId``)
are parsed by Promtail and become queryable in Grafana via LogQL — the same
shape SlurmUI emits, so the Loki log panel + dashboard variables work identically:

    {service="gateway"} | json | status >= 500
    {service="gateway"} | json | durationMs > 1000
    {service="gateway", app_id="tm-fleet"} | json

Without ``LOG_JSON`` the line is human-readable (``POST /tm-fleet/v1/chat/completions
→ 200 (842.301ms)``) for local-dev terminals.

The access logger is independent of the root logger (``propagate=False``) so the
JSON lines stay clean — no ``asctime levelname name:`` prefix wrapping them.

It writes to stdout, and additionally to ``GATEWAY_ACCESS_LOG`` (a file path) when
set. The file tee is what lets a host-side Promtail tail the access log when the
gateway runs as a local process (``.venv/bin/gateway``) rather than in a container.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import Optional

_LOGGER = logging.getLogger("gateway.access")
# Separate stream for serverless-endpoint (vLLM) logs re-emitted from
# /workers/logs, so Loki/Alloy can ingest them as `service="vllm"` alongside
# the gateway access log. Independent logger keeps its JSON lines unprefixed.
_EP_LOGGER = logging.getLogger("gateway.endpoint")
_JSON = False
_INIT = False
# True once endpoint re-emit is actually wired (LOG_JSON=1 → prod Alloy tails
# stdout, or GATEWAY_ENDPOINT_LOG set → dev Promtail tails the file). When False
# `log_endpoint_lines` is a no-op so plain local dev pays nothing.
_EP_ENABLED = False

# vLLM/uvicorn lines carry a level token near the start, e.g.
# "(EngineCore pid=…) ERROR 06-28 …" — lift it to a `level` field for
# `{service="vllm"} | json | level="error"`.
_LEVEL_RE = re.compile(r"\b(CRITICAL|ERROR|WARNING|WARN|INFO|DEBUG)\b")


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def init_access_logging() -> None:
    """Configure the access logger once (idempotent — safe to call per worker)."""
    global _JSON, _INIT
    if _INIT:
        return
    _JSON = _truthy(os.environ.get("LOG_JSON", ""))
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False  # keep JSON lines unprefixed

    fmt = logging.Formatter("%(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    path = os.environ.get("GATEWAY_ACCESS_LOG", "").strip()
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(path))
    for h in handlers:
        h.setFormatter(fmt)
        _LOGGER.addHandler(h)

    # Endpoint (vLLM) log stream. Always JSON (it's a machine stream consumed by
    # Loki/Alloy, never a human terminal). stdout only in LOG_JSON mode (prod
    # Alloy tails the gateway's stdout); the file tee (GATEWAY_ENDPOINT_LOG) is
    # what a host-side dev Promtail tails — kept off stdout so it doesn't clutter
    # the local terminal.
    global _EP_ENABLED
    _EP_LOGGER.setLevel(logging.INFO)
    _EP_LOGGER.propagate = False
    ep_handlers: list[logging.Handler] = []
    if _JSON:
        ep_handlers.append(logging.StreamHandler(sys.stdout))
    ep_path = os.environ.get("GATEWAY_ENDPOINT_LOG", "").strip()
    if ep_path:
        os.makedirs(os.path.dirname(ep_path) or ".", exist_ok=True)
        ep_handlers.append(logging.FileHandler(ep_path))
    for h in ep_handlers:
        h.setFormatter(fmt)
        _EP_LOGGER.addHandler(h)
    _EP_ENABLED = bool(ep_handlers)
    _INIT = True


def log_request(
    *,
    method: str,
    route: str,
    path: str,
    status: int,
    duration_ms: float,
    request_id: str,
    app_id: Optional[str] = None,
    ip: Optional[str] = None,
    nbytes: Optional[int] = None,
) -> None:
    """Emit one access-log record. ``route`` is the templated path
    (``/{app_id}/v1/chat/completions``) for stable dashboards; ``path`` is the raw
    URL for drill-down."""
    status_class = f"{status // 100}xx"
    if _JSON:
        rec = {
            "service": "gateway",
            "kind": "http_access",
            "level": "error" if status >= 500 else "warn" if status >= 400 else "info",
            "method": method,
            "route": route,
            "path": path,
            "status": status,
            "statusClass": status_class,
            "durationMs": round(duration_ms, 3),
            "app_id": app_id,
            "requestId": request_id,
            "ip": ip,
            "bytes": nbytes,
            "time": int(time.time() * 1000),
            "msg": "http_request",
        }
        _LOGGER.info(json.dumps(rec, separators=(",", ":")))
    else:
        _LOGGER.info("%s %s → %d (%.3fms)", method, path, status, duration_ms)


def _line_level(line: str) -> str:
    """Best-effort severity for a vLLM log line (scan only the head — the level
    token sits near the front). Defaults to info."""
    m = _LEVEL_RE.search(line[:120])
    if not m:
        return "info"
    t = m.group(1)
    if t in ("ERROR", "CRITICAL"):
        return "error"
    if t in ("WARNING", "WARN"):
        return "warn"
    if t == "DEBUG":
        return "debug"
    return "info"


def log_endpoint_lines(
    *,
    app_id: str,
    model: Optional[str],
    machine: Optional[str],
    session: Optional[str],
    lines: list[str],
) -> None:
    """Re-emit a batch of serverless-endpoint (vLLM) log lines into the
    ``service="vllm"`` stream for Loki/Alloy. No-op unless endpoint logging is
    wired (see ``_EP_ENABLED``), so the /workers/logs hot path pays nothing in
    plain local dev. Each line becomes a JSON record; ``app_id``/``model`` are
    low-cardinality and meant to be promoted to Loki labels by the collector,
    while ``machine``/``session`` stay queryable JSON fields (``machine`` is an
    unbounded RunPod pod id — a poor label)."""
    if not _EP_ENABLED or not lines:
        return
    now = int(time.time() * 1000)
    for line in lines:
        if not line:
            continue
        rec = {
            "service": "vllm",
            "kind": "endpoint_log",
            "level": _line_level(line),
            "app_id": app_id,
            "model": model,
            "machine": machine,
            "session": session,
            "time": now,
            "msg": line,
        }
        _EP_LOGGER.info(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))
