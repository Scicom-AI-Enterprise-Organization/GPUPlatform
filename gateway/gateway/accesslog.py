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

⚠ ``LOG_JSON=1`` means the **whole stdout stream** is JSON, not only these lines:
``init_root_logging`` swaps the root formatter for ``_JsonLogFormatter`` (so module
logs and tracebacks parse too) and ``_tame_server_loggers`` folds uvicorn's
propagate-False loggers into it, dropping its duplicate access line. Anything that
prints outside the logging module (a library writing to stdout directly) is still
raw text — there is no way to catch that here.

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
from contextvars import ContextVar
from typing import Optional

_LOGGER = logging.getLogger("gateway.access")

# Request-id correlation for ORDINARY module log lines (logger.info(...) in
# training_api/proxy_api/...), not just the access log. metrics_mw sets this per
# request; the filter below stamps it onto every record passing through the root
# handlers, so `%(request_id)s` in the root format renders ` [req-…]` inside a
# request and nothing outside one. ContextVars propagate into tasks spawned by
# the handler, so background work started per-request stays correlated.
request_id_var: ContextVar[Optional[str]] = ContextVar("gateway_request_id", default=None)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rid = request_id_var.get()
        record.request_id = f" [{rid}]" if rid else ""
        return True


class _JsonLogFormatter(logging.Formatter):
    """Render an ORDINARY module log record as one JSON object, so `LOG_JSON=1`
    means the whole stream is JSON — not just the access log.

    Before this, `LOG_JSON=1` only reformatted `gateway.access`/`gateway.endpoint`;
    every `logger.info(...)` in the gateway (and every httpx/paramiko/boto line)
    still went out as `2026-08-14 08:46:43,226 INFO httpx: …`. A Loki pipeline that
    does `| json` drops those lines on the floor as unparseable, so exactly the
    lines you need when something breaks — the tracebacks — were the ones missing
    from the query, while the access log they belong to parsed fine.

    Field names match `log_request`'s (`service`/`level`/`time`/`msg` + the same
    `requestId`) so one LogQL query spans both kinds:
        {service="gateway"} | json | requestId="pxr-4ce76aeb0409d8b5"
    """

    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_var.get()
        rec = {
            "service": "gateway",
            "kind": "app_log",
            "level": record.levelname.lower().replace("warning", "warn"),
            "logger": record.name,
            "requestId": rid,
            "time": int(record.created * 1000),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            rec["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            rec["stack"] = self.formatStack(record.stack_info)
        return json.dumps(rec, separators=(",", ":"), ensure_ascii=False, default=str)


_ROOT_INIT = False


def _tame_server_loggers() -> None:
    """Fold uvicorn's own loggers into the root handler, and silence its access log.

    Two reasons, both visible in a prod log tail:
    1. uvicorn configures `uvicorn` / `uvicorn.access` with **`propagate: False`** and
       its own stderr/stdout handlers, so those lines bypass whatever format the root
       logger has — they stay `INFO:     10.1.28.212:47074 - "GET /ready HTTP/1.1" 200 OK`
       even under LOG_JSON=1.
    2. That line is a strictly WORSE DUPLICATE of our own `http_access` record, which
       `metrics_mw` emits for **every** request (there is no ignore list) with the route
       template, duration, byte count, app_id and the actionable request id. Keeping both
       doubles the log bill to say less.

    So: clear uvicorn's handlers and let its records propagate to the root formatter, and
    turn `uvicorn.access` off entirely. Set `LOG_UVICORN_ACCESS=1` to keep it (useful only
    if you suspect requests are dying BEFORE the middleware — a malformed request line,
    or an ASGI-level rejection, never reaches `log_request`)."""
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "gunicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    # Set BOTH ways, not just the disable: LOG_UVICORN_ACCESS=1 must actually bring the
    # line back, even if something (a re-init, another library) already disabled it.
    logging.getLogger("uvicorn.access").disabled = not truthy(
        os.environ.get("LOG_UVICORN_ACCESS", ""))


def init_root_logging() -> None:
    """Configure the root logger once (idempotent). Called from BOTH run() and
    lifespan so an external ASGI server importing gateway.main:app still gets
    formatted, request-id-correlated logs (basicConfig no-ops if something
    already configured handlers — we still attach the request-id filter).

    `LOG_JSON=1` makes every record a JSON object (see `_JsonLogFormatter`);
    without it the human-readable dev format is unchanged."""
    global _ROOT_INIT
    if _ROOT_INIT:
        return
    json_mode = _truthy(os.environ.get("LOG_JSON", ""))
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s%(request_id)s: %(message)s",
    )
    f = _RequestIdFilter()
    for h in logging.getLogger().handlers:
        h.addFilter(f)
        if json_mode:
            # The request id rides INSIDE the JSON object (`requestId`), so the
            # `%(request_id)s` text prefix is dropped here rather than duplicated.
            h.setFormatter(_JsonLogFormatter())
    _tame_server_loggers()
    _ROOT_INIT = True
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


def truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


_truthy = truthy  # internal alias (pre-existing call sites)


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
