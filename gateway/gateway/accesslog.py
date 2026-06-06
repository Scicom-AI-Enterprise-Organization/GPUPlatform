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
import sys
import time
from typing import Optional

_LOGGER = logging.getLogger("gateway.access")
_JSON = False
_INIT = False


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
