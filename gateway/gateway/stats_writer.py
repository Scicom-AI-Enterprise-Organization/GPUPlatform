"""Background batch-writer for per-request statistics (TTFT, token usage, latency,
completion status).

Request handlers used to each open a fresh DB session and commit the moment a
request finished (``_record_stream_completion`` for serverless streams,
``_finish`` for the proxy). Each completion therefore checked out a pooled
connection; with the default async pool (5 + 10 overflow) a burst of concurrent
requests exhausted the pool and wedged the gateway — every new checkout blocking
on ``pool_timeout``. (This is the prod "stuck under load" incident.)

These stats don't need to be realtime, so handlers now ENQUEUE a tiny update
intent (non-blocking, no DB) and a single consumer coroutine coalesces pending
intents per row and flushes them in batched transactions. N concurrent
completions cost one writer connection — not N. Updates are best-effort: if the
bounded queue overflows we drop (and log) rather than apply backpressure to the
inference path.

Tunables (env): STATS_QUEUE_MAX (20000), STATS_FLUSH_INTERVAL_S (0.5),
STATS_FLUSH_MAX_BATCH (500).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("gateway.stats")

_QUEUE_MAX = int(os.environ.get("STATS_QUEUE_MAX", "20000") or "20000")
_FLUSH_INTERVAL_S = float(os.environ.get("STATS_FLUSH_INTERVAL_S", "0.5") or "0.5")
_FLUSH_MAX_BATCH = int(os.environ.get("STATS_FLUSH_MAX_BATCH", "500") or "500")

_queue: "Optional[asyncio.Queue[dict]]" = None
_task: Optional[asyncio.Task] = None
_sessionmaker = None  # async_sessionmaker, set by start()
_dropped = 0


# ---------- public enqueue API (sync, non-blocking) --------------------------

def _enqueue(item: dict) -> None:
    global _dropped
    q = _queue
    if q is None:
        # Writer not started (unit tests / pre-startup) — stats are best-effort.
        return
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        _dropped += 1
        if _dropped % 1000 == 1:
            logger.warning("stats queue full (max=%d) — dropped %d updates so far",
                           _QUEUE_MAX, _dropped)


def record_stream_completion(request_id: str, ttft_ms: Optional[int],
                             pt: Optional[int], ct: Optional[int],
                             latency_ms: Optional[int] = None) -> None:
    """A streamed serverless request finished — stamp TTFT + token usage.
    `latency_ms` (stream duration) is metrics-only: used to derive TPS, not stored."""
    _enqueue({"kind": "serverless", "id": request_id, "ttft_ms": ttft_ms,
              "pt": pt, "ct": ct, "latency_ms": latency_ms})


def record_serverless_request(request_id: str, app_id: str, owner_id: Optional[int],
                              endpoint: str, model: Optional[str], *, is_stream: bool,
                              status: str, created_at, completed_at,
                              ttft_ms: Optional[int] = None, pt: Optional[int] = None,
                              ct: Optional[int] = None) -> None:
    """INSERT a whole serverless `requests` row (not an update of an existing one).

    Used by the proxy-mode (single-model VM) path, which forwards straight to the VM's
    vLLM and never goes through the queue/worker that normally inserts this row — so its
    traffic is otherwise invisible to the Activity dashboard. Goes through the SAME
    background writer as the update path so a burst of proxy requests costs one writer
    connection, not one pooled checkout per request (the pool-exhaustion incident)."""
    _enqueue({"kind": "serverless_insert", "id": request_id, "app_id": app_id,
              "owner_id": owner_id, "endpoint": endpoint, "model": model,
              "is_stream": is_stream, "status": status, "created_at": created_at,
              "completed_at": completed_at, "ttft_ms": ttft_ms, "pt": pt, "ct": ct})


def record_proxy_finish(request_id: str, status: str, *, status_code: Optional[int] = None,
                        latency_ms: Optional[int] = None, pt: Optional[int] = None,
                        ct: Optional[int] = None, error: Optional[str] = None,
                        upstream: Optional[str] = None, ttft_ms: Optional[int] = None) -> None:
    """A proxied request reached a terminal state — record its outcome + stats."""
    _enqueue({"kind": "proxy", "id": request_id, "status": status, "status_code": status_code,
              "latency_ms": latency_ms, "pt": pt, "ct": ct, "error": error,
              "upstream": upstream, "ttft_ms": ttft_ms})


# ---------- coalescing + flush ----------------------------------------------

_MERGE_SKIP = {"kind", "id"}


def _merge(dst: dict, src: dict) -> None:
    """Fold a later intent for the same row into an earlier one — last non-null wins."""
    for k, v in src.items():
        if k in _MERGE_SKIP:
            continue
        if v is not None:
            dst[k] = v


def _apply_serverless(row, it: dict, now: datetime) -> None:
    # Guards mirror the old _record_stream_completion: never clobber a richer
    # value a terminal-result mirror may have written.
    if it.get("ttft_ms") is not None and row.ttft_ms is None:
        row.ttft_ms = it["ttft_ms"]
    if row.completed_at is None:
        row.completed_at = now
    if row.status not in ("completed", "failed", "cancelled", "timeout"):
        row.status = "completed"
    pt, ct = it.get("pt"), it.get("ct")
    if pt is not None or ct is not None:
        out = dict(row.output or {})
        usage = dict(out.get("usage") or {})
        if pt is not None:
            usage.setdefault("prompt_tokens", pt)
        if ct is not None:
            usage.setdefault("completion_tokens", ct)
        out["usage"] = usage
        row.output = out
    # Per-app Prometheus TTFT/TPS histograms (in-memory; exposed at /{app_id}/metrics).
    try:
        from . import metrics as _metrics
        ttft_ms, latency_ms = it.get("ttft_ms"), it.get("latency_ms")
        ttft_s = (ttft_ms / 1000.0) if ttft_ms is not None else None
        tps = None
        if ct and latency_ms:
            gen_ms = latency_ms - ttft_ms if (ttft_ms is not None and latency_ms > ttft_ms) else latency_ms
            if gen_ms > 0:
                tps = ct / (gen_ms / 1000.0)
        if ttft_s is not None or tps is not None:
            model = (row.payload or {}).get("model") if isinstance(row.payload, dict) else None
            _metrics.observe_serverless_stream(row.app_id, model or "", ttft_s=ttft_s, tps=tps)
    except Exception:  # noqa: BLE001 — metrics are best-effort
        pass


def _apply_proxy(row, it: dict, now: datetime) -> None:
    row.status = it["status"]
    if it.get("status_code") is not None:
        row.status_code = it["status_code"]
    if it.get("latency_ms") is not None:
        row.latency_ms = it["latency_ms"]
    if it.get("ttft_ms") is not None:
        row.ttft_ms = it["ttft_ms"]
    if it.get("pt") is not None:
        row.prompt_tokens = it["pt"]
    if it.get("ct") is not None:
        row.completion_tokens = it["ct"]
    if it.get("upstream"):
        row.upstream = it["upstream"]
    if it.get("error"):
        row.error_text = str(it["error"])[:2048]
    row.completed_at = now
    # Per-proxy Prometheus metric (in-memory) — parity with the old _finish.
    try:
        from . import metrics as _metrics
        ttft_ms, latency_ms, ct = it.get("ttft_ms"), it.get("latency_ms"), it.get("ct")
        ttft_s = (ttft_ms / 1000.0) if ttft_ms is not None else None
        tps = None
        if ct and latency_ms:
            gen_ms = latency_ms - ttft_ms if (ttft_ms is not None and latency_ms > ttft_ms) else latency_ms
            if gen_ms > 0:
                tps = ct / (gen_ms / 1000.0)
        _metrics.observe_proxy(
            row.endpoint_id, row.model, (it.get("upstream") or row.upstream or ""), it["status"],
            (latency_ms / 1000.0) if latency_ms is not None else None, ttft_s=ttft_s, tps=tps,
        )
    except Exception:  # noqa: BLE001 — metrics are best-effort
        pass


def _insert_serverless(s, items: list[dict]) -> None:
    """Stage brand-new serverless rows (proxy-mode requests) for INSERT. Sync — only
    builds + s.add_all()s ORM objects; the caller awaits the batched commit."""
    from .db import Request as ReqRow
    objs = []
    for it in items:
        out = None
        if it.get("pt") is not None or it.get("ct") is not None:
            usage = {}
            if it.get("pt") is not None:
                usage["prompt_tokens"] = it["pt"]
            if it.get("ct") is not None:
                usage["completion_tokens"] = it["ct"]
            out = {"usage": usage}
        objs.append(ReqRow(
            request_id=it["id"], app_id=it["app_id"], owner_id=it["owner_id"],
            endpoint=it["endpoint"],
            payload=({"model": it["model"]} if it.get("model") else {}),
            status=it.get("status") or "completed", output=out,
            is_stream=bool(it.get("is_stream")),
            created_at=it["created_at"], completed_at=it.get("completed_at"),
            ttft_ms=it.get("ttft_ms"),
        ))
    s.add_all(objs)
    try:
        from . import metrics as _metrics
        for it in items:
            _metrics.observe_job_outcome(it["app_id"], it.get("status") or "completed")
    except Exception:  # noqa: BLE001 — metrics are best-effort
        pass


async def _flush(items: list[dict]) -> None:
    if not items or _sessionmaker is None:
        return
    from sqlalchemy import select
    from .db import Request as ReqRow
    from .proxy_api import ProxyRequest

    # Coalesce: the same row finishing/updating twice in one window → one write.
    merged: dict[tuple, dict] = {}
    for it in items:
        key = (it["kind"], it["id"])
        if key in merged:
            _merge(merged[key], it)
        else:
            merged[key] = dict(it)

    sv = {rid: it for (kind, rid), it in merged.items() if kind == "serverless"}
    px = {rid: it for (kind, rid), it in merged.items() if kind == "proxy"}
    ins = {rid: it for (kind, rid), it in merged.items() if kind == "serverless_insert"}
    now = datetime.now(timezone.utc)

    async with _sessionmaker() as s:
        if ins:
            _insert_serverless(s, list(ins.values()))
        if sv:
            rows = (await s.execute(select(ReqRow).where(ReqRow.request_id.in_(list(sv))))).scalars().all()
            for row in rows:
                try:
                    _apply_serverless(row, sv[row.request_id], now)
                except Exception:  # noqa: BLE001
                    logger.warning("stats apply failed for request %s", row.request_id, exc_info=True)
        if px:
            rows = (await s.execute(select(ProxyRequest).where(ProxyRequest.id.in_(list(px))))).scalars().all()
            for row in rows:
                try:
                    _apply_proxy(row, px[row.id], now)
                except Exception:  # noqa: BLE001
                    logger.warning("stats apply failed for proxy req %s", row.id, exc_info=True)
        try:
            await s.commit()
        except Exception:  # noqa: BLE001 — drop the batch rather than crash the writer
            logger.warning("stats batch commit failed (%d rows)", len(merged), exc_info=True)
            await s.rollback()


async def _drain() -> list[dict]:
    q = _queue
    assert q is not None
    first = await q.get()  # block until there is at least one update
    items = [first]
    # Let a small batch accumulate, then sweep everything queued in one txn.
    await asyncio.sleep(_FLUSH_INTERVAL_S)
    while len(items) < _FLUSH_MAX_BATCH:
        try:
            items.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


async def _run() -> None:
    logger.info("stats writer started (queue_max=%d, flush_interval=%.2fs, max_batch=%d)",
                _QUEUE_MAX, _FLUSH_INTERVAL_S, _FLUSH_MAX_BATCH)
    while True:
        try:
            await _flush(await _drain())
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — never let the writer die on a transient error
            logger.warning("stats writer loop error", exc_info=True)
            await asyncio.sleep(1.0)


# ---------- lifecycle --------------------------------------------------------

def start(sessionmaker) -> None:
    """Start the background writer. `sessionmaker` is an async_sessionmaker
    (i.e. db.session_factory())."""
    global _queue, _task, _sessionmaker
    if _task is not None:
        return
    _sessionmaker = sessionmaker
    _queue = asyncio.Queue(maxsize=_QUEUE_MAX)
    _task = asyncio.create_task(_run(), name="stats-writer")


async def stop() -> None:
    """Cancel the writer and flush whatever is still queued (called before the DB
    engine is disposed)."""
    global _task, _queue
    t, q = _task, _queue
    _task = None
    if t is not None:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    if q is not None and not q.empty():
        leftover: list[dict] = []
        while not q.empty():
            try:
                leftover.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break
        if leftover:
            try:
                await _flush(leftover)
            except Exception:  # noqa: BLE001
                logger.warning("final stats flush failed (%d rows)", len(leftover), exc_info=True)
    _queue = None
