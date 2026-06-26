"""Persistent log archival for serverless endpoints.

When an app has a kind="s3" log-archive Storage set (``App.storage_id``, mirrored
to Redis at ``app:{id}:log_storage_id``), the worker-log ingest path enqueues
every batch here. A background flusher periodically appends the buffered lines to
that storage as immutable, append-only parts:

    serverless-logs/{app_id}/{slug}/{session}/{seq:08d}.log

and upserts a ``ServerlessLogArchive`` row — the searchable index the Logs tab
browses. This stores logs UNCAPPED, unlike the Redis live tail (5000 lines / 1h
TTL) which remains the fallback when no storage is configured.

One archived "log file" == one ``(app_id, slug, session)``: a single vLLM launch
(or the worker-agent's run), reconstructed by concatenating its parts in seq order.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select

from . import storage_backends
from .db import ServerlessLogArchive, Storage

logger = logging.getLogger("gateway.log_archive")

ARCHIVE_ROOT = "serverless-logs"
FLUSH_INTERVAL_S = 15

# Pending lines keyed by (app_id, slug, session). Drained under _lock by the flusher.
#   value = {"model": str, "storage_id": str, "lines": list[str]}
_buf: dict[tuple[str, str, str], dict] = {}
_lock = asyncio.Lock()

# storage_id -> StorageBackend. Cached so we don't rebuild a boto3 client per flush;
# cleared on a gateway restart (which also re-reads any storage config change).
_backends: dict[str, storage_backends.StorageBackend] = {}


def archive_prefix(app_id: str, slug: str, session: str) -> str:
    """Object-key prefix (relative to the storage root) for one file's parts."""
    return f"{ARCHIVE_ROOT}/{app_id}/{slug}/{session}"


def session_to_dt(session: str) -> datetime:
    """Parse a launch session ("YYYYMMDD-HHMMSS") to a UTC datetime; now() on miss."""
    try:
        return datetime.strptime(session, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


async def enqueue(
    app_id: str, slug: str, model: str, session: str, storage_id: str, lines: list[str]
) -> None:
    """Buffer a batch of log lines (chronological order) for archival."""
    if not lines:
        return
    async with _lock:
        ent = _buf.get((app_id, slug, session))
        if ent is None:
            ent = {"model": model, "storage_id": storage_id, "lines": []}
            _buf[(app_id, slug, session)] = ent
        ent["model"] = model
        ent["storage_id"] = storage_id
        ent["lines"].extend(lines)


def _backend_for(store: Storage) -> storage_backends.StorageBackend:
    b = _backends.get(store.id)
    if b is None:
        b = storage_backends.resolve_backend(store)
        _backends[store.id] = b
    return b


async def _flush_entry(session_maker, redis, key, ent) -> None:
    app_id, slug, session = key
    lines: list[str] = ent["lines"]
    if not lines:
        return
    storage_id: str = ent["storage_id"]
    model: str = ent["model"]
    prefix = archive_prefix(app_id, slug, session)

    # 1) Short txn: resolve the storage backend + this file's row (creating it on
    #    first sight) and read the next part sequence. We deliberately do NOT hold
    #    the transaction across the S3 upload below — a slow/stuck upload would
    #    otherwise leave an idle-in-transaction connection holding locks (reaped at
    #    DB_IDLE_IN_TXN_TIMEOUT_MS, losing the flush) and add lock churn on deploys.
    async with session_maker() as s:
        store = await s.get(Storage, storage_id)
        if store is None or not getattr(store, "enabled", True) or store.kind != "s3":
            logger.warning(
                "log_archive: dropping %d line(s) for app=%s slug=%s — storage %s unusable",
                len(lines), app_id, slug, storage_id,
            )
            return
        backend = _backend_for(store)
        row = (await s.execute(
            select(ServerlessLogArchive).where(
                ServerlessLogArchive.app_id == app_id,
                ServerlessLogArchive.slug == slug,
                ServerlessLogArchive.session == session,
            )
        )).scalar_one_or_none()
        if row is None:
            row = ServerlessLogArchive(
                id=f"slog-{secrets.token_hex(8)}",
                app_id=app_id, storage_id=storage_id, slug=slug, model=model,
                session=session, key_prefix=prefix, next_seq=0, bytes=0, lines=0,
                started_at=session_to_dt(session),
            )
            s.add(row)
            await s.commit()
        row_id = row.id
        seq = int(row.next_seq or 0)

    # 2) Upload the part OUTSIDE any DB transaction. Single sequential flusher →
    #    no other writer races on `seq` for this (app, slug, session).
    data = ("\n".join(lines) + "\n").encode("utf-8", "replace")
    key_obj = f"{prefix}/{seq:08d}.log"
    await asyncio.to_thread(backend.put_bytes, key_obj, data)  # boto3 is sync
    crash = await redis.get(f"wlog_crash:{app_id}:{slug}:{session}")

    # 3) Short txn: bump the counters now that the bytes are durably stored.
    async with session_maker() as s2:
        row = await s2.get(ServerlessLogArchive, row_id)
        if row is None:
            return
        row.next_seq = seq + 1
        row.bytes = (row.bytes or 0) + len(data)
        row.lines = (row.lines or 0) + len(lines)
        row.model = model
        row.updated_at = datetime.now(timezone.utc)
        if crash:
            row.crash = crash[:2048]
        await s2.commit()


async def flush_once(session_maker, redis) -> int:
    async with _lock:
        if not _buf:
            return 0
        drained = dict(_buf)
        _buf.clear()
    flushed = 0
    for key, ent in drained.items():
        try:
            await _flush_entry(session_maker, redis, key, ent)
            flushed += len(ent["lines"])
        except Exception:  # noqa: BLE001 — one bad file shouldn't stall the rest
            logger.warning("log_archive: flush failed for %s", key, exc_info=True)
    return flushed


async def flusher_loop(session_maker, redis) -> None:
    """Drain the buffer to storage every FLUSH_INTERVAL_S until cancelled."""
    logger.info("log-archive flusher started (interval=%ss)", FLUSH_INTERVAL_S)
    while True:
        try:
            await asyncio.sleep(FLUSH_INTERVAL_S)
            await flush_once(session_maker, redis)
        except asyncio.CancelledError:
            # Final best-effort drain so a clean shutdown doesn't lose the tail.
            try:
                await flush_once(session_maker, redis)
            except Exception:  # noqa: BLE001
                pass
            raise
        except Exception:  # noqa: BLE001 — loop must survive transient failures
            logger.warning("log-archive flusher tick failed", exc_info=True)


# ---- read side (used by the logs endpoints) --------------------------------

def backend_for_storage(store: Storage) -> storage_backends.StorageBackend:
    """Public accessor for the cached backend (endpoints reuse the flusher's cache)."""
    return _backend_for(store)


def list_part_keys(backend: storage_backends.StorageBackend, key_prefix: str) -> list[str]:
    """Sorted (chronological) part keys for one archived file."""
    parts = backend.list_prefix(key_prefix)
    return sorted(p["key"] for p in parts if p["key"].endswith(".log"))


async def read_all_text(backend: storage_backends.StorageBackend, key_prefix: str) -> str:
    """Concatenate every part of an archived file into one string."""
    keys = await asyncio.to_thread(list_part_keys, backend, key_prefix)

    def _read() -> str:
        chunks: list[str] = []
        for k in keys:
            r = backend.open_reader(k)
            try:
                chunks.append(r.read().decode("utf-8", "replace"))
            finally:
                close = getattr(r, "close", None)
                if close:
                    close()
        return "".join(chunks)

    return await asyncio.to_thread(_read)


def iter_parts_bytes(backend: storage_backends.StorageBackend, key_prefix: str):
    """Generator yielding each part's raw bytes in order — for streaming downloads.
    Runs sync (caller wraps it in a StreamingResponse)."""
    for k in list_part_keys(backend, key_prefix):
        r = backend.open_reader(k)
        try:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                yield chunk if isinstance(chunk, bytes) else chunk.encode("utf-8", "replace")
        finally:
            close = getattr(r, "close", None)
            if close:
                close()
