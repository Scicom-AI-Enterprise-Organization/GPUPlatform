"""Redis-backed cross-replica coordination for the LLM proxy.

Two features that make the proxy correct across MANY gateway replicas — both OPT-IN
(`PROXY_CLUSTER=1`) and both degrading to the per-replica in-memory behavior when
disabled or Redis is unreachable, so single-replica deploys are byte-unchanged:

1. GLOBAL concurrency cap (Phase 2). A proxy endpoint's `max_concurrency` is
   enforced ACROSS all replicas instead of per-replica (where N replicas gave an
   effective N× cap). Implemented as a Redis ZSET of leased slots: acquire purges
   expired leases, counts, and admits if under the limit; the lease is renewed by
   the per-replica sync loop so a crashed replica's slots free themselves within a
   lease TTL instead of wedging the cap forever. Acquire/release are on the request
   hot path (one Lua round-trip each); renewal is batched off it.

2. GLOBAL live-request registry + cancel fan-out (Phase 3). Each in-flight request
   is mirrored to Redis (hash + per-endpoint index) by the SAME sync loop — so the
   admin queue view and inflight/queued counts span every replica, not just the one
   that received the API call. This is entirely OFF the request hot path (the loop
   reconciles the local `_live` dict to Redis every couple seconds). Cancel/flush
   publish the request id on a pub/sub channel; every replica's subscriber sets the
   local cancel event if it's the one holding that request.

Key layout (all under the `proxy:` prefix):
  proxy:sem:{endpoint_id}      ZSET  slot_token -> lease_expiry_ms   (limiter)
  proxy:live:{request_id}      HASH  request metadata                (registry)
  proxy:live:ep:{endpoint_id}  SET   request_ids on this endpoint    (index)
  proxy:cancel                 pub/sub channel, payload = request_id (cancel)
"""
from __future__ import annotations

import logging
import os
import socket
import time
from typing import Any, Optional

logger = logging.getLogger("gateway.proxy.cluster")

CANCEL_CHANNEL = "proxy:cancel"
# Live-hash TTL: comfortably > the sync interval so an entry the loop is still
# refreshing never lapses mid-request; a crashed replica's entries expire within it.
LIVE_TTL_S = int(os.environ.get("PROXY_CLUSTER_LIVE_TTL_S", "30") or "30")
SYNC_INTERVAL_S = max(1.0, float(os.environ.get("PROXY_CLUSTER_SYNC_S", "2") or "2"))
# Slot lease TTL — same reasoning: > sync interval so renewal always beats expiry.
SLOT_LEASE_MS = int(os.environ.get("PROXY_CLUSTER_SLOT_LEASE_S", "30") or "30") * 1000


def enabled() -> bool:
    return os.environ.get("PROXY_CLUSTER", "0") == "1"


def replica_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


# ---------- global concurrency limiter (ZSET lease) --------------------------
#
# now_ms is passed from the client (not redis TIME) so the script stays a pure
# write — Redis forbids writes after a non-deterministic command like TIME on
# older servers, and client clock skew of a few ms is irrelevant to a 30s lease.

_ACQUIRE_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local token = ARGV[2]
local now = tonumber(ARGV[3])
local lease = tonumber(ARGV[4])
redis.call('zremrangebyscore', key, '-inf', now)
local n = redis.call('zcard', key)
if n < limit then
  redis.call('zadd', key, now + lease, token)
  redis.call('pexpire', key, lease * 2)
  return 1
else
  return 0
end
"""

_RENEW_LUA = """
local key = KEYS[1]
local token = ARGV[1]
local now = tonumber(ARGV[2])
local lease = tonumber(ARGV[3])
if redis.call('zscore', key, token) then
  redis.call('zadd', key, now + lease, token)
  redis.call('pexpire', key, lease * 2)
  return 1
else
  return 0
end
"""

_RELEASE_LUA = "return redis.call('zrem', KEYS[1], ARGV[1])"


def _sem_key(endpoint_id: str) -> str:
    return f"proxy:sem:{endpoint_id}"


async def limiter_acquire(redis, endpoint_id: str, limit: int, token: str) -> bool:
    """Try to take one global slot. Returns True if admitted (caller must release),
    False if the endpoint is at its global cap. Fails OPEN (True) on a Redis error so
    a Redis blip degrades to letting the request through rather than blocking it."""
    try:
        now_ms = int(time.time() * 1000)
        r = await redis.eval(_ACQUIRE_LUA, 1, _sem_key(endpoint_id), limit, token, now_ms, SLOT_LEASE_MS)
        return bool(r)
    except Exception:
        logger.debug("limiter_acquire failed (fail-open)", exc_info=True)
        return True


async def limiter_renew(redis, endpoint_id: str, token: str) -> None:
    try:
        now_ms = int(time.time() * 1000)
        await redis.eval(_RENEW_LUA, 1, _sem_key(endpoint_id), token, now_ms, SLOT_LEASE_MS)
    except Exception:
        logger.debug("limiter_renew failed", exc_info=True)


async def limiter_release(redis, endpoint_id: str, token: str) -> None:
    try:
        await redis.eval(_RELEASE_LUA, 1, _sem_key(endpoint_id), token)
    except Exception:
        logger.debug("limiter_release failed", exc_info=True)


async def limiter_count(redis, endpoint_id: str) -> int:
    """Live global slot occupancy (expired leases purged first) — for metrics/debug."""
    try:
        now_ms = int(time.time() * 1000)
        await redis.zremrangebyscore(_sem_key(endpoint_id), "-inf", now_ms)
        return int(await redis.zcard(_sem_key(endpoint_id)))
    except Exception:
        return 0


# ---------- global live-request registry -------------------------------------

def _live_key(rid: str) -> str:
    return f"proxy:live:{rid}"


def _ep_index_key(endpoint_id: str) -> str:
    return f"proxy:live:ep:{endpoint_id}"


async def live_upsert(redis, rid: str, meta: dict[str, Any]) -> None:
    """Write/refresh one live request's mirror (hash + endpoint index), TTL-bounded.
    Called by the sync loop, not the hot path. Values are stringified for the hash."""
    try:
        ep = meta.get("endpoint_id") or ""
        flat = {k: ("" if v is None else str(v)) for k, v in meta.items()}
        pipe = redis.pipeline()
        pipe.hset(_live_key(rid), mapping=flat)
        pipe.expire(_live_key(rid), LIVE_TTL_S)
        pipe.sadd(_ep_index_key(ep), rid)
        pipe.expire(_ep_index_key(ep), LIVE_TTL_S * 4)
        await pipe.execute()
    except Exception:
        logger.debug("live_upsert failed", exc_info=True)


async def live_remove(redis, rid: str, endpoint_id: str) -> None:
    try:
        pipe = redis.pipeline()
        pipe.delete(_live_key(rid))
        pipe.srem(_ep_index_key(endpoint_id), rid)
        await pipe.execute()
    except Exception:
        logger.debug("live_remove failed", exc_info=True)


async def live_list_endpoint(redis, endpoint_id: str) -> list[dict[str, Any]]:
    """All live requests for an endpoint across replicas. Prunes index entries whose
    hash has expired (crashed replica) as it goes."""
    out: list[dict[str, Any]] = []
    try:
        rids = list(await redis.smembers(_ep_index_key(endpoint_id)))
        if not rids:
            return out
        pipe = redis.pipeline()
        for rid in rids:
            pipe.hgetall(_live_key(rid))
        rows = await pipe.execute()
        stale: list[str] = []
        for rid, h in zip(rids, rows):
            if not h:
                stale.append(rid)
                continue
            out.append(h)
        if stale:
            await redis.srem(_ep_index_key(endpoint_id), *stale)
    except Exception:
        logger.debug("live_list_endpoint failed", exc_info=True)
    return out


async def live_counts_endpoint(redis, endpoint_id: str) -> tuple[int, int]:
    """(inflight, queued) across replicas for an endpoint."""
    rows = await live_list_endpoint(redis, endpoint_id)
    inflight = sum(1 for r in rows if r.get("state") == "running")
    queued = sum(1 for r in rows if r.get("state") == "queued")
    return inflight, queued


# ---------- cancel fan-out (pub/sub) -----------------------------------------

async def publish_cancel(redis, rid: str) -> bool:
    """Broadcast a cancel for `rid`. Returns True if at least one replica was
    subscribed (best-effort signal — the holder may still be mid-connect)."""
    try:
        n = await redis.publish(CANCEL_CHANNEL, rid)
        return int(n) > 0
    except Exception:
        logger.debug("publish_cancel failed", exc_info=True)
        return False
