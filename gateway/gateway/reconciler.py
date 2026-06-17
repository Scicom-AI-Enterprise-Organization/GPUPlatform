"""Reconciler loop: trust the cloud API as the source of truth for liveness.

Every 5s:
  1. Ask the provider what's actually running (its `list_machines()`).
  2. SCAN Redis for our `worker_index:*` sets.
  3. Compute the diff:
       - in-redis but NOT in-provider  → pod is gone; SREM from index, DEL state key
                                          (this catches manual terminations,
                                          PI-side crashes, billing kills, ...). But
                                          if it's STILL heartbeating it's a ZOMBIE
                                          (the worker process outlived its provider
                                          record) — GC alone is futile because its
                                          next heartbeat re-adds it; re-arm `drain`
                                          so the worker shuts itself down.
       - in-provider but NOT in-redis  → orphan pod; log a warning so the
                                          operator can investigate
  4. Sleep 5s, repeat.

Tracks the "machines we provisioned but never registered" case via the
`worker:{id}:provisioning` ttl set by the autoscaler — if expired AND the
provider doesn't know about the pod either, we clean up our state.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from sqlalchemy import select

from .db import App

if TYPE_CHECKING:
    import redis.asyncio as redis_async
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    from .provider import Provider

logger = logging.getLogger("gateway.reconciler")

TICK_S = 5.0
# An orphan (on the provider, not in Redis) is warned once, then reaped if it's
# STILL unregistered this many seconds later. The grace must exceed worker
# boot+register time so we never reap a machine that's merely still booting.
ORPHAN_REAP_GRACE_S = 90.0
# A machine that's gone from every provider listing but is STILL heartbeating is a
# zombie. We wait this long (to rule out a one-tick provider-list blip) before
# re-arming its drain flag so the next heartbeat shuts the worker down. VM providers
# never blip (their listing is authoritative Redis); this grace guards the RunPod
# fallback, whose API could momentarily return a stale list.
ZOMBIE_DRAIN_GRACE_S = 30.0


async def reconciler_loop(
    rdb: "redis_async.Redis",
    provider: "Provider",
    sm: "async_sessionmaker[AsyncSession] | None" = None,
    provider_cache: dict | None = None,
) -> None:
    logger.info("reconciler running (provider=%s)", provider.name)
    if provider_cache is None:
        provider_cache = {}
    while True:
        try:
            await asyncio.sleep(TICK_S)
            await tick(rdb, provider, sm, provider_cache)
        except asyncio.CancelledError:
            logger.info("reconciler cancelled")
            raise
        except Exception:
            logger.exception("reconciler tick failed")


async def _collect_providers(
    rdb: "redis_async.Redis",
    fallback: "Provider",
    sm: "async_sessionmaker[AsyncSession] | None",
    provider_cache: dict,
) -> list["Provider"]:
    """Every distinct provider referenced by a live app, plus the global
    fallback. Used so VM / per-row providers' machines are visible to the diff
    and never wrongly GC'd."""
    from .provider import resolve_app_provider

    providers: list["Provider"] = []
    seen: set[int] = set()
    if fallback is not None:
        providers.append(fallback)
        seen.add(id(fallback))
    if sm is not None:
        async with sm() as session:
            apps = list((await session.execute(select(App))).scalars().all())
            for app in apps:
                if not getattr(app, "provider_id", None):
                    continue
                try:
                    prov = await resolve_app_provider(
                        session, app, redis=rdb, fallback=fallback, cache=provider_cache
                    )
                except Exception:
                    continue
                if id(prov) not in seen:
                    seen.add(id(prov))
                    providers.append(prov)
    return providers


async def tick(
    rdb: "redis_async.Redis",
    provider: "Provider",
    sm: "async_sessionmaker[AsyncSession] | None" = None,
    provider_cache: dict | None = None,
) -> None:
    if provider_cache is None:
        provider_cache = {}
    providers = await _collect_providers(rdb, provider, sm, provider_cache)

    machine_provider: dict[str, "Provider"] = {}
    for prov in providers:
        try:
            for m in await prov.list_machines():
                machine_provider.setdefault(m, prov)
        except NotImplementedError:
            continue
        except Exception:
            # If ANY provider's listing fails we can't safely diff — skip the
            # whole tick rather than risk GC'ing machines we just couldn't see.
            logger.exception("list_machines failed for provider=%s; skipping tick", getattr(prov, "name", "?"))
            return
    provider_machines = set(machine_provider)

    redis_machines: set[str] = set()
    async for key in rdb.scan_iter(match="worker_index:*"):
        members = await rdb.smembers(key)
        redis_machines.update(members)

    gone = redis_machines - provider_machines
    orphans = provider_machines - redis_machines

    now = time.time()
    # Reset the zombie-watch clock for anything a provider currently lists: a machine
    # that briefly vanished and came back must start a FRESH grace if it later
    # disappears for real, else a stale gone_since would drain it instantly.
    for machine_id in provider_machines:
        await rdb.delete(
            f"reconciler:gone_since:{machine_id}", f"reconciler:drain_logged:{machine_id}"
        )
    for machine_id in gone:
        # A machine in our index but absent from every provider's listing. If its
        # worker process is still alive (heartbeat keeps worker:{id} fresh) it's a
        # ZOMBIE — terminated/forgotten/purged provider-side, yet never actually
        # died. GC'ing it is futile: its next heartbeat re-adds it to worker_index
        # and we churn (+ log-spam) every tick forever. So drain it instead — the
        # worker-agent already shuts itself down on a `drain` heartbeat reply.
        alive = await rdb.exists(f"worker:{machine_id}")  # checked BEFORE _remove deletes it
        await _remove_machine(rdb, machine_id)
        if not alive:
            # Truly gone (no recent heartbeat) → clean, one-time GC.
            await rdb.delete(
                f"reconciler:orphan_since:{machine_id}",
                f"reconciler:gone_since:{machine_id}",
                f"reconciler:drain_logged:{machine_id}",
            )
            logger.info("reconciler: %s no longer in provider, GC'd from Redis", machine_id)
            continue
        # Still heartbeating. Require the gone-but-alive state to persist past a grace
        # window first, so a one-tick provider-list blip can't drain a healthy worker.
        gk = f"reconciler:gone_since:{machine_id}"
        since = await rdb.get(gk)
        if since is None:
            await rdb.set(gk, str(now), ex=3600)
            logger.warning(
                "reconciler: %s gone from provider but still heartbeating — watching for zombie",
                machine_id,
            )
        elif now - float(since) >= ZOMBIE_DRAIN_GRACE_S:
            # Confirmed zombie: re-arm drain (deleted by _remove_machine just above)
            # so its next heartbeat tells the worker-agent to drain + exit. Log once.
            await rdb.set(f"worker:{machine_id}:drain", "1", ex=600)
            if not await rdb.exists(f"reconciler:drain_logged:{machine_id}"):
                await rdb.set(f"reconciler:drain_logged:{machine_id}", "1", ex=3600)
                logger.warning(
                    "reconciler: %s is a zombie (gone from provider, still alive) — sent drain to shut it down",
                    machine_id,
                )

    for machine_id in orphans:
        since_key = f"reconciler:orphan_since:{machine_id}"
        since = await rdb.get(since_key)
        if since is None:
            # First time seen orphaned — record + warn ONCE (don't re-warn every
            # 5s tick). A just-provisioned worker that hasn't registered yet lands
            # here; the grace below lets it register before we'd reap it.
            await rdb.set(since_key, str(now), ex=3600)
            logger.warning(
                "reconciler: %s on provider but not in Redis (orphan) — reaping in %ds if still unregistered",
                machine_id, int(ORPHAN_REAP_GRACE_S),
            )
        elif now - float(since) >= ORPHAN_REAP_GRACE_S:
            # Still unregistered after the grace → a dead provision remnant. Reap it
            # from Redis AND the provider's listing so it stops being re-warned.
            await _remove_machine(rdb, machine_id)
            prov = machine_provider.get(machine_id)
            if prov is not None:
                try:
                    await prov.forget_machine(machine_id)
                except Exception:
                    logger.exception("reconciler: forget_machine(%s) failed", machine_id)
            await rdb.delete(since_key)
            logger.info("reconciler: reaped stale orphan %s (unregistered for >%ds)", machine_id, int(ORPHAN_REAP_GRACE_S))
        # else: within grace — stay quiet (already warned once).


async def _remove_machine(rdb: "redis_async.Redis", machine_id: str) -> None:
    """Delete the worker's state and remove from any index that contains it."""
    await rdb.delete(f"worker:{machine_id}")
    await rdb.delete(f"worker:{machine_id}:drain")
    async for key in rdb.scan_iter(match="worker_index:*"):
        await rdb.srem(key, machine_id)
