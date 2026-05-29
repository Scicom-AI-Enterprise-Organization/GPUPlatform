"""Reconciler loop: trust the cloud API as the source of truth for liveness.

Every 5s:
  1. Ask the provider what's actually running (its `list_machines()`).
  2. SCAN Redis for our `worker_index:*` sets.
  3. Compute the diff:
       - in-redis but NOT in-provider  → pod is gone; SREM from index, DEL state key
                                          (this catches manual terminations,
                                          PI-side crashes, billing kills, ...)
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
from typing import TYPE_CHECKING

from sqlalchemy import select

from .db import App

if TYPE_CHECKING:
    import redis.asyncio as redis_async
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    from .provider import Provider

logger = logging.getLogger("gateway.reconciler")

TICK_S = 5.0


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

    provider_machines: set[str] = set()
    for prov in providers:
        try:
            provider_machines.update(await prov.list_machines())
        except NotImplementedError:
            continue
        except Exception:
            # If ANY provider's listing fails we can't safely diff — skip the
            # whole tick rather than risk GC'ing machines we just couldn't see.
            logger.exception("list_machines failed for provider=%s; skipping tick", getattr(prov, "name", "?"))
            return

    redis_machines: set[str] = set()
    async for key in rdb.scan_iter(match="worker_index:*"):
        members = await rdb.smembers(key)
        redis_machines.update(members)

    gone = redis_machines - provider_machines
    orphans = provider_machines - redis_machines

    for machine_id in gone:
        await _remove_machine(rdb, machine_id)
        logger.info("reconciler: %s no longer in provider, GC'd from Redis", machine_id)

    for machine_id in orphans:
        logger.warning(
            "reconciler: %s exists on provider but not in Redis (orphan)",
            machine_id,
        )


async def _remove_machine(rdb: "redis_async.Redis", machine_id: str) -> None:
    """Delete the worker's state and remove from any index that contains it."""
    await rdb.delete(f"worker:{machine_id}")
    await rdb.delete(f"worker:{machine_id}:drain")
    async for key in rdb.scan_iter(match="worker_index:*"):
        await rdb.srem(key, machine_id)
