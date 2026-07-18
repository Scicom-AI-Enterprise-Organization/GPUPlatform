"""Leader election for the gateway's singleton control-plane loops.

The gateway is a monolith: the same process serves the stateless data plane (the
LLM proxy, API reads) AND runs mutating background controllers (autoscaler,
reconciler, vm_watchdog, the janitors, gitops, log-archive flusher). The data
plane is horizontally scalable — all its state lives in Postgres/Redis — but the
controllers are NOT safe to run on more than one replica at a time (two
autoscalers fight → double-spawn pods; two janitors race → double-finalize).

So for multi-replica HA we split by leadership: EVERY replica serves HTTP, but
only ONE — the leader — runs the controller workload. Leadership is a Redis lock
(`SET <key> <id> NX PX <ttl>`); the holder renews it on an interval, and if the
leader dies its lock lapses so a follower's `SET NX` wins within ~ttl seconds and
takes over the workload (running the same startup orphan-reconciles first).

`GATEWAY_HA` unset/0 (default, single replica) => this replica is ALWAYS the
leader with NO Redis lock in the path — behavior is byte-identical to the old
inline startup, so local dev and single-replica prod are unchanged. Set
`GATEWAY_HA=1` on every replica of a multi-replica Deployment.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from typing import Awaitable, Callable, Optional

logger = logging.getLogger("gateway.leader")

LOCK_KEY = os.environ.get("LEADER_LOCK_KEY", "gateway:leader")
# TTL must exceed renew interval by a comfortable margin so a single slow tick
# (GC pause, Redis blip) doesn't drop leadership. Default 15s ttl / 5s renew.
TTL_MS = max(2000, int(os.environ.get("LEADER_TTL_S", "15") or "15") * 1000)
RENEW_S = max(1.0, float(os.environ.get("LEADER_RENEW_S", "5") or "5"))

# Renew / release only if WE still own the lock — never stomp a lock a follower
# grabbed after our TTL lapsed (the classic distributed-lock correctness bug).
_RENEW_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('pexpire', KEYS[1], ARGV[2])
else
  return 0
end
"""
_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
else
  return 0
end
"""

# workload(app) -> list of asyncio.Task it spawned. Must return promptly (spawn
# tasks, don't await long work inline) so the coordinator keeps renewing while
# the workload's own orphan-reconciles run.
Workload = Callable[[object], Awaitable[list[asyncio.Task]]]


def ha_enabled() -> bool:
    return os.environ.get("GATEWAY_HA", "0") == "1"


def instance_id() -> str:
    # Distinct per pod AND per process restart so a stale lock value never
    # matches a new process (which would let it renew a lock it didn't take).
    return f"{socket.gethostname()}-{os.getpid()}"


class LeaderCoordinator:
    """Owns the leader lock + the leader-only workload's lifecycle.

    Run `run()` as a background task; call `stop()` at shutdown. `app.state.is_leader`
    reflects current leadership for /leader + metrics.
    """

    def __init__(self, app, workload: Workload, redis=None):
        self.app = app
        self.workload = workload
        self.redis = redis if redis is not None else app.state.redis
        self.id = instance_id()
        self.is_leader = False
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        app.state.is_leader = False
        app.state.leader_id = self.id

    async def run(self) -> None:
        if not ha_enabled():
            logger.info("leader: HA disabled (GATEWAY_HA!=1) — this replica is the sole leader")
            await self._acquire()
            await self._stop.wait()
            await self._release_workload()
            return

        renew = self.redis.register_script(_RENEW_LUA)
        release = self.redis.register_script(_RELEASE_LUA)
        logger.info("leader: HA enabled — instance=%s key=%s ttl=%dms renew=%.1fs",
                    self.id, LOCK_KEY, TTL_MS, RENEW_S)
        ttl_s = TTL_MS / 1000.0
        last_renew_ok = time.monotonic()
        try:
            while not self._stop.is_set():
                try:
                    if self.is_leader:
                        ok = await renew(keys=[LOCK_KEY], args=[self.id, TTL_MS])
                        if ok:
                            last_renew_ok = time.monotonic()
                            try:
                                from . import metrics as _metrics
                                _metrics.loop_heartbeat("leader")
                            except Exception:  # noqa: BLE001 — metrics are best-effort
                                pass
                        else:
                            logger.warning("leader: lost lock (renew rejected) — resigning, stopping workload")
                            await self._release_workload()
                    else:
                        got = await self.redis.set(LOCK_KEY, self.id, nx=True, px=TTL_MS)
                        if got:
                            logger.info("leader: acquired leadership (instance=%s)", self.id)
                            await self._acquire()
                            last_renew_ok = time.monotonic()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Redis blip. A single failed renew doesn't mean we lost the lock —
                    # the key's TTL is still ticking, so keep the workload running for now.
                    # BUT if we can't reach Redis for longer than the TTL, the lock has
                    # actually expired and a follower may have promoted → SELF-FENCE
                    # (stop the workload) to bound split-brain to ~TTL. We reacquire once
                    # Redis is back and SET NX succeeds.
                    logger.exception("leader: coordinator tick failed")
                    if self.is_leader and (time.monotonic() - last_renew_ok) > ttl_s:
                        logger.error("leader: no successful renew for >%.0fs (TTL) — self-fencing, "
                                     "stopping workload to avoid split-brain", ttl_s)
                        await self._release_workload()
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=RENEW_S)
                except asyncio.TimeoutError:
                    pass
        finally:
            await self._release_workload()
            # Best-effort hand-off: drop the lock so a follower promotes immediately
            # instead of waiting out the TTL.
            try:
                await release(keys=[LOCK_KEY], args=[self.id])
            except Exception:
                pass

    async def _acquire(self) -> None:
        self.is_leader = True
        self.app.state.is_leader = True
        try:
            self._tasks = await self.workload(self.app)
            logger.info("leader: workload started (%d task groups)", len(self._tasks))
        except Exception:
            logger.exception("leader: workload startup failed")
            self._tasks = []

    async def _release_workload(self) -> None:
        if not self.is_leader and not self._tasks:
            return
        self.is_leader = False
        self.app.state.is_leader = False
        tasks, self._tasks = self._tasks, []
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    def stop(self) -> None:
        self._stop.set()
