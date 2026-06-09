"""Per-app autoscaler loop.

Runs at 1Hz. For every registered app (loaded from Postgres):
  - sample queue length, in-flight tasks, current worker count
  - desired = ceil(queue_len / tasks_per_container), capped at max_containers
  - if desired > current: provider.provision()
  - if queue+inflight=0 AND idle > idle_timeout_s: terminate one worker (until 0)
  - idle_timeout_s == 0 disables teardown entirely (always-on)

Worker liveness is tracked by Redis TTL on `worker:{machine_id}` keys.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import secrets
import time
from typing import TYPE_CHECKING

from sqlalchemy import select

from .db import App

if TYPE_CHECKING:
    import redis.asyncio as redis_async
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    from .provider import Provider

logger = logging.getLogger("gateway.autoscaler")

TICK_S = 1.0
REGISTRATION_TOKEN_TTL_S = 1800  # 30 min — covers slow ECR pulls + vLLM model load
PROVISION_COOLDOWN_S = 60  # back off this long after a provider provision failure
PROVISION_ERROR_TTL_S = 600  # how long the UI keeps showing the last error

# Worker events ring: per-worker timeline of lifecycle events the UI shows
# under "gateway events" — provisioned, registered, scaled, terminated, etc.
WORKER_EVENTS_CAP = 200
WORKER_EVENTS_TTL_S = 3600


def build_multi_model_config(app) -> dict:
    """Translate an App's `models` member spec into the JSON the multi-model
    worker-agent consumes (`MULTI_MODEL_CONFIG`).

    Assigns each member to a GPU "slot" of `tp` consecutive GPUs, packing models
    of the same tp round-robin across the floor(total/tp) slots for that tp. When
    members outnumber slots, the extras reuse earlier slots' GPU sets — the
    worker time-shares those GPUs via vLLM sleep/wake. Per-model vLLM servers
    bind localhost ports starting at MULTI_MODEL_PORT_BASE (default 18001) — high
    enough to dodge anything already running on a shared VM (8000/8001/…).
    """
    total = int(getattr(app, "gpu_count", 0) or 0)
    # Physical GPU ids this endpoint may use. An explicit visible_devices pin
    # (e.g. "4,5,6,7") becomes the universe slots are carved from; otherwise we
    # use 0..total-1 (the historical behaviour). gpu_indices below are real
    # physical ids the worker passes to CUDA_VISIBLE_DEVICES per model.
    vd = (getattr(app, "visible_devices", None) or "").strip()
    if vd:
        phys = [int(x.strip()) for x in vd.split(",") if x.strip() != ""]
    else:
        phys = list(range(total))
    universe = len(phys) or total
    members = getattr(app, "models", None) or []
    default_level = int(getattr(app, "sleep_level", 1) or 1)
    slot_cursor: dict[int, int] = {}
    out: list[dict] = []
    # Per-endpoint port window so multiple fleets sharing ONE VM don't collide on
    # localhost ports. Coexisting endpoints must use disjoint GPUs, so the lowest
    # physical GPU id this endpoint owns gives each a stable, non-overlapping
    # window (e.g. GPUs 0-5 → 18001+, GPUs 6,7 → 18601+ at stride 100).
    port_base = int(os.environ.get("MULTI_MODEL_PORT_BASE", "18001") or "18001")
    port_stride = int(os.environ.get("MULTI_MODEL_PORT_STRIDE", "100") or "100")
    port = port_base + (min(phys) if phys else 0) * port_stride
    for m in members:
        model = m["model"]
        tp = max(1, int(m.get("tp", 1) or 1))
        pp = max(1, int(m.get("pp", 1) or 1))
        width = tp * pp  # GPUs this model occupies = tensor × pipeline parallel
        # Explicit pin wins: the user chose exactly which physical GPUs this model
        # runs on (validated at create/update time). Otherwise auto-pack into the
        # next free (tp*pp)-wide slot, round-robin per slot width.
        pinned = [int(x) for x in (m.get("gpu_indices") or [])]
        if pinned:
            gpu_indices = pinned
        else:
            n_slots = max(1, universe // width) if universe else 1
            si = slot_cursor.get(width, 0) % n_slots
            slot_cursor[width] = si + 1
            start = si * width
            gpu_indices = phys[start:start + width] if phys else list(range(start, start + width))
        out.append({
            "model": model,
            "served_name": model,
            "tp": tp,
            "pp": pp,
            "port": port,
            "gpu_indices": gpu_indices,
            "extra_args": (m.get("extra_args", "") or ""),
            "sleep_level": default_level,
            "task": (m.get("task") or None),
        })
        port += 1
    return {
        "total_gpus": universe,
        "sleep_level": default_level,
        # VM serving: which uv venv to run `vllm serve` from + the version the
        # worker should ensure is installed there. None → bare python3 on PATH.
        "venv_path": (getattr(app, "venv_path", None) or None),
        "vllm_version": (getattr(app, "vllm_version", None) or None),
        "models": out,
    }


def build_metrics_env(app_id: str, provider_name: str) -> dict[str, str]:
    """Per-worker observability env: tells the worker entrypoint to run
    ansible-pull on the gpu-metrics-exporter playbook so it self-installs the
    DCGM/node/vLLM exporter stack and ships metrics to VictoriaMetrics under
    an `endpoint=<app_id>` label.

    Returns an empty dict (= disabled) when the gateway pod is missing the
    METRICS_REMOTE_WRITE_URL/USERNAME/PASSWORD secret triple — keeps local
    dev installs working without forcing the secret to exist."""
    url = (os.environ.get("METRICS_REMOTE_WRITE_URL") or "").strip()
    user = (os.environ.get("METRICS_USERNAME") or "").strip()
    pw = (os.environ.get("METRICS_PASSWORD") or "").strip()
    if not (url and user and pw):
        return {}
    repo = (os.environ.get("METRICS_REPO_URL") or "https://github.com/AIES-Infra/gpu-metrics-exporter.git").strip()
    branch = (os.environ.get("METRICS_REPO_BRANCH") or "main").strip()
    return {
        "ENABLE_METRICS": "true",
        "METRICS_REPO_URL": repo,
        "METRICS_REPO_BRANCH": branch,
        "METRICS_REMOTE_WRITE_URL": url,
        "METRICS_USERNAME": user,
        "METRICS_PASSWORD": pw,
        "METRICS_DATACENTER": provider_name,
        "METRICS_ENDPOINT": app_id,
    }


async def emit_worker_event(
    rdb: "redis_async.Redis",
    machine_id: str,
    app_id: str,
    level: str,
    msg: str,
) -> None:
    """Append a lifecycle event to the worker's capped Redis ring.

    Also stamps `worker_app:{mid}` so the read endpoint can authorize a
    request even after the worker pod has been torn down (worker:{mid}
    expires when the worker stops heartbeating)."""
    if not machine_id:
        return
    try:
        entry = json.dumps({"ts": time.time(), "level": level, "msg": msg})
        key = f"worker_events:{machine_id}"
        await rdb.lpush(key, entry)
        await rdb.ltrim(key, 0, WORKER_EVENTS_CAP - 1)
        await rdb.expire(key, WORKER_EVENTS_TTL_S)
        if app_id:
            await rdb.set(f"worker_app:{machine_id}", app_id, ex=WORKER_EVENTS_TTL_S)
    except Exception:
        logger.exception("emit_worker_event failed for %s", machine_id)


async def autoscaler_loop(
    rdb: "redis_async.Redis",
    provider: "Provider",
    sm: "async_sessionmaker[AsyncSession]",
    provider_cache: dict | None = None,
) -> None:
    logger.info("autoscaler running")
    if provider_cache is None:
        provider_cache = {}
    while True:
        try:
            await asyncio.sleep(TICK_S)
            await tick(rdb, provider, sm, provider_cache)
        except asyncio.CancelledError:
            logger.info("autoscaler cancelled")
            raise
        except Exception:
            logger.exception("autoscaler tick failed")


async def tick(
    rdb: "redis_async.Redis",
    fallback_provider: "Provider",
    sm: "async_sessionmaker[AsyncSession]",
    provider_cache: dict | None = None,
) -> None:
    from .provider import resolve_app_provider

    if provider_cache is None:
        provider_cache = {}
    # Resolve each app's provider while the session is open, then close it so we
    # don't hold a DB connection across slow provider.provision() network/SSH I/O.
    # expire_on_commit=False keeps the detached App attributes readable below.
    async with sm() as session:
        apps = list((await session.execute(select(App))).scalars().all())
        resolved: list[tuple[App, "Provider | None"]] = []
        for app in apps:
            try:
                prov = await resolve_app_provider(
                    session, app, redis=rdb, fallback=fallback_provider, cache=provider_cache
                )
            except Exception as e:
                logger.warning("autoscaler: provider resolve failed for app=%s: %s", app.app_id, e)
                await _record_provision_error(rdb, app.app_id, f"provider resolution failed: {e}")
                prov = None
            resolved.append((app, prov))
    # Reconcile per-app inside a per-app try/except so one bad app (provider
    # rejecting its GPU spec, etc.) doesn't starve the others on this tick.
    for app, prov in resolved:
        if prov is None:
            continue
        try:
            await _reconcile_app(rdb, prov, app)
        except Exception:
            logger.exception("autoscaler: reconcile failed for app=%s", app.app_id)


async def _reconcile_app(rdb: "redis_async.Redis", provider: "Provider", app: App) -> None:
    app_id = app.app_id
    # Self-heal provider-level connectivity (VM reverse tunnel) before we look at
    # workers — after a gateway restart the in-process tunnel is gone, and a
    # steady always-on worker won't trigger a re-provision to rebuild it.
    try:
        await provider.ensure_connectivity()
    except Exception:
        logger.exception("autoscaler: ensure_connectivity failed for app=%s", app_id)
    autoscaler_cfg = app.autoscaler
    max_containers = int(autoscaler_cfg["max_containers"])
    tasks_per_container = int(autoscaler_cfg["tasks_per_container"])
    idle_timeout_s = int(autoscaler_cfg["idle_timeout_s"])

    queue_len = await rdb.llen(f"queue:{app_id}")
    workers = await _live_workers(rdb, app_id)
    current = len(workers)

    # idle_timeout_s == 0 also means "always-on": keep at least one worker
    # warm so the first request doesn't pay cold-start, and respawn if the
    # worker dies.
    always_on = idle_timeout_s == 0
    if queue_len == 0:
        desired = 1 if always_on else 0
    else:
        desired = math.ceil(queue_len / tasks_per_container)
        if always_on:
            desired = max(desired, 1)
    desired = min(max_containers, desired)

    # Killed / paused (POST /apps/{id}/workers/kill): keep the fleet down. Override
    # the always-on bump so we don't immediately respawn the worker the user just
    # killed. Cleared by Redeploy (restart) or editing the model list.
    if await rdb.get(f"app:{app_id}:paused"):
        desired = 0

    last_request_blob = await rdb.get(f"app:{app_id}:last_request_ts")
    last_request_ts = float(last_request_blob) if last_request_blob else 0.0
    idle_for = time.time() - last_request_ts if last_request_ts else float("inf")

    if desired > current:
        from . import metrics as _metrics
        # Cooldown: skip the provision attempt entirely if the last try failed
        # recently. Otherwise we spam the upstream provider every tick when
        # there's no inventory or our spec is wrong, which burns API quota
        # and noses-up the gateway logs.
        cooldown_until_blob = await rdb.get(f"app:{app_id}:provision_cooldown_until")
        if cooldown_until_blob:
            try:
                if time.time() < float(cooldown_until_blob):
                    return
            except (TypeError, ValueError):
                pass
        n_to_add = desired - current
        # Admin global env / secrets, merged into every worker's env (an app env
        # var of the same name overrides). Loaded once per scale-up, not per tick.
        global_env: dict[str, str] = {}
        try:
            from .db import session_factory
            from .global_env_api import load_global_env
            async with session_factory()() as _ges:
                global_env = await load_global_env(_ges)
        except Exception:
            logger.exception("autoscaler: failed to load global env for app=%s", app_id)
        for _ in range(n_to_add):
            token = secrets.token_urlsafe(24)
            env: dict[str, str] = {"REGISTRATION_TOKEN": token}
            extra = (getattr(app, "vllm_args", "") or "").strip()
            if extra:
                env["VLLM_EXTRA_ARGS"] = extra
            if bool(getattr(app, "enable_metrics", True)):
                env.update(build_metrics_env(app_id, provider.name))
            # Multi-model VM fleet: hand the worker its whole model spec.
            if getattr(app, "mode", "single") == "multi":
                env["WORKER_MODE"] = "multi"
                env["MULTI_MODEL_CONFIG"] = json.dumps(build_multi_model_config(app))
                env["SLEEP_LEVEL"] = str(int(getattr(app, "sleep_level", 1) or 1))
                env["TOTAL_GPUS"] = str(int(getattr(app, "gpu_count", 0) or 0))
            elif (getattr(app, "visible_devices", None) or "").strip():
                # Single-model VM GPU pin. Multi sets this per model from
                # gpu_indices, so only apply the global var outside multi.
                env["CUDA_VISIBLE_DEVICES"] = app.visible_devices.strip()
            # Global env/secrets + this app's env vars (app overrides global),
            # applied to every vLLM process on the worker. `secret://KEY` values
            # (e.g. a referenced HF token) resolve against the global secrets.
            from .global_env_api import resolve_env_refs
            _worker_env = resolve_env_refs(
                {**global_env, **(getattr(app, "env_vars", None) or {})}, global_env
            )
            if _worker_env:
                env["WORKER_ENV_JSON"] = json.dumps(_worker_env)
            try:
                result = await provider.provision(
                    app_id=app_id,
                    model=app.model,
                    gpu=app.gpu,
                    env=env,
                    gpu_count=int(getattr(app, "gpu_count", 1) or 1),
                    cloud_type=getattr(app, "cloud_type", None),
                    container_disk_gb=getattr(app, "container_disk_gb", None),
                    volume_gb=getattr(app, "volume_gb", None),
                )
                machine_id = result.machine_id
                _metrics.PROVISION_TOTAL.labels(provider=provider.name, ok="true").inc()
                await emit_worker_event(
                    rdb, machine_id, app_id, "info",
                    f"provisioned on {provider.name} (gpu={app.gpu}x{int(getattr(app, 'gpu_count', 1) or 1)}"
                    + (f", ${result.cost_per_hr:.4f}/hr)" if result.cost_per_hr is not None else ")"),
                )
                # Clear any stale error/cooldown after a successful provision.
                await rdb.delete(
                    f"app:{app_id}:last_provision_error",
                    f"app:{app_id}:last_provision_error_at",
                    f"app:{app_id}:provision_cooldown_until",
                )
            except Exception as e:
                _metrics.PROVISION_TOTAL.labels(provider=provider.name, ok="false").inc()
                error_msg = (str(e) or repr(e))[:1000]
                cooldown_until = time.time() + PROVISION_COOLDOWN_S
                await rdb.set(
                    f"app:{app_id}:provision_cooldown_until",
                    str(cooldown_until),
                    ex=PROVISION_COOLDOWN_S + 30,
                )
                await rdb.set(
                    f"app:{app_id}:last_provision_error",
                    error_msg,
                    ex=PROVISION_ERROR_TTL_S,
                )
                await rdb.set(
                    f"app:{app_id}:last_provision_error_at",
                    str(time.time()),
                    ex=PROVISION_ERROR_TTL_S,
                )
                logger.warning(
                    "provision failed for app=%s gpu=%sx%d: %s — cooldown %ds",
                    app_id, app.gpu,
                    int(getattr(app, "gpu_count", 1) or 1),
                    error_msg[:200], PROVISION_COOLDOWN_S,
                )
                return  # don't try the next slot in n_to_add this tick
            await rdb.set(
                f"register_token:{machine_id}",
                token,
                ex=REGISTRATION_TOKEN_TTL_S,
            )
            await rdb.sadd(f"worker_index:{app_id}", machine_id)
            await rdb.set(
                f"worker:{machine_id}",
                json.dumps({
                    "machine_id": machine_id,
                    "app_id": app_id,
                    "status": "provisioning",
                    "last_seen": time.time(),
                }),
                ex=REGISTRATION_TOKEN_TTL_S,
            )
            # Cost lives in a sidecar key because heartbeat/register overwrites
            # the worker:* blob — we want the original spawn-time quote to stick
            # for the life of the worker.
            if result.cost_per_hr is not None:
                await rdb.set(
                    f"worker_cost:{machine_id}",
                    str(result.cost_per_hr),
                    ex=REGISTRATION_TOKEN_TTL_S,
                )
            await emit_worker_event(
                rdb, machine_id, app_id, "info",
                f"scaled up: +1 worker → {current + 1}/{max_containers}",
            )
            logger.info("scaled up %s: +1 worker (%s) → %d/%d", app_id, machine_id, current + 1, max_containers)
            current += 1
    elif idle_timeout_s > 0 and desired < current and queue_len == 0 and idle_for >= idle_timeout_s:
        if workers:
            from . import metrics as _metrics
            # Don't tear down a worker that's still coming up or actively serving.
            # A multi-model cold start can outlast idle_timeout_s while the waking
            # request is already in flight (dequeued → queue_len==0 — but the model
            # is still loading), so terminating on queue_len+idle alone would kill
            # the pod mid-load and strand that request as stuck "pending". Pick an
            # idle, READY victim; a worker whose heartbeat expired (crashed) is also
            # fair game (cleanup). If none qualify, wait for the next tick.
            victim = None
            for mid in workers:
                wblob = await rdb.get(f"worker:{mid}")
                wst = None
                if wblob:
                    try:
                        wst = json.loads(wblob).get("status")
                    except (json.JSONDecodeError, TypeError):
                        wst = None
                if wst and wst != "ready":
                    continue  # provisioning / loading — let the cold start finish
                inflight = 0
                mblob = await rdb.get(f"worker:{mid}:models")
                if mblob:
                    try:
                        inflight = sum(int(m.get("inflight") or 0) for m in json.loads(mblob))
                    except (ValueError, TypeError, json.JSONDecodeError):
                        inflight = 0
                if inflight > 0:
                    continue  # actively serving in-flight requests
                victim = mid
                break
            if victim is None:
                logger.info(
                    "scale-down %s: holding — worker(s) still loading or serving (idle %.0fs)",
                    app_id, idle_for,
                )
            else:
                try:
                    await provider.terminate(victim)
                    _metrics.TERMINATE_TOTAL.labels(provider=provider.name, ok="true").inc()
                    await emit_worker_event(
                        rdb, victim, app_id, "info",
                        f"terminated (idle for {int(idle_for)}s)",
                    )
                except Exception:
                    _metrics.TERMINATE_TOTAL.labels(provider=provider.name, ok="false").inc()
                    await emit_worker_event(rdb, victim, app_id, "error", "terminate failed (provider error)")
                    raise
                await rdb.delete(f"worker:{victim}")
                await rdb.srem(f"worker_index:{app_id}", victim)
                logger.info("scaled down %s: -1 worker (%s, idle %.0fs)", app_id, victim, idle_for)


async def _record_provision_error(rdb: "redis_async.Redis", app_id: str, msg: str) -> None:
    """Surface a provisioning/resolution failure to the UI and back off, so a
    misconfigured app doesn't spam the upstream every tick."""
    error_msg = (msg or "")[:1000]
    await rdb.set(
        f"app:{app_id}:provision_cooldown_until",
        str(time.time() + PROVISION_COOLDOWN_S),
        ex=PROVISION_COOLDOWN_S + 30,
    )
    await rdb.set(f"app:{app_id}:last_provision_error", error_msg, ex=PROVISION_ERROR_TTL_S)
    await rdb.set(f"app:{app_id}:last_provision_error_at", str(time.time()), ex=PROVISION_ERROR_TTL_S)


async def _live_workers(rdb: "redis_async.Redis", app_id: str) -> list[str]:
    candidates = await rdb.smembers(f"worker_index:{app_id}")
    live: list[str] = []
    for mid in candidates:
        if await rdb.exists(f"worker:{mid}"):
            live.append(mid)
        else:
            await rdb.srem(f"worker_index:{app_id}", mid)
    return live
