"""Compute provider abstraction.

Phase 0/1: a `FakeProvider` spawns workers as in-process asyncio tasks (good
for tests + local dev with no GPUs).

Phase 2: a `PrimeIntellectProvider` calls the PI HTTP API to launch real GPU
pods. Same three-method interface, no other gateway code changes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("gateway.provider")


@dataclass
class GpuAvailability:
    """Result of a provider availability check.

    `available` is tri-state: True/False/None. None means we couldn't reach
    the upstream (auth failure, 5xx, timeout) and the UI should fall back to
    "unknown — try anyway" rather than blocking the user.
    """
    gpu: str
    count: int
    available: Optional[bool]
    cheapest_price_hr: Optional[float] = None
    regions: list[str] = field(default_factory=list)
    reason: Optional[str] = None
    checked_at: float = field(default_factory=lambda: time.time())


@dataclass
class ProvisionResult:
    """What a provider returns when it spawns a worker. `cost_per_hr` is the
    hourly rate quoted by the provider at spawn time (USD); None for providers
    that don't quote hourly prices (PI's per-token model, FakeProvider)."""
    machine_id: str
    cost_per_hr: Optional[float] = None


class Provider(ABC):
    name: str = "abstract"

    @abstractmethod
    async def provision(
        self,
        app_id: str,
        model: str,
        gpu: str,
        env: dict[str, str],
        gpu_count: int = 1,
        cloud_type: Optional[str] = None,
        container_disk_gb: Optional[int] = None,
        volume_gb: Optional[int] = None,
    ) -> ProvisionResult:
        """Spawn a worker for `app_id`. Returns a ProvisionResult with the
        machine_id plus the hourly cost the provider quoted at spawn time
        (None for providers without hourly pricing).

        The worker registers itself back to the gateway asynchronously — this
        method does NOT wait for the worker to be ready.
        """
        ...

    @abstractmethod
    async def terminate(self, machine_id: str) -> None:
        """Tear down the worker. Called when scaling down."""
        ...

    @abstractmethod
    async def list_machines(self) -> list[str]:
        """Authoritative source for what's actually running. Used by the
        reconciler to GC orphans."""
        ...

    async def list_machines_for_app(self, app_id: str) -> list[str]:
        """Machines (registered or orphaned) belonging to a specific app.
        Default falls back to list_machines; providers that can filter
        cheaply (e.g. by pod name prefix) should override."""
        return await self.list_machines()

    def supports_log_tail(self) -> bool:
        """True if the gateway can SSH-tail this provider's worker log files on
        demand (so the worker needn't push logs). VM providers override → True;
        push-only providers (RunPod pods) stay False and keep the log shipper."""
        return False

    async def forget_machine(self, machine_id: str) -> None:
        """Drop a machine from this provider's bookkeeping WITHOUT touching the box
        (the process is already dead). The reconciler uses this to reap a stale
        orphan — a machine left in the provider's listing by a crashed provision
        that never registered. Default no-op; providers with a machine set override."""
        return None

    async def purge_app(self, app_id: str) -> int:
        """Hard cleanup of ALL of an app's worker remnants — including ones the
        reconciler/redis no longer tracks (e.g. crash-loop churn that left stale
        pidfiles + orphan processes on a VM). Kills processes, removes on-disk
        state, and clears redis. Returns the number of remnants purged. Default
        no-op (RunPod pods are reaped by terminate); VM providers override."""
        return 0

    async def check_availability(
        self,
        gpu: str,
        count: int,
        cloud_type: Optional[str] = None,
    ) -> GpuAvailability:
        """Best-effort live check of whether `count` of `gpu` can be
        provisioned right now. `cloud_type` is provider-specific (e.g.
        RunPod COMMUNITY/SECURE); None = provider default."""
        return GpuAvailability(gpu=gpu, count=count, available=True)

    async def shutdown(self) -> None:
        """Kill everything. Called on gateway shutdown."""

    async def ensure_connectivity(self) -> None:
        """Hook called each autoscaler tick before reconciling. Default no-op;
        the VM provider uses it to (re)establish its reverse SSH tunnel so the
        worker can reach the gateway + Redis after a gateway restart."""


class FakeProvider(Provider):
    """In-process worker spawner, for tests and offline dev.

    Each `provision()` runs `worker_agent.main.main_async()` as an asyncio
    task in this process. `terminate()` cancels it.

    NOT for production. The real provider speaks HTTP to a cloud API.
    """

    name = "fake"

    def __init__(self, gateway_url: Optional[str] = None) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._gateway_url = gateway_url or os.environ.get(
            "GATEWAY_URL_FOR_PROVIDER", "http://127.0.0.1:8080"
        )

    async def provision(
        self,
        app_id: str,
        model: str,
        gpu: str,
        env: dict[str, str],
        gpu_count: int = 1,
        cloud_type: Optional[str] = None,  # noqa: ARG002 — fake doesn't care
        container_disk_gb: Optional[int] = None,  # noqa: ARG002
        volume_gb: Optional[int] = None,  # noqa: ARG002
    ) -> ProvisionResult:
        machine_id = f"m-fake-{uuid.uuid4().hex[:8]}"
        task = asyncio.create_task(
            self._spawn(machine_id, app_id, model, env),
            name=f"fake-worker-{machine_id}",
        )
        self._tasks[machine_id] = task
        logger.info("fake-provision: app=%s gpu=%sx%d → %s", app_id, gpu, gpu_count, machine_id)
        return ProvisionResult(machine_id=machine_id, cost_per_hr=None)

    async def _spawn(self, machine_id: str, app_id: str, model: str, env: dict[str, str]) -> None:
        # Set env keys the worker reads and call its main loop directly.
        # We can't share os.environ across multiple tasks safely, so the worker
        # reads from this dict via a small adapter.
        from worker_agent import main as wmain

        full_env = {
            "APP_ID": app_id,
            "MACHINE_ID": machine_id,
            "REGISTRATION_TOKEN": "fake-token",
            "GATEWAY_URL": self._gateway_url,
            "WORKER_MODE": "fake",
            "MODEL_ID": model,
            **env,
        }
        # Best-effort env injection. Tests using FakeProvider should use a
        # single-worker scenario; multi-worker concurrent FakeProviders would
        # race on os.environ.
        for k, v in full_env.items():
            os.environ[k] = v
        try:
            await wmain.main_async()
        except asyncio.CancelledError:
            logger.info("fake-worker %s cancelled", machine_id)
            raise
        except Exception:
            logger.exception("fake-worker %s crashed", machine_id)

    async def terminate(self, machine_id: str) -> None:
        task = self._tasks.pop(machine_id, None)
        if task is None:
            logger.warning("fake-terminate: unknown machine %s", machine_id)
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, BaseException):
            pass
        logger.info("fake-terminate: %s torn down", machine_id)

    async def list_machines(self) -> list[str]:
        return [mid for mid, t in self._tasks.items() if not t.done()]

    async def shutdown(self) -> None:
        for mid in list(self._tasks):
            await self.terminate(mid)


def build_provider(name: str) -> Provider:
    if name == "fake":
        return FakeProvider()
    if name == "primeintellect":
        from .pi_provider import PrimeIntellectProvider
        return PrimeIntellectProvider()
    if name == "runpod":
        from .runpod_provider import RunPodProvider
        return RunPodProvider()
    raise ValueError(f"unknown provider: {name}")


async def resolve_app_provider(session, app, *, redis, fallback: Optional[Provider], cache: dict) -> Provider:
    """Return the Provider an app should use.

    `provider_id is None` → the gateway-wide env-built singleton (`fallback`),
    preserving every existing env-driven deployment. Otherwise build a Provider
    bound to that provider-row's credentials and cache it by id so we don't
    re-decrypt / re-instantiate every autoscaler tick.

    `redis` is needed by VMProvider (it tracks its machines there). Raises
    RuntimeError if the row is missing or its kind is unsupported.
    """
    pid = getattr(app, "provider_id", None)
    if not pid:
        if fallback is None:
            raise RuntimeError("no provider_id and no global provider configured")
        return fallback
    if pid in cache:
        return cache[pid]

    from .db import Provider as ProviderRow
    from . import crypto

    row = await session.get(ProviderRow, pid)
    if row is None:
        raise RuntimeError(f"provider {pid} not found")

    if row.kind == "vm":
        from .vm_serverless_provider import VMProvider
        cfg = row.config or {}
        enc = cfg.get("private_key_enc")
        if not enc:
            raise RuntimeError(f"vm provider {pid} has no stored private key")
        inst: Provider = VMProvider(
            provider_id=pid,
            host=cfg.get("host", ""),
            port=int(cfg.get("port") or 22),
            user=cfg.get("user", "root"),
            private_key_pem=crypto.decrypt(enc),
            gpu_count=int(cfg.get("gpu_count") or 0),
            rdb=redis,
            # Local dev / single-gateway: forward gateway+redis to the VM over
            # SSH so a remote VM can phone home to a non-public gateway.
            reverse_tunnel=os.environ.get("VM_REVERSE_TUNNEL", "").strip() in ("1", "true", "yes"),
        )
    elif row.kind == "runpod":
        from .provider_resolve import resolve_cloud_creds
        from .runpod_provider import RunPodProvider
        creds = await resolve_cloud_creds(session, pid, "runpod")
        inst = RunPodProvider(api_key=creds.api_key)
    elif row.kind == "pi":
        from .provider_resolve import resolve_cloud_creds
        from .pi_provider import PrimeIntellectProvider
        creds = await resolve_cloud_creds(session, pid, "pi")
        inst = PrimeIntellectProvider(api_key=creds.api_key)
    else:
        raise RuntimeError(f"unsupported provider kind {row.kind!r} for serverless")

    logger.info("resolved app=%s → provider %s (kind=%s)", getattr(app, "app_id", "?"), pid, row.kind)
    cache[pid] = inst
    return inst
