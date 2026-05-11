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
    ) -> str:
        """Spawn a worker for `app_id`. Returns a machine_id.

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
    ) -> str:
        machine_id = f"m-fake-{uuid.uuid4().hex[:8]}"
        task = asyncio.create_task(
            self._spawn(machine_id, app_id, model, env),
            name=f"fake-worker-{machine_id}",
        )
        self._tasks[machine_id] = task
        logger.info("fake-provision: app=%s gpu=%sx%d → %s", app_id, gpu, gpu_count, machine_id)
        return machine_id

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
