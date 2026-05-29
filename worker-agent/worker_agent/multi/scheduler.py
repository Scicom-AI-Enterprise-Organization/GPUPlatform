"""GPU-slot scheduler + model router for the multi-model VM fleet.

Contention is at the GPU level: a model may be AWAKE only when every GPU in its
`gpu_indices` is free. To serve a request for an asleep model we evict (sleep)
the least-recently-used AWAKE models that hold any of its GPUs — after letting
their in-flight requests drain — then wake the target.

A single asyncio.Condition guards all state transitions; the slow work (drain
wait, /sleep, /wake_up HTTP) happens with the lock released, and waiters
re-check state on every notify. release() decrements in-flight under the same
Condition, so a drain can never deadlock against a held lock.
"""
from __future__ import annotations

import asyncio
import enum
import logging
import signal
import time

import httpx

from .config import MemberModel, MultiModelConfig
from . import launcher, vllm_ctl

logger = logging.getLogger("worker-agent.scheduler")


class ModelState(str, enum.Enum):
    LAUNCHING = "launching"
    AWAKE = "awake"
    ASLEEP = "asleep"
    WAKING = "waking"
    DRAINING = "draining"
    SLEEPING = "sleeping"
    DEAD = "dead"


class UnknownModelError(Exception):
    pass


class WakeError(Exception):
    pass


class WakeTimeoutError(Exception):
    pass


class ModelRuntime:
    def __init__(self, member: MemberModel):
        self.member = member
        self.state = ModelState.LAUNCHING
        self.proc: asyncio.subprocess.Process | None = None
        self.inflight = 0
        self.last_used = 0.0
        self.swapping = False  # a _swap_to task is converging this model → AWAKE
        self.restart_count = 0
        self.log_path: str | None = None
        self.reason: str | None = None  # human cause when state == DEAD

    @property
    def base_url(self) -> str:
        return self.member.base_url

    @property
    def gpus(self) -> set[int]:
        return set(self.member.gpu_indices)


class MultiModelScheduler:
    def __init__(self, cfg: MultiModelConfig, machine_id: str, log_dir: str = "/var/log/vllm"):
        self.cfg = cfg
        self.machine_id = machine_id
        self.log_dir = log_dir
        # Interpreter that has vLLM: the endpoint's uv venv when set, else PATH python3.
        self._python_exe = f"{cfg.venv_path}/bin/python" if cfg.venv_path else "python3"
        self._client = httpx.AsyncClient(timeout=None)
        self._cond = asyncio.Condition()
        self._runtimes: list[ModelRuntime] = [ModelRuntime(m) for m in cfg.members]
        for rt in self._runtimes:
            rt.log_path = launcher.log_path_for(rt.member, self.log_dir)
        self._by_model: dict[str, ModelRuntime] = {rt.member.served_name: rt for rt in self._runtimes}

    def _mark_dead(self, rt: ModelRuntime) -> None:
        """Set DEAD and distill a human reason from the model's vLLM log (best
        effort). Call under the Condition."""
        rt.state = ModelState.DEAD
        if rt.log_path:
            try:
                rt.reason = launcher.read_failure_reason(rt.log_path) or rt.reason
            except Exception:
                logger.exception("reading failure reason for %s", rt.member.model)
        self._cond.notify_all()

    # ---- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Launch every member sequentially (so two models sharing GPUs never
        load at once and OOM), sleep each after it's healthy, then wake a
        non-overlapping resident set up to GPU capacity."""
        # Ensure the requested vLLM version is present in the venv before any
        # model launches (no-op when venv_path/vllm_version aren't set).
        await launcher.ensure_vllm(self.cfg.venv_path, self.cfg.vllm_version)
        for rt in self._runtimes:
            await self._launch_and_sleep(rt)
        # Greedily choose residents: first model per free GPU set, in config order.
        held: set[int] = set()
        for rt in self._runtimes:
            if rt.state != ModelState.ASLEEP:
                continue
            if rt.gpus & held:
                continue  # GPUs already taken by a chosen resident
            try:
                await vllm_ctl.wake_model(self._client, rt.base_url, rt.member.sleep_level)
                async with self._cond:
                    rt.state = ModelState.AWAKE
                    rt.last_used = time.time()
                    self._cond.notify_all()
                held |= rt.gpus
                logger.info("resident: %s on gpus %s", rt.member.model, sorted(rt.gpus))
            except Exception:
                logger.exception("failed to warm %s", rt.member.model)

    async def _launch_and_sleep(self, rt: ModelRuntime) -> None:
        try:
            rt.reason = None  # clear any stale cause from a prior attempt
            rt.proc = await launcher.launch_member(rt.member, self.log_dir, self._python_exe)
            ok = await launcher.wait_health(self._client, rt.member, rt.proc)
            if not ok:
                async with self._cond:
                    self._mark_dead(rt)
                return
            await vllm_ctl.sleep_model(self._client, rt.base_url, rt.member.sleep_level)
            async with self._cond:
                rt.state = ModelState.ASLEEP
                self._cond.notify_all()
        except Exception:
            logger.exception("launch_and_sleep failed for %s", rt.member.model)
            async with self._cond:
                self._mark_dead(rt)

    # ---- routing / admission ---------------------------------------------

    def resolve(self, served_name: str) -> ModelRuntime | None:
        return self._by_model.get(served_name)

    async def acquire(self, served_name: str, deadline: float) -> ModelRuntime:
        """Ensure the model is AWAKE and reserve an in-flight slot. Triggers
        eviction if its GPUs are held by other awake models. Blocks (≤deadline)
        during a swap. Caller MUST release() when done."""
        rt = self._by_model.get(served_name)
        if rt is None:
            raise UnknownModelError(served_name)
        async with self._cond:
            while True:
                if rt.state == ModelState.AWAKE:
                    rt.inflight += 1
                    rt.last_used = time.time()
                    return rt
                if rt.state == ModelState.DEAD:
                    raise WakeError(f"model {served_name} is dead")
                # If nothing is converging this model to AWAKE yet, kick off a
                # swap. DRAINING/SLEEPING/WAKING mean a transition is in flight —
                # just wait for the next notify and re-evaluate.
                if rt.state == ModelState.ASLEEP and not rt.swapping:
                    rt.swapping = True
                    asyncio.create_task(self._swap_to(rt))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WakeTimeoutError(served_name)
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise WakeTimeoutError(served_name)

    async def release(self, rt: ModelRuntime) -> None:
        async with self._cond:
            rt.inflight = max(0, rt.inflight - 1)
            rt.last_used = time.time()
            self._cond.notify_all()

    # ---- the swap (LRU-drain eviction → wake) -----------------------------

    async def _swap_to(self, rt: ModelRuntime) -> None:
        try:
            async with self._cond:
                rt.state = ModelState.WAKING
                needed = rt.gpus
                victims = sorted(
                    [r for r in self._runtimes
                     if r is not rt and r.state == ModelState.AWAKE and (r.gpus & needed)],
                    key=lambda r: r.last_used,  # LRU first
                )
                for v in victims:
                    v.state = ModelState.DRAINING
                self._cond.notify_all()

            for v in victims:
                await self._drain(v)
                async with self._cond:
                    v.state = ModelState.SLEEPING
                    self._cond.notify_all()
                await vllm_ctl.sleep_model(self._client, v.base_url, v.member.sleep_level)
                async with self._cond:
                    v.state = ModelState.ASLEEP
                    self._cond.notify_all()

            await vllm_ctl.wake_model(self._client, rt.base_url, rt.member.sleep_level)
            async with self._cond:
                rt.state = ModelState.AWAKE
                rt.swapping = False
                self._cond.notify_all()
        except Exception:
            logger.exception("swap_to %s failed", rt.member.model)
            async with self._cond:
                # Leave it asleep so a later request can retry; never strand it.
                if rt.state != ModelState.AWAKE:
                    rt.state = ModelState.ASLEEP
                rt.swapping = False
                self._cond.notify_all()

    async def _drain(self, v: ModelRuntime) -> None:
        """Wait until a DRAINING model has no in-flight requests. Bounded so a
        stuck request can't block a swap forever (~120s)."""
        deadline = time.monotonic() + 120.0
        while True:
            async with self._cond:
                if v.inflight <= 0:
                    return
            if time.monotonic() >= deadline:
                logger.warning("drain timeout for %s (inflight=%d) — sleeping anyway", v.member.model, v.inflight)
                return
            await asyncio.sleep(0.05)

    # ---- monitoring + reporting + shutdown --------------------------------

    async def monitor_loop(self, drain_event: asyncio.Event) -> None:
        """Relaunch crashed vLLM processes (bounded retries)."""
        while not drain_event.is_set():
            for rt in self._runtimes:
                proc = rt.proc
                if proc is not None and proc.returncode is not None and rt.state != ModelState.DEAD:
                    logger.warning("vllm for %s died (rc=%s); relaunching", rt.member.model, proc.returncode)
                    async with self._cond:
                        rt.state = ModelState.LAUNCHING
                        rt.inflight = 0
                        rt.swapping = False
                        rt.reason = None
                        self._cond.notify_all()
                    if rt.restart_count < 5:
                        rt.restart_count += 1
                        await self._launch_and_sleep(rt)
                    else:
                        async with self._cond:
                            self._mark_dead(rt)
            try:
                await asyncio.wait_for(drain_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

    def states_snapshot(self) -> list[dict]:
        return [
            {
                "model": rt.member.served_name,
                "state": rt.state.value,
                "inflight": rt.inflight,
                "gpus": list(rt.member.gpu_indices),
                "tp": rt.member.tp,
                "last_used_ts": rt.last_used or None,
                "reason": rt.reason,
                "port": rt.member.port,
            }
            for rt in self._runtimes
        ]

    def all_ready(self) -> bool:
        """True once startup has settled (no model still LAUNCHING)."""
        return all(rt.state != ModelState.LAUNCHING for rt in self._runtimes)

    async def shutdown(self) -> None:
        for rt in self._runtimes:
            proc = rt.proc
            if proc is None or proc.returncode is not None:
                continue
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        await asyncio.sleep(3.0)
        for rt in self._runtimes:
            proc = rt.proc
            if proc is None or proc.returncode is not None:
                continue
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            await self._client.aclose()
        except Exception:
            pass
