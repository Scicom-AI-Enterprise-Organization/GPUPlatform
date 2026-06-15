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
import os
import signal
import time
from contextlib import asynccontextmanager

import httpx

from .config import MemberModel, MultiModelConfig
from . import launcher, vllm_ctl, cleanup

logger = logging.getLogger("worker-agent.scheduler")

# Consecutive failed /health probes on a settled (awake/asleep) member before we
# treat its engine as dead and force a relaunch. The monitor ticks every ~5s, so
# this is ~15s of unresponsiveness — long enough to ride out a transient blip,
# short enough that a wedged engine doesn't sit "awake" forever.
HEALTH_FAIL_LIMIT = 3


class ModelState(str, enum.Enum):
    QUEUED = "queued"        # waiting for an earlier wave to load + sleep (shares GPUs)
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
        self.pgid: int | None = None  # engine's process group (leader + tp workers)
        self.inflight = 0
        self.last_used = 0.0
        self.swapping = False  # a _swap_to task is converging this model → AWAKE
        self.restart_count = 0
        self.log_path: str | None = None
        self.reason: str | None = None  # human cause when state == DEAD
        self.load_idx = 0  # position in the startup load order (for "in N queue")
        self.health_fail = 0  # consecutive failed /health probes (monitor_loop)

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
        # Per-GPU locks: a load/wake/evict holds the locks for ITS GPUs only, so
        # models on disjoint GPUs (e.g. [0,1], [2,3], [4,5]) run concurrently
        # while overlapping ones serialize on the shared GPU. Acquired in sorted
        # order to avoid deadlock. This is what lets non-overlapping members load
        # at the same time instead of one-by-one.
        self._gpu_locks: dict[int, asyncio.Lock] = {}
        for rt in self._runtimes:
            for g in rt.member.gpu_indices:
                self._gpu_locks.setdefault(g, asyncio.Lock())
        for rt in self._runtimes:
            rt.log_path = launcher.log_path_for(rt.member, self.log_dir)
        self._by_model: dict[str, ModelRuntime] = {rt.member.served_name: rt for rt in self._runtimes}
        # Where we persist the engine process-groups we own, so preflight + the
        # provider's terminate can kill ONLY our processes (never a box-wide
        # pkill). Set by the provider; None → no persistence (cleanup no-ops).
        self._pids_path: str | None = os.environ.get("WORKER_ENGINE_PIDS_PATH")

    @asynccontextmanager
    async def _hold_gpus(self, gpus):
        """Hold the locks for exactly `gpus` (sorted, to avoid deadlock). Two
        operations on disjoint GPU sets proceed concurrently; overlapping ones
        block on the shared GPU's lock."""
        locks = [self._gpu_locks[g] for g in sorted(set(gpus))]
        acquired = []
        try:
            for lk in locks:
                await lk.acquire()
                acquired.append(lk)
            yield
        finally:
            for lk in reversed(acquired):
                lk.release()

    async def _kill_stale_vllm(self) -> None:
        """Reap engines orphaned by a PRIOR run of this worker that was SIGKILLed
        before it could clean up its children.

        We kill ONLY the process groups that prior run recorded in our pids file
        (`WORKER_ENGINE_PIDS_PATH`) — never a box-wide `pkill -f 'VLLM::'`. The VM
        may be shared: another endpoint, or a user outside the platform, can be
        running their own vLLM, and we must not touch it. The cleanup is
        reuse-guarded by start-time (see cleanup.cleanup_records), so a recycled
        pid/pgid can't be mistaken for ours. Safe at start(): we launch nothing
        until after this runs."""
        if not self._pids_path:
            return
        try:
            killed = await asyncio.to_thread(cleanup.cleanup_file, self._pids_path, logger.info)
            if killed:
                logger.info("preflight: reaped %d orphaned engine process(es) from a prior run", len(killed))
        except Exception:
            logger.exception("preflight engine cleanup failed")

    async def kill_model(self, served_name: str) -> None:
        """Operator action: stop a model's engine (and all its tp workers) and
        leave it DEAD. The monitor won't relaunch it (proc is cleared)."""
        rt = self._by_model.get(served_name)
        if rt is None:
            raise UnknownModelError(served_name)
        logger.info("operator kill: %s", served_name)
        async with self._cond:
            old = rt.proc
            rt.proc = None
            rt.state = ModelState.DEAD
            rt.reason = "stopped by operator (kill)"
            rt.inflight = 0
            rt.swapping = False
            self._cond.notify_all()
        await launcher.terminate(old)

    async def restart_model(self, served_name: str) -> None:
        """Operator action: kill + relaunch a model (then sleep it, ready to be
        woken on demand). Resets the crash-retry counter."""
        rt = self._by_model.get(served_name)
        if rt is None:
            raise UnknownModelError(served_name)
        logger.info("operator restart: %s", served_name)
        async with self._cond:
            old = rt.proc
            rt.proc = None
            rt.state = ModelState.LAUNCHING
            rt.reason = None
            rt.inflight = 0
            rt.swapping = False
            rt.restart_count = 0
            self._cond.notify_all()
        await launcher.terminate(old)
        await self._launch_and_sleep(rt)

    async def operator_sleep(self, served_name: str) -> None:
        """Operator action: put an AWAKE model to sleep now (drain in-flight,
        then /sleep at its level), freeing its GPUs. No-op if not awake."""
        rt = self._by_model.get(served_name)
        if rt is None:
            raise UnknownModelError(served_name)
        async with self._hold_gpus(rt.gpus):
            async with self._cond:
                if rt.state != ModelState.AWAKE:
                    return
                rt.state = ModelState.DRAINING
                self._cond.notify_all()
            await self._drain(rt)
            async with self._cond:
                rt.state = ModelState.SLEEPING
                self._cond.notify_all()
            await vllm_ctl.sleep_model(self._client, rt.base_url, rt.member.sleep_level)
            async with self._cond:
                rt.state = ModelState.ASLEEP
                rt.last_used = time.time()
                self._cond.notify_all()
        logger.info("operator sleep: %s", served_name)

    async def operator_sleep_all(self) -> None:
        """Operator action: sleep every awake model (free all GPUs)."""
        names = [rt.member.served_name for rt in self._runtimes if rt.state == ModelState.AWAKE]
        logger.info("operator sleep_all: %s", names)
        for name in names:
            try:
                await self.operator_sleep(name)
            except Exception:
                logger.exception("sleep_all: %s failed", name)

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
        # Kill any vLLM left bound to our ports by a prior crashed/killed worker —
        # we haven't launched ours yet, so anything on these ports is an orphan
        # holding GPU memory that would block our launches.
        await self._kill_stale_vllm()
        # Bootstrap the endpoint's vLLM venv if it doesn't exist yet (install uv,
        # `uv venv` the path, install vLLM) so an endpoint can name a venv_path the
        # VM hasn't created — instead of crashing on a missing {venv_path}/bin/python.
        await launcher.ensure_venv(
            self.cfg.venv_path, self.cfg.vllm_version, self.cfg.vllm_install_args)
        # Ensure the requested vLLM version is present in the venv before any
        # model launches (no-op when venv_path/vllm_version aren't set, or when a
        # custom vllm_install_args spec owns the install).
        if not self.cfg.vllm_install_args:
            await launcher.ensure_vllm(self.cfg.venv_path, self.cfg.vllm_version)
        # vLLM's flashinfer backend JIT-builds kernels at runtime via `ninja` on
        # Blackwell — make sure ninja/cmake are in the venv (launch puts venv/bin on
        # PATH so the JIT finds them).
        await launcher.ensure_build_tools(self.cfg.venv_path)
        # Whisper/ASR members need vLLM's audio-decode deps (librosa, soundfile) or
        # every clip is rejected as "Invalid or unsupported audio file".
        await launcher.ensure_audio_deps(self.cfg.venv_path, self.cfg.members)
        # Optional operator setup script (e.g. building DeepGEMM) — runs once now,
        # with the venv ready and on PATH, before any model launches.
        await launcher.run_pre_script(self.cfg.pre_script, self.cfg.venv_path)
        # Load in WAVES: each wave is a set of mutually NON-overlapping members
        # (disjoint gpu_indices) loaded concurrently; waves run in sequence. On
        # 6 GPUs this loads qwen[0,1] + 35B[2,3] + gemma[4,5] all at once, then
        # Mistral[0,1,2,3] in the next wave — instead of one model at a time.
        # Precompute the waves up front so models waiting for a later wave can show
        # their place in line ("in N queue") instead of a misleading "launching".
        waves: list[list[ModelRuntime]] = []
        remaining = list(self._runtimes)
        while remaining:
            wave: list[ModelRuntime] = []
            rest: list[ModelRuntime] = []
            used: set[int] = set()
            for rt in remaining:
                if rt.gpus & used:
                    rest.append(rt)
                else:
                    wave.append(rt)
                    used |= rt.gpus
            waves.append(wave)
            remaining = rest
        # Stable load order; everything past wave 0 starts QUEUED (waiting its turn).
        idx = 0
        for w, wv in enumerate(waves):
            for rt in wv:
                rt.load_idx = idx
                idx += 1
                if w > 0:
                    rt.state = ModelState.QUEUED
        # Load wave by wave; flip each wave to LAUNCHING as its turn comes.
        for w, wv in enumerate(waves):
            for rt in wv:
                rt.state = ModelState.LAUNCHING
            logger.info("loading wave %d/%d (%d concurrent): %s", w + 1, len(waves), len(wv),
                        [rt.member.served_name for rt in wv])
            await asyncio.gather(
                *(self._launch_and_sleep(rt) for rt in wv),
                return_exceptions=True,
            )
        # Greedily choose residents: first model per free GPU set, in config order.
        held: set[int] = set()
        for rt in self._runtimes:
            if rt.state != ModelState.ASLEEP:
                continue
            if rt.gpus & held:
                continue  # GPUs already taken by a chosen resident
            try:
                await self._wake(rt)
                held |= rt.gpus
                logger.info("resident: %s on gpus %s", rt.member.model, sorted(rt.gpus))
            except Exception:
                logger.exception("failed to warm %s", rt.member.model)

    @staticmethod
    def _safe_pgid(proc: asyncio.subprocess.Process | None) -> int | None:
        """Process-group id of a launched engine (== its pid, since the launcher
        uses start_new_session). The whole tp-worker family shares it."""
        if proc is None or proc.pid is None:
            return None
        try:
            return os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            return None

    def _dump_pids(self) -> None:
        """Persist the engine process-groups we currently own (leader + tp-worker
        children, each with its start-time) so preflight + the provider's
        terminate can reap ONLY our processes — never anything else on the VM."""
        if not self._pids_path:
            return
        live = {
            rt.pgid: rt
            for rt in self._runtimes
            if rt.proc is not None and rt.proc.returncode is None and rt.pgid
        }
        groups = cleanup.snapshot_groups(live.keys())
        records = [
            {"model": live[pgid].member.served_name, "pgid": pgid, "pids": pids}
            for pgid, pids in groups.items()
            if pids
        ]
        try:
            cleanup.dump_records(self._pids_path, records)
        except OSError:
            logger.warning("could not write engine pids file %s", self._pids_path)

    async def _launch_and_sleep(self, rt: ModelRuntime) -> None:
        try:
            rt.reason = None  # clear any stale cause from a prior attempt
            async with self._hold_gpus(rt.gpus):
                # Free rt's GPUs before loading: sleep any AWAKE model that shares
                # them, or the load OOMs against a resident (e.g. relaunching a
                # 2,3 model while Mistral holds 0,1,2,3). At initial startup this
                # is a no-op since nothing is awake yet.
                await self._evict_overlapping(rt)
                rt.proc = await launcher.launch_member(rt.member, self.log_dir, self._python_exe)
                # Record the engine's process group right away (before health) so a
                # worker SIGKILL mid-load still leaves a trackable orphan, not a
                # leak we'd have to pkill box-wide.
                rt.pgid = self._safe_pgid(rt.proc)
                self._dump_pids()
                ok = await launcher.wait_health(self._client, rt.member, rt.proc, log_path=rt.log_path)
                if not ok:
                    # Kill the failed/hung engine so it can't leak and keep holding
                    # GPU/CPU after we give up on it.
                    await launcher.terminate(rt.proc)
                    async with self._cond:
                        self._mark_dead(rt)
                    rt.pgid = None
                    self._dump_pids()
                    return
                await vllm_ctl.sleep_model(self._client, rt.base_url, rt.member.sleep_level)
                async with self._cond:
                    rt.state = ModelState.ASLEEP
                    self._cond.notify_all()
                # Re-dump now that the tp workers have spawned, so the recorded
                # group includes every child.
                self._dump_pids()
        except Exception:
            logger.exception("launch_and_sleep failed for %s", rt.member.model)
            await launcher.terminate(rt.proc)
            async with self._cond:
                self._mark_dead(rt)
            rt.pgid = None
            self._dump_pids()

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

    async def _evict_overlapping(self, rt: ModelRuntime) -> None:
        """Drain + sleep every AWAKE model whose GPUs overlap rt's, so rt's GPUs
        are free to load into or wake into. Caller MUST hold rt's GPU locks."""
        async with self._cond:
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

    async def _wake(self, rt: ModelRuntime) -> None:
        """Free rt's GPUs (sleeping overlapping awake models) then wake rt. Holds
        rt's GPU locks so it can't race a concurrent load/swap on the same GPUs;
        non-overlapping wakes run concurrently. Used by both the on-demand swap
        and the startup resident warm-up."""
        async with self._hold_gpus(rt.gpus):
            async with self._cond:
                if rt.state == ModelState.AWAKE:
                    rt.swapping = False
                    self._cond.notify_all()
                    return
                rt.state = ModelState.WAKING
                self._cond.notify_all()
            await self._evict_overlapping(rt)
            await vllm_ctl.wake_model(self._client, rt.base_url, rt.member.sleep_level)
            async with self._cond:
                rt.state = ModelState.AWAKE
                rt.last_used = time.time()
                rt.swapping = False
                self._cond.notify_all()

    async def _swap_to(self, rt: ModelRuntime) -> None:
        try:
            await self._wake(rt)
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
        """Relaunch dead vLLM engines (bounded retries). Catches BOTH failure modes:

        (a) the api_server PROCESS exits → `proc.returncode` is set; and
        (b) the ENGINE dies while the api_server process lingers — e.g. a CUDA
            OOM during `/wake_up` raises EngineDeadError and segfaults the
            EngineCore + tp workers, but the parent keeps its socket open, so
            `proc.returncode` stays None. `_wake` already flipped the member to
            AWAKE (the /wake_up POST returned 200 before the async OOM), so a
            returncode-only check leaves it stuck "awake" forever and the next
            wake OOMs against the corpse. We actively probe /health on settled
            (awake/asleep) members — it reflects engine liveness, not just the
            socket — and force a relaunch after HEALTH_FAIL_LIMIT misses.
        """
        while not drain_event.is_set():
            for rt in self._runtimes:
                proc = rt.proc
                if proc is None or rt.state == ModelState.DEAD:
                    continue
                died = proc.returncode is not None
                cause = f"rc={proc.returncode}"
                # Engine-liveness probe — only on settled states (a wake/sleep/
                # launch in flight is allowed to be briefly unresponsive).
                if not died and rt.state in (ModelState.AWAKE, ModelState.ASLEEP):
                    if await vllm_ctl.is_healthy(self._client, rt.base_url):
                        rt.health_fail = 0
                    else:
                        rt.health_fail += 1
                        if rt.health_fail >= HEALTH_FAIL_LIMIT:
                            died = True
                            cause = f"engine unresponsive ({rt.health_fail}× /health)"
                if not died:
                    continue
                logger.warning("vllm for %s died (%s); relaunching", rt.member.model, cause)
                async with self._cond:
                    rt.state = ModelState.LAUNCHING
                    rt.inflight = 0
                    rt.swapping = False
                    rt.reason = None
                    rt.health_fail = 0
                    self._cond.notify_all()
                # Engine wedged (process alive but unresponsive)? Group-kill it so
                # it stops holding GPU memory before we reload — else the relaunch
                # OOMs against the corpse. Harmless no-op once the process exited.
                if proc.returncode is None:
                    await launcher.terminate(proc)
                    rt.proc = None
                    rt.pgid = None
                if rt.restart_count < 5:
                    rt.restart_count += 1
                    await self._launch_and_sleep(rt)
                else:
                    async with self._cond:
                        self._mark_dead(rt)
            # Refresh the persisted engine groups: picks up tp workers that
            # spawned after launch and drops any engine that has since died.
            self._dump_pids()
            try:
                await asyncio.wait_for(drain_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

    def states_snapshot(self) -> list[dict]:
        # Models still loading/queued, by load order — used to count how many are
        # ahead of a QUEUED model so the UI can show "in N queue".
        unsettled = [
            rt.load_idx for rt in self._runtimes
            if rt.state in (ModelState.QUEUED, ModelState.LAUNCHING)
        ]
        return [
            {
                "model": rt.member.served_name,
                "state": rt.state.value,
                "queue_ahead": (
                    sum(1 for x in unsettled if x < rt.load_idx)
                    if rt.state == ModelState.QUEUED else 0
                ),
                "inflight": rt.inflight,
                "gpus": list(rt.member.gpu_indices),
                "tp": rt.member.tp,
                "pp": rt.member.pp,
                "last_used_ts": rt.last_used or None,
                "reason": rt.reason,
                "port": rt.member.port,
            }
            for rt in self._runtimes
        ]

    def all_ready(self) -> bool:
        """True once startup has settled (nothing still queued or launching)."""
        return all(rt.state not in (ModelState.QUEUED, ModelState.LAUNCHING) for rt in self._runtimes)

    def live_members(self) -> list[tuple[str, str]]:
        """(served_name, base_url) for members whose engine is running — for the
        metrics shipper to scrape each one's local vLLM /metrics."""
        return [
            (rt.member.served_name, rt.base_url)
            for rt in self._runtimes
            if rt.proc is not None and rt.proc.returncode is None
        ]

    async def shutdown(self) -> None:
        # Group-kill each engine so its tp worker children (VLLM::Worker_TP) and
        # EngineCore die too — signalling only the api_server (rt.proc) orphans
        # them, leaving GPU-hogging processes behind after a drain/delete.
        await asyncio.gather(
            *(launcher.terminate(rt.proc) for rt in self._runtimes if rt.proc is not None),
            return_exceptions=True,
        )
        # Engines are down — drop the pids file so the next start has nothing
        # stale to reap (a crash leaves it in place for terminate/preflight).
        if self._pids_path:
            try:
                os.remove(self._pids_path)
            except OSError:
                pass
        try:
            await self._client.aclose()
        except Exception:
            pass
