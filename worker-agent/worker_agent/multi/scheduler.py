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
# Keep this many timestamped per-launch log files per member on disk (a high
# safety cap so many relaunches can't fill the VM); oldest beyond it are pruned.
LOG_RETAIN = int(os.environ.get("VLLM_LOG_RETAIN", "50") or "50")
# Crash auto-recovery: relaunch a dead engine up to MAX_RELAUNCH times, with
# EXPONENTIAL BACKOFF between attempts (base * 2^n, capped) — so we don't hammer
# a model that's crash-looping (e.g. a wake-OOM) every tick. After the budget is
# spent the member is left DEAD with its distilled reason. A model that stays
# healthy for HEALTHY_RESET_TICKS probes gets its budget reset (fresh incidents).
MAX_RELAUNCH = int(os.environ.get("VLLM_MAX_RELAUNCH", "6") or "6")
RELAUNCH_BACKOFF_BASE_S = float(os.environ.get("VLLM_RELAUNCH_BACKOFF_BASE_S", "15") or "15")
RELAUNCH_BACKOFF_CAP_S = float(os.environ.get("VLLM_RELAUNCH_BACKOFF_CAP_S", "600") or "600")
HEALTHY_RESET_TICKS = int(os.environ.get("VLLM_HEALTHY_RESET_TICKS", "6") or "6")


async def _query_gpu_free_mib() -> "dict[int, tuple[float, float]]":
    """{gpu_index: (free_mib, total_mib)} via nvidia-smi. Empty dict on ANY failure
    (binary absent / parse error / non-NVIDIA box) so callers can fail-open. Async
    so it never blocks the scheduler's event loop."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
    except Exception:  # noqa: BLE001 — nvidia-smi missing/hung: caller fails open
        return {}
    res: "dict[int, tuple[float, float]]" = {}
    for line in out.decode("utf-8", "replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            idx, free, total = int(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        res[idx] = (free, total)
    return res


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
        self.healthy_streak = 0  # consecutive healthy probes; resets the crash budget
        self.retry_pending = False  # a crash backoff is in flight (monitor relaunches at next_retry_ts)
        self.next_retry_ts = 0.0  # monotonic time the next backoff relaunch is due

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
        # Per-fleet auto-retry knobs. Fall back to the module env-var defaults when
        # the config omits them, so fleets created before this feature are unchanged.
        self._retry_max = cfg.retry_max if cfg.retry_max is not None else MAX_RELAUNCH
        self._retry_forever = bool(getattr(cfg, "retry_forever", False))
        self._backoff_base = (cfg.retry_backoff_base_s
                              if cfg.retry_backoff_base_s is not None else RELAUNCH_BACKOFF_BASE_S)
        self._backoff_cap = (cfg.retry_backoff_cap_s
                             if cfg.retry_backoff_cap_s is not None else RELAUNCH_BACKOFF_CAP_S)
        self._require_free_gpu = bool(getattr(cfg, "retry_require_free_gpu", False))
        self._gpu_free_pct = (cfg.retry_gpu_free_pct
                              if cfg.retry_gpu_free_pct is not None else 80.0)
        self._health_fail_limit = (cfg.health_fail_limit
                                   if cfg.health_fail_limit is not None else HEALTH_FAIL_LIMIT)
        logger.info(
            "auto-retry: max=%s backoff=%.0f→%.0fs health_fail_limit=%d require_free_gpu=%s%s",
            "∞" if self._retry_forever else self._retry_max,
            self._backoff_base, self._backoff_cap, self._health_fail_limit,
            self._require_free_gpu,
            f" (≥{self._gpu_free_pct:.0f}% free)" if self._require_free_gpu else "",
        )

    async def _gpus_free_enough(self, gpus: "set[int]") -> "tuple[bool, str]":
        """True when every GPU in `gpus` has ≥ self._gpu_free_pct of its VRAM free.
        Used to gate a crash relaunch (see monitor_loop) so we don't OOM-loop into a
        card a foreign job is hogging. FAIL-OPEN: if nvidia-smi is absent/unreadable
        (e.g. an Ascend box), returns True so the relaunch proceeds rather than
        blocking forever. Returns (ok, why) — `why` names the busiest GPU when not."""
        info = await _query_gpu_free_mib()
        if not info:
            return True, ""
        worst_pct: float | None = None
        worst_g: int | None = None
        for g in sorted(gpus):
            fm = info.get(g)
            if not fm:
                continue  # index not reported → don't block on it
            free, total = fm
            pct = (free / total * 100.0) if total > 0 else 100.0
            if worst_pct is None or pct < worst_pct:
                worst_pct, worst_g = pct, g
        if worst_pct is not None and worst_pct < self._gpu_free_pct:
            return False, f"GPU {worst_g} {worst_pct:.0f}% free < {self._gpu_free_pct:.0f}%"
        return True, ""

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
            rt.retry_pending = False  # operator kill is intentional — no auto-relaunch
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
            rt.retry_pending = False  # fresh start — drop any pending backoff
            rt.healthy_streak = 0
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
            # A failed /sleep must not brick the model in SLEEPING forever — the
            # helper recovers it (back to AWAKE if alive, else the crash path).
            if not await self._sleep_victim(rt):
                return
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
        rt.retry_pending = False
        if rt.log_path:
            try:
                rt.reason = launcher.read_failure_reason(rt.log_path) or rt.reason
            except Exception:
                logger.exception("reading failure reason for %s", rt.member.model)
        self._cond.notify_all()

    def _crash_reason(self, rt: ModelRuntime) -> "str | None":
        """Distill the crashing engine's WHY (CUDA OOM, bad arg, …) from its log."""
        if not rt.log_path:
            return None
        try:
            return launcher.read_failure_reason(rt.log_path)
        except Exception:  # noqa: BLE001
            return None

    def _schedule_retry_or_die(self, rt: ModelRuntime, cause: str) -> None:
        """A launch/engine died. Either schedule a PATIENT (exponential-backoff)
        relaunch or, once MAX_RELAUNCH is spent, leave it DEAD. Call under the
        Condition; the corpse must already be terminated. Sets a human `reason`
        (the distilled crash + the retry plan) for the UI + per-session storage.
        While waiting it's DEAD so acquire() fails fast — the monitor relaunches
        it via `retry_pending` when `next_retry_ts` elapses (not a permanent die)."""
        crash = self._crash_reason(rt)
        prefix = (crash + " ") if crash else ""
        rt.inflight = 0
        rt.swapping = False
        rt.healthy_streak = 0
        rt.state = ModelState.DEAD
        budget = "∞" if self._retry_forever else str(self._retry_max)
        if not self._retry_forever and rt.restart_count >= self._retry_max:
            rt.retry_pending = False
            rt.reason = f"{prefix}(crashed: {cause}; gave up after {self._retry_max} relaunches — restart it from the Workers tab)"
            logger.error("%s: gave up after %d relaunches (%s)", rt.member.model, self._retry_max, cause)
        else:
            # Cap the shift so a long "retry forever" run doesn't compute a giant int;
            # min() clamps to the cap anyway (delays plateau at retry_backoff_cap_s).
            delay = min(self._backoff_base * (2 ** min(rt.restart_count, 30)), self._backoff_cap)
            rt.retry_pending = True
            rt.next_retry_ts = time.monotonic() + delay
            rt.reason = f"{prefix}(crashed: {cause}; auto-retry {rt.restart_count + 1}/{budget} in ~{int(delay)}s)"
            logger.warning("%s died (%s); auto-retry %d/%s in ~%.0fs",
                           rt.member.model, cause, rt.restart_count + 1, budget, delay)
        self._cond.notify_all()

    async def _sleep_victim(self, v: ModelRuntime) -> bool:
        """POST /sleep to a member already flipped to SLEEPING. On success return
        True (the caller flips it ASLEEP). On failure (a 5xx or the ~120s /sleep
        timeout) DON'T strand it in SLEEPING — monitor_loop only health-probes
        AWAKE/ASLEEP members, so a stuck-SLEEPING one would never recover and
        would wedge every request routed to it. Recover instead: if its engine
        still answers /health, put it back AWAKE (alive + visible + serveable;
        it just didn't free its GPUs); otherwise terminate the corpse and route
        it into the crash-retry path so the monitor relaunches it. Returns False.

        Runs with the Condition released (like the other /sleep calls) — the slow
        HTTP happens off the lock; only the state flips take it."""
        try:
            await vllm_ctl.sleep_model(self._client, v.base_url, v.member.sleep_level)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("sleep failed for %s: %s", v.member.model, e)
            if await vllm_ctl.is_healthy(self._client, v.base_url):
                async with self._cond:
                    # Only un-stick it if nothing else moved it meanwhile.
                    if v.state == ModelState.SLEEPING:
                        v.state = ModelState.AWAKE
                        v.last_used = time.time()
                    self._cond.notify_all()
            else:
                await launcher.terminate(v.proc)
                v.proc = None
                v.pgid = None
                async with self._cond:
                    self._schedule_retry_or_die(v, "engine died during sleep")
                self._dump_pids()
            return False

    # ---- lifecycle --------------------------------------------------------

    async def _ensure_deps(self) -> None:
        """Bootstrap / top up the shared vLLM venv: create it if missing, install the
        requested vLLM, ninja/cmake (flashinfer JIT), and audio-decode deps for any
        Whisper member. All idempotent + best-effort — fast no-ops once satisfied — so
        it's safe to re-run before a crash relaunch. That's what lets a launch that
        died on a half-finished install (e.g. a wheel download that timed out on a
        slow uplink) retry the INSTALL on the next backoff attempt and self-heal,
        instead of relaunching the same broken venv until the retry budget is spent.
        pre_script is deliberately excluded (it runs once in start())."""
        await launcher.ensure_venv(
            self.cfg.venv_path, self.cfg.vllm_version, self.cfg.vllm_install_args)
        if not self.cfg.vllm_install_args:
            await launcher.ensure_vllm(self.cfg.venv_path, self.cfg.vllm_version)
        await launcher.ensure_build_tools(self.cfg.venv_path)
        await launcher.ensure_audio_deps(self.cfg.venv_path, self.cfg.members)

    async def start(self) -> None:
        """Launch every member sequentially (so two models sharing GPUs never
        load at once and OOM), sleep each after it's healthy, then wake a
        non-overlapping resident set up to GPU capacity."""
        # Kill any vLLM left bound to our ports by a prior crashed/killed worker —
        # we haven't launched ours yet, so anything on these ports is an orphan
        # holding GPU memory that would block our launches.
        await self._kill_stale_vllm()
        # Bootstrap / top up the shared vLLM venv (idempotent). Also re-run before
        # each crash relaunch (see _launch_and_sleep) so a launch that died on a
        # half-finished install self-heals instead of relaunching a broken venv.
        await self._ensure_deps()
        # Optional operator setup script (e.g. building DeepGEMM) — runs once now,
        # with the venv ready and on PATH, before any model launches. NOT in
        # _ensure_deps: it can be expensive / non-idempotent, so it stays once-only.
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
            # On a crash relaunch (restart_count>0, not the initial wave launch),
            # re-run the venv/dep ensure first — so a launch that died because the
            # install was incomplete (e.g. a wheel download timed out) retries the
            # INSTALL, not just the vLLM process, and self-heals once it succeeds.
            # Done before _hold_gpus: installs need no GPU and can take minutes.
            if rt.restart_count > 0:
                await self._ensure_deps()
            # Fresh timestamped log file for this attempt — never overwrites the
            # prior (possibly crashing) log; the shipper follows it + tags the
            # session so the UI can open historical logs.
            rt.log_path = launcher.new_log_path(rt.member, self.log_dir)
            launcher.prune_log_files(rt.member, self.log_dir, LOG_RETAIN)
            async with self._hold_gpus(rt.gpus):
                # Free rt's GPUs before loading: sleep any AWAKE model that shares
                # them, or the load OOMs against a resident (e.g. relaunching a
                # 2,3 model while Mistral holds 0,1,2,3). At initial startup this
                # is a no-op since nothing is awake yet.
                await self._evict_overlapping(rt)
                rt.proc = await launcher.launch_member(rt.member, self.log_dir, self._python_exe, log_path=rt.log_path)
                # Record the engine's process group right away (before health) so a
                # worker SIGKILL mid-load still leaves a trackable orphan, not a
                # leak we'd have to pkill box-wide.
                rt.pgid = self._safe_pgid(rt.proc)
                self._dump_pids()
                ok = await launcher.wait_health(self._client, rt.member, rt.proc, log_path=rt.log_path)
                if not ok:
                    # Kill the failed/hung engine so it can't leak and keep holding
                    # GPU/CPU, then schedule a patient backoff relaunch (or give up).
                    await launcher.terminate(rt.proc)
                    rt.proc = None
                    rt.pgid = None
                    async with self._cond:
                        self._schedule_retry_or_die(rt, "engine did not become healthy on launch")
                    self._dump_pids()
                    return
                # Only sleep a member whose GPUs ANOTHER member also wants — that's
                # the only reason to evict it (time-sharing). A member that's the
                # sole occupant of its GPUs (e.g. a single-model fleet, or GLM-5.1
                # tp=8 across all 8) has nothing to time-share with, so sleeping it
                # is pointless AND dangerous: waking a huge model back onto the GPU
                # can CUDA-OOM (`/wake_up → 500 out of memory`), leaving it flapping
                # asleep↔waking and wedging every request. Keep it resident (AWAKE).
                contended = any(r is not rt and (r.gpus & rt.gpus) for r in self._runtimes)
                if contended:
                    await vllm_ctl.sleep_model(self._client, rt.base_url, rt.member.sleep_level)
                    async with self._cond:
                        rt.state = ModelState.ASLEEP
                        self._cond.notify_all()
                else:
                    async with self._cond:
                        rt.state = ModelState.AWAKE
                        rt.last_used = time.time()
                        self._cond.notify_all()
                # Re-dump now that the tp workers have spawned, so the recorded
                # group includes every child.
                self._dump_pids()
        except Exception as e:  # noqa: BLE001
            logger.exception("launch_and_sleep failed for %s", rt.member.model)
            await launcher.terminate(rt.proc)
            rt.proc = None
            rt.pgid = None
            async with self._cond:
                self._schedule_retry_or_die(rt, f"launch error: {type(e).__name__}")
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
            # A failed /sleep must not brick the victim in SLEEPING forever — the
            # helper recovers it (back to AWAKE if alive, else the crash path).
            if not await self._sleep_victim(v):
                continue
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
        except Exception as e:
            logger.exception("swap_to %s failed", rt.member.model)
            async with self._cond:
                # Leave it asleep so a later request can retry; never strand it.
                # Surface the cause (e.g. a CUDA-OOM on /wake_up of an oversized
                # model) in the member status so the UI shows WHY it's not serving
                # instead of silently flapping asleep↔waking.
                if rt.state != ModelState.AWAKE:
                    rt.state = ModelState.ASLEEP
                rt.reason = f"wake failed: {str(e)[:200]}"
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
                # (0) A crash backoff is pending → relaunch once the patience timer
                # elapses. Checked BEFORE the DEAD skip: a backing-off model is DEAD
                # (acquire() fails fast) but is NOT a permanent give-up.
                if rt.retry_pending:
                    if time.monotonic() < rt.next_retry_ts:
                        continue
                    # Optionally hold the relaunch until this member's GPUs have free
                    # VRAM — relaunching into a card a foreign job is hogging would
                    # just OOM and burn a retry. Re-poll on the next tick WITHOUT
                    # spending the budget (restart_count unchanged) until it frees.
                    if self._require_free_gpu:
                        ok, why = await self._gpus_free_enough(rt.gpus)
                        if not ok:
                            async with self._cond:
                                if rt.retry_pending:  # no operator action raced us
                                    rt.next_retry_ts = time.monotonic() + self._backoff_base
                                    budget = "∞" if self._retry_forever else str(self._retry_max)
                                    rt.reason = (f"waiting for GPU memory ({why}); auto-retry "
                                                 f"{rt.restart_count + 1}/{budget} once free")
                                self._cond.notify_all()
                            logger.info("%s: retry held — %s", rt.member.model, why)
                            continue
                    async with self._cond:
                        rt.retry_pending = False
                        rt.state = ModelState.LAUNCHING
                        rt.reason = None
                        rt.health_fail = 0
                        self._cond.notify_all()
                    rt.restart_count += 1
                    logger.warning("relaunching %s (backoff attempt %d/%s)",
                                   rt.member.model, rt.restart_count,
                                   "∞" if self._retry_forever else str(self._retry_max))
                    await self._launch_and_sleep(rt)  # success → AWAKE/ASLEEP; fail → schedules next backoff
                    continue
                proc = rt.proc
                if proc is None or rt.state == ModelState.DEAD:
                    continue
                died = proc.returncode is not None
                cause = f"exited rc={proc.returncode}"
                # Engine-liveness probe — only on settled states (a wake/sleep/
                # launch in flight is allowed to be briefly unresponsive).
                if not died and rt.state in (ModelState.AWAKE, ModelState.ASLEEP):
                    if await vllm_ctl.is_healthy(self._client, rt.base_url):
                        rt.health_fail = 0
                        rt.healthy_streak += 1
                        # Sustained health → recovered; reset the crash budget so a
                        # later, unrelated incident gets its own full set of retries.
                        if rt.restart_count and rt.healthy_streak >= HEALTHY_RESET_TICKS:
                            rt.restart_count = 0
                    else:
                        rt.health_fail += 1
                        rt.healthy_streak = 0
                        if rt.health_fail >= self._health_fail_limit:
                            died = True
                            cause = f"engine unresponsive ({rt.health_fail}× /health)"
                if not died:
                    continue
                # Engine wedged (process alive but unresponsive)? Group-kill it so it
                # stops holding GPU memory before any relaunch — else the relaunch
                # OOMs against the corpse. Harmless no-op once the process exited.
                if proc.returncode is None:
                    await launcher.terminate(proc)
                rt.proc = None
                rt.pgid = None
                # Schedule a PATIENT backoff relaunch (or give up after the budget) —
                # captures the crash reason (CUDA OOM, …) into rt.reason per session.
                async with self._cond:
                    self._schedule_retry_or_die(rt, cause)
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
                # The launch session `reason` refers to (the crashing log file) so
                # the gateway can persist the crash reason per session for history.
                "session": launcher.session_of(rt.log_path),
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
