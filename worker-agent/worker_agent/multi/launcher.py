"""Launch a single vLLM server for one member model, pinned to its GPUs and
sleep-mode enabled. Health is gated on vLLM's /health (200 = model loaded)."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import time

import httpx

from .config import MemberModel
from . import vllm_ctl

logger = logging.getLogger("worker-agent.launcher")


# ---- Huawei Ascend (NPU) support -------------------------------------------
# An Ascend box is detected by the driver's manager device (present on every
# CANN install; processes hold THIS device, never /dev/davinciN). On Ascend the
# venv installs vllm + vllm-ascend (the form supplies vllm_install_args), the
# venv python must be >=3.10 (Euler boxes ship 3.9), members pin NPUs via
# ASCEND_RT_VISIBLE_DEVICES, and every install/launch subprocess needs the CANN
# env (LD_LIBRARY_PATH/ASCEND_*/PYTHONPATH from set_env.sh).
_ASCEND_SETUP_SCRIPTS = (
    "/usr/local/Ascend/ascend-toolkit/set_env.sh",
    "/usr/local/Ascend/nnal/atb/set_env.sh",  # optional (ATB); absent on some boxes
)
ASCEND_VENV_PYTHON = "3.11"


def is_ascend() -> bool:
    return os.path.exists("/dev/davinci_manager")


_ascend_env_cache: dict | None = None


def ascend_env() -> dict:
    """Environment after sourcing the CANN setup scripts (cached). Captured via
    `bash -c 'source …; env -0'` so multi-line values survive."""
    global _ascend_env_cache
    if _ascend_env_cache is not None:
        return _ascend_env_cache
    scripts = [s for s in _ASCEND_SETUP_SCRIPTS if os.path.exists(s)]
    env: dict = {}
    if scripts:
        src = "; ".join(f"source {s}" for s in scripts)
        try:
            out = subprocess.run(
                ["bash", "-c", f"{src}; env -0"], capture_output=True, timeout=30,
            ).stdout
            for chunk in out.split(b"\0"):
                if b"=" in chunk:
                    k, _, v = chunk.partition(b"=")
                    env[k.decode()] = v.decode()
        except Exception:
            logger.exception("failed to capture Ascend CANN env — vLLM may not find libascendcl")
            env = {}
    _ascend_env_cache = env
    return env

# Cold model load can take many minutes: a large MoE (e.g. 35B-A3B) at a long
# max-model-len re-downloads tens of GB (no persistent volume) and then spends
# ~13 min in engine init (weight load + compile + CUDA-graph capture) before
# /health flips to 200. 900s used to clip models that were *seconds* from ready,
# so the scheduler killed a nearly-loaded server. Give cold starts real headroom;
# override with VLLM_LAUNCH_HEALTH_TIMEOUT_S. (Fatal-error scan still aborts early.)
LAUNCH_HEALTH_TIMEOUT_S = float(os.environ.get("VLLM_LAUNCH_HEALTH_TIMEOUT_S", "2400"))  # 40 min


def _member_slug(member: MemberModel) -> str:
    return member.served_name.replace("/", "__")


def log_path_for(member: MemberModel, log_dir: str) -> str:
    """Legacy stable per-model vLLM log path (no timestamp). Retained for the
    rare caller that wants the fixed name; live launches now use a timestamped
    file via `new_log_path` so a relaunch never overwrites the crashing log."""
    return os.path.join(log_dir, f"vllm-{_member_slug(member)}-{member.port}.log")


# Each (re)launch writes a NEW timestamped file `vllm-{slug}-{port}-{ts}.log` so
# every attempt's log (crash traceback included) is kept on disk — never
# truncated/overwritten by the next launch. The shipper follows the current one;
# the gateway buckets per session so the UI can open historical logs.
_TS_FMT = "%Y%m%d-%H%M%S"


def new_log_path(member: MemberModel, log_dir: str, ts: str | None = None) -> str:
    ts = ts or time.strftime(_TS_FMT)
    return os.path.join(log_dir, f"vllm-{_member_slug(member)}-{member.port}-{ts}.log")


def session_of(log_path: str | None) -> str | None:
    """The session id (the trailing timestamp) of a timestamped log file, or None
    for the legacy fixed file. `vllm-foo__bar-18001-20260616-120701.log` → that ts."""
    if not log_path:
        return None
    base = os.path.basename(log_path)
    m = re.search(r"-(\d{8}-\d{6})\.log$", base)
    return m.group(1) if m else None


def list_log_files(member: MemberModel, log_dir: str) -> list[str]:
    """All timestamped log files for a member, NEWEST first (lexical ts sort)."""
    import glob
    pat = os.path.join(log_dir, f"vllm-{_member_slug(member)}-{member.port}-*.log")
    return sorted(glob.glob(pat), reverse=True)


def prune_log_files(member: MemberModel, log_dir: str, keep: int) -> None:
    """Delete the OLDEST timestamped logs beyond `keep` (a high safety cap so the
    VM disk can't fill across many relaunches). Best-effort."""
    if keep <= 0:
        return
    files = list_log_files(member, log_dir)  # newest first
    for stale in files[keep:]:
        try:
            os.remove(stale)
        except OSError:
            pass


_LOG_PREFIX_RE = re.compile(
    r"^(?:\([^)]*\)\s*)?"                              # optional (Worker pid=..)
    r"(?:(?:ERROR|WARNING|INFO|DEBUG|CRITICAL)\s+)?"   # log level
    r"(?:\d\d-\d\d\s+[\d:.]+\s+)?"                     # MM-DD HH:MM:SS
    r"(?:\[[^\]]+\]\s*)?"                              # [file.py:123]
)


def _clean_line(line: str) -> str:
    """Strip vLLM's '(Worker pid=..) ERROR 05-29 12:35:48 [file.py:123] ' noise
    so the surfaced reason reads like a sentence, not a log line."""
    return _LOG_PREFIX_RE.sub("", line).strip()


def read_failure_reason(log_path: str, tail_bytes: int = 65536) -> str | None:
    """Best-effort: distill a model's vLLM log tail into one human sentence we
    can show in the UI when it dies. Most specific cause wins (GPU OOM before a
    generic traceback). Returns None if the log is unreadable/empty."""
    try:
        size = os.path.getsize(log_path)
        with open(log_path, "rb") as f:
            if size > tail_bytes:
                f.seek(size - tail_bytes)
                f.readline()  # drop the partial first line
            raw = f.read()
    except OSError:
        return None
    lines = [l for l in raw.decode("utf-8", "replace").splitlines() if l.strip()]
    if not lines:
        return None

    # 1) Pinned GPUs don't have enough free VRAM — the "GPU is not enough" case.
    for line in lines:
        m = re.search(
            r"Free memory on device (\S+) \(([\d.]+)/([\d.]+) GiB\).*?"
            r"utilization \(([\d.]+),\s*([\d.]+) GiB\)", line)
        if m:
            dev, free, total, util, want = m.groups()
            return (
                f"Not enough free GPU memory: {dev} has {free} GiB free of {total} GiB, "
                f"but vLLM wants ~{want} GiB at --gpu-memory-utilization {util}. "
                f"Free other jobs on this GPU, lower --gpu-memory-utilization, or pin idle GPUs."
            )
    # 2) Allocator OOM partway through loading.
    for line in lines:
        if "CUDA out of memory" in line or "torch.OutOfMemoryError" in line:
            return ("CUDA out of memory while loading — lower --gpu-memory-utilization or "
                    "--max-model-len, or raise tensor-parallel size.")
    # 3) Bad/unsupported vLLM arguments.
    for line in lines:
        if re.search(r"(unrecognized arguments|invalid choice|error: argument|"
                     r"the following arguments are required)", line):
            return f"Invalid vLLM argument: {_clean_line(line)}"[:300]
    # 4) Model can't be fetched from Hugging Face.
    for line in lines:
        if re.search(r"(Repository Not Found|Entry Not Found|GatedRepo|401 Client Error|"
                     r"403 Client Error|does not appear to have|is not a valid model)", line):
            return f"Model could not be loaded from Hugging Face: {_clean_line(line)}"[:300]
    # 5) Missing dependency in the venv. (Skip vllm-ascend's benign triton probe —
    #    see _BENIGN_ERROR_RE.)
    for line in lines:
        if _BENIGN_ERROR_RE.search(line):
            continue
        if "ModuleNotFoundError" in line or "No module named" in line or "ImportError" in line:
            return _clean_line(line)[:300]
    # 6) Any explicit exception line (last one — closest to the crash), skipping
    #    vLLM's generic "see root cause above" wrappers.
    for line in reversed(lines):
        if re.search(r"(Error|Exception):\s*\S", line):
            if "Engine core initialization failed" in line or "WorkerProc initialization failed" in line:
                continue
            return _clean_line(line)[:300]
    # 7) Last resort: the final log line.
    return _clean_line(lines[-1])[:300]


async def launch_member(
    member: MemberModel, log_dir: str, python_exe: str = "python3",
    log_path: str | None = None,
) -> asyncio.subprocess.Process:
    """Spawn `vllm.entrypoints.openai.api_server` for one model. Stdout/stderr
    go to a per-model log file the shipper can tail. `python_exe` is the
    interpreter that has vLLM — a uv venv's bin/python when the endpoint set a
    venv_path, else bare `python3` on PATH. `log_path` (a fresh timestamped file
    from `new_log_path`) is where this attempt's output lands; omitted → legacy
    fixed file."""
    os.makedirs(log_dir, exist_ok=True)
    env = dict(os.environ)
    devices = ",".join(str(g) for g in member.gpu_indices)
    if is_ascend():
        env.update(ascend_env())  # CANN libs; without this vLLM can't load libascendcl
        env["ASCEND_RT_VISIBLE_DEVICES"] = devices
    else:
        env["CUDA_VISIBLE_DEVICES"] = devices
    env["VLLM_SERVER_DEV_MODE"] = "1"  # exposes /sleep, /wake_up, /collective_rpc
    # Xet (hf_hub's chunked transfer backend) stalls/hangs on some hosts — the
    # platform already disables it for mirror pushes and dataset transforms.
    # Default OFF for model downloads too; an endpoint env var of 0 re-enables.
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    # vLLM's engine-core-ready has its OWN 600s internal timeout (separate from
    # the worker's health wait). A big model over many devices legitimately loads
    # slower than that — verified: Qwen3-32B TP8 on Ascend blew past 600s in eager
    # init and died with "Timed out waiting for engine core processes to start".
    # Raise it (operator env still wins) so slow multi-device loads aren't clipped.
    env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "2400")
    if is_ascend():
        # Multi-NPU (TP>1) HCCL: the default FFTS+ collective mode DEADLOCKS at
        # model-load on Ascend — verified Qwen3-32B TP8 froze silently at "Starting
        # to load model", 0% AICore, until killed. vLLM itself logs a warning
        # recommending AIV. Setting it makes TP8 init complete + serve (verified).
        env.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
    # Put the venv's bin dir on PATH (running {venv}/bin/python directly does NOT
    # activate the venv). flashinfer JIT-compiles sampling/attention kernels at
    # runtime via the `ninja` console script — without {venv}/bin on PATH it dies
    # with "FileNotFoundError: ninja" on Blackwell. Also exposes the venv's cmake.
    bindir = os.path.dirname(python_exe)
    if bindir and os.path.isabs(bindir):
        env["VIRTUAL_ENV"] = os.path.dirname(bindir)
        env["PATH"] = bindir + ":" + env.get("PATH", "")

    # Ascend-safe defaults, each applied only when the member hasn't set it:
    #  --enforce-eager: vllm-ascend's graph-mode compile ("OOT custom backend")
    #    HANGS at larger scale (verified: silent 30-min stall at TP8/32B, 0%
    #    AICore; tiny models compile fine). Eager trades peak throughput for a
    #    launch that actually completes.
    #  --gpu-memory-utilization 0.8: a 910B3 has ~8 GiB baseline overhead, so the
    #    default 0.9 (~54.9 GiB) exceeds free HBM (~52.8) and trips vLLM's
    #    memory-check — especially racing residual memory from a prior attempt.
    ascend_defaults: list[str] = []
    if is_ascend():
        joined = " ".join(member.extra_args)
        if "--enforce-eager" not in member.extra_args:
            ascend_defaults.append("--enforce-eager")
        if "--gpu-memory-utilization" not in joined and "--gpu_memory_utilization" not in joined:
            ascend_defaults += ["--gpu-memory-utilization", "0.8"]
    args = [
        python_exe, "-m", "vllm.entrypoints.openai.api_server",
        "--model", member.model,
        "--served-model-name", member.served_name,
        "--port", str(member.port),
        "--tensor-parallel-size", str(member.tp),
        # Pipeline parallel: split the model's layers across pp GPU groups. Total
        # GPUs used = tp * pp (the member's gpu_indices). Omitted when pp == 1.
        *(["--pipeline-parallel-size", str(member.pp)] if member.pp > 1 else []),
        # Sleep mode is skipped on Ascend: vllm-ascend's CaMeM allocator
        # (aclrtMallocPhysical) OOMs on hosts without hugepages configured —
        # members stay resident instead of sleep/wake time-sharing.
        *([] if is_ascend() else ["--enable-sleep-mode"]),
        *ascend_defaults,
        *member.extra_args,
    ]
    log_path = log_path or new_log_path(member, log_dir)
    # Fresh per-launch file (timestamped) → this attempt's log is self-contained
    # and a relaunch never overwrites a prior crash. "wb" is safe: the name is
    # unique per launch. The shipper follows the current file + tags its session.
    logf = open(log_path, "wb", buffering=0)
    logger.info(
        "launching vllm: model=%s %s=%s tp=%d pp=%d port=%d → %s",
        member.model, "npus" if is_ascend() else "gpus", devices,
        member.tp, member.pp, member.port, log_path,
    )
    # start_new_session=True puts the engine + its tp worker children in their
    # own process group so terminate() can SIGKILL the whole group (no orphans).
    proc = await asyncio.create_subprocess_exec(
        *args, env=env, stdout=logf, stderr=logf, start_new_session=True,
    )
    return proc


def _find_uv() -> str | None:
    """Locate a usable `uv` binary. The worker process PATH may not include the
    user-local install dirs the standalone installer writes to, so check those
    explicitly after PATH."""
    p = shutil.which("uv")
    if p:
        return p
    home = os.path.expanduser("~")
    for cand in (f"{home}/.local/bin/uv", f"{home}/.cargo/bin/uv",
                 "/root/.local/bin/uv", "/usr/local/bin/uv"):
        if os.path.exists(cand):
            return cand
    return None


async def _stream_subprocess(proc: asyncio.subprocess.Process, prefix: str = "  ") -> int:
    """Drain a process's merged stdout line-by-line into the logger so long-running
    installs stream into the worker's __worker__ log live, instead of going silent
    until they finish (a `uv pip install vllm` on a fresh venv is several minutes of
    torch+vLLM download). Returns the exit code."""
    if proc.stdout is not None:
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                logger.info("%s%s", prefix, line)
    return await proc.wait()


async def ensure_uv() -> str | None:
    """Return a path to a `uv` binary, installing it via Astral's standalone
    installer if the VM doesn't already have it. The vLLM-venv bootstrap needs uv
    and some VMs ship without it. The installer needs no pip/root — it drops a
    static binary in ~/.local/bin. Best-effort; returns None if uv is unavailable."""
    uv = _find_uv()
    if uv:
        return uv
    logger.info("uv not found on VM — installing via astral.sh standalone installer")
    try:
        proc = await asyncio.create_subprocess_shell(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        rc = await _stream_subprocess(proc)
        if rc != 0:
            logger.error("uv install failed rc=%s", rc)
    except Exception:
        logger.exception("uv install failed")
    uv = _find_uv()
    if not uv:
        logger.error("uv still not found after install attempt — cannot bootstrap venvs")
    return uv


# Bootstrapping a vLLM venv pulls huge CUDA/torch wheels (nvidia-cudnn, nvshmem,
# nvjitlink, cmake…). On a throttled uplink uv's default 30s per-request HTTP
# timeout + high download concurrency make these time out — uv itself says so:
# "Failed to download … Try increasing UV_HTTP_TIMEOUT (current value: 30s)".
# Give uv a long per-request timeout and fewer parallel streams (so each download
# gets more of the limited bandwidth). Only set defaults the operator hasn't —
# UV_HTTP_TIMEOUT / UV_CONCURRENT_DOWNLOADS in the worker env still win.
_UV_ENV_DEFAULTS = {
    "UV_HTTP_TIMEOUT": "900",        # 15 min per request (uv default: 30s)
    "UV_CONCURRENT_DOWNLOADS": "8",  # fewer parallel streams on a slow link
}


def _uv_env() -> dict:
    env = dict(os.environ)
    if is_ascend():
        # CANN env for installs too — a vllm-ascend sdist build compiles custom
        # ops against the toolkit (harmless overlay when a wheel is picked).
        env.update(ascend_env())
    for k, v in _UV_ENV_DEFAULTS.items():
        env.setdefault(k, v)
    return env


async def _uv_run(uv: str, args: list[str], what: str, env_extra: dict | None = None) -> bool:
    """Run `uv <args>`, streaming its output to the log. Returns True on success.
    `env_extra` overlays extra env vars on the install subprocess (e.g.
    VLLM_USE_PRECOMPILED=1 for a fast git-fork install)."""
    env = _uv_env()
    if env_extra:
        env.update(env_extra)
    logged = (" ".join(f"{k}={v}" for k, v in (env_extra or {}).items()) + " ").lstrip()
    logger.info("%s: %suv %s", what, logged, " ".join(args))
    try:
        proc = await asyncio.create_subprocess_exec(
            uv, *args, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        rc = await _stream_subprocess(proc)
        if rc != 0:
            logger.error("%s failed rc=%s", what, rc)
            return False
        logger.info("%s ✓", what)
        return True
    except Exception:
        logger.exception("%s raised", what)
        return False


async def _python_has_module(py: str, mod: str) -> bool:
    """True if `{py} -c 'import {mod}'` succeeds — used to tell a fully-bootstrapped
    venv from a half-built one (uv venv made bin/python, but the vLLM install was
    interrupted), so a re-boot RESUMES the install instead of skipping it."""
    try:
        proc = await asyncio.create_subprocess_exec(
            py, "-c", f"import {mod}",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        return (await proc.wait()) == 0
    except Exception:
        return False


async def ensure_venv(
    venv_path: str | None,
    vllm_version: str | None,
    vllm_install_args: str | None = None,
) -> None:
    """Bootstrap the endpoint's vLLM venv when it doesn't exist yet, so an endpoint
    can name a `venv_path` that hasn't been created on the VM and the worker builds
    it on first boot instead of dying with FileNotFoundError on `{venv_path}/bin/python`.

    No-op when venv_path is unset (bare `python3` on PATH) or the venv already has
    vLLM importable. Installs uv if the VM lacks it, `uv venv`s the path (creating
    parent dirs), then installs vLLM. The install is, in priority order:
      • `vllm_install_args` — a full `uv pip install` argument string, used verbatim
        (e.g. a nightly: `-U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu130
        --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match`).
        The operator owns the whole spec, so we DON'T inject --torch-backend.
      • else `vllm==vllm_version` (or bare `vllm`) with `--torch-backend=auto` so uv
        picks the PyTorch CUDA build matching the driver.
    RESUMABLE: a venv with bin/python but no importable vLLM (an interrupted earlier
    boot) re-runs the install rather than launching a broken venv. Install output
    streams to the log so a failure (e.g. an unsupported GPU arch) is visible live."""
    if not venv_path:
        return
    py = f"{venv_path}/bin/python"
    custom = (vllm_install_args or "").strip()
    # Signature of the requested install — recorded in a marker so a re-boot can tell
    # "already built to THIS spec" (skip) from "operator changed the build" (reinstall).
    spec_sig = custom if custom else (f"vllm=={vllm_version}" if vllm_version else "vllm")
    marker = f"{venv_path}/.sgpu_vllm_spec"
    have_py = os.path.exists(py)
    if have_py and await _python_has_module(py, "vllm"):
        try:
            recorded = open(marker).read().strip()
        except OSError:
            recorded = None
        if recorded is None:
            # No marker → a venv WE didn't build (operator-managed, e.g. a shared
            # /share/vllm-venv). Never touch it.
            return
        if recorded == spec_sig:
            return  # already built to exactly this spec
        logger.info("venv %s vLLM spec changed (%r → %r) — reinstalling", venv_path, recorded, spec_sig)
    uv = await ensure_uv()
    if not uv:
        logger.error("cannot bootstrap venv %s — uv unavailable on the VM", venv_path)
        return
    desc = "custom args" if custom else (f"vllm=={vllm_version}" if vllm_version else "vllm")
    if not have_py:
        logger.info("venv %s missing — bootstrapping (uv venv + pip install %s); "
                    "first install can take several minutes", venv_path, desc)
        # uv venv doesn't reliably create missing PARENT dirs (e.g. /share2 on a
        # fresh VM) — make them so `uv venv /share2/vllm-venv2` can't fail on a
        # missing /share2.
        parent = os.path.dirname(venv_path.rstrip("/"))
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError:
                logger.exception("could not create parent dir %s for venv", parent)
        # Ascend: vllm-ascend needs python >=3.10 but Euler boxes ship 3.9 — pin
        # the venv to a uv-managed 3.11 (uv downloads a standalone build).
        venv_args = ["venv", venv_path]
        if is_ascend():
            venv_args = ["venv", "--python", ASCEND_VENV_PYTHON, venv_path]
        if not await _uv_run(uv, venv_args, f"create venv {venv_path}"):
            return
    else:
        logger.info("venv %s exists but vLLM not importable — (re)installing %s", venv_path, desc)
    if custom:
        # Operator-provided install command(s). MULTI-LINE: each non-empty line runs
        # as its OWN `uv pip install`, in order — needed when one resolution can't
        # express the stack (Ascend: vllm and vllm-ascend pin conflicting torch
        # versions, so they only install sequentially; a trailing pin line can also
        # downgrade a dep, e.g. z3-solver for an old-glibcxx host). Run verbatim —
        # no --torch-backend injection, no fallback.
        for line_no, line in enumerate(
                (ln.strip() for ln in custom.splitlines() if ln.strip()), start=1):
            try:
                extra = shlex.split(line)
            except ValueError as e:
                logger.error("vllm_install_args line %d is not a valid shell arg string (%s) — skipping line", line_no, e)
                continue
            # Leading NAME=VALUE tokens are environment assignments for the install
            # subprocess (shell-style), e.g. "VLLM_USE_PRECOMPILED=1 git+https://…@ref":
            # lets a git-fork install reuse precompiled vLLM binaries (fast, no CUDA
            # toolchain) without that var leaking into a real pip arg. Everything from
            # the first non-assignment token on is passed to `uv pip install` verbatim.
            # NAME=VALUE but NOT a pip requirement: `(?!=)` rejects `vllm==0.23.0`
            # (and `name>=…`/`name[extra]==…` already fail the leading-char class).
            env_extra: dict[str, str] = {}
            i = 0
            while i < len(extra) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=(?!=)", extra[i]):
                k, v = extra[i].split("=", 1)
                env_extra[k] = v
                i += 1
            pip_args = extra[i:]
            if not pip_args:
                logger.error("vllm_install_args line %d has no install target after env assignments — skipping line", line_no)
                continue
            await _uv_run(uv, ["pip", "install", "--python", py, *pip_args],
                          f"install vLLM (custom, line {line_no}) into {venv_path}",
                          env_extra=env_extra or None)
    else:
        spec = f"vllm=={vllm_version}" if vllm_version else "vllm"
        if is_ascend():
            # A plain CUDA vLLM can't serve NPUs — the endpoint should carry an
            # Ascend vllm_install_args (vllm==X + vllm-ascend==Y with the Huawei
            # extra index; the create form generates it). Install anyway so the
            # failure is visible in the model log, but say why loudly.
            logger.error(
                "Ascend NPU box but no vllm_install_args — installing plain %s, which "
                "CANNOT serve NPUs. Set the Ascend install line on the endpoint.", spec)
        # --torch-backend=auto lets uv pick the PyTorch CUDA build matching the VM's
        # driver (important on bleeding-edge GPUs); retry without it for older uv
        # that doesn't recognise the flag.
        if not await _uv_run(uv, ["pip", "install", "--python", py, "--torch-backend=auto", spec],
                             f"install {spec} into {venv_path}"):
            await _uv_run(uv, ["pip", "install", "--python", py, spec],
                          f"install {spec} into {venv_path} (no torch-backend)")
    if await _python_has_module(py, "vllm"):
        try:
            with open(marker, "w") as fh:
                fh.write(spec_sig)
        except OSError:
            pass  # marker is best-effort (just affects reinstall-on-spec-change)
        logger.info("bootstrapped venv %s ✓ (spec=%s)", venv_path, spec_sig)
    else:
        logger.error("venv %s bootstrap finished but vLLM still not importable", venv_path)


async def ensure_build_tools(venv_path: str | None) -> None:
    """Make sure `ninja` (and `cmake`) are in the venv. vLLM's flashinfer backend
    JIT-compiles its sampling/attention kernels at runtime on newer GPUs (Blackwell)
    and shells out to the `ninja` build tool — without it the engine dies at
    profile_run with `FileNotFoundError: ninja`. These are cheap pure-Python wheels
    (each bundles a static binary into {venv}/bin); launch_member puts {venv}/bin on
    PATH so the JIT finds them. No-op when venv_path is unset or ninja is present."""
    if not venv_path:
        return
    if os.path.exists(f"{venv_path}/bin/ninja"):
        return  # already there
    py = f"{venv_path}/bin/python"
    if not os.path.exists(py):
        return
    uv = await ensure_uv()
    if not uv:
        logger.warning("`uv` unavailable — cannot ensure ninja/cmake; flashinfer JIT may fail")
        return
    await _uv_run(uv, ["pip", "install", "--python", py, "ninja", "cmake"],
                 f"ensure build tools (ninja, cmake) in {venv_path}")


async def run_pre_script(pre_script: str | None, venv_path: str | None) -> None:
    """Run an optional operator-provided setup script once per worker boot, AFTER the
    vLLM venv is bootstrapped and BEFORE any model launches. Some models need extra
    build/install steps that don't fit a `pip install` — e.g. DeepGEMM:
        bash <(curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh)
    The script runs under bash (so process substitution `<(…)` works) with the vLLM
    venv on PATH and VIRTUAL_ENV set, so `pip`/`python` in the script target that
    venv. Output streams to the worker log. Best-effort: a non-zero exit is logged
    but doesn't abort the fleet (the model launch surfaces a clearer error if the
    script was actually required)."""
    if not pre_script or not pre_script.strip():
        return
    env = _uv_env()  # same long uv HTTP timeout — pre-scripts often pip/uv install
    if venv_path:
        env["VIRTUAL_ENV"] = venv_path
        env["PATH"] = f"{venv_path}/bin:" + env.get("PATH", "")
    logger.info("running pre-script (%d chars) before launch", len(pre_script))
    try:
        # bash -lc (not sh) — pre-scripts commonly use `bash <(curl …)` process
        # substitution, which sh doesn't support.
        proc = await asyncio.create_subprocess_exec(
            "bash", "-lc", pre_script, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        rc = await _stream_subprocess(proc, prefix="  [pre] ")
        if rc != 0:
            logger.error("pre-script exited rc=%s (continuing — launch will validate)", rc)
        else:
            logger.info("pre-script completed ✓")
    except Exception:
        logger.exception("pre-script failed to run")


async def ensure_vllm(venv_path: str | None, vllm_version: str | None) -> None:
    """Best-effort: make sure `vllm==vllm_version` is installed in the endpoint's
    uv venv before launching models. No-op unless both are set. Idempotent — uv
    is a fast no-op when the version is already satisfied. Failures are logged
    (the launch will surface a clearer error if vLLM is genuinely missing)."""
    if not (venv_path and vllm_version):
        return
    py = f"{venv_path}/bin/python"
    if not os.path.exists(py):
        logger.warning("venv_path %s has no bin/python — skipping vllm ensure", venv_path)
        return
    uv = await ensure_uv()
    if not uv:
        logger.warning("`uv` unavailable — cannot ensure vllm==%s; relying on venv as-is", vllm_version)
        return
    await _uv_run(uv, ["pip", "install", "--python", py, f"vllm=={vllm_version}"],
                 f"ensure vllm=={vllm_version} in {venv_path}")


# Heuristic for "this member is a Whisper/ASR transcription model" — used to pull
# in vLLM's audio-decode deps. Errs toward matching (a false positive just means a
# harmless extra install).
_AUDIO_MODEL_RE = re.compile(r"whisper|asr|transcrib|speech[-_]?to[-_]?text", re.I)


def is_audio_model(member) -> bool:
    """A member is audio/ASR if it's explicitly tagged `task="transcription"` (the
    reliable signal — works for any model name, incl. custom finetunes) OR its id
    matches the Whisper-family name heuristic (convenience for the common case)."""
    if getattr(member, "task", None) == "transcription":
        return True
    model = getattr(member, "model", None)
    return bool(model and _AUDIO_MODEL_RE.search(model))


async def ensure_audio_deps(venv_path: str | None, members) -> None:
    """If any member is a transcription/ASR model (Whisper), make sure vLLM's
    audio-decode deps are in the venv. Without them vLLM rejects every clip with
    'Invalid or unsupported audio file', even valid WAVs. Best-effort + idempotent,
    like ensure_vllm.

    `soundfile` (bundled libsndfile ≥1.1) covers wav/flac/ogg/mp3; `resampy` is
    what vLLM's loader uses to resample to the model's 16 kHz (without it, any
    non-16 kHz clip fails — only same-rate WAVs slip through); `av` (PyAV, which
    bundles ffmpeg libraries) is vLLM's fallback decoder for m4a/aac/webm/video.
    vLLM uses PyAV — NOT the system `ffmpeg` binary — so this is fully
    pip-installable into the uv venv. This is the `vllm[audio]` extra set."""
    if not venv_path or not any(is_audio_model(m) for m in members):
        return
    py = f"{venv_path}/bin/python"
    if not os.path.exists(py):
        logger.warning("venv_path %s has no bin/python — skipping audio-deps ensure", venv_path)
        return
    uv = await ensure_uv()
    if not uv:
        logger.warning("`uv` unavailable — cannot ensure audio deps; whisper members will reject audio")
        return
    await _uv_run(uv, ["pip", "install", "--python", py, "librosa", "soundfile", "resampy", "av"],
                 f"ensure audio deps in {venv_path}")


# Definitive "this launch is doomed" markers. Seeing any of these in the log
# means the engine won't recover, so we abort the health wait immediately rather
# than block the whole startup for the full 900s timeout (which is what happens
# when a pinned GPU is held by another tenant and vLLM spins retrying).
_FATAL_MARKERS = (
    "less than desired GPU memory utilization",
    "Free memory on device",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "Engine core initialization failed",
    "WorkerProc initialization failed",
    "unrecognized arguments",
    "ModuleNotFoundError",
    "No module named",
)

# Log lines that LOOK fatal but aren't: vllm-ascend probes optional Triton kernels
# at import ("Failed to import Triton kernels … No module named 'triton.…'") and
# serves fine without them — killing the launch on that line aborts a healthy boot.
_BENIGN_ERROR_RE = re.compile(r"Failed to import Triton kernels")


def has_fatal_error(log_path: str | None, tail_bytes: int = 32768) -> bool:
    """True if the model's log already shows an unrecoverable engine error.
    Per-line so known-benign errors (_BENIGN_ERROR_RE) can be skipped without
    masking a real marker elsewhere in the tail."""
    if not log_path:
        return False
    try:
        size = os.path.getsize(log_path)
        with open(log_path, "rb") as f:
            if size > tail_bytes:
                f.seek(size - tail_bytes)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return False
    for line in text.splitlines():
        if _BENIGN_ERROR_RE.search(line):
            continue
        if any(marker in line for marker in _FATAL_MARKERS):
            return True
    return False


def _pids_on_port(port: int) -> "list[int]":
    """PIDs holding a LISTEN socket on `port`, via `ss`. Best-effort — [] on any error
    (ss missing, parse failure). The port is this endpoint's own window (assigned by
    the gateway's per-endpoint port packer), so a listener on it is ours to reap."""
    try:
        out = subprocess.run(
            ["ss", "-H", "-ltnp", f"sport = :{int(port)}"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:  # noqa: BLE001 — ss absent/slow → caller no-ops
        return []
    return sorted({int(m.group(1)) for m in re.finditer(r"pid=(\d+)", out)})


async def free_port(port: int, grace_s: float = 10.0) -> None:
    """Kill whatever is still LISTENing on `port` and wait for it to release it.

    An orphaned vLLM from a prior/racing launch (a different machine_id, or a rc=-9
    SIGKILL that left tp workers behind) keeps answering /health on this port AND
    hoards its GPU memory. Launching a fresh engine over it makes wait_health return
    a false-instant 200 (it hits the orphan) and the new tp launch OOMs against the
    VRAM the orphan holds. Reaping the port-holder first avoids both."""
    pids = await asyncio.to_thread(_pids_on_port, port)
    if not pids:
        return
    logger.warning("port %d still held by pid(s) %s — reaping before launch", port, pids)
    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)  # whole group (tp workers)
        except (ProcessLookupError, OSError):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not await asyncio.to_thread(_pids_on_port, port):
            return
        await asyncio.sleep(0.5)
    logger.warning("port %d still held after %.0fs — launching anyway", port, grace_s)


async def terminate(proc: asyncio.subprocess.Process | None, grace_s: float = 10.0) -> None:
    """Stop a vLLM engine and ALL its children (SIGTERM group, then SIGKILL).

    A tp>1 vLLM spawns one worker process per GPU; killing only the api_server
    orphans those workers, which keep hoarding GPU memory and OOM the next model
    that touches the same GPUs. We launch each engine in its own process group
    (start_new_session) and signal the whole group here so every worker dies."""
    if proc is None or proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        pgid = None

    def _signal(sig: int) -> None:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except (ProcessLookupError, OSError):
                pass
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass

    _signal(signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_s)
    except asyncio.TimeoutError:
        _signal(signal.SIGKILL)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass


async def wait_health(
    client: httpx.AsyncClient,
    member: MemberModel,
    proc: asyncio.subprocess.Process,
    log_path: str | None = None,
    timeout_s: float = LAUNCH_HEALTH_TIMEOUT_S,
) -> bool:
    """Poll /health until 200, or the process dies, or a fatal error shows up in
    the log, or we time out. The log check lets us fail fast (~seconds) instead
    of blocking the whole fleet startup for 900s when a GPU is unavailable."""
    deadline = time.monotonic() + timeout_s
    i = 0
    while time.monotonic() < deadline:
        if proc.returncode is not None:
            logger.error("vllm for %s exited during startup (rc=%s)", member.model, proc.returncode)
            return False
        if await vllm_ctl.is_healthy(client, member.base_url):
            logger.info("vllm ready: %s on port %d", member.model, member.port)
            return True
        # Cheap log peek every ~5s — abort early on an unrecoverable error.
        if i % 5 == 4 and has_fatal_error(log_path):
            logger.error("vllm for %s hit a fatal error during startup — aborting wait", member.model)
            return False
        i += 1
        await asyncio.sleep(1.0)
    logger.error("vllm for %s never became healthy within %.0fs", member.model, timeout_s)
    return False
