"""Launch a single vLLM server for one member model, pinned to its GPUs and
sleep-mode enabled. Health is gated on vLLM's /health (200 = model loaded)."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import time

import httpx

from .config import MemberModel
from . import vllm_ctl

logger = logging.getLogger("worker-agent.launcher")

LAUNCH_HEALTH_TIMEOUT_S = 900.0  # cold model load can be minutes


def log_path_for(member: MemberModel, log_dir: str) -> str:
    """Per-model vLLM stdout/stderr file. Stable across launch + read + ship so
    the scheduler can read its own failure reason and the shipper can tail it."""
    slug = member.served_name.replace("/", "__")
    return os.path.join(log_dir, f"vllm-{slug}-{member.port}.log")


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
    # 5) Missing dependency in the venv.
    for line in lines:
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
) -> asyncio.subprocess.Process:
    """Spawn `vllm.entrypoints.openai.api_server` for one model. Stdout/stderr
    go to a per-model log file the shipper can tail. `python_exe` is the
    interpreter that has vLLM — a uv venv's bin/python when the endpoint set a
    venv_path, else bare `python3` on PATH."""
    os.makedirs(log_dir, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in member.gpu_indices)
    env["VLLM_SERVER_DEV_MODE"] = "1"  # exposes /sleep, /wake_up, /collective_rpc

    args = [
        python_exe, "-m", "vllm.entrypoints.openai.api_server",
        "--model", member.model,
        "--served-model-name", member.served_name,
        "--port", str(member.port),
        "--tensor-parallel-size", str(member.tp),
        # Pipeline parallel: split the model's layers across pp GPU groups. Total
        # GPUs used = tp * pp (the member's gpu_indices). Omitted when pp == 1.
        *(["--pipeline-parallel-size", str(member.pp)] if member.pp > 1 else []),
        "--enable-sleep-mode",
        *member.extra_args,
    ]
    log_path = log_path_for(member, log_dir)
    # Truncate (not append) per launch: each attempt gets a fresh log so the
    # health-wait fatal-error scan can't match a PREVIOUS attempt's traceback and
    # abort instantly. The log shipper detects the size reset and re-tails from 0.
    logf = open(log_path, "wb", buffering=0)
    logger.info(
        "launching vllm: model=%s gpus=%s tp=%d pp=%d port=%d → %s",
        member.model, env["CUDA_VISIBLE_DEVICES"], member.tp, member.pp, member.port, log_path,
    )
    # start_new_session=True puts the engine + its tp worker children in their
    # own process group so terminate() can SIGKILL the whole group (no orphans).
    proc = await asyncio.create_subprocess_exec(
        *args, env=env, stdout=logf, stderr=logf, start_new_session=True,
    )
    return proc


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
    cmd = ["uv", "pip", "install", "--python", py, f"vllm=={vllm_version}"]
    logger.info("ensuring vllm==%s in %s", vllm_version, venv_path)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            logger.warning("vllm ensure rc=%s: %s", proc.returncode, (out or b"").decode("utf-8", "replace")[-500:])
        else:
            logger.info("vllm==%s present in %s", vllm_version, venv_path)
    except FileNotFoundError:
        logger.warning("`uv` not found on PATH — cannot ensure vllm==%s; relying on venv as-is", vllm_version)
    except Exception:
        logger.exception("vllm ensure failed (continuing — launch will validate)")


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
    cmd = ["uv", "pip", "install", "--python", py, "librosa", "soundfile", "resampy", "av"]
    logger.info("ensuring audio deps (librosa, soundfile) in %s for a transcription model", venv_path)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            logger.warning("audio deps ensure rc=%s: %s", proc.returncode, (out or b"").decode("utf-8", "replace")[-500:])
        else:
            logger.info("audio deps present in %s", venv_path)
    except FileNotFoundError:
        logger.warning("`uv` not found on PATH — cannot ensure audio deps; whisper members will reject audio")
    except Exception:
        logger.exception("audio deps ensure failed (continuing — launch will validate)")


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


def has_fatal_error(log_path: str | None, tail_bytes: int = 32768) -> bool:
    """True if the model's log already shows an unrecoverable engine error."""
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
    return any(marker in text for marker in _FATAL_MARKERS)


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
