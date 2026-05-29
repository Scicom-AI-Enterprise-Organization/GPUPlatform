"""Launch a single vLLM server for one member model, pinned to its GPUs and
sleep-mode enabled. Health is gated on vLLM's /health (200 = model loaded)."""
from __future__ import annotations

import asyncio
import logging
import os
import re
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
        "--enable-sleep-mode",
        *member.extra_args,
    ]
    log_path = log_path_for(member, log_dir)
    logf = open(log_path, "ab", buffering=0)
    logger.info(
        "launching vllm: model=%s gpus=%s tp=%d port=%d → %s",
        member.model, env["CUDA_VISIBLE_DEVICES"], member.tp, member.port, log_path,
    )
    proc = await asyncio.create_subprocess_exec(
        *args, env=env, stdout=logf, stderr=logf,
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


async def wait_health(
    client: httpx.AsyncClient,
    member: MemberModel,
    proc: asyncio.subprocess.Process,
    timeout_s: float = LAUNCH_HEALTH_TIMEOUT_S,
) -> bool:
    """Poll /health until 200, or the process dies, or we time out."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.returncode is not None:
            logger.error("vllm for %s exited during startup (rc=%s)", member.model, proc.returncode)
            return False
        if await vllm_ctl.is_healthy(client, member.base_url):
            logger.info("vllm ready: %s on port %d", member.model, member.port)
            return True
        await asyncio.sleep(1.0)
    logger.error("vllm for %s never became healthy within %.0fs", member.model, timeout_s)
    return False
