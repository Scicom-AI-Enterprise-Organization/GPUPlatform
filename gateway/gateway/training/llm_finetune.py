#!/usr/bin/env python3
"""Standalone LLM finetune orchestrator — shipped to a RunPod pod / VM by the
gateway's Autotrain runner (task_type=llm) and run over SSH.

**Architecture auto-detect.** From the base model id it picks the matching vendored
trainer (sibling `llm/` dir, SFTP'd alongside this file):
  - a **gemma** model  → `llm/gemma4.py`        (dense bf16, custom dual-head_dim attention)
  - a **minimax** model → `llm/minimax/minimax_m2.py` (230B FP8 MoE, QLoRA-style dequant LoRA)
Both consume the SAME packed chat ChiniDataset (kind=llm_packed) — `input_ids/labels/
position_ids/attention_mask` with cu_seqlens varlen packing — and both write a LoRA
checkpoint (`lora.pt` + `lora_meta.json`). The two stacks differ in deps (the `kernels`
pin), the correctness pre-flight, and the launch flags/env — encoded in `_ARCH` below.

The pipeline per run:
  1. download the packed dataset from S3 → ./packed_data,
  2. pre-fetch the base model into the HF cache (once; ranks share it),
  3. the arch's correctness pre-flight (gemma: SDPA/FA3 attention test; minimax: the
     fused-MoE LoRA grad test) — the project's "verify the custom forward before an
     expensive run" gate,
  4. `torchrun <trainer>` — FSDP2 LoRA finetune over the pack,
then upload the LoRA checkpoint to S3 (+ optional HF push).

No gateway imports — config arrives as one JSON file (--config). It emits @@STEP /
@@METRIC / @@ARTIFACT / @@DONE / @@ERROR + `[AUTOTRAIN_PROGRESS]`. The deps install
mirrors the standalone `run.sh`s (torch 2.12 = the FA3-wheel ABI; ChiniDataset is
VENDORED in `llm/`, so no git/pip for it on the box).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import sys
import time
import traceback
import urllib.request

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_DIR = os.path.join(THIS_DIR, "llm")  # gemma4.py + attention.py + minimax/ + chinidataset/
# Match both trainers' loss lines:
#   gemma4.py   "... step: 3, loss: 1.234, tokens/s: 1300.00"
#   minimax_m2.py "epoch 0 step 3 loss 1.2340 tok/s 12400"
_STEP_RE = re.compile(r"step:?\s*(\d+).*?loss:?\s*([0-9.eE+-]+)")
_RUN_WORKDIR = None

TORCH_VERSION = "2.12.0"   # MUST be 2.12.x — the FA3 prebuilt wheel ABI (+ torch._grouped_mm for MoE)
FA3_TAG = "v0.9.18"
_FA3_BASE = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download"

# FA4 (gemma, OPT-IN via cfg.gemma_fa4): the flash-attention-512 fork's cute kernel — symmetric
# head_dim=512 fwd+bwd on Hopper, so it handles BOTH gemma-4 head dims (512 global + 256 sliding),
# replacing the FA3 wheel + the SDPA-tiled 512 path → O(S) memory, long-context training. torch
# stays 2.12 (installed for both paths); only the attention kernel differs. Installed from the
# (PRIVATE) git fork's cute subdir (a `flash_attn.cute` namespace pkg); CuTeDSL JIT-compiles
# kernels at runtime. The cutlass-dsl/quack pins are load-bearing — the `>=` bounds in cute's
# pyproject pull too-new versions that break (quack 0.5 / cutlass-dsl 4.5; see gemma4 CLAUDE.md +
# run.sh). Overridable via cfg (fa4_fork_install / fa4_pins) if the fork URL/branch changes.
_FA4_FORK_INSTALL = "git+https://github.com/Scicom-AI-Enterprise-Organization/flash-attention-512.git#subdirectory=flash_attn/cute"
_FA4_PINS = ["nvidia-cutlass-dsl[cu13]==4.4.2", "quack-kernels==0.3.10"]


def _fa_mode(arch: str, cfg: dict) -> str:
    """Which attention stack to install/run: gemma DEFAULTS to 'fa4' (the head_dim-512
    cute fork — faster, long context) and can opt out with cfg.gemma_fa4=False (→ the
    FA3 wheel + dynamic_attention SDPA-tiled path). minimax/mistral always 'fa3' (stock
    FA, head_dim 128). qwen uses 'kernels' — the GatedDeltaNet hybrid runs the FlashQLA
    chunk_gated_delta_rule kernel + causal_conv1d for the full-attn layers, and
    attn_implementation="kernels-community/flash-attn3" auto-fetched by the `kernels`
    package at model load (no prebuilt FA3 wheel)."""
    if arch == "qwen":
        return "kernels"
    if arch == "gemma":
        return "fa4" if cfg.get("gemma_fa4", True) else "fa3"
    return "fa3"

# Deps shared by all archs (on top of the per-arch torch + attention stack + transformers 5.5.0).
_COMMON_DEPS = ["mlflow", "psutil", "pynvml", "liger-kernel", "wandb",
                "numpy", "tqdm", "boto3", "huggingface_hub", "hf_transfer"]

# Per-architecture trainer wiring. `kernels` pin differs (transformers 5.5.0 import
# crashes outside each arch's range), so the two archs use SEPARATE venvs (the
# gateway picks /share/autotrain-llm-<arch> by default; see training_api).
_ARCH = {
    "gemma": {
        "trainer": "gemma4.py",                 # at LLM_DIR root
        # Default backend = dynamic_attention (FA3 wheel + SDPA-tiled head_dim-512, ~32k ceiling),
        # matching the standalone run.sh's GEMMA_FA4=0 default. FA4 (head_dim-512 cute fork, long
        # context, faster) is opt-in via cfg.gemma_fa4 — see _fa_mode / _install_fa4. The FA3 path's
        # cheap kernel-level gate is test_attention.py; the FA4 gate (compare_logits_fa4.py, shipped
        # for manual use) loads the real model, so the FA4 path skips the every-run preflight.
        "preflight": "test_attention.py",        # FA3/SDPA kernel test (no model); skipped on the FA4 path
        "preflight_cwd": LLM_DIR,
        "preflight_env": {},
        "model_env": "GEMMA_MODEL_ID",
        "default_model": "google/gemma-4-31B-it",
        "deps": ["transformers==5.5.0", "kernels==0.14.1", "peft"],
    },
    "minimax": {
        "trainer": os.path.join("minimax", "minimax_m2.py"),
        "preflight": "test_lora.py",             # fused grouped-MoE LoRA math + grads (CPU; no GPU/model)
        "preflight_cwd": os.path.join(LLM_DIR, "minimax"),
        "preflight_env": {"MINIMAX_GROUPED_FALLBACK": "1"},
        "model_env": "MODEL_ID",
        "default_model": "MiniMaxAI/MiniMax-M2",
        # transformers 5.5.0 needs kernels in [0.12,0.13) or `import transformers`
        # itself crashes (mandatory LayerRepository version); see minimax CLAUDE.md.
        "deps": ["transformers==5.5.0", "kernels>=0.12.0,<0.13", "accelerate"],
    },
    "mistral": {
        "trainer": os.path.join("mistral", "mistral_small.py"),
        "preflight": "test_lora.py",             # per-tensor+block dequant + fused-MoE LoRA grads (CPU)
        "preflight_cwd": os.path.join(LLM_DIR, "mistral"),
        "preflight_env": {"MISTRAL_GROUPED_FALLBACK": "1"},
        "model_env": "MODEL_ID",
        "default_model": "mistralai/Mistral-Small-4-119B-2603",
        # same FP8-MoE stack as minimax (transformers 5.5.0 + kernels pin + accelerate).
        "deps": ["transformers==5.5.0", "kernels>=0.12.0,<0.13", "accelerate"],
        # the Triton per-tensor FP8 dequant fast path (bit-identical; 7-13x).
        "run_env": {"MISTRAL_DEQUANT_TRITON": "1"},
    },
    "qwen": {
        # Qwen3.5/3.6 (dense + MoE) — GatedDeltaNet hybrid. The trainer auto-detects dense
        # (Qwen3_5ForConditionalGeneration) vs MoE (Qwen3_5MoeForConditionalGeneration)
        # from the config, so this one entry serves Qwen/Qwen3.6-27B and Qwen/Qwen3.6-35B-A3B.
        "trainer": os.path.join("qwen", "qwen3_5.py"),
        # No cheap CPU pre-flight: the custom forward (LinearLoRA B=0 + the GatedDeltaNet
        # chunk_gated_delta_rule contiguous patch) needs the real 27B/35B model to check
        # (compare_logits.py is owed — see autotrain/qwen3.5/CLAUDE.md). The practical gate
        # is the smoke run (loss decreases, no NaN). Skipped like the gemma FA4 path.
        "preflight": None,
        "preflight_cwd": os.path.join(LLM_DIR, "qwen"),
        "preflight_env": {},
        "model_env": "MODEL_ID",
        "default_model": "Qwen/Qwen3.6-27B",
        # torch 2.10 + cu13 is the verified Qwen3.5 stack (NOT the FA3-wheel 2.12 the other
        # archs pin — qwen uses kernels-community/flash-attn3 for its full-attn layers, so it
        # doesn't need the 2.12 FA3-wheel ABI). transformers 5.12.1 ships qwen3_5 / qwen3_5_moe
        # + liger's apply_liger_kernel_to_qwen3_5(_moe). kernels<=0.14.0 for the attn fetch.
        "torch": "2.10.0",
        "torchvision": "0.25.0",
        "deps": ["transformers==5.12.1", "kernels<=0.14.0"],
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def emit(tag: str, obj: dict) -> None:
    print(f"@@{tag} {json.dumps(obj)}", flush=True)


def detect_arch(model_id: str) -> str:
    """gemma | minimax | mistral | qwen from the base model id. Raises on anything else."""
    n = (model_id or "").lower()
    if "minimax" in n:
        return "minimax"
    if "mistral" in n:
        return "mistral"
    if "qwen" in n:
        # Qwen3.5 / Qwen3.6 (dense + MoE) — the trainer auto-detects dense vs MoE from
        # the config, so one arch entry serves Qwen/Qwen3.6-27B and Qwen/Qwen3.6-35B-A3B.
        return "qwen"
    if "gemma" in n:
        return "gemma"
    raise RuntimeError(
        f"unsupported LLM base model '{model_id}' — task_type=llm supports gemma-4, "
        f"minimax-m2, mistral-small and qwen3.5/3.6 models (the trainer is chosen by name)")


def _venv_path(cfg: dict, arch: str) -> str:
    return (cfg.get("venv_path") or f"/share/autotrain-llm-{arch}").rstrip("/")


# --------------------------------------------------------------------------
# deps: mirror the standalone run.sh (torch 2.12 + FA3 wheel for host CUDA)
# --------------------------------------------------------------------------
def _cuda_ge(a: str, b: str) -> bool:
    def parts(v):
        return [int(x) for x in re.findall(r"\d+", v)]
    return parts(a) >= parts(b)


def _pick_cuda_backend() -> str:
    """The FA3 wheel ships cu126 / cu130 / cu132; pick from the host driver's max CUDA."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found — the LLM trainer needs a CUDA GPU box")
    m = re.search(r"CUDA Version:\s*([0-9.]+)", out)
    host = m.group(1) if m else ""
    log(f"[deps] host driver supports CUDA up to: {host or 'unknown'}")
    if host and _cuda_ge(host, "13.2"):
        return "cu132"
    if host and _cuda_ge(host, "13.0"):
        return "cu130"
    if host and _cuda_ge(host, "12.6"):
        return "cu126"
    raise RuntimeError(f"host driver CUDA '{host}' too old; need >=12.6 for the FA3 wheel")


def _install_fa3_wheel(cu: str, venv: str, _pip) -> None:
    """The FA3 prebuilt wheel for the host CUDA (head_dim 128 → minimax/mistral, and
    gemma's dynamic_attention fallback). torch 2.12 is installed separately (common)."""
    whl = f"flash_attn_3-3.0.0+{cu}torch2.12gite2743ab-cp39-abi3-linux_x86_64.whl"
    whl_path = os.path.join(venv, whl)
    if not os.path.exists(whl_path):
        url = f"{_FA3_BASE}/{FA3_TAG}/{whl}"
        log(f"[deps] downloading FA3 wheel {whl} …")
        urllib.request.urlretrieve(url, whl_path)
    _pip(whl_path)


def _install_qwen_kernels(py: str, env: dict, _pip) -> None:
    """Qwen3.5/3.6 GatedDeltaNet stack: the FlashQLA `chunk_gated_delta_rule` kernel
    (pure-python, from git) + `causal_conv1d` (a CUDA extension for the short conv).
    torch/torchvision are already installed (per-arch, cu13). The full-attn layers'
    attention (kernels-community/flash-attn3) is fetched by the `kernels` package at
    model load — no install step here. Mirrors autotrain/qwen3.5/run.sh.

    ⚠ causal_conv1d MUST build `--no-build-isolation` so it links the venv's cu13
    torch: under uv's default build isolation it pulls the default (cu12) torch into
    the build env and the resulting `causal_conv1d_cuda.so` links libcudart.so.12 →
    ImportError at runtime against the cu13 torch (the load-bearing gotcha from
    autotrain/qwen3.5/CLAUDE.md). Needs nvcc on PATH (CUDA_HOME, set in _ensure_venv)."""
    # Fast path: both kernels already import in this venv (e.g. a reused
    # /share/qwen3.5-venv) → skip the git fetch + the nvcc source build.
    try:
        subprocess.check_call(
            [py, "-c", "import flash_qla, causal_conv1d"],
            cwd="/tmp", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
        )
        log("[deps] qwen: flash_qla + causal_conv1d already present — skipping rebuild")
        return
    except Exception:
        pass
    # Build deps for the --no-build-isolation CUDA extension below.
    _pip("setuptools", "wheel", "packaging", "ninja")
    # FlashQLA: the GatedDeltaNet linear-attention kernel (pure python).
    _pip("git+https://github.com/QwenLM/FlashQLA.git")
    # causal_conv1d: CUDA extension, --no-build-isolation so it links the venv's cu13 torch.
    _pip("--no-build-isolation", "causal_conv1d")
    subprocess.check_call(
        [py, "-c", "import flash_qla, causal_conv1d; print('qwen kernels import OK')"],
        cwd="/tmp", env=env,
    )


def _install_fa4(cfg: dict, py: str, env: dict, _pip) -> None:
    """FlashAttention-4 fork (gemma head_dim-512). torch 2.12 is already installed
    (common); add the cute subdir from the public git fork + the load-bearing
    cutlass-dsl/quack pins (the cute pyproject's `>=` bounds pull too-new builds that
    break). Mirrors run.sh's GEMMA_FA4=1 path; verifies the import after install."""
    # Fast path: if the cute fork already imports in this venv, skip the (slow,
    # git-based) reinstall. torch may have been reinstalled around it, but the
    # `flash_attn.cute` namespace pkg + its cutlass/quack pins persist — so a venv
    # rebuild (e.g. after a partial install) needn't re-fetch flash-attention-512
    # over a slow/flaky uplink, which is exactly where the install hangs.
    try:
        subprocess.check_call(
            [py, "-c", "from flash_attn.cute.interface import flash_attn_varlen_func"],
            cwd="/tmp", env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        log("[deps] FA4 cute fork already present — skipping reinstall")
        return
    except Exception:
        pass
    spec = cfg.get("fa4_fork_install") or _FA4_FORK_INSTALL
    if spec.startswith("git+"):
        # Clone with SYSTEM git (shallow + abortable) then install from the local
        # checkout — uv's internal git hangs on a flaky github uplink (tm VM).
        # Spec form: git+<url>[@<ref>]#subdirectory=<sub>.
        rest = spec[len("git+"):]
        url, _, frag = rest.partition("#")
        subdir = ""
        for part in frag.split("&"):
            if part.startswith("subdirectory="):
                subdir = part[len("subdirectory="):]
        ref = None
        if url.endswith(".git@") is False and "@" in url.split("://", 1)[-1]:
            url, _, ref = url.rpartition("@")  # url@branch/tag (scheme has no '@')
        dest = os.path.join(tempfile.gettempdir(), "sgpu_fa4_cute_src")
        log("[deps] FlashAttention-4: shallow-cloning the cute fork via system git …")
        _git_clone_resilient(url, dest, env, ref)
        _pip(os.path.join(dest, subdir) if subdir else dest)
    else:
        log("[deps] FlashAttention-4 cute fork + cutlass/quack pins …")
        _pip(spec)
    _pip(*(cfg.get("fa4_pins") or _FA4_PINS))
    # Verify the kernel imports (CuTeDSL loads here; it JIT-compiles only at call time).
    # Run from a dir with NO local flash_attn/ so the namespace pkg resolves.
    subprocess.check_call(
        [py, "-c", "from flash_attn.cute.interface import flash_attn_varlen_func; print('FA4 cute import OK')"],
        cwd="/tmp", env=env,
    )


def _git_clone_resilient(url: str, dest: str, env: dict, ref: Optional[str] = None) -> None:
    """Shallow-clone `url` to `dest` with the SYSTEM git, resiliently. uv's internal
    git client ignores GIT_HTTP_LOW_SPEED and has no timeout/resume, so a `git+`
    install hangs forever on a flaky uplink (the tm VM → github case). System git
    honours GIT_HTTP_LOW_SPEED (aborts a stall) and `--depth 1` keeps it tiny; we
    also retry + hard-cap each attempt (SIGKILL a hang)."""
    attempts = 4
    per_attempt_s = 1200  # 20 min — a shallow clone is small; this only kills a true hang
    for i in range(attempts):
        if os.path.isdir(dest):
            shutil.rmtree(dest, ignore_errors=True)
        cmd = ["git", "clone", "--depth", "1", "--no-tags", "--single-branch"]
        if ref:
            cmd += ["--branch", ref]
        cmd += [url, dest]
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        try:
            rc = proc.wait(timeout=per_attempt_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait()
            rc = -1
            log(f"[deps] git clone attempt {i + 1}/{attempts} exceeded {per_attempt_s}s — killed.")
        if rc == 0:
            return
        if i == attempts - 1:
            raise subprocess.CalledProcessError(rc or 1, cmd)
        wait = 15 * (i + 1)
        log(f"[deps] git clone failed (attempt {i + 1}/{attempts}) — slow/flaky uplink; retrying in {wait}s …")
        time.sleep(wait)


def _clear_uv_git_locks(env: dict) -> None:
    """Remove stale uv `git+` lock files. A uv process killed mid git-fetch (e.g. a
    terminated run) leaves a lock in `<uv-cache>/git-v0/locks/`; later git installs
    then block on it for the lock timeout and fail. The holder is dead, so the lock
    is safe to drop. (Single-user box; runs are serialized per venv.)"""
    try:
        cache = subprocess.check_output(["uv", "cache", "dir"], text=True, env=env, timeout=20).strip()
    except Exception:
        cache = env.get("UV_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "uv")
    locks = os.path.join(cache, "git-v0", "locks")
    removed = 0
    try:
        for f in os.listdir(locks):
            try:
                os.remove(os.path.join(locks, f))
                removed += 1
            except OSError:
                pass
    except OSError:
        return
    if removed:
        log(f"[deps] cleared {removed} stale uv git lock(s) in {locks}")


def _ensure_venv(cfg: dict, arch: str) -> str:
    """Create/reuse an isolated uv venv with the arch's training stack; return its
    python. Idempotent. torch 2.12 is common; the attention kernel is per-mode:
    gemma defaults to the FA4 cute fork (head_dim-512), else the FA3 wheel."""
    venv = _venv_path(cfg, arch)
    py = os.path.join(venv, "bin", "python")
    env = {
        **os.environ, "PIP_CONSTRAINT": "", "PIP_REQUIRE_HASHES": "0",
        # Big CUDA wheels (torch + nvidia-*-cu13, incl. nvidia-nvshmem-cu13 from
        # pypi.nvidia.com) time out on a throttled/slow uplink (e.g. the tm VM's
        # INTL link). Give uv a long per-request timeout; _pip also retries the
        # whole command (and kills a true hang) on failure.
        "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "600"),
        # The FA4 cute fork is a git+ dependency; a stalled `git fetch` of
        # flash-attention-512 over a flaky uplink hangs forever (no HTTP timeout
        # covers it). Abort the fetch if it drops below 1 KB/s for 120s → the fetch
        # fails → _pip retries it.
        "GIT_HTTP_LOW_SPEED_LIMIT": os.environ.get("GIT_HTTP_LOW_SPEED_LIMIT", "1000"),
        "GIT_HTTP_LOW_SPEED_TIME": os.environ.get("GIT_HTTP_LOW_SPEED_TIME", "120"),
        # A uv killed mid `git+` fetch leaves a stale lock in <cache>/git-v0/locks/;
        # the next git install then waits the (default 300s) lock timeout and fails.
        # We proactively clear stale locks (below) AND fail fast if one is contended.
        "UV_LOCK_TIMEOUT": os.environ.get("UV_LOCK_TIMEOUT", "60"),
    }
    # qwen's causal_conv1d builds a CUDA extension from source → nvcc must be on PATH
    # and CUDA_HOME set (matches autotrain/qwen3.5/run.sh). No-op when the kernels are
    # already built (the skip-if-present fast path in _install_qwen_kernels).
    if arch == "qwen" and os.path.isdir("/usr/local/cuda/bin"):
        env.setdefault("CUDA_HOME", "/usr/local/cuda")
        env["PATH"] = "/usr/local/cuda/bin" + os.pathsep + env.get("PATH", "")
    pkgs = list(_ARCH[arch]["deps"]) + list(_COMMON_DEPS)
    fa = _fa_mode(arch, cfg)
    # FA4 installs as the `flash_attn.cute` namespace pkg; FA3 as `flash_attn_interface`;
    # qwen ('kernels') has no prebuilt attn wheel — verify its GatedDeltaNet kernels
    # (flash_qla + causal_conv1d) AND boto3 (a reused /share/qwen3.5-venv carries the
    # training stack but not the orchestrator's S3 deps → force a top-up install).
    if fa == "fa4":
        attn_import = "flash_attn.cute"
    elif fa == "kernels":
        attn_import = "flash_qla, causal_conv1d, boto3"
    else:
        attn_import = "flash_attn_interface"

    def _present() -> bool:
        try:
            subprocess.check_call(
                [py, "-c", f"import torch, transformers, liger_kernel, {attn_import}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
            )
            return True
        except Exception:
            return False

    if os.path.exists(py) and _present():
        log(f"[deps] {arch} LLM venv ready: {py}")
        return py

    have_uv = shutil.which("uv") is not None
    if not os.path.exists(py):
        log(f"[deps] creating venv {venv} …")
        if have_uv:
            subprocess.check_call(["uv", "venv", venv, "--python", "3.12"], env=env)
        else:
            subprocess.check_call([sys.executable, "-m", "venv", venv], env=env)
            subprocess.check_call([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], env=env)

    def _pip(*args: str) -> None:
        cmd = (
            ["uv", "pip", "install", "--python", py, *args] if have_uv
            else [py, "-m", "pip", "install", "-q", "--timeout", "600", "--retries", "5", *args]
        )
        # Retry the whole install, and KILL a true hang — the big CUDA wheels +
        # the FA4 git fetch come over a slow/flaky uplink (tm). A timed-out download
        # fails (→ retry); a stalled git fetch can hang with no output, so each
        # attempt gets a hard wall-clock cap (process-group killed on timeout → retry).
        attempts = 4
        per_attempt_s = 2400  # 40 min — covers a slow torch/cu13 download, kills a real hang
        for i in range(attempts):
            # Drop any stale git-fetch lock left by a killed uv (incl. a prior
            # attempt this loop SIGKILLed) so a `git+` install doesn't block on it.
            if have_uv:
                _clear_uv_git_locks(env)
            failed = False
            proc = subprocess.Popen(cmd, env=env, start_new_session=True)
            try:
                rc = proc.wait(timeout=per_attempt_s)
                failed = rc != 0
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.wait()
                failed = True
                log(f"[deps] install attempt {i + 1}/{attempts} exceeded {per_attempt_s}s "
                    f"(likely a stalled fetch) — killed.")
            if not failed:
                return
            if i == attempts - 1:
                raise subprocess.CalledProcessError(1, cmd)
            wait = 15 * (i + 1)
            log(f"[deps] install failed (attempt {i + 1}/{attempts}) — slow/flaky uplink; "
                f"retrying in {wait}s …")
            time.sleep(wait)

    # torch 2.12 is the common base (the FA3-wheel ABI + torch._grouped_mm for MoE; the
    # FA4 cute fork also runs on it). qwen overrides to torch 2.10 (its verified cu13
    # GatedDeltaNet stack — it uses kernels-community/flash-attn3, not the FA3 wheel, so
    # it needn't share the 2.12 ABI). The attention kernel then differs by mode.
    cu = _pick_cuda_backend()
    torch_ver = _ARCH[arch].get("torch", TORCH_VERSION)
    log(f"[deps] {arch} ({fa}): CUDA backend {cu}; torch=={torch_ver} …")
    _pip(f"torch=={torch_ver}", "--index-url", f"https://download.pytorch.org/whl/{cu}")
    torchvision_ver = _ARCH[arch].get("torchvision")
    if torchvision_ver:
        # qwen is multimodal (Qwen3_5ForConditionalGeneration ships vision blocks);
        # transformers pulls torchvision for the image processor. Match torch's cu13.
        _pip(f"torchvision=={torchvision_ver}", "--index-url", f"https://download.pytorch.org/whl/{cu}")
    if fa == "fa4":
        _install_fa4(cfg, py, env, _pip)
    elif fa == "kernels":
        _install_qwen_kernels(py, env, _pip)  # FlashQLA + causal_conv1d (torch already in place)
    else:
        _install_fa3_wheel(cu, venv, _pip)
    _pip(*pkgs)
    log(f"[deps] {arch} ({fa}) LLM venv ready: {py}")
    return py


# --------------------------------------------------------------------------
# S3 helpers (mirror tts_finetune)
# --------------------------------------------------------------------------
def _s3_client(spec: dict):
    import boto3
    from botocore.client import Config as BotoConfig
    return boto3.client(
        "s3", region_name=spec.get("region") or "us-east-1",
        endpoint_url=spec.get("endpoint") or None,
        aws_access_key_id=spec.get("access_key") or None,
        aws_secret_access_key=spec.get("secret_key") or None,
        config=BotoConfig(signature_version="s3v4"),
    )


def _download_s3_prefix(spec: dict, s3_uri: str, dest_dir: str) -> None:
    assert s3_uri.startswith("s3://"), s3_uri
    bucket, _, prefix = s3_uri[len("s3://"):].partition("/")
    prefix = prefix.rstrip("/") + "/"
    cli = _s3_client(spec)
    os.makedirs(dest_dir, exist_ok=True)
    n = 0
    for page in cli.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel:
                continue
            fp = os.path.join(dest_dir, rel)
            os.makedirs(os.path.dirname(fp) or dest_dir, exist_ok=True)
            cli.download_file(bucket, obj["Key"], fp)
            n += 1
    log(f"[data] downloaded {n} file(s) from {s3_uri} → {dest_dir}")


def _upload_s3_dir(art: dict, local_dir: str, key_prefix: str) -> str:
    cli = _s3_client(art)
    base_key = key_prefix.rstrip("/")
    uploads = []
    for root, _dirs, files in os.walk(local_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            uploads.append((fp, os.path.relpath(fp, local_dir)))
    n = len(uploads)
    log(f"[upload] uploading {n} file(s) → s3://{art['bucket']}/{base_key}/ …")
    log("[AUTOTRAIN_PROGRESS] step=uploading percent=0")
    for i, (fp, rel) in enumerate(uploads, 1):
        cli.upload_file(fp, art["bucket"], f"{base_key}/{rel}")
        log(f"[AUTOTRAIN_PROGRESS] step=uploading percent={int(i * 100 / max(1, n))}")
    return f"s3://{art['bucket']}/{base_key}/"


# --------------------------------------------------------------------------
def _nproc(cfg: dict) -> int:
    cvd = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if cvd:
        return max(1, len([x for x in cvd.split(",") if x.strip()]))
    return max(1, int(cfg.get("gpu_count") or 1))


def _master_port() -> int:
    """A per-run torchrun rendezvous port so two runs on the SAME box (e.g. a dense
    Qwen3.6-27B on GPUs 0-3 and a MoE Qwen3.6-35B-A3B on GPUs 4-7) don't both bind the
    default 29500 → `EADDRINUSE`. Derive it from the first pinned GPU index: disjoint
    GPU pins (the only way two runs coexist on one box) get disjoint ports. Concurrent
    runs on OVERLAPPING GPUs would collide on the GPUs first, so this is sufficient."""
    cvd = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if cvd:
        try:
            return 29500 + (int(cvd.split(",")[0]) % 100)
        except ValueError:
            pass
    return 29500


def _lora_dims(cfg: dict) -> tuple[int, int]:
    """(r, alpha) from the form. The form's LoRA-strength control is lora_alpha_RATIO
    (the UI shows "alpha = round(r × ratio)"), so it WINS — `lora_alpha` carries a
    non-null default (e.g. 32) that would otherwise clobber the ratio and collapse
    scaling (32/256 = 0.125 instead of the intended 2.0)."""
    r = int(cfg.get("lora_r") or 16)
    if cfg.get("lora_alpha_ratio"):
        alpha = int(float(cfg["lora_alpha_ratio"]) * r)
    elif cfg.get("lora_alpha"):
        alpha = int(cfg["lora_alpha"])
    else:
        alpha = 2 * r
    return r, alpha


_MLP_TARGETS = {"gate_proj", "up_proj", "down_proj"}


def _cpu_offload_on(cfg: dict, default: bool) -> bool:
    """Resolve the FSDP CPU-offload choice for a run: an explicit cfg value wins, else
    the per-arch default. gemma/qwen default ON (dense, they hit the VRAM wall at long
    context — turning it off there OOMs); the FP8-MoE trainers default OFF (they fit
    without it, and offload only slows them). The form's toggle sets cfg["cpu_offload"]."""
    v = cfg.get("cpu_offload")
    return default if v is None else bool(v)


def _gemma_cmd(py: str, cfg: dict, nproc: int) -> list[str]:
    r, alpha = _lora_dims(cfg)
    if not cfg.get("lora_r"):
        r, alpha = 256, 512  # gemma proven default (scaling 2.0)
    targets = [str(t) for t in (cfg.get("lora_target_modules") or []) if str(t).strip()] \
        or ["q_proj", "k_proj", "v_proj", "o_proj"]
    cmd = [
        py, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}",
        os.path.join(LLM_DIR, _ARCH["gemma"]["trainer"]),
        # `--lora_r` (not `--r`): torchrun prefix-matches a bare `--r` against its own
        # options and aborts "ambiguous option" on torch 2.12.0. gemma4.py aliases it.
        "--lora_r", str(r), "--alpha", str(alpha),
        "--target_modules", ",".join(targets),
        "--batch_size", str(int(cfg.get("batch_size") or 1)),  # packed bins concatenated per microbatch
        "--grad_accum", str(int(cfg.get("grad_accum") or 1)),  # microbatches accumulated per optimizer step
        "--lr", str(float(cfg.get("learning_rate") or 5e-5)),
        "--max_epochs", str(int(cfg.get("max_epochs") or 3)),
        "--max_steps", str(int(cfg.get("max_steps") or 0)),
        "--checkpointing_step", str(int(cfg.get("save_steps") or cfg.get("logging_steps") or 100)),
    ]
    if _cpu_offload_on(cfg, True):  # dense default: offload ON (long-context VRAM)
        cmd.append("--cpu_offload")
    return cmd


def _qwen_cmd(py: str, cfg: dict, nproc: int) -> list[str]:
    """Qwen3.5/3.6 trainer CLI (qwen/qwen3_5.py). Like gemma it reads ./packed_data
    and writes ./checkpointing (cwd=work); the model (dense Qwen3.6-27B vs MoE
    Qwen3.6-35B-A3B) is selected via the MODEL_ID env (spec['model_env']), auto-detected
    from the config. LoRA wraps only attention q/k/v/o (the MoE experts are untouched),
    so there's no --target_modules. `--lora_r` (long form) sidesteps the torchrun
    bare-`--r` "ambiguous option" clash on torch 2.12.x (the trainer aliases it)."""
    r, alpha = _lora_dims(cfg)
    if not cfg.get("lora_r"):
        r, alpha = 256, 512  # qwen proven default (scaling 2.0), matches the standalone
    cmd = [
        py, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}",
        os.path.join(LLM_DIR, _ARCH["qwen"]["trainer"]),
        "--lora_r", str(r), "--alpha", str(alpha),
        "--batch_size", str(int(cfg.get("batch_size") or 1)),  # packed bins concatenated per microbatch
        "--grad_accum", str(int(cfg.get("grad_accum") or 1)),  # microbatches accumulated per optimizer step
        "--lr", str(float(cfg.get("learning_rate") or 1e-4)),
        "--max_epochs", str(int(cfg.get("max_epochs") or 3)),
        "--max_steps", str(int(cfg.get("max_steps") or 0)),
        "--checkpointing_step", str(int(cfg.get("save_steps") or cfg.get("logging_steps") or 100)),
    ]
    if _cpu_offload_on(cfg, True):  # dense default: offload ON (long-context VRAM)
        cmd.append("--cpu_offload")
    return cmd


def _moe_cmd(py: str, cfg: dict, nproc: int, packed: str, ckpt: str, arch: str) -> list[str]:
    """The FP8-MoE trainers (minimax_m2.py / mistral_small.py) share a CLI: separate
    attention + MoE LoRA ranks, --data_dir/--out_dir, --low_cpu_shard_load. The form
    sends one r/alpha → used for both attn + MoE."""
    r, alpha = _lora_dims(cfg)
    cmd = [
        py, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}",
        os.path.join(LLM_DIR, _ARCH[arch]["trainer"]),
        "--attn_r", str(r), "--attn_alpha", str(float(alpha)),
        "--moe_r", str(r), "--moe_alpha", str(float(alpha)),
        "--batch_size", str(int(cfg.get("batch_size") or 1)),  # packed bins concatenated per microbatch
        "--grad_accum", str(int(cfg.get("grad_accum") or 1)),  # microbatches accumulated per optimizer step
        "--lr", str(float(cfg.get("learning_rate") or 1e-5)),
        "--max_epochs", str(int(cfg.get("max_epochs") or 1)),
        "--max_steps", str(int(cfg.get("max_steps") or 0)),
        "--checkpointing_step", str(int(cfg.get("save_steps") or cfg.get("logging_steps") or 100)),
        "--data_dir", packed, "--out_dir", ckpt,
        # Big FP8 base: stream weights from rank 0 into each shard (caps CPU at ~one
        # model copy instead of ~base × ranks). The default loads on every rank.
        "--low_cpu_shard_load",
    ]
    # These models adapt attention + the MoE experts by DEFAULT (the expert LoRA is
    # the whole point). Only skip the experts when the run EXPLICITLY asks
    # (cfg.no_moe_lora) — NOT inferred from the form's default q/k/v/o target list
    # (that has no MLP entry and would wrongly disable MoE on every run).
    if cfg.get("no_moe_lora"):
        cmd.append("--no_moe_lora")
    if _cpu_offload_on(cfg, False):  # MoE default: offload OFF (fits without it; faster)
        cmd.append("--cpu_offload")
    return cmd


def run(cfg: dict) -> None:
    import tempfile
    global _RUN_WORKDIR

    model_id = cfg.get("base_model") or ""
    arch = detect_arch(model_id)
    spec = _ARCH[arch]
    fa = _fa_mode(arch, cfg)  # gemma: fa4 (default) | fa3; minimax/mistral: fa3

    _root = os.path.join((cfg.get("work_dir") or "/share").rstrip("/"), "checkpoint-llm")
    try:
        os.makedirs(_root, exist_ok=True)
        work = tempfile.mkdtemp(prefix=f"autotrain-{arch}-", dir=_root)
    except OSError:
        work = tempfile.mkdtemp(prefix=f"autotrain-{arch}-")
    _RUN_WORKDIR = work
    log(f"[trainer] arch={arch} · model={model_id} · work dir: {work}")

    ds = cfg.get("dataset") or {}
    if (ds.get("kind") != "llm_packed") or not ds.get("packed_uri"):
        raise RuntimeError("LLM training needs a packed dataset (kind=llm_packed) — pack it first via 'Pack for LLM'")

    report_to = list((cfg.get("tracking") or {}).get("report_to") or [])
    base_env = {**os.environ}
    for k, v in ((cfg.get("tracking") or {}).get("env") or {}).items():
        if v not in (None, ""):
            base_env[k] = str(v)

    py = _ensure_venv(cfg, arch)

    packed = os.path.join(work, "packed_data")
    ckpt_dir = os.path.join(work, "checkpointing")
    _download_s3_prefix(ds, ds["packed_uri"], packed)
    if not os.path.exists(os.path.join(packed, "index.json")):
        raise RuntimeError(f"packed dataset has no index.json under {ds['packed_uri']} (not a ChiniDataset?)")

    hf_token = cfg.get("hf_token") or base_env.get("HF_TOKEN")
    env = {
        **base_env,
        "PYTHONPATH": LLM_DIR + (os.pathsep + base_env["PYTHONPATH"] if base_env.get("PYTHONPATH") else ""),
        spec["model_env"]: model_id,
        "NCCL_NVLS_ENABLE": base_env.get("NCCL_NVLS_ENABLE", "0"),
        "NCCL_CUMEM_ENABLE": base_env.get("NCCL_CUMEM_ENABLE", "0"),
        "PYTORCH_CUDA_ALLOC_CONF": base_env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        "HF_HUB_ENABLE_HF_TRANSFER": base_env.get("HF_HUB_ENABLE_HF_TRANSFER", "1"),
    }
    if arch == "gemma":
        # gemma4.py picks the registered backend from GEMMA_ATTN (set it explicitly so the
        # script default doesn't decide). FA4 = the cute head_dim-512 kernel (default, faster,
        # long context); fa3 = dynamic_attention (SDPA-tiled 512 + FA3 sliding).
        if fa == "fa4":
            env["GEMMA_ATTN"] = "fa4_attention"
            # cache the CuTeDSL JIT kernels across steps/runs + reduce allocator fragmentation.
            env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = base_env.get("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
            env["PYTORCH_ALLOC_CONF"] = base_env.get("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        else:
            env["GEMMA_ATTN"] = "dynamic_attention"
            # query-tiling block size that lets the 32k pack train (see gemma4 CLAUDE.md).
            env["SDPA_QUERY_BLOCK"] = str(cfg.get("sdpa_query_block") or 1024)
    # arch-specific run env (e.g. mistral's Triton FP8 dequant fast path).
    for k, v in (spec.get("run_env") or {}).items():
        env.setdefault(k, str(v))
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if report_to and cfg.get("run_name"):
        env.setdefault("WANDB_NAME", cfg["run_name"])

    # Pre-fetch the base model once (ranks share the HF cache; gemma-4 is gated).
    # Use an ALLOW-list (HF-format `model-*.safetensors` shards + all configs/tokenizer),
    # NOT ignore_patterns: some repos (Mistral-Small-4) ship BOTH HF-format `model-*`
    # AND a mistral-common `consolidated-*` copy, and `ignore_patterns=[consolidated*]`
    # resolved to 0 files on the box's huggingface_hub (downloaded nothing). The allow
    # list grabs exactly the HF weights transformers loads + skips the consolidated /
    # original / gguf / pth dups. Covers gemma / minimax / mistral (all model-* sharded).
    log(f"[model] pre-fetching {model_id} into the HF cache …")
    subprocess.check_call(
        [py, "-c",
         "import sys; from huggingface_hub import snapshot_download; "
         "snapshot_download(sys.argv[1], allow_patterns=['*.json','*.jinja','*.txt','*.model',"
         "'model-*.safetensors','model.safetensors'])",
         model_id],
        env=env,
    )

    # Pre-flight: the arch's correctness test (the "verify the custom forward before an
    # expensive run" gate). Abort the run if it fails. The FA4 gemma path has no cheap
    # unit-test gate (test_attention.py only exercises the FA3/SDPA path; the real FA4
    # gate compare_logits_fa4.py loads the 31B model), so it's skipped — the cute kernel
    # is pre-validated (cosine 0.9998) and shipped for a manual compare_logits_fa4.py run.
    preflight = spec.get("preflight")
    if arch == "gemma" and fa == "fa4":
        preflight = None
    if preflight:
        pf_env = {**env, **spec["preflight_env"]}
        log(f"[preflight] {arch} ({fa}): {preflight} …")
        rc = subprocess.call([py, preflight], cwd=spec["preflight_cwd"], env=pf_env)
        if rc != 0:
            raise RuntimeError(f"{arch} correctness pre-flight ({preflight}) failed (rc={rc}) — refusing to train")
    elif arch == "qwen":
        log(f"[preflight] {arch} ({fa}): no cheap pre-flight — the custom forward "
            f"(LinearLoRA B=0 + GatedDeltaNet contiguous patch) needs the real model; "
            f"compare_logits.py is owed (see autotrain/qwen3.5/CLAUDE.md). Gate = the smoke run.")
    else:
        log(f"[preflight] {arch} ({fa}): no cheap pre-flight (FA4 cute kernel pre-validated; "
            f"run compare_logits_fa4.py manually for a full check)")

    nproc = _nproc(cfg)
    if arch == "gemma":
        cmd = _gemma_cmd(py, cfg, nproc)
    elif arch == "qwen":
        cmd = _qwen_cmd(py, cfg, nproc)  # reads ./packed_data, writes ./checkpointing (like gemma)
    else:  # minimax + mistral share the FP8-MoE CLI
        cmd = _moe_cmd(py, cfg, nproc, packed, ckpt_dir, arch)
    # Unique torchrun rendezvous port per run so two runs on one box don't collide on
    # 29500 (EADDRINUSE). Injected as a torchrun optional right after --nproc_per_node,
    # before the trainer-script positional. Applies to every arch.
    cmd = cmd[:4] + [f"--master_port={_master_port()}"] + cmd[4:]
    if "wandb" in report_to:
        cmd += ["--wandb"]
        if cfg.get("wandb_project"):
            cmd += ["--wandb_project", str(cfg["wandb_project"])]

    log(f"[gateway] $ (cwd={work}) {' '.join(cmd)}")
    log(f"[train] {arch} · {nproc} GPU(s) · lr={cfg.get('learning_rate')} "
        f"epochs={cfg.get('max_epochs')} max_steps={cfg.get('max_steps') or '∞'}")

    p = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_loss = None
    # Gradient accumulation makes a trainer log grad_accum lines per OPTIMIZER step, all
    # sharing the same step number. Collapse them HERE — once, for EVERY arch (current
    # and future) — into a single @@STEP per step, loss = mean over the accumulation
    # window. So the loss curve gets one point per optimizer step without any per-trainer
    # logic: a new model just needs to log the standard "step: N … loss: L" the parser
    # already keys on. A step's point is emitted when the step advances (+ a final flush).
    _acc = {"step": None, "sum": 0.0, "n": 0}

    def _flush_step() -> None:
        nonlocal last_loss
        if _acc["step"] is not None and _acc["n"] > 0:
            last_loss = _acc["sum"] / _acc["n"]
            emit("STEP", {"step": _acc["step"], "loss": last_loss})
        _acc["sum"] = 0.0
        _acc["n"] = 0

    for line in p.stdout:  # type: ignore[union-attr]
        print(line, end="", flush=True)
        m = _STEP_RE.search(line)
        if m:
            try:
                step_i = int(m.group(1))
                loss_f = float(m.group(2))
            except ValueError:
                continue
            if _acc["step"] is not None and step_i != _acc["step"]:
                _flush_step()  # step advanced → emit the previous step's mean
            _acc["step"] = step_i
            _acc["sum"] += loss_f
            _acc["n"] += 1
    _flush_step()  # emit the final step
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)

    if not os.path.isdir(ckpt_dir) or not os.listdir(ckpt_dir):
        raise RuntimeError("training finished but no checkpoint was written (checkpointing/ empty)")
    art = cfg.get("artifacts") or {}
    s3_uri = None
    if art.get("bucket"):
        s3_uri = _upload_s3_dir(art, ckpt_dir, art["prefix"].rstrip("/") + "/checkpoint")
        log(f"[upload] LoRA checkpoint → {s3_uri}")

    hf_repo = None
    if cfg.get("hf_push_repo") and hf_token:
        try:
            from huggingface_hub import HfApi
            log(f"[upload] pushing LoRA checkpoint to Hugging Face → {cfg['hf_push_repo']} …")
            HfApi().upload_folder(folder_path=ckpt_dir, repo_id=cfg["hf_push_repo"],
                                  repo_type="model", token=hf_token)
            hf_repo = cfg["hf_push_repo"]
            log(f"[upload] pushed → https://huggingface.co/{hf_repo}")
        except Exception as e:  # noqa: BLE001
            log(f"[upload] HF push failed: {e}")

    emit("ARTIFACT", {"s3_uri": s3_uri, "hf_repo": hf_repo})
    emit("DONE", {"best": ({"loss": last_loss} if last_loss is not None else None),
                  "epochs": int(cfg.get("max_epochs") or 1), "stopped_early": False})


def _cleanup_workdir(cfg: dict) -> None:
    if not cfg.get("cleanup_checkpoints", True) or not _RUN_WORKDIR:
        return
    shutil.rmtree(_RUN_WORKDIR, ignore_errors=True)
    log(f"[trainer] cleaned work dir: {_RUN_WORKDIR}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--deps-only", action="store_true", help="install dependencies then exit")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    try:
        arch = detect_arch(cfg.get("base_model") or "")
        _ensure_venv(cfg, arch)
        if a.deps_only:
            log(f"[deps] ready (deps-only, arch={arch})")
            return 0
        run(cfg)
        return 0
    except Exception as e:  # noqa: BLE001
        emit("ERROR", {"message": str(e)})
        log(traceback.format_exc())
        return 1
    finally:
        _cleanup_workdir(cfg)


if __name__ == "__main__":
    sys.exit(main())
