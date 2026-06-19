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
import subprocess
import sys
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

# Deps shared by both archs (on top of torch 2.12 + the FA3 wheel + transformers 5.5.0).
_COMMON_DEPS = ["mlflow", "psutil", "pynvml", "liger-kernel", "wandb",
                "numpy", "tqdm", "boto3", "huggingface_hub", "hf_transfer"]

# Per-architecture trainer wiring. `kernels` pin differs (transformers 5.5.0 import
# crashes outside each arch's range), so the two archs use SEPARATE venvs (the
# gateway picks /share/autotrain-llm-<arch> by default; see training_api).
_ARCH = {
    "gemma": {
        "trainer": "gemma4.py",                 # at LLM_DIR root
        "preflight": "test_attention.py",        # SDPA-mask + FA3 cu_seqlens (needs a GPU)
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
}


def log(msg: str) -> None:
    print(msg, flush=True)


def emit(tag: str, obj: dict) -> None:
    print(f"@@{tag} {json.dumps(obj)}", flush=True)


def detect_arch(model_id: str) -> str:
    """gemma | minimax from the base model id. Raises on anything else."""
    n = (model_id or "").lower()
    if "minimax" in n:
        return "minimax"
    if "gemma" in n:
        return "gemma"
    raise RuntimeError(
        f"unsupported LLM base model '{model_id}' — task_type=llm supports gemma-4 and "
        f"minimax-m2 models (the trainer is chosen by name)")


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


def _ensure_venv(cfg: dict, arch: str) -> str:
    """Create/reuse an isolated uv venv with the arch's training stack; return its
    python. Idempotent. torch 2.12 + the FA3 wheel match the host driver CUDA."""
    venv = _venv_path(cfg, arch)
    py = os.path.join(venv, "bin", "python")
    env = {**os.environ, "PIP_CONSTRAINT": "", "PIP_REQUIRE_HASHES": "0"}
    pkgs = list(_ARCH[arch]["deps"]) + list(_COMMON_DEPS)

    def _present() -> bool:
        try:
            subprocess.check_call(
                [py, "-c", "import torch, transformers, flash_attn_interface, liger_kernel, mlflow"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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

    cu = _pick_cuda_backend()
    log(f"[deps] {arch}: CUDA backend {cu}; torch=={TORCH_VERSION} + FlashAttention-3 + stack …")

    def _pip(*args: str) -> None:
        if have_uv:
            subprocess.check_call(["uv", "pip", "install", "--python", py, *args], env=env)
        else:
            subprocess.check_call([py, "-m", "pip", "install", "-q", *args], env=env)

    _pip(f"torch=={TORCH_VERSION}", "--index-url", f"https://download.pytorch.org/whl/{cu}")
    whl = f"flash_attn_3-3.0.0+{cu}torch2.12gite2743ab-cp39-abi3-linux_x86_64.whl"
    whl_path = os.path.join(venv, whl)
    if not os.path.exists(whl_path):
        url = f"{_FA3_BASE}/{FA3_TAG}/{whl}"
        log(f"[deps] downloading FA3 wheel {whl} …")
        urllib.request.urlretrieve(url, whl_path)
    _pip(whl_path)
    _pip(*pkgs)
    log(f"[deps] {arch} LLM venv ready: {py}")
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


def _lora_dims(cfg: dict) -> tuple[int, int]:
    """(r, alpha) from the form: alpha = lora_alpha (absolute) | lora_alpha_ratio*r | 2*r."""
    r = int(cfg.get("lora_r") or 16)
    if cfg.get("lora_alpha"):
        alpha = int(cfg["lora_alpha"])
    elif cfg.get("lora_alpha_ratio"):
        alpha = int(float(cfg["lora_alpha_ratio"]) * r)
    else:
        alpha = 2 * r
    return r, alpha


_MLP_TARGETS = {"gate_proj", "up_proj", "down_proj"}


def _gemma_cmd(py: str, cfg: dict, nproc: int) -> list[str]:
    r, alpha = _lora_dims(cfg)
    if not cfg.get("lora_r"):
        r, alpha = 256, 512  # gemma proven default (scaling 2.0)
    targets = [str(t) for t in (cfg.get("lora_target_modules") or []) if str(t).strip()] \
        or ["q_proj", "k_proj", "v_proj", "o_proj"]
    cmd = [
        py, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}",
        os.path.join(LLM_DIR, _ARCH["gemma"]["trainer"]),
        "--r", str(r), "--alpha", str(alpha),
        "--target_modules", ",".join(targets),
        "--batch_size", "1",  # the collator packs the whole batch into ONE sequence → must be 1
        "--lr", str(float(cfg.get("learning_rate") or 5e-5)),
        "--max_epochs", str(int(cfg.get("max_epochs") or 3)),
        "--max_steps", str(int(cfg.get("max_steps") or 0)),
        "--checkpointing_step", str(int(cfg.get("save_steps") or cfg.get("logging_steps") or 100)),
    ]
    return cmd


def _minimax_cmd(py: str, cfg: dict, nproc: int, packed: str, ckpt: str) -> list[str]:
    r, alpha = _lora_dims(cfg)  # form sends one r/alpha → use for BOTH attn + MoE
    cmd = [
        py, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}",
        os.path.join(LLM_DIR, _ARCH["minimax"]["trainer"]),
        "--attn_r", str(r), "--attn_alpha", str(float(alpha)),
        "--moe_r", str(r), "--moe_alpha", str(float(alpha)),
        "--batch_size", "1",
        "--lr", str(float(cfg.get("learning_rate") or 1e-5)),
        "--max_epochs", str(int(cfg.get("max_epochs") or 1)),
        "--max_steps", str(int(cfg.get("max_steps") or 0)),
        "--checkpointing_step", str(int(cfg.get("save_steps") or cfg.get("logging_steps") or 100)),
        "--data_dir", packed, "--out_dir", ckpt,
        # 230B base: stream weights from rank 0 into each shard (caps CPU at ~one
        # model copy instead of ~230GB × ranks). The default loads on every rank.
        "--low_cpu_shard_load",
    ]
    # minimax adapts attention (fixed q/k/v/o) + the MoE experts. Map the form's
    # LoRA-target picker: if the user selected NO MLP/dense target (gate/up/down),
    # adapt attention only (--no_moe_lora). Explicit cfg flag still wins.
    targets = set(str(t) for t in (cfg.get("lora_target_modules") or []))
    no_moe = bool(cfg.get("no_moe_lora")) or (bool(targets) and not (targets & _MLP_TARGETS))
    if no_moe:
        cmd.append("--no_moe_lora")
    return cmd


def run(cfg: dict) -> None:
    import tempfile
    global _RUN_WORKDIR

    model_id = cfg.get("base_model") or ""
    arch = detect_arch(model_id)
    spec = _ARCH[arch]

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
        # query-tiling block size that lets the 32k pack train (see gemma4 CLAUDE.md).
        env["SDPA_QUERY_BLOCK"] = str(cfg.get("sdpa_query_block") or 1024)
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if report_to and cfg.get("run_name"):
        env.setdefault("WANDB_NAME", cfg["run_name"])

    # Pre-fetch the base model once (ranks share the HF cache; gemma-4 is gated).
    log(f"[model] pre-fetching {model_id} into the HF cache …")
    subprocess.check_call(
        [py, "-c",
         "import sys; from huggingface_hub import snapshot_download; "
         "snapshot_download(sys.argv[1], ignore_patterns=['original/*','*.pth','*.gguf','consolidated*'])",
         model_id],
        env=env,
    )

    # Pre-flight: the arch's correctness test (the "verify the custom forward before
    # an expensive run" gate). Abort the run if it fails.
    pf_env = {**env, **spec["preflight_env"]}
    log(f"[preflight] {arch}: {spec['preflight']} …")
    rc = subprocess.call([py, spec["preflight"]], cwd=spec["preflight_cwd"], env=pf_env)
    if rc != 0:
        raise RuntimeError(f"{arch} correctness pre-flight ({spec['preflight']}) failed (rc={rc}) — refusing to train")

    nproc = _nproc(cfg)
    if arch == "gemma":
        cmd = _gemma_cmd(py, cfg, nproc)
    else:
        cmd = _minimax_cmd(py, cfg, nproc, packed, ckpt_dir)
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
    for line in p.stdout:  # type: ignore[union-attr]
        print(line, end="", flush=True)
        m = _STEP_RE.search(line)
        if m:
            try:
                last_loss = float(m.group(2))
                emit("STEP", {"step": int(m.group(1)), "loss": last_loss})
            except ValueError:
                pass
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
