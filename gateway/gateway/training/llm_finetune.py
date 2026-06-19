#!/usr/bin/env python3
"""Standalone Gemma-4 LLM finetune orchestrator — shipped to a RunPod pod / VM by
the gateway's Autotrain runner (task_type=llm) and run over SSH.

It vendors the proven standalone trainer (sibling `llm/` dir, SFTP'd alongside this
file: `gemma4.py` + `attention.py` + the ChiniDataset package) and runs:
  1. download the pre-packed chat ChiniDataset (kind=llm_packed) from S3 → ./packed_data,
  2. download the gated base model into the HF cache (once, so the torchrun ranks
     don't race),
  3. `python test_attention.py` — the SDPA-mask + FA3-cu_seqlens correctness pre-flight
     (the project's "verify the custom forward before an expensive run" gate),
  4. `torchrun gemma4.py …` — FSDP2 LoRA finetune over the pack,
then upload the LoRA checkpoint (`checkpointing/lora.pt` + `lora_meta.json`) to S3
(+ optional HF push).

This is the chat/LLM analogue of `tts_finetune.py`. No gateway imports — config
arrives as one JSON file (--config). It emits @@STEP / @@METRIC / @@ARTIFACT /
@@DONE / @@ERROR + `[AUTOTRAIN_PROGRESS]`, parsed from the stream by the gateway.

The deps install mirrors `autotrain/gemma4/run.sh` (load-bearing version pinning):
torch 2.12 (the FA3-wheel ABI) for the host-driver CUDA + the FA3 prebuilt wheel +
transformers 5.5.0 + liger/kernels/peft/wandb/mlflow. ChiniDataset is VENDORED (the
`llm/` dir on PYTHONPATH), so no git/pip for it on the box.
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
LLM_DIR = os.path.join(THIS_DIR, "llm")  # gemma4.py + attention.py + chinidataset/
# gemma4.py logs e.g. "Epoch: 0, mb: 3, step: 3, loss: 1.234, tokens/s: 1300.00"
_STEP_RE = re.compile(r"step:\s*(\d+),\s*loss:\s*([0-9.eE+-]+)")
_RUN_WORKDIR = None

DEFAULT_VENV = "/share/autotrain-llm"
TORCH_VERSION = "2.12.0"   # MUST be 2.12.x — the FA3 prebuilt wheel is built for torch 2.12
FA3_TAG = "v0.9.18"
_FA3_BASE = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download"


def log(msg: str) -> None:
    print(msg, flush=True)


def emit(tag: str, obj: dict) -> None:
    print(f"@@{tag} {json.dumps(obj)}", flush=True)


# --------------------------------------------------------------------------
# deps: mirror autotrain/gemma4/run.sh (torch 2.12 + FA3 wheel for host CUDA)
# --------------------------------------------------------------------------
def _cuda_ge(a: str, b: str) -> bool:
    def parts(v):
        return [int(x) for x in re.findall(r"\d+", v)]
    return parts(a) >= parts(b)


def _pick_cuda_backend() -> str:
    """The FA3 wheel ships cu126 / cu130 / cu132; pick from the host driver's max
    CUDA (nvidia-smi), exactly like run.sh."""
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


# gemma4.py's stack. ChiniDataset is vendored (llm/ on PYTHONPATH) — NOT here.
def _base_deps() -> list[str]:
    return [
        "kernels==0.14.1",       # >=0.15.1 needs an explicit pin; keep 0.14.1
        "transformers==5.5.0",   # HF >= 5.0.0 for Gemma4ForConditionalGeneration
        "mlflow", "psutil", "pynvml",
        "liger-kernel",          # LigerFusedLinearCrossEntropyLoss (gemma4.py)
        "peft", "wandb", "numpy", "tqdm", "boto3",
        "huggingface_hub", "hf_transfer",
    ]


def _ensure_venv(cfg: dict) -> str:
    """Create/reuse an isolated uv venv with the gemma-4 stack and return its python.
    Idempotent. Pins torch 2.12 + the FA3 wheel matching the host driver CUDA."""
    venv = (cfg.get("venv_path") or DEFAULT_VENV).rstrip("/")
    py = os.path.join(venv, "bin", "python")
    env = {**os.environ, "PIP_CONSTRAINT": "", "PIP_REQUIRE_HASHES": "0"}
    pkgs = list(_base_deps())

    def _present() -> bool:
        try:
            subprocess.check_call(
                [py, "-c", "import torch, transformers, flash_attn_interface, liger_kernel, mlflow, peft"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    if os.path.exists(py) and _present():
        log(f"[deps] LLM venv ready: {py}")
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
    log(f"[deps] CUDA backend {cu}; installing torch=={TORCH_VERSION} + FlashAttention-3 + stack …")

    def _pip(*args: str) -> None:
        if have_uv:
            subprocess.check_call(["uv", "pip", "install", "--python", py, *args], env=env)
        else:
            subprocess.check_call([py, "-m", "pip", "install", "-q", *args], env=env)

    # torch 2.12 for the host CUDA (the FA3 wheel ABI is torch 2.12).
    _pip(f"torch=={TORCH_VERSION}", "--index-url", f"https://download.pytorch.org/whl/{cu}")
    # FlashAttention-3 prebuilt (Hopper; no JIT build).
    whl = f"flash_attn_3-3.0.0+{cu}torch2.12gite2743ab-cp39-abi3-linux_x86_64.whl"
    whl_path = os.path.join(venv, whl)
    if not os.path.exists(whl_path):
        url = f"{_FA3_BASE}/{FA3_TAG}/{whl}"
        log(f"[deps] downloading FA3 wheel {whl} …")
        urllib.request.urlretrieve(url, whl_path)
    _pip(whl_path)
    _pip(*pkgs)
    log(f"[deps] LLM venv ready: {py}")
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
    """Download every object under an s3://bucket/prefix into dest_dir."""
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
    """Upload every file under local_dir to s3://{bucket}/{key_prefix}/… → the prefix URI."""
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
        pct = int(i * 100 / max(1, n))
        log(f"[AUTOTRAIN_PROGRESS] step=uploading percent={pct}")
    return f"s3://{art['bucket']}/{base_key}/"


# --------------------------------------------------------------------------
def _nproc(cfg: dict) -> int:
    """GPUs visible to this process — the gateway pins CUDA_VISIBLE_DEVICES; else
    fall back to gpu_count."""
    cvd = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if cvd:
        return max(1, len([x for x in cvd.split(",") if x.strip()]))
    return max(1, int(cfg.get("gpu_count") or 1))


def run(cfg: dict) -> None:
    import tempfile
    global _RUN_WORKDIR

    _root = os.path.join((cfg.get("work_dir") or "/share").rstrip("/"), "checkpoint-llm")
    try:
        os.makedirs(_root, exist_ok=True)
        work = tempfile.mkdtemp(prefix="autotrain-llm-", dir=_root)
    except OSError:
        work = tempfile.mkdtemp(prefix="autotrain-llm-")
    _RUN_WORKDIR = work
    log(f"[trainer] work dir: {work}")

    model_id = cfg.get("base_model") or "google/gemma-4-31B-it"
    ds = cfg.get("dataset") or {}
    if (ds.get("kind") != "llm_packed") or not ds.get("packed_uri"):
        raise RuntimeError("LLM training needs a packed dataset (kind=llm_packed) — pack it first via 'Pack for LLM'")

    # LoRA knobs. gemma4.py wants an ABSOLUTE alpha; derive from lora_alpha, else
    # lora_alpha_ratio*r, else 2*r (the proven gemma-4 scaling=2.0).
    r = int(cfg.get("lora_r") or 256)
    if cfg.get("lora_alpha"):
        alpha = int(cfg["lora_alpha"])
    elif cfg.get("lora_alpha_ratio"):
        alpha = int(float(cfg["lora_alpha_ratio"]) * r)
    else:
        alpha = 2 * r
    # Which linear projections to LoRA. Default = attention q/k/v/o; the user can add
    # MLP/dense layers (gate_proj/up_proj/down_proj). gemma4.py warns for any target
    # that wraps nothing on this arch.
    targets = [str(t).strip() for t in (cfg.get("lora_target_modules") or
               ["q_proj", "k_proj", "v_proj", "o_proj"]) if str(t).strip()]
    lr = float(cfg.get("learning_rate") or 5e-5)
    max_epochs = int(cfg.get("max_epochs") or 3)
    max_steps = int(cfg.get("max_steps") or 0)
    ckpt_step = int(cfg.get("save_steps") or cfg.get("logging_steps") or 100)
    report_to = list((cfg.get("tracking") or {}).get("report_to") or [])

    # tracking env (gateway injected resolved secrets in cfg["tracking"]["env"]).
    base_env = {**os.environ}
    for k, v in ((cfg.get("tracking") or {}).get("env") or {}).items():
        if v not in (None, ""):
            base_env[k] = str(v)

    py = _ensure_venv(cfg)

    packed = os.path.join(work, "packed_data")
    _download_s3_prefix(ds, ds["packed_uri"], packed)
    if not os.path.exists(os.path.join(packed, "index.json")):
        raise RuntimeError(f"packed dataset has no index.json under {ds['packed_uri']} (not a ChiniDataset?)")

    # gemma4.py reads ./packed_data + writes ./checkpointing relative to CWD, and
    # imports `chinidataset` + `attention` from its own dir (PYTHONPATH=LLM_DIR).
    hf_token = cfg.get("hf_token") or base_env.get("HF_TOKEN")
    env = {
        **base_env,
        "PYTHONPATH": LLM_DIR + (os.pathsep + base_env["PYTHONPATH"] if base_env.get("PYTHONPATH") else ""),
        "GEMMA_MODEL_ID": model_id,
        # gemma-4 gotchas (see autotrain/gemma4/CLAUDE.md): NVLS off, fragmentation
        # reclaim, and the query-tiling block size that lets the 32k pack train.
        "NCCL_NVLS_ENABLE": base_env.get("NCCL_NVLS_ENABLE", "0"),
        "NCCL_CUMEM_ENABLE": base_env.get("NCCL_CUMEM_ENABLE", "0"),
        "PYTORCH_CUDA_ALLOC_CONF": base_env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        "SDPA_QUERY_BLOCK": str(cfg.get("sdpa_query_block") or 1024),
        "HF_HUB_ENABLE_HF_TRANSFER": base_env.get("HF_HUB_ENABLE_HF_TRANSFER", "1"),
    }
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if report_to and cfg.get("run_name"):
        env.setdefault("WANDB_NAME", cfg["run_name"])

    # Pre-fetch the (gated) base model once so the torchrun ranks share the HF
    # cache instead of racing the download.
    log(f"[model] pre-fetching {model_id} into the HF cache …")
    subprocess.check_call(
        [py, "-c",
         "import sys; from huggingface_hub import snapshot_download; "
         "snapshot_download(sys.argv[1], ignore_patterns=['original/*','*.pth','*.gguf','consolidated*'])",
         model_id],
        env=env,
    )

    # Pre-flight: the SDPA-mask + FA3 cu_seqlens correctness test (run.sh runs it
    # before every expensive job). Abort the run if the custom attention is wrong.
    log("[preflight] attention correctness test (SDPA mask + FA3 cu_seqlens) …")
    rc = subprocess.call([py, "test_attention.py"], cwd=LLM_DIR, env=env)
    if rc != 0:
        raise RuntimeError(f"attention correctness pre-flight failed (rc={rc}) — refusing to train")

    nproc = _nproc(cfg)
    gemma4 = os.path.join(LLM_DIR, "gemma4.py")
    cmd = [
        py, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}", gemma4,
        "--r", str(r), "--alpha", str(alpha),
        "--batch_size", "1",  # the collator packs the whole batch into ONE sequence → must be 1
        "--lr", str(lr), "--max_epochs", str(max_epochs), "--max_steps", str(max_steps),
        "--checkpointing_step", str(ckpt_step),
        "--target_modules", ",".join(targets),
    ]
    if "wandb" in report_to:
        cmd += ["--wandb"]
        if cfg.get("wandb_project"):
            cmd += ["--wandb_project", str(cfg["wandb_project"])]
    log(f"[gateway] $ (cwd={work}) {' '.join(cmd)}")
    log(f"[train] {nproc} GPU(s) · model {model_id} · r={r} alpha={alpha} lr={lr} "
        f"epochs={max_epochs} max_steps={max_steps or '∞'} · lora_targets={targets}")

    # Stream stdout, tee it, and re-emit @@STEP per logged training step so the
    # live loss curve updates (gemma4.py logs plain text, not @@ markers).
    p = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_loss = None
    for line in p.stdout:  # type: ignore[union-attr]
        print(line, end="", flush=True)
        m = _STEP_RE.search(line)
        if m:
            try:
                step_i = int(m.group(1))
                last_loss = float(m.group(2))
                emit("STEP", {"step": step_i, "loss": last_loss})
            except ValueError:
                pass
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)

    # Upload the LoRA checkpoint (lora.pt + lora_meta.json) to S3 (+ optional HF push).
    ckpt_dir = os.path.join(work, "checkpointing")
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
                  "epochs": max_epochs, "stopped_early": False})


def _cleanup_workdir(cfg: dict) -> None:
    if not cfg.get("cleanup_checkpoints", True) or not _RUN_WORKDIR:
        return
    shutil.rmtree(_RUN_WORKDIR, ignore_errors=True)
    log(f"[trainer] cleaned work dir: {_RUN_WORKDIR}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--deps-only", action="store_true",
                    help="install dependencies then exit")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    try:
        _ensure_venv(cfg)
        if a.deps_only:
            log("[deps] ready (deps-only)")
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
