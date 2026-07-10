"""LLM try-it orchestrator (gemma-4) for the autotrain playground.

Pipeline, logged step-by-step (the gateway tails this script's stdout for the
"Load model" progress, then writes a ready marker once vLLM is serving):

  1. download the run's LoRA checkpoint (lora.pt + lora_meta.json) from S3
  2. merge the LoRA into the base model and save the merged model (merge_infer.py
     --merged-out, in a SUBPROCESS so its GPU memory is fully freed before vLLM)
  3. (merge_infer also writes preprocessor_config.json for the multimodal dir)
  4. ensure a dedicated vLLM venv exists (build once with uv if missing)
  5. `vllm serve <merged> --enforce-eager --tensor-parallel-size N` and poll /health

Runs under the TRAINING venv (needs boto3 + transformers 5.5.0 for the merge);
launches vLLM from a separate vLLM venv (`vllm_venv`). The merged model is cached
under `merged_dir` — a re-load reuses it and skips straight to serving.

Config (JSON via --config):
  model_s3, region, endpoint, access_key, secret_key   # S3 checkpoint prefix + creds
  base_model, merged_dir, work_dir, llm_dir, train_py   # merge inputs
  vllm_venv, port, tp, gpus, max_model_len, gpu_mem_util, served_model_name
  ready_file                                            # written when vLLM is healthy
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
import urllib.error


def log(msg: str) -> None:
    print(f"{msg}", flush=True)


def _run_stream(cmd, env=None, cwd=None) -> int:
    """Run a command, streaming its combined output to our stdout (the playground log)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env, cwd=cwd)
    for line in p.stdout:  # type: ignore[union-attr]
        print(line, end="", flush=True)
    p.wait()
    return p.returncode


def s3_download(spec: dict, s3_uri: str, dest_dir: str) -> None:
    """Download every object under s3://bucket/prefix into dest_dir (flat names)."""
    import boto3
    from botocore.client import Config as BotoConfig
    assert s3_uri.startswith("s3://"), s3_uri
    bucket, _, prefix = s3_uri[len("s3://"):].partition("/")
    prefix = prefix.rstrip("/") + "/"
    cli = boto3.client(
        "s3", region_name=spec.get("region") or "us-east-1",
        endpoint_url=spec.get("endpoint") or None,
        aws_access_key_id=spec.get("access_key") or None,
        aws_secret_access_key=spec.get("secret_key") or None,
        config=BotoConfig(signature_version="s3v4"),
    )
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
            log(f"[playground]   ↓ {rel} ({obj.get('Size', 0)/1e6:.1f} MB)")
            n += 1
    log(f"[playground] downloaded {n} file(s) from {s3_uri}")


def ensure_vllm_venv(venv: str, version: str = "0.23.0") -> str:
    """Create the dedicated vLLM venv (uv) if missing; return its python. Installs
    `vllm=={version}` (default 0.23.0). Reuses an existing venv when its vLLM already
    matches the requested version; reinstalls if a different version is pinned."""
    import shutil
    py = os.path.join(venv, "bin", "python")
    marker = os.path.join(venv, ".sgpu_llm_vllm")
    want = (version or "").strip()
    have_uv = shutil.which("uv") is not None
    pip = (["uv", "pip", "install", "--python", py] if have_uv else [py, "-m", "pip", "install"])

    def _imp(mod: str) -> bool:
        try:
            subprocess.check_call([py, "-c", f"import {mod}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def _vllm_ver() -> "str | None":
        try:
            out = subprocess.check_output(
                [py, "-c", "import vllm,sys;sys.stdout.write(getattr(vllm,'__version__',''))"],
                stderr=subprocess.DEVNULL).decode().strip()
            return out or None
        except Exception:
            return None

    def _pkg_ver(mod: str) -> "str | None":
        try:
            return subprocess.check_output(
                [py, "-c", f"import {mod},sys;sys.stdout.write(getattr({mod},'__version__',''))"],
                stderr=subprocess.DEVNULL).decode().strip() or None
        except Exception:
            return None

    def _ensure_build_tools() -> None:
        # vLLM's gemma-4 path JIT-compiles kernels (flashinfer/triton) at runtime →
        # needs `ninja` + `cmake` on PATH, else "[Errno 2] No such file: 'ninja'".
        if not (_imp("ninja") and _imp("cmake")):
            log("[playground] installing build tools (ninja, cmake) for runtime JIT kernels …")
            _run_stream([*pip, "ninja", "cmake"])
        # Mistral-Small-4 / Mistral3 need mistral_common>=1.11.0 for vLLM to resolve the
        # model — older versions fail with "No model architectures are specified" on the
        # inner LM. Idempotent: only (re)install when missing or < 1.11.
        mv = _pkg_ver("mistral_common")
        def _lt(v, lo=(1, 11)):
            try:
                return tuple(int(x) for x in v.split(".")[:2]) < lo
            except Exception:
                return True
        if mv is None or _lt(mv):
            log(f"[playground] installing mistral_common>=1.11.0 (was {mv}) for Mistral model support …")
            _run_stream([*pip, "-U", "mistral_common>=1.11.0"])

    cur = _vllm_ver() if os.path.exists(py) else None
    if cur and (not want or cur == want):
        _ensure_build_tools()
        log(f"[playground] vLLM venv ready: {venv} (vllm {cur})")
        return py
    if cur and want and cur != want:
        log(f"[playground] vLLM {cur} present but {want} requested — reinstalling …")
    if not os.path.exists(py):
        log(f"[playground] creating vLLM venv {venv} (uv) …")
        if have_uv:
            _run_stream(["uv", "venv", venv, "--python", "3.12"])
        else:
            _run_stream([sys.executable, "-m", "venv", venv])
    spec = f"vllm=={want}" if want else "vllm"
    log(f"[playground] installing {spec} (this is a one-time ~15-20 min build) …")
    # CUDA-13 host → let uv resolve the matching torch backend.
    rc = _run_stream([*pip, spec, "--torch-backend=auto"])
    if rc != 0:
        # retry without the torch-backend hint (older uv / non-uv path)
        _run_stream([*pip, spec])
    # 0.23.0 ships a prometheus instrumentator that 500s every route incl. /health.
    _run_stream([*pip, "-U", "prometheus-fastapi-instrumentator>=7"])
    _ensure_build_tools()
    if not _imp("vllm"):
        raise RuntimeError("vLLM did not import after install — see the install log above")
    open(marker, "w").write(want or "ok")
    log(f"[playground] vLLM venv ready: {venv} (vllm {want or 'latest'})")
    return py


def _arch(model_id: str) -> str:
    """gemma | qwen | minimax | mistral | nemotron (mirror of training_api._llm_arch)."""
    n = (model_id or "").lower()
    if "minimax" in n:
        return "minimax"
    if "mistral" in n:
        return "mistral"
    if "qwen" in n:
        return "qwen"
    if "nemotron" in n:
        return "nemotron"
    return "gemma"


def merge_lora_to_dir(base_model, lora, merged, train_py, llm_dir, dtype="fp16", env=None):
    """Per-arch merge-to-disk (shared by the try-it playground + the label export).
    gemma folds bf16 via merge_infer.py; qwen (bf16) via qwen/merge_to_disk.py;
    minimax/mistral dequant FP8→{dtype} via {arch}/merge_to_bf16.py. Each runs from its
    own dir (flat imports) with MODEL_ID set. Returns the subprocess rc (0 = ok)."""
    arch = _arch(base_model)
    menv = dict(env or os.environ)
    menv["MODEL_ID"] = base_model or ""
    if arch == "gemma":
        cmd = [train_py, "merge_infer.py", "--lora", lora, "--merged-out", merged, "--no-generate"]
        cwd = llm_dir
    elif arch == "nemotron":
        # bf16 fold (+ full embed/lm_head copy) via nemotron/merge_infer.py, like gemma.
        cmd = [train_py, "merge_infer.py", "--lora", lora, "--merged-out", merged,
               "--model-id", base_model, "--no-generate"]
        cwd = os.path.join(llm_dir, "nemotron")
    elif arch == "qwen":
        cmd = [train_py, "merge_to_disk.py", "--lora", lora, "--out", merged,
               "--dtype", dtype, "--model-id", base_model]
        cwd = os.path.join(llm_dir, "qwen")
    else:  # minimax / mistral (FP8 MoE)
        cmd = [train_py, "merge_to_bf16.py", "--lora", lora, "--out", merged,
               "--dtype", dtype, "--model-id", base_model]
        cwd = os.path.join(llm_dir, arch)
    log(f"[merge] arch={arch} → {merged} (dtype={dtype if arch != 'gemma' else 'bf16'})")
    return _run_stream(cmd, env=menv, cwd=cwd)


def ensure_processor_configs(base_model: str, merged: str) -> None:
    """Multimodal bases (gemma-4, Mistral3/Pixtral) need the image-processor config in
    the served dir or vLLM refuses to start; model.save_pretrained writes NONE of it.
    Copy whatever processor/preprocessor/chat-template JSONs the base repo has from the
    local HF cache (already downloaded for the merge — no token needed) into the merged
    dir. For a base like gemma-4 that has NO standalone preprocessor_config.json, derive
    it from processor_config.json's embedded `image_processor` block."""
    import glob
    import shutil
    repo = "models--" + base_model.replace("/", "--")
    roots = [os.environ.get("HF_HOME", ""), os.path.expanduser("~/.cache/huggingface"),
             "/share/huggingface", "/root/.cache/huggingface"]
    snap = None
    for r in roots:
        if not r:
            continue
        hits = glob.glob(os.path.join(r, "hub", repo, "snapshots", "*"))
        if hits:
            snap = hits[0]
            break
    copied = []
    for fn in ("processor_config.json", "preprocessor_config.json", "chat_template.json", "chat_template.jinja"):
        dst = os.path.join(merged, fn)
        if os.path.exists(dst):
            continue
        src = os.path.join(snap, fn) if snap else None
        if src and os.path.exists(src):
            shutil.copy(src, dst)
            copied.append(fn)
        elif not snap:
            try:
                from huggingface_hub import hf_hub_download
                shutil.copy(hf_hub_download(base_model, fn), dst)
                copied.append(fn)
            except Exception:  # noqa: BLE001 — optional file may not exist in the repo
                pass
    # gemma-4: no standalone preprocessor_config.json → derive from processor_config's
    # image_processor block.
    pc = os.path.join(merged, "processor_config.json")
    pp = os.path.join(merged, "preprocessor_config.json")
    if os.path.exists(pc) and not os.path.exists(pp):
        with open(pc) as f:
            pj = json.load(f)
        img = pj.get("image_processor")
        if isinstance(img, dict):
            with open(pp, "w") as f:
                json.dump(img, f, indent=2)
            copied.append("preprocessor_config.json(derived)")
    log(f"[playground] processor configs ensured: {copied or 'none needed'}")


def wait_health(port: int, timeout: int = 2400) -> bool:
    """Poll vLLM /health until it answers 200 (big MoE/dense cold loads are slow)."""
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))

    merged = cfg["merged_dir"]
    gpus = str(cfg.get("gpus") or "")
    tp = int(cfg.get("tp") or 1)
    port = int(cfg["port"])
    served = cfg.get("served_model_name") or "model"
    ready_file = cfg["ready_file"]
    base_model = cfg.get("base_model") or ""
    arch = _arch(base_model)
    # FP8 (minimax/mistral) + qwen merge to a plain checkpoint at this dtype (fp16 by
    # default — we don't infer in bf16); gemma folds in its base bf16 via merge_infer.
    merge_dtype = cfg.get("merge_dtype") or "fp16"

    base_env = dict(os.environ)
    if gpus:
        base_env["CUDA_VISIBLE_DEVICES"] = gpus

    # ---- step 1+2+3: download → merge → save (skip if merged dir is already built) ----
    if os.path.exists(os.path.join(merged, "config.json")):
        log(f"[playground] merged model already present at {merged} — reusing (skipping download + merge)")
    else:
        work = cfg["work_dir"]
        os.makedirs(work, exist_ok=True)
        log(f"[playground] step 1/5: downloading LoRA checkpoint from {cfg['model_s3']} …")
        s3_download(cfg, cfg["model_s3"], work)
        lora = os.path.join(work, "lora.pt")
        if not os.path.exists(lora):
            raise RuntimeError(f"lora.pt not found under {cfg['model_s3']} (got: {os.listdir(work)})")

        log(f"[playground] step 2/5: merging LoRA into {cfg['base_model']} "
            f"(arch={arch}, subprocess; GPUs={gpus or 'auto'}) …")
        rc = merge_lora_to_dir(base_model, lora, merged, cfg["train_py"], cfg["llm_dir"],
                               dtype=merge_dtype, env=base_env)
        if rc != 0:
            raise RuntimeError(f"merge for arch={arch} failed (rc={rc})")
        log(f"[playground] step 3/5: merged model saved to {merged}")

    # ---- ensure multimodal processor configs are in the merged dir (gemma + mistral) ----
    # gemma-4 and Mistral3/Pixtral are multimodal; vLLM refuses to start without the
    # image-processor config, which model.save_pretrained does NOT write. minimax/qwen
    # are text-only, so skip it there.
    if arch in ("gemma", "mistral") and not os.path.exists(os.path.join(merged, "preprocessor_config.json")):
        log(f"[playground] ensuring {arch} processor/preprocessor config (vLLM multimodal requirement) …")
        try:
            ensure_processor_configs(cfg["base_model"], merged)
            log(f"[playground] processor configs present: {sorted(f for f in os.listdir(merged) if 'process' in f or 'template' in f)}")
        except Exception as e:  # noqa: BLE001
            log(f"[playground] WARN: processor-config setup failed — vLLM may refuse to start: {e}")

    # ---- step 4: ensure the vLLM venv (pinned version) ----
    log(f"[playground] step 4/5: preparing vLLM venv {cfg['vllm_venv']} (vllm {cfg.get('vllm_version') or '0.23.0'}) …")
    vpy = ensure_vllm_venv(cfg["vllm_venv"], version=cfg.get("vllm_version") or "0.23.0")

    # ---- step 5: serve with vLLM (eager) ----
    log(f"[playground] step 5/5: launching vLLM (eager, TP={tp}, GPUs={gpus or 'auto'}, port={port}) on {merged} …")
    serve_env = dict(base_env)
    # Put the vLLM venv's bin on PATH so the runtime JIT (flashinfer/triton) finds the
    # `ninja` + `cmake` console scripts installed there — else FileNotFoundError: ninja.
    serve_env["PATH"] = os.path.join(cfg["vllm_venv"], "bin") + os.pathsep + serve_env.get("PATH", "")
    # User-supplied vLLM CLI args (verbatim), appended LAST so they can override the
    # soft defaults below (e.g. --max-model-len, --gpu-memory-utilization) and add flags
    # like --enable-auto-tool-choice / --tool-call-parser. The gateway already rejected
    # the reserved flags it controls (model/port/served-name/tp/pp).
    user_args = shlex.split(cfg.get("vllm_args") or "")
    if user_args:
        log(f"[playground] custom vLLM args: {user_args}")
    vbin = os.path.join(cfg["vllm_venv"], "bin", "vllm")
    cmd = [vbin, "serve", merged,
           "--enforce-eager",
           "--tensor-parallel-size", str(tp),
           "--port", str(port),
           "--served-model-name", served,
           "--max-model-len", str(int(cfg.get("max_model_len") or 16384)),
           "--gpu-memory-utilization", str(cfg.get("gpu_mem_util") or 0.90),
           "--trust-remote-code"] + user_args
    if not os.path.exists(vbin):
        cmd = [vpy, "-m", "vllm.entrypoints.openai.api_server", "--model", merged] + cmd[3:]
    log(f"[playground] $ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=serve_env)

    log(f"[playground] waiting for vLLM /health on port {port} (cold load can take several minutes) …")
    if wait_health(port):
        with open(ready_file, "w") as f:
            json.dump({"kind": "llm", "device": gpus or "auto", "port": port, "model": served}, f)
        log(f"[playground] ✅ vLLM is READY — serving '{served}' on port {port} (OpenAI /v1/chat/completions)")
    else:
        log("[playground] ❌ vLLM did not become healthy in time — see the log above")
    # Stay alive as the session leader so the gateway's pgid-stop kills vLLM too.
    proc.wait()
    log(f"[playground] vLLM exited (rc={proc.returncode})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        log(f"[playground] ERROR: {e}")
        log(traceback.format_exc())
        sys.exit(1)
