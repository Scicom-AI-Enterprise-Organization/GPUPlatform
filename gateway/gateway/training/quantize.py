#!/usr/bin/env python3
"""Standalone LLM quantization worker — shipped to a RunPod pod / VM by the
gateway's Quantization runner (quantization_api.py) and executed over SSH. It has
NO gateway imports; everything it needs arrives in a single JSON config file (path
passed via --config). It installs its own deps into an isolated uv venv, pulls a
base model from HuggingFace, quantizes it with llm-compressor (compressed-tensors),
uploads the compressed model to S3, and optionally pushes it to the HF Hub.

Contract with the gateway (parsed from stdout):
  @@PROGRESS {json}  free-form stage markers: {"stage": str, "percent": float}
  @@SIZES {json}     {"source_gb": float, "quantized_gb": float}
  @@ARTIFACT {json}  after upload: {"s3_uri": str, "hf_repo": str|null}
  @@DONE {json}      final: {"scheme": str}
  @@ERROR {json}     fatal: {"message": str}
Every other line is free-form progress and streamed to the job's log.

Two phases (mirrors the autotrain trainers):
  --deps-only : system python, build/reuse the uv venv, then exit.
  (run)       : gateway launches {venv}/bin/python quantize.py --config …; the heavy
                imports resolve from the venv (done lazily inside run()).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback


def log(msg: str) -> None:
    print(msg, flush=True)


def emit(tag: str, obj: dict) -> None:
    print(f"@@{tag} {json.dumps(obj)}", flush=True)


# --------------------------------------------------------------------------
# Quantization schemes — keep in sync with quantization_api._SCHEMES.
# Each entry: (needs_calibration, recipe-builder name). The builder is resolved in
# _build_recipe() lazily (llm-compressor lives in the venv, not system python).
# --------------------------------------------------------------------------
QUANT_SCHEMES = {
    "fp8-dynamic": False,
    "w4a16": True,
    "w8a8-int8": True,
    "fp8": True,
    "nvfp4": True,
    "awq": True,
}


# --------------------------------------------------------------------------
# Dependency bootstrap — an isolated uv venv with the llm-compressor stack.
# --------------------------------------------------------------------------
DEFAULT_VENV = "/share/quant-llmcompressor"


def _ensure_venv(cfg: dict) -> str:
    """Create/reuse an isolated uv venv with the llm-compressor stack; return its
    python. Idempotent — fast when the venv is ready."""
    import shutil

    venv = (cfg.get("venv_path") or DEFAULT_VENV).rstrip("/")
    py = os.path.join(venv, "bin", "python")
    env = {**os.environ, "PIP_CONSTRAINT": "", "PIP_REQUIRE_HASHES": "0"}
    # llmcompressor pulls compressed-tensors + a compatible transformers/torch. We
    # don't pin torch to a CUDA index here: the pod image already carries a matching
    # CUDA torch (cu1300 default), and llmcompressor's torch dep is broad.
    pkgs = [
        "llmcompressor>=0.6", "compressed-tensors", "transformers>=4.48",
        "datasets>=2.20", "accelerate>=0.30", "boto3", "huggingface_hub",
    ]
    check_mods = ["torch", "transformers", "datasets", "llmcompressor", "compressed_tensors", "boto3"]

    def _present() -> bool:
        probe = "import " + ", ".join(check_mods) + "\n"
        try:
            subprocess.check_call([py, "-c", probe],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    if os.path.exists(py) and _present():
        log(f"[deps] llm-compressor venv ready: {py}")
        return py
    have_uv = shutil.which("uv") is not None

    def _pip(*args):
        if have_uv:
            subprocess.check_call(["uv", "pip", "install", "--python", py, *args], env=env)
        else:
            subprocess.check_call([py, "-m", "pip", "install", "-q", *args], env=env)

    if not os.path.exists(py):
        log(f"[deps] creating venv {venv} …")
        if have_uv:
            subprocess.check_call(["uv", "venv", venv, "--python", "3.12"], env=env)
        else:
            subprocess.check_call([sys.executable, "-m", "venv", venv], env=env)
            subprocess.check_call([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], env=env)
    log(f"[deps] installing llm-compressor stack into {venv} …")
    _pip(*pkgs)
    log(f"[deps] llm-compressor venv ready: {py}")
    return py


# --------------------------------------------------------------------------
# Calibration dataset → list[str] of text samples.
# --------------------------------------------------------------------------
def _s3_client(ds: dict):
    import boto3
    from botocore.client import Config as BotoConfig

    return boto3.client(
        "s3",
        region_name=ds.get("region") or "us-east-1",
        endpoint_url=ds.get("endpoint") or None,
        aws_access_key_id=ds.get("access_key") or None,
        aws_secret_access_key=ds.get("secret_key") or None,
        config=BotoConfig(signature_version="s3v4"),
    )


def _read_s3_metadata_rows(ds: dict) -> list[dict]:
    import csv
    import io

    cli = _s3_client(ds)
    body = cli.get_object(Bucket=ds["bucket"], Key=ds["metadata_key"])["Body"].read()
    text = body.decode("utf-8", errors="replace")
    fmt = (ds.get("format") or "").lower()
    if not fmt:
        key = ds["metadata_key"]
        fmt = "jsonl" if key.endswith(".jsonl") else ("json" if key.endswith(".json") else "csv")
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(text)))
    if fmt == "jsonl":
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("data", data.get("rows", []))


def _guess_text_field(row: dict) -> str | None:
    for cand in ("text", "content", "prompt", "instruction", "question", "input", "sentence"):
        if cand in row and isinstance(row[cand], str):
            return cand
    for k, v in row.items():
        if isinstance(v, str) and v.strip():
            return k
    return None


def _messages_to_text(tokenizer, messages) -> str | None:
    """Render an OpenAI-style [{role,content}] list to a training string via the
    tokenizer's chat template (falls back to concatenation)."""
    if not isinstance(messages, list) or not messages:
        return None
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))


def _load_calibration_texts(cfg: dict, tokenizer) -> list[str]:
    """Return up to num_calibration_samples raw text strings from the calibration
    dataset spec (resolved by the gateway from a Datasets resource)."""
    ds = cfg.get("dataset") or {}
    n = int(cfg.get("num_calibration_samples") or 512)
    text_field = cfg.get("calib_text_field")
    msg_field = cfg.get("calib_messages_field")
    kind = ds.get("kind")
    texts: list[str] = []

    if kind in ("hf", "llm"):
        from datasets import load_dataset

        repo = ds.get("hf_repo")
        token = ds.get("hf_token") or None
        log(f"[calib] loading HF dataset {repo} …")
        # Stream so we don't pull a huge dataset to grab a few hundred rows.
        try:
            data = load_dataset(repo, split="train", streaming=True, token=token)
        except Exception:
            data = load_dataset(repo, split="train", token=token)
        for i, row in enumerate(data):
            if len(texts) >= n:
                break
            t = None
            mf = msg_field or ds.get("messages_field") or ("messages" if "messages" in row else None)
            if mf and mf in row:
                t = _messages_to_text(tokenizer, row[mf])
            if t is None:
                # The dataset's transcription_field is a platform default ("transcription")
                # that often doesn't exist on a plain text repo — fall back to guessing
                # rather than silently skipping every row.
                tf = text_field or ds.get("transcription_field")
                if not tf or tf not in row:
                    tf = _guess_text_field(row)
                if tf and tf in row and isinstance(row[tf], str):
                    t = row[tf]
            if t and t.strip():
                texts.append(t)
    elif kind in ("s3", "upload"):
        log("[calib] loading S3 metadata rows …")
        rows = _read_s3_metadata_rows(ds)
        for row in rows:
            if len(texts) >= n:
                break
            tf = text_field or ds.get("transcription_field")
            if not tf or tf not in row:
                tf = _guess_text_field(row)
            if tf and tf in row and isinstance(row[tf], str) and row[tf].strip():
                texts.append(row[tf])
    else:
        raise RuntimeError(
            f"calibration dataset kind '{kind}' is not supported — use a text dataset "
            f"(HuggingFace, LLM chat, or an uploaded/S3 text file)"
        )

    if not texts:
        raise RuntimeError("no calibration text found — check the dataset's text/messages field")
    log(f"[calib] collected {len(texts)} calibration samples")
    return texts


def _build_calibration_dataset(cfg: dict, tokenizer):
    """Tokenize calibration texts into a HF Dataset llm-compressor's oneshot() can
    consume (columns: input_ids, attention_mask)."""
    from datasets import Dataset as HFDataset

    texts = _load_calibration_texts(cfg, tokenizer)
    max_len = int(cfg.get("max_seq_length") or 2048)

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    ds = HFDataset.from_dict({"text": texts})
    ds = ds.map(_tok, batched=True, remove_columns=["text"])
    return ds


# Multimodal (vision / audio) models: the modality towers + input embedders MUST stay
# in full precision. Two failure modes seen on real gemma-4:
#   1. an FP8 vision tower dies in the forward with "RuntimeError: Not yet supported
#      ScalarType 46";
#   2. vLLM's gemma-4 impl builds the vision/audio embedders as plain (unquantized)
#      Linear, so a quantized `embed_audio`/`embed_vision` carries an unexpected
#      `weight_scale` and load fails with "no module or parameter named
#      '…embedding_projection.weight_scale'".
# So for a detected VLM/omni model we union these into the recipe `ignore`. They're
# regexes → they no-op on a model that lacks the matching modules. Covers the common
# gemma / qwen / llava / *_unified naming for vision AND audio.
_MULTIMODAL_IGNORE = [
    "re:.*vision_tower.*",
    "re:.*vision_model.*",
    "re:.*vision_embedder.*",       # gemma-4 unified
    "re:.*visual.*",                # qwen-VL
    "re:.*audio_tower.*",
    "re:.*audio_model.*",
    "re:.*audio_embedder.*",
    "re:.*multi_modal_projector.*",
    "re:.*embed_vision.*",
    "re:.*embed_audio.*",           # gemma-4 unified (omni)
]

# MoE routers/gates MUST stay full precision too: vLLM builds the router as a plain
# Linear, so a quantized router carries an unexpected `weight_scale` → load dies with
# `KeyError: '…router.proj.weight_scale'` (gemma-4) or the analogous mismatch on
# Qwen/Mixtral. These are the router modules ONLY — NOT the experts' gate_proj (which
# should be quantized), hence the `$` anchors / `.router.` scoping.
# vLLM keeps ALL MoE gating/routing Linears unquantized — not just the top-1 router
# (`mlp.gate`) but also per-layer shared-expert gates (`mlp.shared_expert_gate`). A
# quantized gate carries a `weight_scale` vLLM doesn't expect → it skips the scale and
# uses the raw fp8 weight undequantized → GARBAGE output (verified on Qwen3.6-35B-A3B).
# `re:.*gate$` catches every module whose name ENDS in `gate` (mlp.gate,
# shared_expert_gate, …) — never the experts' `gate_proj` (ends in `proj`).
_MOE_IGNORE = [
    "re:.*gate$",                     # MoE routers + shared-expert gates (NOT gate_proj)
    "re:.*\\.router\\..*",            # gemma-4 (router.proj)
    "re:.*\\.router$",
    "re:.*\\.gate\\.wg$",             # deepseek-style
]

# Hybrid state-space models (mamba / gated-DeltaNet linear attention): the SSM /
# linear-attention mixer layers are highly quantization-sensitive — fp8 on them yields
# GARBAGE output (verified on Qwen3.6-35B-A3B GDN: fp8'd `linear_attn.in_proj_*` →
# incoherent text). Keep them bf16 (quantize the full-attention + MoE/MLP layers
# instead), matching NVIDIA's mamba/Nemotron fp8 recipes. Regexes → no-op elsewhere.
_SSM_IGNORE = [
    "re:.*linear_attn.*",   # Qwen3.6 gated DeltaNet (in_proj_qkv/a/b/z, out_proj)
    "re:.*\\.mamba.*",      # Mamba/Mamba2/Nemotron-H mixer
    "re:.*mamba_mixer.*",
]


def _config_is_multimodal(conf) -> bool:
    """True for a vision/audio (VLM / omni) config — a `vision_config`/`audio_config`
    sub-config, or a conditional-generation / image-text / *VL / *Omni architecture."""
    if conf is None:
        return False
    if getattr(conf, "vision_config", None) is not None or getattr(conf, "audio_config", None) is not None:
        return True
    archs = getattr(conf, "architectures", None) or []
    return any(
        ("VL" in a) or ("Vision" in a) or ("ImageTextToText" in a) or ("Omni" in a) or a.endswith("ConditionalGeneration")
        for a in archs
    )


def _is_multimodal(model) -> bool:
    """True for vision/audio models — checks the loaded model's config."""
    return _config_is_multimodal(getattr(model, "config", None))


def _is_moe(model) -> bool:
    """True for mixture-of-experts models (also checks a nested text_config, since a
    multimodal wrapper config puts the expert fields there)."""
    conf = getattr(model, "config", None)
    keys = ("num_experts", "num_local_experts", "n_routed_experts", "num_experts_per_tok")
    for c in (conf, getattr(conf, "text_config", None)):
        if c is None:
            continue
        if any(getattr(c, k, None) for k in keys):
            return True
    archs = getattr(conf, "architectures", None) or []
    return any("moe" in a.lower() for a in archs)


def _is_hybrid_ssm(model) -> bool:
    """True for hybrid state-space models (mamba / linear-attention mixers alongside
    attention) — their SSM layers must stay bf16 (fp8 breaks them)."""
    conf = getattr(model, "config", None)
    keys = ("linear_key_head_dim", "linear_num_key_heads", "linear_conv_kernel_dim",
            "mamba_d_state", "mamba_num_heads", "mamba_expand", "ssm_state_size")
    for c in (conf, getattr(conf, "text_config", None)):
        if c is None:
            continue
        if any(getattr(c, k, None) for k in keys):
            return True
        lt = getattr(c, "layer_types", None) or []
        if any(("linear" in str(t).lower() or "mamba" in str(t).lower()) for t in lt):
            return True
    return False


def _save_processor_config(source: str, out_dir: str, hf_token) -> None:
    """Copy a VLM's processor / preprocessor config into the output next to the weights.
    Prefer AutoProcessor.save_pretrained; if its class isn't importable in this
    transformers (brand-new arch, e.g. gemma-4 unified → no Gemma4UnifiedProcessor),
    fall back to copying the processor/preprocessor JSONs straight from the source
    snapshot (already in the HF cache). Best-effort — never fatal to the quant."""
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(source, token=hf_token, trust_remote_code=True).save_pretrained(out_dir)
        log("[quant] saved multimodal processor via AutoProcessor")
        return
    except Exception as e:  # noqa: BLE001
        log(f"[quant] AutoProcessor unavailable ({type(e).__name__}); copying processor configs from snapshot")
    try:
        import shutil
        from huggingface_hub import snapshot_download
        snap = snapshot_download(source, token=hf_token, allow_patterns=[
            "preprocessor_config.json", "processor_config.json", "image_processor*.json",
            "video_preprocessor_config.json", "audio_processor*.json", "chat_template.json",
        ])
        copied = []
        for fn in os.listdir(snap):
            if fn.endswith(".json") and ("processor" in fn or "preprocess" in fn):
                shutil.copy2(os.path.join(snap, fn), os.path.join(out_dir, fn))
                copied.append(fn)
        log(f"[quant] copied processor configs from snapshot: {copied or 'none found'}")
    except Exception as e:  # noqa: BLE001
        log(f"[quant] WARN: could not obtain processor config: {type(e).__name__}: {e} "
            f"— multimodal serving may need it copied in manually")


def _recipe_ignore(cfg: dict, model=None) -> list:
    """Modules to exclude from quantization. Always `lm_head` (or the caller's
    `ignore_layers`); for a detected VLM, also the vision/audio stack (unless the caller
    opts in with quantize_vision); for a MoE model, also the router/gate. Both are
    modules vLLM keeps in full precision — quantizing them makes the model unloadable."""
    ignore = list(cfg.get("ignore_layers") or ["lm_head"])

    def _add(pats, why):
        added = [p for p in pats if p not in ignore]
        ignore.extend(added)
        if added:
            log(f"[quant] {why} → keeping in full precision: {added}")

    if model is not None and _is_multimodal(model) and not cfg.get("quantize_vision"):
        _add(_MULTIMODAL_IGNORE, "multimodal model detected")
    if model is not None and _is_moe(model):
        _add(_MOE_IGNORE, "MoE model detected (router/gate)")
    if model is not None and _is_hybrid_ssm(model):
        _add(_SSM_IGNORE, "hybrid SSM/linear-attention model detected")
    return ignore


def _load_model(source: str, hf_token):
    """Load the source model for quantization. Multimodal (VLM / omni) models are loaded
    FULL via AutoModelForImageTextToText so the vision/audio tower + the multimodal
    wrapper config survive — AutoModelForCausalLM silently drops them on archs like
    Qwen3.5-MoE-VL (→ a text-only `*ForCausalLM` model whose config lost `vision_config`,
    which vLLM's multimodal impl then refuses to load). Text models use
    AutoModelForCausalLM."""
    from transformers import AutoConfig, AutoModelForCausalLM
    kw = dict(torch_dtype="auto", device_map="auto", token=hf_token)
    try:
        conf = AutoConfig.from_pretrained(source, token=hf_token)
    except Exception as e:  # noqa: BLE001
        log(f"[quant] AutoConfig probe failed ({type(e).__name__}); loading as CausalLM")
        conf = None
    if _config_is_multimodal(conf):
        try:
            from transformers import AutoModelForImageTextToText
            log("[quant] multimodal config → loading full model via AutoModelForImageTextToText "
                "(preserves vision/audio tower + wrapper config)")
            return AutoModelForImageTextToText.from_pretrained(source, **kw)
        except Exception as e:  # noqa: BLE001
            log(f"[quant] AutoModelForImageTextToText failed ({type(e).__name__}: {e}); "
                f"falling back to AutoModelForCausalLM")
    return AutoModelForCausalLM.from_pretrained(source, **kw)


# --------------------------------------------------------------------------
# Recipe builder — scheme id → llm-compressor modifier(s).
# --------------------------------------------------------------------------
def _build_recipe(cfg: dict, model=None):
    scheme = cfg["scheme"]
    ignore = _recipe_ignore(cfg, model)
    from llmcompressor.modifiers.quantization import QuantizationModifier

    if scheme == "fp8-dynamic":
        return QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=ignore)
    if scheme == "fp8":
        # Static FP8 (per-tensor activation scales) — calibration fills the scales.
        return QuantizationModifier(targets="Linear", scheme="FP8", ignore=ignore)
    if scheme == "nvfp4":
        return QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=ignore)
    if scheme == "w4a16":
        from llmcompressor.modifiers.quantization import GPTQModifier
        return GPTQModifier(
            targets="Linear", scheme="W4A16", ignore=ignore,
            dampening_frac=float(cfg.get("dampening_frac") or 0.01),
        )
    if scheme == "w8a8-int8":
        from llmcompressor.modifiers.quantization import GPTQModifier
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        return [
            SmoothQuantModifier(smoothing_strength=float(cfg.get("smoothing_strength") or 0.8)),
            GPTQModifier(
                targets="Linear", scheme="W8A8", ignore=ignore,
                dampening_frac=float(cfg.get("dampening_frac") or 0.01),
            ),
        ]
    if scheme == "awq":
        from llmcompressor.modifiers.awq import AWQModifier
        return AWQModifier(targets="Linear", scheme="W4A16", ignore=ignore)
    raise RuntimeError(f"unknown scheme '{scheme}'")


def _dir_size_gb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return round(total / (1024 ** 3), 3)


# --------------------------------------------------------------------------
def run(cfg: dict) -> None:
    import tempfile
    import torch
    from transformers import AutoTokenizer
    from llmcompressor import oneshot

    scheme = cfg["scheme"]
    needs_calib = QUANT_SCHEMES.get(scheme, True)
    source = cfg["source_model"]
    hf_token = cfg.get("hf_token") or None
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)

    emit("PROGRESS", {"stage": "loading-model", "percent": 5})
    log(f"[quant] loading {source} (scheme={scheme}) …")
    model = _load_model(source, hf_token)
    tokenizer = AutoTokenizer.from_pretrained(source, token=hf_token)

    recipe = _build_recipe(cfg, model)
    oneshot_kwargs: dict = {"model": model, "recipe": recipe}
    if needs_calib:
        emit("PROGRESS", {"stage": "calibrating", "percent": 25})
        calib = _build_calibration_dataset(cfg, tokenizer)
        oneshot_kwargs.update({
            "dataset": calib,
            "num_calibration_samples": min(int(cfg.get("num_calibration_samples") or 512), len(calib)),
            "max_seq_length": int(cfg.get("max_seq_length") or 2048),
        })

    emit("PROGRESS", {"stage": "quantizing", "percent": 45})
    log("[quant] running llm-compressor oneshot …")
    oneshot(**oneshot_kwargs)

    out_dir = os.path.join(tempfile.mkdtemp(prefix="sgpu-quant-"), source.split("/")[-1] + f"-{scheme}")
    os.makedirs(out_dir, exist_ok=True)
    emit("PROGRESS", {"stage": "saving", "percent": 75})
    log(f"[quant] saving compressed model → {out_dir}")
    # Shard the safetensors (default 5GB) instead of one giant file. A single ~30GB
    # shard uploads as one opaque multipart blob → the job sits at "uploading 88%" with
    # no feedback for minutes (looks stuck). Small shards upload per-file (visible
    # progress below), are resumable, and lower peak host memory while writing.
    try:
        model.save_pretrained(out_dir, save_compressed=True,
                              max_shard_size=str(cfg.get("max_shard_size") or "5GB"))
    except TypeError:
        # older compressed-tensors save_pretrained without max_shard_size support
        log("[quant] max_shard_size unsupported here — saving unsharded")
        model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)
    # Multimodal models need the image/audio processor config alongside the weights,
    # else vLLM can't build the feature extractor ("Can't load feature extractor …") —
    # model/tokenizer save_pretrained don't emit preprocessor_config.json /
    # processor_config.json. Get them into the output.
    if _is_multimodal(model):
        _save_processor_config(source, out_dir, hf_token)

    try:
        emit("SIZES", {"quantized_gb": _dir_size_gb(out_dir)})
    except Exception:
        pass

    # ---- upload compressed model to S3 ----
    art = cfg.get("artifacts") or {}
    s3_uri = None
    if art.get("bucket"):
        emit("PROGRESS", {"stage": "uploading", "percent": 88})
        cli = _s3_client(art)
        base_key = art["prefix"].rstrip("/") + "/model"
        files = [os.path.join(r, fn) for r, _d, fns in os.walk(out_dir) for fn in fns]
        files.sort(key=os.path.getsize)  # small (configs) first, big shards last
        total = sum(os.path.getsize(f) for f in files) or 1
        n = len(files)
        # Byte-level progress via a boto3 Callback (invoked as multipart parts complete),
        # so the bar advances 88 → 94 continuously even for one large shard — not frozen.
        prog = {"sent": 0, "last": 88.0}
        def _cb(nbytes):
            prog["sent"] += nbytes
            pct = round(88 + 6 * prog["sent"] / total, 1)
            if pct - prog["last"] >= 0.3:  # throttle marker spam
                prog["last"] = pct
                emit("PROGRESS", {"stage": "uploading", "percent": pct})
        for i, fp in enumerate(files, 1):
            rel = os.path.relpath(fp, out_dir)
            log(f"[upload] s3 {i}/{n} {rel} ({os.path.getsize(fp) / 1024**3:.2f}GB) …")
            cli.upload_file(fp, art["bucket"], f"{base_key}/{rel}", Callback=_cb)
        s3_uri = f"s3://{art['bucket']}/{base_key}/"
        log(f"[upload] quantized model → {s3_uri}")

    # ---- optional HF push ----
    hf_repo = None
    if cfg.get("hf_push_repo") and hf_token:
        try:
            emit("PROGRESS", {"stage": "hf-push", "percent": 95})
            from huggingface_hub import HfApi

            repo = cfg["hf_push_repo"]
            n_files = sum(len(fns) for _r, _d, fns in os.walk(out_dir))
            log(f"[upload] pushing {n_files} files → https://huggingface.co/{repo} "
                f"(large shards upload in the background; per-file progress in the hub client log)")
            api = HfApi(token=hf_token)
            api.create_repo(repo, private=bool(cfg.get("hf_push_private", True)),
                            exist_ok=True, repo_type="model")
            api.upload_folder(folder_path=out_dir, repo_id=repo, repo_type="model")
            hf_repo = repo
            log(f"[upload] pushed quantized model → https://huggingface.co/{repo}")
        except Exception as e:  # noqa: BLE001
            log(f"[upload] HF push failed: {e}")

    emit("ARTIFACT", {"s3_uri": s3_uri, "hf_repo": hf_repo})
    emit("PROGRESS", {"stage": "done", "percent": 100})
    emit("DONE", {"scheme": scheme})


def _redirect_tmp(base: str) -> None:
    """Move TMPDIR / HF cache off the small local /tmp onto a roomy dir (default
    /share) — a big model + its quantized copy easily overflow a pod's /tmp."""
    try:
        os.makedirs(base, exist_ok=True)
        for var in ("TMPDIR", "HF_HOME", "XDG_CACHE_HOME"):
            if not os.environ.get(var):
                d = os.path.join(base, {"TMPDIR": "tmp", "HF_HOME": "huggingface", "XDG_CACHE_HOME": "cache"}[var])
                os.makedirs(d, exist_ok=True)
                os.environ[var] = d
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to JSON config")
    ap.add_argument("--deps-only", action="store_true", help="install dependencies then exit")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    _redirect_tmp(cfg.get("work_dir") or "/share")
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


if __name__ == "__main__":
    sys.exit(main())
