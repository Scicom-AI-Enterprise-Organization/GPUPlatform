#!/usr/bin/env python3
"""Standalone Whisper finetuning script — shipped to a RunPod pod / VM by the
gateway's Autotrain runner and executed over SSH. It has NO gateway imports;
everything it needs arrives in a single JSON config file (path passed via
--config). It installs its own deps, resolves the dataset (S3 or HuggingFace),
finetunes a Whisper model with HF Seq2SeqTrainer, evaluates WER + CER every
epoch, early-stops on patience, then uploads the best model + metrics to S3
(and optionally pushes to the HF Hub).

Contract with the gateway (parsed from stdout):
  @@METRIC {json}   one per epoch: {epoch, wer, cer, eval_loss, train_loss}
  @@ARTIFACT {json} after upload: {s3_uri, hf_repo?}
  @@DONE {json}     final: {best:{epoch,wer,cer}, epochs:int, stopped_early:bool}
  @@ERROR {json}    fatal: {message}
Every other line is free-form progress and streamed to the run's log.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import traceback
import unicodedata


# Under torchrun (DDP) every GPU runs this script as a separate rank; only rank 0
# may write to the gateway's SSH stream, else every log line + @@STEP/@@METRIC is
# duplicated WORLD_SIZE times (and the gateway parses garbage). torchrun sets RANK
# before the script runs; plain `python` (single GPU / DataParallel) leaves it
# unset → rank 0.
_IS_MAIN = os.environ.get("RANK", "0") == "0"


def log(msg: str) -> None:
    if _IS_MAIN:
        print(msg, flush=True)


def emit(tag: str, obj: dict) -> None:
    """Structured line the gateway parses out of the stream (rank-0 only)."""
    if _IS_MAIN:
        print(f"@@{tag} {json.dumps(obj)}", flush=True)


# Set by run() to the run's work dir; rm'd by main() when cleanup_checkpoints is on.
_RUN_WORKDIR = None


def parse_precision(p):
    """'<load>-<amp>' → (torch_dtype_name, amp). The load part is the weight
    dtype the model is loaded in; the amp part is the mixed-precision (AMP)
    training dtype. Back-compat: a bare 'bf16'/'fp16' = load fp32 + that AMP;
    'fp32' = full fp32 (no AMP)."""
    p = (p or "fp32-bf16").lower()
    if "-" in p:
        load, amp = p.split("-", 1)
    elif p == "fp32":
        load, amp = "fp32", ""
    else:
        load, amp = "fp32", p
    load_dt = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}.get(load, "float32")
    return load_dt, (amp if amp in ("bf16", "fp16") else "")


# ==========================================================================
# Text standardization + language detection
# --------------------------------------------------------------------------
# Mirror of autotrain/whisper/cleaning.py (this script ships to the pod with NO
# gateway imports, so it can't import that module — keep the two in sync).
# fix_spacing / whisper_textcleaning / chinese_ratio are verbatim; detect_language
# additionally tolerates a missing model (returns en); format_whisper's
# clean→detect→format is inlined in _prepare_texts (it handles empty rows here).
# Used to (1) standardize/clean each transcription before tokenizing and (2) tag
# each utterance's language for the Whisper prompt. zh is decided by CJK character
# ratio (the bahasa/en fastText model has no `zh` label); see detect_language.
# ==========================================================================

# CJK / full-width punctuation that NFKC does NOT fold to ASCII (ideographic full stop,
# enumeration comma, full-width colon/semicolon, CJK brackets and dashes). We map them to
# a single ASCII punctuation set so the model sees consistent punctuation across en/ms/zh.
# (NFKC already handles the full-width comma '，', question '？' and exclamation '！'.)
_CJK_PUNCT = str.maketrans({
    '。': '.', '、': ',', '〜': '~', '～': '~',
    '；': ';', '：': ':', '·': ' ', '・': ' ',
    '「': '"', '」': '"', '『': '"', '』': '"',
    '《': '"', '》': '"', '〈': '"', '〉': '"',
    '【': '(', '】': ')', '〔': '(', '〕': ')',
})

# Curly quotes / dashes / ellipsis -> ASCII so they don't fragment the vocab.
_QUOTES_DASHES = str.maketrans({
    '‘': "'", '’': "'", '“': '"', '”': '"',
    '–': '-', '—': '-', '―': '-', '−': '-',
})

# Zero-width / BiDi / BOM control characters that carry no acoustic content.
_INVISIBLES = re.compile(r'[­​-‏‪-‮⁠﻿]')


def fix_spacing(text):
    quote_pattern = r'"([^"]*)"'
    def fix_quotes(match):
        content = match.group(1).strip()
        return f'"{content}"'

    text = re.sub(quote_pattern, fix_quotes, text)

    paren_pattern = r'\(([^)]*)\)'
    def fix_parens(match):
        content = match.group(1).strip()
        return f'({content})'

    text = re.sub(paren_pattern, fix_parens, text)
    text = re.sub(r'\s+([,\.!?])', r'\1', text)
    return text

def whisper_textcleaning(text):
    # --- unicode standardization (run first, before any regex) ---
    # NFKC folds full-width letters/digits + '，！？' to ASCII and the ideographic space to ' '.
    text = unicodedata.normalize('NFKC', text)
    text = _INVISIBLES.sub('', text)
    text = text.translate(_QUOTES_DASHES)
    text = text.translate(_CJK_PUNCT)
    text = text.replace('…', '...')

    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    text = re.sub(r'\b(?:ok|oke|okay|okey|okie)\b', 'OK', text, flags=re.IGNORECASE)
    # nasal hesitations (hmm/mm/erm/um/uh/uhm ...) -> single canonical token
    text = re.sub(r'\b(?:h+m+|u+h*m+|u+h+|erm+|mm+)\b', 'herm', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(a+h+|a+\s*a+)(?=[\s,\.!?]|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\.{2,}', ',', text)
    text = re.sub(r'(?<=\s)-(\w+)\b', r'\1', text)
    text = re.sub(r'\b(\w+)-(?=\s|$)', r'\1', text)
    text = re.sub(r'\b(um|uh|aa|erm|herm)(\s+\1)+', r'\1', text, flags=re.IGNORECASE)
    # collapse repeated end-punctuation ("!!" / "??" / ",,")
    text = re.sub(r'([,!?])\1+', r'\1', text)
    # space after , ! ? unless the next char is whitespace or a CJK character
    text = re.sub(r'([,!?])(?=[^\s一-鿿])', r'\1 ', text)
    # space after a sentence-ending period only before a capital letter (keep decimals,
    # domains and emails intact: 3.30 / i.unify.my / name@gmail.com)
    text = re.sub(r'(?<!\d)\.(?=[A-Z])', '. ', text)
    text = fix_spacing(text)
    # tidy punctuation stranded by removed interjections ("Sekejap. Ah, ..." / "Gun, ah?"):
    text = re.sub(r'\s*,\s*(?=[.!?])', '', text)           # drop a comma sitting before . ! ?
    text = re.sub(r'([.!?])\s*,', r'\1', text)              # drop a comma sitting after . ! ?
    text = re.sub(r'([,.!?])(?:\s*[,.!?])+', r'\1', text)   # collapse any remaining run -> first mark
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[\s,.?!:;]+', '', text)                # strip leading punctuation
    text = re.sub(r'\s*[,:;]+$', '', text)                  # strip a dangling trailing , : ;
    return text.strip()


# CJK ideograph ranges (Unified, Extension-A, Compatibility). Used to decide Chinese
# by character ratio — the bahasa/en fastText model has no `zh` label, so Chinese
# transcripts would otherwise be misdetected as `en`. (Full-width CJK *punctuation*
# is already folded to ASCII by whisper_textcleaning, so only ideographs match here.)
_CJK = re.compile(r'[一-鿿㐀-䶿豈-﫿]')


def chinese_ratio(text):
    """Fraction (0..1) of non-whitespace characters that are CJK ideographs."""
    chars = re.sub(r'\s+', '', text)
    if not chars:
        return 0.0
    return len(_CJK.findall(chars)) / len(chars)


def detect_language(text, lang_model, chinese_threshold=0.5):
    """Whisper language code for `text`:

    - `zh` when CJK ideographs are at least `chinese_threshold` of the characters
      (default 50%), checked first since the fastText model can't see Chinese;
    - otherwise the fastText bahasa/en model
      (`mesolitica/fasttext-language-detection-bahasa-en`): `bahasa` -> `ms`,
      anything else (incl. `english` / `other`) -> `en`.

    `lang_model` may be None (model unavailable) — then non-Chinese text falls back
    to `en` (the zh-by-ratio decision still stands).
    """
    if chinese_ratio(text) >= chinese_threshold:
        return "zh"
    if lang_model is None:
        return "en"
    line = text.replace("\n", " ").replace("\r", " ").strip()
    if not line:
        return "en"
    labels, _ = lang_model.predict(line, k=10)
    clean = [l.replace("__label__", "") for l in labels]
    top = clean[0]
    if top == "other" and len(clean) > 1:
        top = clean[1]
    return "ms" if top == "bahasa" else "en"


# --------------------------------------------------------------------------
# Per-utterance language tagging glue (gateway-side; not part of cleaning.py)
# --------------------------------------------------------------------------
_LANG_MODEL = None  # cached fastText model (loaded once per process)


def _auto_lang(cfg) -> bool:
    """Per-utterance language detection (en/ms/zh) is the **default** — ON whenever
    `language` is unset/None/'' (or explicitly 'auto'/'multi'/'multilingual'). Set a
    concrete code (e.g. 'ms') to opt out and pin every utterance to that language."""
    return (cfg.get("language") or "").strip().lower() in ("", "auto", "multi", "multilingual")


def _load_lang_model(cfg):
    """Download + load the bahasa/en fastText model for language detection. Cached.
    Returns None on failure — detection then tags zh by character ratio and defaults
    everything else to en (see detect_language)."""
    global _LANG_MODEL
    if _LANG_MODEL is not None:
        return _LANG_MODEL
    try:
        import fasttext
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="mesolitica/fasttext-language-detection-bahasa-en",
            filename="fasttext.ftz",
            token=cfg.get("hf_token") or None,
        )
        _LANG_MODEL = fasttext.load_model(path)
        log("[lang] fastText bahasa/en model loaded for per-utterance detection")
    except Exception as e:  # noqa: BLE001
        log(f"[lang] WARNING: fastText model unavailable ({e}); "
            "tagging zh by CJK character ratio and defaulting other text to en")
        _LANG_MODEL = None
    return _LANG_MODEL


def _prepare_texts(pairs: list, cfg: dict, lang_model, label: str) -> None:
    """In-place: standardize/clean each pair's transcription and (auto mode) prepend
    the Whisper prompt with a per-utterance language token. Sets pair['preformatted']
    when the text already carries the full `<|startoftranscript|>…<|endoftext|>`
    prompt (so __getitem__ tokenizes with add_special_tokens=False)."""
    if not pairs:
        return
    clean = bool(cfg.get("clean_text", True))
    auto = _auto_lang(cfg)
    task = cfg.get("task") or "transcribe"
    n_clean = 0
    counts: dict[str, int] = {}
    for p in pairs:
        t = p.get("text") or ""
        if clean:
            ct = whisper_textcleaning(t)
            if ct != t:
                n_clean += 1
            t = ct
        if auto:
            # zh decided by CJK character ratio first; ms/en via fastText (or en fallback).
            lang = detect_language(t, lang_model)
            p["text"] = f"<|startoftranscript|><|{lang}|><|{task}|><|notimestamps|> {t}<|endoftext|>"
            p["preformatted"] = True
            counts[lang] = counts.get(lang, 0) + 1
        else:
            p["text"] = t
    if clean:
        log(f"[clean] {label}: standardized {n_clean}/{len(pairs)} transcriptions")
    if auto:
        log(f"[lang] {label}: per-utterance language → "
            + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"))


# --------------------------------------------------------------------------
# Dependency bootstrap — an isolated uv venv (created by --deps-only), so the
# Whisper stack never clobbers the box's system python or the TTS stack.
# --------------------------------------------------------------------------
DEFAULT_VENV = "/share/autotrain-whisper"
# Pin torch to match the pod image (DEFAULT_IMAGE = runpod/pytorch:…-cu1281-torch280…)
# and install it from the matching CUDA wheel index — like the sibling trainers
# (omnivoice/llm_finetune). An UNPINNED `torch` from PyPI's default index pulls the
# latest (a cu13 build) whose CUDA mismatches the cu128 pod → torch.cuda unavailable →
# transformers raises "doesn't support bf16/gpu". A cu128 build also runs on newer
# (cu130) drivers via CUDA backward-compat.
TORCH_VERSION = "2.8.0"
TORCH_CUDA = "cu128"


def _ensure_venv(cfg: dict) -> str:
    """Create/reuse an isolated uv venv with the Whisper stack; return its python.
    datasets is pinned <4.0 (4.x needs torchcodec for the Audio feature; we use
    the soundfile/librosa decoder). Idempotent — fast when the venv is ready."""
    import shutil

    venv = (cfg.get("venv_path") or DEFAULT_VENV).rstrip("/")
    py = os.path.join(venv, "bin", "python")
    env = {**os.environ, "PIP_CONSTRAINT": "", "PIP_REQUIRE_HASHES": "0"}
    # peft is always installed so the same venv serves LoRA + non-LoRA runs (and
    # _present checks it, so a venv first built for a non-LoRA run still gets it).
    # torch is installed SEPARATELY below (pinned + CUDA index); these are the
    # PyPI-default packages.
    pkgs = [
        # fasttext-wheel's compiled extension is built against the numpy 1.x C ABI —
        # under numpy 2.x its predict() raises "Unable to avoid copy" / "dtype size
        # changed". Pin numpy 1.26.4 (last 1.x; compatible with this torch/transformers/
        # datasets stack) so language detection works. Keep first so the resolver honours it.
        "numpy==1.26.4",
        "transformers>=4.44", "datasets>=2.20,<4.0", "evaluate", "jiwer",
        "accelerate>=0.30", "soundfile", "librosa", "boto3", "huggingface_hub", "peft>=0.11",
        # per-utterance bahasa/en language detection (zh is decided by char ratio,
        # no model). fasttext-wheel = prebuilt wheels, no C++ toolchain on the pod.
        "fasttext-wheel",
        # PyAV bundles ffmpeg+libopus, so the livekit/opus augmentations do a REAL
        # in-process codec round-trip instead of the DSP approximation (they log
        # loudly when they have to fall back). Wheels only, no system ffmpeg needed.
        "av",
    ]
    check_mods = ["torch", "transformers", "datasets", "evaluate", "jiwer", "soundfile", "boto3", "peft", "fasttext", "av"]
    report_to = (cfg.get("tracking") or {}).get("report_to") or cfg.get("report_to") or []
    if "wandb" in report_to:
        pkgs.append("wandb"); check_mods.append("wandb")
    if "mlflow" in report_to:
        pkgs.append("mlflow"); check_mods.append("mlflow")

    def _present() -> bool:
        # Require numpy 1.x AND the pinned torch. A venv built before these pins may
        # carry numpy 2.x (breaks fasttext) or a too-new torch (a cu13 wheel → no GPU
        # on the cu128 pod). Either → exit(3) so the install below reconciles it.
        probe = (
            "import " + ", ".join(check_mods) + "\n"
            "import numpy, torch, sys\n"
            f"ok = numpy.__version__.startswith('1.') and torch.__version__.startswith('{TORCH_VERSION}')\n"
            "sys.exit(0 if ok else 3)\n"
        )
        try:
            subprocess.check_call(
                [py, "-c", probe],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    if os.path.exists(py) and _present():
        log(f"[deps] Whisper venv ready: {py}")
        return py
    have_uv = shutil.which("uv") is not None

    def _pip(*args):
        if have_uv:
            subprocess.check_call(["uv", "pip", "install", "--python", py, *args], env=env)
        else:
            subprocess.check_call([py, "-m", "pip", "install", "-q", *args], env=env)

    # Create the venv only if absent (`uv venv` errors on a non-empty dir); then
    # ALWAYS install — idempotent, and adds any missing pkg (e.g. peft) to a venv
    # first built for a different run.
    if not os.path.exists(py):
        log(f"[deps] creating venv {venv} …")
        if have_uv:
            subprocess.check_call(["uv", "venv", venv, "--python", "3.12"], env=env)
        else:
            subprocess.check_call([sys.executable, "-m", "venv", venv], env=env)
            subprocess.check_call([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], env=env)
    # torch first, pinned + from the CUDA wheel index (those wheels aren't on PyPI's
    # default index). A separate call from the PyPI pkgs: passing --index-url with the
    # PyPI-only pkgs would 404 them. Downgrades a previously-resolved too-new torch.
    log(f"[deps] installing torch=={TORCH_VERSION} ({TORCH_CUDA}) into {venv} …")
    _pip(f"torch=={TORCH_VERSION}", "--index-url", f"https://download.pytorch.org/whl/{TORCH_CUDA}")
    log(f"[deps] installing Whisper stack into {venv} …")
    _pip(*pkgs)
    log(f"[deps] Whisper venv ready: {py}")
    return py


# --------------------------------------------------------------------------
# Dataset resolution → list[{"audio": <local path>, "text": str, "split": str?}]
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


def _read_metadata_rows(ds: dict) -> list[dict]:
    """Read the dataset's metadata file (csv/json/jsonl) from S3 into dicts."""
    import csv
    import io

    cli = _s3_client(ds)
    body = cli.get_object(Bucket=ds["bucket"], Key=ds["metadata_key"])["Body"].read()
    text = body.decode("utf-8", errors="replace")
    fmt = (ds.get("format") or "").lower()
    if not fmt:
        fmt = "jsonl" if ds["metadata_key"].endswith(".jsonl") else (
            "json" if ds["metadata_key"].endswith(".json") else "csv"
        )
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(text)))
    if fmt == "jsonl":
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("data", data.get("rows", []))


def _s3_url_key_for_bucket(ref: str, bucket: str | None) -> str | None:
    """If `ref` is an http(s) S3 URL for `bucket` (virtual-hosted
    `bucket.s3….amazonaws.com/key` or path-style `…/bucket/key`), return the
    decoded object key, so the caller can re-fetch via boto3 instead of trusting
    a possibly-expired presigned signature baked into the metadata. Returns None
    for non-S3 / other-bucket URLs (those fall through to a plain HTTP GET)."""
    if not bucket or not (ref.startswith("http://") or ref.startswith("https://")):
        return None
    from urllib.parse import urlparse, unquote

    u = urlparse(ref)
    host, path, b = u.netloc.lower(), u.path.lstrip("/"), bucket.lower()
    if host.startswith(f"{b}.s3") or host.startswith(f"{b}.s3-"):  # virtual-hosted
        return unquote(path)
    if path.lower().startswith(f"{b}/"):  # path-style
        return unquote(path[len(bucket) + 1:])
    return None


def _download_audio_s3(ds: dict, ref: str, dest_dir: str) -> str | None:
    """Resolve a metadata audio reference to a local file. `ref` is either an
    http(s) URL (possibly a presigned S3 link) or a key relative to the storage
    prefix + audio_prefix."""
    import urllib.request

    fname = os.path.basename(ref.split("?")[0]) or "audio.wav"
    local = os.path.join(dest_dir, fname)
    if os.path.exists(local):
        return local
    try:
        own_key = _s3_url_key_for_bucket(ref, ds.get("bucket"))
        if own_key is not None:
            # The metadata stored an http(s) S3 URL for our own bucket — fetch via
            # boto3 with the dataset creds rather than the URL, since a stored
            # presigned link can expire during a long, multi-trial sweep.
            _s3_client(ds).download_file(ds["bucket"], own_key, local)
        elif ref.startswith("http://") or ref.startswith("https://"):
            urllib.request.urlretrieve(ref, local)
        else:
            cli = _s3_client(ds)
            prefix = (ds.get("audio_prefix") or "").strip("/")
            key = "/".join(p for p in [prefix, ref.lstrip("/")] if p)
            cli.download_file(ds["bucket"], key, local)
        return local
    except Exception as e:  # noqa: BLE001
        log(f"[data] skip audio {ref!r}: {e}")
        return None


def load_pairs(ds: dict, work: str) -> list[dict]:
    """Return lightweight metadata items — NO audio decoded/downloaded here.
    Each item carries the text + a lazy handle to its audio; the actual bytes
    are fetched + decoded in _LazyAsrDataset.__getitem__ during training (so a
    big dataset costs ~nothing up front, vs. the old eager HF .map that built a
    multi-GB Arrow table and stalled on slow shared mounts).

      HF item: {src:"hf", hf_split, hf_idx, audio_field, text, split}
      S3 item: {src:"s3", s3_spec, s3_ref, text, split?}
    """
    kind = ds.get("kind")
    audio_field = ds.get("audio_field") or "audio"
    text_field = ds.get("transcription_field") or "transcription"

    if kind == "hf":
        from datasets import Audio, load_dataset

        token = ds.get("hf_token") or None
        repo = ds["hf_repo"]
        log(f"[data] loading HF dataset metadata: {repo} (audio fetched lazily per item)")
        dd = load_dataset(repo, token=token)
        out: list[dict] = []
        for split_name, split in dd.items():
            tf = (ds.get("split_fields") or {}).get(split_name, text_field)
            # lazy 16 kHz resample on access — does NOT trigger a full decode pass
            split = split.cast_column(audio_field, Audio(sampling_rate=16000))
            texts = list(split[tf]) if tf in split.column_names else [""] * split.num_rows
            for idx in range(split.num_rows):
                out.append({
                    "src": "hf", "hf_split": split, "hf_idx": idx,
                    "audio_field": audio_field,
                    "text": texts[idx] if idx < len(texts) else "",
                    "split": split_name,
                })
        log(f"[data] {len(out)} examples indexed (metadata only)")
        return out

    # S3 / upload metadata — keep only the per-row ref + text; download on access.
    rows = _read_metadata_rows(ds)
    # Rows manually un-ticked in the row browser (excluded from training). Indices
    # are positions in this metadata file — the same order the preview shows.
    excluded = {int(x) for x in (ds.get("excluded_rows") or [])}
    log(f"[data] {len(rows)} metadata rows from s3://{ds['bucket']}/{ds['metadata_key']} "
        f"(audio fetched lazily per item)"
        + (f"; {len(excluded)} manually excluded" if excluded else ""))
    out = []
    for i, r in enumerate(rows):
        if i in excluded:
            continue
        ref = r.get(audio_field)
        text = r.get(text_field)
        if not ref or text is None:
            continue
        item = {"src": "s3", "s3_spec": ds, "s3_ref": str(ref), "text": str(text)}
        if r.get("split"):
            item["split"] = str(r["split"])
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Audio augmentation (TRAINING audio only). Each technique is a numpy/scipy/
# librosa transform on a float waveform at sr. The user multi-selects which to
# enable; one is picked at random per augmented sample. Hardens the model
# against phone / noisy conditions. `telephone` is ported from the Scicom STT
# whisper augmentation; the rest are standard ASR augmentations.
#
# The `livekit*` family below models the WebRTC transport a voice agent puts in
# front of the model — see the "LiveKit / WebRTC transport chain" section.
# ---------------------------------------------------------------------------
def _aug_telephone(x, sr):
    """Phone-line degradation: attenuate → 300–3400 Hz band-pass → downsample to
    5/6/8 kHz → additive noise → hard clip → random chunk dropout → upsample."""
    import numpy as np
    from scipy.signal import butter, lfilter, resample
    x = x * (10 ** (-15 / 20.0))
    nyq = 0.5 * sr
    b, a = butter(4, [300 / nyq, 3400 / nyq], btype="band")
    x = lfilter(b, a, x)
    down_sr = random.choice([5000, 6000, 8000])
    x = resample(x, max(1, int(len(x) * down_sr / sr)))
    rms = float(np.sqrt(np.mean(x ** 2))) or 1e-9
    x = x + np.random.normal(0, rms / (10 ** (random.randint(20, 50) / 20)), size=x.shape)
    x = np.clip(x, -0.25, 0.25)
    for i in range(0, len(x), 400):
        if np.random.rand() < 0.03:
            x[i:i + 400] = 0.0
    return resample(x, max(1, int(len(x) * sr / down_sr)))


def _aug_noise(x, sr):
    """Additive Gaussian noise at a random SNR (10–40 dB)."""
    import numpy as np
    rms = float(np.sqrt(np.mean(x ** 2))) or 1e-9
    return x + np.random.normal(0, rms / (10 ** (random.randint(10, 40) / 20)), size=x.shape)


def _aug_dropout(x, sr):
    """Zero out random ~25 ms chunks (packet-loss / clipping)."""
    import numpy as np
    out = x.copy()
    chunk = max(1, int(0.025 * sr))
    for i in range(0, len(out), chunk):
        if np.random.rand() < 0.05:
            out[i:i + chunk] = 0.0
    return out


def _aug_gain(x, sr):
    """Random volume change (−20 … +6 dB)."""
    return x * (10 ** (random.uniform(-20, 6) / 20.0))


def _aug_pitch(x, sr):
    """Pitch shift ±3 semitones (preserves duration)."""
    import librosa
    import numpy as np
    return librosa.effects.pitch_shift(x.astype(np.float32), sr=sr, n_steps=random.uniform(-3, 3))


def _aug_speed(x, sr):
    """Time-stretch 0.9–1.1× (speaking-rate change; alters duration)."""
    import librosa
    import numpy as np
    return librosa.effects.time_stretch(x.astype(np.float32), rate=random.uniform(0.9, 1.1))


def _aug_reverb(x, sr):
    """Light room reverb via convolution with a short decaying-noise impulse."""
    import numpy as np
    from scipy.signal import fftconvolve
    n = max(1, int(0.05 * sr))
    ir = np.exp(-6.0 * np.arange(n) / n) * np.random.normal(0, 1, n)
    ir[0] = 1.0
    y = fftconvolve(x, ir)[: len(x)]
    peak = float(np.max(np.abs(y))) or 1.0
    return y / peak * (float(np.max(np.abs(x))) or 1.0)


def _aug_bandpass(x, sr):
    """Telephone 300–3400 Hz band-pass only (no resample/noise)."""
    from scipy.signal import butter, lfilter
    nyq = 0.5 * sr
    b, a = butter(4, [300 / nyq, 3400 / nyq], btype="band")
    return lfilter(b, a, x)


# ===========================================================================
# LiveKit / WebRTC transport chain
# ===========================================================================
# Why this exists: a Whisper finetune that scores well on a clean held-out set
# collapses once it is served behind a LiveKit voice agent, because LiveKit does
# not hand the model the microphone signal — it hands it the output of a long
# lossy chain. Every stage below was read out of the LiveKit sources rather than
# guessed (versions current as of 2026-08):
#
#   client-sdk-js/src/room/defaults.ts
#     audioDefaults    = autoGainControl:true, echoCancellation:true,
#                        noiseSuppression:true, voiceIsolation:true
#     publishDefaults  = audioPreset: AudioPresets.music, dtx:true, red:true
#   client-sdk-js/src/room/track/options.ts
#     AudioPresets     = telephone 12k, speech 24k, music 48k, musicStereo 64k,
#                        musicHighQuality 96k, musicHighQualityStereo 128k (bps)
#     audioCodecs      = ['opus', 'red']
#   agents/livekit/agents/voice/room_io/_input.py
#     rtc.AudioProcessingModule(auto_gain_control=True) on EVERY inbound frame,
#     then an optional noise-cancellation FrameProcessor (Krisp BVC), then
#     rtc.AudioResampler(48000 -> stt.sample_rate)
#   agents/livekit/agents/stt/stt.py
#     resamples to the plugin's sample rate at AudioResamplerQuality.HIGH
#   livekit-plugins-openai/.../stt.py
#     SAMPLE_RATE = 24000 -> the OpenAI-compatible endpoint is fed 24 kHz WAV
#   livekit-plugins-silero/.../vad.py
#     min_speech_duration 0.05, min_silence_duration 0.55,
#     prefix_padding_duration 0.5, activation_threshold 0.5, sample_rate 16000
#
# so the real signal path is:
#
#   mic 48k -> browser APM (HPF, AEC, NS, voiceIsolation, AGC)
#           -> Opus encode (mono, 20 ms frames, VOIP, DTX + RED)
#           -> network loss/jitter -> Opus decode + PLC
#           -> agent APM (AGC2, a SECOND gain stage) -> optional Krisp BVC
#           -> soxr 48k -> 24k -> Silero VAD crop -> WAV -> this model
#
# Each stage is exposed on its own so a run can isolate one, and `livekit`
# applies the whole thing in order with per-sample randomisation. `livekit_sip`
# is the telephony variant: LiveKit SIP negotiates PCMU/PCMA (G.711, 8 kHz) by
# default on a trunk and transcodes that leg to Opus for the SFU, so a phone
# caller reaches the model through a TANDEM of two codecs.
#
# Everything here is a pure waveform transform, so the transcript stays valid:
# no stage removes a word. In particular `vad_clip` tightens leading/trailing
# SILENCE only and never cuts into speech — cropping real speech against an
# unchanged label is how you train a model to hallucinate.
# ===========================================================================
def _resample(x, sr_in: int, sr_out: int, quality: str = "MQ"):
    """soxr (what LiveKit's rtc.AudioResampler uses) when available, else a
    polyphase fallback. LiveKit's inbound resampler runs at the library default
    quality (AudioResamplerQuality.MEDIUM) — 'MQ' matches it."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if int(sr_in) == int(sr_out) or x.size == 0:
        return x
    try:
        import soxr
        return soxr.resample(x.astype(np.float32), sr_in, sr_out, quality=quality).astype(np.float64)
    except Exception:  # noqa: BLE001
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr_in), int(sr_out))
        return resample_poly(x, int(sr_out) // g, int(sr_in) // g)


def _fit(y, n: int):
    """Trim/zero-pad to exactly n samples (codec + STFT round-trips drift a few)."""
    import numpy as np
    y = np.asarray(y, dtype=np.float64)
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


# --- Opus ------------------------------------------------------------------
_OPUS_BACKEND = None


def _opus_backend() -> str:
    """Resolve the codec backend ONCE: PyAV (in-process libopus, ~ms per clip) >
    ffmpeg CLI (2 subprocesses per clip) > a DSP approximation. The fallback is
    logged loudly — silently training on a filter that is not a codec would look
    exactly like success."""
    global _OPUS_BACKEND
    if _OPUS_BACKEND is not None:
        return _OPUS_BACKEND
    import shutil
    try:
        import av  # noqa: F401
        _OPUS_BACKEND = "av"
    except Exception:  # noqa: BLE001
        _OPUS_BACKEND = "ffmpeg" if shutil.which("ffmpeg") else "dsp"
    if _OPUS_BACKEND == "dsp":
        log("[augment] WARNING: neither PyAV nor ffmpeg is available — opus/livekit "
            "augmentation falls back to a DSP APPROXIMATION, not a real codec "
            "round-trip. Install `av` for parity with production.")
    else:
        log(f"[augment] opus codec backend: {_OPUS_BACKEND}")
    return _OPUS_BACKEND


def _opus_av(x48, bitrate: int, application: str):
    import io

    import av
    import numpy as np

    pcm = np.clip(np.asarray(x48, dtype=np.float64), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16).reshape(1, -1)
    buf = io.BytesIO()
    # ogg is only a container to hold the packets; the encoder/decoder pair is the
    # same libopus WebRTC uses. PyAV's AudioCodecContext FIFOs the input into the
    # encoder's 20 ms frame size for us.
    out = av.open(buf, mode="w", format="ogg")
    # compression_level maps to OPUS_SET_COMPLEXITY. ffmpeg defaults to 10; libwebrtc
    # uses 9 on desktop and 5 on mobile. 5 is ~30 % faster to encode with distortion
    # indistinguishable from 10 (measured: -7.4 vs -7.6 dB residual, corr .908 vs .911
    # at 24 kbps) — and the encode is by far the most expensive part of this chain.
    opts = {"application": application, "frame_duration": "20", "vbr": "on",
            "compression_level": "5"}
    # ⚠ layout="mono" is load-bearing: an audio stream defaults to STEREO, and a
    # stereo libopus stream spends the bitrate budget on two channels, so
    # `bit_rate` stops tracking (verified: 48000 requested -> 22.6 kbps actual,
    # and 12k/24k/48k all produced identical distortion). Mono also matches what
    # WebRTC publishes for a microphone track.
    try:
        stream = out.add_stream("libopus", rate=48000, layout="mono", options=opts)
    except TypeError:  # older PyAV: no options kwarg on add_stream
        stream = out.add_stream("libopus", rate=48000, layout="mono")
    stream.bit_rate = int(bitrate)
    frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
    frame.sample_rate = 48000
    frame.pts = 0
    for pkt in stream.encode(frame):
        out.mux(pkt)
    for pkt in stream.encode(None):
        out.mux(pkt)
    out.close()

    buf.seek(0)
    dec = av.open(buf, mode="r", format="ogg")
    res = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=48000)
    chunks = []
    try:
        for fr in dec.decode(audio=0):
            for rf in res.resample(fr):
                chunks.append(rf.to_ndarray().reshape(-1))
        for rf in res.resample(None):  # flush
            chunks.append(rf.to_ndarray().reshape(-1))
    finally:
        dec.close()
    if not chunks:
        raise RuntimeError("opus decode produced no audio")
    return np.concatenate(chunks).astype(np.float64) / 32768.0


def _opus_ffmpeg(x48, bitrate: int, application: str):
    import numpy as np
    raw = np.clip(np.asarray(x48, dtype=np.float64), -1.0, 1.0).astype(np.float32).tobytes()
    enc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "f32le", "-ar", "48000", "-ac", "1", "-i", "pipe:0",
         "-c:a", "libopus", "-b:a", str(int(bitrate)), "-application", application,
         "-frame_duration", "20", "-vbr", "on", "-compression_level", "5",
         "-ac", "1", "-f", "ogg", "pipe:1"],
        input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    dec = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "ogg", "-i", "pipe:0",
         "-f", "f32le", "-ar", "48000", "-ac", "1", "pipe:1"],
        input=enc, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    return np.frombuffer(dec, dtype=np.float32).astype(np.float64)


def _opus_dsp(x, sr: int, bitrate: int):
    """Approximation used only when no real codec is available: SILK/hybrid
    bandwidth limiting + envelope-shaped coding noise (noise-feedback coding
    keeps a roughly constant LOCAL SNR, which is why plain additive noise is a
    poor stand-in)."""
    import numpy as np
    from scipy.signal import butter, lfilter
    x = np.asarray(x, dtype=np.float64)
    cutoff = 6000 if bitrate <= 14000 else 8000 if bitrate <= 20000 else 12000 if bitrate <= 28000 else 16000
    nyq = 0.5 * sr
    if cutoff < nyq * 0.99:
        b, a = butter(6, cutoff / nyq, btype="low")
        x = lfilter(b, a, x)
    snr = 12.0 + bitrate / 4000.0
    w = max(1, int(0.01 * sr))
    env = np.convolve(np.abs(x), np.ones(w) / w, mode="same") + 1e-6
    return x + np.random.normal(0.0, 1.0, x.shape) * env / (10 ** (snr / 20.0))


def _opus_roundtrip(x, sr: int, bitrate: int = 48000, application: str = "voip"):
    """Encode/decode through libopus the way WebRTC does — mono, 48 kHz, 20 ms
    frames, VOIP application. Returns the waveform at the original sr/length."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    backend = _opus_backend()
    if backend == "dsp":
        return _opus_dsp(x, sr, bitrate)
    try:
        x48 = _resample(x, sr, 48000, "HQ")
        y48 = _opus_av(x48, bitrate, application) if backend == "av" else _opus_ffmpeg(x48, bitrate, application)
        return _fit(_resample(y48, 48000, sr, "HQ"), n)
    except Exception as e:  # noqa: BLE001
        log(f"[augment] opus round-trip via {backend} failed ({e}); using the DSP approximation")
        return _opus_dsp(x, sr, bitrate)


def _g711(x, law: str = "u"):
    """G.711 companding + 8-bit quantisation (PCMU / PCMA), the codec LiveKit SIP
    offers by default on a trunk."""
    import numpy as np
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    if law == "u":
        mu = 255.0
        y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
        q = np.round(y * 127.0) / 127.0
        return np.sign(q) * ((1.0 + mu) ** np.abs(q) - 1.0) / mu
    A = 87.6
    la = 1.0 + np.log(A)
    ax = np.abs(x)
    y = np.sign(x) * np.where(ax < 1.0 / A, A * ax / la, (1.0 + np.log(np.maximum(A * ax, 1e-12))) / la)
    q = np.round(y * 127.0) / 127.0
    aq = np.abs(q)
    return np.sign(q) * np.where(aq < 1.0 / la, aq * la / A, np.exp(aq * la - 1.0) / A)


# --- WebRTC audio processing module ----------------------------------------
def _highpass(x, sr: int, fc: float = 80.0):
    """The APM's high-pass filter — first thing WebRTC does to a capture frame."""
    from scipy.signal import butter, lfilter
    b, a = butter(2, max(fc, 1.0) / (0.5 * sr), btype="high")
    return lfilter(b, a, x)


def _agc(x, sr: int, target_dbfs: float = -3.0, max_gain_db: float = 30.0,
         initial_gain_db: float = 8.0, rate_db_per_s: float = 3.0,
         max_noise_dbfs: float = -50.0):
    """WebRTC AGC2 adaptive-digital gain, on 10 ms sub-frames.

    The artefact that matters for ASR is the SLEW RATE: libwebrtc caps gain
    movement at ~3 dB/s, so the first second or two of a quiet speaker arrives
    10–20 dB under-levelled and then swells. LiveKit applies this TWICE — once in
    the browser (autoGainControl: true) and again agent-side
    (AudioProcessingModule(auto_gain_control=True)) — so the chain below runs it
    twice too. The noise-floor tracker caps the gain so the background is never
    pushed past max_output_noise_level_dbfs, and the fixed limiter closes it out."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    hop = max(1, int(0.01 * sr))
    nf = int(np.ceil(n / hop))
    if nf == 0:
        return x
    fr = np.pad(x, (0, nf * hop - n)).reshape(nf, hop)
    peak_db = 20.0 * np.log10(np.maximum(np.max(np.abs(fr), axis=1), 1e-9))
    rms = np.sqrt(np.mean(fr ** 2, axis=1))

    noise = np.empty(nf)                       # min-statistics floor tracker
    cur = float(rms[0])
    rise = 1.0 + 3.0 * hop / sr                # ~+3 %/s when no new minimum arrives
    for i in range(nf):
        cur = float(rms[i]) if rms[i] < cur else cur * rise
        noise[i] = cur
    noise_db = 20.0 * np.log10(np.maximum(noise, 1e-9))

    want = np.clip(target_dbfs - peak_db, 0.0, max_gain_db)
    want = np.minimum(want, np.maximum(max_noise_dbfs - noise_db, 0.0))

    step = rate_db_per_s * hop / sr
    g = np.empty(nf)
    cur = float(initial_gain_db)
    for i in range(nf):
        cur += float(np.clip(want[i] - cur, -step, step))
        g[i] = cur

    y = x * np.repeat(10.0 ** (g / 20.0), hop)[:n]
    lim = 10.0 ** (-1.0 / 20.0)                # AGC2's fixed limiter, soft knee
    over = np.abs(y) > lim
    if over.any():
        a = np.abs(y[over])
        y[over] = np.sign(y[over]) * (lim + (a - lim) / (1.0 + (a - lim) * 12.0))
    return y


def _spectral_suppress(x, sr: int, over_sub: float = 1.6, floor_db: float = -16.0,
                       musical: float = 0.35, n_fft: int = 512):
    """Noise suppression / voice isolation artefacts.

    LiveKit's browser defaults turn on BOTH noiseSuppression and the much more
    aggressive voiceIsolation, and agents can add Krisp BVC on top. All three are
    spectral suppressors, and all three fail the same way on speech: weak
    unvoiced energy (/s/ /f/ /th/, plosive bursts, the tail of a word) sits near
    the noise floor and gets gated away, and the residual gain fluctuation leaves
    musical noise. That is modelled as a Wiener gain raised to an over-subtraction
    power, with per-bin gain jitter."""
    import numpy as np
    from scipy.signal import istft, stft
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    ov = n_fft * 3 // 4
    _, _, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=ov)
    mag = np.abs(Z)
    noise = np.percentile(mag, 10, axis=1, keepdims=True) + 1e-9
    snr = np.maximum((mag / noise) ** 2 - 1.0, 1e-6)
    gain = (snr / (1.0 + snr)) ** over_sub
    gain = np.maximum(gain, 10.0 ** (floor_db / 20.0))
    if musical > 0:
        gain = np.clip(gain * np.exp(np.random.normal(0.0, musical, gain.shape)), 0.0, 1.0)
    _, y = istft(Z * gain, fs=sr, nperseg=n_fft, noverlap=ov)
    return _fit(y, n)


def _aec_doubletalk(x, sr: int):
    """Acoustic-echo-canceller behaviour during barge-in.

    While the agent is speaking, its TTS comes back through the caller's mic. The
    AEC's non-linear processor removes most of it — and takes a chunk of the
    caller's own speech with it (that is what makes barge-in transcripts drop
    words), leaving a low-level filtered residual behind. Modelled as: pick a few
    'agent is talking' windows, attenuate the near-end inside them, and add a
    scrambled, low-passed, heavily attenuated copy as residual echo (scrambled so
    it is babble, never transcribable words the label does not contain)."""
    import numpy as np
    from scipy.signal import butter, lfilter
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < int(0.6 * sr):
        return x
    blk = max(1, int(random.uniform(0.04, 0.08) * sr))
    nb = n // blk
    if nb < 2:
        return x
    order = np.random.permutation(nb)
    residual = x[: nb * blk].reshape(nb, blk)[order].reshape(-1)
    residual = _fit(residual, n)
    b, a = butter(4, min(2600.0, 0.45 * sr) / (0.5 * sr), btype="low")
    residual = lfilter(b, a, residual)
    rms = float(np.sqrt(np.mean(x ** 2))) or 1e-9
    rres = float(np.sqrt(np.mean(residual ** 2))) or 1e-9
    residual *= (rms / rres) * 10 ** (random.uniform(-45.0, -30.0) / 20.0)

    y = x.copy()
    for _ in range(random.randint(1, 3)):
        w = int(random.uniform(0.3, 1.5) * sr)
        if w >= n:
            continue
        a0 = random.randint(0, n - w)
        att = 10 ** (-random.uniform(4.0, 16.0) / 20.0)
        ramp = min(int(0.02 * sr), w // 2)
        env = np.full(w, att)
        if ramp > 1:                            # the NLP opens/closes over ~20 ms
            env[:ramp] = np.linspace(1.0, att, ramp)
            env[-ramp:] = np.linspace(att, 1.0, ramp)
        y[a0:a0 + w] *= env
    return y + residual


# --- transport -------------------------------------------------------------
def _packet_loss(x, sr: int, loss: float = 0.03, mean_burst: float = 2.0,
                 fec_recovery: float = 0.5, ptime: float = 0.02):
    """RTP packet loss with Opus PLC, on the 20 ms packet grid.

    Loss is bursty (Gilbert-Elliott), not independent — that is what a real
    network does. LiveKit publishes with `red: true`, so a fraction of losses are
    recovered from the redundant copy and never reach the decoder; the rest are
    concealed, and Opus PLC EXTRAPOLATES the last pitch period with a decaying
    gain rather than inserting silence. (The existing `dropout` technique zeroes
    chunks, which is the wrong artefact — the model needs to see extrapolated
    audio, not a hole.)

    Note: this is a waveform-domain model of the decoder's output. Actually
    dropping packets ahead of libopus would need the decoder's own PLC entry
    point, which is not reachable through ffmpeg."""
    import numpy as np
    src = np.asarray(x, dtype=np.float64)
    y = src.copy()
    n = len(src)
    step = max(1, int(ptime * sr))
    q = 1.0 / max(1.0, mean_burst)                      # bad -> good
    p = loss * q / max(1e-6, 1.0 - loss)                # good -> bad
    bad = False
    for i in range(int(np.ceil(n / step))):
        bad = (np.random.rand() > q) if bad else (np.random.rand() < p)
        if not bad or np.random.rand() < fec_recovery:
            continue
        a, b = i * step, min(n, (i + 1) * step)
        m = b - a
        if m <= 0:
            continue
        prev = y[max(0, a - m):a]
        seg = (prev[-m:] * np.linspace(0.7, 0.15, m)) if len(prev) == m else np.zeros(m)
        y[a:b] = seg
        xf = min(int(0.003 * sr), m)                    # overlap-add back to real audio
        if xf > 1:
            w = np.linspace(0.0, 1.0, xf)
            y[b - xf:b] = seg[-xf:] * (1.0 - w) + src[b - xf:b] * w
    return y


def _dtx(x, sr: int, min_gap_ms: float = 200.0):
    """Opus DTX (LiveKit publishes with `dtx: true`).

    During silence the encoder stops sending and the decoder synthesises comfort
    noise, so the real room tone between words is replaced by stationary hiss at
    a held level, with hard transitions at each end. A model trained only on
    natural pauses has never seen that."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    step = max(1, int(0.02 * sr))
    nf = int(np.ceil(n / step))
    if nf < 2:
        return x
    fr = np.pad(x, (0, nf * step - n)).reshape(nf, step)
    e = np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-12
    thr = max(float(np.percentile(e, 20)) * 2.0, float(np.max(e)) * 0.02)
    speech = e > thr
    for i in np.flatnonzero(speech):                    # hangover: DTX lags the drop
        speech[i:i + 6] = True
    out = np.pad(x, (0, nf * step - n))
    quiet = ~speech
    lvl = float(np.median(e[quiet])) if quiet.any() else 1e-4
    min_len = int(np.ceil(min_gap_ms / 20.0))
    i = 0
    while i < nf:
        if speech[i]:
            i += 1
            continue
        j = i
        while j < nf and not speech[j]:
            j += 1
        if j - i >= min_len:
            out[i * step:j * step] = np.random.normal(0.0, lvl, (j - i) * step)
        i = j
    return out[:n]


def _lk_resample_chain(x, sr: int):
    """The resampling LiveKit actually performs: the track arrives at 48 kHz,
    room_io resamples to the STT plugin's rate, and the OpenAI-compatible plugin
    uses SAMPLE_RATE = 24000 — so the server receives 24 kHz and resamples again
    to Whisper's 16 kHz."""
    y = _resample(x, sr, 48000, "MQ")
    y = _resample(y, 48000, 24000, "MQ")
    y = _resample(y, 24000, 16000, "MQ")
    return _fit(_resample(y, 16000, sr, "MQ"), len(x))


def _speech_bounds(x, sr: int):
    """(start, end) sample indices of speech, on a 20 ms energy grid. None when
    nothing crosses the threshold."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    step = max(1, int(0.02 * sr))
    nf = int(np.ceil(n / step))
    if nf < 2:
        return None
    fr = np.pad(x, (0, nf * step - n)).reshape(nf, step)
    e = np.sqrt(np.mean(fr ** 2, axis=1))
    thr = max(float(np.max(e)) * 0.02, float(np.percentile(e, 10)) * 3.0)
    idx = np.flatnonzero(e > thr)
    if idx.size == 0:
        return None
    return int(idx[0]) * step, min(n, (int(idx[-1]) + 1) * step)


def _vad_tighten(x, sr: int, max_pad_ms: float = 250.0):
    """Silero VAD endpointing, as the StreamAdapter applies it.

    The STT never sees a whole recording — it sees one VAD segment, cut close to
    the speech with hard (un-faded) edges. Clean corpora usually carry generous
    room tone at both ends, so the model can arrive at production having never
    seen an utterance that starts on the first phoneme.

    Deliberately conservative: it only trims SILENCE. Cutting into speech while
    keeping the full transcript is how a finetune is taught to hallucinate the
    missing words, so that is not done here."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    bounds = _speech_bounds(x, sr)
    if bounds is None:
        return x
    s, e = bounds
    head = int(random.uniform(0.0, max_pad_ms / 1000.0) * sr)
    tail = int(random.uniform(0.0, max_pad_ms / 1000.0) * sr)
    return x[max(0, s - head):min(len(x), e + tail)].copy()


def _background(x, sr: int, snr_db: float):
    """Pink-ish room tone at a target SNR — the noise floor the capture chain
    then has to suppress (and the thing DTX replaces with comfort noise)."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    spec = np.fft.rfft(np.random.normal(0.0, 1.0, n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    noise = np.fft.irfft(spec / np.sqrt(f), n=n)
    rms_x = float(np.sqrt(np.mean(x ** 2))) or 1e-9
    rms_n = float(np.sqrt(np.mean(noise ** 2))) or 1e-9
    return x + noise * (rms_x / rms_n) / (10 ** (snr_db / 20.0))


# ===========================================================================
# The STREAMING regime — what the VAD + a hesitant speaker do (added 2026-08-06)
# ===========================================================================
# Everything above models the TRANSPORT. The out-of-tree LiveKit STT benchmark
# (a live livekit-server, real WebRTC/Opus, silero VAD, the livekit-plugins-openai
# STT, scored by the same scorer as its batch arm; maintained separately and the
# source of truth for these figures) then measured that transport is NOT where the
# accuracy goes:
#
#   channel only (codec + noise + gain)      4.98% -> 5.33%   (+0.35 pp)
#   + the streaming pipeline, fluent         5.33% -> 7.23%   (+1.90 pp)
#   + the streaming pipeline, HESITANT       6.64% -> 13.39%  (+6.75 pp)
#
# Two findings out of that, and both are trainable properties of the MODEL
# rather than of the pipeline:
#
#   1. Internal silence hurts whisper on its own. The BATCH arm degrades
#      5.33% -> 6.64% when two 0.7 s pauses are inserted mid-utterance, with no
#      LiveKit anywhere in the path. That 1.31 pp is ours to fix here.
#   2. A segment is framed, not tight. Silero hands the STT the utterance plus
#      `prefix_padding_duration` at the front and the `min_silence_duration`
#      hangover at the back — a 3.95 s utterance arrives as a 4.92 s segment —
#      and a DNN enhancer in front of it drives the non-speech to -70 dB. A
#      model that only ever saw tightly-cut clips answers that quiet with
#      phantom text (the benchmark's #2 suspicion for the production gap).
#
# The rest of the streaming cost is over-segmentation: silero splits the turn
# and each fragment is transcribed with no shared context, which destroys a
# number straddling the boundary (`charged 99, two times` -> `charged 92
# times`). That one is NOT modelled here and cannot be, under this file's
# label-preserving rule: cutting the audio at a segment boundary means cutting
# the transcript too, which needs word-level alignment, i.e. a dataset
# transform rather than a waveform augmentation. Serving-side, the fix measured
# best anyway is the VAD's own `min_silence_duration` 0.55 -> 0.9.
# ===========================================================================
_WHISPER_WINDOW_S = 30.0


def _room_left(x, sr: int) -> float:
    """Seconds that may be ADDED before the clip passes whisper's 30 s window.

    The feature extractor truncates at 30 s, so a technique that lengthens audio
    can silently push real speech out of the window while the label still claims
    every word — the same hallucination lesson `vad_clip` avoids, arrived at from
    the other direction. Every stage below that inserts time asks this first."""
    return max(0.0, _WHISPER_WINDOW_S - len(x) / float(sr))


def _room_tone(x, sr: int, n: int, level_scale: float = 1.0):
    """n samples of pink room tone at the clip's OWN noise floor.

    Used to fill time this section inserts. Digital zeros are the wrong filler
    for anything a microphone produced: a real pause, a real pre-roll and a real
    hangover all carry the room, and DTX/comfort noise is the only stage that
    ever hands the model true stationary hiss."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    step = max(1, int(0.02 * sr))
    nf = len(x) // step
    if nf >= 1:
        e = np.sqrt(np.mean(x[: nf * step].reshape(nf, step) ** 2, axis=1))
        floor = float(np.percentile(e, 10))
    else:
        floor = 0.0
    floor = max(floor, 10.0 ** (-70.0 / 20.0)) * level_scale
    if n <= 0:
        return np.zeros(0)
    if n < 8:
        return np.random.normal(0.0, floor, n)
    spec = np.fft.rfft(np.random.normal(0.0, 1.0, n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    tone = np.fft.irfft(spec / np.sqrt(f), n=n)
    rms = float(np.sqrt(np.mean(tone ** 2))) or 1e-9
    return tone * (floor / rms)


def _insert_pauses(x, sr: int, count: int, duration: float, fill: str = "tone"):
    """Mid-utterance hesitation: `count` pauses of `duration` s at the quietest
    inter-word points.

    Real speakers stop and restart; recorded corpora (and TTS especially) don't,
    which is why a finetune meets its first long internal silence in production.
    Costs whisper 1.31 pp in BATCH decoding — before any pipeline is involved —
    and drives the streaming cost from +1.90 to +6.75 pp because a pause past
    `min_silence_duration` makes the VAD close the turn and reopen.

    Deliberately the SAME selection rule as the benchmark's own
    `--pause-count/--pause-duration` transform (20 ms energy grid, lowest-energy
    frames first, >=0.5 s apart, outer 15 % of the clip avoided), so the hesitant
    arm of the benchmark measures the condition training actually saw. Randomised
    here where the benchmark is deterministic — it needs byte-identical audio
    across arms, this needs variety."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if count <= 0 or duration <= 0:
        return x
    duration = min(duration, _room_left(x, sr))
    if duration < 0.05:
        return x
    count = min(count, int(_room_left(x, sr) / duration))
    hop = max(1, int(0.02 * sr))
    nf = len(x) // hop
    if count < 1 or nf < 10:
        return x
    energy = np.abs(x[: nf * hop].reshape(nf, hop)).mean(axis=1)
    lo, hi = int(0.15 * nf), int(0.85 * nf)
    if hi <= lo:
        return x
    min_sep = max(1, int(0.5 * sr / hop))
    chosen: list[int] = []
    for i in sorted(range(lo, hi), key=lambda k: energy[k]):
        if all(abs(i - j) >= min_sep for j in chosen):
            chosen.append(i)
            if len(chosen) == count:
                break
    gap = int(sr * duration)
    parts, prev = [], 0
    for i in sorted(chosen):
        pause = np.zeros(gap) if fill == "silence" else _room_tone(x, sr, gap)
        parts += [x[prev:i * hop], pause]
        prev = i * hop
    parts.append(x[prev:])
    return np.concatenate(parts)


def _segment_frame(x, sr: int, head_s: float, tail_s: float):
    """The VAD segment as `StreamAdapter` actually delivers it: the utterance plus
    silero's `prefix_padding_duration` of pre-roll and its `min_silence_duration`
    hangover, hard-edged at both ends.

    This is the far end of the same axis as `vad_clip`, and it is the common case:
    a turn silero keeps whole arrives wrapped in ~1.4 s of NON-SPEECH (measured:
    3.95 s utterance -> 4.92 s segment), only a split one arrives cut tight. Real
    leading/trailing audio is reused wherever the clip has some — the pre-roll a
    microphone captured is room tone, not silence — and only the shortfall is
    synthesised."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    bounds = _speech_bounds(x, sr)
    if bounds is None:
        return x
    s, e = bounds
    want_head = int(max(0.0, head_s) * sr)
    want_tail = int(max(0.0, tail_s) * sr)
    real_head = min(s, want_head)
    real_tail = min(len(x) - e, want_tail)
    core = x[s - real_head:e + real_tail]
    room = max(0, int(_WHISPER_WINDOW_S * sr) - len(core))
    add_head = min(want_head - real_head, room // 2)
    add_tail = min(want_tail - real_tail, room - add_head)
    return np.concatenate([_room_tone(x, sr, add_head), core,
                           _room_tone(x, sr, add_tail)])


def _receive_ramp(x, sr: int, ramp_s: float, depth_db: float, hard_onset: bool):
    """WebRTC receive ramp-up over the first words of a segment.

    Found in the benchmark harness the hard way: publishing a clip that starts
    talking at t=0 lost or corrupted the FIRST WORD on 80/100 clips (`Ya saya ni`
    -> `Saya ini`, `She's actually` -> `Actually`) — Opus/jitter-buffer priming
    right after subscription, plus silero's prefix padding having no pre-roll to
    prepend. The harness pads a second of silence in front precisely so it stops
    measuring this; production still has it, at session start and after a long
    silence, so the model should have seen it.

    ATTENUATION ONLY, never deletion. A hole where the onset phoneme was, with
    the word still in the label, is how a finetune is taught to invent it."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if hard_onset:
        bounds = _speech_bounds(x, sr)
        if bounds is not None:
            keep = int(random.uniform(0.0, 0.04) * sr)
            x = x[max(0, bounds[0] - keep):].copy()
    n = min(len(x), max(1, int(max(ramp_s, 0.0) * sr)))
    if n < 2:
        return x
    y = x.copy()
    y[:n] *= 10.0 ** (np.linspace(-abs(depth_db), 0.0, n) / 20.0)
    return y


def _deep_denoise(x, sr: int, over_sub: float = 3.5, floor_db: float = -60.0,
                  gate_dbfs: float = -68.0):
    """A single-channel DNN speech enhancer sitting in front of the model — GTCRN
    (the self-hosted plugin, aimed at inbound SIP), Krisp BVC on LiveKit Cloud,
    RNNoise. Much more aggressive than `webrtc_ns`, which models the browser's own
    NS/voiceIsolation.

    Measured on the real one: it WORKS, dropping the p25 frame level from
    -28.9 dB to -70.9 dB while leaving speech peaks within 0.2 dB — and it lost
    WER on every arm (7.23 -> 7.36 fluent, 13.39 -> 15.13 hesitant, 11.00 ->
    12.52 noisy), because whisper tolerates additive noise better than
    enhancement artefacts and answers dead-quiet non-speech with invented content
    (`Can I talk to her later on the line?` -> `My operator was not on the
    line.`). The serving-side conclusion is don't enable it; the TRAINING-side
    conclusion is that a model which has to survive one should have seen its
    artefacts. So: deep over-subtraction, a floor low enough to punch holes in weak
    fricatives, LOW musical noise (a DNN enhancer leaves spectral holes, not the
    musical residue a subtractive filter leaves), non-speech driven down with ~10 ms
    ramps, and the speech peak restored.

    ⚠ `gate_dbfs` is an ABSOLUTE target frame level, not another attenuation.
    Attenuating non-speech by a further N dB stacks with the spectral floor and
    lands 20 dB below anything real (measured -90 dB when calibrating against a
    real segment); the enhancer's signature is a floor AT roughly -70 dBFS, ~63 dB
    under the speech peak, so this scales the non-speech to hit that instead."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    peak0 = float(np.max(np.abs(x)))
    y = _spectral_suppress(x, sr, over_sub=over_sub, floor_db=floor_db,
                           musical=random.uniform(0.02, 0.18))
    step = max(1, int(0.02 * sr))
    nf = int(np.ceil(len(x) / step))
    if nf >= 2:
        fr = np.pad(x, (0, nf * step - len(x))).reshape(nf, step)
        e = np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-12
        thr = max(float(np.percentile(e, 25)) * 3.0, float(np.max(e)) * 0.03)
        speech = e > thr
        for i in np.flatnonzero(speech):        # attack + hangover frames
            speech[max(0, i - 1):i + 4] = True
        quiet = ~speech
        if quiet.any():
            fy = np.pad(y, (0, nf * step - len(y)))[: nf * step].reshape(nf, step)
            lvl = float(np.median(np.sqrt(np.mean(fy[quiet] ** 2, axis=1)))) or 1e-12
            g = min(1.0, 10.0 ** (gate_dbfs / 20.0) / lvl)   # target, never amplify
            env = np.repeat(np.where(speech, 1.0, g), step)[: len(y)]
            k = max(1, int(0.01 * sr))          # the gate opens/closes over ~10 ms
            y = y * np.convolve(env, np.ones(k) / k, mode="same")
    peak1 = float(np.max(np.abs(y)))
    if peak0 > 0 and peak1 > 0:
        y = y * (peak0 / peak1)                 # enhancers hold speech peaks (0.2 dB)
    return y


def _neteq_warp(x, sr: int, n_ops: int, max_op_ms: float):
    """WebRTC's jitter buffer (NetEq) absorbing network jitter: to hold the playout
    buffer near target it ACCELERATES (overlap-add merges two windows into one) or
    PRE-EMPTIVELY EXPANDS (plays a crossfaded duplicate) a few tens of ms at a
    time, so the audio reaching the VAD is mildly time-warped rather than a clean
    copy of what was captured.

    ⚠ Unlike the rest of this section this one is MODELLED, not measured — the
    benchmark runs livekit-server on localhost, where there is no jitter to
    absorb, so it cannot see the artefact and cannot price it either.

    Bounded to <=2 % of the clip (NetEq's budget is tens of ms, not seconds) and
    applied at the lowest-energy windows, which is roughly where its correlation
    search lands anyway — so nothing merges a phoneme away."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    win = max(1, int(min(max_op_ms, 60.0) / 1000.0 * sr))
    budget = int(min(0.02 * n, 0.3 * sr))
    if n < 4 * win or budget < win:
        return x
    nf = n // win
    e = np.sqrt(np.mean(x[: nf * win].reshape(nf, win) ** 2, axis=1))
    picks: list[int] = []
    used = 0
    for i in np.argsort(e):
        i = int(i)
        if i < 1 or i >= nf - 2 or any(abs(i - j) < 2 for j in picks):
            continue
        picks.append(i)
        used += win
        if len(picks) >= n_ops or used + win > budget:
            break
    if not picks:
        return x
    w = np.linspace(0.0, 1.0, win)
    parts, prev = [], 0
    for i in sorted(picks):
        a, b = i * win, (i + 1) * win
        parts.append(x[prev:a])
        if random.random() < 0.5:                       # accelerate: merge 2 -> 1
            parts.append(x[a:b] * (1.0 - w) + x[b:b + win] * w)
            prev = b + win
        else:                                           # expand: 1 -> 2
            parts += [x[a:b], x[a:b] * (1.0 - w) + x[b:b + win] * w]
            prev = b
    parts.append(x[prev:])
    return np.concatenate(parts)


# --- the composites --------------------------------------------------------
def _aug_opus(x, sr):
    """Opus round-trip at a LiveKit AudioPreset bitrate (12/24/48 kbps, mono)."""
    return _opus_roundtrip(x, sr, random.choice([12000, 24000, 24000, 48000, 48000]), "voip")


def _aug_g711(x, sr):
    """G.711 μ-law/A-law at 8 kHz — a LiveKit SIP trunk leg."""
    import numpy as np
    n = len(x)
    y = _resample(x, sr, 8000, "HQ")
    y = _g711(y, random.choice(["u", "u", "a"]))
    return _fit(_resample(y, 8000, sr, "HQ"), n)


def _aug_packet_loss(x, sr):
    """Bursty RTP loss + Opus PLC, 0.5–8 % with RED recovery."""
    return _packet_loss(x, sr, loss=random.uniform(0.005, 0.08),
                        mean_burst=random.uniform(1.0, 4.0),
                        fec_recovery=random.uniform(0.2, 0.7))


def _aug_dtx(x, sr):
    """Opus DTX: comfort noise in place of the real background."""
    return _dtx(x, sr, min_gap_ms=random.uniform(160.0, 400.0))


def _aug_agc(x, sr):
    """WebRTC AGC2 adaptive digital gain (slew-limited, then limited)."""
    return _agc(x, sr, target_dbfs=random.uniform(-6.0, -2.0),
                initial_gain_db=random.uniform(0.0, 12.0),
                rate_db_per_s=random.uniform(2.0, 6.0))


def _aug_webrtc_ns(x, sr):
    """noiseSuppression + voiceIsolation / Krisp BVC artefacts."""
    return _spectral_suppress(x, sr, over_sub=random.uniform(1.1, 2.4),
                              floor_db=random.uniform(-24.0, -10.0),
                              musical=random.uniform(0.15, 0.5))


def _aug_aec(x, sr):
    """AEC non-linear-processor gating during barge-in + residual echo."""
    return _aec_doubletalk(x, sr)


def _aug_vad_clip(x, sr):
    """Silero VAD endpointing — tight, hard-edged segment boundaries."""
    return _vad_tighten(x, sr, max_pad_ms=random.uniform(60.0, 400.0))


def _aug_resample_chain(x, sr):
    """The 48 k → 24 k → 16 k soxr chain the agent + STT plugin perform."""
    return _lk_resample_chain(x, sr)


def _aug_hesitation(x, sr):
    """1–3 mid-utterance pauses at the quietest inter-word points (0.3–1.2 s).

    The range straddles silero's `min_silence_duration` (0.55 default, 0.9 tuned)
    on purpose: some pauses split the turn in production and some don't, and the
    model has to read both."""
    return _insert_pauses(x, sr, random.randint(1, 3), random.uniform(0.3, 1.2),
                          fill=random.choice(["tone", "tone", "silence"]))


def _aug_vad_pad(x, sr):
    """Silero's segment framing — prefix pad (0.3–0.9 s) + min-silence hangover
    (0.4–1.0 s) of room tone, hard-edged. The `vad_clip` counterpart."""
    return _segment_frame(x, sr, random.uniform(0.3, 0.9), random.uniform(0.4, 1.0))


def _aug_rampup(x, sr):
    """WebRTC receive ramp-up: the first 60–300 ms attenuated by 6–24 dB, usually
    with the segment starting on the first phoneme (no pre-roll to pad with)."""
    return _receive_ramp(x, sr, random.uniform(0.06, 0.30),
                         random.uniform(6.0, 24.0), random.random() < 0.7)


def _aug_denoiser(x, sr):
    """DNN speech enhancement in front of the model (GTCRN / Krisp BVC / RNNoise)."""
    return _deep_denoise(x, sr, over_sub=random.uniform(2.5, 5.0),
                         floor_db=random.uniform(-70.0, -45.0),
                         gate_dbfs=random.uniform(-75.0, -55.0))


def _aug_room_tone(x, sr):
    """Pink room/line tone at 6–30 dB SNR — the noisy-call regime the benchmark
    measures with `--noise-snr`. Distinct from `noise`, which is white Gaussian at
    10–40 dB; pink has its energy where speech does, like a real room or trunk."""
    return _background(x, sr, random.uniform(6.0, 30.0))


def _aug_jitter(x, sr):
    """NetEq accelerate / pre-emptive-expand time warping (<=2 % of the clip)."""
    return _neteq_warp(x, sr, n_ops=random.randint(1, 4),
                       max_op_ms=random.uniform(20.0, 60.0))


def _aug_livekit(x, sr):
    """THE ONE TO USE for a WebRTC voice agent: the whole LiveKit capture →
    publish → agent → STT chain, in order, with each stage randomised per sample.
    Roughly what a browser/mobile caller's audio survives before it reaches the
    model."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if random.random() < 0.7:                                   # the room
        x = _background(x, sr, random.uniform(8.0, 32.0))
    x = _highpass(x, sr, random.uniform(60.0, 120.0))           # APM high-pass
    if random.random() < 0.85:                                  # NS + voiceIsolation
        x = _aug_webrtc_ns(x, sr)
    if random.random() < 0.25:                                  # barge-in
        x = _aec_doubletalk(x, sr)
    if random.random() < 0.9:                                   # browser AGC
        x = _agc(x, sr, target_dbfs=random.uniform(-6.0, -2.0),
                 initial_gain_db=random.uniform(0.0, 12.0))
    x = _opus_roundtrip(x, sr, random.choice([12000, 24000, 24000, 48000, 48000]), "voip")
    if random.random() < 0.55:                                  # loss + PLC
        x = _packet_loss(x, sr, loss=random.uniform(0.005, 0.06),
                         mean_burst=random.uniform(1.0, 3.5),
                         fec_recovery=random.uniform(0.3, 0.7))
    if random.random() < 0.5:                                   # DTX comfort noise
        x = _dtx(x, sr, min_gap_ms=random.uniform(160.0, 400.0))
    if random.random() < 0.9:                                   # agent-side AGC2
        x = _agc(x, sr, target_dbfs=random.uniform(-5.0, -2.0),
                 initial_gain_db=random.uniform(0.0, 8.0))
    x = _lk_resample_chain(x, sr)
    if random.random() < 0.6:                                   # VAD segment edges
        x = _vad_tighten(x, sr, max_pad_ms=random.uniform(60.0, 400.0))
    return x


def _aug_livekit_stream(x, sr):
    """`livekit` PLUS the streaming regime the benchmark measured: a speaker who
    hesitates mid-turn, a DNN enhancer that may be in the path, NetEq warping, and
    the VAD segment framed the way `StreamAdapter` actually delivers it.

    **This is the one to use for a voice agent** whose callers are people rather
    than TTS. `livekit` models the transport, which the benchmark priced at ~0.35 pp;
    this models the part that costs +6.75 pp. It is left as a SEPARATE technique
    rather than folded into `livekit` so an earlier run's augmentation is still
    reproducible — pick both if you want the plain-transport draws too.

    Order is the real signal order: the speaker hesitates before the room is
    recorded, the room before the browser processes it, the segment is framed last,
    and the receive ramp lands on the front of the segment that reaches the STT."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if random.random() < 0.6:                                   # the speaker
        x = _aug_hesitation(x, sr)
    if random.random() < 0.7:                                   # the room
        x = _aug_room_tone(x, sr)
    x = _highpass(x, sr, random.uniform(60.0, 120.0))            # APM high-pass
    r = random.random()
    if r < 0.3:                                                 # a DNN enhancer…
        x = _aug_denoiser(x, sr)
    elif r < 0.9:                                               # …or browser NS
        x = _aug_webrtc_ns(x, sr)
    if random.random() < 0.25:                                  # barge-in
        x = _aec_doubletalk(x, sr)
    if random.random() < 0.9:                                   # browser AGC
        x = _agc(x, sr, target_dbfs=random.uniform(-6.0, -2.0),
                 initial_gain_db=random.uniform(0.0, 12.0))
    x = _opus_roundtrip(x, sr, random.choice([12000, 24000, 24000, 48000, 48000]), "voip")
    if random.random() < 0.55:                                  # loss + PLC
        x = _packet_loss(x, sr, loss=random.uniform(0.005, 0.06),
                         mean_burst=random.uniform(1.0, 3.5),
                         fec_recovery=random.uniform(0.3, 0.7))
    if random.random() < 0.5:                                   # DTX comfort noise
        x = _dtx(x, sr, min_gap_ms=random.uniform(160.0, 400.0))
    if random.random() < 0.5:                                   # jitter buffer
        x = _aug_jitter(x, sr)
    if random.random() < 0.9:                                   # agent-side AGC2
        x = _agc(x, sr, target_dbfs=random.uniform(-5.0, -2.0),
                 initial_gain_db=random.uniform(0.0, 8.0))
    x = _lk_resample_chain(x, sr)
    if random.random() < 0.65:      # silero kept the turn whole -> framed segment
        x = _aug_vad_pad(x, sr)
    else:                           # a split segment arrives cut tight
        x = _vad_tighten(x, sr, max_pad_ms=random.uniform(60.0, 400.0))
    # Ramp-up last: it lands on the front of the segment that reaches the STT.
    # hard_onset is only half as likely as in `rampup` alone — the framing above
    # usually just gave this segment its pre-roll back.
    if random.random() < 0.45:
        x = _receive_ramp(x, sr, random.uniform(0.06, 0.30),
                          random.uniform(6.0, 24.0), random.random() < 0.5)
    return x


def _aug_livekit_sip(x, sr):
    """LiveKit SIP (inbound phone call): the PSTN band, a G.711 8 kHz trunk leg,
    then the TANDEM Opus re-encode LiveKit SIP performs to publish into the room.
    Strictly worse than `livekit` — use it if the agent takes phone calls."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if random.random() < 0.8:
        x = _background(x, sr, random.uniform(6.0, 26.0))
    x = _aug_bandpass(x, sr)                                    # analog line band
    x = _aug_g711(x, sr)                                        # PCMU/PCMA trunk leg
    if random.random() < 0.6:
        x = _packet_loss(x, sr, loss=random.uniform(0.005, 0.06),
                         mean_burst=random.uniform(1.0, 3.5),
                         fec_recovery=random.uniform(0.0, 0.3))
    # transcoded to Opus for the SFU — at the low presets, since a SIP leg carries
    # nothing above 4 kHz worth spending bits on
    x = _opus_roundtrip(x, sr, random.choice([12000, 24000, 24000]), "voip")
    if random.random() < 0.9:
        x = _agc(x, sr, target_dbfs=random.uniform(-5.0, -2.0),
                 initial_gain_db=random.uniform(0.0, 8.0))
    x = _lk_resample_chain(x, sr)
    if random.random() < 0.6:
        x = _vad_tighten(x, sr, max_pad_ms=random.uniform(60.0, 400.0))
    return x


_AUG_FUNCS = {
    "telephone": _aug_telephone,
    "noise": _aug_noise,
    "dropout": _aug_dropout,
    "gain": _aug_gain,
    "pitch": _aug_pitch,
    "speed": _aug_speed,
    "reverb": _aug_reverb,
    "bandpass": _aug_bandpass,
    # LiveKit / WebRTC transport (see the section above)
    "livekit": _aug_livekit,
    "livekit_sip": _aug_livekit_sip,
    "opus": _aug_opus,
    "g711": _aug_g711,
    "packet_loss": _aug_packet_loss,
    "dtx": _aug_dtx,
    "agc": _aug_agc,
    "webrtc_ns": _aug_webrtc_ns,
    "aec": _aug_aec,
    "vad_clip": _aug_vad_clip,
    "resample_chain": _aug_resample_chain,
    # The streaming regime — the VAD segment + a hesitant speaker
    "livekit_stream": _aug_livekit_stream,
    "hesitation": _aug_hesitation,
    "vad_pad": _aug_vad_pad,
    "rampup": _aug_rampup,
    "denoiser": _aug_denoiser,
    "room_tone": _aug_room_tone,
    "jitter": _aug_jitter,
}
# Stable list the API/form validate against.
AUG_TECHNIQUES = list(_AUG_FUNCS.keys())


def _augment_audio(data, sr: int, techniques):
    """Apply ONE randomly-chosen enabled technique to the waveform. Falls back to
    the untouched audio if the technique list is empty or a transform errors."""
    import numpy as np
    techs = [t for t in (techniques or []) if t in _AUG_FUNCS]
    if not techs:
        return np.asarray(data, dtype=np.float32)
    x = np.asarray(data, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    try:
        x = _AUG_FUNCS[random.choice(techs)](x, sr)
    except Exception as e:  # noqa: BLE001
        log(f"[augment] skipped ({e})")
        return np.asarray(data, dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


class _LazyAsrDataset:
    """Map-style dataset for HF Seq2SeqTrainer / torch DataLoader. Holds only the
    metadata items; __getitem__ fetches + decodes the audio for one index from
    its source (HF Arrow cache or S3) and returns {input_features, labels}. Plain
    class (not a torch subclass) so it stays picklable for DataLoader workers,
    which run __getitem__ in parallel and overlap audio I/O with GPU compute."""

    def __init__(self, items: list[dict], processor, work: str,
                 augment_techniques=None, augment_prob: float = 0.5):
        self.items = items
        self.processor = processor
        self.audio_dir = os.path.join(work, "audio")
        self.augment_techniques = [t for t in (augment_techniques or []) if t in _AUG_FUNCS]
        self.augment_prob = augment_prob

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        # A single unreachable/corrupt clip must not kill the whole trial — skip
        # to the next item (wrapping) so the batch stays full. Only raise if a
        # whole window is unloadable (a genuinely broken dataset).
        n = len(self.items)
        last_err = None
        for off in range(min(16, n)):
            it = self.items[(idx + off) % n]
            try:
                if it.get("src") == "hf":
                    a = it["hf_split"][it["hf_idx"]][it["audio_field"]]
                    array, sr = a["array"], a["sampling_rate"]
                else:
                    import librosa
                    os.makedirs(self.audio_dir, exist_ok=True)
                    path = _download_audio_s3(it["s3_spec"], it["s3_ref"], self.audio_dir)
                    if path is None:
                        raise RuntimeError(f"audio fetch failed for {it.get('s3_ref')!r}")
                    array, sr = librosa.load(path, sr=16000)  # decode + resample to 16k
                if self.augment_techniques and random.random() < self.augment_prob:
                    array = _augment_audio(array, sr, self.augment_techniques)
                feat = self.processor.feature_extractor(array, sampling_rate=sr).input_features[0]
                # Auto mode: it["text"] already carries the full Whisper prompt
                # (<|startoftranscript|><|lang|>…<|endoftext|>) with a per-utterance
                # language token, so don't let the tokenizer add its own. Fixed mode:
                # plain (cleaned) text — the processor prepends its configured prompt.
                add_special = not it.get("preformatted")
                labels = self.processor.tokenizer(it["text"], add_special_tokens=add_special).input_ids
                return {"input_features": feat, "labels": labels}
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        raise RuntimeError(f"no loadable audio near index {idx}: {last_err}")


# --------------------------------------------------------------------------
# Train / eval split — `split` column wins, else seeded hold-out.
# --------------------------------------------------------------------------
EVAL_SPLITS = {"test", "validation", "valid", "eval", "dev"}


def split_pairs(pairs: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    labelled = [p for p in pairs if p.get("split")]
    if labelled and any(p["split"].lower() in EVAL_SPLITS for p in labelled):
        train = [p for p in pairs if (p.get("split") or "train").lower() not in EVAL_SPLITS]
        ev = [p for p in pairs if (p.get("split") or "").lower() in EVAL_SPLITS]
        log(f"[split] using dataset split column: {len(train)} train / {len(ev)} eval")
        return train, ev
    # The user explicitly chose this dataset as its own test set, but it carries
    # no test/validation split column — fall back to a seeded hold-out and say so
    # loudly (rather than silently evaluating on rows it also trained on).
    if cfg.get("test_from_split"):
        log("[split] WARNING: test==training dataset but no `split` column found "
            f"(values seen: {sorted({(p.get('split') or '').lower() for p in pairs}) or 'none'}); "
            "falling back to a seeded hold-out.")
    import random

    pct = float(cfg.get("eval_split_pct", 10)) / 100.0
    rng = random.Random(int(cfg.get("split_seed", 42)))
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_eval = max(1, int(len(pairs) * pct)) if len(pairs) > 1 else 0
    eval_idx = set(idx[:n_eval])
    train = [pairs[i] for i in idx if i not in eval_idx]
    ev = [pairs[i] for i in idx if i in eval_idx]
    log(f"[split] seeded hold-out {cfg.get('eval_split_pct', 10)}%: "
        f"{len(train)} train / {len(ev)} eval (seed={cfg.get('split_seed', 42)})")
    return train, ev


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def run(cfg: dict) -> None:
    import numpy as np
    import torch
    from datasets import Audio, Dataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
    import evaluate as hf_evaluate
    import jiwer

    # Checkpoints (model + Adam optimizer state) are huge — ~10 GB each, and
    # save_total_limit rotation briefly holds two. Put the run's work dir under
    # the configured work_dir (default /share, a roomy volume); /tmp is a small
    # disk that overflows mid-save ("unexpected pos … inline_container.cc").
    # main() rm's this dir afterwards when cleanup_checkpoints is set (the best
    # model is uploaded to S3 first).
    global _RUN_WORKDIR
    _train_root = os.path.join((cfg.get("work_dir") or "/share").rstrip("/"), "checkpoint-whisper")
    try:
        os.makedirs(_train_root, exist_ok=True)
        work = tempfile.mkdtemp(prefix="autotrain-", dir=_train_root)
    except OSError:
        work = tempfile.mkdtemp(prefix="autotrain-")
    _RUN_WORKDIR = work
    log(f"[trainer] work dir: {work}")
    base_model = cfg["base_model"]
    auto_lang = _auto_lang(cfg)
    # Auto mode → processor/generation language is None (the per-utterance language
    # token comes from the preformatted labels); fixed mode → the configured code.
    language = None if auto_lang else (cfg.get("language") or None)
    task = cfg.get("task") or "transcribe"
    metric_name = (cfg.get("eval_metric") or "wer").lower()

    log(f"[train] base_model={base_model} metric={metric_name} "
        f"max_epochs={cfg['max_epochs']} patience={cfg.get('patience', 0)}")

    no_eval = bool(cfg.get("no_eval"))
    pairs = load_pairs(cfg["dataset"], work)
    if no_eval:
        # "No test set" — train on everything, no held-out eval (no WER/CER, no
        # best-checkpoint selection, no early stop). The final/last model is saved.
        train_pairs, eval_pairs = pairs, []
        log(f"[split] no_eval: training on all {len(train_pairs)} rows — evaluation disabled")
    elif cfg.get("test_dataset"):
        train_pairs = pairs
        eval_pairs = load_pairs(cfg["test_dataset"], work)
        log(f"[split] separate test dataset: {len(train_pairs)} train / {len(eval_pairs)} eval")
    else:
        train_pairs, eval_pairs = split_pairs(pairs, cfg)
    if not train_pairs or (not no_eval and not eval_pairs):
        raise RuntimeError(
            f"need both train and eval examples (got {len(train_pairs)}/{len(eval_pairs)})"
        )

    # Standardize/clean every transcription (cleaning.py) and, in auto mode, tag each
    # utterance's language for the Whisper prompt (zh by CJK character ratio, else
    # fastText ms/en). Done once here, not per __getitem__, so it's cheap across epochs.
    lang_model = _load_lang_model(cfg) if auto_lang else None
    log(f"[train] text: clean={bool(cfg.get('clean_text', True))} "
        f"language={'auto (per-utterance en/ms/zh)' if auto_lang else (language or 'none')}")
    _prepare_texts(train_pairs, cfg, lang_model, "train")
    _prepare_texts(eval_pairs, cfg, lang_model, "eval")

    load_dt, amp = parse_precision(cfg.get("precision"))
    _tdt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[load_dt]
    log(f"[train] precision: load={load_dt} amp={amp or 'none'}")
    processor = WhisperProcessor.from_pretrained(base_model, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(base_model, torch_dtype=_tdt)
    model.generation_config.language = language
    model.generation_config.task = task
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # SpecAugment (time/feature masking on the input mel features, training only —
    # HF applies it in the forward when model.training and apply_spec_augment).
    # The standard Whisper-finetune regularizer for small corpora; independent of
    # the waveform-level augment_techniques.
    if bool(cfg.get("spec_augment", False)):
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = float(cfg.get("mask_time_prob", 0.05))
        model.config.mask_time_length = int(cfg.get("mask_time_length", 10))
        model.config.mask_feature_prob = float(cfg.get("mask_feature_prob", 0.0))
        model.config.mask_feature_length = int(cfg.get("mask_feature_length", 10))
        log(f"[train] SpecAugment on: mask_time_prob={model.config.mask_time_prob} "
            f"mask_feature_prob={model.config.mask_feature_prob}")

    # Freeze the encoder (train decoder only) — faster + less overfitting on
    # small corpora. Independent of LoRA.
    if cfg.get("freeze_encoder"):
        model.freeze_encoder()
        log("[train] encoder frozen — training the decoder only")

    # LoRA / PEFT — train low-rank adapters on the attention projections instead
    # of the full model (far less VRAM + faster). The adapters are merged back
    # into the base weights at save time, so the artifact is a drop-in Whisper
    # checkpoint (no peft needed to load/serve it).
    use_lora = bool(cfg.get("use_lora"))
    if use_lora:
        from peft import LoraConfig, get_peft_model  # in the venv (installed by _ensure_venv)

        _r = int(cfg.get("lora_r", 16))
        # alpha is conventionally a ratio of r (e.g. 2×). When lora_alpha_ratio is
        # set, derive alpha = round(r × ratio) so sweeping r carries alpha with it
        # (no separate alpha dimension to permute); else use an absolute lora_alpha.
        _ratio = cfg.get("lora_alpha_ratio")
        _alpha = int(round(_r * float(_ratio))) if _ratio is not None else int(cfg.get("lora_alpha", 32))
        lconf = LoraConfig(
            r=_r,
            lora_alpha=_alpha,
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            # All linear layers (attn q/k/v/out_proj + MLP fc1/fc2 across encoder
            # & decoder); peft's "all-linear" auto-excludes the tied output proj.
            target_modules="all-linear",
            bias="none",
        )
        model = get_peft_model(model, lconf)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        log(f"[train] LoRA enabled (r={lconf.r}, alpha={lconf.lora_alpha}, "
            f"dropout={lconf.lora_dropout}) — {n_train:,}/{n_all:,} params trainable "
            f"({100 * n_train / max(1, n_all):.2f}%)")

    # Lazy datasets: no upfront decode / Arrow build. Feature extraction happens
    # in __getitem__, parallelized by the DataLoader's workers and overlapped
    # with GPU compute — so training starts immediately.
    aug_techs = [t for t in (cfg.get("augment_techniques") or []) if t in _AUG_FUNCS]
    aug_prob = float(cfg.get("augment_prob", 0.5))
    train_ds = _LazyAsrDataset(train_pairs, processor, work,
                               augment_techniques=aug_techs, augment_prob=aug_prob)
    eval_ds = None if no_eval else _LazyAsrDataset(eval_pairs, processor, work)  # never augment eval
    log(f"[trainer] {len(train_ds)} train / {0 if no_eval else len(eval_ds)} eval examples "
        f"— audio fetched + decoded lazily per item during training"
        + (f"; augment p={aug_prob}: {', '.join(aug_techs)}" if aug_techs else ""))

    class Collator:
        def __call__(self, features):
            inp = [{"input_features": f["input_features"]} for f in features]
            batch = processor.feature_extractor.pad(inp, return_tensors="pt")
            # Match input_features to the model's param dtype. Under nn.DataParallel
            # (multi-GPU, plain `python`) HF half-converts the model for fp16/bf16
            # but autocast doesn't reach the DP replica threads, so float32 input
            # hits half weights → "Input type (float) and bias type (Half)". Casting
            # here is a no-op on single-GPU/fp32 and fixes the DP case.
            try:
                md = next(model.parameters()).dtype
                if batch["input_features"].dtype != md:
                    batch["input_features"] = batch["input_features"].to(md)
            except StopIteration:
                pass
            labels = processor.tokenizer.pad(
                [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
            )
            lab = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
            if (lab[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
                lab = lab[:, 1:]
            batch["labels"] = lab
            return batch

    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    # Whisper-style eval normalization (lowercase + strip punctuation) BEFORE
    # scoring — otherwise WER/CER are inflated by casing/punctuation. Always the
    # multilingual basic normalizer, even for en: the English normalizer also
    # spells out numbers ("25" → "twenty five"), which we deliberately DON'T
    # want. Fall back to lowercasing if the helper isn't available. Opt out via
    # normalize_text=false to score raw (cased + punctuated) text.
    _tok = processor.tokenizer
    _do_norm = bool(cfg.get("normalize_text", True))
    log(f"[train] WER/CER on {'normalized' if _do_norm else 'raw'} text")

    def _normalize(s: str) -> str:
        s = s or ""
        if not _do_norm:
            return s.strip()
        try:
            # basic_normalize maps trailing punctuation to a trailing space —
            # strip it, or CER counts a phantom char when only one side ends
            # with punctuation.
            return _tok.basic_normalize(s).strip()
        except Exception:
            return s.lower().strip()

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids = np.where(label_ids != -100, label_ids, _tok.pad_token_id)
        pred_str = _tok.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = _tok.batch_decode(label_ids, skip_special_tokens=True)
        # jiwer errors on an empty reference — drop pairs whose normalized
        # reference is blank (e.g. punctuation-only labels).
        pairs = [(p, r) for p, r in (
            (_normalize(ps), _normalize(ls)) for ps, ls in zip(pred_str, label_str)
        ) if r.strip()]
        if not pairs:
            return {"wer": 0.0, "cer": 0.0}
        # Outlier filter: a row whose OWN WER or CER exceeds 1.0 (hallucinated
        # loop, truncated/empty decode, language flip) is excluded from the
        # corpus score entirely — one bad decode shouldn't swamp the epoch.
        kept = []
        for p, r in pairs:
            try:
                if jiwer.wer(r, p) > 1.0 or jiwer.cer(r, p) > 1.0:
                    continue
            except Exception:
                continue
            kept.append((p, r))
        if len(kept) < len(pairs):
            log(f"[eval] skipped {len(pairs) - len(kept)}/{len(pairs)} rows with WER/CER > 1.0")
        if not kept:
            return {"wer": 0.0, "cer": 0.0}
        preds = [p for p, _ in kept]
        refs = [r for _, r in kept]
        wer = 100 * wer_metric.compute(predictions=preds, references=refs)
        cer = 100 * cer_metric.compute(predictions=preds, references=refs)
        return {"wer": wer, "cer": cer}

    # ---- experiment tracking (W&B / MLflow via HF Trainer's report_to) ----
    tracking = cfg.get("tracking") or {}
    report_to = list(tracking.get("report_to") or [])
    for k, v in (tracking.get("env") or {}).items():
        if v not in (None, ""):
            os.environ[k] = str(v)
    if "wandb" in report_to and not os.environ.get("WANDB_API_KEY"):
        log("[track] W&B requested but no WANDB_API_KEY — disabling W&B")
        report_to = [r for r in report_to if r != "wandb"]
    if "mlflow" in report_to and not os.environ.get("MLFLOW_TRACKING_URI"):
        log("[track] MLflow requested but no MLFLOW_TRACKING_URI — disabling MLflow")
        report_to = [r for r in report_to if r != "mlflow"]
    if report_to:
        log(f"[track] reporting metrics to: {', '.join(report_to)}")

    out_dir = os.path.join(work, "out")
    # Cap logging_steps to the real step count so SHORT runs still emit a loss
    # curve. HF only logs "loss" every logging_steps; a 1-epoch / tiny-dataset
    # run can have fewer total steps than the default (10), producing an empty
    # loss chart. Aim for ~20 points across the whole run, min 1.
    _bs = max(1, int(cfg.get("batch_size", 8)))
    _ga = max(1, int(cfg.get("grad_accum", 1)))
    _world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    _spe = max(1, (len(train_ds) + (_bs * _ga * _world) - 1) // (_bs * _ga * _world))
    _epochs = max(1, int(cfg["max_epochs"]) or 1)
    _max_steps = int(cfg.get("max_steps", 0) or 0)
    _total_steps = max(1, _spe * _epochs)
    if _max_steps > 0:
        _total_steps = min(_total_steps, _max_steps)
    # Eval + checkpoint cadence (epoch | steps). load_best_model_at_end requires
    # save_strategy == eval_strategy (and save_steps a multiple of eval_steps), so
    # keep them in lockstep. Under a short max_steps cap, shrink eval_steps so at
    # least one eval fires (else there's no "best" checkpoint to load).
    _eval_strat = str(cfg.get("eval_strategy") or "epoch").lower()
    _eval_steps = max(1, int(cfg.get("eval_steps", 500) or 500))
    if _max_steps > 0 and _eval_strat == "steps":
        _eval_steps = min(_eval_steps, _max_steps)
    eff_logging_steps = max(1, min(int(cfg.get("logging_steps", 10)), _total_steps // 20 or 1))
    log(f"[trainer] ~{_total_steps} optimizer steps (world={_world}) → logging_steps={eff_logging_steps}"
        + (f", max_steps={_max_steps}" if _max_steps > 0 else "")
        + (f", eval every {_eval_steps} steps" if _eval_strat == "steps" else ", eval per epoch"))
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=int(cfg.get("batch_size", 8)),
        per_device_eval_batch_size=int(cfg.get("batch_size", 8)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 1)),
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type") or "linear"),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        num_train_epochs=float(cfg["max_epochs"]),
        max_steps=(_max_steps if _max_steps > 0 else -1),
        eval_strategy=("no" if no_eval else _eval_strat),
        save_strategy=_eval_strat,
        eval_steps=_eval_steps,
        save_steps=_eval_steps,
        predict_with_generate=True,
        generation_max_length=int(cfg.get("generation_max_length", 225)),
        fp16=(amp == "fp16"),
        bf16=(amp == "bf16"),
        # torch.compile (opt-in, cfg["torch_compile"]): HF Trainer compiles the (PEFT-wrapped)
        # Whisper model with the default inductor backend. Off by default; the first step is slow
        # (tracing). Whisper is a stock encoder-decoder so it compiles cleanly (no custom kernels).
        torch_compile=bool(cfg.get("torch_compile", False)),
        # No eval → no "best" to load (load_best needs eval==save strategy); keep
        # the last checkpoint instead.
        load_best_model_at_end=(not no_eval),
        metric_for_best_model=(None if no_eval else metric_name),
        greater_is_better=(None if no_eval else False),
        save_total_limit=1,
        logging_steps=eff_logging_steps,
        # Parallel lazy audio fetch/decode in __getitem__, overlapped with GPU.
        dataloader_num_workers=max(0, int(cfg.get("dataloader_num_workers", 4))),
        report_to=report_to,
        run_name=cfg.get("run_name") or None,
    )

    class MetricEmitter(TrainerCallback):
        """Stream per-epoch results so the gateway can chart WER/CER live."""
        def on_log(self, a, state, control, logs=None, **kw):
            # Training-step logs carry "loss" (every logging_steps); eval logs
            # carry "eval_loss". Emit a STEP point per training log so the
            # platform can draw a live loss curve.
            logs = logs or {}
            if "loss" in logs:
                emit("STEP", {
                    "step": int(state.global_step),
                    "loss": logs.get("loss"),
                    "lr": logs.get("learning_rate"),
                    "epoch": round(float(logs.get("epoch") or state.epoch or 0), 3),
                })

        def on_evaluate(self, a, state, control, metrics=None, **kw):
            m = metrics or {}
            train_loss = None
            for h in reversed(state.log_history):
                if "loss" in h:
                    train_loss = h["loss"]
                    break
            emit("METRIC", {
                "epoch": round(float(state.epoch or 0), 3),
                "wer": m.get("eval_wer"),
                "cer": m.get("eval_cer"),
                "eval_loss": m.get("eval_loss"),
                "train_loss": train_loss,
            })

    # Graceful early-stop: the gateway's /stop-early `touch`es $SGPU_STOP_FLAG; we
    # poll it each step and stop cleanly so the partial model is still saved+uploaded.
    class StopFlag(TrainerCallback):
        def __init__(self):
            self.path = os.environ.get("SGPU_STOP_FLAG")
        def on_step_end(self, args, state, control, **kw):
            if self.path and os.path.exists(self.path):
                control.should_training_stop = True
                if state.is_world_process_zero:
                    print("[trainer] early-stop flag seen → stopping after this step; saving model", flush=True)
            return control

    callbacks: list = [MetricEmitter(), StopFlag()]
    patience = int(cfg.get("patience", 0) or 0)
    if patience > 0 and not no_eval:  # early stopping needs eval metrics
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=Collator(),
        compute_metrics=(None if no_eval else compute_metrics),
        processing_class=processor,
        callbacks=callbacks,
    )

    result = trainer.train()
    epochs_ran = int(round(result.metrics.get("epoch", 0)))
    stopped_early = epochs_ran < int(cfg["max_epochs"])

    # evaluate() is a collective op under DDP — EVERY rank must call it or the
    # ranks deadlock. Save + upload + push happen on rank 0 only (below). With
    # no_eval there's no eval set, so every rank skips it (still collective-safe).
    if no_eval:
        best = {"epoch": epochs_ran, "wer": None, "cer": None, "eval_loss": None}
    else:
        # Report the best *observed* epoch by the selection metric (lower is better),
        # read from the eval history — NOT a fresh evaluate() of whatever weights are
        # loaded at the end. Those can disagree: if load_best_model_at_end selects the
        # wrong checkpoint (e.g. HF's pre-fix default treated a higher WER as "better"),
        # a final re-eval reports a metric WORSE than an epoch the run already beat —
        # exactly how a finished run's "Best WER" ended up above an epoch it had passed.
        # log_history is identical across DDP ranks, so this needs no collective op.
        mkey = f"eval_{metric_name}"
        evals = [h for h in trainer.state.log_history if isinstance(h.get(mkey), (int, float))]
        if evals:
            b = min(evals, key=lambda h: h[mkey])
            best = {
                "epoch": int(round(float(b.get("epoch") or epochs_ran))),
                "wer": b.get("eval_wer"),
                "cer": b.get("eval_cer"),
                "eval_loss": b.get("eval_loss"),
            }
        else:
            # No eval metric logged (shouldn't happen with eval on) — fall back to a
            # final evaluate() so we still report something (all ranks call it → safe).
            final = trainer.evaluate()
            best = {
                "epoch": epochs_ran,
                "wer": final.get("eval_wer"),
                "cer": final.get("eval_cer"),
                "eval_loss": final.get("eval_loss"),
            }
    if not _IS_MAIN:
        return  # non-main DDP ranks: nothing more to do; rank 0 saves/uploads.

    best_dir = os.path.join(work, "best-model")
    # Unwrap any DDP/DataParallel wrapper; fold LoRA adapters into the base so the
    # saved checkpoint is a plain Whisper model (loads + serves without peft).
    save_model = trainer.model
    try:
        save_model = trainer.accelerator.unwrap_model(save_model)
    except Exception:  # noqa: BLE001
        pass
    if use_lora and hasattr(save_model, "merge_and_unload"):
        save_model = save_model.merge_and_unload()
        log("[train] merged LoRA adapters into the base weights")
    save_model.save_pretrained(best_dir)
    processor.save_pretrained(best_dir)

    # ---- upload artifacts to S3 (best model + metrics) ----
    art = cfg.get("artifacts") or {}
    s3_uri = None
    if art.get("bucket"):
        cli = _s3_client(art)
        base_key = art["prefix"].rstrip("/") + "/best-model"
        for root, _dirs, files in os.walk(best_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, best_dir)
                cli.upload_file(fp, art["bucket"], f"{base_key}/{rel}")
        metrics_key = art["prefix"].rstrip("/") + "/metrics.json"
        cli.put_object(
            Bucket=art["bucket"], Key=metrics_key,
            Body=json.dumps({"best": best, "epochs": epochs_ran}).encode(),
        )
        s3_uri = f"s3://{art['bucket']}/{base_key}/"
        log(f"[upload] best model → {s3_uri}")

    # ---- optional HF push (the merged model, so it's a drop-in checkpoint) ----
    hf_repo = None
    if cfg.get("hf_push_repo") and cfg.get("hf_token"):
        try:
            save_model.push_to_hub(cfg["hf_push_repo"], token=cfg["hf_token"])
            processor.push_to_hub(cfg["hf_push_repo"], token=cfg["hf_token"])
            hf_repo = cfg["hf_push_repo"]
            log(f"[upload] pushed best model → https://huggingface.co/{hf_repo}")
        except Exception as e:  # noqa: BLE001
            log(f"[upload] HF push failed: {e}")

    emit("ARTIFACT", {"s3_uri": s3_uri, "hf_repo": hf_repo})
    emit("DONE", {"best": best, "epochs": epochs_ran, "stopped_early": stopped_early})


def _redirect_tmp(base: str) -> None:
    """Move TMPDIR off the small local /tmp onto a roomy dir (default /share).
    DataLoader workers' multiprocessing temp (pymp-*), pip, and Python tempfile
    all honour this — /tmp is often a small disk that overflows on big-model
    runs (No space left on device)."""
    base = (base or "/share").rstrip("/")
    d = os.path.join(base, "tmp")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return
    for k in ("TMPDIR", "TEMP", "TMP"):
        os.environ[k] = d
    tempfile.tempdir = d
    log(f"[trainer] TMPDIR → {d} (off small /tmp)")


def _cleanup_workdir(enabled: bool) -> None:
    """rm the run's checkpoints + work dir (the best model is already on S3).
    Rank-0 only: under DDP all ranks share the dir, and a non-main rank must not
    delete it while rank 0 is still uploading."""
    if not enabled or not _RUN_WORKDIR or not _IS_MAIN:
        return
    import shutil
    shutil.rmtree(_RUN_WORKDIR, ignore_errors=True)
    log(f"[trainer] cleaned checkpoints + work dir: {_RUN_WORKDIR}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to JSON config")
    ap.add_argument("--deps-only", action="store_true",
                    help="install dependencies then exit (used by the sweep orchestrator)")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    _redirect_tmp(cfg.get("work_dir") or "/share")
    cleanup = bool(cfg.get("cleanup_checkpoints", True))
    try:
        # --deps-only (deps phase, system python): build the isolated uv venv.
        # Run phase: the gateway launches us with {venv}/bin/python, so the venv
        # is already present (this is a fast no-op) and run()'s imports resolve.
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
        _cleanup_workdir(cleanup)


if __name__ == "__main__":
    sys.exit(main())
