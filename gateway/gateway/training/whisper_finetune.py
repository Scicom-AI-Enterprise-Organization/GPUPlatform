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
    ]
    check_mods = ["torch", "transformers", "datasets", "evaluate", "jiwer", "soundfile", "boto3", "peft", "fasttext"]
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


_AUG_FUNCS = {
    "telephone": _aug_telephone,
    "noise": _aug_noise,
    "dropout": _aug_dropout,
    "gain": _aug_gain,
    "pitch": _aug_pitch,
    "speed": _aug_speed,
    "reverb": _aug_reverb,
    "bandpass": _aug_bandpass,
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

    # Whisper's standard eval normalizes text (lowercase, strip punctuation,
    # spell-out numbers, …) BEFORE scoring — otherwise WER/CER are inflated by
    # casing/punctuation and aren't comparable to any published number. Use the
    # tokenizer's English normalizer for en, the multilingual basic normalizer
    # otherwise; fall back to lowercasing if the helper isn't available. Opt out
    # via normalize_text=false to score raw (cased + punctuated) text.
    _tok = processor.tokenizer
    _is_en = (language or "").lower() in ("en", "english")
    _do_norm = bool(cfg.get("normalize_text", True))
    log(f"[train] WER/CER on {'normalized' if _do_norm else 'raw'} text")

    def _normalize(s: str) -> str:
        s = s or ""
        if not _do_norm:
            return s.strip()
        try:
            return _tok.normalize(s) if _is_en else _tok.basic_normalize(s)
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
        preds = [p for p, _ in pairs]
        refs = [r for _, r in pairs]
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
