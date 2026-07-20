#!/usr/bin/env python3
"""Standalone Qwen3 + NeuCodec TTS finetune orchestrator — shipped to a RunPod
pod / VM by the gateway's Autotrain runner (task_type=tts) and run over SSH.

It vendors three scripts from the `finetune-multilingual-tts` reference (sibling
`tts/` dir, SFTP'd alongside this file) and runs the full pipeline:
  1. resolve the registered {audio, transcription[, speaker]} dataset → write
     audio files + `audio_paths.json` + `meta.jsonl`,
  2. `convert_neucodec.py` — encode audio → NeuCodec speech tokens,
  3. `pack_stage1.py` — pack tokens+text into an MDS streaming dataset,
  4. `torchrun qwen3_tts_flash.py` — finetune Qwen3 as a causal LM over the pack
     (loss-only; metrics go to W&B/MLflow via HF Trainer report_to),
then upload the checkpoint to S3 (+ optional HF push).

No gateway imports — config arrives as a single JSON file (--config). The
sub-scripts emit `[AUTOTRAIN_PROGRESS] step=… percent=…`; the gateway parses
those from the stream. This file emits @@ARTIFACT / @@DONE / @@ERROR.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TTS_DIR = os.path.join(THIS_DIR, "tts")
_LOSS_RE = re.compile(r"'loss':\s*([0-9.eE+-]+)")
# Per-run work dir (set in run()); main() rm's it when cleanup_checkpoints is on.
_RUN_WORKDIR = None


def _free_port() -> int:
    """An OS-assigned free TCP port for torch.distributed's rendezvous. The
    default 29500 collides when concurrent sweep trials each launch their own
    torchrun on the same box (EADDRINUSE) — a unique port per launch avoids it."""
    import socket
    s = socket.socket()
    try:
        s.bind(("", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def log(msg: str) -> None:
    print(msg, flush=True)


def emit(tag: str, obj: dict) -> None:
    print(f"@@{tag} {json.dumps(obj)}", flush=True)


def parse_precision(p):
    """'<load>-<amp>' → (torch_dtype_name, amp): weight load dtype + the
    mixed-precision (AMP) train dtype. Back-compat: bare 'bf16'/'fp16' = load
    fp32 + that AMP; 'fp32' = full fp32 (no AMP)."""
    p = (p or "fp32-bf16").lower()
    if "-" in p:
        load, amp = p.split("-", 1)
    elif p == "fp32":
        load, amp = "fp32", ""
    else:
        load, amp = "fp32", p
    load_dt = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}.get(load, "float32")
    return load_dt, (amp if amp in ("bf16", "fp16") else "")


def _run_loss(cmd: list[str], cwd: str, env: dict) -> float | None:
    """Run a command, tee its stdout, and return the last HF-logged train loss
    (so a sweep can rank TTS trials, which are loss-only)."""
    log(f"[gateway] $ {' '.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = None
    for line in p.stdout:  # type: ignore[union-attr]
        print(line, end="", flush=True)
        m = _LOSS_RE.search(line)
        if m:
            try:
                last = float(m.group(1))
            except ValueError:
                pass
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)
    return last


# The TTS stack the trainer pins. torchaudio MUST match torch (neucodec imports
# it); a mismatched torchaudio gives "undefined symbol: torch_library_impl" on
# `import neucodec`. ChiniDataset (Parquet streaming) is vendored alongside the
# tts scripts (private repo, no git on the box) — not a pip dep.
DEFAULT_VENV = "/share/autotrain-tts"
DEPS = [
    "torch==2.9.1", "torchaudio==2.9.1", "transformers==4.57.3", "accelerate",
    # Pin numba/numpy: librosa pulls numba transitively, and on a FRESH py3.12 venv
    # uv otherwise backtracks numba to 0.53.1 (no py3.12 wheel → sdist build that
    # hard-fails "only versions >=3.6,<3.10 are supported") because torch's numpy 2.3
    # has no compatible numba. Pinning numba>=0.61 (py3.12 wheels) + numpy<2.3 (the
    # range numba 0.61 supports, still fine for torch/transformers) makes it resolve.
    "numba>=0.61", "numpy<2.3",
    "librosa", "soundfile", "datasets", "wandb", "boto3",
    # neucodec is added per-codec-variant in _ensure_venv (upstream vs the Scicom
    # 44k-d20 fork) — see _neucodec_pkg / TTS_CODEC.
    "pandas", "pyarrow", "multiprocess", "liger-kernel",
    "git+https://github.com/apple/ml-cross-entropy",
    "peft>=0.11",  # LoRA (all-linear adapters, merged into base at save)
    "kernels",  # HF kernel-hub loader for the flash-attn-3 attn impl (fresh venvs)
]


# NeuCodec variant (decoder only — the encoder/FSQ codes are identical, so a model
# trained with either decodes with either; this just sets synthesis output fidelity):
#   "neucodec"     → upstream neuphonic/neucodec, 24 kHz  (default)
#   "neucodec-44k" → Scicom-intl/neucodec-44k-d20 fork, 44.1 kHz
_NEUCODEC_FORK_PKG = "git+https://github.com/Scicom-AI-Enterprise-Organization/neucodec-44k.git"


def wants_neucodec_fork(cfg: dict) -> bool:
    return (str(cfg.get("tts_codec") or "neucodec").strip().lower()
            in ("neucodec-44k", "scicom-44k", "44k", "fork"))


def _ensure_venv(cfg: dict) -> str:
    """Create/reuse an isolated uv venv with the TTS stack and return its python.
    Isolation keeps torch 2.9.1 + neucodec off the box's system python (and away
    from the Whisper stack — they need different torch). Idempotent."""
    import shutil

    venv = (cfg.get("venv_path") or DEFAULT_VENV).rstrip("/")
    py = os.path.join(venv, "bin", "python")
    # Bypass the pod image's hashed pip constraint (PIP_CONSTRAINT): it pins the
    # base wheels and rejects torch 2.9.1's nvidia-nccl dep ("do not match the
    # hashes from the requirements file"). A fresh venv has no such constraint.
    env = {**os.environ, "PIP_CONSTRAINT": "", "PIP_REQUIRE_HASHES": "0"}
    want_fork = wants_neucodec_fork(cfg)
    pkgs = list(DEPS) + [_NEUCODEC_FORK_PKG if want_fork else "neucodec"]
    if "mlflow" in ((cfg.get("tracking") or {}).get("report_to") or []):
        pkgs.append("mlflow")

    def _present() -> bool:
        # The installed neucodec must MATCH the requested variant — the fork carries a
        # PEP 610 direct_url.json with the org git URL; upstream PyPI has none. If the
        # venv has the wrong one, reinstall. (Scan all dists — the fork's dist name may
        # differ from the `neucodec` import name.)
        op = "" if want_fork else "not "
        probe = (
            "import torch, torchaudio, transformers, neucodec, pyarrow, boto3; "
            "import importlib.metadata as _m; "
            f"assert {op}any('Scicom-AI-Enterprise-Organization' in (d.read_text('direct_url.json') or '') "
            "for d in _m.distributions()), 'neucodec variant mismatch'"
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
        log(f"[deps] TTS venv ready: {py}")
        return py
    have_uv = shutil.which("uv") is not None
    # Create the venv only if absent (`uv venv` errors on a non-empty dir); then
    # ALWAYS install (idempotent — adds any missing pkg).
    if not os.path.exists(py):
        log(f"[deps] creating venv {venv} …")
        if have_uv:
            subprocess.check_call(["uv", "venv", venv, "--python", "3.12"], env=env)
        else:
            subprocess.check_call([sys.executable, "-m", "venv", venv], env=env)
            subprocess.check_call([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], env=env)
    log(f"[deps] installing TTS stack into {venv} (torch/torchaudio/neucodec/…) …")
    if have_uv:
        subprocess.check_call(["uv", "pip", "install", "--python", py, *pkgs], env=env)
    else:
        subprocess.check_call([py, "-m", "pip", "install", "-q", *pkgs], env=env)
    log(f"[deps] TTS venv ready: {py}")
    return py


# Eval-only extras, installed on demand (per the selected methods) so non-eval
# runs keep a lean venv. The MOS/similarity models come from the Scicom repos.
_EVAL_DEPS = {
    "cer": ["jiwer"],  # transformers/whisper already in the TTS stack
    "mos": ["git+https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2"],
    "similarity": ["scikit-learn", "git+https://github.com/Scicom-AI-Enterprise-Organization/titanet-vectors-fp16"],
}


def _ensure_eval_deps(py: str, methods: list, env: dict) -> None:
    import shutil

    pkgs: list[str] = []
    for m in methods:
        pkgs += _EVAL_DEPS.get(m, [])
    pkgs = sorted(set(pkgs))
    if not pkgs:
        return
    log(f"[eval][deps] installing: {', '.join(pkgs)}")
    if shutil.which("uv"):
        subprocess.check_call(["uv", "pip", "install", "--python", py, *pkgs], env=env)
    else:
        subprocess.check_call([py, "-m", "pip", "install", "-q", *pkgs], env=env)


# --------------------------------------------------------------------------
# dataset → audio files + meta.jsonl + audio_paths.json
# --------------------------------------------------------------------------
def _s3_client(ds: dict):
    import boto3
    from botocore.client import Config as BotoConfig
    return boto3.client(
        "s3", region_name=ds.get("region") or "us-east-1",
        endpoint_url=ds.get("endpoint") or None,
        aws_access_key_id=ds.get("access_key") or None,
        aws_secret_access_key=ds.get("secret_key") or None,
        config=BotoConfig(signature_version="s3v4"),
    )


def _upload_s3_dir(art: dict, local_dir: str, key_prefix: str) -> str:
    """Upload every file under local_dir to s3://{bucket}/{key_prefix}/… and
    return the s3:// URI of the prefix."""
    cli = _s3_client(art)
    base_key = key_prefix.rstrip("/")
    for root, _dirs, files in os.walk(local_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, local_dir)
            cli.upload_file(fp, art["bucket"], f"{base_key}/{rel}")
    return f"s3://{art['bucket']}/{base_key}/"


def _download_s3_prefix(spec: dict, s3_uri: str, dest_dir: str) -> None:
    """Download every object under an s3://bucket/prefix into dest_dir (used to
    pull a pre-packed ChiniDataset before training)."""
    assert s3_uri.startswith("s3://"), s3_uri
    bucket, _, prefix = s3_uri[len("s3://"):].partition("/")
    prefix = prefix.rstrip("/") + "/"
    cli = _s3_client(spec)
    os.makedirs(dest_dir, exist_ok=True)
    for page in cli.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel:
                continue
            fp = os.path.join(dest_dir, rel)
            os.makedirs(os.path.dirname(fp) or dest_dir, exist_ok=True)
            cli.download_file(bucket, obj["Key"], fp)


def _read_metadata_rows(ds: dict) -> list[dict]:
    import csv, io
    cli = _s3_client(ds)
    body = cli.get_object(Bucket=ds["bucket"], Key=ds["metadata_key"])["Body"].read()
    text = body.decode("utf-8", errors="replace")
    fmt = (ds.get("format") or "").lower() or (
        "jsonl" if ds["metadata_key"].endswith(".jsonl")
        else "json" if ds["metadata_key"].endswith(".json") else "csv"
    )
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(text)))
    if fmt == "jsonl":
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("data", data.get("rows", []))


def build_dataset(cfg: dict, work: str) -> tuple[str, str]:
    """Resolve audio sources + write audio_sources.json + meta.jsonl.

    S3 rows are NOT bulk-downloaded: their (presigned) URL is passed through so
    convert_neucodec streams + prefetches them during encoding (overlapping the
    download with GPU work). HF/label rows are materialised locally (their audio
    is an in-memory array / token-gated endpoint, not a plain fetchable URL).
    audio_sources.json is a list of {key, src}: `key` is the token-file key (==
    meta's filename_audio) and `src` is a local path or an S3 URL.
    Returns (audio_sources_json, meta_jsonl)."""
    import soundfile as sf

    ds = cfg["dataset"]
    audio_field = ds.get("audio_field") or "audio"
    text_field = ds.get("transcription_field") or "transcription"
    speaker_field = cfg.get("speaker_field") or "speaker"
    default_speaker = cfg.get("default_speaker") or "speaker"

    audio_dir = os.path.join(work, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    rows: list[dict] = []
    sources: list[dict] = []  # [{key, src}] for convert_neucodec (src = path or URL)

    if ds.get("kind") == "hf":
        from datasets import load_dataset
        dd = load_dataset(ds["hf_repo"], token=ds.get("hf_token") or None)
        idx = 0
        for split_name, split in dd.items():
            tf = (ds.get("split_fields") or {}).get(split_name, text_field)
            for r in split:
                a = r[audio_field]
                rel = f"audio/{idx}.wav"
                sf.write(os.path.join(work, rel), a["array"], a["sampling_rate"])
                rows.append({
                    "filename_audio": rel, "text": str(r.get(tf, "")),
                    "speaker": str(r.get(speaker_field) or default_speaker),
                    "split": split_name,  # preserve the source split → packed per split
                })
                sources.append({"key": rel, "src": rel})  # local (HF array → wav)
                idx += 1
    elif ds.get("kind") == "label":
        # Labeling-platform rows resolved by the gateway. Platform-hosted clips
        # (empty/under-base audio_url) are fetched from the task audio endpoint
        # with the bearer token; presigned audio_urls download directly.
        import urllib.request
        base = (ds.get("label_base_url") or "").rstrip("/")
        pid = ds.get("label_project_id")
        token = ds.get("label_token")
        _exts = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm")
        for i, r in enumerate(ds.get("rows") or []):
            u = (r.get("audio_url") or "").strip()
            tid = r.get("id")
            txt = r.get("text")
            if txt is None:
                continue
            if (not u or u.startswith(base)) and tid is not None:
                req = urllib.request.Request(f"{base}/api/projects/{pid}/tasks/{tid}/audio",
                                             headers={"Authorization": f"Bearer {token}"})
            elif u:
                req = urllib.request.Request(u)
            else:
                continue
            ext = os.path.splitext(os.path.basename(u.split("?")[0]))[1].lower()
            if ext not in _exts:
                ext = ".wav"
            rel = f"audio/task-{tid if tid is not None else i}{ext}"
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    with open(os.path.join(work, rel), "wb") as f:
                        f.write(resp.read())
            except Exception as e:  # noqa: BLE001
                log(f"[data] skip label task {tid}: {e}")
                continue
            rows.append({"filename_audio": rel, "text": str(txt), "speaker": str(default_speaker),
                         "split": str(r.get("split") or "train")})
            sources.append({"key": rel, "src": rel})  # local (downloaded above)
    else:
        cli = _s3_client(ds)
        prefix = (ds.get("audio_prefix") or "").strip("/")
        # Rows manually un-ticked in the row browser → excluded from training.
        excluded = {int(x) for x in (ds.get("excluded_rows") or [])}
        streamed = downloaded = 0
        for i, r in enumerate(_read_metadata_rows(ds)):
            if i in excluded:
                continue
            ref = r.get(audio_field)
            txt = r.get(text_field)
            if not ref or txt is None:
                continue
            ref = str(ref)
            base = os.path.basename(ref.split("?")[0]) or "a.wav"
            rel = f"audio/{base}"
            if ref.startswith(("http://", "https://")):
                # Stream from S3 during encoding — no bulk download to disk.
                src = ref
                streamed += 1
            else:
                # Bare key (no presigned URL to stream) → fetch once via boto3.
                dest = os.path.join(work, rel)
                key = "/".join(p for p in [prefix, ref.lstrip("/")] if p)
                try:
                    cli.download_file(ds["bucket"], key, dest)
                except Exception as e:  # noqa: BLE001
                    log(f"[data] skip {ref!r}: {e}")
                    continue
                src = rel
                downloaded += 1
            rows.append({
                "filename_audio": rel, "text": str(txt),
                "speaker": str(r.get(speaker_field) or default_speaker),
                "split": str(r.get("split") or "train"),  # preserve source split
            })
            sources.append({"key": rel, "src": src})
        log(f"[data] s3 sources: {streamed} streamed from URLs, {downloaded} pre-downloaded")

    if not rows:
        raise RuntimeError("no usable {audio, transcription} rows resolved from the dataset")

    sources_path = os.path.join(work, "audio_sources.json")
    with open(sources_path, "w") as f:
        json.dump(sources, f)
    meta = os.path.join(work, "meta.jsonl")
    with open(meta, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"[data] {len(rows)} rows resolved ({len(sources)} audio sources)")
    return sources_path, meta


# Split subdir names treated as a held-out eval set (never trained on).
_TEST_SPLIT_NAMES = ("test", "validation", "valid", "dev")


def _split_subdirs(root: str) -> list[str]:
    """Immediate subdirs of `root` that are themselves ChiniDataset shards (carry
    their own index.json), sorted. A split-aware pack writes packed/<split>/ where
    <split> is whatever the source rows' `split` field held — it defaults to 'train'
    but can be 'default', 'test', a language code, … (the 'Pack for TTS' transform
    of a single-split HF dataset commonly yields 'default'). The trainer must
    therefore DISCOVER split names rather than assume a literal 'train'."""
    try:
        return sorted(
            n for n in os.listdir(root)
            if os.path.isdir(os.path.join(root, n))
            and os.path.exists(os.path.join(root, n, "index.json"))
        )
    except OSError:
        return []


def _splits_in(meta_jsonl: str) -> list[str]:
    """Distinct splits in meta.jsonl, first-seen order (e.g. train then test).
    Falls back to ['train'] when rows carry no split."""
    seen: list[str] = []
    with open(meta_jsonl) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            sp = json.loads(ln).get("split") or "train"
            if sp not in seen:
                seen.append(sp)
    return seen or ["train"]


# --------------------------------------------------------------------------
def run(cfg: dict) -> None:
    import tempfile
    global _RUN_WORKDIR
    # Per-run work dir under <work_dir>/sgpu-train (mirrors Whisper) so cleanup at
    # the end rm's ONLY this run's dir — never the shared /share root, which holds
    # the venv. Also means each run starts on a fresh dir (no stale checkpoints).
    _root = os.path.join((cfg.get("work_dir") or "/share").rstrip("/"), "checkpoint-tts")
    try:
        os.makedirs(_root, exist_ok=True)
        work = tempfile.mkdtemp(prefix="autotrain-tts-", dir=_root)
    except OSError:
        work = tempfile.mkdtemp(prefix="autotrain-tts-")
    _RUN_WORKDIR = work
    log(f"[trainer] work dir: {work}")

    # tracking env (gateway injected the resolved secrets in cfg["tracking"]["env"])
    tracking = cfg.get("tracking") or {}
    for k, v in (tracking.get("env") or {}).items():
        if v not in (None, ""):
            os.environ[k] = str(v)
    report_to = list(tracking.get("report_to") or [])

    # Deps live in the isolated uv venv (built by --deps-only); the gateway runs
    # us with that venv's python, so this is a fast no-op verifying it's present.
    _ensure_venv(cfg)

    model = cfg.get("base_model") or "Qwen/Qwen3-1.7B-Base"
    tokenizer = cfg.get("tokenizer") or "Scicom-intl/Multilingual-Expressive-TTS-1.7B"
    block_size = int(cfg.get("block_size", 10240))
    seq_len = int(cfg.get("pack_sequence_length", 4096))
    epochs = int(cfg.get("max_epochs", 3))
    batch = int(cfg.get("batch_size", 8))
    grad_accum = int(cfg.get("grad_accum", 4))
    lr = float(cfg.get("learning_rate", 2e-5))
    load_dt, amp = parse_precision(cfg.get("precision"))
    # nproc_per_node = the GPUs actually visible to THIS process. A sweep trial is
    # pinned by the sweep runner to gpus_per_trial GPUs via CUDA_VISIBLE_DEVICES;
    # honoring that (over the run-wide gpu_count) avoids launching N ranks onto a
    # 1-GPU pin. For a single run the gateway sets CVD to the pin = gpu_count, so
    # this is unchanged there.
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    gpus = (len([x for x in _cvd.split(",") if x.strip()]) if _cvd
            else max(1, int(cfg.get("gpu_count", 1))))
    gpus = max(1, gpus)

    packed = os.path.join(work, "packed")
    out_dir = os.path.join(work, "out")
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if report_to and cfg.get("run_name"):
        env.setdefault("WANDB_NAME", cfg["run_name"])

    def step(cmd: list[str], cwd: str) -> None:
        log(f"[gateway] $ {' '.join(cmd)}")
        subprocess.check_call(cmd, cwd=cwd, env=env)

    ds = cfg.get("dataset") or {}
    if ds.get("packed_uri"):
        # Dataset was already NeuCodec-encoded + multipacked (a "Pack for TTS"
        # transform) → pull the ChiniDataset shards and skip convert/pack.
        log(f"[pack] reusing pre-packed ChiniDataset from {ds['packed_uri']}")
        _download_s3_prefix(ds, ds["packed_uri"], packed)
    else:
        sources, meta = build_dataset(cfg, work)
        # 1. NeuCodec tokenization — streams + prefetches audio from S3, overlapping
        # download (comm) with GPU encode (comp); no bulk download to disk.
        step([sys.executable, "-u", os.path.join(TTS_DIR, "convert_neucodec.py"),
              "--file", sources], cwd=work)
        # 2. pack into ONE split-aware ChiniDataset: each SOURCE SPLIT becomes a
        # `packed/<split>/` subdir (its own index.json + shards) so train/test
        # never mix, but the whole `packed/` prefix is ONE dataset (the trainer
        # reads StreamingDataset(local, split=...); the UI shows a split picker).
        if cfg.get("pack_only"):
            art = cfg.get("artifacts") or {}
            splits = _splits_in(meta)
            sys.path.insert(0, TTS_DIR)
            from chinidataset import StreamingDataset
            per_split = {}
            for sp in splits:
                outdir = os.path.join(packed, sp)
                step([sys.executable, "-u", os.path.join(TTS_DIR, "pack_stage1.py"),
                      "--dataset", meta, "--split", sp, "--output_dir", outdir,
                      "--tokenizer", tokenizer, "--sequence_length", str(seq_len)], cwd=work)
                try:
                    per_split[sp] = len(StreamingDataset(local=outdir))
                except Exception as e:  # noqa: BLE001
                    log(f"[pack] could not count packed records for split {sp}: {e}")
                    per_split[sp] = None
                log(f"[pack] split={sp}: {per_split[sp]} records")
            s3_uri = None
            if art.get("bucket"):
                # Upload the whole packed/ tree once → one prefix with train/+test/.
                s3_uri = _upload_s3_dir(art, packed, art["prefix"].rstrip("/") + "/packed")
            total = sum(v for v in per_split.values() if isinstance(v, int))
            log(f"[pack] uploaded split-aware dataset → {s3_uri} · {per_split}")
            emit("PACKED", {"s3_uri": s3_uri, "samples": total, "splits": per_split,
                            "sequence_length": seq_len, "tokenizer": tokenizer})
            return
        step([sys.executable, "-u", os.path.join(TTS_DIR, "pack_stage1.py"),
              "--dataset", meta, "--output_dir", packed,
              "--tokenizer", tokenizer, "--sequence_length", str(seq_len)], cwd=work)

    # A pre-packed dataset may be split-aware: packed/train + packed/test subdirs
    # (each its own index.json), vs a flat dataset with index.json at the packed/
    # root. Train on the train split; "use its own test split" then evals on the
    # held-out packed/test (not a sample of the training shards).
    def _has_index(d):
        return os.path.exists(os.path.join(d, "index.json"))
    subdirs = _split_subdirs(packed)
    # Held-out eval split: first recognised test-ish subdir, if any.
    split_test_dir = next(
        (os.path.join(packed, s) for s in _TEST_SPLIT_NAMES if s in subdirs), None)
    # Train split, in priority order:
    #  1. an explicit train/ subdir — a split-aware pack may ALSO carry a flat
    #     combined index.json at the packed/ root (train+test mixed); training on
    #     that root would leak test records, so train/ takes priority over it;
    #  2. a flat packed/index.json (legacy single-dir pack);
    #  3. the sole non-test split subdir, WHATEVER it's named ('default', a lang
    #     code, …) — this is the reuse-a-pre-packed-dataset case that otherwise
    #     fell through to `packed` and died with "index.json not found".
    # Falling back to `packed` when nothing matched preserves the clear
    # FileNotFoundError downstream rather than masking a genuinely empty download.
    if "train" in subdirs:
        train_dir = os.path.join(packed, "train")
    elif _has_index(packed):
        train_dir = packed
    else:
        non_test = [s for s in subdirs if s not in _TEST_SPLIT_NAMES]
        if non_test:
            train_dir = os.path.join(packed, non_test[0])
            if len(non_test) > 1:
                log(f"[train] multiple non-test split subdirs {non_test}; "
                    f"training on {non_test[0]!r}")
        else:
            train_dir = packed
    log(f"[train] train_dir={train_dir}" + (f"  held-out split={split_test_dir}" if split_test_dir else ""))

    # Fresh output dir. /share/out is reused across runs on a VM, and a leftover
    # checkpoint-N from a prior/aborted run makes HF Trainer try to RESUME from it
    # and die ("Can't find a checkpoint index"). Start each run clean.
    import shutil as _shutil
    if os.path.isdir(out_dir):
        _shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    # 3. finetune via torchrun
    log(f"[train] precision: load={load_dt} amp={amp or 'none'}")
    dtype_args = ["--torch_dtype", load_dt]
    if amp == "bf16":
        dtype_args += ["--bf16"]
    elif amp == "fp16":
        dtype_args += ["--fp16"]
    report_flag = ",".join(report_to) if report_to else "none"
    # LoRA on all linear layers (embeddings + lm_head stay frozen). alpha follows
    # r via lora_alpha_ratio when set (mirrors the Whisper path) so a sweep over r
    # carries alpha with it; the adapters are merged into a plain checkpoint at
    # save (see qwen3_tts_flash.py) so eval/serving need no peft.
    lora_args: list[str] = []
    if cfg.get("use_lora"):
        _r = int(cfg.get("lora_r", 16))
        _ratio = cfg.get("lora_alpha_ratio")
        _alpha = int(round(_r * float(_ratio))) if _ratio is not None else int(cfg.get("lora_alpha", 32))
        # Always "all-linear" — the TTS form never exposes a target-module picker (that's
        # LLM-only UI), so cfg["lora_target_modules"] is really the LLM field's default
        # list ["q_proj", …] passed through unconditionally by the form. str()-ing that
        # list here would send qwen3_tts_flash.py the Python repr "['q_proj', …]", which
        # its `tgt.split(",")` turns into bracket/quote-mangled non-module names — peft
        # then raises "Target modules … not found in the base model" on every TTS LoRA run.
        _tgt = "all-linear"
        lora_args = [
            "--use_lora", "true",
            "--lora_r", str(_r),
            "--lora_alpha", str(_alpha),
            "--lora_dropout", str(float(cfg.get("lora_dropout", 0.05))),
            "--lora_target_modules", _tgt,
        ]
        log(f"[train] LoRA enabled (r={_r}, alpha={_alpha}, dropout={cfg.get('lora_dropout', 0.05)}, target={_tgt})")
    # Eval + checkpoint cadence (epoch | steps) + optional hard step cap — all
    # native HF TrainingArguments. eval_steps == save_steps so a Whisper-style
    # load_best_model_at_end stays valid; qwen3 only turns eval ON when the packed
    # dataset has a test split (else it forces eval_strategy=no).
    _evs = str(cfg.get("eval_strategy") or "epoch").lower()
    _svs = str(cfg.get("save_strategy") or _evs).lower()
    cadence_args = [
        "--eval_strategy", _evs,
        "--save_strategy", _svs,
        "--logging_steps", str(max(1, int(cfg.get("logging_steps", 10) or 10))),
    ]
    if _evs == "steps":
        cadence_args += ["--eval_steps", str(int(cfg.get("eval_steps", 500) or 500))]
    if _svs == "steps":
        cadence_args += ["--save_steps", str(int(cfg.get("save_steps", cfg.get("eval_steps", 500)) or 500))]
    _max_steps = int(cfg.get("max_steps", 0) or 0)
    if _max_steps > 0:
        cadence_args += ["--max_steps", str(_max_steps)]
        log(f"[train] step cap: max_steps={_max_steps} (overrides epochs)")
    # "No test set" → tell qwen3 to skip eval even if the packed dir has a test/
    # subdir (it otherwise force-evals on it).
    if cfg.get("no_eval"):
        cadence_args += ["--skip_eval", "true"]
        log("[train] no_eval: evaluation disabled (ignoring any test/ split)")
    log(f"[train] cadence: eval={_evs} save={_svs}" + (f" every {cfg.get('eval_steps')} steps" if _evs == 'steps' else ""))
    last_loss = _run_loss([
        # venv python's torch.distributed.run (sys.executable is the venv python
        # in the run phase) — not a system `torchrun` that'd miss the venv torch.
        sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={gpus}",
        f"--master_port={_free_port()}",  # unique per launch — concurrent sweep trials else clash on 29500
        os.path.join(TTS_DIR, "qwen3_tts_flash.py"),
        "--model_name_or_path", model,
        # Parent packed dir — qwen3 reads train/ for training and test/ for the
        # per-epoch/step eval loss; a flat dir trains on the root.
        "--train_file", packed,
        "--output_dir", out_dir,
        "--do_train",
        "--num_train_epochs", str(epochs),
        "--per_device_train_batch_size", str(batch),
        "--gradient_accumulation_steps", str(grad_accum),
        "--learning_rate", str(lr),
        "--warmup_steps", str(int(cfg.get("warmup_steps", 0))),
        "--lr_scheduler_type", str(cfg.get("lr_scheduler_type") or "linear"),
        "--block_size", str(block_size),
        "--save_total_limit", "3",
        *cadence_args,
        "--gradient_checkpointing", "true",
        "--ddp_find_unused_parameters", "false",
        "--dataloader_num_workers", "5",
        "--remove_unused_columns", "false",
        "--report_to", report_flag,
        *dtype_args,
        *lora_args,
    ], work, env)

    # ---- evaluation (CER / MOS / similarity) on the test set ----
    eval_methods = [m for m in (cfg.get("eval_methods") or []) if m in ("cer", "mos", "similarity")]
    if eval_methods and os.path.isdir(out_dir):
        # Phase marker → the UI shows "running · evaluating" while the eval model
        # loads + scores (tts_eval then emits its own per-sample progress).
        log("[AUTOTRAIN_PROGRESS] step=evaluating percent=0")
        # eval set, in priority order: (1) a separate packed test dataset, (2) the
        # dataset's own held-out test split (test_from_split), (3) else the train
        # dir (sampled — no held-out set, scores generation on the train shards).
        test_ds = cfg.get("test_dataset") or {}
        eval_dir = train_dir
        if isinstance(test_ds, dict) and test_ds.get("packed_uri"):
            eval_dir = os.path.join(work, "eval_packed")
            _download_s3_prefix(test_ds, test_ds["packed_uri"], eval_dir)
            # a split-aware separate test dataset → prefer a held-out test split,
            # else fall back to its sole split subdir (e.g. a 'default'-named pack)
            # so an arbitrarily-named single split still evals instead of silently
            # finding no index.json at the prefix root.
            if not _has_index(eval_dir):
                esubs = _split_subdirs(eval_dir)
                pick = next((s for s in _TEST_SPLIT_NAMES if s in esubs), None) \
                    or (esubs[0] if esubs else None)
                if pick:
                    eval_dir = os.path.join(eval_dir, pick)
        elif cfg.get("test_from_split") and split_test_dir:
            eval_dir = split_test_dir
        try:
            _ensure_eval_deps(sys.executable, eval_methods, env)
            eval_cmd = [
                sys.executable, "-u", os.path.join(TTS_DIR, "tts_eval.py"),
                "--model_dir", out_dir, "--eval_dir", eval_dir,
                "--out_dir", os.path.join(work, "eval_audio"),
                "--methods", ",".join(eval_methods),
                "--max_samples", str(int(cfg.get("eval_max_samples", 64))),
            ]
            if cfg.get("language"):
                eval_cmd += ["--language", str(cfg["language"])]
            if cfg.get("eval_asr_model"):
                eval_cmd += ["--asr_model", str(cfg["eval_asr_model"])]
            step(eval_cmd, work)  # @@METRIC {tts_eval:{…}} is parsed by the gateway
        except Exception as e:  # noqa: BLE001 — eval is best-effort; training already succeeded
            log(f"[eval] TTS evaluation failed (training is unaffected): {e}")

    # ---- upload checkpoint to S3 + optional HF ----
    art = cfg.get("artifacts") or {}
    s3_uri = None
    if art.get("bucket") and os.path.isdir(out_dir):
        cli = _s3_client(art)
        base_key = art["prefix"].rstrip("/") + "/model"
        # Collect the final-model files first (skip intermediate `checkpoint-N/`
        # dirs — HF keeps save_total_limit of them, several GB each) so we know the
        # total and can log steady progress: a multi-GB upload is otherwise silent
        # for minutes and the run looks stuck on the logs tab.
        uploads: list[tuple[str, str]] = []  # (local_path, rel_key)
        total_bytes = 0
        for root, dirs, files in os.walk(out_dir):
            dirs[:] = [d for d in dirs if not d.startswith("checkpoint-")]
            for fn in files:
                fp = os.path.join(root, fn)
                uploads.append((fp, os.path.relpath(fp, out_dir)))
                try:
                    total_bytes += os.path.getsize(fp)
                except OSError:
                    pass
        n = len(uploads)
        log(f"[upload] uploading {n} file(s) · {total_bytes / 1e6:.0f} MB → s3://{art['bucket']}/{base_key}/ …")
        log("[AUTOTRAIN_PROGRESS] step=uploading percent=0")
        done_bytes = 0
        last_pct = -1
        for i, (fp, rel) in enumerate(uploads, 1):
            cli.upload_file(fp, art["bucket"], f"{base_key}/{rel}")
            try:
                done_bytes += os.path.getsize(fp)
            except OSError:
                pass
            pct = int(done_bytes * 100 / total_bytes) if total_bytes else int(i * 100 / max(1, n))
            # Emit on each ~5% tick (and the final file) so the logs tab + the
            # "running · uploading N%" header stay live without flooding the log.
            if pct >= last_pct + 5 or i == n:
                last_pct = pct
                log(f"[upload] {i}/{n} files · {done_bytes / 1e6:.0f}/{total_bytes / 1e6:.0f} MB ({pct}%)")
                log(f"[AUTOTRAIN_PROGRESS] step=uploading percent={pct}")
        s3_uri = f"s3://{art['bucket']}/{base_key}/"
        log(f"[upload] checkpoint → {s3_uri}")

    hf_repo = None
    if cfg.get("hf_push_repo") and cfg.get("hf_token"):
        try:
            from huggingface_hub import HfApi
            log(f"[upload] pushing model to Hugging Face → {cfg['hf_push_repo']} …")
            log("[AUTOTRAIN_PROGRESS] step=pushing_hf percent=0")
            HfApi().upload_folder(
                folder_path=out_dir, repo_id=cfg["hf_push_repo"],
                repo_type="model", token=cfg["hf_token"],
            )
            hf_repo = cfg["hf_push_repo"]
            log(f"[upload] pushed → https://huggingface.co/{hf_repo}")
        except Exception as e:  # noqa: BLE001
            log(f"[upload] HF push failed: {e}")

    emit("ARTIFACT", {"s3_uri": s3_uri, "hf_repo": hf_repo})
    emit("DONE", {"best": ({"loss": last_loss} if last_loss is not None else None),
                  "epochs": epochs, "stopped_early": False})


def _cleanup_workdir(cfg: dict) -> None:
    """Remove this run's per-run work dir once artifacts are in S3 (mirrors the
    Whisper trainer). Best model lives in S3; the local checkpoints/packed/audio
    are disposable. Skipped when cleanup_checkpoints is off."""
    import shutil
    if not cfg.get("cleanup_checkpoints", True) or not _RUN_WORKDIR:
        return
    shutil.rmtree(_RUN_WORKDIR, ignore_errors=True)
    log(f"[trainer] cleaned work dir: {_RUN_WORKDIR}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--deps-only", action="store_true",
                    help="install dependencies then exit (used by the sweep orchestrator)")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    try:
        _ensure_venv(cfg)  # deps phase builds the uv venv; run phase verifies it
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
