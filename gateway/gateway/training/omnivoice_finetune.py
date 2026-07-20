"""Autotrain orchestrator for an OmniVoice (k2-fsa/OmniVoice) TTS finetune.

Sibling of `tts_finetune.py`, but a fundamentally different TTS stack: OmniVoice
(Qwen3-0.6B diffusion LM) finetuned via OmniVoice's OWN pipeline
(`accelerate -m omnivoice.cli.train`, full finetune — or the vendored
`omnivoice/lora_train.py` when `use_lora` is set, which wraps the Qwen3 backbone
in a peft LoRA adapter and merges it back into a plain checkpoint at the end),
with Higgs-codec audio tokens packed as WebDataset shards (NOT NeuCodec →
ChiniDataset). The standalone authority is `autotrain/omnivoice/` — the
vendored `omnivoice/` dir here mirrors its scripts.

Two modes (same `--config` contract + `@@…` log protocol the gateway parses):
  • pack-only  → resolve the audio dataset, build OmniVoice manifests, Higgs-
                 tokenize to WebDataset shards, upload to S3, emit @@PACKED.
                 The gateway turns that into a `kind=omnivoice_packed` dataset.
  • train      → download the packed shards, count steps/epoch, patch the config,
                 `accelerate` train (emit @@STEP), clean the checkpoint + bundle
                 the Higgs codec, CER/MOS eval, upload, emit @@ARTIFACT/@@DONE.

Deps differ from the gemma/MoE siblings: torch 2.8 + cu128 + OmniVoice's repo
install (NOT the cu13/torch-2.12 + FA stack). Built into a dedicated venv
(default /share/autotrain-omnivoice). The deps install reuses the slow/flaky-
uplink resilience patterns from llm_finetune (system git, retry+kill, lock-clear).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request

# Reuse the (light, NeuCodec-free-at-import) dataset resolver + S3 helpers from the
# sibling TTS orchestrator — the gateway ships tts_finetune.py alongside this for
# omnivoice runs. build_dataset resolves S3/HF/label audio → meta.jsonl + sources.
from tts_finetune import (  # noqa: E402
    build_dataset, emit, log, _s3_client, _upload_s3_dir, _download_s3_prefix,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OMNI_DIR = os.path.join(THIS_DIR, "omnivoice")
DEFAULT_VENV = "/share/autotrain-omnivoice"
OMNIVOICE_REPO = "https://github.com/k2-fsa/OmniVoice"
HIGGS_TOKENIZER = "eustlb/higgs-audio-v2-tokenizer"
_RUN_WORKDIR = None

# OmniVoice trainer console log → step/loss. Confirmed against
# omnivoice/training/checkpoint.py TrainLogger.log_metrics: every logging_steps the
# trainer prints (to stderr, merged into stdout here):
#   "Step 25 | train/loss: 4.5231 | train/learning_rate: 1.00e-05 | train/grad_norm: …"
# It also emits a tqdm bar with a "loss=4.5231" postfix every step — but that line
# carries no "Step N" token, so _STEP_RE never matches it (no double-count). We need
# BOTH a step and a loss on a line to emit @@STEP, so only the console line fires.
_STEP_RE = re.compile(r"(?:global_step|step)[\s=:]+(\d+)", re.I)
_LOSS_RE = re.compile(r"loss['\"]?[\s=:]+([0-9.eE+-]+)", re.I)


# --------------------------------------------------------------------------
# resilient install (mirrors llm_finetune; OmniVoice is a git+ install over a
# possibly slow/flaky uplink — see autotrain/omnivoice/CLAUDE.md HF-Xet note)
# --------------------------------------------------------------------------
def _install_env() -> dict:
    return {
        **os.environ, "PIP_CONSTRAINT": "", "PIP_REQUIRE_HASHES": "0",
        "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT", "600"),
        "UV_LOCK_TIMEOUT": os.environ.get("UV_LOCK_TIMEOUT", "60"),
        "GIT_HTTP_LOW_SPEED_LIMIT": os.environ.get("GIT_HTTP_LOW_SPEED_LIMIT", "1000"),
        "GIT_HTTP_LOW_SPEED_TIME": os.environ.get("GIT_HTTP_LOW_SPEED_TIME", "120"),
        # Xet stalls on the tm/RunPod uplink (603MB higgs blob never finalizes) —
        # force plain HTTPS for every HF pull (higgs codec, OmniVoice, whisper).
        "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1"),
        "HF_HUB_ENABLE_HF_TRANSFER": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0"),
        "TOKENIZERS_PARALLELISM": "false",
    }


def _run_resilient(cmd: list[str], env: dict, per_attempt_s: int = 2400, attempts: int = 4) -> None:
    """Run `cmd`, retrying transient failures and SIGKILLing a true hang (a stalled
    download/clone over a flaky uplink). Backoff; raise after the last attempt."""
    for i in range(attempts):
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
            log(f"[deps] attempt {i + 1}/{attempts} exceeded {per_attempt_s}s (stalled fetch) — killed.")
        if not failed:
            return
        if i == attempts - 1:
            raise subprocess.CalledProcessError(1, cmd)
        wait = 15 * (i + 1)
        log(f"[deps] command failed (attempt {i + 1}/{attempts}) — slow/flaky uplink; retrying in {wait}s …")
        time.sleep(wait)


def _clear_uv_git_locks(env: dict) -> None:
    try:
        cache = subprocess.check_output(["uv", "cache", "dir"], text=True, env=env, timeout=20).strip()
    except Exception:
        cache = env.get("UV_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "uv")
    locks = os.path.join(cache, "git-v0", "locks")
    try:
        for f in os.listdir(locks):
            try:
                os.remove(os.path.join(locks, f))
            except OSError:
                pass
    except OSError:
        pass


def _git_clone_shallow(url: str, dest: str, env: dict) -> None:
    if os.path.isdir(os.path.join(dest, ".git")):
        log(f"[deps] reusing existing clone {dest}")
        return
    if os.path.isdir(dest):
        shutil.rmtree(dest, ignore_errors=True)
    log(f"[deps] cloning {url} → {dest} (shallow, system git) …")
    _run_resilient(["git", "clone", "--depth", "1", "--no-tags", "--single-branch", url, dest],
                   env, per_attempt_s=1200)


# --------------------------------------------------------------------------
def _ensure_venv(cfg: dict) -> str:
    """Create/reuse the OmniVoice uv venv (torch 2.8/cu128 + the OmniVoice repo).
    Idempotent: skips when omnivoice already imports. Separate from the gemma/TTS
    venvs (different torch). Returns its python."""
    venv = (cfg.get("venv_path") or DEFAULT_VENV).rstrip("/")
    py = os.path.join(venv, "bin", "python")
    env = _install_env()
    have_uv = shutil.which("uv") is not None

    def _present() -> bool:
        try:
            subprocess.check_call([py, "-c", "import omnivoice, torch, jiwer, utmosv2, webdataset, peft"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    if os.path.exists(py) and _present():
        log(f"[deps] OmniVoice venv ready: {py}")
        return py

    def _pip(*args: str) -> None:
        if have_uv:
            _clear_uv_git_locks(env)
            _run_resilient(["uv", "pip", "install", "--python", py, *args], env)
        else:
            _run_resilient([py, "-m", "pip", "install", "-q", "--timeout", "600", "--retries", "5", *args], env)

    if not os.path.exists(py):
        log(f"[deps] creating venv {venv} …")
        if have_uv:
            subprocess.check_call(["uv", "venv", venv, "--python", "3.12"], env=env)
        else:
            subprocess.check_call([sys.executable, "-m", "venv", venv], env=env)
            subprocess.check_call([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], env=env)

    # OmniVoice pins torch 2.8 + cu128 (its dep is torch>=2.4, so 2.8 is kept).
    log("[deps] installing torch 2.8.0 (cu128) …")
    _pip("torch==2.8.0", "torchaudio==2.8.0", "--index-url", "https://download.pytorch.org/whl/cu128")
    # OmniVoice itself — git+ install fails inside uv on a flaky uplink, so clone
    # with system git (shallow, abortable) then install the local checkout.
    # NB: install the BASE package, NOT `.[eval]` — that extra (s3prl/funasr) pins
    # an ancient numba==0.53.1 → llvmlite 0.36.0, which has no Python-3.12 wheel and
    # FAILS to build in a fresh py3.12 venv (the standalone only got away with it by
    # pip-ing into the pod's pre-populated system python). We don't use OmniVoice's
    # ASR-eval scripts anyway — our eval is the vendored tts_eval.py (Whisper + jiwer
    # + UTMOSv2 below). Base librosa then resolves a modern, py3.12-compatible numba.
    src = os.path.join(os.path.dirname(venv), "OmniVoice")
    _git_clone_shallow(OMNIVOICE_REPO, src, env)
    log("[deps] installing OmniVoice (base, no [eval]) …")
    # Force a modern numba + a numba-compatible numpy: librosa pulls numba, and
    # omnivoice's unbounded `numpy` lets uv pick numpy 2.3 (which NO numba supports
    # yet) → uv backtracks numba to ==0.53.1 (sdist only → llvmlite 0.36.0, no
    # py3.12 wheel → build FAILS). Pinning numba>=0.59 + numpy<2.3 makes uv resolve
    # a py3.12-installable numba (0.61) with numpy 2.2. Joint resolve with omnivoice.
    _pip("-e", src, "numba>=0.59", "numpy<2.3")
    _pip("git+https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2")
    _pip("jiwer", "webdataset", "hf_transfer", "huggingface_hub[cli]", "pyarrow",
         "boto3", "soundfile", "librosa", "datasets",
         "peft>=0.11")  # LoRA on the llm backbone (lora_train.py) — merged into base at save
    subprocess.check_call([py, "-c", "import omnivoice, jiwer, utmosv2, peft; print('omnivoice import OK')"], env=env)
    log(f"[deps] OmniVoice venv ready: {py}")
    return py


# --------------------------------------------------------------------------
# data: build_dataset (sibling) → local audio → OmniVoice manifests
# --------------------------------------------------------------------------
def _language_of(row: dict, cfg: dict) -> str:
    """Per-row language_id (en/zh/…). From an explicit language_field on the row,
    a per-speaker map, or a single default — NEVER a hard-coded en/zh map."""
    lf = cfg.get("language_field")
    if lf and row.get(lf):
        return str(row[lf]).strip()
    spk_map = cfg.get("speaker_language_map") or {}
    if row.get("speaker") in spk_map:
        return str(spk_map[row["speaker"]])
    return str(cfg.get("default_language") or "en")


def _s3_key_from_url(url: str, bucket: str | None) -> str | None:
    """Object key from an S3 URL (virtual-hosted or path-style), query stripped, %-decoded."""
    from urllib.parse import urlsplit, unquote
    key = unquote(urlsplit(url).path).lstrip("/")
    if bucket and key.startswith(bucket + "/"):
        key = key[len(bucket) + 1:]
    return key or None


def _materialize_local(work: str, sources_path: str, ds: dict) -> dict:
    """Ensure every audio source is a LOCAL file (OmniVoice's Higgs tokenizer reads
    from disk). build_dataset passes S3 rows through as the metadata's PRESIGNED
    URLs (for streaming) — but those expire (7-day X-Amz-Expires), so a plain GET
    403s. Download via boto3 with the dataset's live creds (key parsed from the URL),
    fall back to a direct fetch. Returns key → absolute local path.

    Downloads run on a thread pool: cross-region S3 (e.g. a KL bucket ↔ a US pod) is
    latency-bound, so fetching 1000s of small clips serially crawls — concurrency is
    a ~10× win (boto3 low-level clients are thread-safe; one per worker for safety)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    sources = json.load(open(sources_path))
    out: dict = {}
    bucket = ds.get("bucket")
    remote = []
    for s in sources:
        key, src = s["key"], s["src"]
        if isinstance(src, str) and src.startswith(("http://", "https://")):
            remote.append(s)
        else:
            out[key] = os.path.abspath(os.path.join(work, src))
    if not remote:
        return out

    tl = threading.local()

    def _client():
        c = getattr(tl, "c", None)
        if c is None:
            c = tl.c = _s3_client(ds)
        return c

    def _fetch(s: dict):
        key, src = s["key"], s["src"]
        dest = os.path.join(work, key)
        os.makedirs(os.path.dirname(dest) or work, exist_ok=True)
        obj_key = _s3_key_from_url(src, bucket)
        if obj_key and bucket:
            try:
                _client().download_file(bucket, obj_key, dest)
                return key, os.path.abspath(dest)
            except Exception as e:  # noqa: BLE001
                log(f"[data] s3 get fail {obj_key}: {e}")
        try:
            urllib.request.urlretrieve(src, dest)
            return key, os.path.abspath(dest)
        except Exception as e:  # noqa: BLE001
            log(f"[data] download fail {key}: {e}")
            return key, None

    with ThreadPoolExecutor(max_workers=32) as ex:
        for key, path in ex.map(_fetch, remote):
            if path:
                out[key] = path
    log(f"[data] materialized {sum(1 for s in remote if out.get(s['key']))}/{len(remote)} S3 clips (32-way)")
    return out


def _sanitize_id(speaker: str, filename: str, used: set) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    raw = re.sub(r"[^0-9A-Za-z._-]+", "_", f"{speaker}__{stem}").strip("_") or "x"
    rid = raw
    n = 0
    while rid in used:
        n += 1
        rid = f"{raw}_{n}"
    used.add(rid)
    return rid


def _relocate_data_lst(token_root: str) -> None:
    """Pre-packed shards were tokenized on a DIFFERENT pod (the pack run), so the
    ABSOLUTE tar/jsonl paths tokenize_audio.py baked into each split's data.lst are
    stale here. Rewrite them to the shards' actual downloaded location, preserving
    the `count duration` columns (omnivoice's webdataset_manifest_reader requires
    EXACTLY 4 space-separated fields and opens tar_path directly)."""
    fixed = 0
    for split in sorted(os.listdir(token_root)):
        sdir = os.path.join(token_root, split)
        lst = os.path.join(sdir, "data.lst")
        if not os.path.isfile(lst):
            continue
        out = []
        with open(lst) as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 4:
                    continue
                tar = os.path.join(sdir, "audios", os.path.basename(parts[0]))
                jsonl = os.path.join(sdir, "txts", os.path.basename(parts[1]))
                out.append(f"{tar} {jsonl} {parts[2]} {parts[3]}")
        with open(lst, "w") as f:
            f.write("\n".join(out) + ("\n" if out else ""))
        fixed += len(out)
    log(f"[pack] relocated {fixed} shard path(s) in data.lst under {token_root}")


def _bundle_eval_audio(eval_src: str, token_root: str) -> None:
    """Copy the eval set's reference audio next to the tokens and rewrite ref_audio
    to a RELATIVE path (eval_audio/<file>). The shards hold Higgs *tokens*, not
    audio, but the voice-clone eval (infer_batch) needs the raw ref clip — and a
    separate train pod only downloads this token tree, so bundle the clips here."""
    rows = [json.loads(l) for l in open(eval_src) if l.strip()]
    adir = os.path.join(token_root, "eval_audio")
    os.makedirs(adir, exist_ok=True)
    seen: dict = {}
    for r in rows:
        src = r.get("ref_audio")
        if not src:
            continue
        if src not in seen:
            ext = os.path.splitext(src)[1] or ".wav"
            name = f"ref_{len(seen):05d}{ext}"
            try:
                shutil.copy2(src, os.path.join(adir, name))
                seen[src] = name
            except Exception as e:  # noqa: BLE001
                log(f"[eval] ref audio copy failed {src}: {e}")
                continue
        r["ref_audio"] = os.path.join("eval_audio", seen[src])
    with open(os.path.join(token_root, "eval_test.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"[eval] bundled {len(seen)} ref clip(s) → {adir}")


def _absolutize_eval_audio(token_root: str) -> None:
    """Make eval_test.jsonl ref_audio absolute under token_root (stored relative by
    _bundle_eval_audio so it survives the pack→train pod move). No-op if already abs."""
    p = os.path.join(token_root, "eval_test.jsonl")
    if not os.path.isfile(p):
        return
    rows = [json.loads(l) for l in open(p) if l.strip()]
    changed = False
    for r in rows:
        ra = r.get("ref_audio")
        if ra and not os.path.isabs(ra):
            r["ref_audio"] = os.path.join(token_root, ra)
            changed = True
    if changed:
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_manifests(cfg: dict, work: str) -> dict:
    """Resolve the audio dataset → OmniVoice manifests under work/manifests/:
      train.jsonl / dev.jsonl  {id, audio_path, text, language_id}
      eval_test.jsonl          {id, text, ref_audio, ref_text, language_id}
    Held-out test = the dataset's test/dev split if present, else `n_test` per
    speaker. eval refs are a random same-speaker train clip (zero-shot cloning)."""
    sources_path, meta = build_dataset(cfg, work)
    key2path = _materialize_local(work, sources_path, cfg.get("dataset") or {})

    rows: list[dict] = []
    used: set = set()
    with open(meta) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            path = key2path.get(r["filename_audio"])
            if not path or not os.path.exists(path):
                continue
            spk = str(r.get("speaker") or "speaker")
            rows.append({
                "id": _sanitize_id(spk, r["filename_audio"], used),
                "audio_path": path,
                "text": str(r.get("text") or "").strip(),
                "language_id": _language_of(r, cfg),
                "speaker": spk,
                "split": str(r.get("split") or "train"),
            })
    if not rows:
        raise RuntimeError("no usable {audio, text} rows resolved from the dataset")

    # Split: explicit test/dev split wins; else hold out n_test per speaker.
    test_names = {"test", "validation", "valid", "dev"}
    has_test = any(r["split"] in test_names for r in rows)
    train, test = [], []
    if has_test:
        for r in rows:
            (test if r["split"] in test_names else train).append(r)
    else:
        n_test = int(cfg.get("eval_test_per_speaker") or 25)
        by_spk: dict = {}
        for r in rows:
            by_spk.setdefault(r["speaker"], []).append(r)
        rng = random.Random(int(cfg.get("split_seed") or 42))
        for _spk, items in sorted(by_spk.items()):
            items = items[:]
            rng.shuffle(items)
            n = min(n_test, max(0, len(items) - 1))
            test.extend(items[:n])
            train.extend(items[n:])

    mdir = os.path.join(work, "manifests")
    os.makedirs(mdir, exist_ok=True)

    def _write(path, items, keys):
        with open(path, "w", encoding="utf-8") as f:
            for x in items:
                f.write(json.dumps({k: x[k] for k in keys}, ensure_ascii=False) + "\n")

    keys = ["id", "audio_path", "text", "language_id"]
    _write(os.path.join(mdir, "train.jsonl"), train, keys)
    _write(os.path.join(mdir, "dev.jsonl"), test or train[: min(50, len(train))], keys)

    # eval_test.jsonl: same-speaker train clip as the voice-clone reference.
    train_by_spk: dict = {}
    for x in train:
        train_by_spk.setdefault(x["speaker"], []).append(x)
    rng2 = random.Random(int(cfg.get("split_seed") or 42) + 1)
    eval_rows = []
    for x in test:
        pool = train_by_spk.get(x["speaker"]) or train
        if not pool:
            continue
        ref = rng2.choice(pool)
        eval_rows.append({"id": x["id"], "text": x["text"], "ref_audio": ref["audio_path"],
                          "ref_text": ref["text"], "language_id": x["language_id"]})
    with open(os.path.join(mdir, "eval_test.jsonl"), "w", encoding="utf-8") as f:
        for x in eval_rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    log(f"[data] manifests: train={len(train)} dev/test={len(test)} eval={len(eval_rows)} → {mdir}")
    return {"dir": mdir, "splits": {"train": len(train), "dev": len(test)}, "n_eval": len(eval_rows)}


# --------------------------------------------------------------------------
def run(cfg: dict) -> None:
    global _RUN_WORKDIR
    work = tempfile.mkdtemp(prefix="omnivoice_")
    _RUN_WORKDIR = work
    py = cfg["_python"]
    venv_bin = os.path.dirname(py)
    env = {**os.environ}
    env.update({k: v for k, v in _install_env().items() if k.startswith(("HF_", "TOKENIZERS"))})
    # Activate the venv for child processes (accelerate/omnivoice launched via `py -m`
    # use sys.executable, but venv/bin on PATH + VIRTUAL_ENV lets any console-script /
    # JIT tool resolve from the venv too).
    env["VIRTUAL_ENV"] = os.path.dirname(venv_bin)
    env["PATH"] = venv_bin + ":" + env.get("PATH", "")
    art = cfg.get("artifacts") or {}  # NB: training_api sets "artifacts" (plural)
    # Higgs AUDIO codec id (NOT the LM tokenizer). On a PACK run cfg["tokenizer"] is
    # the codec from the pack request; on a TRAIN run the gateway sets cfg["tokenizer"]
    # to the LM/base model (k2-fsa/OmniVoice), so derive the codec from the packed
    # dataset's recorded value, falling back to the fixed OmniVoice codec — NEVER
    # bundle the base model as audio_tokenizer/ (it lacks preprocessor_config.json).
    if cfg.get("pack_only"):
        tokenizer = cfg.get("tokenizer") or HIGGS_TOKENIZER
    else:
        tokenizer = (cfg.get("dataset") or {}).get("higgs_tokenizer") or HIGGS_TOKENIZER
    token_root = os.path.join(work, "tokens")

    def step(cmd: list[str], cwd: str | None = None) -> None:
        log(f"[gateway] $ {' '.join(cmd)}")
        subprocess.check_call(cmd, cwd=cwd or work, env=env)

    # ---- 1. manifests + Higgs tokenization (shared by pack-only and train-from-raw) ----
    ds = cfg.get("dataset") or {}
    packed_uri = ds.get("packed_uri")
    if not packed_uri:
        man = build_manifests(cfg, work)
        for split in ("train", "dev"):
            jsonl = os.path.join(man["dir"], f"{split}.jsonl")
            if not os.path.exists(jsonl) or os.path.getsize(jsonl) == 0:
                continue
            out = os.path.join(token_root, split)
            step([py, "-u", os.path.join(OMNI_DIR, "tokenize_audio.py"),
                  "--input_jsonl", jsonl, "--out_dir", out,
                  "--tokenizer_path", tokenizer, "--samples_per_shard", "64"])
        # carry the eval manifest + its ref audio next to the tokens (bundled so a
        # separate train pod can run the voice-clone eval — ref clips live only here)
        if os.path.exists(os.path.join(man["dir"], "eval_test.jsonl")):
            _bundle_eval_audio(os.path.join(man["dir"], "eval_test.jsonl"), token_root)

        if cfg.get("pack_only"):
            s3_uri = _upload_s3_dir(art, token_root, art["prefix"].rstrip("/") + "/omnivoice_tokens")
            emit("PACKED", {"s3_uri": s3_uri, "splits": man["splits"], "tokenizer": tokenizer,
                            "format": "omnivoice", "samples": sum(man["splits"].values())})
            log("[pack] done (pack-only)")
            return
    else:
        log(f"[pack] reusing pre-packed OmniVoice tokens from {packed_uri}")
        _download_s3_prefix(ds, packed_uri, token_root)
        _relocate_data_lst(token_root)

    # ---- 2. train: count steps → patch config → accelerate ----
    train_cfg = os.path.join(work, "train_config.json")
    data_cfg = os.path.join(work, "data_config.json")
    shutil.copy2(os.path.join(OMNI_DIR, "train_config.json"), train_cfg)
    base_model = cfg.get("base_model") or "k2-fsa/OmniVoice"
    attn = cfg.get("attn_implementation") or "flex_attention"
    with open(data_cfg, "w") as f:
        json.dump({
            "train": [{"language_id": "mixed",
                       "manifest_path": [os.path.join(token_root, "train", "data.lst")]}],
            "dev": [{"language_id": "mixed",
                     "manifest_path": [os.path.join(token_root, "dev", "data.lst")]}],
        }, f, indent=2)

    c = json.load(open(train_cfg))
    c["init_from_checkpoint"] = base_model
    c["attn_implementation"] = attn
    if cfg.get("learning_rate"):
        c["learning_rate"] = float(cfg["learning_rate"])
    if cfg.get("batch_tokens"):
        c["batch_tokens"] = int(cfg["batch_tokens"])
    json.dump(c, open(train_cfg, "w"), indent=4)

    # steps_per_epoch from the real (packing) dataloader, then steps = epochs × spe.
    spe = None
    try:
        out = subprocess.check_output(
            [py, "-u", os.path.join(OMNI_DIR, "count_steps.py"),
             "--train_config", train_cfg, "--data_config", data_cfg],
            cwd=work, env=env, text=True)
        for ln in out.splitlines():
            if ln.startswith("STEPS_PER_EPOCH="):
                spe = int(ln.split("=", 1)[1])
        log(f"[train] steps_per_epoch={spe}")
    except Exception as e:  # noqa: BLE001
        log(f"[train] count_steps failed ({e}); using config steps as-is")

    max_steps = int(cfg.get("max_steps") or 0)
    epochs = int(cfg.get("max_epochs") or 1)
    c = json.load(open(train_cfg))
    if max_steps > 0:
        c["steps"] = max_steps
    elif spe:
        c["steps"] = spe * epochs
    if spe:
        c["save_steps"] = c["eval_steps"] = spe
    if cfg.get("logging_steps"):
        c["logging_steps"] = int(cfg["logging_steps"])
    json.dump(c, open(train_cfg, "w"), indent=4)
    total_steps = int(c["steps"])
    log(f"[train] steps={total_steps} (epochs={epochs}, max_steps={max_steps})")

    exp_dir = os.path.join(work, "exp")
    os.makedirs(exp_dir, exist_ok=True)
    # `accelerate launch …` via the venv python (== python -m accelerate.commands.launch)
    # so the launched trainer runs under the venv interpreter. `use_lora` swaps
    # `-m omnivoice.cli.train` for the vendored lora_train.py (same build_model_and_
    # tokenizer/OmniTrainer calls, plus a peft wrap on model.llm + a merge-back-to-
    # plain-checkpoint step at the end) — everything downstream (checkpoint-N naming,
    # @@STEP parsing, make_clean_checkpoint.py) is unchanged either way.
    lora_args: list[str] = []
    if cfg.get("use_lora"):
        _r = int(cfg.get("lora_r", 16))
        _ratio = cfg.get("lora_alpha_ratio")
        _alpha = int(round(_r * float(_ratio))) if _ratio is not None else int(cfg.get("lora_alpha", 32))
        _dropout = float(cfg.get("lora_dropout", 0.05))
        lora_args = ["--lora_r", str(_r), "--lora_alpha", str(_alpha),
                     "--lora_dropout", str(_dropout), "--lora_target_modules", "all-linear"]
        log(f"[train] LoRA enabled (r={_r}, alpha={_alpha}, dropout={_dropout}) — adapting the "
            f"Qwen3 backbone only; audio_embeddings/audio_heads stay fully trainable, merged into "
            f"a plain checkpoint at save (see lora_train.py)")
        trainer_entry = [os.path.join(OMNI_DIR, "lora_train.py")]
    else:
        trainer_entry = ["-m", "omnivoice.cli.train"]
    cmd = [py, "-m", "accelerate.commands.launch", "--gpu_ids", "0", "--num_processes", "1",
           *trainer_entry, "--train_config", train_cfg,
           "--data_config", data_cfg, "--output_dir", exp_dir, *lora_args]
    log(f"[gateway] $ {' '.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_loss = None
    last_emit_step = -1
    for line in p.stdout:  # type: ignore[union-attr]
        print(line, end="", flush=True)
        ms, ml = _STEP_RE.search(line), _LOSS_RE.search(line)
        if ms and ml:
            try:
                stp, loss = int(ms.group(1)), float(ml.group(1))
            except ValueError:
                continue
            last_loss = loss
            if stp != last_emit_step:
                last_emit_step = stp
                emit("STEP", {"step": stp, "loss": loss,
                              "percent": round(100 * stp / total_steps, 1) if total_steps else None})
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)

    # ---- 3. clean checkpoint + bundle Higgs ----
    ckpts = sorted(
        (d for d in os.listdir(exp_dir) if d.startswith("checkpoint-")),
        key=lambda d: int(d.split("-")[-1]) if d.split("-")[-1].isdigit() else 0,
    )
    if not ckpts:
        raise RuntimeError("training produced no checkpoint-* dir")
    last_ckpt = os.path.join(exp_dir, ckpts[-1])
    clean = os.path.join(work, "clean")
    step([py, "-u", os.path.join(OMNI_DIR, "make_clean_checkpoint.py"),
          "--ckpt", last_ckpt, "--out", clean, "--higgs", tokenizer])

    # ---- 4. CER/MOS eval (if requested + an eval manifest exists) ----
    methods = [m for m in (cfg.get("eval_methods") or []) if m in ("cer", "mos")]
    _absolutize_eval_audio(token_root)
    eval_manifest = os.path.join(token_root, "eval_test.jsonl")
    eval_results = None
    if methods and os.path.exists(eval_manifest) and os.path.getsize(eval_manifest) > 0:
        res_dir = os.path.join(work, "results")
        os.makedirs(res_dir, exist_ok=True)
        try:
            step([py, "-u", "-m", "omnivoice.cli.infer_batch", "--model", clean,
                  "--test_list", eval_manifest, "--res_dir", res_dir,
                  "--preprocess_prompt", "False", "--postprocess_output", "False",
                  "--batch_duration", "600", "--audio_chunk_threshold", "1000"])
            eval_results = os.path.join(clean, "eval_results.json")
            step([py, "-u", os.path.join(OMNI_DIR, "tts_eval.py"),
                  "--wav_dir", res_dir, "--test_list", eval_manifest,
                  "--methods", ",".join(methods), "--out", eval_results])
            try:
                emit("METRIC", {"tts_eval": json.load(open(eval_results))})
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            log(f"[eval] OmniVoice eval failed (non-fatal): {e}")

    # ---- 5. upload clean checkpoint (+ eval) → S3 (+ optional HF push) ----
    s3_uri = _upload_s3_dir(art, clean, art["prefix"].rstrip("/") + "/best-model")
    hf_repo = (cfg.get("hf_push_repo") or "").strip() or None
    if hf_repo:
        try:
            from huggingface_hub import HfApi
            HfApi(token=os.environ.get("HF_TOKEN")).upload_folder(
                folder_path=clean, repo_id=hf_repo, repo_type="model")
            log(f"[push] pushed clean checkpoint → {hf_repo}")
        except Exception as e:  # noqa: BLE001
            log(f"[push] HF push failed (non-fatal): {e}")
            hf_repo = None

    emit("ARTIFACT", {"s3_uri": s3_uri, "hf_repo": hf_repo})
    emit("DONE", {"best": ({"loss": last_loss} if last_loss is not None else None),
                  "epochs": epochs, "steps": total_steps,
                  "eval": (json.load(open(eval_results)) if eval_results and os.path.exists(eval_results) else None)})


def _cleanup_workdir(cfg: dict) -> None:
    if _RUN_WORKDIR and cfg.get("cleanup_checkpoints", True):
        shutil.rmtree(_RUN_WORKDIR, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--deps-only", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    try:
        py = _ensure_venv(cfg)
        if args.deps_only:
            log("[deps] ready (deps-only, arch=omnivoice)")
            return 0
        cfg["_python"] = py
        run(cfg)
        return 0
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        emit("ERROR", {"message": str(e)})
        return 1
    finally:
        try:
            _cleanup_workdir(cfg)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
