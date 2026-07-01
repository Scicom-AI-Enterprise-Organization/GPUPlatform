#!/usr/bin/env python3
"""Post-training Label-platform export for autotrain TTS.

Shipped to the run's VM as the whole `tts/` dir (so `chinidataset`, `tts_eval`,
and `tts_infer` import as siblings) and run with the TTS trainer venv
(torch/transformers/neucodec/soundfile/boto3). It:

  1. downloads the finetuned model from S3 (reusing tts_infer._download_model),
  2. picks N eval texts — the held-out *test* split if the run has one, else a
     random sample of the *train* split (decoded from the packed ChiniDataset the
     same way tts_eval reads it),
  3. synthesizes each text with the model + NeuCodec (the tts_eval/tts_infer core),
  4. uploads the WAVs to S3 under a per-run key prefix,

then prints ONE structured line the gateway parses:

  @@LABEL {"bucket": "...", "items": [{"key": "<object key>", "text": "<transcript>"}], "count": N}
  @@LABEL {"error": "..."}

Config (JSON via --config):
  {model_s3, region, endpoint, access_key, secret_key,
   packed_uri,                 # s3:// prefix of the packed ChiniDataset to read texts from
   split_subdir,               # "" (flat root) | "train" | "test" — subdir under packed_uri with index.json
   random,                     # true → random.sample N (train); false → first N (test)
   n_samples, seed,
   reject_keywords?,           # phrases to drop from the text pool (case-insensitive, space-collapsed)
   label_speakers?,            # speaker names to voice / group by
   per_speaker?,               # true → N clips from EACH speaker's own utterances (else round-robin)
   label_speaker_prefix?,      # prefix each task transcript with the speaker name
   speaker?, gpu?, max_new_tokens?,
   upload_bucket, upload_prefix}   # upload_prefix = full key prefix, no trailing slash
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Same dir as tts_eval / tts_infer / chinidataset (shipped together).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def emit(obj: dict) -> None:
    print("@@LABEL " + json.dumps(obj), flush=True)


def log(m: str) -> None:
    print(f"[label-export] {m}", flush=True)


def _upload_client(cfg: dict):
    import boto3
    from botocore.client import Config as BotoConfig

    return boto3.client(
        "s3", region_name=cfg.get("region") or "us-east-1",
        endpoint_url=cfg.get("endpoint") or None,
        aws_access_key_id=cfg.get("access_key") or None,
        aws_secret_access_key=cfg.get("secret_key") or None,
        config=BotoConfig(signature_version="s3v4"),
    )


def _collect_texts(cfg: dict) -> list[tuple[str, str, str]]:
    """Download the chosen packed split + decode utterances → [(prompt_text, ref_text, speaker)].
    For the train split a random sample of N is taken (scanning a bounded pool);
    for the test split the first N well-formed utterances are used.

    Two optional filters/groupings (config):
      * reject_keywords — drop any utterance whose transcript contains one of these
        phrases (case-insensitive, whitespace-collapsed, so "E M G S" matches any
        spacing). Applied before sampling, in every mode.
      * label_speakers + per_speaker — see below.

    Speaker handling:
      * per_speaker=true (and label_speakers given): take N clips from EACH listed
        speaker's OWN utterances (matched on the packed "<|im_start|>{speaker}:"
        prefix, case-insensitive). The gateway later groups these into one project
        per speaker. Total returned ≈ N × len(speakers).
      * label_speakers given, per_speaker=false: balance N clips round-robin across
        the names (re-voicing arbitrary transcripts) — the original behaviour.
      * no speakers: keep each clip's original packed voice (speaker recovered from
        the prefix so the optional transcription prefix can still tag it)."""
    import random as _random
    import re as _re

    from transformers import AutoTokenizer
    # tts_eval lives in the same shipped dir; importing it pulls only stdlib.
    from tts_eval import _read_packed, _utterance_parts, _IM_START, _SPEECH_START
    from tts_infer import _download_model

    n = max(1, int(cfg.get("n_samples") or 8))
    packed_uri = cfg["packed_uri"]
    subdir = (cfg.get("split_subdir") or "").strip("/")
    is_random = bool(cfg.get("random"))
    if is_random:
        _random.seed(int(cfg.get("seed") or 42))

    # Reject keywords: case-insensitive substring match on the transcript, with
    # whitespace collapsed on BOTH sides so a spaced keyword ("E M G S") still
    # matches a differently-spaced occurrence.
    def _norm(s: str) -> str:
        return _re.sub(r"\s+", " ", s or "").strip().lower()

    reject = [k for k in (_norm(x) for x in (cfg.get("reject_keywords") or [])) if k]

    def _rejected(text: str) -> bool:
        t = _norm(text)
        return any(k in t for k in reject)

    # Recover a clip's source speaker from its "<|im_start|>{speaker}: …" prefix.
    def _spk(pt: str) -> str:
        left = pt.split(_SPEECH_START, 1)[0].replace(_IM_START, "").strip()
        return left.split(":", 1)[0].strip() if ":" in left else ""

    speakers = [str(s).strip() for s in (cfg.get("label_speakers") or []) if str(s).strip()]
    per_speaker = bool(cfg.get("per_speaker")) and bool(speakers)

    # Reuse the model download helper to pull the packed shards (same listing /
    # size-cached fetch); point it at the packed prefix instead of the model.
    packed_dir = cfg.get("packed_dir") or "/tmp/sgpu-tts-label-packed"
    _download_model({**cfg, "model_s3": packed_uri, "model_dir": packed_dir})
    # Resolve the dir that actually holds index.json: the requested subdir → a flat
    # pack (index.json at the root) → else a single named split subdir (e.g.
    # `default`, produced when the source dataset had no split column), preferring
    # train/. _read_packed reads <read_dir>/index.json directly, so read_dir must be
    # the split dir itself.
    read_dir = os.path.join(packed_dir, subdir) if subdir else packed_dir
    if not os.path.exists(os.path.join(read_dir, "index.json")):
        read_dir = packed_dir  # flat?
        if not os.path.exists(os.path.join(read_dir, "index.json")):
            try:
                subs = [d for d in sorted(os.listdir(packed_dir))
                        if os.path.isdir(os.path.join(packed_dir, d))
                        and os.path.exists(os.path.join(packed_dir, d, "index.json"))]
            except OSError:
                subs = []
            # prefer train/ then default/ then the first split with an index
            pick = next((s for s in ("train", "default") if s in subs), subs[0] if subs else None)
            if pick:
                read_dir = os.path.join(packed_dir, pick)
    log(f"reading texts from {read_dir} (random={is_random}, want {n}"
        f"{', per-speaker' if per_speaker else ''}{f', rejecting {len(reject)} keyword(s)' if reject else ''})")

    tok = AutoTokenizer.from_pretrained(cfg["model_dir"])

    if per_speaker:
        # Bucket utterances by their source speaker (case-insensitive match to the
        # requested names), then take N from each speaker's own clips. Scan a
        # bounded pool per speaker so a huge split doesn't decode end-to-end.
        per_pool = n if not is_random else max(n * 20, 500)
        want = {s.lower(): s for s in speakers}             # lower → canonical name
        buckets: dict[str, list[tuple[str, str, str]]] = {s: [] for s in speakers}
        scan_cap = per_pool * len(speakers)
        scanned = 0
        for utt in _read_packed(read_dir):
            if scanned >= scan_cap or all(len(buckets[s]) >= per_pool for s in speakers):
                break
            parts = _utterance_parts(tok, utt)
            if parts is None:
                continue
            prompt_text, _ref_codes, ref_text = parts
            ref_text = (ref_text or "").strip()
            if not ref_text or _rejected(ref_text):
                continue
            canon = want.get(_spk(prompt_text).lower())
            if canon is None or len(buckets[canon]) >= per_pool:
                continue
            buckets[canon].append((prompt_text, ref_text, canon))
            scanned += 1
        out: list[tuple[str, str, str]] = []
        for spk in speakers:
            b = buckets[spk]
            if not b:
                log(f"speaker {spk!r}: no matching clips in the dataset — skipped")
                continue
            sel = _random.sample(b, min(n, len(b))) if is_random else b[:n]
            out.extend(sel)
        from collections import Counter
        bal = ", ".join(f"{s}×{c}" for s, c in sorted(Counter(s for _, _, s in out).items()))
        log(f"per-speaker selection ({len(out)} clip(s)): {bal or 'none'}")
        return out

    # Single-pool modes: scan a bounded pool of utterances (a packed record holds
    # several) so a huge train split doesn't decode end-to-end just to sample N.
    pool_cap = n if not is_random else max(n * 20, 500)
    pool: list[tuple[str, str]] = []
    for utt in _read_packed(read_dir):
        parts = _utterance_parts(tok, utt)
        if parts is None:
            continue
        prompt_text, _ref_codes, ref_text = parts
        ref_text = (ref_text or "").strip()
        if not ref_text or _rejected(ref_text):
            continue
        pool.append((prompt_text, ref_text))
        if len(pool) >= pool_cap:
            break
    if not pool:
        return []
    selected = _random.sample(pool, min(n, len(pool))) if is_random else pool[:n]

    # Balance the generation across named speakers: reassign each clip's voice
    # round-robin so e.g. 2 speakers + 32 samples → 16 each. The model is
    # speaker-name-conditioned (prompt "<|im_start|>{speaker}: {text}<|speech_start|>",
    # mirrors pack_stage1.py / tts_infer), so we keep each clip's transcript but
    # swap in the chosen speaker name. No speakers given → original packed voices.
    if speakers:
        out = []
        for i, (_pt, ref_text) in enumerate(selected):
            spk = speakers[i % len(speakers)]
            out.append((f"{_IM_START}{spk}: {ref_text}{_SPEECH_START}", ref_text, spk))
        from collections import Counter
        bal = ", ".join(f"{s}×{c}" for s, c in sorted(Counter(s for _, _, s in out).items()))
        log(f"balancing {len(out)} clip(s) across {len(speakers)} speaker(s): {bal}")
        return out
    # No balancing: keep each clip's original packed voice.
    return [(pt, rt, _spk(pt)) for (pt, rt) in selected]


def _synthesize_all(cfg: dict, utts: list[tuple[str, str]]) -> list[dict]:
    """Generate + upload one WAV per utterance. Returns [{key, text}] for the
    clips that produced audio. Model is loaded once and reused."""
    import io

    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from neucodec import NeuCodec

    from tts_eval import _decode_neucodec, _SPEECH_TOK, _IM_END

    use_cuda = torch.cuda.is_available() and str(cfg.get("gpu") or "auto").strip().lower() != "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    log(f"loading model {cfg['model_dir']} + NeuCodec (cuda={use_cuda}) …")
    _t = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model_dir"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_dir"], torch_dtype=dtype)
    model = (model.cuda() if use_cuda else model).eval()
    # NeuCodec decoder per the run's codec choice (cfg["tts_codec"]): upstream
    # neuphonic/neucodec (24 kHz, default) or the Scicom 44k-d20 fork (44.1 kHz).
    if str(cfg.get("tts_codec") or "neucodec").strip().lower() in ("neucodec-44k", "scicom-44k", "44k", "fork"):
        neu = NeuCodec._from_pretrained(model_id="Scicom-intl/neucodec-44k-d20", decoder_depth=20).eval()
    else:
        neu = NeuCodec.from_pretrained("neuphonic/neucodec").eval()
    neu = neu.cuda() if use_cuda else neu
    im_end_id = tok.convert_tokens_to_ids(_IM_END)
    max_new = int(cfg.get("max_new_tokens") or 1024)
    log(f"loaded in {time.time() - _t:.1f}s; synthesizing {len(utts)} clip(s) …")

    import re as _re

    # Optionally prefix each task's transcription with the speaker name (e.g.
    # "TM_Mandarin: hello world") so labellers see who's speaking.
    speaker_prefix = bool(cfg.get("label_speaker_prefix"))

    cli = _upload_client(cfg)
    bucket = cfg["upload_bucket"]
    base_key = cfg["upload_prefix"].strip("/")
    items: list[dict] = []
    for idx, (prompt_text, ref_text, speaker) in enumerate(utts):
        try:
            ids = tok(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids
            if use_cuda:
                ids = ids.cuda()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_new, do_sample=True,
                                     temperature=0.8, top_p=0.95, eos_token_id=im_end_id)
            gen_ids = out[0, ids.shape[-1]:].tolist()
            codes = [int(x) for x in _SPEECH_TOK.findall(tok.decode(gen_ids, skip_special_tokens=False))]
            if not codes:
                log(f"clip {idx}: no speech tokens; skip")
                continue
            wav, sr = _decode_neucodec(neu, codes)
            buf = io.BytesIO()
            sf.write(buf, wav, int(sr), format="WAV", subtype="PCM_16")
            # Tag the filename with the (sanitised) speaker so the balance is
            # visible/traceable in the Label project; key stays S3-safe.
            safe = _re.sub(r"[^A-Za-z0-9._-]+", "-", speaker).strip("-") if speaker else ""
            key = f"{base_key}/{idx:04d}{('_' + safe) if safe else ''}.wav"
            cli.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
            task_text = f"{speaker}: {ref_text}" if (speaker_prefix and speaker) else ref_text
            items.append({"key": key, "text": task_text, "speaker": speaker})
            tag = f" [{speaker}]" if speaker else ""
            log(f"clip {idx + 1}/{len(utts)}{tag}: {len(wav) / int(sr):.2f}s → s3://{bucket}/{key}")
        except Exception as e:  # noqa: BLE001
            log(f"clip {idx + 1}/{len(utts)} failed: {e}")
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)

    # Pin the GPU BEFORE importing torch (chosen index becomes device 0).
    sel = str(cfg.get("gpu") or "auto").strip().lower()
    if sel == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif sel.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = sel
    else:
        from tts_infer import _pick_gpu
        g = _pick_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = g if g is not None else ""

    # Download the model first (also used by the tokenizer in _collect_texts).
    from tts_infer import _download_model
    _download_model(cfg)

    utts = _collect_texts(cfg)
    if not utts:
        emit({"error": "no eval texts found in the packed dataset"})
        return 1
    items = _synthesize_all(cfg, utts)
    if not items:
        emit({"error": "synthesis produced no audio"})
        return 1
    emit({"bucket": cfg["upload_bucket"], "items": items, "count": len(items)})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        emit({"error": str(e)})
        sys.exit(1)
