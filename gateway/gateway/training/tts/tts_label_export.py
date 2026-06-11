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


def _collect_texts(cfg: dict) -> list[tuple[str, str]]:
    """Download the chosen packed split + decode utterances → [(prompt_text, ref_text)].
    For the train split a random sample of N is taken (scanning a bounded pool);
    for the test split the first N well-formed utterances are used."""
    import random as _random

    from transformers import AutoTokenizer
    # tts_eval lives in the same shipped dir; importing it pulls only stdlib.
    from tts_eval import _read_packed, _utterance_parts
    from tts_infer import _download_model

    n = max(1, int(cfg.get("n_samples") or 8))
    packed_uri = cfg["packed_uri"]
    subdir = (cfg.get("split_subdir") or "").strip("/")
    is_random = bool(cfg.get("random"))

    # Reuse the model download helper to pull the packed shards (same listing /
    # size-cached fetch); point it at the packed prefix instead of the model.
    packed_dir = cfg.get("packed_dir") or "/tmp/sgpu-tts-label-packed"
    _download_model({**cfg, "model_s3": packed_uri, "model_dir": packed_dir})
    read_dir = os.path.join(packed_dir, subdir) if subdir else packed_dir
    if not os.path.exists(os.path.join(read_dir, "index.json")):
        # Fall back to the prefix root if the requested subdir has no index.
        read_dir = packed_dir
    log(f"reading texts from {read_dir} (random={is_random}, want {n})")

    tok = AutoTokenizer.from_pretrained(cfg["model_dir"])
    # Scan a bounded pool of utterances (a packed record holds several) so a huge
    # train split doesn't decode end-to-end just to sample N.
    pool_cap = n if not is_random else max(n * 20, 500)
    pool: list[tuple[str, str]] = []
    for utt in _read_packed(read_dir):
        parts = _utterance_parts(tok, utt)
        if parts is None:
            continue
        prompt_text, _ref_codes, ref_text = parts
        ref_text = (ref_text or "").strip()
        if not ref_text:
            continue
        pool.append((prompt_text, ref_text))
        if len(pool) >= pool_cap:
            break
    if not pool:
        return []
    if is_random:
        _random.seed(int(cfg.get("seed") or 42))
        return _random.sample(pool, min(n, len(pool)))
    return pool[:n]


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
    neu = NeuCodec.from_pretrained("neuphonic/neucodec").eval()
    neu = neu.cuda() if use_cuda else neu
    im_end_id = tok.convert_tokens_to_ids(_IM_END)
    max_new = int(cfg.get("max_new_tokens") or 1024)
    log(f"loaded in {time.time() - _t:.1f}s; synthesizing {len(utts)} clip(s) …")

    cli = _upload_client(cfg)
    bucket = cfg["upload_bucket"]
    base_key = cfg["upload_prefix"].strip("/")
    items: list[dict] = []
    for idx, (prompt_text, ref_text) in enumerate(utts):
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
            key = f"{base_key}/{idx:04d}.wav"
            cli.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
            items.append({"key": key, "text": ref_text})
            log(f"clip {idx + 1}/{len(utts)}: {len(wav) / int(sr):.2f}s → s3://{bucket}/{key}")
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
