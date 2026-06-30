#!/usr/bin/env python3
"""Standalone TTS synthesis for the Autotrain "Try it" playground (the TTS twin of
transcribe.py). Shipped to the run's VM over SSH and run with the TTS trainer venv
(torch/transformers/neucodec/soundfile/boto3). It downloads the finetuned model
from S3, generates speech tokens for the given text, NeuCodec-decodes them to a
waveform, and prints a single structured line:

  @@AUDIO {"wav_b64": "...", "sample_rate": 44100, "device": "cuda", "n_codes": N}  on success
  @@AUDIO {"error": "..."}                                                          on failure

Config (JSON via --config): {model_s3, region, endpoint, access_key, secret_key,
model_dir, text, speaker?, gpu?, max_new_tokens?, temperature?, top_p?}. The prompt
mirrors pack_stage1.py exactly: `<|im_start|>{speaker}: {text}<|speech_start|>`.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import threading
import time

_SPEECH_TOK = re.compile(r"<\|s_(\d+)\|>")


def emit(obj: dict) -> None:
    print("@@AUDIO " + json.dumps(obj), flush=True)


def log(m: str) -> None:
    print(f"[tryit] {m}", flush=True)


def _pick_gpu() -> str | None:
    """GPU with the most free memory (>5 GiB), else None (CPU) — so a try-it
    doesn't OOM or disturb training already running on the box."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
    except Exception:
        return None
    best, best_free = None, 0
    for line in out.strip().splitlines():
        try:
            idx, free = (x.strip() for x in line.split(","))
            free_i = int(free)
        except ValueError:
            continue
        if free_i > best_free:
            best, best_free = idx, free_i
    return best if (best is not None and best_free > 5000) else None


def _download_model(cfg: dict) -> str:
    import boto3
    from botocore.client import Config as BotoConfig

    s3 = cfg["model_s3"]
    assert s3.startswith("s3://"), f"bad model_s3: {s3}"
    bucket, _, prefix = s3[len("s3://"):].partition("/")
    prefix = prefix.rstrip("/") + "/"
    cli = boto3.client(
        "s3", region_name=cfg.get("region") or "us-east-1",
        endpoint_url=cfg.get("endpoint") or None,
        aws_access_key_id=cfg.get("access_key") or None,
        aws_secret_access_key=cfg.get("secret_key") or None,
        config=BotoConfig(signature_version="s3v4"),
    )
    dest = cfg.get("model_dir") or "/tmp/sgpu-tts-tryit-model"
    os.makedirs(dest, exist_ok=True)
    log(f"resolving model {s3} → {dest} (cached on the VM after the first call) …")
    t0 = time.time()
    n = fetched = total = 0
    for page in cli.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]; rel = key[len(prefix):]
            if not rel:
                continue
            fp = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(fp) or dest, exist_ok=True)
            if not (os.path.exists(fp) and os.path.getsize(fp) == obj["Size"]):
                size = int(obj["Size"])
                log(f"  ↓ {rel} ({size / 1e6:.0f} MB)")
                if size > 200 * 1e6:  # big shard → stream in-file % (else minutes of silence)
                    prog = {"b": 0, "next": 0.2}
                    lk = threading.Lock()

                    def _cb(chunk: int, _rel=rel, _size=size, _p=prog, _lk=lk) -> None:
                        with _lk:
                            _p["b"] += chunk
                            frac = _p["b"] / max(_size, 1)
                            if frac >= _p["next"] and frac < 1.0:
                                log(f"      {_rel}: {frac * 100:.0f}%  ({_p['b'] / 1e6:.0f}/{_size / 1e6:.0f} MB)")
                                _p["next"] += 0.2
                    cli.download_file(bucket, key, fp, Callback=_cb)
                else:
                    cli.download_file(bucket, key, fp)
                fetched += 1
            n += 1
            total += obj["Size"]
    if n == 0:
        raise RuntimeError(f"no model files found under {s3}")
    log(f"model ready: {n} files · {total / 1e6:.0f} MB · {fetched} fetched / {n - fetched} cached · {time.time() - t0:.1f}s")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)

    # Device pin BEFORE importing torch (chosen GPU becomes device 0).
    sel = str(cfg.get("gpu") or "auto").strip().lower()
    if sel == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""; want_cuda = False
    elif sel.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = sel; want_cuda = True
    else:
        g = _pick_gpu(); os.environ["CUDA_VISIBLE_DEVICES"] = g if g is not None else ""; want_cuda = g is not None

    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from neucodec import NeuCodec

    model_dir = _download_model(cfg)
    use_cuda = want_cuda and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else None
    log(f"device: {device}" + (f" ({gpu_name})" if gpu_name else "") + f" · dtype {str(dtype).replace('torch.', '')}")
    log("loading model + NeuCodec …")
    _t = time.time()
    tok = AutoTokenizer.from_pretrained(model_dir)
    # Plain causal LM for generation (the trainer's custom Model has no logits in
    # forward when labels=None); the saved merged weights are standard Qwen3.
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
    model = (model.cuda() if use_cuda else model).eval()
    # Scicom d20 fork: decoder_depth=20 matches the finetuned depth-20 decoder → 44.1 kHz.
    neu = NeuCodec._from_pretrained(model_id="Scicom-intl/neucodec-44k-d20", decoder_depth=20).eval()
    neu = neu.cuda() if use_cuda else neu
    log(f"loaded model (vocab {len(tok)}) + NeuCodec in {time.time() - _t:.1f}s")

    text = (cfg.get("text") or "").strip()
    if not text:
        emit({"error": "empty text"}); return 1
    speaker = (cfg.get("speaker") or "").strip()
    left = f"{speaker}: {text}" if speaker else text
    prompt = f"<|im_start|>{left}<|speech_start|>"  # mirrors pack_stage1.py
    _maxnew = int(cfg.get("max_new_tokens", 1024)); _temp = float(cfg.get("temperature", 0.8)); _tp = float(cfg.get("top_p", 0.95))
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    if use_cuda:
        ids = ids.cuda()
    log(f"prompt ({ids.shape[-1]} tok): {prompt!r}")
    log(f"generating ≤{_maxnew} speech tokens (temp={_temp}, top_p={_tp}) …")
    _t = time.time()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=_maxnew, do_sample=True,
            temperature=_temp, top_p=_tp, eos_token_id=im_end,
        )
    _gen_t = time.time() - _t
    gen = out[0, ids.shape[-1]:].tolist()
    gen_text = tok.decode(gen, skip_special_tokens=False)  # the raw generation (speech tokens) before NeuCodec
    codes = [int(n) for n in _SPEECH_TOK.findall(gen_text)]
    # Compact preview of the generation: collapse runs of <|s_N|> so the log stays readable.
    _preview = _SPEECH_TOK.sub("", gen_text).replace("<|im_end|>", "").strip()
    log(f"generated {len(gen)} tokens → {len(codes)} speech codes in {_gen_t:.1f}s"
        + (f" · non-speech text: {_preview[:120]!r}" if _preview else " · (pure speech tokens)"))
    if not codes:
        emit({"error": "model produced no speech tokens for this text", "prompt": prompt, "gen_text": gen_text[:4000]}); return 1
    with torch.no_grad():
        fsq = torch.tensor(codes, dtype=torch.long).reshape(1, 1, -1)
        if use_cuda:
            fsq = fsq.cuda()
        wav = neu.decode_code(fsq).squeeze().detach().cpu().float().numpy()
    sr = int(getattr(neu, "sample_rate", 44100))
    _dur = len(wav) / sr
    log(f"NeuCodec → {_dur:.2f}s audio @ {sr}Hz (RTF {_gen_t / max(_dur, 1e-6):.2f}) on {device}")
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")  # half the bytes of float for the b64 round-trip
    emit({"wav_b64": base64.b64encode(buf.getvalue()).decode(), "sample_rate": sr,
          "device": device, "n_codes": len(codes), "prompt": prompt, "gen_text": gen_text[:8000]})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        emit({"error": str(e)})
        sys.exit(1)
