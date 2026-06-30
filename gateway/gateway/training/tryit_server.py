#!/usr/bin/env python3
"""Persistent "Try it" inference server for the Autotrain playground — the small
sibling of the serverless worker. Loads the finetuned model ONCE and serves many
requests over a Unix socket, so the slow model load (often 10-15s) is paid once
instead of per request. Handles BOTH ASR (transcribe) and TTS (synthesize) by
config `kind`. Shipped to the run's VM by the gateway and launched under nohup +
its own session; the gateway tracks its pid and kills the group to stop it.

Protocol (Unix socket, one JSON request per connection, newline-terminated):
  → {text, speaker?, max_new_tokens?, ...}     (tts)   or  {audio_path, ...} (asr)
  ← {wav_b64, sample_rate, device, n_codes, prompt, gen_text}  or  {text, device}
  ← {error: "..."} on failure
Writes `<sock>.ready` (JSON {device, kind}) once the model is loaded. Logs to
stdout (the gateway tails the nohup log). SIGTERM → unlink sock + exit.

Config (JSON via --config): {kind:"asr"|"tts", model_s3, region, endpoint,
access_key, secret_key, model_dir, sock, gpu, language?, task?}.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time

_SPEECH_TOK = re.compile(r"<\|s_(\d+)\|>")


def log(m: str) -> None:
    print(f"[server] {m}", flush=True)


# --- granular load progress: each step() closes out the previous step with its
# elapsed time, so the UI shows "→ loading X …" then "✓ loading X (1.2s)". ---
_step = {"t": 0.0, "label": ""}


def step(m: str) -> None:
    if _step["label"]:
        log(f"  ✓ {_step['label']} ({time.time() - _step['t']:.1f}s)")
    _step["t"] = time.time(); _step["label"] = m
    log(f"→ {m} …")


def step_done() -> None:
    if _step["label"]:
        log(f"  ✓ {_step['label']} ({time.time() - _step['t']:.1f}s)")
        _step["label"] = ""


def _pick_gpu() -> str | None:
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
            f = int(free)
        except ValueError:
            continue
        if f > best_free:
            best, best_free = idx, f
    return best if (best is not None and best_free > 5000) else None


def _download_model(cfg: dict) -> str:
    import boto3
    from botocore.client import Config as BotoConfig

    s3 = cfg["model_s3"]
    bucket, _, prefix = s3[len("s3://"):].partition("/")
    prefix = prefix.rstrip("/") + "/"
    cli = boto3.client(
        "s3", region_name=cfg.get("region") or "us-east-1", endpoint_url=cfg.get("endpoint") or None,
        aws_access_key_id=cfg.get("access_key") or None, aws_secret_access_key=cfg.get("secret_key") or None,
        config=BotoConfig(signature_version="s3v4"),
    )
    dest = cfg["model_dir"]
    os.makedirs(dest, exist_ok=True)

    # Enumerate first so we can report overall + per-file progress (and skip
    # already-cached files) — the big safetensors shards are GBs, so the bare
    # "fetching …" step used to sit silent for minutes.
    objs: list[tuple[str, str, int]] = []
    for page in cli.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if rel:
                objs.append((obj["Key"], rel, int(obj["Size"])))
    if not objs:
        raise RuntimeError(f"no model files under {s3}")
    total = sum(s for _, _, s in objs)
    to_fetch = [
        (k, r, s) for (k, r, s) in objs
        if not (os.path.exists(os.path.join(dest, r)) and os.path.getsize(os.path.join(dest, r)) == s)
    ]
    fetch_bytes = sum(s for _, _, s in to_fetch)
    log(f"  · {len(objs)} files · {total / 1e6:.0f} MB total — {len(to_fetch)} to download "
        f"({fetch_bytes / 1e6:.0f} MB), {len(objs) - len(to_fetch)} already cached")

    done = 0
    for i, (key, rel, size) in enumerate(to_fetch, 1):
        fp = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(fp) or dest, exist_ok=True)
        log(f"  ↓ [{i}/{len(to_fetch)}] {rel} ({size / 1e6:.0f} MB)")
        if size > 200 * 1e6:  # big shard → stream in-file % so it's not a silent multi-minute wait
            prog = {"b": 0, "next": 0.2}
            lock = threading.Lock()

            def _cb(chunk: int, _rel=rel, _size=size, _p=prog, _lk=lock) -> None:
                with _lk:
                    _p["b"] += chunk
                    frac = _p["b"] / max(_size, 1)
                    if frac >= _p["next"] and frac < 1.0:
                        log(f"      {_rel}: {frac * 100:.0f}%  ({_p['b'] / 1e6:.0f}/{_size / 1e6:.0f} MB)")
                        _p["next"] += 0.2
            cli.download_file(bucket, key, fp, Callback=_cb)
        else:
            cli.download_file(bucket, key, fp)
        done += size
        log(f"      ✓ {rel}  [{done / 1e6:.0f}/{fetch_bytes / 1e6:.0f} MB · "
            f"{done / max(fetch_bytes, 1) * 100:.0f}% of download]")
    log(f"model: {len(objs)} files · {total / 1e6:.0f} MB · {len(to_fetch)} fetched / {len(objs) - len(to_fetch)} cached")
    return dest


class TTSEngine:
    def __init__(self, cfg: dict, use_cuda: bool):
        step("importing torch + transformers + neucodec")
        import soundfile
        import torch
        from neucodec import NeuCodec
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch; self.sf = soundfile
        self.use_cuda = use_cuda; self.device = "cuda" if use_cuda else "cpu"
        dtype = torch.bfloat16 if use_cuda else torch.float32

        step("fetching model files from storage")
        md = _download_model(cfg)

        step("loading tokenizer")
        self.tok = AutoTokenizer.from_pretrained(md)

        step(f"loading language-model weights ({str(dtype).replace('torch.', '')})")
        m = AutoModelForCausalLM.from_pretrained(md, torch_dtype=dtype)
        try:
            nparams = sum(p.numel() for p in m.parameters())
            log(f"  · language model: {nparams / 1e9:.2f}B params")
        except Exception:
            pass

        step(f"moving language model to {self.device}")
        self.model = (m.cuda() if use_cuda else m).eval()

        step("loading NeuCodec audio decoder (neuphonic/neucodec)")
        # Upstream neucodec (neuphonic/neucodec, 24 kHz).
        neu = NeuCodec.from_pretrained("neuphonic/neucodec").eval()
        step(f"moving NeuCodec to {self.device}")
        self.neu = neu.cuda() if use_cuda else neu
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        if use_cuda:
            try:
                gb = torch.cuda.memory_allocated() / 1024 ** 3
                log(f"  · resident on {torch.cuda.get_device_name(0)} — {gb:.1f} GiB GPU memory in use")
            except Exception:
                pass

    def infer(self, req: dict) -> dict:
        torch = self.torch
        text = (req.get("text") or "").strip()
        if not text:
            return {"error": "empty text"}
        speaker = (req.get("speaker") or "").strip()
        left = f"{speaker}: {text}" if speaker else text
        prompt = f"<|im_start|>{left}<|speech_start|>"  # mirrors pack_stage1.py
        ids = self.tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        if self.use_cuda:
            ids = ids.cuda()
        t = time.time()
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=int(req.get("max_new_tokens", 1024)), do_sample=True,
                temperature=float(req.get("temperature", 0.8)), top_p=float(req.get("top_p", 0.95)),
                eos_token_id=self.im_end,
            )
        gen = out[0, ids.shape[-1]:].tolist()
        gen_text = self.tok.decode(gen, skip_special_tokens=False)
        codes = [int(n) for n in _SPEECH_TOK.findall(gen_text)]
        log(f"synth: {len(codes)} speech codes in {time.time() - t:.1f}s")
        if not codes:
            return {"error": "model produced no speech tokens", "prompt": prompt, "gen_text": gen_text[:4000]}
        with torch.no_grad():
            fsq = torch.tensor(codes, dtype=torch.long).reshape(1, 1, -1)
            if self.use_cuda:
                fsq = fsq.cuda()
            wav = self.neu.decode_code(fsq).squeeze().detach().cpu().float().numpy()
        sr = int(getattr(self.neu, "sample_rate", 24000))
        buf = io.BytesIO()
        self.sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        return {"wav_b64": base64.b64encode(buf.getvalue()).decode(), "sample_rate": sr,
                "device": self.device, "n_codes": len(codes), "prompt": prompt, "gen_text": gen_text[:8000]}


class DecodeEngine:
    """Persistent NeuCodec decoder for inspecting a `tts_packed` dataset: given the
    speech codes of one packed utterance, decode them straight back to audio — no
    language model, just the codec. Loads only NeuCodec (seconds, not the multi-GB
    LM) and stays resident, so each "play utt N" click is instant until idle-unload."""

    def __init__(self, cfg: dict, use_cuda: bool):
        step("importing torch + neucodec")
        import soundfile
        import torch
        from neucodec import NeuCodec

        self.torch = torch
        self.sf = soundfile
        self.use_cuda = use_cuda
        self.device = "cuda" if use_cuda else "cpu"

        step("loading NeuCodec audio decoder (neuphonic/neucodec)")
        # Upstream neucodec (neuphonic/neucodec, 24 kHz).
        neu = NeuCodec.from_pretrained("neuphonic/neucodec").eval()
        step(f"moving NeuCodec to {self.device}")
        self.neu = neu.cuda() if use_cuda else neu
        if use_cuda:
            try:
                gb = torch.cuda.memory_allocated() / 1024 ** 3
                log(f"  · resident on {torch.cuda.get_device_name(0)} — {gb:.1f} GiB GPU memory in use")
            except Exception:  # noqa: BLE001
                pass

    def infer(self, req: dict) -> dict:
        torch = self.torch
        codes = req.get("codes")
        if not codes:  # also accept the raw decoded text and pull <|s_N|> out of it
            codes = [int(n) for n in _SPEECH_TOK.findall(req.get("gen_text") or req.get("text") or "")]
        try:
            codes = [int(c) for c in (codes or [])]
        except (TypeError, ValueError):
            return {"error": "codes must be a list of integers"}
        if not codes:
            return {"error": "no speech codes to decode for this utterance"}
        t = time.time()
        with torch.no_grad():
            fsq = torch.tensor(codes, dtype=torch.long).reshape(1, 1, -1)
            if self.use_cuda:
                fsq = fsq.cuda()
            wav = self.neu.decode_code(fsq).squeeze().detach().cpu().float().numpy()
        sr = int(getattr(self.neu, "sample_rate", 24000))
        buf = io.BytesIO()
        self.sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        log(f"decoded {len(codes)} codes → {len(wav) / sr:.2f}s @ {sr}Hz in {time.time() - t:.2f}s")
        return {"wav_b64": base64.b64encode(buf.getvalue()).decode(), "sample_rate": sr,
                "device": self.device, "n_codes": len(codes)}


class ASREngine:
    def __init__(self, cfg: dict, use_cuda: bool):
        step("importing torch + transformers + librosa")
        import librosa
        import torch
        from transformers import pipeline

        self.librosa = librosa
        self.use_cuda = use_cuda; self.device = "cuda" if use_cuda else "cpu"
        dtype = torch.float16 if use_cuda else torch.float32

        step("fetching model files from storage")
        md = _download_model(cfg)

        step(f"loading whisper weights ({str(dtype).replace('torch.', '')}) + building ASR pipeline on {self.device}")
        self.asr = pipeline("automatic-speech-recognition", model=md, device=0 if use_cuda else -1,
                            torch_dtype=dtype, chunk_length_s=30)
        self.gen_kwargs = {"task": cfg.get("task") or "transcribe"}
        if cfg.get("language"):
            self.gen_kwargs["language"] = cfg["language"]

    def infer(self, req: dict) -> dict:
        ap = req.get("audio_path")
        if not ap or not os.path.exists(ap):
            return {"error": "audio file not found on the VM"}
        audio, sr = self.librosa.load(ap, sr=16000, mono=True)
        t = time.time()
        # return_timestamps=False on purpose: with transformers 5.x the timestamp
        # logits processor slices with the model's eos_token_id, which isn't a plain
        # int on a merged-LoRA Whisper → "slice indices must be integers". We only
        # need text here; chunk_length_s still handles long clips via token-overlap.
        out = self.asr({"raw": audio, "sampling_rate": sr}, generate_kwargs=self.gen_kwargs, return_timestamps=False)
        text = (out["text"] if isinstance(out, dict) else str(out)) or ""
        log(f"asr: {len(audio) / sr:.1f}s in {time.time() - t:.1f}s → {len(text.split())} words")
        return {"text": text.strip(), "device": self.device}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))

    sel = str(cfg.get("gpu") or "auto").strip().lower()
    if sel == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""; want = False
    elif sel.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = sel; want = True
    else:
        g = _pick_gpu(); os.environ["CUDA_VISIBLE_DEVICES"] = g if g is not None else ""; want = g is not None

    import torch
    use_cuda = want and torch.cuda.is_available()
    kind = cfg["kind"]
    log(f"loading {kind} model on {'cuda' if use_cuda else 'cpu'} …")
    t0 = time.time()
    if kind == "tts":
        engine = TTSEngine(cfg, use_cuda)
    elif kind == "tts_decode":
        engine = DecodeEngine(cfg, use_cuda)
    else:
        engine = ASREngine(cfg, use_cuda)
    step_done()
    log(f"loaded {kind} model in {time.time() - t0:.1f}s — ready on {engine.device}")

    sock = cfg["sock"]
    for p in (sock, sock + ".ready"):
        try:
            os.unlink(p)
        except OSError:
            pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock)
    srv.listen(8)

    cleanup_files = [sock, sock + ".ready"] + ([cfg["pid"]] if cfg.get("pid") else [])

    def _cleanup():
        for p in cleanup_files:
            try:
                os.unlink(p)
            except OSError:
                pass

    def _bye(*_):
        _cleanup()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    with open(sock + ".ready", "w") as f:
        f.write(json.dumps({"device": engine.device, "kind": kind, "pid": os.getpid()}))
    log("listening")

    # Idle auto-unload — free the GPU if no request arrives for idle_timeout seconds,
    # so a forgotten worker doesn't pin a GPU. 0 disables. Runs in a daemon thread;
    # os._exit terminates the whole process (the main thread is blocked on accept()).
    state = {"last": time.time()}
    idle_timeout = float(cfg.get("idle_timeout") or 0)
    if idle_timeout > 0:
        def _watchdog():
            while True:
                time.sleep(min(30.0, idle_timeout))
                if time.time() - state["last"] > idle_timeout:
                    log(f"idle {idle_timeout:.0f}s with no requests — auto-unloading + freeing the GPU")
                    _cleanup()
                    os._exit(0)
        threading.Thread(target=_watchdog, daemon=True).start()
        log(f"idle auto-unload after {idle_timeout:.0f}s of no requests")

    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            break
        state["last"] = time.time()
        try:
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            req = json.loads(data.decode() or "{}")
            try:
                resp = engine.infer(req)
            except Exception as e:  # noqa: BLE001
                resp = {"error": str(e)}
            conn.sendall((json.dumps(resp) + "\n").encode())
        except Exception as e:  # noqa: BLE001
            try:
                conn.sendall((json.dumps({"error": str(e)}) + "\n").encode())
            except Exception:  # noqa: BLE001
                pass
        finally:
            conn.close()
            state["last"] = time.time()  # don't count a long generation as idle
    return 0


if __name__ == "__main__":
    sys.exit(main())
