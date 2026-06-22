#!/usr/bin/env python3
"""OpenAI-compatible, high-concurrency TTS server for OmniVoice.

Endpoint: POST /v1/audio/speech   (OpenAI `audio.speech` schema)
  body: {"model","input","voice","response_format","speed"}
  -> binary audio (mp3/wav/flac/opus/pcm)

Concurrency model (see README):
  * ONE resident model on the GPU (loaded at startup, bf16/fp16).
  * Named "voices" -> reference clips encoded ONCE into VoiceClonePrompt and cached,
    so each request is text -> diffusion -> codec-decode only.
  * A single async **dynamic-batching** loop: concurrent requests land on an
    asyncio.Queue; the loop coalesces up to MAX_BATCH of them (lingering at most
    MAX_WAIT_MS when under-filled) and runs ONE batched `model.generate`. Because
    every item runs the same fixed `num_step`, a batch finishes in lockstep — so
    throughput scales ~linearly with batch size until the GPU saturates.
  * The blocking GPU call runs in a dedicated 1-thread executor (torch releases
    the GIL on CUDA ops); audio container-encoding (mp3/opus) runs on a separate
    CPU threadpool so it never stalls the GPU or the event loop.

Env knobs: OV_MODEL, OV_VOICES (json), OV_NUM_STEP, OV_MAX_BATCH, OV_MAX_WAIT_MS,
OV_GUIDANCE, OV_COMPILE(0/1), OV_DTYPE(float16/bfloat16), PORT.
"""
import asyncio
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

MODEL = os.environ.get("OV_MODEL", "Scicom-intl/omnivoice-tmvoice")
VOICES_JSON = os.environ.get("OV_VOICES", "voices.json")
NUM_STEP = int(os.environ.get("OV_NUM_STEP", "32"))
MAX_BATCH = int(os.environ.get("OV_MAX_BATCH", "32"))
MAX_WAIT_MS = float(os.environ.get("OV_MAX_WAIT_MS", "15"))
BUCKET_RATIO = float(os.environ.get("OV_BUCKET_RATIO", "1.5"))  # max/min length within a batch
GUIDANCE = float(os.environ.get("OV_GUIDANCE", "2.0"))
COMPILE = os.environ.get("OV_COMPILE", "0") == "1"
DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}[os.environ.get("OV_DTYPE", "float16")]
PORT = int(os.environ.get("PORT", "8000"))

app = FastAPI(title="OmniVoice OpenAI-compatible TTS")
STATE = {}  # model, voices, sr, queue, gpu_pool, cpu_pool, stats


class SpeechReq(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    response_format: str | None = "mp3"
    speed: float | None = 1.0


def _load_voices(model):
    """voices.json: {name: {ref_audio, ref_text, language}} -> cached VoiceClonePrompt."""
    cfg = json.load(open(VOICES_JSON, encoding="utf-8"))
    voices = {}
    for name, v in cfg.items():
        prompt = model.create_voice_clone_prompt(
            ref_audio=v["ref_audio"], ref_text=v.get("ref_text"), preprocess_prompt=False)
        voices[name] = {"prompt": prompt, "language": v.get("language")}
        print(f"[serve] voice '{name}' cached (lang={v.get('language')})", flush=True)
    return voices


@app.on_event("startup")
async def _startup():
    from omnivoice import OmniVoice
    print(f"[serve] loading {MODEL} (dtype={DTYPE}) ...", flush=True)
    model = OmniVoice.from_pretrained(MODEL, device_map="cuda:0", dtype=DTYPE)
    model.eval()
    if COMPILE:
        try:
            model.llm = torch.compile(model.llm, mode="max-autotune", dynamic=True)
            print("[serve] torch.compile(llm) enabled", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[serve] compile failed ({e}); continuing eager", flush=True)
    STATE.update(
        model=model,
        sr=int(model.sampling_rate),
        voices=_load_voices(model),
        queue=asyncio.Queue(),
        gpu_pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu"),
        cpu_pool=ThreadPoolExecutor(max_workers=max(4, (os.cpu_count() or 8) // 2)),
        stats={"requests": 0, "batches": 0, "batched_items": 0, "audio_s": 0.0},
    )
    # warm up the batch path (compile/CUDA-graph capture happens here)
    default_voice = next(iter(STATE["voices"]))
    await _run_batch([_FakeReq("warm up warm up warm up.", STATE["voices"][default_voice], 1.0)])
    asyncio.create_task(_batch_loop())
    print(f"[serve] ready on :{PORT}  voices={list(STATE['voices'])}  "
          f"MAX_BATCH={MAX_BATCH} NUM_STEP={NUM_STEP}", flush=True)


class _FakeReq:
    def __init__(self, text, voice, speed):
        self.text, self.voice, self.speed = text, voice, speed
        self.est = len(text)  # length proxy for duration (bucketing)
        self.future = asyncio.get_event_loop().create_future()


def _gpu_generate(texts, prompts, speeds):
    with torch.inference_mode():
        return STATE["model"].generate(
            text=texts, voice_clone_prompt=prompts, speed=speeds,
            num_step=NUM_STEP, guidance_scale=GUIDANCE,
            preprocess_prompt=False, postprocess_output=True,
        )


async def _run_batch(batch):
    loop = asyncio.get_event_loop()
    texts = [b.text for b in batch]
    prompts = [b.voice["prompt"] for b in batch]
    speeds = [b.speed for b in batch]
    try:
        audios = await loop.run_in_executor(STATE["gpu_pool"], _gpu_generate, texts, prompts, speeds)
        for b, wav in zip(batch, audios):
            if not b.future.done():
                b.future.set_result(np.asarray(wav, dtype=np.float32))
        st = STATE["stats"]; st["batches"] += 1; st["batched_items"] += len(batch)
        st["audio_s"] += sum(len(a) for a in audios) / STATE["sr"]
    except Exception as e:  # noqa: BLE001
        for b in batch:
            if not b.future.done():
                b.future.set_exception(e)


async def _batch_loop():
    """Dynamic batching with LENGTH BUCKETING to cut padding waste.

    A batched `generate` pads every item to the longest sequence in the batch, so
    mixing a 3s and an 18s utterance wastes ~6x compute on the short one. Instead we
    drain a candidate pool, sort by length proxy, and emit a batch of *similar-length*
    items (max/min within BUCKET_RATIO); longer items are re-queued for their own
    batch. Padding per batch is then bounded to ~BUCKET_RATIO instead of unbounded.
    """
    q = STATE["queue"]
    while True:
        pool = [await q.get()]                       # block until a request arrives
        deadline = time.monotonic() + MAX_WAIT_MS / 1000.0
        while len(pool) < MAX_BATCH * 4:             # drain a pool to bucket from
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                pool.append(await asyncio.wait_for(q.get(), timeout=timeout))
            except asyncio.TimeoutError:
                break
        pool.sort(key=lambda r: r.est)
        base = pool[0].est
        batch, rest = [], []
        for r in pool:
            if len(batch) < MAX_BATCH and r.est <= base * BUCKET_RATIO + 20:
                batch.append(r)
            else:
                rest.append(r)
        for r in rest:                               # longer items -> their own batch
            q.put_nowait(r)
        await _run_batch(batch)


def _encode(wav: np.ndarray, fmt: str, sr: int) -> tuple[bytes, str]:
    fmt = (fmt or "mp3").lower()
    if fmt == "pcm":
        return (wav * 32767).astype("<i2").tobytes(), "audio/pcm"
    if fmt in ("wav", "flac"):
        buf = io.BytesIO(); sf.write(buf, wav, sr, format=fmt.upper()); return buf.getvalue(), f"audio/{fmt}"
    # mp3/opus/aac via pydub+ffmpeg
    from pydub import AudioSegment
    seg = AudioSegment((wav * 32767).astype("<i2").tobytes(), frame_rate=sr, sample_width=2, channels=1)
    buf = io.BytesIO()
    ct = {"mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac"}.get(fmt, "audio/mpeg")
    seg.export(buf, format="opus" if fmt == "opus" else fmt)
    return buf.getvalue(), ct


@app.get("/health")
async def health():
    return {"status": "ok", **STATE.get("stats", {})}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "voices": list(STATE["voices"])}]}


@app.post("/v1/audio/speech")
async def speech(req: SpeechReq):
    if not req.input or not req.input.strip():
        raise HTTPException(400, "input is empty")
    voices = STATE["voices"]
    name = req.voice if req.voice in voices else next(iter(voices))
    r = _FakeReq(req.input.strip(), voices[name], float(req.speed or 1.0))
    STATE["stats"]["requests"] += 1
    await STATE["queue"].put(r)
    wav = await r.future
    data, ct = await asyncio.get_event_loop().run_in_executor(
        STATE["cpu_pool"], _encode, wav, req.response_format or "mp3", STATE["sr"])
    return Response(content=data, media_type=ct)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
