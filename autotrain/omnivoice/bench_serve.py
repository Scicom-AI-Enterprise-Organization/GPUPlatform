#!/usr/bin/env python3
"""Concurrency-sweep benchmark for the OmniVoice /v1/audio/speech server.

For each concurrency level C it fires a fixed number of requests with C in flight
and reports: throughput (req/s), aggregate RTF (audio-seconds generated per
wall-second = "x realtime"), and end-to-end latency p50/p95/p99. Sweeping C shows
the dynamic-batching speedup and the GPU saturation knee. Run ON the pod (localhost)
so the numbers reflect serving, not network.
"""
import argparse
import asyncio
import io
import json
import time

import httpx
import numpy as np
import soundfile as sf


def load_corpus(test_list, voices):
    name_for = {}
    for n, v in voices.items():
        lang = (v.get("language") or "").lower()
        name_for.setdefault(lang, n)
    items = []
    for line in open(test_list, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        v = name_for.get((d.get("language_id") or "").lower()) or next(iter(voices))
        if d.get("text", "").strip():
            items.append((d["text"].strip(), v))
    return items


async def one(client, url, text, voice, fmt):
    t0 = time.monotonic()
    r = await client.post(f"{url}/v1/audio/speech",
                          json={"input": text, "voice": voice, "response_format": fmt},
                          timeout=120)
    lat = time.monotonic() - t0
    r.raise_for_status()
    try:
        wav, sr = sf.read(io.BytesIO(r.content))
        dur = len(wav) / sr
    except Exception:  # noqa: BLE001
        dur = 0.0
    return lat, dur


async def run_level(url, corpus, c, n, fmt):
    sem = asyncio.Semaphore(c)
    lats, durs = [], []
    errors = 0
    limits = httpx.Limits(max_connections=c + 8, max_keepalive_connections=c + 8)

    async with httpx.AsyncClient(limits=limits) as client:
        async def worker(i):
            nonlocal errors
            text, voice = corpus[i % len(corpus)]
            async with sem:
                try:
                    lat, dur = await one(client, url, text, voice, fmt)
                    lats.append(lat); durs.append(dur)
                except Exception as e:  # noqa: BLE001
                    errors += 1
                    if errors <= 3:
                        print(f"   err: {e}")
        t0 = time.monotonic()
        await asyncio.gather(*[worker(i) for i in range(n)])
        wall = time.monotonic() - t0

    lats.sort()
    pct = lambda p: lats[min(len(lats) - 1, int(p * len(lats)))] if lats else float("nan")
    done = len(lats)
    return {
        "concurrency": c, "done": done, "errors": errors, "wall_s": round(wall, 2),
        "throughput_rps": round(done / wall, 2) if wall else 0,
        "audio_s": round(sum(durs), 1),
        "rtf_x_realtime": round(sum(durs) / wall, 1) if wall else 0,
        "lat_p50_s": round(pct(0.50), 3), "lat_p95_s": round(pct(0.95), 3),
        "lat_p99_s": round(pct(0.99), 3), "lat_max_s": round(lats[-1], 3) if lats else None,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--voices", default="voices.json")
    ap.add_argument("--test_list", default="data/eval_test.jsonl")
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    ap.add_argument("--requests_per_level", type=int, default=128)
    ap.add_argument("--format", default="wav")
    ap.add_argument("--out", default="bench_results.json")
    a = ap.parse_args()

    voices = json.load(open(a.voices, encoding="utf-8"))
    corpus = load_corpus(a.test_list, voices)
    levels = [int(x) for x in a.concurrency.split(",")]
    print(f"corpus={len(corpus)} texts | voices={list(voices)} | fmt={a.format} | "
          f"requests/level={a.requests_per_level}\n")

    # warmup
    await run_level(a.url, corpus, 4, 8, a.format)

    rows = []
    hdr = f"{'conc':>5} {'done':>5} {'err':>4} {'rps':>8} {'RTF(x)':>8} {'p50':>7} {'p95':>7} {'p99':>7}"
    print(hdr); print("-" * len(hdr))
    for c in levels:
        n = max(a.requests_per_level, c * 2)
        r = await run_level(a.url, corpus, c, n, a.format)
        rows.append(r)
        print(f"{r['concurrency']:>5} {r['done']:>5} {r['errors']:>4} {r['throughput_rps']:>8} "
              f"{r['rtf_x_realtime']:>8} {r['lat_p50_s']:>7} {r['lat_p95_s']:>7} {r['lat_p99_s']:>7}")
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
