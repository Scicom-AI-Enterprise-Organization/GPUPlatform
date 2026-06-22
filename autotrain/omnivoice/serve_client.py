#!/usr/bin/env python3
"""Drive the /v1/audio/speech server for a test list and save {id}.wav.

Used to verify the optimized (dynamic-batched + bucketed) serving path produces
the SAME quality as offline inference: synthesize the held-out test texts through
the server under concurrency, then score CER with tts_eval.py and compare.
"""
import argparse
import asyncio
import json
import os

import httpx


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--voices", default="voices.json")
    ap.add_argument("--test_list", default="data/eval_test.jsonl")
    ap.add_argument("--out_dir", default="served_wavs")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--format", default="wav")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    voices = json.load(open(a.voices, encoding="utf-8"))
    name_for = {}
    for n, v in voices.items():
        name_for.setdefault((v.get("language") or "").lower(), n)
    items = read_jsonl(a.test_list)
    sem = asyncio.Semaphore(a.concurrency)
    ok = {"n": 0, "err": 0}
    limits = httpx.Limits(max_connections=a.concurrency + 8)

    async with httpx.AsyncClient(limits=limits, timeout=120) as cl:
        async def one(it):
            voice = name_for.get((it.get("language_id") or "").lower()) or next(iter(voices))
            async with sem:
                try:
                    r = await cl.post(f"{a.url}/v1/audio/speech",
                                      json={"input": it["text"], "voice": voice,
                                            "response_format": a.format})
                    r.raise_for_status()
                    open(os.path.join(a.out_dir, f"{it['id']}.{a.format}"), "wb").write(r.content)
                    ok["n"] += 1
                except Exception as e:  # noqa: BLE001
                    ok["err"] += 1
                    if ok["err"] <= 3:
                        print(f"  err {it['id']}: {e}")
        await asyncio.gather(*[one(it) for it in items])
    print(f"[serve_client] saved {ok['n']} wavs ({ok['err']} errors) -> {a.out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
