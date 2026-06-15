#!/usr/bin/env python3
"""Stress-test token streaming against a serverless-GPU endpoint.

Fires N concurrent streaming chat-completions and reports: success / 429-capacity /
error breakdown (incl. httpx 'incomplete chunked read' = RemoteProtocolError), TTFT,
tokens, wall-clock, throughput. Defaults target the LOCAL gateway's endpoint-rt (GLM).
"""
import argparse
import asyncio
import json
import time
from collections import Counter

import httpx

AP = argparse.ArgumentParser()
AP.add_argument("--base", default="http://localhost:8080")
AP.add_argument("--app", default="endpoint-rt")
AP.add_argument("--model", default="zai-org/GLM-5.1-FP8")
AP.add_argument("--key", default="sgpu_8SoSHhJzCL9FGjwnwtib5BFo_4EswqXo3z3qkLodfis")
AP.add_argument("--n", type=int, default=200)
AP.add_argument("--max-tokens", type=int, default=80)
AP.add_argument("--req-timeout", type=float, default=180.0)
A = AP.parse_args()

URL = f"{A.base}/{A.app}/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {A.key}", "Content-Type": "application/json"}


async def one(client: httpx.AsyncClient, i: int) -> dict:
    body = {
        "model": A.model,
        "messages": [{"role": "user", "content": f"Write a few sentences about the number {i} and prime factorization."}],
        "stream": True,
        "max_tokens": A.max_tokens,
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    try:
        async with client.stream("POST", URL, json=body, headers=HEADERS,
                                 timeout=httpx.Timeout(connect=10, read=A.req_timeout, write=10, pool=30)) as r:
            if r.status_code != 200:
                txt = (await r.aread()).decode("utf-8", "replace")[:200]
                return {"ok": False, "kind": f"http_{r.status_code}", "detail": txt, "t": time.perf_counter() - t0}
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("done") or obj.get("error"):
                    if obj.get("error"):
                        return {"ok": False, "kind": "stream_error", "detail": str(obj.get("error"))[:160], "t": time.perf_counter() - t0}
                    break
                chunks += 1
        return {"ok": True, "kind": "ok", "ttft": ttft, "chunks": chunks, "t": time.perf_counter() - t0}
    except httpx.RemoteProtocolError as e:
        return {"ok": False, "kind": "incomplete_chunked_read", "detail": str(e)[:160], "t": time.perf_counter() - t0}
    except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
        return {"ok": False, "kind": "timeout", "detail": str(e)[:120], "t": time.perf_counter() - t0}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "kind": f"exc_{type(e).__name__}", "detail": str(e)[:160], "t": time.perf_counter() - t0}


async def main():
    limits = httpx.Limits(max_connections=A.n + 20, max_keepalive_connections=A.n + 20)
    print(f"firing {A.n} concurrent streams → {URL}  (model={A.model}, max_tokens={A.max_tokens})", flush=True)
    wall0 = time.perf_counter()
    async with httpx.AsyncClient(limits=limits) as client:
        res = await asyncio.gather(*[one(client, i) for i in range(A.n)])
    wall = time.perf_counter() - wall0

    kinds = Counter(r["kind"] for r in res)
    ok = [r for r in res if r["ok"]]
    ttfts = sorted(r["ttft"] for r in ok if r.get("ttft") is not None)
    lats = sorted(r["t"] for r in ok)
    toks = sum(r.get("chunks", 0) for r in ok)

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else float("nan")

    print(f"\n=== results ({A.n} reqs in {wall:.1f}s wall) ===")
    for k, v in kinds.most_common():
        print(f"  {k:28} {v}")
    print(f"\nsuccess: {len(ok)}/{A.n}")
    if ttfts:
        print(f"TTFT  s  p50={pct(ttfts,.5):.2f} p90={pct(ttfts,.9):.2f} p99={pct(ttfts,.99):.2f} max={ttfts[-1]:.2f}")
    if lats:
        print(f"total s  p50={pct(lats,.5):.2f} p90={pct(lats,.9):.2f} p99={pct(lats,.99):.2f} max={lats[-1]:.2f}")
    print(f"stream chunks total={toks}  (~{toks/wall:.0f} chunks/s aggregate)")
    # show a couple of distinct failure details
    seen = set()
    for r in res:
        if not r["ok"] and r["kind"] not in seen:
            seen.add(r["kind"])
            print(f"  e.g. {r['kind']}: {r.get('detail','')}")

asyncio.run(main())
