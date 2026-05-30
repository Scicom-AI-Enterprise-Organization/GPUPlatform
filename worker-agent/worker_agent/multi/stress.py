"""Worker-side load test: fire N chat completions at a member's LOCAL vLLM
(http://127.0.0.1:<port>) at a target concurrency, measuring throughput + TTFT /
E2E latency. Runs ON the VM with a direct localhost connection, so the numbers
reflect the model + GPU — not the browser → proxy → reverse-tunnel → Redis path,
which serializes requests and makes the UI stress tab ~8x slower.
"""
from __future__ import annotations

import asyncio
import json
import time


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(p / 100 * len(s))) - 1))
    return s[i]


async def run_load_test(
    client,
    base_url: str,
    served_name: str,
    *,
    input_len: int = 128,
    output_len: int = 128,
    num_prompts: int = 50,
    concurrency: int = 10,
) -> dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    prompt = ("word " * max(1, input_len)).strip()
    num_prompts = max(1, num_prompts)
    concurrency = max(1, min(concurrency, num_prompts))
    results: list[tuple] = []  # (ok, ttft_ms, e2e_ms, out_tok, prompt_tok, err)
    launched = 0

    async def one() -> tuple:
        body = {
            "model": served_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        t0 = time.perf_counter()
        tfirst = None
        out = 0
        ptok = 0
        try:
            async with client.stream("POST", url, json=body, timeout=None) as r:
                if r.status_code != 200:
                    txt = (await r.aread()).decode("utf-8", "replace")[:200]
                    return (False, 0.0, 0.0, 0, 0, f"HTTP {r.status_code}: {txt}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    d = line[5:].strip()
                    if not d or d == "[DONE]":
                        continue
                    ch = json.loads(d)
                    u = ch.get("usage")
                    if u:
                        out = u.get("completion_tokens", out)
                        ptok = u.get("prompt_tokens", ptok)
                    dl = ((ch.get("choices") or [{}])[0].get("delta") or {})
                    if dl.get("content") and tfirst is None:
                        tfirst = time.perf_counter()
            t1 = time.perf_counter()
            return (True, ((tfirst or t1) - t0) * 1000, (t1 - t0) * 1000, out, ptok, None)
        except Exception as e:  # noqa: BLE001
            return (False, 0.0, 0.0, 0, 0, str(e)[:200])

    async def worker() -> None:
        nonlocal launched
        while launched < num_prompts:
            launched += 1
            results.append(await one())

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    dur = time.perf_counter() - t0

    ok = [r for r in results if r[0]]
    out_tot = sum(r[3] for r in ok)
    tot = sum(r[3] + r[4] for r in ok)
    ttfts = [r[1] for r in ok]
    e2es = [r[2] for r in ok]
    return {
        "model": served_name,
        "input_len": input_len,
        "output_len": output_len,
        "num_prompts": num_prompts,
        "concurrency": concurrency,
        "successful": len(ok),
        "failed": len(results) - len(ok),
        "duration_s": round(dur, 2),
        "request_throughput_per_s": round(len(ok) / dur, 3) if dur > 0 else 0.0,
        "output_throughput_tok_s": round(out_tot / dur, 1) if dur > 0 else 0.0,
        "total_throughput_tok_s": round(tot / dur, 1) if dur > 0 else 0.0,
        "ttft_ms": {"median": round(_pct(ttfts, 50)), "p99": round(_pct(ttfts, 99))},
        "e2e_ms": {"median": round(_pct(e2es, 50)), "p99": round(_pct(e2es, 99))},
        "errors": [r[5] for r in results if not r[0]][:3],
    }
