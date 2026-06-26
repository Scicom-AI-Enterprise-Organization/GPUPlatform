"""In-gateway benchmark client for the *ingress* provider mode.

For RunPod/VM benchmarks the gateway shells out to `benchmaq … bench`, which
runs the real `vllm bench serve` load generator on the remote box. The ingress
mode has no remote box — the user already serves + ingressed their vLLM and just
wants it benchmarked. The bench client therefore has to run *inside the gateway
container*, where `vllm` is not installed (and we don't want it — it pulls torch
+ CUDA libs into a lightweight FastAPI image).

So this module is a small, dependency-light async load generator built on the
libs the gateway already ships (`httpx`, `transformers`/`tokenizers`,
`huggingface_hub`). It reads the SAME config shape benchmaq consumes
(`benchmark[].bench[]`), drives the OpenAI-compatible endpoint at `base_url`, and
writes per-config result JSON files + prints a `vllm bench serve`-style table so
the gateway's existing result-parsing (`bench.py`) and the web results view work
unchanged.

Run as a subprocess so its stdout streams to the UI via the gateway's log tee:

    python -m gateway.bench_ingress <config.yaml>

Metric definitions follow vLLM's: TTFT = time to first token, ITL = inter-token
latency, TPOT = (e2el - ttft) / (output_tokens - 1), E2EL = end-to-end latency.
Numbers are produced by THIS client, not vLLM's, so they're standard-methodology
but not bit-identical to a RunPod/VM run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

# Loaded HF tokenizers, keyed by model id. None marks a model whose tokenizer we
# failed to load (gated/offline) so we only warn once and fall back to text.
_TOKENIZERS: dict[str, Any] = {}


def _load_tokenizer(model: str) -> Optional[Any]:
    if model in _TOKENIZERS:
        return _TOKENIZERS[model]
    tok = None
    try:
        from transformers import AutoTokenizer  # transformers ships in the gateway image

        token = os.environ.get("HF_TOKEN") or None
        tok = AutoTokenizer.from_pretrained(model, token=token, trust_remote_code=False)
        print(f"[ingress] loaded tokenizer for {model} (vocab≈{tok.vocab_size})", flush=True)
    except Exception as e:  # gated without token, no network, custom code, …
        print(
            f"[ingress] WARNING: could not load tokenizer for {model} ({e}); "
            "falling back to approximate word prompts — input token counts are estimates",
            flush=True,
        )
        tok = None
    _TOKENIZERS[model] = tok
    return tok


# A pool of common English words for the tokenizer-less fallback. ~1.3 tokens/word
# is a rough average for BPE tokenizers, used only to size the fallback prompt.
_FALLBACK_WORDS = (
    "the of and to in a is that for it with as was on are by be this from at or an "
    "which have has had not but they you we can will one all would there their what"
).split()


def _make_prompt(tok: Optional[Any], input_len: int, rng: random.Random) -> tuple[str, int]:
    """Return (prompt_text, actual_input_tokens) of ~input_len tokens.

    With a tokenizer we sample random token ids (vLLM's `random` dataset
    approach) and decode them; without one we stitch random words sized by a
    rough tokens-per-word ratio. The second return value is the real tokenized
    length when a tokenizer is available, else the estimate.
    """
    if tok is not None:
        vocab = max(int(getattr(tok, "vocab_size", 32000)) - 1, 1)
        ids = [rng.randint(0, vocab) for _ in range(max(input_len, 1))]
        try:
            text = tok.decode(ids, skip_special_tokens=True)
            # Decoded text rarely re-tokenizes to exactly input_len; trim/measure.
            reenc = tok.encode(text, add_special_tokens=False)
            if len(reenc) > input_len:
                reenc = reenc[:input_len]
                text = tok.decode(reenc, skip_special_tokens=True)
            return text, len(reenc)
        except Exception:
            pass  # fall through to word fallback
    n_words = max(int(round(input_len / 1.3)), 1)
    text = " ".join(rng.choice(_FALLBACK_WORDS) for _ in range(n_words))
    return text, input_len


# ---------------------------------------------------------------------------
# A single request
# ---------------------------------------------------------------------------


class ReqResult:
    __slots__ = ("ok", "ttft", "itls", "e2el", "out_tokens", "in_tokens", "error")

    def __init__(self) -> None:
        self.ok = False
        self.ttft = 0.0
        self.itls: list[float] = []
        self.e2el = 0.0
        self.out_tokens = 0
        self.in_tokens = 0
        self.error: Optional[str] = None


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    endpoint: str,
    payload: dict,
    tok: Optional[Any],
    in_tokens: int,
) -> ReqResult:
    """Fire one streaming request and time the token stream."""
    r = ReqResult()
    r.in_tokens = in_tokens
    is_chat = endpoint.endswith("/chat/completions")
    start = time.perf_counter()
    last = start
    pieces: list[str] = []
    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:300]
                r.error = f"HTTP {resp.status_code}: {body}"
                return r
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except Exception:
                    continue
                choices = chunk.get("choices") or []
                piece = ""
                if choices:
                    if is_chat:
                        piece = (choices[0].get("delta") or {}).get("content") or ""
                    else:
                        piece = choices[0].get("text") or ""
                if piece:
                    now = time.perf_counter()
                    if not pieces:
                        r.ttft = now - start
                    else:
                        r.itls.append(now - last)
                    last = now
                    pieces.append(piece)
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens"):
                    r.out_tokens = int(usage["completion_tokens"])
        r.e2el = time.perf_counter() - start
        text = "".join(pieces)
        if not r.out_tokens:
            # No usage in stream — count tokens by re-tokenizing, else by pieces.
            if tok is not None and text:
                try:
                    r.out_tokens = len(tok.encode(text, add_special_tokens=False))
                except Exception:
                    r.out_tokens = len(pieces)
            else:
                r.out_tokens = len(pieces)
        r.ok = bool(pieces)
        if not r.ok:
            r.error = "empty response (no tokens streamed)"
    except Exception as e:
        r.error = f"{type(e).__name__}: {e}"
    return r


# ---------------------------------------------------------------------------
# One bench config (one row of the sweep)
# ---------------------------------------------------------------------------


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (q / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def _stat_block(prefix: str, values_ms: list[float]) -> dict:
    if not values_ms:
        return {f"mean_{prefix}_ms": 0.0, f"median_{prefix}_ms": 0.0,
                f"std_{prefix}_ms": 0.0, f"p90_{prefix}_ms": 0.0,
                f"p95_{prefix}_ms": 0.0, f"p99_{prefix}_ms": 0.0}
    return {
        f"mean_{prefix}_ms": statistics.fmean(values_ms),
        f"median_{prefix}_ms": statistics.median(values_ms),
        f"std_{prefix}_ms": statistics.pstdev(values_ms) if len(values_ms) > 1 else 0.0,
        f"p90_{prefix}_ms": _pct(values_ms, 90),
        f"p95_{prefix}_ms": _pct(values_ms, 95),
        f"p99_{prefix}_ms": _pct(values_ms, 99),
    }


async def _run_bench_config(
    base_url: str,
    model: str,
    tok: Optional[Any],
    bench_cfg: dict,
    rng: random.Random,
) -> dict:
    endpoint = str(bench_cfg.get("endpoint") or "/v1/completions")
    input_len = int(bench_cfg.get("random_input_len") or 1024)
    output_len = int(bench_cfg.get("random_output_len") or 128)
    num_prompts = int(bench_cfg.get("num_prompts") or 100)
    max_conc = int(bench_cfg.get("max_concurrency") or num_prompts)
    num_warmups = int(bench_cfg.get("num_warmups") or 0)
    ignore_eos = bool(bench_cfg.get("ignore_eos", True))
    rate = bench_cfg.get("request_rate", "inf")
    try:
        rate_val = float(rate)
    except (TypeError, ValueError):
        rate_val = float("inf")

    url = base_url.rstrip("/") + endpoint
    is_chat = endpoint.endswith("/chat/completions")

    def build_payload() -> tuple[dict, int]:
        prompt, in_tok = _make_prompt(tok, input_len, rng)
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": output_len,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            # vLLM extensions: force exact output length so every request does the
            # same generation work regardless of when the model would emit EOS.
            "min_tokens": output_len,
            "ignore_eos": ignore_eos,
        }
        if is_chat:
            body["messages"] = [{"role": "user", "content": prompt}]
        else:
            body["prompt"] = prompt
        return body, in_tok

    sem = asyncio.Semaphore(max_conc)
    timeout = httpx.Timeout(connect=30.0, read=3600.0, write=120.0, pool=None)
    limits = httpx.Limits(max_connections=max_conc + 8, max_keepalive_connections=max_conc + 8)
    headers = {}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"

    async with httpx.AsyncClient(timeout=timeout, limits=limits, headers=headers) as client:
        # Warmup — fire and discard so JIT/compile/cache effects don't skew the
        # measured window. Bounded by the same concurrency.
        if num_warmups:
            print(f"[ingress] warming up ({num_warmups} requests)…", flush=True)

            async def _warm() -> None:
                async with sem:
                    payload, in_tok = build_payload()
                    await _one_request(client, url, endpoint, payload, tok, in_tok)

            await asyncio.gather(*[_warm() for _ in range(num_warmups)], return_exceptions=True)

        async def _measured(i: int) -> ReqResult:
            async with sem:
                payload, in_tok = build_payload()
                return await _one_request(client, url, endpoint, payload, tok, in_tok)

        print(
            f"[ingress] running: in={input_len} out={output_len} "
            f"prompts={num_prompts} concurrency={max_conc} rate={rate}",
            flush=True,
        )
        bench_start = time.perf_counter()
        tasks: list[asyncio.Task] = []
        for i in range(num_prompts):
            # Poisson arrivals for a finite request rate; fire-immediately for inf.
            if math.isfinite(rate_val) and rate_val > 0 and i > 0:
                await asyncio.sleep(rng.expovariate(rate_val))
            tasks.append(asyncio.create_task(_measured(i)))
        results: list[ReqResult] = list(await asyncio.gather(*tasks))
        duration = time.perf_counter() - bench_start

    ok = [r for r in results if r.ok]
    completed = len(ok)
    errors = [r.error for r in results if not r.ok and r.error]
    total_in = sum(r.in_tokens for r in ok)
    total_out = sum(r.out_tokens for r in ok)
    ttfts = [r.ttft * 1000 for r in ok]
    e2els = [r.e2el * 1000 for r in ok]
    itls_flat = [v * 1000 for r in ok for v in r.itls]
    tpots = [
        ((r.e2el - r.ttft) / (r.out_tokens - 1)) * 1000
        for r in ok
        if r.out_tokens > 1
    ]

    result = {
        "backend": "ingress-httpx",
        "model_id": model,
        "endpoint": endpoint,
        "num_prompts": num_prompts,
        "max_concurrency": max_conc,
        "request_rate": rate if isinstance(rate, str) else rate_val,
        "duration": duration,
        "completed": completed,
        "failed": len(results) - completed,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "request_throughput": (completed / duration) if duration else 0.0,
        "output_throughput": (total_out / duration) if duration else 0.0,
        "total_token_throughput": ((total_in + total_out) / duration) if duration else 0.0,
        **_stat_block("ttft", ttfts),
        **_stat_block("tpot", tpots),
        **_stat_block("itl", itls_flat),
        **_stat_block("e2el", e2els),
    }
    # Heavy per-request arrays — only when save_detailed; the gateway strips these
    # before the DB anyway, but they bloat the streamed/stored file otherwise.
    if bench_cfg.get("_save_detailed"):
        result["ttfts"] = ttfts
        result["itls"] = itls_flat
        result["e2els"] = e2els
        result["tpots"] = tpots
        result["errors"] = errors[:50]

    _print_table(result)
    if errors:
        # Surface a few distinct error reasons so a 0-success run is debuggable.
        seen: list[str] = []
        for e in errors:
            if e not in seen:
                seen.append(e)
            if len(seen) >= 3:
                break
        for e in seen:
            print(f"[ingress] error sample: {e}", flush=True)
    return result


def _print_table(r: dict) -> None:
    """Mimic vLLM bench serve's summary table — incl. the `Successful requests:`
    line the gateway's all-failed detector greps for."""
    def line(label: str, val: Any) -> str:
        return f"{label:<42}{val}"

    print("=" * 50, flush=True)
    print("============ Serving Benchmark Result ============", flush=True)
    print(line("Successful requests:", r["completed"]), flush=True)
    print(line("Failed requests:", r["failed"]), flush=True)
    print(line("Benchmark duration (s):", f"{r['duration']:.2f}"), flush=True)
    print(line("Total input tokens:", r["total_input_tokens"]), flush=True)
    print(line("Total generated tokens:", r["total_output_tokens"]), flush=True)
    print(line("Request throughput (req/s):", f"{r['request_throughput']:.2f}"), flush=True)
    print(line("Output token throughput (tok/s):", f"{r['output_throughput']:.2f}"), flush=True)
    print(line("Total Token throughput (tok/s):", f"{r['total_token_throughput']:.2f}"), flush=True)
    print("---------------Time to First Token----------------", flush=True)
    print(line("Mean TTFT (ms):", f"{r['mean_ttft_ms']:.2f}"), flush=True)
    print(line("Median TTFT (ms):", f"{r['median_ttft_ms']:.2f}"), flush=True)
    print(line("P99 TTFT (ms):", f"{r['p99_ttft_ms']:.2f}"), flush=True)
    print("-----Time per Output Token (excl. 1st token)------", flush=True)
    print(line("Mean TPOT (ms):", f"{r['mean_tpot_ms']:.2f}"), flush=True)
    print(line("Median TPOT (ms):", f"{r['median_tpot_ms']:.2f}"), flush=True)
    print(line("P99 TPOT (ms):", f"{r['p99_tpot_ms']:.2f}"), flush=True)
    print("---------------Inter-token Latency----------------", flush=True)
    print(line("Mean ITL (ms):", f"{r['mean_itl_ms']:.2f}"), flush=True)
    print(line("Median ITL (ms):", f"{r['median_itl_ms']:.2f}"), flush=True)
    print(line("P99 ITL (ms):", f"{r['p99_itl_ms']:.2f}"), flush=True)
    print("----------------End-to-end Latency----------------", flush=True)
    print(line("Mean E2EL (ms):", f"{r['mean_e2el_ms']:.2f}"), flush=True)
    print(line("Median E2EL (ms):", f"{r['median_e2el_ms']:.2f}"), flush=True)
    print(line("P99 E2EL (ms):", f"{r['p99_e2el_ms']:.2f}"), flush=True)
    print("=" * 50, flush=True)


def _result_name(config_name: str, idx: int, bench_cfg: dict) -> str:
    """Match benchmaq's per-config filename scheme so the gateway picks it up as a
    per-config result (anything *.json that isn't literally result.json)."""
    cfg_hash = hashlib.md5(str(sorted(bench_cfg.items())).encode()).hexdigest()[:6]
    parts = [config_name]
    if "random_input_len" in bench_cfg:
        parts.append(f"in{bench_cfg['random_input_len']}")
    if "random_output_len" in bench_cfg:
        parts.append(f"out{bench_cfg['random_output_len']}")
    if "num_prompts" in bench_cfg:
        parts.append(f"p{bench_cfg['num_prompts']}")
    if "max_concurrency" in bench_cfg:
        parts.append(f"c{bench_cfg['max_concurrency']}")
    parts.append(cfg_hash)
    return "_".join(str(p) for p in parts)


async def _run(config: dict) -> int:
    items = config.get("benchmark")
    if not isinstance(items, list) or not items:
        print("[ingress] ERROR: no 'benchmark:' list in config", flush=True)
        return 1

    top_base_url = config.get("base_url")
    any_completed = False
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "ingress")
        base_url = item.get("base_url") or top_base_url
        if not base_url:
            print(f"[ingress] ERROR: '{name}' has no base_url — cannot run ingress bench", flush=True)
            return 1
        model_cfg = item.get("model") or {}
        model = (
            (item.get("serve") or {}).get("model")
            or model_cfg.get("repo_id")
            or model_cfg.get("local_dir")
            or ""
        )
        if not model:
            print(f"[ingress] ERROR: '{name}' has no model.repo_id", flush=True)
            return 1
        bench_list = item.get("bench") or []
        if not isinstance(bench_list, list) or not bench_list:
            print(f"[ingress] '{name}': no bench configs — skipping", flush=True)
            continue

        results_cfg = item.get("results") or {}
        save_result = bool(results_cfg.get("save_result", True))
        save_detailed = bool(results_cfg.get("save_detailed", False))
        result_dir = Path(results_cfg.get("result_dir") or "./benchmark_results")

        tok = _load_tokenizer(model)
        # Deterministic-ish per run but varied per request; seed off the name so
        # reruns of the same config are reproducible.
        rng = random.Random(hash(name) & 0xFFFFFFFF)

        print("=" * 64, flush=True)
        print(f"[ingress] CONFIG: {name}  →  {base_url}  (model={model})", flush=True)
        print("=" * 64, flush=True)

        for i, bench_cfg in enumerate(bench_list):
            if not isinstance(bench_cfg, dict):
                continue
            print(f"\n--- bench {i + 1}/{len(bench_list)} ---", flush=True)
            cfg = dict(bench_cfg)
            cfg["_save_detailed"] = save_detailed
            try:
                result = await _run_bench_config(base_url, model, tok, cfg, rng)
            except Exception as e:
                print(f"[ingress] bench {i + 1} crashed: {type(e).__name__}: {e}", flush=True)
                continue
            if result.get("completed"):
                any_completed = True
            if save_result:
                result_dir.mkdir(parents=True, exist_ok=True)
                fname = _result_name(name, i, bench_cfg) + ".json"
                out = result_dir / fname
                # Drop the private marker before persisting.
                result.pop("_save_detailed", None)
                out.write_text(json.dumps(result, indent=2))
                print(f"[ingress] wrote {out}", flush=True)

    # Exit non-zero only on a hard config error; a run where every request failed
    # still exits 0 and the gateway's "0 successful" detector fails the row (it
    # greps the `Successful requests:` lines we printed).
    return 0 if (any_completed or items) else 1


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m gateway.bench_ingress <config.yaml>", flush=True)
        return 2
    config_path = argv[0]
    try:
        config = yaml.safe_load(Path(config_path).read_text()) or {}
    except Exception as e:
        print(f"[ingress] ERROR: could not read config {config_path}: {e}", flush=True)
        return 1
    if not isinstance(config, dict):
        print("[ingress] ERROR: top-level config must be a mapping", flush=True)
        return 1
    try:
        return asyncio.run(_run(config))
    except KeyboardInterrupt:
        print("[ingress] interrupted", flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
