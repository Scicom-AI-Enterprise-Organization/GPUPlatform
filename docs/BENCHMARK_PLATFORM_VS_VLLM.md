# Benchmark: GPUPlatform vs. direct vLLM

How much overhead does serving a model **through GPUPlatform** add over hitting the
**vLLM engine directly**? This doc is the apples-to-apples comparison, the method to
reproduce it, and where the (small) overhead actually lives.

TL;DR — the serverless core **matches** vLLM (unary ≈ 95% of direct). The only gap is
**streaming**, and it's the per-token Redis relay, which is dominated by the local
reverse-SSH tunnel (a dev-only artifact; on prod with co-located Redis it's far cheaper).

All numbers below: `qwen/qwen3.6-27b`, tensor-parallel 1, one NVIDIA H20-3e, input ≈128
tokens / output 128 tokens, measured 2026-05-30.

## The two paths

```
direct vLLM      client ─────────────► vLLM OpenAI server (localhost:18001)        # zero indirection

GPUPlatform      client ─► gateway /v1/... ─► Redis queue ─► worker BRPOP ─► vLLM (localhost on the VM)
                                          ◄── result ◄── Redis (unary: 1 set;  stream: pub/sub per chunk)
```

Locally the worker reaches the gateway + Redis over a **reverse-SSH tunnel** (the VM
can't see your laptop's localhost). Every Redis op therefore crosses SSH — which is what
makes streaming look slow locally. Prod has no tunnel (public gateway + Redis endpoints).

## Headline comparison (conc 64)

| Path | Output throughput | vs direct | Median TTFT |
|---|---:|---:|---:|
| **Direct vLLM** | **1035 tok/s** | 100% | 7.9 s |
| **GPUPlatform — unary** (`stream:false`) | **985 tok/s** | **95%** ✅ | — |
| GPUPlatform — streaming (`stream:true`) | 670 tok/s | 65% | 11.8 s |
| Browser **Stress tab** (conc 10, ~6-conn cap) | 56 tok/s | 5% ⚠️ | 65 s |

- **Unary matches vLLM.** The enqueue → worker → vLLM → result round-trip adds only ~5%.
- **Streaming is the gap.** The worker publishes every token to Redis and the gateway
  relays each as an SSE event; over the tunnel each token is a round-trip, so the 64
  streams serialize on the tunnel. *(Fix below.)*
- **The browser Stress tab is not a throughput benchmark.** A browser caps ~6
  connections per host (HTTP/1.1), so "concurrency 10" never actually runs 10 — it
  serializes and TTFT explodes. Use it for correctness; drive load from a host client.

## Concurrency curve (direct vLLM, qwen-27b)

```
conc  10 →  442 tok/s   (TTFT  2.9s)
conc  32 →  871 tok/s
conc  64 → 1035 tok/s  ← peak   (TTFT  7.9s)
conc 128 → 1010 tok/s   (TTFT 12.8s)
conc 256 →  972 tok/s   (TTFT 27.6s)   # plateaus — H20 is bf16-compute-bound
```

A single non-batched request is ~61 tok/s; batching at conc 64 is the ~17× win. Beyond
~64 it plateaus (compute wall) and only TTFT grows.

## Per-model (direct vLLM, conc 64)

| Model | Kind | Output throughput | Median TTFT |
|---|---|---:|---:|
| `Qwen/Qwen3.6-35B-A3B` | MoE (~3B active) | **3056 tok/s** | 2.7 s |
| `qwen/qwen3.6-27b` | dense 27B | 1035 tok/s | 7.9 s |
| `google/gemma-4-31b-it` | dense 31B | 995 tok/s | 0.25 s |

The MoE is ~3× the dense models (few active params on a compute-bound GPU); gemma's
prefill/TTFT is dramatically lower.

## How to reproduce

### A. Direct vLLM (ground truth)

On the VM, hit the member's local vLLM port (each fleet member binds `127.0.0.1:<port>`;
see the Workers tab — it shows `localhost:<port>` per model). Async client, fixed
concurrency, streaming, measure aggregate output tok/s:

```bash
# on the VM (or anywhere with direct reach to the vLLM port)
python bench.py --url http://localhost:18001/v1/chat/completions \
  --model qwen/qwen3.6-27b --concurrency 64 --num-prompts 128 \
  --input-len 128 --output-len 128
```

`vllm bench serve` works too if installed in the venv:
```bash
vllm bench serve --backend openai-chat --model qwen/qwen3.6-27b \
  --base-url http://localhost:18001 --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 128 --random-output-len 128 \
  --num-prompts 128 --max-concurrency 64
```

### B. Through GPUPlatform (the real serving path)

Drive the gateway's OpenAI-compatible API from a **host** (not a browser), at real
concurrency. Use the per-endpoint base URL so routing is unambiguous:

```bash
# base_url = <gateway>/<endpoint_id>/v1   — see docs/MULTI_MODEL_FLEET.md
python bench.py --url http://localhost:8080/tm-fleet/v1/chat/completions \
  --model qwen/qwen3.6-27b --concurrency 64 --num-prompts 128 \
  --header "Authorization: Bearer $SGPU_API_KEY"
```

Compare **unary** (`stream:false`) for throughput parity, and **streaming** for the
relay cost. Same `concurrency` / `input-len` / `output-len` as path A.

> The browser **Stress tab** (`/serverless/<id>?tab=stress`) is convenient but
> connection-capped — treat its throughput as a lower bound, not the engine's capacity.

## Where the overhead is (and the streaming fix)

| Stage | Unary | Streaming |
|---|---|---|
| gateway admit + enqueue | 1 Redis op / req | 1 / req |
| worker → vLLM (localhost) | direct (full cost is vLLM) | direct |
| result back | 1 Redis `set` / req | **1 Redis publish per token** |

Streaming's per-token publish (plus a per-token cancel-key check) is the cost; over the
tunnel it serializes. The worker now **pipelines token publishes in small batches**
(one round-trip per ~16 chunks / 20 ms) and **polls the cancel key at most every
~250 ms** instead of per token — cutting per-token Redis round-trips without changing the
SSE wire protocol. On prod (Redis co-located, no tunnel) the per-token cost is small to
begin with, so streaming there tracks unary closely.

## Recommendations

- **Throughput-sensitive / batch:** use unary (`stream:false`) — it already matches vLLM.
- **Latency / UX:** use streaming; the relay cost is per-token, not per-throughput, and
  is minimal on prod.
- **Benchmarking:** drive from a host client at the concurrency you care about; don't
  trust the browser Stress tab for peak numbers.
- **Model choice on H20:** MoE (`35B-A3B`) for raw throughput, gemma for low TTFT.
