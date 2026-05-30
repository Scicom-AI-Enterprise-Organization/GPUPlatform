# GPUPlatform

A multi-tenant GPU workload platform. One control plane, three product surfaces:

- **Serverless** — deploy a model with the `serverlessgpu` Python decorator (or the web UI) and get an autoscaling, OpenAI-compatible HTTP endpoint backed by vLLM (scales to zero when idle) — or stand up a **multi-model fleet on one SSH VM** that time-shares its GPUs via vLLM sleep/wake.
- **Compute** — provision long-lived bare-metal GPU pods with SSH (and JupyterLab on Prime Intellect). Optional admin-approval gate.
- **Benchmark** — SSH-orchestrated [`llm-benchmaq`](https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq) sweeps on RunPod / Prime Intellect / your own VM, with live log streaming, S3-archived results, and a side-by-side compare view.

Users bring their own provider credentials (RunPod / Prime Intellect / VM SSH); the gateway routes each workload to the right account and bills nothing of its own — you pay the provider only while a worker runs.

## Contents

- [How it works](#how-it-works)
- [Quickstart (run locally)](#quickstart-run-locally)
- [Serverless inference](#serverless-inference)
  - [Deploy — SDK + CLI](#deploy--sdk--cli)
  - [Call it — OpenAI-compatible API](#call-it--openai-compatible-api)
  - [Operate it — the endpoint console](#operate-it--the-endpoint-console)
- [Multi-model fleet on one VM](#multi-model-fleet-on-one-vm-gpu-time-sharing)
- [Benchmark](#benchmark)
- [Compute](#compute)
- [Providers (BYO credentials)](#providers-byo-credentials)
- [API keys & the HTTP API](#api-keys--the-http-api)
- [Testing](#testing)
- [Auth, RBAC, and admin](#auth-rbac-and-admin)
- [Repo layout](#repo-layout)
- [Docs](#docs)

## How it works

Split control-plane / data-plane — the same shape as Modal / Beam / RunPod Serverless, but **multi-tenant on top of users' own provider credentials** rather than a single platform-owned fleet:

```
                     ┌──────────── control plane (k8s, CPU-only) ───────────┐
 client              │                                                      │
  │  OpenAI / SDK     │   gateway (FastAPI) ──▶ Postgres (apps, audit, …)    │
  ▼  HTTP             │        │      │                                      │
 web UI ──────────────▶       │      └──▶ Redis  (job queue + pub/sub)       │
                     │  autoscaler / reconciler                             │
                     └────────────────────┬─────────────────────────────────┘
                                          │ provision on demand
                                          │ (RunPod / Prime Intellect / SSH VM)
                                          ▼
                          GPU worker(s) — vLLM + worker-agent
                          BRPOP jobs ◀─ Redis ─▶ publish streamed tokens
```

- **Control plane** (this repo) — gateway + web + Postgres + Redis. CPU-only, runs in k8s (Helm chart included).
- **Data plane** — GPU workers/pods spawned on demand on the user's provider account. A worker registers with the gateway, `BRPOP`s jobs off its Redis queue, runs vLLM, and publishes streaming tokens back via pub/sub.

Deeper dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (component walk-through, Redis key schema, design rationale).

## Quickstart (run locally)

Full stack on your laptop with the `fake` provider — no external GPU billing.

**Pre-reqs:** Docker, [uv](https://docs.astral.sh/uv/), Node 20+.

```bash
# 1. Install Python packages (gateway + sdk + worker-agent, editable)
make install

# 2. Postgres + Redis (leave gateway/worker services off — we run them locally)
docker compose up -d postgres redis

# 3. Gateway — FastAPI on :8080, reads gateway/.env
.venv/bin/gateway

# 4. Web — Next.js on :3000, reads web/.env.local
cd web && npm install && npm run dev
```

Open `http://localhost:3000`. With `AUTH_DISABLED=1` in `gateway/.env`, login is `admin / admin`. Deploy from the UI or `serverlessgpu deploy ...` — the fake worker handles requests in-process.

| Env file | Sets |
|---|---|
| `gateway/.env` | `DATABASE_URL`, `REDIS_URL`, `AUTH_DISABLED`, `PROVIDER` (default `fake`), `AUTOSCALER`, optional `RUNPOD_API_KEY` / `PI_API_KEY` |
| `web/.env.local` | `NEXT_PUBLIC_GATEWAY_URL=http://localhost:8080` |

Both are gitignored — templates live next to them as `.env.example`. **Reset:** `docker compose down -v` (wipes Postgres + Redis volumes).

**Why no real GPU locally?** Real provisioning (`PROVIDER=runpod` / `=pi`) spawns a pod on the provider's network that has to phone home to your gateway + Redis. `localhost` isn't reachable from the public internet. For real workers, deploy the gateway to k8s — see [docs/DEPLOY.md](docs/DEPLOY.md).

## Serverless inference

One model → one autoscaling, OpenAI-compatible endpoint backed by vLLM.

### Deploy — SDK + CLI

The `serverlessgpu` library is the first-class, code-first way to ship an endpoint:

```python
# app.py
from serverlessgpu import endpoint, QueueDepthAutoscaler

@endpoint(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu="H100",
    autoscaler=QueueDepthAutoscaler(max_containers=3, tasks_per_container=30, idle_timeout_s=300),
)
def qwen():
    pass  # vLLM serves the model — no body needed
```

```bash
serverlessgpu deploy app.py:qwen
serverlessgpu run    qwen --payload '{"prompt": "hello"}'
serverlessgpu stream qwen --payload '{"prompt": "tell me a story"}'   # SSE token streaming
serverlessgpu list
serverlessgpu show   qwen
serverlessgpu delete qwen
```

The same endpoint can be created from the web UI at `/serverless/new` (provider, GPU, autoscaler, and vLLM engine args — which are validated at create time, so a bad flag is rejected up front instead of failing the worker later).

### Call it — OpenAI-compatible API

Every endpoint is reachable at `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings`. Point any OpenAI client at the gateway URL with an API key from the **API tokens** page (prefix `sgpu_`). The `model` field is the **endpoint name** for a single-model endpoint, or a **member model name** for a multi-model fleet.

```bash
GATEWAY=http://localhost:8080            # your gateway URL
KEY=sgpu_...                             # from the API tokens page

# Chat completion
curl -s "$GATEWAY/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"hi"}],"max_tokens":256}'

# Streaming (SSE, token-by-token)
curl -sN "$GATEWAY/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stream":true}'

# Discover every served model across the fleet (no token required)
curl -s "$GATEWAY/v1/models"
```

```python
from openai import OpenAI
client = OpenAI(base_url=f"{GATEWAY}/v1", api_key=KEY)
client.chat.completions.create(model="qwen", messages=[{"role": "user", "content": "hi"}])
```

A lower-level async job API also exists — `POST /run/{app}` returns a `request_id` you poll at `/result/{id}`, and `POST /stream/{app}` is its SSE form. The OpenAI routes wrap these.

### Operate it — the endpoint console

The endpoint detail page (`/serverless/{id}`) has a tab for each operational concern:

- **Overview** — model/GPU/autoscaler config, env vars, per-model vLLM args, the OpenAI/cURL snippets, and **Redeploy** / **Delete**.
- **Playground** — fire chat completions interactively: model picker, `reasoning_effort`, `temperature`, "disable thinking" (`chat_template_kwargs.enable_thinking=false`), streaming toggle. Shows **reasoning + answer** panels live, **tokens/sec + TTFT**, and the equivalent **cURL** for the exact request. All settings deep-link into the URL.
- **Stress test** — a `vllm bench serve`-style concurrency load against the live endpoint (input/output length, num prompts, concurrency) reporting throughput + TTFT/TPOT/E2E latency percentiles.
- **Queue** — live `queue:{app}` + `result:*` view; click a request id to deep-link it.
- **Workers** — for a fleet, each model's state (awake / asleep / loading / dead, **why** it died), its GPUs, TP, in-flight count, **localhost:port**, and per-model **Sleep / Restart / Kill / Logs** actions.

## Multi-model fleet on one VM (GPU time-sharing)

Instead of one model per endpoint, a single SSH-reachable VM can host **several vLLM
servers that share its GPUs** by sleeping and waking on demand. You pick the models,
pin each to a tensor-parallel GPU slice, and the gateway packs them onto the VM's
GPUs; the worker-agent loads them, puts the idle ones to sleep, and wakes the right
one when a request arrives.

- **Request-driven sleep/wake** — a request for an asleep model wakes it; if its GPUs
  are held by other awake models, those are slept first (LRU), then the target wakes.
  The OpenAI API returns a clean `503 warming_up` / `503 dead` (with the failure
  reason) instead of hanging.
- **Wave-loading** — models on non-overlapping GPUs load concurrently; overlapping
  ones serialize behind a per-GPU lock, so a 119B model can't OOM a 27B one mid-boot.
- **vLLM sleep levels** — level 1 offloads weights to CPU RAM (fast wake); level 2
  discards them for a smaller footprint (slow wake = reload from disk). Chosen per fleet.
- **Operate from the UI** — the Workers tab shows each model's state and *why* a dead
  one died, with per-model **Sleep / Restart / Kill / Logs** actions, a fleet-wide
  **Sleep all**, and a **Worker log** view of the scheduler itself.
- **Public model list** — `GET /v1/models` lists every served model across the fleet
  (no token required), so any OpenAI client can discover them.

Stand one up and drive it entirely with `curl`: [docs/MULTI_MODEL_FLEET.md](docs/MULTI_MODEL_FLEET.md).

## Benchmark

Run [`llm-benchmaq`](https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq) (which wraps `vllm bench serve`) against a model and archive the results — on a RunPod/PI pod the gateway spins up, or on your own SSH VM (bare-metal).

- **Config** — the benchmaq YAML: one or more `serve` configs × a `bench` sweep matrix (input/output length × concurrency × num prompts). Build it in the form at `/benchmark/new` or paste raw YAML; the gateway validates and rewrites the `remote:` block for the chosen target.
- **VM runs** — pin to specific GPUs with `visible_devices` (→ `CUDA_VISIBLE_DEVICES`; the count is the tensor-parallel size). The gateway delivers the config over SSH, installs benchmaq, runs the sweep, and **never uploads the VM's private key** to S3.
- **Live + archived** — stream the log tail while it runs; on completion an aggregate `result.json` + per-config files land in your S3 storage.
- **Results + Compare** — the detail page charts throughput / TTFT / TPOT / E2EL across the sweep; select multiple runs from the list to **compare** them side by side. Runs are renamable, duplicable, and terminable.

Fire one via the API with a key — see [API keys & the HTTP API](#api-keys--the-http-api) below and [docs/BENCHMARK_PLATFORM_VS_VLLM.md](docs/BENCHMARK_PLATFORM_VS_VLLM.md) (platform vs. direct-vLLM overhead, with numbers).

## Compute

Raw GPU pods for interactive work, not serving. Provision a long-lived pod (`/compute/new`) with a chosen GPU/count, container disk, and template; get back an **SSH command** (and a **JupyterLab** URL + password on Prime Intellect). Live cost tracking per pod, and an optional **admin-approval gate** — pods land in `pending` until cleared (toggle per role). Terminate from the UI; the gateway tears down the provider pod.

## Providers (BYO credentials)

Each user registers their own providers under `/providers`. Supported kinds:

| Kind | What it is | Used by |
|---|---|---|
| `runpod` | RunPod API key | Serverless, Compute, Benchmark |
| `pi` | Prime Intellect API key | Serverless, Compute, Benchmark |
| `vm` | Your own SSH-reachable box | Serverless (multi-model fleet), Benchmark (bare-metal) |
| `fake` | In-process dev provider | Local development only |

Credentials are encrypted at rest. Provider resolution per request: explicit `provider_id` → the user's sole provider of that kind → gateway-wide env fallback (`RUNPOD_API_KEY`, `PI_API_KEY`).

## API keys & the HTTP API

Mint a long-lived bearer token on the **API tokens** page (or `POST /api-keys`). It's shown once, hashed at rest, prefix `sgpu_`. Send it as `Authorization: Bearer sgpu_...`. One key works across every surface:

| Area | Key endpoints |
|---|---|
| Serverless | `POST /apps` · `GET /apps` · `GET /apps/{id}/status` · `POST /v1/chat/completions` · `GET /v1/models` · `POST /run/{app}` · `GET /result/{id}` |
| Benchmark | `POST /benchmarks` · `GET /benchmarks/{id}` · `GET /benchmarks/{id}/logs?tail=` · `POST /benchmarks/{id}/duplicate` · `PATCH /benchmarks/{id}` (rename) · `POST /benchmarks/{id}/terminate` |
| Compute | `POST /compute` · `GET /compute` · `GET /compute/{id}/ssh` |
| Providers / storage | `GET /v1/providers` · `GET /v1/storage` |
| Ops | `GET /health` · `GET /ready` · `GET /metrics` |

```bash
# create a serverless endpoint via the API
curl -s -X POST "$GATEWAY/apps" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"name":"qwen","model":"Qwen/Qwen2.5-7B-Instruct","gpu":"H100","provider_id":"prov-..."}'
```

vLLM args are validated at create time — a stray shell-continuation `\`, unbalanced quotes, or a platform-reserved flag (`--model`, `--port`, `--tensor-parallel-size` on a fleet member, …) is rejected with a clear `400` instead of producing an endpoint that silently fails to launch.

## Testing

### JS unit tests (gateway API layer)

The web client's API calls are covered by [vitest](https://vitest.dev) specs that assert creating serverless endpoints and SSH/VM benchmarks produces the exact request the gateway expects. The network and the `sgpu_token` cookie → `Bearer` auth are mocked, so **no live gateway and no API key are needed**:

```bash
cd web
npm install
npm test            # run once
npm run test:watch  # watch mode
```

Specs live under `web/src/**/*.test.ts`:

| Spec | Covers |
|---|---|
| `src/lib/benchmark-ssh.test.ts` | SSH/VM benchmark create — replicates a 16-cell vLLM sweep pinned to chosen GPUs (`visible_devices`) — plus the get / list / rename / terminate / delete / files routes |
| `src/lib/__tests__/create-inference.test.ts` | Multi-model VM serverless endpoint create — the GPU-pinned `export`-env vLLM fleet |
| `src/lib/__tests__/request-inference.test.ts` | Requesting `/v1/chat/completions` + `/v1/models` for each fleet model (parametrized) |
| `src/app/(app)/serverless/__tests__/deploy-endpoint.test.ts` | Endpoint deploy flow |

One spec — `src/lib/__tests__/real-api.integration.test.ts` — fires the same calls at a
**live gateway** and is skipped unless `SGPU_API_KEY` is set:

```bash
cd web
SGPU_API_KEY=sgpu_... SGPU_URL=http://localhost:8080 npm test -- real-api
```

### Running a real benchmark via the API (with an API key)

The same request the SSH-benchmark spec asserts can be fired at a live gateway with a real key — an end-to-end "battle test" on actual GPUs.

1. **Mint a key** on the **API tokens** page in the UI (or `POST /api-keys`). It's shown once; prefix `sgpu_`.
2. **Find your VM provider + storage ids** — `GET /v1/providers` (a `kind: vm` SSH box) and `GET /v1/storage` (an enabled `s3` backend).
3. **Create the benchmark** — pin it to specific GPUs with `visible_devices` (this becomes `CUDA_VISIBLE_DEVICES` on the VM; the count is the tensor-parallel size). The gateway rewrites the `remote:` block to run on the registered VM over SSH.

```bash
export SGPU_API_KEY=sgpu_...            # your key
export SGPU_URL=http://localhost:8080

# config_yaml is the benchmaq YAML the New-Benchmark form produces; the easiest
# way to get a real one is to copy the `config_yaml` field of an existing run
# (GET /benchmarks/<id>) and tweak it. Minimal one-cell example below.
curl -s -X POST "$SGPU_URL/benchmarks" \
  -H "Authorization: Bearer $SGPU_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "battletest-gpu67",
    "provider_id": "prov-xxxx",
    "storage_id":  "store-xxxx",
    "visible_devices": "6,7",
    "cleanup_model": false,
    "config_yaml": "remote:\n  uv: {path: ~/.benchmark-venv, python_version: \"3.11\"}\n  dependencies: [vllm==0.19.1, huggingface_hub, hf_transfer]\nbenchmark:\n- name: bt\n  engine: vllm\n  model: {repo_id: qwen/qwen3.6-27b, local_dir: ~/models/qwen3p6-27b}\n  serve: {tensor_parallel_size: 2, port: 18017}\n  bench:\n  - {endpoint: /v1/completions, dataset_name: random, random_input_len: 128, random_output_len: 128, num_prompts: 50, max_concurrency: 50}\n  results: {save_result: true}"
  }'
# → {"id":"bench-...","status":"queued"}

# Poll status + stream the log tail:
curl -s -H "Authorization: Bearer $SGPU_API_KEY" "$SGPU_URL/benchmarks/<id>"
curl -s -H "Authorization: Bearer $SGPU_API_KEY" "$SGPU_URL/benchmarks/<id>/logs?tail=200"
```

A finished run reports `status: done`, `exit_code: 0`, and writes an aggregate `result.json` to your S3 storage (readable in the UI **Results** / **Compare** tabs). Same key works for the serverless (`/run/{app}`, `/v1/chat/completions`), compute, and storage APIs.

## Auth, RBAC, and admin

- **GitHub SSO** at `/auth/github/upsert`, plus password login at `/auth/login`. `AUTH_DISABLED=1` short-circuits to `admin/admin` for local dev.
- **Tier roles** (`user` / `developer` / `admin`) gate the sidebar; **policy roles** layer fine-grained RBAC on top.
- **Audit log** at `/admin/audit` records every mutating action.
- **Compute approvals** at `/admin/compute-approvals` — pods land in `pending` until an admin clears them (toggle off per role).
- **Provisioned** at `/admin/provisioned` — live view of every pod + serverless app across users, with cost tracking.

## Repo layout

- **Gateway** (`gateway/`) — FastAPI control plane: `/apps`, `/run`, `/stream`, `/v1/*`, `/workers`, `/compute`, `/benchmarks`, `/v1/providers`, `/v1/storage`, `/api-keys`, `/auth`, `/admin/*`, `/metrics`
- **Worker agent** (`worker-agent/`) — `BRPOP`s jobs from Redis, runs vLLM, publishes streaming tokens via pub/sub; also drives the **multi-model VM fleet** — wave-loads each vLLM, sleeps/wakes them per request, runs operator commands (sleep/restart/kill), and ships per-model + scheduler logs to the gateway
- **SDK + CLI** (`sdk/serverlessgpu/`) — `@endpoint` decorator, `QueueDepthAutoscaler`, and the `serverlessgpu` CLI
- **Web** (`web/`) — Next.js UI: Serverless, Compute, Benchmark, Providers, API tokens, Admin (users / roles / audit / approvals / provisioned)
- **Helm chart** (`deploy/helm/serverlessgpu/`) — gateway + web + Postgres + Redis + Ingress (SSE-safe) + ServiceMonitor
- **Grafana dashboard** (`deploy/grafana/`) — metrics panels wired to the gateway's Prometheus exporter
- **CI** (`.github/workflows/ci.yml`) — helm lint + e2e (kind), and matrix multi-arch image builds (gateway + web → amd64/arm64 → merged manifest; PI worker → amd64)

## Docs

| Doc | What's in it |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Component walk-through, Redis key schema, design rationale |
| [docs/MULTI_MODEL_FLEET.md](docs/MULTI_MODEL_FLEET.md) | Stand up + drive a multi-model VM fleet entirely with `curl` (create / status / request / sleep-wake / logs / delete) |
| [docs/BENCHMARK_PLATFORM_VS_VLLM.md](docs/BENCHMARK_PLATFORM_VS_VLLM.md) | Throughput: serving through GPUPlatform vs. direct vLLM — method, numbers, where the overhead is |
| [docs/WORKER_AGENT_PROVISIONING.md](docs/WORKER_AGENT_PROVISIONING.md) | How the worker-agent gets onto a VM (source-tarball ship over SSH, no git clone) + the prod image-bundling requirement |
| [docs/DEPLOY.md](docs/DEPLOY.md) | Local-fake / local-real / k8s+helm deploy paths |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Auth, health probes, timeouts, observability, tear-down |
| [deploy/helm/serverlessgpu/README.md](deploy/helm/serverlessgpu/README.md) | k8s chart values + production wiring |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Local dev, tests, code layout |

## License

[Apache License 2.0](LICENSE).
