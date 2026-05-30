# GPUPlatform

A multi-tenant GPU workload platform. One control plane, three product surfaces:

- **Serverless** — deploy a model with the `serverlessgpu` Python decorator, get an autoscaling HTTP endpoint backed by vLLM (scales to zero when idle) — or stand up a **multi-model fleet on one SSH VM** that time-shares its GPUs via vLLM sleep/wake.
- **Compute** — provision long-lived bare-metal GPU pods with SSH (and Jupyter on PI). Admin approval gate optional.
- **Benchmark** — SSH-orchestrated `llm-benchmaq` runs on RunPod / Prime Intellect / your own VM, with live log streaming and S3-archived results.

Users bring their own provider credentials (RunPod / Prime Intellect / VM SSH); the gateway routes each workload to the right account.

## The `serverlessgpu` library

The SDK + CLI is the first-class way to ship a serverless endpoint:

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

Endpoints are also reachable via an **OpenAI-compatible API** at `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` — point any OpenAI client at the gateway URL and pass an API key from `/api-keys` in the web UI.

The same endpoint can be created from the web UI (`/serverless/new`); the SDK is for code-first workflows and CI deploys.

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

## What's in here

- **Gateway** (`gateway/`) — FastAPI control plane: `/apps`, `/run`, `/stream`, `/v1/*`, `/workers`, `/compute`, `/benchmarks`, `/v1/providers`, `/auth`, `/admin/*`, `/metrics`
- **Worker agent** (`worker-agent/`) — BRPOPs jobs from Redis, runs vLLM, publishes streaming tokens via pub/sub; also drives the **multi-model VM fleet** — wave-loads each vLLM, sleeps/wakes them per request, runs operator commands (sleep/restart/kill), and ships per-model + scheduler logs to the gateway
- **SDK + CLI** (`sdk/serverlessgpu/`) — `@endpoint` decorator, `QueueDepthAutoscaler`, and the `serverlessgpu` CLI
- **Web** (`web/`) — Next.js UI: Serverless, Compute, Benchmark, Providers, API Keys, Admin (users / roles / audit / approvals / provisioned)
- **Helm chart** (`deploy/helm/serverlessgpu/`) — gateway + web + Postgres + Redis + Ingress (SSE-safe) + ServiceMonitor
- **Grafana dashboard** (`deploy/grafana/`) — metrics panels wired to the gateway's Prometheus exporter

## Providers (BYO credentials)

Each user registers their own providers under `/providers`. Supported kinds:

| Kind | What it is | Used by |
|---|---|---|
| `runpod` | RunPod API key | Serverless, Compute, Benchmark |
| `pi` | Prime Intellect API key | Serverless, Compute, Benchmark |
| `vm` | Your own SSH-reachable box | Serverless (multi-model fleet), Benchmark (bare-metal) |
| `fake` | In-process dev provider | Local development only |

Provider resolution per request: explicit `provider_id` → the user's sole provider of that kind → gateway-wide env fallback (`RUNPOD_API_KEY`, `PI_API_KEY`).

## Running locally

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

Both are gitignored — templates live next to them as `.env.example`.

**Reset:** `docker compose down -v` (wipes Postgres + Redis volumes).

**Why no real GPU locally?** Real provisioning (`PROVIDER=runpod` / `=pi`) spawns a pod on the provider's network that has to phone home to your gateway + Redis. `localhost` isn't reachable from the public internet. For real workers, deploy the gateway to k8s — see [docs/DEPLOY.md](docs/DEPLOY.md).

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

## Going further

| Doc | What's in it |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Component walk-through, Redis key schema, design rationale |
| [docs/MULTI_MODEL_FLEET.md](docs/MULTI_MODEL_FLEET.md) | Stand up + drive a multi-model VM fleet entirely with `curl` (create / status / request / sleep-wake / logs / delete) |
| [docs/BENCHMARK_PLATFORM_VS_VLLM.md](docs/BENCHMARK_PLATFORM_VS_VLLM.md) | Throughput: serving through GPUPlatform vs. direct vLLM — method, numbers, where the overhead is |
| [docs/DEPLOY.md](docs/DEPLOY.md) | Local-fake / local-real / k8s+helm deploy paths |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Auth, health probes, timeouts, observability, tear-down |
| [deploy/helm/serverlessgpu/README.md](deploy/helm/serverlessgpu/README.md) | k8s chart values + production wiring |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Local dev, tests, code layout |

## Architecture

Split control-plane / data-plane:

- **Control plane** (this repo) — gateway + web + Postgres + Redis, runs in k8s, CPU-only
- **GPU workers / pods** — spawned on demand on the user's provider account (RunPod / PI / VM), pay only while running

Same shape as Modal / Beam / RunPod Serverless, but multi-tenant on top of users' own provider credentials rather than a single platform-owned fleet.

## License

[Apache License 2.0](LICENSE).
