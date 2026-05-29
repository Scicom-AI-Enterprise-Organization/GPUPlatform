# GPUPlatform

A multi-tenant GPU workload platform. One control plane, three product surfaces:

- **Serverless** — deploy a model with the `serverlessgpu` Python decorator, get an autoscaling HTTP endpoint backed by vLLM. Scales to zero when idle.
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

## What's in here

- **Gateway** (`gateway/`) — FastAPI control plane: `/apps`, `/run`, `/stream`, `/v1/*`, `/workers`, `/compute`, `/benchmarks`, `/v1/providers`, `/auth`, `/admin/*`, `/metrics`
- **Worker agent** (`worker-agent/`) — BRPOPs jobs from Redis, runs vLLM, publishes streaming tokens via pub/sub
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
| `vm` | Your own SSH-reachable box | Benchmark (bare-metal) |
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
