# GPUPlatform

A multi-tenant GPU workload platform. One control plane, four product surfaces:

- **Serverless** — deploy a model with the `serverlessgpu` Python decorator (or the web UI) and get an autoscaling, OpenAI-compatible HTTP endpoint backed by vLLM — chat / completions / embeddings, plus **Whisper audio** (`/v1/audio/transcriptions`) — that scales to zero when idle. Or stand up a **multi-model fleet** that time-shares its GPUs via vLLM sleep/wake, on your own **SSH VM** or a **cloud RunPod pod** — the cloud fleet scales to zero on idle and re-provisions itself on the next request.
- **Autotrain** — finetune **Whisper (ASR)** or **Qwen3 + NeuCodec (TTS)** on your own datasets. Hyperparameter sweeps, audio augmentation, live per-step loss + per-epoch WER/CER, GPU telemetry, and W&B / MLflow tracking — orchestrated over SSH on your VM or a RunPod pod, with the model pushed to S3 / Hugging Face.
- **Benchmark** — SSH-orchestrated [`llm-benchmaq`](https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq) sweeps on RunPod / Prime Intellect / your own VM, with live log streaming, S3-archived results, and a side-by-side compare view.
- **Compute** — provision long-lived bare-metal GPU pods with SSH (and JupyterLab on Prime Intellect). Optional admin-approval gate.

These sit on shared infrastructure you configure once: **Providers** (BYO RunPod / Prime Intellect / VM SSH credentials), **Storage** (S3 / Hugging Face backends), **Datasets** (audio + transcription corpora for Autotrain), and org-wide **Secrets** (global env vars + W&B / MLflow tracking credentials).

Users bring their own provider credentials; the gateway routes each workload to the right account and bills nothing of its own — you pay the provider only while a worker runs.

## Contents

- [How it works](#how-it-works)
- [Quickstart (run locally)](#quickstart-run-locally)
- [Serverless inference](#serverless-inference)
  - [Deploy — SDK + CLI](#deploy--sdk--cli)
  - [Call it — OpenAI-compatible API](#call-it--openai-compatible-api)
  - [Operate it — the endpoint console](#operate-it--the-endpoint-console)
- [Multi-model fleets (VM or cloud RunPod)](#multi-model-fleets-vm-or-cloud-runpod)
- [Autotrain (finetuning)](#autotrain-finetuning)
- [Datasets](#datasets)
- [Models (self-hosted Hugging Face catalog)](#models-self-hosted-hugging-face-catalog)
- [Benchmark](#benchmark)
- [Compute](#compute)
- [Providers (BYO credentials)](#providers-byo-credentials)
- [Storage](#storage)
- [Secrets (global env + tracking)](#secrets-global-env--tracking)
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
- **Data plane** — two patterns, both on the user's own provider account:
  - **Serverless workers** — a GPU worker (a registered VM, or a RunPod pod the gateway provisions and **reverse-SSH-tunnels** so even a localhost gateway is reachable) `BRPOP`s jobs off its Redis queue, runs vLLM, and publishes streaming tokens back via pub/sub. Cloud fleets scale to zero when idle and re-provision on the next request.
  - **SSH-orchestrated jobs (Autotrain + Benchmark)** — the gateway connects over SSH to a VM (or a freshly-spawned RunPod/PI pod), ships the runner script, runs it pinned to the chosen GPUs, parses structured progress off stdout (loss/metrics/log tail) for the live UI + SSE, and archives the artifacts (trained model / `result.json`) to your S3 storage.

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

**Real GPUs from a local gateway?** A spawned pod must phone home to your gateway + Redis, and `localhost` isn't reachable from the provider's network. The **RunPod and VM providers solve this with a reverse SSH tunnel** (on by default): the gateway SSHes into the pod and forwards the pod's loopback `gateway`+`redis` ports back home, so even a laptop gateway can drive real RunPod/VM workers — no public ingress, no custom worker image. (Prime Intellect has no such tunnel; for PI, or for a hands-off prod deploy, run the gateway in k8s — see [docs/DEPLOY.md](docs/DEPLOY.md).)

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
serverlessgpu pi-check                                                 # preflight a Prime Intellect key + list pods
```

The same endpoint can be created from the web UI at `/serverless/new` (provider, GPU, autoscaler, and vLLM engine args — which are validated at create time, so a bad flag is rejected up front instead of failing the worker later).

### Call it — OpenAI-compatible API

Every endpoint is reachable at `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings` — and, for a Whisper member, `/v1/audio/transcriptions` + `/v1/audio/translations` (multipart file upload). Point any OpenAI client at the gateway URL with an API key from the **API tokens** page (prefix `sgpu_`). The `model` field is the **endpoint name** for a single-model endpoint, or a **member model name** for a multi-model fleet. All routes also exist scoped under `/{app_id}/v1/...`.

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

# Whisper transcription — multipart upload; model = the ASR member's name
curl -s "$GATEWAY/v1/audio/transcriptions" \
  -H "Authorization: Bearer $KEY" \
  -F file=@clip.mp3 -F model=openai/whisper-large-v3

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

- **Overview** — model/GPU/autoscaler config (idle timeout is editable for single-model and **cloud multi-model** endpoints; read-only for an always-on VM fleet), env vars, per-model vLLM args, the OpenAI/cURL snippets, and **Redeploy** / **Delete**.
- **Playground** — a **Chat** ↔ **Audio transcription** toggle. *Chat:* model picker, `reasoning_effort`, `temperature`, "disable thinking" (`chat_template_kwargs.enable_thinking=false`), streaming toggle; live **reasoning + answer** panels, **tokens/sec + TTFT**, and the equivalent **cURL**. *Audio:* upload a clip and transcribe/translate with a Whisper member. All settings deep-link into the URL.
- **Stress test** — a `vllm bench serve`-style concurrency load against the live endpoint (input/output length, num prompts, concurrency) reporting throughput + TTFT/TPOT/E2E latency percentiles.
- **Metrics** — live graphs scraped from the endpoint's own Prometheus exporter at `/{app_id}/metrics`: requests over time + errors, requests-by-route, and latency. Gateway-side HTTP metrics (`serverless_http_requests_total` / `_duration_seconds`), not persisted.
- **Queue** — live request view bucketed **in queue / in progress / completed / failed**; click a request id to deep-link it. **Flush queue** (a styled confirm dialog) drops jobs still waiting *and* clears orphaned/stuck `pending` rows that a worker dequeued but never finalized.
- **Workers** — for a fleet, each model's state (awake / asleep / loading / dead, **why** it died), GPUs, TP, in-flight count, and per-model **Sleep / Restart / Kill / Logs**. For a **cloud (RunPod)** endpoint, also a pods table: each container links to the **RunPod console**, an **Alive** column flags whether the worker actually registered + is heartbeating (vs a pod that's "running" but never phoned home), and a per-container **Delete** frees the GPU (the next request re-provisions).

## Multi-model fleets (VM or cloud RunPod)

Instead of one model per endpoint, a single worker can host **several vLLM servers that
share its GPUs** by sleeping and waking on demand. You pick the models, pin each to a
tensor-parallel GPU slice, and the gateway packs them onto the worker's GPUs; the
worker-agent loads them, sleeps the idle ones, and wakes the right one per request. A
fleet runs on either:

- **your own SSH VM** — always-on; the gateway ships the worker-agent over SSH and the
  GPUs are time-shared by sleep/wake; **or**
- **a cloud RunPod pod** — provisioned on demand from a stock CUDA image (no custom image
  to build). The gateway opens a **reverse SSH tunnel** into the pod, ships the
  worker-agent tarball, `uv pip install`s vLLM into a pod venv, and launches it — all
  reachable from a localhost gateway. The cloud fleet **scales to zero** when idle and
  **re-provisions on the next request**.

- **Request-driven sleep/wake** — a request for an asleep model wakes it; if its GPUs
  are held by other awake models, those are slept first (LRU), then the target wakes.
  The OpenAI API returns a clean `503 warming_up` / `503 dead` (with the failure
  reason) instead of hanging.
- **Wave-loading** — models on non-overlapping GPUs load concurrently; overlapping
  ones serialize behind a per-GPU lock, so a 119B model can't OOM a 27B one mid-boot.
- **vLLM sleep levels** — level 1 offloads weights to CPU RAM (fast wake); level 2
  discards them for a smaller footprint (slow wake = reload from disk). Chosen per fleet.
- **Scale to zero / from zero** (cloud) — with `idle_timeout_s > 0` the idle pod is
  deleted (no GPU bill); the next request re-provisions it, waits out the cold load, and
  is served. The idle-terminator never kills a pod that's still **loading** or has
  **in-flight** requests, so the waking request isn't stranded mid cold-start.
  `idle_timeout_s = 0` keeps the fleet always-on.
- **Cancel on disconnect** — if a client times out / disconnects, the gateway marks the
  request cancelled (it shows immediately as `failed` in the Queue tab), removes it from
  the queue, and signals the worker — which skips it *before* waking a model (and stops
  mid-stream if it was already running), so no GPU burns on abandoned work.
- **Operate from the UI** — the Workers tab shows each model's state and *why* a dead one
  died, with per-model **Sleep / Restart / Kill / Logs**, a fleet-wide **Sleep all**, the
  scheduler's own **Worker log**, and (cloud) the pod's **RunPod-console** link, an
  **Alive** indicator, and a **Delete container** button.
- **Public model list** — `GET /v1/models` lists every served model across the fleet
  (no token required), so any OpenAI client can discover them.

Stand one up and drive it entirely with `curl`: [docs/MULTI_MODEL_FLEET.md](docs/MULTI_MODEL_FLEET.md).

## Autotrain (finetuning)

Finetune a speech model on your own [dataset](#datasets) — orchestrated over SSH on a `vm` provider (bare-metal) or a RunPod pod the gateway spawns, with the trained model pushed to S3 (and optionally Hugging Face). Create a run in the form at `/autotrain/new` or via `POST /v1/training-runs`.

- **Two task types** — **ASR** (Whisper: `whisper-large-v3`, `-v3-turbo`, …), evaluated per-epoch on **WER / CER** with early-stopping on patience; and **TTS** (Qwen3 + NeuCodec), trained on a `tts_packed` [dataset](#datasets) and evaluated post-training on **CER** (Whisper-transcribe), **MOS** (UTMOSv2), and **speaker similarity** (TitaNet).
- **Hyperparameter sweep** — pass a `sweep` grid (e.g. `{"learning_rate":[1e-4,1e-5],"precision":["fp32-bf16","bf16-bf16"]}`) and the gateway runs the cross-product of trials, packing `gpus_per_trial` onto the GPUs you pinned with `visible_devices`. The detail page lists every trial (pending / running / done / failed) and splits the loss + WER/CER curves **per trial**, legended by params.
- **Audio augmentation** — apply any of 8 techniques (`telephone`, `noise`, `dropout`, `gain`, `pitch`, `speed`, `reverb`, `bandpass`) at a chosen probability to the training split only (eval is never augmented).
- **Live + persisted metrics** — per-step training loss, per-epoch eval (WER/CER/eval-loss), and a **GPU telemetry** graph (util / memory / temperature per GPU). All are persisted, so a finished run still renders its charts. Pull everything programmatically from `GET /v1/training-runs/{id}/metrics`.
- **Lazy data loading** — the trainer indexes **metadata only** up front and fetches each clip's audio from S3 / HF inside the DataLoader (`__getitem__`), so a multi-GB corpus costs ~nothing to start and doesn't stall on a slow shared mount.
- **Experiment tracking** — point a run at a [tracking credential](#secrets-global-env--tracking) to log to **Weights & Biases** or **MLflow**.
- **Operate it** — tabbed detail (**Metrics / Logs / Files / Config / Try it**), live log SSE, a Config tab (the VM/provider, GPU type + ids, storage, dataset, base model), per-run **Rename / Restart / Terminate / Delete** (a clean confirmation dialog, no native alert), and `work_dir` + `cleanup_checkpoints` to control scratch/checkpoint disk. Arbitrary OS `env_vars` (e.g. `HF_HOME`, cache dirs) are exported around the trainer.
- **Try it** — once a run finishes, **transcribe a clip** (ASR) or **synthesize speech** (TTS) with the finetuned model directly on the run's VM, from the **Try it** tab. A persistent mode keeps the model resident between calls so you don't pay the load cost on every request.

```bash
curl -s -X POST "$GATEWAY/v1/training-runs" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{
    "name":"whisper-emgs", "task_type":"asr",
    "base_model":"openai/whisper-large-v3-turbo",
    "dataset_id":"ds-...", "test_dataset_id":"ds-...",
    "provider_id":"prov-...", "visible_devices":"6,7", "gpus_per_trial":1,
    "storage_id":"store-...", "max_epochs":10, "batch_size":8,
    "augment_techniques":["telephone","noise","reverb"], "augment_prob":0.5,
    "sweep":{"learning_rate":[1e-4,1e-5],"precision":["fp32-bf16","bf16-bf16"]}
  }'
# poll metrics (steps / per-epoch WER+CER / per-trial status / GPU samples):
curl -s -H "Authorization: Bearer $KEY" "$GATEWAY/v1/training-runs/<id>/metrics"
```

## Datasets

A dataset is a named pointer to a metadata table of `{audio, transcription}` (optionally `speaker`) rows that Autotrain consumes. Manage them at `/datasets` or `POST /v1/datasets`. Five kinds:

| Kind | Source |
|---|---|
| `upload` | a CSV / JSON / JSONL metadata file you upload to an S3 [storage](#storage) backend |
| `s3` | a metadata file already in S3 — a full `s3://bucket/key` URI **or** a key relative to the storage bucket |
| `hf` | a Hugging Face audio dataset (read lazily, per-split, with a token from a HF storage backend) |
| `label` | a live project on a labeling platform |
| `tts_packed` | the output of a **Pack for TTS** step — NeuCodec speech tokens multipacked into a ChiniDataset that TTS training streams directly |

The detail page (`/datasets/{id}`) is tab-organized — **Rows / Columns / Transform / Details**:

- **Rows** — paginate the whole corpus with inline **audio playback + waveform**, and **curate the training split in place**: untick a row to exclude it (`POST …/row-inclusion`) and the trainers skip it. A packed dataset instead shows one row per multipacked block, decoded back to text on demand.
- **Columns** — map the `audio` / `transcription` / `speaker` columns (with per-split transcription overrides); lists the HF **splits**.
- **Transform** — one operation per source kind. An `hf`/`label` dataset stores audio in archives / behind a label export, so it can only **extract a real audio column** (→ materialized to S3 or pushed to HF). An `s3`/`upload` dataset already has audio, so it can **Pack for TTS** — NeuCodec-encode + multipack into a `tts_packed` dataset on a GPU (a RunPod pod or your VM), with live log + progress. Uploaded metadata can also be **synced** to a HF repo.
- **Details** — source/storage metadata, the S3 folder **size** (computed on demand), and — for a transformed dataset — a **Transformed from** link back to the source dataset + its original HF repo.

## Models (self-hosted Hugging Face catalog)

Host your own models **and** datasets on the platform's [Storage](#storage) backends (S3 / local / SFTP) and use **standard Hugging Face tooling** against the gateway — a Hub-compatible mirror mounted at `/hf`. Manage repos in the web **Models** section (`/models`, `/models/new`, detail at `/models/{namespace}/{name}`) or via `POST /v1/catalog`; auth is a platform `sgpu_` API key.

```bash
export HF_ENDPOINT=http://<gateway>:8080/hf
export HF_TOKEN=sgpu_...                               # a platform API key
huggingface_hub.snapshot_download("ns/name")           # read
from_pretrained("ns/name")  /  load_dataset("ns/name") # read (model / dataset)
model.push_to_hub("ns/name")  /  hf upload ns/name .   # write (overwrites main)
```

The bytes live in a `Storage` backend; the file list is the repo's `manifest`. Regular files are stored at `{prefix}/{path}`, large files content-addressed via Git-LFS at `{prefix}/.hf-lfs/{oid}`.

**Revision — always `main`.** The catalog is deliberately single-revision: there are **no branches, tags, commits, or PRs**. Every request that carries a revision — `from_pretrained("ns/name", revision="v1.0")`, `snapshot_download(..., revision="abc123")`, an `@branch` ref, `--revision …` — is **accepted and resolves to `main`** (it won't 404), so existing tooling/configs that pin a revision keep working; the revision is simply ignored. To "publish a new version", **push again** — it overwrites `main`. (If you need true versioning/pinning, use real `huggingface.co` for that repo instead.) The API reports a synthetic `sha` (sha1 over the sorted file manifest) so HF clients have a stable commit id to cache against, but it always points at the one stored snapshot.

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
| `runpod` | RunPod API key | Serverless (single **+ multi-model fleet**, reverse-tunnelled), Compute, Benchmark |
| `pi` | Prime Intellect API key | Serverless, Compute, Benchmark |
| `vm` | Your own SSH-reachable box | Serverless (multi-model fleet), Benchmark (bare-metal) |
| `fake` | In-process dev provider | Local development only |

Credentials are encrypted at rest. Provider resolution per request: explicit `provider_id` → the user's sole provider of that kind → gateway-wide env fallback (`RUNPOD_API_KEY`, `PI_API_KEY`). `GET /v1/providers` also returns each cloud provider's **available GPU catalog** (`available_gpus` — id / label / VRAM), so a client can discover where it can run; `vm` providers report their fixed physical GPUs instead.

## Storage

Where artifacts and datasets live. Register a backend at `/storage` or `POST /v1/storage`:

| Kind | What it holds | Used by |
|---|---|---|
| `s3` | bucket + region + (optional) endpoint + access key — Benchmark `result.json`, trained models, dataset metadata + audio | Benchmark, Autotrain, Datasets |
| `huggingface` | an HF token (or a reference to a [global secret](#secrets-global-env--tracking)) | Datasets (`hf` source / sync), Autotrain (model push) |

Credentials are Fernet-encrypted at rest and never returned by the API; `POST /v1/storage/test` validates connectivity before you save. Reads are org-wide; writes are admin-only.

## Secrets (global env + tracking)

Org-wide secrets managed by admins at `/admin/secrets`:

- **Global env** (`/v1/global-env`) — key/value pairs merged into every workload's environment (benchmark pods, serverless workers, training runs). Values are encrypted; ones flagged secret are masked in API responses. A storage/dataset can reference one by name (e.g. an HF token) instead of inlining it.
- **Tracking credentials** (`/v1/tracking-credentials`) — named **Weights & Biases** / **MLflow** credentials an Autotrain run points at to stream metrics; the runner decrypts the chosen one and injects the tracker's env vars.

## API keys & the HTTP API

Mint a long-lived bearer token on the **API tokens** page (or `POST /api-keys`). It's shown once, hashed at rest, prefix `sgpu_`. Send it as `Authorization: Bearer sgpu_...`. One key works across every surface:

| Area | Key endpoints |
|---|---|
| Serverless | `POST /apps` · `GET /apps` · `GET /apps/{id}/status` · `GET /apps/{id}/workers` · `POST /apps/{id}/workers/{mid}/terminate` · `POST /apps/{id}/queue/flush` · `POST /v1/chat/completions` · `POST /v1/audio/transcriptions` · `GET /v1/models` · `POST /run/{app}` · `GET /result/{id}` · `GET /{app}/metrics` (per-endpoint Prometheus) |
| Autotrain | `POST /v1/training-runs` · `GET /v1/training-runs` · `GET /v1/training-runs/{id}` · `GET /v1/training-runs/{id}/metrics` · `GET /v1/training-runs/{id}/files` · `GET /v1/training-runs/{id}/logs/stream` · `POST /v1/training-runs/{id}/restart` · `POST /v1/training-runs/{id}/terminate` · `POST /v1/training-runs/{id}/transcribe` · `POST /v1/training-runs/{id}/synthesize` |
| Datasets | `GET /v1/datasets` · `POST /v1/datasets` · `POST /v1/datasets/{id}/upload` · `GET /v1/datasets/{id}/preview` · `POST /v1/datasets/{id}/row-inclusion` · `POST /v1/datasets/{id}/transform` · `POST /v1/datasets/{id}/pack-tts` · `POST /v1/datasets/{id}/sync` |
| Benchmark | `POST /benchmarks` · `GET /benchmarks/{id}` · `GET /benchmarks/{id}/logs?tail=` · `POST /benchmarks/{id}/duplicate` · `PATCH /benchmarks/{id}` (rename) · `POST /benchmarks/{id}/terminate` |
| Compute | `POST /compute` · `GET /compute` · `GET /compute/{id}/ssh` |
| Providers / storage / secrets | `GET /v1/providers` · `GET /v1/storage` · `GET /v1/global-env` · `GET /v1/tracking-credentials` |
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
- **Secrets** at `/admin/secrets` — org-wide global env vars + W&B / MLflow [tracking credentials](#secrets-global-env--tracking).
- **Disable a surface** — set `DISABLED_SECTIONS` (comma-separated: `inference,benchmark,compute,datasets`) on the gateway **and** web; the section drops out of the sidebar, its pages 404, and the gateway 403s its routes.

## Repo layout

- **Gateway** (`gateway/`) — FastAPI control plane: `/apps`, `/run`, `/stream`, `/v1/*`, `/workers`, `/compute`, `/benchmarks`, `/v1/training-runs`, `/v1/datasets`, `/v1/providers`, `/v1/storage`, `/v1/global-env`, `/v1/tracking-credentials`, `/api-keys`, `/auth`, `/admin/*`, `/metrics`
- **Trainers** (`gateway/gateway/training/`) — standalone runner scripts shipped over SSH per run: `whisper_finetune.py` (ASR), `tts_finetune.py` (the TTS pipeline — NeuCodec-encode → multipack into a ChiniDataset → Qwen3 causal-LM train → CER/MOS/similarity eval, with a **pack-only** mode that powers the dataset Pack-for-TTS step), and `sweep_runner.py` (GPU-pinned multi-trial orchestrator)
- **Worker agent** (`worker-agent/`) — `BRPOP`s jobs from Redis, runs vLLM, publishes streaming tokens via pub/sub (and rebuilds the multipart upload for Whisper `/v1/audio/*`); also drives the **multi-model fleet** on a VM **or a reverse-tunnelled RunPod pod** — wave-loads each vLLM, sleeps/wakes them per request, skips cancelled jobs before waking a model, runs operator commands (sleep/restart/kill), and ships per-model + scheduler logs to the gateway. The RunPod reverse-tunnel provisioning lives in `gateway/gateway/runpod_provider.py`.
- **SDK + CLI** (`sdk/serverlessgpu/`) — `@endpoint` decorator, `QueueDepthAutoscaler`, and the `serverlessgpu` CLI
- **Web** (`web/`) — Next.js UI: Serverless, Autotrain, Datasets, Benchmark, Compute, Providers, Storage, API tokens, Admin (users / roles / audit / approvals / provisioned / secrets)
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
