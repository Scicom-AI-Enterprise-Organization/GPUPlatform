# Claude project guide — Serverless-GPU

## Local dev is set up. Use it. Don't push to prod just to test.

The full stack runs on the user's laptop. **Default to iterating locally** before
suggesting any deploy/PR. Only push when the user explicitly asks, or when the
change genuinely needs to run in-cluster (e.g. SealedSecrets, ArgoCD wiring).

### What's already running / configured

- **Postgres + Redis** via `docker compose up -d postgres redis` from repo root.
  The user's compose stack stays up between sessions — assume both are healthy
  unless `docker compose ps` says otherwise.
- **`gateway/.env`** — localhost db/redis, `AUTH_DISABLED=1` (login = `admin`/`admin`),
  `AUTOSCALER=1`, `PROVIDER=runpod`, real `RUNPOD_API_KEY` + `RUNPOD_TEMPLATE_ID=gneokrqwe9`.
  ⚠️ Real RunPod billing is live — pods spawned locally cost money.
- **`web/.env.local`** — `NEXT_PUBLIC_GATEWAY_URL=http://localhost:8080` + `RUNPOD_API_KEY`
  for the WorkersTab.

### How to run things (the user already knows; reproduce if asked)

```bash
docker compose up -d postgres redis      # db (idempotent)
.venv/bin/gateway                        # backend, port 8080
cd web && npm run dev                    # frontend, port 3000
```

Python deps: **always `uv`**, never `pip`. New venv: `uv venv .venv && uv pip install -e ./gateway`.

### The localhost ↔ RunPod gotcha (don't forget this)

`PROVIDER=runpod` from a local gateway *does* successfully POST to RunPod's API and
spawn a real pod — the user has confirmed this works. **But** the spawned pod tries
to register at `GATEWAY_PUBLIC_URL` (currently `http://localhost:8080`), which from
RunPod's network points at the pod itself, not the user's laptop. So:

- ✅ Pod appears in RunPod dashboard, billing starts
- ❌ Pod never registers, never serves requests, UI never sees a worker
- 🔥 If the user forgets to terminate, the pod bills indefinitely

When the user reports "no workers showing up" with `PROVIDER=runpod` locally, the
answer is **always** this reachability issue, not a config bug. Suggest one of:
- Switch back to `PROVIDER=fake` for end-to-end UI testing
- Point `web/.env.local` at the prod gateway for real-worker testing
- Tunnel gateway + redis publicly (cloudflared) — only if they explicitly ask

The user has been told this multiple times and may push back. Per
`feedback_just_do_it.md`: don't re-litigate. State the constraint once, do what
they ask, move on.

### Serving Whisper / audio (ASR) on the serverless fleet

The fleet serves **Whisper** via the OpenAI-compatible **`/v1/audio/transcriptions`** and
**`/v1/audio/translations`** (global + scoped `/{app_id}/v1/audio/...`). The job queue is
JSON-only, so the gateway base64s the uploaded clip into the payload and the **worker rebuilds
the multipart** request for vLLM (`worker-agent/worker_agent/main.py` `handle()`). Use it from an
endpoint's **Playground → mode: Audio transcription** (a Chat/Embedding/Audio toggle on the Playground tab —
not a separate tab; its model dropdown lists *every* member so any name works, and it keeps its own
per-browser transcription history alongside the chat history), or:
`curl -F file=@clip.mp3 -F model=<model> -H "Authorization: Bearer sgpu_…" $GW/<app_id>/v1/audio/transcriptions`

Three things silently break audio if missing — check these first when transcription fails:
- **Gateway venv needs `python-multipart`** (now in `gateway/pyproject.toml`). Without it FastAPI
  refuses to boot once the `Form`/`File` routes exist — the gateway crashes on start.
- **vLLM venv needs `librosa soundfile resampy av`** (the `vllm[audio]` set — all pip, **NO system
  ffmpeg**; vLLM decodes via soundfile + PyAV). `resampy` is the sneaky one: without it any
  non-16 kHz clip fails as `"Invalid or unsupported audio file"` while a 16 kHz WAV slips through
  (looks like an mp3/codec bug — it isn't). `launcher.ensure_audio_deps()` auto-installs these for
  any audio member on (re)provision.
- **Tag ASR models** with the per-member **"Audio / ASR (Whisper)" checkbox** (create form +
  Overview → Models → Edit) → stores `task: "transcription"`. Required for ASR finetunes whose name
  doesn't contain "whisper" (a name heuristic auto-covers `whisper-*`); the worker installs audio
  deps when `task=="transcription"` OR the name matches.

To add Whisper to an existing fleet: **Overview → Models → Edit → add the member (TP=1), check
Audio, Save** — that re-provisions and re-ships the worker-agent + ensures the audio deps. (A
member needs a GPU slot; small fleets time-share via sleep/wake. Sleep-mode works fine with Whisper.)

### VM endpoints self-bootstrap their vLLM venv (`venv_path` no longer needs to pre-exist)

A multi-model VM/RunPod worker now **builds its own vLLM venv on boot** — name a `venv_path` that
doesn't exist and the worker installs `uv` (if absent), `uv venv`s the path (mkdir parents), and
installs vLLM, *streaming* the install to the `__worker__` log. Logic in
`worker-agent/.../multi/launcher.py` (`ensure_venv`/`ensure_build_tools`/`run_pre_script`) +
`scheduler.start()`. Knobs (create form **and** Overview → Models → Edit; stored on `App`):
- **vLLM version** (`vllm_version`) — `uv pip install vllm==X --torch-backend=auto`.
- **vLLM install args** (`vllm_install_args`) — a full `uv pip install` arg string used *verbatim*
  (overrides the version), for nightlies/custom CUDA. The form has an **Insert nightly (cu130)** button.
- **Pre-launch script** (`pre_script`) — shell run once after the venv is ready, before launch, with
  `{venv}/bin` on PATH + VIRTUAL_ENV set. The form has a **+ Install DeepGEMM** button. Runs under
  `bash` so `bash <(curl … install_deepgemm.sh)` works.

A `{venv}/.sgpu_vllm_spec` marker means: skip if the spec is unchanged, reinstall if you change it,
**never touch a venv with no marker** (hand-built ones like the tm fleet's `/share/vllm-venv`).

Two Blackwell (B300, sm_103) gotchas, both handled: flashinfer JIT-builds kernels at runtime so the
venv needs `ninja`+`cmake` AND `{venv}/bin` on PATH (`launch_member` prepends it) — else
`FileNotFoundError: ninja`. And vLLM **0.23.0** ships a `prometheus_fastapi_instrumentator` that 500s
*every* route incl. `/health` (`'_IncludedRouter' object has no attribute 'path'`) → worker never goes
healthy; workaround is a `pre_script` `uv pip install -U "prometheus-fastapi-instrumentator>=7"`. After
repeated re-provisions, free orphaned GPU memory with `POST /apps/{id}/workers/purge` (a self-exited
vLLM the per-PID cleanup misses → "No available memory for the cache blocks" on the next launch).

### Activity dashboard (`/activity`) + proxy-mode usage recording

The **Activity** page (`web/.../activity-dashboard.tsx` → `GET /v1/history/activity` in
`history_api.py`) is unified usage analytics: requests, token in/out, TTFT/latency, top
users/models. It aggregates **two tables only** — `requests` (serverless queue) +
`proxy_requests` (the separate LLM-proxy feature). So *anything that doesn't write one of
those rows is invisible to Activity.*

⚠ **Single-model VM endpoints (`mode=proxy`) bypass the queue/worker** — the gateway
HTTP-forwards straight to the VM's vLLM (`main._proxy_to_upstream` / `_proxy_audio_to_upstream`),
and that worker is what normally writes the `requests` row. So proxy traffic used to be recorded
**nowhere** → missing from Activity (and request history). Fixed: the proxy path now records each
request into the `requests` table itself (it's a serverless `App`, so `requests` is the right home —
NOT `proxy_requests`, which belongs to the LLM-proxy and is pruned by *its* health loop). Details:
- **`main._record_proxy_request`** writes a **slim** row — `payload={"model":…}` + `output={"usage":…}`
  + `ttft_ms` + created/completed — exactly the fields the aggregator reads (`payload.model`,
  `output.usage.{prompt,completion}_tokens`, `ttft_ms`, latency = completed−created). Slim on purpose:
  proxy is the high-throughput path (synthetic-data gen). Streams get `stream_options.include_usage`
  injected + the SSE chunks sniffed for the final usage block; TTFT is the first-chunk time.
- **Non-blocking, off the DB hot path**: it ENQUEUEs to the background **`stats_writer`**
  (`record_serverless_request`, a new INSERT path — the writer previously only did UPDATEs on
  existing rows) rather than opening a pooled session per request. One writer connection batches the
  whole burst; a per-request checkout here is exactly the pool-exhaustion incident `stats_writer` was
  built to avoid. Load-verified locally: full proxy path through `_proxy_to_upstream` + a mock vLLM
  sustained **~1200 RPS non-stream / ~514 RPS stream at 100 concurrency with every request recorded**
  (writer hot-path enqueue ~10µs, peak 1 concurrent DB backend). Writer sustained ceiling ≈ **925
  rows/s** on defaults (`STATS_FLUSH_MAX_BATCH=500`, shared across all stat sources); above that the
  20k queue buffers then drops-and-logs (best-effort) — raise the batch env if real load nears it.

### Label platform (data-labelling app)

A separate Next.js app (source: `/home/husein/ssd3/Label`, dev host `http://localhost:3002`)
the gateway talks to for human labelling — both **read** (a `kind=label` Dataset imports
labelled rows) and **write** (autotrain TTS auto-creates a recording+MOS project after a run).
Auth is a `lpat_…` PAT carrying its owner's role (create-project needs an admin PAT). Full API
reference + the audio-filename↔storage prefix gotcha: **`docs/LABEL_PLATFORM.md`**. Gateway
integration lives in `datasets_api.py` (read) and `training_api.py` `_create_label_project_for_run`
+ `training/tts/tts_label_export.py` (write, VM-only — synthesis needs the box).

### Self-hosted HuggingFace catalog (the "Models" section)

Users host their own models/datasets on Storage backends and use standard HF tooling against
the gateway: a Hub-compatible mirror at **`/hf`** (`hf_mirror_api.py`) + a management API at
**`/v1/catalog`** (`catalog_api.py`) + the web **Models** section (`/models`, detail at
`/models/{ns}/{name}`). `export HF_ENDPOINT=<gw>/hf` + `HF_TOKEN=sgpu_…`, then `snapshot_download` /
`from_pretrained` / `load_dataset` / `push_to_hub` just work.

**Revisions (added 2026-06-14).** A repo **created by pushing through the mirror** is *versioned*
(`CatalogRepo.versioned=True`): named, **overwriteable** branches (push to `main` / `checkpoint-v1`
— each independent, NOT immutable commit history), content-addressed blobs at `{prefix}/blobs/{oid}`,
extra branches in the `CatalogRevision` table (`main` stays denormalized on `CatalogRepo.manifest`).
Resolve a revision by branch name OR commit sha; `list_repo_refs`/`create_branch`/`delete_branch` work.
A repo **registered over existing data** (`/v1/catalog` or **Publish dataset** — prefix is the real
S3 layout) stays *flat* (`versioned=False`): single `main`, path-addressed `{prefix}/{path}`, any
`revision` resolves to `main`. The `versioned` flag branches every read/write path in
`hf_mirror_api.py` (`_resolve_revision`, `_blob_key`, `_commit_impl`). Verified via the real `hf`
client (huggingface_hub 1.17.0). Blob GC (orphans from overwrites/deletes) is NOT implemented yet.
⚠️ Still **different from a `kind=hf` Dataset's `hf_revision`**, which pins a real commit/branch/tag
on `huggingface.co` — don't conflate.

### Testing the gateway locally (current `.env` reality)

`gateway/.env` is currently `AUTH_DISABLED=0` + `GATEWAY_RELOAD=0` (despite older notes saying
auth-disabled): backend edits need a **manual gateway restart**, and API calls need **real auth**.
For testing, send an **API key** as `Authorization: Bearer sgpu_…` — do **not** write Redis
`session:<token>` keys to forge a session (that's exactly the prod-Redis-exposure risk; the user
flagged it). No active training run? a gateway restart is safe — runs detach and finalize from log.

### What NOT to do

- Don't suggest `docker compose up gateway` to test backend changes — the compose
  gateway runs the *image*, not their working tree. They want hot reload.
- Don't suggest deploying a branch to prod just to verify a fix. Reproduce locally first.
- Don't run `.venv/bin/gateway` yourself unless asked — the user typically has it
  running in a terminal already. Editing code triggers no auto-reload (uvicorn isn't
  in `--reload` mode), so just tell them to restart it.
- Don't `pip install` anything — use `uv pip install`.
