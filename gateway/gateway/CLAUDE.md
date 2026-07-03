# Claude guide — gateway internals (`gateway/gateway/`)

Area-specific gotchas for the FastAPI control plane. This file **loads automatically when you edit
files here** (lazy — it costs nothing in sessions that don't touch the gateway). Cross-cutting dev
setup, the run commands, the localhost↔RunPod reachability gotcha, and the `.env`/auth/reload reality
live in the **repo-root `CLAUDE.md`** (always loaded).

### VM reverse tunnel — autossh `ssh -R`, keyed by (host, **port**)

A VM worker phones home (register/heartbeat/Redis) over a **reverse SSH tunnel** the gateway opens
(`vm_tunnel.ensure`), needed whenever the gateway isn't publicly reachable (local dev — set
`VM_REVERSE_TUNNEL=1`; the worker then gets `GATEWAY_URL=http://127.0.0.1:{gw_port}` which routes back
through the tunnel). In **prod** the gateway is reachable so the worker connects directly — that's why
"works in prod, not localhost". The reverse tunnel now uses **autossh `ssh -R`** (native OpenSSH, same
as the forward `ensure_forward`) — NOT the old in-process paramiko `request_port_forward`, whose
`_monitor`/`_healthy` close+reconnect loop raced the VM's port release → endless **`TCP forwarding
request denied`** flapping (tunnel up→denied→up, worker's 30 register attempts all miss). A bare
`ssh -R` never flaps; autossh keeps it alive + reconnects.

⚠ **Keyed by `(host, port)`, not host.** Two providers can share one host on different SSH ports —
e.g. the tm box runs **two containers**: `tm`=`8.222.165.68:1024` (prov-5be27d21) and
`tm-2`=`8.222.165.68:1023` (prov-32bb483b). The old `_TUNNELS[host]`/`_REV_PROCS[host]` keying made the
second provider **reuse the first's tunnel** (bound in the wrong container) → its workers' registration
silently failed. `_REV_PROCS`/`_kill_stale_reverse` are now `(host,port)`-scoped so each container gets
its own `-R`. autossh subprocesses are detached (`start_new_session`) so they survive a gateway restart;
`_kill_stale_reverse(host, port, …)` reaps the prior process's `-R` (matched by keyfile + `-p {port}` so
it never kills a sibling provider's tunnel). Verified e2e on tm: both `-p 1023` and `-p 1024` reverse
tunnels coexist, the `:1024` worker registers + serves.

### Benchmarks (benchmaq) + the provider metrics page

The **Benchmark** section (`web/.../benchmark/new/benchmark-form.tsx` → `POST /v1/benchmarks` in
`bench.py`) drives the external **benchmaq** tool (installed in the gateway venv from
`git+…/llm-benchmaq`) to spin up a target and run vLLM/SGLang throughput + accuracy benches. Two backends:
- **RunPod (cloud)** — `benchmaq runpod bench`: benchmaq deploys a pod, SSHes in, `uv pip install`s
  `remote.dependencies`, serves + benches. The pod install runs via **pyremote** (`@remote`).
  ⚠ **Needs the `runpodctl` binary on the gateway's PATH** — benchmaq shells out to it to poll pod
  readiness (`No such file or directory: 'runpodctl'` → the pod spawns but the run hangs, billing the
  whole time). It's a standalone Go CLI, **not** a pip dep, so it isn't in `pyproject.toml`; install
  separately: `brew install runpod/runpodctl/runpodctl`, or drop the `runpodctl-darwin-arm64` release
  binary into `.venv/bin/` (which run_benchmark prepends to the subprocess PATH — no gateway restart).
  Also: local dev has **`RUNPOD_API_KEY` commented out** in `gateway/.env` — pass a **runpod-kind
  provider_id** so creds resolve from `providers.config` (`env["RUNPOD_API_KEY"]` for the subprocess),
  or uncomment the env key + restart.
- **VM (bare-metal)** — `remote.backend: ssh` via the gateway's **`pyremote_shim`** (reconnect-per-command
  paramiko; TM's SSH proxy allows only one exec channel per TCP connection, hence the shim).

**⚠ `HOME=/share/home` breaks RunPod SSH (bit us in prod).** A RunPod pod's boot script installs the
injected key with `echo $PUBLIC_KEY >> ~/.ssh/authorized_keys`, so a `HOME` override lands the key in
`/share/home/.ssh` while sshd reads `/root/.ssh` → every auth fails, pod stays "SSH not ready" to the
ceiling, never runs, bills the whole time. Log tell: `grep: /share/home/.bashrc: No such file` during
"Exporting environment variables". Fix: `bench.py` `_resolve_config` **strips `HOME` from the RunPod
pod-boot env** (keeps it for the VM path, whose sshd is already set up; cache vars like
`XDG_CACHE_HOME`/`HF_HOME` stay — read at runtime, not by the boot script).

**vLLM ≥ 0.23 needs a CUDA-13 image.** vllm 0.23.0 pulls `torch==2.11` built for cu130 → needs a ≥580
driver. A cu1281 image lands the pod on a 12.8-driver host → `NVIDIA driver too old (found 12080)` →
EngineCore crash. Use **`runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404`** (`compute._extract_cuda_version`
→ `allowedCudaVersions=["13.0"]` → ≥580 host). The benchmark form **defaults to cu1300 + vllm 0.23.0**.

**Custom-fork vLLM on benchmaq** (mirrors the endpoint path). Form: "Custom fork / install args" + a
one-click **Gemma-4 FA4** preset → renders `remote.uv.vllm_install_args` (e.g.
`VLLM_USE_PRECOMPILED=1 git+…@ref --torch-backend=auto`). Backends consume it differently:
- **VM** — `pyremote_shim` splits the leading `NAME=VALUE` env tokens off and emits them as a shell
  **prefix** on `uv pip install -U …` (so `VLLM_USE_PRECOMPILED=1` is install env, not a bogus pip arg).
- **RunPod** — `_resolve_config` translates it into `remote.dependencies` (git spec + flags); pyremote's
  install runs in a non-login `bash -c` SSH session that does NOT inherit pod `--env`, so the gateway
  exports the leading env via **`SGPU_PIP_ENV`** (read by a patched pyremote `_install_dependencies`) —
  else the fork silently builds from source (~25 min) and times out.
- Both add **`sentencepiece`** (the fork's precompiled wheel skips it; gemma/llama tokenizers need it,
  else `Couldn't instantiate the backend tokenizer`).

**Serve from an existing HF cache:** omit `model.local_dir` → benchmaq skips the download and runs
`vllm serve <repo_id>` against `HF_HOME` (it downloads only when BOTH `repo_id` AND `local_dir` are set).
This also dodges the VM `/workspace/…`→`~/…` `local_dir` rewrite, which `vllm serve` can't expand (`~`
stays literal → "Invalid repository ID or local directory").

**Crash-abort (RunPod):** `bench.py` `_drain` watches the streamed log for terminal vLLM init failures
(`EngineCore failed to start`, `driver too old`, …) and tears the pod down immediately instead of
polling a dead `/health` to the ceiling.

**⚠ Ephemeral site-packages patches (NOT in the repo — `uv pip install` wipes them; make durable at
gateway startup):** pyremote `_install_dependencies` (the `SGPU_PIP_ENV` prefix — needed for RunPod
forks) and benchmaq `_wait_for_ssh` `timeout=600→1200` (large cu130 images cold-pull past 10 min). Also:
benchmaq's metrics-output `@@`-section splitter in `vm_probe.py` has a **fixed marker whitelist** — any
new `@@SECTION` must be added there or it's silently swallowed.

**Provider metrics page** (`/providers/{id}/metrics` → `providers_api.provider_metrics` → `vm_probe.py`;
VM providers only): live CPU/mem/GPU + per-GPU process list. GPU procs come from **two sources merged**:
- **NVML** (`nvidia-smi --query-compute-apps`, same as nvtop) → per-GPU VRAM, every owner — but on a
  **container** (TM is a PAI-DSW container) it reports **host-namespace pids** whose command can't be
  resolved from inside (`/proc/<hostpid>` doesn't exist → shown as "foreign pid · command not visible").
- **`/proc` cmdline scan** (world-readable) → real commands + **container pids** (killable), catching GPU
  frameworks (vLLM/sglang) whose `/proc/<pid>/fd` is unreadable across the namespace. Device-holders
  attach to the GPUs whose `/dev/nvidiaN` they hold; fd-unreadable framework servers attach to the
  heavy-VRAM GPUs (best-effort — there's no host↔container pid bridge inside the container). nvidia-smi's
  *human* process table is empty in a container, but the `--query-compute-apps` query interface isn't.

Benchmark results show an **"individual TPS"** KPI = output tok/s ÷ concurrency (per-stream decode rate;
`perStreamOutputTps` in `web/src/lib/bench-results.ts`, surfaced in `benchmark/[id]/tabs/results.tsx`).

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
+ `training/tts/tts_label_export.py` (write, VM-only — synthesis needs the box). A `kind=label`
Dataset's import is filtered by `label_status` (review status) and `label_updated_until` (an
optional ISO-8601 point-in-time cutoff → the export's `updated_until`; only tasks last updated
at/before it are pulled). Both are forwarded on every read (`_label_export_rows`, `_label_pairs`)
and editable post-creation via PATCH — changing either re-counts the dataset's rows.

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
