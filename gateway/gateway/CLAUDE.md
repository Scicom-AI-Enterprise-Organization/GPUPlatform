# Claude guide — gateway internals (`gateway/gateway/`)

Area-specific gotchas for the FastAPI control plane. This file **loads automatically when you edit
files here** (lazy — it costs nothing in sessions that don't touch the gateway). Cross-cutting dev
setup, the run commands, the localhost↔RunPod reachability gotcha, and the `.env`/auth/reload reality
live in the **repo-root `CLAUDE.md`** (always loaded).

### Production-hardening conventions (added 2026-07-18 — keep these invariants)

- **Global exception handler** (`main._unhandled_exception_handler`): every unhandled exception
  returns the `{"error": {message, type: "internal_error", request_id}}` envelope + `X-Request-ID`
  header and bumps `gateway_unhandled_exceptions_total{route}`. Deliberate errors stay
  `HTTPException` (handled earlier in the stack, never counted here).
- **Request-id log correlation**: `metrics_mw` sets `accesslog.request_id_var` (a ContextVar) per
  request; `accesslog.init_root_logging()` (idempotent, called from BOTH `run()` and lifespan so
  external ASGI servers get it too) installs a filter rendering ` [req-…]` into every module log
  line via `%(request_id)s` in the root format. Don't `logging.basicConfig` anywhere else.
- **`/ready` = Redis AND Postgres** (each behind a 2s timeout); `/health` stays dependency-free
  liveness. Don't add dependency checks to `/health` — k8s would restart a healthy process over a
  dependency outage.
- **Loop heartbeats**: every background loop stamps `metrics.loop_heartbeat("<name>")` at the END
  of a *successful* tick (never in an except branch). Current names: autoscaler, reconciler,
  vm_watchdog, proxy_health, stats_writer, log_archive, leader (HA only). Alert =
  `time() - max by (loop)(gateway_loop_last_tick_timestamp_seconds) > 600`. **Add the call when
  you add a loop** — an unstamped loop looks permanently stalled once someone alerts on it.
- **/metrics never 500s**: `metrics.render()` degrades on Redis failure (`gateway_redis_up 0`) and
  samples DB-pool (`db.pool_status()`) + stats-writer gauges best-effort. Redis sampling is
  **pipelined** — keep it that way (the old per-app awaits were 2+2·apps+workers round-trips/scrape).
- **Alert rules live in TWO synced places**: `deploy/monitoring/prometheus/alerts.yml`
  (docker stack, promtool-validated) and `deploy/helm/serverlessgpu/templates/prometheusrule.yaml`
  (operator clusters, prom template vars escaped with `{{`…`}}`). Change one → change both.
  Alertmanager (local stack) is `deploy/monitoring/alertmanager/alertmanager.yml` — ships with a
  no-op default receiver; Slack/Telegram/webhook examples are in the header comment.
- **Shared retry helper**: `retry.py` (`retry_async`/`retry_sync` — expo backoff + jitter, logs
  each retry, never swallows `CancelledError`). Use it for new outbound calls instead of another
  inline loop.
- **Opt-in guards**: `MAX_REQUEST_BODY_MB` (Content-Length 413 check in `metrics_mw`, 0=off —
  dataset uploads are multi-GB, only set where ingress doesn't enforce one);
  `PROXY_HTTP_READ_TIMEOUT_S` (read ceiling for the shared proxy httpx client, unset=unbounded
  because per-call sites override with the endpoint's own `timeout_s`).
- **Unit tests**: `gateway/tests/unit/` (pure in-process, no stack — auth/netsafe/pathsafe/crypto/
  metrics/retry/accesslog/stats-writer/exception-handler). Test deps: `[project.optional-dependencies].dev`.
  When you harden something here, add its unit test there.
- **Legacy `GET /v1/training-runs` slims `result_json` to `{"best": …}`** (like `/_page`) — the
  full record is `GET /{run_id}`. Don't fatten the list responses back up; it was the slowest
  control-plane endpoint (146ms p50 at 272 runs) before slimming. See `docs/API_LATENCY_REPORT.md`.

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

**Huawei Ascend NPU + jump host + password auth** (added 2026-07-07): VM providers support an
optional **ProxyJump** (paramiko direct-tcpip through a jump SSHClient) and **password auth** per
hop — `VmConfig.{password,jump_host,jump_port,jump_user,jump_private_key,jump_password}`, secrets
Fernet-encrypted (`*_enc`), resolved by `providers_api._vm_conn_from_cfg`. All of vm_probe
(probe/availability/metrics/bandwidth/kill) falls back to **`npu-smi info`** when nvidia-smi is
absent: `_parse_npu_info` parses the Ascend table (paired rows: id+name/health/power/temp, then
chip/bus-id/AICore%/DDR/HBM — mem = last used/total pair with non-zero total) into `GpuMetric`
`kind="npu"` (+`power_w`/`health`; util=AICore%, mem=HBM), and the NPU process table seeds per-NPU
procs with commands from `/proc` (`@@NPUPROCCMD`). ⚠ In NPU mode the `@@FDPROC` merge is **skipped**:
Ascend procs hold `/dev/davinci_manager`, never `/dev/davinciN`, so fd→device mapping is impossible
and the heavy-fallback would attach every proc to every NPU; npu-smi's table is complete (bare
metal). Verified e2e on the TM box (8× 910B3 via ssh.tma01.gpu.tm.com.my). Metrics UI renders
"NPUs" cards with AICore%/HBM/power/health.

**Serverless through a jump host** (added same day, verified e2e with endpoint `npu-qwen3`):
`VMProvider` + `vm_tunnel` are jump-aware — provisioning SSH goes through
`vm_probe._connect` (jump = paramiko direct-tcpip), and the autossh reverse/forward tunnels
add `-o ProxyCommand=ssh -i <jumpkey> -W %h:%p …` (`vm_tunnel.Jump`; ProxyJump can't take a
per-hop `-i`). ⚠ Tunnels run OpenSSH **BatchMode → key auth required on BOTH hops** —
password-only VM providers get a clear RuntimeError from `_require_key`/`_tunnel_jump`.
`resolve_app_provider` decrypts the full conn (incl. jump) via `providers_api._vm_conn_from_cfg`
(lazy import — circular otherwise). The worker venv (`~/.sgpu/venv`) is created with
`uv venv --python 3.11` (worker-agent needs ≥3.10; the TM NPU box ships 3.9). Training paths
still assume direct key SSH. Ascend serving specifics live in **worker-agent CLAUDE.md**.

Benchmark results show an **"individual TPS"** KPI = output tok/s ÷ concurrency (per-stream decode rate;
`perStreamOutputTps` in `web/src/lib/bench-results.ts`, surfaced in `benchmark/[id]/tabs/results.tsx`).

**Manual GPU identity (`benchmarks.gpu_type`/`gpu_count` columns, added 2026-07-07):** ingress/Slurm
runs (no pod, no provider) have no derivable GPU, so external consumers (the GPU calculator) couldn't
group them. Set it via `CreateBenchmarkRequest.gpu_type`, a top-level or per-benchmark-item `gpu_type:`
key in the YAML, or post-hoc with `PATCH /benchmarks/{id}` (`UpdateBenchmarkRequest` — the old rename
endpoint, `""`/`0` clears; UI = pencil on the Parameters tab's GPU row, plus a dedicated "Hardware"
card for ingress runs). The row value wins over config in `_bench_gpu_meta`; `BenchmarkRecord` and
`public-compare` now carry resolved top-level `gpu_type`/`gpu_count` so nobody has to parse
`config_yaml`.

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

**⚠ Proxy-mode admission control (added 2026-07-15) — bound concurrency, don't 500 the replica.**
A proxy endpoint forwards to ONE vLLM replica; a bursty client (e.g. ~32 concurrent ~100s GLM-5.2
reasoning requests on the TP8 `glm5-2`) overruns vLLM's `max_num_seqs`/KV cache → latency balloons +
a fraction come back as **bare HTTP 500s** (aborted requests). `_proxy_to_upstream` /
`_proxy_audio_to_upstream` now run a per-app **`_ProxyGate`** (`app.state.proxy_gates`, mirrors
`proxy_api._get_sem`): up to `limit` requests forwarded at once, extras **wait** for a slot (bounded
"queue"), and once `queue_max` are already waiting we **shed with 429 + `Retry-After`** instead of
deepening the backlog. Config via `_proxy_concurrency_config`: per-app override in the App's
**`autoscaler` JSON** (`proxy_max_concurrency` / `proxy_queue_max`), else env
**`PROXY_MAX_CONCURRENCY` / `PROXY_QUEUE_MAX`**; **0 = unbounded (the old forward-everything default,
non-breaking)** — set a ceiling on an endpoint that's getting overrun. Two more fixes ride along:
(1) **upstream error bodies are surfaced top-level as `{"error":{…}}`** (was `HTTPException(detail=…)`
→ rendered `{"detail":{…}}`, so OpenAI SDK clients read nothing and logged only the exception class);
(2) every proxy response carries **`X-SGPU-Inflight` / `X-SGPU-Concurrency-Limit`** so clients can
self-throttle. `PROXY_RETRY_AFTER_S` (default 5) tunes the Retry-After hint. Set the ceiling ≈ the
model's vLLM `max_num_seqs` (or just below). NOTE this bounds a single replica (see the multi-replica
section below for the cluster-wide cap on the *separate* LLM-proxy feature).

### Multi-replica HA — leader election (`leader.py`) + proxy cluster (`proxy_cluster.py`)

The gateway is a monolith: it serves a **stateless data plane** (the LLM proxy in `proxy_api.py`, API
reads) but also runs **singleton controllers** (autoscaler/reconciler/vm_watchdog/janitors/gitops/
log-archive) that are NOT safe to run on >1 replica (two autoscalers double-provision). So HA splits by
leadership rather than by deployment:

- **`GATEWAY_HA=1`** (auto-on in the Helm chart when `replicaCount > 1`, or `gateway.ha: true`) → every
  replica serves HTTP, but only the **leader** runs the mutating controllers. Leader = a Redis lock
  `gateway:leader` (`SET NX PX`); the holder renews it every `LEADER_RENEW_S` (5s) and on leader death
  the lock lapses after `LEADER_TTL_S` (15s) so a follower's `SET NX` promotes it. `main.leader_workload()`
  is the leader-only task set; `LeaderCoordinator` starts it on acquire / cancels on loss. **`GATEWAY_HA`
  unset (default) → this replica is the sole leader, no Redis lock, byte-identical to the old inline
  startup** (so local dev + single-replica prod are unchanged). Observe via **`GET /leader`**.
  ⚠ The **provider object stays always-on on every replica** (built in lifespan, not the workload) —
  request handlers use `app.state.provider` as the app-create provision fallback; only the *loops* are
  leader-scoped. Validation still fails startup fast on every replica.
- **`PROXY_CLUSTER=1`** makes the LLM proxy correct across replicas (opt-in; off → per-replica in-memory,
  unchanged; **fails OPEN if Redis is down** so a blip never blocks traffic):
  - **Global concurrency cap.** A proxy endpoint's `max_concurrency` becomes a cluster-wide cap (was
    N×per-replica) via a Redis ZSET of leased slots (`proxy:sem:{endpoint_id}`). Acquire/release are on
    the hot path (one Lua call each, client-time-stamped so the script stays a pure write); the
    per-replica **sync loop renews each in-flight slot's lease**, so a crashed replica's slots self-free
    within `PROXY_CLUSTER_SLOT_LEASE_S` (30s) instead of wedging the cap. Over cap → the caller waits in
    the gate's cancel-aware acquire loop (0.1s poll) — same visible-queue semantics, just global. The
    concurrency gate is `_Gate`/`_get_gate` (local `asyncio.Semaphore` vs Redis), which replaced the raw
    `sem` passed through `_unary`/`_stream`/`_forward_passthrough`/`_handle_audio`.
  - **Global queue + cancel.** The **sync loop** (`proxy_cluster_sync_loop`, per-replica, every
    `PROXY_CLUSTER_SYNC_S`=2s) mirrors the local `_live` dict → Redis (`proxy:live:{rid}` hash +
    `proxy:live:ep:{eid}` index, TTL-bounded) — so the admin queue view + inflight/queued counts span all
    replicas, entirely OFF the request hot path. Cancel/flush publish the request id on `proxy:cancel`
    pub/sub; every replica's **`proxy_cancel_subscriber_loop`** sets the local `cancel_ev` if it's the
    holder. Both loops are per-replica (NOT leader-only) and self-disable when off.
- **Verified e2e locally** (2 clustered gateways + a mock upstream): global cap held at 2 across both
  replicas (8 concurrent split 4/4 → mock saw max 2); a queued request on replica A was visible on B and
  a **flush issued on B cancelled it (499)** cross-replica; streaming requests respect the cap + release
  slots; the ZSET lease self-heals a crashed holder. Leader election: A led, B followed, `kill -9` A →
  B promoted in ~TTL. Single-replica (`PROXY_CLUSTER`/`GATEWAY_HA` off) regression-clean (local semaphore,
  no cluster loops). ⚠ The single-upstream **passthrough non-SSE path still can't abort mid-upstream-read**
  (no `_watch_cancel` there — pre-existing); cancel only lands while the request is *queued* (gate loop)
  or *streaming* (relay loop).

### Audio proxy (STT/TTS) — drift metrics, format conversion, CER/WER, sample capture

The LLM-proxy (`proxy_api.py`) fronts audio backends too: `/v1/audio/{transcriptions,translations}`
(whisper STT), `/v1/audio/speech` + `/v1/audio/speaker` (TTS). All route by the `model` alias like
chat, with failover / gate / `X-Upstream-*` headers. Verified e2e against `stt-engine-tm-l40` (vLLM
whisper-large-v3-turbo) + `tts-api-tm-l40` (real model name **`TTS-model`**; `/v1/models` 404s so map
an alias → `TTS-model` by hand).

- **STT drift metric = `proxy_audio_nll` (NLL, not raw logprob).** whisper `verbose_json` returns
  per-segment `avg_logprob` (≤ 0). We record its duration-weighted mean **NEGATED** (`-avg_logprob`,
  a histogram, labels `proxy`/`model`). ⚠️ **Why negated:** `prometheus_client` DROPS a histogram's
  `_sum` sample once it observes a negative value (spec bans `rate()` over a decreasing sum) → the
  windowed-mean drift query would break. NLL ≥ 0 keeps `_sum`. Drift: `rate(proxy_audio_nll_sum[30m]) /
  rate(proxy_audio_nll_count[30m])` — RISE = worse (whisper turbo baselines ~0.19).
- **STT proxy ALWAYS requests `verbose_json` upstream, then converts back** to the caller's
  `response_format` (`_do_unary_multipart` → `_convert_verbose_audio`: json→`{text}`, verbose_json
  passthrough, text/srt/vtt reconstructed from segments via `_segments_to_{srt,vtt}` + `Response(raw_body,
  media_type)` in `_unary`). So the drift signal populates on **all** non-stream traffic, not only when a
  client opts into verbose_json (the playground sends `json`). On a 4xx from the verbose_json attempt it
  retries once in the caller's own format (never breaks a working request).
- **Streaming.** whisper streams (`stream=true` → SSE `transcription.chunk` deltas) via `_stream_audio`;
  TTS streams raw `audio/pcm` chunks via `_stream_speech` (both relay incrementally + tee bytes). ⚠️
  **`verbose_json` + `stream=true` is mutually exclusive** — vLLM returns `400 "verbose_json format
  doesn't support streaming case"` (surfaced, not masked), so **streaming carries NO logprob** (the
  stream frames are content-only). Stream for latency, `verbose_json` for drift — can't have both.
- **TTS output** is 16-bit mono **24 kHz** — `audio/pcm` (headerless) by default, `audio/wav` with
  `response_format=wav` (bogus `0x7FFFFFFF` size header — the playground patches it client-side for
  playback). `_audio_to_wav` wraps PCM→WAV (parses a real RIFF, else assumes 24k/mono/16) for the STT
  round-trip.
- **TTS CER/WER round-trip (`proxy_tts_cer` / `proxy_tts_wer`, labels proxy/model/voice).** A per-proxy
  **`stt_callback`** config (base_url + model + optional key, set on the proxy form) transcribes each
  generated clip back through a whisper STT; `_cer_wer` scores it (jiwer, case/punct-normalized) vs the
  input text. ⚠️ **CJK voices → use CER** (word-based WER ≈ 1.0 with no spaces). ⚠️ The round-trip's own
  STT call sends **`X-SGPU-TTS-Eval: 1`** so a capture-enabled STT proxy doesn't re-capture it. `jiwer`
  is a lazy import (in `pyproject`; missing wheel ⇒ metric silently skipped, gateway still boots).
- **Drift-sample capture** (per-proxy **`capture`** config: `storage_id` + `prefix` + `logprob_threshold`
  (STT) / `cer_threshold` + `wer_threshold` (TTS)). When a request crosses a threshold, save the audio +
  a `.json` sidecar to the storage backend (`storage_backends.resolve_backend(row).put_bytes`, run via
  `asyncio.to_thread`). Key = **`{prefix}{proxy_id}/{YYYY-MM-DD}/{X-Request-ID}.{ext}`** — honours an
  inbound `X-Request-ID` (tracing) else the `pxr-…` id, **sanitized** (`_SAFE_ID_RE`) against path
  traversal. Metric `proxy_capture_total{proxy,kind,result}`.
- **All off-path work goes through ONE bounded background queue + worker pool** (`_submit_bg` →
  `_bg_queue` drained by `_BG_WORKERS` `_bg_worker`s). Replaced the old per-call inflight caps that
  *dropped* work under load — now bursts buffer and run in parallel; only a full queue sheds (→
  `result="skipped"`). Scale with **`PROXY_BG_WORKERS`** (default 8) / **`PROXY_BG_QUEUE_MAX`** (1000);
  watch **`proxy_bg_queue_depth`** to decide when to raise workers. Verified: 12 concurrent TTS → all 12
  scored + captured, 0 dropped (old cap-of-4 would've dropped ~8). Response latency unaffected (measured
  −80 ms proxy-vs-direct — i.e. noise — with evals+capture firing).
- ⚠️ **Multi-replica**: these are per-replica in-process metrics (no cross-replica sync — `proxy_cluster`
  only shares slots/queue, not Prometheus). The ServiceMonitor scrapes `/metrics` per-pod; aggregate at
  query time — `sum(rate(proxy_tts_cer_sum[30m]))/sum(rate(proxy_tts_cer_count[30m]))`. Histograms
  (`_sum`/`_count`) are additive across pods; that's why drift is a histogram, not a gauge. Don't scrape
  `/proxy/{name}/metrics` (render_proxy) through the LB — it's this-replica-only.

**Per-endpoint health probe** (general, not audio-specific): `GET /proxy/{name}/health` (+ `/healthz`
alias), **auth-exempt**, sibling of `/proxy/{name}/metrics`. Reads the liveness `proxy_health_loop`
already tracks (probes each upstream's `/models` every ~20s; `<500` & not 401/403 = alive) — no probe on
the call. LB/k8s semantics: **200** `healthy`/`degraded` (≥1 upstream not-known-dead; unknown counts OK),
**503** `unhealthy` (all known-dead) / `disabled` / `misconfigured`, **404** unknown endpoint. Per-replica
view (right for a per-pod probe). Excluded from the HTTP metrics (`METRICS_IGNORE_PATHS`).

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

### Quantization (llm-compressor) — `quantization_api.py` + `training/quantize.py`

The **Quantization** section (`/quantization` → `POST /v1/quantization-jobs`) compresses an LLM
with llm-compressor (compressed-tensors, loadable by vLLM): pull from HF → quantize on a VM/pod →
model to S3 → optional HF push. It's a deliberate **sibling of autotrain**: `quantization_api.py`
imports `training_api as ta` and reuses its SSH/pod/dataset/creds plumbing (`_provision_pod`,
`_ssh_*`, `_resolve_dataset_spec`, `_hf_token_for_storage`, …) rather than duplicating it — only
the DB row (`QuantizationJob`, registered in `db.init_db` like TrainingRun), the scheme recipes,
and the worker contract live here. Section key `quantization` in `auth.SECTIONS`.

- **Restart rules**: `training/quantize.py` is SFTP'd from disk per job — worker edits need **NO
  gateway restart**. `quantization_api.py` edits DO (GATEWAY_RELOAD=0).
- **Schemes** live in TWO synced places: `_SCHEMES` (api — labels + needs_calibration, served by
  `GET /v1/quantization-jobs/schemes`, which the web form reads so it never drifts) and
  `QUANT_SCHEMES` + `_build_recipe()` (worker — the actual llm-compressor modifiers). Add a scheme
  → touch both. Data-free: `fp8-dynamic`. Calibrated: `w4a16` (GPTQ), `w8a8-int8`
  (SmoothQuant→GPTQ), `fp8` (static), `nvfp4`, `awq`. All six verified e2e on tm-2 (Qwen3-0.6B).
- **⚠️ Multimodal / VLM vision+audio protection (added 2026-07-15, VERIFIED e2e on tm-2).** Quantizing
  a multimodal model's modality towers/embedders breaks vLLM two ways: (1) an FP8 vision tower dies in
  the forward with `RuntimeError: Not yet supported ScalarType 46`; (2) vLLM's gemma-4 impl builds the
  vision/audio input embedders as plain (unquantized) `ReplicatedLinear`, so a quantized
  `embed_audio`/`embed_vision` carries an unexpected `weight_scale` → load fails with
  `ValueError: no module or parameter named '…embedding_projection.weight_scale'`. The old default
  `ignore=["lm_head"]` FP8'd them → broken model. Fixed in `quantize.py`: `_recipe_ignore(cfg, model)`
  auto-detects a VLM/omni model (`_is_multimodal` — `config.vision_config` or a `*ConditionalGeneration`
  / `*VL` / `*ImageTextToText` arch) and unions `_MULTIMODAL_IGNORE` into the recipe `ignore` so the
  vision AND audio stacks stay full-precision: `re:.*vision_tower.*`, `.*vision_model.*`,
  `.*vision_embedder.*`, `.*visual.*`, `.*audio_tower.*`, `.*audio_model.*`, `.*audio_embedder.*`,
  `.*multi_modal_projector.*`, `.*embed_vision.*`, `.*embed_audio.*` (regexes → no-op on models lacking
  them). Text-only models are unaffected (stays `["lm_head"]`). Opt out with `quantize_vision: true`
  (job cfg / web toggle → `CreateQuantizationJobRequest.quantize_vision`). Also: multimodal `run()`
  saves the processor config into the output via `_save_processor_config` — tries
  `AutoProcessor.save_pretrained`, and when the processor class isn't importable (brand-new arch, e.g.
  gemma-4 unified has no `Gemma4UnifiedProcessor`) **falls back to copying the processor/preprocessor
  JSONs straight from the source snapshot** (model/tokenizer saves don't emit them → else vLLM "Can't
  load feature extractor"). Both take effect on the next job (quantize.py is SFTP'd, no gateway restart).
  **Verified:** `google/gemma-4-12B-it` (omni, `Gemma4UnifiedForConditionalGeneration`) → fp8-dynamic on
  tm-2 (H20-3e, `/share/quant-llmcompressor`): only `model.language_model` FP8'd (328 tensors), all
  vision/audio bf16; served on vLLM 0.23.0 (`/share/vllm-venv`, TP1, `--tool-call-parser gemma4
  --reasoning-parser gemma4`) — loads clean, generation + tool-calling both work. Published to
  `huggingface.co/huseinzolkepliscicom/gemma-4-12B-it-FP8-Dynamic`.
- **⚠️ MoE + hybrid-SSM + full-multimodal-load (added 2026-07-15, VERIFIED serving gemma-4-26B-A4B +
  Qwen3.6-35B-A3B).** Serving fp8 MoE / hybrid / VL models on vLLM surfaced 3 more "must stay bf16 /
  must load full" rules — `_recipe_ignore` + `_load_model` handle them automatically now:
  1. **MoE routers + gates** (`_is_moe` → `_MOE_IGNORE`). vLLM builds every MoE gating Linear
     unquantized — the top-1 router (gemma `router.proj`, Qwen `mlp.gate`) AND per-layer
     **shared-expert gates** (`mlp.shared_expert_gate`). A quantized one → `KeyError:
     '…router.proj.weight_scale'` at load, or a SILENT `weight_scale not found … skipping` warning →
     the fp8 weight is used undequantized → **garbage output**. Pattern is `re:.*gate$` (catches
     `mlp.gate` + `shared_expert_gate`, never the experts' `gate_proj`) + `re:.*router.*`.
  2. **Hybrid state-space / linear-attention** (`_is_hybrid_ssm` → `_SSM_IGNORE`: `re:.*linear_attn.*`,
     `.*mamba.*`). Mamba/GDN mixer layers are quantization-sensitive → fp8 = garbage. Detected via
     `linear_key_head_dim` / `mamba_*` config keys or a `layer_types` with linear/mamba entries.
     (Same sensitivity as Nemotron-H — see [[nemotron-h-training]].)
  3. **Full multimodal load** (`_load_model`): for a multimodal *config* (checked via `AutoConfig`
     BEFORE loading), load with **`AutoModelForImageTextToText`**, NOT `AutoModelForCausalLM` — on
     archs like Qwen3.5-MoE-VL the latter silently returns only the **text sub-model** (arch becomes
     `*ForCausalLM`, config loses `vision_config`) → vLLM's multimodal impl rejects it
     (`TypeError: Invalid type of HuggingFace config. Expected Qwen3_5MoeConfig`). gemma-4's
     AutoModelForCausalLM already returns the full model, so this is a no-op there.
  Diagnose fp8 serve failures by grepping the vLLM log for `weight_scale not found`, `KeyError:
  *weight_scale`, or `ScalarType`; garbage-but-loads → suspect an unignored gate/router or SSM layer.
  **Verified e2e:** `google/gemma-4-26B-A4B-it` (25.3GB, tool-calling works) + `Qwen/Qwen3.6-35B-A3B`
  (35.1GB, coherent — `17×23=391`) on tm-2, served on vLLM 0.23.0, published to
  `huggingface.co/huseinzolkepliscicom/{gemma-4-26B-A4B-it,Qwen3.6-35B-A3B}-FP8-Dynamic`.
  ⚠️ Qwen3.6 GDN uses a FlashInfer kernel that **JIT-compiles on first serve (~8 min, GPU 100%, health
  000)** — not a hang.
- **Worker venv**: one shared uv venv `/share/quant-llmcompressor` (built by `--deps-only`;
  llmcompressor + compressed-tensors + transformers + boto3 + huggingface_hub). Jobs on the same
  box MUST run sequentially-ish on first use (parallel venv builds race).
- **Calibration datasets**: kinds `hf`/`llm`/`upload`/`s3` (`_CALIB_DATASET_KINDS`). Note
  `_resolve_dataset_spec` (training_api) resolves **kind=llm like kind=hf** with `messages_field`
  carried — added for quantization; autotrain never passes kind=llm to it. ⚠️ **Script-based HF
  datasets fail** ("Dataset scripts are no longer supported" — `datasets>=3`); only parquet-native
  repos work (e.g. `roneneldan/TinyStories`). The worker falls back to guessing the text column
  when the dataset's `transcription_field` doesn't exist on the rows; `calib_text_field` /
  `calib_messages_field` in the job config override.
- **HF export** (`POST /{id}/hf-export`): `run_on="gateway"` (default — in-process
  `_hf_push_local`, no GPU) or `"vm"` (reuses `ta._run_hf_export_ssh` + the quant venv). Storage
  resolution mirrors autotrain: `ta._hf_token_for_storage` (handles `hf_token_secret` global-secret
  refs, not just inline `credentials_enc`) + `ta._hf_endpoint_for_storage` (custom endpoint = the
  self-hosted mirror) + `ta._loopback_endpoint` for gateway pushes. ⚠️ **Mirror pushes need Xet
  disabled**: modern huggingface_hub probes `{endpoint}/api/models/{repo}/xet-write-token/{rev}`,
  404s on the mirror, and aborts — `_hf_push_local` sets `HF_HUB_DISABLE_XET=1` **and patches
  `huggingface_hub.constants` on the live module** (hf_hub is usually already imported in the
  gateway process, so env-only is too late — same trick as `dataset_transform.py`). VM pushes to
  the own-mirror endpoint are rejected with a 400 (the box can't reach it without a tunnel).
- **Restart semantics**: `cleanup_orphaned_running` marks queued/running jobs **failed** on gateway
  startup — quant jobs are short and are NOT log-reconciled like autotrain runs; re-run instead.

### LLM chat packing (`llm_pack.py`) — per-arch tool-call `arguments` normalization

The **Pack for LLM** transform (`dataset_transform._run_llm_pack` → `llm_pack.pack_rows`, produces
`kind=llm_packed`; DPO → `pack_dpo_rows` → `kind=llm_dpo_packed`) tokenizes a chat dataset's `messages`
(+ optional `functions`/tools column) through the tokenizer's chat template **once, here** — the trainer
reads the packed ids as-is. CPU-only (threadpool; transformers imported lazily, no torch). The web UI is
the `LlmPackCard` transform tab; `tools_field` (default `functions`) names the tool-declaration column,
rendered as `tools=` (per-row `None`/empty/invalid → `extract_tools` returns `[]` → packed without tools;
mixed with/without-tools datasets are fine).

**⚠ Per-arch tool-call `arguments` MUST be normalized str→dict in `extract_messages`, and the requirement
differs by arch.** SFT parquets store `tool_calls[].function.arguments` as a JSON **string** (OpenAI
spec). The fix belongs in preprocessing (keep the parquet OpenAI-shaped), so `extract_messages(value, arch)`
dispatches to a per-arch `_normalize_<arch>_turn` right before `apply_chat_template`; `detect_arch` maps the
tokenizer name → `gemma|qwen|minimax|mistral|generic`. Whether the str→dict parse is needed depends on the
template:
- **gemma** (`_normalize_gemma_turn`, added 2026-07-09) — **REQUIRED; SILENT bug if missing.** The template
  branches on type: `arguments is mapping` → native `key:<|"|>value<|"|>`; `arguments is string` → dumped
  **verbatim as raw JSON** (`call:NAME{{"k":"v"}}`). A string cell renders wrong-but-valid → the finetune
  trains on a format that fights gemma's native decode (this was the ROOT CAUSE of a real FP8 tool-call
  regression — see the stress-test dir below). Also wraps bare `{name,arguments}` (no `function` key), else
  `tool_call['function']` KeyErrors → row dropped.
- **qwen** (`_normalize_qwen_turn`) / **minimax** (`_normalize_minimax_turn`) — **REQUIRED; LOUD.** Templates
  do `arguments|items` / `.items()` → a string raises → the row is dropped (caught in `pack_rows`). Both
  parse str→dict (+ `None`→`{}`; minimax parses the top-level AND nested `function` holder).
- **mistral** (`_normalize_mistral_turn`) — **str→dict NOT needed** — its native format *is* JSON
  (`[TOOL_CALLS]name[ARGS]{json}`; template `tojson`s a dict, uses a string verbatim). Only `None`→`{}`
  matters (else `None|tojson`→`null`).

So **adding a new arch to LLM packing = add a `_normalize_<arch>_turn` branch** (mirror qwen's) unless
you've confirmed the template tolerates a JSON-string args cell. **Restart rule:** `llm_pack.py` /
`dataset_transform.py` are imported (lazy, then `sys.modules`-cached) → a gateway restart is needed to pick
up edits (unlike `quantize.py`, SFTP'd per job).

**⚠ Assistant-only label masking (gemma-4), added 2026-07-10.** The gemma-4 chat template ships NO
`{% generation %}` block, so `apply_chat_template(return_assistant_tokens_mask=True)` returned an all-zero
mask and `tokenize_row` fell back to `labels = input_ids` — i.e. every gemma SFT run trained on the WHOLE
packed sequence: the system tool-DECLARATIONS, user turns, AND (env) tool-response blocks, not just the
assistant output. Training on the declarations is what taught earlier finetunes to regurgitate them.
`build_chat_template` now injects `{% generation %}` around ONLY the model-generated spans (reasoning
channel, tool_calls, text content, model turn-end) via surgical substring swaps in `_add_gemma_generation_mask`
(`_GEMMA_GENERATION_REPS`); the `<|turn>model` opener and the tool-response forward-scan stay unmasked. The
tags emit no text and abut whitespace-trimmed neighbours, so the **rendered text/token-ids are byte-identical
to stock** — only the label mask changes (**verified against google/gemma-4-31B-it**: same ids, mask covers
exactly the assistant spans; DPO's no-mask `tokenize_pair` render is unaffected). Anchor missing (template
changed upstream) → logs a WARNING + degrades to full-sequence labels rather than crashing. This is gemma-only
(other archs' templates already carry `{% generation %}`); the pack summary's `assistant_masked_rows` now
reports how many rows got a real mask.

**⚠ Nemotron-H (hybrid Mamba2/attention MoE) packs ONE-DOC-PER-BIN, added 2026-07-10.** `detect_arch`
maps `nvidia/…nemotron…` → `nemotron`; `_normalize_nemotron_turn` parses tool-call `arguments` str→dict
(REQUIRED — the template does `arguments|items`) + maps `reasoning`→`reasoning_content`;
`truncate_history_thinking=False` keeps all reasoning. Unlike every other (attention) arch, `pack_rows`
sets `one_doc_per_bin=True` for nemotron and writes each document as its OWN bin (never concatenates):
the HF NemotronH forward has no cu_seqlens/seq_idx plumbing, so multipacking would LEAK the Mamba SSM
state across doc boundaries. The nemotron trainer then PADS a batch of single-doc bins (not the varlen
concatenating collator). No `{% generation %}` (full-sequence labels for now). ⚠ The Nemotron tokenizer
needs **sentencepiece** in the gateway env to instantiate at pack time. See
`training/llm/CLAUDE.md` → "Nemotron-H" for the trainer/merge/dry-run detail.

**Verified e2e (2026-07-09)** on the real source `ds-f2116ddc` (635 tool rows / ~6.5k tool calls) by
rendering each row through the REAL tokenizer template + round-tripping vLLM's own tool parser (gemma4 /
qwen3xml / mistral / minimax_m2): fixed → ~100% native + parseable; unfixed → 0% (gemma raw-JSON,
qwen/minimax render-crash); mistral 100% either way. The packed output `ds-998f5e75` (gemma) decodes to
100% native tool calls. Tooling + the gemma root-cause writeup live in
`~/Documents/ucc_ai_research/stress-test/prompt-correction/` (`verify_dataset.py` gemma,
`verify_multi.py` qwen/minimax, `parse_mistral.py`, run on the tm-2 box's `/share/vllm-venv`).
Verification gotchas: (1) qwen/minimax templates embed a tool-call **example** in the system prompt → isolate
the assistant turn via a longest-common-prefix diff of `render(msgs[:k], gen=True)` vs `render(msgs[:k+1])`,
NOT a whole-render parse; (2) the box's vLLM venv forces **MistralCommonBackend** (rejects this data's
`chatcmpl-tool-*` tool_call ids — must be 9-char alnum — and validates msg structure) while the gateway's
transformers picks **TokenizersBackend** (Jinja, no id check) → render mistral off-box, parse on-box;
(3) mistral's parser reads args greedily to end-of-string → strip the trailing `</s>` before `json.loads`.
⚠️ Separately, `ds-f2116ddc` has a corrupt source row whose tool `name` literally contains `</arg_value>`
(fails cleanly across all archs) — a data-quality issue to scrub, not a packer bug.
