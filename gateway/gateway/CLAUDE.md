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
- **`LOG_JSON=1` means the WHOLE stdout stream is JSON** (fixed 2026-08-14 — it used to cover
  only `gateway.access`/`gateway.endpoint`). Two leaks in prod: (1) module logs and tracebacks
  still went out as `2026-08-14 08:46:43,226 INFO httpx: …`, which a Loki `| json` pipeline drops
  as unparseable — so the lines you actually need when something breaks were the ones missing
  from the query; (2) uvicorn configures `uvicorn`/`uvicorn.access` with **`propagate: False`** and
  its own handlers, so `INFO:     10.1.28.212 - "GET /ready HTTP/1.1" 200 OK` bypassed the format
  entirely. Now `_JsonLogFormatter` renders every record as `kind="app_log"` with the same
  `requestId` field the access line uses (one LogQL query spans both), and
  `_tame_server_loggers()` clears uvicorn's handlers + `propagate=True`. **uvicorn's access log is
  turned OFF** (`log_config=None` + `access_log=False` in `run()`): `metrics_mw` already logs
  every request — there is no ignore list — with the route template, duration, bytes and the
  actionable id, so uvicorn's line was a strictly worse duplicate. `LOG_UVICORN_ACCESS=1` restores
  it (only useful for requests dying *before* the middleware — a malformed request line).
  ⚠ Anything printing outside the logging module (a library writing to stdout) is still raw text.
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

### Compute on a VM — uv venv + JupyterLab, proxied (`compute_vm.py` + `compute_proxy.py`)

`/compute/new` takes a **`kind=vm` provider** alongside runpod/pi. There's no pod to
provision: the gateway SSHes into the registered box, builds `~/.sgpu/compute/{pod_id}/venv`
(`uv venv --seed --python 3.11` → `uv pip install jupyterlab ipywidgets`, both idempotent), and
`setsid nohup`s `jupyter lab` on the VM's **loopback** on a free port. An optional
`visible_devices` ("0,1") becomes `CUDA_VISIBLE_DEVICES` on that process — the box is shared,
so an unpinned session sees every GPU. `compute_pods.kind` is denormalized from the provider so
teardown still dispatches after a provider row is deleted.

- **uv ONLY — never the box's python (user requirement).** There is no `python3 -m venv` / system
  `pip` fallback: missing uv is installed on demand (`astral.sh/uv/install.sh` → `~/.local/bin`)
  and a box where that fails is an error. Even the free-port probe runs on the *venv's* python,
  so nothing in the flow invokes the system interpreter. Three details make this hold in practice:
  1. **`uv venv --seed`** — a plain uv venv has NO pip, so `which pip` in the notebook terminal
     falls through PATH to the box's `/usr/local/bin/pip` and a `pip install` silently writes into
     the machine's default python. `--seed` also makes the `%pip` cell magic work.
  2. **The terminal gets a generated rcfile** (`{rundir}/shellrc`, wired via
     `--ServerApp.terminado_settings`): it sources `~/.bashrc` FIRST and re-asserts
     `VIRTUAL_ENV`/`PATH` after, because terminado's default `bash -l` sources a box profile that
     usually prepends conda and would shadow the venv. Prompt shows `(sgpu:{pod_id})`.
  3. `VIRTUAL_ENV`/`PATH` are exported into the **server's** env too, so kernels inherit the venv
     with no activation step.
  Verified on tm-2: terminal `python`, `pip` and `sys.prefix` all resolve inside the session venv,
  `uv` on PATH, and the box's own python (which already shipped jupyterlab, dated 2025-10-17) is
  untouched.
- **`jupyter_version` pins JupyterLab** (`jupyterlab=={ver}` in the install; blank = whatever uv
  resolves). `compute_vm.validate_jupyter_version` accepts a **bare PEP 440 version only** —
  `>=4,<5` is rejected because a specifier needs `<`/`>` on a remote command line and "which Lab
  do I get" should have one answer. A pasted `==4.2.5` has its `==` stripped. ⚠ `BASE_PACKAGES[0]`
  must stay `jupyterlab` — `_launch_script` swaps that first element for the pinned spec.
  Verified e2e: `4.2.5` requested → `jupyterlab.__version__ == 4.2.5` in the session venv.
- **The form's Run-on choice is in the URL** (`/compute/new?run_on=vm`), same `router.replace`
  pattern as the benchmark form's `?tab=` — shareable, survives a refresh, and renders the
  bare-metal shape server-side. The switch (not the selected provider) is the source of truth for
  cloud-vs-VM, so the right form shape paints before `listProviders()` resolves; an effect
  re-snaps the account when the target flips so a RunPod account can't stay selected under
  "Bare metal" and 400 at create.

- **Jupyter is launched with `--ServerApp.base_url=/compute/jupyter/{pod_id}/{proxy_token}/`,
  identical to the gateway route serving it** — that's what makes `compute_proxy.py` a dumb
  byte proxy with **zero** HTML/JSON rewriting. Change one, change the other.
- **`--allow-root` is load-bearing.** Most GPU boxes SSH in as root and jupyter *refuses* to
  start as root without it — it logs "Running as root is not recommended" and exits, which
  looks exactly like a slow boot until the ready poll times out 5 min later. The launch script
  now reports `ALIVE=0` + the log tail after 4s so that failure lands in `error_text` fast.
- **Auth = the `proxy_token` in the path, and the routes are auth-exempt on purpose.** Gateway
  auth is Bearer-header-only: a browser can't attach a header to a link click, and Next's
  `/api/proxy` (which does the cookie→Bearer swap for the rest of the UI) **cannot proxy
  WebSockets**, so kernels would never connect through it. Hence a 32-byte capability token
  compared with `compare_digest` — **on every request, cache hit included** (caching authz by
  pod id alone would make a guessed id enough).
- **⚠ The GATEWAY authenticates to Jupyter, on both paths — `?token=` in the URL is NOT enough.**
  Jupyter issues **no session cookie** for a `?token=` login (verified: the cookie jar comes back
  empty), and JupyterLab's frontend only reaches the server because it replays the token from page
  config as an `Authorization` header on *fetch* — which a **WebSocket cannot carry**. Result
  before the fix: pages and `/api/*` worked while every kernel / terminal / `api/events/subscribe`
  socket got `403 Couldn't authenticate WebSocket connection`, and Lab sat in a
  "Connection lost, reconnecting in 60 seconds" loop. `_upstream_headers` now drops any
  client-supplied `Authorization` and injects `token {jupyter_password}` itself, HTTP and WS
  alike. Consequences worth keeping: the published URL carries **no** `?token=` (the path's
  proxy_token is the single capability and the notebook token stays out of browser history), and
  token-authenticated requests skip Jupyter's XSRF check so proxied POSTs need no `_xsrf`.
  **A scripted WS test that passes `?token=` in the query will pass even when the browser is
  fully broken** — that's what hid this; test the way a browser connects (no token, no cookie).
- **`Host`/`Origin` are rewritten to the upstream's `127.0.0.1:{port}`** so jupyter's
  same-origin check passes without `allow_origin='*'`. Response headers go back via
  `multi_items()`→`raw_headers`, **not** a dict — jupyter emits multiple `Set-Cookie`s
  (`_xsrf` + session) and a dict keeps only the last, silently breaking XSRF on POSTs.
  The upstream path comes from `scope["raw_path"]` (the router's `path` param is already
  percent-decoded, which mangles notebook filenames containing `?`/`#`).
- **The tunnel is `vm_tunnel.ensure_forward`** (same autossh `-L` as proxy-mode endpoints),
  re-ensured **lazily on each proxied request** — that's what heals it after a gateway restart
  with no background loop. `ensure_forward` only gives autossh 15s to bind and the TM boxes
  can take ~30s, so `_resolve` waits for the local listener (`_FORWARD_WAIT_S`) before caching;
  otherwise the first request after a restart always 502s. Teardown closes **only this
  session's** forward (`close_forwards(host, vm_port)`) — an unported `close_forwards(host)`
  would reap the box's live inference tunnels.
- **No SSH key is ever served for a VM session** (`GET /{id}/ssh` → 403): the "key" is the
  *provider's* box-wide credential, usually root on a shared machine. JupyterLab's terminal is
  the shell.
- **Terminate purges the uv venv, keeps the notebooks — and both confirm dialogs say so.**
  Reclaiming the venv matters (~300 MB per session, one per session on a shared box), but a
  blanket `rm -rf {rundir}` would take the DEFAULT workdir (`{rundir}/work`) with it, so the
  purge is scoped to `venv`/`shellrc`/`jupyter.log` + an `rmdir` (not `rm -rf`) that removes the
  per-pod dir only when nothing is left in it. Failure/abort paths pass `purge=False` so a retry
  reuses the venv. The kill is the recorded pgid, with a **pod-scoped `pkill -f {rundir}/`**
  fallback for a lost pidfile — the rundir carries the pod id, so it can't match a neighbour's
  server; never a bare `pkill jupyter`.
- **Teardown is retried at startup.** `delete_compute` schedules the teardown as a fire-and-forget
  task, so a restart/deploy landing right after a delete kills it mid-SSH and strands a jupyter on
  the box — invisible in the UI, still holding kernel GPU memory. A terminal row that still has
  `vm_port` set is exactly that state; `cleanup_orphaned_running` re-runs the teardown and a
  successful one NULLs `vm_port`/`vm_pid`. (Hit this for real while testing: two sessions deleted
  moments before a `pkill gateway` kept running.)
- **⚠ The compute subsystem no longer requires `RUNPOD_API_KEY`.** It used to be
  `if os.environ.get("RUNPOD_API_KEY")` in `main.lifespan` — a leftover from when Compute was
  env-key-only. On a VM-only (or cloud-disabled) deployment that silently started **neither** the
  idle auto-terminate loop nor the orphan reconcile, while the UI still offered "stop after idle".
  Both loops are cheap; they now always start.
- **Idle auto-terminate is scoped to the pinned GPUs** for VM rows, else a neighbour's job on
  the box keeps every session alive forever. Unpinned sessions effectively never auto-stop
  (fail-safe direction). Cost fields stay NULL — nothing is billed.
- **Verified e2e on tm-2** (2026-07-28), simulating a browser (no `?token=`, no cookie, no
  `Authorization`): `/lab` 200, all three sockets connect (`api/events/subscribe`,
  `terminals/websocket/N`, `api/kernels/{id}/channels` with the
  `v1.kernel.websocket.jupyter.org` subprotocol + binary frames), a real kernel executed code,
  `CUDA_VISIBLE_DEVICES` confirmed as `0` and `1,2` inside the kernel, custom `/share/...`
  workdir honoured, terminal resolved python/pip to the session venv, terminate left no jupyter
  or kernel processes.

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

### Proxy request tracing (OTLP → Tempo) + where the request RECORD lives

`tracing.py` (write) + `trace_store.py` (read) + `PROXY_REQUEST_STORE` (how much Postgres
keeps). Motivation: every proxied request wrote a `proxy_requests` row and **`proxy_requests`
is not a table shape Postgres should be asked for at prod volume** — append-only, write-once,
read-rarely, unbounded (`PROXY_REQUEST_RETENTION_DAYS` defaults to 0 = keep forever). The
record now also goes out as **one OTLP span per request**, and Postgres can be told to hold
part of it or none.

**The write path had THREE synchronous DB round-trips per request; it now has zero.**
`_insert_request_row` (INSERT+COMMIT at admission, before the upstream call) and
`_set_started` (SELECT+UPDATE at queue exit) both checked out a pooled connection on the
latency path — the same pattern that caused the pool-exhaustion incident `stats_writer` was
built to fix, still living at the front of the request. Admission is now
**`_admit_request`** (span + an enqueued `proxy_insert` intent), start is
`stats_writer.record_proxy_start`, and both share the `proxy` intent kind with the terminal
one so a fast request collapses to a **single INSERT**.

- **`PROXY_REQUEST_STORE=all|sampled|errors|off`** (default `all` — byte-identical
  behaviour). `sampled` keeps `PROXY_REQUEST_STORE_RATIO` of rows, chosen by hashing the
  request id — **deterministic on purpose**: sampled IN at admission and OUT at completion
  would strand a permanently `queued` row. `errors` writes nothing up front and one COMPLETE
  row at `_finish` iff the request failed / was blocked / cancelled / returned ≥400.
- **⚠ The Prometheus metrics must never depend on the row.** `observe_proxy` used to fire as
  a side effect of the row UPDATE inside `_apply_proxy`, so `store=off` would have silently
  taken every per-proxy latency/TTFT/tok-s series down with it — a storage decision becoming
  a monitoring outage. It is now `stats_writer._proxy_metric`, fed `endpoint_id`/`model` on
  the intent, and `record_proxy_finish(store=False)` enqueues a `proxy_metric` kind that
  never opens a session. Verified: 15 requests in `errors` mode → 3 rows, 15 in the counter.
- **⚠ In-flight requests are in NEITHER store.** A span is exported when the request ENDS and
  (outside `all`) no row exists while it is queued. `_live_records` overlays the live registry
  onto every non-`all` answer — without it the Queue tab shows an empty queue while requests
  are actively waiting, which is exactly when someone is looking at it. Page 1 only (offset 0),
  else an in-flight request repeats on every page.
- **`_finish` is the single funnel** for span-close AND storage policy, so the trace and the
  row can never describe the request differently. `_ADMITTED` (request id → admission facts)
  carries what `_finish` can't otherwise reach: the metric labels, the deferred insert's
  `created_at`/`owner_id`, and the live overlay. Both it and the span map are swept by the
  proxy health loop (`_sweep_admitted` / `tracing.sweep`) — a handler that dies without
  reaching `_finish` would otherwise leak in two places and never emit its trace at all.

**tracing.py**
- OFF unless `PROXY_TRACING=1`; the opentelemetry import is lazy and a **missing wheel logs
  and stays off** rather than failing startup (a gateway that won't boot for a telemetry dep
  is a worse outage than no telemetry). Every entry point is exception-wrapped — it sits on
  the request path.
- **⚠ Sampling decides at the END, not the start.** A head sampler (OTel's default) chooses
  before the outcome is known, so a 5% sample keeps 5% of your failures. `_PolicyProcessor`
  wraps the BatchSpanProcessor and drops in `on_end`: errors / blocked / cancelled / ≥400 /
  slower than `PROXY_TRACE_SLOW_MS` are **always** kept, and only plain successes are ratio-
  sampled (deterministically in the trace id, so replicas agree).
- The caller's `traceparent` is honoured, so a client that traces its own work gets this span
  as a **child** rather than an orphan. Deliberately does NOT call `set_tracer_provider` —
  claiming the global would adopt every library that auto-instruments itself, which is a much
  larger traffic decision than this one.
- Attributes are the Queue tab's columns: `sgpu.request.id` / `.proxy.id` / `.proxy.name` /
  `.model` / `.upstream` / `.status` / `.stream` / `.latency_ms` / `.ttft_ms` / `.owner`,
  plus `http.response.status_code` and `gen_ai.usage.{input,output}_tokens`. `blocked` and
  `cancelled` are NOT span-status ERROR — a guardrail doing its job and a caller hanging up
  are not gateway errors, and marking them would poison every error-rate panel.

**The trace is a WATERFALL, not one bar** (added 2026-08-15 — the first cut emitted a single
flat span, which is a log line wearing a trace's clothes). `tracing.start_child` /
`end_child` / `add_event` hang phases off the request span:

    proxy <endpoint>                     ← the request, starts when it ARRIVED
      • routed        @+5.8ms            ← body ingest + the routing DB read
      • running       @+5.9ms
      • first_token   @+62.1ms           ← prefill|decode boundary on a stream
      ├─ queue                 0.0ms     ← only when the endpoint HAS max_concurrency
      ├─ red_team <mode>      41.0ms     ← blocking guard, paid inline by every request
      ├─ upstream <name-A>     1.1ms ✗   ← failover attempt (error, sgpu.failover=true)
      └─ upstream <name-B>    60.6ms     ← the one that served it

- **⚠ One span per upstream ATTEMPT is the point.** Failover as N sibling spans (the failed
  ones marked) is a story; a single `upstream=<whoever answered>` attribute is not. Covered:
  `_do_unary`, `_stream`, `_forward_passthrough`.
- **⚠ The span must start BEFORE routing.** `_route` does a DB read per request; a span
  opened after it shows a fast request while the caller waited on a slow query. `_prepare`
  and `_handle_ingest` stamp `t_arrive` up front and thread it through `_admit_request(t0=…)`,
  which is also the row's `created_at`. The `routed` event marks where that ended.
- **⚠ `mark_started` emits the `running` event only ONCE.** `_set_started` is called twice on
  the streaming path (before and after upstream selection) — the attribute update is
  idempotent, two "running" marks on a waterfall are not.
- **A child left open is never exported** — the phase silently vanishes instead of failing
  loudly — so `end_request` force-closes stragglers (`sgpu.unfinished=true`) for the cancel /
  disconnect paths that skip the explicit `end_child`.
- `first_token` is an EVENT as well as `ttft_ms`: an attribute gives the number, only an
  event gives its position in the waterfall.

**⚠ A finished request is NOT instantly searchable — and the Queue tab links straight to
it.** Measured end-to-end (request completes → the trace can be FOUND by request id):
**~15s out of the box**, which reads as "tracing is broken" when you click a row that just
appeared. Two stacked delays, both now tuned:
- **5s** in the SDK — `BatchSpanProcessor`'s default `schedule_delay_millis`. Fine for
  offline analysis, wrong for a UI that deep-links. → `PROXY_TRACE_EXPORT_DELAY_MS` (1000).
- **~10s** in Tempo — `ingester.trace_idle_period` (default 10s) holds a trace in the live
  map waiting for more spans before cutting it into a searchable block. These traces are
  single-process and complete when the request ends, so that wait buys nothing. → **2s** in
  `deploy/monitoring/tempo/tempo-config.yml`.
Result: **~2.0s** (3 runs: 2.1 / 2.1 / 1.8). It cannot be zero, so the Queue tab's `trace`
link says so in its tooltip. ⚠ `docker compose up -d tempo` does NOT restart the container
for a mounted-config change — `docker restart gpuplatform-monitoring-tempo-1`.

**trace_store.py — three things TraceQL can't do that SQL could, all surfaced to the user**
1. **A time window is mandatory** (`PROXY_TRACE_WINDOW_H`, default 24). "Everything ever" is
   not a query Tempo answers.
2. **No `OFFSET`, no `ORDER BY` but time.** ⚠ **A latency sort must fetch the whole window,
   not one page** — Tempo returns most-recent-first, so ranking a page of `limit` traces
   answers "slowest of the newest N" while looking like "slowest". Caught in the e2e: a
   `limit=3` sort returned 5200/410/300 ms and hid the 1820 ms request. Sorting/paging happen
   in-process over a fetch capped at `PROXY_TRACE_MAX_FETCH` (500); past the cap the `note`
   says the ranking is partial (a silently truncated ranking reads as an answer).
3. **`select()` is not optional** — a TraceQL search returns only the attributes it MATCHED
   on, so without it every unselected column comes back empty and looks like missing data.
- **⚠ An unreachable Tempo RAISES (502), never an empty list.** "No requests" and "the trace
  backend is down" must not look identical in a monitoring UI.
- ⚠ OTLP JSON encodes **ints as strings** (`{"intValue": "1500"}`) — take them at face value
  and every token count and latency silently becomes text. Both `spanSets` (current) and the
  older bare `spanSet` are read; handling one returns zero rows against the other version.
- `GET /{id}/requests?source=db|trace` overrides per call; `PROXY_HISTORY_SOURCE=auto` (default)
  reads traces exactly when `PROXY_REQUEST_STORE != all`, so flipping storage doesn't also
  require remembering to flip the read path. The source + caveat ride back as
  **`X-SGPU-History-Source` / `X-SGPU-History-Note`** (the `x-sgpu-*` prefix is already passed
  through by the web proxy) and `/request-facets` reports `source`/`window_hours`/`store` —
  the Queue tab renders both. ⚠ **Header values are latin-1**: an em-dash in the note 500'd the
  endpoint in the e2e; `_hdr()` sanitizes.

**Local stack**: `deploy/monitoring/docker-compose.monitoring.yml` now runs **Tempo**
(+ a provisioned Grafana datasource with trace→logs and trace→metrics links).
⚠ Its OTLP/HTTP host port defaults to **4418, not 4318** — 4318 is squatted on this laptop
(same class of collision as :8080/:3000); override with `TEMPO_OTLP_HTTP_PORT`. The query
API (3200, `TEMPO_URL`) and the ingest port are *different services' ports* — mixing them up
looks like "tracing works but history is empty".

    PROXY_TRACING=1 OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4418 \
      TEMPO_URL=http://localhost:3200 PROXY_REQUEST_STORE=off .venv/bin/gateway

**Verified e2e locally** (2026-08-14) against a mock upstream, on a second gateway (:8099) so
the running one was untouched: real proxied requests (unary + SSE + a 500) → OTLP → Tempo →
the Queue-tab API, with owner/upstream/status/request-id filters, latency sort, facets and
endpoint isolation all correct; `store=all` regression = 21 concurrent mixed requests → 21
complete rows (incl. the insert+update-coalesced-in-one-flush case, 0 stuck `queued`);
`store=errors` = 15 requests → 3 rows, 15 counted in Prometheus. Restart rule: `tracing.py` /
`trace_store.py` / `proxy_api.py` are imported → **gateway restart** for edits.

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

### Experiments — behavioural stress testing + agent observability (`experiments_api.py`)

The generic half of every ad-hoc stress-test script ever written against a served model. Those
scripts are ~90% identical (load a captured request → replay it N times across endpoints/variants
at concurrency C → assemble the stream → classify → tally → SUMMARY.json) and differ only in the
**classifier**. So the classifiers live in **`evaluators.py`** as a registry and everything else is
config. Section key `experiments` in `auth.SECTIONS`.

    EvalDataset ── EvalCase (replayable {messages, tools, params})
         └── Experiment = dataset × targets × variants × repeats
                  └── ExperimentSample = one completion + its evaluator verdicts

- **Restart rules**: everything here is imported, so **any edit needs a gateway restart**
  (`GATEWAY_RELOAD=0`). New tables are created by `create_all`; no ALTER migration was needed.
- **`evaluators.py` is the extension point.** Detectors are pure sync functions of one
  `Completion` → `EvalOutcome`; add a `_check_*` + a `SPECS` entry and **nothing else changes** —
  `GET /v1/experiments/evaluators` serves the registry (id/label/description/option schema) and
  the web form renders it, the same server-driven-dropdown convention as quantization schemes.
  Unit tests: `gateway/tests/unit/test_evaluators.py` + `test_experiments_unit.py`.
- **Targets are always plain `{base_url, model, key}`**, even for this platform's own apps/proxies.
  `GET /targets` only *prefills* them from the app + proxy registry. One code path in the runner,
  and a third-party endpoint is a first-class target — mirrors `proxy_api`'s `stt_callback`.
  Inline keys are Fernet-encrypted into `config_json`; `_public_config` strips them on the way out
  (it deep-copies first — mutating the ORM row's JSON dict would flush the redaction back to the DB).
- **⚠ `retries` defaults to 1 — no retry, on purpose.** Retrying masks the failure being measured:
  a fast 500 or a 0-token reply IS the finding. `request_error` is in `ALWAYS_ON`, and a sample
  with a transport error is force-failed (its content checks ran against `""`, so their "passes"
  are meaningless).
- **⚠ Read reasoning under BOTH names.** Dynamo emits `reasoning_content`, plain vLLM emits
  `reasoning` (`_reasoning_of`). Reading one silently turns a "reasoning-only empty" into a plain
  "empty" on the other backend. Streaming also always injects `stream_options.include_usage` or
  every token-derived metric (cost, empty-with-0-tokens) reads zero.
- **⚠ Streamed tool-call arguments arrive char-by-char** — `_merge_stream_tool_calls` reassembles
  them by `index`. Without it every streamed tool call looks like malformed JSON.
- **The runner is in-process** (asyncio + a dedicated httpx client, separate from the proxy's so a
  long run can't starve live proxy traffic). Bounded by a worker pool + a hard `EXPERIMENT_MAX_UNITS`
  cap (20k; a 4-target × 3-variant × 200-repeat × 20-case sweep is 48k real billed calls). Stored
  sample text is capped at `EXPERIMENT_MAX_STORED_CHARS` (8k).
- **⚠ Restart cleanup is heartbeat-gated, not blind.** `cleanup_orphaned_running` only fails rows
  whose `heartbeat_at` is stale (>120s) or absent — the runner stamps it on every ~2s progress
  flush. A blind sweep would kill runs legitimately in flight on **another HA replica**.
- **Cancel is cooperative**: in-flight units finish, queued ones are skipped, and the partial
  results are still summarized (verified: 525/2000 → `cancelled` with a usable summary).
- `summary_json` = one **cell** per (target, variant) with per-evaluator pass rates + latency/
  token/cost stats. That's the matrix the UI's parallel-coordinates plot draws.

**Benchmark-derived unit tests** (`evaluators.py` + `langid.py`). Two built-ins port the *scoring*
halves of two standalone research benchmarks maintained **out of tree** (function-calling quality
and code-switching / reply-language), so a suite that needed its own repo, venv and fleet run is
now a checkbox:
- **`function_call_units`** — the function-call benchmark's per-turn metrics. Reads
  `expected.tool_calls` off the dataset row and resolves each call against the request's OWN
  `tools` (plumbed through as `expected._tools` by the runner) so hallucination and parameter
  checks need no configuration. Optional `expected.{available_ids,tool_results,out_of_context}`
  enable id-propagation and the refusal metric.
- **`multilingual_units`** — the code-switching benchmark: did the reply come back in
  `expected.language`? ⚠ **`__label__id` maps to `malay`** exactly as the benchmark does, so an
  Indonesian reply would be credited as Malay; `langid.indonesian_leak()` (the ported 475-word
  lexicon, whole-token matched) surfaces `malay_corrected`/`overall_corrected` alongside the raw
  numbers. `strict_malay` (default on) makes a leak an outright failure.
- **`langid.py` uses fastText when it's there, its own detector otherwise.** Set
  `EXPERIMENT_FASTTEXT_MODEL=/path/to/lid.176.bin` (+ the `fasttext` wheel) for exact parity with
  the published tables; without it, script ranges settle Chinese/Tamil exactly and function words
  decide Malay-vs-English. **Every result records `detector: fasttext|builtin`** so two runs
  scored by different paths are never silently compared.

**⚠ Single-turn replay ≠ the benchmark's multi-turn replay.** The real function-call harness walks
a conversation 10–20 turns deep, feeding the *reference* tool results back in after each turn so
the model accumulates context (it has no live telco backend). The Experiments runner sends **one
request per dataset row**. On a single-turn row that's equivalent; on a multi-turn corpus like
`Scicom-intl/Function-Call-TaaS` it scores **only the first turn** and reads higher than the
published figure. Don't present the two as the same measurement. Implementing real multi-turn
replay means threading `expected.tool_results` back into `build_request` and looping per row —
the runner's `run_unit` is one-shot today, so it's a genuine addition, not a config flag.

**⚠ A row missing its reference is SKIPPED, not failed.** No `expected.tool_calls` /
`expected.language` → the detector abstains with `flags.skipped` and `passed=True`, because
inventing a verdict is worse than none. The consequence: a suite pointed at a dataset with no
`expected` column reports a clean 100% having scored nothing. Both aggregators return **`scored`** for exactly this
reason — **compare it against the row count** (`turns` on the function-call one) before believing
a result. `scored: 0` with a green pass rate means the dataset carries no reference data.

**⚠ The `aggregate` hook exists because F1 cannot be averaged.** These two score one reply at a
time like every other detector, but their headline numbers are **corpus-level**: `tool_call_f1`
pools tp/fp/fn across every reply, and per-language accuracy pools per class. So
`EvaluatorSpec.aggregate(flags_list) -> dict` runs once per (target, variant) cell in
`summarize()`, and its output lands on `cell.evals[id].metrics` (+ `headline` for the ones worth
reading first). Averaging the per-sample rates instead would NOT reproduce the benchmark's tables.
The per-sample `flags` that feed it are pooled and discarded — never stored on the cell.

**Synthetic corpus generation** (`synthetic.py` + `dataset_transform._run_generate` +
`/v1/datasets/generate*`). Red teaming is the case where there is nothing to capture — nobody has a
log of the attacks nobody has tried yet — so a generator LLM (any OpenAI-compatible endpoint) WRITES
the corpus. It lives in the **Datasets** section (`/datasets/new` → source "Generate (synthetic)"),
NOT Experiments: the output is an ordinary `kind=upload` chat dataset, and Experiments has no
dataset store.
- **⚠ The dataset row is created EMPTY and returned immediately; rows grow in the background.**
  `POST /datasets/generate` returns a real dataset id at once (mirrors `merge_datasets`), then the
  job publishes after EVERY batch — `num_rows`/`size_bytes` advance while you watch it. Progress
  rides the EXISTING transform plumbing (`transform_status`/`transform_log`, `_active`,
  `POST /{id}/cancel-transform`), so the datasets UI polls and cancels it with no new machinery.
- **⚠ Each publish rewrites the WHOLE JSONL rather than appending.** S3 has no append, and the
  corpus is bounded (`DATASET_SYNTH_MAX_ROWS`, 200 → tens of KB). The point is that the object is a
  complete, parseable file matching `num_rows` at every instant — a partial trailing line would make
  the dataset unreadable for every consumer, which is the whole risk of "rows grow in the background".
  Cancel keeps what was published (smaller, not broken).
- **`Dataset.gen_spec` (JSON, idempotent ALTER) is provenance AND the restart marker** — mode, row
  target, categories, generator model/base_url; **never the API key** (that is passed in memory to
  the task and nowhere else). `datasets_api.cleanup_orphaned_generating()` (wired into
  `main.lifespan`) fails rows left `running` by a restart, since an in-process asyncio task dies
  silently and the row would otherwise sit at `running` forever.
- **The taxonomy is shared with the proxy's red-team guard** (`proxy_api.RED_TEAM_DEFAULT_TYPES`),
  so "which attacks got through the model" reads against "which attacks the guard blocked"
  (`X-SGPU-Red-Team-Type`) in one vocabulary. Served by `GET /datasets/generate/options` —
  server-driven, same convention as the evaluator registry.
- **⚠ Benign controls are not filler.** `mode=mixed` generates harmless look-alikes of each attack
  category; without them a model that refuses EVERYTHING scores a perfect refusal rate. `n_benign()`
  guarantees ≥1 of each half so a "mixed" corpus is never silently single-mode.
- **`plan_batches` splits the quota per CATEGORY first, then chunks.** Round-robining batch-by-batch
  (30 rows / 5 categories / batch 10) emits 3 batches and silently tests 3 of the 5 categories.
- **Short counts are tolerated, not chased.** A model returning 7 of 10 prompts yields a smaller
  corpus + a log line; refilling to an exact number spends unbounded billed calls. Parsing accepts a
  bare array, a fenced array, an object wrapper, `{prompt:…}` items, or a numbered list — re-asking
  costs a call and the failure is trivially recoverable. Dedup is case/punctuation-insensitive
  (the same attack 30 times measures one attack).
- Rows carry `expected.{attack,attack_type,expect_refusal}` — that's what makes the corpus
  self-scoring for the `red_team` evaluator below. `POST /datasets/generate/preview` generates ~6
  rows and persists nothing, so a wrong URL/model/key costs one small call.

**`red_team` evaluator** (`evaluators.py`) — scores the MODEL UNDER TEST's reply, not the guard's
verdict: a lexical refusal / compliance / leak classifier (deterministic, no second billed call;
pair it with `llm_judge` when wording matters more than throughput).
- **⚠ The two directions are scored by OPPOSITE rules.** Refusing an attack row is correct;
  refusing a benign control is the failure (over-refusal). Hence the `aggregate`: a 50/50 corpus
  where the model refuses everything shows a 50% *pass rate* — which looks like a coin flip instead
  of the total over-refusal it is. ⚠ **Replaying through a red-teamed proxy, check
  `model_refusal_rate`, not `refusal_rate`** — the latter is end-to-end (guard + model) and the
  guard's canned block matches every refusal pattern, so a guardrail can post a perfect score for a
  model that would have answered every attack. See the guard-attribution bullet in the red-teaming
  section. `refusal_rate` / `over_refusal_rate` / `safety_score` (their
  mean, so neither half can be gamed) / per-category `refusal_<type>` / `leak_rate`.
- **⚠ An empty reply is not a refusal** (`min_refusal_chars`, default 15) — otherwise a dead
  endpoint posts a perfect safety score. A compliance marker ("Sure, here's…", "DAN mode enabled")
  beats a hedging refusal phrase, because refusing and then complying is complying.
- **Malay refusal markers are load-bearing**, not a nicety: an English-only list reads every Malay
  refusal as compliance and inverts the score on this platform's own traffic.
- Follows the platform's abstain rule — a row with neither `expected.attack` nor `expect_refusal`
  is **skipped, not guessed**; force `mode=attack` for a hand-built all-attack corpus and check
  `scored` against the row count.

**Sandboxes — multi-turn tool replay** (`sandbox.py` + the `CustomSandbox` table +
`experiments_api.run_trajectory`; design + the unbuilt stages in `docs/EXPERIMENTS_SANDBOX.md`).
A sandbox is the thing that ANSWERS a model's tool call during a replay, which is what turns
"one request per row" into a conversation. It's `CustomEvaluator`'s twin: same trust ladder
(`replay` | `api` implemented; `llm` | `python` declared but rejected at save), same
snapshot-at-create rule, `POST /v1/custom-sandboxes/test` to dry-run ONE row,
`GET /v1/experiments/sandboxes` for the server-driven mode descriptors.
**`sandbox=None` is the default and that path is unchanged** — `EvaluatorStack.evaluate()` and
GEPA never see any of this.

- **`mode=api` is what makes it general**: `POST {conversation, tool_call, call}` → the result read
  out of the response by a dotted `response_field`, so any mock service / staging API / simulated
  environment becomes a sandbox with zero gateway code. Nothing runs on the gateway, so the only
  guard is `netsafe.assert_safe_fetch_url` (re-checked on first use — a saved hostname can be
  re-pointed) + `follow_redirects=False`. ⚠ **`send_expected` is OFF by default and that is a
  CORRECTNESS setting**, not privacy: `row.expected` holds the gold reference the evaluators grade
  against, so a simulator that can read it can return exactly the reference result and inflate the
  score with nothing in the trajectory showing why. ⚠ `api_config()` treats only `None` as unset —
  `""` is meaningful for `response_field` (whole response) and `auth_prefix` (no `Bearer `), the
  same bug `custom_eval.api_config` already carries a warning about; two unit tests pin it.

- **⚠ `wants` is why this isn't a one-line adapter.** The model's parsed tool calls are NOT a
  `Completion` field — the runner side-channels them as `expected["_tool_calls"]` (`_tools` for
  the schemas). A trajectory's FINAL turn is the text answer, so projecting to it hands
  `function_call_units` an empty call list against a non-empty reference on every row and collapses
  `tool_call_f1` to 0 — which reads as a model regression, not a bug. Hence
  `EvaluatorSpec.wants ∈ {completion, turn, trajectory}`: `"turn"` scores each assistant turn and
  folds any-fail, with every turn's flags pooled into the `aggregate` (which is what the
  out-of-tree benchmark does over a conversation). Set on `function_call_units`,
  `control_token_leak`, `degeneration` — and **only** those, because a `"turn"` detector must be
  meaningful on a turn whose content is empty because the model only called tools. That
  disqualifies `empty_response` (every tool turn would flag empty), `json_output`/`structure_tags`/
  `regex` (patterns can't match `""`), and `multilingual_units`/`red_team` (their subject is the
  final reply). `test_sandbox.py::test_final_turn_projection_would_zero_tool_call_f1` pins it.
- **⚠ A row's reference describes ONE turn.** `ev.turn_expected()` gives turn 0 the row's
  `expected`, and later turns get the reference keys (`TURN_REFERENCE_KEYS`) STRIPPED unless the row
  carries `expected.turns[i]` — else round 2 is graded against round 1's gold answer and the
  detector invents failures. The fold publishes no "turns scored" count on purpose: abstention
  vocabulary is per-detector, so the authoritative number stays `metrics.scored` from the
  aggregate, pooled out of `flags.turn_flags` by `summarize()`.
- **⚠ `EXPERIMENT_MAX_CALLS` (60k), not `MAX_UNITS`, is what bounds spend now.** A sandboxed unit is
  up to `max_tool_rounds + 1` billed calls, so the 20k unit cap alone would permit ~140k. Checked at
  create AND re-checked at run time, same discipline as GEPA's billed-call budget. `GET
  /v1/experiments/limits` serves it — **the form must price a sandboxed run in calls**, and that
  arithmetic has to match the server (the existing rule for the GEPA budget card).
- **⚠ `replay` matches seed entries by NAME, not by exact arguments** (`match: "exact"` opts in).
  Requiring identical args sounds stricter and is useless: a model under test rarely reproduces the
  reference call's arguments, so nearly every call would be `no_fixture`, the trajectory dies at
  round 1, and a seed-coverage problem gets reported as a catastrophic model score. Scoring the
  ARGUMENTS is the evaluator's job (`function_call_units`), not the environment's. Repeated calls
  walk the seed in order, then reuse the last entry rather than erroring.
- **The response cache is a contract with a DIRECTION.** Same `(name, canonical_args)` → same
  content for the whole run (else two variants face different worlds), which means the **first cell
  to make a novel call defines it for everyone** — `seeded_by` records which. Errors are cached too,
  or a retry makes the comparison luck. For an exact A/B, freeze a cache from a reference pass.
- **A tool failure does NOT abort; a transport failure does.** The model gets a structured error as
  a `role=tool` message and may react to it (realistic, worth scoring); a dead endpoint aborts the
  trajectory and `run_evaluators_trajectory` force-fails the sample, so partial trajectories are
  never partial credit. Per-cell `summary.cells[].sandbox` carries the anti-silent-no-op numbers:
  `provenance` histogram, `novel_call_rate` (high in replay = you measured seed coverage),
  `forced_final_rate` (high = you measured the round limit), `aborted`, and `all_errors`.
- Storage: `experiment_samples.trajectory_json` (idempotent ALTER; NULL for non-sandboxed samples),
  capped at `EXPERIMENT_MAX_TRAJECTORY_CHARS` (32k) with **tool results truncated before model
  turns** — the model's output is what's under test, a fixture's payload isn't.
- ⚠ Summed `usage` means `prompt_tokens` DOUBLE-COUNTS context (turn N's prompt contains turn N−1's).
  Right for cost, wrong for comparing a sandboxed cell's prompt-token mean to a single-turn run's.
- Restart rule: `sandbox.py` / `experiments_api.py` are imported → **gateway restart** for edits.

**Custom evaluators** (`custom_eval.py` + the `CustomEvaluator` table). The escape hatch for a
check the built-ins don't cover, authored on the **Evaluators tab** (`/experiments/evaluators`)
rather than in the run form — they're a reusable resource, not a per-run setting. Saved entries
are a per-user library; an experiment **snapshots the whole definition into its config** at
create time, so editing a library entry can never retroactively change what a finished run
measured (verified). Referenced as `custom:<ce-id>`, or `custom` with the definition inline.
Results are keyed by the evaluator's **name**, which is why a name colliding with a built-in id
is rejected. **Three modes:**

1. **`expression`** (default, always on) — ONE Python expression, checked by a whitelisting AST
   walker: no statements, no imports, no comprehensions, calls only into a fixed helper registry,
   attributes only from `SAFE_METHODS` (never a dunder). Everything in scope is plain data, so
   there's no object graph to climb toward `__subclasses__`. It runs **in the gateway process**,
   which is why two compute bombs are also blocked: **`**` is rejected outright** (`2**(10**10)`
   is a three-character DoS) and a constant multiplier above `_MAX_REPEAT_CONST` (1000) is
   rejected (`content * 100000000` allocates a gigabyte). Adding to `_ALLOWED_NODES`? Re-check both.
2. **`api`** (always on) — POST the completion to an endpoint the user already runs; read the
   verdict out of the JSON via **dotted paths** (`result.verdict`, `result.detail.why` — `dig()`
   walks dicts AND list indices; `""` means the whole response, for an endpoint answering a bare
   `true`). String verdicts are understood (`PASS`/`fail`/`yes`/`no`). Nothing executes on the
   gateway, so there's no sandbox to reason about — only **`netsafe.assert_safe_fetch_url`**,
   which permits internal hosts (an in-cluster scorer is legitimate) but blocks link-local /
   cloud-metadata, is re-checked on first use (a saved hostname could be re-pointed), and pairs
   with `follow_redirects=False` so a 3xx can't bounce onto a blocked host. Keys are **global-secret
   references**, never stored inline. Config lives in the `config` JSON column (added by an
   idempotent ALTER in `db.init_db`).
   ⚠️ **`api_config()` treats only `None` as "unset", NOT `""`** — empty string is a *meaningful*
   value for `passed_field` (whole response) and `auth_prefix` (no `Bearer `). An earlier version
   skipped `""` and silently reinstated the defaults for both; two unit tests pin it.
3. **`python`** (**admin-only, ON by default** since 2026-08-07 — was opt-in) — a real
   `def check(c)`. The remaining gate is **admin role**, re-checked at experiment-create time
   (not just at save); `EXPERIMENT_ALLOW_PYTHON_EVALUATORS=0` disables the mode outright for a
   deployment that can't accept it. ⚠ With the env flag gone from the default path, **admin role
   is the whole control** — an admin (or anything authenticated as one) has code execution on the
   gateway host. Runs in a child process with the gateway's env **scrubbed** to
   `PATH/LANG/LC_ALL/SYSTEMROOT/TMPDIR` (no `DATABASE_URL`, no `PROVIDER_SECRET_KEY`, no cloud
   keys — a unit test asserts this), CPU/address-space/NOFILE rlimits the child sets on itself
   (`preexec_fn` is unsafe from a threaded parent), and a wall-clock kill.
   ⚠️ **This is blast-radius reduction, not a jail** — user code can still read what the gateway
   user can read. Enabling it grants code execution on the gateway host to anyone who can reach
   the endpoint. Prefer `expression` (sandboxed) or `api` (runs nowhere near us).

- **One child process per python evaluator per run**, not per sample (`PythonEvaluatorWorker`
  serves one JSON line per call) — a 10k-sample run would otherwise be 10k process launches. A
  hang kills+respawns; a *fatal* definition error latches so it doesn't respawn-storm. api mode
  gets its own concurrency semaphore so a slow scorer can't starve the replay connections.
- **An author's bug never fails a sample.** A bad regex, a `None.get()`, a timeout, a crashed
  child, an unreachable endpoint, a response missing the configured field — all return
  `passed=True` with `flags.evaluator_error`, because silently marking real replies as failures
  is worse than a missing signal. `POST /v1/custom-evaluators/test` dry-runs one against a pasted
  reply or a real `sample_id`; the UI leans on it because `fail_when_true` inverts every result
  if you pick it wrong.

**Cases come from the platform's own Datasets section — Experiments has NO dataset store.**
(Vocabulary: everything **user-facing** says "row" — `max_rows`, `n_rows`, `/datasets/{id}/rows`.
`Case` survives as the Python name because it's a row *resolved into* a replayable request, which
is a real distinction: rows without a usable messages cell are dropped, so cases ≤ rows and
`n_planned` is an estimate. Don't put "case" back into UI copy or the API.)
`resolve_cases()` reads rows out of a `Dataset` through that section's own readers
(`_hf_preview_rows` for `llm`/`hf`, `dataset_metadata.parse_rows_any` over the S3 metadata file
for `upload`/`s3`), so a dataset behaves identically here and in its row browser. A `Case` is a
plain dataclass built per run, never a table. `dataset_usable()` is the gate: kind must be one of
`CASE_DATASET_KINDS` and a **messages column must be mapped** (`Dataset.messages_field`) — that's
what `GET /v1/experiments/datasets` filters on, and the reason an audio dataset shows up as
`usable=false` with a fixable reason rather than silently missing.
- A row's `tools` OR `functions` column becomes the tool declarations; `params` (or flat
  `temperature`/`top_k`/… columns) become the replay parameters; `expected` feeds the evaluators.
  Each is accepted as a real object OR a JSON **string**, because a CSV/JSONL round-trip gives
  either (`_coerce_json_cell`).
- **`excluded_rows` is honoured** — a row the owner un-ticked in the row browser is excluded from
  training, so excluding it here too keeps one meaning of "this dataset".
- `Experiment.n_planned` is an **estimate** at create time (`Dataset.num_rows` × the matrix); the
  true count is only known once rows are read, and the `MAX_UNITS` cap is re-checked at run time.
- A deleted dataset leaves its runs readable — `_dataset_names` renders `(deleted dataset)`
  rather than failing the row.

**Capturing INTO a dataset** (`langfuse_import.py` + the `/experiments/capture/*` routes). Both
importers now **create a real `kind=upload`, chat-shaped Dataset** (JSONL to the chosen S3
storage, `messages_field="messages"`) instead of writing to a private store — so a captured
corpus is browsable, publishable and packable like anything else. Two sources: a Langfuse trace,
and **this platform's own served traffic** (serverless `requests` rows store the whole OpenAI
body). ⚠ **Proxy-mode rows can't be captured**: `main._record_proxy_request` writes a deliberately
*slim* row (model + usage only, for throughput), so there's no body to replay; the route says so
rather than returning nothing.

Two Langfuse details are load-bearing because they fail **silently**:
1. **`traceId=` beats `peek=`.** In the newer UI `peek=` can be a *span* id that 404s against
   `/api/public/traces/{id}`.
2. **PII-scrubbed JSON-string inputs.** Some observations store the request as a JSON *string*
   whose scrubber left **unquoted** placeholders (`"created_at": <id>`). `json.loads` fails, a
   naive caller iterates the string, and the "request" becomes thousands of **single-character
   messages** with no error. `_repair_scrubbed_json` re-quotes placeholders only in structural
   value positions (never inside customer text), and `extract_request` **refuses** >500 messages
   or non-object messages outright. An AGENT span (`*.arun`) whose input is a task string gets an
   error naming its child GENERATIONs.
Also recovered: `modelParameters` (incl. a JSON-string `extra_body` → `top_k` /
`chat_template_kwargs.enable_thinking`), because replaying with library defaults reproduces a
*different* request than the one that misbehaved.

**Verified e2e locally** (2026-07-30) against a mock upstream with injected faults: 80 units
across 2 targets × 2 variants, every detector firing at its designed rate (control-token leak,
empty+0-token, fenced JSON, degeneration, HTTP 500), the clean target at 100% pass, cancel keeping
partial results, and all five web routes rendering the live data.

### Prompt optimization — GEPA (`prompt_opt.py` + `prompt_opt_api.py`)

Reflective prompt evolution (https://dspy.ai/getting-started/gepa-optimization/): replay the
dataset under a candidate prompt, score every reply, show the failures + their written critique to
a **reflection LM**, keep the rewrite only if it measurably wins, and sample the next parent from a
**Pareto frontier** so the search doesn't collapse onto one strategy. Section key `experiments`;
routes are `/v1/prompt-optimizations/*`; new table `prompt_optimizations` (created by `create_all`,
registered in `db.init_db`).

**The split is the point.** `prompt_opt.py` is the algorithm and touches nothing — it takes
`evaluate(texts, row_ids)` and `reflect(component, current, examples)` as injected coroutines, so a
whole search runs against a two-line fake in `tests/unit/test_prompt_opt.py`. `prompt_opt_api.py`
is the glue, and it deliberately reuses the experiment runner's own parts:
`resolve_cases()` → `build_request()` → `call_once()` → **`EvaluatorStack`**.

- **⚠ Not the `gepa` pypi package, and not dspy.** `gepa`'s *base* deps are litellm + mlflow +
  wandb + datasets + pandas + pyarrow, and its engine is sync (thread-bridging past our cancel flag
  and heartbeat). DSPy optimizes typed `Signature` fields; Experiments replays raw captured chat
  messages that have no field structure to bind to. The published algorithm is ~250 lines.
- **`EvaluatorStack` (in `experiments_api.py`) was extracted for this and is now used by BOTH.**
  Same detectors, same order, same short-circuit on transport errors, same one-child-process-per-run
  rule for python evaluators. Two scoring paths that drift would make an optimized prompt's "+14pp"
  unreproducible in the experiment meant to confirm it. Same reason `snapshot_evaluators()` is
  shared. **Verified**: a search reporting 0%→100% reproduced exactly as 0.00 / 1.00 in an ordinary
  two-variant experiment.
- **The evaluators ARE the metric, `reason` IS the feedback channel.** Score = weighted mean of the
  detectors that scored; the written half is their `reason` strings. So every built-in, custom and
  judge evaluator is a GEPA feedback source with no extra authoring.
  - **ALWAYS_ON diagnostics are excluded from the scalar.** `request_error` passes unconditionally
    on a successful request, so counting it would score a prompt that fails its only real check 0.5
    instead of 0 and halve every reported gain. A request that *did* fail is forced to 0.0.
  - **An abstention is not a pass** (`skipped` / `evaluator_error` / `judge_error` are dropped, not
    counted as 1.0). A dataset with no reference data makes every detector abstain → flat score →
    nothing beats the seed → a "completed" run that measured nothing. `result_json` carries
    **`unscored_rollouts` / `rollouts`** for exactly this, and the UI warns on it.
- **A candidate IS a variant.** Components map onto `VariantSpec` fields (`system_prompt` →
  **`system_override`**, a new field that REPLACES the row's system message where prefix/suffix
  decorate it; `user_suffix` → `user_suffix`). That's why the winner replays through the ordinary
  runner with no special case, and why `/experiments/new?prompt=opt-…` is one click.
- **⚠ Budget is denominated in REAL BILLED CALLS and enforced before each iteration**, covering its
  worst case (two minibatch passes + a validation sweep), so a run cannot overshoot what the user
  approved — at the cost of up to one unspent sweep at the end. Presets are multiples of the
  validation-set size (`AUTO_BUDGETS` light 6 / medium 15 / heavy 40), hard ceiling
  `PROMPT_OPT_MAX_METRIC_CALLS` (5000), rows capped at `PROMPT_OPT_MAX_ROWS` (200 — cost is
  rows × candidates, so a 2000-row corpus would spend everything on one sweep). Reflection calls hit
  a *different* endpoint and are counted separately.
- **⚠ The budget alone does NOT bound the loop — there is an iteration ceiling too.** Parent
  rollouts are cached (round-robin minibatches mean a surviving parent meets the same rows again),
  so a reflection model stuck re-proposing the current text hits the cache, is rejected before the
  child ever runs, spends **zero** metric calls, and spins forever billing the *reflection*
  endpoint. `max_iterations` = what the budget could pay for uncached. This hung the first test run;
  `test_a_zero_cost_iteration_cannot_spin_forever` pins it.
- **Pareto selection keeps a specialist alive**: per-row best scores → each candidate's winning-row
  set → drop strict subsets → sample proportional to breadth. Hill-climbing the mean walks into the
  first local optimum; that's the "genetic-Pareto" half of the name. The paper's **merge** across
  lineages is NOT implemented (nothing to recombine with one component); `Candidate.origin` exists
  so it can be added without a schema change.
- **Seed defaults to the dataset's own most common system message** (`GET /prompt-optimizations/seed`).
  Optimizing a prompt the rows were never captured under measures a different thing.
- Cancel is cooperative (finishes the turn, keeps the best prompt so far, status `cancelled`);
  restart cleanup is heartbeat-gated like experiments. `include_expected` shows the row's `expected`
  block to the reflector — powerful, and the main way a run overfits.
- **⚠ The gain is measured on the validation split, not held-out data.** With too few rows to split
  the minibatches reuse the validation rows (`in_sample: true` on the record) and the number is
  in-sample outright. Treat the confirmation experiment as the real measurement.
- **Verified e2e locally** (2026-08-02) against a mock upstream whose reply quality depends on the
  system prompt: seed 0.0 → best 1.0 in one accepted rewrite, budget respected (34/45), correct
  feedback text, cancel mid-run leaving a usable best prompt at 10/120 calls, both validation errors
  (no evaluators / half-filled reflection endpoint), and all three web routes rendering live data.

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
- **RTF (real-time factor) — response headers + `proxy_audio_rtf`.** Every audio forward reports how
  long the upstream took per second of audio: **`X-RTF`** (processing s / audio s — LOWER is faster,
  `< 1` = faster than real time), **`X-RTFX`** (its reciprocal, the "× real time" figure) and
  **`X-Audio-Seconds`** (the measured duration), plus the histogram `proxy_audio_rtf{proxy,model,kind}`
  (`kind` = `stt|tts` — their scales differ ~10×, so never merge them) and the counters
  `proxy_audio_seconds_total` / `proxy_audio_process_seconds_total`. ⚠️ **Aggregate RTF = the ratio of
  those two counters**, not the histogram's mean (which is a mean of per-request *ratios* — a 1 s clip
  would weigh the same as a 10 min one):
  `rate(proxy_audio_process_seconds_total[30m]) / rate(proxy_audio_seconds_total[30m])`.
  Duration comes from **headers only** (no decoder on the hot path): STT uses the upstream's own
  `duration` (verbose_json — already always requested) and falls back to the uploaded clip's WAV/PCM
  header; TTS parses the generated bytes (`_audio_duration_s`), and the *streaming* TTS relay derives it
  from the relayed **byte count** (`_pcm_bytes_duration_s`, engine PCM at `_TTS_PCM_SR`) since its tee is
  capped. ⚠️ **A compressed body (mp3/opus/flac/m4a) reports NO RTF** — measuring it needs a decoder, so
  `_record_rtf` omits the fields rather than guess. ⚠️ **Streamed requests get the metric but NO header**
  (response headers flush before the first chunk); the browser only sees them because the Next proxy
  route forwards `x-rtf`/`x-rtfx`/`x-audio-seconds` (`PASS_HEADERS`, incl. the *binary* branch — a TTS
  response is binary). Playgrounds render it via `components/playground/rtf.tsx`.
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

### LLM red-teaming guard — inline chat screening (`red_team` proxy config)

Per-endpoint guardrail on **both text-generation paths** (`_RT_GUARDED_PATHS`:
`/v1/chat/completions` **and** the legacy `/v1/completions`): every request is screened by a
detector **before any upstream byte is sent**; a positive verdict is answered by the gateway and the
model never sees the request. ⚠ It screened chat only until 2026-08-07 — the same model behind the
same endpoint answers `/v1/completions`, so re-sending an attack there walked straight past the
guard. Anything else (embeddings, rerank, audio) has no prompt to screen in this sense. Config is a `red_team` block on the
proxy (form card between Routing and STT callback), stored like `stt_callback`/`capture`
(`_build_red_team`, keys Fernet-encrypted / secret-ref'd, resolved+decrypted in `_route`).
⚠ `_build_red_team` REBUILDS the whole block from the spec on every save, so any field the
edit form doesn't send snaps back to its Pydantic default — `max_chars`/`timeout_s` were
silently reset to 8000/15 by a UI save until the form carried them. Add a form input with
every new `RedTeamSpec` field.

- **Two detector modes.** `classifier` → POST `{model, input}` to the classify route; the URL rule
  is `_rt_classifier_url`: a full `/classify` or `/moderations` URL is used verbatim, otherwise
  `/classify` is appended **after stripping a trailing `/v1`** (vLLM serves `/classify` at the
  server ROOT — an OpenAI-style moderation endpoint must be pasted in full). Parses BOTH response
  shapes (vLLM `data[0].label/probs` with `threshold` + `flag_labels`; moderations
  `results[0].flagged/categories`). `llm` → an **OpenAI-compatible chat-completions** call
  (`_rt_chat_url`: server root, `/v1` base, or the full `/chat/completions` URL all normalize —
  same forgiveness as the classifier URL); default prompt is GENERATED from the
  taxonomy ("reply `UNSAFE <category>` or `SAFE`"); `no_system: true` sends the scanned text alone
  for guard models with a baked-in template (Llama-Guard's `unsafe\nS9` parses fine — word-boundary
  regex, **UNSAFE checked before SAFE** since one contains the other; the S-code becomes the type).
  `reasoning` controls thinking on the judge AND responder calls (`disable` →
  `chat_template_kwargs.enable_thinking=false`, vLLM-style; `none|minimal|low|medium|high` →
  `reasoning_effort`) — a reasoning model left thinking can burn its whole `max_tokens` and return
  an EMPTY content field, which reads as an unparseable verdict → the on_error policy.
- **The taxonomy IS the contract.** `types` (default `RED_TEAM_DEFAULT_TYPES`: prompt_injection,
  jailbreak, system_prompt_extraction, harmful_content, pii_exfiltration) drives the judge prompt
  AND the category matching; every block carries `X-SGPU-Red-Team: flagged` +
  **`X-SGPU-Red-Team-Type: <type>`** (sanitized, `unclassified` when the detector names none).
- **Three blocking actions on a hit.** `respond` (default) = an OpenAI-shaped completion with the
  canned `message`, `finish_reason: "content_filter"`, SSE-shaped when the caller streamed (`_rt_sse`);
  `llm_respond` = a responder LLM writes the refusal (falls back to canned on ANY failure — a broken
  responder must never turn a block into a 502; responder defaults to the judge endpoint/key in llm
  mode, and is **required config in classifier mode** — validated at save); `error` = the configured
  `error_status` (default **403 — deliberately the proxy's "guardrail block, never failover" code**)
  with `{"error": {…, "red_team_type": …}}`.
- **`action=ignore` = MONITOR mode: the detector runs CONCURRENTLY with the model, never before
  it** (added 2026-08-12). Classify for observability, forward the request untouched. Two variants,
  `monitor_wait` (default **True**):
  - **`monitor_wait=True`** — `_rt_start_monitor` starts the classification as a task at gate time
    and `_dispatch_buffered` waits for its verdict **just before the response starts**, so the reply
    carries **`X-SGPU-Red-Team-Verdict: flagged|clean|error|pending`** (+ `-Type` on a hit). Cost is
    `max(0, judge − upstream first byte)`, NOT the judge's latency — the two overlap. ⚠ For a
    **stream** that required pulling the generator's first chunk as a task (`_rt_first_chunk`) while
    awaiting the verdict, because SSE headers flush before the body; the pulled chunk is replayed by
    `_rt_replay` so nothing is lost. Consequence: a streamed response's headers now arrive at the
    first upstream byte instead of immediately — that is the price of a verdict on a stream.
  - **`monitor_wait=False`** — fire-and-forget via `_rt_submit_monitor` → the shared bounded
    background queue (`_submit_bg`, same pool as the TTS evals / drift captures). **Zero** added
    latency, no verdict on the response, counter + log line are the whole record.
  - ⚠ **The verdict header is `X-SGPU-Red-Team-Verdict`, deliberately NOT `X-SGPU-Red-Team`.**
    The latter means "this gateway refused the request" and `experiments_api._guard_verdict` reads
    it to attribute a refusal to the guard instead of the model. In monitor mode the MODEL wrote the
    reply, so reusing it would drop a genuine model response out of scoring and credit a guardrail
    that let the attack through — exactly the confusion the guard-attribution work removed.
  - The wait is bounded by `min(timeout_s, RED_TEAM_MONITOR_WAIT_MAX_S)` (default 10 s) — a 30 s
    detector timeout is fine for a background classification, not for a blocked client. Over the
    cap the header says `pending` and the task keeps running under `asyncio.shield`, so the counter
    and the log line still land. **Never cancel the classification to satisfy the wait.**
  - `_rt_monitor_tasks` holds a strong reference to every in-flight task: asyncio only weakly
    references tasks, so without it a verdict can be GC'd mid-flight and the detection silently
    vanishes under load.

  What follows from the detector not gating the request, all deliberate:
  - **`on_error` is inert** — the request is already forwarded when the verdict fails, so there is
    nothing left to fail closed on. A failure is one WARNING + `result="error"` (and
    `X-SGPU-Red-Team-Verdict: error`). The form says so.
  - **No `blocked` ProxyRequest row** — the request genuinely completed. Experiments' guard
    attribution sees an unguarded target here, which is *correct*: the model really did answer the
    attack, so `model_refusal_rate` is the honest number.
  - **A full background queue SHEDS the classification** (counted `skipped` + a WARNING naming
    `PROXY_BG_WORKERS`/`PROXY_BG_QUEUE_MAX`) rather than pushing backpressure onto live traffic.
  - ⚠ **The WARNING line is the only path from a detection back to its request** (the counter
    carries no id), so `_job` re-attaches `accesslog.request_id_var` and spells `req=pxr-…` into the
    message. `_bg_worker` now CLEARS that var before every job for the same reason: a worker task
    inherits the context of whichever request lazily started the pool, so every off-path log line
    for the rest of the process was stamped with that one stale id (visible on the TTS-eval and
    capture jobs too — this fixed those as a side effect).
  - **The body is still buffered and parsed** — only the detector CALL moved off the path; the scan
    needs the messages, so the streaming-passthrough fast path stays bypassed.
  - `_build_red_team` does NOT require a responder for `ignore` (nothing writes a reply), and the
    stored `message`/`error_status`/responder fields survive untouched so flipping the action back
    to a blocking one restores the previous behaviour.
- **What's scanned**: `scan` = `last_user` (default) | `user` | `full`; multimodal content lists
  contribute their text parts; **TAIL-truncated** to `max_chars` (8k) — injections ride at the END
  of long context. A `/v1/completions` body has no `messages`, so `_rt_prompt_text` reads `prompt`
  as one turn (scan modes don't apply — there's only one); a **batched** `prompt` list scans EVERY
  element (one poisoned element is enough), and a token-id array is skipped, not stringified.
  Blocks on that path get the LEGACY body shape (`object: text_completion`, `text`, `cmpl-` id) —
  a chat-shaped body reads as an *empty* reply to a completions client, i.e. a silent block.
- **Failure policy is explicit**: `on_error: allow` (default, fail-open — logged + counted, request
  forwards) or `block` (fail-closed, type `detector_error`). An unparseable verdict (ambiguous judge
  reply, unknown classifier shape) is an *error*, not a pass/fail.
- **⚠ Red teaming forces the buffered path** — the streaming-passthrough fast path is skipped for
  chat on a red-teamed endpoint (the detector needs parsed messages), and detector latency is paid
  inline by every request. Blocked requests get ProxyRequest status **`blocked`**
  (`error_text = "red-team[<type>]: <reason>"`), visible in the Queue tab (new bucket/stat).
- **Metrics**: `proxy_red_team_total{proxy,model,result=safe|unsafe|error|skipped}`,
  `proxy_red_team_hits_total{proxy,type}` (attack-type breakdown),
  **`proxy_red_team_monitor_hits_total{proxy,type}`** (monitor mode's detections — flagged AND
  forwarded; deliberately a SEPARATE series so `hits_total`'s == `blocked` invariant below survives,
  and because the two are opposite outcomes: one blocked an attack, the other let it reach the
  model. It is the only record of the detection, hence the `ProxyRedTeamAttackDetected` alert fires
  with no `for:` window — in `deploy/monitoring/prometheus/alerts.yml` **and** the Helm
  PrometheusRule, the two-synced-files rule),
  `proxy_red_team_seconds{proxy,mode}` — all four included in the per-proxy
  `/proxy/{name}/metrics` exposition and rendered on the web Metrics tab
  (screened/safe/blocked cards + a flagged-by-type bar chart; monitor hits add a
  "Flagged, allowed" card and a second sky-coloured bar series, and only appear once
  the counter is non-zero so a blocking endpoint's tab is unchanged). ⚠ `blocked` is NOT an
  error in the tab's error rate — the guard working as designed isn't an upstream
  failure. ⚠ **The two counters measure different axes**: `_total` is the detector's
  VERDICT, `_hits_total` is the BLOCK. They diverge under `on_error: block`, where a
  fail-closed request counts `result=error` *and* `type=detector_error`. `_hits_total`
  is therefore bumped at the block point in `_red_team_gate` (`observe_red_team_hit`),
  not derived from `result="unsafe"` — deriving it hid fail-closed blocks from the
  breakdown exactly when the detector was down, and left `sum(hits)` short of the
  endpoint's `blocked` request count. That invariant — **sum(hits_total) == blocked
  ProxyRequests** — is what the Metrics tab's "Blocked" card and Queue tab agree on.
- **Dry run before saving**: `POST /v1/proxy/red-team/test` (admin) — the "Test detector" button on
  the form card. Runs the LIVE form values through `_rt_detect` (so URL normalization, key
  resolution, the reasoning toggle and both verdict parsers are the real ones) against **two**
  probes: a known attack (`RED_TEAM_TEST_ATTACK`, expect flagged) and a benign control
  (`RED_TEAM_TEST_SAFE`, expect passed). ⚠ The control is the load-bearing half — an
  attack-only test passes against a detector that flags EVERYTHING (wrong model, inverted
  classifier labels, a judge prompt that always answers UNSAFE), which would refuse all chat
  traffic the moment it's saved. The four verdicts are distinct messages: ok / over-blocks /
  doesn't flag / answered backwards, plus the raw detector error when it can't be reached.
  Key handling mirrors the form's three modes; `proxy_id` (sent for "keep existing") tests the
  key already stored on the endpoint, forcing `enabled: True` so a switched-off guard is still
  testable. Exercised live, not by a unit test — same as the upstream `POST /v1/proxy/test`.
- **Measuring the detector, not just smoke-testing it**: `POST /v1/proxy/red-team/evaluate`
  (admin) — the "Evaluate against a labelled dataset" panel under the test button. Takes a
  `dataset_id` and runs every row's scanned text through the SAME `_rt_detect` (shared resolver
  `_rt_detector_from_request`, so test and eval can never disagree about which detector they
  measured), bounded by `limit` (≤500, detector calls are billed per row) and `concurrency` (≤16).
  Ground truth is `expected.attack`/`expect_refusal` + `attack_type` — exactly what the synthetic
  red-team generator writes, so the corpus generator and the guard close the loop. Reports the
  confusion matrix, recall, **recall per attack category** (which class the guard is blind to),
  `type_accuracy` (did it also NAME the attack right), latency p50/p95, and the rows it got wrong
  — the tuning list. Two abstain rules, both load-bearing: a row with no label is **skipped, not
  guessed** (`scored` vs the row count), and **precision is `null` unless the corpus has benign
  rows** — on an attack-only corpus `fp` is 0 by construction, so `tp/(tp+fp)` would report a
  flawless 100% while measuring nothing. Measured on this platform: 30/30 recall across all five
  categories (EN+MS), p50 ~400 ms, `type_accuracy` 0.83; on a 16-row mixed corpus, precision 0.89
  (the one FP was a benign row *discussing* prompt injection).
- **Guard vs model attribution in Experiments** (`evaluators.py`): replaying a corpus through a
  red-teamed proxy, the guard's canned block matches every refusal pattern — so the MODEL used to
  score a refusal for a prompt it never saw. `Completion.guard_blocked`/`guard_type` (read from
  `X-SGPU-Red-Team*` by `experiments_api._guard_verdict`, on all four return paths incl. the
  `action=error` 403) short-circuits `_check_red_team`, and `_agg_red_team` adds
  `guard_block_rate` / `guard_over_block_rate` / `model_refusal_rate` / `model_saw_attacks`
  alongside the end-to-end `refusal_rate`. ⚠ "The endpoint is safe" and "the model is safe" are
  different claims — only the second survives turning the guardrail off. All guard keys are
  omitted entirely for an unguarded target, so old runs read exactly as before.
- Unit tests: `gateway/tests/unit/test_red_team.py` (scan extraction incl. the `/v1/completions`
  prompt shapes, both verdict parsers, both block-body shapes, builder validation + key handling,
  hits-at-block-point) and `test_evaluators.py` (guard-vs-model attribution + aggregate split). ⚠ `test_custom_eval`'s env-scrub test must RESTORE
  `PROVIDER_SECRET_KEY` (not pop it) or the red-team crypto tests fail suite-order-dependently.
- Restart rule: `proxy_api.py` is imported → **gateway restart** to pick up edits.

### Proxy failover — what moves a request to the next upstream

`_should_failover(u, status)` is the single decision point, used by every forwarder
(`_do_unary`, `_do_unary_multipart`, `_stream`, `_do_speech`, `_stream_speech`,
`_stream_audio`). A request moves to the next candidate on **`status >= 500`**, on a
transport failure (`ConnectError` / `ConnectTimeout` / `ReadTimeout` / `ReadError` /
`RemoteProtocolError`), or on one of the endpoint's **`failover_status`** codes.

- **`failover_status` defaults to `(402, 408, 429)`** — the three OpenRouter documents as
  *transient*: out of credits, request timed out, rate limited. The 4xx blanket rule used
  to be "never fail over", which meant a 429 went back to the caller while a healthy
  standby sat idle. Measured before the change: a 30-request burst at a `:free` model
  returned `{429: 28, 200: 2}` with all 30 pinned to the throttled upstream; after, `{200: 30}`.
- ⚠️ **`403` is deliberately excluded.** OpenRouter uses it for guardrail / moderation
  blocks — retrying that on another provider just re-triggers the block. Same for every
  other 4xx: those are caller bugs, and failing over multiplies them across backends.
- **A sub-500 failover does NOT mark the upstream dead** (`_mark_after_failover`). It is up
  and refusing, so a dead-mark would sink it behind its standbys for `HEALTH_TTL_S` (120s)
  and paint it red in the UI for something that clears in seconds. Cost: each throttled
  request pays one wasted round-trip before rolling over.
- **We do not honour `Retry-After`.** Trying the next upstream immediately beats sleeping
  when a standby exists. On a single-upstream endpoint the 429 is returned as-is and the
  hint is dropped — worth revisiting if one ever fronts OpenRouter alone.
- `>= 500` is not configurable. Note OpenRouter classes a bare `500` as *permanent* while
  502/503/504 are transient; we fail over on all of them, which is right here — a 500 from
  one backend says nothing about the next.
- **⚠ A non-failover error on the STREAMING path is now re-framed as an SSE `data:` frame**
  (fixed 2026-08-13). An upstream 4xx answers with a plain JSON body, not SSE, and `_stream`
  used to relay those bytes verbatim — producing an HTTP **200** `text/event-stream` whose
  body was bare JSON. Every SSE client (including this platform's own playground) reads only
  `data:` lines, so the caller saw an **empty stream and no error at all**. Found for real:
  an image request to an OpenRouter-backed endpoint returned
  `{"error":{"message":"No endpoints found for google/gemma-4-31b-it."}}` with 404 —
  non-stream showed it, stream showed nothing. The status is already committed to 200 by the
  time the generator runs (headers flush first), so the error is delivered the only way a
  stream can carry one: `data: {"error":…}` + `data: [DONE]`, and the row is marked `failed`
  with the real upstream status. A body that is ALREADY SSE-shaped is passed through
  unwrapped. ⚠ This branch only covers **non-failover** statuses — 402/408/429 and every 5xx
  exhaust the candidate list instead and hit the loop's own (already SSE-shaped)
  "all upstreams failed" frame. Tests: `test_red_team.py::test_stream_*`.
- Config lives on the endpoint (`failover_status` in the config JSON), is stamped onto each
  candidate by `_route` / `_resolve_speech_route` as `_failover`, and is surfaced in the API
  record + the "Fail over on status" field. `[]` restores the old never-fail-over-on-4xx
  behaviour.

### Rerank / score proxy (cross-encoders) — `/v1/rerank`, `/v1/score`

The LLM-proxy also fronts cross-encoder rerankers. `POST /proxy/{name}/v1/rerank`
(+ `/v2/rerank` and bare `/rerank` — the Jina/Cohere client spellings, all mapped onto the ONE
upstream path `/rerank`, mirroring what vLLM itself serves) and `POST /proxy/{name}/v1/score`
(+ `/score`). Both are plain unary JSON with a top-level `model`, so they ride the same buffered
engine as embeddings (`_handle` → `_do_unary`): alias→real rewrite, priority + failover, gate,
`X-Upstream-*` headers, request-history rows. Verified e2e against `tm-h20-reranker`
(Qwen/Qwen3-Reranker-8B). Nothing rerank-specific in the forwarding path — the only rerank-aware
code is the **upstream test probe** and the playground.

- ⚠️ **The failure mode is a silent HTTP 200, and it's why the probe asserts the RANKING.** vLLM
  never auto-applies a reranker's chat template, so a job started without `--chat-template` scores
  a bare query+document concatenation and still returns 200 — with meaningless scores. Signature:
  everything bunched in ~0.3–0.85 with no separation, often the relevant doc NOT first (a relevant
  doc once ranked last at 0.318 under an unrelated one at 0.851). That is **not** a borderline
  match. Real scores are calibrated hard toward 0/1 — relevant ~0.9+, irrelevant ~0.0.
- **So `mode=rerank` on `POST /v1/proxy/test` sends THREE documents** — one relevant, one
  same-domain-wrong-intent, one unrelated — and fails the upstream if the relevant one isn't
  ranked #1, warning on a top-to-next gap < 0.2. A single-document smoke test passes even when the
  template is broken, which is exactly the trap. The playground surfaces the same gap + warning.
- **`instruction` genuinely changes the ranking**, it isn't cosmetic — the default is generic
  web-search retrieval; override it to score for a domain task (churn intent, policy match,
  triage). Passed through verbatim like any other body field.
- **Serverless (`/{app_id}/v1/...`) has NO rerank** — that data plane goes through the worker
  queue, which has no rerank job type. This is proxy-only; the playground mode lives in the proxy
  playground, not the serverless tabs.

### ASR augmentation — the LiveKit/WebRTC transport family (`training/whisper_finetune.py`)

Autotrain ASR's `augment_techniques` gained a second family (2026-08-05) that models the
**transport a voice agent puts in front of the model**. Motivating symptom: a Whisper finetune with a
good held-out WER was **terrible** once served as an OpenAI-compatible API behind LiveKit. That is not
a serving bug — LiveKit never hands the model the microphone signal, only what survives a long lossy
chain, and a finetune on clean audio has seen none of it.

**The chain, read out of the LiveKit sources (not guessed) — versions current 2026-08:**

| where | what | source |
|---|---|---|
| browser capture | `autoGainControl`, `echoCancellation`, `noiseSuppression`, **`voiceIsolation`** all default **true** | `client-sdk-js/src/room/defaults.ts` |
| publish | Opus, `audioPreset: music` (**48 kbps**), `dtx: true`, `red: true` | same file |
| presets | telephone 12k / speech 24k / music 48k / musicStereo 64k / …HQ 96k / …HQStereo 128k bps | `client-sdk-js/src/room/track/options.ts` |
| agent | `rtc.AudioProcessingModule(auto_gain_control=True)` on **every inbound frame** — a SECOND AGC — then optional Krisp BVC, then `rtc.AudioResampler` (default quality MEDIUM) | `agents/.../voice/room_io/_input.py` |
| STT plugin | `SAMPLE_RATE = 24000` → the OpenAI-compatible endpoint is fed **24 kHz** WAV, resampled at quality HIGH | `livekit-plugins-openai/.../stt.py`, `agents/.../stt/stt.py` |
| VAD | Silero `min_speech_duration 0.05`, `min_silence_duration 0.55`, `prefix_padding_duration 0.5`, `activation_threshold 0.5`, 16 kHz | `livekit-plugins-silero/.../vad.py` |
| SIP | trunks negotiate **PCMU/PCMA (G.711, 8 kHz)** by default and LiveKit SIP transcodes that leg to Opus → a phone caller is a **codec tandem** | LiveKit telephony docs |

**Techniques:** `livekit` (the whole chain in order, per-sample randomised — *this is the one to use*),
`livekit_sip` (the telephony variant), and the isolatable stages `opus`, `g711`, `packet_loss`, `dtx`,
`agc`, `webrtc_ns`, `aec`, `vad_clip`, `resample_chain`.

- **⚠ `layout="mono"` in `_opus_av` is load-bearing and fails SILENTLY.** A PyAV audio stream defaults
  to **stereo**; a stereo libopus stream splits the bitrate budget across two channels, so
  `stream.bit_rate` stops tracking and every "bitrate" produces the same distortion. Measured before
  the fix: 48000 requested → **22.6 kbps actual**, and 12k/24k/48k all landed within 0.2 dB of each
  other. It looks like it works. Verify a codec change by checking encoded **bytes/second**, never by
  eyeballing the waveform.
- **⚠ The opus round-trip is REAL, with a loud fallback.** `_opus_backend()` resolves once: PyAV
  (in-process libopus) → ffmpeg CLI → a **DSP approximation**. `av` is in the trainer's venv `pkgs` +
  `check_mods`, so an older venv is reconciled on the next run. The DSP path logs a WARNING because a
  band-pass filter that is not a codec would look exactly like success in the metrics.
- **`compression_level=5`, not ffmpeg's default 10.** Maps to `OPUS_SET_COMPLEXITY`; libwebrtc uses 9
  desktop / 5 mobile. Measured identical distortion (−7.4 vs −7.6 dB residual, corr .908 vs .911 at
  24 kbps) for ~30 % less encode time — and the encode is ~95 % of the chain's cost.
- **⚠ `vad_clip` trims SILENCE only, deliberately.** Real VAD segmentation does clip words, but
  cropping speech while keeping the full transcript is exactly how you train a model to **hallucinate
  the missing words**. It tightens leading/trailing silence to a random 0–400 ms and leaves hard
  (un-faded) edges. Every technique here is label-preserving; keep it that way.
- **Cost: budget dataloader workers.** Steady-state on one core for a 7 s clip: most techniques
  0.1–13 ms, but `opus`/`livekit`/`livekit_sip` are **~130–210 ms** (PyAV) / ~80–100 ms (ffmpeg CLI) /
  ~25 ms (DSP). At `augment_prob=0.5` that is ~100 ms per sample amortised — raise
  `dataloader_num_workers` or the GPU starves. `_LazyAsrDataset.__getitem__` runs in the workers, so
  it parallelises.
- **⚠ Four registries must stay in sync**, else a technique is silently dropped from the run config:
  `whisper_finetune._AUG_FUNCS` (the trainer), `training_api._AUG_TECHNIQUES` (validation — request
  values not in this set are **filtered out silently**), `training-form.tsx` `AUG_OPTIONS` +
  `LIVEKIT_AUG_OPTIONS` + `STREAM_AUG_OPTIONS` (the UI), `automation/run_pipeline.py`
  `ALL_AUGMENTATIONS` + `LIVEKIT_AUGMENTATIONS` + `STREAM_AUGMENTATIONS`.
  `gateway/tests/unit/test_audio_augment.py` pins **all four** (the form test scrapes the three
  `*_OPTIONS` consts by name — a fourth UI group needs adding to that regex or its ids read as
  missing).
- **Restart rules**: `whisper_finetune.py` is **SFTP'd per run** → trainer edits need no gateway
  restart. `training_api.py` is imported → its `_AUG_TECHNIQUES` edit **does**.
- **`resolve_augment` aliases** (automation YAML): `all` = the 8 classic techniques, `livekit_all`
  = the 11 transport ones, `stream_all` = the 7 streaming-regime ones, `voice_all` = both voice-agent
  families. Each alias keeps its meaning as techniques are added — **one technique is picked per
  augmented sample, so growing `livekit_all` would silently halve how often an existing YAML's
  transport stages fire**. Plain `livekit` / `livekit_stream` are real techniques, not aliases.
- Verified against a real speech clip: opus distortion falls monotonically with bitrate, G.711 leaves
  <1 % energy above 4.2 kHz, PLC conceals instead of zeroing, AGC's 3 dB/s slew leaves the clip head
  15 dB under the tail, and 20 random draws of both chains stay finite/unclipped on all three codec
  backends.

#### The streaming regime — the VAD segment + a hesitant speaker (added 2026-08-06)

A second wave, driven by what the out-of-tree LiveKit STT benchmark (a live `livekit-server`, real
WebRTC/Opus, silero VAD, the `livekit-plugins-openai` STT, graded by the same scorer as its batch
arm — maintained separately, and the source of truth for these figures) actually **measured**:
**the transport is not where the accuracy goes.**

| what | WER |
|---|---|
| channel alone (codec + noise + gain) | 4.98 % → 5.33 % (**+0.35 pp**) |
| + the streaming pipeline, fluent speech | 5.33 % → 7.23 % (+1.90 pp) |
| + the streaming pipeline, **hesitant** speech | 6.64 % → 13.39 % (**+6.75 pp**) |

Everything in the family above buys the first row. **Techniques:** `livekit_stream` (the transport
chain plus all of the below — *the one to use for an agent that talks to people*), `hesitation`,
`vad_pad`, `rampup`, `denoiser`, `room_tone`, `jitter`.

- **`hesitation` fixes a MODEL weakness, not a pipeline one.** The benchmark's *batch* arm — no
  LiveKit anywhere in the path — degrades 5.33 % → 6.64 % when two 0.7 s pauses are inserted
  mid-utterance. That 1.31 pp is trainable here. It deliberately uses the **same selection rule** as
  the benchmark's `--pause-count/--pause-duration` (20 ms energy grid, lowest-energy frames first,
  ≥ 0.5 s apart, outer 15 % avoided), so the hesitant benchmark arm measures the condition training
  saw; durations 0.3–1.2 s straddle silero's `min_silence_duration` (0.55 default / 0.9 tuned) so
  some pauses split the turn and some don't.
- **⚠ `vad_pad` is the OPPOSITE of `vad_clip`, and it is the common case.** A turn silero keeps
  whole arrives *framed* — `prefix_padding_duration` of pre-roll + the `min_silence_duration`
  hangover, measured as a 3.95 s utterance reaching the STT as a **4.92 s segment**. Only a *split*
  segment arrives cut tight. Both ends of that axis are now covered; a model that has only seen
  tightly-cut clips answers ~1.4 s of non-speech with phantom text (the benchmark's #2 suspicion for
  the production gap). Pre-roll is filled with the clip's **own noise floor** (`_room_tone`), not
  digital zeros — only DTX ever hands the model true stationary hiss.
- **⚠ Anything that inserts time asks `_room_left()` first.** Whisper's feature extractor truncates
  at **30 s**, so lengthening a clip past it pushes real speech out of the window while the label
  still claims every word — the `vad_clip` hallucination lesson arrived at from the other direction.
  `hesitation` shrinks/drops pauses to fit, `vad_pad` caps the synthesised padding, `livekit_stream`
  stays under 30 s across draws (verified on a 29.2 s clip). ⚠ The **pre-existing `speed` technique
  has no such guard** — `time_stretch(0.9)` on a > 27 s clip can overflow the window. Rare enough
  that it was left alone; don't add a length-increasing technique without the guard.
- **⚠ `denoiser`'s `gate_dbfs` is an ABSOLUTE target level, not another attenuation.** Attenuating
  non-speech by a further N dB *stacks* with `_spectral_suppress`'s floor: the first calibration
  landed the floor at **−90 dB**, ~20 dB below anything a real enhancer produces. The measured GTCRN
  signature is a floor **at** ≈ −70 dBFS with speech peaks held within 0.2 dB, so the gate scales
  non-speech to hit that (and never amplifies). Re-verified on a real dumped segment: −24.7 → **−71
  to −72 dB** p25 frame level, peak delta **0.00 dB**.
- **Why train on a denoiser that LOST at serving time.** GTCRN measurably works (floor −28.9 →
  −70.9 dB, speech peaks within 0.2 dB) and still lost WER on **every** benchmark arm (7.23 → 7.36
  fluent, 13.39 → 15.13 hesitant, 11.00 → 12.52 noisy) — whisper tolerates additive noise better
  than enhancement artefacts and answers dead-quiet non-speech with invented content. The serving
  conclusion is *don't enable it*; the training conclusion is that a model which may end up behind
  one should have seen its artefacts. Much more aggressive than `webrtc_ns`, which stays the
  browser's own NS/voiceIsolation.
- **`rampup` is attenuation only, never deletion.** Found in the harness: publishing a clip that
  starts talking at t=0 lost or corrupted the **first word on 80/100 clips** (`Ya saya ni` → `Saya
  ini`) — Opus/jitter-buffer priming after subscription plus silero's prefix padding having no
  pre-roll to prepend. The harness pads a second of silence in front precisely so it stops measuring
  this; production still has it at session start. A *hole* where the onset phoneme was, with the word
  still in the label, is the hallucination lesson again — so 6–24 dB of fade, no zeroing.
- **⚠ `jitter` is MODELLED, not measured.** NetEq's accelerate / pre-emptive-expand is unambiguously
  in the receive path, but the benchmark runs livekit-server on **localhost** where there is no
  jitter to absorb, so it can neither see nor price the artefact. Bounded to ≤ 2 % of the clip and
  applied at its lowest-energy windows (roughly where NetEq's correlation search lands), so nothing
  merges a phoneme away.
- **⚠ Over-segmentation is NOT modelled and cannot be here.** It is the *rest* of the streaming cost:
  silero splits the turn, each fragment is transcribed with no shared context, and a number
  straddling the boundary is destroyed (`charged 99, two times` → `charged 92 times`). Clips silero
  kept whole cost 0.35 pp; the 12/100 it split were 14.5 % of the words but ~⅔ of the extra errors.
  Modelling it means cutting the audio at a segment boundary **and cutting the transcript with it**,
  which needs word-level alignment — a dataset transform, not a label-preserving waveform augment.
  Serving-side the measured fix is the VAD's own `min_silence_duration` 0.55 → **0.9** (halves the
  fluent cost, cuts the hesitant cost by two-thirds, ~0.35 s more end-of-turn latency).
- **`livekit_stream` is a separate technique, not folded into `livekit`.** Same reason `all` never
  grew: an earlier run's augmentation stays reproducible. Select both if you want the
  plain-transport draws too.
- **Cost** (real 16 kHz speech, one core): `jitter` 0.1 ms, `rampup` 0.2, `vad_pad` 0.6,
  `hesitation` 1.2, `room_tone` 4.8, `denoiser` 10 — and `livekit_stream` **~100 ms** (vs `livekit`
  ~81 ms), still dominated by the opus encode. Same dataloader-worker budgeting as above.
- **Recommended**: `augment_techniques: ["livekit_stream"]` at `augment_prob` ~0.5 for a
  human-facing voice agent (add `livekit_sip` if it takes phone calls). Verified on real segments
  the benchmark had uploaded to the STT: 20 draws finite/bounded/under 30 s, room tone at the
  clip's own floor, jitter warp within ±1.22 %, `room_tone` hitting its requested SNR exactly.

### kind=hf audio import — subset scoping + embedded (`Audio`) clips (`dataset_transform.py`)

The HF→S3 transform originally assumed one repo shape: audio files in archives, referenced by a
**path column** in a metadata table. Two additions (2026-08-07) cover the other common shape and
stop a whole-repo pull:

- **Subset scoping.** `TransformRequest.hf_subsets` (or, persisted, `Dataset.hf_subsets` — an
  idempotent-ALTER JSON column set by `/datasets/new?source=hf`) restricts the run to some of the repo's
  **declared** configs/splits. `_declared_entries` parses the README `configs:` front-matter into
  `{label, config, split, glob}` — `label` being the SAME config-vs-split name `_hf_split_ident`
  gives the row browser, so a name picked in the UI resolves here. `_resolve_hf_subsets` matches by
  label, by bare config (= all its splits), or `config/split`, and returns the globs → the README is
  fetched **first** (`_fetch_declared_entries`, one `hf_hub_download`) so they become
  `snapshot_download(allow_patterns=…)`. ⚠ **An unmatched name RAISES**, listing the available
  labels: selecting nothing would materialise an empty dataset and selecting everything would pull
  the configs the caller was excluding. The `repo_info` byte total is filtered by the same patterns
  or the download % stalls partway.
- **Embedded audio.** An HF `Audio` feature stores the clip INLINE (`struct<bytes, path>`), so the
  old path — `isinstance(ref, str)` → join to a file on disk — skipped every row and the transform
  died with "no metadata rows matched an extracted audio file". `_table_with_embedded_audio` detects
  the struct via the arrow schema and writes each clip under `work/_embedded/{label}/`, replacing the
  column with that path so everything downstream is unchanged. Extension comes from the struct's own
  `path` when it's a known audio suffix, else sniffed from the container magic (`_audio_ext`).
  - ⚠ **Streamed in `_EMBED_BATCH_ROWS` (64) batches, not `_load_table`.** A shard of 24 kHz speech
    is ~0.5 MB/row, so reading one 500 MB parquet into pandas would put half a gigabyte of audio in
    the gateway heap — per file.
  - ⚠ **The subset label MUST be in the clip filename.** `_materialise_s3` keys S3 objects by
    BASENAME, and this shape repeats both the shard stem (`train-00000-of-00005`) and the inner
    `path` (`gen00047.wav`) across configs — so two configs' clips would silently collapse onto one
    object. Names are `{label}_{shard}_{row}{ext}`; a unit test pins the uniqueness.
  - `_read_metadata` resolves each table's split label BEFORE loading it, so an unselected (or
    undeclared) multi-GB shard is never read.

**⚠ The source repo's token/endpoint must never come from the MIRROR storage** (`_hf_source_store`,
fixed 2026-08-09). A `kind=hf` dataset usually has `storage_id=None`, and the transform borrowed
"any huggingface Storage" — `.limit(1)`, which on this deployment is the self-hosted mirror
(`endpoint=http://localhost:8080/hf` + an `sgpu_` key). Every file of a public huggingface.co repo
then 404s, and the first casualty is the README: `_fetch_declared_entries` returned None, which the
caller reported as **"<repo> declares no configs in its README"** — for a repo whose README declares
18. The row browser was unaffected the whole time because `datasets_api` passes **no storage** for a
storage-less dataset, which is why the same subsets picked fine and only the transform failed. The
fallback now skips any storage carrying a custom `endpoint` (the dataset's OWN huggingface storage
still wins, endpoint and all — that's a repo genuinely hosted on the mirror), and no huggingface
storage at all is a valid answer: public HF + `HF_TOKEN`. Same helper on the merge + llm-pack paths.
`ReadmeUnavailable` now separates **"couldn't read the README"** (names the endpoint + the real
error) from **"the README declares nothing"** — conflating them is what sent the last debug at the
repo instead of at us. Unit tests in `test_dataset_transform_hf.py`.

**⚠ A stored scope has to be honoured by the READ paths too, not just the transform.** The row
browser and the `/splits` picker enumerate the repo live off the datasets-server and used to
default to **`splits[0]`** — so a dataset scoped to `synthetic` opened on `default/train`, a
text-only chat config whose every audio/transcription cell is null, which reads as "the subset
setting did nothing" (reported from the UI). `_dataset_scope` + `_hf_resolve_scope` now filter
`GET /{id}/splits` and supply the preview's default selection; both share
`dataset_transform.subset_matches` with the transform, so label / bare-config / `config/split`
mean the same thing in all three. A scope matching nothing (free text, and the repo can change
under it) is an **error naming the available subsets**, never a silent widening back to the whole
repo. ⚠ `GET /{id}/splits?all=1` is the escape hatch the scope EDITOR uses: every subset with
`in_scope` flags, and no error on an unmatched scope — a filtered listing can only narrow a scope,
and erroring would lock the user out of the control that fixes it. The preview clamps an explicit
`?split=` to the scope too (a URL or a remembered selection outlives a narrowing). Note the preview's `_stamp_detected_fields` self-heal still picks the column mapping from the
rows it sees — on a TTS corpus that lands on `transcription` (the readback), so set
`transcription_field` by hand if the script column is the label.

**⚠ `_materialise_s3` drops carried-through columns that collide with the three it owns**
(`audio` / `<transcription_field>` / `split`). Emitting both put a DUPLICATE header in the metadata
CSV and every reader (csv.DictReader, pandas) keeps the LAST — silently replacing the real label
with the passenger value. Live case: a source labelled by `text` that also carries an ASR-readback
`transcription` column, merged into an output whose transcription column IS `transcription` — which
would have blanked every *other* source's transcript too (they have no such extra).

**⚠ The same collision bit the READ path, in the opposite direction** (fixed 2026-08-09). When the
output's transcription column is `text`, a passenger column literally named `transcription` is
legitimately written — and `preview_dataset` built each row as `{"transcription": r.get(tf), **r}`,
so the `**r` spread (last) let that empty passenger overwrite the value just resolved from `text`.
Every transcript in the row browser read blank while the CSV was perfectly fine (hit on
`synthetic-tts-user-v2-audio`, transformed out of `Scicom-intl/Synthetic-User-Turn-TTS`). Both
branches now spread `**r` FIRST and assign the computed `audio_url`/`transcription`/`row_index`
after — the rule the chat (`messages_field`) branch already followed for the same reason.

### Merging s3 sources — REFERENCE in place, and where the key comes from

A same-account s3→s3 merge (`_run_merge` → `_s3_copy_pairs(reference=True)` → `("s3ref", …)`
markers) writes a metadata CSV pointing at the source objects **where they already are**. It used
to server-side copy every clip: merging two 13k-row exports duplicated 26,283 objects to produce a
file whose only real content is the row list. Now nothing is moved and the merge takes seconds.
Consequences to keep in mind:

- **The merged dataset owns only its `metadata.csv`.** `DELETE ?purge=true` on it therefore
  deletes just that (verified: 1 object) — but purging a SOURCE now breaks the merge. Same tradeoff
  the normalize output already carries.
- **⚠ A referenced clip is NOT renamed, so basenames must stay unique across sources.** The
  `s{idx}-` prefix that guaranteed that is a property of the COPY. Nothing downstream keys on the
  URL alone — `whisper_finetune` caches each download by basename — so two sources holding
  different clips under one name would train the second row on the first row's audio. `_run_merge`
  therefore checks `_pair_stem` uniqueness across all pairs and, on any clash, **downgrades the
  whole merge back to copying** (the marker tags swap; the 4th element is the copy name, which is
  why references carry one). Verified both ways: distinct names → 0 copies, 1 file in the output
  folder; same names → the `s0-`/`s1-` copy path and 200 unique basenames.
- References may span buckets in one account, so `_materialise_s3` presigns **grouped by bucket**.

**⚠ The audio key comes from the CSV cell's own URL (`_s3_key_from_ref`), never from the metadata's
folder.** `{metadata_dir}/audio/{basename}` only holds when metadata sits next to its audio, and a
**normalized** dataset deliberately breaks that: its CSV lives in `normalized-<hex>/` and references
the PARENT's `audio/`. Merging one guessed a key one directory too deep and died with
`NoSuchKey` on CopyObject — after copying the other source's 13k objects (hit in prod on
`asr-merged-v3-normalized`). The URL's path IS the key; virtual-host vs path-style is
disambiguated by the host, and the reconstructed key survives only as the fallback for a bare
filename. `_s3_pairs` (the download path) reads the key the same way; it had been surviving on its
"fall back to fetching the raw URL" branch, which `_s3_copy_pairs` has no equivalent of.

**⚠ A subset's split label is `config/split`, so a HF `test` subset is NOT eval.** The trainer's
`EVAL_SPLITS` is `{test, validation, valid, eval, dev}` and `synthetic/test` matches none of them —
it trains. Carve eval with `test_split_pct`/`test_split_count` instead.

**Verified e2e locally** (2026-08-07) against `Scicom-intl/Synthetic-User-Turn-TTS` (audio inline,
5 configs): `synthetic/test` alone selected → only that config downloaded, 100 clips materialised to
S3 with the right split label and `text` (the TTS script) as the transcript, not `transcription` (a
Whisper readback of the same clip); then merged with a 188-row label-derived dataset → 288 rows, one
`transcription` column, neither half blanked. Unit tests: `gateway/tests/unit/test_dataset_transform_hf.py`.
Automation side: `automation/run_pipeline.py` grew an `hf_repo` dataset entry (see its README) and
`automation/config-v4.yaml.prod`.

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

### Storage file viewer (`/storage/{id}/files` → `storage_api.py` browse/object routes)

A read-only browser over a **`kind=s3` or `kind=local`** storage (`BROWSABLE_KINDS`; huggingface
browses on the Hub, sftp would need a connection per listing): `GET /v1/storage/{id}/browse` (one
**delimited** `list_objects_v2` / one `readdir` per directory — folders + files + a continuation
token, O(page), NOT the full-bucket walk `usage`/`purge-scan` do), `GET /{id}/object` (bytes through
the gateway for the preview pane, `?download=1` for the uncapped fetch) and `GET /{id}/object-url`
(presigned, s3 only). All are **admin-only**, like usage/cleanup — one storage row backs every
feature's data. The s3 primitives (`s3_list_page` / `s3_head` / `s3_get_head_bytes`) live in
`bench.py` with the rest.

- **Paths are relative to the storage's own scope** — the configured `prefix` (s3) or `path`
  (local) — and that scope is a hard boundary. `_safe_rel` rejects `..` outright rather than
  resolving it; for local, `_local_path` **realpaths both sides**, because a symlink inside the
  root has no `..` and would otherwise walk straight to `/etc`. Escaping symlinks are also
  **hidden from listings** (`_escapes_root`, which only pays a realpath for entries that ARE
  symlinks), so the viewer shows the configured folder and nothing merely reachable from it.
- **⚠ Nothing here may build or sort a whole directory** — a million objects in one prefix is
  normal (per-clip audio). S3 pages on its continuation token; local pages on an offset token and
  **only name-sorts under `LOCAL_SORT_MAX` (20k)** — above that it streams in readdir order,
  stat()ing just the page, and returns a `note` saying it isn't sorted. Measured: 22 ms for a page
  of a 25k-entry directory, 45 ms at offset 24000.
- **The filter is a server-side name PREFIX (`q`), not a client-side substring.** Prefix is all
  `list_objects_v2` can push down without scanning, and filtering the loaded page would answer the
  wrong question on a directory the UI has only seen 300 of. Local applies the same rule during its
  scan so both kinds behave identically (and a filtered local listing drops back under the sort
  threshold, so it comes back sorted).
- **Sorting by size / modified is SERVER-side too, over a capped scan** (`sort=name|size|modified`
  + `order=asc|desc`, added 2026-08-07; sortable column headers in the UI, state in `?sort=`/
  `?order=` so a sorted view is linkable). Same reasoning as the filter — re-ordering the 300 rows
  the browser happens to hold would rank the wrong set — but neither backend can push it down:
  **S3 LIST only ever returns keys in ascending lexicographic order**, and readdir has no order at
  all. So `sort=name&order=asc` is the native path (unchanged, strictly page-bounded) and
  *everything else* reads the directory via `_s3_scan_dir` / `_local_scan_dir`, capped at
  `BROWSE_SORT_SCAN_MAX` (20k), sorts in `_sort_entries`, and pages by an **offset** token (so a
  later page re-scans — the same trade `_local_browse` already makes, and the server keeps no
  per-client state). Measured: 0.1 s on a ~250-object day directory, 1.3 s on a few-thousand-object
  prefix. Three rules the tests pin:
  - **Over the cap, `note` says the ranking is partial** — "biggest file" over the first 20k keys of
    a million-key directory is not the biggest file, and a silently-truncated ranking reads as an
    answer.
  - **Folders lead and are always name-ordered.** An S3 "folder" is a CommonPrefix — no size, no
    LastModified. A local directory *does* have an mtime, but letting it rank would make the two
    kinds behave differently for the same click.
  - **A missing value sorts LAST in both directions** (unranked, not "the smallest"). `modified`
    compares as a string: both producers emit UTC ISO-8601, so lexicographic *is* chronological.
- **⚠ A delimited S3 page can be EMPTY while more remain** — MaxKeys counts keys *scanned*, and a
  page whose keys all collapse into already-returned folders returns nothing. `_s3_browse` pulls up
  to `S3_EMPTY_PAGE_RETRIES` pages before handing the UI a blank screen with a "load more" button.
- **Download splits by kind**: local streams off the gateway's disk (`FileResponse`, forced
  `octet-stream` so the web proxy takes its byte-exact binary branch), s3 **307s to a presigned
  URL** so a multi-GB checkpoint never crosses the control plane.

- **⚠ S3's stored `Content-Type` LIES — the filename decides the type.** It's arbitrary
  caller-supplied metadata: this platform's own bucket stamps `.jsonl` objects
  **`application/jsonl`** (no whitelist calls that text) and extensionless files
  `application/octet-stream`. Trusting it made a 21 KB JSONL un-previewable (413 "too large" on a
  text file). `_media_type_for(name)` wins; the stored type is only the fallback for an extension
  we don't know — and those get **sniffed** (`_looks_textual`: no NUL + valid utf-8, tolerating a
  codepoint split at the tail of a ranged read), which is how `.persist_marker` previews. A known
  extension is NEVER re-typed by sniffing: a `.wav` whose head happens to decode is still a `.wav`.
- **Text is head-read, media is all-or-nothing.** Over `max_bytes` a text file serves its first
  chunk with `X-Object-Truncated: 1`; binary is refused with 413 pointing at the download (half a
  WAV is not a smaller WAV). `X-Object-Size`/`X-Object-Truncated` only reach the browser because
  they're in the Next proxy's `PASS_HEADERS` — add response headers there or the UI can't see them.
- Previews go **through the gateway** (same-origin + authed) so they work with no bucket CORS
  policy; downloads are presigned so a multi-GB checkpoint never streams through the control plane.

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

**⚠ Correction (verified 2026-07-22): the other archs do NOT carry `{% generation %}`.** The claim
above that qwen/minimax/mistral templates "already carry it" is false — rendered each with
`return_assistant_tokens_mask=True` and transformers warns + returns an all-zero mask, so
**qwen3.6 / MiniMax-M2.7 / Mistral-Small-4 have always packed FULL-SEQUENCE labels** (tokenize_row's
fallback). That's also why none of them ever exhibited gemma's tool-call loop: their stop tokens
(`<|im_end|>` / `[e~[` after `</minimax:tool_call>` / `</s>`) are trained for free under full-seq.
If anyone adds generation-masking for those archs, the stop-token-after-tool-calls lesson below
applies to them too.

**⚠ The silent-degrade path FIRED for real (found 2026-07-21):** Google updated gemma-4's upstream
template and both the reasoning guard and 2 of the 4 generation-mask anchors stopped matching —
every gemma pack since silently trained full-sequence labels with history-reasoning stripped, and
the resulting finetune regressed hard (part of the Tool-F1 0.918→0.481 post-mortem). Fixed:
`_REASONING_GUARDS` is now per-arch **candidate lists** (new anchor first, old kept as fallback,
then other archs' candidates), and the `_GEMMA_GENERATION_REPS` B (tool_calls-open, now
`message.get('tool_calls')`) + D (turn-end, now with `and not next_nt.found`) anchors match the
live template. **After any gemma pack, check `assistant_masked_rows` ≈ row count** — a collapse
to 0 means Google moved the template again; update the anchors, don't trust the WARNING alone.

**⚠ Masking MUST include the model's STOP token (anchor E, found 2026-07-22 the hard way):**
gemma-4 stops after tool calls by emitting `<|tool_response>` (it's in generation_config
`eos_token_id` alongside `<eos>`/`<turn|>`) — in transcripts that token doubles as the env's
response opener, so assistant-only masking left it label=-100 and the finetune NEVER LEARNED TO
STOP calling tools (identical call looped 12x+ per turn; parallel-call examples trained
`<|tool_call>`-after-`<tool_call|>`, making another call the preferred continuation). Anchors
E1/E2 wrap the tool-response opener in `{% generation %}` so the stop is a trained target.
General rule: for ANY masked arch, every token in `eos_token_id` that terminates an assistant
span must be INSIDE the mask. The pack card also has a **full-sequence-labels** option
(`LlmPackRequest.full_seq_labels` → `_llm_pack.full_seq_labels` in the packed metadata) — the
escape hatch that skips masking entirely (pre-07-10 behaviour, structurally immune to this bug).

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

**⚠ Pack shard-upload concurrency is capped at 6 (`dataset_transform._upload_chinidataset_dir`,
`max_workers=6`, was 16 — fixed 2026-07-21).** 16 concurrent HTTPS/TLS handshakes to S3 hit a
genuine thread-safety bug in this Python build's bundled OpenSSL (`SIGSEGV` in `X509_verify_cert`
on a background pthread — two identical macOS crash reports, the whole gateway process died
mid-pack both times, leaving the dataset stuck at `transform_status="running"` with no automatic
un-sticking; reset the row via direct DB write before retrying). Don't raise it back for speed.

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
