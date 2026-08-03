# Claude project guide — Serverless-GPU

## Local dev is set up. Use it. Don't push to prod just to test.

The full stack runs on the user's laptop. **Default to iterating locally** before
suggesting any deploy/PR. Only push when the user explicitly asks, or when the
change genuinely needs to run in-cluster (e.g. SealedSecrets, ArgoCD wiring).

### Where the rest lives (nested CLAUDE.md — load lazily when you edit there)

This root file is the always-loaded, cross-cutting stuff. Area-specific gotchas live next to the
code and are pulled in automatically only when you touch that subtree:
- **`gateway/gateway/CLAUDE.md`** — gateway internals: benchmaq + the `HOME`-breaks-RunPod-SSH /
  cu1300 / fork-install / crash-abort gotchas, the provider metrics page (NVML + `/proc`), the VM
  reverse tunnel, VM Compute sessions (uv venv + proxied JupyterLab), Activity + proxy-mode
  recording, the Label platform, the HF catalog/mirror, Quantization (llm-compressor —
  scheme sync, calib datasets, mirror-push Xet gotcha), Experiments (the evaluator registry,
  the no-retry rule, and the silent Langfuse-import corruptions), and GEPA prompt optimization
  (shared `EvaluatorStack`, the billed-call budget, the zero-cost-iteration spin).
- **`worker-agent/worker_agent/CLAUDE.md`** — the multi-model fleet: vLLM venv self-bootstrap,
  `vllm_version` / `vllm_install_args` / git forks / `pre_script`, and serving Whisper/audio.
- **`web/src/app/(app)/experiments/CLAUDE.md`** — the Experiments UI (agent observability): the
  Runs/Optimize/Evaluators tabs, the server-driven evaluator form, the two capture sources, the
  row-not-case vocabulary rule, and the CVD-validated tradeoff-plot palette.
- **`web/src/app/(app)/quantization/CLAUDE.md`** — the Quantization UI: mirrors Autotrain 1:1
  (the file↔file mapping table), server-driven scheme dropdown, HF-export tab conventions.

### Experiments: the two benchmark unit tests (ports of out-of-tree benchmarks)

`/experiments` ships two evaluators — **Function Call Unit Tests** (`function_call_units`) and
**Multilingual Unit Tests** (`multilingual_units`) — that are **ports of standalone research
benchmarks maintained out of tree**, not original work here.

Those benchmarks (their own harnesses, venvs and model fleet) **remain the source of truth for the
published numbers.** Only the *scoring* halves were ported, into `gateway/gateway/evaluators.py`
+ `langid.py`; the replay harnesses, eval datasets and result tables stayed there. If a metric
definition changes, change it upstream first and mirror it here — and don't present this platform's
figures as the benchmark's own (see caveat 3).

**Three things that make the numbers wrong if you forget them:**
1. **They need reference data on the dataset row.** `function_call_units` reads
   `expected.tool_calls`; `multilingual_units` reads `expected.language`. A row without it is
   **skipped, not failed** — so a suite can report a clean 100% while having scored almost nothing.
   Both aggregators return `scored` — check it against the row count (`turns` on the
   function-call one). `scored: 0` with a green pass rate means the dataset carries no
   reference data at all.
2. **fastText is needed for parity.** `multilingual_units` matches the published table only with
   `EXPERIMENT_FASTTEXT_MODEL=/path/to/lid.176.bin` (+ the `fasttext` wheel). Without it a built-in
   detector runs — good, but not identical. Every result stamps `detector: fasttext|builtin`;
   **never compare two runs whose detector differs.**
3. **⚠ Experiments replays ONE request per row; the function-call benchmark replays a whole
   conversation.** The real harness walks 10–20 turns, injecting reference tool results so the
   model accumulates context. Here each row is a single request, so on a multi-turn corpus (e.g.
   `Scicom-intl/Function-Call-TaaS`) this **effectively scores the first turn only** and will read
   higher than the benchmark. Fine for regression-watching one turn; **not a substitute for the
   real run**. Multi-turn replay is unimplemented — it needs the runner to thread tool results
   back into the request, which is the one genuinely missing piece.

Both report **corpus-level** metrics (F1, per-language accuracy) that cannot be averaged from
per-sample rates — see the `aggregate` hook in `gateway/gateway/CLAUDE.md`.

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

(The serverless/benchmark workers solve this with a reverse SSH tunnel — see
`gateway/gateway/CLAUDE.md` "VM reverse tunnel".)

The user has been told this multiple times and may push back. Per
`feedback_just_do_it.md`: don't re-litigate. State the constraint once, do what
they ask, move on.

### Testing the gateway locally (current `.env` reality)

`gateway/.env` is back to `AUTH_DISABLED=1` (checked 2026-07-18) + `GATEWAY_RELOAD=0`: backend
edits need a **manual gateway restart**; anonymous requests act as the seeded admin. Still prefer
sending a real **API key** as `Authorization: Bearer sgpu_…` (one lives in `automation/config.yaml`)
so tests behave the same against prod — and never write Redis `session:<token>` keys to forge a
session (that's exactly the prod-Redis-exposure risk; the user flagged it). No active training run?
a gateway restart is safe — runs detach and finalize from log. NOTE `AUTH_DISABLED=1` makes
`gateway/tests/test_hf_mirror.py::test_pull_requires_auth` self-skip (anonymous = admin is
intentional in that mode).

**Unit tests** (no stack needed, ~1s): `.venv/bin/pytest gateway/tests/unit`. Test deps are a
declared extra now: `uv pip install -e './gateway[dev]'`. The rest of `gateway/tests/` is the
live-stack integration suite and self-skips when nothing is listening.

### What NOT to do

- Don't suggest `docker compose up gateway` to test backend changes — the compose
  gateway runs the *image*, not their working tree. They want hot reload.
- Don't suggest deploying a branch to prod just to verify a fix. Reproduce locally first.
- Don't run `.venv/bin/gateway` yourself unless asked — the user typically has it
  running in a terminal already. Editing code triggers no auto-reload (uvicorn isn't
  in `--reload` mode), so just tell them to restart it.
- Don't `pip install` anything — use `uv pip install`.
