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

### What NOT to do

- Don't suggest `docker compose up gateway` to test backend changes — the compose
  gateway runs the *image*, not their working tree. They want hot reload.
- Don't suggest deploying a branch to prod just to verify a fix. Reproduce locally first.
- Don't run `.venv/bin/gateway` yourself unless asked — the user typically has it
  running in a terminal already. Editing code triggers no auto-reload (uvicorn isn't
  in `--reload` mode), so just tell them to restart it.
- Don't `pip install` anything — use `uv pip install`.
