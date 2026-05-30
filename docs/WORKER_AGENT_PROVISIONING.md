# Worker-agent provisioning on a VM (no git clone)

How a `kind: vm` serverless endpoint gets the **worker-agent** onto the box, why it
does **not** `git clone`, and how to make it work on prod. Background for
[docs/MULTI_MODEL_FLEET.md](MULTI_MODEL_FLEET.md).

## Why not git

The repo (`github.com/Scicom-AI-Enterprise-Organization/GPUPlatform`) is **private** and
the VMs have no GitHub credentials, so `uv pip install git+https://…` fails on the VM
with `could not read Username for 'https://github.com'`. The old default install spec
also pointed at the wrong org. So the gateway no longer clones — it **ships the
worker-agent source itself**.

## How it works (`gateway/gateway/vm_serverless_provider.py`)

On `provision()` the gateway SSHes into the VM and, in order:

1. Writes the worker config (`MULTI_MODEL_CONFIG` + token) via **base64 over the exec
   channel** — proxied SSH front-ends (Alibaba PAI DSW) have no SFTP.
2. **Installs the worker-agent**, preferring to ship the source:
   - If the gateway has the worker-agent source (`WORKER_AGENT_SRC_DIR`, else the
     `worker-agent/` sibling of the gateway package), it tars it, base64-ships it over
     the same exec channel, and `uv pip install`s it into `~/.sgpu/venv`. A gateway
     code change therefore propagates to the VM on the next provision — no registry, no
     git.
   - **Fallback:** if no source is bundled, it uses whatever worker-agent is already
     installed in the VM venv (and errors clearly only if neither is present). This is
     the `f69a447` fix — previously a missing bundled source hard-failed the whole
     provision with *"worker-agent source not found"*.
3. `nohup`s `python -m worker_agent.main` against the config.

The worker reaches the gateway + Redis at **public** endpoints on prod
(`serverlessgpu.aies.scicom.dev`, `sgpu-redis-public-…`); locally it uses a
**reverse-SSH tunnel** (`VM_REVERSE_TUNNEL=1`) so the VM can see your laptop's localhost.

## Prod requirement: bundle the source in the gateway image

The gateway runs as a **container image** on prod, not a source checkout, so the
`worker-agent/` sibling isn't present. Two ways to satisfy step 2's preferred path so
prod is self-sufficient (doesn't depend on a pre-installed VM venv):

```dockerfile
# gateway image — bundle the worker-agent source (build context = repo root)
COPY worker-agent ./worker-agent
ENV WORKER_AGENT_SRC_DIR=/app/worker-agent
```
…or set `WORKER_AGENT_SRC_DIR` to a path you mount in. **TODO (follow-up):** wire this
into `gateway/Dockerfile` + the CI gateway build context. Until then, prod relies on the
fallback — fine for VMs whose `~/.sgpu/venv` already has the worker-agent (e.g. TM-H20),
but a brand-new VM needs the source bundled.

## Deploying a fleet (prod)

```bash
export SGPU_URL=https://serverlessgpu.aies.scicom.dev
export SGPU_API_KEY=sgpu_...
curl -s "$SGPU_URL/apps" -H "Authorization: Bearer $SGPU_API_KEY" \
  -H 'Content-Type: application/json' -X POST -d @fleet.json
```

See [docs/MULTI_MODEL_FLEET.md](MULTI_MODEL_FLEET.md) for the full `fleet.json` shape
(models, `tp`, `visible_devices`, `venv_path`, `env_vars`, `sleep_level`).

### Live reference deployment

`tm-fleet` on prod (provider `prov-3318238b` "TM-H20", 8× H20-3e), GPU 0–5,
`--gpu-memory-utilization 0.8` on every model:

| Model | tp | GPUs |
|---|---|---|
| `qwen/qwen3.6-27b` | 2 | 0,1 |
| `Qwen/Qwen3.6-35B-A3B` | 2 | 2,3 |
| `Qwen/Qwen3.5-122B-A10B` | 4 | 0,1,2,3 |
| `mistralai/Mistral-Small-4-119B-2603` | 4 | 0,1,2,3 |
| `google/gemma-4-31b-it` | 2 | 4,5 |

All five load (~9 min) and serve; the three non-overlapping models stay resident while
the two tp-4 giants share GPU 0–3 via sleep/wake. Address each by its model id at
`$SGPU_URL/tm-fleet/v1/chat/completions`.
