#!/usr/bin/env bash
# Entrypoint for the multi-model worker (WORKER_MODE=multi).
#
# Unlike the single-model entrypoint, this does NOT start a vLLM server itself:
# the worker-agent's scheduler launches one vLLM per member model (each pinned
# to its GPUs, sleep-mode enabled) from MULTI_MODEL_CONFIG, and routes/evicts
# them at runtime. We just (optionally) join the tailnet, then exec the agent.
set -euo pipefail

: "${APP_ID:?APP_ID env var required}"
: "${MACHINE_ID:?MACHINE_ID env var required}"
: "${GATEWAY_URL:?GATEWAY_URL env var required}"
: "${REGISTRATION_TOKEN:?REGISTRATION_TOKEN env var required}"
: "${MULTI_MODEL_CONFIG:?MULTI_MODEL_CONFIG env var required for multi mode}"

export WORKER_MODE=multi
export WORKER_LOG_DIR="${WORKER_LOG_DIR:-/var/log/vllm}"
mkdir -p "$WORKER_LOG_DIR"

# Tailscale: join the user's tailnet so the worker can reach gateway-side Redis
# via MagicDNS. Skipped when TS_AUTHKEY is unset (bare-metal hosts on a LAN).
if [ -n "${TS_AUTHKEY:-}" ]; then
  echo "[entrypoint-multi] starting tailscaled (kernel mode)"
  mkdir -p /var/run/tailscale /var/lib/tailscale
  tailscaled --state=/var/lib/tailscale/tailscaled.state \
             --socket=/var/run/tailscale/tailscaled.sock &
  for i in $(seq 1 30); do
    [ -S /var/run/tailscale/tailscaled.sock ] && break
    sleep 1
  done
  tailscale up --auth-key="${TS_AUTHKEY}" \
               --hostname="${MACHINE_ID}" \
               --ephemeral=true \
               --accept-dns=true \
               --reset
fi

echo "[entrypoint-multi] starting worker-agent (fleet launched by scheduler)"
exec python3 -m worker_agent.main
