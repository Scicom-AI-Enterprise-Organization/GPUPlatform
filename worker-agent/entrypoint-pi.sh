#!/usr/bin/env bash
# Entrypoint for the PI custom-template worker image.
#
# Two processes run in the same container:
#   - vllm OpenAI-compatible server on $VLLM_PORT (background)
#   - worker-agent (foreground; container exits when it does)
#
# worker-agent waits for vllm to be /health-ready before BRPOPping the queue.
set -euo pipefail

: "${APP_ID:?APP_ID env var required}"
: "${MACHINE_ID:?MACHINE_ID env var required}"
: "${GATEWAY_URL:?GATEWAY_URL env var required}"
: "${REGISTRATION_TOKEN:?REGISTRATION_TOKEN env var required}"

# Reverse-tunnel support: the gateway can SSH back into this pod and forward the
# gateway+redis onto the pod's loopback (so a non-public / localhost gateway is
# reachable — see runpod_provider's reverse_tunnel mode). When it injects
# PUBLIC_KEY, authorize that key and start a real sshd (RunPod's ssh.runpod.io
# proxy can't do `-R`). No-op when PUBLIC_KEY is unset. Needs the pod to expose
# 22/tcp.
if [ -n "${PUBLIC_KEY:-}" ]; then
  echo "[entrypoint] PUBLIC_KEY set → enabling sshd for the gateway reverse tunnel"
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
  if [ ! -x /usr/sbin/sshd ]; then
    # Baked in via Dockerfile.pi; this is just a cold-start fallback.
    apt-get update && apt-get install -y --no-install-recommends openssh-server && rm -rf /var/lib/apt/lists/* || true
  fi
  mkdir -p /run/sshd
  ssh-keygen -A 2>/dev/null || true
  # AllowTcpForwarding must be on for the reverse forward; key-only root login.
  sed -ri 's/^#?PermitRootLogin.*/PermitRootLogin prohibit-password/; s/^#?AllowTcpForwarding.*/AllowTcpForwarding yes/' /etc/ssh/sshd_config 2>/dev/null || true
  /usr/sbin/sshd && echo "[entrypoint] sshd up on :22" || echo "[entrypoint] sshd failed — reverse tunnel won't connect"
fi

# Multi-model fleet: worker-agent launches + time-shares the vLLM servers itself
# (no single MODEL_ID, no pre-launched vllm) — hand straight to it.
if [ "${WORKER_MODE:-vllm}" = "multi" ]; then
  echo "[entrypoint] starting worker-agent (multi-model fleet): app=$APP_ID machine=$MACHINE_ID"
  exec python3 -m worker_agent.main
fi

# Single-model path below launches one vllm + proxies to it, so it needs a model.
: "${MODEL_ID:?MODEL_ID env var required}"

# Tailscale: when TS_AUTHKEY is set, join the user's tailnet so we can
# reach gateway-side Redis proxy via MagicDNS. Skipped when unset (e.g.
# bare-metal hosts that already run tailscaled, or local dev).
if [ -n "${TS_AUTHKEY:-}" ]; then
  echo "[entrypoint] starting tailscaled (kernel mode)"
  mkdir -p /var/run/tailscale /var/lib/tailscale
  tailscaled --state=/var/lib/tailscale/tailscaled.state \
             --socket=/var/run/tailscale/tailscaled.sock &
  for i in $(seq 1 30); do
    [ -S /var/run/tailscale/tailscaled.sock ] && break
    sleep 1
  done
  echo "[entrypoint] tailscale up (hostname=${MACHINE_ID}, ephemeral)"
  tailscale up --auth-key="${TS_AUTHKEY}" \
               --hostname="${MACHINE_ID}" \
               --ephemeral=true \
               --accept-dns=true \
               --reset
  echo "[entrypoint] tailnet status:"
  tailscale status | head -5 || true
fi

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
WORKER_LOG_PATH="${WORKER_LOG_PATH:-/var/log/vllm.log}"
mkdir -p "$(dirname "$WORKER_LOG_PATH")"
: > "$WORKER_LOG_PATH"
export WORKER_LOG_PATH

echo "[entrypoint] starting vllm: model=$MODEL_ID served-as=$APP_ID port=$VLLM_PORT extra=$VLLM_EXTRA_ARGS log=$WORKER_LOG_PATH"
# stdbuf forces line-buffered output so the worker-agent's log shipper sees
# vllm output as it's produced instead of in 4-KB stdio chunks. Output goes
# straight to the log file (the agent tails it); we also background a tail
# so anyone attached to the container can still see vllm's stdout.
stdbuf -oL -eL python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$APP_ID" \
  --port "$VLLM_PORT" \
  $VLLM_EXTRA_ARGS \
  > "$WORKER_LOG_PATH" 2>&1 &
VLLM_PID=$!
tail -F "$WORKER_LOG_PATH" &
TAIL_PID=$!

# Wait up to 10 minutes for vllm to be ready (large models take a while)
echo "[entrypoint] waiting for vllm /health on :$VLLM_PORT ..."
vllm_ready=0
for i in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "[entrypoint] vllm ready after ${i}s"
    vllm_ready=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[entrypoint] vllm died during startup" >&2
    exit 1
  fi
  sleep 1
done

# If the loop expired without /health ever passing (vllm alive but wedged —
# stuck loading, deadlocked, unhealthy), do NOT fall through to the worker-agent:
# a broken worker would register + poll while burning full GPU cost and serving
# nothing. Exit non-zero so the provider/autoscaler tears it down + reprovisions.
if [ "$vllm_ready" != "1" ]; then
  echo "[entrypoint] vllm never became healthy after 600s — exiting so the provider reprovisions" >&2
  kill -TERM "$VLLM_PID" "$TAIL_PID" 2>/dev/null || true
  exit 1
fi

# Worker-agent runs in foreground; if it exits, kill vllm + tail too.
trap 'kill -TERM "$VLLM_PID" "$TAIL_PID" 2>/dev/null || true' EXIT

# Per-worker observability — fire-and-forget ansible-pull AFTER vllm is ready
# so it never delays the first request. Failures are non-fatal.
if [ "${ENABLE_METRICS:-false}" = "true" ]; then
  (
    echo "[metrics] running ansible-pull (endpoint=$APP_ID)"
    ansible-pull \
      -U "${METRICS_REPO_URL:-https://github.com/AIES-Infra/gpu-metrics-exporter.git}" \
      -C "${METRICS_REPO_BRANCH:-main}" \
      -i localhost, \
      playbooks/serverless_metrics_local.yml \
      -e "endpoint=$APP_ID" \
      -e "datacenter=${METRICS_DATACENTER:-runpod}" \
      -e "vllm_port=$VLLM_PORT" \
      -e "alloy_remote_write_url=$METRICS_REMOTE_WRITE_URL" \
      -e "alloy_vllm_remote_write_url=$METRICS_REMOTE_WRITE_URL" \
      -e "alloy_username=$METRICS_USERNAME" \
      -e "alloy_vllm_username=$METRICS_USERNAME" \
      -e "alloy_password=$METRICS_PASSWORD" \
      -e "alloy_vllm_password=$METRICS_PASSWORD" \
      > /var/log/ansible-pull.log 2>&1 \
      && echo "[metrics] install ok" \
      || echo "[metrics] ansible-pull failed (non-fatal); see /var/log/ansible-pull.log"
  ) &
fi

echo "[entrypoint] starting worker-agent: app=$APP_ID machine=$MACHINE_ID"
exec python3 -m worker_agent.main
