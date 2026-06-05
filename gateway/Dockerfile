FROM python:3.12-slim
WORKDIR /app

# Runtime deps for benchmaq's runpod flow:
#   git           — uv pip installs benchmaq from git
#   openssh-client— benchmaq SSHs into the spawned RunPod pod
#   ca-certificates + curl — fetch runpodctl binary
RUN apt-get update && apt-get install -y --no-install-recommends \
      git openssh-client ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# runpodctl — benchmaq shells out to it for `pod create / ssh info / delete`.
# TARGETARCH is set by BuildKit per platform in multi-arch builds (amd64/arm64);
# without it we'd bake the host arch into both image variants → "exec format
# error" when the wrong-arch pod boots on an EKS node.
ARG TARGETARCH
RUN curl -fsSL -o /usr/local/bin/runpodctl \
      "https://github.com/runpod/runpodctl/releases/download/v2.1.9/runpodctl-linux-${TARGETARCH}" \
    && chmod +x /usr/local/bin/runpodctl \
    && /usr/local/bin/runpodctl version || true

RUN pip install --no-cache-dir uv
# Build context is the repo ROOT (see .github/workflows/ci.yml + Makefile) so we
# can also COPY the sibling worker-agent/ below. Paths here are repo-relative.
COPY gateway/pyproject.toml ./
COPY gateway/gateway ./gateway
RUN uv pip install --system --no-cache-dir -e .

# Bundle the worker-agent SOURCE so the gateway can ship it to VM fleets over the
# SSH exec channel on provision (vm_serverless_provider._worker_agent_tarball_b64
# tarballs this dir — there is no git clone of the private repo). WITHOUT this the
# shipper finds no source and a fleet Redeploy silently reuses the STALE
# worker-agent already installed on the VM, so worker-agent fixes never land.
COPY worker-agent ./worker-agent
ENV WORKER_AGENT_SRC_DIR=/app/worker-agent

# Bake the release version (git short-sha) so /health + /version report which
# build is live. CI passes `--build-arg APP_VERSION=<sha>`; defaults to "dev".
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}

EXPOSE 8080
ENV GATEWAY_BIND=0.0.0.0:8080
CMD ["python", "-m", "gateway.main"]
