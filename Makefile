# Serverless-GPU developer ergonomics.
#
# Common entrypoints:
#   make install       create .venv and install all three packages editable + dev deps
#   make test          run pytest against the consolidated test suite
#   make compose-up    docker compose up --build (default profile: static fake worker)
#   make compose-down  docker compose down (preserves redis volume)
#   make compose-nuke  docker compose down -v (wipes redis volume too)
#   make helm-lint     lint the k8s helm chart
#   make helm-template render the helm chart with sample values
#   make clean         remove .venv and __pycache__

.PHONY: install runpodctl test test-fast compose-up compose-down compose-nuke \
        helm-lint helm-template clean pi-check pi-up verify-values \
        ecr-create-repos ecr-build-push ecr-build-push-gateway ecr-build-push-worker

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:
	uv venv $(VENV)
	VIRTUAL_ENV=$(PWD)/$(VENV) uv pip install -e ./gateway -e ./worker-agent -e ./sdk
	VIRTUAL_ENV=$(PWD)/$(VENV) uv pip install \
	  pytest pytest-asyncio fakeredis httpx
	$(MAKE) runpodctl

# runpodctl — RunPod's CLI. benchmaq shells out to it for RunPod-target runs,
# and the gateway prepends $(VENV)/bin to benchmaq's PATH, so it must live here
# (alongside the gateway/benchmaq binaries), not rely on a system-wide install.
runpodctl:
	@os=$$(uname -s | tr 'A-Z' 'a-z'); arch=$$(uname -m); \
	  case "$$arch" in x86_64|amd64) arch=amd64;; aarch64|arm64) arch=arm64;; esac; \
	  url="https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-$$os-$$arch"; \
	  echo "installing runpodctl ($$os-$$arch) -> $(VENV)/bin/runpodctl"; \
	  curl -fsSL -o "$(VENV)/bin/runpodctl" "$$url" && chmod +x "$(VENV)/bin/runpodctl"; \
	  "$(VENV)/bin/runpodctl" version

test:
	$(PY) -m pytest tests/ -v

test-fast:
	$(PY) -m pytest tests/ -x --ff -v

compose-up:
	docker compose up --build

compose-down:
	docker compose down

compose-nuke:
	docker compose down -v

helm-lint:
	helm lint deploy/helm/serverlessgpu/

helm-template:
	helm template test deploy/helm/serverlessgpu/ \
	  --set primeintellect.apiKey=fake-render-key \
	  --set primeintellect.customTemplateId=tmpl-x \
	  --set ingress.host=api.example.com \
	  --set gateway.publicUrl=https://api.example.com \
	  --set gateway.apiKeys=key1,key2

# Catch the most common helm-install footgun: leftover TODO_* placeholders
# in your my-values.yaml. Fails before you accidentally helm install with
# `image.registry: ghcr.io/TODO_YOUR_GH_USER` and end up debugging
# ImagePullBackOff for an hour.
#
# Usage: make verify-values VALUES=my-values.yaml
VALUES ?= my-values.yaml
verify-values:
	@if [ ! -f "$(VALUES)" ]; then echo "$(VALUES) not found — copy from deploy/helm/serverlessgpu/values-prod.example.yaml first"; exit 1; fi
	@if grep -nE "TODO_" "$(VALUES)"; then \
	  echo ""; \
	  echo "✗ leftover TODO_ placeholders in $(VALUES) — fill them in before deploying."; \
	  exit 1; \
	fi
	@echo "✓ $(VALUES) looks clean (no TODO_ placeholders)"
	@helm lint deploy/helm/serverlessgpu/ -f "$(VALUES)"

# ============================================================================
# Prime Intellect operator helpers
# ============================================================================

# Pre-flight check against the real PI API. Reads PI_API_KEY from your shell.
pi-check:
	@if [ -z "$$PI_API_KEY" ]; then echo "set PI_API_KEY in your shell first"; exit 1; fi
	$(PY) -m serverlessgpu.cli pi-check

# Path B from docs/DEPLOY.md: local gateway + real PI workers via cloudflared.
# Compresses the manual env-var dance into one command. Requires:
#   - cloudflared installed (brew install cloudflared)
#   - PI_API_KEY, PI_CUSTOM_TEMPLATE_ID set in shell
pi-up:
	@if [ -z "$$PI_API_KEY" ]; then echo "set PI_API_KEY in your shell first"; exit 1; fi
	@if [ -z "$$PI_CUSTOM_TEMPLATE_ID" ]; then echo "set PI_CUSTOM_TEMPLATE_ID in your shell first"; exit 1; fi
	@command -v cloudflared >/dev/null || { echo "cloudflared not installed (brew install cloudflared)"; exit 1; }
	@echo ""
	@echo "Step 1/3: opening cloudflared tunnel to localhost:8080..."
	@echo "  --- watch the next ~5 lines for your https://...trycloudflare.com URL ---"
	@cloudflared tunnel --url http://localhost:8080 --logfile /tmp/sgpu-cf.log &
	@sleep 5
	@URL=$$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/sgpu-cf.log | head -1); \
	  if [ -z "$$URL" ]; then echo "couldn't extract tunnel URL — check /tmp/sgpu-cf.log"; exit 1; fi; \
	  echo "  tunnel URL: $$URL"; \
	  echo ""; \
	  echo "Step 2/3: starting gateway + redis with PROVIDER=primeintellect..."; \
	  AUTOSCALER=1 PROVIDER=primeintellect \
	    PI_API_KEY=$$PI_API_KEY \
	    PI_CUSTOM_TEMPLATE_ID=$$PI_CUSTOM_TEMPLATE_ID \
	    GATEWAY_PUBLIC_URL=$$URL \
	    docker compose up --build gateway redis &
	@echo ""
	@echo "Step 3/3: deploy + run from another terminal:"
	@echo "  source .venv/bin/activate"
	@echo "  serverlessgpu deploy sdk/examples/qwen.py:qwen"
	@echo "  serverlessgpu run qwen --payload '{\"prompt\": \"hello\"}'"
	@echo ""
	@echo "ctrl-c here to tear everything down."

clean:
	rm -rf $(VENV) **/__pycache__ **/*.egg-info .pytest_cache

# ============================================================================
# ECR — local build & push (no GitHub, no CI)
# ============================================================================
#
# Tested flow: build images on your laptop with buildx, push to ECR directly.
# Useful when you don't want to commit/push to GitHub yet but want to deploy.
#
# Prereqs:
#   - aws CLI configured (aws sso login)
#   - docker / docker buildx (Docker Desktop on macOS has both)
#   - The ECR repos exist (run `make ecr-create-repos` once)
#
# Defaults match the helm chart (and CI). Override via env if you want to push
# to a different account / alias.

AWS_REGION       ?= ap-southeast-5
AWS_ACCOUNT_ID   ?= 865626945255
ECR_REGISTRY     ?= $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
# AWS auto-generated public-ECR alias; claim a memorable one via Console if you want
ECR_PUBLIC_ALIAS ?= o6x1g6b0
GATEWAY_REPO     ?= aies/serverlessgpu-gateway
WORKER_REPO      ?= serverlessgpu-pi-worker
TAG              ?= latest

# One-time: create both ECR repos. Idempotent — re-running is fine, AWS just
# logs "RepositoryAlreadyExistsException" which we swallow.
ecr-create-repos:
	@echo "Creating private ECR repo: $(GATEWAY_REPO) in $(AWS_REGION)"
	@aws ecr create-repository \
	  --repository-name $(GATEWAY_REPO) \
	  --region $(AWS_REGION) \
	  --image-tag-mutability MUTABLE 2>&1 | grep -v RepositoryAlreadyExistsException || true
	@echo ""
	@echo "Creating ECR Public repo: $(WORKER_REPO) (us-east-1)"
	@aws ecr-public create-repository \
	  --repository-name $(WORKER_REPO) \
	  --region us-east-1 \
	  --catalog-data 'description=Serverless-GPU worker (vLLM + worker-agent), pulled by Prime Intellect.,architectures=x86-64,operatingSystems=Linux' \
	  2>&1 | grep -v RepositoryAlreadyExistsException || true
	@echo ""
	@echo "Verify:"
	@aws ecr describe-repositories --region $(AWS_REGION) --repository-names $(GATEWAY_REPO) --query 'repositories[0].repositoryUri' --output text
	@aws ecr-public describe-repositories --region us-east-1 --repository-names $(WORKER_REPO) --query 'repositories[0].repositoryUri' --output text

# Build & push the gateway image.
# Default platforms = arm64 only because:
#   1. EKS Graviton nodes are ARM64 (the only place this image runs in prod)
#   2. QEMU emulation of amd64 on Apple Silicon segfaults running `uv`
# Override with: GATEWAY_PLATFORMS=linux/amd64,linux/arm64 (needs pip-not-uv
# in Dockerfile, or an x86 build host).
GATEWAY_PLATFORMS ?= linux/arm64
ecr-build-push-gateway:
	@aws ecr get-login-password --region $(AWS_REGION) | \
	  docker login --username AWS --password-stdin $(ECR_REGISTRY)
	docker buildx build \
	  --platform $(GATEWAY_PLATFORMS) \
	  --tag $(ECR_REGISTRY)/$(GATEWAY_REPO):$(TAG) \
	  --file ./gateway/Dockerfile \
	  --push \
	  .

# Build & push the worker image. amd64 only (vLLM is x86, PI hosts are x86).
ecr-build-push-worker:
	@aws ecr-public get-login-password --region us-east-1 | \
	  docker login --username AWS --password-stdin public.ecr.aws
	docker buildx build \
	  --platform linux/amd64 \
	  --tag public.ecr.aws/$(ECR_PUBLIC_ALIAS)/$(WORKER_REPO):$(TAG) \
	  --file ./worker-agent/Dockerfile.pi \
	  --push \
	  .

# Build + push both, in series.
ecr-build-push: ecr-build-push-gateway ecr-build-push-worker
	@echo ""
	@echo "Pushed:"
	@echo "  $(ECR_REGISTRY)/$(GATEWAY_REPO):$(TAG)"
	@echo "  public.ecr.aws/$(ECR_PUBLIC_ALIAS)/$(WORKER_REPO):$(TAG)"
