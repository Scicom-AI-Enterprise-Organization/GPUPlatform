#!/usr/bin/env bash
# Gemma-4 31B LoRA finetune — 2x H100 SXM on RunPod.
#
# Pre-flight: installs torch 2.12 (the ABI the FA3 wheel needs) for the host CUDA, installs the
# attention kernel + deps, runs the correctness test, downloads the model + packed dataset, launches
# torchrun. Attention backend is chosen by GEMMA_FA4: 0 (default) = FA3 wheel + dynamic_attention
# (SDPA-tiled head_dim-512, ~32k ceiling); 1 = clone+install the flash-attention-512 FA4 fork and use
# fa4_attention (head_dim-512 cute kernel, long context — but see the activation-memory ceiling note).
#
# Usage on the pod:
#   export HF_TOKEN=hf_...            # gated google/gemma-4-31B-it
#   bash run.sh                       # full run (FA3 / dynamic_attention)
#   GEMMA_FA4=1 bash run.sh           # FA4 path: clone flash-attention-512 + fa4_attention
#                                     #   (scp the private fork to ./flash-attention-512 first, or
#                                     #    have a GH token; then run compare_logits_fa4.py as the gate)
#   GEMMA_DEPS_ONLY=1 bash run.sh     # install + correctness test only (no train)
set -euo pipefail
cd "$(dirname "$0")"

# ---------- config (override via env) ----------
: "${HF_TOKEN:?HF_TOKEN required for the gated google/gemma-4-31B-it download (export HF_TOKEN=hf_...)}"
export HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
TORCH_VERSION="${TORCH_VERSION:-2.12.0}"   # MUST be 2.12.x — the FA3 wheel is built for torch2.12
FA3_TAG="${FA3_TAG:-v0.9.18}"
MODEL_ID="${MODEL_ID:-google/gemma-4-31B-it}"
DATA_ID="${DATA_ID:-huseinzolkepliscicom/gemma4-multipack}"
GEMMA_DEPS_ONLY="${GEMMA_DEPS_ONLY:-0}"

# ---------- attention backend ----------
# GEMMA_FA4=1 → install the FlashAttention-4 head_dim-512 fork and train via fa4_attention
# (memory-efficient O(S), enables long context). GEMMA_FA4=0 (default) → the prebuilt FA3 wheel +
# the SDPA-tiled dynamic_attention path (FA3 caps head_dim 256, so the 512 global layers use SDPA).
GEMMA_FA4="${GEMMA_FA4:-0}"
# The FA4 head_dim-512 fork (adds symmetric head_dim=512 fwd+bwd on SM90 for gemma-4's global layers):
FA4_FORK_REPO="${FA4_FORK_REPO:-https://github.com/Scicom-AI-Enterprise-Organization/flash-attention-512.git}"
FA4_FORK_DIR="${FA4_FORK_DIR:-./flash-attention-512}"   # PRIVATE repo — clone needs a GH token, or
                                                        # scp the fork here first (run.sh reuses it if present).
# gemma4.py reads GEMMA_ATTN to pick the registered backend:
export GEMMA_ATTN="${GEMMA_ATTN:-$([ "$GEMMA_FA4" = "1" ] && echo fa4_attention || echo dynamic_attention)}"

# NCCL: disable NVLink-SHARP (NVLS) multicast. Inside RunPod containers the multicast memory
# bind fails ("Failed to bind NVLink SHARP (NVLS) Multicast memory ... CUDA error 401") and
# crashes the first FSDP all-gather. NCCL falls back to ring/tree over NVLink/PCIe — works fine.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"

# GPUs: default to every GPU on the pod.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
NPROC="$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)"
echo ">> GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  nproc_per_node=$NPROC"

# ---------- pick CUDA backend from the host driver ----------
# nvidia-smi reports the MAX CUDA the driver supports. FA3 v0.9.18 ships cu126 / cu130 / cu132.
HOST_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
echo ">> host driver supports CUDA up to: ${HOST_CUDA:-unknown}"
cuda_ge() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]; }  # $1 >= $2 ?
if   [ -n "$HOST_CUDA" ] && cuda_ge "$HOST_CUDA" "13.2"; then CU=cu132
elif [ -n "$HOST_CUDA" ] && cuda_ge "$HOST_CUDA" "13.0"; then CU=cu130
elif [ -n "$HOST_CUDA" ] && cuda_ge "$HOST_CUDA" "12.6"; then CU=cu126
else echo "ERROR: host driver CUDA '${HOST_CUDA}' too old; need >=12.6 for the FA3 wheel"; exit 1; fi
echo ">> using CUDA backend: $CU"

# ---------- uv ----------
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# Install into the active python (RunPod images use a system/conda python, not a venv).
# --break-system-packages: the base image's /usr python is PEP-668 externally-managed; this is
# an ephemeral, single-purpose pod so installing system-wide is fine.
UV_FLAGS="--system --break-system-packages"; [ -n "${VIRTUAL_ENV:-}" ] && UV_FLAGS=""
PIP="uv pip install $UV_FLAGS"

# ---------- torch 2.12 matching the host CUDA (FA3 wheel ABI: torch2.12) ----------
$PIP "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${CU}"

# ---------- attention kernel: FA4 fork (head_dim 512) OR the FA3 prebuilt wheel ----------
if [ "$GEMMA_FA4" = "1" ]; then
  echo ">> FA4 head_dim-512 fork → fa4_attention (replaces FA3; no FA3 wheel needed)"
  # PRIVATE repo: clone with a GH token (gh auth / GH_TOKEN), or scp the fork to $FA4_FORK_DIR first.
  if [ ! -d "$FA4_FORK_DIR/flash_attn/cute" ]; then
    echo ">> cloning $FA4_FORK_REPO -> $FA4_FORK_DIR"
    git clone --depth 1 "$FA4_FORK_REPO" "$FA4_FORK_DIR"
  else
    echo ">> reusing existing $FA4_FORK_DIR"
  fi
  # CuTeDSL kernels JIT-compile at runtime (FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 caches them).
  $PIP -e "$FA4_FORK_DIR/flash_attn/cute"
  # Pin the deps: the >= bounds in cute's pyproject pull too-new builds that break (quack 0.5.0 /
  # cutlass-dsl 4.5.x). [cu13] = CUDA-13 cutlass libs. Verified combo on H100/H20 cu130.
  $PIP "nvidia-cutlass-dsl[cu13]==4.4.2" "quack-kernels==0.3.10"
  python -c "from flash_attn.cute.interface import flash_attn_varlen_func; print('FA4 cute import OK')"
else
  echo ">> FlashAttention-3 prebuilt wheel → dynamic_attention (Hopper; no JIT build)"
  WHL="flash_attn_3-3.0.0+${CU}torch2.12gite2743ab-cp39-abi3-linux_x86_64.whl"
  wget -nc -q "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/${FA3_TAG}/${WHL}"
  $PIP "$WHL"
fi

# ---------- remaining deps ----------
$PIP kernels==0.14.1                # >=0.15.1 needs an explicit version pin; keep 0.14.1
$PIP -U "transformers==5.5.0"       # HF >= 5.0.0 for Gemma4ForConditionalGeneration
$PIP mlflow psutil pynvml           # logging + GPU monitoring
$PIP liger-kernel                   # LigerFusedLinearCrossEntropyLoss (used by gemma4.py)
$PIP peft                           # LoRA (B=0 init, tested merge) — replaces the custom LinearLoRA
$PIP wandb                          # metrics logging (gemma4.py --wandb)
$PIP "git+https://github.com/Scicom-AI-Enterprise-Organization/ChiniDataset.git"
$PIP -U huggingface_hub hf_transfer # `hf` CLI + fast download

# ---------- pre-flight: attention correctness ----------
# FA3/SDPA path → test_attention.py (kernel-level, no model). FA4 path → compare_logits_fa4.py is the
# gate, but it needs the model on the GPU, so run it after the model download (not here):
#   python compare_logits_fa4.py   # asserts fa4_attention logits == default (cosine > 0.99)
if [ "$GEMMA_FA4" = "1" ]; then
  echo ">> FA4 backend: skipping test_attention.py (run compare_logits_fa4.py post-download as the gate)"
else
  echo ">> attention correctness test"
  python test_attention.py
fi

if [ "$GEMMA_DEPS_ONLY" = "1" ]; then echo ">> GEMMA_DEPS_ONLY=1 — install + test done, skipping train"; exit 0; fi

# ---------- data + model (download ONCE; both ranks then read the shared HF cache) ----------
# SKIP_DATA_DOWNLOAD=1 keeps an existing ./packed_data (e.g. one scp'd from the laptop that is
# newer than / sized differently from the published DATA_ID) instead of re-pulling it.
if [ "${SKIP_DATA_DOWNLOAD:-0}" = "1" ] && [ -f ./packed_data/index.json ]; then
  echo ">> SKIP_DATA_DOWNLOAD=1 — reusing existing ./packed_data ($(python3 -c "import json;print(json.load(open('packed_data/index.json'))['shards'][0]['samples'])" 2>/dev/null) bins)"
else
  rm -rf ./packed_data
  hf download "$DATA_ID" --repo-type dataset --local-dir ./packed_data
fi
# populate HF cache so the two torchrun ranks don't race the download. Exclude redundant raw
# formats (original/*.pth, gguf) — from_pretrained only needs the safetensors + config/tokenizer.
# NOTE: `hf download --exclude` takes ONE pattern per flag (huggingface_hub >=1.x). Passing several
# space-separated patterns makes the CLI treat the extras as explicit FILENAMES → "Fetching 0
# files" (silent no-op). Repeat --exclude per pattern.
hf download "$MODEL_ID" --exclude "original/*" --exclude "*.pth" --exclude "*.gguf" --exclude "consolidated*"

# ---------- train ----------
# FA4: cache the JIT-compiled CuTeDSL kernels + reduce allocator fragmentation. (Activation memory is
# O(seq*layers) and unshardable, so even with FA4 the no-offload context ceiling is ~64k on 80GB
# H100 / ~70k on 144GB H20 — pack shorter bins, or add activation offload, for longer context.)
if [ "$GEMMA_FA4" = "1" ]; then
  export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED="${FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED:-1}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
fi
echo ">> launching training on $NPROC GPUs (GEMMA_ATTN=$GEMMA_ATTN)"
torchrun --nproc_per_node="$NPROC" gemma4.py "$@"
