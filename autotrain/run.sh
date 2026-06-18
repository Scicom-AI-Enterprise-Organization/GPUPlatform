#!/usr/bin/env bash
# Gemma-4 31B LoRA finetune — 2x H100 SXM on RunPod.
#
# Pre-flight: this installs torch 2.12 (the ABI the FA3 wheel is built against) for whatever
# CUDA the host driver supports, installs FlashAttention-3 + deps, runs the attention
# correctness test, downloads the model + packed dataset once, then launches torchrun.
#
# Usage on the pod:
#   export HF_TOKEN=hf_...            # gated google/gemma-4-31B-it
#   bash run.sh                       # full run
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

# ---------- FlashAttention-3 (Hopper; prebuilt, no JIT build) ----------
WHL="flash_attn_3-3.0.0+${CU}torch2.12gite2743ab-cp39-abi3-linux_x86_64.whl"
wget -nc -q "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/${FA3_TAG}/${WHL}"
$PIP "$WHL"

# ---------- remaining deps ----------
$PIP kernels==0.14.1                # >=0.15.1 needs an explicit version pin; keep 0.14.1
$PIP -U "transformers==5.5.0"       # HF >= 5.0.0 for Gemma4ForConditionalGeneration
$PIP mlflow psutil pynvml           # logging + GPU monitoring
$PIP liger-kernel                   # LigerFusedLinearCrossEntropyLoss (used by gemma4.py)
$PIP peft                           # LoRA (B=0 init, tested merge) — replaces the custom LinearLoRA
$PIP wandb                          # metrics logging (gemma4.py --wandb)
$PIP "git+https://github.com/Scicom-AI-Enterprise-Organization/ChiniDataset.git"
$PIP -U huggingface_hub hf_transfer # `hf` CLI + fast download

# ---------- pre-flight: attention correctness (SDPA mask + FA3 cu_seqlens) ----------
echo ">> attention correctness test"
python test_attention.py

if [ "$GEMMA_DEPS_ONLY" = "1" ]; then echo ">> GEMMA_DEPS_ONLY=1 — install + test done, skipping train"; exit 0; fi

# ---------- data + model (download ONCE; both ranks then read the shared HF cache) ----------
rm -rf ./packed_data
hf download "$DATA_ID" --repo-type dataset --local-dir ./packed_data
# populate HF cache so the two torchrun ranks don't race the download. Exclude redundant raw
# formats (original/*.pth, gguf) — from_pretrained only needs the safetensors + config/tokenizer.
hf download "$MODEL_ID" --exclude "original/*" "*.pth" "*.gguf" "consolidated*"

# ---------- train ----------
echo ">> launching training on $NPROC GPUs"
torchrun --nproc_per_node="$NPROC" gemma4.py "$@"
