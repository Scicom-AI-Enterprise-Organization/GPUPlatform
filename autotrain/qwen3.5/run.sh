#!/usr/bin/env bash
# Qwen3.6 LoRA finetune (dense + MoE) — FSDP2 on N GPUs (tm 8× H20-3e box).
# Qwen3.6 reuses the Qwen3.5 arch; qwen3_5.py auto-detects dense vs MoE from the config:
#   MODEL_ID=Qwen/Qwen3.6-27B      (dense, Qwen3_5ForConditionalGeneration)      [default]
#   MODEL_ID=Qwen/Qwen3.6-35B-A3B  (MoE,   Qwen3_5MoeForConditionalGeneration)
#
# Pre-flight: creates/activates a uv venv, installs torch 2.10 + the Qwen3.5 stack (transformers,
# FlashQLA GatedDeltaNet kernel, causal_conv1d, liger-kernel, kernels<=0.14 for the
# kernels-community/flash-attn3 attention, ChiniDataset), packs the dataset, then launches torchrun.
#
# Usage on the box:
#   export HF_TOKEN=hf_...            # for the (gated) Scicom-intl/Function-Call-TaaS dataset
#   bash run.sh                       # full run of Qwen/Qwen3.6-27B (deps + pack + train)
#   MODEL_ID=Qwen/Qwen3.6-35B-A3B bash run.sh        # train the MoE model instead
#   QWEN_DEPS_ONLY=1 bash run.sh      # install deps only (no pack, no train)
#   bash run.sh --max_steps 50 --limit_samples 10   # short smoke -> a LoRA checkpoint
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh         # pin specific GPUs
set -euo pipefail
cd "$(dirname "$0")"

# ---------- config (override via env) ----------
: "${HF_TOKEN:?HF_TOKEN required for the (gated) Scicom-intl/Function-Call-TaaS dataset pull (export HF_TOKEN=hf_...)}"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
# tm box: Xet stalls big pulls; hf_transfer off (the pack default). HF_HOME defaults to the shared cache.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HOME="${HF_HOME:-/share/huggingface}"

TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
# Qwen3.6 reuses the Qwen3.5 arch. Train either: Qwen/Qwen3.6-27B (dense) or
# Qwen/Qwen3.6-35B-A3B (MoE) — qwen3_5.py auto-detects dense vs MoE from the config.
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-50000}"          # pack_dataset.py --max-seq-len (50k fits the H20 activation budget)
QWEN_DEPS_ONLY="${QWEN_DEPS_ONLY:-0}"
VENV_PATH="${VENV_PATH:-.venv}"              # uv venv to install into (created if absent & no venv active)
# Per-model checkpoint dir so 27B and 35B-A3B runs don't clobber each other's lora.pt.
_MODEL_SLUG="$(printf '%s' "$MODEL_ID" | sed 's#.*/##' | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '-')"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpointing-${_MODEL_SLUG}}"

# NCCL: disable NVLink-SHARP (NVLS) multicast — the bind fails inside these containers and crashes
# the first FSDP all-gather; ring/tree over NVLink/PCIe works fine.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
# Reduce allocator fragmentation for the long packed bins.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# GPUs: default to every GPU on the box (override with CUDA_VISIBLE_DEVICES=0,1,2,3).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
NPROC="$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)"
echo ">> GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  nproc_per_node=$NPROC"

# ---------- CUDA toolkit (nvcc) for source-building causal_conv1d / FlashQLA ----------
# These build CUDA extensions from source, so nvcc must be on PATH and match torch's CUDA.
if [ -d /usr/local/cuda/bin ]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
fi
# Pick the CUDA backend from the host driver so the torch wheel matches nvcc (avoids ABI mismatch
# in the source builds). nvidia-smi reports the MAX CUDA the driver supports.
HOST_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
cuda_ge() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]; }   # $1 >= $2 ?
if   [ -n "$HOST_CUDA" ] && cuda_ge "$HOST_CUDA" "13.0"; then CU=cu130
elif [ -n "$HOST_CUDA" ] && cuda_ge "$HOST_CUDA" "12.8"; then CU=cu128
else CU=cu126; fi
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/$CU}"
echo ">> host driver CUDA=${HOST_CUDA:-unknown} -> torch backend $CU ($TORCH_INDEX_URL)"

# ---------- uv + venv ----------
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
if [ -z "${VIRTUAL_ENV:-}" ]; then
  [ -d "$VENV_PATH" ] || { echo ">> creating venv $VENV_PATH (python 3.12)"; uv venv "$VENV_PATH" --python 3.12; }
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi
echo ">> venv: ${VIRTUAL_ENV:-<none>}"
PIP="uv pip install"

# ---------- deps (the proven Qwen3.5 stack) ----------
$PIP "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "$TORCH_INDEX_URL"
$PIP setuptools wheel packaging ninja   # build deps for the --no-build-isolation CUDA extension below
$PIP huggingface_hub transformers ipykernel liger-kernel wandb mlflow pandas pyarrow tqdm accelerate
$PIP "kernels<=0.14.0"                                                  # for attn_implementation=kernels-community/flash-attn3
$PIP "git+https://github.com/QwenLM/FlashQLA.git"                       # chunk_gated_delta_rule (GatedDeltaNet linear attn; pure-python)
# causal_conv1d builds a CUDA extension and MUST be --no-build-isolation so it links the venv's
# cu130 torch. Under uv's default build isolation it pulls the default (cu12) torch into the build
# env, and the resulting causal_conv1d_cuda.so links libcudart.so.12 -> ImportError at runtime
# ("libcudart.so.12: cannot open shared object file") against our cu130 torch.
$PIP --no-build-isolation causal_conv1d                                 # Qwen3.5 GatedDeltaNet short conv
$PIP "git+https://github.com/Scicom-AI-Enterprise-Organization/ChiniDataset.git"

if [ "$QWEN_DEPS_ONLY" = "1" ]; then echo ">> QWEN_DEPS_ONLY=1 — deps installed, skipping pack + train"; exit 0; fi

# ---------- data: pack the dataset (skip if ./packed_data already present) ----------
# SKIP_DATA_PACK=1 reuses an existing ./packed_data (e.g. one packed earlier or scp'd in).
if [ "${SKIP_DATA_PACK:-0}" = "1" ] && [ -d ./packed_data ]; then
  echo ">> SKIP_DATA_PACK=1 — reusing existing ./packed_data"
else
  echo ">> packing dataset (max-seq-len=$MAX_SEQ_LEN, tokenizer=$MODEL_ID)"
  python pack_dataset.py --out ./packed_data --max-seq-len "$MAX_SEQ_LEN" --tokenizer "$MODEL_ID"
fi

# ---------- model (download ONCE into the shared HF cache; the torchrun ranks then read it) ----------
# Pre-fetch so the N ranks don't race the same ~54GB download in from_pretrained. Idempotent: a
# already-cached model is a fast no-op. SKIP_MODEL_DOWNLOAD=1 bypasses it entirely.
if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
  echo ">> ensuring $MODEL_ID is in the HF cache ($HF_HOME)"
  hf download "$MODEL_ID" --exclude "original/*" --exclude "*.pth" --exclude "*.gguf"
fi

# ---------- train ----------
# --model_id/--checkpoint_dir come first so an explicit override in "$@" wins (argparse: last).
echo ">> launching training on $NPROC GPUs (model=$MODEL_ID, checkpoint_dir=$CHECKPOINT_DIR)"
torchrun --nproc_per_node="$NPROC" qwen3_5.py \
  --model_id "$MODEL_ID" --checkpoint_dir "$CHECKPOINT_DIR" "$@"
