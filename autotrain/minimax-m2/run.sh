#!/usr/bin/env bash
# MiniMax-M2 (230B/10B FP8 MoE) LoRA finetune — 4x H100 SXM on RunPod.
#
# Installs torch 2.12 (FA3 wheel ABI + has torch._grouped_mm for the fused MoE) for the host
# CUDA, the prebuilt FlashAttention-3 wheel (head_dim 128 works on FA3), transformers 5.5.0
# (native MiniMaxM2 + FP8 + fused-experts), liger-kernel, kernels, ChiniDataset. Runs the LoRA
# correctness test (CPU), builds/downloads the packed dataset, then torchrun.
#
#   export HF_TOKEN=hf_...                 # only if your account needs it for the download
#   MINIMAX_DEPS_ONLY=1 bash run.sh        # install + LoRA correctness test only (cheap smoke)
#   bash run.sh -- --max_steps 50          # short run -> a LoRA checkpoint for merge/inference
#   bash run.sh                            # full epoch
set -euo pipefail
cd "$(dirname "$0")"

# ---------- config (override via env) ----------
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
TORCH_VERSION="${TORCH_VERSION:-2.12.0}"     # MUST be 2.12.x — FA3 wheel ABI + torch._grouped_mm
FA3_TAG="${FA3_TAG:-v0.9.18}"
MODEL_ID="${MODEL_ID:-MiniMaxAI/MiniMax-M2}"
DATA_ID="${DATA_ID:-}"                        # optional prebuilt packed dataset repo; else pack locally
PACK_MAX_SEQ_LEN="${PACK_MAX_SEQ_LEN:-131072}"
PACK_MAX_ROWS="${PACK_MAX_ROWS:-}"           # cap source rows for a quick build (e.g. 20)
MINIMAX_DEPS_ONLY="${MINIMAX_DEPS_ONLY:-0}"

# NCCL: disable NVLink-SHARP multicast (the bind fails inside RunPod containers; ring/tree is fine).
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
NPROC="$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)"
echo ">> GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  nproc_per_node=$NPROC"

# ---------- pick CUDA backend from the host driver (FA3 v0.9.18 ships cu126/cu130/cu132) ----------
HOST_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
echo ">> host driver supports CUDA up to: ${HOST_CUDA:-unknown}"
cuda_ge() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]; }
if   [ -n "$HOST_CUDA" ] && cuda_ge "$HOST_CUDA" "13.2"; then CU=cu132
elif [ -n "$HOST_CUDA" ] && cuda_ge "$HOST_CUDA" "13.0"; then CU=cu130
elif [ -n "$HOST_CUDA" ] && cuda_ge "$HOST_CUDA" "12.6"; then CU=cu126
else echo "ERROR: host driver CUDA '${HOST_CUDA}' too old; need >=12.6 for the FA3 wheel"; exit 1; fi
echo ">> using CUDA backend: $CU"

# ---------- uv ----------
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
UV_FLAGS="--system --break-system-packages"; [ -n "${VIRTUAL_ENV:-}" ] && UV_FLAGS=""
PIP="uv pip install $UV_FLAGS"

# ---------- torch 2.12 matching the host CUDA ----------
$PIP "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${CU}"

# ---------- FlashAttention-3 (Hopper; prebuilt, no JIT build) ----------
WHL="flash_attn_3-3.0.0+${CU}torch2.12gite2743ab-cp39-abi3-linux_x86_64.whl"
wget -nc -q "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/${FA3_TAG}/${WHL}"
$PIP "$WHL"

# ---------- remaining deps ----------
$PIP -U "transformers==5.5.0"   # native MiniMaxM2 + FP8 (FineGrainedFP8) + fused-experts (grouped_mm)
$PIP liger-kernel               # LigerFusedLinearCrossEntropyLoss (avoids the 200k-vocab logits)
# kernels MUST be pinned to transformers 5.5.0's range (>=0.12,<0.13). The latest kernels (0.15.x)
# makes LayerRepository(version=...) MANDATORY, but transformers 5.5.0's hub_kernels.py builds its
# _KERNEL_MAPPING with version-less LayerRepository(...) at import time -> `import transformers`
# itself dies with "Either a revision or a version must be specified." (USE_HUB_KERNELS=NO does
# NOT help — the mapping is built unconditionally).
$PIP "kernels>=0.12.0,<0.13"    # finegrained-fp8 / deep-gemm hub kernels (inference only; training dequants)
$PIP accelerate                 # device_map="auto" for merge_infer + low_cpu_mem_usage load
$PIP wandb psutil pynvml
$PIP "git+https://github.com/Scicom-AI-Enterprise-Organization/ChiniDataset.git"
$PIP -U huggingface_hub hf_transfer

# ---------- pre-flight: LoRA + fused-MoE correctness (CPU, no GPU/model needed) ----------
echo ">> LoRA correctness test (fused grouped-MoE math + grads)"
MINIMAX_GROUPED_FALLBACK=1 python test_lora.py

if [ "$MINIMAX_DEPS_ONLY" = "1" ]; then echo ">> MINIMAX_DEPS_ONLY=1 — install + test done, skipping train"; exit 0; fi

# ---------- data ----------
if [ ! -d ./packed_data ]; then
  if [ -n "$DATA_ID" ]; then
    echo ">> downloading prebuilt packed dataset $DATA_ID"
    hf download "$DATA_ID" --repo-type dataset --local-dir ./packed_data
  else
    echo ">> building packed dataset locally (pack_dataset.py, max_seq_len=$PACK_MAX_SEQ_LEN)"
    python pack_dataset.py --out ./packed_data --max-seq-len "$PACK_MAX_SEQ_LEN" \
      ${PACK_MAX_ROWS:+--max-rows "$PACK_MAX_ROWS"}
  fi
fi

# ---------- model (download once; all ranks then read the shared HF cache) ----------
# ~230GB FP8 across 130 shards. Exclude redundant formats.
hf download "$MODEL_ID" --exclude "original/*" "*.pth" "*.gguf" "consolidated*"

# ---------- train ----------
echo ">> launching training on $NPROC GPUs"
torchrun --nproc_per_node="$NPROC" minimax_m2.py "$@"
