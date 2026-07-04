#!/usr/bin/env bash
# Ring (context-parallel) attention correctness test — install the attention kernels and run
# test_ring_attn.py across every GPU on the pod (needs >=2). No model / HF token needed: the test
# is pure random tensors on Gemma-4 attention geometry.
#
#   bash run_ring_test.sh                 # cute-only (pod torch) — head_dim-512 ring
#   GEMMA_FA3=1 bash run_ring_test.sh     # HYBRID: torch 2.12 + FA3 wheel + cute → tests the fa3 backend too
#   RING_DEPS_ONLY=1 [GEMMA_FA3=1] bash run_ring_test.sh   # install only
set -euo pipefail
cd "$(dirname "$0")"

FA4_FORK_REPO="${FA4_FORK_REPO:-https://github.com/Scicom-AI-Enterprise-Organization/flash-attention-512.git}"
FA4_FORK_DIR="${FA4_FORK_DIR:-./flash-attention-512}"
GEMMA_FA3="${GEMMA_FA3:-0}"            # 1 = also install torch 2.12 + the FA3 prebuilt wheel (hybrid)
TORCH_VERSION="${TORCH_VERSION:-2.12.0}"
FA3_TAG="${FA3_TAG:-v0.9.18}"

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"       # NVLS multicast bind fails in RunPod containers
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED="${FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
NPROC="$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)"
echo ">> GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  nproc_per_node=$NPROC"
[ "$NPROC" -ge 2 ] || { echo "ERROR: ring test needs >=2 GPUs"; exit 1; }

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
UV_FLAGS="--system --break-system-packages"; [ -n "${VIRTUAL_ENV:-}" ] && UV_FLAGS=""
PIP="uv pip install $UV_FLAGS"

# ---- host CUDA backend (FA3 wheel ships cu126/cu130/cu132) ----
HOST_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
cuda_ge() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]; }
if   cuda_ge "$HOST_CUDA" "13.2"; then CU=cu132
elif cuda_ge "$HOST_CUDA" "13.0"; then CU=cu130
else CU=cu126; fi
echo ">> host CUDA $HOST_CUDA -> $CU  (GEMMA_FA3=$GEMMA_FA3)"

if [ "$GEMMA_FA3" = "1" ]; then
  # Hybrid env: torch 2.12 (FA3 wheel ABI) + the FA3 prebuilt wheel. cute is torch-flexible and
  # rides along on 2.12. FA3 installs as `flash_attn_interface`; cute is `flash_attn.cute` — distinct
  # top-level modules, no shadow clash (verified on the pod).
  $PIP "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${CU}"
  WHL="flash_attn_3-3.0.0+${CU}torch2.12gite2743ab-cp39-abi3-linux_x86_64.whl"
  wget -nc -q "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/${FA3_TAG}/${WHL}"
  $PIP "$WHL"
  python -c "import flash_attn_interface as f; print('FA3 import OK:', [n for n in dir(f) if 'varlen' in n])"
fi

# FA4 cute fork (adds symmetric head_dim=512 fwd+bwd on SM90).
if [ ! -d "$FA4_FORK_DIR/flash_attn/cute" ]; then
  echo ">> cloning $FA4_FORK_REPO -> $FA4_FORK_DIR"; git clone --depth 1 "$FA4_FORK_REPO" "$FA4_FORK_DIR"
else echo ">> reusing existing $FA4_FORK_DIR"; fi
$PIP -e "$FA4_FORK_DIR/flash_attn/cute"
# Pin: the >= bounds in cute's pyproject pull too-new builds that break (quack 0.5 / cutlass 4.5).
$PIP "nvidia-cutlass-dsl[cu13]==4.4.2" "quack-kernels==0.3.10"
python -c "from flash_attn.cute.interface import _flash_attn_fwd; print('FA4 cute import OK')"

if [ "${RING_DEPS_ONLY:-0}" = "1" ]; then echo ">> RING_DEPS_ONLY=1 — done"; exit 0; fi

echo ">> launching ring test on $NPROC GPUs"
torchrun --nproc_per_node="$NPROC" test_ring_attn.py
