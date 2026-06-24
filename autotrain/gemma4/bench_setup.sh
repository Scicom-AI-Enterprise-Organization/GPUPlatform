#!/usr/bin/env bash
# Set up bench_attention.py (FA4 head_dim-512 cute vs vLLM Triton unified attention).
#
# Two pieces:
#   1. Copy vLLM's Triton attention kernel (+ its pure-python helper) from a vLLM checkout
#      into bench_vllm_shim/ — so the benchmark runs the EXACT kernel source without a
#      multi-hour full vLLM CUDA build. bench_vllm_shim/ already ships tiny stand-ins for
#      the handful of vllm internals the kernel imports (envs / logger / platforms /
#      triton_utils / KVQuantMode), faithful to upstream defaults.
#   2. (optional, --venv) build a venv with torch (cu130) + the FA4 cute fork + cutlass-dsl.
#
# Usage:
#   VLLM_REPO=/home/husein/ssd3/vllm bash bench_setup.sh                 # copy kernel files only
#   VLLM_REPO=/path/to/vllm FA4_FORK=/path/to/flash-attention-512 \
#     bash bench_setup.sh --venv /share/gemma4-bench-venv               # + build the venv
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_REPO="${VLLM_REPO:-/home/husein/ssd3/vllm}"
OPS_SRC="$VLLM_REPO/vllm/v1/attention/ops"
OPS_DST="$HERE/bench_vllm_shim/vllm/v1/attention/ops"

echo ">> copying vLLM Triton kernel from $VLLM_REPO"
for f in triton_unified_attention.py triton_attention_helpers.py; do
  [ -f "$OPS_SRC/$f" ] || { echo "MISSING $OPS_SRC/$f — set VLLM_REPO to a vLLM checkout"; exit 1; }
  cp "$OPS_SRC/$f" "$OPS_DST/$f"
  echo "   $f"
done
echo ">> kernel staged into bench_vllm_shim/"

# ---- optional venv build (proven recipe; see gemma4/CLAUDE.md "FA4 ... benchmark") ----
if [ "${1:-}" = "--venv" ]; then
  VENV="${2:?--venv needs a path}"
  FA4_FORK="${FA4_FORK:?set FA4_FORK to the flash-attention-512 checkout}"
  : "${UV_HTTP_TIMEOUT:=300}"; export UV_HTTP_TIMEOUT
  echo ">> building venv at $VENV"
  uv venv "$VENV" --python 3.12
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  uv pip install torch --index-url https://download.pytorch.org/whl/cu130   # torch 2.12 + triton
  uv pip install numpy
  uv pip install -e "$FA4_FORK/flash_attn/cute"
  uv pip install "nvidia-cutlass-dsl[cu13]==4.4.2" "quack-kernels==0.3.10"  # pins; newer break
  python -c "import torch,triton; print('torch',torch.__version__,'triton',triton.__version__)"
  python -c "from flash_attn.cute.interface import flash_attn_varlen_func; print('FA4 cute OK')"
  echo ">> venv ready: source $VENV/bin/activate"
fi

cat <<'EOF'

run it (FA4 cute import needs a dir with NO local flash_attn/ folder — this dir is fine):
  CUDA_VISIBLE_DEVICES=7 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    python bench_attention.py --iters 50 --warmup 10 --out results.json
EOF
