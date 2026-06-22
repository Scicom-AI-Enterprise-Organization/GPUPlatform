#!/bin/bash
# A/B: base k2-fsa/OmniVoice vs the TM-Voice finetune, same 100 held-out items.
# Generates with both models, then scores CER + speaker-similarity (ECAPA).
set -euo pipefail
JOB_DIR="$(cd "$(dirname "$0")" && pwd)"
OMNI_DIR="${OMNI_DIR:-/root/OmniVoice}"
BASE_MODEL="${BASE_MODEL:-k2-fsa/OmniVoice}"
FT_MODEL="${FT_MODEL:-Scicom-intl/omnivoice-tmvoice}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$OMNI_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
cd "$JOB_DIR"
INFER_OPTS="--preprocess_prompt False --postprocess_output False --batch_duration 600 --audio_chunk_threshold 1000"

echo "### install"
if [ ! -d "$OMNI_DIR/.git" ]; then git clone https://github.com/k2-fsa/OmniVoice "$OMNI_DIR"; fi
PIP="pip install --break-system-packages -q"
$PIP -e "$OMNI_DIR[eval]"
$PIP "git+https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2" "huggingface_hub[cli]" pyarrow speechbrain
python -c "import omnivoice, jiwer, speechbrain; print('imports OK')"

echo "### prepare data (same seed -> same 100 test items)"
python "$JOB_DIR/prepare_data.py" --out_dir "$JOB_DIR/data" --n_test 50 --seed 42

echo "### generate with BASE ($BASE_MODEL)"
CUDA_VISIBLE_DEVICES=0 python -m omnivoice.cli.infer_batch --model "$BASE_MODEL" \
  --test_list "$JOB_DIR/data/eval_test.jsonl" --res_dir "$JOB_DIR/results_base" $INFER_OPTS
echo "  base wavs: $(ls "$JOB_DIR/results_base" | wc -l)"

echo "### generate with FINETUNE ($FT_MODEL)"
CUDA_VISIBLE_DEVICES=0 python -m omnivoice.cli.infer_batch --model "$FT_MODEL" \
  --test_list "$JOB_DIR/data/eval_test.jsonl" --res_dir "$JOB_DIR/results_ft" $INFER_OPTS
echo "  ft wavs: $(ls "$JOB_DIR/results_ft" | wc -l)"

echo "### A/B scoring (CER + speaker similarity)"
CUDA_VISIBLE_DEVICES=0 python "$JOB_DIR/ab_eval.py" \
  --base_dir "$JOB_DIR/results_base" --ft_dir "$JOB_DIR/results_ft" \
  --test_list "$JOB_DIR/data/eval_test.jsonl" --gt_list "$JOB_DIR/data/dev.jsonl" \
  --out "$JOB_DIR/ab_results.json"
echo "=== ab done ==="
