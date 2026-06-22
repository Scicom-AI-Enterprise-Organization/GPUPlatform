#!/bin/bash
# Generate a voice-cloning demo set with the finetuned model: per speaker,
# 1 reference + N synthesized held-out sentences. Assembles into samples/.
set -euo pipefail
JOB_DIR="$(cd "$(dirname "$0")" && pwd)"
OMNI_DIR="${OMNI_DIR:-/root/OmniVoice}"
FT_MODEL="${FT_MODEL:-Scicom-intl/omnivoice-tmvoice}"
N="${N:-5}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$OMNI_DIR:${PYTHONPATH:-}"
cd "$JOB_DIR"

echo "### install"
if [ ! -d "$OMNI_DIR/.git" ]; then git clone https://github.com/k2-fsa/OmniVoice "$OMNI_DIR"; fi
pip install --break-system-packages -q -e "$OMNI_DIR" "huggingface_hub[cli]" pyarrow
python -c "import omnivoice; print('omnivoice OK')"

echo "### prepare data (need the real TM clips as reference voices)"
python "$JOB_DIR/prepare_data.py" --out_dir "$JOB_DIR/data" --n_test 50 --seed 42

echo "### build sample list"
python "$JOB_DIR/make_samples.py" --data_dir "$JOB_DIR/data" \
  --out_jsonl "$JOB_DIR/samples_test.jsonl" --ref_dir "$JOB_DIR/samples_ref" --n "$N"

echo "### synthesize with finetune ($FT_MODEL)"
CUDA_VISIBLE_DEVICES=0 python -m omnivoice.cli.infer_batch --model "$FT_MODEL" \
  --test_list "$JOB_DIR/samples_test.jsonl" --res_dir "$JOB_DIR/samples_gen" \
  --preprocess_prompt False --postprocess_output False --batch_duration 600 --audio_chunk_threshold 1000

echo "### assemble samples/"
rm -rf "$JOB_DIR/samples" && mkdir -p "$JOB_DIR/samples"
cp "$JOB_DIR/samples_gen/"*.wav "$JOB_DIR/samples/" 2>/dev/null || true
cp "$JOB_DIR/samples_ref/"*.wav "$JOB_DIR/samples/" 2>/dev/null || true
cp "$JOB_DIR/samples_ref/manifest.json" "$JOB_DIR/samples/" 2>/dev/null || true
echo "  samples/: $(ls "$JOB_DIR/samples" | wc -l) files"
ls -la "$JOB_DIR/samples"
echo "=== sample done ==="
