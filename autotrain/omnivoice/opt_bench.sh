#!/bin/bash
# Optimization campaign: sweep serving configs and measure SPEED + ACCURACY (CER)
# together, so we only adopt changes that don't drop accuracy.
#   - compile off/on   (torch.compile reduce-overhead = CUDA graphs; accuracy-neutral)
#   - num_step 32/24/16 (fewer diffusion steps = faster but may cost CER -> gated)
# Throughput from bench_serve.py (c=32), CER from serving 100 held-out texts -> Whisper.
set -euo pipefail
JOB_DIR="$(cd "$(dirname "$0")" && pwd)"
OMNI_DIR="${OMNI_DIR:-/root/OmniVoice}"
FT="${FT_MODEL:-Scicom-intl/omnivoice-tmvoice}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$OMNI_DIR:${PYTHONPATH:-}"
cd "$JOB_DIR"; mkdir -p logs opt

echo "### install"
if [ ! -d "$OMNI_DIR/.git" ]; then git clone https://github.com/k2-fsa/OmniVoice "$OMNI_DIR"; fi
pip install --break-system-packages -q -e "$OMNI_DIR" "huggingface_hub[cli]" pyarrow \
  "fastapi" "uvicorn[standard]" "httpx" "pydub" jiwer
python -c "import omnivoice, fastapi, httpx, pydub, jiwer; print('deps OK')"

echo "### prepare data + voices"
python "$JOB_DIR/prepare_data.py" --out_dir "$JOB_DIR/data" --n_test 50 --seed 42
python - <<'PY'
import json, soundfile as sf
spk={"en":"tm_english","zh":"tm_mandarin"}; rows=[json.loads(l) for l in open("data/train.jsonl") if l.strip()]
by={}
[by.setdefault(r["language_id"],[]).append(r) for r in rows]
v={}
for lang,items in by.items():
    pick=items[0]
    for r in items:
        try: d=sf.info(r["audio_path"]).duration
        except Exception: continue
        if 3.0<=d<=10.0: pick=r; break
    v[spk.get(lang,lang)]={"ref_audio":pick["audio_path"],"ref_text":pick["text"],"language":lang}
json.dump(v,open("voices.json","w"),ensure_ascii=False,indent=2); print("voices",list(v))
PY

RESULTS="opt/opt_results.tsv"; echo -e "config\tcompile\tnum_step\trps_c32\tRTF_c32\tCER" > "$RESULTS"

run_cfg() {  # label compile num_step
  local label=$1 comp=$2 step=$3
  echo "### CONFIG $label (compile=$comp num_step=$step)"
  [ -f serve.pid ] && kill "$(cat serve.pid)" 2>/dev/null || true; sleep 3
  OV_MODEL="$FT" OV_VOICES="$JOB_DIR/voices.json" OV_MAX_BATCH=32 OV_NUM_STEP="$step" \
    OV_COMPILE="$comp" PORT=8000 nohup python "$JOB_DIR/serve.py" > "logs/serve_$label.log" 2>&1 &
  echo $! > serve.pid
  for i in $(seq 1 180); do
    curl -sf localhost:8000/health >/dev/null 2>&1 && { echo "  up"; break; }
    kill -0 "$(cat serve.pid)" 2>/dev/null || { echo "  DIED"; tail -25 "logs/serve_$label.log"; return 0; }
    sleep 5
  done
  python "$JOB_DIR/bench_serve.py" --concurrency 1,16,32 --requests_per_level 96 --format wav \
    --out "opt/bench_$label.json" | tee "logs/bench_$label.txt"
  rm -rf "served_$label"
  python "$JOB_DIR/serve_client.py" --test_list data/eval_test.jsonl --out_dir "served_$label" --concurrency 32
  CUDA_VISIBLE_DEVICES=0 python "$JOB_DIR/tts_eval.py" --wav_dir "served_$label" \
    --test_list data/eval_test.jsonl --methods cer --out "opt/cer_$label.json"
  kill "$(cat serve.pid)" 2>/dev/null || true; sleep 2
  python - "$label" "$comp" "$step" >> "$RESULTS" <<'PY'
import json, sys
label, comp, step = sys.argv[1], sys.argv[2], sys.argv[3]
b = json.load(open(f"opt/bench_{label}.json")); x = [z for z in b if z["concurrency"] == 32][0]
cer = json.load(open(f"opt/cer_{label}.json"))["cer"]
print(f"{label}\t{comp}\t{step}\t{x['throughput_rps']}\t{x['rtf_x_realtime']}\t{cer:.4f}")
PY
}

run_cfg baseline   0 32
run_cfg compiled   1 32
run_cfg comp_s24   1 24
run_cfg comp_s16   1 16

echo "=== OPT RESULTS (speed vs accuracy) ==="; column -t -s $'\t' "$RESULTS"
echo "=== opt_bench done ==="
