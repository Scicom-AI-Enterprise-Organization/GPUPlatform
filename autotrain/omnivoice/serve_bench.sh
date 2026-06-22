#!/bin/bash
# Stand up the OmniVoice /v1/audio/speech server and benchmark it on the H100.
# Runs the sweep twice: MAX_BATCH=1 (dynamic batching OFF = baseline) vs
# MAX_BATCH=32 (ON), to isolate the batching speedup. Client runs on localhost.
set -euo pipefail
JOB_DIR="$(cd "$(dirname "$0")" && pwd)"
OMNI_DIR="${OMNI_DIR:-/root/OmniVoice}"
FT_MODEL="${FT_MODEL:-Scicom-intl/omnivoice-tmvoice}"
NUM_STEP="${NUM_STEP:-32}"
CONC="${CONC:-1,2,4,8,16,32,64}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$OMNI_DIR:${PYTHONPATH:-}"
cd "$JOB_DIR"; mkdir -p logs

echo "### install"
if [ ! -d "$OMNI_DIR/.git" ]; then git clone https://github.com/k2-fsa/OmniVoice "$OMNI_DIR"; fi
pip install --break-system-packages -q -e "$OMNI_DIR" "huggingface_hub[cli]" pyarrow \
  "fastapi" "uvicorn[standard]" "httpx" "pydub"
python -c "import omnivoice, fastapi, uvicorn, httpx, pydub; print('serve deps OK')"

echo "### prepare data (need TM clips as reference voices)"
python "$JOB_DIR/prepare_data.py" --out_dir "$JOB_DIR/data" --n_test 50 --seed 42

echo "### build voices.json (one 3-10s reference per speaker)"
python - <<'PY'
import json, soundfile as sf
spk = {"en": "tm_english", "zh": "tm_mandarin"}
rows = [json.loads(l) for l in open("data/train.jsonl") if l.strip()]
by = {}
for r in rows: by.setdefault(r["language_id"], []).append(r)
voices = {}
for lang, items in by.items():
    pick = items[0]
    for r in items:
        try:
            d = sf.info(r["audio_path"]).duration
        except Exception: continue
        if 3.0 <= d <= 10.0: pick = r; break
    voices[spk.get(lang, lang)] = {"ref_audio": pick["audio_path"], "ref_text": pick["text"], "language": lang}
json.dump(voices, open("voices.json", "w"), ensure_ascii=False, indent=2)
print("voices:", list(voices))
PY

start_server() {  # $1 = MAX_BATCH
  [ -f serve.pid ] && kill "$(cat serve.pid)" 2>/dev/null || true; sleep 2
  OV_MODEL="$FT_MODEL" OV_VOICES="$JOB_DIR/voices.json" OV_MAX_BATCH="$1" \
    OV_NUM_STEP="$NUM_STEP" PORT=8000 nohup python "$JOB_DIR/serve.py" > "logs/serve_b$1.log" 2>&1 &
  echo $! > serve.pid
  echo "  waiting for /health (MAX_BATCH=$1) ..."
  for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then echo "  server up"; return 0; fi
    if ! kill -0 "$(cat serve.pid)" 2>/dev/null; then echo "  SERVER DIED"; tail -20 "logs/serve_b$1.log"; return 1; fi
    sleep 5
  done; echo "  health timeout"; return 1
}

echo "### BASELINE: dynamic batching OFF (MAX_BATCH=1)"
start_server 1
python "$JOB_DIR/bench_serve.py" --concurrency "$CONC" --requests_per_level 96 \
  --format wav --out "$JOB_DIR/bench_nobatch.json" | tee logs/bench_nobatch.txt

echo "### OPTIMIZED: dynamic batching ON (MAX_BATCH=32)"
start_server 32
( nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 1 > logs/gpu_bench.log 2>&1 & echo $! > gpu.pid )
python "$JOB_DIR/bench_serve.py" --concurrency "$CONC" --requests_per_level 96 \
  --format wav --out "$JOB_DIR/bench_batch.json" | tee logs/bench_batch.txt
kill "$(cat gpu.pid)" 2>/dev/null || true
kill "$(cat serve.pid)" 2>/dev/null || true

echo "### peak GPU util during batched run:"
sort -t, -k1 -n logs/gpu_bench.log | tail -1
echo "=== serve_bench done ==="
