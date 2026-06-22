#!/bin/bash
# Round 2: push num_step lower (12, 8) and gate on BOTH CER (intelligibility) and
# MOS (naturalness) — fewer diffusion steps mainly hurt MOS, which CER won't catch.
# Reuses the round-1 served audio for 16/24/32. (Run on the same pod, server compiled.)
set -euo pipefail
JOB_DIR="$(cd "$(dirname "$0")" && pwd)"
OMNI_DIR="${OMNI_DIR:-/root/OmniVoice}"
FT="${FT_MODEL:-Scicom-intl/omnivoice-tmvoice}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$OMNI_DIR:${PYTHONPATH:-}"
cd "$JOB_DIR"; mkdir -p logs opt

gen() {  # num_step
  local step=$1
  echo "### generate num_step=$step"
  [ -f serve.pid ] && kill "$(cat serve.pid)" 2>/dev/null || true; sleep 3
  OV_MODEL="$FT" OV_VOICES="$JOB_DIR/voices.json" OV_MAX_BATCH=32 OV_NUM_STEP="$step" \
    OV_COMPILE=1 PORT=8000 nohup python "$JOB_DIR/serve.py" > "logs/serve_s$step.log" 2>&1 &
  echo $! > serve.pid
  for i in $(seq 1 180); do
    curl -sf localhost:8000/health >/dev/null 2>&1 && { echo "  up"; break; }
    kill -0 "$(cat serve.pid)" 2>/dev/null || { echo "  DIED"; tail -25 "logs/serve_s$step.log"; return 0; }
    sleep 5
  done
  python "$JOB_DIR/bench_serve.py" --concurrency 32 --requests_per_level 64 --format wav \
    --out "opt/bench_s$step.json" | tee "logs/bench_s$step.txt"
  rm -rf "served_s$step"
  python "$JOB_DIR/serve_client.py" --test_list data/eval_test.jsonl --out_dir "served_s$step" --concurrency 32
  kill "$(cat serve.pid)" 2>/dev/null || true; sleep 2
}

gen 12
gen 8

echo "### score step curve (CER + MOS)"
CUDA_VISIBLE_DEVICES=0 python "$JOB_DIR/score_dirs.py" --test_list data/eval_test.jsonl \
  --dirs s32=served_baseline s24=served_comp_s24 s16=served_comp_s16 s12=served_s12 s8=served_s8 \
  --out opt/score_steps.json

echo "### throughput @c32 (rps / RTF)"
for s in 32 24 16 12 8; do
  f="opt/bench_s$s.json"; [ "$s" = 32 ] && f="opt/bench_baseline.json"
  [ "$s" = 24 ] && f="opt/bench_comp_s24.json"; [ "$s" = 16 ] && f="opt/bench_comp_s16.json"
  python3 -c "import json
r=json.load(open('$f')); x=[z for z in r if z['concurrency']==32][0]
print('s$s:', x['throughput_rps'],'rps', x['rtf_x_realtime'],'RTF')" 2>/dev/null || echo "s$s: NA"
done
echo "=== round2 done ==="
