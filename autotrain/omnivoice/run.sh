#!/bin/bash
# End-to-end OmniVoice finetune on TM-Voice (RunPod, 1x H100).
#
# Staged like OmniVoice's own scripts so it can be run/verified incrementally:
#   STAGE=0 STOP_STAGE=3 bash run.sh   # install -> prepare -> tokenize -> count
#   STAGE=4 STOP_STAGE=4 bash run.sh   # train (run under tmux/nohup; "patient")
#   STAGE=5 STOP_STAGE=8 bash run.sh   # clean -> generate -> eval -> push
#
# Requires HF_TOKEN in env (export from ../.env before scp/ssh).  Run from the
# job dir (this file's dir).  Hard rules: work under /root, never /workspace.
set -euo pipefail

stage=${STAGE:-0}
stop_stage=${STOP_STAGE:-8}

JOB_DIR="$(cd "$(dirname "$0")" && pwd)"
OMNI_DIR="${OMNI_DIR:-/root/OmniVoice}"
EXP_DIR="${EXP_DIR:-$JOB_DIR/exp/omnivoice_tmvoice}"
TOKEN_DIR="${TOKEN_DIR:-$JOB_DIR/data/finetune/tokens}"
CLEAN_DIR="${CLEAN_DIR:-$JOB_DIR/exp/clean}"
RESULTS_DIR="${RESULTS_DIR:-$JOB_DIR/results}"
HF_REPO="${HF_REPO:-Scicom-intl/omnivoice-tmvoice}"
EPOCHS="${EPOCHS:-2}"
TOKENIZER_PATH="${TOKENIZER_PATH:-eustlb/higgs-audio-v2-tokenizer}"
ATTN="${ATTN:-flex_attention}"   # set ATTN=sdpa to fall back

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
# Xet downloads STALL on this pod (603MB higgs model.safetensors never finalizes,
# leaving *.incomplete blobs) — same issue as the gateway's HF-Xet stall note.
# Force plain HTTPS downloads for every HF pull (higgs codec, k2-fsa/OmniVoice, whisper).
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$OMNI_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
cd "$JOB_DIR"

echo "=== OmniVoice/TM-Voice run | stages $stage..$stop_stage | attn=$ATTN | repo=$HF_REPO ==="

# Stage 0: clone + install OmniVoice (reuse the pod's preinstalled torch)
if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
  echo "### Stage 0: install"
  if [ ! -d "$OMNI_DIR/.git" ]; then
    git clone https://github.com/k2-fsa/OmniVoice "$OMNI_DIR"
  fi
  python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available())"
  # RunPod base-image /usr python is PEP-668 externally-managed; install into it
  # (torch 2.8+cu128 already lives there) with --break-system-packages.
  PIP="pip install --break-system-packages"
  $PIP -e "$OMNI_DIR[eval]"
  # UTMOSv2 MOS — same package the gateway TTS eval uses (imports as `utmosv2`).
  $PIP "git+https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2"
  $PIP hf_transfer "huggingface_hub[cli]" pyarrow
  python -c "import omnivoice, jiwer, utmosv2; print('omnivoice import OK')"
fi

# Stage 1: download + unzip + per-speaker split -> train/dev/eval_test jsonl
if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
  echo "### Stage 1: prepare data"
  python "$JOB_DIR/prepare_data.py" --out_dir "$JOB_DIR/data" --n_test 50 --seed 42
fi

# Stage 2: tokenize audio -> WebDataset shards (needs GPU)
if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
  echo "### Stage 2: tokenize audio"
  # Use the sequential tokenizer (tokenize_audio.py) instead of the stock
  # omnivoice.scripts.extract_audio_tokens: the stock script's ProcessPoolExecutor
  # + 24-worker DataLoader spawns ~27 torch processes that all hammer the (Xet-
  # stalling) higgs download at once. The sequential version loads the codec once
  # and encodes ~2.4k short clips on one GPU in a few minutes — same shard layout.
  for split in train dev; do
    echo "  tokenizing $split -> $TOKEN_DIR/$split"
    CUDA_VISIBLE_DEVICES=0 python "$JOB_DIR/tokenize_audio.py" \
      --input_jsonl "$JOB_DIR/data/${split}.jsonl" \
      --out_dir "$TOKEN_DIR/${split}" \
      --tokenizer_path "$TOKENIZER_PATH" --samples_per_shard 64
    echo "  -> $TOKEN_DIR/${split}/data.lst ($(wc -l < "$TOKEN_DIR/${split}/data.lst") shards)"
  done
fi

# Stage 3: count steps/epoch and patch the config for EPOCHS passes
if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
  echo "### Stage 3: count steps/epoch -> patch train_config.json"
  # keep attn + workers consistent with training when counting
  python - "$ATTN" <<'PY'
import json,sys
p="train_config.json"; c=json.load(open(p)); c["attn_implementation"]=sys.argv[1]
json.dump(c,open(p,"w"),indent=4)
PY
  spe=$(CUDA_VISIBLE_DEVICES=0 python "$JOB_DIR/count_steps.py" \
        --train_config "$JOB_DIR/train_config.json" \
        --data_config "$JOB_DIR/data_config.json" | grep STEPS_PER_EPOCH | cut -d= -f2)
  echo "  steps_per_epoch=$spe"
  python - "$spe" "$EPOCHS" <<'PY'
import json,sys
spe=int(sys.argv[1]); ep=int(sys.argv[2]); p="train_config.json"
c=json.load(open(p))
c["steps"]=spe*ep
c["save_steps"]=spe        # checkpoint at the end of each epoch
c["eval_steps"]=spe        # eval-loss at the end of each epoch
c["warmup_steps"]=0; c["warmup_type"]="ratio"
json.dump(c,open(p,"w"),indent=4)
print(f"[patch] steps={spe*ep} save_steps={spe} eval_steps={spe}")
PY
fi

# Stage 4: finetune (run me under tmux/nohup -- "patient")
if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
  echo "### Stage 4: train"
  mkdir -p "$EXP_DIR"
  accelerate launch --gpu_ids 0 --num_processes 1 \
    -m omnivoice.cli.train \
    --train_config "$JOB_DIR/train_config.json" \
    --data_config "$JOB_DIR/data_config.json" \
    --output_dir "$EXP_DIR" 2>&1 | tee "$EXP_DIR/run_train.log"
fi

# Stage 5: build clean, self-contained inference checkpoint
if [ "$stage" -le 5 ] && [ "$stop_stage" -ge 5 ]; then
  echo "### Stage 5: clean checkpoint"
  last_ckpt=$(ls -d "$EXP_DIR"/checkpoint-* | sort -t- -k2 -n | tail -1)
  echo "  final checkpoint: $last_ckpt"
  python "$JOB_DIR/make_clean_checkpoint.py" --ckpt "$last_ckpt" --out "$CLEAN_DIR" \
    --higgs "$TOKENIZER_PATH"
fi

# Stage 6: synthesize the 100 held-out test files (voice cloning)
if [ "$stage" -le 6 ] && [ "$stop_stage" -ge 6 ]; then
  echo "### Stage 6: generate eval audio"
  mkdir -p "$RESULTS_DIR"
  CUDA_VISIBLE_DEVICES=0 python -m omnivoice.cli.infer_batch \
    --model "$CLEAN_DIR" \
    --test_list "$JOB_DIR/data/eval_test.jsonl" \
    --res_dir "$RESULTS_DIR" \
    --preprocess_prompt False --postprocess_output False \
    --batch_duration 600 --audio_chunk_threshold 1000
  echo "  generated: $(ls "$RESULTS_DIR" | wc -l) wavs"
fi

# Stage 7: CER + MOS (gateway methodology: Whisper-large-v3 + UTMOSv2)
if [ "$stage" -le 7 ] && [ "$stop_stage" -ge 7 ]; then
  echo "### Stage 7: eval CER + MOS"
  CUDA_VISIBLE_DEVICES=0 python "$JOB_DIR/tts_eval.py" \
    --wav_dir "$RESULTS_DIR" \
    --test_list "$JOB_DIR/data/eval_test.jsonl" \
    --methods cer,mos \
    --out "$JOB_DIR/eval_results.json"
fi

# Stage 8: push the clean checkpoint + eval results to HF
if [ "$stage" -le 8 ] && [ "$stop_stage" -ge 8 ]; then
  echo "### Stage 8: push to HF ($HF_REPO)"
  # Write a model card (embeds the eval metrics) into the clean dir, then push.
  python - "$CLEAN_DIR" "$JOB_DIR/eval_results.json" "$HF_REPO" <<'PY'
import json, sys, os
clean, evalp, repo = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(evalp)) if os.path.exists(evalp) else {}
def fmt(x): return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"
cer, mos = r.get("cer"), r.get("mos")
cerL, mosL = r.get("cer_per_language", {}), r.get("mos_per_language", {})
card = f"""---
license: apache-2.0
base_model: k2-fsa/OmniVoice
tags: [text-to-speech, tts, omnivoice, voice-cloning, finetune]
language: [en, zh]
---

# OmniVoice finetuned on TM-Voice

Full finetune of [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) (multilingual
zero-shot TTS, Qwen3-0.6B-based diffusion LM) on **Scicom-intl/TM-Voice** for **2 epochs**.

## Data
Per-speaker split, holding out 50 test clips per speaker:
- **TM_English**: 1203 train / 50 test
- **TM_Mandarin**: 1141 train / 50 test (Hanzi transcripts; pinyin duplicates dropped)

## Evaluation ({r.get('num_test','?')} held-out clips, zero-shot voice cloning)
Gateway-style metrics — Whisper-large-v3 CER + UTMOSv2 MOS:

| metric | overall | en | zh |
|---|---|---|---|
| CER ↓ | {fmt(cer)} | {fmt(cerL.get('en'))} | {fmt(cerL.get('zh'))} |
| MOS ↑ | {fmt(mos)} | {fmt(mosL.get('en'))} | {fmt(mosL.get('zh'))} |

## Usage
```python
from omnivoice import OmniVoice
import soundfile as sf, torch
model = OmniVoice.from_pretrained("{repo}", device_map="cuda:0", dtype=torch.float16)
audio = model.generate(text="Hello from TM-Voice.", ref_audio="ref.wav", ref_text="...")
sf.write("out.wav", audio[0], 24000)
```
Trained via `autotrain/omnivoice/` (RunPod 1xH100). Includes the bundled Higgs audio tokenizer.
"""
open(os.path.join(clean, "README.md"), "w").write(card)
print("[card] wrote README.md")
PY
  hf upload "$HF_REPO" "$CLEAN_DIR" . --repo-type model
  hf upload "$HF_REPO" "$JOB_DIR/eval_results.json" eval_results.json --repo-type model
  echo "  pushed -> https://huggingface.co/$HF_REPO"
fi

echo "=== done (stages $stage..$stop_stage) ==="
