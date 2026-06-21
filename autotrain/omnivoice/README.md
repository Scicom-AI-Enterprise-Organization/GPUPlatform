# OmniVoice × TM-Voice finetune

Full finetune of [`k2-fsa/OmniVoice`](https://github.com/k2-fsa/OmniVoice) (multilingual
zero-shot TTS, Qwen3-0.6B-based diffusion LM) on the gated HF dataset
[`Scicom-intl/TM-Voice`](https://huggingface.co/datasets/Scicom-intl/TM-Voice) for **2 epochs**,
with **CER + MOS** evaluation on a held-out test set, pushed to
[`Scicom-intl/omnivoice-tmvoice`](https://huggingface.co/Scicom-intl/omnivoice-tmvoice).

## Files
| file | role |
|---|---|
| `prepare_data.py` | download + unzip TM-Voice, per-speaker 50-test split → `train/dev/eval_test.jsonl` |
| `count_steps.py` | count steps-per-epoch (token-packed batches) so 2 epochs = `2 × spe` |
| `train_config.json` / `data_config.json` | OmniVoice finetune configs (steps patched at runtime) |
| `make_clean_checkpoint.py` | strip optimizer state + bundle Higgs `audio_tokenizer/` |
| `tts_eval.py` | Whisper-large-v3 **CER** + UTMOSv2 **MOS** (mirrors the gateway TTS eval) |
| `run.sh` | staged end-to-end driver (`STAGE`/`STOP_STAGE` 0..8) |

## Run (on a RunPod 1× H100, CUDA 12.8 image)
```bash
export HF_TOKEN=...                       # from ../.env
hf auth login --token "$HF_TOKEN"
STAGE=0 STOP_STAGE=3 bash run.sh          # install -> prepare -> tokenize -> count
STAGE=4 STOP_STAGE=4 bash run.sh          # train 2 epochs (run under tmux/nohup)
STAGE=5 STOP_STAGE=8 bash run.sh          # clean -> generate -> CER/MOS -> push
```
See `CLAUDE.md` for the design, the step-based-trainer / 2-epoch math, and gotchas.
Data split: **TM_English** 1203 train / 50 test, **TM_Mandarin** 2332 train / 50 test.
