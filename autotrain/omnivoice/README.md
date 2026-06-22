# OmniVoice × TM-Voice — finetune, eval & optimized serving

Full finetune of [`k2-fsa/OmniVoice`](https://github.com/k2-fsa/OmniVoice) (multilingual zero-shot TTS,
Qwen3-0.6B diffusion LM) on the gated dataset
[`Scicom-intl/TM-Voice`](https://huggingface.co/datasets/Scicom-intl/TM-Voice), with CER/MOS +
speaker-similarity evaluation and an OpenAI-compatible high-concurrency server. Model:
[`Scicom-intl/omnivoice-tmvoice`](https://huggingface.co/Scicom-intl/omnivoice-tmvoice).

## Results (1× H100)
| | CER ↓ (all/en/zh) | MOS ↑ | speaker-sim → ground-truth ↑ |
|---|---|---|---|
| base k2-fsa/OmniVoice | 0.246 | 3.00 | 0.718 |
| **finetune (2 ep)** | 0.246 / .194 / .299 | 3.00 | — |
| **finetune (20 ep, on HF)** | 0.247 / .200 / .294 | 3.00 | **0.768** (en .639→.709) |

CER/MOS are flat 2↔20 epochs (they don't measure timbre); the **finetune's real win is speaker
similarity to the target TM voices** (+0.05). Served-output CER = **0.243 == offline** → the optimized
batched serving path preserves quality.

**Serving** (`/v1/audio/speech`, dynamic batching + length bucketing): **2.0 → 3.5 rps** mixed-length
(1.75×), **5.6 rps** uniform (2.8×, single-GPU ceiling). **Optimization** (gated on CER+MOS): `num_step`
32→**16** gives **~2× more (6.84 rps @c32, RTF 65×)** at flat CER (0.245) + ~flat MOS — the optimized
default. Below 16 plateaus; at 8 quality drops. Scale-out beyond ~2× = data-parallel replicas (linear).

## Files
| file | role |
|---|---|
| `prepare_data.py` | download + unzip TM-Voice; dedup Mandarin pinyin/Hanzi (keep Hanzi); per-speaker 50-test split → `train/dev/eval_test.jsonl` |
| `tokenize_audio.py` | sequential Higgs-codec tokenizer → WebDataset shards (deadlock-free replacement for the stock script) |
| `count_steps.py` | steps-per-epoch (token-packed) so N epochs = `N × spe` |
| `train_config.json` / `data_config.json` | OmniVoice finetune configs (steps patched at runtime) |
| `make_clean_checkpoint.py` | strip optimizer state + bundle Higgs `audio_tokenizer/` |
| `tts_eval.py` | Whisper-large-v3 **CER** + UTMOSv2 **MOS** (mirrors gateway TTS eval) |
| `run.sh` | staged end-to-end driver (`STAGE`/`STOP_STAGE` 0..8, `EPOCHS=`) |
| `ab_eval.py` / `ab.sh` | base-vs-finetune A/B: CER + ECAPA speaker-similarity |
| `make_samples.py` / `sample.sh` | per-speaker demo `samples/` (1 reference + 5 synthesized) |
| `serve.py` | OpenAI `/v1/audio/speech` server (cached voices, dynamic batching, length bucketing) |
| `bench_serve.py` / `serve_client.py` / `serve_bench.sh` | concurrency benchmark + served-CER verifier |
| `opt_bench.sh` / `opt_round2.sh` / `score_dirs.py` | optimization sweep: compile/num_step vs CER+MOS (→ `num_step=16`) |
| `samples/` | committed demo audio (5 clips + 1 reference per speaker) |

## Run (RunPod 1× H100, image `runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2404`)
```bash
export HF_TOKEN=...                 # from ../.env;  export HF_HUB_DISABLE_XET=1  (Xet stalls on RunPod!)
EPOCHS=20 STAGE=0 STOP_STAGE=8 bash run.sh        # full: install→prep→tokenize→count→train→clean→gen→CER/MOS→push
bash ab.sh                                        # base-vs-finetune speaker-similarity A/B
bash sample.sh                                    # generate demo samples/
bash serve_bench.sh                               # stand up /v1/audio/speech + benchmark (batch off vs on)
```
Serve standalone:
```bash
OV_MODEL=Scicom-intl/omnivoice-tmvoice OV_VOICES=voices.json OV_MAX_BATCH=32 python serve.py
curl -s localhost:8000/v1/audio/speech -H 'content-type: application/json' \
  -d '{"input":"Hello from TM-Voice.","voice":"tm_english","response_format":"wav"}' -o out.wav
```
Pip needs `--break-system-packages` on the pod. **Always `runpodctl pod delete <id>`** when done — this job
names pods `omnivoice-*`; `runpodctl pod list | grep omnivoice` should be empty after a session (don't delete
unrelated pods the list may show).
See `CLAUDE.md` for the design, accuracy details, serving/parallelization, and gotchas (esp. the Xet stall).
