# omnivoice — finetune k2-fsa/OmniVoice on Scicom-intl/TM-Voice

Self-contained training job (sibling of `gemma4/`, `minimax-m2/`, `mistral-small/`), but
**unlike** them this one **uses OmniVoice's own stock training pipeline** (`omnivoice.cli.train`)
— a **full** finetune of a small (Qwen3-0.6B-based) diffusion-LM zero-shot TTS model, **not a
LoRA**, and with **no custom forward path**. So the autotrain `compare_logits.py` rule does
**not** apply here (nothing replaces a stock transformers forward / no custom kernel/dequant).

## What it does
1. **Data** — `Scicom-intl/TM-Voice` (gated): `english.zip` + `Mandarin_2026-06-12.zip` (must be
   **unzipped**) + a parquet `[speaker, filename_audio, text]` (3635 rows). Two speakers:
   `TM_English` (1253), `TM_Mandarin` (2382). `prepare_data.py` unzips, resolves each
   `filename_audio` → wav, and does a **per-speaker** split: hold out **50 test per speaker**
   (→ 100 test, 3535 train). Emits OmniVoice manifests: `train.jsonl`, `dev.jsonl` (= the test
   set, for eval-loss), `eval_test.jsonl` (id/text/ref_audio/ref_text/language_id, ref = a random
   **same-speaker train clip** for zero-shot voice-cloning eval). `language_id`=`en`/`zh`.
2. **Train** — OmniVoice Stage 0 (`extract_audio_tokens`, tokenizer
   `eustlb/higgs-audio-v2-tokenizer`, needs GPU) → WebDataset shards; Stage 1
   `accelerate launch -m omnivoice.cli.train`. The trainer is **step-based** (`trainer.py`
   loops `while global_step < steps`; epoch++ on `StopIteration`), so **2 epochs** = set
   `steps = 2 × steps_per_epoch`. `count_steps.py` builds the *real* train dataloader (1 proc,
   no DDP) and counts packed batches in one pass; `run.sh` patches `train_config.json`
   (`steps`, `save_steps`=`eval_steps`=`steps_per_epoch` → a checkpoint + eval each epoch).
3. **Eval** — `make_clean_checkpoint.py` strips Accelerate optimizer state + bundles the Higgs
   `audio_tokenizer/` (self-contained) → `omnivoice-infer-batch` synthesizes the 100 test files
   → `tts_eval.py` scores **CER** (Whisper-large-v3 + `jiwer.cer`, per-language) + **MOS**
   (UTMOSv2), **mirroring `gateway/.../training/tts/tts_eval.py`**.
4. **Push** — clean checkpoint + `eval_results.json` → `Scicom-intl/omnivoice-tmvoice`.

`run.sh` is staged (`STAGE`/`STOP_STAGE`, 0..8) so you can run + verify incrementally.

## RunPod recipe (1× H100, US)
Keys from `../.env` (`RUNPOD_API_KEY`, `HF_TOKEN`). Single GPU keeps the epoch→steps count
exact (no DDP shard split). Hard rules: `--country-code US`, work under `/root/...` (never
`/workspace`), generous `--container-disk-in-gb`.
- OmniVoice pins **torch 2.8.0 + cu128** → pick a **CUDA 12.8** PyTorch pod image (NOT the
  cu13/torch-2.12 + FA3 image the gemma4/MoE siblings use). `pip install -e .[eval]` reuses the
  pod's preinstalled torch (dep is `torch>=2.4`, so 2.8 is kept).
- `hf auth login --token "$HF_TOKEN"`; run Stage 4 (train) under **tmux/nohup** ("patient").
- **Always `runpodctl pod delete <id>`** when done — real billing.

## Gotchas
- **flex_attention** (default) may fail to build on some H100 images → set `ATTN=sdpa` (uses the
  length-grouped padding path; README documents this). `count_steps.py` must use the *same* attn
  + `num_workers` as training, else the per-epoch batch count drifts.
- Training checkpoints are HF-format **but don't include the Higgs codec** → `from_pretrained`
  re-downloads `eustlb/higgs-audio-v2-tokenizer` unless `make_clean_checkpoint.py` bundled
  `audio_tokenizer/` (it does).
- `infer_batch --model` accepts a **local dir** (`_resolve_model_path`). Output wavs are named
  `{id}.wav` @ 24 kHz; `tts_eval.py` reads them back by id.
- Pushing to the `Scicom-intl` org needs the `HF_TOKEN` to have **write** to that org.
