# Autotrain — OmniVoice TTS (vendored)

A SECOND TTS model in autotrain, alongside the Qwen3+NeuCodec path. OmniVoice
(`k2-fsa/OmniVoice`, a Qwen3-0.6B diffusion-LM zero-shot TTS) is a fundamentally
different stack — its OWN trainer (`accelerate -m omnivoice.cli.train`), **Higgs**
codec audio tokens packed as **WebDataset shards** (not NeuCodec→ChiniDataset), a
**full** finetune (no LoRA), torch **2.8/cu128**. Standalone authority:
`autotrain/omnivoice/` — these files are vendored copies; **re-sync on change**.

```
omnivoice_finetune.py   (one dir up) the orchestrator shipped to the box for a TTS run
                        whose base model is OmniVoice. Self-contained deps-resilience;
                        imports build_dataset + S3 helpers from tts_finetune.py (shipped
                        alongside). Two modes: pack-only (→ @@PACKED) and train.
tokenize_audio.py       Higgs-codec encode each clip → WebDataset shards (verbatim)
count_steps.py          steps-per-epoch from the real packing dataloader (verbatim)
make_clean_checkpoint.py strip optimizer state + bundle the Higgs codec (verbatim)
tts_eval.py             Whisper-v3 CER + UTMOSv2 MOS (verbatim; mirrors ../tts/tts_eval.py)
train_config.json / data_config.json   OmniVoice configs (steps/init patched at runtime)
```

## Selection + dispatch
- `task_type=tts` + base model matching OmniVoice → `_tts_arch()=="omnivoice"`
  (`training_api.py`). The dispatch ships `omnivoice_finetune.py` + `tts_finetune.py`
  + this `omnivoice/` dir; venv `/share/autotrain-omnivoice` (separate from the
  Qwen3 `/share/autotrain-tts`).
- Data transform: **Pack for OmniVoice** (`POST /v1/datasets/{id}/pack-omnivoice`,
  a pack-only run with base `k2-fsa/OmniVoice`) → Higgs WebDataset shards on S3 →
  `kind=omnivoice_packed`. Training requires that kind (not `tts_packed`).
- Web: `training-form.tsx` lists `k2-fsa/OmniVoice` (gates the dataset to
  `omnivoice_packed`); `transformation-card.tsx` audio datasets get a TTS/OmniVoice
  pack toggle (reuses `TtsPackCard` with `variant`).

## Gotchas
- **CUDA 12.8** — OmniVoice pins torch 2.8/cu128; on RunPod use a cu1281 pytorch
  image (NOT the cu13 gemma image). The omnivoice venv is separate from cu13 stacks.
- **Higgs/OmniVoice fetched over the box uplink** — the orchestrator reuses the
  slow/flaky-uplink resilience (system-git shallow clone of OmniVoice, `_pip`
  retry+kill, stale-uv-lock clear, `HF_HUB_DISABLE_XET=1` + hf_transfer off).
- **`language_id`** is a config knob (`default_language` / `language_field` /
  per-speaker map) — NOT hard-coded en/zh.
- **`@@STEP` parsing** keys off OmniVoice's trainer stdout (`_STEP_RE`/`_LOSS_RE` in
  `omnivoice_finetune.py`). Verify against `OmniVoice/omnivoice/training/trainer.py`
  the first real run; loosen the regex / fall back to per-checkpoint ticks if needed.
- Standalone `prepare_data.py` is TM-Voice-specific (zips+parquet); the gateway path
  uses `omnivoice_finetune.build_manifests` (generic, off `build_dataset`'s meta.jsonl).
