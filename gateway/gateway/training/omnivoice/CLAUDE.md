# Autotrain — OmniVoice TTS (vendored)

A SECOND TTS model in autotrain, alongside the Qwen3+NeuCodec path. OmniVoice
(`k2-fsa/OmniVoice`, a Qwen3-0.6B diffusion-LM zero-shot TTS) is a fundamentally
different stack — its OWN trainer (`accelerate -m omnivoice.cli.train`, or the
vendored `lora_train.py` when `use_lora` is set — see below), **Higgs** codec audio
tokens packed as **WebDataset shards** (not NeuCodec→ChiniDataset), torch **2.8/cu128**.
Standalone authority: `autotrain/omnivoice/` — these files are vendored copies;
**re-sync on change** (except `lora_train.py`, which is gateway-only — see below).

```
omnivoice_finetune.py   (one dir up) the orchestrator shipped to the box for a TTS run
                        whose base model is OmniVoice. Self-contained deps-resilience;
                        imports build_dataset + S3 helpers from tts_finetune.py (shipped
                        alongside). Two modes: pack-only (→ @@PACKED) and train.
lora_train.py           GATEWAY-ONLY (not in the standalone repo — no upstream LoRA support
                        to sync against). Drop-in alternative to `-m omnivoice.cli.train`:
                        same build_model_and_tokenizer/build_dataloaders/OmniTrainer calls,
                        plus a peft LoRA wrap on `model.llm` (the Qwen3 backbone) and a
                        merge-back-to-plain-checkpoint step at the end. See "LoRA" below.
tokenize_audio.py       Higgs-codec encode each clip → WebDataset shards (verbatim)
count_steps.py          steps-per-epoch from the real packing dataloader (verbatim)
make_clean_checkpoint.py strip optimizer state + bundle the Higgs codec (verbatim)
tts_eval.py             Whisper-v3 CER + UTMOSv2 MOS (verbatim; mirrors ../tts/tts_eval.py)
train_config.json / data_config.json   OmniVoice configs (steps/init patched at runtime)
```

## LoRA (added 2026-07-20)

Upstream OmniVoice's own trainer has no PEFT/LoRA wiring (`omnivoice.training.builder.
build_model_and_tokenizer` loads a plain `AutoModel` backbone into `OmniVoice.llm`, full
finetune only). `omnivoice_finetune.py`'s `run()` reads `cfg["use_lora"]` (same field the
web form / sweep already send for ASR/TTS) and, when set, launches the vendored
`lora_train.py` instead of `-m omnivoice.cli.train` — everything else (data config, step
counting, checkpoint cadence, the `@@STEP`-parseable console log) is identical, since
`lora_train.py` calls the exact same public `build_model_and_tokenizer` / `build_dataloaders`
/ `OmniTrainer` functions upstream's own CLI does.

- **Only `model.llm` (the Qwen3 backbone) is LoRA-wrapped** (`target_modules="all-linear"`,
  `task_type=None` → peft's generic `PeftModel`, since `model.llm` has no `lm_head`/
  `generate()` of its own — OmniVoice's `forward()` calls it directly with custom kwargs).
  `LoraConfig(r, lora_alpha, lora_dropout)` from `cfg["lora_r"]` / `lora_alpha_ratio` (α =
  round(r × ratio), mirrors the Whisper/qwen3_tts_flash convention) / `lora_dropout`.
- **OmniVoice's own TTS-specific heads stay fully trainable** — `audio_embeddings` (nn.Embedding,
  per-codebook audio-token embed) and `audio_heads` (nn.Linear, audio-token output projection)
  live OUTSIDE `model.llm` (see `OmniVoice.__init__` upstream) and are never touched by the
  peft wrap. This is deliberate: LoRA the pretrained LLM prior, fully adapt the small
  audio-specific machinery, same split as freezing an encoder while training a head elsewhere
  in this codebase.
- **Merge at save, same contract as qwen3_tts_flash.py.** `lora_train.py` runs
  `trainer.train()` unmodified (so ALL of OmniTrainer's periodic `save_checkpoint` calls
  during training write UN-merged peft-format checkpoints — key names like
  `llm.base_model.model….lora_A/lora_B` — these are never read by anything; the gateway
  orchestrator only ever consumes the highest-numbered `checkpoint-*` dir). After `train()`
  returns, `lora_train.py` calls `model.llm.merge_and_unload()` (destructive, fine — training
  is done) and re-`save_pretrained`s into the **same** `checkpoint-{global_step}` dir
  OmniTrainer's own final save already used, OVERWRITING it with a plain, peft-free state
  dict. `omnivoice_finetune.py`'s existing `ckpts[-1]` selection and `make_clean_checkpoint.py`
  (a dumb file copy) need no changes — they see an ordinary full-finetune-shaped checkpoint
  either way, so CER/MOS eval and any later `OmniVoice.from_pretrained()` need no peft install.
- **Deps**: `peft>=0.11` is now in the shared `/share/autotrain-omnivoice` venv deps
  unconditionally (mirrors `tts_finetune.py`'s `DEPS` — cheap, avoids a conditional install
  path). `_ensure_venv`'s presence check now also probes `import peft`.
- ⚠️ **Not GPU-verified yet** — reasoned from OmniVoice's real upstream source (`builder.py`
  / `trainer.py` / `checkpoint.py` / `models/omnivoice.py`, fetched 2026-07-20) rather than a
  real run. Verify the first real LoRA run: check `model.llm.print_trainable_parameters()`
  in the log shows a small trainable fraction, and that the final checkpoint's `.safetensors`
  keys have NO `lora_`/`base_layer` substrings (confirms the merge actually replaced
  `model.llm` before save, not just merged-in-place on the still-wrapped module).
- Sibling bug fixed alongside this: `tts_finetune.py`'s own `_tgt = str(cfg.get(
  "lora_target_modules") or "all-linear")` always hit the `str(list)` branch (the LLM-only
  target-module field is sent by the form for every task type, never empty) →
  `qwen3_tts_flash.py` received the Python repr `"['q_proj', …]"` → `tgt.split(",")` produced
  bracket/quote-mangled non-module names → peft raised "Target modules … not found" on every
  Scicom TTS LoRA run. Now hardcoded to `"all-linear"` (the TTS form never exposes a picker
  anyway, matching whisper_finetune.py's own hardcoded default).

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
