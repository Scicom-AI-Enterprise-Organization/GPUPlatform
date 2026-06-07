# Audio model training — Whisper (ASR) and TTS (LLM + NeuCodec)

> Intern onboarding doc. Read this top-to-bottom once, then keep it open while you
> read the scripts it points at. Everything here is grounded in the actual code in
> this directory — file + function names are real, go look at them.

This directory holds two **standalone** training pipelines that the gateway's
AutoTrain runner ships to a GPU box (a RunPod pod or the TM VM) over SSH and runs
with `python <script> --config <cfg.json>`:

| Task | Entry script | What it produces |
|------|--------------|------------------|
| **ASR** (speech → text) | `whisper_finetune.py` | a finetuned Whisper checkpoint |
| **TTS** (text → speech) | `tts_finetune.py` (orchestrates `tts/*`) | a finetuned Qwen3 that emits NeuCodec speech tokens |

**"Standalone" matters:** these scripts have **no `gateway` imports**. Everything
they need arrives in one JSON config file. They install their own deps into an
isolated `uv` venv, resolve the dataset (S3 or HuggingFace), train, and upload the
result to S3 (optionally pushing to the HF Hub). They talk back to the gateway
**only** through stdout markers it parses:

```
@@METRIC {json}     # per epoch (whisper): {epoch, wer, cer, eval_loss, train_loss}
@@ARTIFACT {json}   # after upload: {s3_uri, hf_repo?}
@@DONE {json}       # final summary
@@ERROR {json}      # fatal: {message}
[AUTOTRAIN_PROGRESS] step=… processed=N total=M percent=P   # progress bars
```
Under `torchrun` (multi-GPU DDP) only **rank 0** prints these (else the gateway
parses every line `WORLD_SIZE` times) — see `_IS_MAIN` / `log()` / `emit()`.

---

## 0. Concepts you need first

If these are new, read this section slowly — the rest of the doc assumes them.

- **ASR vs TTS.** ASR = automatic speech recognition (audio in, text out). TTS =
  text-to-speech (text in, audio out). They are inverses, and we train them with
  two *very different* architectures below.

- **Encoder–decoder (seq2seq) vs decoder-only (causal LM).**
  - *Encoder–decoder* (Whisper): one stack (encoder) reads the **whole input** and
    builds a representation; a second stack (decoder) generates the output one token
    at a time while **cross-attending** to the encoder's representation.
  - *Decoder-only / causal LM* (Qwen3, GPT-style): one stack that just predicts the
    **next token** given everything before it. We turn TTS into "next-token
    prediction over audio tokens" so we can reuse an ordinary LLM.

- **Mel spectrogram.** Raw audio is ~16,000 numbers per second — too long and too
  low-level to feed a transformer directly. A *mel spectrogram* is a compact
  image-like representation: short overlapping windows of audio → frequency content,
  with the frequency axis warped to the **mel scale** (mimics human hearing). Whisper
  consumes the **log-mel** spectrogram, not the raw waveform.

- **Neural audio codec / "audio tokens".** A codec like **NeuCodec** is a learned
  model that compresses a waveform into a short sequence of **discrete integers**
  (a fixed vocabulary, like a dictionary of sound fragments) and can decode those
  integers back into a waveform. Once audio is a sequence of integers, an LLM can
  model it exactly like text. This is what makes "LLM-based TTS" possible.

- **Teacher forcing.** When training a generator, you feed the **ground-truth**
  previous tokens (not the model's own guesses) and ask it to predict the next one.
  Fast and stable; the whole target sequence is supervised in one forward pass.

- **Multipacking.** Concatenating many short training examples into one fixed-length
  block so almost no compute is wasted on padding — explained in full in §2.3,
  because it's the trickiest idea here.

---

## 1. Whisper ASR finetuning — `whisper_finetune.py`

Whisper is an **encoder–decoder transformer**. We finetune it with HuggingFace's
`Seq2SeqTrainer` (`WhisperForConditionalGeneration` + `WhisperProcessor`).

### 1.1 Data flow (what happens to one (audio, text) pair)

```
   raw audio file (.wav/.mp3, any sample rate)
        │  librosa.load(..., sr=16000)         ← decode + resample to 16 kHz mono
        ▼
   waveform  (1-D float array @ 16 kHz)
        │  processor.feature_extractor(...)    ← short-window FFT → log-mel
        ▼
   input_features  = log-Mel spectrogram   (80 or 128 mel bins × time frames,
        │                                    padded/trimmed to Whisper's 30 s window)
        │
        │                                      transcript text
        │                                          │ processor.tokenizer(text)
        ▼                                          ▼
   ┌──────────────┐                          labels  (text token ids)
   │   ENCODER    │  conv stem → transformer
   │ (reads mel)  │  → audio hidden states
   └──────┬───────┘
          │  encoder outputs (one vector per audio frame)
          ▼
   ┌──────────────────────────────┐
   │           DECODER            │  text-token transformer, generated left→right
   │  • causal self-attention     │  (a token only sees earlier text tokens)
   │  • CROSS-ATTENTION ──────────┼──► attends to the ENCODER outputs  ← the link!
   └──────────────┬───────────────┘     (this is how text is conditioned on audio)
                  ▼
        predicted next text token  →  (loss vs `labels`, or sampled at inference)
```

The **cross-attention in every decoder layer is the bridge**: at each step the
decoder looks back at the full encoded audio to decide the next character/word
piece. Self-attention keeps generation causal (no peeking ahead at future text).

### 1.2 Training objective

Teacher forcing + cross-entropy. The decoder is fed the ground-truth transcript
shifted right (it starts from special prompt tokens — language + `transcribe`/
`translate` + optional timestamps — set via `WhisperProcessor(..., language=, task=)`
and `model.generation_config`). For each position it predicts the next token; loss
is cross-entropy against `labels`. Padding positions are set to **`-100`** so they're
ignored by the loss (see the collator: `masked_fill(attention_mask.ne(1), -100)`, and
it strips a leading BOS that the decoder re-adds).

### 1.3 What the script actually does (read in this order)

- `parse_precision()` — `"<load>-<amp>"`, default **`fp32-bf16`** (load weights in
  fp32, train with bf16 mixed precision). Don't default to `bf16-bf16`.
- `_ensure_venv()` — isolated venv at **`/share/autotrain-whisper`** (transformers,
  accelerate, `soundfile`+`librosa` for audio decode, `peft` for LoRA, boto3).
- `load_pairs()` / `_LazyAsrDataset` — resolve the dataset (S3 metadata rows or a HF
  dataset), and **lazily** per item: decode+resample to 16 kHz → `feature_extractor`
  → `input_features`; `tokenizer(text)` → `labels`. Lazy = we don't hold every mel in
  RAM.
- `_augment_audio()` / `_AUG_FUNCS` — optional robustness augmentations (telephone
  band-pass, additive noise, packet-loss dropout, gain, pitch, speed, reverb).
- the data **collator** — pads a batch of variable-length `input_features` and
  `labels` (labels padded with `-100`).
- `compute_metrics()` — decodes generated ids and computes **WER** and **CER**
  against references (this is why `predict_with_generate=True`).
- `Seq2SeqTrainingArguments` + `Seq2SeqTrainer` — the training loop; evaluates every
  epoch, **early-stops on patience**, keeps the best checkpoint, uploads to S3.

Multi-GPU is `torchrun` (DDP): each GPU is a separate process/rank running the same
script; only rank 0 logs and uploads.

### 1.4 Evaluation — WER / CER (built into the training loop)

ASR eval is **inline**: there's no separate eval script. Every epoch (or every
`eval_steps`) the `Seq2SeqTrainer` runs the eval set with **`predict_with_generate=True`**
— it *actually generates* transcripts (real autoregressive decoding), not just the
teacher-forced loss — and `compute_metrics()` scores them.

Two metrics (both lower = better), computed via HuggingFace `evaluate` (jiwer under
the hood) and reported ×100:
- **WER (Word Error Rate)** — edit distance at the **word** level.
- **CER (Character Error Rate)** — edit distance at the **character** level (more
  forgiving; the headline number for non-spaced languages and small models).

**Text normalization matters and is on by default.** Before scoring, both prediction
and reference go through Whisper's normalizer (`tokenizer.normalize` for English,
`tokenizer.basic_normalize` otherwise) — lowercase, strip punctuation, spell out
numbers, etc. Without it, WER/CER are inflated by casing/punctuation and aren't
comparable to any published Whisper number. Opt out with `normalize_text=false` to
score raw text. (Empty references are dropped — jiwer errors on a blank reference.)

How the metric is consumed:
- emitted per epoch as `@@METRIC {epoch, wer, cer, eval_loss, train_loss}` (the
  AutoTrain UI plots these) and reported to W&B/MLflow if configured;
- drives **early stopping** (`patience`) and `load_best_model_at_end` — the checkpoint
  with the best eval metric is what gets uploaded, not the last one.

```
eval audio ─► model.generate() ─► predicted text ┐
                                                  ├─► normalize both ─► WER, CER (×100)
ground-truth transcript ──────────────────────────┘                    → best-checkpoint + early stop
```

---

## 2. TTS finetuning — `tts_finetune.py` (orchestrator) + `tts/`

Key idea: **we do not train an audio model.** We turn speech into **discrete tokens**
with NeuCodec, then finetune a normal **causal LLM (Qwen3)** to predict those tokens
given the speaker + text. At inference the LLM emits speech tokens and NeuCodec
decodes them to a waveform. So "TTS training" here = "language-model training on a
vocabulary that includes speech tokens."

`tts_finetune.py` is just the conductor — it runs four stages on the GPU box:

```
 (0) resolve dataset  →  audio files + meta.jsonl {audio, transcription, speaker}
 (1) convert_neucodec.py   audio  ───────────────►  NeuCodec speech tokens (ints)
 (2) pack_stage1.py        tokens+text  ─────────►  MULTIPACKED fixed-length blocks
 (3) qwen3_tts_flash.py    finetune Qwen3 (causal LM) over the packed blocks
 (4) upload checkpoint to S3 (+ optional HF push);  eval via tts/tts_eval.py
```
Venv: **`/share/autotrain-tts`** (torch/torchaudio, `neucodec`, flash-attn, the
`chinidataset` Parquet streaming lib vendored under `tts/chinidataset`).

### 2.1 Stage 1 — audio → speech tokens (`tts/convert_neucodec.py`)

```
waveform @ 16 kHz ──► NeuCodec.encode_code(wav) ──► [3812, 91, 1190, 7, …]  (ints)
                                                     saved as <key>_neucodec/….json
```
- `model = NeuCodec.from_pretrained("neuphonic/neucodec")`; `codes = model.encode_code(wav.unsqueeze(1))`;
  `tokens = codes[0, 0].tolist()` → one integer stream per utterance, written to JSON.
- **⚠️ Sample rate — NeuCodec is asymmetric. Its *encoder* expects 16 kHz; its
  *decoder* reconstructs at 24 kHz** (`model.sample_rate == 24000`). That's why
  `convert_neucodec.py` hardcodes `_SR = 16000` on the encode side — and it *matters*:
  in `neucodec/model.py`, `encode_code` only auto-resamples when you hand it a **file
  path**; when you hand it a **tensor** (which this script does, `wav.unsqueeze(1)`),
  it does **no resampling** and the internal feature extractor is hardcoded to
  `sampling_rate=16000`. So you must feed 16 kHz — load at 24 k and pass the tensor and
  you'd get silently-wrong tokens. The "24 kHz" you've seen is the **decode/output**
  rate, used by `tts_infer.py` / `tts_eval.py` when they write the synthesized wav
  (`getattr(neucodec, "sample_rate", 24000)`).
- It's a **streaming** producer/consumer pipeline: a pool of downloader threads fetch
  + decode audio (from S3/HTTP, **in memory**) into a bounded queue while the **GPU**
  drains it and encodes. Download and encode overlap; disk use stays ~constant.
- **Resume-safe:** clips whose token JSON already exists are skipped.
- The token-file path (`new_path`) **must** match `pack_stage1.new_path` so stage 2
  can find each utterance's tokens.

Think of the output integers as "words in a sound language." NeuCodec defines that
vocabulary; the LLM will learn to speak it.

### 2.2 Stage 2 — build training examples (`tts/pack_stage1.py`)

Each utterance becomes one text prompt that interleaves the **conditioning** (who +
what to say) and the **target** (the speech tokens):

```
<|im_start|>{speaker}: {text}<|speech_start|><|s_3812|><|s_91|><|s_1190|>…<|im_end|>
└──────────── conditioning prefix ───────────┘└──────────── speech tokens ─────────┘
```
The TTS tokenizer (`Scicom-intl/Multilingual-Expressive-TTS-1.7B`) already has the
`<|s_N|>` speech tokens and the `<|im_start|>` / `<|speech_start|>` / `<|im_end|>`
control tokens **in its vocabulary**, so the whole line tokenizes into ordinary
`input_ids`. (NeuCodec integer `3812` → the literal string token `<|s_3812|>`.)

Then these examples are **multipacked** (next section) into fixed-length blocks and
written as Parquet/MDS shards (via the vendored `chinidataset`) with columns:
`input_ids`, `position_ids`, `attention_mask` — note `attention_mask` here is **not**
a 0/1 mask (see §2.3).

### 2.3 Multipacking — *read this carefully*

**The problem.** Utterances vary a lot in length (a 1-second clip vs a 15-second
clip). If you batch them naively, every sequence is padded to the longest one in the
batch, and the GPU burns compute on padding tokens that contribute nothing. For
speech-token sequences that waste can be enormous.

**The fix: pack, don't pad.** Concatenate many whole utterances end-to-end into one
block of a **fixed length** (`sequence_length`, default **4096** tokens). When the
next utterance wouldn't fit, you close the current block and start a new one
(`if count + length > block_size: flush; start new block`). Utterances longer than
the block are dropped. Result: blocks are ~entirely real tokens — **almost no
padding** — so throughput goes way up.

```
block (4096 tokens) = [ utterance A | utterance B | utterance C | … ]   no padding
```

**The danger, and how we avoid it.** If you just concatenate, the transformer's
attention would let utterance C "see" utterances A and B, and the position counter
would keep climbing across them — both are wrong (they're unrelated samples). Two
mechanisms keep packed utterances fully independent:

1. **`position_ids` reset per utterance.** In `loop()` each utterance contributes
   `range(length)` — so positions restart at 0 at every utterance boundary
   (`[0,1,…,nA-1, 0,1,…,nB-1, 0,1,…]`). Each utterance thinks it's at the start of a
   sequence.
2. **Block-diagonal ("varlen") attention.** `pack_stage1.collator` stores, in the
   `attention_mask` column, the **list of per-utterance lengths** (e.g. `[nA, nB, nC]`),
   *not* a 0/1 mask. At train time those lengths become **`cu_seq_lens`** (cumulative
   sequence lengths) handed to **flash-attention's varlen kernel**, which restricts
   attention so each utterance attends **only within itself** — never across a pack
   boundary.

> ⚠️ This is why `qwen3_tts_flash.py` **requires a flash-attention backend** (FA3 on
> the H20 / sm_90 via the `kernels` hub, or FA2). `sdpa`/`eager` would silently let
> utterances attend across boundaries and corrupt training — so the script tries FA3,
> falls back to FA2, and **fails loudly** otherwise rather than degrade.

**One-line summary:** multipacking = pack many variable-length samples into one
full-length block to kill padding waste, then use per-sample `position_ids` + varlen
(block-diagonal) attention so the packed samples can't contaminate each other.

### 2.4 Stage 3 — finetune the LLM (`tts/qwen3_tts_flash.py`)

- `AutoModelForCausalLM.from_pretrained(base_model)` — default `Qwen/Qwen3-1.7B-Base`
  — loaded with `attn_implementation` = FA3 then FA2 (see warning above).
- Dataset: streams the packed Parquet blocks (`chinidataset.StreamingDataset`),
  passing `input_ids`, `position_ids`, and the segment-length `attention_mask` through
  to the model so flash-attn can build `cu_seq_lens`.
- Objective: **standard causal-LM next-token cross-entropy** — labels are the inputs
  shifted by one (`labels = labels[..., 1:]` in `forward`). The model learns: given
  `<|im_start|>{speaker}: {text}<|speech_start|>`, continue with the right `<|s_N|>`
  tokens, then `<|im_end|>`. Metrics are loss-only (W&B/MLflow via HF Trainer
  `report_to`).

### 2.5 Inference (`tts/tts_infer.py`)

Build the prefix `<|im_start|>{speaker}: {text}<|speech_start|>` and let the LLM
generate tokens until `<|im_end|>`. Strip the `<|s_N|>` tokens back to integers and
call **`NeuCodec` decode** → waveform (written at the codec's 24 kHz output rate;
see §2.1). Generate via `AutoModelForCausalLM` — **not** a custom `Model` wrapper
(that bug produced silence/garbage and was one of the validated fixes).

### 2.6 Evaluation — CER / MOS / similarity (`tts/tts_eval.py`)

Unlike Whisper, TTS eval is a **separate script run after training** (you can't score
"how good is the speech" with a training loss). It re-synthesizes the **test split**
and scores the generated audio with any combination of three metrics (pick via config;
heavy models are imported lazily so you only pay for what you select):

| Metric | What it answers | How (`tts_eval.py`) |
|--------|-----------------|---------------------|
| **CER** | *Did it say the right words?* (intelligibility) | ASR the generated audio with **Whisper** (`AutoModelForSpeechSeq2Seq`), then `jiwer.cer` vs. the reference transcript (capped at 1.0). Lower = better. |
| **MOS** | *Does it sound natural?* (quality) | Predicted naturalness via **UTMOSv2** (`utmosv2.create_model(pretrained=True).predict(..., num_repetitions=5)`). No reference needed; higher = better (~1–5). |
| **similarity** | *Does it sound like the target speaker?* (voice fidelity) | **TitaNet** speaker embeddings of generated vs. **reference** audio → cosine similarity. Higher = better (0–1). |

The harness, per utterance in the packed test set:
1. take the prompt up to `<|speech_start|>`, **generate** the speech tokens, NeuCodec-decode → **generated** wav;
2. decode the **reference** speech tokens already in the record → **reference** wav;
3. the **reference transcript** is the text;
4. score: generated-wav-vs-text (CER), generated-wav (MOS), generated-vs-reference-wav (similarity).

Emits one `@@METRIC {json}` per method + `@@DONE`. Notes / gotchas baked into the code
(learned the hard way — see the validated fixes):
- the eval tokenizer **must match the one used in `pack_stage1`** (same `<|s_N|>` vocab) or the speech tokens won't line up;
- decode/transcribe with **librosa**, not system ffmpeg (the box has no ffmpeg);
- the Whisper ASR step reads the generated wav at **16 kHz** (its encoder rate), while NeuCodec writes it at **24 kHz** — resampling is handled in the loader;
- scorer APIs mirror the Scicom `Multilingual-TTS` `calculate_{cer,mos,similarity}.py` verbatim, so numbers are comparable to that reference.

---

## 3. How to run

**Through AutoTrain (normal path):** create a run in the web UI (task type ASR or
TTS); the gateway writes the JSON config, ships these scripts + the dataset refs to
the GPU box, and runs them over SSH, streaming logs back and parsing the `@@…` /
`[AUTOTRAIN_PROGRESS]` markers.

**Directly on a GPU box (for debugging):**
```bash
# ASR
python whisper_finetune.py --config /path/to/whisper_cfg.json
# TTS (runs all four stages)
python tts_finetune.py --config /path/to/tts_cfg.json
# or run a TTS stage in isolation:
python tts/convert_neucodec.py --file audio_sources.json
python tts/pack_stage1.py --dataset meta.jsonl --output_dir /tmp/packed --sequence_length 4096
torchrun --nproc_per_node=<N> tts/qwen3_tts_flash.py  # args from the orchestrator
```

**Config keys you'll see** (both pipelines, in the `--config` JSON): `base_model`,
`tokenizer`, dataset descriptor (S3 creds/prefix or HF id), `max_epochs`,
`learning_rate`, `precision` (`fp32-bf16` etc.), `venv_path`, `cleanup_checkpoints`,
optional HF push target. Whisper adds `language` + `task` (`transcription`/
`translation`) and augmentation toggles.

---

## 4. Glossary

| Term | Meaning |
|------|---------|
| **WER / CER** | Word / Character Error Rate — edit distance between predicted and reference text (lower is better). Whisper's eval metrics; CER also scores TTS intelligibility (ASR the synthesized audio, compare to the input text). |
| **MOS** | Mean Opinion Score — speech naturalness/quality, ~1–5, higher better. We predict it (no human raters) with **UTMOSv2**. |
| **UTMOSv2** | A no-reference model that predicts MOS from a waveform — the TTS naturalness metric. |
| **TitaNet** | A speaker-embedding model; cosine similarity between the generated and reference embeddings = TTS speaker/voice similarity (higher better). |
| **Text normalization** | Lowercasing / punctuation-stripping / number spell-out applied to both sides before WER/CER, so the score reflects content, not formatting. On by default for Whisper eval. |
| **Log-mel spectrogram** | Audio as a frequency×time image on the perceptual mel scale, log-scaled. Whisper's encoder input (`input_features`). |
| **NeuCodec** | Neural audio codec (`neuphonic/neucodec`) that encodes a waveform to discrete integer tokens and decodes them back. Turns audio into something an LLM can model. **Asymmetric SR: encoder in = 16 kHz, decoder out = 24 kHz** (`model.sample_rate == 24000`). |
| **Speech token `<|s_N|>`** | A NeuCodec integer `N` rendered as a vocabulary token the LLM predicts. |
| **Cross-attention** | Decoder attending to the encoder's outputs — how Whisper conditions text on audio. |
| **Teacher forcing** | Training a generator on ground-truth previous tokens. |
| **Multipacking** | Concatenating many samples into one fixed-length block to eliminate padding waste; correctness via reset `position_ids` + varlen (block-diagonal) flash-attention. |
| **`cu_seq_lens`** | Cumulative per-sample lengths flash-attn varlen uses to keep packed samples from attending across each other. |
| **`-100`** | The ignore index for cross-entropy — masks padding/non-target positions out of the loss. |
| **DDP / `torchrun`** | Distributed Data Parallel: one process per GPU; only rank 0 emits the gateway markers. |
| **`fp32-bf16`** | Default precision: fp32 weights, bf16 mixed-precision compute. |
| **`@@METRIC` / `[AUTOTRAIN_PROGRESS]`** | stdout markers the gateway parses for metrics and progress bars. |
