# omnivoice — finetune + serve k2-fsa/OmniVoice on Scicom-intl/TM-Voice

Self-contained job (sibling of `gemma4/`, `minimax-m2/`, `mistral-small/`), but **unlike** them it
**uses OmniVoice's own stock training pipeline** (`omnivoice.cli.train`) — a **full** finetune of a
small (Qwen3-0.6B-based) diffusion-LM zero-shot TTS model, **not LoRA**, **no custom forward**. So the
autotrain `compare_logits.py` rule does **not** apply (nothing replaces a stock transformers forward).

Covers the whole lifecycle: **data prep → finetune → eval (CER/MOS + speaker-similarity A/B) → demo
samples → optimized OpenAI-compatible serving + H100 benchmark**. Pushed to
**[`Scicom-intl/omnivoice-tmvoice`](https://huggingface.co/Scicom-intl/omnivoice-tmvoice)** (the repo
currently holds the **20-epoch** checkpoint; self-contained with bundled Higgs `audio_tokenizer/`).

## Pipeline
1. **Data** (`prepare_data.py`) — `Scicom-intl/TM-Voice` (gated): `english.zip` + `Mandarin_2026-06-12.zip`
   (must **unzip**) + a parquet `[speaker, filename_audio, text]` (3635 rows; TM_English 1253, TM_Mandarin
   2382). **Each Mandarin clip ships TWICE** — once pinyin, once Hanzi → dedup per audio file keeping the
   **Hanzi** text (max-CJK) → **2444 unique files**. Per-speaker split holds out **50 test/speaker** →
   **2344 train (en 1203 / zh 1141), 100 test**. Emits `train.jsonl`, `dev.jsonl` (=test, for eval-loss),
   `eval_test.jsonl` (ref = a random same-speaker train clip for zero-shot voice-cloning eval). `language_id`=en/zh.
2. **Tokenize** (`tokenize_audio.py`) — Higgs codec (`eustlb/higgs-audio-v2-tokenizer`) encodes each clip →
   WebDataset shards + `data.lst` (same layout `omnivoice.data.dataset` reads). **Sequential, single-process
   replacement** for the stock `extract_audio_tokens` (whose 27-process default deadlocks — see Gotchas).
3. **Train** — `accelerate launch -m omnivoice.cli.train`. The trainer is **step-based** (`while
   global_step < steps`; epoch++ on `StopIteration`), so **N epochs = N × steps_per_epoch**. `count_steps.py`
   builds the *real* train dataloader (1 proc, no DDP) and counts packed batches (got **82/epoch**). `run.sh`
   patches `train_config.json` (`steps`, `save_steps`=`eval_steps`=82 → checkpoint+eval per epoch). `EPOCHS=` env.
4. **Clean** (`make_clean_checkpoint.py`) — strip Accelerate optimizer state + bundle the Higgs
   `audio_tokenizer/` → self-contained inference dir.
5. **Eval** (`tts_eval.py`) — `omnivoice-infer-batch` synthesizes the 100 test files → **CER** (Whisper-large-v3
   + `jiwer.cer`, per-language) + **MOS** (UTMOSv2), **mirroring `gateway/.../training/tts/tts_eval.py`**.
6. **A/B** (`ab.sh` + `ab_eval.py`) — base vs finetune on the same items: CER + **ECAPA speaker-similarity**
   (gen↔reference prompt, gen↔ground-truth TM clip). This is the metric that shows the finetune working.
7. **Samples** (`sample.sh` + `make_samples.py`) — per speaker, 1 reference + 5 synthesized held-out sentences.
8. **Serve + bench** (`serve.py`, `bench_serve.py`, `serve_bench.sh`, `serve_client.py`) — see Serving below.

`run.sh` is staged (`STAGE`/`STOP_STAGE`, 0..8) for incremental runs.

## Results (verified on RunPod 1× H100, 2026-06-22)
**Accuracy** (100 held-out clips; CER = Whisper-large-v3 char error ↓, MOS = UTMOSv2 naturalness ↑):

| run | CER (all / en / zh) | MOS (all / en / zh) | note |
|---|---|---|---|
| **2 epochs** | 0.246 / 0.194 / 0.299 | 3.00 / 3.11 / 2.89 | final eval-loss ~4.5 |
| **20 epochs** (current HF) | 0.247 / 0.200 / 0.294 | 3.00 / 3.09 / 2.91 | eval-loss ↓ 4.5→3.8 |

**CER/MOS are flat across 2↔20 epochs** even though eval-loss dropped — those metrics don't measure speaker
timbre. The **A/B speaker-similarity** is where the finetune shows up:

| metric (overall) | base k2-fsa/OmniVoice | finetune | Δ |
|---|---|---|---|
| CER ↓ | 0.246 | 0.243 | −0.003 |
| spk-sim → reference prompt ↑ | 0.803 | 0.797 | ~flat (base already clones prompt) |
| **spk-sim → ground-truth TM clip ↑** | 0.718 | **0.768** | **+0.051** (en .639→.709, zh .797→.828) |

→ **The finetune genuinely adapts the voice toward the real TM speakers** (sim-to-GT +0.05, en ~+11% rel).
Judge TTS finetunes by **speaker-sim-to-target**, not CER/MOS (both saturate on a strong base model).
Takeaway: 2 epochs already captures the gain; more epochs aren't worth the GPU time for this data.

## Serving — OpenAI `/v1/audio/speech`, high concurrency (`serve.py`)
Resident model; named **voices** = reference clips encoded once into `VoiceClonePrompt` and **cached**;
async **dynamic-batching** loop with **length bucketing** (group similar-duration requests so the rectangular
diffusion forward doesn't pad short clips to the longest); GPU work on a 1-thread executor; audio
container-encode (mp3/wav/opus/pcm) on a CPU pool. Endpoints: `/v1/audio/speech`, `/v1/models`, `/health`.
Env: `OV_MODEL OV_VOICES OV_MAX_BATCH OV_NUM_STEP OV_BUCKET_RATIO OV_GUIDANCE OV_COMPILE OV_DTYPE PORT`.

**Benchmark** (1× H100, finetuned model, mixed real TM texts, `bench_serve.py`):

| concurrency | no batching | dynamic batching | + length bucketing |
|---|---|---|---|
| 1 | 2.0 rps | 1.9 | 1.9 |
| 32 | 2.0 (p50 15.6s) | 2.9 | **3.5 rps, RTF 33×** |
| 64 | 2.0 (p50 31s) | 2.9 | 3.3 |

Uniform-length burst hits **5.6 rps (2.8×)** = the single-GPU ceiling — the H100 runs **100% util at small
batch**, so it's launch/overhead+memory-bound (0.6B model × 32 steps × CFG = many tiny kernels), not FLOPS-bound.
**Served-output CER = 0.243 == offline 0.247** → batching/bucketing **preserve quality** (verified by feeding
the server's audio back through Whisper).

**Optimization campaign** (every config gated on served-CER + MOS; `opt_bench.sh` + `opt_round2.sh` + `score_dirs.py`):

| num_step | CER ↓ | MOS ↑ | rps @c32 | RTF | verdict |
|---|---|---|---|---|---|
| 32 (orig) | 0.2445 | 2.997 | 3.5 | 33× | baseline |
| 24 | 0.2439 | 2.939 | 5.4 | 52× | ok |
| **16 (default)** | 0.2454 | 2.953 | **6.84** | **65×** | **~2× speed, no accuracy drop** |
| 12 | 0.2441 | 2.947 | 6.73 | 65× | no extra speed |
| 8 | 0.2509 ↑ | 2.888 ↓ | 7.76 | 75× | rejected (CER+MOS drop) |

→ **`num_step=16` is the optimized default** (baked into `serve.py`): **2× throughput at flat CER + ~flat MOS**.
Below 16 the diffusion loop stops being the bottleneck (decode/encode/batching floor) so speed plateaus; at 8 quality
drops. **`torch.compile(reduce-overhead)`** (CUDA graphs) adds only ~9% @c32 and *hurts* single-request latency
(0.55→0.97s) because varying seq-len causes per-shape graph **recapture** — so it's opt-in (`OV_COMPILE=1`) and needs
**fixed-shape buckets** to pay off cleanly. Remaining levers (not done): fixed-length-grid CUDA graphs; **varlen/packed
attention** (zero padding — the *training* forward already has the `document_ids`+`create_block_mask`/flex_attention
machinery); uniform-chunk + step-level admit/retire (true continuous batching); prefix-KV reuse across steps; fp8 (gate
on CER); **data-parallel replicas** (one per GPU, linear) for scale-out beyond the single-GPU ~2× ceiling.

## RunPod recipe (1× H100, US)
Keys from `../.env` (`RUNPOD_API_KEY`, `HF_TOKEN`). Hard rules: `--country-code US`, work under `/root/...`
(never `/workspace`), generous `--container-disk-in-gb`.
- Image **`runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2404`** (OmniVoice pins **torch 2.8 + cu128** — NOT the
  cu13/torch-2.12 + FA3 image the gemma4/MoE siblings use). pip needs **`--break-system-packages`** (PEP-668).
- `export HF_HUB_DISABLE_XET=1` (see Gotchas). Run long stages under **tmux/nohup** ("patient").
- **Always `runpodctl pod delete <id>`** when done — real billing (~$3.3/hr H100). This job names its pods
  `omnivoice-*` (ft / ft20 / ab / samples / serve / opt); delete each when its stage finishes.
- **Pod hygiene**: `runpodctl pod list` shows **all** account pods, including unrelated ones
  (`sgpu-train-*`, `neucodec-*`, `aura-*`, …). Only delete pods you created (the `omnivoice-*` ones) — don't
  touch others' running jobs. After a full session, `runpodctl pod list | grep omnivoice` should be empty.
- SSH endpoint: `runpodctl pod create ... --ports "22/tcp"`, then GraphQL `pod(...).runtime.ports` for the
  public ip:port (or `runpodctl get pod -a`); key is `~/.ssh/id_rsa`.

## Gotchas (load-bearing)
- ⭐ **HF Xet STALLS on RunPod** — the Higgs `model.safetensors` (603 MB) downloads to full size but never
  finalizes (leaves `*.incomplete` blobs); `from_pretrained` then hangs **forever**. It *masquerades as a
  tokenizer multiprocessing deadlock* (workers stuck on the download, GPU idle). Fix: **`HF_HUB_DISABLE_XET=1`**
  for every HF pull (`HF_HUB_ENABLE_HF_TRANSFER` is ignored now; Xet is default).
- The stock `omnivoice.scripts.extract_audio_tokens` defaults (`nj_per_gpu=3`, `loader_workers=24`) spawn ~27
  torch processes that hammer the (Xet-stalling) download at once → use **`tokenize_audio.py`** instead.
- **`keep_last_n_checkpoints=-1`** keeps ALL checkpoints (~6.9 GB each: 2.3 model + 4.6 optimizer). 20 epochs
  blow the 120 GB disk → run a background "keep latest 2" GC (`ckpt_gc.sh` pattern) for many-epoch runs.
- `count_steps.py` must use the **same `attn_implementation` + `num_workers`** as training or the per-epoch
  count drifts. flex_attention works on H100; fall back to `ATTN=sdpa` (length-grouped padding) if it errors.
- Training checkpoints are HF-format but **don't bundle the Higgs codec** → `make_clean_checkpoint.py` copies
  `audio_tokenizer/` in (else `from_pretrained` re-downloads it).
- `infer_batch --model` / `OmniVoice.from_pretrained` accept a **local dir** OR an HF repo id. Output wavs are
  `{id}.wav` @ 24 kHz. UTMOSv2 = `git+.../faster-UTMOSv2` (imports as `utmosv2`), matching the gateway.
- Pushing to the `Scicom-intl` org needs the `HF_TOKEN` to have **write** to it (huseinzolkepliscicom is admin).
- Driving the pod over SSH: **`pkill -f <pattern>` self-matches your own ssh shell** when the pattern string is
  in its command line (kills the session, exit 255). Kill from a **script file** (clean cmdline) instead.

## File index
`prepare_data.py` · `tokenize_audio.py` · `count_steps.py` · `train_config.json` · `data_config.json` ·
`make_clean_checkpoint.py` · `tts_eval.py` · `run.sh` (stages 0–8) · `ab_eval.py` + `ab.sh` (base-vs-FT A/B) ·
`make_samples.py` + `sample.sh` (demo `samples/`) · `serve.py` + `bench_serve.py` + `serve_client.py` +
`serve_bench.sh` (OpenAI server + H100 benchmark) · `opt_bench.sh` + `opt_round2.sh` + `score_dirs.py`
(speed-vs-CER/MOS optimization sweep). Result JSONs (`eval_results.json`, `ab_results.json`,
`bench_*.json`, `served_cer.json`, `opt/*.json`) are produced on the pod; headline numbers are above. `samples/` (committed
demo audio) holds 5 synthesized clips + 1 reference per speaker.
