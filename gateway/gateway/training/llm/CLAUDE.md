# Autotrain — LLM finetune subsystem (gemma-4 / MiniMax-M2 / Mistral-Small-4)

This directory holds the **vendored standalone trainers** the gateway's Autotrain feature ships to
a GPU box to LoRA-finetune large LLMs. It's the LLM analogue of the TTS path
(`../tts_finetune.py` + `../tts/`). The trainer to run is **auto-detected from the base model**.

```
../llm_finetune.py     the orchestrator shipped + run on the box (task_type=llm). Builds the venv,
                       downloads the packed dataset, runs the per-arch correctness pre-flight, then
                       `torchrun <trainer>`, parses loss → @@STEP, uploads the LoRA checkpoint.
../../llm_pack.py      IN-PROCESS (gateway) chat → multipack: messages column → kind=llm_packed
                       ChiniDataset. Arch-aware (gemma / minimax / mistral templates).
llm/gemma4.py          gemma-4-31B dense bf16 trainer (custom dynamic_attention OR FA4 fork)
llm/attention.py       gemma's dynamic_attention (head_dim-512 SDPA + 256 FA3) — the FA3 fallback
llm/gemma4_fa4_attention.py  gemma's FA4 backend (flash-attention-512 cute kernel; THE DEFAULT)
llm/test_attention.py / compare_logits_fa4.py   gemma correctness gates (FA3 unit test / FA4 model compare)
llm/minimax/minimax_m2.py    MiniMax-M2 230B FP8 MoE trainer (QLoRA-style dequant LoRA)
llm/mistral/mistral_small.py Mistral-Small-4 119B FP8 MoE (MLA) trainer
llm/minimax/lora.py · mistral/lora.py · */dequant_triton.py · */test_lora.py   per-arch LoRA + FP8 dequant
llm/chinidataset/      vendored ChiniDataset (Parquet StreamingDataset) — SHARED by all trainers
llm/_trainer_common.py SHARED scaffolding: add_common_args() (the CLI flags every trainer takes)
                       + fsdp_kwargs()/shard_layers()/checkpoint_layers() (the identical FSDP2 setup).
                       Importable by bare name (llm_finetune sets PYTHONPATH=<llm dir>). See below.
```

### Shared trainer scaffolding (`_trainer_common.py`) — don't re-copy CLI/FSDP per trainer

The four trainers used to each inline the **same** argparse block and the **same** FSDP2
shard/checkpoint code, so every new flag (batch_size, grad_accum, cpu_offload…) meant editing four
files. That's now in `_trainer_common.py`:
- **`add_common_args(parser, *, lr_default, wandb_project)`** — the flags EVERY trainer takes
  (`--batch_size --grad_accum --cpu_offload --max_epochs --max_steps --checkpointing_step
  --limit_samples --lr --wandb --wandb_project`). Arch-specific flags (LoRA dims, model id, data/out
  dirs) stay in each trainer's parser. **A new common flag is a one-line edit here.**
- **`fsdp_kwargs(mesh, *, param_dtype, cpu_offload)`** — the `fully_shard` kwargs (mp_policy +
  optional CPUOffloadPolicy + mesh). THE one place the offload/param_dtype policy lives.
- **`shard_layers(model, decoder_classes, kw, *, reshard_after_forward=None)`** +
  **`checkpoint_layers(model, decoder_classes)`** — the shard-loop + non-reentrant activation
  checkpointing (identical across archs; qwen passes a `(decoder, vision)` tuple + `reshard_after_forward`).

What stays per-trainer (legitimately arch-specific): **`param_dtype`** (bf16 dense; **None** for FP8
MoE so the fp8 weights aren't cast), the decoder-layer class, and the FP8-MoE steps interleaved with
sharding (`_promote_scalar_params`, `low_cpu_shard_load`). A new model = call the four helpers + its
own FP8 bits; no CLI/FSDP copy-paste.

**`--cpu_offload`** (FSDP2 CPUOffloadPolicy — params/optimizer in host RAM: big VRAM saver, PCIe-bound
so ~7% MFU / slow). **Arch-aware default** (`llm_finetune._cpu_offload_on`): gemma/qwen default **ON**
(dense — hit the VRAM wall at long context), minimax/mistral default **OFF** (fit without it). The
form exposes a toggle (arch-aware default via `llmCpuOffloadDefault`); `training_api` carries
`cpu_offload` (None = per-arch default) → `cfg` → the cmd builders append `--cpu_offload` when on.
On big-VRAM GPUs (143GB H20) turning it OFF is a large speedup — but only if the context/bins fit
without it (a 64k-block run was at 99% VRAM *with* offload → can't drop it there without OOM).

These are **copies** of the standalone `autotrain/{gemma4,minimax-m2,mistral-small}/` jobs (the
authority for the model-specific design — read their CLAUDE.md for the deep details). The gateway
copy adds two edits to each `*.py` it runs: a `MODEL_ID`/`GEMMA_MODEL_ID` env override (so a run can
pick a size) and, for gemma, the `--target_modules` flag. When the user improves a standalone
trainer, RE-SYNC the vendored copy here (preserving those gateway edits).

## The end-to-end pipeline

1. **Dataset.** A chat dataset (`kind=llm`, or a `kind=hf` dataset with a `messages` column mapped)
   on the Datasets page.
2. **Pack for LLM** (Transform tab → in-process, CPU, no GPU). `../../llm_pack.py` renders each
   conversation through the chosen tokenizer's chat template (messages + tools) and bin-packs into a
   ChiniDataset → a new **`kind=llm_packed`** dataset on S3 (`input_ids/labels/position_ids/
   attention_mask`, cu_seqlens varlen). The pack arch (gemma/minimax/mistral) is recorded in
   `split_fields._llm_pack.arch`.
3. **Autotrain → New → LLM.** Pick the `llm_packed` dataset + a base model. `task_type=llm`. The
   create-run rejects a base-model/dataset **arch mismatch** (a gemma-packed dataset trained as
   minimax = garbage ids — the trainer reads the packed ids verbatim, no re-tokenization).
4. **Run.** `training_api.run_training` ships `llm_finetune.py` + this `llm/` dir to the box, runs
   `python llm_finetune.py --deps-only` (builds the arch venv), then launches it from that venv.
   `llm_finetune.run()` downloads the pack → `./packed_data`, pre-fetches the base model, runs the
   pre-flight, then `torchrun <arch trainer>`. The LoRA checkpoint (`lora.pt` + `lora_meta.json`)
   uploads to S3 (+ optional HF push).

## Architecture auto-detect (the core of this subsystem)

`detect_arch(model_id)` (in `llm_finetune.py`, `llm_pack.py`, and `training_api._llm_arch` — keep
all three in sync): `"minimax"`→minimax, `"mistral"`→mistral, else gemma. Everything per-arch lives
in `llm_finetune._ARCH` + `_fa_mode`:

| | gemma-4 | MiniMax-M2 | Mistral-Small-4 |
|---|---|---|---|
| trainer | `gemma4.py` | `minimax/minimax_m2.py` | `mistral/mistral_small.py` |
| size / dtype | 31B dense bf16 | 230B FP8 MoE | 119B FP8 MoE (MLA) |
| attention | **FA4 cute fork** (default; head_dim-512), or FA3+dynamic_attention | stock FA3 (head_dim 128) | stock FA3 (head_dim 128) |
| venv | `/share/autotrain-llm-gemma` | `/share/autotrain-llm-minimax` | `/share/autotrain-llm-mistral` |
| `kernels` pin | `==0.14.1` | `>=0.12,<0.13` | `>=0.12,<0.13` |
| extra deps | `peft` | `accelerate` | `accelerate` |
| pre-flight | `test_attention.py` (FA3) / skipped (FA4) | `test_lora.py` (CPU) | `test_lora.py` (CPU) |
| run env | `GEMMA_ATTN`, FA4 JIT cache | — | `MISTRAL_DEQUANT_TRITON=1` |
| LoRA CLI | `--r --alpha --target_modules` | `--attn_r --moe_r --attn_alpha --moe_alpha [--no_moe_lora]` | + `--no_shared_lora` |

Common to all: **torch 2.12** (the FA3-wheel ABI + `torch._grouped_mm` for fused MoE), transformers
5.5.0, liger-kernel, the FA3 prebuilt wheel (cu126/130/132 from the host driver) — EXCEPT gemma's
default FA4 path which swaps the wheel for the cute fork. **`--batch_size` / `--grad_accum` are now
configurable** (all four trainers): `batch_size` = how many packed bins the collator concatenates into
ONE varlen sequence per microbatch (each bin is already a full sequence, so this raises per-step memory),
`grad_accum` = microbatches accumulated before an optimizer step. **Effective batch = batch_size ×
grad_accum × world_size.** Grad-accum is the memory-safe lever (step + zero_grad only every N
microbatches, partial window flushed at epoch end; `max_steps`/checkpointing now count OPTIMIZER steps).
`llm_finetune.py` passes both from the run config (default 1 each → prior behavior). The UI defaults LLM
`batch_size` to 1 (ASR/TTS stay 8). minimax/mistral always pass `--low_cpu_shard_load`.

**⚠ Token-weighted loss (grad-accum + DDP), mirrors HF Trainer's `average_tokens_across_devices`.** The
naive `loss/grad_accum` is a *mean-of-means* — it over-weights short bins and, under FSDP grad averaging,
mis-weights ranks with fewer tokens. Instead each microbatch back-props the token-**SUM** loss
(`model_mean_loss × n_tok`, where `n_tok = (labels[:,1:] != -100).sum()` == the LigerFusedLinearCE
shifted denominator — numerically identical to a `reduction="sum"` loss, no overflow). A running
`win_tokens` accumulates `n_tok` over the window; at each optimizer step it's `all_reduce(SUM)`'d across
ranks to `N_total`, and the accumulated `.grad` is scaled by **`world_size / N_total`** (the `world_size`
factor cancels FSDP's grad averaging; ÷`N_total` is the true token-mean over the whole effective batch —
all ranks × all accumulation microbatches). Reduces EXACTLY to the old `loss.backward()` for single-GPU
+ `grad_accum=1` (`scale = 1/n_tok`, no cross-rank term). Grads are scaled via
`optimizer.param_groups` (works for gemma's CPU-offloaded grads too — `scale` is `.item()`'d to a py float
so there's no CPU↔CUDA device mismatch). Logged loss stays the per-microbatch mean (human-readable / the
`@@STEP` parser).

## gemma FA4 (the default, faster, long-context path)

`gemma4.py` registers two backends and `GEMMA_ATTN` selects (the gateway sets it explicitly):
`fa4_attention` (the `Scicom-AI-Enterprise-Organization/flash-attention-512` cute kernel — symmetric
head_dim-512 fwd+bwd on Hopper, O(S) memory, lifts the ~32k SDPA ceiling) is the **default**;
`dynamic_attention` (FA3 + SDPA-tiled 512) is the opt-out (`cfg.gemma_fa4=False`).

FA4 install (public fork, no token): `git+…/flash-attention-512.git#subdirectory=flash_attn/cute`
then the **load-bearing pins** `nvidia-cutlass-dsl[cu13]==4.4.2 quack-kernels==0.3.10` (the `>=`
bounds in cute's pyproject pull builds that break). CuTeDSL **JIT-compiles** the kernel on the first
forward — the **first step is slow (~2 min)**; `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` caches it.
The cute pkg is a `flash_attn.cute` namespace package — it imports fine as long as no top-level
`flash_attn/__init__.py` (FA2) and no local `flash_attn/` dir shadow it (the trainer runs from the
workdir, which has neither). ⚠ FA4 fixes the *attention* memory wall but NOT the
activation-checkpoint memory (O(seq×layers), unshardable) → ~64k context ceiling on 80GB GPUs;
144GB H20s clear it. (See `autotrain/gemma4/CLAUDE.md` for the full FA4 + memory notes.)

## Running / debugging (verified end-to-end on the `tm` 8× H20 VM, 2026-06-20)

- **gemma-4-31B FA4** (`ds-512802b5`, 8× H20): ✅ done, exit 0. FA4 cute kernel JIT-compiled +
  trained; loss 7.09 → 4.84; LoRA checkpoint → S3. First step ~114s (JIT), then fast.
- **MiniMax-M2.7** (`ds-30fa1cce`, 8× H20): ✅ done, exit 0, **10 steps**, LoRA checkpoint → S3. Full
  attn+MoE LoRA wraps correctly (248 attn blocks + 62 MoE blocks = **2756.9M trainable**), FSDP2-of-fp8
  + the QLoRA-style dequant works, loss moves, 744-tensor LoRA saved per epoch. (Loss is noisy on a
  4-bin smoke at lr 1e-5 — expected.)
- **Mistral-Small-4** (`ds-d690354e`): wiring complete + the dataset packs; the live 10-step smoke
  was skipped (user) — its code path is the SAME FP8-MoE stack as MiniMax (which passed), just
  `mistral_small.py` instead of `minimax_m2.py` (+ `--no_shared_lora` available, MLA attention).

**Also validated on a freshly-provisioned RunPod pod (2026-06-20/21)** — the gateway spawns the pod,
ships the trainer, runs it detached, uploads to S3, and **tears the pod down** (no orphaned billing,
confirmed `pod … torn down` for every run). TTS (1× H100) ✅, STT/Whisper (1× H100) ✅, gemma-4-31B
(4× H100) ✅ **both attention paths**:
- **FA3** (`gemma_fa4=false`, default `cu1281` image): loss 7.08→1.77, ~1.0–1.2k tokens/s.
- **FA4** (`gemma_fa4=true`, image `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404` → CUDA-13 host):
  loss 7.10→1.83, **~4.0–4.4k tokens/s (~3.5× FA3)**. `cu130` torch wheel + the cute fork +
  `cutlass-dsl[cu13]` all install on the pod; `FA4 cute import OK`, first step ~38s.

**To run FA4 on RunPod you MUST pass a CUDA-13 image** (`…-cu1300-…`) so the pod lands on a ≥580
driver — the FA4 cute kernel JIT-compiles CUDA-13 cubins via `cutlass-dsl[cu13]` and fails on an
older host. `_extract_cuda_version` parses the image tag → `allowedCudaVersions=["13.0"]`, which is
what pins the host. RunPod-specific breaks found: the `--r`/torchrun collision and the cuda-tag
parse bug (both below).

To drive a run without the web UI (admin API key as `Authorization: Bearer sgpu_…`):
`POST /v1/training-runs` with `{task_type:"llm", dataset_id, base_model, provider_id, storage_id,
gpu_count, visible_devices, max_steps, max_epochs, lora_r, lora_alpha_ratio, no_eval:true,
env_vars:{HF_HOME, HF_HUB_DISABLE_XET}}`.

### Real bugs found + fixed wiring these up (2026-06-20)

- **`base_model` MUST match the dataset's pack tokenizer EXACTLY.** `ds-30fa1cce` was packed with
  `MiniMaxAI/MiniMax-M2.7` (note the `.7`) — training it as plain `MiniMaxAI/MiniMax-M2` is both an
  uncached re-download AND a tokenizer mismatch (the trainer reads packed ids verbatim). The
  `_llm_pack.tokenizer` field records the right id; use it.
- **LoRA alpha precedence** (`_lora_dims`): the form's default `lora_alpha` was clobbering
  `lora_alpha_ratio` (scaling 0.125 vs the intended 2.0). The ratio is the form's real control → it wins.
- **`--no_moe_lora` misfire** (`_moe_cmd`): the form's default `q/k/v/o` target list (no MLP) wrongly
  disabled the MoE expert LoRA on every MoE run. Now only on an explicit flag — minimax/mistral
  default to full attn+MoE.
- **Model pre-fetch must use `allow_patterns`, not `ignore_patterns`.** On the box's huggingface_hub,
  `ignore_patterns=[…,'consolidated*']` resolved to **0 files** (downloaded nothing). Mistral-Small-4
  ships BOTH HF-format `model-*.safetensors` AND `consolidated-*` dups, so the right move is an
  allow-list (`model-*.safetensors` + `*.json/*.jinja/*.txt/*.model`).
- **`HF_HOME=/share/huggingface`** (via `env_vars`) so runs reuse the box's model cache + persist
  downloads there (root's `~/.cache` would re-download). `HF_HUB_DISABLE_XET=1` avoids Xet stalls.
- **gemma `--lora_r`, never a bare `--r`** (`_gemma_cmd` + `gemma4.py`). `torchrun`
  (`torch.distributed.run`) argparse **prefix-matches** a bare `--r` against its OWN options
  (`--rdzv-backend`, `--role`, `--run-path`, `--redirects`, …) and aborts `ambiguous option: --r`
  before the script ever runs. Only bites on RunPod's torch **2.12.0** venv — the `tm` VM had
  2.12.1 which resolved it. Fix: `gemma4.py` adds a `--r/--lora_r` alias (`dest="r"`) and the
  orchestrator passes the long form `--lora_r`. (minimax/mistral were never affected — their LoRA
  flags are already `--attn_r/--moe_r`, no bare `--r`.)
- **`_extract_cuda_version` (compute.py) must parse RunPod's short `cuNNNN` tag.** It only matched
  the dotted `cuda12.4.1` / `cuda_12_6` forms → returned `None` for modern `runpod/pytorch` tags
  (`…-cu1300-…`, `…-cu1281-…`), so **`allowedCudaVersions` was never set**. Harmless for FA3 (any
  ≥12.6 host works) but FATAL for FA4 — without the filter the cu1300 image could land on a CUDA-12
  host and the cute kernel's CUDA-13 JIT fails. Fixed to also parse `cu1300→13.0`, `cu1281→12.8`
  (2-digit major + 1-digit minor). This is what makes a CUDA-13 image actually pin a CUDA-13 host.

### Deps-install resilience on a slow/flaky uplink (the `tm` VM, 2026-06-22)

A clean `/share/autotrain-llm-gemma` rebuild on tm hit a cascade of NON-network-looking failures, all
now fixed in `_ensure_venv`/`_pip`/`_install_fa4` (they ship per-run, so a re-run picks them up):
- **CUDA wheel download timeout** (`nvidia-nvshmem-cu13` from pypi.nvidia.com) → `UV_HTTP_TIMEOUT=600`
  + `_pip` retries the whole command, and a per-attempt **wall-clock cap SIGKILLs a true hang** →
  retry (so a stalled download can't wedge the run forever).
- **Stale `uv` git lock**: terminating a run mid-`git+` fetch leaves a lock in `<uv-cache>/git-v0/locks/`;
  every later FA4 install then waits the lock timeout (300s) and fails *identically*. `_pip` now
  **clears stale uv git locks before each attempt** + `UV_LOCK_TIMEOUT=60`. (Looks like "no internet" — it isn't.)
- **uv's internal git hangs** on the `flash-attention-512` clone over a flaky uplink (it ignores
  `GIT_HTTP_LOW_SPEED`, no timeout/resume). `_install_fa4` now **clones with SYSTEM git** (`--depth 1`,
  abortable via `GIT_HTTP_LOW_SPEED`, retry + 20-min kill-cap) and `uv pip install`s the local checkout.
  Plus a **skip-if-present** fast path (`flash_attn.cute` already imports → no re-fetch).
- **Model pre-fetch**: set `env_vars.HF_HOME=/share/huggingface` (reuse/persist the 62GB cache, NOT
  root's `~/.cache`) + `HF_HUB_DISABLE_XET=1` AND `HF_HUB_ENABLE_HF_TRANSFER=0` (both Xet *and*
  hf_transfer stall on tm → plain HTTP is slow ~1.5–3h for 62GB but doesn't hang). The first run caches
  the model in `/share`; later gemma runs skip the download.

Net: `train-250eeaa2` (gemma-4-31B FA4, 2× H20 GPUs 6,7, `batch_size=2 grad_accum=1`, 64k block) trained
end-to-end (~73→120GB/GPU, 100% util). `batch_size=2` is what makes 64k context fit on H20s.

**Monitoring gotcha:** the gateway's live log (`/tmp/sgpu-train/<id>/_full.log`, streamed via an SSH
`tail -F`) can **freeze mid-run on long jobs** when that tail SSH connection drops — but the run is
detached (`setsid`) on the box and still finishes + **finalizes-from-log** correctly (status flips to
done/failed once reconciled). For live progress on a long run, SSH the box and tail the authoritative
**`/tmp/sgpu_train_<run_id>.log`** (the full llm_finetune + torchrun output) directly; the gateway
sweeps it after the run finalizes.

**Step count:** steps/epoch = ⌈bins / n_gpus⌉. The smoke datasets are ~26–32 bins, so on 8 GPUs an
epoch is only ~3–4 steps — set `max_epochs` high enough that `max_steps` is reachable (e.g. 5).

**Shared box etiquette** (the `tm` VM): other users run here. Check `nvidia-smi` and use free GPUs;
H20s are 143GB each so big-model FSDP shards fit alongside small jobs. `--container/share` venvs are
cached between runs (the per-arch venv + the HF model cache make re-runs fast).

## Context parallelism — zigzag ring attention (gemma-4 only), 2026-07-04

`--cp_size N` (>1) shards ONE packed sequence across N GPUs (a CP group) and computes exact attention
by ringing K/V — trains context longer than one GPU's VRAM. New files (vendored copies of the
standalone `autotrain/gemma4/` authorities — **re-sync from there**): `ring_zigzag_attn.py` (the ring
kernels) + `context_parallel.py` (CP process group, zigzag `shard_batch`, `cp_ring_attention` backend).
The full design (hybrid cute-512-global + position-aware-256-sliding, the LSE convention, the
non-contiguous-recv permutation bug, the per-doc loss pre-shift) lives in **`autotrain/gemma4/CLAUDE.md`
→ "Context parallelism"**. Wiring here:

- **`gemma4.py`** takes `--cp_size`: builds a CP process group (FSDP still shards params over ALL ranks —
  orthogonal; DP is across CP groups), a dp-based `DistributedSampler` (whole CP group sees the same bin),
  per-batch `cp.shard_batch`, and a CP loss branch (labels are pre-shifted per-doc targets → no
  `hidden[:-1]/labels[1:]` shift, which is invalid across zigzag chunk boundaries). All gated on `cp_size>1`.
- **`llm_finetune._gemma_cmd`** appends `--cp_size {nproc}` when `cfg["context_parallel"]` and nproc≥2
  (CP over ALL run GPUs, dp=1 — the form's toggle is boolean).
- **`training_api`** carries `context_parallel` in the run config; the web form
  (`/autotrain/new?task=llm`) shows a **Context parallelism** toggle, gemma4-only, gated to ≥2 GPUs.

**Verified 2× H100 SXM 2026-07-04:** ring primitives (out+dq/dk/dv == non-ring, causal + sliding),
`shard_batch` round-trip, and the real Gemma4 backbone under CP == non-CP hidden states (both layer
types). ⚠ A full gemma-4-31B CP **training run** (optimizer + checkpoint) is wired + ready
(`HF_TOKEN` in `autotrain/.env`) but NOT yet run end-to-end — needs the 62GB model + an `llm_packed`
dataset. Only gemma-4 is wired (qwen/minimax/mistral would each need the same hybrid dispatch).
