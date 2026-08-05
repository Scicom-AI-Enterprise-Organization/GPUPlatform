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
trainer, RE-SYNC the vendored copy here (preserving those gateway edits). ⚠ **The DPO additions
(qwen3_5.py + gemma4.py: `dpo_collator`, `lora_disabled()`, the DPO forward branch, `--dpo/--dpo_beta`,
and qwen's `make_custom_cls(dpo_beta=)` / gemma's `enable_dpo()`) currently live ONLY in these
vendored gateway copies — NOT in the standalone `autotrain/{gemma4,qwen3.5}/`.** Treat them like the
MODEL_ID/target_modules edits: preserve on re-sync (or port DPO to the standalone first). The
`triton_func.py` + `triton_dpo.py` kernels ARE synced from `small-ablation/multipacking-dpo`.

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
   uploads to S3 (+ optional HF push). Checkpoints save at every EPOCH END + every
   `checkpointing_step` + at `max_steps` — `lora.pt` is a single overwritten file, so to keep an
   exact epoch-boundary snapshot, copy it off the work dir before the next save lands.
   **`/stop-early` works for gemma since 2026-07-21** (`gemma4.py` polls `$SGPU_STOP_FLAG` per
   optimizer step, rank-0 decision broadcast so all ranks checkpoint+break on the same step —
   trainer files ship per-run, no gateway restart). ⚠ The other LLM trainers (qwen/minimax/
   mistral/nemotron) still IGNORE the flag — /stop-early on those runs does nothing; port the
   same block if needed.

## Architecture auto-detect (the core of this subsystem)

`detect_arch(model_id)` (in `llm_finetune.py`, `llm_pack.py`, and `training_api._llm_arch` — keep
all three in sync): `"minimax"`→minimax, `"mistral"`→mistral, else gemma. Everything per-arch lives
in `llm_finetune._ARCH` + `_fa_mode`:

| | gemma-4 | MiniMax-M2 | Mistral-Small-4 |
|---|---|---|---|
| trainer | `gemma4.py` | `minimax/minimax_m2.py` | `mistral/mistral_small.py` |
| size / dtype | 31B dense bf16 | 230B FP8 MoE | 119B FP8 MoE (MLA) |
| attention | **FA4 cute fork** (default; head_dim-512), or FA3+dynamic_attention | stock FA3 (head_dim 128) | stock FA3 (head_dim 128) |
| venv | `/share/autotrain-llm-gemma-v2` | `/share/autotrain-llm-minimax` | `/share/autotrain-llm-mistral` |
| `kernels` pin | `==0.14.1` | `>=0.12,<0.13` | `>=0.12,<0.13` |
| extra deps | `peft` | `accelerate` | `accelerate` |
| pre-flight | `test_attention.py` (FA3) / skipped (FA4) | `test_lora.py` (CPU) | `test_lora.py` (CPU) |
| run env | `GEMMA_ATTN`, FA4 JIT cache | — | `MISTRAL_DEQUANT_TRITON=1` |
| LoRA CLI | `--r --alpha --target_modules [--train_embeddings]` | `--attn_r --moe_r --attn_alpha --moe_alpha [--no_moe_lora --train_embeddings]` | + `--no_shared_lora [--train_embeddings]` |

**Per-arch venv version (`_LLM_VENV_VERSION`, 2026-07-12).** The box venv is
`/share/autotrain-llm-{arch}` unless an arch is bumped in the `_LLM_VENV_VERSION` map — gemma is now
pinned to **`gemma-v2`**. Bumping an arch's entry forces a FRESH venv on its next run (the old dir is
left untouched for in-flight runs); this is how you roll deps/kernels forward, because the deps
installer's "FA4 cute already imports → skip reinstall" fast path otherwise keeps a stale kernel in a
reused venv. `gemma-v2` re-clones the FA4 fork → picks up the head_dim-512 forward retile
(m64n80→m64n64, +15–20%; see `autotrain/gemma4/CLAUDE.md`). ⚠ The map lives in BOTH `training_api.py`
(gateway — ships `venv_path` + re-derives it in every post-train op) and `training/llm_finetune.py`
(box-side fallback); keep the two copies in sync or a run trains in one dir and its export/merge looks
in another ("venv python not found").

**`--train_embeddings` (added 2026-07-10, ALL LLM archs).** Also FULL-trains the token embeddings + LM
head on top of LoRA — attention-only LoRA can only nudge the output distribution via hidden states, so
teaching a model to reliably emit a specific special token (e.g. gemma's `<|tool_call>`) is far easier when
the head/embeddings adapt. The shared helper **`_trainer_common.unfreeze_embeddings(model, rank, logger)`**
unfreezes `get_input_embeddings()` + `get_output_embeddings()` right after LoRA is applied, BEFORE FSDP
sharding (root `fully_shard` shards them; the loss already projects hidden states through
`self.lm_head.weight` and Liger FLCE returns its gradient → trained from BOTH the input-embed lookup and the
output projection). It handles **tied** heads (gemma-4, small qwen — `tie_word_embeddings=True` → ONE weight)
and **untied** (Qwen3.6-27B, MiniMax-M2, Mistral-Small-4 — two separate weights). For the FP8-MoE pair the
embeddings/lm_head are the bf16 non-FP8 parts, so they train as-is (requires_grad survives the meta-init
`to_empty`/broadcast path). Checkpoint saves the whole weight(s) into `lora.pt` (non-`.lora_*` keys;
`lora_meta.json` gets `train_embeddings:true`); the merges apply them back: gemma `merge_infer.py` +
qwen `merge_to_disk.py` COPY the full weight (replace, not add-delta), and minimax/mistral `merge_to_bf16.py`
+ `merge_infer.py` already load-by-name so they auto-apply. Pairs naturally with adding the MLP
(`gate/up/down_proj`) to `--target_modules` (gemma/qwen) — the FP8-MoE trainers adapt experts by default.
⚠ **Incompatible with `--dpo`** (the LoRA-disabled reference assumes the base weights stay frozen; training
the shared head makes the reference drift) — rejected in the gemma+qwen trainers, create-run (400), and hidden
in the form for DPO. Wired: `_gemma_cmd`/`_qwen_cmd`/`_moe_cmd` (`cfg["train_embeddings"]`) → `training_api`
(`train_embeddings` field + config) → form toggle (`/autotrain/new`, all LLM archs + SFT only). Costs extra
optimizer state — the trainers' `cpu_offload` default helps (ON for gemma/qwen). Lives ONLY in these vendored
gateway copies (like DPO/CP), not the standalone `autotrain/`.

**✅ GPU-smoked (2026-07-10): gemma tied + nemotron untied.** Gemma `train-66c888ed` — gemma-4-31B on a
**65k pack** (`ds-fcd94aac`, 548 bins @ cap 65536), 4× H20, r16 q/k/v/o + `train_embeddings`, 10 steps:
unfroze "embeddings + lm_head (**tied, one weight**) — 1409.3M", losses 3.86→2.84 (finite, trending down,
~1.1–1.4k tok/s at S≈54k), and the checkpoint saved **461 tensors = 460 LoRA + 1 full tied-embed weight**
(the 1.4B CPU-offloaded DTensor gathered over the gloo backend — the exact collective `cpu:gloo` exists
for) → S3. Nemotron (untied → 2 weights) smoked earlier on the tiny model + merge (2/2 full weights
replaced). ⚠ **gemma at 65k on H20s REQUIRES `cpu_offload` even on 4 GPUs**: an explicit
`cpu_offload=false` attempt (`train-ecdd6e11`) OOM'd at S=54k in the FA4 backward at ~133 GB/GPU — the
O(seq×layers) activation-checkpoint memory + the 1.4B trainable embed's optimizer state don't fit beside
the unsharded activations; the arch default (ON) is load-bearing, don't override it for ≥64k runs.

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
  the model in `/share`; later gemma runs skip the download. ⚠ `training/hf_export.py` used to
  **re-enable hf_transfer itself** (`env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")`) for the
  base-model download during merge — contradicting this exact finding; removed 2026-07-21 (the file
  ships per-request, no gateway restart needed). If an hf-export stalls at the download step again,
  check nothing reintroduced that env.

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

## DPO — Direct Preference Optimization (qwen + gemma-4), 2026-07-08

`training_type="dpo"` (create-run) trains the **qwen or gemma-4** trainer on preference pairs with
the **fused multipacked DPO loss** from
`Scicom-AI-Enterprise-Organization/small-ablation/multipacking-dpo` (vendored: `triton_func.py` =
the fused linear+log-prob Triton kernel, `triton_dpo.py` = `fused_dpo_loss` — logits NEVER
materialized; forward keeps only the [T] logsumexp, backward recomputes logits chunk-by-chunk.
4.2× lower peak than Liger's padded DPO; **re-sync from small-ablation**, its `__main__` blocks
are GPU correctness gates). The qwen and gemma trainers share the **identical** `dpo_collator` +
DPO-forward + `lora_disabled()` logic (byte-equal collators, verified) — keep them in sync; the
only structural difference is qwen builds the custom class via `make_custom_cls(dpo_beta=...)`
(two base classes) while gemma sets it on the single subclass via `model.enable_dpo(beta)`.

- **Dataset**: `kind=llm_dpo_packed` — packed by 'Pack for LLM' with **objective=dpo**
  (`llm_pack.pack_dpo_rows`). Three source shapes (`extract_pair_messages`): **continuation** —
  a `prompt_field` messages column + `chosen`/`rejected` holding only the turns AFTER it (agentic
  trajectories: differing lengths, `role: tool` results interleaved); **ultrafeedback** — full
  message lists sharing the prompt turns, no prompt column; **plain strings** + a `prompt_field`.
  See "Agentic DPO" below — the first shape needs the tools column and env masking. Same ChiniDataset
  columns as llm_packed but: every bin holds WHOLE pairs, doc count is even, **first half of the
  docs are the chosen responses** (pair k = (doc k, doc K+k)), and `labels` are **pre-aligned
  next-token targets** (targets[j]=ids[j+1] on response positions, −100 on prompt + final; NO
  shift at loss time). The prompt/response boundary comes from rendering the prompt with
  `add_generation_prompt=True`; non-prefix-stable templates fall back to the longest common
  prefix of the two renders (equivalent for DPO — shared-prefix log-probs cancel in the loss).
- **Trainer** (`qwen/qwen3_5.py` or `gemma4.py`, `--dpo --dpo_beta β`): `dpo_collator` reorders
  multi-bin batches chosen-first across the row; the **reference model is THIS model with LoRA
  disabled** (`lora_disabled()` flips a module flag in `LinearLoRA.forward` — base frozen + B=0
  init ⇒ ref == initial policy, no second model copy), one extra no-grad backbone pass per
  microbatch. lm_head isn't LoRA-wrapped so one weight serves both sides. First loss ≈ **ln 2 =
  0.693** (the policy==ref sanity). Metrics are appended after the loss (the `@@STEP` parser
  keys on "step: N … loss: L" unchanged) — see "DPO metrics" below. Grad-accum weighting mirrors the SFT token-weighted
  scheme with the **pair** as the unit (the DPO loss is a pair-mean — token-weighting would re-add
  a length bias). `lora_meta.json` gets `objective: "dpo"` + `dpo_beta`.
- **Constraints** (validated in create-run AND llm_finetune AND the trainer): qwen or gemma-4 arch;
  **incompatible with context parallelism** (per-sequence log-prob sums would need cross-rank
  reduction); qwen's `SGPU_MAX_SEQ_LEN` truncation is skipped under DPO (would cut pairs).
- **Verified locally (CPU)**: pack invariants (real Qwen tokenizer, both single- and multi-turn +
  string/message-list source shapes, drop paths), collator reorder/pairing over multi-bin batches,
  `dpo_loss_reference` on the packed layout == a from-scratch per-pair computation (+ policy==ref →
  ln 2), and gemma's `dpo_collator` byte-equal to qwen's.
- **✅ GPU-verified end-to-end (2026-07-08, tm-2 = prov-940055fc, 8× H20).** Two 10-step dry runs
  (β=0.1, lr 5e-6, LoRA r=16, 4 GPUs each), both **exit 0** with textbook curves — **step-0 loss =
  0.6931471824645996 = ln2 EXACTLY on both** (the policy==reference sanity), loss ↓, reward_acc →
  ~1.0, margin → +:
  - gemma (`google/gemma-4-31B-it`, FA4 cute kernel, GPUs 4-7): 0.693 → 0.475, reward_acc 0 → 1.0,
    margin 0 → 0.556 — the DPO double-forward runs fine through FA4.
  - qwen (`Qwen/Qwen3.6-27B`, GatedDeltaNet, GPUs 0-3): 0.693 → 0.638, reward_acc → ~1.0, margin
    0 → 0.118 (early per-step wobble is just the 2-bin sampler alternating pair sets — not a bug).
  Both wrote `lora.pt` + `lora_meta.json` (`objective=dpo, dpo_beta=0.1`) to S3.
  ⚠ **DPO datasets MUST be packed with the base model's real tokenizer** (Qwen3.6 vocab 248077 ≠
  Qwen3-0.6B's 151936; gemma-4 262144) — a small proxy tokenizer produces wrong ids. The local
  gateway needs a gemma-authorized `HF_TOKEN` in its env to pack the gated gemma tokenizer. The
  Triton kernel `__main__` gates (`python triton_func.py` / `triton_dpo.py`) are still worth a run
  when re-syncing from small-ablation.

### DPO metrics — 5 numbers, pooled across ranks, 2026-08-03

Emitted per microbatch on the trainer's loss line and carried into `@@STEP` →
`result_json.steps` → the **Preference (DPO)** card on the run page:

| key | meaning |
|---|---|
| `reward_acc` | fraction of pairs where chosen outscored rejected ("chosen win rate") |
| `reward_margin` | mean(chosen − rejected) — the quantity DPO maximizes |
| `reward_chosen` | mean implicit reward β·log(π/π_ref) on the **preferred** side |
| `reward_rejected` | same on the dispreferred side |
| `pairs` | preference pairs in the step (reward_acc's sample size) |

- **⚠ Watch `reward_chosen`, not just accuracy.** The loss only sees the DIFFERENCE, so it is
  perfectly happy to drive BOTH rewards down — the model unlearns the preferred answers too, just
  more slowly than the rejected ones. Win rate and margin look great throughout. A chosen line well
  below 0 means lower the LR, raise β, or stop. This is the single most common way a "successful"
  DPO run produces a worse model, and it was **invisible** before: only `reward_acc`/`margin`
  existed, and only in the raw log text + wandb — nothing reached `result_json` or the UI, so a run
  without a wandb key had no preference signal anywhere in the platform.
- **⚠ The LOSS is pooled in the same collective (DPO only) — it has to be.** `fused_dpo_loss`
  returns a mean over the LOCAL pairs and nothing reduces it, so `output['loss']` is rank-0-only.
  Once the rewards went global, the two lived on different populations, and a chart pairing them
  is apples-to-oranges: run `train-2d624e00` logged a step with loss **0.5982** while
  `-logσ(margin)` was **0.7295** — below a floor convexity forbids for a single population, which
  reads as a broken metric. `st[5]` carries a pair-count-weighted loss so the pooled mean is exact.
  **Logged only** — backward still runs on the local loss, which is correct (FSDP averages grads).
  SFT keeps the historical rank-0-local loss; only the DPO branch pools.
- **⚠ The stats are all-reduced across ranks; the collective must run on EVERY rank.** They used to
  be computed inside `if rank == 0`, i.e. on 1/world_size of the pairs (`DistributedSampler` shards
  the bins) — with 4 GPUs and a couple of pairs per bin that quantized `reward_acc` to a handful of
  values. SUMS are reduced (not means) so the pooled mean is exact under an uneven split. Moving
  the `dist.all_reduce` under the rank-0 guard **deadlocks the job** — it is deliberately above it.
- Both trainers must stay byte-similar here (`gemma4.py` + `qwen/qwen3_5.py`), same as the
  `dpo_collator`. Parsed by `_DPO_METRIC_RE` in `llm_finetune.py` (`margin` → `reward_margin` via
  `_DPO_METRIC_ALIAS`); SFT lines carry none of these, so the payload is unchanged for SFT.
- UI: `DpoCurve` in `autotrain/[runId]/training-detail.tsx`, self-hiding when no step carries
  `reward_acc`. **Two charts, never a dual axis** — a 0..1 rate and a signed unbounded reward share
  no scale. Series colours are CVD/contrast-validated for BOTH themes (`dataviz` skill's
  `validate_palette.js`); the lighter -400/-500 steps fail the dark-mode lightness band.

### Agentic DPO — tool declarations + environment masking, 2026-08-02

The original DPO packer assumed the textbook shape: **one** assistant reply per side, no tools. A
**tool-calling** preference corpus (each side a whole trajectory — assistant turns interleaved with
`role: tool` results) broke it three ways, all of which packed *successfully* and trained the wrong
objective. Found on `ucc_ai_research/synthetic-generation/tm-text-assist-simulation/dpo-v2` (678
rows / 320 with `has_pair & pair_guard_ok`), gemma-4:

1. **⚠ A JSON-STRING message-list cell was packed as a literal JSON dump.** `extract_pair_messages`
   dispatched on `isinstance(cell, str)` → the plain-response branch → `{"role": "assistant",
   "content": '[{"role": "assistant", "content": "", "tool_calls": …}]'}`. The finetune learns to
   emit the serialized transcript. Parquets store message lists as JSON strings all the time (it's
   why `extract_messages` exists) — so **parse first, and treat a cell as a response string only
   when it does NOT parse into messages**. Without a `prompt_field` the same rows instead hit the
   ultrafeedback equal-length + identical-prefix check and **all 320 dropped → 0 bins**.
2. **⚠ `pack_dpo_rows` never rendered the tools column** (no `tools_field` at all, while
   `pack_rows` has had one since day one). A preference over tool CALLS trained against a prompt
   that never declared the tools — train/serve mismatch on every row. Now plumbed end-to-end
   (`datasets_api` already forwarded `tools_field`; only the DPO branch of `dataset_transform`
   dropped it), and `dataset_transform` reads the column for DPO too.
3. **⚠ Environment tokens were inside the DPO log-ratio — `mask_env=True` is the new default.**
   This is NOT the SFT masking rationale, and the usual "DPO needs no mask, mask the prompt and
   score the rest" instinct is right *only* for a single-reply completion. Here **37.0% of the
   scored tokens were KB search results** — text the policy cannot emit — and the chosen/rejected
   env spans share **48% of their vocabulary**, so the loss pushed the *same* KB sentences up on
   one side and down on the other. 41/320 pairs had env tokens outnumbering model tokens.
   `tokenize_pair` now scores only tokens the template's `{% generation %}` mask marks. A template
   with **no** generation mask (all-zero → every arch but gemma-4, see the 2026-07-22 correction in
   `gateway/CLAUDE.md`) **silently degrades to the old full-completion behaviour** rather than
   scoring nothing; `full_seq_labels=true` on the pack card is the explicit opt-out.

**The gemma stop token stays scored.** Mask anchor E1 wraps `<|tool_response>` — gemma-4's actual
stop after a tool call — so the scored span runs reasoning → `<|tool_call>…<tool_call|>` →
`<|tool_response>` and then cuts, excluding the response body. Same lesson as the SFT
tool-call-looping bug: whatever masks, the eos that terminates an assistant span must stay inside.

**Read `env_tokens_masked` / `scored_tokens` / `rows_with_tools` off the pack summary.**
`env_tokens_masked: 0` on a corpus you know carries tool results means the generation mask didn't
apply and the loss is scoring environment output — the same failure mode as `assistant_masked_rows`
collapsing to 0 on the SFT side.

**Verified on the real corpus** (local, `google/gemma-4-31B-it`): 320/320 pairs packed, 0 dropped,
67 bins @ 65536, 89.1% efficiency, 320 with tools, 568,014 scored vs 333,545 env tokens masked
(37.0%). Bin layout re-checked (even doc count, first-half chosen, `position_ids` reset per doc,
`sum(attention_mask) == len(input_ids)`), targets confirmed pre-aligned, and the prompt boundary
verified to land **past** the true prompt on all 320 pairs (0 prompt tokens scored). ⚠ gemma-4's
prompt render is **not** prefix-stable — `add_generation_prompt=True` emits an *empty* thought
channel (`<|channel>thought\n<channel|>`) where the real turn emits reasoning, so it diverges ~1
token before the end and `tokenize_pair` takes the common-prefix fallback on **every** row. That's
benign (it lands +6..+698 past the prompt, shared tokens cancel) — don't "fix" it. Sizing: the
pair (chosen+rejected) max is **46,618 tokens**, so `sequence_length` must be ≥ 65536 to keep every
pair; at 32768 four pairs drop, at 16384 sixty-three do.

Unit tests: `gateway/tests/unit/test_llm_pack_dpo.py` (tokenizer-free — a fake template reports a
generation mask the way gemma's injected blocks do).

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

## Nemotron-H (NVIDIA Nemotron-3-Nano-30B-A3B) — hybrid Mamba2/attention MoE, 2026-07-10

A 5th LLM arch. Nemotron-H is a **hybrid**: 52 `NemotronHBlock` layers = **23 Mamba2 SSM + 23 MoE +
6 attention** (per `config.layers_block_type`, derived from `hybrid_override_pattern`). bf16, untied
embeddings (`tie_word_embeddings=false`), vocab 131072. Native in **transformers 5.12.1** (incl. the
MoE `NemotronHMoE`/`NemotronHExperts` — 5.5.0 predates it).

- **`nemotron/nemotron_h.py`** — the trainer. `CustomNemotronHForCausalLM` runs the backbone → Liger
  FLCE on hidden states (never materializes [B,S,V] logits). LoRA (custom `LinearLoRA`, B=0 init)
  wraps attention **q/k/v/o_proj** + Mamba **in_proj/out_proj** (nn.Linear); the MoE experts are 3D
  `nn.Parameter` (`NemotronHExperts.up_proj/down_proj`) so the Linear LoRA leaves them frozen.
  `--train_embeddings` (shared `tc.unfreeze_embeddings`) full-trains the untied embed+lm_head.
  FSDP2 shards the single `NemotronHBlock` class; `--cpu_offload` default OFF (MoE bulk frozen → fits
  sharded on H20s). No DPO, no context parallelism.
- **⚠ NO multipacking — ONE DOCUMENT PER BIN.** The HF NemotronH forward has no cu_seqlens/seq_idx
  plumbing, so concatenating docs would LEAK the Mamba SSM state across document boundaries (no
  per-doc reset). `llm_pack` packs `arch=nemotron` **one-doc-per-bin** (`one_doc_per_bin`), and the
  trainer's collator **PADS** a batch of single-doc bins (right-pad; `attention_mask` marks real
  tokens → `create_causal_mask` + the mamba mask zero the pad; correct for both attention and Mamba)
  — NOT the concatenating varlen collator the attention archs use.
- **Fast path (default since 2026-07-10; NVIDIA's own recipe = Megatron-Bridge → TE flash-attn +
  fused mamba kernels — we get the HF equivalents):** attention = **hub FA3**
  (`NEMOTRON_ATTN=kernels-community/flash-attn3`, head_dim 128 GQA 32/2; sdpa + a padding mask
  materializes an O(S²) causal mask at long context — `NEMOTRON_ATTN=sdpa` is the fallback), and
  Mamba = the **fused Triton kernels** (`NEMOTRON_MAMBA_KERNELS=1` → `config.use_mamba_kernels=True`;
  `mamba-ssm` is `lazy_load_kernel`-fetched from the kernels hub at model load, but its `ssd_combined`
  does a plain `import causal_conv1d_cuda` and asserts on it — that compiled ext is NOT bundled with
  the hub kernel and is NOT a `kernels`-hub kernel, so **`causal-conv1d` is BUILT FROM SOURCE in the
  venv** by `_install_nemotron_kernels` (same cu13/`--no-build-isolation` recipe as qwen; needs
  `einops` + nvcc). ⚠ **Regression fixed 2026-07-12:** causal-conv1d was NOT in `_ARCH["nemotron"]`
  deps, so a venv rebuild dropped it → the fused path assert-crashed (worked earlier only because a
  prior venv had it). Two-part fix: (a) `nemotron_h.py` probes `causal_conv1d_cuda` (import torch
  first! the .so links libc10.so) and degrades to the torch SSD path with a warning if absent —
  graceful, not a crash; (b) `_ensure_venv`/`_install_nemotron_kernels` builds it so the fast path
  actually works at 32k. The torch SSD fallback (`NEMOTRON_MAMBA_KERNELS=0` OR causal-conv1d missing)
  materializes fp32 chunk intermediates in autograd across the 23 mamba layers — THE long-context
  memory/speed hog, and it **OOMs at 32k** (~47GB single alloc on H20) → the fused path is REQUIRED
  for ≥32k. (All 8 dropdown models smoke-verified at 32k / 3 steps on tm-2, 2026-07-12.)
  ⚠ **Three LoRA/kernel traps, all handled in the trainer (found via the NaN bisect + the
  autotrain/nemotron gates — READ THAT CLAUDE.md before touching the backends):**
  (1) the dispatch reads `self.in_proj.weight.device.type` → `LinearLoRA` exposes `weight`/`bias`
  properties delegating to the wrapped Linear; (2) the mem-eff split kernel
  (`mamba_split_conv1d_scan_combined`) fuses the RAW `out_proj.weight` into the kernel → a LoRA-wrapped
  out_proj's adapter would be SILENTLY bypassed, AND the hub build's split kernel hard-asserts
  `causal_conv1d_cuda is not None` (its OWN reference is None even when the pip causal-conv1d IS
  installed) → it can't run here at all. So the trainer forces `use_mem_eff_path=False` on **every**
  mamba mixer (unconditionally, when the fused path is on — falls to the non-split fast path:
  causal_conv1d_fn + mamba_chunk_scan_combined + module-call out_proj). ⚠ **Fixed 2026-07-12:** the
  old guard only disabled it for a LoRA-wrapped out_proj, so a q/k/v/o-only target set (out_proj stays
  a plain Linear) left the split kernel active → assert-crash; (3) **the hub mamba-ssm kernel's bf16 BACKWARD is broken** —
  NaN grads in exactly the ddt/dA path (dt_bias, A_log, dx→conv→in_proj; D/norm/out_proj grads stay
  cos 1.0), NOT contiguity, forward fine (cos 0.999988), fp32 exact → `_patch_mamba_kernel_fp32`
  wraps the kernel to upcast (x,dt,A,B,C)→fp32 + downcast the output (verified: xgrad cos 0.9999 vs
  torch). Without the patch: step-0 loss finite, then **NaN from step 1** (the step-0 backward poisons
  the LoRA weights at the first optimizer step — `train-6a672e67`/`train-9d5d1448`). FA3 attention is
  grad-exact (q/k/v/o weight grads cos 1.0 vs sdpa — `autotrain/nemotron/test_attention_kernels.py`).
  `_fa_mode` returns `"none"` → the deps install skips the FA3-wheel/FA4/qwen-kernel step (the hub
  kernels fetch at model load instead).
- **Deps/venv** (`_ARCH["nemotron"]`, `/share/autotrain-llm-nemotron`): torch 2.10+cu13 (same base as
  qwen) + `transformers==5.12.1 kernels<=0.14.0 sentencepiece einops`, PLUS `causal-conv1d` built from
  source by `_install_nemotron_kernels` (in the `_fa_mode=="none"` branch of `_ensure_venv`; nvcc/CUDA_HOME
  wired for nemotron like qwen). ⚠ the Nemotron tokenizer needs sentencepiece to instantiate — at pack
  time on the gateway AND at merge/serve time; the hub mamba-ssm kernel imports einops.
- **Chat template:** `_normalize_nemotron_turn` (llm_pack) parses tool-call `arguments` str→dict
  (REQUIRED — the template does `arguments|items`, LOUD crash otherwise) + maps `reasoning`→
  `reasoning_content`; `truncate_history_thinking=False` keeps all reasoning. NO `{% generation %}`
  block → assistant-only masking falls back to full-sequence labels (a follow-up, like gemma pre-fix).
- **Merge** (`nemotron/merge_infer.py`, used by `llm_playground.merge_lora_to_dir`): folds
  `scaling·(B@A)` into the base Linears + COPIES the full embed/lm_head (from `--train_embeddings`);
  tokenizer load + `device_map=auto` are best-effort (fall back so the weight merge always completes).

**✅ Dry-run verified end-to-end on tm-2 (2026-07-10, 2× H20 FSDP, a tiny random 52-layer NemotronH
so no 60GB download):** LoRA wrapped **q/k/v/o ×6 (the 6 attn layers) + in/out_proj ×23 (the 23 mamba
layers)**; `--train_embeddings` unfroze embed+lm_head (untied); FSDP sharded `NemotronHBlock`
(574.9M/rank); forward+backward+optimizer ran (**loss 8.19→8.00→7.41**, Mamba torch backward + Liger
loss + padding collator all work); checkpoint saved; **merge folded 70/70 adapters + replaced 2/2 full
weights** and saved a merged model.

**✅ REAL 30B run VERIFIED end-to-end (2026-07-10, `train-d0efd849`, tm-2 4×H20 via the gateway API).**
Packed `ds-f2116ddc` (12814 tool rows) → `ds-a077af13` (arch=nemotron, **12000 one-doc bins**,
seq_len 16384). Gateway create-run accepted nemotron; tm-2 built `/share/autotrain-llm-nemotron`
(transformers 5.12.1, `_fa_mode=none` → no FA wheel), downloaded the 59GB model, loaded **30115M
params**, LoRA wrapped **q/k/v/o×6 + in/out_proj×23** (8.72M trainable), FSDP across 4 GPUs (~15GB/GPU),
ran **10 steps** (finite loss, noisy — batch_size=1 single-doc bins so each step sees one document),
**Saved LoRA (140 tensors)** → S3 `training-runs/train-d0efd849/checkpoint/`, `@@DONE`, workdir torn
down. Torch Mamba SSD path ~500–690 tok/s.
⚠ **`training_api._LORA_TARGET_MODULES` must include `in_proj`/`out_proj`** (added) — else the Mamba LoRA
targets are silently filtered → only the 6 attention layers adapt.

**✅ Fast path verified end-to-end (2026-07-10, `train-c7957876`, same config):** FA3 + mamba kernels +
the fp32 kernel patch — all 10 steps finite (same per-bin loss shape as the baseline), checkpoint saved,
**~720–1250 tok/s ≈ 1.45× the torch baseline at short (~750-tok) docs** (the O(S)-attention/fused-scan
advantage grows with context; 16k-bin throughput not yet measured). Real-30B `compare_logits.py`:
PASSED (cosine 0.999929, argmax match, top-5 5/5). Gates + the full NaN investigation:
**`autotrain/nemotron/CLAUDE.md`**.
