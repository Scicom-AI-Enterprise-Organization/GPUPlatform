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
```

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
default FA4 path which swaps the wheel for the cute fork. `--batch_size 1` always (the collator packs
each bin into ONE sequence). minimax/mistral always pass `--low_cpu_shard_load`.

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

**Also validated on a freshly-provisioned RunPod pod (2026-06-20)** — the gateway spawns the pod,
ships the trainer, runs it detached, uploads to S3, and **tears the pod down** (no orphaned billing,
confirmed `pod … torn down` for every run). TTS (1× H100) ✅, STT/Whisper (1× H100) ✅, gemma-4-31B
(4× H100) ✅ — gemma ran the **FA3 path** (`gemma_fa4=false`) on the default `cu1281` image, since
the FA4 cute fork needs a guaranteed CUDA-13 host (`cutlass-dsl[cu13]`) the stock image can't
promise. The only RunPod-specific break was the `--r`/torchrun collision (below).

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
