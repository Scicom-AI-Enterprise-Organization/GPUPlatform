# autotrain — Qwen3.6 LoRA finetune, dense + MoE (FSDP2, tm 8× H20-3e)

Standalone training job (NOT part of the gateway). Finetunes the **Qwen3.6** family with a custom
LoRA on a packed function-calling dataset, using FSDP2 (`fully_shard`) + CPU offload across N GPUs.
Sibling of `../gemma4`, `../minimax-m2`, `../mistral-small` — same multipack + custom-LoRA shape.

**Qwen3.6 reuses the Qwen3.5 architecture** (`model_type` `qwen3_5` / `qwen3_5_moe`), so this one
job dir (still named `qwen3.5/`) trains BOTH targets — `qwen3_5.py` auto-detects dense vs MoE from
the config and picks the right classes:

| model | kind | class / module | geometry |
|-------|------|----------------|----------|
| `Qwen/Qwen3.6-27B`     | dense | `Qwen3_5ForConditionalGeneration` / `modeling_qwen3_5`         | 64 layers, head_dim 256, GatedDeltaNet hybrid |
| `Qwen/Qwen3.6-35B-A3B` | MoE   | `Qwen3_5MoeForConditionalGeneration` / `modeling_qwen3_5_moe`  | 40 layers, 256 experts / 8 active (~3B active), GatedDeltaNet hybrid |

Both are GatedDeltaNet hybrids and multimodal `*ForConditionalGeneration` (vision blocks present; we
train text-only). LoRA wraps only the attention `q/k/v/o_proj`, so the MoE experts are untouched and
the **same training path serves both** — the only per-model difference is which module's classes get
sharded / activation-checkpointed / GatedDeltaNet-patched (resolved in `resolve_arch`).

```
qwen3_5.py        training entrypoint (torchrun, FSDP2, custom LinearLoRA, packed varlen, Liger FLCE loss)
pack_dataset.py   render the Qwen3.6 chat template + multipack a HF chat-parquet → ChiniDataset
merge_infer.py    fold the custom LoRA back into the base (dense or MoE) + generate (coherence check)
run.sh            pod/box bootstrap: venv + deps + dataset pack + torchrun
CLAUDE.md         this file
README.md         short overview
.gitignore        packed_data*, .venv, checkpointing*, memory pickles
```

The original single-file `one_script_run.sh` was split into these files to match the `../gemma4`
layout. `one_script_run.sh` is kept for history.

## The GatedDeltaNet hybrid attention (same for dense + MoE)

Most layers are **GatedDeltaNet linear attention** (O(S) state recurrence, not softmax attention),
interleaved with periodic **full-attention** layers (`layer_types` = mostly `linear_attention` with a
`full_attention` every ~4th). Only the full-attention layers have `q/k/v/o_proj`, so LoRA lands on
those (~16 of 64 layers dense → 128 LoRA tensors; ~10 of 40 MoE → 80 tensors).

- **Linear-attention layers** run the **FlashQLA** `chunk_gated_delta_rule` kernel
  (`git+https://github.com/QwenLM/FlashQLA.git`) + `causal_conv1d` for the short conv. `qwen3_5.py`
  monkeypatches each `Qwen3_5[Moe]GatedDeltaNet.chunk_gated_delta_rule` to **`.contiguous()` the `v`**
  tensor first — the TileLang kernel requires `stride[-1] == 1` or it errors. The kernels **JIT-compile
  on the first step** (TileLang, ~2–4 min the first time; cached to disk after, so warm steps are fast).
- **Full-attention layers** use `attn_implementation="kernels-community/flash-attn3"`, auto-fetched at
  model load by the `kernels` package (pinned `kernels<=0.14.0`). No FA3 wheel, no torch-2.12 pin
  (Qwen3.6's full layers are head_dim ≤ 256 so the kernels-community FA3 just works on torch 2.10).

The collator carries packing as `cu_seq_lens_q/k` + `position_ids` (no dense `(1,S,S)` mask), so both
the linear and full layers see per-document boundaries.

## Training design (`qwen3_5.py`) — same shape as the gemma4/minimax siblings

- **`resolve_arch(model_id)`** reads the config, branches on `"Moe" in architectures[0]`, imports the
  dense **or** MoE classes (`base` / `decoder` / `vision` / `gdn`), and applies the matching
  `apply_liger_kernel_to_qwen3_5[_moe]` (both exist; liger lazy-loads so `dir()` won't list them).
  `make_custom_cls(base)` builds the loss-overriding subclass on top of whichever base class. Both
  `--model_id`-selectable.
- **Custom `LinearLoRA`** (not PEFT): `y = Wx + (alpha/r)·B(Ax)`, wraps `q/k/v/o_proj` (vision skipped),
  `B` zero-init (→ no-op at init), bf16. `r=256`, `alpha=512` (`scaling=2.0`) by default.
- **FSDP2** `fully_shard` per `Qwen3_5[Moe]DecoderLayer` + `Qwen3_5[Moe]VisionBlock`, then the root,
  under **`CPUOffloadPolicy()`** + `MixedPrecisionPolicy(bf16 param / fp32 reduce)`.
- **`init_process_group(backend="cpu:gloo,cuda:nccl")`** — required: the checkpoint's `full_tensor()`
  all-gather of CPU-resident DTensor shards is a CPU collective NCCL can't service.
- **Activation checkpointing** (`NO_REENTRANT`) per `Qwen3_5[Moe]DecoderLayer`.
- **Loss**: `LigerFusedLinearCrossEntropyLoss` (skips materializing the lm_head logits) — the wrapper
  forward returns only `{"loss"}`.
- **Optimizer**: AdamW over LoRA params only, `fused=False` (fused AdamW is CUDA-only; CPU-offload
  runs the step on CPU-resident params).
- **Checkpoint**: saves only the LoRA adapters to **`--checkpoint_dir`** (`lora.pt` + `lora_meta.json`,
  meta carries `model_id` + `is_moe`), never the full base. Use a per-model dir (`run.sh` derives
  `checkpointing-<model-slug>` automatically) so the two models don't clobber each other.
- `torch.cuda.set_device(rank)` before NCCL init (else every rank lands on cuda:0).
- `--batch_size 1` (the collator `np.concatenate`s the whole DataLoader batch into ONE packed
  sequence; >1 multiplies the per-step sequence length).

✅ **`mm_token_type_ids` is accepted by BOTH dense and MoE** (`Qwen3_5[Moe]Model.forward` takes it +
`**kwargs`), so the collator's all-zero `mm_token_type_ids` + `cu_seq_lens_*` pass straight through —
the gemma4-inherited kwarg does NOT need dropping (confirmed on real runs, 2026-07-01).

## Dataset packing (`pack_dataset.py`) — Qwen3.6 chat template

Identical multipack to the siblings, with Qwen3.6-specific rendering:
- **All-turn reasoning** (default ON): the stock template only emits `<think>…</think>` on turns AFTER
  `ns.last_query_index`. Qwen3.6's template guards on
  `(preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index)`,
  so we pass **`preserve_thinking=True`** to render EVERY model turn's reasoning. (The legacy Qwen3.5
  string-swap `loop.index0 > ns.last_query_index` → `>= 0` is kept as a fallback; it's a no-op for 3.6
  since that guard string isn't present — the two mechanisms coexist harmlessly.) `--native-reasoning`
  disables both.
- **GLM → Qwen normalization**: `role: observation`→`tool`, `content: null`→`""`, bare
  `tool_calls`/`functions` wrapped to OpenAI `{type:function, function:{…}}` with **dict** (not
  JSON-string) `arguments`.
- **Tokenizer default `Qwen/Qwen3.6-27B`.** Qwen3.6-27B and Qwen3.6-35B-A3B share the **same tokenizer
  + chat template** (vocab 248077), so **ONE pack serves both**. It differs from Qwen3.5-27B (different
  template) → a 3.5 pack is NOT valid for 3.6, and vice versa. `run.sh` passes `--tokenizer "$MODEL_ID"`.
- **Assistant-only labels** via `return_assistant_tokens_mask=True` when the template has
  `{% generation %}`; the Qwen3.6 template has **no `{% generation %}`**, so labels fall back to
  `labels = input_ids` → trains on the FULL packed sequence (same as before; revisit if assistant-only
  masking is wanted).
- Source default: `Scicom-intl/Function-Call-TaaS` `glm5.1-fp8-test/test-00000-of-00001.parquet` (GATED
  — needs the HF token; on the box it's at `/share/huggingface/token`, org `Scicom-intl` authorized).
- Bins longer than `--max-seq-len` are DROPPED (never split). Output: `./packed_data` ChiniDataset.

## Merge + inference (`merge_infer.py`)

Folds the custom LoRA back into the base and generates — the sanity check that the finetune didn't
collapse the model. Merge is `W += (alpha/r)·(B@A)` per wrapped Linear; it strips the
`._checkpoint_wrapped_module` segment from the saved keys to map onto the clean base weight names,
resolves the dense/MoE base class from the config, and re-applies the GatedDeltaNet contiguous-v patch
(needed for the generate() prefill step). Needs **`accelerate`** (for `device_map="auto"`; now in
`run.sh`). A 27B/35B model fits on ONE 144 GB H20.

```bash
python merge_infer.py --lora checkpointing-qwen3.6-27b/lora.pt --max-new-tokens 256
python merge_infer.py --lora checkpointing-qwen3.6-35b-a3b/lora.pt --no-merge   # base, for an A/B check
```

⚠️ Generation opens a `<think>` block (the model was trained with all-turn reasoning via
`preserve_thinking=True`), so short `--max-new-tokens` shows only chain-of-thought; give it room
(≥256) to reach the final answer.

## ⚠️ Still owed: `compare_logits.py`

Per `../CLAUDE.md`: the custom forward (`resolve_arch`/`make_custom_cls` + `LinearLoRA` B=0 + the
`chunk_gated_delta_rule` contiguous patch) should be gated by a `compare_logits.py` asserting the no-op
customization reproduces stock logits (argmax match, cosine ~1). **Not written.** The practical gate
used instead is the **overfit smoke** (loss must drop, no NaN/collapse) + the post-merge generation
coherence check — both passed for both models (see Status).

## Run recipe — the tm 8× H20-3e box (port 1023)

See `../CLAUDE.md` "tm H20 VM" for box access/etiquette. **Everything under `/share`** (small `/`):
job dir `/share/autotrain-qwen3.5`, venv `/share/qwen3.5-venv`, **`HF_HOME=/share/huggingface`**.

```bash
ssh -i ../../scicom -p 1023 root@8.222.165.68      # the scicom key = GPUPlatform/scicom (mode 600)

# stage the job in — scp is FLAKY on this box (intermittent sshd); pipe via stdin instead:
for f in *.py *.sh; do ssh -i ../../scicom -p 1023 root@8.222.165.68 "cat > /share/autotrain-qwen3.5/$f" < "$f"; done

# on the box — HF token is already stored; export it for the gated dataset pack:
cd /share/autotrain-qwen3.5
export HF_TOKEN=$(cat /share/huggingface/token) HF_HOME=/share/huggingface HF_HUB_DISABLE_XET=1
VENV_PATH=/share/qwen3.5-venv QWEN_DEPS_ONLY=1 bash run.sh   # deps only first (already built)

# full run — dense (default) on 4 GPUs; MoE on another 4 (distinct --master_port if concurrent!):
CUDA_VISIBLE_DEVICES=0,1,2,3 VENV_PATH=/share/qwen3.5-venv \
  nohup bash run.sh --lr 1e-4 --max_epochs 3 > train27.log 2>&1 &
MODEL_ID=Qwen/Qwen3.6-35B-A3B CUDA_VISIBLE_DEVICES=4,5,6,7 VENV_PATH=/share/qwen3.5-venv \
  nohup bash run.sh --lr 1e-4 --max_epochs 3 > train35.log 2>&1 &
```

Helper scripts on the box (bypass `run.sh`'s deps/pack when they're already done, drive torchrun
directly): **`smoke_model.sh`** (overfit smoke) and **`train_full.sh`** (full run) — both take env
`MODEL_ID` / `GPUS` / `CKPT` / `MASTER_PORT` / `LR` / `EPOCHS`. **`pack36.sh`** repacks with the 3.6
tokenizer. Run them detached in **tmux** (installed; `nohup setsid` over ssh leaves the channel hanging).

`run.sh` knobs (env): `MODEL_ID` (dense/MoE; default `Qwen/Qwen3.6-27B`), `CHECKPOINT_DIR` (derived
per-model), `CUDA_VISIBLE_DEVICES`, `VENV_PATH`, `MAX_SEQ_LEN` (pack length, default 50000),
`SKIP_DATA_PACK=1`, `SKIP_MODEL_DOWNLOAD=1`, `QWEN_DEPS_ONLY=1`. Args after the flags pass to
`qwen3_5.py` (`--lr --max_epochs --max_steps --limit_samples --rank --alpha --batch_size
--checkpointing_step --model_id --checkpoint_dir --wandb --mlflow`).

### Gotchas (confirmed on the real runs, 2026-07-01)

- **Concurrent torchrun jobs need distinct `--master_port`** — both default to 29500 →
  `EADDRINUSE`. The helper scripts take `MASTER_PORT` (used 29500 + 29520 to run dense + MoE in
  parallel on 0–3 / 4–7).
- **`causal_conv1d` cu13 (load-bearing):** under uv's default build isolation it builds against the
  default cu12 torch → `causal_conv1d_cuda.so` links `libcudart.so.12` → ImportError against our cu130
  torch. Fix = build **`--no-build-isolation`** (in `run.sh`). Needs `nvcc` (`/usr/local/cuda` = 13.0;
  `run.sh` exports CUDA_HOME). Full re-fix combo: `--no-build-isolation --no-cache --reinstall --no-deps`.
- **scp is flaky** on this box (intermittent sshd); transfer files by piping to `cat >` over ssh.
- **HF token** lives at `/share/huggingface/token` (`hf auth whoami` = `huseinzolkepliscicom`, orgs
  incl. `Scicom-intl` → authorized for the gated source dataset). No `~/.netrc` / `WANDB_API_KEY` on
  the box, so the runs logged loss to the tmux log, not wandb.

## Status

**2026-07-01 — BOTH Qwen3.6 models trained end-to-end on the tm box (port 1023), no errors.**
Deps/venv/model-cache were already prepped (see 2026-06-24 note); this session refactored the trainer
to dense+MoE, repacked for 3.6, smoked both, and ran the full finetunes.

- **Env** (unchanged, verified): torch **2.10.0+cu130**, transformers **5.12.1** (ships both `qwen3_5`
  and `qwen3_5_moe`), flash_qla, causal_conv1d (cu130), liger (`apply_liger_kernel_to_qwen3_5[_moe]`),
  chinidataset. Both models pre-cached in `/share/huggingface` (Qwen3.6-27B ~52 GB, Qwen3.6-35B-A3B).
- **Dataset repacked** with the Qwen3.6 tokenizer + `preserve_thinking=True`: **67 bins** from 77 rows
  (6 dropped >50k), mean **35.2k tok/bin**, 70.4% packing efficiency. (`packed_data`; the old 3.5 pack
  was moved to `packed_data_qwen3.5_old`.)
- **Smokes** (4× H20, overfit 8 bins, `--max_steps 15 --limit_samples 8 --max_epochs 20`): both PASS,
  loss drops, no OOM/NaN, LoRA saved.
  - Qwen3.6-27B: loss step0 **7.16 → 0.34**; ~2.5k tok/s; 128 LoRA tensors.
  - Qwen3.6-35B-A3B: loss step0 **3.74 → 0.37**; ~11k tok/s; 80 LoRA tensors; 33.48B params / **52.5M**
    trainable — proves `resolve_arch` picked the MoE classes and LoRA hit only attention.
- **Full runs** (4× H20 each, in parallel, `--lr 1e-4 --max_epochs 3` = 51 steps over 67 bins):
  | model | GPUs | loss step0→step50 | tok/s (warm) | wall | LoRA (`lora.pt`) |
  |-------|------|-------------------|--------------|------|------------------|
  | **Qwen3.6-27B** (dense) | 0–3 @ port 29500 | **4.40 → 0.25** | ~2.4k | ~50 min | `checkpointing-qwen3.6-27b/` 335 MB |
  | **Qwen3.6-35B-A3B** (MoE) | 4–7 @ port 29520 | **2.67 → 0.28** | ~10–12k | ~13 min | `checkpointing-qwen3.6-35b-a3b/` 110 MB |
  - Memory: dense ~38–63 GB/GPU, MoE ~18–22 GB/GPU (both huge headroom on the 144 GB H20s). MoE is
    ~4× faster/step (A3B = only ~3B active params per token). Both loss curves settle ~0.25–0.45 (small
    dataset, overfit-ish); clean NCCL shutdown, no collapse.
  - `lora_meta.json` records `model_id` + `is_moe` + `r/alpha/scaling` + target modules for merge.

- **Merge + inference verified** (`merge_infer.py`, 2026-07-01): both checkpoints merge cleanly
  (**64/64** dense adapters, **40/40** MoE) and generate **coherent, on-topic text** — no collapse
  (contrast the earlier gemma4 overfit-collapse). Each base fits on one 144 GB H20. Needed
  `accelerate` (added to `run.sh`). Output opens with the `<think>` reasoning block (all-turn reasoning
  was trained), so use `--max-new-tokens ≥256` to see the final answer.

**STILL TO DO:** `compare_logits.py` (the owed logits gate) is not yet written. No wandb (no key on the
box) — loss is in the tmux logs.
