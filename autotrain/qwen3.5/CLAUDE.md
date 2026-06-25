# autotrain — Qwen3.5-27B LoRA finetune (FSDP2, tm 8× H20-3e)

Standalone training job (NOT part of the gateway). Finetunes `Qwen/Qwen3.5-27B` with a custom
LoRA on a packed function-calling dataset, using FSDP2 (`fully_shard`) + CPU offload across N GPUs.
Sibling of `../gemma4`, `../minimax-m2`, `../mistral-small` — same multipack + custom-LoRA shape.

```
qwen3_5.py        training entrypoint (torchrun, FSDP2, custom LinearLoRA, packed varlen, Liger FLCE loss)
pack_dataset.py   render Qwen3.5 chat template + multipack a HF chat-parquet → ChiniDataset
run.sh            pod/box bootstrap: venv + deps + dataset pack + torchrun
CLAUDE.md         this file
README.md         short overview
.gitignore        packed_data*, .venv, checkpointing*, memory pickles
```

The original single-file `one_script_run.sh` (heredoc'd both python files + the install/run steps)
was split into these files to match the `../gemma4` layout. `one_script_run.sh` is kept for history.

## What's different about Qwen3.5: the GatedDeltaNet hybrid attention

Qwen3.5-27B is a **hybrid** model — most layers are **GatedDeltaNet linear attention** (O(S) state
recurrence, not softmax attention), interleaved with a few **full-attention** layers. It is also a
multimodal `Qwen3_5ForConditionalGeneration` (vision blocks present; we train text-only).

- **Linear-attention layers** run the **FlashQLA** `chunk_gated_delta_rule` kernel
  (`git+https://github.com/QwenLM/FlashQLA.git`) + `causal_conv1d` for the short conv. `qwen3_5.py`
  monkeypatches each `Qwen3_5GatedDeltaNet.chunk_gated_delta_rule` to **`.contiguous()` the `v`**
  tensor first — the TileLang kernel requires `stride[-1] == 1` or it errors.
- **Full-attention layers** use `attn_implementation="kernels-community/flash-attn3"`, auto-fetched
  at model load by the `kernels` package (pinned `kernels<=0.14.0`). No FA3 wheel, no torch-2.12 pin
  (unlike gemma4 — its head_dim-512 layers forced the prebuilt FA3 wheel; Qwen3.5's full layers are
  head_dim ≤ 256 so the kernels-community FA3 just works on torch 2.10).

The collator carries packing as `cu_seq_lens_q/k` + `position_ids` (no dense `(1,S,S)` mask), so both
the linear and full layers see per-document boundaries.

## Training design (`qwen3_5.py`) — same shape as the gemma4/minimax siblings

- **Custom `LinearLoRA`** (not PEFT): `y = Wx + (alpha/r)·B(Ax)`, wraps `q/k/v/o_proj` (vision skipped),
  `B` zero-init (→ no-op at init), bf16. `r=256`, `alpha=512` (`scaling=2.0`) by default.
- **FSDP2** `fully_shard` per `Qwen3_5DecoderLayer` + `Qwen3_5VisionBlock`, then the root, under
  **`CPUOffloadPolicy()`** (param shards live on CPU) + `MixedPrecisionPolicy(bf16 param / fp32 reduce)`.
- **`init_process_group(backend="cpu:gloo,cuda:nccl")`** — required: the checkpoint's `full_tensor()`
  all-gather of CPU-resident DTensor shards is a CPU collective NCCL can't service.
- **Activation checkpointing** (`NO_REENTRANT`) per `Qwen3_5DecoderLayer`.
- **Loss**: `LigerFusedLinearCrossEntropyLoss` (skips materializing the lm_head logits) — the wrapper
  forward returns only `{"loss"}`.
- **Optimizer**: AdamW over LoRA params only, `fused=False` (fused AdamW is CUDA-only; CPU-offload
  runs the step on CPU-resident params).
- **Checkpoint**: saves only the LoRA adapters (`checkpointing/lora.pt` + `lora_meta.json`), never the
  full 27B.
- `torch.cuda.set_device(rank)` before NCCL init (else every rank lands on cuda:0).
- `--batch_size 1` (the collator `np.concatenate`s the whole DataLoader batch into ONE packed
  sequence; >1 multiplies the per-step sequence length).

⚠️ **Inherited-from-gemma4 bits to watch** (gemma4 was the template; confirm on the first run):
the collator emits `mm_token_type_ids` (all-zero, text-only) and the wrapper forwards it to
`self.model`. gemma4 *requires* this; Qwen3.5 may not accept the kwarg — if `from_pretrained`/forward
raises `unexpected keyword argument 'mm_token_type_ids'`, drop it from both the collator and the
wrapper's `self.model(...)` call.

## Dataset packing (`pack_dataset.py`) — Qwen3.5 chat template

Identical multipack to the siblings, with Qwen3.5-specific rendering:
- **Reasoning-guard relaxation** (default ON): Qwen3.5's stock template only emits `<think>…</think>`
  on assistant turns AFTER `ns.last_query_index`, so intermediate agentic turns lose their reasoning.
  We swap `{%- if loop.index0 > ns.last_query_index %}` → `{%- if loop.index0 >= 0 %}` so EVERY model
  turn's reasoning trains. `--native-reasoning` keeps the stock template. The swap is defensive: a
  missing guard string warns + falls back to stock (never crashes).
- **GLM → Qwen3.5 normalization**: `role: observation`→`tool`, `content: null`→`""`, bare
  `tool_calls`/`functions` wrapped to OpenAI `{type:function, function:{…}}` with **dict** (not
  JSON-string) `arguments` — the template iterates `tool_call.function.arguments | items`.
- **Assistant-only labels** via `return_assistant_tokens_mask=True` when the template has
  `{% generation %}`; else falls back to `labels = input_ids`.
- Source default: `Scicom-intl/Function-Call-TaaS` `glm5.1-fp8-test/test-00000-of-00001.parquet`.
- Bins longer than `--max-seq-len` are DROPPED (never split). Output: `./packed_data` ChiniDataset.
- Reads `HF_TOKEN` from the gateway `.env` if present, else ambient (graceful if the file is missing,
  e.g. on the tm box where you `export HF_TOKEN`).

## ⚠️ Owed: `compare_logits.py` before any expensive run

Per `../CLAUDE.md` ("Always compare logits when adding/altering a custom implementation"): the custom
forward (`CustomQwen3_5ForConditionalGeneration` + `LinearLoRA` B=0 + the `chunk_gated_delta_rule`
contiguous patch) should be gated by a `compare_logits.py` that asserts the no-op customization
reproduces stock Qwen3.5 logits (argmax match, cosine ~1) before training. **Not yet written.** Until
it exists, the practical gate is the smoke run below (loss must decrease, no NaN/collapse).

## Run recipe — the tm 8× H20-3e box (port 1023)

See `../CLAUDE.md` "tm H20 VM" for box access/etiquette (shared box; use only the GPUs you're given;
everything under `/share`). This job was developed against the **fresh box at port 1023**
(`dsw-464391…`), NOT the port-1024 box.

```bash
ssh -i ../../scicom -p 1023 root@8.222.165.68      # the scicom key = GPUPlatform/scicom (mode 600)

# stage the job in
scp -i ../../scicom -P 1023 *.py *.sh root@8.222.165.68:/share/autotrain-qwen3.5/

# on the box — venv under /share, deps + pack + train via run.sh
cd /share/autotrain-qwen3.5
export HF_TOKEN=hf_...                              # gated dataset (from ../.env)
export HF_HOME=/share/huggingface HF_HUB_DISABLE_XET=1
VENV_PATH=/share/qwen3.5-venv QWEN_DEPS_ONLY=1 bash run.sh    # deps only first (FlashQLA/causal_conv1d build)
# pick the GPUs you're given (the box is shared; check `nvidia-smi` first):
CUDA_VISIBLE_DEVICES=0,1,2,3 VENV_PATH=/share/qwen3.5-venv \
  bash run.sh --max_steps 50 --limit_samples 10 --lr 1e-4    # smoke -> a LoRA checkpoint
# full run (detach — survives ssh drop):
CUDA_VISIBLE_DEVICES=0,1,2,3 VENV_PATH=/share/qwen3.5-venv \
  nohup bash run.sh --lr 1e-4 --max_epochs 3 --wandb > train.log 2>&1 &
```

`run.sh` knobs (env): `CUDA_VISIBLE_DEVICES` (which GPUs), `VENV_PATH`, `MAX_SEQ_LEN` (pack length,
default 50000), `SKIP_DATA_PACK=1` (reuse `./packed_data`), `QWEN_DEPS_ONLY=1`. Training args after
`--` pass straight to `qwen3_5.py` (`--lr --max_epochs --max_steps --limit_samples --rank --alpha
--batch_size --checkpointing_step --wandb --mlflow`).

### Bootstrap gotchas (anticipated; confirm on first run)

- **`causal_conv1d` + FlashQLA build from source** → need a CUDA toolkit (`nvcc`) matching torch's
  CUDA on the box. If the build fails, that's the first thing to check.
- **Qwen3.5-27B is NOT in the tm HF cache** (only gemma-4-31B + GLM-5.2-FP8 were, 2026-06-24) → first
  run downloads ~54 GB over tm's slow uplink (~20 MB/s). `HF_HUB_DISABLE_XET=1` is set.
- **`transformers` must be new enough** to ship `Qwen3_5ForConditionalGeneration` /
  `transformers.models.qwen3_5` and `liger_kernel` must ship `apply_liger_kernel_to_qwen3_5`.

## Status

**2026-06-24** — split `one_script_run.sh` → standardized files; prepped + verified the env on the tm
H20 box (port 1023), `/share/qwen3.5-venv`. Everything GPU-independent is **done and verified**; the
training launch was blocked only by the box's intermittently-unreachable sshd (see below).

Verified on the box:
- **Deps build works** (torch **2.10.0+cu130**, transformers **5.12.1** ships Qwen3.5, flash_qla
  0.1.0 pure-python, chinidataset 0.2.0). **`causal_conv1d` cu13 gotcha (load-bearing):** under uv's
  default build isolation it builds against the default **cu12** torch → `causal_conv1d_cuda.so` links
  `libcudart.so.12` → `ImportError` at runtime against our cu130 torch. Fix = build it
  **`--no-build-isolation`** so it uses the venv's cu130 torch (now in `run.sh`). If a poisoned uv
  wheel cache forces a reinstall, the full combo is `--no-build-isolation --no-cache --reinstall
  --no-deps` (dropping `--no-deps` pulls cu12 torch back in). After the fix the `.so` links
  `libcudart.so.13`. Needs `nvcc` on PATH (`/usr/local/cuda` = cuda-13.0; `run.sh` exports CUDA_HOME).
- **Forward is compatible** (no GPU needed to check): `Qwen3_5Model.forward` (the `.model` attr)
  accepts `mm_token_type_ids` AND `**kwargs`, so the gemma4-inherited collator kwargs + `cu_seq_lens_*`
  pass through. The "copied blind" risk is cleared at the signature level (runtime still unproven).
- **Model cached** (52 GB at `/share/huggingface`). **Dataset packed**: 67 bins from 77 rows (6 dropped
  >50k), mean 35.2k tok/bin. ⚠️ The Qwen3.5 template has **no `{% generation %}`**, so labels fall back
  to `labels = input_ids` (trains on the FULL packed sequence, not assistant-only — same as the
  original script; revisit if assistant-only masking is wanted).

**tm box (port 1023) operational notes (2026-06-24):** shared/multi-tenant + a 7-GPU gemma4 autotrain
run hammered CPU/IO → **sshd intermittently unreachable** for long stretches (connect timeouts /
`kex_exchange_identification: Connection closed`). Detached launches: **`tmux` was NOT installed**
(`apt-get install -y tmux` fixed it) — use it, since `nohup setsid …` over ssh leaves the ssh channel
hanging (looks like a failed launch but the process started → easy to spawn duplicates; always make the
launcher idempotent: `pkill -f qwen3_5.py; tmux kill-session -t smoke` first). GPUs 0–3 freed once the
gemma4 run finished (box load 7→0.1).

**STILL TO DO (when the box is reachable):** land the smoke (`smoke.sh` on the box, or the snippet in
the run recipe above), confirm loss decreases / no OOM at 50k on 4× H20, then the full
`--lr 1e-4 --max_epochs 3 --wandb` run; record loss curve + tok/s here. Also still **owed**:
`compare_logits.py` (see above). A `merge_infer.py` for the custom LoRA is not yet ported either.
