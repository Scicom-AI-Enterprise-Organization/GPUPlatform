# Gemma-4 31B LoRA finetuning

LoRA finetune of **`google/gemma-4-31B-it`** on a packed dataset, sharded across **2× H100 SXM**
with PyTorch **FSDP2** (`fully_shard`) + activation checkpointing + CPU offload. Designed to run
on a fresh RunPod pod via [`runpodctl`](https://github.com/runpod/runpodctl).

The interesting part is the **custom per-layer attention**: Gemma-4 has full-attention layers
with `head_dim=512` and sliding-window layers with `head_dim=256`. FlashAttention-3 only supports
`head_dim ≤ 256`, so the run **dispatches per layer** — SDPA for the 512-dim full-attention
layers, FA3 varlen for the 256-dim sliding layers — over a packed (varlen) sequence.

## Status — what's done & verified

Bugs found and fixed (the original script didn't run; these are the fixes):

1. **Syntax** — `mm_token_type_ids='********'` (stray string, no comma) → compile error. Fixed.
2. **SDPA mask was a `float` 0/1 mask** → SDPA treats float `attn_mask` as an *additive bias*, so it
   masked **nothing** (attention leaked across docs + into the future). Fixed to a **boolean**
   block-diagonal causal mask. (`test_attention.py`, exact to ~1e-7.)
3. **Attention `scaling` was dropped** — transformers calls the attention interface with `scaling=`
   (gemma-4 uses `scaling=1.0`), but `dynamic_attention` named the param `scale`, so `scaling=1.0`
   fell into `**kwargs` and SDPA/FA3 used `1/√512` → wrong attention everywhere (incl. training).
   Fixed: param renamed to `scaling`. **Verified end-to-end by `compare_logits.py`**: dynamic vs
   default attention → identical next-token argmax + top-5, cosine 0.998 (was a total mismatch before).
4. **LoRA `lora_b` not zero-initialised** — the adapter started as a large *random* perturbation on
   every q/k/v/o (× scaling), so the model trained from a corrupted state and merged to garbage
   (collapse to `<|"|>`). Fixed: `nn.init.zeros_(lora_b.weight)` (adapter = no-op at step 0).
5. **FA3 cu_seqlens path** hardened (int32 on-device cu_seqlens, `window_size` guard, tuple-return,
   GQA). **PEFT can't replace the custom LoRA** — gemma-4 q/k/v/o are `Gemma4ClippableLinear`, which
   PEFT rejects; the custom `LinearLoRA` wraps the real module (preserving clipping).
6. **Distributed/infra** — `torch.cuda.set_device` per rank; `init_process_group("cpu:gloo,cuda:nccl")`
   (CPU-offloaded checkpoint all-gather needs gloo); `NCCL_NVLS_ENABLE=0` (RunPod multicast bind fails);
   LoRA-only optimizer with `fused=False`; LoRA-only checkpoints.

Verified on real H100s: `test_attention.py` (SDPA 1.9e-6, FA3 9.4e-3), `compare_logits.py` (dynamic ==
default), training loss decreasing + wandb logging, `merge_infer.py` merge (230/230) + coherent
generation. Base function-calling eval (`eval_funccall.py`, SyntheticGen scoring on `glm5.1-fp8-test`):
`tool_call_f1 ≈ 0.715`.

**Context parallelism — zigzag ring attention (`ring_zigzag_attn.py`), verified 2× H100 SXM
2026-07-04.** Shards one long packed sequence across GPUs and rings the K/V blocks, combining the
per-block partials via the online-softmax LSE the FA4 forward returns (`return_lse=True`, natural-log
with the scale folded in). The head_dim>256 backward can't take an external LSE (the fork's
`_flash_attn_bwd_large_headdim` re-softmaxes each block *locally*), so the ring backward recomputes
each partial block with `p = exp(scale·S − LSE_global)` — the same math as the fork's backward, just
consuming the global LSE. `torchrun --nproc_per_node=2 test_ring_attn.py`: the ring (zigzag layout,
2 GPUs) **output AND dq/dk/dv match single-GPU non-ring FA4 to rel ~3e-3 and an fp32 ground truth to
~3e-3** for gemma4-global hd512, sliding-geom hd256, multi-doc varlen, a tiny config, AND the
**sliding-window** configs (position-aware ring) — `ALL RING CONFIGS PASS ✅`. The hybrid: hd512-global
→ fused cute zigzag ring; hd256-sliding → position-aware ring (a fused kernel's window is wrong under
zigzag). Wired into training via `context_parallel.py` + `gemma4.py --cp_size` + the `/autotrain/new`
Context-parallelism toggle (gemma4-only); the real Gemma4 backbone under CP matches non-CP hidden
states (2× H100, tiny random model). A full gemma-4-31B CP training run is wired + ready but not yet
run end-to-end. See `CLAUDE.md` for the design, the interface facts, and the non-contiguous-recv bug.

⚠️ **Open constraint — long-context training is memory-bound by the head_dim-512 full-attention.** It
has *no* memory-efficient kernel in PyTorch (flash/efficient/cuDNN all reject head_dim 512; flex
fails to compile), so it runs SDPA-math = `O(S²)` score matrix, which FSDP2 **cannot shard** (it's an
activation, not a parameter). The Function-Call conversations are 19k–50k tokens (tool-schema heavy),
so: ~8k packed bins train fine; 22–24k OOMs on 2× H100 (~44–60 GB score); 32k needs ~74 GB. Levers:
more GPUs (4× lowers the model shard, not the score matrix), trim the tool schemas at pack time, or a
chunked / FlexAttention full-attention kernel. **Do NOT truncate bins in training** — control length
at pack time (`pack_dataset.py --max-seq-len`); the bin must still contain the conversation turns
(the first user turn is ~25k tokens in, after the schemas).

## Files

| file | what |
|------|------|
| `gemma4.py` | training entrypoint — run with `torchrun` |
| `attention.py` | `dynamic_attention`: the SDPA / FA3 per-layer dispatch (torch-only) |
| `test_attention.py` | unit test for both attention paths vs a per-document causal reference |
| `compare_logits.py` | end-to-end check: `dynamic_attention` logits == default-attention logits |
| `merge_infer.py` | merge the custom LoRA into the base model and generate |
| `pack_dataset.py` | build the multipacked ChiniDataset from a chat parquet (tools + reasoning) |
| `eval_funccall.py` | function-calling accuracy eval (SyntheticGen scoring) via a vLLM server |
| `bench_attention.py` | latency benchmark: FA4 cute (head_dim-512 fork) vs vLLM Triton attention |
| `decode_attention.py` | **purpose-built Triton flash-decode kernel** — fastest decode (beats FA4 + vLLM) |
| `bench_decode_kernel.py` | 3-way decode comparison: custom Triton vs vLLM vs FA4-cute |
| `bench_decode_cudagraph.py` | eager-vs-CUDA-graph decode micro-bench (short-context analysis) |
| `bench_setup.sh` + `bench_vllm_shim/` | copies vLLM's Triton kernel into a minimal shim (no full vLLM build) |
| `ring_zigzag_attn.py` | **zigzag ring (context-parallel) attention**: fused cute-512 ring + position-aware sliding ring |
| `context_parallel.py` | CP trainer glue: CP process group, zigzag `shard_batch`, `cp_ring_attention` backend |
| `test_ring_attn.py` | 2-GPU test: ring out+dq/dk/dv == non-ring FA4 == fp32 (causal + sliding) |
| `run_ring_test.sh` | pod bootstrap for the ring test (fork + cutlass-dsl [+ FA3 via GEMMA_FA3=1]; no model needed) |
| `run.sh` | one-shot pod bootstrap: deps → correctness test → download → train |
| `CLAUDE.md` | design notes, gotchas, the full runpodctl workflow |

## Quickstart (on a 2× H100 pod)

```bash
cd /workspace/autotrain
export HF_TOKEN=hf_...                       # gated: accept the gemma-4 license on HF first

# 1) install deps + run the attention correctness test only (cheap smoke test)
GEMMA_DEPS_ONLY=1 bash run.sh

# 2) full bootstrap + training (downloads model + dataset, then torchrun on all GPUs)
bash run.sh

#    ...or a short run that produces a LoRA checkpoint fast (for merge/inference validation):
bash run.sh -- --max_steps 120

# 3) merge the adapters back into the base model and generate
python merge_infer.py --prompt "Terangkan konsep AI." --max-new-tokens 64
python merge_infer.py --merged-out ./gemma4-merged    # also save the merged weights
```

`run.sh` is idempotent — re-running it re-checks deps and skips what's already installed.

## What `run.sh` does

1. Picks the CUDA backend from the **host driver** (`nvidia-smi`): ≥13.0 → `cu130`, ≥12.6 → `cu126`.
2. Installs **torch 2.12** for that backend — the FA3 wheel is built for torch 2.12 (this is a
   hard requirement, see below).
3. Installs the prebuilt **FlashAttention-3** wheel, `transformers==5.5.0`, `liger-kernel`,
   `kernels==0.14.1`, `ChiniDataset`, `huggingface_hub`.
4. Runs `test_attention.py` as a **pre-flight** (don't pay for a GPU to discover a broken mask).
5. Downloads `google/gemma-4-31B-it` + the `huseinzolkepliscicom/gemma4-multipack` dataset **once**
   (so the two torchrun ranks read a shared cache instead of racing the download). Override the
   dataset with `DATA_ID=<repo> bash run.sh`.
6. `torchrun --nproc_per_node=<#gpus> gemma4.py`.

### CLI flags (`gemma4.py`)

| flag | default | meaning |
|------|---------|---------|
| `--r` | 256 | LoRA rank |
| `--alpha` | 512 | LoRA alpha (scaling = alpha/r) |
| `--batch_size` | 1 | packed bins concatenated per step |
| `--max_epochs` | 1 | epochs over the dataset (set high to overfit a small set) |
| `--lr` | 1e-4 | AdamW learning rate |
| `--max_steps` | 0 | hard stop after N steps (0 = run all epochs); for quick checkpoints |
| `--checkpointing_step` | 100 | save LoRA adapters every N steps |
| `--limit_samples` | 0 | cap the dataset to the first N bins (0 = all) |
| `--wandb` / `--wandb_project` | off | log loss/lr/tps to Weights & Biases |

Training consumes packed bins **as-is** (no in-training truncation — length is a `pack_dataset.py`
concern). Checkpoints (LoRA only) land in `checkpointing/lora.pt` + `checkpointing/lora_meta.json`.

## Dataset (multipacking)

`pack_dataset.py` builds the **multipacked ChiniDataset** that `gemma4.py` trains on, following the
bin-packing scheme of `gateway/.../tts/pack_stage1.py`. Each row's `messages` is rendered through the
`google/gemma-4-31B-it` chat template, tokenized, and whole conversations are greedily packed into
`--max-seq-len` (default **131072 = 128k**) token bins. Conversations are **never split**, so each
document's `cu_seqlens` / per-doc RoPE positions stay coherent — a single conversation longer than a
bin is dropped (loudly), so raise `--max-seq-len` rather than splitting.

- **Tools** — the `functions` column (bare function objects) is wrapped as OpenAI tools and passed via
  `apply_chat_template(tools=...)` → a `<|tool>…<tool|>` block in the system turn. (`--no-tools` to skip.)
- **Reasoning** — every assistant turn's `reasoning` is rendered as a `<|channel>thought…<channel|>`
  block. The stock gemma-4 template only keeps the last user turn's tool-call reasoning; `pack_dataset.py`
  relaxes that one guard so all of it is trained. Use `--native-reasoning` for stock behavior.
- **Labels** — `labels == input_ids` (train on the full packed sequence; the gemma-4 template has no
  `{% generation %}` block, so assistant-only masking isn't available and is auto-skipped).
- **Columns** consumed by `gemma4.py`: `input_ids`, `labels` (`int64[]`), `position_ids` (per-doc reset),
  `attention_mask` (per-doc lengths → cu_seqlens). Invariant: `sum(attention_mask) == len(input_ids)`.

```bash
# build locally -> ./packed_data (what gemma4.py reads via StreamingDataset)
python pack_dataset.py --out ./packed_data
# any chat parquet / context length / quick smoke test:
python pack_dataset.py --repo Scicom-intl/Function-Call-TaaS \
  --file glm5.1-fp8-test/test-00000-of-00001.parquet --max-seq-len 131072 --max-rows 10
```

The prebuilt dataset is published at **`huseinzolkepliscicom/gemma4-multipack`** (public HF dataset) —
`run.sh` downloads it by default. After re-packing, refresh it with:
`hf upload huseinzolkepliscicom/gemma4-multipack ./packed_data --repo-type dataset`.

## Attention correctness (the thing this run lives or dies on)

`dynamic_attention` must reproduce **per-document causal** attention over the packed sequence in
both branches. `test_attention.py` checks each against an independent per-document reference:

- **SDPA branch** (`head_dim=512`): the block-diagonal causal mask must be a **boolean** mask.
  SDPA reads a *float* `attn_mask` as an additive bias — a `0/1` float mask masks **nothing**.
  Verified to ~1e-7 (fp32).
- **FA3 branch** (`head_dim=256`): `flash_attn_varlen_func` with int32 on-device `cu_seqlens`,
  unbatched `(total_tokens, H, D)` layout, `window_size` guard. Verified to ~1e-2 (bf16).

```bash
python test_attention.py     # SDPA runs anywhere; FA3 needs a GPU + flash_attn_interface
```

`test_attention.py` checks the kernels in **isolation**; `compare_logits.py` checks the **full
model** — it builds the exact packed input (the trainer's collator dict) and asserts that
`dynamic_attention` gives the same last-token logits as the default attention. This is what caught
the `scaling` bug (the unit test passed because it called `dynamic_attention(scaling=...)` directly;
only the full-model path exercised transformers' `scaling=` kwarg):

```bash
python compare_logits.py     # loads the model twice (dynamic vs default), asserts argmax match
# PASS: argmax+top5 identical, cosine 0.998 (residual = bf16 noise from FA3+SDPA mix)
```

## Attention kernel benchmark — FA4 cute vs vLLM Triton (`bench_attention.py`)

gemma-4's `head_dim=512` global layers rule out FA2/FA3 (cap ≤ 256), so the two real options are
**FA4 cute** (the `flash-attention-512` fork — a *training* kernel: fwd+bwd, contiguous varlen) and
**vLLM's Triton `unified_attention`** (the only vLLM backend that serves head_dim 512; paged KV with
a 3D split-KV decode kernel). `bench_attention.py` times both on the real gemma-4-31B geometries —
**full** (hd 512, 32 q / 4 kv) and **sliding** (hd 256, 32 q / 16 kv, window 1024) — for **prefill**
(q=kv=L) and **decode** (q=1, kv=L) at L ∈ {1k…32k}. Measured on **one H20-3e (SM 9.0)**, bf16,
torch 2.12.1+cu130 / triton 3.7.1, median of 50 iters.

Latency in **ms** (median of 50 iters) — **FA4 vs vLLM Triton**, prefill (q=kv=L) and decode
(q=1, kv=L). For decode, FA4 is shown both no-split (naive) and with manual FlashDecoding (the fix
below); the faster FA4 decode is **bold**. Every shape: FA4↔Triton cosine ≥0.99999, both bit-match
an fp32 SDPA reference.

**full attention — head_dim 512, 32 q / 4 kv heads**

| ms | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|----|------|------|------|------|-------|-------|
| prefill — FA4              | 0.524 | 1.69  | 6.15  | 23.7  | 93.5  | 371   |
| prefill — Triton (vLLM)    | 0.766 | 2.82  | 10.7  | 42.0  | 166   | 662   |
| decode — FA4 FlashDecoding | 0.088 | 0.087 | 0.092 | 0.090 | **0.131** | **0.243** |
| decode — FA4 naive         | 0.184 | 0.297 | 0.528 | 0.965 | 1.843 | 3.596 |
| decode — Triton (vLLM)     | 0.082 | 0.082 | 0.087 | 0.087 | 0.140 | 0.263 |

**sliding attention — head_dim 256, 32 q / 16 kv heads, window 1024**

| ms | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|----|------|------|------|------|-------|-------|
| prefill — FA4              | 0.222 | 0.500 | 1.04  | 2.14  | 4.31  | 8.63  |
| prefill — Triton (vLLM)    | 0.637 | 1.75  | 4.00  | 8.48  | 17.5  | 35.4  |
| decode — FA4 FlashDecoding | 0.087 | 0.088 | 0.090 | 0.086 | 0.086 | 0.087 |
| decode — Triton (vLLM)     | 0.081 | 0.081 | 0.081 | 0.087 | 0.086 | 0.086 |

**Prefill — FA4 wins** (FA2/FA3 can't even do head_dim 512): ~**1.8×** Triton on full hd512 (95 vs
53 TFLOP/s at 32k), ~**4×** on sliding hd256 (125 vs 31). gemma4 trains with FA4
(`GEMMA_ATTN=fa4_attention`, prefill-heavy long context).

**Decode — FA4 now beats Triton at ≥16k.** FA4's varlen forward had no usable split-KV on Hopper
(its native FlashDecoding asserts out on SM 9.0 — Blackwell-only), so decode grew ~linearly with
context (full-hd512 naive: **3.6 ms at 32k**, ~14× Triton). Fixed with **manual FlashDecoding**
(`run_fa4_decode_split`, no kernel change): split the KV into N chunks run as one varlen batch
(`return_lse=True`), combined with a **fused Triton kernel**; with precomputed cu_seqlens and a
zero-copy broadcast query, the per-token path is just kernel + combine. That drops full-hd512 decode
to **0.243 ms at 32k (14.8× the naive)**, ~flat across context — **0.92× Triton at 32k, 0.94× at
16k**, and ~1.03–1.13× below at ≤8k (both ~0.09 ms; there Triton's decode-specialized kernel does
genuinely less GPU work for tiny KV — confirmed by CUDA-graphing both, which strips launch overhead
and leaves Triton ahead at short L). Everything stays bit-exact (cosine ≥0.99999, fp32-SDPA match).

**Decode winner — `decode_attention.py` (custom Triton flash-decode).** The cute kernel can't win short
decode (WGMMA forces a 64-row tile for GQA-8's 8 query rows), so the fix is a separate small-M kernel,
not a cute-fork change. This ~80-line Triton split-KV flash-decode (BLOCK_M=16, contiguous KV) **beats
both FA4-cute and vLLM's general Triton at every context length**, bit-exact:

| geom | L | **custom** | vLLM | FA4-cute | custom/vLLM |
|------|---|-----------|------|----------|-------------|
| full hd512    | 1024  | **0.038 ms** | 0.085 | 0.093 | **0.45×** |
| full hd512    | 32768 | **0.195 ms** | 0.263 | 0.244 | 0.74× |
| sliding hd256 | any   | **~0.038 ms** | ~0.088 | ~0.094 | **~0.44×** |

i.e. ~2× faster than vLLM at short context (2.3× on sliding), 1.35× at 32k. Use `decode_attention`
for decode; FA4-cute stays the prefill/training path. Full details, the CUDA-graph short-context
analysis + the run recipe (and the tiny `bench_vllm_shim/` that runs vLLM's exact kernel without a full
vLLM build) are in `CLAUDE.md`.

```bash
VLLM_REPO=/path/to/vllm bash bench_setup.sh          # copy vLLM's Triton kernel into the shim
CUDA_VISIBLE_DEVICES=7 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  python bench_attention.py --iters 50 --warmup 10 --out results.json
```

## FA4 forward-kernel speedup — m64n80 → m64n64 (2026-07-12)

The FA4 fork's head_dim-512 SM90 **forward** default was retiled from `tile_mn=(64,80)` + RS-replicated
QK to **`(64,64)` cooperative QK N-split + intra-wg overlap** (`flash-attention-512-cute` commit
`48ecadc`). The public `flash_attn_varlen_func` API is unchanged → a **drop-in speedup for
`fa4_attention`**, no trainer edits. Verified on **H100 SXM** (gemma-4-31B, cu130, torch 2.12):

| measurement | baseline `4c964fa` (m64n80) | latest `48ecadc` (m64n64) | speedup |
|---|---|---|---|
| forward kernel — hd512 32q/4kv @ 32k (`sweep_fwd.py`) | 375.4 TFLOP/s (93.7 ms) | 433.7 TFLOP/s (81.1 ms) | **+15.5%** |
| forward kernel — hd512 @ 8k | 378.3 TFLOP/s | 444.7 TFLOP/s | +17.6% |
| end-to-end 32k LoRA training (4× H100, FSDP2 + `--cpu_offload`) | 4178 tok/s | 5031 tok/s | **+20.4%** |

Numerically identical: `compare_logits_fa4.py` cosine **0.999848**, argmax+top5 match; training losses
track step-for-step. End-to-end exceeds the raw forward gain because activation checkpointing recomputes
the forward in backward (kernel runs twice/step). **Memory caveat:** gemma-4-31B *fwd+bwd* at 32k does
NOT fit unoffloaded on 2× or 4× H100 — the head_dim-512 **backward** (`_bwd_large_headdim_block`)
materialises score-sized fp32 tensors and FSDP can't shard activations; run with `--cpu_offload` (LoRA
offload stays compute-bound, ~100% util) or on 144 GB/GPU (H20). This is the FA4 *backward* wall,
distinct from the SDPA-path forward `O(S²)` wall noted above.

```bash
# isolated kernel A/B (both configs benched in one process, correctness-checked before timing):
CUDA_VISIBLE_DEVICES=0 python flash-attention-512-cute/dev512/sweep_fwd.py --s 32768 --b 1 --hq 32 --hkv 4 --causal 1
# training A/B: checkout each commit, FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=0 so the source recompiles
GEMMA_ATTN=fa4_attention torchrun --nproc_per_node=4 gemma4.py --batch_size 1 --cpu_offload --max_steps 16
```

## Evaluation (function-calling accuracy)

`eval_funccall.py` reuses **SyntheticGen's exact scoring** (`SyntheticGen/synthetic/
evaluate_function_calling.py`) — headline metric `tool_call_f1` — and runs the model through a
**vLLM** server (transformers SDPA OOMs on the ~22k-token eval prompts; vLLM's tensor-parallel kernels
handle head_dim 512 + long context). It builds the prompt via the gemma-4 chat template and parses
gemma's native `call:NAME{...}` output (converting the unquoted gemma args to JSON for the scorer).

```bash
# serve base (or a merged model) with vLLM, then:
python eval_funccall.py --vllm-url http://localhost:8000 --served-model gemma4 --out base_results.json
```

Base `google/gemma-4-31B-it` on `glm5.1-fp8-test`: **`tool_call_f1 ≈ 0.715`** (full 77 convs).

## Why torch is pinned to 2.12

The prebuilt FA3 wheel (`mjun0812/flash-attention-prebuild-wheels` v0.9.18) is compiled against
**torch 2.12** and only ships `cu126 / cu130 / cu132` builds. So the env *must* use torch 2.12,
and the CUDA build is chosen to match the host driver. `run.sh` handles this automatically and
force-installs torch even if the base image ships a different version.

## Deploy a pod from scratch

```bash
runpodctl config --apiKey "$RUNPOD_API_KEY"
runpodctl ssh add-key --key-file ~/.ssh/id_rsa.pub

runpodctl pod create --name gemma4-ft \
  --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 2 \
  --image runpod/pytorch:1.0.6-cu1300-torch291-ubuntu2404 \
  --container-disk-in-gb 260 --cloud-type SECURE \
  --min-cuda-version 13.0 --ssh --ports "22/tcp" \
  --terminate-after "$(date -u -d '+8 hours' +%Y-%m-%dT%H:%M:%SZ)"

runpodctl ssh info <pod-id>          # connection string (says "pod not ready" while booting)
scp -P <port> -i ~/.ssh/id_rsa *.py run.sh root@<ip>:/workspace/autotrain/
# ... run.sh on the pod ...
runpodctl pod delete <pod-id>        # ⚠ billing stops only here (~$6.58/hr for 2× H100 SXM)
```

## Troubleshooting

- **`dynamic_attention requires cu_seq_lens_q ... was None`** — transformers didn't propagate the
  packing metadata to the attention call. The collator emits `cu_seq_lens_q/k` + `max_length_q/k`;
  make sure they reach `model(**batch)` (they're passed as kwargs).
- **`No module named 'flash_attn_interface'`** — the FA3 wheel didn't install; check the host CUDA
  matches a shipped wheel (`cu126/cu130/cu132`) and that torch is 2.12.
- **403 / gated repo on download** — the `HF_TOKEN` account hasn't accepted the gemma-4 license.
- **CPU OOM during model load** — every rank loads the full 62GB model before sharding; needs
  ~`62GB × #gpus` system RAM (our 2-GPU pods have 503GB, 4-GPU have ~1TB).
- **CUDA OOM "Tried to allocate NN GiB" mid-step** — the head_dim-512 full-attention score matrix
  (`O(S²)`, unshardable). ~44 GB at 24k, ~74 GB at 32k. Pack shorter bins (`--max-seq-len`), add GPUs
  (4× lowers the model shard, not the score), trim schemas, or use chunked/Flex attention. ~8k is safe.
- **Generated garbage / repeats one token (`<|"|>`) / argmax ≠ default** — attention or LoRA bug:
  confirm `dynamic_attention`'s param is `scaling` (not `scale`) and `lora_b` is zero-init; run
  `compare_logits.py`. Also a sign of over-truncated training (model only saw the schema preamble).
- **vLLM serving a *merged* gemma-4 dir fails on `preprocessor_config.json`** — `save_pretrained`
  doesn't write the multimodal processor; build it from `processor_config.json`'s `image_processor`
  block. vLLM 0.23.0 also needs `prometheus-fastapi-instrumentator>=7` pinned, and `--enable-lora`
  does NOT support gemma-4, so serve a merged model.
- **No workers / pod idle-bills** — `runpodctl pod delete <id>`. Billing only stops on delete.
