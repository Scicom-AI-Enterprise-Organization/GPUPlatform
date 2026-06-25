# autotrain — Gemma-4 31B LoRA finetune (2× H100 SXM on RunPod)

Standalone training job (NOT part of the gateway). Finetunes `google/gemma-4-31B-it`
with a custom LoRA on a packed dataset, using FSDP2 (`fully_shard`) across 2 GPUs.

```
gemma4.py        training entrypoint (torchrun, FSDP2, custom LoRA, packed varlen)
attention.py     dynamic_attention — the per-layer SDPA/FA3 dispatch (torch-only, unit-tested)
test_attention.py  correctness test: SDPA mask + FA3 cu_seqlens vs a per-doc causal reference
merge_infer.py   fold LoRA adapters back into the base model and generate
run.sh           pod bootstrap: deps + FA3 wheel + correctness test + download + torchrun
.env             RUNPOD_API_KEY
```

## The whole point: dynamic per-layer attention (`attention.py`)

Gemma-4 mixes two attention layer types with **different head dims**, and FlashAttention-3
only supports `head_dim <= 256`:

| layer type      | head_dim | kernel                         | why                                  |
|-----------------|----------|--------------------------------|--------------------------------------|
| FULL attention  | **512**  | **SDPA** (math backend)        | FA3 can't do head_dim > 256          |
| SLIDING window  | **256**  | **FA3** `flash_attn_varlen_func` | fast, supports the sliding window    |

`dynamic_attention` is registered via `AttentionInterface.register("dynamic_attention", …)`
and branches on `query.shape[-1] <= 256`. Both branches must reproduce **per-document causal**
attention over the packed sequence (B is always 1; the collator concatenates every document
into one long sequence and records `cu_seq_lens_*`).

### Gotcha #1 — SDPA `attn_mask` must be BOOLEAN, not a 0/1 float (this was the bug)

`torch.nn.functional.scaled_dot_product_attention` treats a **float** `attn_mask` as an
**additive bias** and a **bool** mask as keep/drop. The original code built a `tril` block-
diagonal mask of `1.0`/`0.0` floats → it added `+1` to allowed scores and masked **nothing**
(attention leaked across documents and into the future). Fix: `.bool()` the block-diagonal
mask (`True` = attend). Confirmed by `test_attention.py` (`test_sdpa_float_mask_would_be_wrong`).

### Gotcha #2 — FA3 varlen needs the cu_seqlens path right

`flash_attn_varlen_func` wants **unbatched** `(total_tokens, num_head, head_dim)` and **int32
cu_seqlens on the GPU**. The function permutes `(B,H,S,D)->(S,H,D)`, casts cu_seqlens to
`int32`/device, guards `window_size=(-1,-1)` when `sliding_window is None`, and handles builds
that return `(out, lse)`. GQA works natively (q has more heads than k/v) — no `pack_gqa` flag.

### Memory-bounded full attention — query tiling (the long-context fix)

The head_dim-512 SDPA-math backend would materialize one `(H, S, S)` score (78 GiB at 32k → OOM,
the wall that capped training to ~20k). `attention.py` now **tiles the QUERY axis**
(`_packed_sdpa_full` + `SDPA_QUERY_BLOCK`, default 2048): process `block` queries at a time so the
live score is `(H, block, S)`, and wrap each block in `torch.utils.checkpoint` so the BACKWARD
recomputes one block at a time (peak ~one block, not O(S²)). This is **bit-exact** — each query row
still softmaxes over the *whole* key axis (ordinary softmax, **NO online/streaming softmax**, since
the *key* axis is never tiled; only key-tiling would force online softmax). Verified in
`test_attention.py::test_sdpa_query_block_fwd_bwd`: forward AND q/k/v `.grad` match the per-doc
reference to ~1e-6 (fp32), block-size invariant incl. the partial last block. Tune memory↔speed via
env `SDPA_QUERY_BLOCK` (0 / ≥S = single call = legacy). This is what lets the 32k pack train.

⚠ **The default `SDPA_QUERY_BLOCK=2048` still OOMs the longest ~32k bins on 2× H100** (verified
2026-06-19: trained fine through step 4 then OOM'd on backward of a 32k bin, needing ~7.5 GiB more
than fit, with ~15 GiB lost to allocator fragmentation). Two fixes, use **both**: set
**`SDPA_QUERY_BLOCK=1024`** (proven-good; 512 also works, ~2× more recompute = slower) AND
**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (reclaims the fragmentation). At 512 there was
~25 GiB headroom, so 1024 is the sweet spot. torchrun propagates both envs to the workers.

### Confirm both paths (always run before an expensive job)

```bash
python test_attention.py     # SDPA runs on CPU/GPU; FA3 needs a GPU + flash_attn_interface
```
SDPA matches the reference to ~1e-7 (fp32). FA3 matches to ~1e-2 (bf16). `run.sh` runs this
automatically as a pre-flight before downloading the model.

## Version pinning is load-bearing: FA3 wheel ⇒ torch 2.12 ⇒ CUDA backend

The prebuilt FA3 wheel (`mjun0812/flash-attention-prebuild-wheels` v0.9.18) is built for
**torch 2.12** and ships only **cu126 / cu130 / cu132** (no cu128). So:

- torch is pinned to **2.12.x** regardless of the base image's torch.
- The CUDA backend is chosen **from the host driver** at runtime (`run.sh` parses `nvidia-smi`):
  driver ≥13.2→cu132, ≥13.0→cu130, ≥12.6→cu126. The FA3 wheel name follows the same `$CU`.
- `torch-2.12.0+cu130` exists on the stable PyTorch index — verified.

This is why `run.sh` force-installs torch even on a `runpod/pytorch` image (whose newest torch
is 2.9.1): the wheel ABI must match.

## RunPod workflow (use `runpodctl` — https://github.com/runpod/runpodctl)

Install: `curl -sL …/releases/download/v2.4.0/runpodctl-linux-amd64 -o ~/.local/bin/runpodctl`.
Config from `.env`: `runpodctl config --apiKey "$RUNPOD_API_KEY"`.

```bash
# register your SSH key once (so --ssh authorizes you)
runpodctl ssh add-key --key-file ~/.ssh/id_rsa.pub

# create 2× H100 SXM, CUDA>=13 host, auto-terminate safety net
runpodctl pod create --name gemma4-ft \
  --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 2 \
  --image runpod/pytorch:1.0.6-cu1300-torch291-ubuntu2404 \
  --container-disk-in-gb 260 --cloud-type SECURE \
  --min-cuda-version 13.0 --ssh --ports "22/tcp" \
  --terminate-after "$(date -u -d '+8 hours' +%Y-%m-%dT%H:%M:%SZ)"

runpodctl pod list
runpodctl ssh info <pod-id>        # prints the ssh command (returns {"error":"pod not ready"} while booting)
runpodctl pod delete <pod-id>      # TERMINATE — billing stops only here
```

Gotchas, all real:
- **GPU id for H100 SXM = `NVIDIA H100 80GB HBM3`** (`runpodctl gpu list`).
- The **modern** `runpodctl pod create` (not the deprecated `create pod`) is the one with
  `--min-cuda-version`, `--gpu-id`, `--image`, `--cloud-type`, `--terminate-after`. The
  deprecated `create pod` uses `--gpuType/--imageName/--secureCloud/--cost` and a price ceiling.
- `--min-cuda-version 13.0` is how you guarantee a CUDA-13 host (needed for cu130). The
  deprecated `create pod` can't pin CUDA — you'd have to verify `nvidia-smi` and re-roll.
- Container disk must be big: model ~62GB + dataset + deps. We use 260GB; volume not needed.
- H100 SXM ×2 secure ≈ **$6.58/hr, real billing** — `runpodctl pod delete` when done.

### Run on the pod

```bash
# from your laptop: copy the job in (runpodctl send/receive, or scp via `runpodctl ssh info`)
scp -P <port> -i ~/.ssh/id_rsa *.py run.sh root@<ip>:/workspace/autotrain/

# on the pod
cd /workspace/autotrain
export HF_TOKEN=hf_...                 # gated google/gemma-4-31B-it
GEMMA_DEPS_ONLY=1 bash run.sh          # install + attention test only (cheap smoke)
bash run.sh -- --max_steps 120         # short run -> a LoRA checkpoint for merge/inference
bash run.sh                            # full epoch

# merge adapters back + generate
python merge_infer.py --prompt "..." --max-new-tokens 64
python merge_infer.py --merged-out ./gemma4-merged   # also persist the merged weights
```

## Other fixes applied to `gemma4.py` (vs the original failing script)

- **Syntax error**: `mm_token_type_ids='********'` (stray string, no comma) → `=mm_token_type_ids,`.
- **Distributed init**: added `torch.cuda.set_device(rank)` + un-commented `ddp_setup()`
  (NCCL PG) before `init_device_mesh` — otherwise every rank lands on cuda:0.
- **Batch not on GPU**: the training loop now moves the collated CPU batch to `cuda:{rank}`.
- **Optimizer**: only the trainable LoRA params, and `fused=False` (fused AdamW is CUDA-only
  but `CPUOffloadPolicy` runs the step on CPU-resident params).
- **Collator** no longer materializes a dense `(1,S,S)` mask (O(S²) + confuses HF mask prep);
  packing is carried by `cu_seq_lens_*` + `position_ids`, `attention_mask=None`.
- **Checkpoint** saves only the LoRA adapters (`checkpointing/lora.pt` + `lora_meta.json`),
  not the full 31B every 100 steps (~62GB → disk blowup).
- Backbone call no longer passes `labels` (loss is computed in the wrapper with Liger FLCE).

## merge_infer.py — custom LoRA, not PEFT

The adapters are a custom `LinearLoRA` (`y = Wx + (alpha/r)·B(Ax)`), so merge is
`W += (alpha/r)·(B@A)` per wrapped Linear (`q/k/v/o_proj`, vision skipped). `merge_infer.py`
loads a clean base model with **`attn_implementation="sdpa"`** (generation uses a normal causal
mask — NOT the packed `dynamic_attention`, which needs cu_seqlens), folds the deltas, and
generates. `scaling`/`r`/`alpha` come from `checkpointing/lora_meta.json`. NOTE the checkpoint
keys carry the FSDP/activation-checkpoint wrapper segment `._checkpoint_wrapped_module` — the
merge strips it to map onto the clean base weight names (else 0/N adapters merge).

## Runtime gotchas hit on a real 2× H100 pod (all fixed; in order of appearance)

1. **PEP 668** — base image `/usr` python is externally-managed; `uv pip install --system`
   needs `--break-system-packages` (ephemeral pod, fine). In `run.sh`.
2. **NCCL NVLS** — first FSDP all-gather dies with `Failed to bind NVLink SHARP (NVLS) Multicast
   memory … CUDA error 401`. Fix: `export NCCL_NVLS_ENABLE=0` (+ `NCCL_CUMEM_ENABLE=0`). In `run.sh`.
3. **`mm_token_type_ids` required when training** — gemma4's `create_causal_mask_mapping` raises
   if absent. The collator emits an all-zero `mm_token_type_ids` (text-only) of shape (1, S).
4. **`per_layer_inputs` double-pass** — the wrapper must NOT forward `per_layer_inputs` to
   `self.model`; Gemma4Model computes it internally and re-passes it → "got multiple values for
   keyword argument 'per_layer_inputs'". The wrapper now passes only the text-training essentials.
5. **CPU-offload checkpoint needs gloo** — `full_tensor()` all-gathers CPU-resident DTensor shards;
   NCCL can't ("No backend type associated with device type cpu"). Init
   `init_process_group(backend="cpu:gloo,cuda:nccl")`.
6. **Sequence length is memory-bound by the FULL-attention layers.** head_dim 512 → SDPA, and an
   explicit attn_mask forces the **math backend**. A *single* `(H, S, S)` score is 78 GiB at 32k →
   OOM even on 4× H100 (FSDP shards params, NOT that per-rank activation — every GPU builds the whole
   thing; confirmed 32k/24k both OOM at batch_size=1). **FIXED** by query tiling in `attention.py`
   (`SDPA_QUERY_BLOCK`, see "Memory-bounded full attention" above): live score → `(H, block, S)`,
   checkpointed backward, bit-exact incl. grads. With it, the 32k pack trains; without it the ceiling
   was ~16–20k. ⚠ Two collateral notes: `gemma4.py`'s collator `np.concatenate`s the whole DataLoader
   batch into ONE packed sequence, so use **`--batch_size 1`** (default 2 doubles the per-step
   sequence). And the `glm5.1-fp8-test` convs (with tool schemas + all_reasoning) are min 19k /
   **median 35k** / max 81k tokens — only 1 of 77 fits ≤20k, which is *why* the query-tiling fix
   (train at 32k → 32 of 77) matters here.

## Verified end-to-end on 2× H100 SXM (driver 580, CUDA 13)

- `test_attention.py` on GPU: SDPA (1.9e-6) + **FA3 cu_seqlens varlen (9.4e-3)** + float≠bool — all PASS.
- Training runs: 29.8B params, 686.88M LoRA trainable, loss decreases, ~1.3k tok/s; checkpoints
  `lora.pt`+`lora_meta.json`. Metrics push to **wandb** (`gemma4.py --wandb`, key from `~/.netrc`).
- `merge_infer.py`: 230/230 adapters merged; base (scaling 0) generates coherent text.
- New `gemma4.py` flags: `--max_epochs`, `--lr`, `--max_seq_len_train`, `--limit_samples`,
  `--wandb`, `--wandb_project`. Launch env: `NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0`.

## Function-calling eval (base vs finetuned) — `eval_funccall.py` + vLLM

Goal: SyntheticGen function-calling accuracy (headline = `tool_call_f1`) on `glm5.1-fp8-test`,
base vs finetuned, to test "did the overfit help". Hard-won facts:

- **Inference MUST use vLLM, not transformers.** The eval prompts are ~22–56k tokens (full tool
  schemas + agentic history), and gemma-4's head_dim-512 full-attention layers have **NO
  memory-efficient SDPA kernel in PyTorch** (flash caps at 256; EFFICIENT + cuDNN both return
  "No available kernel"; flex_attention fails: Triton shared-mem >227KB at head_dim 512, and the
  small-block monkeypatch hits an Inductor lowering error). So transformers falls to the math
  backend → O(S²) → 61 GiB OOM at 22k tokens. `device_map="auto"` only splits *weights* (pipeline
  parallel), not the single 61 GiB score tensor. vLLM tensor-parallel (TRITON_ATTN backend, head
  split across GPUs) + paged KV handles it. Serve TP=2, `--max-model-len 65536` (single convs reach
  ~56k), `--gpu-memory-utilization 0.85`; pin `prometheus-fastapi-instrumentator>=7` (0.23.0 bug).
- **Serving a MERGED multimodal model from a local dir** needs `preprocessor_config.json` +
  `processor_config.json` — `save_pretrained()` doesn't write them. gemma-4 ships only
  `processor_config.json` (with an embedded `image_processor` block); build the missing
  `preprocessor_config.json` from that block. (Serving the base by repo-id tolerates the absence;
  a local dir does not.)
- **`eval_funccall.py --vllm-url` API mode** tokenizes the prompt locally (chat template + tools),
  sends token ids to `/v1/completions`, and parses gemma-4's NATIVE output `call:NAME{k:v,...}`
  (no `<|tool_call>` wrapper in generated text; args are unquoted gemma-format, not JSON — a small
  recursive parser converts them to JSON so the upstream scorer's json.loads/coverage/type checks
  work). DON'T pass an attention_mask to generate in the local-transformers path either.
- **Result (this overfit):** base `tool_call_f1` **0.7154** (prec .87 / rec .61, full 77 convs);
  finetuned **0.0** — the 25-epoch / lr 3e-4 / scaling-2.0 / 4k-truncated overfit **collapsed**
  the model: it generates the single token `<|"|>` forever (even on "hi"), 0 tool calls. Low train
  loss (44→6) ≠ usable model. For a real improvement: lr ~1e-5–5e-5, 2–3 epochs, scaling ≤1,
  include MLP, and don't train on 4k-truncated bins while evaluating on 22k prompts.

## Successful re-run + merged model on HF (2026-06-19)

The collapse was traced to two now-fixed bugs (the `scaling` kwarg + non-zero `lora_b` init; see the
README "Status"), so the corrected pipeline was re-run on the **full-length 32-bin pack** (`packed_data`,
median 26k / max 32k tokens — NOT the 4k-truncated bins): **lr 5e-5, 3 epochs (48 steps), scaling 2.0,
q/k/v/o, FSDP2 on 2× H100**, with `SDPA_QUERY_BLOCK=512` + `expandable_segments:True` (see the OOM note
above). Train loss **7.08 → 0.459**, no collapse, generation coherent. `merge_infer.py --merged-out`
folded **230/230** adapters and the merged model generates clean text.

The merged model is pushed (private) to **`huseinzolkepliscicom/gemma4-31b-funccall-merged`** for vLLM
reuse — it includes `processor_config.json` + a `preprocessor_config.json` built from the `image_processor`
block (so vLLM can load the multimodal dir). Serve: `vllm serve <repo> --tensor-parallel-size 2
--max-model-len 65536 --gpu-memory-utilization 0.85` (+ pin `prometheus-fastapi-instrumentator>=7` on
0.23.0). **Eval verdict (2026-06-19): the finetune did NOT improve function-calling.** On the same
first 25 convs (apples-to-apples) base `tool_call_f1` **0.6975 vs finetuned 0.6485** (Δ −0.049; lower
precision AND recall — it under-calls tools). ⚠ 6/25 finetuned convs errored on the 65k context limit
(prompt ≤62k + 16384 max_new > 65536 → vLLM 400), which drags the number down — for a clean re-run
lower `--max-new-tokens` (≤~3k) or raise `--max-model-len`; even so, no improvement signal. Run it
parallel + resumable: `eval_funccall.py --vllm-url … --workers 12 --max-rows 25` (new `--workers`
= ThreadPoolExecutor over convs, identical scores; `--cache` JSONL skip-if-cached resume;
`--metrics-only` aggregates the cache so far). RunPod: pin `--country-code US` (an IN pod pulled the
62GB model at ~25 MB/s + Xet stalled; US Xet-turbo did ~212 MB/s with `HF_XET_HIGH_PERFORMANCE=1`).

## FA4 head_dim-512 attention → long-context (128k) training (2026-06-20)

To train at **128k** context, swap the SDPA-tiled head_dim-512 path for the **FlashAttention-4 fork**
`Scicom-AI-Enterprise-Organization/flash-attention-512` (`dev512/`), which adds **symmetric
head_dim=512 support on SM90 (Hopper), forward AND backward** (memory-efficient recompute path).

- **`gemma4_fa4_attention.py`** (`fa4_attention`) routes **ALL** layers (512 global + 256 sliding)
  through `flash_attn.cute.interface.flash_attn_varlen_func` — O(S) memory, no SDPA O(S²) score, no
  query-tiling. Drop-in for `AttentionInterface`; consumes the same `cu_seq_lens_*` the collator emits.
- **`gemma4.py`** registers both backends; env **`GEMMA_ATTN`** (default `fa4_attention`) selects it.
  FA4 replaces FA3 entirely (handles both head dims) → **no FA3 wheel, no torch-2.12 pin**; use the
  pod's torch (2.9.1+cu130 worked).
- **Install (CUDA 13)**: **`run.sh` does this when `GEMMA_FA4=1`** — it clones `FA4_FORK_REPO`
  (`Scicom-AI-Enterprise-Organization/flash-attention-512`, default `./flash-attention-512`; scp the
  private fork there first or supply a GH token), `uv pip install -e <fork>/flash_attn/cute`, then pins
  `uv pip install "nvidia-cutlass-dsl[cu13]==4.4.2" "quack-kernels==0.3.10"` (the `>=` bounds pull
  too-new deps that break — quack 0.5.0 / cutlass-dsl 4.5.x). CuTeDSL JIT-compiles kernels at runtime
  (`FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` to cache). The cute package installs as the
  `flash_attn.cute` namespace pkg — **import only works from a dir with NO local `flash_attn/` folder**
  (else the FA2 top-level `flash_attn/__init__.py` tries `flash_attn_2_cuda` and dies). Run training
  from the job dir, not inside the fork.
- **`compare_logits_fa4.py` PASSED on the real model: cosine 0.999646**, argmax+top5 identical to
  default attention (tighter than the SDPA path's 0.998). FA4 head_dim-512 is numerically correct.
- **Memory finding (the real ceiling):** FA4 fixes the *attention* O(S²) wall, but the model's
  **activation-checkpoint-saved layer inputs are O(seq×layers) and FSDP can't shard them** (they're
  activations, not params) — so on **4× H100 80GB** training OOMs around ~64–90k regardless of context
  (flat ~73GB live; 64k OOM'd by a mere 0.12GB with plain checkpointing). 128k dataset pack
  `huseinzolkepliscicom/gemma4-multipack` = 26 bins, median 109k / max 126k tokens. Repack at N with
  `pack_dataset.py --max-seq-len N` (max single conv = 81k, so <82k drops 1 conv).
  - **`offload_wrapper(checkpoint_wrapper(...))`** (activation CPU offload to the pod's RAM) WOULD lift
    this to 100k+, but we chose NOT to use it (keep parity with minimax's plain checkpointing).
  - **Resolution: run long-context on bigger-VRAM GPUs.** Prod is **8× H20 144GB** (1152GB total,
    144GB/GPU) — the activation ceiling that capped the 80GB H100s is a non-issue there, so 64k–128k
    train without offload/repack hacks. The 4× H100 run only *validated the FA4 integration*.

### FA4 128k on the `tm` H20 VM — the run recipe (2026-06-20)

128k trains here because 144 GB/GPU clears the activation ceiling. Setup (see `../CLAUDE.md` "tm H20
VM" for box access/etiquette — shared box, use GPUs you're given, everything under `/share`):

```bash
ssh -i ../../scicom root@8.222.165.68 -p 1024
# stage: scp *.py + the flash-attention-512 fork to /share/autotrain-gemma4 + /share/
# venv + deps (NO FA3 wheel; FA4 replaces it):
uv venv /share/gemma4-fa4-venv --python 3.12 && source /share/gemma4-fa4-venv/bin/activate
uv pip install torch --torch-backend=cu130          # got torch 2.12.1+cu130; FA4 cute is torch-flexible
uv pip install kernels==0.14.1 "transformers==5.5.0" liger-kernel peft wandb pandas pyarrow \
  mlflow psutil pynvml "git+…/ChiniDataset.git" -U huggingface_hub hf_transfer
uv pip install -e /share/flash-attention-512/flash_attn/cute
uv pip install "nvidia-cutlass-dsl[cu13]==4.4.2" "quack-kernels==0.3.10"   # pin; newer break

export HF_HOME=/share/huggingface HF_HUB_DISABLE_XET=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
hf download huseinzolkepliscicom/gemma4-multipack --repo-type dataset --local-dir ./packed_data  # 128k pack
CUDA_VISIBLE_DEVICES=0 python compare_logits_fa4.py     # gate: PASSED, cosine 0.99984
# train FA4 128k (GEMMA_ATTN=fa4_attention is gemma4.py's default), detach with nohup:
GEMMA_ATTN=fa4_attention PYTORCH_ALLOC_CONF=expandable_segments:True FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
  NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  torchrun --nproc_per_node=6 gemma4.py --lr 5e-5 --max_epochs 3 --wandb
```

`compare_logits_fa4.py` PASSED on the H20 (cosine **0.999843**). gemma-4-31B is pre-cached in
`/share/huggingface` (no download). Post-train: `merge_infer.py --merged-out` then push/serve/eval
as for the earlier runs.

## FA4 cute vs vLLM Triton attention — kernel benchmark (`bench_attention.py`, 2026-06-24)

Head-to-head latency of the **two attention kernels that can actually do gemma-4's head_dim-512
global layers** — FA2/FA3 and vLLM's *FlashAttention* backend all cap at head_dim ≤ 256, so the
realistic choices are **FA4 cute** (`flash_attn.cute.interface.flash_attn_varlen_func`, the
`flash-attention-512` fork — built for *training*: fwd+bwd, contiguous varlen) and **vLLM's
Triton** `unified_attention` (`vllm/v1/attention/ops/triton_unified_attention.py` — the only vLLM
backend that serves head_dim 512; paged KV, 3D split-KV decode kernel). Both run on the two real
gemma-4-31B geometries: **full** (hd 512, 32 q-heads / **4** kv-heads, full causal) and **sliding**
(hd 256, 32 q / **16** kv, window 1024), for **prefill** (q=kv=L) and **decode** (q=1, kv=L) at
L ∈ {1024, 2048, 4096, 8192, 16384, 32768}. Measured on **one H20-3e (SM 9.0), GPU 7**, bf16, torch
2.12.1+cu130 / triton 3.7.1, 50 timed iters (median, CUDA events).

**`fa4_ms` / `triton_ms` (ms); `r = fa4/tri` (<1 ⇒ FA4 faster). Decode `fa4_ms` is FA4 with manual
FlashDecoding (the decode fix below); the no-split baseline is in the decode-fix table.**

| geom | regime | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|------|--------|------|------|------|------|-------|-------|
| full hd512    | prefill | 0.52 / 0.77 (0.69) | 1.69 / 2.82 (0.60) | 6.15 / 10.7 (0.57) | 23.7 / 42.0 (0.56) | 93.5 / 166.2 (0.56) | **371 / 662 (0.56)** |
| full hd512    | decode  | 0.09 / 0.08 (1.06) | 0.09 / 0.08 (1.06) | 0.09 / 0.09 (1.06) | 0.09 / 0.09 (1.03) | 0.13 / 0.14 (**0.94**) | **0.24 / 0.26 (0.92)** |
| sliding hd256 | prefill | 0.22 / 0.64 (0.35) | 0.50 / 1.75 (0.28) | 1.04 / 4.00 (0.26) | 2.14 / 8.48 (0.25) | 4.31 / 17.50 (0.25) | **8.63 / 35.4 (0.24)** |
| sliding hd256 | decode  | 0.09 / 0.08 (1.07) | 0.09 / 0.08 (1.08) | 0.09 / 0.08 (1.11) | 0.09 / 0.09 (**0.99**) | 0.09 / 0.09 (1.01) | **0.09 / 0.09 (1.01)** |

**Verdict — FA4 wins prefill, and now wins long-context decode too** (after the decode fix below):
- **Prefill: FA4 wins big.** ~**1.8×** faster on full hd512 (95 vs 53 TFLOP/s at 32k) and ~**4×**
  on sliding hd256 (125 vs 31 TFLOP/s) — the Triton kernel isn't tuned for head_dim 512 and tiles
  the long key axis poorly. Ratio is flat across L (0.56 full / 0.24–0.35 sliding). This is the
  long-context-training regime, which is *why* gemma4 trains with FA4 (`GEMMA_ATTN=fa4_attention`).
- **Decode: FA4-cute FlashDecoding beats Triton at ≥16k** (0.94× at 16k, 0.92× at 32k; ~1.03–1.13×
  below that — the cute kernel's WGMMA M=64 can't tile GQA-8's 8 query rows tightly at short L). **But
  the actual decode winner is a purpose-built Triton small-M flash-decode kernel** (`decode_attention.py`)
  that beats *both* FA4-cute and vLLM at **every** L (~2× vLLM at short context, ~2.3× on sliding) — see
  "Decode winner" below. FA4-cute stays the prefill/training path.
- **Both are numerically correct.** All shapes: FA4↔Triton **cosine ≥0.99999**, max-abs ≤ 8e-3,
  and **both bit-match an fp32 SDPA reference** (`fa1.0000/tr1.0000`) — including the fused
  FlashDecoding combine. Confirms head_dim-512 runs *correctly* through vLLM Triton too.

### Decode fix — manual FlashDecoding (FA4's native SplitKV is Blackwell-only)

FA4 decode was originally slow because **`flash_attn_varlen_func(num_splits=1)` launches one CTA per
(query-block, head)** — for q_len=1 that's a handful of CTAs, each walking the *entire* KV serially,
so latency ∝ context (0.18→1.84→**3.6 ms** at 16k→32k on full hd512, up to 14× Triton). The fork
*has* a split-KV path + combine kernel, but it **asserts out on Hopper** (`assert not is_split_kv,
"SplitKV not supported on SM 9.0"`; `FlashAttentionForwardSm90` takes no `is_split_kv` — split-KV is
wired only for Blackwell sm100/110). So `num_splits=0` (the auto heuristic) hits that assert on the
H20; `pack_gqa` was already auto-on and isn't enough.

Fix (`run_fa4_decode_split` / `make_fa4_decode_runner`, **no kernel change**): do FlashDecoding
*around* the kernel — split the KV into N contiguous chunks, run them as ONE varlen batch (the decode
query repeated once per chunk, `return_lse=True`), then LSE-combine the N partials
(`O = Σ_s exp(lse_s−M)·O_s / Σ_s exp(lse_s−M)`). N× the CTAs → the long KV is processed in parallel.
Exact for decode: the single query sits at the end and attends every key with **no causal mask**, so
each chunk is a plain full attention over its key range (a sliding layer attends only the last
`window` keys, so only those are split).

To actually **beat** Triton needed three more things — profiling showed the split *kernel alone* already
beat Triton (16k 0.128 vs 0.140 ms, 32k 0.241 vs 0.264), so the whole gap was Python/launch overhead:
1. **Fused Triton combine** (`_fa4_combine_kernel`) — the LSE merge in ONE kernel launch instead of
   ~5 torch ops (`amax`/`exp`/`einsum`/`sum`/`div` ≈ 40–70 µs of launch latency).
2. **GPU-built `cu_seqlens` + precomputed static metadata** — the per-chunk `cu_seqlens` depend only
   on (kv_len, N), so they're built once on-device (no per-call `torch.tensor(...)` H2D sync) and
   reused across decode steps, exactly like the Triton runner's preallocated `block_table`. Only
   query-repeat + kernel + combine sit on the per-token path.
3. **Broadcast query, no copy** — the decode query is repeated to N split-rows as a stride-(0,D,1)
   `q.expand(...)` *view* (all docs read the same memory), skipping the `.contiguous()` copy (~10–14 µs,
   the last small-L gap). Verified bit-exact vs the contiguous copy / fp32 SDPA.

Result — **FA4 now beats Triton at ≥16k and ties below** (full hd512; best N∈{2..64}; FA4 ~flat, naive ∝ L):

| L | naive | FA4 FlashDecoding | speedup | Triton | FA4 vs Triton |
|---|-------|-------------------|---------|--------|---------------|
| 1024  | 0.184 ms | 0.088 ms | 2.1×  | 0.082 ms | 1.06× (Triton) |
| 4096  | 0.528 ms | 0.092 ms | 5.7×  | 0.087 ms | 1.06× (Triton) |
| 8192  | 0.965 ms | 0.090 ms | 10.7× | 0.087 ms | 1.03× (Triton) |
| 16384 | 1.843 ms | 0.131 ms | 14.1× | 0.140 ms | **0.94× (FA4 wins)** |
| 32768 | 3.596 ms | 0.243 ms | **14.8×** | 0.263 ms | **0.92× (FA4 wins)** |

So FA4 beats Triton's purpose-built paged decode at **≥16k**, and is ~1.03–1.13× below it at ≤8k (both
~0.09 ms). **Sliding decode** is window-bounded (~0.09 ms, only 1024 keys); split helps (best N≈8) and
FA4 ties Triton (0.99–1.1×). Timing uses the precomputed-metadata runner (per-token path =
broadcast-query + kernel + combine), comparable to the Triton runner. FA4 **prefill is untouched** (the
heuristic keeps 1 split — split-KV is a decode-only win). `run_fa4_decode_split` is the reference if FA4
is used for inference decode; the packed *training* path (`gemma4_fa4_attention.py`) is prefill-only and
unaffected.

**Why the *cute* kernel can't win short-context decode (CUDA-graph evidence).** It's tempting to blame
the ≤8k gap on FA4's per-call Python/launch overhead — but **CUDA-graphing the per-token path (both
kernels) disproves that** (and the fix turned out to be a separate small-M kernel — see below). Graphs strip the launch overhead and *both* drop hard, yet Triton pulls **ahead** at short L:

| L | eager fa4/tri | graphed fa4 / tri |
|---|---------------|-------------------|
| 1024  | 1.07 | 0.031 / 0.019 ms (**1.66**) |
| 2048  | 1.13 | 0.048 / 0.026 ms (**1.86**) |
| 8192  | 1.06 | 0.079 / 0.077 ms (1.03) |
| 16384 | 0.94 | 0.131 / 0.140 ms (0.94) |
| 32768 | 0.92 | 0.244 / 0.265 ms (0.92) |

So the short-L gap is **GPU work, not launch latency**: FA4's prefill-derived cute kernel under-utilizes
its MMA at q_len=1 / tiny KV, and graphs expose that (eager mode's launch floor was actually masking it
— flattering FA4). At ≥16k the KV scan dominates, graphs change nothing, and FA4 wins regardless.

**The wall is WGMMA, and it can't be tuned away (verified).** The decode under-utilization is structural:
GQA-8 packs only **8 query rows**, but the SM90 forward is **WGMMA**-based with an **M=64 atom**, so it
runs a 64-row tile with 8 valid rows (~8× waste). Sweeping the fork's `tile_mn` confirms there's no
escape from the interface — `tile_m<64` is rejected outright (`ValueError: Expected size in shape to be
strictly…`), the only valid hd512 config is the default `64×80` (explicit alternates
`cudaErrorIllegalInstruction`), and `pack_gqa`/`tile_n` don't change the M-waste. Triton wins short
decode precisely because its `tl.dot` tiles small-M (BLOCK_M=16) efficiently. So beating Triton at
short-context decode needs a **non-WGMMA / small-M decode kernel** — *not* a tile/config tweak of the
cute kernel (its WGMMA M=64 is hard). **So I wrote that kernel** (below).

### Decode winner — a purpose-built Triton flash-decode kernel (`decode_attention.py`)

Rather than fight WGMMA in the cute fork, the small-M decode path is a tidy ~80-line **Triton** kernel:
split-KV, online softmax, **BLOCK_M=16** (so GQA-8's 8 query rows waste ½ a tile, not ⅞), contiguous
KV (no paging/quant/sink/3D-segment machinery), `num_stages=1` (the hd512 K+V tiles would blow smem
otherwise), + the same fused LSE combine. It **beats BOTH the cute FA4 FlashDecoding AND vLLM's general
Triton at every context length** — including the short contexts that were the holdout — at cosine 1.0
vs fp32 SDPA (`make_decode_runner`; bench `bench_decode_kernel.py`, one H20, median of 80):

| geom | L | **custom** | vLLM | FA4-cute | custom/vLLM | custom/FA4 |
|------|---|-----------|------|----------|-------------|-----------|
| full hd512    | 1024  | **0.038** | 0.085 | 0.093 | 0.45× | 0.41× |
| full hd512    | 8192  | **0.057** | 0.089 | 0.097 | 0.64× | 0.59× |
| full hd512    | 32768 | **0.195** | 0.263 | 0.244 | 0.74× | 0.80× |
| sliding hd256 | (any) | **~0.038** | ~0.088 | ~0.094 | ~0.44× | ~0.41× |

i.e. **~2× faster than vLLM at short context, 1.35× at 32k, ~2.3× on sliding** — the general vLLM
kernel's paging/branch overhead and the cute kernel's WGMMA M-waste both vanish. **This is the decode
kernel to use** (`flash_decode` / `make_decode_runner` in `decode_attention.py`); FA4-cute stays the
**prefill/training** path (`gemma4_fa4_attention.py`), where it wins ~1.8–4×. The cute fork is left
untouched (no training-path risk). Note this kernel uses contiguous KV — fine for the autotrain decode
path; a paged-serving deployment would add a block-table indirection (as vLLM does).

**Run it** (results in `bench_attention_results.json`). Needs torch+triton (for the Triton kernel)
+ the FA4 cute fork + cutlass-dsl — same venv as the FA4 training recipe above. The benchmark runs
vLLM's *exact* kernel source via `bench_vllm_shim/` (tiny stand-ins for the ~5 vllm internals the
kernel imports — `envs`/`logger`/`platforms`/`triton_utils`/`KVQuantMode`, faithful to upstream;
none touch the attention math), so **no full vLLM build** is needed:

```bash
# 1) copy vLLM's Triton kernel into the shim (pins to whatever vLLM checkout you point at):
VLLM_REPO=/home/husein/ssd3/vllm bash bench_setup.sh            # add: --venv /share/gemma4-bench-venv (FA4_FORK=…) to also build the venv
# 2) run on one GPU (FA4 cute JIT-compiles per shape; warmup absorbs it). Import needs a dir
#    with NO local flash_attn/ folder — the job dir is fine:
ssh -i scicom root@8.222.165.68 -p 1023   # the tm H20 box (port 1023)
CUDA_VISIBLE_DEVICES=7 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python bench_attention.py --iters 50 --warmup 10 --out results.json   # --quick for a smoke
```

⚠ The tm box this ran on (`dsw-464391…`, port **1023**) is a *fresh* box — the FA4 venv/fork from
the recipe above didn't exist, so it was rebuilt: torch 2.12 had to install via the explicit index
(`uv pip install torch --index-url https://download.pytorch.org/whl/cu130`; this `uv`'s
`--torch-backend` caps at cu129), and the box has **no `rsync`** (stage with `tar | ssh 'tar x'`,
not rsync). gemma-4 is pre-cached in `/share/huggingface`.
