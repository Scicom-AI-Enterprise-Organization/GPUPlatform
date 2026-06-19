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
  ~56k), `--gpu-memory-utilization 0.92`; pin `prometheus-fastapi-instrumentator>=7` (0.23.0 bug).
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
--max-model-len 65536 --gpu-memory-utilization 0.92` (+ pin `prometheus-fastapi-instrumentator>=7` on
0.23.0). **Function-calling eval (`eval_funccall.py`) vs base 0.7154 not yet run** for this checkpoint.
