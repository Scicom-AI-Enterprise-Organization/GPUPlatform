# autotrain/mistral-small — Mistral-Small-4-119B FP8 MoE LoRA finetune (4× H100 on RunPod)

Standalone training job (NOT part of the gateway), the Mistral-Small-4 sibling of
`../minimax-m2` (and `../gemma4`). LoRA-finetunes the **text** model of
**`mistralai/Mistral-Small-4-119B-2603`** with PyTorch **FSDP2** (`fully_shard`) across 4 GPUs.

The published model is a **`Mistral3ForConditionalGeneration`** (multimodal): a Pixtral vision
tower + projector wrapping a **`mistral4`** text decoder. We finetune **text only** — freeze the
vision tower/projector and run the language model directly. The text model is **119B-total** MoE:
128 routed experts (top-4) + **1 shared expert**, 36 layers, hidden 4096, vocab 131072, **MLA
attention**, FP8 per-tensor.

```
mistral_small.py   training entrypoint (torchrun, FSDP2, packed varlen, Liger FLCE)
lora.py            the interesting part — LinearLoRA (MLA q_a/q_b/kv_a/kv_b/o + shared MLP)
                   + fused grouped-MoE routed-expert LoRA + per-tensor FP8 dequant
test_lora.py       CPU correctness: per-tensor+block dequant, LinearLoRA, fused MoE fwd+grads
compare_logits.py  GPU logits check on the real model: our LoRA(B=0) vs an independent bf16 ref
pack_dataset.py    build the multipacked ChiniDataset from a chat parquet (Mistral-Small-4 template)
merge_infer.py     attach the trained LoRA to the FP8 base and generate
run.sh             pod bootstrap: deps + FA3 wheel + LoRA test + pack/download + torchrun
```

## How Mistral-Small-4 differs from minimax-m2 (this reshaped the design)

| | minimax-m2 (`../minimax-m2`) | Mistral-Small-4 (here) |
|---|---|---|
| size | 230B / 10B-active MoE | **119B-total** MoE (128 routed top-4 + **1 shared**, 36 layers) |
| attention | plain GQA q/k/v/o, head_dim 128 | **MLA** (DeepSeek-style: q_lora_rank 1024, kv_lora_rank 256), still **uniform head_dim 128** |
| FP8 | **128×128 block**-scaled | **per-tensor** (`weight_block_size=null`), static activations |
| wrapper | `MiniMaxM2ForCausalLM` (text-only) | **`Mistral3ForConditionalGeneration`** (vision + `mistral4` text) |
| layer class | `MiniMaxM2DecoderLayer` | `Mistral4DecoderLayer` (`self_attn` MLA + `mlp`=`Mistral4MoE`) |

What's the SAME as minimax-m2: head_dim is a uniform **128** (qk_head_dim == v_head_dim), so every
layer runs stock `flash_attention_2`/`_3` + FlashAttention's native varlen packing — **no custom
attention**. The collator emits `cu_seq_lens_q/k` + per-doc-reset `position_ids`; transformers'
`_flash_attention_forward` consumes them directly. FA varlen is O(S) so long packed bins are cheap.

## ⚠ The transformers 5.5.0 FP8 load bug (load-bearing workaround)

transformers 5.5.0's `FP8Experts.__init__` does `getattr(config, "num_local_experts",
config.num_experts)`. Python evaluates the **default** `config.num_experts` *eagerly*, and
`Mistral4Config` only exposes `num_local_experts` (via `attribute_map`) — there is **no**
`num_experts` — so `from_pretrained` crashes with `AttributeError` before getattr even runs.
**Every load path here sets `config.text_config.num_experts = num_local_experts` first**
(`load_patched_config()` in `mistral_small.py` / `compare_logits.py` / `merge_infer.py`). Without
it the model does not load at all. (Verified: with the patch, MLA projections load as `FP8Linear`,
routed experts as `FP8Experts` with per-tensor `(E,1,1)` `*_scale_inv`.)

## The crux: FP8 base is frozen, but the FP8 kernels are inference-only

transformers loads the MLA q/kv/o + the shared-expert MLP as **`FP8Linear`** (per-tensor scalar
`weight_scale_inv` + static `activation_scale`) and the routed experts as **`FP8Experts`** (3D
`gate_up_proj` (E,2I,H) + `down_proj` (E,H,I), per-expert `(E,1,1)` `*_scale_inv` + per-expert
activation scales). Their forward kernels (Triton `w8a8`, the per-expert activation quant) are an
**inference path and NOT autograd-differentiable**. Worse for this model: the fused
`grouped_mm`/`batched_mm`/`deepgemm` experts dispatches all **`raise NotImplementedError` for
`activation_scheme="static"`**, so even stock inference only runs the eager per-expert FP8 loop.
None of that trains.

**Fix (QLoRA's trick), in `lora.py`:** keep the weight stored in FP8 (cheap, sharded by FSDP), and
inside the forward **dequantize the per-tensor-scaled weight to bf16 on the fly** and run a normal,
differentiable matmul. The dequantized bf16 weight is transient — **activation checkpointing** on
each `Mistral4DecoderLayer` recomputes it in the backward instead of retaining it, so peak memory
stays ~ one layer's bf16 weights. ⚠ Activation checkpointing is **load-bearing**, not a nicety.

- **Attention LoRA** — `LinearLoRA` wraps each frozen MLA `FP8Linear`
  (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`):
  `y = dequant_linear(x) + scaling · B(A(x))`.
- **Shared-expert LoRA** — `LinearLoRA` wraps the shared MLP's `gate_proj`/`up_proj`/`down_proj`.
- **Routed-expert LoRA** — per-expert low-rank adapters folded into a **fused grouped_mm** experts
  forward. `gate_up` and `down` each compute `(frozen dequant grouped_mm) + (bf16 LoRA
  grouped_mm)`, with the **SwiGLU gate applied to the sum (base+LoRA)** before the down projection
  (the gate is non-linear, so base and LoRA must be combined at each projection).

`lora_b` / expert-`*_lora_b` are **zero-initialised** so the adapter is a no-op at step 0.

### Per-tensor dequant (`dequantize_fp8`)

`block_size=None` (Mistral-Small-4) → `w_deq = fp8_to_fp32(w) * scale_inv`, broadcasting a scalar
(2D `FP8Linear`) or a per-expert `(E,1,1)` scalar (3D `FP8Experts`). The 3D path chunks over experts
(`MISTRAL_DEQUANT_EXPERT_CHUNK`, default 16) so the fp32 transient is bounded — only matters for the
full-model compare/merge paths (under FSDP each rank holds 1/world_size of the experts). The helper
also handles 128×128 block layouts (for the unit test / block-quantized siblings).

### Fused MoE — built into transformers 5.5.0

`Mistral4NaiveMoe` (the routed `.experts`) is `@use_experts_implementation`-decorated, stores 3D
expert params, and `torch._grouped_mm` (torch ≥ 2.x) backs `grouped_mm_experts_forward`. Our
training forward reuses the *same* grouped/sorted layout + `_moe._grouped_linear` helper but on
dequantized bf16 weights + bf16 LoRA, so it's differentiable. The shared expert + router `gate`
(both kept bf16, the router is a raw Parameter so never FP8-converted) live in `Mistral4MoE.forward`
*around* the routed experts — `Mistral4MoE.forward` calls `self.experts(...)` (our fused LoRA) and
`+ self.shared_experts(residual)` (our LinearLoRA-wrapped MLP); we touch neither's orchestration.

## LoRA targets (analogue of minimax's "q/k/v/o + dense MoE layers")

`apply_mistral_lora()` freezes the whole base, wraps the **5 MLA attention projections**, adds
per-expert LoRA to every routed-experts block, and wraps the **shared-expert MLP** (gate/up/down).
The router `gate`, the q/kv layernorms, embeddings, vision tower/projector, and lm_head stay frozen
and un-adapted. Defaults: `attn_r=16`, `moe_r=16`, scaling 1.0 (lesson from gemma4/minimax: LoRA
`lr ~1e-5..5e-5`, scaling ≤ 1, few epochs — over-training collapsed the gemma4 model). Flags:
`--no_moe_lora` (attention only), `--no_shared_lora` (keep routed, skip the shared MLP).

⚠ **Trainable-param note.** The routed-expert LoRA dominates: at `moe_r=16`, 128 experts × 36
layers ≈ **1.05B** trainable bf16 params (~2.1GB) + AdamW fp32 state (~8.5GB) + grads, sharded /4 ≈
~3GB/GPU. The frozen FP8 base is ~119GB → ~30GB/GPU sharded across 4. Plenty of headroom on 80GB
(4× H100 ≈ 122GB base /4 ≈ 30GB + ~3GB LoRA + one layer's transient bf16 dequant + activations).
2× H100 (~61GB base/GPU) is *possible* but tight for long bins — prefer 4. Drop `moe_r` / use
`--no_moe_lora` / shorter packed bins if you OOM.

## FSDP2 sharding

`fully_shard` each `Mistral4DecoderLayer` then the root. `MixedPrecisionPolicy(param_dtype=None,
reduce_dtype=fp32)` — **param_dtype MUST stay None** so FSDP does not cast the frozen FP8 weights to
bf16 (that would defeat the per-tensor dequant); every param keeps its storage dtype (fp8 frozen,
bf16 LoRA/norms/embed/lm_head/vision, fp32 scales) and only gradient reduction is fp32. The frozen
vision tower + embeddings + lm_head sit in the root shard. `--cpu_offload` adds `CPUOffloadPolicy`
for tight VRAM (slow). `init_process_group("cpu:gloo,cuda:nccl")` so the LoRA-checkpoint
`full_tensor()` all-gather works even under CPU offload.

The training forward (`CustomMistral3ForCausalLM`) bypasses the vision path: it calls
`self.model.language_model(...)` directly (text-only) and computes the loss with **Liger
FusedLinearCrossEntropy** on `self.lm_head.weight` (vocab 131,072 — materializing (1, S, 131k)
logits OOMs at long S; FLCE fuses the lm_head matmul + cross-entropy).

## Logits parity (`compare_logits.py`)

Always compare logits against a trusted reference when you touch a custom forward (rule in
`../CLAUDE.md`). Like minimax-m2, the native FP8 inference path is not a safe reference here
(per-tensor static w8a8 / eager FP8Experts loop on this stack), so `compare_logits.py` builds an
**independent bf16 reference** — a naive per-token/per-expert dequant loop, separate code from
`lora.py`'s fused grouped path — and checks our LoRA(B=0) against it. Pass condition: **top-1 argmax
match + cosine > 0.997** (near-tied 4th/5th top-k ranks may swap = bf16 fused-vs-loop accumulation
noise). Plus: all `*_lora_b` zero at init (no-op), and poking a non-zero B *does* move the logits.
The native FP8 pass is run for information (`MISTRAL_RUN_NATIVE=0` to skip). It loads with
`device_map="auto"` capped to `MISTRAL_PER_GPU_GIB` (default 30 GiB/GPU) so the bf16 dequant
transient fits. **Run it on the pod before training.** ⚠ This uses `device_map` pipeline-parallel,
NOT FSDP — so the FSDP2 fp8 all-gather risk (#1 below) is still unverified.

## Status — what's verified vs what needs the pod

**Verified locally on CPU (`autotrain/.venv`, torch 2.12.1 + transformers 5.5.0):**
- `test_lora.py`: per-tensor (2D + 3D) AND block FP8 dequant round-trip; `LinearLoRA` zero-B no-op
  (bf16 + per-tensor-FP8 base); fused grouped LoRA-MoE forward matches an independent per-expert
  reference to ~1e-7, **LoRA gradients match to ~1e-7, frozen base gets no grad**.
- **End-to-end wiring on a tiny REAL `Mistral3ForConditionalGeneration`** (bf16, eager attn): the 5
  MLA projections + routed MoE + shared MLP wrap correctly (10/2/6 modules on a 2-layer model);
  LoRA(B=0) reproduces the base logits exactly (~1e-7) through the full model forward; a non-zero B
  moves them; backward puts grads ONLY on the LoRA tensors.
- `pack_dataset.py`: Mistral-Small-4 tokenizer + chat template load; reasoning folds into
  `[THINK]` blocks (rendered), tools render in `[AVAILABLE_TOOLS]`, tool calls render; BOS appears
  once (no double-BOS); full-sequence `labels = input_ids` fallback (no `{% generation %}`).
- The transformers 5.5.0 `num_experts` load bug + the `config.text_config.num_experts` workaround.
- `mistral_small.py` / `compare_logits.py` / `merge_infer.py` import + compile; all
  transformers/FSDP symbols exist as used.

**NOT yet run on hardware (validate on the pod — treat the first run as a smoke test):**
1. **FSDP2 all-gather of `float8_e4m3fn` params** (the #1 risk, same as minimax-m2). If it errors:
   `device_map="auto"` pipeline-parallel (slower, no FSDP), or upcast-to-bf16 + CPU offload.
2. **`compare_logits.py` on the real 119B model** — our LoRA(B=0) vs the independent bf16 ref
   (top-1 + cosine > 0.997), and whether the native per-tensor-static FP8 forward NaNs/diverges.
3. **CPU RAM at load.** Default path: every rank `from_pretrained`s ~119GB to CPU → ~119GB × ranks.
   Use `--low_cpu_shard_load` (meta-init + DCP broadcast from rank 0) on a tight-RAM box (ported
   from minimax-m2; itself unverified on hardware).
4. End-to-end loss-decrease, tok/s, the FA3 varlen path, and `merge_infer.py` generation.

First run: `MISTRAL_DEPS_ONLY=1 bash run.sh`, then `python compare_logits.py`, then
`bash run.sh -- --max_steps 20 --limit_samples 4` before a full epoch.

## RunPod workflow (use `runpodctl`; `RUNPOD_API_KEY` + `HF_TOKEN` + `WANDB_API_KEY` in `../.env`)

```bash
runpodctl config --apiKey "$(grep -E '^RUNPOD_API_KEY=' ../.env | cut -d= -f2-)"
runpodctl ssh add-key --key-file ~/.ssh/id_rsa.pub

# 4× H100 SXM, CUDA>=13 host, big disk (119GB model + dataset + deps), US region, auto-terminate.
runpodctl pod create --name mistral-small-4-ft \
  --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 4 \
  --image runpod/pytorch:1.0.6-cu1300-torch291-ubuntu2404 \
  --container-disk-in-gb 400 --cloud-type SECURE --country-code US \
  --min-cuda-version 13.0 --ssh --ports "22/tcp" \
  --terminate-after "$(date -u -d '+8 hours' +%Y-%m-%dT%H:%M:%SZ)"

runpodctl ssh info <pod-id>        # ssh string ({"error":"pod not ready"} while booting)
# Work under / (container disk), NEVER /workspace (slow network volume):
scp -P <port> -i ~/.ssh/id_rsa *.py *.sh root@<ip>:/root/autotrain-mistral/
export HF_TOKEN="$(grep -E '^HF_TOKEN=' ../.env | cut -d= -f2-)"   # gated download
# on the pod:  cd /root/autotrain-mistral && hf auth login --token "$HF_TOKEN" && bash run.sh
runpodctl pod delete <pod-id>      # ⚠ billing stops only here. 4× H100 SXM ≈ $13/hr.
```

Gotchas (mostly inherited from minimax-m2/gemma4; all real):
- **GPU id for H100 SXM = `NVIDIA H100 80GB HBM3`**; use the modern `runpodctl pod create`.
- **`--country-code US`** is a hard rule (user requested) — without it RunPod may place the pod in
  IN/elsewhere. **`--min-cuda-version 13.0`** guarantees a cu130 host for the torch-2.12+cu130 / FA3
  stack. **Work under `/` (`/root/...`), NEVER `/workspace`** (slow network volume); big
  `--container-disk-in-gb` (400) instead of a volume; HF cache stays on the container disk.
- **NCCL NVLS** bind fails in RunPod containers → `NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0` (in run.sh).
- **Pin `kernels>=0.12.0,<0.13`** — the latest kernels (0.15.x) makes `LayerRepository(version=...)`
  mandatory and transformers 5.5.0 builds its `_KERNEL_MAPPING` version-less at import → `import
  transformers` itself crashes. (Same gotcha as minimax-m2; in run.sh.)
- Version pinning is load-bearing: the FA3 wheel is built for **torch 2.12**, which also provides
  `torch._grouped_mm` for the fused MoE — keep torch 2.12.x. CUDA backend is chosen from the host
  driver (`run.sh`: ≥13.2→cu132, ≥13.0→cu130, ≥12.6→cu126).
- **Download `model-*.safetensors`, EXCLUDE `consolidated*`** — the repo ships both the HF-format
  FP8 weights (3 shards, what transformers loads) and a mistral-common `consolidated-*` copy (7
  shards). run.sh excludes the consolidated set to avoid downloading ~119GB twice.
