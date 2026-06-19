# autotrain/minimax-m2 — MiniMax-M2 230B FP8 MoE LoRA finetune (4× H100 on RunPod)

Standalone training job (NOT part of the gateway), the MiniMax-M2 sibling of `../` (gemma4).
LoRA-finetunes **`MiniMaxAI/MiniMax-M2`** (230B total / 10B active MoE) on a packed dataset
with PyTorch **FSDP2** (`fully_shard`) across 4 GPUs.

```
minimax_m2.py    training entrypoint (torchrun, FSDP2, packed varlen, Liger FLCE loss)
lora.py          the interesting part — LinearLoRA (q/k/v/o) + fused grouped-MoE expert LoRA
test_lora.py     CPU correctness test: fused grouped LoRA-MoE fwd+grads vs a per-expert reference
compare_logits.py  GPU logits check on the real 230B model: our LoRA(B=0) vs an independent bf16
                 reference (the stock FP8 forward NaNs — see below). POD-VERIFIED 2026-06-19.
pack_dataset.py  build the multipacked ChiniDataset from a chat parquet (MiniMax-M2 template)
merge_infer.py   attach the trained LoRA to the FP8 base and generate
run.sh           pod bootstrap: deps + FA3 wheel + LoRA test + pack/download + torchrun
```

## How MiniMax-M2 differs from gemma-4 (this is what reshaped the whole design)

| | gemma-4 (`../`) | MiniMax-M2 (here) |
|---|---|---|
| size | 31B dense | **230B / 10B-active MoE** (256 experts, top-8, 62 layers) |
| precision | bf16 | **FP8 only** (`float8_e4m3fn`, block 128×128; 130 shards, ~230GB) |
| attention | dual head_dim **512 (SDPA) + 256 (FA3)** → custom `dynamic_attention` | **uniform head_dim 128, all full attn, GQA 48/8** → stock FA, **no custom attention** |
| layer class | `Gemma4TextDecoderLayer` | `MiniMaxM2DecoderLayer` (`self_attn` + `mlp`=`MiniMaxM2SparseMoeBlock`) |
| loading | local class | native `MiniMaxM2ForCausalLM` (transformers 5.5.0; **not** trust_remote_code) |

The gemma-4 run's whole reason for `attention.py` (head_dim 512 has no flash kernel) **does not
exist here** — MiniMax-M2's head_dim is a uniform 128, so every layer runs stock
`flash_attention_2`/`_3` and FlashAttention's native varlen packing. `minimax_m2.py`'s collator
emits the same `cu_seq_lens_q/k` + per-doc-reset `position_ids`; transformers'
`_flash_attention_forward` consumes them directly (verified in the 5.5.0 source).

## The crux: FP8 base is frozen, but the FP8 kernels are inference-only

transformers 5.5.0 loads MiniMax-M2's FP8 weights as **`FP8Linear`** (q/k/v/o) and
**`FP8Experts`** (3D stacked expert params `gate_up_proj` (E,2I,H) + `down_proj` (E,H,I) with
per-block fp32 `*_scale_inv`). Their forward kernels — Triton `w8a8` / DeepGEMM grouped GEMM, and
the per-expert **activation quant** — are **NOT autograd-differentiable** (they're an inference
path). LoRA training needs gradients to flow *through* the frozen base of every layer to reach
LoRA params in earlier layers, so a non-differentiable frozen forward silently breaks training
for everything below it.

**Fix (this is exactly QLoRA's trick), in `lora.py`:** keep the weight stored in FP8 (cheap,
sharded by FSDP), and inside the forward **dequantize the block-scaled weight to bf16 on the fly**
and run a normal, differentiable matmul. The dequantized bf16 weight is transient — **activation
checkpointing** on each decoder layer recomputes it in the backward instead of retaining it, so
peak memory stays ~ one layer's bf16 weights, not the whole model's. ⚠ Activation checkpointing
is therefore **load-bearing**, not just a memory nicety.

- **Attention LoRA** — `LinearLoRA` wraps the frozen `FP8Linear`:
  `y = dequant_linear(x) + scaling · B(A(x))`.
- **MoE expert LoRA** ("the dense layers in the MoE") — per-expert low-rank adapters folded into a
  **fused grouped_mm** experts forward. `gate_up` and `down` each compute
  `(frozen dequant grouped_mm) + (bf16 LoRA grouped_mm)`, with the **SwiGLU gate applied to the
  sum (base+LoRA) before the down projection** — the gate is non-linear, so base and LoRA cannot
  be split after the block; they must be combined at each projection.

`lora_b`/expert-`*_lora_b` are **zero-initialised** so the adapter is a no-op at step 0 (a non-zero
B corrupts the frozen model from the first step — the bug that produced garbage in the gemma4 run).

### Fused MoE — built into transformers 5.5.0

`torch._grouped_mm` (torch ≥ 2.x; present in the pinned 2.12) backs transformers'
`grouped_mm_experts_forward`, registered as `experts_implementation="grouped_mm"` (Triton fp8 +
`"deepgemm"` variants exist for **inference**). Our training forward reuses the *same* grouped/sorted
layout and `_grouped_linear` helper but on dequantized bf16 weights + bf16 LoRA, so it's
differentiable. (So for training the `experts_implementation` kwarg is moot — we override the
experts `forward` entirely; the fast fp8 kernels are only for inference/eval.)

## LoRA targets (as requested: q/k/v/o + the dense MoE layers)

`apply_minimax_lora()` freezes the whole base, wraps attention **q/k/v/o** with `LinearLoRA`, and
adds per-expert LoRA to every MoE block's `gate_up_proj` + `down_proj`. The router `gate` (kept
bf16, not fp8) and the layernorms stay frozen and un-adapted. Defaults: `attn_r=16`, `moe_r=16`,
`alpha=scaling 1.0` (gemma4 lesson: LoRA `lr ~1e-5..5e-5`, scaling ≤ 1, few epochs — over-training
collapsed the gemma4 model). `--no_moe_lora` adapts attention only.

⚠ **Trainable-param / memory note.** The expert LoRA dominates: at `moe_r=16`, 256 experts × 62
layers ≈ **2.7B** trainable bf16 params (~5.4GB) + AdamW fp32 state (~21GB) + grads, sharded /4 ≈
~8GB/GPU. The frozen FP8 base is ~230GB → ~58GB/GPU sharded. Peak/GPU ≈ 58 (base) + ~8 (LoRA opt) +
one layer's transient bf16 dequant (~7GB) + activations. Tight on 80GB — drop `moe_r` (8/4) or use
`--no_moe_lora` or shorter packed bins if you OOM. FA varlen is O(S) (no S² score like gemma-4), so
long context is far cheaper here.

## FSDP2 sharding

`fully_shard` each `MiniMaxM2DecoderLayer` then the root. `MixedPrecisionPolicy(param_dtype=None,
reduce_dtype=fp32)` — **param_dtype MUST stay None** so FSDP does not cast the frozen FP8 weights to
bf16 (that would defeat the block-scale dequant); every param keeps its storage dtype (fp8 frozen,
bf16 LoRA/norms, fp32 scales) and only gradient reduction is fp32. `--cpu_offload` adds
`CPUOffloadPolicy` for tight VRAM (slow). `init_process_group("cpu:gloo,cuda:nccl")` so the
LoRA-checkpoint `full_tensor()` all-gather works even under CPU offload.

## Logits parity (`compare_logits.py`) — POD-VERIFIED 2026-06-19 on 8× H100

Always compare logits against a trusted reference when you touch a custom forward (the rule lives
in `../CLAUDE.md`). `compare_logits.py` does this for `lora.py` on the **real 230B model**:

- **The stock FP8 forward is NOT a usable reference — it returns NaN.** transformers 5.5.0's native
  MiniMax-M2 FP8 *inference* kernels (FP8Linear / FP8Experts w8a8) produce **NaN logits** on this
  stack: the hidden state is finite through the embeddings but goes **NaN around decoder layer ~17**
  (argmax collapses to token 0, `cosine=nan`). It is **not** an attention issue — both paths use
  `flash_attention_3`; only the fp8 compute kernel NaNs. Our differentiable **bf16-dequant path does
  NOT NaN** and produces correct text (`"The capital of France is"` → ` Paris`). So our QLoRA-style
  dequant is *more* numerically robust than the stock fp8 inference path here.
- Therefore `compare_logits.py` builds an **independent bf16 reference** (a naive per-token /
  per-expert dequant loop, separate code from `lora.py`'s fused grouped path) and checks our
  LoRA(B=0) against it. **Result: top-1 argmax matches on every prompt** (` Paris`, `n`, …),
  **cosine 0.998–0.9997**. Top-5 occasionally swaps a near-tied 4th/5th rank (fused grouped_mm vs
  naive loop accumulate in different order) — that's bf16 noise, so the pass condition is
  **top-1 argmax + cosine > 0.997**, not exact top-k. Plus: all `*_lora_b` are zero at init (no-op),
  and poking a non-zero B *does* move the logits (adapter is wired in, not dead).

Three real bugs this run surfaced + fixed (all now in the code):
- **`run.sh` pinned `kernels` unpinned → 0.15.2, which makes `import transformers` itself crash**
  (`LayerRepository(... version=...)` mandatory; transformers 5.5.0 builds its `_KERNEL_MAPPING`
  with version-less `LayerRepository` at import time; `USE_HUB_KERNELS=NO` does NOT help). Pin
  **`kernels>=0.12.0,<0.13`** (transformers 5.5.0's declared range). Fixed in `run.sh`.
- **`LinearLoRA` created its `lora_a`/`lora_b` on CPU** (`nn.Linear` default device) → `mat2 is on
  cpu` whenever the base is already on GPU at apply time (`device_map="auto"` in compare/merge).
  Fixed: create adapters on `base.weight.device` (under FSDP the base is on CPU at apply time, so
  still correct — `fully_shard` moves them onto the mesh afterwards).
- **`dequantize_fp8_blockwise` upcast a whole 256-expert tensor to fp32 + an fp32 product (~36GB
  transient per MoE layer) → OOM** in a full per-layer (non-FSDP) forward. Fixed: dequant stacked
  expert weights in **expert-chunks** into a preallocated bf16 buffer (bit-identical; bounds the
  fp32 transient; `MINIMAX_DEQUANT_EXPERT_CHUNK`, default 16). Only matters for the full-model paths
  (compare/merge) — under FSDP each rank holds 1/world_size of the experts.

Running it (the model is FP8 230GB; the differentiable forward dequants a layer's experts to bf16
on the fly, so device_map must leave headroom): `compare_logits.py` loads with `device_map="auto"`
**capped to `MINIMAX_PER_GPU_GIB` (default 40GiB/GPU)** so the bf16 dequant transient fits — at the
default greedy cap a busy GPU packs ~72GB and OOMs on the transient. Use `MINIMAX_RUN_NATIVE=0` to
skip the (NaN) native pass. ⚠ This used **`device_map` pipeline-parallel, NOT FSDP** — so the FSDP2
fp8 all-gather risk (#1 below) is still unverified.

## Status — what's verified vs what needs the pod

**Verified locally on CPU (`autotrain/.venv`, torch 2.12.1 + transformers 5.5.0):**
- `test_lora.py` (run it; it's also `run.sh`'s pre-flight): fused grouped LoRA-MoE forward matches
  an independent per-expert/per-token reference to ~1e-7, **LoRA gradients match to ~1e-7, and the
  frozen base params receive no grad**; zero-B adapter reduces to the pure base MoE; `LinearLoRA`
  zero-B is a no-op and the non-zero case equals `base + scaling·B(A(x))`; blockwise FP8 dequant
  round-trips within fp8 error.
- `pack_dataset.py`: MiniMax-M2 tokenizer + chat template load; the reasoning-guard relaxation
  renders ALL assistant `<think>` blocks (2 vs the stock 1 on a 2-user trajectory); tools render;
  `reasoning→reasoning_content` mapping works; full-sequence label fallback (no `{% generation %}`).
- `minimax_m2.py` / `lora.py` import + compile; all transformers/FSDP symbols
  (`MiniMaxM2ForCausalLM`, `MiniMaxM2DecoderLayer`, `fully_shard`, `MixedPrecisionPolicy
  (param_dtype=None)`, `experts_implementation`) exist as used.

**Verified on a real 8× H100 pod (2026-06-19, via `compare_logits.py`):** the FP8 230GB model loads
across GPUs (`device_map="auto"`), our `lora.py` differentiable dequant forward (LinearLoRA +
fused expert LoRA) runs end-to-end and its **logits match an independent bf16 reference (top-1 +
cosine>0.997)**; the stock FP8 inference forward NaNs (see the parity section above). This exercises
the same dequant path `merge_infer.py` uses — but via `device_map`, **not FSDP**.

**NOT yet run on hardware (validate on the pod):**
1. **FSDP2 all-gather of `float8_e4m3fn` params.** Sharding/gathering fp8 DTensors across ranks is
   the #1 risk. If it errors, fallback options: `device_map="auto"` pipeline-parallel for a frozen
   base (slower, no FSDP), or upcast-to-bf16 + CPU offload (needs ~460GB host RAM).
2. **CPU RAM at load.** Each rank `from_pretrained`s to CPU (~230GB) before sharding → ~230GB ×
   ranks unless the pod has a smarter loader. On a tight box, reduce ranks or stage the load.
3. End-to-end loss-decrease, tok/s, and the FA3 varlen path on the *real* model.
4. `merge_infer.py` generation (uses the dequant path, not the fast fp8 kernels).

Treat the first run as a smoke test: `MINIMAX_DEPS_ONLY=1 bash run.sh`, then
`bash run.sh -- --max_steps 20 --limit_samples 4` before a full epoch.

## RunPod workflow (use `runpodctl`; `RUNPOD_API_KEY` in `../.env`)

```bash
runpodctl config --apiKey "$RUNPOD_API_KEY"
runpodctl ssh add-key --key-file ~/.ssh/id_rsa.pub

# 4× H100 SXM, CUDA>=13 host, big disk (230GB model + dataset + deps), auto-terminate safety net
runpodctl pod create --name minimax-m2-ft \
  --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 4 \
  --image runpod/pytorch:1.0.6-cu1300-torch291-ubuntu2404 \
  --container-disk-in-gb 600 --cloud-type SECURE \
  --min-cuda-version 13.0 --ssh --ports "22/tcp" \
  --terminate-after "$(date -u -d '+8 hours' +%Y-%m-%dT%H:%M:%SZ)"

runpodctl ssh info <pod-id>        # ssh string ({"error":"pod not ready"} while booting)
scp -P <port> -i ~/.ssh/id_rsa *.py *.sh root@<ip>:/workspace/autotrain-mm2/
# on the pod:  cd /workspace/autotrain-mm2 && bash run.sh
runpodctl pod delete <pod-id>      # ⚠ billing stops only here. 4× H100 SXM ≈ $13/hr.
```

Gotchas (mostly inherited from gemma4; all real):
- **GPU id for H100 SXM = `NVIDIA H100 80GB HBM3`**; use the modern `runpodctl pod create` (has
  `--min-cuda-version`/`--gpu-id`/`--terminate-after`), not the deprecated `create pod`.
- `--min-cuda-version 13.0` guarantees a cu130 host for the torch-2.12+cu130 / FA3-wheel stack.
- Container disk must be large: model ~230GB + dataset + deps. 600GB is safe.
- **NCCL NVLS** bind fails in RunPod containers → `NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0` (in run.sh).
- Version pinning is load-bearing: the FA3 wheel is built for **torch 2.12**, which also provides
  `torch._grouped_mm` for the fused MoE — keep torch 2.12.x. CUDA backend is chosen from the host
  driver (`run.sh`: ≥13.2→cu132, ≥13.0→cu130, ≥12.6→cu126).
