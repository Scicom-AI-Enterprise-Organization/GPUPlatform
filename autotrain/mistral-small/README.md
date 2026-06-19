# Mistral-Small-4-119B FP8 MoE — LoRA finetuning

LoRA finetune of the **text** model of **`mistralai/Mistral-Small-4-119B-2603`** (a
`Mistral3ForConditionalGeneration` multimodal model whose language model is a 119B-total
`mistral4` MoE: 128 routed experts top-4 + 1 shared expert, 36 layers, **MLA** attention, FP8)
on a packed chat dataset, sharded across **4× H100** with PyTorch **FSDP2** + activation
checkpointing. The Mistral-Small-4 sibling of `../minimax-m2`. Designed to run on a fresh RunPod
pod via [`runpodctl`](https://github.com/runpod/runpodctl).

The interesting part is **LoRA on a frozen FP8 MoE with MLA attention**:
- The text model's attention head_dim is a uniform 128 (MLA, but qk_head_dim == v_head_dim), so
  every layer fits FlashAttention — stock `flash_attention_3` + varlen packing, **no custom attn**.
- The released weights are **FP8 (per-tensor)** and transformers' FP8 kernels are **inference-only
  / non-differentiable** (the fused experts dispatches even refuse `activation_scheme="static"`).
  We keep the base frozen in FP8 and, for training, run each frozen matmul through a
  **differentiable on-the-fly dequant → bf16 matmul / fused `grouped_mm`** (QLoRA-style), with
  activation checkpointing keeping the transient bf16 weights cheap.
- LoRA adapts the **MLA projections** (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`,
  `o_proj`), the **routed MoE experts** (`gate_up_proj`, `down_proj`, fused), and the **shared
  expert MLP** (`gate_proj`, `up_proj`, `down_proj`).
- Multimodal wrapper: we load the full model, freeze the vision tower + projector, and run the
  text-only forward through `model.language_model`.

## Files

| file | what |
|------|------|
| `mistral_small.py` | training entrypoint — run with `torchrun` (FSDP2, packed varlen, Liger FLCE) |
| `lora.py` | `LinearLoRA` (MLA + shared MLP) + fused grouped-MoE routed-expert LoRA + per-tensor FP8 dequant |
| `dequant_triton.py` | fused Triton per-tensor FP8→bf16 dequant (~7–13×; `MISTRAL_DEQUANT_TRITON=1`) |
| `bench_dequant.py` | GPU benchmark: Triton vs torch dequant — fwd/bwd parity + speed + peak memory |
| `test_lora.py` | CPU correctness: dequant (per-tensor + block), LinearLoRA, fused LoRA-MoE fwd+grads |
| `compare_logits.py` | GPU parity: our LoRA(B=0) vs an independent bf16 dequant reference |
| `pack_dataset.py` | build the multipacked ChiniDataset from a chat parquet (Mistral-Small-4 template) |
| `merge_infer.py` | attach the trained LoRA to the FP8 base and generate |
| `run.sh` | one-shot pod bootstrap: deps → LoRA test → pack/download → train |
| `CLAUDE.md` | design notes, the FP8/MoE/MLA gotchas, the load-bug workaround, verified-vs-TODO |

## Quickstart (on a 4× H100 pod)

```bash
cd /root/autotrain-mistral
export HF_TOKEN=hf_...                   # gated download (canonical: autotrain/.env)

# 1) install deps + run the LoRA correctness test only (cheap, CPU)
MISTRAL_DEPS_ONLY=1 bash run.sh

# 2) MANDATORY before training: logits parity on the real model (LoRA(B=0) vs bf16 ref)
python compare_logits.py

# 3) short run -> a LoRA checkpoint fast (smoke test before a full epoch)
bash run.sh -- --max_steps 20 --limit_samples 4

# 4) full bootstrap + training (packs the dataset, downloads the ~119GB FP8 model, torchrun on 4 GPUs)
bash run.sh

# 5) attach the trained adapters to the FP8 base and generate
python merge_infer.py --prompt "Write a function to reverse a linked list." --max-new-tokens 128
```

`run.sh` is idempotent — it re-checks deps, reuses `./packed_data` and the HF cache.

### CLI flags (`mistral_small.py`)

| flag | default | meaning |
|------|---------|---------|
| `--attn_r` / `--attn_alpha` | 16 / 16 | LoRA rank/alpha for the MLA projections (scaling = alpha/r) |
| `--moe_r` / `--moe_alpha` | 16 / 16 | LoRA rank/alpha for the routed + shared expert FFNs (dominant params) |
| `--no_moe_lora` | off | adapt attention only (skip all experts) |
| `--no_shared_lora` | off | keep routed-expert LoRA but skip the shared-expert MLP |
| `--lr` | 1e-5 | AdamW LR (lesson: 1e-5..5e-5, scaling ≤1, few epochs) |
| `--batch_size` | 1 | packed bins per step (keep 1; the collator concatenates the batch) |
| `--max_epochs` / `--max_steps` | 1 / 0 | epochs / hard step cap (0 = all epochs) |
| `--limit_samples` | 0 | cap dataset to first N bins (0 = all) |
| `--low_cpu_shard_load` | off | meta-init + stream base weights from rank 0 (caps CPU RAM; unverified on HW) |
| `--cpu_offload` | off | FSDP2 CPUOffloadPolicy for tight VRAM (slow) |
| `--checkpointing_step` | 100 | save LoRA adapters every N steps |
| `--wandb` / `--wandb_project` | off | log loss/lr/tps to Weights & Biases |

Checkpoints (LoRA only) land in `checkpointing/lora.pt` + `checkpointing/lora_meta.json`.

## Triton FP8 dequant — Triton vs PyTorch (measured on H20-3e, 2026-06-19)

`dequant_triton.py` fuses the per-tensor FP8→bf16 dequant into a single pass (no fp32 transient);
`bench_dequant.py` benchmarks it against the torch reference (`lora.dequantize_fp8`). Enable with
`MISTRAL_DEQUANT_TRITON=1` (default in `run.sh`).

**Correctness — bit-identical (max abs diff `0.0e+00`):**

| check | result |
|---|---|
| forward dequant (2D `o_proj`, 2D `kv_b`, 3D experts) | **0.0e+00** (bit-identical) |
| LinearLoRA fwd+bwd: output, `d/dx`, `d/d lora_a`, `d/d lora_b` | **0.0e+00** (bit-identical) |

**Speed + peak memory** (mean over 50 iters, CUDA events):

| weight (per-tensor FP8) | elems | torch | Triton | speedup | peak mem |
|---|---|---|---|---|---|
| 2D `o_proj` 4096×4096 | 17M | 0.183 ms / 0.29 GB | 0.025 ms / 0.18 GB | **7.4×** | 1.55× less |
| 3D experts `gate_up` (128, 4096, 4096) | 2147M | 24.696 ms / 17.25 GB | 1.894 ms / 15.10 GB | **13.0×** | 1.14× less |
| 3D experts `down` (128, 4096, 2048) | 1074M | 12.408 ms / 8.66 GB | 0.921 ms / 7.58 GB | **13.5×** | 1.14× less |

The frozen base needs no gradient (QLoRA), so this is a plain forward op; gradients flow through
`F.linear` to the activations / adapters. Reproduce: `python bench_dequant.py`.

## Dataset (multipacking)

`pack_dataset.py` renders each conversation through the **Mistral-Small-4** chat template (tools +
`[THINK]` reasoning), tokenizes, and greedily packs whole conversations into `--max-seq-len`
(default 131072 = 128k) token bins. Conversations are never split. It folds each assistant turn's
flat `reasoning` field into a `[THINK]` content block so reasoning trains (`--native-reasoning` for
stock), normalises tool calls, and falls back to `labels = input_ids` (the template has no
`{% generation %}` block).

```bash
python pack_dataset.py --out ./packed_data                      # default Function-Call-TaaS parquet
python pack_dataset.py --repo <repo> --file <parquet> --max-rows 10   # any chat parquet / quick test
```

## What is verified

**On real hardware (4× H20-3e, the `tm` VM, 2026-06-19):** `compare_logits.py` on the real 119B
model PASSED (LoRA(B=0) matches an independent bf16 ref top-1 + cosine 0.999, "...France is" →
` Paris`); **FSDP2 fp8 all-gather works** (after reshaping per-tensor scalar scales to `(1,)`);
end-to-end FSDP2 LoRA training ran 3 epochs, **loss 0.95 → ~0.50**, ~5k tok/s with the Triton
dequant; the Triton dequant is **bit-identical** to torch on fwd+bwd and **7–13×** faster
(`bench_dequant.py`). LoRA saved to `checkpointing/lora.pt`.

**Locally (CPU, torch 2.12 + transformers 5.5.0):** the fused grouped LoRA-MoE forward **and
gradients** vs an independent per-expert reference (`test_lora.py`, ~1e-7); per-tensor + block FP8
dequant; end-to-end B=0 no-op + LoRA wiring on a tiny real `Mistral3ForConditionalGeneration`.

**Still not verified:** `merge_infer.py` generation on the real model; `--low_cpu_shard_load`. See
**CLAUDE.md → "Status"** + **"H20 node"** for the exact commands and fallbacks.

## Deploy a pod from scratch

See **CLAUDE.md → "RunPod workflow"** (4× H100 SXM, cu130 host, 400GB disk, US region, ~$13/hr —
`runpodctl pod delete` when done; billing stops only on delete).
