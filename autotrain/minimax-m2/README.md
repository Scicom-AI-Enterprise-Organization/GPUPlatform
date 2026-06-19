# MiniMax-M2 230B FP8 MoE — LoRA finetuning

LoRA finetune of **`MiniMaxAI/MiniMax-M2`** (230B total / 10B active, 256-expert MoE, FP8) on a
packed chat dataset, sharded across **4× H100** with PyTorch **FSDP2** + activation checkpointing.
The MiniMax-M2 sibling of the gemma-4 job in `../`. Designed to run on a fresh RunPod pod via
[`runpodctl`](https://github.com/runpod/runpodctl).

The interesting part is **LoRA on a frozen FP8 MoE**:
- MiniMax-M2's attention is a uniform head_dim 128 (every layer fits FlashAttention), so unlike
  gemma-4 there is **no custom attention** — stock `flash_attention_3` + varlen packing.
- The released weights are **FP8** and transformers' FP8 kernels are **inference-only /
  non-differentiable**. We keep the base frozen in FP8 and, for training, run each frozen matmul
  through a **differentiable on-the-fly dequant → fused bf16 `grouped_mm`** (QLoRA-style), with
  activation checkpointing keeping the transient bf16 weights cheap.
- LoRA adapts **attention q/k/v/o + the MoE expert FFNs** (`gate_up_proj`, `down_proj`), the
  expert LoRA folded into the fused grouped-MoE forward.

## Files

| file | what |
|------|------|
| `minimax_m2.py` | training entrypoint — run with `torchrun` (FSDP2, packed varlen, Liger FLCE) |
| `lora.py` | `LinearLoRA` (q/k/v/o) + fused grouped-MoE expert LoRA + FP8 block dequant |
| `dequant_triton.py` | Triton blockwise FP8→bf16 dequant (autograd) — `lora.py`'s CUDA fast path |
| `bench_dequant.py` | Triton-vs-PyTorch dequant correctness + speed (fwd & bwd), random fp8 weights |
| `test_lora.py` | CPU correctness test: fused LoRA-MoE fwd+grads vs a per-expert reference |
| `compare_logits.py` | GPU logits check on the real model: LoRA(B=0) vs an independent bf16 reference |
| `pack_dataset.py` | build the multipacked ChiniDataset from a chat parquet (MiniMax-M2 template) |
| `merge_infer.py` | attach the trained LoRA to the FP8 base and generate |
| `run.sh` | one-shot pod bootstrap: deps → LoRA test → pack/download → train |
| `CLAUDE.md` | design notes, the FP8/MoE gotchas, the runpodctl workflow, verified-vs-TODO |

## Quickstart (on a 4× H100 pod)

```bash
cd /workspace/autotrain-mm2
export HF_TOKEN=hf_...                  # only if your account needs it for the download

# 1) install deps + run the LoRA correctness test only (cheap, CPU)
MINIMAX_DEPS_ONLY=1 bash run.sh

# 2) short run -> a LoRA checkpoint fast (smoke test before a full epoch)
bash run.sh -- --max_steps 20 --limit_samples 4

# 3) full bootstrap + training (packs the dataset, downloads the 230GB FP8 model, torchrun on 4 GPUs)
bash run.sh

# 4) attach the trained adapters to the FP8 base and generate
python merge_infer.py --prompt "Write a function to reverse a linked list." --max-new-tokens 128
```

`run.sh` is idempotent — it re-checks deps, reuses `./packed_data` and the HF cache.

### CLI flags (`minimax_m2.py`)

| flag | default | meaning |
|------|---------|---------|
| `--attn_r` / `--attn_alpha` | 16 / 16 | LoRA rank/alpha for attention q/k/v/o (scaling = alpha/r) |
| `--moe_r` / `--moe_alpha` | 16 / 16 | LoRA rank/alpha for the MoE expert FFNs (the dominant params) |
| `--no_moe_lora` | off | adapt attention only (skip the 256-expert FFNs) |
| `--lr` | 1e-5 | AdamW LR (gemma4 lesson: 1e-5..5e-5, scaling ≤1, few epochs) |
| `--batch_size` | 1 | packed bins per step (keep 1; the collator concatenates the batch) |
| `--max_epochs` / `--max_steps` | 1 / 0 | epochs / hard step cap (0 = all epochs) |
| `--limit_samples` | 0 | cap dataset to first N bins (0 = all) |
| `--cpu_offload` | off | FSDP2 CPUOffloadPolicy for tight VRAM (slow) |
| `--checkpointing_step` | 100 | save LoRA adapters every N steps |
| `--wandb` / `--wandb_project` | off | log loss/lr/tps to Weights & Biases |

Checkpoints (LoRA only) land in `checkpointing/lora.pt` + `checkpointing/lora_meta.json`.

## Dataset (multipacking)

`pack_dataset.py` renders each conversation through the **MiniMax-M2** chat template (tools +
`<think>` reasoning), tokenizes, and greedily packs whole conversations into `--max-seq-len`
(default 131072 = 128k) token bins. Conversations are never split. It relaxes the template's
reasoning guard so EVERY assistant turn's reasoning trains (`--native-reasoning` for stock), maps
each assistant turn's `reasoning`→`reasoning_content`, and falls back to `labels = input_ids`
(the template has no `{% generation %}` block).

```bash
python pack_dataset.py --out ./packed_data                      # default Function-Call-TaaS parquet
python pack_dataset.py --repo <repo> --file <parquet> --max-rows 10   # any chat parquet / quick test
```

## Triton FP8 dequant — speed (`dequant_triton.py`)

The block-scaled FP8→bf16 dequant runs on every frozen attn proj + every MoE expert tensor, every
layer, every step — so `lora.py` dispatches it to a Triton kernel on CUDA (fuses upcast·scale·
downcast in one pass, one scale load per 128×128 block; env `MINIMAX_DEQUANT_TRITON=0` to force the
PyTorch path; CPU / triton-less boxes fall back automatically). Backward **is** supported as a
`torch.autograd.Function`: the FP8 weight is non-differentiable → grad `None`; `scale_inv` gets an
analytic grad. Reproduce with `python bench_dequant.py`.

**Microbenchmark — 1× H100, Triton vs PyTorch (`bench_dequant.py`):**

| tensor | shape | forward | fwd **speedup** | fwd+bwd | fwd+bwd speedup | correctness (vs PyTorch) |
|--------|-------|---------|-----------------|---------|-----------------|--------------------------|
| attn proj (2D) | `(3072, 3072)` | 23.7µs / 89.8µs | **3.8×** | 306µs / 367µs | 1.2× | fwd **bit-exact**, bwd rel 8.8e-8 |
| MoE `gate_up` | `(256, 3072, 3072)` | 2.68ms / 23.8ms | **8.9×** | 44.3ms / 142.3ms | 3.2× | fwd **bit-exact**, bwd rel 1.5e-7 |
| MoE `down` | `(256, 3072, 1536)` | 1.33ms / 12.0ms | **9.0×** | 22.2ms / 71.6ms | 3.2× | fwd **bit-exact**, bwd rel 1.0e-7 |

(times are `triton / pytorch`; forward is bit-exact so loss is unchanged. backward grad is w.r.t.
`scale_inv` — matches PyTorch autograd to ~1e-7.)

**End-to-end** (full 32k finetune, 8× H100): **~12.4k tok/s** with Triton vs **~10.3k** with the
PyTorch dequant → **~20% higher training throughput**, identical loss curve.

## What is verified vs what needs the pod

**Verified locally (CPU):** the fused grouped LoRA-MoE forward **and gradients** vs an independent
per-expert reference (`test_lora.py`, ~1e-7); the FP8 block dequant; chat-template packing.

**Verified on an 8× H100 pod (2026-06-19):** the full **32k finetune runs end-to-end** — FSDP2
all-gather of FP8 params works, loss decreases (7.4 → ~5.1, lr 5e-5, 3 epochs), LoRA pushed to
[`huseinzolkepliscicom/minimax-m2-funccall-lora-32k`](https://huggingface.co/huseinzolkepliscicom/minimax-m2-funccall-lora-32k).
`compare_logits.py` matched our forward to an independent bf16 reference (top-1 + cosine>0.997;
the stock FP8 *inference* forward NaNs). The Triton dequant gave ~20% more tok/s with no loss
change. `--low_cpu_shard_load` (meta-init + rank-0 broadcast) caps load-time CPU at ~one model
copy (265GB) instead of ~230GB×ranks. See **CLAUDE.md → "Status"** for details. First pod run is
still best treated as a smoke test (`MINIMAX_DEPS_ONLY=1`, then `--max_steps 20 --limit_samples 4`).

## Deploy a pod from scratch

See **CLAUDE.md → "RunPod workflow"** (4× H100 SXM, cu130 host, 600GB disk, ~$13/hr — `runpodctl
pod delete` when done; billing stops only on delete).
