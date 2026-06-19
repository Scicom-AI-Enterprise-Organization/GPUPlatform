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
| `test_lora.py` | CPU correctness test: fused LoRA-MoE fwd+grads vs a per-expert reference |
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

## What is verified vs what needs the pod

**Verified locally (CPU, torch 2.12 + transformers 5.5.0):** the fused grouped LoRA-MoE forward
**and gradients** vs an independent per-expert reference (`test_lora.py`, ~1e-7); the FP8 block
dequant; the chat-template reasoning relaxation + packing invariants; imports/symbols.

**Needs validation on a 4× H100 pod (no such hardware available here):** FSDP2 all-gather of FP8
params, CPU RAM at load (~230GB×ranks), end-to-end loss/throughput, and the FA3 varlen path on the
real model. See **CLAUDE.md → "Status"** for the full list and fallbacks. Treat the first pod run
as a smoke test (`MINIMAX_DEPS_ONLY=1`, then `--max_steps 20 --limit_samples 4`).

## Deploy a pod from scratch

See **CLAUDE.md → "RunPod workflow"** (4× H100 SXM, cu130 host, 600GB disk, ~$13/hr — `runpodctl
pod delete` when done; billing stops only on delete).
