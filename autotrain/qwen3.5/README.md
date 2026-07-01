# Qwen3.6 LoRA finetune — dense + MoE (autotrain)

Standalone FSDP2 LoRA finetune of the **Qwen3.6** family on a packed function-calling dataset, run on
the tm 8× H20-3e box. Sibling of `../gemma4`, `../minimax-m2`, `../mistral-small`.

Qwen3.6 reuses the **Qwen3.5 architecture** (`model_type` `qwen3_5` / `qwen3_5_moe`), so one job dir
trains both variants — `qwen3_5.py` auto-detects dense vs MoE from the config:

| model | kind | class | notes |
|-------|------|-------|-------|
| `Qwen/Qwen3.6-27B`     | dense | `Qwen3_5ForConditionalGeneration`    | 64 layers, GatedDeltaNet hybrid |
| `Qwen/Qwen3.6-35B-A3B` | MoE   | `Qwen3_5MoeForConditionalGeneration` | 40 layers, 256 experts / 8 active, ~3B active |

Both are GatedDeltaNet hybrids; LoRA wraps only the attention `q/k/v/o_proj`, so the MoE experts are
untouched and the same training path serves both.

## Files

| file | what |
|------|------|
| `qwen3_5.py`      | torchrun trainer — FSDP2 + CPU offload, custom `LinearLoRA` on q/k/v/o, Liger FLCE loss, FlashQLA GatedDeltaNet kernel. `--model_id` selects dense/MoE (auto-detected); `--checkpoint_dir` per model |
| `pack_dataset.py` | render the Qwen3.6 chat template (`preserve_thinking=True` → all turns' reasoning) + multipack a HF chat-parquet → ChiniDataset |
| `run.sh`          | bootstrap: uv venv + deps + dataset pack (Qwen3.6 tokenizer) + `torchrun qwen3_5.py` |
| `one_script_run.sh` | original single-file version (kept for history) |

## Quickstart (tm box, port 1023)

```bash
ssh -i ../../scicom -p 1023 root@8.222.165.68
# scp is flaky on this box — pipe via stdin instead:
for f in *.py *.sh; do ssh -i ../../scicom -p 1023 root@8.222.165.68 "cat > /share/autotrain-qwen3.5/$f" < "$f"; done

cd /share/autotrain-qwen3.5
export HF_TOKEN=$(cat /share/huggingface/token) HF_HOME=/share/huggingface HF_HUB_DISABLE_XET=1
# deps only first (FlashQLA / causal_conv1d build from source):
VENV_PATH=/share/qwen3.5-venv QWEN_DEPS_ONLY=1 bash run.sh
# full run of the dense model (default MODEL_ID=Qwen/Qwen3.6-27B):
CUDA_VISIBLE_DEVICES=0,1,2,3 VENV_PATH=/share/qwen3.5-venv bash run.sh --lr 1e-4 --max_epochs 3
# ...or the MoE model:
MODEL_ID=Qwen/Qwen3.6-35B-A3B CUDA_VISIBLE_DEVICES=4,5,6,7 VENV_PATH=/share/qwen3.5-venv \
  bash run.sh --lr 1e-4 --max_epochs 3
```

Everything lives under **`/share`** (job dir `/share/autotrain-qwen3.5`, venv `/share/qwen3.5-venv`,
HF cache `/share/huggingface`) — the box's `/` is small. See `CLAUDE.md` for the design (the
GatedDeltaNet hybrid attention, the dense/MoE class resolution, the custom LoRA/FSDP shape, the
chat-template reasoning relaxation) and the gotchas.

## What it produces

Per-model `checkpointing-<model-slug>/lora.pt` + `lora_meta.json` (adapter weights only; the base is
never saved). Merge/serve as for the sibling jobs (a `merge_infer.py` is not yet ported here).
