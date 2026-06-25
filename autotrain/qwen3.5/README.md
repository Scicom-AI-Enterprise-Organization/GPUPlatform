# Qwen3.5-27B LoRA finetune (autotrain)

Standalone FSDP2 LoRA finetune of `Qwen/Qwen3.5-27B` on a packed function-calling dataset, run on
the tm 8× H20-3e box. Sibling of `../gemma4`, `../minimax-m2`, `../mistral-small`.

## Files

| file | what |
|------|------|
| `qwen3_5.py`      | torchrun trainer — FSDP2 + CPU offload, custom `LinearLoRA` on q/k/v/o, Liger FLCE loss, FlashQLA GatedDeltaNet kernel |
| `pack_dataset.py` | render Qwen3.5 chat template (reasoning-relaxed) + multipack a HF chat-parquet → ChiniDataset |
| `run.sh`          | bootstrap: uv venv + deps + dataset pack + `torchrun qwen3_5.py` |
| `one_script_run.sh` | original single-file version (kept for history) |

## Quickstart (tm box, port 1023)

```bash
ssh -i ../../scicom -p 1023 root@8.222.165.68
scp -i ../../scicom -P 1023 *.py *.sh root@8.222.165.68:/share/autotrain-qwen3.5/

cd /share/autotrain-qwen3.5
export HF_TOKEN=hf_... HF_HOME=/share/huggingface HF_HUB_DISABLE_XET=1
# deps only first (FlashQLA / causal_conv1d build from source):
VENV_PATH=/share/qwen3.5-venv QWEN_DEPS_ONLY=1 bash run.sh
# smoke (pick the GPUs you're given — the box is shared, check nvidia-smi):
CUDA_VISIBLE_DEVICES=0,1,2,3 VENV_PATH=/share/qwen3.5-venv \
  bash run.sh --max_steps 50 --limit_samples 10 --lr 1e-4
```

See `CLAUDE.md` for the design (the GatedDeltaNet hybrid attention, the custom LoRA/FSDP shape, the
chat-template reasoning relaxation) and the gotchas.

## What it produces

`checkpointing/lora.pt` + `checkpointing/lora_meta.json` (adapter weights only; the 27B base is never
saved). Merge/serve as for the sibling jobs (a `merge_infer.py` is not yet ported here).
