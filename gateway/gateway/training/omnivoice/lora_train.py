#!/usr/bin/env python3
"""LoRA training entrypoint for OmniVoice — a drop-in alternative to
`accelerate launch -m omnivoice.cli.train` (see that file upstream) that wraps
`model.llm` (the Qwen3 backbone `build_model_and_tokenizer` returns) in a peft
LoRA adapter before training, and merges the adapter back into a plain
checkpoint after `trainer.train()` finishes — mirrors qwen3_tts_flash.py's own
use_lora/merge_and_unload contract (../tts/qwen3_tts_flash.py) so downstream
eval/serving (make_clean_checkpoint.py, omnivoice.cli.infer_batch,
OmniVoice.from_pretrained) sees an ordinary full-finetune-shaped checkpoint
with no peft dependency.

Everything else — data loading, the training loop, checkpoint cadence, the
`@@STEP`-parseable console log — stays exactly OmniVoice's own public
`build_model_and_tokenizer` / `build_dataloaders` / `OmniTrainer` (this file
only adds the LoRA wrap + final merge around those same calls, so it tracks
upstream OmniVoice automatically instead of forking its trainer).

OmniVoice's own TTS-specific heads — `audio_embeddings` (the per-codebook
audio-token embedding) and `audio_heads` (the audio-token output projection)
— live OUTSIDE `model.llm` (see omnivoice/models/omnivoice.py `OmniVoice.
__init__`) and are left FULLY trainable; only the pretrained Qwen3 backbone is
adapter-wrapped. `task_type=None` on the LoraConfig gets peft's generic
`PeftModel` (not `PeftModelForCausalLM`) — `model.llm` is a bare
`AutoModel`-loaded backbone (no lm_head of its own; OmniVoice's forward()
calls it directly with custom kwargs, not the standard causal-LM signature),
so the causal-LM-specific wrapper doesn't apply here.

Usage (mirrors omnivoice.cli.train, plus --lora_*):
    accelerate launch --gpu_ids 0 --num_processes 1 \\
        lora_train.py --train_config train_config.json \\
        --data_config data_config.json --output_dir output/ \\
        --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \\
        --lora_target_modules all-linear
"""
import argparse
import os

from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
from omnivoice.training.config import TrainingConfig
from omnivoice.training.trainer import OmniTrainer


def main():
    parser = argparse.ArgumentParser(description="OmniVoice LoRA training entry point")
    parser.add_argument("--train_config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save checkpoints")
    parser.add_argument("--data_config", type=str, required=True, help="Path to data config JSON")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="all-linear")
    args = parser.parse_args()

    # 1. Load Configuration (same as omnivoice.cli.train)
    config = TrainingConfig.from_json(args.train_config)
    config.output_dir = args.output_dir
    config.data_config = args.data_config

    # 2. Build Components, then wrap the LLM backbone in LoRA before dataloaders/
    #    trainer are built (dataloaders don't touch the model; order doesn't matter
    #    for them, but the wrap must happen before OmniTrainer builds the optimizer).
    model, tokenizer = build_model_and_tokenizer(config)

    from peft import LoraConfig, get_peft_model

    tgt = (args.lora_target_modules or "all-linear").strip()
    target_modules = (
        "all-linear" if tgt in ("", "all-linear", "all")
        else [t.strip() for t in tgt.split(",") if t.strip()]
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=None,  # generic PeftModel — model.llm has no lm_head/generate() of its own
        target_modules=target_modules,
    )
    model.llm = get_peft_model(model.llm, peft_config)
    model.llm.print_trainable_parameters()

    train_loader, eval_loader = build_dataloaders(config, tokenizer)

    # 3. Initialize Trainer and Start (same as omnivoice.cli.train)
    trainer = OmniTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        tokenizer=tokenizer,
    )
    trainer.train()

    # 4. Merge the adapter into the base weights and OVERWRITE the trainer's own
    #    final checkpoint dir (train() already ran its last save_checkpoint(global_step)
    #    in un-merged peft format — same path, same name) with a plain, peft-free
    #    state dict. merge_and_unload() is destructive, but training is done so that's
    #    fine here (unlike a mid-training checkpoint, which we never touch/use — the
    #    gateway orchestrator only reads the highest-numbered checkpoint-* dir).
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    unwrapped.llm = unwrapped.llm.merge_and_unload()
    final_dir = os.path.join(config.output_dir, f"checkpoint-{trainer.global_step}")
    if trainer.accelerator.is_main_process:
        unwrapped.save_pretrained(final_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
