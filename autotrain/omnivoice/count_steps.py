#!/usr/bin/env python3
"""Count steps-per-epoch for an OmniVoice finetune.

The OmniVoice trainer is *step*-based (`omnivoice/training/trainer.py` loops
`while global_step < config.steps`).  To train for exactly N epochs we set
`steps = N * steps_per_epoch`.  Because batching is dynamic token-packing
(`PackingIterableDataset`, ~`batch_tokens` per batch), steps-per-epoch isn't
known a priori — so we build the *real* train dataloader (same config the
trainer uses, single process, no DDP) and count the batches in one full pass.

Prints `STEPS_PER_EPOCH=<n>` on the last line for the caller (run.sh) to grep.

Usage:
    python count_steps.py --train_config train_config.json --data_config data_config.json
"""
import argparse
import sys

import torch
from transformers import AutoTokenizer

from omnivoice.models.omnivoice import _resolve_model_path
from omnivoice.training.builder import build_dataloaders
from omnivoice.training.config import TrainingConfig

# Mirrors omnivoice/training/builder.py:build_model_and_tokenizer
NEW_TOKENS = [
    "<|denoise|>", "<|lang_start|>", "<|lang_end|>", "<|instruct_start|>",
    "<|instruct_end|>", "<|text_start|>", "<|text_end|>",
]


def build_tokenizer(config):
    path = config.init_from_checkpoint or config.llm_name_or_path
    tok = AutoTokenizer.from_pretrained(_resolve_model_path(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    add = [t for t in NEW_TOKENS if t not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_config", required=True)
    ap.add_argument("--data_config", required=True)
    ap.add_argument("--epoch", type=int, default=0, help="epoch seed to count under")
    args = ap.parse_args()

    config = TrainingConfig.from_json(args.train_config)
    config.data_config = args.data_config
    config.output_dir = "/tmp/_count_steps"

    tok = build_tokenizer(config)
    train_loader, _ = build_dataloaders(config, tok)

    # Reproduce the trainer's per-epoch shuffle seed so the packed batch count
    # matches what training will see on this epoch.
    ds = train_loader.dataset
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(args.epoch)

    n = 0
    for _ in train_loader:
        n += 1
    print(f"[count_steps] attn={config.attn_implementation} batch_tokens={config.batch_tokens} "
          f"num_workers={config.num_workers} epoch={args.epoch}", file=sys.stderr, flush=True)
    print(f"STEPS_PER_EPOCH={n}")


if __name__ == "__main__":
    main()
