"""Shared scaffolding for the LLM finetune trainers (gemma4 / qwen3_5 / minimax_m2 /
mistral_small). Those trainers differ only in the model class, its decoder-layer type,
the LoRA layout, and — for the FP8 MoE pair — a few dequant-specific steps. Everything
else (the CLI surface and the FSDP2 sharding/checkpointing) is identical, so it lives
here ONCE. Adding a new model, a new common flag, or changing the FSDP/offload policy
is then a single-file edit instead of four.

Importable by bare name from every trainer because llm_finetune ships this dir with
`PYTHONPATH=<llm dir>` (so both the root gemma4.py and the subdir trainers see it).

What stays per-trainer (legitimately arch-specific, NOT forced in here):
  - the LoRA CLI (gemma/qwen: --lora_r/--alpha/--target_modules; MoE: --attn_r/--moe_r…),
  - param_dtype (bf16 for dense; None for FP8 MoE so FSDP doesn't cast the fp8 weights),
  - the FP8 MoE steps interleaved with sharding (_promote_scalar_params, low-CPU load).
"""
from __future__ import annotations

from functools import partial

import torch
import torch.distributed.fsdp as fsdp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


def add_common_args(parser, *, lr_default: float = 1e-4, wandb_project: str = "autotrain") -> None:
    """Register the CLI flags EVERY LLM trainer shares. Call this from each trainer's
    argparse; add arch-specific flags (LoRA dims, model id, data/out dirs) alongside.
    `dest` names match what the trainers already read (args.batch_size, args.grad_accum,
    args.cpu_offload, …), so wiring a new flag is a one-line change here."""
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Packed bins concatenated into ONE varlen sequence per microbatch "
             "(the collator concatenates them). 1 = one bin per microbatch.")
    parser.add_argument(
        "--grad_accum", type=int, default=1,
        help="Microbatches to accumulate before an optimizer step. "
             "Effective batch = batch_size × grad_accum × world_size. 1 = off.")
    parser.add_argument(
        "--cpu_offload", action="store_true",
        help="FSDP2 CPUOffloadPolicy: keep params/optimizer in host RAM — a big VRAM "
             "saver but PCIe-bound (slow). Off by default; enable for tight VRAM / "
             "long context that won't otherwise fit.")
    parser.add_argument("--max_epochs", type=int, default=1, help="Epochs over the dataset.")
    parser.add_argument(
        "--max_steps", type=int, default=0,
        help="Stop after this many OPTIMIZER steps (0 = run the full epochs).")
    parser.add_argument(
        "--checkpointing_step", type=int, default=100,
        help="Save the LoRA adapters every N optimizer steps.")
    parser.add_argument(
        "--limit_samples", type=int, default=0,
        help="Cap the dataset to the first N packed bins (0 = all).")
    parser.add_argument("--lr", type=float, default=lr_default, help="AdamW learning rate.")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    parser.add_argument("--wandb_project", default=wandb_project, help="wandb project name.")


def fsdp_kwargs(mesh, *, param_dtype, cpu_offload: bool) -> dict:
    """The FSDP2 `fully_shard` kwargs shared by every trainer: the mixed-precision
    policy (param_dtype=bf16 for dense; None for FP8 MoE so the fp8 weights are NOT
    cast), fp32 gradient reduction, the device mesh, and — only when requested — CPU
    offload. THIS is the one place the cpu_offload / param_dtype policy lives."""
    kw: dict = {
        "mp_policy": fsdp.MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32),
        "mesh": mesh,
    }
    if cpu_offload:
        kw["offload_policy"] = fsdp.CPUOffloadPolicy()
    return kw


def shard_layers(model, decoder_classes, kw: dict, *, reshard_after_forward=None) -> None:
    """`fully_shard` each decoder layer at layer granularity, then the root module.
    `decoder_classes` is a class or tuple of classes (dense vs MoE decoder types)."""
    classes = decoder_classes if isinstance(decoder_classes, tuple) else (decoder_classes,)
    layer_kw = dict(kw)
    if reshard_after_forward is not None:
        layer_kw["reshard_after_forward"] = reshard_after_forward
    for module in model.modules():
        if isinstance(module, classes):
            fsdp.fully_shard(module, **layer_kw)
    fsdp.fully_shard(model, **kw)  # root


def checkpoint_layers(model, decoder_classes) -> None:
    """Non-reentrant activation checkpointing on each decoder layer. Activation memory
    is O(seq × layers) — the long-context wall — so recompute-in-backward trades it for
    time; REQUIRED for the FP8 dequant memory trick on the MoE trainers."""
    classes = decoder_classes if isinstance(decoder_classes, tuple) else (decoder_classes,)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, classes),
    )
