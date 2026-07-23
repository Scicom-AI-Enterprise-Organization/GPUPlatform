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
    parser.add_argument(
        "--torch_compile", action="store_true",
        help="Per-block torch.compile(dynamic=True) on the decoder layers (see maybe_torch_compile). "
             "dynamic avoids per-bin-length recompiles; custom attention/SSM kernels graph-break. "
             "Off by default — smoke-test + check the recompiles log before relying on it.")


def unfreeze_embeddings(model, *, rank: int = 0, logger=None) -> int:
    """FULL-train the token embeddings + LM head on top of LoRA (the `--train_embeddings`
    flag), shared by every LLM trainer. Attention-only LoRA can only nudge the output
    distribution via hidden states, so teaching a model to reliably emit a specific special
    token (e.g. gemma's `<|tool_call>`) is far easier when the head/embeddings themselves
    adapt. The loss already projects hidden states through `self.lm_head.weight` (Liger FLCE
    returns its gradient), so unfreezing that tensor trains it from BOTH the input-embed
    lookup and the output projection.

    Handles TIED heads (gemma-4, small qwen — `tie_word_embeddings=True` → ONE weight) and
    UNTIED heads (minimax/mistral, larger qwen — two separate weights). Call AFTER LoRA is
    applied + the base frozen, and BEFORE FSDP sharding (root `fully_shard` then shards the
    now-trainable weight; requires_grad survives to_empty/broadcast on the meta-init path).
    Returns the #params unfrozen. The checkpoint save (every `requires_grad` param via
    `full_tensor()`) then captures the whole weight, and the arch's merge loads/folds it back.
    """
    in_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()
    in_w = getattr(in_emb, "weight", None)
    out_w = getattr(out_emb, "weight", None)
    n = 0
    if in_w is not None:
        in_w.requires_grad_(True)
        n += in_w.numel()
    tied = in_w is not None and out_w is not None and out_w is in_w
    if out_w is not None and not tied:
        out_w.requires_grad_(True)
        n += out_w.numel()
    if logger is not None and rank == 0:
        which = ("embeddings + lm_head (tied, one weight)" if tied
                 else "embeddings + lm_head" if out_w is not None else "embeddings")
        logger.info(f"[train_embeddings] unfroze {which} — {n/1e6:.1f}M params "
                    f"full-trained alongside LoRA")
    return n


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


def maybe_torch_compile(model, decoder_classes, *, enabled: bool, rank: int = 0, logger=None) -> int:
    """Opt-in (`--torch_compile`) PER-BLOCK `torch.compile(dynamic=True)` on each decoder layer,
    shared by every LLM trainer. Returns the number of layers compiled (0 when disabled).

    Call AFTER `shard_layers` (fully_shard) but BEFORE `checkpoint_layers`:
      * PER-BLOCK (not whole-model): whole-model `torch.compile` wraps the root in an
        `OptimizedModule` and prefixes every param with `_orig_mod.` — that would break the
        LoRA checkpoint keys the merges read. `layer.compile()` is in-place and leaves
        `named_parameters()` untouched. It's also the FSDP2-recommended shape.
      * BEFORE activation checkpointing: the clean nesting is `checkpoint(compiled_fn)`, so
        compile the raw sharded layer, then `checkpoint_layers` wraps the compiled forward.
      * `dynamic=True` is LOAD-BEARING: multipacked bins (and nemotron's padded batches) have a
        DIFFERENT length every step, so a static compile would recompile each step (minutes each
        → unusable). dynamic builds ONE symbolic-shape graph reused for all lengths.

    Each arch's custom attention / state-space kernel (gemma FA4 cute, qwen GatedDeltaNet,
    minimax/mistral FP8 dequant, nemotron Mamba2) is a custom op dynamo can't trace → it
    graph-breaks there automatically (expected; only the norm/activation/elementwise regions
    fuse). A per-layer compile failure is caught and the run stays eager rather than dying.

    ⚠ Opt-in + UNVERIFIED per arch: enable on a smoke run first and confirm the recompiles log
    stays quiet after step 1 ("make sure it will not recompile").
    """
    if not enabled:
        return 0
    import torch._dynamo as _dynamo
    _dynamo.config.cache_size_limit = 64   # headroom; blowing past this == something IS recompiling
    try:
        _dynamo.config.recompile_limit = 16
    except Exception:  # noqa: BLE001 — older torch names it differently; non-fatal
        pass
    # Log every (re)compilation with its guard reason so a smoke test can PROVE it compiles
    # once. Equivalent to running with TORCH_LOGS=recompiles.
    try:
        torch._logging.set_logs(recompiles=True)
    except Exception:  # noqa: BLE001
        pass
    classes = decoder_classes if isinstance(decoder_classes, tuple) else (decoder_classes,)
    n = 0
    for m in model.modules():
        if isinstance(m, classes):
            try:
                m.compile(dynamic=True)
                n += 1
            except Exception as e:  # noqa: BLE001 — a compile failure must NOT kill the run
                if rank == 0 and logger is not None:
                    logger.warning(f"[torch.compile] layer compile failed, staying eager: {e}")
                break
    if rank == 0 and logger is not None:
        logger.info(f"[torch.compile] per-block dynamic compile on {n} decoder layers "
                    f"(custom attention/SSM kernels graph-break; recompiles are logged). The FIRST "
                    f"optimizer step will be SLOW (tracing/compiling); steps after should be steady "
                    f"with NO further 'recompiling' log lines.")
    return n
