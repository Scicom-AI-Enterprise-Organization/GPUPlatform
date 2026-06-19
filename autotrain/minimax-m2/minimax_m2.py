"""MiniMax-M2 (230B/10B-active FP8 MoE) LoRA finetune — FSDP2 across 4x H100.

Standalone training job (NOT part of the gateway), the MiniMax-M2 analogue of `gemma4.py`.

Architecture vs gemma-4 (see CLAUDE.md for the full table):
  * MoE: 256 experts, top-8, 62 layers; stored as 3D fused expert params (FP8Experts).
  * Attention: uniform head_dim 128, all full attention, GQA 48/8 -> stock
    `flash_attention_2` with native varlen packing (NO custom attention needed).
  * Weights are FP8 block-quantized; we keep the base FROZEN in FP8 and train bf16 LoRA on
    q/k/v/o + the MoE expert FFNs (see `lora.py`). The frozen base forward is run through a
    DIFFERENTIABLE on-the-fly dequant (transformers' FP8 Triton/DeepGEMM kernels are
    inference-only / non-differentiable), with activation checkpointing keeping the
    transient bf16 weights memory-cheap.

Launch (run.sh wraps this):
    NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 torchrun --nproc_per_node=4 minimax_m2.py [flags]
"""
import argparse
import json
import logging
import os
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import destroy_process_group, fsdp, init_process_group
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import AutoConfig, MiniMaxM2ForCausalLM
from transformers.models.minimax_m2 import modeling_minimax_m2

from chinidataset import StreamingDataset
from lora import apply_minimax_lora

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger()

MODEL_ID = os.environ.get("MODEL_ID", "MiniMaxAI/MiniMax-M2")
# FA3 (Hopper) reuses gemma4's proven prebuilt wheel + torch 2.12; head_dim 128 works on FA2 or
# FA3. transformers routes both to the same varlen integration (consumes cu_seq_lens_q/k).
ATTN_IMPL = os.environ.get("MINIMAX_ATTN_IMPL", "flash_attention_3")


def ddp_setup():
    # cpu:gloo + cuda:nccl. gloo services CPU collectives (the LoRA-checkpoint full_tensor()
    # all-gather when CPUOffloadPolicy puts shards on CPU); nccl is the GPU fast path.
    init_process_group(backend="cpu:gloo,cuda:nccl")


class PackedDataset(TorchDataset):
    """Reads the multipacked ChiniDataset built by pack_dataset.py (same format as gemma4)."""

    def __init__(self, local: str = "./packed_data", limit: int = 0):
        self.dataset = StreamingDataset(local=local)
        self._len = min(limit, len(self.dataset)) if limit and limit > 0 else len(self.dataset)

    def __getitem__(self, idx):
        s = self.dataset[idx]
        return {
            "input_ids": s["input_ids"],
            "attention_mask": s["attention_mask"],  # per-doc lengths
            "labels": s["labels"],
            "position_ids": s["position_ids"],       # reset per doc
        }

    def __len__(self):
        return self._len


def collator(batch):
    """Concatenate packed bins into ONE varlen sequence + cu_seqlens (B is always 1).

    MiniMax-M2 is text-only with uniform head_dim 128, so there is no mm_token_type_ids and
    no custom attention: stock flash_attention_2 consumes cu_seq_lens_q/k (+ max_length_q/k)
    and the per-doc-reset position_ids to keep attention per-document causal over the pack.
    Use --batch_size 1 so a step is one packed bin (the collator concatenates the whole batch).
    """
    batch = [b for b in batch if b is not None]
    input_ids = np.concatenate([b["input_ids"] for b in batch])
    position_ids = np.concatenate([b["position_ids"] for b in batch])
    labels = np.concatenate([b["labels"] for b in batch])
    query_lens = np.concatenate([b["attention_mask"] for b in batch])  # per-doc lengths

    cu = [0] + np.cumsum(query_lens).tolist()
    cu_seq_lens = torch.tensor(cu, dtype=torch.int32)
    max_seqlen = int(np.max(query_lens))

    input_ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    return {
        "input_ids": input_ids_t,
        "position_ids": torch.tensor(position_ids, dtype=torch.long).unsqueeze(0),
        "attention_mask": None,  # packing carried by cu_seqlens + position_ids, not a dense mask
        "labels": torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        "cu_seq_lens_q": cu_seq_lens,
        "cu_seq_lens_k": cu_seq_lens,
        "max_length_q": max_seqlen,
        "max_length_k": max_seqlen,
    }


class CustomMiniMaxM2ForCausalLM(MiniMaxM2ForCausalLM):
    """Compute the loss with Liger FusedLinearCrossEntropy directly from the hidden states.

    MiniMax-M2's vocab is 200,064 — materializing (1, S, 200k) logits OOMs at long S.
    Liger FLCE fuses the lm_head matmul + cross-entropy without ever building the full
    logits tensor (same trick as gemma4's CustomGemma4ForConditionalGeneration).
    """

    def __init__(self, config):
        super().__init__(config)
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        self.loss_fn = LigerFusedLinearCrossEntropyLoss()

    def forward(self, input_ids=None, position_ids=None, attention_mask=None,
                labels=None, use_cache=False, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            **kwargs,  # carries cu_seq_lens_q/k + max_length_q/k through to flash attention
        )
        hidden_states = outputs.last_hidden_state

        if labels is None:
            raise NotImplementedError("training-only forward; labels required")
        shifted_hidden = hidden_states[:, :-1, :].contiguous().reshape(-1, hidden_states.shape[-1])
        shifted_labels = labels[:, 1:].contiguous().reshape(-1)
        loss = self.loss_fn(self.lm_head.weight, shifted_hidden, shifted_labels)
        return {"loss": loss}


def main(args):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)  # pin BEFORE init'ing NCCL / the mesh (else every rank -> cuda:0)
    ddp_setup()
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("shard",))

    # ---- load the FP8 model (frozen base) -----------------------------------
    # Native transformers class (NOT trust_remote_code) so we get the built-in FP8 +
    # fused-experts integration. The fp8 quantization_config lives in the model config, so
    # from_pretrained loads q/k/v/o as FP8Linear and the experts as FP8Experts automatically.
    config = AutoConfig.from_pretrained(MODEL_ID)
    logger.info(f"[Rank{rank}] loading {MODEL_ID} (FP8, frozen base)")
    model = CustomMiniMaxM2ForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        dtype=torch.bfloat16,                 # non-quantized parts (norms, router gate, lm_head)
        attn_implementation=ATTN_IMPL,
        low_cpu_mem_usage=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[Rank{rank}] params: {total_params/1e9:.1f}B")

    # ---- LoRA: freeze base, wrap attn q/k/v/o + MoE experts -----------------
    stats = apply_minimax_lora(
        model,
        attn_r=args.attn_r, attn_alpha=args.attn_alpha,
        moe_r=args.moe_r, moe_alpha=args.moe_alpha,
        include_moe=not args.no_moe_lora,
    )
    logger.info(
        f"[Rank{rank}] LoRA: wrapped {stats['attn_modules_wrapped']} attn blocks, "
        f"adapted {stats['moe_blocks_adapted']} MoE blocks | "
        f"attn {stats['attn_lora_params']/1e6:.1f}M + moe {stats['moe_lora_params']/1e6:.1f}M "
        f"= {stats['trainable_params']/1e6:.1f}M trainable"
    )

    # ---- FSDP2: shard each decoder layer + root -----------------------------
    # param_dtype is left at None: FSDP2 must NOT cast the frozen FP8 weights to bf16 (that
    # would defeat the block-scale dequant). Each param keeps its storage dtype (fp8 frozen,
    # bf16 LoRA/norms, fp32 scales); only gradient reduction is forced to fp32.
    fsdp_kwargs = {
        "mp_policy": fsdp.MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32),
        "mesh": mesh,
    }
    if args.cpu_offload:
        fsdp_kwargs["offload_policy"] = fsdp.CPUOffloadPolicy()

    for module in model.modules():
        if isinstance(module, modeling_minimax_m2.MiniMaxM2DecoderLayer):
            fsdp.fully_shard(module, **fsdp_kwargs)
    fsdp.fully_shard(model, **fsdp_kwargs)

    # ---- activation checkpointing (REQUIRED for the dequant memory trick) ----
    # Each decoder layer's forward (incl. the transient bf16 weight dequant) is recomputed in
    # the backward instead of retained, so peak memory stays ~ one layer's bf16 weights.
    non_reentrant = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant,
        check_fn=lambda m: isinstance(m, modeling_minimax_m2.MiniMaxM2DecoderLayer),
    )

    model_sd = model.state_dict()
    local_shard = sum(v.to_local().numel() if hasattr(v, "to_local") else v.numel()
                      for v in model_sd.values())
    logger.info(f"[Rank{rank}] local shard params: {local_shard/1e9:.2f}B")

    # ---- data ----------------------------------------------------------------
    dataset = PackedDataset(local=args.data_dir, limit=args.limit_samples)
    sampler = DistributedSampler(dataset, num_replicas=mesh.size(), rank=mesh.get_rank())
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collator, sampler=sampler,
        num_workers=2, prefetch_factor=2,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, fused=False)
    model.train()

    wandb_run = None
    if args.wandb and rank == 0:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, config={
            "model": MODEL_ID, "attn_r": args.attn_r, "attn_alpha": args.attn_alpha,
            "moe_r": args.moe_r, "moe_alpha": args.moe_alpha, "moe_lora": not args.no_moe_lora,
            "lr": args.lr, "batch_size": args.batch_size, "max_epochs": args.max_epochs,
            "max_steps": args.max_steps, "world_size": world_size, "num_bins": len(dataset),
            "trainable_params_M": round(stats["trainable_params"] / 1e6, 2),
        })
        logger.info(f"wandb run: {wandb_run.url}")

    reached_max = False
    for epoch in range(args.max_epochs):
        sampler.set_epoch(epoch)
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=(rank != 0)):
            global_step = epoch * len(dataloader) + idx
            if rank == 0:
                t0 = time.time()

            batch = {k: (v.to(f"cuda:{rank}", non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            out = model(**batch, use_cache=False)
            out["loss"].backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            tok = torch.tensor(batch["input_ids"].numel(), device=f"cuda:{rank}")
            dist.all_reduce(tok, op=dist.ReduceOp.SUM)

            if rank == 0:
                loss = out["loss"].item()
                tps = tok.item() / (time.time() - t0)
                logger.info(f"epoch {epoch} step {global_step} loss {loss:.4f} tok/s {tps:.0f}")
                if wandb_run is not None:
                    try:
                        wandb_run.log({"loss": loss, "lr": optimizer.param_groups[0]["lr"],
                                       "tps": tps, "epoch": epoch}, step=global_step)
                    except Exception as e:
                        logger.warning(f"wandb.log failed: {e}")

            reached_max = args.max_steps > 0 and (global_step + 1) >= args.max_steps
            last = idx == len(dataloader) - 1
            if last or (idx + 1) % args.checkpointing_step == 0 or reached_max:
                save_lora(model, args, rank, stats)
            if reached_max:
                logger.info(f"reached max_steps={args.max_steps}, stopping")
                break
        if reached_max:
            break

    if wandb_run is not None:
        wandb_run.finish()
    destroy_process_group()


def save_lora(model, args, rank, stats):
    """Gather + save ONLY the trainable LoRA tensors. full_tensor() is a collective, so every
    rank must iterate the SAME (requires_grad) params; only rank 0 writes."""
    logger.info("checkpointing LoRA adapters..")
    sd = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        full = param.full_tensor() if hasattr(param, "full_tensor") else param
        if rank == 0:
            sd[name] = full.detach().to("cpu")
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(sd, os.path.join(args.out_dir, "lora.pt"))
        with open(os.path.join(args.out_dir, "lora_meta.json"), "w") as f:
            json.dump({
                "model_id": MODEL_ID,
                "attn_r": args.attn_r, "attn_alpha": args.attn_alpha,
                "attn_scaling": args.attn_alpha / args.attn_r,
                "moe_r": args.moe_r, "moe_alpha": args.moe_alpha,
                "moe_scaling": args.moe_alpha / args.moe_r,
                "moe_lora": not args.no_moe_lora,
                "attn_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "moe_targets": ["gate_up_proj", "down_proj"],
            }, f, indent=2)
        logger.info(f"saved LoRA ({len(sd)} tensors) -> {args.out_dir}/lora.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attn_r", type=int, default=16, help="LoRA rank for attention q/k/v/o.")
    p.add_argument("--attn_alpha", type=float, default=16.0, help="LoRA alpha for attention (scaling=alpha/r).")
    p.add_argument("--moe_r", type=int, default=16, help="LoRA rank for MoE expert FFNs.")
    p.add_argument("--moe_alpha", type=float, default=16.0, help="LoRA alpha for MoE experts.")
    p.add_argument("--no_moe_lora", action="store_true", help="Adapt attention only (skip MoE experts).")
    p.add_argument("--batch_size", type=int, default=1, help="Packed bins per step (keep 1; collator concatenates).")
    p.add_argument("--lr", type=float, default=1e-5, help="AdamW LR (gemma4 lesson: 1e-5..5e-5 for LoRA).")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=0, help="Hard stop after N steps (0 = all epochs).")
    p.add_argument("--checkpointing_step", type=int, default=100)
    p.add_argument("--limit_samples", type=int, default=0, help="Cap dataset to first N bins (0 = all).")
    p.add_argument("--data_dir", default="./packed_data")
    p.add_argument("--out_dir", default="./checkpointing")
    p.add_argument("--cpu_offload", action="store_true", help="FSDP2 CPUOffloadPolicy (slow; for tight VRAM).")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="minimax-m2-autotrain")
    main(p.parse_args())
