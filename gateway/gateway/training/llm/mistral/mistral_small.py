"""Mistral-Small-4-119B (FP8 MoE + MLA) LoRA finetune — FSDP2 across 4x H100.

Standalone training job (NOT part of the gateway), the Mistral-Small-4 analogue of the
minimax-m2 sibling. LoRA-finetunes the TEXT model of `mistralai/Mistral-Small-4-119B-2603`
(a `Mistral3ForConditionalGeneration` multimodal model whose language model is a 119B-total
`mistral4` MoE: 128 routed experts top-4 + 1 shared expert, 36 layers, MLA attention).

Architecture vs minimax-m2 (see CLAUDE.md for the full table):
  * MoE: 128 routed experts (top-4) + 1 SHARED expert, 36 layers; routed experts stored as
    3D fused expert params (FP8Experts). Shared expert is a dense SwiGLU MLP (FP8Linear).
  * Attention: MLA (DeepSeek-style) — compressed-latent q (q_lora_rank 1024) + kv
    (kv_lora_rank 256) projections, but a UNIFORM head_dim 128 (qk_head_dim == v_head_dim)
    so every layer runs stock flash attention with native varlen packing (NO custom attn).
  * Weights are FP8 PER-TENSOR (`weight_block_size=null`, static activations). We keep the
    base FROZEN in FP8 and train bf16 LoRA on the MLA projections + routed/shared experts
    (see `lora.py`), running the frozen base through a DIFFERENTIABLE on-the-fly dequant
    (transformers' FP8 kernels are inference-only; the fused/grouped/deepgemm experts
    dispatches even refuse activation_scheme="static"). Activation checkpointing keeps the
    transient bf16 weights memory-cheap.
  * Multimodal wrapper: we load the full `Mistral3ForConditionalGeneration`, freeze the
    vision tower + projector, and run the TEXT-only forward through `model.language_model`.

Launch (run.sh wraps this):
    NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 torchrun --nproc_per_node=4 mistral_small.py [flags]
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

from transformers import AutoConfig, Mistral3ForConditionalGeneration
from transformers.models.mistral4 import modeling_mistral4

from chinidataset import StreamingDataset
import _trainer_common as tc
from lora import ATTN_TARGETS, apply_mistral_lora

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger()

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-Small-4-119B-2603")
# FA3 (Hopper) reuses the proven prebuilt wheel + torch 2.12; head_dim 128 works on FA2 or FA3.
# transformers routes both to the same varlen integration (consumes cu_seq_lens_q/k).
ATTN_IMPL = os.environ.get("MISTRAL_ATTN_IMPL", "flash_attention_3")


def load_patched_config(model_id: str = MODEL_ID):
    """AutoConfig for Mistral-Small-4 + the transformers 5.5.0 FP8 workaround.

    transformers 5.5.0's `FP8Experts.__init__` does `getattr(config, "num_local_experts",
    config.num_experts)`. Python evaluates the *default* `config.num_experts` eagerly, and
    `Mistral4Config` only exposes `num_local_experts` (via attribute_map) — there is no
    `num_experts` — so the load crashes with AttributeError before getattr even runs. Set
    `num_experts` on the text config (== num_local_experts) to dodge the eager default.
    """
    config = AutoConfig.from_pretrained(model_id)
    tc = config.get_text_config()
    if not hasattr(tc, "num_experts"):
        tc.num_experts = tc.num_local_experts
    return config


def ddp_setup():
    # cpu:gloo + cuda:nccl. gloo services CPU collectives (the LoRA-checkpoint full_tensor()
    # all-gather when CPUOffloadPolicy puts shards on CPU); nccl is the GPU fast path.
    init_process_group(backend="cpu:gloo,cuda:nccl")


class PackedDataset(TorchDataset):
    """Reads the multipacked ChiniDataset built by pack_dataset.py (same format as minimax-m2)."""

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

    Mistral-Small-4's text model is uniform head_dim 128, so there is no custom attention:
    stock flash attention consumes cu_seq_lens_q/k (+ max_length_q/k) and the per-doc-reset
    position_ids to keep attention per-document causal over the pack. Use --batch_size 1 so
    a step is one packed bin (the collator concatenates the whole batch).
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


class CustomMistral3ForCausalLM(Mistral3ForConditionalGeneration):
    """Text-only training forward + Liger FusedLinearCrossEntropy loss.

    We bypass the vision path entirely (no pixel inputs) by calling the language model
    submodule directly, and compute the loss with Liger FLCE straight from the hidden
    states: Mistral-Small-4's vocab is 131,072 — materialising (1, S, 131k) logits OOMs at
    long S, and FLCE fuses the lm_head matmul + cross-entropy without ever building it.
    """

    def __init__(self, config):
        super().__init__(config)
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        self.loss_fn = LigerFusedLinearCrossEntropyLoss()

    def forward(self, input_ids=None, position_ids=None, attention_mask=None,
                labels=None, use_cache=False, **kwargs):
        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **kwargs,  # carries cu_seq_lens_q/k + max_length_q/k through to flash attention
        )
        hidden_states = outputs.last_hidden_state

        if labels is None:
            raise NotImplementedError("training-only forward; labels required")
        shifted_hidden = hidden_states[:, :-1, :].contiguous().reshape(-1, hidden_states.shape[-1])
        shifted_labels = labels[:, 1:].contiguous().reshape(-1)
        loss = self.loss_fn(self.lm_head.weight, shifted_hidden, shifted_labels)
        return {"loss": loss}


# ---------------------------------------------------------------------------
# Low-CPU sharded load (--low_cpu_shard_load)
# ---------------------------------------------------------------------------
# The DEFAULT path has every rank `from_pretrained` the full ~119GB model to CPU before FSDP2
# shards it to GPU -> ~119GB x world_size of CPU. This path instead builds the FP8 module
# structure on META (no weights) on every rank, shards it, then streams the base weights from
# rank 0 only (DCP broadcast_from_rank0) straight into each rank's local shard.
def build_meta_lora_model(config, args):
    """Meta-init the FP8 structure on EVERY rank (no weights) + apply LoRA (meta, uninitialised)."""
    from transformers.quantizers import AutoHfQuantizer
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # non-fp8 parts (norms/embed/lm_head/router/vision) -> bf16
    try:
        with torch.device("meta"):
            model = CustomMistral3ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev)
    hf_quantizer = AutoHfQuantizer.from_config(config.quantization_config)
    hf_quantizer._process_model_before_weight_loading(model)  # Linear->FP8Linear, experts->FP8Experts
    stats = apply_mistral_lora(
        model,
        attn_r=args.attn_r, attn_alpha=args.attn_alpha,
        moe_r=args.moe_r, moe_alpha=args.moe_alpha,
        include_moe=not args.no_moe_lora, include_shared=not args.no_shared_lora,
        use_dora=args.use_dora,
    )
    return model, stats


# LinearLoRA wraps the frozen Linear, so the model FQN becomes `...{proj}.base.{weight,...}`.
# Insert `.base` after the wrapped projection segment so the rank-0 state_dict matches. This
# segment-replace catches weight, weight_scale_inv AND activation_scale in one shot.
_WRAPPED_SEGMENTS = tuple(f".self_attn.{p}." for p in ATTN_TARGETS) + (
    ".shared_experts.gate_proj.", ".shared_experts.up_proj.", ".shared_experts.down_proj.",
)


def _remap_base_keys_for_lora(sd):
    out = {}
    for k, v in sd.items():
        nk = k
        for seg in _WRAPPED_SEGMENTS:
            if seg in k:
                nk = k.replace(seg, seg[:-1] + "base.", 1)
                break
        # Match _promote_scalar_params: 0-dim scalars (per-tensor weight_scale_inv/activation_scale)
        # become shape (1,) on the sharded model, so reshape the source tensors too.
        out[nk] = v.reshape(1) if hasattr(v, "dim") and v.dim() == 0 else v
    return out


def load_base_weights_broadcast(model, config, rank):
    """Fill the (materialised, sharded) model's base weights from rank 0's full CPU state_dict via
    DCP broadcast. Only rank 0 holds the full ~119GB copy; other ranks receive their shard."""
    from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
    full_sd = {}
    if rank == 0:
        logger.info(f"[Rank0] loading full base state_dict to CPU for broadcast ({MODEL_ID})")
        src = CustomMistral3ForCausalLM.from_pretrained(
            MODEL_ID, config=config, dtype=torch.bfloat16, attn_implementation=ATTN_IMPL,
            low_cpu_mem_usage=True)
        full_sd = _remap_base_keys_for_lora(src.state_dict())
        del src
    set_model_state_dict(
        model, model_state_dict=full_sd,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False),
    )


def _promote_scalar_params(model):
    """FSDP2 `fully_shard` rejects 0-dim (scalar) parameters. Mistral-Small-4 is PER-TENSOR FP8,
    so each `FP8Linear` stores `weight_scale_inv` (and, with static activations, `activation_scale`)
    as a *scalar* Parameter -> "fully_shard doesn't support scalar parameters". Reshape every 0-dim
    param to shape (1,) before sharding. This is numerically transparent: the per-tensor dequant
    (`w.float() * scale_inv`) and the Triton kernel both broadcast a (1,) scale exactly like a
    scalar. (minimax-m2 was 128x128 block-scaled, so its scale_inv was already >=2D and never hit
    this.)"""
    n = 0
    for module in model.modules():
        for name, p in list(module.named_parameters(recurse=False)):
            if p is not None and p.dim() == 0:
                new = nn.Parameter(p.detach().reshape(1), requires_grad=p.requires_grad)
                setattr(module, name, new)
                n += 1
    return n


def _reinit_lora_(model):
    """Re-initialise LoRA/DoRA adapters on the materialised (sharded) tensors: A=kaiming, B=0, and
    (DoRA only) magnitude = base-weight row-norm. See minimax_m2._reinit_lora_ for the sharding
    rationale (attention gathers the full weight; experts use whole-expert local shards)."""
    from lora import LinearLoRA, dequantize_fp8
    from torch.distributed.tensor import DTensor, distribute_tensor

    def loc(t):
        return t.to_local() if isinstance(t, DTensor) else t

    def _dequant_local(experts, which):
        if which == "gate_up":
            w, s = experts.gate_up_proj, getattr(experts, "gate_up_proj_scale_inv", None)
        else:
            w, s = experts.down_proj, getattr(experts, "down_proj_scale_inv", None)
        wl = loc(w)
        if wl.element_size() == 1 and s is not None:
            return dequantize_fp8(wl, loc(s), getattr(experts, "block_size", None), out_dtype=torch.float32)
        return wl.to(torch.float32)

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LinearLoRA):
                nn.init.kaiming_uniform_(loc(m.lora_a.weight))
                nn.init.zeros_(loc(m.lora_b.weight))
                if getattr(m, "use_dora", False):
                    w = m.base.weight
                    w_full = w.full_tensor() if isinstance(w, DTensor) else w
                    if m._fp8:
                        s = m.base.weight_scale_inv
                        s_full = s.full_tensor() if isinstance(s, DTensor) else s
                        deq = dequantize_fp8(w_full, s_full, m._block_size, out_dtype=torch.float32)
                    else:
                        deq = w_full.to(torch.float32)
                    mag_full = deq.norm(dim=1).to(m.magnitude.dtype)
                    mag = m.magnitude
                    if isinstance(mag, DTensor):
                        mag.copy_(distribute_tensor(mag_full, mag.device_mesh, mag.placements))
                    else:
                        mag.copy_(mag_full)
            if hasattr(m, "gate_up_lora_a"):  # FP8Experts carrying routed-expert LoRA/DoRA
                nn.init.kaiming_uniform_(loc(m.gate_up_lora_a))
                nn.init.kaiming_uniform_(loc(m.down_lora_a))
                nn.init.zeros_(loc(m.gate_up_lora_b))
                nn.init.zeros_(loc(m.down_lora_b))
                if getattr(m, "use_dora", False):
                    loc(m.gate_up_mag).copy_(_dequant_local(m, "gate_up").norm(dim=2).to(loc(m.gate_up_mag).dtype))
                    loc(m.down_mag).copy_(_dequant_local(m, "down").norm(dim=2).to(loc(m.down_mag).dtype))


def main(args):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)  # pin BEFORE init'ing NCCL / the mesh (else every rank -> cuda:0)
    ddp_setup()
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("shard",))

    # ---- load the FP8 model (frozen base) -----------------------------------
    # Native transformers class (NOT trust_remote_code). The fp8 quantization_config lives in
    # the model config, so from_pretrained loads the MLA q/kv/o + shared expert as FP8Linear
    # and the routed experts as FP8Experts automatically (after the num_experts workaround).
    config = load_patched_config(MODEL_ID)
    # MoE expert rank: AUTO-derive attn_r ÷ active-experts (top-k), scaling held constant, unless
    # --moe_r was explicitly set (>0). Set on args so BOTH build paths use the same rank.
    # https://thinkingmachines.ai/blog/lora/
    if not args.no_moe_lora and (not args.moe_r or args.moe_r <= 0):
        import moe_adapter as _MA
        _k = _MA.num_active_experts(config) or 8
        args.moe_r, args.moe_alpha = _MA.derive_moe_rank(args.attn_r, args.attn_alpha, _k)
        logger.info(f"[Rank{rank}] MoE expert rank auto-derived: r={args.moe_r} alpha={args.moe_alpha:.1f} "
                    f"(attn_r={args.attn_r} ÷ top_k={_k}; scaling {args.attn_alpha/args.attn_r:.2f})")
    if args.low_cpu_shard_load:
        logger.info(f"[Rank{rank}] meta-init {MODEL_ID} (FP8 structure, low-CPU sharded load)")
        model, stats = build_meta_lora_model(config, args)
    else:
        logger.info(f"[Rank{rank}] loading {MODEL_ID} (FP8, frozen base)")
        model = CustomMistral3ForCausalLM.from_pretrained(
            MODEL_ID,
            config=config,
            dtype=torch.bfloat16,             # non-quantized parts (norms, router, lm_head, vision)
            attn_implementation=ATTN_IMPL,
            low_cpu_mem_usage=True,
        )
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"[Rank{rank}] params: {total_params/1e9:.1f}B")

        # ---- LoRA: freeze base, wrap MLA attn + routed/shared experts -----------
        stats = apply_mistral_lora(
            model,
            attn_r=args.attn_r, attn_alpha=args.attn_alpha,
            moe_r=args.moe_r, moe_alpha=args.moe_alpha,
            include_moe=not args.no_moe_lora, include_shared=not args.no_shared_lora,
        )
    logger.info(
        f"[Rank{rank}] LoRA: wrapped {stats['attn_modules_wrapped']} attn + "
        f"{stats['shared_modules_wrapped']} shared blocks, adapted {stats['moe_blocks_adapted']} "
        f"routed-MoE blocks | attn {stats['attn_lora_params']/1e6:.1f}M + moe "
        f"{stats['moe_lora_params']/1e6:.1f}M + shared {stats['shared_lora_params']/1e6:.1f}M "
        f"= {stats['trainable_params']/1e6:.1f}M trainable"
    )

    # Optionally FULL-train the token embeddings + LM head on top of LoRA (they're the bf16
    # non-FP8 parts). Mistral-Small-4 is UNTIED → both weights unfrozen. Done BEFORE FSDP
    # sharding; the save loop captures them; merge attaches them by name.
    if getattr(args, "train_embeddings", False):
        tc.unfreeze_embeddings(model, rank=rank, logger=logger)

    # ---- FSDP2: shard each decoder layer + root -----------------------------
    # param_dtype=None so FSDP2 does NOT cast the frozen FP8 weights to bf16 (that would
    # defeat the per-tensor dequant); only gradient reduction is forced to fp32.
    kw = tc.fsdp_kwargs(mesh, param_dtype=None, cpu_offload=args.cpu_offload)

    # Per-tensor FP8 stores scalar weight_scale_inv/activation_scale; FSDP2 needs them 1D.
    n_promoted = _promote_scalar_params(model)
    logger.info(f"[Rank{rank}] promoted {n_promoted} scalar (0-dim) params to shape (1,) for FSDP2")

    tc.shard_layers(model, modeling_mistral4.Mistral4DecoderLayer, kw)

    # ---- low-CPU path: materialise the sharded meta model, then stream base weights in ----
    if args.low_cpu_shard_load:
        model.to_empty(device=f"cuda:{rank}")          # allocate real (empty) local shards
        load_base_weights_broadcast(model, config, rank)  # rank-0 full sd -> broadcast into shards
        _reinit_lora_(model)                            # A=kaiming, B=0 on the real tensors
        logger.info(f"[Rank{rank}] low-CPU sharded load complete")

    # torch.compile (opt-in, --torch_compile): per-block dynamic compile after sharding, before
    # AC. The FP8 dequant + fused-MoE grouped_mm + MLA are custom ops that graph-break; only the
    # norm/activation/elementwise regions fuse.
    tc.maybe_torch_compile(model, modeling_mistral4.Mistral4DecoderLayer,
                           enabled=args.torch_compile, rank=rank, logger=logger)

    # ---- activation checkpointing (REQUIRED for the dequant memory trick) ----
    tc.checkpoint_layers(model, modeling_mistral4.Mistral4DecoderLayer)

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
            "shared_lora": not args.no_shared_lora,
            "lr": args.lr, "batch_size": args.batch_size, "max_epochs": args.max_epochs,
            "max_steps": args.max_steps, "world_size": world_size, "num_bins": len(dataset),
            "trainable_params_M": round(stats["trainable_params"] / 1e6, 2),
        })
        logger.info(f"wandb run: {wandb_run.url}")

    # opt_step counts OPTIMIZER steps (one per grad_accum microbatches).
    opt_step = 0
    reached_max = False
    # Running count of contributing (non -100) label tokens in the current accumulation
    # window on THIS rank; summed across ranks each optimizer step for token-weighted loss.
    win_tokens = torch.zeros((), device=f"cuda:{rank}", dtype=torch.long)
    for epoch in range(args.max_epochs):
        sampler.set_epoch(epoch)
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=(rank != 0)):
            if rank == 0:
                t0 = time.time()

            batch = {k: (v.to(f"cuda:{rank}", non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            out = model(**batch, use_cache=False)
            # Back-prop the token-SUM loss (mean × #label tokens). Accumulating sums, then
            # normalizing by the GLOBAL token count at the step below, yields the exact
            # token-weighted mean over the whole effective batch — not a naive mean-of-means
            # that mis-weights variable-length bins / ranks (HF grad-accum loss fix).
            n_tok = (batch["labels"][:, 1:] != -100).sum()  # == the loss_fn's shifted denominator
            (out["loss"] * n_tok).backward()
            win_tokens += n_tok

            # Step once every grad_accum microbatches; flush a partial window at epoch
            # end. grad_accum=1 → step every microbatch (unchanged).
            do_step = ((idx + 1) % args.grad_accum == 0) or (idx == len(dataloader) - 1)
            if do_step:
                # Gather total label tokens over the window AND across ranks, then ÷
                # world_size to counteract FSDP's gradient averaging → the accumulated
                # grad becomes the true token-mean (HF average_tokens_across_devices).
                dist.all_reduce(win_tokens, op=dist.ReduceOp.SUM)
                scale = (world_size / win_tokens.clamp(min=1)).item()
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            p.grad.mul_(scale)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                win_tokens.zero_()

            tok = torch.tensor(batch["input_ids"].numel(), device=f"cuda:{rank}")
            dist.all_reduce(tok, op=dist.ReduceOp.SUM)

            if rank == 0:
                loss = out["loss"].item()
                tps = tok.item() / (time.time() - t0)
                logger.info(f"epoch {epoch} step {opt_step} loss {loss:.4f} tok/s {tps:.0f}")
                if wandb_run is not None and do_step:
                    try:
                        wandb_run.log({"loss": loss, "lr": optimizer.param_groups[0]["lr"],
                                       "tps": tps, "epoch": epoch}, step=opt_step)
                    except Exception as e:
                        logger.warning(f"wandb.log failed: {e}")

            reached_max = args.max_steps > 0 and do_step and (opt_step + 1) >= args.max_steps
            last = idx == len(dataloader) - 1
            if last or (do_step and (idx + 1) % args.checkpointing_step == 0) or reached_max:
                save_lora(model, args, rank, stats)
            if do_step:
                opt_step += 1
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
                "shared_lora": not args.no_shared_lora,
                "attn_targets": list(ATTN_TARGETS),
                "moe_targets": ["gate_up_proj", "down_proj"],
                "shared_targets": ["gate_proj", "up_proj", "down_proj"],
                "train_embeddings": bool(getattr(args, "train_embeddings", False)),
                "use_dora": bool(args.use_dora),
            }, f, indent=2)
        logger.info(f"saved LoRA ({len(sd)} tensors) -> {args.out_dir}/lora.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attn_r", type=int, default=16, help="LoRA rank for MLA q_a/q_b/kv_a/kv_b/o.")
    p.add_argument("--attn_alpha", type=float, default=16.0, help="LoRA alpha for attention (scaling=alpha/r).")
    p.add_argument("--moe_r", type=int, default=0,
                   help="LoRA/DoRA rank for routed + shared expert FFNs. 0 = AUTO: attn_r ÷ active-experts "
                        "(top-k), scaling held constant (https://thinkingmachines.ai/blog/lora/). >0 overrides.")
    p.add_argument("--moe_alpha", type=float, default=0.0, help="alpha for the expert FFNs (0 = auto, keeps attn scaling).")
    p.add_argument("--no_moe_lora", action="store_true", help="Adapt attention only (skip all experts).")
    p.add_argument("--no_shared_lora", action="store_true", help="Skip the shared-expert MLP (keep routed).")
    p.add_argument("--use_dora", action="store_true",
                   help="Use DoRA (weight-decomposed LoRA) instead of plain LoRA for attention + "
                        "routed/shared experts: adds a trainable per-output-row magnitude.")
    p.add_argument("--train_embeddings", action="store_true",
                   help="Also FULL-train the (bf16) token embeddings + LM head on top of LoRA "
                        "(Mistral-Small-4 is untied → both weights). Helps the finetune reliably "
                        "emit special tokens. Costs extra optimizer state — combine with --cpu_offload.")
    # batch_size / grad_accum / cpu_offload / max_epochs / max_steps / checkpointing_step
    # / limit_samples / lr / wandb[_project] — shared across all LLM trainers.
    tc.add_common_args(p, lr_default=1e-5, wandb_project="mistral-small-4-autotrain")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--data_dir", default="./packed_data")
    p.add_argument("--out_dir", default="./checkpointing")
    p.add_argument("--low_cpu_shard_load", action="store_true",
                   help="Meta-init the FP8 structure on every rank + stream base weights from rank 0 "
                        "(DCP broadcast) into each shard. Caps CPU at ~one model copy instead of "
                        "~119GB x world_size. Default path loads the full model on every rank.")
    main(p.parse_args())
