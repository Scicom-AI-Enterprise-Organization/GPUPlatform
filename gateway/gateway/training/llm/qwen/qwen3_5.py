"""Qwen3.5-27B LoRA finetune — FSDP2 (fully_shard) + CPU offload, packed varlen.

torchrun entrypoint. Wraps q/k/v/o_proj with a custom LinearLoRA (B=0 init), shards each
Qwen3_5DecoderLayer / Qwen3_5VisionBlock with FSDP2 under CPUOffloadPolicy, and trains on the
ChiniDataset packed bins produced by pack_dataset.py. Loss is LigerFusedLinearCrossEntropyLoss
(materialized lm_head logits are skipped). The Qwen3.5 GatedDeltaNet (linear attention) path runs
the FlashQLA `chunk_gated_delta_rule` kernel (v patched contiguous for the TileLang kernel).

Attention backend: kernels-community/flash-attn3 (auto-fetched by the `kernels` package).

`--dpo` switches the objective to Direct Preference Optimization over a DPO-packed
dataset (kind=llm_dpo_packed: every bin holds whole preference pairs, first half of
its docs chosen then rejected, labels PRE-ALIGNED next-token targets). The loss is
triton_dpo.fused_dpo_loss (multipacked, logits never materialized — see
small-ablation/multipacking-dpo); the frozen reference model is THIS model with the
LoRA branches disabled (base weights are frozen and B=0-init ⇒ policy == reference at
step 0 ⇒ first loss ≈ ln 2), so no second model copy is loaded. Incompatible with
--cp_size > 1 (per-sequence log-probs would need cross-rank reduction).
"""
from transformers import AutoConfig
from transformers.cache_utils import Cache
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import logging
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group, fsdp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
    offload_wrapper
)
import os
import json
import time
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
import argparse
from chinidataset import StreamingDataset
import _trainer_common as tc
from flash_qla import chunk_gated_delta_rule
import context_parallel as cp  # GatedDeltaNet-hybrid context parallelism (opt-in via --cp_size > 1)
from transformers import AttentionInterface
# Full-attention (softmax) layers under CP use the contiguous position-aware ring; the GatedDeltaNet
# layers relay their conv+recurrent state across the CP group (installed in main() when cp_size > 1).
# Registering the ring backend is always safe (only dispatched when attn_implementation selects it).
AttentionInterface.register("cp_full_attention", cp.cp_full_attention)


logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def resolve_arch(model_id: str):
    """Pick the dense vs MoE Qwen3.5-family classes from the model config, apply the
    matching Liger patches, and return (config, is_moe, classes).

    Qwen3.6 reuses the Qwen3.5 architecture code (model_type qwen3_5 / qwen3_5_moe):
      - dense  (Qwen3.6-27B)      -> Qwen3_5ForConditionalGeneration     / modeling_qwen3_5
      - MoE    (Qwen3.6-35B-A3B)  -> Qwen3_5MoeForConditionalGeneration  / modeling_qwen3_5_moe
    Both are GatedDeltaNet hybrids; we only LoRA the attention q/k/v/o_proj, so the MoE
    experts are untouched and the same training path serves both. Liger MUST be applied
    before from_pretrained (it monkeypatches the module's RMSNorm/SwiGLU classes).
    """
    import inspect
    config = AutoConfig.from_pretrained(model_id)
    arch = (config.architectures or [""])[0]
    is_moe = "Moe" in arch
    if is_moe:
        from transformers import Qwen3_5MoeForConditionalGeneration as Base
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5_moe as liger
        classes = {
            "base": Base,
            "decoder": M.Qwen3_5MoeDecoderLayer,
            "vision": M.Qwen3_5MoeVisionBlock,
            "gdn": M.Qwen3_5MoeGatedDeltaNet,
        }
    else:
        from transformers import Qwen3_5ForConditionalGeneration as Base
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5 as liger
        classes = {
            "base": Base,
            "decoder": M.Qwen3_5DecoderLayer,
            "vision": M.Qwen3_5VisionBlock,
            "gdn": M.Qwen3_5GatedDeltaNet,
        }
    kw = {"rms_norm": True, "cross_entropy": False}
    if "swiglu" in inspect.signature(liger).parameters:
        kw["swiglu"] = True  # fuses gate+act+up, cuts dense-MLP peak memory ~50%
    liger(**kw)
    logger.info(f"[resolve_arch] {model_id}: arch={arch} is_moe={is_moe} liger_kwargs={kw}")
    return config, is_moe, classes


def ddp_setup():
    # cpu:gloo + cuda:nccl. With CPUOffloadPolicy the LoRA params are DTensors whose shards live
    # on CPU; the checkpoint's full_tensor() all-gather is then a CPU collective that NCCL can't
    # service ("No backend type associated with device type cpu"). gloo handles CPU collectives;
    # nccl stays the fast path for the GPU all-gather/reduce-scatter during forward/backward.
    init_process_group(backend="cpu:gloo,cuda:nccl")


class Dataset(Dataset):
    def __init__(self, limit: int = 0):
        self.dataset = StreamingDataset(local="./packed_data")
        # limit > 0 caps the dataset to the first N packed bins.
        self._len = min(limit, len(self.dataset)) if limit and limit > 0 else len(self.dataset)
        # NOTE: training consumes packed bins AS-IS (no truncation here). The packed-bin length is
        # controlled at dataset-prep time (pack_dataset.py --max-seq-len) so bins are both trainable
        # AND contain the actual conversation turns. (Truncating here to a small window only fed the
        # model the system/tool-schema preamble — the conversation starts ~25k tokens in — which is
        # why the earlier finetune degenerated.)

    def __getitem__(self, idx):
        return {
            "input_ids": self.dataset[idx]["input_ids"],
            "attention_mask": self.dataset[idx]["attention_mask"],  # np.array of per-doc lengths
            "labels": self.dataset[idx]["labels"],
            "position_ids": self.dataset[idx]["position_ids"],
        }

    def __len__(self):
        return self._len

def collator(batch):
    batch = [b for b in batch if b is not None]
    input_ids = [b['input_ids'] for b in batch]
    position_ids = [b['position_ids'] for b in batch]
    labels = [b['labels'] for b in batch]
    attention_mask = [b['attention_mask'] for b in batch]

    # query_lens = the length of every packed document across the whole batch.
    query_lens = np.concatenate(attention_mask)

    input_ids = np.concatenate(input_ids)
    position_ids = np.concatenate(position_ids)
    labels = np.concatenate(labels)

    cumsum = [0] + np.cumsum(query_lens).tolist()
    cu_seq_lens_q = torch.tensor(cumsum, dtype=torch.int32)
    max_seqlen_q = int(np.max(query_lens))

    # No dense (1, S, S) attention_mask: the packing is fully described by cu_seq_lens_* +
    # position_ids. dynamic_attention rebuilds the causal block-diagonal mask itself in the
    # SDPA branch and consumes cu_seqlens directly in the FA3 branch. A dense (1, S, S) mask
    # both wastes O(S^2) memory and confuses transformers' mask preparation (it expects a
    # 2D padding mask or a 4D causal mask, not 3D).
    input_ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    return {
        'input_ids': input_ids_t,
        'position_ids': torch.tensor(position_ids, dtype=torch.long).unsqueeze(0),
        'attention_mask': None,
        # Multimodal ConditionalGeneration models create_causal_mask_mapping may require
        # mm_token_type_ids during training. Text-only packing => all-zero (every token is text).
        'mm_token_type_ids': torch.zeros_like(input_ids_t),
        'labels': torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        'cu_seq_lens_q': cu_seq_lens_q,
        'cu_seq_lens_k': cu_seq_lens_q,
        'max_length_q': max_seqlen_q,
        'max_length_k': max_seqlen_q
    }

def dpo_collator(batch):
    """Packed DPO microbatch. Each DPO bin holds K whole preference pairs as 2K docs,
    first K chosen then K rejected (llm_pack.collate_dpo_bin). When batch_size > 1
    concatenates several bins, the docs are REORDERED so all chosen sequences come
    first across the whole row — preserving triton_dpo's pair contract (pair k =
    (seq k, seq K_total+k); chosen and rejected keep the same bin order, so pairing
    stays aligned). `labels` are pre-aligned next-token targets (no shift at loss
    time); `seq_boundaries` (int64 cu_seqlens) drives the fused DPO loss."""
    batch = [b for b in batch if b is not None]
    chosen, rejected = [], []  # per-bin (ids, labels, pos, lens) slices
    for b in batch:
        lens = np.asarray(b["attention_mask"])
        assert len(lens) % 2 == 0, "DPO bin must hold K chosen + K rejected docs — is this a kind=llm_dpo_packed dataset?"
        K = len(lens) // 2
        cut = int(lens[:K].sum())
        chosen.append((b["input_ids"][:cut], b["labels"][:cut], b["position_ids"][:cut], lens[:K]))
        rejected.append((b["input_ids"][cut:], b["labels"][cut:], b["position_ids"][cut:], lens[K:]))
    halves = chosen + rejected
    input_ids = np.concatenate([h[0] for h in halves])
    labels = np.concatenate([h[1] for h in halves])
    position_ids = np.concatenate([h[2] for h in halves])
    query_lens = np.concatenate([h[3] for h in halves])

    cumsum = [0] + np.cumsum(query_lens).tolist()
    cu_seq_lens_q = torch.tensor(cumsum, dtype=torch.int32)
    input_ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    return {
        "input_ids": input_ids_t,
        "position_ids": torch.tensor(position_ids, dtype=torch.long).unsqueeze(0),
        "attention_mask": None,
        "mm_token_type_ids": torch.zeros_like(input_ids_t),
        "labels": torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        "seq_boundaries": torch.tensor(cumsum, dtype=torch.long),
        "cu_seq_lens_q": cu_seq_lens_q,
        "cu_seq_lens_k": cu_seq_lens_q,
        "max_length_q": int(np.max(query_lens)),
        "max_length_k": int(np.max(query_lens)),
    }


# Optional hard cap on packed-sequence length (env SGPU_MAX_SEQ_LEN). 0 = off (bins trained as-is).
# Used to prove the CP path end-to-end at a SMALL context (the 32k-packed bins otherwise blow past
# memory with AC disabled under CP). Truncates to a multiple of `mult` (the CP size) and rebuilds
# cu_seqlens to the doc boundaries within the cap (the last doc is cut at the cap).
_MAX_SEQ_LEN = int(os.environ.get("SGPU_MAX_SEQ_LEN", "0") or 0)


def _truncate_packed(batch, n, mult=1):
    n = (n // mult) * mult
    if n <= 0 or batch["input_ids"].shape[1] <= n:
        return batch
    cu = batch["cu_seq_lens_q"]
    cu2 = cu[cu <= n]
    if int(cu2[-1]) < n:                              # cut the straddling doc at the cap
        cu2 = torch.cat([cu2, torch.tensor([n], dtype=cu.dtype, device=cu.device)])
    seg = (cu2[1:] - cu2[:-1])
    out = dict(batch)
    for k in ("input_ids", "position_ids", "labels", "mm_token_type_ids"):
        if isinstance(batch.get(k), torch.Tensor):
            out[k] = batch[k][:, :n].contiguous()
    out["cu_seq_lens_q"] = out["cu_seq_lens_k"] = cu2
    out["max_length_q"] = out["max_length_k"] = int(seg.max())
    return out


def apply_linear_lora(base_model: nn.Module, r: int = 8, alpha:int = 16):
    # no nn.Linear instance for up_proj, down_proj, gate_proj after attention block
    linear_layers = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
    for name, module in list(base_model.named_modules()):
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear) and child_name in linear_layers:
                if 'vision' in name:
                    continue

                lora = LinearLoRA(child_module, r, alpha)
                setattr(module, child_name, lora)

def make_custom_cls(base_cls, dpo_beta=None):
    """Build a CustomForConditionalGeneration subclass of the resolved base class
    (dense or MoE) that overrides forward to compute the loss without ever
    materializing the lm_head logits. Same forward for both variants.

    dpo_beta=None → the Liger fused-linear-CE SFT loss. dpo_beta set → the fused
    multipacked DPO loss (triton_dpo): the batch carries `seq_boundaries` (cu_seqlens,
    first half of the sequences chosen) + pre-aligned `labels` targets, the reference
    log-probs come from a no-grad forward of THIS model with LoRA disabled, and the
    lm_head weight serves both sides (it isn't LoRA-wrapped, so policy head == ref head)."""

    class CustomForConditionalGeneration(base_cls):
        def __init__(self, config):
            super().__init__(config)
            self.dpo_beta = dpo_beta
            if dpo_beta is None:
                from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
                self.loss_fn = LigerFusedLinearCrossEntropyLoss()
            else:
                from triton_dpo import fused_dpo_loss
                self.loss_fn = fused_dpo_loss

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            pixel_values: torch.Tensor | None = None,
            pixel_values_videos: torch.FloatTensor | None = None,
            image_grid_thw: torch.LongTensor | None = None,
            video_grid_thw: torch.LongTensor | None = None,
            mm_token_type_ids: torch.IntTensor | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            seq_boundaries: torch.LongTensor | None = None,
            **kwargs
        ):

            # Text-only packed training: pass just the essentials + **kwargs (carries the packing
            # metadata cu_seq_lens_q/k + max_length_q/k through to the attention). The multimodal
            # inputs are all None here, and per_layer_inputs must NOT be passed — the model computes
            # it internally and re-passes it to the language model ("got multiple values for keyword
            # argument 'per_layer_inputs'" otherwise).
            def _backbone():
                return self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    mm_token_type_ids=mm_token_type_ids,
                    **kwargs,
                )

            outputs = _backbone()
            hidden_states = outputs.last_hidden_state
            # overwrite to disable the logits materializatioon

            if self.dpo_beta is not None:
                assert seq_boundaries is not None and labels is not None, \
                    "DPO needs seq_boundaries + pre-aligned targets (the dpo_collator batch)"
                # Frozen reference = same weights, LoRA bypassed (base is frozen; B=0
                # init ⇒ ref == initial policy). Second backbone pass, no grad.
                with torch.no_grad(), lora_disabled():
                    ref_hidden = _backbone().last_hidden_state
                loss, chosen_rewards, rejected_rewards = self.loss_fn(
                    hidden_states[0], ref_hidden[0], labels[0], seq_boundaries,
                    self.lm_head.weight, self.lm_head.weight, beta=self.dpo_beta,
                )
                return {
                    "loss": loss,
                    "chosen_rewards": chosen_rewards,
                    "rejected_rewards": rejected_rewards,
                }

            loss = None
            if labels is not None:
                if cp.cp_active():
                    # Context parallel: `labels` are already per-token NEXT-token targets (pre-shifted
                    # per-doc in cp.shard_batch), because the usual global hidden[:-1]/labels[1:] shift
                    # is invalid across contiguous rank boundaries. Align hidden↔labels 1:1.
                    hs = hidden_states.contiguous().reshape(-1, hidden_states.shape[-1])
                    lbl = labels.contiguous().reshape(-1)
                    loss = self.loss_fn(self.lm_head.weight, hs, lbl)
                else:
                    # B, S, D -> (B*S, D)
                    shifted_hidden_states = hidden_states[:, :-1, :].contiguous().reshape(-1, hidden_states.shape[-1])
                    shifted_labels = labels[:, 1:].contiguous().reshape(-1)
                    loss = self.loss_fn(
                        self.lm_head.weight,
                        shifted_hidden_states,
                        shifted_labels,
                    )
            else:
                raise NotImplementedError("Loss calculation is not implemented yet.")

            return {
                "loss": loss
            }

    return CustomForConditionalGeneration


# Process-wide LoRA bypass: the DPO reference forward runs THIS model with every
# LoRA branch skipped. The base weights are frozen (requires_grad=False) and only
# the adapters train, so base+no-LoRA IS the frozen initial policy — no second
# model copy in memory. A plain python flag: no FSDP/AC interaction.
_LORA_ENABLED = True


@contextmanager
def lora_disabled():
    global _LORA_ENABLED
    _LORA_ENABLED = False
    try:
        yield
    finally:
        _LORA_ENABLED = True


class LinearLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_a = nn.Linear(in_features, r, bias=False, dtype=torch.bfloat16)
        self.lora_b = nn.Linear(r, out_features, bias=False, dtype=torch.bfloat16)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_a.weight)
            nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        out_non_lora = self.linear(x)
        if not _LORA_ENABLED:  # DPO reference pass (see lora_disabled)
            return out_non_lora
        lora_out = self.lora_b(self.lora_a(x))
        return out_non_lora + self.scaling * lora_out

def main(
        r:int=256,
        alpha: int=512,
        batch_size:int = 2,
        max_steps:int = 0,
        checkpointing_step:int = 100,
        limit_samples:int = 0,
        max_epochs:int = 1,
        lr:float = 1e-4,
        use_wandb:bool = False,
        wandb_project:str = "qwen3.5-autotrain",
        use_mlflow:bool = False,
        mlflow_experiment:str = "qwen3.5-autotrain",
        model_id:str = "Qwen/Qwen3.6-27B",
        checkpoint_dir:str = "checkpointing",
        grad_accum:int = 1,
        cpu_offload:bool = False,
        cp_size:int = 1,
        dpo:bool = False,
        dpo_beta:float = 0.1,
        train_embeddings:bool = False,
    ):
    if dpo and cp_size > 1:
        # per-sequence log-prob sums (and the pairing) live on whole sequences; a CP
        # shard only sees a chunk — would need a cross-rank logprob reduction + coeff
        # broadcast that isn't wired. Reject up front instead of training garbage.
        raise RuntimeError("--dpo is incompatible with --cp_size > 1 (context parallelism)")
    if dpo and train_embeddings:
        # The DPO reference = THIS model with LoRA disabled, only the frozen INITIAL policy
        # if the base stays frozen. Training embed_tokens/lm_head (the reference shares the
        # head) makes the reference drift with the policy → a wrong DPO objective.
        raise RuntimeError("--train_embeddings is incompatible with --dpo (the LoRA-disabled "
                           "reference assumes the base weights stay frozen)")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # Pin this process to its GPU BEFORE init'ing NCCL / the device mesh, otherwise every
    # rank lands on cuda:0 (NCCL hang / OOM).
    torch.cuda.set_device(rank)
    ddp_setup()  # init_process_group(nccl) — required before init_device_mesh / fully_shard
    # FSDP shards params across ALL ranks (1D mesh). Context parallelism is a SEPARATE process group
    # over cp_size consecutive ranks (sequence sharded + GDN state / full-attn KV relayed within it);
    # data parallelism is across CP groups. Orthogonal to FSDP.
    mesh_device = init_device_mesh("cuda", (world_size, ), mesh_dim_names=("shard", ))
    cp_group = dp_size = dp_rank = cp_rank = None
    attn_impl = "kernels-community/flash-attn3"
    if cp_size > 1:
        cp_group, dp_size, dp_rank, cp_rank = cp.setup_cp(world_size, cp_size, rank, torch.device(f"cuda:{rank}"))
        attn_impl = "cp_full_attention"   # full-attn (softmax) layers ring; GDN layers patched below
        logger.info(f"[cp] cp_size={cp_size} dp_size={dp_size} (world={world_size})")
        if os.environ.get("SGPU_MEM_PROFILE") == "1" and rank == 0:
            torch.cuda.memory._record_memory_history()   # dump: /share/qwen_cp_mem_r0.pickle at i0/idx1
    else:
        torch.cuda.memory._record_memory_history()

    # Resolve dense vs MoE classes from the config + apply the matching Liger patches.
    config, is_moe, classes = resolve_arch(model_id)
    CustomCls = make_custom_cls(classes["base"], dpo_beta=(dpo_beta if dpo else None))
    if dpo:
        logger.info(f"[dpo] objective=DPO beta={dpo_beta} — fused multipacked loss, "
                    f"reference = frozen base (LoRA disabled), expect first loss ≈ ln2 = 0.693")
    model = CustomCls.from_pretrained(
        model_id,
        config=config,
        dtype=torch.bfloat16, # native bf16 training
        attn_implementation = attn_impl
    )
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {total_params/(1024*1024):.0f}M")

    for param in model.parameters():
        param.requires_grad = False
    apply_linear_lora(model, r=r, alpha=alpha)

    # Optionally FULL-train the token embeddings + LM head on top of LoRA. Qwen3.6-27B is
    # UNTIED (embed_tokens + lm_head are two weights, both unfrozen); smaller/tied qwens get
    # one weight. Done BEFORE FSDP sharding / CP replication. See tc.unfreeze_embeddings.
    if train_embeddings:
        tc.unfreeze_embeddings(model, rank=rank, logger=logger)

    total_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info(f"Total trainable parameters: {total_trainable_params/(1024*1024):.2f}M")

    # ── Parallelism ────────────────────────────────────────────────────────────────────────────
    # LoRA + Context Parallel: REPLICATE the frozen base per GPU (NO FSDP) and shard only the sequence.
    # Rationale (the fix for the dp==1 CP+FSDP deadlock — see qwen-cp-deadlock diagnosis): FSDP-sharding
    # a FROZEN base saves ~nothing (no base gradient, no base optimizer state — only the tiny LoRA
    # trains) but its per-layer all-gather/reduce-scatter + iter-1 prefetch get ENQUEUED in a different
    # order than the CP ring/relay on the shared default PG → NCCL wedges at step 1. Removing FSDP leaves
    # the default PG carrying ONLY the (symmetric) full-attn ring + the step-end LoRA-grad all-reduce →
    # nothing can reorder against them → deadlock-free at ANY cp_size (500k = one CP group over N GPUs).
    # qwen-27B is ~54GB bf16 → fits one 143GB H20 with ample room for the 1/cp_size sequence-sharded,
    # activation-checkpointed activations. SGPU_CP_FSDP=1 forces the old FSDP path (e.g. a base too big
    # to replicate — then the deadlock is back; use dp>1 or a smaller cp_size).
    replicate_cp = cp_size > 1 and os.environ.get("SGPU_CP_FSDP") != "1"
    if replicate_cp:
        model = model.to(f"cuda:{rank}")
        # Replicated training REQUIRES identical params on every rank (grads are all-reduced). The base
        # comes identically from disk, but each process randomly initializes its LoRA A matrices —
        # broadcast every trainable param from rank 0 so all replicas start in lockstep.
        for p in model.parameters():
            if p.requires_grad:
                dist.broadcast(p.data, src=0)
        logger.info("[cp] FSDP DISABLED — frozen base REPLICATED per GPU, sequence sharded across the "
                    "CP group (deadlock-free; no FSDP collectives to reorder against the ring/relay); "
                    "LoRA init broadcast from rank 0.")
    else:
        # FSDP2 shard + activation-checkpoint via the shared helper. Qwen3.5 is multimodal — shard the
        # decoder AND vision blocks (reshard_after_forward); param_dtype=bf16 dense; cpu_offload opt-in.
        kw = tc.fsdp_kwargs(mesh_device, param_dtype=torch.bfloat16, cpu_offload=cpu_offload)
        tc.shard_layers(model, (classes["decoder"], classes["vision"]), kw, reshard_after_forward=True)
        model_sd = model.state_dict()
        local_shard = sum(v.to_local().numel() if hasattr(v, "to_local") else v.numel() for _, v in model_sd.items())
        logger.info(f"[Rank{rank}]: Total local/shard param: {local_shard/(1024*1024):.2f}M")

    # Activation checkpointing (NO_REENTRANT) — the DOMINANT memory lever (64 layers × intermediate
    # 17408 MLP intermediates); disabling it is what forced the CP OOM at long context. ON by default
    # (both the replicate-CP and FSDP paths); SGPU_CP_NO_AC=1 forces it off for A/B debugging.
    if cp_size <= 1 or os.environ.get("SGPU_CP_NO_AC") != "1":
        tc.checkpoint_layers(model, classes["decoder"])
    else:
        logger.info("[cp] activation checkpointing force-disabled (SGPU_CP_NO_AC=1)")

    def _patched_chunk_gated_delta_rule(q, k, v, *args, **kwargs):
        # TileLang kernel requires v to be contiguous (stride[-1] == 1)
        v = v.contiguous()
        return chunk_gated_delta_rule(q, k, v, *args, **kwargs)

    # use qwen flash qla
    for module in model.modules():
        if isinstance(module, classes["gdn"]):
            module.chunk_gated_delta_rule = _patched_chunk_gated_delta_rule

    # Context parallel: wrap each GDN layer's conv + delta-rule kernels to relay their state across the
    # CP group (one differentiable pair-P2P per layer). AFTER the contiguous-v patch so the relay wraps
    # the patched (flash_qla TileLang) kernel.
    if cp_size > 1:
        cp.install_gdn_cp(model, classes["gdn"])

    # max_steps > 0 caps the run regardless of epochs (0 = run all max_epochs). For overfitting a
    # small dataset, set max_epochs high and max_steps 0.
    dataset = Dataset(limit=limit_samples)
    # Under CP the whole CP group must see the SAME bin (each rank trains a different contiguous shard
    # of it), so the sampler is DP-based: num_replicas = #CP-groups (dp_size), rank = this rank's group.
    sampler = DistributedSampler(
        dataset,
        num_replicas=(dp_size if cp_size > 1 else mesh_device.size()),
        rank=(dp_rank if cp_size > 1 else mesh_device.get_rank()),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=(dpo_collator if dpo else collator),
        sampler=sampler,
        prefetch_factor=4,
        num_workers=4,
    )

    # Optimize ONLY the trainable LoRA params. fused=False: with CPUOffloadPolicy the
    # optimizer step runs on CPU-resident params, and fused AdamW is CUDA-only.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, fused=False)
    model.train()

    # ---- wandb (rank 0 only) ----
    wandb_run = None
    if use_wandb and rank == 0:
        import wandb
        wandb_run = wandb.init(
            project=wandb_project,
            config={
                "model": model_id, "is_moe": is_moe, "r": r, "alpha": alpha,
                "batch_size": batch_size, "lr": lr, "max_epochs": max_epochs,
                "max_steps": max_steps,
                "limit_samples": limit_samples, "world_size": world_size,
                "num_bins": len(dataset), "trainable_params_M": round(total_trainable_params/1e6, 2),
            },
        )
        logger.info(f"wandb run: {wandb_run.url}")
    # ---- mlflow (rank 0 only) ----
    if use_mlflow and rank == 0:
        import mlflow
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run(run_name=f"{model_id.split('/')[-1]}_r{r}_alpha{alpha}_bs{batch_size}_lr{lr}")
        mlflow.log_params({
            "model": model_id, "is_moe": is_moe, "r": r, "alpha": alpha,
            "batch_size": batch_size, "lr": lr, "max_epochs": max_epochs,
            "max_steps": max_steps,
            "limit_samples": limit_samples, "world_size": world_size,
            "num_bins": len(dataset), "trainable_params_M": round(total_trainable_params/1e6, 2),
        })

    # seen_tokens = torch.tensor(0, dtype=torch.long, device=f'cuda:{rank}')
    # opt_step counts OPTIMIZER steps (one per grad_accum microbatches), so max_steps /
    # checkpointing / the @@STEP progress track weight updates, not forward passes.
    opt_step = 0
    reached_max = False
    # Running count of contributing (non -100) label tokens in the current accumulation
    # window on THIS rank; summed across ranks each optimizer step for token-weighted loss.
    win_tokens = torch.zeros((), device=f'cuda:{rank}', dtype=torch.long)
    for i in range(max_epochs):
        sampler.set_epoch(i)  # reshuffle each epoch
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if rank == 0:
                start_time = time.time()

            # Move the packed batch onto this rank's GPU (collator builds CPU tensors).
            # Python ints (max_length_q/k) are left as-is.
            batch = {
                k: (v.to(f'cuda:{rank}', non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }

            # Optional context cap (SGPU_MAX_SEQ_LEN) — truncate to a multiple of cp_size so the CP
            # shard is even. Off (0) by default. Never under DPO: cutting docs would break the
            # whole-pair layout (and seq_boundaries) the fused loss depends on.
            if _MAX_SEQ_LEN and not dpo:
                batch = _truncate_packed(batch, _MAX_SEQ_LEN, mult=max(cp_size, 1))

            # Context parallel: replace the full packed batch with THIS rank's contiguous chunk (+ set
            # the per-token position/doc ids the full-attn ring reads; labels pre-shifted per-doc).
            if cp_size > 1:
                batch = cp.shard_batch(batch, cp_size, cp_rank)

            output = model(**batch, use_cache=False) # forward pass and calculate losses
            # Back-prop the SUM loss (mean × #units). Accumulating sums, then normalizing
            # by the GLOBAL unit count at the step below, yields the exact weighted mean
            # over the whole effective batch — not a naive mean-of-means that mis-weights
            # variable-length bins / ranks (HF grad-accum loss fix). The unit is the label
            # TOKEN for SFT (the loss is a token-mean) and the preference PAIR for DPO
            # (fused_dpo_loss is a pair-mean — weighting by tokens would re-introduce a
            # length bias into the pairwise objective).
            # Under CP labels are already pre-shifted per-token targets (1:1), so count them directly;
            # otherwise count the shifted denominator (labels[:,1:]) the loss_fn uses.
            if dpo:
                n_tok = torch.tensor((batch["seq_boundaries"].numel() - 1) // 2,
                                     device=f'cuda:{rank}', dtype=torch.long)
            else:
                n_tok = ((batch["labels"] != -100).sum() if cp_size > 1
                         else (batch["labels"][:, 1:] != -100).sum())
            (output["loss"] * n_tok).backward() # calculate gradient
            win_tokens += n_tok

            # Step once every grad_accum microbatches; always flush a partial window at
            # epoch end so no gradient is dropped. grad_accum=1 → step every microbatch.
            do_step = ((idx + 1) % grad_accum == 0) or (idx == len(dataloader) - 1)
            if do_step:
                # Token-weighted loss: sum the window's label tokens across ALL ranks → N_total.
                dist.all_reduce(win_tokens, op=dist.ReduceOp.SUM)
                if replicate_cp:
                    # No FSDP → grads are this rank's chunk grads (NOT averaged). All-reduce-SUM each
                    # LoRA grad over ALL ranks (default PG), then ÷ N_total → the exact token-mean over
                    # the whole effective batch (every chunk of every row). Symmetric collective on the
                    # default PG (no FSDP collectives to interleave) → deadlock-free. No world_size term
                    # (nothing averaged the grads, unlike FSDP's reduce-scatter-mean).
                    for p in trainable_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    scale = (1.0 / win_tokens.clamp(min=1)).item()
                else:
                    # FSDP reduce-scatters grads as a MEAN over world_size; ×world_size cancels that,
                    # ÷N_total gives the token-mean (HF average_tokens_across_devices).
                    scale = (world_size / win_tokens.clamp(min=1)).item()
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            p.grad.mul_(scale)
                optimizer.step()
                optimizer.zero_grad()
                win_tokens.zero_()

            # synchronize
            # last_seen_tokens = seen_tokens.item()
            token_count = batch["input_ids"].numel()
            token_count = torch.tensor(token_count, device=f'cuda:{rank}')
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
            # seen_tokens += token_count

            if rank == 0:
                loss = output['loss'].item()
                delta_time = time.time() - start_time
                tps = token_count.item() / delta_time
                # DPO extras: reward accuracy (chosen > rejected) + margin. Appended AFTER
                # loss so the orchestrator's "step: N … loss: L" parser keys on unchanged.
                dpo_extra = ""
                if dpo:
                    cr, rr = output["chosen_rewards"], output["rejected_rewards"]
                    reward_acc = (cr > rr).float().mean().item()
                    reward_margin = (cr - rr).mean().item()
                    dpo_extra = f", reward_acc: {reward_acc:.3f}, margin: {reward_margin:.4f}"
                logger.info(f"Epoch: {i}, mb: {idx}, step: {opt_step}, loss: {loss}, tokens/s: {tps:.2f}{dpo_extra}")
                # Post-microbatch CUDA residue — the leak/retention tell for long-context CP: after
                # backward+step the allocator should be back near the steady floor; a climbing series
                # here means graph/activation retention across steps.
                logger.info(f"[mem] mb {idx}: allocated={torch.cuda.memory_allocated()/2**30:.2f}GB "
                            f"reserved={torch.cuda.memory_reserved()/2**30:.2f}GB")
                metrics = {"loss": loss, "lr": optimizer.param_groups[0]['lr'], "tps": tps, "epoch": i}
                if dpo:
                    metrics.update({"reward_acc": reward_acc, "reward_margin": reward_margin})
                if wandb_run is not None and do_step:
                    try:
                        wandb_run.log(metrics, step=opt_step)
                    except Exception as e:
                        logger.warning(f"wandb.log failed (continuing): {e}")
                if use_mlflow and do_step:
                    try:
                        mlflow.log_metrics(metrics, step=opt_step)
                    except Exception as e:
                        logger.warning(f"mlflow.log_metrics failed (continuing): {e}")

            if i == 0 and idx == 1 and cp_size <= 1:  # history only recorded in the non-CP branch
                torch.cuda.memory._dump_snapshot(f"memory_profile_{rank}.pickle")
            # CP memory profiling (SGPU_MEM_PROFILE=1): dump rank 0's allocation history after the
            # SECOND microbatch — captures whatever step 0 left resident (the OOM residue) with
            # allocation stacks. Absolute path: the run's workdir is cleaned on failure.
            if (i == 0 and idx == 1 and cp_size > 1 and rank == 0
                    and os.environ.get("SGPU_MEM_PROFILE") == "1"):
                torch.cuda.memory._dump_snapshot("/share/qwen_cp_mem_r0.pickle")
                logger.info("[mem] snapshot dumped to /share/qwen_cp_mem_r0.pickle")

            reached_max = max_steps > 0 and do_step and (opt_step + 1) >= max_steps
            if idx == len(dataloader)-1 or (do_step and (idx + 1) % checkpointing_step == 0) or reached_max:
                logger.info("Checkpointing LoRA adapters..")
                # Save only the trainable LoRA params. full_tensor() is a collective, so every rank
                # must iterate the SAME params; requires_grad is identical across ranks.
                lora_state_dict = {}
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    full_param = param.full_tensor() if hasattr(param, "full_tensor") else param
                    if rank == 0:
                        lora_state_dict[name] = full_param.detach().to('cpu')
                if rank == 0:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(lora_state_dict, f"{checkpoint_dir}/lora.pt")
                    with open(f"{checkpoint_dir}/lora_meta.json", "w") as f:
                        json.dump({
                            "model_id": model_id,
                            "is_moe": is_moe,
                            "r": r, "alpha": alpha, "scaling": alpha / r,
                            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                            "wrapped_attr": "linear",
                            "train_embeddings": train_embeddings,
                            "objective": "dpo" if dpo else "sft",
                            **({"dpo_beta": dpo_beta} if dpo else {}),
                        }, f, indent=2)
                    logger.info(f"Saved LoRA ({len(lora_state_dict)} tensors) to {checkpoint_dir}/lora.pt")

            if do_step:
                opt_step += 1
            if reached_max:
                logger.info(f"Reached max_steps={max_steps}, stopping.")
                break
        if reached_max:
            break

    if wandb_run is not None:
        wandb_run.finish()
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # `--lora_r` is the long form the gateway orchestrator passes (llm_finetune._qwen_cmd);
    # `--rank` is the standalone alias. NOT a bare `--r`: torchrun (torch.distributed.run)
    # prefix-matches a bare `--r` against its own options and aborts "ambiguous option" on
    # torch 2.12.x (see gemma4). dest="rank" keeps main(args.rank) working.
    parser.add_argument(
        "--lora_r", "--rank",
        dest="rank",
        type=int,
        default=256,
        help="LoRA rank"
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=512,
        help="LoRA alpha scaling factor"
    )
    # batch_size / grad_accum / cpu_offload / max_epochs / max_steps / checkpointing_step
    # / limit_samples / lr / wandb[_project] — shared across all LLM trainers.
    tc.add_common_args(parser, lr_default=1e-4, wandb_project="qwen3.5-autotrain")
    parser.add_argument("--mlflow", action="store_true", help="Log metrics to MLflow.")
    parser.add_argument("--mlflow_experiment", default="qwen3.5-autotrain", help="mlflow experiment name.")
    parser.add_argument(
        "--model_id",
        # The gateway orchestrator (llm_finetune.run) selects the model via the MODEL_ID
        # env var (spec["model_env"]="MODEL_ID"), like the minimax/mistral trainers, so a
        # run picks dense (Qwen3.6-27B) vs MoE (Qwen3.6-35B-A3B) without a CLI flag.
        default=os.environ.get("MODEL_ID", "Qwen/Qwen3.6-27B"),
        help="HF model id to finetune. Dense (Qwen3_5ForConditionalGeneration) or MoE "
             "(Qwen3_5MoeForConditionalGeneration) is auto-detected from the config. "
             "Qwen3.6 reuses the Qwen3.5 arch: Qwen/Qwen3.6-27B (dense), Qwen/Qwen3.6-35B-A3B (MoE).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpointing",
        help="Directory to save the LoRA adapters (lora.pt + lora_meta.json). Use a "
             "per-model dir so multiple models don't clobber each other.",
    )
    parser.add_argument(
        "--cp_size", type=int, default=1,
        help="Context-parallel group size (>1 shards one packed sequence across cp_size GPUs and "
             "relays GatedDeltaNet state + rings the full-attn KV; data-parallel across CP groups). "
             "1 = off. The gateway sets it to the GPU count when the run enables context parallelism.",
    )
    parser.add_argument(
        "--dpo", action="store_true",
        help="Direct Preference Optimization over a DPO-packed dataset (kind=llm_dpo_packed: whole "
             "preference pairs per bin, chosen-first layout, pre-aligned targets). Loss = the fused "
             "multipacked DPO loss (triton_dpo); reference model = this model with LoRA disabled. "
             "Incompatible with --cp_size > 1.",
    )
    parser.add_argument(
        "--dpo_beta", type=float, default=0.1,
        help="DPO temperature β on the policy/reference log-ratio (only with --dpo).",
    )
    parser.add_argument(
        "--train_embeddings", action="store_true",
        help="Also FULL-train the token embeddings + LM head (Qwen3.6-27B is untied → both "
             "weights), not just LoRA. Helps the finetune reliably emit special tokens that "
             "attention-only LoRA can only nudge. Costs extra optimizer state — combine with "
             "--cpu_offload. Incompatible with --dpo.",
    )
    args = parser.parse_args()
    main(
        args.rank,
        args.alpha,
        args.batch_size,
        args.max_steps,
        args.checkpointing_step,
        args.limit_samples,
        args.max_epochs,
        args.lr,
        args.wandb,
        args.wandb_project,
        args.mlflow,
        args.mlflow_experiment,
        args.model_id,
        args.checkpoint_dir,
        args.grad_accum,
        args.cpu_offload,
        args.cp_size,
        args.dpo,
        args.dpo_beta,
        args.train_embeddings,
    )
