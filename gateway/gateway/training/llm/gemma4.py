from transformers import (
    AutoConfig,
    Gemma4ForConditionalGeneration, 
    AttentionInterface, 
    AutoTokenizer
)
from transformers.models.gemma4 import modeling_gemma4
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
)
import os
import json
import time
from functools import partial
from tqdm import tqdm
import argparse
from typing import Optional
import mlflow
from chinidataset import StreamingDataset
import _trainer_common as tc
from contextlib import nullcontext, contextmanager
from attention import dynamic_attention, block_diagonal_concat
from gemma4_fa4_attention import fa4_attention
import context_parallel as cp  # zigzag ring context-parallelism (opt-in via --cp_size > 1)

logging.basicConfig(
    level=logging.INFO, 
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Base model. Defaults to gemma-4-31B-it (standalone runs unchanged); the gateway
# autotrain runner overrides it via GEMMA_MODEL_ID so a run can pick a gemma-4 size.
MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-31B-it")


# Two interchangeable packed-attention backends, selected by env GEMMA_ATTN:
#   "dynamic_attention" (attention.py)  — head_dim-512 via tiled SDPA-math (O(S^2) score → ~32k ceiling, needs FA3)
#   "fa4_attention" (gemma4_fa4_attention.py) — ALL layers through the FA4 head_dim-512 cute kernel
#       (flash-attention-512 fork): memory-efficient O(S), lifts the ceiling to 128k. THE FAST PATH.
# fa4_attention's flash_attn.cute import is lazy (inside the fn), so registering it is safe even if
# the FA4 package isn't installed; it only fires when that backend is actually used.
AttentionInterface.register("dynamic_attention", dynamic_attention)
AttentionInterface.register("fa4_attention", fa4_attention)
# Context-parallel hybrid ring backend: head_dim-512 global → fused zigzag ring (cute), head_dim-256
# sliding → position-aware ring. Selected automatically when cp_size > 1.
AttentionInterface.register("cp_ring_attention", cp.cp_ring_attention)
ATTN_IMPL = os.environ.get("GEMMA_ATTN", "fa4_attention")


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
        # Gemma-4 is multimodal; create_causal_mask_mapping requires mm_token_type_ids during
        # training. Text-only packing => all-zero (every token is a text token).
        'mm_token_type_ids': torch.zeros_like(input_ids_t),
        'labels': torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        'cu_seq_lens_q': cu_seq_lens_q,
        'cu_seq_lens_k': cu_seq_lens_q,
        'max_length_q': max_seqlen_q,
        'max_length_k': max_seqlen_q
    }

def dpo_collator(batch):
    """Packed DPO microbatch (mirrors qwen3_5.dpo_collator). Each DPO bin holds K whole
    preference pairs as 2K docs, first K chosen then K rejected (llm_pack.collate_dpo_bin).
    When batch_size > 1 concatenates several bins, the docs are REORDERED so ALL chosen
    sequences come first across the whole row — preserving triton_dpo's pair contract
    (pair k = (seq k, seq K_total+k); chosen/rejected keep the same bin order, so pairing
    stays aligned). `labels` are pre-aligned next-token targets (no shift at loss time);
    `seq_boundaries` (int64 cu_seqlens) drives the fused DPO loss."""
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
        'input_ids': input_ids_t,
        'position_ids': torch.tensor(position_ids, dtype=torch.long).unsqueeze(0),
        'attention_mask': None,
        'mm_token_type_ids': torch.zeros_like(input_ids_t),
        'labels': torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        'seq_boundaries': torch.tensor(cumsum, dtype=torch.long),
        'cu_seq_lens_q': cu_seq_lens_q,
        'cu_seq_lens_k': cu_seq_lens_q,
        'max_length_q': int(np.max(query_lens)),
        'max_length_k': int(np.max(query_lens)),
    }


DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

def apply_linear_lora(base_model: nn.Module, r: int = 8, alpha: int = 16, target_modules=None):
    # gemma-4's q/k/v/o are Gemma4ClippableLinear (an nn.Linear subclass), so isinstance
    # matches them. The MLP projections (gate_proj/up_proj/down_proj) may NOT be plain
    # nn.Linear on every arch — so any requested target that wraps nothing is reported
    # (warning) instead of silently ignored, making a bad selection visible.
    target_modules = list(target_modules or DEFAULT_LORA_TARGETS)
    wrapped = {t: 0 for t in target_modules}
    for name, module in list(base_model.named_modules()):
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear) and child_name in target_modules:
                if 'vision' in name:
                    continue

                lora = LinearLoRA(child_module, r, alpha)
                setattr(module, child_name, lora)
                wrapped[child_name] = wrapped.get(child_name, 0) + 1
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        logger.info(f"LoRA target_modules={target_modules}; wrapped per module: {wrapped}")
        zero = [t for t, n in wrapped.items() if n == 0]
        if zero:
            logger.warning(f"LoRA targets matched NO nn.Linear modules and will NOT be trained: "
                           f"{zero} — this architecture may fuse/rename them.")
    return wrapped

class CustomGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        self.loss_fn = LigerFusedLinearCrossEntropyLoss()
        self.dpo_beta = None  # set by enable_dpo() → the DPO forward branch

    def enable_dpo(self, beta: float) -> None:
        """Switch to the fused multipacked DPO objective (triton_dpo.fused_dpo_loss —
        logits never materialized; see small-ablation/multipacking-dpo). The reference
        model is THIS model with LoRA disabled (base frozen + B=0 init ⇒ ref == initial
        policy), so no second model copy is loaded. Call before FSDP sharding."""
        from triton_dpo import fused_dpo_loss
        self.dpo_beta = beta
        self.dpo_loss_fn = fused_dpo_loss

    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            pixel_values: torch.FloatTensor | None = None,
            pixel_values_videos: torch.FloatTensor | None = None,
            input_features: torch.FloatTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            input_features_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            image_position_ids: torch.LongTensor | None = None,
            video_position_ids: torch.LongTensor | None = None,
            past_key_values = None,
            mm_token_type_ids: torch.LongTensor | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            use_cache: bool | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            per_layer_inputs: torch.Tensor | None = None,
            seq_boundaries: torch.LongTensor | None = None,
            **kwargs
        ):

        # Text-only packed training: pass just the essentials + **kwargs (carries the packing
        # metadata cu_seq_lens_q/k + max_length_q/k through to dynamic_attention). The multimodal
        # inputs are all None here, and per_layer_inputs must NOT be passed — Gemma4Model computes
        # it internally and re-passes it to the language model ("got multiple values for keyword
        # argument 'per_layer_inputs'" otherwise). seq_boundaries is pulled out here (DPO-only) so
        # it doesn't reach the attention as a stray kwarg.
        def _backbone():
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                return_dict=True,
                **kwargs,
            )

        outputs = _backbone()
        hidden_states = outputs.last_hidden_state
        # overwrite to disable the logits materializatioon

        if self.dpo_beta is not None:
            assert seq_boundaries is not None and labels is not None, \
                "DPO needs seq_boundaries + pre-aligned targets (the dpo_collator batch)"
            # Frozen reference = same weights, LoRA bypassed (base is frozen; B=0 init ⇒
            # ref == initial policy). Second backbone pass, no grad. lm_head isn't LoRA-
            # wrapped so one weight serves both policy and reference heads.
            with torch.no_grad(), lora_disabled():
                ref_hidden = _backbone().last_hidden_state
            loss, chosen_rewards, rejected_rewards = self.dpo_loss_fn(
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
                # is invalid across zigzag chunk boundaries. So align hidden↔labels 1:1 (no shift).
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

# Process-wide LoRA bypass: the DPO reference forward runs THIS model with every LoRA
# branch skipped. The base weights are frozen (requires_grad=False) and only the adapters
# train, so base+no-LoRA IS the frozen initial policy — no second model copy in memory.
# A plain python flag: no FSDP/AC interaction (mirrors qwen3_5.lora_disabled).
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
            # CRITICAL: lora_b MUST be zero so the adapter is a no-op at init
            # (delta = scaling * B @ A = 0). Without this, every q/k/v/o starts with a large
            # RANDOM perturbation (x scaling) -> the model trains from a corrupted state and the
            # merged result is degenerate (the bug that produced 0 accuracy / <|"|> garbage).
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
        wandb_project:str = "gemma4-autotrain",
        grad_accum:int = 1,
        cpu_offload:bool = False,
        target_modules=None,
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
        # The DPO reference = THIS model with LoRA disabled, which is only the frozen
        # INITIAL policy if the base weights stay frozen. Training embed_tokens/lm_head
        # (shared with the reference's head) makes the reference drift with the policy →
        # a wrong DPO objective. Would need a separate frozen reference copy (not wired).
        raise RuntimeError("--train_embeddings is incompatible with --dpo (the LoRA-disabled "
                           "reference assumes the base weights stay frozen)")
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # Pin this process to its GPU BEFORE init'ing NCCL / the device mesh, otherwise every
    # rank lands on cuda:0 (NCCL hang / OOM).
    torch.cuda.set_device(rank)
    ddp_setup()  # init_process_group(nccl) — required before init_device_mesh / fully_shard
    # FSDP shards params across ALL ranks (1D mesh, unchanged). Context parallelism is a SEPARATE
    # process group over cp_size consecutive ranks (sequence sharded + KV ringed within it);
    # data parallelism is across CP groups. The two are orthogonal — FSDP all-gathers params over
    # every rank regardless of the CP grouping.
    mesh_device = init_device_mesh("cuda", (world_size, ), mesh_dim_names=("shard", ))
    cp_group = dp_size = dp_rank = cp_rank = None
    attn_impl = ATTN_IMPL
    if cp_size > 1:
        cp_group, dp_size, dp_rank, cp_rank = cp.setup_cp(world_size, cp_size, rank)
        attn_impl = "cp_ring_attention"
        if rank == 0:
            logger.info(f"context parallel: cp_size={cp_size} dp_size={dp_size} (world={world_size})")

    config = AutoConfig.from_pretrained(MODEL_ID)
    if rank == 0:
        logger.info(f"attention backend: {attn_impl}")
    # modify the confg
    model = CustomGemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        config=config,
        dtype=torch.bfloat16, # native bf16 training
        attn_implementation = attn_impl
    )
    if dpo:
        model.enable_dpo(dpo_beta)
        if rank == 0:
            logger.info(f"[dpo] objective=DPO beta={dpo_beta} — fused multipacked loss, "
                        f"reference = frozen base (LoRA disabled), expect first loss ≈ ln2 = 0.693")
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {total_params/(1024*1024):.0f}M")
    
    # Custom LinearLoRA (NOT PEFT): gemma-4's q/k/v/o are `Gemma4ClippableLinear`, which PEFT
    # refuses ("Target module ... not supported; only torch.nn.Linear ..."), and wrapping them as
    # plain Linear would drop the clipping. LinearLoRA wraps the real module and calls its forward,
    # preserving clipping. With the lora_b=0 init fix it no longer corrupts the model at start.
    for param in model.parameters():
        param.requires_grad = False
    target_modules = list(target_modules or DEFAULT_LORA_TARGETS)
    apply_linear_lora(model, r=r, alpha=alpha, target_modules=target_modules)

    # Optionally FULL-train the token embeddings + LM head (NOT LoRA — the real weights).
    # gemma-4 ties them (tie_word_embeddings=True) so it's ONE ~1.4B weight (vocab 262144 ×
    # hidden 5376). Done BEFORE FSDP sharding; pairs naturally with adding the MLP
    # (gate_proj,up_proj,down_proj) to --target_modules. See tc.unfreeze_embeddings.
    if train_embeddings:
        tc.unfreeze_embeddings(model, rank=rank, logger=logger)

    total_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info(f"Total trainable parameters: {total_trainable_params/(1024*1024):.2f}M")

    # FSDP2 shard + activation-checkpoint via the shared helper (param_dtype=bf16 dense;
    # cpu_offload is the opt-in VRAM↔speed knob — see _trainer_common.fsdp_kwargs).
    kw = tc.fsdp_kwargs(mesh_device, param_dtype=torch.bfloat16, cpu_offload=cpu_offload)
    tc.shard_layers(model, modeling_gemma4.Gemma4TextDecoderLayer, kw)

    # Get the local shard
    model_sd = model.state_dict()
    local_shard = sum(v.to_local().numel() if hasattr(v, "to_local") else v.numel() for _ , v in model_sd.items())
    logger.info(f"[Rank{rank}]: Total local/shard param: {local_shard/(1024*1024):.2f}M")

    tc.checkpoint_layers(model, modeling_gemma4.Gemma4TextDecoderLayer)
    
    # max_steps > 0 caps the run regardless of epochs (0 = run all max_epochs). For overfitting a
    # small dataset, set max_epochs high and max_steps 0.

    dataset = Dataset(limit=limit_samples)
    # Under CP the whole CP group must see the SAME bin (each rank trains a different sequence
    # chunk of it), so the sampler splits bins across DATA-PARALLEL groups only.
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
                "model": MODEL_ID, "r": r, "alpha": alpha,
                "batch_size": batch_size, "lr": lr, "max_epochs": max_epochs,
                "max_steps": max_steps,
                "limit_samples": limit_samples, "world_size": world_size,
                "num_bins": len(dataset), "trainable_params_M": round(total_trainable_params/1e6, 2),
            },
        )
        logger.info(f"wandb run: {wandb_run.url}")

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

            # Context parallel: replace the full packed batch with THIS rank's zigzag shard (and
            # set the local position/doc ids the ring attention masks read). n_tok below then counts
            # only this rank's real (non -100) targets; summed across ranks for the token-weighted loss.
            if cp_size > 1:
                batch = cp.shard_batch(batch, cp_size, cp_rank, pad_id=0)

            if rank == 0 and idx == 0:
                S = batch["input_ids"].shape[-1]
                alloc_gb = torch.cuda.memory_allocated(rank) / 2**30
                res_gb   = torch.cuda.memory_reserved(rank)  / 2**30
                logger.info(f"PRE-FWD: S={S}, allocated={alloc_gb:.2f}GB, reserved={res_gb:.2f}GB")
            output = model(**batch, use_cache=False) # forward pass and calculate losses
            # Back-prop the SUM loss (mean × #units). Accumulating sums, then normalizing by
            # the GLOBAL unit count at the step below, yields the exact weighted mean over the
            # whole effective batch — not a naive mean-of-means that mis-weights variable-length
            # bins / ranks (HF grad-accum loss fix). The unit is the label TOKEN for SFT (a
            # token-mean loss) and the preference PAIR for DPO (fused_dpo_loss is a pair-mean —
            # token-weighting would re-add a length bias into the pairwise objective).
            # CP labels are pre-shifted per-token targets (no [:,1:] shift); non-CP shifts.
            if dpo:
                n_tok = torch.tensor((batch["seq_boundaries"].numel() - 1) // 2,
                                     device=f'cuda:{rank}', dtype=torch.long)
            else:
                n_tok = ((batch["labels"] if cp_size > 1 else batch["labels"][:, 1:]) != -100).sum()
            (output["loss"] * n_tok).backward() # calculate gradient
            win_tokens += n_tok

            # Step once every grad_accum microbatches; always flush a partial window at
            # epoch end so no gradient is dropped. grad_accum=1 → step every microbatch.
            do_step = ((idx + 1) % grad_accum == 0) or (idx == len(dataloader) - 1)
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
                metrics = {"loss": loss, "lr": optimizer.param_groups[0]['lr'], "tps": tps, "epoch": i}
                if dpo:
                    metrics.update({"reward_acc": reward_acc, "reward_margin": reward_margin})
                if wandb_run is not None and do_step:
                    try:
                        wandb_run.log(metrics, step=opt_step)
                    except Exception as e:
                        logger.warning(f"wandb.log failed (continuing): {e}")

            reached_max = max_steps > 0 and do_step and (opt_step + 1) >= max_steps
            if idx == len(dataloader)-1 or (do_step and (idx + 1) % checkpointing_step == 0) or reached_max:
                logger.info("Checkpointing LoRA adapters..")
                # Save every trainable param — the LoRA adapters AND (with --train_embeddings)
                # the full embed_tokens/lm_head weight. full_tensor() is a collective, so every
                # rank must iterate the SAME params; requires_grad is identical across ranks.
                lora_state_dict = {}
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    full_param = param.full_tensor() if hasattr(param, "full_tensor") else param
                    if rank == 0:
                        lora_state_dict[name] = full_param.detach().to('cpu')
                if rank == 0:
                    os.makedirs("checkpointing", exist_ok=True)
                    torch.save(lora_state_dict, "checkpointing/lora.pt")
                    with open("checkpointing/lora_meta.json", "w") as f:
                        json.dump({
                            "model_id": MODEL_ID,
                            "r": r, "alpha": alpha, "scaling": alpha / r,
                            "target_modules": target_modules,
                            "wrapped_attr": "linear",
                            "train_embeddings": train_embeddings,
                            "objective": "dpo" if dpo else "sft",
                            **({"dpo_beta": dpo_beta} if dpo else {}),
                        }, f, indent=2)
                    logger.info(f"Saved LoRA ({len(lora_state_dict)} tensors) to checkpointing/lora.pt")

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
    parser.add_argument(
        "--r", "--lora_r",
        dest="r",
        type=int,
        default=256,
        # `--lora_r` alias: torchrun's argparse prefix-matches the bare `--r` against
        # its OWN options (--rdzv-*, --role, --run-path …) and aborts with "ambiguous
        # option" on some torch builds (2.12.0). The gateway passes `--lora_r`, which
        # collides with nothing; the standalone `--r` still works.
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
    tc.add_common_args(parser, lr_default=1e-4, wandb_project="gemma4-autotrain")
    parser.add_argument(
        "--target_modules",
        type=str,
        default=",".join(DEFAULT_LORA_TARGETS),
        help="Comma-separated linear module names to apply LoRA to. Default is the attention "
             "projections (q_proj,k_proj,v_proj,o_proj); add MLP/dense layers "
             "(gate_proj,up_proj,down_proj) to adapt those too.",
    )
    parser.add_argument(
        "--cp_size",
        type=int,
        default=1,
        help="Context-parallel (zigzag ring) degree: shard each packed sequence across this many "
             "GPUs (KV ringed between them) to train context longer than one GPU's VRAM. 1 = off. "
             "world_size must be divisible by it; dp_size = world_size / cp_size.",
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
        help="Also FULL-train the token embeddings + LM head (gemma-4 ties them → one "
             "~1.4B weight), not just LoRA. Helps the finetune reliably emit special tokens "
             "(e.g. <|tool_call>) that attention-only LoRA can only nudge. Costs extra VRAM/"
             "optimizer state — combine with --cpu_offload. Pairs well with adding the MLP "
             "(gate_proj,up_proj,down_proj) to --target_modules.",
    )
    args = parser.parse_args()
    main(
        args.r,
        args.alpha,
        args.batch_size,
        args.max_steps,
        args.checkpointing_step,
        args.limit_samples,
        args.max_epochs,
        args.lr,
        args.wandb,
        args.wandb_project,
        grad_accum=args.grad_accum,
        cpu_offload=args.cpu_offload,
        target_modules=[t.strip() for t in (args.target_modules or "").split(",") if t.strip()],
        cp_size=args.cp_size,
        dpo=args.dpo,
        dpo_beta=args.dpo_beta,
        train_embeddings=args.train_embeddings,
    )