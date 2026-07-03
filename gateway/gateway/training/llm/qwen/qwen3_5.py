"""Qwen3.5-27B LoRA finetune — FSDP2 (fully_shard) + CPU offload, packed varlen.

torchrun entrypoint. Wraps q/k/v/o_proj with a custom LinearLoRA (B=0 init), shards each
Qwen3_5DecoderLayer / Qwen3_5VisionBlock with FSDP2 under CPUOffloadPolicy, and trains on the
ChiniDataset packed bins produced by pack_dataset.py. Loss is LigerFusedLinearCrossEntropyLoss
(materialized lm_head logits are skipped). The Qwen3.5 GatedDeltaNet (linear attention) path runs
the FlashQLA `chunk_gated_delta_rule` kernel (v patched contiguous for the TileLang kernel).

Attention backend: kernels-community/flash-attn3 (auto-fetched by the `kernels` package).
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
from functools import partial
from tqdm import tqdm
import argparse
from chinidataset import StreamingDataset
from flash_qla import chunk_gated_delta_rule


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

def make_custom_cls(base_cls):
    """Build a CustomForConditionalGeneration subclass of the resolved base class
    (dense or MoE) that overrides forward to compute the Liger fused-linear-CE loss
    without ever materializing the lm_head logits. Same forward for both variants."""

    class CustomForConditionalGeneration(base_cls):
        def __init__(self, config):
            super().__init__(config)
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            self.loss_fn = LigerFusedLinearCrossEntropyLoss()

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
            **kwargs
        ):

            # Text-only packed training: pass just the essentials + **kwargs (carries the packing
            # metadata cu_seq_lens_q/k + max_length_q/k through to the attention). The multimodal
            # inputs are all None here, and per_layer_inputs must NOT be passed — the model computes
            # it internally and re-passes it to the language model ("got multiple values for keyword
            # argument 'per_layer_inputs'" otherwise).
            outputs = self.model(
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

            hidden_states = outputs.last_hidden_state
            # overwrite to disable the logits materializatioon

            loss = None
            if labels is not None:
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
    ):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # Pin this process to its GPU BEFORE init'ing NCCL / the device mesh, otherwise every
    # rank lands on cuda:0 (NCCL hang / OOM).
    torch.cuda.set_device(rank)
    ddp_setup()  # init_process_group(nccl) — required before init_device_mesh / fully_shard
    mesh_device = init_device_mesh("cuda", (world_size, ), mesh_dim_names=("shard", ))
    torch.cuda.memory._record_memory_history()

    # Resolve dense vs MoE classes from the config + apply the matching Liger patches.
    config, is_moe, classes = resolve_arch(model_id)
    CustomCls = make_custom_cls(classes["base"])
    model = CustomCls.from_pretrained(
        model_id,
        config=config,
        dtype=torch.bfloat16, # native bf16 training
        attn_implementation = "kernels-community/flash-attn3"
    )
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {total_params/(1024*1024):.0f}M")

    for param in model.parameters():
        param.requires_grad = False
    apply_linear_lora(model, r=r, alpha=alpha)

    total_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info(f"Total trainable parameters: {total_trainable_params/(1024*1024):.2f}M")

    fsdp_kwargs = {}
    fsdp_kwargs["mp_policy"] = fsdp.MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    fsdp_kwargs["offload_policy"] = fsdp.CPUOffloadPolicy()
    shard_modules = (
        classes["decoder"],   # Qwen3_5DecoderLayer / Qwen3_5MoeDecoderLayer
        classes["vision"],    # Qwen3_5VisionBlock  / Qwen3_5MoeVisionBlock
    )

    for module in model.modules():
        if isinstance(module, shard_modules):
            fsdp.fully_shard(module, **fsdp_kwargs, mesh=mesh_device, reshard_after_forward=True) # shard on submodules
    fsdp.fully_shard(model, **fsdp_kwargs, mesh=mesh_device) # full shard on root module

    # Get the local shard
    model_sd = model.state_dict()
    local_shard = sum(v.to_local().numel() if hasattr(v, "to_local") else v.numel() for _ , v in model_sd.items())
    logger.info(f"[Rank{rank}]: Total local/shard param: {local_shard/(1024*1024):.2f}M")

    # checkpointing module
    checkpointing_modules = [
        classes["decoder"],   # Qwen3_5DecoderLayer / Qwen3_5MoeDecoderLayer
    ]
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn = non_reentrant_wrapper,
        check_fn=lambda x: isinstance(x, tuple(checkpointing_modules))
    )

    def _patched_chunk_gated_delta_rule(q, k, v, *args, **kwargs):
        # TileLang kernel requires v to be contiguous (stride[-1] == 1)
        v = v.contiguous()
        return chunk_gated_delta_rule(q, k, v, *args, **kwargs)

    # use qwen flash qla
    for module in model.modules():
        if isinstance(module, classes["gdn"]):
            module.chunk_gated_delta_rule = _patched_chunk_gated_delta_rule

    # max_steps > 0 caps the run regardless of epochs (0 = run all max_epochs). For overfitting a
    # small dataset, set max_epochs high and max_steps 0.
    dataset = Dataset(limit=limit_samples)
    sampler = DistributedSampler(
        dataset,
        num_replicas=mesh_device.size(),
        rank=mesh_device.get_rank(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
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

            output = model(**batch, use_cache=False) # forward pass and calculate losses
            # Back-prop the token-SUM loss (mean × #label tokens). Accumulating sums, then
            # normalizing by the GLOBAL token count at the step below, yields the exact
            # token-weighted mean over the whole effective batch — not a naive mean-of-means
            # that mis-weights variable-length bins / ranks (HF grad-accum loss fix).
            n_tok = (batch["labels"][:, 1:] != -100).sum()  # == the loss_fn's shifted denominator
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
                logger.info(f"Epoch: {i}, mb: {idx}, step: {opt_step}, loss: {loss}, tokens/s: {tps:.2f}")
                metrics = {"loss": loss, "lr": optimizer.param_groups[0]['lr'], "tps": tps, "epoch": i}
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

            if i == 0 and idx == 1:
                torch.cuda.memory._dump_snapshot(f"memory_profile_{rank}.pickle")

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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Packed bins concatenated into ONE sequence per optimizer microbatch "
             "(the collator varlen-packs them). 1 = one bin per step."
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Accumulate gradients over this many microbatches before an optimizer "
             "step. Effective batch = batch_size × grad_accum × world_size. 1 = off.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Stop after this many optimizer steps (0 = full epoch). Use a small value to "
             "produce a LoRA checkpoint quickly for merge/inference validation.",
    )
    parser.add_argument(
        "--checkpointing_step",
        type=int,
        default=100,
        help="Save the LoRA adapters every N steps.",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=0,
        help="Cap the dataset to the first N packed bins (0 = all). Use a small N to deliberately "
             "overfit a tiny subset as an end-to-end sanity check.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs over the dataset (set high to overfit a small dataset).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    parser.add_argument("--mlflow", action="store_true", help="Log metrics to MLflow.")
    parser.add_argument("--wandb_project", default="qwen3.5-autotrain", help="wandb project name.")
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
    )
