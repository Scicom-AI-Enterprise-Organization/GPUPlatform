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
from contextlib import nullcontext
from attention import dynamic_attention, block_diagonal_concat

logging.basicConfig(
    level=logging.INFO, 
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


# dynamic_attention + block_diagonal_concat live in attention.py (torch-only, unit-tested).
AttentionInterface.register("dynamic_attention", dynamic_attention)


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

class CustomGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        self.loss_fn = LigerFusedLinearCrossEntropyLoss()
    
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
            **kwargs
        ):
        
        # Text-only packed training: pass just the essentials + **kwargs (carries the packing
        # metadata cu_seq_lens_q/k + max_length_q/k through to dynamic_attention). The multimodal
        # inputs are all None here, and per_layer_inputs must NOT be passed — Gemma4Model computes
        # it internally and re-passes it to the language model ("got multiple values for keyword
        # argument 'per_layer_inputs'" otherwise).
        outputs = self.model(
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
    ):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # Pin this process to its GPU BEFORE init'ing NCCL / the device mesh, otherwise every
    # rank lands on cuda:0 (NCCL hang / OOM).
    torch.cuda.set_device(rank)
    ddp_setup()  # init_process_group(nccl) — required before init_device_mesh / fully_shard
    mesh_device = init_device_mesh("cuda", (world_size, ), mesh_dim_names=("shard", ))
    
    config = AutoConfig.from_pretrained("google/gemma-4-31B-it")
    # modify the confg
    model = CustomGemma4ForConditionalGeneration.from_pretrained(
        "google/gemma-4-31B-it", 
        config=config, 
        dtype=torch.bfloat16, # native bf16 training
        attn_implementation = "dynamic_attention"
    )
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {total_params/(1024*1024):.0f}M")
    
    # Custom LinearLoRA (NOT PEFT): gemma-4's q/k/v/o are `Gemma4ClippableLinear`, which PEFT
    # refuses ("Target module ... not supported; only torch.nn.Linear ..."), and wrapping them as
    # plain Linear would drop the clipping. LinearLoRA wraps the real module and calls its forward,
    # preserving clipping. With the lora_b=0 init fix it no longer corrupts the model at start.
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
        modeling_gemma4.Gemma4TextDecoderLayer
    )

    for module in model.modules():
        if isinstance(module, shard_modules):
            fsdp.fully_shard(module, **fsdp_kwargs, mesh=mesh_device)
    fsdp.fully_shard(model, **fsdp_kwargs, mesh=mesh_device) # full shard on root module

    # Get the local shard
    model_sd = model.state_dict()
    local_shard = sum(v.to_local().numel() if hasattr(v, "to_local") else v.numel() for _ , v in model_sd.items())
    logger.info(f"[Rank{rank}]: Total local/shard param: {local_shard/(1024*1024):.2f}M")

    # checkpointing module 
    checkpointing_modules = [
        modeling_gemma4.Gemma4TextDecoderLayer
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
                "model": "google/gemma-4-31B-it", "r": r, "alpha": alpha,
                "batch_size": batch_size, "lr": lr, "max_epochs": max_epochs,
                "max_steps": max_steps,
                "limit_samples": limit_samples, "world_size": world_size,
                "num_bins": len(dataset), "trainable_params_M": round(total_trainable_params/1e6, 2),
            },
        )
        logger.info(f"wandb run: {wandb_run.url}")

    # seen_tokens = torch.tensor(0, dtype=torch.long, device=f'cuda:{rank}')
    for i in range(max_epochs):
        sampler.set_epoch(i)  # reshuffle each epoch
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            global_step = i * len(dataloader) + idx
            if rank == 0:
                start_time = time.time()

            # Move the packed batch onto this rank's GPU (collator builds CPU tensors).
            # Python ints (max_length_q/k) are left as-is.
            batch = {
                k: (v.to(f'cuda:{rank}', non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }

            output = model(**batch, use_cache=False) # forward pass and calculate losses
            output["loss"].backward() # calculate gradient 
            
            optimizer.step()
            optimizer.zero_grad()

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
                logger.info(f"Epoch: {i}, mb: {idx}, step: {global_step}, loss: {loss}, tokens/s: {tps:.2f}")
                metrics = {"loss": loss, "lr": optimizer.param_groups[0]['lr'], "tps": tps, "epoch": i}
                if wandb_run is not None:
                    try:
                        wandb_run.log(metrics, step=global_step)
                    except Exception as e:
                        logger.warning(f"wandb.log failed (continuing): {e}")

            reached_max = max_steps > 0 and (global_step + 1) >= max_steps
            if idx == len(dataloader)-1 or (idx+1) % checkpointing_step == 0 or reached_max:
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
                    os.makedirs("checkpointing", exist_ok=True)
                    torch.save(lora_state_dict, "checkpointing/lora.pt")
                    with open("checkpointing/lora_meta.json", "w") as f:
                        json.dump({
                            "model_id": "google/gemma-4-31B-it",
                            "r": r, "alpha": alpha, "scaling": alpha / r,
                            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                            "wrapped_attr": "linear",
                        }, f, indent=2)
                    logger.info(f"Saved LoRA ({len(lora_state_dict)} tensors) to checkpointing/lora.pt")

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
        "--r", 
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
        help="Batch size"
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
    parser.add_argument("--wandb_project", default="gemma4-autotrain", help="wandb project name.")
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
    )