"""NVIDIA Nemotron-H (Nemotron-3-Nano-30B-A3B) LoRA finetune — FSDP2, bf16.

Nemotron-H is a HYBRID model: each `NemotronHBlock` is a Mamba2 SSM, an attention layer,
a dense MLP, or an MoE block (per `config.layers_block_type`). Unlike the attention-only
trainers (gemma/qwen/minimax/mistral) this one does NOT multipack: the HF NemotronH forward
has no cu_seqlens/seq_idx plumbing, so concatenating documents would LEAK the Mamba SSM
state across document boundaries (there is no per-doc reset). So the dataset is packed
ONE-DOC-PER-BIN (llm_pack, arch=nemotron) and this trainer PADS a batch of docs instead of
concatenating — each sequence is a single document, which is correct for both the attention
(standard causal mask) and Mamba (state runs over one doc only) layers.

LoRA wraps the attention `q/k/v/o_proj` and the Mamba `in_proj/out_proj` (nn.Linear). The MoE
experts are 3D `nn.Parameter` (`NemotronHExperts.up_proj/down_proj`), which the nn.Linear LoRA
does not touch → they stay frozen (like the other MoE trainers' default). `--train_embeddings`
additionally full-trains the (untied) token embeddings + LM head.

Mamba kernels: we set `use_mamba_kernels=False` so the Mamba layers run transformers' torch SSD
fallback (autograd-friendly, no mamba-ssm/causal-conv1d build). Attention runs `sdpa` (env
NEMOTRON_ATTN). Reads ./packed_data, writes ./checkpointing/lora.pt (+ lora_meta.json).
"""
from transformers import AutoConfig, AutoTokenizer, NemotronHForCausalLM
from transformers.models.nemotron_h import modeling_nemotron_h
import numpy as np
import torch
import torch.nn as nn
import logging
import os
import json
import time
import argparse
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh
from tqdm import tqdm
from chinidataset import StreamingDataset
import _trainer_common as tc

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger()

# Base model — the gateway autotrain runner overrides via MODEL_ID.
MODEL_ID = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
# Attention backend for the (sparse) attention layers: hub FA3 (kernels-community — fetched by
# the `kernels` package at model load, no build; head_dim 128 GQA supported) is the default —
# sdpa with a padding mask materializes an O(S^2) causal mask at long context. NEMOTRON_ATTN=sdpa
# is the no-network fallback.
ATTN_IMPL = os.environ.get("NEMOTRON_ATTN", "kernels-community/flash-attn3")
# Fused Mamba2 Triton kernels (mamba-ssm + causal-conv1d, BOTH lazy-fetched from the kernels
# hub — no source build): the torch SSD fallback materializes fp32 chunk intermediates in
# autograd across the 23 mamba layers, THE long-context memory/speed hog. 0 = torch fallback.
MAMBA_KERNELS = os.environ.get("NEMOTRON_MAMBA_KERNELS", "1") == "1"


def _causal_conv1d_available() -> bool:
    """The hub mamba-ssm kernel's forward (both the mem-eff split path AND the non-split
    `causal_conv1d_fn` path) hard-asserts `causal_conv1d_cuda is not None`. That compiled
    extension ships with a pip/built `causal-conv1d` — it is NOT bundled with the mamba-ssm
    hub kernel and is NOT in the nemotron venv deps, so a fresh venv lacks it. transformers'
    `is_fast_path_available` only checks mamba-ssm (True even when causal_conv1d is missing),
    so it can't guard this. Probe the extension directly so we can degrade to the torch SSD
    path (use_mamba_kernels=False) instead of crashing at the first Mamba forward."""
    try:
        import causal_conv1d_cuda  # noqa: F401
        return True
    except Exception:
        return False


def ddp_setup():
    # cpu:gloo + cuda:nccl — gloo services the CPU collective in the checkpoint's full_tensor()
    # all-gather under CPUOffloadPolicy (NCCL can't); nccl stays the GPU fast path.
    init_process_group(backend="cpu:gloo,cuda:nccl")


def _patch_mamba_kernel_fp32():
    """⚠ The hub mamba-ssm kernel's BACKWARD produces NaN grads in bf16 — specifically the
    ddt/dA path (dt_bias, A_log, and dx→conv→in_proj grads go NaN; D/norm/out_proj grads are
    fine), verified at the real 30B geometry in autotrain/nemotron/test_mamba_kernels.py.
    The SAME kernel in fp32 is exact (fwd + grad cos 1.0 vs the torch reference). So wrap
    mamba_chunk_scan_combined to upcast (x, dt, A, B, C) → fp32 and downcast the output —
    autograd then runs the verified-clean fp32 backward, and the casts route grads back to
    the bf16 params. Still far cheaper than the torch SSD fallback (fused Triton scan vs
    autograd-materialized chunk einsums)."""
    orig = modeling_nemotron_h.mamba_chunk_scan_combined
    if orig is None or getattr(orig, "_sgpu_fp32_wrapped", False):
        return orig is not None

    def fp32_wrapped(x, dt, A, B, C, **kw):
        outs = orig(x.float(), dt.float(), A.float(), B.float(), C.float(), **kw)
        if isinstance(outs, tuple):
            return tuple(o.to(x.dtype) if torch.is_tensor(o) and o.is_floating_point() else o
                         for o in outs)
        return outs.to(x.dtype)

    fp32_wrapped._sgpu_fp32_wrapped = True
    modeling_nemotron_h.mamba_chunk_scan_combined = fp32_wrapped
    return True


# Attention q/k/v/o + Mamba2 in_proj/out_proj (attention layers are SPARSE in the hybrid
# pattern, so q/k/v/o alone barely adapts the model — include the Mamba projections).
DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"]


class LinearLoRA(nn.Module):
    """y = W x + (alpha/r)·B(A x). Wraps the real Linear (calls its forward) so any
    module-specific behaviour is preserved. B=0 init ⇒ a no-op adapter at start."""

    def __init__(self, linear: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r
        self.lora_a = nn.Linear(linear.in_features, r, bias=False, dtype=torch.bfloat16)
        self.lora_b = nn.Linear(r, linear.out_features, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_a.weight)
            nn.init.zeros_(self.lora_b.weight)  # CRITICAL: adapter delta = 0 at init

    # The Mamba2 fast-path dispatch reads `self.in_proj.weight.device.type` (and the mem-eff
    # kernel reads `.weight`/`.bias` directly) — expose the wrapped Linear's tensors so a
    # LoRA-wrapped projection still looks like a Linear to that code.
    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        return self.linear(x) + self.scaling * self.lora_b(self.lora_a(x))


def apply_linear_lora(base_model: nn.Module, r: int, alpha: int, target_modules=None):
    target_modules = list(target_modules or DEFAULT_LORA_TARGETS)
    wrapped = {t: 0 for t in target_modules}
    for name, module in list(base_model.named_modules()):
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear) and child_name in target_modules:
                setattr(module, child_name, LinearLoRA(child_module, r, alpha))
                wrapped[child_name] = wrapped.get(child_name, 0) + 1
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        logger.info(f"LoRA target_modules={target_modules}; wrapped per module: {wrapped}")
        zero = [t for t, n in wrapped.items() if n == 0]
        if zero:
            logger.warning(f"LoRA targets matched NO nn.Linear modules and will NOT be trained: "
                           f"{zero} — this architecture may fuse/rename them.")
    return wrapped


class CustomNemotronHForCausalLM(NemotronHForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        self.loss_fn = LigerFusedLinearCrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                labels=None, use_cache=None, **kwargs):
        # Run the backbone to hidden states, then the fused linear+CE against lm_head.weight
        # (never materializes the full [B,S,V] logits — the long-vocab memory saver).
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if labels is None:
            raise NotImplementedError("nemotron_h trainer computes loss only (labels required)")
        # standard next-token shift; Liger FLCE ignores -100 (pad + non-trained) internally.
        shifted_hidden = hidden_states[:, :-1, :].contiguous().reshape(-1, hidden_states.shape[-1])
        shifted_labels = labels[:, 1:].contiguous().reshape(-1)
        loss = self.loss_fn(self.lm_head.weight, shifted_hidden, shifted_labels)
        return {"loss": loss}


class Dataset(TorchDataset):
    def __init__(self, limit: int = 0):
        self.dataset = StreamingDataset(local="./packed_data")
        self._len = min(limit, len(self.dataset)) if limit and limit > 0 else len(self.dataset)

    def __getitem__(self, idx):
        # Each bin is ONE document (llm_pack one_doc_per_bin for nemotron). We only need its
        # ids + labels; position_ids/attention_mask are rebuilt by the padding collator.
        return {
            "input_ids": self.dataset[idx]["input_ids"],
            "labels": self.dataset[idx]["labels"],
        }

    def __len__(self):
        return self._len


def make_collator(pad_id: int):
    """Right-pad a batch of single-doc bins to the longest in the batch. attention_mask marks
    real tokens (create_causal_mask + the mamba mask zero the pad); labels pad with -100;
    position_ids reset 0..L-1 per doc. Right-padding is correct for Mamba (trailing pad never
    corrupts an earlier token's state)."""
    def collator(batch):
        batch = [b for b in batch if b is not None]
        ids = [np.asarray(b["input_ids"], dtype=np.int64) for b in batch]
        labs = [np.asarray(b["labels"], dtype=np.int64) for b in batch]
        B = len(ids)
        max_len = max(len(x) for x in ids)
        input_ids = np.full((B, max_len), pad_id, dtype=np.int64)
        labels = np.full((B, max_len), -100, dtype=np.int64)
        attn = np.zeros((B, max_len), dtype=np.int64)
        pos = np.zeros((B, max_len), dtype=np.int64)
        for i, (x, y) in enumerate(zip(ids, labs)):
            L = len(x)
            input_ids[i, :L] = x
            labels[i, :L] = y
            attn[i, :L] = 1
            pos[i, :L] = np.arange(L)
        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attn),
            "position_ids": torch.from_numpy(pos),
            "labels": torch.from_numpy(labels),
        }
    return collator


def main(
        r: int = 32,
        alpha: int = 64,
        batch_size: int = 1,
        max_steps: int = 0,
        checkpointing_step: int = 100,
        limit_samples: int = 0,
        max_epochs: int = 1,
        lr: float = 5e-5,
        use_wandb: bool = False,
        wandb_project: str = "nemotron-h-autotrain",
        grad_accum: int = 1,
        cpu_offload: bool = False,
        target_modules=None,
        train_embeddings: bool = False,
    ):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    ddp_setup()
    mesh_device = init_device_mesh("cuda", (world_size,), mesh_dim_names=("shard",))

    config = AutoConfig.from_pretrained(MODEL_ID)
    # Fused Mamba2 Triton kernels (hub-fetched) vs the torch SSD fallback. The fused path
    # needs the compiled `causal_conv1d_cuda` extension (separate `causal-conv1d` package,
    # NOT bundled with the mamba-ssm hub kernel); when it's absent the kernel forward would
    # assert-crash. Probe it and degrade to the (autograd-friendly, always-available) torch
    # SSD path instead of dying — the fallback the comment used to only claim.
    use_mamba = MAMBA_KERNELS and _causal_conv1d_available()
    if MAMBA_KERNELS and not use_mamba and rank == 0:
        logger.warning("nemotron-h: causal_conv1d_cuda is unavailable — the fused Mamba2 "
                       "kernels would assert; falling back to the torch SSD path "
                       "(use_mamba_kernels=False). Install `causal-conv1d` in the venv "
                       "to restore the ~1.5x fast path.")
    config.use_mamba_kernels = use_mamba
    if rank == 0:
        logger.info(f"nemotron-h: attn={ATTN_IMPL}, use_mamba_kernels={use_mamba}")
    model = CustomNemotronHForCausalLM.from_pretrained(
        MODEL_ID, config=config, dtype=torch.bfloat16, attn_implementation=ATTN_IMPL,
    )
    if rank == 0:
        logger.info(f"nemotron-h: mamba fast path available = "
                    f"{getattr(modeling_nemotron_h, 'is_fast_path_available', None)}")
    if use_mamba and getattr(modeling_nemotron_h, "is_fast_path_available", False):
        patched = _patch_mamba_kernel_fp32()
        if rank == 0:
            logger.info(f"nemotron-h: mamba_chunk_scan_combined fp32-upcast patch applied = {patched} "
                        f"(hub kernel's bf16 backward NaNs — see autotrain/nemotron)")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {total_params/(1024*1024):.0f}M")

    for param in model.parameters():
        param.requires_grad = False
    target_modules = list(target_modules or DEFAULT_LORA_TARGETS)
    apply_linear_lora(model, r=r, alpha=alpha, target_modules=target_modules)

    # ⚠ Force the NON-split fast path (use_mem_eff_path=False) on EVERY mamba mixer when the
    # fused kernels are on — for TWO independent reasons:
    #   (1) the mem-eff split kernel (mamba_split_conv1d_scan_combined) fuses the RAW
    #       out_proj.weight into the kernel, so a LoRA-wrapped out_proj's adapter is SILENTLY
    #       BYPASSED (trains but never fires in the forward); and
    #   (2) the hub mamba-ssm build's split kernel hard-asserts `causal_conv1d_cuda is not None`
    #       — its OWN reference is None even when the pip `causal-conv1d` IS installed — so it
    #       CANNOT run here regardless of LoRA.
    # The non-split path (causal_conv1d_fn + mamba_chunk_scan_combined + a module-call out_proj)
    # uses the pip causal-conv1d and honors LoRA. Disable the split path UNCONDITIONALLY — the
    # earlier `out_proj is LinearLoRA` guard missed a q/k/v/o-only target set (out_proj stays a
    # plain Linear) → the split kernel ran → assert-crash. in_proj is a module call either way.
    n_meff_off = 0
    if use_mamba:
        for m in model.modules():
            if isinstance(m, modeling_nemotron_h.NemotronHMamba2Mixer) and \
                    getattr(m, "use_mem_eff_path", False):
                m.use_mem_eff_path = False
                n_meff_off += 1
    if rank == 0 and n_meff_off:
        logger.info(f"[mamba] forced non-split fast path on {n_meff_off} mixers "
                    f"(hub split kernel asserts on causal_conv1d_cuda + would bypass out_proj LoRA)")

    # Optionally FULL-train the (untied) token embeddings + LM head on top of LoRA. Done BEFORE
    # FSDP sharding. See tc.unfreeze_embeddings.
    if train_embeddings:
        tc.unfreeze_embeddings(model, rank=rank, logger=logger)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_trainable_params/(1024*1024):.2f}M")

    kw = tc.fsdp_kwargs(mesh_device, param_dtype=torch.bfloat16, cpu_offload=cpu_offload)
    tc.shard_layers(model, modeling_nemotron_h.NemotronHBlock, kw)
    tc.checkpoint_layers(model, modeling_nemotron_h.NemotronHBlock)

    model_sd = model.state_dict()
    local_shard = sum(v.to_local().numel() if hasattr(v, "to_local") else v.numel()
                      for _, v in model_sd.items())
    logger.info(f"[Rank{rank}]: Total local/shard param: {local_shard/(1024*1024):.2f}M")

    # pad id (only matters as a valid vocab id — pad positions are masked out of the loss).
    pad_id = getattr(config, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(config, "eos_token_id", None) or 0
    if isinstance(pad_id, (list, tuple)):
        pad_id = pad_id[0]

    dataset = Dataset(limit=limit_samples)
    sampler = DistributedSampler(dataset, num_replicas=mesh_device.size(), rank=mesh_device.get_rank())
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=make_collator(int(pad_id)),
        sampler=sampler, prefetch_factor=4, num_workers=4,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, fused=False)
    model.train()

    wandb_run = None
    if use_wandb and rank == 0:
        import wandb
        wandb_run = wandb.init(
            project=wandb_project,
            config={"model": MODEL_ID, "r": r, "alpha": alpha, "batch_size": batch_size,
                    "lr": lr, "max_epochs": max_epochs, "max_steps": max_steps,
                    "limit_samples": limit_samples, "world_size": world_size,
                    "num_bins": len(dataset), "trainable_params_M": round(total_trainable_params/1e6, 2)},
        )
        logger.info(f"wandb run: {wandb_run.url}")

    opt_step = 0
    reached_max = False
    win_tokens = torch.zeros((), device=f'cuda:{rank}', dtype=torch.long)
    for i in range(max_epochs):
        sampler.set_epoch(i)
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if rank == 0:
                start_time = time.time()
            batch = {k: (v.to(f'cuda:{rank}', non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            if rank == 0 and idx == 0:
                S = batch["input_ids"].shape[-1]
                logger.info(f"PRE-FWD: B={batch['input_ids'].shape[0]}, S={S}, "
                            f"allocated={torch.cuda.memory_allocated(rank)/2**30:.2f}GB")
            output = model(**batch, use_cache=False)
            # Token-weighted grad-accum (mirrors gemma/qwen): back-prop the token-SUM loss, then
            # normalize by the GLOBAL non-pad token count → exact token-mean over the effective batch.
            n_tok = (batch["labels"][:, 1:] != -100).sum()
            (output["loss"] * n_tok).backward()
            win_tokens += n_tok

            do_step = ((idx + 1) % grad_accum == 0) or (idx == len(dataloader) - 1)
            if do_step:
                dist.all_reduce(win_tokens, op=dist.ReduceOp.SUM)
                scale = (world_size / win_tokens.clamp(min=1)).item()
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            p.grad.mul_(scale)
                optimizer.step()
                optimizer.zero_grad()
                win_tokens.zero_()

            token_count = torch.tensor(batch["input_ids"].numel(), device=f'cuda:{rank}')
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

            if rank == 0:
                loss = output['loss'].item()
                dt = time.time() - start_time
                tps = token_count.item() / dt
                logger.info(f"Epoch: {i}, mb: {idx}, step: {opt_step}, loss: {loss}, tokens/s: {tps:.2f}")
                if wandb_run is not None and do_step:
                    try:
                        wandb_run.log({"loss": loss, "lr": optimizer.param_groups[0]['lr'],
                                       "tps": tps, "epoch": i}, step=opt_step)
                    except Exception as e:
                        logger.warning(f"wandb.log failed (continuing): {e}")

            reached_max = max_steps > 0 and do_step and (opt_step + 1) >= max_steps
            if idx == len(dataloader) - 1 or (do_step and (idx + 1) % checkpointing_step == 0) or reached_max:
                logger.info("Checkpointing LoRA adapters..")
                # Save every trainable param — LoRA adapters AND (with --train_embeddings) the full
                # embed/lm_head weight. full_tensor() is a collective → every rank iterates the SAME params.
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
                            "objective": "sft",
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
    parser.add_argument("--r", "--lora_r", dest="r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=64, help="LoRA alpha (scaling=alpha/r)")
    tc.add_common_args(parser, lr_default=5e-5, wandb_project="nemotron-h-autotrain")
    parser.add_argument(
        "--target_modules", type=str, default=",".join(DEFAULT_LORA_TARGETS),
        help="Comma-separated nn.Linear module names to LoRA. Default: attention q/k/v/o + Mamba "
             "in_proj/out_proj. Add MLP (up_proj,down_proj) to adapt the dense-MLP layers; the MoE "
             "experts are 3D nn.Parameter and are NOT reachable by this Linear LoRA.")
    parser.add_argument(
        "--train_embeddings", action="store_true",
        help="Also FULL-train the (untied) token embeddings + LM head, not just LoRA. Helps the "
             "finetune reliably emit special tokens. Costs extra optimizer state — combine with --cpu_offload.")
    args = parser.parse_args()
    main(
        args.r, args.alpha, args.batch_size, args.max_steps, args.checkpointing_step,
        args.limit_samples, args.max_epochs, args.lr, args.wandb, args.wandb_project,
        grad_accum=args.grad_accum, cpu_offload=args.cpu_offload,
        target_modules=[t.strip() for t in (args.target_modules or "").split(",") if t.strip()],
        train_embeddings=args.train_embeddings,
    )
