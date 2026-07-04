"""FAST reproducer for the Qwen CP *backward* deadlock under FSDP2 (no 27B / no HF download).

Builds a TINY but FAITHFUL hybrid Qwen3.6 (8 layers = [GDN,GDN,GDN,FULL]*2, REAL GDN head dims so the
cached TileLang kernels are reused, tiny vocab/intermediate/seq), wires it EXACTLY like qwen3_5.py under
CP (setup_cp + cp_full_attention + contiguous-v patch + install_gdn_cp), FSDP2 fully_shard per decoder
layer + CPUOffload, activation-checkpointing OFF (the CP path). Then one forward + CP loss + backward.

    torchrun --nproc_per_node=2 cp_fsdp_repro.py

Prints per-rank BEGIN/FWD_DONE/BWD_DONE. If it hangs at 100% GPU with no BWD_DONE it has reproduced the
27B backward deadlock — iterate the fix in context_parallel.py and re-run (seconds/cycle, not minutes).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group, fsdp
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoConfig, AttentionInterface
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
import context_parallel as cp

AttentionInterface.register("cp_full_attention", cp.cp_full_attention)

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); lrank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lrank); dev = torch.device(f"cuda:{lrank}")
init_process_group(backend="cpu:gloo,cuda:nccl")
CP_SIZE = world
cp_group, dp_size, dp_rank, cp_rank = cp.setup_cp(world, CP_SIZE, rank, dev)
mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("shard",))


def log(msg):
    print(f"[repro r{rank}] {msg}", flush=True)


# ---- tiny faithful config: keep GDN head dims (kernel-shape identical -> reuse cached kernels) ----
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-27B").get_text_config()
_NL = int(os.environ.get("REPRO_LAYERS", "64"))
cfg.num_hidden_layers = _NL
cfg.layer_types = cfg.layer_types[:_NL]        # full depth by default = same GDN/FULL interleave density
cfg.hidden_size = 2048                          # shrink width (GDN kernel shapes are head-dim based, unchanged)
cfg.intermediate_size = 2048                   # shrink MLP (irrelevant to the CP deadlock)
cfg.vocab_size = 1024                           # shrink embedding/head
cfg.tie_word_embeddings = False
cfg._attn_implementation = "cp_full_attention"  # full-attn layers ring over the CP group
log(f"layer_types={cfg.layer_types}")

torch.manual_seed(0)
model = M.Qwen3_5TextModel(cfg).to(dev).to(torch.bfloat16)
lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).to(dev).to(torch.bfloat16)
for p in model.parameters():
    dist.broadcast(p.detach(), src=0)
dist.broadcast(lm_head.weight.detach(), src=0)

# ---- freeze base + LoRA the attention q/k/v/o_proj (EXACTLY like qwen3_5.py: GDN layers stay fully
# frozen, only full-attn LoRA adapters train). This reshapes the backward graph — the scale-free
# suspect for the 27B backward deadlock — so the repro must match it. ----
class _LoRA(nn.Module):
    def __init__(self, lin, r=8, alpha=16):
        super().__init__(); self.lin = lin; self.s = alpha / r
        self.a = nn.Linear(lin.in_features, r, bias=False, dtype=torch.bfloat16)
        self.b = nn.Linear(r, lin.out_features, bias=False, dtype=torch.bfloat16)
        nn.init.kaiming_uniform_(self.a.weight); nn.init.zeros_(self.b.weight)
    def forward(self, x):
        return self.lin(x) + self.s * self.b(self.a(x))

for p in model.parameters():
    p.requires_grad_(False)
lm_head.weight.requires_grad_(False)                      # frozen, like 27B (only LoRA trains)
_LORA_TGT = {"q_proj", "k_proj", "v_proj", "o_proj"}
for mod in list(model.modules()):
    for cn, child in list(mod.named_children()):
        if isinstance(child, nn.Linear) and cn in _LORA_TGT:
            setattr(mod, cn, _LoRA(child).to(dev).to(torch.bfloat16))
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"trainable(LoRA)={n_train/1e6:.2f}M")

# ---- FSDP2 fully_shard per decoder layer + root, CPUOffload, AC OFF (the CP path) ----
fsdp_kwargs = {"mp_policy": fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
               "offload_policy": fsdp.CPUOffloadPolicy()}
_RESHARD = os.environ.get("CP_NORESHARD") != "1"      # CP_NORESHARD=1 -> keep params gathered (no re-AG)
_fsdp_layers = []
for m in model.modules():
    if isinstance(m, M.Qwen3_5DecoderLayer):
        fsdp.fully_shard(m, **fsdp_kwargs, mesh=mesh, reshard_after_forward=_RESHARD)
        _fsdp_layers.append(m)
fsdp.fully_shard(model, **fsdp_kwargs, mesh=mesh, reshard_after_forward=_RESHARD)
log(f"reshard_after_forward={_RESHARD}")

# FIX (toggle CP_NOPREFETCH=1): FSDP2 turns on implicit all-gather prefetch from iteration 1 (it
# records module order on iter 0). Under CP that prefetched AG gets enqueued ahead of the GDN relay
# P2P on one rank but after it on the other -> cross-communicator circular wait -> backward/step-1
# deadlock. Emptying the prefetch lists disables the reordering so the AG stays in-order with the P2P.
if os.environ.get("CP_NOPREFETCH") == "1":
    for m in _fsdp_layers + [model]:
        m.set_modules_to_forward_prefetch([])
        m.set_modules_to_backward_prefetch([])
    log("FSDP prefetch DISABLED (CP_NOPREFETCH=1)")

# ---- inject the REAL flash_qla TileLang kernel + contiguous-v patch, then GDN CP relay (as qwen3_5.py) ----
from flash_qla import chunk_gated_delta_rule as _flash_gdr
def _patched_gdr(q, k, v, *a, **kw):
    return _flash_gdr(q, k, v.contiguous(), *a, **kw)
for m in model.modules():
    if isinstance(m, M.Qwen3_5GatedDeltaNet):
        m.chunk_gated_delta_rule = _patched_gdr
cp.install_gdn_cp(model, M.Qwen3_5GatedDeltaNet)

opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4)

# ---- synthetic single-doc packed batch (S multiple of cp_size), like the trainer feeds shard_batch ----
S = int(os.environ.get("REPRO_SEQ", "512"))
ids = torch.randint(1, cfg.vocab_size, (S,), device=dev)
batch = {"input_ids": ids.unsqueeze(0),
         "position_ids": torch.arange(S, device=dev).unsqueeze(0),
         "labels": ids.clone().unsqueeze(0),
         "cu_seq_lens_q": torch.tensor([0, S], dtype=torch.int32, device=dev)}

for step in range(2):
    log(f"BEGIN step {step}")
    b = cp.shard_batch(batch, CP_SIZE, cp_rank)
    labels = b.pop("labels"); b.pop("attention_mask"); b.pop("mm_token_type_ids")
    out = model(**b, use_cache=False)
    hs = out.last_hidden_state
    logits = lm_head(hs).float()
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), labels.reshape(-1), ignore_index=-100)
    log(f"FWD_DONE step {step} loss={loss.item():.4f}")
    loss.backward()
    log(f"BWD_DONE step {step}")
    opt.step(); opt.zero_grad()
    log(f"STEP_DONE step {step}")

log("REPRO_OK — CP forward+backward+step completed under FSDP2")
dist.barrier()
destroy_process_group()
