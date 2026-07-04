"""Full tiny-Qwen3.6 CP integration test: the REAL text backbone (mix of GatedDeltaNet + full-attn
layers) under context parallelism (cp_size=2, contiguous shard, GDN state-relay + full-attn ring)
vs the non-CP full-sequence forward. Compares last_hidden_state per rank's chunk. Random weights;
no checkpoint. This exercises multi-layer comm interleaving + wiring end to end.

    torchrun --nproc_per_node=2 test_cp_model.py
"""
import os, sys, torch, torch.distributed as dist

from transformers import AutoConfig, AttentionInterface
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
import context_parallel as cp

rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr); dev = torch.device(f"cuda:{lr}")
dist.init_process_group("nccl")
cp.setup_cp(W, W, rank, dev)                    # cp_size = W (one CP group)
AttentionInterface.register("cp_full_attention", cp.cp_full_attention)

# tiny config: a few layers mixing linear + full attention, small hidden, real head dims.
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-27B").get_text_config()
cfg.num_hidden_layers = 4
cfg.layer_types = ["linear_attention", "linear_attention", "full_attention", "linear_attention"]
cfg.hidden_size = cfg.num_attention_heads * 64  # keep head count consistent, shrink hidden
cfg.intermediate_size = 512
cfg.vocab_size = 256
if hasattr(cfg, "num_experts"):
    cfg.num_experts = 4                          # (dense here anyway; harmless)
if rank == 0:
    print(f"layers={cfg.layer_types} hidden={cfg.hidden_size} head_dim={cfg.head_dim}", flush=True)

torch.manual_seed(0)
model = M.Qwen3_5TextModel(cfg).to(dev).to(torch.bfloat16).eval()
for p in model.parameters():
    p.requires_grad_(False); dist.broadcast(p, src=0)

# patch chunk_gated_delta_rule to .contiguous() v (TileLang needs it) — as the trainer does
_orig = {}
for m in model.modules():
    if isinstance(m, M.Qwen3_5GatedDeltaNet):
        real = m.chunk_gated_delta_rule
        m.chunk_gated_delta_rule = (lambda real: (lambda q, k, v, *a, **kw: real(q, k, v.contiguous(), *a, **kw)))(real)

S = 2 * W * 24; cu_full = torch.tensor([0, S], dtype=torch.int32, device=dev)
pos_full = torch.arange(S, device=dev)
g = torch.Generator(device="cpu").manual_seed(1)
ids = torch.randint(0, cfg.vocab_size, (S,), generator=g).to(dev)

def run(attn_impl, input_ids, position_ids, cu):
    for mm in model.modules():
        if hasattr(mm, "config"):
            mm.config._attn_implementation = attn_impl
    model.config._attn_implementation = attn_impl
    with torch.no_grad():
        out = model(input_ids=input_ids.unsqueeze(0), position_ids=position_ids.unsqueeze(0),
                    cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=int(cu[-1]), max_length_k=int(cu[-1]),
                    use_cache=False)
    return out.last_hidden_state[0].float()

# non-CP full reference (redundant on each rank); eager full-attn over the whole sequence
h_full = run("eager", ids, pos_full, cu_full)

# CP: this rank's contiguous chunk (+ GDN relay installed + full-attn ring backend)
cp.install_gdn_cp(model, M.Qwen3_5GatedDeltaNet)
c = S // W
batch = {"input_ids": ids.unsqueeze(0), "position_ids": pos_full.unsqueeze(0),
         "labels": ids.unsqueeze(0), "cu_seq_lens_q": cu_full}
b = cp.shard_batch(batch, W, rank)
h_loc = run("cp_full_attention", b["input_ids"][0], b["position_ids"][0], b["cu_seq_lens_q"])

ref = h_full[rank * c:(rank + 1) * c]
rel = (h_loc - ref).norm().item() / (ref.norm().item() or 1.0)
t = torch.tensor([rel], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX)
if rank == 0:
    print(f"[{'PASS' if t.item() < 2e-2 else 'FAIL'}] full tiny-Qwen CP vs non-CP hidden rel={t.item():.3e}", flush=True)
dist.destroy_process_group()
