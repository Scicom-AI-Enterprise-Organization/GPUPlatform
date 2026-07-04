"""MULTI-DOC (varlen) tiny-Qwen3.6 CP correctness test — the gap test_cp_model.py (single-doc) left
open. Packs several documents into one sequence such that:
  * one doc SPANS the shard boundary  -> exercises `first_cont` GDN recurrent-state continuation,
  * a doc boundary falls INSIDE a rank's chunk -> exercises per-doc state reset + multi-segment local
    cu_seqlens + the doc-masked full-attn ring (no cross-doc attention).
Compares CP last_hidden_state (per rank chunk) to the non-CP VARLEN reference (flash-attn, which masks
by doc via cu_seqlens). Random weights, no checkpoint.

    torchrun --nproc_per_node=2 test_cp_model_varlen.py
"""
import os, torch, torch.distributed as dist
from transformers import AutoConfig, AttentionInterface
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
from flash_qla import chunk_gated_delta_rule as _flash_gdr
import context_parallel as cp

rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr); dev = torch.device(f"cuda:{lr}")
dist.init_process_group("nccl")
cp.setup_cp(W, W, rank, dev)
AttentionInterface.register("cp_full_attention", cp.cp_full_attention)
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-27B").get_text_config()
cfg.num_hidden_layers = 4
cfg.layer_types = ["linear_attention", "linear_attention", "full_attention", "linear_attention"]
cfg.hidden_size = cfg.num_attention_heads * 64
cfg.intermediate_size = 512
cfg.vocab_size = 256
cfg._attn_implementation = "eager"                        # eager builds + propagates cleanly on tm
if hasattr(cfg, "num_experts"):
    cfg.num_experts = 4

torch.manual_seed(0)
model = M.Qwen3_5TextModel(cfg).to(dev).to(torch.bfloat16).eval()
for p in model.parameters():
    p.requires_grad_(False); dist.broadcast(p, src=0)
# Use the REAL flash_qla TileLang kernel (as the trainer does) — its varlen path supports per-doc
# [n_docs, Hv, K, V] initial_state + cu_seqlens, which the CP relay needs. (The model's torch fallback
# does not.) Both the non-CP reference AND the CP run then use the same kernel.
def _patched_gdr(q, k, v, *a, **kw):
    return _flash_gdr(q, k, v.contiguous(), *a, **kw)
for m in model.modules():
    if isinstance(m, M.Qwen3_5GatedDeltaNet):
        m.chunk_gated_delta_rule = _patched_gdr

# ---- multi-doc pack: with W=2, chunk size c = S/2. Boundaries chosen so doc 0 crosses c and doc 2
#      starts inside rank 1's chunk. S multiple of W. ----
c = 48                                    # per-rank chunk (W=2 -> S=96)
S = c * W
doc_lens = [70, 26] if W == 2 else [S]    # doc0 (0..69) spans c=48; doc1 (70..95) inside rank1
assert sum(doc_lens) == S, (doc_lens, S)
cu_full = torch.tensor([0] + list(torch.tensor(doc_lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
# per-doc position ids (varlen packing resets positions each doc)
pos_full = torch.cat([torch.arange(L, device=dev) for L in doc_lens])
gcpu = torch.Generator(device="cpu").manual_seed(1)
ids = torch.randint(0, cfg.vocab_size, (S,), generator=gcpu).to(dev)
if rank == 0:
    print(f"docs={doc_lens} cu={cu_full.tolist()} S={S} c={c} full_attn_layers=1", flush=True)


def run(attn_impl, input_ids, position_ids, cu, attn_mask=None):
    # model.config is shared by every submodule (verified) -> one assignment switches all layers.
    model.config._attn_implementation = attn_impl
    kw = {} if attn_mask is None else {"attention_mask": attn_mask}
    with torch.no_grad():
        out = model(input_ids=input_ids.unsqueeze(0), position_ids=position_ids.unsqueeze(0),
                    cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=int((cu[1:] - cu[:-1]).max()),
                    max_length_k=int((cu[1:] - cu[:-1]).max()), use_cache=False, **kw)
    return out.last_hidden_state[0].float()


# non-CP varlen reference: eager full-attn with an EXPLICIT block-diagonal causal mask (attend only
# within the same doc, and only to earlier positions) — the doc-masking the packed forward must do.
# (GDN layers ignore attention_mask; they reset per-doc via cu_seqlens.)
docid_full = torch.cat([torch.full((L,), i, device=dev) for i, L in enumerate(doc_lens)])
idx = torch.arange(S, device=dev)
keep = (docid_full[:, None] == docid_full[None, :]) & (idx[None, :] <= idx[:, None])   # (S,S) bool
mask4d = torch.where(keep, 0.0, float("-inf")).to(torch.bfloat16)[None, None]           # (1,1,S,S)
h_full = run("eager", ids, pos_full, cu_full, attn_mask=mask4d)

# CP: shard the SAME multi-doc pack; GDN relay + full-attn ring
cp.install_gdn_cp(model, M.Qwen3_5GatedDeltaNet)
batch = {"input_ids": ids.unsqueeze(0), "position_ids": pos_full.unsqueeze(0),
         "labels": ids.unsqueeze(0), "cu_seq_lens_q": cu_full}
b = cp.shard_batch(batch, W, rank)
print(f"[r{rank}] first_cont={cp._CP['first_cont']} local_cu={b['cu_seq_lens_q'].tolist()} "
      f"docs_local={sorted(set(cp._CP['doc'].tolist()))}", flush=True)
h_loc = run("cp_full_attention", b["input_ids"][0], b["position_ids"][0], b["cu_seq_lens_q"])

ref = h_full[rank * c:(rank + 1) * c]
rel = (h_loc - ref).norm().item() / (ref.norm().item() or 1.0)
t = torch.tensor([rel], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX)
if rank == 0:
    ok = t.item() < 2e-2
    print(f"[{'PASS' if ok else 'FAIL'}] MULTI-DOC Qwen CP vs non-CP varlen hidden rel={t.item():.3e}", flush=True)
dist.destroy_process_group()
