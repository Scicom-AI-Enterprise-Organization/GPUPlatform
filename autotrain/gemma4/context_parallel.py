"""Context-parallel (zigzag ring) plumbing for gemma-4 packed training.

Enabled by `--cp_size N` (>1): shard ONE packed sequence across N GPUs (a "CP group") and compute
exact attention by ringing K/V, so context longer than a single GPU's VRAM can train. The ranks of
a CP group all process the SAME packed bin (different sequence chunks); data parallelism is across
CP groups. The hybrid attention dispatch mirrors gemma-4's two layer types:

  * head_dim-512 GLOBAL (full causal)  → zigzag_ring_flash_attn_varlen_func (FA4 cute fwd + recompute
    bwd) — the fast, memory-efficient fused ring.
  * head_dim-256 SLIDING window        → posaware_ring_attn_func (global-position masks) — a fused
    kernel's window is bottom-right-aligned per call, which is WRONG for zigzag's non-contiguous
    chunks, so the sliding layers use the position-aware recompute ring.

Both verified (out + dq/dk/dv == single-GPU) in test_ring_attn.py.

Zigzag layout: each document is padded to a multiple of 2*cp_size and split into 2*cp_size equal
chunks; CP rank r holds chunk r and chunk (2*cp_size-1-r), concatenated [front, back] per doc.
Padding tokens get label -100 (no loss) and sit at each doc's end, so the causal/sliding mask never
lets a real token attend them — no extra masking needed.
"""
import os
import numpy as np
import torch
import torch.distributed as dist

from ring_zigzag_attn import zigzag_ring_flash_attn_varlen_func, posaware_ring_attn_func

# Per-batch context set before each model forward (the AttentionInterface fn can't see these
# otherwise): the CP process group + this rank's local per-token position ids / global doc ids.
_CP = {"group": None, "pos": None, "doc": None, "active": False}


def cp_active() -> bool:
    return _CP["active"]


def setup_cp(world_size: int, cp_size: int, global_rank: int):
    """Partition ranks into world_size/cp_size CP groups of cp_size CONSECUTIVE ranks. Returns
    (cp_group, dp_size, dp_rank, cp_rank). Every rank must call this (new_group is collective).

    ⚠ dp==1 (cp_size == world_size) REUSES THE DEFAULT process group (None): the ring must share
    FSDP's communicator so there is ONE global comm ordering. A separate all-rank `new_group` puts
    the ring P2P and FSDP's all-gather/reduce-scatter on TWO NCCL comms over the same GPUs; their
    per-rank enqueue orders differ → cross-communicator circular wait. Empirically: gemma cp4
    survived the race, cp8 @512k WEDGED every step-0 backward (all ranks blocked in the autograd
    engine, GPU memory flat at 100% util). Same class as the qwen deadlock — default-PG sharing is
    the proven fix (see qwen/context_parallel.py). RingComm already handles group=None."""
    assert world_size % cp_size == 0, f"world_size {world_size} not divisible by cp_size {cp_size}"
    dp_size = world_size // cp_size
    cp_group = None
    if dp_size > 1:
        for d in range(dp_size):
            ranks = list(range(d * cp_size, (d + 1) * cp_size))
            g = dist.new_group(ranks)      # collective: all ranks create all groups
            if global_rank in ranks:
                cp_group = g
    dp_rank = global_rank // cp_size
    cp_rank = global_rank % cp_size
    _CP["group"] = cp_group
    _CP["active"] = True
    return cp_group, dp_size, dp_rank, cp_rank


def shard_batch(batch: dict, cp_size: int, cp_rank: int, pad_id: int = 0) -> dict:
    """Zigzag-shard a collated full batch for this CP rank. In: input_ids/position_ids/labels
    (1, S) + cu_seq_lens_q. Out: the local shard (1, S_local) + LOCAL cu_seq_lens + max_length,
    and sets _CP['pos']/_CP['doc'] (this rank's global per-token position + doc ids) for the
    sliding-window ring. Every doc is padded to a multiple of 2*cp_size."""
    twoW = 2 * cp_size
    ids = batch["input_ids"][0]
    pos = batch["position_ids"][0]
    lab = batch["labels"][0]
    cu = batch["cu_seq_lens_q"].tolist()
    dev = ids.device
    loc_ids, loc_pos, loc_lab, loc_doc, loc_lens = [], [], [], [], []
    for d in range(len(cu) - 1):
        s, e = cu[d], cu[d + 1]
        di, dp, dl = ids[s:e], pos[s:e], lab[s:e]
        L = e - s
        # Causal-LM target = the NEXT token's label, computed PER-DOC here (not with the loss's
        # global hidden[:-1]/labels[1:] shift, which is wrong under zigzag: adjacent local tokens
        # aren't globally adjacent). tgt[i] = dl[i+1]; the doc's last real token has no next -> -100.
        # The CP loss then uses these targets directly (no shift) — see the trainer's cp branch.
        dt = torch.cat([dl[1:], torch.full((1,), -100, dtype=dl.dtype, device=dev)])
        pad = (-L) % twoW
        if pad:
            di = torch.cat([di, torch.full((pad,), pad_id, dtype=di.dtype, device=dev)])
            last = int(dp[-1].item()) + 1
            dp = torch.cat([dp, torch.arange(last, last + pad, dtype=dp.dtype, device=dev)])
            dt = torch.cat([dt, torch.full((pad,), -100, dtype=dl.dtype, device=dev)])  # pad targets ignored
        dl = dt                                            # local "labels" are now per-token targets
        c = (L + pad) // twoW
        for chunk in (cp_rank, twoW - 1 - cp_rank):        # [front, back] per doc
            sl = slice(chunk * c, (chunk + 1) * c)
            loc_ids.append(di[sl]); loc_pos.append(dp[sl]); loc_lab.append(dl[sl])
            loc_doc.append(torch.full((c,), d, dtype=torch.long, device=dev))
        loc_lens.append(2 * c)
    li = torch.cat(loc_ids).unsqueeze(0)
    lp = torch.cat(loc_pos)
    ld = torch.cat(loc_doc)
    cumsum = [0] + np.cumsum(loc_lens).tolist()
    _CP["pos"], _CP["doc"] = lp, ld
    return {
        "input_ids": li,
        "position_ids": lp.unsqueeze(0),
        "labels": torch.cat(loc_lab).unsqueeze(0),
        "attention_mask": None,
        "mm_token_type_ids": torch.zeros_like(li),
        "cu_seq_lens_q": torch.tensor(cumsum, dtype=torch.int32, device=dev),
        "cu_seq_lens_k": torch.tensor(cumsum, dtype=torch.int32, device=dev),
        "max_length_q": int(max(loc_lens)) if loc_lens else 0,
        "max_length_k": int(max(loc_lens)) if loc_lens else 0,
    }


def cp_ring_attention(module, query, key, value, attention_mask,
                      cu_seq_lens_q=None, cu_seq_lens_k=None,
                      max_length_q=None, max_length_k=None,
                      sliding_window=None, scaling=None, **kwargs):
    """AttentionInterface backend used under CP. Dispatches per gemma-4 layer type: sliding-window
    (head_dim 256) → position-aware ring; full-causal global (head_dim 512) → fused zigzag ring."""
    group = _CP["group"]
    # (B=1, H, S, D) -> packed (S, H, D)
    q = query.permute(0, 2, 1, 3).squeeze(0).contiguous().to(torch.bfloat16)
    k = key.permute(0, 2, 1, 3).squeeze(0).contiguous().to(torch.bfloat16)
    v = value.permute(0, 2, 1, 3).squeeze(0).contiguous().to(torch.bfloat16)
    if sliding_window is not None:
        out = posaware_ring_attn_func(
            q, k, v, _CP["pos"], _CP["doc"], softmax_scale=scaling,
            window=int(sliding_window), group=group)
    else:
        cu = cu_seq_lens_q.to(device=q.device, dtype=torch.int32)
        out = zigzag_ring_flash_attn_varlen_func(
            q, k, v, cu, max_length_q, softmax_scale=scaling, causal=True,
            window_size=(None, None), group=group)
    return out.unsqueeze(0), None
