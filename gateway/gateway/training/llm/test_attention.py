"""Correctness test for dynamic_attention (gemma4 packed training).

Confirms the two things the training run depends on:

  1. SDPA branch  (head_dim 512, full attention): the rebuilt block-diagonal *boolean*
     mask reproduces true per-document causal attention — i.e. no token attends across a
     document boundary and no token attends to the future. This is the bug that was fixed:
     a 0/1 *float* mask is an additive bias to SDPA and masks nothing.

  2. FA3 branch   (head_dim 256, sliding/full): the cu_seqlens varlen kernel reproduces the
     same per-document causal attention. Requires a GPU + flash_attn_interface (FA3);
     skipped with a clear message otherwise.

Both are checked against an independent reference that loops over documents and runs a
plain causal softmax — if dynamic_attention leaked across docs or dropped causality, the
reference would disagree.

    python test_attention.py
"""
import sys

import torch
import torch.nn.functional as F

from attention import dynamic_attention


def reference_packed_causal(q, k, v, cu, scale):
    """Per-document causal attention reference. q:(1,H,S,D) k/v:(1,Hkv,S,D) -> (1,S,H,D)."""
    H = q.shape[1]
    Hkv = k.shape[1]
    if H != Hkv:  # GQA: replicate kv heads
        rep = H // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    outs = []
    qf, kf, vf = q.float(), k.float(), v.float()
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
        qd, kd, vd = qf[:, :, a:b], kf[:, :, a:b], vf[:, :, a:b]
        scores = torch.matmul(qd, kd.transpose(-1, -2)) * scale  # (1,H,L,L)
        L = b - a
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.matmul(probs, vd))  # (1,H,L,D)
    out = torch.cat(outs, dim=2)  # (1,H,S,D)
    return out.transpose(1, 2).contiguous()  # (1,S,H,D)


def make_inputs(H, Hkv, D, doc_lens, dtype, device):
    S = sum(doc_lens)
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(1, H, S, D, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    k = torch.randn(1, Hkv, S, D, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    v = torch.randn(1, Hkv, S, D, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    cu = torch.tensor([0] + torch.cumsum(torch.tensor(doc_lens), 0).tolist(), dtype=torch.int32, device=device)
    return q, k, v, cu


def report(name, got, ref, atol, rtol):
    got, ref = got.float(), ref.float()
    max_abs = (got - ref).abs().max().item()
    denom = ref.norm().item() or 1.0
    rel = (got - ref).norm().item() / denom
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max_abs={max_abs:.3e} rel_l2={rel:.3e} (atol={atol}, rtol={rtol})")
    return ok


def test_sdpa():
    """head_dim 512 -> SDPA full-attention branch. fp32, runs on CPU or GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, Hkv, D = 8, 2, 512  # GQA, head_dim>256 forces the SDPA branch
    doc_lens = [5, 7, 4]
    scale = 1.0 / (D ** 0.5)
    q, k, v, cu = make_inputs(H, Hkv, D, doc_lens, torch.float32, device)

    out, _ = dynamic_attention(
        None, q, k, v, None,
        cu_seq_lens_q=cu, cu_seq_lens_k=cu,
        max_length_q=max(doc_lens), max_length_k=max(doc_lens),
        sliding_window=None, scaling=scale,
    )
    ref = reference_packed_causal(q, k, v, cu, scale)
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape}"
    return report("SDPA block-diagonal causal (head_dim=512, GQA)", out, ref, atol=1e-4, rtol=1e-3)


def test_sdpa_float_mask_would_be_wrong():
    """Document the fix: a 0/1 FLOAT mask masks nothing in SDPA (it is an additive bias)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, D, L = 4, 64, 6
    g = torch.Generator(device="cpu").manual_seed(1)
    q = torch.randn(1, H, L, D, generator=g).to(device)
    k = torch.randn(1, H, L, D, generator=g).to(device)
    v = torch.randn(1, H, L, D, generator=g).to(device)
    causal = torch.tril(torch.ones(L, L, device=device))
    bool_out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal.bool())
    float_out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)  # the OLD buggy behaviour
    differ = not torch.allclose(bool_out, float_out, atol=1e-3)
    print(f"[{'PASS' if differ else 'FAIL'}] float-mask != bool-mask (proves the bool fix matters): differ={differ}")
    return differ


def test_fa3():
    """head_dim 256 -> FA3 varlen branch. Needs GPU + flash_attn_interface."""
    if not torch.cuda.is_available():
        print("[SKIP] FA3 cu_seqlens varlen: no CUDA device")
        return None
    try:
        import flash_attn_interface  # noqa: F401
    except Exception as e:
        print(f"[SKIP] FA3 cu_seqlens varlen: flash_attn_interface not importable ({e})")
        return None

    device = "cuda"
    H, Hkv, D = 8, 2, 256  # head_dim<=256 -> FA3 branch; GQA
    doc_lens = [16, 24, 8]
    scale = 1.0 / (D ** 0.5)
    q, k, v, cu = make_inputs(H, Hkv, D, doc_lens, torch.bfloat16, device)

    out, _ = dynamic_attention(
        None, q, k, v, None,
        cu_seq_lens_q=cu, cu_seq_lens_k=cu,
        max_length_q=max(doc_lens), max_length_k=max(doc_lens),
        sliding_window=None, scaling=scale,
    )
    ref = reference_packed_causal(q, k, v, cu, scale)
    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape}"
    # bf16 inputs -> loose tolerance vs the fp32 reference.
    return report("FA3 cu_seqlens varlen causal (head_dim=256, GQA, bf16)", out, ref, atol=2e-2, rtol=2e-2)


def test_sdpa_query_block_fwd_bwd():
    """Query-blocked SDPA (the head_dim-512 memory-bounded path) must match the per-doc
    causal reference on BOTH the forward output AND the input gradients (q/k/v .grad).

    We run the SAME inputs through (a) the manual per-doc causal reference and (b)
    dynamic_attention with SDPA_QUERY_BLOCK forced small (so the query-tiling +
    per-block activation-checkpoint path is exercised), then backprop the SAME upstream
    gradient through each and compare .grad. If tiling/checkpointing changed the math —
    dropped a block's contribution to k/v, mis-sliced the mask, or broke grad
    accumulation across blocks — the gradients would diverge.
    """
    import attention as A
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, Hkv, D = 8, 2, 512                 # GQA + head_dim>256 -> SDPA full-attention branch
    doc_lens = [9, 13, 6]                  # packed, multi-doc; S=28
    S = sum(doc_lens)
    scale = 1.0 / (D ** 0.5)
    q, k, v, cu = make_inputs(H, Hkv, D, doc_lens, torch.float32, device)

    # two independent grad-tracking copies of the same values
    qr, kr, vr = (t.clone().requires_grad_(True) for t in (q, k, v))
    qb, kb, vb = (t.clone().requires_grad_(True) for t in (q, k, v))

    ref = reference_packed_causal(qr, kr, vr, cu, scale)          # (1, S, H, D)

    old = A.SDPA_QUERY_BLOCK
    A.SDPA_QUERY_BLOCK = 4                                        # force tiling: blocks of 4 over S=28
    try:
        out, _ = A.dynamic_attention(
            None, qb, kb, vb, None,
            cu_seq_lens_q=cu, cu_seq_lens_k=cu,
            max_length_q=max(doc_lens), max_length_k=max(doc_lens),
            sliding_window=None, scaling=scale,
        )
    finally:
        A.SDPA_QUERY_BLOCK = old

    fwd_ok = report("query-blocked SDPA fwd vs reference (head_dim=512, tiled)", out, ref, atol=1e-4, rtol=1e-3)

    g = torch.randn_like(out)                                    # same upstream grad for both
    (out * g).sum().backward()
    (ref * g).sum().backward()
    grad_checks = [
        report("  d/dquery", qb.grad, qr.grad, atol=1e-4, rtol=1e-3),
        report("  d/dkey  ", kb.grad, kr.grad, atol=1e-4, rtol=1e-3),
        report("  d/dvalue", vb.grad, vr.grad, atol=1e-4, rtol=1e-3),
    ]
    return fwd_ok and all(grad_checks)


if __name__ == "__main__":
    print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")
    results = [
        test_sdpa(),
        test_sdpa_float_mask_would_be_wrong(),
        test_sdpa_query_block_fwd_bwd(),
        test_fa3(),
    ]
    checked = [r for r in results if r is not None]
    if all(checked):
        print(f"\nALL {len(checked)} CHECKS PASSED")
        sys.exit(0)
    print(f"\n{sum(1 for r in checked if not r)} CHECK(S) FAILED")
    sys.exit(1)
