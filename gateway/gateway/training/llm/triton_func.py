"""Fused linear + log-prob Triton kernel (online softmax over vocab tiles).

VENDORED from small-ablation/multipacking-dpo/triton_func.py (the authority —
re-sync from there). Consumed by triton_dpo.fused_dpo_loss; the `__main__`
block is a correctness gate (kernel vs fp32 torch reference), not a demo.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_linear_logprob_fwd_kernel(
    X,
    W,
    T,
    LOGPROB,
    LSE,
    N,
    H,
    V,
    stride_xn,
    stride_xh,
    stride_wv,
    stride_wh,
    ignore_index,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    DOT_BF16: tl.constexpr,
):
    """
    Fused linear + log-prob forward with online softmax.

    For a block of BLOCK_N tokens, iterate over the vocab in BLOCK_V tiles.
    Each tile of logits is computed with tl.dot over BLOCK_H slices of the
    hidden dimension, so the full [N, V] logits matrix is never materialized.

    Outputs per token:
        LOGPROB[n] = logits[n, T[n]] - logsumexp(logits[n, :])  (0 if T[n] == ignore_index)
        LSE[n]     = logsumexp(logits[n, :])                    (saved for backward)

    When both operands are bf16 the dot runs on bf16 tensor cores with fp32
    accumulate (exact for bf16 data); otherwise operands are upcast to fp32 and
    the dot uses tf32x3, which is within ~1e-6 relative of an IEEE fp32 matmul.
    """
    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    targets = tl.load(T + n_offs, mask=n_mask, other=ignore_index)

    m_i = tl.full([BLOCK_N], float("-inf"), tl.float32)  # running max
    l_i = tl.zeros([BLOCK_N], tl.float32)  # running sum of exp
    t_logit = tl.zeros([BLOCK_N], tl.float32)  # logit at target index

    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < V

        acc = tl.zeros([BLOCK_N, BLOCK_V], tl.float32)
        for h_start in range(0, H, BLOCK_H):
            h_offs = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offs < H

            x_tile = tl.load(
                X + n_offs[:, None] * stride_xn + h_offs[None, :] * stride_xh,
                mask=n_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            w_tile = tl.load(
                W + v_offs[:, None] * stride_wv + h_offs[None, :] * stride_wh,
                mask=v_mask[:, None] & h_mask[None, :],
                other=0.0,
            )

            if DOT_BF16:
                acc += tl.dot(x_tile.to(tl.bfloat16), tl.trans(w_tile.to(tl.bfloat16)))
            else:
                acc += tl.dot(x_tile.to(tl.float32), tl.trans(w_tile.to(tl.float32)),
                              input_precision="tf32x3")

        acc = tl.where(v_mask[None, :], acc, float("-inf"))

        # online softmax update
        tile_max = tl.max(acc, axis=1)
        new_m = tl.maximum(m_i, tile_max)
        l_i = l_i * tl.exp(m_i - new_m) + tl.sum(tl.exp(acc - new_m[:, None]), axis=1)
        m_i = new_m

        # pick out the target logit if it lives in this vocab tile
        is_target = (v_offs[None, :] == targets[:, None]) & v_mask[None, :]
        t_logit += tl.sum(tl.where(is_target, acc, 0.0), axis=1)

    lse = m_i + tl.log(l_i)
    valid = (targets != ignore_index) & n_mask
    log_prob = tl.where(valid, t_logit - lse, 0.0)

    tl.store(LOGPROB + n_offs, log_prob, mask=n_mask)
    tl.store(LSE + n_offs, lse, mask=n_mask)


def fused_linear_logprob(inputs, targets, weight, ignore_index=-100):
    """
    Per-token log p(target | input) for a linear head, without materializing logits.

    Args:
        inputs: [N, H] bf16/fp16/fp32 hidden states (packed, no padding)
        targets: [N] int64 token ids, ignore_index entries get log-prob 0
        weight: [V, H] head weight

    Returns:
        log_probs: [N] fp32
        lse: [N] fp32 logsumexp per token, reusable by the backward pass
    """
    N, H = inputs.shape
    V = weight.shape[0]
    log_probs = torch.empty(N, device=inputs.device, dtype=torch.float32)
    lse = torch.empty(N, device=inputs.device, dtype=torch.float32)

    BLOCK_N, BLOCK_V, BLOCK_H = 64, 128, 64
    grid = (triton.cdiv(N, BLOCK_N),)
    _fused_linear_logprob_fwd_kernel[grid](
        inputs,
        weight,
        targets,
        log_probs,
        lse,
        N,
        H,
        V,
        inputs.stride(0),
        inputs.stride(1),
        weight.stride(0),
        weight.stride(1),
        ignore_index,
        BLOCK_N=BLOCK_N,
        BLOCK_V=BLOCK_V,
        BLOCK_H=BLOCK_H,
        DOT_BF16=(inputs.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16),
        num_warps=8,
    )
    return log_probs, lse


def get_sum_logprob_triton(inputs, targets, weight, ignore_index=-100, average_log_prob=False):
    """Sum (or mean) of per-token log-probs via the fused Triton kernel."""
    log_probs, _ = fused_linear_logprob(inputs, targets, weight, ignore_index)
    sum_log_prob = log_probs.sum()
    if average_log_prob:
        sum_log_prob = sum_log_prob / (targets != ignore_index).sum()
    return sum_log_prob


def get_sum_logprob_reference(inputs, targets, weight, chunk_size=512, ignore_index=-100, average_log_prob=False):
    """Chunked fp32 torch reference."""
    sum_log_prob = 0.0
    BT, H = inputs.shape

    for start_idx in range(0, BT, chunk_size):
        end_idx = min(start_idx + chunk_size, BT)
        _inputs_chunk = inputs[start_idx:end_idx].float()
        _targets_chunk = targets[start_idx:end_idx]

        logits = _inputs_chunk @ weight.float().T
        log_probs_chunk = F.log_softmax(logits, dim=-1)

        loss_mask = _targets_chunk != ignore_index
        label_chunk = torch.where(loss_mask, _targets_chunk, 0)
        per_token_logps = log_probs_chunk.gather(-1, label_chunk.unsqueeze(-1)).squeeze(-1)
        sum_log_prob += (per_token_logps * loss_mask).sum(-1)

    if average_log_prob:
        sum_log_prob = sum_log_prob / (targets != ignore_index).sum()
    return sum_log_prob


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda")

    # correctness on a small shape, including ignore_index handling
    BT, H, V = 1000, 4096, 32000
    inputs = torch.randn(BT, H, device=device, dtype=torch.float32)
    targets = torch.randint(0, V, (BT,), device=device)
    targets[::7] = -100
    weight = torch.randn(V, H, device=device, dtype=torch.float32) / H**0.5

    ref = get_sum_logprob_reference(inputs, targets, weight)
    tri = get_sum_logprob_triton(inputs, targets, weight)
    print(f"small fp32   reference={ref.item():.6f} triton={tri.item():.6f} diff={abs(ref.item() - tri.item()):.6e}")
    assert torch.allclose(ref, tri, atol=5e-2, rtol=0)

    ref16 = get_sum_logprob_reference(inputs.bfloat16(), targets, weight)
    tri16 = get_sum_logprob_triton(inputs.bfloat16(), targets, weight)
    print(f"small bf16   reference={ref16.item():.6f} triton={tri16.item():.6f} diff={abs(ref16.item() - tri16.item()):.6e}")
    assert torch.allclose(ref16, tri16, atol=5e-2, rtol=0)

    # per-token agreement
    log_probs, lse = fused_linear_logprob(inputs, targets, weight)
    logits = inputs @ weight.T
    ref_lse = torch.logsumexp(logits, dim=-1)
    print(f"per-token lse max abs diff: {(lse - ref_lse).abs().max().item():.6e}")
    assert torch.allclose(lse, ref_lse, atol=1e-3, rtol=0)

    # Qwen3-32B head shape benchmark: H=5120, V=151936
    # https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json
    BT, H, V = 10000, 5120, 151936
    inputs = torch.randn(BT, H, device=device, dtype=torch.bfloat16)
    targets = torch.randint(0, V, (BT,), device=device)
    weight = torch.nn.Linear(H, V).cuda().weight.detach()

    for name, fn in [
        ("triton fused (no logits materialized)", lambda: get_sum_logprob_triton(inputs, targets, weight)),
        ("torch chunked fp32", lambda: get_sum_logprob_reference(inputs, targets, weight)),
    ]:
        fn()  # warmup / compile
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        print(f"{name}: {start.elapsed_time(end):.1f} ms, result={out.item():.4f}, "
              f"peak={torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
