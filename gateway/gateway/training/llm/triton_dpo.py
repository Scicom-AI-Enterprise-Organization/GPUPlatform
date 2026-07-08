"""Fused DPO loss for multipacked (padding-free) sequences — logits never materialized.

VENDORED from small-ablation/multipacking-dpo/triton_dpo.py (the authority — re-sync
from there). Packed layout contract: hidden [T, H] holds 2K sequences concatenated on
the sequence dim, boundaries in cu_seqlens [2K+1], FIRST K CHOSEN then K rejected;
targets[T] are pre-aligned next-token ids (-100 on prompt tokens + each sequence's
final position) — no shifting at loss time. The `__main__` block is a correctness
gate (fused vs full-autograd fp32 reference), not a demo.
"""
import torch
import torch.nn.functional as F

from triton_func import fused_linear_logprob


class TritonFusedDPO(torch.autograd.Function):
    """
    DPO loss for packed (multipacked, padding-free) sequences.

    Layout: `hidden`/`ref_hidden` are [T, H] with 2K sequences concatenated on the
    sequence dimension, boundaries given by `cu_seqlens` [2K + 1]. The first K
    sequences are the chosen responses, the last K the rejected ones, so pair k
    is (sequence k, sequence K + k).

    `targets` [T] holds the next-token id for every position (already aligned,
    no shifting here) with `ignore_index` on prompt tokens and final positions.

    Memory strategy:
      - forward: the Triton kernel returns per-token log-probs and logsumexp
        without materializing logits, so nothing besides the [T] lse vector is
        kept for backward.
      - backward: logits are recomputed chunk-by-chunk, converted in place to
        d(loss)/d(logits) = coeff * (onehot(target) - softmax) using the saved
        lse, then reduced into grad_hidden and a single grad_weight buffer.

    Unlike the forward-time gradient stashing of the chunked v2/v3 notebooks,
    the DPO coefficients (one per pair) are known here, so any number of packed
    pairs works with one [V, H] gradient buffer.
    """

    @staticmethod
    def forward(ctx, hidden, ref_hidden, targets, cu_seqlens, weight, ref_weight,
                beta=0.1, chunk_size=1024, ignore_index=-100):
        num_seqs = cu_seqlens.numel() - 1
        assert num_seqs % 2 == 0, "expect K chosen followed by K rejected sequences"
        K = num_seqs // 2

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        seq_id = torch.repeat_interleave(
            torch.arange(num_seqs, device=hidden.device), lengths)

        log_probs, lse = fused_linear_logprob(hidden, targets, weight, ignore_index)
        with torch.no_grad():
            ref_log_probs, _ = fused_linear_logprob(ref_hidden, targets, ref_weight, ignore_index)

        seq_logprob = torch.zeros(num_seqs, device=hidden.device, dtype=torch.float32)
        seq_logprob.index_add_(0, seq_id, log_probs)
        ref_seq_logprob = torch.zeros(num_seqs, device=hidden.device, dtype=torch.float32)
        ref_seq_logprob.index_add_(0, seq_id, ref_log_probs)

        chosen_logratios = seq_logprob[:K] - ref_seq_logprob[:K]
        rejected_logratios = seq_logprob[K:] - ref_seq_logprob[K:]
        z = beta * (chosen_logratios - rejected_logratios)
        loss = -F.logsigmoid(z).mean()

        chosen_rewards = beta * chosen_logratios
        rejected_rewards = beta * rejected_logratios

        ctx.save_for_backward(hidden, targets, weight, lse, z, seq_id)
        ctx.beta = beta
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.mark_non_differentiable(chosen_rewards, rejected_rewards)
        return loss, chosen_rewards, rejected_rewards

    @staticmethod
    def backward(ctx, grad_loss, *_):
        hidden, targets, weight, lse, z, seq_id = ctx.saved_tensors
        K = z.numel()
        T = hidden.shape[0]

        # dL/d(seq_logprob): -beta/K * sigmoid(-z_k) for chosen, + for rejected
        c_pair = grad_loss * ctx.beta * torch.sigmoid(-z) / K
        coeff_seq = torch.cat([-c_pair, c_pair])
        valid = targets != ctx.ignore_index
        coeff_tok = coeff_seq[seq_id] * valid

        w32 = weight.float()
        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)

        for start in range(0, T, ctx.chunk_size):
            end = min(start + ctx.chunk_size, T)
            x_chunk = hidden[start:end].float()

            # recompute logits, then reuse the buffer for d(loss)/d(logits)
            g = x_chunk @ w32.T
            g.sub_(lse[start:end, None]).exp_().neg_()  # -softmax
            rows = torch.arange(end - start, device=g.device)
            safe_t = torch.where(valid[start:end], targets[start:end], 0)
            g[rows, safe_t] += valid[start:end].float()  # onehot - softmax
            g.mul_(coeff_tok[start:end, None])

            grad_hidden[start:end] = (g @ w32).to(hidden.dtype)
            grad_weight.addmm_(g.T, x_chunk)

        return (grad_hidden, None, None, None, grad_weight.to(weight.dtype),
                None, None, None, None)


def fused_dpo_loss(hidden, ref_hidden, targets, cu_seqlens, weight, ref_weight,
                   beta=0.1, chunk_size=1024, ignore_index=-100):
    """
    Returns (loss, chosen_rewards [K], rejected_rewards [K]).
    See TritonFusedDPO for the packed layout contract.
    """
    return TritonFusedDPO.apply(hidden, ref_hidden, targets, cu_seqlens, weight,
                                ref_weight, beta, chunk_size, ignore_index)


def dpo_loss_reference(hidden, ref_hidden, targets, cu_seqlens, weight, ref_weight,
                       beta=0.1, ignore_index=-100):
    """Materialize-everything fp32 autograd reference, for validation only."""
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    num_seqs = lengths.numel()
    K = num_seqs // 2
    seq_id = torch.repeat_interleave(torch.arange(num_seqs, device=hidden.device), lengths)
    valid = targets != ignore_index
    safe_t = torch.where(valid, targets, 0)

    def seq_logprobs(h, w):
        log_probs = F.log_softmax(h.float() @ w.float().T, dim=-1)
        tok = log_probs.gather(1, safe_t[:, None]).squeeze(1) * valid
        out = torch.zeros(num_seqs, device=h.device, dtype=tok.dtype)
        out.index_add_(0, seq_id, tok)
        return out

    seq_logprob = seq_logprobs(hidden, weight)
    with torch.no_grad():
        ref_seq_logprob = seq_logprobs(ref_hidden, ref_weight)

    z = beta * ((seq_logprob[:K] - ref_seq_logprob[:K]) - (seq_logprob[K:] - ref_seq_logprob[K:]))
    loss = -F.logsigmoid(z).mean()
    return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda")

    # ---- correctness on a small shape (loss + grads vs full autograd) ----
    H, V, K = 256, 1203, 3
    lens = torch.tensor([120, 75, 200, 90, 133, 61])  # 3 chosen + 3 rejected
    cu = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]).to(device)
    T = int(lens.sum())

    hidden = torch.randn(T, H, device=device) / H**0.5
    ref_hidden = torch.randn(T, H, device=device) / H**0.5
    targets = torch.randint(0, V, (T,), device=device)
    # mask "prompt" prefixes and final positions like real usage
    for i in range(2 * K):
        targets[cu[i]:cu[i] + 10] = -100
        targets[cu[i + 1] - 1] = -100
    weight = (torch.randn(V, H, device=device) / H**0.5).requires_grad_(True)
    ref_weight = torch.randn(V, H, device=device) / H**0.5
    hidden.requires_grad_(True)

    loss, cr, rr = fused_dpo_loss(hidden, ref_hidden, targets, cu, weight, ref_weight, chunk_size=97)
    loss.backward()
    gh, gw = hidden.grad.clone(), weight.grad.clone()
    hidden.grad = None
    weight.grad = None

    ref_loss = dpo_loss_reference(hidden, ref_hidden, targets, cu, weight, ref_weight)
    ref_loss.backward()

    print(f"loss: fused={loss.item():.6f} reference={ref_loss.item():.6f} "
          f"diff={abs(loss.item() - ref_loss.item()):.3e}")
    print(f"grad_hidden max abs diff: {(gh - hidden.grad).abs().max().item():.3e}")
    print(f"grad_weight max abs diff: {(gw - weight.grad).abs().max().item():.3e}")
    assert torch.allclose(loss, ref_loss, atol=1e-4, rtol=0)
    assert torch.allclose(gh, hidden.grad, atol=1e-4, rtol=0)
    assert torch.allclose(gw, weight.grad, atol=1e-4, rtol=0)

    # ---- bf16 hidden states, fp32 weight (study setup) ----
    hidden16 = hidden.detach().bfloat16().requires_grad_(True)
    ref_hidden16 = ref_hidden.detach().bfloat16()
    loss16, _, _ = fused_dpo_loss(hidden16, ref_hidden16, targets, cu, weight, ref_weight)
    loss16.backward()
    ref_loss16 = dpo_loss_reference(hidden16, ref_hidden16, targets, cu, weight, ref_weight)
    print(f"bf16 loss: fused={loss16.item():.6f} reference={ref_loss16.item():.6f}")
    assert torch.allclose(loss16, ref_loss16, atol=1e-3, rtol=0)
    print("all correctness checks passed")

    # ---- Qwen3-32B head shape timing: one packed pair, 10k + 5k tokens ----
    # https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json
    H, V = 5120, 151936
    cu = torch.tensor([0, 10000, 15000], device=device)
    hidden = torch.randn(15000, H, device=device, dtype=torch.bfloat16, requires_grad=True)
    ref_hidden = torch.randn(15000, H, device=device, dtype=torch.bfloat16)
    targets = torch.randint(0, V, (15000,), device=device)
    weight = torch.nn.Linear(H, V).cuda().weight
    ref_weight = torch.nn.Linear(H, V).cuda().weight.detach()

    def run():
        loss, _, _ = fused_dpo_loss(hidden, ref_hidden, targets, cu, weight, ref_weight)
        loss.backward()
        return loss

    run()  # warmup
    weight.grad, hidden.grad = None, None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss = run()
    end.record()
    torch.cuda.synchronize()
    print(f"fwd+bwd 15k tokens H={H} V={V}: {start.elapsed_time(end):.1f} ms, "
          f"loss={loss.item():.4f}, peak={torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
