"""Correctness gate for the NemotronH ATTENTION fast path (hub FA3) vs sdpa — including
per-projection q/k/v/o weight GRADIENTS (finiteness + cosine vs reference), the attention
analogue of test_mamba_kernels.py.

One NemotronHAttention at the real 30B geometry (head_dim 128, 32q/2kv GQA, causal, NO
rope — the module applies none), bf16, single GPU. The attention backend is selected via
`config._attn_implementation` at forward time, so the SAME module (same weights) runs:

  reference : sdpa            (what the clean baseline trained with; is_causal internally)
  candidate : kernels-community/flash-attn3 (the trainer's default)

(NOT eager: with attention_mask=None the eager path applies NO causal mask — sdpa/FA
implementations handle causality via `is_causal` internally, eager expects an explicit
4D mask — so eager here computes bidirectional attention and is not a valid reference.)

PASS per candidate = fwd cosine ≥ 0.999 AND all of x/q/k/v/o grads finite AND each grad
cosine ≥ 0.99 vs the sdpa reference.
"""
import os
os.environ.setdefault("HF_HOME", "/share/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import AutoConfig
from transformers.models.nemotron_h import modeling_nemotron_h as M

MODEL_ID = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
CAND_ATTN = os.environ.get("NEMOTRON_ATTN", "kernels-community/flash-attn3")
SEQ_LENS = [int(s) for s in os.environ.get("TEST_SEQ_LENS", "749,2048").split(",")]

cfg = AutoConfig.from_pretrained(MODEL_ID)
torch.manual_seed(0)
dev = "cuda"

# from_pretrained registers hub attention kernels during model init; a bare module test
# must register the candidate explicitly (same helper modeling_utils uses).
from transformers.integrations.hub_kernels import load_and_register_attn_kernel  # noqa: E402
load_and_register_attn_kernel(CAND_ATTN)
# layer 5 is the first 'attention' entry in the 30B hybrid pattern (M E M E M *…)
attn_idx = list(cfg.layers_block_type).index("attention")
attn = M.NemotronHAttention(cfg, layer_idx=attn_idx).to(dev, torch.bfloat16)
print(f"geometry: heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
      f"head_dim={attn.head_dim} hidden={cfg.hidden_size} (layer_idx={attn_idx})")


def run(impl, x):
    cfg._attn_implementation = impl
    attn.train()
    attn.zero_grad(set_to_none=True)
    xi = x.clone().requires_grad_(True)
    out, _ = attn(xi, attention_mask=None)
    out.float().pow(2).mean().backward()
    grads = {"x": xi.grad.detach().clone()}
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        g = getattr(attn, n).weight.grad
        grads[n] = g.detach().clone() if g is not None else None
    return out.detach(), grads


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


failures = 0
for L in SEQ_LENS:
    x = torch.randn(1, L, cfg.hidden_size, device=dev, dtype=torch.bfloat16) * 0.5
    ref_out, ref_g = run("sdpa", x)
    print(f"\n== L={L} · sdpa reference: |out|={ref_out.float().norm():.3f}")

    cand_out, cand_g = run(CAND_ATTN, x)
    fc = cos(cand_out, ref_out)
    fin = {n: bool(torch.isfinite(g).all()) if g is not None else None for n, g in cand_g.items()}
    gcs = {n: round(cos(cand_g[n], ref_g[n]), 4) for n in cand_g
           if cand_g[n] is not None and ref_g[n] is not None and fin[n]}
    ok = fc >= 0.999 and all(fin.values()) and all(v >= 0.99 for v in gcs.values())
    print(f"  {CAND_ATTN}: {'PASS' if ok else 'FAIL'} — fwd_cos={fc:.6f} "
          f"grad_finite={fin} grad_cos={gcs}")
    if not ok:
        failures += 1

print(f"\n{'ALL ATTENTION-KERNEL CHECKS PASS ✅' if failures == 0 else f'{failures} FAILURE(S) ❌'}")
raise SystemExit(1 if failures else 0)
