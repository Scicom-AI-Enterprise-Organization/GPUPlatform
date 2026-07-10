"""Correctness gate for the NemotronH Mamba2 fast path (kernels hub mamba-ssm/causal-conv1d)
vs transformers' torch_forward reference — the mixer-level analogue of gemma's
test_attention.py. Run BEFORE any training that enables NEMOTRON_MAMBA_KERNELS.

One NemotronHMamba2Mixer at the real Nemotron-3-Nano-30B geometry (random weights),
single GPU, no FSDP/checkpointing:

  1. torch_forward (reference)     — fwd out + input/param grads
  2. cuda non-split path           — use_mem_eff_path=False (what the out_proj-LoRA
                                     guard forces in the gateway trainer): causal_conv1d_fn
                                     + mamba_chunk_scan_combined + module-call out_proj
  3. cuda mem-eff (split) path     — use_mem_eff_path=True: mamba_split_conv1d_scan_combined
                                     (known: asserts causal_conv1d_cuda in the hub build —
                                     expected FAIL in this env; kept to detect if that changes)
  4. (2) again with kernel inputs forced .contiguous() — isolates stride bugs

PASS = fwd cosine ≥ 0.999 vs torch AND all grads finite AND grad cosine ≥ 0.99.
Findings 2026-07-10 (tm-2 H20, torch 2.10 cu130, hub mamba-ssm a39ff24): see CLAUDE.md.
"""
import os
os.environ.setdefault("HF_HOME", "/share/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import AutoConfig
from transformers.models.nemotron_h import modeling_nemotron_h as M

MODEL_ID = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
SEQ_LENS = [int(s) for s in os.environ.get("TEST_SEQ_LENS", "749,2048").split(",")]
# bf16 (training dtype) by default; TEST_DTYPE=fp32 discriminates bf16-numerics vs kernel bug.
DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("TEST_DTYPE", "bf16")]

cfg = AutoConfig.from_pretrained(MODEL_ID)
cfg.use_mamba_kernels = True

torch.manual_seed(0)
dev = "cuda"
mixer = M.NemotronHMamba2Mixer(cfg, layer_idx=0).to(dev, DTYPE)
assert M.is_fast_path_available, "mamba fast path unavailable — kernels hub fetch failed?"
print(f"geometry: heads={cfg.mamba_num_heads} head_dim={cfg.mamba_head_dim} "
      f"groups={cfg.n_groups} state={cfg.ssm_state_size} hidden={cfg.hidden_size}")


def run(path, x, mem_eff=False, force_contiguous=False, fp32_kernel=False):
    """One fwd+bwd through the mixer; returns (out, x.grad, param_grads, err)."""
    mixer.use_mem_eff_path = mem_eff
    mixer.train()
    mixer.zero_grad(set_to_none=True)
    for p in mixer.parameters():
        p.requires_grad_(True)
    xi = x.clone().requires_grad_(True)

    patches = []
    if force_contiguous:
        # wrap the module-global kernel fns to force-contiguous every tensor arg
        def wrap(fn):
            def inner(*args, **kw):
                args = [a.contiguous() if torch.is_tensor(a) else a for a in args]
                kw = {k: (v.contiguous() if torch.is_tensor(v) else v) for k, v in kw.items()}
                return fn(*args, **kw)
            return inner
        for name in ("mamba_chunk_scan_combined", "causal_conv1d_fn"):
            patches.append((name, getattr(M, name)))
            setattr(M, name, wrap(getattr(M, name)))
    if fp32_kernel:
        # THE TRAINER'S FIX (nemotron_h._patch_mamba_kernel_fp32): upcast the kernel inputs
        # to fp32 (its bf16 backward NaNs in the ddt/dA path; fp32 is exact) + downcast out.
        orig = M.mamba_chunk_scan_combined
        def fp32_wrapped(x_, dt, A, B_, C, **kw):
            outs = orig(x_.float(), dt.float(), A.float(), B_.float(), C.float(), **kw)
            if isinstance(outs, tuple):
                return tuple(o.to(x_.dtype) if torch.is_tensor(o) and o.is_floating_point() else o
                             for o in outs)
            return outs.to(x_.dtype)
        patches.append(("mamba_chunk_scan_combined", orig))
        M.mamba_chunk_scan_combined = fp32_wrapped
    try:
        if path == "cuda":
            out = mixer.cuda_kernels_forward(xi, cache_params=None, attention_mask=None)
        else:
            out = mixer.torch_forward(xi, cache_params=None, attention_mask=None)
        out.float().pow(2).mean().backward()
        pgrads = {n: (p.grad.detach().clone() if p.grad is not None else None)
                  for n, p in mixer.named_parameters()}
        return out.detach(), xi.grad.detach().clone(), pgrads, None
    except Exception as e:  # noqa: BLE001
        return None, None, None, f"{type(e).__name__}: {str(e)[:140]}"
    finally:
        for name, orig in patches:
            setattr(M, name, orig)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


failures = 0
for L in SEQ_LENS:
    x = torch.randn(1, L, cfg.hidden_size, device=dev, dtype=DTYPE) * 0.5
    ref_out, ref_g, ref_pg, err = run("torch", x)
    assert err is None, f"torch reference failed: {err}"
    print(f"\n== L={L} · torch reference: |out|={ref_out.float().norm():.3f} "
          f"grad_finite={bool(torch.isfinite(ref_g).all())}")

    # Gating: the "+fp32k" variant is what the gateway trainer ships
    # (_patch_mamba_kernel_fp32) and MUST pass. The raw-bf16 variants are KNOWN upstream
    # failures (hub kernel bf16 backward NaN — informational only). fp32 mode gates all.
    for tag, gating, kw in (
            ("cuda non-split", DTYPE != torch.bfloat16, dict(mem_eff=False)),
            ("cuda non-split +contig", DTYPE != torch.bfloat16, dict(mem_eff=False, force_contiguous=True)),
            ("cuda non-split +fp32k", True, dict(mem_eff=False, fp32_kernel=True)),
            ("cuda mem-eff (split)", False, dict(mem_eff=True))):
        out, g, pg, err = run("cuda", x, **kw)
        if err is not None:
            print(f"  {tag:26s}: {'FAIL' if gating else 'EXPECTED-FAIL'} — {err}")
            failures += 1 if gating else 0
            continue
        fc = cos(out, ref_out)
        out_fin = bool(torch.isfinite(out).all())
        g_fin = bool(torch.isfinite(g).all())
        gc = cos(g, ref_g) if g_fin else float("nan")
        bad_pg = [n for n, v in pg.items() if v is not None and not torch.isfinite(v).all()]
        # per-param grad cosine for the ones that exist in both
        pgc = {n: round(cos(pg[n], ref_pg[n]), 4) for n in pg
               if pg[n] is not None and ref_pg[n] is not None and torch.isfinite(pg[n]).all()}
        ok = fc >= 0.999 and out_fin and g_fin and not bad_pg and gc >= 0.99
        status = "PASS" if ok else ("KNOWN-FAIL (upstream bf16 bwd)" if not gating else "FAIL")
        print(f"  {tag:26s}: {status} — fwd_cos={fc:.6f} out_finite={out_fin} "
              f"xgrad_finite={g_fin} xgrad_cos={gc:.4f} nan_param_grads={bad_pg or 'none'}")
        if not ok and gating:
            failures += 1
            print(f"    param grad cosines: {pgc}")

print(f"\n{'ALL MAMBA-KERNEL CHECKS PASS ✅' if failures == 0 else f'{failures} FAILURE(S) ❌'}")
raise SystemExit(1 if failures else 0)
