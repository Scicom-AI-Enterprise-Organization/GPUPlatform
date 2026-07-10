# autotrain — Nemotron-H correctness gates (kernel fast-path verification)

**This dir is the VERIFICATION TOOLING for the Nemotron-H trainer** (which lives vendored at
`gateway/gateway/training/llm/nemotron/nemotron_h.py` — unlike the other archs there is no
standalone trainer here yet, only the gates). Nemotron-3-Nano-30B-A3B = hybrid **23 Mamba2 +
23 MoE + 6 attention** layers, bf16, untied. Per the repo rule ("always compare logits before
an expensive run"), run these BEFORE any training that changes the attention/mamba backend:

```
test_mamba_kernels.py      mixer-level gate: torch_forward vs the kernels-hub fast path
                           (fwd cosine + input/param grad finiteness & cosine), real 30B
                           geometry, single GPU. TEST_DTYPE=bf16|fp32, TEST_SEQ_LENS=...
test_attention_kernels.py  attention-level gate: sdpa vs hub FA3 — fwd cosine + per-projection
                           q/k/v/o weight-grad finiteness & cosine. (NOT eager as reference:
                           with attention_mask=None eager applies NO causal mask.)
compare_logits.py          REAL-30B gate: sdpa+torch reference vs FA3+mamba-kernels+fp32-patch
                           candidate — next-token argmax, top-5 overlap, logits cosine.
```

Run on the tm-2 box (`/share/autotrain-nemotron`, venv `/share/autotrain-llm-nemotron`,
model cached in `/share/huggingface`):

```bash
export HF_HOME=/share/huggingface HF_HUB_DISABLE_XET=1
CUDA_VISIBLE_DEVICES=0 /share/autotrain-llm-nemotron/bin/python test_mamba_kernels.py
CUDA_VISIBLE_DEVICES=0 /share/autotrain-llm-nemotron/bin/python test_attention_kernels.py
CUDA_VISIBLE_DEVICES=0 /share/autotrain-llm-nemotron/bin/python compare_logits.py
```

## Findings (2026-07-10, tm-2 H20, torch 2.10 cu130, transformers 5.12.1)

The fast path = hub FA3 attention (`kernels-community/flash-attn3`) + hub Mamba2 Triton
kernels (`kernels-community/mamba-ssm` snapshot `a39ff24`, `causal-conv1d`), both fetched by
`lazy_load_kernel`/`kernels` at model load — no source build. What the gates found when the
first fast-path training run went **NaN from step 1** (step-0 forward finite → the step-0
BACKWARD poisoned the LoRA weights at the first optimizer step):

- **FA3 attention: exact.** fwd cos ≥ 0.999998 vs sdpa; **q/k/v/o weight grads all finite,
  cosine 1.0**. Confirmed independently by a 10-step training bisect (FA3 + torch-mamba run
  tracked the sdpa baseline loss to bf16 noise).
- **Hub mamba-ssm `mamba_chunk_scan_combined` bf16 BACKWARD is broken.** Forward is fine
  (cos 0.999988). In bf16 the backward produces **NaN grads in exactly the ddt/dA path**:
  `dt_bias`, `A_log`, and dx→conv→`in_proj` go NaN while `D`/`norm`/`out_proj` grads stay
  perfect (cos 1.0). NOT a stride/contiguity issue (forcing `.contiguous()` on every kernel
  arg changes nothing). **fp32 through the same kernel is exact** (fwd + all grads cos 1.0).
- **Fix (shipped in the gateway trainer, `_patch_mamba_kernel_fp32`):** wrap
  `mamba_chunk_scan_combined` to upcast (x, dt, A, B, C) → fp32 and downcast the output.
  Verified: bf16 module + fp32 kernel → all grads finite, xgrad cos 0.9999 vs the torch
  reference. Still far cheaper than the torch SSD fallback (fused Triton scan vs
  autograd-materialized chunk einsums).
- **The mem-eff split kernel (`mamba_split_conv1d_scan_combined`) cannot run here at all**:
  the hub build asserts `causal_conv1d_cuda is not available` (it wants the *compiled*
  causal-conv1d package ext, not the hub kernel). Irrelevant in practice — the gateway
  trainer forces `use_mem_eff_path=False` anyway on LoRA-wrapped mixers (the split kernel
  fuses the RAW `out_proj.weight`, silently bypassing an out_proj LoRA adapter).
- The hub repo's newer `main` revision changed its build layout (`module has no attribute
  'ops'`) and can't be dropped in via the transformers `lazy_load_kernel` path — dead end.
- `LinearLoRA` wrappers need `weight`/`bias` properties (delegating to the wrapped Linear):
  the mamba dispatch reads `self.in_proj.weight.device.type` at every forward.

**65k long-context exercise (2026-07-10, synthetic 8× 65,536-token one-doc bins, real 30B, 4× H20
`/share/autotrain-nemotron-65k` recipe):**
- **Fast path (FA3 + mamba kernels + fp32 patch): 3 steps finite** (12.89→12.82→12.74 ≈ ln V for
  random tokens), **~11,000 tok/s steady-state (~24 s/step), peak 58.6 GB/GPU** of 143 — headroom
  for ~2× longer context or batch_size 2.
- **Torch fallback (sdpa + SSD path): CANNOT RUN 65k — instant OOM trying to allocate a single
  256 GiB tensor** on the first forward (the torch scan materializes a chunked intermediate).
  At 65k the kernels aren't an optimization, they're the difference between training and not.

**Training evidence (10-step runs, 4× H20, real 30B, one-doc-per-bin pack `ds-a077af13`):**
baseline sdpa+torch `train-d0efd849` clean ~500–690 tok/s; first fast-path run
`train-6a672e67` NaN@step1 (~2600–5200 tok/s — bf16 kernel, wrong); bisect `train-9d5d1448`
(mamba-kernels only) NaN@step1 / `train-423d8e6c` (FA3 only) clean → kernels' backward guilty;
**fixed run `train-c7957876` (FA3 + mamba kernels + fp32 patch): ALL 10 STEPS FINITE**
(1.543, 1.447, 1.204, 1.967, 0.871, 4.872, 1.832, 4.943, 3.580, 4.442 — same per-bin shape as
the baseline), checkpoint saved, **~720–1250 tok/s steady-state ≈ 1.45× the baseline at these
short (~750-tok) docs**. `compare_logits.py` on the real 30B: **PASSED** — logits cosine
0.999929, argmax match, top-5 identical. ⚠ The structural wins (FA3's O(S) unpadded attention
vs sdpa's O(S²) padding mask; the fused scan vs the torch path's autograd-materialized chunk
einsums) grow with context — the ~1.45× here is the SHORT-doc floor where per-step overhead
dominates; long-context (16k bins) throughput/memory not yet measured.
