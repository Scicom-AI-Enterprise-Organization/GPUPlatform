# GLM‑5.1 serving benchmark — 8× NVIDIA B300

**Model:** `zai-org/GLM-5.1-FP8` (a large MoE with DeepSeek‑style Sparse Attention, arch `GlmMoeDsaForCausalLM`) and the NVFP4 variant `nvidia/GLM-5.1-NVFP4`.
**Hardware:** 8× NVIDIA B300 (Blackwell Ultra, sm_103, ~275 GB HBM each), single node, TP=8.
**Engine:** vLLM 0.23.0 (+ a `0.23.1rc1` nightly probe).
**Workload (fixed unless noted):** input 8192, output 256, 200 prompts, `--ignore-eos`, `request_rate=inf`, **prefix caching OFF** (clean numbers), per‑run warm‑up.
**Driver:** the platform benchmark API (benchmaq over SSH to the `fpt` VM provider), results in the `husein` S3 storage.

---

## TL;DR — two real wins, one hard ceiling

| change | throughput | vs prior |
|---|---|---|
| Baseline — user's exact command (`--enable-expert-parallel`, EP **on**) | 13,118 tok/s @ conc 50 | — |
| Drop `--enable-expert-parallel` (TP‑only) | 14,159 tok/s @ conc 50 | +8% |
| EP‑off, **`--kv-cache-dtype fp8`** | **21,211 tok/s @ conc 200** | **+31%** vs EP‑off @ conc 200 |
| **`nvidia/GLM-5.1-NVFP4` (FP4 weights, auto fp8‑KV)** | **23,729 tok/s @ conc 200** | **+47%** vs FP8 EP‑off @ conc 200 |

**Two recommendations:**
- **No model change:** FP8, TP=8, **no expert‑parallel**, **`--kv-cache-dtype fp8`** → ~21.2k tok/s (+31% over EP‑off; the biggest flag‑only win). Validate accuracy (fp8 KV is lossy).
- **Absolute fastest:** **`nvidia/GLM-5.1-NVFP4`**, TP=8, no expert‑parallel (it already uses fp8 KV) → ~23.7k tok/s. Validate output quality (experimental ModelOpt quant).
- Both: prefix‑cache off; operate around conc 200.

**The remaining ceiling is the prefill attention.** This is a *prefill‑bound* workload (8192‑token inputs, ~256 decode tokens), and vLLM runs the MLA **prefill** on `FLASH_ATTN MLA`, which on Blackwell **falls back to FlashAttention‑2** (`FA4 does not support head_size=256 due to TMEM capacity limits`). That FA2 prefill is unchanged by attention‑backend flags, by weight precision (FP8/FP4), and by the `0.23.1` nightly. It's a current vLLM kernel gap, not a config you can flip.

---

## 1. Expert parallelism — OFF wins (+8%)

GLM‑5.1 uses DeepSeek Sparse Attention (DSA/MLA). vLLM's own guidance for DSA models is plain TP, because expert‑parallel has been buggy/slower for them; on a single NVLink node EP's all‑to‑all can cost more than the TP all‑reduce it replaces.

| config (conc 50, prefix‑cache off) | total tok/s | output tok/s | mean TTFT |
|---|---|---|---|
| baseline, EP on (user's command) | 13,118 | 397.5 | 5.06 s |
| **EP off (TP‑only)** | **14,159** | 429.1 | 4.85 s |

`--moe-backend flashinfer_cutlass` **crashed** (`FP8 MoE backend FLASHINFER_CUTLASS does not support … block‑scaled FP8`) — incompatible with GLM's FP8 quant scheme. The auto‑selected MoE backend (`flashinfer_trtllm` + DeepGEMM) is already optimal.

## 2. Per‑knob results

**2a. Plain knobs (EP‑off, conc 50) — no‑ops, one regression:**

| variant | total tok/s | Δ vs EP‑off ref (14,081) |
|---|---|---|
| reference | 14,081 | — |
| `--async-scheduling` | 14,078 | 0% (**already default‑on**) |
| `--max-num-seqs 128` | 14,119 | +0.3% |
| `--gpu-memory-utilization 0.95` | 14,080 | 0% |
| `--max-num-batched-tokens 16384` | 13,455 | **−4%** (regression) |

`async-scheduling` is auto‑enabled by vLLM on Blackwell (confirmed in the no‑flag baseline's log), so passing it explicitly is a no‑op.

**2b. fp8 KV cache — the surprise win (EP‑off, conc 200):**

| variant | total tok/s | output tok/s | Δ vs EP‑off (16,152) |
|---|---|---|---|
| **`--kv-cache-dtype fp8`** | **21,211** | 642.7 | **+31%** |
| `--speculative-config mtp` (1 token, async off) | 14,875 | 450.8 | **−8%** |

**fp8 KV cache is the biggest flag‑only win — but the mechanism is KV *capacity*, not faster compute.** The 16‑bit KV cache holds 1,608,064 tokens (max concurrency **7.93×**) and at conc 200 with 8192‑token prompts it runs **97–99% full** → requests queue and preempt (188 running / 10 waiting) → throughput caps at the ~16k plateau. fp8 KV holds **2,904,960 tokens (14.33×)**, runs only **~56% full**, and keeps ~197 requests truly in flight (≈0 waiting) → +31%. Implication: this gain is **concurrency/context‑dependent** — at low concurrency (KV not full) fp8 KV gives ~nothing; it pays off exactly when you're KV‑bound. It logs `Using standard fp8 KV cache format` (the DeepSeek `fp8_ds_mla` format would need `--attention-backend FLASHMLA_SPARSE`, which was *slower* in §4). fp8 KV is **lossy** → validate accuracy. MTP speculative decoding **regresses** here: at 256 output tokens / conc 200 the decode is a small slice and the spec‑verify overhead isn't repaid.

## 3. Concurrency — throughput plateaus (~compute‑bound)

| concurrency (EP‑off) | total tok/s | mean TTFT |
|---|---|---|
| 50 | 14,032 | 7.2 s |
| 100 | 15,462 | 13.5 s |
| 200 | 16,139 | 27.6 s |
| 400 | 16,240 | **105 s** |
| 400 + `max-num-batched-tokens=32768` | 16,568 | 105 s |

Throughput rises only ~15% from conc 50→400, then flattens at **~16k tok/s**; conc 400 just queues requests into 100 s+ TTFT. The 24% KV‑cache utilization at conc 50 was misleading — the bottleneck is **prefill compute**, not KV/concurrency. The throughput "sweet spot" is around **conc 200** (16.1k tok/s, TTFT 27 s).

## 4. Attention backend — the prefill is hardwired to FA2

| backend (EP‑off, conc 200) | total tok/s | note |
|---|---|---|
| default (auto → `FLASHINFER_MLA_SPARSE` decode) | 16,152 | what it already picks |
| `FLASHINFER_MLA_SPARSE` (forced) | 16,135 | ≈ default |
| `FLASHMLA_SPARSE` (forced) | 14,450 | slower |
| `FLASHMLA`, `CUTLASS_MLA` | ❌ crash | `sparse not supported` — DSA needs a sparse backend |

Every surviving variant logs `Using FLASH_ATTN MLA prefill backend` → `FA4 … head_size=256 → FA2`. **`--attention-backend` only controls the *decode* backend** (already optimal); the **prefill** is separately fixed to FLASH_ATTN MLA. So the FA2 prefill cannot be changed by flags in 0.23.0.

**Nightly probe (`0.23.1rc1.dev38`):** loaded and showed the **same** `FLASH_ATTN MLA → FA2` — the kernel gap isn't fixed in the newer build either.

## 5. NVFP4 — the big win (+47%), and why it's not 2×

| metric (EP‑off, conc 200) | FP8 | **NVFP4** | Δ |
|---|---|---|---|
| total token throughput | 16,152 | **23,729** | **+47%** |
| output token throughput | ~490 | **719** | **+47%** |
| mean TTFT | 27.6 s | 17.8 s | −36% |

`nvidia/GLM-5.1-NVFP4` (`modelopt_fp4`, `FLASHINFER_TRTLLM` NvFp4 MoE, fp8 KV auto). FP4 weight GEMMs run at ~2× FP8 FLOPS on B300, so the MoE/FFN gets much faster — **but the attention prefill still computes in bf16 and still falls back to FA2** (`FLASH_ATTN MLA → head_size=256 → FA2`). Because the workload is prefill‑bound, the net is **+47%, not the full 2×** — the FA2 attention is the remaining limiter. (This is the answer to "what about fp16?": go *down* to FP4, not up to fp16 — fp16 would be ~2× slower on Blackwell and wouldn't touch the attention.)

---

## Recommendations

1. **Absolute fastest — `nvidia/GLM-5.1-NVFP4`, `--tensor-parallel-size 8`, NO `--enable-expert-parallel`** (~23.7k tok/s, +47% over FP8 EP‑off). It already uses fp8 KV. Validate output quality (experimental ModelOpt quant).
2. **No model change — FP8, drop `--enable-expert-parallel`, add `--kv-cache-dtype fp8`** → ~21.2k tok/s (+31% over EP‑off; nearly NVFP4's level from two flags). fp8 KV is lossy — validate accuracy.
3. **Don't bother** with `--async-scheduling` (default‑on), `--max-num-seqs`, `--gpu-memory-utilization`, MoE‑backend overrides, MTP (−8% here), or raising `--max-num-batched-tokens` (−4%).
4. **To break the ~21k FP8 / ~24k NVFP4 ceiling** you need the **prefill attention off FA2** — a future vLLM with FA4 head_size=256 support on Blackwell (or a Blackwell FlashMLA *prefill* kernel). It's a clear upstream kernel gap for `head_size=256` MLA models.
5. Concurrency: operate around **conc 200** (throughput sweet spot); higher just inflates TTFT.

---

## Appendix

**Benchmark runs (platform IDs, storage `husein`):**
`bench-c64df065` baseline · `bench-f82059c0` clean single‑change compare · `bench-5f05063d` EP‑off knob sweep · `bench-64d0b993` concurrency sweep · `bench-cb52d132` attention‑backend hunt · `bench-db4406b3` nightly · `bench-6247e3b7` **NVFP4** · `bench-61f7ff46` **kv‑fp8/MTP** (the conc‑200 fp8‑KV win; two earlier attempts `bench-13f2ed3a`/`bench-4b47023f` were orphaned by gateway restarts).

**Notes:** the 729 GB FP8 + 438 GB NVFP4 caches live on the VM at `/share/huggingface`; runs used `cleanup_model=false` to preserve them. NVFP4 is Xet‑native — download needs Xet **enabled** + an HF token (disabling Xet stalls it).
