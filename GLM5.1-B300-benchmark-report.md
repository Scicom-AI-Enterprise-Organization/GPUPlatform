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
- **No model change:** FP8, TP=8, **no expert‑parallel**, **`--kv-cache-dtype fp8`** → ~21.2k tok/s (+31% over EP‑off; the biggest flag‑only win).
- **Absolute fastest:** **`nvidia/GLM-5.1-NVFP4`**, TP=8, no expert‑parallel (it already uses fp8 KV) → ~23.7k tok/s.
- Both: prefix‑cache off; operate around conc 200.
- **Short‑form accuracy holds ([section 6](#6-accuracy--does-lower-precision-cost-quality)):** GSM8K and MMLU stay flat (~87–89%) across FP8 → fp8‑KV → NVFP4 — no precision loss on plain reasoning, and fp8 KV is clean.
- **⚠️ NVFP4 breaks function‑calling ([section 6b](#6b-functioncalling--nvfp4-breaks-tool-use-fp8-kv-doesnt)):** fresh, isolated per‑config runs now confirm it — FP8 and FP8+fp8‑KV both score **tool‑call F1 0.97** (recall ~1.0, calls every tool it should), but **NVFP4 collapses to F1 0.20** (recall **0.12** — misses **88% of required tool calls**, and ~6× slower as it rambles instead of calling tools). **fp8 KV is innocent** (0.968 ≈ 0.970) — the cause is the **FP4 weights**. Short‑form accuracy ([section 6](#6-accuracy--does-lower-precision-cost-quality)) does *not* predict this. **Don't use NVFP4 for agentic / tool‑calling workloads** (use FP8 + fp8 KV).

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

**fp8 KV cache is the biggest flag‑only win — but the mechanism is KV *capacity*, not faster compute.** The 16‑bit KV cache holds 1,608,064 tokens (max concurrency **7.93×**) and at conc 200 with 8192‑token prompts it runs **97–99% full** → requests queue and preempt (188 running / 10 waiting) → throughput caps at the ~16k plateau. fp8 KV holds **2,904,960 tokens (14.33×)**, runs only **~56% full**, and keeps ~197 requests truly in flight (≈0 waiting) → +31%. Implication: this gain is **concurrency/context‑dependent** — at low concurrency (KV not full) fp8 KV gives ~nothing; it pays off exactly when you're KV‑bound. It logs `Using standard fp8 KV cache format` (the DeepSeek `fp8_ds_mla` format would need `--attention-backend FLASHMLA_SPARSE`, which was *slower* in [section 4](#4-attention-backend--the-prefill-is-hardwired-to-fa2)). fp8 KV is **lossy** → validate accuracy. MTP speculative decoding **regresses** here: at 256 output tokens / conc 200 the decode is a small slice and the spec‑verify overhead isn't repaid.

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

## 6. Accuracy — does lower precision cost quality?

Measured quality on three evals via the platform's **accuracy benchmark** — greedy, reasoning on, served with the *exact same args* as the speed runs, **one fresh isolated run per config** (`glm51-bench-*`): **GSM8K** (200 q), **multilingual MMLU** (`openai/MMMLU`, 200 q across FR/DE/ES/ZH/JA), and the hard multi‑turn **Function‑Call‑TaaS** (20 agentic tool‑calling conversations).

| config | GSM8K | MMLU (5‑lang) | **Func‑Call F1** | throughput ([section 3](#3-concurrency--throughput-plateaus-computebound)/[section 5](#5-nvfp4--the-big-win-47-and-why-its-not-2)) |
|---|---|---|---|---|
| FP8, 16‑bit KV (baseline) | 88.0% | 88.0% | **0.970** | 16,152 tok/s |
| **FP8 + `--kv-cache-dtype fp8`** | 89.5% | 87.5% | **0.968** | 21,211 tok/s (+31%) |
| **`nvidia/GLM-5.1-NVFP4`** (fp8 KV, auto) | 87.5% | 86.5% | **0.204** ⬇️ | 23,729 tok/s (+47%) |

**Short‑form reasoning is robust to precision; multi‑turn tool‑calling is not.** GSM8K and MMLU stay flat (~±1.5 pts, within noise) across all three — fp8 KV and even FP4 weights cost nothing there. But the agentic function‑calling eval tells a completely different story: **NVFP4 collapses from F1 0.97 → 0.20**, while fp8 KV holds (0.968 ≈ 0.970). So the real IQ‑vs‑speed tradeoff only surfaces on a hard agentic task — a short‑form eval alone would have **missed it entirely**. Tool‑call breakdown in **[section 6b](#6b-functioncalling--nvfp4-breaks-tool-use-fp8-kv-doesnt)**. (200 samples ⇒ ±~3 pts on GSM8K/MMLU.)

---

## 6b. Function‑calling — NVFP4 breaks tool use (fp8 KV doesn't)

`Function-Call-TaaS` (`test-basic`, 20 multi‑turn conversations, ~255 reference tool calls) replays each conversation turn‑by‑turn and scores the model's tool calls against the reference. Fresh, **isolated per‑config** runs (`glm51-bench-*`):

| config | tool‑call **F1** | precision | **recall** | calls made / needed | eval wall‑time (20 conv) |
|---|---|---|---|---|---|
| FP8, 16‑bit KV | **0.970** | 0.941 | **1.00** | 422 / 255 | ~25 min |
| FP8 + fp8 KV | **0.968** | 0.941 | 0.996 | 412 / 255 | ~22 min |
| **NVFP4** (fp8 KV, auto) | **0.204** | 0.633 | **0.122** | **83 / 255** | **~147 min** |

**NVFP4 misses ~88% of the tool calls it should make** (recall 0.12 — 224 of 255 missed). The few calls it *does* make are well‑formed (valid JSON, correct arg types), so it's **not a format break** — NVFP4 simply **fails to invoke tools when it should**, generating prose instead. That rambling also makes it **~6× slower** on the same 20 conversations (it generates toward the 16k‑token cap each turn instead of emitting a short tool call).

**The cause is the FP4 weights, not fp8 KV.** FP8 + fp8 KV (0.968) is statistically identical to plain FP8 (0.970) — **fp8 KV is innocent**; it preserves tool‑use just as it preserves GSM8K/MMLU. Only swapping to NVFP4 weights triggers the collapse. And it's **invisible to short‑form evals**: NVFP4's GSM8K/MMLU sit at a normal ~87% ([section 6](#6-accuracy--does-lower-precision-cost-quality)), but its agentic tool‑use is broken.

> **Where's "NVFP4 + fp8 KV"?** The NVFP4 row above **is** it. The `modelopt_fp4` checkpoint **auto‑enables fp8 KV** (`kv_cache_dtype=fp8_e4m3` in the engine config — logged even without passing `--kv-cache-dtype fp8`), so NVFP4 always runs with fp8 KV; there is **no NVFP4 + 16‑bit‑KV** cell (you'd have to force fp8 KV *off*, and it wouldn't matter — the regression is the weights, not the KV dtype). A separate explicit‑`--kv-cache-dtype fp8` NVFP4 run (N=50) corroborates: **F1 0.13, recall 0.075**.

**Implication:** NVFP4's +47% throughput is real, but **do not use NVFP4 for agentic / tool‑calling / function‑calling workloads** — use **FP8 + `--kv-cache-dtype fp8`** (+31% throughput, tool‑use fully intact). NVFP4 is fine for plain text generation where tools aren't involved.

*(An earlier batch reported a bogus identical 0.941 for every config: a benchmark checkpoint/output‑path collision keyed only by dataset index made later configs **resume / re‑read the first config's results** — fc_eval checkpoints by model name, and FP8‑epoff and FP8‑kvfp8 share a model name. Fixed by keying scratch paths per config + failing loudly on a missing output; the table above is from the clean re‑run.)*

---

## Recommendations

1. **Fastest — but text‑only — `nvidia/GLM-5.1-NVFP4`, `--tensor-parallel-size 8`, NO `--enable-expert-parallel`** (~23.7–23.9k tok/s, +47% over FP8 EP‑off; short‑form GSM8K/MMLU held — [section 6](#6-accuracy--does-lower-precision-cost-quality)). **⚠️ Do NOT use NVFP4 for tool‑calling / agentic workloads:** fresh isolated runs ([section 6b](#6b-functioncalling--nvfp4-breaks-tool-use-fp8-kv-doesnt)) show **tool‑call F1 0.20 vs FP8's 0.97** — recall 0.12, it misses 88% of required tool calls and runs ~6× slower. Use NVFP4 only for plain text generation.
2. **Best all‑round — FP8, drop `--enable-expert-parallel`, add `--kv-cache-dtype fp8`** → ~21.2k tok/s (+31% over EP‑off; nearly NVFP4's level from two flags). **No quality cost anywhere:** GSM8K/MMLU flat ([section 6](#6-accuracy--does-lower-precision-cost-quality)) *and* tool‑calling fully intact (F1 0.968 ≈ FP8's 0.970, [section 6b](#6b-functioncalling--nvfp4-breaks-tool-use-fp8-kv-doesnt)). This is the config to use for agentic / tool workloads.
3. **Don't bother** with `--async-scheduling` (default‑on), `--max-num-seqs`, `--gpu-memory-utilization`, MoE‑backend overrides, MTP (−8% here), or raising `--max-num-batched-tokens` (−4%).
4. **To break the ~21k FP8 / ~24k NVFP4 ceiling** you need the **prefill attention off FA2** — a future vLLM with FA4 head_size=256 support on Blackwell (or a Blackwell FlashMLA *prefill* kernel). It's a clear upstream kernel gap for `head_size=256` MLA models.
5. Concurrency: operate around **conc 200** (throughput sweet spot); higher just inflates TTFT.

---

## Appendix

**Speed runs (storage `husein`):** baseline (EP on) · clean single‑change compare · EP‑off knob sweep · concurrency sweep · attention‑backend hunt · 0.23.1 nightly · NVFP4 · kv‑fp8/MTP (the conc‑200 fp8‑KV win).

**Accuracy runs ([section 6](#6-accuracy--does-lower-precision-cost-quality)):** first measured all three configs in one combined run, then re‑run as three **separate** records (one per config) for 1:1 mapping to the speed configs. Driven by the platform's new **accuracy benchmark** (Benchmark → New → type **Accuracy**): serves the model, scores GSM8K + `openai/MMMLU`, reports accuracy + a decode tok/s.

**Clean per‑config function‑call runs ([section 6](#6-accuracy--does-lower-precision-cost-quality)/[section 6b](#6b-functioncalling--nvfp4-breaks-tool-use-fp8-kv-doesnt) — the trustworthy set):** one fresh isolated run per config (FP8 · FP8+fp8‑KV · NVFP4) — each GSM8K + MMLU + `Function-Call-TaaS` (20 conv), with fresh isolated scratch paths after the checkpoint‑collision fix. **These are the numbers in [section 6](#6-accuracy--does-lower-precision-cost-quality)/[section 6b](#6b-functioncalling--nvfp4-breaks-tool-use-fp8-kv-doesnt).**

**Earlier (buggy) accuracy runs, superseded:** a combined run plus an "all" batch whose byte‑identical 0.94 function‑call values were a checkpoint/output‑path collision (since fixed). The per‑eval `metrics` bag (tool‑call P/R/F1, name/type accuracy, json‑valid, hallucination, etc.) now renders in full on the **Results** tab for any eval that reports it; the single‑run **IQ‑vs‑speed** scatter was moved out of the per‑run Results tab to the multi‑select **compare** view on `/benchmark`.

**Notes:** the 729 GB FP8 + 438 GB NVFP4 caches live on the VM at `/share/huggingface`; runs used `cleanup_model=false` to preserve them. NVFP4 is Xet‑native — download needs Xet **enabled** + an HF token (disabling Xet stalls it).
