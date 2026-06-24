# bench_vllm_shim

A minimal `vllm` package so `bench_attention.py` can run vLLM's **Triton unified attention**
kernel (`unified_attention`) without building all of vLLM. It contains:

- **Tiny stand-ins** (committed) for the handful of vllm internals the kernel imports, each
  faithful to upstream defaults: `envs.VLLM_BATCH_INVARIANT` (off), `logger.init_logger`,
  `platforms.current_platform` (`fp8_dtype`, `is_device_capability_family`), `triton_utils`
  (re-exports `triton`/`tl`), and `v1.kv_cache_interface.KVQuantMode`.
- **The real kernel** — `v1/attention/ops/triton_unified_attention.py` +
  `triton_attention_helpers.py`, **copied verbatim** from a vLLM checkout by `bench_setup.sh`
  (gitignored here — they are vLLM's source, pinned to whatever checkout you copy from).

So the numbers come from the *exact* kernel code in your vLLM tree; only the ~5 supporting
modules are shimmed, none of which affects the attention math on an H20 (SM 9.0).

Populate the kernel files:  `VLLM_REPO=/home/husein/ssd3/vllm bash ../bench_setup.sh`
