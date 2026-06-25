"""3-way decode comparison: the purpose-built Triton flash-decode kernel (decode_attention) vs
vLLM's general Triton unified_attention vs FA4-cute FlashDecoding. Full hd512 (32q/4kv) and sliding
hd256 (32q/16kv, window 1024), q_len=1, L in {1k..32k}. Reuses bench_attention's runners + the fp32
SDPA correctness anchor. The custom kernel wins everywhere — see CLAUDE.md.

    CUDA_VISIBLE_DEVICES=7 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 python bench_decode_kernel.py
"""
import torch

from bench_attention import (CONFIGS, CTX_LENS, DEVICE, time_call, diff_stats,
                             make_qkv, build_paged_cache, make_triton_runner,
                             make_fa4_decode_runner, sdpa_reference)
from decode_attention import make_decode_runner


def main():
    cap = torch.cuda.get_device_capability(0)
    print(f"device={torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}  (decode, q_len=1)\n")
    hdr = (f"{'cfg':8} {'L':>6} {'custom':>8} {'vllm':>8} {'fa4':>8} | "
           f"{'cus/vllm':>8} {'cus/fa4':>8} {'N':>3} {'cos_vs_sdpa':>11}")
    print(hdr); print("-" * len(hdr))
    for cfg_name, cfg in CONFIGS.items():
        D = cfg["head_dim"]; scale = D ** -0.5
        for L in CTX_LENS:
            q, k, v, q_len, kv_len = make_qkv("decode", L, cfg)
            # custom Triton flash-decode
            cust_fn, N = make_decode_runner(q, k, v, scale, window=cfg["window"])
            out = cust_fn()
            ref = sdpa_reference(q, k, v, q_len, kv_len, cfg, scale)
            cos = torch.nn.functional.cosine_similarity(out.float().flatten(),
                                                        ref.float().flatten(), dim=0).item()
            # vLLM triton (paged)
            kc, vc, bt = build_paged_cache(k, v, kv_len, cfg)
            vllm_fn, _ = make_triton_runner(q, kc, vc, bt, q_len, kv_len, cfg, scale)
            # FA4 cute FlashDecoding (best split)
            Nfa = 8 if L <= 2048 else 16
            fa4_fn, _ = make_fa4_decode_runner(q, k, v, cfg, scale, Nfa)
            ct, vt, ft = time_call(cust_fn, 80, 20), time_call(vllm_fn, 80, 20), time_call(fa4_fn, 80, 20)
            print(f"{cfg_name:8} {L:>6} {ct:>8.4f} {vt:>8.4f} {ft:>8.4f} | "
                  f"{ct/vt:>8.2f} {ct/ft:>8.2f} {N:>3} {cos:>11.5f}")
            del q, k, v, kc, vc, bt, out, ref
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
