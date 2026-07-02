"""Fold a trained Mistral-Small-4 LoRA into a plain **bf16/fp16** Mistral3 checkpoint for vLLM serving.

Vendored from `autotrain/mistral-small/merge_to_bf16.py` (keep in sync). Gateway edit vs the
standalone: a `--dtype {fp16,bf16}` flag (default **fp16**) + a `MODEL_ID` env override.

`merge_infer.py` only *attaches* the adapter (FP8 base + on-the-fly dequant); transformers' FP8
generate path is broken for this model, so for real generation we serve with **vLLM**, which needs
the weights on disk. vLLM also can't apply our custom fused routed-expert LoRA as an adapter, so we
**fold everything into the weights**:

  * MLA q_a/q_b/kv_a/kv_b/o + shared MLP (LinearLoRA): W <- dequant(W_fp8) + scaling·(B@A)
  * routed experts (FP8Experts + our LoRA): per-expert gu[e]/dn[e] <- dequant + moe_scaling·(B[e]@A[e])
  * everything else (router gate, norms, embeddings, lm_head, vision tower) -> merged-dtype copy

Standard `Mistral3ForConditionalGeneration` with no quantization_config; vLLM serves it. Merged
fp16 ≈ 238GB on disk.

    python merge_to_bf16.py --lora checkpointing/lora.pt --out /share/merged-mistral --dtype fp16
    MERGE_SELFTEST=1 python merge_to_bf16.py      # tiny-model correctness check (no real weights)

Run this from its own dir (the flat `from lora import` / `import merge_infer` need it on sys.path).
"""
import argparse
import os

import torch
import torch.nn as nn

from lora import LinearLoRA, dequantize_fp8
import merge_infer

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-Small-4-119B-2603")

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "float16": torch.float16, "bfloat16": torch.bfloat16}


@torch.no_grad()
def build_merged_state_dict(model, attn_scaling, moe_scaling, out_dtype=torch.bfloat16):
    """Walk the LoRA-attached model and produce a clean state_dict (in `out_dtype`) keyed by the
    *unwrapped* (vanilla Mistral3) FQNs. LinearLoRA -> `<path>.weight`; expert LoRA ->
    `<path>.gate_up_proj` / `.down_proj`; every other param is copied as `out_dtype`."""
    merged = {}
    handled_prefixes = []

    for name, module in model.named_modules():
        if isinstance(module, LinearLoRA):
            base = module.base
            if module._fp8:
                w = dequantize_fp8(base.weight, base.weight_scale_inv, module._block_size,
                                   out_dtype=torch.float32)
            else:
                w = base.weight.to(torch.float32)
            delta = module.lora_b.weight.to(torch.float32) @ module.lora_a.weight.to(torch.float32)
            merged[f"{name}.weight"] = (w + module.scaling * delta).to(out_dtype)
            if getattr(base, "bias", None) is not None:
                merged[f"{name}.bias"] = base.bias.to(out_dtype)
            handled_prefixes.append(name + ".")

        elif hasattr(module, "gate_up_lora_a"):  # routed experts carrying our LoRA
            bs = getattr(module, "block_size", None)
            gu = (dequantize_fp8(module.gate_up_proj, module.gate_up_proj_scale_inv, bs, out_dtype=torch.float32)
                  if module.gate_up_proj.element_size() == 1 else module.gate_up_proj.to(torch.float32))
            dn = (dequantize_fp8(module.down_proj, module.down_proj_scale_inv, bs, out_dtype=torch.float32)
                  if module.down_proj.element_size() == 1 else module.down_proj.to(torch.float32))
            sc = module.lora_scaling
            for e in range(module.num_experts):
                gu[e] += sc * (module.gate_up_lora_b[e].float() @ module.gate_up_lora_a[e].float())
                dn[e] += sc * (module.down_lora_b[e].float() @ module.down_lora_a[e].float())
            merged[f"{name}.gate_up_proj"] = gu.to(out_dtype)
            merged[f"{name}.down_proj"] = dn.to(out_dtype)
            handled_prefixes.append(name + ".")

    for pname, p in model.named_parameters():
        if pname in merged:
            continue
        if any(pname.startswith(h) for h in handled_prefixes):
            continue
        merged[pname] = p.detach().to(out_dtype)
    return merged


def _write_base_config(model_id: str, out: str) -> bool:
    """Overwrite the merged dir's config.json with the BASE repo's own config.json,
    stripped of fp8 quantization_config. save_pretrained re-serialises the composite
    Mistral3 config and DROPS text_config.architectures → vLLM's inner-LM resolver
    raises 'No model architectures are specified'. The merged model is the base arch
    minus fp8, so the base config (minus quant) is exactly right. Returns False if the
    base config.json isn't in the local HF cache (caller keeps save_pretrained's copy)."""
    import glob
    import json as _j
    repo = "models--" + model_id.replace("/", "--")
    roots = [os.environ.get("HF_HOME", ""), "/share/huggingface",
             os.path.expanduser("~/.cache/huggingface"), "/root/.cache/huggingface"]
    src = None
    for r in roots:
        if not r:
            continue
        hits = glob.glob(os.path.join(r, "hub", repo, "snapshots", "*", "config.json"))
        if hits:
            src = hits[0]
            break
    if not src:
        return False
    with open(src) as f:
        cfg = _j.load(f)
    cfg.pop("quantization_config", None)
    cfg.pop("auto_map", None)  # force vLLM's native class, not missing trust-remote-code .py
    tc = cfg.get("text_config")
    if isinstance(tc, dict):
        tc.pop("quantization_config", None)
        tc.pop("auto_map", None)
        tc.pop("num_experts", None)  # the transformers-5.5.0 load-bug shim; not a real field
    with open(os.path.join(out, "config.json"), "w") as f:
        _j.dump(cfg, f, indent=2)
    return True


def _clean_config(config):
    """Drop the fp8 quantization_config so the saved model is a plain (bf16/fp16) checkpoint."""
    import copy
    cfg = copy.deepcopy(config)
    for attr in ("quantization_config",):
        if hasattr(cfg, attr):
            try:
                delattr(cfg, attr)
            except Exception:
                setattr(cfg, attr, None)
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_experts"):
        try:
            delattr(cfg.text_config, "num_experts")
        except Exception:
            pass
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="checkpointing/lora.pt")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--out", default="/share/merged-mistral")
    ap.add_argument("--dtype", default="fp16", choices=list(_DTYPES), help="merged model dtype (default fp16)")
    args = ap.parse_args()

    out_dtype = _DTYPES[args.dtype]
    from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

    model_id = args.model_id or MODEL_ID
    meta = merge_infer.load_meta(args.lora)
    attn_scaling = meta.get("attn_scaling", meta.get("attn_alpha", 16.0) / meta.get("attn_r", 16))
    moe_scaling = meta.get("moe_scaling", meta.get("moe_alpha", 16.0) / meta.get("moe_r", 16))
    config = merge_infer.load_patched_config(model_id)

    print(f">> loading FP8 base {model_id} on CPU", flush=True)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id, config=config, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    merge_infer.attach_lora(model, args.lora, meta)

    print(f">> folding LoRA -> {args.dtype} state_dict", flush=True)
    merged = build_merged_state_dict(model, attn_scaling, moe_scaling, out_dtype=out_dtype)
    del model

    print(f">> building clean {args.dtype} model + loading {len(merged)} tensors", flush=True)
    cfg2 = _clean_config(config)
    with torch.device("meta"):
        model2 = Mistral3ForConditionalGeneration(cfg2)
    missing, unexpected = model2.load_state_dict(merged, strict=False, assign=True)
    missing = [m for m in missing if "rotary_emb.inv_freq" not in m]  # buffers, recomputed
    assert not missing, f"merged sd missing {len(missing)} keys, e.g. {missing[:5]}"
    if unexpected:
        print(f">> WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")
    del merged

    print(f">> saving merged {args.dtype} model -> {args.out}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    model2.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(args.out)
    # Preserve the base composite config (keeps text_config.architectures for vLLM's
    # inner-LM resolver); save_pretrained drops it.
    if _write_base_config(model_id, args.out):
        print(">> wrote base config.json (quant stripped) for vLLM", flush=True)
    print(">> done.", flush=True)


# ---------------------------------------------------------------------------
def _selftest():
    """Validate the fold math on a tiny REAL Mistral3 (bf16, no fp8): merged dense model must
    reproduce the LoRA-attached model's logits."""
    import torch.nn.functional as F  # noqa
    from transformers import Mistral3ForConditionalGeneration
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
    from lora import apply_mistral_lora
    os.environ.setdefault("MISTRAL_GROUPED_FALLBACK", "1")
    text = dict(model_type="mistral4", vocab_size=128, hidden_size=64, intermediate_size=128,
        moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        n_shared_experts=1, n_routed_experts=8, num_experts_per_tok=2, kv_lora_rank=16, q_lora_rank=32,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16, n_group=1, topk_group=1,
        first_k_dense_replace=0, max_position_embeddings=4096)
    vision = dict(model_type="pixtral", hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, head_dim=8, image_size=28, patch_size=14, num_channels=3)
    cfg = Mistral3Config(text_config=text, vision_config=vision, tie_word_embeddings=False)
    torch.manual_seed(0)
    m = Mistral3ForConditionalGeneration(cfg).to(torch.float32).eval()
    m.config.get_text_config()._experts_implementation = "eager"
    m.config._attn_implementation = m.model.language_model.config._attn_implementation = "eager"
    stats = apply_mistral_lora(m, attn_r=4, attn_alpha=8.0, moe_r=4, moe_alpha=8.0, lora_dtype=torch.float32)
    with torch.no_grad():  # non-zero adapter so the merge is non-trivial
        for n, p in m.named_parameters():
            if n.endswith("lora_b.weight") or n.endswith("lora_b"):
                p.normal_(std=0.05)
    ids = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        ref = m(input_ids=ids, use_cache=False).logits

    merged = build_merged_state_dict(m, 8.0 / 4, 8.0 / 4)
    cfg2 = _clean_config(cfg)
    m2 = Mistral3ForConditionalGeneration(cfg2).to(torch.float32).eval()
    m2.config.get_text_config()._experts_implementation = "eager"
    m2.config._attn_implementation = m2.model.language_model.config._attn_implementation = "eager"
    miss, unexp = m2.load_state_dict({k: v.float() for k, v in merged.items()}, strict=False, assign=True)
    miss = [x for x in miss if "rotary_emb.inv_freq" not in x]
    assert not miss, f"missing {miss[:5]}"
    with torch.no_grad():
        got = m2(input_ids=ids, use_cache=False).logits
    d = (ref - got).abs().max().item()
    print(f"[selftest] attn/moe wrapped={stats['attn_modules_wrapped']}/{stats['moe_blocks_adapted']} "
          f"unexpected_keys={len(unexp)} | max|attached - merged| = {d:.3e}")
    assert d < 8e-3, f"merge mismatch {d}"
    print("[selftest] PASS — folded model reproduces the LoRA-attached logits (weight-rounding precision)")


if __name__ == "__main__":
    if os.environ.get("MERGE_SELFTEST") == "1":
        _selftest()
    else:
        main()
