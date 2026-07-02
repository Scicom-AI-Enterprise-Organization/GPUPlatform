"""Fold a trained MiniMax-M2 LoRA into a plain **bf16/fp16** MiniMaxM2ForCausalLM for vLLM serving.

Vendored from `autotrain/minimax-m2/merge_to_bf16.py` (keep in sync). The gateway edit vs the
standalone: a `--dtype {fp16,bf16}` flag (default **fp16** — inference on the merged model, per the
platform's "we don't infer in bf16" preference) and a `MODEL_ID` env override.

vLLM 0.23.0 supports `MiniMaxM2ForCausalLM`, but can't apply our custom fused routed-expert LoRA as
an adapter, so we fold everything into the weights and save a standard checkpoint (no
quantization_config), then vLLM serves it.

  * attention q/k/v/o (LinearLoRA): W <- dequant_blockwise(W_fp8) + scaling·(B@A)
  * routed experts (FP8Experts + our LoRA): per-expert gu[e]/dn[e] <- dequant + moe_scaling·(B[e]@A[e])
  * everything else (router gate, norms, embeddings, lm_head) -> merged dtype copy

MiniMax-M2: 128x128 BLOCK-scaled FP8 (dequantize_fp8_blockwise), 256 experts, NO shared experts,
NO vision tower (text-only MiniMaxM2ForCausalLM). Merged fp16 ≈ 460GB on disk.

    python merge_to_bf16.py --lora checkpointing/lora.pt --out /share/merged-minimax --dtype fp16

Run this from its own dir (the flat `from lora import` / `import merge_infer` need it on sys.path).
"""
import argparse
import copy
import os

import torch

from lora import LinearLoRA, dequantize_fp8_blockwise
import merge_infer

MODEL_ID = os.environ.get("MODEL_ID", "MiniMaxAI/MiniMax-M2")

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "float16": torch.float16, "bfloat16": torch.bfloat16}


@torch.no_grad()
def build_base_state_dict(model, out_dtype=torch.float16):
    """Dequant the FROZEN FP8 base to `out_dtype` with NO LoRA (for serving the BASE model on vLLM,
    which chokes on MiniMax-M2's block-FP8 MoE dims). Mirror of build_merged_state_dict minus the fold."""
    merged = {}
    handled = []
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if isinstance(w, torch.Tensor) and w.element_size() == 1 and getattr(module, "weight_scale_inv", None) is not None:
            merged[f"{name}.weight"] = dequantize_fp8_blockwise(
                w, module.weight_scale_inv, getattr(module, "block_size", None), out_dtype=out_dtype)
            if getattr(module, "bias", None) is not None:
                merged[f"{name}.bias"] = module.bias.to(out_dtype)
            handled.append(name + ".")
        elif getattr(module, "gate_up_proj", None) is not None and module.gate_up_proj.element_size() == 1:
            bs = getattr(module, "block_size", None)
            merged[f"{name}.gate_up_proj"] = dequantize_fp8_blockwise(
                module.gate_up_proj, module.gate_up_proj_scale_inv, bs, out_dtype=out_dtype)
            merged[f"{name}.down_proj"] = dequantize_fp8_blockwise(
                module.down_proj, module.down_proj_scale_inv, bs, out_dtype=out_dtype)
            handled.append(name + ".")
    for pname, p in model.named_parameters():
        if pname in merged or any(pname.startswith(h) for h in handled):
            continue
        merged[pname] = p.detach().to(out_dtype)
    for bname, b in model.named_buffers():
        if bname in merged or any(bname.startswith(h) for h in handled) or "rotary" in bname or "inv_freq" in bname:
            continue
        merged[bname] = b.detach()
    return merged


@torch.no_grad()
def build_merged_state_dict(model, out_dtype=torch.float16):
    merged = {}
    handled = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLoRA):
            base = module.base
            if module._fp8:
                w = dequantize_fp8_blockwise(base.weight, base.weight_scale_inv, module._block_size,
                                             out_dtype=torch.float32)
            else:
                w = base.weight.to(torch.float32)
            delta = module.lora_b.weight.to(torch.float32) @ module.lora_a.weight.to(torch.float32)
            merged[f"{name}.weight"] = (w + module.scaling * delta).to(out_dtype)
            if getattr(base, "bias", None) is not None:
                merged[f"{name}.bias"] = base.bias.to(out_dtype)
            handled.append(name + ".")
        elif hasattr(module, "gate_up_lora_a"):
            bs = getattr(module, "block_size", None)
            gu = (dequantize_fp8_blockwise(module.gate_up_proj, module.gate_up_proj_scale_inv, bs, out_dtype=torch.float32)
                  if module.gate_up_proj.element_size() == 1 else module.gate_up_proj.to(torch.float32))
            dn = (dequantize_fp8_blockwise(module.down_proj, module.down_proj_scale_inv, bs, out_dtype=torch.float32)
                  if module.down_proj.element_size() == 1 else module.down_proj.to(torch.float32))
            sc = module.lora_scaling
            for e in range(module.num_experts):
                gu[e] += sc * (module.gate_up_lora_b[e].float() @ module.gate_up_lora_a[e].float())
                dn[e] += sc * (module.down_lora_b[e].float() @ module.down_lora_a[e].float())
            merged[f"{name}.gate_up_proj"] = gu.to(out_dtype)
            merged[f"{name}.down_proj"] = dn.to(out_dtype)
            handled.append(name + ".")
    for pname, p in model.named_parameters():
        if pname in merged or any(pname.startswith(h) for h in handled):
            continue
        merged[pname] = p.detach().to(out_dtype)
    # persistent buffers too (e.g. the MoE router's e_score_correction_bias) — named_parameters
    # misses these, but the clean model expects them in its state_dict.
    for bname, b in model.named_buffers():
        if bname in merged or any(bname.startswith(h) for h in handled):
            continue
        if "rotary" in bname or "inv_freq" in bname:
            continue  # recomputed at init
        merged[bname] = b.detach()  # preserve original dtype (e.g. fp32 router correction bias)
    return merged


def _write_base_config(model_id: str, out: str) -> bool:
    """Overwrite the merged dir's config.json with the BASE repo's own config.json
    (fp8 quantization_config stripped). The merged model is the base arch minus fp8, so
    the base config is exactly right and preserves every field save_pretrained may drop.
    Returns False if the base config.json isn't in the local HF cache."""
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
    # Drop auto_map: it points at trust-remote-code .py (configuration_minimax_m2.py /
    # modeling_minimax_m2.py) that we don't ship in the merged dir — vLLM has a NATIVE
    # MiniMaxM2 impl, so strip it to force the native path (else vLLM tries to load the
    # missing custom module and OSErrors).
    cfg.pop("auto_map", None)
    with open(os.path.join(out, "config.json"), "w") as f:
        _j.dump(cfg, f, indent=2)
    return True


def _clean_config(config):
    cfg = copy.deepcopy(config)
    if hasattr(cfg, "quantization_config"):
        try:
            delattr(cfg, "quantization_config")
        except Exception:
            cfg.quantization_config = None
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="checkpointing/lora.pt")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--out", default="/share/merged-minimax")
    ap.add_argument("--dtype", default="fp16", choices=list(_DTYPES), help="merged model dtype (default fp16)")
    ap.add_argument("--no-lora", action="store_true", help="just dequant the FP8 base (no LoRA fold) — for serving the BASE model")
    args = ap.parse_args()

    out_dtype = _DTYPES[args.dtype]
    from transformers import AutoConfig, AutoTokenizer, MiniMaxM2ForCausalLM

    model_id = args.model_id or MODEL_ID
    config = AutoConfig.from_pretrained(model_id)

    print(f">> loading FP8 base {model_id} on CPU", flush=True)
    model = MiniMaxM2ForCausalLM.from_pretrained(
        model_id, config=config, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    if args.no_lora:
        print(f">> dequant base FP8 -> {args.dtype} (no LoRA)", flush=True)
        merged = build_base_state_dict(model, out_dtype=out_dtype)
    else:
        merge_infer.attach_lora(model, args.lora, merge_infer.load_meta(args.lora))
        print(f">> folding LoRA -> {args.dtype} state_dict", flush=True)
        merged = build_merged_state_dict(model, out_dtype=out_dtype)
    del model

    print(f">> building clean {args.dtype} model + loading {len(merged)} tensors", flush=True)
    cfg2 = _clean_config(config)
    with torch.device("meta"):
        model2 = MiniMaxM2ForCausalLM(cfg2)
    missing, unexpected = model2.load_state_dict(merged, strict=False, assign=True)
    missing = [m for m in missing if "rotary_emb.inv_freq" not in m and "rotary" not in m]
    assert not missing, f"merged sd missing {len(missing)} keys, e.g. {missing[:5]}"
    if unexpected:
        print(f">> WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")
    del merged

    print(f">> saving merged {args.dtype} model -> {args.out}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    model2.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True).save_pretrained(args.out)
    if _write_base_config(model_id, args.out):
        print(">> wrote base config.json (quant stripped) for vLLM", flush=True)
    print(">> done.", flush=True)


if __name__ == "__main__":
    main()
