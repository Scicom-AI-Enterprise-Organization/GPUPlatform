"""Attach the trained MiniMax-M2 LoRA adapters to the FP8 base and run inference.

Unlike gemma4's `merge_infer.py` (which folds the LoRA delta into a bf16 base weight and can
save a merged model), MiniMax-M2's base is FP8 and ~230GB — materializing a merged bf16 copy
(~460GB) needs far more than one box. So the default here ATTACHES the adapters (no weight
merge): it loads the FP8 base, recreates the LoRA structure from `lora_meta.json`, loads the
trained `lora.pt` into the adapters, and generates through the SAME LoRA-aware forward used in
training (differentiable dequant of the frozen FP8 base + bf16 LoRA). Memory ~= FP8 base +
small bf16 LoRA. This uses the (slower) dequant path rather than the fast inference FP8 kernels
— it's a correctness sanity check, not a serving path.

For production serving: serve the base with vLLM + the LoRA adapter (vLLM supports MiniMax-M2),
or fold + re-quantize to FP8 on a big-RAM box. Both are out of scope for this script.

    python merge_infer.py --prompt "..." --max-new-tokens 128
    # spread the FP8 base across all visible GPUs:
    python merge_infer.py --device-map auto --prompt "..."
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer, MiniMaxM2ForCausalLM

from lora import apply_minimax_lora

# Strip the FSDP / activation-checkpoint wrapper segments the trainer inserts into param names,
# so the saved keys map onto the clean (unwrapped) inference model.
_WRAPPER_SEGMENTS = ("._checkpoint_wrapped_module", "._fsdp_wrapped_module", "._orig_mod")


def _strip_wrappers(name: str) -> str:
    for seg in _WRAPPER_SEGMENTS:
        name = name.replace(seg, "")
    return name


def load_meta(lora_path):
    meta_path = os.path.join(os.path.dirname(lora_path) or ".", "lora_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


@torch.no_grad()
def attach_lora(model, lora_path, meta):
    """Recreate the LoRA structure (zeros) then load the trained adapter tensors into it."""
    stats = apply_minimax_lora(
        model,
        attn_r=meta.get("attn_r", 16), attn_alpha=meta.get("attn_alpha", 16.0),
        moe_r=meta.get("moe_r", 16), moe_alpha=meta.get("moe_alpha", 16.0),
        include_moe=meta.get("moe_lora", True),
        use_dora=meta.get("use_dora", False),
    )
    print(f">> LoRA structure: {stats['attn_modules_wrapped']} attn + "
          f"{stats['moe_blocks_adapted']} MoE blocks ({stats['trainable_params']/1e6:.1f}M params)")

    lora = torch.load(lora_path, map_location="cpu")
    target = dict(model.named_parameters())
    loaded, missing = 0, []
    for k, v in lora.items():
        clean = _strip_wrappers(k)
        if clean in target:
            target[clean].copy_(v.to(target[clean].dtype).to(target[clean].device))
            loaded += 1
        else:
            missing.append(clean)
    print(f">> loaded {loaded}/{len(lora)} adapter tensors")
    if missing:
        print(f">> WARNING: {len(missing)} adapter keys had no match, e.g. {missing[:3]}")
    return loaded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="checkpointing/lora.pt")
    ap.add_argument("--model-id", default=None, help="overrides lora_meta.json / default")
    ap.add_argument("--prompt", default="Write a Python function that reverses a linked list.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device-map", default="auto", help='"auto" spreads the FP8 base across GPUs.')
    ap.add_argument("--attn-impl", default=os.environ.get("MINIMAX_ATTN_IMPL", "flash_attention_3"))
    args = ap.parse_args()

    meta = load_meta(args.lora)
    model_id = args.model_id or meta.get("model_id", "MiniMaxAI/MiniMax-M2")

    print(f">> loading base {model_id} (FP8, device_map={args.device_map})")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = MiniMaxM2ForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation=args.attn_impl, device_map=args.device_map,
    )
    model.eval()

    if os.path.exists(args.lora):
        attach_lora(model, args.lora, meta)
    else:
        print(f">> NOTE: {args.lora} not found — generating with the BASE model (no adapters).")

    msgs = [{"role": "user", "content": args.prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    input_ids = enc["input_ids"].to(model.device)

    print(f">> generating ({args.max_new_tokens} new tokens)")
    out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print("\n===== PROMPT =====\n" + args.prompt)
    print("\n===== OUTPUT =====\n" + text)


if __name__ == "__main__":
    main()
