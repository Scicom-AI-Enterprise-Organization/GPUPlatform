"""Assert that gemma-4 + custom `dynamic_attention` gives the same logits as the default attention.

  (1) load gemma with attn_implementation="dynamic_attention", feed the CUSTOM packed-dataset input
      (same dict the trainer's collator builds), get last-token logits.
  (2) load gemma with the default attn, feed the plain input, get last-token logits.
  (3) compare + assert (next-token argmax must match; logits ~equal up to bf16 noise).

For one document spanning the whole prompt, dynamic_attention's block-diagonal causal mask reduces
to ordinary causal attention, so the two MUST agree.

    python compare_logits.py
"""
import os
import sys

if not os.environ.get("HF_TOKEN"):
    try:
        for line in open("/home/husein/ssd3/GPUPlatform/gateway/.env"):
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("'\"")
                break
    except FileNotFoundError:
        pass

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration, AttentionInterface
from attention import dynamic_attention

AttentionInterface.register("dynamic_attention", dynamic_attention)

MODEL = os.environ.get("MODEL_ID", "google/gemma-4-31B-it")
TEXT = "hello world, how are you today?"

tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok(TEXT)["input_ids"]          # list[int], includes <bos>
L = len(ids)
input_ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)   # (1, L)

# === the CUSTOM dataset input — exactly the dict gemma4.collator produces, single document ===
custom_batch = {
    "input_ids": input_ids_t,
    "position_ids": torch.arange(L, dtype=torch.long).unsqueeze(0),
    "attention_mask": None,
    "mm_token_type_ids": torch.zeros_like(input_ids_t),
    "cu_seq_lens_q": torch.tensor([0, L], dtype=torch.int32),     # one doc -> [0, L]
    "cu_seq_lens_k": torch.tensor([0, L], dtype=torch.int32),
    "max_length_q": L,
    "max_length_k": L,
}
print(f"prompt={TEXT!r}  L={L} tokens", flush=True)


def to_dev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def load(attn_impl):
    kw = dict(dtype=torch.bfloat16, device_map="cuda")
    if attn_impl is not None:
        kw["attn_implementation"] = attn_impl
    m = Gemma4ForConditionalGeneration.from_pretrained(MODEL, **kw)
    return m.eval()

# (1) dynamic_attention + custom packed input
print(">> [1] dynamic_attention + custom dataset input", flush=True)
m = load("dynamic_attention")
b = to_dev(custom_batch, m.device)
with torch.no_grad():
    logit_dyn = m(**b, use_cache=False).logits[0, -1].float().cpu()
del m
torch.cuda.empty_cache()

# (2) default attention + plain input
print(">> [2] default attention + plain input", flush=True)
m = load(None)
with torch.no_grad():
    logit_def = m(input_ids=input_ids_t.to(m.device), use_cache=False).logits[0, -1].float().cpu()
del m
torch.cuda.empty_cache()

# (3) compare + assert
am_dyn, am_def = logit_dyn.argmax().item(), logit_def.argmax().item()
max_abs = (logit_dyn - logit_def).abs().max().item()
cos = F.cosine_similarity(logit_dyn, logit_def, dim=0).item()
print("\n================ result ================")
print(f"argmax  dynamic={am_dyn} ({tok.decode([am_dyn])!r})   default={am_def} ({tok.decode([am_def])!r})")
print(f"max_abs_diff={max_abs:.4f}   cosine={cos:.6f}")
print(f"top5 dynamic={logit_dyn.topk(5).indices.tolist()}")
print(f"top5 default={logit_def.topk(5).indices.tolist()}")

assert am_dyn == am_def, f"ARGMAX MISMATCH: dynamic={am_dyn} default={am_def}"
assert cos > 0.99, f"COSINE TOO LOW: {cos}"   # bf16 (FA3 sliding + SDPA-math full) noise
print("\nPASS ✅  dynamic_attention logits match the default attention")
