"""Real-model correctness gate for the Nemotron-H fast path — the compare-logits
discipline from autotrain/CLAUDE.md ("a wrong-but-runnable forward silently trains
garbage"). Run BEFORE any training run that changes the attention / mamba backend.

Loads the REAL nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 twice on one GPU (30B bf16
≈ 60GB — fits one 143GB H20; loads sequentially, frees in between):

  reference : attn=sdpa, use_mamba_kernels=False (the torch path that trained cleanly)
  candidate : attn=$NEMOTRON_ATTN (default kernels-community/flash-attn3),
              use_mamba_kernels=True + the trainer's fp32-upcast kernel patch

Asserts on a chat-formatted prompt: next-token argmax match, top-5 overlap, logits
cosine ≥ 0.99. Usage (tm-2):
  HF_HOME=/share/huggingface CUDA_VISIBLE_DEVICES=0 python compare_logits.py
"""
import os
os.environ.setdefault("HF_HOME", "/share/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import gc
import torch
from transformers import AutoConfig, AutoTokenizer, NemotronHForCausalLM
from transformers.models.nemotron_h import modeling_nemotron_h as M

MODEL_ID = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
CAND_ATTN = os.environ.get("NEMOTRON_ATTN", "kernels-community/flash-attn3")

PROMPT = [{"role": "user", "content": "Explain in one sentence why the sky is blue."}]


def patch_fp32():
    """The gateway trainer's fix (nemotron_h._patch_mamba_kernel_fp32): the hub kernel's
    bf16 backward NaNs; fp32 is exact. Applied here so the candidate == what trains."""
    orig = M.mamba_chunk_scan_combined
    if orig is None:
        return False
    def w(x, dt, A, B, C, **kw):
        outs = orig(x.float(), dt.float(), A.float(), B.float(), C.float(), **kw)
        if isinstance(outs, tuple):
            return tuple(o.to(x.dtype) if torch.is_tensor(o) and o.is_floating_point() else o for o in outs)
        return outs.to(x.dtype)
    M.mamba_chunk_scan_combined = w
    return True


def logits_for(attn, kernels, tag):
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    cfg.use_mamba_kernels = kernels
    print(f"[{tag}] loading (attn={attn}, mamba_kernels={kernels}) …", flush=True)
    model = NemotronHForCausalLM.from_pretrained(
        MODEL_ID, config=cfg, dtype=torch.bfloat16, attn_implementation=attn,
    ).to("cuda").eval()
    if kernels:
        assert M.is_fast_path_available, "kernels requested but fast path unavailable"
        assert patch_fp32(), "fp32 kernel patch failed"
        print(f"[{tag}] fast path ON + fp32 kernel patch applied", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = tok.apply_chat_template(PROMPT, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].to("cuda")
    with torch.no_grad():
        out = model(input_ids=ids).logits[0, -1].float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out, tok


ref, tok = logits_for("sdpa", False, "reference")
cand, _ = logits_for(CAND_ATTN, True, "candidate")

cos = torch.nn.functional.cosine_similarity(ref, cand, dim=0).item()
r5 = ref.topk(5).indices.tolist()
c5 = cand.topk(5).indices.tolist()
argmax_match = r5[0] == c5[0]
overlap = len(set(r5) & set(c5))
print(f"\nlogits cosine       : {cos:.6f}")
print(f"next-token argmax   : ref={tok.decode([r5[0]])!r} cand={tok.decode([c5[0]])!r} "
      f"match={argmax_match}")
print(f"top-5 overlap       : {overlap}/5  (ref={r5} cand={c5})")
ok = cos >= 0.99 and argmax_match and overlap >= 4
print("\nCOMPARE-LOGITS " + ("PASSED ✅" if ok else "FAILED ❌"))
raise SystemExit(0 if ok else 1)
