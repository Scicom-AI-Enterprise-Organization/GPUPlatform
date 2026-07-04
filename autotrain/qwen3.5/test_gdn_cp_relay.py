"""Module-level GatedDeltaNet CP validation: a REAL Qwen3_5GatedDeltaNet layer, full sequence vs
2-way contiguous split with BOTH state relays (conv initial/final_states + delta-rule
initial_state/final_state). If output + input-grad match, the CP state-passing is correct — and
whether the grads match tells us if the relays are auto-differentiable.

Uses real Qwen3.6-27B GDN dims; random weights; no 52GB checkpoint."""
import os, torch
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

torch.manual_seed(0)
dev = "cuda"; dt = torch.bfloat16
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-27B").get_text_config()
H = cfg.hidden_size
S = 256; half = S // 2                      # single doc; both halves multiples of chunk 64

layer = M.Qwen3_5GatedDeltaNet(cfg, layer_idx=0).to(dev).to(dt)
for p in layer.parameters():
    p.requires_grad_(False)                 # only test input grad (activations), weights frozen

hs_full = (torch.randn(1, S, H, device=dev, dtype=dt) * 0.2).requires_grad_(True)
do = torch.randn(1, S, H, device=dev, dtype=dt) * 0.2

# ---- relay closure: the wrappers read/write these between the two split forwards ----
relay = {"conv_in": None, "conv_final": None, "rec_in": None, "rec_final": None, "on": False}
_real_conv = layer.causal_conv1d_fn
_real_gdr = layer.chunk_gated_delta_rule

def conv_wrap(x, weight=None, bias=None, seq_idx=None, initial_states=None,
              return_final_states=False, final_states_out=None, activation=None):
    if not relay["on"]:
        return _real_conv(x, weight, bias, seq_idx=seq_idx, activation=activation)
    out, fin = _real_conv(x, weight, bias, seq_idx=seq_idx, activation=activation,
                          initial_states=relay["conv_in"], return_final_states=True)
    relay["conv_final"] = fin
    return out

def gdr_wrap(q, k, v, g=None, beta=None, initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=False, cu_seqlens=None):
    if not relay["on"]:
        return _real_gdr(q, k, v, g=g, beta=beta, initial_state=initial_state,
                         output_final_state=output_final_state,
                         use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens)
    o, fs = _real_gdr(q, k, v, g=g, beta=beta, initial_state=relay["rec_in"],
                      output_final_state=True, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                      cu_seqlens=cu_seqlens)
    relay["rec_final"] = fs
    return o, fs

layer.causal_conv1d_fn = conv_wrap
layer.chunk_gated_delta_rule = gdr_wrap

def run(hs, s):
    # NOTE: no seq_idx — causal_conv1d_fn forbids return_final_states when seq_idx is set. For the
    # single-doc long-context CP case we don't need per-doc conv resets, so drop it and use the
    # native conv initial_states/return_final_states relay. (Multi-doc-within-chunk would need a
    # manual conv halo instead.)
    cu = torch.tensor([0, s], dtype=torch.int32, device=dev)
    r = layer(hs, cu_seq_lens_q=cu)
    return r[0] if isinstance(r, tuple) else r

# ---- FULL ----
relay["on"] = False
out_full = run(hs_full, S)
out_full.backward(do)
g_full = hs_full.grad.clone()

# ---- SPLIT with relay ----
hs2 = hs_full.detach().clone().requires_grad_(True)
relay["on"] = True
relay["conv_in"] = relay["rec_in"] = None
o0 = run(hs2[:, :half], half)
relay["conv_in"], relay["rec_in"] = relay["conv_final"], relay["rec_final"]   # relay across the split
o1 = run(hs2[:, half:], half)
out_split = torch.cat([o0, o1], dim=1)
out_split.backward(do)
g_split = hs2.grad.clone()

def rep(name, a, b):
    a, b = a.float(), b.float()
    rel = (a - b).norm().item() / (b.norm().item() or 1.0)
    print(f"[{'PASS' if rel < 1e-2 else 'FAIL'}] {name}: rel={rel:.3e} max_abs={(a-b).abs().max().item():.3e}")

rep("out (fwd relay)", out_split, out_full)
rep("d_hidden (bwd relay)", g_split, g_full)
