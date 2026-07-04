"""2-rank distributed GatedDeltaNet CP: each rank holds a contiguous half of the sequence and
relays the conv-state + recurrent-state across ranks via DIFFERENTIABLE P2P (send in fwd,
recv-grad in bwd). Compares each rank's local out+grad to the non-CP full-layer reference sliced
to that rank's contiguous chunk. Real Qwen3.6-27B GDN layer, random weights.

    torchrun --nproc_per_node=2 test_gdn_cp_dist.py

STATUS: PASSES — out rel 1.4e-4, d_hidden rel 4.9e-4. The two deadlock fixes are baked in here:
(1) ONE combined (conv+recurrent) pair-relay per layer → deterministic backward comm order across
ranks; (2) a grad-requiring `_ANCHOR` input on the recv op → its backward actually fires (else the
receiving rank never sends the state-grads and the sender hangs). These are the same primitives in
context_parallel.py. See CLAUDE.md "Context parallelism".
"""
import os, torch, torch.distributed as dist
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr); dev = torch.device(f"cuda:{lr}")
dist.init_process_group("nccl")
_ANCHOR = torch.zeros((), device=dev, requires_grad=True)   # grad-requiring anchor for recv ops
dt = torch.bfloat16


# ---- differentiable cross-rank state relay (ONE PAIR-op per GDN layer) ----
# The deadlock fix: relay BOTH stateful pieces (conv-state + recurrent-state) of a layer through a
# SINGLE autograd Function with a FIXED internal send/recv order. Two separate relays per layer let
# the autograd engine fire their backward P2P in a rank-dependent order → deadlock. One pair-op per
# layer removes the intra-layer ambiguity; across layers the order is forced by the data dependency,
# so the backward comm order is identical on every rank (deadlock-free). Blocking send/recv are fine
# here because the order is now deterministic.
def _isend(x, dst):
    dist.isend(x.contiguous(), dst).wait()

def _irecv(shape, dtype, src):
    b = torch.empty(shape, dtype=dtype, device=dev); dist.irecv(b, src).wait(); return b

class _SendPairToNext(torch.autograd.Function):
    """fwd: send (a, b) to rank+1 in fixed order; bwd: recv (da, db) in the same order."""
    @staticmethod
    def forward(ctx, a, b, dst):
        ctx.dst = dst
        ctx.sa, ctx.da = a.shape, a.dtype
        ctx.sb, ctx.db = b.shape, b.dtype
        _isend(a, dst); _isend(b, dst)
        return a.new_zeros(())
    @staticmethod
    def backward(ctx, _g):
        ga = _irecv(ctx.sa, ctx.da, ctx.dst)
        gb = _irecv(ctx.sb, ctx.db, ctx.dst)
        return ga, gb, None

class _RecvPairFromPrev(torch.autograd.Function):
    """fwd: recv (a, b) from rank-1 in fixed order; bwd: send (da, db) in the same order.

    Takes a grad-requiring `anchor` input: an autograd.Function whose inputs are all non-tensors has
    outputs that DON'T require grad, so its backward is never called — which meant the receiving rank
    never sent the state-grads back and the sender hung on recv. The anchor forces the outputs to
    require grad so the backward fires."""
    @staticmethod
    def forward(ctx, anchor, sa, da, sb, db, src):
        ctx.src = src
        a = _irecv(sa, da, src)
        b = _irecv(sb, db, src)
        return a, b
    @staticmethod
    def backward(ctx, ga, gb):
        _isend(ga, ctx.src); _isend(gb, ctx.src)
        return None, None, None, None, None, None

# ---- GDN CP wrappers: ONE combined (conv+recurrent) pair-relay per layer ----
# conv_wrap (layer start) RECVs both states; gdr_wrap (layer end) SENDs both. One recv + one send
# per layer, fixed order -> deterministic backward comm -> deadlock-free.
def install_cp(layer):
    _conv = layer.causal_conv1d_fn
    _gdr = layer.chunk_gated_delta_rule
    st = {"rec_init": None, "conv_final": None}   # shared between conv_wrap and gdr_wrap

    def conv_wrap(x, weight=None, bias=None, seq_idx=None, initial_states=None,
                  return_final_states=False, final_states_out=None, activation=None):
        B, D, _ = x.shape
        k1 = layer.conv_kernel_size - 1
        conv_init = st["rec_init"] = None
        if rank > 0:
            # recv (conv_state (B,k1,D), rec_state (1,Hv,K,V)) as ONE pair op.
            conv_buf, st["rec_init"] = _RecvPairFromPrev.apply(
                _ANCHOR, (B, k1, D), x.dtype, layer._cp_rec_shape, layer._cp_rec_dtype, rank - 1)
            conv_init = conv_buf.transpose(1, 2)          # -> (B,D,k1), stride(1)==1
        out, fin = _conv(x, weight, bias, activation=activation,
                         initial_states=conv_init, return_final_states=True)
        st["conv_final"] = fin.transpose(1, 2).contiguous()   # (B,k1,D) for the relay
        return out

    def gdr_wrap(q, k, v, g=None, beta=None, initial_state=None, output_final_state=False,
                 use_qk_l2norm_in_kernel=False, cu_seqlens=None):
        layer._cp_rec_shape = (1, v.shape[2], q.shape[3], v.shape[3])  # (1,Hv,K,V) — for next recv
        o, fs = _gdr(q, k, v, g=g, beta=beta, initial_state=st["rec_init"], output_final_state=True,
                     use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens)
        layer._cp_rec_dtype = fs.dtype
        if rank < W - 1:
            ph = _SendPairToNext.apply(st["conv_final"], fs.contiguous(), rank + 1)
            o = o + ph                # ph==0: connects the send to the graph so its bwd fires
        return o, fs

    # rec-state shape/dtype must be known at conv_wrap (recv) BEFORE gdr runs; seed from config.
    layer._cp_rec_shape = (1, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim)
    layer._cp_rec_dtype = torch.float32
    layer.causal_conv1d_fn = conv_wrap
    layer.chunk_gated_delta_rule = gdr_wrap


# ---- build layer with IDENTICAL weights on all ranks ----
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-27B").get_text_config()
Hn = cfg.hidden_size
S = 256; c = S // W                       # contiguous chunk per rank
torch.manual_seed(0)
layer = M.Qwen3_5GatedDeltaNet(cfg, layer_idx=0).to(dev).to(dt)
for p in layer.parameters():
    p.requires_grad_(False); dist.broadcast(p, src=0)          # same weights everywhere

# identical full input on all ranks
if rank == 0:
    hs_full = torch.randn(1, S, Hn, device=dev, dtype=dt) * 0.2
    do = torch.randn(1, S, Hn, device=dev, dtype=dt) * 0.2
    blob = torch.cat([hs_full.reshape(-1), do.reshape(-1)])
else:
    blob = torch.empty(2 * S * Hn, device=dev, dtype=dt)
dist.broadcast(blob, src=0)
hs_full = blob[:S * Hn].reshape(1, S, Hn)
do = blob[S * Hn:].reshape(1, S, Hn)

def fwd(hs, s):
    cu = torch.tensor([0, s], dtype=torch.int32, device=dev)
    r = layer(hs, cu_seq_lens_q=cu)
    return r[0] if isinstance(r, tuple) else r

# ---- non-CP reference (full, redundant on each rank) ----
hf = hs_full.detach().clone().requires_grad_(True)
out_full = fwd(hf, S)
out_full.backward(do)
ref_out = out_full.detach()[:, rank * c:(rank + 1) * c]
ref_g = hf.grad[:, rank * c:(rank + 1) * c]

# ---- CP (this rank's contiguous chunk + P2P relay) ----
install_cp(layer)
hl = hs_full[:, rank * c:(rank + 1) * c].detach().clone().requires_grad_(True)
out_cp = fwd(hl, c)
out_cp.backward(do[:, rank * c:(rank + 1) * c])

def diff(a, b):
    a, b = a.float(), b.float()
    return (a - b).norm().item() / (b.norm().item() or 1.0)

t = torch.tensor([diff(out_cp.detach(), ref_out), diff(hl.grad, ref_g)], device=dev)
dist.all_reduce(t, op=dist.ReduceOp.MAX)
if rank == 0:
    ok = t[0] < 1e-2 and t[1] < 1e-2
    print(f"[{'PASS' if ok else 'FAIL'}] GDN CP 2-rank: out rel={t[0]:.3e}  d_hidden rel={t[1]:.3e}", flush=True)
dist.destroy_process_group()
