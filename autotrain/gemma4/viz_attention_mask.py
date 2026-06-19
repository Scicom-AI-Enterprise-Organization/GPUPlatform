"""Render the packed attention mask that dynamic_attention builds, to a PNG — so you can eyeball
that it makes sense.

It uses the SAME construction as attention.dynamic_attention's SDPA branch (block_diagonal_concat
of per-document lower-triangular blocks, as a boolean mask). The FA3 branch doesn't materialize a
mask — it gets the identical effect from cu_seqlens — so this image represents both paths.

    python viz_attention_mask.py --doc-lens 6,4,5 --out attention_mask.png
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attention import block_diagonal_concat


def packed_block_diag_causal(doc_lens):
    # identical to dynamic_attention's SDPA branch
    masks = [torch.tril(torch.ones(int(s), int(s))) for s in doc_lens]
    return block_diagonal_concat(*masks).bool()[0].numpy()  # (S, S), True = attend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-lens", default="6,4,5", help="comma-sep document lengths in one packed bin")
    ap.add_argument("--out", default="attention_mask.png")
    args = ap.parse_args()

    doc_lens = [int(x) for x in args.doc_lens.split(",")]
    S = sum(doc_lens)
    bounds = np.cumsum([0] + doc_lens)

    packed = packed_block_diag_causal(doc_lens)              # what dynamic_attention uses
    plain = torch.tril(torch.ones(S, S)).bool().numpy()      # plain causal (no doc blocking), for contrast
    float_noop = np.ones((S, S))                             # a 0/1 FLOAT mask is additive bias -> everyone attends

    panels = [
        (packed, f"packed block-diagonal causal  (docs={doc_lens})\nBOOL mask used by dynamic_attention — correct"),
        (plain, f"plain causal over {S} tokens (single doc)\nfor contrast: would leak across doc boundaries"),
        (float_noop, "what a 0/1 FLOAT mask actually does\n(additive bias -> nothing masked: the bug)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (mat, title) in zip(axes, panels):
        ax.imshow(mat, cmap="Greys", interpolation="nearest", origin="upper", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("key position (attended to)")
        ax.set_ylabel("query position")
        for b in bounds[1:-1]:
            ax.axhline(b - 0.5, color="red", lw=0.8, ls="--")
            ax.axvline(b - 0.5, color="red", lw=0.8, ls="--")
        ax.set_xticks(bounds)
        ax.set_yticks(bounds)
    fig.suptitle("Black = attend (True) · White = masked (False) · red dashes = document boundaries", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")

    # sanity: the packed mask must be causal within each doc and never cross a boundary
    ok = True
    for a, b in zip(bounds[:-1], bounds[1:]):
        block = packed[a:b, a:b]
        if not np.array_equal(block, np.tril(np.ones_like(block))):
            ok = False
    cross = packed.copy()
    for a, b in zip(bounds[:-1], bounds[1:]):
        cross[a:b, a:b] = 0
    leaks = cross.sum()
    print(f"saved {args.out}  (S={S}, docs={doc_lens})")
    print(f"per-document causal: {'OK' if ok else 'BROKEN'} | cross-document attention entries: {int(leaks)} (must be 0)")


if __name__ == "__main__":
    main()
