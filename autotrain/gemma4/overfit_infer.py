"""End-to-end overfit sanity check: can the merged model reproduce its FIRST training row?

After deliberately overfitting the dataset, this:
  1. loads the base model + merges the trained LoRA (via merge_infer.merge_lora_),
  2. takes row `--row` of the packed dataset (the SAME truncation used in training),
  3. feeds the first `--prefix-tokens` tokens of that row's first document,
  4. greedily generates and compares to the memorized continuation.

A properly overfit model reproduces its training data, so the next-token match should be high —
which exercises the whole chain: attention masking -> training -> checkpoint -> merge -> inference.

    python overfit_infer.py --row 0 --prefix-tokens 64 --gen-tokens 256 --max-seq-len 2048
"""
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from chinidataset import StreamingDataset

from merge_infer import load_meta, merge_lora_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="checkpointing/lora.pt")
    ap.add_argument("--data", default="./packed_data")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--max-seq-len", type=int, default=2048, help="match the training truncation")
    ap.add_argument("--prefix-tokens", type=int, default=64)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--scaling", type=float, default=None)
    args = ap.parse_args()

    meta = load_meta(args.lora)
    model_id = args.model_id or meta.get("model_id", "google/gemma-4-31B-it")
    scaling = args.scaling if args.scaling is not None else meta.get("scaling")
    if scaling is None:
        raise SystemExit("scaling unknown: pass --scaling alpha/r")

    # The first document of row `--row`, truncated exactly like training.
    ds = StreamingDataset(local=args.data)
    row = ds[args.row]
    ids = np.asarray(row["input_ids"])[: args.max_seq_len]
    am = np.asarray(row["attention_mask"])
    doc0 = min(int(am[0]) if len(am) else len(ids), len(ids))
    doc_ids = ids[:doc0].astype(np.int64)
    print(f">> row {args.row}: truncated_len={len(ids)} first_doc_len={doc0}")

    tok = AutoTokenizer.from_pretrained(model_id)
    print(f">> loading base {model_id} + merging LoRA")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
    )
    model.eval()
    merge_lora_(model, args.lora, scaling)

    k = max(1, min(args.prefix_tokens, doc0 - 1))
    n_gen = min(args.gen_tokens, doc0 - k)
    prefix = torch.tensor(doc_ids[:k], dtype=torch.long)[None].to(model.device)
    target = doc_ids[k : k + n_gen]

    print(f">> greedily generating {n_gen} tokens from a {k}-token prefix")
    out = model.generate(prefix, max_new_tokens=n_gen, do_sample=False)
    gen = out[0][k : k + n_gen].cpu().numpy().astype(np.int64)

    m = min(len(gen), len(target))
    match = float((gen[:m] == target[:m]).mean()) if m else 0.0
    print(f"\n===== greedy next-token reproduction of row {args.row}: {match*100:.1f}% ({m} tokens) =====")
    print("\n----- prompt (decoded tail of the prefix) -----")
    print(tok.decode(doc_ids[max(0, k - 48):k], skip_special_tokens=False))
    print("\n----- GENERATED (merged overfit model) -----")
    print(tok.decode(gen, skip_special_tokens=False))
    print("\n----- GROUND TRUTH (the training row) -----")
    print(tok.decode(target[:m], skip_special_tokens=False))


if __name__ == "__main__":
    main()
