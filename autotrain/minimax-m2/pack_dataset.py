"""Pack a HuggingFace chat-parquet dataset into the multipacking ChiniDataset
(StreamingDataset) format consumed by `minimax_m2.py`.

This is the MiniMax-M2 sibling of gemma4's `pack_dataset.py` — same bin-packing scheme
(tokenize each conversation via the chat template, greedily pack whole conversations into
`--max-seq-len` token bins), same emitted columns:

  - input_ids     : concatenated token ids of every doc in the bin
  - labels        : same length; -100 masks non-trained tokens (here: full-seq, see below)
  - position_ids  : RESET per document (0..L-1 per packed doc), concatenated
  - attention_mask: the per-document SEQUENCE LENGTHS (NOT a dense mask) -> cu_seqlens

  Invariant: sum(attention_mask) == len(input_ids) == len(position_ids) == len(labels)

Differences from the gemma-4 packer (all because the chat template differs):
  * TOKENIZER defaults to `MiniMaxAI/MiniMax-M2` (GPT2Tokenizer + chat_template.jinja).
  * REASONING: the MiniMax-M2 template renders an assistant's `<think>` reasoning ONLY when
    the turn is AFTER the last user message
    (`if reasoning_content and loop.index0 > ns.last_user_index`). In a multi-user agentic
    trajectory that drops most reasoning. By default we relax that one guard to
    `if reasoning_content` so EVERY assistant turn's reasoning trains. `--native-reasoning`
    keeps the stock behaviour. The template reads `message.reasoning_content`, so we also
    map each assistant turn's `reasoning` field -> `reasoning_content`.
  * LABELS: the MiniMax-M2 template has no `{% generation %}` block, so an assistant-token
    mask is unavailable (all-zero) and we fall back to `labels = input_ids` (train on the
    full packed sequence), detected per-row at runtime — same as gemma-4.
  * TOOLS: the template renders `<tool>{{ tool.function | tojson }}</tool>`, so the bare
    `functions` objects are wrapped as `{"type": "function", "function": <fn>}` and passed
    via `apply_chat_template(tools=...)`. (`--no-tools` to skip.)

MiniMax-M2's attention is O(S) (FlashAttention varlen, uniform head_dim 128), so long bins
are far more feasible than gemma-4's O(S^2) full-attention — `--max-seq-len 131072` packs the
Function-Call conversations (19k-81k tokens) whole. Drop bin size if training is VRAM-bound.
"""
import argparse
import json
import os
import sys

import numpy as np

# --- Read HF_TOKEN from the gateway .env (done BEFORE importing transformers) ----
GATEWAY_ENV = "/home/husein/ssd3/GPUPlatform/gateway/.env"


def _load_hf_token_from_env_file(path: str) -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        os.environ["HF_TOKEN"] = val
                        os.environ["HUGGING_FACE_HUB_TOKEN"] = val
                    return
    except FileNotFoundError:
        print(f"[pack_dataset] WARN: {path} not found; relying on ambient HF_TOKEN", file=sys.stderr)


_load_hf_token_from_env_file(GATEWAY_ENV)

# Xet/hf_transfer have stalled large HF downloads on this box (see MEMORY); disable defensively.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

COLUMNS = {
    "input_ids": "int64[]",
    "labels": "int64[]",
    "position_ids": "uint32[]",
    "attention_mask": "uint32[]",
}
IGNORE_INDEX = -100

DEFAULT_REPO = "Scicom-intl/Function-Call-TaaS"
DEFAULT_FILE = "glm5.1-fp8-test/test-00000-of-00001.parquet"
DEFAULT_TOKENIZER = "MiniMaxAI/MiniMax-M2"

# The MiniMax-M2 chat template's reasoning guard (verified against chat_template.jinja):
#   {%- if reasoning_content and loop.index0 > ns.last_user_index -%}
# Relax it to render EVERY assistant turn's reasoning.
_REASONING_GUARD_STOCK = "if reasoning_content and loop.index0 > ns.last_user_index"
_REASONING_GUARD_ALL = "if reasoning_content"


def build_chat_template(tokenizer, all_reasoning: bool):
    stock = tokenizer.chat_template
    if not all_reasoning:
        return stock
    if _REASONING_GUARD_STOCK not in stock:
        raise KeyError(
            "[pack_dataset] could not find the MiniMax-M2 reasoning guard in the chat "
            "template; it changed upstream. Inspect tokenizer.chat_template and update "
            "_REASONING_GUARD_STOCK (or pass --native-reasoning)."
        )
    return stock.replace(_REASONING_GUARD_STOCK, _REASONING_GUARD_ALL)


def load_source_dataframe(repo: str, filename: str):
    import pandas as pd

    hf_uri = f"hf://datasets/{repo}/{filename}"
    try:
        print(f"[pack_dataset] reading {hf_uri}", flush=True)
        return pd.read_parquet(hf_uri)
    except Exception as e:  # noqa: BLE001
        print(f"[pack_dataset] hf:// read failed ({type(e).__name__}: {e}); falling back to hf_hub_download",
              flush=True)
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        local = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", token=token)
        return pd.read_parquet(local)


def extract_messages(value):
    """Normalize a `messages` cell into list[dict], mapping `reasoning` -> `reasoning_content`
    (the field the MiniMax-M2 template reads) for assistant turns."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return None
    out = []
    for turn in value:
        if not (isinstance(turn, dict) and "role" in turn):
            continue
        turn = dict(turn)
        # The MiniMax-M2 template reads `reasoning_content`; the source dataset stores
        # per-turn reasoning under `reasoning`. Map it across so reasoning renders.
        if turn.get("role") in ("assistant", "model") and turn.get("reasoning") and not turn.get("reasoning_content"):
            turn["reasoning_content"] = turn["reasoning"]
        out.append(turn)
    return out or None


def extract_tools(value):
    """Wrap bare `functions` objects as OpenAI tools (`{"type":"function","function":<fn>}`)."""
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return []
    tools = []
    for fn in value:
        if not isinstance(fn, dict):
            continue
        if fn.get("type") == "function" and isinstance(fn.get("function"), dict):
            tools.append(dict(fn))
        else:
            tools.append({"type": "function", "function": dict(fn)})
    return tools


def tokenize_row(tokenizer, messages, tools=None, chat_template=None):
    """Render+tokenize one conversation; return (input_ids, labels, used_assistant_mask).

    Prefers assistant-only labels via return_assistant_tokens_mask; falls back to
    labels = input_ids when the template lacks {% generation %} (MiniMax-M2 case)."""
    kw = {}
    if tools:
        kw["tools"] = tools
    if chat_template is not None:
        kw["chat_template"] = chat_template
    try:
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True, **kw,
        )
        ids = list(out["input_ids"])
        mask = out.get("assistant_masks")
        if mask is not None and any(mask):
            labels = [tid if m else IGNORE_INDEX for tid, m in zip(ids, mask)]
            return ids, labels, True
        return ids, list(ids), False
    except Exception:
        pass
    out = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, **kw)
    ids = list(out["input_ids"])
    return ids, list(ids), False


def collate_bin(docs_ids, docs_labels):
    input_ids, labels, position_ids, lengths = [], [], [], []
    for ids, labs in zip(docs_ids, docs_labels):
        L = len(ids)
        input_ids.extend(ids)
        labels.extend(labs)
        position_ids.extend(range(L))
        lengths.append(L)
    return {
        "input_ids": np.array(input_ids, dtype=np.int64),
        "labels": np.array(labels, dtype=np.int64),
        "position_ids": np.array(position_ids, dtype=np.uint32),
        "attention_mask": np.array(lengths, dtype=np.uint32),
    }


def assert_invariants(sample):
    n = len(sample["input_ids"])
    assert len(sample["labels"]) == n, "labels length mismatch"
    assert len(sample["position_ids"]) == n, "position_ids length mismatch"
    assert int(np.sum(sample["attention_mask"])) == n, (
        f"sum(attention_mask)={int(np.sum(sample['attention_mask']))} != len(input_ids)={n}"
    )


def pack(df, tokenizer, out_dir, max_seq_len, max_rows=None, chat_template=None, enable_tools=True):
    from chinidataset import ParquetWriter

    if max_rows is not None:
        df = df.head(max_rows)
    has_functions = enable_tools and ("functions" in df.columns)

    total_rows = len(df)
    n_bins = n_docs = n_drop_long = n_drop_empty = n_masked = n_tools = total_tokens = 0
    cur_ids, cur_labels, cur_count = [], [], 0
    first_sample = None

    writer = ParquetWriter(out=out_dir, columns=COLUMNS, exist_ok=True)
    with writer as out:
        for i in range(total_rows):
            row = df.iloc[i]
            messages = extract_messages(row.get("messages") if hasattr(row, "get") else row["messages"])
            if not messages:
                n_drop_empty += 1
                continue
            tools = extract_tools(row["functions"]) if has_functions else []
            if tools:
                n_tools += 1
            try:
                ids, labels, used_mask = tokenize_row(tokenizer, messages, tools=tools, chat_template=chat_template)
            except Exception as e:  # noqa: BLE001
                print(f"[pack_dataset] row {i}: tokenize failed ({type(e).__name__}: {e}); skipping", flush=True)
                n_drop_empty += 1
                continue

            length = len(ids)
            if length == 0:
                n_drop_empty += 1
                continue
            if length > max_seq_len:
                n_drop_long += 1
                continue
            if used_mask:
                n_masked += 1

            if cur_count + length > max_seq_len:
                if cur_ids:
                    sample = collate_bin(cur_ids, cur_labels)
                    assert_invariants(sample)
                    first_sample = first_sample or sample
                    out.write(sample)
                    n_bins += 1
                    total_tokens += len(sample["input_ids"])
                cur_ids, cur_labels, cur_count = [ids], [labels], length
            else:
                cur_ids.append(ids)
                cur_labels.append(labels)
                cur_count += length
            n_docs += 1

            if (i + 1) % 100 == 0:
                print(f"[pack_dataset] processed {i + 1}/{total_rows} rows, {n_bins} bins so far", flush=True)

        if cur_ids:
            sample = collate_bin(cur_ids, cur_labels)
            assert_invariants(sample)
            first_sample = first_sample or sample
            out.write(sample)
            n_bins += 1
            total_tokens += len(sample["input_ids"])

    efficiency = (total_tokens / n_bins / max_seq_len) if n_bins else 0.0
    return {
        "total_rows": total_rows, "docs_packed": n_docs, "dropped_long": n_drop_long,
        "dropped_empty": n_drop_empty, "assistant_masked_rows": n_masked, "rows_with_tools": n_tools,
        "n_bins": n_bins, "total_tokens": total_tokens, "max_seq_len": max_seq_len,
        "efficiency": efficiency, "first_sample": first_sample,
    }


def main():
    parser = argparse.ArgumentParser(description="Pack a HF chat-parquet into ChiniDataset multipacking format (MiniMax-M2).")
    parser.add_argument("--out", default="./packed_data")
    parser.add_argument("--max-seq-len", type=int, default=131072,
                        help="Max tokens per packed bin (default 131072 = 128k; fits the longest conv whole).")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="Tokenizer (tokenizer only, no weights).")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--native-reasoning", action="store_true",
                        help="Use the stock template (reasoning only on last-user turns). Default: ALL reasoning.")
    parser.add_argument("--no-tools", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"[pack_dataset] loading tokenizer {args.tokenizer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    all_reasoning = not args.native_reasoning
    chat_template = build_chat_template(tokenizer, all_reasoning=all_reasoning)
    print(f"[pack_dataset] reasoning: {'ALL assistant turns' if all_reasoning else 'stock (last-user turns only)'}"
          f" | tools: {'OFF' if args.no_tools else 'from functions column'}", flush=True)

    df = load_source_dataframe(args.repo, args.file)
    print(f"[pack_dataset] source rows: {len(df)}; columns: {list(df.columns)}", flush=True)
    if "messages" not in df.columns:
        raise SystemExit("[pack_dataset] ERROR: source parquet has no 'messages' column")
    if not args.no_tools and "functions" not in df.columns:
        print("[pack_dataset] WARN: no 'functions' column — packing without tool definitions", flush=True)

    stats = pack(df, tokenizer, args.out, args.max_seq_len, max_rows=args.max_rows,
                 chat_template=chat_template, enable_tools=not args.no_tools)

    print("\n========== PACK SUMMARY ==========", flush=True)
    print(f"source rows seen        : {stats['total_rows']}")
    print(f"docs packed             : {stats['docs_packed']}")
    print(f"dropped (too long)      : {stats['dropped_long']} (len > max_seq_len={stats['max_seq_len']}; never split)")
    print(f"dropped (empty/bad)     : {stats['dropped_empty']}")
    print(f"rows with tool defs     : {stats['rows_with_tools']}")
    print(f"packed bins written     : {stats['n_bins']}")
    print(f"total packed tokens     : {stats['total_tokens']}")
    mean = (stats['total_tokens'] / stats['n_bins']) if stats['n_bins'] else 0
    print(f"mean tokens / bin       : {mean:.1f}")
    print(f"packing efficiency      : {stats['efficiency'] * 100:.1f}%")
    if stats["assistant_masked_rows"] > 0:
        print(f"label masking           : assistant-only on {stats['assistant_masked_rows']} rows")
    else:
        print("label masking           : FALLBACK labels = input_ids (no {% generation %}) -> full-sequence")
    print("==================================\n", flush=True)

    from chinidataset import StreamingDataset

    ds = StreamingDataset(local=args.out)
    print(f"[pack_dataset] read back {len(ds)} packed records from {args.out}", flush=True)
    for idx in range(min(3, len(ds))):
        s = ds[idx]
        ii, lab, pos, am = (np.asarray(s[k]) for k in ("input_ids", "labels", "position_ids", "attention_mask"))
        n = len(ii)
        assert len(lab) == n and len(pos) == n and int(am.sum()) == n, f"sample {idx}: invariant broken"
        off = 0
        for L in am.tolist():
            assert list(pos[off:off + int(L)]) == list(range(int(L))), f"sample {idx}: position_ids not reset per doc"
            off += int(L)
        print(f"[pack_dataset] sample {idx}: len={n} docs={len(am)} OK", flush=True)
    print("[pack_dataset] invariants verified. Done.", flush=True)


if __name__ == "__main__":
    main()
