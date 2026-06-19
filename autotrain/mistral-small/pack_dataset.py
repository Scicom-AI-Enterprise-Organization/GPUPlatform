"""Pack a HuggingFace chat-parquet dataset into the multipacking ChiniDataset
(StreamingDataset) format consumed by `mistral_small.py`.

The Mistral-Small-4 sibling of minimax-m2's `pack_dataset.py` — same bin-packing scheme
(tokenize each conversation via the chat template, greedily pack whole conversations into
`--max-seq-len` token bins), same emitted columns:

  - input_ids     : concatenated token ids of every doc in the bin
  - labels        : same length; -100 masks non-trained tokens (here: full-seq, see below)
  - position_ids  : RESET per document (0..L-1 per packed doc), concatenated
  - attention_mask: the per-document SEQUENCE LENGTHS (NOT a dense mask) -> cu_seqlens

  Invariant: sum(attention_mask) == len(input_ids) == len(position_ids) == len(labels)

Differences from the minimax-m2 packer (all because the chat template differs):
  * TOKENIZER defaults to `mistralai/Mistral-Small-4-119B-2603` (tekken + chat_template.jinja).
  * REASONING: Mistral-Small-4 renders assistant reasoning as a `[THINK]...[/THINK]` block,
    produced only when the assistant turn's `content` is a LIST containing a
    `{"type": "thinking", "thinking": ...}` chunk. The source dataset stores per-turn
    reasoning under a flat `reasoning` field, so by default we rewrite each assistant turn's
    content into `[{"type":"thinking",...}, {"type":"text",...}]` so EVERY assistant turn's
    reasoning trains, and pass `reasoning_effort="high"`. `--native-reasoning` leaves messages
    untouched (reasoning dropped — the template ignores the flat `reasoning` field).
  * LABELS: the Mistral-Small-4 template has no `{% generation %}` block, so an assistant-token
    mask is unavailable (all-zero) and we fall back to `labels = input_ids` (train on the full
    packed sequence), detected per-row at runtime — same as minimax-m2.
  * TOOLS: the template emits `[AVAILABLE_TOOLS]{{ tools | tojson }}[/AVAILABLE_TOOLS]`
    verbatim, so the bare `functions` objects are wrapped as `{"type":"function","function":<fn>}`
    and passed via `apply_chat_template(tools=...)`. (`--no-tools` to skip.) Tool calls are
    normalised so each carries a nested `function.{name,arguments}` (the template reads those).

Mistral-Small-4's attention is O(S) (FlashAttention varlen, uniform head_dim 128, MLA), so long
bins are far more feasible than a full-attention model — `--max-seq-len 131072` packs the
Function-Call conversations (19k-81k tokens) whole. Drop bin size if training is VRAM-bound.
"""
import argparse
import json
import os
import sys

import numpy as np

# --- Read HF_TOKEN from autotrain/.env (../.env) BEFORE importing transformers ----
# ../.env (autotrain root) is the canonical place per autotrain/CLAUDE.md; fall back to
# gateway/.env then ambient.
_ENV_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", ".env"),
    "/home/husein/ssd3/GPUPlatform/gateway/.env",
]


def _load_hf_token_from_env_files(paths) -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    for path in paths:
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
            continue
    print("[pack_dataset] WARN: no HF_TOKEN in ../.env or gateway/.env; relying on ambient", file=sys.stderr)


_load_hf_token_from_env_files(_ENV_CANDIDATES)

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
DEFAULT_TOKENIZER = "mistralai/Mistral-Small-4-119B-2603"


def _as_thinking_content(content, reasoning):
    """Rewrite an assistant turn's content into a [thinking, text] block list so the
    Mistral-Small-4 template renders `[THINK]<reasoning>[/THINK]<text>`."""
    blocks = [{"type": "thinking", "thinking": str(reasoning)}]
    if isinstance(content, str):
        if content != "":
            blocks.append({"type": "text", "text": content})
    elif isinstance(content, (list, tuple)):
        blocks.extend(content)  # already block-structured (rare for text chat)
    return blocks


def extract_messages(value, all_reasoning: bool):
    """Normalize a `messages` cell into list[dict]. When `all_reasoning`, fold each assistant
    turn's flat `reasoning` field into a `[THINK]` content block. Normalise tool calls to the
    nested `function.{name,arguments}` shape the template reads."""
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
        role = turn.get("role")
        if role == "model":
            role = turn["role"] = "assistant"

        # Fold per-turn reasoning into a [THINK] block (Mistral renders thinking only when
        # the content is a list with a `thinking` chunk).
        if all_reasoning and role == "assistant" and turn.get("reasoning"):
            turn["content"] = _as_thinking_content(turn.get("content"), turn["reasoning"])
        turn.pop("reasoning", None)
        turn.pop("reasoning_content", None)  # not read by this template

        # Normalise tool calls -> [{"function": {"name":.., "arguments":..}}]. The dataset
        # (OpenAI format) usually nests under `function` already; handle bare name/arguments too.
        tcs = turn.get("tool_calls")
        if isinstance(tcs, (list, tuple)):
            norm = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                tc = dict(tc)
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
                if fn is None:
                    fn = {"name": tc.get("name"), "arguments": tc.get("arguments", {})}
                else:
                    fn = dict(fn)
                # The template handles both string and object arguments; leave as-is but
                # default missing args to an empty object.
                if fn.get("arguments") is None:
                    fn["arguments"] = {}
                tc["function"] = fn
                norm.append(tc)
            turn["tool_calls"] = norm
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


def tokenize_row(tokenizer, messages, tools=None, reasoning_effort="none"):
    """Render+tokenize one conversation; return (input_ids, labels, used_assistant_mask).

    Prefers assistant-only labels via return_assistant_tokens_mask; falls back to
    labels = input_ids when the template lacks {% generation %} (Mistral-Small-4 case)."""
    kw = {"reasoning_effort": reasoning_effort}
    if tools:
        kw["tools"] = tools
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


def pack(df, tokenizer, out_dir, max_seq_len, max_rows=None, all_reasoning=True,
         reasoning_effort="high", enable_tools=True):
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
            messages = extract_messages(
                row.get("messages") if hasattr(row, "get") else row["messages"], all_reasoning)
            if not messages:
                n_drop_empty += 1
                continue
            tools = extract_tools(row["functions"]) if has_functions else []
            if tools:
                n_tools += 1
            try:
                ids, labels, used_mask = tokenize_row(
                    tokenizer, messages, tools=tools, reasoning_effort=reasoning_effort)
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
    parser = argparse.ArgumentParser(description="Pack a HF chat-parquet into ChiniDataset multipacking format (Mistral-Small-4).")
    parser.add_argument("--out", default="./packed_data")
    parser.add_argument("--max-seq-len", type=int, default=131072,
                        help="Max tokens per packed bin (default 131072 = 128k; fits the longest conv whole).")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="Tokenizer (tokenizer only, no weights).")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--native-reasoning", action="store_true",
                        help="Leave messages untouched (the flat `reasoning` field is dropped). "
                             "Default: fold reasoning into [THINK] blocks so it trains.")
    parser.add_argument("--reasoning-effort", default=None, choices=["none", "high"],
                        help="reasoning_effort rendered in [MODEL_SETTINGS] (default: high when "
                             "training reasoning, else none).")
    parser.add_argument("--no-tools", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"[pack_dataset] loading tokenizer {args.tokenizer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    all_reasoning = not args.native_reasoning
    reasoning_effort = args.reasoning_effort or ("high" if all_reasoning else "none")
    print(f"[pack_dataset] reasoning: {'ALL assistant turns ([THINK] blocks)' if all_reasoning else 'native (dropped)'}"
          f" | reasoning_effort={reasoning_effort}"
          f" | tools: {'OFF' if args.no_tools else 'from functions column'}", flush=True)

    df = load_source_dataframe(args.repo, args.file)
    print(f"[pack_dataset] source rows: {len(df)}; columns: {list(df.columns)}", flush=True)
    if "messages" not in df.columns:
        raise SystemExit("[pack_dataset] ERROR: source parquet has no 'messages' column")
    if not args.no_tools and "functions" not in df.columns:
        print("[pack_dataset] WARN: no 'functions' column — packing without tool definitions", flush=True)

    stats = pack(df, tokenizer, args.out, args.max_seq_len, max_rows=args.max_rows,
                 all_reasoning=all_reasoning, reasoning_effort=reasoning_effort,
                 enable_tools=not args.no_tools)

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
