"""Pack a HF chat-parquet dataset into ChiniDataset multipacking format for Qwen3.5.

Reads a chat dataset (messages + optional functions columns) from the Hub, renders each
conversation through Qwen3.5's chat template (with an optional reasoning-guard relaxation so
EVERY model turn's <think> block is trained, not just the last), tokenizes, and multipacks
documents into fixed-length bins written as a ChiniDataset (ParquetWriter).

COLUMN ENCODINGS (per packed bin):
  input_ids      int64[]   concatenated token ids of all docs in the bin
  labels         int64[]   assistant-only labels (IGNORE_INDEX elsewhere) or = input_ids
  position_ids   uint32[]  per-doc 0..L-1 (resets at each document boundary)
  attention_mask uint32[]  per-doc lengths (sum == len(input_ids)); NOT a 0/1 mask

The trainer's collator rebuilds cu_seqlens from `attention_mask` (the per-doc lengths) so the
varlen attention sees per-document causal blocks.
"""
import os
import sys
import json
import argparse

import numpy as np

# --- Read HF_TOKEN from the gateway .env (the model repo may be GATED) --------
GATEWAY_ENV = "/home/husein/ssd3/GPUPlatform/gateway/.env"


def _load_hf_token_from_env_file(path: str) -> None:
    """Set HF_TOKEN / HUGGING_FACE_HUB_TOKEN from `grep 'HF_TOKEN=' <gateway/.env>`.

    Does not clobber an HF_TOKEN already present in the environment.
    """
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
        print(f"[pack_dataset] WARN: {path} not found; relying on ambient HF_TOKEN",
              file=sys.stderr)


_load_hf_token_from_env_file(GATEWAY_ENV)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# COLUMN ENCODINGS -- see module docstring.
COLUMNS = {
    "input_ids": "int64[]",
    "labels": "int64[]",
    "position_ids": "uint32[]",
    "attention_mask": "uint32[]",
}

IGNORE_INDEX = -100

DEFAULT_REPO = "Scicom-intl/Function-Call-TaaS"
DEFAULT_FILE = "glm5.1-fp8-test/test-00000-of-00001.parquet"
# Qwen3.6-27B and Qwen3.6-35B-A3B share the SAME tokenizer + chat template, so one pack
# serves both. (It differs from Qwen3.5-27B — different template — so a 3.5 pack is NOT valid
# for 3.6 and vice versa.)
DEFAULT_TOKENIZER = "Qwen/Qwen3.6-27B"

# --- Reasoning rendering ------------------------------------------------------
# Both Qwen3.5 and Qwen3.6 stock templates only emit <think>...</think> on assistant
# turns AFTER `ns.last_query_index` (the last real user query, not a tool-response turn).
# In a multi-step agentic trajectory every intermediate assistant turn is before that
# index, so their reasoning is silently dropped. We enable ALL turns' reasoning two ways
# (whichever the loaded template supports; harmless if it doesn't):
#
#   1. Qwen3.6 (preferred): the template guards on
#        (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index)
#      so we simply pass `preserve_thinking=True` to apply_chat_template (see tokenize_row).
#   2. Qwen3.5 (legacy): the template has no preserve_thinking hook, so we surgically swap
#        STOCK : loop.index0 > ns.last_query_index   ->   ALL : loop.index0 >= 0  (always true)
#      on the template string. If the guard string is absent (e.g. Qwen3.6) the swap is a
#      no-op that warns and returns the stock template rather than crashing — the
#      preserve_thinking kwarg carries the all-reasoning behaviour there.
_REASONING_GUARD_STOCK = (
    "{%- if loop.index0 > ns.last_query_index %}"
)
_REASONING_GUARD_ALL = (
    "{%- if loop.index0 >= 0 %}"
)


def build_chat_template(tokenizer, all_reasoning: bool):
    """Return the chat-template string to use (stock, or reasoning-relaxed).

    ``all_reasoning=False`` -> use the tokenizer's stock template unchanged.
    ``all_reasoning=True``  -> render every model turn's thinking block.

    Unlike the original version this function does NOT raise when the guard string
    is absent -- it warns and returns the stock template so the script stays runnable
    even if the upstream template changes or the guard constant above needs updating.
    """
    stock = tokenizer.chat_template
    if not all_reasoning:
        return stock
    if _REASONING_GUARD_STOCK not in stock:
        print(
            "[pack_dataset] INFO: Qwen3.5 reasoning-guard string not found in the chat "
            "template (expected for Qwen3.6). Using the stock template + passing "
            "preserve_thinking=True to render ALL turns' reasoning.",
            file=sys.stderr,
        )
        return stock
    return stock.replace(_REASONING_GUARD_STOCK, _REASONING_GUARD_ALL)


def load_source_dataframe(repo: str, filename: str):
    """Load the source parquet as a pandas DataFrame.

    Tries the ``hf://`` pandas path first (honors HF_TOKEN), falling back to an
    explicit ``hf_hub_download(repo_type="dataset")`` + ``read_parquet``.
    """
    import pandas as pd

    hf_uri = f"hf://datasets/{repo}/{filename}"
    try:
        print(f"[pack_dataset] reading {hf_uri}", flush=True)
        return pd.read_parquet(hf_uri)
    except Exception as e:  # noqa: BLE001
        print(f"[pack_dataset] hf:// read failed ({type(e).__name__}: {e}); "
              f"falling back to hf_hub_download", flush=True)
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        local = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
        return pd.read_parquet(local)


def _normalize_tool_calls(tool_calls):
    """Convert GLM-style tool_calls to the format Qwen3.5's template expects.

    The Qwen3.5 chat template iterates tool_call.function.arguments with:
        for args_name, args_value in tool_call.function.arguments | items
    so `arguments` MUST be a plain Python dict, never a JSON string.

    GLM stores calls as ``[{"name": "fn", "arguments": {...}}]`` -- bare format.
    Qwen3.5 needs ``[{"type": "function", "function": {"name": "fn", "arguments": {...}}}]``.
    Already-wrapped entries are passed through; JSON-string arguments are parsed back to dict.
    """
    normalized = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            continue
        # Already in OpenAI format.
        if tc.get("type") == "function" and isinstance(tc.get("function"), dict):
            fn = dict(tc["function"])
            # arguments must be a dict, not a JSON string.
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except (json.JSONDecodeError, TypeError):
                    fn["arguments"] = {}
            elif fn.get("arguments") is None:
                fn["arguments"] = {}
            entry = {"type": "function", "function": fn}
            entry["id"] = tc.get("id", f"call_{i}")
            normalized.append(entry)
        elif "name" in tc:
            # Bare GLM format: wrap it.
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            elif not isinstance(args, dict):
                args = {}
            normalized.append({
                "type": "function",
                "function": {"name": tc["name"], "arguments": args},
                "id": tc.get("id", f"call_{i}"),
            })
    return normalized


def extract_messages(value):
    """Normalize a ``messages`` cell into a list[dict] compatible with Qwen3.5.

    The source column stores ``messages`` as a JSON *string* (verified on the real
    parquet), but also tolerate a row that is already a list / numpy array of dicts.

    Normalizations applied for Qwen3.5 compatibility:
      - ``content: null``       -> ``content: ""``  (null crashes the template)
      - ``role: "observation"`` -> ``role: "tool"``  (GLM role name)
      - ``tool_calls``          -> OpenAI format with JSON-string arguments
      - tool messages           -> ensure ``tool_call_id`` is present
    """
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

    # First pass: collect tool-call ids in order so we can assign matching
    # tool_call_ids to subsequent tool-response turns.
    call_ids = []
    for turn in value:
        if isinstance(turn, dict) and turn.get("role") in ("assistant", "model"):
            for i, tc in enumerate(turn.get("tool_calls") or []):
                if isinstance(tc, dict):
                    cid = tc.get("id") or (
                        tc.get("function", {}).get("id") if isinstance(tc.get("function"), dict) else None
                    ) or f"call_{i}"
                    call_ids.append(cid)

    tool_response_idx = 0
    out = []
    for turn in value:
        if not (isinstance(turn, dict) and "role" in turn):
            continue
        t = dict(turn)

        # Remap GLM observation role.
        if t["role"] == "observation":
            t["role"] = "tool"

        # Null content -> empty string.
        if t.get("content") is None:
            t["content"] = ""

        # Normalize tool_calls in assistant turns.
        if t.get("tool_calls"):
            t["tool_calls"] = _normalize_tool_calls(t["tool_calls"])

        # Ensure tool-response turns have a tool_call_id.
        if t["role"] == "tool" and not t.get("tool_call_id"):
            if tool_response_idx < len(call_ids):
                t["tool_call_id"] = call_ids[tool_response_idx]
            else:
                t["tool_call_id"] = f"call_{tool_response_idx}"
            tool_response_idx += 1

        out.append(t)
    return out or None


def extract_tools(value):
    """Normalize a ``functions`` cell into the OpenAI tool list the template wants.

    The source ``functions`` column is a JSON *string* of BARE function objects
    (``{"name", "description", "parameters", ...}``). The Qwen3.5 template's
    ``format_function_declaration`` accesses ``tool['function']['name']``, so each
    bare function must be wrapped as ``{"type": "function", "function": <fn>}`` --
    passing the bare list is exactly what raises
    ``'dict object' has no attribute 'function'``.
    Returns a list (possibly empty) suitable for ``apply_chat_template(tools=...)``.
    """
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


def tokenize_row(tokenizer, messages, tools=None, chat_template=None,
                 preserve_thinking=False):
    """Render+tokenize one conversation; return (input_ids, labels, used_mask).

    ``tools``            : OpenAI-wrapped tool list (from extract_tools) -- rendered
                           into the system block so the model conditions on the
                           available functions. An empty list means no tool defs.
    ``chat_template``    : the template string to render with (stock or
                           reasoning-relaxed); None uses the tokenizer's default.
    ``preserve_thinking``: pass ``preserve_thinking=True`` to apply_chat_template so
                           Qwen3.6 renders EVERY turn's <think> block (templates that
                           don't reference the variable ignore it).

    PREFERS assistant-only labels via ``return_assistant_tokens_mask=True``; falls
    back to ``labels = input_ids`` when the template lacks ``{% generation %}``
    (mask all-zero) or the kwarg is unsupported.
    """
    kw = {}
    if tools:
        kw["tools"] = tools
    if chat_template is not None:
        kw["chat_template"] = chat_template
    if preserve_thinking:
        kw["preserve_thinking"] = True

    # First attempt: assistant-token mask.
    try:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            **kw,
        )
        ids = list(out["input_ids"])
        mask = out.get("assistant_masks")
        if mask is not None and any(mask):
            labels = [tid if m else IGNORE_INDEX for tid, m in zip(ids, mask)]
            return ids, labels, True
        return ids, list(ids), False
    except Exception:
        # return_assistant_tokens_mask not accepted — try without it.
        pass

    # Fallback: plain tokenization (no assistant mask).
    import traceback as _tb
    try:
        out = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, **kw)
        ids = list(out["input_ids"])
        return ids, list(ids), False
    except Exception as e:
        # Re-raise with the full traceback attached so pack() can log it.
        raise RuntimeError(
            f"{type(e).__name__}: {e}\n" + _tb.format_exc()
        ) from e


def collate_bin(docs_ids, docs_labels):
    """Build one packed-bin sample dict from a list of per-doc token/label arrays.

    Mirrors pack_stage1.py's ``collator``: concatenate ids, build per-doc reset
    position_ids, and store per-doc lengths in attention_mask.
    """
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
    """Assert sum(attention_mask) == len(input_ids) == len(position_ids) == len(labels)."""
    n = len(sample["input_ids"])
    assert len(sample["labels"]) == n, "labels length mismatch"
    assert len(sample["position_ids"]) == n, "position_ids length mismatch"
    assert int(np.sum(sample["attention_mask"])) == n, (
        f"sum(attention_mask)={int(np.sum(sample['attention_mask']))} != len(input_ids)={n}"
    )


def pack(df, tokenizer, out_dir, max_seq_len, max_rows=None,
         chat_template=None, enable_tools=True, preserve_thinking=False):
    """Tokenize + multipack ``df`` into a ChiniDataset at ``out_dir``.

    Each row is rendered with its ``messages`` AND (when ``enable_tools``) the
    wrapped ``functions`` column as ``tools=``, using ``chat_template`` (stock or
    reasoning-relaxed).

    Long-document handling: a single conversation whose tokenized length exceeds
    ``max_seq_len`` is DROPPED (same policy as pack_stage1.py). We never split a
    conversation across bins, so position_ids/labels stay coherent per document.

    Returns a stats dict for the run summary.
    """
    from chinidataset import ParquetWriter

    if max_rows is not None:
        df = df.head(max_rows)

    has_functions = enable_tools and ("functions" in df.columns)

    total_rows = len(df)
    n_bins = 0
    n_docs_packed = 0
    n_dropped_long = 0
    n_dropped_empty = 0
    n_assistant_masked = 0
    n_rows_with_tools = 0
    total_tokens = 0

    cur_ids, cur_labels, cur_count = [], [], 0
    first_sample_for_check = None

    writer = ParquetWriter(out=out_dir, columns=COLUMNS, exist_ok=True)
    with writer as out:
        for i in range(total_rows):
            row = df.iloc[i]
            messages = extract_messages(
                row.get("messages") if hasattr(row, "get") else row["messages"]
            )
            if not messages:
                n_dropped_empty += 1
                continue

            tools = extract_tools(row["functions"]) if has_functions else []
            if tools:
                n_rows_with_tools += 1

            try:
                ids, labels, used_mask = tokenize_row(
                    tokenizer, messages, tools=tools, chat_template=chat_template,
                    preserve_thinking=preserve_thinking,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[pack_dataset] row {i}: tokenize failed "
                      f"({type(e).__name__}: {e}); skipping", flush=True)
                n_dropped_empty += 1
                continue

            length = len(ids)
            if length == 0:
                n_dropped_empty += 1
                continue
            if length > max_seq_len:
                n_dropped_long += 1
                continue

            if used_mask:
                n_assistant_masked += 1

            if cur_count + length > max_seq_len:
                if cur_ids:
                    sample = collate_bin(cur_ids, cur_labels)
                    assert_invariants(sample)
                    if first_sample_for_check is None:
                        first_sample_for_check = sample
                    out.write(sample)
                    n_bins += 1
                    total_tokens += len(sample["input_ids"])
                cur_ids, cur_labels, cur_count = [ids], [labels], length
            else:
                cur_ids.append(ids)
                cur_labels.append(labels)
                cur_count += length
            n_docs_packed += 1

            if (i + 1) % 100 == 0:
                print(f"[pack_dataset] processed {i + 1}/{total_rows} rows, "
                      f"{n_bins} bins so far", flush=True)

        # Flush the final partial bin.
        if cur_ids:
            sample = collate_bin(cur_ids, cur_labels)
            assert_invariants(sample)
            if first_sample_for_check is None:
                first_sample_for_check = sample
            out.write(sample)
            n_bins += 1
            total_tokens += len(sample["input_ids"])

    efficiency = (total_tokens / n_bins / max_seq_len) if n_bins else 0.0
    return {
        "total_rows": total_rows,
        "docs_packed": n_docs_packed,
        "dropped_long": n_dropped_long,
        "dropped_empty": n_dropped_empty,
        "assistant_masked_rows": n_assistant_masked,
        "rows_with_tools": n_rows_with_tools,
        "n_bins": n_bins,
        "total_tokens": total_tokens,
        "max_seq_len": max_seq_len,
        "efficiency": efficiency,
        "first_sample": first_sample_for_check,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pack a HF chat-parquet dataset into ChiniDataset multipacking format "
                    "for Qwen3.5 training."
    )
    parser.add_argument("--out", default="./packed_data",
                        help="Output ChiniDataset directory (default: ./packed_data).")
    parser.add_argument("--max-seq-len", type=int, default=131072,
                        help="Max tokens per packed bin (default: 131072 = 128k).")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help=f"Tokenizer to load (tokenizer only, no weights). "
                             f"Default: {DEFAULT_TOKENIZER}.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap source rows for a quick test (default: all).")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"HF dataset repo (default: {DEFAULT_REPO}).")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help=f"Parquet file within the repo (default: {DEFAULT_FILE}).")
    parser.add_argument("--native-reasoning", action="store_true",
                        help="Use the stock Qwen3.5 template unchanged (may suppress "
                             "reasoning on most turns). Default relaxes the guard to train "
                             "ALL model turn reasoning.")
    parser.add_argument("--no-tools", action="store_true",
                        help="Do NOT pass the `functions` column as tools to the chat template.")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"[pack_dataset] loading tokenizer {args.tokenizer} (tokenizer only)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    all_reasoning = not args.native_reasoning
    chat_template = build_chat_template(tokenizer, all_reasoning=all_reasoning)
    print(
        f"[pack_dataset] reasoning: "
        f"{'ALL model turns' if all_reasoning else 'stock (native template)'}"
        f" | tools: {'OFF' if args.no_tools else 'from functions column'}",
        flush=True,
    )

    df = load_source_dataframe(args.repo, args.file)
    print(f"[pack_dataset] source rows: {len(df)}; columns: {list(df.columns)}", flush=True)
    if "messages" not in df.columns:
        raise SystemExit("[pack_dataset] ERROR: source parquet has no 'messages' column")
    if not args.no_tools and "functions" not in df.columns:
        print("[pack_dataset] WARN: no 'functions' column -- packing without tool definitions",
              flush=True)

    stats = pack(df, tokenizer, args.out, args.max_seq_len, max_rows=args.max_rows,
                 chat_template=chat_template, enable_tools=not args.no_tools,
                 preserve_thinking=all_reasoning)

    # ----- Summary -----------------------------------------------------------
    print("\n========== PACK SUMMARY ==========", flush=True)
    print(f"source rows seen        : {stats['total_rows']}")
    print(f"docs packed             : {stats['docs_packed']}")
    print(f"dropped (too long)      : {stats['dropped_long']} "
          f"(len > max_seq_len={stats['max_seq_len']}; dropped, never split)")
    print(f"dropped (empty/bad)     : {stats['dropped_empty']}")
    print(f"rows with tool defs     : {stats['rows_with_tools']}")
    print(f"packed bins written     : {stats['n_bins']}")
    print(f"total packed tokens     : {stats['total_tokens']}")
    mean_per_bin = (stats['total_tokens'] / stats['n_bins']) if stats['n_bins'] else 0
    print(f"mean tokens / bin       : {mean_per_bin:.1f}")
    print(f"packing efficiency      : {stats['efficiency'] * 100:.1f}% "
          f"(mean tokens-per-bin / max_seq_len)")
    if stats["assistant_masked_rows"] > 0:
        print(f"label masking           : assistant-only on "
              f"{stats['assistant_masked_rows']} rows (template has {{% generation %}})")
    else:
        print("label masking           : FALLBACK labels = input_ids "
              "(template lacks {% generation %}; assistant mask all-zero) "
              "-> trains on the full packed sequence")
    print("==================================\n", flush=True)

    # ----- Read back + verify invariants on a few samples --------------------
    from chinidataset import StreamingDataset

    ds = StreamingDataset(local=args.out)
    print(f"[pack_dataset] read back {len(ds)} packed records from {args.out}", flush=True)
    n_check = min(3, len(ds))
    for idx in range(n_check):
        s = ds[idx]
        ii = np.asarray(s["input_ids"])
        lab = np.asarray(s["labels"])
        pos = np.asarray(s["position_ids"])
        am = np.asarray(s["attention_mask"])
        n = len(ii)
        assert len(lab) == n, f"sample {idx}: labels length mismatch"
        assert len(pos) == n, f"sample {idx}: position_ids length mismatch"
        assert int(am.sum()) == n, (
            f"sample {idx}: sum(attention_mask)={int(am.sum())} != len(input_ids)={n}"
        )
        off = 0
        for L in am.tolist():
            seg = pos[off:off + int(L)]
            assert list(seg) == list(range(int(L))), (
                f"sample {idx}: position_ids do not reset per doc at offset {off}"
            )
            off += int(L)
        print(f"[pack_dataset] sample {idx}: len(input_ids)={n} "
              f"docs={len(am)} sum(am)={int(am.sum())} OK", flush=True)

    print("[pack_dataset] invariants verified. Done.", flush=True)


if __name__ == "__main__":
    main()
