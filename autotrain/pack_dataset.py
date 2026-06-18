"""Pack a HuggingFace chat-parquet dataset into the multipacking ChiniDataset
(StreamingDataset) format consumed by the Gemma-4 training script (`gemma4.py`).

This is the *chat-text* analogue of `gateway/gateway/training/tts/pack_stage1.py`.
It follows the same multipacking (bin-packing) approach: tokenize each document
(here: a chat conversation rendered via the tokenizer's chat template), then greedily
pack documents into fixed-size bins of `--max-seq-len` tokens. Per bin we emit:

  - input_ids     : concatenated token ids of every doc in the bin
  - labels        : same length as input_ids; -100 masks non-trained tokens
  - position_ids  : RESET per document (0..L-1 for each packed doc), concatenated
  - attention_mask: the per-document SEQUENCE LENGTHS within the bin
                    (NOT a dense mask). e.g. packing docs of lengths [120, 88, 300]
                    => np.array([120, 88, 300]). gemma4.py's collator concatenates
                    these across the batch and builds cu_seqlens for FlashAttention.

Invariant enforced for EVERY written sample:
    sum(attention_mask) == len(input_ids) == len(position_ids) == len(labels)

---------------------------------------------------------------------------------
WRITER API (verified from source, NOT guessed):
  ChiniDataset is *Parquet-native* — it is NOT a wrapper over MosaicML's MDSWriter.
  Source: github.com/Scicom-AI-Enterprise-Organization/ChiniDataset @ v0.2.0
    - chinidataset/__init__.py            exports `ParquetWriter`, `StreamingDataset`
    - chinidataset/writer/parquet.py      `ParquetWriter(out=..., columns={name: type})`
                                          context-manager + `.write(dict)`; array
                                          columns are typed `"<base>[]"` (e.g.
                                          "int64[]", "uint32[]"); `_parse_column_type`
                                          maps these to `pa.list_(<base>)`. numpy
                                          arrays are written natively (zero-copy).
    - chinidataset/dataset/reader.py      reads each shard via pandas
                                          `to_dict(orient='records')` so array columns
                                          come back as Python lists / numpy arrays,
                                          which gemma4.py's collator feeds to
                                          `np.concatenate(...)` directly.
  Mirrors pack_stage1.py's use of `ParquetWriter(out=..., columns=COLUMNS)`.

COLUMN ENCODINGS we use:
  input_ids    : "int64[]"   (signed: holds full Gemma vocab ids; matches collator's
                              `torch.tensor(..., dtype=torch.long)`)
  labels       : "int64[]"   (MUST be signed — -100 is the ignore index; uint would
                              wrap it. -100 -> torch.long for LigerFusedLinearCE)
  position_ids : "uint32[]"  (always >= 0; per-doc reset)
  attention_mask: "uint32[]" (always >= 0; the per-doc length array)

LABEL-MASKING DECISION (verified at runtime against google/gemma-4-31B-it):
  We PREFER training only on assistant turns and therefore *try*
  `apply_chat_template(..., return_assistant_tokens_mask=True)`. However, the
  gemma-4-31B-it chat template does NOT contain the `{% generation %}` block that
  the mask relies on, so the returned `assistant_masks` is ALL ZEROS (transformers
  even warns: "return_assistant_tokens_mask==True but chat template does not contain
  `{% generation %}` keyword"). Verified with transformers 5.3.0.
  => When the assistant mask is unavailable OR all-zero for a row, we FALL BACK to
     `labels = input_ids` (train on the full packed sequence). This is detected per
     row at runtime, so if a future template gains a `{% generation %}` block the
     script will automatically switch to assistant-only masking with no code change.
  The run summary prints how many rows used real assistant masking vs. the fallback.
---------------------------------------------------------------------------------
TOOLS + REASONING (rendered into every row by default):
  - TOOLS: the source `functions` column is a JSON string of BARE function objects.
    The gemma-4 template reads `tool['function']['name']`, so each is wrapped as
    `{"type": "function", "function": <fn>}` and passed via
    `apply_chat_template(tools=...)` -> a `<|tool>declaration:...<tool|>` block in the
    system turn. (Passing the bare list raised `'dict object' has no attribute
    'function'`.) Disable with `--no-tools`.
  - REASONING: assistant `reasoning` is rendered as a `<|channel>thought ... <channel|>`
    block. The stock template only emits it on tool-call turns AFTER the last user
    message (2 of 15 on row 0); by default we relax that single guard so EVERY
    assistant turn's reasoning is trained (15 of 15). Use `--native-reasoning` to keep
    the stock behavior. See build_chat_template().
---------------------------------------------------------------------------------
"""

import os
import sys
import json
import argparse

import numpy as np

# --- Read HF_TOKEN from the gateway .env (the gemma-4 repo is GATED) ----------
# Done BEFORE importing transformers / huggingface_hub so the token is honored.
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

# Xet/hf_transfer have stalled large HF downloads on this box (see MEMORY); disable
# them defensively for the source-parquet fetch.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# COLUMN ENCODINGS — see module docstring. int64 for the two columns that may carry
# the -100 ignore index or large vocab ids; uint32 for the always-non-negative ones.
COLUMNS = {
    "input_ids": "int64[]",
    "labels": "int64[]",
    "position_ids": "uint32[]",
    "attention_mask": "uint32[]",
}

IGNORE_INDEX = -100

DEFAULT_REPO = "Scicom-intl/Function-Call-TaaS"
DEFAULT_FILE = "glm5.1-fp8-test/test-00000-of-00001.parquet"

# --- Reasoning rendering ------------------------------------------------------
# The stock gemma-4-31B-it chat template only emits an assistant's `reasoning`
# (`<|channel>thought ... <channel|>`) when the turn (a) comes AFTER the LAST user
# message AND (b) has tool_calls. In a multi-user agentic trajectory that silently
# drops most reasoning (verified: 2 of 15 thought blocks rendered on row 0). We want
# ALL reasoning trained, so we relax that one guard to "every assistant turn that
# carries reasoning". This is a single, surgical substring swap on the stock
# template — if transformers ever changes the template, the swap fails loudly
# (KeyError below) rather than silently reverting to the stock behavior.
_REASONING_GUARD_STOCK = (
    "if thinking_text and loop.index0 > ns_turn.last_user_idx "
    "and message.get('tool_calls')"
)
_REASONING_GUARD_ALL = "if thinking_text and role == 'model'"


def build_chat_template(tokenizer, all_reasoning: bool):
    """Return the chat-template string to use (stock, or reasoning-relaxed).

    `all_reasoning=False` -> use the tokenizer's stock template unchanged.
    `all_reasoning=True`  -> render every assistant turn's reasoning.
    """
    stock = tokenizer.chat_template
    if not all_reasoning:
        return stock
    if _REASONING_GUARD_STOCK not in stock:
        raise KeyError(
            "[pack_dataset] could not find the gemma-4 reasoning guard in the chat "
            "template; the template changed upstream. Inspect tokenizer.chat_template "
            "and update _REASONING_GUARD_STOCK (or pass --native-reasoning)."
        )
    return stock.replace(_REASONING_GUARD_STOCK, _REASONING_GUARD_ALL)


def load_source_dataframe(repo: str, filename: str):
    """Load the source parquet as a pandas DataFrame.

    Tries the `hf://` pandas path first (honors HF_TOKEN), falling back to an
    explicit `hf_hub_download(repo_type="dataset")` + `read_parquet`.
    """
    import pandas as pd

    hf_uri = f"hf://datasets/{repo}/{filename}"
    try:
        print(f"[pack_dataset] reading {hf_uri}", flush=True)
        return pd.read_parquet(hf_uri)
    except Exception as e:  # noqa: BLE001 — fall back to explicit download
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


def extract_messages(value):
    """Normalize a `messages` cell into a list[dict].

    The source column stores `messages` as a JSON *string* (verified on the real
    parquet), but also tolerate a row that is already a list / numpy array of dicts.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    # numpy arrays of dicts -> list
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return None
    out = []
    for turn in value:
        if isinstance(turn, dict) and "role" in turn:
            out.append(dict(turn))
    return out or None


def extract_tools(value):
    """Normalize a `functions` cell into the OpenAI tool list the template wants.

    The source `functions` column is a JSON *string* of BARE function objects
    (`{"name", "description", "parameters", ...}`). The gemma-4 template's
    `format_function_declaration` accesses `tool['function']['name']`, so each bare
    function must be wrapped as `{"type": "function", "function": <fn>}` — passing
    the bare list is exactly what raised `'dict object' has no attribute 'function'`.
    Returns a list (possibly empty) suitable for `apply_chat_template(tools=...)`.
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
        # Already OpenAI-wrapped? pass through; else wrap the bare function.
        if fn.get("type") == "function" and isinstance(fn.get("function"), dict):
            tools.append(dict(fn))
        else:
            tools.append({"type": "function", "function": dict(fn)})
    return tools


def tokenize_row(tokenizer, messages, tools=None, chat_template=None):
    """Render+tokenize one conversation; return (input_ids, labels, used_mask).

    `tools`         : OpenAI-wrapped tool list (from extract_tools) — rendered into the
                      system block so the model conditions on the available functions.
                      An empty list means no tool definitions for this row.
    `chat_template` : the template string to render with (stock or reasoning-relaxed);
                      None uses the tokenizer's default.

    PREFERS assistant-only labels via `return_assistant_tokens_mask=True`; falls back
    to `labels = input_ids` when the template lacks `{% generation %}` (mask all-zero)
    or the kwarg is unsupported. The gemma-4 template has no `{% generation %}`, so in
    practice this always falls back to full-sequence labels (train on everything).
    """
    kw = {}
    if tools:
        kw["tools"] = tools
    if chat_template is not None:
        kw["chat_template"] = chat_template

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
            # Train only on assistant tokens: -100 everywhere mask==0.
            labels = [tid if m else IGNORE_INDEX for tid, m in zip(ids, mask)]
            return ids, labels, True
        # mask unsupported / all-zero -> fall through to full-sequence labels.
        return ids, list(ids), False
    except Exception:
        # return_assistant_tokens_mask not accepted by this tokenizer version.
        pass

    # Fallback path: plain tokenized ids (handle both BatchEncoding-dict and list
    # returns across transformers versions).
    out = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, **kw)
    ids = list(out["input_ids"])
    return ids, list(ids), False


def collate_bin(docs_ids, docs_labels):
    """Build one packed-bin sample dict from a list of per-doc token/label arrays.

    Mirrors pack_stage1.py's `collator`: concatenate ids, build per-doc reset
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
         chat_template=None, enable_tools=True):
    """Tokenize + multipack `df` into a ChiniDataset at `out_dir`.

    Each row is rendered with its `messages` AND (when `enable_tools`) the wrapped
    `functions` column as `tools=`, using `chat_template` (stock or reasoning-relaxed).

    Long-document handling: a single conversation whose tokenized length exceeds
    `max_seq_len` is DROPPED (same policy as pack_stage1.py, which `continue`s past
    any doc with `length > sequence_length`). We never split a conversation across
    bins, so position_ids/labels stay coherent per document.

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

    # Current bin accumulator.
    cur_ids, cur_labels, cur_count = [], [], 0
    first_sample_for_check = None

    # Overwrite any existing dataset dir (pack_stage1.py rm -rf's the folder).
    writer = ParquetWriter(out=out_dir, columns=COLUMNS, exist_ok=True)
    with writer as out:
        for i in range(total_rows):
            row = df.iloc[i]
            messages = extract_messages(row.get("messages") if hasattr(row, "get") else row["messages"])
            if not messages:
                n_dropped_empty += 1
                continue

            tools = extract_tools(row["functions"]) if has_functions else []
            if tools:
                n_rows_with_tools += 1

            try:
                ids, labels, used_mask = tokenize_row(
                    tokenizer, messages, tools=tools, chat_template=chat_template
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
                # Drop docs longer than a single bin (pack_stage1.py policy).
                n_dropped_long += 1
                continue

            if used_mask:
                n_assistant_masked += 1

            # Greedy bin-packing: flush the current bin before it overflows.
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
        description="Pack a HF chat-parquet dataset into ChiniDataset multipacking format."
    )
    parser.add_argument("--out", default="./packed_data",
                        help="Output ChiniDataset directory (default: ./packed_data).")
    parser.add_argument("--max-seq-len", type=int, default=131072,
                        help="Max tokens per packed bin (default: 131072 = 128k). With tools + "
                             "all reasoning the longest row is ~81k; 128k fits every row whole.")
    parser.add_argument("--tokenizer", default="google/gemma-4-31B-it",
                        help="Tokenizer to load (tokenizer only, no weights).")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap source rows for a quick test (default: all).")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"HF dataset repo (default: {DEFAULT_REPO}).")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help=f"Parquet file within the repo (default: {DEFAULT_FILE}).")
    parser.add_argument("--native-reasoning", action="store_true",
                        help="Use the stock gemma-4 template (renders reasoning ONLY on the "
                             "last-user tool-call turns). Default renders ALL assistant reasoning.")
    parser.add_argument("--no-tools", action="store_true",
                        help="Do NOT pass the `functions` column as tools to the chat template.")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"[pack_dataset] loading tokenizer {args.tokenizer} (tokenizer only)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    all_reasoning = not args.native_reasoning
    chat_template = build_chat_template(tokenizer, all_reasoning=all_reasoning)
    print(f"[pack_dataset] reasoning: {'ALL assistant turns' if all_reasoning else 'stock (last-user tool-call turns only)'}"
          f" | tools: {'OFF' if args.no_tools else 'from functions column'}", flush=True)

    df = load_source_dataframe(args.repo, args.file)
    print(f"[pack_dataset] source rows: {len(df)}; columns: {list(df.columns)}", flush=True)
    if "messages" not in df.columns:
        raise SystemExit("[pack_dataset] ERROR: source parquet has no 'messages' column")
    if not args.no_tools and "functions" not in df.columns:
        print("[pack_dataset] WARN: no 'functions' column — packing without tool definitions",
              flush=True)

    stats = pack(df, tokenizer, args.out, args.max_seq_len, max_rows=args.max_rows,
                 chat_template=chat_template, enable_tools=not args.no_tools)

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
              "(gemma-4 template lacks {% generation %}; assistant mask all-zero) "
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
        # position_ids must reset per doc: each doc segment is 0..L-1.
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
