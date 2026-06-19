"""In-process chat → multipack (ChiniDataset) packer for kind=llm datasets.

The web/gateway analogue of `autotrain/gemma4/pack_dataset.py`: tokenize each chat
conversation (its `messages` array + an optional `functions`/tools column) via the
tokenizer's chat template, then greedily bin-pack documents into fixed-size bins,
writing the SAME ChiniDataset columns the gemma4 trainer (`autotrain/gemma4/gemma4.py`)
consumes:

    input_ids     int64[]   concatenated token ids of every doc in the bin
    labels        int64[]   same length; -100 masks non-trained tokens (assistant-only
                            when the template has `{% generation %}`, else == input_ids)
    position_ids  uint32[]  RESET per document (0..L-1 for each packed doc), concatenated
    attention_mask uint32[] the per-document SEQUENCE LENGTHS within the bin (NOT a dense
                            mask) — e.g. docs of length [120, 88, 300] -> [120, 88, 300].
                            gemma4.py's collator turns these into cu_seqlens.

Invariant asserted for EVERY written bin:
    sum(attention_mask) == len(input_ids) == len(position_ids) == len(labels)

This runs IN-PROCESS in the gateway (no GPU — pure CPU tokenization). Two deps the
gateway otherwise lacks are handled deliberately:
  * `transformers` is imported LAZILY (inside pack_rows). Tokenization + chat templates
    need no torch, so a tokenizer-only transformers works on the torch-less gateway.
  * ChiniDataset's `ParquetWriter` is imported via a torch-free stub (its sibling
    StreamingDataset reader pulls torch; we only need the writer) so the gateway need
    not install torch just to produce the shards.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import types
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("gateway.llm_pack")

IGNORE_INDEX = -100

# ChiniDataset Parquet column encodings — must match the gemma4 trainer's reader.
# int64 for the two columns that may carry -100 / large vocab ids; uint32 for the
# always-non-negative ones.
COLUMNS = {
    "input_ids": "int64[]",
    "labels": "int64[]",
    "position_ids": "uint32[]",
    "attention_mask": "uint32[]",
}

# --- gemma-4 reasoning rendering (optional; no-op on other templates) ---------
# The stock gemma-4-31B-it chat template only emits an assistant's `reasoning`
# (`<|channel>thought ... <channel|>`) on tool-call turns AFTER the last user
# message — silently dropping most reasoning in a multi-user agentic trajectory.
# `all_reasoning=True` relaxes that one guard so EVERY assistant turn's reasoning
# is trained. The swap is a surgical substring replace; if the substring isn't
# present (a non-gemma template, or an upstream template change) we fall back to
# the stock template UNCHANGED rather than raising — this packer is generic.
_REASONING_GUARD_STOCK = (
    "if thinking_text and loop.index0 > ns_turn.last_user_idx "
    "and message.get('tool_calls')"
)
_REASONING_GUARD_ALL = "if thinking_text and role == 'model'"


def _import_parquet_writer():
    """Return ChiniDataset's ``ParquetWriter`` without importing torch.

    The `chinidataset` package's __init__ does `from chinidataset.dataset import
    StreamingDataset`, and that reader subpackage imports torch (absent on the
    gateway). We only need the WRITER (pyarrow/numpy only), so pre-register a
    torch-free stub for the reader subpackage, then import the package — the
    __init__'s `from chinidataset.dataset import StreamingDataset` binds to the
    stub and the real (writer) import path runs untouched."""
    if "chinidataset" not in sys.modules:
        stub = types.ModuleType("chinidataset.dataset")
        stub.StreamingDataset = None  # placeholder; the writer never touches it
        sys.modules.setdefault("chinidataset.dataset", stub)
        # The vendored ChiniDataset lives next to the (shipped) TTS trainer scripts.
        tts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training", "tts")
        if tts_dir not in sys.path:
            sys.path.insert(0, tts_dir)
    from chinidataset import ParquetWriter  # noqa: PLC0415
    return ParquetWriter


def build_chat_template(tokenizer, all_reasoning: bool) -> Optional[str]:
    """Return the chat-template string to render with (stock, or gemma reasoning
    relaxed). Non-fatal: an unrecognised template falls back to stock."""
    stock = getattr(tokenizer, "chat_template", None)
    if not all_reasoning or not stock:
        return stock
    if _REASONING_GUARD_STOCK in stock:
        return stock.replace(_REASONING_GUARD_STOCK, _REASONING_GUARD_ALL)
    logger.info("llm_pack: all_reasoning requested but the gemma-4 reasoning guard "
                "was not found in the chat template — using the stock template.")
    return stock


def extract_messages(value: Any) -> Optional[list]:
    """Normalize a `messages` cell into a list[dict]. HF parquet often stores it
    as a JSON *string*; also tolerate a list / numpy array of dicts."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return None
    out = []
    for turn in value:
        if isinstance(turn, dict) and "role" in turn:
            out.append(dict(turn))
    return out or None


def extract_tools(value: Any) -> list:
    """Normalize a `functions` cell into the OpenAI tool list the chat template
    wants. The source column is typically a JSON *string* of BARE function objects
    (`{"name", "description", "parameters", ...}`); the template accesses
    `tool['function']['name']`, so each bare function is wrapped as
    `{"type": "function", "function": <fn>}`. Returns a (possibly empty) list."""
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
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
    """Render + tokenize one conversation; return (input_ids, labels, used_mask).

    Prefers assistant-only labels via `return_assistant_tokens_mask=True`; falls
    back to `labels = input_ids` when the template lacks `{% generation %}` (mask
    all-zero) or the kwarg is unsupported."""
    kw: dict[str, Any] = {}
    if tools:
        kw["tools"] = tools
    if chat_template is not None:
        kw["chat_template"] = chat_template

    try:
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True,
            return_assistant_tokens_mask=True, **kw,
        )
        ids = list(out["input_ids"])
        mask = out.get("assistant_masks")
        if mask is not None and any(mask):
            labels = [tid if m else IGNORE_INDEX for tid, m in zip(ids, mask)]
            return ids, labels, True
        return ids, list(ids), False
    except Exception:  # noqa: BLE001 — kwarg unsupported by this tokenizer version
        pass

    out = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, **kw)
    ids = list(out["input_ids"])
    return ids, list(ids), False


def collate_bin(docs_ids: list, docs_labels: list) -> dict:
    """Build one packed-bin sample: concatenate ids/labels, reset position_ids per
    doc, store per-doc lengths in attention_mask."""
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


def assert_invariants(sample: dict) -> None:
    n = len(sample["input_ids"])
    assert len(sample["labels"]) == n, "labels length mismatch"
    assert len(sample["position_ids"]) == n, "position_ids length mismatch"
    assert int(np.sum(sample["attention_mask"])) == n, (
        f"sum(attention_mask)={int(np.sum(sample['attention_mask']))} != len(input_ids)={n}"
    )


def pack_rows(
    rows: Any,
    *,
    tokenizer_name: str,
    out_dir: str,
    messages_field: str = "messages",
    tools_field: str = "functions",
    max_seq_len: int = 131072,
    hf_token: Optional[str] = None,
    hf_endpoint: Optional[str] = None,
    all_reasoning: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
    progress_every: int = 100,
) -> dict:
    """Tokenize + greedily multipack `rows` into a ChiniDataset at `out_dir`.

    `rows` is anything index-accessible with len (a list[dict] or a HF Dataset).
    Each row is rendered with its messages AND (when present) its `tools_field`
    column as `tools=`. A conversation longer than `max_seq_len` whole is DROPPED
    (never split across bins — keeps position_ids/labels coherent). Returns a
    stats dict for the run summary/log.

    BLOCKING (CPU-bound): call from a threadpool, not the event loop.
    """
    # Lazy: transformers is not a gateway boot dep. Tokenization needs no torch.
    from transformers import AutoTokenizer  # noqa: PLC0415

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    logger.info("llm_pack: loading tokenizer %s (tokenizer only)", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token or None)
    chat_template = build_chat_template(tokenizer, all_reasoning)

    ParquetWriter = _import_parquet_writer()

    total_rows = len(rows)
    n_bins = n_docs = n_dropped_long = n_dropped_empty = 0
    n_assistant_masked = n_rows_with_tools = total_tokens = 0
    cur_ids: list = []
    cur_labels: list = []
    cur_count = 0

    def _get(row: Any, key: str) -> Any:
        if isinstance(row, dict):
            return row.get(key)
        try:
            return row[key]
        except (KeyError, IndexError, TypeError):
            return None

    # Overwrite any existing dataset dir (a re-run repacks from scratch).
    writer = ParquetWriter(out=out_dir, columns=COLUMNS, exist_ok=True)
    with writer as out:
        for i in range(total_rows):
            row = rows[i]
            messages = extract_messages(_get(row, messages_field))
            if not messages:
                n_dropped_empty += 1
            else:
                tools = extract_tools(_get(row, tools_field)) if tools_field else []
                if tools:
                    n_rows_with_tools += 1
                try:
                    ids, labels, used_mask = tokenize_row(
                        tokenizer, messages, tools=tools, chat_template=chat_template
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("llm_pack: row %d tokenize failed (%s); skipping", i, e)
                    ids = []
                    used_mask = False
                    labels = []

                length = len(ids)
                if length == 0:
                    n_dropped_empty += 1
                elif length > max_seq_len:
                    n_dropped_long += 1  # too long for a single bin — drop, never split
                else:
                    if used_mask:
                        n_assistant_masked += 1
                    # Greedy bin-packing: flush the current bin before it overflows.
                    if cur_count + length > max_seq_len:
                        if cur_ids:
                            sample = collate_bin(cur_ids, cur_labels)
                            assert_invariants(sample)
                            out.write(sample)
                            n_bins += 1
                            total_tokens += len(sample["input_ids"])
                        cur_ids, cur_labels, cur_count = [ids], [labels], length
                    else:
                        cur_ids.append(ids)
                        cur_labels.append(labels)
                        cur_count += length
                    n_docs += 1

            if progress and total_rows and (i + 1) % progress_every == 0:
                progress(i + 1, total_rows)

        # Flush the final partial bin.
        if cur_ids:
            sample = collate_bin(cur_ids, cur_labels)
            assert_invariants(sample)
            out.write(sample)
            n_bins += 1
            total_tokens += len(sample["input_ids"])

    if progress and total_rows:
        progress(total_rows, total_rows)

    efficiency = (total_tokens / n_bins / max_seq_len) if n_bins else 0.0
    return {
        "total_rows": total_rows,
        "docs_packed": n_docs,
        "dropped_long": n_dropped_long,
        "dropped_empty": n_dropped_empty,
        "assistant_masked_rows": n_assistant_masked,
        "rows_with_tools": n_rows_with_tools,
        "n_bins": n_bins,
        "total_tokens": total_tokens,
        "max_seq_len": max_seq_len,
        "efficiency": efficiency,
        "tokenizer": tokenizer_name,
    }
