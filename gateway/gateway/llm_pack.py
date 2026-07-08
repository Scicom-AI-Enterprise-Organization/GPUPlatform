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

# --- reasoning rendering (optional; arch-specific; no-op on unknown templates) -
# Several chat templates only emit an assistant's reasoning on tool-call turns
# AFTER the last user message — silently dropping most reasoning in a multi-user
# agentic trajectory. `all_reasoning=True` relaxes that one guard so EVERY
# assistant turn's reasoning is trained. The swap is a surgical substring replace
# per arch; if no known guard substring is present we fall back to the stock
# template UNCHANGED rather than raising — this packer is generic.
# Each entry: arch -> (stock_guard_substring, relaxed_replacement). Mirrors the
# standalone packers: autotrain/gemma4/pack_dataset.py + autotrain/minimax-m2/pack_dataset.py.
_REASONING_GUARDS: dict[str, tuple[str, str]] = {
    "gemma": (
        "if thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls')",
        "if thinking_text and role == 'model'",
    ),
    "minimax": (
        "if reasoning_content and loop.index0 > ns.last_user_index",
        "if reasoning_content",
    ),
    # Qwen3.5 (legacy) template guards reasoning to turns after the last user query.
    # Qwen3.6 combines it with a `preserve_thinking` clause (so this exact substring is
    # absent) — there build_chat_template falls back to stock and pack_rows passes
    # preserve_thinking=True instead (see tokenize_row). Harmless either way.
    "qwen": (
        "{%- if loop.index0 > ns.last_query_index %}",
        "{%- if loop.index0 >= 0 %}",
    ),
}


def detect_arch(tokenizer_name: Optional[str]) -> str:
    """Map a tokenizer/model id to a packing arch: 'gemma' | 'minimax' | 'mistral' |
    'qwen' | 'generic'. Drives the reasoning relaxation + per-turn message
    normalization. Keep in sync with llm_finetune.detect_arch + training_api._llm_arch."""
    n = (tokenizer_name or "").lower()
    if "minimax" in n:
        return "minimax"
    if "mistral" in n:
        return "mistral"
    if "qwen" in n:
        return "qwen"
    if "gemma" in n:
        return "gemma"
    return "generic"


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


def build_chat_template(tokenizer, all_reasoning: bool, arch: str = "generic") -> Optional[str]:
    """Return the chat-template string to render with (stock, or arch-specific
    reasoning relaxed). Non-fatal: an unrecognised template falls back to stock."""
    stock = getattr(tokenizer, "chat_template", None)
    if not all_reasoning or not stock:
        return stock
    # Prefer this arch's guard; otherwise try any known guard (the arch detection
    # is a name heuristic — a custom finetune name might not match).
    candidates = []
    if arch in _REASONING_GUARDS:
        candidates.append(_REASONING_GUARDS[arch])
    candidates += [g for a, g in _REASONING_GUARDS.items() if a != arch]
    for stock_guard, relaxed in candidates:
        if stock_guard in stock:
            return stock.replace(stock_guard, relaxed)
    logger.info("llm_pack: all_reasoning requested but no known reasoning guard "
                "matched the chat template (arch=%s) — using the stock template.", arch)
    return stock


def _normalize_minimax_turn(turn: dict) -> dict:
    """MiniMax-M2 template quirks (mirrors autotrain/minimax-m2/pack_dataset.py):
    it reads `reasoning_content` (the dataset stores per-turn `reasoning`), and it
    iterates `tool_call.arguments.items()` (the dataset stores `arguments` as a
    JSON STRING → '"str" has no attribute items'). Map + parse so the row renders."""
    turn = dict(turn)
    if turn.get("role") in ("assistant", "model") and turn.get("reasoning") and not turn.get("reasoning_content"):
        turn["reasoning_content"] = turn["reasoning"]
    tcs = turn.get("tool_calls")
    if isinstance(tcs, (list, tuple)):
        norm = []
        for tc in tcs:
            if isinstance(tc, dict):
                tc = dict(tc)
                holders = [tc]
                if isinstance(tc.get("function"), dict):
                    tc["function"] = dict(tc["function"])
                    holders.append(tc["function"])
                for holder in holders:
                    a = holder.get("arguments")
                    if isinstance(a, str):
                        try:
                            holder["arguments"] = json.loads(a)
                        except (json.JSONDecodeError, ValueError):
                            holder["arguments"] = {}
                    elif a is None and "arguments" in holder:
                        # explicit null → {} (the template does `arguments.items()`).
                        holder["arguments"] = {}
            norm.append(tc)
        turn["tool_calls"] = norm
    return turn


def _normalize_qwen_turn(turn: dict) -> dict:
    """Qwen3.5/3.6 template quirks (mirrors autotrain/qwen3.5/pack_dataset.py):
    - GLM `role: observation` → `tool`; `content: null` → `""` (null crashes the template).
    - The template reads `reasoning_content` (the dataset stores per-turn `reasoning`) to
      render `<think>…</think>` — map it on assistant turns.
    - It iterates `tool_call.function.arguments | items`, so `arguments` MUST be a plain
      dict, never a JSON string — parse it; wrap a bare GLM `{name, arguments}` call into
      the OpenAI `{type:function, function:{…}}` shape."""
    turn = dict(turn)
    if turn.get("role") == "observation":
        turn["role"] = "tool"
    if turn.get("content") is None:
        turn["content"] = ""
    if turn.get("role") in ("assistant", "model") and turn.get("reasoning") and not turn.get("reasoning_content"):
        turn["reasoning_content"] = turn["reasoning"]
    tcs = turn.get("tool_calls")
    if isinstance(tcs, (list, tuple)):
        norm = []
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            tc = dict(tc)
            fn = dict(tc["function"]) if isinstance(tc.get("function"), dict) else {
                "name": tc.get("name"), "arguments": tc.get("arguments", {})}
            a = fn.get("arguments")
            if isinstance(a, str):
                try:
                    fn["arguments"] = json.loads(a)
                except (json.JSONDecodeError, ValueError):
                    fn["arguments"] = {}
            elif a is None:
                fn["arguments"] = {}
            tc["function"] = fn
            norm.append(tc)
        turn["tool_calls"] = norm
    return turn


def _as_thinking_content(content: Any, reasoning: Any) -> list:
    """Rewrite an assistant turn's content into a [thinking, text] block list so the
    Mistral-Small-4 template renders `[THINK]<reasoning>[/THINK]<text>`."""
    blocks: list = [{"type": "thinking", "thinking": str(reasoning)}]
    if isinstance(content, str):
        if content != "":
            blocks.append({"type": "text", "text": content})
    elif isinstance(content, (list, tuple)):
        blocks.extend(content)
    return blocks


def _normalize_mistral_turn(turn: dict, all_reasoning: bool) -> dict:
    """Mistral-Small-4 template quirks (mirrors autotrain/mistral-small/pack_dataset.py):
    role 'model'→'assistant'; when all_reasoning, fold the flat `reasoning` field into a
    `[THINK]` content block (the template renders thinking only from a structured content
    list); normalise tool_calls to the nested `function.{name,arguments}` shape it reads."""
    turn = dict(turn)
    if turn.get("role") == "model":
        turn["role"] = "assistant"
    if all_reasoning and turn.get("role") == "assistant" and turn.get("reasoning"):
        turn["content"] = _as_thinking_content(turn.get("content"), turn["reasoning"])
    turn.pop("reasoning", None)
    turn.pop("reasoning_content", None)  # not read by this template
    tcs = turn.get("tool_calls")
    if isinstance(tcs, (list, tuple)):
        norm = []
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            tc = dict(tc)
            fn = dict(tc["function"]) if isinstance(tc.get("function"), dict) else {
                "name": tc.get("name"), "arguments": tc.get("arguments", {})}
            if fn.get("arguments") is None:
                fn["arguments"] = {}
            tc["function"] = fn
            norm.append(tc)
        turn["tool_calls"] = norm
    return turn


def _normalize_gemma_turn(turn: dict) -> dict:
    """Gemma-4 template quirks: it reads `tool_call['function']['name']` and renders
    `function['arguments']` in its NATIVE `key:<|"|>value<|"|>` form ONLY when arguments
    is a *mapping* — a JSON-*string* cell is dumped verbatim as raw JSON (wrong format to
    train on), and a bare `{name, arguments}` tool_call with no `function` wrapper
    KeyErrors the template → the row is silently dropped. So parse `arguments` str→dict
    and wrap bare calls into the OpenAI `{type:function, function:{…}}` shape. (Reasoning
    needs no mapping — the template reads `reasoning`/`reasoning_content` directly.)"""
    turn = dict(turn)
    tcs = turn.get("tool_calls")
    if isinstance(tcs, (list, tuple)):
        norm = []
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            tc = dict(tc)
            fn = dict(tc["function"]) if isinstance(tc.get("function"), dict) else {
                "name": tc.get("name"), "arguments": tc.get("arguments", {})}
            a = fn.get("arguments")
            if isinstance(a, str):
                try:
                    fn["arguments"] = json.loads(a)
                except (json.JSONDecodeError, ValueError):
                    fn["arguments"] = {}
            elif a is None:
                fn["arguments"] = {}
            tc["function"] = fn
            norm.append(tc)
        turn["tool_calls"] = norm
    return turn


def extract_messages(value: Any, arch: str = "generic", all_reasoning: bool = True) -> Optional[list]:
    """Normalize a `messages` cell into a list[dict]. HF parquet often stores it as a
    JSON *string*; also tolerate a list / numpy array of dicts. Apply the arch-specific
    per-turn normalization (minimax: reasoning_content + tool-arg parse; mistral: [THINK]
    folding + tool-call shape)."""
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
        if not (isinstance(turn, dict) and "role" in turn):
            continue
        if arch == "minimax":
            out.append(_normalize_minimax_turn(turn))
        elif arch == "mistral":
            out.append(_normalize_mistral_turn(turn, all_reasoning))
        elif arch == "qwen":
            out.append(_normalize_qwen_turn(turn))
        elif arch == "gemma":
            out.append(_normalize_gemma_turn(turn))
        else:
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


def tokenize_row(tokenizer, messages, tools=None, chat_template=None, reasoning_effort=None,
                 preserve_thinking=False):
    """Render + tokenize one conversation; return (input_ids, labels, used_mask).

    Prefers assistant-only labels via `return_assistant_tokens_mask=True`; falls
    back to `labels = input_ids` when the template lacks `{% generation %}` (mask
    all-zero) or the kwarg is unsupported. `reasoning_effort` (mistral) is passed
    to the template only when set. `preserve_thinking` (qwen3.6) makes the template
    render EVERY assistant turn's `<think>` block (templates that don't reference the
    variable ignore it)."""
    kw: dict[str, Any] = {}
    if tools:
        kw["tools"] = tools
    if reasoning_effort is not None:
        kw["reasoning_effort"] = reasoning_effort
    if preserve_thinking:
        kw["preserve_thinking"] = True
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


def extract_pair_messages(row_get: Callable[[str], Any], *, chosen_field: str,
                          rejected_field: str, prompt_field: Optional[str],
                          arch: str, all_reasoning: bool) -> Optional[tuple[list, list, list]]:
    """Normalize one preference row into (prompt_msgs, chosen_msgs, rejected_msgs),
    each a full message list (chosen/rejected INCLUDE the prompt turns + the final
    assistant turn). Two source shapes are accepted:
      * chosen/rejected are message lists sharing the prompt turns (ultrafeedback-
        binarized style) — the prompt is everything before the final assistant turn;
      * chosen/rejected are plain strings + a `prompt_field` column (a string or a
        messages list) holding the shared prompt.
    Returns None when the row doesn't parse (caller counts it dropped)."""
    def _as_prompt_msgs(value: Any) -> Optional[list]:
        msgs = extract_messages(value, arch, all_reasoning)
        if msgs:
            return msgs
        if isinstance(value, str) and value.strip():
            return [{"role": "user", "content": value}]
        return None

    c_raw, r_raw = row_get(chosen_field), row_get(rejected_field)
    if isinstance(c_raw, str) and isinstance(r_raw, str):
        if not prompt_field:
            return None
        prompt = _as_prompt_msgs(row_get(prompt_field))
        if not prompt or not c_raw.strip() or not r_raw.strip():
            return None
        chosen = prompt + [{"role": "assistant", "content": c_raw}]
        rejected = prompt + [{"role": "assistant", "content": r_raw}]
        return prompt, chosen, rejected

    chosen = extract_messages(c_raw, arch, all_reasoning)
    rejected = extract_messages(r_raw, arch, all_reasoning)
    if not chosen or not rejected:
        return None
    if chosen[-1].get("role") not in ("assistant", "model") or \
            rejected[-1].get("role") not in ("assistant", "model"):
        return None
    prompt = chosen[:-1]
    # The pair must share its prompt turns — otherwise the DPO log-ratio compares
    # apples to oranges. (Content compare, not identity: rows often duplicate them.)
    if len(rejected) != len(chosen) or \
            json.dumps(prompt, sort_keys=True, default=str) != \
            json.dumps(rejected[:-1], sort_keys=True, default=str):
        return None
    if not prompt:
        return None
    return prompt, chosen, rejected


def _common_prefix_len(a: list, b: list) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def tokenize_pair(tokenizer, prompt_msgs, chosen_msgs, rejected_msgs,
                  chat_template=None, **kw) -> Optional[tuple[list, list, list, list]]:
    """Tokenize one preference pair → (chosen_ids, chosen_targets, rejected_ids,
    rejected_targets), with targets PRE-ALIGNED per the fused-DPO contract:
    targets[j] = ids[j+1] on response positions, IGNORE_INDEX on prompt tokens and
    the final position — no shifting happens at loss time.

    The prompt/response boundary comes from rendering the prompt turns with
    add_generation_prompt=True and checking it prefixes both full renders; templates
    that aren't prefix-stable fall back to the longest common prefix of the two full
    renders (equivalent for DPO: any shared-prefix tokens contribute identical
    log-probs to both sides and cancel in the pairwise loss)."""
    if chat_template is not None:
        kw["chat_template"] = chat_template

    def _render(msgs, **extra):
        # return_dict=True like tokenize_row — transformers 5.x returns a dict
        # (not a bare id list) from apply_chat_template.
        out = tokenizer.apply_chat_template(msgs, tokenize=True, return_dict=True, **kw, **extra)
        return list(out["input_ids"])

    c_ids = _render(chosen_msgs)
    r_ids = _render(rejected_msgs)
    try:
        p_ids = _render(prompt_msgs, add_generation_prompt=True)
    except Exception:  # noqa: BLE001 — template without generation-prompt support
        p_ids = []
    if p_ids and c_ids[:len(p_ids)] == p_ids and r_ids[:len(p_ids)] == p_ids:
        prompt_len = len(p_ids)
    else:
        prompt_len = _common_prefix_len(c_ids, r_ids)
    # Need >=1 trained (response) token per side, and a non-empty prompt.
    if prompt_len < 1 or len(c_ids) <= prompt_len or len(r_ids) <= prompt_len:
        return None

    def _targets(ids: list) -> list:
        t = [IGNORE_INDEX] * len(ids)
        for j in range(prompt_len - 1, len(ids) - 1):
            t[j] = ids[j + 1]
        return t

    return c_ids, _targets(c_ids), r_ids, _targets(r_ids)


def collate_dpo_bin(pairs: list) -> dict:
    """One packed DPO bin: K pairs → 2K docs laid out FIRST K CHOSEN then K rejected
    (the triton_dpo.fused_dpo_loss contract — pair k is (doc k, doc K+k)). Columns
    are the standard ChiniDataset layout; `labels` holds the pre-aligned targets."""
    docs_ids = [p[0] for p in pairs] + [p[2] for p in pairs]
    docs_labels = [p[1] for p in pairs] + [p[3] for p in pairs]
    return collate_bin(docs_ids, docs_labels)


def pack_dpo_rows(
    rows: Any,
    *,
    tokenizer_name: str,
    out_dir: str,
    chosen_field: str = "chosen",
    rejected_field: str = "rejected",
    prompt_field: Optional[str] = None,
    max_seq_len: int = 131072,
    hf_token: Optional[str] = None,
    hf_endpoint: Optional[str] = None,
    all_reasoning: bool = True,
    arch: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    progress_every: int = 100,
) -> dict:
    """Tokenize + greedily multipack preference pairs into a DPO ChiniDataset at
    `out_dir` (kind=llm_dpo_packed). Same columns/invariants as pack_rows, plus:
    every bin holds WHOLE pairs, doc count is even, and the first half of each
    bin's docs are the chosen responses (see collate_dpo_bin). A pair whose
    chosen+rejected renders don't both fit one bin is dropped (never split).

    BLOCKING (CPU-bound): call from a threadpool, not the event loop.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415 — lazy, like pack_rows

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    arch = arch or detect_arch(tokenizer_name)
    logger.info("llm_pack: DPO pack — loading tokenizer %s (arch=%s)", tokenizer_name, arch)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token or None)
    chat_template = build_chat_template(tokenizer, all_reasoning, arch)
    tpl_kw: dict[str, Any] = {}
    if arch == "mistral":
        tpl_kw["reasoning_effort"] = "high" if all_reasoning else "none"
    if all_reasoning and arch == "qwen":
        tpl_kw["preserve_thinking"] = True

    ParquetWriter = _import_parquet_writer()

    total_rows = len(rows)
    n_bins = n_pairs = n_dropped_long = n_dropped_bad = 0
    total_tokens = 0
    cur_pairs: list = []
    cur_count = 0

    def _get_factory(row: Any) -> Callable[[str], Any]:
        def _get(key: str) -> Any:
            if isinstance(row, dict):
                return row.get(key)
            try:
                return row[key]
            except (KeyError, IndexError, TypeError):
                return None
        return _get

    writer = ParquetWriter(out=out_dir, columns=COLUMNS, exist_ok=True)
    with writer as out:
        for i in range(total_rows):
            trip = extract_pair_messages(
                _get_factory(rows[i]), chosen_field=chosen_field,
                rejected_field=rejected_field, prompt_field=prompt_field,
                arch=arch, all_reasoning=all_reasoning)
            pair = None
            if trip is not None:
                try:
                    pair = tokenize_pair(tokenizer, *trip, chat_template=chat_template, **tpl_kw)
                except Exception as e:  # noqa: BLE001
                    logger.warning("llm_pack: DPO row %d tokenize failed (%s); skipping", i, e)
            if pair is None:
                n_dropped_bad += 1
            else:
                length = len(pair[0]) + len(pair[2])
                if length > max_seq_len:
                    n_dropped_long += 1  # the pair must fit one bin whole
                else:
                    if cur_count + length > max_seq_len:
                        if cur_pairs:
                            sample = collate_dpo_bin(cur_pairs)
                            assert_invariants(sample)
                            out.write(sample)
                            n_bins += 1
                            total_tokens += len(sample["input_ids"])
                        cur_pairs, cur_count = [pair], length
                    else:
                        cur_pairs.append(pair)
                        cur_count += length
                    n_pairs += 1

            if progress and total_rows and (i + 1) % progress_every == 0:
                progress(i + 1, total_rows)

        if cur_pairs:
            sample = collate_dpo_bin(cur_pairs)
            assert_invariants(sample)
            out.write(sample)
            n_bins += 1
            total_tokens += len(sample["input_ids"])

    if progress and total_rows:
        progress(total_rows, total_rows)

    efficiency = (total_tokens / n_bins / max_seq_len) if n_bins else 0.0
    return {
        "total_rows": total_rows,
        "pairs_packed": n_pairs,
        "docs_packed": 2 * n_pairs,
        "dropped_long": n_dropped_long,
        "dropped_empty": n_dropped_bad,
        "n_bins": n_bins,
        "total_tokens": total_tokens,
        "max_seq_len": max_seq_len,
        "efficiency": efficiency,
        "tokenizer": tokenizer_name,
        "arch": arch,
        "objective": "dpo",
    }


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
    arch: Optional[str] = None,
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
    arch = arch or detect_arch(tokenizer_name)
    logger.info("llm_pack: loading tokenizer %s (arch=%s, tokenizer only)", tokenizer_name, arch)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token or None)
    chat_template = build_chat_template(tokenizer, all_reasoning, arch)
    # mistral relaxes reasoning by folding [THINK] blocks in extract_messages (not a
    # template guard swap) AND passing reasoning_effort to the template.
    reasoning_effort = ("high" if all_reasoning else "none") if arch == "mistral" else None
    # qwen3.6's template renders every turn's reasoning when preserve_thinking=True
    # (its guard is combined with a preserve_thinking clause, so the substring swap in
    # build_chat_template is a no-op there — this kwarg is the real control).
    preserve_thinking = bool(all_reasoning and arch == "qwen")

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
            messages = extract_messages(_get(row, messages_field), arch, all_reasoning)
            if not messages:
                n_dropped_empty += 1
            else:
                tools = extract_tools(_get(row, tools_field)) if tools_field else []
                if tools:
                    n_rows_with_tools += 1
                try:
                    ids, labels, used_mask = tokenize_row(
                        tokenizer, messages, tools=tools, chat_template=chat_template,
                        reasoning_effort=reasoning_effort, preserve_thinking=preserve_thinking,
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
        "arch": arch,
    }
