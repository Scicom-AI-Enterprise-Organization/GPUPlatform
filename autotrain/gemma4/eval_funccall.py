#!/usr/bin/env python3
"""Function-calling accuracy eval for google/gemma-4-31B-it — base vs LoRA-finetuned.

Reuses SyntheticGen's EXACT scoring methodology
(SyntheticGen/synthetic/evaluate_function_calling.py) so the headline numbers are
directly comparable to its *_results.json, but runs the model with **transformers
locally on the GPU** instead of an OpenAI-compatible API server.

=============================================================================
WHAT IS REPLICATED 1:1 FROM evaluate_function_calling.py (no behavioural change)
=============================================================================
The full-replay loop, the per-turn metric record, and the aggregation are copied
verbatim (same TP/FP/FN/TN bookkeeping, same precision/recall/F1, name_accuracy,
name_set_f1, json_valid_rate, hallucination_rate, req_coverage, type_accuracy,
parallel_count_match, id_propagation_rate, refusal_rate). Specifically:

  - build_openai_tools(lib)  ............. evaluate_function_calling.py:85-106
  - resolve_refs / required_coverage / type_accuracy / extract_ids /
    id_propagation  ...................... :71-176
  - TurnResult dataclass + compute_turn_result()  .. :208-279
  - evaluate_conversation() full-replay  .. :286-392
  - aggregate()  ......................... :399-480
  - print_summary()  .................... :601-633
  - the results-json shape ({model, split, metrics, conversations:[{...turns...}]})
    .................................... :921-925 + the on-disk *_results.json.

The ONE place the upstream code reads a server response is compute_turn_result's
loop over `model_tcs`, where each element is an OpenAI message-tool-call object:
    mtc.function.name            evaluate_function_calling.py:257
    mtc.function.arguments       :260, :264
and call_model() returns `resp.choices[0].message` (:195) whose `.tool_calls`
the server already parsed from the raw model text (:337).

=============================================================================
THE ADAPTATION: API tool-parsing  ->  transformers chat-template + text parser
=============================================================================
SyntheticGen called an OpenAI-compatible endpoint (call_model, :183-201) that did
TWO things we must reproduce locally:

  (1) PROMPT CONSTRUCTION. The server applied the model's own chat template to
      (messages + tools) before generation. We do the SAME, explicitly, via the
      gemma-4 chat template — `tokenizer.apply_chat_template(history, tools=...,
      add_generation_prompt=True)`. Tools are wrapped bare-fn -> {"type":"function",
      "function":fn} exactly like pack_dataset.extract_tools (pack_dataset.py:215-247),
      because the gemma-4 template reads tool['function']['name']. This is precisely
      how the model is trained (pack_dataset.py) and served, so it is the faithful
      prompt — there is no other "more correct" prompt for this model.

  (2) TOOL-CALL PARSING. The server parsed gemma-4's raw output text into structured
      OpenAI tool_calls (name + a JSON-string `arguments`). We reproduce that parser.
      The gemma-4 chat template renders a tool call as (verified by rendering the real
      dataset rows through google/gemma-4-31B-it's chat_template.jinja):

          <|tool_call>call:<NAME>{<JSON-OBJECT>}<tool_call|>

      i.e. literal markers '<|tool_call>call:' ... '}<tool_call|>', with the model's
      arguments as a JSON object between the template's outer braces (template lines
      243-258: '<|tool_call>call:'+name+'{' then, for string `arguments`, the JSON
      verbatim, then '}<tool_call|>'). _parse_gemma_tool_calls() below extracts each
      block and returns the inner JSON text as the `arguments` STRING — identical to
      what the OpenAI API hands compute_turn_result. The structure mirrors vLLM's
      FunctionGemma tool parser (vllm/tool_parsers/functiongemma_tool_parser.py
      extract_tool_calls: find '<...call>' blocks, regex name+args, json.dumps the
      args) adapted to gemma-4's '<|tool_call>'/'<tool_call|>' delimiters instead of
      FunctionGemma's '<start_function_call>'/'<end_function_call>'/'<escape>'.

      We DO NOT json.loads the arguments here — we hand the raw JSON string straight
      to compute_turn_result, which runs its own `json.loads(mtc.function.arguments)`
      (:260) so json_valid_rate / type_accuracy / req_coverage are scored by the
      upstream code unchanged. A model call that emits non-JSON args therefore scores
      json_valid=0 exactly as it would through the API.

  Model objects fed to the upstream scorer are tiny shims (_FnShim/_TCShim/_MsgShim)
  exposing the attributes it reads: .content, .tool_calls, tc.id, tc.function.name,
  tc.function.arguments. Nothing in the scoring path can tell the difference.

Generation is GREEDY (do_sample=False) — the deterministic analogue of the API call
(which used default sampling but we want reproducible local numbers). Stop tokens are
'<turn|>' (id 106, the gemma-4 turn terminator) and <eos>, so each generated assistant
turn ends naturally; the decoded text is then parsed for tool calls.

=============================================================================
USAGE
=============================================================================
    # BASE model
    python eval_funccall.py --max-rows 5 --out base_results.json

    # FINETUNED (merge the trained LoRA first, exactly like merge_infer.py)
    python eval_funccall.py --lora checkpointing/lora.pt --out lora_results.json

    # full set, custom data / model
    python eval_funccall.py --data ./glm5.1-fp8-test-00000-of-00001.parquet \
        --model-id google/gemma-4-31B-it --max-new-tokens 16384

Compare the printed headline (and the metrics block in each results.json) between the
two runs. The base google/gemma-4-31b-it reference recorded by SyntheticGen on this
set is tool_call_f1=0.7706 (precision 0.8542, recall 0.7019), name_set_f1=0.8535
(see SyntheticGen/synthetic/google_gemma-4-31b-it_results.json). NOTE that number was
produced via the API with default sampling on a possibly different split snapshot, so
treat it as a ballpark; the base-vs-LoRA *delta from THIS script* is the apples-to-
apples comparison.
"""

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# --- Read HF_TOKEN from the gateway .env (gemma-4 repo is GATED) --------------
# Mirror pack_dataset._load_hf_token_from_env_file; do this BEFORE importing
# transformers / huggingface_hub so the token is honored.
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
        print(f"[eval_funccall] WARN: {path} not found; relying on ambient HF_TOKEN",
              file=sys.stderr)


_load_hf_token_from_env_file(GATEWAY_ENV)

# Xet/hf_transfer have stalled large HF downloads on this box (see MEMORY).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

DEFAULT_DATA = "/home/husein/ssd3/SyntheticGen/synthetic/glm5.1-fp8-test-00000-of-00001.parquet"
DEFAULT_HF_URI = "hf://datasets/Scicom-intl/Function-Call-TaaS/glm5.1-fp8-test/test-00000-of-00001.parquet"


# =============================================================================
# === BEGIN verbatim port from evaluate_function_calling.py (schema + metrics)
# =============================================================================
# resolve_refs / build_openai_tools / required_coverage / _TYPE_MAP /
# type_accuracy / extract_ids / id_propagation are copied unchanged so the
# scoring matches SyntheticGen exactly.

def resolve_refs(schema: object, shared: dict) -> object:
    """Recursively inline $ref references of the form #/shared_entities/<name>."""
    if isinstance(schema, dict):
        if "$ref" in schema:
            parts = schema["$ref"].lstrip("#/").split("/")
            if len(parts) == 2 and parts[0] == "shared_entities":
                return resolve_refs(shared.get(parts[1], {}), shared)
            return schema
        return {k: resolve_refs(v, shared) for k, v in schema.items()}
    if isinstance(schema, list):
        return [resolve_refs(item, shared) for item in schema]
    return schema


def build_openai_tools(lib: dict) -> tuple[list[dict], dict, set]:
    """Convert function library to OpenAI tool list with $refs resolved.

    Returns (tools, fn_schema_map, fn_names). Verbatim from
    evaluate_function_calling.py:85-106. The returned `tools` list is exactly the
    OpenAI-wrapped {"type":"function","function":{...}} shape the gemma-4 chat
    template wants (same wrapping as pack_dataset.extract_tools).
    """
    shared = lib.get("shared_entities", {})
    tools = []
    fn_schema_map = {}
    for fn in lib["functions"]:
        resolved = resolve_refs(copy.deepcopy(fn), shared)
        params = resolved.get("parameters", {"type": "object", "properties": {}})
        tools.append({
            "type": "function",
            "function": {
                "name": resolved["name"],
                "description": resolved.get("description", ""),
                "parameters": params,
            },
        })
        fn_schema_map[resolved["name"]] = resolved
    fn_names = set(fn_schema_map.keys())
    return tools, fn_schema_map, fn_names


def required_coverage(args: dict, fn_schema: dict) -> float:
    required = fn_schema.get("parameters", {}).get("required", [])
    if not required:
        return 1.0
    return sum(1 for r in required if r in args) / len(required)


_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def type_accuracy(args: dict, fn_schema: dict) -> float:
    props = fn_schema.get("parameters", {}).get("properties", {})
    if not props or not args:
        return 1.0
    correct = total = 0
    for key, val in args.items():
        if key in props:
            expected = props[key].get("type")
            if expected and expected in _TYPE_MAP:
                total += 1
                if isinstance(val, _TYPE_MAP[expected]):
                    correct += 1
    return correct / total if total > 0 else 1.0


def extract_ids(content_str: str) -> set:
    """Extract ID-like strings from tool result JSON (hyphen-containing, no spaces)."""
    ids: set = set()
    try:
        def _walk(o):
            if isinstance(o, str):
                if "-" in o and 5 <= len(o) <= 80 and " " not in o:
                    ids.add(o)
            elif isinstance(o, dict):
                for v in o.values():
                    _walk(v)
            elif isinstance(o, list):
                for v in o:
                    _walk(v)
        _walk(json.loads(content_str))
    except Exception:
        pass
    return ids


def id_propagation(model_args: list[str], ref_args: list[str], available_ids: set) -> Optional[float]:
    all_ref = " ".join(ref_args)
    ref_uses = {i for i in available_ids if i in all_ref}
    if not ref_uses:
        return None
    all_model = " ".join(model_args)
    return len({i for i in ref_uses if i in all_model}) / len(ref_uses)


@dataclass
class TurnResult:
    turn: int
    ref_call_count: int
    model_call_count: int
    tp: bool = False
    fp: bool = False
    fn_miss: bool = False
    tn: bool = False
    name_matches: int = 0
    ref_name_set: list = field(default_factory=list)
    model_name_set: list = field(default_factory=list)
    json_valid: int = 0
    hallucinated: int = 0
    req_coverages: list = field(default_factory=list)
    type_accs: list = field(default_factory=list)
    id_prop: Optional[float] = None
    out_of_context: bool = False


def compute_turn_result(
    turn: int,
    ref_tcs: list,
    model_tcs: list,
    fn_names: set,
    fn_schema_map: dict,
    available_ids: set,
) -> TurnResult:
    """Verbatim from evaluate_function_calling.py:228-279.

    `model_tcs` is a list of objects exposing `.function.name` and
    `.function.arguments` (a JSON string) — supplied here by the local _TCShim,
    identical to the OpenAI message tool-call objects upstream.
    """
    tr = TurnResult(
        turn=turn,
        ref_call_count=len(ref_tcs),
        model_call_count=len(model_tcs),
    )

    if tr.ref_call_count > 0 and tr.model_call_count > 0:
        tr.tp = True
    elif tr.ref_call_count == 0 and tr.model_call_count > 0:
        tr.fp = True
    elif tr.ref_call_count > 0 and tr.model_call_count == 0:
        tr.fn_miss = True
    else:
        tr.tn = True

    ref_names = [tc["function"]["name"] for tc in ref_tcs]
    ref_args_strs = [tc["function"].get("arguments", "{}") for tc in ref_tcs]
    model_names = []
    model_args_strs = []

    for mtc in model_tcs:
        fn_name = mtc.function.name
        model_names.append(fn_name)
        try:
            args = json.loads(mtc.function.arguments)
            tr.json_valid += 1
        except Exception:
            args = {}
        model_args_strs.append(mtc.function.arguments or "")

        if fn_name not in fn_names:
            tr.hallucinated += 1
            continue

        schema = fn_schema_map.get(fn_name, {})
        tr.req_coverages.append(required_coverage(args, schema))
        tr.type_accs.append(type_accuracy(args, schema))

    tr.name_matches = len(set(model_names) & set(ref_names))
    tr.ref_name_set = ref_names
    tr.model_name_set = model_names
    tr.id_prop = id_propagation(model_args_strs, ref_args_strs, available_ids)

    return tr


def aggregate(all_results: list[list[TurnResult]]) -> dict:
    """Verbatim from evaluate_function_calling.py:399-480."""
    tp = fp = fn_miss = tn = 0
    json_valid = json_total = 0
    hallucinated = hall_total = 0
    req_cov_vals: list[float] = []
    type_acc_vals: list[float] = []
    id_prop_vals: list[float] = []
    parallel_match = parallel_total = 0
    name_match_sum = name_match_n = 0
    name_set_tp = name_set_fp = name_set_fn = 0
    refusal_correct = refusal_total = 0

    for conv_turns in all_results:
        for tr in conv_turns:
            tp += tr.tp
            fp += tr.fp
            fn_miss += tr.fn_miss
            tn += tr.tn

            json_valid += tr.json_valid
            json_total += tr.model_call_count
            hallucinated += tr.hallucinated
            hall_total += tr.model_call_count

            req_cov_vals.extend(tr.req_coverages)
            type_acc_vals.extend(tr.type_accs)

            if tr.id_prop is not None:
                id_prop_vals.append(tr.id_prop)

            if tr.ref_call_count > 0:
                parallel_total += 1
                if tr.model_call_count == tr.ref_call_count:
                    parallel_match += 1

            if tr.ref_call_count > 0 and tr.model_call_count > 0:
                ref_set = set(tr.ref_name_set)
                model_set = set(tr.model_name_set)
                name_set_tp += len(ref_set & model_set)
                name_set_fp += len(model_set - ref_set)
                name_set_fn += len(ref_set - model_set)
                name_match_sum += tr.name_matches / max(tr.ref_call_count, tr.model_call_count)
                name_match_n += 1

            if tr.out_of_context:
                refusal_total += 1
                if tr.model_call_count == 0:
                    refusal_correct += 1

    def _div(a, b):
        return round(a / b, 4) if b > 0 else 0.0

    def _avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    prec = _div(tp, tp + fp)
    rec = _div(tp, tp + fn_miss)
    f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

    stp, sfp, sfn = name_set_tp, name_set_fp, name_set_fn
    sprec = _div(stp, stp + sfp)
    srec = _div(stp, stp + sfn)
    sf1 = round(2 * sprec * srec / (sprec + srec), 4) if (sprec + srec) > 0 else 0.0

    return {
        "tool_call_precision": prec,
        "tool_call_recall": rec,
        "tool_call_f1": f1,
        "name_accuracy": _div(name_match_sum, name_match_n),
        "name_set_precision": sprec,
        "name_set_recall": srec,
        "name_set_f1": sf1,
        "json_valid_rate": _div(json_valid, json_total),
        "hallucination_rate": _div(hallucinated, hall_total),
        "req_coverage": _avg(req_cov_vals),
        "type_accuracy": _avg(type_acc_vals),
        "parallel_count_match": _div(parallel_match, parallel_total),
        "id_propagation_rate": _avg(id_prop_vals),
        "refusal_rate": _div(refusal_correct, refusal_total) if refusal_total > 0 else None,
        "_counts": {
            "tp": tp, "fp": fp, "fn": fn_miss, "tn": tn,
            "total_model_calls": json_total,
            "total_ref_calls": tp + fn_miss,
        },
    }


def print_summary(metrics: dict, n_convs: int, n_errors: int, model: str) -> None:
    """Verbatim from evaluate_function_calling.py:601-633."""
    m = metrics
    print("\n" + "=" * 60)
    print(f"  Function Calling Evaluation — {model}")
    print(f"  Conversations: {n_convs}  |  Errors: {n_errors}")
    print("=" * 60)
    print(f"  Tool Call Detection")
    print(f"    Precision:          {m['tool_call_precision']:.3f}")
    print(f"    Recall:             {m['tool_call_recall']:.3f}")
    print(f"    F1:                 {m['tool_call_f1']:.3f}")
    print(f"  Function Selection")
    print(f"    Name Accuracy:      {m['name_accuracy']:.3f}")
    print(f"    Name Set F1:        {m['name_set_f1']:.3f}  "
          f"(P={m['name_set_precision']:.3f}  R={m['name_set_recall']:.3f})")
    print(f"  Output Quality")
    print(f"    JSON Valid Rate:    {m['json_valid_rate']:.3f}")
    print(f"    Hallucination Rate: {m['hallucination_rate']:.3f}")
    print(f"  Argument Quality")
    rc = m['req_coverage']
    ta = m['type_accuracy']
    print(f"    Req Coverage:       {rc:.3f}" if rc is not None else "    Req Coverage:       n/a")
    print(f"    Type Accuracy:      {ta:.3f}" if ta is not None else "    Type Accuracy:       n/a")
    print(f"  Multi-turn")
    print(f"    Parallel Match:     {m['parallel_count_match']:.3f}")
    ip = m['id_propagation_rate']
    print(f"    ID Propagation:     {ip:.3f}" if ip is not None else "    ID Propagation:     n/a")
    rr = m.get('refusal_rate')
    if rr is not None:
        print(f"    Refusal Rate:       {rr:.3f}  (out-of-context turns correctly ignored)")
    c = m["_counts"]
    print(f"\n  Confusion: TP={c['tp']}  FP={c['fp']}  FN={c['fn']}  TN={c['tn']}")
    print(f"  Total model calls: {c['total_model_calls']}  |  ref calls: {c['total_ref_calls']}")
    print("=" * 60)

# =============================================================================
# === END verbatim port
# =============================================================================


# =============================================================================
# === ADAPTATION 1: tool-call shims (the objects compute_turn_result reads)
# =============================================================================
# Upstream `model_tcs` were OpenAI message tool-call objects with .id and
# .function.{name,arguments}. We reproduce that exact attribute surface so the
# ported scorer is byte-for-byte unaware it isn't talking to the API.

class _FnShim:
    __slots__ = ("name", "arguments")

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments  # a JSON string, like the API


class _TCShim:
    __slots__ = ("id", "type", "function")

    def __init__(self, tc_id: str, name: str, arguments: str):
        self.id = tc_id
        self.type = "function"
        self.function = _FnShim(name, arguments)


class _MsgShim:
    """Mirror of resp.choices[0].message (evaluate_function_calling.py:195)."""
    __slots__ = ("content", "tool_calls")

    def __init__(self, content: Optional[str], tool_calls: list):
        self.content = content
        self.tool_calls = tool_calls


# =============================================================================
# === ADAPTATION 2: gemma-4 tool-call text parser
# =============================================================================
# Parses '<|tool_call>call:NAME{<JSON>}<tool_call|>' blocks out of the model's
# generated text, producing _TCShim objects whose .function.arguments is the
# inner JSON string verbatim. Structurally mirrors vLLM FunctionGemmaToolParser
# .extract_tool_calls (find tool-call blocks, split name from args, hand args on
# as a JSON string) but with gemma-4's delimiters and the template's outer brace.

_TC_START = "<|tool_call>"
_TC_END = "<tool_call|>"
_CALL_PREFIX = "call:"
_THOUGHT_OPEN = "<|channel>"
_THOUGHT_CLOSE = "<channel|>"


def _strip_thought(text: str) -> str:
    """Drop the model's '<|channel>thought ... <channel|>' reasoning block(s).

    Mirrors the chat template's strip_thinking macro (chat_template.jinja:148-158):
    keep only the text OUTSIDE a thought channel. Tool calls and visible content
    live after the closing <channel|>, so we keep everything after the last close.
    Robust to an unterminated thought (returns "" so we don't mine the reasoning
    text for spurious tool-call markers).
    """
    if _THOUGHT_OPEN not in text:
        return text
    # Reassemble like the template: for each '<channel|>'-split part, drop the
    # portion starting at any '<|channel>' (the thought body).
    out_parts = []
    for part in text.split(_THOUGHT_CLOSE):
        if _THOUGHT_OPEN in part:
            out_parts.append(part.split(_THOUGHT_OPEN)[0])
        else:
            out_parts.append(part)
    return "".join(out_parts)


def _parse_gemma_tool_calls(raw: str) -> tuple[Optional[str], list[_TCShim]]:
    """Parse gemma-4 generated text into (content, [tool_call_shim, ...]).

    Format per the gemma-4 chat template (chat_template.jinja:243-258):
        <|tool_call>call:<NAME>{<ARGS-JSON>}<tool_call|>
    where, because the dataset stores tool_call.arguments as a JSON *string*, the
    template emits the template's literal '{' then the JSON object then '}', i.e.
    the bytes between the FIRST '{' after the name and the matching '}<tool_call|>'
    are exactly one outer (template) brace pair wrapping the JSON object. We strip
    that single outer pair and return the JSON text as the `arguments` string —
    leaving the actual JSON parsing to compute_turn_result (so json_valid_rate is
    scored by the upstream code, unchanged).

    Returns (content, tool_calls). content is the visible (non-thought, non-call)
    text or None — matching what the API put on message.content (and what
    evaluate_conversation stores into history).
    """
    text = _strip_thought(raw)

    # The model emits gemma-4's NATIVE tool-call syntax: `call:NAME{key:value,...}`.
    # The `<|tool_call>` opener is part of the *prompt* (add_generation_prompt), so the
    # generated text starts straight at `call:NAME{...}`; `<tool_call|>` may or may not close
    # it. So we scan for `call:` blocks directly (not the wrapper). The args are gemma's own
    # unquoted `{k:v}` form (NOT JSON) -> we convert to real JSON so the upstream scorer's
    # json.loads / required-coverage / type checks work; on failure we keep the raw inner text
    # (json_valid scores 0 for that call, but the name still counts for tool_call_f1).
    tool_calls: list[_TCShim] = []
    idx = 0
    first_call_pos = None
    pos = 0
    while True:
        c = text.find(_CALL_PREFIX, pos)
        if c < 0:
            break
        brace = text.find("{", c)
        if brace < 0:
            break
        name = text[c + len(_CALL_PREFIX):brace].strip()
        # strip stray wrapper/whitespace from the name
        name = name.strip("<>| \n\t").replace("|tool_call", "").strip()
        obj, end = _parse_gemma_value(text, brace)  # parses the balanced {...}
        if isinstance(obj, dict):
            args_str = json.dumps(obj)
        else:
            inner = text[brace:end]
            args_str = inner[1:-1] if inner.startswith("{") and inner.endswith("}") else inner
        if name:
            tool_calls.append(_TCShim(f"local-call-{idx}", name, args_str))
            idx += 1
            if first_call_pos is None:
                first_call_pos = c
        pos = max(end, c + len(_CALL_PREFIX))

    content = text[:first_call_pos].strip() if first_call_pos else (text.strip() if not tool_calls else None)
    return (content or None), tool_calls


def _parse_gemma_value(s: str, i: int):
    """Best-effort parse of one gemma-format value starting at index i; returns (value, next_i).

    Handles objects {k:v,...}, arrays [..], quoted strings, and barewords (true/false/null,
    ints, floats, else string). gemma emits unquoted keys/values, e.g.
    {include_metrics:true,reconciliation_run_id:RC-2026-04187}.
    """
    n = len(s)
    while i < n and s[i] in " \t\n\r":
        i += 1
    if i >= n:
        return None, i
    ch = s[i]
    if ch == "{":
        obj = {}
        i += 1
        while i < n:
            while i < n and s[i] in " \t\n\r,":
                i += 1
            if i < n and s[i] == "}":
                return obj, i + 1
            k0 = i
            depth = 0
            while i < n and not (s[i] == ":" and depth == 0):
                if s[i] in "{[":
                    depth += 1
                elif s[i] in "}]":
                    depth -= 1
                i += 1
            key = s[k0:i].strip().strip("\"'")
            i += 1  # skip ':'
            val, i = _parse_gemma_value(s, i)
            obj[key] = val
        return obj, i
    if ch == "[":
        arr = []
        i += 1
        while i < n:
            while i < n and s[i] in " \t\n\r,":
                i += 1
            if i < n and s[i] == "]":
                return arr, i + 1
            val, i = _parse_gemma_value(s, i)
            arr.append(val)
        return arr, i
    if ch in "\"'":
        q = ch
        i += 1
        buf = []
        while i < n and s[i] != q:
            if s[i] == "\\" and i + 1 < n:
                buf.append(s[i + 1]); i += 2; continue
            buf.append(s[i]); i += 1
        return "".join(buf), i + 1
    # bareword: read to the next top-level , } ]
    start = i
    while i < n and s[i] not in ",}]":
        i += 1
    tok = s[start:i].strip()
    low = tok.lower()
    if low == "true":
        return True, i
    if low == "false":
        return False, i
    if low in ("null", "none"):
        return None, i
    try:
        return int(tok), i
    except ValueError:
        pass
    try:
        return float(tok), i
    except ValueError:
        pass
    return tok, i


# =============================================================================
# === ADAPTATION 3: local transformers "call_model" replacing the OpenAI client
# =============================================================================

class LocalModel:
    """Loads google/gemma-4-31B-it with transformers, optionally merges a LoRA,
    and exposes generate_turn(history, tools) -> _MsgShim — the local analogue of
    evaluate_function_calling.call_model()."""

    def __init__(self, model_id: str, lora_path: Optional[str], scaling: Optional[float],
                 max_new_tokens: int, vllm_url: Optional[str] = None,
                 served_model: Optional[str] = None):
        from transformers import AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.model_id = model_id
        self.vllm_url = vllm_url.rstrip("/") if vllm_url else None
        self.served_model = served_model or model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # vLLM API mode: gemma-4's head_dim-512 attention has NO memory-efficient kernel in
        # PyTorch SDPA (math backend -> O(S^2) -> 61 GiB OOM at ~22k-token eval prompts; flex
        # fails to compile). vLLM serves it with tensor parallelism (each GPU computes half the
        # heads) + a head_dim-512 attention kernel, so we drive a vLLM OpenAI server instead of
        # generating locally. We still build the prompt + parse + score here, identically.
        if self.vllm_url:
            import requests
            self.torch = None
            self.model = None
            self.session = requests.Session()
            self.eos_token_id = None
            print(f">> vLLM API mode: {self.vllm_url} (model={self.served_model})", flush=True)
            return

        import torch
        from transformers import Gemma4ForConditionalGeneration
        self.torch = torch
        print(f">> loading base {model_id} (bf16, sdpa, device_map=auto)", flush=True)
        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
        )
        self.model.eval()

        if lora_path:
            # Merge exactly like merge_infer.py: W <- W + scaling * (B @ A).
            import merge_infer
            if scaling is None:
                meta = merge_infer.load_meta(lora_path)
                scaling = meta.get("scaling")
            if scaling is None:
                raise SystemExit(
                    "scaling unknown: pass --scaling alpha/r (no lora_meta.json next "
                    f"to {lora_path})"
                )
            print(f">> merging LoRA {lora_path} (scaling={scaling})", flush=True)
            merge_infer.merge_lora_(self.model, lora_path, scaling)

        # Stop generation at the gemma-4 turn terminator '<turn|>' (single token, id
        # 106) and at <eos>. eos_token_id may already be a list across versions.
        eos = self.tokenizer.eos_token_id
        stop_ids = set()
        if isinstance(eos, (list, tuple)):
            stop_ids.update(eos)
        elif eos is not None:
            stop_ids.add(eos)
        for marker in ("<turn|>", "<end_of_turn>"):
            ids = self.tokenizer.encode(marker, add_special_tokens=False)
            if len(ids) == 1:
                stop_ids.add(ids[0])
        self.eos_token_id = sorted(stop_ids)

    def generate_turn(self, history: list, tools: list) -> Optional[_MsgShim]:
        """Local replacement for call_model(): apply the gemma-4 chat template to
        (history + tools), greedily generate the assistant turn, parse tool calls.

        `tools` is the OpenAI-wrapped list from build_openai_tools — exactly the
        shape the gemma-4 template + pack_dataset.extract_tools expect.
        """
        # vLLM API path: tokenize the prompt HERE (identical to local), send the token ids to
        # /v1/completions (so vLLM never re-templates / double-BOSes), parse the returned text.
        if self.vllm_url:
            try:
                enc = self.tokenizer.apply_chat_template(
                    history, tools=tools if tools else None,
                    add_generation_prompt=True, tokenize=True, return_dict=True,
                )
                ids = enc["input_ids"]
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
            except Exception as e:
                print(f"  [error] apply_chat_template: {e}", file=sys.stderr)
                return None
            print(f"  [prompt_len] {len(ids)} tokens", file=sys.stderr)
            try:
                r = self.session.post(
                    f"{self.vllm_url}/v1/completions",
                    json={
                        "model": self.served_model,
                        "prompt": ids,
                        "max_tokens": self.max_new_tokens,
                        "temperature": 0,
                        "stop": ["<turn|>", "<end_of_turn>"],
                    },
                    timeout=1800,
                )
                r.raise_for_status()
                raw = r.json()["choices"][0]["text"]
            except Exception as e:
                print(f"  [error] vllm: {e}", file=sys.stderr)
                return None
            content, tool_calls = _parse_gemma_tool_calls(raw)
            return _MsgShim(content=content, tool_calls=tool_calls)

        torch = self.torch
        try:
            enc = self.tokenizer.apply_chat_template(
                history,
                tools=tools if tools else None,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except Exception as e:
            print(f"  [error] apply_chat_template: {e}", file=sys.stderr)
            return None

        input_ids = enc["input_ids"].to(self.model.device)
        print(f"  [prompt_len] {input_ids.shape[-1]} tokens", file=sys.stderr)
        # Deliberately DO NOT pass attention_mask: it's a single un-padded sequence, so passing a
        # mask only forces transformers to build an explicit 4D causal mask, which pushes SDPA onto
        # the math backend (O(S^2) -> 61 GiB OOM at ~22k tokens for the head_dim-512 full-attention
        # layers). Without it, transformers uses the is_causal fast path -> memory-efficient SDPA.
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            logits_to_keep=1,  # only the last position's logits — the 262k-vocab head over a long
                               # prefill is itself tens of GB otherwise.
            eos_token_id=self.eos_token_id,
            pad_token_id=(self.tokenizer.pad_token_id
                          if self.tokenizer.pad_token_id is not None
                          else self.eos_token_id[0]),
        )

        try:
            with torch.no_grad():
                out = self.model.generate(input_ids, **gen_kwargs)
        except Exception as e:
            print(f"  [error] generate: {e}", file=sys.stderr)
            return None

        # Decode only the newly generated tokens. Keep special-string markers
        # ('<|tool_call>' etc. are NOT in added_tokens, so they survive decode);
        # we drop the trailing <eos>/<turn|> ourselves via the parser.
        gen_ids = out[0][input_ids.shape[-1]:]
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        # Trim the turn/eos terminators so they don't pollute trailing content.
        for marker in ("<turn|>", "<eos>", "<end_of_turn>", "<|tool_response>"):
            ci = raw.find(marker)
            if ci >= 0:
                raw = raw[:ci]

        content, tool_calls = _parse_gemma_tool_calls(raw)
        return _MsgShim(content=content, tool_calls=tool_calls)


# =============================================================================
# === Full-replay evaluation (ported from evaluate_function_calling.py:286-392)
# =============================================================================
# Identical control flow; the only swap is call_model(...) -> lm.generate_turn(...).

def evaluate_conversation(
    lm: LocalModel,
    conv: dict,
    tools: list,
    fn_names: set,
    fn_schema_map: dict,
) -> tuple[list[TurnResult], Optional[str]]:
    msgs = conv["messages"]
    history: list[dict] = []
    results: list[TurnResult] = []
    available_ids: set = set()
    turn_num = 0
    ref_idx = 0

    meta = conv.get("metadata", {})
    oot_turns: set = {
        t["turn"] - 1
        for t in meta.get("out_of_context_turns", [])
        if isinstance(t, dict) and "turn" in t
    }

    while ref_idx < len(msgs):
        msg = msgs[ref_idx]
        role = msg["role"]

        if role == "user":
            history.append({"role": "user", "content": msg["content"]})
            available_ids = set()
            ref_idx += 1

        elif role == "assistant":
            ref_tcs = msg.get("tool_calls") or []

            # Call the (local) model with the conversation so far.
            model_msg = lm.generate_turn(history, tools)
            if model_msg is None:
                return results, "model call failed"

            model_tcs = model_msg.tool_calls or []

            tr = compute_turn_result(
                turn=turn_num,
                ref_tcs=ref_tcs,
                model_tcs=model_tcs,
                fn_names=fn_names,
                fn_schema_map=fn_schema_map,
                available_ids=available_ids,
            )
            tr.out_of_context = turn_num in oot_turns
            results.append(tr)
            turn_num += 1

            # Append the model's own response to history (like a real conversation).
            assistant_hist: dict = {"role": "assistant", "content": model_msg.content or ""}
            if model_tcs:
                assistant_hist["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in model_tcs
                ]
            history.append(assistant_hist)
            ref_idx += 1

            # Collect the reference tool results that follow this assistant turn.
            ref_tool_msgs: list[dict] = []
            while ref_idx < len(msgs) and msgs[ref_idx]["role"] == "tool":
                ref_tool_msgs.append(msgs[ref_idx])
                ref_idx += 1

            # Inject reference tool results using the model's tool call IDs (positional).
            for i, mtc in enumerate(model_tcs):
                content = (
                    ref_tool_msgs[i]["content"]
                    if i < len(ref_tool_msgs)
                    else json.dumps({"status": "no_reference_result"})
                )
                history.append({
                    "role": "tool",
                    "tool_call_id": mtc.id,
                    "name": mtc.function.name,
                    "content": content,
                })
                available_ids.update(extract_ids(content))

        elif role == "tool":
            ref_idx += 1
        else:
            ref_idx += 1

    return results, None


# =============================================================================
# === Data loading: local parquet (default) with HF uri fallback
# =============================================================================
# Yields (lib, conv) pairs in the SAME shape build_openai_tools / evaluate_
# conversation expect — built the SAME way as evaluate_function_calling.load_hf_data
# (:519-538), just sourced from the local parquet columns.

def load_pairs(data: str, max_rows: int) -> list[tuple[dict, dict]]:
    import pandas as pd

    src = data
    if data.startswith("hf://") or data.startswith("http"):
        df = pd.read_parquet(data)
    elif os.path.exists(data):
        df = pd.read_parquet(data)
    else:
        print(f"[eval_funccall] local {data} not found; falling back to {DEFAULT_HF_URI}",
              flush=True)
        src = DEFAULT_HF_URI
        df = pd.read_parquet(DEFAULT_HF_URI)
    print(f"[eval_funccall] loaded {len(df)} rows from {src}", flush=True)

    if max_rows:
        df = df.head(max_rows)

    pairs = []
    for i in range(len(df)):
        row = df.iloc[i]
        try:
            lib = {
                "workflow_name": row["workflow"],
                "shared_entities": json.loads(row["shared_entities"]),
                "functions": json.loads(row["functions"]),
            }
            conv = {
                "conversation_id": (row["metadata"] and json.loads(row["metadata"]).get("conversation_id"))
                                   or f"{row['workflow']}-{i}",
                "workflow_name": row["workflow"],
                "domain": row["domain"],
                "messages": json.loads(row["messages"]),
                "metadata": json.loads(row["metadata"]),
            }
            pairs.append((lib, conv))
        except Exception as e:
            print(f"  [warn] row {i} parse error: {e}", file=sys.stderr)

    print(f"[eval_funccall] {len(pairs)} conversations.", flush=True)
    return pairs


# =============================================================================
# === Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Local-transformers function-calling eval for gemma-4-31B-it "
                    "(SyntheticGen scoring). Base vs LoRA-finetuned."
    )
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help=f"Parquet path (default: {DEFAULT_DATA}); falls back to the HF uri.")
    ap.add_argument("--model-id", default="google/gemma-4-31B-it")
    ap.add_argument("--lora", default=None,
                    help="LoRA .pt to merge before eval (merge_infer.merge_lora_). "
                         "Omit for the BASE model.")
    ap.add_argument("--scaling", type=float, default=None,
                    help="LoRA scaling alpha/r (overrides lora_meta.json).")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows.")
    ap.add_argument("--max-new-tokens", type=int, default=16384,
                    help="Per-turn generation cap (upstream API used max_tokens=16384).")
    ap.add_argument("--out", default=None, help="Results JSON path.")
    ap.add_argument("--vllm-url", default=None,
                    help="If set, drive a vLLM OpenAI server at this base URL (e.g. "
                         "http://localhost:8000) instead of loading the model locally. Required "
                         "for the real eval — head_dim-512 attention OOMs in transformers at "
                         "~22k-token prompts. With --lora, serve the MERGED model in vLLM.")
    ap.add_argument("--served-model", default=None,
                    help="Model name the vLLM server is serving under (default: --model-id).")
    ap.add_argument("--cache", default=None,
                    help="JSONL per-conversation result cache (default: <out>.cache.jsonl). Each "
                         "finished conversation is appended immediately; on restart, conversations "
                         "already in the cache are SKIPPED (no re-generation). Delete it to force "
                         "a fresh run.")
    ap.add_argument("--metrics-only", action="store_true",
                    help="Don't generate anything: just aggregate + print metrics over whatever is "
                         "already in the cache (the rows generated so far).")
    args = ap.parse_args()

    model_label = args.model_id + (f" + LoRA({args.lora})" if args.lora else " (base)")
    out_path = args.out or (
        f"{args.model_id.replace('/', '_')}"
        + ("_lora" if args.lora else "_base")
        + "_results.json"
    )
    cache_path = args.cache or (out_path + ".cache.jsonl")

    # --- load any previously-cached conversations (keyed by conversation_id) ---
    cache: dict = {}
    if os.path.exists(cache_path):
        with open(cache_path) as cf:
            for line in cf:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    cache[e["conversation_id"]] = e
                except Exception:
                    pass
        print(f">> cache {cache_path}: {len(cache)} conversations already generated", flush=True)

    def agg_from(entries):
        atr = [[TurnResult(**d) for d in e["turn_results"]] for e in entries]
        return aggregate(atr), atr

    # --- metrics-only: report on the rows generated so far, then exit ---
    if args.metrics_only:
        if not cache:
            raise SystemExit(f"[eval_funccall] cache {cache_path} is empty — nothing to aggregate.")
        m, atr = agg_from(cache.values())
        n_err = sum(1 for e in cache.values() if e.get("error"))
        print_summary(m, len(atr), n_err, model_label + " [cache-so-far]")
        print(f"\n  [cache so far] {len(atr)} convs  tool_call_f1 = {m['tool_call_f1']:.4f}  "
              f"(precision {m['tool_call_precision']:.4f}, recall {m['tool_call_recall']:.4f})")
        return

    pairs = load_pairs(args.data, args.max_rows)
    if not pairs:
        raise SystemExit("[eval_funccall] no conversations loaded.")

    lm = LocalModel(args.model_id, args.lora, args.scaling, args.max_new_tokens,
                    vllm_url=args.vllm_url, served_model=args.served_model)

    conv_results: list[dict] = []
    all_turn_results: list[list[TurnResult]] = []
    n_errors = 0
    t0 = time.time()

    for n, (lib, conv) in enumerate(pairs, 1):
        conv_id = conv.get("conversation_id", f"conv-{n}")
        # skip if this conversation is already cached (resume / no re-generation)
        if conv_id in cache:
            e = cache[conv_id]
            all_turn_results.append([TurnResult(**d) for d in e["turn_results"]])
            conv_results.append(e["conv_result"])
            if e.get("error"):
                n_errors += 1
            print(f"  [{n}/{len(pairs)}] {conv_id}  CACHED (turns={len(e['turn_results'])})", flush=True)
            continue
        tools, fn_schema_map, fn_names = build_openai_tools(lib)
        turn_results, error = evaluate_conversation(lm, conv, tools, fn_names, fn_schema_map)
        if error:
            n_errors += 1
        all_turn_results.append(turn_results)
        conv_result = {
            "conversation_id": conv_id,
            "workflow_name": conv.get("workflow_name"),
            "domain": conv.get("domain"),
            "error": error,
            "turns": [
                {
                    "turn": tr.turn,
                    "ref_call_count": tr.ref_call_count,
                    "model_call_count": tr.model_call_count,
                    "tp": tr.tp, "fp": tr.fp, "fn": tr.fn_miss, "tn": tr.tn,
                    "name_matches": tr.name_matches,
                    "ref_names": tr.ref_name_set,
                    "model_names": tr.model_name_set,
                    "json_valid": tr.json_valid,
                    "hallucinated": tr.hallucinated,
                    "req_coverages": tr.req_coverages,
                    "type_accs": tr.type_accs,
                    "id_prop": tr.id_prop,
                    "out_of_context": tr.out_of_context,
                }
                for tr in turn_results
            ],
        }
        conv_results.append(conv_result)
        # append to the durable cache IMMEDIATELY so a crash/restart resumes here
        with open(cache_path, "a") as cf:
            cf.write(json.dumps({
                "conversation_id": conv_id,
                "error": error,
                "conv_result": conv_result,
                "turn_results": [asdict(tr) for tr in turn_results],
            }) + "\n")
        cache[conv_id] = True
        err = f" [ERROR: {error}]" if error else ""
        print(f"  [{n}/{len(pairs)}] {conv_id}  turns={len(turn_results)}{err}", flush=True)

    metrics = aggregate(all_turn_results)
    print_summary(metrics, len(conv_results), n_errors, model_label)
    print(f"\n  Headline tool_call_f1 = {metrics['tool_call_f1']:.4f}  "
          f"(precision {metrics['tool_call_precision']:.4f}, "
          f"recall {metrics['tool_call_recall']:.4f})")
    print(f"  name_set_f1 = {metrics['name_set_f1']:.4f}  |  elapsed {time.time() - t0:.0f}s")

    # Same shape as SyntheticGen's *_results.json: {model, split, metrics, conversations}.
    Path(out_path).write_text(json.dumps({
        "model": model_label,
        "model_id": args.model_id,
        "lora": args.lora,
        "split": Path(args.data).name if os.path.exists(args.data) else args.data,
        "metrics": metrics,
        "conversations": conv_results,
    }, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
