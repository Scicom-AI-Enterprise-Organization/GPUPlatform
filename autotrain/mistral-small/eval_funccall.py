#!/usr/bin/env python3
"""Function-calling accuracy eval for Mistral-Small-4-119B — base vs LoRA-finetuned.

The Mistral-Small-4 sibling of `../gemma4/eval_funccall.py`. Reuses SyntheticGen's EXACT scoring
(SyntheticGen/synthetic/evaluate_function_calling.py) verbatim — build_openai_tools, the per-turn
TP/FP/FN/TN bookkeeping, compute_turn_result, evaluate_conversation full-replay, aggregate,
print_summary — so the metrics are directly comparable. Only the two model-specific bits differ:

  (1) PROMPT: the Mistral-Small-4 chat template (tools wrapped {"type":"function","function":fn},
      reasoning_effort), applied exactly like pack_dataset.py.
  (2) TOOL-CALL PARSING: Mistral-Small-4 emits  [TOOL_CALLS]<NAME>[ARGS]{<json-args>}  (repeated
      per call), reasoning in [THINK]...[/THINK]. `_parse_mistral_tool_calls` extracts each block
      and hands the JSON-string args to the upstream scorer (which json.loads them), identical to
      what the OpenAI API handed compute_turn_result.

Generation is driven through **vLLM** (`--vllm-url`), because transformers' FP8 generate path is
broken for this model (static-activation experts NotImplementedError; the dequant path degenerates
under KV-cache). Merge the LoRA first with `merge_to_bf16.py`, serve the merged model with vLLM,
then point `--vllm-url` at it. Greedy (temperature 0) for reproducible numbers.

    # serve merged model:  vllm serve /share/merged-mistral-64k --served-model-name mistral-lora ...
    python eval_funccall.py --vllm-url http://localhost:8000 --served-model mistral-lora \
        --max-rows 25 --out lora25_results.json
"""
import argparse
import copy
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# HF_TOKEN from autotrain/.env (../.env) then gateway/.env (mistral convention).
_ENV_CANDIDATES = [os.path.join(os.path.dirname(__file__), "..", ".env"),
                   "/home/husein/ssd3/GPUPlatform/gateway/.env"]


def _load_hf_token(paths):
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    for path in paths:
        try:
            with open(path) as f:
                for line in f:
                    if line.strip().startswith("HF_TOKEN="):
                        v = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if v:
                            os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = v
                        return
        except FileNotFoundError:
            continue


_load_hf_token(_ENV_CANDIDATES)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

DEFAULT_HF_URI = "hf://datasets/Scicom-intl/Function-Call-TaaS/glm5.1-fp8-test/test-00000-of-00001.parquet"
DEFAULT_MODEL_ID = "mistralai/Mistral-Small-4-119B-2603"

# =============================================================================
# === BEGIN verbatim port from evaluate_function_calling.py (schema + metrics)
# =============================================================================

def resolve_refs(schema, shared):
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


def build_openai_tools(lib):
    shared = lib.get("shared_entities", {})
    tools, fn_schema_map = [], {}
    for fn in lib["functions"]:
        resolved = resolve_refs(copy.deepcopy(fn), shared)
        params = resolved.get("parameters", {"type": "object", "properties": {}})
        tools.append({"type": "function", "function": {
            "name": resolved["name"], "description": resolved.get("description", ""), "parameters": params}})
        fn_schema_map[resolved["name"]] = resolved
    return tools, fn_schema_map, set(fn_schema_map.keys())


def required_coverage(args, fn_schema):
    required = fn_schema.get("parameters", {}).get("required", [])
    if not required:
        return 1.0
    return sum(1 for r in required if r in args) / len(required)


_TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}


def type_accuracy(args, fn_schema):
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


def extract_ids(content_str):
    ids = set()
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


def id_propagation(model_args, ref_args, available_ids):
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


def compute_turn_result(turn, ref_tcs, model_tcs, fn_names, fn_schema_map, available_ids):
    tr = TurnResult(turn=turn, ref_call_count=len(ref_tcs), model_call_count=len(model_tcs))
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
    model_names, model_args_strs = [], []
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


def aggregate(all_results):
    tp = fp = fn_miss = tn = 0
    json_valid = json_total = hallucinated = hall_total = 0
    req_cov_vals, type_acc_vals, id_prop_vals = [], [], []
    parallel_match = parallel_total = 0
    name_match_sum = name_match_n = 0
    name_set_tp = name_set_fp = name_set_fn = 0
    refusal_correct = refusal_total = 0
    for conv_turns in all_results:
        for tr in conv_turns:
            tp += tr.tp; fp += tr.fp; fn_miss += tr.fn_miss; tn += tr.tn
            json_valid += tr.json_valid; json_total += tr.model_call_count
            hallucinated += tr.hallucinated; hall_total += tr.model_call_count
            req_cov_vals.extend(tr.req_coverages); type_acc_vals.extend(tr.type_accs)
            if tr.id_prop is not None:
                id_prop_vals.append(tr.id_prop)
            if tr.ref_call_count > 0:
                parallel_total += 1
                if tr.model_call_count == tr.ref_call_count:
                    parallel_match += 1
            if tr.ref_call_count > 0 and tr.model_call_count > 0:
                ref_set, model_set = set(tr.ref_name_set), set(tr.model_name_set)
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
    sprec = _div(name_set_tp, name_set_tp + name_set_fp)
    srec = _div(name_set_tp, name_set_tp + name_set_fn)
    sf1 = round(2 * sprec * srec / (sprec + srec), 4) if (sprec + srec) > 0 else 0.0
    return {
        "tool_call_precision": prec, "tool_call_recall": rec, "tool_call_f1": f1,
        "name_accuracy": _div(name_match_sum, name_match_n),
        "name_set_precision": sprec, "name_set_recall": srec, "name_set_f1": sf1,
        "json_valid_rate": _div(json_valid, json_total),
        "hallucination_rate": _div(hallucinated, hall_total),
        "req_coverage": _avg(req_cov_vals), "type_accuracy": _avg(type_acc_vals),
        "parallel_count_match": _div(parallel_match, parallel_total),
        "id_propagation_rate": _avg(id_prop_vals),
        "refusal_rate": _div(refusal_correct, refusal_total) if refusal_total > 0 else None,
        "_counts": {"tp": tp, "fp": fp, "fn": fn_miss, "tn": tn,
                    "total_model_calls": json_total, "total_ref_calls": tp + fn_miss},
    }


def print_summary(metrics, n_convs, n_errors, model):
    m = metrics
    print("\n" + "=" * 60)
    print(f"  Function Calling Evaluation — {model}")
    print(f"  Conversations: {n_convs}  |  Errors: {n_errors}")
    print("=" * 60)
    print(f"  Tool Call Detection\n    Precision:          {m['tool_call_precision']:.3f}\n"
          f"    Recall:             {m['tool_call_recall']:.3f}\n    F1:                 {m['tool_call_f1']:.3f}")
    print(f"  Function Selection\n    Name Accuracy:      {m['name_accuracy']:.3f}\n"
          f"    Name Set F1:        {m['name_set_f1']:.3f}  (P={m['name_set_precision']:.3f}  R={m['name_set_recall']:.3f})")
    print(f"  Output Quality\n    JSON Valid Rate:    {m['json_valid_rate']:.3f}\n    Hallucination Rate: {m['hallucination_rate']:.3f}")
    rc, ta = m['req_coverage'], m['type_accuracy']
    print(f"  Argument Quality\n    Req Coverage:       {rc if rc is None else round(rc,3)}\n    Type Accuracy:      {ta if ta is None else round(ta,3)}")
    ip = m['id_propagation_rate']
    print(f"  Multi-turn\n    Parallel Match:     {m['parallel_count_match']:.3f}\n    ID Propagation:     {ip if ip is None else round(ip,3)}")
    c = m["_counts"]
    print(f"\n  Confusion: TP={c['tp']}  FP={c['fp']}  FN={c['fn']}  TN={c['tn']}")
    print(f"  Total model calls: {c['total_model_calls']}  |  ref calls: {c['total_ref_calls']}")
    print("=" * 60)

# =============================================================================
# === Tool-call shims (the objects compute_turn_result reads)
# =============================================================================

class _FnShim:
    __slots__ = ("name", "arguments")

    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _TCShim:
    __slots__ = ("id", "type", "function")

    def __init__(self, tc_id, name, arguments):
        self.id = tc_id
        self.type = "function"
        self.function = _FnShim(name, arguments)


class _MsgShim:
    __slots__ = ("content", "tool_calls")

    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls


# =============================================================================
# === Mistral-Small-4 tool-call text parser:  [TOOL_CALLS]NAME[ARGS]{json}
# =============================================================================
_TC = "[TOOL_CALLS]"
_ARGS = "[ARGS]"
_THINK_O, _THINK_C = "[THINK]", "[/THINK]"


def _strip_think(text):
    while _THINK_O in text:
        a = text.find(_THINK_O)
        b = text.find(_THINK_C, a)
        if b < 0:
            text = text[:a]  # unterminated think -> drop the rest
            break
        text = text[:a] + text[b + len(_THINK_C):]
    return text


def _first_json_obj(s):
    """Return the first balanced {...} substring of s (the tool-call args), or None."""
    i = s.find("{")
    if i < 0:
        return None
    depth, instr, esc = 0, False, False
    for j in range(i, len(s)):
        ch = s[j]
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
        else:
            if ch == '"':
                instr = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[i:j + 1]
    return None


def _parse_mistral_tool_calls(raw):
    """Parse '[TOOL_CALLS]NAME[ARGS]{json}' blocks -> (content, [_TCShim]). Args handed on as the
    raw JSON string (compute_turn_result runs its own json.loads, scoring json_valid_rate)."""
    text = _strip_think(raw)
    for stop in ("</s>", "[/INST]", "[TOOL_RESULTS]"):
        i = text.find(stop)
        if i >= 0:
            text = text[:i]
    if _TC not in text:
        return (text.strip() or None), []
    head, *blocks = text.split(_TC)
    content = head.strip() or None
    tool_calls = []
    for idx, blk in enumerate(blocks):
        if _ARGS in blk:
            name, args = blk.split(_ARGS, 1)
        else:
            name, args = blk, "{}"
        name = name.strip()
        args_obj = _first_json_obj(args)
        args_str = args_obj if args_obj is not None else (args.strip() or "{}")
        if name:
            tool_calls.append(_TCShim(f"local-call-{idx}", name, args_str))
    return content, tool_calls


# =============================================================================
# === Model driver: vLLM OpenAI server (primary) or local transformers
# =============================================================================

class Model:
    """vLLM-server mode (--vllm-url) OR local-transformers mode (--merged-dir, the MERGED bf16
    model — vLLM 0.23.0 has no mistral4 support, but the plain bf16 merge generates fine locally)."""

    def __init__(self, model_id, max_new_tokens, vllm_url, served_model, reasoning_effort,
                 merged_dir=None):
        from transformers import AutoTokenizer
        self.max_new_tokens = max_new_tokens
        self.reasoning_effort = reasoning_effort
        self.vllm_url = vllm_url.rstrip("/") if vllm_url else None
        self.served_model = served_model or model_id
        self.merged_dir = merged_dir
        tok_src = merged_dir or model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        # Relax the template's strict user/assistant alternation guard: a full model-vs-reference
        # replay legitimately produces non-alternating histories (e.g. the model answers in text
        # where the reference called a tool), which the stock template would `raise_exception` on,
        # aborting the conversation. Drop just that raise so the replay can continue.
        ct = self.tokenizer.chat_template or ""
        _alt = ("raise_exception('After the optional system message, conversation roles must "
                "alternate user and assistant roles except for tool calls and results.')")
        self.chat_template = ct.replace(_alt, "''") if _alt in ct else None
        if self.vllm_url:
            import requests
            self.session = requests.Session()
            print(f">> vLLM API mode: {self.vllm_url} (model={self.served_model}, "
                  f"reasoning_effort={reasoning_effort})", flush=True)
            return
        if not merged_dir:
            raise SystemExit("pass --merged-dir (the merge_to_bf16.py output) or --vllm-url.")
        import torch
        from transformers import Mistral3ForConditionalGeneration
        self.torch = torch
        rank = int(os.environ.get("LOCAL_RANK", -1))
        self.tp = rank >= 0  # launched under torchrun -> tensor parallel (fast); else device_map (slow)
        kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
        if self.tp:
            kw["tp_plan"] = "auto"
            print(f">> [rank{rank}] loading MERGED bf16 {merged_dir} (sdpa, TP={os.environ.get('WORLD_SIZE')})", flush=True)
        else:
            kw["device_map"] = "auto"
            print(f">> loading MERGED bf16 {merged_dir} (sdpa, device_map=auto)", flush=True)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(merged_dir, **kw).eval()
        # NB: keep the default experts dispatch — overriding to grouped_mm breaks the tp_plan expert
        # sharding (moe_tp_experts) and causes NCCL collective mismatch under tensor parallel.
        self.device = f"cuda:{rank}" if self.tp else self.model.device
        eos = self.tokenizer.eos_token_id
        self.eos_ids = sorted(set(eos if isinstance(eos, (list, tuple)) else [eos]))

    def _render(self, history, tools, tensors=False):
        kw = {"reasoning_effort": self.reasoning_effort}
        if self.chat_template is not None:
            kw["chat_template"] = self.chat_template
        return self.tokenizer.apply_chat_template(
            history, tools=tools if tools else None, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt" if tensors else None, **kw)["input_ids"]

    def generate_turn(self, history, tools):
        if self.vllm_url:
            try:
                ids = self._render(history, tools)
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
            except Exception as e:
                print(f"  [error] apply_chat_template: {e}", file=sys.stderr)
                return None
            try:
                r = self.session.post(f"{self.vllm_url}/v1/completions", json={
                    "model": self.served_model, "prompt": ids,
                    "max_tokens": self.max_new_tokens, "temperature": 0,
                    "stop": ["</s>"]}, timeout=1800)
                r.raise_for_status()
                raw = r.json()["choices"][0]["text"]
            except Exception as e:
                print(f"  [error] vllm: {e}", file=sys.stderr)
                return None
            content, tool_calls = _parse_mistral_tool_calls(raw)
            return _MsgShim(content, tool_calls)

        # local transformers (merged bf16 model)
        torch = self.torch
        try:
            input_ids = self._render(history, tools, tensors=True).to(self.device)
        except Exception as e:
            print(f"  [error] apply_chat_template: {e}", file=sys.stderr)
            return None
        try:
            with torch.no_grad():
                out = self.model.generate(input_ids, max_new_tokens=self.max_new_tokens,
                                          do_sample=False, use_cache=True,
                                          pad_token_id=self.eos_ids[0])
        except Exception as e:
            print(f"  [error] generate: {e}", file=sys.stderr)
            return None
        raw = self.tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=False)
        content, tool_calls = _parse_mistral_tool_calls(raw)
        return _MsgShim(content, tool_calls)


# =============================================================================
# === Full-replay evaluation (ported from evaluate_function_calling.py)
# =============================================================================

def evaluate_conversation(lm, conv, tools, fn_names, fn_schema_map):
    msgs = conv["messages"]
    history, results, available_ids = [], [], set()
    turn_num, ref_idx = 0, 0
    meta = conv.get("metadata", {})
    oot_turns = {t["turn"] - 1 for t in meta.get("out_of_context_turns", [])
                 if isinstance(t, dict) and "turn" in t}
    while ref_idx < len(msgs):
        msg = msgs[ref_idx]
        role = msg["role"]
        if role == "user":
            history.append({"role": "user", "content": msg["content"]})
            available_ids = set()
            ref_idx += 1
        elif role == "assistant":
            ref_tcs = msg.get("tool_calls") or []
            model_msg = lm.generate_turn(history, tools)
            if model_msg is None:
                return results, "model call failed"
            model_tcs = model_msg.tool_calls or []
            tr = compute_turn_result(turn_num, ref_tcs, model_tcs, fn_names, fn_schema_map, available_ids)
            tr.out_of_context = turn_num in oot_turns
            results.append(tr)
            turn_num += 1
            assistant_hist = {"role": "assistant", "content": model_msg.content or ""}
            if model_tcs:
                assistant_hist["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in model_tcs]
            elif not (assistant_hist["content"] or "").strip():
                # the chat template rejects an assistant turn with no content AND no tool calls
                assistant_hist["content"] = "(no response)"
            history.append(assistant_hist)
            ref_idx += 1
            ref_tool_msgs = []
            while ref_idx < len(msgs) and msgs[ref_idx]["role"] == "tool":
                ref_tool_msgs.append(msgs[ref_idx])
                ref_idx += 1
            for i, mtc in enumerate(model_tcs):
                content = ref_tool_msgs[i]["content"] if i < len(ref_tool_msgs) else json.dumps({"status": "no_reference_result"})
                history.append({"role": "tool", "tool_call_id": mtc.id, "name": mtc.function.name, "content": content})
                available_ids.update(extract_ids(content))
        else:
            ref_idx += 1
    return results, None


def load_pairs(data, max_rows):
    import pandas as pd
    src = data
    if data.startswith("hf://") or data.startswith("http") or os.path.exists(data):
        df = pd.read_parquet(data)
    else:
        print(f"[eval] local {data} not found; using {DEFAULT_HF_URI}", flush=True)
        src = DEFAULT_HF_URI
        df = pd.read_parquet(DEFAULT_HF_URI)
    print(f"[eval] loaded {len(df)} rows from {src}", flush=True)
    if max_rows:
        df = df.head(max_rows)
    pairs = []
    for i in range(len(df)):
        row = df.iloc[i]
        try:
            lib = {"workflow_name": row["workflow"], "shared_entities": json.loads(row["shared_entities"]),
                   "functions": json.loads(row["functions"])}
            conv = {
                "conversation_id": (row["metadata"] and json.loads(row["metadata"]).get("conversation_id")) or f"{row['workflow']}-{i}",
                "workflow_name": row["workflow"], "domain": row["domain"],
                "messages": json.loads(row["messages"]), "metadata": json.loads(row["metadata"])}
            pairs.append((lib, conv))
        except Exception as e:
            print(f"  [warn] row {i} parse error: {e}", file=sys.stderr)
    print(f"[eval] {len(pairs)} conversations.", flush=True)
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Function-calling eval for Mistral-Small-4 (SyntheticGen scoring, vLLM).")
    ap.add_argument("--data", default=DEFAULT_HF_URI)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="tokenizer/template source (the BASE id).")
    ap.add_argument("--vllm-url", default=None, help="vLLM OpenAI server base URL (serve the merged model).")
    ap.add_argument("--merged-dir", default=None, help="local MERGED bf16 model dir (transformers mode; "
                    "use when vLLM lacks mistral4 support).")
    ap.add_argument("--served-model", default=None, help="model name vLLM serves under / eval label.")
    ap.add_argument("--reasoning-effort", default="high", choices=["none", "high"])
    ap.add_argument("--max-rows", type=int, default=25)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument("--workers", type=int, default=None,
                    help="concurrent convs (default: 8 for vLLM, 1 for local transformers).")
    args = ap.parse_args()
    if not (args.vllm_url or args.merged_dir or args.metrics_only):
        ap.error("pass --vllm-url or --merged-dir")
    RANK = int(os.environ.get("LOCAL_RANK", -1))   # >=0 -> under torchrun (TP); all ranks run the
    IS_MAIN = RANK <= 0                              # eval loop (collective generate), only rank0 writes
    if args.workers is None:
        args.workers = 8 if args.vllm_url else 1
    if RANK >= 0:
        args.workers = 1  # TP generate is one collective call; no threading

    label = (args.served_model or args.merged_dir or args.model_id)
    out_path = args.out or (label.replace("/", "_") + "_results.json")
    cache_path = args.cache or (out_path + ".cache.jsonl")

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as cf:
            for line in cf:
                line = line.strip()
                if line:
                    try:
                        e = json.loads(line)
                        cache[e["conversation_id"]] = e
                    except Exception:
                        pass
        print(f">> cache {cache_path}: {len(cache)} conversations already done", flush=True)

    if args.metrics_only:
        atr = [[TurnResult(**d) for d in e["turn_results"]] for e in cache.values()]
        m = aggregate(atr)
        print_summary(m, len(atr), sum(1 for e in cache.values() if e.get("error")), label + " [cache]")
        return

    pairs = load_pairs(args.data, args.max_rows)
    lm = Model(args.model_id, args.max_new_tokens, args.vllm_url, args.served_model,
               args.reasoning_effort, merged_dir=args.merged_dir)

    all_turn_results, conv_results, n_errors = [], [], 0
    t0 = time.time()
    io_lock = threading.Lock()

    def process_one(n, lib, conv):
        conv_id = conv.get("conversation_id", f"conv-{n}")
        if conv_id in cache:
            e = cache[conv_id]
            with io_lock:
                print(f"  [{n}/{len(pairs)}] {conv_id} CACHED", flush=True)
            return [TurnResult(**d) for d in e["turn_results"]], e["conv_result"], e.get("error")
        tools, fn_schema_map, fn_names = build_openai_tools(lib)
        turn_results, error = evaluate_conversation(lm, conv, tools, fn_names, fn_schema_map)
        conv_result = {"conversation_id": conv_id, "workflow_name": conv.get("workflow_name"),
                       "domain": conv.get("domain"), "error": error,
                       "turns": [{"turn": tr.turn, "ref_call_count": tr.ref_call_count,
                                  "model_call_count": tr.model_call_count, "tp": tr.tp, "fp": tr.fp,
                                  "fn": tr.fn_miss, "tn": tr.tn, "ref_names": tr.ref_name_set,
                                  "model_names": tr.model_name_set, "json_valid": tr.json_valid,
                                  "hallucinated": tr.hallucinated} for tr in turn_results]}
        if IS_MAIN:
            with io_lock:
                with open(cache_path, "a") as cf:
                    cf.write(json.dumps({"conversation_id": conv_id, "error": error,
                                         "conv_result": conv_result,
                                         "turn_results": [asdict(tr) for tr in turn_results]}) + "\n")
                print(f"  [{n}/{len(pairs)}] {conv_id} turns={len(turn_results)}{' ERR:'+error if error else ''}", flush=True)
        return turn_results, conv_result, error

    if args.workers <= 1:
        # sequential (REQUIRED under TP: generate is a collective; must run in the main thread,
        # in identical order on every rank, so the NCCL collectives stay in lockstep).
        for n, (lib, conv) in enumerate(pairs, 1):
            try:
                tr, cr, err = process_one(n, lib, conv)
            except Exception as e:
                n_errors += 1
                print(f"  [conv crashed] {e}", file=sys.stderr, flush=True)
                continue
            all_turn_results.append(tr)
            conv_results.append(cr)
            if err:
                n_errors += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one, n, lib, conv) for n, (lib, conv) in enumerate(pairs, 1)]
            for fut in as_completed(futs):
                try:
                    tr, cr, err = fut.result()
                except Exception as e:
                    n_errors += 1
                    print(f"  [conv crashed] {e}", file=sys.stderr, flush=True)
                    continue
                all_turn_results.append(tr)
                conv_results.append(cr)
                if err:
                    n_errors += 1

    if not IS_MAIN:
        return  # non-rank0 TP workers just participated in the collective generates
    metrics = aggregate(all_turn_results)
    print_summary(metrics, len(conv_results), n_errors, label)
    print(f"\n  Headline tool_call_f1 = {metrics['tool_call_f1']:.4f}  "
          f"(P {metrics['tool_call_precision']:.4f}, R {metrics['tool_call_recall']:.4f})  "
          f"name_set_f1 = {metrics['name_set_f1']:.4f}  |  elapsed {time.time()-t0:.0f}s")
    Path(out_path).write_text(json.dumps({"model": label, "model_id": args.model_id,
        "served_model": args.served_model, "split": args.data, "metrics": metrics,
        "conversations": conv_results}, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
