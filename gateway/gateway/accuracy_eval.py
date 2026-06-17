#!/usr/bin/env python3
"""Accuracy eval for the GPUPlatform benchmark feature.

Runs *on the GPU box* (shipped to the VM/pod by ``pyremote_shim``), alongside
benchmaq. For every ``benchmark:`` item in the run config that carries an
``accuracy:`` block, it:

  1. serves the item's model with benchmaq's :class:`VLLMServer` (same serve
     args, same kebab-case kwarg convention as the throughput bench, so a
     config can reuse the exact same ``serve:`` block);
  2. loads each requested dataset (GSM8K, ``openai/MMMLU`` multilingual MMLU)
     from the HF hub;
  3. fires the questions concurrently at the local OpenAI-compatible endpoint
     and scores the answers (flexible last-number match for GSM8K, single
     letter for multiple-choice MMLU);
  4. times those same requests for a decode tok/s number, so each served
     config yields BOTH an accuracy and a speed — the two axes of the
     IQ-vs-speed plot — from a single serve.

Each (config, dataset) result is emitted as one ``@@ACCURACY {json}`` line on
stdout; the gateway (`bench.py`) scans the streamed log for these and folds
them into the run's ``result_json['accuracy']``. Marker convention matches the
``@@AUDIO``/``@@METRIC``/``@@LABEL`` lines used elsewhere in the platform.

Self-contained: only ``benchmaq`` (already installed for the bench), ``requests``
(a benchmaq dep), ``datasets`` (installed by the shim when accuracy is enabled)
and ``yaml``. No platform imports — it runs in the isolated benchmark venv.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import yaml

try:
    from benchmaq.vllm.core import VLLMServer
except Exception:  # pragma: no cover - import shape can vary across benchmaq versions
    from benchmaq.vllm import VLLMServer  # type: ignore


# Default multilingual spread for openai/MMMLU when the config doesn't pin
# languages. openai/MMMLU configs are language codes (no English — that's the
# original cais/mmlu); this is a representative cross-script set.
DEFAULT_MMMLU_LANGS = ["FR_FR", "DE_DE", "ES_LA", "ZH_CN", "JA_JP"]

# GLM-5.1 (and most modern models served here) are reasoning models: they emit
# a long <think> block before the answer. The generation must be allowed to
# finish thinking AND emit the final answer, or `content` comes back empty and
# every question scores wrong. So the default budget is generous; override via
# `accuracy.max_tokens`. With a reasoning_parser configured, vLLM puts the
# thinking in `reasoning_content` and the clean answer in `content`.
DEFAULT_MAX_TOKENS = 2048

# Names that select the hard multi-turn function-calling benchmark
# (Scicom-intl/Function-Call-TaaS). It's scored by the vendored fc_eval.py
# (SyntheticGen's evaluator) run as a subprocess against the live endpoint,
# not by the simple Q→A path — so it's dispatched separately.
FUNCTION_CALL_NAMES = {
    "function-call", "function_call", "function-calling", "functioncall",
    "taas", "scicom-intl/function-call-taas",
}
# Where the shim drops the vendored evaluator (+ a local-test fallback).
FC_EVAL_CANDIDATES = [
    "/tmp/sgpu_fc_eval.py",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fc_eval.py"),
]


def _is_function_call(spec) -> bool:
    name = spec if isinstance(spec, str) else (spec.get("name") if isinstance(spec, dict) else "")
    return str(name).strip().lower() in FUNCTION_CALL_NAMES


def _fc_eval_path():
    for p in FC_EVAL_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _emit(obj: dict) -> None:
    """Print one machine-readable marker line for the gateway to parse."""
    sys.stdout.write("@@ACCURACY " + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    sys.stdout.write(f"[accuracy] {msg}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------- datasets ---
def _load_gsm8k(limit, max_tokens):
    from datasets import load_dataset

    last_err = None
    ds = None
    # The bare `gsm8k` id no longer resolves on datasets>=4 ("repo id must be
    # namespace/name"); openai/gsm8k is the canonical mirror.
    for repo in ("openai/gsm8k", "gsm8k"):
        try:
            ds = load_dataset(repo, "main", split="test")
            break
        except Exception as e:  # noqa: PERF203
            last_err = e
            ds = None
    if ds is None:
        raise RuntimeError(f"could not load gsm8k: {last_err}")
    if limit:
        ds = ds.select(range(min(int(limit), len(ds))))
    items = []
    for r in ds:
        gold = str(r["answer"]).split("####")[-1].strip().replace(",", "").replace("$", "")
        prompt = (
            str(r["question"]).strip()
            + "\n\nSolve this step by step. End your reply with the final answer "
            "on its own line as:\n#### <number>"
        )
        items.append({"prompt": prompt, "gold": gold, "kind": "number", "max_tokens": max_tokens})
    return items


def _first(row, *keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _load_mmmlu(limit, languages, max_tokens):
    from datasets import load_dataset

    langs = languages or DEFAULT_MMMLU_LANGS
    if isinstance(langs, str):
        langs = [langs]
    items = []
    # Split the total budget across the requested languages so `limit` is a
    # total sample count, not per-language.
    per = max(1, int(limit) // len(langs)) if limit else None
    for lang in langs:
        try:
            ds = load_dataset("openai/MMMLU", lang, split="test")
        except Exception as e:
            _log(f"openai/MMMLU config {lang!r} failed to load ({e}); skipping")
            continue
        if per:
            ds = ds.select(range(min(per, len(ds))))
        for r in ds:
            q = _first(r, "Question", "question")
            choices = [_first(r, k, k.lower()) for k in ("A", "B", "C", "D")]
            gold = str(_first(r, "Answer", "answer") or "").strip().upper()[:1]
            if q is None or gold not in {"A", "B", "C", "D"}:
                continue
            body = str(q).strip() + "\n" + "\n".join(
                f"{ltr}. {ch}" for ltr, ch in zip("ABCD", choices)
            )
            items.append({
                "prompt": body + "\n\nReply with just the single letter (A, B, C, or D) of the correct answer.",
                "gold": gold,
                "kind": "letter",
                "max_tokens": max_tokens,
                "lang": lang,
            })
    return items


def _load_dataset_spec(spec, default_limit, default_langs, default_max_tokens):
    """Normalise a dataset spec (str or dict) → (display_name, items)."""
    if isinstance(spec, str):
        name, limit, langs, max_tokens = spec, default_limit, default_langs, default_max_tokens
    else:
        name = spec.get("name")
        limit = spec.get("limit", default_limit)
        langs = spec.get("languages", default_langs)
        max_tokens = spec.get("max_tokens", default_max_tokens)
    key = str(name).strip().lower()
    if key in ("gsm8k", "openai/gsm8k"):
        return "gsm8k", _load_gsm8k(limit, max_tokens)
    if key in ("openai/mmmlu", "mmmlu", "mmlu_multilingual"):
        return "openai/MMMLU", _load_mmmlu(limit, langs, max_tokens)
    raise RuntimeError(f"unknown accuracy dataset: {name!r}")


# ---------------------------------------------------------------- scoring ----
def _extract_number(text: str):
    text = text or ""
    # Prefer an explicit `#### <n>` marker (what the prompt asks for); else the
    # last number in the text — for a reasoning model the final answer is last.
    m = re.findall(r"####\s*(-?\$?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "").replace("$", "")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def _extract_letter(text: str):
    t = (text or "").strip().upper()
    if not t:
        return None
    # Prefer an explicit "answer is/: X"; else the LAST standalone A–D (the
    # conclusion comes last, after any reasoning). Fall back to any A–D.
    m = re.findall(r"ANSWER\s*(?:IS|:|=)?\s*\(?([ABCD])\)?", t)
    if m:
        return m[-1]
    m = re.findall(r"\b([ABCD])\b", t)
    if m:
        return m[-1]
    m = re.findall(r"([ABCD])", t)
    return m[-1] if m else None


def _is_correct(item: dict, completion: str) -> bool:
    if item["kind"] == "number":
        pred = _extract_number(completion)
        if pred is None:
            return False
        try:
            return abs(float(pred) - float(item["gold"])) < 1e-4
        except ValueError:
            return pred == item["gold"]
    return _extract_letter(completion) == item["gold"]


# ---------------------------------------------------------------- requests ---
def _run_dataset(base_url: str, model: str, items: list, concurrency: int) -> dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    results = [None] * len(items)

    def worker(idx_item):
        idx, it = idx_item
        body = {
            "model": model,
            "messages": [{"role": "user", "content": it["prompt"]}],
            "temperature": 0,
            "max_tokens": it["max_tokens"],
        }
        out_tok = 0
        try:
            resp = requests.post(url, json=body, timeout=1800)
            resp.raise_for_status()
            j = resp.json()
            msg = j["choices"][0]["message"]
            # With a reasoning_parser the clean answer is in `content` and the
            # <think> block in `reasoning_content`. Fall back to reasoning_content
            # if content is empty (some configs/parsers leave the answer there).
            txt = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "")
            out_tok = int((j.get("usage") or {}).get("completion_tokens", 0) or 0)
        except Exception as e:  # noqa: BLE001
            results[idx] = (False, 0, str(e))
            return
        results[idx] = (_is_correct(it, txt), out_tok, None)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        list(ex.map(worker, enumerate(items)))
    elapsed = time.time() - t0

    correct = sum(1 for r in results if r and r[0])
    out_tokens = sum(r[1] for r in results if r)
    errors = sum(1 for r in results if r and r[2])
    n = len(items)
    return {
        "n": n,
        "correct": correct,
        "errors": errors,
        "accuracy": (correct / n) if n else 0.0,
        "output_tokens": out_tokens,
        "elapsed_s": round(elapsed, 2),
        "output_tok_s": round(out_tokens / elapsed, 1) if elapsed > 0 else 0.0,
    }


# ---------------------------------------------------------------- serve -------
def _run_function_call(base_url, model, spec, default_limit, concurrency, idx, config_name) -> dict:
    """Score the multi-turn function-calling benchmark by running the vendored
    SyntheticGen evaluator (fc_eval.py) against the live endpoint. Returns a
    result dict shaped like _run_dataset's, with the headline `tool_call_f1` as
    `accuracy` and the full metric set under `metrics`."""
    import shutil

    cfg = spec.get("config", "test-basic") if isinstance(spec, dict) else "test-basic"
    split = spec.get("split", "train") if isinstance(spec, dict) else "train"
    limit = spec.get("limit", default_limit) if isinstance(spec, dict) else default_limit
    workers = int(spec.get("workers", min(concurrency, 16))) if isinstance(spec, dict) else min(concurrency, 16)

    path = _fc_eval_path()
    if not path:
        raise RuntimeError("fc_eval.py not found on the box (not shipped?)")
    # Unique paths PER CONFIG — fc_eval checkpoints by *model name*, and two
    # serve configs can share a model (fp8-epoff vs fp8-kvfp8 are both
    # zai-org/GLM-5.1-FP8), so an idx-only path made the 2nd config resume the
    # 1st's checkpoint / read its stale output. Key by config name + idx, and
    # wipe both first so every run is fresh and a crash fails loudly (no stale
    # read) rather than silently reporting the previous config's numbers.
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{config_name}_{idx}")
    out_path = f"/tmp/sgpu_fc_out_{safe}.json"
    ckpt_dir = f"/tmp/sgpu_fc_ckpt_{safe}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    try:
        os.remove(out_path)
    except OSError:
        pass
    cmd = [
        sys.executable, path,
        "--base-url", base_url.rstrip("/") + "/v1",
        "--api-key", "none",
        "--model", model,
        "--config", str(cfg),
        "--split", str(split),
        "--max-conversations", str(int(limit or 0)),
        "--workers", str(workers),
        "--checkpoint-dir", ckpt_dir,
        "--output", out_path,
    ]
    _log(f"function-call: {cfg}/{split}, max_conversations={limit or 'all'}, workers={workers}")
    t0 = time.time()
    # Stream the evaluator's progress straight to our stdout (→ bench log).
    subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    if not os.path.exists(out_path):
        raise RuntimeError("fc_eval wrote no output (crashed before finishing all conversations?)")
    with open(out_path) as f:
        data = json.load(f)
    m = data.get("metrics", {}) or {}
    convs = data.get("conversations", []) or []
    n = len(convs)
    errors = sum(1 for c in convs if c.get("error"))
    return {
        "n": n,
        "errors": errors,
        "accuracy": float(m.get("tool_call_f1") or 0.0),  # headline for the IQ axis
        "metrics": m,
        "elapsed_s": round(elapsed, 2),
        "output_tok_s": None,  # the evaluator doesn't meter tokens
    }


def _served_model_name(base_url: str, fallback: str) -> str:
    """vLLM registers the model under the path/repo it was served with; ask the
    endpoint so the chat requests don't 404 on a name mismatch."""
    try:
        r = requests.get(base_url.rstrip("/") + "/v1/models", timeout=30)
        r.raise_for_status()
        data = (r.json() or {}).get("data") or []
        if data and data[0].get("id"):
            return data[0]["id"]
    except Exception:
        pass
    return fallback


def _resolve_model(item: dict, serve: dict):
    """Mirror benchmaq's model resolution, but prefer an on-disk local_dir and
    fall back to repo_id (lets a config serve straight from the HF_HOME cache,
    the way the platform's big-model benchmarks do)."""
    model = serve.pop("model", None) or serve.pop("model_path", None)
    if model:
        return model
    mc = item.get("model") or {}
    local_dir = mc.get("local_dir")
    if local_dir and os.path.isdir(local_dir) and os.listdir(local_dir):
        return local_dir
    return mc.get("repo_id") or local_dir or ""


def _eval_item(item: dict) -> None:
    name = item.get("name", "benchmark")
    acc = item.get("accuracy") or {}
    serve = dict(item.get("serve") or {})
    model = _resolve_model(item, serve)
    port = int(serve.pop("port", 8000))
    concurrency = int(acc.get("concurrency", 32))
    default_limit = acc.get("limit")
    default_langs = acc.get("languages")
    default_max_tokens = int(acc.get("max_tokens", DEFAULT_MAX_TOKENS))
    dataset_specs = acc.get("datasets") or ["gsm8k"]

    if not model:
        _emit({"event": "error", "config": name, "error": "no model specified"})
        return

    _emit({"event": "serving", "config": name, "model": model, "serve_kwargs": serve})
    server = VLLMServer(model=model, port=port, **serve)
    healthy = False
    try:
        healthy = server.start()
        if not healthy:
            _emit({"event": "error", "config": name, "error": "vLLM server failed to become healthy"})
            return
        # vLLM may register the model under a normalised name — ask it so the
        # requests don't 404.
        req_model = _served_model_name(server.base_url, model)
        for idx, spec in enumerate(dataset_specs):
            # Hard multi-turn function-calling benchmark — scored by the
            # vendored evaluator subprocess, not the simple Q→A path.
            if _is_function_call(spec):
                try:
                    res = _run_function_call(server.base_url, req_model, spec, default_limit, concurrency, idx, name)
                    _emit({"event": "result", "config": name, "dataset": "Function-Call-TaaS", **res})
                    _log(f"{name}/Function-Call-TaaS: tool_call_f1={res['accuracy']:.3f} (n={res['n']})")
                except Exception as e:  # noqa: BLE001
                    _emit({"event": "error", "config": name, "dataset": "Function-Call-TaaS", "error": repr(e)})
                continue
            try:
                display, items = _load_dataset_spec(spec, default_limit, default_langs, default_max_tokens)
            except Exception as e:  # noqa: BLE001
                _emit({"event": "error", "config": name, "dataset": str(spec), "error": str(e)})
                continue
            if not items:
                _emit({"event": "error", "config": name, "dataset": display, "error": "no eval items loaded"})
                continue
            _log(f"{name}: evaluating {display} ({len(items)} items, conc {concurrency})")
            res = _run_dataset(server.base_url, req_model, items, concurrency)
            _emit({"event": "result", "config": name, "dataset": display, **res})
            _log(
                f"{name}/{display}: accuracy={res['accuracy']:.3f} "
                f"({res['correct']}/{res['n']}), {res['output_tok_s']} tok/s"
            )
    finally:
        try:
            server.stop()
        except Exception:
            pass


def main() -> int:
    if len(sys.argv) < 2:
        _log("usage: accuracy_eval.py <config.yaml>")
        return 2
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f) or {}
    items = [it for it in config.get("benchmark", []) if it.get("accuracy")]
    if not items:
        _log("no benchmark items with an accuracy: block — nothing to do")
        return 0
    _log(f"running accuracy eval for {len(items)} config(s)")
    for item in items:
        try:
            _eval_item(item)
        except Exception as e:  # noqa: BLE001
            _emit({"event": "error", "config": item.get("name", "benchmark"), "error": repr(e)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
