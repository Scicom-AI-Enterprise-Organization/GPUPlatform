#!/usr/bin/env python3
"""Offline vLLM batched generation for the LLM label export.

Runs in the dedicated **vLLM venv** (not the training venv) on a MERGED model dir (produced by the
arch's merge-to-disk). Loads the model once, batch-generates one assistant response per eval row via
`LLM.chat`, and emits the same manifest line the gateway parses from the label-export stream:

    @@LABEL {"items": [{"messages":[…,{"role":"assistant","content":…}], "lov":"…", "language":"…"}], "count": N}
    @@LABEL {"error": "…"}

Config (JSON via --config):
  {
    "merged_dir":     "/share/…/merged",      # standard HF checkpoint vLLM loads
    "eval_rows":      [{"messages":[{"role":"user","content":"…"}], "lov":"…", "language":"…"}, …],
    "n_samples":      110,
    "max_new_tokens": 512,
    "tp":             1,                        # tensor-parallel size (= GPU count)
    "gpu_mem_util":   0.85,
    "max_model_len":  65536
  }
"""
from __future__ import annotations

import argparse
import json
import sys


def emit(obj: dict) -> None:
    print("@@LABEL " + json.dumps(obj), flush=True)


def log(m: str) -> None:
    print(f"[llm-vllm-infer] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)

    merged = cfg["merged_dir"]
    eval_rows = cfg.get("eval_rows") or []
    n = int(cfg.get("n_samples") or 0) or len(eval_rows)
    eval_rows = eval_rows[:n]
    if not eval_rows:
        emit({"error": "no eval rows in config"})
        return 1
    max_new = int(cfg.get("max_new_tokens") or 512)
    tp = max(1, int(cfg.get("tp") or 1))
    gmu = float(cfg.get("gpu_mem_util") or 0.85)
    max_len = int(cfg.get("max_model_len") or 65536)

    from vllm import LLM, SamplingParams

    log(f"loading {merged} (tp={tp}, gpu_mem_util={gmu}, max_model_len={max_len}) …")
    llm = LLM(
        model=merged,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gmu,
        max_model_len=max_len,
        trust_remote_code=True,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)

    # Keep only rows with a non-empty messages list; remember their original index so the
    # lov/language passthrough lines up with the generated outputs.
    convs, keep = [], []
    for i, row in enumerate(eval_rows):
        msgs = list(row.get("messages") or [])
        if msgs:
            convs.append(msgs)
            keep.append(i)
    if not convs:
        emit({"error": "all eval rows had empty messages"})
        return 1

    log(f"generating {len(convs)} response(s) (max_new_tokens={max_new}) …")
    outs = llm.chat(convs, sp)

    items = []
    for idx, out in zip(keep, outs):
        row = eval_rows[idx]
        resp = (out.outputs[0].text if out.outputs else "").strip()
        items.append({
            "messages": list(row.get("messages") or []) + [{"role": "assistant", "content": resp}],
            "lov": row.get("lov", ""),
            "language": row.get("language", ""),
        })
        log(f"[{len(items)}/{len(convs)}] {len(resp)} chars")

    if not items:
        emit({"error": "generation produced no items"})
        return 1
    emit({"items": items, "count": len(items)})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        emit({"error": str(e)})
        sys.exit(1)
