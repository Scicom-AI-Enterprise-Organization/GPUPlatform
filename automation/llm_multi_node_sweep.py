#!/usr/bin/env python3
"""LLM LoRA sweep across MULTIPLE nodes — google/gemma-4-26B-A4B-it (MoE) on
ds-a2b2a342, distributed over several (provider, gpu-range) node specs. This is
a NORMAL POST /v1/training-runs with a `nodes` list instead of a single
provider_id/visible_devices — the run behaves exactly like an ASR/TTS/single-box
LLM sweep (one TrainingRun row, result_json.trials/best), except each trial can
land on a different box. Each trial is still a single-node run (no cross-host
distributed training); the gateway fans the sweep grid across nodes and queues
overflow trials onto whichever node frees up next.

Usage:
    .venv/bin/python automation/llm_multi_node_sweep.py
    .venv/bin/python automation/llm_multi_node_sweep.py --status train-xxxx   # re-attach
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Any

try:
    import httpx
except ImportError as e:  # pragma: no cover
    sys.exit(f"missing dependency: {e}. Run with the repo venv: .venv/bin/python {sys.argv[0]}")

GATEWAY_URL = "http://localhost:8080"
API_KEY = "sgpu_mDDk7sDuLaNzuTcU0q_a_4T0QrAXdOzFRbE25STXg9c"
STORAGE_ID = "store-22bad781"
DATASET_ID = "ds-a2b2a342"
BASE_MODEL = "google/gemma-4-26B-A4B-it"

NODES = [
    {"provider_id": "prov-91b8f69a", "visible_devices": "0,1,2,3"},        # tm      GPUs 0-3
    {"provider_id": "prov-940055fc", "visible_devices": "0,1,2,3"},        # tm-2    GPUs 0-3
    {"provider_id": "prov-940055fc", "visible_devices": "4,5,6,7"},        # tm-2    GPUs 4-7
]

RUN_TERMINAL = {"done", "failed", "cancelled"}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Gateway:
    def __init__(self, base_url: str, api_key: str):
        self.cli = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=180.0, follow_redirects=True,
        )

    def _req(self, method: str, path: str, **kw) -> Any:
        r = self.cli.request(method, path, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:800]}")
        return r.json()

    def get(self, path: str, **kw):
        return self._req("GET", path, **kw)

    def post(self, path: str, body: dict, **kw):
        return self._req("POST", path, json=body, **kw)


def build_body(*, max_steps: int) -> dict:
    return {
        "name": "gemma-4-26b-a4b-lora-multinode",
        "dataset_id": DATASET_ID,
        "base_model": BASE_MODEL,
        "task_type": "llm",
        "training_type": "sft",
        "no_eval": True,
        "max_epochs": 100,
        "max_steps": max_steps,
        "batch_size": 1,
        "grad_accum": 1,
        "learning_rate": 5e-5,          # overridden per-trial by sweep.learning_rate
        "warmup_steps": 0,
        "lr_scheduler_type": "constant",
        "weight_decay": 0.0,
        "use_lora": True,
        "lora_r": 64,                   # overridden per-trial by sweep.lora_r
        "lora_alpha_ratio": 2.0,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
        "gemma_fa4": True,
        "train_embeddings": True,
        "no_moe_lora": False,
        "logging_steps": 1,
        "storage_id": STORAGE_ID,
        "work_dir": "/share",
        "venv_path": "/share/autotrain-llm-gemma",
        "cleanup_checkpoints": True,
        "env_vars": {"HF_HOME": "/share/huggingface", "HF_HUB_DISABLE_XET": "1"},
        "sweep": {"lora_r": [64, 128], "learning_rate": [5e-5, 1e-4]},
        "gpus_per_trial": 4,
        "nodes": NODES,
    }


def poll(gw: Gateway, run_id: str, *, watch_timeout: float) -> dict:
    deadline = time.monotonic() + watch_timeout
    last_status = None
    last_trials: dict = {}
    while True:
        rec = gw.get(f"/v1/training-runs/{run_id}")
        st = rec.get("status") or "?"
        if st != last_status:
            log(f"{run_id}: {st}")
            last_status = st
        for t in (rec.get("result_json") or {}).get("trials", []):
            key = t["trial"]
            snap = (t.get("run_id"), t.get("status"), t.get("metric"))
            if last_trials.get(key) != snap:
                last_trials[key] = snap
                node = t.get("node") or {}
                log(f"trial {t['trial']} params={t['params']} node={node.get('provider_id')}:"
                    f"{node.get('visible_devices')} run_id={t.get('run_id')} "
                    f"status={t.get('status')} metric={t.get('metric')}")
        if st in RUN_TERMINAL:
            return rec
        if time.monotonic() > deadline:
            log(f"STOPPED WATCHING after {watch_timeout:.0f}s; sweep continues server-side")
            return rec
        time.sleep(10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--status", default=None, help="poll an existing run_id instead of creating one")
    ap.add_argument("--watch-timeout", type=float, default=12 * 3600)
    args = ap.parse_args()

    gw = Gateway(GATEWAY_URL, API_KEY)

    if args.status:
        rec = poll(gw, args.status, watch_timeout=args.watch_timeout)
        print(json.dumps(rec.get("result_json"), indent=2))
        return

    body = build_body(max_steps=args.max_steps)
    log("=" * 72)
    log(f"creating multi-node sweep: {len(NODES)} nodes, max_steps={args.max_steps}")
    log(json.dumps(body, indent=2))
    run = gw.post("/v1/training-runs", body)
    run_id = run["id"]
    log(f"-> {run_id} status={run.get('status')}")

    rec = poll(gw, run_id, watch_timeout=args.watch_timeout)
    log("=" * 72)
    log(f"FINAL status={rec.get('status')}")
    print(json.dumps(rec.get("result_json"), indent=2))


if __name__ == "__main__":
    main()
