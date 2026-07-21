#!/usr/bin/env python3
"""LLM LoRA finetune — google/gemma-4-26B-A4B-it (MoE) on ds-a2b2a342. Each invocation
launches ONE single-node run (FSDP2 over its pinned GPUs) via the gateway HTTP API —
no `sweep` dict, no cross-host distributed training. Several independent runs (e.g. one
on `tm` 0-3, two on `tm-2` split 0-3/4-7) are launched separately and tracked by hand.

Usage:
    .venv/bin/python automation/llm_sweep_gemma_a4b.py --probe --max-steps 3                     # 1-GPU timing probe
    .venv/bin/python automation/llm_sweep_gemma_a4b.py --provider tm  --visible-devices 0,1,2,3 --max-steps 800 --lora-r 64  --lr 5e-5
    .venv/bin/python automation/llm_sweep_gemma_a4b.py --provider tm2 --visible-devices 0,1,2,3 --max-steps 400 --lora-r 64  --lr 1e-4
    .venv/bin/python automation/llm_sweep_gemma_a4b.py --provider tm2 --visible-devices 4,5,6,7 --max-steps 400 --lora-r 128 --lr 5e-5
    .venv/bin/python automation/llm_sweep_gemma_a4b.py --status train-xxxx                        # re-attach + poll
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
API_KEY = "sgpu_mDDk7sDuLaNzuTcU0q_a_4T0QrAXdOzFRbE25STXg9c"   # automation/config.yaml's local key
STORAGE_ID = "store-22bad781"          # store-22bad781 -> s3://huseinlabel-app-test
DATASET_ID = "ds-a2b2a342"             # LLM-merge-2026-07-20-llm-packed, arch=gemma, 65384 seq_len, 3128 bins
BASE_MODEL = "google/gemma-4-26B-A4B-it"
PROVIDER_IDS = {"tm": "prov-91b8f69a", "tm2": "prov-940055fc"}

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
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError:
            return r.text

    def get(self, path: str, **kw):
        return self._req("GET", path, **kw)

    def post(self, path: str, body: dict, **kw):
        return self._req("POST", path, json=body, **kw)


def build_body(*, max_steps: int, lora_r: int, learning_rate: float, run_name: str,
               provider_id: str, visible_devices: str) -> dict:
    n_gpus = len([x for x in visible_devices.split(",") if x.strip()])
    return {
        "name": run_name,
        "dataset_id": DATASET_ID,
        "base_model": BASE_MODEL,
        "task_type": "llm",
        "training_type": "sft",
        "no_eval": True,                # ds-a2b2a342 has no held-out test split
        "max_epochs": 100,               # high ceiling; max_steps is the real cap
        "max_steps": max_steps,
        "batch_size": 1,
        "grad_accum": 1,
        "learning_rate": learning_rate,
        "warmup_steps": 0,
        "lr_scheduler_type": "constant",
        "weight_decay": 0.0,
        "use_lora": True,
        "lora_r": lora_r,
        "lora_alpha_ratio": 2.0,
        "lora_dropout": 0.05,
        # attention (q/k/v/o) + MLP (gate/up/down) -- adapts the dense FFN alongside attention.
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
        "gemma_fa4": True,
        # gemma default cpu_offload=ON (cfg omits it -> per-arch default); 65k context
        # on H20 needs it (see gateway/gateway/training/llm/CLAUDE.md).
        "train_embeddings": True,       # full-train embed+lm_head (tied on gemma-4) alongside LoRA
        "no_moe_lora": False,           # keep the fused MoE expert-adapter ON (the gemma-MoE default)
        "logging_steps": 1,
        "provider_id": provider_id,
        "visible_devices": visible_devices,
        "gpu_count": n_gpus,
        "use_ddp": True,                # multi-GPU single run -> torchrun DDP/FSDP over the pinned GPUs
        "storage_id": STORAGE_ID,
        "work_dir": "/share",
        "venv_path": "/share/autotrain-llm-gemma",
        "cleanup_checkpoints": True,
        "env_vars": {"HF_HOME": "/share/huggingface", "HF_HUB_DISABLE_XET": "1"},
    }


def poll(gw: Gateway, run_id: str, *, watch_timeout: float) -> dict:
    deadline = time.monotonic() + watch_timeout
    last_status = None
    last_log_len = 0
    while True:
        rec = gw.get(f"/v1/training-runs/{run_id}")
        st = rec.get("status") or "?"
        if st != last_status:
            log(f"{run_id}: {st}")
            last_status = st
        try:
            logs = gw.get(f"/v1/training-runs/{run_id}/logs", params={"tail": 400})
            lines = logs.get("lines") or []
            for line in lines[last_log_len:]:
                s = line.strip()
                if s and ("@@STEP" in s or "@@DONE" in s or "[train]" in s
                          or "[trainer]" in s or "error" in s.lower()):
                    print(f"    {s}")
            last_log_len = len(lines)
        except Exception as e:  # noqa: BLE001
            log(f"(log fetch failed: {e})")
        if st in RUN_TERMINAL:
            return rec
        if time.monotonic() > deadline:
            log(f"STOPPED WATCHING after {watch_timeout:.0f}s (status={st}); run continues server-side")
            return rec
        time.sleep(10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=3, help="cap on optimizer steps")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--name", default=None)
    ap.add_argument("--provider", choices=["tm", "tm2"], default="tm")
    ap.add_argument("--visible-devices", default="0,1,2,3")
    ap.add_argument("--status", default=None, help="poll an existing run id instead of creating one")
    ap.add_argument("--watch-timeout", type=float, default=12 * 3600)
    args = ap.parse_args()

    gw = Gateway(GATEWAY_URL, API_KEY)

    if args.status:
        rec = poll(gw, args.status, watch_timeout=args.watch_timeout)
        print(json.dumps(rec.get("result_json"), indent=2))
        return

    provider_id = PROVIDER_IDS[args.provider]
    run_name = args.name or f"gemma-4-26b-a4b-lora-{args.provider}-{args.visible_devices.replace(',', '')}"
    body = build_body(max_steps=args.max_steps, lora_r=args.lora_r, learning_rate=args.lr,
                       run_name=run_name, provider_id=provider_id, visible_devices=args.visible_devices)

    log("=" * 72)
    log(f"creating run {run_name!r} provider={args.provider} gpus={args.visible_devices} "
        f"max_steps={args.max_steps} lora_r={args.lora_r} lr={args.lr}")
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
