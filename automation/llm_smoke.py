#!/usr/bin/env python3
"""LLM multi-model smoke sweep — pack a chat dataset per-arch, then run a tiny
(max_steps) LoRA finetune of every dropdown base model on the VM box, one at a
time, capturing pass/fail + trainer logs for each.

Purpose: shake out per-arch autotrain bugs. Each model runs a real but minimal
run (default 3 optimizer steps) so "does this arch train end-to-end?" is answered
without a full run.

Flow:
  1. For each unique arch, POST /v1/datasets/{src}/pack-llm (seq_len 32k, that
     arch's tokenizer) → poll the source's transform_status → grab the created
     kind=llm_packed dataset id. Packs share across models of the same arch.
  2. For each model, POST /v1/training-runs (task_type=llm, max_steps=N, LoRA,
     per-arch cpu_offload/lora_r) on its arch's pack → watch to terminal.
  3. Print a summary; a JSON state file makes the whole thing resumable and
     records each model's final status + error + last log lines.

Sequential by necessity: the packs all mutate one source dataset (the endpoint
409s on concurrent transforms), and each run uses the whole box.

Usage:
    .venv/bin/python automation/llm_smoke.py [--dry-run] [--until pack]
        [--only MODEL_SUBSTR] [--fresh] [--max-steps N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import httpx
except ImportError as e:  # pragma: no cover
    sys.exit(f"missing dependency: {e}. Run with the repo venv: .venv/bin/python {sys.argv[0]}")

# --------------------------------------------------------------------------- #
# config — the /autotrain/new?task=llm dropdown, in cached/cheap → risky order
# --------------------------------------------------------------------------- #
GATEWAY_URL = "http://localhost:8080"
API_KEY = "admin"          # AUTH_DISABLED=1 locally → any bearer resolves to admin
SOURCE_DATASET = "ds-04c8899b"
STORAGE_ID = "store-22bad781"
PROVIDER_ID = "prov-940055fc"   # tm-2, VM, 8x H20-3e
SEQ_LEN = 32768
MAX_STEPS = 3

# arch → tokenizer used to pack + (cpu_offload, lora_r) training defaults
# (mirrors the web form's llmArch / llmCpuOffloadDefault / llmLoraDefaults).
ARCH = {
    "gemma":    {"tokenizer": "google/gemma-4-31B-it",                       "cpu_offload": True,  "lora_r": 64},
    "qwen":     {"tokenizer": "Qwen/Qwen3.6-27B",                            "cpu_offload": True,  "lora_r": 64},
    "nemotron": {"tokenizer": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",  "cpu_offload": False, "lora_r": 32},
    "minimax":  {"tokenizer": "MiniMaxAI/MiniMax-M2.7",                      "cpu_offload": False, "lora_r": 16},
    "mistral":  {"tokenizer": "mistralai/Mistral-Small-4-119B-2603",         "cpu_offload": False, "lora_r": 16},
}
MODELS = [
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.6-27B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "google/gemma-4-26B-A4B-it",
    "Qwen/Qwen3.6-35B-A3B",
    "mistralai/Mistral-Small-4-119B-2603",
    "MiniMaxAI/MiniMax-M2.7",
    "Qwen/Qwen3.5-122B-A10B",
]

RUN_TERMINAL = {"done", "failed", "cancelled"}
STATE_PATH = Path(__file__).resolve().parent / "state" / "llm_smoke.json"


def llm_arch(model: str) -> str:
    n = model.lower()
    if "minimax" in n:
        return "minimax"
    if "mistral" in n:
        return "mistral"
    if "qwen" in n:
        return "qwen"
    if "nemotron" in n:
        return "nemotron"
    return "gemma"


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
        attempts = 0
        while True:
            try:
                r = self.cli.request(method, path, **kw)
                break
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                attempts += 1
                if attempts > 30:
                    raise
                log(f"gateway unreachable ({e}); retry {attempts}/30 in 5s…")
                time.sleep(5)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:600]}")
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


class State:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {"packs": {}, "runs": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
                self.data.setdefault("packs", {})
                self.data.setdefault("runs", {})
            except Exception as e:  # noqa: BLE001
                log(f"WARNING: bad state {path}: {e} (fresh)")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


# --------------------------------------------------------------------------- #
# phase 1 — pack per arch
# --------------------------------------------------------------------------- #
def ensure_pack(gw: Gateway, state: State, arch: str, *, dry_run: bool, poll_timeout: float = 3600) -> Optional[str]:
    entry = state.data["packs"].get(arch)
    if entry and entry.get("packed_id"):
        log(f"[pack:{arch}] up-to-date ({entry['packed_id']}) — skipping")
        return entry["packed_id"]

    tok = ARCH[arch]["tokenizer"]
    log(f"[pack:{arch}] packing {SOURCE_DATASET} @ seq_len={SEQ_LEN} with tokenizer {tok}")
    if dry_run:
        log(f"[pack:{arch}] DRY-RUN: POST /v1/datasets/{SOURCE_DATASET}/pack-llm "
            f"{{tokenizer:{tok}, sequence_length:{SEQ_LEN}, storage_id:{STORAGE_ID}}}")
        return f"<dry-pack:{arch}>"

    gw.post(f"/v1/datasets/{SOURCE_DATASET}/pack-llm", {
        "storage_id": STORAGE_ID,
        "tokenizer": tok,
        "sequence_length": SEQ_LEN,
        "objective": "sft",
    })
    # poll the SOURCE dataset's transform_status; the pack runs in-process
    deadline = time.monotonic() + poll_timeout
    seen: set[str] = set()
    while True:
        rec = gw.get(f"/v1/datasets/{SOURCE_DATASET}")
        status = rec.get("transform_status") or ""
        tlog = rec.get("transform_log") or ""
        for line in tlog.splitlines():
            s = line.strip()
            if s and "[AUTOTRAIN_PROGRESS]" not in s and s not in seen:
                seen.add(s)
                log(f"    [pack:{arch}] {s}")
        if status == "done":
            m = re.search(r"created dataset (ds-[0-9a-f]+)", tlog)
            if not m:
                log(f"[pack:{arch}] FAILED to find created dataset id in log")
                return None
            packed_id = m.group(1)
            # verify arch stamped on the pack matches
            pd = gw.get(f"/v1/datasets/{packed_id}")
            pack_arch = (((pd.get("split_fields") or {}).get("_llm_pack")) or {}).get("arch")
            log(f"[pack:{arch}] -> {packed_id} (kind={pd.get('kind')}, rows={pd.get('num_rows')}, pack_arch={pack_arch})")
            if pack_arch and pack_arch != arch:
                log(f"[pack:{arch}] WARNING: pack arch '{pack_arch}' != expected '{arch}'")
            state.data["packs"][arch] = {"packed_id": packed_id, "tokenizer": tok, "pack_arch": pack_arch}
            state.save()
            return packed_id
        if status == "failed":
            log(f"[pack:{arch}] FAILED: {tlog[-400:]}")
            state.data["packs"][arch] = {"packed_id": None, "error": tlog[-400:]}
            state.save()
            return None
        if time.monotonic() > deadline:
            log(f"[pack:{arch}] TIMEOUT after {poll_timeout:.0f}s (status={status!r})")
            return None
        time.sleep(5)


# --------------------------------------------------------------------------- #
# phase 2 — smoke-train per model
# --------------------------------------------------------------------------- #
def run_model(gw: Gateway, state: State, model: str, packed_id: str, *, dry_run: bool,
              max_steps: int, watch_timeout: float = 3 * 3600) -> dict:
    arch = llm_arch(model)
    ad = ARCH[arch]
    key = model
    existing = state.data["runs"].get(key)
    # Only skip a model that actually PASSED — leave failed / cancelled / watch-timeout
    # retryable so a re-run after a bug fix re-attempts them (done ones stay skipped).
    if existing and existing.get("final_status") == "done":
        log(f"[run:{model}] already done ({existing.get('run_id')}) — skipping")
        return existing

    body = {
        "name": f"smoke-{model.split('/')[-1]}",
        "dataset_id": packed_id,
        "base_model": model,
        "task_type": "llm",
        "training_type": "sft",
        "max_steps": max_steps,
        "max_epochs": 1,
        "no_eval": True,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": 10_000,     # > max_steps → no eval/save mid-run
        "save_steps": 10_000,
        "batch_size": 1,
        "grad_accum": 1,
        "logging_steps": 1,
        "warmup_steps": 0,
        "learning_rate": 5e-5,
        "use_lora": True,
        "lora_r": ad["lora_r"],
        "lora_alpha_ratio": 2.0,
        "cpu_offload": ad["cpu_offload"],
        "train_embeddings": False,
        "gemma_fa4": True,               # ignored by non-gemma archs
        "storage_id": STORAGE_ID,
        "provider_id": PROVIDER_ID,
    }
    log(f"[run:{model}] arch={arch} pack={packed_id} steps={max_steps} "
        f"cpu_offload={ad['cpu_offload']} lora_r={ad['lora_r']}")
    if dry_run:
        log(f"[run:{model}] DRY-RUN: POST /v1/training-runs {json.dumps(body)}")
        return {"run_id": None, "final_status": "dry-run"}

    run = gw.post("/v1/training-runs", body)
    rid = run["id"]
    log(f"[run:{model}] -> {rid} status={run.get('status')}")
    state.data["runs"][key] = {"run_id": rid, "arch": arch, "packed_id": packed_id, "final_status": None}
    state.save()

    # watch to terminal
    deadline = time.monotonic() + watch_timeout
    last = None
    while True:
        rec = gw.get(f"/v1/training-runs/{rid}")
        st = rec.get("status") or "?"
        if st != last:
            log(f"[run:{model}] {rid}: {st}")
            last = st
        if st in RUN_TERMINAL:
            logs = gw.get(f"/v1/training-runs/{rid}/logs", params={"tail": 60})
            tail = (logs.get("lines") or [])[-40:]
            res = {
                "run_id": rid, "arch": arch, "packed_id": packed_id,
                "final_status": st, "error_text": rec.get("error_text"),
                "log_tail": tail,
            }
            state.data["runs"][key] = res
            state.save()
            mark = "✅" if st == "done" else "❌"
            log(f"[run:{model}] {mark} {st}" + (f" — {rec.get('error_text','')[:200]}" if rec.get("error_text") else ""))
            return res
        if time.monotonic() > deadline:
            log(f"[run:{model}] STOPPED WATCHING after {watch_timeout:.0f}s (status={st}); continues server-side")
            state.data["runs"][key]["final_status"] = f"watch-timeout:{st}"
            state.save()
            return state.data["runs"][key]
        time.sleep(15)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--until", choices=["pack", "train"], default="train")
    ap.add_argument("--only", default=None, help="substring filter on model id")
    ap.add_argument("--fresh", action="store_true", help="ignore existing state")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    if args.fresh and STATE_PATH.exists() and not args.dry_run:
        STATE_PATH.unlink()
        log(f"--fresh: removed {STATE_PATH}")
    state = State(STATE_PATH)
    gw = Gateway(GATEWAY_URL, API_KEY)

    models = [m for m in MODELS if (not args.only or args.only.lower() in m.lower())]
    archs = []
    for m in models:
        a = llm_arch(m)
        if a not in archs:
            archs.append(a)

    log("=" * 72)
    log(f"LLM smoke sweep — {len(models)} model(s), {len(archs)} arch(s), "
        f"seq_len={SEQ_LEN}, max_steps={args.max_steps}, provider={PROVIDER_ID}")
    log(f"source={SOURCE_DATASET} storage={STORAGE_ID} state={STATE_PATH}")
    if args.dry_run:
        log("DRY-RUN — no changes")

    # phase 1: pack per arch
    log("=" * 72)
    log(f"PHASE 1/2 — packing {len(archs)} arch(s) @ {SEQ_LEN}")
    packs: dict[str, Optional[str]] = {}
    for a in archs:
        packs[a] = ensure_pack(gw, state, a, dry_run=args.dry_run)

    if args.until == "pack":
        log("=" * 72)
        log("stopping after pack (--until pack)")
        log("packs: " + json.dumps({a: packs[a] for a in archs}))
        return

    # phase 2: train per model
    log("=" * 72)
    log(f"PHASE 2/2 — smoke-training {len(models)} model(s)")
    results: dict[str, dict] = {}
    for m in models:
        a = llm_arch(m)
        pid = packs.get(a)
        if not pid or str(pid).startswith("<"):
            if args.dry_run:
                run_model(gw, state, m, pid or f"<dry-pack:{a}>", dry_run=True, max_steps=args.max_steps)
                continue
            log(f"[run:{m}] SKIP — no pack for arch {a}")
            results[m] = {"final_status": f"skipped:no-pack-{a}"}
            continue
        results[m] = run_model(gw, state, m, pid, dry_run=args.dry_run, max_steps=args.max_steps)

    # summary
    log("=" * 72)
    log("SUMMARY")
    for m in models:
        r = state.data["runs"].get(m) or results.get(m) or {}
        st = r.get("final_status") or "?"
        mark = {"done": "✅", "dry-run": "·"}.get(st, "❌")
        log(f"  {mark} {m:48s} {st}" + (f"  ({r.get('run_id')})" if r.get("run_id") else ""))
    log("done.")


if __name__ == "__main__":
    main()
