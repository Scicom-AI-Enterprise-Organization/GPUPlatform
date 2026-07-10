#!/usr/bin/env python3
"""Autotrain pipeline orchestrator — config-driven, API-only.

Reads a YAML config (see config.yaml) and drives the whole flow through the
gateway HTTP API:

  1. import each Label-platform project        -> a kind=label dataset
  2. transform each to S3 (per-dataset test split) -> a kind=s3 "-audio" dataset
  3. merge all transformed datasets into ONE combined audio dataset
  4. fine-tune each model on the merged dataset (RunPod secure H100 SXM)

A JSON state file (default: automation/state/<config-stem>.json) makes re-runs
idempotent + resumable: add more datasets / models to the config and re-run —
already-created resources are skipped, only the new work (and a re-merge) runs.

Usage:
    .venv/bin/python automation/run_pipeline.py [--config PATH] [--cutoff "..."]
        [--gateway-url URL] [--api-key KEY] [--state PATH]
        [--fresh] [--dry-run] [--no-run-watch] [--watch-timeout SECONDS]

Nothing here needs the gateway source — it's a plain API client. Run it with any
python that has `httpx` and `pyyaml` (the repo .venv has both).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import httpx
    import yaml
except ImportError as e:  # pragma: no cover
    sys.exit(f"missing dependency: {e}. Run with the repo venv: .venv/bin/python {sys.argv[0]}")

ALL_AUGMENTATIONS = ["telephone", "noise", "dropout", "gain", "pitch", "speed", "reverb", "bandpass"]
DATASET_TERMINAL = {"done", "failed", "cancelled"}
RUN_TERMINAL = {"done", "failed", "cancelled"}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def _tz_from_offset(off: str) -> timezone:
    off = (off or "+00:00").strip()
    sign = 1
    if off and off[0] in "+-":
        sign = -1 if off[0] == "-" else 1
        off = off[1:]
    parts = off.split(":")
    hh = int(parts[0] or 0)
    mm = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    return timezone(sign * timedelta(hours=hh, minutes=mm))


def parse_cutoff(raw: Optional[str], default_offset: str) -> Optional[str]:
    """Parse a cutoff into a canonical ISO-8601 string with an offset. Accepts
    ISO-8601 (with/without offset or trailing Z) or friendly forms like
    '2026-07-01 11:59PM' / '2026-07-01 23:59'. A naive value gets `default_offset`."""
    s = (raw or "").strip()
    if not s:
        return None
    dt: Optional[datetime] = None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        for fmt in (
            "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %I:%M%p", "%Y-%m-%d %I:%M %p", "%Y-%m-%d %I%p", "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        die(f"cannot parse cutoff {raw!r} — use ISO-8601 or 'YYYY-MM-DD HH:MM' / '... 11:59PM'")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz_from_offset(default_offset))
    return dt.isoformat()


def resolve_augment(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        if v.strip().lower() == "all":
            return list(ALL_AUGMENTATIONS)
        return [t.strip() for t in v.split(",") if t.strip()]
    return [str(t).strip() for t in v if str(t).strip()]


# --------------------------------------------------------------------------- #
# gateway API client
# --------------------------------------------------------------------------- #
class GatewayError(RuntimeError):
    pass


class Gateway:
    def __init__(self, base_url: str, api_key: str, timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.cli = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            follow_redirects=True,
        )

    def request(self, method: str, path: str, **kw) -> Any:
        # Retry on connection-refused / connect-timeout so a gateway restart
        # mid-run (e.g. picking up a backend edit) doesn't kill the pipeline. A
        # ConnectError means the request never reached the server, so retrying is
        # safe for any method (no risk of a double POST).
        attempts = 0
        while True:
            try:
                r = self.cli.request(method, path, **kw)
                break
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                attempts += 1
                if attempts > 30:  # ~150s of gateway downtime tolerance
                    raise
                log(f"gateway unreachable ({e}); retry {attempts}/30 in 5s…")
                time.sleep(5)
        if r.status_code >= 400:
            raise GatewayError(f"{method} {path} -> HTTP {r.status_code}: {r.text[:600]}")
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError:
            return r.text

    def get(self, path: str, **kw) -> Any:
        return self.request("GET", path, **kw)

    def post(self, path: str, body: Optional[dict] = None, **kw) -> Any:
        return self.request("POST", path, json=body or {}, **kw)


# --------------------------------------------------------------------------- #
# state (idempotency / resume)
# --------------------------------------------------------------------------- #
class State:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {"datasets": {}, "merge": {}, "runs": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
                self.data.setdefault("datasets", {})
                self.data.setdefault("merge", {})
                self.data.setdefault("runs", {})
            except Exception as e:  # noqa: BLE001
                log(f"WARNING: could not read state {path}: {e} (starting fresh)")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


# --------------------------------------------------------------------------- #
# polling
# --------------------------------------------------------------------------- #
def wait_dataset(gw: Gateway, ds_id: str, *, timeout: float, label: str) -> dict:
    """Poll a dataset until its transform/merge finishes. Prints new log tail lines."""
    deadline = time.monotonic() + timeout
    emitted: set[str] = set()  # dedup by content — the gateway log is a rolling
    #                            buffer, so a prefix diff re-prints once it truncates.
    while True:
        rec = gw.get(f"/v1/datasets/{ds_id}")
        status = rec.get("transform_status") or ""
        for line in (rec.get("transform_log") or "").splitlines():
            s = line.strip()
            if s and "[AUTOTRAIN_PROGRESS]" not in s and s not in emitted:
                emitted.add(s)
                log(f"    {label}: {s}")
        if status in DATASET_TERMINAL:
            return rec
        if time.monotonic() > deadline:
            die(f"{label}: timed out after {timeout:.0f}s (status={status!r})")
        time.sleep(5)


def watch_runs(gw: Gateway, run_ids: list[str], *, timeout: float) -> None:
    """Poll training runs until all reach a terminal state (or timeout)."""
    deadline = time.monotonic() + timeout
    last: dict[str, str] = {}
    while True:
        statuses = {}
        for rid in run_ids:
            rec = gw.get(f"/v1/training-runs/{rid}")
            st = rec.get("status") or "?"
            statuses[rid] = st
            if last.get(rid) != st:
                extra = ""
                if st == "failed" and rec.get("error_text"):
                    extra = f" — {rec['error_text'][:200]}"
                log(f"    run {rid} ({rec.get('base_model','?')}): {st}{extra}")
                last[rid] = st
        if all(s in RUN_TERMINAL for s in statuses.values()):
            return
        if time.monotonic() > deadline:
            log(f"    (stopped watching after {timeout:.0f}s; runs continue server-side: "
                + ", ".join(f"{r}={s}" for r, s in statuses.items()) + ")")
            return
        time.sleep(15)


# --------------------------------------------------------------------------- #
# pipeline steps
# --------------------------------------------------------------------------- #
def resolve_storage(gw: Gateway, cfg: dict) -> str:
    sid = (cfg.get("storage_id") or "").strip()
    storages = gw.get("/v1/storage")
    by_id = {s["id"]: s for s in storages}
    if sid:
        s = by_id.get(sid)
        if not s:
            die(f"storage_id {sid!r} not found on the gateway")
        if s["kind"] != "s3":
            die(f"storage_id {sid!r} is kind={s['kind']} — must be kind=s3")
        return sid
    s3s = [s for s in storages if s["kind"] == "s3" and s.get("enabled", True)]
    if not s3s:
        die("no enabled kind=s3 storage on the gateway — set storage_id in the config")
    log(f"auto-picked S3 storage {s3s[0]['id']} ({s3s[0].get('name')})")
    return s3s[0]["id"]


def resolve_provider(gw: Gateway, want: Optional[str]) -> Optional[str]:
    if want and want.strip():
        return want.strip()
    provs = gw.get("/v1/providers")
    runpods = [p for p in provs if p.get("kind") == "runpod" and p.get("enabled", True)]
    if not runpods:
        die("no enabled RunPod provider on the gateway — set train.provider_id in the config")
    log(f"auto-picked RunPod provider {runpods[0]['id']} ({runpods[0].get('name')})")
    return runpods[0]["id"]


def dataset_signature(ds: dict, cutoff: Optional[str], status: str,
                      min_chars: Any, exclude_regex: Any, ref_id: Any = None) -> dict:
    sig = {
        "project_id": ds.get("project_id"),
        "cutoff": cutoff,
        "status": status,
        "test_split_pct": ds.get("test_split_pct"),
        "test_split_count": ds.get("test_split_count"),
    }
    # Only record these when set, so datasets created before the option existed
    # (and those not using it) still match their stored signature.
    if min_chars:
        sig["test_min_chars"] = int(min_chars)
    if exclude_regex:
        sig["test_exclude_regex"] = str(exclude_regex)
    if ref_id:
        sig["test_split_ref_dataset_id"] = str(ref_id)
    return sig


def ensure_dataset(
    gw: Gateway, cfg: dict, ds: dict, state: State, storage_id: str,
    *, cli_cutoff: Optional[str], dry_run: bool, poll_timeout: float,
) -> str:
    """Import + transform one dataset. Returns the transformed (kind=s3) id."""
    name = ds.get("name") or die("every dataset needs a `name`")
    project_id = ds.get("project_id") or die(f"dataset {name!r} needs a `project_id`")

    # cutoff precedence: per-dataset > CLI --cutoff > config default
    raw_cutoff = ds.get("cutoff") or cli_cutoff or cfg.get("cutoff")
    cutoff = parse_cutoff(raw_cutoff, cfg.get("timezone_offset", "+00:00"))
    status = (ds.get("label_status") or cfg.get("label_status") or "approved").strip()
    pct = ds.get("test_split_pct")
    cnt = ds.get("test_split_count")
    # Reuse ANOTHER dataset's exact test set instead of carving a random one: the
    # rows whose audio matches that dataset's `test` split become this dataset's
    # test set, everything else train. Guarantees no train/test overlap when this
    # dataset is a superset of the reference (e.g. a v2 import with a later cutoff).
    # Value = a dataset id (an S3/exported dataset, or a label dataset resolved to
    # its exported S3 twin). Mutually exclusive with pct/count.
    ref_id = (ds.get("test_split_ref_dataset_id") or "").strip() or None
    if ref_id and (pct not in (None, 0) or cnt not in (None, 0)):
        die(f"[{name}] set test_split_ref_dataset_id OR test_split_pct/test_split_count, not both")
    # Min transcription length (chars) for a row to be eligible for the test split;
    # per-dataset value, else the config-level default. Keeps junk transcripts
    # ("[silent]", "[unintelligible]") out of eval. Only applies when there's a test split.
    min_chars = ds.get("test_min_chars")
    if min_chars is None:
        min_chars = cfg.get("test_min_chars")
    # Regex; transcripts matching it are excluded from the test split (kept in train).
    exclude_regex = ds.get("test_exclude_regex")
    if exclude_regex is None:
        exclude_regex = cfg.get("test_exclude_regex")
    exclude_regex = (str(exclude_regex).strip() if exclude_regex else "") or None
    if exclude_regex:
        import re as _re
        try:
            _re.compile(exclude_regex)
        except _re.error as e:
            die(f"[{name}] test_exclude_regex is not a valid regex: {e}")
    has_test = bool(ref_id) or pct not in (None, 0) or cnt not in (None, 0)
    # min_chars / exclude_regex are eligibility filters for the RANDOM split only —
    # a reused test set is taken verbatim, so they don't apply in ref mode.
    eligibility = has_test and not ref_id
    sig = dataset_signature(
        ds, cutoff, status,
        min_chars if eligibility else None, exclude_regex if eligibility else None,
        ref_id=ref_id,
    )

    entry = state.data["datasets"].get(name) or {}
    if entry.get("transformed_id") and entry.get("signature") == sig:
        log(f"[{name}] up-to-date (transformed={entry['transformed_id']}) — skipping")
        return entry["transformed_id"]

    _notes = []
    if eligibility and min_chars:
        _notes.append(f"min {int(min_chars)} chars")
    if eligibility and exclude_regex:
        _notes.append(f"excl /{exclude_regex}/")
    _n = (", " + ", ".join(_notes)) if _notes else ""
    test_desc = (
        f"reuse test set of {ref_id}" if ref_id else
        f"{pct}% test{_n}" if pct not in (None, 0) else
        f"{cnt} test rows{_n}" if cnt not in (None, 0) else "no test set"
    )
    log(f"[{name}] import project {project_id} (status={status}, cutoff={cutoff}) + transform to S3 [{test_desc}]")

    if dry_run:
        log(f"[{name}] DRY-RUN: would create kind=label dataset + transform")
        return f"<dry-run:{name}>"

    # 1. create the kind=label dataset (verifies token + project + counts rows)
    label_id = entry.get("label_id")
    if not label_id or entry.get("signature") != sig:
        created = gw.post("/v1/datasets", {
            "name": f"{name}-label",
            "kind": "label",
            "label_base_url": cfg["label_base_url"],
            "label_project_id": project_id,
            "label_token_secret": cfg["label_token_secret"],
            "label_status": status,
            "label_updated_until": cutoff,
        })
        label_id = created["id"]
        log(f"[{name}] created label dataset {label_id} (num_rows={created.get('num_rows')})")

    # 2. transform to S3 with the requested test split
    body: dict[str, Any] = {"target": "s3", "storage_id": storage_id}
    if ref_id:
        body["test_split_ref_dataset_id"] = ref_id
    elif pct not in (None, 0):
        body["test_split_pct"] = float(pct)
    elif cnt not in (None, 0):
        body["test_split_count"] = int(cnt)
    if eligibility and min_chars not in (None, 0):
        body["test_min_chars"] = int(min_chars)
    if eligibility and exclude_regex:
        body["test_exclude_regex"] = exclude_regex
    gw.post(f"/v1/datasets/{label_id}/transform", body)
    log(f"[{name}] transform started; waiting…")
    rec = wait_dataset(gw, label_id, timeout=poll_timeout, label=name)
    if rec.get("transform_status") != "done":
        die(f"[{name}] transform ended as {rec.get('transform_status')!r}: {(rec.get('transform_log') or '')[-400:]}")
    transformed_id = rec.get("audio_dataset_id")
    if not transformed_id:
        die(f"[{name}] transform done but no audio_dataset_id on the source record")
    log(f"[{name}] transformed -> {transformed_id}")

    state.data["datasets"][name] = {
        "label_id": label_id, "transformed_id": transformed_id, "signature": sig,
    }
    state.save()
    return transformed_id


def ensure_merge(
    gw: Gateway, cfg: dict, transformed_ids: list[str], state: State, storage_id: str,
    *, dry_run: bool, poll_timeout: float,
) -> str:
    """Merge the transformed datasets into one. Returns the training dataset id."""
    merge_cfg = cfg.get("merge") or {}
    if not merge_cfg.get("enabled", True) or len(transformed_ids) < 2:
        log(f"merge disabled or <2 datasets — training on {transformed_ids[0]}")
        return transformed_ids[0]

    sig = {"sources": list(transformed_ids), "name": merge_cfg.get("name")}
    entry = state.data.get("merge") or {}
    if entry.get("merged_id") and entry.get("signature") == sig:
        log(f"merge up-to-date (merged={entry['merged_id']}) — skipping")
        return entry["merged_id"]

    name = merge_cfg.get("name") or "merged-dataset"
    log(f"merging {len(transformed_ids)} datasets -> {name}")
    if dry_run:
        log("DRY-RUN: would merge " + ", ".join(transformed_ids))
        return "<dry-run:merged>"

    created = gw.post("/v1/datasets/merge", {
        "source_ids": transformed_ids,
        "target": "s3",
        "storage_id": storage_id,
        "name": name,
    })
    merged_id = created["id"]
    log(f"merge dataset {merged_id} created; waiting…")
    rec = wait_dataset(gw, merged_id, timeout=poll_timeout, label="merge")
    if rec.get("transform_status") != "done":
        die(f"merge ended as {rec.get('transform_status')!r}: {(rec.get('transform_log') or '')[-400:]}")
    log(f"merged -> {merged_id} ({rec.get('num_rows')} rows)")

    state.data["merge"] = {"merged_id": merged_id, "signature": sig}
    state.save()
    return merged_id


def ensure_runs(
    gw: Gateway, cfg: dict, dataset_id: str, state: State, storage_id: str,
    *, provider_id: Optional[str], dry_run: bool,
) -> list[str]:
    train = cfg.get("train") or {}
    models = cfg.get("models") or []
    if not models:
        die("no `models` in the config")
    aug = resolve_augment(train.get("augment_techniques"))
    run_ids: list[str] = []

    for m in models:
        mc = dict(m) if isinstance(m, dict) else {"base_model": m}
        base_model = mc.get("base_model") or die("each model needs a `base_model`")
        # per-model overrides win over the shared train block
        merged = {**train, **mc}
        key = f"{dataset_id}::{base_model}"
        existing = state.data["runs"].get(key)
        if existing and existing.get("run_id"):
            log(f"run for {base_model} on {dataset_id} exists ({existing['run_id']}) — skipping")
            run_ids.append(existing["run_id"])
            continue

        run_name = mc.get("name") or f"{base_model.split('/')[-1]}-{dataset_id}"
        body: dict[str, Any] = {
            "name": run_name,
            "dataset_id": dataset_id,
            "base_model": base_model,
            "task_type": merged.get("task_type", "asr"),
            "max_epochs": int(merged.get("max_epochs", 5)),
            # max_steps > 0 caps optimizer steps (overrides max_epochs) — a quick
            # smoke run. 0 = run the full epochs.
            "max_steps": int(merged.get("max_steps", 0)),
            "eval_strategy": merged.get("eval_strategy", "epoch"),
            "save_strategy": merged.get("save_strategy", merged.get("eval_strategy", "epoch")),
            "eval_steps": int(merged.get("eval_steps", 500)),
            "save_steps": int(merged.get("save_steps", merged.get("eval_steps", 500))),
            "no_eval": bool(merged.get("no_eval", False)),
            "patience": int(merged.get("patience", 0)),
            "batch_size": int(merged.get("batch_size", 8)),
            "grad_accum": int(merged.get("grad_accum", 1)),
            "warmup_steps": int(merged.get("warmup_steps", 0)),
            "lr_scheduler_type": merged.get("lr_scheduler_type", "linear"),
            "augment_techniques": resolve_augment(mc["augment_techniques"]) if "augment_techniques" in mc else aug,
            "augment_prob": float(merged.get("augment_prob", 0.5)),
            "gpu_type": merged.get("gpu_type", "NVIDIA H100 80GB HBM3"),
            "gpu_count": int(merged.get("gpu_count", 1)),
            "secure_cloud": bool(merged.get("secure_cloud", True)),
            "storage_id": storage_id,
            "provider_id": provider_id,
        }
        if merged.get("learning_rate") is not None:
            body["learning_rate"] = float(merged["learning_rate"])
        if merged.get("data_center_id"):
            body["data_center_id"] = merged["data_center_id"]

        _dur = f"max_steps={body['max_steps']}" if body["max_steps"] > 0 else f"epochs={body['max_epochs']}"
        log(f"launching training: {base_model} on {dataset_id} "
            f"({_dur}, patience={body['patience']}, "
            f"batch={body['batch_size']}, warmup={body['warmup_steps']}, "
            f"aug={len(body['augment_techniques'])}, {body['gpu_type']} secure={body['secure_cloud']})")
        if dry_run:
            log(f"DRY-RUN: would POST /v1/training-runs {json.dumps(body)}")
            continue

        run = gw.post("/v1/training-runs", body)
        rid = run["id"]
        log(f"  -> run {rid} status={run.get('status')}")
        state.data["runs"][key] = {"run_id": rid, "base_model": base_model}
        state.save()
        run_ids.append(rid)

    return run_ids


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Config-driven Autotrain pipeline (import -> transform -> merge -> train).")
    here = Path(__file__).resolve().parent
    ap.add_argument("--config", default=str(here / "config.yaml"), help="path to the YAML config")
    ap.add_argument("--cutoff", default=None, help='global import cutoff override, e.g. "2026-07-01 11:59PM"')
    ap.add_argument("--gateway-url", default=None, help="override gateway_url from the config")
    ap.add_argument("--api-key", default=None, help="override api_key (or set AUTOTRAIN_API_KEY)")
    ap.add_argument("--state", default=None, help="state file path (default: automation/state/<config-stem>.json)")
    ap.add_argument("--until", choices=["transform", "merge", "train"], default="train",
                    help="stop after this stage: 'transform' (per-dataset only), 'merge' (through the merge, no training), or 'train' (full, default)")
    ap.add_argument("--fresh", action="store_true", help="ignore + overwrite existing state (recreate everything)")
    ap.add_argument("--dry-run", action="store_true", help="print what would happen, make no changes")
    ap.add_argument("--no-run-watch", action="store_true", help="launch training runs but don't wait for them")
    ap.add_argument("--watch-timeout", type=float, default=6 * 3600, help="max seconds to watch training runs")
    ap.add_argument("--transform-timeout", type=float, default=3 * 3600, help="max seconds to wait per transform/merge")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    gateway_url = args.gateway_url or cfg.get("gateway_url") or "http://localhost:8080"
    api_key = args.api_key or os.environ.get("AUTOTRAIN_API_KEY") or cfg.get("api_key")
    if not api_key:
        die("no api_key (config.api_key, --api-key, or AUTOTRAIN_API_KEY)")
    for req in ("label_base_url", "label_token_secret"):
        if not cfg.get(req):
            die(f"config is missing `{req}`")

    state_path = Path(args.state) if args.state else (here / "state" / f"{Path(args.config).stem}.json")
    if args.fresh and state_path.exists() and not args.dry_run:
        state_path.unlink()
        log(f"--fresh: removed {state_path}")
    state = State(state_path)

    log(f"gateway   = {gateway_url}")
    log(f"config    = {args.config}")
    log(f"state     = {state_path}")
    if args.cutoff:
        log(f"cutoff    = {args.cutoff} (CLI override)")
    if args.dry_run:
        log("DRY-RUN — no changes will be made")

    gw = Gateway(gateway_url, api_key)

    # 0. resolve storage + provider up front (fail fast on a bad config)
    storage_id = resolve_storage(gw, cfg) if not args.dry_run else (cfg.get("storage_id") or "<auto>")
    provider_id = resolve_provider(gw, (cfg.get("train") or {}).get("provider_id")) if not args.dry_run \
        else ((cfg.get("train") or {}).get("provider_id") or "<auto>")
    log(f"storage   = {storage_id}")
    log(f"provider  = {provider_id}")

    datasets = cfg.get("datasets") or []
    if not datasets:
        die("config has no `datasets`")

    # Preflight: make sure the running gateway supports every transform field the
    # config uses. Older gateways silently drop unknown request fields (pydantic
    # extra=ignore), which would quietly put junk transcripts back in the test set —
    # fail loudly instead of misbehaving silently.
    def _needed_fields(d: dict) -> set:
        out: set = set()
        if (d.get("test_split_ref_dataset_id") or "").strip():
            out.add("test_split_ref_dataset_id")
        has_test = d.get("test_split_pct") not in (None, 0) or d.get("test_split_count") not in (None, 0)
        if not has_test:
            return out
        mc = d.get("test_min_chars")
        if (mc if mc is not None else cfg.get("test_min_chars")):
            out.add("test_min_chars")
        rx = d.get("test_exclude_regex")
        rx = rx if rx is not None else cfg.get("test_exclude_regex")
        if rx and str(rx).strip():
            out.add("test_exclude_regex")
        return out
    needed: set = set().union(*(_needed_fields(d) for d in datasets)) if datasets else set()
    if needed and not args.dry_run:
        try:
            props = (gw.get("/openapi.json").get("components", {}).get("schemas", {})
                     .get("TransformRequest", {}).get("properties", {}))
        except GatewayError as e:
            die(f"could not read the gateway OpenAPI to verify transform support: {e}")
        missing = sorted(f for f in needed if f not in props)
        if missing:
            die(f"the running gateway doesn't support {', '.join(missing)} yet — restart it "
                "(.venv/bin/gateway) to pick up the new code, then re-run")

    # 1 + 2. import + transform each dataset
    log("=" * 70)
    log(f"STEP 1/3 — importing + transforming {len(datasets)} dataset(s)")
    transformed_ids: list[str] = []
    for ds in datasets:
        tid = ensure_dataset(
            gw, cfg, ds, state, storage_id,
            cli_cutoff=args.cutoff, dry_run=args.dry_run, poll_timeout=args.transform_timeout,
        )
        transformed_ids.append(tid)

    if args.until == "transform":
        log("=" * 70)
        log(f"stopping after transform (--until transform) — transformed: {', '.join(transformed_ids)}")
        return

    # 3. merge
    log("=" * 70)
    log("STEP 2/3 — merging")
    training_dataset_id = ensure_merge(
        gw, cfg, transformed_ids, state, storage_id,
        dry_run=args.dry_run, poll_timeout=args.transform_timeout,
    )

    if args.until == "merge":
        log("=" * 70)
        log(f"stopping after merge (--until merge) — merged dataset: {training_dataset_id}")
        return

    # 4. train
    log("=" * 70)
    log("STEP 3/3 — launching training")
    run_ids = ensure_runs(
        gw, cfg, training_dataset_id, state, storage_id,
        provider_id=provider_id, dry_run=args.dry_run,
    )

    log("=" * 70)
    if args.dry_run:
        log("DRY-RUN complete.")
        return
    log(f"merged dataset : {training_dataset_id}")
    log(f"training runs  : {', '.join(run_ids) if run_ids else '(none)'}")
    if run_ids and not args.no_run_watch:
        log("watching training runs (Ctrl-C to stop watching — runs continue server-side)…")
        watch_runs(gw, run_ids, timeout=args.watch_timeout)
    log("done.")


if __name__ == "__main__":
    main()
