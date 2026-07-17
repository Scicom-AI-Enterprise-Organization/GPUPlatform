#!/usr/bin/env python3
"""Hyperparameter-sweep orchestrator (AutoTrain) — runs N trials of a worker
trainer (whisper_finetune.py / tts_finetune.py) in a GPU-pinned pool on ONE box,
ranks by the sweep metric, and reports the best. Shipped to the pod/VM alongside
the worker by the gateway when a run is in sweep mode.

Config (JSON via --config) = the base trainer config PLUS:
  sweep:          {param: [values], ...}     cross-product of these = the trials
  gpus_per_trial: int                         GPUs each trial pins
  sweep_gpus:     ["6","7", …] | []           the GPU ids to schedule across
  sweep_metric:   "wer" | "cer" | "loss"      ranked ascending (lower = better)
  task_type:      "asr" | "tts"               selects the worker

Each trial gets its own config (base + the swept overrides), its own work_dir,
its own artifacts prefix (…/trials/<i>/), and CUDA_VISIBLE_DEVICES pinned to its
GPU slice. Concurrency = floor(#gpus / gpus_per_trial).

Emits (parsed by the gateway):
  @@TRIAL {trial, params, metric, status}     one per finished trial
  @@DONE  {best:{trial,params,metric}, trials:[…]}
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
import threading

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKERS = {"asr": "whisper_finetune.py", "tts": "tts_finetune.py"}
# Tasks whose worker is an ORCHESTRATOR — it launches its own torch.distributed.run
# (TTS: tts_finetune spawns torchrun for qwen3). Such a worker must be run DIRECTLY
# (never under an outer torchrun, which would nest launchers); it reads the GPU
# slice from CUDA_VISIBLE_DEVICES and picks its own nproc. ASR's worker IS the
# trainer, so it's run as N ranks under torchrun for a multi-GPU trial.
ORCHESTRATOR_TASKS = {"tts"}


def log(m: str) -> None:
    print(m, flush=True)


def emit(tag: str, obj: dict) -> None:
    print(f"@@{tag} {json.dumps(obj)}", flush=True)


def expand(sweep: dict) -> list[dict]:
    """{param: [v1, v2]} → [{param: v1}, {param: v2}, …] (cross-product)."""
    keys = [k for k, v in (sweep or {}).items() if isinstance(v, list) and v]
    if not keys:
        return [{}]
    grids = [sweep[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*grids)]


def run(cfg: dict) -> None:
    task = cfg.get("task_type", "asr")
    worker = os.path.join(THIS_DIR, WORKERS.get(task, WORKERS["asr"]))
    metric = (cfg.get("sweep_metric") or "wer").lower()
    per = max(1, int(cfg.get("gpus_per_trial", 1)))
    gpus = [str(g) for g in (cfg.get("sweep_gpus") or [])]

    if gpus:
        slices = [gpus[i:i + per] for i in range(0, len(gpus), per)]
        slices = [s for s in slices if len(s) == per] or [gpus[:per]]
    else:
        slices = [None]  # no pin → one slot, all visible GPUs
    concurrency = max(1, len(slices))

    combos = expand(cfg.get("sweep") or {})
    log(f"[sweep] {len(combos)} trial(s) · {concurrency} GPU slot(s) · "
        f"{per} GPU/trial · rank by {metric} (asc)")

    # Install deps ONCE up front (avoids N concurrent pip races on first run).
    log("[sweep] preparing dependencies …")
    subprocess.call([sys.executable, worker, "--deps-only", "--config", cfg["_config_path"]])

    base_prefix = (cfg.get("artifacts") or {}).get("prefix", "").rstrip("/")
    # Namespace the sweep workdir by RUN — concurrent sweeps on one box used to
    # share {work_dir}/trial{i}.json + trial{i}/ and silently swap each other's
    # trial configs (observed: a turbo sweep training the sibling run's
    # large-v2 config). basename(artifacts.prefix) = the run id.
    run_tag = os.path.basename(base_prefix) or f"pid{os.getpid()}"
    work = os.path.abspath(os.path.join(cfg.get("work_dir") or "/workspace",
                                        f"autotrain-sweep-{run_tag}"))
    os.makedirs(work, exist_ok=True)

    results: list[dict | None] = [None] * len(combos)
    sem = threading.Semaphore(concurrency)
    slot_lock = threading.Lock()
    free_slots = list(range(len(slices)))
    io_lock = threading.Lock()

    def emit_line(s: str) -> None:
        with io_lock:
            sys.stdout.write(s)
            sys.stdout.flush()

    def run_trial(i: int, params: dict) -> None:
        with sem:
            with slot_lock:
                slot = free_slots.pop()
            gpu_slice = slices[slot]
            try:
                tcfg = copy.deepcopy(cfg)
                for k in ("sweep", "sweep_gpus", "sweep_metric", "gpus_per_trial", "_config_path"):
                    tcfg.pop(k, None)
                tcfg.update(params)  # override the swept hyperparameters
                # The `augment` sweep dimension is a scalar flag (on/off) for a
                # clean leaderboard label; translate it into the trainer's real
                # knob: "on" reuses the base augment_techniques, "off" clears
                # them. (params keeps `augment` so @@TRIAL still reports it.)
                if "augment" in tcfg:
                    on = str(tcfg.pop("augment")).lower() in ("on", "true", "1", "yes")
                    tcfg["augment_techniques"] = (
                        list(cfg.get("augment_techniques") or []) if on else []
                    )
                # freeze_encoder may arrive as an on/off sweep value — coerce to a
                # real bool, else the string "off" is truthy and freezes anyway.
                if "freeze_encoder" in tcfg:
                    tcfg["freeze_encoder"] = str(tcfg["freeze_encoder"]).lower() in ("on", "true", "1", "yes")
                # Sweeping a LoRA knob (r/alpha) implies LoRA is on for those trials.
                if ("lora_r" in tcfg or "lora_alpha" in tcfg) and not tcfg.get("use_lora"):
                    tcfg["use_lora"] = True
                tcfg["work_dir"] = os.path.join(work, f"trial{i}")
                tcfg["run_name"] = f"{cfg.get('run_name', 'run')}-t{i}"
                if base_prefix and tcfg.get("artifacts"):
                    tcfg["artifacts"] = {**cfg["artifacts"], "prefix": f"{base_prefix}/trials/{i}"}
                tpath = os.path.join(work, f"trial{i}.json")
                with open(tpath, "w") as f:
                    json.dump(tcfg, f)

                env = dict(os.environ)
                if gpu_slice:
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_slice)
                pin = env.get("CUDA_VISIBLE_DEVICES", "(all)")
                emit_line(f"[sweep] trial {i} START params={json.dumps(params)} gpus={pin}\n")

                # Multi-GPU trial → DDP via torch's launcher (one process per GPU
                # in the slice); the worker rank-guards its @@ output. Single-GPU
                # trials run plain `python`. Per-trial master_port avoids clashes
                # between concurrent trials.
                slice_n = len(gpu_slice) if gpu_slice else 1
                if slice_n > 1 and tcfg.get("use_ddp", True) and task not in ORCHESTRATOR_TASKS:
                    # ASR multi-GPU trial: run the trainer as N ranks under torchrun
                    # (per-trial master_port avoids clashes between concurrent trials).
                    port = 29500 + (i % 4000)
                    cmd = [sys.executable, "-m", "torch.distributed.run",
                           f"--nproc_per_node={slice_n}", f"--master_port={port}",
                           worker, "--config", tpath]
                else:
                    # Single-GPU trial, OR an orchestrator worker (TTS) that launches
                    # its own torchrun from the pinned CUDA_VISIBLE_DEVICES slice.
                    cmd = [sys.executable, "-u", worker, "--config", tpath]
                proc = subprocess.Popen(
                    cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                best = None
                artifact = None
                for line in proc.stdout:  # type: ignore[union-attr]
                    emit_line(f"[trial {i}] {line}")
                    # Tags can be preceded by tqdm \r fragments on the same
                    # \n-line — find them anywhere, not just at the start.
                    di = line.find("@@DONE ")
                    if di >= 0:
                        try:
                            best = json.loads(line[di + len("@@DONE "):]).get("best")
                        except Exception:
                            pass
                    ai = line.find("@@ARTIFACT ")
                    if ai >= 0:
                        try:
                            artifact = json.loads(line[ai + len("@@ARTIFACT "):])
                        except Exception:
                            pass
                proc.wait()
                m = best.get(metric) if isinstance(best, dict) else None
                results[i] = {
                    "trial": i, "params": params, "metric": m,
                    "status": "done" if proc.returncode == 0 else "failed",
                    "artifact": artifact,
                }
                emit("TRIAL", results[i])
            except Exception as e:  # noqa: BLE001
                results[i] = {"trial": i, "params": params, "metric": None,
                              "status": "failed", "error": str(e)}
                emit("TRIAL", results[i])
            finally:
                with slot_lock:
                    free_slots.append(slot)

    threads = [threading.Thread(target=run_trial, args=(i, c)) for i, c in enumerate(combos)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    done = [r for r in results if r]
    ranked = sorted((r for r in done if r["metric"] is not None), key=lambda r: r["metric"])
    best = ranked[0] if ranked else None
    if best:
        log(f"[sweep] best: trial {best['trial']} · {metric}={best['metric']} · {best['params']}")
        # Re-emit the WINNING trial's artifact as the sweep-level one, LAST —
        # trials' own @@ARTIFACT lines can lose their "[trial N]" prefix to the
        # log stream's \r-splitting and clobber result_json.artifact with
        # whichever trial finished last; this final emit always wins.
        if best.get("artifact"):
            emit("ARTIFACT", best["artifact"])
    emit("DONE", {"best": best, "trials": done})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    cfg["_config_path"] = a.config
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
