"""Parse + validate the multi-model fleet spec the gateway hands the worker.

Shape of MULTI_MODEL_CONFIG (JSON):
    {
      "total_gpus": 4,
      "sleep_level": 1,
      "models": [
        {"model": "...", "served_name": "...", "tp": 2, "port": 8001,
         "gpu_indices": [0,1], "extra_args": "...", "sleep_level": 1},
        ...
      ]
    }

Contention is modelled at the GPU level: a model may be awake only when every
GPU in its `gpu_indices` is free. Two models sharing any GPU are mutually
exclusive and time-share VRAM via sleep/wake. This handles uniform TP, mixed
TP, and a wide model overlapping several narrow ones — no explicit "slots".
"""
from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemberModel:
    model: str                      # HF id / path → vLLM --model
    served_name: str                # what clients send as payload["model"]
    tp: int                         # tensor-parallel size
    port: int                       # localhost port for this model's vLLM
    gpu_indices: tuple[int, ...]    # CUDA_VISIBLE_DEVICES (len == tp * pp)
    extra_args: list[str] = field(default_factory=list)
    sleep_level: int = 1
    pp: int = 1                     # pipeline-parallel size; GPUs needed = tp * pp
    task: str | None = None         # "transcription" → audio/ASR model (drives audio-dep install)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass(frozen=True)
class MultiModelConfig:
    total_gpus: int
    default_sleep_level: int
    members: tuple[MemberModel, ...]
    # uv venv to run `vllm serve` from (its bin/python). None → bare `python3`.
    venv_path: str | None = None
    # vLLM version the worker ensures is installed in venv_path. None → as-is.
    vllm_version: str | None = None
    # Full `uv pip install` arg string for vLLM, used verbatim instead of the
    # version (e.g. a nightly with extra index URLs). None → use vllm_version.
    vllm_install_args: str | None = None
    # Optional shell snippet run once per worker boot, after the venv is ready and
    # before any model launches (e.g. building DeepGEMM). None → skip.
    pre_script: str | None = None


def parse_multi_config(raw_json: str | None, path: str | None = None) -> MultiModelConfig:
    if (not raw_json) and path:
        with open(path) as fh:
            raw_json = fh.read()
    if not raw_json:
        raise ValueError("MULTI_MODEL_CONFIG (or _PATH) is required for WORKER_MODE=multi")
    cfg = json.loads(raw_json)

    total = int(cfg.get("total_gpus") or 0)
    default_level = int(cfg.get("sleep_level") or 1)
    raw_members = cfg.get("models") or []
    if not raw_members:
        raise ValueError("multi config has no models")

    members: list[MemberModel] = []
    seen_names: set[str] = set()
    seen_ports: set[int] = set()
    for i, m in enumerate(raw_members):
        model = (m.get("model") or "").strip()
        if not model:
            raise ValueError(f"member {i} has no model")
        served = (m.get("served_name") or model).strip()
        tp = int(m.get("tp") or 1)
        pp = max(1, int(m.get("pp") or 1))
        width = tp * pp  # GPUs this member occupies (tensor × pipeline parallel)
        port = int(m.get("port") or (8001 + i))
        explicit_idxs = bool(m.get("gpu_indices"))
        idxs = tuple(int(x) for x in (m.get("gpu_indices") or []))
        if not idxs:
            # Auto-assign: pack `width` consecutive GPUs round-robin (deterministic).
            start = (i * width) % max(1, total) if total else 0
            idxs = tuple((start + j) % max(1, total) for j in range(width)) if total else tuple(range(width))
        if len(idxs) != width:
            raise ValueError(f"{model}: gpu_indices {idxs} length != tp*pp {width} (tp={tp}, pp={pp})")
        if any(g < 0 for g in idxs):
            raise ValueError(f"{model}: negative gpu index in {idxs}")
        # Only range-check AUTO-assigned indices against the device count. Explicit
        # gpu_indices are physical CUDA ids — an endpoint pinned to a high subset
        # (e.g. visible_devices "6,7") legitimately has ids >= the count, so the
        # count is not a valid upper bound for them.
        if not explicit_idxs and total and any(g >= total for g in idxs):
            raise ValueError(f"{model}: gpu_indices {idxs} out of range for total_gpus={total}")
        if served in seen_names:
            raise ValueError(f"duplicate served_name {served}")
        if port in seen_ports:
            raise ValueError(f"duplicate port {port}")
        seen_names.add(served)
        seen_ports.add(port)
        extra = m.get("extra_args") or ""
        extra_list = shlex.split(extra) if isinstance(extra, str) else list(extra)
        members.append(MemberModel(
            model=model,
            served_name=served,
            tp=tp,
            pp=pp,
            port=port,
            gpu_indices=idxs,
            extra_args=extra_list,
            sleep_level=int(m.get("sleep_level") or default_level),
            task=((m.get("task") or "").strip().lower() or None),
        ))
    return MultiModelConfig(
        total_gpus=total,
        default_sleep_level=default_level,
        members=tuple(members),
        venv_path=((cfg.get("venv_path") or "").strip() or None),
        vllm_version=((cfg.get("vllm_version") or "").strip() or None),
        vllm_install_args=((cfg.get("vllm_install_args") or "").strip() or None),
        pre_script=((cfg.get("pre_script") or "").strip() or None),
    )
