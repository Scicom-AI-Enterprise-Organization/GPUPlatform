"""Compute — provision raw RunPod pods for ad-hoc SSH access.

User picks GPU + disk size + image; gateway POSTs RunPod's REST API and
polls the pod for SSH coords (publicIp + the public port mapped to 22/tcp).
The user connects with the same private key the bench feature uses — its
public half is registered on the RunPod account, so every pod RunPod
provisions auto-gets it injected into authorized_keys.

Pod lifetime is **explicit** — there's no idle TTL. Users terminate from
the UI when done. We do mark `creating` rows as failed on gateway restart
to surface orphans, mirroring bench.cleanup_orphaned_running.
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from . import audit
from . import vm_probe
from .auth import current_user, require_admin, require_section
from .db import Base, User, get_session, session_factory
from .provider import CloudProviderDisabled, cloud_providers_disabled
from .runpod_provider import _map_gpu

logger = logging.getLogger("gateway.compute")

# Default image — RunPod's pytorch image has sshd preconfigured and respects
# the account-level SSH key for authorized_keys injection.
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 600  # 10min — RunPod cold pulls of pytorch images can be slow
PI_POLL_TIMEOUT_S = 1500  # 25min — PI's Lambda Labs / hyperstack sub-providers
                          # can sit in PROVISIONING for >15 min on first launch

# Curated favourites. Every image here has both sshd and JupyterLab
# pre-baked — `JUPYTER_PASSWORD` env triggers `start.sh` to launch jupyter
# on port 8888, which we proxy via RunPod's per-pod proxy domain. Kept to 2
# entries; the UI surfaces the full RunPod catalogue via a search box that
# hits /compute/runpod/templates.
CURATED_TEMPLATES: list[dict[str, str]] = [
    {
        "id": "pytorch-2.8-cuda12.8",
        "name": "PyTorch 2.8 + CUDA 12.8",
        "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "description": "PyTorch 2.8.0, CUDA 12.8.1, Ubuntu 24.04. Newest stack — default for most workloads.",
    },
    {
        "id": "pytorch-2.4-cuda12.4",
        "name": "PyTorch 2.4 + CUDA 12.4",
        "image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "description": "PyTorch 2.4, Python 3.11, CUDA 12.4. Pick for older-CUDA workloads.",
    },
]

# Prime Intellect — pod-create only accepts values from this enum. We hardcode
# the list (it's small and stable) rather than calling out to PI for it. First
# entry is the default; we lead with PI's newest CUDA/PyTorch combo since most
# sub-providers carry it.
PI_IMAGES: list[dict[str, str]] = [
    {
        "id": "cuda_12_6_pytorch_2_7",
        "name": "PyTorch 2.7 + CUDA 12.6",
        "description": "Newest CUDA/PyTorch combo PI offers. Broadest sub-provider support.",
    },
    {
        "id": "cuda_12_4_pytorch_2_6",
        "name": "PyTorch 2.6 + CUDA 12.4",
        "description": "Slightly older — pick if you need PyTorch ≤ 2.6.",
    },
    {
        "id": "cuda_12_4_pytorch_2_5",
        "name": "PyTorch 2.5 + CUDA 12.4",
        "description": "Same CUDA as our RunPod default. Good fallback when 12.6 is short of stock.",
    },
    {
        "id": "ubuntu_22_cuda_12",
        "name": "Ubuntu 22.04 + CUDA 12",
        "description": "Minimal Ubuntu + CUDA 12. Bring your own framework.",
    },
    {
        "id": "vllm_llama_70b",
        "name": "vLLM + Llama-3 70B",
        "description": "vLLM pre-loaded with a Llama-3 70B endpoint.",
    },
    {
        "id": "stable_diffusion",
        "name": "Stable Diffusion",
        "description": "Stable Diffusion pre-configured.",
    },
]


def _resolve_template(template_id: str) -> Optional[dict[str, str]]:
    for t in CURATED_TEMPLATES:
        if t["id"] == template_id:
            return t
    return None


def _resolve_pi_image(value: str) -> Optional[dict[str, str]]:
    for t in PI_IMAGES:
        if t["id"] == value:
            return t
    return None


# Per-api-key cache of RunPod's public-templates blob. Keyed by sha256 of the
# key so we don't hold raw bearer tokens in a process-wide map. ~33k lines of
# JSON per fetch, refreshed every 5 min.
_RUNPOD_TEMPLATES_TTL_S = 300.0
_runpod_templates_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}


def _key_hash(api_key: str) -> str:
    import hashlib
    return hashlib.sha256(api_key.encode()).hexdigest()


async def _fetch_runpod_templates(api_key: str) -> list[dict[str, Any]]:
    """Pull RunPod's templates (user + public + runpod). Cached 5 min per key.

    The upstream response has no pagination today and returns the whole
    catalogue in one shot (~33k lines). We slim each row to the fields the
    UI cares about so the cache + API responses stay light.
    """
    h = _key_hash(api_key)
    now = time.time()
    cached = _runpod_templates_cache.get(h)
    if cached is not None and cached[1] > now:
        return cached[0]

    base = _api_base()
    async with httpx.AsyncClient(timeout=60.0) as cli:
        r = await cli.get(
            f"{base}/templates",
            headers={"Authorization": f"Bearer {api_key}"},
            params={
                "includePublicTemplates": "true",
                "includeRunpodTemplates": "true",
            },
        )
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"RunPod templates fetch failed: HTTP {r.status_code}: {r.text[:200]}",
        )
    raw = r.json() or []
    if isinstance(raw, dict):
        # Tolerate {data: [...]} shape if RunPod ever switches.
        raw = raw.get("data") or raw.get("templates") or []

    slim: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        # Pod templates only — drop serverless endpoint templates which can't
        # be launched as raw pods.
        if row.get("isServerless"):
            continue
        slim.append({
            "id": row.get("id") or "",
            "name": row.get("name") or "",
            "imageName": row.get("imageName") or row.get("image") or "",
            "category": row.get("category") or "",
            "isPublic": bool(row.get("isPublic")),
            "isRunpod": bool(row.get("isRunpod") or row.get("isRunpodTemplate")),
        })
    _runpod_templates_cache[h] = (slim, now + _RUNPOD_TEMPLATES_TTL_S)
    return slim


def _jupyter_url(runpod_pod_id: str, token: Optional[str] = None) -> str:
    # RunPod proxies any HTTP port via `https://<podId>-<port>.proxy.runpod.net/`.
    # The runpod/pytorch images launch jupyter with `--ServerApp.token=$JUPYTER_PASSWORD`,
    # and the working URL pattern (confirmed against a live pod) is the root
    # path with `?token=…` — Jupyter consumes the token there and redirects
    # to /lab itself. `/lab?token=…` 404s through RunPod's proxy because
    # Jupyter would issue a /login redirect that the proxy doesn't handle.
    base = f"https://{runpod_pod_id}-8888.proxy.runpod.net/"
    if token:
        from urllib.parse import quote
        return f"{base}?token={quote(token, safe='')}"
    return base


# ---------- DB model ---------------------------------------------------


class ComputePod(Base):
    __tablename__ = "compute_pods"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    gpu_type: Mapped[str] = mapped_column(String(64))
    gpu_count: Mapped[int] = mapped_column(Integer, default=1)
    container_disk_gb: Mapped[int] = mapped_column(Integer, default=40)
    volume_gb: Mapped[int] = mapped_column(Integer, default=0)
    image: Mapped[str] = mapped_column(String(255))
    template_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    cloud_type: Mapped[str] = mapped_column(String(16), default="COMMUNITY")
    # creating | running | failed | terminated | auto_terminated | pending_approval | rejected
    # (auto_terminated = killed by the idle sweep; terminated = manual delete)
    status: Mapped[str] = mapped_column(String(20), default="creating", index=True)
    runpod_pod_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    public_ip: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    ssh_port: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ssh_user: Mapped[str] = mapped_column(String(32), default="root")
    # Per-pod random JupyterLab password — start.sh in runpod/* images uses
    # this to launch jupyter on 8888. Owner-visible only; not a system secret.
    jupyter_password: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    cost_per_hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_text: Mapped[Optional[str]] = mapped_column(String(4096), nullable=True)
    # Set when an admin rejects an approval request. Distinct from error_text
    # (which is for provisioning failures) so the UI can render the right copy.
    reject_reason: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    ready_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    terminated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Auto-terminate after this many seconds with no GPU compute AND no GPU
    # memory in use. 0 = disabled (the pod runs until manually terminated).
    idle_terminate_after_s: Mapped[int] = mapped_column(Integer, default=0)
    # Last time the idle monitor observed the pod busy (or couldn't measure it —
    # we bump it to "now" then, so an unmeasurable pod is never auto-killed).
    # Seeded to the ready time so the idle clock starts when SSH lands.
    last_active_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Per-pod cloud-account selection. NULL = platform default (env var).
    provider_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    # Persisted Jupyter URL. For RunPod we derive it from runpod_pod_id at
    # render time so we leave it NULL; for PI we resolve the external 8888
    # port at poll time and write the full URL here so the renderer stays
    # kind-agnostic.
    jupyter_url_override: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)


# ---------- RunPod REST helpers ----------------------------------------


def _api_base() -> str:
    return os.environ.get("RUNPOD_API_BASE", "https://rest.runpod.io/v1").rstrip("/")


async def _resolve_api_key(
    provider_id: Optional[str],
    expected_kind: str = "runpod",
) -> str:
    """Look up the right API key for a compute pod row.

    Falls back to the kind-appropriate gateway env var (`RUNPOD_API_KEY` /
    `PI_API_KEY`) when no provider_id is set so the legacy single-tenant
    path keeps working without a provider registered."""
    # Central kill-switch — also closes the env-key fallback below, which would
    # otherwise use a stray RUNPOD_API_KEY/PI_API_KEY even with cloud disabled.
    if cloud_providers_disabled():
        raise CloudProviderDisabled(expected_kind)
    if provider_id:
        from .provider_resolve import resolve_cloud_creds
        from .db import session_factory
        async with session_factory()() as s:
            creds = await resolve_cloud_creds(s, provider_id, expected_kind)
        return creds.api_key
    env_name = "RUNPOD_API_KEY" if expected_kind == "runpod" else "PI_API_KEY"
    k = os.environ.get(env_name, "").strip()
    if not k:
        raise RuntimeError(f"{env_name} not set — Compute requires a {expected_kind} provider or env key")
    return k


def _api_key() -> str:
    """Synchronous env-only lookup. Retained for the few startup-time call
    sites that don't yet have a provider_id context."""
    if cloud_providers_disabled():
        raise CloudProviderDisabled("runpod")
    k = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not k:
        raise RuntimeError("RUNPOD_API_KEY not set — Compute requires a real RunPod account")
    return k


def _name_prefix() -> str:
    return os.environ.get("COMPUTE_NAME_PREFIX", "sgpu-compute")


def _client(api_key: Optional[str] = None) -> httpx.AsyncClient:
    """Build an httpx client for the RunPod REST API.

    Pass `api_key` to use a provider-specific key; otherwise falls back to
    the gateway env var. Async callers that know the compute pod's
    provider_id should resolve via _resolve_api_key() first and pass it in.
    """
    return httpx.AsyncClient(
        base_url=_api_base(),
        headers={
            "Authorization": f"Bearer {api_key or _api_key()}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )


def _ssh_key_path() -> str:
    p = os.environ.get("BENCHMARK_SSH_KEY_PATH", "").strip()
    if not p:
        p = str(Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go")
    return os.path.expanduser(p)


def _extract_ssh(pod: dict) -> tuple[Optional[str], Optional[int]]:
    """Pull the public-IP / SSH port pair out of a RunPod pod object.

    RunPod's REST API returns `portMappings` as **either** a dict
    `{"22": 39342}` (current shape, observed against rest.runpod.io/v1) or a
    list of `{privatePort, publicPort, ...}` records (older shape / GraphQL).
    `runtime.ports` is the legacy field used by the GraphQL API. Be defensive
    about all three; we only need (ip, port_for_22). Returns (None, None) if
    SSH isn't assigned yet.
    """
    public_ip = pod.get("publicIp") or pod.get("public_ip")

    pms = pod.get("portMappings")

    # Current REST shape: {"22": 39342, "8888": 12345}
    if isinstance(pms, dict):
        for k, v in pms.items():
            try:
                if int(k) == 22 and v:
                    return public_ip, int(v)
            except (TypeError, ValueError):
                continue

    # Older shape: [{privatePort: 22, publicPort: 39342, type: "tcp"}, ...]
    if isinstance(pms, list):
        for pm in pms:
            if not isinstance(pm, dict):
                continue
            priv = pm.get("privatePort") or pm.get("private_port")
            proto = (pm.get("type") or pm.get("protocol") or "").lower()
            if priv == 22 and proto in ("", "tcp"):
                pub = pm.get("publicPort") or pm.get("public_port")
                ip = pm.get("ip") or public_ip
                if pub:
                    return ip, int(pub)

    runtime = pod.get("runtime") or {}
    for p in (runtime.get("ports") or []):
        if not isinstance(p, dict):
            continue
        if (p.get("privatePort") == 22 or p.get("private_port") == 22) and (
            (p.get("type") or "tcp").lower() == "tcp"
        ):
            pub = p.get("publicPort") or p.get("public_port")
            ip = p.get("ip") or public_ip
            if pub:
                return ip, int(pub)

    return public_ip, None


def _extract_cuda_version(image: str) -> Optional[str]:
    """Pull a CUDA major.minor version out of an image name.

    RunPod-style: 'runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04'
    Bare:        'nvidia/cuda:12.8.0-devel-ubuntu22.04'
    PI-style:    'cuda_12_6_pytorch_2_7' (handled too, returns '12.6')

    Returns None if no recognisable version is present. We pass the result
    through to RunPod's allowedCudaVersions so a CUDA-12.8 image lands on a
    host with a ≥555 driver instead of failing at runtime with 'forward
    compatibility was attempted on non-supported HW'.
    """
    if not image:
        return None
    import re
    # Match dotted form first (runpod/* / bare images).
    m = re.search(r"cuda[:\-_]?(\d+)\.(\d+)", image, re.IGNORECASE)
    if not m:
        # Underscore form (PI enum).
        m = re.search(r"cuda_(\d+)_(\d+)", image, re.IGNORECASE)
    if not m:
        return None
    return f"{m.group(1)}.{m.group(2)}"


# ---------- Prime Intellect helpers ------------------------------------


def _pi_api_base() -> str:
    return os.environ.get("PI_API_BASE", "https://api.primeintellect.ai").rstrip("/")


def _pi_client(api_key: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=_pi_api_base(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )


# Provider-native GPU catalogs. Each provider exposes its own list (served by
# /compute/{kind}/gpu-types) so the new-pod form can show real per-provider
# hardware instead of pretending one universal naming scheme exists. The `id`
# field is the *provider-native* enum we send and store in `compute_pods.gpu_type`.

RUNPOD_GPU_TYPES: list[dict[str, Any]] = [
    {"id": "NVIDIA RTX A4000", "label": "RTX A4000", "vram_gb": 16, "hint": "cheap baseline"},
    {"id": "NVIDIA RTX A5000", "label": "RTX A5000", "vram_gb": 24, "hint": ""},
    {"id": "NVIDIA RTX A6000", "label": "RTX A6000", "vram_gb": 48, "hint": ""},
    {"id": "NVIDIA GeForce RTX 4090", "label": "RTX 4090", "vram_gb": 24, "hint": "consumer"},
    {"id": "NVIDIA GeForce RTX 5090", "label": "RTX 5090", "vram_gb": 32, "hint": "consumer · Blackwell"},
    {"id": "NVIDIA L4", "label": "L4", "vram_gb": 24, "hint": ""},
    {"id": "NVIDIA L40", "label": "L40", "vram_gb": 48, "hint": ""},
    {"id": "NVIDIA L40S", "label": "L40S", "vram_gb": 48, "hint": "faster L40"},
    {"id": "NVIDIA A40", "label": "A40", "vram_gb": 48, "hint": ""},
    {"id": "NVIDIA A100 80GB PCIe", "label": "A100 80GB PCIe", "vram_gb": 80, "hint": "datacenter"},
    {"id": "NVIDIA A100-SXM4-80GB", "label": "A100 80GB SXM", "vram_gb": 80, "hint": "datacenter"},
    {"id": "NVIDIA H100 PCIe", "label": "H100 PCIe", "vram_gb": 80, "hint": ""},
    {"id": "NVIDIA H100 80GB HBM3", "label": "H100 80GB SXM", "vram_gb": 80, "hint": "fastest H100"},
    {"id": "NVIDIA H100 NVL", "label": "H100 NVL", "vram_gb": 94, "hint": ""},
    {"id": "NVIDIA H200", "label": "H200", "vram_gb": 141, "hint": "newest"},
    {"id": "NVIDIA B200", "label": "B200", "vram_gb": 180, "hint": "Blackwell datacenter"},
]

PI_GPU_TYPES: list[dict[str, Any]] = [
    {"id": "RTX3090_24GB", "label": "RTX 3090", "vram_gb": 24, "hint": "consumer"},
    {"id": "RTX4090_24GB", "label": "RTX 4090", "vram_gb": 24, "hint": "consumer"},
    {"id": "RTX5090_32GB", "label": "RTX 5090", "vram_gb": 32, "hint": "consumer · Blackwell"},
    {"id": "A4000_16GB", "label": "RTX A4000", "vram_gb": 16, "hint": "cheap baseline"},
    {"id": "A5000_24GB", "label": "RTX A5000", "vram_gb": 24, "hint": ""},
    {"id": "A6000_48GB", "label": "RTX A6000", "vram_gb": 48, "hint": ""},
    {"id": "A10_24GB", "label": "A10", "vram_gb": 24, "hint": ""},
    {"id": "L4_24GB", "label": "L4", "vram_gb": 24, "hint": ""},
    {"id": "L40_48GB", "label": "L40", "vram_gb": 48, "hint": ""},
    {"id": "L40S_48GB", "label": "L40S", "vram_gb": 48, "hint": "faster L40"},
    {"id": "A100_40GB", "label": "A100 40GB", "vram_gb": 40, "hint": ""},
    {"id": "A100_80GB", "label": "A100 80GB", "vram_gb": 80, "hint": "datacenter"},
    {"id": "H100_80GB", "label": "H100 80GB", "vram_gb": 80, "hint": "fastest H100"},
    {"id": "H200_141GB", "label": "H200", "vram_gb": 141, "hint": "newest"},
    {"id": "B200_180GB", "label": "B200", "vram_gb": 180, "hint": "Blackwell datacenter"},
    {"id": "MI300X_192GB", "label": "MI300X", "vram_gb": 192, "hint": "AMD"},
]


# Back-compat: rows created before the per-provider catalog (or via the old
# benchmark form) store PI gpu_type in RunPod long-form ("NVIDIA H100 80GB
# HBM3"). _map_pi_gpu translates those legacy values to PI's enum at create /
# availability time so existing pods keep working. For new rows submitted from
# the per-provider form, gpu_type is already PI-native and this is a no-op.
_PI_GPU_MAP = {
    # RunPod-form (what new-pod-form.tsx ships)
    "NVIDIA H100 80GB HBM3": "H100_80GB",
    "NVIDIA A100 80GB PCIe": "A100_80GB",
    "NVIDIA GeForce RTX 4090": "RTX4090_24GB",
    "NVIDIA L40S": "L40S_48GB",
    "NVIDIA L40": "L40_48GB",
    "NVIDIA RTX A6000": "A6000_48GB",
    "NVIDIA RTX A5000": "A5000_24GB",
    "NVIDIA RTX A4000": "A4000_16GB",
    # Short aliases
    "H100": "H100_80GB",
    "H100_80GB": "H100_80GB",
    "A100": "A100_80GB",
    "A100_80GB": "A100_80GB",
    "A100-40G": "A100_40GB",
    "A100_40GB": "A100_40GB",
    "A10G": "A10_24GB",
    "A10": "A10_24GB",
    "L40S": "L40S_48GB",
    "L40": "L40_48GB",
    "L4": "L4_24GB",
    "RTX4090": "RTX4090_24GB",
    "RTX4090_24GB": "RTX4090_24GB",
    "RTX3090": "RTX3090_24GB",
    "RTX3090_24GB": "RTX3090_24GB",
    "A6000": "A6000_48GB",
    "A5000": "A5000_24GB",
    "A4000": "A4000_16GB",
}


def _map_pi_gpu(name: str) -> str:
    return _PI_GPU_MAP.get(name, name)


def _pi_security(cloud_type: str) -> str:
    """Map our COMMUNITY/SECURE knob to PI's security enum."""
    return "secure_cloud" if cloud_type.upper() == "SECURE" else "community_cloud"


async def _pi_pick_cloud_candidates(
    api_key: str,
    gpu: str,
    count: int,
    security: str,
) -> list[dict[str, Any]]:
    """Hit PI's availability endpoint, return all in-stock matches ordered
    cheapest-first.

    The pod-creator walks this list until one succeeds. We don't pre-filter
    by image because PI's `images` array on availability rows is unreliable
    (sometimes empty, sometimes display names rather than enum values).
    Raises HTTPException(409) if no in-stock rows.
    """
    pi_gpu = _map_pi_gpu(gpu)
    params: dict[str, Any] = {
        "gpu_type": pi_gpu,
        "gpu_count": count,
        "security": security,
        "page_size": 100,
    }
    async with _pi_client(api_key) as cli:
        r = await cli.get("/api/v1/availability/gpus", params=params)
    if r.status_code == 422:
        # PI returns a giant 422 with the full enum dump when the gpu_type
        # isn't in its catalogue. Don't pass that wall of text to the UI.
        raise HTTPException(
            status_code=400,
            detail={"error": f"Prime Intellect doesn't carry GPU '{gpu}' (mapped to '{pi_gpu}'). Try H100, A100, RTX 4090, L40S, or A6000."},
        )
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"PI availability fetch failed: HTTP {r.status_code}: {r.text[:200]}",
        )
    payload = r.json()
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        if pi_gpu in payload and isinstance(payload[pi_gpu], list):
            rows = [x for x in payload[pi_gpu] if isinstance(x, dict)]
        elif "data" in payload and isinstance(payload["data"], list):
            rows = [x for x in payload["data"] if isinstance(x, dict)]
        else:
            for v in payload.values():
                if isinstance(v, list):
                    rows.extend(x for x in v if isinstance(x, dict))
    elif isinstance(payload, list):
        rows = [x for x in payload if isinstance(x, dict)]

    in_stock = [
        r for r in rows
        if (r.get("stockStatus") or "").lower() in ("available", "low", "")
        and (not r.get("security") or r.get("security") == security)
    ]
    if not in_stock:
        other_tier = "secure_cloud" if security == "community_cloud" else "community_cloud"
        other_in_stock = any(
            (r.get("stockStatus") or "").lower() in ("available", "low", "")
            and (not r.get("security") or r.get("security") == other_tier)
            for r in rows
        )
        hint = (
            f" — but {other_tier.replace('_', ' ')} has capacity, try switching the Cloud tier."
            if other_in_stock else ""
        )
        raise HTTPException(
            status_code=409,
            detail={"error": f"Prime Intellect: no {gpu}×{count} pods available in {security.replace('_', ' ')} right now{hint}"},
        )

    def _price(row: dict[str, Any]) -> float:
        v = row.get("prices") or {}
        if isinstance(v, dict):
            for k in ("onDemand", "communityPrice", "price"):
                if k in v:
                    try:
                        return float(v[k])
                    except (TypeError, ValueError):
                        pass
        return float("inf")

    in_stock.sort(key=_price)
    out: list[dict[str, Any]] = []
    for row in in_stock:
        out.append({
            "cloudId": row.get("cloudId") or "",
            "provider": row.get("provider") or "primeintellect",
            "socket": row.get("socket") or "PCIe",
            "price_hr": _price(row) if _price(row) != float("inf") else None,
            "dataCenterId": row.get("dataCenter") or row.get("dataCenterId"),
            "country": row.get("country"),
        })
    return out


def _pi_extract_ssh(pod: dict[str, Any]) -> tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
    """Pull (public_ip, ssh_port, jupyter_port, ssh_user) from a PI pod object.

    PI returns `primePortMapping` as `[{internal, external, protocol, usedBy,
    description}, ...]`. Different sub-providers map SSH to different
    internal ports (Lambda Labs uses 1234, not 22) so we prefer the explicit
    `usedBy` label and only fall back to the well-known ports. SSH user
    comes from the `sshConnection: "user@ip"` field (sub-providers like
    massedcompute use `ubuntu`, not the default `root`).
    """
    ip: Optional[str] = None
    raw_ip = pod.get("ip") or pod.get("publicIp")
    if isinstance(raw_ip, str):
        ip = raw_ip
    elif isinstance(raw_ip, list) and raw_ip:
        ip = raw_ip[0] if isinstance(raw_ip[0], str) else None

    ssh_user: Optional[str] = None
    sc = pod.get("sshConnection")
    candidates = sc if isinstance(sc, list) else ([sc] if isinstance(sc, str) else [])
    for s in candidates:
        if not isinstance(s, str):
            continue
        # Common shapes: "ubuntu@1.2.3.4", "ssh root@1.2.3.4 -p 1234"
        import re
        m = re.search(r"(?:ssh\s+)?([a-zA-Z][\w\-]*)@([\w\.\-]+)", s)
        if m:
            ssh_user = m.group(1)
            if not ip:
                ip = m.group(2)
            break

    ssh_port: Optional[int] = None
    jupyter_port: Optional[int] = None
    for pm in pod.get("primePortMapping") or []:
        if not isinstance(pm, dict):
            continue
        used_by = (pm.get("usedBy") or "").upper()
        internal = pm.get("internal") or pm.get("privatePort")
        external = pm.get("external") or pm.get("publicPort")
        try:
            internal_i = int(internal) if internal is not None else None
            external_i = int(external) if external is not None else None
        except (TypeError, ValueError):
            continue
        if not external_i:
            continue
        if used_by == "SSH" or internal_i == 22:
            ssh_port = external_i
        elif used_by in ("JUPYTER", "JUPYTER_NOTEBOOK", "JUPYTERLAB") or internal_i == 8888:
            jupyter_port = external_i

    # Last-resort port fallback: parse "-p 1234" or ":1234" off sshConnection.
    if ip and not ssh_port:
        sc2 = pod.get("sshConnection")
        candidates2 = sc2 if isinstance(sc2, list) else ([sc2] if isinstance(sc2, str) else [])
        for s in candidates2:
            if not isinstance(s, str):
                continue
            import re
            m = re.search(r"(?:-p\s+(\d+)|:(\d+))", s)
            if m:
                port_str = m.group(1) or m.group(2)
                try:
                    ssh_port = int(port_str)
                except ValueError:
                    pass
                break
    # Bare-host SSH (massedcompute "ubuntu@1.2.3.4") with no port spec → 22.
    if ip and ssh_user and not ssh_port:
        ssh_port = 22

    return ip, ssh_port, jupyter_port, ssh_user


async def _pi_ensure_ssh_key_id(provider_id: str) -> str:
    """Return the provider's PI sshKeyId, uploading the stored pubkey if the
    provider row pre-dates the SSH-upload-at-create change."""
    from . import crypto
    from .db import Provider
    async with session_factory()() as s:
        prov = await s.get(Provider, provider_id)
        if prov is None:
            raise RuntimeError(f"provider {provider_id} vanished")
        cfg = dict(prov.config or {})
        key_id = cfg.get("pi_ssh_key_id")
        if key_id:
            return key_id
        api_enc = cfg.get("api_key_enc")
        ssh_pub = cfg.get("ssh_pub")
        if not api_enc or not ssh_pub:
            raise RuntimeError(f"provider {provider_id} missing api_key/ssh_pub")
        from .providers_api import _pi_upload_ssh_key
        api_key = crypto.decrypt(api_enc)
        key_id = await _pi_upload_ssh_key(api_key, f"sgpu-{provider_id}", ssh_pub)
        cfg["pi_ssh_key_id"] = key_id
        prov.config = cfg
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(prov, "config")
        await s.commit()
        return key_id


async def _create_pi_pod(pod_id: str) -> None:
    """Background task for PI: availability → POST /pods → poll until ACTIVE."""
    async with session_factory()() as s:
        row = await s.get(ComputePod, pod_id)
        if row is None:
            return

    if not row.provider_id:
        await _mark_failed(pod_id, "PI compute requires a provider_id")
        return

    try:
        api_key = await _resolve_api_key(row.provider_id, expected_kind="pi")
    except Exception as e:
        await _mark_failed(pod_id, f"PI creds resolve failed: {e}"[:4000])
        return

    # Resume path: row already has an upstream pi_pod_id, skip the cloud-pick
    # and create POST and jump straight to polling.
    resuming = bool(row.runpod_pod_id)
    if resuming:
        jupyter_password = row.jupyter_password or secrets.token_urlsafe(18)
        if not row.jupyter_password:
            async with session_factory()() as s:
                r2 = await s.get(ComputePod, pod_id)
                if r2 is not None:
                    r2.jupyter_password = jupyter_password
                    await s.commit()
        pi_pod_id = row.runpod_pod_id
        async with _pi_client(api_key) as cli:
            await _pi_poll_until_ready(pod_id, pi_pod_id, jupyter_password, cli)
        return

    try:
        ssh_key_id = await _pi_ensure_ssh_key_id(row.provider_id)
    except Exception as e:
        await _mark_failed(pod_id, f"PI ssh key registration failed: {e}"[:4000])
        return

    security = _pi_security(row.cloud_type)
    image_for_pick = row.image or "cuda_12_6_pytorch_2_7"
    try:
        candidates = await _pi_pick_cloud_candidates(
            api_key, row.gpu_type, row.gpu_count, security,
        )
    except HTTPException as he:
        msg = he.detail if isinstance(he.detail, str) else str(he.detail)
        await _mark_failed(pod_id, f"PI availability: {msg}"[:4000])
        return
    except Exception as e:
        await _mark_failed(pod_id, f"PI availability crashed: {e}"[:4000])
        return

    candidates = [c for c in candidates if c.get("cloudId")]
    if not candidates:
        await _mark_failed(pod_id, "PI availability returned no cloudId")
        return

    # Cap the number of cloudIds we try — each failed create round-trip is
    # ~2-5s and ends up in the audit log; bound the noise.
    MAX_PI_CREATE_ATTEMPTS = 5
    candidates = candidates[:MAX_PI_CREATE_ATTEMPTS]

    # JupyterLab password (PI's pre-baked images respect jupyterPassword).
    jupyter_password = secrets.token_urlsafe(18)
    async with session_factory()() as s:
        r2 = await s.get(ComputePod, pod_id)
        if r2 is None:
            return
        r2.jupyter_password = jupyter_password
        await s.commit()

    def _build_body(cloud: dict[str, Any]) -> dict[str, Any]:
        pod_body: dict[str, Any] = {
            "name": f"{_name_prefix()}-{pod_id}",
            "cloudId": cloud["cloudId"],
            "gpuType": _map_pi_gpu(row.gpu_type),
            "socket": cloud["socket"],
            "gpuCount": max(1, int(row.gpu_count)),
            "diskSize": int(row.container_disk_gb) + int(row.volume_gb),
            "image": row.image or "cuda_12_6_pytorch_2_7",
            "security": security,
            "sshKeyId": ssh_key_id,
            "jupyterPassword": jupyter_password,
            "autoRestart": False,
        }
        if cloud.get("dataCenterId"):
            pod_body["dataCenterId"] = cloud["dataCenterId"]
        if cloud.get("country"):
            pod_body["country"] = cloud["country"]
        return {"pod": pod_body, "provider": {"type": cloud["provider"]}}

    # Skip-on-retry hints: PI rejects with these substrings when the chosen
    # sub-provider can't serve the image we want. Falling through to the next
    # candidate is the right move; anything else is a real failure.
    SKIPPABLE_HINTS = (
        "is not supported for image",
        "image is not supported",
        "image not available",
        "not available for image",
    )

    pi_pod_id: Optional[str] = None
    cost_f: Optional[float] = None
    cloud: Optional[dict[str, Any]] = None
    last_error: str = ""

    async with _pi_client(api_key) as cli:
        for idx, candidate in enumerate(candidates):
            body = _build_body(candidate)
            try:
                r = await cli.post("/api/v1/pods/", json=body)
            except Exception as e:
                last_error = f"PI create request crashed (candidate {idx + 1}/{len(candidates)} {candidate['provider']}/{candidate['cloudId']}): {e}"
                logger.warning("compute %s: %s", pod_id, last_error)
                continue

            if r.status_code >= 400:
                resp_text = r.text or ""
                last_error = f"PI refused create (candidate {idx + 1}/{len(candidates)} {candidate['provider']}/{candidate['cloudId']}): HTTP {r.status_code} {resp_text}"
                logger.warning("compute %s: %s", pod_id, last_error)
                lower = resp_text.lower()
                if any(h in lower for h in SKIPPABLE_HINTS):
                    continue
                # Non-skippable 4xx/5xx — bail with this error.
                break

            data = r.json()
            pi_pod_id = data.get("id") or (data.get("data") or {}).get("id")
            if not pi_pod_id:
                last_error = f"PI response missing id (candidate {idx + 1}): {str(data)[:200]}"
                continue

            cost_raw = data.get("priceHr") or data.get("price_hr") or candidate.get("price_hr")
            try:
                cost_f = float(cost_raw) if cost_raw is not None else None
            except (TypeError, ValueError):
                cost_f = None
            cloud = candidate
            logger.info(
                "compute %s: PI pod created on %s/%s after %d attempt(s) (price_hr=%s)",
                pod_id, candidate["provider"], candidate["cloudId"], idx + 1, cost_f,
            )
            break

        if not pi_pod_id or cloud is None:
            await _mark_failed(pod_id, last_error[:4000] or "PI: exhausted all candidates")
            return

        async with session_factory()() as s:
            row = await s.get(ComputePod, pod_id)
            if row is None:
                try:
                    await cli.delete(f"/api/v1/pods/{pi_pod_id}")
                except Exception:
                    pass
                return
            # We re-use the runpod_pod_id column for PI's pod id — same shape,
            # rename later. provider_id distinguishes the kind for teardown.
            row.runpod_pod_id = pi_pod_id
            row.cost_per_hr = cost_f
            await s.commit()

        await _pi_poll_until_ready(pod_id, pi_pod_id, jupyter_password, cli)


async def _pi_poll_until_ready(
    pod_id: str,
    pi_pod_id: str,
    jupyter_password: str,
    cli: httpx.AsyncClient,
) -> None:
    """Poll PI until the pod is ACTIVE with SSH coords. Shared by create
    and the resume-on-restart path. Marks the row failed on terminal status
    or after PI_POLL_TIMEOUT_S."""
    deadline = time.time() + PI_POLL_TIMEOUT_S
    last_logged_status: Optional[str] = None
    while time.time() < deadline:
        await asyncio.sleep(POLL_INTERVAL_S)

        async with session_factory()() as s:
            row = await s.get(ComputePod, pod_id)
            if row is None:
                try:
                    await cli.delete(f"/api/v1/pods/{pi_pod_id}")
                except Exception:
                    pass
                return
            if row.status == "terminated":
                return

        try:
            pr = await cli.get(f"/api/v1/pods/{pi_pod_id}")
        except Exception as e:
            logger.warning("compute %s: PI poll crashed: %s", pod_id, e)
            continue

        if pr.status_code >= 400:
            logger.warning(
                "compute %s: PI poll HTTP %s %s",
                pod_id, pr.status_code, pr.text[:200],
            )
            continue

        pod = pr.json() or {}
        status = (pod.get("status") or "").upper()
        install = (pod.get("installationStatus") or "").upper()
        transition = f"{status}/{install}"
        if transition != last_logged_status:
            logger.info("compute %s: PI status %s", pod_id, transition)
            last_logged_status = transition
        if status in ("ERROR", "TERMINATED"):
            await _mark_failed(
                pod_id, f"PI pod reached status {status}: {pod.get('statusMessage') or ''}"[:4000],
            )
            return
        if status != "ACTIVE":
            continue
        if install in ("PENDING", "") and not pod.get("primePortMapping"):
            continue

        ip, ssh_port, jup_port, ssh_user = _pi_extract_ssh(pod)
        if ip and ssh_port:
            from urllib.parse import quote
            jurl = None
            if jup_port:
                jurl = f"http://{ip}:{jup_port}/?token={quote(jupyter_password, safe='')}"
            async with session_factory()() as s:
                row = await s.get(ComputePod, pod_id)
                if row is None or row.status == "terminated":
                    return
                row.public_ip = ip
                row.ssh_port = ssh_port
                if ssh_user:
                    row.ssh_user = ssh_user
                row.status = "running"
                row.ready_at = datetime.now(timezone.utc)
                row.last_active_at = row.ready_at  # start the idle clock at ready
                row.error_text = None
                if jurl:
                    row.jupyter_url_override = jurl
                await s.commit()
            logger.info(
                "compute %s: PI pod ready ssh=%s@%s:%s jupyter=%s",
                pod_id, ssh_user or "root", ip, ssh_port, jurl,
            )
            return

    await _mark_failed(
        pod_id,
        f"PI pod SSH not ready after {PI_POLL_TIMEOUT_S}s — check Prime Intellect dashboard.",
    )


async def _delete_pi(pi_pod_id: str, provider_id: Optional[str] = None) -> None:
    if not provider_id:
        logger.warning("compute: PI delete %s without provider_id — skipping", pi_pod_id)
        return
    try:
        api_key = await _resolve_api_key(provider_id, expected_kind="pi")
    except Exception as e:
        logger.warning("compute: PI delete %s — cred resolve failed: %s", pi_pod_id, e)
        return
    async with _pi_client(api_key) as cli:
        try:
            r = await cli.delete(f"/api/v1/pods/{pi_pod_id}")
            if r.status_code >= 400 and r.status_code != 404:
                logger.warning(
                    "compute: PI delete %s returned %s %s",
                    pi_pod_id, r.status_code, r.text[:200],
                )
        except Exception:
            logger.exception("compute: PI delete request for %s crashed", pi_pod_id)


# ---------- Provision flow ---------------------------------------------


async def _create_pod(pod_id: str) -> None:
    """Background task: POST /pods, then poll until SSH coords land."""
    async with session_factory()() as s:
        row = await s.get(ComputePod, pod_id)
        if row is None:
            return
    # Resolve RunPod creds up front so we use the same account through the
    # whole provision + poll cycle even if the user later edits providers.
    try:
        api_key = await _resolve_api_key(row.provider_id)
    except Exception as e:
        await _mark_failed(pod_id, f"runpod creds resolve failed: {e}"[:4000])
        return
    # When the pod was launched via a kind=runpod provider, fetch its
    # generated public key so we can inject it into the pod's
    # authorized_keys via env.PUBLIC_KEY. SSH-info later returns the
    # matching private half.
    ssh_pub: Optional[str] = None
    if row.provider_id:
        from . import crypto
        from .db import Provider
        async with session_factory()() as s:
            prov = await s.get(Provider, row.provider_id)
            if prov is not None:
                ssh_pub = (prov.config or {}).get("ssh_pub")

    # JupyterLab password is generated up front so we can persist it on the
    # row before the pod actually exists — that way if the create succeeds
    # but persistence races, we don't lose the only copy.
    jupyter_password = row.jupyter_password or secrets.token_urlsafe(18)
    if not row.jupyter_password:
        async with session_factory()() as s:
            r2 = await s.get(ComputePod, pod_id)
            if r2 is None:
                return
            r2.jupyter_password = jupyter_password
            await s.commit()

    # Resume path: row already has a runpod_pod_id — skip the create POST,
    # only poll. Keeps gateway restarts non-destructive to billing pods.
    if row.runpod_pod_id:
        runpod_id = row.runpod_pod_id
        async with _client(api_key=api_key) as cli:
            await _runpod_poll_until_ready(pod_id, runpod_id, cli)
        return

    body: dict[str, Any] = {
        "name": f"{_name_prefix()}-{pod_id}",
        "imageName": row.image,
        "gpuTypeIds": [_map_gpu(row.gpu_type)],
        "gpuCount": max(1, int(row.gpu_count)),
        "cloudType": row.cloud_type,
        "containerDiskInGb": int(row.container_disk_gb),
        "volumeInGb": int(row.volume_gb),
        # 22/tcp → public TCP port for SSH; 8888/http is the JupyterLab
        # listener which RunPod proxies at <podId>-8888.proxy.runpod.net.
        "ports": ["22/tcp", "8888/http"],
        # JUPYTER_PASSWORD is the contract runpod/pytorch start.sh uses to
        # decide whether to launch JupyterLab. Setting it = always-on.
        # PUBLIC_KEY (when present) is what runpod/pytorch start.sh appends
        # to ~/.ssh/authorized_keys so the user can SSH in with the key we
        # generated for this provider row.
        "env": {
            "JUPYTER_PASSWORD": jupyter_password,
            **({"PUBLIC_KEY": ssh_pub} if ssh_pub else {}),
        },
    }
    # If the chosen image embeds a CUDA version, restrict host selection so
    # the pod lands on a machine with a matching (or newer) driver. Without
    # this a CUDA 12.8 image can be scheduled onto a 12.4-driver host and
    # CUDA init fails at container start.
    cuda_v = _extract_cuda_version(row.image or "")
    if cuda_v:
        body["allowedCudaVersions"] = [cuda_v]

    async with _client(api_key=api_key) as cli:
        try:
            r = await cli.post("/pods", json=body)
        except Exception as e:
            logger.exception("compute %s: create request crashed", pod_id)
            await _mark_failed(pod_id, f"network error creating pod: {e}"[:4000])
            return

        if r.status_code >= 400:
            await _mark_failed(
                pod_id,
                f"RunPod refused create: HTTP {r.status_code} {r.text}"[:4000],
            )
            return

        data = r.json()
        runpod_id = data.get("id")
        if not runpod_id:
            await _mark_failed(pod_id, f"RunPod response missing id: {data}"[:4000])
            return

        # Cost is sometimes returned at create time; fall back to None.
        cost = data.get("costPerHr") or data.get("cost_per_hr")
        try:
            cost_f = float(cost) if cost is not None else None
        except (TypeError, ValueError):
            cost_f = None

        async with session_factory()() as s:
            row = await s.get(ComputePod, pod_id)
            if row is None:
                # User deleted before create returned — clean up the orphaned RunPod pod.
                try:
                    await cli.delete(f"/pods/{runpod_id}")
                except Exception:
                    logger.warning("compute %s: orphan cleanup failed", pod_id)
                return
            row.runpod_pod_id = runpod_id
            row.cost_per_hr = cost_f
            await s.commit()

        await _runpod_poll_until_ready(pod_id, runpod_id, cli)


async def _runpod_poll_until_ready(
    pod_id: str,
    runpod_id: str,
    cli: httpx.AsyncClient,
) -> None:
    """Poll RunPod until SSH coords land or POLL_TIMEOUT_S elapses. Shared
    between fresh-create and resume-on-restart paths."""
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        await asyncio.sleep(POLL_INTERVAL_S)

        async with session_factory()() as s:
            row = await s.get(ComputePod, pod_id)
            if row is None:
                try:
                    await cli.delete(f"/pods/{runpod_id}")
                except Exception:
                    pass
                return
            if row.status == "terminated":
                return

        try:
            pr = await cli.get(f"/pods/{runpod_id}")
        except Exception as e:
            logger.warning("compute %s: poll crashed: %s", pod_id, e)
            continue

        if pr.status_code >= 400:
            logger.warning(
                "compute %s: poll HTTP %s %s",
                pod_id, pr.status_code, pr.text[:200],
            )
            continue

        pod = pr.json() or {}
        ip, port = _extract_ssh(pod)
        if ip and port:
            async with session_factory()() as s:
                row = await s.get(ComputePod, pod_id)
                if row is None or row.status == "terminated":
                    return
                row.public_ip = ip
                row.ssh_port = port
                row.status = "running"
                row.ready_at = datetime.now(timezone.utc)
                row.last_active_at = row.ready_at  # start the idle clock at ready
                row.error_text = None
                await s.commit()
            logger.info("compute %s: SSH ready at %s:%s", pod_id, ip, port)
            return

    await _mark_failed(
        pod_id,
        f"SSH not ready after {POLL_TIMEOUT_S}s — pod may still be pulling. "
        "Check RunPod dashboard.",
    )


async def _mark_failed(pod_id: str, error: str) -> None:
    async with session_factory()() as s:
        row = await s.get(ComputePod, pod_id)
        if row is None or row.status in ("terminated", "failed"):
            return
        row.status = "failed"
        row.error_text = error[:4000]
        row.terminated_at = datetime.now(timezone.utc)
        await s.commit()
    logger.warning("compute %s: failed — %s", pod_id, error)


async def _delete_runpod(runpod_id: str, provider_id: Optional[str] = None) -> None:
    try:
        api_key = await _resolve_api_key(provider_id)
    except Exception as e:
        logger.warning("compute: delete %s — cred resolve failed: %s", runpod_id, e)
        return
    async with _client(api_key=api_key) as cli:
        try:
            r = await cli.delete(f"/pods/{runpod_id}")
            if r.status_code >= 400 and r.status_code != 404:
                logger.warning(
                    "compute: delete %s returned %s %s",
                    runpod_id, r.status_code, r.text[:200],
                )
        except Exception:
            logger.exception("compute: delete request for %s crashed", runpod_id)


# ---------- Idle auto-terminate ----------------------------------------
#
# A pod with idle_terminate_after_s > 0 is torn down once it has shown no GPU
# compute AND no GPU memory in use for that many seconds. We SSH into the pod
# each sweep and read `nvidia-smi` for live utilization + memory — RunPod's
# REST API returns an empty `runtime` (no per-GPU stats), so the provider API
# can't tell us. Works the same for RunPod and Prime Intellect (both give SSH).
# Deliberately FAIL-SAFE: if the pod is unreachable or nvidia-smi can't be
# parsed, we treat it as active and never auto-kill it.

# A GPU counts as "in use" if compute utilization exceeds this percent OR
# resident GPU memory exceeds this many MiB. The memory floor sits ABOVE the
# idle driver/runtime baseline — a bare RunPod card with nothing running still
# reports ~0.5-1 GiB used (observed 767 MiB on an idle A100), while a loaded
# model occupies many GiB. 2 GiB cleanly separates "nothing running" from
# "a workload is resident".
_IDLE_UTIL_PCT = 1
_IDLE_MEM_MIB = 2048
# How often the sweep runs. Small idle windows are only as precise as this.
_IDLE_CHECK_INTERVAL_S = 30


async def _resolve_pod_private_key(s: AsyncSession, pod: ComputePod) -> Optional[str]:
    """The private key whose public half is in the pod's authorized_keys: the
    spawning provider's generated key, else the gateway's file-based key."""
    if pod.provider_id:
        from . import crypto
        from .db import Provider
        prov = await s.get(Provider, pod.provider_id)
        if prov is not None:
            enc = (prov.config or {}).get("ssh_priv_enc")
            if enc:
                try:
                    return crypto.decrypt(enc)
                except Exception:
                    pass  # fall through to the file key
    try:
        return Path(_ssh_key_path()).read_text()
    except FileNotFoundError:
        return None


async def _fetch_pod_busy(pod_id: str) -> Optional[bool]:
    """SSH into the pod, read nvidia-smi. True=busy, False=idle, None=can't tell
    (unreachable / unparseable → caller treats as active, fail-safe)."""
    async with session_factory()() as s:
        pod = await s.get(ComputePod, pod_id)
        if (
            pod is None or pod.status != "running"
            or not pod.public_ip or not pod.ssh_port
        ):
            return None
        host, port, user = pod.public_ip, pod.ssh_port, pod.ssh_user
        key = await _resolve_pod_private_key(s, pod)
    if not key:
        logger.info("compute %s: idle-probe no SSH key resolved -> treat as active", pod_id)
        return None
    try:
        res = await vm_probe.availability_vm(host, port, user, key)
    except Exception as e:
        logger.warning("compute %s: nvidia-smi probe crashed: %s", pod_id, e)
        return None
    if not res.ok or not res.gpus:
        logger.info(
            "compute %s: idle-probe inconclusive (ok=%s msg=%r gpus=%d) -> treat as active",
            pod_id, res.ok, res.message, len(res.gpus),
        )
        return None
    busy = False
    detail = []
    for g in res.gpus:
        mem_used_mib = g.mem_total_mib - g.mem_free_mib
        detail.append(f"gpu{g.index} util={g.util_pct}% mem_used={mem_used_mib}MiB")
        if g.util_pct > _IDLE_UTIL_PCT or mem_used_mib > _IDLE_MEM_MIB:
            busy = True
    logger.info("compute %s: idle-probe busy=%s [%s]", pod_id, busy, "; ".join(detail))
    return busy


async def _idle_terminate(pod_id: str) -> None:
    """Tear down a pod the sweep judged idle. Mirrors delete_compute's teardown,
    but marks the pod `auto_terminated` (a manual delete stays `terminated`) and
    attributes the audit row to the owner (logged as compute.idle_terminate)."""
    async with session_factory()() as s:
        row = await s.get(ComputePod, pod_id)
        if row is None or row.status != "running":
            return  # raced with a manual delete / state change
        runpod_id = row.runpod_pod_id
        pod_provider_id = row.provider_id
        pod_name = row.name
        owner_id = row.owner_id
        ready_at = row.ready_at
        cost_per_hr = row.cost_per_hr
        timeout_s = row.idle_terminate_after_s
        row.status = "auto_terminated"
        row.terminated_at = datetime.now(timezone.utc)
        terminated_at = row.terminated_at
        await s.commit()

    kind = await _resolve_pod_kind(pod_id)
    if runpod_id:
        if kind == "pi":
            asyncio.create_task(_delete_pi(runpod_id, provider_id=pod_provider_id))
        else:
            asyncio.create_task(_delete_runpod(runpod_id, provider_id=pod_provider_id))
    logger.info(
        "compute %s: auto-terminated after %ds idle (no GPU compute or memory in use)",
        pod_id, timeout_s,
    )

    # Attribute the audit row to the owner (the system acted on their behalf).
    try:
        async with session_factory()() as s:
            owner = await s.get(User, owner_id)
            if owner is not None:
                details: dict[str, Any] = {
                    "reason": "idle_timeout",
                    "idle_terminate_after_s": timeout_s,
                }
                if runpod_id:
                    details["runpod_pod_id"] = runpod_id
                cost = audit.cost_breakdown(ready_at, terminated_at, cost_per_hr)
                if cost is not None:
                    details.update(cost)
                await audit.record(
                    owner, "compute.idle_terminate", "compute", pod_id, pod_name,
                    details=details,
                )
    except Exception:
        logger.exception("compute %s: idle-terminate audit failed", pod_id)


async def _idle_sweep_once() -> None:
    """One pass: refresh activity for busy/unmeasurable pods, terminate the
    ones idle past their window."""
    async with session_factory()() as s:
        rows = (
            await s.execute(
                select(ComputePod).where(
                    ComputePod.status == "running",
                    ComputePod.idle_terminate_after_s > 0,
                )
            )
        ).scalars().all()
        # Snapshot the fields we need so we don't hold the session during SSH.
        cands = [
            (r.id, r.idle_terminate_after_s, r.last_active_at, r.ready_at, r.created_at)
            for r in rows
            if r.public_ip and r.ssh_port
        ]

    now = datetime.now(timezone.utc)
    for pod_id, timeout_s, last_active, ready_at, created_at in cands:
        busy = await _fetch_pod_busy(pod_id)
        if busy is None or busy:
            # Active, or we couldn't measure it — reset the idle clock so an
            # unmeasurable pod is never auto-killed.
            async with session_factory()() as s:
                r = await s.get(ComputePod, pod_id)
                if r is not None and r.status == "running":
                    r.last_active_at = now
                    await s.commit()
            continue
        base = last_active or ready_at or created_at
        if base is None:
            continue
        if base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)
        if (now - base).total_seconds() >= timeout_s:
            await _idle_terminate(pod_id)


async def compute_idle_loop() -> None:
    """Background task: auto-terminate idle compute pods. Started at gateway
    startup whenever Compute is enabled; runs until the process stops."""
    logger.info(
        "compute idle-terminate loop started (poll every %ds)", _IDLE_CHECK_INTERVAL_S
    )
    while True:
        try:
            await asyncio.sleep(_IDLE_CHECK_INTERVAL_S)
            await _idle_sweep_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("compute: idle sweep crashed (continuing)")


# ---------- Startup hook -----------------------------------------------


async def cleanup_orphaned_running() -> int:
    """Resume polling on `creating` rows whose upstream pod_id is already
    set; mark the rest as failed.

    A `creating` row with `runpod_pod_id` set means the create call landed
    upstream but our polling task died with the previous gateway process —
    upstream is still billing, so we spawn a fresh poll task that picks up
    where the previous one left off.

    A `creating` row with NO `runpod_pod_id` means the upstream create
    never returned (or the row was just inserted and we crashed before
    posting). Safe to mark failed since nothing is billing yet.
    """
    resumed = 0
    async with session_factory()() as s:
        rows = await s.execute(
            select(ComputePod).where(ComputePod.status == "creating")
        )
        all_rows = list(rows.scalars().all())

    failed_ids: list[str] = []
    for row in all_rows:
        if row.runpod_pod_id:
            # Has an upstream id — resume polling on the new event loop.
            asyncio.create_task(_safe_create(row.id))
            resumed += 1
            logger.info(
                "compute %s: resuming poll after gateway restart (upstream=%s)",
                row.id, row.runpod_pod_id,
            )
        else:
            failed_ids.append(row.id)

    if failed_ids:
        async with session_factory()() as s:
            await s.execute(
                update(ComputePod)
                .where(ComputePod.id.in_(failed_ids))
                .values(
                    status="failed",
                    error_text="orphaned by gateway restart before upstream create completed",
                    terminated_at=datetime.now(timezone.utc),
                )
            )
            await s.commit()
        logger.info("compute: marked %d row(s) failed (no upstream id)", len(failed_ids))

    return resumed + len(failed_ids)


# ---------- Pydantic schemas -------------------------------------------


class CreateComputeRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    gpu_type: str = Field(min_length=1, max_length=64)
    gpu_count: int = Field(default=1, ge=1, le=8)
    container_disk_gb: int = Field(default=40, ge=10, le=2000)
    volume_gb: int = Field(default=0, ge=0, le=2000)
    template_id: str = Field(default="pytorch-2.8-cuda12.8", max_length=128)
    # When `template_id` isn't one of the curated favourites the client must
    # pass the resolved image — for RunPod that's the template's `imageName`
    # picked from the search results, for PI it can be omitted (the image
    # IS the template id, an enum value).
    image: Optional[str] = Field(default=None, max_length=512)
    cloud_type: str = Field(default="SECURE", pattern=r"^(COMMUNITY|SECURE)$")
    # Auto-terminate after this many idle seconds (no GPU compute AND no GPU
    # memory used). 0 = disabled. Capped at 24h so a typo can't strand a pod
    # billing "forever".
    idle_terminate_after_s: int = Field(default=0, ge=0, le=86400)
    # NULL = use the gateway-wide RUNPOD_API_KEY env var. When set, must
    # refer to a kind=runpod or kind=pi Provider row.
    provider_id: Optional[str] = None


class ComputeRecord(BaseModel):
    id: str
    name: str
    gpu_type: str
    gpu_count: int
    container_disk_gb: int
    volume_gb: int
    image: str
    template_id: Optional[str] = None
    cloud_type: str
    status: str
    runpod_pod_id: Optional[str] = None
    public_ip: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_user: str
    # JupyterLab is always on; URL is derivable once we know the RunPod pod id.
    jupyter_url: Optional[str] = None
    jupyter_password: Optional[str] = None
    cost_per_hr: Optional[float] = None
    error_text: Optional[str] = None
    reject_reason: Optional[str] = None
    provider_id: Optional[str] = None
    created_by: str
    created_at: str
    ready_at: Optional[str] = None
    terminated_at: Optional[str] = None
    idle_terminate_after_s: int = 0
    last_active_at: Optional[str] = None


class SshInfoResponse(BaseModel):
    ssh_command: str
    ssh_user: str
    ssh_host: str
    ssh_port: int
    private_key: str  # PEM, fetched once at SSH-modal time


class TemplateRecord(BaseModel):
    id: str
    name: str
    image: str
    description: str


class RunpodTemplateSearchResult(BaseModel):
    id: str
    name: str
    image: str
    category: Optional[str] = None
    is_public: bool = False
    is_runpod: bool = False


class PiImageOption(BaseModel):
    id: str
    name: str
    description: str


class RejectRequest(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=1024)


# ---------- HTTP API ---------------------------------------------------


router = APIRouter(prefix="/compute", tags=["compute"])


def _to_record(p: ComputePod, owner_username: str) -> ComputeRecord:
    # Embed the token in the URL so the user clicks once and is logged in.
    # Keeping `jupyter_password` in the response too is harmless — it's the
    # same token, just exposed separately for advanced uses (e.g. reusing in
    # a different client). The UI hides the password field by default.
    if p.jupyter_url_override:
        jurl = p.jupyter_url_override
    elif p.runpod_pod_id:
        jurl = _jupyter_url(p.runpod_pod_id, p.jupyter_password)
    else:
        jurl = None
    return ComputeRecord(
        id=p.id,
        name=p.name,
        gpu_type=p.gpu_type,
        gpu_count=p.gpu_count,
        container_disk_gb=p.container_disk_gb,
        volume_gb=p.volume_gb,
        image=p.image,
        template_id=p.template_id,
        cloud_type=p.cloud_type,
        status=p.status,
        runpod_pod_id=p.runpod_pod_id,
        public_ip=p.public_ip,
        ssh_port=p.ssh_port,
        ssh_user=p.ssh_user,
        jupyter_url=jurl,
        jupyter_password=p.jupyter_password,
        cost_per_hr=p.cost_per_hr,
        error_text=p.error_text,
        reject_reason=p.reject_reason,
        provider_id=p.provider_id,
        created_by=owner_username,
        created_at=p.created_at.isoformat() if p.created_at else "",
        ready_at=p.ready_at.isoformat() if p.ready_at else None,
        terminated_at=p.terminated_at.isoformat() if p.terminated_at else None,
        idle_terminate_after_s=p.idle_terminate_after_s or 0,
        last_active_at=p.last_active_at.isoformat() if p.last_active_at else None,
    )


def _gen_id() -> str:
    return f"cmp-{uuid.uuid4().hex[:8]}"


# Templates route is defined BEFORE /{pod_id} so the literal `/templates`
# isn't swallowed by the path-param matcher. Same trick bench.py uses.


@router.get("/templates", response_model=list[TemplateRecord])
async def list_templates(_: User = Depends(require_section("compute"))):
    return [TemplateRecord(**t) for t in CURATED_TEMPLATES]


class GpuTypeOption(BaseModel):
    id: str
    label: str
    vram_gb: int
    hint: str = ""


# Static GPU catalogs (just names + VRAM) — any authenticated user can read them
# (the Benchmark form needs them too, not only Compute users).
@router.get("/runpod/gpu-types", response_model=list[GpuTypeOption])
async def list_runpod_gpu_types(_: User = Depends(current_user)):
    return [GpuTypeOption(**g) for g in RUNPOD_GPU_TYPES]


@router.get("/pi/gpu-types", response_model=list[GpuTypeOption])
async def list_pi_gpu_types(_: User = Depends(current_user)):
    return [GpuTypeOption(**g) for g in PI_GPU_TYPES]


@router.get("/runpod/templates", response_model=list[RunpodTemplateSearchResult])
async def search_runpod_templates(
    q: str = "",
    limit: int = 50,
    provider_id: Optional[str] = None,
    _: User = Depends(require_section("compute")),
):
    """Search RunPod's full templates catalogue (user + public + runpod).

    Upstream returns the whole list in one shot, so we cache per-key for 5min
    and filter server-side by `q` against name + imageName + category.
    Drops serverless-endpoint templates.
    """
    try:
        api_key = await _resolve_api_key(provider_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail={"error": str(e)})

    rows = await _fetch_runpod_templates(api_key)

    needle = q.strip().lower()
    if needle:
        rows = [
            r for r in rows
            if needle in (r.get("name") or "").lower()
            or needle in (r.get("imageName") or "").lower()
            or needle in (r.get("category") or "").lower()
        ]
    # Sort: runpod-official first, then public, then by name. Most users want
    # the canonical runpod/* images near the top.
    rows.sort(key=lambda r: (not r.get("isRunpod"), not r.get("isPublic"), r.get("name") or ""))
    rows = rows[: max(1, min(limit, 200))]

    return [
        RunpodTemplateSearchResult(
            id=r["id"],
            name=r["name"],
            image=r["imageName"],
            category=r.get("category") or None,
            is_public=bool(r.get("isPublic")),
            is_runpod=bool(r.get("isRunpod")),
        )
        for r in rows
    ]


@router.get("/pi/images", response_model=list[PiImageOption])
async def list_pi_images(_: User = Depends(require_section("compute"))):
    return [PiImageOption(**t) for t in PI_IMAGES]


@router.get("/pi/images/compatible", response_model=list[PiImageOption])
async def list_pi_compatible_images(
    gpu: str,
    count: int = 1,
    cloud_type: str = "COMMUNITY",
    provider_id: Optional[str] = None,
    _: User = Depends(require_section("compute")),
):
    """Return the subset of PI_IMAGES that at least one in-stock sub-provider
    advertises support for, given (gpu, count, security).

    Falls back to the full PI_IMAGES list if no availability row has a usable
    `images` array — better to let the user pick and let the create-retry
    loop sort it out than to show an empty dropdown.
    """
    security = _pi_security(cloud_type)
    pi_gpu = _map_pi_gpu(gpu)

    try:
        api_key = await _resolve_api_key(provider_id, expected_kind="pi")
    except Exception:
        # No creds yet — return everything; the form will still work.
        return [PiImageOption(**t) for t in PI_IMAGES]

    try:
        async with _pi_client(api_key) as cli:
            r = await cli.get(
                "/api/v1/availability/gpus",
                params={
                    "gpu_type": pi_gpu,
                    "gpu_count": count,
                    "security": security,
                    "page_size": 100,
                },
            )
    except Exception:
        return [PiImageOption(**t) for t in PI_IMAGES]

    if r.status_code >= 400:
        return [PiImageOption(**t) for t in PI_IMAGES]

    payload = r.json()
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        if pi_gpu in payload and isinstance(payload[pi_gpu], list):
            rows = [x for x in payload[pi_gpu] if isinstance(x, dict)]
        elif "data" in payload and isinstance(payload["data"], list):
            rows = [x for x in payload["data"] if isinstance(x, dict)]
        else:
            for v in payload.values():
                if isinstance(v, list):
                    rows.extend(x for x in v if isinstance(x, dict))
    elif isinstance(payload, list):
        rows = [x for x in payload if isinstance(x, dict)]

    in_stock = [
        row for row in rows
        if (row.get("stockStatus") or "").lower() in ("available", "low", "")
        and (not row.get("security") or row.get("security") == security)
    ]

    # Collect the union of `images` strings (case-insensitive).
    supported: set[str] = set()
    any_with_images = False
    for row in in_stock:
        imgs = row.get("images")
        if isinstance(imgs, list) and imgs:
            any_with_images = True
            for img in imgs:
                if isinstance(img, str):
                    supported.add(img.lower())

    if not any_with_images:
        # PI didn't populate images on any in-stock row — we can't filter
        # confidently, so show the full catalogue.
        return [PiImageOption(**t) for t in PI_IMAGES]

    matching = [t for t in PI_IMAGES if t["id"].lower() in supported]
    # If our curated list and PI's union don't overlap, fall back to the full
    # catalogue rather than show an empty dropdown.
    if not matching:
        return [PiImageOption(**t) for t in PI_IMAGES]
    return [PiImageOption(**t) for t in matching]


class PiAvailabilityResult(BaseModel):
    gpu: str
    count: int
    available: Optional[bool] = None
    cheapest_price_hr: Optional[float] = None
    regions: list[str] = []
    reason: Optional[str] = None
    checked_at: float
    provider: str = "pi"


@router.get("/pi/availability", response_model=PiAvailabilityResult)
async def check_pi_availability(
    gpu: str,
    count: int = 1,
    cloud_type: str = "COMMUNITY",
    provider_id: Optional[str] = None,
    _: User = Depends(require_section("compute")),
):
    """Live PI capacity check used by the new-pod form to render the
    availability badge for Prime Intellect providers. Mirrors the shape of
    /v1/availability (RunPod) so the frontend can use the same badge."""
    security = _pi_security(cloud_type)
    pi_gpu = _map_pi_gpu(gpu)
    now = datetime.now(timezone.utc).timestamp()

    try:
        api_key = await _resolve_api_key(provider_id, expected_kind="pi")
    except Exception as e:
        return PiAvailabilityResult(
            gpu=gpu, count=count, available=None,
            reason=f"no PI credentials: {e}", checked_at=now,
        )

    params: dict[str, Any] = {
        "gpu_type": pi_gpu,
        "gpu_count": count,
        "security": security,
        "page_size": 100,
    }
    try:
        async with _pi_client(api_key) as cli:
            r = await cli.get("/api/v1/availability/gpus", params=params)
    except Exception as e:
        return PiAvailabilityResult(
            gpu=gpu, count=count, available=None,
            reason=f"network error: {e}", checked_at=now,
        )

    if r.status_code == 422:
        return PiAvailabilityResult(
            gpu=gpu, count=count, available=False,
            reason=f"Prime Intellect doesn't carry '{gpu}'", checked_at=now,
        )
    if r.status_code >= 400:
        return PiAvailabilityResult(
            gpu=gpu, count=count, available=None,
            reason=f"PI HTTP {r.status_code}", checked_at=now,
        )

    payload = r.json()
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        if pi_gpu in payload and isinstance(payload[pi_gpu], list):
            rows = [x for x in payload[pi_gpu] if isinstance(x, dict)]
        elif "data" in payload and isinstance(payload["data"], list):
            rows = [x for x in payload["data"] if isinstance(x, dict)]
        else:
            for v in payload.values():
                if isinstance(v, list):
                    rows.extend(x for x in v if isinstance(x, dict))
    elif isinstance(payload, list):
        rows = [x for x in payload if isinstance(x, dict)]

    in_stock = [
        row for row in rows
        if (row.get("stockStatus") or "").lower() in ("available", "low", "")
        and (not row.get("security") or row.get("security") == security)
    ]
    if not in_stock:
        # If the other tier has stock, hint at it.
        other = "secure_cloud" if security == "community_cloud" else "community_cloud"
        other_in = any(
            (row.get("stockStatus") or "").lower() in ("available", "low", "")
            and (not row.get("security") or row.get("security") == other)
            for row in rows
        )
        hint = f" — try {other.replace('_', ' ')}" if other_in else ""
        return PiAvailabilityResult(
            gpu=gpu, count=count, available=False,
            reason=f"out of stock on {security.replace('_', ' ')}{hint}",
            checked_at=now,
        )

    prices: list[float] = []
    regions: list[str] = []
    for row in in_stock:
        v = row.get("prices") or {}
        if isinstance(v, dict):
            for k in ("onDemand", "communityPrice", "price"):
                if k in v:
                    try:
                        prices.append(float(v[k]))
                        break
                    except (TypeError, ValueError):
                        pass
        for k in ("dataCenter", "data_center", "region", "country"):
            r2 = row.get(k)
            if isinstance(r2, str) and r2 and r2 not in regions:
                regions.append(r2)
                break

    return PiAvailabilityResult(
        gpu=gpu, count=count, available=True,
        cheapest_price_hr=min(prices) if prices else None,
        regions=regions,
        checked_at=now,
    )


# Admin-only approval queue. Routes live under /compute/approvals/* (rather
# than /admin/compute-approvals) so the entire compute surface stays under
# one router prefix and the admin UI doesn't have to think about two roots.


@router.get("/approvals", response_model=list[ComputeRecord])
async def list_approvals(
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    rows = await session.execute(
        select(ComputePod)
        .where(ComputePod.status == "pending_approval")
        .order_by(ComputePod.created_at.asc())
    )
    out: list[ComputeRecord] = []
    for p in rows.scalars().all():
        owner = await session.get(User, p.owner_id)
        out.append(_to_record(p, owner.username if owner else ""))
    return out


@router.post("/{pod_id}/approve", response_model=ComputeRecord)
async def approve_compute(
    pod_id: str,
    actor: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    p = await session.get(ComputePod, pod_id)
    if p is None:
        raise HTTPException(status_code=404, detail={"error": "compute pod not found"})
    if p.status != "pending_approval":
        raise HTTPException(
            status_code=409,
            detail={"error": f"pod is in status '{p.status}', not pending_approval"},
        )
    p.status = "creating"
    p.reject_reason = None
    await session.commit()
    asyncio.create_task(_safe_create(pod_id))
    await audit.record(
        actor, "compute.approve", "compute", pod_id, p.name,
        details={"requester_id": p.owner_id},
    )
    owner = await session.get(User, p.owner_id)
    return _to_record(p, owner.username if owner else "")


@router.post("/{pod_id}/reject", response_model=ComputeRecord)
async def reject_compute(
    pod_id: str,
    body: RejectRequest,
    actor: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    p = await session.get(ComputePod, pod_id)
    if p is None:
        raise HTTPException(status_code=404, detail={"error": "compute pod not found"})
    if p.status != "pending_approval":
        raise HTTPException(
            status_code=409,
            detail={"error": f"pod is in status '{p.status}', not pending_approval"},
        )
    p.status = "rejected"
    p.reject_reason = (body.reason or "").strip() or None
    p.terminated_at = datetime.now(timezone.utc)
    await session.commit()
    await audit.record(
        actor, "compute.reject", "compute", pod_id, p.name,
        details={"requester_id": p.owner_id, "reason": p.reject_reason},
    )
    owner = await session.get(User, p.owner_id)
    return _to_record(p, owner.username if owner else "")


@router.post("", response_model=ComputeRecord)
async def create_compute(
    body: CreateComputeRequest,
    user: User = Depends(require_section("compute")),
    session: AsyncSession = Depends(get_session),
):
    # Compute pods are cloud-only (runpod/pi — there is no vm kind here), so the
    # whole feature is refused when cloud providers are disabled (CAE/CCE).
    if cloud_providers_disabled():
        raise HTTPException(
            status_code=403,
            detail={"error": "cloud GPU providers (RunPod / Prime Intellect) are disabled on this deployment"},
        )

    # Dispatch on provider kind. NULL provider_id = legacy RunPod env path.
    kind = "runpod"
    if body.provider_id:
        from .db import Provider
        prov = await session.get(Provider, body.provider_id)
        if prov is None:
            raise HTTPException(status_code=400, detail={"error": "unknown provider_id"})
        if prov.kind not in ("runpod", "pi"):
            raise HTTPException(
                status_code=400,
                detail={"error": f"provider {body.provider_id} is kind={prov.kind}, compute requires runpod or pi"},
            )
        kind = prov.kind
    elif not os.environ.get("RUNPOD_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail={"error": "no RunPod credentials — register a provider or set RUNPOD_API_KEY"},
        )

    # Resolve the image + template_id we'll persist on the row.
    if kind == "pi":
        pi_img = _resolve_pi_image(body.template_id)
        if pi_img is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"unknown PI image '{body.template_id}'",
                    "available": [t["id"] for t in PI_IMAGES],
                },
            )
        image_resolved = pi_img["id"]
        template_id_resolved = pi_img["id"]
    else:
        curated = _resolve_template(body.template_id)
        if curated is not None:
            image_resolved = curated["image"]
            template_id_resolved = curated["id"]
        else:
            # Non-curated → must be a real RunPod template id; client supplies
            # the resolved imageName so we don't have to round-trip.
            if not body.image or not body.image.strip():
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": f"unknown template_id '{body.template_id}' and no image provided",
                        "available_curated": [t["id"] for t in CURATED_TEMPLATES],
                    },
                )
            image_resolved = body.image.strip()
            template_id_resolved = body.template_id

    # Approval gate: admins and anyone on the seeded `full-access` policy
    # role bypass entirely (treated as trusted). Everyone else lands in
    # `pending_approval` until an admin clicks Approve.
    needs_approval = not user.is_admin and user.policy_role_id != "full-access"
    pod_id = _gen_id()
    row = ComputePod(
        id=pod_id,
        name=body.name.strip(),
        gpu_type=body.gpu_type,
        gpu_count=body.gpu_count,
        container_disk_gb=body.container_disk_gb,
        volume_gb=body.volume_gb,
        image=image_resolved,
        template_id=template_id_resolved,
        cloud_type=body.cloud_type,
        status="pending_approval" if needs_approval else "creating",
        idle_terminate_after_s=body.idle_terminate_after_s,
        owner_id=user.id,
        provider_id=body.provider_id,
    )
    session.add(row)
    await session.commit()

    if not needs_approval:
        asyncio.create_task(_safe_create(pod_id))

    await audit.record(
        user,
        "compute.request" if needs_approval else "compute.create",
        "compute", pod_id, body.name.strip(),
        details={
            "gpu_type": body.gpu_type, "gpu_count": body.gpu_count,
            "template_id": template_id_resolved, "cloud_type": body.cloud_type,
            "provider_kind": kind,
        },
    )

    row = await session.get(ComputePod, pod_id)
    return _to_record(row, user.username)


async def _resolve_pod_kind(pod_id: str) -> str:
    """Look up the provider kind for a compute pod row. NULL provider → runpod
    (legacy env-key path)."""
    from .db import Provider
    async with session_factory()() as s:
        row = await s.get(ComputePod, pod_id)
        if row is None or not row.provider_id:
            return "runpod"
        prov = await s.get(Provider, row.provider_id)
        return prov.kind if prov is not None else "runpod"


async def _safe_create(pod_id: str) -> None:
    try:
        kind = await _resolve_pod_kind(pod_id)
        if kind == "pi":
            await _create_pi_pod(pod_id)
        else:
            await _create_pod(pod_id)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("compute %s: provisioner crashed", pod_id)
        await _mark_failed(pod_id, f"provisioner crashed: {e}"[:4000])


@router.get("", response_model=list[ComputeRecord])
async def list_compute(
    scope: str = "mine",
    user: User = Depends(require_section("compute")),
    session: AsyncSession = Depends(get_session),
):
    # Hide terminated pods — once a pod is gone there's no useful action left
    # and the list gets noisy. Detail page (`/compute/{id}`) still resolves
    # for terminated rows so links from audit / direct URLs keep working.
    #
    # Admins default to their own pods; pass ?scope=all to see everyone's.
    # Non-admins are always scoped to own regardless of the param.
    show_all = user.is_admin and scope == "all"
    # Hide MANUAL terminations (the user did it, knows it's gone) but keep
    # auto_terminated pods visible — the system killed them, so they're worth
    # reviewing (and it's why the status is distinct in the first place).
    stmt = (
        select(ComputePod)
        .where(ComputePod.status != "terminated")
        .order_by(ComputePod.created_at.desc())
    )
    if not show_all:
        stmt = stmt.where(ComputePod.owner_id == user.id)
    rows = await session.execute(stmt)
    out: list[ComputeRecord] = []
    for p in rows.scalars().all():
        owner = await session.get(User, p.owner_id)
        out.append(_to_record(p, owner.username if owner else ""))
    return out


@router.get("/{pod_id}", response_model=ComputeRecord)
async def get_compute(
    pod_id: str,
    user: User = Depends(require_section("compute")),
    session: AsyncSession = Depends(get_session),
):
    p = await session.get(ComputePod, pod_id)
    if not p:
        raise HTTPException(status_code=404, detail={"error": "compute pod not found"})
    if not user.is_admin and p.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    owner = await session.get(User, p.owner_id)
    return _to_record(p, owner.username if owner else "")


@router.get("/{pod_id}/ssh", response_model=SshInfoResponse)
async def get_ssh_info(
    pod_id: str,
    user: User = Depends(require_section("compute")),
    session: AsyncSession = Depends(get_session),
):
    p = await session.get(ComputePod, pod_id)
    if not p:
        raise HTTPException(status_code=404, detail={"error": "compute pod not found"})
    if not user.is_admin and p.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    if p.status != "running" or not p.public_ip or not p.ssh_port:
        raise HTTPException(
            status_code=409,
            detail={"error": "pod is not ready for SSH yet", "status": p.status},
        )
    # When the pod was spawned via a kind=runpod provider, serve THAT
    # provider's generated private key — it's the one whose public half we
    # injected into the pod's authorized_keys at create time. Falls back to
    # the gateway's default file-based key for legacy / env-spawned pods.
    private_key = await _resolve_pod_private_key(session, p)
    if private_key is None:
        raise HTTPException(
            status_code=503,
            detail={"error": f"SSH key not found at {_ssh_key_path()} on gateway"},
        )
    # Suggest a filename that hints at the source so users with multiple
    # providers don't overwrite a stash that points at someone else's pods.
    key_hint = f"sgpu-{p.provider_id}" if p.provider_id else "sgpu-runpod"
    cmd = f"ssh -i ~/.ssh/{key_hint} -p {p.ssh_port} {p.ssh_user}@{p.public_ip}"
    return SshInfoResponse(
        ssh_command=cmd,
        ssh_user=p.ssh_user,
        ssh_host=p.public_ip,
        ssh_port=p.ssh_port,
        private_key=private_key,
    )


@router.delete("/{pod_id}")
async def delete_compute(
    pod_id: str,
    user: User = Depends(require_section("compute")),
    session: AsyncSession = Depends(get_session),
):
    p = await session.get(ComputePod, pod_id)
    if not p:
        raise HTTPException(status_code=404, detail={"error": "compute pod not found"})
    if not user.is_admin and p.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"error": "forbidden"})

    runpod_id = p.runpod_pod_id
    pod_provider_id = p.provider_id
    pod_name = p.name
    # Snapshot billing inputs BEFORE we overwrite terminated_at — the audit
    # helper measures cost as (terminated_at or now - ready_at) × rate, and
    # we want it pinned to "the moment of deletion".
    ready_at = p.ready_at
    cost_per_hr = p.cost_per_hr
    p.status = "terminated"
    p.terminated_at = datetime.now(timezone.utc)
    await session.commit()

    # Dispatch teardown by provider kind. NULL provider_id → legacy env-RunPod.
    pod_kind = "runpod"
    if pod_provider_id:
        from .db import Provider
        prov = await session.get(Provider, pod_provider_id)
        if prov is not None:
            pod_kind = prov.kind

    if runpod_id:
        # Fire-and-forget — gateway shouldn't block on the upstream API.
        if pod_kind == "pi":
            asyncio.create_task(_delete_pi(runpod_id, provider_id=pod_provider_id))
        else:
            asyncio.create_task(_delete_runpod(runpod_id, provider_id=pod_provider_id))

    details: dict[str, Any] = {}
    if runpod_id:
        details["runpod_pod_id"] = runpod_id
    cost = audit.cost_breakdown(ready_at, p.terminated_at, cost_per_hr)
    if cost is not None:
        details.update(cost)
    await audit.record(
        user, "compute.delete", "compute", pod_id, pod_name,
        details=details or None,
    )
    return {"ok": True, "id": pod_id}
