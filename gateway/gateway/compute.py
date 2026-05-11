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
from .auth import require_admin, require_section
from .db import Base, User, get_session, session_factory
from .runpod_provider import _map_gpu

logger = logging.getLogger("gateway.compute")

# Default image — RunPod's pytorch image has sshd preconfigured and respects
# the account-level SSH key for authorized_keys injection.
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 600  # 10min — RunPod cold pulls of pytorch images can be slow

# Curated template list. Every image here has both sshd and JupyterLab
# pre-baked — `JUPYTER_PASSWORD` env triggers `start.sh` to launch jupyter
# on port 8888, which we proxy via RunPod's per-pod proxy domain. We
# deliberately keep the list small so the "JupyterLab always-on" promise
# holds; a free-text image field would let users pick something that
# doesn't ship Jupyter and the URL would 404.
CURATED_TEMPLATES: list[dict[str, str]] = [
    {
        "id": "pytorch-2.4-cuda12.4",
        "name": "PyTorch 2.4 + CUDA 12.4",
        "image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "description": "PyTorch 2.4, Python 3.11, CUDA 12.4. Default for most workloads.",
    },
    {
        "id": "pytorch-2.1-cuda11.8",
        "name": "PyTorch 2.1 + CUDA 11.8",
        "image": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "description": "Older CUDA for libraries pinned to 11.x (e.g. some Triton kernels).",
    },
    {
        "id": "tensorflow-latest",
        "name": "TensorFlow",
        "image": "runpod/tensorflow:latest",
        "description": "Latest TensorFlow GPU build with JupyterLab.",
    },
    {
        "id": "cuda-12.4-base",
        "name": "CUDA 12.4 (bare)",
        "image": "runpod/base:0.5.0-cuda12.4.1",
        "description": "Minimal CUDA-only image — bring your own framework via pip.",
    },
]


def _resolve_template(template_id: str) -> Optional[dict[str, str]]:
    for t in CURATED_TEMPLATES:
        if t["id"] == template_id:
            return t
    return None


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
    # creating | running | failed | terminated | pending_approval | rejected
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


# ---------- RunPod REST helpers ----------------------------------------


def _api_base() -> str:
    return os.environ.get("RUNPOD_API_BASE", "https://rest.runpod.io/v1").rstrip("/")


def _api_key() -> str:
    k = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not k:
        raise RuntimeError("RUNPOD_API_KEY not set — Compute requires a real RunPod account")
    return k


def _name_prefix() -> str:
    return os.environ.get("COMPUTE_NAME_PREFIX", "sgpu-compute")


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=_api_base(),
        headers={
            "Authorization": f"Bearer {_api_key()}",
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


# ---------- Provision flow ---------------------------------------------


async def _create_pod(pod_id: str) -> None:
    """Background task: POST /pods, then poll until SSH coords land."""
    async with session_factory()() as s:
        row = await s.get(ComputePod, pod_id)
        if row is None:
            return

    # JupyterLab password is generated up front so we can persist it on the
    # row before the pod actually exists — that way if the create succeeds
    # but persistence races, we don't lose the only copy.
    jupyter_password = secrets.token_urlsafe(18)
    async with session_factory()() as s:
        r2 = await s.get(ComputePod, pod_id)
        if r2 is None:
            return
        r2.jupyter_password = jupyter_password
        await s.commit()

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
        "env": {"JUPYTER_PASSWORD": jupyter_password},
    }

    async with _client() as cli:
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

        # Poll for SSH readiness.
        deadline = time.time() + POLL_TIMEOUT_S
        while time.time() < deadline:
            await asyncio.sleep(POLL_INTERVAL_S)

            # Bail if the row was deleted (user cancelled) — we don't want to
            # keep polling a pod that nobody owns anymore.
            async with session_factory()() as s:
                row = await s.get(ComputePod, pod_id)
                if row is None:
                    try:
                        await cli.delete(f"/pods/{runpod_id}")
                    except Exception:
                        pass
                    return
                if row.status == "terminated":
                    return  # delete handler is doing its thing

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
                    await s.commit()
                logger.info("compute %s: SSH ready at %s:%s", pod_id, ip, port)
                return

        # Timed out waiting for SSH.
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


async def _delete_runpod(runpod_id: str) -> None:
    async with _client() as cli:
        try:
            r = await cli.delete(f"/pods/{runpod_id}")
            if r.status_code >= 400 and r.status_code != 404:
                logger.warning(
                    "compute: delete %s returned %s %s",
                    runpod_id, r.status_code, r.text[:200],
                )
        except Exception:
            logger.exception("compute: delete request for %s crashed", runpod_id)


# ---------- Startup hook -----------------------------------------------


async def cleanup_orphaned_running() -> int:
    """Mark `creating` rows from previous gateway processes as failed.

    Pods that were `running` are real and stay running on RunPod — no
    reason to disturb them. But a `creating` row means the polling task
    died with the gateway, and the user needs to know.
    """
    async with session_factory()() as s:
        rows = await s.execute(
            update(ComputePod)
            .where(ComputePod.status == "creating")
            .values(
                status="failed",
                error_text="orphaned by gateway restart — check RunPod dashboard for any pod still billing",
                terminated_at=datetime.now(timezone.utc),
            )
            .returning(ComputePod.id)
        )
        ids = [r[0] for r in rows.all()]
        await s.commit()
    return len(ids)


# ---------- Pydantic schemas -------------------------------------------


class CreateComputeRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    gpu_type: str = Field(min_length=1, max_length=64)
    gpu_count: int = Field(default=1, ge=1, le=8)
    container_disk_gb: int = Field(default=40, ge=10, le=2000)
    volume_gb: int = Field(default=0, ge=0, le=2000)
    template_id: str = Field(default="pytorch-2.4-cuda12.4")
    cloud_type: str = Field(default="COMMUNITY", pattern=r"^(COMMUNITY|SECURE)$")


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
    created_by: str
    created_at: str
    ready_at: Optional[str] = None
    terminated_at: Optional[str] = None


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


class RejectRequest(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=1024)


# ---------- HTTP API ---------------------------------------------------


router = APIRouter(prefix="/compute", tags=["compute"])


def _to_record(p: ComputePod, owner_username: str) -> ComputeRecord:
    # Embed the token in the URL so the user clicks once and is logged in.
    # Keeping `jupyter_password` in the response too is harmless — it's the
    # same token, just exposed separately for advanced uses (e.g. reusing in
    # a different client). The UI hides the password field by default.
    jurl = (
        _jupyter_url(p.runpod_pod_id, p.jupyter_password)
        if p.runpod_pod_id
        else None
    )
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
        created_by=owner_username,
        created_at=p.created_at.isoformat() if p.created_at else "",
        ready_at=p.ready_at.isoformat() if p.ready_at else None,
        terminated_at=p.terminated_at.isoformat() if p.terminated_at else None,
    )


def _gen_id() -> str:
    return f"cmp-{uuid.uuid4().hex[:8]}"


# Templates route is defined BEFORE /{pod_id} so the literal `/templates`
# isn't swallowed by the path-param matcher. Same trick bench.py uses.


@router.get("/templates", response_model=list[TemplateRecord])
async def list_templates(_: User = Depends(require_section("compute"))):
    return [TemplateRecord(**t) for t in CURATED_TEMPLATES]


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
    if not os.environ.get("RUNPOD_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail={"error": "RUNPOD_API_KEY not configured on gateway"},
        )

    tpl = _resolve_template(body.template_id)
    if tpl is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"unknown template_id '{body.template_id}'",
                "available": [t["id"] for t in CURATED_TEMPLATES],
            },
        )

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
        image=tpl["image"],
        template_id=tpl["id"],
        cloud_type=body.cloud_type,
        status="pending_approval" if needs_approval else "creating",
        owner_id=user.id,
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
            "template_id": tpl["id"], "cloud_type": body.cloud_type,
        },
    )

    row = await session.get(ComputePod, pod_id)
    return _to_record(row, user.username)


async def _safe_create(pod_id: str) -> None:
    try:
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
    key_path = _ssh_key_path()
    try:
        private_key = Path(key_path).read_text()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail={"error": f"SSH key not found at {key_path} on gateway"},
        )
    cmd = f"ssh -i ~/.ssh/sgpu-runpod -p {p.ssh_port} {p.ssh_user}@{p.public_ip}"
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
    pod_name = p.name
    p.status = "terminated"
    p.terminated_at = datetime.now(timezone.utc)
    await session.commit()

    if runpod_id:
        # Fire-and-forget — gateway shouldn't block on RunPod's API.
        asyncio.create_task(_delete_runpod(runpod_id))

    await audit.record(
        user, "compute.delete", "compute", pod_id, pod_name,
        details={"runpod_pod_id": runpod_id} if runpod_id else None,
    )
    return {"ok": True, "id": pod_id}
