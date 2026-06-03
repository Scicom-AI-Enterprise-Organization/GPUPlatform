"""HTTP routes for user-registered cloud providers.

Three kinds today:
- `vm`   — bare-metal SSH, user uploads a PEM.
- `runpod` / `pi` — cloud accounts, user pastes an API key. Gateway validates
  it with a cheap GET and auto-generates an ed25519 keypair so spawned pods
  can be SSH'd later without a manual upload step.
"""
from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Optional

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import audit as audit_module
from . import crypto
from .auth import current_user, require_admin
from .db import Provider, User, get_session
from .vm_probe import availability_vm, metrics_vm, probe_vm

logger = logging.getLogger("gateway.providers")

router = APIRouter(prefix="/v1/providers", tags=["providers"])

VM_DEFAULT_PORT = 22
SUPPORTED_KINDS = ("vm", "runpod", "pi")
API_KEY_KINDS = ("runpod", "pi")


class VmConfig(BaseModel):
    host: str
    port: int = VM_DEFAULT_PORT
    user: str = "root"
    # Full PEM body. Required on create; on update the client may omit it to
    # keep the existing key — handled by `CreateProviderRequest` validation.
    private_key: Optional[str] = None


class ApiKeyConfig(BaseModel):
    api_key: Optional[str] = None


class CreateProviderRequest(BaseModel):
    name: str
    kind: str  # "vm" | "runpod" | "pi"
    vm: Optional[VmConfig] = None
    api: Optional[ApiKeyConfig] = None


class TestProviderRequest(BaseModel):
    """Test against an arbitrary config without persisting it. The frontend
    calls this from the new-provider form so users can verify SSH (vm) or
    the API key (runpod/pi) before they commit to saving the row.

    Alternately, callers may pass `provider_id` to test an already-saved
    provider — useful for the list page's per-row "Re-test" button.
    """
    kind: str
    vm: Optional[VmConfig] = None
    api: Optional[ApiKeyConfig] = None
    provider_id: Optional[str] = None


class ProviderRecord(BaseModel):
    """Public shape. Never includes the private key body or the raw API key."""
    id: str
    name: str
    kind: str
    created_at: str
    created_by: str
    # VM-specific summary; absent for other kinds.
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    gpus: Optional[list[str]] = None
    gpu_count: Optional[int] = None
    # API-key-kind summary; absent for vm.
    api_key_last4: Optional[str] = None
    ssh_pub: Optional[str] = None
    validated_at: Optional[str] = None
    account_email: Optional[str] = None


class TestProviderResponse(BaseModel):
    ok: bool
    message: str
    gpus: list[str] = []
    gpu_count: int = 0


class GpuLiveInfo(BaseModel):
    index: int
    name: str
    mem_free_mib: int
    mem_total_mib: int
    util_pct: int


class AvailabilityResponse(BaseModel):
    ok: bool
    message: str
    gpus: list[GpuLiveInfo] = []
    checked_at: float


class GpuMetricInfo(BaseModel):
    index: int
    name: str
    util_pct: int
    mem_used_mib: int
    mem_total_mib: int
    temp_c: int


class ProviderMetricsResponse(BaseModel):
    """Live VM host metrics for the provider metrics page (polled, not stored)."""
    ok: bool
    message: str
    cpu_pct: float = -1.0          # overall CPU busy %, -1 if unavailable
    cpu_cores: list[float] = []    # per-core busy % (htop-style)
    mem_used_mib: int = 0
    mem_total_mib: int = 0
    gpus: list[GpuMetricInfo] = []
    checked_at: float


def _to_record(p: Provider, owner_username: str) -> ProviderRecord:
    cfg = p.config or {}
    api_key_last4: Optional[str] = None
    if p.kind in API_KEY_KINDS and cfg.get("api_key_enc"):
        try:
            api_key_last4 = crypto.decrypt(cfg["api_key_enc"])[-4:]
        except Exception:
            api_key_last4 = None
    return ProviderRecord(
        id=p.id,
        name=p.name,
        kind=p.kind,
        created_at=p.created_at.isoformat() if p.created_at else "",
        created_by=owner_username,
        host=cfg.get("host"),
        port=cfg.get("port"),
        user=cfg.get("user"),
        gpus=cfg.get("gpus"),
        gpu_count=cfg.get("gpu_count"),
        api_key_last4=api_key_last4,
        ssh_pub=cfg.get("ssh_pub"),
        validated_at=cfg.get("validated_at"),
        account_email=cfg.get("account_email"),
    )


def _validate_vm(vm: Optional[VmConfig]) -> VmConfig:
    if vm is None:
        raise HTTPException(status_code=400, detail="vm config required for kind=vm")
    if not vm.host.strip():
        raise HTTPException(status_code=400, detail="vm.host is required")
    if vm.port < 1 or vm.port > 65535:
        raise HTTPException(status_code=400, detail="vm.port must be 1..65535")
    if not vm.user.strip():
        raise HTTPException(status_code=400, detail="vm.user is required")
    return vm


def _gen_ssh_keypair(label: str) -> tuple[str, str]:
    """Return (public_openssh, private_openssh_pem). Used so api-key providers
    have an SSH key available for spawned pods without forcing the user to
    upload one."""
    sk = Ed25519PrivateKey.generate()
    priv = sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub_raw = sk.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode()
    return f"{pub_raw} {label}", priv


async def _runpod_validate(api_key: str) -> tuple[bool, str, dict]:
    """Cheap GET that succeeds on any valid RunPod key. Lists 1 pod —
    always authorised regardless of whether the account has any pods."""
    base = os.environ.get("RUNPOD_API_BASE", "https://rest.runpod.io/v1").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as cli:
            r = await cli.get(
                f"{base}/pods",
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.HTTPError as e:
        return False, f"network error: {e}", {}
    if r.status_code == 200:
        return True, "ok", {}
    if r.status_code in (401, 403):
        return False, "unauthorized", {}
    return False, f"HTTP {r.status_code}: {r.text[:200]}", {}


async def _pi_validate(api_key: str) -> tuple[bool, str, dict]:
    """Cheap GET against Prime Intellect — list pods with limit=1."""
    base = os.environ.get("PI_API_BASE", "https://api.primeintellect.ai").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as cli:
            r = await cli.get(
                f"{base}/api/v1/pods/",
                headers={"Authorization": f"Bearer {api_key}"},
                params={"limit": 1, "offset": 0},
            )
    except httpx.HTTPError as e:
        return False, f"network error: {e}", {}
    if r.status_code == 200:
        return True, "ok", {}
    if r.status_code in (401, 403):
        return False, "unauthorized", {}
    return False, f"HTTP {r.status_code}: {r.text[:200]}", {}


async def _validate_api_key(kind: str, api_key: str) -> tuple[bool, str, dict]:
    if kind == "runpod":
        return await _runpod_validate(api_key)
    if kind == "pi":
        return await _pi_validate(api_key)
    raise HTTPException(status_code=400, detail=f"kind {kind} not an api-key kind")


async def _pi_upload_ssh_key(api_key: str, label: str, public_key: str) -> str:
    """Register a public key on the Prime Intellect account and return its id.

    PI's pod-create requires an `sshKeyId` referencing an account-level key —
    there's no inline pub-key field on the pod object. We upload once at
    provider-create time so every compute pod created with this provider can
    pass the stored id without an extra round-trip."""
    base = os.environ.get("PI_API_BASE", "https://api.primeintellect.ai").rstrip("/")
    async with httpx.AsyncClient(timeout=15.0) as cli:
        r = await cli.post(
            f"{base}/api/v1/ssh_keys/",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"name": label, "publicKey": public_key},
        )
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"PI ssh key upload failed: HTTP {r.status_code}: {r.text[:200]}",
        )
    data = r.json()
    key_id = data.get("id") or (data.get("data") or {}).get("id")
    if not key_id:
        raise HTTPException(
            status_code=502,
            detail=f"PI ssh key upload returned no id: {str(data)[:200]}",
        )
    return key_id


@router.get("", response_model=list[ProviderRecord])
async def list_providers(
    user: User = Depends(current_user),  # noqa: ARG001 — auth-only; list is org-wide
    session: AsyncSession = Depends(get_session),
):
    """All providers are visible to every authenticated user so the resource
    forms (benchmark, serverless) can show the dropdown. Writes (POST/DELETE)
    require admin — see those handlers."""
    # Join to users so we can populate `created_by` without a per-row lookup.
    from sqlalchemy.orm import selectinload
    result = await session.execute(
        select(Provider).order_by(Provider.created_at.desc())
    )
    rows = list(result.scalars().all())
    # Resolve owner usernames in one extra query.
    owner_ids = {p.owner_id for p in rows}
    owner_map: dict[int, str] = {}
    if owner_ids:
        from .db import User as _User
        users = await session.execute(select(_User).where(_User.id.in_(owner_ids)))
        for u in users.scalars().all():
            owner_map[u.id] = u.username
    return [_to_record(p, owner_map.get(p.owner_id, "?")) for p in rows]


@router.post("", response_model=ProviderRecord)
async def create_provider(
    req: CreateProviderRequest,
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    if req.kind not in SUPPORTED_KINDS:
        raise HTTPException(status_code=400, detail=f"unsupported kind: {req.kind}")
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="name is required")

    pid = f"prov-{secrets.token_hex(4)}"
    config: dict
    if req.kind == "vm":
        vm = _validate_vm(req.vm)
        if not vm.private_key or not vm.private_key.strip():
            raise HTTPException(status_code=400, detail="vm.private_key is required")
        config = {
            "host": vm.host.strip(),
            "port": int(vm.port),
            "user": vm.user.strip(),
            "private_key_enc": crypto.encrypt(vm.private_key),
        }
    elif req.kind in API_KEY_KINDS:
        api = req.api or ApiKeyConfig()
        if not api.api_key or not api.api_key.strip():
            raise HTTPException(status_code=400, detail="api.api_key is required")
        key = api.api_key.strip()
        ok, msg, account = await _validate_api_key(req.kind, key)
        if not ok:
            raise HTTPException(status_code=400, detail=f"{req.kind} key invalid: {msg}")
        ssh_pub, ssh_priv = _gen_ssh_keypair(label=f"gateway@{pid}")
        config = {
            "api_key_enc": crypto.encrypt(key),
            "ssh_pub": ssh_pub,
            "ssh_priv_enc": crypto.encrypt(ssh_priv),
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "account_email": account.get("email") if isinstance(account, dict) else None,
        }
        if req.kind == "pi":
            # PI requires an account-registered key referenced by id on
            # pod-create. Upload now so compute create stays single-round-trip.
            pi_key_id = await _pi_upload_ssh_key(key, f"sgpu-{pid}", ssh_pub)
            config["pi_ssh_key_id"] = pi_key_id
    else:  # pragma: no cover — guarded above
        raise HTTPException(status_code=400, detail=f"kind {req.kind} not implemented")

    row = Provider(
        id=pid,
        owner_id=user.id,
        name=req.name.strip(),
        kind=req.kind,
        config=config,
        created_at=datetime.now(timezone.utc),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)

    await audit_module.record(
        user, "provider.create", "provider", pid, req.name,
        details={"kind": req.kind},
    )
    logger.info("created provider %s (%s) for user=%s", pid, req.kind, user.username)
    return _to_record(row, user.username)


@router.delete("/{provider_id}")
async def delete_provider(
    provider_id: str,
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    row = await session.get(Provider, provider_id)
    if row is None:
        raise HTTPException(status_code=404, detail="provider not found")
    name = row.name
    kind = row.kind
    await session.delete(row)
    await session.commit()
    await audit_module.record(
        user, "provider.delete", "provider", provider_id, name,
        details={"kind": kind},
    )
    return {"ok": True, "id": provider_id}


@router.post("/test", response_model=TestProviderResponse)
async def test_provider(
    req: TestProviderRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    if req.kind not in SUPPORTED_KINDS:
        raise HTTPException(status_code=400, detail=f"unsupported kind: {req.kind}")

    # ---- API-key kinds: cheap HTTP probe, no SSH ----
    if req.kind in API_KEY_KINDS:
        if req.provider_id:
            row = await session.get(Provider, req.provider_id)
            if row is None:
                raise HTTPException(status_code=404, detail="provider not found")
            if row.kind != req.kind:
                raise HTTPException(status_code=400, detail="kind mismatch")
            enc = (row.config or {}).get("api_key_enc")
            if not enc:
                raise HTTPException(status_code=500, detail="provider missing stored key")
            api_key = crypto.decrypt(enc)
        else:
            api = req.api or ApiKeyConfig()
            if not api.api_key or not api.api_key.strip():
                raise HTTPException(status_code=400, detail="api.api_key required for test")
            api_key = api.api_key.strip()
        ok, msg, _ = await _validate_api_key(req.kind, api_key)
        if req.provider_id and ok:
            row = await session.get(Provider, req.provider_id)
            if row is not None:
                cfg = dict(row.config or {})
                cfg["validated_at"] = datetime.now(timezone.utc).isoformat()
                row.config = cfg
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(row, "config")
                await session.commit()
        return TestProviderResponse(ok=ok, message=msg, gpus=[], gpu_count=0)

    # ---- VM: SSH probe (existing path) ----
    # Resolve config: either inline (new-provider form) or from a saved row.
    if req.provider_id:
        row = await session.get(Provider, req.provider_id)
        if row is None:
            raise HTTPException(status_code=404, detail="provider not found")
        if row.kind != req.kind:
            raise HTTPException(status_code=400, detail="kind mismatch")
        cfg = row.config or {}
        host = cfg.get("host", "")
        port = int(cfg.get("port") or VM_DEFAULT_PORT)
        ssh_user = cfg.get("user", "root")
        enc = cfg.get("private_key_enc")
        if not enc:
            raise HTTPException(status_code=500, detail="provider missing stored key")
        private_key = crypto.decrypt(enc)
    else:
        vm = _validate_vm(req.vm)
        if not vm.private_key or not vm.private_key.strip():
            raise HTTPException(status_code=400, detail="vm.private_key required for test")
        host = vm.host.strip()
        port = int(vm.port)
        ssh_user = vm.user.strip()
        private_key = vm.private_key

    result = await probe_vm(host=host, port=port, user=ssh_user, private_key=private_key)

    # On a saved provider, persist the probe result so the list view can show
    # the GPU summary without re-running SSH on every page load.
    if req.provider_id and result.ok:
        row = await session.get(Provider, req.provider_id)
        if row is not None:
            cfg = dict(row.config or {})
            cfg["gpus"] = result.gpus
            cfg["gpu_count"] = result.gpu_count
            row.config = cfg
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(row, "config")
            await session.commit()

    return TestProviderResponse(
        ok=result.ok,
        message=result.message,
        gpus=result.gpus,
        gpu_count=result.gpu_count,
    )


@router.get("/{provider_id}/availability", response_model=AvailabilityResponse)
async def provider_availability(
    provider_id: str,
    user: User = Depends(current_user),  # noqa: ARG001 — auth-only; open to all
    session: AsyncSession = Depends(get_session),
):
    """Live SSH probe — returns per-GPU memory + utilisation. Used by the
    benchmark form to surface availability the same way RunPod's API check
    does for cloud runs. Open to all authenticated users so non-admins can
    still see whether a provider has free capacity before picking it."""
    row = await session.get(Provider, provider_id)
    if row is None:
        raise HTTPException(status_code=404, detail="provider not found")
    if row.kind in API_KEY_KINDS:
        enc = (row.config or {}).get("api_key_enc")
        if not enc:
            raise HTTPException(status_code=500, detail="provider missing stored key")
        ok, msg, _ = await _validate_api_key(row.kind, crypto.decrypt(enc))
        return AvailabilityResponse(
            ok=ok, message=msg, gpus=[], checked_at=datetime.now(timezone.utc).timestamp(),
        )
    if row.kind != "vm":
        raise HTTPException(status_code=400, detail="availability check not supported for kind={}".format(row.kind))
    cfg = row.config or {}
    enc = cfg.get("private_key_enc")
    if not enc:
        raise HTTPException(status_code=500, detail="provider missing stored key")
    private_key = crypto.decrypt(enc)
    result = await availability_vm(
        host=cfg.get("host", ""),
        port=int(cfg.get("port") or VM_DEFAULT_PORT),
        user=cfg.get("user", "root"),
        private_key=private_key,
    )
    return AvailabilityResponse(
        ok=result.ok,
        message=result.message,
        gpus=[GpuLiveInfo(
            index=g.index, name=g.name,
            mem_free_mib=g.mem_free_mib, mem_total_mib=g.mem_total_mib,
            util_pct=g.util_pct,
        ) for g in result.gpus],
        checked_at=result.checked_at,
    )


@router.get("/{provider_id}/metrics", response_model=ProviderMetricsResponse)
async def provider_metrics(
    provider_id: str,
    user: User = Depends(current_user),  # noqa: ARG001 — auth-only
    session: AsyncSession = Depends(get_session),
):
    """Live host metrics (CPU% + memory + per-GPU util/mem/temp) for a VM provider,
    over SSH. Not stored — the metrics page polls this and graphs it client-side."""
    row = await session.get(Provider, provider_id)
    if row is None:
        raise HTTPException(status_code=404, detail="provider not found")
    if row.kind != "vm":
        raise HTTPException(status_code=400, detail="metrics are only available for VM providers")
    cfg = row.config or {}
    enc = cfg.get("private_key_enc")
    if not enc:
        raise HTTPException(status_code=500, detail="provider missing stored key")
    result = await metrics_vm(
        host=cfg.get("host", ""),
        port=int(cfg.get("port") or VM_DEFAULT_PORT),
        user=cfg.get("user", "root"),
        private_key=crypto.decrypt(enc),
    )
    return ProviderMetricsResponse(
        ok=result.ok,
        message=result.message,
        cpu_pct=result.cpu_pct,
        cpu_cores=result.cpu_cores,
        mem_used_mib=result.mem_used_mib,
        mem_total_mib=result.mem_total_mib,
        gpus=[GpuMetricInfo(
            index=g.index, name=g.name, util_pct=g.util_pct,
            mem_used_mib=g.mem_used_mib, mem_total_mib=g.mem_total_mib, temp_c=g.temp_c,
        ) for g in result.gpus],
        checked_at=result.checked_at,
    )
