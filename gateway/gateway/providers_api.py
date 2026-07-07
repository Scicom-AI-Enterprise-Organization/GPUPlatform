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
from .compute import PI_GPU_TYPES, RUNPOD_GPU_TYPES, GpuTypeOption
from .db import Provider, User, get_session
from .provider import cloud_providers_disabled
from .vm_probe import availability_vm, bandwidth_vm, kill_pid_vm, metrics_vm, probe_vm

logger = logging.getLogger("gateway.providers")

router = APIRouter(prefix="/v1/providers", tags=["providers"])

VM_DEFAULT_PORT = 22
SUPPORTED_KINDS = ("vm", "runpod", "pi")
API_KEY_KINDS = ("runpod", "pi")


class VmConfig(BaseModel):
    host: str
    port: int = VM_DEFAULT_PORT
    user: str = "root"
    # Full PEM body. Key OR password required on create; on update the client
    # may omit both to keep the existing credentials.
    private_key: Optional[str] = None
    password: Optional[str] = None
    # Optional jump host (ProxyJump) for boxes not directly reachable — e.g.
    # the TM Huawei NPU machines behind ssh.tma01.gpu.tm.com.my. Jump auth is
    # its own key or password.
    jump_host: Optional[str] = None
    jump_port: int = 22
    jump_user: Optional[str] = None
    jump_private_key: Optional[str] = None
    jump_password: Optional[str] = None


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
    jump_host: Optional[str] = None  # set when the VM is reached via ProxyJump
    gpus: Optional[list[str]] = None
    gpu_count: Optional[int] = None
    # Cloud (runpod/pi): the catalog of GPU types this provider can provision, so
    # API consumers (e.g. the labeling platform discovering where to run autotrain)
    # see what's available. VM providers list their fixed physical GPUs in `gpus`.
    available_gpus: Optional[list[GpuTypeOption]] = None
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


class GpuProcInfo(BaseModel):
    pid: int          # container-namespace pid (what ps/kill see on the box)
    comm: str
    cmd: str
    gpu_mem_mib: int = 0   # this process's VRAM on the GPU (from nvidia-smi)
    gpus: str = ""         # host procs: GPU device indices it has open (if readable)


class GpuMetricInfo(BaseModel):
    index: int
    name: str
    util_pct: int
    mem_used_mib: int
    mem_total_mib: int
    temp_c: int
    pcie_gen_cur: int = 0
    pcie_width_cur: int = 0
    pcie_gen_max: int = 0
    pcie_width_max: int = 0
    nvlink_supported: bool = False
    nvlink_active: int = 0
    nvlink_gbps: float = 0.0
    # Huawei Ascend (npu-smi): kind="npu", util_pct=AICore%, mem=HBM.
    kind: str = "gpu"
    power_w: float = 0.0
    health: str = ""
    processes: list[GpuProcInfo] = []


class DiskInfo(BaseModel):
    mount: str
    used_bytes: int
    total_bytes: int


class ProviderMetricsResponse(BaseModel):
    """Live VM host metrics for the provider metrics page (polled, not stored)."""
    ok: bool
    message: str
    cpu_pct: float = -1.0          # overall CPU busy %, -1 if unavailable
    cpu_cores: list[float] = []    # per-core busy % (htop-style)
    mem_used_mib: int = 0
    mem_total_mib: int = 0
    gpus: list[GpuMetricInfo] = []
    disks: list[DiskInfo] = []     # real filesystems (df), largest first
    # GPU processes found via /proc (command + container pid) that NVML can't map to
    # a GPU here (host / other-container tenants on a shared box). Surfaced host-level.
    host_gpu_procs: list[GpuProcInfo] = []
    checked_at: float


class ProviderBandwidthResponse(BaseModel):
    """On-demand disk / memory / CPU bandwidth benchmark (button-triggered)."""
    ok: bool
    message: str
    disk_write_mbps: float = 0.0
    disk_read_mbps: float = 0.0
    mem_mbps: float = 0.0
    cpu_model: str = ""
    cpu_mhz: float = 0.0
    checked_at: float


# Cloud providers don't have fixed installed GPUs — they expose a catalog of
# selectable types. Reuse the same lists the new-pod / benchmark forms read from
# (/compute/{kind}/gpu-types) so a single source defines what's available.
_GPU_CATALOG_BY_KIND: dict[str, list[dict]] = {
    "runpod": RUNPOD_GPU_TYPES,
    "pi": PI_GPU_TYPES,
}


def _to_record(p: Provider, owner_username: str) -> ProviderRecord:
    cfg = p.config or {}
    api_key_last4: Optional[str] = None
    if p.kind in API_KEY_KINDS and cfg.get("api_key_enc"):
        try:
            api_key_last4 = crypto.decrypt(cfg["api_key_enc"])[-4:]
        except Exception:
            api_key_last4 = None
    catalog = _GPU_CATALOG_BY_KIND.get(p.kind)
    available_gpus = [GpuTypeOption(**g) for g in catalog] if catalog else None
    return ProviderRecord(
        id=p.id,
        name=p.name,
        kind=p.kind,
        created_at=p.created_at.isoformat() if p.created_at else "",
        created_by=owner_username,
        host=cfg.get("host"),
        port=cfg.get("port"),
        user=cfg.get("user"),
        jump_host=cfg.get("jump_host"),
        gpus=cfg.get("gpus"),
        gpu_count=cfg.get("gpu_count"),
        available_gpus=available_gpus,
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
    if vm.jump_host and vm.jump_host.strip():
        if vm.jump_port < 1 or vm.jump_port > 65535:
            raise HTTPException(status_code=400, detail="vm.jump_port must be 1..65535")
        if not ((vm.jump_private_key and vm.jump_private_key.strip()) or vm.jump_password):
            raise HTTPException(status_code=400, detail="jump host needs a private key or a password")
    return vm


def _vm_has_credentials(vm: VmConfig) -> bool:
    return bool((vm.private_key and vm.private_key.strip()) or vm.password)


def _vm_config_dict(vm: VmConfig) -> dict:
    """Provider.config for a validated VmConfig — secrets Fernet-encrypted."""
    config: dict = {
        "host": vm.host.strip(),
        "port": int(vm.port),
        "user": vm.user.strip(),
    }
    if vm.private_key and vm.private_key.strip():
        config["private_key_enc"] = crypto.encrypt(vm.private_key)
    if vm.password:
        config["password_enc"] = crypto.encrypt(vm.password)
    if vm.jump_host and vm.jump_host.strip():
        config["jump_host"] = vm.jump_host.strip()
        config["jump_port"] = int(vm.jump_port)
        config["jump_user"] = (vm.jump_user or "root").strip()
        if vm.jump_private_key and vm.jump_private_key.strip():
            config["jump_private_key_enc"] = crypto.encrypt(vm.jump_private_key)
        if vm.jump_password:
            config["jump_password_enc"] = crypto.encrypt(vm.jump_password)
    return config


def _vm_conn_from_cfg(cfg: dict) -> dict:
    """Decrypt a stored VM provider config into kwargs for the vm_probe
    functions (probe/availability/metrics/bandwidth/kill)."""
    if not cfg.get("private_key_enc") and not cfg.get("password_enc"):
        raise HTTPException(status_code=500, detail="provider missing stored credentials")
    conn: dict = {
        "host": cfg.get("host", ""),
        "port": int(cfg.get("port") or VM_DEFAULT_PORT),
        "user": cfg.get("user", "root"),
        "private_key": crypto.decrypt(cfg["private_key_enc"]) if cfg.get("private_key_enc") else None,
        "password": crypto.decrypt(cfg["password_enc"]) if cfg.get("password_enc") else None,
    }
    if cfg.get("jump_host"):
        conn.update(
            jump_host=cfg["jump_host"],
            jump_port=int(cfg.get("jump_port") or 22),
            jump_user=cfg.get("jump_user") or "root",
            jump_private_key=crypto.decrypt(cfg["jump_private_key_enc"]) if cfg.get("jump_private_key_enc") else None,
            jump_password=crypto.decrypt(cfg["jump_password_enc"]) if cfg.get("jump_password_enc") else None,
        )
    return conn


def _vm_conn_from_inline(vm: VmConfig) -> dict:
    """Kwargs for the vm_probe functions from an unsaved form config."""
    return {
        "host": vm.host.strip(),
        "port": int(vm.port),
        "user": vm.user.strip(),
        "private_key": vm.private_key,
        "password": vm.password,
        "jump_host": (vm.jump_host or "").strip() or None,
        "jump_port": int(vm.jump_port or 22),
        "jump_user": (vm.jump_user or "root").strip(),
        "jump_private_key": vm.jump_private_key,
        "jump_password": vm.jump_password,
    }


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


async def _runpod_balance(api_key: str) -> tuple[Optional[float], str]:
    """RunPod account credit in USD via the GraphQL `myself.clientBalance` — the
    REST /v1 API has no balance route. Returns (balance, message); balance is None
    on any error (network/auth/parse) and the message says why."""
    base = os.environ.get("RUNPOD_GRAPHQL_BASE", "https://api.runpod.io/graphql").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as cli:
            r = await cli.post(
                base,
                params={"api_key": api_key},
                json={"query": "query { myself { clientBalance } }"},
            )
    except httpx.HTTPError as e:
        return None, f"network error: {e}"
    if r.status_code in (401, 403):
        return None, "unauthorized"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    try:
        bal = r.json()["data"]["myself"]["clientBalance"]
    except (ValueError, KeyError, TypeError):
        return None, "unexpected response"
    try:
        return (float(bal) if bal is not None else None), "ok"
    except (TypeError, ValueError):
        return None, "non-numeric balance"


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
    if req.kind in API_KEY_KINDS and cloud_providers_disabled():
        raise HTTPException(
            status_code=403,
            detail="cloud GPU providers (RunPod / Prime Intellect) are disabled on this deployment",
        )
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="name is required")

    pid = f"prov-{secrets.token_hex(4)}"
    config: dict
    if req.kind == "vm":
        vm = _validate_vm(req.vm)
        if not _vm_has_credentials(vm):
            raise HTTPException(status_code=400, detail="vm.private_key or vm.password is required")
        config = _vm_config_dict(vm)
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
    if req.kind in API_KEY_KINDS and cloud_providers_disabled():
        raise HTTPException(
            status_code=403,
            detail="cloud GPU providers (RunPod / Prime Intellect) are disabled on this deployment",
        )

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
        conn = _vm_conn_from_cfg(row.config or {})
    else:
        vm = _validate_vm(req.vm)
        if not _vm_has_credentials(vm):
            raise HTTPException(status_code=400, detail="vm.private_key or vm.password required for test")
        conn = _vm_conn_from_inline(vm)

    result = await probe_vm(**conn)

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
    result = await availability_vm(**_vm_conn_from_cfg(row.config or {}))
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


class ProviderBalanceResponse(BaseModel):
    ok: bool
    balance: Optional[float] = None
    currency: str = "USD"
    message: str = "ok"


@router.get("/{provider_id}/balance", response_model=ProviderBalanceResponse)
async def provider_balance(
    provider_id: str,
    user: User = Depends(current_user),  # noqa: ARG001 — auth-only; any signed-in user
    session: AsyncSession = Depends(get_session),
):
    """RunPod account credit (USD). Surfaced on the providers page card + the
    create-endpoint form so you can see the selected account's budget before
    spawning pods. Read-only; never returns the key."""
    row = await session.get(Provider, provider_id)
    if row is None:
        raise HTTPException(status_code=404, detail="provider not found")
    if row.kind != "runpod":
        raise HTTPException(status_code=400, detail="balance is only available for RunPod providers")
    enc = (row.config or {}).get("api_key_enc")
    if not enc:
        raise HTTPException(status_code=500, detail="provider missing stored key")
    bal, msg = await _runpod_balance(crypto.decrypt(enc))
    return ProviderBalanceResponse(ok=bal is not None, balance=bal, message=msg)


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
    result = await metrics_vm(**_vm_conn_from_cfg(row.config or {}))
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
            pcie_gen_cur=g.pcie_gen_cur, pcie_width_cur=g.pcie_width_cur,
            pcie_gen_max=g.pcie_gen_max, pcie_width_max=g.pcie_width_max,
            nvlink_supported=g.nvlink_supported, nvlink_active=g.nvlink_active,
            nvlink_gbps=g.nvlink_gbps,
            kind=g.kind, power_w=g.power_w, health=g.health,
            processes=[GpuProcInfo(pid=p.pid, comm=p.comm, cmd=p.cmd, gpu_mem_mib=p.gpu_mem_mib) for p in g.processes],
        ) for g in result.gpus],
        disks=[DiskInfo(mount=d.mount, used_bytes=d.used_bytes, total_bytes=d.total_bytes)
               for d in result.disks],
        host_gpu_procs=[GpuProcInfo(pid=p.pid, comm=p.comm, cmd=p.cmd, gpus=p.gpus)
                        for p in result.host_procs],
        checked_at=result.checked_at,
    )


class KillPidRequest(BaseModel):
    pid: int
    sig: int = 9  # SIGKILL by default — the metrics "Terminate" button frees a stuck GPU


class KillPidResponse(BaseModel):
    ok: bool
    message: str


@router.post("/{provider_id}/kill-pid", response_model=KillPidResponse)
async def provider_kill_pid(
    provider_id: str,
    req: KillPidRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    """Kill a process by pid on a VM provider over SSH — the metrics page "Terminate"
    button, to free a GPU held by a stuck/orphaned process. Owner or admin only.
    Reports the real kill outcome (a pid in another container's PID namespace — the
    orphaned-GPU case — isn't visible here and reports "No such process")."""
    row = await session.get(Provider, provider_id)
    if row is None:
        raise HTTPException(status_code=404, detail="provider not found")
    if row.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="not your provider")
    if row.kind != "vm":
        raise HTTPException(status_code=400, detail="kill-pid is only available for VM providers")
    if req.pid <= 1:
        raise HTTPException(status_code=400, detail=f"refusing to kill pid {req.pid}")
    result = await kill_pid_vm(
        **_vm_conn_from_cfg(row.config or {}),
        pid=req.pid,
        sig=req.sig,
    )
    await audit_module.record(
        user, "provider.kill-pid", "provider", provider_id, row.name,
        details={"pid": req.pid, "sig": req.sig, "ok": result.ok, "message": result.message},
    )
    return KillPidResponse(ok=result.ok, message=result.message)


@router.get("/{provider_id}/bandwidth", response_model=ProviderBandwidthResponse)
async def provider_bandwidth(
    provider_id: str,
    user: User = Depends(current_user),  # noqa: ARG001 — auth-only
    session: AsyncSession = Depends(get_session),
):
    """On-demand disk/memory/CPU bandwidth benchmark for a VM provider, over SSH.
    Heavier than the live metrics poll (writes a ~512 MiB temp file) — the metrics
    page triggers this from a button, not the polling loop."""
    row = await session.get(Provider, provider_id)
    if row is None:
        raise HTTPException(status_code=404, detail="provider not found")
    if row.kind != "vm":
        raise HTTPException(status_code=400, detail="bandwidth test is only available for VM providers")
    result = await bandwidth_vm(**_vm_conn_from_cfg(row.config or {}))
    return ProviderBandwidthResponse(
        ok=result.ok,
        message=result.message,
        disk_write_mbps=result.disk_write_mbps,
        disk_read_mbps=result.disk_read_mbps,
        mem_mbps=result.mem_mbps,
        cpu_model=result.cpu_model,
        cpu_mhz=result.cpu_mhz,
        checked_at=result.checked_at,
    )
