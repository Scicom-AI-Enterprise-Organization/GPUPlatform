"""HTTP routes for storage backends.

A "storage" is a reusable destination the platform writes to — where AutoTrain
datasets, benchmark logs, serverless inference logs, etc. get persisted. It is
NOT a single dataset; features reference a storage by id.

Two kinds:
- `s3`          — an S3 (or S3-compatible: R2, MinIO) bucket.
- `huggingface` — a HuggingFace token holder for pushing repos.

Credentials are Fernet-encrypted into `config.credentials_enc` and never
returned to the UI (the record only exposes `has_credentials`). When absent the
runtime falls back to env (AWS_* for s3, HF_TOKEN for huggingface).

Reads are org-wide (any authenticated user) so feature forms can offer the
dropdown. Writes (create / update / delete) are admin-only — these hold shared
credentials and platform-wide infra config, like GPU Providers.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Optional

import boto3
import httpx
from botocore.config import Config as BotoConfig
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from . import audit as audit_module
from . import crypto
from .auth import current_user, require_admin
from .db import Storage, User, get_session

logger = logging.getLogger("gateway.storage")

router = APIRouter(prefix="/v1/storage", tags=["storage"])

SUPPORTED_KINDS = ("s3", "huggingface", "local", "sftp")


# ---------- request / response models ----------------------------------


class CreateStorageRequest(BaseModel):
    name: str
    kind: str  # "s3" | "huggingface" | "local" | "sftp"
    # s3 fields
    bucket: Optional[str] = None
    prefix: Optional[str] = None
    region: Optional[str] = None
    endpoint: Optional[str] = None
    # s3 credentials — both blank → fall back to AWS_* env vars
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    # huggingface credentials — blank → fall back to HF_TOKEN env var
    hf_token: Optional[str] = None
    # Reference a global secret (admin Secrets) by key instead of pasting a token;
    # resolved at use-time. Takes precedence over `hf_token` when set.
    hf_token_secret: Optional[str] = None
    # local fields
    path: Optional[str] = None
    # sftp fields (credentials: password OR private_key)
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    private_key: Optional[str] = None
    base_path: Optional[str] = None
    notes: Optional[str] = None
    enabled: bool = True


class UpdateStorageRequest(BaseModel):
    """All fields optional. Omitted credential fields keep the stored values;
    `enabled` doubles as the toggle. `kind` is immutable."""
    name: Optional[str] = None
    bucket: Optional[str] = None
    prefix: Optional[str] = None
    region: Optional[str] = None
    endpoint: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    hf_token: Optional[str] = None
    hf_token_secret: Optional[str] = None
    path: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    private_key: Optional[str] = None
    base_path: Optional[str] = None
    notes: Optional[str] = None
    enabled: Optional[bool] = None


class TestStorageRequest(BaseModel):
    """Validate connectivity for an unsaved config (the new-storage form calls
    this before letting the user commit). Uses the supplied credentials, or
    falls back to the gateway env exactly like the runtime would."""
    kind: str
    bucket: Optional[str] = None
    region: Optional[str] = None
    endpoint: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    hf_token: Optional[str] = None
    hf_token_secret: Optional[str] = None
    path: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    private_key: Optional[str] = None
    base_path: Optional[str] = None


class TestStorageResponse(BaseModel):
    ok: bool
    message: str


class StorageRecord(BaseModel):
    """Public shape — never includes raw credentials."""
    id: str
    name: str
    kind: str
    bucket: Optional[str] = None
    prefix: Optional[str] = None
    region: Optional[str] = None
    endpoint: Optional[str] = None
    has_credentials: bool = False
    # For huggingface: the global-secret key its token is resolved from (if any).
    hf_token_secret: Optional[str] = None
    # local
    path: Optional[str] = None
    # sftp (non-secret fields only)
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    base_path: Optional[str] = None
    enabled: bool = True
    notes: Optional[str] = None
    created_at: str
    created_by: str


# ---------- helpers -----------------------------------------------------


def _to_record(s: Storage, owner_username: str) -> StorageRecord:
    cfg = s.config or {}
    return StorageRecord(
        id=s.id,
        name=s.name,
        kind=s.kind,
        bucket=cfg.get("bucket"),
        prefix=cfg.get("prefix"),
        region=cfg.get("region"),
        endpoint=cfg.get("endpoint"),
        has_credentials=bool(cfg.get("credentials_enc")) or bool(cfg.get("hf_token_secret")),
        hf_token_secret=cfg.get("hf_token_secret"),
        path=cfg.get("path"),
        host=cfg.get("host"),
        port=cfg.get("port"),
        username=cfg.get("username"),
        base_path=cfg.get("base_path"),
        enabled=bool(s.enabled),
        notes=s.description,
        created_at=s.created_at.isoformat() if s.created_at else "",
        created_by=owner_username,
    )


def _encrypt_sftp_creds(password: Optional[str], private_key: Optional[str]) -> Optional[str]:
    """Encrypt whichever sftp credential was supplied (password or private key).
    Returns None if neither given (keeps an existing blob on update)."""
    pw = (password or "").strip()
    pk = (private_key or "").strip()
    if not pw and not pk:
        return None
    blob: dict = {}
    if pw:
        blob["password"] = pw
    if pk:
        blob["privateKey"] = pk
    return crypto.encrypt(json.dumps(blob))


def _encrypt_s3_creds(access_key_id: Optional[str], secret_access_key: Optional[str]) -> Optional[str]:
    """Return an encrypted blob if a complete key pair was supplied, else None.
    Raises 400 if only one half is given."""
    has_any = bool((access_key_id or "").strip()) or bool((secret_access_key or "").strip())
    if not has_any:
        return None
    if not (access_key_id or "").strip() or not (secret_access_key or "").strip():
        raise HTTPException(
            status_code=400,
            detail="provide both access_key_id and secret_access_key, or leave both blank to use env",
        )
    return crypto.encrypt(json.dumps({
        "accessKeyId": access_key_id.strip(),
        "secretAccessKey": secret_access_key.strip(),
    }))


def _encrypt_hf_token(token: Optional[str]) -> Optional[str]:
    if not (token or "").strip():
        return None
    return crypto.encrypt(json.dumps({"token": token.strip()}))


async def _name_taken(session: AsyncSession, name: str, exclude_id: Optional[str] = None) -> bool:
    result = await session.execute(select(Storage).where(Storage.name == name))
    for row in result.scalars().all():
        if row.id != exclude_id:
            return True
    return False


async def _owner_map(session: AsyncSession, rows: list[Storage]) -> dict[int, str]:
    owner_ids = {s.owner_id for s in rows}
    out: dict[int, str] = {}
    if owner_ids:
        users = await session.execute(select(User).where(User.id.in_(owner_ids)))
        for u in users.scalars().all():
            out[u.id] = u.username
    return out


# ---------- connectivity tests -----------------------------------------


def _test_s3_sync(
    bucket: str,
    region: Optional[str],
    endpoint: Optional[str],
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
) -> None:
    """head_bucket against the target. Raises on any failure (bad creds, wrong
    region, missing bucket, unreachable endpoint). Synchronous — call via
    run_in_threadpool so it doesn't block the event loop."""
    region = (region or os.environ.get("AWS_REGION") or "us-east-1").strip()
    endpoint = (endpoint or "").strip() or None
    kwargs: dict = {
        "region_name": region,
        "config": BotoConfig(
            connect_timeout=5,
            read_timeout=10,
            retries={"max_attempts": 1},
            signature_version="s3v4",
            # Custom endpoints (MinIO, some R2 setups) usually need path-style.
            s3={"addressing_style": "path" if endpoint else "virtual"},
        ),
        # Default to the regional AWS host; a custom endpoint overrides it.
        "endpoint_url": endpoint or f"https://s3.{region}.amazonaws.com",
    }
    akid = (access_key_id or "").strip() or os.environ.get("AWS_ACCESS_KEY_ID", "")
    sak = (secret_access_key or "").strip() or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if akid and sak:
        kwargs["aws_access_key_id"] = akid
        kwargs["aws_secret_access_key"] = sak
    boto3.client("s3", **kwargs).head_bucket(Bucket=bucket)


def _s3_error_message(e: Exception) -> str:
    resp = getattr(e, "response", None)
    if isinstance(resp, dict):
        err = resp.get("Error", {})
        code = err.get("Code")
        msg = err.get("Message")
        # Wrong-region redirects expose the right one in this header.
        region = (resp.get("ResponseMetadata", {}) or {}).get("HTTPHeaders", {}).get("x-amz-bucket-region")
        parts = [p for p in (code, msg) if p]
        base = ": ".join(parts) if parts else str(e)
        if region:
            base += f" (bucket region is {region})"
        return base
    return str(e)


def _test_local_sync(path: str) -> None:
    """Ensure the local path exists (create it) and is writable. Raises on failure."""
    path = os.path.abspath(os.path.expanduser((path or "").strip()))
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(path):
        raise RuntimeError(f"{path} is not a directory")
    probe = os.path.join(path, ".sgpu-write-test")
    with open(probe, "w") as f:
        f.write("ok")
    os.remove(probe)


def _test_sftp_sync(
    host: str, port: Optional[int], username: str, base_path: Optional[str],
    password: Optional[str], private_key: Optional[str],
) -> None:
    """Connect over SFTP and stat the base path. Raises on any failure."""
    from .storage_backends import SFTPBackend
    enc = _encrypt_sftp_creds(password, private_key)
    cfg = {
        "host": host, "port": port or 22, "username": username,
        # rstrip only — keep a leading slash so an absolute base_path stays absolute.
        "base_path": (base_path or "").strip().rstrip("/"),
    }
    if enc:
        cfg["credentials_enc"] = enc
    SFTPBackend(cfg).ping()  # raises StorageError on connect / base-path failure


async def _test_hf(token: Optional[str]) -> tuple[bool, str]:
    token = (token or "").strip() or os.environ.get("HF_TOKEN", "").strip()
    if not token:
        return False, "no token provided and HF_TOKEN env is empty"
    try:
        async with httpx.AsyncClient(timeout=10.0) as cli:
            r = await cli.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
            )
    except httpx.HTTPError as e:
        return False, f"network error: {e}"
    if r.status_code == 200:
        data = r.json()
        who = data.get("name") or data.get("fullname") or "ok"
        return True, f"authenticated as {who}"
    if r.status_code in (401, 403):
        return False, "invalid token"
    return False, f"HTTP {r.status_code}: {r.text[:200]}"


# ---------- endpoints ---------------------------------------------------


@router.get("", response_model=list[StorageRecord])
async def list_storage(
    user: User = Depends(current_user),  # noqa: ARG001 — auth-only; list is org-wide
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Storage).order_by(Storage.name.asc()))
    rows = list(result.scalars().all())
    owners = await _owner_map(session, rows)
    return [_to_record(s, owners.get(s.owner_id, "?")) for s in rows]


@router.post("/test", response_model=TestStorageResponse)
async def test_storage(
    req: TestStorageRequest,
    user: User = Depends(require_admin),  # noqa: ARG001 — admin-only create flow
    session: AsyncSession = Depends(get_session),
):
    if req.kind not in SUPPORTED_KINDS:
        raise HTTPException(status_code=400, detail=f"unsupported kind: {req.kind}")
    if req.kind == "s3":
        bucket = (req.bucket or "").strip()
        if not bucket:
            raise HTTPException(status_code=400, detail="bucket is required to test")
        if bool((req.access_key_id or "").strip()) != bool((req.secret_access_key or "").strip()):
            raise HTTPException(
                status_code=400,
                detail="provide both access_key_id and secret_access_key, or leave both blank",
            )
        try:
            await run_in_threadpool(
                _test_s3_sync, bucket, req.region, req.endpoint,
                req.access_key_id, req.secret_access_key,
            )
        except Exception as e:  # botocore ClientError / endpoint / network
            return TestStorageResponse(ok=False, message=_s3_error_message(e))
        return TestStorageResponse(ok=True, message=f"reached bucket {bucket}")
    if req.kind == "local":
        path = (req.path or "").strip()
        if not path:
            raise HTTPException(status_code=400, detail="path is required to test")
        try:
            await run_in_threadpool(_test_local_sync, path)
        except Exception as e:  # noqa: BLE001
            return TestStorageResponse(ok=False, message=str(e))
        return TestStorageResponse(ok=True, message=f"{path} is writable")
    if req.kind == "sftp":
        host = (req.host or "").strip()
        username = (req.username or "").strip()
        if not host or not username:
            raise HTTPException(status_code=400, detail="host and username are required to test")
        try:
            await run_in_threadpool(
                _test_sftp_sync, host, req.port, username, req.base_path,
                req.password, req.private_key,
            )
        except Exception as e:  # noqa: BLE001
            return TestStorageResponse(ok=False, message=str(e))
        return TestStorageResponse(ok=True, message=f"reached {username}@{host}")
    # huggingface — resolve a global-secret reference to its value before testing.
    token = req.hf_token
    ref = (req.hf_token_secret or "").strip()
    if ref:
        from .global_env_api import load_global_env
        token = (await load_global_env(session)).get(ref)
        if not token:
            return TestStorageResponse(ok=False, message=f"global secret '{ref}' is not set")
    ok, msg = await _test_hf(token)
    return TestStorageResponse(ok=ok, message=msg)


@router.post("", response_model=StorageRecord)
async def create_storage(
    req: CreateStorageRequest,
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    if req.kind not in SUPPORTED_KINDS:
        raise HTTPException(status_code=400, detail=f"unsupported kind: {req.kind}")
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if await _name_taken(session, name):
        raise HTTPException(status_code=400, detail=f"a storage named '{name}' already exists")

    config: dict
    if req.kind == "s3":
        bucket = (req.bucket or "").strip()
        if not bucket:
            raise HTTPException(status_code=400, detail="bucket is required for kind=s3")
        config = {
            "bucket": bucket,
            "prefix": (req.prefix or "").strip() or None,
            "region": (req.region or "").strip() or None,
            "endpoint": (req.endpoint or "").strip() or None,
        }
        enc = _encrypt_s3_creds(req.access_key_id, req.secret_access_key)
        if enc:
            config["credentials_enc"] = enc
    elif req.kind == "local":
        path = (req.path or "").strip()
        if not path:
            raise HTTPException(status_code=400, detail="path is required for kind=local")
        config = {"path": path}
    elif req.kind == "sftp":
        host = (req.host or "").strip()
        username = (req.username or "").strip()
        if not host or not username:
            raise HTTPException(status_code=400, detail="host and username are required for kind=sftp")
        config = {
            "host": host,
            "port": int(req.port or 22),
            "username": username,
            "base_path": (req.base_path or "").strip().rstrip("/"),
        }
        enc = _encrypt_sftp_creds(req.password, req.private_key)
        if enc:
            config["credentials_enc"] = enc
    else:  # huggingface
        config = {}
        ref = (req.hf_token_secret or "").strip()
        if ref:
            config["hf_token_secret"] = ref  # resolve from global secrets at use-time
        else:
            enc = _encrypt_hf_token(req.hf_token)
            if enc:
                config["credentials_enc"] = enc

    sid = f"store-{secrets.token_hex(4)}"
    row = Storage(
        id=sid,
        owner_id=user.id,
        name=name,
        kind=req.kind,
        description=(req.notes or "").strip() or None,
        enabled=req.enabled,
        config=config,
        created_at=datetime.now(timezone.utc),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)

    await audit_module.record(
        user, "storage.create", "storage", sid, name,
        details={"kind": req.kind},
    )
    logger.info("created storage %s (%s) for user=%s", sid, req.kind, user.username)
    return _to_record(row, user.username)


@router.patch("/{storage_id}", response_model=StorageRecord)
async def update_storage(
    storage_id: str,
    req: UpdateStorageRequest,
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    row = await session.get(Storage, storage_id)
    if row is None:
        raise HTTPException(status_code=404, detail="storage not found")

    cfg = dict(row.config or {})

    if req.name is not None:
        new_name = req.name.strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="name cannot be blank")
        if new_name != row.name and await _name_taken(session, new_name, exclude_id=storage_id):
            raise HTTPException(status_code=400, detail=f"a storage named '{new_name}' already exists")
        row.name = new_name

    if req.notes is not None:
        row.description = req.notes.strip() or None

    if req.enabled is not None:
        row.enabled = req.enabled

    if row.kind == "s3":
        if req.bucket is not None:
            b = req.bucket.strip()
            if not b:
                raise HTTPException(status_code=400, detail="bucket cannot be blank for s3")
            cfg["bucket"] = b
        if req.prefix is not None:
            cfg["prefix"] = req.prefix.strip() or None
        if req.region is not None:
            cfg["region"] = req.region.strip() or None
        if req.endpoint is not None:
            cfg["endpoint"] = req.endpoint.strip() or None
        enc = _encrypt_s3_creds(req.access_key_id, req.secret_access_key)
        if enc:  # only replace when new creds supplied; omit keeps existing
            cfg["credentials_enc"] = enc
    elif row.kind == "local":
        if req.path is not None:
            p = req.path.strip()
            if not p:
                raise HTTPException(status_code=400, detail="path cannot be blank for local")
            cfg["path"] = p
    elif row.kind == "sftp":
        if req.host is not None:
            h = req.host.strip()
            if not h:
                raise HTTPException(status_code=400, detail="host cannot be blank for sftp")
            cfg["host"] = h
        if req.port is not None:
            cfg["port"] = int(req.port)
        if req.username is not None:
            u = req.username.strip()
            if not u:
                raise HTTPException(status_code=400, detail="username cannot be blank for sftp")
            cfg["username"] = u
        if req.base_path is not None:
            cfg["base_path"] = req.base_path.strip().rstrip("/")
        enc = _encrypt_sftp_creds(req.password, req.private_key)
        if enc:  # only replace when new creds supplied
            cfg["credentials_enc"] = enc
    else:  # huggingface
        # Switching to a global-secret reference clears any stored token, and
        # vice-versa. Omitting both keeps whatever's there.
        if req.hf_token_secret is not None:
            ref = req.hf_token_secret.strip()
            if ref:
                cfg["hf_token_secret"] = ref
                cfg.pop("credentials_enc", None)
            else:
                cfg.pop("hf_token_secret", None)
        enc = _encrypt_hf_token(req.hf_token)
        if enc:
            cfg["credentials_enc"] = enc
            cfg.pop("hf_token_secret", None)

    row.config = cfg
    flag_modified(row, "config")
    await session.commit()
    await session.refresh(row)

    owners = await _owner_map(session, [row])
    await audit_module.record(
        user, "storage.update", "storage", storage_id, row.name,
        details={"kind": row.kind},
    )
    return _to_record(row, owners.get(row.owner_id, user.username))


@router.delete("/{storage_id}")
async def delete_storage(
    storage_id: str,
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    row = await session.get(Storage, storage_id)
    if row is None:
        raise HTTPException(status_code=404, detail="storage not found")
    name = row.name
    kind = row.kind
    await session.delete(row)
    await session.commit()
    await audit_module.record(
        user, "storage.delete", "storage", storage_id, name,
        details={"kind": kind},
    )
    return {"ok": True, "id": storage_id}
