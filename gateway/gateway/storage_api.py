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

Reads of the RECORDS are org-wide (any authenticated user) so feature forms can
offer the dropdown. Writes (create / update / delete) are admin-only — these hold
shared credentials and platform-wide infra config, like GPU Providers. Reads of
the bucket's CONTENTS (usage, cleanup, the file viewer) are admin-only too: one
storage row backs every feature's data, so browsing it is an infra operation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import posixpath
import re
import secrets
from datetime import datetime, timezone
from typing import Optional

import boto3
import httpx
from botocore.config import Config as BotoConfig
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from . import audit as audit_module
from . import crypto
from .auth import current_user, require_admin
from .db import Storage, User, get_session, session_factory

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
    # s3 credentials by reference: a global-secret (admin Secrets) key resolved at
    # use-time, instead of a pasted key. Take precedence over the pasted values.
    access_key_id_secret: Optional[str] = None
    secret_access_key_secret: Optional[str] = None
    # huggingface credentials — blank → fall back to HF_TOKEN env var
    hf_token: Optional[str] = None
    # Reference a global secret (admin Secrets) by key instead of pasting a token;
    # resolved at use-time. Takes precedence over `hf_token` when set.
    hf_token_secret: Optional[str] = None
    # huggingface: a custom Hub endpoint (HF_ENDPOINT) — blank → huggingface.co.
    # Either a literal URL (`endpoint`) or a global-secret key (`endpoint_secret`,
    # resolved at use-time, takes precedence). Note: `endpoint` is reused by s3.
    endpoint_secret: Optional[str] = None
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
    access_key_id_secret: Optional[str] = None
    secret_access_key_secret: Optional[str] = None
    hf_token: Optional[str] = None
    hf_token_secret: Optional[str] = None
    endpoint_secret: Optional[str] = None
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
    access_key_id_secret: Optional[str] = None
    secret_access_key_secret: Optional[str] = None
    hf_token: Optional[str] = None
    hf_token_secret: Optional[str] = None
    endpoint_secret: Optional[str] = None
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
    # For huggingface: the global-secret key its custom HF_ENDPOINT resolves from.
    endpoint_secret: Optional[str] = None
    # For s3: the global-secret keys its credentials resolve from (if any).
    access_key_id_secret: Optional[str] = None
    secret_access_key_secret: Optional[str] = None
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
    # Cached storage usage (s3 only) — computed on demand (usage/scan), NOT per list
    # request (a full-bucket walk is O(objects)). Null until first computed.
    total_size_bytes: Optional[int] = None
    object_count: Optional[int] = None
    size_computed_at: Optional[str] = None
    # True while a background cleanup is deleting from this storage (s3 only).
    purge_running: bool = False


# ---------- helpers -----------------------------------------------------


def _to_record(s: Storage, owner_username: str) -> StorageRecord:
    cfg = s.config or {}
    usage = cfg.get("usage") or {}
    return StorageRecord(
        id=s.id,
        name=s.name,
        kind=s.kind,
        bucket=cfg.get("bucket"),
        prefix=cfg.get("prefix"),
        region=cfg.get("region"),
        endpoint=cfg.get("endpoint"),
        total_size_bytes=usage.get("bytes"),
        object_count=usage.get("objects"),
        size_computed_at=usage.get("computed_at"),
        purge_running=(_PURGE_JOBS.get(s.id, {}).get("state") == "running"),
        has_credentials=(
            bool(cfg.get("credentials_enc"))
            or bool(cfg.get("hf_token_secret"))
            or bool(cfg.get("access_key_id_secret"))
            or bool(cfg.get("secret_access_key_secret"))
        ),
        hf_token_secret=cfg.get("hf_token_secret"),
        endpoint_secret=cfg.get("endpoint_secret"),
        access_key_id_secret=cfg.get("access_key_id_secret"),
        secret_access_key_secret=cfg.get("secret_access_key_secret"),
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


async def _test_hf(token: Optional[str], endpoint: Optional[str] = None) -> tuple[bool, str]:
    token = (token or "").strip() or os.environ.get("HF_TOKEN", "").strip()
    if not token:
        return False, "no token provided and HF_TOKEN env is empty"
    base = (endpoint or "").strip().rstrip("/") or "https://huggingface.co"
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as cli:
            r = await cli.get(
                f"{base}/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
            )
    except httpx.HTTPError as e:
        return False, f"network error: {e}"
    if r.status_code == 200:
        data = r.json()
        who = data.get("name") or data.get("fullname") or "ok"
        where = "" if base == "https://huggingface.co" else f" at {base}"
        return True, f"authenticated as {who}{where}"
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
        # Resolve credentials: a global-secret reference (resolved here) takes
        # precedence over a pasted key, mirroring runtime (_resolve_s3_creds).
        akid = (req.access_key_id or "").strip()
        sak = (req.secret_access_key or "").strip()
        ak_ref = (req.access_key_id_secret or "").strip()
        sk_ref = (req.secret_access_key_secret or "").strip()
        if ak_ref or sk_ref:
            from .global_env_api import load_global_env
            ge = await load_global_env(session)
            if ak_ref:
                akid = (ge.get(ak_ref) or "").strip()
                if not akid:
                    return TestStorageResponse(ok=False, message=f"global secret '{ak_ref}' is not set")
            if sk_ref:
                sak = (ge.get(sk_ref) or "").strip()
                if not sak:
                    return TestStorageResponse(ok=False, message=f"global secret '{sk_ref}' is not set")
        if bool(akid) != bool(sak):
            raise HTTPException(
                status_code=400,
                detail="provide both access_key_id and secret_access_key, or leave both blank",
            )
        try:
            await run_in_threadpool(
                _test_s3_sync, bucket, req.region, req.endpoint,
                akid or None, sak or None,
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
    # huggingface — resolve global-secret references (token + custom endpoint) to
    # their values before testing.
    token = req.hf_token
    endpoint = (req.endpoint or "").strip() or None
    ref = (req.hf_token_secret or "").strip()
    ep_ref = (req.endpoint_secret or "").strip()
    if ref or ep_ref:
        from .global_env_api import load_global_env
        ge = await load_global_env(session)
        if ref:
            token = ge.get(ref)
            if not token:
                return TestStorageResponse(ok=False, message=f"global secret '{ref}' is not set")
        if ep_ref:
            endpoint = (ge.get(ep_ref) or "").strip() or None
            if not endpoint:
                return TestStorageResponse(ok=False, message=f"global secret '{ep_ref}' is not set")
    ok, msg = await _test_hf(token, endpoint)
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
        ak_ref = (req.access_key_id_secret or "").strip()
        sk_ref = (req.secret_access_key_secret or "").strip()
        if ak_ref or sk_ref:
            # Credentials by global-secret reference — resolved at use-time.
            if ak_ref:
                config["access_key_id_secret"] = ak_ref
            if sk_ref:
                config["secret_access_key_secret"] = sk_ref
        else:
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
        # Custom HF_ENDPOINT (blank → huggingface.co): a global-secret reference
        # takes precedence over a pasted URL.
        ep_ref = (req.endpoint_secret or "").strip()
        if ep_ref:
            config["endpoint_secret"] = ep_ref
        elif (req.endpoint or "").strip():
            config["endpoint"] = req.endpoint.strip()

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
        # Credentials by global-secret reference: a non-None field updates it
        # (empty string clears it); setting any ref drops the pasted blob.
        if req.access_key_id_secret is not None:
            if req.access_key_id_secret.strip():
                cfg["access_key_id_secret"] = req.access_key_id_secret.strip()
            else:
                cfg.pop("access_key_id_secret", None)
        if req.secret_access_key_secret is not None:
            if req.secret_access_key_secret.strip():
                cfg["secret_access_key_secret"] = req.secret_access_key_secret.strip()
            else:
                cfg.pop("secret_access_key_secret", None)
        if cfg.get("access_key_id_secret") or cfg.get("secret_access_key_secret"):
            cfg.pop("credentials_enc", None)
        enc = _encrypt_s3_creds(req.access_key_id, req.secret_access_key)
        if enc:  # pasted creds replace the blob AND clear any references
            cfg["credentials_enc"] = enc
            cfg.pop("access_key_id_secret", None)
            cfg.pop("secret_access_key_secret", None)
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
        # Custom HF_ENDPOINT: a global-secret reference and a literal URL are
        # mutually exclusive; setting one clears the other. Empty string clears.
        if req.endpoint_secret is not None:
            if req.endpoint_secret.strip():
                cfg["endpoint_secret"] = req.endpoint_secret.strip()
                cfg.pop("endpoint", None)
            else:
                cfg.pop("endpoint_secret", None)
        if req.endpoint is not None:
            if req.endpoint.strip():
                cfg["endpoint"] = req.endpoint.strip()
                cfg.pop("endpoint_secret", None)
            else:
                cfg.pop("endpoint", None)

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


# ---------- usage + cleanup (s3 only) -----------------------------------
# Computing total size or scanning for junk means a full-bucket LIST (O(objects)),
# so it's on-demand + cached in config["usage"], never inline on the list route.

DEFAULT_PURGE_AGE_DAYS = 30


class PurgeScanRequest(BaseModel):
    max_age_days: Optional[int] = None  # ephemeral-dir age cutoff; None → 30, 0 → no age rule


class PurgeRequest(BaseModel):
    prefixes: list[str]                 # exactly the group prefixes the scan proposed
    max_age_days: Optional[int] = None  # re-validated at delete time with the same rule


def _s3_for_storage(s: Storage):
    """(S3Target, normalized base prefix) for a kind=s3 storage — absolute keys are
    passed to the s3 helpers, so `prefix_root` is irrelevant here."""
    from . import bench
    target = bench._target_from_storage_row(s)
    base = ((s.config or {}).get("prefix") or "").strip().strip("/")
    return target, (f"{base}/" if base else "")


async def _live_owner_ids(session: AsyncSession) -> dict[str, set[str]]:
    """The set of ids still alive per owner-kind, for orphan detection. Ids are
    globally unique, so membership (not storage_id) decides ownership."""
    from .db import Dataset, App
    from .bench import Benchmark
    from .training_api import TrainingRun
    from .quantization_api import QuantizationJob

    async def ids(col):
        return set((await session.execute(select(col))).scalars().all())

    return {
        "dataset": await ids(Dataset.id),
        "app": await ids(App.app_id),  # serverless-logs/<app_id>/… — App PK is app_id
        "benchmark": await ids(Benchmark.id),
        "training_run": await ids(TrainingRun.id),
        "quant_job": await ids(QuantizationJob.id),
    }


async def _live_repo_prefixes(session: AsyncSession, storage_id: str) -> list[str]:
    """Key prefixes of live catalog (HF-mirror) repos on this storage — protected."""
    from .db import CatalogRepo
    rows = (await session.execute(
        select(CatalogRepo.prefix).where(CatalogRepo.storage_id == storage_id)
    )).scalars().all()
    return [p for p in rows if p]


async def _require_s3_storage(session: AsyncSession, storage_id: str) -> Storage:
    s = await session.get(Storage, storage_id)
    if s is None:
        raise HTTPException(status_code=404, detail="storage not found")
    if s.kind != "s3":
        raise HTTPException(status_code=400, detail="usage/cleanup is only available for s3 storage")
    return s


async def _cache_usage(session: AsyncSession, s: Storage, total_bytes: int, total_objects: int) -> None:
    cfg = dict(s.config or {})
    cfg["usage"] = {
        "bytes": int(total_bytes),
        "objects": int(total_objects),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    s.config = cfg
    flag_modified(s, "config")
    await session.commit()


@router.get("/{storage_id}/usage")
async def storage_usage(
    storage_id: str,
    refresh: bool = False,
    user: User = Depends(require_admin),  # noqa: ARG001 — a full-bucket walk is an infra op
    session: AsyncSession = Depends(get_session),
):
    """Total size + object count for an s3 storage. Returns the cached value unless
    `refresh=true`, which re-walks the bucket (O(objects)) and re-caches it."""
    s = await _require_s3_storage(session, storage_id)
    cfg = s.config or {}
    if not refresh and cfg.get("usage"):
        return {"storage_id": storage_id, "cached": True, **cfg["usage"]}
    target, base = _s3_for_storage(s)
    from . import bench
    objs = await run_in_threadpool(bench.s3_list, base, target)
    total_bytes = sum(int(o.get("size") or 0) for o in objs)
    await _cache_usage(session, s, total_bytes, len(objs))
    return {"storage_id": storage_id, "cached": False,
            "bytes": total_bytes, "objects": len(objs),
            "computed_at": (s.config or {}).get("usage", {}).get("computed_at")}


async def _scan_storage(session: AsyncSession, s: Storage, max_age_days: Optional[int]) -> dict:
    """List the bucket once, classify (orphan + aged), cache usage. Shared by the
    dry-run scan and the delete re-validation so both see identical classification."""
    from . import bench
    from . import storage_purge
    from datetime import timedelta
    target, base = _s3_for_storage(s)
    objs = await run_in_threadpool(bench.s3_list, base, target)
    live = await _live_owner_ids(session)
    repos = await _live_repo_prefixes(session, s.id)
    days = DEFAULT_PURGE_AGE_DAYS if max_age_days is None else int(max_age_days)
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat() if days > 0 else None
    result = storage_purge.categorize(
        objs, base=base, live_ids=live, repo_prefixes=repos, cutoff_iso=cutoff_iso,
    )
    await _cache_usage(session, s, result["total_bytes"], result["total_objects"])
    result["age_days"] = days
    return result


@router.post("/{storage_id}/purge-scan")
async def storage_purge_scan(
    storage_id: str,
    req: PurgeScanRequest,
    user: User = Depends(require_admin),  # noqa: ARG001
    session: AsyncSession = Depends(get_session),
):
    """DRY RUN — list what cleanup WOULD delete (orphaned + aged), grouped, with
    reclaimable bytes. Deletes nothing. Also refreshes the cached total size."""
    s = await _require_s3_storage(session, storage_id)
    result = await _scan_storage(session, s, req.max_age_days)
    return {"storage_id": storage_id, **result}


# Deleting hundreds of GB / 100k+ objects takes minutes, so purge runs as a
# BACKGROUND task and the UI polls progress — the request returns as soon as the
# job is launched. One job per storage at a time; state is in-memory (a manual
# admin op, re-runnable, so it's not persisted like training runs).
_PURGE_JOBS: dict[str, dict] = {}
_PURGE_TASKS: set = set()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_public(job: dict) -> dict:
    """The status shape the UI polls (drops internal bookkeeping keys)."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


async def _run_purge(storage_id: str, job: dict, todo: list[dict], target) -> None:
    """Delete each group's prefix, updating `job` progress after every batch /
    prefix. Refreshes the cached usage at the end. Never raises to the caller —
    failures land in job['state']='error'."""
    from . import bench
    try:
        for g in todo:
            prefix = g["prefix"]
            n = await run_in_threadpool(
                bench.s3_delete_prefix, prefix, target,
                lambda k: job.__setitem__("deleted_objects", job["deleted_objects"] + k),
            )
            job["freed_bytes"] += g["bytes"]
            job["done_prefixes"] += 1
            job["deleted"].append({"prefix": prefix, "objects": n, "bytes": g["bytes"]})
        job["state"] = "done"
    except Exception as e:  # noqa: BLE001 — surface to the poller, don't crash the loop
        job["state"] = "error"
        job["error"] = str(e)
        logger.warning("storage purge %s failed: %s", storage_id, e)
    finally:
        job["finished_at"] = _now_iso()
        try:
            async with session_factory()() as sess:
                s = await sess.get(Storage, storage_id)
                if s is not None:
                    await _cache_usage(
                        sess, s,
                        max(0, job["_scan_bytes"] - job["freed_bytes"]),
                        max(0, job["_scan_objects"] - job["deleted_objects"]),
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning("storage purge %s usage refresh failed: %s", storage_id, e)


@router.post("/{storage_id}/purge")
async def storage_purge_delete(
    storage_id: str,
    req: PurgeRequest,
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    """Launch a background delete of the confirmed prefixes. RE-VALIDATES each
    against a fresh scan first (a prefix that became live is skipped, never
    deleted), then returns immediately with a running job; poll `/purge-status`."""
    s = await _require_s3_storage(session, storage_id)
    requested = {p.strip() for p in (req.prefixes or []) if p and p.strip()}
    if not requested:
        raise HTTPException(status_code=400, detail="no prefixes to delete")
    running = _PURGE_JOBS.get(storage_id)
    if running and running.get("state") == "running":
        raise HTTPException(status_code=409, detail="a cleanup is already running for this storage")
    scan = await _scan_storage(session, s, req.max_age_days)
    purgeable = {g["prefix"]: g for g in scan["groups"] if g["purgeable"]}
    todo = [purgeable[p] for p in requested if p in purgeable]
    skipped = [{"prefix": p, "reason": "no longer purgeable (now live or gone)"}
               for p in requested if p not in purgeable]
    target, _base = _s3_for_storage(s)
    job = _PURGE_JOBS[storage_id] = {
        "storage_id": storage_id, "job_id": "purge-" + secrets.token_hex(6),
        "state": "running", "started_at": _now_iso(), "finished_at": None,
        "total_prefixes": len(todo), "done_prefixes": 0,
        "target_objects": sum(g["objects"] for g in todo),
        "target_bytes": sum(g["bytes"] for g in todo),
        "deleted_objects": 0, "freed_bytes": 0,
        "deleted": [], "skipped": skipped, "error": None,
        "_scan_bytes": scan["total_bytes"], "_scan_objects": scan["total_objects"],
    }
    task = asyncio.create_task(_run_purge(storage_id, job, todo, target))
    _PURGE_TASKS.add(task)
    task.add_done_callback(_PURGE_TASKS.discard)
    await audit_module.record(
        user, "storage.purge", "storage", storage_id, s.name,
        details={"prefixes": len(todo), "target_objects": job["target_objects"],
                 "target_bytes": job["target_bytes"]},
    )
    return _job_public(job)


# ---------- file browser (s3 + local) -----------------------------------
# A read-only viewer over the backing store: one delimited LIST / one readdir per
# directory (O(page), NOT the full-bucket walk usage/purge-scan do), plus a capped
# read of a single object. Admin-only, like the other raw-bucket routes — a
# storage row is shared infra and its objects are every feature's data.
#
# ⚠ Everything here must stay page-bounded: a single directory with a million
# entries is normal in these buckets (per-clip audio), so no route may build a
# whole-directory list in memory or sort one.

BROWSABLE_KINDS = ("s3", "local")
# Served THROUGH the gateway (same-origin + authed, so it works regardless of
# bucket CORS). Anything bigger is a presigned direct-from-S3 download instead.
MAX_INLINE_BYTES = 25 * 1024 * 1024
# Default head-read for the text preview pane.
DEFAULT_PREVIEW_BYTES = 1024 * 1024
BROWSE_PAGE_MAX = 1000
# A local directory is name-sorted only up to this many entries — above it we
# page in readdir order instead. Sorting means materializing every name, and a
# million-entry directory is exactly where that stops being free.
LOCAL_SORT_MAX = 20_000
# S3 counts every key it SCANS against MaxKeys, so a delimited page can come back
# with zero entries (all keys collapsed into already-returned folders) while more
# remain. Pull a few pages before handing the UI an empty one.
S3_EMPTY_PAGE_RETRIES = 5

# Extensions mimetypes doesn't know (or gets wrong) for the kinds of files this
# platform writes.
_EXTRA_MEDIA_TYPES = {
    ".jsonl": "application/x-ndjson",
    ".ndjson": "application/x-ndjson",
    ".log": "text/plain; charset=utf-8",
    ".yaml": "text/yaml; charset=utf-8",
    ".yml": "text/yaml; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".tsv": "text/tab-separated-values; charset=utf-8",
    ".py": "text/plain; charset=utf-8",
    ".sh": "text/plain; charset=utf-8",
    ".toml": "text/plain; charset=utf-8",
    ".cfg": "text/plain; charset=utf-8",
    ".ini": "text/plain; charset=utf-8",
    ".pcm": "audio/L16",
}
# A filename safe to put in a Content-Disposition header (quotes/CR/LF stripped).
_UNSAFE_FILENAME = re.compile(r'[\r\n"\\]')


class StorageEntry(BaseModel):
    """One row in a directory listing. `path` is relative to the storage's own
    configured prefix — that's what every other route here takes back."""
    name: str
    path: str
    kind: str  # "folder" | "file"
    size: Optional[int] = None
    modified: Optional[str] = None


class StorageBrowseResponse(BaseModel):
    storage_id: str
    kind: str  # "s3" | "local"
    # s3: the bucket. local: "" (the root IS the filesystem path).
    bucket: str
    # s3: the storage's configured prefix ("" = whole bucket).
    # local: the absolute filesystem root.
    root: str
    # Directory being listed, relative to root ("" = root itself).
    path: str
    entries: list[StorageEntry]
    # Opaque continuation token (S3's for s3, an offset for local) — pass back as
    # `token` for the next page.
    next_token: Optional[str] = None
    # Something the user should know about THIS listing (e.g. a directory too big
    # to sort). Rendered as a hint line, not an error.
    note: Optional[str] = None


class StorageObjectUrlResponse(BaseModel):
    url: str
    expires_in: int


def _storage_root(s: Storage) -> str:
    return ((s.config or {}).get("prefix") or "").strip().strip("/")


def _safe_rel(path: Optional[str]) -> str:
    """Normalize a browser-supplied path to a clean relative one, or 400.

    Rejects `..` outright rather than resolving it: the storage's own prefix is a
    scope boundary (a viewer must not read a sibling tenant's objects in a shared
    bucket), so escaping it is never a legitimate request.
    """
    rel = (path or "").strip().lstrip("/")
    if not rel:
        return ""
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise HTTPException(status_code=400, detail="invalid path")
    return "/".join(parts)


def _abs_key(root: str, rel: str) -> str:
    return posixpath.join(root, rel) if root else rel


def _media_type_for(name: str) -> Optional[str]:
    """Media type from the FILENAME, or None when the extension is unknown."""
    ext = os.path.splitext(name.lower())[1]
    if ext in _EXTRA_MEDIA_TYPES:
        return _EXTRA_MEDIA_TYPES[ext]
    guessed, _enc = mimetypes.guess_type(name)
    if not guessed:
        return None
    return f"{guessed}; charset=utf-8" if guessed.startswith("text/") else guessed


def _is_textual(media_type: str) -> bool:
    mt = media_type.split(";", 1)[0].strip()
    return (
        mt.startswith("text/")
        or mt in ("application/json", "application/x-ndjson", "application/xml",
                  "application/javascript", "application/x-yaml")
        or mt.endswith("+json")
    )


def _looks_textual(data: bytes) -> bool:
    """Sniff a head-read for text. Extensionless files are everywhere in these
    buckets (`.persist_marker`, `metadata`, `README`) and S3 stores them as
    octet-stream, so without this they'd all be unpreviewable."""
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError as e:
        # A ranged read can split a multi-byte codepoint — only the last few
        # bytes are allowed to be an incomplete sequence.
        return e.start >= len(data) - 4


async def _require_browsable_storage(session: AsyncSession, storage_id: str) -> Storage:
    s = await session.get(Storage, storage_id)
    if s is None:
        raise HTTPException(status_code=404, detail="storage not found")
    if s.kind not in BROWSABLE_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"the file viewer supports s3 and local storage (this one is {s.kind})",
        )
    return s


# ---------- local filesystem backing ------------------------------------


def _local_root(s: Storage) -> str:
    path = ((s.config or {}).get("path") or "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="local storage has no path configured")
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def _local_path(root: str, rel: str) -> str:
    """Resolve a path inside a local root, or 400.

    ⚠ realpath BOTH sides. `_safe_rel` already rejects `..`, but a *symlink*
    inside the root pointing at `/etc` would sail through a string check — this
    is what actually confines the viewer to the configured directory.
    """
    target = os.path.realpath(os.path.join(root, rel)) if rel else root
    if target != root and os.path.commonpath([root, target]) != root:
        raise HTTPException(status_code=400, detail="path escapes the storage root")
    return target


def _escapes_root(root: str, full_path: str) -> bool:
    """True when `full_path` is a symlink leading OUT of the storage root.

    Only symlinks pay the realpath — an ordinary child of the directory cannot
    escape. Such entries are hidden from listings (not just refused on open) so
    the viewer shows exactly the configured folder and nothing reachable from it.
    """
    if not os.path.islink(full_path):
        return False
    try:
        real = os.path.realpath(full_path)
        return real != root and os.path.commonpath([root, real]) != root
    except ValueError:  # different drives — treat as outside
        return True


def _local_entry(de: os.DirEntry, rel: str) -> StorageEntry:
    is_dir = de.is_dir(follow_symlinks=True)
    size: Optional[int] = None
    modified: Optional[str] = None
    try:  # a broken symlink / racing unlink still lists, just without stats
        st = de.stat(follow_symlinks=True)
        size = None if is_dir else st.st_size
        modified = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()
    except OSError:
        pass
    return StorageEntry(
        name=de.name,
        path=f"{rel}/{de.name}" if rel else de.name,
        kind="folder" if is_dir else "file",
        size=size,
        modified=modified,
    )


def _local_browse(
    root: str, rel: str, offset: int, limit: int, q: str,
) -> tuple[list[StorageEntry], Optional[str], Optional[str]]:
    """One page of a local directory → (entries, next_token, note).

    ⚠ Page-bounded on purpose. Sorting a directory means materializing every
    name, so that only happens under `LOCAL_SORT_MAX`; a bigger directory pages
    in **readdir order** (stable for an unchanged directory) and only the page's
    entries are stat()ed. `q` is a name-prefix filter applied during the scan —
    same semantics as the S3 side, where a prefix is all the API can do.
    """
    d = _local_path(root, rel)
    if not os.path.isdir(d):
        raise HTTPException(
            status_code=404,
            detail=f"{d} is not a directory on the gateway host",
        )

    # Names only (no stat) while deciding whether the directory is small enough
    # to sort — bounded at LOCAL_SORT_MAX + 1, so memory can't run away.
    names: list[str] = []
    huge = False
    with os.scandir(d) as it:
        for de in it:
            if q and not de.name.startswith(q):
                continue
            if _escapes_root(root, de.path):
                continue  # a symlink out of the root isn't part of this folder
            names.append(de.name)
            if len(names) > LOCAL_SORT_MAX:
                huge = True
                break

    entries: list[StorageEntry] = []
    note: Optional[str] = None
    if not huge:
        names.sort(key=str.lower)
        page = names[offset : offset + limit]
        for name in page:
            try:
                de_path = os.path.join(d, name)
                st = os.stat(de_path)
                is_dir = os.path.isdir(de_path)
            except OSError:
                st, is_dir = None, False
            entries.append(StorageEntry(
                name=name,
                path=f"{rel}/{name}" if rel else name,
                kind="folder" if is_dir else "file",
                size=None if (is_dir or st is None) else st.st_size,
                modified=(datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()
                          if st else None),
            ))
        next_offset = offset + len(page)
        next_token = str(next_offset) if next_offset < len(names) else None
    else:
        # Stream: skip `offset`, take `limit`, stat only what we return.
        seen = 0
        more = False
        with os.scandir(d) as it:
            for de in it:
                if q and not de.name.startswith(q):
                    continue
                if _escapes_root(root, de.path):
                    continue
                seen += 1
                if seen <= offset:
                    continue
                if len(entries) >= limit:
                    more = True
                    break
                entries.append(_local_entry(de, rel))
        next_token = str(offset + len(entries)) if more else None
        note = (
            f"over {LOCAL_SORT_MAX:,} entries in this directory — listed in "
            "filesystem order, not sorted"
        )

    entries.sort(key=lambda e: (e.kind != "folder", e.name.lower()))
    return entries, next_token, note


def _local_offset(token: Optional[str]) -> int:
    if not token:
        return 0
    try:
        return max(0, int(token))
    except ValueError as e:
        raise HTTPException(status_code=400, detail="invalid page token") from e


def _s3_target_checked(s: Storage):
    target, _base = _s3_for_storage(s)
    if not target.bucket:
        raise HTTPException(status_code=400, detail="storage has no bucket configured")
    return target


def _s3_browse(target, key_prefix: str, rel: str, token: Optional[str], limit: int, q: str):
    """One page of a bucket directory → (entries, next_token). Runs in a thread.

    `q` is a name-prefix filter pushed into the S3 `Prefix` — the only filter the
    LIST API can do without scanning, and the reason the UI's filter is
    prefix-not-substring: on a million-object directory a client-side filter over
    the loaded page would just be wrong.
    """
    from . import bench

    scan_prefix = key_prefix + q
    entries: list[StorageEntry] = []
    next_token = token
    for _ in range(S3_EMPTY_PAGE_RETRIES):
        page = bench.s3_list_page(scan_prefix, target, token=next_token, limit=limit)
        next_token = page["next_token"]
        for folder_key in page["folders"]:
            name = folder_key[len(key_prefix):].rstrip("/")
            if name:
                entries.append(StorageEntry(
                    name=name, path=f"{rel}/{name}" if rel else name, kind="folder",
                ))
        for obj in page["files"]:
            name = obj["key"][len(key_prefix):]
            if name:
                entries.append(StorageEntry(
                    name=name, path=f"{rel}/{name}" if rel else name, kind="file",
                    size=obj["size"], modified=obj["modified"],
                ))
        # An empty page with more to come is normal under a delimiter (every key
        # scanned collapsed into a folder already returned) — keep pulling rather
        # than handing the UI a blank screen with a "load more" button.
        if entries or not next_token:
            break
    entries.sort(key=lambda e: (e.kind != "folder", e.name.lower()))
    return entries, next_token


@router.get("/{storage_id}/browse", response_model=StorageBrowseResponse)
async def storage_browse(
    storage_id: str,
    path: str = "",
    token: Optional[str] = None,
    q: str = Query("", description="name-prefix filter within this directory"),
    limit: int = Query(300, ge=1, le=BROWSE_PAGE_MAX),
    user: User = Depends(require_admin),  # noqa: ARG001 — raw store access is an infra op
    session: AsyncSession = Depends(get_session),
):
    """One directory: child folders + files, folders first, ALWAYS paged — a
    directory holding a million objects returns a page and a `next_token`, never
    the whole listing. `q` filters by name prefix on the server (see `_s3_browse`)."""
    s = await _require_browsable_storage(session, storage_id)
    rel = _safe_rel(path)
    query = (q or "").strip()

    if s.kind == "local":
        root = _local_root(s)
        entries, next_token, note = await run_in_threadpool(
            _local_browse, root, rel, _local_offset(token), limit, query,
        )
        return StorageBrowseResponse(
            storage_id=storage_id, kind="local", bucket="", root=root, path=rel,
            entries=entries, next_token=next_token, note=note,
        )

    target = _s3_target_checked(s)
    root = _storage_root(s)
    key_prefix = _abs_key(root, rel)
    if key_prefix:
        key_prefix += "/"
    try:
        entries, next_token = await run_in_threadpool(
            _s3_browse, target, key_prefix, rel, token, limit, query,
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — bad creds / missing bucket / network
        raise HTTPException(status_code=502, detail=_s3_error_message(e)) from e
    return StorageBrowseResponse(
        storage_id=storage_id, kind="s3", bucket=target.bucket, root=root, path=rel,
        entries=entries, next_token=next_token,
    )


def _object_head_sync(s: Storage, rel: str) -> Optional[dict]:
    """{size, content_type} for one object on either backing store, or None."""
    if s.kind == "local":
        p = _local_path(_local_root(s), rel)
        if not os.path.isfile(p):
            return None
        return {"size": os.path.getsize(p), "content_type": ""}
    from . import bench

    return bench.s3_head(_abs_key(_storage_root(s), rel), _s3_target_checked(s))


def _object_read_sync(s: Storage, rel: str, max_bytes: Optional[int]) -> Optional[bytes]:
    """The first `max_bytes` (all of it when None) of one object, or None."""
    if s.kind == "local":
        p = _local_path(_local_root(s), rel)
        try:
            with open(p, "rb") as f:
                return f.read(max_bytes) if max_bytes else f.read()
        except OSError:
            return None
    from . import bench

    return bench.s3_get_head_bytes(
        _abs_key(_storage_root(s), rel), _s3_target_checked(s), max_bytes=max_bytes,
    )


async def _object_download(s: Storage, rel: str, name: str):
    """The uncapped path. Local streams off disk; S3 redirects to a presigned URL
    so a multi-GB checkpoint never streams through the gateway."""
    safe_name = _UNSAFE_FILENAME.sub("_", name)
    if s.kind == "local":
        p = _local_path(_local_root(s), rel)
        if not os.path.isfile(p):
            raise HTTPException(status_code=404, detail="object not found")
        # octet-stream (not the sniffed type) so the browser saves it, and so the
        # web proxy takes its byte-exact binary branch rather than decoding text.
        return FileResponse(
            p, media_type="application/octet-stream", filename=safe_name,
        )
    from . import bench

    target = _s3_target_checked(s)
    key = _abs_key(_storage_root(s), rel)
    if await run_in_threadpool(bench.s3_head, key, target) is None:
        raise HTTPException(status_code=404, detail="object not found")
    url = await run_in_threadpool(bench.s3_presign_get, key, 3600, target)
    return RedirectResponse(url, status_code=307)


@router.get("/{storage_id}/object")
async def storage_object(
    storage_id: str,
    path: str,
    max_bytes: int = Query(DEFAULT_PREVIEW_BYTES, ge=1, le=MAX_INLINE_BYTES),
    download: bool = False,
    user: User = Depends(require_admin),  # noqa: ARG001
    session: AsyncSession = Depends(get_session),
):
    """Serve one object's bytes through the gateway for the viewer (same-origin +
    authed, so previews work without a bucket CORS policy).

    Text is head-read to `max_bytes` — a truncated log still previews fine, and
    the response says how much was cut. Binary (image/audio/…) is all-or-nothing:
    half a WAV is not a smaller WAV, so anything over the cap is refused with a
    413 pointing at the download instead.

    `download=1` is the uncapped path: a **local** file streams off disk, an S3
    object 307s to a presigned URL so its bytes never cross the control plane.
    """
    s = await _require_browsable_storage(session, storage_id)
    rel = _safe_rel(path)
    if not rel:
        raise HTTPException(status_code=400, detail="path is required")
    name = rel.rsplit("/", 1)[-1]

    if download:
        return await _object_download(s, rel, name)

    head = await run_in_threadpool(_object_head_sync, s, rel)
    if head is None:
        raise HTTPException(status_code=404, detail="object not found")
    # ⚠ The FILENAME decides the type, not S3's stored Content-Type — that's
    # arbitrary caller-supplied metadata and it lies routinely (this platform's
    # own bucket stamps .jsonl objects `application/jsonl`, which no whitelist
    # would call text). The stored type is only the fallback for an extension we
    # don't know, and such files are sniffed below anyway.
    ext_type = _media_type_for(name)
    media_type = ext_type or head["content_type"] or "application/octet-stream"
    size = head["size"]
    textual = _is_textual(media_type)
    # Only an unrecognized extension may be re-typed by sniffing — a .wav whose
    # first bytes happen to decode as utf-8 is still a .wav.
    sniffable = ext_type is None and not textual
    read_bytes: Optional[int] = None
    too_large = HTTPException(
        status_code=413,
        detail=(
            f"{name} is {size} bytes — too large to preview inline; "
            "use the download link instead"
        ),
    )
    if size > max_bytes:
        if sniffable:
            # Sniff before refusing: an extensionless multi-GB log heads fine.
            probe = await run_in_threadpool(_object_read_sync, s, rel, 4096)
            if probe is not None and _looks_textual(probe):
                textual, media_type = True, "text/plain; charset=utf-8"
        if not textual:
            raise too_large
        read_bytes = max_bytes

    data = await run_in_threadpool(_object_read_sync, s, rel, read_bytes)
    if data is None:
        raise HTTPException(status_code=404, detail="object not found")
    if sniffable and not textual and _looks_textual(data):
        textual, media_type = True, "text/plain; charset=utf-8"
    if textual:
        # A head-read can land mid-codepoint; drop the partial tail rather than
        # emitting invalid utf-8.
        data = data.decode("utf-8", "replace").encode("utf-8")
    safe_name = _UNSAFE_FILENAME.sub("_", name)
    return Response(
        content=data,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{safe_name}"',
            "X-Object-Size": str(size),
            "X-Object-Truncated": "1" if read_bytes is not None else "0",
            "Cache-Control": "private, max-age=60",
        },
    )


@router.get("/{storage_id}/object-url", response_model=StorageObjectUrlResponse)
async def storage_object_url(
    storage_id: str,
    path: str,
    expires: int = Query(3600, ge=60, le=7 * 24 * 3600),
    user: User = Depends(require_admin),  # noqa: ARG001
    session: AsyncSession = Depends(get_session),
):
    """A presigned GET url for one S3 object — the download path, so multi-GB
    artifacts stream from S3 directly instead of through the gateway. Local
    storage has nothing to presign; its download is `object?download=1`."""
    from . import bench

    s = await _require_browsable_storage(session, storage_id)
    if s.kind != "s3":
        raise HTTPException(
            status_code=400,
            detail="local storage has no presigned URL — download via object?download=1",
        )
    target = _s3_target_checked(s)
    rel = _safe_rel(path)
    if not rel:
        raise HTTPException(status_code=400, detail="path is required")
    key = _abs_key(_storage_root(s), rel)
    if await run_in_threadpool(bench.s3_head, key, target) is None:
        raise HTTPException(status_code=404, detail="object not found")
    url = await run_in_threadpool(bench.s3_presign_get, key, expires, target)
    return StorageObjectUrlResponse(url=url, expires_in=expires)


@router.get("/{storage_id}/purge-status")
async def storage_purge_status(
    storage_id: str,
    user: User = Depends(require_admin),  # noqa: ARG001
    session: AsyncSession = Depends(get_session),
):
    """Progress of the current/last cleanup for this storage (`state=idle` if none).
    The UI polls this while a delete runs, and on reopen to resume the progress view."""
    await _require_s3_storage(session, storage_id)
    job = _PURGE_JOBS.get(storage_id)
    if not job:
        return {"storage_id": storage_id, "state": "idle"}
    return _job_public(job)
