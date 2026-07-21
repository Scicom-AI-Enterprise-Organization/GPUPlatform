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

import asyncio
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
