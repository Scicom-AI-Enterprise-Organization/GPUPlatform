"""Global environment variables / secrets — org-wide, admin-managed.

Admins set key/value pairs in the UI; the gateway merges them into every
workload's environment (benchmark pods + serverless workers) so things like
`HF_TOKEN` work everywhere without baking them into the gateway's own env. A
per-resource var of the same name overrides the global one.

Values are Fernet-encrypted at rest (same key as provider/storage secrets) and
secret values are never returned in plaintext — the list endpoint masks them.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import crypto
from .auth import require_admin
from .db import GlobalEnv, User, get_session

router = APIRouter(prefix="/v1/global-env", tags=["global-env"])

# POSIX-ish env var name: letter/underscore then letters/digits/underscores.
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_VALUE = 8000


class GlobalEnvUpsert(BaseModel):
    value: str
    is_secret: bool = True
    description: Optional[str] = None


class GlobalEnvRecord(BaseModel):
    key: str
    is_secret: bool
    value: Optional[str]          # plaintext for non-secrets; None for secrets
    value_preview: Optional[str]  # masked hint for secrets (e.g. "hf_…XbBh")
    description: Optional[str]
    updated_by: str
    updated_at: datetime


def _mask(v: str) -> str:
    if len(v) <= 8:
        return "•" * max(len(v), 4)
    return f"{v[:3]}…{v[-4:]}"


def _to_record(r: GlobalEnv) -> GlobalEnvRecord:
    try:
        plain = crypto.decrypt(r.value_enc)
    except Exception:
        plain = ""
    return GlobalEnvRecord(
        key=r.key,
        is_secret=r.is_secret,
        value=None if r.is_secret else plain,
        value_preview=_mask(plain) if r.is_secret else None,
        description=r.description,
        updated_by=r.updated_by,
        updated_at=r.updated_at,
    )


@router.get("", response_model=list[GlobalEnvRecord])
async def list_global_env(
    user: User = Depends(require_admin),  # noqa: ARG001 — admin-only
    session: AsyncSession = Depends(get_session),
):
    rows = (await session.execute(select(GlobalEnv).order_by(GlobalEnv.key))).scalars().all()
    return [_to_record(r) for r in rows]


@router.put("/{key}", response_model=GlobalEnvRecord)
async def upsert_global_env(
    key: str,
    req: GlobalEnvUpsert,
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    key = key.strip()
    if not _KEY_RE.match(key):
        raise HTTPException(status_code=400, detail="key must be a valid env var name (letters, digits, underscore; not starting with a digit)")
    if not req.value:
        raise HTTPException(status_code=400, detail="value is required")
    if len(req.value) > _MAX_VALUE:
        raise HTTPException(status_code=400, detail=f"value too long (max {_MAX_VALUE} chars)")
    row = await session.get(GlobalEnv, key)
    if row is None:
        row = GlobalEnv(key=key)
        session.add(row)
    row.value_enc = crypto.encrypt(req.value)
    row.is_secret = bool(req.is_secret)
    row.description = (req.description or "").strip() or None
    row.updated_by = user.username
    row.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(row)
    return _to_record(row)


@router.delete("/{key}")
async def delete_global_env(
    key: str,
    user: User = Depends(require_admin),  # noqa: ARG001 — admin-only
    session: AsyncSession = Depends(get_session),
):
    row = await session.get(GlobalEnv, key.strip())
    if row is None:
        raise HTTPException(status_code=404, detail="no such key")
    await session.delete(row)
    await session.commit()
    return {"ok": True, "key": key.strip()}


async def load_global_env(session: AsyncSession) -> dict[str, str]:
    """Every global var as {key: decrypted_value} for injecting into workloads.
    Skips any row that fails to decrypt (e.g. PROVIDER_SECRET_KEY rotated)."""
    rows = (await session.execute(select(GlobalEnv))).scalars().all()
    out: dict[str, str] = {}
    for r in rows:
        try:
            out[r.key] = crypto.decrypt(r.value_enc)
        except Exception:
            continue
    return out
