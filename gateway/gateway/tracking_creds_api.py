"""Experiment-tracker credentials — named W&B / MLflow creds, org-wide and
admin-managed (shown as a card on the Secrets page). Autotrain runs reference
one by id; the runner decrypts it and injects the canonical tracker env.

Secrets are Fernet-encrypted at rest (same key as global-env / storage) and are
never returned in plaintext — the list endpoint masks them. Listing is open to
developers (so the Autotrain form can populate its pickers — names + masked
previews aren't secret); creating / deleting is admin-only, like the Secrets page.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import crypto
from .auth import require_admin, require_developer
from .db import TrackingCredential, User, get_session, session_factory

router = APIRouter(prefix="/v1/tracking-credentials", tags=["tracking-credentials"])

KINDS = ("wandb", "mlflow")


class CreateTrackingCredentialRequest(BaseModel):
    name: str
    kind: str                       # "wandb" | "mlflow"
    api_key: Optional[str] = None   # wandb
    uri: Optional[str] = None       # mlflow
    username: Optional[str] = None  # mlflow
    password: Optional[str] = None  # mlflow


class TrackingCredentialRecord(BaseModel):
    id: str
    name: str
    kind: str
    preview: str                    # masked hint, never the secret
    created_by: str
    created_at: datetime


def _mask(v: str) -> str:
    if not v:
        return ""
    return "•" * max(len(v), 4) if len(v) <= 8 else f"{v[:3]}…{v[-4:]}"


def _preview(kind: str, cfg: dict) -> str:
    if kind == "wandb":
        return _mask(cfg.get("api_key") or "")
    uri = cfg.get("uri") or "—"
    user = cfg.get("username") or "—"
    return f"{user} @ {uri}"


def _to_record(row: TrackingCredential) -> TrackingCredentialRecord:
    try:
        cfg = json.loads(crypto.decrypt(row.config_enc))
    except Exception:
        cfg = {}
    return TrackingCredentialRecord(
        id=row.id, name=row.name, kind=row.kind, preview=_preview(row.kind, cfg),
        created_by=row.created_by, created_at=row.created_at,
    )


@router.get("", response_model=list[TrackingCredentialRecord])
async def list_tracking_credentials(
    kind: Optional[str] = None,
    user: User = Depends(require_developer),  # noqa: ARG001
    session: AsyncSession = Depends(get_session),
):
    q = select(TrackingCredential).order_by(TrackingCredential.name)
    if kind in KINDS:
        q = q.where(TrackingCredential.kind == kind)
    rows = (await session.execute(q)).scalars().all()
    return [_to_record(r) for r in rows]


@router.post("", response_model=TrackingCredentialRecord)
async def create_tracking_credential(
    req: CreateTrackingCredentialRequest,
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if req.kind not in KINDS:
        raise HTTPException(status_code=400, detail=f"kind must be one of {KINDS}")
    if req.kind == "wandb":
        if not (req.api_key or "").strip():
            raise HTTPException(status_code=400, detail="api_key is required for wandb")
        cfg = {"api_key": req.api_key.strip()}
    else:  # mlflow
        if not (req.uri or "").strip():
            raise HTTPException(status_code=400, detail="uri is required for mlflow")
        cfg = {
            "uri": req.uri.strip(),
            "username": (req.username or "").strip(),
            "password": (req.password or "").strip(),
        }
    row = TrackingCredential(
        id="track-" + uuid.uuid4().hex[:8], name=name, kind=req.kind,
        config_enc=crypto.encrypt(json.dumps(cfg)), created_by=user.username,
        created_at=datetime.now(timezone.utc),
    )
    session.add(row)
    await session.commit()
    return _to_record(row)


@router.delete("/{cred_id}")
async def delete_tracking_credential(
    cred_id: str,
    user: User = Depends(require_admin),  # noqa: ARG001
    session: AsyncSession = Depends(get_session),
):
    row = await session.get(TrackingCredential, cred_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such credential")
    await session.delete(row)
    await session.commit()
    return {"ok": True, "id": cred_id}


async def resolve_tracking_env(cred_id: Optional[str]) -> dict[str, str]:
    """Decrypt a credential by id → canonical tracker env vars. {} if missing.
    Opens its own short-lived session (called from the Autotrain runner)."""
    if not cred_id:
        return {}
    async with session_factory()() as s:
        row = await s.get(TrackingCredential, cred_id)
    if row is None:
        return {}
    try:
        cfg = json.loads(crypto.decrypt(row.config_enc))
    except Exception:
        return {}
    if row.kind == "wandb":
        return {"WANDB_API_KEY": cfg.get("api_key") or ""}
    env = {}
    if cfg.get("uri"):
        env["MLFLOW_TRACKING_URI"] = cfg["uri"]
    if cfg.get("username"):
        env["MLFLOW_TRACKING_USERNAME"] = cfg["username"]
    if cfg.get("password"):
        env["MLFLOW_TRACKING_PASSWORD"] = cfg["password"]
    return env
