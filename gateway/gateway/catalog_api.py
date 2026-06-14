"""Management API for the self-hosted HuggingFace catalog (`/v1/catalog`).

Drives the web "Models" section: list/register/inspect/delete hosted model &
dataset repos. The actual HF Hub wire protocol (push/pull) lives in
`hf_mirror_api.py`; this is the CRUD + browse surface the UI talks to.

A repo points at a `Storage` backend (s3/local/sftp) under `prefix`. You can
either push to it with HF tooling (`hf upload` auto-creates the repo) or
**register** an existing prefix that already holds HF-layout files and
`reindex` to seed the manifest.
"""
from __future__ import annotations

import hashlib
import logging
import re
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import storage_backends as sb
from .auth import require_section
from .db import CatalogRepo, Storage, User, get_session
from .hf_mirror_api import _compute_sha, _is_lfs

logger = logging.getLogger("gateway.catalog")

router = APIRouter(prefix="/v1/catalog", tags=["catalog"])

_catalog = require_section("catalog")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# ---------- models ------------------------------------------------------


class CreateCatalogRequest(BaseModel):
    repo_type: str = "model"  # model | dataset
    namespace: str
    name: str
    storage_id: str
    prefix: Optional[str] = None
    private: bool = True
    description: Optional[str] = None


class UpdateCatalogRequest(BaseModel):
    private: Optional[bool] = None
    description: Optional[str] = None
    storage_id: Optional[str] = None
    prefix: Optional[str] = None


class CatalogFile(BaseModel):
    path: str
    size: Optional[int] = None
    lfs: bool = False
    oid: Optional[str] = None


class CatalogRecord(BaseModel):
    id: str
    repo_type: str
    namespace: str
    name: str
    full_id: str
    storage_id: Optional[str] = None
    storage_name: Optional[str] = None
    prefix: str
    sha: Optional[str] = None
    private: bool = True
    description: Optional[str] = None
    size_bytes: Optional[int] = None
    num_files: Optional[int] = None
    created_at: str
    updated_at: str
    created_by: str
    files: Optional[list[CatalogFile]] = None


# ---------- helpers -----------------------------------------------------


def _to_record(r: CatalogRepo, owner: str, storage_name: Optional[str],
               with_files: bool = False) -> CatalogRecord:
    files = None
    if with_files:
        files = [
            CatalogFile(path=e.get("path"), size=e.get("size"),
                        lfs=bool(e.get("lfs")), oid=e.get("oid"))
            for e in (r.manifest or [])
        ]
    return CatalogRecord(
        id=r.id,
        repo_type=r.repo_type,
        namespace=r.namespace,
        name=r.name,
        full_id=r.full_id,
        storage_id=r.storage_id,
        storage_name=storage_name,
        prefix=r.prefix,
        sha=r.sha,
        private=bool(r.private),
        description=r.description,
        size_bytes=r.size_bytes,
        num_files=r.num_files,
        created_at=r.created_at.isoformat() if r.created_at else "",
        updated_at=r.updated_at.isoformat() if r.updated_at else "",
        created_by=owner,
        files=files,
    )


async def _name_maps(session: AsyncSession, rows: list[CatalogRepo]) -> tuple[dict, dict]:
    owner_ids = {r.owner_id for r in rows}
    store_ids = {r.storage_id for r in rows if r.storage_id}
    owners: dict[int, str] = {}
    stores: dict[str, str] = {}
    if owner_ids:
        res = await session.execute(select(User).where(User.id.in_(owner_ids)))
        owners = {u.id: u.username for u in res.scalars().all()}
    if store_ids:
        res = await session.execute(select(Storage).where(Storage.id.in_(store_ids)))
        stores = {s.id: s.name for s in res.scalars().all()}
    return owners, stores


def _reindex_manifest(backend: sb.StorageBackend, prefix: str) -> tuple[list[dict], int]:
    """Rebuild a manifest by listing a storage prefix (register-existing flow).
    Skips the internal `.hf-lfs/` blob area; treats every file as path-stored."""
    objs = backend.list_prefix(prefix)
    base = prefix.strip("/") + "/"
    manifest: list[dict] = []
    total = 0
    for o in objs:
        key = o["key"]
        rel = key[len(base):] if key.startswith(base) else key
        if not rel or rel.startswith(".hf-lfs/"):
            continue
        size = int(o.get("size") or 0)
        etag = hashlib.sha1(f"{rel}:{size}".encode()).hexdigest()
        manifest.append({"path": rel, "size": size, "oid": etag,
                         "lfs": _is_lfs(rel, size), "etag": etag})
        total += size
    manifest.sort(key=lambda e: e["path"])
    return manifest, total


# ---------- endpoints ---------------------------------------------------


@router.get("", response_model=list[CatalogRecord])
async def list_catalog(
    scope: str = "mine",
    repo_type: Optional[str] = None,
    user: User = Depends(_catalog),
    session: AsyncSession = Depends(get_session),
):
    q = select(CatalogRepo).order_by(CatalogRepo.updated_at.desc())
    if scope != "all":
        q = q.where(CatalogRepo.owner_id == user.id)
    if repo_type in ("model", "dataset"):
        q = q.where(CatalogRepo.repo_type == repo_type)
    rows = list((await session.execute(q)).scalars().all())
    owners, stores = await _name_maps(session, rows)
    return [_to_record(r, owners.get(r.owner_id, "?"), stores.get(r.storage_id)) for r in rows]


@router.post("", response_model=CatalogRecord)
async def create_catalog(
    req: CreateCatalogRequest,
    user: User = Depends(_catalog),
    session: AsyncSession = Depends(get_session),
):
    repo_type = (req.repo_type or "model").lower()
    if repo_type not in ("model", "dataset"):
        raise HTTPException(status_code=400, detail="repo_type must be model or dataset")
    ns = (req.namespace or "").strip().strip("/")
    name = (req.name or "").strip().strip("/")
    if not _NAME_RE.match(ns) or not _NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="namespace/name must be alphanumeric (._- allowed)")

    store = await session.get(Storage, req.storage_id)
    if store is None:
        raise HTTPException(status_code=400, detail="storage not found")
    if store.kind not in ("s3", "local", "sftp"):
        raise HTTPException(status_code=400, detail=f"storage kind {store.kind} can't host repos (need s3/local/sftp)")

    existing = await session.execute(
        select(CatalogRepo).where(
            CatalogRepo.repo_type == repo_type,
            CatalogRepo.namespace == ns,
            CatalogRepo.name == name,
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail=f"{repo_type} {ns}/{name} already exists")

    prefix = (req.prefix or "").strip().strip("/") or f"catalog/{ns}/{name}"

    # Seed the manifest if the prefix already holds files (register-existing).
    manifest: list[dict] = []
    total = 0
    try:
        backend = await run_in_threadpool(sb.resolve_backend, store)
        manifest, total = await run_in_threadpool(_reindex_manifest, backend, prefix)
    except sb.StorageError as e:
        raise HTTPException(status_code=400, detail=f"storage error: {e}") from e
    except Exception as e:  # noqa: BLE001 — empty/unreachable prefix → start empty
        logger.info("catalog: prefix %s not pre-seeded (%s)", prefix, e)

    repo = CatalogRepo(
        id=f"repo-{secrets.token_hex(4)}",
        owner_id=user.id,
        repo_type=repo_type,
        namespace=ns,
        name=name,
        full_id=f"{ns}/{name}",
        storage_id=store.id,
        prefix=prefix,
        sha=_compute_sha(manifest),
        private=bool(req.private),
        description=(req.description or "").strip() or None,
        manifest=manifest,
        size_bytes=total,
        num_files=len(manifest),
    )
    session.add(repo)
    await session.commit()
    await session.refresh(repo)
    logger.info("catalog: registered %s %s/%s on %s (%d files)", repo_type, ns, name, store.id, len(manifest))
    return _to_record(repo, user.username, store.name, with_files=True)


@router.get("/lookup", response_model=CatalogRecord)
async def lookup_catalog(
    repo_type: str,
    namespace: str,
    name: str,
    user: User = Depends(_catalog),
    session: AsyncSession = Depends(get_session),
):
    """Resolve a repo by its HF id (repo_type + namespace/name) — backs the
    name-based detail URLs (/models/<ns>/<name>, /datasets/hosted/<ns>/<name>)."""
    res = await session.execute(select(CatalogRepo).where(
        CatalogRepo.repo_type == repo_type,
        CatalogRepo.namespace == namespace,
        CatalogRepo.name == name,
    ))
    repo = res.scalar_one_or_none()
    if repo is None or (repo.owner_id != user.id and not user.is_admin and repo.private):
        raise HTTPException(status_code=404, detail="repo not found")
    owners, stores = await _name_maps(session, [repo])
    return _to_record(repo, owners.get(repo.owner_id, "?"), stores.get(repo.storage_id), with_files=True)


@router.get("/{repo_id}", response_model=CatalogRecord)
async def get_catalog(
    repo_id: str,
    user: User = Depends(_catalog),
    session: AsyncSession = Depends(get_session),
):
    repo = await session.get(CatalogRepo, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="repo not found")
    if repo.owner_id != user.id and not user.is_admin and repo.private:
        raise HTTPException(status_code=404, detail="repo not found")
    owners, stores = await _name_maps(session, [repo])
    return _to_record(repo, owners.get(repo.owner_id, "?"), stores.get(repo.storage_id), with_files=True)


@router.patch("/{repo_id}", response_model=CatalogRecord)
async def update_catalog(
    repo_id: str,
    req: UpdateCatalogRequest,
    user: User = Depends(_catalog),
    session: AsyncSession = Depends(get_session),
):
    repo = await session.get(CatalogRepo, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="repo not found")
    if repo.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="not your repo")
    if req.private is not None:
        repo.private = req.private
    if req.description is not None:
        repo.description = req.description.strip() or None
    if req.storage_id is not None:
        store = await session.get(Storage, req.storage_id)
        if store is None or store.kind not in ("s3", "local", "sftp"):
            raise HTTPException(status_code=400, detail="invalid storage")
        repo.storage_id = req.storage_id
    if req.prefix is not None:
        repo.prefix = req.prefix.strip().strip("/") or repo.prefix
    await session.commit()
    await session.refresh(repo)
    owners, stores = await _name_maps(session, [repo])
    return _to_record(repo, owners.get(repo.owner_id, "?"), stores.get(repo.storage_id), with_files=True)


@router.post("/{repo_id}/reindex", response_model=CatalogRecord)
async def reindex_catalog(
    repo_id: str,
    user: User = Depends(_catalog),
    session: AsyncSession = Depends(get_session),
):
    repo = await session.get(CatalogRepo, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="repo not found")
    if repo.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="not your repo")
    store = await session.get(Storage, repo.storage_id) if repo.storage_id else None
    if store is None:
        raise HTTPException(status_code=400, detail="repo has no storage")
    try:
        backend = await run_in_threadpool(sb.resolve_backend, store)
        manifest, total = await run_in_threadpool(_reindex_manifest, backend, repo.prefix)
    except sb.StorageError as e:
        raise HTTPException(status_code=400, detail=f"storage error: {e}") from e
    repo.manifest = manifest
    repo.sha = _compute_sha(manifest)
    repo.size_bytes = total
    repo.num_files = len(manifest)
    await session.commit()
    await session.refresh(repo)
    owners, stores = await _name_maps(session, [repo])
    return _to_record(repo, owners.get(repo.owner_id, "?"), stores.get(repo.storage_id), with_files=True)


@router.delete("/{repo_id}")
async def delete_catalog(
    repo_id: str,
    wipe: bool = False,
    user: User = Depends(_catalog),
    session: AsyncSession = Depends(get_session),
):
    repo = await session.get(CatalogRepo, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="repo not found")
    if repo.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="not your repo")
    if wipe and repo.storage_id:
        store = await session.get(Storage, repo.storage_id)
        if store is not None:
            try:
                backend = await run_in_threadpool(sb.resolve_backend, store)
                objs = await run_in_threadpool(backend.list_prefix, repo.prefix)
                for o in objs:
                    await run_in_threadpool(backend.delete, o["key"])
            except Exception as e:  # noqa: BLE001
                logger.warning("catalog: wipe of %s failed: %s", repo.full_id, e)
    await session.delete(repo)
    await session.commit()
    return {"ok": True, "id": repo_id}
