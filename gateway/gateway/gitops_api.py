"""HTTP routes + background poll loop for GitOps.

Register a git repo (URL + branch + optional sub-path); the gateway reconciles
the platform resources declared in its YAML manifests to match. Three triggers:

- **auto-poll** — `gitops_reconcile_loop` (started in main's lifespan) fetches each
  enabled repo whose `poll_interval` has elapsed and reconciles drift.
- **manual** — `POST /v1/gitops/{id}/sync` forces an immediate reconcile.
- **webhook** — `POST /v1/gitops/webhook` (GitHub-style push hook) verifies a
  per-repo HMAC and reconciles on push.

Writes are admin-only (these hold shared infra + credentials, like Providers /
Storage). The actual create/update/delete is done by `gitops_engine.reconcile_repo`,
which calls the existing resource handlers in-process.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from . import crypto
from .auth import current_user, require_admin
from .db import GitopsRepo, GitopsResource, User, get_session, session_factory
from . import gitops_engine

logger = logging.getLogger("gateway.gitops")

router = APIRouter(prefix="/v1/gitops", tags=["gitops"])

MIN_POLL_INTERVAL = 30
# How often the background loop wakes to check which repos are due.
POLL_TICK_S = 30


# ---------- request / response models ---------------------------------------

class CreateGitopsRepoRequest(BaseModel):
    name: str
    url: str
    branch: str = "main"
    path: Optional[str] = None
    token_secret: Optional[str] = None      # GlobalEnv key holding a git token (private repos)
    webhook_secret: Optional[str] = None     # HMAC secret for the push webhook
    prune: bool = True
    poll_interval: int = 300
    enabled: bool = True


class UpdateGitopsRepoRequest(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    branch: Optional[str] = None
    path: Optional[str] = None
    token_secret: Optional[str] = None
    # None = keep existing; "" = clear; non-empty = replace.
    webhook_secret: Optional[str] = None
    prune: Optional[bool] = None
    poll_interval: Optional[int] = None
    enabled: Optional[bool] = None


class GitopsRepoRecord(BaseModel):
    id: str
    name: str
    url: str
    branch: str
    path: Optional[str] = None
    token_secret: Optional[str] = None
    has_webhook_secret: bool = False
    prune: bool = True
    poll_interval: int = 300
    enabled: bool = True
    last_synced_sha: Optional[str] = None
    last_sync_at: Optional[str] = None
    last_sync_status: str = "never"
    last_sync_error: Optional[str] = None
    resource_count: int = 0
    created_at: str
    created_by: str


class GitopsResourceRecord(BaseModel):
    id: str
    kind: str
    name: str
    resource_id: Optional[str] = None
    generation: int = 1
    status: str = "applied"
    error: Optional[str] = None
    last_synced_at: str


class SyncResponse(BaseModel):
    ok: bool
    skipped: bool = False
    sha: Optional[str] = None
    created: list[str] = []
    updated: list[str] = []
    pruned: list[str] = []
    unchanged: int = 0
    errors: list[str] = []


# ---------- helpers ----------------------------------------------------------

async def _resource_count(session: AsyncSession, repo_id: str) -> int:
    return int((await session.execute(
        select(func.count()).select_from(GitopsResource).where(GitopsResource.repo_id == repo_id)
    )).scalar() or 0)


def _to_record(r: GitopsRepo, owner_username: str, resource_count: int) -> GitopsRepoRecord:
    return GitopsRepoRecord(
        id=r.id,
        name=r.name,
        url=r.url,
        branch=r.branch,
        path=r.path,
        token_secret=r.token_secret,
        has_webhook_secret=bool(r.webhook_secret_enc),
        prune=bool(r.prune),
        poll_interval=r.poll_interval,
        enabled=bool(r.enabled),
        last_synced_sha=r.last_synced_sha,
        last_sync_at=r.last_sync_at.isoformat() if r.last_sync_at else None,
        last_sync_status=r.last_sync_status,
        last_sync_error=r.last_sync_error,
        resource_count=resource_count,
        created_at=r.created_at.isoformat() if r.created_at else "",
        created_by=owner_username,
    )


def _normalize_git_url(url: str) -> str:
    """Loose normalization for webhook repo matching: drop scheme userinfo, a
    trailing .git and slash, lowercase. `https://t@github.com/o/r.git` and
    `git@github.com:o/r` both → `github.com/o/r`."""
    u = (url or "").strip().lower()
    for pre in ("https://", "http://", "ssh://", "git://"):
        if u.startswith(pre):
            u = u[len(pre):]
            break
    if u.startswith("git@"):
        u = u[len("git@"):]
    if "@" in u.split("/", 1)[0]:  # strip token@host userinfo
        u = u.split("@", 1)[1]
    u = u.replace(":", "/", 1) if "/" not in u.split(":", 1)[0] else u
    if u.endswith(".git"):
        u = u[:-4]
    return u.rstrip("/")


# ---------- CRUD -------------------------------------------------------------

@router.get("", response_model=list[GitopsRepoRecord])
async def list_repos(
    user: User = Depends(require_admin),  # noqa: ARG001 — admin-only infra
    session: AsyncSession = Depends(get_session),
):
    rows = (await session.execute(select(GitopsRepo).order_by(GitopsRepo.created_at.desc()))).scalars().all()
    owner_ids = {r.owner_id for r in rows}
    owners: dict[int, str] = {}
    if owner_ids:
        for u in (await session.execute(select(User).where(User.id.in_(owner_ids)))).scalars().all():
            owners[u.id] = u.username
    out = []
    for r in rows:
        out.append(_to_record(r, owners.get(r.owner_id, "?"), await _resource_count(session, r.id)))
    return out


@router.post("", response_model=GitopsRepoRecord)
async def create_repo(
    req: CreateGitopsRepoRequest,
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    name = req.name.strip()
    url = req.url.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    rid = f"gitops-{secrets.token_hex(4)}"
    row = GitopsRepo(
        id=rid,
        owner_id=user.id,
        name=name,
        url=url,
        branch=(req.branch or "main").strip() or "main",
        path=(req.path or "").strip() or None,
        token_secret=(req.token_secret or "").strip() or None,
        webhook_secret_enc=crypto.encrypt(req.webhook_secret.strip()) if (req.webhook_secret or "").strip() else None,
        prune=bool(req.prune),
        poll_interval=max(MIN_POLL_INTERVAL, int(req.poll_interval or 300)),
        enabled=bool(req.enabled),
        last_sync_status="never",
        created_at=datetime.now(timezone.utc),
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    logger.info("created gitops repo %s (%s @ %s) for user=%s", rid, url, row.branch, user.username)
    return _to_record(row, user.username, 0)


class TestGitopsRepoRequest(BaseModel):
    url: str
    branch: str = "main"
    token_secret: Optional[str] = None  # GlobalEnv key holding a git token


class TestGitopsRepoResponse(BaseModel):
    ok: bool
    message: str
    sha: Optional[str] = None


@router.post("/test", response_model=TestGitopsRepoResponse)
async def test_repo(
    req: TestGitopsRepoRequest,
    user: User = Depends(require_admin),  # noqa: ARG001 — admin-only infra
    session: AsyncSession = Depends(get_session),
):
    """Validate a repo's reachability/branch/auth WITHOUT creating it (a
    `git ls-remote`, no clone). Lets the create form gate "Add repository" on a
    passing test. Returns ok=false with a reason instead of 500 on a bad repo."""
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    token: Optional[str] = None
    key = (req.token_secret or "").strip()
    if key:
        env = await gitops_engine.load_global_env(session)
        token = env.get(key)
        if token is None:
            return TestGitopsRepoResponse(
                ok=False, message=f"secret key '{key}' not found in Secrets"
            )
    branch = (req.branch or "main").strip() or "main"
    try:
        sha = await gitops_engine.test_remote(url, branch, token)
    except Exception as e:  # noqa: BLE001 — surface the (sanitized) reason
        return TestGitopsRepoResponse(ok=False, message=str(e))
    return TestGitopsRepoResponse(
        ok=True, message=f"Reachable — {branch} @ {sha[:12]}", sha=sha
    )


@router.get("/{repo_id}", response_model=GitopsRepoRecord)
async def get_repo(
    repo_id: str,
    user: User = Depends(require_admin),  # noqa: ARG001
    session: AsyncSession = Depends(get_session),
):
    row = await session.get(GitopsRepo, repo_id)
    if row is None:
        raise HTTPException(status_code=404, detail="repo not found")
    owner = await session.get(User, row.owner_id)
    return _to_record(row, owner.username if owner else "?", await _resource_count(session, repo_id))


@router.patch("/{repo_id}", response_model=GitopsRepoRecord)
async def update_repo(
    repo_id: str,
    req: UpdateGitopsRepoRequest,
    user: User = Depends(require_admin),  # noqa: ARG001
    session: AsyncSession = Depends(get_session),
):
    row = await session.get(GitopsRepo, repo_id)
    if row is None:
        raise HTTPException(status_code=404, detail="repo not found")
    if req.name is not None:
        n = req.name.strip()
        if not n:
            raise HTTPException(status_code=400, detail="name cannot be blank")
        row.name = n
    if req.url is not None:
        u = req.url.strip()
        if not u:
            raise HTTPException(status_code=400, detail="url cannot be blank")
        row.url = u
    if req.branch is not None:
        row.branch = req.branch.strip() or "main"
    if req.path is not None:
        row.path = req.path.strip() or None
    if req.token_secret is not None:
        row.token_secret = req.token_secret.strip() or None
    if req.webhook_secret is not None:
        s = req.webhook_secret.strip()
        row.webhook_secret_enc = crypto.encrypt(s) if s else None
    if req.prune is not None:
        row.prune = bool(req.prune)
    if req.poll_interval is not None:
        row.poll_interval = max(MIN_POLL_INTERVAL, int(req.poll_interval))
    if req.enabled is not None:
        row.enabled = bool(req.enabled)
    await session.commit()
    await session.refresh(row)
    owner = await session.get(User, row.owner_id)
    return _to_record(row, owner.username if owner else "?", await _resource_count(session, repo_id))


@router.delete("/{repo_id}")
async def delete_repo(
    repo_id: str,
    request: Request,
    prune: bool = False,
    user: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    """Unregister a repo. `?prune=true` also deletes every resource it manages;
    otherwise the resources are orphaned (left running) and only the ledger is
    dropped."""
    row = await session.get(GitopsRepo, repo_id)
    if row is None:
        raise HTTPException(status_code=404, detail="repo not found")
    if prune:
        ledger = (await session.execute(
            select(GitopsResource).where(GitopsResource.repo_id == repo_id)
        )).scalars().all()
        # delete in reverse dependency order
        owner = await session.get(User, row.owner_id)
        order = {k.canonical: k.order for k in gitops_engine._KINDS}
        for gr in sorted(ledger, key=lambda r: order.get(r.kind, 99), reverse=True):
            if gr.resource_id and owner is not None:
                try:
                    await gitops_engine._delete(gr.kind, gr.resource_id, app=request.app, user=owner, session=session)
                except Exception:
                    await session.rollback()
                    logger.warning("gitops delete %s: prune %s/%s failed", repo_id, gr.kind, gr.name)
    await session.delete(row)  # ledger rows cascade
    await session.commit()
    logger.info("deleted gitops repo %s (prune=%s) by user=%s", repo_id, prune, user.username)
    return {"ok": True, "id": repo_id, "pruned": prune}


@router.get("/{repo_id}/resources", response_model=list[GitopsResourceRecord])
async def list_resources(
    repo_id: str,
    user: User = Depends(require_admin),  # noqa: ARG001
    session: AsyncSession = Depends(get_session),
):
    row = await session.get(GitopsRepo, repo_id)
    if row is None:
        raise HTTPException(status_code=404, detail="repo not found")
    rows = (await session.execute(
        select(GitopsResource).where(GitopsResource.repo_id == repo_id).order_by(GitopsResource.kind, GitopsResource.name)
    )).scalars().all()
    return [
        GitopsResourceRecord(
            id=r.id, kind=r.kind, name=r.name, resource_id=r.resource_id,
            generation=r.generation, status=r.status, error=r.error,
            last_synced_at=r.last_synced_at.isoformat() if r.last_synced_at else "",
        )
        for r in rows
    ]


@router.post("/{repo_id}/sync", response_model=SyncResponse)
async def sync_repo(
    repo_id: str,
    request: Request,
    user: User = Depends(require_admin),  # noqa: ARG001
    session: AsyncSession = Depends(get_session),
):
    """Force an immediate reconcile and return what changed."""
    row = await session.get(GitopsRepo, repo_id)
    if row is None:
        raise HTTPException(status_code=404, detail="repo not found")
    res = await gitops_engine.reconcile_repo(request.app, session, row, force=True)
    return SyncResponse(
        ok=res.ok, skipped=res.skipped, sha=res.sha,
        created=res.created, updated=res.updated, pruned=res.pruned,
        unchanged=res.unchanged, errors=res.errors,
    )


# ---------- webhook ----------------------------------------------------------

@router.post("/webhook")
async def webhook(request: Request):
    """GitHub/GitLab-style push webhook. Matches the repo by payload URL + branch,
    verifies the per-repo HMAC over the raw body, and reconciles in the background."""
    raw = await request.body()
    import json as _json
    try:
        payload = _json.loads(raw or b"{}")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")

    # repo url: GitHub `repository.clone_url|html_url`; GitLab `project.git_http_url`.
    repo_obj = payload.get("repository") or payload.get("project") or {}
    url = (
        repo_obj.get("clone_url") or repo_obj.get("html_url")
        or repo_obj.get("git_http_url") or repo_obj.get("url") or ""
    )
    ref = payload.get("ref") or ""
    branch = ref.split("/", 2)[-1] if ref.startswith("refs/heads/") else None
    norm = _normalize_git_url(url)
    if not norm:
        raise HTTPException(status_code=400, detail="could not determine repository from payload")

    async with session_factory()() as session:
        candidates = (await session.execute(select(GitopsRepo).where(GitopsRepo.enabled == True))).scalars().all()  # noqa: E712
        matched: Optional[GitopsRepo] = None
        for r in candidates:
            if _normalize_git_url(r.url) == norm and (branch is None or (r.branch or "main") == branch):
                matched = r
                break
        if matched is None:
            raise HTTPException(status_code=404, detail="no enabled repo matches this push")

        # verify signature
        sig = (
            request.headers.get("x-hub-signature-256")
            or request.headers.get("X-Hub-Signature-256")
            or ""
        )
        if matched.webhook_secret_enc:
            secret = crypto.decrypt(matched.webhook_secret_enc)
            expected = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
            if not (sig and hmac.compare_digest(sig, expected)):
                raise HTTPException(status_code=401, detail="invalid or missing signature")
        repo_id = matched.id

    # reconcile off the request path so the webhook returns immediately
    _spawn_background_sync(request.app, repo_id)
    return {"ok": True, "repo_id": repo_id, "queued": True}


# ---------- background sync + poll loop --------------------------------------

_bg_tasks: set[asyncio.Task] = set()


def _spawn_background_sync(app: Any, repo_id: str, *, force: bool = True) -> None:
    task = asyncio.create_task(_run_sync(app, repo_id, force=force))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def _run_sync(app: Any, repo_id: str, *, force: bool) -> None:
    try:
        async with session_factory()() as session:
            repo = await session.get(GitopsRepo, repo_id)
            if repo is not None:
                await gitops_engine.reconcile_repo(app, session, repo, force=force)
    except Exception:
        logger.exception("gitops background sync %s failed", repo_id)


def _is_due(repo: GitopsRepo, now: datetime) -> bool:
    if not repo.enabled:
        return False
    last = repo.last_sync_at
    if last is None:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (now - last).total_seconds() >= max(MIN_POLL_INTERVAL, repo.poll_interval)


async def gitops_reconcile_loop(app: Any) -> None:
    """Background loop: reconcile every enabled repo whose poll_interval elapsed.
    Disable with GITOPS_POLL=0 (manual sync + webhook still work)."""
    if os.environ.get("GITOPS_POLL", "1") == "0":
        logger.info("gitops poll loop disabled (GITOPS_POLL=0)")
        return
    await asyncio.sleep(10)  # let startup settle
    logger.info("gitops poll loop started (tick=%ss)", POLL_TICK_S)
    while True:
        try:
            now = datetime.now(timezone.utc)
            async with session_factory()() as session:
                repos = (await session.execute(select(GitopsRepo).where(GitopsRepo.enabled == True))).scalars().all()  # noqa: E712
                due = [r.id for r in repos if _is_due(r, now)]
            for rid in due:
                async with session_factory()() as session:
                    repo = await session.get(GitopsRepo, rid)
                    if repo is not None:
                        await gitops_engine.reconcile_repo(app, session, repo, force=False)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("gitops poll loop tick failed")
        try:
            await asyncio.sleep(POLL_TICK_S)
        except asyncio.CancelledError:
            break
