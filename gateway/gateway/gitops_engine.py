"""GitOps reconcile engine.

Declares platform resources (apps, storage, datasets, providers, benchmarks,
training runs) in a git repo as YAML manifests and reconciles the live state to
match. Git is the source of truth:

- a manifest with no ledger row  -> create the resource
- a manifest whose spec changed  -> update (or delete+recreate where there's no
  PATCH endpoint: app/provider), EXCEPT jobs (benchmark/training_run) which are
  submit-once-by-name and only resubmit when `generation` increases
- a ledger row with no manifest   -> prune (delete the resource) when repo.prune

The engine calls the EXISTING create/update/delete handler coroutines in-process
(synthesizing the repo owner as `user`, a DB `session`, and a `SimpleNamespace(app=app)`
request shim) so all validation, Fernet encryption and pod-spawn side effects are
reused — nothing is duplicated.

Secrets are never stored in git. Manifests reference `GlobalEnv` keys by name via
`*_secret` fields; the engine resolves them in-memory at apply time. Cross-resource
references (`provider_id`, `storage_id`, `dataset_id`, …) may name another manifest in
the same repo and are resolved to the live platform id.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets as _secrets
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import GitopsRepo, GitopsResource, User
from .global_env_api import load_global_env

logger = logging.getLogger("gateway.gitops")


# ---------- kind registry ----------------------------------------------------

@dataclass(frozen=True)
class _KindSpec:
    canonical: str
    aliases: tuple[str, ...]
    is_job: bool      # submit-once-by-name semantics (no in-place update)
    has_patch: bool   # an in-place PATCH update handler exists
    order: int        # apply order (lower first); prune runs in reverse


_KINDS: tuple[_KindSpec, ...] = (
    _KindSpec("provider",     ("provider", "gpuprovider"),                         False, False, 0),
    _KindSpec("storage",      ("storage",),                                        False, True,  1),
    _KindSpec("dataset",      ("dataset",),                                        False, True,  2),
    _KindSpec("app",          ("app", "serverlessapp", "serverless"),              False, False, 3),
    _KindSpec("benchmark",    ("benchmark",),                                      True,  False, 4),
    _KindSpec("training_run", ("trainingrun", "training_run", "training", "autotrain"), True, False, 5),
)
_ALIAS_TO_KIND: dict[str, _KindSpec] = {a: k for k in _KINDS for a in k.aliases}
_CANON: dict[str, _KindSpec] = {k.canonical: k for k in _KINDS}

# Cross-reference fields → the manifest kind they point at. A value matching a
# managed manifest name is rewritten to that resource's live platform id.
_REF_FIELDS: dict[str, str] = {
    "provider_id": "provider",
    "storage_id": "storage",
    "dataset_id": "dataset",
    "test_dataset_id": "dataset",
    "pack_source_dataset_id": "dataset",
}

# Reserved top-level manifest keys that are NOT part of the resource spec.
_RESERVED = {"kind", "name", "generation", "apiversion", "apiVersion", "metadata"}


# ---------- parsed manifest + reconcile result -------------------------------

@dataclass
class Manifest:
    kind: str          # canonical kind
    name: str
    generation: int
    spec: dict
    source: str        # file path (for error messages)


@dataclass
class ReconcileResult:
    sha: Optional[str] = None
    skipped: bool = False          # nothing to do (sha unchanged, not forced)
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    unchanged: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------- git -------------------------------------------------------------

def _work_root() -> Path:
    root = os.environ.get("GITOPS_WORK_DIR", "").strip()
    if not root:
        root = os.path.join(tempfile.gettempdir(), "sgpu-gitops")
    return Path(root)


def _auth_url(url: str, token: Optional[str]) -> str:
    """Inject a token into an https URL for private clone/fetch. The token lands
    in argv (visible via ps on this host) — acceptable for a single-tenant
    gateway; use a public repo or a short-lived deploy token to limit exposure."""
    if not token or not url.startswith("https://"):
        return url
    return "https://" + token + "@" + url[len("https://"):]


def _sanitize(msg: str, token: Optional[str]) -> str:
    if token:
        msg = msg.replace(token, "***")
    return msg


async def _git(cwd: Optional[str], *args: str, timeout: float = 120.0) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "true"},
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("git command timed out")
    return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


async def fetch_repo(repo: GitopsRepo, token: Optional[str]) -> tuple[Path, str]:
    """Clone (first time) or fetch+hard-reset the repo into a persistent work dir.
    Returns (checkout_dir, HEAD sha). Raises RuntimeError with a token-sanitized
    message on failure."""
    work = _work_root() / repo.id
    work.parent.mkdir(parents=True, exist_ok=True)
    url = _auth_url(repo.url, token)
    branch = (repo.branch or "main").strip() or "main"

    if not (work / ".git").exists():
        if work.exists():
            # stale non-git dir — start clean
            import shutil
            shutil.rmtree(work, ignore_errors=True)
        rc, _, err = await _git(None, "clone", "--depth", "1", "--branch", branch, url, str(work))
        if rc != 0:
            raise RuntimeError(_sanitize(f"git clone failed: {err.strip()}", token))
    else:
        # Point the remote at the (possibly re-tokenized) url, then fetch+reset.
        await _git(str(work), "remote", "set-url", "origin", url)
        rc, _, err = await _git(str(work), "fetch", "--depth", "1", "origin", branch)
        if rc != 0:
            raise RuntimeError(_sanitize(f"git fetch failed: {err.strip()}", token))
        rc, _, err = await _git(str(work), "reset", "--hard", "FETCH_HEAD")
        if rc != 0:
            raise RuntimeError(_sanitize(f"git reset failed: {err.strip()}", token))
        await _git(str(work), "clean", "-fdx")

    rc, out, err = await _git(str(work), "rev-parse", "HEAD")
    if rc != 0:
        raise RuntimeError(_sanitize(f"git rev-parse failed: {err.strip()}", token))
    return work, out.strip()


async def test_remote(url: str, branch: str, token: Optional[str]) -> str:
    """Lightweight connectivity/auth/branch check via `git ls-remote` — no clone,
    no disk. Returns the branch's HEAD sha on success; raises RuntimeError (with
    the token redacted) if the repo is unreachable, the token is wrong, or the
    branch doesn't exist."""
    branch = (branch or "main").strip() or "main"
    auth = _auth_url(url, token)
    rc, out, err = await _git(None, "ls-remote", "--heads", auth, branch, timeout=30.0)
    if rc != 0:
        raise RuntimeError(_sanitize(f"git ls-remote failed: {err.strip()}", token))
    out = out.strip()
    if not out:
        raise RuntimeError(f"branch '{branch}' not found in the repository")
    return out.split()[0]  # "<sha>\trefs/heads/<branch>"


# ---------- manifest parsing -------------------------------------------------

def parse_manifests(root: Path) -> tuple[list[Manifest], list[str]]:
    """Recursively load every *.yaml/*.yml doc under `root`. Returns
    (manifests, errors). One bad doc never aborts the others."""
    manifests: list[Manifest] = []
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    if not root.exists():
        return manifests, [f"path '{root.name}' not found in repo"]
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".yaml", ".yml") and p.is_file())
    for f in files:
        rel = str(f.relative_to(root))
        try:
            docs = list(yaml.safe_load_all(f.read_text()))
        except Exception as e:
            errors.append(f"{rel}: YAML parse error: {e}")
            continue
        for doc in docs:
            if doc is None:
                continue
            if not isinstance(doc, dict):
                errors.append(f"{rel}: each document must be a mapping")
                continue
            raw_kind = str(doc.get("kind", "")).strip().lower()
            kspec = _ALIAS_TO_KIND.get(raw_kind)
            if kspec is None:
                errors.append(f"{rel}: unknown or missing kind '{doc.get('kind')}'")
                continue
            meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
            name = str(doc.get("name") or meta.get("name") or "").strip()
            if not name:
                errors.append(f"{rel}: {kspec.canonical} is missing 'name'")
                continue
            key = (kspec.canonical, name)
            if key in seen:
                errors.append(f"{rel}: duplicate {kspec.canonical}/{name}")
                continue
            seen.add(key)
            spec = doc.get("spec")
            if not isinstance(spec, dict):
                spec = {k: v for k, v in doc.items() if k not in _RESERVED}
            try:
                generation = int(doc.get("generation", 1))
            except (TypeError, ValueError):
                generation = 1
            manifests.append(Manifest(kspec.canonical, name, generation, spec, rel))
    return manifests, errors


# ---------- spec resolution + hashing ---------------------------------------

def _resolve_refs(spec: dict, name_index: dict[tuple[str, str], str]) -> dict:
    """Rewrite id-bearing fields that name a managed manifest into its live id.
    A value with no matching manifest is left as a literal id."""
    out = json.loads(json.dumps(spec))  # deep copy
    for fld, target_kind in _REF_FIELDS.items():
        val = out.get(fld)
        if isinstance(val, str) and val:
            resolved = name_index.get((target_kind, val))
            if resolved:
                out[fld] = resolved
    return out


def spec_hash(kind: str, name: str, generation: int, resolved_spec: dict) -> str:
    """Stable hash of the cross-ref-resolved spec (secret names, not values).
    Changing a referenced resource's id changes the resolved spec → re-apply."""
    blob = json.dumps(
        {"kind": kind, "name": name, "generation": generation, "spec": resolved_spec},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _take_secret(d: dict, ref_key: str, dest_key: str, env: dict[str, str]) -> None:
    """Pop `d[ref_key]` (a GlobalEnv key name); if set, resolve it and write the
    plaintext into `d[dest_key]`. Raises ValueError if the secret is missing."""
    ref = d.pop(ref_key, None)
    if ref is None:
        return
    ref = str(ref).strip()
    if not ref:
        return
    if ref not in env:
        raise ValueError(f"global secret '{ref}' (for {ref_key}) is not set")
    d[dest_key] = env[ref]


def _resolve_secrets(kind: str, spec: dict, env: dict[str, str]) -> dict:
    """Replace `*_secret` references with the GlobalEnv value, injected into the
    field the create handler expects. Returns a copy; never mutates `spec`.
    Raises ValueError if a referenced secret is missing."""
    out = json.loads(json.dumps(spec))
    if kind == "provider":
        if isinstance(out.get("vm"), dict):
            _take_secret(out["vm"], "private_key_secret", "private_key", env)
        if isinstance(out.get("api"), dict):
            _take_secret(out["api"], "api_key_secret", "api_key", env)
    elif kind == "storage":
        _take_secret(out, "access_key_id_secret", "access_key_id", env)
        _take_secret(out, "secret_access_key_secret", "secret_access_key", env)
        # hf_token_secret / label_token_secret are real model fields — pass through.
    return out


# ---------- in-process resource handlers ------------------------------------

def _shim(app: Any):
    from types import SimpleNamespace
    return SimpleNamespace(app=app)


def _build_request(kind: str, name: str, spec: dict):
    # The manifest `name` is a top-level field, not part of `spec`; every
    # Create*Request requires it. The manifest name always wins.
    spec = {**spec, "name": name}
    if kind == "app":
        from .main import CreateAppRequest
        return CreateAppRequest(**spec)
    if kind == "storage":
        from .storage_api import CreateStorageRequest
        return CreateStorageRequest(**spec)
    if kind == "dataset":
        from .datasets_api import CreateDatasetRequest
        return CreateDatasetRequest(**spec)
    if kind == "provider":
        from .providers_api import CreateProviderRequest
        return CreateProviderRequest(**spec)
    if kind == "benchmark":
        from .bench import CreateBenchmarkRequest
        return CreateBenchmarkRequest(**spec)
    if kind == "training_run":
        from .training_api import CreateTrainingRunRequest
        return CreateTrainingRunRequest(**spec)
    raise ValueError(f"unknown kind {kind}")


async def _create(kind: str, req, *, app, user: User, session: AsyncSession) -> str:
    """Call the existing create handler; return the new platform id."""
    if kind == "app":
        from .main import create_app
        resp = await create_app(req, _shim(app), user=user, session=session)
        return resp.app_id
    if kind == "storage":
        from .storage_api import create_storage
        return (await create_storage(req, user=user, session=session)).id
    if kind == "dataset":
        from .datasets_api import create_dataset
        return (await create_dataset(req, user=user, session=session)).id
    if kind == "provider":
        from .providers_api import create_provider
        return (await create_provider(req, user=user, session=session)).id
    if kind == "benchmark":
        from .bench import create_benchmark
        return (await create_benchmark(req, _shim(app), user=user, session=session)).id
    if kind == "training_run":
        from .training_api import create_training_run
        return (await create_training_run(req, _shim(app), user=user, session=session)).id
    raise ValueError(f"unknown kind {kind}")


async def _update(kind: str, resource_id: str, spec: dict, *, app, user: User, session: AsyncSession) -> str:
    """In-place PATCH for kinds that support it (storage, dataset). Returns the id."""
    if kind == "storage":
        from .storage_api import update_storage, UpdateStorageRequest
        await update_storage(resource_id, UpdateStorageRequest(**spec), user=user, session=session)
        return resource_id
    if kind == "dataset":
        from .datasets_api import update_dataset, UpdateDatasetRequest
        # UpdateDatasetRequest only accepts a subset of fields; ignore the rest.
        allowed = set(UpdateDatasetRequest.model_fields.keys())
        await update_dataset(resource_id, UpdateDatasetRequest(**{k: v for k, v in spec.items() if k in allowed}),
                             user=user, session=session)
        return resource_id
    raise ValueError(f"{kind} has no in-place update")


async def _delete(kind: str, resource_id: str, *, app, user: User, session: AsyncSession) -> None:
    if kind == "app":
        from .main import delete_app
        await delete_app(resource_id, _shim(app), user=user, session=session)
        return
    if kind == "storage":
        from .storage_api import delete_storage
        await delete_storage(resource_id, user=user, session=session)
        return
    if kind == "dataset":
        from .datasets_api import delete_dataset
        await delete_dataset(resource_id, user=user, session=session)
        return
    if kind == "provider":
        from .providers_api import delete_provider
        await delete_provider(resource_id, user=user, session=session)
        return
    if kind == "benchmark":
        from .bench import delete_benchmark
        await delete_benchmark(resource_id, user=user, session=session)
        return
    if kind == "training_run":
        from .training_api import delete_training_run
        await delete_training_run(resource_id, user=user, session=session)
        return
    raise ValueError(f"unknown kind {kind}")


def _err(e: Exception) -> str:
    from fastapi import HTTPException
    if isinstance(e, HTTPException):
        d = e.detail
        if isinstance(d, dict):
            return str(d.get("error") or d)
        return str(d)
    return f"{type(e).__name__}: {e}"


# ---------- ledger -----------------------------------------------------------

@dataclass
class _Row:
    id: str
    kind: str
    name: str
    resource_id: Optional[str]
    generation: int
    spec_hash: Optional[str]
    status: str


async def _load_ledger(session: AsyncSession, repo_id: str) -> dict[tuple[str, str], _Row]:
    rows = (await session.execute(
        select(GitopsResource).where(GitopsResource.repo_id == repo_id)
    )).scalars().all()
    return {
        (r.kind, r.name): _Row(r.id, r.kind, r.name, r.resource_id, r.generation, r.spec_hash, r.status)
        for r in rows
    }


async def _ledger_upsert(session: AsyncSession, repo_id: str, kind: str, name: str, *,
                         resource_id: Optional[str], generation: int,
                         spec_hash: Optional[str], status: str, error: Optional[str]) -> None:
    row = (await session.execute(
        select(GitopsResource).where(
            GitopsResource.repo_id == repo_id,
            GitopsResource.kind == kind,
            GitopsResource.name == name,
        )
    )).scalar_one_or_none()
    if row is None:
        row = GitopsResource(id=f"gres-{_secrets.token_hex(4)}", repo_id=repo_id, kind=kind, name=name)
        session.add(row)
    row.resource_id = resource_id
    row.generation = generation
    row.spec_hash = spec_hash
    row.status = status
    row.error = error
    row.last_synced_at = datetime.now(timezone.utc)
    await session.commit()


async def _ledger_delete(session: AsyncSession, repo_id: str, kind: str, name: str) -> None:
    row = (await session.execute(
        select(GitopsResource).where(
            GitopsResource.repo_id == repo_id,
            GitopsResource.kind == kind,
            GitopsResource.name == name,
        )
    )).scalar_one_or_none()
    if row is not None:
        await session.delete(row)
        await session.commit()


# ---------- reconcile --------------------------------------------------------

async def resolve_token(session: AsyncSession, repo: GitopsRepo) -> Optional[str]:
    if not repo.token_secret:
        return None
    env = await load_global_env(session)
    return env.get(repo.token_secret)


async def reconcile_repo(app: Any, session: AsyncSession, repo: GitopsRepo, *, force: bool = False) -> ReconcileResult:
    """Fetch the repo and reconcile every manifest. `session` is shared with the
    in-process handlers (which commit themselves); per-doc failures are isolated
    via rollback and recorded in the ledger."""
    res = ReconcileResult()
    repo_id = repo.id

    repo.last_sync_status = "syncing"
    await session.commit()

    # ---- fetch ----
    try:
        token = await resolve_token(session, repo)
        work, sha = await fetch_repo(repo, token)
    except Exception as e:
        msg = _err(e)
        res.errors.append(msg)
        await _finalize(session, repo_id, None, "error", msg)
        return res
    res.sha = sha

    if sha == repo.last_synced_sha and not force:
        res.skipped = True
        await _finalize(session, repo_id, sha, "ok", None)
        return res

    # ---- parse ----
    scan_root = work
    if repo.path and repo.path.strip().strip("/"):
        scan_root = work / repo.path.strip().strip("/")
    manifests, parse_errors = parse_manifests(scan_root)
    res.errors.extend(parse_errors)

    ledger = await _load_ledger(session, repo_id)
    env = await load_global_env(session)
    owner = await session.get(User, repo.owner_id)
    if owner is None:
        msg = "repo owner no longer exists"
        res.errors.append(msg)
        await _finalize(session, repo_id, sha, "error", msg)
        return res

    name_index: dict[tuple[str, str], str] = {
        (r.kind, r.name): r.resource_id for r in ledger.values() if r.resource_id
    }
    declared: set[tuple[str, str]] = set()

    # ---- apply (dependency order) ----
    for m in sorted(manifests, key=lambda x: _CANON[x.kind].order):
        declared.add((m.kind, m.name))
        kspec = _CANON[m.kind]
        label = f"{m.kind}/{m.name}"
        try:
            resolved_spec = _resolve_refs(m.spec, name_index)
            h = spec_hash(m.kind, m.name, m.generation, resolved_spec)
            req_spec = _resolve_secrets(m.kind, resolved_spec, env)
            row = ledger.get((m.kind, m.name))

            if row is None or row.resource_id is None or row.status == "error":
                # never created (or last attempt failed) → create
                req = _build_request(m.kind, m.name, req_spec)
                rid = await _create(m.kind, req, app=app, user=owner, session=session)
                name_index[(m.kind, m.name)] = rid
                await _ledger_upsert(session, repo_id, m.kind, m.name,
                                     resource_id=rid, generation=m.generation,
                                     spec_hash=h, status="applied", error=None)
                res.created.append(label)
            elif kspec.is_job:
                # submit-once-by-name: only resubmit when generation increases
                if m.generation > row.generation:
                    try:
                        await _delete(m.kind, row.resource_id, app=app, user=owner, session=session)
                    except Exception:
                        await session.rollback()  # old run may already be gone
                    req = _build_request(m.kind, m.name, req_spec)
                    rid = await _create(m.kind, req, app=app, user=owner, session=session)
                    name_index[(m.kind, m.name)] = rid
                    await _ledger_upsert(session, repo_id, m.kind, m.name,
                                         resource_id=rid, generation=m.generation,
                                         spec_hash=h, status="applied", error=None)
                    res.updated.append(label)
                else:
                    name_index[(m.kind, m.name)] = row.resource_id
                    res.unchanged += 1
            elif h != row.spec_hash:
                if kspec.has_patch:
                    rid = await _update(m.kind, row.resource_id, req_spec, app=app, user=owner, session=session)
                else:
                    # no PATCH endpoint (app/provider) → delete + recreate
                    await _delete(m.kind, row.resource_id, app=app, user=owner, session=session)
                    req = _build_request(m.kind, m.name, req_spec)
                    rid = await _create(m.kind, req, app=app, user=owner, session=session)
                name_index[(m.kind, m.name)] = rid
                await _ledger_upsert(session, repo_id, m.kind, m.name,
                                     resource_id=rid, generation=m.generation,
                                     spec_hash=h, status="applied", error=None)
                res.updated.append(label)
            else:
                name_index[(m.kind, m.name)] = row.resource_id
                res.unchanged += 1
        except Exception as e:
            await session.rollback()
            msg = _err(e)
            res.errors.append(f"{label}: {msg}")
            logger.warning("gitops %s: apply %s failed: %s", repo_id, label, msg)
            try:
                existing = ledger.get((m.kind, m.name))
                await _ledger_upsert(
                    session, repo_id, m.kind, m.name,
                    resource_id=existing.resource_id if existing else None,
                    generation=existing.generation if existing else m.generation,
                    spec_hash=existing.spec_hash if existing else None,
                    status="error", error=msg,
                )
            except Exception:
                await session.rollback()

    # ---- prune (reverse dependency order) ----
    if repo.prune:
        stale = [(k, n) for (k, n) in ledger if (k, n) not in declared]
        stale.sort(key=lambda kn: _CANON[kn[0]].order, reverse=True)
        for (kind, name) in stale:
            row = ledger[(kind, name)]
            label = f"{kind}/{name}"
            try:
                if row.resource_id:
                    await _delete(kind, row.resource_id, app=app, user=owner, session=session)
                await _ledger_delete(session, repo_id, kind, name)
                res.pruned.append(label)
            except Exception as e:
                await session.rollback()
                msg = _err(e)
                res.errors.append(f"prune {label}: {msg}")
                logger.warning("gitops %s: prune %s failed: %s", repo_id, label, msg)

    status = "ok" if res.ok else "error"
    err_text = None if res.ok else "\n".join(res.errors[:20])
    await _finalize(session, repo_id, sha, status, err_text)
    logger.info(
        "gitops %s reconciled @ %s: +%d ~%d -%d =%d (%d errors)",
        repo_id, sha[:8] if sha else "?", len(res.created), len(res.updated),
        len(res.pruned), res.unchanged, len(res.errors),
    )
    return res


async def _finalize(session: AsyncSession, repo_id: str, sha: Optional[str], status: str, error: Optional[str]) -> None:
    """Write terminal sync status back to the repo row (re-fetched fresh)."""
    repo = await session.get(GitopsRepo, repo_id)
    if repo is None:
        return
    if sha is not None and status == "ok":
        repo.last_synced_sha = sha
    repo.last_sync_status = status
    repo.last_sync_error = error
    repo.last_sync_at = datetime.now(timezone.utc)
    await session.commit()
