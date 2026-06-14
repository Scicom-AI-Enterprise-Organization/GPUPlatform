"""HuggingFace Hub–compatible mirror, mounted at `/hf`.

Lets users host their own models/datasets in GPUPlatform and use standard HF
tooling against the platform:

    export HF_ENDPOINT=http://<gateway>:8080/hf
    export HF_TOKEN=sgpu_...              # a platform API key
    huggingface_hub.snapshot_download("ns/name")          # read
    model.push_to_hub("ns/name")  /  hf upload ns/name .  # write

The bytes live in a `Storage` backend (s3 / local / sftp) via
`storage_backends.resolve_backend`; the file list of the single `main` revision
is the `CatalogRepo.manifest`. Regular (git) files are stored at
`{prefix}/{path}`; large files use Git-LFS and are stored content-addressed at
`{prefix}/.hf-lfs/{oid}`.

Auth: every route (except the LFS upload PUT) authenticates the
`Authorization: Bearer sgpu_…` header via the normal API-key path (`current_user`).
The LFS single-part PUT carries **no** auth header (huggingface_hub uploads it
bare), so its URL is signed with a short-lived HMAC token minted during the
authenticated `…/objects/batch` call.

Scope is deliberately basic: only `main`, no branches/tags/PRs, no pull-through
caching of huggingface.co, no Xet (clients fall back to plain HTTP).

Protocol verified against huggingface_hub 1.17.0 (file_download.py, hf_api.py,
_commit_api.py, lfs.py).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
import time
from datetime import timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import storage_backends as sb
from .auth import current_user
from .db import CatalogRepo, Storage, User, get_session

logger = logging.getLogger("gateway.hf_mirror")

router = APIRouter(prefix="/hf", tags=["hf-mirror"])

# Files at/above this size, or matching a binary extension, go through Git-LFS.
LFS_THRESHOLD = 10 * 1024 * 1024  # 10 MiB
LFS_EXTS = {
    ".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".h5", ".gguf", ".ggml",
    ".onnx", ".msgpack", ".npy", ".npz", ".tar", ".gz", ".tgz", ".zip", ".7z",
    ".parquet", ".arrow", ".pickle", ".pkl", ".model", ".bin.index", ".pb",
    ".tflite", ".wasm", ".mlmodel", ".ot", ".joblib", ".rar",
}
# How long a minted LFS upload URL stays valid.
LFS_TOKEN_TTL_S = 6 * 3600

CHUNK = sb.CHUNK


# ---------- helpers -----------------------------------------------------


def _rt(rtype: str) -> str:
    """`models`|`datasets` (URL plural) → `model`|`dataset`. 404 otherwise."""
    if rtype == "models":
        return "model"
    if rtype == "datasets":
        return "dataset"
    raise HTTPException(status_code=404, detail="unknown repo type")


def _url_prefix(repo_type: str) -> str:
    """URL segment prefix HF uses per repo type (`""` for models, `datasets/`)."""
    return "datasets/" if repo_type == "dataset" else ""


# huggingface_hub's `fix_hf_endpoint_in_url` rewrites this exact host in any URL
# we return to the client's configured HF_ENDPOINT. Emitting client-facing URLs
# (LFS upload href, commit/repo URLs) with this host means they resolve correctly
# whether the client hits the gateway directly OR through the web proxy
# (`<origin>/api/proxy/hf`) — the gateway never needs to know its external URL.
HF_DEFAULT_HOST = "https://huggingface.co"


def _client_url(repo_type: str, suffix: str) -> str:
    """Build a client-facing URL rooted at the HF default host (NOT our `/hf`
    mount). `fix_hf_endpoint_in_url` swaps the host for the client's HF_ENDPOINT
    — which already includes the `/hf` mount (direct) or `/api/proxy/hf` (web
    proxy) — so we must NOT add `/hf` here or it doubles up. `suffix` is the
    huggingface.co-style path, e.g. `ns/name.git/lfs/<oid>`."""
    return f"{HF_DEFAULT_HOST}/{_url_prefix(repo_type)}{suffix}"


def _secret() -> bytes:
    return (os.environ.get("PROVIDER_SECRET_KEY") or "sgpu-dev-secret").encode()


def _sign_lfs(repo_id: str, oid: str) -> str:
    exp = int(time.time()) + LFS_TOKEN_TTL_S
    msg = f"{repo_id}:{oid}:{exp}"
    sig = hmac.new(_secret(), msg.encode(), hashlib.sha256).hexdigest()
    raw = f"{repo_id}|{oid}|{exp}|{sig}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def _verify_lfs(token: str, oid: str) -> Optional[str]:
    """Return the repo_id the token authorizes for `oid`, or None if invalid."""
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        repo_id, t_oid, exp_s, sig = raw.split("|")
        exp = int(exp_s)
    except Exception:  # noqa: BLE001
        return None
    if t_oid != oid or time.time() > exp:
        return None
    expect = hmac.new(_secret(), f"{repo_id}:{oid}:{exp}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expect, sig):
        return None
    return repo_id


def _is_lfs(path: str, size: int) -> bool:
    if size >= LFS_THRESHOLD:
        return True
    low = path.lower()
    return any(low.endswith(ext) for ext in LFS_EXTS)


def _compute_sha(manifest: list[dict]) -> str:
    """Synthetic commit id: sha1 over the sorted (path, oid) manifest."""
    items = sorted(((e.get("path"), e.get("oid"), e.get("size")) for e in manifest))
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()


def _hf_dt(dt) -> Optional[str]:
    """HF's parse_datetime is strict: `%Y-%m-%dT%H:%M:%S.%fZ` (UTC, trailing Z).
    `.isoformat()` emits `+00:00` which it rejects, so format explicitly."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _manifest(repo: CatalogRepo) -> list[dict]:
    return list(repo.manifest or [])


def _find(repo: CatalogRepo, path: str) -> Optional[dict]:
    for e in repo.manifest or []:
        if e.get("path") == path:
            return e
    return None


def _blob_key(repo: CatalogRepo, entry: dict) -> str:
    """Storage key for a manifest entry (lfs → content-addressed, else by path)."""
    if entry.get("lfs"):
        return f"{repo.prefix}/.hf-lfs/{entry['oid']}"
    return f"{repo.prefix}/{entry['path']}"


async def _get_repo(session: AsyncSession, repo_type: str, ns: str, name: str) -> Optional[CatalogRepo]:
    name = name[:-4] if name.endswith(".git") else name  # tolerate `name.git` from LFS paths
    res = await session.execute(
        select(CatalogRepo).where(
            CatalogRepo.repo_type == repo_type,
            CatalogRepo.namespace == ns,
            CatalogRepo.name == name,
        )
    )
    return res.scalar_one_or_none()


async def _backend_for(session: AsyncSession, repo: CatalogRepo) -> sb.StorageBackend:
    row = await session.get(Storage, repo.storage_id) if repo.storage_id else None
    if row is None:
        raise HTTPException(status_code=500, detail="repo storage missing or unconfigured")
    try:
        return await run_in_threadpool(sb.resolve_backend, row)
    except sb.StorageError as e:
        raise HTTPException(status_code=500, detail=f"storage error: {e}") from e


def _can_read(repo: CatalogRepo, user: User) -> bool:
    return (not repo.private) or repo.owner_id == user.id or bool(user.is_admin)


def _require_owner(repo: CatalogRepo, user: User) -> None:
    if repo.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="no write permission on this repo")


def _repo_not_found(repo_type: str, ns: str, name: str):
    # 404 shape HF maps to RepositoryNotFoundError.
    return HTTPException(status_code=404, detail=f"{repo_type} {ns}/{name} not found")


# ---------- whoami ------------------------------------------------------


@router.get("/api/whoami-v2")
async def whoami_v2(user: User = Depends(current_user)):
    # push_to_hub reads ["name"] to resolve the default namespace.
    return {
        "name": user.username,
        "fullname": user.username,
        "email": user.email,
        "type": "user",
        "orgs": [],
        # `hf auth login` reads auth.accessToken.{role,displayName}.
        "auth": {
            "type": "access_token",
            "accessToken": {"displayName": "gpuplatform-api-key", "role": "write"},
        },
    }


@router.post("/api/validate-yaml")
async def validate_yaml(payload: dict, user: User = Depends(current_user)):  # noqa: ARG001
    # `hf upload` validates README.md front-matter before committing. We accept
    # anything (basic mirror) — return no errors/warnings so the upload proceeds.
    return {"errors": [], "warnings": []}


# ---------- repo info (enumerate files for download) --------------------


async def _repo_info_impl(rtype: str, ns: str, name: str, user: User, session: AsyncSession):
    repo_type = _rt(rtype)
    repo = await _get_repo(session, repo_type, ns, name)
    if repo is None or not _can_read(repo, user):
        raise _repo_not_found(repo_type, ns, name)
    manifest = _manifest(repo)
    sha = repo.sha or _compute_sha(manifest)
    last = _hf_dt(repo.updated_at)
    siblings = [{"rfilename": e["path"], "size": e.get("size")} for e in manifest]
    info = {
        "_id": repo.id,
        "id": repo.full_id,
        "author": ns,
        "sha": sha,
        "lastModified": last,
        "createdAt": _hf_dt(repo.created_at) or last,
        "private": bool(repo.private),
        "disabled": False,
        "gated": False,
        "downloads": 0,
        "likes": 0,
        "tags": [],
        "siblings": siblings,
    }
    if repo_type == "model":
        info["modelId"] = repo.full_id
        info["pipeline_tag"] = None
    return info


@router.get("/api/{rtype}/{ns}/{name}")
async def repo_info(rtype: str, ns: str, name: str,
                    user: User = Depends(current_user),
                    session: AsyncSession = Depends(get_session)):
    return await _repo_info_impl(rtype, ns, name, user, session)


@router.get("/api/{rtype}/{ns}/{name}/revision/{revision:path}")
async def repo_info_rev(rtype: str, ns: str, name: str, revision: str,  # noqa: ARG001 — single revision
                        user: User = Depends(current_user),
                        session: AsyncSession = Depends(get_session)):
    return await _repo_info_impl(rtype, ns, name, user, session)


# ---------- tree (list files under a path) — used by Dataset.push_to_hub ------


def _tree_file_entry(e: dict) -> dict:
    """A tree `file` entry in the shape huggingface_hub's RepoFile(**) expects."""
    entry = {"type": "file", "path": e["path"], "size": int(e.get("size") or 0),
             "oid": e.get("oid") or ""}
    if e.get("lfs"):
        entry["lfs"] = {"size": int(e.get("size") or 0), "oid": e.get("oid") or "", "pointerSize": 0}
    return entry


async def _tree_impl(rtype: str, ns: str, name: str, path: str, recursive: bool,
                     user: User, session: AsyncSession):
    repo_type = _rt(rtype)
    repo = await _get_repo(session, repo_type, ns, name)
    if repo is None or not _can_read(repo, user):
        raise _repo_not_found(repo_type, ns, name)
    prefix = (path or "").strip("/")
    manifest = _manifest(repo)
    # A non-root path that matches no file/dir must 404 with EntryNotFound — HF
    # clients (HfFileSystem.find → glob) rely on that to treat it as "empty".
    if prefix and not any(p == prefix or p.startswith(prefix + "/") for p in (e["path"] for e in manifest)):
        raise HTTPException(status_code=404, detail=f"{path} not found",
                            headers={"X-Error-Code": "EntryNotFound"})
    out: list[dict] = []
    dirs: set[str] = set()
    for e in manifest:
        p = e["path"]
        if prefix:
            if p != prefix and not p.startswith(prefix + "/"):
                continue
            rel = p[len(prefix) + 1:] if p.startswith(prefix + "/") else ""
        else:
            rel = p
        if not rel:
            continue
        if recursive or "/" not in rel:
            out.append(_tree_file_entry(e))
        else:  # non-recursive: surface the immediate subdirectory once
            child = f"{prefix}/{rel.split('/', 1)[0]}" if prefix else rel.split("/", 1)[0]
            if child not in dirs:
                dirs.add(child)
                out.append({"type": "directory", "path": child, "oid": ""})
    return JSONResponse(out)


@router.get("/api/{rtype}/{ns}/{name}/tree/{revision}/{path:path}")
async def repo_tree(rtype: str, ns: str, name: str, revision: str, path: str,  # noqa: ARG001
                    recursive: bool = False, expand: bool = False,  # noqa: ARG001 — expand ignored
                    user: User = Depends(current_user),
                    session: AsyncSession = Depends(get_session)):
    return await _tree_impl(rtype, ns, name, path, recursive, user, session)


@router.get("/api/{rtype}/{ns}/{name}/tree/{revision}")
async def repo_tree_root(rtype: str, ns: str, name: str, revision: str,  # noqa: ARG001
                         recursive: bool = False, expand: bool = False,  # noqa: ARG001
                         user: User = Depends(current_user),
                         session: AsyncSession = Depends(get_session)):
    return await _tree_impl(rtype, ns, name, "", recursive, user, session)


# ---------- paths-info — stat specific paths (HfFileSystem; push_to_hub/load_dataset) ----


@router.post("/api/{rtype}/{ns}/{name}/paths-info/{revision:path}")
async def paths_info(rtype: str, ns: str, name: str, revision: str, request: Request,  # noqa: ARG001
                     user: User = Depends(current_user),
                     session: AsyncSession = Depends(get_session)):
    repo_type = _rt(rtype)
    repo = await _get_repo(session, repo_type, ns, name)
    if repo is None or not _can_read(repo, user):
        raise _repo_not_found(repo_type, ns, name)
    # `HfApi.get_paths_info` sends form-encoded `paths` (repeated key); some
    # callers send JSON — accept both.
    paths: list[str] = []
    try:
        form = await request.form()
        paths = [str(p) for p in form.getlist("paths")]
    except Exception:  # noqa: BLE001
        paths = []
    if not paths:
        try:
            body = await request.json()
            paths = body.get("paths") or []
        except Exception:  # noqa: BLE001
            paths = []

    by_path = {e["path"]: e for e in _manifest(repo)}
    out: list[dict] = []
    for raw in paths:
        p = (raw or "").strip("/")
        if p == "":  # repo root → directory
            out.append({"type": "directory", "path": "", "oid": repo.sha or ""})
        elif p in by_path:
            out.append(_tree_file_entry(by_path[p]))
        elif any(mp == p or mp.startswith(p + "/") for mp in by_path):
            out.append({"type": "directory", "path": p, "oid": ""})
        # else: path doesn't exist → omit (HF returns it absent)
    return JSONResponse(out)


# ---------- file resolve (HEAD metadata + GET bytes) --------------------


def _meta_headers(repo: CatalogRepo, entry: dict) -> dict:
    sha = repo.sha or ""
    etag = entry.get("etag") or entry.get("oid") or ""
    return {
        "X-Repo-Commit": sha,
        "ETag": f'"{etag}"',
        "Content-Length": str(int(entry.get("size") or 0)),
        "Accept-Ranges": "bytes",
        "Content-Type": "application/octet-stream",
    }


async def _resolve_head_impl(repo_type: str, ns: str, name: str, path: str,
                             user: User, session: AsyncSession) -> Response:
    repo = await _get_repo(session, repo_type, ns, name)
    if repo is None or not _can_read(repo, user):
        raise _repo_not_found(repo_type, ns, name)
    entry = _find(repo, path)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"{path} not found in {ns}/{name}")
    # lowercase key so Starlette keeps our Content-Length on the empty HEAD body.
    h = _meta_headers(repo, entry)
    headers = {k.lower(): v for k, v in h.items()}
    return Response(status_code=200, headers=headers)


def _parse_range(range_header: Optional[str], size: int) -> Optional[tuple[int, int]]:
    """Parse a single `bytes=start-end` range → (start, end_inclusive). None if absent."""
    if not range_header or not range_header.startswith("bytes="):
        return None
    spec = range_header[len("bytes="):].split(",")[0].strip()
    if "-" not in spec:
        return None
    a, _, b = spec.partition("-")
    if a == "":  # suffix range: bytes=-N
        n = int(b)
        return (max(0, size - n), size - 1)
    start = int(a)
    end = int(b) if b else size - 1
    return (start, min(end, size - 1))


async def _resolve_get_impl(repo_type: str, ns: str, name: str, path: str,
                            request: Request, user: User, session: AsyncSession):
    repo = await _get_repo(session, repo_type, ns, name)
    if repo is None or not _can_read(repo, user):
        raise _repo_not_found(repo_type, ns, name)
    entry = _find(repo, path)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"{path} not found in {ns}/{name}")
    backend = await _backend_for(session, repo)
    key = _blob_key(repo, entry)
    size = int(entry.get("size") or 0)

    rng = _parse_range(request.headers.get("range"), size)
    if rng is not None:
        start, end = rng
        length = max(0, end - start + 1)
        status = 206
    else:
        start, length, end = 0, size, size - 1
        status = 200

    try:
        reader = await run_in_threadpool(backend.open_reader, key, start)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="blob missing in storage")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"storage read error: {e}") from e

    async def _gen():
        remaining = length
        try:
            while remaining > 0:
                chunk = await run_in_threadpool(reader.read, min(CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
        finally:
            await run_in_threadpool(reader.close)

    headers = {
        "X-Repo-Commit": repo.sha or "",
        "ETag": f'"{entry.get("etag") or entry.get("oid") or ""}"',
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(_gen(), status_code=status, headers=headers,
                             media_type="application/octet-stream")


# model resolve
@router.head("/{ns}/{name}/resolve/{revision}/{path:path}")
async def model_resolve_head(ns: str, name: str, revision: str, path: str,  # noqa: ARG001
                             user: User = Depends(current_user),
                             session: AsyncSession = Depends(get_session)):
    return await _resolve_head_impl("model", ns, name, path, user, session)


@router.get("/{ns}/{name}/resolve/{revision}/{path:path}")
async def model_resolve_get(ns: str, name: str, revision: str, path: str, request: Request,  # noqa: ARG001
                            user: User = Depends(current_user),
                            session: AsyncSession = Depends(get_session)):
    return await _resolve_get_impl("model", ns, name, path, request, user, session)


# dataset resolve
@router.head("/datasets/{ns}/{name}/resolve/{revision}/{path:path}")
async def dataset_resolve_head(ns: str, name: str, revision: str, path: str,  # noqa: ARG001
                               user: User = Depends(current_user),
                               session: AsyncSession = Depends(get_session)):
    return await _resolve_head_impl("dataset", ns, name, path, user, session)


@router.get("/datasets/{ns}/{name}/resolve/{revision}/{path:path}")
async def dataset_resolve_get(ns: str, name: str, revision: str, path: str, request: Request,  # noqa: ARG001
                              user: User = Depends(current_user),
                              session: AsyncSession = Depends(get_session)):
    return await _resolve_get_impl("dataset", ns, name, path, request, user, session)


# ---------- write: create / delete repo ---------------------------------


async def _default_storage(session: AsyncSession, user: User) -> Optional[Storage]:
    """Pick a storage to host an auto-created repo: the user's first enabled
    s3/local/sftp storage (any owner falls back to admin-shared)."""
    res = await session.execute(
        select(Storage).where(Storage.enabled == True).order_by(Storage.created_at.asc())  # noqa: E712
    )
    rows = [r for r in res.scalars().all() if r.kind in ("s3", "local", "sftp")]
    mine = [r for r in rows if r.owner_id == user.id]
    return (mine or rows or [None])[0]


@router.post("/api/repos/create")
async def repos_create(payload: dict, request: Request,
                       user: User = Depends(current_user),
                       session: AsyncSession = Depends(get_session)):
    repo_type = (payload.get("type") or "model").lower()
    if repo_type not in ("model", "dataset"):
        raise HTTPException(status_code=400, detail="only model/dataset repos are supported")
    name = (payload.get("name") or "").strip().strip("/")
    org = (payload.get("organization") or "").strip().strip("/")
    if "/" in name and not org:
        org, name = name.split("/", 1)
    ns = org or user.username
    if not name:
        raise HTTPException(status_code=400, detail="repo name is required")
    visibility = payload.get("visibility")
    private = (visibility == "private") if visibility else bool(payload.get("private", True))

    repo = await _get_repo(session, repo_type, ns, name)
    if repo is None:
        store = await _default_storage(session, user)
        if store is None:
            raise HTTPException(
                status_code=400,
                detail="no storage available — register an s3/local/sftp storage first",
            )
        repo = CatalogRepo(
            id=f"repo-{secrets.token_hex(4)}",
            owner_id=user.id,
            repo_type=repo_type,
            namespace=ns,
            name=name,
            full_id=f"{ns}/{name}",
            storage_id=store.id,
            prefix=f"catalog/{ns}/{name}",
            sha=_compute_sha([]),
            private=private,
            manifest=[],
            size_bytes=0,
            num_files=0,
        )
        session.add(repo)
        await session.commit()
        logger.info("hf: created %s repo %s/%s (storage=%s)", repo_type, ns, name, store.id)
    return {
        "url": _client_url(repo_type, f"{ns}/{name}"),
        "name": f"{ns}/{name}",
        "id": repo.id,
        "type": repo_type,
        "private": bool(repo.private),
    }


@router.api_route("/api/repos/delete", methods=["DELETE", "POST"])
async def repos_delete(payload: dict,
                       user: User = Depends(current_user),
                       session: AsyncSession = Depends(get_session)):
    repo_type = (payload.get("type") or "model").lower()
    name = (payload.get("name") or "").strip().strip("/")
    org = (payload.get("organization") or "").strip().strip("/")
    if "/" in name and not org:
        org, name = name.split("/", 1)
    ns = org or user.username
    repo = await _get_repo(session, repo_type, ns, name)
    if repo is None:
        raise _repo_not_found(repo_type, ns, name)
    _require_owner(repo, user)
    # Best-effort wipe of the repo's storage prefix (it's repo-specific).
    try:
        backend = await _backend_for(session, repo)
        objs = await run_in_threadpool(backend.list_prefix, repo.prefix)
        for o in objs:
            await run_in_threadpool(backend.delete, o["key"])
    except Exception as e:  # noqa: BLE001 — leave bytes if storage is unreachable
        logger.warning("hf: could not wipe %s storage: %s", repo.full_id, e)
    await session.delete(repo)
    await session.commit()
    return {"ok": True}


# ---------- write: preupload (decide lfs vs regular) --------------------


async def _preupload_impl(rtype: str, ns: str, name: str, payload: dict,
                          user: User, session: AsyncSession):
    repo_type = _rt(rtype)
    repo = await _get_repo(session, repo_type, ns, name)
    if repo is None:
        raise _repo_not_found(repo_type, ns, name)
    _require_owner(repo, user)
    out = []
    for f in (payload.get("files") or []):
        path = f.get("path")
        size = int(f.get("size") or 0)
        mode = "lfs" if _is_lfs(path or "", size) else "regular"
        out.append({"path": path, "uploadMode": mode, "shouldIgnore": False})
    return {"files": out}


@router.post("/api/{rtype}/{ns}/{name}/preupload/{revision:path}")
async def preupload(rtype: str, ns: str, name: str, revision: str, payload: dict,  # noqa: ARG001
                    user: User = Depends(current_user),
                    session: AsyncSession = Depends(get_session)):
    return await _preupload_impl(rtype, ns, name, payload, user, session)


# ---------- write: LFS batch (issue signed upload URLs) -----------------


async def _batch_impl(repo_type: str, ns: str, name: str, payload: dict, request: Request,
                      user: User, session: AsyncSession):
    repo = await _get_repo(session, repo_type, ns, name)
    if repo is None:
        raise _repo_not_found(repo_type, ns, name)
    _require_owner(repo, user)
    backend = await _backend_for(session, repo)
    operation = payload.get("operation", "upload")
    objects = []
    for obj in (payload.get("objects") or []):
        oid = obj.get("oid")
        size = int(obj.get("size") or 0)
        if operation == "upload":
            blob_key = f"{repo.prefix}/.hf-lfs/{oid}"
            existing = await run_in_threadpool(backend.stat, blob_key)
            if existing == size:
                objects.append({"oid": oid, "size": size})  # already present → skip
                continue
            token = _sign_lfs(repo.id, oid)
            # Use the repo's canonical name (the route captured it as "<name>.git").
            href = _client_url(repo_type, f"{repo.namespace}/{repo.name}.git/lfs/{oid}?token={token}")
            objects.append({"oid": oid, "size": size, "actions": {"upload": {"href": href}}})
        else:
            objects.append({"oid": oid, "size": size})
    return JSONResponse({"transfer": "basic", "objects": objects},
                        media_type="application/vnd.git-lfs+json")


@router.post("/{ns}/{name}/info/lfs/objects/batch")
async def model_lfs_batch(ns: str, name: str, payload: dict, request: Request,
                          user: User = Depends(current_user),
                          session: AsyncSession = Depends(get_session)):
    return await _batch_impl("model", ns, name, payload, request, user, session)


@router.post("/datasets/{ns}/{name}/info/lfs/objects/batch")
async def dataset_lfs_batch(ns: str, name: str, payload: dict, request: Request,
                            user: User = Depends(current_user),
                            session: AsyncSession = Depends(get_session)):
    return await _batch_impl("dataset", ns, name, payload, request, user, session)


# ---------- write: LFS upload PUT (signed, no bearer auth) --------------


async def _lfs_put_impl(repo_type: str, ns: str, name: str, oid: str,
                        request: Request, session: AsyncSession):
    token = request.query_params.get("token", "")
    repo_id = _verify_lfs(token, oid)
    if repo_id is None:
        raise HTTPException(status_code=403, detail="invalid or expired LFS upload token")
    repo = await session.get(CatalogRepo, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="repo no longer exists")
    backend = await _backend_for(session, repo)

    # Spool the (potentially huge) body to a temp file, hashing as we go, then
    # hand the file to the backend (S3 upload_file does managed multipart).
    hasher = hashlib.sha256()
    size = 0
    tmp = tempfile.NamedTemporaryFile(delete=False, prefix="sgpu-lfs-")
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            hasher.update(chunk)
            size += len(chunk)
            await run_in_threadpool(tmp.write, chunk)
        tmp.flush()
        tmp.close()
        digest = hasher.hexdigest()
        if digest != oid:
            raise HTTPException(status_code=400, detail=f"oid mismatch: got {digest}, expected {oid}")
        await run_in_threadpool(backend.put_file, f"{repo.prefix}/.hf-lfs/{oid}", tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return Response(status_code=200)


@router.put("/{ns}/{name}/lfs/{oid}")
async def model_lfs_put(ns: str, name: str, oid: str, request: Request,  # noqa: ARG001 — repo via token
                        session: AsyncSession = Depends(get_session)):
    return await _lfs_put_impl("model", ns, name, oid, request, session)


@router.put("/datasets/{ns}/{name}/lfs/{oid}")
async def dataset_lfs_put(ns: str, name: str, oid: str, request: Request,  # noqa: ARG001
                          session: AsyncSession = Depends(get_session)):
    return await _lfs_put_impl("dataset", ns, name, oid, request, session)


# ---------- write: commit (NDJSON apply) --------------------------------


async def _commit_impl(rtype: str, ns: str, name: str, request: Request,
                       user: User, session: AsyncSession):
    repo_type = _rt(rtype)
    repo = await _get_repo(session, repo_type, ns, name)
    if repo is None:
        raise _repo_not_found(repo_type, ns, name)
    _require_owner(repo, user)
    backend = await _backend_for(session, repo)

    raw = await request.body()
    by_path: dict[str, dict] = {e["path"]: e for e in _manifest(repo)}

    for line in raw.split(b"\n"):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="malformed NDJSON in commit")
        key = item.get("key")
        val = item.get("value") or {}
        if key == "header":
            continue
        if key == "file":
            path = val.get("path")
            content_b64 = val.get("content") or ""
            data = base64.b64decode(content_b64) if val.get("encoding") == "base64" else content_b64.encode()
            await run_in_threadpool(backend.put_bytes, f"{repo.prefix}/{path}", data)
            etag = hashlib.sha256(data).hexdigest()
            by_path[path] = {"path": path, "size": len(data), "oid": etag, "lfs": False, "etag": etag}
        elif key == "lfsFile":
            path = val.get("path")
            oid = val.get("oid")
            size = val.get("size")
            if size is None:  # size omitted on copy ops — stat the blob
                size = await run_in_threadpool(backend.stat, f"{repo.prefix}/.hf-lfs/{oid}") or 0
            by_path[path] = {"path": path, "size": int(size), "oid": oid, "lfs": True, "etag": oid}
        elif key == "deletedFile":
            path = val.get("path")
            old = by_path.pop(path, None)
            if old and not old.get("lfs"):
                await run_in_threadpool(backend.delete, f"{repo.prefix}/{path}")
        elif key == "deletedFolder":
            folder = (val.get("path") or "").rstrip("/") + "/"
            for p in [p for p in by_path if p.startswith(folder)]:
                old = by_path.pop(p, None)
                if old and not old.get("lfs"):
                    await run_in_threadpool(backend.delete, f"{repo.prefix}/{p}")

    manifest = sorted(by_path.values(), key=lambda e: e["path"])
    sha = _compute_sha(manifest)
    repo.manifest = manifest
    repo.sha = sha
    repo.size_bytes = sum(int(e.get("size") or 0) for e in manifest)
    repo.num_files = len(manifest)
    await session.commit()

    return {
        "commitUrl": _client_url(repo_type, f"{ns}/{name}/commit/{sha}"),
        "commitOid": sha,
        "pullRequestUrl": None,
    }


@router.post("/api/{rtype}/{ns}/{name}/commit/{revision:path}")
async def commit(rtype: str, ns: str, name: str, revision: str, request: Request,  # noqa: ARG001
                 user: User = Depends(current_user),
                 session: AsyncSession = Depends(get_session)):
    return await _commit_impl(rtype, ns, name, request, user, session)
