"""HTTP routes for Autotrain datasets.

A dataset is a named pointer to a metadata file of {audio, transcription} rows
living in a `Storage` backend (S3) or on a HuggingFace repo. It mirrors the
Benchmark ownership model (owned by a user, references a Storage by id) and the
Storage admin module's crypto/boto3 patterns.

Source kinds:
- `upload` — CSV/JSON/JSONL uploaded through the UI, written to the dataset's S3
  storage under `{storage.prefix}/datasets/{id}/{filename}`.
- `s3`     — references a metadata file already in S3 (`s3_metadata_uri`).
- `hf`     — references an existing HuggingFace dataset repo (`hf_repo`).

S3 I/O reuses the benchmark helpers (`bench._target_from_storage_row`,
`bench.s3_*`). All routes require the `datasets` section; reads are owner-scoped
(admins may pass `?scope=all`).
"""
from __future__ import annotations

import dataclasses
import io
import json
import logging
import os
import secrets
import tempfile
from datetime import datetime, timezone
from typing import Any, Optional
import posixpath
from urllib.parse import quote, urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import audit as audit_module
from . import bench, crypto, dataset_metadata
from .auth import require_section
from .db import Dataset, Storage, User, get_session

logger = logging.getLogger("gateway.datasets")

router = APIRouter(prefix="/v1/datasets", tags=["datasets"])

KINDS = ("upload", "s3", "hf", "label")
_UPLOAD_EXTS = (".csv", ".json", ".jsonl", ".ndjson")


# ---------- request / response models ----------------------------------


class CreateDatasetRequest(BaseModel):
    name: str
    kind: str = "upload"  # upload | s3 | hf | label
    storage_id: Optional[str] = None
    description: Optional[str] = None
    audio_prefix: Optional[str] = None
    s3_metadata_uri: Optional[str] = None  # kind=s3
    hf_repo: Optional[str] = None  # kind=hf
    # kind=label — live import from a labeling-platform project's export API.
    label_base_url: Optional[str] = None     # e.g. http://localhost:3002
    label_project_id: Optional[str] = None   # project UUID
    label_token: Optional[str] = None        # lpat_… (stored Fernet-encrypted, never returned)
    label_token_secret: Optional[str] = None # OR: a global-secret key holding the lpat token
    label_status: Optional[str] = None       # approved | rejected | not_reviewed | all (default approved)


class UpdateDatasetRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    audio_prefix: Optional[str] = None
    audio_field: Optional[str] = None
    transcription_field: Optional[str] = None
    # Per-split transcription column overrides, e.g. {"train": "text", "test": "after"}.
    # Pass {} to clear. Splits not listed fall back to transcription_field.
    split_fields: Optional[dict[str, str]] = None


class SyncRequest(BaseModel):
    hf_repo: str
    private: bool = True


class TransformRequest(BaseModel):
    # Turn a zip/tar-of-audio + metadata HF dataset into one with an audio column.
    target: str  # "hf" | "s3"
    hf_repo: Optional[str] = None     # required for target=hf (owner/name)
    storage_id: Optional[str] = None  # required for target=s3 (a kind=s3 storage)
    # target=s3: destination folder within the storage (under its configured
    # prefix). Blank → datasets/{id}/transformed.
    s3_folder: Optional[str] = None


class TtsPackRequest(BaseModel):
    """NeuCodec-encode + multipack a {audio, transcription} dataset into a
    ChiniDataset, on a GPU provider over SSH, → a new packed dataset."""
    provider_id: str                       # GPU box (vm/runpod) — NeuCodec needs a GPU
    storage_id: str                        # s3 storage for the packed shards
    tokenizer: Optional[str] = None        # speech-token tokenizer (pack_stage1)
    sequence_length: int = 4096            # multipack block length
    gpu_count: int = 1
    visible_devices: Optional[str] = None


class DatasetRecord(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    kind: str
    storage_id: Optional[str] = None
    storage_name: Optional[str] = None
    audio_prefix: Optional[str] = None
    s3_metadata_uri: Optional[str] = None
    size_bytes: Optional[int] = None
    metadata_filename: Optional[str] = None
    format: Optional[str] = None
    num_rows: Optional[int] = None
    audio_field: str = "audio"
    transcription_field: str = "transcription"
    split_fields: Optional[dict[str, str]] = None
    audio_dataset_id: Optional[str] = None  # materialised S3 audio dataset (if any)
    hf_repo: Optional[str] = None
    hf_revision: Optional[str] = None
    hf_synced_at: Optional[str] = None
    label_base_url: Optional[str] = None     # kind=label source (token never returned)
    label_project_id: Optional[str] = None
    label_status: Optional[str] = None
    label_token_secret: Optional[str] = None  # global-secret key (if used instead of a stored token)
    transform_status: Optional[str] = None  # "" | running | done | failed
    transform_log: Optional[str] = None     # short tail of progress lines
    created_at: str
    updated_at: str
    created_by: str


class UploadResult(BaseModel):
    filename: str
    format: str
    num_rows: int
    columns: list[str]
    audio_field: str
    transcription_field: str
    preview: list[dict[str, Any]]


class PreviewResponse(BaseModel):
    audio_field: str
    transcription_field: str
    rows: list[dict[str, Any]]
    offset: int = 0
    limit: int = 0
    total: Optional[int] = None  # full row count when known (for pagination)
    split: Optional[str] = None  # which HF split these rows came from
    splits: Optional[list[str]] = None  # available HF splits (for a picker)
    error: Optional[str] = None


class SplitInfo(BaseModel):
    split: str
    columns: list[str]
    num_rows: Optional[int] = None


class SplitsResponse(BaseModel):
    splits: list[SplitInfo]
    error: Optional[str] = None


# ---------- helpers -----------------------------------------------------


def _gen_id() -> str:
    return f"ds-{secrets.token_hex(4)}"


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _to_record(d: Dataset, owner_username: str, storage_name: Optional[str]) -> DatasetRecord:
    return DatasetRecord(
        id=d.id,
        name=d.name,
        description=d.description,
        kind=d.kind,
        storage_id=d.storage_id,
        storage_name=storage_name,
        audio_prefix=d.audio_prefix,
        s3_metadata_uri=d.s3_metadata_uri,
        size_bytes=d.size_bytes,
        metadata_filename=d.metadata_filename,
        format=d.format,
        num_rows=d.num_rows,
        audio_field=d.audio_field,
        transcription_field=d.transcription_field,
        split_fields=getattr(d, "split_fields", None) or None,
        audio_dataset_id=getattr(d, "audio_dataset_id", None) or None,
        hf_repo=d.hf_repo,
        hf_revision=d.hf_revision,
        hf_synced_at=_iso(d.hf_synced_at),
        label_base_url=getattr(d, "label_base_url", None),
        label_project_id=getattr(d, "label_project_id", None),
        label_status=getattr(d, "label_status", None),
        label_token_secret=getattr(d, "label_token_secret", None),
        transform_status=getattr(d, "transform_status", None),
        transform_log=getattr(d, "transform_log", None),
        created_at=_iso(d.created_at) or "",
        updated_at=_iso(d.updated_at) or "",
        created_by=owner_username,
    )


async def _owner_map(session: AsyncSession, rows: list[Dataset]) -> dict[int, str]:
    ids = {d.owner_id for d in rows}
    out: dict[int, str] = {}
    if ids:
        res = await session.execute(select(User).where(User.id.in_(ids)))
        for u in res.scalars().all():
            out[u.id] = u.username
    return out


async def _storage_name_map(session: AsyncSession, rows: list[Dataset]) -> dict[str, str]:
    ids = {d.storage_id for d in rows if d.storage_id}
    out: dict[str, str] = {}
    if ids:
        res = await session.execute(select(Storage).where(Storage.id.in_(ids)))
        for s in res.scalars().all():
            out[s.id] = s.name
    return out


async def _require_dataset(session: AsyncSession, dataset_id: str, user: User) -> Dataset:
    d = await session.get(Dataset, dataset_id)
    if d is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    if not user.is_admin and d.owner_id != user.id:
        raise HTTPException(status_code=403, detail="forbidden")
    return d


async def _load_storage(session: AsyncSession, storage_id: str) -> Storage:
    s = await session.get(Storage, storage_id)
    if s is None:
        raise HTTPException(status_code=400, detail=f"storage {storage_id} not found")
    return s


def _s3_target_and_prefix(storage: Storage):
    """(S3Target, base_prefix) for a dataset's storage. base_prefix is the
    storage's configured prefix with no surrounding slashes; keys are built as
    `{base_prefix}/datasets/{id}/...`. Reuses the benchmark credential resolver
    (its `prefix_root` is irrelevant — we pass absolute keys to the s3 helpers)."""
    target = bench._target_from_storage_row(storage)
    base = (storage.config or {}).get("prefix") or ""
    return target, base.strip().strip("/")


def _join_key(*parts: Optional[str]) -> str:
    return "/".join(p.strip("/") for p in parts if p and p.strip("/"))


def _metadata_key(storage: Storage, dataset_id: str, filename: str) -> str:
    _, base = _s3_target_and_prefix(storage)
    return _join_key(base, "datasets", dataset_id, filename)


async def _hf_token(storage: Optional[Storage], session: AsyncSession) -> Optional[str]:
    """HF token for a kind=huggingface storage. Precedence: a referenced global
    secret (config.hf_token_secret) > the storage's own encrypted token > the
    HF_TOKEN env var."""
    cfg = (storage.config or {}) if storage is not None else {}
    ref = cfg.get("hf_token_secret")
    if ref:
        from .global_env_api import load_global_env
        tok = (await load_global_env(session)).get(ref)
        if tok:
            return tok
    if cfg.get("credentials_enc"):
        try:
            return json.loads(crypto.decrypt(cfg["credentials_enc"])).get("token")
        except Exception:
            pass
    return os.environ.get("HF_TOKEN", "").strip() or None


def _resolve_audio_url(target, base_prefix: str, audio_prefix: Optional[str], value: Any) -> Optional[str]:
    """Turn a row's audio reference into a playable URL: http(s) passthrough;
    `s3://bucket/key` or a relative key → a presigned GET (TTL 1h)."""
    if not value or not isinstance(value, str):
        return None
    v = value.strip()
    if v.startswith("http://") or v.startswith("https://"):
        return v
    if v.startswith("s3://"):
        u = urlparse(v)
        bucket, key = u.netloc, u.path.lstrip("/")
        if not bucket or not key:
            return None
        t = dataclasses.replace(target, bucket=bucket)
        return bench.s3_presign_get(key, expires=3600, target=t)
    # relative key under the storage prefix (+ dataset audio_prefix)
    key = _join_key(base_prefix, audio_prefix, v)
    try:
        return bench.s3_presign_get(key, expires=3600, target=target)
    except Exception:
        return None


# ---------- CRUD --------------------------------------------------------


@router.get("", response_model=list[DatasetRecord])
async def list_datasets(
    scope: str = "mine",
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    show_all = user.is_admin and scope == "all"
    stmt = select(Dataset).order_by(Dataset.created_at.desc())
    if not show_all:
        stmt = select(Dataset).where(Dataset.owner_id == user.id).order_by(Dataset.created_at.desc())
    rows = list((await session.execute(stmt)).scalars().all())
    owners = await _owner_map(session, rows)
    storages = await _storage_name_map(session, rows)
    return [_to_record(d, owners.get(d.owner_id, "?"), storages.get(d.storage_id or "")) for d in rows]


@router.post("", response_model=DatasetRecord)
async def create_dataset(
    req: CreateDatasetRequest,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    if req.kind not in KINDS:
        raise HTTPException(status_code=400, detail=f"unsupported kind: {req.kind}")
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    storage_name: Optional[str] = None
    label_base_url: Optional[str] = None
    label_project_id: Optional[str] = None
    label_token_enc: Optional[str] = None
    label_token_secret_val: Optional[str] = None
    label_status_val: Optional[str] = None
    label_num_rows: Optional[int] = None
    if req.kind in ("upload", "s3"):
        if not req.storage_id:
            raise HTTPException(status_code=400, detail="storage_id (an S3 storage) is required")
        storage = await _load_storage(session, req.storage_id)
        if storage.kind != "s3":
            raise HTTPException(status_code=400, detail="storage must be kind=s3 for upload/s3 datasets")
        storage_name = storage.name
        if req.kind == "s3" and not (req.s3_metadata_uri or "").strip():
            raise HTTPException(status_code=400, detail="s3_metadata_uri is required for kind=s3")
    elif req.kind == "hf":
        if not (req.hf_repo or "").strip():
            raise HTTPException(status_code=400, detail="hf_repo (owner/name) is required for kind=hf")
        if req.storage_id:
            storage = await _load_storage(session, req.storage_id)
            storage_name = storage.name
    else:  # label — live import from a labeling-platform project
        label_base_url = (req.label_base_url or "").strip().rstrip("/")
        label_project_id = (req.label_project_id or "").strip()
        tok = (req.label_token or "").strip()
        sec = (req.label_token_secret or "").strip()
        if not (label_base_url and label_project_id):
            raise HTTPException(status_code=400, detail="label_base_url and label_project_id are required for kind=label")
        if not tok and not sec:
            raise HTTPException(status_code=400, detail="provide an API token or pick a global secret for kind=label")
        # Resolve the actual token to verify with — either pasted, or pulled from
        # the referenced global secret.
        verify_tok = tok
        if not verify_tok and sec:
            from .global_env_api import load_global_env
            verify_tok = (await load_global_env(session)).get(sec)
            if not verify_tok:
                raise HTTPException(status_code=400, detail=f"global secret '{sec}' not found or empty")
        label_status_val = (req.label_status or "approved").strip() or "approved"
        if label_status_val not in ("approved", "rejected", "not_reviewed", "all"):
            raise HTTPException(status_code=400, detail=f"invalid label_status: {label_status_val}")
        # Verify the token + project reachable, and grab the row count, by reading
        # the export header (limit=0 stops after the first line).
        try:
            _, label_num_rows = await _run_sync(
                _label_export_rows, label_base_url, label_project_id, verify_tok, label_status_val, 0, 0,
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail=f"could not reach the labeling platform or verify the token: {e}",
            ) from e
        # Store the token reference: a global-secret key (no copy) or the pasted
        # token Fernet-encrypted.
        if sec and not tok:
            label_token_secret_val = sec
        else:
            label_token_enc = crypto.encrypt(json.dumps({"token": tok}))

    did = _gen_id()
    row = Dataset(
        id=did,
        owner_id=user.id,
        name=name,
        description=(req.description or "").strip() or None,
        kind=req.kind,
        storage_id=req.storage_id,
        audio_prefix=(req.audio_prefix or "").strip() or None,
        s3_metadata_uri=(req.s3_metadata_uri or "").strip() or None,
        hf_repo=(req.hf_repo or "").strip() or None,
        label_base_url=label_base_url,
        label_project_id=label_project_id,
        label_token_enc=label_token_enc,
        label_token_secret=label_token_secret_val,
        label_status=label_status_val,
    )
    if req.kind == "label":
        # The Label export uses audio_url + transcription columns.
        row.audio_field = "audio_url"
        row.transcription_field = "transcription"
        row.num_rows = label_num_rows
    session.add(row)
    await session.commit()
    await session.refresh(row)
    await audit_module.record(user, "dataset.create", "dataset", did, name, details={"kind": req.kind})
    logger.info("created dataset %s (%s) for user=%s", did, req.kind, user.username)
    return _to_record(row, user.username, storage_name)


@router.get("/{dataset_id}", response_model=DatasetRecord)
async def get_dataset(
    dataset_id: str,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    d = await _require_dataset(session, dataset_id, user)
    owner = await session.get(User, d.owner_id)
    storage = await session.get(Storage, d.storage_id) if d.storage_id else None
    return _to_record(d, owner.username if owner else "", storage.name if storage else None)


@router.patch("/{dataset_id}", response_model=DatasetRecord)
async def update_dataset(
    dataset_id: str,
    req: UpdateDatasetRequest,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    d = await _require_dataset(session, dataset_id, user)
    if req.name is not None:
        n = req.name.strip()
        if not n:
            raise HTTPException(status_code=400, detail="name cannot be blank")
        d.name = n
    if req.description is not None:
        d.description = req.description.strip() or None
    if req.audio_prefix is not None:
        d.audio_prefix = req.audio_prefix.strip() or None
    if req.audio_field is not None and req.audio_field.strip():
        d.audio_field = req.audio_field.strip()
    if req.transcription_field is not None and req.transcription_field.strip():
        d.transcription_field = req.transcription_field.strip()
    if req.split_fields is not None:
        # {} clears the overrides; otherwise keep only non-blank split→column pairs.
        cleaned = {
            str(k).strip(): str(v).strip()
            for k, v in req.split_fields.items()
            if str(k).strip() and str(v).strip()
        }
        d.split_fields = cleaned or None
    await session.commit()
    await session.refresh(d)
    storage = await session.get(Storage, d.storage_id) if d.storage_id else None
    owner = await session.get(User, d.owner_id)
    await audit_module.record(user, "dataset.update", "dataset", d.id, d.name)
    return _to_record(d, owner.username if owner else "", storage.name if storage else None)


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    d = await _require_dataset(session, dataset_id, user)
    name = d.name
    await session.delete(d)
    await session.commit()
    await audit_module.record(user, "dataset.delete", "dataset", dataset_id, name)
    return {"ok": True, "id": dataset_id}


# ---------- upload / preview / sync ------------------------------------


@router.post("/{dataset_id}/upload", response_model=UploadResult)
async def upload_metadata(
    dataset_id: str,
    request: Request,
    filename: str = Query(..., description="original file name, used for format + S3 key"),
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Accept the raw metadata file body, parse + validate it, write it to the
    dataset's S3 storage, and stamp the parsed shape onto the row."""
    d = await _require_dataset(session, dataset_id, user)
    if not d.storage_id:
        raise HTTPException(status_code=400, detail="dataset has no S3 storage attached")
    fname = os.path.basename(filename.strip())
    if not fname or os.path.splitext(fname)[1].lower() not in _UPLOAD_EXTS:
        raise HTTPException(status_code=400, detail="filename must end in .csv, .json, .jsonl, or .ndjson")

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        parsed = dataset_metadata.parse_metadata_bytes(fname, body)
    except dataset_metadata.DatasetParseError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    storage = await _load_storage(session, d.storage_id)
    target, _ = _s3_target_and_prefix(storage)
    key = _metadata_key(storage, dataset_id, fname)
    try:
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(body)
            tmp.flush()
            bench.s3_put_file(key, tmp.name, target=target)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"S3 upload failed: {e}") from e

    d.metadata_filename = fname
    d.format = parsed["format"]
    d.num_rows = parsed["num_rows"]
    d.audio_field = parsed["audio_field"]
    d.transcription_field = parsed["transcription_field"]
    d.size_bytes = len(body)
    d.hf_synced_at = None  # content changed → mark out of sync
    await session.commit()
    await audit_module.record(
        user, "dataset.upload", "dataset", dataset_id, d.name,
        details={"filename": fname, "num_rows": parsed["num_rows"]},
    )
    return UploadResult(
        filename=fname,
        format=parsed["format"],
        num_rows=parsed["num_rows"],
        columns=parsed["columns"],
        audio_field=parsed["audio_field"],
        transcription_field=parsed["transcription_field"],
        preview=parsed["preview"],
    )


def _audio_str(v: Any) -> Optional[str]:
    """Normalise an audio cell to a single *playable* URL string. The HF
    datasets-server returns audio as `[{"src": url, "type": ...}]`; S3/upload
    rows already hold a presigned URL. A bare filename (e.g. an HF source whose
    audio lives in a zip, no Audio feature) is NOT playable → None, so the UI
    shows "no audio" instead of a 404 in the player."""
    if isinstance(v, str):
        return v if v.startswith(("http://", "https://")) else None
    if isinstance(v, dict):
        return _audio_str(v.get("src"))
    if isinstance(v, list) and v:
        return _audio_str(v[0])
    return None


_AUDIO_CT = {
    ".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg", ".opus": "audio/opus", ".m4a": "audio/mp4", ".aac": "audio/aac",
}


def _proxy_audio_url(dataset_id: str, presigned: Optional[str]) -> Optional[str]:
    """Wrap a presigned S3 URL in a same-origin gateway path so the browser
    fetches it through the gateway (no S3 CORS). The frontend prefixes
    `/api/proxy`. Returns None if there's nothing to wrap."""
    if not presigned:
        return None
    return f"/v1/datasets/{dataset_id}/audio?src={quote(presigned, safe='')}"


def _fetch_url_bytes(url: str) -> tuple[bytes, str]:
    """GET a URL server-side (no browser CORS) and return (bytes, content_type)."""
    with httpx.Client(timeout=30.0, follow_redirects=True) as cli:
        r = cli.get(url)
        r.raise_for_status()
        ext_ct = _AUDIO_CT.get(posixpath.splitext(urlparse(url).path)[1].lower())
        upstream = (r.headers.get("content-type") or "").split(";")[0].strip()
        # S3 often serves these as generic octet-stream; prefer the real audio
        # type inferred from the extension so the browser/<audio> handles it.
        ct = ext_ct if (ext_ct and upstream in ("", "binary/octet-stream", "application/octet-stream")) else (upstream or ext_ct or "application/octet-stream")
        return r.content, ct


def _byte_range_response(data: bytes, ctype: str, range_header: Optional[str]) -> Response:
    """Serve bytes with HTTP Range support so media elements can seek. Returns a
    206 with Content-Range for a valid `Range: bytes=…` request, else a 200 that
    advertises `Accept-Ranges: bytes`."""
    total = len(data)
    base = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"}
    if range_header and range_header.strip().startswith("bytes="):
        try:
            spec = range_header.strip()[len("bytes="):].split(",")[0].strip()
            start_s, _, end_s = spec.partition("-")
            if start_s == "":  # suffix: bytes=-N → last N bytes
                start, end = max(0, total - int(end_s)), total - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s else total - 1
            if 0 <= start < total and start <= end:
                end = min(end, total - 1)
                chunk = data[start:end + 1]
                return Response(
                    content=chunk, status_code=206, media_type=ctype,
                    headers={**base, "Content-Range": f"bytes {start}-{end}/{total}", "Content-Length": str(len(chunk))},
                )
        except (ValueError, IndexError):
            pass
    return Response(content=data, media_type=ctype, headers={**base, "Content-Length": str(total)})


def _compute_peaks(data: bytes, buckets: int) -> tuple[list[list[float]], float]:
    """Decode audio bytes (via libsndfile, which handles the awkward MP3s the
    browser's Web Audio decoder rejects) and reduce to `buckets` [min,max] peak
    pairs + duration — so the waveform renders server-side, reliably."""
    import io as _io

    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(_io.BytesIO(data), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    n = int(mono.shape[0])
    duration = (n / sr) if sr else 0.0
    if n == 0:
        return [[0.0, 0.0]] * buckets, 0.0
    per = max(1, n // buckets)
    peaks: list[list[float]] = []
    for i in range(buckets):
        seg = mono[i * per: min((i + 1) * per, n)]
        if seg.size:
            peaks.append([float(seg.min()), float(seg.max())])
        else:
            peaks.append([0.0, 0.0])
    return peaks, float(duration)


async def _audio_s3_storage(session: AsyncSession, d: Dataset) -> tuple[Optional[Storage], Optional[str]]:
    """The (Storage, bucket) backing this dataset's audio: its own S3 storage for
    s3/upload, or the materialised output's storage for an HF source linked via
    audio_dataset_id. bucket override comes from the relevant s3:// metadata URI."""
    def _bucket(uri: Optional[str]) -> Optional[str]:
        return urlparse(uri).netloc if (uri or "").startswith("s3://") else None

    if d.kind in ("s3", "upload") and d.storage_id:
        return await session.get(Storage, d.storage_id), _bucket(d.s3_metadata_uri)
    if d.audio_dataset_id:
        out = await session.get(Dataset, d.audio_dataset_id)
        if out and out.kind == "s3" and out.storage_id:
            return await session.get(Storage, out.storage_id), _bucket(out.s3_metadata_uri)
    return None, None


async def _allowed_audio_hosts(session: AsyncSession, d: Dataset) -> set[str]:
    """Hosts the audio proxy may fetch from — the S3 endpoint host(s) of the
    dataset's own / materialised storage. A presigned probe yields the exact
    host we'd ever generate, so anything else is rejected (anti-SSRF)."""
    storage, bucket = await _audio_s3_storage(session, d)
    if storage is None:
        return set()
    try:
        target, _ = _s3_target_and_prefix(storage)
        if bucket:
            target = dataclasses.replace(target, bucket=bucket)
        return {urlparse(bench.s3_presign_get("__probe__", 60, target)).netloc}
    except Exception:  # noqa: BLE001
        return set()


async def _source_audio_resolver(session: AsyncSession, d: Dataset):
    """For an HF source materialised to S3 (audio_dataset_id), return fn(value) →
    a proxied presigned URL, joining the row's audio basename to the output's
    `audio/` folder. None when not materialised."""
    if not d.audio_dataset_id:
        return None
    out = await session.get(Dataset, d.audio_dataset_id)
    if not (out and out.kind == "s3" and out.s3_metadata_uri and out.storage_id):
        return None
    storage = await session.get(Storage, out.storage_id)
    if storage is None:
        return None
    target, _ = _s3_target_and_prefix(storage)
    u = urlparse(out.s3_metadata_uri)
    if u.netloc:
        target = dataclasses.replace(target, bucket=u.netloc)
    audio_base = posixpath.dirname(u.path.lstrip("/")) + "/audio"  # …/<folder>/audio

    def resolve(value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value:
            return None
        key = f"{audio_base}/{os.path.basename(value)}"
        try:
            return _proxy_audio_url(d.id, bench.s3_presign_get(key, 3600, target))
        except Exception:  # noqa: BLE001
            return None

    return resolve


def _hf_preview_rows(
    hf_repo: str, token: Optional[str], limit: int, offset: int = 0, split: Optional[str] = None,
) -> tuple[list[dict[str, Any]], Optional[int], Optional[str], list[str]]:
    """Fetch a page of rows for one split via the HF datasets-server API.
    Returns (rows, total_rows, used_split, all_split_names). `split` selects which
    split to read (default: the first); a split's full row count drives paging."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(timeout=20.0) as cli:
        sp = cli.get(
            "https://datasets-server.huggingface.co/splits",
            params={"dataset": hf_repo}, headers=headers,
        )
        sp.raise_for_status()
        splits = sp.json().get("splits", [])
        if not splits:
            return [], 0, None, []
        # Identify each entry by whichever of config/split is distinct, so the
        # label matches the parquet-dir names used by /splits + the per-split
        # mapping. This dataset uses configs "test"/"train" (each split "train");
        # a normal dataset uses one config "default" with distinct splits.
        configs = [s["config"] for s in splits]
        snames = [s["split"] for s in splits]
        if len(set(configs)) == len(splits):
            ident = lambda s: s["config"]  # noqa: E731
        elif len(set(snames)) == len(splits):
            ident = lambda s: s["split"]  # noqa: E731
        else:
            ident = lambda s: f'{s["config"]}/{s["split"]}'  # noqa: E731
        names = [ident(s) for s in splits]
        chosen = next((s for s in splits if ident(s) == split), splits[0])
        rows = cli.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": hf_repo, "config": chosen["config"], "split": chosen["split"],
                "offset": max(0, offset), "length": min(limit, 100),
            },
            headers=headers,
        )
        rows.raise_for_status()
        body = rows.json()
        total = body.get("num_rows_total")
        return [r.get("row", {}) for r in body.get("rows", [])], total, ident(chosen), names


def _hf_split_columns(hf_repo: str, token: Optional[str]) -> list[dict[str, Any]]:
    """Per-split column names for an HF dataset by reading each split's parquet
    footer (cheap — only the metadata/footer is fetched, not the data). Splits
    can have *different* schemas; the UI uses this to offer a transcription-column
    picker per split. Returns [{split, columns, num_rows}]."""
    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as pq

    api = HfApi(token=token)
    files = [f for f in api.list_repo_files(hf_repo, repo_type="dataset") if f.endswith(".parquet")]
    if not files:
        return []
    fs = HfFileSystem(token=token)
    agg: dict[str, dict[str, Any]] = {}
    for f in files:
        # HF parquet layout is "<split>/<shard>.parquet"; bare files → "default".
        split = f.split("/")[0] if "/" in f else "default"
        try:
            with fs.open(f"datasets/{hf_repo}/{f}") as fh:
                pf = pq.ParquetFile(fh)
                cols = list(pf.schema_arrow.names)
                nrows = pf.metadata.num_rows
        except Exception:
            logger.warning("split-columns read failed for %s/%s", hf_repo, f)
            continue
        a = agg.setdefault(split, {"columns": set(), "num_rows": 0})
        a["columns"].update(cols)
        a["num_rows"] += nrows
    return [
        {"split": k, "columns": sorted(v["columns"]), "num_rows": v["num_rows"]}
        for k, v in sorted(agg.items())
    ]


# ---------- labeling-platform source (kind=label) ----------------------


async def _label_token(d: Dataset, session: AsyncSession) -> Optional[str]:
    """Resolve a label dataset's access token. A referenced global secret
    (`label_token_secret`) takes precedence; otherwise the per-dataset
    Fernet-encrypted token (`label_token_enc`)."""
    ref = getattr(d, "label_token_secret", None)
    if ref:
        from .global_env_api import load_global_env
        tok = (await load_global_env(session)).get(ref)
        if tok:
            return tok
    enc = getattr(d, "label_token_enc", None)
    if enc:
        try:
            return json.loads(crypto.decrypt(enc)).get("token")
        except Exception:
            return None
    return None


def _label_export_rows(
    base_url: str, project_id: str, token: str, status: str, limit: int, offset: int,
) -> tuple[list[dict[str, Any]], Optional[int]]:
    """Stream a Label-platform project's `export.v1.jsonl` (one task per line, with
    `audio_url` + `transcription`) and return (rows[offset:offset+limit], total).
    Auth is `Authorization: Bearer <lpat token>`. Reads `X-Total-Tasks` for the
    count so we can stop early once the requested page is collected; `limit=0`
    just verifies the token + returns the total. Sync — call via `_run_sync`."""
    base = base_url.rstrip("/")
    url = f"{base}/api/projects/{project_id}/export.v1.jsonl"
    rows: list[dict[str, Any]] = []
    total: Optional[int] = None
    with httpx.Client(timeout=120.0, follow_redirects=True) as cli:
        with cli.stream(
            "GET", url, params={"status": status or "approved"},
            headers={"Authorization": f"Bearer {token}"},
        ) as r:
            r.raise_for_status()
            try:
                total = int(r.headers.get("x-total-tasks") or 0) or None
            except ValueError:
                total = None
            i = 0
            for line in r.iter_lines():
                s = line.strip()
                if not s:
                    continue
                if offset <= i < offset + limit:
                    try:
                        rows.append(json.loads(s))
                    except Exception:
                        pass
                i += 1
                if total is not None and i >= offset + limit:
                    break
            if total is None:
                total = i
    return rows, total


@router.get("/{dataset_id}/preview", response_model=PreviewResponse)
async def preview_dataset(
    dataset_id: str,
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    split: Optional[str] = Query(None, description="HF split to read (default: first)"),
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Return a page (`offset`..`offset+limit`) of rows with each audio reference
    resolved to a presigned (or passthrough) URL the browser can play, plus the
    full `total` row count so the UI can paginate through the whole dataset. For
    HF sources, the transcription column honours the dataset's per-split mapping."""
    d = await _require_dataset(session, dataset_id, user)
    af, tf = d.audio_field, d.transcription_field

    def _resp(**kw):
        return PreviewResponse(audio_field=af, transcription_field=tf, offset=offset, limit=limit, **kw)

    if d.kind == "tts_packed":
        # Packed = tokenized NeuCodec multipacks (no audio/text rows to show).
        return _resp(rows=[], total=d.num_rows or 0,
                     error=f"Packed TTS dataset ({d.num_rows or '?'} records) — tokenized multipacks, not row-previewable.")

    try:
        if d.kind == "label":
            tok = await _label_token(d, session)
            if not (d.label_base_url and d.label_project_id and tok):
                return _resp(rows=[], error="labeling-platform source not fully configured")
            raw, total = await _run_sync(
                _label_export_rows, d.label_base_url, d.label_project_id, tok,
                d.label_status or "approved", limit, offset,
            )
            base = (d.label_base_url or "").rstrip("/")

            def _label_audio(r: dict) -> Optional[str]:
                u = str(r.get("audio_url") or "")
                # Proxy-mode export URLs ({base}/api/…) need the lpat token, which
                # the browser can't send — route them through our audio proxy
                # (binary-safe via the web /api/datasets/{id}/label-audio route).
                # Presigned-S3 URLs are browser-playable as-is.
                if u.startswith(base + "/api/") and r.get("id"):
                    return f"/api/datasets/{d.id}/label-audio?task_id={r['id']}"
                return _audio_str(u)

            rows = [
                {**r, "audio_url": _label_audio(r), "transcription": r.get("transcription")}
                for r in raw
            ]
            return _resp(rows=rows, total=total)

        if d.kind == "hf":
            storage = await session.get(Storage, d.storage_id) if d.storage_id else None
            if not d.hf_repo:
                return _resp(rows=[], error="no hf_repo set")
            raw, total, used_split, names = await _run_sync(
                _hf_preview_rows, d.hf_repo, await _hf_token(storage, session), limit, offset, split,
            )
            # Honour the per-split transcription mapping (e.g. test→after) so rows
            # aren't blank when a split uses a different column than the default.
            tcol = (d.split_fields or {}).get(used_split or "") or tf
            # Source audio lives in a zip (no playable column). If it's been
            # materialised to S3, resolve audio by basename through the proxy.
            resolver = await _source_audio_resolver(session, d)
            rows = [
                {
                    "audio_url": resolver(r.get(af)) if resolver else _audio_str(r.get(af)),
                    "transcription": r.get(tcol),
                    **r,
                }
                for r in raw
            ]
            return _resp(rows=rows, total=total, split=used_split, splits=names)

        # upload / s3 → read the metadata file from S3
        if not d.storage_id:
            return _resp(rows=[], error="no storage attached")
        storage = await _load_storage(session, d.storage_id)
        target, base = _s3_target_and_prefix(storage)
        if d.kind == "s3":
            if not d.s3_metadata_uri:
                return _resp(rows=[], error="no s3_metadata_uri")
            u = urlparse(d.s3_metadata_uri)
            t = dataclasses.replace(target, bucket=u.netloc) if u.scheme == "s3" else target
            key = u.path.lstrip("/") if u.scheme == "s3" else d.s3_metadata_uri
            mdname = os.path.basename(key)
        else:  # upload
            if not d.metadata_filename:
                return _resp(rows=[], error="no metadata uploaded yet")
            t = target
            key = _metadata_key(storage, dataset_id, d.metadata_filename)
            mdname = d.metadata_filename

        text = await _run_sync(bench.s3_get_text, key, t)
        if text is None:
            return _resp(rows=[], error="metadata file not found in storage")
        # Metadata tables are small (text + URLs) → parse all to get the true row
        # count, then slice the requested page.
        all_rows = dataset_metadata.parse_rows(mdname, text.encode("utf-8"), 10**9)
        # A `split` column (from a split-preserving transform) → expose the splits
        # and page within the chosen one, mirroring the HF split picker.
        used_split: Optional[str] = None
        splits_list: Optional[list[str]] = None
        if all_rows and "split" in all_rows[0]:
            splits_list = sorted({str(r["split"]) for r in all_rows if r.get("split")})
            used_split = split if (split and split in splits_list) else (splits_list[0] if splits_list else None)
            if used_split is not None:
                all_rows = [r for r in all_rows if str(r.get("split")) == used_split]
        total = len(all_rows)
        page = all_rows[offset:offset + limit]
        rows = [
            {
                # Proxy the presigned URL through the gateway (avoids S3 CORS in
                # the browser). _audio_str drops bare filenames first.
                "audio_url": _proxy_audio_url(
                    dataset_id, _audio_str(_resolve_audio_url(target, base, d.audio_prefix, r.get(af)))
                ),
                "transcription": r.get(tf),
                **r,
            }
            for r in page
        ]
        return _resp(rows=rows, total=total, split=used_split, splits=splits_list)
    except dataset_metadata.DatasetParseError as e:
        return _resp(rows=[], error=str(e))
    except Exception as e:  # noqa: BLE001 — surface as an inline error, not a 500
        logger.warning("preview failed for %s: %s", dataset_id, e)
        return _resp(rows=[], error=str(e))


@router.get("/{dataset_id}/audio")
async def dataset_audio(
    dataset_id: str,
    request: Request,
    src: str = Query(..., description="presigned S3 URL to stream (must belong to the dataset's storage)"),
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Stream an audio object through the gateway so the browser fetches it
    same-origin (the S3 bucket has no CORS config). Honours Range so the player
    can seek. `src` is restricted to the dataset's own / materialised S3 host
    (anti-SSRF)."""
    d = await _require_dataset(session, dataset_id, user)
    allowed = await _allowed_audio_hosts(session, d)
    pu = urlparse(src)
    if pu.scheme != "https" or pu.netloc not in allowed:
        raise HTTPException(status_code=400, detail="src is not an allowed audio URL for this dataset")
    try:
        data, ctype = await _run_sync(_fetch_url_bytes, src)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"audio fetch failed: {e}") from e
    return _byte_range_response(data, ctype, request.headers.get("range"))


@router.get("/{dataset_id}/audio-peaks")
async def dataset_audio_peaks(
    dataset_id: str,
    src: str = Query(..., description="presigned S3 URL (same allow-list as /audio)"),
    buckets: int = Query(200, ge=20, le=2000),
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Server-side waveform: decode the audio with libsndfile and return
    {peaks:[[min,max],…], duration}. Reliable for codecs the browser's Web Audio
    decoder chokes on (e.g. MPEG 2.5 / 8 kHz MP3)."""
    d = await _require_dataset(session, dataset_id, user)
    allowed = await _allowed_audio_hosts(session, d)
    pu = urlparse(src)
    if pu.scheme != "https" or pu.netloc not in allowed:
        raise HTTPException(status_code=400, detail="src is not an allowed audio URL for this dataset")
    try:
        data, _ = await _run_sync(_fetch_url_bytes, src)
        peaks, duration = await _run_sync(_compute_peaks, data, buckets)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"peaks failed: {e}") from e
    return {"peaks": peaks, "duration": duration}


@router.get("/{dataset_id}/splits", response_model=SplitsResponse)
async def dataset_splits(
    dataset_id: str,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Per-split column names for an HF source, so the UI can offer a
    transcription-column picker per split (splits can differ in schema)."""
    d = await _require_dataset(session, dataset_id, user)
    if d.kind != "hf" or not d.hf_repo:
        return SplitsResponse(splits=[])
    try:
        storage = await session.get(Storage, d.storage_id) if d.storage_id else None
        token = await _hf_token(storage, session)
        splits = await _run_sync(_hf_split_columns, d.hf_repo, token)
        return SplitsResponse(splits=[SplitInfo(**s) for s in splits])
    except Exception as e:  # noqa: BLE001
        logger.warning("splits lookup failed for %s: %s", dataset_id, e)
        return SplitsResponse(splits=[], error=str(e))


@router.get("/{dataset_id}/label-audio")
async def label_audio(
    dataset_id: str,
    request: Request,
    task_id: str = Query(..., description="labeling-platform task id"),
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Proxy a labeling-platform task's audio. The export's `audio_url` needs the
    lpat token (which a browser <audio> can't send), and we don't want to leak the
    upstream URL/token to the client — so the preview points audio here and we
    fetch upstream with the stored token, then serve the bytes with Range support.
    The Label endpoint returns the whole file (no Range), so we buffer it (TTS
    clips are small) and slice locally via `_byte_range_response` — that's what the
    browser <audio> element needs to load metadata + seek without erroring."""
    d = await _require_dataset(session, dataset_id, user)
    if d.kind != "label" or not (d.label_base_url and d.label_project_id):
        raise HTTPException(status_code=400, detail="not a labeling-platform dataset")
    tok = await _label_token(d, session)
    if not tok:
        raise HTTPException(status_code=400, detail="no stored labeling-platform token")

    base = d.label_base_url.rstrip("/")
    url = f"{base}/api/projects/{d.label_project_id}/tasks/{task_id}/audio"
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {tok}"})
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"upstream audio fetch failed ({resp.status_code})")
    ctype = resp.headers.get("content-type", "audio/wav")
    return _byte_range_response(resp.content, ctype, request.headers.get("range"))


@router.post("/{dataset_id}/sync", response_model=DatasetRecord)
async def sync_to_hf(
    dataset_id: str,
    req: SyncRequest,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Push the uploaded metadata file to a HuggingFace dataset repo."""
    d = await _require_dataset(session, dataset_id, user)
    if d.kind != "upload":
        raise HTTPException(status_code=400, detail="sync is only for kind=upload datasets")
    if not d.storage_id or not d.metadata_filename:
        raise HTTPException(status_code=400, detail="upload a metadata file first")
    repo = req.hf_repo.strip()
    if "/" not in repo:
        raise HTTPException(status_code=400, detail="hf_repo must be owner/name")

    storage = await _load_storage(session, d.storage_id)
    target, _ = _s3_target_and_prefix(storage)
    key = _metadata_key(storage, dataset_id, d.metadata_filename)
    text = await _run_sync(bench.s3_get_text, key, target)
    if text is None:
        raise HTTPException(status_code=502, detail="could not read metadata file from storage")

    # HF token: prefer a kind=huggingface storage if one exists, else env.
    hf_store = (await session.execute(
        select(Storage).where(Storage.kind == "huggingface").limit(1)
    )).scalars().first()
    token = await _hf_token(hf_store, session)
    if not token:
        raise HTTPException(status_code=400, detail="no HuggingFace token — add a HF storage or set HF_TOKEN")

    try:
        rev = await _run_sync(_hf_upload, repo, d.metadata_filename, text.encode("utf-8"), token, req.private)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"HuggingFace sync failed: {e}") from e

    d.hf_repo = repo
    d.hf_revision = rev
    d.hf_synced_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(d)
    await audit_module.record(user, "dataset.sync", "dataset", dataset_id, d.name, details={"hf_repo": repo})
    owner = await session.get(User, d.owner_id)
    return _to_record(d, owner.username if owner else "", storage.name)


@router.post("/{dataset_id}/transform", response_model=DatasetRecord)
async def transform_dataset(
    dataset_id: str,
    req: TransformRequest,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Kick off the audio-zip → audio-column transform as a gateway background
    job. Poll GET /{id} (transform_status / transform_log) for progress."""
    d = await _require_dataset(session, dataset_id, user)
    if d.kind == "label":
        tok = await _label_token(d, session)
        if not (d.label_base_url and d.label_project_id and tok):
            raise HTTPException(status_code=400, detail="label dataset needs a base URL, project, and stored token to transform")
    elif not (d.hf_repo and "/" in d.hf_repo):
        raise HTTPException(status_code=400, detail="transform needs a source HuggingFace repo (owner/name) on the dataset")
    if d.transform_status == "running":
        raise HTTPException(status_code=409, detail="a transform is already running for this dataset")
    if req.target not in ("hf", "s3"):
        raise HTTPException(status_code=400, detail="target must be 'hf' or 's3'")
    if req.target == "hf" and not (req.hf_repo and "/" in req.hf_repo):
        raise HTTPException(status_code=400, detail="hf_repo (owner/name) is required for target=hf")
    if req.target == "s3":
        if not req.storage_id:
            raise HTTPException(status_code=400, detail="storage_id is required for target=s3")
        st = await session.get(Storage, req.storage_id)
        if st is None or st.kind != "s3":
            raise HTTPException(status_code=400, detail="storage_id must reference a kind=s3 storage")

    from . import dataset_transform
    await dataset_transform.start_transform(
        dataset_id, req.target, (req.hf_repo or "").strip() or None, req.storage_id,
        s3_folder=(req.s3_folder or "").strip() or None,
    )
    await audit_module.record(user, "dataset.transform", "dataset", dataset_id, d.name, details={"target": req.target})
    await session.refresh(d)
    return _to_record(d, user.username, None)


@router.post("/{dataset_id}/pack-tts", response_model=DatasetRecord)
async def pack_tts_dataset(
    dataset_id: str,
    req: TtsPackRequest,
    request: Request,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """NeuCodec-encode + multipack this {audio, transcription} dataset into a
    ChiniDataset on a GPU provider over SSH, then create a new packed dataset.
    Implemented as a pack-only TTS training job (reuses the training runner); the
    source dataset's transform_status/transform_log track it for the UI."""
    d = await _require_dataset(session, dataset_id, user)
    if d.kind == "tts_packed":
        raise HTTPException(status_code=400, detail="this dataset is already packed")
    if d.kind == "label":
        tok = await _label_token(d, session)
        if not (d.label_base_url and d.label_project_id and tok):
            raise HTTPException(status_code=400, detail="label dataset needs a base URL, project, and stored token to pack")
    if d.transform_status == "running":
        raise HTTPException(status_code=409, detail="a transform is already running for this dataset")
    st = await session.get(Storage, req.storage_id)
    if st is None or st.kind != "s3":
        raise HTTPException(status_code=400, detail="storage_id must reference a kind=s3 storage")

    from .training_api import CreateTrainingRunRequest, create_training_run
    body = CreateTrainingRunRequest(
        name=f"pack-{d.name}"[:128],
        dataset_id=dataset_id,
        base_model="Qwen/Qwen3-1.7B-Base",  # unused in pack-only (no training)
        task_type="tts",
        provider_id=req.provider_id, storage_id=req.storage_id,
        gpu_count=req.gpu_count, visible_devices=req.visible_devices,
        tokenizer=req.tokenizer or None,
        pack_sequence_length=req.sequence_length,
        pack_only=True, pack_source_dataset_id=dataset_id,
        max_epochs=1,
    )
    run = await create_training_run(body, request, user, session)
    d.transform_status = "running"
    d.transform_log = f"TTS pack queued (run {run.id}) · seq_len {req.sequence_length} · provider {req.provider_id}"
    await session.commit()
    await audit_module.record(user, "dataset.pack-tts", "dataset", dataset_id, d.name,
                              details={"run": run.id, "sequence_length": req.sequence_length})
    await session.refresh(d)
    return _to_record(d, user.username, None)


def _hf_upload(repo: str, path_in_repo: str, body: bytes, token: str, private: bool) -> Optional[str]:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=True)
    info = api.upload_file(
        path_or_fileobj=io.BytesIO(body),
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Upload {path_in_repo} via GPU Platform",
    )
    return getattr(info, "oid", None) or getattr(info, "commit_oid", None)


async def _run_sync(fn, *args):
    """Run a blocking (boto3/httpx/hf) call off the event loop."""
    from fastapi.concurrency import run_in_threadpool
    return await run_in_threadpool(fn, *args)
