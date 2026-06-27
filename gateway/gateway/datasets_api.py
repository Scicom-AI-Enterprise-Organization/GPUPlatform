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

KINDS = ("upload", "s3", "hf", "label", "tts_packed", "llm", "llm_packed")
_UPLOAD_EXTS = (".csv", ".json", ".jsonl", ".ndjson")


# ---------- request / response models ----------------------------------


class CreateDatasetRequest(BaseModel):
    name: str
    kind: str = "upload"  # upload | s3 | hf | label | tts_packed
    storage_id: Optional[str] = None
    description: Optional[str] = None
    audio_prefix: Optional[str] = None
    s3_metadata_uri: Optional[str] = None  # kind=s3 (metadata file) / kind=tts_packed|llm_packed (shards prefix)
    # kind=tts_packed / llm_packed — register existing ChiniDataset parquet shards already in S3.
    tokenizer: Optional[str] = None        # tokenizer used when packing
    sequence_length: Optional[int] = None  # multipack sequence length
    # kind=llm_packed — the source subset/config that was packed (descriptive metadata).
    subset: Optional[str] = None
    hf_repo: Optional[str] = None  # kind=hf / kind=llm
    # kind=hf / kind=llm — git revision to pin: a commit SHA (full/short), branch
    # ("main", "dev"), or tag ("v1.0.0"). Blank → the repo's default branch.
    hf_revision: Optional[str] = None
    # kind=llm / kind=llm_packed — which column holds the OpenAI messages array ([{role,content}]).
    messages_field: Optional[str] = None
    # kind=label — live import from a labeling-platform project's export API.
    label_base_url: Optional[str] = None     # e.g. http://localhost:3002
    label_project_id: Optional[str] = None   # project UUID
    label_token: Optional[str] = None        # lpat_… (stored Fernet-encrypted, never returned)
    label_token_secret: Optional[str] = None # OR: a global-secret key holding the lpat token
    label_status: Optional[str] = None       # approved | rejected | not_reviewed | all (default approved)
    label_updated_until: Optional[str] = None # ISO-8601 cutoff — import only tasks last updated at/before it


class UpdateDatasetRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    audio_prefix: Optional[str] = None
    audio_field: Optional[str] = None
    transcription_field: Optional[str] = None
    # TTS-only speaker column. Pass "" to clear (one-voice packing).
    speaker_field: Optional[str] = None
    # Per-split transcription column overrides, e.g. {"train": "text", "test": "after"}.
    # Pass {} to clear. Splits not listed fall back to transcription_field.
    split_fields: Optional[dict[str, str]] = None
    # kind=llm — which column holds the messages array. Pass "" to reset to default.
    messages_field: Optional[str] = None
    # kind=label import filters. None → leave unchanged. label_status switches the
    # review-status filter; label_updated_until sets/clears (pass "") the ISO-8601
    # point-in-time cutoff. Changing either re-counts the dataset's rows.
    label_status: Optional[str] = None
    label_updated_until: Optional[str] = None


class RowInclusionRequest(BaseModel):
    """Tick / un-tick rows in the row browser. included=False excludes the given
    metadata-file row indices from training; included=True re-includes them.
    clear=True re-includes ALL rows (ignores indices)."""
    indices: list[int] = []
    included: bool = True
    clear: bool = False


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
    ChiniDataset, on a GPU provider over SSH, → a new packed dataset. Provisioning
    mirrors Autotrain (pack runs through the training runner): a VM provider_id
    SSHes onto that box; otherwise a fresh RunPod pod is spawned (provider_id =
    a RunPod account, or null for the gateway default key) with gpu_type/tier."""
    provider_id: Optional[str] = None      # kind=vm → bare metal; runpod acct / null → spawn a pod
    storage_id: str                        # s3 storage for the packed shards
    tokenizer: Optional[str] = None        # speech-token tokenizer (pack_stage1)
    sequence_length: int = 4096            # multipack block length
    gpu_count: int = 1
    visible_devices: Optional[str] = None
    # Isolated uv venv for the NeuCodec/TTS deps (mirrors Autotrain). Reused +
    # cached across packs/runs. None → /share/neucodec-tts (dedicated NeuCodec venv).
    venv_path: Optional[str] = None
    # RunPod pod knobs (ignored for a VM provider).
    gpu_type: str = "NVIDIA L40S"
    secure_cloud: bool = True
    disk_gb: int = 60
    volume_gb: int = 80


class OmnivoicePackRequest(BaseModel):
    """Higgs-codec tokenize a {audio, transcription} dataset into OmniVoice
    WebDataset shards (→ kind=omnivoice_packed), on a GPU box over SSH. Same
    provisioning as Pack-for-TTS, but the OmniVoice stack (torch 2.8/cu128) — so a
    RunPod pod should use a CUDA-12.8 image (NOT the cu13 gemma image)."""
    provider_id: Optional[str] = None
    storage_id: str
    tokenizer: Optional[str] = None        # Higgs codec (default eustlb/higgs-audio-v2-tokenizer)
    default_language: Optional[str] = "en"  # language_id when the dataset has no language column
    language_field: Optional[str] = None    # dataset column holding per-row language_id
    eval_test_per_speaker: int = 25         # held-out clips/speaker for the voice-clone eval set
    gpu_count: int = 1
    visible_devices: Optional[str] = None
    venv_path: Optional[str] = None         # None → /share/autotrain-omnivoice
    # RunPod pod knobs (ignored for a VM provider). OmniVoice needs CUDA 12.8.
    gpu_type: str = "NVIDIA H100 80GB HBM3"
    image: Optional[str] = None             # None → a cu128 pytorch image (set by the endpoint)
    secure_cloud: bool = True
    disk_gb: int = 80
    volume_gb: int = 80


class LlmPackRequest(BaseModel):
    """Chat → multipack a kind=llm dataset (its `messages` column, + an optional
    tools column) into a ChiniDataset (kind=llm_packed), the SAME layout the
    gemma4 LLM trainer consumes. Pure CPU tokenization → runs IN-PROCESS in the
    gateway (no GPU box). The user picks the subset + tokenizer."""
    storage_id: str                        # kind=s3 storage for the packed shards
    tokenizer: str                         # HF tokenizer (chat template), e.g. google/gemma-4-31B-it
    subset: Optional[str] = None           # single subset/split label to pack (None → first); legacy/fallback
    # Multiple subset/split labels packed together into ONE ChiniDataset (their rows
    # concatenated in selection order). Takes precedence over `subset`; None/empty →
    # fall back to `subset` (or the first split).
    subsets: Optional[list[str]] = None
    sequence_length: int = 32768           # multipack bin length (tokens); convs longer are dropped
    # Source column of OpenAI-style tool/function declarations rendered as tools=
    # into the chat template. Blank/None → no tools. Default "functions".
    tools_field: Optional[str] = "functions"
    # For templates that gate reasoning to tool-call turns after the last user
    # message (gemma-4, MiniMax-M2, …): render EVERY assistant turn's reasoning.
    # No-op on templates without that guard. See llm_pack.build_chat_template.
    all_reasoning: bool = True


class DatasetMergeRequest(BaseModel):
    """Concatenate several existing kind=label datasets into ONE combined audio
    dataset (their clips downloaded + paired with transcription, then written to
    HF or S3 — the SAME output a single-project transform produces). The merge
    runs as an in-process background job; status/log live on the NEW dataset."""
    source_ids: list[str]                  # >= 2 kind=label datasets to concatenate
    target: str = "s3"                     # "hf" | "s3"
    hf_repo: Optional[str] = None          # owner/name (target=hf)
    storage_id: Optional[str] = None       # kind=s3 storage (target=s3)
    s3_folder: Optional[str] = None        # blank → datasets/{new_id}/transformed
    name: Optional[str] = None             # output dataset name (blank → auto)


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
    speaker_field: Optional[str] = None  # TTS-only speaker column (None → one voice)
    # Usually {split: transcription_column} (str→str), but a packed (tts_packed)
    # dataset stashes a nested {"_tts_pack": {...}} metadata blob here — so the
    # response type tolerates non-string values (don't tighten back to str).
    split_fields: Optional[dict[str, Any]] = None
    audio_dataset_id: Optional[str] = None  # materialised S3 audio dataset (if any)
    # Lineage: a transformed dataset (hf/label → audio) records the dataset it was
    # derived from + that source's original HF repo (computed, not stored).
    source_dataset_id: Optional[str] = None
    source_name: Optional[str] = None
    source_hf_repo: Optional[str] = None
    hf_repo: Optional[str] = None
    hf_revision: Optional[str] = None
    hf_synced_at: Optional[str] = None
    label_base_url: Optional[str] = None     # kind=label source (token never returned)
    label_project_id: Optional[str] = None
    label_status: Optional[str] = None
    label_updated_until: Optional[str] = None  # ISO-8601 point-in-time import cutoff (None → no upper bound)
    label_token_secret: Optional[str] = None  # global-secret key (if used instead of a stored token)
    transform_status: Optional[str] = None  # "" | running | done | failed
    transform_log: Optional[str] = None     # short tail of progress lines
    # kind=llm: which column holds the OpenAI messages array
    messages_field: Optional[str] = None
    # When published to the self-hosted HF mirror: the CatalogRepo id serving it
    # over /hf (None = not published). The dataset page links + shows pull snippets.
    catalog_repo_id: Optional[str] = None
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
    speakers: Optional[list[str]] = None  # distinct speaker values (for a filter dropdown)
    speaker: Optional[str] = None  # the selected speaker filter, echoed back
    excluded_count: int = 0  # rows manually un-ticked (excluded from training)
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


def _norm_cutoff(raw: Optional[str]) -> Optional[str]:
    """Validate + canonicalise a kind=label import cutoff (the `updated_until`
    filter). Accepts any ISO-8601 instant (a trailing `Z` or an offset); a naive
    value is read as UTC. Returns a canonical UTC ISO-8601 string (`…+00:00`) the
    Label export compares lexicographically, or None for a blank/cleared value.
    Raises HTTPException(400) on an unparseable timestamp."""
    s = (raw or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid timestamp: {s!r} (expected ISO-8601)") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _to_record(
    d: Dataset,
    owner_username: str,
    storage_name: Optional[str],
    *,
    source_dataset_id: Optional[str] = None,
    source_name: Optional[str] = None,
    source_hf_repo: Optional[str] = None,
) -> DatasetRecord:
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
        speaker_field=getattr(d, "speaker_field", None) or None,
        split_fields=getattr(d, "split_fields", None) or None,
        audio_dataset_id=getattr(d, "audio_dataset_id", None) or None,
        source_dataset_id=source_dataset_id,
        source_name=source_name,
        source_hf_repo=source_hf_repo,
        hf_repo=d.hf_repo,
        hf_revision=d.hf_revision,
        hf_synced_at=_iso(d.hf_synced_at),
        label_base_url=getattr(d, "label_base_url", None),
        label_project_id=getattr(d, "label_project_id", None),
        label_status=getattr(d, "label_status", None),
        label_updated_until=getattr(d, "label_updated_until", None),
        label_token_secret=getattr(d, "label_token_secret", None),
        messages_field=getattr(d, "messages_field", None) or None,
        transform_status=getattr(d, "transform_status", None),
        transform_log=getattr(d, "transform_log", None),
        catalog_repo_id=getattr(d, "catalog_repo_id", None),
        created_at=_iso(d.created_at) or "",
        updated_at=_iso(d.updated_at) or "",
        created_by=owner_username,
    )


async def _s3_folder_size(storage: Storage, s3_metadata_uri: str) -> Optional[int]:
    """Total bytes under a materialised dataset's S3 folder (the directory holding
    its metadata file + audio/). Best-effort: returns None on any S3 error."""
    try:
        target, _ = _s3_target_and_prefix(storage)
        u = urlparse(s3_metadata_uri)
        if u.scheme == "s3":
            target = dataclasses.replace(target, bucket=u.netloc)
            key = u.path.lstrip("/")
        else:
            key = s3_metadata_uri
        folder = (key.rsplit("/", 1)[0] + "/") if "/" in key else ""
        objs = await _run_sync(bench.s3_list, folder, target)
        return sum(int(o.get("size") or 0) for o in objs)
    except Exception as e:  # noqa: BLE001
        logger.warning("folder-size compute failed for %s: %s", s3_metadata_uri, e)
        return None


async def _tts_pack_prefix_size(storage: Storage, s3_uri: str) -> Optional[int]:
    """Total bytes under a packed dataset's S3 PREFIX (s3_metadata_uri is the dir
    holding the parquet shards, incl. per-split subdirs) — not the parent folder.
    Best-effort: returns None on any S3 error."""
    try:
        target, _ = _s3_target_and_prefix(storage)
        u = urlparse(s3_uri)
        if u.scheme == "s3":
            target = dataclasses.replace(target, bucket=u.netloc)
            key = u.path.lstrip("/")
        else:
            key = s3_uri
        prefix = key.rstrip("/") + "/"
        objs = await _run_sync(bench.s3_list, prefix, target)
        return sum(int(o.get("size") or 0) for o in objs)
    except Exception as e:  # noqa: BLE001
        logger.warning("packed prefix-size compute failed for %s: %s", s3_uri, e)
        return None


def _introspect_packed_s3(target: "S3Target", prefix: str) -> dict:
    """Inspect existing ChiniDataset shards under `prefix` for registering them as a
    tts_packed dataset: per-split record counts (from the <prefix>/<split>/*.parquet
    layout; flat shards → no split dimension) + total bytes. Counts come from each
    shard's parquet footer (num_rows) — no row data loaded. Sync; call via _run_sync."""
    import io as _io
    import pyarrow.parquet as pq

    objs = bench.s3_list(prefix, target)
    if not objs:
        raise RuntimeError("no objects found under the prefix")
    size = sum(int(o.get("size") or 0) for o in objs)
    shards = sorted(o["key"] for o in objs if o["key"].endswith(".parquet"))
    if not shards:
        raise RuntimeError("no .parquet shards under the prefix — is this a packed (ChiniDataset) folder?")
    cli = bench._s3_client(target)
    splits: dict[str, int] = {}
    flat_total = 0
    for key in shards:
        rel = key[len(prefix):] if key.startswith(prefix) else key
        split = rel.split("/", 1)[0] if "/" in rel else None
        body = cli.get_object(Bucket=target.bucket, Key=key)["Body"].read()
        n = int(pq.read_metadata(_io.BytesIO(body)).num_rows)
        if split:
            splits[split] = splits.get(split, 0) + n
        else:
            flat_total += n
    # When split subdirs exist they're authoritative (preview/training read
    # <prefix>/<split>/); any flat top-level shards are a combined/duplicate copy,
    # so don't double-count them. Flat-only datasets fall back to the flat total.
    total = sum(splits.values()) if splits else flat_total
    return {"splits": splits, "total_rows": total, "size_bytes": size}


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


async def _hf_endpoint(storage: Optional[Storage], session: AsyncSession) -> Optional[str]:
    """Custom HF Hub endpoint (HF_ENDPOINT) for a kind=huggingface storage, or
    None for huggingface.co. Precedence: a referenced global secret
    (config.endpoint_secret) > the storage's literal `endpoint`."""
    cfg = (storage.config or {}) if storage is not None else {}
    ref = cfg.get("endpoint_secret")
    if ref:
        from .global_env_api import load_global_env
        val = (await load_global_env(session)).get(ref)
        if val and val.strip():
            return val.strip().rstrip("/")
    ep = (cfg.get("endpoint") or "").strip()
    return ep.rstrip("/") or None


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
    label_updated_until_val: Optional[str] = None
    label_num_rows: Optional[int] = None
    if req.kind in ("upload", "s3", "tts_packed", "llm_packed"):
        if not req.storage_id:
            raise HTTPException(status_code=400, detail="storage_id (an S3 storage) is required")
        storage = await _load_storage(session, req.storage_id)
        if storage.kind != "s3":
            raise HTTPException(status_code=400, detail="storage must be kind=s3 for upload / s3 / tts_packed / llm_packed datasets")
        storage_name = storage.name
        if req.kind in ("s3", "tts_packed", "llm_packed") and not (req.s3_metadata_uri or "").strip():
            raise HTTPException(
                status_code=400,
                detail="s3_metadata_uri is required (the metadata file for s3, the shards prefix for tts_packed / llm_packed)",
            )
    elif req.kind in ("hf", "llm"):
        if not (req.hf_repo or "").strip():
            raise HTTPException(status_code=400, detail="hf_repo (owner/name) is required for kind=hf/llm")
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
        label_updated_until_val = _norm_cutoff(req.label_updated_until)
        # Verify the token + project reachable, and grab the row count (under the
        # status + cutoff filters), by reading the export header (limit=0 stops after
        # the first line).
        try:
            _, label_num_rows = await _run_sync(
                _label_export_rows, label_base_url, label_project_id, verify_tok,
                label_status_val, 0, 0, label_updated_until_val,
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
        hf_revision=(req.hf_revision or "").strip() or None if req.kind in ("hf", "llm") else None,
        label_base_url=label_base_url,
        label_project_id=label_project_id,
        label_token_enc=label_token_enc,
        label_token_secret=label_token_secret_val,
        label_status=label_status_val,
        label_updated_until=label_updated_until_val,
    )
    if req.kind == "label":
        # The Label export uses audio_url + transcription columns.
        row.audio_field = "audio_url"
        row.transcription_field = "transcription"
        row.num_rows = label_num_rows
    elif req.kind == "s3":
        # Introspect the metadata file up front so format/num_rows + the
        # audio/transcription/speaker mappings are correct from the first view
        # (CreateDatasetRequest carries no field hints). Best-effort: a transient
        # read/parse issue must not block creation — the preview re-heals on open.
        try:
            target, _base = _s3_target_and_prefix(storage)
            u = urlparse(row.s3_metadata_uri or "")
            t = dataclasses.replace(target, bucket=u.netloc) if u.scheme == "s3" else target
            key = u.path.lstrip("/") if u.scheme == "s3" else (row.s3_metadata_uri or "")
            mdname = os.path.basename(key)
            text = await _run_sync(bench.s3_get_text, key, t)
            if text:
                all_rows = dataset_metadata.parse_rows(mdname, text.encode("utf-8"), 10**9)
                cols = list(all_rows[0].keys()) if all_rows else []
                try:
                    fmt = dataset_metadata.detect_format(mdname)
                except dataset_metadata.DatasetParseError:
                    fmt = None
                _stamp_detected_fields(row, cols, len(all_rows), fmt)
        except Exception as e:  # noqa: BLE001 — best-effort; preview re-heals
            logger.warning("s3 metadata introspect failed for new dataset %s: %s", did, e)
    elif req.kind == "llm":
        # LLM / chat dataset: the primary column is messages ([{role,content}]).
        # We don't introspect HF up front (same as kind=hf) — preview does it lazily.
        mf = (req.messages_field or "").strip() or "messages"
        row.messages_field = mf
    elif req.kind == "tts_packed":
        # Register existing ChiniDataset shards: introspect the prefix for splits +
        # per-split counts + total size, and stamp the _tts_pack metadata so preview
        # / decode / training treat it exactly like a packed dataset the pack job
        # produced. Fail loudly if the prefix has no shards — a broken packed dataset
        # is worse than a clear error.
        try:
            target, _base = _s3_target_and_prefix(storage)
            u = urlparse(row.s3_metadata_uri or "")
            t = dataclasses.replace(target, bucket=u.netloc) if u.scheme == "s3" else target
            prefix = (u.path.lstrip("/") if u.scheme == "s3" else (row.s3_metadata_uri or "")).rstrip("/") + "/"
            info = await _run_sync(_introspect_packed_s3, t, prefix)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail=f"could not read packed shards at {row.s3_metadata_uri}: {e}",
            )
        row.format = "chinidataset"
        row.num_rows = info["total_rows"]
        row.size_bytes = info["size_bytes"]
        row.audio_field = "audio"
        row.transcription_field = "text"
        row.split_fields = {"_tts_pack": {
            "tokenizer": (req.tokenizer or "").strip() or _DEFAULT_TTS_TOKENIZER,
            "sequence_length": int(req.sequence_length or 4096),
            "splits": info["splits"],
        }}
    elif req.kind == "llm_packed":
        # Register existing chat-multipack ChiniDataset shards already in S3 (the
        # same layout the LLM pack job produces). Introspect the prefix for counts +
        # size and stamp the _llm_pack metadata so preview / packed-row / training
        # treat it like a packed dataset the job made. Fail loudly on an empty prefix.
        try:
            target, _base = _s3_target_and_prefix(storage)
            u = urlparse(row.s3_metadata_uri or "")
            t = dataclasses.replace(target, bucket=u.netloc) if u.scheme == "s3" else target
            prefix = (u.path.lstrip("/") if u.scheme == "s3" else (row.s3_metadata_uri or "")).rstrip("/") + "/"
            info = await _run_sync(_introspect_packed_s3, t, prefix)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail=f"could not read packed shards at {row.s3_metadata_uri}: {e}",
            )
        mf = (req.messages_field or "").strip() or "messages"
        row.format = "chinidataset"
        row.num_rows = info["total_rows"]
        row.size_bytes = info["size_bytes"]
        row.messages_field = mf
        row.split_fields = {"_llm_pack": {
            "tokenizer": (req.tokenizer or "").strip() or None,
            "sequence_length": int(req.sequence_length or 32768),
            "messages_field": mf,
            "subset": (req.subset or "").strip() or None,
            "splits": info["splits"],
        }}
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

    # Transformed datasets are created without a size — compute it once from the
    # materialised S3 folder (metadata + audio/) and cache it on the row.
    if (
        d.kind == "s3" and d.size_bytes is None and d.s3_metadata_uri
        and storage and d.transform_status != "running"
    ):
        size = await _s3_folder_size(storage, d.s3_metadata_uri)
        if size:
            d.size_bytes = size
            await session.commit()
            await session.refresh(d)

    # Packed datasets: the pack job records rows + the s3 prefix but not the format
    # or on-disk size. Backfill both once — format is always "chinidataset" (the
    # multipacked NeuCodec layout); size = sum of the parquet shards under the
    # prefix (train/ + test/ subdirs included).
    if d.kind == "tts_packed" and d.s3_metadata_uri and storage and d.transform_status != "running":
        _changed = False
        if not d.format:
            d.format = "chinidataset"
            _changed = True
        if d.size_bytes is None:
            size = await _tts_pack_prefix_size(storage, d.s3_metadata_uri)
            if size:
                d.size_bytes = size
                _changed = True
        if _changed:
            await session.commit()
            await session.refresh(d)

    # Lineage: the transform sets the SOURCE dataset's audio_dataset_id → this one,
    # so surface the original dataset (+ its HF repo) this was derived from.
    src = (
        await session.execute(select(Dataset).where(Dataset.audio_dataset_id == d.id))
    ).scalars().first()

    return _to_record(
        d, owner.username if owner else "", storage.name if storage else None,
        source_dataset_id=src.id if src else None,
        source_name=src.name if src else None,
        source_hf_repo=(src.hf_repo if src else None),
    )


class DatasetFile(BaseModel):
    name: str            # key relative to the listed prefix
    key: str             # full S3 key
    size: int = 0
    modified: str = ""   # ISO8601
    download_url: str    # presigned GET (1h)


@router.get("/{dataset_id}/files", response_model=list[DatasetFile])
async def list_dataset_files(
    dataset_id: str,
    split: Optional[str] = Query(None, description="for a split-aware tts_packed dataset, list only this split's subdir"),
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """List the S3 objects backing a dataset, with presigned download links. For a
    split-aware tts_packed dataset, `?split=` narrows to that split's subdir.
    Returns [] for datasets with no S3 backing (hf / label)."""
    d = await _require_dataset(session, dataset_id, user)
    storage = await session.get(Storage, d.storage_id) if d.storage_id else None
    if storage is None or storage.kind != "s3":
        return []
    target, base = _s3_target_and_prefix(storage)

    # Resolve the bucket + prefix to list under, from s3_metadata_uri (an absolute
    # s3:// URI for tts_packed/transformed datasets, else a key under the storage
    # prefix), falling back to the dataset's own datasets/<id>/ folder.
    uri = (d.s3_metadata_uri or "").strip()
    if uri.startswith("s3://"):
        u = urlparse(uri)
        if u.netloc:
            target = dataclasses.replace(target, bucket=u.netloc)
        listing = u.path.lstrip("/").rstrip("/")
    elif uri:
        listing = _join_key(base, uri)
    else:
        listing = _join_key(base, "datasets", dataset_id)

    # If s3_metadata_uri points at a single metadata FILE (has an extension), list
    # its parent dir instead. tts_packed's URI is already a prefix dir.
    if listing and d.kind != "tts_packed":
        last = listing.rsplit("/", 1)[-1]
        if "." in last:
            listing = listing.rsplit("/", 1)[0] if "/" in listing else ""

    # Split-aware tts_packed: narrow to <prefix>/<split>/ when that subdir exists.
    sp = (split or "").strip().strip("/")
    if sp and d.kind == "tts_packed" and listing:
        cand = f"{listing}/{sp}"
        try:
            if bench.s3_list(cand + "/", target=target):
                listing = cand
        except Exception:  # noqa: BLE001
            pass

    if not listing:
        return []
    pfx = listing.rstrip("/") + "/"
    out: list[DatasetFile] = []
    try:
        for obj in bench.s3_list(pfx, target=target):
            key = obj["key"]
            name = key[len(pfx):] if key.startswith(pfx) else key
            out.append(DatasetFile(
                name=name, key=key, size=obj.get("size", 0),
                modified=obj.get("modified", ""),
                download_url=bench.s3_presign_get(key, target=target),
            ))
            if len(out) >= 1000:  # cap — huge audio datasets can have many objects
                break
    except Exception as e:  # noqa: BLE001
        logger.warning("dataset %s: file list failed: %s", dataset_id, e)
    return out


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
    if req.speaker_field is not None:
        # Blank clears it → the packer falls back to a single constant speaker.
        d.speaker_field = req.speaker_field.strip() or None
    if req.split_fields is not None:
        # {} clears the overrides; otherwise keep only non-blank split→column pairs.
        cleaned = {
            str(k).strip(): str(v).strip()
            for k, v in req.split_fields.items()
            if str(k).strip() and str(v).strip()
        }
        d.split_fields = cleaned or None
    if req.messages_field is not None:
        d.messages_field = req.messages_field.strip() or None
    # kind=label import filters. Changing the status or the point-in-time cutoff
    # changes which tasks the dataset materialises, so re-count the rows (best-effort
    # — a transient platform/token issue must not block the metadata edit).
    label_filter_changed = False
    if d.kind == "label":
        if req.label_status is not None:
            st = req.label_status.strip() or "approved"
            if st not in ("approved", "rejected", "not_reviewed", "all"):
                raise HTTPException(status_code=400, detail=f"invalid label_status: {st}")
            if st != (d.label_status or "approved"):
                d.label_status = st
                label_filter_changed = True
        if req.label_updated_until is not None:
            cutoff = _norm_cutoff(req.label_updated_until)  # "" clears it → None
            if cutoff != d.label_updated_until:
                d.label_updated_until = cutoff
                label_filter_changed = True
    if label_filter_changed:
        tok = await _label_token(d, session)
        if d.label_base_url and d.label_project_id and tok:
            try:
                _, d.num_rows = await _run_sync(
                    _label_export_rows, d.label_base_url, d.label_project_id, tok,
                    d.label_status or "approved", 0, 0, d.label_updated_until,
                )
            except Exception:  # noqa: BLE001 — keep the stale count rather than fail the edit
                pass
    await session.commit()
    await session.refresh(d)
    storage = await session.get(Storage, d.storage_id) if d.storage_id else None
    owner = await session.get(User, d.owner_id)
    await audit_module.record(user, "dataset.update", "dataset", d.id, d.name)
    return _to_record(d, owner.username if owner else "", storage.name if storage else None)


def _dataset_storage_prefix(d: Dataset, storage: Optional[Storage]) -> Optional[str]:
    """The S3 key prefix holding this dataset's materialised files (for
    purge-on-delete), or None when nothing under our storage is the dataset's to
    delete — kind=hf is pushed to a HF repo, kind=label lives on the labeling
    platform, and a dataset with no S3 backing has nothing here."""
    uri = d.s3_metadata_uri or ""
    if uri.startswith("s3://"):
        key = urlparse(uri).path.lstrip("/")
        if d.kind == "s3":
            # metadata.csv lives at {base}/metadata.csv → purge the {base}/ folder.
            return (key.rsplit("/", 1)[0] + "/") if "/" in key else None
        if d.kind in ("tts_packed", "llm_packed"):
            return key.rstrip("/") + "/"  # the URI already points at the shards prefix
    if d.kind == "upload" and storage is not None and storage.kind == "s3" and d.metadata_filename:
        _, base = _s3_target_and_prefix(storage)
        return _join_key(base, "datasets", d.id) + "/"
    return None


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    purge: bool = Query(False, description="also delete the dataset's files in S3 storage"),
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Delete the dataset record. With `purge=true`, ALSO delete the dataset's
    files in S3 storage first (only for S3-backed kinds: s3 / tts_packed /
    llm_packed / upload). If a requested purge fails, the record is kept so the
    caller can retry rather than orphaning files."""
    d = await _require_dataset(session, dataset_id, user)
    name = d.name
    purged = 0
    if purge:
        storage = await session.get(Storage, d.storage_id) if d.storage_id else None
        prefix = _dataset_storage_prefix(d, storage)
        if prefix and storage is not None and storage.kind == "s3":
            try:
                target, _ = _s3_target_and_prefix(storage)
                purged = await _run_sync(bench.s3_delete_prefix, prefix, target)
            except Exception as e:  # noqa: BLE001
                logger.warning("purge failed for dataset %s (prefix=%s): %s", dataset_id, prefix, e)
                raise HTTPException(status_code=502, detail=f"failed to purge storage files: {e}") from e
    await session.delete(d)
    await session.commit()
    await audit_module.record(user, "dataset.delete", "dataset", dataset_id, name,
                              details={"purge": purge, "purged_objects": purged})
    return {"ok": True, "id": dataset_id, "purged_objects": purged}


# ---------- publish to the self-hosted HF mirror -----------------------


class PublishResult(BaseModel):
    repo_id: str       # CatalogRepo id (repo-…)
    full_id: str       # "ns/name"
    repo_type: str     # always "dataset"
    num_files: int
    size_bytes: int


def _sanitize_repo_part(s: str) -> str:
    """Coerce a string into a valid HF repo namespace/name segment
    (`[A-Za-z0-9][A-Za-z0-9._-]*`)."""
    out = "".join(c if (c.isalnum() or c in "._-") else "-" for c in (s or "").strip())
    while out and not out[0].isalnum():
        out = out[1:]
    return out or "x"


@router.post("/{dataset_id}/publish", response_model=PublishResult)
async def publish_dataset(
    dataset_id: str,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Expose an S3-backed Autotrain dataset over the HF mirror as a hosted
    dataset repo (a CatalogRepo over its S3 prefix), so it's pullable with
    `hf download <ns>/<name> --repo-type dataset`. Idempotent."""
    from fastapi.concurrency import run_in_threadpool
    from sqlalchemy import select as _select
    from . import storage_backends as sb
    from .catalog_api import _reindex_manifest
    from .db import CatalogRepo
    from .hf_mirror_api import _compute_sha

    PUBLISHABLE = ("s3", "upload", "tts_packed")
    d = await _require_dataset(session, dataset_id, user)

    # Resolve which dataset's S3 files we actually serve. An hf/label dataset
    # holds only an external reference, but if it's been materialised to S3
    # (audio_dataset_id), publish that twin and link BOTH back to the same repo.
    target = d
    if d.kind not in PUBLISHABLE:
        twin = await session.get(Dataset, d.audio_dataset_id) if d.audio_dataset_id else None
        if twin is not None and twin.kind in PUBLISHABLE and twin.storage_id and twin.s3_metadata_uri:
            target = twin
        else:
            hint = (f" — its data lives on huggingface.co ({d.hf_repo}); use that directly"
                    if d.hf_repo else " — materialise it to S3 first (Transform → S3)")
            raise HTTPException(status_code=400,
                                detail=f"a '{d.kind}' dataset has no files in your storage to serve{hint}.")
    if not target.storage_id or not target.s3_metadata_uri:
        raise HTTPException(status_code=400, detail="dataset has no S3 storage/location to publish")
    storage = await _load_storage(session, target.storage_id)
    if storage.kind != "s3":
        raise HTTPException(status_code=400, detail="dataset storage is not S3-backed")

    def _link(repo: "CatalogRepo") -> None:
        d.catalog_repo_id = repo.id
        target.catalog_repo_id = repo.id

    # Idempotent — if the requested dataset OR its target twin is already published.
    existing_id = d.catalog_repo_id or target.catalog_repo_id
    if existing_id:
        existing = await session.get(CatalogRepo, existing_id)
        if existing is not None:
            _link(existing)
            await session.commit()
            return PublishResult(repo_id=existing.id, full_id=existing.full_id,
                                 repo_type=existing.repo_type, num_files=existing.num_files or 0,
                                 size_bytes=existing.size_bytes or 0)

    # Derive the repo prefix = the metadata file's S3 dir, relative to the
    # storage's base prefix (the S3 backend roots keys at storage.config.prefix).
    uri = target.s3_metadata_uri
    if uri.startswith("s3://"):
        bucket, _, key = uri[5:].partition("/")
        st_bucket = (storage.config or {}).get("bucket")
        if st_bucket and bucket and bucket != st_bucket:
            raise HTTPException(status_code=400,
                                detail=f"dataset files are in bucket '{bucket}' but its storage points at '{st_bucket}'")
    else:
        key = uri.lstrip("/")
    keydir = "/".join(key.split("/")[:-1])
    base = ((storage.config or {}).get("prefix") or "").strip().strip("/")
    if not base:
        repo_prefix = keydir
    elif keydir == base:
        repo_prefix = ""
    elif keydir.startswith(base + "/"):
        repo_prefix = keydir[len(base) + 1:]
    else:
        raise HTTPException(status_code=400,
                            detail=f"dataset files ({keydir}) aren't under the storage base prefix ({base})")
    if not repo_prefix:
        raise HTTPException(status_code=400, detail="could not derive a repo prefix for this dataset")

    owner = await session.get(User, target.owner_id)
    ns = _sanitize_repo_part(owner.username if owner else "user")
    name = _sanitize_repo_part(target.name) or target.id
    # Avoid colliding with a different repo of the same id.
    clash = (await session.execute(_select(CatalogRepo).where(
        CatalogRepo.repo_type == "dataset", CatalogRepo.namespace == ns, CatalogRepo.name == name,
    ))).scalar_one_or_none()
    if clash is not None:
        name = f"{name}-{target.id.split('-')[-1]}"
    full_id = f"{ns}/{name}"

    try:
        backend = await run_in_threadpool(sb.resolve_backend, storage)
        manifest, total = await run_in_threadpool(_reindex_manifest, backend, repo_prefix)
    except sb.StorageError as e:
        raise HTTPException(status_code=400, detail=f"storage error: {e}") from e
    if not manifest:
        raise HTTPException(status_code=400, detail=f"no files found under '{repo_prefix}' to publish")

    repo = CatalogRepo(
        id=f"repo-{secrets.token_hex(4)}",
        owner_id=target.owner_id,
        repo_type="dataset",
        namespace=ns,
        name=name,
        full_id=full_id,
        storage_id=storage.id,
        prefix=repo_prefix,
        sha=_compute_sha(manifest),
        private=True,
        description=f"Published from Autotrain dataset “{target.name}” ({target.id})",
        manifest=manifest,
        size_bytes=total,
        num_files=len(manifest),
    )
    session.add(repo)
    _link(repo)
    await session.commit()
    await audit_module.record(user, "dataset.publish", "dataset", dataset_id, d.name,
                              details={"repo": full_id, "served_from": target.id})
    logger.info("dataset %s published to HF mirror as %s (%d files, served from %s)",
                dataset_id, full_id, len(manifest), target.id)
    return PublishResult(repo_id=repo.id, full_id=full_id, repo_type="dataset",
                         num_files=len(manifest), size_bytes=total)


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
    # Auto-detect the speaker column too (None when absent → single speaker); a
    # re-upload re-detects against the new file.
    d.speaker_field = parsed.get("speaker_field")
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


def _hf_split_ident(splits: list[dict[str, Any]]):
    """Return a function that names each datasets-server split entry by whichever
    of config/split is the distinct one, so the row preview, the /splits picker
    and the per-split column map all agree on labels. A multi-config dataset
    (configs "test"/"train", each with a single split "train") is labelled by
    config; a normal single-config dataset ("default") is labelled by its split
    (train/test/validation). Reading the parquet directory names directly is
    unreliable — it surfaces the top-level dir (e.g. "data"/"default") rather than
    the actual split."""
    configs = [s["config"] for s in splits]
    snames = [s["split"] for s in splits]
    if splits and len(set(configs)) == len(splits):
        return lambda s: s["config"]  # noqa: E731
    if splits and len(set(snames)) == len(splits):
        return lambda s: s["split"]  # noqa: E731
    return lambda s: f'{s["config"]}/{s["split"]}'  # noqa: E731


def _hf_preview_rows(
    hf_repo: str, token: Optional[str], limit: int, offset: int = 0, split: Optional[str] = None,
    revision: Optional[str] = None,
) -> tuple[list[dict[str, Any]], Optional[int], Optional[str], list[str]]:
    """Fetch a page of rows for one split via the HF datasets-server API.
    Returns (rows, total_rows, used_split, all_split_names). `split` selects which
    split to read (default: the first); a split's full row count drives paging.
    `revision` (commit/branch/tag) pins the source ref when set."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    rev = {"revision": revision} if revision else {}
    with httpx.Client(timeout=20.0) as cli:
        sp = cli.get(
            "https://datasets-server.huggingface.co/splits",
            params={"dataset": hf_repo, **rev}, headers=headers,
        )
        sp.raise_for_status()
        splits = sp.json().get("splits", [])
        if not splits:
            return [], 0, None, []
        ident = _hf_split_ident(splits)
        names = [ident(s) for s in splits]
        chosen = next((s for s in splits if ident(s) == split), splits[0])
        rows = cli.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": hf_repo, "config": chosen["config"], "split": chosen["split"],
                "offset": max(0, offset), "length": min(limit, 100), **rev,
            },
            headers=headers,
        )
        rows.raise_for_status()
        body = rows.json()
        total = body.get("num_rows_total")
        return [r.get("row", {}) for r in body.get("rows", [])], total, ident(chosen), names


def _hf_preview_rows_multi(
    hf_repo: str, token: Optional[str], limit: int, offset: int, want: list[str],
    revision: Optional[str] = None,
) -> tuple[list[dict[str, Any]], Optional[int], Optional[str], list[str]]:
    """Like `_hf_preview_rows` but reads ACROSS several splits, concatenated in the
    dataset's split order with a COMBINED total — so the row browser pages through
    multiple subsets as one list. Each row is tagged `__split`. `want` = selected
    split idents (empty / no match → the first split). Costs one extra count probe
    per selected split, then one /rows call per split the page window spans."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    rev = {"revision": revision} if revision else {}
    with httpx.Client(timeout=20.0) as cli:
        sp = cli.get(
            "https://datasets-server.huggingface.co/splits",
            params={"dataset": hf_repo, **rev}, headers=headers,
        )
        sp.raise_for_status()
        splits = sp.json().get("splits", [])
        if not splits:
            return [], 0, None, []
        ident = _hf_split_ident(splits)
        names = [ident(s) for s in splits]
        wanted = set(want or [])
        chosen = [s for s in splits if ident(s) in wanted] or [splits[0]]

        def _count(s: dict) -> int:
            rr = cli.get(
                "https://datasets-server.huggingface.co/rows",
                params={"dataset": hf_repo, "config": s["config"], "split": s["split"],
                        "offset": 0, "length": 1, **rev}, headers=headers,
            )
            rr.raise_for_status()
            return int(rr.json().get("num_rows_total") or 0)

        totals = [_count(s) for s in chosen]
        combined = sum(totals)
        out: list[dict[str, Any]] = []
        need = limit
        start = 0  # running global start index of the current split
        for s, tot in zip(chosen, totals):
            if need <= 0:
                break
            s_start, s_end = start, start + tot
            start = s_end
            if offset >= s_end or offset + limit <= s_start:
                continue  # page window doesn't intersect this split
            local_off = max(0, offset - s_start)
            take = min(need, tot - local_off, 100)
            if take <= 0:
                continue
            rr = cli.get(
                "https://datasets-server.huggingface.co/rows",
                params={"dataset": hf_repo, "config": s["config"], "split": s["split"],
                        "offset": local_off, "length": take, **rev}, headers=headers,
            )
            rr.raise_for_status()
            for item in rr.json().get("rows", []):
                row = item.get("row", {})
                row["__split"] = ident(s)
                out.append(row)
            need -= take
        return out, combined, ",".join(ident(s) for s in chosen), names


def _hf_split_columns(hf_repo: str, token: Optional[str], revision: Optional[str] = None) -> list[dict[str, Any]]:
    """Per-split column names + row counts for an HF dataset, from the HF
    datasets-server: `/splits` for the authoritative config/split list and
    `/info` for each config's feature columns and per-split row counts. The UI
    uses this to offer a transcription/speaker column picker per split, so the
    labels MUST match `_hf_preview_rows` (train/test, …) — not the parquet
    directory names. `revision` pins the source ref. Returns [{split, columns, num_rows}]."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    rev = {"revision": revision} if revision else {}
    with httpx.Client(timeout=20.0) as cli:
        sp = cli.get(
            "https://datasets-server.huggingface.co/splits",
            params={"dataset": hf_repo, **rev}, headers=headers,
        )
        sp.raise_for_status()
        splits = sp.json().get("splits", [])
        if not splits:
            return []
        ident = _hf_split_ident(splits)
        info = cli.get(
            "https://datasets-server.huggingface.co/info",
            params={"dataset": hf_repo, **rev}, headers=headers,
        )
        info.raise_for_status()
        # dataset_info maps config → {features: {col: …}, splits: {split: {num_examples}}}.
        di = info.json().get("dataset_info", {}) or {}
    out = []
    for s in splits:
        cfg = di.get(s["config"], {}) or {}
        cols = sorted((cfg.get("features") or {}).keys())
        nrows = ((cfg.get("splits") or {}).get(s["split"]) or {}).get("num_examples")
        out.append({"split": ident(s), "columns": cols, "num_rows": nrows})
    return out


def _resolve_hf_subset(
    hf_repo: str, token: Optional[str], subset: Optional[str], revision: Optional[str] = None,
) -> tuple[str, str, str]:
    """Resolve a UI subset label (as produced by `_hf_split_ident` — the same
    label the preview/split picker shows) back to the datasets-server (config,
    split) pair, plus the canonical label. `subset` None/unknown → the first
    entry. Raises ValueError if the dataset has no splits."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    rev = {"revision": revision} if revision else {}
    with httpx.Client(timeout=30.0) as cli:
        sp = cli.get(
            "https://datasets-server.huggingface.co/splits",
            params={"dataset": hf_repo, **rev}, headers=headers,
        )
        sp.raise_for_status()
        splits = sp.json().get("splits", [])
    if not splits:
        raise ValueError(f"{hf_repo} exposes no splits on the HF datasets-server")
    ident = _hf_split_ident(splits)
    chosen = next((s for s in splits if ident(s) == subset), splits[0])
    return chosen["config"], chosen["split"], ident(chosen)


def _hf_parquet_files(
    hf_repo: str, token: Optional[str], config: str, split: str, revision: Optional[str] = None,
) -> list[str]:
    """The parquet file URLs backing one (config, split) on the HF datasets-server
    — the authoritative file list regardless of the repo's on-disk layout. Used by
    the in-process LLM pack to read the FULL split (the /rows preview caps at 100)."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    rev = {"revision": revision} if revision else {}
    with httpx.Client(timeout=60.0) as cli:
        r = cli.get(
            "https://datasets-server.huggingface.co/parquet",
            params={"dataset": hf_repo, **rev}, headers=headers,
        )
        r.raise_for_status()
        files = r.json().get("parquet_files", [])
    urls = [f["url"] for f in files if f.get("config") == config and f.get("split") == split and f.get("url")]
    return urls


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
    until: Optional[str] = None,
) -> tuple[list[dict[str, Any]], Optional[int]]:
    """Stream a Label-platform project's `export.v1.jsonl` (one task per line, with
    `audio_url` + `transcription`) and return (rows[offset:offset+limit], total).
    Auth is `Authorization: Bearer <lpat token>`. Reads `X-Total-Tasks` for the
    count so we can stop early once the requested page is collected; `limit=0`
    just verifies the token + returns the total. `until` (ISO-8601) is forwarded as
    the export's `updated_until` cutoff so the platform filters server-side (total +
    pagination stay accurate). Sync — call via `_run_sync`."""
    base = base_url.rstrip("/")
    url = f"{base}/api/projects/{project_id}/export.v1.jsonl"
    params: dict[str, str] = {"status": status or "approved"}
    if until:
        params["updated_until"] = until
    rows: list[dict[str, Any]] = []
    total: Optional[int] = None
    with httpx.Client(timeout=120.0, follow_redirects=True) as cli:
        with cli.stream(
            "GET", url, params=params,
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


# ---- packed (tts_packed) inspection: read ChiniDataset parquet + decode ----
_PACKED_CACHE: dict[str, list[dict]] = {}        # dataset_id → [{input_ids, attention_mask}]
_TTS_TOKENIZER_CACHE: dict[str, Any] = {}        # repo_id → tokenizers.Tokenizer
_DEFAULT_TTS_TOKENIZER = "Scicom-intl/Multilingual-Expressive-TTS-1.7B"


def _pack_meta(d: Dataset) -> dict:
    """The pack-metadata blob for a packed dataset — `_tts_pack` (tts_packed) or
    `_llm_pack` (llm_packed); both stash tokenizer/sequence_length/splits here."""
    sf = d.split_fields or {}
    return (sf.get("_tts_pack") or sf.get("_llm_pack") or {})


def _pack_tokenizer(d: Dataset) -> str:
    """The tokenizer repo to decode a packed dataset's blocks back to text."""
    return _pack_meta(d).get("tokenizer") or _DEFAULT_TTS_TOKENIZER


def _read_packed_parquet(target: "S3Target", s3_uri: str, cap: int = 5000) -> list[dict]:
    """Read the ChiniDataset parquet shards under the `s3_uri` prefix (multipacked
    NeuCodec records) → [{input_ids, attention_mask}], capped. attention_mask is
    the list of per-utterance lengths the packer stored, so the UI can split a
    block back into its constituent utterances."""
    import io as _io
    import pyarrow.parquet as pq

    u = urlparse(s3_uri)
    t = dataclasses.replace(target, bucket=u.netloc) if u.scheme == "s3" else target
    prefix = (u.path.lstrip("/") if u.scheme == "s3" else s3_uri).rstrip("/") + "/"
    shards = sorted(o["key"] for o in bench.s3_list(prefix, t) if o["key"].endswith(".parquet"))
    cli = bench._s3_client(t)
    out: list[dict] = []
    for key in shards:
        body = cli.get_object(Bucket=t.bucket, Key=key)["Body"].read()
        tbl = pq.read_table(_io.BytesIO(body), columns=["input_ids", "attention_mask"])
        ids_col = tbl.column("input_ids").to_pylist()
        mask_col = tbl.column("attention_mask").to_pylist()
        for ids, mask in zip(ids_col, mask_col):
            out.append({"input_ids": list(ids or []), "attention_mask": list(mask or [])})
            if len(out) >= cap:
                return out
    return out


def _decode_packed_record(repo_id: str, hf_token: Optional[str], rec: dict) -> dict:
    """Decode one packed record's token ids to text with the run's Qwen3 tokenizer
    (cached). Splits by attention_mask so each multipacked utterance shows on its
    own; speech tokens render as `<|s_N|>`, control tokens kept (not stripped)."""
    from tokenizers import Tokenizer

    tok = _TTS_TOKENIZER_CACHE.get(repo_id)
    if tok is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id, "tokenizer.json", token=hf_token or None)
        tok = Tokenizer.from_file(path)
        _TTS_TOKENIZER_CACHE[repo_id] = tok

    ids = rec.get("input_ids") or []
    mask = rec.get("attention_mask") or []
    utts, pos = [], 0
    for length in mask:
        seg = ids[pos:pos + length]
        pos += length
        utts.append({"tokens": len(seg), "text": tok.decode(seg, skip_special_tokens=False)})
    return {
        "num_tokens": len(ids),
        "num_utterances": len(utts),
        "utterances": utts,
        "full_text": tok.decode(ids, skip_special_tokens=False),
    }


async def _packed_records_cached(
    dataset_id: str, target: "S3Target", s3_uri: str, split: Optional[str] = None,
) -> list[dict]:
    # Split-aware packed datasets keep shards under <prefix>/<split>/; cache per split.
    ckey = f"{dataset_id}::{split}" if split else dataset_id
    recs = _PACKED_CACHE.get(ckey)
    if recs is None:
        uri = (s3_uri.rstrip("/") + "/" + split) if split else s3_uri
        recs = await _run_sync(_read_packed_parquet, target, uri)
        _PACKED_CACHE[ckey] = recs
    return recs


def _stamp_detected_fields(
    d: Dataset, columns: list[str], num_rows: Optional[int], fmt: Optional[str]
) -> bool:
    """Backfill format/num_rows and auto-map the audio/transcription/speaker
    columns onto a dataset row from the metadata's ACTUAL columns. Only fills a
    null and repairs a mapping that doesn't point at a real column — it never
    clobbers a valid, user-chosen field. Returns True if anything changed.

    This is why an s3-metadata import (which can't know the columns up front) and
    any older row with stale defaults self-heal: `transcription` defaulting to a
    column that doesn't exist (the file uses `text`) gets repaired, and `speaker`
    gets discovered."""
    changed = False
    if fmt and not d.format:
        d.format = fmt
        changed = True
    if num_rows is not None and d.num_rows is None:
        d.num_rows = num_rows
        changed = True
    if not columns:
        return changed
    det = dataset_metadata.detect_fields(columns)
    if (not d.audio_field or d.audio_field not in columns) and det["audio_field"]:
        if det["audio_field"] != d.audio_field:
            d.audio_field = det["audio_field"]
            changed = True
    if (not d.transcription_field or d.transcription_field not in columns) and det["transcription_field"]:
        if det["transcription_field"] != d.transcription_field:
            d.transcription_field = det["transcription_field"]
            changed = True
    if not d.speaker_field and det["speaker_field"]:
        d.speaker_field = det["speaker_field"]
        changed = True
    return changed


@router.get("/{dataset_id}/preview", response_model=PreviewResponse)
async def preview_dataset(
    dataset_id: str,
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    split: Optional[str] = Query(None, description="HF split to read (default: first)"),
    speaker: Optional[str] = Query(None, description="filter rows to one speaker (S3/upload)"),
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

    if d.kind in ("tts_packed", "llm_packed"):
        # Packed = multipacked token blocks (NeuCodec speech for tts_packed, chat
        # text for llm_packed). Show one row per block with its token/segment
        # counts; the UI decodes a block to text on expand (GET /{id}/packed-row).
        if not (d.storage_id and d.s3_metadata_uri):
            return _resp(rows=[], total=d.num_rows or 0, error="packed dataset has no storage / shards")
        # Split-aware: `_*_pack.splits` = {split: count}; tts shards live under
        # <prefix>/<split>/, llm packs to a single flat prefix. Offer a picker.
        tp = _pack_meta(d)
        pack_splits = list((tp.get("splits") or {}).keys())
        used_split = (split if (split and split in pack_splits) else (pack_splits[0] if pack_splits else None))
        try:
            storage = await _load_storage(session, d.storage_id)
            target, _base = _s3_target_and_prefix(storage)
            recs = await _packed_records_cached(dataset_id, target, d.s3_metadata_uri, used_split)
        except Exception as e:  # noqa: BLE001
            logger.warning("packed preview failed for %s: %s", dataset_id, e)
            return _resp(rows=[], total=d.num_rows or 0, error=f"could not read packed shards: {e}")
        total = len(recs)
        page = recs[offset:offset + limit]
        rows = [
            {"packed": True, "index": offset + i, "split": used_split,
             "tokens": len(r["input_ids"]), "utterances": len(r["attention_mask"])}
            for i, r in enumerate(page)
        ]
        return _resp(rows=rows, total=total, split=used_split, splits=(pack_splits or None))

    try:
        if d.kind == "label":
            tok = await _label_token(d, session)
            if not (d.label_base_url and d.label_project_id and tok):
                return _resp(rows=[], error="labeling-platform source not fully configured")
            raw, total = await _run_sync(
                _label_export_rows, d.label_base_url, d.label_project_id, tok,
                d.label_status or "approved", limit, offset,
                getattr(d, "label_updated_until", None),
            )
            base = (d.label_base_url or "").rstrip("/")

            def _label_audio(r: dict) -> Optional[str]:
                u = str(r.get("audio_url") or "")
                # Prefer our audio proxy whenever the task has an id: it streams the
                # clip via the platform's task-audio endpoint (authenticated, binary-
                # safe via the web /api/datasets/{id}/label-audio route). This both
                # hides the lpat token (proxy-mode export URLs need it) AND survives
                # presigned `audio_url`s whose S3 endpoint the browser can't resolve
                # (some platform storage configs presign a non-routable region host).
                # Only fall back to the raw URL when there's no task id to proxy by.
                if r.get("id"):
                    return f"/api/datasets/{d.id}/label-audio?task_id={r['id']}"
                return _audio_str(u)

            rows = [
                {**r, "audio_url": _label_audio(r), "transcription": r.get("transcription")}
                for r in raw
            ]
            return _resp(rows=rows, total=total)

        if d.kind in ("hf", "llm"):
            storage = await session.get(Storage, d.storage_id) if d.storage_id else None
            if not d.hf_repo:
                return _resp(rows=[], error="no hf_repo set")
            mf = getattr(d, "messages_field", None) or None  # None = not configured
            tok = await _hf_token(storage, session)
            # `split` may be a comma-separated list (the row browser's multiselect):
            # >1 → read all of them merged into one paged list with a combined total.
            sel = [s.strip() for s in (split or "").split(",") if s.strip()]
            if len(sel) > 1:
                raw, total, used_split, names = await _run_sync(
                    _hf_preview_rows_multi, d.hf_repo, tok, limit, offset, sel, d.hf_revision,
                )
            else:
                raw, total, used_split, names = await _run_sync(
                    _hf_preview_rows, d.hf_repo, tok, limit, offset, (sel[0] if sel else None),
                    d.hf_revision,
                )
            # Honour the per-split transcription mapping (e.g. test→after), per row in
            # the merged case (each row carries its own `__split`).
            sfmap = d.split_fields or {}
            # Source audio lives in a zip (no playable column). If it's been
            # materialised to S3, resolve audio by basename through the proxy.
            resolver = await _source_audio_resolver(session, d)

            def _parse_messages(v: Any) -> Any:
                """HF parquet often stores the messages array as a JSON string.
                Auto-parse to a list so the viewer always gets a real array."""
                if isinstance(v, str):
                    try:
                        parsed = json.loads(v)
                        if isinstance(parsed, list):
                            return parsed
                    except (json.JSONDecodeError, ValueError):
                        pass
                return v

            rows = []
            for r in raw:
                tcol = sfmap.get(r.get("__split") or used_split or "") or tf
                row: dict[str, Any] = {
                    "audio_url": resolver(r.get(af)) if resolver else _audio_str(r.get(af)),
                    "transcription": r.get(tcol),
                    **r,
                }
                # If messages_field is configured, parse + surface it so the chat
                # viewer always gets a list regardless of how HF stored it.
                if mf:
                    row["messages"] = _parse_messages(r.get(mf))
                rows.append(row)
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
        # Self-heal a stale record: an s3-metadata import can't know the columns up
        # front (CreateDatasetRequest has no field hints), so format/num_rows stay
        # null and the field mappings fall to defaults that may not match the file
        # (e.g. `transcription` when the column is `text`). Backfill from the actual
        # columns now — only filling nulls / repairing broken mappings — so this page
        # and the trainers see the right columns. Cheap: the file is already parsed.
        try:
            _fmt = dataset_metadata.detect_format(mdname)
        except dataset_metadata.DatasetParseError:
            _fmt = None
        _cols = list(all_rows[0].keys()) if all_rows else []
        if _stamp_detected_fields(d, _cols, len(all_rows), _fmt):
            await session.commit()
            af, tf = d.audio_field, d.transcription_field
        excluded = {int(x) for x in (d.excluded_rows or [])}
        # Pair each row with its stable GLOBAL index (file order) before any split
        # filtering — that's the identity the trainers use (they read the same
        # file in the same order), so an un-ticked row maps 1:1 to a skipped row.
        indexed = list(enumerate(all_rows))
        # A `split` column (from a split-preserving transform) → expose the splits
        # and page within the chosen one, mirroring the HF split picker.
        used_split: Optional[str] = None
        splits_list: Optional[list[str]] = None
        if all_rows and "split" in all_rows[0]:
            splits_list = sorted({str(r["split"]) for r in all_rows if r.get("split")})
            used_split = split if (split and split in splits_list) else (splits_list[0] if splits_list else None)
            if used_split is not None:
                indexed = [(gi, r) for gi, r in indexed if str(r.get("split")) == used_split]
        # Speaker filter: materialised datasets carry a speaker column. Expose the
        # distinct speakers (within the current split) for a dropdown, and page
        # within the chosen one. Computed before applying the filter so the
        # dropdown always lists every speaker, not just the selected one.
        spk_col = getattr(d, "speaker_field", None) or "speaker"
        speakers_list: Optional[list[str]] = None
        used_speaker: Optional[str] = None
        if all_rows and spk_col in all_rows[0]:
            speakers_list = sorted({str(r.get(spk_col)) for _gi, r in indexed if str(r.get(spk_col) or "").strip()})
            if speaker and speaker in speakers_list:
                used_speaker = speaker
                indexed = [(gi, r) for gi, r in indexed if str(r.get(spk_col)) == used_speaker]
        total = len(indexed)
        page = indexed[offset:offset + limit]
        rows = [
            {
                # Proxy the presigned URL through the gateway (avoids S3 CORS in
                # the browser). _audio_str drops bare filenames first.
                "audio_url": _proxy_audio_url(
                    dataset_id, _audio_str(_resolve_audio_url(target, base, d.audio_prefix, r.get(af)))
                ),
                "transcription": r.get(tf),
                "row_index": gi,
                "included": gi not in excluded,
                **r,
            }
            for gi, r in page
        ]
        return _resp(rows=rows, total=total, split=used_split, splits=splits_list,
                     speakers=speakers_list, speaker=used_speaker, excluded_count=len(excluded))
    except dataset_metadata.DatasetParseError as e:
        return _resp(rows=[], error=str(e))
    except Exception as e:  # noqa: BLE001 — surface as an inline error, not a 500
        logger.warning("preview failed for %s: %s", dataset_id, e)
        return _resp(rows=[], error=str(e))


@router.post("/{dataset_id}/row-inclusion")
async def set_row_inclusion(
    dataset_id: str,
    req: RowInclusionRequest,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Manually curate which rows train. Stores the excluded metadata-row indices
    on the dataset (default: none → all rows train); the S3/upload trainers skip
    these. Returns the new excluded count."""
    d = await _require_dataset(session, dataset_id, user)
    cur = {int(x) for x in (d.excluded_rows or [])}
    if req.clear:
        cur = set()
    else:
        idx = {int(i) for i in req.indices if int(i) >= 0}
        cur = (cur - idx) if req.included else (cur | idx)
    d.excluded_rows = sorted(cur) or None
    await session.commit()
    await audit_module.record(user, "dataset.row-inclusion", "dataset", d.id, d.name,
                              details={"excluded_count": len(cur)})
    return {"excluded_count": len(cur)}


@router.get("/{dataset_id}/packed-row")
async def packed_row(
    dataset_id: str,
    index: int = Query(..., ge=0, description="packed record index (within the split)"),
    split: Optional[str] = Query(None, description="which packed split to decode from"),
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Decode one multipacked record to text with the run's Qwen3 tokenizer — for
    inspecting what got packed together. Returns per-utterance + full decoded text.
    The datasets UI calls this when a packed row's collapse is opened."""
    d = await _require_dataset(session, dataset_id, user)
    if d.kind not in ("tts_packed", "llm_packed"):
        raise HTTPException(status_code=400, detail="not a packed (tts_packed / llm_packed) dataset")
    if not (d.storage_id and d.s3_metadata_uri):
        raise HTTPException(status_code=400, detail="packed dataset has no storage / shards")
    storage = await _load_storage(session, d.storage_id)
    target, _base = _s3_target_and_prefix(storage)
    pack_splits = list((_pack_meta(d).get("splits") or {}).keys())
    used_split = (split if (split and split in pack_splits) else (pack_splits[0] if pack_splits else None))
    recs = await _packed_records_cached(dataset_id, target, d.s3_metadata_uri, used_split)
    if index >= len(recs):
        raise HTTPException(status_code=404, detail=f"index {index} out of range (have {len(recs)})")

    repo_id = _pack_tokenizer(d)
    from .global_env_api import load_global_env
    hf_token = (await load_global_env(session)).get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    try:
        decoded = await _run_sync(_decode_packed_record, repo_id, hf_token, recs[index])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"decode failed (tokenizer {repo_id}): {e}")
    return {"index": index, "tokenizer": repo_id, **decoded}


# ---------- persistent NeuCodec decoder (play a packed utterance as audio) ------
# A `tts_packed` row is speech codes, not waveform. To hear it you load NeuCodec
# on a GPU and decode codes→WAV. Rather than a one-shot job per click, we keep the
# codec RESIDENT on a chosen VM (idle auto-unloads) so each "play utt N" is instant.
# Reuses the run try-it's persistent-server machinery (see training_api).

_DECODER_DEFAULT_VENV = "/share/autotrain-tts"


class DecoderLoadRequest(BaseModel):
    target: str = "vm"                  # "vm" (registered box) | "cloud" (spawn a RunPod pod)
    provider_id: Optional[str] = None   # vm provider (target=vm); runpod account or null=default (target=cloud)
    gpu: Optional[str] = None           # "auto" (default) | "cpu" | a GPU index — VM only
    gpu_type: Optional[str] = None      # cloud GPU type
    gpu_count: int = 1                  # cloud GPU count
    secure_cloud: bool = True           # cloud tier
    idle_timeout_s: int = 600           # auto-unload after this many idle seconds (0 = never)
    venv_path: Optional[str] = None


class DecoderDecodeRequest(BaseModel):
    provider_id: str
    index: int                     # packed record index (within the split)
    utt: int = 0                   # which utterance within that record to decode
    split: Optional[str] = None
    venv_path: Optional[str] = None


class DecoderActionRequest(BaseModel):
    provider_id: str
    venv_path: Optional[str] = None


async def _resolve_vm_ssh(session: AsyncSession, provider_id: str) -> tuple[str, int, str, str]:
    """(host, port, user, key_file) for a kind=vm provider — decrypt its stored key
    to a 600 temp file. The decoder runs on a registered VM (always-on, no spawn)."""
    from .db import Provider
    prov = await session.get(Provider, provider_id)
    if prov is None or prov.kind != "vm":
        raise HTTPException(status_code=400, detail="decoder needs a kind=vm provider")
    pc = prov.config or {}
    host, enc = pc.get("host"), pc.get("private_key_enc")
    if not (host and enc):
        raise HTTPException(status_code=400, detail="VM provider is missing host / SSH key")
    keyp = f"/tmp/sgpu_dsdec_key_{provider_id}"
    if not os.path.exists(keyp):
        with open(keyp, "w") as f:
            f.write(crypto.decrypt(enc))
        os.chmod(keyp, 0o600)
    return host, int(pc.get("port") or 22), pc.get("user") or "root", keyp


def _require_packed(d: Dataset) -> None:
    if d.kind != "tts_packed":
        raise HTTPException(status_code=400, detail="audio decode is only for tts_packed datasets")


@router.post("/{dataset_id}/decoder/load")
async def decoder_load(
    dataset_id: str,
    req: DecoderLoadRequest,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Load NeuCodec persistently on the chosen compute (idle-unloads). Poll status."""
    d = await _require_dataset(session, dataset_id, user)
    _require_packed(d)
    if (req.target or "vm").lower() != "vm":
        # A *resident* RunPod decoder = a pod kept alive (spawn + cold-install the
        # codec + idle-teardown) = the serverless-pod lifecycle. Not wired yet — a
        # registered VM (always-on, TTS venv present) is the supported persistent box.
        raise HTTPException(
            status_code=400,
            detail="the resident audio decoder runs on a registered VM for now — pick a VM under 'Run on'. "
                   "(A kept-alive RunPod pod is a heavier follow-up.)",
        )
    if not req.provider_id:
        raise HTTPException(status_code=400, detail="pick a VM provider under 'Run on'")
    host, port, suser, key = await _resolve_vm_ssh(session, req.provider_id)
    venv = (req.venv_path or "").strip() or _DECODER_DEFAULT_VENV
    from .training_api import dataset_decoder_start_ssh
    try:
        st = await _run_sync(dataset_decoder_start_ssh, host, port, suser, key, dataset_id, venv,
                             req.gpu, int(req.idle_timeout_s), {})
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"could not start decoder on the VM: {e}")
    return st


@router.get("/{dataset_id}/decoder/status")
async def decoder_status(
    dataset_id: str,
    provider_id: str = Query(...),
    venv_path: Optional[str] = Query(None),
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    d = await _require_dataset(session, dataset_id, user)
    _require_packed(d)
    host, port, suser, key = await _resolve_vm_ssh(session, provider_id)
    venv = (venv_path or "").strip() or _DECODER_DEFAULT_VENV
    from .training_api import dataset_decoder_status_ssh
    try:
        return await _run_sync(dataset_decoder_status_ssh, host, port, suser, key, dataset_id, venv)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"could not reach the VM: {e}")


@router.post("/{dataset_id}/decoder/decode")
async def decoder_decode(
    dataset_id: str,
    req: DecoderDecodeRequest,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Decode one packed utterance's speech codes → WAV on the resident decoder."""
    d = await _require_dataset(session, dataset_id, user)
    _require_packed(d)
    if not (d.storage_id and d.s3_metadata_uri):
        raise HTTPException(status_code=400, detail="packed dataset has no storage / shards")
    storage = await _load_storage(session, d.storage_id)
    target, _base = _s3_target_and_prefix(storage)
    pack_splits = list((((d.split_fields or {}).get("_tts_pack") or {}).get("splits") or {}).keys())
    used_split = (req.split if (req.split and req.split in pack_splits) else (pack_splits[0] if pack_splits else None))
    recs = await _packed_records_cached(dataset_id, target, d.s3_metadata_uri, used_split)
    if req.index >= len(recs):
        raise HTTPException(status_code=404, detail=f"index {req.index} out of range (have {len(recs)})")
    repo_id = ((d.split_fields or {}).get("_tts_pack") or {}).get("tokenizer") or _DEFAULT_TTS_TOKENIZER
    from .global_env_api import load_global_env
    hf_token = (await load_global_env(session)).get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    decoded = await _run_sync(_decode_packed_record, repo_id, hf_token, recs[req.index])
    utts = decoded.get("utterances") or []
    if req.utt >= len(utts):
        raise HTTPException(status_code=404, detail=f"utterance {req.utt} out of range (record has {len(utts)})")
    text = utts[req.utt].get("text") or ""

    host, port, suser, key = await _resolve_vm_ssh(session, req.provider_id)
    venv = (req.venv_path or "").strip() or _DECODER_DEFAULT_VENV
    from .training_api import dataset_decoder_decode_ssh
    try:
        resp = await _run_sync(dataset_decoder_decode_ssh, host, port, suser, key, dataset_id, venv, text)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"decode failed (is the decoder loaded?): {e}")
    if not resp or resp.get("error"):
        raise HTTPException(status_code=502, detail=(resp or {}).get("error") or "decoder returned no audio")
    return {"index": req.index, "utt": req.utt, "split": used_split, **resp}


@router.post("/{dataset_id}/decoder/unload")
async def decoder_unload(
    dataset_id: str,
    req: DecoderActionRequest,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    d = await _require_dataset(session, dataset_id, user)
    _require_packed(d)
    host, port, suser, key = await _resolve_vm_ssh(session, req.provider_id)
    venv = (req.venv_path or "").strip() or _DECODER_DEFAULT_VENV
    from .training_api import dataset_decoder_stop_ssh
    try:
        return await _run_sync(dataset_decoder_stop_ssh, host, port, suser, key, dataset_id, venv)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"could not stop the decoder: {e}")


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
    if d.kind not in ("hf", "llm") or not d.hf_repo:
        return SplitsResponse(splits=[])
    try:
        storage = await session.get(Storage, d.storage_id) if d.storage_id else None
        token = await _hf_token(storage, session)
        splits = await _run_sync(_hf_split_columns, d.hf_repo, token, d.hf_revision)
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
    endpoint = await _hf_endpoint(hf_store, session)

    try:
        rev = await _run_sync(_hf_upload, repo, d.metadata_filename, text.encode("utf-8"), token, req.private, endpoint)
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
        gpu_type=req.gpu_type, gpu_count=req.gpu_count,
        secure_cloud=req.secure_cloud, disk_gb=req.disk_gb, volume_gb=req.volume_gb,
        visible_devices=req.visible_devices,
        venv_path=(req.venv_path or "").strip() or "/share/neucodec-tts",
        tokenizer=req.tokenizer or None,
        pack_sequence_length=req.sequence_length,
        # The speaker column is set on the dataset's column mapping; pack_stage1
        # prepends it to each transcript. None → a single constant speaker.
        speaker_field=d.speaker_field or None,
        pack_only=True, pack_source_dataset_id=dataset_id,
        max_epochs=1,
    )
    run = await create_training_run(body, request, user, session)
    d.transform_status = "running"
    _where = req.provider_id or f"RunPod {req.gpu_type} ×{req.gpu_count}"
    d.transform_log = f"TTS pack queued (run {run.id}) · seq_len {req.sequence_length} · {_where}"
    await session.commit()
    await audit_module.record(user, "dataset.pack-tts", "dataset", dataset_id, d.name,
                              details={"run": run.id, "sequence_length": req.sequence_length})
    await session.refresh(d)
    return _to_record(d, user.username, None)


@router.post("/{dataset_id}/pack-omnivoice", response_model=DatasetRecord)
async def pack_omnivoice_dataset(
    dataset_id: str,
    req: OmnivoicePackRequest,
    request: Request,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Higgs-codec tokenize this {audio, transcription} dataset into OmniVoice
    WebDataset shards on a GPU box over SSH, then create a kind=omnivoice_packed
    dataset. Implemented as a pack-only TTS run with base_model=k2-fsa/OmniVoice
    (→ the omnivoice trainer is dispatched). Reuses Pack-for-TTS provisioning."""
    d = await _require_dataset(session, dataset_id, user)
    if d.kind == "omnivoice_packed":
        raise HTTPException(status_code=400, detail="this dataset is already OmniVoice-packed")
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
        name=f"pack-omni-{d.name}"[:128],
        dataset_id=dataset_id,
        base_model="k2-fsa/OmniVoice",  # → _tts_arch=omnivoice → omnivoice_finetune
        task_type="tts",
        provider_id=req.provider_id, storage_id=req.storage_id,
        gpu_type=req.gpu_type, gpu_count=req.gpu_count,
        # OmniVoice pins torch 2.8/cu128 → a CUDA-12.8 pod image (NOT the cu13 one).
        image=(req.image or "").strip() or "runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2404",
        secure_cloud=req.secure_cloud, disk_gb=req.disk_gb, volume_gb=req.volume_gb,
        visible_devices=req.visible_devices,
        venv_path=(req.venv_path or "").strip() or "/share/autotrain-omnivoice",
        tokenizer=req.tokenizer or "eustlb/higgs-audio-v2-tokenizer",
        speaker_field=d.speaker_field or None,
        default_language=req.default_language or "en",
        language_field=req.language_field or None,
        eval_test_per_speaker=req.eval_test_per_speaker,
        pack_only=True, pack_source_dataset_id=dataset_id,
        max_epochs=1,
    )
    run = await create_training_run(body, request, user, session)
    d.transform_status = "running"
    _where = req.provider_id or f"RunPod {req.gpu_type} ×{req.gpu_count}"
    d.transform_log = f"OmniVoice pack queued (run {run.id}) · Higgs codec · {_where}"
    await session.commit()
    await audit_module.record(user, "dataset.pack-omnivoice", "dataset", dataset_id, d.name,
                              details={"run": run.id})
    await session.refresh(d)
    return _to_record(d, user.username, None)


@router.post("/{dataset_id}/pack-llm", response_model=DatasetRecord)
async def pack_llm_dataset(
    dataset_id: str,
    req: LlmPackRequest,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Tokenize + multipack this chat (kind=llm) dataset's messages column into a
    ChiniDataset on S3, then create a new kind=llm_packed dataset. Runs IN-PROCESS
    (CPU tokenization — no GPU box). The source dataset's transform_status /
    transform_log track progress (poll GET /{id})."""
    d = await _require_dataset(session, dataset_id, user)
    # A kind=llm dataset, OR any HF-backed dataset with a messages column mapped.
    if not (d.kind == "llm" or (d.kind in ("hf",) and getattr(d, "messages_field", None))):
        raise HTTPException(
            status_code=400,
            detail="LLM pack needs a kind=llm dataset (or an hf dataset with a messages column set)",
        )
    if not (d.hf_repo and "/" in d.hf_repo):
        raise HTTPException(status_code=400, detail="LLM pack needs a source HuggingFace repo (owner/name)")
    if d.transform_status == "running":
        raise HTTPException(status_code=409, detail="a transform is already running for this dataset")
    if not (req.tokenizer or "").strip():
        raise HTTPException(status_code=400, detail="a tokenizer (chat template) is required")
    if req.sequence_length < 1:
        raise HTTPException(status_code=400, detail="sequence_length must be >= 1")
    st = await session.get(Storage, req.storage_id)
    if st is None or st.kind != "s3":
        raise HTTPException(status_code=400, detail="storage_id must reference a kind=s3 storage")

    # Accept a multiselect (`subsets`); fall back to the single `subset`. Each label
    # is resolved + packed, with all selected subsets concatenated into one dataset.
    raw = req.subsets if req.subsets else ([req.subset] if req.subset else [])
    subsets = [s.strip() for s in raw if s and s.strip()]

    from . import dataset_transform
    await dataset_transform.start_llm_pack(
        dataset_id,
        subsets=subsets or None,
        tokenizer=req.tokenizer.strip(),
        sequence_length=int(req.sequence_length),
        storage_id=req.storage_id,
        tools_field=(req.tools_field or "").strip() or None,
        all_reasoning=bool(req.all_reasoning),
    )
    await audit_module.record(user, "dataset.pack-llm", "dataset", dataset_id, d.name,
                              details={"tokenizer": req.tokenizer, "subsets": subsets,
                                       "sequence_length": req.sequence_length})
    await session.refresh(d)
    return _to_record(d, user.username, None)


@router.post("/merge", response_model=DatasetRecord)
async def merge_datasets(
    req: DatasetMergeRequest,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Concatenate >=2 kind=label datasets into one new audio dataset (HF or S3).
    Creates the merged dataset row up front (transform_status=running) and runs
    the export+download+write in-process — poll GET /{new_id}."""
    ids: list[str] = []
    for sid in (req.source_ids or []):
        sid = (sid or "").strip()
        if sid and sid not in ids:   # de-dup, preserve order
            ids.append(sid)
    if len(ids) < 2:
        raise HTTPException(status_code=400, detail="merge needs at least 2 distinct source dataset ids")

    sources: list[Dataset] = []
    for sid in ids:
        d = await _require_dataset(session, sid, user)  # 404 / 403 as needed
        if d.kind != "label":
            raise HTTPException(
                status_code=400,
                detail=f"source {sid} is kind={d.kind}; merge currently supports kind=label datasets only",
            )
        sources.append(d)

    if req.target not in ("hf", "s3"):
        raise HTTPException(status_code=400, detail="target must be 'hf' or 's3'")
    if req.target == "hf" and not (req.hf_repo and "/" in req.hf_repo):
        raise HTTPException(status_code=400, detail="target HF repo must be owner/name")
    if req.target == "s3":
        st = await session.get(Storage, req.storage_id) if req.storage_id else None
        if st is None or st.kind != "s3":
            raise HTTPException(status_code=400, detail="storage_id must reference a kind=s3 storage")

    # Create the merged OUTPUT dataset row up front so the UI has an id to poll.
    transcription_field = next((s.transcription_field for s in sources if s.transcription_field), None) or "transcription"
    name = (req.name or "").strip() or f"Merged: {' + '.join(s.name for s in sources)}"
    new = Dataset(
        id=_gen_id(),
        owner_id=user.id,
        name=name[:255],
        description=f"Merge of {len(sources)} label datasets ({', '.join(s.id for s in sources)})",
        kind=("hf" if req.target == "hf" else "s3"),
        storage_id=(req.storage_id if req.target == "s3" else None),
        hf_repo=((req.hf_repo or "").strip() or None) if req.target == "hf" else None,
        audio_field="audio",
        transcription_field=transcription_field,
        num_rows=0,
        transform_status="running",
    )
    session.add(new)
    await session.commit()

    from . import dataset_transform
    await dataset_transform.start_merge(
        new.id,
        source_ids=ids,
        target=req.target,
        hf_repo=(req.hf_repo or "").strip() or None,
        storage_id=req.storage_id,
        s3_folder=(req.s3_folder or "").strip() or None,
    )
    await audit_module.record(user, "dataset.merge", "dataset", new.id, name,
                              details={"sources": ids, "target": req.target})
    await session.refresh(new)
    return _to_record(new, user.username, None)


@router.post("/{dataset_id}/cancel-transform", response_model=DatasetRecord)
async def cancel_dataset_transform(
    dataset_id: str,
    user: User = Depends(require_section("datasets")),
    session: AsyncSession = Depends(get_session),
):
    """Cancel a running transform for this dataset — whether it's an in-process
    audio-extraction job (hf/label → audio) or a TTS pack (a training run)."""
    d = await _require_dataset(session, dataset_id, user)
    if d.transform_status != "running":
        owner = await session.get(User, d.owner_id)
        return _to_record(d, owner.username if owner else "", None)

    from . import dataset_transform
    from .training_api import cancel_pack_run_for_dataset

    cancelled = await dataset_transform.cancel_transform(dataset_id)
    cancelled = (await cancel_pack_run_for_dataset(dataset_id)) or cancelled
    # Stamp cancelled if nothing else already moved it off "running".
    await session.refresh(d)
    if d.transform_status == "running":
        d.transform_status = "cancelled"
        prev = d.transform_log or ""
        d.transform_log = (prev + ("\n" if prev else "") + "transform cancelled by user")[-8000:]
        await session.commit()
        await session.refresh(d)
    await audit_module.record(user, "dataset.cancel-transform", "dataset", dataset_id, d.name)
    owner = await session.get(User, d.owner_id)
    return _to_record(d, owner.username if owner else "", None)


def _hf_upload(repo: str, path_in_repo: str, body: bytes, token: str, private: bool,
               endpoint: Optional[str] = None) -> Optional[str]:
    from huggingface_hub import HfApi
    api = HfApi(token=token, endpoint=endpoint or None)
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
