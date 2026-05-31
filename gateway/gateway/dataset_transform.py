"""Dataset transform — turn a HuggingFace dataset that stores audio as zip/tar
archives + a metadata table into a dataset with a real `audio` column.

Runs as a gateway background task (asyncio.create_task) with milestone progress
written to Dataset.transform_log so the UI can poll it. The heavy/blocking steps
(download, unzip, build, push/upload) run in a threadpool. Output goes to a HF
repo (datasets.Dataset with an Audio() feature) or is materialised to S3 (audio
files + a metadata file whose audio column holds presigned URLs).

Deps (`datasets`, `soundfile`, `huggingface_hub`) are imported lazily so the
gateway still boots if they're missing — the job just fails with a clear error.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .db import Dataset, Storage, session_factory

logger = logging.getLogger("gateway.dataset_transform")

_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac")
_META_NAMES = ("metadata", "train", "data")
_META_EXTS = (".csv", ".tsv", ".parquet", ".jsonl", ".json")


def _ts() -> str:
    return f"[{datetime.now(timezone.utc):%H:%M:%S}]"


def _append_log(existing: Optional[str], line: str, max_chars: int = 8000) -> str:
    text = existing or ""
    text = (text + ("\n" if text else "") + f"{_ts()} {line}")
    return text[-max_chars:]


# ---------------- async orchestration ----------------


async def start_transform(
    dataset_id: str,
    target: str,
    hf_repo: Optional[str],
    storage_id: Optional[str],
    s3_folder: Optional[str] = None,
) -> None:
    """Mark the dataset running and kick off the background job."""
    async with session_factory()() as s:
        d = await s.get(Dataset, dataset_id)
        if d is None:
            return
        d.transform_status = "running"
        d.transform_log = _append_log(None, f"transform queued (target={target})")
        await s.commit()
    asyncio.create_task(_run(dataset_id, target, hf_repo, storage_id, s3_folder))


async def _log(dataset_id: str, line: str) -> None:
    async with session_factory()() as s:
        d = await s.get(Dataset, dataset_id)
        if d is not None:
            d.transform_log = _append_log(d.transform_log, line)
            await s.commit()
    logger.info("transform %s: %s", dataset_id, line)


async def _finish(dataset_id: str, status: str, line: str, **updates) -> None:
    async with session_factory()() as s:
        d = await s.get(Dataset, dataset_id)
        if d is not None:
            d.transform_status = status
            d.transform_log = _append_log(d.transform_log, line)
            for k, v in updates.items():
                setattr(d, k, v)
            await s.commit()
    logger.info("transform %s: %s (%s)", dataset_id, line, status)


async def _create_output(
    source_id: str,
    *,
    kind: str,
    transcription_field: str,
    num_rows: int,
    hf_repo: Optional[str] = None,
    storage_id: Optional[str] = None,
    s3_metadata_uri: Optional[str] = None,
) -> str:
    """Create the transformed dataset as a NEW row (audio_field='audio'),
    leaving the source intact + re-runnable. Returns the new dataset id."""
    import secrets

    async with session_factory()() as s:
        src = await s.get(Dataset, source_id)
        if src is None:
            return ""
        new = Dataset(
            id=f"ds-{secrets.token_hex(4)}",
            owner_id=src.owner_id,
            name=f"{src.name}-audio",
            description=f"Audio-column transform of {src.hf_repo or src.name}",
            kind=kind,
            storage_id=storage_id if kind == "s3" else src.storage_id,  # hf keeps the HF storage for token resolution
            s3_metadata_uri=s3_metadata_uri,
            hf_repo=hf_repo if kind == "hf" else None,
            audio_field="audio",
            transcription_field=transcription_field,
            num_rows=num_rows,
        )
        s.add(new)
        await s.commit()
        return new.id


async def _run(
    dataset_id: str,
    target: str,
    hf_repo: Optional[str],
    storage_id: Optional[str],
    s3_folder: Optional[str] = None,
) -> None:
    from fastapi.concurrency import run_in_threadpool
    from sqlalchemy import select
    from .datasets_api import _hf_token, _s3_target_and_prefix

    work: Optional[str] = None
    try:
        # Resolve source repo + columns + tokens + output storage.
        async with session_factory()() as s:
            d = await s.get(Dataset, dataset_id)
            if d is None:
                return
            src_repo = d.hf_repo
            audio_field, transcription_field = d.audio_field, d.transcription_field
            split_fields = dict(d.split_fields or {})  # {split: transcription_column}
            hf_store = (
                await s.execute(select(Storage).where(Storage.kind == "huggingface").limit(1))
            ).scalars().first()
            token = await _hf_token(hf_store, s)
            out_storage = await s.get(Storage, storage_id) if (target == "s3" and storage_id) else None
            s3_target, s3_prefix = (_s3_target_and_prefix(out_storage) if out_storage else (None, ""))

        if not src_repo:
            await _finish(dataset_id, "failed", "dataset has no hf_repo to transform")
            return
        if target == "hf" and not (hf_repo and "/" in hf_repo):
            await _finish(dataset_id, "failed", "target HF repo must be owner/name")
            return
        if target == "s3" and out_storage is None:
            await _finish(dataset_id, "failed", "pick an S3 storage for the S3 target")
            return

        await _log(dataset_id, f"downloading {src_repo} …")
        work = await run_in_threadpool(_download, src_repo, token)

        await _log(dataset_id, "extracting audio archives …")
        n_audio = await run_in_threadpool(_extract_archives, work)
        await _log(dataset_id, f"~{n_audio} audio files on disk; reading metadata …")

        if split_fields:
            await _log(dataset_id, f"per-split transcription columns: {split_fields}")
        pairs = await run_in_threadpool(_build_pairs, work, audio_field, transcription_field, split_fields)
        if not pairs:
            await _finish(dataset_id, "failed", "no metadata rows matched an extracted audio file — check the audio column")
            return
        await _log(dataset_id, f"matched {len(pairs)} rows to audio; building output …")

        # Non-destructive: build the output, then create a NEW "-audio" dataset
        # pointing at it. The source dataset is left intact + re-runnable.
        if target == "hf":
            await run_in_threadpool(_push_hf, pairs, transcription_field, hf_repo, token)
            new_id = await _create_output(
                dataset_id, kind="hf", transcription_field=transcription_field, num_rows=len(pairs),
                hf_repo=hf_repo,
            )
            await _finish(dataset_id, "done", f"pushed → {hf_repo}; created dataset {new_id} ({len(pairs)} rows)")
        else:
            uri = await run_in_threadpool(
                _materialise_s3, pairs, transcription_field, s3_target, s3_prefix, dataset_id, s3_folder,
            )
            new_id = await _create_output(
                dataset_id, kind="s3", transcription_field=transcription_field, num_rows=len(pairs),
                storage_id=storage_id, s3_metadata_uri=uri,
            )
            # Link the source → its materialised audio so the source page can
            # resolve + play audio by basename through the gateway proxy.
            await _finish(
                dataset_id, "done",
                f"materialised → {uri}; created dataset {new_id} ({len(pairs)} rows)",
                audio_dataset_id=new_id,
            )
    except ModuleNotFoundError as e:
        await _finish(dataset_id, "failed", f"missing dependency: {e}. Install datasets + soundfile in the gateway.")
    except Exception as e:  # noqa: BLE001
        await _finish(dataset_id, "failed", f"transform failed: {e}")
        logger.exception("transform %s failed", dataset_id)
    finally:
        if work:
            shutil.rmtree(work, ignore_errors=True)


# ---------------- blocking ETL steps (run in threadpool) ----------------


def _download(repo: str, token: Optional[str]) -> str:
    """snapshot_download the dataset repo into a fresh temp dir; return its path."""
    import tempfile
    from huggingface_hub import snapshot_download

    work = tempfile.mkdtemp(prefix="sgpu-transform-")
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=work, token=token)
    return work


def _extract_archives(work: str) -> int:
    """Unzip/untar every archive under `work` in place. Returns a rough count of
    audio files present afterwards."""
    root = Path(work)
    for p in list(root.rglob("*")):
        if not p.is_file():
            continue
        try:
            if p.suffix.lower() == ".zip" and zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as zf:
                    zf.extractall(p.parent)
            elif tarfile.is_tarfile(p):
                with tarfile.open(p) as tf:
                    tf.extractall(p.parent)  # noqa: S202 — trusted org dataset
        except Exception:
            logger.exception("extract failed for %s", p)
    return sum(1 for f in root.rglob("*") if f.suffix.lower() in _AUDIO_EXTS)


def _load_table(meta: Path):
    import pandas as pd  # bundled with `datasets`
    ext = meta.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(meta)
    if ext in (".jsonl", ".json"):
        return pd.read_json(meta, lines=(ext == ".jsonl"))
    return pd.read_csv(meta, sep="\t" if ext == ".tsv" else ",")


def _split_of(meta: Path, root: Path) -> str:
    """Infer a metadata table's split from its location. HF parquet layout is
    `<split>/<shard>.parquet`, so the first path component is the split; a bare
    file at the root falls back to its stem (e.g. `metadata`)."""
    try:
        rel = meta.relative_to(root)
    except ValueError:
        return meta.stem
    return rel.parts[0] if len(rel.parts) > 1 else meta.stem


def _read_metadata(
    work: str,
    audio_field: str,
    transcription_field: str,
    split_fields: Optional[dict] = None,
) -> list[dict]:
    """Load + concatenate every metadata table (csv/tsv/parquet/jsonl) that has
    the audio column — so train/test split parquets are all included. Each split
    can name its transcription column differently (`split_fields`, e.g.
    {"train": "text", "test": "after"}); the chosen column is normalised into a
    single `__text__` column so the output is unified. Raises with the columns it
    *did* find when nothing matches (the usual cause: wrong column mapping)."""
    import pandas as pd

    split_fields = split_fields or {}
    root = Path(work)
    candidates: list[Path] = []
    for ext in _META_EXTS:
        candidates += sorted(root.rglob(f"*{ext}"))
    if not candidates:
        raise RuntimeError("no metadata table (csv/parquet/jsonl) found in the repo")

    frames = []
    seen_cols: set[str] = set()
    for meta in candidates:
        try:
            df = _load_table(meta)
        except Exception:
            continue
        seen_cols.update(map(str, df.columns))
        if audio_field not in df.columns:
            continue
        split = _split_of(meta, root)
        tcol = split_fields.get(split) or transcription_field
        out = pd.DataFrame({audio_field: df[audio_field]})
        # Missing column for this split → blank, not a hard failure (the split may
        # genuinely lack a transcription, or the mapping only covers some splits).
        out["__text__"] = df[tcol] if tcol in df.columns else ""
        out["__split__"] = split  # carried through so the output keeps its splits
        frames.append(out)
    if not frames:
        raise RuntimeError(
            f"no metadata table has the audio column '{audio_field}'. "
            f"Columns found: {sorted(seen_cols)}. Set the right Audio/Transcription columns and retry."
        )
    return pd.concat(frames, ignore_index=True).to_dict(orient="records")


def _build_pairs(
    work: str,
    audio_field: str,
    transcription_field: str,
    split_fields: Optional[dict] = None,
) -> list[tuple[str, str, str]]:
    """Join each metadata row's audio reference to an extracted file on disk.
    Returns [(split, abs_audio_path, transcription)] so the output keeps splits."""
    root = Path(work)
    by_name: dict[str, str] = {}
    for f in root.rglob("*"):
        if f.suffix.lower() in _AUDIO_EXTS:
            by_name.setdefault(f.name, str(f))  # basename → path
    rows = _read_metadata(work, audio_field, transcription_field, split_fields)
    pairs: list[tuple[str, str, str]] = []
    for r in rows:
        ref = r.get(audio_field)
        if not isinstance(ref, str) or not ref:
            continue
        cand = root / ref
        path = str(cand) if cand.is_file() else by_name.get(Path(ref).name)
        if not path:
            continue
        text = r.get("__text__")
        # `__text__` is the per-split transcription; pandas yields NaN when a row
        # came from a split missing the chosen column.
        if text is None or (isinstance(text, float) and text != text):
            text = ""
        split = r.get("__split__") or "train"
        pairs.append((str(split), path, str(text)))
    return pairs


def _push_hf(pairs: list[tuple[str, str, str]], transcription_field: str, out_repo: str, token: Optional[str]) -> None:
    from datasets import Audio, Dataset as HFDataset, DatasetDict

    by_split: dict[str, dict[str, list]] = {}
    for split, path, text in pairs:
        d = by_split.setdefault(split, {"audio": [], transcription_field: []})
        d["audio"].append(path)
        d[transcription_field].append(text)
    dd = DatasetDict({
        split: HFDataset.from_dict(d).cast_column("audio", Audio())
        for split, d in by_split.items()
    })
    dd.push_to_hub(out_repo, token=token, private=True)


def _materialise_s3(
    pairs: list[tuple[str, str, str]],
    transcription_field: str,
    s3_target,
    s3_prefix: str,
    dataset_id: str,
    s3_folder: Optional[str] = None,
) -> str:
    """Upload each audio file to S3 and write a metadata CSV with columns
    [audio (presigned URL), <transcription>, split] — keeping the source's
    splits. Writes under the storage's prefix + `s3_folder` (default
    datasets/{id}/transformed). Re-runs skip audio already in the bucket.
    Returns the s3:// URI of the metadata."""
    import csv
    import io as _io

    from . import bench

    folder = (s3_folder or f"datasets/{dataset_id}/transformed").strip().strip("/")
    base = "/".join(p for p in [(s3_prefix or "").strip().strip("/"), folder] if p)
    # Existing audio keys → skip re-upload (idempotent, fast re-runs).
    existing: set[str] = set()
    try:
        for obj in bench.s3_list(f"{base}/audio/", s3_target):
            if obj.get("key"):
                existing.add(obj["key"])
    except Exception:  # noqa: BLE001 — best-effort; fall back to always uploading
        pass
    buf = _io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["audio", transcription_field, "split"])
    uploaded: dict[str, str] = {}  # src path → presigned URL (dedupe repeats across splits)
    for split, path, text in pairs:
        url = uploaded.get(path)
        if url is None:
            key = f"{base}/audio/{os.path.basename(path)}"
            if key not in existing:
                bench.s3_put_file(key, path, s3_target)
            url = bench.s3_presign_get(key, 7 * 24 * 3600, s3_target)
            uploaded[path] = url
        writer.writerow([url, text, split])
    meta_key = f"{base}/metadata.csv"
    bench.s3_put_text(meta_key, buf.getvalue(), s3_target)
    bucket = getattr(s3_target, "bucket", "")
    return f"s3://{bucket}/{meta_key}"
