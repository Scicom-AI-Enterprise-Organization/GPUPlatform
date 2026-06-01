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
from typing import Callable, Optional

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


# In-flight transform tasks, keyed by dataset_id, so the datasets "Cancel"
# button can abort an audio-extraction job (hf/label → audio) mid-run.
_active: dict[str, asyncio.Task] = {}


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
    task = asyncio.create_task(_run(dataset_id, target, hf_repo, storage_id, s3_folder))
    _active[dataset_id] = task
    task.add_done_callback(lambda _t: _active.pop(dataset_id, None))


async def cancel_transform(dataset_id: str) -> bool:
    """Abort an in-flight audio-extraction transform for this dataset. Returns
    True if one was running. (Pack-only TTS runs are training runs — cancelled
    separately via training_api.cancel_pack_run_for_dataset.)"""
    t = _active.get(dataset_id)
    if t is None or t.done():
        return False
    t.cancel()
    await _finish(dataset_id, "cancelled", "transform cancelled by user")
    return True


async def _log(dataset_id: str, line: str) -> None:
    async with session_factory()() as s:
        d = await s.get(Dataset, dataset_id)
        if d is not None:
            d.transform_log = _append_log(d.transform_log, line)
            await s.commit()
    logger.info("transform %s: %s", dataset_id, line)


def _make_progress(dataset_id: str, loop: asyncio.AbstractEventLoop) -> Callable[[str, int, int], None]:
    """A thread-safe progress callback for the blocking ETL steps (which run in a
    threadpool). Appends an `[AUTOTRAIN_PROGRESS]` marker to transform_log so the
    UI can show a percentage + ETA — the same marker format the TTS pack emits."""
    def progress(step: str, processed: int, total: int) -> None:
        pct = (processed / total * 100.0) if total else 0.0
        line = f"[AUTOTRAIN_PROGRESS] step={step} processed={processed} total={total} percent={pct:.1f}"
        try:
            fut = asyncio.run_coroutine_threadsafe(_log(dataset_id, line), loop)
            fut.add_done_callback(lambda f: f.cancelled() or f.exception())  # retrieve → no "never awaited" warning
        except Exception:  # noqa: BLE001 — progress is best-effort
            pass
    return progress


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
    from .datasets_api import _hf_token, _label_token, _s3_target_and_prefix

    # Stable per-dataset scratch dir: a re-run reuses an earlier download +
    # extraction (no re-fetch of a huge repo over a slow link). Kept on
    # failure/cancel for resume; only removed after a fully successful run.
    work = _work_dir(dataset_id)
    success = False
    progress = _make_progress(dataset_id, asyncio.get_running_loop())
    try:
        os.makedirs(work, exist_ok=True)
        # Resolve source repo + columns + tokens + output storage.
        async with session_factory()() as s:
            d = await s.get(Dataset, dataset_id)
            if d is None:
                return
            kind = d.kind
            src_repo = d.hf_repo
            audio_field, transcription_field = d.audio_field, d.transcription_field
            split_fields = dict(d.split_fields or {})  # {split: transcription_column}
            hf_store = (
                await s.execute(select(Storage).where(Storage.kind == "huggingface").limit(1))
            ).scalars().first()
            token = await _hf_token(hf_store, s)
            out_storage = await s.get(Storage, storage_id) if (target == "s3" and storage_id) else None
            s3_target, s3_prefix = (_s3_target_and_prefix(out_storage) if out_storage else (None, ""))
            # Labeling-platform source (kind="label"): base URL + project + token.
            label_base_url = d.label_base_url
            label_project_id = d.label_project_id
            label_status = d.label_status or "approved"
            label_token = await _label_token(d, s) if kind == "label" else None

        if kind == "label":
            if not (label_base_url and label_project_id and label_token):
                await _finish(dataset_id, "failed", "labeling-platform source not fully configured (base URL, project, token)")
                return
        elif not src_repo:
            await _finish(dataset_id, "failed", "dataset has no hf_repo to transform")
            return
        if target == "hf" and not (hf_repo and "/" in hf_repo):
            await _finish(dataset_id, "failed", "target HF repo must be owner/name")
            return
        if target == "s3" and out_storage is None:
            await _finish(dataset_id, "failed", "pick an S3 storage for the S3 target")
            return

        if kind == "label":
            # Stream the project export + download each task's audio. No archive
            # extraction — the export already pairs audio with its transcription.
            await _log(dataset_id, f"exporting label project {label_project_id} (status={label_status}) …")
            pairs = await run_in_threadpool(
                _label_pairs, label_base_url, label_project_id, label_token, label_status, work,
            )
            if not pairs:
                await _finish(dataset_id, "failed", "no tasks with downloadable audio in the label export (check the status filter / token)")
                return
        else:
            await _log(dataset_id, f"downloading {src_repo} … (reuses cached files on re-run)")
            await run_in_threadpool(_download, src_repo, token, work, progress)

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
            success = True
        else:
            uri = await run_in_threadpool(
                _materialise_s3, pairs, transcription_field, s3_target, s3_prefix, dataset_id, s3_folder, progress,
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
            success = True
    except ModuleNotFoundError as e:
        await _finish(dataset_id, "failed", f"missing dependency: {e}. Install datasets + soundfile in the gateway.")
    except Exception as e:  # noqa: BLE001
        await _finish(dataset_id, "failed", f"transform failed: {e}")
        logger.exception("transform %s failed", dataset_id)
    finally:
        # Keep the scratch dir on failure/cancel so a re-run resumes from the
        # already-downloaded + extracted files; only clean up after success.
        if work and success:
            shutil.rmtree(work, ignore_errors=True)


# ---------------- blocking ETL steps (run in threadpool) ----------------


def _work_dir(dataset_id: str) -> str:
    """Stable per-dataset scratch dir under the system temp root. Reused across
    runs so a re-run doesn't re-download/re-extract a big repo."""
    import tempfile
    return os.path.join(tempfile.gettempdir(), "sgpu-transform", dataset_id)


def _download(
    repo: str,
    token: Optional[str],
    dest: str,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> str:
    """snapshot_download the dataset repo into the STABLE `dest` — re-runs skip
    files already present (no network re-fetch). Emits a byte-based `download`
    progress marker (snapshot_download itself is opaque, so the UI otherwise
    shows no %) by polling the on-disk size against the repo's total size."""
    import threading
    from huggingface_hub import HfApi, snapshot_download

    os.makedirs(dest, exist_ok=True)
    total_bytes = 0
    try:
        info = HfApi(token=token).repo_info(repo, repo_type="dataset", files_metadata=True)
        total_bytes = sum(int(getattr(s, "size", 0) or 0) for s in (info.siblings or []))
    except Exception:  # noqa: BLE001 — size lookup is best-effort (just disables %)
        logger.warning("repo_info size lookup failed for %s; download %% unavailable", repo)

    mb = 1024 * 1024

    def _dir_bytes() -> int:
        tot = 0
        for root, _dirs, files in os.walk(dest):
            for f in files:
                try:
                    tot += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return tot

    stop = threading.Event()

    def _poll() -> None:
        while not stop.wait(2.0):
            if progress and total_bytes:
                progress("download", min(_dir_bytes(), total_bytes) // mb, total_bytes // mb)

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest, token=token)
    finally:
        stop.set()
        poller.join(timeout=3.0)
    if progress and total_bytes:
        progress("download", total_bytes // mb, total_bytes // mb)
    return dest


# Content-type → extension, for label-platform audio that arrives without a
# usable filename (e.g. proxied task-audio endpoints).
_CT_EXT = {
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
    "audio/flac": ".flac", "audio/x-flac": ".flac",
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
    "audio/ogg": ".ogg", "audio/opus": ".opus",
    "audio/mp4": ".m4a", "audio/aac": ".aac", "audio/x-m4a": ".m4a",
}


def _label_pairs(
    base_url: str,
    project_id: str,
    token: str,
    status: str,
    work: str,
) -> list[tuple[str, str, str, dict]]:
    """Stream a labeling-platform project export (export.v1.jsonl) and download
    each task's audio into `work/audio`. Returns [(split, abs_audio_path,
    transcription, extra)] — the same shape `_build_pairs` produces for HF
    sources (extra is empty here), so the HF/S3 target writers are reused.

    Audio resolution mirrors the gateway's label-audio proxy: tasks whose
    `audio_url` lives under the platform (or that only expose a task id) are
    fetched from `/api/projects/{pid}/tasks/{tid}/audio` with the bearer token;
    presigned-S3 `audio_url`s are downloaded directly (no auth header — the
    signature lives in the query string)."""
    import os as _os
    from urllib.parse import unquote, urlparse

    import httpx

    from .datasets_api import _label_export_rows

    base = (base_url or "").rstrip("/")
    # limit=huge → read every line; offset=0. _label_export_rows stops only when
    # it has `total` rows, so a limit past the end safely yields all of them.
    rows, _total = _label_export_rows(base_url, project_id, token, status, 10**9, 0)

    audio_dir = Path(work) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, str, str, dict]] = []
    with httpx.Client(timeout=120.0, follow_redirects=True) as cli:
        for i, r in enumerate(rows):
            u = str(r.get("audio_url") or "").strip()
            tid = r.get("id")
            # Decide download URL + whether to attach the platform token.
            if (not u or u.startswith(base)) and tid is not None:
                dl_url = f"{base}/api/projects/{project_id}/tasks/{tid}/audio"
                headers = {"Authorization": f"Bearer {token}"}
            elif u:
                dl_url, headers = u, {}
            else:
                continue
            try:
                resp = cli.get(dl_url, headers=headers)
                resp.raise_for_status()
            except Exception:  # noqa: BLE001 — skip the task, keep going
                logger.warning("label audio download failed (task=%s)", tid)
                continue
            # Pick an extension: URL basename, else the response content-type.
            ext = _os.path.splitext(unquote(_os.path.basename(urlparse(u).path)))[1].lower()
            if ext not in _AUDIO_EXTS:
                ext = _CT_EXT.get((resp.headers.get("content-type") or "").split(";")[0].strip().lower(), ".wav")
            name = f"task-{tid}{ext}" if tid is not None else f"row-{i}{ext}"
            dest = audio_dir / name
            dest.write_bytes(resp.content)
            text = r.get("transcription")
            text = "" if text is None else str(text)
            split = str(r.get("split") or "train")
            pairs.append((split, str(dest), text, {}))
    return pairs


def _extract_archives(work: str) -> int:
    """Unzip/untar every archive under `work` in place. Idempotent: an archive
    already expanded on a previous run (a sibling `.extracted` marker exists) is
    skipped, so a resumed transform doesn't re-extract. Returns a rough count of
    audio files present afterwards."""
    root = Path(work)
    for p in list(root.rglob("*")):
        if not p.is_file() or p.suffix == ".extracted":
            continue
        marker = p.with_name(p.name + ".extracted")
        if marker.exists():
            continue
        try:
            if p.suffix.lower() == ".zip" and zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as zf:
                    zf.extractall(p.parent)
                marker.write_text("")
            elif tarfile.is_tarfile(p):
                with tarfile.open(p) as tf:
                    tf.extractall(p.parent)  # noqa: S202 — trusted org dataset
                marker.write_text("")
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


def _scalar(v) -> Optional[str]:
    """Stringify a metadata value if it's a simple scalar (str/number/bool);
    return None for nulls/NaN or complex values (audio structs, byte blobs,
    arrays) that can't live in a metadata CSV — so carried-through columns like
    `speaker` survive but the audio feature itself is skipped."""
    import numpy as _np

    if v is None or isinstance(v, (dict, list, tuple, set, bytes, bytearray, _np.ndarray)):
        return None
    try:
        import pandas as _pd
        if _pd.isna(v):  # NaN / NaT / pd.NA (scalars only — complex types excluded above)
            return None
    except (TypeError, ValueError):
        pass
    s = str(v)
    return s if s.strip() != "" else None


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
    single `__text__` column so the output is unified. All other columns (e.g.
    `speaker`) are preserved so the output dataset keeps them. Raises with the
    columns it *did* find when nothing matches (usually a wrong column mapping)."""
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
        out = df.copy()  # keep every column (speaker, etc.); audio col is replaced later
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
) -> list[tuple[str, str, str, dict]]:
    """Join each metadata row's audio reference to an extracted file on disk.
    Returns [(split, abs_audio_path, transcription, extra)] so the output keeps
    splits; `extra` carries the row's other simple columns (e.g. speaker)."""
    root = Path(work)
    by_name: dict[str, str] = {}
    for f in root.rglob("*"):
        if f.suffix.lower() in _AUDIO_EXTS:
            by_name.setdefault(f.name, str(f))  # basename → path
    rows = _read_metadata(work, audio_field, transcription_field, split_fields)
    # Don't carry the audio col (it's replaced by the URL), the chosen
    # transcription col (it becomes `transcription_field`), or the markers.
    skip = {audio_field, transcription_field, "__text__", "__split__"}
    pairs: list[tuple[str, str, str, dict]] = []
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
        extra = {str(k): s for k, v in r.items() if k not in skip and (s := _scalar(v)) is not None}
        pairs.append((str(split), path, str(text), extra))
    return pairs


def _push_hf(pairs: list[tuple[str, str, str, dict]], transcription_field: str, out_repo: str, token: Optional[str]) -> None:
    from datasets import Audio, Dataset as HFDataset, DatasetDict

    # Union of carried-through columns (e.g. speaker) so every row has every key.
    extra_cols = sorted({k for *_rest, extra in pairs for k in extra})
    by_split: dict[str, dict[str, list]] = {}
    for split, path, text, extra in pairs:
        d = by_split.setdefault(
            split, {"audio": [], transcription_field: [], **{c: [] for c in extra_cols}}
        )
        d["audio"].append(path)
        d[transcription_field].append(text)
        for c in extra_cols:
            d[c].append(extra.get(c, ""))
    dd = DatasetDict({
        split: HFDataset.from_dict(d).cast_column("audio", Audio())
        for split, d in by_split.items()
    })
    dd.push_to_hub(out_repo, token=token, private=True)


def _materialise_s3(
    pairs: list[tuple[str, str, str, dict]],
    transcription_field: str,
    s3_target,
    s3_prefix: str,
    dataset_id: str,
    s3_folder: Optional[str] = None,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> str:
    """Upload each audio file to S3 and write a metadata CSV with columns
    [audio (presigned URL), <transcription>, split, <carried-through cols…>] —
    keeping the source's splits and extra columns (e.g. speaker). Writes under
    the storage's prefix + `s3_folder` (default datasets/{id}/transformed).
    Re-runs skip audio already in the bucket. Returns the s3:// URI."""
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
    expires = 7 * 24 * 3600
    # Unique source files (the same clip can repeat across splits) → S3 key.
    key_of: dict[str, str] = {}
    for _split, path, _text, _extra in pairs:
        key_of.setdefault(path, f"{base}/audio/{os.path.basename(path)}")

    # Upload concurrently, reusing one client; re-runs skip clips already present.
    to_upload = [(key, path) for path, key in key_of.items() if key not in existing]
    total_up = len(to_upload)
    if total_up:
        every = max(1, total_up // 50)  # ~50 progress markers over the upload

        def _on_done(done: int) -> None:
            if progress and (done % every == 0 or done == total_up):
                progress("upload_s3", done, total_up)

        bench.s3_put_files(to_upload, s3_target, max_workers=16, on_done=_on_done)

    # Presign every unique key (local signing, one client), then write the CSV
    # with the carried-through columns (e.g. speaker) after audio/text/split.
    urls = bench.s3_presign_many(list(key_of.values()), expires, s3_target)
    extra_cols = sorted({k for *_rest, extra in pairs for k in extra})
    buf = _io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["audio", transcription_field, "split"] + extra_cols)
    for split, path, text, extra in pairs:
        writer.writerow([urls[key_of[path]], text, split] + [extra.get(c, "") for c in extra_cols])
    meta_key = f"{base}/metadata.csv"
    bench.s3_put_text(meta_key, buf.getvalue(), s3_target)
    bucket = getattr(s3_target, "bucket", "")
    return f"s3://{bucket}/{meta_key}"
