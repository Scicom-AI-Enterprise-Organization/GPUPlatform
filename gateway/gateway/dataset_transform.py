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
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .db import Dataset, Storage, session_factory

logger = logging.getLogger("gateway.dataset_transform")

# Force HF downloads onto the plain HTTPS path. The accelerated backends —
# `hf_transfer` (Rust multi-conn) and Xet (content-addressed dedup) — stall
# mid-transfer on some networks: the dataset clone hangs at a few % (observed
# wedging at ~14% of Scicom-intl/TM-Voice). Set here at import so a fresh
# huggingface_hub import reads them; `_download` also patches the live constants
# in case hf_hub was already imported by another module (its download path reads
# `constants.HF_HUB_DISABLE_XET`, computed once at import — so env-only is too
# late in a long-running gateway).
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"

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
    test_split_pct: Optional[float] = None,
    test_split_count: Optional[int] = None,
    test_min_chars: Optional[int] = None,
    test_exclude_regex: Optional[str] = None,
) -> None:
    """Mark the dataset running and kick off the background job."""
    async with session_factory()() as s:
        d = await s.get(Dataset, dataset_id)
        if d is None:
            return
        d.transform_status = "running"
        d.transform_log = _append_log(None, f"transform queued (target={target})")
        await s.commit()
    task = asyncio.create_task(_run(
        dataset_id, target, hf_repo, storage_id, s3_folder,
        test_split_pct=test_split_pct, test_split_count=test_split_count,
        test_min_chars=test_min_chars, test_exclude_regex=test_exclude_regex,
    ))
    _active[dataset_id] = task
    task.add_done_callback(lambda _t: _active.pop(dataset_id, None))


async def start_llm_pack(
    dataset_id: str,
    *,
    subsets: Optional[list[str]],
    tokenizer: str,
    sequence_length: int,
    storage_id: str,
    tools_field: Optional[str] = None,
    all_reasoning: bool = True,
) -> None:
    """Mark the dataset running and kick off the in-process chat→multipack job
    (kind=llm → kind=llm_packed ChiniDataset on S3). CPU-only tokenization, so it
    runs here (no GPU box) — the threadpool steps stay off the event loop.
    `subsets` may name several subset/split labels — their rows are concatenated
    into one packed dataset; None/empty packs the first split."""
    async with session_factory()() as s:
        d = await s.get(Dataset, dataset_id)
        if d is None:
            return
        d.transform_status = "running"
        d.transform_log = _append_log(
            None, f"LLM pack queued · subsets={', '.join(subsets) if subsets else '(first)'} "
                  f"· tokenizer={tokenizer} · seq_len {sequence_length}")
        await s.commit()
    task = asyncio.create_task(_run_llm_pack(
        dataset_id, subsets=subsets, tokenizer=tokenizer, sequence_length=sequence_length,
        storage_id=storage_id, tools_field=tools_field, all_reasoning=all_reasoning,
    ))
    _active[dataset_id] = task
    task.add_done_callback(lambda _t: _active.pop(dataset_id, None))


async def cancel_transform(dataset_id: str) -> bool:
    """Abort an in-flight audio-extraction OR LLM-pack transform for this dataset.
    Returns True if one was running. (Pack-only TTS runs are training runs —
    cancelled separately via training_api.cancel_pack_run_for_dataset.)"""
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


def _make_progress(dataset_id: str, loop: asyncio.AbstractEventLoop) -> Callable[..., None]:
    """A thread-safe progress callback for the blocking ETL steps (which run in a
    threadpool). Appends an `[AUTOTRAIN_PROGRESS]` marker to transform_log so the
    UI can show a percentage + ETA — the same marker format the TTS pack emits.
    `rate`, when given, is appended as `rate=<MB/s>` (the download step's transfer
    speed); the UI marker parser ignores unknown keys, so it's purely additive."""
    def progress(step: str, processed: int, total: int, rate: Optional[float] = None) -> None:
        pct = (processed / total * 100.0) if total else 0.0
        line = f"[AUTOTRAIN_PROGRESS] step={step} processed={processed} total={total} percent={pct:.1f}"
        if rate is not None:
            # Trailing unit reads cleanly in the log; the marker parser tokenises on
            # whitespace and ignores the unit-only `MB/s` token (and the `rate` key).
            line += f" rate={rate:.1f} MB/s"
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
    test_split_pct: Optional[float] = None,
    test_split_count: Optional[int] = None,
    test_min_chars: Optional[int] = None,
    test_exclude_regex: Optional[str] = None,
) -> None:
    from fastapi.concurrency import run_in_threadpool
    from sqlalchemy import select
    from .datasets_api import _hf_endpoint, _hf_token, _label_token, _s3_target_and_prefix

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
            src_revision = d.hf_revision  # commit/branch/tag to fetch (None → default)
            audio_field, transcription_field = d.audio_field, d.transcription_field
            split_fields = dict(d.split_fields or {})  # {split: transcription_column}
            hf_store = (
                await s.execute(select(Storage).where(Storage.kind == "huggingface").limit(1))
            ).scalars().first()
            token = await _hf_token(hf_store, s)
            hf_endpoint = await _hf_endpoint(hf_store, s)  # custom Hub endpoint, or None
            out_storage = await s.get(Storage, storage_id) if (target == "s3" and storage_id) else None
            s3_target, s3_prefix = (_s3_target_and_prefix(out_storage) if out_storage else (None, ""))
            # Labeling-platform source (kind="label"): base URL + project + token.
            label_base_url = d.label_base_url
            label_project_id = d.label_project_id
            label_status = d.label_status or "approved"
            label_updated_until = d.label_updated_until
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
            _cut = f", until={label_updated_until}" if label_updated_until else ""
            await _log(dataset_id, f"exporting label project {label_project_id} (status={label_status}{_cut}) …")
            pairs = await run_in_threadpool(
                _label_pairs, label_base_url, label_project_id, label_token, label_status, work, "",
                label_updated_until,
            )
            if not pairs:
                await _finish(dataset_id, "failed", "no tasks with downloadable audio in the label export (check the status filter / token)")
                return
        else:
            rev_note = f" @ {src_revision}" if src_revision else ""
            await _log(dataset_id, f"downloading {src_repo}{rev_note} … (reuses cached files on re-run)")
            await run_in_threadpool(_download, src_repo, token, work, progress, src_revision, hf_endpoint)

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

        # Optional held-out test split: reassign each row's split to train/test.
        if test_split_pct is not None or test_split_count is not None:
            pairs, n_test = _apply_test_split(
                pairs, test_split_pct, test_split_count,
                min_chars=int(test_min_chars or 0), exclude_regex=(test_exclude_regex or None),
            )
            _filters = []
            if test_min_chars:
                _filters.append(f"≥ {int(test_min_chars)} chars")
            if test_exclude_regex:
                _filters.append(f"excl /{test_exclude_regex}/")
            _f = f" (test eligibility: {', '.join(_filters)})" if _filters else ""
            await _log(
                dataset_id,
                f"test split → {n_test} test / {len(pairs) - n_test} train rows{_f}",
            )

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


# ---------------- merge label datasets → one combined audio dataset ----------


async def start_merge(
    output_id: str,
    *,
    source_ids: list[str],
    target: str,
    hf_repo: Optional[str],
    storage_id: Optional[str],
    s3_folder: Optional[str] = None,
) -> None:
    """Mark the (already-created) merged OUTPUT dataset running and kick off the
    background merge: export + download each source label project's audio and
    write ONE combined audio dataset (HF or S3). Status/log live on the OUTPUT
    dataset (poll GET /{output_id})."""
    async with session_factory()() as s:
        d = await s.get(Dataset, output_id)
        if d is None:
            return
        d.transform_status = "running"
        d.transform_log = _append_log(
            None, f"merge queued · {len(source_ids)} source(s) → {target}")
        await s.commit()
    task = asyncio.create_task(_run_merge(
        output_id, source_ids=source_ids, target=target, hf_repo=hf_repo,
        storage_id=storage_id, s3_folder=s3_folder,
    ))
    _active[output_id] = task
    task.add_done_callback(lambda _t: _active.pop(output_id, None))


async def _run_merge(
    output_id: str,
    *,
    source_ids: list[str],
    target: str,
    hf_repo: Optional[str],
    storage_id: Optional[str],
    s3_folder: Optional[str],
) -> None:
    """Concatenate several kind=label datasets into one audio dataset. Each source
    is exported + its clips downloaded into its own scratch subdir (per-source
    filename prefix → unique S3 basenames), then all pairs are concatenated and
    written once via the SAME HF/S3 writers the single-source transform uses."""
    from fastapi.concurrency import run_in_threadpool
    from sqlalchemy import select

    from .datasets_api import _hf_endpoint, _hf_token, _label_token, _s3_target_and_prefix

    work = _work_dir(output_id) + "-merge"
    success = False
    progress = _make_progress(output_id, asyncio.get_running_loop())
    try:
        if os.path.isdir(work):
            shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work, exist_ok=True)

        # Resolve output storage + HF token, and each source's label config.
        async with session_factory()() as s:
            out = await s.get(Dataset, output_id)
            if out is None:
                return
            transcription_field = out.transcription_field or "transcription"
            hf_store = (
                await s.execute(select(Storage).where(Storage.kind == "huggingface").limit(1))
            ).scalars().first()
            token = await _hf_token(hf_store, s)
            _hf_endpoint_unused = await _hf_endpoint(hf_store, s)  # noqa: F841 — parity with _run
            out_storage = await s.get(Storage, storage_id) if (target == "s3" and storage_id) else None
            s3_target, s3_prefix = (_s3_target_and_prefix(out_storage) if out_storage else (None, ""))
            sources: list[dict] = []
            for sid in source_ids:
                sd = await s.get(Dataset, sid)
                if sd is None:
                    await _finish(output_id, "failed", f"source dataset {sid} not found")
                    return
                if sd.kind not in ("label", "s3"):
                    await _finish(output_id, "failed",
                                  f"source {sid} is kind={sd.kind}; merge supports kind=label and kind=s3 sources")
                    return
                if sd.kind == "label":
                    sources.append({
                        "id": sid, "name": sd.name, "kind": "label",
                        "base_url": sd.label_base_url, "project_id": sd.label_project_id,
                        "status": sd.label_status or "approved",
                        "until": sd.label_updated_until,
                        "token": await _label_token(sd, s),
                    })
                else:  # s3 — read its materialised metadata.csv + audio/ folder.
                    src_storage = await s.get(Storage, sd.storage_id) if sd.storage_id else None
                    src_target, _src_prefix = (
                        _s3_target_and_prefix(src_storage) if src_storage else (None, "")
                    )
                    sources.append({
                        "id": sid, "name": sd.name, "kind": "s3",
                        "s3_target": src_target,
                        "s3_metadata_uri": sd.s3_metadata_uri,
                        "audio_field": sd.audio_field or "audio",
                        "transcription_field": sd.transcription_field or "transcription",
                    })

        if target == "hf" and not (hf_repo and "/" in hf_repo):
            await _finish(output_id, "failed", "target HF repo must be owner/name")
            return
        if target == "s3" and out_storage is None:
            await _finish(output_id, "failed", "pick an S3 storage for the S3 target")
            return

        all_pairs: list[tuple[str, str, str, dict]] = []
        for idx, src in enumerate(sources):
            sub = os.path.join(work, f"src-{idx}")
            if src["kind"] == "label":
                if not (src["base_url"] and src["project_id"] and src["token"]):
                    await _finish(output_id, "failed",
                                  f"source '{src['name']}' is not a fully-configured label dataset "
                                  f"(base URL, project, token)")
                    return
                _cut = f", until={src['until']}" if src.get("until") else ""
                await _log(output_id,
                           f"[{idx + 1}/{len(sources)}] exporting '{src['name']}' "
                           f"(project {src['project_id']}, status={src['status']}{_cut}) …")
                pairs = await run_in_threadpool(
                    _label_pairs, src["base_url"], src["project_id"], src["token"],
                    src["status"], sub, f"s{idx}-", src["until"],
                )
            else:  # s3
                if not (src["s3_target"] and src["s3_metadata_uri"]):
                    await _finish(output_id, "failed",
                                  f"source '{src['name']}' is an s3 dataset without storage / a metadata file")
                    return
                # Fast path: when the source and destination S3 are the same account
                # (same creds/region/endpoint) and we're writing to S3, copy the audio
                # objects server-side (S3→S3) instead of pulling every clip down through
                # the gateway and re-uploading. Falls back to download for target=hf or
                # a cross-account source.
                if target == "s3" and _same_account(src["s3_target"], s3_target):
                    await _log(output_id,
                               f"[{idx + 1}/{len(sources)}] copying s3 dataset '{src['name']}' "
                               f"server-side ({src['s3_metadata_uri']}) …")
                    pairs = await run_in_threadpool(
                        _s3_copy_pairs, src["s3_target"], src["s3_metadata_uri"],
                        src["audio_field"], src["transcription_field"], f"s{idx}-",
                    )
                else:
                    await _log(output_id,
                               f"[{idx + 1}/{len(sources)}] reading s3 dataset '{src['name']}' "
                               f"({src['s3_metadata_uri']}) …")
                    pairs = await run_in_threadpool(
                        _s3_pairs, src["s3_target"], src["s3_metadata_uri"],
                        src["audio_field"], src["transcription_field"], sub, f"s{idx}-", progress,
                    )
            await _log(output_id, f"  ↳ {len(pairs)} clip(s) from '{src['name']}'")
            all_pairs.extend(pairs)

        if not all_pairs:
            await _finish(output_id, "failed",
                          "no tasks with downloadable audio across the selected sources "
                          "(check each project's status filter / token)")
            return
        await _log(output_id,
                   f"merged {len(all_pairs)} rows from {len(sources)} sources; building output …")

        if target == "hf":
            await run_in_threadpool(_push_hf, all_pairs, transcription_field, hf_repo, token)
            await _finish(
                output_id, "done",
                f"pushed → {hf_repo} ({len(all_pairs)} rows from {len(sources)} sources)",
                hf_repo=hf_repo, num_rows=len(all_pairs))
            success = True
        else:
            uri = await run_in_threadpool(
                _materialise_s3, all_pairs, transcription_field, s3_target, s3_prefix,
                output_id, s3_folder, progress,
            )
            await _finish(
                output_id, "done",
                f"materialised → {uri} ({len(all_pairs)} rows from {len(sources)} sources)",
                s3_metadata_uri=uri, num_rows=len(all_pairs))
            success = True
    except ModuleNotFoundError as e:
        await _finish(output_id, "failed",
                      f"missing dependency: {e}. Install datasets + soundfile in the gateway.")
    except Exception as e:  # noqa: BLE001
        await _finish(output_id, "failed", f"merge failed: {e}")
        logger.exception("merge %s failed", output_id)
    finally:
        if work and success:
            shutil.rmtree(work, ignore_errors=True)


# ---------------- LLM chat → multipack (kind=llm → kind=llm_packed) ----------


async def _create_llm_packed_output(
    source_id: str, *, new_id: str, name: str, description: str, storage_id: str,
    s3_uri: str, num_rows: int, messages_field: str, tokenizer: str,
    sequence_length: int, subset: Optional[str], arch: Optional[str] = None,
) -> str:
    """Create the packed (chat multipack) dataset as a NEW row, leaving the source
    intact + re-runnable. `_llm_pack` metadata (in split_fields) mirrors tts_packed's
    `_tts_pack` so the packed-row preview/decode + the trainer can read it. Returns id."""
    async with session_factory()() as s:
        src = await s.get(Dataset, source_id)
        owner_id = src.owner_id if src else None
        new = Dataset(
            id=new_id,
            owner_id=owner_id,
            name=name[:255],
            description=description[:2048],
            kind="llm_packed",
            format="chinidataset",  # multipacked chat layout (ChiniDataset parquet shards)
            storage_id=storage_id,
            s3_metadata_uri=s3_uri,
            num_rows=num_rows,
            messages_field=messages_field,
            # tokenizer + sequence_length live in the _llm_pack blob (mirrors how
            # tts_packed stashes them under _tts_pack — there are no DB columns).
            # NB: no `splits` map → the pack is a single FLAT ChiniDataset (shards
            # at the prefix root), so the packed-row reader doesn't append a split
            # subdir (unlike tts_packed's <prefix>/<split>/ layout).
            split_fields={"_llm_pack": {
                "tokenizer": tokenizer,
                "sequence_length": sequence_length,
                "subset": subset,
                "messages_field": messages_field,
                "samples": num_rows,
                # Packing arch (gemma|minimax|generic) — drives the trainer choice
                # and lets a run reject a base-model/dataset arch mismatch.
                "arch": arch,
            }},
        )
        s.add(new)
        await s.commit()
        return new.id


def _read_split_columns(parquet_paths: list[str], cols: list[str]) -> list[dict]:
    """Read only `cols` from the downloaded parquet shards → list[dict] (blocking).
    Keeping it to the messages/tools columns keeps memory bounded for big splits."""
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for p in parquet_paths:
        tbl = pq.read_table(p)
        present = [c for c in cols if c in tbl.column_names]
        rows.extend(tbl.select(present).to_pylist())
    return rows


def _upload_chinidataset_dir(out_dir: str, target, key_prefix: str,
                             progress: Optional[Callable[..., None]] = None) -> str:
    """Upload every file under a local ChiniDataset dir (index.json + shards) to
    S3 under `key_prefix`, preserving relative paths. Returns the s3:// prefix URI
    (what the packed-row preview + trainer download)."""
    from . import bench

    to_upload: list[tuple[str, str]] = []
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            local = os.path.join(root, f)
            rel = os.path.relpath(local, out_dir).replace(os.sep, "/")
            to_upload.append((f"{key_prefix}/{rel}", local))
    total = len(to_upload)
    every = max(1, total // 25)

    def _on_done(done: int) -> None:
        if progress and (done % every == 0 or done == total):
            progress("upload_s3", done, total)

    bench.s3_put_files(to_upload, target, max_workers=16, on_done=_on_done)
    bucket = getattr(target, "bucket", "")
    return f"s3://{bucket}/{key_prefix}"


async def _run_llm_pack(
    dataset_id: str,
    *,
    subsets: Optional[list[str]],
    tokenizer: str,
    sequence_length: int,
    storage_id: str,
    tools_field: Optional[str],
    all_reasoning: bool,
) -> None:
    from fastapi.concurrency import run_in_threadpool

    from sqlalchemy import select

    from . import llm_pack
    from .datasets_api import (
        _hf_endpoint, _hf_parquet_files, _hf_token, _join_key, _resolve_hf_subset,
        _s3_target_and_prefix,
    )

    work = _work_dir(dataset_id) + "-llmpack"
    success = False
    progress = _make_progress(dataset_id, asyncio.get_running_loop())
    try:
        if os.path.isdir(work):
            shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work, exist_ok=True)

        async with session_factory()() as s:
            d = await s.get(Dataset, dataset_id)
            if d is None:
                return
            src_repo = d.hf_repo
            src_revision = d.hf_revision
            src_name = d.name
            src_metadata_filename = d.metadata_filename
            messages_field = (getattr(d, "messages_field", None) or "messages")
            # HF token + endpoint: prefer the dataset's OWN storage when it's a
            # huggingface backend (carries the token for a gated repo), else fall
            # back to any configured huggingface storage (mirrors _run).
            own = await s.get(Storage, d.storage_id) if d.storage_id else None
            hf_store = own if (own and own.kind == "huggingface") else (
                await s.execute(select(Storage).where(Storage.kind == "huggingface").limit(1))
            ).scalars().first()
            token = await _hf_token(hf_store, s)
            hf_endpoint = await _hf_endpoint(hf_store, s)
            out_storage = await s.get(Storage, storage_id)

        if out_storage is None or out_storage.kind != "s3":
            await _finish(dataset_id, "failed", "pick a kind=s3 storage for the packed output")
            return
        target, base_prefix = _s3_target_and_prefix(out_storage)

        has_hf = bool(src_repo and "/" in src_repo)
        has_file = bool(own is not None and own.kind == "s3" and src_metadata_filename)
        if not (has_hf or has_file):
            await _finish(dataset_id, "failed",
                          "dataset has no source HuggingFace repo or uploaded file to pack")
            return

        if has_hf:
            # 1. Resolve each chosen subset → (config, split) + its parquet files. Several
            #    subsets are concatenated into one packed dataset (one combined URL list,
            #    so the download step indexes filenames globally → no collisions).
            want = subsets or [None]
            labels: list[str] = []
            urls: list[str] = []
            for sub in want:
                await _log(dataset_id, f"resolving subset {sub or '(first)'} on {src_repo} …")
                config, split, label = await run_in_threadpool(
                    _resolve_hf_subset, src_repo, token, sub, src_revision)
                sub_urls = await run_in_threadpool(
                    _hf_parquet_files, src_repo, token, config, split, src_revision)
                if not sub_urls:
                    await _finish(dataset_id, "failed",
                                  f"no parquet files for subset '{label}' (config={config}, split={split}) "
                                  f"on the HF datasets-server")
                    return
                labels.append(label)
                urls.extend(sub_urls)
                await _log(dataset_id,
                           f"subset '{label}' → config={config} split={split}; {len(sub_urls)} parquet file(s)")
            label = ", ".join(labels)
            if len(labels) > 1:
                await _log(dataset_id, f"{len(labels)} subsets [{label}] → {len(urls)} parquet file(s) total")

            # 2. Download the parquet shards (auth via the HF token).
            await _log(dataset_id, "downloading source parquet …")
            paths = await run_in_threadpool(_download_parquet_urls, urls, token, work, progress)

            # 3. Read messages (+ tools) columns → rows.
            cols = [messages_field] + ([tools_field] if tools_field else [])
            rows = await run_in_threadpool(_read_split_columns, paths, cols)
        else:
            # S3-backed upload: read the uploaded chat file (json / jsonl / parquet)
            # directly from the dataset's own storage — no HF datasets-server.
            import json

            from . import bench, dataset_metadata
            from .datasets_api import _metadata_key

            src_target, _ = _s3_target_and_prefix(own)
            key = _metadata_key(own, dataset_id, src_metadata_filename)
            await _log(dataset_id, f"reading uploaded chat file {src_metadata_filename} from S3 …")
            body = await run_in_threadpool(bench.s3_get_bytes, key, src_target)
            if not body:
                await _finish(dataset_id, "failed",
                              f"uploaded file {src_metadata_filename} not found in storage")
                return
            rows = await run_in_threadpool(
                dataset_metadata.parse_rows_any, src_metadata_filename, body, 10 ** 9)
            if rows and messages_field not in rows[0]:
                await _finish(dataset_id, "failed",
                              f"no column '{messages_field}' in {src_metadata_filename} — "
                              f"check the messages column mapping")
                return
            # Some exports store the messages array as a JSON string — parse it back
            # to a real list so pack_rows sees a conversation (matches the preview).
            for r in rows:
                v = r.get(messages_field)
                if isinstance(v, str):
                    try:
                        parsed = json.loads(v)
                        if isinstance(parsed, list):
                            r[messages_field] = parsed
                    except (json.JSONDecodeError, ValueError):
                        pass
            label = src_name

        if not rows:
            await _finish(dataset_id, "failed",
                          f"no rows read for column '{messages_field}' — check the messages column mapping")
            return
        await _log(dataset_id, f"read {len(rows)} rows; tokenizing + multipacking with {tokenizer} …")

        # 4. Tokenize + bin-pack into a ChiniDataset (CPU; threadpool).
        out_dir = os.path.join(work, "packed")

        def _pack_progress(p: int, t: int) -> None:
            progress("pack", p, t)

        stats = await run_in_threadpool(
            llm_pack.pack_rows, rows,
            tokenizer_name=tokenizer, out_dir=out_dir,
            messages_field=messages_field, tools_field=(tools_field or ""),
            max_seq_len=int(sequence_length), hf_token=token, hf_endpoint=hf_endpoint,
            all_reasoning=all_reasoning, progress=_pack_progress,
        )
        n_bins = int(stats.get("n_bins") or 0)
        if n_bins == 0:
            await _finish(dataset_id, "failed",
                          f"packed 0 bins (rows={stats.get('total_rows')}, dropped_long="
                          f"{stats.get('dropped_long')}, dropped_empty={stats.get('dropped_empty')}) "
                          f"— is seq_len {sequence_length} too small for every conversation?")
            return
        await _log(
            dataset_id,
            f"packed {n_bins} bins from {stats.get('docs_packed')} docs "
            f"(dropped {stats.get('dropped_long')} too-long, {stats.get('dropped_empty')} empty; "
            f"{stats.get('rows_with_tools')} rows had tools; efficiency "
            f"{stats.get('efficiency', 0) * 100:.1f}%); uploading shards …")

        # 5. Upload the ChiniDataset dir to S3, then register the packed dataset.
        # Mint the new dataset id up front so its S3 prefix matches its row id.
        import secrets as _secrets
        new_id = f"ds-{_secrets.token_hex(4)}"
        key_prefix = _join_key(base_prefix, "datasets", new_id, "packed")
        s3_uri = await run_in_threadpool(_upload_chinidataset_dir, out_dir, target, key_prefix, progress)

        created_id = await _create_llm_packed_output(
            dataset_id, new_id=new_id,
            name=f"{src_name}-llm-packed",
            description=(f"Chat multipack (seq_len {sequence_length}, tokenizer {tokenizer}, "
                         f"subset {label}, arch {stats.get('arch')}) of "
                         f"{src_repo or src_metadata_filename} → {n_bins} bins"),
            storage_id=storage_id, s3_uri=s3_uri, num_rows=n_bins,
            messages_field=messages_field, tokenizer=tokenizer,
            sequence_length=int(sequence_length), subset=label, arch=stats.get("arch"),
        )
        await _finish(
            dataset_id, "done",
            f"packed → {s3_uri} ({n_bins} bins); created dataset {created_id}")
        success = True
    except ModuleNotFoundError as e:
        await _finish(dataset_id, "failed",
                      f"missing dependency: {e}. The gateway needs transformers (+ jinja2) for chat templates.")
        logger.exception("llm pack %s failed (missing dep)", dataset_id)
    except Exception as e:  # noqa: BLE001
        await _finish(dataset_id, "failed", f"LLM pack failed: {e}")
        logger.exception("llm pack %s failed", dataset_id)
    finally:
        if work and success:
            shutil.rmtree(work, ignore_errors=True)


def _download_parquet_urls(
    urls: list[str], token: Optional[str], dest: str,
    progress: Optional[Callable[..., None]] = None,
) -> list[str]:
    """Download each parquet URL into `dest` (auth header from the HF token).
    Returns local paths. Emits a per-file `download` progress marker."""
    import httpx

    os.makedirs(dest, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    paths: list[str] = []
    total = len(urls)
    with httpx.Client(timeout=None, follow_redirects=True, headers=headers) as cli:
        for i, url in enumerate(urls):
            local = os.path.join(dest, f"source-{i:05d}.parquet")
            with cli.stream("GET", url) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1 << 20):
                        f.write(chunk)
            paths.append(local)
            if progress:
                progress("download", i + 1, total)
    return paths


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
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> str:
    """snapshot_download the dataset repo into the STABLE `dest` — re-runs skip
    files already present (no network re-fetch). Emits a byte-based `download`
    progress marker (snapshot_download itself is opaque, so the UI otherwise
    shows no %) by polling the on-disk size against the repo's total size.
    `revision` (commit/branch/tag) pins which ref to fetch (None → default)."""
    import threading
    from huggingface_hub import HfApi, snapshot_download

    # Belt-and-suspenders to the import-time env set above: if hf_hub was already
    # imported by another module, its constants are fixed, so patch them on the
    # live module (the download path reads `constants.HF_HUB_DISABLE_XET`). Forces
    # the plain HTTPS path and avoids the hf_transfer/Xet mid-transfer stall.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        import huggingface_hub.constants as _hfc
        _hfc.HF_HUB_DISABLE_XET = True
        if hasattr(_hfc, "HF_HUB_ENABLE_HF_TRANSFER"):
            _hfc.HF_HUB_ENABLE_HF_TRANSFER = False
    except Exception:  # noqa: BLE001 — best-effort; env vars still apply
        pass

    os.makedirs(dest, exist_ok=True)
    total_bytes = 0
    try:
        info = HfApi(token=token, endpoint=endpoint or None).repo_info(repo, repo_type="dataset", revision=revision, files_metadata=True)
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
        prev_bytes = _dir_bytes()
        prev_t = time.monotonic()
        while not stop.wait(2.0):
            if not (progress and total_bytes):
                continue
            cur_bytes = _dir_bytes()
            now = time.monotonic()
            dt = now - prev_t
            # MB/s over this interval — the live download speed (0 if the dir didn't
            # grow, e.g. a cached re-run where snapshot_download fetches nothing).
            rate = max(0.0, (cur_bytes - prev_bytes) / dt / mb) if dt > 0 else 0.0
            prev_bytes, prev_t = cur_bytes, now
            progress("download", min(cur_bytes, total_bytes) // mb, total_bytes // mb, rate)

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest, token=token, revision=revision, endpoint=endpoint or None)
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
    name_prefix: str = "",
    until: Optional[str] = None,
) -> list[tuple[str, str, str, dict]]:
    """Stream a labeling-platform project export (export.v1.jsonl) and download
    each task's audio into `work/audio`. Returns [(split, abs_audio_path,
    transcription, extra)] — the same shape `_build_pairs` produces for HF
    sources (extra is empty here), so the HF/S3 target writers are reused.

    `name_prefix` is prepended to each downloaded filename — used by the merge
    job to keep basenames unique ACROSS projects (S3 keys collide on basename;
    two projects can each have a `task-5.wav`). Empty for the single-project path.
    `until` (ISO-8601) is the point-in-time cutoff forwarded to the export so only
    tasks last updated at or before it are pulled.

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
    rows, _total = _label_export_rows(base_url, project_id, token, status, 10**9, 0, until)

    audio_dir = Path(work) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, str, str, dict]] = []
    with httpx.Client(timeout=120.0, follow_redirects=True) as cli:
        for i, r in enumerate(rows):
            u = str(r.get("audio_url") or "").strip()
            tid = r.get("id")
            # Candidate download URLs, in priority order. A presigned off-platform
            # `audio_url` is fastest (no platform load); but some storage configs
            # presign an S3 endpoint we can't resolve (e.g. a non-routable region
            # host), so fall back to the platform's authenticated task-audio
            # endpoint whenever there's a task id.
            candidates: list[tuple[str, dict]] = []
            if u and not u.startswith(base):
                candidates.append((u, {}))                       # direct presigned
            if tid is not None:
                candidates.append((f"{base}/api/projects/{project_id}/tasks/{tid}/audio",
                                   {"Authorization": f"Bearer {token}"}))
            elif u and u.startswith(base):
                candidates.append((u, {"Authorization": f"Bearer {token}"}))  # on-platform; needs token
            if not candidates:
                continue
            resp = None
            for dl_url, dl_headers in candidates:
                try:
                    rr = cli.get(dl_url, headers=dl_headers)
                    rr.raise_for_status()
                    resp = rr
                    break
                except Exception:  # noqa: BLE001 — try the next candidate
                    continue
            if resp is None:
                logger.warning("label audio download failed (task=%s)", tid)
                continue
            # Pick an extension: URL basename, else the response content-type.
            ext = _os.path.splitext(unquote(_os.path.basename(urlparse(u).path)))[1].lower()
            if ext not in _AUDIO_EXTS:
                ext = _CT_EXT.get((resp.headers.get("content-type") or "").split(";")[0].strip().lower(), ".wav")
            name = f"{name_prefix}task-{tid}{ext}" if tid is not None else f"{name_prefix}row-{i}{ext}"
            dest = audio_dir / name
            dest.write_bytes(resp.content)
            text = r.get("transcription")
            text = "" if text is None else str(text)
            split = str(r.get("split") or "train")
            pairs.append((split, str(dest), text, {}))
    return pairs


def _same_account(a, b) -> bool:
    """Whether two S3Targets can server-side copy between each other — same creds,
    region, and endpoint, so a CopyObject issued by the dest client can read the
    source bucket. Bucket may differ (cross-bucket copy in one account is fine)."""
    if a is None or b is None:
        return False
    return (
        (a.access_key or "") == (b.access_key or "")
        and (a.region or "") == (b.region or "")
        and (a.endpoint or "") == (b.endpoint or "")
    )


def _s3_copy_pairs(
    s3_target,
    s3_metadata_uri: str,
    audio_field: str,
    transcription_field: str,
    name_prefix: str = "",
) -> list[tuple[str, tuple, str, dict]]:
    """Like `_s3_pairs`, but does NOT download: read the source metadata CSV and
    return copy-markers so `_materialise_s3` server-side copies each clip S3→S3.
    Each pair's audio element is an ("s3cp", src_bucket, src_key, dest_basename)
    tuple. `name_prefix` keeps dest basenames unique across merged sources — the
    SAME contract as `_s3_pairs`, so the two can be mixed in one merge."""
    import csv
    import io as _io
    from urllib.parse import unquote, urlparse

    from . import bench

    u = urlparse(s3_metadata_uri or "")
    meta_key = (u.path if u.scheme == "s3" else (s3_metadata_uri or "")).lstrip("/")
    if not meta_key:
        raise RuntimeError(f"unusable s3 metadata uri: {s3_metadata_uri!r}")
    base = meta_key.rsplit("/", 1)[0] if "/" in meta_key else ""
    src_bucket = getattr(s3_target, "bucket", "")

    text = bench.s3_get_text(meta_key, s3_target)
    if not text:
        raise RuntimeError(f"could not read metadata at {s3_metadata_uri}")
    reader = csv.DictReader(_io.StringIO(text))
    cols = reader.fieldnames or []
    a_col = audio_field if audio_field in cols else ("audio" if "audio" in cols else None)
    t_col = transcription_field if transcription_field in cols else (
        "transcription" if "transcription" in cols else None
    )
    if not a_col:
        raise RuntimeError(f"audio column '{audio_field}' not found in {meta_key} (have {cols})")
    skip = {a_col, t_col, "split"}
    pairs: list[tuple[str, tuple, str, dict]] = []
    for i, r in enumerate(reader):
        ref = (r.get(a_col) or "").strip()
        if not ref:
            continue
        basename = unquote(os.path.basename(urlparse(ref).path)) or f"row-{i}.wav"
        src_key = f"{base}/audio/{basename}" if base else f"audio/{basename}"
        marker = ("s3cp", src_bucket, src_key, f"{name_prefix}{basename}")
        text_val = (r.get(t_col) or "") if t_col else ""
        split = (r.get("split") or "train").strip() or "train"
        extra = {k: v for k, v in r.items() if k not in skip and isinstance(v, str) and v != ""}
        pairs.append((split, marker, str(text_val), extra))
    return pairs


def _s3_pairs(
    s3_target,
    s3_metadata_uri: str,
    audio_field: str,
    transcription_field: str,
    work: str,
    name_prefix: str = "",
    progress: Optional[Callable[..., None]] = None,
) -> list[tuple[str, str, str, dict]]:
    """Read a materialised S3 audio dataset (a metadata CSV + an `audio/` folder,
    the layout `_materialise_s3` writes) and download each clip into `work/audio`,
    returning [(split, abs_audio_path, transcription, extra)] — the SAME shape
    `_label_pairs` produces, so the merge's HF/S3 writers are reused.

    Audio keys are resolved as `{base}/audio/{basename}` under the metadata file's
    folder; the CSV's audio cell may be a presigned URL, an s3:// uri, or a path —
    only its basename matters. `name_prefix` keeps basenames unique across sources
    (S3 keys collide on basename), matching `_label_pairs`'s merge contract."""
    import csv
    import io as _io
    from urllib.parse import unquote, urlparse

    from . import bench

    u = urlparse(s3_metadata_uri or "")
    meta_key = (u.path if u.scheme == "s3" else (s3_metadata_uri or "")).lstrip("/")
    if not meta_key:
        raise RuntimeError(f"unusable s3 metadata uri: {s3_metadata_uri!r}")
    base = meta_key.rsplit("/", 1)[0] if "/" in meta_key else ""

    text = bench.s3_get_text(meta_key, s3_target)
    if not text:
        raise RuntimeError(f"could not read metadata at {s3_metadata_uri}")
    reader = csv.DictReader(_io.StringIO(text))
    cols = reader.fieldnames or []
    a_col = audio_field if audio_field in cols else ("audio" if "audio" in cols else None)
    t_col = transcription_field if transcription_field in cols else (
        "transcription" if "transcription" in cols else None
    )
    if not a_col:
        raise RuntimeError(f"audio column '{audio_field}' not found in {meta_key} (have {cols})")

    audio_dir = Path(work) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = list(reader)
    total = len(rows)
    every = max(1, total // 50)
    skip = {a_col, t_col, "split"}
    pairs: list[tuple[str, str, str, dict]] = []
    for i, r in enumerate(rows):
        ref = (r.get(a_col) or "").strip()
        if not ref:
            continue
        basename = unquote(os.path.basename(urlparse(ref).path)) or f"row-{i}.wav"
        # Prefer the {base}/audio/{basename} key (re-signed by our own client, so
        # an expired URL in the CSV doesn't bite); fall back to the raw URL.
        key = f"{base}/audio/{basename}" if base else f"audio/{basename}"
        data = bench.s3_get_bytes(key, s3_target)
        if data is None and ref.lower().startswith(("http://", "https://")):
            try:
                import httpx
                rr = httpx.get(ref, timeout=120)
                rr.raise_for_status()
                data = rr.content
            except Exception:  # noqa: BLE001 — skip a row we can't fetch
                data = None
        if data is None:
            continue
        dest = audio_dir / f"{name_prefix}{basename}"
        dest.write_bytes(data)
        text_val = (r.get(t_col) or "") if t_col else ""
        split = (r.get("split") or "train").strip() or "train"
        extra = {k: v for k, v in r.items() if k not in skip and isinstance(v, str) and v != ""}
        pairs.append((split, str(dest), str(text_val), extra))
        if progress and (i % every == 0 or i == total - 1):
            progress("download_s3", i + 1, total)
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


def _declared_splits(work: str) -> Optional[list[tuple[str, str]]]:
    """Parse the repo README's `configs:` front-matter → [(split_label, path_glob)],
    so a metadata file maps to the dataset's DECLARED split (the same train/test
    HF shows) instead of being guessed from its filename. Files matching no
    declared pattern aren't part of the dataset and are dropped by the caller.
    Returns None when no configs are declared (caller falls back to filename
    inference). Labels mirror the row preview's config-vs-split naming."""
    import re

    import yaml

    readme = Path(work) / "README.md"
    if not readme.is_file():
        return None
    try:
        text = readme.read_text(errors="replace")
        m = re.match(r"^﻿?---\s*\n(.*?)\n---", text, re.S)
        meta = yaml.safe_load(m.group(1)) if m else None
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    triples: list[tuple[str, str, str]] = []  # (config, split, glob)
    for c in (meta.get("configs") or []):
        cfg = str((c or {}).get("config_name") or "default")
        for df in ((c or {}).get("data_files") or []):
            sp, path = (df or {}).get("split"), (df or {}).get("path")
            if not sp or not path:
                continue
            for p in ([path] if isinstance(path, str) else path):
                triples.append((cfg, str(sp), str(p)))
    if not triples:
        return None
    # Label by whichever of config/split is distinct (matches _hf_split_ident).
    pairs = sorted({(c, s) for c, s, _ in triples})
    if len({c for c, _ in pairs}) == len(pairs):
        ident = lambda c, s: c  # noqa: E731
    elif len({s for _, s in pairs}) == len(pairs):
        ident = lambda c, s: s  # noqa: E731
    else:
        ident = lambda c, s: f"{c}/{s}"  # noqa: E731
    return [(ident(c, s), g) for c, s, g in triples]


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

    import fnmatch

    # Prefer the source's DECLARED splits (README configs) so the output follows
    # the original train/test rather than parquet directory/file names; falls
    # back to filename inference for plain repos with no configs.
    declared = _declared_splits(work)
    frames = []
    seen_cols: set[str] = set()
    skipped_undeclared = 0
    for meta in candidates:
        try:
            df = _load_table(meta)
        except Exception:
            continue
        seen_cols.update(map(str, df.columns))
        if audio_field not in df.columns:
            continue
        if declared is not None:
            try:
                rel = str(meta.relative_to(root)).replace(os.sep, "/")
            except ValueError:
                rel = meta.name
            split = next((lbl for lbl, glob in declared if fnmatch.fnmatch(rel, glob)), None)
            if split is None:
                # Not referenced by any declared config → not part of the dataset.
                skipped_undeclared += 1
                continue
        else:
            split = _split_of(meta, root)
        tcol = split_fields.get(split) or transcription_field
        out = df.copy()  # keep every column (speaker, etc.); audio col is replaced later
        # Missing column for this split → blank, not a hard failure (the split may
        # genuinely lack a transcription, or the mapping only covers some splits).
        out["__text__"] = df[tcol] if tcol in df.columns else ""
        out["__split__"] = split  # carried through so the output keeps its splits
        frames.append(out)
    if declared is not None and skipped_undeclared:
        logger.info(
            "transform: kept %d declared-split tables, dropped %d not referenced by the "
            "source's configs (e.g. loose root parquets)", len(frames), skipped_undeclared,
        )
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


def _apply_test_split(
    pairs: list[tuple[str, str, str, dict]],
    pct: Optional[float],
    count: Optional[int],
    seed: int = 42,
    min_chars: int = 0,
    exclude_regex: Optional[str] = None,
) -> tuple[list[tuple[str, str, str, dict]], int]:
    """Reassign each pair's split to `train`/`test`, carving out a random held-out
    subset (collapsing any source splits). `count` (absolute) wins over `pct`
    (percentage of all matched rows). A row is ELIGIBLE to become test only if its
    transcription (after stripping) is at least `min_chars` characters AND does not
    match `exclude_regex` (a Python regex, `re.search`). So junk/placeholder
    transcripts (e.g. `[silent]`, `[unintelligible]`, matched by `^\\s*\\[.*\\]\\s*$`)
    stay in train instead of polluting eval; ineligible rows are never chosen as
    test. Deterministic for a given `seed`. Returns (pairs, n_test)."""
    import random
    import re

    n = len(pairs)
    mc = max(0, int(min_chars or 0))
    pat = re.compile(exclude_regex) if exclude_regex else None

    def _eligible(text: str) -> bool:
        t = (text or "").strip()
        if len(t) < mc:
            return False
        if pat is not None and pat.search(t):
            return False
        return True

    # Indices eligible to become test (mc=0 + no regex → all rows).
    eligible = [i for i, (_s, _p, text, _e) in enumerate(pairs) if _eligible(text)]
    if count is not None:
        k = count
    elif pct is not None:
        k = round(n * pct / 100.0)
    else:
        return pairs, 0
    # Can't hold out more than the eligible pool (short transcripts stay in train).
    k = max(0, min(k, len(eligible)))
    if k == 0:
        return [("train", path, text, extra) for _split, path, text, extra in pairs], 0
    random.Random(seed).shuffle(eligible)
    test_idx = set(eligible[:k])
    return [
        ("test" if i in test_idx else "train", path, text, extra)
        for i, (_split, path, text, extra) in enumerate(pairs)
    ], k


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
    # Each unique audio source (same clip can repeat across splits) → its S3 dest key.
    # A source is either a local file path (str, uploaded) or an
    # ("s3cp", src_bucket, src_key, dest_basename) marker (server-side copied S3→S3,
    # no bytes through the gateway — emitted by _s3_copy_pairs for a same-account merge).
    def _dest_key(src) -> str:
        if isinstance(src, tuple) and src and src[0] == "s3cp":
            return f"{base}/audio/{src[3]}"
        return f"{base}/audio/{os.path.basename(src)}"

    key_of: dict = {}
    for _split, src, _text, _extra in pairs:
        key_of.setdefault(src, _dest_key(src))

    # Re-runs skip clips already in the bucket. Split the rest into local uploads
    # and server-side copies.
    to_upload: list[tuple[str, str]] = []      # (dest_key, local_path)
    to_copy: list[tuple[str, str, str]] = []   # (dest_key, src_bucket, src_key)
    for src, dest_key in key_of.items():
        if dest_key in existing:
            continue
        if isinstance(src, tuple) and src and src[0] == "s3cp":
            to_copy.append((dest_key, src[1], src[2]))
        else:
            to_upload.append((dest_key, src))

    total = len(to_copy) + len(to_upload)
    every = max(1, total // 50) if total else 1  # ~50 progress markers overall

    def _tick(done: int) -> None:
        if progress and (done % every == 0 or done == total):
            progress("upload_s3", done, total)

    # Server-side copies first (fast metadata ops), then local uploads.
    if to_copy:
        bench.s3_copy_many(to_copy, s3_target, max_workers=16, on_done=_tick)
    if to_upload:
        offset = len(to_copy)
        bench.s3_put_files(to_upload, s3_target, max_workers=16, on_done=lambda n: _tick(offset + n))

    # Presign every unique key (local signing, one client), then write the CSV
    # with the carried-through columns (e.g. speaker) after audio/text/split.
    urls = bench.s3_presign_many(list(key_of.values()), expires, s3_target)
    extra_cols = sorted({k for *_rest, extra in pairs for k in extra})
    buf = _io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["audio", transcription_field, "split"] + extra_cols)
    for split, src, text, extra in pairs:
        writer.writerow([urls[key_of[src]], text, split] + [extra.get(c, "") for c in extra_cols])
    meta_key = f"{base}/metadata.csv"
    bench.s3_put_text(meta_key, buf.getvalue(), s3_target)
    bucket = getattr(s3_target, "bucket", "")
    return f"s3://{bucket}/{meta_key}"
