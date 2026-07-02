"""Parsers for the metadata files users upload through the datasets UI —
{audio, transcription} rows for audio datasets, or a `messages` column for chat
datasets. Supports CSV, JSON, JSONL, and Parquet.

Python port of the AutoTrain web app's `src/lib/dataset-metadata.ts`. Callers
hand us the raw bytes + the filename so we can detect the format and validate
that the expected audio + transcription columns are present.
"""
from __future__ import annotations

import csv
import io
import json
import os
from typing import Any

DatasetFormat = str  # "csv" | "json" | "jsonl" | "parquet"

PREVIEW_ROWS = 5

# When the metadata file doesn't use the literal `audio`/`transcription` column
# names, fall back to these common aliases (first match wins).
REQUIRED_FIELDS_FALLBACK = {
    "audio": ["audio", "audio_path", "filename_audio", "audio_url", "url"],
    "transcription": ["transcription", "text", "sentence", "transcript", "label"],
}

# Optional speaker column — detected so multi-speaker datasets map the speaker
# picker + TTS packer to the right column (None when absent → single speaker).
SPEAKER_FIELD_FALLBACK = ["speaker", "speaker_id", "speaker_name", "spk", "spk_id", "speaker_label"]


class DatasetParseError(Exception):
    """Raised on an unparseable / invalid metadata file. The API maps this to a
    400 so the user sees the message inline."""


def detect_format(filename: str) -> DatasetFormat:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext == ".csv":
        return "csv"
    if ext in (".jsonl", ".ndjson"):
        return "jsonl"
    if ext == ".json":
        return "json"
    if ext == ".parquet":
        return "parquet"
    raise DatasetParseError(
        f"unsupported file extension {ext or '(none)'} — use .csv, .json, .jsonl, or .parquet"
    )


def _pick_field(columns: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _decode(body: bytes) -> str:
    return body.decode("utf-8", "replace")


def _parse_json(text: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except Exception as e:
        raise DatasetParseError(f"invalid JSON: {e}") from e
    if not isinstance(parsed, list):
        raise DatasetParseError("JSON must be an array of objects")
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(parsed):
        if not isinstance(row, dict):
            raise DatasetParseError(f"row {i} is not an object")
        rows.append(row)
    return rows


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, line in enumerate(text.splitlines()):
        trimmed = line.strip()
        if not trimmed:
            continue
        try:
            parsed = json.loads(trimmed)
        except Exception as e:
            raise DatasetParseError(f"JSONL line {i + 1}: {e}") from e
        if not isinstance(parsed, dict):
            raise DatasetParseError(f"JSONL line {i + 1} is not an object")
        out.append(parsed)
    return out


def _parse_csv(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return []
    return [dict(row) for row in reader]


def _parse_parquet(body: bytes) -> list[dict[str, Any]]:
    """Read a parquet file (binary) into a list of row dicts. Nested columns
    (e.g. a `messages` array of {role, content}) come back as native Python
    lists/dicts via `to_pylist()`."""
    try:
        import pyarrow.parquet as pq  # lazy — only chat/parquet uploads need it
    except Exception as e:  # noqa: BLE001
        raise DatasetParseError(f"parquet support unavailable: {e}") from e
    try:
        table = pq.read_table(io.BytesIO(body))
    except Exception as e:  # noqa: BLE001
        raise DatasetParseError(f"invalid parquet file: {e}") from e
    return table.to_pylist()


def _parse_rows_by_format(fmt: DatasetFormat, text: str) -> list[dict[str, Any]]:
    if fmt == "csv":
        return _parse_csv(text)
    if fmt == "jsonl":
        return _parse_jsonl(text)
    return _parse_json(text)


def parse_rows_any(filename: str, body: bytes, limit: int) -> list[dict[str, Any]]:
    """Like `parse_rows`, but handles parquet (binary) in addition to the text
    formats. Returns up to `limit` rows without column validation."""
    fmt = detect_format(filename)
    rows = _parse_parquet(body) if fmt == "parquet" else _parse_rows_by_format(fmt, _decode(body))
    return rows[: max(0, limit)]


def parse_metadata_bytes(
    filename: str, body: bytes, messages_field: str | None = None
) -> dict[str, Any]:
    """Parse + validate an uploaded metadata file.

    Audio datasets (``messages_field`` unset) return
    {format, columns, num_rows, preview, audio_field, transcription_field} and
    require an audio + transcription column. Chat datasets (``messages_field``
    set) instead validate that the messages column exists and return
    {format, columns, num_rows, preview, messages_field} — no audio required.
    Raises DatasetParseError on any problem (bad format, empty, missing columns).
    """
    fmt = detect_format(filename)
    rows = _parse_parquet(body) if fmt == "parquet" else _parse_rows_by_format(fmt, _decode(body))
    if not rows:
        raise DatasetParseError("file is empty — no rows parsed")

    columns = list(rows[0].keys())

    # Chat upload: validate the messages column, skip the audio/transcription
    # requirement entirely.
    if messages_field:
        if messages_field not in columns:
            raise DatasetParseError(
                f"messages column '{messages_field}' not found. Columns: "
                + ", ".join(columns)
            )
        return {
            "format": fmt,
            "columns": columns,
            "num_rows": len(rows),
            "preview": rows[:PREVIEW_ROWS],
            "messages_field": messages_field,
        }

    audio_field = _pick_field(columns, REQUIRED_FIELDS_FALLBACK["audio"])
    transcription_field = _pick_field(columns, REQUIRED_FIELDS_FALLBACK["transcription"])
    speaker_field = _pick_field(columns, SPEAKER_FIELD_FALLBACK)

    if not audio_field:
        raise DatasetParseError(
            "no audio column found. Expected one of: "
            + ", ".join(REQUIRED_FIELDS_FALLBACK["audio"])
        )
    if not transcription_field:
        raise DatasetParseError(
            "no transcription column found. Expected one of: "
            + ", ".join(REQUIRED_FIELDS_FALLBACK["transcription"])
        )

    return {
        "format": fmt,
        "columns": columns,
        "num_rows": len(rows),
        "preview": rows[:PREVIEW_ROWS],
        "audio_field": audio_field,
        "transcription_field": transcription_field,
        "speaker_field": speaker_field,
    }


def detect_fields(columns: list[str]) -> dict[str, Any]:
    """Best-effort column → role mapping for a metadata table (no validation, no
    raising). Used to backfill a dataset row's field mappings from the file's
    actual columns. Any role with no matching column comes back None."""
    return {
        "audio_field": _pick_field(columns, REQUIRED_FIELDS_FALLBACK["audio"]),
        "transcription_field": _pick_field(columns, REQUIRED_FIELDS_FALLBACK["transcription"]),
        "speaker_field": _pick_field(columns, SPEAKER_FIELD_FALLBACK),
    }


def parse_rows(filename: str, body: bytes, limit: int) -> list[dict[str, Any]]:
    """Return up to `limit` rows of a metadata file without column validation —
    used by the detail page's audio preview (the caller already knows the
    field names from the stored Dataset row)."""
    fmt = detect_format(filename)
    return _parse_rows_by_format(fmt, _decode(body))[: max(0, limit)]
