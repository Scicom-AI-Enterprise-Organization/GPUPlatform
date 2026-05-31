"""Parsers for the {audio, transcription} metadata files users upload through
the datasets UI. Supports CSV, JSON, and JSONL.

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

DatasetFormat = str  # "csv" | "json" | "jsonl"

PREVIEW_ROWS = 5

# When the metadata file doesn't use the literal `audio`/`transcription` column
# names, fall back to these common aliases (first match wins).
REQUIRED_FIELDS_FALLBACK = {
    "audio": ["audio", "audio_path", "filename_audio", "audio_url", "url"],
    "transcription": ["transcription", "text", "sentence", "transcript", "label"],
}


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
    raise DatasetParseError(
        f"unsupported file extension {ext or '(none)'} — use .csv, .json, or .jsonl"
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


def _parse_rows_by_format(fmt: DatasetFormat, text: str) -> list[dict[str, Any]]:
    if fmt == "csv":
        return _parse_csv(text)
    if fmt == "jsonl":
        return _parse_jsonl(text)
    return _parse_json(text)


def parse_metadata_bytes(filename: str, body: bytes) -> dict[str, Any]:
    """Parse + validate an uploaded metadata file. Returns
    {format, columns, num_rows, preview, audio_field, transcription_field}.
    Raises DatasetParseError on any problem (bad format, empty, missing columns).
    """
    fmt = detect_format(filename)
    rows = _parse_rows_by_format(fmt, _decode(body))
    if not rows:
        raise DatasetParseError("file is empty — no rows parsed")

    columns = list(rows[0].keys())
    audio_field = _pick_field(columns, REQUIRED_FIELDS_FALLBACK["audio"])
    transcription_field = _pick_field(columns, REQUIRED_FIELDS_FALLBACK["transcription"])

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
    }


def parse_rows(filename: str, body: bytes, limit: int) -> list[dict[str, Any]]:
    """Return up to `limit` rows of a metadata file without column validation —
    used by the detail page's audio preview (the caller already knows the
    field names from the stored Dataset row)."""
    fmt = detect_format(filename)
    return _parse_rows_by_format(fmt, _decode(body))[: max(0, limit)]
