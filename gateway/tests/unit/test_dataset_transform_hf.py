"""Unit tests for the kind=hf transform's two additions:

  * subset selection — resolving declared config/split names into the labels to
    keep and the file globs to fetch;
  * embedded audio — a HuggingFace `Audio` column (the clip bytes inline in the
    parquet, not a path to a file in the repo) materialised to disk.

Plus the metadata-CSV column-collision guard, which is what stops a carried-
through column silently overwriting the transcription it collides with.

All pure/in-process — a tiny parquet is written on the fly, no network.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from gateway import dataset_transform as dt

pytest.importorskip("pyarrow")
pytest.importorskip("pandas")


README = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
- config_name: synthetic
  data_files:
  - split: train
    path: synthetic/train-*
  - split: test
    path: synthetic/test-*
- config_name: synthetic_podcast
  data_files:
  - split: train
    path: synthetic_podcast/train-*
---

# a dataset
"""

# Minimal but real containers, so `_audio_ext` sniffs them the way it would a
# genuine clip.
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 24
FLAC = b"fLaC" + b"\x00" * 16


def _write_parquet(path: Path, rows: list[dict], *, embedded: bool) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if embedded:
        schema = pa.schema([
            ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
            ("text", pa.string()),
            ("transcription", pa.string()),
            ("speaker", pa.string()),
        ])
        pq.write_table(pa.Table.from_pandas(df, schema=schema), path)
    else:
        pq.write_table(pa.Table.from_pandas(df), path)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A work dir shaped like a snapshot_download of a multi-config repo."""
    (tmp_path / "README.md").write_text(README)
    _write_parquet(
        tmp_path / "synthetic" / "train-00000-of-00001.parquet",
        [
            {"audio": {"bytes": WAV, "path": "gen00001.wav"}, "text": "hello",
             "transcription": "helo", "speaker": "spk-a"},
            {"audio": {"bytes": FLAC, "path": "gen00002.flac"}, "text": "world",
             "transcription": "word", "speaker": "spk-b"},
        ],
        embedded=True,
    )
    # A DIFFERENT config whose shard has the same file stem AND whose rows carry
    # the same inner `path` values — the basename-collision trap.
    _write_parquet(
        tmp_path / "synthetic_podcast" / "train-00000-of-00001.parquet",
        [
            {"audio": {"bytes": WAV, "path": "gen00001.wav"}, "text": "podcast",
             "transcription": "podcst", "speaker": "spk-c"},
        ],
        embedded=True,
    )
    _write_parquet(
        tmp_path / "data" / "train-00000-of-00001.parquet",
        [{"audio": {"bytes": WAV, "path": "gen00009.wav"}, "text": "unwanted",
          "transcription": "x", "speaker": "spk-z"}],
        embedded=True,
    )
    return tmp_path


# ---------------------------------------------------------------- subsets ----


def test_declared_entries_labels_by_config_and_split(repo: Path):
    entries = dt._declared_entries(str(repo))
    assert {e["label"] for e in entries} == {
        "default/train", "synthetic/train", "synthetic/test", "synthetic_podcast/train",
    }


def test_resolve_subsets_by_label_and_by_bare_config(repo: Path):
    entries = dt._declared_entries(str(repo))

    keep, globs = dt._resolve_hf_subsets(entries, ["synthetic/train"])
    assert keep == {"synthetic/train"}
    assert globs == ["synthetic/train-*"]

    # A bare config name takes every split of it.
    keep, globs = dt._resolve_hf_subsets(entries, ["synthetic"])
    assert keep == {"synthetic/train", "synthetic/test"}
    assert sorted(globs) == ["synthetic/test-*", "synthetic/train-*"]


def test_unknown_subset_raises_rather_than_selecting_nothing(repo: Path):
    """A typo must fail loudly: selecting nothing would materialise an empty
    dataset and selecting everything would pull the configs being excluded."""
    entries = dt._declared_entries(str(repo))
    with pytest.raises(ValueError) as e:
        dt._resolve_hf_subsets(entries, ["synthetic/traon"])
    assert "synthetic/train" in str(e.value)  # lists what IS available


def test_subset_matches_is_the_one_rule_everything_shares():
    """The transform, the /splits picker and the row preview all resolve a stored
    scope through this — they must never disagree about what a name selects."""
    assert dt.subset_matches("synthetic/train", "synthetic/train", "synthetic", "train")
    assert dt.subset_matches("synthetic", "synthetic/train", "synthetic", "train")
    # A single-config repo is labelled by split alone; the config/split spelling
    # still has to resolve.
    assert dt.subset_matches("default/train", "train", "default", "train")
    # Not a prefix match, not a substring match.
    assert not dt.subset_matches("synth", "synthetic/train", "synthetic", "train")
    assert not dt.subset_matches("synthetic_podcast", "synthetic/train", "synthetic", "train")
    assert not dt.subset_matches("", "synthetic/train", "synthetic", "train")


def test_no_declared_configs_is_none(tmp_path: Path):
    assert dt._declared_entries(str(tmp_path)) is None       # no README at all
    (tmp_path / "README.md").write_text("# plain repo\n")
    assert dt._declared_entries(str(tmp_path)) is None       # no front-matter


# --------------------------------------------------------- embedded audio ----


def test_embedded_audio_is_materialised_and_selection_is_honoured(repo: Path):
    pairs = dt._build_pairs(str(repo), "audio", "text", None, {"synthetic/train"})
    assert len(pairs) == 2

    splits = {p[0] for p in pairs}
    assert splits == {"synthetic/train"}          # the other configs are absent
    assert [p[2] for p in pairs] == ["hello", "world"]

    for _split, path, _text, _extra in pairs:
        assert Path(path).is_file() and Path(path).stat().st_size > 0

    # Extension comes from the clip's own name when it's a known audio suffix.
    assert Path(pairs[0][1]).suffix == ".wav"
    assert Path(pairs[1][1]).suffix == ".flac"

    # Non-audio columns are carried through as `extra`.
    assert pairs[0][3]["speaker"] == "spk-a"
    assert pairs[0][3]["transcription"] == "helo"


def test_clip_basenames_are_unique_across_configs(repo: Path):
    """⚠ `_materialise_s3` keys S3 objects by BASENAME. Two configs whose shards
    share a stem and whose rows share an inner `path` would collapse onto one
    object — so the subset label has to be part of the filename."""
    pairs = dt._build_pairs(
        str(repo), "audio", "text", None, {"synthetic/train", "synthetic_podcast/train"},
    )
    assert len(pairs) == 3
    names = [Path(p[1]).name for p in pairs]
    assert len(set(names)) == 3, names


def test_rerun_reuses_already_written_clips(repo: Path):
    first = dt._build_pairs(str(repo), "audio", "text", None, {"synthetic/train"})
    stamps = {p[1]: Path(p[1]).stat().st_mtime_ns for p in first}
    second = dt._build_pairs(str(repo), "audio", "text", None, {"synthetic/train"})
    assert [p[1] for p in second] == [p[1] for p in first]
    assert all(Path(p).stat().st_mtime_ns == stamps[p] for p in stamps)


def test_plain_path_column_still_works(tmp_path: Path):
    """The non-embedded shape (a metadata table referencing extracted files) must
    keep behaving exactly as before."""
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "a.wav").write_bytes(WAV)
    _write_parquet(
        tmp_path / "metadata.parquet",
        [{"audio": "audio/a.wav", "text": "hi", "transcription": "hi", "speaker": "s"}],
        embedded=False,
    )
    pairs = dt._build_pairs(str(tmp_path), "audio", "text")
    assert len(pairs) == 1
    assert Path(pairs[0][1]).name == "a.wav"
    assert pairs[0][2] == "hi"


def test_audio_ext_sniffs_when_the_name_is_useless():
    assert dt._audio_ext(WAV, None) == ".wav"
    assert dt._audio_ext(FLAC, "") == ".flac"
    assert dt._audio_ext(b"OggS...", "clip") == ".ogg"
    assert dt._audio_ext(b"\x00\x00\x00\x18ftypM4A ", "clip") == ".m4a"
    # A usable suffix on the struct's own path wins over sniffing.
    assert dt._audio_ext(WAV, "gen1.flac") == ".flac"


# ------------------------------------------------ metadata column collision ---


def test_materialise_s3_drops_extras_that_collide_with_its_own_columns(monkeypatch, tmp_path):
    """A carried-through column named like one of the three the writer owns would
    emit a DUPLICATE header; every CSV reader keeps the last, so the passenger
    value would silently replace the real transcription."""
    written: dict[str, bytes] = {}

    class _FakeBench:
        @staticmethod
        def s3_list(prefix, target):
            return []

        @staticmethod
        def s3_copy_many(items, target, max_workers=16, on_done=None):
            return None

        @staticmethod
        def s3_put_files(items, target, max_workers=16, on_done=None):
            return None

        @staticmethod
        def s3_presign_many(keys, expires, target):
            return {k: f"https://example.test/{k}" for k in keys}

        @staticmethod
        def s3_put_text(key, text, target):
            written[key] = text.encode("utf-8")

    # `_materialise_s3` does `from . import bench` at call time, which resolves the
    # attribute on the PACKAGE once gateway.bench is imported — so patching
    # sys.modules alone leaves the real client in place (and it only shows up when
    # another test imported bench first, i.e. suite-order-dependently).
    import sys

    import gateway as gateway_pkg

    monkeypatch.setattr(gateway_pkg, "bench", _FakeBench, raising=False)
    monkeypatch.setitem(sys.modules, "gateway.bench", _FakeBench)

    clip = tmp_path / "clip.wav"
    clip.write_bytes(WAV)
    # `transcription` here is a passenger column while the OUTPUT's transcription
    # column is also `transcription` — exactly the merge case.
    pairs = [("train", str(clip), "the real label",
              {"transcription": "an asr readback", "speaker": "spk", "split": "bogus"})]

    uri = dt._materialise_s3(pairs, "transcription", object(), "", "ds-test")
    assert uri.endswith("metadata.csv")

    body = next(v for k, v in written.items() if k.endswith("metadata.csv"))
    rows = list(csv.DictReader(io.StringIO(body.decode())))
    reader = csv.reader(io.StringIO(body.decode()))
    header = next(reader)

    assert header.count("transcription") == 1
    assert header.count("split") == 1
    assert rows[0]["transcription"] == "the real label"
    assert rows[0]["split"] == "train"
    assert rows[0]["speaker"] == "spk"          # a non-colliding extra survives
