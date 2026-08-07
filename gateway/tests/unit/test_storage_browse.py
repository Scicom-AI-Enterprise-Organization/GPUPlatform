"""Unit tests for the storage file viewer's path, paging + content-type handling.

Pure helpers and a tmp_path directory — no S3, no DB. Three things worth pinning:
  * `_safe_rel` / `_local_path` are the scope boundary (a viewer must not escape
    the storage's own prefix into a sibling tenant's objects, and a symlink
    inside a local root must not reach `/etc`);
  * type resolution must NOT trust S3's stored Content-Type, which is arbitrary
    caller-supplied metadata (this platform's bucket stamps .jsonl objects
    `application/jsonl`) — that bug made a text file un-previewable;
  * every listing stays page-bounded: a directory too big to sort must still
    return a page, in filesystem order, with a note saying so.
"""
import os

import pytest
from fastapi import HTTPException

from gateway import storage_api
from gateway.storage_api import (
    StorageEntry,
    _abs_key,
    _is_textual,
    _local_browse,
    _local_offset,
    _local_path,
    _local_scan_dir,
    _looks_textual,
    _media_type_for,
    _safe_rel,
    _sort_entries,
    _sorted_page,
)


@pytest.mark.parametrize("raw,expected", [
    ("", ""),
    (None, ""),
    ("/", ""),
    ("a/b.json", "a/b.json"),
    ("/a/b.json", "a/b.json"),
    ("a//b/", "a/b"),
    ("./a/./b", "a/b"),
    ("  a/b  ", "a/b"),
])
def test_safe_rel_normalizes(raw, expected):
    assert _safe_rel(raw) == expected


@pytest.mark.parametrize("raw", ["..", "../etc/passwd", "a/../../b", "a/.."])
def test_safe_rel_rejects_traversal(raw):
    with pytest.raises(HTTPException) as e:
        _safe_rel(raw)
    assert e.value.status_code == 400


def test_abs_key_scopes_to_the_storage_prefix():
    assert _abs_key("tenant/x", "a/b.json") == "tenant/x/a/b.json"
    assert _abs_key("", "a/b.json") == "a/b.json"


@pytest.mark.parametrize("name,expected", [
    ("x.jsonl", "application/x-ndjson"),
    ("x.log", "text/plain; charset=utf-8"),
    ("x.json", "application/json"),
    ("x.wav", "audio/x-wav"),
    ("x.png", "image/png"),
])
def test_media_type_from_extension(name, expected):
    assert _media_type_for(name) == expected


@pytest.mark.parametrize("name", [".persist_marker", "metadata", "README", "x.zzz"])
def test_media_type_none_for_unknown_extension(name):
    # None is the signal that sniffing may re-type the object.
    assert _media_type_for(name) is None


def test_textual_classification():
    assert _is_textual("text/plain; charset=utf-8")
    assert _is_textual("application/json")
    assert _is_textual("application/x-ndjson")
    assert _is_textual("application/ld+json")
    assert not _is_textual("audio/x-wav")
    assert not _is_textual("application/octet-stream")
    # The exact type that broke the preview when S3's stored value was trusted.
    assert not _is_textual("application/jsonl")


# ---------- local backing: confined to the configured folder --------------


@pytest.fixture()
def local_root(tmp_path):
    """A local storage root: 3 files + a subdir, plus a sibling directory
    OUTSIDE the root and a symlink pointing at it."""
    root = tmp_path / "store"
    (root / "sub").mkdir(parents=True)
    for n in ("b.txt", "a.txt", "c.log"):
        (root / n).write_text(n)
    (root / "sub" / "nested.txt").write_text("nested")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not yours")
    os.symlink(outside / "secret.txt", root / "escape.txt")
    os.symlink(outside, root / "escape-dir")
    return os.path.realpath(str(root))


def test_local_browse_lists_only_that_folder(local_root):
    entries, next_token, note = _local_browse(local_root, "", 0, 100, "")
    names = [e.name for e in entries]
    # Exactly the root's own children — folders first, then files, name-sorted.
    # The two symlinks leading OUT of the root are hidden, not merely refused on
    # open: the listing is the configured folder, nothing reachable from it.
    assert names == ["sub", "a.txt", "b.txt", "c.log"]
    assert next_token is None and note is None


@pytest.mark.parametrize("rel", ["escape.txt", "escape-dir", "escape-dir/secret.txt"])
def test_local_path_refuses_symlinks_out_of_the_root(local_root, rel):
    # `_safe_rel` can't catch this — the path has no `..`. realpath is what
    # confines the viewer to the configured folder.
    with pytest.raises(HTTPException) as e:
        _local_path(local_root, rel)
    assert e.value.status_code == 400


def test_local_path_allows_paths_inside_the_root(local_root):
    assert _local_path(local_root, "sub/nested.txt") == os.path.join(local_root, "sub", "nested.txt")
    assert _local_path(local_root, "") == local_root


def test_local_browse_subdir_and_prefix_filter(local_root):
    entries, _, _ = _local_browse(local_root, "sub", 0, 100, "")
    assert [e.name for e in entries] == ["nested.txt"]
    assert entries[0].path == "sub/nested.txt"  # paths stay relative to the root
    hits, _, _ = _local_browse(local_root, "", 0, 100, "a")
    assert [e.name for e in hits] == ["a.txt"]


def test_local_browse_pages_with_an_offset_token(local_root):
    first, token, _ = _local_browse(local_root, "", 0, 2, "")
    assert len(first) == 2 and token == "2"
    second, token2, _ = _local_browse(local_root, "", _local_offset(token), 2, "")
    assert len(second) == 2
    # Pages don't overlap and together cover the folder; the last page ends the walk.
    assert not ({e.name for e in first} & {e.name for e in second})
    assert {e.name for e in first} | {e.name for e in second} == {
        "sub", "a.txt", "b.txt", "c.log",
    }
    assert token2 is None


def test_a_directory_too_big_to_sort_still_returns_a_page(local_root, monkeypatch):
    """The 1M-file case: no whole-directory list, no sort — a page, in readdir
    order, and a note that says so."""
    monkeypatch.setattr(storage_api, "LOCAL_SORT_MAX", 2)
    entries, token, note = _local_browse(local_root, "", 0, 3, "")
    assert len(entries) == 3
    assert token == "3"  # more to come
    assert note and "not sorted" in note


def test_local_offset_rejects_a_junk_token():
    assert _local_offset(None) == 0 and _local_offset("") == 0
    with pytest.raises(HTTPException) as e:
        _local_offset("not-a-number")
    assert e.value.status_code == 400


# ---------- sorting by size / modified ------------------------------------
# Neither backend can push these down (S3 LIST is lexicographic-only, readdir is
# unordered), so they are computed over a capped scan. What must hold: folders
# stay first and name-ordered (an S3 CommonPrefix has neither field), a missing
# value is unranked rather than "smallest", and a scan that hit the cap says so.


def _e(name, kind="file", size=None, modified=None):
    return StorageEntry(name=name, path=name, kind=kind, size=size, modified=modified)


@pytest.fixture()
def mixed():
    return [
        _e("zed", kind="folder"),
        _e("alpha", kind="folder"),
        _e("big.bin", size=900, modified="2026-08-01T00:00:00+00:00"),
        _e("small.txt", size=12, modified="2026-08-07T00:00:00+00:00"),
        _e("mid.wav", size=300, modified="2026-08-05T00:00:00+00:00"),
    ]


def test_sort_by_size(mixed):
    desc = _sort_entries(mixed, "size", "desc")
    assert [e.name for e in desc] == ["alpha", "zed", "big.bin", "mid.wav", "small.txt"]
    asc = _sort_entries(mixed, "size", "asc")
    assert [e.name for e in asc] == ["alpha", "zed", "small.txt", "mid.wav", "big.bin"]


def test_sort_by_modified(mixed):
    desc = _sort_entries(mixed, "modified", "desc")
    assert [e.name for e in desc] == ["alpha", "zed", "small.txt", "mid.wav", "big.bin"]


def test_folders_lead_and_never_rank_by_size_or_time(mixed):
    """An S3 folder is a CommonPrefix — no size, no LastModified. Letting one
    participate would also make local (which has an mtime) behave differently
    for the same click."""
    for order in ("asc", "desc"):
        for field in ("size", "modified"):
            out = _sort_entries(mixed, field, order)
            assert [e.name for e in out[:2]] == ["alpha", "zed"]


def test_sort_by_name_both_ways(mixed):
    assert [e.name for e in _sort_entries(mixed, "name", "asc")] == [
        "alpha", "zed", "big.bin", "mid.wav", "small.txt",
    ]
    assert [e.name for e in _sort_entries(mixed, "name", "desc")] == [
        "zed", "alpha", "small.txt", "mid.wav", "big.bin",
    ]


def test_a_file_with_no_value_is_unranked_not_smallest():
    entries = [_e("known", size=5), _e("unknown"), _e("bigger", size=50)]
    for order in ("asc", "desc"):
        out = _sort_entries(entries, "size", order)
        assert out[-1].name == "unknown", order


def test_equal_values_tie_break_alphabetically():
    entries = [_e("c.txt", size=7), _e("a.txt", size=7), _e("b.txt", size=7)]
    assert [e.name for e in _sort_entries(entries, "size", "desc")] == [
        "a.txt", "b.txt", "c.txt",
    ]


def test_sorted_page_cuts_pages_and_ends_the_walk(mixed):
    first, token, note = _sorted_page(mixed, False, "size", "desc", 0, 3)
    assert [e.name for e in first] == ["alpha", "zed", "big.bin"]
    assert token == "3" and note is None
    second, token2, _ = _sorted_page(mixed, False, "size", "desc", _local_offset(token), 3)
    assert [e.name for e in second] == ["mid.wav", "small.txt"]
    assert token2 is None


def test_a_truncated_scan_says_the_ranking_is_partial(mixed):
    """The trap this note exists for: "biggest file" over the first 20k keys of a
    million-key directory is not the biggest file."""
    _, _, note = _sorted_page(mixed, True, "size", "desc", 0, 3)
    assert note and "not the whole folder" in note


def test_local_scan_dir_materializes_stats_and_caps(local_root):
    entries, truncated = _local_scan_dir(local_root, "", "", 100)
    assert not truncated
    # The escaping symlinks stay hidden here too, and every file carries the
    # stat that a size/mtime ranking needs.
    assert {e.name for e in entries} == {"sub", "a.txt", "b.txt", "c.log"}
    assert all(e.size is not None for e in entries if e.kind == "file")
    capped, truncated = _local_scan_dir(local_root, "", "", 2)
    assert len(capped) == 2 and truncated


def test_looks_textual_sniff():
    assert _looks_textual(b"run@2026-07-21 hello\n")
    assert _looks_textual("héllo wörld".encode())
    assert not _looks_textual(b"RIFF\x00\x00\x00WAVEfmt ")
    assert not _looks_textual(b"\xff\xd8\xff\xe0\x10JFIF")
    # A head-read may split a multi-byte codepoint at the tail — still text.
    assert _looks_textual("hello wörld".encode()[:-1])
    # …but garbage early in the buffer is not.
    assert not _looks_textual(b"\xff\xfe" + b"a" * 100)
