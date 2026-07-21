"""storage_purge.categorize — the storage-cleanup classifier (orphan + aged),
the pure core the manual scan/confirm flow re-runs identically."""
from gateway.storage_purge import categorize


def _objs():
    return [
        # live dataset — kept
        {"key": "datasets/ds-live/metadata.csv", "size": 100, "modified": "2026-07-20T00:00:00+00:00"},
        # deleted dataset — ORPHAN
        {"key": "datasets/ds-gone/a.parquet", "size": 500, "modified": "2026-07-19T00:00:00+00:00"},
        {"key": "datasets/ds-gone/b.parquet", "size": 500, "modified": "2026-07-19T00:00:00+00:00"},
        # old benchmark of a LIVE bench — AGED (ephemeral + older than cutoff)
        {"key": "benchmarks/bm-live/out.json", "size": 200, "modified": "2026-01-01T00:00:00+00:00"},
        # recent benchmark of a live bench — kept
        {"key": "benchmarks/bm-fresh/out.json", "size": 300, "modified": "2026-07-21T00:00:00+00:00"},
        # unrecognized top-level folder — kept for safety
        {"key": "random-stuff/x.bin", "size": 999, "modified": "2020-01-01T00:00:00+00:00"},
    ]


def _classify(**over):
    kw = dict(
        base="",
        live_ids={"dataset": {"ds-live"}, "benchmark": {"bm-live", "bm-fresh"}},
        repo_prefixes=[],
        cutoff_iso="2026-06-01T00:00:00+00:00",
    )
    kw.update(over)
    return categorize(_objs(), **kw)


def _by_prefix(res):
    return {g["prefix"]: g for g in res["groups"]}


def test_orphan_and_aged_flagged_kept_protected():
    res = _classify()
    g = _by_prefix(res)
    assert g["datasets/ds-gone/"]["category"] == "orphan" and g["datasets/ds-gone/"]["purgeable"]
    assert g["datasets/ds-gone/"]["bytes"] == 1000 and g["datasets/ds-gone/"]["objects"] == 2
    assert g["benchmarks/bm-live/"]["category"] == "aged" and g["benchmarks/bm-live/"]["purgeable"]
    assert g["datasets/ds-live/"]["category"] == "kept" and not g["datasets/ds-live/"]["purgeable"]
    assert g["benchmarks/bm-fresh/"]["category"] == "kept"
    assert g["random-stuff/"]["category"] == "kept"  # unrecognized never deleted
    # totals + reclaimable
    assert res["total_bytes"] == 100 + 1000 + 200 + 300 + 999
    assert res["reclaimable_bytes"] == 1000 + 200
    assert res["reclaimable_objects"] == 3


def test_no_age_rule_keeps_aged_bench():
    res = _classify(cutoff_iso=None)
    g = _by_prefix(res)
    assert g["benchmarks/bm-live/"]["category"] == "kept"  # only orphans without an age rule
    assert res["reclaimable_bytes"] == 1000


def test_catalog_repo_prefix_protects_objects():
    # ds-gone is a deleted Dataset but its prefix is a LIVE catalog repo → kept.
    res = _classify(repo_prefixes=["datasets/ds-gone"])
    g = _by_prefix(res)
    assert g["datasets/ds-gone/"]["category"] == "kept"
    assert g["datasets/ds-gone/"]["purgeable"] is False


def test_nonempty_base_prefix():
    objs = [{"key": "myroot/datasets/ds-gone/a", "size": 10, "modified": "2026-07-19T00:00:00+00:00"}]
    res = categorize(objs, base="myroot", live_ids={"dataset": set()}, repo_prefixes=[], cutoff_iso=None)
    g = _by_prefix(res)
    assert "myroot/datasets/ds-gone/" in g
    assert g["myroot/datasets/ds-gone/"]["category"] == "orphan"
