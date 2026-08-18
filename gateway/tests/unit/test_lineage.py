"""Lineage tree shaping — the pure parts, no DB.

The walk itself is exercised against the live gateway; these pin the properties that
would silently corrupt an audit trail: a cycle must not recurse forever, roots must be
the leaves and nothing else, and `flatten` must not lose or duplicate nodes.
"""
from gateway.lineage import flatten, record, roots


def _node(i, sources=None, **kw):
    n = {"id": i, "name": f"name-{i}", "kind": "s3", "num_rows": 1, **kw}
    if sources:
        n["sources"] = sources
        n.setdefault("op", "merge")
    return n


def test_record_drops_empty_sources_and_none_params():
    r = record("merge", ["ds-a", "", None, "ds-b"], target="s3", extra=None)
    assert r["sources"] == ["ds-a", "ds-b"]
    assert r["params"] == {"target": "s3"}      # `extra=None` must not be stored
    assert r["op"] == "merge"


def test_roots_are_the_leaves_only():
    tree = _node("ds-top", [
        _node("ds-mid", [_node("ds-leaf1"), _node("ds-leaf2")]),
        _node("ds-leaf3"),
    ])
    assert [r["id"] for r in roots(tree)] == ["ds-leaf1", "ds-leaf2", "ds-leaf3"]


def test_flatten_visits_every_node_once():
    tree = _node("ds-top", [_node("ds-mid", [_node("ds-leaf")])])
    ids = [n["id"] for n in flatten(tree)]
    assert ids == ["ds-top", "ds-mid", "ds-leaf"]


def test_a_dataset_reachable_by_two_paths_appears_twice_in_flat():
    """A diamond is legal — the same source merged in twice. flatten() reports both
    occurrences; de-duplication is the caller's job (the run endpoint keys by id), and
    doing it here would hide that a corpus was included twice."""
    shared = _node("ds-shared")
    tree = _node("ds-top", [_node("ds-a", [shared]), _node("ds-b", [shared])])
    assert [n["id"] for n in flatten(tree)].count("ds-shared") == 2
    assert [r["id"] for r in roots(tree)] == ["ds-shared", "ds-shared"]


def test_cycle_and_missing_markers_terminate():
    """Markers set by resolve() must be leaves, so a corrupted graph still renders."""
    tree = _node("ds-top", [{"id": "ds-top", "cycle": True}, {"id": "ds-gone", "missing": True}])
    assert [r["id"] for r in roots(tree)] == ["ds-top", "ds-gone"]
    assert len(flatten(tree)) == 3


def test_empty_inputs_are_safe():
    assert flatten(None) == [] and roots(None) == []
    assert roots(_node("ds-solo")) == [{"id": "ds-solo", "name": "name-ds-solo",
                                        "kind": "s3", "hf_repo": None, "hf_subsets": None}]
