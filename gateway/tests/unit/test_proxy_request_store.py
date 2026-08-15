"""Unit tests for PROXY_REQUEST_STORE — which proxied requests get a Postgres row,
and the invariant that the Prometheus metrics do NOT depend on that answer."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest


@pytest.fixture
def px(monkeypatch):
    """proxy_api under a given PROXY_REQUEST_STORE, with stats_writer's enqueue
    captured instead of executed.

    ⚠ The mode is patched onto the module, NOT applied by reloading it: proxy_api
    declares SQLAlchemy models at import time, so `importlib.reload` raises
    "Table 'proxy_endpoints' is already defined for this MetaData instance"."""
    import gateway.proxy_api as p
    import gateway.stats_writer as sw

    def _load(mode="all", ratio="0.05"):
        resolved = mode if mode in ("all", "sampled", "errors", "off") else "all"
        monkeypatch.setattr(p, "REQUEST_STORE", resolved)
        monkeypatch.setattr(p, "REQUEST_STORE_RATIO", float(ratio))
        p._ADMITTED.clear()
        sent: list[dict] = []
        monkeypatch.setattr(sw, "_enqueue", sent.append)
        return p, sent

    yield _load
    p._ADMITTED.clear()


class _User:
    id = 7
    username = "admin"


def _admit(p, rid="pxr-1", stream=False):
    asyncio.run(p._admit_request(rid, "proxy-1", _User(), "m1", stream,
                                 endpoint_name="ep"))


def _finish(p, rid="pxr-1", **kw):
    kw.setdefault("status_code", 200)
    kw.setdefault("latency_ms", 12)
    asyncio.run(p._finish(rid, kw.pop("status", "completed"), **kw))


def test_mode_all_inserts_at_admission_and_updates_at_finish(px):
    p, sent = px("all")
    _admit(p)
    assert [i["kind"] for i in sent] == ["proxy_insert"]
    assert sent[0]["status"] == "queued" and sent[0]["endpoint_id"] == "proxy-1"
    assert sent[0]["owner_id"] == 7 and sent[0]["model"] == "m1"
    sent.clear()
    _finish(p)
    assert [i["kind"] for i in sent] == ["proxy"]     # an UPDATE of that row
    assert sent[0]["status"] == "completed"


def test_mode_off_writes_no_row_but_still_reports_the_metric(px):
    """The regression this guards: the per-proxy Prometheus series used to be a side
    effect of the row UPDATE, so turning storage off would have silently taken every
    latency/TTFT/tok-s panel down with it."""
    p, sent = px("off")
    _admit(p)
    assert sent == []                                  # nothing at admission
    _finish(p, ct=5)
    assert [i["kind"] for i in sent] == ["proxy_metric"]   # metric only, no session
    assert sent[0]["endpoint_id"] == "proxy-1" and sent[0]["model"] == "m1"


def test_mode_errors_writes_one_complete_row_only_when_it_went_wrong(px):
    p, sent = px("errors")
    _admit(p, "pxr-ok")
    _finish(p, "pxr-ok")
    assert [i["kind"] for i in sent] == ["proxy_metric"]    # success → no row
    sent.clear()

    _admit(p, "pxr-bad")
    _finish(p, "pxr-bad", status="failed", status_code=502, error="upstream died")
    kinds = [i["kind"] for i in sent]
    assert kinds == ["proxy_insert", "proxy_metric"]
    row = sent[0]
    # Written ONCE, complete — not insert-then-update.
    assert row["status"] == "failed" and row["status_code"] == 502
    assert row["error"] == "upstream died" and row["completed_at"] is not None
    assert row["created_at"] is not None


def test_mode_errors_also_keeps_4xx_and_blocked(px):
    p, sent = px("errors")
    for rid, status, code in (("a", "completed", 429), ("b", "blocked", 200),
                              ("c", "cancelled", 499)):
        _admit(p, rid)
        sent.clear()
        _finish(p, rid, status=status, status_code=code)
        assert "proxy_insert" in [i["kind"] for i in sent], (rid, status, code)


def test_sampled_mode_is_deterministic_and_consistent_end_to_end(px):
    """A request sampled IN at admission must not be sampled OUT at completion —
    that would strand a permanently `queued` row."""
    p, _ = px("sampled", ratio="0.5")
    ids = [f"pxr-{i:04d}" for i in range(400)]
    chosen = {i for i in ids if p._store_row(i)}
    assert 0 < len(chosen) < len(ids)                  # actually sampling
    assert all(p._store_row(i) for i in chosen)        # stable across calls
    assert {i for i in ids if p._store_row(i)} == chosen
    p2, _ = px("sampled", ratio="1.0")
    assert all(p2._store_row(i) for i in ids)
    p3, _ = px("sampled", ratio="0.0")
    assert not any(p3._store_row(i) for i in ids)


def test_unknown_store_mode_falls_back_to_all():
    """An unreadable value must not silently mean "store nothing" — the safe
    fallback is the old behaviour, and it is chosen at import time."""
    import gateway.proxy_api as p
    src = open(p.__file__.replace(".pyc", ".py")).read()
    assert 'if REQUEST_STORE not in ("all", "sampled", "errors", "off"):' in src
    assert 'REQUEST_STORE = "all"' in src


def test_admission_record_is_released_at_finish(px):
    """`_ADMITTED` is per-in-flight-request state on the hot path; a leak there is a
    slow OOM."""
    p, _ = px("errors")
    _admit(p, "pxr-x")
    assert "pxr-x" in p._ADMITTED
    _finish(p, "pxr-x")
    assert "pxr-x" not in p._ADMITTED


def test_sweeper_reaps_records_whose_request_never_finished(px):
    p, _ = px("errors")
    _admit(p, "pxr-stale")
    p._ADMITTED["pxr-stale"]["t"] = 0.0        # admitted in 1970
    _admit(p, "pxr-fresh")
    assert p._sweep_admitted(max_age_s=60) == 1
    assert "pxr-stale" not in p._ADMITTED and "pxr-fresh" in p._ADMITTED


def test_finish_without_an_admission_record_still_updates(px):
    """A request admitted before a gateway restart (or evicted by the cap) must still
    have its terminal outcome applied to whatever row exists."""
    p, sent = px("all")
    _finish(p, "pxr-orphan", status="completed")
    assert [i["kind"] for i in sent] == ["proxy"]


def test_history_source_follows_the_storage_mode(px, monkeypatch):
    p, _ = px("all")
    monkeypatch.setattr(p.trace_store, "TEMPO_URL", "http://tempo:3200")
    monkeypatch.delenv("PROXY_HISTORY_SOURCE", raising=False)
    assert p._history_source() == "db"          # store=all → the table is complete
    p2, _ = px("off")
    monkeypatch.setattr(p2.trace_store, "TEMPO_URL", "http://tempo:3200")
    assert p2._history_source() == "trace"      # rows aren't kept → read traces
    monkeypatch.setattr(p2.trace_store, "TEMPO_URL", "")
    assert p2._history_source() == "db"         # no trace store configured → db
    monkeypatch.setenv("PROXY_HISTORY_SOURCE", "db")
    monkeypatch.setattr(p2.trace_store, "TEMPO_URL", "http://tempo:3200")
    assert p2._history_source() == "db"         # explicit override wins


def test_header_values_are_ascii_safe(px):
    """Headers are latin-1; an em-dash in a human-readable note 500'd the endpoint."""
    p, _ = px("all")
    assert p._hdr("store=off — see traces").isascii()


# ---------- the writer side: start/finish coalescing ---------------------------

def test_start_intent_does_not_stamp_a_terminal_state():
    """`record_proxy_start` and `record_proxy_finish` share one intent kind so they
    coalesce; the start half must not look like a completion."""
    from gateway import stats_writer as sw

    class Row:
        status = "queued"
        started_at = None
        completed_at = None
        upstream = None
        endpoint_id = "p1"
        model = "m1"

    row = Row()
    now = datetime.now(timezone.utc)
    sw._apply_proxy(row, {"running": True, "started_at": now, "upstream": "mock"}, now)
    assert row.status == "running"
    assert row.started_at == now
    assert row.completed_at is None          # ⚠ the bug this test exists for
    assert row.upstream == "mock"


def test_coalesced_start_and_finish_keep_both_halves():
    from gateway import stats_writer as sw

    class Row:
        status = "queued"
        started_at = None
        completed_at = None
        upstream = None
        status_code = None
        latency_ms = None
        ttft_ms = None
        prompt_tokens = None
        completion_tokens = None
        error_text = None
        endpoint_id = "p1"
        model = "m1"

    row = Row()
    now = datetime.now(timezone.utc)
    merged = {"running": True, "started_at": now, "status": "completed",
              "status_code": 200, "latency_ms": 42, "upstream": "mock"}
    sw._apply_proxy(row, merged, now)
    assert row.started_at == now and row.completed_at == now
    assert row.status == "completed" and row.latency_ms == 42


# ---------- the live overlay ---------------------------------------------------

def _app_with_live(entries):
    import types
    return types.SimpleNamespace(state=types.SimpleNamespace(proxy_live=entries))


def test_live_overlay_renders_in_flight_requests_the_response_model_accepts(px):
    """⚠ Outside `store=all` an in-flight request exists in NEITHER store — no row
    yet, and no span until it ENDS — so this overlay is the only thing standing
    between the Queue tab and an empty queue while requests are actively waiting."""
    p, _ = px("off")
    app = _app_with_live({
        "pxr-a": {"cancel": None, "state": "running", "endpoint_id": "proxy-1", "model": "m1",
                  "upstream": "mock", "created_at": 1786700000.5, "owner": "admin",
                  "is_stream": True, "id": "pxr-a"},
        "pxr-b": {"cancel": None, "state": "queued", "endpoint_id": "proxy-1", "model": "m1",
                  "upstream": None, "created_at": 1786700001.0, "owner": "admin",
                  "is_stream": False, "id": "pxr-b"},
        "pxr-z": {"cancel": None, "state": "queued", "endpoint_id": "proxy-9", "model": "x",
                  "upstream": None, "created_at": 1786700002.0, "owner": "u2",
                  "is_stream": False, "id": "pxr-z"},
    })
    recs = asyncio.run(p._live_records(app, "proxy-1"))
    assert {r["id"] for r in recs} == {"pxr-a", "pxr-b"}      # other endpoint excluded
    # The dicts must satisfy the response model — this is the shape both the db and
    # the trace branch hand to FastAPI.
    models = {m.id: m for m in (p.ProxyRequestRecord(**r) for r in recs)}
    assert models["pxr-a"].status == "running" and models["pxr-a"].live is True
    assert models["pxr-b"].status == "queued" and models["pxr-b"].is_stream is False
    assert models["pxr-a"].created_at.startswith("2026-")


def test_overlay_obeys_the_same_filters_as_the_stored_rows(px):
    p, _ = px("off")
    app = _app_with_live({
        "pxr-a": {"cancel": None, "state": "running", "endpoint_id": "proxy-1", "model": "m1",
                  "upstream": "mock", "created_at": 1786700000.5, "owner": "admin",
                  "is_stream": True, "id": "pxr-a"},
        "pxr-b": {"cancel": None, "state": "queued", "endpoint_id": "proxy-1", "model": "m1",
                  "upstream": None, "created_at": 1786700001.0, "owner": "bob",
                  "is_stream": False, "id": "pxr-b"},
    })
    recs = asyncio.run(p._live_records(app, "proxy-1"))
    f = p._matches_filters
    assert [r["id"] for r in recs if f(r, None, None, "queued", None)] == ["pxr-b"]
    assert [r["id"] for r in recs if f(r, None, "mock", None, None)] == ["pxr-a"]
    assert [r["id"] for r in recs if f(r, "admin", None, None, None)] == ["pxr-a"]
    assert [r["id"] for r in recs if f(r, "nobody", None, None, None)] == []
    assert [r["id"] for r in recs if f(r, None, None, None, "pxr-b")] == ["pxr-b"]
