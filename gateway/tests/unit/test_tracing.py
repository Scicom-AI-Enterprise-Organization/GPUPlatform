"""Unit tests for proxy request tracing (gateway/tracing.py) + the Tempo read path
(gateway/trace_store.py). No collector and no Tempo needed — the export decision, the
attribute mapping and the TraceQL builder are all pure."""

from __future__ import annotations

import importlib

import pytest


# ---------- sampling: the decision is made at the END, on the outcome -----------

def _fresh(monkeypatch, **env):
    """Re-import tracing with a given env (its knobs are read at import time)."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import gateway.tracing as t
    return importlib.reload(t)


def test_errors_are_kept_at_any_sample_ratio(monkeypatch):
    t = _fresh(monkeypatch, PROXY_TRACE_SAMPLE_RATIO="0.0")
    try:
        # A head sampler would drop these too — which is the whole reason the
        # decision is deferred to on_end.
        for status in ("failed", "cancelled", "blocked", "unknown"):
            assert t._keep(status, 200, 100, trace_id=1) is True
        assert t._keep("completed", 500, 100, trace_id=1) is True
        assert t._keep("completed", 429, 100, trace_id=1) is True
        # …and a plain success at ratio 0 is dropped.
        assert t._keep("completed", 200, 100, trace_id=1) is False
    finally:
        importlib.reload(t)


def test_slow_requests_are_kept(monkeypatch):
    t = _fresh(monkeypatch, PROXY_TRACE_SAMPLE_RATIO="0.0", PROXY_TRACE_SLOW_MS="5000")
    try:
        assert t._keep("completed", 200, 5001, trace_id=1) is True
        assert t._keep("completed", 200, 4999, trace_id=1) is False
        assert t._keep("completed", 200, None, trace_id=1) is False
    finally:
        importlib.reload(t)


def test_ratio_is_deterministic_in_the_trace_id(monkeypatch):
    t = _fresh(monkeypatch, PROXY_TRACE_SAMPLE_RATIO="0.5")
    try:
        low, high = 1, (1 << 64) - 1          # below and above the 50% cut
        assert t._keep("completed", 200, 10, trace_id=low) is True
        assert t._keep("completed", 200, 10, trace_id=high) is False
        # Same id → same answer, every time (replicas must agree).
        assert all(t._keep("completed", 200, 10, trace_id=low) for _ in range(5))
    finally:
        importlib.reload(t)


def test_disabled_tracing_is_a_total_no_op():
    """Every entry point must be safe to call when tracing is off — they sit on the
    request path and are called unconditionally."""
    import gateway.tracing as t
    importlib.reload(t)
    assert t.enabled() is False
    t.start_request("pxr-1", endpoint_id="p1", endpoint_name="n", model="m",
                    owner="u", owner_id=1, is_stream=False)
    t.mark_started("pxr-1", "up")
    t.end_request("pxr-1", "completed", status_code=200, latency_ms=5)
    assert t.trace_id("pxr-1") is None
    assert t.sweep() == 0
    assert t.pending_count() == 0


# ---------- trace_store: TraceQL + span→record ---------------------------------

def test_traceql_quotes_hostile_filter_values():
    from gateway import trace_store as ts
    q = ts.build_query('proxy-1', owner='ev"il\\user')
    # The injected quote must be escaped, not close the literal.
    assert '"ev\\"il\\\\user"' in q
    assert q.count("{") == 1 and q.startswith("{ ")


def test_traceql_selects_every_mapped_column():
    """Without an explicit select() Tempo returns only the attributes it matched on,
    so every unselected column would come back empty — data that looks missing."""
    from gateway import trace_store as ts
    q = ts.build_query("proxy-1")
    for attr in ts._ATTRS:
        assert f"span.{attr}" in q, attr


def test_traceql_filters_are_anded_into_the_query():
    from gateway import trace_store as ts
    q = ts.build_query("proxy-1", owner="admin", upstream="mock", status="failed",
                       request_id="pxr-abc")
    for frag in ('span.sgpu.proxy.id="proxy-1"', 'span.sgpu.owner="admin"',
                 'span.sgpu.upstream="mock"', 'span.sgpu.status="failed"',
                 'span.sgpu.request.id="pxr-abc"', 'span.sgpu.request.kind="proxy"'):
        assert frag in q, frag


def _span(attrs: dict, start_ns=1_700_000_000_000_000_000, dur_ns=1_500_000_000):
    def val(v):
        if isinstance(v, bool):
            return {"boolValue": v}
        if isinstance(v, int):
            return {"intValue": str(v)}   # OTLP JSON sends ints as STRINGS
        return {"stringValue": str(v)}
    return {
        "spanID": "abc", "startTimeUnixNano": str(start_ns), "durationNanos": str(dur_ns),
        "attributes": [{"key": k, "value": val(v)} for k, v in attrs.items()],
    }


def test_span_to_record_maps_every_field_and_coerces_int_strings():
    from gateway import trace_store as ts
    rec = ts.span_to_record({"traceID": "deadbeef"}, _span({
        "sgpu.request.id": "pxr-1", "sgpu.proxy.id": "proxy-1", "sgpu.owner": "admin",
        "sgpu.model": "m1", "sgpu.upstream": "mock", "sgpu.status": "completed",
        "sgpu.stream": True, "sgpu.latency_ms": 1500, "sgpu.ttft_ms": 120,
        "http.response.status_code": 200,
        "gen_ai.usage.input_tokens": 7, "gen_ai.usage.output_tokens": 3,
    }))
    assert rec["id"] == "pxr-1"
    assert rec["status"] == "completed"
    assert rec["is_stream"] is True
    # ints, not the strings OTLP puts on the wire
    assert rec["latency_ms"] == 1500 and isinstance(rec["latency_ms"], int)
    assert rec["status_code"] == 200 and rec["prompt_tokens"] == 7 and rec["completion_tokens"] == 3
    assert rec["trace_id"] == "deadbeef"
    assert rec["live"] is False
    assert rec["created_at"].startswith("2023-11-14T")
    assert rec["completed_at"] > rec["created_at"]


def test_span_without_a_request_id_is_dropped():
    """Anything else in the trace store (another service's spans) must not surface as
    a phantom row in the Queue tab."""
    from gateway import trace_store as ts
    assert ts.span_to_record({"traceID": "x"}, _span({"http.route": "/other"})) is None


def test_latency_falls_back_to_the_span_duration():
    from gateway import trace_store as ts
    rec = ts.span_to_record({"traceID": "x"}, _span({"sgpu.request.id": "pxr-1"},
                                                    dur_ns=2_000_000_000))
    assert rec["latency_ms"] == 2000


@pytest.mark.parametrize("payload_key", ["spanSets", "spanSet"])
def test_both_tempo_response_shapes_are_read(monkeypatch, payload_key):
    """Tempo has emitted `spanSets` (current) and a bare `spanSet`; handling one only
    would silently return zero rows against the other version."""
    import asyncio

    from gateway import trace_store as ts

    span = _span({"sgpu.request.id": "pxr-1", "sgpu.status": "completed"})
    sets = {"spans": [span], "matched": 1}
    trace = {"traceID": "t1", "startTimeUnixNano": "1700000000000000000",
             payload_key: ([sets] if payload_key == "spanSets" else sets)}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"traces": [trace]}

    class _Client:
        async def get(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(ts, "_http", lambda: _Client())
    monkeypatch.setattr(ts, "TEMPO_URL", "http://tempo:3200")
    rows = asyncio.run(ts._search("{}", limit=10, since_hours=1))
    assert [r["id"] for r in rows] == ["pxr-1"]


def test_backend_failure_raises_rather_than_returning_empty(monkeypatch):
    """An unreachable trace store must not render as 'no requests'."""
    import asyncio

    from gateway import trace_store as ts

    class _Client:
        async def get(self, *a, **k):
            raise OSError("connection refused")

    monkeypatch.setattr(ts, "_http", lambda: _Client())
    monkeypatch.setattr(ts, "TEMPO_URL", "http://tempo:3200")
    with pytest.raises(ts.TraceStoreError):
        asyncio.run(ts._search("{}", limit=10, since_hours=1))


def test_latency_sort_fetches_the_whole_window_not_one_page(monkeypatch):
    """Tempo returns most-recent-first and cannot sort, so ranking a page of `limit`
    traces by latency answers 'slowest of the newest N' while looking like 'slowest'.
    Measured for real: a limit=3 sort hid the 1820 ms request behind three shorter,
    newer ones."""
    import asyncio

    from gateway import trace_store as ts

    seen = {}

    async def fake_search(q, *, limit, since_hours):
        seen["limit"] = limit
        return [
            {"id": "a", "latency_ms": 300, "created_at": "2026-01-01T00:00:03"},
            {"id": "b", "latency_ms": 5200, "created_at": "2026-01-01T00:00:02"},
            {"id": "c", "latency_ms": 1820, "created_at": "2026-01-01T00:00:01"},
        ]

    monkeypatch.setattr(ts, "_search", fake_search)
    rows, _ = asyncio.run(ts.search_requests("p1", limit=2, sort="latency"))
    assert seen["limit"] == ts.MAX_FETCH            # not `limit`
    assert [r["id"] for r in rows] == ["b", "c"]    # 5200, 1820 — the actual slowest


def test_truncation_note_appears_only_at_the_fetch_cap(monkeypatch):
    """A note on every full page would be noise; a MISSING note when the ranking is
    partial would be a lie. The line is the fetch cap, not the page size."""
    import asyncio

    from gateway import trace_store as ts

    rows_in = [{"id": str(i), "latency_ms": i, "created_at": f"2026-01-01T00:00:{i:02d}"}
               for i in range(3)]

    async def fake_search(q, *, limit, since_hours):
        return rows_in[:limit]

    monkeypatch.setattr(ts, "_search", fake_search)
    monkeypatch.setattr(ts, "MAX_FETCH", 3)
    _, note = asyncio.run(ts.search_requests("p1", limit=2))          # page < cap
    assert "only" not in note
    _, note = asyncio.run(ts.search_requests("p1", limit=2, sort="latency"))  # hits cap
    assert "only" in note


def test_trace_ui_url_templating(monkeypatch):
    from gateway import trace_store as ts
    monkeypatch.delenv("TRACE_UI_URL", raising=False)
    assert ts.trace_ui_url("abc") is None          # not configured → no dead link
    monkeypatch.setenv("TRACE_UI_URL", "http://g/explore?traceId={trace}")
    assert ts.trace_ui_url("abc") == "http://g/explore?traceId=abc"
    monkeypatch.setenv("TRACE_UI_URL", "http://jaeger/trace/")
    assert ts.trace_ui_url("abc") == "http://jaeger/trace/abc"
    assert ts.trace_ui_url(None) is None
