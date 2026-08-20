"""Correlation-id propagation — the parts that fail silently when broken.

The point of a correlation id is that ONE value reaches Tempo, Loki and the upstream
worker. Every failure mode here is quiet: the id is still minted, logs still ship,
requests still succeed — you only find out when you go looking for a request's logs and
the field is null or the upstream never saw it. So these pin the plumbing rather than
the happy path.

`wan` is optional and OFF by default (`GATEWAY_CORRELATION`), so everything is asserted
in the degraded, no-wan configuration — which is the one that actually runs today.
"""
import asyncio

import httpx
from gateway.accesslog import correlation_id_var, request_id_var, trace_ids
from gateway.proxy_api import _stamp_correlation

from gateway import observability


def _clear():
    correlation_id_var.set(None)
    request_id_var.set(None)


def test_correlation_id_reads_the_gateway_var_not_just_wan():
    """⚠ The regression this file exists for. `correlation_id()` originally read only
    wan's contextvar, so with GATEWAY_CORRELATION unset (the DEFAULT) it returned None
    even though metrics_mw had minted and set an id — and upstream propagation silently
    never happened in the configuration everyone actually runs."""
    _clear()
    assert observability.enabled() is False          # wan not installed in unit env
    correlation_id_var.set("cid-abc123")
    assert observability.correlation_id() == "cid-abc123"


def test_correlation_id_is_none_outside_a_request():
    _clear()
    assert observability.correlation_id() is None


def test_inject_adds_the_header_without_wan():
    _clear()
    correlation_id_var.set("cid-abc123")
    h = observability.inject({"Content-Type": "application/json"})
    assert h["X-Correlation-ID"] == "cid-abc123"
    assert h["Content-Type"] == "application/json"   # must not clobber what was there


def test_inject_never_overwrites_an_explicit_header():
    """A caller that set the header deliberately outranks the ambient context."""
    _clear()
    correlation_id_var.set("cid-ambient")
    h = observability.inject({"X-Correlation-ID": "cid-explicit"})
    assert h["X-Correlation-ID"] == "cid-explicit"


def test_inject_is_a_noop_with_no_id():
    _clear()
    assert observability.inject({}) == {}


def test_upstream_hook_stamps_both_ids():
    """The httpx event hook on the shared proxy client is the single choke point for
    upstream calls — if it stops stamping, all ~15 call sites lose propagation at once."""
    _clear()
    correlation_id_var.set("cid-abc123")
    request_id_var.set("pxr-293d632640b55ad3")
    req = httpx.Request("POST", "http://upstream/v1/chat/completions",
                        headers={"Authorization": "Bearer k"})
    asyncio.run(_stamp_correlation(req))
    assert req.headers["x-correlation-id"] == "cid-abc123"
    assert req.headers["x-request-id"] == "pxr-293d632640b55ad3"
    assert req.headers["authorization"] == "Bearer k"   # untouched


def test_upstream_hook_never_raises_without_context():
    """Telemetry must not fail a proxied request. No ids set, no OTel span, no wan."""
    _clear()
    req = httpx.Request("GET", "http://upstream/health")
    asyncio.run(_stamp_correlation(req))                # must not raise
    assert "x-correlation-id" not in req.headers


def test_trace_ids_degrade_to_none_without_a_span():
    """The OTel wheels are optional and there is no active span in a unit run; a missing
    trace id must cost a log FIELD, not the log line."""
    assert trace_ids() == (None, None)


def test_new_id_is_unique_and_prefixed():
    a, b = observability.new_id(), observability.new_id()
    assert a != b
    assert a.startswith("cid-")


# --------------------------------------------------------------- span parenting
# The Tempo↔Loki jump depends entirely on the proxy span sharing a trace id with the
# log lines. Log lines take theirs from the AMBIENT span (the HTTP server span), so the
# proxy span must join it unless the caller supplied a traceparent of its own.

def _proxy_span_trace(monkeypatch, headers):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    from gateway import tracing

    monkeypatch.setenv("PROXY_TRACING", "1")
    monkeypatch.setattr(tracing, "_init_done", False)
    monkeypatch.setattr(tracing, "_enabled", False)
    tracing.start()
    outer = TracerProvider().get_tracer("ambient").start_span("http server span")
    with trace.use_span(outer, end_on_exit=False):
        ambient = outer.get_span_context().trace_id
        rid = f"pxr-{len(headers)}{id(headers) & 0xffff:x}"
        tracing.start_request(rid, endpoint_id="e", endpoint_name="n", model="m",
                              owner=None, owner_id=None, is_stream=False, headers=headers)
        pending = tracing._PENDING.get(rid)
        got = pending["span"].get_span_context().trace_id if pending else None
    return ambient, got


def test_span_joins_the_ambient_trace_when_caller_sends_no_traceparent(monkeypatch):
    """⚠ The regression. `extract()` on headers with no traceparent returns an EMPTY
    context, and `start_span(context=<empty>)` starts a NEW ROOT TRACE rather than
    inheriting. That put the span in a different trace from the log lines, so going from
    a Tempo trace to its Loki logs silently returned nothing."""
    ambient, got = _proxy_span_trace(monkeypatch, {"authorization": "Bearer k"})
    assert got == ambient


def test_caller_traceparent_still_wins_over_the_ambient_span(monkeypatch):
    """The fix must not cost the original behaviour: a client already tracing its own
    work owns the trace, so its leg and ours stay one trace."""
    tp = "00-11111111111111111111111111111111-2222222222222222-01"
    ambient, got = _proxy_span_trace(monkeypatch, {"traceparent": tp})
    assert got == int("1" * 32, 16)
    assert got != ambient
