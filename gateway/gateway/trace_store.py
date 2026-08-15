"""Read proxy-request history back OUT of the tracing backend (Grafana Tempo).

`tracing.py` writes one span per proxied request; this reads them back so the Queue
tab keeps working when Postgres no longer holds the rows (`PROXY_REQUEST_STORE=off`).
Tempo's TraceQL search is the query language; the mapping is deliberately narrow —
span attributes → the exact `ProxyRequestRecord` fields the tab renders.

    TEMPO_URL=http://localhost:3200      # Tempo's query API (NOT the OTLP port)
    TEMPO_ORG_ID=…                       # X-Scope-OrgID, for multi-tenant Tempo
    TEMPO_TOKEN=…                        # Bearer, e.g. Grafana Cloud
    TEMPO_TIMEOUT_S=8
    PROXY_HISTORY_SOURCE=auto|db|trace   # auto: trace iff rows aren't being stored
    TRACE_UI_URL=http://localhost:3001/explore?…  # optional deep-link base

**Three ways this differs from the SQL it replaces, all visible to the user:**

1. **A time window is mandatory.** Tempo searches blocks by time; "everything ever"
   is not a query it can answer. Every search carries `since_hours`
   (`PROXY_TRACE_WINDOW_H`, default 24) and the answer is scoped to it — the Queue
   tab says so rather than implying the window is the whole history.
2. **No `OFFSET`, and no `ORDER BY` but time.** Tempo returns most-recent-first and
   pages by narrowing the window, so deep paging and `sort=latency` are done HERE,
   over an over-fetched page capped at `PROXY_TRACE_MAX_FETCH`. Past that cap the
   ranking is partial and the caller is told (`note`) — a silently truncated "slowest
   first" reads as an answer.
3. **In-flight requests are not in it.** A span is exported when the request ENDS.
   Queued/running requests come from the live registry and are overlaid by
   `proxy_api`, which is why that overlay is not optional in trace mode.

**Never lie about an outage.** A Tempo that is down/unreachable raises — an empty
list would render as "no requests", which is the one answer a monitoring UI must
never invent.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger("gateway.trace_store")

TEMPO_URL = (os.environ.get("TEMPO_URL") or "").strip().rstrip("/")
TEMPO_ORG_ID = (os.environ.get("TEMPO_ORG_ID") or "").strip()
TEMPO_TOKEN = (os.environ.get("TEMPO_TOKEN") or "").strip()
TIMEOUT_S = float(os.environ.get("TEMPO_TIMEOUT_S", "8") or "8")
WINDOW_H = float(os.environ.get("PROXY_TRACE_WINDOW_H", "24") or "24")
# Ceiling on how many traces one page may pull back to satisfy offset/sort locally.
MAX_FETCH = int(os.environ.get("PROXY_TRACE_MAX_FETCH", "500") or "500")

_client: Optional[httpx.AsyncClient] = None

# span attribute -> ProxyRequestRecord field. The single place the two vocabularies
# meet; `tracing.py` writes exactly these keys.
_ATTRS = {
    "sgpu.request.id": "id",
    "sgpu.proxy.id": "endpoint_id",
    "sgpu.owner": "owner",
    "sgpu.model": "model",
    "sgpu.upstream": "upstream",
    "sgpu.status": "status",
    "sgpu.stream": "is_stream",
    "sgpu.latency_ms": "latency_ms",
    "sgpu.ttft_ms": "ttft_ms",
    "sgpu.error": "error_text",
    "http.response.status_code": "status_code",
    "gen_ai.usage.input_tokens": "prompt_tokens",
    "gen_ai.usage.output_tokens": "completion_tokens",
}


def enabled() -> bool:
    return bool(TEMPO_URL)


class TraceStoreError(RuntimeError):
    """The trace backend could not answer. Surfaced to the caller as a 502 — never
    swallowed into an empty result set."""


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        headers = {}
        if TEMPO_ORG_ID:
            headers["X-Scope-OrgID"] = TEMPO_ORG_ID
        if TEMPO_TOKEN:
            headers["Authorization"] = f"Bearer {TEMPO_TOKEN}"
        _client = httpx.AsyncClient(timeout=TIMEOUT_S, headers=headers)
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ---------- TraceQL ----------------------------------------------------------

def _q(value: str) -> str:
    """Quote a value for TraceQL. Backslash and double-quote are the only two
    characters that can break out of a string literal — the filters here are fed
    from user-chosen usernames / upstream names, so this is not decorative."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_query(endpoint_id: str, *, owner: Optional[str] = None,
                upstream: Optional[str] = None, status: Optional[str] = None,
                request_id: Optional[str] = None) -> str:
    """The TraceQL for one proxy endpoint's requests, plus the `select()` that makes
    Tempo return the columns the Queue tab renders. Without `select()` a search only
    returns the attributes it MATCHED on, and every other column would come back
    empty — which looks like missing data, not a missing projection."""
    parts = ['span.sgpu.request.kind="proxy"', f"span.sgpu.proxy.id={_q(endpoint_id)}"]
    if request_id:
        parts.append(f"span.sgpu.request.id={_q(request_id)}")
    if owner:
        parts.append(f"span.sgpu.owner={_q(owner)}")
    if upstream:
        parts.append(f"span.sgpu.upstream={_q(upstream)}")
    if status:
        parts.append(f"span.sgpu.status={_q(status)}")
    selects = ", ".join(f"span.{a}" for a in _ATTRS)
    return "{ " + " && ".join(parts) + " } | select(" + selects + ")"


def _attr_value(v: dict) -> Any:
    """Unwrap one OTLP AnyValue (`{"stringValue": …}` / intValue / boolValue / …).
    intValue arrives as a STRING in OTLP JSON — a detail that silently turns every
    token count and latency into text if you take it at face value."""
    if not isinstance(v, dict):
        return None
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        try:
            return int(v["intValue"])
        except (TypeError, ValueError):
            return None
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "doubleValue" in v:
        try:
            return float(v["doubleValue"])
        except (TypeError, ValueError):
            return None
    return None


def _iso(ns: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc).isoformat()


def span_to_record(trace: dict, span: dict) -> Optional[dict]:
    """One Tempo span → one ProxyRequestRecord-shaped dict."""
    attrs = {}
    for a in span.get("attributes") or []:
        if isinstance(a, dict) and a.get("key"):
            attrs[a["key"]] = _attr_value(a.get("value"))
    rid = attrs.get("sgpu.request.id")
    if not rid:
        return None
    rec: dict[str, Any] = {f: None for f in _ATTRS.values()}
    for key, field in _ATTRS.items():
        if attrs.get(key) is not None:
            rec[field] = attrs[key]
    rec["is_stream"] = bool(rec.get("is_stream"))
    try:
        start_ns = int(span.get("startTimeUnixNano") or trace.get("startTimeUnixNano") or 0)
        dur_ns = int(span.get("durationNanos") or 0)
    except (TypeError, ValueError):
        start_ns, dur_ns = 0, 0
    rec["created_at"] = _iso(start_ns) if start_ns else ""
    # The span covers admission → terminal, so its end IS completed_at. started_at is
    # a span EVENT (queue exit), not a field — the search API doesn't return events,
    # so it stays None here rather than being guessed from the start time.
    rec["completed_at"] = _iso(start_ns + dur_ns) if start_ns else None
    rec["started_at"] = None
    if rec.get("latency_ms") is None and dur_ns:
        rec["latency_ms"] = int(dur_ns / 1_000_000)
    rec["trace_id"] = trace.get("traceID")
    rec["live"] = False
    return rec


# ---------- search -----------------------------------------------------------

async def _search(query: str, *, limit: int, since_hours: float) -> list[dict]:
    now = int(time.time())
    params = {
        "q": query,
        "start": str(now - int(since_hours * 3600)),
        "end": str(now + 60),          # small forward margin for clock skew
        "limit": str(max(1, min(limit, MAX_FETCH))),
        "spss": "1",                   # one span per trace — a proxy trace has one
    }
    try:
        r = await _http().get(f"{TEMPO_URL}/api/search", params=params)
    except Exception as e:  # noqa: BLE001
        raise TraceStoreError(f"trace store unreachable at {TEMPO_URL}: {type(e).__name__}: {e}") from e
    if r.status_code >= 400:
        raise TraceStoreError(f"trace store returned HTTP {r.status_code}: {r.text[:300]}")
    try:
        data = r.json()
    except Exception as e:  # noqa: BLE001
        raise TraceStoreError(f"trace store returned an unparseable body: {type(e).__name__}") from e
    out: list[dict] = []
    for tr in (data.get("traces") or []):
        # Tempo has emitted both `spanSets` (current) and a single `spanSet` (older);
        # reading only one silently returns zero rows against the other version.
        sets = tr.get("spanSets") or ([tr["spanSet"]] if isinstance(tr.get("spanSet"), dict) else [])
        for ss in sets:
            for sp in (ss.get("spans") or []):
                rec = span_to_record(tr, sp)
                if rec is not None:
                    out.append(rec)
    return out


async def search_requests(endpoint_id: str, *, owner: Optional[str] = None,
                          upstream: Optional[str] = None, status: Optional[str] = None,
                          request_id: Optional[str] = None, limit: int = 50,
                          offset: int = 0, sort: str = "created", order: str = "desc",
                          since_hours: Optional[float] = None) -> tuple[list[dict], Optional[str]]:
    """One page of an endpoint's request history from the trace store.

    Returns `(records, note)`. `note` is non-None whenever the answer is bounded by
    something the caller can't see — the fetch cap, or a sort applied to a partial
    set — and the UI is expected to show it."""
    window = since_hours if since_hours is not None else WINDOW_H
    q = build_query(endpoint_id, owner=owner, upstream=upstream, status=status,
                    request_id=request_id)
    # ⚠ A non-time sort must fetch as much of the window as it is allowed to, NOT
    # `limit` rows. Tempo returns most-recent-first, so sorting a page of `limit`
    # traces by latency answers "the slowest of the 5 most recent" while looking
    # exactly like "the 5 slowest" — measured: a limit=3 latency sort returned the
    # 5200/410/300 ms requests and hid the 1820 ms one.
    fetch = MAX_FETCH if sort == "latency" else min(max(0, offset) + max(1, limit), MAX_FETCH)
    rows = await _search(q, limit=fetch, since_hours=window)
    # Truncation worth telling the user about is hitting the FETCH CAP — not merely
    # filling the page they asked for (that is what pagination is). Otherwise every
    # full page would carry a scary note.
    truncated = len(rows) >= fetch >= MAX_FETCH
    if sort == "latency":
        rows.sort(key=lambda r: (r.get("latency_ms") is None, r.get("latency_ms") or 0),
                  reverse=(order != "asc"))
    else:
        rows.sort(key=lambda r: r.get("created_at") or "", reverse=(order != "asc"))
    page = rows[offset:offset + limit] if offset else rows[:limit]
    note = f"from traces, last {window:g}h"
    if truncated:
        note += (f" (ranked over the {len(rows)} most recent traces only)"
                 if sort == "latency" else f" (newest {len(rows)} traces only)")
    return page, note


async def facet_users(endpoint_id: str, since_hours: Optional[float] = None) -> list[str]:
    """Distinct `sgpu.owner` values for one endpoint — the Queue tab's user filter.
    Tempo answers this natively (tag values scoped by a TraceQL filter), so it is one
    request, not a scan of the search results."""
    window = since_hours if since_hours is not None else WINDOW_H
    now = int(time.time())
    params = {
        "q": "{ " + f'span.sgpu.request.kind="proxy" && span.sgpu.proxy.id={_q(endpoint_id)}' + " }",
        "start": str(now - int(window * 3600)),
        "end": str(now + 60),
    }
    try:
        r = await _http().get(f"{TEMPO_URL}/api/v2/search/tag/span.sgpu.owner/values", params=params)
        if r.status_code >= 400:
            logger.debug("tempo tag values HTTP %s: %s", r.status_code, r.text[:200])
            return []
        data = r.json()
    except Exception:  # noqa: BLE001 — a missing FILTER list is a degraded UI, not an
        # error page; the request list itself still fails loudly if Tempo is down.
        logger.debug("tempo tag values failed", exc_info=True)
        return []
    vals = []
    for v in (data.get("tagValues") or []):
        if isinstance(v, dict) and v.get("value"):
            vals.append(str(v["value"]))
        elif isinstance(v, str):
            vals.append(v)
    return sorted(set(vals))


def trace_ui_url(trace_id: Optional[str], request_id: Optional[str] = None) -> Optional[str]:
    """Deep link to one request's trace in Grafana/Jaeger, when TRACE_UI_URL is set.

    Two placeholders, and `{request}` is the one that matters:

    * `{trace}`   — substituted with the trace id. Only available for rows that came
      FROM the trace store, because a trace id is minted by the tracer and never
      stored anywhere else.
    * `{request}` — substituted with the `pxr-…` request id. This works for a
      Postgres row too, which is the whole point: in the default
      `PROXY_REQUEST_STORE=all` mode the Queue tab reads rows that have no trace id,
      so a `{trace}`-only template renders NO link on the very page people are
      looking at. Point it at a TraceQL search — `{ span.sgpu.request.id="{request}" }` —
      and one template serves both sources.

    A template with neither placeholder gets the id appended (the Jaeger
    `/trace/<id>` shape)."""
    base = (os.environ.get("TRACE_UI_URL") or "").strip()
    if not base:
        return None
    if "{request}" in base:
        if not request_id:
            return None
        return base.replace("{request}", request_id)
    if not trace_id:
        return None
    if "{trace}" in base:
        return base.replace("{trace}", trace_id)
    return base.rstrip("/") + "/" + trace_id
