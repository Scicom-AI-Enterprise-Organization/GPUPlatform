"""Import replayable requests from Langfuse traces.

The manual workflow this replaces: open a trace in the Langfuse UI, realise the
UI's "download JSON" strips the observation payloads, run a separate CLI against
the *public* API instead, then hand-repair the extracted request when it comes
back mangled. All of that is mechanical, and all of it is here.

Two hard-won details are load-bearing — both produce a *silently wrong* result
rather than an error, so they're implemented, not documented-around:

1. **`traceId` beats `peek`.** In the current UI layout `peek=` can be a *span*
   id, which 404s against `/api/public/traces/{id}`. The real trace is `traceId=`
   when both are present.
2. **PII-scrubbed JSON-string inputs.** Some observations store the request as a
   JSON *string* whose scrubber replaced bare numeric values with **unquoted**
   tokens (`"created_at": <id>`). That is invalid JSON, so a naive `json.loads`
   fails, the caller treats the string as an iterable of messages, and the
   extract becomes thousands of **single-character** messages — with no error
   raised. `_repair_scrubbed_json` re-quotes placeholders only in structural
   value positions (never inside customer text), and `extract_request` refuses
   an implausible message shape outright.

Langfuse credentials are per-import (project-scoped public/secret key pair) and
may be supplied inline or as global-secret references — the same
`api_key_secret` convention the proxy upstreams use.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import httpx

logger = logging.getLogger("gateway.langfuse")

# A scrubbed placeholder sitting in a JSON *value* slot: `: <id>,` / `: <id>}`.
# The lookahead is what keeps the substitution out of customer content strings —
# quoting a `<email>` inside a message body would corrupt the text.
_STRUCTURAL_PLACEHOLDER = re.compile(r":\s*<([a-z_]+)>\s*(?=[,}\]])")

_RANGE_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}

# A replayable request should have a handful of messages, not thousands. The
# single-character-message corruption above always blows past this.
_MAX_PLAUSIBLE_MESSAGES = 500


class LangfuseError(RuntimeError):
    """Import failed in a way the user needs to act on (bad URL, auth, shape)."""


def _repair_scrubbed_json(raw: str) -> str:
    """Quote PII placeholders that sit in structural value positions."""
    return _STRUCTURAL_PLACEHOLDER.sub(r': "<\1>"', raw)


def parse_date_range(spec: str) -> Optional[timedelta]:
    """'90d' / '24h' / '30m' / '2w' → timedelta; None if unparseable."""
    spec = (spec or "").strip().lower()
    if not spec or spec in ("all", "none"):
        return None
    unit = spec[-1:]
    if unit not in _RANGE_UNITS or not spec[:-1].isdigit():
        return None
    return timedelta(**{_RANGE_UNITS[unit]: int(spec[:-1])})


def parse_langfuse_url(raw: str) -> dict[str, Any]:
    """Pull {project_id, trace_id, observation_id, days} out of a Langfuse UI URL.

    Accepts a list-view URL with `?peek=`/`?traceId=`, a trace permalink
    (`/traces/<id>`), or a bare trace id.
    """
    raw = (raw or "").strip()
    if not raw:
        raise LangfuseError("empty URL")

    # A bare id (32-hex or similar) — not a URL at all.
    if "://" not in raw and "/" not in raw and "?" not in raw:
        return {"trace_id": raw, "observation_id": None, "project_id": None, "days": None}

    parsed = urlparse(raw)
    qs = parse_qs(parsed.query)
    parts = [p for p in parsed.path.split("/") if p]

    project_id = None
    if "project" in parts:
        i = parts.index("project")
        if i + 1 < len(parts):
            project_id = parts[i + 1]

    # traceId wins over peek — see the module docstring.
    trace_id = (qs.get("traceId") or [None])[0] or (qs.get("peek") or [None])[0]
    if not trace_id and "traces" in parts:
        i = parts.index("traces")
        if i + 1 < len(parts):
            trace_id = parts[i + 1]

    observation_id = (qs.get("observation") or [None])[0]
    days = None
    dr = (qs.get("dateRange") or [None])[0]
    if dr:
        delta = parse_date_range(dr)
        if delta:
            days = max(1, int(delta.total_seconds() // 86400) or 1)

    return {
        "trace_id": trace_id,
        "observation_id": observation_id,
        "project_id": project_id,
        "days": days,
        "used_trace_id_param": bool((qs.get("traceId") or [None])[0]),
    }


def _auth_header(public_key: str, secret_key: str) -> dict[str, str]:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


async def fetch_trace(
    client: httpx.AsyncClient,
    base_url: str,
    public_key: str,
    secret_key: str,
    trace_id: str,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """GET one full trace (all observations) from the public API."""
    url = f"{base_url.rstrip('/')}/api/public/traces/{trace_id}"
    resp = await client.get(url, headers=_auth_header(public_key, secret_key), timeout=timeout)
    if resp.status_code == 401:
        # Keys are PROJECT-scoped: a URL's project id is informational only and
        # does NOT switch projects, so a mismatch reads as an auth failure.
        raise LangfuseError(
            "401 from Langfuse — the key pair is project-scoped and doesn't match this "
            "trace's project (a URL's project id does not switch projects)."
        )
    if resp.status_code == 404:
        raise LangfuseError(
            f"trace {trace_id} not found on {base_url}. If the URL came from the newer UI, "
            "`peek=` may be a span id — use the `traceId=` param. Also check the trace isn't "
            "on a different Langfuse instance (self-hosted vs cloud)."
        )
    resp.raise_for_status()
    return resp.json()


async def list_traces(
    client: httpx.AsyncClient,
    base_url: str,
    public_key: str,
    secret_key: str,
    *,
    days: Optional[int] = None,
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    environment: Optional[str] = None,
    limit: int = 50,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Trace summaries, newest first, honouring filters + a hard limit.

    ⚠ `name` is an **exact** match — the UI's search box is fuzzy, so pasting the
    UI's search text here silently returns nothing.
    """
    url = f"{base_url.rstrip('/')}/api/public/traces"
    params: dict[str, Any] = {"limit": min(limit, 100), "orderBy": "timestamp.desc"}
    if days:
        params["fromTimestamp"] = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat()
    if name:
        params["name"] = name
    if user_id:
        params["userId"] = user_id
    if session_id:
        params["sessionId"] = session_id
    if environment:
        params["environment"] = environment
    if tags:
        params["tags"] = tags

    out: list[dict[str, Any]] = []
    page = 1
    headers = _auth_header(public_key, secret_key)
    while len(out) < limit and page <= 20:
        resp = await client.get(
            url, headers=headers, params=dict(params, page=page), timeout=timeout
        )
        if resp.status_code == 401:
            raise LangfuseError("401 from Langfuse — check the project-scoped key pair.")
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data") or []
        if not data:
            break
        out.extend(data)
        meta = body.get("meta") or {}
        if page >= (meta.get("totalPages") or page):
            break
        page += 1
    return out[:limit]


def _coerce_input(raw: Any) -> Any:
    """Parse an observation `input` that may be an object, or a JSON string that
    may itself be PII-scrub-corrupted."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_repair_scrubbed_json(raw))
    except json.JSONDecodeError:
        # Leave it as a string — extract_request rejects it with a useful message
        # rather than iterating it into single-character "messages".
        return raw


def descendant_generations(trace: dict[str, Any], obs_id: str) -> list[dict[str, Any]]:
    """GENERATION observations anywhere beneath `obs_id` in the span tree."""
    observations = [o for o in trace.get("observations", []) if isinstance(o, dict)]
    by_id = {o.get("id"): o for o in observations}
    found: list[dict[str, Any]] = []
    for obs in observations:
        if obs.get("type") != "GENERATION":
            continue
        cur, hops = obs, 0
        while cur and cur.get("parentObservationId") and hops < 50:
            if cur["parentObservationId"] == obs_id:
                found.append(obs)
                break
            cur = by_id.get(cur["parentObservationId"])
            hops += 1
    return found


def list_generations(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Every GENERATION observation, with just enough detail to pick one in the UI."""
    out = []
    for obs in trace.get("observations", []):
        if not isinstance(obs, dict) or obs.get("type") != "GENERATION":
            continue
        inp = _coerce_input(obs.get("input"))
        messages = inp.get("messages") if isinstance(inp, dict) else inp
        n_msgs = len(messages) if isinstance(messages, list) else 0
        out.append({
            "id": obs.get("id"),
            "name": obs.get("name"),
            "model": obs.get("model"),
            "start_time": obs.get("startTime"),
            "n_messages": n_msgs,
            "n_tools": len((inp or {}).get("tools") or []) if isinstance(inp, dict) else 0,
            "replayable": n_msgs > 0 and n_msgs <= _MAX_PLAUSIBLE_MESSAGES,
        })
    return out


def _model_params(obs: dict[str, Any]) -> dict[str, Any]:
    """Recorded sampling params, normalized to the replay shape.

    Langfuse stores these as `modelParameters`, sometimes with `extra_body`
    nested as a JSON string. Recovering them matters: replaying with library
    defaults instead of the trace's actual `temperature`/`enable_thinking`
    reproduces a *different* request than the one that misbehaved.
    """
    raw = obs.get("modelParameters") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    if not isinstance(raw, dict):
        return {}

    out: dict[str, Any] = {}
    for key in ("temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"):
        if raw.get(key) is not None:
            out[key] = raw[key]

    extra = raw.get("extra_body")
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except json.JSONDecodeError:
            extra = None
    if isinstance(extra, dict):
        for key in ("top_k", "repetition_penalty", "min_p"):
            if extra.get(key) is not None:
                out[key] = extra[key]
        ctk = extra.get("chat_template_kwargs")
        if isinstance(ctk, dict) and ctk.get("enable_thinking") is not None:
            out["enable_thinking"] = bool(ctk["enable_thinking"])
    # Some SDKs flatten these onto modelParameters directly.
    for key in ("top_k", "repetition_penalty"):
        if raw.get(key) is not None and key not in out:
            out[key] = raw[key]
    return out


def extract_request(trace: dict[str, Any], obs_id: Optional[str] = None) -> dict[str, Any]:
    """One observation's request as `{messages, tools, params, model}`.

    With no `obs_id`, picks the first replayable GENERATION in the trace.
    Raises `LangfuseError` with an actionable message when the named span isn't
    a chat request — for an agno AGENT span (`*.arun`) the input is the task
    string and the real request is a child GENERATION.
    """
    observations = [o for o in trace.get("observations", []) if isinstance(o, dict)]
    if not observations:
        raise LangfuseError(
            "trace has no observations — a metadata-only export. Pull via the public API "
            "(this importer does), not the UI's download button."
        )

    obs: Optional[dict[str, Any]] = None
    if obs_id:
        obs = next((o for o in observations if o.get("id") == obs_id), None)
        if obs is None:
            raise LangfuseError(f"observation {obs_id} not found in trace {trace.get('id')}")
    else:
        for cand in observations:
            if cand.get("type") != "GENERATION":
                continue
            inp = _coerce_input(cand.get("input"))
            msgs = inp.get("messages") if isinstance(inp, dict) else inp
            if isinstance(msgs, list) and msgs:
                obs = cand
                break
        if obs is None:
            raise LangfuseError("no GENERATION observation with input messages in this trace")

    inp = _coerce_input(obs.get("input"))

    if isinstance(inp, str):
        raise LangfuseError(
            f"observation {obs.get('id')} stores its input as a string that isn't valid JSON "
            "even after repairing PII placeholders. Pick a child GENERATION instead."
        )

    messages = inp.get("messages") if isinstance(inp, dict) else inp
    if not isinstance(messages, list) or not messages:
        kids = descendant_generations(trace, obs.get("id") or "")
        hint = ""
        if kids:
            ids = ", ".join(str(k.get("id")) for k in kids[:5])
            hint = f" Its child GENERATION observations are replayable: {ids}"
        raise LangfuseError(
            f"observation {obs.get('id')} (type={obs.get('type')}, name={obs.get('name')}) "
            f"has no replayable chat request — its input is not a messages payload.{hint}"
        )

    # The corruption guard: single-character "messages" from an unrepaired scrub.
    if len(messages) > _MAX_PLAUSIBLE_MESSAGES:
        raise LangfuseError(
            f"extracted {len(messages)} messages from observation {obs.get('id')} — that is the "
            "signature of a PII-scrubbed JSON string that failed to parse, not a real request. "
            "Refusing to import a corrupted case."
        )
    if not all(isinstance(m, dict) for m in messages):
        raise LangfuseError(
            f"observation {obs.get('id')} produced non-object messages — the input was likely "
            "a string iterated character-by-character. Refusing to import."
        )

    payload: dict[str, Any] = {
        "messages": clean_messages(messages),
        "tools": (inp.get("tools") if isinstance(inp, dict) else None) or [],
        "params": _model_params(obs),
        "model": obs.get("model"),
        "observation_id": obs.get("id"),
        "observation_name": obs.get("name"),
    }
    return payload


def clean_messages(raw: list[Any]) -> list[dict[str, Any]]:
    """Keep exactly the fields that make a request replay byte-for-byte."""
    out: list[dict[str, Any]] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        msg: dict[str, Any] = {"role": m.get("role") or "user", "content": m.get("content")}
        for key in ("tool_calls", "tool_call_id", "name"):
            if m.get(key):
                msg[key] = m[key]
        out.append(msg)
    return out


# Internal alias kept for readability inside this module.
_clean_messages = clean_messages
