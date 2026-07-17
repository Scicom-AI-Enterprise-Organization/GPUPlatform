"""LLM API proxy — a centralized OpenAI-compatible router to external providers.

One stable endpoint + model name (`POST /proxy/{endpoint}/v1/chat/completions`,
`model: "qwen"`) fans out to multiple external OpenAI-compatible backends. The
team never changes anything; backends are added/swapped/health-routed here.

- **Priority + failover** routing across the upstreams that serve a model alias,
  skipping ones a background health loop marked dead (look-ahead).
- **Per-endpoint concurrency cap** → excess requests wait in a visible queue.
- **Auto-cancel on client disconnect** (unary: a disconnect watcher cancels the
  forward task; streaming: the generator's finally closes the upstream stream),
  plus a manual cancel API. A queued request cancels by never being dispatched.
- Recent requests are persisted to Postgres for history/audit.

Data plane auth = a platform API key (`sgpu_…`, via `current_user`); management
routes are admin-only. Upstream API keys are referenced by a GlobalEnv secret key
(resolved per call) or pasted (Fernet-encrypted at rest), like Storage.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets as _secrets
import struct
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from . import crypto
from . import proxy_cluster
from .auth import current_user, require_admin
from .db import Base, User, get_session, get_user_by_username, session_factory
from .global_env_api import load_global_env

logger = logging.getLogger("gateway.proxy")

# Management router (admin) + data-plane router (API-key) — both included in main.
router = APIRouter(prefix="/v1/proxy", tags=["proxy"])
data_router = APIRouter(tags=["proxy-data"])

HEALTH_TTL_S = 120                 # a probe older than this is "stale/unknown"
# 0 (default) = keep proxy request history INDEFINITELY (no prune). Set
# PROXY_REQUEST_RETENTION_DAYS=N to prune rows older than N days.
REQUEST_RETENTION_DAYS = int(os.environ.get("PROXY_REQUEST_RETENTION_DAYS", "0") or "0")
DEFAULT_TIMEOUT_S = 3600.0
import re as _re
_NAME_RE = _re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


# ---------- DB models (registered via init_db side-effect import) -----------

class ProxyEndpoint(Base):
    __tablename__ = "proxy_endpoints"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # proxy-<hex8>
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)  # the {endpoint} path segment
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
    # Public proxies are visible (read-only, secret-stripped) to every logged-in
    # user via /v1/proxy/public[/{id}] — name, serving URL, model aliases only.
    # Admin-managed flag (proxies have no per-user ownership). Default private.
    public: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    # { max_concurrency:int, timeout_s:int, upstreams:[{id,name,base_url,
    #   api_key_secret?|api_key_enc?, models:{alias:real}, priority, enabled}] }
    config: Mapped[dict] = mapped_column(JSON, default=dict, server_default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ProxyRequest(Base):
    """Persisted record of a proxied request (history/audit). Inserted at
    admission, updated at terminal; pruned by the health loop."""
    __tablename__ = "proxy_requests"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # pxr-<hex8>
    endpoint_id: Mapped[str] = mapped_column(String(64), index=True)
    owner_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    upstream: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    # queued | running | completed | cancelled | failed
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    is_stream: Mapped[bool] = mapped_column(Boolean, default=False)
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ttft_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # time-to-first-token (stream)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_text: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


# ---------- request / response models ---------------------------------------

class UpstreamSpec(BaseModel):
    id: Optional[str] = None
    name: str
    base_url: str
    api_key_secret: Optional[str] = None  # GlobalEnv key (resolved at call time)
    api_key: Optional[str] = None         # OR a pasted key (stored Fernet-encrypted); write-only
    models: dict[str, str] = {}           # alias -> real upstream model name
    priority: int = 0
    enabled: bool = True
    # Extra top-level JSON merged into every forwarded body for this upstream, e.g.
    # OpenRouter provider pinning: {"provider": {"order": ["ModelRun"], "allow_fallbacks": false}}.
    # Optional; the upstream's own keys win over the caller's, `model` always wins over both.
    extra_body: Optional[dict] = None


class SttCallbackSpec(BaseModel):
    """A whisper-compatible STT endpoint the TTS proxy calls back to transcribe its
    own generated audio, for async CER/WER drift metrics. Not a data-plane upstream —
    it never serves client requests. Key handling mirrors UpstreamSpec."""
    enabled: bool = True
    base_url: str = ""                    # e.g. https://…/proxy/stt/v1 (an /audio/transcriptions base)
    model: str = ""                       # STT model to send (real upstream name)
    api_key_secret: Optional[str] = None  # GlobalEnv key ref
    api_key: Optional[str] = None         # OR a pasted key (stored encrypted); write-only


class CaptureSpec(BaseModel):
    """Persist audio (+ a JSON sidecar) to a storage backend when a request crosses a
    quality threshold — STT low confidence (avg_logprob < threshold) / TTS high error
    (CER or WER > threshold). For collecting drift/failure samples to inspect or retrain."""
    enabled: bool = False
    storage_id: str = ""                    # a /v1/storage backend id
    prefix: str = "drift/"                  # key prefix within that backend
    logprob_threshold: Optional[float] = None  # STT: capture if avg_logprob < this
    cer_threshold: Optional[float] = None      # TTS: capture if cer > this
    wer_threshold: Optional[float] = None      # TTS: capture if wer > this


class CreateProxyRequest(BaseModel):
    name: str
    max_concurrency: int = 0              # 0 = unlimited (no queue)
    # Same source as the runtime fallback (DEFAULT_TIMEOUT_S) so the per-endpoint
    # "Timeout (s)" default and the fallback can't drift — change the constant once.
    timeout_s: int = int(DEFAULT_TIMEOUT_S)
    enabled: bool = True
    public: bool = False
    upstreams: list[UpstreamSpec] = []
    stt_callback: Optional[SttCallbackSpec] = None
    capture: Optional[CaptureSpec] = None


class UpdateProxyRequest(BaseModel):
    name: Optional[str] = None
    max_concurrency: Optional[int] = None
    timeout_s: Optional[int] = None
    enabled: Optional[bool] = None
    public: Optional[bool] = None
    upstreams: Optional[list[UpstreamSpec]] = None
    stt_callback: Optional[SttCallbackSpec] = None
    capture: Optional[CaptureSpec] = None


class UpstreamRecord(BaseModel):
    id: str
    name: str
    base_url: str
    api_key_secret: Optional[str] = None
    has_inline_key: bool = False
    models: dict[str, str] = {}
    priority: int = 0
    enabled: bool = True
    extra_body: dict = {}


class SttCallbackRecord(BaseModel):
    enabled: bool = True
    base_url: str = ""
    model: str = ""
    api_key_secret: Optional[str] = None
    has_inline_key: bool = False


class CaptureRecord(BaseModel):
    enabled: bool = False
    storage_id: str = ""
    prefix: str = "drift/"
    logprob_threshold: Optional[float] = None
    cer_threshold: Optional[float] = None
    wer_threshold: Optional[float] = None


class ProxyEndpointRecord(BaseModel):
    id: str
    name: str
    enabled: bool
    public: bool = False
    max_concurrency: int
    timeout_s: int
    upstreams: list[UpstreamRecord]
    stt_callback: Optional[SttCallbackRecord] = None
    capture: Optional[CaptureRecord] = None
    inflight: int = 0
    queued: int = 0
    created_at: str
    created_by: str


class UpstreamHealth(BaseModel):
    upstream_id: str
    name: str
    alive: Optional[bool] = None          # None = not probed yet
    latency_ms: Optional[int] = None
    checked_at: Optional[float] = None
    error: Optional[str] = None
    stale: bool = False


class ProxyRequestRecord(BaseModel):
    id: str
    endpoint_id: str
    owner: Optional[str] = None  # username of the API key / session that made the request
    model: Optional[str] = None
    upstream: Optional[str] = None
    status: str
    is_stream: bool = False
    status_code: Optional[int] = None
    latency_ms: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    error_text: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    live: bool = False


class TestUpstreamRequest(BaseModel):
    base_url: str
    api_key_secret: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None   # real upstream model to end-to-end test; None = just probe /models
    mode: str = "chat"            # "chat" | "embedding" — which endpoint to end-to-end test
    extra_body: Optional[dict] = None  # merged into the chat test body (e.g. provider pinning)


class TestUpstreamResponse(BaseModel):
    ok: bool
    message: str
    latency_ms: Optional[int] = None
    models: list[str] = []


# ---------- app.state accessors ---------------------------------------------

def _http(app) -> httpx.AsyncClient:
    cli = getattr(app.state, "proxy_http", None)
    if cli is None:
        cli = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=10.0),
            follow_redirects=True,
        )
        app.state.proxy_http = cli
    return cli


def _health(app) -> dict:
    h = getattr(app.state, "proxy_health", None)
    if h is None:
        h = {}
        app.state.proxy_health = h
    return h


def _live(app) -> dict:
    lv = getattr(app.state, "proxy_live", None)
    if lv is None:
        lv = {}
        app.state.proxy_live = lv
    return lv


def _get_sem(app, endpoint_id: str, max_conc: int) -> Optional[asyncio.Semaphore]:
    """Per-endpoint admission gate. None = unlimited. Recreated when the cap changes."""
    if max_conc <= 0:
        return None
    sems = getattr(app.state, "proxy_sems", None)
    if sems is None:
        sems = {}
        app.state.proxy_sems = sems
    cur = sems.get(endpoint_id)
    if cur is None or cur[1] != max_conc:
        sem = asyncio.Semaphore(max_conc)
        sems[endpoint_id] = (sem, max_conc)
        return sem
    return cur[0]


class _Gate:
    """One endpoint's concurrency gate. Two backends behind a single async
    acquire→token / release(token) interface so the forwarding engine doesn't
    branch on mode:
      • LOCAL (default) — a per-replica ``asyncio.Semaphore`` (``sem`` set). The cap
        applies per replica; behavior identical to the old raw-semaphore path.
      • GLOBAL (``PROXY_CLUSTER=1``, ``sem is None``) — a Redis ZSET-lease slot shared
        by every replica, so ``max_concurrency`` is a cluster-wide cap. Over cap →
        the caller waits (polls) for a slot, honoring ``cancel_ev`` — same visible-queue
        semantics as the local semaphore, just global."""

    def __init__(self, app, endpoint_id: str, max_conc: int, sem: Optional[asyncio.Semaphore]):
        self.app = app
        self.endpoint_id = endpoint_id
        self.max_conc = max_conc
        self.sem = sem  # None => Redis global mode

    async def acquire(self, cancel_ev: Optional[asyncio.Event] = None, live_entry: Optional[dict] = None):
        if self.sem is not None:
            await self.sem.acquire()
            return "__local__"
        token = f"{proxy_cluster.replica_id()}-{_secrets.token_hex(6)}"
        while True:
            if cancel_ev is not None and cancel_ev.is_set():
                raise asyncio.CancelledError()
            if await proxy_cluster.limiter_acquire(self.app.state.redis, self.endpoint_id, self.max_conc, token):
                # Stash the slot token on the live entry so the sync loop renews its
                # lease (long streams) and releases it if this replica dies.
                if live_entry is not None:
                    live_entry["slot_token"] = token
                return token
            await asyncio.sleep(0.1)  # at global cap — wait in the (visible) queue

    async def release(self, token) -> None:
        if token is None:
            return
        if token == "__local__":
            if self.sem is not None:
                self.sem.release()
            return
        await proxy_cluster.limiter_release(self.app.state.redis, self.endpoint_id, token)


def _get_gate(app, endpoint_id: str, max_conc: int) -> Optional["_Gate"]:
    """Concurrency gate for an endpoint, or None when uncapped (max_conc<=0).
    Redis-global when PROXY_CLUSTER=1, else the per-replica semaphore."""
    if max_conc <= 0:
        return None
    if proxy_cluster.enabled():
        return _Gate(app, endpoint_id, max_conc, sem=None)
    return _Gate(app, endpoint_id, max_conc, sem=_get_sem(app, endpoint_id, max_conc))


def _mark_health(app, endpoint_id: str, upstream_id: str, alive: bool,
                 latency_ms: Optional[int] = None, error: Optional[str] = None) -> None:
    _health(app)[(endpoint_id, upstream_id)] = {
        "alive": alive, "latency_ms": latency_ms, "error": error,
        "checked_at": time.time(),
    }


# ---------- helpers ----------------------------------------------------------

def _upstream_record(u: dict) -> UpstreamRecord:
    return UpstreamRecord(
        id=u.get("id", ""), name=u.get("name", ""), base_url=u.get("base_url", ""),
        api_key_secret=u.get("api_key_secret"), has_inline_key=bool(u.get("api_key_enc")),
        models=u.get("models") or {}, priority=int(u.get("priority", 0)),
        enabled=bool(u.get("enabled", True)), extra_body=u.get("extra_body") or {},
    )


def _endpoint_record(app, e: ProxyEndpoint, owner_username: str,
                     counts: Optional[tuple[int, int]] = None) -> ProxyEndpointRecord:
    cfg = e.config or {}
    if counts is None:  # per-replica counts (local _live); pass global counts when clustered
        live = _live(app)
        mine = [v for v in live.values() if v.get("endpoint_id") == e.id]
        inflight = sum(1 for v in mine if v.get("state") == "running")
        queued = sum(1 for v in mine if v.get("state") == "queued")
    else:
        inflight, queued = counts
    return ProxyEndpointRecord(
        id=e.id, name=e.name, enabled=bool(e.enabled), public=bool(getattr(e, "public", False)),
        max_concurrency=int(cfg.get("max_concurrency") or 0),
        timeout_s=int(cfg.get("timeout_s") or int(DEFAULT_TIMEOUT_S)),
        upstreams=[_upstream_record(u) for u in cfg.get("upstreams", [])],
        stt_callback=_stt_callback_record(cfg),
        capture=_capture_record(cfg),
        inflight=inflight,
        queued=queued,
        created_at=e.created_at.isoformat() if e.created_at else "",
        created_by=owner_username,
    )


async def _endpoint_counts(app, endpoint_id: str) -> Optional[tuple[int, int]]:
    """Global (inflight, queued) for an endpoint when PROXY_CLUSTER=1, else None so
    the caller falls back to this replica's local counts."""
    if not proxy_cluster.enabled():
        return None
    return await proxy_cluster.live_counts_endpoint(app.state.redis, endpoint_id)


def _public_endpoint_record(app, e: ProxyEndpoint) -> ProxyEndpointRecord:
    """Secret-stripped view of a PUBLIC proxy for non-admin viewers. Returns the
    serving name, model aliases, and per-upstream priority/enabled only — NEVER
    upstream base_urls, API-key references, real upstream model names, upstream
    names, or the owning admin's username. Reuses ProxyEndpointRecord so the web
    type is shared with the admin view."""
    cfg = e.config or {}
    live = _live(app)
    mine = [v for v in live.values() if v.get("endpoint_id") == e.id]
    safe_upstreams = [
        UpstreamRecord(
            id=u.get("id", ""), name="", base_url="",
            api_key_secret=None, has_inline_key=False,
            # keep the alias KEYS (callers need them) but blank the real model names
            models={a: "" for a in (u.get("models") or {}).keys()},
            priority=int(u.get("priority", 0)), enabled=bool(u.get("enabled", True)),
        )
        for u in cfg.get("upstreams", [])
    ]
    return ProxyEndpointRecord(
        id=e.id, name=e.name, enabled=bool(e.enabled), public=True,
        max_concurrency=int(cfg.get("max_concurrency") or 0),
        timeout_s=int(cfg.get("timeout_s") or int(DEFAULT_TIMEOUT_S)),
        upstreams=safe_upstreams,
        inflight=sum(1 for v in mine if v.get("state") == "running"),
        queued=sum(1 for v in mine if v.get("state") == "queued"),
        created_at=e.created_at.isoformat() if e.created_at else "",
        created_by="",
    )


def _build_upstreams(specs: list[UpstreamSpec], existing: Optional[list[dict]] = None) -> list[dict]:
    """Turn API specs into stored upstream dicts. Inline keys are encrypted; a
    blank inline key preserves the existing encrypted key (matched by id)."""
    by_id = {u.get("id"): u for u in (existing or []) if u.get("id")}
    out: list[dict] = []
    for sp in specs:
        uid = sp.id or f"up-{_secrets.token_hex(4)}"
        prev = by_id.get(uid, {})
        u: dict[str, Any] = {
            "id": uid,
            "name": sp.name.strip(),
            "base_url": sp.base_url.strip().rstrip("/"),
            "models": {k.strip(): v.strip() for k, v in (sp.models or {}).items() if k.strip() and v.strip()},
            "priority": int(sp.priority),
            "enabled": bool(sp.enabled),
        }
        # extra_body is admin intent, not a secret — always round-tripped by the form,
        # so store it verbatim when it's a non-empty object, drop it otherwise (no
        # preserve-on-omit like keys: sending {} / null clears it).
        if isinstance(sp.extra_body, dict) and sp.extra_body:
            u["extra_body"] = sp.extra_body
        ref = (sp.api_key_secret or "").strip()
        if ref:
            u["api_key_secret"] = ref  # secret reference wins; no inline key kept
        elif (sp.api_key or "").strip():
            u["api_key_enc"] = crypto.encrypt(sp.api_key.strip())
        elif prev.get("api_key_enc"):
            u["api_key_enc"] = prev["api_key_enc"]  # preserve on edit when omitted
        elif prev.get("api_key_secret"):
            u["api_key_secret"] = prev["api_key_secret"]
        out.append(u)
    return out


def _resolve_key(u: dict, genv: dict[str, str]) -> str:
    ref = (u.get("api_key_secret") or "").strip()
    if ref:
        return genv.get(ref, "")
    enc = u.get("api_key_enc")
    if enc:
        try:
            return crypto.decrypt(enc)
        except Exception:
            return ""
    return ""


def _build_stt_callback(spec: Optional[SttCallbackSpec], existing: Optional[dict] = None) -> Optional[dict]:
    """Turn the API spec into the stored stt_callback dict (or None to clear it).
    None spec on UPDATE = field omitted → keep existing. A spec with a blank
    base_url/model clears it. Inline key encrypted; blank key preserved on edit."""
    if spec is None:
        return existing  # PATCH: omitted → unchanged
    base = (spec.base_url or "").strip().rstrip("/")
    model = (spec.model or "").strip()
    if not base or not model:
        return None  # incomplete/cleared → disable the callback
    prev = existing or {}
    out: dict[str, Any] = {"enabled": bool(spec.enabled), "base_url": base, "model": model}
    ref = (spec.api_key_secret or "").strip()
    if ref:
        out["api_key_secret"] = ref
    elif (spec.api_key or "").strip():
        out["api_key_enc"] = crypto.encrypt(spec.api_key.strip())
    elif prev.get("api_key_enc"):
        out["api_key_enc"] = prev["api_key_enc"]
    elif prev.get("api_key_secret"):
        out["api_key_secret"] = prev["api_key_secret"]
    return out


def _stt_callback_record(cfg: dict) -> Optional[SttCallbackRecord]:
    c = cfg.get("stt_callback")
    if not isinstance(c, dict) or not c.get("base_url"):
        return None
    return SttCallbackRecord(
        enabled=bool(c.get("enabled", True)),
        base_url=c.get("base_url", ""),
        model=c.get("model", ""),
        api_key_secret=c.get("api_key_secret"),
        has_inline_key=bool(c.get("api_key_enc")),
    )


def _build_capture(spec: Optional[CaptureSpec], existing: Optional[dict] = None) -> Optional[dict]:
    """Stored capture dict (or None to clear). None spec on UPDATE = keep existing; a spec
    with no storage_id clears it. Thresholds omitted (None) mean that dimension won't trigger."""
    if spec is None:
        return existing
    sid = (spec.storage_id or "").strip()
    if not sid:
        return None
    out: dict[str, Any] = {"enabled": bool(spec.enabled), "storage_id": sid,
                           "prefix": (spec.prefix or "").strip()}
    for key, val in (("logprob_threshold", spec.logprob_threshold),
                     ("cer_threshold", spec.cer_threshold), ("wer_threshold", spec.wer_threshold)):
        if val is not None:
            out[key] = float(val)
    return out


def _capture_record(cfg: dict) -> Optional[CaptureRecord]:
    c = cfg.get("capture")
    if not isinstance(c, dict) or not c.get("storage_id"):
        return None
    return CaptureRecord(
        enabled=bool(c.get("enabled", False)), storage_id=c.get("storage_id", ""),
        prefix=c.get("prefix", "drift/"), logprob_threshold=c.get("logprob_threshold"),
        cer_threshold=c.get("cer_threshold"), wer_threshold=c.get("wer_threshold"),
    )


def _select_candidates(app, endpoint_id: str, cfg: dict, alias: str) -> list[dict]:
    """Upstreams serving `alias`, enabled, ordered: alive-or-unknown first, then
    by priority (lower = preferred); known-dead pushed to the back (still tried)."""
    ups = [u for u in cfg.get("upstreams", []) if u.get("enabled", True) and alias in (u.get("models") or {})]
    health = _health(app)

    def sort_key(u: dict):
        h = health.get((endpoint_id, u.get("id")))
        dead = bool(h) and not h.get("alive", True) and (time.time() - h.get("checked_at", 0)) < HEALTH_TTL_S
        return (1 if dead else 0, int(u.get("priority", 0)))

    return sorted(ups, key=sort_key)


# ---------- DB record updates (own session, safe from disconnect path) -------

async def _set_started(request_id: str, upstream: Optional[str] = None) -> None:
    async with session_factory()() as s:
        row = await s.get(ProxyRequest, request_id)
        if row is None:
            return
        row.status = "running"
        row.started_at = datetime.now(timezone.utc)
        if upstream:
            row.upstream = upstream
        await s.commit()


async def _finish(request_id: str, status: str, *, status_code: Optional[int] = None,
                  latency_ms: Optional[int] = None, pt: Optional[int] = None,
                  ct: Optional[int] = None, error: Optional[str] = None,
                  upstream: Optional[str] = None, ttft_ms: Optional[int] = None,
                  avg_logprob: Optional[float] = None) -> None:
    """Record a proxied request's terminal outcome (status / latency / TTFT / tokens
    / audio logprob) + its per-proxy Prometheus metric. Enqueued to the batch stats
    writer rather than committed inline: opening a pooled connection per completion
    exhausted the DB pool under load and wedged the gateway. Still `async def` so every
    existing `await _finish(...)` / `create_task(_finish(...))` call site is unchanged;
    the DB write + metric now happen in the writer's batched flush (parity preserved)."""
    from . import stats_writer
    stats_writer.record_proxy_finish(
        request_id, status, status_code=status_code, latency_ms=latency_ms,
        pt=pt, ct=ct, error=error, upstream=upstream, ttft_ms=ttft_ms,
        avg_logprob=avg_logprob,
    )


# ---------- forwarding engine ------------------------------------------------

async def _do_unary(app, endpoint_id: str, candidates: list[dict], alias: str,
                    payload: dict, upstream_path: str, timeout_s: float) -> dict:
    """Try candidates in order; failover on connect error / 5xx. Returns the
    upstream's JSON + status. Raises HTTPException(502) if all fail."""
    cli = _http(app)
    last_err = "no upstream"
    for u in candidates:
        body = {**payload, **(u.get("extra_body") or {}), "model": u["models"][alias]}
        headers = {"Content-Type": "application/json"}
        if u.get("_key"):
            headers["Authorization"] = f"Bearer {u['_key']}"
        url = u["base_url"].rstrip("/") + upstream_path
        t0 = time.perf_counter()
        try:
            r = await cli.post(url, json=body, headers=headers,
                               timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=timeout_s, pool=10.0))
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError) as e:
            _mark_health(app, endpoint_id, u["id"], False, error=str(e))
            last_err = f"{u['name']}: {type(e).__name__}"
            continue
        lat = int((time.perf_counter() - t0) * 1000)
        if r.status_code >= 500:
            _mark_health(app, endpoint_id, u["id"], False, latency_ms=lat, error=f"HTTP {r.status_code}")
            last_err = f"{u['name']}: HTTP {r.status_code}"
            continue
        _mark_health(app, endpoint_id, u["id"], True, latency_ms=lat)
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}
        usage = (data.get("usage") or {}) if isinstance(data, dict) else {}
        return {"upstream": u["name"], "upstream_url": u["base_url"], "status_code": r.status_code,
                "data": data, "latency_ms": lat, "pt": usage.get("prompt_tokens"), "ct": usage.get("completion_tokens")}
    raise HTTPException(status_code=502, detail={"error": f"all upstreams failed: {last_err}"})


def _extract_audio_logprob(data: object) -> Optional[float]:
    """Pull a single mean token log-probability out of a whisper transcription/
    translation response, for the drift metric. Prefers the duration-weighted mean
    of per-segment `avg_logprob` (response_format=verbose_json); falls back to the
    mean of top-level token `logprobs` (OpenAI include[]=logprobs shape). Returns
    None when neither is present (e.g. response_format=json/text, or empty result).
    Best-effort: never raises — a malformed body must not break the passthrough."""
    if not isinstance(data, dict):
        return None
    try:
        segs = data.get("segments")
        if isinstance(segs, list) and segs:
            num = 0.0
            wsum = 0.0
            for s in segs:
                if not isinstance(s, dict) or s.get("avg_logprob") is None:
                    continue
                w = (s.get("end") or 0) - (s.get("start") or 0)
                if w <= 0:
                    w = 1.0
                num += float(s["avg_logprob"]) * w
                wsum += w
            if wsum > 0:
                return num / wsum
        lps = data.get("logprobs")
        if isinstance(lps, list) and lps:
            vals = [float(x["logprob"]) for x in lps
                    if isinstance(x, dict) and x.get("logprob") is not None]
            if vals:
                return sum(vals) / len(vals)
    except (TypeError, ValueError):
        return None
    return None


def _fmt_ts(sec: object, sep: str) -> str:
    """Seconds → an SRT/VTT timestamp `HH:MM:SS<sep>mmm` (sep=',' for SRT, '.' for VTT)."""
    try:
        v = max(0.0, float(sec or 0))
    except (TypeError, ValueError):
        v = 0.0
    whole = int(v)
    ms = int(round((v - whole) * 1000))
    if ms == 1000:  # rounding carry
        whole += 1
        ms = 0
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _segments_to_srt(vj: dict) -> str:
    out: list[str] = []
    for i, s in enumerate(vj.get("segments") or [], 1):
        if not isinstance(s, dict):
            continue
        out += [str(i), f"{_fmt_ts(s.get('start'), ',')} --> {_fmt_ts(s.get('end'), ',')}",
                (s.get("text") or "").strip(), ""]
    return "\n".join(out)


def _segments_to_vtt(vj: dict) -> str:
    out: list[str] = ["WEBVTT", ""]
    for s in vj.get("segments") or []:
        if not isinstance(s, dict):
            continue
        out += [f"{_fmt_ts(s.get('start'), '.')} --> {_fmt_ts(s.get('end'), '.')}",
                (s.get("text") or "").strip(), ""]
    return "\n".join(out)


# response_format values we can reconstruct FROM a verbose_json body — so we can always
# ask the upstream for verbose_json (→ segment avg_logprob → drift metric) and downgrade.
_CONVERTIBLE_AUDIO_FORMATS = ("", "json", "verbose_json", "text", "srt", "vtt")


def _convert_verbose_audio(vj: dict, client_fmt: str) -> dict:
    """Downgrade a verbose_json transcription body to the caller's response_format.
    Returns fields to merge into the result: either `data` (a JSON body) or
    `raw_body` + `media_type` (text/srt/vtt). verbose_json passes through unchanged."""
    if client_fmt in ("", "json"):
        return {"data": {"text": vj.get("text", "")}}
    if client_fmt == "verbose_json":
        return {"data": vj}
    if client_fmt == "text":
        return {"raw_body": (vj.get("text") or ""), "media_type": "text/plain; charset=utf-8"}
    if client_fmt == "srt":
        return {"raw_body": _segments_to_srt(vj), "media_type": "text/plain; charset=utf-8"}
    if client_fmt == "vtt":
        return {"raw_body": _segments_to_vtt(vj), "media_type": "text/vtt; charset=utf-8"}
    return {"data": vj}  # unreachable given the convertible gate


async def _do_unary_multipart(app, endpoint_id: str, candidates: list[dict], alias: str,
                              upstream_path: str, file_name: str, file_bytes: bytes,
                              content_type: str, form_fields: dict, timeout_s: float,
                              capture: Optional[dict] = None, request_id: Optional[str] = None,
                              sample_id: Optional[str] = None) -> dict:
    """Multipart forward (audio transcriptions/translations). Rebuilds the multipart body
    (uploaded file + passthrough form fields, model rewritten). For the standard whisper
    response formats it ALWAYS asks the upstream for `verbose_json` — its per-segment
    `avg_logprob` is the drift signal (metrics.PROXY_AUDIO_NLL) and it's a superset of
    json/text/srt/vtt — then converts the body back to exactly the format the caller asked
    for. If an upstream rejects verbose_json (4xx) we retry once in the caller's own format
    so a working request never breaks (forgoing the logprob). Failover on connect error/5xx."""
    cli = _http(app)
    last_err = "no upstream"
    client_fmt = str(form_fields.get("response_format") or "json").strip().lower()
    upgradable = client_fmt in _CONVERTIBLE_AUDIO_FORMATS
    for u in candidates:
        files = {"file": (file_name, file_bytes, content_type)}
        headers = {"Authorization": f"Bearer {u['_key']}"} if u.get("_key") else {}
        url = u["base_url"].rstrip("/") + upstream_path
        force_verbose = upgradable
        while True:
            data = {**form_fields, "model": u["models"][alias]}
            if force_verbose:
                data["response_format"] = "verbose_json"
            t0 = time.perf_counter()
            try:
                r = await cli.post(url, data=data, files=files, headers=headers,
                                   timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=timeout_s, pool=10.0))
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError) as e:
                _mark_health(app, endpoint_id, u["id"], False, error=str(e))
                last_err = f"{u['name']}: {type(e).__name__}"
                break  # → next candidate
            lat = int((time.perf_counter() - t0) * 1000)
            if r.status_code >= 500:
                _mark_health(app, endpoint_id, u["id"], False, latency_ms=lat, error=f"HTTP {r.status_code}")
                last_err = f"{u['name']}: HTTP {r.status_code}"
                break  # → next candidate (failover)
            if force_verbose and 400 <= r.status_code < 500:
                # Upstream won't do verbose_json — retry once in the caller's own format
                # (no logprob) rather than fail a request that would otherwise have worked.
                force_verbose = False
                continue
            _mark_health(app, endpoint_id, u["id"], True, latency_ms=lat)
            res = {"upstream": u["name"], "upstream_url": u["base_url"], "status_code": r.status_code,
                   "latency_ms": lat, "pt": None, "ct": None, "avg_logprob": None}
            try:
                body = r.json()
            except Exception:
                body = None
            if force_verbose and isinstance(body, dict):
                res["avg_logprob"] = _extract_audio_logprob(body)
                res.update(_convert_verbose_audio(body, client_fmt))
                lp = res["avg_logprob"]
                if (capture and lp is not None and capture.get("logprob_threshold") is not None
                        and lp < capture["logprob_threshold"]):
                    ext = os.path.splitext(file_name or "")[1].lstrip(".").lower() or "wav"
                    sid = sample_id or request_id
                    _fire_capture(app, endpoint_id, "stt", file_bytes, ext,
                                  {"kind": "stt", "model": alias, "avg_logprob": round(lp, 4),
                                   "transcription": body.get("text", ""), "threshold": capture["logprob_threshold"],
                                   "request_id": sid, "proxy_request_id": request_id,
                                   "ts": datetime.now(timezone.utc).isoformat()},
                                  capture, sid)
            elif body is None:
                res["raw_body"] = r.text          # response_format=text → plain string
                res["media_type"] = r.headers.get("content-type") or "text/plain; charset=utf-8"
            else:
                res["data"] = body                # already in the caller's format (fallback path)
            return res
    raise HTTPException(status_code=502, detail={"error": f"all upstreams failed: {last_err}"})


async def _watch_cancel(request: Request, cancel_ev: asyncio.Event) -> None:
    """Resolve when the client disconnects OR a manual cancel fires."""
    while True:
        if cancel_ev.is_set():
            return
        try:
            if await request.is_disconnected():
                cancel_ev.set()
                return
        except Exception:
            return
        await asyncio.sleep(0.25)


async def _unary(app, request: Request, request_id: str, forward,
                 gate: Optional["_Gate"], cancel_ev: asyncio.Event) -> Response:
    """Run one non-streaming forward (`forward` is a zero-arg coroutine returning the
    {upstream,status_code,data,latency_ms,pt,ct} dict) with disconnect-cancel + the
    concurrency slot + request tracking. Works for JSON (chat/completions/embeddings)
    and multipart (audio) forwards alike."""
    live = _live(app)

    async def work() -> Response:
        token = None
        try:
            if gate is not None:
                token = await gate.acquire(cancel_ev, live.get(request_id))
            if cancel_ev.is_set():
                raise asyncio.CancelledError()
            if request_id in live:
                live[request_id]["state"] = "running"
            await _set_started(request_id)
            res = await forward()
            if request_id in live:
                live[request_id]["upstream"] = res["upstream"]
            await _finish(request_id, "completed", status_code=res["status_code"],
                          latency_ms=res["latency_ms"], pt=res["pt"], ct=res["ct"], upstream=res["upstream"],
                          avg_logprob=res.get("avg_logprob"))
            # Surface which upstream actually served this request so callers can tell
            # them apart (failover/health routing means it isn't always the primary).
            hdrs = {"X-Request-Id": request_id}
            if res.get("upstream_url"):
                hdrs["X-Upstream-Url"] = res["upstream_url"]
            if res.get("upstream"):
                hdrs["X-Upstream-Name"] = res["upstream"]
            # Audio forwards may return a non-JSON body (response_format=text/srt/vtt,
            # reconstructed from the upstream's verbose_json); everything else is JSON.
            if res.get("raw_body") is not None:
                return Response(res["raw_body"], status_code=res["status_code"],
                                media_type=res.get("media_type") or "text/plain; charset=utf-8", headers=hdrs)
            return JSONResponse(res.get("data"), status_code=res["status_code"], headers=hdrs)
        finally:
            if gate is not None:
                await gate.release(token)

    wtask = asyncio.create_task(work())
    ctask = asyncio.create_task(_watch_cancel(request, cancel_ev))
    try:
        done, _ = await asyncio.wait({wtask, ctask}, return_when=asyncio.FIRST_COMPLETED)
        if wtask in done:
            ctask.cancel()
            try:
                return wtask.result()
            except HTTPException as e:
                await _finish(request_id, "failed", status_code=getattr(e, "status_code", 502),
                              error=json.dumps(e.detail) if not isinstance(e.detail, str) else e.detail)
                raise
            except asyncio.CancelledError:
                await _finish(request_id, "cancelled", status_code=499, error="cancelled")
                return Response(status_code=499)
        # client disconnected / manual cancel — abort the forward
        wtask.cancel()
        try:
            await wtask
        except BaseException:
            pass
        await _finish(request_id, "cancelled", status_code=499, error="client disconnected")
        return Response(status_code=499)
    finally:
        live.pop(request_id, None)


async def _stream(app, request_id: str, endpoint_id: str, candidates: list[dict], alias: str,
                  payload: dict, upstream_path: str, timeout_s: float,
                  gate: Optional["_Gate"], cancel_ev: asyncio.Event):
    """SSE passthrough. Failover only before the first byte. The finally releases
    the slot, drops the live entry, and (on disconnect/cancel) marks the row
    cancelled via a detached task (no await during generator close)."""
    live = _live(app)
    cli = _http(app)
    token = None
    gate_held = False
    finished = False
    try:
        if gate is not None:
            token = await gate.acquire(cancel_ev, live.get(request_id))
            gate_held = True
        if cancel_ev.is_set():
            return
        if request_id in live:
            live[request_id]["state"] = "running"
        await _set_started(request_id)
        last_err = "no upstream"
        for u in candidates:
            body = {**payload, **(u.get("extra_body") or {}), "model": u["models"][alias], "stream": True}
            # Ask for a final usage chunk so streamed requests record token counts too
            # (vLLM/OpenAI emit it last when include_usage=true). Don't clobber a caller's.
            body.setdefault("stream_options", {"include_usage": True})
            headers = {"Content-Type": "application/json"}
            if u.get("_key"):
                headers["Authorization"] = f"Bearer {u['_key']}"
            url = u["base_url"].rstrip("/") + upstream_path
            t0 = time.perf_counter()
            try:
                async with cli.stream("POST", url, json=body, headers=headers,
                                      timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=timeout_s, pool=10.0)) as r:
                    if r.status_code >= 500:
                        _mark_health(app, endpoint_id, u["id"], False, error=f"HTTP {r.status_code}")
                        last_err = f"{u['name']}: HTTP {r.status_code}"
                        continue  # failover (nothing sent yet)
                    _mark_health(app, endpoint_id, u["id"], True, latency_ms=int((time.perf_counter() - t0) * 1000))
                    if request_id in live:
                        live[request_id]["upstream"] = u["name"]
                    await _set_started(request_id, upstream=u["name"])
                    # With failover candidates the chosen upstream isn't known when the
                    # SSE response HEADERS flush, so X-Upstream-* can't be set there. Emit
                    # it instead as a leading SSE COMMENT (a line starting with ":", which
                    # the SSE spec says consumers ignore — OpenAI/EventSource parsers skip
                    # it, so it's safe for every client) that the playground surfaces.
                    if len(candidates) > 1:
                        yield (": sgpu-upstream " + json.dumps({"name": u["name"], "url": u["base_url"]}) + "\n\n").encode()
                    sent_any = False
                    ttft_ms: Optional[int] = None
                    pt = ct = None
                    usage_buf = ""
                    try:
                        async for chunk in r.aiter_bytes():
                            if cancel_ev.is_set():
                                break
                            if not sent_any:
                                sent_any = True
                                ttft_ms = int((time.perf_counter() - t0) * 1000)  # time-to-first-token
                            yield chunk
                            # Light usage scan — the `"usage"` substring guard keeps the
                            # JSON parser off content chunks; only the final usage frame hits it.
                            try:
                                usage_buf += chunk.decode("utf-8", "ignore")
                                while "\n\n" in usage_buf:
                                    frame, usage_buf = usage_buf.split("\n\n", 1)
                                    if '"usage"' not in frame:
                                        continue
                                    for ln in frame.split("\n"):
                                        if not ln.startswith("data:"):
                                            continue
                                        d = ln[5:].strip()
                                        if not d or d == "[DONE]":
                                            continue
                                        try:
                                            us = json.loads(d).get("usage") or {}
                                            if us.get("prompt_tokens") is not None:
                                                pt = us["prompt_tokens"]
                                            if us.get("completion_tokens") is not None:
                                                ct = us["completion_tokens"]
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                    except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout) as e:
                        # Upstream dropped the socket mid-stream (GPU node cold-start /
                        # recycle / overload → httpx "incomplete chunked read"). It's an
                        # infra hiccup, not a client bug. If we haven't yielded a byte yet
                        # we can still fail over to another upstream; once partial output
                        # is out we can't retry, so close the SSE cleanly with a terminator
                        # instead of letting the protocol error break the client's stream.
                        _mark_health(app, endpoint_id, u["id"], False, error=str(e))
                        if not sent_any:
                            last_err = f"{u['name']}: {type(e).__name__} (mid-stream, pre-byte)"
                            continue
                        yield b"data: [DONE]\n\n"
                        finished = True
                        await _finish(request_id, "completed", status_code=r.status_code,
                                      latency_ms=int((time.perf_counter() - t0) * 1000),
                                      ttft_ms=ttft_ms, pt=pt, ct=ct, upstream=u["name"],
                                      error=f"upstream closed mid-stream: {type(e).__name__}")
                        return
                    finished = True
                    await _finish(request_id, "completed", status_code=r.status_code,
                                  latency_ms=int((time.perf_counter() - t0) * 1000),
                                  ttft_ms=ttft_ms, pt=pt, ct=ct, upstream=u["name"])
                    return
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                _mark_health(app, endpoint_id, u["id"], False, error=str(e))
                last_err = f"{u['name']}: {type(e).__name__}"
                continue
        finished = True
        err = json.dumps({"error": {"message": f"all upstreams failed: {last_err}", "type": "upstream_error"}})
        yield f"data: {err}\n\n".encode()
        await _finish(request_id, "failed", status_code=502, error=last_err)
    finally:
        if gate_held and gate is not None:
            await gate.release(token)
        live.pop(request_id, None)
        if not finished:
            # disconnect/manual cancel — record without awaiting (may be mid-aclose)
            asyncio.create_task(_finish(request_id, "cancelled", status_code=499, error="client disconnected"))


async def _route(app, endpoint_name: str, alias: str):
    """Resolve the endpoint + candidate upstreams for `alias` and their keys (NO DB row).
    Returns (endpoint_id, candidates, timeout_s, max_conc). Raises 404 if the endpoint or
    model alias is unknown/disabled. Split out of _prepare so the streaming-passthrough path
    can decide single-vs-multi upstream BEFORE committing a request row."""
    async with session_factory()() as s:
        ep = (await s.execute(select(ProxyEndpoint).where(ProxyEndpoint.name == endpoint_name))).scalar_one_or_none()
        if ep is None or not ep.enabled:
            raise HTTPException(status_code=404, detail={"error": f"proxy endpoint '{endpoint_name}' not found or disabled"})
        cfg = ep.config or {}
        endpoint_id = ep.id
        candidates = _select_candidates(app, endpoint_id, cfg, alias)
        if not candidates:
            raise HTTPException(status_code=404, detail={"error": f"model '{alias}' is not served by endpoint '{endpoint_name}'"})
        genv = await load_global_env(s)
        for u in candidates:
            u["_key"] = _resolve_key(u, genv)
        max_conc = int(cfg.get("max_concurrency") or 0)
        timeout_s = float(cfg.get("timeout_s") or DEFAULT_TIMEOUT_S)
    return endpoint_id, candidates, timeout_s, max_conc


async def _resolve_speech_route(app, endpoint_name: str, alias: str):
    """Like _route, but also resolves the endpoint's stt_callback (base_url/model +
    decrypted key) for the TTS CER/WER round-trip. Returns
    (endpoint_id, candidates, timeout_s, max_conc, stt) where stt is None if unset/disabled."""
    async with session_factory()() as s:
        ep = (await s.execute(select(ProxyEndpoint).where(ProxyEndpoint.name == endpoint_name))).scalar_one_or_none()
        if ep is None or not ep.enabled:
            raise HTTPException(status_code=404, detail={"error": f"proxy endpoint '{endpoint_name}' not found or disabled"})
        cfg = ep.config or {}
        endpoint_id = ep.id
        candidates = _select_candidates(app, endpoint_id, cfg, alias)
        if not candidates:
            raise HTTPException(status_code=404, detail={"error": f"model '{alias}' is not served by endpoint '{endpoint_name}'"})
        genv = await load_global_env(s)
        for u in candidates:
            u["_key"] = _resolve_key(u, genv)
        max_conc = int(cfg.get("max_concurrency") or 0)
        timeout_s = float(cfg.get("timeout_s") or DEFAULT_TIMEOUT_S)
        stt = None
        sc = cfg.get("stt_callback")
        if isinstance(sc, dict) and sc.get("enabled", True) and sc.get("base_url") and sc.get("model"):
            stt = {"base_url": sc["base_url"], "model": sc["model"], "_key": _resolve_key(sc, genv)}
        capture = _resolve_capture(cfg)
    return endpoint_id, candidates, timeout_s, max_conc, stt, capture


async def _insert_request_row(request_id, endpoint_id, user, alias, is_stream) -> None:
    async with session_factory()() as s:
        s.add(ProxyRequest(
            id=request_id, endpoint_id=endpoint_id, owner_id=getattr(user, "id", None),
            model=alias, status="queued", is_stream=is_stream,
            created_at=datetime.now(timezone.utc),
        ))
        await s.commit()


async def _prepare(app, endpoint_name: str, alias: str, user: User, is_stream: bool):
    """Resolve routing (via _route) + record a queued ProxyRequest. Returns
    (endpoint_id, candidates, timeout_s, max_conc, request_id)."""
    request_id = f"pxr-{_secrets.token_hex(8)}"
    endpoint_id, candidates, timeout_s, max_conc = await _route(app, endpoint_name, alias)
    await _insert_request_row(request_id, endpoint_id, user, alias, is_stream)
    return endpoint_id, candidates, timeout_s, max_conc, request_id


def _register_live(app, request_id, endpoint_id, alias, user, cancel_ev, is_stream) -> None:
    _live(app)[request_id] = {
        "cancel": cancel_ev, "state": "queued", "endpoint_id": endpoint_id,
        "model": alias, "upstream": None, "created_at": time.time(),
        "owner": getattr(user, "username", "?"), "is_stream": is_stream, "id": request_id,
    }


async def _handle(request: Request, user: User, endpoint_name: str, payload: dict, upstream_path: str) -> Response:
    """Classic path: a fully-buffered payload dict. Routes + records the row, then dispatches
    (supports failover across candidates). Used by embeddings and by the buffered fallbacks
    of _handle_ingest."""
    app = request.app
    alias = payload.get("model")
    if not isinstance(alias, str) or not alias.strip():
        raise HTTPException(status_code=400, detail={"error": "missing 'model' in request body"})
    alias = alias.strip()
    is_stream = bool(payload.get("stream"))
    endpoint_id, candidates, timeout_s, max_conc, request_id = await _prepare(app, endpoint_name, alias, user, is_stream)
    return await _dispatch_buffered(app, request, user, endpoint_id, candidates, alias, is_stream,
                                    timeout_s, max_conc, request_id, payload, upstream_path)


async def _dispatch_buffered(app, request, user, endpoint_id, candidates, alias, is_stream,
                             timeout_s, max_conc, request_id, payload, upstream_path) -> Response:
    """Forward a fully-buffered payload dict — the classic engine (_stream / _do_unary),
    which supports failover across multiple candidate upstreams."""
    gate = _get_gate(app, endpoint_id, max_conc)
    cancel_ev = asyncio.Event()
    _register_live(app, request_id, endpoint_id, alias, user, cancel_ev, is_stream)
    if is_stream:
        sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-Id": request_id}
        # Response headers must be flushed before the body, but the streaming
        # forward only picks an upstream once it starts (and may fail over before
        # the first byte). So we can only name the upstream up front when there's
        # exactly one candidate — then it's guaranteed accurate. With failover
        # candidates, the actual upstream is recorded on the ProxyRequest row.
        if len(candidates) == 1:
            sse_headers["X-Upstream-Url"] = candidates[0]["base_url"]
            sse_headers["X-Upstream-Name"] = candidates[0]["name"]
        return StreamingResponse(
            _stream(app, request_id, endpoint_id, candidates, alias, payload, upstream_path, timeout_s, gate, cancel_ev),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    return await _unary(app, request, request_id,
                        lambda: _do_unary(app, endpoint_id, candidates, alias, payload, upstream_path, timeout_s),
                        gate, cancel_ev)


# ---------- streaming request-body passthrough (big-payload latency fix) -----
#
# A chat/completions body carrying a base64 image can be ~1 MB. Declaring the route
# `payload: dict` makes FastAPI BUFFER + parse the whole body before we forward, so the
# client→gateway upload and the gateway→upstream relay run SERIALLY (measured: +~500 ms
# TTFT on a ~960 KB image through a WAN gateway hop). Instead we read the body as a stream,
# pull just enough to find + rewrite the top-level `model` alias, then relay the remainder
# to the upstream AS IT ARRIVES — overlapping the two uploads. Kill-switch:
# PROXY_STREAM_PASSTHROUGH=0 reverts to the classic buffered path.

_STREAM_PASSTHROUGH_CAP = int(os.environ.get("PROXY_STREAM_HEAD_CAP", "65536") or "65536")


def _scan_top_level_model(buf: bytes):
    """Find the top-level "model" string value in a (possibly PARTIAL) JSON object body.
    Returns ("found", vstart, vend, alias) with byte offsets of the value (excluding the
    surrounding quotes), ("absent",) if the root object closed / body isn't a JSON object
    without a top-level model, or ("need_more",) if the buffer ends before we can decide.
    base64 image data can't contain a double-quote, so a depth-1 '"model"' key is
    unambiguous; we still depth-gate + skip string contents to be safe."""
    n = len(buf)
    i = 0
    while i < n and buf[i] in b" \t\r\n":
        i += 1
    if i >= n:
        return ("need_more",)
    if buf[i] != 0x7b:  # '{'
        return ("absent",)
    i += 1
    depth = 1
    expect_key = True
    while i < n:
        c = buf[i]
        if c in b" \t\r\n":
            i += 1; continue
        if c == 0x7d:  # '}'
            depth -= 1; i += 1; expect_key = False
            if depth == 0:
                return ("absent",)
            continue
        if c == 0x5d:  # ']'
            depth -= 1; i += 1; expect_key = False; continue
        if c == 0x2c:  # ','
            i += 1
            if depth == 1:
                expect_key = True
            continue
        if c == 0x3a:  # ':'
            i += 1; continue
        if c in b"[{":
            depth += 1; i += 1; expect_key = False; continue
        if c == 0x22:  # '"' — a string token
            j = i + 1
            while j < n:
                if buf[j] == 0x5c:  # backslash escape
                    j += 2; continue
                if buf[j] == 0x22:
                    break
                j += 1
            if j >= n:
                return ("need_more",)
            s = buf[i+1:j]
            end = j + 1
            if depth == 1 and expect_key:
                k = end
                while k < n and buf[k] in b" \t\r\n":
                    k += 1
                if k >= n:
                    return ("need_more",)
                if buf[k] != 0x3a:  # not a ':' — malformed, let caller buffer + parse
                    return ("absent",)
                k += 1
                while k < n and buf[k] in b" \t\r\n":
                    k += 1
                if k >= n:
                    return ("need_more",)
                if s == b"model":
                    if buf[k] != 0x22:  # model value isn't a string
                        return ("absent",)
                    v = k + 1
                    while v < n:
                        if buf[v] == 0x5c:
                            v += 2; continue
                        if buf[v] == 0x22:
                            break
                        v += 1
                    if v >= n:
                        return ("need_more",)
                    return ("found", k+1, v, buf[k+1:v].decode("utf-8", "ignore"))
                i = k; expect_key = False; continue  # non-model key: skip to its value
            else:
                i = end; expect_key = False; continue
        i += 1; expect_key = False  # scalar char (number / true / false / null)
    return ("need_more",)


async def _ingest_body(request: Request) -> dict:
    """Read the request body, locating the top-level `model` within the first
    _STREAM_PASSTHROUGH_CAP bytes. Returns either:
      {"mode":"stream", "alias", "head", "vstart", "vend", "it"} — model found early; `it`
          is the (partially-consumed) body iterator whose remainder can be relayed live, or
      {"mode":"buffer", "payload": dict|None} — whole body buffered + parsed (classic path).
    Degrades to buffer mode on any scan/read hiccup (never streams uncertain input)."""
    it = request.stream().__aiter__()
    buf = bytearray()
    ended = False
    try:
        while True:
            st = _scan_top_level_model(bytes(buf))
            if st[0] == "found":
                return {"mode": "stream", "alias": st[3], "head": bytes(buf),
                        "vstart": st[1], "vend": st[2], "it": it}
            if st[0] == "absent" or len(buf) >= _STREAM_PASSTHROUGH_CAP:
                break
            try:
                chunk = await it.__anext__()
            except StopAsyncIteration:
                ended = True; break
            if not chunk:
                ended = True; break
            buf.extend(chunk)
    except Exception:
        pass  # fall through to buffered mode
    if not ended:
        while True:
            try:
                chunk = await it.__anext__()
            except StopAsyncIteration:
                break
            except Exception:
                break
            if chunk:
                buf.extend(chunk)
    try:
        payload = json.loads(bytes(buf)) if buf else None
    except Exception:
        payload = None
    return {"mode": "buffer", "payload": payload}


async def _forward_passthrough(app, request, user, endpoint_id, cand, alias, upstream_path,
                               head: bytes, body_it, timeout_s, max_conc, is_stream_hint) -> Response:
    """Single-upstream forward that STREAMS the request body (with `model` rewritten) to the
    upstream as it keeps arriving from the client — overlapping the client→gateway upload
    with the gateway→upstream relay. Branches the RESPONSE on the upstream's content-type
    (SSE vs JSON), so we needn't pre-know the client's `stream` flag. No failover (single
    candidate by construction); this also surfaces a non-2xx upstream (e.g. nginx 413) with
    its REAL status instead of masking it inside a 200 SSE envelope."""
    request_id = f"pxr-{_secrets.token_hex(8)}"
    await _insert_request_row(request_id, endpoint_id, user, alias, is_stream_hint)
    gate = _get_gate(app, endpoint_id, max_conc)
    cancel_ev = asyncio.Event()
    _register_live(app, request_id, endpoint_id, alias, user, cancel_ev, is_stream_hint)
    live = _live(app)
    cli = _http(app)
    url = cand["base_url"].rstrip("/") + upstream_path
    headers = {"Content-Type": "application/json"}
    if cand.get("_key"):
        headers["Authorization"] = f"Bearer {cand['_key']}"

    async def _agen():
        yield head
        async for chunk in body_it:
            yield chunk

    token = None
    gate_held = False
    handed_off = False
    try:
        if gate is not None:
            token = await gate.acquire(cancel_ev, live.get(request_id)); gate_held = True
        if cancel_ev.is_set():
            raise asyncio.CancelledError()
        if request_id in live:
            live[request_id]["state"] = "running"; live[request_id]["upstream"] = cand["name"]
        await _set_started(request_id, upstream=cand["name"])
        t0 = time.perf_counter()
        req = cli.build_request("POST", url, content=_agen(), headers=headers,
                                timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=timeout_s, pool=10.0))
        r = await cli.send(req, stream=True)
        lat = int((time.perf_counter() - t0) * 1000)
        _mark_health(app, endpoint_id, cand["id"], r.status_code < 500,
                     latency_ms=lat, error=None if r.status_code < 500 else f"HTTP {r.status_code}")
        base_hdrs = {"X-Request-Id": request_id, "X-Upstream-Url": cand["base_url"], "X-Upstream-Name": cand["name"]}
        if "text/event-stream" not in r.headers.get("content-type", ""):
            data = await r.aread()
            await r.aclose()
            if gate_held and gate is not None:
                await gate.release(token); gate_held = False
            try:
                obj = json.loads(data)
            except Exception:
                obj = {"raw": data.decode("utf-8", "ignore")}
            usage = (obj.get("usage") or {}) if isinstance(obj, dict) else {}
            ok = r.status_code < 400
            await _finish(request_id, "completed" if ok else "failed", status_code=r.status_code,
                          latency_ms=lat, pt=usage.get("prompt_tokens"), ct=usage.get("completion_tokens"),
                          upstream=cand["name"],
                          error=None if ok else (json.dumps(obj)[:500] if isinstance(obj, dict) else str(obj)[:500]))
            live.pop(request_id, None)
            return JSONResponse(obj, status_code=r.status_code, headers=base_hdrs)
        # SSE relay — hand the concurrency slot + cleanup off to the generator's finally.
        handed_off = True
        sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **base_hdrs}

        async def _relay():
            nonlocal token, gate_held
            ttft = None; pt = ct2 = None; usage_buf = ""; finished = False
            try:
                async for chunk in r.aiter_bytes():
                    if cancel_ev.is_set():
                        break
                    if ttft is None:
                        ttft = int((time.perf_counter() - t0) * 1000)
                    yield chunk
                    try:  # light usage sniff — same as _stream
                        usage_buf += chunk.decode("utf-8", "ignore")
                        while "\n\n" in usage_buf:
                            frame, usage_buf = usage_buf.split("\n\n", 1)
                            if '"usage"' not in frame:
                                continue
                            for ln in frame.split("\n"):
                                if not ln.startswith("data:"):
                                    continue
                                d = ln[5:].strip()
                                if not d or d == "[DONE]":
                                    continue
                                try:
                                    us = json.loads(d).get("usage") or {}
                                    if us.get("prompt_tokens") is not None:
                                        pt = us["prompt_tokens"]
                                    if us.get("completion_tokens") is not None:
                                        ct2 = us["completion_tokens"]
                                except Exception:
                                    pass
                    except Exception:
                        pass
                finished = True
                await _finish(request_id, "completed", status_code=r.status_code,
                              latency_ms=int((time.perf_counter() - t0) * 1000),
                              ttft_ms=ttft, pt=pt, ct=ct2, upstream=cand["name"])
            finally:
                try:
                    await r.aclose()
                except Exception:
                    pass
                if gate_held and gate is not None:
                    await gate.release(token); gate_held = False
                live.pop(request_id, None)
                if not finished:
                    asyncio.create_task(_finish(request_id, "cancelled", status_code=499, error="client disconnected"))

        return StreamingResponse(_relay(), media_type="text/event-stream", headers=sse_headers)
    except asyncio.CancelledError:
        # Cancelled while queued (manual cancel / cross-replica flush set cancel_ev,
        # caught in the gate's acquire loop) OR the client disconnected before we sent
        # anything upstream — a cancellation, not an upstream failure. Record 499.
        await _finish(request_id, "cancelled", status_code=499, error="cancelled")
        live.pop(request_id, None)
        return Response(status_code=499)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.ReadTimeout,
            httpx.RemoteProtocolError, httpx.WriteError) as e:
        _mark_health(app, endpoint_id, cand["id"], False, error=str(e))
        await _finish(request_id, "failed", status_code=502, error=f"{type(e).__name__}: {e}")
        live.pop(request_id, None)
        raise HTTPException(status_code=502, detail={"error": f"upstream failed: {type(e).__name__}"})
    finally:
        if gate_held and gate is not None and not handed_off:
            await gate.release(token)


async def _handle_ingest(request: Request, user: User, endpoint_name: str, upstream_path: str) -> Response:
    """Entry for chat/completions + completions. Streams the body straight through to a
    single upstream when possible (overlap); otherwise buffers + uses the classic failover
    path. PROXY_STREAM_PASSTHROUGH=0 forces the classic path for all requests."""
    app = request.app
    if os.environ.get("PROXY_STREAM_PASSTHROUGH", "1") == "0":
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail={"error": "invalid JSON body"})
        return await _handle(request, user, endpoint_name, payload, upstream_path)

    ing = await _ingest_body(request)
    if ing["mode"] == "buffer":
        payload = ing["payload"]
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail={"error": "invalid JSON body"})
        return await _handle(request, user, endpoint_name, payload, upstream_path)

    alias = (ing["alias"] or "").strip()
    it = ing["it"]
    if not alias:  # defensive: unusable alias — buffer the rest, classic path
        rest = bytearray(ing["head"])
        async for c in it:
            rest.extend(c)
        try:
            payload = json.loads(bytes(rest))
        except Exception:
            raise HTTPException(status_code=400, detail={"error": "invalid JSON body"})
        return await _handle(request, user, endpoint_name, payload, upstream_path)

    endpoint_id, candidates, timeout_s, max_conc = await _route(app, endpoint_name, alias)
    # The passthrough fast path only rewrites the `model` value in-place in the byte
    # stream; it can't inject new top-level keys. An upstream with extra_body (e.g.
    # OpenRouter provider pinning) must go through the buffered path so extra_body is
    # merged into the parsed payload.
    if len(candidates) == 1 and not candidates[0].get("extra_body"):
        cand = candidates[0]
        real = cand["models"][alias]
        head = ing["head"]
        new_head = head[:ing["vstart"]] + real.encode("utf-8") + head[ing["vend"]:]
        flat = head.replace(b" ", b"").replace(b"\n", b"").replace(b"\t", b"").replace(b"\r", b"")
        is_stream_hint = b'"stream":true' in flat[:4096]
        return await _forward_passthrough(app, request, user, endpoint_id, cand, alias, upstream_path,
                                          new_head, it, timeout_s, max_conc, is_stream_hint)

    # multiple candidates → preserve failover: buffer the rest, parse, classic dispatch
    rest = bytearray(ing["head"])
    async for c in it:
        rest.extend(c)
    try:
        payload = json.loads(bytes(rest))
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "invalid JSON body"})
    is_stream = bool(payload.get("stream"))
    request_id = f"pxr-{_secrets.token_hex(8)}"
    await _insert_request_row(request_id, endpoint_id, user, alias, is_stream)
    return await _dispatch_buffered(app, request, user, endpoint_id, candidates, alias, is_stream,
                                    timeout_s, max_conc, request_id, payload, upstream_path)


# ---------- background job queue (off-path eval + capture) -------------------
# TTS CER/WER evals + drift-sample uploads run OFF the request path. Rather than a
# per-call cap that DROPS work under load, they go through one bounded queue drained by
# a worker pool: bursts buffer and run in parallel, and only a genuinely full queue sheds
# (backpressure). Scale parallelism with PROXY_BG_WORKERS, backlog with PROXY_BG_QUEUE_MAX.
_BG_WORKERS = int(os.environ.get("PROXY_BG_WORKERS", "8") or "8")
_BG_QUEUE_MAX = int(os.environ.get("PROXY_BG_QUEUE_MAX", "1000") or "1000")
_bg_queue: "Optional[asyncio.Queue]" = None
_bg_started = False


async def _bg_worker() -> None:
    assert _bg_queue is not None
    while True:
        factory = await _bg_queue.get()
        try:
            await factory()
        except Exception as e:  # noqa: BLE001 — a bad job must never kill the worker
            logger.warning("background job failed: %s", e)
        finally:
            _bg_queue.task_done()
            from . import metrics as _metrics
            _metrics.set_bg_queue_depth(_bg_queue.qsize())


def _ensure_bg_workers() -> None:
    """Lazily start the worker pool on first use (runs on the request's event loop).
    Synchronous → atomic on the single-threaded loop, so no double-start race."""
    global _bg_queue, _bg_started
    if _bg_started:
        return
    _bg_queue = asyncio.Queue(maxsize=max(1, _BG_QUEUE_MAX))
    for _ in range(max(1, _BG_WORKERS)):
        asyncio.create_task(_bg_worker())
    _bg_started = True


def _submit_bg(factory) -> bool:
    """Enqueue a zero-arg coroutine factory for a background worker. Returns False (shed)
    when the queue is full — never blocks the caller."""
    _ensure_bg_workers()
    assert _bg_queue is not None
    try:
        _bg_queue.put_nowait(factory)
    except asyncio.QueueFull:
        return False
    from . import metrics as _metrics
    _metrics.set_bg_queue_depth(_bg_queue.qsize())
    return True


# ---------- TTS speech proxy + async CER/WER round-trip ----------------------
# The TTS API returns raw audio (audio/pcm headerless, or audio/wav with response_format=wav;
# this engine is 16-bit mono 24 kHz). We relay it verbatim, and — off the response path — tee
# the bytes, transcribe them through the configured whisper STT, and score CER/WER vs the input
# text into Prometheus. Never blocks or alters the client response.
_TTS_PCM_SR = int(os.environ.get("TTS_PCM_SAMPLE_RATE", "24000") or "24000")
_TTS_EVAL_CAP_BYTES = int(os.environ.get("TTS_EVAL_MAX_BYTES", str(50 * 1024 * 1024)))


def _pcm_to_wav(pcm: bytes, sr: int, ch: int, bits: int) -> bytes:
    """Wrap raw PCM in a correct 44-byte WAV header (so a strict STT decoder accepts it)."""
    byte_rate = sr * ch * bits // 8
    block = ch * bits // 8
    dl = len(pcm)
    return (b"RIFF" + struct.pack("<I", 36 + dl) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, sr, byte_rate, block, bits)
            + b"data" + struct.pack("<I", dl) + pcm)


def _parse_wav(data: bytes):
    """(sr, ch, bits, pcm) from a RIFF/WAVE buffer, or None. Takes the data chunk to EOF —
    the engine streams WAV with a bogus 0x7FFFFFFF size, so the declared length is ignored."""
    try:
        i, n = 12, len(data)
        sr, ch, bits, pcm = _TTS_PCM_SR, 1, 16, None
        while i + 8 <= n:
            cid = data[i:i + 4]
            size = struct.unpack("<I", data[i + 4:i + 8])[0]
            i += 8
            if cid == b"fmt " and i + 16 <= n:
                _fmt, ch, sr, _br, _ba, bits = struct.unpack("<HHIIHH", data[i:i + 16])
                i += size
            elif cid == b"data":
                pcm = data[i:]
                break
            else:
                i += size
            if size % 2:
                i += 1  # chunks are word-aligned
        return (sr, ch, bits, pcm) if pcm is not None else None
    except Exception:
        return None


def _audio_to_wav(data: bytes, content_type: str = "") -> bytes:
    """Normalize TTS output to a clean WAV for the STT round-trip. RIFF/WAVE → re-wrap with a
    correct header; anything else → treat as raw PCM at the engine's 16-bit/mono/24 kHz."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        p = _parse_wav(data)
        if p:
            sr, ch, bits, pcm = p
            return _pcm_to_wav(pcm, sr, ch, bits)
    return _pcm_to_wav(data, _TTS_PCM_SR, 1, 16)


def _cer_wer(ref: str, hyp: str):
    """(cer, wer) via jiwer, on case/punctuation-normalized text; (None, None) if unscoreable.
    For CJK (no word spaces) WER is coarse (~1.0 on any mismatch) — CER is the real signal."""
    ref = (ref or "").strip()
    hyp = (hyp or "").strip()
    if not ref:
        return None, None
    try:
        import re
        import jiwer
    except Exception:
        return None, None
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()
    rn, hn = _norm(ref), _norm(hyp)
    if not rn:
        return None, None
    try:
        return float(jiwer.cer(rn, hn)), float(jiwer.wer(rn, hn))
    except Exception:
        return None, None


async def _tts_eval_async(app, endpoint_id: str, model_alias: str, voice: str,
                          input_text: str, audio_bytes: bytes, content_type: str, stt: dict,
                          capture: Optional[dict] = None, request_id: Optional[str] = None,
                          sample_id: Optional[str] = None) -> None:
    """Fire-and-forget: transcribe generated audio via the STT callback, score CER/WER,
    record to Prometheus, and (if a capture threshold is crossed) persist the clip. Runs on
    a background worker (bounded by the pool); never raises."""
    from . import metrics as _metrics
    try:
        wav = _audio_to_wav(audio_bytes, content_type)
        cli = _http(app)
        url = stt["base_url"].rstrip("/") + "/audio/transcriptions"
        # Mark this as the TTS round-trip's own transcription so, if the STT callback
        # points at a capture-enabled STT proxy on THIS gateway, it doesn't also file the
        # internal call as an STT drift sample (would double-capture the same audio).
        headers = {"X-SGPU-TTS-Eval": "1"}
        if stt.get("_key"):
            headers["Authorization"] = f"Bearer {stt['_key']}"
        r = await cli.post(url, data={"model": stt["model"], "response_format": "json"},
                           files={"file": ("tts.wav", wav, "audio/wav")}, headers=headers,
                           timeout=httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0))
        if r.status_code != 200:
            _metrics.observe_tts_eval_outcome(endpoint_id, "stt_error")
            return
        try:
            hyp = (r.json() or {}).get("text") or ""
        except Exception:
            hyp = r.text or ""
        cer, wer = _cer_wer(input_text, hyp)
        if cer is None:
            _metrics.observe_tts_eval_outcome(endpoint_id, "stt_error")
            return
        _metrics.observe_tts_eval(endpoint_id, model_alias, voice, cer, wer)
        # Capture bad syntheses (CER or WER over threshold) — save the WAV we already built.
        if capture:
            ct_thr, wt_thr = capture.get("cer_threshold"), capture.get("wer_threshold")
            if (ct_thr is not None and cer > ct_thr) or (wt_thr is not None and wer > wt_thr):
                sid = sample_id or request_id
                _fire_capture(app, endpoint_id, "tts", wav, "wav",
                              {"kind": "tts", "model": model_alias, "voice": voice, "input": input_text,
                               "hypothesis": hyp, "cer": round(cer, 4), "wer": round(wer, 4),
                               "request_id": sid, "proxy_request_id": request_id,
                               "ts": datetime.now(timezone.utc).isoformat()},
                              capture, sid)
    except Exception as e:  # noqa: BLE001 — eval is best-effort, never touches the response
        logger.warning("tts CER/WER eval failed (endpoint=%s): %s", endpoint_id, e)
        _metrics.observe_tts_eval_outcome(endpoint_id, "stt_error")


def _fire_tts_eval(app, endpoint_id: str, alias: str, ev: dict, audio: bytes, content_type: str) -> None:
    """Queue the CER/WER round-trip if a callback + input text + audio are present."""
    stt = ev.get("stt")
    text = ev.get("input")
    if not stt or not text or not audio:
        return
    from . import metrics as _metrics
    if len(audio) > _TTS_EVAL_CAP_BYTES:
        _metrics.observe_tts_eval_outcome(endpoint_id, "skipped")
        return
    ok = _submit_bg(lambda: _tts_eval_async(app, endpoint_id, alias, ev.get("voice", ""),
                                            text, bytes(audio), content_type, stt,
                                            ev.get("capture"), ev.get("request_id"), ev.get("sample_id")))
    if not ok:
        _metrics.observe_tts_eval_outcome(endpoint_id, "skipped")


# ---------- drift-sample capture (threshold-triggered audio → storage) -------
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]")


def _trace_sample_id(request: Request, fallback: str) -> str:
    """Storage key stem for a captured sample = the request's X-Request-ID so it lines up
    with tracing/logs. Honours an inbound X-Request-ID (an external tracer's id); else the
    proxy's own `pxr-…` id (which is what the response + access log carry). Sanitized —
    the inbound value is client-controlled, so strip it to a safe, bounded key segment."""
    raw = (request.headers.get("x-request-id") or "").strip()
    if not raw:
        return fallback
    safe = _SAFE_ID_RE.sub("-", raw)[:80].strip("-.")
    return safe or fallback


def _resolve_capture(cfg: dict) -> Optional[dict]:
    """The endpoint's capture config if enabled + a storage target is set, else None."""
    c = cfg.get("capture")
    if isinstance(c, dict) and c.get("enabled") and c.get("storage_id"):
        return c
    return None


async def _resolve_capture_cfg(endpoint_name: str) -> Optional[dict]:
    """Load just the capture config for an endpoint (used by the STT path, which doesn't
    otherwise need the full endpoint row)."""
    async with session_factory()() as s:
        ep = (await s.execute(select(ProxyEndpoint).where(ProxyEndpoint.name == endpoint_name))).scalar_one_or_none()
        return _resolve_capture(ep.config or {}) if ep is not None else None


async def _capture_sample_async(app, endpoint_id: str, kind: str, ext: str,
                                audio_bytes: bytes, meta: dict, capture: dict, sample_id: str) -> None:
    """Persist audio + a JSON sidecar to the configured storage backend. Runs on a background
    worker; boto3/SFTP puts go through a thread so they never block the event loop, best-effort."""
    from . import metrics as _metrics
    try:
        from .storage_api import Storage
        from .storage_backends import resolve_backend
        async with session_factory()() as s:
            row = await s.get(Storage, capture["storage_id"])
        if row is None:
            logger.warning("capture: storage %s not found (endpoint=%s)", capture.get("storage_id"), endpoint_id)
            _metrics.observe_capture(endpoint_id, kind, "error")
            return
        backend = resolve_backend(row)
        prefix = (capture.get("prefix") or "").strip()
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        base = f"{prefix}{endpoint_id}/{day}/{sample_id}"
        await asyncio.to_thread(backend.put_bytes, f"{base}.{ext}", audio_bytes)
        await asyncio.to_thread(backend.put_bytes, f"{base}.json",
                                json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
        _metrics.observe_capture(endpoint_id, kind, "saved")
    except Exception as e:  # noqa: BLE001 — capture is best-effort, never touches the response
        logger.warning("drift-sample capture failed (endpoint=%s kind=%s): %s", endpoint_id, kind, e)
        _metrics.observe_capture(endpoint_id, kind, "error")


def _fire_capture(app, endpoint_id: str, kind: str, audio: bytes, ext: str,
                  meta: dict, capture: Optional[dict], sample_id: Optional[str]) -> None:
    """Queue a threshold-crossing sample save (if a storage target is configured)."""
    if not capture or not audio or not capture.get("storage_id"):
        return
    from . import metrics as _metrics
    if len(audio) > _TTS_EVAL_CAP_BYTES:
        _metrics.observe_capture(endpoint_id, kind, "skipped")
        return
    ok = _submit_bg(lambda: _capture_sample_async(app, endpoint_id, kind, ext, bytes(audio),
                                                  meta, capture, sample_id or _secrets.token_hex(8)))
    if not ok:
        _metrics.observe_capture(endpoint_id, kind, "skipped")


async def _do_speech(app, endpoint_id: str, candidates: list[dict], alias: str,
                     payload: dict, timeout_s: float, ev: dict) -> dict:
    """Non-streaming TTS forward. Returns the binary audio as raw_body (relayed verbatim) and
    fires the async CER/WER eval. Failover on connect error / 5xx."""
    cli = _http(app)
    last_err = "no upstream"
    for u in candidates:
        body = {**payload, **(u.get("extra_body") or {}), "model": u["models"][alias]}
        body.pop("stream", None)
        headers = {"Content-Type": "application/json"}
        if u.get("_key"):
            headers["Authorization"] = f"Bearer {u['_key']}"
        url = u["base_url"].rstrip("/") + "/audio/speech"
        t0 = time.perf_counter()
        try:
            r = await cli.post(url, json=body, headers=headers,
                               timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=timeout_s, pool=10.0))
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError) as e:
            _mark_health(app, endpoint_id, u["id"], False, error=str(e))
            last_err = f"{u['name']}: {type(e).__name__}"
            continue
        lat = int((time.perf_counter() - t0) * 1000)
        if r.status_code >= 500:
            _mark_health(app, endpoint_id, u["id"], False, latency_ms=lat, error=f"HTTP {r.status_code}")
            last_err = f"{u['name']}: HTTP {r.status_code}"
            continue
        _mark_health(app, endpoint_id, u["id"], True, latency_ms=lat)
        audio = r.content
        ct = r.headers.get("content-type") or "application/octet-stream"
        res = {"upstream": u["name"], "upstream_url": u["base_url"], "status_code": r.status_code,
               "latency_ms": lat, "pt": None, "ct": None, "avg_logprob": None}
        if r.status_code >= 400:
            try:
                res["data"] = r.json()
            except Exception:
                res["data"] = {"error": r.text[:500]}
            return res
        _fire_tts_eval(app, endpoint_id, alias, ev, audio, ct)
        res["raw_body"] = audio
        res["media_type"] = ct
        return res
    raise HTTPException(status_code=502, detail={"error": f"all upstreams failed: {last_err}"})


async def _stream_speech(app, request_id: str, endpoint_id: str, candidates: list[dict], alias: str,
                         payload: dict, timeout_s: float, gate, cancel_ev: asyncio.Event, ev: dict) -> Response:
    """Streaming TTS forward: relays the upstream binary audio as it arrives and tees it into a
    bounded buffer for the async CER/WER eval fired on completion. Failover before the first byte."""
    live = _live(app)
    cli = _http(app)
    token = None
    gate_held = False
    handed_off = False
    try:
        if gate is not None:
            token = await gate.acquire(cancel_ev, live.get(request_id)); gate_held = True
        if cancel_ev.is_set():
            raise asyncio.CancelledError()
        if request_id in live:
            live[request_id]["state"] = "running"
        await _set_started(request_id)
        last_err = "no upstream"
        chosen = None
        for u in candidates:
            body = {**payload, **(u.get("extra_body") or {}), "model": u["models"][alias], "stream": True}
            headers = {"Content-Type": "application/json"}
            if u.get("_key"):
                headers["Authorization"] = f"Bearer {u['_key']}"
            url = u["base_url"].rstrip("/") + "/audio/speech"
            t0 = time.perf_counter()
            try:
                req = cli.build_request("POST", url, json=body, headers=headers,
                                        timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=timeout_s, pool=10.0))
                r = await cli.send(req, stream=True)
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                _mark_health(app, endpoint_id, u["id"], False, error=str(e))
                last_err = f"{u['name']}: {type(e).__name__}"
                continue
            lat = int((time.perf_counter() - t0) * 1000)
            if r.status_code >= 500:
                _mark_health(app, endpoint_id, u["id"], False, latency_ms=lat, error=f"HTTP {r.status_code}")
                last_err = f"{u['name']}: HTTP {r.status_code}"
                await r.aclose()
                continue
            _mark_health(app, endpoint_id, u["id"], True, latency_ms=lat)
            chosen = (r, u, t0, lat)
            break
        if chosen is None:
            await _finish(request_id, "failed", status_code=502, error=last_err)
            live.pop(request_id, None)
            raise HTTPException(status_code=502, detail={"error": f"all upstreams failed: {last_err}"})

        r, u, t0, lat = chosen
        if request_id in live:
            live[request_id]["upstream"] = u["name"]
        await _set_started(request_id, upstream=u["name"])
        ct = r.headers.get("content-type") or "application/octet-stream"
        base_hdrs = {"X-Request-Id": request_id, "X-Upstream-Url": u["base_url"], "X-Upstream-Name": u["name"]}
        if r.status_code >= 400:  # upstream client error → surface it as JSON, don't stream
            body = await r.aread()
            await r.aclose()
            if gate_held and gate is not None:
                await gate.release(token); gate_held = False
            await _finish(request_id, "failed", status_code=r.status_code, latency_ms=lat,
                          upstream=u["name"], error=body[:500].decode("utf-8", "ignore"))
            live.pop(request_id, None)
            try:
                obj = json.loads(body)
            except Exception:
                obj = {"error": body.decode("utf-8", "ignore")[:500]}
            return JSONResponse(obj, status_code=r.status_code, headers=base_hdrs)

        handed_off = True
        hdrs = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **base_hdrs}

        async def _relay():
            nonlocal token, gate_held
            ttft = None
            finished = False
            truncated = False
            buf = bytearray()
            try:
                async for chunk in r.aiter_bytes():
                    if cancel_ev.is_set():
                        break
                    if ttft is None:
                        ttft = int((time.perf_counter() - t0) * 1000)
                    yield chunk
                    if not truncated:
                        if len(buf) + len(chunk) <= _TTS_EVAL_CAP_BYTES:
                            buf.extend(chunk)
                        else:
                            truncated = True
                finished = True
                await _finish(request_id, "completed", status_code=r.status_code,
                              latency_ms=int((time.perf_counter() - t0) * 1000), ttft_ms=ttft, upstream=u["name"])
                if truncated:
                    from . import metrics as _metrics
                    _metrics.observe_tts_eval_outcome(endpoint_id, "skipped")
                elif buf:
                    _fire_tts_eval(app, endpoint_id, alias, ev, bytes(buf), ct)
            finally:
                try:
                    await r.aclose()
                except Exception:
                    pass
                if gate_held and gate is not None:
                    await gate.release(token); gate_held = False
                live.pop(request_id, None)
                if not finished:
                    asyncio.create_task(_finish(request_id, "cancelled", status_code=499, error="client disconnected"))

        return StreamingResponse(_relay(), media_type=ct, headers=hdrs)
    except asyncio.CancelledError:
        await _finish(request_id, "cancelled", status_code=499, error="cancelled")
        live.pop(request_id, None)
        return Response(status_code=499)
    except HTTPException:
        raise
    except (httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.WriteError) as e:
        await _finish(request_id, "failed", status_code=502, error=f"{type(e).__name__}: {e}")
        live.pop(request_id, None)
        raise HTTPException(status_code=502, detail={"error": f"upstream failed: {type(e).__name__}"})
    finally:
        if gate_held and gate is not None and not handed_off:
            await gate.release(token)


async def _handle_speech(request: Request, user: User, endpoint_name: str) -> Response:
    """TTS text→speech forward (POST /v1/audio/speech). JSON body in, binary audio out
    (stream or unary). Routes by the `model` alias; fires an async CER/WER round-trip."""
    app = request.app
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "invalid JSON body"})
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail={"error": "body must be a JSON object"})
    alias = payload.get("model")
    if not isinstance(alias, str) or not alias.strip():
        raise HTTPException(status_code=400, detail={"error": "missing 'model' in request body"})
    alias = alias.strip()
    input_text = payload["input"] if isinstance(payload.get("input"), str) else ""
    voice = payload["voice"] if isinstance(payload.get("voice"), str) else ""
    is_stream = bool(payload.get("stream"))
    endpoint_id, candidates, timeout_s, max_conc, stt, capture = await _resolve_speech_route(app, endpoint_name, alias)
    request_id = f"pxr-{_secrets.token_hex(8)}"
    ev = {"stt": stt, "input": input_text, "voice": voice, "capture": capture,
          "request_id": request_id, "sample_id": _trace_sample_id(request, request_id)}
    await _insert_request_row(request_id, endpoint_id, user, alias, is_stream)
    gate = _get_gate(app, endpoint_id, max_conc)
    cancel_ev = asyncio.Event()
    _register_live(app, request_id, endpoint_id, alias, user, cancel_ev, is_stream)
    if is_stream:
        return await _stream_speech(app, request_id, endpoint_id, candidates, alias, payload,
                                    timeout_s, gate, cancel_ev, ev)
    return await _unary(app, request, request_id,
                        lambda: _do_speech(app, endpoint_id, candidates, alias, payload, timeout_s, ev),
                        gate, cancel_ev)


async def _handle_speaker(request: Request, user: User, endpoint_name: str) -> Response:
    """GET /v1/audio/speaker — forward the upstream's speaker/voice list (endpoint-wide, not
    per-model). Cheap read: no gate / request-row tracking. Failover across enabled upstreams."""
    app = request.app
    async with session_factory()() as s:
        ep = (await s.execute(select(ProxyEndpoint).where(ProxyEndpoint.name == endpoint_name))).scalar_one_or_none()
        if ep is None or not ep.enabled:
            raise HTTPException(status_code=404, detail={"error": f"proxy endpoint '{endpoint_name}' not found or disabled"})
        cfg = ep.config or {}
        genv = await load_global_env(s)
        ups = [dict(u) for u in cfg.get("upstreams", []) if u.get("enabled", True)]
        for u in ups:
            u["_key"] = _resolve_key(u, genv)
        timeout_s = float(cfg.get("timeout_s") or DEFAULT_TIMEOUT_S)
    if not ups:
        raise HTTPException(status_code=404, detail={"error": "endpoint has no enabled upstreams"})
    cli = _http(app)
    last_err = "no upstream"
    for u in ups:
        headers = {"Authorization": f"Bearer {u['_key']}"} if u.get("_key") else {}
        url = u["base_url"].rstrip("/") + "/audio/speaker"
        try:
            r = await cli.get(url, headers=headers,
                              timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=timeout_s, pool=10.0))
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError) as e:
            last_err = f"{u['name']}: {type(e).__name__}"
            continue
        if r.status_code >= 500:
            last_err = f"{u['name']}: HTTP {r.status_code}"
            continue
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}
        return JSONResponse(data, status_code=r.status_code, headers={"X-Upstream-Name": u["name"]})
    raise HTTPException(status_code=502, detail={"error": f"all upstreams failed: {last_err}"})


def _truthy(v: object) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


async def _stream_audio(app, request, user, endpoint_id, candidates, alias, upstream_path,
                        file_name, file_bytes, content_type, form_fields, timeout_s, max_conc) -> Response:
    """Streaming audio forward (whisper SSE, `stream=true`). Buffers the (small) upload,
    POSTs the multipart with `model` rewritten, then RELAYS the upstream response. Branches
    on the upstream content-type — SSE `transcription.chunk` frames are relayed as they
    arrive; a buffered JSON body (upstream ignored `stream`, or an error) is returned as
    JSON — mirroring `_forward_passthrough`. Failover only before the first byte. Gate /
    cancel / TTFT / logprob-sniff / `_finish` match the JSON streaming path."""
    request_id = f"pxr-{_secrets.token_hex(8)}"
    await _insert_request_row(request_id, endpoint_id, user, alias, True)
    gate = _get_gate(app, endpoint_id, max_conc)
    cancel_ev = asyncio.Event()
    _register_live(app, request_id, endpoint_id, alias, user, cancel_ev, True)
    live = _live(app)
    cli = _http(app)
    token = None
    gate_held = False
    handed_off = False
    try:
        if gate is not None:
            token = await gate.acquire(cancel_ev, live.get(request_id)); gate_held = True
        if cancel_ev.is_set():
            raise asyncio.CancelledError()
        if request_id in live:
            live[request_id]["state"] = "running"
        await _set_started(request_id)

        # Open the upstream stream, failing over before any byte reaches the client.
        last_err = "no upstream"
        chosen = None
        for u in candidates:
            data = {**form_fields, "model": u["models"][alias]}
            files = {"file": (file_name, file_bytes, content_type)}
            headers = {}
            if u.get("_key"):
                headers["Authorization"] = f"Bearer {u['_key']}"
            url = u["base_url"].rstrip("/") + upstream_path
            t0 = time.perf_counter()
            try:
                req = cli.build_request("POST", url, data=data, files=files, headers=headers,
                                        timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=timeout_s, pool=10.0))
                r = await cli.send(req, stream=True)
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                _mark_health(app, endpoint_id, u["id"], False, error=str(e))
                last_err = f"{u['name']}: {type(e).__name__}"
                continue
            lat = int((time.perf_counter() - t0) * 1000)
            if r.status_code >= 500:
                _mark_health(app, endpoint_id, u["id"], False, latency_ms=lat, error=f"HTTP {r.status_code}")
                last_err = f"{u['name']}: HTTP {r.status_code}"
                await r.aclose()
                continue
            _mark_health(app, endpoint_id, u["id"], True, latency_ms=lat)
            chosen = (r, u, t0, lat)
            break
        if chosen is None:
            await _finish(request_id, "failed", status_code=502, error=last_err)
            live.pop(request_id, None)
            raise HTTPException(status_code=502, detail={"error": f"all upstreams failed: {last_err}"})

        r, u, t0, lat = chosen
        if request_id in live:
            live[request_id]["upstream"] = u["name"]
        await _set_started(request_id, upstream=u["name"])
        base_hdrs = {"X-Request-Id": request_id, "X-Upstream-Url": u["base_url"], "X-Upstream-Name": u["name"]}

        # Upstream didn't stream (ignored `stream`, or an error body) → buffer + return JSON.
        if "text/event-stream" not in r.headers.get("content-type", ""):
            body = await r.aread()
            await r.aclose()
            if gate_held and gate is not None:
                await gate.release(token); gate_held = False
            try:
                obj = json.loads(body)
            except Exception:
                obj = {"text": body.decode("utf-8", "ignore")}
            ok = r.status_code < 400
            await _finish(request_id, "completed" if ok else "failed", status_code=r.status_code,
                          latency_ms=lat, upstream=u["name"],
                          avg_logprob=(_extract_audio_logprob(obj) if ok else None),
                          error=None if ok else (json.dumps(obj)[:500] if isinstance(obj, dict) else str(obj)[:500]))
            live.pop(request_id, None)
            return JSONResponse(obj, status_code=r.status_code, headers=base_hdrs)

        # SSE relay — hand the concurrency slot + cleanup to the generator's finally.
        handed_off = True
        sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **base_hdrs}

        async def _relay():
            nonlocal token, gate_held
            ttft = None; lp = None; buf = ""; finished = False
            try:
                async for chunk in r.aiter_bytes():
                    if cancel_ev.is_set():
                        break
                    if ttft is None:
                        ttft = int((time.perf_counter() - t0) * 1000)
                    yield chunk
                    # Best-effort logprob sniff — only frames that could carry it (the
                    # `avg_logprob`/`logprobs`/`segments` guard keeps the JSON parser off
                    # the content deltas). Most whisper streams don't emit it; harmless if so.
                    try:
                        buf += chunk.decode("utf-8", "ignore")
                        while "\n\n" in buf:
                            frame, buf = buf.split("\n\n", 1)
                            if "logprob" not in frame and "segments" not in frame:
                                continue
                            for ln in frame.split("\n"):
                                if not ln.startswith("data:"):
                                    continue
                                d = ln[5:].strip()
                                if not d or d == "[DONE]":
                                    continue
                                try:
                                    got = _extract_audio_logprob(json.loads(d))
                                    if got is not None:
                                        lp = got
                                except Exception:
                                    pass
                    except Exception:
                        pass
                finished = True
                await _finish(request_id, "completed", status_code=r.status_code,
                              latency_ms=int((time.perf_counter() - t0) * 1000),
                              ttft_ms=ttft, upstream=u["name"], avg_logprob=lp)
            finally:
                try:
                    await r.aclose()
                except Exception:
                    pass
                if gate_held and gate is not None:
                    await gate.release(token); gate_held = False
                live.pop(request_id, None)
                if not finished:
                    asyncio.create_task(_finish(request_id, "cancelled", status_code=499, error="client disconnected"))

        return StreamingResponse(_relay(), media_type="text/event-stream", headers=sse_headers)
    except asyncio.CancelledError:
        await _finish(request_id, "cancelled", status_code=499, error="cancelled")
        live.pop(request_id, None)
        return Response(status_code=499)
    except HTTPException:
        raise
    except (httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.WriteError) as e:
        await _finish(request_id, "failed", status_code=502, error=f"{type(e).__name__}: {e}")
        live.pop(request_id, None)
        raise HTTPException(status_code=502, detail={"error": f"upstream failed: {type(e).__name__}"})
    finally:
        if gate_held and gate is not None and not handed_off:
            await gate.release(token)


async def _handle_audio(request: Request, user: User, endpoint_name: str, upstream_path: str) -> Response:
    """Multipart audio forward (transcriptions / translations). Routes by the `model` form
    field and rebuilds the multipart for the chosen upstream. Unary by default; when the
    caller sends `stream=true` (whisper SSE), relays the upstream stream via _stream_audio."""
    app = request.app
    form = await request.form()
    alias = form.get("model")
    if not isinstance(alias, str) or not alias.strip():
        raise HTTPException(status_code=400, detail={"error": "missing 'model' form field"})
    alias = alias.strip()
    up = form.get("file")
    if up is None or not hasattr(up, "read"):
        raise HTTPException(status_code=400, detail={"error": "missing 'file' upload (multipart/form-data)"})
    file_bytes = await up.read()
    file_name = getattr(up, "filename", None) or "audio"
    content_type = getattr(up, "content_type", None) or "application/octet-stream"
    # Pass through every other text form field (language, prompt, response_format,
    # temperature, stream, timestamp_granularities[], …) verbatim to the upstream.
    extra = {k: v for k, v in form.multi_items() if k not in ("file", "model") and isinstance(v, str)}
    is_stream = _truthy(extra.get("stream"))
    if is_stream:
        endpoint_id, candidates, timeout_s, max_conc = await _route(app, endpoint_name, alias)
        return await _stream_audio(app, request, user, endpoint_id, candidates, alias, upstream_path,
                                   file_name, file_bytes, content_type, extra, timeout_s, max_conc)
    endpoint_id, candidates, timeout_s, max_conc, request_id = await _prepare(app, endpoint_name, alias, user, False)
    # Low-logprob sample capture (best-effort) — skipped for a TTS proxy's own round-trip
    # transcription so it doesn't double-capture the same audio as an STT sample.
    capture = None if request.headers.get("x-sgpu-tts-eval") else await _resolve_capture_cfg(endpoint_name)
    gate = _get_gate(app, endpoint_id, max_conc)
    cancel_ev = asyncio.Event()
    _register_live(app, request_id, endpoint_id, alias, user, cancel_ev, False)
    sample_id = _trace_sample_id(request, request_id)  # key captures by X-Request-ID (tracing)
    return await _unary(app, request, request_id,
                        lambda: _do_unary_multipart(app, endpoint_id, candidates, alias, upstream_path,
                                                    file_name, file_bytes, content_type, extra, timeout_s,
                                                    capture=capture, request_id=request_id, sample_id=sample_id),
                        gate, cancel_ev)


# ---------- data-plane routes ------------------------------------------------

@data_router.post("/proxy/{endpoint}/v1/chat/completions")
async def proxy_chat(endpoint: str, request: Request, user: User = Depends(current_user)):
    # No `payload: dict` param on purpose — that would make FastAPI buffer the whole (possibly
    # ~1 MB image) body before we forward. _handle_ingest streams it through instead.
    return await _handle_ingest(request, user, endpoint, "/chat/completions")


@data_router.post("/proxy/{endpoint}/v1/completions")
async def proxy_completions(endpoint: str, request: Request, user: User = Depends(current_user)):
    return await _handle_ingest(request, user, endpoint, "/completions")


@data_router.post("/proxy/{endpoint}/v1/embeddings")
async def proxy_embeddings(endpoint: str, payload: dict, request: Request, user: User = Depends(current_user)):
    payload.pop("stream", None)  # embeddings are unary — never SSE
    return await _handle(request, user, endpoint, payload, "/embeddings")


@data_router.post("/proxy/{endpoint}/v1/audio/transcriptions")
async def proxy_audio_transcriptions(endpoint: str, request: Request, user: User = Depends(current_user)):
    return await _handle_audio(request, user, endpoint, "/audio/transcriptions")


@data_router.post("/proxy/{endpoint}/v1/audio/translations")
async def proxy_audio_translations(endpoint: str, request: Request, user: User = Depends(current_user)):
    return await _handle_audio(request, user, endpoint, "/audio/translations")


@data_router.post("/proxy/{endpoint}/v1/audio/speech")
async def proxy_audio_speech(endpoint: str, request: Request, user: User = Depends(current_user)):
    return await _handle_speech(request, user, endpoint)


@data_router.get("/proxy/{endpoint}/v1/audio/speaker")
async def proxy_audio_speaker(endpoint: str, request: Request, user: User = Depends(current_user)):
    return await _handle_speaker(request, user, endpoint)


@data_router.get("/proxy/{endpoint}/v1/models")
async def proxy_models(endpoint: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    ep = (await session.execute(select(ProxyEndpoint).where(ProxyEndpoint.name == endpoint))).scalar_one_or_none()
    if ep is None or not ep.enabled:
        raise HTTPException(status_code=404, detail={"error": "proxy endpoint not found or disabled"})
    aliases: set[str] = set()
    for u in (ep.config or {}).get("upstreams", []):
        if u.get("enabled", True):
            aliases.update((u.get("models") or {}).keys())
    return {"object": "list", "data": [
        {"id": a, "object": "model", "created": 0, "owned_by": endpoint} for a in sorted(aliases)
    ]}


@data_router.get("/proxy/{endpoint}/metrics")
async def proxy_metrics(endpoint: str, session: AsyncSession = Depends(get_session)):
    """Per-proxy Prometheus scrape — request counts + latency by model / upstream /
    outcome for this endpoint. Public like `/{app_id}/metrics` (pure telemetry, no
    secrets); the natural sibling of the serving URL `/proxy/{name}/v1/...`."""
    from . import metrics as _metrics
    ep = (await session.execute(select(ProxyEndpoint).where(ProxyEndpoint.name == endpoint))).scalar_one_or_none()
    if ep is None:
        raise HTTPException(status_code=404, detail={"error": f"proxy endpoint '{endpoint}' not found"})
    return Response(content=_metrics.render_proxy(ep.id), media_type="text/plain; version=0.0.4")


# ---------- management routes (admin) ----------------------------------------

async def _owner_username(session: AsyncSession, owner_id: int) -> str:
    u = await session.get(User, owner_id)
    return u.username if u else "?"


@router.get("", response_model=list[ProxyEndpointRecord])
async def list_proxies(request: Request, user: User = Depends(require_admin),  # noqa: ARG001
                       session: AsyncSession = Depends(get_session)):
    rows = (await session.execute(select(ProxyEndpoint).order_by(ProxyEndpoint.created_at.desc()))).scalars().all()
    owners = {r.owner_id for r in rows}
    names: dict[int, str] = {}
    if owners:
        for u in (await session.execute(select(User).where(User.id.in_(owners)))).scalars().all():
            names[u.id] = u.username
    out = []
    for e in rows:
        out.append(_endpoint_record(request.app, e, names.get(e.owner_id, "?"),
                                    counts=await _endpoint_counts(request.app, e.id)))
    return out


@router.post("", response_model=ProxyEndpointRecord)
async def create_proxy(req: CreateProxyRequest, request: Request, user: User = Depends(require_admin),
                       session: AsyncSession = Depends(get_session)):
    name = (req.name or "").strip().lower()
    if not _NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="name must be lowercase letters/digits/._- (<=64 chars)")
    if (await session.execute(select(ProxyEndpoint).where(ProxyEndpoint.name == name))).scalar_one_or_none():
        raise HTTPException(status_code=400, detail=f"a proxy named '{name}' already exists")
    cfg = {
        "max_concurrency": max(0, int(req.max_concurrency)),
        "timeout_s": max(1, int(req.timeout_s)),
        "upstreams": _build_upstreams(req.upstreams),
    }
    stt = _build_stt_callback(req.stt_callback)
    if stt:
        cfg["stt_callback"] = stt
    cap = _build_capture(req.capture)
    if cap:
        cfg["capture"] = cap
    row = ProxyEndpoint(id=f"proxy-{_secrets.token_hex(4)}", owner_id=user.id, name=name,
                        enabled=bool(req.enabled), public=bool(req.public), config=cfg,
                        created_at=datetime.now(timezone.utc))
    session.add(row)
    await session.commit()
    await session.refresh(row)
    logger.info("created proxy endpoint %s (%s) for user=%s", row.id, name, user.username)
    return _endpoint_record(request.app, row, user.username)


# Public read routes — ANY logged-in user (not just admins). Declared BEFORE the
# /{proxy_id} matcher so "/v1/proxy/public" isn't captured as proxy_id="public".
@router.get("/public", response_model=list[ProxyEndpointRecord])
async def list_public_proxies(request: Request, user: User = Depends(current_user),  # noqa: ARG001
                              session: AsyncSession = Depends(get_session)):
    """Read-only list of PUBLIC proxy endpoints for any logged-in user — the
    secret-stripped info cards (name, serving URL, model aliases). Lets non-admins
    discover proxies they're allowed to use even though management is admin-only."""
    rows = (await session.execute(
        select(ProxyEndpoint).where(ProxyEndpoint.public.is_(True)).order_by(ProxyEndpoint.created_at.desc())
    )).scalars().all()
    return [_public_endpoint_record(request.app, e) for e in rows]


@router.get("/{proxy_id}", response_model=ProxyEndpointRecord)
async def get_proxy(proxy_id: str, request: Request, user: User = Depends(require_admin),  # noqa: ARG001
                    session: AsyncSession = Depends(get_session)):
    row = await session.get(ProxyEndpoint, proxy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="proxy endpoint not found")
    return _endpoint_record(request.app, row, await _owner_username(session, row.owner_id),
                            counts=await _endpoint_counts(request.app, row.id))


@router.get("/{proxy_id}/public", response_model=ProxyEndpointRecord)
async def get_public_proxy(proxy_id: str, request: Request, user: User = Depends(current_user),  # noqa: ARG001
                           session: AsyncSession = Depends(get_session)):
    """Read-only view of ONE public proxy for any logged-in user. 404s a private
    proxy so non-admins can't probe for existence/config of non-public endpoints."""
    row = await session.get(ProxyEndpoint, proxy_id)
    if row is None or not bool(getattr(row, "public", False)):
        raise HTTPException(status_code=404, detail="proxy endpoint not found")
    return _public_endpoint_record(request.app, row)


@router.patch("/{proxy_id}", response_model=ProxyEndpointRecord)
async def update_proxy(proxy_id: str, req: UpdateProxyRequest, request: Request,
                       user: User = Depends(require_admin),  # noqa: ARG001
                       session: AsyncSession = Depends(get_session)):
    row = await session.get(ProxyEndpoint, proxy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="proxy endpoint not found")
    cfg = dict(row.config or {})
    if req.name is not None:
        n = req.name.strip().lower()
        if not _NAME_RE.match(n):
            raise HTTPException(status_code=400, detail="invalid name")
        if n != row.name and (await session.execute(select(ProxyEndpoint).where(ProxyEndpoint.name == n))).scalar_one_or_none():
            raise HTTPException(status_code=400, detail=f"a proxy named '{n}' already exists")
        row.name = n
    if req.enabled is not None:
        row.enabled = bool(req.enabled)
    if req.public is not None:
        row.public = bool(req.public)
    if req.max_concurrency is not None:
        cfg["max_concurrency"] = max(0, int(req.max_concurrency))
    if req.timeout_s is not None:
        cfg["timeout_s"] = max(1, int(req.timeout_s))
    if req.upstreams is not None:
        cfg["upstreams"] = _build_upstreams(req.upstreams, existing=cfg.get("upstreams"))
    if req.stt_callback is not None:
        stt = _build_stt_callback(req.stt_callback, existing=cfg.get("stt_callback"))
        if stt:
            cfg["stt_callback"] = stt
        else:
            cfg.pop("stt_callback", None)  # cleared (blank base_url/model)
    if req.capture is not None:
        cap = _build_capture(req.capture, existing=cfg.get("capture"))
        if cap:
            cfg["capture"] = cap
        else:
            cfg.pop("capture", None)  # cleared (no storage_id)
    row.config = cfg
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(row, "config")
    await session.commit()
    await session.refresh(row)
    return _endpoint_record(request.app, row, await _owner_username(session, row.owner_id))


@router.delete("/{proxy_id}")
async def delete_proxy(proxy_id: str, user: User = Depends(require_admin),  # noqa: ARG001
                       session: AsyncSession = Depends(get_session)):
    row = await session.get(ProxyEndpoint, proxy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="proxy endpoint not found")
    await session.delete(row)
    await session.commit()
    return {"ok": True, "id": proxy_id}


@router.get("/{proxy_id}/health", response_model=list[UpstreamHealth])
async def proxy_health(proxy_id: str, request: Request, user: User = Depends(require_admin),  # noqa: ARG001
                       session: AsyncSession = Depends(get_session)):
    row = await session.get(ProxyEndpoint, proxy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="proxy endpoint not found")
    health = _health(request.app)
    out: list[UpstreamHealth] = []
    for u in (row.config or {}).get("upstreams", []):
        h = health.get((row.id, u.get("id")))
        if h is None:
            out.append(UpstreamHealth(upstream_id=u.get("id", ""), name=u.get("name", "")))
        else:
            out.append(UpstreamHealth(
                upstream_id=u.get("id", ""), name=u.get("name", ""), alive=h.get("alive"),
                latency_ms=h.get("latency_ms"), checked_at=h.get("checked_at"), error=h.get("error"),
                stale=(time.time() - h.get("checked_at", 0)) > HEALTH_TTL_S,
            ))
    return out


@router.get("/{proxy_id}/request-facets")
async def proxy_request_facets(proxy_id: str, user: User = Depends(require_admin),  # noqa: ARG001
                               session: AsyncSession = Depends(get_session)):
    """Distinct users + upstreams seen on this endpoint — to populate the queue
    tab's filter dropdowns (server-side, so it covers the full history)."""
    row = await session.get(ProxyEndpoint, proxy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="proxy endpoint not found")
    oids = [o for o in (await session.execute(
        select(ProxyRequest.owner_id).where(ProxyRequest.endpoint_id == proxy_id).distinct()
    )).scalars().all() if o is not None]
    users: list[str] = []
    if oids:
        users = sorted(u.username for u in (await session.execute(
            select(User).where(User.id.in_(oids)))).scalars().all())
    upstreams = sorted({u.get("name") for u in (row.config or {}).get("upstreams", []) if u.get("name")})
    return {"users": users, "upstreams": upstreams}


@router.get("/{proxy_id}/requests", response_model=list[ProxyRequestRecord])
async def proxy_requests(proxy_id: str, request: Request, limit: int = 50, offset: int = 0,
                         owner: Optional[str] = None, upstream: Optional[str] = None,
                         status: Optional[str] = None, sort: str = "created", order: str = "desc",
                         request_id: Optional[str] = None,
                         user: User = Depends(require_admin),  # noqa: ARG001
                         session: AsyncSession = Depends(get_session)):
    row = await session.get(ProxyEndpoint, proxy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="proxy endpoint not found")
    live = _live(request.app)
    live_ids = {rid for rid, v in live.items() if v.get("endpoint_id") == proxy_id}
    if proxy_cluster.enabled():
        # The request ROWS are already global (shared Postgres); only the in-memory
        # "is it live right now" flag was per-replica — union the global registry so a
        # request in flight on another replica still shows live here.
        for r in await proxy_cluster.live_list_endpoint(request.app.state.redis, proxy_id):
            if r.get("id"):
                live_ids.add(r["id"])
    q = select(ProxyRequest).where(ProxyRequest.endpoint_id == proxy_id)
    if request_id:
        # Direct primary-key lookup — finds the row anywhere in the FULL history (not
        # just the 200-row page); other filters/sort are irrelevant for a unique id.
        rows = (await session.execute(q.where(ProxyRequest.id == request_id))).scalars().all()
    else:
        if owner:
            ou = await get_user_by_username(session, owner)
            q = q.where(ProxyRequest.owner_id == (ou.id if ou else -1))  # -1 matches nothing
        if upstream:
            q = q.where(ProxyRequest.upstream == upstream)
        if status:
            q = q.where(ProxyRequest.status == status)
        col = ProxyRequest.latency_ms if sort == "latency" else ProxyRequest.created_at
        direction = col.asc() if order == "asc" else col.desc()
        # latency is NULL while queued/running — keep those out of a latency sort's way
        q = q.order_by(direction.nulls_last() if sort == "latency" else direction)
        rows = (await session.execute(
            q.offset(max(0, offset)).limit(min(limit, 200))
        )).scalars().all()
    owner_ids = {r.owner_id for r in rows if r.owner_id is not None}
    owners: dict[int, str] = {}
    if owner_ids:
        for u in (await session.execute(select(User).where(User.id.in_(owner_ids)))).scalars().all():
            owners[u.id] = u.username
    return [
        ProxyRequestRecord(
            id=r.id, endpoint_id=r.endpoint_id, owner=owners.get(r.owner_id) if r.owner_id is not None else None,
            model=r.model, upstream=r.upstream, status=r.status,
            is_stream=r.is_stream, status_code=r.status_code, latency_ms=r.latency_ms,
            prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens, error_text=r.error_text,
            created_at=r.created_at.isoformat() if r.created_at else "",
            started_at=r.started_at.isoformat() if r.started_at else None,
            completed_at=r.completed_at.isoformat() if r.completed_at else None,
            live=r.id in live_ids,
        )
        for r in rows
    ]


@router.post("/{proxy_id}/requests/{req_id}/cancel")
async def cancel_proxy_request(proxy_id: str, req_id: str, request: Request,
                               user: User = Depends(require_admin)):  # noqa: ARG001
    app = request.app
    entry = _live(app).get(req_id)
    if entry is not None and entry.get("endpoint_id") == proxy_id:
        entry["cancel"].set()  # served by THIS replica — cancel directly
        if proxy_cluster.enabled():
            await proxy_cluster.publish_cancel(app.state.redis, req_id)  # + any dup holder
        return {"ok": True, "id": req_id}
    if proxy_cluster.enabled():
        # Not local — it may be in flight on another replica. Confirm it exists in the
        # global registry, then fan the cancel out via pub/sub to whoever holds it.
        rows = await proxy_cluster.live_list_endpoint(app.state.redis, proxy_id)
        if any(r.get("id") == req_id for r in rows):
            await proxy_cluster.publish_cancel(app.state.redis, req_id)
            return {"ok": True, "id": req_id}
    raise HTTPException(status_code=404, detail="request not found / already finished")


@router.post("/{proxy_id}/flush")
async def flush_proxy_queue(proxy_id: str, request: Request,
                            user: User = Depends(require_admin)):  # noqa: ARG001
    """Cancel every request still WAITING for a slot (state=queued). Already-running
    requests are left alone — same semantics as the serverless queue flush. With
    PROXY_CLUSTER=1 this spans replicas (fan out via pub/sub over the global registry)."""
    app = request.app
    live = _live(app)
    n = 0
    for entry in list(live.values()):
        if entry.get("endpoint_id") == proxy_id and entry.get("state") == "queued":
            entry["cancel"].set()
            n += 1
    if proxy_cluster.enabled():
        rows = await proxy_cluster.live_list_endpoint(app.state.redis, proxy_id)
        queued = [r.get("id") for r in rows if r.get("state") == "queued" and r.get("id")]
        for rid in queued:
            await proxy_cluster.publish_cancel(app.state.redis, rid)
        n = len(queued)  # global queued count (local ones are mirrored into it)
    return {"ok": True, "flushed": n}


@router.post("/test", response_model=TestUpstreamResponse)
async def test_upstream(req: TestUpstreamRequest, request: Request,
                        user: User = Depends(require_admin),  # noqa: ARG001
                        session: AsyncSession = Depends(get_session)):
    base = (req.base_url or "").strip().rstrip("/")
    if not base:
        raise HTTPException(status_code=400, detail="base_url is required")
    key = ""
    ref = (req.api_key_secret or "").strip()
    if ref:
        key = (await load_global_env(session)).get(ref, "")
        if not key:
            return TestUpstreamResponse(ok=False, message=f"global secret '{ref}' is not set")
    elif (req.api_key or "").strip():
        key = req.api_key.strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    cli = _http(request.app)
    t0 = time.perf_counter()
    model = (req.model or "").strip()
    mode = (req.mode or "chat").strip().lower()

    # Embedding upstreams don't speak /chat/completions — end-to-end test the
    # matching endpoint instead: a tiny "hello" embedding, validating a vector back.
    if model and mode == "embedding":
        try:
            r = await cli.post(
                base + "/embeddings",
                json={"model": model, "input": "hello"},
                headers=headers, timeout=httpx.Timeout(30.0),
            )
        except httpx.HTTPError as e:
            return TestUpstreamResponse(ok=False, message=f"network error: {e}")
        lat = int((time.perf_counter() - t0) * 1000)
        if r.status_code in (401, 403):
            return TestUpstreamResponse(ok=False, message="unauthorized — check the API key", latency_ms=lat)
        if r.status_code != 200:
            return TestUpstreamResponse(ok=False, message=f"HTTP {r.status_code}: {r.text[:160]}", latency_ms=lat)
        dim = 0
        try:
            data = (r.json().get("data") or [])
            emb = (data[0].get("embedding") if data else None) or []
            dim = len(emb) if isinstance(emb, list) else 0
        except Exception:
            pass
        if dim <= 0:
            return TestUpstreamResponse(ok=False, message=f"no embedding returned — is '{model}' an embedding model?", latency_ms=lat)
        return TestUpstreamResponse(ok=True, message=f"embedding ok ({model}): dim {dim}", latency_ms=lat, models=[model])

    # With a model, do a real end-to-end check: a tiny "hello" chat completion.
    if model:
        extra = req.extra_body if isinstance(req.extra_body, dict) else {}
        try:
            r = await cli.post(
                base + "/chat/completions",
                json={**extra, "model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 16, "stream": False},
                headers=headers, timeout=httpx.Timeout(30.0),
            )
        except httpx.HTTPError as e:
            return TestUpstreamResponse(ok=False, message=f"network error: {e}")
        lat = int((time.perf_counter() - t0) * 1000)
        if r.status_code in (401, 403):
            return TestUpstreamResponse(ok=False, message="unauthorized — check the API key", latency_ms=lat)
        if r.status_code != 200:
            return TestUpstreamResponse(ok=False, message=f"HTTP {r.status_code}: {r.text[:160]}", latency_ms=lat)
        reply = ""
        try:
            ch = (r.json().get("choices") or [])
            if ch:
                reply = (ch[0].get("message") or {}).get("content") or ch[0].get("text") or ""
        except Exception:
            pass
        reply = " ".join((reply or "").split())[:60]
        msg = f"chat ok ({model})" + (f": “{reply}”" if reply else "")
        return TestUpstreamResponse(ok=True, message=msg, latency_ms=lat, models=[model])

    # No model yet → just probe reachability via /models.
    try:
        r = await cli.get(base + "/models", headers=headers, timeout=httpx.Timeout(10.0))
    except httpx.HTTPError as e:
        return TestUpstreamResponse(ok=False, message=f"network error: {e}")
    lat = int((time.perf_counter() - t0) * 1000)
    if r.status_code in (401, 403):
        return TestUpstreamResponse(ok=False, message="unauthorized — check the API key", latency_ms=lat)
    if r.status_code != 200:
        return TestUpstreamResponse(ok=False, message=f"HTTP {r.status_code}: {r.text[:160]}", latency_ms=lat)
    models: list[str] = []
    try:
        models = [m.get("id") for m in (r.json().get("data") or []) if m.get("id")][:50]
    except Exception:
        pass
    return TestUpstreamResponse(ok=True, message=f"reachable ({len(models)} models) — add a model to chat-test", latency_ms=lat, models=models)


# ---------- cross-replica cluster loops (PROXY_CLUSTER=1) ---------------------

async def proxy_cluster_sync_loop(app) -> None:
    """Reconcile this replica's in-memory live requests to the Redis registry (so the
    admin queue view + inflight/queued counts span every replica) and renew each
    in-flight request's global concurrency-slot lease (so long streams keep their slot
    and a crashed replica's slots free themselves within the lease TTL). Off unless
    PROXY_CLUSTER=1 — then it's the ONLY thing that writes the registry, keeping the
    request hot path free of extra Redis writes."""
    if not proxy_cluster.enabled():
        logger.info("proxy cluster sync loop disabled (PROXY_CLUSTER!=1)")
        return
    redis = app.state.redis
    registered: dict[str, str] = {}  # rid -> endpoint_id we've mirrored to Redis
    app.state.proxy_cluster_registered = registered
    interval = proxy_cluster.SYNC_INTERVAL_S
    await asyncio.sleep(1)
    logger.info("proxy cluster sync loop started (interval=%ss)", interval)
    while True:
        try:
            live = _live(app)
            seen = set()
            for rid, e in list(live.items()):
                seen.add(rid)
                ep = e.get("endpoint_id") or ""
                await proxy_cluster.live_upsert(redis, rid, {
                    "id": rid, "endpoint_id": ep, "model": e.get("model"),
                    "upstream": e.get("upstream"), "state": e.get("state"),
                    "owner": e.get("owner"), "is_stream": e.get("is_stream"),
                    "created_at": e.get("created_at"), "replica": proxy_cluster.replica_id(),
                })
                registered[rid] = ep
                tok = e.get("slot_token")
                if tok:
                    await proxy_cluster.limiter_renew(redis, ep, tok)
            # Requests that ended locally since the last tick: drop their mirror now
            # (don't wait for TTL) so the global queue view stays tight.
            for rid in [r for r in registered if r not in seen]:
                await proxy_cluster.live_remove(redis, rid, registered.pop(rid))
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("proxy cluster sync tick failed")
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break


async def proxy_cancel_subscriber_loop(app) -> None:
    """Subscribe to the cross-replica cancel channel; when a cancel for a request THIS
    replica is holding arrives, set its local cancel event. Lets an admin cancel/flush
    issued on ANY replica reach the replica actually serving the request. Reconnects on
    error. Off unless PROXY_CLUSTER=1."""
    if not proxy_cluster.enabled():
        return
    redis = app.state.redis
    await asyncio.sleep(1)
    while True:
        try:
            pubsub = redis.pubsub()
            await pubsub.subscribe(proxy_cluster.CANCEL_CHANNEL)
            logger.info("proxy cancel subscriber started (channel=%s)", proxy_cluster.CANCEL_CHANNEL)
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                rid = msg.get("data")
                entry = _live(app).get(rid)
                if entry is not None and entry.get("cancel") is not None:
                    entry["cancel"].set()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("proxy cancel subscriber failed; reconnecting")
            try:
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                break


# ---------- background health loop -------------------------------------------

async def _probe(app, cli: httpx.AsyncClient, endpoint_id: str, upstream: dict, key: str) -> None:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    url = upstream.get("base_url", "").rstrip("/") + "/models"
    t0 = time.perf_counter()
    try:
        r = await cli.get(url, headers=headers, timeout=httpx.Timeout(8.0))
        lat = int((time.perf_counter() - t0) * 1000)
        if r.status_code < 500 and r.status_code not in (401, 403):
            _mark_health(app, endpoint_id, upstream["id"], True, latency_ms=lat)
        else:
            _mark_health(app, endpoint_id, upstream["id"], False, latency_ms=lat, error=f"HTTP {r.status_code}")
    except Exception as e:
        _mark_health(app, endpoint_id, upstream["id"], False, error=type(e).__name__)


async def proxy_health_loop(app) -> None:
    """Probe each enabled upstream's /models on an interval (look-ahead liveness)
    and prune old request rows. Disable with PROXY_HEALTHCHECK=0."""
    if os.environ.get("PROXY_HEALTHCHECK", "1") == "0":
        logger.info("proxy health loop disabled (PROXY_HEALTHCHECK=0)")
        return
    await asyncio.sleep(8)
    interval = max(5, int(os.environ.get("PROXY_HEALTHCHECK_INTERVAL_S", "20")))
    logger.info("proxy health loop started (interval=%ss)", interval)
    while True:
        try:
            targets: list[tuple[str, dict, str]] = []
            async with session_factory()() as s:
                eps = (await s.execute(select(ProxyEndpoint).where(ProxyEndpoint.enabled == True))).scalars().all()  # noqa: E712
                genv = await load_global_env(s)
                for ep in eps:
                    for u in (ep.config or {}).get("upstreams", []):
                        if u.get("enabled", True) and u.get("base_url"):
                            targets.append((ep.id, u, _resolve_key(u, genv)))
                if REQUEST_RETENTION_DAYS > 0:  # 0 = keep indefinitely (no prune)
                    cutoff = datetime.now(timezone.utc) - timedelta(days=REQUEST_RETENTION_DAYS)
                    await s.execute(delete(ProxyRequest).where(ProxyRequest.created_at < cutoff))
                await s.commit()
            cli = _http(app)
            for endpoint_id, u, key in targets:
                await _probe(app, cli, endpoint_id, u, key)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("proxy health loop tick failed")
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
