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
import secrets as _secrets
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
from .auth import current_user, require_admin
from .db import Base, User, get_session, session_factory
from .global_env_api import load_global_env

logger = logging.getLogger("gateway.proxy")

# Management router (admin) + data-plane router (API-key) — both included in main.
router = APIRouter(prefix="/v1/proxy", tags=["proxy"])
data_router = APIRouter(tags=["proxy-data"])

HEALTH_TTL_S = 120                 # a probe older than this is "stale/unknown"
REQUEST_RETENTION_DAYS = 7
DEFAULT_TIMEOUT_S = 600.0
import re as _re
_NAME_RE = _re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


# ---------- DB models (registered via init_db side-effect import) -----------

class ProxyEndpoint(Base):
    __tablename__ = "proxy_endpoints"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # proxy-<hex8>
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)  # the {endpoint} path segment
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
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


class CreateProxyRequest(BaseModel):
    name: str
    max_concurrency: int = 0              # 0 = unlimited (no queue)
    timeout_s: int = 600
    enabled: bool = True
    upstreams: list[UpstreamSpec] = []


class UpdateProxyRequest(BaseModel):
    name: Optional[str] = None
    max_concurrency: Optional[int] = None
    timeout_s: Optional[int] = None
    enabled: Optional[bool] = None
    upstreams: Optional[list[UpstreamSpec]] = None


class UpstreamRecord(BaseModel):
    id: str
    name: str
    base_url: str
    api_key_secret: Optional[str] = None
    has_inline_key: bool = False
    models: dict[str, str] = {}
    priority: int = 0
    enabled: bool = True


class ProxyEndpointRecord(BaseModel):
    id: str
    name: str
    enabled: bool
    max_concurrency: int
    timeout_s: int
    upstreams: list[UpstreamRecord]
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
    model: Optional[str] = None   # real upstream model to chat-test; None = just probe /models


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
        enabled=bool(u.get("enabled", True)),
    )


def _endpoint_record(app, e: ProxyEndpoint, owner_username: str) -> ProxyEndpointRecord:
    cfg = e.config or {}
    live = _live(app)
    mine = [v for v in live.values() if v.get("endpoint_id") == e.id]
    return ProxyEndpointRecord(
        id=e.id, name=e.name, enabled=bool(e.enabled),
        max_concurrency=int(cfg.get("max_concurrency") or 0),
        timeout_s=int(cfg.get("timeout_s") or 600),
        upstreams=[_upstream_record(u) for u in cfg.get("upstreams", [])],
        inflight=sum(1 for v in mine if v.get("state") == "running"),
        queued=sum(1 for v in mine if v.get("state") == "queued"),
        created_at=e.created_at.isoformat() if e.created_at else "",
        created_by=owner_username,
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
                  upstream: Optional[str] = None) -> None:
    async with session_factory()() as s:
        row = await s.get(ProxyRequest, request_id)
        if row is None:
            return
        row.status = status
        if status_code is not None:
            row.status_code = status_code
        if latency_ms is not None:
            row.latency_ms = latency_ms
        if pt is not None:
            row.prompt_tokens = pt
        if ct is not None:
            row.completion_tokens = ct
        if upstream:
            row.upstream = upstream
        if error:
            row.error_text = error[:2048]
        row.completed_at = datetime.now(timezone.utc)
        await s.commit()
        # Per-proxy Prometheus metric (served at GET /proxy/{name}/metrics).
        try:
            from . import metrics as _metrics
            _metrics.observe_proxy(
                row.endpoint_id, row.model, (upstream or row.upstream or ""), status,
                (latency_ms / 1000.0) if latency_ms is not None else None,
            )
        except Exception:
            pass


# ---------- forwarding engine ------------------------------------------------

async def _do_unary(app, endpoint_id: str, candidates: list[dict], alias: str,
                    payload: dict, upstream_path: str, timeout_s: float) -> dict:
    """Try candidates in order; failover on connect error / 5xx. Returns the
    upstream's JSON + status. Raises HTTPException(502) if all fail."""
    cli = _http(app)
    last_err = "no upstream"
    for u in candidates:
        body = {**payload, "model": u["models"][alias]}
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
        return {"upstream": u["name"], "status_code": r.status_code, "data": data,
                "latency_ms": lat, "pt": usage.get("prompt_tokens"), "ct": usage.get("completion_tokens")}
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


async def _unary(app, request: Request, request_id: str, endpoint_id: str, candidates: list[dict],
                 alias: str, payload: dict, upstream_path: str, timeout_s: float,
                 sem: Optional[asyncio.Semaphore], cancel_ev: asyncio.Event) -> Response:
    live = _live(app)

    async def work() -> Response:
        acquired = False
        try:
            if sem is not None:
                await sem.acquire()
                acquired = True
            if cancel_ev.is_set():
                raise asyncio.CancelledError()
            if request_id in live:
                live[request_id]["state"] = "running"
            await _set_started(request_id)
            res = await _do_unary(app, endpoint_id, candidates, alias, payload, upstream_path, timeout_s)
            if request_id in live:
                live[request_id]["upstream"] = res["upstream"]
            await _finish(request_id, "completed", status_code=res["status_code"],
                          latency_ms=res["latency_ms"], pt=res["pt"], ct=res["ct"], upstream=res["upstream"])
            return JSONResponse(res["data"], status_code=res["status_code"])
        finally:
            if acquired and sem is not None:
                sem.release()

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
                  sem: Optional[asyncio.Semaphore], cancel_ev: asyncio.Event):
    """SSE passthrough. Failover only before the first byte. The finally releases
    the slot, drops the live entry, and (on disconnect/cancel) marks the row
    cancelled via a detached task (no await during generator close)."""
    live = _live(app)
    cli = _http(app)
    acquired = False
    finished = False
    try:
        if sem is not None:
            await sem.acquire()
            acquired = True
        if cancel_ev.is_set():
            return
        if request_id in live:
            live[request_id]["state"] = "running"
        await _set_started(request_id)
        last_err = "no upstream"
        for u in candidates:
            body = {**payload, "model": u["models"][alias], "stream": True}
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
                    async for chunk in r.aiter_bytes():
                        if cancel_ev.is_set():
                            break
                        yield chunk
                    finished = True
                    await _finish(request_id, "completed", status_code=r.status_code,
                                  latency_ms=int((time.perf_counter() - t0) * 1000), upstream=u["name"])
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
        if acquired and sem is not None:
            sem.release()
        live.pop(request_id, None)
        if not finished:
            # disconnect/manual cancel — record without awaiting (may be mid-aclose)
            asyncio.create_task(_finish(request_id, "cancelled", status_code=499, error="client disconnected"))


async def _handle(request: Request, user: User, endpoint_name: str, payload: dict, upstream_path: str) -> Response:
    app = request.app
    alias = payload.get("model")
    if not isinstance(alias, str) or not alias.strip():
        raise HTTPException(status_code=400, detail={"error": "missing 'model' in request body"})
    alias = alias.strip()
    is_stream = bool(payload.get("stream"))
    request_id = f"pxr-{_secrets.token_hex(8)}"

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
        s.add(ProxyRequest(
            id=request_id, endpoint_id=endpoint_id, owner_id=getattr(user, "id", None),
            model=alias, status="queued", is_stream=is_stream,
            created_at=datetime.now(timezone.utc),
        ))
        await s.commit()

    sem = _get_sem(app, endpoint_id, max_conc)
    cancel_ev = asyncio.Event()
    _live(app)[request_id] = {
        "cancel": cancel_ev, "state": "queued", "endpoint_id": endpoint_id,
        "model": alias, "upstream": None, "created_at": time.time(),
        "owner": getattr(user, "username", "?"), "is_stream": is_stream, "id": request_id,
    }

    if is_stream:
        return StreamingResponse(
            _stream(app, request_id, endpoint_id, candidates, alias, payload, upstream_path, timeout_s, sem, cancel_ev),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-Id": request_id},
        )
    return await _unary(app, request, request_id, endpoint_id, candidates, alias, payload, upstream_path, timeout_s, sem, cancel_ev)


# ---------- data-plane routes ------------------------------------------------

@data_router.post("/proxy/{endpoint}/v1/chat/completions")
async def proxy_chat(endpoint: str, payload: dict, request: Request, user: User = Depends(current_user)):
    return await _handle(request, user, endpoint, payload, "/chat/completions")


@data_router.post("/proxy/{endpoint}/v1/completions")
async def proxy_completions(endpoint: str, payload: dict, request: Request, user: User = Depends(current_user)):
    return await _handle(request, user, endpoint, payload, "/completions")


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
    return [_endpoint_record(request.app, e, names.get(e.owner_id, "?")) for e in rows]


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
    row = ProxyEndpoint(id=f"proxy-{_secrets.token_hex(4)}", owner_id=user.id, name=name,
                        enabled=bool(req.enabled), config=cfg, created_at=datetime.now(timezone.utc))
    session.add(row)
    await session.commit()
    await session.refresh(row)
    logger.info("created proxy endpoint %s (%s) for user=%s", row.id, name, user.username)
    return _endpoint_record(request.app, row, user.username)


@router.get("/{proxy_id}", response_model=ProxyEndpointRecord)
async def get_proxy(proxy_id: str, request: Request, user: User = Depends(require_admin),  # noqa: ARG001
                    session: AsyncSession = Depends(get_session)):
    row = await session.get(ProxyEndpoint, proxy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="proxy endpoint not found")
    return _endpoint_record(request.app, row, await _owner_username(session, row.owner_id))


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
    if req.max_concurrency is not None:
        cfg["max_concurrency"] = max(0, int(req.max_concurrency))
    if req.timeout_s is not None:
        cfg["timeout_s"] = max(1, int(req.timeout_s))
    if req.upstreams is not None:
        cfg["upstreams"] = _build_upstreams(req.upstreams, existing=cfg.get("upstreams"))
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


@router.get("/{proxy_id}/requests", response_model=list[ProxyRequestRecord])
async def proxy_requests(proxy_id: str, request: Request, limit: int = 50,
                         user: User = Depends(require_admin),  # noqa: ARG001
                         session: AsyncSession = Depends(get_session)):
    row = await session.get(ProxyEndpoint, proxy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="proxy endpoint not found")
    live = _live(request.app)
    live_ids = {rid for rid, v in live.items() if v.get("endpoint_id") == proxy_id}
    rows = (await session.execute(
        select(ProxyRequest).where(ProxyRequest.endpoint_id == proxy_id)
        .order_by(ProxyRequest.created_at.desc()).limit(min(limit, 200))
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
    live = _live(request.app)
    entry = live.get(req_id)
    if entry is None or entry.get("endpoint_id") != proxy_id:
        raise HTTPException(status_code=404, detail="request not found / already finished")
    entry["cancel"].set()
    return {"ok": True, "id": req_id}


@router.post("/{proxy_id}/flush")
async def flush_proxy_queue(proxy_id: str, request: Request,
                            user: User = Depends(require_admin)):  # noqa: ARG001
    """Cancel every request still WAITING for a slot (state=queued). Already-running
    requests are left alone — same semantics as the serverless queue flush."""
    live = _live(request.app)
    n = 0
    for entry in list(live.values()):
        if entry.get("endpoint_id") == proxy_id and entry.get("state") == "queued":
            entry["cancel"].set()
            n += 1
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

    # With a model, do a real end-to-end check: a tiny "hello" chat completion.
    if model:
        try:
            r = await cli.post(
                base + "/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 16, "stream": False},
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
