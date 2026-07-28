"""Reverse-proxy a VM Compute session's JupyterLab through the gateway.

A VM session's Jupyter binds the **VM's loopback** — nothing about it is
reachable from a browser. The gateway holds an `ssh -L` forward to it
(`vm_tunnel.ensure_forward`) and serves it at

    /compute/jupyter/{pod_id}/{proxy_token}/…      ← both HTTP and WebSocket

Two things make this a *dumb* proxy (no body rewriting anywhere):

- **Jupyter is launched with `--ServerApp.base_url` equal to that same path**, so
  every URL it generates — HTML, the `/api/*` JSON, the WS endpoints, its
  cookie path — already points back through us.
- **`Host` and `Origin` are rewritten to the upstream's own `127.0.0.1:{port}`**
  so Jupyter's same-origin / host checks pass without `allow_origin='*'`.

**Auth is the unguessable `proxy_token` in the path**, not the gateway's Bearer
header — a browser can't attach an Authorization header when you click a link,
and Next's `/api/proxy` (which does the cookie→Bearer swap for the rest of the
UI) can't proxy WebSockets at all, so kernels would never connect. The token is
a 32-byte per-session secret compared in constant time, and Jupyter's own token
sits behind it; same capability-URL model as the RunPod path, whose jupyter URL
already embeds its token. These routes are therefore deliberately auth-exempt.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, WebSocket
from starlette.responses import RedirectResponse, Response, StreamingResponse
from starlette.websockets import WebSocketDisconnect

from . import compute_vm

logger = logging.getLogger("gateway.compute_proxy")

router = APIRouter(tags=["compute"])

# Jupyter fires a lot of small requests; re-resolving the pod row + SSH conn per
# request would hammer the DB. Cache the resolved session briefly and invalidate
# explicitly on teardown.
_CACHE_TTL_S = 15.0
# How long to let a freshly-spawned `ssh -L` bind before giving up on a request.
_FORWARD_WAIT_S = 40.0
# Never buffer a response — notebook downloads and `/api/events` (SSE) both need
# to stream. Connect fast, read unbounded.
_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=None, pool=10.0)

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}
_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


@dataclass
class _Session:
    token: str
    jupyter_token: str
    conn: dict[str, Any]
    vm_port: int
    local_port: int
    expires: float


_cache: dict[str, _Session] = {}
_client: Optional[httpx.AsyncClient] = None


def invalidate(pod_id: str) -> None:
    """Drop a cached session — called on terminate so a torn-down pod stops
    proxying immediately instead of for up to the cache TTL."""
    _cache.pop(pod_id, None)


async def _wait_local_port(port: int, timeout_s: float) -> bool:
    """Wait for the forward's local listener to accept.

    `ensure_forward` gives autossh 15s to bind, which isn't always enough — the
    TM boxes sit behind a slow SSH front-end and a cold connect can take ~30s.
    Without this, the FIRST request after a gateway restart reliably fails with
    'All connection attempts failed' even though the tunnel comes up a moment
    later."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            return True
        except OSError:
            await asyncio.sleep(0.5)
    return False


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=False)
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        finally:
            _client = None


async def _resolve(pod_id: str, token: str) -> _Session:
    """Validate the capability token and return a live forward to the session.

    ⚠ The token is compared on EVERY call, cache hit included — caching only the
    pod id would make a guessed/enumerated pod id enough to reach a warm session.
    The DB lookup is what's cached (`_CACHE_TTL_S`), never the authorization."""
    from .compute import ComputePod
    from .db import session_factory

    cached = _cache.get(pod_id)
    now = time.monotonic()
    if cached is not None and cached.expires > now:
        if not secrets.compare_digest(cached.token, token):
            raise HTTPException(status_code=404, detail={"error": "no such session"})
        return cached

    async with session_factory()() as s:
        pod = await s.get(ComputePod, pod_id)
        if pod is None or pod.kind != "vm":
            raise HTTPException(status_code=404, detail={"error": "no such session"})
        stored = pod.proxy_token or ""
        if not stored or not secrets.compare_digest(stored, token):
            raise HTTPException(status_code=404, detail={"error": "no such session"})
        if pod.status != "running" or not pod.vm_port:
            raise HTTPException(
                status_code=409,
                detail={"error": f"session is {pod.status}, not running"},
            )
        if not pod.provider_id:
            raise HTTPException(status_code=500, detail={"error": "session has no provider"})
        conn = await compute_vm.vm_conn_for_provider(s, pod.provider_id)
        vm_port = int(pod.vm_port)
        jupyter_token = pod.jupyter_password or ""

    # ensure_forward re-spawns a dead autossh (and can block up to ~15s doing
    # it), so never run it on the event loop.
    try:
        local_port = await asyncio.to_thread(compute_vm.ensure_forward_sync, conn, vm_port)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail={"error": f"could not reach the VM's JupyterLab: {e}"},
        ) from e

    # Don't cache (or serve) a forward that isn't listening yet — a cached dead
    # port would fail every request for the whole TTL.
    if not await _wait_local_port(local_port, _FORWARD_WAIT_S):
        raise HTTPException(
            status_code=503,
            detail={"error": "the SSH tunnel to this session is still coming up — retry in a moment"},
            headers={"Retry-After": "5"},
        )

    sess = _Session(token=stored, jupyter_token=jupyter_token, conn=conn,
                    vm_port=vm_port, local_port=local_port,
                    expires=now + _CACHE_TTL_S)
    _cache[pod_id] = sess
    return sess


def _raw_target(scope_holder, pod_id: str, token: str, path: str) -> bytes:
    """`{raw path}?{query}` for the upstream request.

    ASGI's `scope["path"]` (and therefore the router's `path` param) is
    percent-DECODED; `raw_path` keeps the bytes the client actually sent.
    Prefer it, and fall back to rebuilding from the decoded parts."""
    raw = scope_holder.scope.get("raw_path")
    if raw:
        target = raw.split(b"?", 1)[0]
    else:
        target = f"{compute_vm.base_path(pod_id, token)}{path}".encode()
    qs = scope_holder.url.query
    if qs:
        target += b"?" + (qs.encode() if isinstance(qs, str) else qs)
    return target


def _upstream_headers(request_headers, local_port: int, jupyter_token: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    upstream_host = f"127.0.0.1:{local_port}"
    for k, v in request_headers.items():
        lk = k.lower()
        if lk in _HOP_BY_HOP or lk == "host":
            continue
        # We authenticate to Jupyter ourselves (below) — never forward a
        # client-supplied Authorization, or a caller could try to override it.
        if lk == "authorization":
            continue
        # Make Jupyter see a same-origin request: it compares Origin's host to
        # its own (allow_remote_access covers Host, not Origin).
        if lk == "origin":
            v = f"http://{upstream_host}"
        elif lk == "referer":
            v = f"http://{upstream_host}/"
        out.append((k, v))
    out.append(("host", upstream_host))
    # ⚠ The gateway authenticates to Jupyter, on BOTH the HTTP and WS paths.
    # Jupyter does NOT set a session cookie for a `?token=` login, and a browser
    # cannot attach headers to a WebSocket — so JupyterLab's fetches (which send
    # the token from page config) worked while every kernel/terminal/events
    # socket got a 403 "Couldn't authenticate WebSocket connection". Injecting
    # it here is also why the published URL no longer needs `?token=`: the
    # path's proxy_token is the single capability, and Jupyter's own token stays
    # out of browser history. Token-authenticated requests also skip Jupyter's
    # XSRF check, so proxied POSTs work without an `_xsrf` cookie.
    if jupyter_token:
        out.append(("authorization", f"token {jupyter_token}"))
    return out


@router.get("/compute/jupyter/{pod_id}/{token}")
async def jupyter_root_redirect(pod_id: str, token: str, request: Request):
    """`…/{token}` → `…/{token}/` — Jupyter's base_url has a trailing slash and
    it 404s (or builds broken relative URLs) without one."""
    qs = request.url.query
    return RedirectResponse(
        url=f"/compute/jupyter/{pod_id}/{token}/" + (f"?{qs}" if qs else ""),
        status_code=307,
    )


@router.api_route("/compute/jupyter/{pod_id}/{token}/{path:path}", methods=_METHODS)
async def jupyter_http(pod_id: str, token: str, path: str, request: Request):
    sess = await _resolve(pod_id, token)
    # Forward the RAW (still percent-encoded) path — `path` from the router is
    # already decoded, and re-encoding it mangles notebook filenames containing
    # '?' or '#'. Jupyter's base_url is our prefix, so the raw path is exactly
    # what upstream expects.
    url = httpx.URL(
        scheme="http", host="127.0.0.1", port=sess.local_port,
        raw_path=_raw_target(request, pod_id, token, path),
    )

    cli = _http()
    # Stream the request body through rather than buffering — notebook file
    # uploads go through this path.
    req = cli.build_request(
        request.method, url,
        headers=_upstream_headers(request.headers, sess.local_port, sess.jupyter_token),
        content=request.stream() if request.method not in ("GET", "HEAD") else None,
    )
    try:
        resp = await cli.send(req, stream=True)
    except httpx.HTTPError as e:
        invalidate(pod_id)
        raise HTTPException(
            status_code=502,
            detail={"error": f"JupyterLab is not answering on the VM: {e}"},
        ) from e

    # multi_items() + raw_headers, NOT a dict: Jupyter can emit several
    # Set-Cookie headers on one response (_xsrf + the session cookie) and a dict
    # would keep only the last, silently breaking XSRF on POSTs.
    out_headers = [
        (k.encode("latin-1"), v.encode("latin-1"))
        for k, v in resp.headers.multi_items()
        if k.lower() not in _HOP_BY_HOP
    ]

    async def body():
        try:
            # aiter_raw: pass the wire bytes through untouched so the
            # content-encoding header we forwarded stays truthful.
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()

    if request.method == "HEAD":
        await resp.aclose()
        out: Response = Response(status_code=resp.status_code)
    else:
        out = StreamingResponse(body(), status_code=resp.status_code)
    out.raw_headers = out_headers
    return out


@router.websocket("/compute/jupyter/{pod_id}/{token}/{path:path}")
async def jupyter_ws(websocket: WebSocket, pod_id: str, token: str, path: str):
    """Relay Jupyter's kernel/terminal WebSockets. This is the reason the proxy
    lives on the gateway and not in Next's `/api/proxy` route handler, which
    can't upgrade a connection."""
    try:
        sess = await _resolve(pod_id, token)
    except HTTPException as e:
        # Nothing has been accepted yet — 1008 (policy violation) is the closest
        # WS analogue of the HTTP status we'd have returned.
        await websocket.close(code=1008, reason=str(e.detail)[:120])
        return

    from websockets.asyncio.client import connect as ws_connect

    target = _raw_target(websocket, pod_id, token, path).decode("latin-1")
    uri = f"ws://127.0.0.1:{sess.local_port}{target}"
    # Same auth story as the HTTP path: WE authenticate to Jupyter with the
    # session's token, because a browser cannot put a header on a WebSocket and
    # Jupyter issues no cookie for a `?token=` login. Without this every
    # kernel / terminal / events socket 403s while plain HTTP works.
    hdrs = [
        (k, v) for k, v in websocket.headers.items()
        if k.lower() in ("cookie", "user-agent")
    ]
    hdrs.append(("origin", f"http://127.0.0.1:{sess.local_port}"))
    if sess.jupyter_token:
        hdrs.append(("authorization", f"token {sess.jupyter_token}"))
    subprotocols = [
        p.strip() for p in (websocket.headers.get("sec-websocket-protocol") or "").split(",")
        if p.strip()
    ]

    try:
        upstream = await ws_connect(
            uri, additional_headers=hdrs,
            subprotocols=subprotocols or None,
            max_size=None, open_timeout=20, ping_interval=20, ping_timeout=20,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("compute-proxy %s: kernel WS upstream failed: %s", pod_id, e)
        invalidate(pod_id)
        await websocket.close(code=1011, reason="upstream websocket failed")
        return

    await websocket.accept(subprotocol=upstream.subprotocol)

    async def client_to_upstream() -> None:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("text") is not None:
                await upstream.send(msg["text"])
            elif msg.get("bytes") is not None:
                await upstream.send(msg["bytes"])

    async def upstream_to_client() -> None:
        async for message in upstream:
            if isinstance(message, str):
                await websocket.send_text(message)
            else:
                await websocket.send_bytes(message)

    tasks = [asyncio.create_task(client_to_upstream()), asyncio.create_task(upstream_to_client())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            exc = t.exception()
            if exc is not None and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                logger.debug("compute-proxy %s: ws relay ended: %s", pod_id, exc)
    finally:
        try:
            await upstream.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass
