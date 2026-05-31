"""Integration-test harness that drives the **web platform** (Next.js), not the
gateway directly.

Every request goes to `WEB/api/proxy/*` (default http://localhost:3000), exactly
like a browser client. The web proxy forwards the `sgpu_token` cookie to the
gateway as a Bearer token — and an API key is a valid bearer token, so the tests
authenticate by setting that cookie to a real `sgpu_…` **API key**.

Setup: log in via the web (`SGPU_ADMIN_USER`/`SGPU_ADMIN_PASS`, default
admin/admin) to get a session, mint an API key through the proxy, then run the
tests with that key. The key is revoked on teardown. Set `SGPU_API_KEY` to use
an existing key. The whole suite skips if the web platform isn't reachable.
"""
from __future__ import annotations

import os
import secrets

import httpx
import pytest
import pytest_asyncio

WEB = os.environ.get("WEB", "http://localhost:3000").rstrip("/")
ADMIN_USER = os.environ.get("SGPU_ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("SGPU_ADMIN_PASS", "admin")


@pytest.fixture(scope="session")
def api_key():
    """A `sgpu_…` API key. Uses SGPU_API_KEY if set, else logs in through the web
    and mints one via the proxy, revoking it on teardown. Skips if unavailable."""
    existing = os.environ.get("SGPU_API_KEY")
    minted_id = None
    # httpx.Client keeps a cookie jar: the web login sets sgpu_token (a session
    # token), which is then auto-sent to mint/revoke through the proxy.
    with httpx.Client(base_url=WEB, timeout=15.0) as c:
        if existing:
            yield existing
            return
        try:
            r = c.post("/api/auth/login", json={"username": ADMIN_USER, "password": ADMIN_PASS})
        except httpx.HTTPError as e:
            pytest.skip(f"web platform not reachable at {WEB} ({e}); set WEB / SGPU_API_KEY to run")
        if r.status_code != 200:
            pytest.skip(f"web login failed ({r.status_code}); set SGPU_API_KEY to run integration tests")
        rk = c.post("/api/proxy/api-keys", json={"name": f"pytest-{secrets.token_hex(3)}"})
        if rk.status_code != 200:
            pytest.skip(f"could not mint API key via the web proxy ({rk.status_code})")
        body = rk.json()
        key, minted_id = body["key"], body["id"]

        yield key

        try:
            c.delete(f"/api/proxy/api-keys/{minted_id}")
        except httpx.HTTPError:
            pass


class _WebClient(httpx.AsyncClient):
    """AsyncClient that routes every gateway path through the web's /api/proxy,
    so tests can keep writing plain `/v1/...` paths."""

    async def request(self, method, url, **kwargs):
        if isinstance(url, str) and url.startswith("/") and not url.startswith("/api/proxy"):
            url = "/api/proxy" + url
        return await super().request(method, url, **kwargs)


@pytest_asyncio.fixture
async def client(api_key):
    """Client pointed at the web platform, authenticated as an API key via the
    `sgpu_token` cookie (which the proxy forwards to the gateway as Bearer)."""
    async with _WebClient(base_url=WEB, cookies={"sgpu_token": api_key}, timeout=30.0) as c:
        yield c


@pytest_asyncio.fixture
async def cleanup(client):
    """Collect resource paths to DELETE after the test (runs even on failure)."""
    paths: list[str] = []
    yield paths
    for p in reversed(paths):
        try:
            await client.delete(p)
        except httpx.HTTPError:
            pass
