"""The global unhandled-exception handler added by the hardening pass: clients
get a structured JSON envelope + request id instead of Starlette's bare 500."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway import metrics
from gateway.main import _unhandled_exception_handler


def _mini_app() -> FastAPI:
    # A minimal app wired with the REAL handler — exercising it without booting
    # the gateway's lifespan (Redis/Postgres).
    app = FastAPI()
    app.add_exception_handler(Exception, _unhandled_exception_handler)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("kaboom")

    return app


def test_unhandled_exception_returns_structured_envelope():
    client = TestClient(_mini_app(), raise_server_exceptions=False)
    r = client.get("/boom", headers={"x-request-id": "req-unit-1"})
    assert r.status_code == 500
    err = r.json()["error"]
    assert err["type"] == "internal_error"
    assert err["message"] == "internal server error"
    # The id echoes back in body AND header so the client can quote it.
    assert err["request_id"] == "req-unit-1"
    assert r.headers["x-request-id"] == "req-unit-1"


def test_unhandled_exception_bumps_metric():
    client = TestClient(_mini_app(), raise_server_exceptions=False)
    before = metrics.UNHANDLED_EXCEPTIONS.labels(route="/boom")._value.get()
    client.get("/boom")
    after = metrics.UNHANDLED_EXCEPTIONS.labels(route="/boom")._value.get()
    assert after == before + 1


def test_no_inbound_id_still_returns_envelope():
    client = TestClient(_mini_app(), raise_server_exceptions=False)
    r = client.get("/boom")
    assert r.status_code == 500
    assert r.json()["error"]["type"] == "internal_error"
