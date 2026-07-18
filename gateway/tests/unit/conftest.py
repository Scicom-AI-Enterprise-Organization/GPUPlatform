"""Unit-test fixtures. Unlike the integration suite one level up (which drives a
LIVE gateway/web stack over HTTP and self-skips when it's unreachable), these
tests are pure in-process — no Postgres, no Redis, no network. Run with:

    .venv/bin/pytest gateway/tests/unit
"""
from __future__ import annotations

import os

import pytest

# crypto._fernet is lru_cached and reads PROVIDER_SECRET_KEY at first use — give
# every unit-test process a deterministic key before any module touches it.
os.environ.setdefault(
    "PROVIDER_SECRET_KEY", "Dvhc4l60UsMYCS-4CQvuwUyLKb3EMPBHt2p0O5vFvBc="
)


@pytest.fixture()
def anyio_backend():
    return "asyncio"
