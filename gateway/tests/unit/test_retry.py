"""retry — the shared backoff helper introduced by the hardening pass."""
import asyncio

import pytest

from gateway.retry import retry_async, retry_sync


async def test_async_succeeds_after_transient_failures():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("blip")
        return "ok"

    out = await retry_async(flaky, attempts=3, base_delay_s=0.001, what="flaky")
    assert out == "ok"
    assert calls["n"] == 3


async def test_async_exhausts_and_reraises_last_error():
    async def always_fails():
        raise TimeoutError("dead")

    with pytest.raises(TimeoutError, match="dead"):
        await retry_async(always_fails, attempts=2, base_delay_s=0.001)


async def test_async_cancellation_propagates_immediately():
    calls = {"n": 0}

    async def cancels():
        calls["n"] += 1
        raise asyncio.CancelledError()

    # CancelledError must never be swallowed/retried — it would stall shutdown.
    with pytest.raises(asyncio.CancelledError):
        await retry_async(cancels, attempts=5, base_delay_s=0.001)
    assert calls["n"] == 1


async def test_async_only_retries_listed_exceptions():
    async def type_error():
        raise TypeError("bug, not blip")

    with pytest.raises(TypeError):
        await retry_async(
            type_error, attempts=5, base_delay_s=0.001, retry_on=(ConnectionError,)
        )


def test_sync_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise OSError("blip")
        return 42

    assert retry_sync(flaky, attempts=3, base_delay_s=0.001) == 42
    assert calls["n"] == 2


def test_sync_exhausts_and_reraises():
    def always_fails():
        raise OSError("still dead")

    with pytest.raises(OSError, match="still dead"):
        retry_sync(always_fails, attempts=3, base_delay_s=0.001)
