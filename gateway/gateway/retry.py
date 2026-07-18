"""Shared retry-with-backoff helpers for outbound calls (provider APIs, SSH,
storage, HTTP probes).

The codebase grew several one-off inline retry loops (vm_serverless_provider's
SSH connect, dataset_transform's clip downloads, fc_eval's exponential sleep) —
this is the one importable implementation for new code. Deliberately tiny: no
decorator magic, no tenacity dependency, works for sync and async callables.

    from .retry import retry_async, retry_sync

    machines = await retry_async(
        lambda: provider.list_machines(),
        attempts=3, base_delay_s=0.5, what="runpod list_machines",
    )

Backoff is exponential (base * 2^attempt) capped at `max_delay_s`, with ±25%
jitter so N replicas retrying the same dead dependency don't stampede in phase.
Exceptions in `retry_on` (default: any Exception) are retried; the last one is
re-raised when attempts are exhausted. CancelledError always propagates —
retrying past cancellation would hold up graceful shutdown.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Awaitable, Callable, Tuple, Type, TypeVar

logger = logging.getLogger("gateway.retry")

T = TypeVar("T")


def _delay(attempt: int, base_delay_s: float, max_delay_s: float) -> float:
    d = min(base_delay_s * (2 ** attempt), max_delay_s)
    return d * random.uniform(0.75, 1.25)


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 30.0,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    what: str = "operation",
) -> T:
    """Await `fn()` up to `attempts` times. Retries only on `retry_on`;
    re-raises the final failure. Logs each retry at WARNING with the reason so
    flaky dependencies are visible before they become outages."""
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except asyncio.CancelledError:
            raise
        except retry_on as e:
            last = e
            if attempt + 1 >= attempts:
                break
            delay = _delay(attempt, base_delay_s, max_delay_s)
            logger.warning(
                "%s failed (attempt %d/%d, retrying in %.1fs): %s",
                what, attempt + 1, attempts, delay, e,
            )
            await asyncio.sleep(delay)
    assert last is not None
    raise last


def retry_sync(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 30.0,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    what: str = "operation",
) -> T:
    """Blocking twin of retry_async for the sync `_ssh_*`-style helpers that
    already run inside a thread executor. Never call from an async handler."""
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except retry_on as e:
            last = e
            if attempt + 1 >= attempts:
                break
            delay = _delay(attempt, base_delay_s, max_delay_s)
            logger.warning(
                "%s failed (attempt %d/%d, retrying in %.1fs): %s",
                what, attempt + 1, attempts, delay, e,
            )
            time.sleep(delay)
    assert last is not None
    raise last
