"""vLLM sleep-mode control-plane calls.

Requires each vLLM server launched with `--enable-sleep-mode` and env
`VLLM_SERVER_DEV_MODE=1` (exposes /sleep, /wake_up, /collective_rpc,
/reset_prefix_cache). See https://vllm.ai/blog/2025-10-26-sleep-mode.
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("worker-agent.vllm_ctl")

# Wake budgets: level 1 (CPU-RAM offload) is near-instant; level 2 reloads
# weights from disk so it needs much longer.
WAKE_TIMEOUT_L1_S = 60.0
WAKE_TIMEOUT_L2_S = 300.0
SLEEP_TIMEOUT_S = 120.0
HEALTH_TIMEOUT_S = 5.0


async def is_healthy(client: httpx.AsyncClient, base_url: str) -> bool:
    try:
        r = await client.get(f"{base_url}/health", timeout=HEALTH_TIMEOUT_S)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


async def sleep_model(client: httpx.AsyncClient, base_url: str, level: int) -> None:
    r = await client.post(f"{base_url}/sleep", params={"level": int(level)}, timeout=SLEEP_TIMEOUT_S)
    if r.status_code >= 400:
        raise RuntimeError(f"/sleep?level={level} → {r.status_code}: {r.text[:200]}")
    logger.info("slept %s (level=%s)", base_url, level)


async def reload_weights(client: httpx.AsyncClient, base_url: str) -> None:
    r = await client.post(
        f"{base_url}/collective_rpc",
        json={"method": "reload_weights"},
        timeout=WAKE_TIMEOUT_L2_S,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"/collective_rpc reload_weights → {r.status_code}: {r.text[:200]}")


async def reset_prefix_cache(client: httpx.AsyncClient, base_url: str) -> None:
    r = await client.post(f"{base_url}/reset_prefix_cache", timeout=WAKE_TIMEOUT_L2_S)
    if r.status_code >= 400:
        raise RuntimeError(f"/reset_prefix_cache → {r.status_code}: {r.text[:200]}")


async def wake_model(client: httpx.AsyncClient, base_url: str, level: int) -> None:
    """Wake a sleeping model. Level 2 additionally reloads weights from disk and
    resets the prefix cache (required because the weights were discarded)."""
    budget = WAKE_TIMEOUT_L2_S if level == 2 else WAKE_TIMEOUT_L1_S
    r = await client.post(f"{base_url}/wake_up", timeout=budget)
    if r.status_code >= 400:
        raise RuntimeError(f"/wake_up → {r.status_code}: {r.text[:200]}")
    if level == 2:
        await reload_weights(client, base_url)
        await reset_prefix_cache(client, base_url)
    logger.info("woke %s (level=%s)", base_url, level)
