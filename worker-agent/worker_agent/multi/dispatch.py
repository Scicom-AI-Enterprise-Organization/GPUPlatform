"""Multi-model job dispatcher.

BRPOPs the single `queue:{app_id}` (same gateway contract as single-mode),
resolves each job's target model, ensures it's awake (evicting via the
scheduler if needed), then reuses the existing unary/stream runners against
that model's local vLLM port. Each job runs in a child task so the poll loop
keeps draining while a swap is in progress.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from .scheduler import (
    MultiModelScheduler,
    UnknownModelError,
    WakeError,
    WakeTimeoutError,
)

logger = logging.getLogger("worker-agent.dispatch")

MAX_CONCURRENT_JOBS = 64


async def multi_poll_loop(
    rdb,
    queue_key: str,
    machine_id: str,
    sched: MultiModelScheduler,
    drain_event: asyncio.Event,
) -> None:
    # Lazy import avoids a circular import at module load (main imports us).
    from worker_agent.main import _run_unary, _run_stream

    sem = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    tasks: set[asyncio.Task] = set()
    logger.info("multi dispatcher polling %s", queue_key)

    only_member = sched.cfg.members[0].served_name if len(sched.cfg.members) == 1 else None

    while not drain_event.is_set():
        try:
            res = await rdb.brpop(queue_key, timeout=2)
            if res is None:
                continue
            _key, blob = res
            job = json.loads(blob)
            await sem.acquire()
            t = asyncio.create_task(
                _handle_job(rdb, job, machine_id, sched, only_member, _run_unary, _run_stream, sem)
            )
            tasks.add(t)
            t.add_done_callback(tasks.discard)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("multi dispatcher loop error")
            await asyncio.sleep(1.0)

    # Drain in-flight job tasks on shutdown.
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _handle_job(rdb, job, machine_id, sched, only_member, run_unary, run_stream, sem):
    try:
        request_id = job["request_id"]
        # The client may have disconnected / timed out while this job waited in the
        # queue — the gateway then sets cancel:{request_id}. Bail BEFORE the model
        # wake + prefill so we don't burn GPU on a request nobody's waiting for.
        # One EXISTS per job (not per token) → off the streaming hot path, so it
        # can't affect token throughput. Mid-stream cancel (run_stream) still covers
        # requests abandoned *after* dispatch starts.
        if await rdb.exists(f"cancel:{request_id}"):
            await _error_result(
                rdb, request_id, machine_id,
                "client cancelled before dispatch", status="cancelled",
            )
            return
        payload = job.get("payload", {})
        stream = bool(job.get("stream"))
        timeout_s = float(job.get("timeout_s", 600))
        endpoint = job.get("endpoint", "/v1/completions")
        served = job.get("target_model") or payload.get("model") or only_member

        if not served:
            await _error_result(rdb, request_id, machine_id, "missing target model")
            return

        deadline = time.monotonic() + timeout_s
        try:
            rt = await sched.acquire(served, deadline)
        except UnknownModelError:
            await _error_result(
                rdb, request_id, machine_id,
                f"unknown model '{served}'", available=[m.served_name for m in sched.cfg.members],
            )
            return
        except WakeTimeoutError:
            await _error_result(rdb, request_id, machine_id, f"timed out waiting for model '{served}' to wake", status="timeout")
            return
        except WakeError as e:
            await _error_result(rdb, request_id, machine_id, str(e))
            return

        try:
            if stream:
                await run_stream(rdb, request_id, machine_id, "vllm", served, payload, timeout_s, endpoint, base_url=rt.base_url)
            else:
                await run_unary(rdb, request_id, machine_id, "vllm", served, payload, timeout_s, endpoint, base_url=rt.base_url)
        finally:
            await sched.release(rt)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("multi job handler crashed")
    finally:
        sem.release()


async def _error_result(rdb, request_id, machine_id, message, *, status="failed", available=None):
    out = {"error": message}
    if available is not None:
        out["available_models"] = available
    result = {"status": status, "output": out, "machine_id": machine_id}
    await rdb.set(f"result:{request_id}", json.dumps(result), ex=3600)
    logger.warning("job %s → %s: %s", request_id, status, message)
