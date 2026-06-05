import asyncio
import base64
import json
import logging
import os
import signal
import time
import uuid
from typing import Any

import httpx
import redis.asyncio as redis_async
from dotenv import load_dotenv

logger = logging.getLogger("worker-agent")

# Resilient redis client kwargs. The worker's redis (a public ELB) occasionally
# times out; a bare client throws on every blip, which crashed the worker (the
# unguarded startup ping) and churned the fleet. Bound socket ops, health-check
# the connection, and auto-retry connection/timeout errors on EVERY command.
_REDIS_KW: dict[str, Any] = dict(
    decode_responses=True,
    socket_connect_timeout=10,
    socket_timeout=20,
    socket_keepalive=True,
    health_check_interval=30,
    retry_on_timeout=True,
)
try:  # full retry policy (retries ConnectionError too) — best-effort across redis-py versions
    from redis.asyncio.retry import Retry as _Retry
    from redis.backoff import ExponentialBackoff as _ExpBackoff
    from redis.exceptions import ConnectionError as _RedisConnErr, TimeoutError as _RedisTOErr
    _REDIS_KW["retry"] = _Retry(_ExpBackoff(cap=10, base=0.5), retries=10)
    _REDIS_KW["retry_on_error"] = [_RedisConnErr, _RedisTOErr]
except Exception:  # noqa: BLE001 — older redis-py: socket timeouts + retry_on_timeout still apply
    pass


async def _redis_ready(rdb, *, timeout_s: float = 120.0) -> None:
    """Wait for redis to answer, retrying transient blips with backoff. Crashing on
    a startup blip just spawns a replacement that hits the same blip (crash-loop) —
    so retry for up to timeout_s, then give up (genuinely-down redis still fails)."""
    deadline = time.monotonic() + timeout_s
    delay = 0.5
    while True:
        try:
            await rdb.ping()
            return
        except Exception as e:  # noqa: BLE001
            if time.monotonic() >= deadline:
                raise
            logger.warning("redis not ready (%s) — retrying in %.1fs", e, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10.0)


async def register(gateway_url: str, machine_id: str, app_id: str, token: str) -> str:
    url = f"{gateway_url.rstrip('/')}/workers/register"
    body = {"machine_id": machine_id, "app_id": app_id, "token": token}
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(1, 31):
            try:
                r = await client.post(url, json=body)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("ok"):
                        return data["redis_url"]
                    raise RuntimeError(f"gateway rejected registration: {data}")
                logger.warning("register attempt %d → status=%s", attempt, r.status_code)
            except httpx.HTTPError as e:
                logger.warning("register attempt %d → %s", attempt, e)
            await asyncio.sleep(1.0)
    raise RuntimeError("gateway never accepted registration after 30 attempts")


# OpenAI-compatible audio endpoints take a multipart file upload, so the worker
# rebuilds multipart from the base64'd clip the gateway put in the JSON payload.
AUDIO_PATHS = ("/v1/audio/transcriptions", "/v1/audio/translations")


async def handle(mode: str, model_id: str, payload: dict, endpoint: str = "/v1/completions", base_url: str | None = None) -> Any:
    """Run a unary (non-streaming) request.

    `endpoint` is the path on localhost vLLM to POST to. Defaults to the
    legacy /v1/completions; OpenAI-compat /run uses /v1/chat/completions or
    /v1/embeddings. The body is forwarded verbatim — vLLM is OpenAI-shaped
    natively, so the gateway's job is just queue + auth + autoscale.

    `base_url` overrides the target vLLM (multi-model: one server per model on a
    distinct port); single-mode leaves it None and uses $VLLM_URL.
    """
    if mode == "fake":
        return {
            "echo": payload,
            "fake": True,
            "model": model_id,
            "endpoint": endpoint,
            "completion": f"[fake response from {model_id}] you sent: {payload}",
        }
    if mode == "vllm":
        url = base_url or os.environ.get("VLLM_URL", "http://localhost:8000")
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                if endpoint in AUDIO_PATHS:
                    # Audio endpoints (Whisper) take multipart/form-data, not JSON.
                    # The gateway base64'd the clip into the payload (the queue is
                    # JSON-only); rebuild the multipart upload vLLM expects.
                    audio = base64.b64decode(payload["_audio_b64"])
                    files = {"file": (payload.get("_filename") or "audio.wav", audio)}
                    data = {"model": payload["model"], **(payload.get("_form") or {})}
                    r = await client.post(f"{url}{endpoint}", files=files, data=data)
                else:
                    r = await client.post(f"{url}{endpoint}", json=payload)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:
                return {"error": str(e)}
    return {"error": f"unknown WORKER_MODE: {mode}"}


async def handle_stream(mode: str, model_id: str, payload: dict, endpoint: str = "/v1/completions", base_url: str | None = None):
    """Async generator yielding chunks. Final chunk is `{"done": True}`."""
    if mode == "fake":
        # Simulate token-by-token output from a real LLM.
        words = ["[fake", "stream", "from", model_id + "]", "you", "sent:", str(payload)]
        for i, word in enumerate(words):
            yield {"index": i, "delta": word + " "}
            await asyncio.sleep(0.02)
        yield {"done": True}
        return

    if mode == "vllm":
        url = base_url or os.environ.get("VLLM_URL", "http://localhost:8000")
        body = {**payload, "stream": True}
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", f"{url}{endpoint}", json=body) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[len("data: "):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            yield {"raw": data}
                yield {"done": True}
                return
            except httpx.HTTPError as e:
                yield {"error": str(e), "done": True}
                return

    yield {"error": f"unknown WORKER_MODE: {mode}", "done": True}


async def log_shipper_loop(
    gateway_url: str,
    machine_id: str,
    app_id: str,
    log_path: str,
    drain_event: asyncio.Event,
    source: str | None = None,
) -> None:
    """Tail the vLLM stdout file and ship batches to the gateway.

    Tracks a byte offset across iterations. If the file is rotated/truncated
    we reset to 0. Failures are logged but never raise — log shipping is
    best-effort, never block the worker.

    `source`, when set (multi-model: one shipper per member), tags the batch so
    the gateway buckets logs per model and the UI can show each model's tail."""
    url = f"{gateway_url.rstrip('/')}/workers/logs"
    offset = 0
    leftover = b""
    BATCH_INTERVAL_S = 3.0
    MAX_BATCH_LINES = 2000          # big batches → a chatty model (vLLM load spam) ships a
    MAX_POST_BYTES = 512 * 1024     # few POSTs per tick, not dozens of 200-line ones
    MAX_LINE_BYTES = 4096

    async with httpx.AsyncClient(timeout=10.0) as client:
        while not drain_event.is_set():
            try:
                if not os.path.exists(log_path):
                    # File hasn't been created yet (vLLM still booting).
                    try:
                        await asyncio.wait_for(drain_event.wait(), timeout=BATCH_INTERVAL_S)
                    except asyncio.TimeoutError:
                        pass
                    continue
                size = os.path.getsize(log_path)
                if size < offset:
                    # Truncated or rotated — start over from the top.
                    offset = 0
                    leftover = b""
                if size > offset:
                    with open(log_path, "rb") as f:
                        f.seek(offset)
                        chunk = f.read(size - offset)
                    offset = size
                    data = leftover + chunk
                    parts = data.split(b"\n")
                    leftover = parts[-1]
                    new_lines = parts[:-1]
                    # Ship in batches bounded by BOTH line count and byte size, so a
                    # backlog drains in a few large POSTs instead of one-per-200-lines
                    # (which floods the gateway when a model logs heavily / crash-loops).
                    i = 0
                    while i < len(new_lines):
                        batch: list[str] = []
                        nbytes = 0
                        while i < len(new_lines) and len(batch) < MAX_BATCH_LINES and nbytes < MAX_POST_BYTES:
                            line = new_lines[i][:MAX_LINE_BYTES].decode("utf-8", errors="replace")
                            batch.append(line)
                            nbytes += len(line) + 1
                            i += 1
                        body: dict[str, Any] = {"machine_id": machine_id, "app_id": app_id, "lines": batch}
                        if source is not None:
                            body["source"] = source
                        try:
                            await client.post(url, json=body)
                        except httpx.HTTPError as e:
                            logger.warning("log shipper post failed: %s", e)
            except Exception:
                logger.exception("log shipper iteration failed")
            try:
                await asyncio.wait_for(drain_event.wait(), timeout=BATCH_INTERVAL_S)
            except asyncio.TimeoutError:
                pass


async def metrics_shipper_loop(gateway_url, machine_id, app_id, members_fn, drain_event, interval_s: float = 15.0) -> None:
    """Scrape each live member's local vLLM /metrics and ship them to the gateway
    for the combined /metrics/workers scrape target. `members_fn()` returns
    [(served_name, base_url)]. Best-effort — never blocks the worker."""
    url = f"{gateway_url.rstrip('/')}/workers/metrics"
    async with httpx.AsyncClient(timeout=10.0) as client:
        while not drain_event.is_set():
            metrics: dict[str, str] = {}
            for served, base_url in members_fn():
                try:
                    r = await client.get(f"{base_url}/metrics", timeout=5.0)
                    if r.status_code == 200 and r.text:
                        metrics[served] = r.text
                except Exception:
                    pass
            if metrics:
                try:
                    await client.post(url, json={"machine_id": machine_id, "app_id": app_id, "metrics": metrics})
                except httpx.HTTPError as e:
                    logger.warning("metrics shipper post failed: %s", e)
            try:
                await asyncio.wait_for(drain_event.wait(), timeout=interval_s)
            except asyncio.TimeoutError:
                pass


async def _safe_cmd(cmd_fn, model: str, action: str) -> None:
    try:
        await cmd_fn(model, action)
    except Exception:
        logger.exception("worker command failed: %s %s", action, model)


async def heartbeat_loop(
    gateway_url: str,
    machine_id: str,
    app_id: str,
    drain_event: asyncio.Event,
    snapshot_fn=None,
    cmd_fn=None,
) -> None:
    """Heartbeat to gateway every 5s. Set drain_event if gateway tells us to drain.

    `snapshot_fn`, when given (multi-model), returns the per-model state list the
    gateway surfaces in the endpoint status; `status` flips to "loading" until
    the fleet has finished launching.

    `cmd_fn`, when given, runs operator commands the gateway returns in the
    heartbeat response (kill/restart a member). Each runs in its own task so a
    slow restart (model reload) never stalls the heartbeat — a stalled heartbeat
    would let the worker's TTL lapse and trigger a needless re-provision."""
    url = f"{gateway_url.rstrip('/')}/workers/heartbeat"
    async with httpx.AsyncClient(timeout=5.0) as client:
        while not drain_event.is_set():
            body = {"machine_id": machine_id, "app_id": app_id, "status": "ready"}
            if snapshot_fn is not None:
                try:
                    models, ready = snapshot_fn()
                    body["models"] = models
                    body["status"] = "ready" if ready else "loading"
                except Exception:
                    logger.exception("snapshot_fn failed")
            try:
                r = await client.post(url, json=body)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("drain"):
                        logger.info("drain signal received from gateway")
                        drain_event.set()
                        return
                    if cmd_fn is not None:
                        for cmd in data.get("commands") or []:
                            model, action = cmd.get("model"), cmd.get("action")
                            if model and action:
                                logger.info("received command: %s %s", action, model)
                                asyncio.create_task(_safe_cmd(cmd_fn, model, action))
            except httpx.HTTPError as e:
                logger.warning("heartbeat error: %s", e)
            try:
                await asyncio.wait_for(drain_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass


async def poll_loop(rdb, queue_key: str, machine_id: str, mode: str, model_id: str, drain_event: asyncio.Event) -> None:
    logger.info("ready, polling %s", queue_key)
    while not drain_event.is_set():
        try:
            res = await rdb.brpop(queue_key, timeout=2)
            if res is None:
                continue
            _key, blob = res
            job = json.loads(blob)
            request_id = job["request_id"]
            payload = job.get("payload", {})
            stream = bool(job.get("stream"))
            timeout_s = float(job.get("timeout_s", 600))
            endpoint = job.get("endpoint", "/v1/completions")
            logger.info("picked up %s (stream=%s endpoint=%s timeout=%ss)", request_id, stream, endpoint, timeout_s)

            if stream:
                await _run_stream(rdb, request_id, machine_id, mode, model_id, payload, timeout_s, endpoint)
            else:
                await _run_unary(rdb, request_id, machine_id, mode, model_id, payload, timeout_s, endpoint)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("loop error, sleeping 1s")
            await asyncio.sleep(1.0)


async def _run_unary(rdb, request_id, machine_id, mode, model_id, payload, timeout_s, endpoint="/v1/completions", base_url=None):
    try:
        output = await asyncio.wait_for(handle(mode, model_id, payload, endpoint, base_url=base_url), timeout=timeout_s)
        result = {"status": "completed", "output": output, "machine_id": machine_id}
    except asyncio.TimeoutError:
        logger.warning("%s timed out after %ss", request_id, timeout_s)
        result = {
            "status": "timeout",
            "output": {"error": f"request exceeded timeout_s={timeout_s}"},
            "machine_id": machine_id,
        }
    await rdb.set(f"result:{request_id}", json.dumps(result), ex=3600)
    logger.info("wrote result for %s status=%s", request_id, result["status"])


async def _run_stream(rdb, request_id, machine_id, mode, model_id, payload, timeout_s, endpoint="/v1/completions", base_url=None):
    """Stream with per-request timeout + mid-stream cancel.

    Token chunks are PIPELINED to Redis in small batches (one round-trip per
    batch instead of one per token), and the cancel key is polled at most every
    ~250ms rather than per token. Over a reverse-SSH tunnel the per-token Redis
    round-trip is what serialized streaming throughput (~670 vs ~985 tok/s
    unary); batching closes that gap. The gateway still gets one pub/sub message
    per chunk, so the SSE wire protocol is unchanged."""
    loop = asyncio.get_event_loop()
    channel = f"stream:{request_id}"
    cancel_key = f"cancel:{request_id}"
    last = None
    cancelled = False
    timed_out = False
    deadline = loop.time() + timeout_s

    buf: list[dict] = []
    last_flush = loop.time()
    last_cancel_check = 0.0
    FLUSH_S = 0.02   # flush at least every 20ms (keeps streaming smooth)
    FLUSH_N = 16     # ...or every 16 chunks, whichever comes first

    async def flush():
        if not buf:
            return
        pipe = rdb.pipeline(transaction=False)
        for ch in buf:
            pipe.publish(channel, json.dumps(ch))
        await pipe.execute()
        buf.clear()

    try:
        gen = handle_stream(mode, model_id, payload, endpoint, base_url=base_url)
        while True:
            now = loop.time()
            remaining = deadline - now
            if remaining <= 0:
                timed_out = True
                break
            try:
                chunk = await asyncio.wait_for(gen.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                timed_out = True
                break

            if now - last_cancel_check > 0.25:
                last_cancel_check = now
                if await rdb.exists(cancel_key):
                    logger.info("client cancelled %s, stopping mid-stream", request_id)
                    cancelled = True
                    last = {"cancelled": True, "done": True}
                    buf.append(last)
                    await flush()
                    break
            last = chunk
            buf.append(chunk)
            if len(buf) >= FLUSH_N or (now - last_flush) >= FLUSH_S:
                await flush()
                last_flush = now
    finally:
        await flush()
        if timed_out:
            chunk = {"timeout": True, "timeout_s": timeout_s, "done": True}
            last = chunk
            await rdb.publish(channel, json.dumps(chunk))

    if timed_out:
        status = "timeout"
    elif cancelled:
        status = "cancelled"
    else:
        status = "completed"
    final = {"status": status, "output": last, "machine_id": machine_id, "streamed": True}
    await rdb.set(f"result:{request_id}", json.dumps(final), ex=3600)
    logger.info("streamed %s status=%s", request_id, status)


def _load_config_file() -> None:
    """When launched over SSH on a VM, the provider drops all env in a JSON file
    and points WORKER_CONFIG_FILE at it (keeps the big MULTI_MODEL_CONFIG + the
    registration token off the process arg list). Load it into os.environ
    without clobbering anything already set in the real environment."""
    path = os.environ.get("WORKER_CONFIG_FILE")
    if not path or not os.path.exists(path):
        return
    try:
        with open(path) as fh:
            data = json.load(fh)
        for k, v in (data or {}).items():
            os.environ.setdefault(str(k), str(v))
        logger.info("loaded worker config from %s (%d keys)", path, len(data or {}))
    except Exception:
        logger.exception("failed to load WORKER_CONFIG_FILE=%s", path)


_RESERVED_ENV = {
    "APP_ID", "MACHINE_ID", "GATEWAY_URL", "WORKER_REDIS_URL", "REGISTRATION_TOKEN",
    "WORKER_MODE", "MULTI_MODEL_CONFIG", "MULTI_MODEL_CONFIG_PATH", "SLEEP_LEVEL",
    "TOTAL_GPUS", "WORKER_CONFIG_FILE", "WORKER_ENV_JSON",
}


def _apply_user_env() -> None:
    """Apply endpoint-level env vars (WORKER_ENV_JSON) to this process so every
    vLLM subprocess inherits them, and mkdir -p any absolute-path values (HF_HOME,
    TRITON_CACHE_DIR, …). Reserved control vars can't be overridden.

    CUDA_VISIBLE_DEVICES is intentionally allowed here as a global default, but
    the multi-model launcher sets it per-model afterwards, so the per-model
    pinning always wins."""
    raw = os.environ.get("WORKER_ENV_JSON")
    if not raw:
        return
    try:
        custom = json.loads(raw)
    except Exception:
        logger.exception("failed to parse WORKER_ENV_JSON")
        return
    for k, v in (custom or {}).items():
        if k in _RESERVED_ENV:
            continue
        v = str(v)
        os.environ[k] = v
        if v.startswith("/"):
            try:
                os.makedirs(v, exist_ok=True)
            except Exception as e:
                logger.warning("could not mkdir %s=%s: %s", k, v, e)
    logger.info("applied %d user env var(s)", len(custom or {}))


async def main_async() -> None:
    _load_config_file()
    _apply_user_env()
    app_id = os.environ.get("APP_ID")
    if not app_id:
        raise SystemExit("APP_ID env var required")

    machine_id = os.environ.get("MACHINE_ID") or f"m-{uuid.uuid4().hex[:8]}"
    token = os.environ.get("REGISTRATION_TOKEN", "dev-token")
    gateway_url = os.environ.get("GATEWAY_URL", "http://gateway:8080")
    mode = os.environ.get("WORKER_MODE", "fake")
    model_id = os.environ.get("MODEL_ID", "fake-model")

    logger.info(
        "worker booting: app=%s machine=%s mode=%s model=%s gateway=%s",
        app_id, machine_id, mode, model_id, gateway_url,
    )

    redis_url = await register(gateway_url, machine_id, app_id, token)
    logger.info("registered with gateway, redis=%s", redis_url)

    rdb = redis_async.from_url(redis_url, **_REDIS_KW)
    drain_event = asyncio.Event()
    # Drain gracefully on SIGTERM/SIGINT so the `finally` runs sched.shutdown(),
    # which group-kills every vLLM engine + its tp workers. Without this, the
    # provider terminating the worker (SIGTERM) skips cleanup and the engines
    # orphan, holding GPU ("vllm worker still in top" after a delete).
    try:
        loop = asyncio.get_running_loop()
        for _sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(_sig, drain_event.set)
    except (NotImplementedError, RuntimeError):
        pass  # add_signal_handler unsupported on this platform — best effort
    log_path = os.environ.get("WORKER_LOG_PATH", "/var/log/vllm.log")

    if mode == "multi":
        await _run_multi(rdb, app_id, machine_id, gateway_url, drain_event)
        return

    try:
        await _redis_ready(rdb)
        hb_task = asyncio.create_task(
            heartbeat_loop(gateway_url, machine_id, app_id, drain_event)
        )
        # Skip log shipping in fake mode — there's no vllm.log to tail.
        log_task: asyncio.Task[Any] | None = None
        if mode != "fake":
            log_task = asyncio.create_task(
                log_shipper_loop(gateway_url, machine_id, app_id, log_path, drain_event)
            )
        try:
            await poll_loop(rdb, f"queue:{app_id}", machine_id, mode, model_id, drain_event)
        finally:
            drain_event.set()
            for t in (hb_task, log_task):
                if t is None:
                    continue
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, BaseException):
                    pass
    finally:
        await rdb.aclose()


async def _run_multi(rdb, app_id, machine_id, gateway_url, drain_event) -> None:
    """WORKER_MODE=multi: launch the vLLM fleet, route jobs by model name, and
    evict idle models via sleep/wake to fit the GPU budget."""
    from .multi.config import parse_multi_config
    from .multi.scheduler import MultiModelScheduler
    from .multi.dispatch import multi_poll_loop
    from .multi.launcher import log_path_for

    cfg = parse_multi_config(
        os.environ.get("MULTI_MODEL_CONFIG"),
        os.environ.get("MULTI_MODEL_CONFIG_PATH"),
    )
    log_dir = os.environ.get("WORKER_LOG_DIR", "/var/log/vllm")
    sched = MultiModelScheduler(cfg, machine_id, log_dir=log_dir)
    logger.info("multi mode: %d models, %d GPUs", len(cfg.members), cfg.total_gpus)

    def snapshot():
        return sched.states_snapshot(), sched.all_ready()

    async def cmd_fn(model: str, action: str) -> None:
        if action == "kill":
            await sched.kill_model(model)
        elif action == "restart":
            await sched.restart_model(model)
        elif action == "sleep":
            await sched.operator_sleep(model)
        elif action == "sleep_all":
            await sched.operator_sleep_all()
        else:
            logger.warning("ignoring unknown command action: %s", action)

    hb_task = asyncio.create_task(
        heartbeat_loop(gateway_url, machine_id, app_id, drain_event, snapshot_fn=snapshot, cmd_fn=cmd_fn)
    )
    # One log shipper per member, each tagged with the model's served_name so the
    # gateway buckets logs per model (the single-file shipper can't — every model
    # has its own vLLM process + log file).
    log_tasks = [
        asyncio.create_task(
            log_shipper_loop(
                gateway_url, machine_id, app_id,
                log_path_for(m, log_dir), drain_event, source=m.served_name,
            )
        )
        for m in cfg.members
    ]
    # Also ship the worker-agent's OWN stdout log (scheduler events: wave-loading,
    # sleep/wake, operator commands, preflight, dead reasons) under a reserved
    # "__worker__" source, so the UI can show the fleet's control-plane log too.
    self_log = os.environ.get("WORKER_SELF_LOG_PATH")
    if self_log:
        log_tasks.append(asyncio.create_task(
            log_shipper_loop(gateway_url, machine_id, app_id, self_log, drain_event, source="__worker__")
        ))
    # Ship each member's vLLM /metrics to the gateway's combined scrape target.
    log_tasks.append(asyncio.create_task(
        metrics_shipper_loop(gateway_url, machine_id, app_id, sched.live_members, drain_event)
    ))
    monitor_task = asyncio.create_task(sched.monitor_loop(drain_event))
    try:
        await _redis_ready(rdb)
        await sched.start()
        await multi_poll_loop(rdb, f"queue:{app_id}", machine_id, sched, drain_event)
    finally:
        drain_event.set()
        for t in (hb_task, monitor_task, *log_tasks):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, BaseException):
                pass
        await sched.shutdown()
        await rdb.aclose()


def run() -> None:
    load_dotenv()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx logs an INFO line per request (heartbeat + log-ship POSTs every few
    # seconds); silence it so the worker's own log stays readable + shippable.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main_async())


if __name__ == "__main__":
    run()
