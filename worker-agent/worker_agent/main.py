import asyncio
import base64
import json
import logging
import os
import random
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


_NODE_META: dict[str, Any] | None = None


def node_meta() -> dict[str, Any]:
    """Best-effort identity of the node this worker runs on, stamped into every
    result so the gateway can persist *where* a request was served (history API).
    Probed once per process: hostname + provider pod id from env, GPU inventory
    via nvidia-smi (never fatal — a CPU box / missing binary just omits GPUs)."""
    global _NODE_META
    if _NODE_META is not None:
        return _NODE_META
    import socket
    import subprocess
    meta: dict[str, Any] = {"hostname": socket.gethostname()}
    if os.environ.get("RUNPOD_POD_ID"):
        meta["runpod_pod_id"] = os.environ["RUNPOD_POD_ID"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        meta["visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        gpus = [
            [p.strip() for p in line.split(",")]
            for line in out.splitlines() if line.strip()
        ]
        if gpus:
            meta["gpu_name"] = gpus[0][0]
            meta["gpu_count"] = len(gpus)
            if len(gpus[0]) > 1:
                meta["gpu_memory"] = gpus[0][1]
            if len(gpus[0]) > 2:
                meta["driver_version"] = gpus[0][2]
    except Exception:  # noqa: BLE001 — nvidia-smi absent/broken: report without GPUs
        pass
    if "gpu_name" not in meta:
        # Huawei Ascend box: name the NPUs from the npu-smi table instead
        # (rows pair as "<id> <name> | health | …" — grab the first chip row).
        try:
            out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10).stdout
            import re as _re
            names = [n for n in _re.findall(r"^\|\s*\d+\s+(\S+)\s*\|\s*\S+\s*\|", out, _re.M)
                     if not n.isdigit()]  # drop process-table rows ("| 0  0 | pid |…")
            if names:
                meta["gpu_name"] = f"Ascend {names[0]}"
                meta["gpu_count"] = len(names)
        except Exception:  # noqa: BLE001 — npu-smi absent: report without NPUs
            pass
    _NODE_META = meta
    return meta


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


async def register(gateway_url: str, machine_id: str, app_id: str, token: str,
                   *, timeout_s: float = 900.0) -> str:
    """Register with the gateway, retrying transient failures with backoff.

    The gateway may still be starting / rolling / briefly unreachable when a
    freshly-provisioned pod boots — and giving up after ~30s (the old fixed
    30×1s loop) exited the process while the pod kept billing indefinitely.
    Retry patiently for up to `timeout_s` (~15 min) with a bounded exponential
    backoff (cap 30s) plus jitter — modelled on `_redis_ready` — so a whole
    fleet booting together doesn't hammer a recovering gateway in lockstep. An
    explicit reject (`ok:false`) is deterministic, so it fails fast (no retry)."""
    url = f"{gateway_url.rstrip('/')}/workers/register"
    body = {"machine_id": machine_id, "app_id": app_id, "token": token}
    deadline = time.monotonic() + timeout_s
    delay = 1.0
    attempt = 0
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            attempt += 1
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
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"gateway never accepted registration after {attempt} attempts "
                    f"(~{int(timeout_s)}s)")
            # Bounded exponential backoff + small jitter (decorrelates a fleet).
            sleep_s = min(delay, 30.0) + random.uniform(0.0, 1.0)
            logger.warning("register retrying in %.1fs", sleep_s)
            await asyncio.sleep(sleep_s)
            delay = min(delay * 2, 30.0)


# OpenAI-compatible audio endpoints take a multipart file upload, so the worker
# rebuilds multipart from the base64'd clip the gateway put in the JSON payload.
AUDIO_PATHS = ("/v1/audio/transcriptions", "/v1/audio/translations")

# Cap the upstream body we echo into a result so a pathological error page can't
# bloat the Redis result key (vLLM's error JSON is normally tiny).
_MAX_UPSTREAM_BODY = 4096


class UpstreamError(Exception):
    """A vLLM upstream call failed. Carries a structured `payload` (the error
    message + the upstream status/body when it was an HTTP status error) so the
    job runner can write it as a FAILED result instead of a bogus COMPLETED one."""

    def __init__(self, payload: dict):
        super().__init__(payload.get("error", "upstream error"))
        self.payload = payload


def _upstream_error_payload(e: httpx.HTTPError) -> dict:
    """Structured error info for a failed vLLM upstream call. For an HTTP status
    error, preserve vLLM's OWN status code + response body — that's the actual
    reason (bad request, model not found, OOM) — instead of collapsing it to a
    bare `str(e)` that discards it."""
    out: dict[str, Any] = {"error": str(e)}
    if isinstance(e, httpx.HTTPStatusError):
        out["status_code"] = e.response.status_code
        try:
            out["body"] = e.response.text[:_MAX_UPSTREAM_BODY]
        except Exception:  # noqa: BLE001 — response body not available/decodable
            pass
    return out


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
                # An upstream/generation failure is NOT a completed job — raise so
                # the runner records it as FAILED (preserving vLLM's status+body)
                # rather than wrapping {"error": …} as a successful result.
                raise UpstreamError(_upstream_error_payload(e)) from e
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
                # Terminal upstream error mid-stream (bad request, engine dead,
                # connection drop). Emit an error chunk carrying vLLM's own
                # status+body; the runner keys off the "error" field to record the
                # job as FAILED rather than completed.
                yield {**_upstream_error_payload(e), "done": True}
                return

    yield {"error": f"unknown WORKER_MODE: {mode}", "done": True}


async def log_shipper_loop(
    gateway_url: str,
    machine_id: str,
    app_id: str,
    log_path: str | None,
    drain_event: asyncio.Event,
    source: str | None = None,
    get_log_path: "Callable[[], str | None] | None" = None,
    fixed_session: str | None = None,
) -> None:
    """Tail the vLLM stdout file and ship batches to the gateway.

    Tracks a byte offset across iterations. If the file is rotated/truncated
    we reset to 0. Failures are logged but never raise — log shipping is
    best-effort, never block the worker.

    `source`, when set (multi-model: one shipper per member), tags the batch so
    the gateway buckets logs per model and the UI can show each model's tail.

    `get_log_path`, when set, is polled each tick for the member's CURRENT log
    file (it changes on every (re)launch — a fresh timestamped file). When it
    changes we reset the offset and start shipping the new file under a new
    `session` (the file's timestamp), so the gateway keeps each launch's log
    separately and the UI can open historical logs. The crashing log is never
    overwritten — it stays its own session."""
    url = f"{gateway_url.rstrip('/')}/workers/logs"
    offset = 0
    leftover = b""
    cur_path = log_path
    # `fixed_session` pins the session (the agent's own __worker__ log = one session
    # per agent launch); otherwise derive it from the rotating per-launch filename.
    session = fixed_session if fixed_session is not None else _session_of(cur_path)
    BATCH_INTERVAL_S = 3.0
    MAX_BATCH_LINES = 2000          # big batches → a chatty model (vLLM load spam) ships a
    MAX_POST_BYTES = 512 * 1024     # few POSTs per tick, not dozens of 200-line ones
    MAX_LINE_BYTES = 4096

    async with httpx.AsyncClient(timeout=10.0) as client:
        async def drain(path: "str | None", sess: "str | None") -> None:
            """Ship bytes appended to `path` since `offset` (newline-split) under
            session `sess`. Mutates offset/leftover. Best-effort: never raises."""
            nonlocal offset, leftover
            if not path or not os.path.exists(path):
                return
            size = os.path.getsize(path)
            if size < offset:          # truncated/rotated in place → restart from top
                offset = 0
                leftover = b""
            if size <= offset:
                return
            with open(path, "rb") as f:
                f.seek(offset)
                chunk = f.read(size - offset)
            offset = size
            data = leftover + chunk
            parts = data.split(b"\n")
            leftover = parts[-1]
            new_lines = parts[:-1]
            # Ship in batches bounded by BOTH line count and byte size, so a backlog
            # drains in a few large POSTs instead of one-per-200-lines (which floods
            # the gateway when a model logs heavily / crash-loops).
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
                if sess is not None:
                    body["session"] = sess
                try:
                    await client.post(url, json=body)
                except httpx.HTTPError as e:
                    logger.warning("log shipper post failed: %s", e)

        while not drain_event.is_set():
            try:
                if get_log_path is not None:
                    latest = get_log_path()
                    if latest and latest != cur_path:
                        # Rotated to a fresh per-launch file. FIRST flush the dying
                        # file's remaining tail (the crash traceback!) under its OWN
                        # session — else those final, most-important lines are lost
                        # because we'd jump to the new file — THEN switch.
                        await drain(cur_path, session)
                        cur_path = latest
                        if fixed_session is None:
                            session = _session_of(cur_path)
                        offset = 0
                        leftover = b""
                await drain(cur_path, session)
            except Exception:
                logger.exception("log shipper iteration failed")
            try:
                await asyncio.wait_for(drain_event.wait(), timeout=BATCH_INTERVAL_S)
            except asyncio.TimeoutError:
                pass

        # Drain fired (graceful shutdown / reconciler drain / SIGTERM) → ship whatever
        # was written since the last tick BEFORE the process exits. Without this final
        # flush, a worker that boots and dies within one batch interval — common on a
        # slow node whose vLLM bootstrap stalls — ships NOTHING: its boot log + crash
        # reason never reach the gateway and the Workers tab shows a blank screen.
        try:
            if get_log_path is not None:
                latest = get_log_path()
                if latest and latest != cur_path:
                    await drain(cur_path, session)  # dying file's tail under its own session
                    cur_path = latest
                    if fixed_session is None:
                        session = _session_of(cur_path)
                    offset = 0
                    leftover = b""
            await drain(cur_path, session)
        except Exception:
            logger.exception("final log flush failed")


def _session_of(log_path: str | None) -> str | None:
    """Session id (trailing timestamp) of a timestamped log file, else None."""
    if not log_path:
        return None
    import re as _re
    m = _re.search(r"-(\d{8}-\d{6})\.log$", os.path.basename(log_path))
    return m.group(1) if m else None


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


# Max jobs a single-model worker runs at once. It used to serialize (one job
# fully finished before the next BRPOP), starving vLLM's continuous-batching
# engine — one in-flight request per worker no matter the GPU headroom. Now we
# dispatch each popped job as a semaphore-bounded child task, matching multi
# mode (dispatch.MAX_CONCURRENT_JOBS). Env-overridable for small/large boxes.
SINGLE_MODE_MAX_CONCURRENT_JOBS = int(
    os.environ.get("WORKER_MAX_CONCURRENT_JOBS", "64") or "64")


async def _dispatch_job(rdb, blob: str, machine_id: str, mode: str, model_id: str,
                        sem: asyncio.Semaphore, processing_key: str) -> None:
    """Run one popped job (parse → unary/stream), then remove it from the
    per-machine processing list. Result-writing + error handling live entirely
    in _run_unary/_run_stream (unchanged); this just bounds concurrency and does
    the processing-list bookkeeping. Never raises out (except cancellation)."""
    try:
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
        logger.exception("job handler crashed")
    finally:
        # Result written (or the job was poison/crashed) → drop it from the
        # processing list. If the worker instead DIES before reaching here, the
        # entry stays and can be recovered. See poll_loop for the follow-up note.
        try:
            await rdb.lrem(processing_key, 1, blob)
        except Exception:  # noqa: BLE001 — best-effort cleanup, never fail the job
            logger.warning("could not LREM job from %s", processing_key)
        sem.release()


async def poll_loop(rdb, queue_key: str, machine_id: str, mode: str, model_id: str, drain_event: asyncio.Event) -> None:
    logger.info("ready, polling %s", queue_key)
    sem = asyncio.Semaphore(SINGLE_MODE_MAX_CONCURRENT_JOBS)
    tasks: set[asyncio.Task] = set()
    # Reliability: BRPOPLPUSH atomically moves the popped job into a per-machine
    # processing list so a job in-flight when the worker dies (SIGKILL / OOM /
    # crash between pop and result-write) isn't silently lost — it survives in
    # `processing:{machine_id}`. We LREM it once the result is written (in
    # _dispatch_job). FOLLOW-UP (gateway-side, intentionally NOT done here to
    # avoid uncoordinated cross-service change): when a worker's heartbeat TTL
    # lapses, the gateway should requeue any leftover entries in
    # `processing:{machine_id}` back onto `queue:{app_id}` (and mark still-live
    # request ids) to fully close the at-least-once loop.
    processing_key = f"processing:{machine_id}"
    while not drain_event.is_set():
        try:
            blob = await rdb.brpoplpush(queue_key, processing_key, timeout=2)
            if blob is None:
                continue
            await sem.acquire()
            t = asyncio.create_task(
                _dispatch_job(rdb, blob, machine_id, mode, model_id, sem, processing_key))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("loop error, sleeping 1s")
            await asyncio.sleep(1.0)
    # Graceful shutdown: let in-flight jobs finish (each writes its result +
    # clears its processing-list entry) before returning.
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _run_unary(rdb, request_id, machine_id, mode, model_id, payload, timeout_s, endpoint="/v1/completions", base_url=None):
    try:
        output = await asyncio.wait_for(handle(mode, model_id, payload, endpoint, base_url=base_url), timeout=timeout_s)
        result = {"status": "completed", "output": output, "machine_id": machine_id, "node": node_meta()}
    except asyncio.TimeoutError:
        logger.warning("%s timed out after %ss", request_id, timeout_s)
        result = {
            "status": "timeout",
            "output": {"error": f"request exceeded timeout_s={timeout_s}"},
            "machine_id": machine_id,
            "node": node_meta(),
        }
    except UpstreamError as e:
        # vLLM errored — a FAILED job, not a completed one. Preserve vLLM's own
        # status/body (in e.payload) so the caller sees the real reason.
        logger.warning("%s upstream error: %s", request_id, e)
        result = {
            "status": "failed",
            "output": e.payload,
            "machine_id": machine_id,
            "node": node_meta(),
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
    elif isinstance(last, dict) and "error" in last:
        # handle_stream's terminal chunk carries an "error" field only when the
        # upstream vLLM call failed (raise_for_status / mid-stream drop). Don't
        # report a generation that errored as completed.
        status = "failed"
    else:
        status = "completed"
    final = {"status": status, "output": last, "machine_id": machine_id, "streamed": True, "node": node_meta()}
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

    if mode in ("multi", "proxy"):
        # proxy = a single-model VM endpoint the gateway proxies to directly over a
        # forward tunnel: same fleet machinery (launch + per-member log shipping +
        # auto-restart monitor + registration/heartbeat), but it does NOT consume
        # the job queue (no queue, no sleep — see _run_multi's queue_poll branch).
        await _run_multi(
            rdb, app_id, machine_id, gateway_url, drain_event,
            queue_poll=(mode == "multi"),
        )
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


async def _run_multi(rdb, app_id, machine_id, gateway_url, drain_event, queue_poll: bool = True) -> None:
    """WORKER_MODE=multi: launch the vLLM fleet, route jobs by model name, and
    evict idle models via sleep/wake to fit the GPU budget.

    `queue_poll=False` is the **proxy** mode (single-model VM endpoint): launch the
    one member and run all the same side loops (heartbeat, per-member + control-plane
    log shipping, metrics, the auto-restart monitor), but do NOT consume the job
    queue — the gateway proxies requests straight to the member's vLLM port over a
    forward tunnel. With no queue dispatch there's no acquire()/eviction, so the lone
    member (woken as the resident by sched.start()) simply stays awake → "no sleep"."""
    from .multi.config import parse_multi_config
    from .multi.scheduler import MultiModelScheduler
    from .multi.dispatch import multi_poll_loop

    cfg = parse_multi_config(
        os.environ.get("MULTI_MODEL_CONFIG"),
        os.environ.get("MULTI_MODEL_CONFIG_PATH"),
    )
    log_dir = os.environ.get("WORKER_LOG_DIR", "/var/log/vllm")
    sched = MultiModelScheduler(cfg, machine_id, log_dir=log_dir)
    logger.info("%s mode: %d models, %d GPUs", "multi" if queue_poll else "proxy",
                len(cfg.members), cfg.total_gpus)

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
                None, drain_event, source=m.served_name,
                # Follow the member's CURRENT log file — it changes to a fresh
                # timestamped file on every (re)launch; each becomes its own session.
                get_log_path=(lambda sn=m.served_name: (sched.resolve(sn).log_path if sched.resolve(sn) else None)),
            )
        )
        for m in cfg.members
    ]
    # Also ship the worker-agent's OWN stdout log (scheduler events: wave-loading,
    # sleep/wake, operator commands, preflight, dead reasons) under a reserved
    # "__worker__" source, so the UI can show the fleet's control-plane log too.
    self_log = os.environ.get("WORKER_SELF_LOG_PATH")
    if self_log:
        # One worker-log session per agent launch (provision): the agent's own
        # stdout file isn't timestamped (it's `worker-{mid}.log`, already unique
        # per provision), so pin a startup-timestamp session explicitly — keeps
        # historical worker logs across re-provisions, same as the model logs.
        worker_session = time.strftime("%Y%m%d-%H%M%S")
        log_tasks.append(asyncio.create_task(
            log_shipper_loop(gateway_url, machine_id, app_id, self_log, drain_event,
                             source="__worker__", fixed_session=worker_session)
        ))
    # Ship each member's vLLM /metrics to the gateway's combined scrape target.
    log_tasks.append(asyncio.create_task(
        metrics_shipper_loop(gateway_url, machine_id, app_id, sched.live_members, drain_event)
    ))
    monitor_task = asyncio.create_task(sched.monitor_loop(drain_event))
    try:
        await _redis_ready(rdb)
        await sched.start()
        if queue_poll:
            await multi_poll_loop(rdb, f"queue:{app_id}", machine_id, sched, drain_event)
        else:
            # Proxy mode: the gateway forwards HTTP straight to the member's vLLM
            # port (no queue consumer here). Stay up until drained; the monitor task
            # above keeps the lone engine healthy and auto-restarts it on crash.
            logger.info("proxy mode: model served directly (no queue); idling until drain")
            await drain_event.wait()
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
