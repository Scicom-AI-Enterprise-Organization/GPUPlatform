import asyncio
import base64
import json
import logging
import os
import re
import secrets
import shlex
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import redis.asyncio as redis_async
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from . import metrics
from . import accesslog
from .auth import (
    SECTIONS,
    create_session,
    current_user,
    has_section,
    hash_api_key,
    hash_password,
    mint_api_key,
    require_admin,
    require_developer,
    require_section,
    revoke_session,
    verify_password,
)
from .db import ApiKey, App, AuditLog, PolicyRole, Request as ReqRow, StressRun, User, WorkerEvent, get_session, init_db, get_user_by_username, list_all_apps, seed_admin_user, session_factory, shutdown_db
from . import audit as audit_module
from . import bench as bench_module
from . import compute as compute_module
from . import providers_api as providers_module
from . import storage_api as storage_module
from . import datasets_api as datasets_module
from . import global_env_api as global_env_module
from . import training_api as training_module
from . import tracking_creds_api as tracking_creds_module
from . import gitops_api as gitops_module
from . import proxy_api as proxy_module
from . import history_api as history_module
from . import catalog_api as catalog_module
from . import hf_mirror_api as hf_mirror_module

logger = logging.getLogger("gateway")

# Heartbeat liveness TTL for worker:{mid}. Long (1h) on purpose: a brief
# heartbeat gap — a Redis pod reschedule (AOF reload preserves a key whose 1h
# expiry hasn't elapsed), a tunnel blip, an engine stall — must NOT make the
# autoscaler think the worker died and provision a duplicate. The worker still
# heartbeats every 5s; this only governs how long a *silent* worker is presumed
# alive. Trade-off: a genuinely dead VM is presumed alive (and not auto-replaced)
# for up to this long. Override with WORKER_TTL_S env.
WORKER_TTL_S = int(os.environ.get("WORKER_TTL_S", "3600"))


async def _worker_ttl_s(rdb, app_id: str) -> int:
    """Per-app heartbeat TTL (set on the autoscaler config, mirrored to Redis by the
    autoscaler loop). Cheap GET on the heartbeat hot path; falls back to the global
    WORKER_TTL_S for older apps that predate the per-app setting."""
    try:
        raw = await rdb.get(f"app:{app_id}:worker_ttl_s")
        if raw:
            v = int(raw)
            if v > 0:
                return v
    except (TypeError, ValueError):
        pass
    return WORKER_TTL_S
# How long a client-disconnect cancel marker (cancel:{request_id}) lives. Must
# outlast the time a job can sit in the queue before the worker dequeues it —
# otherwise a 60s marker expires long before a deeply-queued request is picked
# up and the cancel is silently lost (worker then burns GPU on a dead request).
# Aligned with the result-key TTL (3600s); a job queued longer than that is dead
# anyway (its result key has expired too).
_CANCEL_TTL_S = 3600


class AutoscalerSpec(BaseModel):
    max_containers: int = 1
    tasks_per_container: int = 30
    idle_timeout_s: int = 300
    # Heartbeat liveness TTL for this app's workers. Long by default (1h) so a
    # brief heartbeat gap (Redis reschedule, tunnel blip, engine stall) can't make
    # the autoscaler presume the worker dead and spawn a duplicate. Falls back to
    # the global WORKER_TTL_S env default when unset on older apps.
    worker_ttl_s: int = 3600


class UpdateAutoscalerRequest(BaseModel):
    max_containers: Optional[int] = None
    tasks_per_container: Optional[int] = None
    idle_timeout_s: Optional[int] = None
    worker_ttl_s: Optional[int] = None
    vllm_args: Optional[str] = None
    gpu_count: Optional[int] = None


class MultiModelMember(BaseModel):
    # A model served by a multi-model endpoint. `model` is the HuggingFace id;
    # it doubles as the served-model-name clients send in payload["model"].
    model: str
    tp: int = 1                 # tensor-parallel size
    pp: int = 1                 # pipeline-parallel size; GPUs this model needs = tp * pp
    extra_args: str = ""        # per-model vLLM CLI args
    # "transcription" marks an audio/ASR (Whisper) model so the worker installs the
    # audio decode deps for it (name-independent — works for custom finetunes).
    task: Optional[str] = None
    # Optional explicit GPU pin: the physical ids (within `visible_devices`) this
    # model runs on, e.g. [0,1,2,3]. len must == tp * pp. None/empty = auto-pack
    # into the next free (tp*pp)-wide slot. Models with disjoint pins load
    # concurrently; overlapping pins time-share those GPUs via vLLM sleep/wake.
    gpu_indices: Optional[list[int]] = None


class UpdateModelsRequest(BaseModel):
    # Edit a multi-model fleet in place (PATCH /apps/{id}/models): the full new
    # member list (add/remove/retp/re-arg), optionally re-pinning the GPUs.
    models: list[MultiModelMember]
    sleep_level: Optional[int] = None
    visible_devices: Optional[str] = None
    # Optional setup script run once per worker boot before model launch. None
    # leaves it unchanged; "" clears it.
    pre_script: Optional[str] = None
    # Full `uv pip install` arg string for vLLM (overrides the version). None leaves
    # it unchanged; "" clears it.
    vllm_install_args: Optional[str] = None


class StressRunCreate(BaseModel):
    # A completed browser-driven stress run to persist (POST /apps/{id}/stress-runs).
    model: str = ""
    input_len: int
    output_len: int
    num_prompts: int
    concurrency: int
    summary: dict  # opaque client-computed metric block (throughput + percentiles)


class StressRunRecord(BaseModel):
    id: str
    app_id: str
    created_by: str
    model: str
    input_len: int
    output_len: int
    num_prompts: int
    concurrency: int
    summary: dict
    created_at: datetime


class CreateAppRequest(BaseModel):
    name: str
    model: str = ""
    gpu: str
    gpu_count: int = 1
    autoscaler: AutoscalerSpec = Field(default_factory=AutoscalerSpec)
    cpu: int = 2
    memory: str = "16Gi"
    request_timeout_s: int = 600
    vllm_args: str = ""
    enable_metrics: bool = True
    # RunPod cloud tier. None = provider default. Only "COMMUNITY" / "SECURE".
    cloud_type: Optional[str] = None
    # Per-worker disk sizing. None = provider default.
    container_disk_gb: Optional[int] = None
    volume_gb: Optional[int] = None
    # Per-app cloud-account selection (kind=runpod / pi / vm provider row).
    # NULL = use gateway-wide env keys.
    provider_id: Optional[str] = None
    # Serving mode. "single" (default) = one model. "multi" = a vLLM fleet on a
    # VM with sleep/wake eviction; requires a kind="vm" provider_id and `models`.
    mode: str = "single"
    models: Optional[list[MultiModelMember]] = None
    sleep_level: int = 1
    # Extra env applied to every vLLM process (HF_HOME, cache dirs, …). Absolute
    # path values are created on the worker before launch.
    env_vars: Optional[dict[str, str]] = None
    # VM-only GPU pin, e.g. "0,1,2,3". None/empty = all the VM's GPUs.
    visible_devices: Optional[str] = None
    # VM-only: uv venv the worker runs `vllm serve` from (e.g. "/share/vllm-venv").
    venv_path: Optional[str] = None
    # VM-only: pin vLLM to this version in venv_path (e.g. "0.19.1").
    vllm_version: Optional[str] = None
    # Full `uv pip install` arg string for vLLM, used verbatim instead of the version
    # (e.g. a nightly with extra index URLs). Overrides vllm_version when set.
    vllm_install_args: Optional[str] = None
    # Optional setup script the worker runs once after the venv is ready and before
    # launching models (e.g. `bash <(curl -fsSL …/install_deepgemm.sh)`).
    pre_script: Optional[str] = None


class CreateAppResponse(BaseModel):
    app_id: str
    url: str


class AppRecord(BaseModel):
    app_id: str
    name: str
    model: str
    gpu: str
    gpu_count: int = 1
    autoscaler: AutoscalerSpec
    cpu: int = 2
    memory: str = "16Gi"
    request_timeout_s: int = 600
    vllm_args: str = ""
    enable_metrics: bool = True
    cloud_type: Optional[str] = None
    container_disk_gb: Optional[int] = None
    volume_gb: Optional[int] = None
    provider_id: Optional[str] = None
    mode: str = "single"
    models: Optional[list[MultiModelMember]] = None
    sleep_level: int = 1
    env_vars: Optional[dict[str, str]] = None
    visible_devices: Optional[str] = None
    venv_path: Optional[str] = None
    vllm_version: Optional[str] = None
    vllm_install_args: Optional[str] = None
    pre_script: Optional[str] = None
    is_public: bool = False
    created_at: str
    owner: str


class RunResponse(BaseModel):
    request_id: str
    poll_url: str


class ResultResponse(BaseModel):
    request_id: str
    status: str
    output: Optional[Any] = None


class WorkerRegisterRequest(BaseModel):
    machine_id: str
    app_id: str
    token: str


class WorkerRegisterResponse(BaseModel):
    ok: bool
    redis_url: str


class WorkerHeartbeatRequest(BaseModel):
    machine_id: str
    app_id: str
    status: str = "ready"
    # Multi-model workers report per-model state here:
    #   [{"model","state","inflight","slot","last_used_ts","queue_len"}]
    models: Optional[list[dict]] = None


class WorkerLogsRequest(BaseModel):
    machine_id: str
    app_id: str
    lines: list[str] = Field(default_factory=list)
    # Multi-model workers tag each batch with the member's served_name so logs
    # bucket per model (one vLLM process per model). None → single-mode worker.
    source: Optional[str] = None
    # The member's current log SESSION (per-launch timestamp "YYYYMMDD-HHMMSS").
    # Changes on every (re)launch; lets the gateway keep each launch's log
    # separately (keyed by app+model+session, surviving re-provision) so the UI
    # can open historical logs. None → legacy single live buffer only.
    session: Optional[str] = None


class WorkerMetricsRequest(BaseModel):
    # A worker ships each member's raw vLLM /metrics (Prometheus text), keyed by
    # served_name, for the gateway's combined /metrics/workers scrape target.
    machine_id: str
    app_id: str
    metrics: dict[str, str] = Field(default_factory=dict)


class ModelActionRequest(BaseModel):
    """Operator action on a multi-model VM endpoint. `model` targets one member
    (kill/restart/sleep); it's ignored by the fleet-wide `sleep_all`."""
    model: Optional[str] = None
    action: str  # "kill" | "restart" | "sleep" | "sleep_all"


# Per-worker container log retention. The worker-agent ships batches every
# few seconds; we cap the list so a chatty worker can't blow up Redis.
WORKER_LOGS_CAP = 5000
# Historical per-launch log sessions are kept this long (keyed by app+model+
# session, so they survive a worker re-provision / mid change). Long enough to
# debug a crash days later, short enough that Redis self-cleans.
WLOG_SESSION_TTL = int(os.environ.get("WLOG_SESSION_TTL_S", str(7 * 24 * 3600)) or str(7 * 24 * 3600))


def _logs_slug(s: str) -> str:
    """Stable, Redis-key-safe slug for a model's served_name (used to bucket
    per-model logs). Mirrors nothing on the worker — the gateway owns the key."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s or "")[:128]


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    password: str = Field(min_length=8, max_length=128)
    email: str = Field(min_length=3, max_length=255, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class LoginRequest(BaseModel):
    # Accept either email or username so existing username-based clients keep
    # working. UI now uses email; older callers can still send username.
    email: Optional[str] = None
    username: Optional[str] = None
    password: str


class TokenResponse(BaseModel):
    token: str
    username: str


class WhoamiResponse(BaseModel):
    user_id: int
    username: str
    email: Optional[str] = None
    is_admin: bool = False
    role: str = "user"
    policy_role_id: Optional[str] = None
    # Effective per-section access — collapsed (admin → all true; developer
    # → resolved through policy role; user → all false).
    sections: dict[str, bool] = Field(default_factory=dict)


class UserRecord(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    role: str
    is_admin: bool
    policy_role_id: Optional[str] = None
    policy_role_name: Optional[str] = None
    section_permissions: dict[str, bool] = Field(default_factory=dict)
    created_at: str
    # "github" if the row was created via SSO, "password" otherwise. Derived
    # from the github_id column so the frontend can show a small badge on
    # the profile page without us inventing a separate column.
    auth_provider: str = "password"
    github_id: Optional[str] = None


class SetRoleRequest(BaseModel):
    role: str  # validated against {"user","developer","admin"} in the handler


class SetPolicyRoleRequest(BaseModel):
    # null/missing = detach (no sections)
    policy_role_id: Optional[str] = None


class PolicyRoleRecord(BaseModel):
    id: str
    name: str
    sections: dict[str, bool]
    is_system: bool
    created_at: str


class CreatePolicyRoleRequest(BaseModel):
    id: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9-]+$")
    name: str = Field(min_length=1, max_length=128)
    sections: dict[str, bool] = Field(default_factory=dict)


class UpdatePolicyRoleRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=128)
    sections: Optional[dict[str, bool]] = None


class AuditLogRecord(BaseModel):
    id: int
    actor_id: Optional[int] = None
    actor_username: str
    actor_email: Optional[str] = None  # current email of the actor (if still a user)
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    details: Optional[dict] = None
    created_at: str


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class GithubUpsertRequest(BaseModel):
    """Used by the web layer's GitHub OAuth callback. The web has just
    completed the OAuth code exchange and verified the user; we trust it
    (auth via INTERNAL_AUTH_TOKEN header) to mint a session for that user."""
    github_id: str = Field(min_length=1, max_length=64)
    login: str = Field(min_length=1, max_length=64)  # GitHub username
    email: Optional[str] = Field(default=None, max_length=255)
    name: Optional[str] = Field(default=None, max_length=128)


def _to_app_record(app: App, *, redacted: bool = False) -> AppRecord:
    """Serialize an App. With redacted=True (a non-owner viewing a *public*
    endpoint read-only), the infra/secret-bearing fields are blanked so a public
    viewer never sees env vars, launch args, pre-scripts, the venv path, or the
    cloud-account binding — only the shape of the endpoint (name/model/gpu/scale).
    Per-member vLLM args (which can carry tokens) are stripped too."""
    members = [MultiModelMember(**m) for m in (getattr(app, "models", None) or [])] or None
    if redacted and members:
        members = [m.model_copy(update={"extra_args": ""}) for m in members]
    return AppRecord(
        app_id=app.app_id,
        name=app.name,
        model=app.model,
        gpu=app.gpu,
        gpu_count=getattr(app, "gpu_count", 1) or 1,
        autoscaler=AutoscalerSpec(**app.autoscaler),
        cpu=app.cpu,
        memory=app.memory,
        request_timeout_s=app.request_timeout_s,
        vllm_args="" if redacted else (app.vllm_args or ""),
        enable_metrics=bool(getattr(app, "enable_metrics", True)),
        cloud_type=getattr(app, "cloud_type", None),
        container_disk_gb=getattr(app, "container_disk_gb", None),
        volume_gb=getattr(app, "volume_gb", None),
        provider_id=None if redacted else getattr(app, "provider_id", None),
        mode=getattr(app, "mode", "single") or "single",
        models=members,
        sleep_level=int(getattr(app, "sleep_level", 1) or 1),
        env_vars=None if redacted else (getattr(app, "env_vars", None) or None),
        visible_devices=None if redacted else (getattr(app, "visible_devices", None) or None),
        venv_path=None if redacted else (getattr(app, "venv_path", None) or None),
        vllm_version=getattr(app, "vllm_version", None) or None,
        vllm_install_args=None if redacted else (getattr(app, "vllm_install_args", None) or None),
        pre_script=None if redacted else (getattr(app, "pre_script", None) or None),
        is_public=bool(getattr(app, "is_public", False)),
        created_at=app.created_at.isoformat() if app.created_at else "",
        owner=app.owner.username if app.owner else "",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    accesslog.init_access_logging()  # idempotent; also covers the uvicorn reload subprocess
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")
    logger.info("connecting to redis at %s", redis_url)
    # Keepalive matters: prod Redis sits behind a public ELB with a ~60s idle
    # timeout. Without health checks, the SSE relay's pubsub connection (gen()
    # in _openai_endpoint) gets silently dropped by the ELB during a brief lull
    # in publishing; with no socket_timeout, pubsub.listen() then hangs on the
    # dead socket for ~60s before recovery — surfacing as a mid-stream SSE stall
    # that "resumes" after ~60s. health_check_interval sends periodic PINGs that
    # keep the connection under the idle timeout AND detect a dropped socket
    # fast; TCP keepalive is a second line of defence. (Tunable via env.)
    _redis_hc = int(os.environ.get("REDIS_HEALTH_CHECK_INTERVAL_S", "30") or "30")
    app.state.redis = redis_async.from_url(
        redis_url,
        decode_responses=True,
        health_check_interval=_redis_hc,
        socket_keepalive=True,
    )
    await app.state.redis.ping()
    logger.info("redis ready (health_check_interval=%ss, socket_keepalive=on)", _redis_hc)

    logger.info("initializing postgres")
    await init_db()
    await seed_admin_user()
    logger.info("postgres ready")

    # Warm the global-secret cache so sync credential resolvers (S3 key refs)
    # never see a cold cache before the first async load_global_env runs.
    try:
        from .global_env_api import load_global_env
        async with session_factory()() as _s:
            await load_global_env(_s)
    except Exception:  # noqa: BLE001 — best-effort; refreshed on first use anyway
        logger.warning("global-secret cache warm-up failed (non-fatal)", exc_info=True)

    app.state.provider = None
    app.state.provider_cache = {}
    app.state.autoscaler_task = None
    app.state.reconciler_task = None
    app.state.bench_janitor_task = None
    app.state.compute_idle_task = None
    app.state.gitops_task = None
    if os.environ.get("BENCHMARK_S3_BUCKET", "").strip():
        # Materialize SSH key from env (prod) before anything else uses it.
        bench_module.bootstrap_ssh_key_from_env()
        orphaned = await bench_module.cleanup_orphaned_running(app.state.redis)
        if orphaned:
            logger.warning(
                "bench: marked %d previously-running benchmark(s) as failed (gateway restart). "
                "Check RunPod for any pods left billing.", orphaned,
            )
        app.state.bench_janitor_task = asyncio.create_task(
            bench_module.janitor_loop(app.state.redis)
        )
        logger.info("bench enabled (bucket=%s)", os.environ.get("BENCHMARK_S3_BUCKET"))
    if os.environ.get("RUNPOD_API_KEY", "").strip():
        # Compute uses the same RunPod creds; cleanup any rows the previous
        # gateway process was mid-creating. Running pods stay running on
        # RunPod and the row keeps its `running` status.
        orphaned = await compute_module.cleanup_orphaned_running()
        if orphaned:
            logger.warning(
                "compute: marked %d previously-creating pod(s) as failed (gateway restart). "
                "Check RunPod dashboard for any pod still billing.", orphaned,
            )
        # Idle auto-terminate sweep — tears down pods left idle (no GPU/memory
        # use) past their per-pod window. No-op for pods with the window unset.
        app.state.compute_idle_task = asyncio.create_task(
            compute_module.compute_idle_loop()
        )
        logger.info("compute enabled")
    # Autotrain: reconcile training runs left 'running'/'queued' by a previous
    # gateway process. Trainers run detached on the VM/pod, so a restart no longer
    # kills them — cleanup SSH-checks each (alive → kept running; exited → finalized
    # from its log; unreachable → failed), and the janitor finalizes survivors when
    # they later exit. Runs without the env bucket (runs can target per-storage S3).
    try:
        t_orphaned = await training_module.cleanup_orphaned_running(app.state.redis)
        if t_orphaned:
            logger.warning("autotrain: finalized %d orphaned training run(s) after gateway restart", t_orphaned)
        app.state.training_janitor_task = asyncio.create_task(
            training_module.training_janitor_loop(app.state.redis)
        )
    except Exception:
        logger.exception("autotrain: orphan reconcile failed")
    # GitOps: reconcile platform resources declared in registered git repos.
    # The loop self-disables with GITOPS_POLL=0; manual sync + webhook always work.
    try:
        app.state.gitops_task = asyncio.create_task(
            gitops_module.gitops_reconcile_loop(app)
        )
    except Exception:
        logger.exception("gitops: failed to start poll loop")
    # LLM API proxy: shared httpx client + live state + health-check loop.
    app.state.proxy_http = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=10.0),
        follow_redirects=True,
    )
    app.state.proxy_health = {}
    app.state.proxy_live = {}
    app.state.proxy_sems = {}
    app.state.proxy_healthcheck_task = None
    try:
        app.state.proxy_healthcheck_task = asyncio.create_task(
            proxy_module.proxy_health_loop(app)
        )
    except Exception:
        logger.exception("proxy: failed to start health loop")
    if os.environ.get("AUTOSCALER", "0") == "1":
        from .provider import build_provider, cloud_providers_disabled, CLOUD_PROVIDER_NAMES
        from .autoscaler import autoscaler_loop, vm_watchdog_loop
        from .reconciler import reconciler_loop

        provider_name = os.environ.get("PROVIDER", "fake")

        # vm-only / cloud-disabled deployments (CAE/CCE) have no global cloud
        # provider: vm workers are per-app provider rows (resolve_app_provider
        # builds a VMProvider for each), so the autoscaler's global fallback is
        # None and every app must reference its own vm provider. build_provider()
        # has no 'vm' form, so we skip it rather than crash on "unknown provider".
        if cloud_providers_disabled() and provider_name in CLOUD_PROVIDER_NAMES:
            raise RuntimeError(
                f"PROVIDER={provider_name} but DISABLE_CLOUD_PROVIDERS is set — "
                f"set PROVIDER=vm (per-app providers) or PROVIDER=fake"
            )

        if provider_name == "vm":
            app.state.provider = None
        else:
            if provider_name == "primeintellect":
                missing = []
                for var in ("PI_API_KEY", "PI_CUSTOM_TEMPLATE_ID", "GATEWAY_PUBLIC_URL"):
                    v = os.environ.get(var, "")
                    if not v or v in ("replace-me", "changeme"):
                        missing.append(var)
                if missing:
                    raise RuntimeError(
                        f"PROVIDER=primeintellect requires {missing} to be set "
                        f"(or set PROVIDER=fake for local dev with no real GPU)"
                    )

            if provider_name == "runpod":
                missing = []
                for var in ("RUNPOD_API_KEY", "RUNPOD_TEMPLATE_ID", "GATEWAY_PUBLIC_URL"):
                    v = os.environ.get(var, "")
                    if not v or v in ("replace-me", "changeme"):
                        missing.append(var)
                if missing:
                    raise RuntimeError(
                        f"PROVIDER=runpod requires {missing} to be set "
                        f"(or set PROVIDER=fake for local dev with no real GPU)"
                    )

            app.state.provider = build_provider(provider_name)
        # Per-app provider instances (kind=vm/runpod/pi from a providers row) are
        # cached in app.state.provider_cache (initialised above) and shared by
        # the autoscaler + reconciler so we don't re-decrypt creds every tick.
        # (provider may be None for PROVIDER=vm — apps then resolve their own row.)
        app.state.autoscaler_task = asyncio.create_task(
            autoscaler_loop(
                app.state.redis, app.state.provider, session_factory(), app.state.provider_cache
            )
        )
        app.state.reconciler_task = asyncio.create_task(
            reconciler_loop(
                app.state.redis, app.state.provider, session_factory(), app.state.provider_cache
            )
        )
        app.state.vm_watchdog_task = asyncio.create_task(
            vm_watchdog_loop(
                app.state.redis, session_factory(), app.state.provider_cache
            )
        )
        logger.info(
            "autoscaler + reconciler + vm_watchdog enabled (provider=%s)",
            app.state.provider.name if app.state.provider else f"{provider_name} (per-app rows)",
        )

    try:
        yield
    finally:
        for task_attr in ("autoscaler_task", "reconciler_task", "vm_watchdog_task", "bench_janitor_task", "compute_idle_task", "gitops_task", "proxy_healthcheck_task"):
            t = getattr(app.state, task_attr, None)
            if t:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, BaseException):
                    pass
        proxy_http = getattr(app.state, "proxy_http", None)
        if proxy_http is not None:
            try:
                await proxy_http.aclose()
            except Exception:
                pass
        if app.state.provider:
            await app.state.provider.shutdown()
        await app.state.redis.aclose()
        await shutdown_db()


app = FastAPI(
    title="serverless-gpu gateway",
    lifespan=lifespan,
    # Swagger / ReDoc UIs stay off (the web UI ships its own /api-docs), but the
    # machine-readable schema at /openapi.json is served publicly — it carries no
    # auth dependency, so any client / tool can fetch it without a token.
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",
)
app.include_router(bench_module.router)
app.include_router(compute_module.router)
app.include_router(providers_module.router)
app.include_router(storage_module.router)
app.include_router(datasets_module.router)
app.include_router(global_env_module.router)
app.include_router(training_module.router)
app.include_router(tracking_creds_module.router)
app.include_router(gitops_module.router)
app.include_router(proxy_module.router)
app.include_router(proxy_module.data_router)
app.include_router(history_module.router)
app.include_router(catalog_module.router)
app.include_router(hf_mirror_module.router)


@app.middleware("http")
async def metrics_mw(request: Request, call_next):
    metrics.INFLIGHT.inc()
    # Correlate metrics ↔ logs ↔ client. Honour an upstream X-Request-ID, else mint one.
    request_id = request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex[:12]}"
    request.state.request_id = request_id
    start = time.perf_counter()
    status_code = 500  # default if call_next raises before producing a response
    resp = None
    try:
        resp = await call_next(request)
        status_code = resp.status_code
        resp.headers["x-request-id"] = request_id
        return resp
    finally:
        elapsed = time.perf_counter() - start
        route_obj = request.scope.get("route")
        route = getattr(route_obj, "path", request.url.path) if route_obj else request.url.path
        # Serverless-API HTTP instrumentation (serverless_http_requests_total +
        # serverless_http_request_duration_seconds, labelled method/route/http_status/app_id).
        # Match by route TEMPLATE (collapsing unmatched paths so 404 scanners can't
        # explode cardinality) and skip probes + the metrics scrapes themselves.
        endpoint = getattr(route_obj, "path", None) or "<unmatched>"
        app_id = (request.scope.get("path_params") or {}).get("app_id", "") or ""
        # Recorded in finally so exceptions (status 500) still count.
        metrics.REQUESTS_TOTAL.labels(route=route, status=str(status_code)).inc()
        if endpoint not in metrics.IGNORE_PATHS:
            metrics.observe_http(request.method, endpoint, status_code, app_id, elapsed)
        metrics.INFLIGHT.dec()
        # Structured access log (per-app filterable in Loki).
        fwd = request.headers.get("x-forwarded-for")
        ip = (fwd.split(",")[0].strip() if fwd else None) or (
            request.client.host if request.client else None
        )
        nbytes = None
        if resp is not None and resp.headers.get("content-length"):
            try:
                nbytes = int(resp.headers["content-length"])
            except (TypeError, ValueError):
                nbytes = None
        accesslog.log_request(
            method=request.method,
            route=route,
            path=request.url.path,
            status=status_code,
            duration_ms=elapsed * 1000.0,
            request_id=request_id,
            app_id=app_id or None,
            ip=ip,
            nbytes=nbytes,
        )


# The build the gateway is serving. CI bakes the git short-sha in as APP_VERSION
# at image-build time (see the Dockerfile + ci.yml build-args); "dev" for local /
# unbaked runs.
GATEWAY_VERSION = os.environ.get("APP_VERSION", "dev")


# Routes that genuinely need no token — kept off the global security requirement
# below so /openapi.json doesn't read as if login itself is gated.
_OPENAPI_PUBLIC_PATHS = {
    "/", "/health", "/ready", "/version", "/openapi.json", "/v1/models",
    "/auth/login", "/auth/register", "/auth/github/upsert",
}


def _custom_openapi():
    """FastAPI can't infer auth from our hand-rolled `current_user` dependency, so
    the generated schema would show every route as public. Inject an HTTP-bearer
    security scheme + a global requirement (minus the public routes) so the spec
    honestly documents `Authorization: Bearer <sgpu_… key | session token>`."""
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=GATEWAY_VERSION,
        description=(
            "Serverless-GPU control-plane API. Authenticate with an API key "
            "(prefix `sgpu_`, from the API tokens page) sent as "
            "`Authorization: Bearer <key>`."
        ),
        routes=app.routes,
    )
    schema.setdefault("components", {})["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "description": "API key (prefix sgpu_) or a session token.",
        },
    }
    schema["security"] = [{"bearerAuth": []}]
    for path, ops in schema.get("paths", {}).items():
        if path in _OPENAPI_PUBLIC_PATHS:
            for op in ops.values():
                if isinstance(op, dict):
                    op["security"] = []
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi  # type: ignore[method-assign]


@app.get("/")
async def root():
    """Service identity at the bare URL. Workers reach the gateway via their
    `GATEWAY_URL` (on a VM, the reverse-tunnelled loopback) and ping the root as a
    reachability check — without this they'd get a 404 that spams the access log.
    A plain 200 confirms reachability; real probes use /health, /ready, /version."""
    return {"service": "serverless-gpu-gateway", "version": GATEWAY_VERSION, "ok": True}


@app.get("/health")
async def health():
    return {"ok": True, "version": GATEWAY_VERSION}


@app.get("/version")
async def version():
    """Which build is live — so you can confirm what the platform is serving."""
    return {"version": GATEWAY_VERSION}


@app.get("/ready")
async def ready(request: Request):
    rdb = request.app.state.redis
    try:
        await rdb.ping()
    except Exception as e:
        raise HTTPException(status_code=503, detail={"redis": "unreachable", "error": str(e)[:200]})
    return {"ok": True, "redis": "ok"}


@app.get("/metrics")
async def metrics_endpoint(request: Request):
    body, ctype = await metrics.render(request.app.state.redis)
    return Response(content=body, media_type=ctype)


@app.get("/metrics/resources")
async def resource_metrics_endpoint(request: Request):
    """Prometheus exporter for platform resources (serverless apps, benchmarks,
    storage, datasets, GPU providers, GitOps) sampled from Postgres — separate
    from the infra `/metrics` target. The web app re-exposes this at
    `/api/metrics` for scrapers hitting the web host. Path is `/metrics/resources`
    (not `/api/metrics`) so it can't be shadowed by the `/{app_id}/metrics` route.
    Autotrain runs are the headline: alert on a NEW failure with
    `increase(platform_autotrain_runs_finished_total{status="failed"}[10m]) > 0`.
    Auth-exempt like /metrics — gate via ingress/network if needed."""
    async with session_factory()() as session:
        body, ctype = await metrics.render_resources(session, request.app.state.redis)
    return Response(content=body, media_type=ctype)


# ----- auth -----

@app.post("/auth/register", response_model=TokenResponse)
async def register(req: RegisterRequest, request: Request, session: AsyncSession = Depends(get_session)):
    user = User(
        username=req.username,
        email=req.email.lower(),
        password_hash=hash_password(req.password),
    )
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=409, detail={"error": "username or email already taken"})
    await session.refresh(user)
    token = await create_session(session, user.id)
    logger.info("registered user %s (id=%d)", user.username, user.id)
    return TokenResponse(token=token, username=user.username)


@app.post("/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest, request: Request, session: AsyncSession = Depends(get_session)):
    if not req.email and not req.username:
        raise HTTPException(status_code=400, detail={"error": "email or username required"})
    user = None
    if req.email:
        from sqlalchemy import select
        result = await session.execute(select(User).where(User.email == req.email.lower()))
        user = result.scalar_one_or_none()
    if user is None and req.username:
        user = await get_user_by_username(session, req.username)
    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail={"error": "invalid credentials"})
    token = await create_session(session, user.id)
    return TokenResponse(token=token, username=user.username)


@app.post("/auth/change-password")
async def change_password(
    req: ChangePasswordRequest,
    request: Request,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    if not verify_password(req.current_password, user.password_hash):
        raise HTTPException(status_code=401, detail={"error": "current password is wrong"})
    user.password_hash = hash_password(req.new_password)
    await session.commit()
    # Invalidate the current session so the user re-logs with the new password.
    header = request.headers.get("authorization", "")
    token = header[len("Bearer "):].strip() if header.startswith("Bearer ") else ""
    if token:
        await revoke_session(session, token)
    logger.info("password changed: user=%s", user.username)
    return {"ok": True}


@app.post("/auth/github/upsert", response_model=TokenResponse)
async def github_upsert(
    req: GithubUpsertRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Mint a session for a GitHub user. Looks up by github_id first, then
    by email; creates the row if neither matches. Called server-to-server
    by the web's OAuth callback — gated by a shared INTERNAL_AUTH_TOKEN
    so the public can't bypass password auth by hitting this directly."""
    expected = os.environ.get("INTERNAL_AUTH_TOKEN", "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail={"error": "INTERNAL_AUTH_TOKEN not configured on gateway"},
        )
    presented = request.headers.get("x-internal-token", "").strip()
    if not secrets.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail={"error": "internal auth failed"})

    from sqlalchemy import select as _select
    user: Optional[User] = None
    # Match by github_id (stable) first.
    res = await session.execute(_select(User).where(User.github_id == req.github_id))
    user = res.scalar_one_or_none()
    if user is None and req.email:
        # Fallback: link an existing local account that registered with the
        # same email. Updates the row's github_id so future logins go through
        # the fast path above.
        res = await session.execute(_select(User).where(User.email == req.email.lower()))
        user = res.scalar_one_or_none()
        if user is not None:
            user.github_id = req.github_id
    if user is None:
        # First-time sign-in via GitHub. Create the row with a random,
        # unusable password hash — they can't password-login this account
        # (and shouldn't need to, since they SSO).
        username = req.login
        # If GitHub login collides with an existing local username, append
        # a short suffix so we don't 409 on the unique index.
        existing = await get_user_by_username(session, username)
        if existing is not None:
            username = f"{req.login}-{secrets.token_hex(3)}"
        user = User(
            username=username,
            email=(req.email.lower() if req.email else None),
            password_hash=hash_password(secrets.token_urlsafe(32)),
            github_id=req.github_id,
            role="user",  # admins promote manually via /admin/users
        )
        session.add(user)
    await session.commit()
    await session.refresh(user)
    token = await create_session(session, user.id)
    logger.info("github sso: user=%s id=%d gh=%s", user.username, user.id, req.github_id)
    return TokenResponse(token=token, username=user.username)


@app.post("/auth/logout")
async def logout(request: Request, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)):
    header = request.headers.get("authorization", "")
    token = header[len("Bearer "):].strip()
    await revoke_session(session, token)
    return {"ok": True}


# ----- API keys -----
# Long-lived, revocable tokens for scripting the platform API. A key
# authenticates as its owner and inherits that user's role + section access.


class CreateApiKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)


class ApiKeyRecord(BaseModel):
    id: str
    name: str
    prefix: str
    created_at: str
    last_used_at: Optional[str] = None


class CreateApiKeyResponse(ApiKeyRecord):
    # The full plaintext key — returned ONCE at creation, never again.
    key: str


@app.get("/api-keys", response_model=list[ApiKeyRecord])
async def list_api_keys(
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    from sqlalchemy import select
    rows = (
        await session.execute(
            select(ApiKey)
            .where(ApiKey.owner_id == user.id, ApiKey.revoked_at.is_(None))
            .order_by(ApiKey.created_at.desc())
        )
    ).scalars().all()
    return [
        ApiKeyRecord(
            id=k.id,
            name=k.name,
            prefix=k.prefix,
            created_at=k.created_at.isoformat() if k.created_at else "",
            last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
        )
        for k in rows
    ]


@app.post("/api-keys", response_model=CreateApiKeyResponse)
async def create_api_key(
    req: CreateApiKeyRequest,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    raw = mint_api_key()
    rec = ApiKey(
        id=f"ak-{uuid.uuid4().hex[:12]}",
        owner_id=user.id,
        name=req.name.strip(),
        prefix=raw[:12],
        key_hash=hash_api_key(raw),
        created_at=datetime.now(timezone.utc),
    )
    session.add(rec)
    await session.commit()
    await audit_module.record(user, "apikey.create", "apikey", rec.id, rec.name)
    logger.info("api key created: id=%s name=%s user=%s", rec.id, rec.name, user.username)
    return CreateApiKeyResponse(
        id=rec.id,
        name=rec.name,
        prefix=rec.prefix,
        created_at=rec.created_at.isoformat(),
        last_used_at=None,
        key=raw,
    )


@app.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    rec = await session.get(ApiKey, key_id)
    if rec is None or rec.owner_id != user.id or rec.revoked_at is not None:
        raise HTTPException(status_code=404, detail="no such api key")
    rec.revoked_at = datetime.now(timezone.utc)
    await session.commit()
    await audit_module.record(user, "apikey.revoke", "apikey", rec.id, rec.name)
    return {"ok": True, "id": key_id}


async def _user_to_record(u: User, session: AsyncSession) -> UserRecord:
    role_obj = (
        await session.get(PolicyRole, u.policy_role_id) if u.policy_role_id else None
    )
    sections = {s: await has_section(u, s, session) for s in SECTIONS}
    return UserRecord(
        id=u.id,
        username=u.username,
        email=u.email,
        role=u.role,
        is_admin=u.is_admin,
        policy_role_id=u.policy_role_id,
        policy_role_name=role_obj.name if role_obj else None,
        section_permissions=sections,
        created_at=u.created_at.isoformat() if u.created_at else "",
        auth_provider="github" if u.github_id else "password",
        github_id=u.github_id,
    )


@app.get("/auth/me", response_model=WhoamiResponse)
async def whoami(
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
):
    sections = {s: await has_section(user, s, session) for s in SECTIONS}
    return WhoamiResponse(
        user_id=user.id,
        username=user.username,
        email=user.email,
        is_admin=user.is_admin,
        role=user.role,
        policy_role_id=user.policy_role_id,
        sections=sections,
    )


# ----- admin: tier role + policy role management -----

@app.get("/admin/users", response_model=list[UserRecord])
async def list_users(
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    from sqlalchemy import select
    result = await session.execute(select(User).order_by(User.id))
    out: list[UserRecord] = []
    for u in result.scalars().all():
        out.append(await _user_to_record(u, session))
    return out


@app.get("/admin/users/{user_id}", response_model=UserRecord)
async def get_user(
    user_id: int,
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    target = await session.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="user not found")
    return await _user_to_record(target, session)


@app.delete("/admin/users/{user_id}")
async def delete_user(
    user_id: int,
    actor: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    if user_id == actor.id:
        raise HTTPException(status_code=400, detail={"error": "you cannot delete yourself"})
    target = await session.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="user not found")
    username = target.username
    await session.delete(target)
    await session.commit()
    logger.info("user deleted: actor=%s target=%s", actor.username, username)
    await audit_module.record(actor, "user.delete", "user", str(user_id), username)
    return {"ok": True, "username": username}


@app.patch("/admin/users/{user_id}/role", response_model=UserRecord)
async def set_user_role(
    user_id: int,
    req: SetRoleRequest,
    actor: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    """Tier role: user / developer / admin. Promoting to developer for the
    first time auto-attaches the `full-access` policy role so the user can
    actually do anything. Admin overrides everything anyway."""
    if req.role not in ("user", "developer", "admin"):
        raise HTTPException(status_code=400, detail={"error": "role must be one of: user, developer, admin"})
    target = await session.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="user not found")
    if target.id == actor.id and req.role != "admin":
        raise HTTPException(status_code=400, detail={"error": "you cannot demote yourself"})
    old_role = target.role
    target.role = req.role
    target.is_admin = req.role == "admin"
    if req.role in ("developer", "admin") and target.policy_role_id is None:
        target.policy_role_id = "full-access"
    await session.commit()
    await session.refresh(target)
    logger.info("role change: actor=%s target=%s old=%s new=%s",
                actor.username, target.username, old_role, target.role)
    await audit_module.record(
        actor, "user.role_change", "user", str(user_id), target.username,
        details={"old_role": old_role, "new_role": req.role},
    )
    return await _user_to_record(target, session)


@app.patch("/admin/users/{user_id}/policy-role", response_model=UserRecord)
async def set_user_policy_role(
    user_id: int,
    req: SetPolicyRoleRequest,
    actor: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    """Attach (or detach with null) a policy role to a user."""
    target = await session.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="user not found")
    if req.policy_role_id:
        role = await session.get(PolicyRole, req.policy_role_id)
        if role is None:
            raise HTTPException(status_code=404, detail={"error": "policy role not found"})
    old_role_id = target.policy_role_id
    target.policy_role_id = req.policy_role_id
    await session.commit()
    await session.refresh(target)
    await audit_module.record(
        actor, "user.permissions_change", "user", str(user_id), target.username,
        details={"old_policy_role_id": old_role_id, "new_policy_role_id": req.policy_role_id},
    )
    return await _user_to_record(target, session)


# ----- admin: policy roles CRUD -----

@app.get("/admin/policy-roles", response_model=list[PolicyRoleRecord])
async def list_policy_roles(
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    from sqlalchemy import select
    rows = await session.execute(select(PolicyRole).order_by(PolicyRole.created_at))
    return [
        PolicyRoleRecord(
            id=r.id,
            name=r.name,
            sections={s: bool((r.sections or {}).get(s, False)) for s in SECTIONS},
            is_system=r.is_system,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in rows.scalars().all()
    ]


@app.post("/admin/policy-roles", response_model=PolicyRoleRecord)
async def create_policy_role(
    req: CreatePolicyRoleRequest,
    actor: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    if await session.get(PolicyRole, req.id):
        raise HTTPException(status_code=409, detail={"error": "policy role id already exists"})
    sections = {s: bool(req.sections.get(s, False)) for s in SECTIONS}
    row = PolicyRole(id=req.id, name=req.name.strip(), sections=sections, is_system=False)
    session.add(row)
    await session.commit()
    await session.refresh(row)
    await audit_module.record(
        actor, "policy_role.create", "policy_role", row.id, row.name,
        details={"sections": sections},
    )
    return PolicyRoleRecord(
        id=row.id, name=row.name, sections=sections,
        is_system=row.is_system,
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


@app.patch("/admin/policy-roles/{role_id}", response_model=PolicyRoleRecord)
async def update_policy_role(
    role_id: str,
    req: UpdatePolicyRoleRequest,
    actor: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    row = await session.get(PolicyRole, role_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"error": "policy role not found"})
    changes: dict[str, Any] = {}
    if req.name is not None:
        new_name = req.name.strip()
        if new_name and new_name != row.name:
            row.name = new_name
            changes["name"] = new_name
    if req.sections is not None:
        cur = dict(row.sections or {})
        for k, v in req.sections.items():
            if k in SECTIONS:
                cur[k] = bool(v)
        row.sections = cur
        flag_modified(row, "sections")
        changes["sections"] = cur
    if not changes:
        raise HTTPException(status_code=400, detail={"error": "no changes"})
    await session.commit()
    await session.refresh(row)
    await audit_module.record(
        actor, "policy_role.update", "policy_role", row.id, row.name,
        details=changes,
    )
    return PolicyRoleRecord(
        id=row.id, name=row.name,
        sections={s: bool((row.sections or {}).get(s, False)) for s in SECTIONS},
        is_system=row.is_system,
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


@app.delete("/admin/policy-roles/{role_id}")
async def delete_policy_role(
    role_id: str,
    actor: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
):
    row = await session.get(PolicyRole, role_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"error": "policy role not found"})
    if row.is_system:
        raise HTTPException(
            status_code=400,
            detail={"error": "system roles cannot be deleted (you can edit their sections instead)"},
        )
    name = row.name
    await session.delete(row)
    await session.commit()
    await audit_module.record(
        actor, "policy_role.delete", "policy_role", role_id, name,
    )
    return {"ok": True, "id": role_id}


# ----- admin: audit log list -----

@app.get("/admin/audit-logs", response_model=list[AuditLogRecord])
async def list_audit_logs(
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
    limit: int = 200,
    actor: Optional[str] = None,
    resource_type: Optional[str] = None,
    action: Optional[str] = None,
):
    """Most recent events first. Filters compose with AND. Caps at 1000 to
    keep the admin page responsive — older history can be paged later."""
    from sqlalchemy import select, desc
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be 1..1000")
    stmt = select(AuditLog).order_by(desc(AuditLog.created_at)).limit(limit)
    if actor:
        stmt = stmt.where(AuditLog.actor_username == actor)
    if resource_type:
        stmt = stmt.where(AuditLog.resource_type == resource_type)
    if action:
        stmt = stmt.where(AuditLog.action == action)
    rows = list((await session.execute(stmt)).scalars().all())
    # The audit row only snapshots the username; resolve each actor's current
    # email from the users table (NULL for since-deleted users).
    actor_ids = {r.actor_id for r in rows if r.actor_id is not None}
    emails: dict[int, Optional[str]] = {}
    if actor_ids:
        ures = await session.execute(select(User.id, User.email).where(User.id.in_(actor_ids)))
        emails = {uid: email for uid, email in ures.all()}
    return [
        AuditLogRecord(
            id=r.id,
            actor_id=r.actor_id,
            actor_username=r.actor_username,
            actor_email=emails.get(r.actor_id) if r.actor_id is not None else None,
            action=r.action,
            resource_type=r.resource_type,
            resource_id=r.resource_id,
            resource_name=r.resource_name,
            details=r.details,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in rows
    ]


# ----- apps (owner-scoped) -----

@app.get("/v1/availability")
async def get_gpu_availability(
    request: Request,
    gpu: str,
    count: int = 1,
    cloud_type: Optional[str] = None,
    # Cross-section: both Inference and Benchmark forms call this, so we gate
    # on the broader developer role instead of a specific section.
    user: User = Depends(require_developer),
):
    """Live check whether `count` of `gpu` can be provisioned right now on
    the active provider. UI uses this to render a green/red/yellow badge
    next to the GPU picker. Provider-side caches keep upstream RPS bounded.

    `cloud_type` is optional and provider-specific (RunPod: COMMUNITY/SECURE).
    When omitted the provider's configured default is used.
    """
    if count < 1 or count > 8:
        raise HTTPException(status_code=400, detail="count must be 1..8")
    if not gpu or len(gpu) > 64:
        raise HTTPException(status_code=400, detail="gpu name required (≤64 chars)")
    if cloud_type is not None and cloud_type.upper() not in ("COMMUNITY", "SECURE"):
        raise HTTPException(status_code=400, detail="cloud_type must be COMMUNITY or SECURE")
    provider = getattr(request.app.state, "provider", None)
    if provider is None:
        return {
            "gpu": gpu, "count": count, "available": True,
            "cheapest_price_hr": None, "regions": [], "reason": None,
            "checked_at": time.time(), "provider": "fake",
        }
    try:
        result = await provider.check_availability(gpu, count, cloud_type=cloud_type)
    except Exception:
        logger.exception("availability check failed for %s x%d", gpu, count)
        return {
            "gpu": gpu, "count": count, "available": None,
            "cheapest_price_hr": None, "regions": [], "reason": "internal error",
            "checked_at": time.time(), "provider": getattr(provider, "name", "unknown"),
        }
    return {
        "gpu": result.gpu,
        "count": result.count,
        "available": result.available,
        "cheapest_price_hr": result.cheapest_price_hr,
        "regions": result.regions,
        "reason": result.reason,
        "checked_at": result.checked_at,
        "provider": getattr(provider, "name", "unknown"),
    }


# Flags the platform sets on the `vllm serve` command itself — a user passing
# them in their args makes vLLM see a duplicate argument and refuse to start, so
# we reject them at create time rather than letting the worker fail to provision.
_VLLM_RESERVED_SINGLE = {"--model", "--served-model-name", "--port"}
_VLLM_RESERVED_MULTI = _VLLM_RESERVED_SINGLE | {"--tensor-parallel-size", "-tp", "--pipeline-parallel-size", "-pp", "--enable-sleep-mode"}


def _validate_vllm_args(args: Optional[str], *, label: str, reserved: set[str]) -> None:
    """Reject obviously-broken vLLM arg strings *before* an endpoint is created,
    so a bad config surfaces as a clear create-time error instead of a worker
    that silently fails to launch. Catches the common copy-paste mistakes:
    a stray shell line-continuation backslash, unbalanced quotes, and flags the
    platform already sets itself. Raises HTTPException(400) on the first problem."""
    s = (args or "").strip()
    if not s:
        return
    # A bare `\` token is almost always a multi-line shell continuation pasted
    # onto one line (e.g. `--mm-encoder-tp-mode data \ --mm-processor-cache-type shm`).
    if re.search(r"(^|\s)\\(\s|$)", s):
        raise HTTPException(
            status_code=400,
            detail=f"{label}: stray '\\' in vLLM args — looks like a pasted shell line-continuation. Put all args on one line.",
        )
    try:
        tokens = shlex.split(s)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"{label}: can't parse vLLM args ({e}). Check for unbalanced quotes.",
        )
    for tok in tokens:
        flag = tok.split("=", 1)[0]
        if flag in reserved:
            raise HTTPException(
                status_code=400,
                detail=f"{label}: remove '{flag}' — the platform sets it automatically.",
            )


def _normalize_member_gpu_indices(m, *, vd_ids, usable_gpus: int, label: str):
    """Validate + normalize an optional per-model GPU pin (`gpu_indices`). Returns
    the cleaned list of physical ids, or None when unset → the packer auto-assigns
    a tp-wide slot. `vd_ids` is the endpoint's visible_devices id list (or falsy
    when it spans 0..usable_gpus-1). Raises ValueError(msg) on a bad pin."""
    idxs = [int(x) for x in (m.gpu_indices or [])]
    if not idxs:
        return None
    if any(g < 0 for g in idxs):
        raise ValueError(f"{label}: gpu_indices must be >= 0")
    if len(set(idxs)) != len(idxs):
        raise ValueError(f"{label}: gpu_indices has duplicate ids {idxs}")
    width = int(m.tp) * max(1, int(getattr(m, "pp", 1) or 1))
    if len(idxs) != width:
        raise ValueError(
            f"{label}: gpu_indices {idxs} must have exactly tp*pp={width} id(s) "
            f"(tp={m.tp}, pp={getattr(m, 'pp', 1)})"
        )
    if vd_ids:
        bad = [g for g in idxs if g not in set(vd_ids)]
        if bad:
            raise ValueError(f"{label}: gpu_indices {bad} not in this endpoint's GPUs {sorted(set(vd_ids))}")
    elif usable_gpus and any(g >= usable_gpus for g in idxs):
        raise ValueError(f"{label}: gpu_indices {idxs} out of range — pick from 0..{usable_gpus - 1}")
    return idxs


@app.post("/apps", response_model=CreateAppResponse)
async def create_app(
    req: CreateAppRequest,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    if req.gpu_count < 1 or req.gpu_count > 8:
        raise HTTPException(status_code=400, detail="gpu_count must be 1..8")
    if req.cloud_type is not None and req.cloud_type.upper() not in ("COMMUNITY", "SECURE"):
        raise HTTPException(status_code=400, detail="cloud_type must be COMMUNITY or SECURE")
    cloud_type_norm = req.cloud_type.upper() if req.cloud_type else None
    if req.container_disk_gb is not None and (req.container_disk_gb < 1 or req.container_disk_gb > 2000):
        raise HTTPException(status_code=400, detail="container_disk_gb must be 1..2000")
    if req.volume_gb is not None and (req.volume_gb < 0 or req.volume_gb > 4000):
        raise HTTPException(status_code=400, detail="volume_gb must be 0..4000")

    mode = (req.mode or "single").lower()
    if mode not in ("single", "multi", "proxy"):
        raise HTTPException(status_code=400, detail="mode must be 'single', 'multi' or 'proxy'")

    # Validate the chosen provider row. VM is now allowed for serverless;
    # multi-model REQUIRES a probed VM provider (a fixed multi-GPU node).
    prov = None
    if req.provider_id:
        from .db import Provider
        prov = await session.get(Provider, req.provider_id)
        if prov is None:
            raise HTTPException(status_code=400, detail="unknown provider_id")
        if prov.kind not in ("runpod", "pi", "vm"):
            raise HTTPException(
                status_code=400,
                detail=f"provider {req.provider_id} is kind={prov.kind}, serverless requires runpod, pi or vm",
            )
    is_vm = prov is not None and prov.kind == "vm"

    # VM-only GPU pin (e.g. "0,1,2,3"). Validated against the VM's probed GPU
    # count. The number of ids becomes the endpoint's effective gpu_count.
    visible_devices_norm: Optional[str] = None
    vd_ids: Optional[list[int]] = None
    if is_vm and (req.visible_devices or "").strip():
        try:
            parsed = [int(x.strip()) for x in req.visible_devices.split(",") if x.strip() != ""]
        except ValueError:
            raise HTTPException(status_code=400, detail="visible_devices must be comma-separated GPU ids, e.g. 0,1,2,3")
        if parsed:
            if len(set(parsed)) != len(parsed):
                raise HTTPException(status_code=400, detail="visible_devices has duplicate GPU ids")
            vm_total = int((prov.config or {}).get("gpu_count") or 0)
            if any(i < 0 for i in parsed):
                raise HTTPException(status_code=400, detail="visible_devices ids must be >= 0")
            if vm_total and any(i >= vm_total for i in parsed):
                raise HTTPException(status_code=400, detail=f"visible_devices ids must be within 0..{vm_total - 1}")
            vd_ids = parsed
            visible_devices_norm = ",".join(str(i) for i in parsed)

    # Per-mode normalization of the fields we persist.
    gpu = req.gpu
    gpu_count = req.gpu_count
    model = req.model
    models_json: Optional[list] = None
    sleep_level = 1
    autoscaler_dict = req.autoscaler.model_dump()

    if mode in ("multi", "proxy"):
        members = list(req.models or [])
        # proxy = a single-model VM endpoint the gateway proxies to directly (no
        # queue, no sleep). It's modeled as a 1-member fleet so it reuses all the
        # multi-model packing/validation + worker machinery below. Accept either a
        # 1-element `models` list or a bare `model`; normalize to exactly one member.
        if mode == "proxy":
            if not is_vm:
                raise HTTPException(status_code=400, detail="mode='proxy' is for VM providers only")
            if not members and (req.model or "").strip():
                members = [MultiModelMember(
                    model=req.model.strip(), tp=1, pp=1,
                    extra_args=(req.vllm_args or "").strip(),
                )]
            if len(members) != 1:
                raise HTTPException(status_code=400, detail="mode='proxy' requires exactly one model")
        if not members:
            raise HTTPException(status_code=400, detail="multi-model mode requires at least one model in 'models'")
        # Usable GPU universe to pack the fleet onto. Each model needs tp*pp GPUs
        # (must be <= usable); build_multi_model_config packs each onto a (tp*pp)-
        # wide slot and handles the width NOT dividing the total (e.g. width=4 on 6
        # GPUs → [0,1,2,3], leaving [4,5]). So we only require tp*pp <= usable.
        #   VM    → the provider's probed gpu_count (optionally narrowed by a
        #           visible_devices pin); always-on (you own the box).
        #   cloud → the requested gpu_count (the RunPod/PI pod's GPU count);
        #           honours idle-timeout auto-delete of the whole pod.
        if is_vm:
            total_gpus = int((prov.config or {}).get("gpu_count") or 0)
            if total_gpus <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="VM provider has no probed GPUs — run Test on the provider first",
                )
            usable_gpus = len(vd_ids) if vd_ids else total_gpus
        else:
            usable_gpus = int(gpu_count or 1)
            vd_ids = None  # cloud pods have no GPU pin
        seen: set[str] = set()
        for m in members:
            if m.tp < 1:
                raise HTTPException(status_code=400, detail=f"model {m.model}: tp must be >= 1")
            if getattr(m, "pp", 1) < 1:
                raise HTTPException(status_code=400, detail=f"model {m.model}: pp must be >= 1")
            width = m.tp * max(1, int(getattr(m, "pp", 1) or 1))
            if width > usable_gpus:
                raise HTTPException(
                    status_code=400,
                    detail=f"model {m.model}: tp×pp={width} (tp={m.tp}, pp={getattr(m, 'pp', 1)}) exceeds the {usable_gpus} selected GPUs",
                )
            if m.model in seen:
                raise HTTPException(status_code=400, detail=f"duplicate model in members: {m.model}")
            seen.add(m.model)
            _validate_vllm_args(m.extra_args, label=f"model {m.model}", reserved=_VLLM_RESERVED_MULTI)
            try:
                m.gpu_indices = _normalize_member_gpu_indices(
                    m, vd_ids=vd_ids, usable_gpus=usable_gpus, label=f"model {m.model}")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        if req.sleep_level not in (1, 2):
            raise HTTPException(status_code=400, detail="sleep_level must be 1 or 2")
        model = ""
        models_json = [m.model_dump() for m in members]
        sleep_level = req.sleep_level
        # One fleet box/pod with a deeper queue (it time-shares GPUs via sleep/wake).
        autoscaler_dict["max_containers"] = 1
        if int(autoscaler_dict.get("tasks_per_container", 30) or 30) < 64:
            autoscaler_dict["tasks_per_container"] = 64
        if is_vm:
            # Always-on VM fleet — no scale-to-zero (you own the box).
            gpu = "vm"
            gpu_count = usable_gpus
            autoscaler_dict["idle_timeout_s"] = 0
        else:
            # Cloud multi-model fleet (RunPod/PI): keep the requested GPU type +
            # count and HONOUR idle_timeout_s — 0 = always-on, N>0 = delete the
            # pod after N idle seconds (re-provisioned on the next request).
            gpu = req.gpu
            gpu_count = usable_gpus
    else:
        # VM providers serve multi-model fleets only — a 1-model fleet covers the
        # single-model case and adds sleep/wake, and single-model VM serving was
        # never wired up. Single-model mode is for cloud (RunPod / PI) scale-to-zero.
        if is_vm:
            raise HTTPException(
                status_code=400,
                detail="VM providers serve multi-model fleets — use mode='multi' with at least one model.",
            )
        if not (req.model or "").strip():
            raise HTTPException(status_code=400, detail="model is required for single-model endpoints")
        _validate_vllm_args(req.vllm_args, label="vLLM args", reserved=_VLLM_RESERVED_SINGLE)

    record = App(
        app_id=req.name,
        owner_id=user.id,
        name=req.name,
        model=model,
        gpu=gpu,
        gpu_count=gpu_count,
        enable_metrics=req.enable_metrics,
        autoscaler=autoscaler_dict,
        cpu=req.cpu,
        memory=req.memory,
        request_timeout_s=req.request_timeout_s,
        vllm_args=(req.vllm_args or "").strip(),
        cloud_type=cloud_type_norm,
        container_disk_gb=req.container_disk_gb,
        volume_gb=req.volume_gb,
        provider_id=req.provider_id,
        mode=mode,
        models=models_json,
        sleep_level=sleep_level,
        env_vars={k: str(v) for k, v in (req.env_vars or {}).items()} or None,
        visible_devices=visible_devices_norm,
        # venv_path + vllm_version are honoured for a VM venv AND a RunPod multi-model
        # fleet (installed into the pod venv — a volume path survives a re-provision);
        # ignored for a single cloud pod (its image brings vLLM).
        venv_path=((req.venv_path or "").strip() or None) if (is_vm or mode == "multi") else None,
        vllm_version=((req.vllm_version or "").strip() or None) if (is_vm or mode == "multi") else None,
        vllm_install_args=((req.vllm_install_args or "").strip() or None) if (is_vm or mode == "multi") else None,
        # Pre-script runs on the worker (VM venv or RunPod multi-model pod), not a
        # single cloud pod.
        pre_script=((req.pre_script or "").strip() or None) if (is_vm or mode == "multi") else None,
        created_at=datetime.now(timezone.utc),
    )
    session.add(record)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=409, detail={"error": "app name already taken"})
    logger.info("created app %s by user=%s (%s on %s)", req.name, user.username, req.model, req.gpu)

    # Pre-flight: when always-on (idle_timeout_s == 0) the autoscaler will
    # immediately try to provision. Do that one attempt synchronously so we
    # can fail the create cleanly if the provider rejects the spec
    # (out of stock, GPU not on this cloud tier, etc.) — instead of leaving
    # the user with a phantom endpoint they have to delete.
    rdb = request.app.state.redis
    effective_idle = int(autoscaler_dict.get("idle_timeout_s", 300) or 0)
    provider = None
    if effective_idle == 0:
        from .provider import resolve_app_provider
        try:
            provider = await resolve_app_provider(
                session, record, redis=rdb,
                fallback=getattr(request.app.state, "provider", None),
                cache=request.app.state.provider_cache,
            )
        except Exception as e:
            # Bad provider config (unreachable VM, missing template env, …) — fail
            # the create cleanly rather than leaving a phantom endpoint.
            try:
                await session.delete(record)
                await session.commit()
            except Exception:
                await session.rollback()
            raise HTTPException(
                status_code=400,
                detail={"error": "provider unavailable", "reason": (str(e) or repr(e))[:500]},
            )
    if effective_idle == 0 and provider is not None:
        # Block the autoscaler tick from racing this attempt.
        await rdb.set(
            f"app:{req.name}:provision_cooldown_until",
            str(time.time() + 30),
            ex=60,
        )
        token = secrets.token_urlsafe(24)
        env: dict[str, str] = {"REGISTRATION_TOKEN": token}
        extra = (req.vllm_args or "").strip()
        if extra:
            env["VLLM_EXTRA_ARGS"] = extra
        from .autoscaler import (
            REGISTRATION_TOKEN_TTL_S, emit_worker_event, build_metrics_env, build_multi_model_config,
        )
        if req.enable_metrics:
            env.update(build_metrics_env(req.name, provider.name))
        if mode in ("multi", "proxy"):
            # proxy = 1-member fleet served via direct forward-tunnel proxy (no
            # queue); WORKER_MODE=proxy so the worker skips the queue consumer.
            env["WORKER_MODE"] = mode
            env["MULTI_MODEL_CONFIG"] = json.dumps(build_multi_model_config(record))
            env["SLEEP_LEVEL"] = str(sleep_level)
            env["TOTAL_GPUS"] = str(gpu_count)
        elif visible_devices_norm:
            # Single-model VM GPU pin (multi sets it per model from gpu_indices).
            env["CUDA_VISIBLE_DEVICES"] = visible_devices_norm
        # Global env/secrets + this app's env vars (app overrides global), with
        # any `secret://KEY` references (e.g. a chosen HF token) resolved.
        from .global_env_api import load_global_env, resolve_env_refs
        _g = await load_global_env(session)
        _worker_env = resolve_env_refs({**_g, **(record.env_vars or {})}, _g)
        if _worker_env:
            env["WORKER_ENV_JSON"] = json.dumps(_worker_env)
        try:
            result = await provider.provision(
                app_id=req.name,
                model=record.model,
                gpu=record.gpu,
                env=env,
                gpu_count=record.gpu_count,
                cloud_type=cloud_type_norm,
                container_disk_gb=req.container_disk_gb,
                volume_gb=req.volume_gb,
            )
            machine_id = result.machine_id
        except Exception as e:
            error_msg = (str(e) or repr(e))[:500]
            logger.warning(
                "create_app pre-flight provision failed for %s gpu=%sx%d: %s",
                req.name, record.gpu, record.gpu_count, error_msg,
            )
            # Roll back: delete the app so the user can retry with a different
            # combo without bumping into the unique-name 409.
            try:
                await session.delete(record)
                await session.commit()
            except Exception:
                await session.rollback()
                logger.exception("create_app rollback failed for %s", req.name)
            await rdb.delete(f"app:{req.name}:provision_cooldown_until")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "GPU not available right now",
                    "reason": error_msg,
                    "gpu": record.gpu,
                    "gpu_count": record.gpu_count,
                },
            )
        # Success — register the worker so the autoscaler sees current=1.
        await rdb.set(f"register_token:{machine_id}", token, ex=REGISTRATION_TOKEN_TTL_S)
        await rdb.sadd(f"worker_index:{req.name}", machine_id)
        await rdb.set(
            f"worker:{machine_id}",
            json.dumps({
                "machine_id": machine_id,
                "app_id": req.name,
                "status": "provisioning",
                "last_seen": time.time(),
            }),
            ex=REGISTRATION_TOKEN_TTL_S,
        )
        if result.cost_per_hr is not None:
            await rdb.set(
                f"worker_cost:{machine_id}",
                str(result.cost_per_hr),
                ex=REGISTRATION_TOKEN_TTL_S,
            )
        await emit_worker_event(
            rdb, machine_id, req.name, "info",
            f"provisioned on {provider.name} (gpu={record.gpu}x{record.gpu_count}, pre-flight at create"
            + (f", ${result.cost_per_hr:.4f}/hr)" if result.cost_per_hr is not None else ")"),
        )
        await rdb.delete(f"app:{req.name}:provision_cooldown_until")
        logger.info("create_app pre-flight provisioned %s for %s", machine_id, req.name)

    await audit_module.record(
        user, "inference.create", "app", req.name, req.name,
        details={
            "mode": mode,
            "model": record.model or None,
            "models": [m.get("model") for m in (models_json or [])] or None,
            "gpu": record.gpu,
            "gpu_count": record.gpu_count,
            "visible_devices": record.visible_devices,
            "member_gpu_indices": {
                str(m.get("model")): m.get("gpu_indices")
                for m in (models_json or [])
                if isinstance(m, dict) and m.get("gpu_indices")
            } or None,
        },
    )
    return CreateAppResponse(app_id=req.name, url=f"/run/{req.name}")


@app.get("/apps", response_model=list[AppRecord])
async def list_apps(
    scope: str = "mine",
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    # Admins default to their own apps; pass ?scope=all to see everyone's.
    # Non-admins are always scoped to own regardless of the param.
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    show_all = user.is_admin and scope == "all"
    stmt = select(App).options(selectinload(App.owner))
    if show_all:
        pass  # admin + ?scope=all → every endpoint
    elif user.is_admin:
        stmt = stmt.where(App.owner_id == user.id)  # admin "mine" stays strictly own
    else:
        # Non-admins see their own endpoints plus any public ones (read-only),
        # mirroring the benchmark list. Edit/delete is still owner-gated server-side.
        stmt = stmt.where((App.owner_id == user.id) | (App.is_public.is_(True)))
    result = await session.execute(stmt)
    apps = result.scalars().all()
    return [_to_app_record(a, redacted=not _viewer_is_owner(a, user)) for a in apps]


async def _load_owned_app(
    session: AsyncSession, app_id: str, user: User, *, allow_public: bool = False
) -> App:
    """Load an app, enforcing access. Default (allow_public=False) is owner-or-admin
    only — used by every MUTATING route, so writes stay strictly owner-scoped. Pass
    allow_public=True on READ-ONLY routes that should also serve a *public* endpoint
    to non-owners (record/status/workers/worker-events/stress-runs). It never relaxes
    writes, the inference data plane, logs, or request history."""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    result = await session.execute(
        select(App).where(App.app_id == app_id).options(selectinload(App.owner))
    )
    app = result.scalar_one_or_none()
    if app is None:
        raise HTTPException(status_code=404, detail="no such app")
    if app.owner_id != user.id and not user.is_admin:
        if allow_public and bool(getattr(app, "is_public", False)):
            return app
        raise HTTPException(status_code=403, detail="not your app")
    return app


def _viewer_is_owner(app: App, user: User) -> bool:
    """True when `user` may see the full (un-redacted) record + manage the app."""
    return bool(user.is_admin or app.owner_id == user.id)


async def _provider_for_app(session: AsyncSession, app_id: str, user: User, app_state) -> Optional["Provider"]:
    """Resolve the Provider an app's workers run on (vm/runpod/pi per-row, else
    the global fallback). Used by teardown paths so e.g. a VM fleet is actually
    SSH-killed. Falls back to the global provider if resolution fails."""
    app = await _load_owned_app(session, app_id, user)
    from .provider import resolve_app_provider
    try:
        return await resolve_app_provider(
            session, app, redis=app_state.redis,
            fallback=getattr(app_state, "provider", None),
            cache=getattr(app_state, "provider_cache", {}),
        )
    except Exception:
        logger.exception("provider resolve failed for app=%s during teardown", app_id)
        return getattr(app_state, "provider", None)


@app.get("/apps/{app_id}", response_model=AppRecord)
async def get_app_endpoint(
    app_id: str,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    app = await _load_owned_app(session, app_id, user, allow_public=True)
    return _to_app_record(app, redacted=not _viewer_is_owner(app, user))


class SetAppVisibilityRequest(BaseModel):
    is_public: bool


@app.post("/apps/{app_id}/visibility", response_model=AppRecord)
async def set_app_visibility(
    app_id: str,
    body: SetAppVisibilityRequest,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Make an endpoint public (read-only visible to every logged-in user) or
    private again. Owner (or admin) only — goes through the strict loader, so a
    non-owner can never flip someone else's endpoint."""
    app = await _load_owned_app(session, app_id, user)  # strict: owner/admin only
    app.is_public = bool(body.is_public)
    await session.commit()
    await audit_module.record(
        user, "inference.visibility", "app", app_id, app.name,
        details={"is_public": app.is_public},
    )
    return _to_app_record(app)


class AppProxyLink(BaseModel):
    id: str
    name: str
    public: bool
    serving_path: str          # /proxy/{name}/v1
    models: list[str] = []     # alias(es) on this proxy that route to this endpoint


@app.get("/apps/{app_id}/proxies", response_model=list[AppProxyLink])
async def list_app_proxies(
    app_id: str,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """LLM API proxies that front this endpoint, matched by upstream URL (a
    backend pointed at `/{app_id}/v1`) or by served model name. Read-only and
    secret-stripped: only the proxy name, its stable serving path, and the model
    aliases are returned — never upstream base_urls or API keys. Non-admins see
    only PUBLIC proxies; admins see every matching proxy."""
    app = await _load_owned_app(session, app_id, user, allow_public=True)
    from .proxy_api import ProxyEndpoint
    from sqlalchemy import select
    # Models this endpoint serves (single: [model]; multi/proxy fleet: members).
    served: set[str] = set()
    if app.model:
        served.add(app.model)
    for m in (getattr(app, "models", None) or []):
        mm = m.get("model") if isinstance(m, dict) else None
        if mm:
            served.add(mm)
    rows = (await session.execute(select(ProxyEndpoint))).scalars().all()
    out: list[AppProxyLink] = []
    for ep in rows:
        # Non-admins only ever see public proxies (proxies are admin-managed).
        if not (user.is_admin or bool(getattr(ep, "public", False))):
            continue
        cfg = ep.config or {}
        matched: set[str] = set()
        for u in cfg.get("upstreams", []):
            if not u.get("enabled", True):
                continue
            base = u.get("base_url") or ""
            models_map = u.get("models") or {}  # alias -> real upstream model
            if f"/{app_id}/" in base:
                matched.update(models_map.keys())
            else:
                matched.update(a for a, real in models_map.items() if real in served)
        if matched:
            out.append(AppProxyLink(
                id=ep.id, name=ep.name, public=bool(getattr(ep, "public", False)),
                serving_path=f"/proxy/{ep.name}/v1", models=sorted(matched),
            ))
    return out


@app.get("/apps/{app_id}/status")
async def get_app_status(
    app_id: str,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Operational state for the overview tab: live worker count, queue depth,
    and the most recent provision error (if any). Empty error means the
    autoscaler is either idle or scaling cleanly."""
    app = await _load_owned_app(session, app_id, user, allow_public=True)
    rdb = request.app.state.redis
    paused = bool(await rdb.get(f"app:{app_id}:paused"))
    queue_len = await rdb.llen(f"queue:{app_id}")
    workers = await rdb.smembers(f"worker_index:{app_id}")
    live_workers = 0
    models_state: list[dict] = []
    # proxy is a 1-member fleet — it reports member state the same way, so surface
    # its model in the status (Workers/Visual tabs) like multi.
    is_fleet = (getattr(app, "mode", "single") or "single") in ("multi", "proxy")
    for mid in workers:
        if await rdb.exists(f"worker:{mid}"):
            live_workers += 1
            if is_fleet:
                blob = await rdb.get(f"worker:{mid}:models")
                if blob:
                    try:
                        models_state.extend(json.loads(blob))
                    except (json.JSONDecodeError, TypeError):
                        pass
    # Collapse to one entry per model. With >1 live worker every worker reports
    # the same members (replicas of the fleet config), so the per-worker lists
    # above contain duplicates — left as-is the UI gets two rows with the same
    # key. Keep the "most awake" state across workers so a model that's serving
    # on any worker shows as awake.
    if models_state:
        _STATE_RANK = {
            "awake": 7, "waking": 6, "launching": 5, "queued": 4,
            "draining": 3, "sleeping": 2, "asleep": 1, "dead": 0,
        }
        deduped: dict[str, dict] = {}
        for ms in models_state:
            name = ms.get("model")
            if name is None:
                continue
            cur = deduped.get(name)
            if cur is None or _STATE_RANK.get(ms.get("state"), -1) > _STATE_RANK.get(cur.get("state"), -1):
                deduped[name] = ms
        models_state = list(deduped.values())
    # Attach each model's localhost vLLM port. The gateway assigns these
    # deterministically when it builds the fleet config, so we can recompute
    # them here and surface `localhost:<port>` per model in the Workers tab —
    # no worker change needed.
    if is_fleet and models_state:
        try:
            from .autoscaler import build_multi_model_config
            cfg = build_multi_model_config(app)
            port_by_model = {m["model"]: m.get("port") for m in cfg.get("models", [])}
            for ms in models_state:
                p = port_by_model.get(ms.get("model"))
                if p is not None:
                    ms["port"] = p
                    ms["base_url"] = f"localhost:{p}"
        except Exception:
            pass
    err = await rdb.get(f"app:{app_id}:last_provision_error")
    err_at_blob = await rdb.get(f"app:{app_id}:last_provision_error_at")
    cooldown_blob = await rdb.get(f"app:{app_id}:provision_cooldown_until")
    cooldown_remaining = 0
    if cooldown_blob:
        try:
            remaining = float(cooldown_blob) - time.time()
            cooldown_remaining = int(max(0, remaining))
        except (TypeError, ValueError):
            pass
    err_at: Optional[float] = None
    if err_at_blob:
        try:
            err_at = float(err_at_blob)
        except (TypeError, ValueError):
            pass
    return {
        "app_id": app_id,
        "queue_len": queue_len,
        "workers": live_workers,
        "last_provision_error": err,
        "last_provision_error_at": err_at,
        "provision_cooldown_remaining_s": cooldown_remaining,
        "mode": getattr(app, "mode", "single") or "single",
        "models": models_state,
        "sleep_level": int(getattr(app, "sleep_level", 1) or 1),
        "paused": paused,
    }


@app.get("/apps/{app_id}/models/logs")
async def get_app_model_logs(
    app_id: str,
    request: Request,
    model: str,
    tail: int = 400,
    log_session: Optional[str] = None,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Per-model vLLM stdout/stderr for a multi-model VM endpoint. The worker
    ships one log stream per member tagged by served_name. Without `log_session`
    we return the LIVE tail (`worker_logs:{mid}:{slug}`). With `log_session` (a
    per-launch timestamp from /models/log-sessions) we return that historical
    launch's log (`wlog:{app}:{slug}:{session}`), which survives a re-provision —
    so a past crash stays openable."""
    if tail < 1 or tail > WORKER_LOGS_CAP:
        raise HTTPException(status_code=400, detail=f"tail must be 1..{WORKER_LOGS_CAP}")
    app = await _load_owned_app(session, app_id, user)
    rdb = request.app.state.redis
    slug = _logs_slug(model)
    lines: list[str] = []
    if log_session:
        raw = await rdb.lrange(f"wlog:{app_id}:{slug}:{log_session}", 0, tail - 1)
        lines = list(reversed(raw))
    else:
        workers = await rdb.smembers(f"worker_index:{app_id}")
        # Take the first worker that still has buffered logs for this model — the
        # crash log outlives the heartbeat TTL (~1h), so a dead model's reason
        # remains visible after its worker stops beating.
        for mid in workers:
            raw = await rdb.lrange(f"worker_logs:{mid}:{slug}", 0, tail - 1)
            if raw:
                lines = list(reversed(raw))  # stored newest-first → chronological
                break
    return {"app_id": app.app_id, "model": model, "session": log_session, "lines": lines, "count": len(lines)}


@app.get("/apps/{app_id}/models/log-sessions")
async def get_app_model_log_sessions(
    app_id: str,
    request: Request,
    model: str,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Historical log sessions (one per vLLM launch) for a member, newest first,
    for the UI's session picker. Each is `{session, started_at, lines}` where
    `session` is the launch timestamp ("YYYYMMDD-HHMMSS") to pass back as
    `?log_session=`."""
    app = await _load_owned_app(session, app_id, user)
    rdb = request.app.state.redis
    slug = _logs_slug(model)
    raw = await rdb.zrevrange(f"wlog_sessions:{app_id}:{slug}", 0, 199)
    out = []
    for s in raw:
        # "YYYYMMDD-HHMMSS" → ISO for display; best-effort.
        iso = s
        try:
            d = s.split("-")
            iso = f"{d[0][:4]}-{d[0][4:6]}-{d[0][6:8]}T{d[1][:2]}:{d[1][2:4]}:{d[1][4:6]}"
        except (IndexError, ValueError):
            pass
        n = await rdb.llen(f"wlog:{app_id}:{slug}:{s}")
        crash = await rdb.get(f"wlog_crash:{app_id}:{slug}:{s}")
        out.append({"session": s, "started_at": iso, "lines": n, "crash": crash})
    return {"app_id": app.app_id, "model": model, "sessions": out}


@app.post("/apps/{app_id}/model-action")
async def app_model_action(
    app_id: str,
    req: ModelActionRequest,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Queue an action for a multi-model VM endpoint; the worker applies it on
    its next heartbeat (≤5s) via the scheduler:
      kill      → stop the engine + all its tp workers, leave it dead
      restart   → kill + relaunch (then asleep, woken on demand)
      sleep     → drain + /sleep one awake model now (free its GPUs)
      sleep_all → sleep every awake model (free all GPUs)
    `sleep_all` ignores `model`; the others target one member."""
    app = await _load_owned_app(session, app_id, user)
    app_mode = (getattr(app, "mode", "single") or "single")
    if app_mode not in ("multi", "proxy"):
        raise HTTPException(status_code=400, detail="model actions apply only to multi-model / proxy endpoints")
    action = (req.action or "").strip().lower()
    if action not in ("kill", "restart", "sleep", "sleep_all"):
        raise HTTPException(status_code=400, detail="action must be one of: kill, restart, sleep, sleep_all")
    # proxy serves one always-on model directly (no queue/sleep) — sleeping it would
    # brick the endpoint (nothing wakes it). Only kill/restart make sense.
    if app_mode == "proxy" and action in ("sleep", "sleep_all"):
        raise HTTPException(status_code=400, detail="proxy endpoints don't support sleep — only kill/restart")
    rdb = request.app.state.redis
    if action == "sleep_all":
        await rdb.rpush(f"app:{app_id}:model_cmds", json.dumps({"model": "*", "action": "sleep_all"}))
        await rdb.expire(f"app:{app_id}:model_cmds", 300)
        await audit_module.record(
            user, "inference.model_action", "app", app_id, app_id,
            details={"action": "sleep_all", "model": "*"},
        )
        return {"ok": True, "queued": action}
    members = [m.get("model") for m in (app.models or [])]
    if req.model not in members:
        raise HTTPException(
            status_code=404,
            detail={"error": "unknown model for this endpoint", "model": req.model, "members": members},
        )
    await rdb.rpush(f"app:{app_id}:model_cmds", json.dumps({"model": req.model, "action": action}))
    await rdb.expire(f"app:{app_id}:model_cmds", 300)
    await audit_module.record(
        user, "inference.model_action", "app", app_id, app_id,
        details={"action": action, "model": req.model},
    )
    return {"ok": True, "queued": action, "model": req.model}


@app.patch("/apps/{app_id}/autoscaler", response_model=AppRecord)
async def update_app_autoscaler(
    app_id: str,
    req: UpdateAutoscalerRequest,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    target = await _load_owned_app(session, app_id, user)
    cfg = dict(target.autoscaler or {})
    updates = req.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="no fields to update")
    for k in ("max_containers", "tasks_per_container", "idle_timeout_s"):
        if k in updates:
            v = int(updates[k])
            if v < 0:
                raise HTTPException(status_code=400, detail=f"{k} must be >= 0")
            cfg[k] = v
    if "worker_ttl_s" in updates:
        ttl = int(updates["worker_ttl_s"])
        if ttl < 30 or ttl > 86400:
            raise HTTPException(status_code=400, detail="worker_ttl_s must be 30..86400 seconds")
        cfg["worker_ttl_s"] = ttl
        # Mirror immediately so the heartbeat hot path honours it without waiting
        # for the next autoscaler tick.
        await request.app.state.redis.set(f"app:{app_id}:worker_ttl_s", str(ttl))
    target.autoscaler = cfg
    flag_modified(target, "autoscaler")
    if "vllm_args" in updates:
        new_args = (updates["vllm_args"] or "").strip()
        if len(new_args) > 2048:
            raise HTTPException(status_code=400, detail="vllm_args too long (max 2048 chars)")
        target.vllm_args = new_args
    if "gpu_count" in updates:
        new_count = int(updates["gpu_count"])
        if new_count < 1 or new_count > 8:
            raise HTTPException(status_code=400, detail="gpu_count must be 1..8")
        target.gpu_count = new_count
    await session.commit()
    await session.refresh(target)
    # Reset the idle clock when idle_timeout_s changes — otherwise switching
    # always-on (0) → finite tears down immediately because last_request_ts
    # is already far in the past.
    if "idle_timeout_s" in updates:
        await request.app.state.redis.set(
            f"app:{app_id}:last_request_ts", str(time.time())
        )
    logger.info("autoscaler updated app=%s by user=%s: %s", app_id, user.username, updates)
    await audit_module.record(
        user, "inference.update_autoscaler", "app", app_id, target.name,
        details={"changes": updates},
    )
    return _to_app_record(target)


async def _reprovision_workers(rdb, session, app_id: str, user: User, app_state) -> int:
    """Drain + terminate every worker for this app so the autoscaler respawns it
    with the app's latest config (updated vllm_args, model list, tp, …). Drain
    lets in-flight requests wrap up; terminate() actually frees the pod/engines
    (RunPod doesn't reap exited containers; VM engines are reaped per-PID). Picks
    up orphan machines (in the provider but not in worker_index) too. Returns the
    number of machines terminated."""
    from .autoscaler import emit_worker_event

    tracked = set(await rdb.smembers(f"worker_index:{app_id}"))
    for mid in tracked:
        await rdb.set(f"worker:{mid}:drain", "1", ex=600)
        await emit_worker_event(rdb, mid, app_id, "warning", "drain signal sent (reprovision)",
                                actor=getattr(user, "username", None))

    provider = await _provider_for_app(session, app_id, user, app_state)
    all_machines = set(tracked)
    if provider is not None:
        try:
            orphans = set(await provider.list_machines_for_app(app_id)) - tracked
            if orphans:
                logger.info("reprovision app %s: also terminating %d orphan machines", app_id, len(orphans))
                all_machines |= orphans
        except Exception:
            logger.exception("reprovision app %s: list_machines_for_app failed", app_id)
        for mid in all_machines:
            try:
                await provider.terminate(mid)
                await emit_worker_event(rdb, mid, app_id, "info", "terminated (reprovision)",
                                        actor=getattr(user, "username", None))
            except Exception:
                logger.exception("reprovision app %s: provider.terminate(%s) failed", app_id, mid)
                await emit_worker_event(rdb, mid, app_id, "error", "terminate failed (reprovision)",
                                        actor=getattr(user, "username", None))

    for mid in all_machines:
        await rdb.delete(f"worker:{mid}", f"register_token:{mid}", f"worker:{mid}:models")
    if all_machines:
        await rdb.srem(f"worker_index:{app_id}", *all_machines)
    logger.info("reprovision app %s: drained=%d terminated=%d", app_id, len(tracked), len(all_machines))
    return len(all_machines)


@app.post("/apps/{app_id}/restart")
async def restart_app_workers(
    app_id: str,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Drain + terminate every worker so the autoscaler respawns with the latest
    config (e.g. updated vllm_args). Also resumes a killed/paused fleet."""
    await _load_owned_app(session, app_id, user)
    rdb = request.app.state.redis
    await rdb.delete(f"app:{app_id}:paused")  # resume if it was killed
    n = await _reprovision_workers(rdb, session, app_id, user, request.app.state)
    await audit_module.record(
        user, "inference.restart", "app", app_id, app_id,
        details={"drained_workers": n},
    )
    return {"ok": True, "app_id": app_id, "drained_workers": n}


@app.post("/apps/{app_id}/queue/flush")
async def flush_app_queue(
    app_id: str,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Drop every job still waiting in the queue (not yet picked up by a worker).
    Already-running requests are left alone — we only clear the Redis list
    `queue:{app_id}` and mark the corresponding still-queued rows as cancelled.
    The LRANGE+DEL is transactional so a job enqueued mid-flush isn't silently
    lost (it stays on the fresh list, recorded, for a worker to pick up)."""
    app = await _load_owned_app(session, app_id, user)
    rdb = request.app.state.redis
    key = f"queue:{app_id}"
    async with rdb.pipeline(transaction=True) as pipe:
        pipe.lrange(key, 0, -1)
        pipe.delete(key)
        raw, _ = await pipe.execute()

    request_ids: list[str] = []
    for item in raw or []:
        try:
            rid = json.loads(item).get("request_id")
        except (json.JSONDecodeError, TypeError):
            rid = None
        if rid:
            request_ids.append(rid)

    now = datetime.now(timezone.utc)
    cancelled = 0
    for rid in request_ids:
        row = await session.get(ReqRow, rid)
        if row is not None and row.status in ("pending", "queued"):
            row.status = "cancelled"
            if row.completed_at is None:
                row.completed_at = now
            cancelled += 1
        # Unblock any client polling /result/<id>: surface a terminal status.
        await rdb.set(
            f"result:{rid}",
            json.dumps({"status": "cancelled", "error": "flushed from queue"}),
            ex=3600,
        )

    # Also clear ORPHANED pending/queued rows: ones no longer in the queue and not
    # actually being served. A worker can dequeue a job then die / be terminated
    # before finalizing it (common after idle-delete or a failed cold start) — the
    # row stays stuck "pending" (shown as "in queue") and the redis flush above
    # can't reach it. Cancel such a row only when it CAN'T be in flight: no live
    # worker for the app, or it's older than the request timeout (a real request
    # resolves within that). Genuinely in-flight requests are left alone.
    from datetime import timedelta
    from sqlalchemy import select as _select
    live_worker = False
    degraded = False  # a member reporting a failure reason (e.g. a CUDA-OOM on
    # /wake_up) can't serve — its pending requests are wedged, not in flight, so
    # the "leave recent rows on a live worker alone" guard must not apply.
    for _mid in await rdb.smembers(f"worker_index:{app_id}"):
        if await rdb.exists(f"worker:{_mid}"):
            live_worker = True
            blob = await rdb.get(f"worker:{_mid}:models")
            if blob:
                try:
                    if any((m or {}).get("reason") for m in json.loads(blob)):
                        degraded = True
                except (json.JSONDecodeError, TypeError):
                    pass
    timeout_s = int(getattr(app, "request_timeout_s", 600) or 600)
    stale_before = now - timedelta(seconds=timeout_s)
    flushed_ids = set(request_ids)
    stuck_rows = (await session.execute(
        _select(ReqRow).where(ReqRow.app_id == app_id, ReqRow.status.in_(("pending", "queued")))
    )).scalars().all()
    for row in stuck_rows:
        if row.request_id in flushed_ids:
            continue  # already handled via the redis-queue path above
        if live_worker and not degraded and row.created_at and row.created_at > stale_before:
            continue  # could be genuinely in flight on a healthy live worker — leave it
        row.status = "cancelled"
        if row.completed_at is None:
            row.completed_at = now
        cancelled += 1
        await rdb.set(
            f"result:{row.request_id}",
            json.dumps({"status": "cancelled", "error": "flushed (orphaned / not being served)"}),
            ex=3600,
        )
    await session.commit()

    logger.info("flush queue app=%s by user=%s: dropped %d, cancelled %d rows",
                app_id, user.username, len(request_ids), cancelled)
    await audit_module.record(
        user, "inference.queue_flush", "app", app_id, app_id,
        details={"flushed": len(request_ids), "cancelled": cancelled},
    )
    return {"ok": True, "app_id": app_id, "flushed": len(request_ids), "cancelled": cancelled}


@app.post("/apps/{app_id}/workers/kill")
async def kill_app_workers(
    app_id: str,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Drain + terminate every worker AND pause the autoscaler so it stays down —
    unlike /restart, which respawns. Frees the GPUs until you resume with Restart
    all / Redeploy (/restart) or by editing the model list."""
    await _load_owned_app(session, app_id, user)
    rdb = request.app.state.redis
    await rdb.set(f"app:{app_id}:paused", "1")
    n = await _reprovision_workers(rdb, session, app_id, user, request.app.state)
    logger.info("kill workers app=%s by user=%s: terminated %d, paused", app_id, user.username, n)
    await audit_module.record(
        user, "inference.stop", "app", app_id, app_id,
        details={"killed_workers": n, "paused": True},
    )
    return {"ok": True, "app_id": app_id, "killed_workers": n, "paused": True}


@app.post("/apps/{app_id}/workers/purge")
async def purge_app_workers(
    app_id: str,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Hard reset: drain+terminate tracked/orphan workers, THEN sweep every on-disk
    worker remnant for this endpoint on the box (stale pidfiles/logs/configs +
    orphan vLLM engines left by crash-loop churn — which Redeploy/Kill don't touch)
    and clear redis. Use to clean up after redis blips churned the fleet. Leaves the
    paused state as-is: if the fleet wasn't killed, the autoscaler respawns a clean
    worker; if it was, it stays down until Redeploy."""
    await _load_owned_app(session, app_id, user)
    rdb = request.app.state.redis
    tracked = len(await rdb.smembers(f"worker_index:{app_id}"))
    provider = await _provider_for_app(session, app_id, user, request.app.state)
    purged = 0
    if provider is not None:
        # purge_app does the whole cleanup (kill processes + sweep on-disk state +
        # clear redis) in a SINGLE SSH connection. We deliberately DON'T call
        # _reprovision_workers first — that opens one 20s-timeout SSH per machine,
        # which with crash-loop churn (dozens of stale machines) blows past the
        # client timeout. The hard sweep covers tracked + orphan + stale alike.
        try:
            purged = await provider.purge_app(app_id)
        except Exception:
            logger.exception("purge workers app=%s: provider.purge_app failed", app_id)
            raise HTTPException(status_code=502, detail={"error": "purge failed on the provider — see gateway logs"})
    logger.info("purge workers app=%s by user=%s: tracked=%d purged=%d", app_id, user.username, tracked, purged)
    await audit_module.record(
        user, "inference.purge", "app", app_id, app_id,
        details={"terminated": tracked, "purged": purged},
    )
    return {"ok": True, "app_id": app_id, "terminated": tracked, "purged": purged}


@app.post("/apps/{app_id}/clear-restart")
async def clear_restart_app_workers(
    app_id: str,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """One-shot 'Clear & Restart': force-purge every worker remnant on the box
    (process tree + vLLM engines + on-disk pidfiles/logs/configs + redis state),
    even a zombie we may be *wrongly* presuming alive under the long heartbeat TTL,
    then resume the fleet so the autoscaler brings up one clean worker.

    This is the safety valve that makes a long WORKER_TTL_S acceptable: a stale or
    mis-detected worker is never stranded — the operator force-clears the actual VM
    (best-effort kill; harmless no-op if it was already dead) and gets a fresh one."""
    await _load_owned_app(session, app_id, user)
    rdb = request.app.state.redis
    tracked = len(await rdb.smembers(f"worker_index:{app_id}"))
    provider = await _provider_for_app(session, app_id, user, request.app.state)
    purged = 0
    if provider is not None:
        try:
            purged = await provider.purge_app(app_id)
        except Exception:
            logger.exception("clear-restart app=%s: provider.purge_app failed", app_id)
            raise HTTPException(status_code=502, detail={"error": "clear failed on the provider — see gateway logs"})
    # Resume AFTER the purge so the autoscaler respawns a clean worker (and so a
    # previously-killed fleet comes back too).
    await rdb.delete(f"app:{app_id}:paused")
    logger.info("clear-restart app=%s by user=%s: tracked=%d purged=%d", app_id, user.username, tracked, purged)
    await audit_module.record(
        user, "inference.clear_restart", "app", app_id, app_id,
        details={"terminated": tracked, "purged": purged},
    )
    return {"ok": True, "app_id": app_id, "terminated": tracked, "purged": purged}


@app.post("/apps/{app_id}/workers/{machine_id}/terminate")
async def terminate_app_worker(
    app_id: str,
    machine_id: str,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Delete ONE worker's container (terminate its pod) WITHOUT pausing the
    autoscaler — unlike /workers/kill, which pauses so the fleet stays down. The
    fleet drops toward zero, freeing the GPU / stopping billing; the next request
    re-provisions it (scale-from-zero). For the Workers-tab per-pod Delete button."""
    await _load_owned_app(session, app_id, user)
    rdb = request.app.state.redis
    from .autoscaler import emit_worker_event
    provider = await _provider_for_app(session, app_id, user, request.app.state)
    if provider is None:
        raise HTTPException(status_code=400, detail={"error": "no provider configured for this endpoint"})
    try:
        await provider.terminate(machine_id)
    except Exception as e:  # noqa: BLE001
        logger.exception("terminate worker app=%s machine=%s failed", app_id, machine_id)
        await emit_worker_event(rdb, machine_id, app_id, "error", "terminate failed (manual delete)")
        raise HTTPException(status_code=502, detail={"error": f"provider terminate failed: {(str(e) or repr(e))[:300]}"})
    await rdb.delete(
        f"worker:{machine_id}", f"worker:{machine_id}:models",
        f"register_token:{machine_id}", f"worker_cost:{machine_id}", f"worker:{machine_id}:drain",
        f"worker:{machine_id}:ready_since", f"worker:{machine_id}:provisioned_at",
    )
    await rdb.srem(f"worker_index:{app_id}", machine_id)
    await emit_worker_event(
        rdb, machine_id, app_id, "info",
        "container deleted (manual) — autoscaler will re-provision on the next request",
    )
    logger.info("manual terminate app=%s machine=%s by user=%s (autoscaler NOT paused)", app_id, machine_id, user.username)
    return {"ok": True, "app_id": app_id, "machine_id": machine_id, "paused": False}


@app.post("/apps/{app_id}/chaos/kill-engine")
async def chaos_kill_engine(
    app_id: str,
    request: Request,
    model: Optional[str] = None,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Chaos-engineering: SIGKILL the vLLM engine process group(s) on a worker pod
    OUT-OF-BAND (the scheduler doesn't know), simulating a real crash — to exercise
    the auto-restart backoff. `model` (served_name) targets one member's port; omitted
    kills every vLLM on the pod. Provider must support SSH exec (RunPod/VM)."""
    await _load_owned_app(session, app_id, user)
    rdb = request.app.state.redis
    workers = sorted(await rdb.smembers(f"worker_index:{app_id}"))
    if not workers:
        raise HTTPException(status_code=400, detail={"error": "no live worker to chaos-kill"})
    mid = workers[0]
    provider = await _provider_for_app(session, app_id, user, request.app.state)
    if provider is None or not hasattr(provider, "exec_on"):
        raise HTTPException(status_code=400, detail={"error": "this provider doesn't support SSH exec"})
    # Match the api_server process; narrow to a member's --port when `model` given.
    pat = "vllm.entrypoints.openai.api_server"
    cmd = (f"PIDS=$(pgrep -f {shlex.quote(pat)} || true); echo \"victims: $PIDS\"; "
           f"for p in $PIDS; do kill -9 -- -$(ps -o pgid= -p $p | tr -d ' ') 2>/dev/null || kill -9 $p; done; "
           f"echo killed")
    try:
        res = await provider.exec_on(mid, cmd)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail={"error": f"chaos exec failed: {(str(e) or repr(e))[:300]}"})
    logger.warning("CHAOS: killed vLLM engine on app=%s machine=%s by user=%s", app_id, mid, user.username)
    return {"ok": True, "app_id": app_id, "machine_id": mid, "model": model, "result": res}


@app.patch("/apps/{app_id}/models", response_model=AppRecord)
async def update_app_models(
    app_id: str,
    req: UpdateModelsRequest,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Edit a multi-model VM fleet in place: add/remove members, change a model's
    `tp`, or tweak its vLLM `extra_args` — then re-provision so the worker reloads
    the new fleet. (Same validation as create.) Single-model endpoints use
    `PATCH /apps/{id}/autoscaler` (vllm_args) instead."""
    target = await _load_owned_app(session, app_id, user)
    target_mode = (getattr(target, "mode", "single") or "single")
    if target_mode not in ("multi", "proxy"):
        raise HTTPException(status_code=400, detail={"error": "only multi-model / proxy endpoints have an editable model list"})
    if not req.models:
        raise HTTPException(status_code=400, detail={"error": "models cannot be empty — delete the endpoint instead"})
    if target_mode == "proxy" and len(req.models) != 1:
        raise HTTPException(status_code=400, detail={"error": "a proxy endpoint serves exactly one model"})
    if req.sleep_level is not None and req.sleep_level not in (1, 2):
        raise HTTPException(status_code=400, detail={"error": "sleep_level must be 1 or 2"})

    # GPU universe: an updated visible_devices pin (else the app's current one).
    vd = (req.visible_devices if req.visible_devices is not None else (target.visible_devices or "")).strip()
    vd_ids: list[int] = []
    if vd:
        try:
            vd_ids = [int(x.strip()) for x in vd.split(",") if x.strip() != ""]
        except ValueError:
            raise HTTPException(status_code=400, detail={"error": "visible_devices must be comma-separated GPU ids, e.g. 6,7"})
        if len(set(vd_ids)) != len(vd_ids):
            raise HTTPException(status_code=400, detail={"error": "visible_devices has duplicate GPU ids"})
    usable_gpus = len(vd_ids) if vd_ids else int(target.gpu_count or 0)

    seen: set[str] = set()
    for m in req.models:
        if m.tp < 1:
            raise HTTPException(status_code=400, detail={"error": f"model {m.model}: tp must be >= 1"})
        if getattr(m, "pp", 1) < 1:
            raise HTTPException(status_code=400, detail={"error": f"model {m.model}: pp must be >= 1"})
        width = m.tp * max(1, int(getattr(m, "pp", 1) or 1))
        if usable_gpus and width > usable_gpus:
            raise HTTPException(status_code=400, detail={"error": f"model {m.model}: tp×pp={width} (tp={m.tp}, pp={getattr(m, 'pp', 1)}) exceeds the {usable_gpus} GPUs"})
        if m.model in seen:
            raise HTTPException(status_code=400, detail={"error": f"duplicate model in members: {m.model}"})
        seen.add(m.model)
        _validate_vllm_args(m.extra_args, label=f"model {m.model}", reserved=_VLLM_RESERVED_MULTI)
        try:
            m.gpu_indices = _normalize_member_gpu_indices(
                m, vd_ids=vd_ids, usable_gpus=usable_gpus, label=f"model {m.model}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": str(e)})

    target.models = [m.model_dump() for m in req.models]
    flag_modified(target, "models")
    if req.visible_devices is not None:
        target.visible_devices = vd or None
        if vd_ids:
            target.gpu_count = len(vd_ids)
    if req.sleep_level is not None:
        target.sleep_level = req.sleep_level
    if req.pre_script is not None:
        target.pre_script = req.pre_script.strip() or None
    if req.vllm_install_args is not None:
        target.vllm_install_args = req.vllm_install_args.strip() or None
    await session.commit()
    await session.refresh(target)

    rdb = request.app.state.redis
    await rdb.delete(f"app:{app_id}:paused")  # editing implies you want it running
    n = await _reprovision_workers(rdb, session, app_id, user, request.app.state)
    logger.info("models updated app=%s by user=%s: %d models, reprovisioned %d worker(s)",
                app_id, user.username, len(req.models), n)
    await audit_module.record(
        user, "inference.update_models", "app", app_id, target.name,
        details={"models": [m.model for m in req.models], "reprovisioned_workers": n},
    )
    return _to_app_record(target)


@app.get("/apps/{app_id}/worker-events")
async def list_worker_events(
    app_id: str,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 2000,
):
    """Durable worker lifecycle timeline for one endpoint — survives the Redis
    ring's 1h TTL, so this is what an on/off calendar/analytics view reads.

    Returns rows **oldest-first** (natural for a left-to-right timeline). `since`
    / `until` accept ISO-8601 or a unix epoch (seconds); both optional. Pair
    `provisioned`/`registered` with `terminated`/`idle_terminated` per
    `machine_id` to reconstruct serving spans."""
    from sqlalchemy import select, and_
    await _load_owned_app(session, app_id, user, allow_public=True)  # owner/admin or public read
    if limit < 1 or limit > 10000:
        raise HTTPException(status_code=400, detail="limit must be 1..10000")

    def _parse_ts(v: str) -> datetime:
        v = v.strip()
        try:
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"bad timestamp: {v!r}")
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    conds = [WorkerEvent.app_id == app_id]
    if since:
        conds.append(WorkerEvent.created_at >= _parse_ts(since))
    if until:
        conds.append(WorkerEvent.created_at <= _parse_ts(until))
    # Take the newest `limit` rows for the window, then flip to chronological —
    # so a tight cap keeps the most recent activity rather than the oldest.
    stmt = (
        select(WorkerEvent)
        .where(and_(*conds))
        .order_by(WorkerEvent.created_at.desc())
        .limit(limit)
    )
    rows = list((await session.execute(stmt)).scalars().all())
    rows.reverse()
    return {
        "app_id": app_id,
        "count": len(rows),
        "events": [
            {
                "id": r.id,
                "machine_id": r.machine_id,
                "event": r.event,
                "level": r.level,
                "message": r.message,
                "actor": r.actor_username,
                "details": r.details,
                "ts": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@app.get("/admin/worker-events")
async def admin_list_worker_events(
    _: User = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
    since: Optional[str] = None,
    until: Optional[str] = None,
    app_id: Optional[str] = None,
    limit: int = 5000,
):
    """Cross-endpoint workload feed for the admin GPU Timeline. Returns the
    durable `worker_events` rows (inference on/off spans) AND benchmark runs
    (started/ended spans) across every endpoint, each resolved to its node +
    GPU ids, so the analytics page can lay out a unified per-node/per-GPU
    occupancy calendar. Admin only. `since`/`until` accept ISO-8601 or unix
    epoch; worker-event rows come back oldest-first."""
    from sqlalchemy import select, and_
    if limit < 1 or limit > 20000:
        raise HTTPException(status_code=400, detail="limit must be 1..20000")

    def _parse_ts(v: str) -> datetime:
        v = v.strip()
        try:
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"bad timestamp: {v!r}")
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    since_dt = _parse_ts(since) if since else None
    until_dt = _parse_ts(until) if until else None

    # NOTE: we deliberately do NOT apply a `created_at >= since` lower bound here.
    # Worker events are lifecycle transitions only (no heartbeat rows), so a worker
    # that registered BEFORE the window and is still serving has its single ON event
    # older than `since`. Filtering it out would drop a currently-occupied GPU from
    # the window — the opposite of the benchmark span-overlap filter below. Instead
    # we fetch up to `until` (newest `limit` rows) and let the client pair events
    # into spans and clip to the window, matching the benchmark semantics.
    conds = []
    if app_id:
        conds.append(WorkerEvent.app_id == app_id)
    if until_dt is not None:
        conds.append(WorkerEvent.created_at <= until_dt)
    stmt = select(WorkerEvent)
    if conds:
        stmt = stmt.where(and_(*conds))
    stmt = stmt.order_by(WorkerEvent.created_at.desc()).limit(limit)
    rows = list((await session.execute(stmt)).scalars().all())
    rows.reverse()

    def _parse_gpu_ids(vd: "str | None") -> list[int]:
        vd = (vd or "").strip()
        if not vd:
            return []
        try:
            return [int(x.strip()) for x in vd.split(",") if x.strip() != ""]
        except ValueError:
            return []

    # Benchmarks are GPU workloads on the same nodes as inference, so the unified
    # GPU Timeline shows them too. A benchmark = one span [started_at, ended_at]
    # (ended_at NULL → still running). Only started runs have a span. Filter to
    # the window: started before `until` AND not ended before `since`.
    from .bench import Benchmark
    bconds = [Benchmark.started_at.isnot(None)]
    if app_id:
        bconds.append(Benchmark.id == app_id)  # allow targeting a single run too
    if until_dt is not None:
        bconds.append(Benchmark.started_at <= until_dt)
    if since_dt is not None:
        bconds.append((Benchmark.ended_at.is_(None)) | (Benchmark.ended_at >= since_dt))
    brows = (
        await session.execute(
            select(Benchmark).where(and_(*bconds)).order_by(Benchmark.started_at.desc()).limit(limit)
        )
    ).scalars().all()

    # Resolve node (the provider the workload is bound to) + GPU ids, by joining
    # App/Benchmark → Provider. Done here at read time so the timeline gets
    # node/GPU columns without a schema change and works for old rows.
    from .db import Provider
    app_ids = {r.app_id for r in rows if r.app_id}
    prov_ids = set()
    arows = []
    if app_ids:
        arows = (await session.execute(select(App).where(App.app_id.in_(app_ids)))).scalars().all()
        prov_ids |= {a.provider_id for a in arows if a.provider_id}
    prov_ids |= {b.provider_id for b in brows if b.provider_id}
    pname: dict[str, str] = {}
    if prov_ids:
        prows = (await session.execute(select(Provider).where(Provider.id.in_(prov_ids)))).scalars().all()
        pname = {p.id: p.name for p in prows}

    apps_meta: dict[str, dict] = {}
    for a in arows:
        gpu_ids = _parse_gpu_ids(getattr(a, "visible_devices", None)) or list(range(int(a.gpu_count or 1)))
        node = (pname.get(a.provider_id) if a.provider_id else None) or (a.gpu or "shared")
        apps_meta[a.app_id] = {"name": a.name, "node": node, "gpu": a.gpu, "gpu_ids": gpu_ids}

    # Owner usernames for benchmark attribution.
    owner_ids = {b.owner_id for b in brows}
    onames: dict[int, str] = {}
    if owner_ids:
        ures = await session.execute(select(User.id, User.username).where(User.id.in_(owner_ids)))
        onames = {uid: uname for uid, uname in ures.all()}

    benchmarks = []
    for b in brows:
        node = (
            (pname.get(b.provider_id) if b.provider_id else None)
            or ("runpod" if b.runpod_pod_id else "shared")
        )
        benchmarks.append({
            "id": b.id,
            "name": b.name,
            "node": node,
            "gpu_ids": _parse_gpu_ids(b.visible_devices),
            "status": b.status,
            "owner": onames.get(b.owner_id),
            "started": b.started_at.isoformat() if b.started_at else None,
            "ended": b.ended_at.isoformat() if b.ended_at else None,
        })

    return {
        "count": len(rows),
        "apps": apps_meta,
        "events": [
            {
                "id": r.id,
                "app_id": r.app_id,
                "machine_id": r.machine_id,
                "event": r.event,
                "level": r.level,
                "message": r.message,
                "actor": r.actor_username,
                "ts": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
        "benchmarks": benchmarks,
    }


# ── Stress-test run history: persisted per endpoint so runs / models can be
# compared and the comparison shared by link. Access mirrors the endpoint
# (owner/admin via _load_owned_app), so a shared link works for anyone who can
# already see the app.
def _to_stress_record(r: StressRun) -> StressRunRecord:
    return StressRunRecord(
        id=r.id, app_id=r.app_id, created_by=r.created_by, model=r.model or "",
        input_len=r.input_len, output_len=r.output_len, num_prompts=r.num_prompts,
        concurrency=r.concurrency, summary=r.summary or {}, created_at=r.created_at,
    )


_STRESS_MAX_RUNS = 200  # per app; oldest pruned beyond this


@app.get("/apps/{app_id}/stress-runs", response_model=list[StressRunRecord])
async def list_stress_runs(
    app_id: str,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Saved stress runs for an endpoint, newest first. Visible to anyone who can
    access the app — that's what makes a shared comparison link resolve."""
    from sqlalchemy import select
    await _load_owned_app(session, app_id, user, allow_public=True)
    rows = (
        await session.execute(
            select(StressRun)
            .where(StressRun.app_id == app_id)
            .order_by(StressRun.created_at.desc())
            .limit(_STRESS_MAX_RUNS)
        )
    ).scalars().all()
    return [_to_stress_record(r) for r in rows]


@app.post("/apps/{app_id}/stress-runs", response_model=StressRunRecord)
async def create_stress_run(
    app_id: str,
    req: StressRunCreate,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    from sqlalchemy import delete as _delete, select
    await _load_owned_app(session, app_id, user)
    for field, val, lo, hi in (
        ("input_len", req.input_len, 1, 1_048_576),
        ("output_len", req.output_len, 1, 1_048_576),
        ("num_prompts", req.num_prompts, 1, 1_000_000),
        ("concurrency", req.concurrency, 1, 100_000),
    ):
        if not (lo <= val <= hi):
            raise HTTPException(status_code=400, detail=f"{field} must be between {lo} and {hi}")
    if len(req.model) > 255:
        raise HTTPException(status_code=400, detail="model name too long")
    row = StressRun(
        id=f"sr-{uuid.uuid4().hex[:12]}",
        app_id=app_id,
        created_by=user.username,
        model=req.model or "",
        input_len=req.input_len,
        output_len=req.output_len,
        num_prompts=req.num_prompts,
        concurrency=req.concurrency,
        summary=req.summary or {},
    )
    session.add(row)
    # Prune the oldest beyond the cap so history can't grow unbounded.
    excess = (
        await session.execute(
            select(StressRun.id)
            .where(StressRun.app_id == app_id)
            .order_by(StressRun.created_at.desc())
            .offset(_STRESS_MAX_RUNS - 1)
        )
    ).scalars().all()
    if excess:
        await session.execute(_delete(StressRun).where(StressRun.id.in_(excess)))
    await session.commit()
    await session.refresh(row)
    logger.info("stress run saved app=%s by user=%s model=%s", app_id, user.username, req.model)
    return _to_stress_record(row)


@app.delete("/apps/{app_id}/stress-runs/{run_id}")
async def delete_stress_run(
    app_id: str,
    run_id: str,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    from sqlalchemy import select
    await _load_owned_app(session, app_id, user)
    row = (
        await session.execute(
            select(StressRun).where(StressRun.id == run_id, StressRun.app_id == app_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="no such stress run")
    await session.delete(row)
    await session.commit()
    return {"ok": True, "id": run_id}


@app.delete("/apps/{app_id}/stress-runs")
async def clear_stress_runs(
    app_id: str,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    from sqlalchemy import delete as _delete, func, select
    await _load_owned_app(session, app_id, user)
    n = (
        await session.execute(
            select(func.count()).select_from(StressRun).where(StressRun.app_id == app_id)
        )
    ).scalar_one()
    await session.execute(_delete(StressRun).where(StressRun.app_id == app_id))
    await session.commit()
    return {"ok": True, "deleted": int(n or 0)}


@app.delete("/apps/{app_id}")
async def delete_app(
    app_id: str,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    rdb = request.app.state.redis
    app = await _load_owned_app(session, app_id, user)
    from .autoscaler import emit_worker_event

    tracked = set(await rdb.smembers(f"worker_index:{app_id}"))
    for mid in tracked:
        await rdb.set(f"worker:{mid}:drain", "1", ex=600)
        await emit_worker_event(rdb, mid, app_id, "warning", "drain signal sent (app deleted)",
                                actor=getattr(user, "username", None))
    logger.info("delete app %s: marked %d tracked workers for drain", app_id, len(tracked))

    provider = await _provider_for_app(session, app_id, user, request.app.state)
    all_machines = set(tracked)
    if provider is not None:
        try:
            orphans = set(await provider.list_machines_for_app(app_id)) - tracked
            if orphans:
                logger.info("delete app %s: also terminating %d orphan workers", app_id, len(orphans))
                all_machines |= orphans
        except Exception:
            logger.exception("delete app %s: list_machines_for_app failed", app_id)
        for mid in all_machines:
            try:
                await provider.terminate(mid)
                await emit_worker_event(rdb, mid, app_id, "info", "terminated (app deleted)",
                                        actor=getattr(user, "username", None))
            except Exception:
                logger.exception("delete app %s: provider.terminate(%s) failed", app_id, mid)
                await emit_worker_event(rdb, mid, app_id, "error", "terminate failed (app delete)",
                                        actor=getattr(user, "username", None))

    for mid in all_machines:
        await rdb.delete(f"worker:{mid}", f"register_token:{mid}", f"worker:{mid}:models")
    await rdb.delete(
        f"queue:{app_id}",
        f"app:{app_id}:last_request_ts",
        f"worker_index:{app_id}",
    )
    # proxy endpoint: drop the forward-tunnel upstream + the per-provider proxy-app
    # set, else ensure_connectivity keeps re-opening a tunnel (and re-publishing the
    # upstream) for a deleted app every autoscaler tick.
    if (getattr(app, "mode", "single") or "single") == "proxy":
        await rdb.delete(f"proxy:{app_id}:upstream", f"proxy:{app_id}:vmport")
        if app.provider_id:
            await rdb.srem(f"vm_proxy_apps:{app.provider_id}", app_id)
    app_name = app.name
    await session.delete(app)
    await session.commit()

    await audit_module.record(
        user, "inference.delete", "app", app_id, app_name,
        details={"drained_workers": len(all_machines)},
    )
    return {"ok": True, "app_id": app_id, "drained_workers": len(all_machines)}


# ----- run / result / stream -----

async def _admit_and_enqueue(
    rdb,
    db_session: AsyncSession,
    app_id: str,
    user: User,
    payload: dict,
    *,
    stream: bool,
    endpoint: str = "/v1/completions",
    target_model: Optional[str] = None,
) -> tuple[str, int]:
    app = await _load_owned_app(db_session, app_id, user)
    cfg = app.autoscaler
    cap = int(cfg["max_containers"]) * int(cfg["tasks_per_container"])
    queue_len = await rdb.llen(f"queue:{app_id}")
    if queue_len >= cap:
        raise HTTPException(
            status_code=429,
            detail={"error": "capacity exceeded", "queue_length": queue_len, "cap": cap, "retry_after_s": 5},
        )
    # Multi-model fleets only serve once every member has finished loading and gone
    # to sleep (the worker's warm-up runs before its request dispatcher starts).
    # Fail fast with a precise reason instead of letting the request sit in the
    # queue until the sync timeout: warming-up, a dead member, or no worker yet.
    if (getattr(app, "mode", "single") or "single") == "multi":
        target = target_model or (payload.get("model") if isinstance(payload, dict) else None)
        workers = await rdb.smembers(f"worker_index:{app_id}")
        live = [m for m in workers if await rdb.exists(f"worker:{m}")]
        if not live:
            # No worker up. A fleet that scales to zero (idle_timeout_s > 0) must
            # WAKE on a request: fall through to enqueue so the autoscaler
            # provisions from zero — the queued job keeps the booting pod alive and
            # is served once it's ready (cancel-on-disconnect cleans up if the
            # client gives up mid cold-start). Always-on fleets (idle=0) with no
            # worker are mid-initial-provision → fail fast so the caller retries
            # instead of holding this request through a multi-minute cold start.
            idle_timeout_s = int((cfg or {}).get("idle_timeout_s", 0) or 0)
            if idle_timeout_s <= 0:
                raise HTTPException(
                    status_code=503,
                    detail={"error": "no worker is up yet — the endpoint is still provisioning. Retry shortly.", "state": "provisioning"},
                )
            logger.info("scale-from-zero: waking idle-scaled multi fleet %s (no live worker) via enqueue", app_id)
        else:
            mid = live[0]
            try:
                wstate = json.loads(await rdb.get(f"worker:{mid}") or "{}")
            except (json.JSONDecodeError, TypeError):
                wstate = {}
            try:
                models_state = json.loads(await rdb.get(f"worker:{mid}:models") or "[]")
            except (json.JSONDecodeError, TypeError):
                models_state = []
            tinfo = next((m for m in models_state if m.get("model") == target), None) if target else None
            if tinfo and tinfo.get("state") == "dead":
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": f"model '{target}' is not running (dead) — restart it from the Workers tab.",
                        "state": "dead", "model": target, "reason": tinfo.get("reason"),
                    },
                )
            if wstate.get("status") and wstate.get("status") != "ready":
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "the model fleet is still warming up — every model must finish loading and go to sleep before any can serve. Retry shortly.",
                        "state": "warming_up",
                        "models": [{"model": m.get("model"), "state": m.get("state")} for m in models_state],
                    },
                )
    request_id = f"req-{uuid.uuid4().hex[:12]}"
    timeout_s = int(app.request_timeout_s)
    job = {
        "request_id": request_id,
        "payload": payload,
        "timeout_s": timeout_s,
        "endpoint": endpoint,
    }
    # Multi-model: the worker routes by the real member-model name. Single-mode
    # leaves this unset (the worker serves one model and ignores the field).
    if target_model:
        job["target_model"] = target_model
    if stream:
        job["stream"] = True
    db_session.add(ReqRow(
        request_id=request_id,
        app_id=app_id,
        owner_id=app.owner_id,
        endpoint=endpoint,
        payload=payload,
        is_stream=stream,
    ))
    await db_session.commit()
    await rdb.lpush(f"queue:{app_id}", json.dumps(job))
    await rdb.set(f"result:{request_id}", json.dumps({"status": "pending"}), ex=3600)
    await rdb.set(f"app:{app_id}:last_request_ts", str(time.time()))
    return request_id, timeout_s


@app.post("/run/{app_id}", response_model=RunResponse)
async def run(
    app_id: str,
    payload: dict,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    rdb = request.app.state.redis
    # Optional control field: which path on the worker's local vLLM to POST to.
    # Defaults to legacy /v1/completions; the playground sends
    # /v1/chat/completions so chat-only params (reasoning_effort,
    # chat_template_kwargs.enable_thinking) actually apply. Popped so it isn't
    # forwarded to vLLM as an unknown body field.
    ep = payload.pop("endpoint", None)
    endpoint = ep if isinstance(ep, str) and ep.startswith("/v1/") else "/v1/completions"
    # Multi-model endpoints route by the member-model name in `payload.model`;
    # single-mode workers serve one model and ignore target_model.
    model_field = payload.get("model")
    target_model = model_field if isinstance(model_field, str) and model_field else None
    request_id, _ = await _admit_and_enqueue(
        rdb, session, app_id, user, payload, stream=False,
        endpoint=endpoint, target_model=target_model,
    )
    logger.info("enqueued %s on %s (user=%s, endpoint=%s)", request_id, app_id, user.username, endpoint)
    return RunResponse(request_id=request_id, poll_url=f"/result/{request_id}")


@app.post("/stream/{app_id}")
async def stream(
    app_id: str,
    payload: dict,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    rdb = request.app.state.redis
    app = await _load_owned_app(session, app_id, user)
    cfg = app.autoscaler
    cap = int(cfg["max_containers"]) * int(cfg["tasks_per_container"])
    queue_len = await rdb.llen(f"queue:{app_id}")
    if queue_len >= cap:
        raise HTTPException(
            status_code=429,
            detail={"error": "capacity exceeded", "queue_length": queue_len, "cap": cap, "retry_after_s": 5},
        )

    request_id = f"req-{uuid.uuid4().hex[:12]}"
    channel = f"stream:{request_id}"

    pubsub = rdb.pubsub()
    await pubsub.subscribe(channel)

    # Same control fields as /run: which vLLM path to POST to, and the
    # member-model name multi-model workers route by.
    ep = payload.pop("endpoint", None)
    endpoint = ep if isinstance(ep, str) and ep.startswith("/v1/") else "/v1/completions"
    model_field = payload.get("model")
    target_model = model_field if isinstance(model_field, str) and model_field else None

    timeout_s = int(app.request_timeout_s)
    job = {
        "request_id": request_id, "payload": payload, "stream": True,
        "timeout_s": timeout_s, "endpoint": endpoint,
    }
    if target_model:
        job["target_model"] = target_model
    job_blob = json.dumps(job)
    await rdb.lpush(f"queue:{app_id}", job_blob)
    await rdb.set(f"result:{request_id}", json.dumps({"status": "pending"}), ex=3600)
    await rdb.set(f"app:{app_id}:last_request_ts", str(time.time()))
    logger.info("enqueued stream %s on %s (timeout=%ss)", request_id, app_id, timeout_s)

    async def gen():
        yield f"event: meta\ndata: {json.dumps({'request_id': request_id})}\n\n"
        finished_normally = False
        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                data = msg["data"]
                yield f"data: {data}\n\n"
                try:
                    parsed = json.loads(data)
                    if parsed.get("done") or parsed.get("error"):
                        finished_normally = True
                        break
                except json.JSONDecodeError:
                    continue
        finally:
            if not finished_normally:
                await _cancel_on_disconnect(rdb, request_id, app_id, job_blob)
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-Id": request_id,
        },
    )


@app.get("/result/{request_id}", response_model=ResultResponse)
async def get_result(
    request_id: str,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Result lookup. Lazily mirrors completed Redis state into the requests
    table so the UI can show the result long after Redis TTL expires."""
    rdb = request.app.state.redis
    blob = await rdb.get(f"result:{request_id}")
    if blob is None:
        # Fall back to Postgres — Redis result key may have expired.
        row = await session.get(ReqRow, request_id)
        if row is None:
            raise HTTPException(status_code=404, detail="not found")
        if row.owner_id != user.id and not user.is_admin:
            raise HTTPException(status_code=403, detail="not your request")
        return ResultResponse(request_id=request_id, status=row.status, output=row.output)

    raw = json.loads(blob)
    status = raw.get("status", "unknown")
    output = raw.get("output")

    row = await session.get(ReqRow, request_id)
    if row is not None:
        if row.owner_id != user.id and not user.is_admin:
            raise HTTPException(status_code=403, detail="not your request")
        # Mirror redis -> postgres so the row reflects current state, surviving Redis TTL.
        if row.status != status or row.output != output:
            if row.status != status:
                metrics.observe_job_outcome(row.app_id, status)
            row.status = status
            row.output = output
            wm = _worker_meta_from_result(raw)
            if wm is not None:
                row.worker_meta = wm
            if status != "pending" and row.completed_at is None:
                from datetime import datetime, timezone
                row.completed_at = datetime.now(timezone.utc)
            await session.commit()

    return ResultResponse(request_id=request_id, status=status, output=output)


class RequestRecord(BaseModel):
    request_id: str
    app_id: str
    endpoint: str
    payload: dict
    status: str
    output: Optional[Any] = None
    is_stream: bool
    created_at: str
    completed_at: Optional[str] = None
    # Username of the caller who submitted the request (resolved from owner_id).
    requested_by: Optional[str] = None


def _to_request_record(r: ReqRow, requested_by: Optional[str] = None) -> RequestRecord:
    return RequestRecord(
        request_id=r.request_id,
        app_id=r.app_id,
        endpoint=r.endpoint,
        payload=r.payload,
        status=r.status,
        output=r.output,
        is_stream=r.is_stream,
        created_at=r.created_at.isoformat() if r.created_at else "",
        completed_at=r.completed_at.isoformat() if r.completed_at else None,
        requested_by=requested_by,
    )


@app.get("/requests/{request_id}", response_model=RequestRecord)
async def get_request(
    request_id: str,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    row = await session.get(ReqRow, request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    if row.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="not your request")
    return _to_request_record(row)


class AppWorkerRow(BaseModel):
    machine_id: str
    status: str                      # provisioning | loading | ready | … (from the heartbeat)
    alive: bool                      # worker:{mid} still present (heartbeated within WORKER_TTL_S)
    last_seen: Optional[float] = None


@app.get("/apps/{app_id}/workers", response_model=list[AppWorkerRow])
async def list_app_workers(
    app_id: str,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Workers the gateway currently tracks for this app, and whether each is still
    alive (heartbeating). The Workers tab cross-references this with the provider's
    pod list: a pod can be `running` on RunPod while its worker never registered /
    stopped heartbeating (the localhost-registration gotcha) — here that shows as a
    machine_id with `alive=false` (or absent entirely)."""
    await _load_owned_app(session, app_id, user, allow_public=True)
    rdb = request.app.state.redis
    mids = await rdb.smembers(f"worker_index:{app_id}")
    out: list[AppWorkerRow] = []
    now = time.time()
    for mid in mids:
        blob = await rdb.get(f"worker:{mid}")
        if not blob:
            out.append(AppWorkerRow(machine_id=mid, status="gone", alive=False))
            continue
        try:
            w = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            w = {}
        last_seen = w.get("last_seen")
        # `alive` = actually HEARTBEATING, not merely "the redis key exists". The
        # autoscaler writes a `provisioning` placeholder at provision time with a
        # longer TTL, so a worker that never phoned home (broken reverse tunnel,
        # failed bootstrap) would otherwise read as alive+provisioning forever. A
        # live worker refreshes worker:{mid} every <WORKER_TTL_S, so a stale
        # last_seen means it isn't heartbeating.
        fresh = isinstance(last_seen, (int, float)) and (now - last_seen) <= WORKER_TTL_S * 2
        out.append(AppWorkerRow(
            machine_id=mid,
            status=str(w.get("status") or "unknown"),
            alive=bool(fresh),
            last_seen=last_seen,
        ))
    return out


@app.get("/apps/{app_id}/requests", response_model=list[RequestRecord])
async def list_app_requests(
    app_id: str,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
    limit: int = 50,
    status_filter: Optional[str] = None,
):
    """Recent requests for an app, newest first. Owner-scoped (admin sees all).
    Reconciles unsettled rows against Redis on the way out so the queue UI shows
    reality: a still-queued/pending row that completed without ever being polled,
    AND a `timeout` row whose client gave up at the 60s unary poll but whose worker
    kept going and produced a result afterwards (the cold-start case — the worker
    isn't cancelled by the client's 504). The Redis result lives ~1h, which bounds
    which rows can still be upgraded."""
    app = await _load_owned_app(session, app_id, user)
    from sqlalchemy import select, desc
    from datetime import datetime, timezone, timedelta
    stmt = select(ReqRow).where(ReqRow.app_id == app_id).order_by(desc(ReqRow.created_at)).limit(min(limit, 200))
    if status_filter:
        stmt = stmt.where(ReqRow.status == status_filter)
    result = await session.execute(stmt)
    rows = list(result.scalars().all())

    rdb = request.app.state.redis
    # A still-pending/queued row older than this, with Redis having forgotten it
    # (result TTL 1h ≥ request_timeout_s), was orphaned and will never resolve.
    orphan_timeout_s = int(getattr(app, "request_timeout_s", 600) or 600)
    orphan_before = datetime.now(timezone.utc) - timedelta(seconds=orphan_timeout_s)
    for r in rows:
        # `timeout` is reconcilable too: the gateway marks a unary request `timeout`
        # when its 60s poll lapses, but the job stays queued and the worker may
        # complete it later — Redis then holds the real terminal result.
        if r.status not in ("queued", "pending", "timeout"):
            continue
        blob = await rdb.get(f"result:{r.request_id}")
        if blob:
            raw = json.loads(blob)
            rstatus = raw.get("status")
            if rstatus and rstatus != "pending" and rstatus != r.status:
                await _mirror_status_to_db(session, r.request_id, rstatus, raw.get("output"),
                                           worker_meta=_worker_meta_from_result(raw))
                r.status = rstatus
                r.output = raw.get("output")
            continue
        # No Redis result key. A still-pending/queued row older than the request
        # timeout was orphaned — a gateway restart killed its poll loop mid-flight,
        # or its queued job was flushed — and won't resolve on its own. Settle it
        # `failed` so the queue UI stops showing a phantom "in queue" job. A genuinely
        # in-flight request (even a long cold start) still has result:{id}=pending in
        # Redis, so it never reaches here. `timeout` rows with no result: leave them.
        if r.status in ("queued", "pending"):
            ca = r.created_at
            if ca is not None and ca.tzinfo is None:
                ca = ca.replace(tzinfo=timezone.utc)
            if ca is None or ca < orphan_before:
                out = {"error": "orphaned — never completed (gateway restart mid-request, or its queued job was flushed / result expired)"}
                await _mirror_status_to_db(session, r.request_id, "failed", out)
                r.status = "failed"
                r.output = out

    # Resolve owner_id → username in one query so each row can show who called.
    owner_ids = {r.owner_id for r in rows}
    umap: dict[int, str] = {}
    if owner_ids:
        from .db import User as _User
        urows = await session.execute(select(_User).where(_User.id.in_(owner_ids)))
        umap = {u.id: u.username for u in urows.scalars().all()}

    return [_to_request_record(r, umap.get(r.owner_id)) for r in rows]


def _worker_meta_from_result(raw: dict) -> Optional[dict]:
    """Node identity from a worker's Redis result blob — `node` (hostname, GPU
    inventory, CUDA_VISIBLE_DEVICES, runpod_pod_id) plus the worker's machine_id.
    None when the result never came from a worker (orphan/gateway-side failures)."""
    if not isinstance(raw, dict):
        return None
    meta = dict(raw.get("node") or {})
    if raw.get("machine_id"):
        meta["machine_id"] = raw["machine_id"]
    return meta or None


async def _mirror_status_to_db(
    session: AsyncSession, request_id: str, status: str, output: Any,
    worker_meta: Optional[dict] = None,
) -> None:
    """Reflect a terminal Redis result back into the requests table so the
    request-history UI shows it as completed/failed instead of stuck queued.
    Workers write only to Redis; without this, postgres never sees the update."""
    row = await session.get(ReqRow, request_id)
    if row is None:
        return
    if row.status == status and row.output == output and (worker_meta is None or row.worker_meta == worker_meta):
        return
    if row.status != status:
        metrics.observe_job_outcome(row.app_id, status)
    row.status = status
    row.output = output
    if worker_meta is not None:
        row.worker_meta = worker_meta
    if row.completed_at is None and status != "pending":
        from datetime import datetime, timezone
        row.completed_at = datetime.now(timezone.utc)
    await session.commit()


async def _record_stream_completion(request_id: str, ttft_ms: Optional[int],
                                    pt: Optional[int], ct: Optional[int]) -> None:
    """Stamp TTFT + token usage onto a streamed request's row. The streaming relay
    doesn't go through the result:{id} poll/mirror, so without this a streamed
    serverless request never records ttft/tokens/completed_at. Best-effort + guarded
    so it never clobbers a richer output a later lazy-mirror writes."""
    from datetime import datetime, timezone
    try:
        async with session_factory()() as s:
            row = await s.get(ReqRow, request_id)
            if row is None:
                return
            if ttft_ms is not None and row.ttft_ms is None:
                row.ttft_ms = ttft_ms
            if row.completed_at is None:
                row.completed_at = datetime.now(timezone.utc)
            if row.status not in ("completed", "failed", "cancelled", "timeout"):
                row.status = "completed"
            if pt is not None or ct is not None:
                out = dict(row.output or {})
                usage = dict(out.get("usage") or {})
                if pt is not None:
                    usage.setdefault("prompt_tokens", pt)
                if ct is not None:
                    usage.setdefault("completion_tokens", ct)
                out["usage"] = usage
                row.output = out
            await s.commit()
    except Exception:
        logger.warning("stream completion record failed for %s", request_id, exc_info=True)


async def _cancel_on_disconnect(
    rdb, request_id: str, app_id: Optional[str] = None, job_blob: Optional[str] = None
) -> None:
    """A streaming client disconnected before the response finished. Two jobs:
    (1) signal the worker to stop — `cancel:{request_id}`, honored by its
        pre-dequeue check AND mid-stream poll;
    (2) if the request hasn't already produced a terminal result, mark it
        cancelled NOW — Redis result + the Postgres row — so the queue UI flips to
        `failed` immediately instead of waiting for the worker to reach it. When we
        still hold the exact enqueued blob, also LREM it so the worker never even
        dequeues it (the OpenAI path has no blob in scope → the worker's pre-check
        skip-drains it cheaply instead).
    Best-effort: never raises into the stream teardown. The Postgres write uses a
    FRESH short-lived session — never the streamed request's, which would pin a pool
    connection for the whole stream (see the SSE pool-exhaustion gotcha)."""
    try:
        await rdb.set(f"cancel:{request_id}", "1", ex=_CANCEL_TTL_S)
    except Exception:
        logger.warning("cancel-on-disconnect: couldn't set cancel flag for %s", request_id)
        return
    try:
        blob = await rdb.get(f"result:{request_id}")
        cur = json.loads(blob).get("status") if blob else None
        if cur not in (None, "pending"):
            return  # worker already wrote a terminal result — don't clobber it
        out = {"error": "client disconnected"}
        await rdb.set(
            f"result:{request_id}",
            json.dumps({"status": "cancelled", "output": out}),
            ex=_CANCEL_TTL_S,
        )
        if job_blob and app_id:
            try:
                await rdb.lrem(f"queue:{app_id}", 0, job_blob)
            except Exception:
                pass
        async with session_factory()() as s:
            await _mirror_status_to_db(s, request_id, "cancelled", out)
        logger.info("client disconnected from %s → cancelled (queue shows failed)", request_id)
    except Exception:
        logger.exception("cancel-on-disconnect: couldn't finalize %s", request_id)


async def _resolve_model_to_app(session: AsyncSession, user: User, model_name: str) -> tuple[str, Optional[str]]:
    """Map an OpenAI `model` field to the endpoint that serves it.

    Returns (app_id, target_model). `target_model` is the real member-model name
    for a multi-model endpoint (so the worker can route locally), else None.

    Resolution order, scoped to the requesting user (admins keep the fast path):
      1. Exact app_id match — back-compat: single-mode clients send the endpoint
         name as `model`, and that's how it's worked.
      2. A multi-model endpoint whose `models[]` contains this model name.
      3. A single-mode endpoint whose `model` equals this name (real HF id).
    Ambiguity (>1 owned endpoint) → 409; none → 404.
    """
    from sqlalchemy import select

    # 1. Exact endpoint-name match.
    app = await session.get(App, model_name)
    if app is not None and (app.owner_id == user.id or user.is_admin):
        if (getattr(app, "mode", "single") or "single") == "multi":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "this is a multi-model endpoint — set 'model' to one of its member models, not the endpoint name",
                    "endpoint": model_name,
                    "models": [m.get("model") for m in (app.models or [])],
                },
            )
        return model_name, None

    # 2 + 3. Search the user's own endpoints for a model match.
    rows = (await session.execute(select(App).where(App.owner_id == user.id))).scalars().all()
    matches: list[tuple[str, Optional[str]]] = []
    for a in rows:
        if (a.mode or "single") == "multi":
            if any((m or {}).get("model") == model_name for m in (a.models or [])):
                matches.append((a.app_id, model_name))
        elif a.model == model_name:
            matches.append((a.app_id, None))
    # de-dupe by app_id, preserve order
    seen: set[str] = set()
    uniq = [m for m in matches if not (m[0] in seen or seen.add(m[0]))]
    if len(uniq) == 1:
        return uniq[0]
    if len(uniq) > 1:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "ambiguous model name — multiple of your endpoints serve it",
                "model": model_name,
                "candidates": [m[0] for m in uniq],
                "hint": "send the endpoint name as 'model' to disambiguate",
            },
        )
    raise HTTPException(
        status_code=404,
        detail={"error": "no endpoint serves this model for your account", "model": model_name},
    )


async def _resolve_endpoint_path(
    session: AsyncSession, user: User, app_id: str, model_field: Optional[str]
) -> tuple[str, Optional[str]]:
    """Path-scoped OpenAI route (`/{app_id}/v1/...`): the endpoint is fixed by the
    URL, so there's no global model-name resolution (and no cross-endpoint
    ambiguity). For a multi-model fleet `model` must name one of its members; for a
    single-model endpoint `model` is ignored (the endpoint serves its one model).
    Returns (app_id, target_model) — `target_model` is None for single-model."""
    app = await session.get(App, app_id)
    if app is None or not (app.owner_id == user.id or user.is_admin):
        raise HTTPException(
            status_code=404,
            detail={"error": "no such endpoint for your account", "endpoint": app_id},
        )
    if (getattr(app, "mode", "single") or "single") == "multi":
        members = [m.get("model") for m in (app.models or [])]
        if not model_field:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "this is a multi-model endpoint — set 'model' to one of its member models",
                    "endpoint": app_id,
                    "models": members,
                },
            )
        if model_field not in members:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "this endpoint does not serve that model",
                    "endpoint": app_id,
                    "model": model_field,
                    "models": members,
                },
            )
        return app_id, model_field
    return app_id, None


async def _proxy_to_upstream(
    request: Request,
    app_id: str,
    payload: dict,
    vllm_path: str,
    timeout_s: int,
):
    """Proxy-mode dispatch: forward the OpenAI request straight to the single
    model's vLLM over the gateway→VM forward tunnel. No Redis queue, no admission,
    no sleep/wake — a transparent reverse proxy. The upstream URL
    (`http://127.0.0.1:{local_forward_port}`) is published by the VM provider when
    it opens the tunnel (see vm_serverless_provider._wire_proxy_forward)."""
    rdb = request.app.state.redis
    upstream = await rdb.get(f"proxy:{app_id}:upstream")
    if isinstance(upstream, (bytes, bytearray)):
        upstream = upstream.decode()
    if not upstream:
        raise HTTPException(
            status_code=503,
            detail={"error": "proxy endpoint not ready — the worker is still starting "
                             "(no upstream tunnel yet). Check the Workers tab."},
        )
    cli: httpx.AsyncClient = request.app.state.proxy_http
    url = upstream.rstrip("/") + vllm_path
    httpx_to = httpx.Timeout(connect=10.0, read=timeout_s, write=timeout_s, pool=10.0)
    is_stream = bool(payload.get("stream"))

    if not is_stream:
        try:
            r = await cli.post(url, json=payload, timeout=httpx_to)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail={"error": f"proxy upstream unreachable: {type(e).__name__}: {e}"})
        if r.status_code >= 400:
            try:
                detail = r.json()
            except Exception:
                detail = {"error": r.text[:500]}
            raise HTTPException(status_code=r.status_code, detail=detail)
        return r.json()

    async def gen():
        try:
            async with cli.stream("POST", url, json=payload, timeout=httpx_to) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode("utf-8", "replace")[:500]
                    yield f"data: {json.dumps({'error': {'message': body, 'code': r.status_code}})}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return
                # vLLM already emits OpenAI SSE framing ("data: {…}\n\n") — pass bytes through.
                async for chunk in r.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as e:
            yield f"data: {json.dumps({'error': {'message': f'upstream stream error: {type(e).__name__}'}})}\n\n".encode()
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _proxy_audio_to_upstream(
    request: Request,
    app_id: str,
    payload: dict,
    vllm_path: str,
    timeout_s: int,
):
    """Proxy-mode audio dispatch: rebuild the multipart request from the base64
    payload and forward it straight to the model's vLLM (which serves
    /v1/audio/{transcriptions,translations} as native multipart) over the forward
    tunnel. The queue path base64s + has the worker rebuild multipart; here we
    rebuild + proxy directly (no queue)."""
    rdb = request.app.state.redis
    upstream = await rdb.get(f"proxy:{app_id}:upstream")
    if isinstance(upstream, (bytes, bytearray)):
        upstream = upstream.decode()
    if not upstream:
        raise HTTPException(status_code=503, detail={"error": "proxy endpoint not ready — the worker is still starting."})
    audio = base64.b64decode(payload["_audio_b64"])
    files = {"file": (payload.get("_filename") or "audio.wav", audio)}
    data = {"model": payload["model"], **{k: str(v) for k, v in (payload.get("_form") or {}).items()}}
    cli: httpx.AsyncClient = request.app.state.proxy_http
    url = upstream.rstrip("/") + vllm_path
    try:
        r = await cli.post(url, files=files, data=data,
                           timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=timeout_s, pool=10.0))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail={"error": f"proxy upstream unreachable: {type(e).__name__}: {e}"})
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = {"error": r.text[:500]}
        raise HTTPException(status_code=r.status_code, detail=detail)
    return r.json()


async def _openai_endpoint(
    request: Request,
    db_session: AsyncSession,
    user: User,
    payload: dict,
    vllm_path: str,
    explicit_app_id: Optional[str] = None,
):
    rdb = request.app.state.redis
    model_field = payload.get("model")
    if explicit_app_id is not None:
        # Path-scoped route (/{app_id}/v1/...): endpoint fixed by the URL; `model`
        # only selects the member of a multi-model fleet.
        app_id, target_model = await _resolve_endpoint_path(
            db_session, user, explicit_app_id, model_field
        )
    else:
        if not model_field:
            raise HTTPException(status_code=400, detail={"error": "missing 'model' field in request body"})
        app_id, target_model = await _resolve_model_to_app(db_session, user, model_field)

    # Proxy endpoints (single-model VM): bypass the queue and forward straight to
    # the model's vLLM over the gateway→VM tunnel. Fetch the App once to branch on
    # mode, then release the request-scoped DB connection (the proxy path only
    # talks to Redis + httpx — holding it would pin a pool slot for the stream).
    app_row = await db_session.get(App, app_id)
    if app_row is not None and getattr(app_row, "mode", "single") == "proxy":
        proxy_timeout_s = int(getattr(app_row, "request_timeout_s", 600) or 600)
        if target_model:
            payload = {**payload, "model": target_model}
        await db_session.close()
        return await _proxy_to_upstream(request, app_id, payload, vllm_path, proxy_timeout_s)

    is_stream = bool(payload.get("stream"))
    if is_stream:
        # Ask vLLM for a final usage chunk so streamed requests record token counts
        # too (the worker forwards the body verbatim; the relay below parses + stores
        # it). Don't clobber a caller's own stream_options.
        _so = dict(payload.get("stream_options") or {})
        _so.setdefault("include_usage", True)
        payload = {**payload, "stream_options": _so}
    request_id, timeout_s = await _admit_and_enqueue(
        rdb, db_session, app_id, user, payload, stream=is_stream, endpoint=vllm_path,
        target_model=target_model,
    )

    if not is_stream:
        # Non-stream: the worker posts result:{id} ONLY when generation FINISHES, so
        # the wait must cover the whole generation — not just a cold start. The old
        # hardcoded 60s killed any legit long generation (big max_tokens, a slow/large
        # model like GLM-5.2-FP8, long reasoning) and mislabelled it "cold-starting".
        # Honour the endpoint's configured request_timeout_s (the intended per-request
        # budget; default 600s) instead. Close the DB session first — the poll only
        # talks to Redis, and holding the Depends(get_session) connection for a minute+
        # pins a pool slot (the same leak the streaming path below avoids); re-open a
        # short-lived session only to mirror the terminal status.
        await db_session.close()
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            blob = await rdb.get(f"result:{request_id}")
            if blob:
                raw = json.loads(blob)
                status = raw.get("status")
                if status == "completed":
                    async with session_factory()() as s:
                        await _mirror_status_to_db(s, request_id, "completed", raw.get("output"), worker_meta=_worker_meta_from_result(raw))
                    return raw.get("output", {})
                if status in ("timeout", "cancelled", "failed"):
                    async with session_factory()() as s:
                        await _mirror_status_to_db(s, request_id, status, raw.get("output"), worker_meta=_worker_meta_from_result(raw))
                    raise HTTPException(status_code=504, detail=raw.get("output"))
            await asyncio.sleep(0.2)
        msg = (f"no completion in {timeout_s}s (endpoint request_timeout_s). The worker may be "
               f"cold-starting, OR the generation is taking longer than the timeout — raise "
               f"request_timeout_s on the endpoint, or use stream:true for long outputs.")
        async with session_factory()() as s:
            await _mirror_status_to_db(s, request_id, "timeout", {"error": msg})
        raise HTTPException(status_code=504, detail={"error": msg, "request_id": request_id})

    # Release the request-scoped DB connection BEFORE streaming. With
    # Depends(get_session) FastAPI keeps the dependency (and its pooled
    # connection) checked out until the streamed body finishes — but gen()
    # below talks only to Redis, never the DB. Holding it pins a pool slot for
    # the whole (often minute-plus) stream; under concurrent streams the async
    # pool exhausts and new requests block on connection checkout. Closing now
    # is safe (idempotent; the get_session ctx-exit will close again as a noop).
    await db_session.close()

    channel = f"stream:{request_id}"
    pubsub = rdb.pubsub()
    await pubsub.subscribe(channel)
    _t0 = time.perf_counter()

    async def gen():
        finished = False
        ttft_ms: Optional[int] = None
        pt = ct = None
        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                data = msg["data"]
                if ttft_ms is None:
                    ttft_ms = int((time.perf_counter() - _t0) * 1000)  # time-to-first-token
                yield f"data: {data}\n\n"
                try:
                    parsed = json.loads(data)
                    us = parsed.get("usage")
                    if isinstance(us, dict):
                        if us.get("prompt_tokens") is not None:
                            pt = us["prompt_tokens"]
                        if us.get("completion_tokens") is not None:
                            ct = us["completion_tokens"]
                    if parsed.get("done") or parsed.get("error"):
                        finished = True
                        break
                except json.JSONDecodeError:
                    continue
            yield "data: [DONE]\n\n"
        finally:
            if finished:
                # Stamp ttft/tokens/completed onto the row (detached — we may be mid-aclose).
                asyncio.create_task(_record_stream_completion(request_id, ttft_ms, pt, ct))
            else:
                await _cancel_on_disconnect(rdb, request_id, app_id)
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-Id": request_id,
        },
    )


@app.post("/v1/chat/completions")
async def openai_chat_completions(
    payload: dict,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    return await _openai_endpoint(request, session, user, payload, "/v1/chat/completions")


@app.post("/v1/completions")
async def openai_completions(
    payload: dict,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    return await _openai_endpoint(request, session, user, payload, "/v1/completions")


@app.post("/v1/embeddings")
async def openai_embeddings(
    payload: dict,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    payload.pop("stream", None)
    return await _openai_endpoint(request, session, user, payload, "/v1/embeddings")


@app.get("/v1/models")
async def openai_list_models(session: AsyncSession = Depends(get_session)):
    """OpenAI-compatible model list. Public (no auth) — pure model discovery.
    Lists every servable model id across all endpoints: multi-model members +
    single endpoint names (what clients send in the `model` field)."""
    from sqlalchemy import select

    rows = (await session.execute(select(App))).scalars().all()
    data: list[dict] = []
    seen: set[str] = set()
    for a in rows:
        created = int(a.created_at.timestamp()) if a.created_at else 0
        if (getattr(a, "mode", "single") or "single") == "multi":
            for m in (a.models or []):
                mid = m.get("model")
                if mid and mid not in seen:
                    seen.add(mid)
                    data.append({"id": mid, "object": "model", "created": created, "owned_by": a.app_id})
        elif a.app_id not in seen:
            seen.add(a.app_id)
            data.append({"id": a.app_id, "object": "model", "created": created, "owned_by": a.app_id})
    return {"object": "list", "data": data}


# ----- per-endpoint OpenAI-compatible routes -----
# `/{app_id}/v1/...` scopes the request to one endpoint by URL, so a client can
# point an OpenAI SDK at `base_url=<gateway>/{app_id}/v1` and run many endpoints
# side by side without the global `model`-name routing (no cross-endpoint
# ambiguity). The fixed `/v1/...` suffix keeps these from shadowing the literal
# global routes above (different segment count).

@app.post("/{app_id}/v1/chat/completions")
async def openai_chat_completions_scoped(
    app_id: str,
    payload: dict,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    return await _openai_endpoint(
        request, session, user, payload, "/v1/chat/completions", explicit_app_id=app_id
    )


@app.post("/{app_id}/v1/completions")
async def openai_completions_scoped(
    app_id: str,
    payload: dict,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    return await _openai_endpoint(
        request, session, user, payload, "/v1/completions", explicit_app_id=app_id
    )


@app.post("/{app_id}/v1/embeddings")
async def openai_embeddings_scoped(
    app_id: str,
    payload: dict,
    request: Request,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    payload.pop("stream", None)
    return await _openai_endpoint(
        request, session, user, payload, "/v1/embeddings", explicit_app_id=app_id
    )


@app.get("/{app_id}/v1/models")
async def openai_list_models_scoped(app_id: str, session: AsyncSession = Depends(get_session)):
    """OpenAI-compatible model list for ONE endpoint (public, like the global
    `/v1/models`). A multi-model fleet lists its members; a single-model endpoint
    lists its own name."""
    app = await session.get(App, app_id)
    if app is None:
        raise HTTPException(status_code=404, detail={"error": "no such endpoint", "endpoint": app_id})
    created = int(app.created_at.timestamp()) if app.created_at else 0
    data: list[dict] = []
    if (getattr(app, "mode", "single") or "single") == "multi":
        for m in (app.models or []):
            mid = m.get("model")
            if mid:
                data.append({"id": mid, "object": "model", "created": created, "owned_by": app.app_id})
    else:
        data.append({"id": app.app_id, "object": "model", "created": created, "owned_by": app.app_id})
    return {"object": "list", "data": data}


# ----- OpenAI-compatible AUDIO routes (Whisper: transcriptions / translations) -----
# These take multipart/form-data (a file upload), not JSON. The queue carries JSON
# only, so we base64 the clip into the job payload under `_audio_b64`; the worker
# rebuilds the multipart request for vLLM (see worker_agent.main.handle). Unary
# only — transcription returns one JSON body, no streaming here.

_AUDIO_MAX_BYTES = 25 * 1024 * 1024


async def _build_audio_payload(file: UploadFile, model: str, **form) -> dict:
    """Read + cap the uploaded clip, base64 it, and assemble the queue payload.
    `form` holds the optional OpenAI fields (language/prompt/response_format/…) —
    only the non-None ones are forwarded to vLLM."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail={"error": "empty audio upload"})
    if len(data) > _AUDIO_MAX_BYTES:
        raise HTTPException(status_code=413, detail={"error": "audio too large (max 25 MB)"})
    return {
        "model": model,
        "_audio_b64": base64.b64encode(data).decode(),
        "_filename": file.filename or "audio.wav",
        "_form": {k: v for k, v in form.items() if v is not None},
    }


async def _openai_audio_endpoint(
    request: Request,
    db_session: AsyncSession,
    user: User,
    payload: dict,
    vllm_path: str,
    explicit_app_id: Optional[str] = None,
):
    """Resolve the endpoint/model, enqueue the audio job, and (unary) wait for the
    worker's result. Mirrors `_openai_endpoint` but with a longer deadline — a cold
    Whisper member may need to load/wake before it can transcribe."""
    rdb = request.app.state.redis
    model_field = payload.get("model")
    if explicit_app_id is not None:
        app_id, target_model = await _resolve_endpoint_path(db_session, user, explicit_app_id, model_field)
    else:
        if not model_field:
            raise HTTPException(status_code=400, detail={"error": "missing 'model' field in request"})
        app_id, target_model = await _resolve_model_to_app(db_session, user, model_field)

    # proxy endpoints: rebuild + forward the multipart straight to the model's vLLM
    # (no queue), same as the JSON proxy path.
    app_row = await db_session.get(App, app_id)
    if app_row is not None and getattr(app_row, "mode", "single") == "proxy":
        if target_model:
            payload["model"] = target_model
        ptimeout = int(getattr(app_row, "request_timeout_s", 600) or 600)
        await db_session.close()
        return await _proxy_audio_to_upstream(request, app_id, payload, vllm_path, ptimeout)

    request_id, _timeout_s = await _admit_and_enqueue(
        rdb, db_session, app_id, user, payload, stream=False, endpoint=vllm_path,
        target_model=target_model,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        blob = await rdb.get(f"result:{request_id}")
        if blob:
            raw = json.loads(blob)
            status = raw.get("status")
            if status == "completed":
                await _mirror_status_to_db(db_session, request_id, "completed", raw.get("output"), worker_meta=_worker_meta_from_result(raw))
                return raw.get("output", {})
            if status in ("timeout", "cancelled", "failed"):
                await _mirror_status_to_db(db_session, request_id, status, raw.get("output"), worker_meta=_worker_meta_from_result(raw))
                raise HTTPException(status_code=504, detail=raw.get("output"))
        await asyncio.sleep(0.2)
    await _mirror_status_to_db(
        db_session, request_id, "timeout",
        {"error": "no completion in 120s — worker probably cold-starting / waking the model"},
    )
    raise HTTPException(
        status_code=504,
        detail={"error": "no completion in 120s — worker probably cold-starting; retry", "request_id": request_id},
    )


@app.post("/v1/audio/transcriptions")
async def openai_audio_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    payload = await _build_audio_payload(
        file, model, language=language, prompt=prompt,
        response_format=response_format, temperature=temperature,
    )
    return await _openai_audio_endpoint(request, session, user, payload, "/v1/audio/transcriptions")


@app.post("/v1/audio/translations")
async def openai_audio_translations(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    payload = await _build_audio_payload(
        file, model, prompt=prompt, response_format=response_format, temperature=temperature,
    )
    return await _openai_audio_endpoint(request, session, user, payload, "/v1/audio/translations")


@app.post("/{app_id}/v1/audio/transcriptions")
async def openai_audio_transcriptions_scoped(
    app_id: str,
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    payload = await _build_audio_payload(
        file, model, language=language, prompt=prompt,
        response_format=response_format, temperature=temperature,
    )
    return await _openai_audio_endpoint(
        request, session, user, payload, "/v1/audio/transcriptions", explicit_app_id=app_id
    )


@app.post("/{app_id}/v1/audio/translations")
async def openai_audio_translations_scoped(
    app_id: str,
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    payload = await _build_audio_payload(
        file, model, prompt=prompt, response_format=response_format, temperature=temperature,
    )
    return await _openai_audio_endpoint(
        request, session, user, payload, "/v1/audio/translations", explicit_app_id=app_id
    )


# ----- workers (machine auth, not user auth) -----

@app.post("/workers/register", response_model=WorkerRegisterResponse)
async def register_worker(req: WorkerRegisterRequest, request: Request):
    rdb = request.app.state.redis

    if os.environ.get("AUTOSCALER", "0") == "1":
        token_key = f"register_token:{req.machine_id}"
        expected = await rdb.get(token_key)
        if expected is None or expected != req.token:
            logger.warning(
                "register rejected: machine=%s token=%s (expected=%s)",
                req.machine_id, req.token[:8] + "...", "<set>" if expected else "<missing>",
            )
            raise HTTPException(status_code=401, detail="invalid or expired token")
        await rdb.delete(token_key)

    state = {
        "machine_id": req.machine_id,
        "app_id": req.app_id,
        "status": "registered",
        "last_seen": time.time(),
    }
    await rdb.set(f"worker:{req.machine_id}", json.dumps(state), ex=await _worker_ttl_s(rdb, req.app_id))
    await rdb.sadd(f"worker_index:{req.app_id}", req.machine_id)
    from .autoscaler import emit_worker_event
    await emit_worker_event(rdb, req.machine_id, req.app_id, "info", "registered with gateway")
    logger.info("worker registered: machine=%s app=%s", req.machine_id, req.app_id)
    redis_url = os.environ.get(
        "WORKER_REDIS_URL",
        os.environ.get("REDIS_URL", "redis://redis:6379"),
    )
    return WorkerRegisterResponse(ok=True, redis_url=redis_url)


@app.post("/workers/heartbeat")
async def heartbeat(req: WorkerHeartbeatRequest, request: Request):
    rdb = request.app.state.redis
    state = {
        "machine_id": req.machine_id,
        "app_id": req.app_id,
        "status": req.status,
        "last_seen": time.time(),
    }
    await rdb.set(f"worker:{req.machine_id}", json.dumps(state), ex=await _worker_ttl_s(rdb, req.app_id))
    await rdb.sadd(f"worker_index:{req.app_id}", req.machine_id)
    # Stamp the instant this worker first became servable. The autoscaler uses it
    # to give a freshly-ready worker a full idle window before teardown — a cold
    # start can outlast idle_timeout_s, so a request that woke a scale-to-zero
    # fleet would otherwise be stranded when the pod is torn down the moment it
    # goes ready (idle_for is measured from request *arrival*, not fulfilment).
    # set-once (NX): survives the loading→ready→loading churn of a model swap.
    if req.status == "ready":
        from .autoscaler import REGISTRATION_TOKEN_TTL_S as _REG_TTL
        await rdb.set(f"worker:{req.machine_id}:ready_since", str(time.time()), ex=_REG_TTL, nx=True)
    # Multi-model workers report per-model state (awake/asleep/loading + queue);
    # stash it for the status endpoint. Longer TTL than the heartbeat so a missed
    # beat doesn't blank the UI.
    if req.models is not None:
        await rdb.set(
            f"worker:{req.machine_id}:models",
            json.dumps(req.models),
            ex=WORKER_TTL_S * 3,
        )
        # Persist a member's crash reason (CUDA OOM, …) PER launch session, so the
        # UI can show WHY a past launch died long after the worker is gone (keyed
        # by app+model+session, like the historical logs). `reason` is set while a
        # model is dead/backing-off and points at the crashing session.
        for m in req.models:
            reason = m.get("reason")
            sess = m.get("session")
            model = m.get("model")
            if reason and sess and model:
                await rdb.set(
                    f"wlog_crash:{req.app_id}:{_logs_slug(model)}:{sess}",
                    str(reason)[:1024],
                    ex=WLOG_SESSION_TTL,
                )
    # Hand back any queued operator commands (kill/restart a member). One worker
    # per multi-model VM endpoint, so app-scoped is unambiguous; we pop them so
    # each command is delivered exactly once.
    commands: list[dict] = []
    cmd_key = f"app:{req.app_id}:model_cmds"
    raw_cmds = await rdb.lrange(cmd_key, 0, -1)
    if raw_cmds:
        await rdb.delete(cmd_key)
        for x in raw_cmds:
            try:
                commands.append(json.loads(x))
            except (json.JSONDecodeError, TypeError):
                pass
    drain = await rdb.exists(f"worker:{req.machine_id}:drain")
    return {"ok": True, "drain": bool(drain), "commands": commands}


@app.post("/workers/logs")
async def ingest_worker_logs(req: WorkerLogsRequest, request: Request):
    """Worker-agent ships batches of vLLM stdout lines here. We cap the list
    so a chatty worker can't fill Redis. No auth — workers are identified by
    machine_id, same trust model as /workers/heartbeat."""
    if not req.lines:
        return {"ok": True, "stored": 0}
    rdb = request.app.state.redis
    # Multi-model batches carry a `source` (served_name) → bucket per model so
    # each model's tab shows only its own vLLM output. Single-mode → flat key.
    key = (
        f"worker_logs:{req.machine_id}:{_logs_slug(req.source)}"
        if req.source else f"worker_logs:{req.machine_id}"
    )
    # Newest first (LPUSH); LTRIM keeps only the most recent N. Each line is
    # bounded to 4 KB to keep one runaway log line from eating the cap budget.
    truncated = [l[:4096] for l in req.lines if l]
    if not truncated:
        return {"ok": True, "stored": 0}
    await rdb.lpush(key, *reversed(truncated))
    await rdb.ltrim(key, 0, WORKER_LOGS_CAP - 1)
    # 1h TTL so logs naturally expire after the worker is gone.
    await rdb.expire(key, 3600)
    # Historical: also bucket per launch session (app+model+session), kept longer
    # and surviving re-provision, so the UI can open old logs (incl. a crash).
    if req.session and req.source:
        slug = _logs_slug(req.source)
        skey = f"wlog:{req.app_id}:{slug}:{req.session}"
        await rdb.lpush(skey, *reversed(truncated))
        await rdb.ltrim(skey, 0, WORKER_LOGS_CAP - 1)
        await rdb.expire(skey, WLOG_SESSION_TTL)
        idx = f"wlog_sessions:{req.app_id}:{slug}"
        digits = req.session.replace("-", "")
        score = float(digits) if digits.isdigit() else 0.0  # "YYYYMMDD-HHMMSS" sorts chronologically
        await rdb.zadd(idx, {req.session: score})
        await rdb.expire(idx, WLOG_SESSION_TTL)
    return {"ok": True, "stored": len(truncated)}


_METRIC_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?(\s+.+)$")


def _relabel_metrics(text: str, extra: str) -> str:
    """Rewrite a vLLM /metrics block so its series carry `extra` labels
    (sgpu_app/sgpu_machine/sgpu_model). Drops # HELP/# TYPE — they'd duplicate
    across sources — and merges the labels into each metric line, so many
    workers' identical metric names don't collide into one series."""
    out: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        name, labels, rest = m.group(1), m.group(2), m.group(3)
        if labels:
            inner = labels[1:-1].strip()
            merged = "{" + (f"{inner},{extra}" if inner else extra) + "}"
        else:
            merged = "{" + extra + "}"
        out.append(f"{name}{merged}{rest}")
    return "\n".join(out)


@app.post("/workers/metrics")
async def ingest_worker_metrics(req: WorkerMetricsRequest, request: Request):
    """A worker ships each member's raw vLLM /metrics here; we stash per
    machine+model with a short TTL (metrics are a live snapshot). No auth —
    same machine_id trust model as /workers/heartbeat + /workers/logs."""
    rdb = request.app.state.redis
    stored = 0
    for model, text in (req.metrics or {}).items():
        if not text:
            continue
        key = f"worker_metrics:{req.machine_id}:{_logs_slug(model)}"
        await rdb.set(
            key,
            json.dumps({"app_id": req.app_id, "machine_id": req.machine_id, "model": model, "text": text[:200_000]}),
            ex=120,
        )
        stored += 1
    return {"ok": True, "stored": stored}


async def _collect_worker_metrics(rdb: Any, app_id: Optional[str] = None) -> str:
    """Concatenate every worker's vLLM /metrics (or just one app's, when app_id is
    given), each relabeled with sgpu_app/sgpu_machine/sgpu_model so series from
    different workers/models don't collide."""
    keys = await rdb.keys("worker_metrics:*")
    parts: list[str] = []
    for k in keys:
        blob = await rdb.get(k)
        if not blob:
            continue
        try:
            d = json.loads(blob)
        except (ValueError, TypeError):
            continue
        if app_id is not None and d.get("app_id") != app_id:
            continue
        extra = (
            f'sgpu_app="{d.get("app_id","")}",'
            f'sgpu_machine="{d.get("machine_id","")}",'
            f'sgpu_model="{d.get("model","")}"'
        )
        block = _relabel_metrics(d.get("text", ""), extra)
        if block:
            parts.append(block)
    return ("\n".join(parts) + "\n") if parts else "# no worker metrics\n"


@app.get("/metrics/workers")
async def workers_metrics(request: Request):
    """Combined Prometheus scrape target: every worker's vLLM /metrics, relabeled
    with sgpu_app/sgpu_machine/sgpu_model so the series don't collide. Public
    (like /metrics) — pure telemetry, scrape it with Prometheus/VictoriaMetrics."""
    body = await _collect_worker_metrics(request.app.state.redis)
    return Response(content=body, media_type="text/plain; version=0.0.4")


@app.get("/{app_id}/metrics")
async def app_metrics(app_id: str, request: Request):
    """Per-endpoint Prometheus scrape: this app's workers' vLLM /metrics (relabeled)
    PLUS the gateway's own HTTP metrics for this app (serverless_http_requests_total /
    serverless_http_request_duration_seconds, labelled by http_status). The natural
    sibling of the serving URL `{base}/{app_id}/v1/...`. Public like /metrics — pure
    telemetry. Grafana scrapes this and can alert on non-2xx, e.g.:
        sum(increase(serverless_http_requests_total{http_status!~"2.."}[5m])) > 0"""
    worker_body = await _collect_worker_metrics(request.app.state.redis, app_id=app_id)
    gw_body = metrics.render_app(app_id).decode("utf-8")
    combined = f"{worker_body}\n{gw_body}" if worker_body else gw_body
    return Response(content=combined, media_type="text/plain; version=0.0.4")


async def _resolve_worker_app_id(rdb: Any, machine_id: str) -> Optional[str]:
    """Find which app a worker belongs to. Prefers worker:{mid} (live state)
    but falls back to worker_app:{mid} sidecar so /events still works after
    the worker has been terminated and worker:{mid} expired."""
    state_blob = await rdb.get(f"worker:{machine_id}")
    if state_blob:
        try:
            return json.loads(state_blob).get("app_id")
        except json.JSONDecodeError:
            pass
    return await rdb.get(f"worker_app:{machine_id}")


@app.get("/workers/{machine_id}/events")
async def get_worker_events(
    machine_id: str,
    request: Request,
    tail: int = 100,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Return the worker's lifecycle event timeline (provisioned, registered,
    scaled, drained, terminated, etc.). Auth: the requesting user must own
    the app this worker belongs to."""
    if tail < 1 or tail > 200:
        raise HTTPException(status_code=400, detail="tail must be 1..200")
    rdb = request.app.state.redis
    app_id = await _resolve_worker_app_id(rdb, machine_id)
    if not app_id:
        raise HTTPException(status_code=404, detail="worker not found or expired")
    target_app = await _load_owned_app(session, app_id, user)
    raw = await rdb.lrange(f"worker_events:{machine_id}", 0, tail - 1)
    events: list[dict[str, Any]] = []
    # Stored newest-first; flip to chronological for the UI.
    for entry in reversed(raw):
        try:
            events.append(json.loads(entry))
        except (json.JSONDecodeError, TypeError):
            continue
    return {
        "machine_id": machine_id,
        "app_id": target_app.app_id,
        "events": events,
        "count": len(events),
    }


@app.get("/workers/{machine_id}/logs")
async def get_worker_logs(
    machine_id: str,
    request: Request,
    tail: int = 300,
    user: User = Depends(require_section("inference")),
    session: AsyncSession = Depends(get_session),
):
    """Return the last `tail` container-stdout lines for a worker. Auth: the
    requesting user must own the app this worker belongs to."""
    if tail < 1 or tail > WORKER_LOGS_CAP:
        raise HTTPException(status_code=400, detail=f"tail must be 1..{WORKER_LOGS_CAP}")
    rdb = request.app.state.redis
    app_id = await _resolve_worker_app_id(rdb, machine_id)
    if not app_id:
        raise HTTPException(status_code=404, detail="worker not found or expired")
    target_app = await _load_owned_app(session, app_id, user)
    raw = await rdb.lrange(f"worker_logs:{machine_id}", 0, tail - 1)
    if not raw:
        # A multi-model worker ships PER source (one stream per member + a reserved
        # "__worker__" stream for the agent's own stdout) and never the flat key, so
        # "container logs" would be empty. The __worker__ stream IS the container's
        # main process output → use it. (Single-mode workers use the flat key.)
        raw = await rdb.lrange(f"worker_logs:{machine_id}:__worker__", 0, tail - 1)
    # Stored newest-first; flip to chronological for the UI.
    lines = list(reversed(raw))
    return {
        "machine_id": machine_id,
        "app_id": target_app.app_id,
        "lines": lines,
        "count": len(lines),
    }


def run():
    load_dotenv()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    accesslog.init_access_logging()
    host, port = os.environ.get("GATEWAY_BIND", "0.0.0.0:8080").rsplit(":", 1)
    # Local-dev hot reload: set GATEWAY_RELOAD=1 to restart the server on any
    # edit under the gateway package. Off by default so prod runs a single
    # stable process. We pass the app as an import string (above) which is what
    # uvicorn needs to respawn the worker; reload_dirs is scoped to this
    # package so the watcher doesn't crawl .venv / the web/ tree.
    reload = os.environ.get("GATEWAY_RELOAD", "").strip().lower() in ("1", "true", "yes")
    uvicorn.run(
        "gateway.main:app",
        host=host,
        port=int(port),
        log_level="info",
        reload=reload,
        reload_dirs=[os.path.dirname(__file__)] if reload else None,
    )


if __name__ == "__main__":
    run()
