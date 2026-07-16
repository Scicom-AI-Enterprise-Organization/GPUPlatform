"""Postgres-backed durable state: users + apps.

Redis still holds the hot path (queues, worker registrations, results, sessions
with TTL). Postgres holds anything that must survive restarts and have an owner.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from sqlalchemy import BigInteger, JSON, Boolean, Float, ForeignKey, String, Text, DateTime, Integer, UniqueConstraint, select, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PolicyRole(Base):
    """Admin-managed role template — a named bundle of section-access flags.

    Users attach to a role; their effective permissions come from the role's
    `sections` map. Admins bypass entirely. System roles (`is_system=True`)
    are seeded on first init and can't be deleted from the UI; their sections
    can still be edited if the admin wants to broaden / narrow them.
    """
    __tablename__ = "policy_roles"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # slug
    name: Mapped[str] = mapped_column(String(128), unique=True)
    sections: Mapped[dict] = mapped_column(JSON, default=dict, server_default="{}", nullable=False)
    is_system: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True, index=True, nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Tier roles: "user" (default, no platform access), "developer" (can use
    # platform sections, gated by policy_role), "admin" (everything).
    role: Mapped[str] = mapped_column(String(16), default="user", server_default="user", nullable=False)
    # Attached policy role — defines which sections this user can access.
    # NULL = no sections. Admins ignore this and have all access.
    policy_role_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("policy_roles.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # GitHub user ID for accounts linked via GitHub SSO. Stored as string
    # since GitHub returns numeric ids that fit easily but we keep room
    # for other SSO providers later.
    github_id: Mapped[Optional[str]] = mapped_column(String(64), unique=True, index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    apps: Mapped[list["App"]] = relationship(back_populates="owner", cascade="all, delete-orphan")


class AuditLog(Base):
    """Immutable record of every state-changing action across the platform.

    `actor_username` is captured as a snapshot so deleted users still appear
    in history. `details` is a free-form dict for action-specific extras
    (gpu type, model id, etc.) — keep it small; this table grows linearly.
    """
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    actor_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    actor_username: Mapped[str] = mapped_column(String(64), index=True)
    # Dotted action key, e.g. "compute.create" / "user.permissions_change".
    action: Mapped[str] = mapped_column(String(64), index=True)
    # "compute" | "benchmark" | "app" | "user" | …
    resource_type: Mapped[str] = mapped_column(String(32), index=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    resource_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )


class WorkerEvent(Base):
    """Durable per-worker lifecycle timeline — the persistent twin of the Redis
    `worker_events:{mid}` ring (which has a 1h TTL and vanishes once the worker
    stops heartbeating). One row per lifecycle transition so analytics can
    reconstruct when an endpoint was *actually serving* (provisioned →
    terminated spans) long after the worker is gone — e.g. a calendar-style
    on/off canvas per endpoint.

    `event` is a short stable key (see `_classify_worker_event`); `message` is
    the human string mirrored from the Redis ring. System-driven events
    (autoscaler scale-up / idle-teardown) have a NULL `actor_username`;
    user-triggered ones (kill / purge / redeploy / delete) stamp the actor.
    Grows fast under churn, hence a BigInteger PK and indexed (app_id,
    created_at) for range scans.
    """
    __tablename__ = "worker_events"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    app_id: Mapped[str] = mapped_column(String(64), index=True)
    machine_id: Mapped[str] = mapped_column(String(128), index=True)
    # Stable key: "provisioned" | "registered" | "scaled_up" | "scaled_down"
    # | "idle_terminated" | "drained" | "terminated" | "terminate_failed" | …
    event: Mapped[str] = mapped_column(String(48), index=True)
    level: Mapped[str] = mapped_column(String(16), default="info")
    message: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    actor_username: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )


class App(Base):
    __tablename__ = "apps"
    app_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(255))
    gpu: Mapped[str] = mapped_column(String(64))
    gpu_count: Mapped[int] = mapped_column(Integer, default=1, server_default="1", nullable=False)
    enable_metrics: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
    # Public endpoints surface (read-only) in every logged-in user's endpoint
    # list and let non-owners view the overview/workers/metrics — but never edit,
    # delete, run inference, or read logs/requests. Only the owner (or an admin)
    # can flip this or mutate the endpoint. Mirrors benchmarks.is_public.
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    autoscaler: Mapped[dict] = mapped_column(JSON)
    cpu: Mapped[int] = mapped_column(Integer, default=2)
    memory: Mapped[str] = mapped_column(String(32), default="16Gi")
    request_timeout_s: Mapped[int] = mapped_column(Integer, default=600)
    vllm_args: Mapped[str] = mapped_column(String(2048), default="", server_default="", nullable=False)
    # RunPod cloud tier the autoscaler should provision on. NULL = use provider
    # default (RUNPOD_CLOUD_TYPE env var, typically COMMUNITY). Only meaningful
    # for the RunPod provider; ignored by Fake/PI.
    cloud_type: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    # RunPod region allowlist (dataCenterIds) — comma-separated ids, or NULL/empty →
    # auto (RunPod picks any DC with capacity). RunPod-only; ignored by Fake/PI/VM.
    data_center_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # Per-worker disk sizing. NULL = use provider defaults
    # (RUNPOD_CONTAINER_DISK_GB / RUNPOD_VOLUME_GB).
    container_disk_gb: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    volume_gb: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Per-app cloud-account selection. NULL = use the global env-driven
    # provider singleton built at gateway startup. Per-app routing through
    # the autoscaler is still a follow-up — the column lands now so the
    # API surface is stable and we can backfill the resolver wiring later
    # without a second migration.
    provider_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    # Persistent log-archive storage (a kind="s3" Storage row) for worker/model
    # logs. NULL = no archival → logs stay in Redis with the 5000-line cap and 1h
    # TTL (the default). When set, every log line is archived uncapped to S3.
    storage_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    # Serving mode. "single" = one model per endpoint (the original behaviour).
    # "multi" = a fleet of vLLM servers on one VM, model-routed with vLLM
    # sleep/wake eviction. Multi requires a kind="vm" provider_id.
    mode: Mapped[str] = mapped_column(String(8), default="single", server_default="single", nullable=False)
    # Multi-model member spec; NULL/empty for single mode. Shape:
    #   [{"model": str, "tp": int, "extra_args": str}]
    # For multi, `model` is "" and `gpu_count` holds the VM's total GPU count.
    models: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # vLLM sleep level used when evicting an idle model (1 = offload weights to
    # CPU RAM, fast wake; 2 = discard weights, reload from disk). Multi only.
    sleep_level: Mapped[int] = mapped_column(Integer, default=1, server_default="1", nullable=False)
    # Extra environment variables applied to every vLLM process on the worker
    # (e.g. HF_HOME=/share/huggingface, TRITON_CACHE_DIR=…). Absolute-path values
    # are mkdir -p'd on the worker before launch. NULL/empty = none.
    env_vars: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # VM-only: pin to specific physical GPU indices, e.g. "0,1,2,3". NULL/empty =
    # use all of the VM's GPUs. Single-model sets CUDA_VISIBLE_DEVICES on the
    # worker; multi-model maps its slot packer onto exactly these ids (instead
    # of always starting at GPU 0). `gpu_count` is set to the count of these ids.
    visible_devices: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    # VM-only: the uv venv the worker runs `vllm serve` from, e.g.
    # "/share/vllm-venv" → launches {venv_path}/bin/python -m vllm. NULL = bare
    # `python3` on PATH.
    venv_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    # VM-only: pin vLLM to this version in venv_path — the worker `uv pip install`s
    # it if missing/mismatched, e.g. "0.19.1". NULL = use whatever's installed.
    vllm_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # Full `uv pip install` arg string for vLLM, used verbatim by the worker instead
    # of vllm_version (e.g. a nightly: "-U vllm --pre --extra-index-url …"). NULL =
    # install vllm[==vllm_version] with --torch-backend=auto.
    vllm_install_args: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Optional shell snippet the worker runs once per boot, after the venv is ready
    # and before any model launches — for model-specific setup that isn't a plain
    # pip install (e.g. building DeepGEMM). NULL = none.
    pre_script: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # ---- Multi-model fleet auto-retry (crash recovery) tuning; NULL = worker default ----
    # Max relaunch attempts before a crashed member is left DEAD.
    retry_max: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Never give up — relaunch indefinitely, ignoring retry_max (waits for free GPU
    # memory forever when retry_require_free_gpu is on). NULL/False = honour retry_max.
    retry_forever: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    # Backoff between relaunches (seconds): initial delay, doubled per attempt up to
    # retry_backoff_cap_s (the patience ceiling — longest wait before a retry).
    retry_backoff_base_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    retry_backoff_cap_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Hold a relaunch until the member's GPUs have free VRAM (poll, don't spend the
    # budget) instead of OOM-looping against a foreign job. NULL/False = retry now.
    retry_require_free_gpu: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    # Min free GPU memory (% of total) required to relaunch when the above is on.
    retry_gpu_free_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Consecutive failed /health probes before a settled engine is declared dead.
    health_fail_limit: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    owner: Mapped[User] = relationship(back_populates="apps")


class StressRun(Base):
    """A saved stress-test run for a serverless endpoint. The stress test is a
    browser-driven load generator (web tabs/stress.tsx); this persists each
    completed run's metric summary so runs / models can be compared over time and
    the comparison shared by link. Scoped to an app — visible to anyone who can
    access that app (same owner/admin check as the endpoint itself)."""
    __tablename__ = "stress_runs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    app_id: Mapped[str] = mapped_column(
        ForeignKey("apps.app_id", ondelete="CASCADE"), index=True
    )
    created_by: Mapped[str] = mapped_column(String(64), index=True)
    model: Mapped[str] = mapped_column(String(255), default="", server_default="", nullable=False)
    input_len: Mapped[int] = mapped_column(Integer, nullable=False)
    output_len: Mapped[int] = mapped_column(Integer, nullable=False)
    num_prompts: Mapped[int] = mapped_column(Integer, nullable=False)
    concurrency: Mapped[int] = mapped_column(Integer, nullable=False)
    # The client-computed metric block (throughput + latency percentiles). Opaque
    # JSON — the gateway stores and returns it verbatim.
    summary: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class Provider(Base):
    """User-registered cloud provider — VM (bare metal), RunPod, PI.

    `kind` selects the runtime adapter; `config` is a free-form JSON blob whose
    schema depends on kind. Any secret fields (private_key, api_key) are
    Fernet-encrypted before being stored here and decrypted on read.

    For phase 1 only `kind == "vm"` is supported. Expected config keys:
        { "host": str, "port": int, "user": str, "private_key": <encrypted str>,
          "gpus": [str, ...], "gpu_count": int }   # gpus/gpu_count set by last Test
    """
    __tablename__ = "providers"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(16), index=True)  # "vm" | "runpod" | "pi"
    config: Mapped[dict] = mapped_column(JSON, default=dict, server_default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class GlobalEnv(Base):
    """Org-wide environment variables / secrets, set by admins in the UI and
    merged into every workload's environment (benchmark pods + serverless VM/
    RunPod workers). A per-resource var of the same name overrides the global one.

    `value_enc` is always Fernet-encrypted at rest. `is_secret` controls whether
    the API ever returns the plaintext (secrets are masked; non-secrets are shown
    so admins can read back e.g. a default region)."""
    __tablename__ = "global_env"
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value_enc: Mapped[str] = mapped_column(String(8192), nullable=False)
    is_secret: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    updated_by: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class TrackingCredential(Base):
    """A named experiment-tracker credential (Weights & Biases / MLflow), shown
    as a card on the Secrets page and selectable per Autotrain run. Org-wide +
    admin-managed, like GlobalEnv. `config_enc` is Fernet-encrypted JSON whose
    shape depends on `kind`:
        wandb : {"api_key": str}
        mlflow: {"uri": str, "username": str, "password": str}
    The Autotrain runner decrypts the referenced credential and injects the
    canonical env (WANDB_API_KEY / MLFLOW_TRACKING_URI/USERNAME/PASSWORD)."""
    __tablename__ = "tracking_credentials"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # track-<hex8>
    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(16), index=True)  # "wandb" | "mlflow"
    config_enc: Mapped[str] = mapped_column(String(8192), nullable=False)
    created_by: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class Storage(Base):
    """A storage backend the platform writes to — where AutoTrain datasets,
    benchmark logs, serverless inference logs, etc. get persisted. NOT a single
    dataset; a reusable destination that features reference by id.

    Two kinds:
    - `s3`          — an S3 (or S3-compatible: R2, MinIO) bucket.
    - `huggingface` — a HuggingFace token holder for pushing repos.

    `config` is a JSON blob whose schema depends on `kind`. Secrets live under
    `credentials_enc` (Fernet-encrypted JSON) and are never returned to the UI:
        s3:          { "bucket": str, "prefix": str|None, "region": str|None,
                       "endpoint": str|None,
                       "credentials_enc": <enc {accessKeyId, secretAccessKey}> }
        huggingface: { "credentials_enc": <enc {token}> }
    When `credentials_enc` is absent the runtime falls back to env (AWS_* for
    s3, HF_TOKEN for huggingface). `description` holds free-form notes.
    """
    __tablename__ = "storage"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(16), index=True)  # "s3" | "huggingface"
    description: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)  # notes
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
    config: Mapped[dict] = mapped_column(JSON, default=dict, server_default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ServerlessLogArchive(Base):
    """Index of a serverless endpoint's persisted log files (one row per launch
    session per source). Populated by the gateway's log-archive flusher when an
    app has a log-archive `storage_id` set; the actual log bytes live in that
    storage under `serverless-logs/{app_id}/{slug}/{session}/{seq}.log` parts.

    This table is the searchable/paginated index the Logs tab browses — it
    scales to thousands of files where the Redis session list (capped, 1h TTL)
    does not. One file = one (app_id, slug, session).
    """
    __tablename__ = "serverless_log_archive"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # slog-<hex>
    app_id: Mapped[str] = mapped_column(String(64), index=True)
    storage_id: Mapped[str] = mapped_column(String(64))
    # _logs_slug(source); "__worker__" for the worker-agent scheduler log.
    slug: Mapped[str] = mapped_column(String(128), index=True)
    # Human label: the served model name, or "__worker__" for the agent log.
    model: Mapped[str] = mapped_column(String(255))
    # Per-launch timestamp "YYYYMMDD-HHMMSS" — the unit the worker tags batches by.
    session: Mapped[str] = mapped_column(String(32))
    # Object-key prefix for this file's parts (within the storage's own root).
    key_prefix: Mapped[str] = mapped_column(String(1024))
    next_seq: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    bytes: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0", nullable=False)
    lines: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0", nullable=False)
    # Crash reason for this launch (mirrored from wlog_crash:*), NULL if clean.
    crash: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    __table_args__ = (
        UniqueConstraint("app_id", "slug", "session", name="uq_slog_app_slug_session"),
    )


class Dataset(Base):
    """An Autotrain dataset — a named pointer to a metadata file of
    {audio, transcription} rows living in a `Storage` backend (S3) or on a
    HuggingFace repo. Mirrors the Benchmark/Storage ownership model: owned by a
    user, references a Storage by id.

    `kind` discriminates the source:
    - `upload` — a CSV/JSON/JSONL metadata file uploaded through the UI and
      written to the dataset's S3 storage under `datasets/{id}/{filename}`.
    - `s3`     — references a metadata file already living in S3 (`s3_metadata_uri`).
    - `hf`     — references an existing HuggingFace dataset repo (`hf_repo`).

    The audio referenced by each row resolves against the storage prefix +
    `audio_prefix`. `audio_field`/`transcription_field` are the column names the
    parser detected (defaults `audio`/`transcription`). `hf_*` track a push of
    the metadata file to a HuggingFace repo.
    """
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # ds-<hex8>
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    kind: Mapped[str] = mapped_column(String(16), default="upload", server_default="upload", nullable=False)
    # Storage row (kind=s3 for upload/s3, kind=huggingface for hf). Plain string
    # ref (not a hard FK) so deleting a storage doesn't cascade-delete datasets.
    storage_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    audio_prefix: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    s3_metadata_uri: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    metadata_filename: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    format: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # csv|json|jsonl|chinidataset
    num_rows: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    audio_field: Mapped[str] = mapped_column(String(128), default="audio", server_default="audio", nullable=False)
    transcription_field: Mapped[str] = mapped_column(String(128), default="transcription", server_default="transcription", nullable=False)
    # TTS-only: which column holds the speaker label/name. The TTS pack step
    # (pack_stage1) prepends "<speaker>: " to each transcript. None → the packer
    # falls back to a constant speaker (one voice).
    speaker_field: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    # Rows manually un-ticked in the row browser so they're EXCLUDED from training
    # (a JSON list of metadata-file row indices). Default/empty → every row is
    # included. Applied by the S3/upload dataset readers in the trainers.
    excluded_rows: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Per-split transcription column overrides for HF sources whose splits use
    # different column names (e.g. {"train": "text", "test": "after"}). Empty/None
    # → use `transcription_field` for every split. The audio column is assumed
    # consistent across splits (`audio_field`).
    split_fields: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # When this (zip-backed HF) dataset has been materialised to S3, the id of
    # the resulting audio dataset. Lets the source page resolve + play audio by
    # joining each row's audio basename against the materialised S3 audio.
    audio_dataset_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    hf_repo: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    hf_revision: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    hf_synced_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Audio-zip → audio-column transform job: "" (idle) | running | done | failed.
    # `transform_log` holds a short tail of progress lines for the UI to poll.
    transform_status: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    transform_log: Mapped[Optional[str]] = mapped_column(String(8192), nullable=True)
    # Labeling-platform source (kind=label): a live reference to a Label project.
    # We store the base URL + project id and the Fernet-encrypted access token;
    # preview/import fetch the project's export.v1.jsonl (audio_url + transcription)
    # on demand with `Authorization: Bearer <lpat token>`.
    label_base_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    label_project_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    label_token_enc: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    # Which review status to import from the project: approved | rejected |
    # not_reviewed | all (the export endpoint's `status` filter). Default approved.
    label_status: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    # Optional point-in-time cutoff (ISO-8601, UTC) for the import: only tasks whose
    # last_updated_at is at or before this instant are pulled (the export endpoint's
    # `updated_until` filter). None → no upper bound (every task). Set on /datasets/new
    # and editable later — lets a label dataset materialise a stable snapshot.
    label_updated_until: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # Alternatively the lpat token can come from a named global secret instead of
    # being stored per-dataset; resolved via load_global_env() at use time.
    label_token_secret: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    # How many times to retry a per-clip audio download during the transform before
    # giving up on that clip. None / ≤ 0 → RETRY UNTIL SUCCESS (the default): the
    # labelling-platform ingress flakes under the rapid-fire load of a big project,
    # and a single-attempt fetch used to SILENTLY drop 20–50% of rows. Set a positive
    # cap to bound the attempts (a give-up is then counted + surfaced, not hidden).
    label_download_retries: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # kind=llm: which column holds the OpenAI-compatible messages array
    # ([{role,content}]). Default "messages". In DPO (preference) mode this names
    # the CHOSEN column and `rejected_field` names the rejected column.
    messages_field: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    # kind=llm preference (DPO) mode: which column holds the REJECTED response
    # (chosen = messages_field). NULL → chat/SFT mode (no preference pairing). Its
    # presence is what flips the dataset into "dpo" mode (columns card + preview).
    rejected_field: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    # When this (s3-backed) dataset has been published to the self-hosted HF mirror
    # — the id of the CatalogRepo (repo_type=dataset) serving its files over /hf, so
    # the dataset page can show pull snippets + link to it. NULL = not published.
    catalog_repo_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class CatalogRepo(Base):
    """A self-hosted HuggingFace repo (model or dataset) served over the HF Hub
    protocol (`hf_mirror_api.py`) and managed via the catalog UI (`catalog_api.py`).

    Clients set `HF_ENDPOINT=<gateway>/hf` + `HF_TOKEN=sgpu_…` and then
    `from_pretrained`/`hf download` (read) and `push_to_hub`/`hf upload` (write)
    hit GPUPlatform instead of huggingface.co. The bytes live in a `Storage`
    backend (s3 / local / sftp) under `prefix`:
    - regular (git) files at `{prefix}/{path}`
    - LFS blobs (content-addressed) at `{prefix}/.hf-lfs/{oid}` — `manifest` maps
      each repo path → its oid/size/etag so a resolve can find the blob.

    `manifest` is the file list of the single `main` revision:
        [{ "path": str, "size": int, "oid": str, "lfs": bool, "etag": str }]
    `sha` is a synthetic commit id (sha1 of the sorted manifest), bumped on every
    commit. We keep only `main` — no branches/tags/history (basic mirror).
    """
    __tablename__ = "catalog_repos"
    __table_args__ = (
        UniqueConstraint("repo_type", "namespace", "name", name="uq_catalog_repo_id"),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # repo-<hex8>
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    repo_type: Mapped[str] = mapped_column(String(16), default="model", server_default="model", nullable=False, index=True)  # model|dataset
    namespace: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    full_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)  # "namespace/name"
    # Storage row (kind=s3|local|sftp). Soft string ref (not a hard FK) so deleting
    # a storage doesn't cascade-delete repo rows — matches the Dataset model.
    storage_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    prefix: Mapped[str] = mapped_column(String(1024), nullable=False)  # key prefix within the storage
    sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    private: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    manifest: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    num_files: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Revisioned repos (mirror-native pushes) keep extra named branches in
    # `CatalogRevision` and store file bytes CONTENT-ADDRESSED at `{prefix}/blobs/{oid}`
    # (so two branches' same-named files don't collide). Flat repos (False — the
    # default; published/registered-over-existing-data repos) keep the single
    # `manifest`/`sha` "main" at path-addressed `{prefix}/{path}` (unchanged). The
    # `manifest`/`sha` above are ALWAYS the head of `default_branch` (main).
    versioned: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    default_branch: Mapped[str] = mapped_column(String(255), default="main", server_default="main", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class CatalogRevision(Base):
    """A named, OVERWRITEABLE revision (branch) of a versioned `CatalogRepo` other
    than the default `main` branch. Each row is a branch's CURRENT file set — a
    push to that revision overwrites this row's `manifest`/`sha` (not an immutable
    commit history; main lives denormalized on `CatalogRepo.manifest`). Pulling
    `revision=<name>` (or its sha) resolves to this manifest; bytes are
    content-addressed under the repo's `{prefix}/blobs/{oid}` shared with main."""
    __tablename__ = "catalog_revisions"
    __table_args__ = (
        UniqueConstraint("repo_id", "name", name="uq_catalog_revision"),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # rev-<hex8>
    repo_id: Mapped[str] = mapped_column(ForeignKey("catalog_repos.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)  # branch name, e.g. "checkpoint-v1"
    sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    manifest: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    num_files: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class GitopsRepo(Base):
    """A git repository the platform reconciles platform resources from (GitOps).

    YAML manifests under `path` declare Apps / Storage / Datasets / Providers /
    Benchmarks / TrainingRuns; the reconciler creates, updates and prunes the
    live resources to match. Secrets are NEVER committed to git — manifests
    reference `GlobalEnv` keys by name, resolved in-memory at apply time.

    `token_secret` names a GlobalEnv key holding a git access token (private
    repos). `webhook_secret_enc` is the Fernet-encrypted HMAC secret for the
    push webhook. `prune=True` (the default) means a resource whose manifest is
    removed gets deleted on the next sync (full GitOps)."""
    __tablename__ = "gitops_repos"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # gitops-<hex8>
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    url: Mapped[str] = mapped_column(String(1024))
    branch: Mapped[str] = mapped_column(String(255), default="main", server_default="main", nullable=False)
    # Subdirectory within the repo to scan; NULL/empty → repo root.
    path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    # GlobalEnv key holding a git token for private clone/fetch (NULL → public).
    token_secret: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    webhook_secret_enc: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    prune: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
    poll_interval: Mapped[int] = mapped_column(Integer, default=300, server_default="300", nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
    last_synced_sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    last_sync_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # never | syncing | ok | error
    last_sync_status: Mapped[str] = mapped_column(String(16), default="never", server_default="never", nullable=False)
    last_sync_error: Mapped[Optional[str]] = mapped_column(String(8192), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class GitopsResource(Base):
    """Ledger mapping a GitopsRepo manifest (kind, name) → the live platform
    resource it created, plus the last-applied spec hash. Drives drift detection
    (hash changed → re-apply) and prune (a ledger row whose manifest disappeared
    → delete the underlying resource). `resource_id` is the platform id
    (app name / store-… / ds-… / prov-… / bench-… / train-…); NULL if the last
    create attempt failed (`status='error'`)."""
    __tablename__ = "gitops_resources"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # gres-<hex8>
    repo_id: Mapped[str] = mapped_column(ForeignKey("gitops_repos.id", ondelete="CASCADE"), index=True)
    # app | storage | dataset | provider | benchmark | training_run
    kind: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(255))
    resource_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    generation: Mapped[int] = mapped_column(Integer, default=1, server_default="1", nullable=False)
    spec_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="applied", server_default="applied", nullable=False)  # applied|error
    error: Mapped[Optional[str]] = mapped_column(String(8192), nullable=True)
    last_synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class Request(Base):
    __tablename__ = "requests"
    request_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    app_id: Mapped[str] = mapped_column(String(64), ForeignKey("apps.app_id", ondelete="CASCADE"), index=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    endpoint: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    output: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Where the request was served — written by the worker into its Redis result
    # ({machine_id, hostname, gpu_name, gpu_count, gpu_memory, driver_version,
    # visible_devices, runpod_pod_id}) and mirrored here alongside status/output.
    # NULL for requests that never reached a worker (or pre-upgrade rows).
    worker_meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    is_stream: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Time-to-first-token (ms): wall time from dispatch to the first streamed content
    # chunk reaching the gateway. Captured by the stream relay; NULL for non-stream /
    # failed / pre-upgrade rows. Total latency is (completed_at - created_at).
    ttft_ms: Mapped[Optional[int]] = mapped_column(nullable=True)


class ApiKey(Base):
    """A long-lived, revocable API token tied to a user. The plaintext key
    (`sgpu_…`) is shown once at creation; only its SHA-256 hash is stored. A key
    authenticates as its owner and inherits that user's role + section access —
    there are no separate per-key scopes. Revoking sets `revoked_at`."""
    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # ak-<hex>
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    # First ~12 chars of the key ("sgpu_AbCd…") for display — never the secret.
    prefix: Mapped[str] = mapped_column(String(16))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # sha256 hex
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class UserSession(Base):
    """Browser / script session token stored in Postgres. No TTL — sessions
    persist until explicitly revoked (logout) or the user is deleted."""
    __tablename__ = "user_sessions"
    token: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


_engine = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    return url


async def init_db() -> None:
    global _engine, _sessionmaker
    # Import side-effect: registers Benchmark / BenchmarkJob / ComputePod /
    # TrainingRun tables on Base before create_all runs.
    from . import bench  # noqa: F401
    from . import compute  # noqa: F401
    from . import training_api  # noqa: F401
    from . import quantization_api  # noqa: F401  # registers QuantizationJob
    from . import proxy_api  # noqa: F401  # registers ProxyEndpoint / ProxyRequest
    # Explicit pool sizing — the SQLAlchemy async default (pool_size=5,
    # max_overflow=10 → 15 conns, pool_timeout=30s) is small; under load a burst
    # of concurrent requests exhausted it and every new checkout blocked 30s,
    # wedging the gateway. pool_recycle also evicts connections idle past the prod
    # Redis/PG ELB timeout. All tunable via env.
    _engine = create_async_engine(
        get_database_url(),
        pool_pre_ping=True,
        pool_size=int(os.environ.get("DB_POOL_SIZE", "20") or "20"),
        max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", "20") or "20"),
        pool_timeout=float(os.environ.get("DB_POOL_TIMEOUT_S", "10") or "10"),
        pool_recycle=int(os.environ.get("DB_POOL_RECYCLE_S", "1800") or "1800"),
        connect_args={
            "server_settings": {
                # Reap any session left idle-in-transaction — e.g. a redeployed/killed
                # gateway pod whose in-flight txn still holds ACCESS SHARE on a hot
                # table (a leaked SSE get_session). Postgres won't reap a dead pod's
                # TCP connection for a long time, so without this its lock blocks the
                # next boot's ALTER TABLE migrations forever ("stuck initializing
                # postgres" → manual Postgres restart). Now the holder self-clears.
                "idle_in_transaction_session_timeout":
                    os.environ.get("DB_IDLE_IN_TXN_TIMEOUT_MS", "120000") or "120000",
            },
        },
    )
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    async with _engine.begin() as conn:
        # Backstop so an idempotent ALTER can't hang forever waiting on ACCESS
        # EXCLUSIVE (a conflicting lock from a just-killed pod). Set > the
        # idle-in-txn reap above so the holder self-clears first and the migration
        # then proceeds; if something genuinely holds the lock longer, we fail
        # loudly (→ restart + retry) instead of hanging until someone bounces PG.
        _lock_ms = int(os.environ.get("DB_MIGRATION_LOCK_TIMEOUT_MS", "150000") or "150000")
        await conn.execute(text(f"SET LOCAL lock_timeout = {_lock_ms}"))
        await conn.run_sync(Base.metadata.create_all)
        # ── Steady-state fast-path ──────────────────────────────────────────
        # The `ADD COLUMN IF NOT EXISTS` statements below still take an ACCESS
        # EXCLUSIVE lock on each table EVEN when the column already exists (the
        # lock is grabbed before the existence check). During a rolling deploy the
        # previous pod is still live and constantly reading the hot tables
        # (apps/benchmarks/requests), so those no-op locks queue behind its traffic
        # and the new pod wedges right here — the "stuck initializing postgres"
        # failure the connect_args above try to bound (they only help a *dead idle*
        # holder; a live old pod never self-clears). So once the DB already has
        # every column our models declare, skip the whole idempotent block: the
        # inspection below hits only the system catalogs (no table locks).
        # create_all already added any missing TABLES, and the concurrent-index
        # block at the end is a catalog-only no-op once built. The first boot after
        # a new column is added still runs the block (once) to apply it.
        # NOTE: an index-only migration (a new index with no new column) won't be
        # picked up by this column check — add such migrations above create_all or
        # extend this guard.
        def _schema_has_all_model_columns(sync_conn) -> bool:
            from sqlalchemy import inspect as _sa_inspect
            insp = _sa_inspect(sync_conn)
            tables = set(insp.get_table_names())
            for tbl in Base.metadata.tables.values():
                if tbl.name not in tables:
                    return False
                have = {c["name"] for c in insp.get_columns(tbl.name)}
                if any(col.name not in have for col in tbl.columns):
                    return False
            return True

        if await conn.run_sync(_schema_has_all_model_columns):
            import logging as _logging
            _logging.getLogger("gateway.db").info(
                "schema already current — skipping idempotent migrations (no table locks)"
            )
            return
        # Idempotent column adds for in-place upgrades — Base.metadata.create_all
        # only creates missing tables, not missing columns on existing ones.
        await conn.execute(text(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE"
        ))
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS catalog_repo_id VARCHAR(64)"
        ))
        # Point-in-time cutoff for kind=label imports (export `updated_until`). NULL on
        # existing rows → unchanged behaviour (import every task).
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS label_updated_until VARCHAR(64)"
        ))
        # kind=llm DPO (preference) mode: the rejected-response column (chosen =
        # messages_field). NULL on existing rows → chat/SFT mode (unchanged).
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS rejected_field VARCHAR(128)"
        ))
        # Per-clip audio-download retry cap for kind=label transforms. NULL on existing
        # rows → retry until success (the new default; fixes the silent-drop bug).
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS label_download_retries INTEGER"
        ))
        # Catalog revisions: existing repos stay flat (versioned=false) → no behaviour
        # change; only mirror-native pushes set versioned=true. catalog_revisions
        # itself is created by create_all above.
        await conn.execute(text(
            "ALTER TABLE catalog_repos ADD COLUMN IF NOT EXISTS versioned BOOLEAN NOT NULL DEFAULT FALSE"
        ))
        await conn.execute(text(
            "ALTER TABLE catalog_repos ADD COLUMN IF NOT EXISTS default_branch VARCHAR(255) NOT NULL DEFAULT 'main'"
        ))
        await conn.execute(text(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR(255)"
        ))
        # Public benchmarks: an owner can flip a run to public so it shows up
        # (read-only) in everyone's benchmark list. Existing rows stay private.
        await conn.execute(text(
            "ALTER TABLE benchmarks ADD COLUMN IF NOT EXISTS is_public BOOLEAN NOT NULL DEFAULT FALSE"
        ))
        # Public serverless endpoints: an owner can flip an endpoint public so it
        # shows up (read-only) in every user's endpoint list. Existing rows stay
        # private. Same shape as benchmarks.is_public above.
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS is_public BOOLEAN NOT NULL DEFAULT FALSE"
        ))
        # Public LLM API proxies: an admin can flip a proxy public so any logged-in
        # user can view its read-only info card (serving URL + model aliases) and
        # use the data plane. Existing rows stay private/admin-only.
        await conn.execute(text(
            "ALTER TABLE proxy_endpoints ADD COLUMN IF NOT EXISTS public BOOLEAN NOT NULL DEFAULT FALSE"
        ))
        # Time-to-first-token (ms) for the usage/activity analytics — recorded by the
        # streaming relays (serverless requests + proxy_requests).
        await conn.execute(text(
            "ALTER TABLE requests ADD COLUMN IF NOT EXISTS ttft_ms INTEGER"
        ))
        await conn.execute(text(
            "ALTER TABLE proxy_requests ADD COLUMN IF NOT EXISTS ttft_ms INTEGER"
        ))
        # NOTE: the queue-history indexes on the high-volume requests/proxy_requests
        # tables are built CONCURRENTLY *after* this transaction (see below) so they
        # never take an ACCESS EXCLUSIVE lock during startup on a large prod table.
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS vllm_args VARCHAR(2048) NOT NULL DEFAULT ''"
        ))
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS gpu_count INTEGER NOT NULL DEFAULT 1"
        ))
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS enable_metrics BOOLEAN NOT NULL DEFAULT TRUE"
        ))
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS cloud_type VARCHAR(16)"
        ))
        # RunPod region allowlist (comma-separated dataCenterIds). ADD first (fresh
        # DBs get 255); the widen ALTERs bump any column left at the earlier 32 width.
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS data_center_id VARCHAR(255)"
        ))
        await conn.execute(text(
            "ALTER TABLE apps ALTER COLUMN data_center_id TYPE VARCHAR(255)"
        ))
        await conn.execute(text(
            "ALTER TABLE compute_pods ADD COLUMN IF NOT EXISTS data_center_id VARCHAR(255)"
        ))
        await conn.execute(text(
            "ALTER TABLE compute_pods ALTER COLUMN data_center_id TYPE VARCHAR(255)"
        ))
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS container_disk_gb INTEGER"
        ))
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS volume_gb INTEGER"
        ))
        # Widen datasets.format (was VARCHAR(8), sized for csv/json/jsonl) so the
        # packed format label "chinidataset" fits. Idempotent: widening only.
        await conn.execute(text(
            "ALTER TABLE datasets ALTER COLUMN format TYPE VARCHAR(32)"
        ))
        # Per-app cloud-account selection — API surface; autoscaler still
        # uses the global env-driven provider for now.
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS provider_id VARCHAR(64)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_apps_provider_id ON apps(provider_id)"
        ))
        # Multi-model serving on VMs: per-endpoint mode, member-model spec, and
        # vLLM sleep level. Existing rows backfill to single-model behaviour.
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS mode VARCHAR(8) NOT NULL DEFAULT 'single'"
        ))
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS models JSON"
        ))
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS sleep_level INTEGER NOT NULL DEFAULT 1"
        ))
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS env_vars JSON"
        ))
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS visible_devices VARCHAR(128)"
        ))
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS venv_path VARCHAR(512)"
        ))
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS vllm_version VARCHAR(32)"
        ))
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS pre_script TEXT"
        ))
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS vllm_install_args TEXT"
        ))
        # Multi-model fleet auto-retry (crash recovery) tuning — NULL = worker default.
        for _col, _type in (
            ("retry_max", "INTEGER"),
            ("retry_forever", "BOOLEAN"),
            ("retry_backoff_base_s", "DOUBLE PRECISION"),
            ("retry_backoff_cap_s", "DOUBLE PRECISION"),
            ("retry_require_free_gpu", "BOOLEAN"),
            ("retry_gpu_free_pct", "DOUBLE PRECISION"),
            ("health_fail_limit", "INTEGER"),
        ):
            await conn.execute(text(f"ALTER TABLE apps ADD COLUMN IF NOT EXISTS {_col} {_type}"))
        # Per-endpoint log-archive storage (a kind=s3 Storage). NULL = Redis-only.
        await conn.execute(text(
            "ALTER TABLE apps ADD COLUMN IF NOT EXISTS storage_id VARCHAR(64)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_apps_storage_id ON apps(storage_id)"
        ))
        await conn.execute(text(
            "ALTER TABLE benchmark_shares ADD COLUMN IF NOT EXISTS pairing JSON"
        ))
        await conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email ON users(email) WHERE email IS NOT NULL"
        ))
        # Role rollout: only backfill on the migration that first adds the
        # column. After that, new users default to 'user' (no access) and
        # admins promote them manually. Existing users at migration time get
        # promoted to 'developer' so we don't break their current access.
        await conn.execute(text("""
            DO $$
            DECLARE col_exists boolean;
            BEGIN
              SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'users' AND column_name = 'role'
              ) INTO col_exists;
              IF NOT col_exists THEN
                ALTER TABLE users ADD COLUMN role VARCHAR(16) NOT NULL DEFAULT 'user';
                UPDATE users SET role = CASE WHEN is_admin THEN 'admin' ELSE 'developer' END;
              END IF;
            END $$;
        """))
        # Policy roles rollout. We seed four system roles below; `policy_role_id`
        # is added to users with an FK to policy_roles. Existing developers
        # are auto-attached to "full-access" so we don't lock anyone out.
        await conn.execute(text(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS policy_role_id VARCHAR(64) "
            "REFERENCES policy_roles(id) ON DELETE SET NULL"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_users_policy_role_id ON users(policy_role_id)"
        ))
        # Idempotent seed of system roles.
        await conn.execute(text("""
            INSERT INTO policy_roles (id, name, sections, is_system, created_at)
            VALUES
              ('full-access', 'Full access',
                '{"inference": true, "benchmark": true, "compute": true}'::jsonb,
                true, NOW()),
              ('inference-only', 'Inference only',
                '{"inference": true, "benchmark": false, "compute": false}'::jsonb,
                true, NOW()),
              ('benchmark-only', 'Benchmark only',
                '{"inference": false, "benchmark": true, "compute": false}'::jsonb,
                true, NOW()),
              ('compute-only', 'Compute only',
                '{"inference": false, "benchmark": false, "compute": true}'::jsonb,
                true, NOW())
            ON CONFLICT (id) DO NOTHING
        """))
        # Backfill: any developer/admin without an attached role gets full-access
        # so the migration doesn't strip access from existing users.
        await conn.execute(text("""
            UPDATE users SET policy_role_id = 'full-access'
            WHERE policy_role_id IS NULL AND role IN ('developer', 'admin')
        """))
        # Compute approval workflow: widen status column from 16 → 20 to fit
        # 'pending_approval', and add reject_reason for admin-supplied notes.
        await conn.execute(text(
            "ALTER TABLE compute_pods ALTER COLUMN status TYPE VARCHAR(20)"
        ))
        await conn.execute(text(
            "ALTER TABLE compute_pods ADD COLUMN IF NOT EXISTS reject_reason VARCHAR(1024)"
        ))
        # Per-pod RunPod-account selection. NULL = use gateway env key.
        await conn.execute(text(
            "ALTER TABLE compute_pods ADD COLUMN IF NOT EXISTS provider_id VARCHAR(64)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_compute_pods_provider_id ON compute_pods(provider_id)"
        ))
        # Persisted Jupyter URL for non-RunPod kinds (PI). NULL = derive from
        # runpod_pod_id at render time (RunPod proxy domain).
        await conn.execute(text(
            "ALTER TABLE compute_pods ADD COLUMN IF NOT EXISTS jupyter_url_override VARCHAR(512)"
        ))
        # Idle auto-terminate: window in seconds (0 = off) + last time the idle
        # monitor saw the pod busy (seeded to ready time).
        await conn.execute(text(
            "ALTER TABLE compute_pods ADD COLUMN IF NOT EXISTS idle_terminate_after_s INTEGER NOT NULL DEFAULT 0"
        ))
        await conn.execute(text(
            "ALTER TABLE compute_pods ADD COLUMN IF NOT EXISTS last_active_at TIMESTAMP WITH TIME ZONE"
        ))
        # GitHub SSO: column for linking platform accounts to GitHub user IDs.
        await conn.execute(text(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS github_id VARCHAR(64)"
        ))
        await conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_github_id ON users(github_id) WHERE github_id IS NOT NULL"
        ))
        # Benchmark cost tracking: captured at spawn by scraping benchmaq's
        # `Pod created: <id>` line and querying RunPod /pods/{id} for costPerHr.
        await conn.execute(text(
            "ALTER TABLE benchmarks ADD COLUMN IF NOT EXISTS cost_per_hr DOUBLE PRECISION"
        ))
        await conn.execute(text(
            "ALTER TABLE benchmarks ADD COLUMN IF NOT EXISTS runpod_pod_id VARCHAR(64)"
        ))
        # Per-benchmark provider selection: NULL = platform default cloud.
        await conn.execute(text(
            "ALTER TABLE benchmarks ADD COLUMN IF NOT EXISTS provider_id VARCHAR(64)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_benchmarks_provider_id ON benchmarks(provider_id)"
        ))
        # VM cleanup flag — only honoured when provider_id is set.
        await conn.execute(text(
            "ALTER TABLE benchmarks ADD COLUMN IF NOT EXISTS cleanup_model BOOLEAN NOT NULL DEFAULT TRUE"
        ))
        # Extra env applied to the benchmark run (exported on the VM / passed to
        # the RunPod pod) + CUDA_VISIBLE_DEVICES pinning. NULL = none.
        await conn.execute(text(
            "ALTER TABLE benchmarks ADD COLUMN IF NOT EXISTS env_vars JSON"
        ))
        await conn.execute(text(
            "ALTER TABLE benchmarks ADD COLUMN IF NOT EXISTS visible_devices VARCHAR(128)"
        ))
        # Per-benchmark storage backend for logs + results. NULL = env bucket.
        await conn.execute(text(
            "ALTER TABLE benchmarks ADD COLUMN IF NOT EXISTS storage_id VARCHAR(64)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_benchmarks_storage_id ON benchmarks(storage_id)"
        ))
        # Per-benchmark HF token: a global-secret key aliased to HF_TOKEN at launch.
        await conn.execute(text(
            "ALTER TABLE benchmarks ADD COLUMN IF NOT EXISTS hf_token_secret VARCHAR(128)"
        ))
        # Ingress endpoint API key: a global-secret key aliased to OPENAI_API_KEY.
        await conn.execute(text(
            "ALTER TABLE benchmarks ADD COLUMN IF NOT EXISTS api_key_secret VARCHAR(128)"
        ))
        # Manual GPU identity for runs the platform didn't provision (ingress/
        # Slurm-submitted via API — no pod block, usually no provider). Set at
        # create or via PATCH /benchmarks/{id}; wins over config-derived GPU.
        await conn.execute(text(
            "ALTER TABLE benchmarks ADD COLUMN IF NOT EXISTS gpu_type VARCHAR(64)"
        ))
        await conn.execute(text(
            "ALTER TABLE benchmarks ADD COLUMN IF NOT EXISTS gpu_count INTEGER"
        ))
        # Worker/node identity per inference request (history API): mirrored
        # from the worker's Redis result alongside status/output.
        await conn.execute(text(
            "ALTER TABLE requests ADD COLUMN IF NOT EXISTS worker_meta JSON"
        ))
        # Providers table is created by Base.metadata.create_all above; nothing
        # to migrate yet since it landed on a fresh schema.
        # Storage: `enabled` toggle landed after the table's first cut, so add
        # it in place for any DB that created `storage` without it.
        await conn.execute(text(
            "ALTER TABLE storage ADD COLUMN IF NOT EXISTS enabled BOOLEAN NOT NULL DEFAULT TRUE"
        ))
        # Dataset audio-zip → audio-column transform job tracking.
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS transform_status VARCHAR(16)"
        ))
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS transform_log VARCHAR(8192)"
        ))
        # Per-split transcription column overrides (HF splits with differing schemas).
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS split_fields JSON"
        ))
        # TTS-only speaker column mapping (consumed by the pack step).
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS speaker_field VARCHAR(128)"
        ))
        # Manually-excluded row indices (un-ticked in the row browser).
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS excluded_rows JSON"
        ))
        # Link a zip-backed source dataset to its materialised S3 audio dataset.
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS audio_dataset_id VARCHAR(64)"
        ))
        # Labeling-platform source (kind=label): base URL + project id + encrypted token.
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS label_base_url VARCHAR(512)"
        ))
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS label_project_id VARCHAR(128)"
        ))
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS label_token_enc VARCHAR(2048)"
        ))
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS label_status VARCHAR(16)"
        ))
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS label_token_secret VARCHAR(128)"
        ))
        await conn.execute(text(
            "ALTER TABLE datasets ADD COLUMN IF NOT EXISTS messages_field VARCHAR(128)"
        ))
        # Worker lifecycle timeline: the calendar/analytics query is always
        # "events for app X between t0 and t1", so a composite (app_id,
        # created_at) index serves it without scanning the whole table.
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_worker_events_app_created "
            "ON worker_events(app_id, created_at)"
        ))
        # GitOps: one ledger row per (repo, kind, manifest name). Tables are
        # created fresh by create_all above; the unique index is what the
        # reconciler relies on to upsert the ledger.
        await conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_gitops_resources_repo_kind_name "
            "ON gitops_resources(repo_id, kind, name)"
        ))

    # Queue-history indexes on the HIGH-VOLUME, now-indefinitely-retained request
    # tables — built CONCURRENTLY in autocommit (OUTSIDE the txn above) so they take
    # no ACCESS EXCLUSIVE lock: a large prod `requests`/`proxy_requests` stays
    # readable/writable while the index builds. Each runs in its own implicit txn
    # (CONCURRENTLY can't run inside one); `IF NOT EXISTS` makes it a no-op once
    # built, and a failure (e.g. a leftover INVALID index) is logged, not fatal.
    _conc_indexes = (
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_proxy_req_endpoint_created ON proxy_requests (endpoint_id, created_at DESC)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_proxy_req_endpoint_latency ON proxy_requests (endpoint_id, latency_ms)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_proxy_req_owner ON proxy_requests (owner_id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_proxy_req_upstream ON proxy_requests (endpoint_id, upstream)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_requests_app_created ON requests (app_id, created_at DESC)",
    )
    import logging as _logging
    # AUTOCOMMIT engine variant (AsyncEngine.execution_options is sync, returns an
    # engine) → each statement runs outside a txn, which CONCURRENTLY requires.
    _ac_engine = _engine.execution_options(isolation_level="AUTOCOMMIT")
    async with _ac_engine.connect() as conn:
        for _ix in _conc_indexes:
            try:
                await conn.execute(text(_ix))
            except Exception:
                _logging.getLogger("gateway.db").warning(
                    "concurrent index build skipped: %s", _ix.split(" ON ")[0], exc_info=True)


async def seed_admin_user() -> None:
    """If ADMIN_USERNAME + ADMIN_PASSWORD env are set, upsert that user with
    is_admin=true. ADMIN_EMAIL is optional and backfilled when set (never
    cleared). Idempotent: re-running is safe and doesn't overwrite the password
    if the user already exists."""
    import os as _os
    from .auth import hash_password
    username = _os.environ.get("ADMIN_USERNAME", "").strip()
    password = _os.environ.get("ADMIN_PASSWORD", "").strip()
    email = _os.environ.get("ADMIN_EMAIL", "").strip() or None
    if not username or not password:
        return
    async with session_factory()() as session:
        existing = await get_user_by_username(session, username)
        if existing is None:
            session.add(User(
                username=username,
                email=email,
                password_hash=hash_password(password),
                is_admin=True,
                role="admin",
            ))
            await session.commit()
        elif not existing.is_admin or existing.role != "admin" or (email and existing.email != email):
            existing.is_admin = True
            existing.role = "admin"
            if email:
                existing.email = email
            await session.commit()


async def shutdown_db() -> None:
    if _engine is not None:
        await _engine.dispose()


def session_factory() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        raise RuntimeError("db not initialized")
    return _sessionmaker


async def get_session() -> AsyncIterator[AsyncSession]:
    async with session_factory()() as session:
        yield session


async def list_all_apps(session: AsyncSession) -> list[App]:
    result = await session.execute(select(App))
    return list(result.scalars().all())


async def get_app(session: AsyncSession, app_id: str) -> Optional[App]:
    return await session.get(App, app_id)


async def get_user_by_username(session: AsyncSession, username: str) -> Optional[User]:
    result = await session.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def get_user_by_id(session: AsyncSession, user_id: int) -> Optional[User]:
    return await session.get(User, user_id)
