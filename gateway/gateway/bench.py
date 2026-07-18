"""Benchmark feature — approach B (SSH-orchestrated via llm-benchmaq).

Gateway shells out to `benchmaq runpod bench config.yaml` as a subprocess,
streams stdout/stderr to a redis list (capped) for live SSE replay, and on
exit syncs any result files into S3 + parses them for the Metrics tab.

Subprocess lives only in the gateway process. If gateway dies mid-run the
bench is orphaned (pod is alive on RunPod but nobody's collecting). On
startup we mark all `running` rows as `failed` with a clear message — the
user can re-submit or terminate the dangling pod from RunPod's dashboard.
"""
from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import boto3
import httpx
import yaml
from botocore.client import Config as BotoConfig
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    func,
    or_,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from dataclasses import dataclass

from . import audit
from . import crypto
from .auth import require_section
from .db import Base, Storage, User, get_session, session_factory

logger = logging.getLogger("gateway.bench")

LOG_LIST_CAP = 5000          # max lines retained in redis per bench
LOG_LIST_TTL_S = 12_960_000  # ~5 months after benchmark completes


# ---------- DB model ----------------------------------------------------


class BenchmarkTemplate(Base):
    __tablename__ = "benchmark_templates"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    config_yaml: Mapped[str] = mapped_column(String(65535))
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class BenchmarkShare(Base):
    """A public, no-auth comparison link: a random token → an ordered list of
    benchmark ids. Minted by an authed owner; resolved by anyone with the token
    (only explicitly-shared comparisons are world-readable)."""
    __tablename__ = "benchmark_shares"
    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    bench_ids: Mapped[list] = mapped_column(JSON)
    # Optional markdown summary/notes shown above the comparison (e.g. the report
    # summary + extra text). Captured when the link is minted; shown on the page.
    notes: Mapped[str] = mapped_column(String(1048576), default="")
    # Frozen accuracy-run→speed-run pairing (accId→speedId) captured when the link
    # is minted, so the public IQ-vs-speed chart reproduces exactly what the owner
    # saw (otherwise it re-auto-pairs, which can differ).
    pairing: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class Benchmark(Base):
    __tablename__ = "benchmarks"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    config_yaml: Mapped[str] = mapped_column(String(65535))
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    s3_prefix: Mapped[str] = mapped_column(String(255))
    exit_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_text: Mapped[Optional[str]] = mapped_column(String(4096), nullable=True)
    result_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # Public runs surface (read-only) in every user's benchmark list. Only the
    # owner (or an admin) can flip this or mutate the run.
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # RunPod $/hour quote at spawn time, captured by scraping the pod_id out of
    # benchmaq stdout then querying RunPod /pods/{id}. NULL while the pod isn't
    # up yet, and stays at the original quoted rate for the life of the run.
    cost_per_hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    runpod_pod_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # User-selected cloud provider. NULL = use platform default (RunPod via env).
    # FK omitted to keep this column nullable without cascade headaches; we
    # validate ownership at create time.
    provider_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    # User-selected storage backend (Storage row, kind=s3) for logs + result
    # files. NULL = fall back to the BENCHMARK_S3_BUCKET env. The chosen
    # storage's bucket/prefix/region/endpoint/creds are baked into `s3_prefix`
    # and used by the S3 helpers via the resolved target.
    storage_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    # VM runs only: SSH back in after benchmaq exits to rm -rf the model's
    # local_dir + HF hub cache. Default true so users don't fill the VM disk.
    cleanup_model: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
    # Extra env exported for the run (VM: exported in the remote script + mkdir'd
    # for path values; RunPod: passed to the pod via runpod.env). NULL = none.
    env_vars: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # CUDA_VISIBLE_DEVICES pin, e.g. "0,1,2,3". NULL/empty = all GPUs.
    visible_devices: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    # A global-secret key (admin Secrets) whose value is injected as HF_TOKEN for
    # this run — resolved fresh at launch so rotating the secret takes effect.
    # NULL = none (a pasted token lands in env_vars["HF_TOKEN"] instead).
    hf_token_secret: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    # Ingress only: a global-secret key whose value is injected as OPENAI_API_KEY
    # at launch (the in-gateway ingress client sends it as Authorization: Bearer).
    # NULL = none (a pasted key lands in env_vars["OPENAI_API_KEY"] instead).
    api_key_secret: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    # Manually-set GPU identity, for runs the platform didn't provision (ingress/
    # Slurm-submitted): no runpod.pod block and usually no provider, so there's
    # nothing to derive hardware from. Set at create or via PATCH; wins over the
    # config-derived value everywhere GPU is read (record/stats/aggregate/export).
    # NULL = derive from config_yaml / provider as before.
    gpu_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    gpu_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


# ---------- S3 ----------------------------------------------------------


@dataclass
class S3Target:
    """Resolved destination for a benchmark's S3 I/O. Built either from the
    gateway env (the default) or from a user-selected Storage row."""
    bucket: str
    region: str
    endpoint: Optional[str]
    access_key: Optional[str]
    secret_key: Optional[str]
    prefix_root: str  # always ends with "/"


def _env_s3_target() -> S3Target:
    """The historical env-driven destination (BENCHMARK_S3_BUCKET et al.)."""
    prefix = os.environ.get("BENCHMARK_S3_PREFIX", "benchmarks/").strip().lstrip("/")
    if not prefix.endswith("/"):
        prefix += "/"
    return S3Target(
        bucket=os.environ.get("BENCHMARK_S3_BUCKET", "").strip(),
        region=os.environ.get("AWS_REGION", "ap-southeast-5"),
        endpoint=None,
        access_key=os.environ.get("AWS_ACCESS_KEY_ID") or None,
        secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
        prefix_root=prefix,
    )


def _resolve_s3_creds(cfg: dict) -> tuple[Optional[str], Optional[str]]:
    """(access_key, secret_key) for a kind=s3 storage config. Precedence per key:
    a referenced global secret (`access_key_id_secret` / `secret_access_key_secret`
    — a Secrets key name, resolved from the in-process cache) > the encrypted
    literal blob (`credentials_enc`) > the AWS_* env vars. Synchronous: the global
    secret comes from global_env_api's cache (no DB session needed here)."""
    from .global_env_api import global_secret_sync
    access_key = global_secret_sync(cfg.get("access_key_id_secret"))
    secret_key = global_secret_sync(cfg.get("secret_access_key_secret"))
    if (access_key is None or secret_key is None) and cfg.get("credentials_enc"):
        try:
            creds = json.loads(crypto.decrypt(cfg["credentials_enc"]))
        except Exception:  # noqa: BLE001 — undecryptable blob → fall through to env
            creds = {}
        if access_key is None:
            access_key = creds.get("accessKeyId")
        if secret_key is None:
            secret_key = creds.get("secretAccessKey")
    if access_key is None:
        access_key = os.environ.get("AWS_ACCESS_KEY_ID") or None
    if secret_key is None:
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or None
    return access_key, secret_key


def _target_from_storage_row(row: Optional[Storage]) -> S3Target:
    """Build a target from a kind=s3 Storage row, decrypting its credentials.
    Falls back to env for any field the row leaves blank. None → env target."""
    if row is None:
        return _env_s3_target()
    cfg = row.config or {}
    access_key, secret_key = _resolve_s3_creds(cfg)
    prefix = (cfg.get("prefix") or "").strip().strip("/")
    prefix_root = f"{prefix}/benchmarks/" if prefix else "benchmarks/"
    return S3Target(
        bucket=(cfg.get("bucket") or "").strip(),
        region=(cfg.get("region") or os.environ.get("AWS_REGION", "ap-southeast-5")),
        endpoint=(cfg.get("endpoint") or None),
        access_key=access_key,
        secret_key=secret_key,
        prefix_root=prefix_root,
    )


async def _bench_s3_target(storage_id: Optional[str]) -> S3Target:
    """Resolve the target for a benchmark by its storage_id. Opens its own
    short-lived session; storage_id=None (or a missing row) → env target."""
    if not storage_id:
        return _env_s3_target()
    async with session_factory()() as s:
        row = await s.get(Storage, storage_id)
    return _target_from_storage_row(row)


def _s3_client(target: S3Target, max_pool_connections: Optional[int] = None):
    if not target.bucket:
        raise RuntimeError(
            "no S3 bucket configured — set BENCHMARK_S3_BUCKET or select a storage"
        )
    cfg_kwargs: dict = {
        "signature_version": "s3v4",
        "s3": {"addressing_style": "path" if target.endpoint else "virtual"},
    }
    # Bump the connection pool for concurrent uploaders (botocore defaults to 10,
    # which would serialise a wider thread pool and spam pool-full warnings).
    if max_pool_connections:
        cfg_kwargs["max_pool_connections"] = max_pool_connections
    kwargs: dict = {
        "region_name": target.region,
        # Pin to the regional endpoint. Default `s3.amazonaws.com` redirects to
        # the bucket's region, but presigned URLs signed with a non-default
        # region get a 400 on the global host before the redirect can happen.
        # A custom endpoint (R2 / MinIO) overrides this and wants path-style.
        "endpoint_url": target.endpoint or f"https://s3.{target.region}.amazonaws.com",
        "config": BotoConfig(**cfg_kwargs),
    }
    if target.access_key and target.secret_key:
        kwargs["aws_access_key_id"] = target.access_key
        kwargs["aws_secret_access_key"] = target.secret_key
    return boto3.client("s3", **kwargs)


def s3_put_text(key: str, body: str, target: Optional[S3Target] = None) -> None:
    t = target or _env_s3_target()
    _s3_client(t).put_object(Bucket=t.bucket, Key=key, Body=body.encode("utf-8"))


def s3_put_file(key: str, path: str, target: Optional[S3Target] = None) -> None:
    t = target or _env_s3_target()
    with open(path, "rb") as f:
        _s3_client(t).put_object(Bucket=t.bucket, Key=key, Body=f.read())


def s3_put_files(
    items: list[tuple[str, str]],
    target: Optional[S3Target] = None,
    *,
    max_workers: int = 16,
    on_done: Optional["Callable[[int], None]"] = None,
) -> None:
    """Upload many (key, local_path) files to S3 concurrently, reusing ONE client.
    The sequential path paid a fresh-client build + a serial round-trip per file,
    which crawls on 10k+ clips. Calls on_done(n_completed) as each finishes;
    raises the first upload error (parity with the per-file path)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not items:
        return
    t = target or _env_s3_target()
    workers = max(1, min(max_workers, len(items)))
    cli = _s3_client(t, max_pool_connections=workers)

    def _put(key_path: tuple[str, str]) -> None:
        key, path = key_path
        with open(path, "rb") as f:
            cli.put_object(Bucket=t.bucket, Key=key, Body=f.read())

    err: Optional[BaseException] = None
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_put, it) for it in items]):
            done += 1
            exc = fut.exception()
            if exc is not None and err is None:
                err = exc
            if on_done is not None:
                try:
                    on_done(done)
                except Exception:  # noqa: BLE001 — progress is best-effort
                    pass
    if err is not None:
        raise err


def s3_copy_many(
    items: list[tuple[str, str, str]],
    target: Optional[S3Target] = None,
    *,
    max_workers: int = 16,
    on_done: Optional["Callable[[int], None]"] = None,
) -> None:
    """Server-side copy many (dest_key, src_bucket, src_key) objects into `target`'s
    bucket — the bytes move S3→S3 (CopyObject), never through the gateway. Reuses ONE
    dest client. Requires that client's creds to read the source bucket (same account,
    or a granted cross-account policy). Calls on_done(n_completed) as each finishes;
    raises the first error (parity with s3_put_files)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not items:
        return
    t = target or _env_s3_target()
    workers = max(1, min(max_workers, len(items)))
    cli = _s3_client(t, max_pool_connections=workers)

    def _cp(item: tuple[str, str, str]) -> None:
        dest_key, src_bucket, src_key = item
        cli.copy_object(
            Bucket=t.bucket, Key=dest_key,
            CopySource={"Bucket": src_bucket, "Key": src_key},
        )

    err: Optional[BaseException] = None
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_cp, it) for it in items]):
            done += 1
            exc = fut.exception()
            if exc is not None and err is None:
                err = exc
            if on_done is not None:
                try:
                    on_done(done)
                except Exception:  # noqa: BLE001 — progress is best-effort
                    pass
    if err is not None:
        raise err


def s3_presign_many(
    keys: list[str], expires: int = 3600, target: Optional[S3Target] = None
) -> dict[str, str]:
    """Presign GET URLs for many keys reusing one client (avoids a per-key client
    build — presigning itself is local, no network). Returns {key: url}."""
    t = target or _env_s3_target()
    cli = _s3_client(t)
    return {
        k: cli.generate_presigned_url(
            "get_object", Params={"Bucket": t.bucket, "Key": k}, ExpiresIn=expires
        )
        for k in dict.fromkeys(keys)  # dedupe, keep order
    }


def s3_get_text(key: str, target: Optional[S3Target] = None) -> Optional[str]:
    """Read an S3 object as utf-8 text. Returns None if the key is missing."""
    t = target or _env_s3_target()
    try:
        obj = _s3_client(t).get_object(Bucket=t.bucket, Key=key)
        return obj["Body"].read().decode("utf-8", "replace")
    except Exception:
        return None


def s3_get_bytes(key: str, target: Optional[S3Target] = None) -> Optional[bytes]:
    """Read an S3 object's raw bytes. Returns None if the key is missing."""
    t = target or _env_s3_target()
    try:
        return _s3_client(t).get_object(Bucket=t.bucket, Key=key)["Body"].read()
    except Exception:
        return None


def s3_put_bytes(key: str, data: bytes, target: Optional[S3Target] = None) -> None:
    """Write raw bytes to S3."""
    t = target or _env_s3_target()
    _s3_client(t).put_object(Bucket=t.bucket, Key=key, Body=data)


def s3_list(prefix: str, target: Optional[S3Target] = None) -> list[dict]:
    t = target or _env_s3_target()
    cli = _s3_client(t)
    out: list[dict] = []
    token = None
    while True:
        kwargs = {"Bucket": t.bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        r = cli.list_objects_v2(**kwargs)
        for obj in r.get("Contents", []):
            out.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "modified": obj["LastModified"].isoformat() if obj.get("LastModified") else "",
            })
        if not r.get("IsTruncated"):
            break
        token = r.get("NextContinuationToken")
    return out


def s3_delete_prefix(prefix: str, target: Optional[S3Target] = None) -> int:
    """Delete every object under `prefix` (batched, 1000/req). Returns the count
    deleted. Refuses an empty/root prefix — that would wipe the whole bucket."""
    if not (prefix and prefix.strip("/")):
        raise ValueError("refusing to delete an empty/root S3 prefix")
    t = target or _env_s3_target()
    cli = _s3_client(t)
    keys = [o["key"] for o in s3_list(prefix, t)]
    for i in range(0, len(keys), 1000):
        batch = [{"Key": k} for k in keys[i : i + 1000]]
        cli.delete_objects(Bucket=t.bucket, Delete={"Objects": batch})
    return len(keys)


def s3_presign_get(key: str, expires: int = 3600, target: Optional[S3Target] = None) -> str:
    t = target or _env_s3_target()
    return _s3_client(t).generate_presigned_url(
        "get_object", Params={"Bucket": t.bucket, "Key": key}, ExpiresIn=expires
    )


# ---------- Helpers -----------------------------------------------------


def benchmark_s3_prefix(bench_id: str, target: Optional[S3Target] = None) -> str:
    t = target or _env_s3_target()
    return f"{t.prefix_root}{bench_id}/"


def _gen_id() -> str:
    import uuid
    return f"bench-{uuid.uuid4().hex[:8]}"


def _work_dir(bench_id: str) -> Path:
    base = Path(os.environ.get("BENCHMARK_WORK_DIR", "/tmp/sgpu-bench"))
    p = base / bench_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ssh_key_path() -> str:
    p = os.environ.get("BENCHMARK_SSH_KEY_PATH", "").strip()
    if not p:
        # benchmaq's own default
        p = str(Path.home() / ".runpod" / "ssh" / "RunPod-Key-Go")
    return os.path.expanduser(p)


def _merge_run_env(
    env_vars: Optional[dict],
    visible_devices: Optional[str],
    global_env: Optional[dict] = None,
) -> dict:
    """Build the run env. Precedence (low → high): legacy gateway HF_TOKEN env <
    admin global env/secrets < per-benchmark env vars. benchmaq forwards this into
    the pod (runpod.env) / VM remote env, where `hf download` reads HF_TOKEN — so
    gated models work without pasting a token per benchmark."""
    env: dict[str, str] = {}
    gw_tok = os.environ.get("HF_TOKEN", "").strip()  # legacy; superseded by the global-env UI
    if gw_tok:
        env["HF_TOKEN"] = gw_tok
    for k, v in (global_env or {}).items():
        env[str(k)] = str(v)
    for k, v in (env_vars or {}).items():
        env[str(k)] = str(v)
    vd = (visible_devices or "").strip()
    if vd:
        env["CUDA_VISIBLE_DEVICES"] = vd
    return env


def _split_leading_env(args: str) -> tuple[dict[str, str], list[str]]:
    """Split a leading run of `NAME=VALUE` tokens (install-time env, e.g.
    `VLLM_USE_PRECOMPILED=1`) off the front of a `uv pip install` arg string.

    Returns (env, remaining_tokens). Stops at the first token that isn't a bare
    `NAME=VALUE` assignment — a flag (`--torch-backend=auto`), a VCS spec
    (`git+https://…`), or a pin (`vllm==0.23.0`, rejected via the `(?!=)`
    look-ahead). Used so a custom-fork vLLM spec installs with its env applied,
    not passed to pip as a bogus requirement."""
    import shlex
    toks = shlex.split(args or "")
    env: dict[str, str] = {}
    i = 0
    for t in toks:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(?!=)(.*)$", t)
        if not m:
            break
        env[m.group(1)] = m.group(2)
        i += 1
    return env, toks[i:]


def _resolve_config(
    raw_yaml: str,
    vm_target: Optional[dict] = None,
    env_vars: Optional[dict] = None,
    visible_devices: Optional[str] = None,
    runpod_key_path: Optional[str] = None,
    bench_id: Optional[str] = None,
    global_env: Optional[dict] = None,
    ingress: bool = False,
) -> str:
    """Inject runtime values (SSH key path, RunPod API key) into the user's YAML.

    Users paste a config that may have `ssh_private_key: "path/to/your/private/key"`
    or empty `runpod_api_key: ""`. We replace those with real values from env so
    they don't have to know about the runpodctl-managed key location.

    When `vm_target` is provided (bare-metal mode), we rewrite the `remote:`
    block to point at the user-registered VM and drop the `runpod:` block so
    benchmaq's vllm runner doesn't accidentally pick it up. Expected shape:
        {"host": str, "port": int, "user": str, "key_filename": str}
    """
    cfg = yaml.safe_load(raw_yaml) or {}
    if not isinstance(cfg, dict):
        return raw_yaml

    run_env = _merge_run_env(env_vars, visible_devices, global_env)

    if ingress:
        # Ingress: no machine is provisioned. The in-gateway client
        # (gateway.bench_ingress) only needs base_url + benchmark; drop the
        # runpod/remote blocks so a leftover provider/pod spec can't confuse it.
        # base_url already rides on the config (top-level or per-item) from the
        # form. run_env (e.g. HF_TOKEN) reaches the client via the process env.
        cfg.pop("runpod", None)
        cfg.pop("remote", None)
        return yaml.safe_dump(cfg, sort_keys=False)

    if vm_target is None:
        # Prefer the per-run ephemeral key (minted in run_benchmark); fall back to
        # the legacy static BENCHMARK_SSH_KEY_PATH only if no ephemeral key was
        # passed (keeps any old explicit-path configs working).
        rp_key = runpod_key_path or _ssh_key_path()
        rp = cfg.setdefault("runpod", {})
        if not rp.get("ssh_private_key") or "path/to/your" in str(rp.get("ssh_private_key")):
            rp["ssh_private_key"] = rp_key
        if not rp.get("runpod_api_key"):
            rp["runpod_api_key"] = os.environ.get("RUNPOD_API_KEY", "")
        # Custom-fork vLLM: benchmaq's RunPod runner installs `remote.dependencies`
        # via `uv pip install` (no `vllm_install_args` hook like the SSH path has).
        # So translate a `remote.uv.vllm_install_args` spec into that mechanism:
        # leading NAME=VALUE tokens (e.g. VLLM_USE_PRECOMPILED=1) become pod env so
        # they're set when uv runs; the rest (git+… spec, --torch-backend=auto, …)
        # replaces the `vllm==` pin in dependencies. huggingface_hub + hf_transfer
        # are re-appended so the model still downloads.
        fork_env: dict[str, str] = {}
        _uv = cfg.get("remote", {}).get("uv", {}) if isinstance(cfg.get("remote"), dict) else {}
        _install_args = str(_uv.get("vllm_install_args") or "").strip()
        if _install_args:
            fork_env, _fork_tokens = _split_leading_env(_install_args)
            _rem = cfg.setdefault("remote", {})
            if _fork_tokens:
                # sentencepiece: fork precompiled wheels skip it, but gemma/llama
                # tokenizers need it (else "Couldn't instantiate the backend tokenizer").
                _rem["dependencies"] = [*_fork_tokens, "sentencepiece", "huggingface_hub", "hf_transfer"]
            _rem.get("uv", {}).pop("vllm_install_args", None)  # consumed; not read by the runpod runner
            logger.info("bench: RunPod fork install — deps=%s env=%s", _fork_tokens, list(fork_env))
        # RunPod path: benchmaq forwards `runpod.env` into the pod (--env) — and the
        # pod's boot script has it in scope when it installs the injected SSH key
        # into `$HOME/.ssh/authorized_keys`. HOME is therefore poison here: a HOME
        # override (e.g. /share/home) lands the key off /root, sshd (which reads
        # /root/.ssh) never sees it, and the pod stays "SSH not ready" until the
        # wait ceiling — burning credits, never running. Strip HOME from the *pod*
        # env; runtime cache redirects (XDG_CACHE_HOME / HF_HOME / VLLM_CACHE_ROOT /
        # …) are read by the benchmark process, not the boot script, so they stay.
        # The VM path (below) keeps HOME — its sshd auth is already established.
        pod_env = {**(rp.get("env") or {}), **run_env, **fork_env}
        dropped_home = pod_env.pop("HOME", None)
        if dropped_home is not None:
            logger.warning(
                "bench: dropped HOME=%r from RunPod pod env — it breaks the pod's "
                "boot-time SSH-key install (key would land off /root); caches still "
                "honour XDG_CACHE_HOME/HF_HOME/etc.",
                dropped_home,
            )
        if pod_env:
            rp["env"] = pod_env

        # benchmaq's RunPod (pyremote) path runs `hf download` inside the pod over a
        # pyremote exec session that does NOT inherit the pod `--env` (the same reason
        # the fork-install env has to ride SGPU_PIP_ENV rather than runpod.env). So an
        # HF_TOKEN placed only in runpod.env never reaches the download — it goes out
        # "unauthenticated" (throttled, and fails outright on gated repos). benchmaq
        # DOES read `benchmark[].model.hf_token` from the config, which pyremote ships
        # into the pod via the serialized-closure config (not env), so fold the resolved
        # token in there. A token the user pinned in their own YAML wins; benchmaq never
        # prints the model block, so this doesn't leak into the bench logs.
        hf_tok = (run_env or {}).get("HF_TOKEN")
        if hf_tok:
            for item in cfg.get("benchmark") or []:
                if not isinstance(item, dict):
                    continue
                model = item.get("model")
                if isinstance(model, dict) and not str(model.get("hf_token") or "").strip():
                    model["hf_token"] = hf_tok

        rem = cfg.setdefault("remote", {})
        if not rem.get("key_filename") or "path/to/your" in str(rem.get("key_filename")):
            rem["key_filename"] = rp_key
    else:
        # Bare-metal VM: drop runpod block (irrelevant + would confuse benchmaq)
        # and rewrite remote to use benchmaq's `backend: ssh` runner — a
        # paramiko-based path with proper live-streaming, idempotent
        # uv+benchmaq install on the VM, and zero dependency on pyremote.
        # benchmaq[vllm] pulls vLLM transitively via the extra; we pin the
        # vllm version explicitly when the user picked one in the form.
        cfg.pop("runpod", None)
        rem = cfg.setdefault("remote", {})
        rem["backend"] = "ssh"
        rem["host"] = vm_target["host"]
        rem["port"] = int(vm_target.get("port") or 22)
        rem["username"] = vm_target.get("user", "root")
        rem["key_filename"] = vm_target["key_filename"]
        # Consumed by pyremote_shim's run_remote_ssh: exported (and path values
        # mkdir'd) on the VM before install + benchmark. Stripped from the
        # config uploaded to the VM, so it only shapes the remote shell env.
        if run_env:
            rem["env"] = run_env
        uv = rem.setdefault("uv", {})
        uv.setdefault("path", "~/.bench-venv")
        uv.setdefault("python_version", "3.11")
        uv.setdefault(
            "benchmaq_ref",
            "git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq.git@main",
        )
        # If the form rendered a vLLM pin under remote.dependencies (legacy
        # path), surface it as uv.vllm_version so the new ssh backend picks
        # it up. Otherwise leave unset = latest.
        if "vllm_version" not in uv:
            for dep in (rem.get("dependencies") or []):
                if isinstance(dep, str) and dep.startswith("vllm==") and len(dep) > 6:
                    uv["vllm_version"] = dep.split("==", 1)[1].strip()
                    break
        # The new backend installs benchmaq[vllm] + vllm itself; the legacy
        # `dependencies` field is unused here.
        rem.pop("dependencies", None)

        # `/workspace/...` is RunPod's per-pod mount and doesn't exist on
        # bare-metal VMs (where the SSH user is typically `ubuntu` and only
        # has write access under $HOME). Rewrite any model.local_dir or
        # results.result_dir that starts with `/workspace/` to live under
        # the user's home instead.
        items = cfg.get("benchmark") or []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                model = item.get("model")
                if isinstance(model, dict):
                    ld = str(model.get("local_dir") or "")
                    if ld.startswith("/workspace/"):
                        model["local_dir"] = "~/" + ld[len("/workspace/"):]
                results = item.get("results")
                if not isinstance(results, dict):
                    results = {}
                    item["results"] = results
                rd = str(results.get("result_dir") or "")
                if rd.startswith("/workspace/"):
                    results["result_dir"] = "~/" + rd[len("/workspace/"):]
                # Isolate this run's output in a per-bench dir. benchmaq's default
                # is the *relative* `./benchmark_results`, which on a shared VM
                # resolves under the SSH user's home alongside every prior run's
                # files — `_download_results` then `ls`es them all back, polluting
                # the aggregate and (since the gateway keeps the first .json it
                # finds) storing some unrelated run's metrics on this row. Only
                # inject when the user didn't pin a dir themselves.
                if bench_id and not results.get("result_dir"):
                    results["result_dir"] = f"/tmp/sgpu-bench-results/{bench_id}"

    return yaml.safe_dump(cfg, sort_keys=False)


_REDACTED = "***redacted***"


def _redact_config_for_display(resolved_yaml: str) -> str:
    """Strip secrets from a RESOLVED config before it's persisted to S3 and
    served through the files API.

    `_resolve_config` injects the platform `RUNPOD_API_KEY`, the resolved
    `HF_TOKEN`, and every admin global env/secret into the config (under
    `runpod.runpod_api_key`, `runpod.env` / `remote.env`, and
    `benchmark[].model.hf_token`). The run owner — and, for a public benchmark,
    every logged-in user — can download that file, so it must never carry live
    credentials. Redact those fields (values only; key names are harmless) and
    return YAML. On any parse failure we return a placeholder rather than risk
    emitting the un-redacted text."""
    try:
        cfg = yaml.safe_load(resolved_yaml) or {}
    except Exception:  # noqa: BLE001 — never leak the raw (secret-bearing) text
        return "# config.yaml withheld — could not parse for secret redaction\n"
    if not isinstance(cfg, dict):
        return "# config.yaml withheld — unexpected structure\n"

    def _redact_env(d: dict) -> None:
        env = d.get("env")
        if isinstance(env, dict):
            d["env"] = {k: _REDACTED for k in env}

    for section in ("runpod", "remote"):
        blk = cfg.get(section)
        if isinstance(blk, dict):
            if blk.get("runpod_api_key"):
                blk["runpod_api_key"] = _REDACTED
            _redact_env(blk)
    _redact_env(cfg)  # any top-level env
    for item in cfg.get("benchmark") or []:
        if isinstance(item, dict):
            model = item.get("model")
            if isinstance(model, dict) and model.get("hf_token"):
                model["hf_token"] = _REDACTED
    return yaml.safe_dump(cfg, sort_keys=False)


def _pick_engine_subcommand(raw_yaml: str) -> list[str]:
    """Read engine from the first benchmark item; default to vllm.

    benchmaq separates engines into top-level subcommands (`vllm bench`,
    `sglang bench`) and only those honour the `remote:` block. The `runpod`
    subcommand is only for the spawn-a-pod-then-bench flow we use for the
    cloud target.
    """
    try:
        cfg = yaml.safe_load(raw_yaml) or {}
        items = cfg.get("benchmark") or []
        if items and isinstance(items, list):
            engine = str(items[0].get("engine") or "vllm").lower()
            if engine == "sglang":
                return ["sglang", "bench"]
    except Exception:  # noqa: BLE001 — unparseable YAML falls back to the vllm default
        logger.debug("bench: engine sniff failed, defaulting to vllm", exc_info=True)
    return ["vllm", "bench"]


def _ingress_base_url(raw_yaml: str) -> Optional[str]:
    """Return the base_url when this config is an *ingress* run, else None.

    Ingress = bench an already-served, ingressed vLLM with no machine to
    provision. The signal is a `base_url` on the config — top-level or on a
    benchmark item — which the web form emits (and only emits) for ingress mode.
    The caller additionally requires no machine provider to be selected, so a
    stray base_url on a runpod/vm config never silently skips provisioning.
    """
    try:
        cfg = yaml.safe_load(raw_yaml) or {}
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    top = cfg.get("base_url")
    if isinstance(top, str) and top.strip():
        return top.strip()
    for item in cfg.get("benchmark") or []:
        if isinstance(item, dict):
            bu = item.get("base_url")
            if isinstance(bu, str) and bu.strip():
                return bu.strip()
    return None


async def _materialise_vm_key(work_dir: Path, provider_id: str) -> dict:
    """Look up the provider, decrypt its private key, write to `work/vm_key`
    with 0600, and return the dict `_resolve_config` expects as `vm_target`.

    Raises if the provider is missing or the key can't be decrypted.
    """
    from . import crypto
    from .db import Provider
    async with session_factory()() as s:
        prov = await s.get(Provider, provider_id)
    if prov is None:
        raise RuntimeError(f"provider {provider_id} not found")
    if prov.kind != "vm":
        raise RuntimeError(f"provider {provider_id} is kind={prov.kind}, expected vm")
    cfg = prov.config or {}
    enc = cfg.get("private_key_enc")
    if not enc:
        raise RuntimeError(f"provider {provider_id} has no stored key")
    pk_text = crypto.decrypt(enc)
    key_path = work_dir / "vm_key"
    key_path.write_text(pk_text + ("\n" if not pk_text.endswith("\n") else ""))
    os.chmod(key_path, 0o600)
    return {
        "host": cfg.get("host", ""),
        "port": int(cfg.get("port") or 22),
        "user": cfg.get("user", "root"),
        "key_filename": str(key_path),
    }


def _gen_ephemeral_runpod_key(work_dir: Path, bench_id: str) -> str:
    """Mint a throwaway SSH keypair for a single RunPod run and return the
    private-key path.

    benchmaq's RunPod runner reads `<priv>.pub` to inject PUBLIC_KEY into the pod
    (so the pod's authorized_keys gets our key at boot) and SSHes in with `<priv>`.
    Generating the pair per-run means the gateway needs no pre-provisioned key on
    disk (the old BENCHMARK_SSH_KEY_PATH footgun) and works identically in local
    dev and prod. Both files live under the per-run work dir, are excluded from S3
    upload (see `_NO_UPLOAD`), and are deleted once benchmaq exits.
    """
    import paramiko
    key = paramiko.RSAKey.generate(2048)
    priv = work_dir / "rp_key"
    key.write_private_key_file(str(priv))
    os.chmod(priv, 0o600)
    (work_dir / "rp_key.pub").write_text(
        f"{key.get_name()} {key.get_base64()} sgpu-bench-{bench_id}\n"
    )
    return str(priv)


def _hf_cache_dir(repo_id: str) -> str:
    """HuggingFace's on-disk cache layout: `~/.cache/huggingface/hub/models--<org>--<name>`.
    Slashes in the repo id become double-dashes. Used to clean up after a VM run."""
    sanitised = repo_id.replace("/", "--")
    return f"~/.cache/huggingface/hub/models--{sanitised}"


def _ssh_cleanup_paths_sync(vm_target: dict, paths: list[str]) -> tuple[bool, str]:
    """Open SSH and `rm -rf` each path. Returns (ok, message)."""
    import paramiko
    try:
        pkey = paramiko.PKey.from_path(vm_target["key_filename"])
    except Exception:
        # Older paramiko / non-standard key — fall back to type-probing.
        from .vm_probe import _load_pkey
        with open(vm_target["key_filename"], "r") as f:
            pkey = _load_pkey(f.read())
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=vm_target["host"],
            port=int(vm_target["port"]),
            username=vm_target["user"],
            pkey=pkey,
            timeout=15,
            banner_timeout=15,
            auth_timeout=15,
            look_for_keys=False,
            allow_agent=False,
        )
    except Exception as e:
        return False, f"SSH connect failed: {e}"

    try:
        # Quote each path with single quotes; reject any with embedded quotes
        # (we generated them ourselves, but belt-and-braces).
        safe = [p for p in paths if "'" not in p]
        if not safe:
            return False, "no safe paths to clean"
        # `rm -rf` is fine for missing paths; we use bash -lc so ~ expands.
        # Quote the whole inner command as ONE arg — inside `bash -lc "..."` the
        # single quotes below lose their power and $()/backticks in a path would
        # still execute; shlex.quote closes that.
        import shlex as _shlex
        cmd = "; ".join(f"rm -rf '{p}'" for p in safe)
        full = "bash -lc " + _shlex.quote(cmd)
        stdin, stdout, stderr = client.exec_command(full, timeout=60)
        rc = stdout.channel.recv_exit_status()
        err = stderr.read().decode(errors="replace").strip()
        if rc != 0:
            return False, f"rm exited {rc}: {err[:200]}"
        return True, f"removed {len(safe)} path{'s' if len(safe) != 1 else ''}"
    finally:
        try:
            client.close()
        except Exception:
            pass


def _ssh_kill_bench_procs_sync(vm_target: dict) -> tuple[bool, str]:
    """SSH in and pkill any benchmaq/huggingface-cli/vllm processes running
    under the bench venv. Best-effort — used by the terminate endpoint.
    Pattern matches `.benchmark-venv/bin/python` so we don't touch unrelated
    python processes the user might be running on the VM."""
    import paramiko
    try:
        pkey = paramiko.PKey.from_path(vm_target["key_filename"])
    except Exception:
        from .vm_probe import _load_pkey
        with open(vm_target["key_filename"], "r") as f:
            pkey = _load_pkey(f.read())
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=vm_target["host"],
            port=int(vm_target["port"]),
            username=vm_target["user"],
            pkey=pkey,
            timeout=15,
            banner_timeout=15,
            auth_timeout=15,
            look_for_keys=False,
            allow_agent=False,
        )
    except Exception as e:
        return False, f"SSH connect failed: {e}"

    try:
        cmd = (
            "pkill -9 -f '.benchmark-venv/bin/python' 2>/dev/null; "
            "pkill -9 -f 'huggingface-cli' 2>/dev/null; "
            "pkill -9 -f 'benchmaq' 2>/dev/null; true"
        )
        _, stdout, _ = client.exec_command(f"bash -lc \"{cmd}\"", timeout=30)
        stdout.channel.recv_exit_status()
        return True, "killed remote bench processes"
    finally:
        try:
            client.close()
        except Exception:
            pass


async def _cleanup_vm_model(
    redis,
    bench_id: str,
    vm_target: dict,
    raw_yaml: str,
) -> None:
    """After a bare-metal run ends, remove the model from the VM so the disk
    doesn't fill up with stale downloads. Targets both the user's `local_dir`
    (if set) and the standard HF hub cache. Best-effort — failures log but
    never bubble up since the benchmark itself already finished."""
    try:
        cfg = yaml.safe_load(raw_yaml) or {}
        items = cfg.get("benchmark") or []
        first = items[0] if items and isinstance(items, list) else {}
        model = first.get("model") or {}
        repo_id = str(model.get("repo_id") or "").strip()
        local_dir = str(model.get("local_dir") or "").strip()
    except Exception as e:
        await _push_log(redis, bench_id, f"[gateway] vm cleanup: could not parse YAML: {e}")
        return

    paths: list[str] = []
    if local_dir:
        paths.append(local_dir)
    if repo_id:
        paths.append(_hf_cache_dir(repo_id))
    if not paths:
        return

    await _push_log(redis, bench_id, f"[gateway] vm cleanup: removing {', '.join(paths)}")
    try:
        ok, msg = await asyncio.to_thread(_ssh_cleanup_paths_sync, vm_target, paths)
    except Exception as e:
        await _push_log(redis, bench_id, f"[gateway] vm cleanup failed: {e}")
        return
    level = "info" if ok else "warning"
    await _push_log(redis, bench_id, f"[gateway] vm cleanup [{level}]: {msg}")


# ---------- Subprocess runner ------------------------------------------


# Tracks live runs so DELETE can kill the subprocess. {bench_id: asyncio.subprocess.Process}
_LIVE: dict[str, asyncio.subprocess.Process] = {}


def _full_log_path(bench_id: str) -> Path:
    """On-disk file that captures *every* log line for a run, uncapped.
    Uploaded to S3 as `{prefix}logs.txt` on completion so the UI can replay
    the full log even after the redis list has been TTL'd or LRU-trimmed."""
    return _work_dir(bench_id) / "_full.log"


async def _push_log(redis, bench_id: str, line: str) -> None:
    if not line:
        return
    # Append to the full on-disk log (best-effort). This is the canonical
    # record — redis is just the live-tail cache.
    try:
        with _full_log_path(bench_id).open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    key = f"bench:logs:{bench_id}"
    try:
        await redis.rpush(key, line)
        await redis.ltrim(key, -LOG_LIST_CAP, -1)
    except Exception:
        # Logs are best-effort — never let log-pipe failures kill the runner.
        pass


# benchmaq prints `Pod created: <runpod-id>` exactly once when the pod comes
# up. We use that to capture the RunPod $/hour rate so the UI can display
# a live cost ticker. Captured once per bench (set keeps us idempotent).
_POD_CREATED_RE = re.compile(r"Pod created:\s*(\S+)")
_COST_CAPTURED: set[str] = set()

# On the RunPod path benchmaq keeps polling the vLLM /health endpoint to its
# (gateway-bumped) ceiling even after the engine has already crashed on startup
# — burning pod credits on a pod that will never serve. (The "abort on process
# death" patch in pyremote_shim only covers the ssh/bare-metal backend.) When we
# spot an unambiguous engine-init failure on the streamed log we tear the pod
# down + kill the run right away. {bench_id: pod_id}; aborted-set keeps the two
# (stdout+stderr) drains from firing it twice.
_POD_FOR_BENCH: dict[str, str] = {}
_CRASH_ABORTED: set[str] = set()

# Terminal vLLM startup-failure signatures → a clean one-line reason. Each only
# prints once the engine/worker process has already exited (the serve will never
# come up), so a match means "stop waiting, tear down". Deliberately narrow: a
# cold-start 504 or a slow torch.compile must NOT match — those still go healthy.
_VLLM_FATAL_SIGNATURES: list[tuple[str, str]] = [
    (r"The NVIDIA driver on your system is too old",
     "the pod's GPU driver is too old for the installed torch/vLLM build "
     "(e.g. vllm==0.23.0 ships a CUDA-13 torch, but a cu128 image lands the pod "
     "on a CUDA-12.x-driver host) — use a CUDA-13 image (…-cu1300-…) or pin a "
     "cu128 vLLM build"),
    (r"forward compatibility was attempted on non-supported",
     "the pod's GPU driver is older than the CUDA the torch/vLLM build needs "
     "— use a matching-CUDA image"),
    (r"EngineCore failed to start|Engine core initialization failed",
     "the vLLM engine core failed to initialise (see the traceback above)"),
    (r"WorkerProc (initialization )?failed",
     "a vLLM worker process failed to start (see the traceback above)"),
]
_VLLM_FATAL_RES = [(re.compile(rx), msg) for rx, msg in _VLLM_FATAL_SIGNATURES]


def _vllm_fatal_reason(text: str) -> Optional[str]:
    """Return a clean reason if a log line is a terminal vLLM startup failure
    (engine will never serve), else None. Used to fail-fast + tear the pod down
    instead of polling a dead /health for the whole health-wait ceiling."""
    for rx, msg in _VLLM_FATAL_RES:
        if rx.search(text):
            return msg
    return None


async def _abort_on_vllm_crash(
    redis, bench_id: str, reason: str, provider_id: Optional[str] = None
) -> None:
    """Tear down a run whose vLLM engine crashed on startup — don't let benchmaq
    poll a dead /health to its ceiling and burn pod credits. Tears the (expensive)
    RunPod pod down first, then kills the local benchmaq subprocess so the runner
    proceeds to mark the row failed."""
    await _push_log(
        redis, bench_id,
        f"[gateway] vLLM engine crashed on startup — {reason}. "
        "Aborting the health wait and tearing the pod down to stop credit burn.",
    )
    pod_id = _POD_FOR_BENCH.get(bench_id)
    if pod_id:
        try:
            await _terminate_runpod_pod(pod_id, provider_id=provider_id)
            await _push_log(redis, bench_id, f"[gateway] runpod pod {pod_id} torn down (crash-abort)")
        except Exception as e:
            await _push_log(redis, bench_id, f"[gateway] crash-abort: runpod teardown failed: {e}")
    proc = _LIVE.get(bench_id)
    if proc and proc.returncode is None:
        try:
            proc.kill()
        except Exception:
            pass


async def _fetch_runpod_cost(pod_id: str, provider_id: Optional[str] = None) -> Optional[float]:
    """Return RunPod's costPerHr for a pod by id, or None if anything goes
    sideways (no API key, pod not found, transient network error). Best-effort
    — never raises."""
    from .provider_resolve import try_resolve_cloud_creds
    async with session_factory()() as s:
        creds = await try_resolve_cloud_creds(s, provider_id, "runpod")
    if creds is None:
        return None
    base = os.environ.get("RUNPOD_API_BASE", "https://rest.runpod.io/v1")
    try:
        async with httpx.AsyncClient(
            base_url=base,
            headers={"Authorization": f"Bearer {creds.api_key}"},
            timeout=15.0,
        ) as cli:
            r = await cli.get(f"/pods/{pod_id}")
            if r.status_code >= 400:
                return None
            data = r.json()
            cost = data.get("costPerHr") or data.get("cost_per_hr")
            return float(cost) if cost is not None else None
    except Exception:
        return None


async def _terminate_runpod_pod(pod_id: str, provider_id: Optional[str] = None) -> None:
    """Delete a RunPod pod by id. Raises if the API call fails (caller logs)."""
    from .provider_resolve import resolve_cloud_creds
    async with session_factory()() as s:
        creds = await resolve_cloud_creds(s, provider_id, "runpod")
    base = os.environ.get("RUNPOD_API_BASE", "https://rest.runpod.io/v1")
    async with httpx.AsyncClient(
        base_url=base,
        headers={"Authorization": f"Bearer {creds.api_key}"},
        timeout=30.0,
    ) as cli:
        r = await cli.delete(f"/pods/{pod_id}")
        if r.status_code >= 400 and r.status_code != 404:
            raise RuntimeError(f"RunPod terminate {pod_id}: {r.status_code} {r.text[:200]}")


async def _capture_runpod_cost(bench_id: str, pod_id: str, provider_id: Optional[str] = None) -> None:
    """Look up the hourly rate for the pod benchmaq just spawned and store it
    on the row. Logged but otherwise best-effort — a cost-tracking failure must
    never break the run."""
    cost = await _fetch_runpod_cost(pod_id, provider_id=provider_id)
    try:
        async with session_factory()() as s:
            row = await s.get(Benchmark, bench_id)
            if row is None:
                return
            row.runpod_pod_id = pod_id
            row.cost_per_hr = cost
            await s.commit()
    except Exception:
        logger.warning("bench %s: failed to persist cost for pod %s", bench_id, pod_id)
        return
    logger.info("bench %s: pod=%s cost=%s/hr", bench_id, pod_id, cost)


async def _drain(stream: asyncio.StreamReader, prefix: str, redis, bench_id: str, provider_id: Optional[str] = None) -> None:
    """Read lines from a subprocess pipe and fan them out to redis + python log."""
    while True:
        line = await stream.readline()
        if not line:
            return
        text = line.decode("utf-8", "replace").rstrip()
        await _push_log(redis, bench_id, f"{prefix}{text}")
        # First-seen `Pod created: <id>` → kick off a cost lookup. Only watch
        # stdout (prefix == "") since stderr can echo unrelated text.
        if not prefix and bench_id not in _COST_CAPTURED:
            m = _POD_CREATED_RE.search(text)
            if m:
                _COST_CAPTURED.add(bench_id)
                _POD_FOR_BENCH[bench_id] = m.group(1)
                asyncio.create_task(_capture_runpod_cost(bench_id, m.group(1), provider_id=provider_id))
        # Fail fast on a terminal vLLM engine crash — otherwise benchmaq polls a
        # dead /health to its ceiling and burns pod credits. Fires once per bench.
        if bench_id not in _CRASH_ABORTED:
            reason = _vllm_fatal_reason(text)
            if reason:
                _CRASH_ABORTED.add(bench_id)
                asyncio.create_task(_abort_on_vllm_crash(redis, bench_id, reason, provider_id=provider_id))


# Recognised model-download (hf) failure signatures → a clean one-liner. When a
# benchmaq run dies at the download step it prints a generic "Model download
# failed with exit code N" and then the pod-cleanup epilogue, which buries the
# real `hf` error — so we dig it out of the log and name the likely cause.
_DL_FAILURE_SIGNATURES: list[tuple[str, str]] = [
    (r"gated repo|GatedRepo|awaiting a review of your request|access to model.*restricted|must agree|accept the conditions",
     "the model is gated — accept access on HuggingFace and pass a token that can see it"),
    (r"401 Client Error|Invalid (user )?token|Unauthorized|authentication.*fail",
     "HuggingFace auth failed (401) — set a valid HF token with access to this repo"),
    (r"404 Client Error|RepositoryNotFound|Repository Not Found|does not (exist|appear)|could not be found",
     "the repo wasn't found (404) — check the model id (and that the token can see it)"),
    (r"hf_transfer",
     "hf_transfer is enabled (HF_HUB_ENABLE_HF_TRANSFER=1) but not installed in the pod"),
    (r"No space left on device|Disk quota exceeded|\[Errno 28\]",
     "the pod ran out of disk while downloading — use a larger container/volume disk"),
    (r"Connection (error|reset|timed out)|Max retries exceeded|Temporary failure in name resolution|Read timed out|Failed to establish a new connection",
     "a network error reaching HuggingFace (transient) — retry"),
]


def _download_failure_reason(text: str) -> Optional[str]:
    """If a run failed at the model-download step, return a clean one-line reason
    (a recognised hf error, or the tail of the hf output) instead of the generic
    'exit code N' + pod-cleanup noise. None when it wasn't a download failure."""
    if "Model download failed" not in text and "DOWNLOADING MODEL" not in text:
        return None
    m = re.search(r"DOWNLOADING MODEL:\s*(\S+)", text)
    head = f"Model download failed{f' for {m.group(1)}' if m else ''}"
    for rx, msg in _DL_FAILURE_SIGNATURES:
        if re.search(rx, text, re.IGNORECASE):
            return f"{head}: {msg}."
    # Unrecognised — surface the hf output between the download banner and the
    # failure line (the real error), dropping progress bars + boilerplate.
    di, fi = text.rfind("DOWNLOADING MODEL"), text.find("Model download failed")
    seg = text[di:fi] if (di >= 0 and fi > di) else text
    lines = [
        ln.strip() for ln in seg.splitlines()
        if ln.strip()
        and not re.search(r"\d+%\|", ln)          # tqdm progress bars
        and "DOWNLOADING MODEL" not in ln
        and "Model download failed" not in ln     # the generic line we're replacing
        and not re.match(r"(?i)^(error|warning):?\s*$", ln.strip())
        and not ln.startswith(("Running:", "Destination:", "="))
    ]
    tail = "\n".join(lines[-12:]).strip()
    return f"{head}:\n{tail}" if tail else (
        f"{head} — hf produced no error output (often a bad model id, or a missing/wrong HF token)."
    )


async def run_benchmark(redis, bench_id: str, raw_yaml: str) -> None:
    """End-to-end runner for one benchmark. Owns the subprocess from spawn → S3 sync."""
    work = _work_dir(bench_id)

    # Mark running + start time, and read provider_id so we know whether to
    # take the RunPod path or the VM/SSH path.
    async with session_factory()() as s:
        b = await s.get(Benchmark, bench_id)
        if b is None:
            return
        b.status = "running"
        b.started_at = datetime.now(timezone.utc)
        provider_id = b.provider_id
        storage_id = b.storage_id
        s3_prefix = b.s3_prefix
        cleanup_model = bool(getattr(b, "cleanup_model", True))
        run_env_vars = getattr(b, "env_vars", None) or None
        run_visible_devices = getattr(b, "visible_devices", None) or None
        # Admin global env / secrets (e.g. HF_TOKEN) merged into the run; a
        # per-benchmark env var of the same name overrides it.
        from .global_env_api import load_global_env
        global_env = await load_global_env(s)
        # A benchmark-selected HF token secret is aliased to HF_TOKEN at the
        # per-run env layer (highest precedence) so it wins over the global one,
        # resolved fresh here so rotating the secret takes effect on the next run.
        hf_secret_key = getattr(b, "hf_token_secret", None)
        if hf_secret_key and global_env.get(hf_secret_key):
            run_env_vars = {**(run_env_vars or {}), "HF_TOKEN": global_env[hf_secret_key]}
        # Same for the ingress endpoint API key → OPENAI_API_KEY (the ingress
        # client reads it and sends Authorization: Bearer).
        api_secret_key = getattr(b, "api_key_secret", None)
        if api_secret_key and global_env.get(api_secret_key):
            run_env_vars = {**(run_env_vars or {}), "OPENAI_API_KEY": global_env[api_secret_key]}
        await s.commit()

    # Ingress: bench an already-served endpoint, no machine. Detected from a
    # base_url in the config + no machine provider picked. Runs the in-gateway
    # httpx client instead of benchmaq; skips provisioning, cloud check and keys.
    is_ingress = (not provider_id) and (_ingress_base_url(raw_yaml) is not None)

    # Disambiguate provider kind. A runpod-kind provider_id picks WHICH
    # RunPod account to bill against (cred override only); a vm-kind id
    # switches us onto the bare-metal SSH path.
    provider_kind: Optional[str] = None
    if provider_id:
        from .db import Provider
        async with session_factory()() as s:
            prov = await s.get(Provider, provider_id)
            if prov is None:
                await _push_log(redis, bench_id, f"[gateway] provider {provider_id} not found")
                async with session_factory()() as s2:
                    b2 = await s2.get(Benchmark, bench_id)
                    if b2 is not None:
                        b2.status = "failed"
                        b2.error_text = f"provider {provider_id} not found"
                        b2.ended_at = datetime.now(timezone.utc)
                        await s2.commit()
                return
            provider_kind = prov.kind

    # Cloud-disabled: refuse anything that isn't a physical vm provider before we
    # reach the `benchmaq runpod bench` path (provider_kind None → that default).
    # Defense in depth for rows queued/retried from before the flag was set; the
    # create API rejects these up front too.
    from .provider import ensure_benchmark_provider_allowed, CloudProviderDisabled
    try:
        # Ingress provisions nothing (no pod, no cloud spend) so the cloud-
        # disabled gate doesn't apply — it only guards the spawn-a-pod path.
        if not is_ingress:
            ensure_benchmark_provider_allowed(provider_kind)
    except CloudProviderDisabled:
        msg = ("cloud GPU providers are disabled on this deployment — register a "
               "physical 'vm' provider and re-run this benchmark")
        await _push_log(redis, bench_id, f"[gateway] {msg}")
        async with session_factory()() as s:
            b2 = await s.get(Benchmark, bench_id)
            if b2 is not None:
                b2.status = "failed"
                b2.error_text = msg
                b2.ended_at = datetime.now(timezone.utc)
                await s.commit()
        return

    vm_target: Optional[dict] = None
    runpod_creds = None
    runpod_key_path: Optional[str] = None
    if provider_kind == "vm":
        # Guard against benchmaq version drift: the VM path needs the `backend:
        # ssh` runner (run_remote_ssh). If the installed benchmaq is older and
        # only has the pyremote run_remote, fail loudly here instead of silently
        # falling back to it (which breaks on VMs that print an SSH MOTD banner).
        try:
            import benchmaq.runner as _bmr
            has_ssh_backend = hasattr(_bmr, "run_remote_ssh")
        except Exception:
            has_ssh_backend = False
        if not has_ssh_backend:
            msg = (
                "installed benchmaq lacks the SSH backend (run_remote_ssh) — VM "
                "benchmarks need the pinned ref. Reinstall: uv pip install "
                "--reinstall-package benchmaq "
                "'benchmaq @ git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq.git@main'"
            )
            await _push_log(redis, bench_id, f"[gateway] {msg}")
            async with session_factory()() as s:
                b2 = await s.get(Benchmark, bench_id)
                if b2 is not None:
                    b2.status = "failed"
                    b2.error_text = msg[:4000]
                    b2.ended_at = datetime.now(timezone.utc)
                    await s.commit()
            return
        try:
            vm_target = await _materialise_vm_key(work, provider_id)
            await _push_log(redis, bench_id, f"[gateway] bare-metal target: {vm_target['user']}@{vm_target['host']}:{vm_target['port']}")
        except Exception as e:
            await _push_log(redis, bench_id, f"[gateway] could not prepare VM target: {e}")
            async with session_factory()() as s:
                b2 = await s.get(Benchmark, bench_id)
                if b2 is not None:
                    b2.status = "failed"
                    b2.error_text = f"VM target setup failed: {e}"[:4000]
                    b2.ended_at = datetime.now(timezone.utc)
                    await s.commit()
            return
    elif provider_kind == "runpod":
        from .provider_resolve import resolve_cloud_creds
        try:
            async with session_factory()() as s:
                runpod_creds = await resolve_cloud_creds(s, provider_id, "runpod")
            await _push_log(redis, bench_id, f"[gateway] runpod provider {provider_id} (key ****{runpod_creds.api_key[-4:]})")
        except Exception as e:
            await _push_log(redis, bench_id, f"[gateway] could not resolve runpod provider: {e}")
            async with session_factory()() as s:
                b2 = await s.get(Benchmark, bench_id)
                if b2 is not None:
                    b2.status = "failed"
                    b2.error_text = f"runpod provider resolve failed: {e}"[:4000]
                    b2.ended_at = datetime.now(timezone.utc)
                    await s.commit()
            return

    # RunPod path (no VM target): mint a throwaway SSH keypair per run so we
    # don't depend on a pre-provisioned key on disk. benchmaq injects the .pub
    # into the pod (PUBLIC_KEY) and SSHes with the private half; both are deleted
    # once benchmaq exits.
    if vm_target is None and not is_ingress:
        try:
            runpod_key_path = _gen_ephemeral_runpod_key(work, bench_id)
            await _push_log(redis, bench_id, "[gateway] minted ephemeral SSH keypair for pod access")
        except Exception as e:
            await _push_log(redis, bench_id, f"[gateway] could not mint SSH keypair: {e}")

    cfg_path = work / "config.yaml"
    cfg_path.write_text(_resolve_config(
        raw_yaml, vm_target=vm_target, ingress=is_ingress,
        env_vars=run_env_vars, visible_devices=run_visible_devices,
        runpod_key_path=runpod_key_path, bench_id=bench_id, global_env=global_env,
    ))

    sub_cmd: list[str] = []
    if is_ingress:
        await _push_log(
            redis, bench_id,
            f"[gateway] ingress mode — benchmarking {_ingress_base_url(raw_yaml)} "
            f"with the in-gateway client (no pod) (cwd={work})",
        )
    elif vm_target:
        sub_cmd = _pick_engine_subcommand(raw_yaml)
        await _push_log(redis, bench_id, f"[gateway] starting benchmaq {' '.join(sub_cmd)} (cwd={work})")
    else:
        sub_cmd = ["runpod", "bench"]
        await _push_log(redis, bench_id, f"[gateway] starting benchmaq runpod bench (cwd={work})")

    env = dict(os.environ)
    # Cred override: when the user picked a runpod-kind provider, the key
    # stored in providers.config wins over the gateway-wide env. benchmaq
    # picks RUNPOD_API_KEY straight from the subprocess env.
    if runpod_creds is not None:
        env["RUNPOD_API_KEY"] = runpod_creds.api_key
        if runpod_creds.cloud_type:
            env["RUNPOD_CLOUD_TYPE"] = runpod_creds.cloud_type
    else:
        env["RUNPOD_API_KEY"] = os.environ.get("RUNPOD_API_KEY", "")
    env["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    # benchmaq writes results into the cwd by default unless config says otherwise.

    # Prefer the venv-local `benchmaq` (sibling of the running python) since
    # the gateway process inherits PATH from however it was launched, which
    # may not include .venv/bin. Fall back to PATH lookup, then bare name.
    sibling = Path(sys.executable).parent / "benchmaq"
    if sibling.exists():
        benchmaq_bin = str(sibling)
    else:
        benchmaq_bin = shutil.which("benchmaq") or "benchmaq"

    # Make sure the venv's bin is on PATH for the subprocess too — benchmaq
    # itself shells out to runpodctl, uv, etc., and may need them.
    env_path = env.get("PATH", "")
    venv_bin = str(Path(sys.executable).parent)
    if venv_bin not in env_path.split(":"):
        env["PATH"] = f"{venv_bin}:{env_path}" if env_path else venv_bin

    # Force unbuffered stdout/stderr — without this, benchmaq's print()s sit
    # in the pipe buffer and the UI sees nothing until the run finishes.
    env["PYTHONUNBUFFERED"] = "1"

    # Custom-fork installs need their leading env (e.g. VLLM_USE_PRECOMPILED=1) ON
    # the `uv pip install` command. The RunPod path installs via benchmaq→pyremote,
    # whose non-login `bash -c` SSH session does NOT inherit the pod `--env` — so the
    # fork would silently build from source (very slow). Pass the install env to our
    # patched pyremote `_install_dependencies` via SGPU_PIP_ENV. (The VM path applies
    # this itself in pyremote_shim and doesn't use pyremote's installer, so no-op there.)
    try:
        _cfg0 = yaml.safe_load(raw_yaml) or {}
        _ia = str((((_cfg0.get("remote") or {}).get("uv") or {}).get("vllm_install_args")) or "").strip()
        if _ia:
            _ienv, _ = _split_leading_env(_ia)
            if _ienv:
                env["SGPU_PIP_ENV"] = "".join(f"{k}={v} " for k, v in _ienv.items())
                await _push_log(redis, bench_id, f"[gateway] fork install env: {' '.join(_ienv)}")
    except Exception:
        pass

    await _push_log(redis, bench_id, f"[gateway] benchmaq binary: {benchmaq_bin}")

    # Invoke through python -u so even C-level stdio is line-buffered, in case
    # benchmaq spawns subprocesses (runpodctl) whose output also needs to flow.
    # For VM (bare-metal) runs, we route through a thin wrapper that installs
    # the pyremote reconnect-per-command shim before benchmaq's CLI runs.
    # This sidesteps Go-based SSH proxies (e.g. TM's `ssh.*.gpu.tm.com.my`)
    # that enforce one exec channel per TCP connection.
    if is_ingress:
        # No machine: run the lightweight in-gateway httpx load generator
        # against base_url. No benchmaq, no SSH wrapper.
        cmd_argv = [sys.executable, "-u", "-m", "gateway.bench_ingress", str(cfg_path)]
    elif vm_target is not None:
        cmd_argv = [sys.executable, "-u", "-m", "gateway.bench_remote_wrapper", *sub_cmd, str(cfg_path)]
    else:
        cmd_argv = [sys.executable, "-u", benchmaq_bin, *sub_cmd, str(cfg_path)]
    proc = await asyncio.create_subprocess_exec(
        *cmd_argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(work),
        env=env,
    )
    _LIVE[bench_id] = proc

    try:
        await asyncio.gather(
            _drain(proc.stdout, "", redis, bench_id, provider_id=provider_id),
            _drain(proc.stderr, "[stderr] ", redis, bench_id, provider_id=provider_id),
        )
        rc = await proc.wait()
    except asyncio.CancelledError:
        await _push_log(redis, bench_id, "[gateway] cancelled — killing subprocess")
        try:
            proc.kill()
        except Exception:
            pass
        rc = -1
        raise
    finally:
        _LIVE.pop(bench_id, None)
        _POD_FOR_BENCH.pop(bench_id, None)
        _CRASH_ABORTED.discard(bench_id)
        _COST_CAPTURED.discard(bench_id)

    await _push_log(redis, bench_id, f"[gateway] benchmaq exited rc={rc}")

    # Shred the per-run ephemeral SSH keypair now that the pod work is done. The
    # pod (and its injected PUBLIC_KEY) is torn down separately; the private key
    # has no further use and shouldn't linger in /tmp.
    if runpod_key_path:
        for _p in (Path(runpod_key_path), Path(runpod_key_path + ".pub")):
            try:
                _p.unlink(missing_ok=True)
            except Exception:
                pass

    # Bare-metal runs leave the model in the VM's HF cache + local_dir. Clean
    # both up so a series of benchmarks on different models doesn't fill the
    # VM's disk. Best-effort: failures are logged but don't change the run
    # outcome (the benchmark already finished).
    if vm_target is not None and cleanup_model:
        try:
            await _cleanup_vm_model(redis, bench_id, vm_target, raw_yaml)
        except Exception as e:
            await _push_log(redis, bench_id, f"[gateway] vm cleanup crashed: {e}")
    elif vm_target is not None:
        await _push_log(redis, bench_id, "[gateway] vm cleanup: skipped (cleanup_model=false)")

    # Sync any result files dropped under work/ into S3 — to the storage the
    # user picked at create time (or the env bucket when none was set). Use the
    # prefix baked into the row so reads + writes always agree.
    target = await _bench_s3_target(storage_id)
    prefix = s3_prefix
    # Redact secrets from the resolved config before persisting — this file is
    # downloadable via the files API (and by every logged-in user if the bench is
    # public); the un-redacted resolve carries the platform RunPod/HF creds and
    # all global secrets.
    s3_put_text(
        f"{prefix}config.yaml",
        _redact_config_for_display(
            _resolve_config(raw_yaml, vm_target=vm_target, ingress=is_ingress, env_vars=run_env_vars, visible_devices=run_visible_devices, runpod_key_path=runpod_key_path, bench_id=bench_id, global_env=global_env)
        ),
        target=target,
    )
    result_json: Optional[dict] = None
    error_excerpt: Optional[str] = None

    # Upload the complete, uncapped log to S3 as logs.txt. The stream endpoint
    # falls back to this for terminal benches so the UI can replay the full
    # log forever, even after the redis list has been trimmed or TTL'd.
    full_log = _full_log_path(bench_id)
    if full_log.exists():
        try:
            s3_put_file(f"{prefix}logs.txt", str(full_log), target=target)
        except Exception as e:
            await _push_log(redis, bench_id, f"[gateway] s3 upload failed for logs.txt: {e}")

    # Never upload secrets/control files. `vm_key` is the decrypted SSH private
    # key materialised for the run — uploading it leaks the VM credential to S3.
    _NO_UPLOAD = {"config.yaml", "_full.log", "vm_key", "vm_key.pub", "rp_key", "rp_key.pub"}
    # Big per-request arrays — dropped from the aggregate result.json summary.
    _DROP_KEYS = {
        "generated_texts", "input_texts", "itls", "tpots", "ttfts", "e2els",
        "input_lens", "output_lens", "errors",
    }
    per_config_results: list[dict] = []
    _raw_candidates: list[dict] = []
    for path in sorted(work.rglob("*")):
        if not path.is_file() or path.name in _NO_UPLOAD:
            continue
        rel = path.relative_to(work).as_posix()
        try:
            s3_put_file(f"{prefix}{rel}", str(path), target=target)
        except Exception as e:
            await _push_log(redis, bench_id, f"[gateway] s3 upload failed for {rel}: {e}")
        # Collect each per-config result.json (benchmaq writes one per bench row).
        if path.suffix == ".json" and path.name != "result.json":
            try:
                with path.open() as f:
                    candidate = json.load(f)
                if isinstance(candidate, dict):
                    _raw_candidates.append(candidate)
                    per_config_results.append(
                        {"file": rel, **{k: v for k, v in candidate.items() if k not in _DROP_KEYS}}
                    )
            except Exception:
                pass

    # DB result_json (the single summary column) = the completed config with the
    # highest throughput. A crashed/failed config reports completed=0 and 0
    # throughput, so it never shadows a real result; this falls back to the first
    # candidate only when none completed (max() keeps the first on a tie).
    if _raw_candidates and result_json is None:
        def _cand_score(c: dict) -> float:
            completed = c.get("completed") or 0
            if not (isinstance(completed, (int, float)) and completed > 0):
                return -1.0
            tp = c.get("total_token_throughput") or c.get("output_throughput") or 0
            return float(tp) if isinstance(tp, (int, float)) else 0.0
        # Strip the heavy per-request arrays before they land in the DB column. The
        # raw per-config files (with full itls/generated_texts) are already in S3
        # (uploaded above), so keeping them here just bloats result_json to multi-MB
        # — which made GET /benchmarks load tens of MB out of Postgres per request
        # (~1.4s, event-loop block) and embedded NUL bytes (from generated_texts)
        # that break any SQL-level json access. Keep scalars + the accuracy summary.
        result_json = _slim_result_json(max(_raw_candidates, key=_cand_score))

    # Build a canonical, aggregate `result.json` (summary across all bench
    # configs, heavy per-request arrays stripped) and put it in storage if one
    # doesn't already exist. benchmaq only emits per-config files; this is the
    # single artifact tools/UI can read for the whole run.
    if per_config_results:
        try:
            if s3_get_text(f"{prefix}result.json", target=target) is None:
                agg = {"bench_id": bench_id, "count": len(per_config_results), "results": per_config_results}
                s3_put_text(f"{prefix}result.json", json.dumps(agg, indent=2), target=target)
                await _push_log(redis, bench_id, f"[gateway] built result.json ({len(per_config_results)} configs)")
        except Exception as e:
            await _push_log(redis, bench_id, f"[gateway] failed to build result.json: {e}")

    # Accuracy mode: accuracy_eval.py emits one `@@ACCURACY {json}` line per
    # (config, dataset). Scan the full log, collect the result events, and fold
    # them into result_json so the UI can draw the IQ-vs-speed plot. Provider-
    # agnostic — the markers ride the streamed log, not S3 result files.
    accuracy_results: list[dict] = []
    accuracy_errors: list[dict] = []
    if full_log.exists():
        try:
            _marker = "@@ACCURACY "
            for line in full_log.read_text(encoding="utf-8", errors="replace").splitlines():
                idx = line.find(_marker)
                if idx == -1:
                    continue
                try:
                    obj = json.loads(line[idx + len(_marker):])
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("event") == "result":
                    accuracy_results.append(obj)
                elif obj.get("event") == "error":
                    accuracy_errors.append(obj)
        except Exception:
            pass
    # An accuracy run that produced no results but did emit errors (e.g. the
    # server never came up) is a failure even though the script returns 0.
    accuracy_failed = bool(accuracy_errors) and not accuracy_results
    if accuracy_results:
        if result_json is None:
            result_json = {"bench_id": bench_id}
        result_json["accuracy"] = accuracy_results
        try:
            existing = s3_get_text(f"{prefix}result.json", target=target)
            agg = json.loads(existing) if existing else {"bench_id": bench_id}
            agg["accuracy"] = accuracy_results
            s3_put_text(f"{prefix}result.json", json.dumps(agg, indent=2), target=target)
        except Exception as e:
            await _push_log(redis, bench_id, f"[gateway] failed to write accuracy to result.json: {e}")
        await _push_log(redis, bench_id, f"[gateway] parsed {len(accuracy_results)} accuracy result(s)")

    # A run where benchmaq exits 0 but EVERY request failed (0 successful across
    # all bench configs) is not a valid result — almost always the vLLM server
    # never actually served (port already in use, 404s, model-load failure). The
    # exit code lies in that case, so we inspect the per-config "Successful
    # requests" tallies and fail the run if they're all zero.
    all_requests_failed = False
    if rc == 0 and full_log.exists():
        try:
            _text = full_log.read_text(encoding="utf-8", errors="replace")
            _succ = [int(m) for m in re.findall(r"Successful requests:\s*(\d+)", _text)]
            if _succ and all(s == 0 for s in _succ):
                all_requests_failed = True
        except Exception:
            pass

    if rc != 0 or all_requests_failed:
        # Tail the full on-disk log for the error_text card on the list page.
        try:
            if full_log.exists():
                with full_log.open("r", encoding="utf-8", errors="replace") as f:
                    all_lines = f.readlines()
                error_excerpt = "".join(all_lines[-50:])[-4000:]
                # Surface a clean one-liner for known failure patterns so the
                # list page doesn't show a raw log wall.
                # A vLLM engine crash we already crash-aborted on: lead with the
                # reason we logged (it sits just below the traceback, so scan a
                # wider window than the 50-line tail).
                _crash_m = re.search(
                    r"\[gateway\] vLLM engine crashed on startup[^\n]*",
                    "".join(all_lines[-300:]),
                )
                _cuda_m = re.search(
                    r"CUDA mismatch[^\n]{0,200}", error_excerpt, re.IGNORECASE
                )
                if _crash_m:
                    error_excerpt = (
                        _crash_m.group(0).replace("[gateway] ", "").strip()
                        + "\n\n— log tail —\n" + (error_excerpt or "")
                    )[:4000]
                elif _cuda_m:
                    error_excerpt = _cuda_m.group(0).strip()
                else:
                    # Model-download failures: the real hf error sits ABOVE the
                    # pod-cleanup epilogue, so the last-50-lines tail misses it.
                    # Scan a wider window and lead with the actual cause.
                    _dl = _download_failure_reason("".join(all_lines[-2000:]))
                    if _dl:
                        error_excerpt = (_dl + "\n\n— log tail —\n" + (error_excerpt or ""))[:4000]
        except Exception:
            error_excerpt = None
        if all_requests_failed:
            # Prepend a clear reason so the UI doesn't just show a metrics wall.
            # Scan a wider window for the health-wait timeout — the most common cause
            # for a large model: it never finished loading + compiling + capturing
            # CUDA graphs before benchmaq's health wait elapsed, so the bench fired
            # against a not-yet-ready server (ConnectionRefused → 0 successful).
            _wide = ""
            try:
                if full_log.exists():
                    _wide = "".join(full_log.open("r", encoding="utf-8", errors="replace").readlines()[-2000:])
            except Exception:
                _wide = error_excerpt or ""
            if "failed to become healthy" in _wide:
                _addr = ("the model didn't become healthy within benchmaq's health wait "
                         "(a large model's load + torch.compile + CUDA-graph capture can "
                         "exceed it on the first run) — give it a longer wait or pre-warm the compile cache")
            elif full_log.exists() and "Address already in use" in (error_excerpt or ""):
                _addr = "the vLLM serve port was already in use"
            else:
                _addr = "the vLLM server didn't serve the model (check for 404s / load errors)"
            error_excerpt = (
                f"All requests failed — 0 successful across every bench config. Likely {_addr}.\n\n"
                + (error_excerpt or "")
            )[:4000]

    async with session_factory()() as s:
        b = await s.get(Benchmark, bench_id)
        if b is None:
            return
        b.status = "done" if (rc == 0 and not all_requests_failed and not accuracy_failed) else "failed"
        b.exit_code = rc
        if accuracy_failed and not error_excerpt:
            _first_err = (accuracy_errors[0] or {}).get("error", "unknown error")
            error_excerpt = (
                "Accuracy eval produced no results — the model never served or "
                f"every dataset failed to load. First error: {_first_err}"
            )[:4000]
        b.error_text = error_excerpt
        b.result_json = result_json
        b.ended_at = datetime.now(timezone.utc)
        await s.commit()
        if all_requests_failed:
            await _push_log(redis, bench_id, "[gateway] marked failed: 0 successful requests across all bench configs")

    # TTL on the log list so old runs eventually drop out of redis.
    try:
        await redis.expire(f"bench:logs:{bench_id}", LOG_LIST_TTL_S)
    except Exception:
        pass


# ---------- Startup hooks -----------------------------------------------


def bootstrap_ssh_key_from_env() -> None:
    """Prod delivers the RunPod SSH private key via env var
    (BENCHMARK_SSH_PRIVATE_KEY) from a SealedSecret — pods can't read the
    developer's ~/.runpod/ssh/RunPod-Key-Go file. We materialize the env
    value to disk at BENCHMARK_SSH_KEY_PATH (chmod 0600) on startup so the
    rest of the code keeps using a normal file path.

    Idempotent: skips if the file already exists, so local dev keeps using
    the runpodctl-managed key without disturbance.
    """
    key = os.environ.get("BENCHMARK_SSH_PRIVATE_KEY", "")
    if not key.strip():
        return
    path = Path(_ssh_key_path())
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    text = key if key.endswith("\n") else key + "\n"
    path.write_text(text)
    try:
        path.chmod(0o600)
    except Exception:
        pass
    logger.info("bench: wrote SSH key from env to %s", path)


# In-process registry of running bench tasks. Populated when a bench is
# kicked off (see create_benchmark) and used by the janitor to detect rows
# that say `running` in the DB but have no live task in this process —
# usually the result of an asyncio task GC, a crashed coroutine that
# couldn't update the DB, or a SIGKILL that bypassed _safe_run.
_active_runners: dict[str, asyncio.Task] = {}

# How long a bench can sit at status='running' with no live in-process task
# and no recent redis log activity before the janitor reaps it. Generous
# enough to tolerate brief gaps (model warmup, vllm compile) while still
# catching genuinely dead rows.
_JANITOR_STALL_SECONDS = 600


async def janitor_loop(redis) -> None:
    """Periodically sweep for `running` benchmark rows that have no live task
    in this process and no recent log activity, and mark them failed.

    Triggered by the asyncio-task-GC bug (a fire-and-forget create_task can
    vanish silently if no strong ref is held). The strong-ref fix in
    create_benchmark prevents that going forward; the janitor is the safety
    net for any other path that leaves a row stranded (OOMKill, SIGTERM
    after _safe_run started but before it could update the DB, etc.).
    """
    while True:
        try:
            await _janitor_sweep(redis)
        except Exception as e:
            logger.warning("bench janitor sweep failed: %s", e)
        await asyncio.sleep(60)


async def _janitor_sweep(redis) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=_JANITOR_STALL_SECONDS)
    async with session_factory()() as s:
        rows = (
            await s.execute(
                select(Benchmark).where(
                    Benchmark.status == "running",
                    Benchmark.started_at < cutoff,
                )
            )
        ).scalars().all()

    for b in rows:
        if b.id in _active_runners:
            continue  # live task in this process — leave alone
        logger.warning("bench janitor: reaping stuck row %s (no live task)", b.id)
        async with session_factory()() as s:
            row = await s.get(Benchmark, b.id)
            if row is None or row.status != "running":
                continue
            row.status = "failed"
            row.exit_code = -1
            row.ended_at = datetime.now(timezone.utc)
            row.error_text = "subprocess vanished — reaped by gateway janitor"
            await s.commit()
        try:
            await _push_log(redis, b.id, "[gateway] reaped by janitor — runner vanished")
        except Exception:
            pass


async def cleanup_orphaned_running(redis) -> int:
    """Called from main.py lifespan. Marks any rows still 'running' (left over
    from a previous gateway process) as 'failed' with a recovery message."""
    async with session_factory()() as s:
        rows = await s.execute(
            update(Benchmark)
            .where(Benchmark.status.in_(["running", "queued"]))
            .values(
                status="failed",
                error_text="orphaned by gateway restart — any worker spawned for this run "
                           "(VM process or cloud pod) may still be live; check its provider",
                ended_at=datetime.now(timezone.utc),
            )
            .returning(Benchmark.id)
        )
        ids = [row[0] for row in rows.all()]
        await s.commit()
    for bid in ids:
        # Push the marker line first so it lands in _full.log before we
        # upload that file as the canonical logs.txt in S3.
        try:
            await _push_log(redis, bid, "[gateway] orphaned by gateway restart — marking failed")
        except Exception:
            pass
        # Upload whatever we managed to capture on disk so the Logs tab can
        # replay it. Without this, the stream falls back to the trimmed redis
        # tail and the user loses the head of the run.
        full_log = _full_log_path(bid)
        if full_log.exists():
            try:
                async with session_factory()() as s:
                    _row = await s.get(Benchmark, bid)
                _target = await _bench_s3_target(_row.storage_id if _row else None)
                _prefix = _row.s3_prefix if _row else benchmark_s3_prefix(bid, _target)
                s3_put_file(f"{_prefix}logs.txt", str(full_log), target=_target)
            except Exception as e:
                logger.warning("orphan %s: failed to upload _full.log: %s", bid, e)
    return len(ids)


# ---------- Pydantic schemas -------------------------------------------


class CreateBenchmarkRequest(BaseModel):
    name: str
    config_yaml: str
    # NULL/absent means use the platform default cloud (RunPod). Set to a
    # provider id (from /v1/providers) to bind this run to a user-registered
    # VM. Phase 2 just persists the choice; phase 3 will route execution.
    provider_id: Optional[str] = None
    # Storage backend (Storage row, kind=s3) for logs + result files. NULL/absent
    # falls back to the BENCHMARK_S3_BUCKET env. The web form requires it.
    storage_id: Optional[str] = None
    # VM runs only: remove the model from the VM after the run. Ignored for
    # cloud runs since the RunPod pod is torn down anyway.
    cleanup_model: Optional[bool] = None
    # Extra env for the run (cache/home dirs, etc.). Absolute-path values are
    # mkdir -p'd on the VM. RunPod runs pass these to the pod.
    env_vars: Optional[dict[str, str]] = None
    # CUDA_VISIBLE_DEVICES pin, e.g. "0,1,2,3". Empty/None = all GPUs.
    visible_devices: Optional[str] = None
    # HuggingFace token for gated models. A global-secret KEY (resolved to
    # HF_TOKEN at launch). Pasted tokens come through env_vars["HF_TOKEN"].
    hf_token_secret: Optional[str] = None
    # Ingress endpoint API key. A global-secret KEY (resolved to OPENAI_API_KEY at
    # launch). Pasted keys come through env_vars["OPENAI_API_KEY"].
    api_key_secret: Optional[str] = None
    # Create the run already shared publicly (read-only visible to every logged-in
    # user). Default False = private. Same flag the post-creation
    # /{bench_id}/visibility toggle flips; set here to choose at create time.
    is_public: bool = False
    # GPU identity for runs the platform doesn't provision (ingress/Slurm): the
    # hardware behind base_url, e.g. "NVIDIA H20". Also settable as a top-level
    # `gpu_type:`/`gpu_count:` key in the YAML itself. Cloud/VM runs don't need
    # this — their GPU is derived from runpod.pod / the provider as usual.
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None


class UpdateBenchmarkRequest(BaseModel):
    # All fields optional — absent means "leave unchanged". `name` renames (must
    # be non-empty when present). `gpu_type`/`gpu_count` set the manual hardware
    # identity ("" / 0 clears back to config-derived) so ingress/Slurm runs group
    # correctly in stats/aggregate/export and external API consumers.
    name: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None


class BenchmarkRecord(BaseModel):
    id: str
    name: str
    status: str
    s3_prefix: str
    config_yaml: str
    exit_code: Optional[int] = None
    error_text: Optional[str] = None
    result_json: Optional[dict] = None
    created_by: str
    # Whether this run is shared publicly, and whether the requesting user owns
    # it. is_owner is computed per-request (None when the caller is unknown).
    is_public: bool = False
    is_owner: Optional[bool] = None
    created_at: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    cost_per_hr: Optional[float] = None
    provider_id: Optional[str] = None
    storage_id: Optional[str] = None
    env_vars: Optional[dict[str, str]] = None
    visible_devices: Optional[str] = None
    hf_token_secret: Optional[str] = None
    api_key_secret: Optional[str] = None
    # Exposed so the UI "Re-run" button can faithfully recreate the run — without
    # it, a re-run would default cleanup_model=true and wipe a pre-downloaded cache.
    cleanup_model: Optional[bool] = None
    # Effective GPU identity, top-level so API consumers (e.g. the GPU calculator)
    # never have to parse config_yaml: the manually-set row value when present
    # (the only way ingress/Slurm runs get one), else the config's runpod.pod /
    # top-level gpu_type. NULL when neither knows.
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None


class FileRecord(BaseModel):
    name: str
    size: int
    modified: str
    download_url: str


# ---- Portable export/import (move a finished benchmark between deployments) --
# The export is a self-contained JSON: the DB row's results + config plus the
# run's S3 artifacts inlined as base64, so the destination needs no access to
# the source's bucket. Import re-creates the row + writes the files into the
# destination's own benchmark bucket.

EXPORT_KIND = "gpuplatform.benchmark.export"
# Never embed control/secret artifacts (SSH keys, raw config) in the export.
_EXPORT_SKIP_FILES = {"vm_key", "vm_key.pub", "rp_key", "rp_key.pub", "config.yaml"}
_EXPORT_PER_FILE_CAP = 25 * 1024 * 1024   # 25 MiB per file
_EXPORT_TOTAL_CAP = 50 * 1024 * 1024      # 50 MiB total embedded


class ImportBenchmarkData(BaseModel):
    name: str
    config_yaml: str = ""
    status: str = "done"
    exit_code: Optional[int] = None
    error_text: Optional[str] = None
    result_json: Optional[dict] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    cost_per_hr: Optional[float] = None
    env_vars: Optional[dict] = None
    visible_devices: Optional[str] = None
    # Resolved hardware from the export (top-level in the export JSON since the
    # source's provider doesn't travel). Stored on the imported row so GPU
    # grouping survives the round-trip.
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None


class ImportBenchmarkFile(BaseModel):
    name: str          # path relative to the benchmark's S3 prefix
    content_b64: str


class ImportBenchmarkBody(BaseModel):
    kind: str
    version: int = 1
    source_bench_id: Optional[str] = None
    benchmark: ImportBenchmarkData
    files: list[ImportBenchmarkFile] = []


class TemplateRecord(BaseModel):
    id: str
    name: str
    config_yaml: str
    created_at: str


class CreateTemplateRequest(BaseModel):
    name: str
    config_yaml: str


# ---------- HTTP API ----------------------------------------------------


router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


def _to_record(
    b: Benchmark, owner_username: str, is_owner: Optional[bool] = None
) -> BenchmarkRecord:
    # Effective GPU identity: manual row value first, else the config's
    # runpod.pod / top-level gpu_type (same YAML parse _stats does per row).
    # No provider fallback here — that needs a DB lookup (_bench_gpu_meta),
    # and _to_record must stay sync/session-free.
    gpu_type = getattr(b, "gpu_type", None)
    gpu_count = getattr(b, "gpu_count", None)
    if not gpu_type:
        parsed = _parse_config(b.config_yaml or "")
        gpu_type = parsed.get("gpu_type")
        gpu_count = gpu_count or (parsed.get("gpu_count") if gpu_type else None)
    return BenchmarkRecord(
        id=b.id,
        name=b.name,
        status=b.status,
        s3_prefix=b.s3_prefix,
        config_yaml=b.config_yaml,
        exit_code=b.exit_code,
        error_text=b.error_text,
        result_json=b.result_json,
        created_by=owner_username,
        is_public=bool(getattr(b, "is_public", False)),
        is_owner=is_owner,
        created_at=b.created_at.isoformat() if b.created_at else "",
        started_at=b.started_at.isoformat() if b.started_at else None,
        ended_at=b.ended_at.isoformat() if b.ended_at else None,
        cost_per_hr=b.cost_per_hr,
        provider_id=b.provider_id,
        storage_id=b.storage_id,
        env_vars=getattr(b, "env_vars", None) or None,
        visible_devices=getattr(b, "visible_devices", None) or None,
        hf_token_secret=getattr(b, "hf_token_secret", None) or None,
        api_key_secret=getattr(b, "api_key_secret", None) or None,
        cleanup_model=getattr(b, "cleanup_model", None),
        gpu_type=gpu_type,
        gpu_count=gpu_count,
    )


def _can_read(b: Benchmark, user: User) -> bool:
    """Read access: the owner, an admin, or anyone when the run is public."""
    return user.is_admin or b.owner_id == user.id or bool(getattr(b, "is_public", False))


# Raw per-request arrays that benchmaq dumps into result_json (one entry per prompt
# × the full ITL trace). They dominate the blob — `itls` + `generated_texts` alone
# run ~6 MB/run — yet the list/card view only reads scalar throughput + the small
# `accuracy` summary. Stripping them from the LIST response cuts ~66 MB → ~0.3 MB,
# so the page loads fast AND serializing it stops blocking the asyncio event loop
# (the big synchronous json encode was stalling the loop long enough to trip the
# k8s /health + /ready probes → pod restarts). The detail / results / compare pages
# fetch each run's full result_json separately, so nothing user-facing loses data.
_LIST_HEAVY_KEYS = frozenset({
    "itls", "tpots", "e2els", "ttfts", "start_times", "input_lens", "output_lens",
    "errors", "generated_texts", "prompt_lens", "generated_token_ids",
    "output_token_ids", "input_texts", "prompts",
})


def _slim_result_json(rj: Optional[dict]) -> Optional[dict]:
    """A list-view copy of result_json with the heavy per-request arrays removed.
    Keeps every scalar metric + the `accuracy` summary (drives the card chips)."""
    if not isinstance(rj, dict):
        return rj
    out: dict = {}
    for k, v in rj.items():
        if k == "accuracy":
            out[k] = v  # small summary the avg-accuracy chip needs — always keep
        elif k in _LIST_HEAVY_KEYS:
            continue
        elif isinstance(v, list) and len(v) > 64:
            continue  # backstop for any other unanticipated per-request array
        else:
            out[k] = v
    return out


# ---------- Templates --------------------------------------------------
# These come BEFORE the /benchmarks/{id}/* routes so /benchmarks/templates
# isn't captured by the {bench_id} path parameter.


@router.get("/templates", response_model=list[TemplateRecord])
async def list_templates(
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    rows = await session.execute(
        select(BenchmarkTemplate)
        .where(BenchmarkTemplate.owner_id == user.id)
        .order_by(BenchmarkTemplate.created_at.desc())
    )
    return [
        TemplateRecord(
            id=t.id,
            name=t.name,
            config_yaml=t.config_yaml,
            created_at=t.created_at.isoformat() if t.created_at else "",
        )
        for t in rows.scalars().all()
    ]


@router.post("/templates", response_model=TemplateRecord)
async def create_template(
    body: CreateTemplateRequest,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    # Validate the YAML at least parses — saving garbage helps no one.
    try:
        cfg = yaml.safe_load(body.config_yaml) or {}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail={"error": f"invalid YAML: {e}"})
    if not isinstance(cfg, dict):
        raise HTTPException(status_code=400, detail={"error": "top-level YAML must be a mapping"})

    import uuid
    t = BenchmarkTemplate(
        id=f"tpl-{uuid.uuid4().hex[:8]}",
        name=body.name.strip()[:128] or "untitled",
        config_yaml=body.config_yaml,
        owner_id=user.id,
    )
    session.add(t)
    await session.commit()
    return TemplateRecord(
        id=t.id, name=t.name, config_yaml=t.config_yaml,
        created_at=t.created_at.isoformat() if t.created_at else "",
    )


@router.delete("/templates/{template_id}")
async def delete_template(
    template_id: str,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    t = await session.get(BenchmarkTemplate, template_id)
    if not t:
        raise HTTPException(status_code=404, detail={"error": "template not found"})
    if not user.is_admin and t.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    await session.delete(t)
    await session.commit()
    return {"ok": True, "id": template_id}


# ---------- Benchmarks -------------------------------------------------


@router.post("", response_model=BenchmarkRecord)
async def create_benchmark(
    body: CreateBenchmarkRequest,
    request: Request,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    try:
        cfg = yaml.safe_load(body.config_yaml) or {}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail={"error": f"invalid YAML: {e}"})
    if not isinstance(cfg, dict):
        raise HTTPException(status_code=400, detail={"error": "top-level YAML must be a mapping"})

    prov_kind: Optional[str] = None
    if body.provider_id:
        from .db import Provider
        prov = await session.get(Provider, body.provider_id)
        if prov is None:
            raise HTTPException(status_code=400, detail={"error": "unknown provider_id"})
        prov_kind = prov.kind

    # Cloud-disabled deployments (CAE/CCE): a benchmark with no provider — or a
    # runpod/pi one — defaults to the `benchmaq runpod bench` cloud path and
    # spawns a pod. Refuse at submit time so the user gets a clear 403 instead of
    # a cryptic failure deep inside the benchmaq subprocess (no RUNPOD_API_KEY).
    # Ingress runs provision nothing (no pod / no cloud spend), so the cloud-
    # disabled gate doesn't apply — it only guards the spawn-a-pod path. Detect
    # the same way the runner does: a base_url in the config + no machine provider.
    is_ingress = (not body.provider_id) and (_ingress_base_url(body.config_yaml) is not None)
    if is_ingress:
        # SSRF guard: the ingress base_url is fetched server-side (and non-200
        # bodies land in the run log) — block link-local/metadata + bad schemes.
        from .netsafe import assert_safe_fetch_url
        try:
            assert_safe_fetch_url(_ingress_base_url(body.config_yaml))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"unsafe ingress base_url: {e}")
    from .provider import ensure_benchmark_provider_allowed, CloudProviderDisabled
    try:
        if not is_ingress:
            ensure_benchmark_provider_allowed(prov_kind)
    except CloudProviderDisabled:
        raise HTTPException(
            status_code=403,
            detail={"error": "cloud GPU providers are disabled on this deployment — "
                             "register and select a physical 'vm' provider for this benchmark"},
        )

    # Resolve the storage backend for logs + result files. Precedence:
    #   1. the explicit storage_id field (web form path), else
    #   2. a `storage:` key on a benchmark item inside config_yaml — a backend
    #      id or name — so a full YAML config (pasted in, or POSTed via the API)
    #      can name its own s3 storage without the separate field, else
    #   3. the env bucket (API/back-compat).
    storage_id = body.storage_id
    if not storage_id:
        ref = None
        for item in cfg.get("benchmark") or []:
            if isinstance(item, dict):
                sv = item.get("storage")
                if isinstance(sv, str) and sv.strip():
                    ref = sv.strip()
                    break
        if ref:
            st = await session.get(Storage, ref)  # try id first
            if st is None:
                # fall back to a name lookup, scoped to the caller's storages.
                res = await session.execute(
                    select(Storage).where(
                        Storage.owner_id == user.id, Storage.name == ref
                    )
                )
                st = res.scalars().first()
            if st is None:
                raise HTTPException(
                    status_code=400,
                    detail={"error": f"unknown storage in config: {ref!r}"},
                )
            storage_id = st.id

    if storage_id:
        st = await session.get(Storage, storage_id)
        if st is None:
            raise HTTPException(status_code=400, detail={"error": "unknown storage_id"})
        if st.kind != "s3":
            raise HTTPException(status_code=400, detail={"error": "storage must be kind=s3 for benchmark logs"})
        if not st.enabled:
            raise HTTPException(status_code=400, detail={"error": "selected storage is disabled"})
        target = _target_from_storage_row(st)
    else:
        target = _env_s3_target()

    bench_id = _gen_id()
    s3_prefix = benchmark_s3_prefix(bench_id, target)

    bench = Benchmark(
        id=bench_id,
        name=body.name,
        config_yaml=body.config_yaml,
        status="queued",
        s3_prefix=s3_prefix,
        owner_id=user.id,
        provider_id=body.provider_id,
        storage_id=storage_id,
        is_public=bool(body.is_public),
        # Only honoured when provider_id is set (VM path). Default True.
        cleanup_model=True if body.cleanup_model is None else bool(body.cleanup_model),
        env_vars={k: str(v) for k, v in (body.env_vars or {}).items()} or None,
        visible_devices=(body.visible_devices or "").strip() or None,
        hf_token_secret=(body.hf_token_secret or "").strip() or None,
        api_key_secret=(body.api_key_secret or "").strip() or None,
        gpu_type=(body.gpu_type or "").strip()[:64] or None,
        gpu_count=int(body.gpu_count) if body.gpu_count and body.gpu_count > 0 else None,
    )
    session.add(bench)
    await session.commit()

    # Kick off the runner. We MUST keep a strong reference to the task —
    # asyncio's docs warn that "tasks can be garbage-collected mid-execution"
    # if the only ref is the loop's weakref. _active_runners is also what
    # the janitor uses to tell stuck-in-DB rows apart from in-flight ones.
    redis = request.app.state.redis
    task = asyncio.create_task(_safe_run(redis, bench_id, body.config_yaml))
    _active_runners[bench_id] = task
    task.add_done_callback(lambda _t, _bid=bench_id: _active_runners.pop(_bid, None))

    await audit.record(user, "benchmark.create", "benchmark", bench_id, body.name)

    bench = await session.get(Benchmark, bench_id)
    return _to_record(bench, user.username)


async def _safe_run(redis, bench_id: str, raw_yaml: str) -> None:
    try:
        await run_benchmark(redis, bench_id, raw_yaml)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("benchmark %s crashed: %s", bench_id, e)
        async with session_factory()() as s:
            b = await s.get(Benchmark, bench_id)
            if b and b.status not in ("done", "failed", "cancelled"):
                b.status = "failed"
                b.error_text = f"runner crashed: {e}"[:4000]
                b.ended_at = datetime.now(timezone.utc)
                await s.commit()


@router.get("", response_model=list[BenchmarkRecord])
async def list_benchmarks(
    scope: str = "mine",
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    # Admins default to their own runs; pass ?scope=all to see everyone's.
    # Everyone (incl. non-admins) additionally sees public runs shared by others.
    show_all = user.is_admin and scope == "all"
    if show_all:
        rows = await session.execute(select(Benchmark).order_by(Benchmark.created_at.desc()))
    else:
        rows = await session.execute(
            select(Benchmark)
            .where((Benchmark.owner_id == user.id) | (Benchmark.is_public.is_(True)))
            .order_by(Benchmark.created_at.desc())
        )
    benches = rows.scalars().all()
    # Resolve all owner usernames in ONE query (was a session.get per row → an N+1
    # that made the list scale linearly in DB round-trips).
    owner_ids = {b.owner_id for b in benches}
    names: dict[str, str] = {}
    if owner_ids:
        urows = await session.execute(
            select(User.id, User.username).where(User.id.in_(owner_ids))
        )
        names = {uid: uname for uid, uname in urows.all()}
    out: list[BenchmarkRecord] = []
    for b in benches:
        rec = _to_record(b, names.get(b.owner_id, ""), is_owner=b.owner_id == user.id)
        # The list ships ~66 MB of unused per-request arrays otherwise; slim it so the
        # response is small + the json encode doesn't block the event loop.
        rec.result_json = _slim_result_json(rec.result_json)
        out.append(rec)
    return out


# ---------- Paged list + slim stats (lazy pagination) -------------------
# With hundreds of runs, the full list above is megabytes and the web page pays
# for all of it up front. The web list now fetches /_page per page (server-side
# search/filter/sort so results cover ALL runs, not just loaded ones), and the
# dashboard KPIs read /_stats — a few numbers per run instead of full records.
# Both are declared BEFORE /{bench_id} (Starlette matches in declaration order);
# the plain list endpoint stays for back-compat (external API users).


class BenchmarkPageResponse(BaseModel):
    total: int
    items: list[BenchmarkRecord]


def _bench_scope_stmt(user: User, scope: str):
    stmt = select(Benchmark)
    if not (user.is_admin and scope == "all"):
        stmt = stmt.where((Benchmark.owner_id == user.id) | (Benchmark.is_public.is_(True)))
    return stmt


@router.get("/_page", response_model=BenchmarkPageResponse)
async def list_benchmarks_page(
    scope: str = "mine",
    q: str = "",
    status: str = "",
    sort: str = "newest",
    limit: int = Query(12, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    stmt = _bench_scope_stmt(user, scope)
    if status:
        stmt = stmt.where(Benchmark.status == status)
    # Multi-token search (every token must match), mirroring the old client-side
    # filter: name/id/status + owner username + the config YAML text (which is
    # where model / gpu_type / tp / dp live — it's a plain String column).
    for tok in (q or "").lower().split():
        like = f"%{tok}%"
        stmt = stmt.where(
            or_(
                Benchmark.id.ilike(like),
                Benchmark.name.ilike(like),
                Benchmark.status.ilike(like),
                Benchmark.config_yaml.ilike(like),
                Benchmark.owner_id.in_(select(User.id).where(User.username.ilike(like))),
            )
        )
    total = (
        await session.execute(select(func.count()).select_from(stmt.subquery()))
    ).scalar_one()
    order = Benchmark.created_at.asc() if sort == "oldest" else Benchmark.created_at.desc()
    rows = await session.execute(stmt.order_by(order).limit(limit).offset(offset))
    benches = rows.scalars().all()
    owner_ids = {b.owner_id for b in benches}
    names: dict[str, str] = {}
    if owner_ids:
        urows = await session.execute(select(User.id, User.username).where(User.id.in_(owner_ids)))
        names = {uid: uname for uid, uname in urows.all()}
    items: list[BenchmarkRecord] = []
    for b in benches:
        rec = _to_record(b, names.get(b.owner_id, ""), is_owner=b.owner_id == user.id)
        rec.result_json = _slim_result_json(rec.result_json)
        items.append(rec)
    return BenchmarkPageResponse(total=total, items=items)


class BenchStat(BaseModel):
    """The handful of numbers the dashboard KPI row needs, per run."""

    id: str
    status: str
    model: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = None
    output_throughput: Optional[float] = None
    duration_s: Optional[float] = None


@router.get("/_stats", response_model=list[BenchStat])
async def benchmark_stats(
    scope: str = "mine",
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    bench_rows = (await session.execute(_bench_scope_stmt(user, scope))).scalars().all()
    # Prefetch every bound provider in ONE query instead of a session-open +
    # Provider.get per row (an N+1 that checked out a pooled connection each time).
    prov_ids = {b.provider_id for b in bench_rows if b.provider_id}
    prov_cache: dict = {}
    if prov_ids:
        from .db import Provider as _Provider
        for _p in (await session.execute(select(_Provider).where(_Provider.id.in_(prov_ids)))).scalars().all():
            prov_cache[_p.id] = _p
    out: list[BenchStat] = []
    for b in bench_rows:
        meta = await _bench_gpu_meta(b, prov_cache)
        rj = b.result_json if isinstance(b.result_json, dict) else {}
        tput = rj.get("output_throughput")
        dur = (
            (b.ended_at - b.started_at).total_seconds()
            if b.started_at and b.ended_at
            else None
        )
        out.append(
            BenchStat(
                id=b.id,
                status=b.status,
                model=meta.get("model"),
                gpu_type=meta.get("gpu_type"),
                gpu_count=meta.get("gpu_count"),
                output_throughput=tput if isinstance(tput, (int, float)) else None,
                duration_s=dur,
            )
        )
    return out


# ---------- Aggregate (cross-benchmark dashboard) ----------------------
# Defined BEFORE /{bench_id} so the literal path "/_aggregate" isn't captured
# by the path-param matcher. (FastAPI/Starlette match in declaration order.)


class AggregatePoint(BaseModel):
    benchmark_id: str
    benchmark_name: str
    model: str | None = None
    gpu_type: str | None = None
    gpu_count: int = 1
    engine: str = "vllm"
    tp: int = 1
    dp: int = 1
    context_len: int = 0
    output_len: int = 0
    concurrency: int = 0
    num_prompts: int = 0
    duration_s: float | None = None
    output_throughput: float | None = None
    output_throughput_per_gpu: float | None = None
    request_throughput: float | None = None
    median_ttft_ms: float | None = None
    p99_ttft_ms: float | None = None
    median_tpot_ms: float | None = None
    p99_tpot_ms: float | None = None
    median_itl_ms: float | None = None
    median_e2el_ms: float | None = None
    p99_e2el_ms: float | None = None


_AGG_CACHE: dict[str, tuple[float, list[AggregatePoint]]] = {}
_AGG_TTL_S = 600.0          # in-process freshness window (was 60s — expired constantly, forcing cold S3 rebuilds)
_AGG_REDIS_TTL_S = 3600     # shared Redis copy lives longer: survives a gateway restart and backs a cold in-process cache
_AGG_WARM_MARGIN_S = 90.0   # background warmer re-warms each key this long before its in-process entry goes stale
# cache_key -> (show_all, owner_id). Populated by aggregate() on every request so the
# background warmer knows exactly which cache entries to keep hot.
_AGG_WARM: dict[str, tuple[bool, int | None]] = {}


def _safe_num(d: dict, k: str) -> float | None:
    v = d.get(k)
    if isinstance(v, (int, float)) and v == v:
        return float(v)
    return None


def _parse_dims_from_filename(name: str) -> dict:
    import re
    base = name.split("/")[-1]
    out = {"context_len": 0, "output_len": 0, "num_prompts": 0, "concurrency": 0}
    m = re.search(r"_in(\d+)_out(\d+)_p(\d+)_c(\d+)", base)
    if m:
        out["context_len"] = int(m.group(1))
        out["output_len"] = int(m.group(2))
        out["num_prompts"] = int(m.group(3))
        out["concurrency"] = int(m.group(4))
    return out


def _safe_int(v, default: int = 1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _parse_config(yaml_text: str) -> dict:
    try:
        cfg = yaml.safe_load(yaml_text) or {}
    except Exception:
        return {}
    if not isinstance(cfg, dict):
        return {}
    pod = ((cfg.get("runpod") or {}).get("pod") or {})
    benches = cfg.get("benchmark") or []
    first = benches[0] if isinstance(benches, list) and benches else {}
    if not isinstance(first, dict):
        first = {}
    serve = (first.get("serve") or {})
    # GPU identity: runpod.pod for cloud runs; ingress/Slurm configs have no pod
    # block, so also honor a top-level (or first-benchmark-item) gpu_type /
    # gpu_count key — same YAML-author ergonomics as base_url/storage.
    gpu_type = pod.get("gpu_type") or cfg.get("gpu_type") or first.get("gpu_type")
    return {
        "gpu_type": gpu_type,
        "gpu_count": _safe_int(
            pod.get("gpu_count") or cfg.get("gpu_count") or first.get("gpu_count") or 1
        ),
        "engine": first.get("engine") or "vllm",
        "model": ((first.get("model") or {}).get("repo_id")),
        "tp": _safe_int(serve.get("tensor_parallel_size") or 1),
        "dp": _safe_int(serve.get("data_parallel_size") or 1),
    }


async def _bench_gpu_meta(b: "Benchmark", prov_cache: Optional[dict] = None) -> dict:
    """Resolve a benchmark's hardware/serve metadata: ``{gpu_type, gpu_count,
    model, tp, dp, engine}``. RunPod runs carry ``gpu_type`` in the YAML's
    ``runpod.pod`` block; VM (bare-metal) runs don't, so fall back to the bound
    provider's stored GPU info (populated by its last Test/availability probe),
    e.g. ``"H20-3e (tm-2)"``.

    ``prov_cache`` (provider_id → Provider) lets a caller resolving MANY rows
    prefetch every provider in one ``IN`` query and avoid the per-row
    session-open + query (an N+1 — one pooled-connection checkout per benchmark).
    Omit it and this opens its own session, so it's still safe to call standalone
    from concurrent gather() contexts."""
    meta = _parse_config(b.config_yaml or "")
    # Manual row values (set at create / via PATCH — ingress and Slurm runs)
    # override whatever the config says.
    if getattr(b, "gpu_type", None):
        meta["gpu_type"] = b.gpu_type
    if getattr(b, "gpu_count", None):
        meta["gpu_count"] = int(b.gpu_count)
    if b.provider_id and not meta.get("gpu_type"):
        try:
            if prov_cache is not None:
                prov = prov_cache.get(b.provider_id)
            else:
                async with session_factory()() as _s:
                    from .db import Provider as _Provider
                    prov = await _s.get(_Provider, b.provider_id)
            if prov is not None:
                pcfg = prov.config or {}
                gpus_list = pcfg.get("gpus") or []
                if isinstance(gpus_list, list) and gpus_list:
                    # Just the GPU model (e.g. "H20-3e") — no provider/VM-name suffix.
                    meta["gpu_type"] = str(gpus_list[0]).replace("NVIDIA ", "").strip()
                meta["gpu_count"] = int(pcfg.get("gpu_count") or meta.get("gpu_count") or 1)
        except Exception as e:
            logger.warning("gpu meta: provider lookup for %s failed: %s", b.id, e)
    return meta


@router.post("/_compact")
async def compact_result_json(
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    """Admin one-time cleanup: rewrite every benchmark's result_json through the
    slimmer, dropping the heavy per-request arrays (itls / generated_texts / …) that
    bloat the column. Those arrays already live in each run's S3 per-config files, so
    nothing is lost — this just compacts rows finalized BEFORE result_json was slimmed
    at the source. Idempotent: already-slim rows are skipped. Route name leads with
    `_` so it's matched before the /{bench_id} path param."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="admin only")
    rows = (await session.execute(select(Benchmark))).scalars().all()
    scanned = compacted = 0
    before = after = 0
    for b in rows:
        scanned += 1
        rj = b.result_json
        if not isinstance(rj, dict):
            continue
        slim = _slim_result_json(rj)
        if slim is None or slim.keys() == rj.keys():
            continue  # nothing dropped → already slim (cheap key-set check, no serialize)
        before += len(json.dumps(rj, default=str))
        after += len(json.dumps(slim, default=str))
        b.result_json = slim
        compacted += 1
    await session.commit()
    logger.info("compact: scanned=%d compacted=%d saved=%.1fMB", scanned, compacted, (before - after) / 1e6)
    return {
        "scanned": scanned,
        "compacted": compacted,
        "bytes_before": before,
        "bytes_after": after,
        "saved_mb": round((before - after) / 1e6, 1),
    }


async def _build_aggregate(
    session: AsyncSession, *, show_all: bool, owner_id: int | None
) -> list[AggregatePoint]:
    """Build the benchmark aggregate points by fanning out over each benchmark's S3
    results. No caching and no request/user context, so both the HTTP endpoint and the
    background warmer can call it. `show_all` mirrors admin + scope=all (every
    benchmark); otherwise it's scoped to `owner_id`."""
    if show_all:
        rows = await session.execute(
            select(Benchmark).where(Benchmark.status.in_(["done", "running", "failed"]))
        )
    else:
        rows = await session.execute(
            select(Benchmark)
            .where(Benchmark.owner_id == owner_id)
            .where(Benchmark.status.in_(["done", "running", "failed"]))
        )
    benches = list(rows.scalars().all())

    async def fetch_one(b: Benchmark) -> list[AggregatePoint]:
        # Config's runpod.pod gpu_type, else the bound VM provider's stored GPU
        # (e.g. "L40S (TM-VM1)") so the Performance explorer labels series
        # instead of showing "—". Shared with the export endpoint.
        cfg_meta = await _bench_gpu_meta(b)
        try:
            # Resolve this bench's storage so aggregates read from wherever its
            # results actually live — each bench may use a different backend.
            target = await _bench_s3_target(b.storage_id)
            cli = _s3_client(target)
            bucket = target.bucket
            keys = []
            token = None
            while True:
                kwargs = {"Bucket": bucket, "Prefix": b.s3_prefix}
                if token: kwargs["ContinuationToken"] = token
                r = await asyncio.to_thread(cli.list_objects_v2, **kwargs)
                for obj in r.get("Contents", []):
                    k = obj["Key"]
                    if k.lower().endswith(".json") and not k.endswith("_DONE"):
                        keys.append(k)
                if not r.get("IsTruncated"): break
                token = r.get("NextContinuationToken")
        except Exception as e:
            logger.warning("aggregate: list %s failed: %s", b.id, e)
            return []

        async def fetch_json(key: str) -> AggregatePoint | None:
            try:
                obj = await asyncio.to_thread(cli.get_object, Bucket=bucket, Key=key)
                body = obj["Body"].read()
                data = json.loads(body)
            except Exception:
                return None
            if not isinstance(data, dict):
                return None
            dims = _parse_dims_from_filename(key)
            gpu_count = cfg_meta.get("gpu_count") or 1
            tput = _safe_num(data, "output_throughput")
            return AggregatePoint(
                benchmark_id=b.id,
                benchmark_name=b.name,
                model=cfg_meta.get("model"),
                gpu_type=cfg_meta.get("gpu_type"),
                gpu_count=gpu_count,
                engine=cfg_meta.get("engine") or "vllm",
                tp=cfg_meta.get("tp") or 1,
                dp=cfg_meta.get("dp") or 1,
                context_len=dims["context_len"] or int(_safe_num(data, "random_input_len") or 0),
                output_len=dims["output_len"] or int(_safe_num(data, "random_output_len") or 0),
                concurrency=dims["concurrency"] or int(_safe_num(data, "max_concurrency") or 0),
                num_prompts=dims["num_prompts"] or int(_safe_num(data, "num_prompts") or 0),
                duration_s=_safe_num(data, "duration"),
                output_throughput=tput,
                output_throughput_per_gpu=(tput / gpu_count) if (tput and gpu_count) else None,
                request_throughput=_safe_num(data, "request_throughput"),
                median_ttft_ms=_safe_num(data, "median_ttft_ms"),
                p99_ttft_ms=_safe_num(data, "p99_ttft_ms"),
                median_tpot_ms=_safe_num(data, "median_tpot_ms"),
                p99_tpot_ms=_safe_num(data, "p99_tpot_ms"),
                median_itl_ms=_safe_num(data, "median_itl_ms"),
                median_e2el_ms=_safe_num(data, "median_e2el_ms"),
                p99_e2el_ms=_safe_num(data, "p99_e2el_ms"),
            )

        results = await asyncio.gather(*[fetch_json(k) for k in keys])
        return [p for p in results if p is not None]

    nested = await asyncio.gather(*[fetch_one(b) for b in benches])
    return [p for sub in nested for p in sub]


async def _get_or_build_aggregate(
    session: AsyncSession,
    redis,
    cache_key: str,
    *,
    show_all: bool,
    owner_id: int | None,
    force: bool = False,
) -> list[AggregatePoint]:
    """Return aggregate points for `cache_key` using the in-process cache, then the
    shared Redis copy (survives a gateway restart), then a fresh S3 build as a last
    resort. `force` skips the read caches (used by the warmer and the client Refresh)."""
    now = time.time()
    if not force:
        cached = _AGG_CACHE.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]
        # In-process cold/expired (e.g. right after a restart) — try the shared Redis
        # copy before paying for a full S3 rebuild.
        if redis is not None:
            try:
                raw = await redis.get(f"agg:{cache_key}")
            except Exception:
                raw = None
            if raw:
                try:
                    pts = [AggregatePoint(**d) for d in json.loads(raw)]
                    _AGG_CACHE[cache_key] = (now + _AGG_TTL_S, pts)
                    return pts
                except Exception:
                    logger.warning("aggregate: unreadable redis cache for %s; rebuilding", cache_key)

    pts = await _build_aggregate(session, show_all=show_all, owner_id=owner_id)
    _AGG_CACHE[cache_key] = (time.time() + _AGG_TTL_S, pts)
    if redis is not None:
        try:
            await redis.set(
                f"agg:{cache_key}",
                json.dumps([p.model_dump() for p in pts]),
                ex=_AGG_REDIS_TTL_S,
            )
        except Exception:
            logger.warning("aggregate: redis write failed for %s (non-fatal)", cache_key, exc_info=True)
    return pts


@router.get("/_aggregate", response_model=list[AggregatePoint])
async def aggregate(
    request: Request,
    scope: str = "mine",
    refresh: bool = False,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    show_all = user.is_admin and scope == "all"
    cache_key = "admin-all" if show_all else f"u{user.id}"
    owner_id = None if show_all else user.id
    # Register this key so the background warmer keeps its cache entry hot.
    _AGG_WARM[cache_key] = (show_all, owner_id)
    redis = getattr(request.app.state, "redis", None)
    # refresh=1 (the calculator's Refresh button) rebuilds from S3 now instead of
    # serving the cache — so a just-finished benchmark shows up immediately.
    return await _get_or_build_aggregate(
        session, redis, cache_key, show_all=show_all, owner_id=owner_id, force=refresh,
    )


async def aggregate_warmer_loop(session_maker, redis) -> None:
    """Keep the _aggregate cache hot so no user ever waits on a cold S3 rebuild. Every
    ~(_AGG_TTL_S - margin) it rebuilds each key that has actually been requested
    (tracked in _AGG_WARM) and refreshes both the in-process cache and Redis. Read-only
    w.r.t. the platform: reads benchmarks + S3, writes only the agg:* cache keys."""
    interval = max(30.0, _AGG_TTL_S - _AGG_WARM_MARGIN_S)
    while True:
        try:
            await asyncio.sleep(interval)
            targets = list(_AGG_WARM.items())
            if not targets:
                continue
            async with session_maker() as session:
                for cache_key, (show_all, owner_id) in targets:
                    try:
                        await _get_or_build_aggregate(
                            session, redis, cache_key,
                            show_all=show_all, owner_id=owner_id, force=True,
                        )
                    except Exception:
                        logger.warning("aggregate warmer: rebuild %s failed", cache_key, exc_info=True)
            logger.info("aggregate warmer: refreshed %d cache key(s)", len(targets))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("aggregate warmer: loop iteration failed")


@router.get("/{bench_id}", response_model=BenchmarkRecord)
async def get_benchmark(
    bench_id: str,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    b = await session.get(Benchmark, bench_id)
    if not b:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    if not _can_read(b, user):
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    owner = await session.get(User, b.owner_id)
    return _to_record(b, owner.username if owner else "", is_owner=b.owner_id == user.id)


@router.patch("/{bench_id}", response_model=BenchmarkRecord)
async def update_benchmark(
    bench_id: str,
    body: UpdateBenchmarkRequest,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    """Update mutable metadata: rename, and/or set the manual GPU identity (for
    ingress/Slurm runs the platform didn't provision — nothing to derive hardware
    from otherwise). Owner (or admin) only. Cosmetic — the run, S3 prefix, and
    config are untouched."""
    b = await session.get(Benchmark, bench_id)
    if not b:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    if not user.is_admin and b.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    changes: dict = {}
    if body.name is not None:
        name = (body.name or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail={"error": "name must not be empty"})
        name = name[:200]
        if name != b.name:
            changes["name"] = {"from": b.name, "to": name}
            b.name = name
    if body.gpu_type is not None:
        # "" clears the manual value (back to config-derived).
        gpu_type = body.gpu_type.strip()[:64] or None
        if gpu_type != b.gpu_type:
            changes["gpu_type"] = {"from": b.gpu_type, "to": gpu_type}
            b.gpu_type = gpu_type
    if body.gpu_count is not None:
        # 0 (or negative) clears the manual value.
        gpu_count = int(body.gpu_count) if body.gpu_count > 0 else None
        if gpu_count != b.gpu_count:
            changes["gpu_count"] = {"from": b.gpu_count, "to": gpu_count}
            b.gpu_count = gpu_count
    if changes:
        await session.commit()
        # Keep the pre-existing rename event name so audit trails stay consistent;
        # anything else lands under benchmark.update.
        action = "benchmark.rename" if set(changes) == {"name"} else "benchmark.update"
        await audit.record(
            user, action, "benchmark", bench_id, b.name,
            details=changes["name"] if action == "benchmark.rename" else {"changes": changes},
        )
    owner = await session.get(User, b.owner_id)
    return _to_record(b, owner.username if owner else "", is_owner=b.owner_id == user.id)


class VisibilityRequest(BaseModel):
    is_public: bool


@router.post("/{bench_id}/visibility", response_model=BenchmarkRecord)
async def set_benchmark_visibility(
    bench_id: str,
    body: VisibilityRequest,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    """Make a benchmark public (visible read-only to everyone) or private again.
    Owner (or admin) only."""
    b = await session.get(Benchmark, bench_id)
    if not b:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    if not user.is_admin and b.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    b.is_public = bool(body.is_public)
    await session.commit()
    await audit.record(
        user, "benchmark.visibility", "benchmark", bench_id, b.name,
        details={"is_public": b.is_public},
    )
    owner = await session.get(User, b.owner_id)
    return _to_record(b, owner.username if owner else "", is_owner=b.owner_id == user.id)


@router.get("/{bench_id}/logs")
async def get_benchmark_logs(
    bench_id: str,
    request: Request,
    tail: int = 200,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    """Plain (non-streaming) log fetch for scripting/polling — returns the last
    `tail` lines as JSON. Source priority mirrors the stream endpoint: on-disk
    full log → S3 logs.txt (terminal runs) → Redis live-tail."""
    b = await session.get(Benchmark, bench_id)
    if not b:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    if not _can_read(b, user):
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    n = max(1, min(int(tail or 200), 5000))

    lines: list[str] = []
    full_log = _full_log_path(bench_id)
    if full_log.exists():
        try:
            lines = full_log.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            lines = []
    if not lines and b.status in ("done", "failed", "cancelled"):
        _target = await _bench_s3_target(b.storage_id)
        txt = s3_get_text(f"{b.s3_prefix}logs.txt", target=_target)
        if txt:
            lines = txt.splitlines()
    if not lines:
        try:
            lines = await request.app.state.redis.lrange(f"bench:logs:{bench_id}", 0, -1)
        except Exception:
            lines = []

    return {
        "bench_id": bench_id,
        "status": b.status,
        "error_text": b.error_text,
        "total_lines": len(lines),
        "lines": lines[-n:],
    }


@router.post("/{bench_id}/duplicate", response_model=BenchmarkRecord)
async def duplicate_benchmark(
    bench_id: str,
    request: Request,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    """Re-run: create a fresh benchmark copying this one's config + provider +
    storage + env/GPU settings, and start it. The copy is owned by the caller."""
    src = await session.get(Benchmark, bench_id)
    if not src:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    # Re-run mints a new run owned by the caller, so reading a public run to copy
    # its config is fine — the original is never mutated.
    if not _can_read(src, user):
        raise HTTPException(status_code=403, detail={"error": "forbidden"})

    target = await _bench_s3_target(src.storage_id)
    new_id = _gen_id()
    new = Benchmark(
        id=new_id,
        name=f"{src.name}-copy",
        config_yaml=src.config_yaml,
        status="queued",
        s3_prefix=benchmark_s3_prefix(new_id, target),
        owner_id=user.id,
        provider_id=src.provider_id,
        storage_id=src.storage_id,
        cleanup_model=bool(getattr(src, "cleanup_model", True)),
        env_vars=getattr(src, "env_vars", None),
        visible_devices=getattr(src, "visible_devices", None),
        gpu_type=getattr(src, "gpu_type", None),
        gpu_count=getattr(src, "gpu_count", None),
    )
    session.add(new)
    await session.commit()

    redis = request.app.state.redis
    task = asyncio.create_task(_safe_run(redis, new_id, src.config_yaml))
    _active_runners[new_id] = task
    task.add_done_callback(lambda _t, _bid=new_id: _active_runners.pop(_bid, None))

    await audit.record(user, "benchmark.create", "benchmark", new_id, new.name)
    logger.info("benchmark %s duplicated from %s by user=%s", new_id, bench_id, user.username)
    new = await session.get(Benchmark, new_id)
    return _to_record(new, user.username)


# Strong refs to terminate-cleanup tasks so they don't get GC'd mid-flight.
_active_terminations: dict[str, asyncio.Task] = {}


@router.post("/{bench_id}/terminate")
async def terminate_benchmark(
    bench_id: str,
    request: Request,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    """Stop a running benchmark: cancel the runner task, kill the local
    subprocess, mark the row `cancelled`, then run cleanup in the background
    (SSH-pkill remote bench procs, rm the VM model dir, terminate any RunPod
    pod). Returns immediately; cleanup progress is appended to the bench log."""
    b = await session.get(Benchmark, bench_id)
    if not b:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    if not user.is_admin and b.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    if b.status in ("done", "failed", "cancelled"):
        raise HTTPException(status_code=409, detail={"error": f"benchmark already {b.status}"})

    redis = request.app.state.redis
    provider_id = b.provider_id
    raw_yaml = b.config_yaml
    cleanup_model_flag = b.cleanup_model
    runpod_pod_id = b.runpod_pod_id
    bench_name = b.name

    # Figure out the provider kind so cleanup knows whether to SSH (vm) or
    # only tear down a runpod pod (runpod). A None provider_id implies legacy
    # env-based RunPod cred (no per-row override).
    provider_kind: Optional[str] = None
    if provider_id:
        from .db import Provider
        prov = await session.get(Provider, provider_id)
        if prov is not None:
            provider_kind = prov.kind

    await _push_log(redis, bench_id, "[gateway] terminate requested")

    # Cancel the runner task — its CancelledError handler kills the local
    # subprocess, which closes the SSH channel and SIGHUPs the remote bash.
    task = _active_runners.get(bench_id)
    if task and not task.done():
        task.cancel()

    # Safety net: hard-kill the local subprocess if the task isn't tracked
    # (e.g. orphan from before a gateway restart that left the row running).
    proc = _LIVE.pop(bench_id, None)
    if proc and proc.returncode is None:
        try:
            proc.kill()
        except Exception:
            pass

    # Mark cancelled now so the UI flips immediately. Cleanup runs async.
    b.status = "cancelled"
    b.exit_code = -1
    if b.ended_at is None:
        b.ended_at = datetime.now(timezone.utc)
    await session.commit()

    async def _cleanup():
        # SSH-side cleanup for VM benches: kill any survivors + remove model.
        if provider_kind == "vm":
            work = _work_dir(bench_id)
            try:
                vm_target = await _materialise_vm_key(work, provider_id)
            except Exception as e:
                await _push_log(redis, bench_id, f"[gateway] terminate: vm key materialise failed: {e}")
                vm_target = None
            if vm_target is not None:
                await _push_log(redis, bench_id, "[gateway] terminate: killing remote bench processes")
                try:
                    ok, msg = await asyncio.to_thread(_ssh_kill_bench_procs_sync, vm_target)
                    level = "info" if ok else "warning"
                    await _push_log(redis, bench_id, f"[gateway] terminate [{level}]: {msg}")
                except Exception as e:
                    await _push_log(redis, bench_id, f"[gateway] terminate: pkill failed: {e}")
                if cleanup_model_flag:
                    try:
                        await _cleanup_vm_model(redis, bench_id, vm_target, raw_yaml)
                    except Exception as e:
                        await _push_log(redis, bench_id, f"[gateway] terminate: model cleanup failed: {e}")

        # RunPod pod teardown — only when benchmaq spawned a pod itself.
        # Use the per-row provider_id (when it's a runpod-kind one) so we
        # hit the same account benchmaq used to spawn the pod.
        if runpod_pod_id:
            pid_for_teardown = provider_id if provider_kind == "runpod" else None
            try:
                await _terminate_runpod_pod(runpod_pod_id, provider_id=pid_for_teardown)
                await _push_log(redis, bench_id, f"[gateway] terminate: runpod pod {runpod_pod_id} torn down")
            except Exception as e:
                await _push_log(redis, bench_id, f"[gateway] terminate: runpod teardown failed: {e}")

        # Upload the final log to S3 so the cancelled row stays viewable.
        try:
            full = _full_log_path(bench_id)
            if full.exists():
                async with session_factory()() as s:
                    _row = await s.get(Benchmark, bench_id)
                _target = await _bench_s3_target(_row.storage_id if _row else None)
                _prefix = _row.s3_prefix if _row else benchmark_s3_prefix(bench_id, _target)
                s3_put_file(f"{_prefix}logs.txt", str(full), target=_target)
        except Exception as e:
            await _push_log(redis, bench_id, f"[gateway] terminate: s3 log upload failed: {e}")

    cleanup_task = asyncio.create_task(_cleanup())
    _active_terminations[bench_id] = cleanup_task
    cleanup_task.add_done_callback(lambda _t, _bid=bench_id: _active_terminations.pop(_bid, None))

    await audit.record(user, "benchmark.terminate", "benchmark", bench_id, bench_name)
    return {"ok": True, "id": bench_id, "status": "cancelled"}


@router.delete("/{bench_id}")
async def delete_benchmark(
    bench_id: str,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    b = await session.get(Benchmark, bench_id)
    if not b:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    if not user.is_admin and b.owner_id != user.id:
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    proc = _LIVE.pop(bench_id, None)
    if proc and proc.returncode is None:
        try:
            proc.kill()
        except Exception:
            pass
    bench_name = b.name
    # Snapshot pod/provider before the row is gone: deleting a still-running
    # cloud bench (Delete instead of Terminate) otherwise leaves its RunPod pod
    # billing with no DB record left to drive a teardown.
    runpod_pod_id = b.runpod_pod_id
    del_provider_id = b.provider_id
    del_provider_kind: Optional[str] = None
    if del_provider_id:
        from .db import Provider
        _prov = await session.get(Provider, del_provider_id)
        if _prov is not None:
            del_provider_kind = _prov.kind
    # Snapshot billing inputs before the row is gone. If the user deletes a
    # bench that's still running, ended_at will be None and the audit helper
    # treats "now" as the end — giving us a "spent so far at deletion" total.
    cost = audit.cost_breakdown(b.started_at, b.ended_at, b.cost_per_hr)
    await session.delete(b)
    await session.commit()
    # Tear the pod down AFTER the row is committed-gone (off the request session)
    # in a tracked background task — same teardown Terminate runs. Best-effort.
    if runpod_pod_id:
        pid_for_teardown = del_provider_id if del_provider_kind == "runpod" else None

        async def _delete_teardown():
            try:
                await _terminate_runpod_pod(runpod_pod_id, provider_id=pid_for_teardown)
            except Exception as e:  # noqa: BLE001 — best-effort; row is already gone
                logger.warning("benchmark %s: delete-time pod %s teardown failed: %s",
                               bench_id, runpod_pod_id, e)

        _t = asyncio.create_task(_delete_teardown())
        _active_terminations[bench_id] = _t
        _t.add_done_callback(lambda _tt, _bid=bench_id: _active_terminations.pop(_bid, None))
    await audit.record(
        user, "benchmark.delete", "benchmark", bench_id, bench_name,
        details=cost,
    )
    return {"ok": True, "id": bench_id}


@router.get("/{bench_id}/logs/stream")
async def stream_logs(
    bench_id: str,
    request: Request,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    b = await session.get(Benchmark, bench_id)
    if not b:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    if not _can_read(b, user):
        raise HTTPException(status_code=403, detail={"error": "forbidden"})

    redis = request.app.state.redis
    initial_status = b.status
    # Source-of-truth for logs is the on-disk _full.log (uncapped) while the
    # bench is live, and S3 logs.txt once it's been uploaded. Redis is only a
    # last-resort fallback for benches that ran before the on-disk tee landed.
    s3_full_log: Optional[str] = None
    if initial_status in ("done", "failed", "cancelled"):
        _target = await _bench_s3_target(b.storage_id)
        s3_full_log = s3_get_text(f"{b.s3_prefix}logs.txt", target=_target)
    full_log = _full_log_path(bench_id)

    async def gen() -> AsyncIterator[bytes]:
        # 1) Terminal + S3 has it → stream the canonical copy and close.
        if s3_full_log is not None:
            for line in s3_full_log.splitlines():
                yield f"data: {line}\n\n".encode("utf-8")
            yield f"event: end\ndata: {initial_status}\n\n".encode("utf-8")
            return

        # 2) Terminal but no S3 copy and no on-disk file → legacy bench, fall
        # back to whatever redis still has (will be trimmed, but it's all we've
        # got). Newer terminal benches always have one of the above.
        if initial_status in ("done", "failed", "cancelled") and not full_log.exists():
            key = f"bench:logs:{bench_id}"
            try:
                lines = await redis.lrange(key, 0, -1)
            except Exception:
                lines = []
            for line in lines:
                yield f"data: {line}\n\n".encode("utf-8")
            yield f"event: end\ndata: {initial_status}\n\n".encode("utf-8")
            return

        # 3) Live or recently-terminal: tail _full.log from disk. This is the
        # uncapped, canonical record — the file is appended to on every
        # _push_log call, so we just keep reading from where we left off.
        pos = 0
        buf = b""
        while True:
            chunk = b""
            if full_log.exists():
                try:
                    with full_log.open("rb") as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos += len(chunk)
                except Exception:
                    chunk = b""
            if chunk:
                buf += chunk
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line = buf[:nl].decode("utf-8", "replace")
                    buf = buf[nl + 1:]
                    yield f"data: {line}\n\n".encode("utf-8")
                continue
            # No new bytes — check whether the run finished.
            async with session_factory()() as s:
                cur = await s.get(Benchmark, bench_id)
            if cur and cur.status in ("done", "failed", "cancelled"):
                # Flush any trailing partial line (run ended mid-write).
                if buf:
                    yield f"data: {buf.decode('utf-8', 'replace')}\n\n".encode("utf-8")
                    buf = b""
                yield f"event: end\ndata: {cur.status}\n\n".encode("utf-8")
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )



@router.get("/{bench_id}/files", response_model=list[FileRecord])
async def list_files(
    bench_id: str,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    b = await session.get(Benchmark, bench_id)
    if not b:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    if not _can_read(b, user):
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    target = await _bench_s3_target(b.storage_id)
    # No bucket resolvable here (this deployment has no BENCHMARK_S3_BUCKET and the
    # run's storage row is null/deleted) → there's nothing we can list. Degrade to
    # an empty list rather than 500ing the results/files tabs.
    if not target.bucket:
        return []
    items = s3_list(b.s3_prefix, target=target)
    out: list[FileRecord] = []
    for it in items:
        rel = it["key"][len(b.s3_prefix):] if it["key"].startswith(b.s3_prefix) else it["key"]
        out.append(FileRecord(
            name=rel or it["key"],
            size=it["size"],
            modified=it["modified"],
            download_url=s3_presign_get(it["key"], target=target),
        ))
    return out


@router.get("/{bench_id}/files/content")
async def get_file_content(
    bench_id: str,
    path: str,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    """Serve a result file's bytes THROUGH the gateway (same-origin, authed)
    instead of a presigned S3 URL. The browser can't fetch presigned S3 URLs
    cross-origin unless the bucket has a CORS policy for the web origin — this
    endpoint sidesteps that so the results/files tabs work regardless of bucket
    CORS. `path` is relative to the benchmark's S3 prefix."""
    b = await session.get(Benchmark, bench_id)
    if not b:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    if not _can_read(b, user):
        raise HTTPException(status_code=403, detail={"error": "forbidden"})
    rel = (path or "").lstrip("/")
    if not rel or ".." in rel:
        raise HTTPException(status_code=400, detail={"error": "invalid path"})
    target = await _bench_s3_target(b.storage_id)
    if not target.bucket:
        # No resolvable bucket on this deployment → the file isn't reachable here.
        raise HTTPException(status_code=404, detail={"error": "file not found"})
    txt = s3_get_text(f"{b.s3_prefix}{rel}", target=target)
    if txt is None:
        raise HTTPException(status_code=404, detail={"error": "file not found"})
    media = "application/json" if rel.endswith(".json") else "text/plain; charset=utf-8"
    return Response(content=txt, media_type=media)


@router.get("/{bench_id}/export")
async def export_benchmark(
    bench_id: str,
    include_files: bool = True,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    """Download a finished benchmark as a self-contained JSON: the DB row's
    results + config, plus the run's S3 artifacts inlined as base64 (so the
    destination needs no access to this instance's bucket). Pair with the
    /benchmarks/import endpoint on another deployment."""
    import base64

    b = await session.get(Benchmark, bench_id)
    if not b:
        raise HTTPException(status_code=404, detail={"error": "benchmark not found"})
    if not _can_read(b, user):
        raise HTTPException(status_code=403, detail={"error": "forbidden"})

    files: list[dict] = []
    omitted: list[dict] = []
    # No bucket resolvable here (no BENCHMARK_S3_BUCKET + null/deleted storage row)
    # → export the metadata without inlined files instead of 500ing.
    target = await _bench_s3_target(b.storage_id) if include_files else None
    if include_files and target and target.bucket:
        total = 0
        for it in s3_list(b.s3_prefix, target=target):
            key = it["key"]
            rel = key[len(b.s3_prefix):] if key.startswith(b.s3_prefix) else key
            base = rel.rsplit("/", 1)[-1]
            if not rel or base in _EXPORT_SKIP_FILES:
                continue
            size = int(it.get("size") or 0)
            if size > _EXPORT_PER_FILE_CAP or total + size > _EXPORT_TOTAL_CAP:
                omitted.append({"name": rel, "size": size, "reason": "exceeds export size cap"})
                continue
            data = s3_get_bytes(key, target=target)
            if data is None:
                omitted.append({"name": rel, "size": size, "reason": "unreadable"})
                continue
            total += len(data)
            files.append({"name": rel, "content_b64": base64.b64encode(data).decode("ascii")})

    # Resolve the GPU/serve metadata (config's runpod.pod block, else the VM
    # provider's stored GPU info) so the export is self-describing — the
    # importing deployment has no access to this instance's providers.
    gpu_meta = await _bench_gpu_meta(b)
    # Bake the resolved GPU into the EXPORTED config (not the original row) under
    # the canonical runpod.pod block that _parse_config reads. VM runs resolve
    # their GPU from the provider, which doesn't travel — so without this the
    # importer (which drops provider_id) shows a blank GPU. Baking it into
    # config_yaml means GPU survives the round-trip regardless of the importer's
    # code version, since every deployment derives GPU from config_yaml.
    export_config_yaml = b.config_yaml or ""
    if gpu_meta.get("gpu_type"):
        try:
            cfg = yaml.safe_load(export_config_yaml) or {}
            if isinstance(cfg, dict):
                pod = cfg.setdefault("runpod", {}).setdefault("pod", {})
                if not pod.get("gpu_type"):
                    pod["gpu_type"] = gpu_meta["gpu_type"]
                    pod.setdefault("gpu_count", gpu_meta.get("gpu_count") or 1)
                    export_config_yaml = yaml.safe_dump(cfg, sort_keys=False)
        except Exception as e:
            logger.warning("export %s: could not bake gpu into config: %s", b.id, e)
    export = {
        "kind": EXPORT_KIND,
        "version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_bench_id": b.id,
        "benchmark": {
            "name": b.name,
            "config_yaml": export_config_yaml,
            "status": b.status,
            "exit_code": b.exit_code,
            "error_text": b.error_text,
            "result_json": b.result_json,
            "created_at": b.created_at.isoformat() if b.created_at else None,
            "started_at": b.started_at.isoformat() if b.started_at else None,
            "ended_at": b.ended_at.isoformat() if b.ended_at else None,
            "cost_per_hr": b.cost_per_hr,
            "env_vars": b.env_vars,
            "visible_devices": b.visible_devices,
            # Hardware + serve shape (resolved, not just what's in the YAML).
            "gpu_type": gpu_meta.get("gpu_type"),
            "gpu_count": gpu_meta.get("gpu_count") or 1,
            "model": gpu_meta.get("model"),
            "engine": gpu_meta.get("engine") or "vllm",
            "tensor_parallel_size": gpu_meta.get("tp") or 1,
            "data_parallel_size": gpu_meta.get("dp") or 1,
        },
        "files": files,
        "files_omitted": omitted,
    }
    await audit.record(
        user, "benchmark.export", "benchmark", b.id, b.name,
        details={"files": len(files), "omitted": len(omitted)},
    )
    return Response(
        content=json.dumps(export),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{b.id}.benchmark.json"'},
    )


@router.post("/import", response_model=BenchmarkRecord)
async def import_benchmark(
    body: ImportBenchmarkBody,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    """Re-create a benchmark exported from another deployment. Mints a fresh id,
    owns it as the importer, writes any embedded files into THIS deployment's
    benchmark bucket, and stores results/config so the dashboard renders fully.
    Instance-specific fields (provider/storage/secret) are intentionally dropped."""
    import base64

    if body.kind != EXPORT_KIND:
        raise HTTPException(
            status_code=400,
            detail={"error": f"not a benchmark export (kind={body.kind!r})"},
        )
    data = body.benchmark
    new_id = _gen_id()
    target = _env_s3_target()  # write into this deployment's own bucket
    prefix = benchmark_s3_prefix(new_id, target)

    written = 0
    for f in body.files:
        rel = (f.name or "").lstrip("/")
        if not rel or ".." in rel:
            continue
        try:
            raw = base64.b64decode(f.content_b64)
        except Exception:
            continue
        try:
            s3_put_bytes(f"{prefix}{rel}", raw, target=target)
            written += 1
        except Exception as e:
            logger.warning("import %s: failed to write %s: %s", new_id, rel, e)

    def _parse(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

    b = Benchmark(
        id=new_id,
        name=(data.name or "imported")[:128],
        config_yaml=data.config_yaml or "",
        status=data.status or "done",
        s3_prefix=prefix,
        exit_code=data.exit_code,
        error_text=data.error_text,
        result_json=data.result_json,
        owner_id=user.id,
        created_at=_parse(data.created_at) or datetime.now(timezone.utc),
        started_at=_parse(data.started_at),
        ended_at=_parse(data.ended_at),
        cost_per_hr=data.cost_per_hr,
        provider_id=None,
        storage_id=None,
        env_vars=data.env_vars,
        visible_devices=data.visible_devices,
        hf_token_secret=None,
        api_key_secret=None,
        gpu_type=(data.gpu_type or "").strip()[:64] or None,
        gpu_count=data.gpu_count if (data.gpu_count or 0) > 0 else None,
    )
    session.add(b)
    await session.commit()
    await audit.record(
        user, "benchmark.import", "benchmark", new_id, b.name,
        details={"source_bench_id": body.source_bench_id, "files_written": written},
    )
    return _to_record(b, user.username)


# ---------- Public share (no-auth comparison links) ---------------------


class CreateShareBody(BaseModel):
    ids: list[str]
    notes: str = ""
    # Frozen accuracy→speed pairing (accId→speedId) for the public IQ-vs-speed chart.
    pairing: dict[str, str] = {}


class ShareResponse(BaseModel):
    token: str


@router.post("/share", response_model=ShareResponse)
async def create_share(
    body: CreateShareBody,
    user: User = Depends(require_section("benchmark")),
    session: AsyncSession = Depends(get_session),
):
    """Mint a public comparison link for a set of benchmarks the caller can see.
    Only explicitly-shared comparisons become world-readable (via the token)."""
    import uuid

    ids = [i for i in (body.ids or []) if i]
    if not ids:
        raise HTTPException(status_code=400, detail={"error": "no benchmark ids"})
    for bid in ids:
        b = await session.get(Benchmark, bid)
        if b is None:
            raise HTTPException(status_code=404, detail={"error": f"benchmark {bid} not found"})
        if not user.is_admin and b.owner_id != user.id:
            raise HTTPException(status_code=403, detail={"error": f"forbidden: {bid}"})
    token = "cmp_" + uuid.uuid4().hex[:16]
    session.add(BenchmarkShare(token=token, bench_ids=ids, notes=(body.notes or ""),
                               pairing=(body.pairing or {}), owner_id=user.id))
    await session.commit()
    await audit.record(user, "benchmark.share", "benchmark", token, ",".join(ids), details={"ids": ids})
    return ShareResponse(token=token)


@router.get("/public-compare/{token}")
async def public_compare(
    token: str,
    session: AsyncSession = Depends(get_session),
):
    """PUBLIC (no auth): resolve a share token → a self-contained comparison
    payload (per-bench record + per-config result rows read from S3). Instance/
    secret fields (provider, storage, env vars, hf token) are NOT included."""
    share = await session.get(BenchmarkShare, token)
    if share is None:
        raise HTTPException(status_code=404, detail={"error": "share link not found"})
    out: list[dict] = []
    for bid in (share.bench_ids or []):
        b = await session.get(Benchmark, bid)
        if b is None:
            continue
        rows: list = []
        try:
            target = await _bench_s3_target(b.storage_id)
            # s3_get_text is blocking boto3 — this is an UNAUTHENTICATED route, so a
            # sync call here lets anyone stall the whole gateway event loop.
            agg = await asyncio.to_thread(lambda: s3_get_text(f"{b.s3_prefix}result.json", target=target))
            if agg:
                rows = json.loads(agg).get("results") or []
        except Exception:
            rows = []
        meta = await _bench_gpu_meta(b)
        out.append({
            "id": b.id,
            "name": b.name,
            "status": b.status,
            "config_yaml": b.config_yaml,
            "result_json": b.result_json,
            "result_rows": rows,
            # Resolved hardware (manual value → config → provider) so the public
            # page / external consumers don't have to parse config_yaml.
            "gpu_type": meta.get("gpu_type"),
            "gpu_count": meta.get("gpu_count") or 1,
        })
    return {"token": token, "notes": share.notes or "", "pairing": share.pairing or {}, "benchmarks": out}
