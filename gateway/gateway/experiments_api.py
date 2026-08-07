"""Experiments — agent observability + behavioural stress testing.

The generic half of every stress-test study ever hand-written against a served
model. Those scripts are ~90% identical: load a captured request, replay it N
times across some endpoints and prompt variants at concurrency C, assemble the
stream (minding that Dynamo says `reasoning_content` while plain vLLM says
`reasoning`), classify each completion, tally, print a table. Only the classifier
differs — and those live in `evaluators.py`.

So the model here is:

    Dataset (the platform's own, /datasets)
         │  rows carrying a `messages` column → Cases
         └── Experiment = dataset × targets × variants × repeats
                  │
                  └── ExperimentSample = one completion + its evaluator verdicts

Experiments deliberately has **no dataset store of its own**. A corpus of captured
requests is a dataset like any other, so it lives in the Datasets section — where
it can be browsed, published, packed, or reused by anything else — and
`resolve_cases()` reads rows out of it through that section's own readers.

A run's `summary_json` holds one **cell** per (target, variant) with each
evaluator's pass rate plus latency/token/cost stats — that's the matrix the
tradeoff plot draws, and the thing you actually compare across experiments.

Design notes worth keeping:

- **Targets are always plain `{base_url, model, key}`**, even when they point at
  this platform's own apps or proxy endpoints. `GET /targets` prefills those from
  the app/proxy registry so it's one click in the UI, but the runner has exactly
  one code path and a third-party endpoint is a first-class target. This mirrors
  the `stt_callback` convention in `proxy_api.py`.
- **Errors are data, not noise.** `retries` defaults to 1 (no retry): retrying is
  how you mask the very failure you're measuring — a fast 500 or a 0-token reply
  is itself the finding. The `request_error` evaluator is always on.
- **The runner is in-process** (asyncio + the shared httpx client): this is pure
  HTTP fan-out with no GPU, so there's no pod to provision. It's bounded by a
  worker pool and a hard unit cap; in-flight runs are marked failed on restart
  (`cleanup_orphaned_running`), same as quantization jobs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import statistics
import time
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    cast,
    delete as _delete,
    func,
    or_,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from . import crypto
from . import custom_eval as ce
from . import evaluators as ev
from . import langfuse_import as lf
from .auth import require_section
from .db import App, Base, Dataset, Request as ReqRow, Storage, User, get_session, session_factory
from .global_env_api import load_global_env
from urllib.parse import urlparse

logger = logging.getLogger("gateway.experiments")


def bench_s3_get_bytes(key: str, target):
    from .bench import s3_get_bytes
    return s3_get_bytes(key, target)


def _s3_put_bytes(key: str, body: bytes, target) -> None:
    """Write bytes to S3 via the same helper the dataset upload route uses."""
    import tempfile
    from .bench import s3_put_file
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(body)
        tmp.flush()
        s3_put_file(key, tmp.name, target=target)

SECTION = "experiments"

# Hard ceiling on units (rows × targets × variants × repeats) for one
# experiment. A 4-target × 3-variant × 200-repeat sweep over 20 cases is 48k
# calls — enough to bill real money and saturate an endpoint, so it needs an
# explicit ceiling rather than discovering it in production.
MAX_UNITS = int(os.environ.get("EXPERIMENT_MAX_UNITS", "20000") or "20000")
# Stored per-sample text is capped: a 200-repeat run of 8k-char replies is
# ~1.6MB of JSONB per cell otherwise.
MAX_STORED_CHARS = int(os.environ.get("EXPERIMENT_MAX_STORED_CHARS", "8000") or "8000")
DEFAULT_CONCURRENCY = 8
MAX_CONCURRENCY = 64
PROGRESS_FLUSH_S = 2.0

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,127}$")


# --------------------------------------------------------------------------- #
# DB models
# --------------------------------------------------------------------------- #


@dataclass
class Case:
    """One replayable request, resolved from a row of a platform Dataset.

    Deliberately NOT a table. Experiments used to own a parallel `eval_datasets`
    concept; cases now come from the Datasets section like every other corpus on
    the platform, so this is a transient view of a row, built per run.
    """
    id: str
    name: str
    messages: list[dict[str, Any]]
    tools: Optional[list[dict[str, Any]]] = None
    # Sampling parameters AS RECORDED at capture time — replaying with library
    # defaults reproduces a different request than the one that misbehaved.
    params: dict[str, Any] = dataclass_field(default_factory=dict)
    # Per-row expectations evaluators can read, e.g. {"json_keys": [...]}.
    expected: dict[str, Any] = dataclass_field(default_factory=dict)


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # exp-<hex8>
    name: Mapped[str] = mapped_column(String(128))
    dataset_id: Mapped[str] = mapped_column(String(64), index=True)
    # queued | running | completed | failed | cancelled
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    # {targets, variants, repeats, concurrency, retries, timeout_s, stream, evaluators, judge}
    # Target API keys are stored ENCRYPTED inside this blob, never in plaintext.
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    # {cells: [...], totals: {...}} — the comparison matrix.
    summary_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    n_planned: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    n_completed: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    n_failed: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    error_text: Mapped[Optional[str]] = mapped_column(String(4096), nullable=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Stamped by the runner on every progress flush. This is what makes the
    # restart cleanup safe under multi-replica HA: a row still being driven by
    # ANOTHER live replica has a fresh heartbeat, so this replica's startup
    # sweep leaves it alone instead of failing a healthy run.
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class CustomEvaluator(Base):
    """A user-written detector, reusable across experiments.

    The library entry is editable; an experiment **snapshots** the code into its
    own config at create time, so editing an evaluator later can never change
    what an already-finished run meant.
    """
    __tablename__ = "custom_evaluators"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # ce-<hex8>
    name: Mapped[str] = mapped_column(String(64), index=True)
    description: Mapped[str] = mapped_column(Text, default="", server_default="", nullable=False)
    # expression | python  (see custom_eval.MODES)
    mode: Mapped[str] = mapped_column(String(16), default="expression", server_default="expression", nullable=False)
    code: Mapped[str] = mapped_column(Text, default="", server_default="", nullable=False)
    # api mode only: {url, method, headers, auth_*, *_field, timeout_s, concurrency}.
    # No secret is stored here — `api_key_secret` names a global secret.
    config: Mapped[dict] = mapped_column(JSON, default=dict, server_default="{}", nullable=False)
    # True = the expression matching means FAIL (hunting a bug) rather than PASS.
    fail_when_true: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class ExperimentSample(Base):
    """One replay. High volume — deliberately narrow columns plus a capped text
    body, so a 20k-unit run stays a few hundred MB rather than multiple GB."""
    __tablename__ = "experiment_samples"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    case_name: Mapped[str] = mapped_column(String(255), default="", server_default="", nullable=False)
    target: Mapped[str] = mapped_column(String(128), default="", server_default="", nullable=False)
    variant: Mapped[str] = mapped_column(String(128), default="", server_default="", nullable=False)
    repeat: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true", nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", server_default="", nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, default="", server_default="", nullable=False)
    finish_reason: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ttft_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_text: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    # {evaluator_id: {passed, score, reason, flags}}
    evals_json: Mapped[dict] = mapped_column(JSON, default=dict, server_default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


# In-flight runner tasks, keyed by experiment id. Strong refs so the event loop
# doesn't garbage-collect a running task mid-flight; the cancel flag lets a
# cancel request unwind the worker pool cooperatively.
_RUNNERS: dict[str, asyncio.Task] = {}
_CANCELLED: set[str] = set()


# --------------------------------------------------------------------------- #
# API models
# --------------------------------------------------------------------------- #


class DatasetOption(BaseModel):
    """A platform Dataset that can supply replayable requests."""
    id: str
    name: str
    kind: str
    messages_field: Optional[str]
    num_rows: Optional[int]
    owner: str
    usable: bool
    reason: Optional[str] = None


class RowPreview(BaseModel):
    """One dataset row, as the request it will be replayed as."""
    id: str
    name: str
    n_messages: int
    n_tools: int
    prompt_chars: int
    params: dict[str, Any]


class TargetSpec(BaseModel):
    label: str
    base_url: str
    model: str
    # Either a global-secret name or an inline key (encrypted before storage).
    api_key_secret: Optional[str] = None
    api_key: Optional[str] = None
    # Merged into the request body for this target only.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # Optional: pin the OpenAI path (audio endpoints etc.). Default chat.
    path: str = "/v1/chat/completions"


class VariantSpec(BaseModel):
    """One mutation of the captured request. `label` names the column in the
    comparison matrix."""
    label: str = "baseline"
    params: dict[str, Any] = Field(default_factory=dict)
    # REPLACES the row's system message (prefix/suffix decorate it instead).
    # This is the axis prompt optimization writes into: a GEPA-optimized prompt
    # is a whole system prompt, so it has to replace, not wrap.
    system_override: str = ""
    system_prefix: str = ""
    system_suffix: str = ""
    user_suffix: str = ""
    # Seeds the assistant turn — the "force prefill" axis.
    assistant_prefill: str = ""
    # "json_object", or a full response_format object, or None.
    response_format: Optional[Any] = None
    strip_tools: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)


class EvaluatorSelection(BaseModel):
    id: str
    options: dict[str, Any] = Field(default_factory=dict)


class CreateExperimentRequest(BaseModel):
    name: str
    dataset_id: str
    targets: list[TargetSpec]
    variants: list[VariantSpec] = Field(default_factory=lambda: [VariantSpec()])
    evaluators: list[EvaluatorSelection] = Field(default_factory=list)
    repeats: int = 1
    concurrency: int = DEFAULT_CONCURRENCY
    # 1 = no retry. Retrying masks the failure being measured.
    retries: int = 1
    timeout_s: float = 300.0
    stream: bool = True
    # Cap rows pulled from the dataset (0 = all of them).
    max_rows: int = 0


class ExperimentRecord(BaseModel):
    id: str
    name: str
    dataset_id: str
    dataset_name: str
    status: str
    config: dict[str, Any]
    summary: Optional[dict[str, Any]]
    n_planned: int
    n_completed: int
    n_failed: int
    error_text: Optional[str]
    owner: str
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]


class ExperimentPage(BaseModel):
    total: int
    items: list[ExperimentRecord]


class SampleRecord(BaseModel):
    id: str
    case_id: str
    case_name: str
    target: str
    variant: str
    repeat: int
    passed: bool
    content: str
    reasoning: str
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency_ms: Optional[int]
    ttft_ms: Optional[int]
    status_code: Optional[int]
    error_text: Optional[str]
    evals: dict[str, Any]


class SamplePage(BaseModel):
    total: int
    items: list[SampleRecord]


class LangfusePreviewRequest(BaseModel):
    url: str
    base_url: Optional[str] = None
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    public_key_secret: Optional[str] = None
    secret_key_secret: Optional[str] = None




class CustomEvaluatorSpec(BaseModel):
    name: str
    description: str = ""
    mode: str = "expression"
    code: str = ""
    fail_when_true: bool = False
    config: dict[str, Any] = Field(default_factory=dict)


class CustomEvaluatorUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    mode: Optional[str] = None
    code: Optional[str] = None
    fail_when_true: Optional[bool] = None
    config: Optional[dict[str, Any]] = None


class CustomEvaluatorRecord(BaseModel):
    id: str
    name: str
    description: str
    mode: str
    code: str
    fail_when_true: bool
    config: dict[str, Any]
    owner: str
    created_at: datetime
    updated_at: Optional[datetime]


class TestCustomEvaluatorRequest(BaseModel):
    """Try an evaluator against a sample reply before committing to a run."""
    mode: str = "expression"
    code: str = ""
    fail_when_true: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    name: str = "preview"
    content: str = ""
    reasoning: str = ""
    finish_reason: Optional[str] = "stop"
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    ttft_ms: Optional[int] = None
    expected: dict[str, Any] = Field(default_factory=dict)
    # Optional: pull a real reply from a finished experiment instead of pasting one.
    sample_id: Optional[str] = None




# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(4)}"


def _clip(text: str, limit: int = MAX_STORED_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"


def _experiment_record(row: Experiment, owner: str, dataset_name: str) -> ExperimentRecord:
    return ExperimentRecord(
        id=row.id, name=row.name, dataset_id=row.dataset_id, dataset_name=dataset_name,
        status=row.status, config=_public_config(row.config_json or {}),
        summary=row.summary_json, n_planned=row.n_planned, n_completed=row.n_completed,
        n_failed=row.n_failed, error_text=row.error_text, owner=owner,
        created_at=row.created_at, started_at=row.started_at, ended_at=row.ended_at,
    )


def _public_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Strip encrypted key material before a config ever leaves the gateway.

    Deep-copied first — mutating the ORM row's JSON dict in place would mark it
    dirty and flush the redacted version back to the database.
    """
    out = json.loads(json.dumps(cfg, default=str))
    for t in out.get("targets") or []:
        if isinstance(t, dict):
            t["has_inline_key"] = bool(t.pop("api_key_enc", None))
    for sel in out.get("evaluators") or []:
        opts = (sel or {}).get("options")
        if isinstance(opts, dict):
            opts.pop("api_key", None)
            opts.pop("api_key_enc", None)
    return out


def _store_targets(specs: list[TargetSpec]) -> list[dict[str, Any]]:
    """Persist targets with any inline key encrypted at rest."""
    out: list[dict[str, Any]] = []
    for t in specs:
        d: dict[str, Any] = {
            "label": t.label.strip() or t.base_url,
            "base_url": t.base_url.strip().rstrip("/"),
            "model": t.model.strip(),
            "extra_body": t.extra_body or {},
            "path": t.path or "/v1/chat/completions",
        }
        ref = (t.api_key_secret or "").strip()
        if ref:
            d["api_key_secret"] = ref
        elif (t.api_key or "").strip():
            d["api_key_enc"] = crypto.encrypt(t.api_key.strip())
            d["has_inline_key"] = True
        out.append(d)
    return out


def _resolve_key(target: dict[str, Any], genv: dict[str, str]) -> str:
    ref = (target.get("api_key_secret") or "").strip()
    if ref:
        return genv.get(ref, "")
    enc = target.get("api_key_enc")
    if enc:
        try:
            return crypto.decrypt(enc)
        except Exception:
            return ""
    return ""


def _http(app) -> httpx.AsyncClient:
    """Shared client for experiment traffic. Separate from the proxy's client so a
    long stress run can't exhaust the connection pool serving live proxy traffic."""
    cli = getattr(app.state, "experiments_http", None)
    if cli is None:
        cli = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=None, write=None, pool=15.0),
            limits=httpx.Limits(max_connections=MAX_CONCURRENCY * 2,
                                max_keepalive_connections=MAX_CONCURRENCY),
            follow_redirects=True,
        )
        app.state.experiments_http = cli
    return cli


# --------------------------------------------------------------------------- #
# Request building — the variant axis
# --------------------------------------------------------------------------- #

# Params that belong in `extra_body` for an OpenAI-compatible vLLM server rather
# than as top-level fields.
_EXTRA_BODY_PARAMS = ("top_k", "repetition_penalty", "min_p", "length_penalty")
_TOP_LEVEL_PARAMS = (
    "temperature", "top_p", "max_tokens", "seed",
    "frequency_penalty", "presence_penalty", "n",
)


def build_request(
    case: Case,
    target: dict[str, Any],
    variant: dict[str, Any],
    stream: bool,
) -> dict[str, Any]:
    """Compose the OpenAI request body for one (case, target, variant) unit.

    Precedence: case-recorded params < variant params. The case's recorded
    sampling parameters are the baseline so a replay reproduces the captured
    request; a variant exists precisely to deviate from it.
    """
    messages = [dict(m) for m in (case.messages or [])]
    vparams = dict(variant.get("params") or {})
    params = {**(case.params or {}), **vparams}

    # -- prompt mutations -------------------------------------------------- #
    # Replacement first, decoration second, so a variant can do both: swap the
    # system prompt for an optimized one AND still pin a suffix on top of it.
    sys_override = variant.get("system_override") or ""
    if sys_override:
        idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
        if idx is None:
            messages.insert(0, {"role": "system", "content": sys_override})
        else:
            messages[idx] = {**messages[idx], "content": sys_override}

    sys_prefix = variant.get("system_prefix") or ""
    sys_suffix = variant.get("system_suffix") or ""
    if sys_prefix or sys_suffix:
        idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
        if idx is None:
            # No system turn to amend — insert one rather than silently dropping
            # the instruction the variant is meant to test.
            messages.insert(0, {"role": "system", "content": f"{sys_prefix}{sys_suffix}".strip()})
        else:
            body = str(messages[idx].get("content") or "")
            messages[idx] = {
                **messages[idx],
                "content": f"{sys_prefix}{body}{sys_suffix}" if sys_prefix or sys_suffix else body,
            }

    user_suffix = variant.get("user_suffix") or ""
    if user_suffix:
        idx = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
            None,
        )
        if idx is None:
            messages.append({"role": "user", "content": user_suffix})
        else:
            messages[idx] = {
                **messages[idx],
                "content": f"{messages[idx].get('content') or ''}{user_suffix}",
            }

    prefill = variant.get("assistant_prefill") or ""
    if prefill:
        # An assistant turn last = "continue this" — vLLM continues the final
        # assistant message when it isn't followed by a user turn.
        messages.append({"role": "assistant", "content": prefill})

    # -- body -------------------------------------------------------------- #
    body: dict[str, Any] = {"model": target.get("model") or "", "messages": messages}

    tools = None if variant.get("strip_tools") else (case.tools or None)
    if tools:
        body["tools"] = tools

    for key in _TOP_LEVEL_PARAMS:
        if params.get(key) is not None:
            body[key] = params[key]

    extra: dict[str, Any] = {}
    for key in _EXTRA_BODY_PARAMS:
        if params.get(key) is not None:
            extra[key] = params[key]
    if params.get("enable_thinking") is not None:
        extra["chat_template_kwargs"] = {"enable_thinking": bool(params["enable_thinking"])}
    extra.update(target.get("extra_body") or {})
    extra.update(variant.get("extra_body") or {})
    if extra:
        body.update(extra)

    rf = variant.get("response_format")
    if rf:
        body["response_format"] = {"type": rf} if isinstance(rf, str) else rf

    if stream:
        body["stream"] = True
        # Without this a streamed reply reports no usage at all, and every
        # token-derived metric (cost, empty-with-0-tokens) silently reads zero.
        body["stream_options"] = {"include_usage": True}
    return body


def _reasoning_of(obj: Any) -> str:
    """Read the reasoning field under either name.

    The two backends disagree — NVIDIA Dynamo emits `reasoning_content`, plain
    vLLM emits `reasoning`. Reading one silently reports zero reasoning on the
    other, which turns a 'reasoning-only empty' into a plain 'empty'.
    """
    if not isinstance(obj, dict):
        return ""
    for attr in ("reasoning_content", "reasoning"):
        val = obj.get(attr)
        if val:
            return str(val)
    return ""


# --------------------------------------------------------------------------- #
# The call
# --------------------------------------------------------------------------- #


async def call_once(
    client: httpx.AsyncClient,
    base_url: str,
    path: str,
    api_key: str,
    body: dict[str, Any],
    timeout: float,
) -> ev.Completion:
    """One replay. Never raises — a failed call is recorded as a sample.

    Broad exception handling is deliberate: an SSE read timeout arrives while
    *iterating* the stream, long after the request future resolved, and letting
    it escape kills the whole run rather than one unit.
    """
    url = f"{base_url.rstrip('/')}{path}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.perf_counter()
    stream = bool(body.get("stream"))
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    tool_calls: list[dict[str, Any]] = []
    ttft_ms: Optional[int] = None
    status_code: Optional[int] = None

    try:
        if not stream:
            resp = await client.post(url, headers=headers, json=body, timeout=timeout)
            status_code = resp.status_code
            if resp.status_code >= 400:
                return ev.Completion(
                    error=f"HTTP {resp.status_code}: {resp.text[:400]}",
                    status_code=resp.status_code,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            return ev.Completion(
                content=str(msg.get("content") or ""),
                reasoning=_reasoning_of(msg),
                finish_reason=choice.get("finish_reason"),
                usage=data.get("usage"),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                status_code=status_code,
                expected={"_tool_calls": msg.get("tool_calls") or []},
            )

        async with client.stream(
            "POST", url, headers=headers, json=body, timeout=timeout
        ) as resp:
            status_code = resp.status_code
            if resp.status_code >= 400:
                raw = (await resp.aread()).decode("utf-8", "replace")
                return ev.Completion(
                    error=f"HTTP {resp.status_code}: {raw[:400]}",
                    status_code=resp.status_code,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    if ttft_ms is None:
                        ttft_ms = int((time.perf_counter() - t0) * 1000)
                    content_parts.append(str(piece))
                rc = _reasoning_of(delta)
                if rc:
                    if ttft_ms is None:
                        ttft_ms = int((time.perf_counter() - t0) * 1000)
                    reasoning_parts.append(rc)
                for tc in delta.get("tool_calls") or []:
                    tool_calls.append(tc)
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
    except Exception as exc:  # noqa: BLE001 — see docstring
        return ev.Completion(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            error=f"{exc.__class__.__name__}: {exc}",
            status_code=status_code,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            ttft_ms=ttft_ms,
        )

    return ev.Completion(
        content="".join(content_parts),
        reasoning="".join(reasoning_parts),
        finish_reason=finish_reason,
        usage=usage,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        ttft_ms=ttft_ms,
        status_code=status_code,
        expected={"_tool_calls": _merge_stream_tool_calls(tool_calls)},
    )


def _merge_stream_tool_calls(fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reassemble streamed tool-call deltas (arguments arrive char-by-char)."""
    by_index: dict[int, dict[str, Any]] = {}
    for frag in fragments:
        idx = frag.get("index", 0)
        slot = by_index.setdefault(idx, {"id": frag.get("id"), "type": "function",
                                         "function": {"name": "", "arguments": ""}})
        fn = frag.get("function") or {}
        if fn.get("name"):
            slot["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            slot["function"]["arguments"] += fn["arguments"]
        if frag.get("id"):
            slot["id"] = frag["id"]
    return [by_index[k] for k in sorted(by_index)]


# --------------------------------------------------------------------------- #
# LLM judge
# --------------------------------------------------------------------------- #


async def _judge_sample(
    client: httpx.AsyncClient,
    judge: dict[str, Any],
    api_key: str,
    content: str,
    timeout: float,
) -> ev.EvalOutcome:
    """Ask the judge model for a PASS/FAIL verdict on one reply."""
    instruction = judge.get("prompt") or (
        "Answer PASS or FAIL, then one short sentence of justification."
    )
    body = {
        "model": judge.get("model") or "",
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"<reply>\n{content}\n</reply>"},
        ],
        "temperature": float(judge.get("temperature") or 0.0),
        "max_tokens": int(judge.get("max_tokens") or 256),
    }
    comp = await call_once(
        client, judge.get("base_url") or "", "/v1/chat/completions", api_key, body, timeout
    )
    if comp.error:
        # A judge outage must not be scored as a content failure.
        return ev.EvalOutcome(
            id="llm_judge", passed=True, score=None,
            reason=f"judge unavailable: {comp.error}",
            flags={"judge_error": True},
        )
    verdict = comp.content.strip()
    head = verdict[:200].upper()
    failed = "FAIL" in head and "PASS" not in head.split("FAIL")[0]
    return ev.EvalOutcome(
        id="llm_judge",
        passed=not failed,
        score=0.0 if failed else 1.0,
        reason=verdict[:300] if failed else None,
        flags={"verdict": verdict[:1000]},
    )


# --------------------------------------------------------------------------- #
# The evaluator stack — built-ins + custom evaluators + the judge
# --------------------------------------------------------------------------- #


class EvaluatorStack:
    """Everything that scores one completion, assembled once per run.

    Extracted from the experiment runner so prompt optimization
    (`prompt_opt_api.py`) grades a rollout with **exactly** the detectors an
    experiment would — same order, same short-circuit on transport errors, same
    one-child-process-per-run rule for python evaluators. Two scoring paths that
    drift apart would make an optimized prompt's reported gain unreproducible in
    the experiment that's supposed to confirm it.
    """

    def __init__(
        self,
        selections: list[dict[str, Any]],
        client: httpx.AsyncClient,
        genv: dict[str, str],
        *,
        default_target: Optional[dict[str, Any]] = None,
        default_key: str = "",
        timeout_s: float = 300.0,
    ) -> None:
        self.selections = selections or []
        self.client = client
        self.timeout_s = timeout_s

        # Custom evaluators come from the run's own SNAPSHOT, never the library
        # row — editing an entry must not change what a finished run measured.
        self.custom_specs = [
            ce.CustomSpec.from_dict(s["custom"]) for s in self.selections if s.get("custom")
        ]
        # One long-lived child per python evaluator for the whole run: a 10k-sample
        # run would otherwise be 10k process launches.
        self._py = {
            spec.name: ce.PythonEvaluatorWorker(spec)
            for spec in self.custom_specs
            if spec.mode == "python"
        }
        # api mode needs no sandbox — it's an outbound call, bounded so a slow
        # scorer can't consume the connections the replays need.
        self._api = {
            spec.name: ce.ApiEvaluatorClient(
                spec, client, genv.get((spec.config.get("api_key_secret") or "").strip(), ""),
            )
            for spec in self.custom_specs
            if spec.mode == "api"
        }

        judge_sel = next((s for s in self.selections if s.get("id") == "llm_judge"), None)
        opts = (judge_sel or {}).get("options") or {}
        self.judge: Optional[dict[str, Any]] = None
        self.judge_key = ""
        if judge_sel:
            base = default_target or {}
            self.judge = {
                "base_url": (opts.get("base_url") or base.get("base_url") or "").rstrip("/"),
                "model": opts.get("model") or base.get("model") or "",
                "prompt": opts.get("prompt"),
                "temperature": opts.get("temperature"),
                "max_tokens": opts.get("max_tokens"),
            }
            ref = (opts.get("api_key_secret") or "").strip()
            self.judge_key = genv.get(ref, "") if ref else default_key
        # The prefilter is what keeps judging affordable: only replies that look
        # suspicious are spent on a judge call.
        self._prefilters = ev.as_list(opts.get("prefilter_any")) if judge_sel else []
        self._judge_sem = asyncio.Semaphore(max(1, int(opts.get("concurrency") or 4)))

    @property
    def ids(self) -> list[str]:
        """Result keys, in report order. Custom evaluators are keyed by NAME."""
        return [
            (s["custom"]["name"] if s.get("custom") else s["id"]) for s in self.selections
        ] + list(ev.ALWAYS_ON)

    async def evaluate(self, comp: ev.Completion) -> tuple[list[ev.EvalOutcome], bool]:
        outcomes, passed = ev.run_evaluators(comp, self.selections)

        # A transport error already decided the verdict; running user code
        # against an empty string would only add noise.
        if self.custom_specs and not comp.error:
            for spec in self.custom_specs:
                if spec.mode == "python":
                    outcome = await self._py[spec.name].evaluate(comp)
                elif spec.mode == "api":
                    outcome = await self._api[spec.name].evaluate(comp)
                else:
                    outcome = ce.run_expression_evaluator(spec, comp)
                outcomes.append(outcome)
                passed = passed and outcome.passed

        if self.judge and not comp.error:
            text = comp.content
            if not self._prefilters or any(
                re.search(p, text, re.IGNORECASE) for p in self._prefilters
            ):
                async with self._judge_sem:
                    verdict = await _judge_sample(
                        self.client, self.judge, self.judge_key, text, self.timeout_s
                    )
                outcomes.append(verdict)
                passed = passed and verdict.passed

        return outcomes, passed

    async def close(self) -> None:
        """Never leave a child process behind, including on cancel/failure."""
        for worker in self._py.values():
            await worker.close()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _pct(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return round(ordered[k], 2)


_AGGREGATING = {eid for eid, spec in ev.SPECS.items() if spec.aggregate}


def summarize(samples: list[dict[str, Any]], evaluator_ids: list[str]) -> dict[str, Any]:
    """Fold samples into one cell per (target, variant).

    Each cell carries per-evaluator pass rates plus latency/token/cost stats —
    the axes of the tradeoff plot. Rates are computed over *completed* samples
    (errors counted separately) so a 50%-error run doesn't read as 50% clean.
    """
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for s in samples:
        key = (s["target"], s["variant"])
        cell = cells.setdefault(key, {
            "target": s["target"], "variant": s["variant"],
            "n": 0, "n_error": 0, "n_passed": 0,
            "latencies": [], "ttfts": [], "prompt_tokens": [], "completion_tokens": [],
            "costs": [], "evals": {},
        })
        cell["n"] += 1
        if s.get("error_text"):
            cell["n_error"] += 1
        if s.get("passed"):
            cell["n_passed"] += 1
        if s.get("latency_ms") is not None:
            cell["latencies"].append(float(s["latency_ms"]))
        if s.get("ttft_ms") is not None:
            cell["ttfts"].append(float(s["ttft_ms"]))
        if s.get("prompt_tokens") is not None:
            cell["prompt_tokens"].append(float(s["prompt_tokens"]))
        if s.get("completion_tokens") is not None:
            cell["completion_tokens"].append(float(s["completion_tokens"]))
        for eid, res in (s.get("evals") or {}).items():
            slot = cell["evals"].setdefault(
                eid, {"n": 0, "n_failed": 0, "scores": [], "flags": []}
            )
            slot["n"] += 1
            if not res.get("passed"):
                slot["n_failed"] += 1
            if res.get("score") is not None:
                slot["scores"].append(float(res["score"]))
            # Kept only long enough for the corpus-level rollup below; never stored.
            if eid in _AGGREGATING and isinstance(res.get("flags"), dict):
                slot["flags"].append(res["flags"])
            if eid == "cost":
                c = (res.get("flags") or {}).get("cost_usd")
                if c is not None:
                    cell["costs"].append(float(c))

    out_cells = []
    for cell in cells.values():
        n = cell["n"]
        n_ok = n - cell["n_error"]
        lat, ttft = cell["latencies"], cell["ttfts"]
        evals_out = {}
        for eid, slot in cell["evals"].items():
            entry = {
                "n": slot["n"],
                "n_failed": slot["n_failed"],
                "fail_rate": round(slot["n_failed"] / slot["n"], 4) if slot["n"] else 0.0,
                "pass_rate": round(1 - slot["n_failed"] / slot["n"], 4) if slot["n"] else 1.0,
                "mean_score": round(statistics.fmean(slot["scores"]), 4) if slot["scores"] else None,
            }
            # Detectors whose headline number is an F1 or per-class accuracy pool
            # their raw counts here — averaging per-sample rates would not
            # reproduce the benchmark's published figures.
            spec = ev.SPECS.get(eid)
            if spec is not None and spec.aggregate and slot["flags"]:
                try:
                    entry["metrics"] = spec.aggregate(slot["flags"])
                    entry["headline"] = list(spec.headline)
                except Exception:  # noqa: BLE001 — a bad rollup must not void the run
                    logger.exception("aggregate failed for evaluator %s", eid)
            evals_out[eid] = entry
        out_cells.append({
            "target": cell["target"],
            "variant": cell["variant"],
            "n": n,
            "n_error": cell["n_error"],
            "error_rate": round(cell["n_error"] / n, 4) if n else 0.0,
            "n_passed": cell["n_passed"],
            "pass_rate": round(cell["n_passed"] / n, 4) if n else 0.0,
            "latency_ms": {
                "mean": round(statistics.fmean(lat), 1) if lat else None,
                "p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95),
            },
            "ttft_ms": {
                "mean": round(statistics.fmean(ttft), 1) if ttft else None,
                "p50": _pct(ttft, 0.5), "p95": _pct(ttft, 0.95),
            },
            "prompt_tokens_mean": round(statistics.fmean(cell["prompt_tokens"]), 1)
                if cell["prompt_tokens"] else None,
            "completion_tokens_mean": round(statistics.fmean(cell["completion_tokens"]), 1)
                if cell["completion_tokens"] else None,
            "cost_usd_total": round(sum(cell["costs"]), 6) if cell["costs"] else None,
            "cost_usd_mean": round(statistics.fmean(cell["costs"]), 6) if cell["costs"] else None,
            "n_ok": n_ok,
            "evals": evals_out,
        })

    out_cells.sort(key=lambda c: (c["target"], c["variant"]))
    total = len(samples)
    n_err = sum(c["n_error"] for c in out_cells)
    return {
        "cells": out_cells,
        "evaluator_ids": evaluator_ids,
        "totals": {
            "n": total,
            "n_error": n_err,
            "n_passed": sum(c["n_passed"] for c in out_cells),
            "pass_rate": round(sum(c["n_passed"] for c in out_cells) / total, 4) if total else 0.0,
            "error_rate": round(n_err / total, 4) if total else 0.0,
        },
    }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


async def _run_experiment(app, experiment_id: str) -> None:
    """Execute one experiment end to end. Owns its own DB sessions (it outlives
    the request that created it)."""
    started = datetime.now(timezone.utc)
    try:
        async with session_factory()() as session:
            exp = await session.get(Experiment, experiment_id)
            if exp is None:
                return
            cfg = exp.config_json or {}
            ds = await session.get(Dataset, exp.dataset_id)
            if ds is None:
                raise RuntimeError(f"dataset {exp.dataset_id} no longer exists")
            # `max_cases` is the pre-rename key — kept so an older run still clones.
            max_rows = int(cfg.get("max_rows") or cfg.get("max_cases") or 0)
            cases = await resolve_cases(session, ds, limit=max_rows)
            genv = await load_global_env(session)
            exp.status = "running"
            exp.started_at = started
            await session.commit()

        if not cases:
            raise RuntimeError("no rows in that dataset are replayable")

        targets: list[dict[str, Any]] = cfg.get("targets") or []
        variants: list[dict[str, Any]] = cfg.get("variants") or [{"label": "baseline"}]
        repeats = max(1, int(cfg.get("repeats") or 1))
        concurrency = max(1, min(MAX_CONCURRENCY, int(cfg.get("concurrency") or DEFAULT_CONCURRENCY)))
        retries = max(1, int(cfg.get("retries") or 1))
        timeout_s = float(cfg.get("timeout_s") or 300.0)
        stream = bool(cfg.get("stream", True))
        selections: list[dict[str, Any]] = cfg.get("evaluators") or []

        keys = {t["label"]: _resolve_key(t, genv) for t in targets}
        client = _http(app)

        units: list[tuple[Case, dict, dict, int]] = [
            (case, target, variant, r)
            for case in cases
            for target in targets
            for variant in variants
            for r in range(repeats)
        ]
        if len(units) > MAX_UNITS:
            raise RuntimeError(
                f"{len(units)} units exceeds the {MAX_UNITS} cap "
                f"({len(cases)} rows × {len(targets)} targets × {len(variants)} variants "
                f"× {repeats} repeats). Lower repeats or trim the matrix."
            )

        # Judge setup defaults to the first target's endpoint/key, so the common
        # case ("judge with the same model") needs no extra configuration.
        first = targets[0] if targets else {}
        stack = EvaluatorStack(
            selections, client, genv,
            default_target=first,
            default_key=keys.get(first.get("label", ""), ""),
            timeout_s=timeout_s,
        )

        results: list[dict[str, Any]] = []
        pending_rows: list[ExperimentSample] = []
        counters = {"done": 0, "failed": 0}
        lock = asyncio.Lock()
        last_flush = time.monotonic()

        async def flush(force: bool = False) -> None:
            nonlocal last_flush, pending_rows
            if not pending_rows and not force:
                return
            if not force and (time.monotonic() - last_flush) < PROGRESS_FLUSH_S:
                return
            rows, pending_rows = pending_rows, []
            last_flush = time.monotonic()
            async with session_factory()() as s2:
                if rows:
                    s2.add_all(rows)
                exp2 = await s2.get(Experiment, experiment_id)
                if exp2 is not None:
                    exp2.n_completed = counters["done"]
                    exp2.n_failed = counters["failed"]
                    exp2.heartbeat_at = datetime.now(timezone.utc)
                await s2.commit()

        async def run_unit(unit: tuple[Case, dict, dict, int]) -> None:
            case, target, variant, repeat = unit
            if experiment_id in _CANCELLED:
                return
            body = build_request(case, target, variant, stream)
            comp = ev.Completion()
            for attempt in range(retries):
                comp = await call_once(
                    client, target["base_url"], target.get("path") or "/v1/chat/completions",
                    keys.get(target["label"], ""), body, timeout_s,
                )
                if not comp.error or attempt == retries - 1:
                    break
                await asyncio.sleep(min(2 ** attempt, 15))

            tool_calls = (comp.expected or {}).get("_tool_calls") or []
            # `_tools` lets the function-call detector resolve each call against
            # the schema the model actually saw (hallucination + param checks).
            comp.expected = {
                **(case.expected or {}),
                "_tool_calls": tool_calls,
                "_tools": case.tools or [],
            }

            outcomes, passed = await stack.evaluate(comp)

            evals = {
                o.id: {"passed": o.passed, "score": o.score, "reason": o.reason, "flags": o.flags}
                for o in outcomes
            }
            record = {
                "case_id": case.id, "case_name": case.name or case.id,
                "target": target["label"], "variant": variant.get("label") or "baseline",
                "repeat": repeat, "passed": passed,
                "latency_ms": comp.latency_ms, "ttft_ms": comp.ttft_ms,
                "prompt_tokens": comp.prompt_tokens, "completion_tokens": comp.completion_tokens,
                "error_text": comp.error, "evals": evals,
            }
            row = ExperimentSample(
                id=_new_id("exs"), experiment_id=experiment_id, case_id=case.id,
                case_name=(case.name or case.id)[:255], target=target["label"][:128],
                variant=(variant.get("label") or "baseline")[:128], repeat=repeat,
                passed=passed, content=_clip(comp.content), reasoning=_clip(comp.reasoning),
                finish_reason=comp.finish_reason, prompt_tokens=comp.prompt_tokens,
                completion_tokens=comp.completion_tokens, latency_ms=comp.latency_ms,
                ttft_ms=comp.ttft_ms, status_code=comp.status_code,
                error_text=(comp.error or None) and str(comp.error)[:2048],
                evals_json=evals,
            )
            async with lock:
                results.append(record)
                pending_rows.append(row)
                counters["done"] += 1
                if comp.error:
                    counters["failed"] += 1
                await flush()

        queue: asyncio.Queue = asyncio.Queue()
        for unit in units:
            queue.put_nowait(unit)

        async def worker() -> None:
            while True:
                try:
                    unit = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await run_unit(unit)
                except Exception as exc:  # noqa: BLE001 — one unit must not kill the pool
                    logger.warning("experiment %s unit failed: %s", experiment_id, exc)
                finally:
                    queue.task_done()

        async with session_factory()() as s0:
            exp0 = await s0.get(Experiment, experiment_id)
            if exp0 is not None:
                exp0.n_planned = len(units)
                await s0.commit()

        try:
            await asyncio.gather(*[worker() for _ in range(min(concurrency, len(units)))])
        finally:
            await stack.close()
        await flush(force=True)

        summary = summarize(results, sorted(set(stack.ids)))

        async with session_factory()() as s3:
            exp3 = await s3.get(Experiment, experiment_id)
            if exp3 is not None:
                cancelled = experiment_id in _CANCELLED
                exp3.status = "cancelled" if cancelled else "completed"
                exp3.summary_json = summary
                exp3.n_completed = counters["done"]
                exp3.n_failed = counters["failed"]
                exp3.ended_at = datetime.now(timezone.utc)
                await s3.commit()
        logger.info(
            "experiment %s %s: %d/%d units, %d errors",
            experiment_id, "cancelled" if experiment_id in _CANCELLED else "completed",
            counters["done"], len(units), counters["failed"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("experiment %s failed", experiment_id)
        try:
            async with session_factory()() as s4:
                exp4 = await s4.get(Experiment, experiment_id)
                if exp4 is not None:
                    exp4.status = "failed"
                    exp4.error_text = f"{exc.__class__.__name__}: {exc}"[:4096]
                    exp4.ended_at = datetime.now(timezone.utc)
                    await s4.commit()
        except Exception:
            logger.exception("experiment %s: could not record failure", experiment_id)
    finally:
        _RUNNERS.pop(experiment_id, None)
        _CANCELLED.discard(experiment_id)


# A run whose heartbeat is older than this is presumed dead. Comfortably above
# PROGRESS_FLUSH_S so a merely-slow flush is never mistaken for a dead runner.
STALE_HEARTBEAT_S = 120.0


async def cleanup_orphaned_running() -> int:
    """Fail experiments whose runner died, at startup.

    The runner is in-process, so a restart orphans anything mid-run. Experiments
    are cheap to re-run and are not log-reconciled, so fail them loudly rather
    than leaving a row that says 'running' forever — same rule as quantization.

    ⚠ Under multi-replica HA another replica may be legitimately driving a run,
    so this only reaps rows with a **stale or absent heartbeat**; a live run
    elsewhere keeps stamping `heartbeat_at` every couple of seconds and survives.
    """
    reaped = 0
    try:
        cutoff = datetime.now(timezone.utc).timestamp() - STALE_HEARTBEAT_S
        async with session_factory()() as session:
            rows = (await session.execute(
                select(Experiment).where(Experiment.status.in_(("queued", "running")))
            )).scalars().all()
            for row in rows:
                hb = row.heartbeat_at
                if hb is not None and hb.timestamp() > cutoff:
                    continue  # another replica is still driving this one
                # Never reap a run this process itself just started.
                if row.id in _RUNNERS:
                    continue
                row.status = "failed"
                row.error_text = "the gateway restarted while this experiment was running"
                row.ended_at = datetime.now(timezone.utc)
                reaped += 1
            if reaped:
                await session.commit()
                logger.info("marked %d orphaned experiment(s) failed at startup", reaped)
    except Exception:
        logger.exception("experiment startup cleanup failed")
    return reaped


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

router = APIRouter(prefix="/v1", tags=["experiments"])


async def _owner_names(session: AsyncSession, ids: set[int]) -> dict[int, str]:
    if not ids:
        return {}
    rows = await session.execute(select(User.id, User.username).where(User.id.in_(ids)))
    return {uid: uname for uid, uname in rows.all()}


# ---- registry ------------------------------------------------------------- #


def _allow_python(user: User) -> bool:
    """python-mode evaluators execute arbitrary code on the gateway host, so they
    need BOTH the env opt-in and admin role."""
    return ce.python_mode_enabled() and bool(user.is_admin)


def _custom_record(row: CustomEvaluator, owner: str) -> CustomEvaluatorRecord:
    return CustomEvaluatorRecord(
        id=row.id, name=row.name, description=row.description, mode=row.mode,
        code=row.code, fail_when_true=row.fail_when_true,
        config=_public_api_config(row.config or {}), owner=owner,
        created_at=row.created_at, updated_at=row.updated_at,
    )


@router.get("/experiments/evaluators")
async def list_evaluators(
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    """The evaluator registry — the web form renders its options from this, so
    the UI can never drift from what the runner actually supports. Also carries
    the caller's custom evaluators and the authoring context, so the form needs
    a single fetch."""
    payload = ev.specs_payload()
    rows = (await session.execute(
        select(CustomEvaluator)
        .where(CustomEvaluator.owner_id == user.id)
        .order_by(CustomEvaluator.created_at.desc())
    )).scalars().all()
    payload["custom"] = [_custom_record(r, user.username).model_dump() for r in rows]
    payload["custom_context"] = ce.describe_context()
    payload["custom_context"]["python_allowed"] = _allow_python(user)
    return payload


@router.get("/custom-evaluators", response_model=list[CustomEvaluatorRecord])
async def list_custom_evaluators(
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    rows = (await session.execute(
        select(CustomEvaluator)
        .where(CustomEvaluator.owner_id == user.id)
        .order_by(CustomEvaluator.created_at.desc())
    )).scalars().all()
    return [_custom_record(r, user.username) for r in rows]


def _public_api_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """api-mode config is safe to echo — it holds a secret *reference*, never a
    secret — but strip anything that looks like an inline key just in case."""
    out = dict(cfg or {})
    out.pop("api_key", None)
    out.pop("api_key_enc", None)
    return out


def _validate_custom(
    name: str, mode: str, code: str, user: User, config: Optional[dict[str, Any]] = None
) -> str:
    name = (name or "").strip()
    if not re.match(r"^[a-z0-9][a-z0-9 _-]{1,47}$", name, re.IGNORECASE):
        raise HTTPException(
            status_code=400,
            detail="name must be 2–48 chars: letters, digits, spaces, - or _",
        )
    # A custom evaluator's name becomes its column in the results, so it must not
    # collide with a built-in id or the two would merge in the summary.
    if name in ev.SPECS:
        raise HTTPException(status_code=400, detail=f"{name!r} is a built-in evaluator id")
    try:
        ce.validate_spec(mode, code, allow_python=_allow_python(user), config=config)
    except ce.CustomEvalError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return name


@router.post("/custom-evaluators", response_model=CustomEvaluatorRecord)
async def create_custom_evaluator(
    req: CustomEvaluatorSpec,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    name = _validate_custom(req.name, req.mode, req.code, user, req.config)
    dupe = (await session.execute(
        select(CustomEvaluator).where(
            CustomEvaluator.owner_id == user.id, CustomEvaluator.name == name
        )
    )).scalar_one_or_none()
    if dupe is not None:
        raise HTTPException(status_code=400, detail=f"you already have an evaluator named {name!r}")
    row = CustomEvaluator(
        id=_new_id("ce"), name=name, description=(req.description or "").strip(),
        mode=req.mode, code=req.code, fail_when_true=bool(req.fail_when_true),
        config=req.config or {}, owner_id=user.id,
    )
    session.add(row)
    await session.commit()
    logger.info("custom evaluator %s (%s) created by %s", row.id, row.mode, user.username)
    return _custom_record(row, user.username)


async def _get_custom(session: AsyncSession, ce_id: str, user: User) -> CustomEvaluator:
    row = await session.get(CustomEvaluator, ce_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such evaluator")
    if row.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="not your evaluator")
    return row


@router.patch("/custom-evaluators/{ce_id}", response_model=CustomEvaluatorRecord)
async def update_custom_evaluator(
    ce_id: str,
    req: CustomEvaluatorUpdate,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    row = await _get_custom(session, ce_id, user)
    name = req.name if req.name is not None else row.name
    mode = req.mode if req.mode is not None else row.mode
    code = req.code if req.code is not None else row.code
    config = req.config if req.config is not None else (row.config or {})
    row.name = _validate_custom(name, mode, code, user, config)
    row.mode, row.code, row.config = mode, code, config
    if req.description is not None:
        row.description = req.description.strip()
    if req.fail_when_true is not None:
        row.fail_when_true = bool(req.fail_when_true)
    row.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return _custom_record(row, user.username)


@router.delete("/custom-evaluators/{ce_id}")
async def delete_custom_evaluator(
    ce_id: str,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    row = await _get_custom(session, ce_id, user)
    await session.delete(row)
    await session.commit()
    return {"ok": True}


@router.post("/custom-evaluators/test")
async def test_custom_evaluator(
    req: TestCustomEvaluatorRequest,
    request: Request,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    """Run an evaluator against one sample reply and return its verdict.

    Authoring without this is guesswork — `fail_when_true` in particular silently
    inverts every result if you pick it wrong.
    """
    try:
        ce.validate_spec(
            req.mode, req.code, allow_python=_allow_python(user), config=req.config
        )
    except ce.CustomEvalError as exc:
        return {"ok": False, "error": str(exc)}

    content, reasoning = req.content, req.reasoning
    finish_reason = req.finish_reason
    usage: dict[str, Any] = {}
    latency_ms, ttft_ms = req.latency_ms, req.ttft_ms
    if req.prompt_tokens is not None:
        usage["prompt_tokens"] = req.prompt_tokens
    if req.completion_tokens is not None:
        usage["completion_tokens"] = req.completion_tokens

    # Try it against a REAL reply from a finished run rather than a paste.
    if req.sample_id:
        srow = await session.get(ExperimentSample, req.sample_id)
        if srow is None:
            raise HTTPException(status_code=404, detail="no such sample")
        exp = await session.get(Experiment, srow.experiment_id)
        if exp is None or (exp.owner_id != user.id and not user.is_admin):
            raise HTTPException(status_code=403, detail="not your sample")
        content, reasoning = srow.content, srow.reasoning
        finish_reason = srow.finish_reason
        usage = {"prompt_tokens": srow.prompt_tokens, "completion_tokens": srow.completion_tokens}
        latency_ms, ttft_ms = srow.latency_ms, srow.ttft_ms

    comp = ev.Completion(
        content=content, reasoning=reasoning, finish_reason=finish_reason,
        usage=usage or None, latency_ms=latency_ms, ttft_ms=ttft_ms,
        expected=dict(req.expected or {}),
    )
    spec = ce.CustomSpec(
        id="preview", name=(req.name or "preview").strip() or "preview",
        mode=req.mode, code=req.code, fail_when_true=bool(req.fail_when_true),
        config=req.config or {},
    )

    if spec.mode == "python":
        worker = ce.PythonEvaluatorWorker(spec)
        try:
            outcome = await worker.evaluate(comp)
        finally:
            await worker.close()
    elif spec.mode == "api":
        genv = await load_global_env(session)
        key = genv.get((spec.config.get("api_key_secret") or "").strip(), "")
        outcome = await ce.ApiEvaluatorClient(spec, _http(request.app), key).evaluate(comp)
    else:
        outcome = ce.run_expression_evaluator(spec, comp)

    return {
        "ok": not outcome.flags.get("evaluator_error"),
        "passed": outcome.passed,
        "score": outcome.score,
        "reason": outcome.reason,
        "flags": outcome.flags,
        "error": outcome.reason if outcome.flags.get("evaluator_error") else None,
    }


@router.get("/experiments/limits")
async def experiment_limits(user: User = Depends(require_section(SECTION))):
    """The run-size ceilings, so the form can enforce them BEFORE submit rather
    than letting the user compose a matrix the gateway will reject."""
    return {
        "max_units": MAX_UNITS,
        "max_rows": MAX_ROWS_PER_RUN,
        "max_concurrency": MAX_CONCURRENCY,
        "default_concurrency": DEFAULT_CONCURRENCY,
        # Above this many rows a dataset is a corpus, not a captured trace, so the
        # form defaults to one pass over a sample instead of a repeat sweep.
        "sweep_row_threshold": 20,
        "sweep_sample_rows": 200,
    }


@router.get("/experiments/targets")
async def list_targets(
    request: Request,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    """Suggested targets: this platform's serverless apps and proxy endpoints,
    prefilled with base_url + model. These are suggestions only — the experiment
    stores plain {base_url, model, key}, so any OpenAI-compatible endpoint works.
    """
    public = (os.environ.get("GATEWAY_PUBLIC_URL", "") or "").strip().rstrip("/")
    if not public:
        public = str(request.base_url).rstrip("/")

    out: list[dict[str, Any]] = []
    stmt = select(App)
    if not user.is_admin:
        stmt = stmt.where(App.owner_id == user.id)
    for a in (await session.execute(stmt.limit(200))).scalars().all():
        out.append({
            "kind": "app",
            "id": a.app_id,
            # A serverless app is served at {gateway}/{app_id}/v1/chat/completions,
            # so the app id is the base path — not a /v1 suffix.
            "label": a.name or a.app_id,
            "base_url": f"{public}/{a.app_id}",
            "model": a.model or "",
        })

    try:
        from .proxy_api import ProxyEndpoint  # local import: avoids a cycle
        prows = (await session.execute(
            select(ProxyEndpoint).where(ProxyEndpoint.enabled.is_(True)).limit(200)
        )).scalars().all()
        for p in prows:
            aliases: list[str] = []
            for u in (p.config or {}).get("upstreams") or []:
                aliases.extend((u.get("models") or {}).keys())
            out.append({
                "kind": "proxy",
                "id": p.id,
                "label": p.name,
                "base_url": f"{public}/proxy/{p.name}",
                "model": aliases[0] if aliases else "",
                "models": sorted(set(aliases)),
            })
    except Exception:
        logger.debug("proxy targets unavailable", exc_info=True)

    return {"targets": out, "gateway_url": public}


# ---- resolving cases from a platform Dataset ------------------------------- #

# Dataset kinds that can carry a chat request. `llm`/`hf` read from the HF hub;
# `upload`/`s3` read a metadata table (jsonl/csv/parquet) out of S3.
CASE_DATASET_KINDS = ("llm", "hf", "upload", "s3")
# Columns searched for the tool declarations that accompanied the request.
_TOOLS_COLUMNS = ("tools", "functions")
_PARAMS_COLUMNS = ("params", "sampling_params", "model_parameters")
_EXPECTED_COLUMNS = ("expected", "expectations")
# Hard ceiling on rows pulled for one run — the same reason experiments cap units.
MAX_ROWS_PER_RUN = int(os.environ.get("EXPERIMENT_MAX_ROWS", "2000") or "2000")


def _dataset_messages_field(d: Dataset) -> Optional[str]:
    """The column holding the chat messages, if this dataset is chat-shaped."""
    return (getattr(d, "messages_field", None) or "").strip() or None


def dataset_usable(d: Dataset) -> tuple[bool, Optional[str]]:
    """Can this Dataset supply replayable requests? Returns (usable, why-not)."""
    if d.kind not in CASE_DATASET_KINDS:
        return False, f"kind={d.kind} has no chat rows"
    if not _dataset_messages_field(d):
        return False, "no messages column mapped — set one on the dataset"
    if d.kind in ("upload", "s3") and not d.storage_id:
        return False, "no storage attached"
    if d.kind in ("llm", "hf") and not d.hf_repo:
        return False, "no HF repo set"
    return True, None


def _coerce_json_cell(value: Any) -> Any:
    """A JSONL round-trip may hand back a dict or its JSON string."""
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            return None
    return None


def _row_to_case(row: dict[str, Any], idx: int, messages_field: str, prefix: str) -> Optional[Case]:
    """Build one Case from a dataset row, or None when the row has no request."""
    from .datasets_api import _parse_messages  # local: avoids an import cycle

    messages = _parse_messages(row.get(messages_field))
    if not isinstance(messages, list) or not messages:
        return None
    clean = lf.clean_messages(messages)
    if not clean:
        return None

    tools = None
    for col in _TOOLS_COLUMNS:
        val = _coerce_json_cell(row.get(col))
        if isinstance(val, list) and val:
            tools = val
            break

    params: dict[str, Any] = {}
    for col in _PARAMS_COLUMNS:
        val = _coerce_json_cell(row.get(col))
        if isinstance(val, dict):
            params = val
            break
    # Also accept flat per-row columns, which is how a captured trace exports.
    for key in ("temperature", "top_p", "top_k", "max_tokens", "repetition_penalty",
                "seed", "enable_thinking"):
        if row.get(key) is not None and key not in params:
            params[key] = row[key]

    expected: dict[str, Any] = {}
    for col in _EXPECTED_COLUMNS:
        val = _coerce_json_cell(row.get(col))
        if isinstance(val, dict):
            expected = val
            break

    name = str(row.get("name") or row.get("case") or row.get("id") or f"row-{idx + 1}")[:255]
    return Case(
        id=f"{prefix}:{idx}", name=name, messages=clean,
        tools=tools, params=params, expected=expected,
    )


async def resolve_cases(
    session: AsyncSession,
    d: Dataset,
    limit: int = 0,
) -> list[Case]:
    """Read replayable requests out of a platform Dataset.

    Reuses the Datasets section's own readers rather than reimplementing them, so
    a dataset behaves identically here and in its row browser.
    """
    from . import datasets_api as da  # local: avoids an import cycle

    usable, why = dataset_usable(d)
    if not usable:
        raise HTTPException(status_code=400, detail=f"dataset {d.id} is not replayable: {why}")
    mf = _dataset_messages_field(d) or "messages"
    cap = min(limit or MAX_ROWS_PER_RUN, MAX_ROWS_PER_RUN)

    rows: list[dict[str, Any]] = []
    if d.kind in ("llm", "hf"):
        storage = await session.get(Storage, d.storage_id) if d.storage_id else None
        token = await da._hf_token(storage, session)
        raw, _total, _split, _names = await da._run_sync(
            da._hf_preview_rows, d.hf_repo, token, cap, 0, None, d.hf_revision,
        )
        rows = list(raw or [])
    else:
        storage = await da._load_storage(session, d.storage_id)
        target, _base = da._s3_target_and_prefix(storage)
        if d.kind == "s3":
            if not d.s3_metadata_uri:
                raise HTTPException(status_code=400, detail="dataset has no s3_metadata_uri")
            parsed = urlparse(d.s3_metadata_uri)
            if parsed.scheme == "s3":
                import dataclasses as _dc
                target = _dc.replace(target, bucket=parsed.netloc)
                key = parsed.path.lstrip("/")
            else:
                key = d.s3_metadata_uri
            mdname = os.path.basename(key)
        else:
            if not d.metadata_filename:
                raise HTTPException(status_code=400, detail="dataset has no uploaded metadata file")
            key = da._metadata_key(storage, d.id, d.metadata_filename)
            mdname = d.metadata_filename
        body = await da._run_sync(bench_s3_get_bytes, key, target)
        if body is None:
            raise HTTPException(status_code=400, detail="metadata file not found in storage")
        rows = da.dataset_metadata.parse_rows_any(mdname, body, cap)

    # Rows the dataset owner un-ticked in the row browser are excluded from
    # training; excluding them here too keeps one meaning of "this dataset".
    excluded = {int(x) for x in (d.excluded_rows or [])}
    cases: list[Case] = []
    for i, row in enumerate(rows):
        if i in excluded:
            continue
        case = _row_to_case(row, i, mf, d.id)
        if case is not None:
            cases.append(case)
        if len(cases) >= cap:
            break
    if not cases:
        raise HTTPException(
            status_code=400,
            detail=f"no rows in {d.name} have a usable {mf!r} column",
        )
    return cases


async def _require_source_dataset(session: AsyncSession, dataset_id: str, user: User) -> Dataset:
    d = await session.get(Dataset, dataset_id)
    if d is None:
        raise HTTPException(status_code=404, detail="no such dataset")
    if d.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="not your dataset")
    return d


@router.get("/experiments/datasets", response_model=list[DatasetOption])
async def list_case_datasets(
    include_unusable: bool = False,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    """Platform Datasets that can supply replayable requests.

    Experiments has no dataset store of its own — this is the Datasets section,
    filtered to the kinds that carry chat rows.
    """
    stmt = select(Dataset).where(Dataset.kind.in_(CASE_DATASET_KINDS))
    if not user.is_admin:
        stmt = stmt.where(Dataset.owner_id == user.id)
    rows = (await session.execute(stmt.order_by(Dataset.created_at.desc()).limit(300))).scalars().all()
    names = await _owner_names(session, {r.owner_id for r in rows})
    out: list[DatasetOption] = []
    for d in rows:
        usable, why = dataset_usable(d)
        if not usable and not include_unusable:
            continue
        out.append(DatasetOption(
            id=d.id, name=d.name, kind=d.kind,
            messages_field=_dataset_messages_field(d), num_rows=d.num_rows,
            owner=names.get(d.owner_id, "?"), usable=usable, reason=why,
        ))
    return out


@router.get("/experiments/datasets/{dataset_id}/rows", response_model=list[RowPreview])
async def preview_rows(
    dataset_id: str,
    limit: int = Query(20, ge=1, le=200),
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    """The rows this dataset would replay — shown before you commit to a run."""
    d = await _require_source_dataset(session, dataset_id, user)
    cases = await resolve_cases(session, d, limit=limit)
    return [
        RowPreview(
            id=c.id, name=c.name, n_messages=len(c.messages), n_tools=len(c.tools or []),
            prompt_chars=sum(len(str(m.get("content") or "")) for m in c.messages),
            params=c.params,
        )
        for c in cases
    ]


# ---- capturing requests INTO a platform Dataset ---------------------------- #


class LangfusePreviewRequest(BaseModel):
    url: str
    base_url: Optional[str] = None
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    public_key_secret: Optional[str] = None
    secret_key_secret: Optional[str] = None


class CaptureLangfuseRequest(LangfusePreviewRequest):
    """Pull observations from a Langfuse trace into a NEW platform Dataset."""
    name: str
    storage_id: str
    observation_ids: list[str] = Field(default_factory=list)


class CapturePlatformRequest(BaseModel):
    """Turn traffic this platform already served into a NEW platform Dataset."""
    name: str
    storage_id: str
    app_id: Optional[str] = None
    limit: int = 20
    status: str = ""
    search: str = ""


class CaptureResult(BaseModel):
    dataset_id: str
    name: str
    n_rows: int


def _langfuse_creds(req: LangfusePreviewRequest, genv: dict[str, str]) -> tuple[str, str, str]:
    base = (req.base_url or os.environ.get("LANGFUSE_BASE_URL", "") or "").strip().rstrip("/")
    pk = (req.public_key or "").strip()
    sk = (req.secret_key or "").strip()
    if req.public_key_secret:
        pk = genv.get(req.public_key_secret.strip(), "")
    if req.secret_key_secret:
        sk = genv.get(req.secret_key_secret.strip(), "")
    if not base:
        raise HTTPException(status_code=400, detail="Langfuse base URL required")
    if not pk or not sk:
        raise HTTPException(
            status_code=400,
            detail="Langfuse public + secret key required (inline or as global-secret names). "
                   "Keys are project-scoped — they must match the trace's project.",
        )
    return base, pk, sk


async def _create_case_dataset(
    session: AsyncSession,
    user: User,
    name: str,
    storage_id: str,
    rows: list[dict[str, Any]],
    description: str,
) -> CaptureResult:
    """Write captured requests into a real Dataset (kind=upload, chat-shaped).

    This is the whole point of dropping the parallel store: a captured corpus is
    a dataset like any other, so it lands in /datasets where it can be browsed,
    published, packed, or reused by anything else on the platform.
    """
    from . import datasets_api as da

    if not rows:
        raise HTTPException(status_code=400, detail="nothing to capture")
    storage = await session.get(Storage, storage_id)
    if storage is None:
        raise HTTPException(status_code=400, detail="no such storage")
    if storage.kind != "s3":
        raise HTTPException(status_code=400, detail="storage must be kind=s3")
    if storage.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="not your storage")

    dataset_id = f"ds-{secrets.token_hex(4)}"
    body = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode()
    filename = "cases.jsonl"

    target, _ = da._s3_target_and_prefix(storage)
    key = da._metadata_key(storage, dataset_id, filename)
    try:
        await da._run_sync(_s3_put_bytes, key, body, target)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"S3 upload failed: {exc}") from exc

    row = Dataset(
        id=dataset_id, owner_id=user.id, name=name.strip(), description=description,
        kind="upload", storage_id=storage_id, metadata_filename=filename,
        format="jsonl", num_rows=len(rows), size_bytes=len(body),
        messages_field="messages",
    )
    session.add(row)
    await session.commit()
    logger.info(
        "captured %d request(s) into dataset %s by %s", len(rows), dataset_id, user.username
    )
    return CaptureResult(dataset_id=dataset_id, name=row.name, n_rows=len(rows))


@router.post("/experiments/capture/langfuse/preview")
async def langfuse_preview(
    req: LangfusePreviewRequest,
    request: Request,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    """Resolve a Langfuse URL and list the replayable generations in that trace."""
    genv = await load_global_env(session)
    base, pk, sk = _langfuse_creds(req, genv)
    try:
        parsed = lf.parse_langfuse_url(req.url)
    except lf.LangfuseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    trace_id = parsed.get("trace_id")
    if not trace_id:
        raise HTTPException(
            status_code=400,
            detail="no trace id in that URL — expected a ?peek=/?traceId= param or /traces/<id>",
        )
    client = _http(request.app)
    try:
        trace = await lf.fetch_trace(client, base, pk, sk, trace_id)
    except lf.LangfuseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Langfuse unreachable: {exc}")
    return {
        "trace_id": trace_id,
        "trace_name": trace.get("name"),
        "used_trace_id_param": parsed.get("used_trace_id_param"),
        "suggested_observation": parsed.get("observation_id"),
        "generations": lf.list_generations(trace),
    }


@router.post("/experiments/capture/langfuse", response_model=CaptureResult)
async def capture_langfuse(
    req: CaptureLangfuseRequest,
    request: Request,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    genv = await load_global_env(session)
    base, pk, sk = _langfuse_creds(req, genv)
    parsed = lf.parse_langfuse_url(req.url)
    trace_id = parsed.get("trace_id")
    if not trace_id:
        raise HTTPException(status_code=400, detail="no trace id in that URL")
    client = _http(request.app)
    try:
        trace = await lf.fetch_trace(client, base, pk, sk, trace_id)
    except lf.LangfuseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Langfuse unreachable: {exc}")

    obs_ids = req.observation_ids or (
        [parsed["observation_id"]] if parsed.get("observation_id") else [None]
    )
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for oid in obs_ids:
        try:
            payload = lf.extract_request(trace, oid)
        except lf.LangfuseError as exc:
            errors.append(str(exc))
            continue
        rows.append({
            "name": f"{trace.get('name') or trace_id} · {payload.get('observation_name') or payload.get('observation_id')}"[:255],
            "messages": payload["messages"],
            "tools": payload.get("tools") or [],
            "params": payload.get("params") or {},
            "source_ref": f"langfuse:{trace_id}:{payload.get('observation_id')}",
        })
    if not rows:
        raise HTTPException(
            status_code=400, detail="; ".join(errors) or "nothing replayable in that trace"
        )
    return await _create_case_dataset(
        session, user, req.name, req.storage_id, rows,
        description=f"Captured from Langfuse trace {trace_id}",
    )


@router.post("/experiments/capture/platform", response_model=CaptureResult)
async def capture_platform(
    req: CapturePlatformRequest,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    """Turn requests this platform already served into a replayable dataset.

    ⚠ Proxy-mode rows (`proxy_requests`) are deliberately slim — model + usage
    only, for throughput — so they carry no body and cannot be captured here.
    """
    stmt = select(ReqRow).order_by(ReqRow.created_at.desc())
    if not user.is_admin:
        stmt = stmt.where(ReqRow.owner_id == user.id)
    if req.app_id:
        stmt = stmt.where(ReqRow.app_id == req.app_id)
    if req.status:
        stmt = stmt.where(ReqRow.status == req.status)
    if req.search:
        stmt = stmt.where(cast(ReqRow.payload, Text).ilike(f"%{req.search}%"))

    limit = max(1, min(500, req.limit))
    found = (await session.execute(stmt.limit(limit * 4))).scalars().all()
    rows: list[dict[str, Any]] = []
    for r in found:
        if len(rows) >= limit:
            break
        payload = r.payload if isinstance(r.payload, dict) else {}
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            continue  # slim proxy row or a non-chat request
        params = {
            k: payload[k] for k in
            ("temperature", "top_p", "max_tokens", "top_k", "repetition_penalty", "seed")
            if payload.get(k) is not None
        }
        ctk = payload.get("chat_template_kwargs")
        if isinstance(ctk, dict) and ctk.get("enable_thinking") is not None:
            params["enable_thinking"] = bool(ctk["enable_thinking"])
        rows.append({
            "name": f"{r.app_id} · {r.request_id[:12]}"[:255],
            "messages": lf.clean_messages(messages),
            "tools": payload.get("tools") or [],
            "params": params,
            "source_ref": f"platform:{r.request_id}",
        })
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="no replayable requests found. Proxy-mode rows store only model + usage "
                   "(no body) and can't be captured.",
        )
    return await _create_case_dataset(
        session, user, req.name, req.storage_id, rows,
        description="Captured from this platform's served traffic",
    )


# ---- experiments ---------------------------------------------------------- #


@router.get("/experiments/_page", response_model=ExperimentPage)
async def list_experiments_page(
    scope: str = "mine",
    q: str = "",
    status: str = "",
    dataset_id: str = "",
    limit: int = Query(12, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Experiment)
    if not (scope == "all" and user.is_admin):
        stmt = stmt.where(Experiment.owner_id == user.id)
    if status:
        stmt = stmt.where(Experiment.status == status)
    if dataset_id:
        stmt = stmt.where(Experiment.dataset_id == dataset_id)
    for tok in (q or "").lower().split():
        like = f"%{tok}%"
        stmt = stmt.where(or_(
            Experiment.id.ilike(like),
            Experiment.name.ilike(like),
            Experiment.status.ilike(like),
            cast(Experiment.config_json, Text).ilike(like),
        ))
    total = (await session.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    rows = (await session.execute(
        stmt.order_by(Experiment.created_at.desc()).limit(limit).offset(offset)
    )).scalars().all()
    names = await _owner_names(session, {r.owner_id for r in rows})
    ds_names = await _dataset_names(session, {r.dataset_id for r in rows})
    return ExperimentPage(
        total=total,
        items=[
            _experiment_record(r, names.get(r.owner_id, "?"), ds_names.get(r.dataset_id, "?"))
            for r in rows
        ],
    )


async def _dataset_names(session: AsyncSession, ids: set[str]) -> dict[str, str]:
    """Names for the run list. A deleted dataset still leaves a readable run."""
    if not ids:
        return {}
    rows = await session.execute(
        select(Dataset.id, Dataset.name).where(Dataset.id.in_(ids))
    )
    found = {i: n for i, n in rows.all()}
    return {i: found.get(i, "(deleted dataset)") for i in ids}


@router.post("/experiments", response_model=ExperimentRecord)
async def create_experiment(
    req: CreateExperimentRequest,
    request: Request,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    name = (req.name or "").strip()
    if not _ID_RE.match(name):
        raise HTTPException(status_code=400, detail="invalid experiment name")
    ds = await _require_source_dataset(session, req.dataset_id, user)
    usable, why = dataset_usable(ds)
    if not usable:
        raise HTTPException(status_code=400, detail=f"dataset {ds.name} is not replayable: {why}")
    if not req.targets:
        raise HTTPException(status_code=400, detail="at least one target is required")

    labels = [t.label.strip() or t.base_url for t in req.targets]
    if len(set(labels)) != len(labels):
        raise HTTPException(status_code=400, detail="target labels must be unique")
    for t in req.targets:
        if not (t.base_url or "").strip():
            raise HTTPException(status_code=400, detail=f"target {t.label!r} has no base URL")
        if not (t.model or "").strip():
            raise HTTPException(status_code=400, detail=f"target {t.label!r} has no model")

    variants = req.variants or [VariantSpec()]
    vlabels = [v.label.strip() or "baseline" for v in variants]
    if len(set(vlabels)) != len(vlabels):
        raise HTTPException(status_code=400, detail="variant labels must be unique")

    stored_evaluators = await snapshot_evaluators(session, user, req.evaluators)

    # Row count is the dataset's own tally where it has one; the exact figure is
    # only known once the rows are read, so the cap below is re-checked at run time.
    known_rows = ds.num_rows or 0
    n_rows = min(known_rows, req.max_rows) if req.max_rows > 0 else known_rows
    n_rows = n_rows or (req.max_rows or 1)
    planned = n_rows * len(req.targets) * len(variants) * max(1, req.repeats)
    if planned > MAX_UNITS:
        raise HTTPException(
            status_code=400,
            detail=f"{planned} units exceeds the {MAX_UNITS} cap "
                   f"({n_rows} rows × {len(req.targets)} targets × {len(variants)} variants "
                   f"× {req.repeats} repeats)",
        )

    judge = next((s for s in stored_evaluators if s.get("id") == "llm_judge"), None)
    if judge:
        opts = judge.get("options") or {}
        if not (opts.get("model") or "").strip() and not (opts.get("base_url") or "").strip():
            # Falls back to the first target — fine, but make that explicit in the
            # stored config so the run is reproducible from the record alone.
            opts["base_url"] = req.targets[0].base_url
            opts["model"] = req.targets[0].model
            judge["options"] = opts

    cfg = {
        "targets": _store_targets(req.targets),
        "variants": [v.model_dump() for v in variants],
        "evaluators": stored_evaluators,
        "repeats": max(1, req.repeats),
        "concurrency": max(1, min(MAX_CONCURRENCY, req.concurrency)),
        "retries": max(1, req.retries),
        "timeout_s": max(1.0, req.timeout_s),
        "stream": bool(req.stream),
        "max_rows": max(0, req.max_rows),
    }

    row = Experiment(
        id=_new_id("exp"), name=name, dataset_id=req.dataset_id, status="queued",
        config_json=cfg, n_planned=planned, owner_id=user.id,
    )
    session.add(row)
    await session.commit()

    task = asyncio.create_task(_run_experiment(request.app, row.id))
    _RUNNERS[row.id] = task
    logger.info(
        "experiment %s queued by %s: %d units (%d rows × %d targets × %d variants × %d repeats)",
        row.id, user.username, planned, n_rows, len(req.targets), len(variants), req.repeats,
    )
    return _experiment_record(row, user.username, ds.name)


async def snapshot_evaluators(
    session: AsyncSession,
    user: User,
    selections: list[EvaluatorSelection],
) -> list[dict[str, Any]]:
    """Resolve an evaluator selection into the form a runner consumes.

    Custom evaluators are referenced as `custom:<ce-id>` (a library entry) or
    bare `custom` with the definition inline. Either way the definition is
    **snapshotted** into the caller's config: editing the library entry later
    must not retroactively change what a finished run measured.

    Shared by experiments and prompt optimization so both grade with the same
    evaluators resolved the same way.
    """
    stored: list[dict[str, Any]] = []
    for sel in selections:
        if not sel.id.startswith("custom"):
            if sel.id not in ev.SPECS:
                raise HTTPException(status_code=400, detail=f"unknown evaluator: {sel.id}")
            stored.append(sel.model_dump())
            continue

        opts = sel.options or {}
        _, _, ref = sel.id.partition(":")
        if ref:
            row = await session.get(CustomEvaluator, ref)
            if row is None:
                raise HTTPException(status_code=400, detail=f"no such custom evaluator: {ref}")
            if row.owner_id != user.id and not user.is_admin:
                raise HTTPException(status_code=403, detail=f"not your evaluator: {ref}")
            snap = {
                "id": row.id, "name": row.name, "mode": row.mode,
                "code": row.code, "fail_when_true": row.fail_when_true,
                "config": row.config or {},
            }
        else:
            snap = {
                "id": "inline",
                "name": str(opts.get("name") or "custom").strip() or "custom",
                "mode": str(opts.get("mode") or "expression"),
                "code": str(opts.get("code") or ""),
                "fail_when_true": bool(opts.get("fail_when_true", False)),
                "config": dict(opts.get("config") or {}),
            }
        # Re-validate at run-create time: python mode may have been disabled, or
        # the caller may not be an admin any more, since the entry was saved.
        try:
            ce.validate_spec(
                snap["mode"], snap["code"],
                allow_python=_allow_python(user), config=snap.get("config"),
            )
        except ce.CustomEvalError as exc:
            raise HTTPException(
                status_code=400, detail=f"custom evaluator {snap['name']!r}: {exc}"
            )
        if snap["name"] in ev.SPECS:
            raise HTTPException(
                status_code=400,
                detail=f"custom evaluator {snap['name']!r} collides with a built-in id",
            )
        stored.append({"id": f"custom:{snap['id']}", "custom": snap})

    names = [(s.get("custom") or {}).get("name") for s in stored if s.get("custom")]
    if len(set(names)) != len(names):
        raise HTTPException(status_code=400, detail="custom evaluator names must be unique")
    return stored


async def _get_experiment(session: AsyncSession, exp_id: str, user: User) -> Experiment:
    row = await session.get(Experiment, exp_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such experiment")
    if row.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="not your experiment")
    return row


@router.get("/experiments/{exp_id}", response_model=ExperimentRecord)
async def get_experiment(
    exp_id: str,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    row = await _get_experiment(session, exp_id, user)
    names = await _owner_names(session, {row.owner_id})
    ds_names = await _dataset_names(session, {row.dataset_id})
    return _experiment_record(row, names.get(row.owner_id, "?"), ds_names.get(row.dataset_id, "?"))


@router.get("/experiments/{exp_id}/samples", response_model=SamplePage)
async def list_samples(
    exp_id: str,
    target: str = "",
    variant: str = "",
    case_id: str = "",
    only_failed: bool = False,
    q: str = "",
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    await _get_experiment(session, exp_id, user)
    stmt = select(ExperimentSample).where(ExperimentSample.experiment_id == exp_id)
    if target:
        stmt = stmt.where(ExperimentSample.target == target)
    if variant:
        stmt = stmt.where(ExperimentSample.variant == variant)
    if case_id:
        stmt = stmt.where(ExperimentSample.case_id == case_id)
    if only_failed:
        stmt = stmt.where(ExperimentSample.passed.is_(False))
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(
            ExperimentSample.content.ilike(like),
            ExperimentSample.reasoning.ilike(like),
            ExperimentSample.error_text.ilike(like),
            ExperimentSample.case_name.ilike(like),
        ))
    total = (await session.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    rows = (await session.execute(
        stmt.order_by(ExperimentSample.passed.asc(), ExperimentSample.created_at.asc())
        .limit(limit).offset(offset)
    )).scalars().all()
    return SamplePage(
        total=total,
        items=[
            SampleRecord(
                id=r.id, case_id=r.case_id, case_name=r.case_name, target=r.target,
                variant=r.variant, repeat=r.repeat, passed=r.passed, content=r.content,
                reasoning=r.reasoning, finish_reason=r.finish_reason,
                prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
                latency_ms=r.latency_ms, ttft_ms=r.ttft_ms, status_code=r.status_code,
                error_text=r.error_text, evals=r.evals_json or {},
            )
            for r in rows
        ],
    )


@router.post("/experiments/{exp_id}/cancel", response_model=ExperimentRecord)
async def cancel_experiment(
    exp_id: str,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    """Stop a run cooperatively — in-flight units finish, queued ones are skipped,
    and whatever completed is still summarized."""
    row = await _get_experiment(session, exp_id, user)
    if row.status not in ("queued", "running"):
        raise HTTPException(status_code=400, detail=f"experiment is {row.status}")
    _CANCELLED.add(exp_id)
    logger.info("experiment %s cancel requested by %s", exp_id, user.username)
    names = await _owner_names(session, {row.owner_id})
    ds_names = await _dataset_names(session, {row.dataset_id})
    return _experiment_record(row, names.get(row.owner_id, "?"), ds_names.get(row.dataset_id, "?"))


@router.delete("/experiments/{exp_id}")
async def delete_experiment(
    exp_id: str,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    row = await _get_experiment(session, exp_id, user)
    if row.status == "running":
        raise HTTPException(status_code=400, detail="cancel the run before deleting it")
    await session.execute(
        _delete(ExperimentSample).where(ExperimentSample.experiment_id == exp_id)
    )
    await session.delete(row)
    await session.commit()
    logger.info("experiment %s deleted by %s", exp_id, user.username)
    return {"ok": True}


@router.get("/experiments/{exp_id}/compare")
async def compare_experiments(
    exp_id: str,
    against: str = Query("", description="Comma-separated experiment ids to compare with"),
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    """Summaries for this experiment plus any others, for the tradeoff plot.

    Cells are keyed `{experiment} / {target} / {variant}` so runs from different
    days line up on the same axes.
    """
    ids = [exp_id] + [i.strip() for i in (against or "").split(",") if i.strip()]
    out = []
    for eid in dict.fromkeys(ids):
        row = await session.get(Experiment, eid)
        if row is None:
            continue
        if row.owner_id != user.id and not user.is_admin:
            continue
        out.append({
            "id": row.id,
            "name": row.name,
            "status": row.status,
            "created_at": row.created_at,
            "summary": row.summary_json,
        })
    if not out:
        raise HTTPException(status_code=404, detail="no such experiment")
    return {"experiments": out}
