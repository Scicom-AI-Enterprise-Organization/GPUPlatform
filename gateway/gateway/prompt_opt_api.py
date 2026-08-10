"""Prompt optimization (GEPA) — the platform half.

`prompt_opt.py` is the search; this module is everything it needs from the
platform, and nothing more:

    rows      ← the same `resolve_cases()` an experiment replays
    rollout   ← the same `build_request()` + `call_once()` an experiment sends
    score     ← the same `EvaluatorStack` an experiment grades with
    feedback  ← the evaluators' own `reason` strings

That reuse is the design, not a convenience. An optimizer that scores rollouts
its own way produces a "+14pp" nobody can reproduce in the experiment that is
supposed to confirm it; sharing one scoring path means the confirmation run is
the same measurement. For the same reason the winning candidate is written back
as a plain **variant** (`system_override`) rather than a bespoke artifact — one
click turns the result into an ordinary experiment.

    Dataset ── rows ─┬─ validation rows → every candidate's score VECTOR
                     └─ train rows      → the reflection minibatches
    Optimization = dataset × target × evaluators × budget
              └── candidates (a lineage) → the best system prompt

⚠ **Every metric call is a real billed request.** Budget is denominated in those
and enforced before each iteration starts (see `prompt_opt.optimize`), a hard
ceiling sits above it (`PROMPT_OPT_MAX_METRIC_CALLS`), and the form shows the
number before you submit. Reflection calls are counted and reported separately —
they hit the reflection endpoint, not the target.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    cast,
    func,
    or_,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from . import evaluators as ev
from . import prompt_opt as po
from .auth import require_section
from .db import Base, Dataset, User, get_session, session_factory
from .experiments_api import (
    DEFAULT_CONCURRENCY,
    MAX_CONCURRENCY,
    MAX_STORED_CHARS,
    EvaluatorSelection,
    EvaluatorStack,
    TargetSpec,
    _clip,
    _http,
    _owner_names,
    _require_source_dataset,
    _resolve_key,
    _store_targets,
    build_request,
    call_once,
    dataset_usable,
    resolve_cases,
    snapshot_evaluators,
)
from .global_env_api import load_global_env

logger = logging.getLogger("gateway.prompt_opt")

SECTION = "experiments"

# Hard ceiling on billed calls for ONE optimization, whatever the form asks for.
MAX_METRIC_CALLS = int(os.environ.get("PROMPT_OPT_MAX_METRIC_CALLS", "5000") or "5000")
# Rows pulled from the dataset for one run. Small on purpose: GEPA's cost is
# (rows × candidates), so a 2000-row corpus would spend the whole budget on a
# single validation sweep and never get to iterate.
MAX_OPT_ROWS = int(os.environ.get("PROMPT_OPT_MAX_ROWS", "200") or "200")
DEFAULT_ROWS = 50
# Budget presets, as multiples of the validation-set size — roughly "how many
# full validation sweeps may this run afford". `light` ≈ the dspy preset's
# half-dozen candidate evaluations.
AUTO_BUDGETS = {"light": 6, "medium": 15, "heavy": 40}
PROGRESS_FLUSH_S = 2.0
STALE_HEARTBEAT_S = 120.0
# Reflection replies are instructions, not essays; also the ceiling that stops a
# runaway reflection model from costing more than the search it serves.
DEFAULT_REFLECTION_MAX_TOKENS = 4096

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,127}$")

_RUNNERS: dict[str, asyncio.Task] = {}
_CANCELLED: set[str] = set()


# --------------------------------------------------------------------------- #
# DB model
# --------------------------------------------------------------------------- #


class PromptOptimization(Base):
    """One GEPA run. The candidate lineage lives in `result_json` rather than a
    child table: it is bounded by the budget (a few dozen accepted candidates at
    most) and is always read whole, so a join would buy nothing."""
    __tablename__ = "prompt_optimizations"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # opt-<hex8>
    name: Mapped[str] = mapped_column(String(128))
    dataset_id: Mapped[str] = mapped_column(String(64), index=True)
    # queued | running | completed | failed | cancelled
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    # {target, reflection, evaluators, components, budget, …}. Any inline API key
    # is stored ENCRYPTED inside this blob, never in plaintext.
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    # {seed, best, candidates, iterations, …} — grows as the run progresses, so
    # the detail page can watch a live search rather than a spinner.
    result_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    budget: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    metric_calls: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    reflection_calls: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    # Denormalized for the run list — the headline is "did it get better", and
    # parsing result_json for every row of a list page to answer that is silly.
    seed_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    best_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_text: Mapped[Optional[str]] = mapped_column(String(4096), nullable=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Same role as Experiment.heartbeat_at: it is what makes the restart sweep
    # safe when another HA replica is legitimately driving this run.
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


# --------------------------------------------------------------------------- #
# API models
# --------------------------------------------------------------------------- #


class CreateOptimizationRequest(BaseModel):
    name: str
    dataset_id: str
    # The model whose prompt is being optimized.
    target: TargetSpec
    # The model that WRITES the new prompts. Blank = reuse the target, which
    # works but is the weaker setup — GEPA's published result comes from a small
    # student and a strong reflector.
    reflection: Optional[TargetSpec] = None
    reflection_guidance: str = ""
    reflection_temperature: float = 1.0
    reflection_max_tokens: int = DEFAULT_REFLECTION_MAX_TOKENS
    evaluators: list[EvaluatorSelection] = Field(default_factory=list)
    components: list[str] = Field(default_factory=lambda: list(po.DEFAULT_COMPONENTS))
    # Blank = infer from the dataset's own system messages.
    seed_prompt: str = ""
    seed_user_suffix: str = ""
    max_rows: int = 0
    # 0 = auto (a fraction of the rows pulled).
    val_rows: int = 0
    budget: str = "light"       # light | medium | heavy | custom
    max_metric_calls: int = 0   # honoured when budget == "custom"
    minibatch_size: int = 3
    max_examples: int = 5
    concurrency: int = DEFAULT_CONCURRENCY
    timeout_s: float = 300.0
    stream: bool = False
    # Per-evaluator weight in the scalar score. Absent = 1.0.
    weights: dict[str, float] = Field(default_factory=dict)
    # Show the row's `expected` block to the reflection model.
    include_expected: bool = True
    rng_seed: int = 0


class OptimizationRecord(BaseModel):
    id: str
    name: str
    dataset_id: str
    dataset_name: str
    status: str
    config: dict[str, Any]
    result: Optional[dict[str, Any]]
    budget: int
    metric_calls: int
    reflection_calls: int
    seed_score: Optional[float]
    best_score: Optional[float]
    error_text: Optional[str]
    owner: str
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]


class OptimizationPage(BaseModel):
    total: int
    items: list[OptimizationRecord]


class SeedPromptResponse(BaseModel):
    """What the form prefills as the starting prompt."""
    seed_prompt: str
    n_rows: int
    n_with_system: int
    distinct_system: int
    source: str  # dataset | none


# --------------------------------------------------------------------------- #
# Scoring — evaluator verdicts → (scalar, written feedback)
# --------------------------------------------------------------------------- #

# Verdicts that mean "this detector had nothing to say", NOT "this reply is
# fine". Counting an abstention as a pass is how a suite reports 100% having
# scored nothing (the same trap the benchmark evaluators warn about), and here
# it would additionally teach the optimizer that a broken reply was good.
_ABSTAIN_FLAGS = ("skipped", "evaluator_error", "judge_error")


def outcome_value(o: ev.EvalOutcome) -> Optional[float]:
    """One detector's contribution to the scalar, or None if it doesn't count.

    ⚠ The ALWAYS_ON diagnostics (`request_error`) are excluded. They are not
    objectives: on a successful request one passes unconditionally, so including
    it just adds a constant — a prompt that fails its only real check would score
    0.5 instead of 0, and every reported gain would be halved. A request that DID
    fail is already forced to 0 by the caller.
    """
    if o.id in ev.ALWAYS_ON:
        return None
    if any((o.flags or {}).get(f) for f in _ABSTAIN_FLAGS):
        return None
    if o.score is not None:
        return max(0.0, min(1.0, float(o.score)))
    return 1.0 if o.passed else 0.0


def score_outcomes(
    outcomes: list[ev.EvalOutcome],
    weights: Optional[dict[str, float]] = None,
) -> tuple[float, int]:
    """Weighted mean of the detectors that actually scored. Returns (score, n)."""
    weights = weights or {}
    num = den = 0.0
    n = 0
    for o in outcomes:
        val = outcome_value(o)
        if val is None:
            continue
        w = float(weights.get(o.id, 1.0))
        if w <= 0:
            continue
        num += w * val
        den += w
        n += 1
    return (num / den if den else 0.0), n


def render_feedback(
    outcomes: list[ev.EvalOutcome],
    comp: ev.Completion,
    expected: Optional[dict[str, Any]] = None,
) -> str:
    """The written half of the metric — what GEPA actually reads.

    Every detector already explains itself in `EvalOutcome.reason`; this just
    lays those out so the reflection model can see which rule was broken and
    why. A run whose evaluators all abstain says so explicitly, because
    "no feedback" and "nothing to fix" would otherwise look identical.
    """
    if comp.error:
        return f"The request failed and produced no answer: {comp.error}"

    lines: list[str] = []
    for o in outcomes:
        flags = o.flags or {}
        if any(flags.get(f) for f in _ABSTAIN_FLAGS):
            continue
        verdict = "PASS" if o.passed else "FAIL"
        detail = (o.reason or "").strip()
        lines.append(f"{verdict} {o.id}" + (f": {detail}" if detail else ""))
    if not lines:
        lines.append("No evaluator produced a verdict on this reply.")

    if expected:
        public = {k: v for k, v in expected.items() if not k.startswith("_")}
        if public:
            lines.append(
                "The reference answer for this row was: "
                + _clip(json.dumps(public, ensure_ascii=False), 1200)
            )
    return "\n".join(lines)


def render_input(case: Any, limit: int = 2000) -> str:
    """The row as the reflection model should see it: the task, minus the very
    instruction being rewritten (showing it back invites the model to copy it)."""
    parts: list[str] = []
    for m in case.messages or []:
        role = str(m.get("role") or "")
        if role == "system":
            continue
        content = str(m.get("content") or "").strip()
        calls = m.get("tool_calls") or []
        if calls:
            names = ", ".join(
                str((c.get("function") or {}).get("name") or "?") for c in calls
            )
            content = f"{content}\n(called: {names})".strip()
        if content:
            parts.append(f"[{role}] {content}")
    if case.tools:
        names = ", ".join(
            str((t.get("function") or {}).get("name") or t.get("name") or "?")
            for t in case.tools
        )
        parts.append(f"[tools available] {names}")
    return _clip("\n".join(parts), limit)


# --------------------------------------------------------------------------- #
# Budget + split
# --------------------------------------------------------------------------- #


def resolve_budget(kind: str, custom: int, n_val: int, minibatch: int) -> int:
    """Metric-call budget for a run, clamped to something that can actually run.

    The floor is one seed sweep plus one full iteration — a budget below that
    would evaluate the seed, discover it cannot afford a mutation, and report a
    "result" identical to the input.
    """
    floor = 2 * max(1, n_val) + 2 * max(1, minibatch)
    if kind == "custom":
        want = int(custom or 0)
    else:
        want = AUTO_BUDGETS.get(kind, AUTO_BUDGETS["light"]) * max(1, n_val)
    return max(floor, min(MAX_METRIC_CALLS, want or floor))


def split_rows(row_ids: list[str], val_rows: int, rng: random.Random) -> tuple[list[str], list[str]]:
    """Shuffle, then take the validation set off the front.

    Shuffled because dataset row order is rarely arbitrary — a corpus sorted by
    language or by tool would otherwise put a whole category in one split, and
    the optimizer would tune against a slice it never validates on.
    """
    order = list(row_ids)
    rng.shuffle(order)
    n = len(order)
    want = val_rows if val_rows > 0 else max(1, round(n * 0.4))
    want = max(1, min(want, n))
    val = order[:want]
    train = order[want:]
    # Too few rows to split: reuse the validation rows for reflection. Honest,
    # and better than optimizing against an empty minibatch stream — but it does
    # mean the reported gain is in-sample, which the record states.
    return (train or list(val)), val


def auto_val_rows(n_rows: int) -> int:
    return max(1, min(n_rows, round(n_rows * 0.4)))


def infer_seed_prompt(cases: list[Any]) -> tuple[str, int, int]:
    """The dataset's own most common system message.

    Optimizing a prompt the rows were never captured under measures something
    else, so the seed defaults to what the corpus actually used. Returns
    (prompt, n_with_system, n_distinct).
    """
    counts: dict[str, int] = {}
    for c in cases:
        for m in c.messages or []:
            if m.get("role") == "system":
                text = str(m.get("content") or "").strip()
                if text:
                    counts[text] = counts.get(text, 0) + 1
                break
    if not counts:
        return "", 0, 0
    best = max(counts.items(), key=lambda kv: (kv[1], -len(kv[0])))
    return best[0], sum(counts.values()), len(counts)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def _public_opt_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Strip encrypted key material before a config leaves the gateway."""
    out = json.loads(json.dumps(cfg, default=str))
    for key in ("target", "reflection"):
        t = out.get(key)
        if isinstance(t, dict):
            t["has_inline_key"] = bool(t.pop("api_key_enc", None))
    for sel in out.get("evaluators") or []:
        opts = (sel or {}).get("options")
        if isinstance(opts, dict):
            opts.pop("api_key", None)
            opts.pop("api_key_enc", None)
    return out


def _candidate_payload(c: po.Candidate) -> dict[str, Any]:
    return {
        "index": c.index,
        "parent": c.parent,
        "iteration": c.iteration,
        "component": c.component,
        "origin": c.origin,
        "score": round(c.score, 4),
        "texts": {k: _clip(v, MAX_STORED_CHARS) for k, v in c.texts.items()},
    }


def _iteration_payload(log: po.IterationLog) -> dict[str, Any]:
    return {
        "i": log.i,
        "parent": log.parent,
        "component": log.component,
        "origin": log.origin,
        "row_ids": log.row_ids,
        "parent_score": log.parent_score,
        "child_score": log.child_score,
        "accepted": log.accepted,
        "val_score": log.val_score,
        "candidate": log.candidate,
        "calls": log.calls,
        "note": log.note,
        "proposal": _clip(log.proposal, 4000),
        # Three is enough to show WHY the prompt changed without turning the
        # record into a transcript store.
        "examples": [
            {
                "row_id": ex.get("row_id"),
                "row_name": ex.get("row_name"),
                "score": ex.get("score"),
                "prompt": _clip(str(ex.get("prompt") or ""), 1200),
                "output": _clip(str(ex.get("output") or ""), 1200),
                "feedback": _clip(str(ex.get("feedback") or ""), 1200),
            }
            for ex in (log.examples or [])[:3]
        ],
    }


async def _run_optimization(app, opt_id: str) -> None:
    """Execute one GEPA run end to end. Owns its own DB sessions — it outlives
    the request that created it."""
    started = datetime.now(timezone.utc)
    stack: Optional[EvaluatorStack] = None
    state: dict[str, Any] = {
        "iterations": [], "metric_calls": 0, "reflection_calls": 0,
        # A dataset whose rows carry no reference data makes every detector
        # abstain: the score is then flat, nothing ever beats the seed, and the
        # run "completes" having measured nothing. Counted so the result can say
        # so outright — the same trap the benchmark evaluators warn about.
        "rollouts": 0, "unscored": 0,
    }

    try:
        async with session_factory()() as session:
            row = await session.get(PromptOptimization, opt_id)
            if row is None:
                return
            cfg = row.config_json or {}
            ds = await session.get(Dataset, row.dataset_id)
            if ds is None:
                raise RuntimeError(f"dataset {row.dataset_id} no longer exists")
            cases = await resolve_cases(session, ds, limit=int(cfg.get("max_rows") or DEFAULT_ROWS))
            genv = await load_global_env(session)
            row.status = "running"
            row.started_at = started
            await session.commit()

        target: dict[str, Any] = cfg.get("target") or {}
        reflection: dict[str, Any] = cfg.get("reflection") or target
        selections: list[dict[str, Any]] = cfg.get("evaluators") or []
        components: list[str] = cfg.get("components") or list(po.DEFAULT_COMPONENTS)
        weights: dict[str, float] = cfg.get("weights") or {}
        timeout_s = float(cfg.get("timeout_s") or 300.0)
        stream = bool(cfg.get("stream", False))
        concurrency = max(1, min(MAX_CONCURRENCY, int(cfg.get("concurrency") or DEFAULT_CONCURRENCY)))
        include_expected = bool(cfg.get("include_expected", True))
        minibatch = max(1, int(cfg.get("minibatch_size") or 3))
        max_examples = max(1, int(cfg.get("max_examples") or 5))
        rng_seed = int(cfg.get("rng_seed") or 0)

        client = _http(app)
        target_key = _resolve_key(target, genv)
        reflection_key = _resolve_key(reflection, genv) or (
            target_key if reflection is target else ""
        )
        stack = EvaluatorStack(
            selections, client, genv,
            default_target=target, default_key=target_key, timeout_s=timeout_s,
        )

        by_id = {c.id: c for c in cases}
        rng = random.Random(rng_seed)
        train_rows, val_rows = split_rows(
            [c.id for c in cases], int(cfg.get("val_rows") or 0), rng
        )
        budget = resolve_budget(
            str(cfg.get("budget") or "light"),
            int(cfg.get("max_metric_calls") or 0),
            len(val_rows), minibatch,
        )

        sem = asyncio.Semaphore(concurrency)

        def variant_of(texts: dict[str, str]) -> dict[str, Any]:
            """A candidate IS a variant — that is what makes the winning prompt
            replayable by the ordinary experiment runner with no special case."""
            v: dict[str, Any] = {"label": "candidate"}
            for name in components:
                field = po.COMPONENTS[name]["variant_field"]
                v[field] = texts.get(name, "")
            return v

        async def rollout(texts: dict[str, str], row_id: str) -> po.Rollout:
            case = by_id[row_id]
            body = build_request(case, target, variant_of(texts), stream)
            async with sem:
                comp = await call_once(
                    client, target.get("base_url") or "",
                    target.get("path") or "/v1/chat/completions",
                    target_key, body, timeout_s,
                )
            # `_tools` lets the function-call detector resolve each call against
            # the schema the model actually saw — same contract as the runner.
            comp.expected = {
                **(case.expected or {}),
                "_tool_calls": (comp.expected or {}).get("_tool_calls") or [],
                "_tools": case.tools or [],
            }
            outcomes, passed = await stack.evaluate(comp)
            score, n_scored = score_outcomes(outcomes, weights)
            state["rollouts"] += 1
            if comp.error:
                score = 0.0
            elif n_scored == 0:
                state["unscored"] += 1
            return po.Rollout(
                row_id=row_id,
                row_name=case.name or row_id,
                score=score,
                feedback=render_feedback(
                    outcomes, comp, case.expected if include_expected else None
                ),
                prompt=render_input(case),
                output=_clip(comp.content or comp.reasoning, 2000),
                passed=passed,
                error=comp.error,
            )

        async def evaluate(texts: dict[str, str], row_ids: list[str]) -> list[po.Rollout]:
            # gather preserves order, which the score VECTOR depends on — index i
            # must mean the same row for every candidate or Pareto selection is
            # comparing different questions.
            out = await asyncio.gather(*[rollout(texts, rid) for rid in row_ids])
            state["metric_calls"] += len(row_ids)
            return list(out)

        async def reflect(component: str, current: str, examples: list[dict[str, Any]]) -> str:
            prompt = po.build_reflection_prompt(
                component, current, examples, str(cfg.get("reflection_guidance") or "")
            )
            body = {
                "model": reflection.get("model") or "",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": float(cfg.get("reflection_temperature") or 1.0),
                "max_tokens": int(
                    cfg.get("reflection_max_tokens") or DEFAULT_REFLECTION_MAX_TOKENS
                ),
            }
            body.update(reflection.get("extra_body") or {})
            comp = await call_once(
                client, reflection.get("base_url") or "",
                reflection.get("path") or "/v1/chat/completions",
                reflection_key, body, timeout_s,
            )
            state["reflection_calls"] += 1
            if comp.error:
                raise RuntimeError(comp.error)
            # A reasoning model can spend its whole reply in the reasoning
            # channel; the instruction is there rather than nowhere.
            return comp.content or comp.reasoning

        def on_event(kind: str, payload: Any) -> None:
            if kind == "iteration":
                state["iterations"].append(_iteration_payload(payload))
            elif kind == "seed":
                state["seed_score"] = round(float(payload.get("score") or 0.0), 4)

        # A flusher task rather than a flush inside `on_event`: the callback is
        # sync (the engine stays free of DB concerns) and the heartbeat has to
        # keep beating during a long validation sweep, not just between them.
        async def flusher() -> None:
            while True:
                await asyncio.sleep(PROGRESS_FLUSH_S)
                try:
                    async with session_factory()() as s2:
                        r2 = await s2.get(PromptOptimization, opt_id)
                        if r2 is not None:
                            r2.metric_calls = int(state["metric_calls"])
                            r2.reflection_calls = int(state["reflection_calls"])
                            r2.seed_score = state.get("seed_score")
                            r2.heartbeat_at = datetime.now(timezone.utc)
                            r2.result_json = {
                                **(r2.result_json or {}),
                                "iterations": list(state["iterations"]),
                                "progress": {
                                    "metric_calls": int(state["metric_calls"]),
                                    "budget": budget,
                                },
                            }
                        await s2.commit()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — a failed flush must not kill the run
                    logger.debug("prompt-opt %s progress flush failed", opt_id, exc_info=True)

        async with session_factory()() as s1:
            r1 = await s1.get(PromptOptimization, opt_id)
            if r1 is not None:
                r1.budget = budget
                r1.heartbeat_at = datetime.now(timezone.utc)
                await s1.commit()

        seed_texts = {
            "system_prompt": str(cfg.get("seed_prompt") or ""),
            "user_suffix": str(cfg.get("seed_user_suffix") or ""),
        }
        seed = {k: seed_texts.get(k, "") for k in components}

        beat = asyncio.create_task(flusher())
        try:
            result = await po.optimize(
                seed=seed,
                components=components,
                train_rows=train_rows,
                val_rows=val_rows,
                evaluate=evaluate,
                reflect=reflect,
                budget=budget,
                minibatch_size=minibatch,
                max_examples=max_examples,
                rng_seed=rng_seed,
                should_stop=lambda: opt_id in _CANCELLED,
                on_event=on_event,
            )
        finally:
            beat.cancel()
            try:
                await beat
            except asyncio.CancelledError:
                pass
            await stack.close()

        best = next(c for c in result.candidates if c.index == result.best)
        seed_cand = result.candidates[0]
        payload = {
            "components": components,
            "seed": _candidate_payload(seed_cand),
            "best": _candidate_payload(best),
            "improved": round(best.score - seed_cand.score, 4),
            "candidates": [_candidate_payload(c) for c in result.candidates],
            "iterations": [_iteration_payload(i) for i in result.iterations],
            "metric_calls": result.metric_calls,
            "reflection_calls": int(state["reflection_calls"]),
            "budget": budget,
            "stopped": result.stopped,
            "val_rows": len(val_rows),
            "train_rows": len(train_rows),
            # An in-sample run (too few rows to split) reports a gain measured on
            # the rows it tuned against; say so on the record itself.
            "in_sample": train_rows == val_rows,
            "n_rows": len(cases),
            "rollouts": int(state["rollouts"]),
            # > 0 means some replies were graded by NOTHING. All of them means the
            # whole run is meaningless however green the numbers look.
            "unscored_rollouts": int(state["unscored"]),
        }

        async with session_factory()() as s3:
            r3 = await s3.get(PromptOptimization, opt_id)
            if r3 is not None:
                r3.status = "cancelled" if result.stopped == "cancelled" else "completed"
                r3.result_json = payload
                r3.metric_calls = result.metric_calls
                r3.reflection_calls = int(state["reflection_calls"])
                r3.seed_score = round(seed_cand.score, 4)
                r3.best_score = round(best.score, 4)
                r3.ended_at = datetime.now(timezone.utc)
                await s3.commit()
        logger.info(
            "prompt optimization %s %s: %d metric calls, %d candidates, %.3f → %.3f",
            opt_id, result.stopped, result.metric_calls, len(result.candidates),
            seed_cand.score, best.score,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("prompt optimization %s failed", opt_id)
        if stack is not None:
            try:
                await stack.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            async with session_factory()() as s4:
                r4 = await s4.get(PromptOptimization, opt_id)
                if r4 is not None:
                    r4.status = "failed"
                    r4.error_text = f"{exc.__class__.__name__}: {exc}"[:4096]
                    r4.metric_calls = int(state["metric_calls"])
                    r4.ended_at = datetime.now(timezone.utc)
                    await s4.commit()
        except Exception:
            logger.exception("prompt optimization %s: could not record failure", opt_id)
    finally:
        _RUNNERS.pop(opt_id, None)
        _CANCELLED.discard(opt_id)


async def cleanup_orphaned_running() -> int:
    """Fail runs whose runner died, at startup — heartbeat-gated so a run being
    driven by another HA replica survives. Same rule as experiments."""
    reaped = 0
    try:
        cutoff = datetime.now(timezone.utc).timestamp() - STALE_HEARTBEAT_S
        async with session_factory()() as session:
            rows = (await session.execute(
                select(PromptOptimization).where(
                    PromptOptimization.status.in_(("queued", "running"))
                )
            )).scalars().all()
            for row in rows:
                hb = row.heartbeat_at
                if hb is not None and hb.timestamp() > cutoff:
                    continue
                if row.id in _RUNNERS:
                    continue
                row.status = "failed"
                row.error_text = "the gateway restarted while this optimization was running"
                row.ended_at = datetime.now(timezone.utc)
                reaped += 1
            if reaped:
                await session.commit()
                logger.info("marked %d orphaned prompt optimization(s) failed at startup", reaped)
    except Exception:
        logger.exception("prompt-optimization startup cleanup failed")
    return reaped


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

router = APIRouter(prefix="/v1", tags=["prompt-optimization"])


def _record(row: PromptOptimization, owner: str, dataset_name: str) -> OptimizationRecord:
    return OptimizationRecord(
        id=row.id, name=row.name, dataset_id=row.dataset_id, dataset_name=dataset_name,
        status=row.status, config=_public_opt_config(row.config_json or {}),
        result=row.result_json, budget=row.budget, metric_calls=row.metric_calls,
        reflection_calls=row.reflection_calls, seed_score=row.seed_score,
        best_score=row.best_score, error_text=row.error_text, owner=owner,
        created_at=row.created_at, started_at=row.started_at, ended_at=row.ended_at,
    )


async def _dataset_names(session: AsyncSession, ids: set[str]) -> dict[str, str]:
    if not ids:
        return {}
    rows = await session.execute(select(Dataset.id, Dataset.name).where(Dataset.id.in_(ids)))
    found = {i: n for i, n in rows.all()}
    return {i: found.get(i, "(deleted dataset)") for i in ids}


@router.get("/prompt-optimizations/limits")
async def optimization_limits(user: User = Depends(require_section(SECTION))):
    """Ceilings and presets, so the form can price a run BEFORE submitting it."""
    return {
        "max_metric_calls": MAX_METRIC_CALLS,
        "max_rows": MAX_OPT_ROWS,
        "default_rows": DEFAULT_ROWS,
        "max_concurrency": MAX_CONCURRENCY,
        "default_concurrency": DEFAULT_CONCURRENCY,
        "auto_budgets": AUTO_BUDGETS,
        "default_minibatch": 3,
        "components": [
            {"id": cid, "label": spec["label"], "description": spec["description"]}
            for cid, spec in po.COMPONENTS.items()
        ],
    }


@router.get("/prompt-optimizations/seed", response_model=SeedPromptResponse)
async def seed_prompt(
    dataset_id: str,
    limit: int = Query(50, ge=1, le=200),
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    """The dataset's own most-common system message, as the starting prompt.

    Optimizing a prompt the rows were never captured under measures a different
    thing than the one that misbehaved, so the form starts from what the corpus
    actually used.
    """
    d = await _require_source_dataset(session, dataset_id, user)
    cases = await resolve_cases(session, d, limit=limit)
    text, n_with, n_distinct = infer_seed_prompt(cases)
    return SeedPromptResponse(
        seed_prompt=text, n_rows=len(cases), n_with_system=n_with,
        distinct_system=n_distinct, source="dataset" if text else "none",
    )


@router.get("/prompt-optimizations/_page", response_model=OptimizationPage)
async def list_optimizations_page(
    q: str = "",
    status: str = "",
    dataset_id: str = "",
    limit: int = Query(12, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(PromptOptimization)
        # Everyone with the section sees every row — section access IS the
        # boundary here, so there is no mine-vs-all split (see main.list_apps).
    if status:
        stmt = stmt.where(PromptOptimization.status == status)
    if dataset_id:
        stmt = stmt.where(PromptOptimization.dataset_id == dataset_id)
    for tok in (q or "").lower().split():
        like = f"%{tok}%"
        stmt = stmt.where(or_(
            PromptOptimization.id.ilike(like),
            PromptOptimization.name.ilike(like),
            PromptOptimization.status.ilike(like),
            cast(PromptOptimization.config_json, Text).ilike(like),
        ))
    total = (await session.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    rows = (await session.execute(
        stmt.order_by(PromptOptimization.created_at.desc()).limit(limit).offset(offset)
    )).scalars().all()
    names = await _owner_names(session, {r.owner_id for r in rows})
    ds_names = await _dataset_names(session, {r.dataset_id for r in rows})
    return OptimizationPage(
        total=total,
        items=[
            _record(r, names.get(r.owner_id, "?"), ds_names.get(r.dataset_id, "?"))
            for r in rows
        ],
    )


@router.post("/prompt-optimizations", response_model=OptimizationRecord)
async def create_optimization(
    req: CreateOptimizationRequest,
    request: Request,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    name = (req.name or "").strip()
    if not _ID_RE.match(name):
        raise HTTPException(status_code=400, detail="invalid optimization name")
    ds = await _require_source_dataset(session, req.dataset_id, user)
    usable, why = dataset_usable(ds)
    if not usable:
        raise HTTPException(status_code=400, detail=f"dataset {ds.name} is not replayable: {why}")
    if not (req.target.base_url or "").strip() or not (req.target.model or "").strip():
        raise HTTPException(status_code=400, detail="the target needs a base URL and a model")

    components = [c for c in req.components if c in po.COMPONENTS]
    if not components:
        raise HTTPException(
            status_code=400,
            detail=f"pick at least one component to optimize: {', '.join(po.COMPONENTS)}",
        )
    # An optimizer with no metric has nothing to climb: it would replay the seed,
    # score every reply 0, and reject every proposal. Fail at create rather than
    # after the user has paid for a validation sweep.
    stored_evaluators = await snapshot_evaluators(session, user, req.evaluators)
    if not stored_evaluators:
        raise HTTPException(
            status_code=400,
            detail="pick at least one evaluator — it is the score GEPA optimizes against",
        )

    max_rows = min(req.max_rows or DEFAULT_ROWS, MAX_OPT_ROWS)
    known = ds.num_rows or max_rows
    n_rows = min(known, max_rows)
    n_val = min(req.val_rows, n_rows) if req.val_rows > 0 else auto_val_rows(n_rows)
    minibatch = max(1, req.minibatch_size)
    budget = resolve_budget(req.budget, req.max_metric_calls, n_val, minibatch)

    # Blank reflection fields mean "reuse the target" — a half-filled one is a
    # mistake worth surfacing rather than silently ignoring.
    reflection = req.reflection
    if reflection is not None and not (reflection.base_url or "").strip():
        reflection = None
    if reflection is not None and not (reflection.model or "").strip():
        raise HTTPException(status_code=400, detail="the reflection endpoint needs a model")

    cfg: dict[str, Any] = {
        "target": _store_targets([req.target])[0],
        "reflection": _store_targets([reflection])[0] if reflection else None,
        "reflection_guidance": (req.reflection_guidance or "").strip(),
        "reflection_temperature": float(req.reflection_temperature),
        "reflection_max_tokens": int(req.reflection_max_tokens or DEFAULT_REFLECTION_MAX_TOKENS),
        "evaluators": stored_evaluators,
        "components": components,
        "seed_prompt": req.seed_prompt or "",
        "seed_user_suffix": req.seed_user_suffix or "",
        "max_rows": max_rows,
        "val_rows": max(0, req.val_rows),
        "budget": req.budget,
        "max_metric_calls": max(0, req.max_metric_calls),
        "minibatch_size": minibatch,
        "max_examples": max(1, req.max_examples),
        "concurrency": max(1, min(MAX_CONCURRENCY, req.concurrency)),
        "timeout_s": max(1.0, req.timeout_s),
        "stream": bool(req.stream),
        "weights": {k: float(v) for k, v in (req.weights or {}).items()},
        "include_expected": bool(req.include_expected),
        "rng_seed": int(req.rng_seed),
    }

    row = PromptOptimization(
        id=f"opt-{secrets.token_hex(4)}", name=name, dataset_id=req.dataset_id,
        status="queued", config_json=cfg, budget=budget, owner_id=user.id,
    )
    session.add(row)
    await session.commit()

    task = asyncio.create_task(_run_optimization(request.app, row.id))
    _RUNNERS[row.id] = task
    logger.info(
        "prompt optimization %s queued by %s: %d rows (%d val) × budget %d metric calls",
        row.id, user.username, n_rows, n_val, budget,
    )
    return _record(row, user.username, ds.name)


async def _get_optimization(session: AsyncSession, opt_id: str, user: User) -> PromptOptimization:
    row = await session.get(PromptOptimization, opt_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such optimization")
    if row.owner_id != user.id and not user.is_admin:
        raise HTTPException(status_code=403, detail="not your optimization")
    return row


@router.get("/prompt-optimizations/{opt_id}", response_model=OptimizationRecord)
async def get_optimization(
    opt_id: str,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    row = await _get_optimization(session, opt_id, user)
    names = await _owner_names(session, {row.owner_id})
    ds_names = await _dataset_names(session, {row.dataset_id})
    return _record(row, names.get(row.owner_id, "?"), ds_names.get(row.dataset_id, "?"))


@router.post("/prompt-optimizations/{opt_id}/cancel", response_model=OptimizationRecord)
async def cancel_optimization(
    opt_id: str,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    """Stop after the current iteration. The candidates found so far are kept and
    summarized — a partial search still has a best prompt."""
    row = await _get_optimization(session, opt_id, user)
    if row.status not in ("queued", "running"):
        raise HTTPException(status_code=400, detail=f"optimization is {row.status}")
    _CANCELLED.add(opt_id)
    logger.info("prompt optimization %s cancel requested by %s", opt_id, user.username)
    names = await _owner_names(session, {row.owner_id})
    ds_names = await _dataset_names(session, {row.dataset_id})
    return _record(row, names.get(row.owner_id, "?"), ds_names.get(row.dataset_id, "?"))


@router.delete("/prompt-optimizations/{opt_id}")
async def delete_optimization(
    opt_id: str,
    user: User = Depends(require_section(SECTION)),
    session: AsyncSession = Depends(get_session),
):
    row = await _get_optimization(session, opt_id, user)
    if row.status == "running":
        raise HTTPException(status_code=400, detail="cancel the run before deleting it")
    await session.delete(row)
    await session.commit()
    logger.info("prompt optimization %s deleted by %s", opt_id, user.username)
    return {"ok": True}
