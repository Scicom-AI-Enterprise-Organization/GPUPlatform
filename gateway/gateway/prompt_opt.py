"""GEPA — reflective prompt evolution, the algorithm half.

GEPA (Genetic-Pareto, https://dspy.ai/getting-started/gepa-optimization/) improves
a prompt by *reading its own failures*: run the prompt, score each reply, hand the
low scorers plus a written critique to a reflection LM, and ask it for a better
instruction. Keep whatever wins, sample the next parent from the Pareto frontier
so the search doesn't collapse onto one strategy, repeat until the budget is out.

This module is the algorithm ONLY — no HTTP, no database, no evaluator registry.
Everything that touches the world arrives as an injected coroutine:

    evaluate(texts, row_ids) -> [Rollout]     replay + score N rows under a candidate
    reflect(component, current, examples)     ask the reflection LM for a new text

which is what makes the search unit-testable with a two-line fake (see
`tests/unit/test_prompt_opt.py`) and lets `prompt_opt_api.py` own all the
platform glue — targets, evaluators, cancellation, cost accounting.

Why not the `gepa` package (or dspy)? Its base install pulls litellm, mlflow,
wandb, datasets, pandas and pyarrow onto the gateway image, and its engine is
synchronous — it would have to be thread-bridged back into the runner's event
loop, past our cancel flag and heartbeat. DSPy is a further mismatch: it
optimizes typed `Signature` fields, while Experiments replays raw captured chat
messages that have no field structure to bind to. The search below follows the
published algorithm (reflective mutation, minibatch accept/reject, Pareto parent
sampling on a per-instance score vector); the pieces it deliberately leaves out
are noted at `optimize()`.

Vocabulary note: a "metric call" here is **one real, billed request** to the
target endpoint. Budget is denominated in those, never in iterations, because
that is the number that costs money.
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("gateway.prompt_opt")


# --------------------------------------------------------------------------- #
# Components — the pieces of the request GEPA is allowed to rewrite
# --------------------------------------------------------------------------- #

# A candidate is a {component: text} mapping, not a bare string, because the
# search generalizes to several mutable slots at once (the paper's multi-module
# case). Each key here maps onto a field of the experiment's VariantSpec, so an
# optimized candidate can be replayed by the ordinary experiment runner with no
# special-casing — that's what makes "use this prompt in an experiment" one click.
COMPONENTS: dict[str, dict[str, str]] = {
    "system_prompt": {
        "label": "System prompt",
        "variant_field": "system_override",
        "description": "Replaces the system message on every replayed row.",
        "role": "the system prompt that configures the assistant",
    },
    "user_suffix": {
        "label": "User-turn reminder",
        "variant_field": "user_suffix",
        "description": "Appended to the last user message — a just-in-time reminder.",
        "role": "a short reminder appended to the end of the user's message",
    },
}
DEFAULT_COMPONENTS = ("system_prompt",)

# A reflection LM that runs away (repeating the examples back, emitting a whole
# transcript) produces an instruction nobody can afford to send on every request.
MAX_INSTRUCTION_CHARS = 20000


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #


@dataclass
class Rollout:
    """One row replayed under one candidate, scored.

    `feedback` is the whole point: a bare number tells the reflection LM that
    something is wrong, the text tells it *what*. It is rendered from the
    evaluators' own `reason` strings, so every detector on the platform — the
    built-ins, a user's expression evaluator, an LLM judge — is already a GEPA
    feedback source with no extra authoring.
    """
    row_id: str
    score: float
    feedback: str
    row_name: str = ""
    prompt: str = ""
    output: str = ""
    passed: bool = True
    error: Optional[str] = None


@dataclass
class Candidate:
    """A point in the search: one full set of component texts plus its score
    vector over the validation rows.

    The **vector**, not its mean, is what Pareto selection needs — a candidate
    that is mediocre on average but the single best answer to three awkward rows
    is exactly the parent worth mutating next, and an average would hide it.
    """
    index: int
    texts: dict[str, str]
    parent: Optional[int] = None
    iteration: int = 0
    component: Optional[str] = None
    origin: str = "seed"  # seed | mutation | merge
    val_scores: list[float] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.val_scores:
            return 0.0
        return sum(self.val_scores) / len(self.val_scores)


@dataclass
class IterationLog:
    """One accept/reject decision, as the UI renders it."""
    i: int
    parent: int
    component: str
    origin: str
    row_ids: list[str]
    parent_score: float
    child_score: Optional[float] = None
    accepted: bool = False
    val_score: Optional[float] = None
    candidate: Optional[int] = None
    calls: int = 0
    note: str = ""
    proposal: str = ""
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OptimizeResult:
    candidates: list[Candidate]
    iterations: list[IterationLog]
    best: int
    metric_calls: int
    stopped: str  # budget | iterations | cancelled


# --------------------------------------------------------------------------- #
# Pareto candidate selection
# --------------------------------------------------------------------------- #


def pareto_frontier(pool: list[Candidate]) -> dict[int, int]:
    """Candidates worth mutating, mapped to how many rows each one *wins*.

    For every validation row, find the best score any candidate reached and note
    which candidates reached it. A candidate whose winning-row set is a strict
    subset of another's is dominated — anything it can do, the other does too —
    and is dropped. What survives is the frontier, weighted by breadth.

    This is the part that keeps the search from collapsing. Hill-climbing on the
    mean walks into the first local optimum and stays there; sampling the
    frontier keeps a specialist alive as long as it owns even one row.
    """
    scored = [c for c in pool if c.val_scores]
    if not scored:
        return {}
    n = min(len(c.val_scores) for c in scored)
    wins: dict[int, set[int]] = {}
    for i in range(n):
        best = max(c.val_scores[i] for c in scored)
        for c in scored:
            # Ties all count as wins: two candidates that both nail a row are
            # both legitimate parents for it.
            if c.val_scores[i] >= best - 1e-9:
                wins.setdefault(c.index, set()).add(i)

    kept: dict[int, int] = {}
    for cid, rows in wins.items():
        if any(cid != other and rows < other_rows for other, other_rows in wins.items()):
            continue  # strictly dominated
        kept[cid] = len(rows)
    # Every score identical (a flat metric) leaves every candidate tied and
    # nothing dominated — fall back to a uniform draw rather than an empty set.
    return kept or {c.index: 1 for c in scored}


def select_candidate(pool: list[Candidate], rng: random.Random) -> Candidate:
    """Sample a parent from the frontier, proportional to rows won."""
    weights = pareto_frontier(pool)
    if not weights:
        return pool[-1]
    ids = sorted(weights)
    pick = rng.choices(ids, weights=[weights[i] for i in ids], k=1)[0]
    return next(c for c in pool if c.index == pick)


# --------------------------------------------------------------------------- #
# The reflection meta-prompt
# --------------------------------------------------------------------------- #

_META_PROMPT = """I gave an assistant the following instruction, used as {role}:

```
{current}
```

Below are task inputs the assistant was given, the response it produced, and
feedback on how each response should have been better.

{examples}

Your task is to write a NEW instruction for the assistant.

Read every example carefully. Identify the domain-specific facts, conventions and
constraints the feedback reveals — the assistant will NOT see these examples at
run time, so anything it needs to know has to be stated in the instruction
itself. If the successful responses share a strategy, spell that strategy out.
Keep instructions that are already working; fix only what the feedback shows is
broken. Write it as a direct instruction to the assistant, not as commentary
about these examples, and do not refer to "the examples" — the assistant cannot
see them.

Return ONLY the new instruction, inside a single ``` code block."""

_EXAMPLE_TEMPLATE = """### Example {n} (score {score:.2f})
Task input:
```
{prompt}
```
Assistant response:
```
{output}
```
Feedback:
```
{feedback}
```"""


def build_reflection_prompt(
    component: str,
    current: str,
    examples: list[dict[str, Any]],
    guidance: str = "",
) -> str:
    """Render the meta-prompt handed to the reflection LM.

    Deliberately close to the published GEPA meta-prompt: it is the part of the
    method that was actually tuned, and drifting from it for style silently
    changes the optimizer's behaviour.
    """
    role = COMPONENTS.get(component, {}).get("role", "an instruction")
    body = "\n\n".join(
        _EXAMPLE_TEMPLATE.format(
            n=i + 1,
            score=float(ex.get("score") or 0.0),
            prompt=(ex.get("prompt") or "(no textual input)").strip(),
            output=(ex.get("output") or "(empty response)").strip(),
            feedback=(ex.get("feedback") or "(no feedback)").strip(),
        )
        for i, ex in enumerate(examples)
    )
    prompt = _META_PROMPT.format(role=role, current=current.strip() or "(empty)", examples=body)
    if guidance.strip():
        prompt += f"\n\nAdditional requirements for the new instruction:\n{guidance.strip()}"
    return prompt


_FENCE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


def extract_instruction(reply: str) -> str:
    """Pull the proposed instruction out of the reflection LM's reply.

    Take the LONGEST fenced block, not the first: a chatty model likes to quote
    the old instruction back before offering the new one, and the first block is
    then the thing we already had. Unfenced replies are used whole — refusing
    them would throw away a perfectly good proposal over formatting.
    """
    blocks = [b.strip() for b in _FENCE_BLOCK_RE.findall(reply or "")]
    if blocks:
        return max(blocks, key=len)
    return (reply or "").strip()


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #

EvaluateFn = Callable[[dict[str, str], list[str]], Awaitable[list[Rollout]]]
ReflectFn = Callable[[str, str, list[dict[str, Any]]], Awaitable[str]]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


class _Minibatches:
    """Round-robin minibatches over a shuffled train set.

    Round-robin rather than random draws: over a short budget, sampling with
    replacement leaves rows the optimizer never sees, and an unseen row is a
    failure mode the reflection LM is never told about.
    """

    def __init__(self, row_ids: list[str], size: int, rng: random.Random) -> None:
        self._rows = list(row_ids)
        rng.shuffle(self._rows)
        self.size = max(1, min(size, len(self._rows) or 1))
        self._cursor = 0

    def next(self) -> list[str]:
        if not self._rows:
            return []
        out: list[str] = []
        while len(out) < self.size:
            if self._cursor >= len(self._rows):
                self._cursor = 0
            out.append(self._rows[self._cursor])
            self._cursor += 1
        return out


async def optimize(
    *,
    seed: dict[str, str],
    components: list[str],
    train_rows: list[str],
    val_rows: list[str],
    evaluate: EvaluateFn,
    reflect: ReflectFn,
    budget: int,
    minibatch_size: int = 3,
    max_examples: int = 5,
    max_iterations: Optional[int] = None,
    rng_seed: int = 0,
    should_stop: Callable[[], bool] = lambda: False,
    on_event: Optional[Callable[[str, Any], None]] = None,
) -> OptimizeResult:
    """Run the search until the metric-call budget is spent.

    One iteration:
      1. sample a parent from the Pareto frontier
      2. replay it over the next train minibatch  →  scores + written feedback
      3. ask the reflection LM to rewrite ONE component from that feedback
      4. replay the child over the SAME minibatch; keep it only if it scores better
      5. a kept child is scored over the whole validation set, which is what
         earns it a place on the frontier

    Budget is checked *before* an iteration starts and covers its worst case
    (two minibatch passes plus a validation sweep), so a run cannot overshoot the
    number of billed calls the user approved. The cost of that guarantee is up to
    one validation sweep of unspent budget at the end.

    Deliberately not implemented: the paper's system-aware **merge** across
    lineages, which recombines components optimized in different branches. With
    the single-component default there is nothing to recombine, and a wrong merge
    is indistinguishable from a bad mutation in the logs. `Candidate.origin`
    exists so it can be added without a schema change.
    """
    emit = on_event or (lambda kind, payload: None)
    rng = random.Random(rng_seed)
    components = [c for c in components if c in COMPONENTS] or list(DEFAULT_COMPONENTS)
    val_rows = list(val_rows)
    train_rows = list(train_rows) or list(val_rows)
    calls = 0

    # Re-evaluating a parent on a minibatch it has already seen is a pure waste of
    # real money — round-robin batches mean a surviving parent meets the same rows
    # again within a few iterations.
    cache: dict[tuple[int, tuple[str, ...]], list[Rollout]] = {}

    async def run(cand: Candidate, rows: list[str]) -> list[Rollout]:
        nonlocal calls
        key = (cand.index, tuple(rows))
        hit = cache.get(key)
        if hit is not None:
            return hit
        out = await evaluate(cand.texts, rows)
        calls += len(rows)
        cache[key] = out
        return out

    seed_cand = Candidate(index=0, texts=dict(seed), origin="seed")
    pool = [seed_cand]

    # The seed's validation vector is the baseline every later candidate is read
    # against, so it is charged to the budget before the loop starts.
    if val_rows:
        seed_rollouts = await evaluate(seed_cand.texts, val_rows)
        calls += len(val_rows)
        seed_cand.val_scores = [r.score for r in seed_rollouts]
    emit("seed", {"score": seed_cand.score, "calls": calls})

    batches = _Minibatches(train_rows, minibatch_size, rng)
    iterations: list[IterationLog] = []
    per_iteration = 2 * batches.size + len(val_rows)
    # ⚠ The budget alone does NOT bound the loop, because the rollout cache can
    # make an iteration cost zero metric calls: a reflection model that keeps
    # proposing the same text hits a cached parent, is rejected before the child
    # is ever run, spends nothing — and spins forever, billing the *reflection*
    # endpoint on every turn. So cap the turns at what the budget could pay for
    # if nothing were cached; caching then buys extra search inside that bound
    # rather than an unbounded loop.
    if max_iterations is None:
        max_iterations = max(4, budget // max(1, 2 * batches.size))
    stopped = "budget"
    turn = 0

    while calls + per_iteration <= budget:
        if should_stop():
            stopped = "cancelled"
            break
        if turn >= max_iterations:
            stopped = "iterations"
            break
        turn += 1
        parent = select_candidate(pool, rng)
        # Round-robin the component too: with one component this is a no-op, with
        # several it stops the search from polishing the first one forever.
        component = components[(turn - 1) % len(components)]
        rows = batches.next()
        log = IterationLog(
            i=turn, parent=parent.index, component=component, origin="mutation",
            row_ids=list(rows), parent_score=0.0,
        )

        before = calls
        parent_rollouts = await run(parent, rows)
        log.parent_score = round(_mean([r.score for r in parent_rollouts]), 4)

        # Worst first: a perfect minibatch teaches the reflection LM nothing, and
        # the failures are where the domain knowledge it needs is hiding.
        ranked = sorted(parent_rollouts, key=lambda r: r.score)[:max_examples]
        examples = [
            {
                "row_id": r.row_id, "row_name": r.row_name, "prompt": r.prompt,
                "output": r.output, "feedback": r.feedback, "score": r.score,
            }
            for r in ranked
        ]
        log.examples = examples

        try:
            reply = await reflect(component, parent.texts.get(component, ""), examples)
        except Exception as exc:  # noqa: BLE001 — a reflection outage skips a turn, never kills the run
            logger.warning("reflection failed on iteration %d: %s", turn, exc)
            reply = ""
        proposal = extract_instruction(reply)

        rejected = None
        if not proposal:
            rejected = "the reflection model returned no instruction"
        elif len(proposal) > MAX_INSTRUCTION_CHARS:
            rejected = f"proposed instruction is {len(proposal)} chars (max {MAX_INSTRUCTION_CHARS})"
        elif proposal.strip() == (parent.texts.get(component, "") or "").strip():
            rejected = "the reflection model proposed the current instruction unchanged"
        if rejected:
            log.note = rejected
            log.calls = calls - before
            iterations.append(log)
            emit("iteration", log)
            continue

        log.proposal = proposal
        child_texts = {**parent.texts, component: proposal}
        child = Candidate(
            index=len(pool), texts=child_texts, parent=parent.index,
            iteration=turn, component=component, origin="mutation",
        )
        child_rollouts = await evaluate(child_texts, rows)
        calls += len(rows)
        cache[(child.index, tuple(rows))] = child_rollouts
        log.child_score = round(_mean([r.score for r in child_rollouts]), 4)

        # Strictly better, not merely equal: an equal child costs a full
        # validation sweep to learn it changed nothing, and admits drift.
        if log.child_score is None or log.child_score <= log.parent_score:
            log.note = "no improvement on the minibatch"
            log.calls = calls - before
            iterations.append(log)
            emit("iteration", log)
            continue

        if val_rows:
            val_rollouts = await evaluate(child_texts, val_rows)
            calls += len(val_rows)
            child.val_scores = [r.score for r in val_rollouts]
        else:
            child.val_scores = [log.child_score]
        pool.append(child)
        log.accepted = True
        log.candidate = child.index
        log.val_score = round(child.score, 4)
        log.calls = calls - before
        log.note = (
            f"kept — validation {child.score:.3f} vs seed {seed_cand.score:.3f}"
        )
        iterations.append(log)
        emit("iteration", log)

    if should_stop():
        stopped = "cancelled"

    # Ties go to the earlier candidate: it is the shorter lineage and the one
    # that survived more comparisons.
    best = max(pool, key=lambda c: (round(c.score, 6), -c.index)).index
    return OptimizeResult(
        candidates=pool, iterations=iterations, best=best,
        metric_calls=calls, stopped=stopped,
    )
