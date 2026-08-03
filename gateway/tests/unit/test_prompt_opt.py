"""Unit tests for GEPA prompt optimization — the search and the metric bridge.

No stack, no HTTP: `prompt_opt.optimize` takes its world as two coroutines, so a
whole search runs against a scripted fake in milliseconds. What's pinned here is
the behaviour that costs money or silently corrupts a result — budget never
overshoots, an abstaining evaluator is not a pass, the score vector stays aligned
with the validation rows, and a dominated candidate never gets picked as a parent.
"""
from __future__ import annotations

import pytest

from gateway import evaluators as ev
from gateway import prompt_opt as po
from gateway import prompt_opt_api as api


# --------------------------------------------------------------------------- #
# Pareto frontier
# --------------------------------------------------------------------------- #


def _cand(index: int, scores: list[float]) -> po.Candidate:
    return po.Candidate(index=index, texts={"system_prompt": f"p{index}"}, val_scores=scores)


def test_pareto_drops_strictly_dominated_candidates():
    # b wins nothing a doesn't also win → dominated, must never be a parent.
    a = _cand(0, [1.0, 1.0, 0.0])
    b = _cand(1, [1.0, 0.5, 0.0])
    front = po.pareto_frontier([a, b])
    assert 1 not in front
    assert front[0] == 3  # ties count: a is best-or-equal on all three rows


def test_pareto_keeps_a_specialist_with_a_worse_mean():
    # generalist has the better average; specialist owns row 2 outright. GEPA's
    # whole point is that the specialist survives as a parent.
    generalist = _cand(0, [0.9, 0.9, 0.0])
    specialist = _cand(1, [0.1, 0.1, 1.0])
    front = po.pareto_frontier([generalist, specialist])
    assert set(front) == {0, 1}
    assert front[0] == 2 and front[1] == 1


def test_pareto_flat_scores_fall_back_to_uniform():
    pool = [_cand(0, [0.5, 0.5]), _cand(1, [0.5, 0.5])]
    assert po.pareto_frontier(pool) == {0: 2, 1: 2}


# --------------------------------------------------------------------------- #
# Reflection prompt / proposal parsing
# --------------------------------------------------------------------------- #


def test_extract_instruction_takes_the_longest_fenced_block():
    # A chatty reflector quotes the OLD instruction first; taking the first block
    # would "optimize" straight back to the input.
    reply = "Here is the current one:\n```\nold\n```\nAnd my new version:\n```\na much better instruction\n```"
    assert po.extract_instruction(reply) == "a much better instruction"


def test_extract_instruction_accepts_an_unfenced_reply():
    assert po.extract_instruction("  just prose  ") == "just prose"


def test_reflection_prompt_carries_feedback_and_guidance():
    prompt = po.build_reflection_prompt(
        "system_prompt",
        "be brief",
        [{"prompt": "hi", "output": "hello", "feedback": "FAIL language: replied in english", "score": 0.0}],
        guidance="always answer in Malay",
    )
    assert "be brief" in prompt
    assert "replied in english" in prompt
    assert "always answer in Malay" in prompt
    assert "system prompt" in prompt  # the component's role, not a bare id


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #


def _fake_world(scores_by_text: dict[str, float], proposals: list[str]):
    """A scripted world: each prompt has a fixed score, reflection emits `proposals`."""
    calls = {"evaluate": 0, "reflect": 0, "rows": 0}

    async def evaluate(texts, row_ids):
        calls["evaluate"] += 1
        calls["rows"] += len(row_ids)
        score = scores_by_text.get(texts["system_prompt"], 0.0)
        return [
            po.Rollout(row_id=r, score=score, feedback=f"score {score}", output="out")
            for r in row_ids
        ]

    async def reflect(component, current, examples):
        i = calls["reflect"]
        calls["reflect"] += 1
        return f"```\n{proposals[i % len(proposals)]}\n```"

    return evaluate, reflect, calls


@pytest.mark.asyncio
async def test_optimize_keeps_the_better_prompt():
    evaluate, reflect, _ = _fake_world({"seed": 0.2, "better": 0.9}, ["better"])
    res = await po.optimize(
        seed={"system_prompt": "seed"}, components=["system_prompt"],
        train_rows=["t1", "t2"], val_rows=["v1", "v2"],
        evaluate=evaluate, reflect=reflect, budget=200, minibatch_size=2,
    )
    best = next(c for c in res.candidates if c.index == res.best)
    assert best.texts["system_prompt"] == "better"
    assert best.score == pytest.approx(0.9)
    assert res.candidates[0].score == pytest.approx(0.2)  # the seed is candidate 0


@pytest.mark.asyncio
async def test_optimize_rejects_a_worse_prompt_and_never_scores_it_on_val():
    evaluate, reflect, calls = _fake_world({"seed": 0.8, "worse": 0.1}, ["worse"])
    res = await po.optimize(
        seed={"system_prompt": "seed"}, components=["system_prompt"],
        train_rows=["t1"], val_rows=["v1", "v2"],
        evaluate=evaluate, reflect=reflect, budget=60, minibatch_size=1,
    )
    assert len(res.candidates) == 1  # nothing was admitted to the pool
    assert all(not it.accepted for it in res.iterations)
    # A rejected child costs its own minibatch and nothing else: the seed's
    # validation sweep (2), the parent's minibatch once (cached thereafter), and
    # one child rollout per turn. No validation sweep is ever spent on a loser —
    # that is the expensive half.
    assert calls["rows"] == 2 + 1 + len(res.iterations)


@pytest.mark.asyncio
async def test_optimize_never_exceeds_the_budget():
    evaluate, reflect, _ = _fake_world({"seed": 0.1, "a": 0.2, "b": 0.3}, ["a", "b"])
    budget = 37
    res = await po.optimize(
        seed={"system_prompt": "seed"}, components=["system_prompt"],
        train_rows=[f"t{i}" for i in range(6)], val_rows=[f"v{i}" for i in range(5)],
        evaluate=evaluate, reflect=reflect, budget=budget, minibatch_size=3,
    )
    assert res.metric_calls <= budget
    assert res.stopped == "budget"


@pytest.mark.asyncio
async def test_optimize_stops_on_cancel_and_still_returns_a_best():
    evaluate, reflect, _ = _fake_world({"seed": 0.1, "better": 0.9}, ["better"])
    stop = {"now": False}

    async def reflect_then_cancel(component, current, examples):
        stop["now"] = True
        return await reflect(component, current, examples)

    res = await po.optimize(
        seed={"system_prompt": "seed"}, components=["system_prompt"],
        train_rows=["t1"], val_rows=["v1"],
        evaluate=evaluate, reflect=reflect_then_cancel, budget=1000, minibatch_size=1,
        should_stop=lambda: stop["now"],
    )
    assert res.stopped == "cancelled"
    assert res.candidates  # a partial search still has a best prompt


@pytest.mark.asyncio
async def test_identical_proposal_is_rejected_without_a_second_rollout():
    evaluate, reflect, calls = _fake_world({"seed": 0.5}, ["seed"])
    res = await po.optimize(
        seed={"system_prompt": "seed"}, components=["system_prompt"],
        train_rows=["t1"], val_rows=["v1"],
        evaluate=evaluate, reflect=reflect, budget=20, minibatch_size=1,
    )
    assert len(res.candidates) == 1
    assert all("unchanged" in it.note for it in res.iterations)
    # The child is never rolled out: 1 seed validation row + 1 parent minibatch
    # (cached from then on) is the entire bill, however many turns it takes.
    assert calls["rows"] == 2


@pytest.mark.asyncio
async def test_a_zero_cost_iteration_cannot_spin_forever():
    """The regression that hung the first test run.

    The rollout cache can make a whole iteration cost NO metric calls — a
    reflection model stuck proposing the current text hits a cached parent and is
    rejected before the child runs. The budget guard alone never advances then,
    so the loop spins, billing the reflection endpoint on every turn. The
    iteration ceiling is what stops it.
    """
    evaluate, reflect, calls = _fake_world({"seed": 0.5}, ["seed"])
    res = await po.optimize(
        seed={"system_prompt": "seed"}, components=["system_prompt"],
        train_rows=["t1"], val_rows=["v1"],
        evaluate=evaluate, reflect=reflect, budget=1000, minibatch_size=1,
    )
    assert res.stopped == "iterations"
    assert res.metric_calls < 10          # the budget was never touched…
    assert calls["reflect"] == len(res.iterations) <= 500  # …and reflection stayed bounded


@pytest.mark.asyncio
async def test_reflection_failure_skips_the_turn_instead_of_killing_the_run():
    evaluate, _, _ = _fake_world({"seed": 0.5}, ["x"])

    async def exploding_reflect(component, current, examples):
        raise RuntimeError("reflection endpoint down")

    res = await po.optimize(
        seed={"system_prompt": "seed"}, components=["system_prompt"],
        train_rows=["t1"], val_rows=["v1"],
        evaluate=evaluate, reflect=exploding_reflect, budget=20, minibatch_size=1,
    )
    assert res.iterations and all(not it.accepted for it in res.iterations)


@pytest.mark.asyncio
async def test_parent_minibatch_results_are_cached_across_iterations():
    # Round-robin batches mean a surviving parent meets the same rows again;
    # re-billing those rollouts is pure waste.
    evaluate, reflect, calls = _fake_world({"seed": 0.5}, ["seed"])
    await po.optimize(
        seed={"system_prompt": "seed"}, components=["system_prompt"],
        train_rows=["t1"], val_rows=["v1"],
        evaluate=evaluate, reflect=reflect, budget=30, minibatch_size=1,
    )
    # 1 seed val + exactly ONE parent minibatch rollout, reused thereafter.
    assert calls["rows"] == 2


@pytest.mark.asyncio
async def test_multi_component_rotates_which_slot_is_rewritten():
    seen: list[str] = []

    async def evaluate(texts, row_ids):
        return [po.Rollout(row_id=r, score=0.5, feedback="f") for r in row_ids]

    async def reflect(component, current, examples):
        seen.append(component)
        return ""  # no proposal: keeps the pool at one candidate, cheap turns

    await po.optimize(
        seed={"system_prompt": "s", "user_suffix": "u"},
        components=["system_prompt", "user_suffix"],
        train_rows=["t1"], val_rows=["v1"],
        evaluate=evaluate, reflect=reflect, budget=20, minibatch_size=1,
    )
    assert seen[:4] == ["system_prompt", "user_suffix", "system_prompt", "user_suffix"]


# --------------------------------------------------------------------------- #
# The metric bridge: evaluator verdicts → (score, feedback)
# --------------------------------------------------------------------------- #


def test_always_on_diagnostics_are_not_part_of_the_objective():
    # request_error passes unconditionally on a successful request. Counting it
    # would score a prompt that fails its only real check 0.5 instead of 0, and
    # halve every reported gain.
    outcomes = [
        ev.EvalOutcome(id="json_output", passed=False, score=0.0),
        ev.EvalOutcome(id="request_error", passed=True, score=1.0),
    ]
    score, n = api.score_outcomes(outcomes)
    assert n == 1 and score == pytest.approx(0.0)


def test_abstaining_evaluators_do_not_count_as_passes():
    # A skipped detector means "no reference data on this row". Scoring it 1.0
    # would teach the optimizer that an unscored reply is a perfect reply.
    outcomes = [
        ev.EvalOutcome(id="a", passed=True, score=1.0),
        ev.EvalOutcome(id="skipped_one", passed=True, flags={"skipped": True}),
        ev.EvalOutcome(id="broken", passed=True, flags={"evaluator_error": True}),
    ]
    score, n = api.score_outcomes(outcomes)
    assert n == 1 and score == pytest.approx(1.0)


def test_score_is_the_weighted_mean_of_the_detectors_that_scored():
    outcomes = [
        ev.EvalOutcome(id="a", passed=False, score=0.0),
        ev.EvalOutcome(id="b", passed=True, score=1.0),
    ]
    assert api.score_outcomes(outcomes)[0] == pytest.approx(0.5)
    assert api.score_outcomes(outcomes, {"b": 3.0})[0] == pytest.approx(0.75)


def test_passed_flag_stands_in_for_a_missing_numeric_score():
    outcomes = [ev.EvalOutcome(id="a", passed=False), ev.EvalOutcome(id="b", passed=True)]
    assert api.score_outcomes(outcomes)[0] == pytest.approx(0.5)


def test_feedback_names_the_failing_rule_and_hides_abstentions():
    outcomes = [
        ev.EvalOutcome(id="json_output", passed=False, reason="trailing comma at 142"),
        ev.EvalOutcome(id="latency", passed=True),
        ev.EvalOutcome(id="quiet", passed=True, flags={"skipped": True}),
    ]
    text = api.render_feedback(outcomes, ev.Completion(content="{...}"), {"answer": "42", "_priv": 1})
    assert "FAIL json_output: trailing comma at 142" in text
    assert "PASS latency" in text
    assert "quiet" not in text
    assert '"answer": "42"' in text
    assert "_priv" not in text  # runner-internal keys stay out of the prompt


def test_feedback_for_a_transport_error_says_so():
    text = api.render_feedback([], ev.Completion(error="HTTP 500: boom"), None)
    assert "HTTP 500" in text


def test_feedback_says_when_nothing_scored_the_reply():
    text = api.render_feedback(
        [ev.EvalOutcome(id="a", passed=True, flags={"skipped": True})],
        ev.Completion(content="hi"), None,
    )
    assert "No evaluator produced a verdict" in text


# --------------------------------------------------------------------------- #
# Budget + split
# --------------------------------------------------------------------------- #


def test_budget_floor_guarantees_at_least_one_iteration():
    # A budget below "seed sweep + one iteration" would return the input prompt
    # and call it a result.
    assert api.resolve_budget("custom", 1, n_val=10, minibatch=3) == 2 * 10 + 2 * 3


def test_budget_presets_scale_with_the_validation_set():
    assert api.resolve_budget("light", 0, n_val=20, minibatch=3) == 20 * api.AUTO_BUDGETS["light"]
    assert api.resolve_budget("heavy", 0, n_val=20, minibatch=3) == 20 * api.AUTO_BUDGETS["heavy"]


def test_budget_is_clamped_to_the_hard_ceiling():
    assert api.resolve_budget("custom", 10_000_000, n_val=5, minibatch=3) == api.MAX_METRIC_CALLS


def test_split_is_disjoint_and_deterministic():
    import random
    rows = [f"r{i}" for i in range(10)]
    train, val = api.split_rows(rows, 4, random.Random(7))
    assert len(val) == 4 and len(train) == 6
    assert not set(train) & set(val)
    assert api.split_rows(rows, 4, random.Random(7)) == (train, val)


def test_tiny_dataset_reuses_val_rows_for_reflection():
    import random
    train, val = api.split_rows(["only"], 1, random.Random(0))
    assert train == val == ["only"]


class _Case:
    def __init__(self, messages, tools=None, name="row"):
        self.messages, self.tools, self.name = messages, tools, name


def test_seed_prompt_is_the_datasets_own_most_common_system_message():
    cases = [
        _Case([{"role": "system", "content": "you are a bot"}, {"role": "user", "content": "hi"}]),
        _Case([{"role": "system", "content": "you are a bot"}, {"role": "user", "content": "yo"}]),
        _Case([{"role": "system", "content": "other"}, {"role": "user", "content": "hm"}]),
    ]
    text, n_with, distinct = api.infer_seed_prompt(cases)
    assert text == "you are a bot" and n_with == 3 and distinct == 2


def test_seed_prompt_is_empty_when_no_row_has_a_system_message():
    assert api.infer_seed_prompt([_Case([{"role": "user", "content": "hi"}])]) == ("", 0, 0)


def test_render_input_hides_the_system_turn_being_optimized():
    case = _Case(
        [
            {"role": "system", "content": "SECRET SEED PROMPT"},
            {"role": "user", "content": "what is the weather"},
        ],
        tools=[{"function": {"name": "get_weather"}}],
    )
    text = api.render_input(case)
    assert "SECRET SEED PROMPT" not in text
    assert "[user] what is the weather" in text
    assert "get_weather" in text
