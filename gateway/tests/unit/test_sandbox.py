"""Sandboxes: call identity, the replay provider, the response cache, and the
per-turn scoring fold. Pure logic — no DB, no network.

The tests that matter most here pin the two things a naive implementation gets
wrong and NOBODY notices, because both produce a green run:

  * `test_final_turn_projection_would_zero_tool_call_f1` — scoring a trajectory's
    LAST turn hands `function_call_units` an empty call list against a non-empty
    reference, collapsing tool-call F1 to 0. That reads as a model regression.
  * `test_leak_in_middle_turn_fails_the_sample` — a model that degenerated in
    round 2 and recovered in round 5 must not read clean.
"""
import json

import pytest

from gateway import evaluators as ev
from gateway import sandbox as sb
from gateway.experiments_api import (
    Case,
    _normalize_calls,
    _sandbox_rollup,
    _sum_usage,
    _trajectory_payload,
    summarize,
)


def _call(name, args=None, cid="c1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args if args is not None else "{}"}}


def _spec(**cfg):
    return sb.SandboxSpec(id="sb-1", name="s", mode="replay", config=cfg)


# --------------------------------------------------------------------------- #
# Call identity
# --------------------------------------------------------------------------- #


def test_canonical_args_is_key_order_insensitive():
    # Key order forking the cache would hand two identical calls two different
    # worlds — the exact thing the cache exists to prevent.
    assert sb.canonical_args('{"a": 1, "b": 2}') == sb.canonical_args('{"b": 2, "a": 1}')
    assert sb.canonical_args({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'


def test_canonical_args_tolerates_malformed_arguments():
    # A malformed call is still a call; the detectors want to see it, so this
    # must not raise.
    assert sb.canonical_args("{not json") == "{not json"
    assert sb.canonical_args(None) == "{}"
    assert sb.canonical_args("") == "{}"


def test_call_key_reads_both_shapes():
    assert sb.call_key(_call("get_bill", '{"x":1}')) == 'get_bill({"x": 1})'
    assert sb.call_key({"name": "get_bill", "arguments": {"x": 1}}) == 'get_bill({"x": 1})'


# --------------------------------------------------------------------------- #
# replay provider
# --------------------------------------------------------------------------- #


def test_replay_matches_by_name_not_by_exact_arguments():
    # The load-bearing default. Requiring identical arguments would make almost
    # every call `no_fixture`, kill the trajectory at round one, and report a
    # seed-coverage problem as a catastrophic model score.
    p = sb.ReplayProvider(_spec())
    expected = {"tool_seed": [{"name": "get_bill", "arguments": {"month": "jan"},
                               "content": "RM 42"}]}
    resp = p.respond(_call("get_bill", '{"month":"feb"}'), expected)
    assert resp.content == "RM 42"
    assert resp.error is False
    assert resp.detail == "name"


def test_replay_exact_mode_refuses_a_different_call():
    p = sb.ReplayProvider(_spec(replay={"match": "exact"}))
    expected = {"tool_seed": [{"name": "get_bill", "arguments": {"month": "jan"},
                               "content": "RM 42"}]}
    resp = p.respond(_call("get_bill", '{"month":"feb"}'), expected)
    assert resp.error is True
    assert json.loads(resp.content)["error"] == sb.ERR_NO_FIXTURE


def test_replay_exact_match_wins_over_name_match():
    p = sb.ReplayProvider(_spec())
    expected = {"tool_seed": [
        {"name": "get_bill", "arguments": {"month": "jan"}, "content": "JAN"},
        {"name": "get_bill", "arguments": {"month": "feb"}, "content": "FEB"},
    ]}
    assert p.respond(_call("get_bill", '{"month":"feb"}'), expected).content == "FEB"


def test_replay_consumes_the_seed_in_order_for_repeated_calls():
    # A conversation that queries the same tool twice must get two results.
    p = sb.ReplayProvider(_spec())
    expected = {"tool_seed": [
        {"name": "usage", "content": "first"},
        {"name": "usage", "content": "second"},
    ]}
    used: set[int] = set()
    assert p.respond(_call("usage"), expected, used).content == "first"
    assert p.respond(_call("usage"), expected, used).content == "second"
    # Seed exhausted → reuse the last rather than killing the trajectory.
    assert p.respond(_call("usage"), expected, used).content == "second"


def test_replay_never_fabricates_a_success():
    p = sb.ReplayProvider(_spec())
    resp = p.respond(_call("unknown_tool"), {"tool_seed": [{"name": "x", "content": "y"}]})
    assert resp.error is True
    assert json.loads(resp.content)["error"] == sb.ERR_UNKNOWN_FUNCTION

    missing = p.respond(_call("x"), {})
    assert missing.error is True
    assert json.loads(missing.content)["error"] == sb.ERR_NO_FIXTURE


def test_replay_accepts_a_name_keyed_dict_and_a_json_string():
    p = sb.ReplayProvider(_spec())
    assert p.respond(_call("a"), {"tool_seed": {"a": "A"}}).content == "A"
    assert p.respond(_call("a"), {"tool_seed": '[{"name":"a","content":"A"}]'}).content == "A"
    # A dict fixture reaches the model as JSON, not a Python repr.
    assert p.respond(_call("a"), {"tool_seed": {"a": {"k": 1}}}).content == '{"k": 1}'


# --------------------------------------------------------------------------- #
# api provider
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text if payload is None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _FakeClient:
    """Records the outbound request so the payload contract can be asserted."""

    def __init__(self, response=None, boom=None):
        self.response = response or _FakeResponse(payload={"content": "RM 42"})
        self.boom = boom
        self.calls: list[dict] = []

    async def request(self, method, url, **kw):
        self.calls.append({"method": method, "url": url, **kw})
        if self.boom:
            raise self.boom
        return self.response


def _api_spec(**api):
    cfg = {"url": "http://127.0.0.1:9/tool", **api}
    return sb.SandboxSpec(id="sb-2", name="tm", mode="api", config={"api": cfg})


@pytest.mark.asyncio
async def test_api_mode_posts_the_call_and_reads_the_configured_path():
    client = _FakeClient(_FakeResponse(payload={"result": {"output": "RM 42"}}))
    rt = sb.SandboxRuntime(_api_spec(response_field="result.output"), client=client)

    resp = await rt.respond(
        _call("get_bill", '{"month":"jan"}'), {"tool_calls": [{"name": "get_bill"}]},
        conversation=[{"role": "user", "content": "bill?"}],
    )
    assert resp.content == "RM 42"
    assert resp.provenance == "api"

    body = client.calls[0]["json"]
    assert body["tool_call"]["id"] == "c1"
    assert body["call"] == {"name": "get_bill", "arguments": {"month": "jan"}}
    assert body["conversation"][0]["content"] == "bill?"
    # A 3xx must not bounce a validated host onto a blocked one.
    assert client.calls[0]["follow_redirects"] is False


@pytest.mark.asyncio
async def test_api_mode_withholds_the_gold_reference_by_default():
    # `expected` holds the answer the evaluators grade against — a simulator that
    # can read it can return the reference result and inflate the score.
    client = _FakeClient()
    rt = sb.SandboxRuntime(_api_spec(), client=client)
    await rt.respond(_call("a"), {"tool_calls": [{"name": "a"}], "secret": 1})
    assert "row" not in client.calls[0]["json"]

    optin = _FakeClient()
    rt2 = sb.SandboxRuntime(_api_spec(send_expected=True), client=optin)
    await rt2.respond(_call("a"), {"tool_calls": [{"name": "a"}]})
    assert optin.calls[0]["json"]["row"] == {"tool_calls": [{"name": "a"}]}


@pytest.mark.asyncio
async def test_api_failures_become_structured_errors_not_fabricated_successes():
    for client, kind in (
        (_FakeClient(_FakeResponse(status_code=500, text="boom")), "sandbox_http_error"),
        (_FakeClient(boom=RuntimeError("connect failed")), "sandbox_unreachable"),
        (_FakeClient(_FakeResponse(payload={"wrong": 1})), "sandbox_bad_response"),
        (_FakeClient(_FakeResponse(text="<html>")), "sandbox_bad_response"),
    ):
        rt = sb.SandboxRuntime(_api_spec(), client=client)
        resp = await rt.respond(_call("a"), {})
        assert resp.error is True
        assert json.loads(resp.content)["error"] == kind


@pytest.mark.asyncio
async def test_api_blank_result_path_takes_the_whole_response():
    # `""` is a MEANINGFUL value, not "unset" — the custom_eval bug, replicated
    # here on purpose: a service answering a bare string must work.
    client = _FakeClient(_FakeResponse(text="RM 42"))
    rt = sb.SandboxRuntime(_api_spec(response_field=""), client=client)
    assert (await rt.respond(_call("a"), {})).content == "RM 42"


def test_api_config_treats_empty_string_as_a_value_not_as_unset():
    cfg = sb.api_config({"api": {"response_field": "", "auth_prefix": ""}})
    assert cfg["response_field"] == ""
    assert cfg["auth_prefix"] == ""
    # ...while None really does mean "not set".
    assert sb.api_config({"api": {"response_field": None}})["response_field"] == "content"


def test_api_auth_prefix_can_be_dropped_entirely():
    p = sb.ApiProvider(_api_spec(auth_prefix=""), None, api_key="k")
    assert p._headers()["Authorization"] == "k"
    assert sb.ApiProvider(_api_spec(), None, api_key="k")._headers()["Authorization"] == "Bearer k"


def test_api_validation_rejects_a_blocked_host_and_a_missing_url():
    with pytest.raises(sb.SandboxError):
        sb.validate_spec("api", "", {"api": {"url": ""}})
    with pytest.raises(sb.SandboxError):
        # Cloud-metadata address — the SSRF guard.
        sb.validate_spec("api", "", {"api": {"url": "http://169.254.169.254/latest/meta-data"}})
    with pytest.raises(sb.SandboxError):
        sb.validate_spec("api", "", {"api": {"url": "http://127.0.0.1:9/x", "method": "GET"}})
    sb.validate_spec("api", "", {"api": {"url": "http://127.0.0.1:9/x"}})


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cache_gives_every_cell_the_same_world_and_records_who_seeded_it():
    rt = sb.SandboxRuntime(_spec())
    expected = {"tool_seed": [{"name": "a", "content": "A"}]}
    first = await rt.respond(_call("a"), expected, cell="base/v1")
    second = await rt.respond(_call("a"), expected, cell="tuned/v2")
    assert first.content == second.content == "A"
    # The second cell was measured in a world the first one built — that is a
    # different claim from having resolved the call itself.
    assert first.provenance == "seed"
    assert second.provenance == "cache"
    assert second.detail == "base/v1"


@pytest.mark.asyncio
async def test_errors_are_cached_too():
    # Otherwise the same bad call fails for one variant and (on a retry) succeeds
    # for another, and the comparison measures luck.
    rt = sb.SandboxRuntime(_spec())
    a = await rt.respond(_call("nope"), {"tool_seed": [{"name": "x", "content": "y"}]})
    b = await rt.respond(_call("nope"), {"tool_seed": [{"name": "x", "content": "y"}]})
    assert a.error and b.error
    assert a.content == b.content


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_loop_config_clamps_rather_than_raising():
    # This runs inside the runner, where a config that outlived a validation
    # change must degrade, not kill a run.
    cfg = sb.loop_config({"loop": {"max_tool_rounds": 9999, "trajectory_timeout_s": "nope"}})
    assert cfg["max_tool_rounds"] == sb.MAX_TOOL_ROUNDS_CAP
    assert cfg["trajectory_timeout_s"] == sb.DEFAULT_TRAJECTORY_TIMEOUT_S


def test_validate_spec_rejects_unimplemented_modes_and_absurd_round_counts():
    with pytest.raises(sb.SandboxError):
        sb.validate_spec("api", "", {})
    with pytest.raises(sb.SandboxError):
        sb.validate_spec("python", "", {}, allow_python=False)
    with pytest.raises(sb.SandboxError):
        sb.validate_spec("replay", "", {"loop": {"max_tool_rounds": 500}})
    sb.validate_spec("replay", "", {})  # the happy path must not raise


# --------------------------------------------------------------------------- #
# Trajectory + per-turn scoring
# --------------------------------------------------------------------------- #


def _traj(*turns, **kw):
    return ev.Trajectory(turns=list(turns), **kw)


def _turn(content="", tool_calls=None, expected=None):
    c = ev.Completion(content=content)
    c.expected = dict(expected or {})
    c.expected["_tool_calls"] = tool_calls or []
    c.expected.setdefault("_tools", [])
    return c


def test_turn_expected_narrows_reference_data_after_the_first_turn():
    row = {"tool_calls": [{"name": "a"}], "language": "malay", "note": "keep me"}
    first = ev.turn_expected(row, 0, tool_calls=[], tools=[])
    assert first["tool_calls"] == [{"name": "a"}]

    later = ev.turn_expected(row, 1, tool_calls=[], tools=[])
    # The row's reference describes the turn the row is ABOUT. Re-using it for
    # round 2 would invent failures, so a later turn abstains.
    assert "tool_calls" not in later
    assert "language" not in later
    assert later["note"] == "keep me"


def test_turn_expected_uses_a_per_turn_reference_when_the_row_carries_one():
    row = {"tool_calls": [{"name": "a"}],
           "turns": [{"tool_calls": [{"name": "a"}]}, {"tool_calls": [{"name": "b"}]}]}
    assert ev.turn_expected(row, 1, tool_calls=[], tools=[])["tool_calls"] == [{"name": "b"}]


def test_as_completion_projects_the_final_turn_with_trajectory_totals():
    traj = _traj(
        _turn("", tool_calls=[_call("a")]),
        _turn("done"),
        latency_ms=900, ttft_ms=40, usage={"completion_tokens": 7},
    )
    comp = traj.as_completion()
    assert comp.content == "done"
    assert comp.latency_ms == 900 and comp.ttft_ms == 40
    assert comp.completion_tokens == 7


def test_final_turn_projection_would_zero_tool_call_f1():
    """The bug the three-valued `wants` exists to prevent.

    The final turn of a trajectory is the text answer, so it carries no tool
    calls. Scored there, `function_call_units` compares an empty call list
    against a non-empty reference on EVERY row — F1 collapses to 0 and it reads
    exactly like a model regression. Per-turn scoring recovers the real number.
    """
    ref = [{"name": "get_bill", "arguments": {}}]
    traj = _traj(
        _turn("", tool_calls=[_call("get_bill")], expected={"tool_calls": ref}),
        _turn("Your bill is RM 42."),
        expected={"tool_calls": ref},
    )
    sel = [{"id": "function_call_units", "options": {}}]

    # What the naive adapter would have done:
    naive, _ = ev.run_evaluators(traj.as_completion(), sel)
    naive_f1 = ev.SPECS["function_call_units"].aggregate(
        [o.flags for o in naive if o.id == "function_call_units"]
    )
    assert naive_f1["tool_call_f1"] == 0.0

    # What the per-turn fold actually does.
    outcomes, _ = ev.run_evaluators_trajectory(traj, sel)
    fcu = next(o for o in outcomes if o.id == "function_call_units")
    pooled = ev.SPECS["function_call_units"].aggregate(fcu.flags["turn_flags"])
    assert pooled["tool_call_f1"] == 1.0
    # Turn 2 had no reference of its own, so it abstained rather than counting.
    assert pooled["scored"] == 1


def test_leak_in_middle_turn_fails_the_sample():
    # Recovery in a later round does not undo the leak.
    traj = _traj(
        _turn("<|channel|>thought", tool_calls=[_call("a")]),
        _turn("all good"),
    )
    outcomes, passed = ev.run_evaluators_trajectory(
        traj, [{"id": "control_token_leak", "options": {}}]
    )
    leak = next(o for o in outcomes if o.id == "control_token_leak")
    assert leak.passed is False
    assert leak.flags["failed_turn"] == 1
    assert passed is False


def test_a_tool_call_turn_never_trips_the_empty_response_detector():
    # `empty_response` stays wants="completion" precisely because an assistant
    # turn that only calls tools has empty content BY DESIGN.
    assert ev.SPECS["empty_response"].wants == "completion"
    traj = _traj(_turn("", tool_calls=[_call("a")]), _turn("here you go"))
    outcomes, passed = ev.run_evaluators_trajectory(
        traj, [{"id": "empty_response", "options": {}}]
    )
    assert next(o for o in outcomes if o.id == "empty_response").passed is True
    assert passed is True


def test_a_trajectory_with_no_reference_data_scores_nothing():
    # The standing guard, one level up: a green pass rate over a corpus carrying
    # no `expected` block must still report `scored: 0`. Abstention here is
    # IMPLICIT (no reference and no model call), which is why the fold publishes
    # no scored-count of its own — the aggregate is authoritative.
    traj = _traj(_turn("hi"), _turn("bye"))
    outcomes, passed = ev.run_evaluators_trajectory(
        traj, [{"id": "function_call_units", "options": {}}]
    )
    fcu = next(o for o in outcomes if o.id == "function_call_units")
    assert passed is True
    pooled = ev.SPECS["function_call_units"].aggregate(fcu.flags["turn_flags"])
    assert pooled["scored"] == 0
    assert pooled["turns"] == 2


def test_an_aborted_trajectory_is_a_failure_not_partial_credit():
    traj = _traj(_turn("partial"), error="ReadTimeout", error_round=2)
    outcomes, passed = ev.run_evaluators_trajectory(traj, [])
    assert passed is False
    assert any(o.id == "request_error" for o in outcomes)


# --------------------------------------------------------------------------- #
# Storage + rollup
# --------------------------------------------------------------------------- #


def test_trajectory_payload_truncates_tool_results_before_model_turns():
    # The model's own output is the thing under test; a fixture's payload is not.
    big = "x" * (2 * 32000)
    traj = ev.Trajectory(messages=[
        {"role": "assistant", "content": "short answer"},
        {"role": "tool", "tool_call_id": "c1", "content": big},
    ])
    payload = _trajectory_payload(traj)
    assert payload["messages"][0]["content"] == "short answer"
    assert payload["messages"][1]["truncated"] is True
    assert len(payload["messages"][1]["content"]) < len(big)


def test_sandbox_rollup_flags_a_run_that_never_really_ran():
    payloads = [
        {"rounds": 1, "tool_calls_total": 2, "novel_calls": 2, "forced_final": False,
         "provenance": {"error": 2}},
        {"rounds": 1, "tool_calls_total": 2, "novel_calls": 2, "forced_final": True,
         "provenance": {"error": 2}},
    ]
    roll = _sandbox_rollup(payloads)
    assert roll["all_errors"] is True          # every tool response failed
    assert roll["novel_call_rate"] == 1.0      # the seed covered nothing
    assert roll["forced_final_rate"] == 0.5    # half hit the round limit


def test_summarize_pools_turn_flags_into_the_corpus_metric():
    # An F1 pools tp/fp/fn across every scored TURN, not across folded samples.
    sample = {
        "target": "t", "variant": "v", "passed": True,
        "evals": {"function_call_units": {"passed": True, "score": 1.0, "flags": {
            "turn_flags": [
                {"n_ref": 1, "n_model": 1, "tp": 1, "fp": 0, "fn": 0},
                {"n_ref": 1, "n_model": 1, "tp": 1, "fp": 0, "fn": 0},
            ],
        }}},
    }
    out = summarize([sample], ["function_call_units"])
    metrics = out["cells"][0]["evals"]["function_call_units"]["metrics"]
    assert metrics["scored"] == 2   # two TURNS, from one sample


def test_summarize_puts_the_sandbox_rollup_on_the_cell():
    sample = {
        "target": "t", "variant": "v", "passed": True, "evals": {},
        "trajectory": {"rounds": 2, "tool_calls_total": 2, "novel_calls": 0,
                       "forced_final": False, "provenance": {"seed": 2}},
    }
    cell = summarize([sample], [])["cells"][0]
    assert cell["sandbox"]["trajectories"] == 1
    assert cell["sandbox"]["rounds_mean"] == 2.0
    assert cell["sandbox"]["all_errors"] is False


def test_summarize_without_a_sandbox_is_unchanged():
    cell = summarize([{"target": "t", "variant": "v", "passed": True, "evals": {}}], [])["cells"][0]
    assert "sandbox" not in cell


# --------------------------------------------------------------------------- #
# Runner helpers
# --------------------------------------------------------------------------- #


def test_normalize_calls_always_produces_an_id():
    # A `role=tool` message without a matching tool_call_id is rejected by the
    # very server we're about to send it back to.
    out = _normalize_calls([{"function": {"name": "a", "arguments": "{}"}}])
    assert out[0]["id"]
    assert out[0]["type"] == "function"


def test_sum_usage_accumulates_across_turns():
    total = _sum_usage(None, {"prompt_tokens": 10, "completion_tokens": 2})
    total = _sum_usage(total, {"prompt_tokens": 14, "completion_tokens": 3})
    assert total == {"prompt_tokens": 24, "completion_tokens": 5}
    assert _sum_usage(total, None) == total


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


TARGET = {"label": "t", "base_url": "http://x/v1", "model": "m"}
VARIANT = {"label": "baseline"}


def _case(expected=None, tools=None):
    return Case(
        id="ds-1:0", name="row", params={},
        messages=[{"role": "user", "content": "what's my bill?"}],
        tools=tools or [{"type": "function", "function": {"name": "get_bill"}}],
        expected=expected or {"tool_seed": [{"name": "get_bill", "content": "RM 42"}]},
    )


def _scripted(*replies):
    """Stand in for `call_once`, returning one canned reply per turn and
    recording the request body it was handed."""
    seen: list[dict] = []
    queue = list(replies)

    async def fake_call_once(client, base_url, path, key, body, timeout):
        seen.append(json.loads(json.dumps(body)))   # snapshot: the loop mutates it
        return queue.pop(0) if queue else ev.Completion(content="fallback")

    fake_call_once.seen = seen
    return fake_call_once


def _reply(content="", calls=None, **kw):
    return ev.Completion(
        content=content, expected={"_tool_calls": calls or []}, **kw,
    )


@pytest.mark.asyncio
async def test_a_tool_call_is_answered_and_the_conversation_continues(monkeypatch):
    import gateway.experiments_api as api

    fake = _scripted(
        _reply(calls=[_call("get_bill", '{"month":"jan"}', cid="abc")], ttft_ms=30),
        _reply(content="Your bill is RM 42.", usage={"completion_tokens": 5}),
    )
    monkeypatch.setattr(api, "call_once", fake)
    rt = sb.SandboxRuntime(_spec())

    traj = await api.run_trajectory(
        None, _case(), TARGET, VARIANT, False, "", 30.0, rt, cell="t/baseline",
    )

    assert traj.rounds == 1
    assert len(traj.turns) == 2
    assert traj.provenance == ["seed"]
    assert traj.error is None
    assert traj.ttft_ms == 30            # the FIRST turn's, not the last

    # The second request carried the assistant turn and its tool result, and the
    # tool message's id matches the call — without which the server rejects it.
    second = fake.seen[1]["messages"]
    assert second[-2]["tool_calls"][0]["id"] == "abc"
    assert second[-1] == {"role": "tool", "tool_call_id": "abc",
                          "name": "get_bill", "content": "RM 42"}


@pytest.mark.asyncio
async def test_the_round_limit_forces_one_toolless_final_answer(monkeypatch):
    import gateway.experiments_api as api

    # A model that would loop forever.
    fake = _scripted(
        _reply(calls=[_call("get_bill", cid="a")]),
        _reply(calls=[_call("get_bill", cid="b")]),
        _reply(content="RM 42."),
    )
    monkeypatch.setattr(api, "call_once", fake)
    rt = sb.SandboxRuntime(_spec(loop={"max_tool_rounds": 1}))

    traj = await api.run_trajectory(None, _case(), TARGET, VARIANT, False, "", 30.0, rt)

    assert traj.forced_final is True
    assert traj.turns[-1].content == "RM 42."
    # The final ask had the tools removed, which is what makes it answer in text.
    assert "tools" in fake.seen[0] and "tools" not in fake.seen[-1]


@pytest.mark.asyncio
async def test_the_loop_terminates_even_when_the_model_never_stops(monkeypatch):
    import gateway.experiments_api as api

    always_calls = _scripted(*[_reply(calls=[_call("get_bill", cid=str(i))])
                               for i in range(12)])
    monkeypatch.setattr(api, "call_once", always_calls)
    rt = sb.SandboxRuntime(_spec(loop={"max_tool_rounds": 2}))

    traj = await api.run_trajectory(None, _case(), TARGET, VARIANT, False, "", 30.0, rt)
    assert traj.rounds <= 4
    assert len(always_calls.seen) <= 5


@pytest.mark.asyncio
async def test_a_transport_failure_aborts_the_trajectory(monkeypatch):
    import gateway.experiments_api as api

    fake = _scripted(
        _reply(calls=[_call("get_bill", cid="a")]),
        _reply(error="ReadTimeout: too slow"),
    )
    monkeypatch.setattr(api, "call_once", fake)
    rt = sb.SandboxRuntime(_spec())

    traj = await api.run_trajectory(None, _case(), TARGET, VARIANT, False, "", 30.0, rt)
    assert traj.error == "ReadTimeout: too slow"
    assert traj.error_round == 2
    # And an aborted trajectory is a failure, never partial credit.
    _, passed = ev.run_evaluators_trajectory(traj, [])
    assert passed is False


@pytest.mark.asyncio
async def test_a_tool_failure_does_NOT_abort_the_trajectory(monkeypatch):
    # The model is handed a structured error and gets to react to it — that's
    # realistic behaviour and worth scoring, unlike a dead endpoint.
    import gateway.experiments_api as api

    fake = _scripted(
        _reply(calls=[_call("not_seeded", cid="a")]),
        _reply(content="Sorry, I couldn't look that up."),
    )
    monkeypatch.setattr(api, "call_once", fake)
    rt = sb.SandboxRuntime(_spec())

    traj = await api.run_trajectory(None, _case(), TARGET, VARIANT, False, "", 30.0, rt)
    assert traj.error is None
    assert traj.provenance == ["error"]
    assert traj.novel_calls == 1
    assert json.loads(fake.seen[1]["messages"][-1]["content"])["error"] == "unknown_function"


@pytest.mark.asyncio
async def test_each_turn_is_scored_against_its_own_reference(monkeypatch):
    import gateway.experiments_api as api

    fake = _scripted(
        _reply(calls=[_call("get_bill", cid="a")]),
        _reply(content="RM 42."),
    )
    monkeypatch.setattr(api, "call_once", fake)
    rt = sb.SandboxRuntime(_spec())
    case = _case(expected={
        "tool_seed": [{"name": "get_bill", "content": "RM 42"}],
        "tool_calls": [{"name": "get_bill", "arguments": {}}],
    })

    traj = await api.run_trajectory(None, case, TARGET, VARIANT, False, "", 30.0, rt)
    # Turn 1 carries the row's reference; turn 2 has none of its own, so it
    # abstains instead of being compared against turn 1's.
    assert traj.turns[0].expected["tool_calls"] == [{"name": "get_bill", "arguments": {}}]
    assert "tool_calls" not in traj.turns[1].expected
    assert traj.turns[0].expected["_tools"] == case.tools
