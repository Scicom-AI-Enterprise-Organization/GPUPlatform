"""Pure logic of the Experiments feature: request building, aggregation, and the
Langfuse import guards. No DB, no network.
"""
import json

import pytest

from gateway import langfuse_import as lf
from gateway.experiments_api import (
    Case,
    _coerce_json_cell,
    _merge_stream_tool_calls,
    _public_config,
    _row_to_case,
    build_request,
    dataset_usable,
    summarize,
)


def _case(**kw) -> Case:
    defaults = dict(
        id="ds-1:0", name="c",
        messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}],
        tools=None, params={}, expected={},
    )
    defaults.update(kw)
    return Case(**defaults)


TARGET = {"label": "t", "base_url": "http://x", "model": "m", "extra_body": {}}


# ------------------------------------------------------------- request building


def test_build_request_replays_recorded_params():
    """The captured sampling params are the baseline — replaying with library
    defaults reproduces a different request than the one that misbehaved."""
    case = _case(params={"temperature": 0.7, "top_k": 64, "enable_thinking": False})
    body = build_request(case, TARGET, {"label": "baseline"}, stream=False)
    assert body["temperature"] == 0.7
    assert body["top_k"] == 64                                  # goes to extra_body keys
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_variant_params_override_case_params():
    case = _case(params={"temperature": 0.7})
    body = build_request(case, TARGET, {"params": {"temperature": 0.0}}, stream=False)
    assert body["temperature"] == 0.0


def test_stream_always_requests_usage():
    """Without stream_options a streamed reply reports no usage at all, and every
    token-derived metric silently reads zero."""
    body = build_request(_case(), TARGET, {}, stream=True)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_system_suffix_appends_to_existing_system_turn():
    body = build_request(_case(), TARGET, {"system_suffix": "\nBE TERSE"}, stream=False)
    assert body["messages"][0]["content"] == "SYS\nBE TERSE"
    assert len(body["messages"]) == 2


def test_system_suffix_inserts_when_no_system_turn():
    case = _case(messages=[{"role": "user", "content": "hi"}])
    body = build_request(case, TARGET, {"system_prefix": "RULES"}, stream=False)
    assert body["messages"][0] == {"role": "system", "content": "RULES"}


def test_user_suffix_targets_the_last_user_turn():
    case = _case(messages=[
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ])
    body = build_request(case, TARGET, {"user_suffix": " NOW"}, stream=False)
    assert body["messages"][2]["content"] == "second NOW"
    assert body["messages"][0]["content"] == "first"


def test_assistant_prefill_appends_a_trailing_assistant_turn():
    body = build_request(_case(), TARGET, {"assistant_prefill": '{"intent":'}, stream=False)
    assert body["messages"][-1] == {"role": "assistant", "content": '{"intent":'}


def test_strip_tools_removes_declarations():
    case = _case(tools=[{"type": "function", "function": {"name": "f"}}])
    assert "tools" in build_request(case, TARGET, {}, stream=False)
    assert "tools" not in build_request(case, TARGET, {"strip_tools": True}, stream=False)


def test_response_format_shorthand_and_object():
    b1 = build_request(_case(), TARGET, {"response_format": "json_object"}, stream=False)
    assert b1["response_format"] == {"type": "json_object"}
    schema = {"type": "json_schema", "json_schema": {"name": "s"}}
    b2 = build_request(_case(), TARGET, {"response_format": schema}, stream=False)
    assert b2["response_format"] == schema


def test_variant_extra_body_wins_over_target_extra_body():
    target = {**TARGET, "extra_body": {"guided_decoding_backend": "outlines"}}
    body = build_request(
        _case(), target, {"extra_body": {"guided_decoding_backend": "xgrammar"}}, stream=False
    )
    assert body["guided_decoding_backend"] == "xgrammar"


def test_build_request_does_not_mutate_the_case():
    case = _case()
    build_request(case, TARGET, {"system_suffix": "X", "assistant_prefill": "Y"}, stream=False)
    assert case.messages[0]["content"] == "SYS"
    assert len(case.messages) == 2


# ------------------------------------------------------------- stream tool calls


def test_merge_stream_tool_calls_reassembles_arguments():
    """Streamed arguments arrive char-by-char; a naive reader sees only the last
    fragment and every tool call looks malformed."""
    frags = [
        {"index": 0, "id": "call_1", "function": {"name": "get_bill", "arguments": '{"a'}},
        {"index": 0, "function": {"arguments": '": 1}'}},
    ]
    merged = _merge_stream_tool_calls(frags)
    assert len(merged) == 1
    assert json.loads(merged[0]["function"]["arguments"]) == {"a": 1}
    assert merged[0]["function"]["name"] == "get_bill"


def test_merge_stream_tool_calls_handles_parallel_calls():
    frags = [
        {"index": 0, "function": {"name": "a", "arguments": "{}"}},
        {"index": 1, "function": {"name": "b", "arguments": "{}"}},
    ]
    assert [c["function"]["name"] for c in _merge_stream_tool_calls(frags)] == ["a", "b"]


# ------------------------------------------------------------- summarize


def _s(target, variant, passed=True, error=None, latency=100, evals=None):
    return {
        "target": target, "variant": variant, "passed": passed, "error_text": error,
        "latency_ms": latency, "ttft_ms": 10, "prompt_tokens": 5, "completion_tokens": 7,
        "evals": evals or {},
    }


def test_summarize_builds_one_cell_per_target_variant():
    samples = [
        _s("a", "base"), _s("a", "base"), _s("a", "tuned"), _s("b", "base"),
    ]
    out = summarize(samples, ["request_error"])
    keys = {(c["target"], c["variant"]) for c in out["cells"]}
    assert keys == {("a", "base"), ("a", "tuned"), ("b", "base")}
    assert out["totals"]["n"] == 4


def test_summarize_reports_per_evaluator_fail_rate():
    evals_bad = {"json_output": {"passed": False, "score": 0.0, "flags": {}}}
    evals_ok = {"json_output": {"passed": True, "score": 1.0, "flags": {}}}
    samples = [
        _s("a", "base", passed=False, evals=evals_bad),
        _s("a", "base", passed=True, evals=evals_ok),
        _s("a", "base", passed=True, evals=evals_ok),
        _s("a", "base", passed=True, evals=evals_ok),
    ]
    cell = summarize(samples, ["json_output"])["cells"][0]
    assert cell["evals"]["json_output"]["fail_rate"] == 0.25
    assert cell["pass_rate"] == 0.75


def test_summarize_counts_errors_separately_from_failures():
    samples = [_s("a", "base", passed=False, error="HTTP 500"), _s("a", "base")]
    cell = summarize(samples, [])["cells"][0]
    assert cell["n_error"] == 1
    assert cell["error_rate"] == 0.5
    assert cell["n_ok"] == 1


def test_summarize_percentiles_and_cost():
    evals = {"cost": {"passed": True, "score": 0.01, "flags": {"cost_usd": 0.01}}}
    samples = [_s("a", "b", latency=lat, evals=evals) for lat in (10, 20, 30, 40, 1000)]
    cell = summarize(samples, ["cost"])["cells"][0]
    assert cell["latency_ms"]["p50"] == 30
    assert cell["latency_ms"]["p95"] == 1000
    assert cell["cost_usd_total"] == pytest.approx(0.05)


def test_summarize_empty_is_safe():
    out = summarize([], [])
    assert out["cells"] == []
    assert out["totals"]["n"] == 0


# ------------------------------------------------------------- config redaction


def test_public_config_strips_encrypted_keys_without_mutating_source():
    cfg = {
        "targets": [{"label": "t", "api_key_enc": "gAAAA…", "base_url": "http://x"}],
        "evaluators": [{"id": "llm_judge", "options": {"api_key": "sk-secret", "model": "m"}}],
    }
    out = _public_config(cfg)
    assert "api_key_enc" not in out["targets"][0]
    assert out["targets"][0]["has_inline_key"] is True
    assert "api_key" not in out["evaluators"][0]["options"]
    # The ORM row's dict must be untouched, or the redaction is flushed back.
    assert cfg["targets"][0]["api_key_enc"] == "gAAAA…"
    assert cfg["evaluators"][0]["options"]["api_key"] == "sk-secret"


# ------------------------------------------------------------- langfuse import


def test_parse_url_prefers_traceid_over_peek():
    """In the newer UI `peek=` can be a span id that 404s; traceId is the trace."""
    url = "https://lf.example/project/p1/traces?peek=SPAN&traceId=TRACE&observation=OBS"
    out = lf.parse_langfuse_url(url)
    assert out["trace_id"] == "TRACE"
    assert out["observation_id"] == "OBS"
    assert out["used_trace_id_param"] is True


def test_parse_url_falls_back_to_peek():
    out = lf.parse_langfuse_url("https://lf.example/project/p1/traces?peek=ABC")
    assert out["trace_id"] == "ABC"
    assert out["used_trace_id_param"] is False


def test_parse_url_accepts_bare_id_and_permalink():
    assert lf.parse_langfuse_url("deadbeef")["trace_id"] == "deadbeef"
    assert lf.parse_langfuse_url(
        "https://lf.example/project/p/traces/xyz"
    )["trace_id"] == "xyz"


def test_repair_quotes_only_structural_placeholders():
    """Quoting a placeholder inside customer text would corrupt the message body."""
    raw = '{"created_at": <id>, "content": "Email address: <email>\\n"}'
    fixed = json.loads(lf._repair_scrubbed_json(raw))
    assert fixed["created_at"] == "<id>"
    assert fixed["content"] == "Email address: <email>\n"


def test_extract_refuses_character_iterated_corruption():
    """The signature failure: an unparseable JSON-string input yields thousands of
    single-character 'messages' with no error raised."""
    trace = {
        "id": "t1",
        "observations": [{
            "id": "o1", "type": "GENERATION",
            "input": [{"role": "user", "content": "x"}] * 900,
        }],
    }
    with pytest.raises(lf.LangfuseError, match="signature of a PII-scrubbed"):
        lf.extract_request(trace, "o1")


def test_extract_rejects_non_object_messages():
    trace = {"id": "t", "observations": [
        {"id": "o1", "type": "GENERATION", "input": ["a", "b", "c"]}
    ]}
    with pytest.raises(lf.LangfuseError, match="non-object messages"):
        lf.extract_request(trace, "o1")


def test_extract_points_at_child_generations_for_an_agent_span():
    trace = {"id": "t", "observations": [
        {"id": "agent", "type": "AGENT", "name": "x.arun", "input": "just a task string"},
        {"id": "gen", "type": "GENERATION", "parentObservationId": "agent",
         "input": {"messages": [{"role": "user", "content": "hi"}]}},
    ]}
    with pytest.raises(lf.LangfuseError, match="gen"):
        lf.extract_request(trace, "agent")


def test_extract_repairs_scrubbed_json_string_input():
    payload = '{"messages": [{"role": "user", "content": "hi"}], "created_at": <id>}'
    trace = {"id": "t", "observations": [
        {"id": "o1", "type": "GENERATION", "input": payload}
    ]}
    out = lf.extract_request(trace, "o1")
    assert out["messages"] == [{"role": "user", "content": "hi"}]


def test_extract_recovers_model_parameters_including_extra_body():
    trace = {"id": "t", "observations": [{
        "id": "o1", "type": "GENERATION", "model": "gemma",
        "input": {"messages": [{"role": "user", "content": "hi"}]},
        "modelParameters": {
            "temperature": 0.7, "top_p": 0.95,
            "extra_body": '{"top_k": 64, "chat_template_kwargs": {"enable_thinking": false}}',
        },
    }]}
    out = lf.extract_request(trace, "o1")
    assert out["params"] == {
        "temperature": 0.7, "top_p": 0.95, "top_k": 64, "enable_thinking": False
    }
    assert out["model"] == "gemma"


def test_extract_picks_first_replayable_generation_when_unspecified():
    trace = {"id": "t", "observations": [
        {"id": "span", "type": "SPAN", "input": {"foo": 1}},
        {"id": "gen", "type": "GENERATION",
         "input": {"messages": [{"role": "user", "content": "hi"}], "tools": [{"x": 1}]}},
    ]}
    out = lf.extract_request(trace)
    assert out["observation_id"] == "gen"
    assert out["tools"] == [{"x": 1}]


def test_clean_messages_preserves_tool_fields():
    msgs = lf._clean_messages([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}], "extra": "drop"},
        {"role": "tool", "content": "res", "tool_call_id": "1", "name": "f"},
    ])
    assert msgs[0]["tool_calls"] == [{"id": "1"}]
    assert "extra" not in msgs[0]
    assert msgs[1]["tool_call_id"] == "1" and msgs[1]["name"] == "f"


def test_list_generations_marks_corrupt_ones_unreplayable():
    trace = {"observations": [
        {"id": "ok", "type": "GENERATION",
         "input": {"messages": [{"role": "user", "content": "hi"}]}},
        {"id": "bad", "type": "GENERATION",
         "input": [{"role": "user"}] * 900},
    ]}
    gens = {g["id"]: g for g in lf.list_generations(trace)}
    assert gens["ok"]["replayable"] is True
    assert gens["bad"]["replayable"] is False


# ------------------------------------------------- cases from a platform Dataset


class _Ds:
    """Enough of a Dataset row for dataset_usable()."""
    def __init__(self, **kw):
        self.kind = kw.get("kind", "upload")
        self.messages_field = kw.get("messages_field", "messages")
        self.storage_id = kw.get("storage_id", "st-1")
        self.hf_repo = kw.get("hf_repo", None)
        self.id = "ds-1"


def test_dataset_usable_accepts_a_chat_upload():
    ok, why = dataset_usable(_Ds())
    assert ok is True and why is None


def test_dataset_usable_rejects_wrong_kind():
    ok, why = dataset_usable(_Ds(kind="tts_packed"))
    assert ok is False and "kind=tts_packed" in why


def test_dataset_usable_rejects_missing_messages_column():
    """An audio/plain dataset has rows but nothing replayable in them."""
    ok, why = dataset_usable(_Ds(messages_field=None))
    assert ok is False and "messages column" in why


def test_dataset_usable_rejects_unattached_sources():
    assert dataset_usable(_Ds(storage_id=None))[0] is False
    assert dataset_usable(_Ds(kind="llm", hf_repo=None))[0] is False
    assert dataset_usable(_Ds(kind="llm", hf_repo="a/b", storage_id=None))[0] is True


def test_coerce_json_cell_handles_string_round_trips():
    """A JSONL round-trip may hand back a dict or its JSON string."""
    assert _coerce_json_cell({"a": 1}) == {"a": 1}
    assert _coerce_json_cell('{"a": 1}') == {"a": 1}
    assert _coerce_json_cell("[1,2]") == [1, 2]
    assert _coerce_json_cell("not json") is None
    assert _coerce_json_cell("") is None
    assert _coerce_json_cell(None) is None


def test_row_to_case_basic():
    row = {"messages": [{"role": "user", "content": "hi"}], "name": "greeting"}
    case = _row_to_case(row, 0, "messages", "ds-9")
    assert case is not None
    assert case.id == "ds-9:0" and case.name == "greeting"
    assert case.messages == [{"role": "user", "content": "hi"}]


def test_row_to_case_reads_messages_as_a_json_string():
    row = {"messages": '[{"role": "user", "content": "hi"}]'}
    case = _row_to_case(row, 3, "messages", "ds-9")
    assert case is not None and case.messages[0]["content"] == "hi"
    assert case.name == "row-4"      # 1-based fallback name


def test_row_to_case_honours_a_custom_messages_column():
    row = {"conversation": [{"role": "user", "content": "hi"}]}
    assert _row_to_case(row, 0, "conversation", "ds-9") is not None
    assert _row_to_case(row, 0, "messages", "ds-9") is None


def test_row_to_case_picks_up_tools_from_either_column():
    tools = [{"type": "function", "function": {"name": "f"}}]
    base = {"messages": [{"role": "user", "content": "hi"}]}
    assert _row_to_case({**base, "tools": tools}, 0, "messages", "d").tools == tools
    assert _row_to_case({**base, "functions": tools}, 0, "messages", "d").tools == tools
    # A JSON-string cell (what a CSV export gives) parses too.
    assert _row_to_case(
        {**base, "tools": json.dumps(tools)}, 0, "messages", "d"
    ).tools == tools
    assert _row_to_case(base, 0, "messages", "d").tools is None


def test_row_to_case_reads_params_from_a_blob_or_flat_columns():
    base = {"messages": [{"role": "user", "content": "hi"}]}
    blob = _row_to_case({**base, "params": {"temperature": 0.7}}, 0, "messages", "d")
    assert blob.params == {"temperature": 0.7}
    flat = _row_to_case({**base, "temperature": 0.7, "top_k": 64}, 0, "messages", "d")
    assert flat.params == {"temperature": 0.7, "top_k": 64}


def test_row_to_case_reads_expected_for_evaluators():
    row = {"messages": [{"role": "user", "content": "hi"}],
           "expected": {"json_keys": ["intent"]}}
    assert _row_to_case(row, 0, "messages", "d").expected == {"json_keys": ["intent"]}


def test_row_to_case_skips_rows_with_no_request():
    for bad in ({}, {"messages": None}, {"messages": []}, {"messages": "nope"},
                {"messages": ["a", "b"]}):
        assert _row_to_case(bad, 0, "messages", "d") is None


def test_row_to_case_keeps_tool_message_fields():
    row = {"messages": [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}], "junk": "x"},
        {"role": "tool", "content": "res", "tool_call_id": "1", "name": "f"},
    ]}
    case = _row_to_case(row, 0, "messages", "d")
    assert case.messages[0]["tool_calls"] == [{"id": "1"}]
    assert "junk" not in case.messages[0]
    assert case.messages[1]["tool_call_id"] == "1"


# --------------------------------------------------------------------------- #
# Target URL joining — the /v1/v1 404
# --------------------------------------------------------------------------- #


def test_default_path_normalizes_a_base_that_already_has_v1():
    """A platform proxy base is `…/proxy/x/v1`; joined naively with the default
    path it became `…/v1/v1/chat/completions` and 404'd every replay."""
    from gateway.experiments_api import DEFAULT_CHAT_PATH, _join_chat_url

    assert _join_chat_url("https://gw/proxy/x/v1", DEFAULT_CHAT_PATH) == \
        "https://gw/proxy/x/v1/chat/completions"
    assert _join_chat_url("http://vllm:8000", DEFAULT_CHAT_PATH) == \
        "http://vllm:8000/v1/chat/completions"
    assert _join_chat_url("https://gw/v1/chat/completions", DEFAULT_CHAT_PATH) == \
        "https://gw/v1/chat/completions"


def test_a_custom_path_is_joined_verbatim():
    """Setting a non-default path is deliberate — don't second-guess it."""
    from gateway.experiments_api import _join_chat_url

    assert _join_chat_url("https://gw/api", "/v2/generate") == "https://gw/api/v2/generate"
    # An empty path means the caller already resolved the full URL.
    assert _join_chat_url("https://gw/v1/chat/completions", "") == "https://gw/v1/chat/completions"
