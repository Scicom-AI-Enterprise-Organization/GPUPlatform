"""evaluators — the detector library behind the Experiments feature.

Each test pins a behaviour that a hand-written stress script got wrong at least
once, so a regression here is a regression in a real finding.
"""
import pytest

from gateway import evaluators as ev


def _c(**kw) -> ev.Completion:
    return ev.Completion(**kw)


# ---------------------------------------------------------------- control tokens


@pytest.mark.parametrize("text", [
    "<|channel|>final Hello",
    "<|channel>thought\n<channel|>No worries!",   # the malformed pair seen in prod
    "answer <|start|> leaked",
    "text <end_of_turn>",
])
def test_control_token_leak_catches_wellformed_and_malformed(text):
    out = ev._check_control_token_leak(_c(content=text), {})
    assert out.passed is False
    assert out.flags["leaked_tokens"]


def test_control_token_leak_clean_text_passes():
    out = ev._check_control_token_leak(_c(content="Your account is active."), {})
    assert out.passed is True
    assert out.flags["leaked_tokens"] == []


def test_control_token_leak_channel_only_ignores_other_markers():
    out = ev._check_control_token_leak(_c(content="a <|start|> b"), {"channel_only": True})
    assert out.passed is True          # not a channel marker
    assert out.flags["leaked_tokens"]  # still reported


def test_control_token_leak_reports_channel_role():
    out = ev._check_control_token_leak(_c(content="<|channel>thought hmm"), {})
    assert out.flags["channel_role"] == "thought"


# ---------------------------------------------------------------- empty response


def test_empty_response_distinguishes_zero_usage():
    """Empty + 0 completion tokens means no forward pass ran — a different fault
    from a model that generated nothing."""
    out = ev._check_empty_response(
        _c(content="", usage={"prompt_tokens": 0, "completion_tokens": 0}), {}
    )
    assert out.passed is False
    assert out.flags["empty_zero_usage"] is True


def test_empty_response_flags_reasoning_only():
    out = ev._check_empty_response(_c(content="  ", reasoning="thinking hard"), {})
    assert out.passed is False
    assert out.flags["reasoning_only"] is True


def test_empty_response_reasoning_only_can_be_allowed():
    out = ev._check_empty_response(
        _c(content="", reasoning="thinking"), {"allow_reasoning_only": True}
    )
    assert out.passed is True


def test_empty_response_min_chars():
    out = ev._check_empty_response(_c(content="hi"), {"min_chars": 10})
    assert out.passed is False
    out2 = ev._check_empty_response(_c(content="hi there friend"), {"min_chars": 10})
    assert out2.passed is True


# ---------------------------------------------------------------- degeneration


def test_degeneration_catches_repetition_loop():
    out = ev._check_degeneration(_c(content="spam " * 200), {})
    assert out.passed is False
    assert out.flags["content_degenerate"] is True


def test_degeneration_ignores_short_replies():
    """A 3-word reply trivially trips distinct-ratio; below min_tokens we must
    never flag, or every short answer reads as degenerate."""
    out = ev._check_degeneration(_c(content="yes yes yes"), {})
    assert out.passed is True


def test_degeneration_passes_normal_prose():
    prose = (
        "Your account is currently active and there are no bars applied. "
        "The most recent invoice was settled on the third of the month, so "
        "nothing further is outstanding at this time. Let me know if you would "
        "like a copy of the statement sent to your registered email address."
    ) * 3
    out = ev._check_degeneration(_c(content=prose), {})
    assert out.passed is True


def test_degeneration_detects_newline_runaway():
    out = ev._check_degeneration(_c(content="done." + "\n" * 60, finish_reason="length"), {})
    assert out.passed is False
    assert out.flags["newline_loop"] is True


# ---------------------------------------------------------------- json output


def test_json_output_accepts_bare_object():
    out = ev._check_json_output(_c(content='{"intent": "billing"}'), {})
    assert out.passed is True
    assert out.flags["strict_ok"] is True


def test_json_output_fenced_fails_strict_by_default():
    """A ```json fence is the exact failure that broke an agent 100/100 times
    while looking perfectly fine to a human."""
    out = ev._check_json_output(_c(content='```json\n{"a": 1}\n```'), {})
    assert out.passed is False
    assert out.flags["fenced"] is True
    assert out.flags["repaired_ok"] is True   # recoverable, but still a failure


def test_json_output_accept_repaired_passes_fenced():
    out = ev._check_json_output(
        _c(content='```json\n{"a": 1}\n```'), {"accept_repaired": True}
    )
    assert out.passed is True


def test_json_output_trailing_comma_reported():
    out = ev._check_json_output(_c(content='{"a": 1,}'), {})
    assert out.passed is False
    assert out.flags["trailing_comma"] is True


def test_json_output_missing_expected_keys():
    out = ev._check_json_output(
        _c(content='{"intent": "billing"}'), {"expected_keys": ["intent", "confidence"]}
    )
    assert out.passed is False
    assert out.flags["missing_keys"] == ["confidence"]


def test_json_output_expected_keys_fall_back_to_case():
    out = ev._check_json_output(
        _c(content='{"intent": "x"}', expected={"json_keys": ["intent"]}), {}
    )
    assert out.passed is True


def test_json_output_ignores_braces_inside_strings():
    out = ev._check_json_output(_c(content='prose {"msg": "a } b", "k": 1} tail'), {})
    assert out.flags["repaired_ok"] is True
    assert out.flags["prose_suffix"] == "tail"


# ---------------------------------------------------------------- structure tags


def test_structure_tags_wellformed():
    out = ev._check_structure_tags(
        _c(content="<reason>because</reason><respond>hello</respond>"), {}
    )
    assert out.passed is True


def test_structure_tags_unclosed_fails():
    out = ev._check_structure_tags(_c(content="<reason>because<respond>hi</respond>"), {})
    assert out.passed is False


def test_structure_tags_wrong_order_fails():
    out = ev._check_structure_tags(
        _c(content="<respond>hi</respond><reason>because</reason>"), {}
    )
    assert out.passed is False
    assert "order" in (out.reason or "")


def test_structure_tags_text_outside_optional():
    text = "preamble <reason>r</reason><respond>x</respond>"
    assert ev._check_structure_tags(_c(content=text), {}).passed is True
    strict = ev._check_structure_tags(_c(content=text), {"no_text_outside": True})
    assert strict.passed is False


# ---------------------------------------------------------------- regex


def test_regex_forbid_and_require():
    out = ev._check_regex(_c(content="-- weird dash opener"), {"forbid": [r"^\s*-{2,}"]})
    assert out.passed is False
    out2 = ev._check_regex(_c(content="Hello there"), {"require": [r"[Hh]ello"]})
    assert out2.passed is True


def test_regex_bad_pattern_is_reported_not_raised():
    out = ev._check_regex(_c(content="x"), {"forbid": ["(unclosed"]})
    assert out.passed is False
    assert "bad forbid pattern" in (out.reason or "")


def test_regex_accepts_newline_separated_string():
    """The UI hands textareas through as one string, not a list."""
    out = ev._check_regex(_c(content="alpha"), {"require": "alpha\nbeta"})
    assert out.passed is False       # beta missing
    assert out.flags["matches"]["require:alpha"] is True


# ---------------------------------------------------------------- tool calls


def test_tool_calls_flags_unparseable_arguments():
    comp = _c(expected={"_tool_calls": [
        {"function": {"name": "get_bill", "arguments": "{not json"}}
    ]})
    out = ev._check_tool_calls(comp, {})
    assert out.passed is False
    assert out.flags["unparseable_arguments"] == ["get_bill"]


def test_tool_calls_expected_names():
    comp = _c(expected={"_tool_calls": [
        {"function": {"name": "get_bill", "arguments": "{}"}}
    ]})
    assert ev._check_tool_calls(comp, {"expected_names": ["get_bill"]}).passed is True
    assert ev._check_tool_calls(comp, {"expected_names": ["escalate"]}).passed is False


def test_tool_calls_max_calls_catches_a_loop():
    comp = _c(expected={"_tool_calls": [
        {"function": {"name": "f", "arguments": "{}"}} for _ in range(12)
    ]})
    out = ev._check_tool_calls(comp, {"max_calls": 3})
    assert out.passed is False


# ---------------------------------------------------------------- misc detectors


def test_finish_length():
    assert ev._check_finish_length(_c(finish_reason="length"), {}).passed is False
    assert ev._check_finish_length(_c(finish_reason="stop"), {}).passed is True


def test_latency_thresholds():
    out = ev._check_latency(_c(latency_ms=5000, ttft_ms=100), {"max_latency_ms": 1000})
    assert out.passed is False
    assert out.score == 5000.0


def test_cost_computation():
    out = ev._check_cost(
        _c(usage={"prompt_tokens": 1000, "completion_tokens": 500}),
        {"input_per_1k": 0.01, "output_per_1k": 0.02},
    )
    assert out.flags["cost_usd"] == pytest.approx(0.02)


def test_request_error():
    assert ev._check_request_error(_c(error="HTTP 500"), {}).passed is False
    assert ev._check_request_error(_c(), {}).passed is True


# ---------------------------------------------------------------- the driver


def test_run_evaluators_always_includes_request_error():
    outcomes, passed = ev.run_evaluators(_c(content="fine"), [])
    assert [o.id for o in outcomes] == ["request_error"]
    assert passed is True


def test_run_evaluators_error_short_circuits_verdict():
    """Content checks ran against an empty string; their passes are meaningless,
    so a failed call must never be reported as a clean sample."""
    outcomes, passed = ev.run_evaluators(
        _c(error="HTTP 500: boom"), [{"id": "empty_response", "options": {"min_chars": 0}}]
    )
    assert passed is False


def test_run_evaluators_skips_unknown_ids():
    outcomes, passed = ev.run_evaluators(_c(content="ok"), [{"id": "no_such_detector"}])
    assert [o.id for o in outcomes] == ["request_error"]
    assert passed is True


def test_run_evaluators_survives_a_broken_detector(monkeypatch):
    def boom(_c, _o):
        raise ValueError("detector exploded")

    monkeypatch.setattr(ev.SPECS["degeneration"], "fn", boom)
    outcomes, passed = ev.run_evaluators(_c(content="hi"), [{"id": "degeneration"}])
    deg = next(o for o in outcomes if o.id == "degeneration")
    assert deg.flags.get("evaluator_error") is True
    assert passed is True   # a broken detector must not fail the sample


def test_run_evaluators_deduplicates():
    outcomes, _ = ev.run_evaluators(
        _c(content="hi"), [{"id": "finish_length"}, {"id": "finish_length"}]
    )
    assert [o.id for o in outcomes].count("finish_length") == 1


def test_specs_payload_is_json_serializable():
    import json
    payload = ev.specs_payload()
    json.dumps(payload)   # the web form reads this verbatim
    assert any(e["id"] == "llm_judge" and e["deferred"] for e in payload["evaluators"])
    assert "request_error" in payload["always_on"]


# ------------------------------------------------- Function Call Unit Tests


TOOLS = [
    {"function": {"name": "get_bill", "parameters": {
        "type": "object", "required": ["account_id"],
        "properties": {"account_id": {"type": "string"}, "months": {"type": "integer"}}}}},
    {"function": {"name": "escalate", "parameters": {"type": "object", "properties": {}}}},
]


def _fc(model_calls, ref_calls, **extra) -> ev.EvalOutcome:
    expected = {"tool_calls": ref_calls, "_tool_calls": model_calls, "_tools": TOOLS}
    expected.update(extra)
    return ev._check_function_call_units(_c(expected=expected), {})


def _call(name, args='{"account_id": "A-1"}'):
    return {"function": {"name": name, "arguments": args}}


def test_fc_exact_match_passes():
    out = _fc([_call("get_bill")], [_call("get_bill")])
    assert out.passed is True
    assert out.flags["tp"] == 1 and out.flags["hallucinated"] == 0


def test_fc_missing_call_is_a_false_negative():
    out = _fc([], [_call("get_bill")])
    assert out.passed is False
    assert out.flags["fn"] == 1
    assert "should have called a tool" in (out.reason or "")


def test_fc_spurious_call_is_a_false_positive():
    out = _fc([_call("get_bill")], [])
    assert out.passed is False
    assert out.flags["fp"] == 1


def test_fc_no_call_expected_and_none_made_passes():
    out = _fc([], [])
    assert out.passed is True
    assert out.flags["tp"] == 0 and out.flags["fp"] == 0 and out.flags["fn"] == 0


def test_fc_wrong_function_name():
    out = _fc([_call("escalate")], [_call("get_bill")])
    assert out.passed is False
    assert out.flags["name_fp"] == 1 and out.flags["name_fn"] == 1


def test_fc_hallucinated_function():
    """A call to a function not in the schema the model was given."""
    out = _fc([_call("delete_everything")], [_call("get_bill")])
    assert out.passed is False
    assert out.flags["hallucinated"] == 1


def test_fc_invalid_json_arguments():
    out = _fc([_call("get_bill", "{not json")], [_call("get_bill")])
    assert out.passed is False
    assert out.flags["json_valid"] == 0


def test_fc_required_coverage_and_type_accuracy():
    ok = _fc([_call("get_bill", '{"account_id": "A-1", "months": 3}')], [_call("get_bill")])
    assert ok.flags["req_covs"] == [1.0] and ok.flags["type_accs"] == [1.0]
    # required param missing, and `months` is a string where an integer is declared
    bad = _fc([_call("get_bill", '{"months": "three"}')], [_call("get_bill")])
    assert bad.flags["req_covs"] == [0.0] and bad.flags["type_accs"] == [0.0]


def test_fc_parallel_count_match():
    two_ref = [_call("get_bill"), _call("escalate")]
    assert _fc(two_ref, two_ref).flags["parallel_match"] == 1
    assert _fc([_call("get_bill")], two_ref).flags["parallel_match"] == 0


def test_fc_id_propagation():
    """Of the ids the reference forwards from an earlier tool result, how many
    did the model forward too?"""
    ref = [_call("get_bill", '{"run_id": "recon-2026-07-01"}')]
    good = _fc(ref, ref, available_ids=["recon-2026-07-01"])
    assert good.flags["id_prop"] == 1.0
    bad = _fc([_call("get_bill", '{"run_id": "made-up"}')], ref,
              available_ids=["recon-2026-07-01"])
    assert bad.flags["id_prop"] == 0.0
    # No ids in play → not scored, so it can't drag the average down.
    assert _fc(ref, ref).flags["id_prop"] is None


def test_fc_ids_extracted_from_tool_results():
    ref = [_call("get_bill", '{"run_id": "recon-2026-07-01"}')]
    out = _fc(ref, ref, tool_results=['{"reconciliation_run_id": "recon-2026-07-01"}'])
    assert out.flags["id_prop"] == 1.0


def test_fc_out_of_context_turn_must_refuse():
    """The refusal metric: off-topic turns should draw no tool call at all."""
    silent = _fc([], [], out_of_context=True)
    assert silent.passed is True and silent.flags["refusal_ok"] == 1
    called = _fc([_call("get_bill")], [], out_of_context=True)
    assert called.passed is False and called.flags["refusal_ok"] == 0


def test_fc_aggregate_reproduces_corpus_f1():
    """Corpus F1 pools counts; averaging per-turn rates gives a different number,
    which is the whole reason the aggregate hook exists."""
    rows = [
        _fc([_call("get_bill")], [_call("get_bill")]).flags,   # tp
        _fc([_call("get_bill")], [_call("get_bill")]).flags,   # tp
        _fc([], [_call("get_bill")]).flags,                    # fn
        _fc([_call("get_bill")], []).flags,                    # fp
    ]
    m = ev._agg_function_call_units(rows)
    assert m["tool_call_precision"] == 0.6667       # 2 / (2+1)
    assert m["tool_call_recall"] == 0.6667          # 2 / (2+1)
    assert m["tool_call_f1"] == 0.6667
    assert m["json_valid_rate"] == 1.0
    assert m["hallucination_rate"] == 0.0
    assert m["parallel_count_match"] == 0.6667      # 2 of 3 turns with a ref call


def test_fc_aggregate_refusal_only_when_applicable():
    plain = ev._agg_function_call_units([_fc([], []).flags])
    assert "refusal_rate" not in plain
    withref = ev._agg_function_call_units([_fc([], [], out_of_context=True).flags])
    assert withref["refusal_rate"] == 1.0


# --------------------------------------------------- Multilingual Unit Tests


def _ml(content, expected_lang=None, **opts) -> ev.EvalOutcome:
    exp = {"language": expected_lang} if expected_lang else {}
    return ev._check_multilingual_units(_c(content=content, expected=exp), opts)


def test_ml_matching_language_passes():
    assert _ml("Your account is active, no bars applied.", "english").passed is True
    assert _ml("Akaun anda masih aktif dan tiada sekatan.", "malay").passed is True
    assert _ml("您的账户目前处于正常状态，没有任何限制。", "chinese").passed is True
    assert _ml("உங்கள் கணக்கு தற்போது செயலில் உள்ளது.", "tamil").passed is True


def test_ml_wrong_language_fails_with_both_named():
    out = _ml("Your account is active.", "malay")
    assert out.passed is False
    assert "replied in english, expected malay" in (out.reason or "")


def test_ml_indonesian_counted_as_a_malay_failure():
    """fastText maps __label__id to malay, so an Indonesian reply would otherwise
    be credited as correct Malay — the benchmark's corrected column removes it."""
    text = "Nomor rekening Anda sudah dikonfirmasi karena tidak ada kendala."
    strict = _ml(text, "malay")
    assert strict.passed is False
    assert strict.flags["indonesian_leak"]
    lenient = _ml(text, "malay", strict_malay=False)
    assert lenient.passed is True          # matches the benchmark's RAW malay column


def test_ml_option_can_force_the_target_language():
    assert _ml("Akaun anda aktif.", None, language="malay").passed is True


def test_ml_abstains_without_a_target():
    out = _ml("Some reply with no expectation attached.")
    assert out.passed is True
    assert out.flags["skipped"] is True
    assert "no target language" in (out.reason or "")


def test_ml_abstains_on_an_empty_reply():
    """empty_response is the detector for blank replies; this one must not also
    count it as a language failure."""
    out = _ml("   ", "malay")
    assert out.flags["skipped"] is True
    assert out.passed is True


def test_ml_records_which_detector_ran():
    """Two runs scored by fastText vs the builtin are not comparable, so the
    detector is stamped on every result."""
    assert _ml("Your account is active.", "english").flags["detector"] in ("builtin", "fasttext")


def test_ml_aggregate_per_language_and_correction():
    rows = [
        _ml("Your account is active.", "english").flags,
        _ml("Akaun anda masih aktif.", "malay").flags,
        _ml("Nomor rekening Anda dikonfirmasi karena kendala.", "malay").flags,
        _ml("您的账户正常。", "chinese").flags,
        _ml("   ", "malay").flags,                      # skipped, must not count
    ]
    m = ev._agg_multilingual_units(rows)
    assert m["scored"] == 4
    assert m["english"] == 1.0 and m["chinese"] == 1.0
    assert m["malay"] == 1.0                            # raw: both credited
    assert m["indonesian_leak"] == 1
    assert m["malay_corrected"] == 0.5                  # one was really Indonesian
    assert m["overall"] == 1.0 and m["overall_corrected"] == 0.75


def test_ml_aggregate_empty_is_safe():
    assert ev._agg_multilingual_units([{"skipped": True}])["scored"] == 0


def test_new_evaluators_are_in_the_registry_payload():
    payload = ev.specs_payload()
    ids = {e["id"] for e in payload["evaluators"]}
    assert {"function_call_units", "multilingual_units"} <= ids
    fc = next(e for e in payload["evaluators"] if e["id"] == "function_call_units")
    assert fc["headline"] == ["tool_call_f1", "name_set_f1", "hallucination_rate"]


def test_fc_aggregate_reports_scored_so_a_no_op_suite_is_visible():
    """A dataset with no expected.tool_calls produces turns where nothing is
    compared — every metric then reads 0.0/1.0 while the suite looks green.
    `scored` is what makes that visible."""
    empty = [_fc([], []).flags for _ in range(5)]
    m = ev._agg_function_call_units(empty)
    assert m["turns"] == 5
    assert m["scored"] == 0            # nothing was actually compared
    real = ev._agg_function_call_units(empty + [_fc([_call("get_bill")], [_call("get_bill")]).flags])
    assert real["scored"] == 1 and real["turns"] == 6
