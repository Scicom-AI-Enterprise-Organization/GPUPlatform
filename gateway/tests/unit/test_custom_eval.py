"""custom_eval — user-written evaluators.

The expression tests are a security boundary, not a feature checklist: each
`pytest.raises` pins an escape that must stay closed.
"""
import pytest

from gateway import custom_eval as ce
from gateway import evaluators as ev


def _c(**kw) -> ev.Completion:
    return ev.Completion(**kw)


def run(code: str, fail_when_true: bool = False, **completion) -> ev.EvalOutcome:
    spec = ce.CustomSpec(id="t", name="t", mode="expression", code=code,
                         fail_when_true=fail_when_true)
    return ce.run_expression_evaluator(spec, _c(**completion))


# ------------------------------------------------------------------ the sandbox


@pytest.mark.parametrize("code", [
    "__import__('os').system('id')",
    "().__class__",
    "().__class__.__bases__",
    "content.__class__.__mro__",
    "open('/etc/passwd')",
    "eval('1+1')",
    "exec('x=1')",
    "globals()",
    "locals()",
    "vars()",
    "getattr(content, 'upper')",
    "content.encode",                  # not in SAFE_METHODS
    "[x for x in content]",            # comprehensions are refused outright
    "content.__len__()",
])
def test_expression_rejects_escapes(code):
    with pytest.raises(ce.CustomEvalError):
        ce.validate_expression(code, {"content"})


@pytest.mark.parametrize("code,match", [
    ("2**(10**10)", r"\*\*"),                 # classic 3-char CPU bomb
    ("content ** 2", r"\*\*"),
    ("content * 100000000", "multiplying by"),  # gigabyte allocation
    ("100000 * content", "multiplying by"),
])
def test_expression_rejects_compute_bombs(code, match):
    """Expression mode runs IN the gateway process — these would take it down."""
    with pytest.raises(ce.CustomEvalError, match=match):
        ce.validate_expression(code, {"content"})


def test_expression_allows_small_multipliers():
    ce.validate_expression("content * 3", {"content"})
    ce.validate_expression("len(content) * len(content)", {"content"})


def test_expression_rejects_statements_and_assignment():
    for code in ("x = 1", "import os", "lambda: 1", "content := 1"):
        with pytest.raises(ce.CustomEvalError):
            ce.validate_expression(code, {"content"})


def test_expression_rejects_unknown_names():
    with pytest.raises(ce.CustomEvalError, match="unknown name"):
        ce.validate_expression("secret_thing > 1", {"content"})


def test_expression_rejects_non_helper_calls():
    with pytest.raises(ce.CustomEvalError):
        ce.validate_expression("print(content)", {"content"})


def test_expression_rejects_empty_and_oversized():
    with pytest.raises(ce.CustomEvalError, match="empty"):
        ce.validate_expression("   ", {"content"})
    with pytest.raises(ce.CustomEvalError, match="too long"):
        ce.validate_expression("1 + " * 2000 + "1", {"content"})


def test_syntax_error_is_reported_not_raised_as_500():
    with pytest.raises(ce.CustomEvalError, match="syntax error"):
        ce.validate_expression("content ==", {"content"})


# ------------------------------------------------------------------ evaluating


def test_simple_pass_and_fail():
    assert run("len(content) > 3", content="hello").passed is True
    assert run("len(content) > 3", content="hi").passed is False


def test_fail_when_true_inverts_the_sense():
    """The detector idiom: write the bug, not the health check."""
    fenced = "```json\n{}\n```"
    assert run('re_search("```", content)', fail_when_true=True, content=fenced).passed is False
    assert run('re_search("```", content)', fail_when_true=True, content="{}").passed is True


def test_safe_methods_work():
    assert run('content.lower().startswith("hel")', content="HELlo").passed is True
    assert run('content.count("a") == 2', content="banana"[:4]).passed is True


def test_helpers_regex_and_json():
    assert run('is_json_object(content)', content='{"a":1}').passed is True
    assert run('is_json_object(content)', content='```{"a":1}```').passed is False
    assert run('json_loads(content).get("a") == 1', content='{"a":1}').passed is True
    assert run('"intent" in json_keys(content)', content='{"intent":"x"}').passed is True
    assert run('re_count("a", content) == 3', content="banana").passed is True


def test_json_loads_returns_none_instead_of_raising():
    assert run("json_loads(content) is None", content="not json").passed is True


def test_degeneration_helpers_are_reusable():
    assert run("max_repeat(content) >= 10", content="spam " * 50).passed is True
    assert run("distinct_ratio(content) < 0.2", content="spam " * 50).passed is True
    assert run("compression_ratio(content) < 0.2", content="spam " * 200).passed is True


def test_control_tokens_helper_matches_the_builtin():
    assert run("len(control_tokens(content)) > 0", content="<|channel>x").passed is True


def test_tool_calls_are_exposed_as_plain_data():
    comp = _c(expected={"_tool_calls": [
        {"function": {"name": "get_bill", "arguments": '{"id": 1}'}}
    ]})
    spec = ce.CustomSpec(id="t", name="t", mode="expression",
                         code='tool_calls[0]["name"] == "get_bill"')
    assert ce.run_expression_evaluator(spec, comp).passed is True
    spec2 = ce.CustomSpec(id="t", name="t", mode="expression",
                          code='tool_calls[0]["parsed_arguments"].get("id") == 1')
    assert ce.run_expression_evaluator(spec2, comp).passed is True


def test_private_expected_keys_are_hidden():
    """`_tool_calls` is plumbing — it must not leak into `expected`."""
    comp = _c(expected={"_tool_calls": [], "json_keys": ["a"]})
    vars_ = ce.completion_vars(comp)
    assert vars_["expected"] == {"json_keys": ["a"]}


def test_latency_and_usage_are_available():
    assert run("latency_ms > 100", latency_ms=200).passed is True
    assert run("completion_tokens == 7",
               usage={"completion_tokens": 7}).passed is True


def test_runtime_error_is_reported_and_does_not_fail_the_sample():
    """An author bug must not silently mark real replies as failures."""
    out = run("json_loads(content).get('a') == 1", content="not json")
    assert out.flags.get("evaluator_error") is True
    assert out.passed is True
    assert "AttributeError" in (out.reason or "")


def test_invalid_expression_is_reported_not_raised():
    out = run("__import__('os')")
    assert out.flags.get("evaluator_error") is True
    assert out.passed is True


# ------------------------------------------------------------------ normalizing


def test_normalize_bool_and_number():
    assert ce.normalize_result(True, False, "n").passed is True
    assert ce.normalize_result(False, False, "n").passed is False
    assert ce.normalize_result(True, True, "n").passed is False      # inverted
    out = ce.normalize_result(4, False, "n")
    assert out.passed is True and out.score == 4.0


def test_normalize_dict_states_its_own_verdict():
    """An explicit dict is authoritative — flipping it too would be a trap."""
    out = ce.normalize_result(
        {"passed": False, "score": 0.25, "reason": "nope", "flags": {"k": 1}},
        fail_when_true=True, name="n",
    )
    assert out.passed is False
    assert out.score == 0.25
    assert out.reason == "nope"
    assert out.flags == {"k": 1}


def test_normalize_supplies_a_reason_on_failure():
    assert ce.normalize_result(False, False, "mycheck").reason


# ------------------------------------------------------------------ validation


def test_validate_spec_rejects_unknown_mode():
    with pytest.raises(ce.CustomEvalError, match="mode must be"):
        ce.validate_spec("javascript", "1", allow_python=True)


def test_python_mode_blocked_unless_enabled():
    with pytest.raises(ce.CustomEvalError, match="disabled"):
        ce.validate_spec("python", "def check(c):\n    return True", allow_python=False)


def test_python_mode_requires_a_check_function():
    with pytest.raises(ce.CustomEvalError, match="named check"):
        ce.validate_spec("python", "def other(c):\n    return True", allow_python=True)
    ce.validate_spec("python", "def check(c):\n    return True", allow_python=True)


def test_python_mode_reports_syntax_errors_with_a_line():
    with pytest.raises(ce.CustomEvalError, match="line"):
        ce.validate_spec("python", "def check(c)\n    return True", allow_python=True)


def test_describe_context_is_serializable_and_examples_validate():
    import json
    ctx = ce.describe_context()
    json.dumps(ctx)
    names = {v["name"] for v in ctx["variables"]}
    assert {"content", "reasoning", "tool_calls", "latency_ms"} <= names
    # Every code-bearing example must actually be accepted by the validator, or
    # the help text teaches people syntax the engine rejects. The api example is
    # a starting template — it has no URL until the author supplies one.
    for ex in ctx["examples"]:
        if ex["mode"] == "api":
            continue
        ce.validate_spec(ex["mode"], ex["code"], allow_python=True)
    assert any(ex["mode"] == "api" for ex in ctx["examples"])
    assert "passed_field" in ctx["api_defaults"]
    assert "api_key_secret" not in ctx["api_defaults"]   # never echo a secret ref


# ------------------------------------------------------------------ python mode


async def test_python_worker_runs_and_reuses_the_process():
    spec = ce.CustomSpec(id="t", name="bullets", mode="python", code=(
        "def check(c):\n"
        "    n = c['content'].count('- ')\n"
        "    return {'passed': n >= 2, 'score': n, 'flags': {'bullets': n}}\n"
    ))
    w = ce.PythonEvaluatorWorker(spec)
    try:
        ok = await w.evaluate(_c(content="- a\n- b"))
        assert ok.passed is True and ok.score == 2.0
        bad = await w.evaluate(_c(content="- a"))
        assert bad.passed is False and bad.flags == {"bullets": 1}
    finally:
        await w.close()


async def test_python_worker_env_is_scrubbed_of_gateway_secrets():
    """The child must not inherit DB / Fernet / cloud credentials."""
    spec = ce.CustomSpec(id="t", name="env", mode="python", code=(
        "import os\n"
        "def check(c):\n"
        "    leaked = [k for k in ('DATABASE_URL','PROVIDER_SECRET_KEY','RUNPOD_API_KEY',\n"
        "                          'AWS_SECRET_ACCESS_KEY','HF_TOKEN') if os.environ.get(k)]\n"
        "    return {'passed': not leaked, 'flags': {'leaked': leaked}}\n"
    ))
    import os
    # RESTORE the prior key afterwards, don't pop it — conftest seeds
    # PROVIDER_SECRET_KEY for the whole run, and later tests (e.g. the red-team
    # builder's Fernet encryption) re-read it after test_crypto clears the cache.
    prev_key = os.environ.get("PROVIDER_SECRET_KEY")
    os.environ["DATABASE_URL"] = "postgresql://user:pw@host/db"
    os.environ["PROVIDER_SECRET_KEY"] = "super-secret"
    w = ce.PythonEvaluatorWorker(spec)
    try:
        out = await w.evaluate(_c(content="x"))
        assert out.flags.get("leaked") == [], f"secrets reached the child: {out.flags}"
        assert out.passed is True
    finally:
        await w.close()
        os.environ.pop("DATABASE_URL", None)
        if prev_key is None:
            os.environ.pop("PROVIDER_SECRET_KEY", None)
        else:
            os.environ["PROVIDER_SECRET_KEY"] = prev_key


async def test_python_worker_reports_a_raising_check_without_failing_the_sample():
    spec = ce.CustomSpec(id="t", name="boom", mode="python", code=(
        "def check(c):\n    raise ValueError('kaboom')\n"
    ))
    w = ce.PythonEvaluatorWorker(spec)
    try:
        out = await w.evaluate(_c(content="x"))
        assert out.flags.get("evaluator_error") is True
        assert out.passed is True
        assert "kaboom" in (out.reason or "")
    finally:
        await w.close()


async def test_python_worker_survives_a_hang_and_marks_it():
    spec = ce.CustomSpec(id="t", name="hang", mode="python", code=(
        "import time\ndef check(c):\n    time.sleep(30)\n    return True\n"
    ))
    w = ce.PythonEvaluatorWorker(spec)
    original = ce.PY_CALL_TIMEOUT_S
    ce.PY_CALL_TIMEOUT_S = 1.0
    try:
        out = await w.evaluate(_c(content="x"))
        assert out.flags.get("timeout") is True
        assert out.passed is True
    finally:
        ce.PY_CALL_TIMEOUT_S = original
        await w.close()


async def test_python_worker_fatal_definition_stops_retrying():
    spec = ce.CustomSpec(id="t", name="nodef", mode="python", code="x = 1\n")
    w = ce.PythonEvaluatorWorker(spec)
    try:
        first = await w.evaluate(_c(content="x"))
        assert first.flags.get("evaluator_error") is True
        assert w._fatal is not None       # latched — no respawn storm
        second = await w.evaluate(_c(content="x"))
        assert second.flags.get("evaluator_error") is True
    finally:
        await w.close()


# ---------------------------------------------------------------- api mode


def _api_spec(**cfg) -> ce.CustomSpec:
    return ce.CustomSpec(
        id="t", name="scorer", mode="api",
        code="", fail_when_true=cfg.pop("fail_when_true", False),
        config={"url": "https://scorer.example/score", **cfg},
    )


def _parse(data, **cfg) -> ev.EvalOutcome:
    client = ce.ApiEvaluatorClient(_api_spec(**cfg), client=None)  # type: ignore[arg-type]
    return client.parse_response(data)


def test_dig_walks_dicts_and_lists():
    obj = {"a": {"b": [{"c": 7}]}}
    assert ce.dig(obj, "a.b.0.c") == 7
    assert ce.dig(obj, "a.missing") is None
    assert ce.dig(obj, "a.b.9.c") is None
    assert ce.dig(obj, "") is obj          # "" = the whole response


def test_api_parses_the_default_shape():
    out = _parse({"passed": True, "score": 0.8, "reason": "fine", "flags": {"k": 1}})
    assert out.passed is True
    assert out.score == 0.8
    assert out.reason == "fine"
    assert out.flags == {"k": 1}


def test_api_parses_nested_paths():
    out = _parse(
        {"result": {"verdict": False, "detail": {"why": "too long"}}},
        passed_field="result.verdict", reason_field="result.detail.why",
    )
    assert out.passed is False
    assert out.reason == "too long"


@pytest.mark.parametrize("verdict,expected", [
    ("PASS", True), ("pass", True), ("yes", True), ("ok", True), ("true", True),
    ("FAIL", False), ("fail", False), ("no", False), ("false", False),
])
def test_api_accepts_string_verdicts(verdict, expected):
    """Real scorers answer 'PASS'/'fail' at least as often as a JSON bool."""
    assert _parse({"passed": verdict}).passed is expected


def test_api_bare_boolean_response():
    """An endpoint that answers `true` with no wrapper — passed_field=""."""
    assert _parse(True, passed_field="").passed is True
    assert _parse(False, passed_field="").passed is False


def test_api_fail_when_true_inverts():
    assert _parse({"passed": True}, fail_when_true=True).passed is False


def test_api_score_only_response_infers_the_verdict():
    out = _parse({"score": 0.0}, passed_field="passed")
    assert out.passed is False and out.score == 0.0


def test_api_missing_verdict_is_an_evaluator_error_not_a_failure():
    out = _parse({"unexpected": 1})
    assert out.flags.get("evaluator_error") is True
    assert out.passed is True
    assert "no 'passed' field" in (out.reason or "")


def test_api_non_numeric_score_falls_back():
    out = _parse({"passed": True, "score": "high"})
    assert out.passed is True and out.score == 1.0


# ---- config validation / SSRF ------------------------------------------------


def test_api_config_requires_a_url():
    with pytest.raises(ce.CustomEvalError, match="needs a URL"):
        ce.validate_api_config({})


def test_api_config_rejects_bad_scheme():
    with pytest.raises(ce.CustomEvalError, match="unsafe URL"):
        ce.validate_api_config({"url": "file:///etc/passwd"})


def test_api_config_rejects_cloud_metadata():
    """The classic SSRF target — 169.254.169.254 must never be reachable."""
    with pytest.raises(ce.CustomEvalError, match="unsafe URL"):
        ce.validate_api_config({"url": "http://169.254.169.254/latest/meta-data/"})


def test_api_config_rejects_get_and_bad_timeout():
    with pytest.raises(ce.CustomEvalError, match="method must be"):
        ce.validate_api_config({"url": "http://127.0.0.1:1/x", "method": "GET"})
    with pytest.raises(ce.CustomEvalError, match="timeout"):
        ce.validate_api_config({"url": "http://127.0.0.1:1/x", "timeout_s": 9999})


def test_api_config_allows_internal_hosts():
    """An in-cluster scorer is a legitimate target — don't block private ranges."""
    cfg = ce.validate_api_config({"url": "http://127.0.0.1:9000/score"})
    assert cfg["method"] == "POST"
    assert cfg["passed_field"] == "passed"


def test_validate_spec_api_mode_needs_no_code():
    ce.validate_spec("api", "", allow_python=False,
                     config={"url": "http://127.0.0.1:9000/score"})


def test_api_payload_shape_and_reasoning_toggle():
    comp = _c(content="hi", reasoning="thinking", latency_ms=12)
    with_reasoning = ce.ApiEvaluatorClient(_api_spec(), client=None).payload(comp)  # type: ignore[arg-type]
    assert with_reasoning["content"] == "hi"
    assert with_reasoning["reasoning"] == "thinking"
    assert with_reasoning["evaluator"] == "scorer"
    without = ce.ApiEvaluatorClient(
        _api_spec(include_reasoning=False), client=None,  # type: ignore[arg-type]
    ).payload(comp)
    assert "reasoning" not in without


def test_api_auth_header_uses_the_resolved_key():
    client = ce.ApiEvaluatorClient(_api_spec(), client=None, api_key="sk-123")  # type: ignore[arg-type]
    assert client._headers()["Authorization"] == "Bearer sk-123"
    custom = ce.ApiEvaluatorClient(
        _api_spec(auth_header="X-Api-Key", auth_prefix=""),
        client=None, api_key="sk-123",  # type: ignore[arg-type]
    )
    assert custom._headers()["X-Api-Key"] == "sk-123"


def test_api_no_key_means_no_auth_header():
    client = ce.ApiEvaluatorClient(_api_spec(), client=None)  # type: ignore[arg-type]
    assert "Authorization" not in client._headers()
