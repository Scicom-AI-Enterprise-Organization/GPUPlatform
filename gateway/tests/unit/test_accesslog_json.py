"""LOG_JSON=1 must make the WHOLE stream JSON — not just the access lines.

The prod symptom this pins: a log tail interleaving parseable access records with
plain-text `2026-08-14 08:46:43,226 INFO httpx: …` module lines and uvicorn's own
`INFO:     1.2.3.4 - "GET /ready HTTP/1.1" 200 OK`, so `{service="gateway"} | json`
silently dropped exactly the lines carrying the errors.
"""

from __future__ import annotations

import json
import logging

from gateway import accesslog


def _record(msg="hello", name="httpx", level=logging.INFO, exc_info=None):
    return logging.LogRecord(name=name, level=level, pathname=__file__, lineno=1,
                             msg=msg, args=(), exc_info=exc_info)


def test_module_log_renders_as_json_with_the_access_log_field_names():
    out = json.loads(accesslog._JsonLogFormatter().format(_record()))
    assert out["service"] == "gateway"          # same stream label as http_access
    assert out["kind"] == "app_log"             # …distinguishable from it
    assert out["level"] == "info"
    assert out["logger"] == "httpx"
    assert out["msg"] == "hello"
    assert isinstance(out["time"], int)


def test_warning_level_is_normalised_to_warn():
    """`http_access` records emit `warn`; a `warning` here would split every
    level-filtered query in two."""
    out = json.loads(accesslog._JsonLogFormatter().format(_record(level=logging.WARNING)))
    assert out["level"] == "warn"


def test_request_id_is_carried_into_the_record():
    token = accesslog.request_id_var.set("pxr-4ce76aeb0409d8b5")
    try:
        out = json.loads(accesslog._JsonLogFormatter().format(_record()))
        assert out["requestId"] == "pxr-4ce76aeb0409d8b5"
    finally:
        accesslog.request_id_var.reset(token)


def test_traceback_is_a_field_not_a_multi_line_spill():
    """A traceback printed raw becomes N unparseable lines; as a field it stays one
    JSON record attached to its request."""
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        out = json.loads(accesslog._JsonLogFormatter().format(_record(exc_info=sys.exc_info())))
    assert "ValueError: boom" in out["exception"]
    assert out["msg"] == "hello"
    assert len(out["exception"].splitlines()) > 1   # the traceback survived intact


def test_non_serialisable_message_does_not_break_the_line():
    class Weird:
        def __str__(self):
            return "weird-obj"

    out = json.loads(accesslog._JsonLogFormatter().format(_record(msg=Weird())))
    assert out["msg"] == "weird-obj"


def test_uvicorn_loggers_are_folded_into_root_and_its_access_log_silenced(monkeypatch):
    """uvicorn sets propagate=False + its own handlers; left alone, its lines ignore
    LOG_JSON entirely and duplicate our access log."""
    monkeypatch.delenv("LOG_UVICORN_ACCESS", raising=False)
    for name in ("uvicorn", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = [logging.StreamHandler()]
        lg.propagate = False
        lg.disabled = False
    accesslog._tame_server_loggers()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        assert lg.handlers == [] and lg.propagate is True, name
    assert logging.getLogger("uvicorn.access").disabled is True
    # …and it comes back when explicitly asked for.
    monkeypatch.setenv("LOG_UVICORN_ACCESS", "1")
    accesslog._tame_server_loggers()
    assert logging.getLogger("uvicorn.access").disabled is False
    logging.getLogger("uvicorn.access").disabled = False


def test_truthy_accepts_the_usual_spellings():
    for v in ("1", "true", "TRUE", "yes", "on", " 1 "):
        assert accesslog.truthy(v) is True
    for v in ("", "0", "false", "no", "off"):
        assert accesslog.truthy(v) is False
