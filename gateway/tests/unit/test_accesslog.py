"""accesslog — JSON record shape, vLLM line severity lifting, and the
request-id logging context added by the hardening pass."""
import json
import logging

from gateway import accesslog


def test_log_request_json_shape(monkeypatch, capsys):
    # Force a clean re-init in JSON mode.
    monkeypatch.setattr(accesslog, "_INIT", False)
    monkeypatch.setattr(accesslog, "_JSON", False)
    logging.getLogger("gateway.access").handlers.clear()
    logging.getLogger("gateway.endpoint").handlers.clear()
    monkeypatch.setenv("LOG_JSON", "1")
    monkeypatch.delenv("GATEWAY_ACCESS_LOG", raising=False)
    monkeypatch.delenv("GATEWAY_ENDPOINT_LOG", raising=False)
    accesslog.init_access_logging()

    accesslog.log_request(
        method="POST", route="/{app_id}/v1/chat/completions",
        path="/tm/v1/chat/completions", status=502, duration_ms=12.3456,
        request_id="req-abc", app_id="tm", ip="1.2.3.4", nbytes=17,
    )
    line = capsys.readouterr().out.strip().splitlines()[-1]
    rec = json.loads(line)
    # The fields Promtail/Grafana LogQL queries rely on.
    assert rec["service"] == "gateway"
    assert rec["status"] == 502
    assert rec["statusClass"] == "5xx"
    assert rec["level"] == "error"          # 5xx → error
    assert rec["route"] == "/{app_id}/v1/chat/completions"
    assert rec["requestId"] == "req-abc"
    assert rec["durationMs"] == 12.346
    assert rec["app_id"] == "tm"


def test_line_level_classification():
    assert accesslog._line_level("(EngineCore pid=3) ERROR 06-28 boom") == "error"
    assert accesslog._line_level("WARNING something") == "warn"
    assert accesslog._line_level("INFO fine") == "info"
    assert accesslog._line_level("no level token here") == "info"
    assert accesslog._line_level("DEBUG verbose") == "debug"


def test_request_id_filter_stamps_records():
    f = accesslog._RequestIdFilter()
    rec = logging.LogRecord("gateway.x", logging.INFO, __file__, 1, "msg", (), None)

    token = accesslog.request_id_var.set("req-123")
    try:
        f.filter(rec)
        assert rec.request_id == " [req-123]"
    finally:
        accesslog.request_id_var.reset(token)

    # Outside a request the field renders as nothing, not "None".
    f.filter(rec)
    assert rec.request_id == ""
