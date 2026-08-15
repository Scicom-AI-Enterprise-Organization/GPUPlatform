"""A switched-off upstream must stay VISIBLE everywhere it still matters.

Disabling an upstream never removes it from the config — but every read surface used
to behave as if it had: the routing graph erased it, `/proxy/{name}/health` didn't
count it, and the request error said "not served by this endpoint", which reads like a
config typo rather than "your standby is switched off". Reported from prod as
"I disabled the fallback and it went missing forever" — the fallback was one toggle
away the whole time, with nothing on the page connecting the outage to it.

These pin the gateway half: the error names the disabled backend, and the health
report separates "no backend" from "a backend, switched off".
"""
from __future__ import annotations

import time
import types

from gateway.proxy_api import (
    HEALTH_TTL_S,
    _disabled_serving,
    _no_candidate_error,
    _proxy_health_report,
)


def _cfg():
    return {"upstreams": [
        {"id": "up-1", "name": "gemma-h20", "base_url": "http://a/v1", "enabled": True,
         "models": {"gemma": "gemma-4-31b-it"}},
        {"id": "up-2", "name": "openrouter", "base_url": "http://b/v1", "enabled": False,
         "models": {"gemma": "gemma-4-31b-it", "gemma-free": "google/gemma:free"}},
    ]}


def _app(health=None):
    app = types.SimpleNamespace(state=types.SimpleNamespace())
    app.state.proxy_health = health or {}
    return app


def _ep(cfg, enabled=True):
    return types.SimpleNamespace(id="proxy-1", name="repro", enabled=enabled, config=cfg)


# ---------- which upstreams are switched off for an alias --------------------

def test_disabled_serving_finds_only_off_upstreams_mapping_the_alias():
    assert [u["name"] for u in _disabled_serving(_cfg(), "gemma")] == ["openrouter"]
    assert [u["name"] for u in _disabled_serving(_cfg(), "gemma-free")] == ["openrouter"]
    assert _disabled_serving(_cfg(), "not-a-model") == []


def test_an_enabled_upstream_is_never_reported_as_disabled():
    cfg = {"upstreams": [{"id": "u", "name": "on", "enabled": True, "models": {"m": "m"}}]}
    assert _disabled_serving(cfg, "m") == []


def test_missing_enabled_key_defaults_to_on():
    """Legacy rows predate the flag — absent must mean enabled, not disabled."""
    cfg = {"upstreams": [{"id": "u", "name": "legacy", "models": {"m": "m"}}]}
    assert _disabled_serving(cfg, "m") == []


# ---------- the error a client gets ------------------------------------------

def test_no_candidate_error_names_the_switched_off_backend():
    d = _no_candidate_error(_cfg(), "gemma-free", "repro")
    assert "switched off" in d["error"]
    assert "openrouter" in d["error"]
    assert d["disabled_upstreams"] == ["openrouter"]


def test_no_candidate_error_is_unchanged_when_nothing_serves_the_alias():
    """No disabled backend to point at → don't invent a hint."""
    d = _no_candidate_error(_cfg(), "mistral", "repro")
    assert d["error"] == "model 'mistral' is not served by endpoint 'repro'"
    assert "disabled_upstreams" not in d


def test_forced_upstream_keeps_its_own_error():
    d = _no_candidate_error(_cfg(), "gemma", "repro", force="openrouter")
    assert "forced upstream 'openrouter'" in d["error"]


# ---------- the health report ------------------------------------------------

def test_disabled_upstream_is_counted_separately_not_omitted():
    body, code = _proxy_health_report(_app(), _ep(_cfg()))
    assert code == 200
    assert body["upstreams_total"] == 1        # only the enabled one serves traffic
    assert body["upstreams_disabled"] == 1     # ...but the standby is still reported
    assert [u["name"] for u in body["disabled"]] == ["openrouter"]


def test_unhealthy_endpoint_with_a_switched_off_standby_says_so():
    """The whole point: 503 because the primary is dead, while a usable backup sits
    one toggle away. Without the hint the two situations are indistinguishable."""
    health = {("proxy-1", "up-1"): {"alive": False, "checked_at": time.time(), "error": "refused"}}
    body, code = _proxy_health_report(_app(health), _ep(_cfg()))
    assert code == 503 and body["status"] == "unhealthy"
    assert "openrouter" in body["hint"]
    assert "enable one" in body["hint"]


def test_no_hint_when_healthy():
    body, code = _proxy_health_report(_app(), _ep(_cfg()))
    assert code == 200 and "hint" not in body


def test_no_disabled_keys_at_all_when_every_upstream_is_on():
    cfg = {"upstreams": [{"id": "u", "name": "on", "enabled": True, "models": {"m": "m"}}]}
    body, _ = _proxy_health_report(_app(), _ep(cfg))
    assert body["upstreams_disabled"] == 0
    assert "disabled" not in body and "hint" not in body


def test_a_stale_probe_still_counts_as_unknown_not_dead():
    """HEALTH_TTL_S is the existing rule — a stale probe sorts as alive-or-unknown, so
    a disabled backend whose last probe aged out must not turn the endpoint unhealthy."""
    old = time.time() - HEALTH_TTL_S - 5
    health = {("proxy-1", "up-1"): {"alive": False, "checked_at": old, "error": "refused"}}
    body, code = _proxy_health_report(_app(health), _ep(_cfg()))
    assert code == 200 and body["status"] == "healthy"
