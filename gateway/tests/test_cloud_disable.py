"""Unit tests for the DISABLE_CLOUD_PROVIDERS gate (CAE/CCE lock-down).

Pure-function tests — no running platform required (unlike the integration
suite). They exercise the central kill-switch and the benchmark policy helper
so a regression that re-opens a cloud path is caught in CI.
"""
from __future__ import annotations

import pytest

from gateway.provider import (
    CLOUD_PROVIDER_NAMES,
    CloudProviderDisabled,
    FakeProvider,
    build_provider,
    cloud_providers_disabled,
    ensure_benchmark_provider_allowed,
)
from gateway.provider_resolve import resolve_cloud_creds, try_resolve_cloud_creds


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("", False), ("  ", False),
    ],
)
def test_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("DISABLE_CLOUD_PROVIDERS", value)
    assert cloud_providers_disabled() is expected


def test_flag_absent_defaults_off(monkeypatch):
    monkeypatch.delenv("DISABLE_CLOUD_PROVIDERS", raising=False)
    assert cloud_providers_disabled() is False


def test_build_provider_blocks_cloud_when_disabled(monkeypatch):
    monkeypatch.setenv("DISABLE_CLOUD_PROVIDERS", "1")
    for name in ("runpod", "primeintellect"):
        assert name in CLOUD_PROVIDER_NAMES
        with pytest.raises(CloudProviderDisabled):
            build_provider(name)


def test_build_provider_fake_unaffected_when_disabled(monkeypatch):
    # `fake` (and per-app `vm`) must keep working — only cloud is gated.
    monkeypatch.setenv("DISABLE_CLOUD_PROVIDERS", "1")
    assert isinstance(build_provider("fake"), FakeProvider)


def test_ensure_benchmark_provider_allowed_enabled_is_noop(monkeypatch):
    monkeypatch.delenv("DISABLE_CLOUD_PROVIDERS", raising=False)
    for kind in (None, "vm", "runpod", "pi"):
        ensure_benchmark_provider_allowed(kind)  # no raise


def test_ensure_benchmark_provider_allowed_disabled(monkeypatch):
    monkeypatch.setenv("DISABLE_CLOUD_PROVIDERS", "1")
    # vm is allowed; no-provider (defaults to runpod bench) and cloud kinds are not.
    ensure_benchmark_provider_allowed("vm")
    for kind in (None, "runpod", "pi"):
        with pytest.raises(CloudProviderDisabled):
            ensure_benchmark_provider_allowed(kind)


async def test_resolve_cloud_creds_blocked_before_db(monkeypatch):
    # The kill-switch fires before any DB/session access, so session=None is safe.
    monkeypatch.setenv("DISABLE_CLOUD_PROVIDERS", "1")
    with pytest.raises(CloudProviderDisabled):
        await resolve_cloud_creds(None, None, "runpod")
    with pytest.raises(CloudProviderDisabled):
        await resolve_cloud_creds(None, "prov-abc", "pi")


async def test_try_resolve_cloud_creds_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv("DISABLE_CLOUD_PROVIDERS", "1")
    assert await try_resolve_cloud_creds(None, None, "runpod") is None
