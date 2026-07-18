"""netsafe — the SSRF guard for user-supplied server-side fetch URLs."""
import socket

import pytest

from gateway import netsafe
from gateway.netsafe import assert_safe_fetch_url


def _fake_resolver(mapping):
    def _getaddrinfo(host, port, proto=None, **kw):
        ips = mapping.get(host)
        if ips is None:
            raise OSError(f"unresolvable: {host}")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 80)) for ip in ips]
    return _getaddrinfo


@pytest.mark.parametrize("url", [
    "ftp://example.com/x",
    "file:///etc/passwd",
    "gopher://example.com",
    "",
    "not-a-url",
])
def test_bad_scheme_or_shape_rejected(url):
    with pytest.raises(ValueError):
        assert_safe_fetch_url(url)


def test_link_local_metadata_ip_literal_rejected(monkeypatch):
    monkeypatch.setattr(
        netsafe.socket, "getaddrinfo",
        _fake_resolver({"169.254.169.254": ["169.254.169.254"]}),
    )
    with pytest.raises(ValueError, match="blocked address"):
        assert_safe_fetch_url("http://169.254.169.254/latest/meta-data/")


def test_hostname_resolving_to_metadata_rejected(monkeypatch):
    # The metadata.google.internal shape: benign hostname, link-local A record.
    monkeypatch.setattr(
        netsafe.socket, "getaddrinfo",
        _fake_resolver({"metadata.internal": ["169.254.169.254"]}),
    )
    with pytest.raises(ValueError, match="blocked address"):
        assert_safe_fetch_url("http://metadata.internal/creds")


def test_private_and_loopback_allowed(monkeypatch):
    # Internal services are legitimate targets here (Label platform, ingresses).
    monkeypatch.setattr(
        netsafe.socket, "getaddrinfo",
        _fake_resolver({"internal.svc": ["10.1.2.3"], "localhost": ["127.0.0.1"]}),
    )
    assert assert_safe_fetch_url("http://internal.svc:8080/x") == "http://internal.svc:8080/x"
    assert assert_safe_fetch_url("https://localhost/x") == "https://localhost/x"


def test_unresolvable_host_rejected(monkeypatch):
    monkeypatch.setattr(netsafe.socket, "getaddrinfo", _fake_resolver({}))
    with pytest.raises(ValueError, match="could not resolve"):
        assert_safe_fetch_url("http://nope.invalid/x")


def test_multi_a_record_any_blocked_rejects(monkeypatch):
    # DNS answers with one clean + one link-local record → reject (rebinding).
    monkeypatch.setattr(
        netsafe.socket, "getaddrinfo",
        _fake_resolver({"evil.example": ["93.184.216.34", "169.254.1.1"]}),
    )
    with pytest.raises(ValueError, match="blocked address"):
        assert_safe_fetch_url("http://evil.example/")
