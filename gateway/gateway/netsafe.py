"""SSRF guard for user-supplied fetch URLs (label_base_url, ingress base_url).

These fields name an external service the gateway then GETs server-side (and, for
label audio, streams the body back to the browser) — the classic SSRF shape. We
CANNOT simply block private ranges: the Label platform and inference ingresses
legitimately live on the internal network / localhost. So we block the parts that
are never a legitimate target:

  * non-http(s) schemes (file://, gopher://, … — used to reach the FS / other
    protocols), and
  * link-local + cloud-metadata addresses (169.254.0.0/16 incl. 169.254.169.254,
    fe80::/10) — the usual credential-exfil target.

DNS is resolved so a hostname that points at a link-local address (e.g.
metadata.google.internal) is caught too, not just IP literals. Redirect-following
must be disabled at the call site so a 3xx can't bounce a validated host onto a
blocked one.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def _blocked_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # unparseable → refuse
    return addr.is_link_local or addr.is_multicast or addr.is_unspecified


def assert_safe_fetch_url(url: str) -> str:
    """Return `url` if it's a safe server-side fetch target; raise ValueError
    otherwise. Allows public AND private/loopback hosts (internal services are
    legitimate here) but rejects bad schemes and link-local/metadata addresses."""
    if not url or not isinstance(url, str):
        raise ValueError("empty URL")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL scheme must be http or https, got {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError("URL has no host")
    # If the host is an IP literal, check it directly; otherwise resolve it and
    # check every address it maps to (catches metadata.* style hostnames).
    try:
        infos = socket.getaddrinfo(host, parsed.port or None, proto=socket.IPPROTO_TCP)
    except OSError as e:
        raise ValueError(f"could not resolve host {host!r}: {e}") from e
    for info in infos:
        ip = info[4][0]
        if _blocked_ip(ip):
            raise ValueError(f"host {host!r} resolves to a blocked address ({ip})")
    return url
