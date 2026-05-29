"""Reverse SSH tunnels so a remote VM worker can reach a gateway + Redis that
live on a non-public host.

The VM worker phones home over HTTP (register/heartbeat/logs) and pulls jobs
straight from Redis. When the gateway + Redis aren't reachable from the VM —
local dev on a laptop, or a single-gateway deploy that doesn't want to expose
Redis publicly — we open a reverse tunnel over the SSH connection the VM
provider already uses: the VM's `127.0.0.1:<port>` is forwarded back to the
gateway host's gateway/redis. The worker is then told
`GATEWAY_URL`/`WORKER_REDIS_URL = 127.0.0.1`, which routes through the tunnel.

One long-lived paramiko connection per VM host, kept in a module-level registry
so it survives across the per-request `VMProvider` instances. A monitor thread
reconnects on drop. The autoscaler calls `ensure()` (via the provider) every
tick, so the tunnel also re-establishes after a gateway restart.

Single-process only: the forwards bind the VM's loopback, so two gateway
replicas can't both forward the same VM port. For HA, point `WORKER_REDIS_URL`
at a reachable Redis instead of using this.
"""
from __future__ import annotations

import logging
import select
import socket
import threading
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger("gateway.vm_tunnel")

SOCK_BUF = 8192
CONNECT_TIMEOUT_S = 20
MONITOR_INTERVAL_S = 15


@dataclass(frozen=True)
class Forward:
    """One reverse forward: bind `vm_port` on the VM's loopback and pump it to
    `local_host:local_port` (reachable from the gateway process)."""
    vm_port: int
    local_host: str
    local_port: int


@dataclass
class _Tunnel:
    host: str
    port: int
    user: str
    pkey_pem: str
    forwards: tuple[Forward, ...]
    client: object = None
    stop: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None


_LOCK = threading.Lock()
_TUNNELS: dict[str, _Tunnel] = {}  # keyed by VM host


def parse_host_port(url_or_hostport: str, default_port: int) -> tuple[str, int]:
    """Accept `redis://h:p`, `http://h:p`, or `h:p` → (host, port)."""
    s = (url_or_hostport or "").strip()
    if "://" in s:
        u = urlparse(s)
        return (u.hostname or "127.0.0.1", u.port or default_port)
    if ":" in s:
        h, _, p = s.rpartition(":")
        try:
            return (h or "127.0.0.1", int(p))
        except ValueError:
            return (s, default_port)
    return (s or "127.0.0.1", default_port)


def _pump(chan, target_host: str, target_port: int) -> None:
    """Bidirectionally copy bytes between a forwarded channel and a fresh socket
    to the local target. Runs in paramiko's per-connection handler thread."""
    try:
        sock = socket.create_connection((target_host, target_port), timeout=10)
    except Exception as e:
        logger.warning("vm-tunnel: cannot reach local target %s:%s: %s", target_host, target_port, e)
        try:
            chan.close()
        except Exception:
            pass
        return
    try:
        while True:
            r, _, _ = select.select([chan, sock], [], [], 60)
            if chan in r:
                data = chan.recv(SOCK_BUF)
                if not data:
                    break
                sock.sendall(data)
            if sock in r:
                data = sock.recv(SOCK_BUF)
                if not data:
                    break
                chan.sendall(data)
    except Exception:
        pass
    finally:
        for c in (chan, sock):
            try:
                c.close()
            except Exception:
                pass


def _run(client, cmd: str, timeout: float = 15.0) -> str:
    _, out, _ = client.exec_command(cmd, timeout=timeout)
    out.channel.recv_exit_status()
    return out.read().decode("utf-8", "replace").strip()


def _free_stale_listeners(client, ports: tuple[int, ...]) -> None:
    """Kill orphaned reverse-forward listeners on the VM holding our ports.

    A gateway reload (or an aborted attempt) drops the SSH connection but leaves
    its `sshd-session` forward listener bound on the VM with a dead pump — which
    both blocks a fresh `request_port_forward` ("TCP forwarding request denied")
    and answers worker connections with an immediate disconnect. We scope the
    kill to `sshd-session` owners so we never touch a real service on the port."""
    if not ports:
        return
    alt = "|".join(str(p) for p in ports)
    cmd = (
        f"pids=$(ss -ltnpH 2>/dev/null | grep -E '127.0.0.1:({alt}) ' | grep sshd "
        f"| grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); "
        f"for p in $pids; do kill -9 \"$p\" 2>/dev/null; done; true"
    )
    try:
        _run(client, cmd)
    except Exception as e:
        logger.warning("vm-tunnel %s: stale-listener cleanup failed: %s", getattr(client, "_host", "?"), e)


def _connect(t: _Tunnel) -> None:
    import paramiko

    from . import vm_probe

    pkey = vm_probe._load_pkey(t.pkey_pem)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=t.host, port=t.port, username=t.user, pkey=pkey,
        timeout=CONNECT_TIMEOUT_S, banner_timeout=CONNECT_TIMEOUT_S,
        auth_timeout=CONNECT_TIMEOUT_S, look_for_keys=False, allow_agent=False,
    )
    # Free any orphaned listeners before binding, so re-binding the same ports
    # after a reload/abort doesn't get denied.
    _free_stale_listeners(client, tuple(f.vm_port for f in t.forwards))
    transport = client.get_transport()
    transport.set_keepalive(30)
    # paramiko stores ONE `_tcp_handler` per transport (the last one set wins), so
    # a per-forward closure would route every forward to whichever was registered
    # last. Use a single handler that routes by the forwarded port — the 3rd arg
    # is (bind_addr, bind_port), i.e. the VM-side port the connection arrived on.
    port_map = {f.vm_port: (f.local_host, f.local_port) for f in t.forwards}

    def handler(chan, _src, dst):
        target = port_map.get(dst[1]) or next(iter(port_map.values()))
        # Called inline on the transport's receive thread — return immediately
        # and pump in a dedicated thread so we don't block channel dispatch.
        threading.Thread(target=_pump, args=(chan, target[0], target[1]), daemon=True).start()

    for fwd in t.forwards:
        transport.request_port_forward("127.0.0.1", fwd.vm_port, handler=handler)
    t.client = client
    logger.info(
        "vm-tunnel up: %s ← %s", t.host,
        ", ".join(f"127.0.0.1:{f.vm_port}→{f.local_host}:{f.local_port}" for f in t.forwards),
    )


def _safe_close(t: _Tunnel) -> None:
    try:
        if t.client is not None:
            t.client.close()
    except Exception:
        pass
    t.client = None


def _is_up(t: _Tunnel) -> bool:
    if t.client is None:
        return False
    try:
        tr = t.client.get_transport()
        return tr is not None and tr.is_active()
    except Exception:
        return False


def _forwards_alive(t: _Tunnel) -> bool:
    """Verify the reverse-forward listeners are actually still bound on the VM —
    not just that the SSH connection is up. A listener can be killed server-side
    (e.g. a competing tunnel, or our own prior cleanup) while the transport stays
    active; checking is_active() alone would never notice, so the worker would
    keep hitting a dead port. Cheap `ss` probe over the existing connection."""
    if t.client is None:
        return False
    try:
        alt = "|".join(str(f.vm_port) for f in t.forwards)
        n = _run(t.client, f"ss -ltnH 2>/dev/null | grep -Ec '127.0.0.1:({alt}) '", timeout=10)
        return int(n or 0) >= len(t.forwards)
    except Exception:
        return False


def _healthy(t: _Tunnel) -> bool:
    return _is_up(t) and _forwards_alive(t)


def _monitor(t: _Tunnel) -> None:
    while not t.stop.is_set():
        if not _healthy(t):
            logger.warning("vm-tunnel %s unhealthy — reconnecting", t.host)
            _safe_close(t)
            try:
                _connect(t)
            except Exception as e:
                logger.warning("vm-tunnel %s reconnect failed: %s", t.host, e)
        t.stop.wait(MONITOR_INTERVAL_S)


def ensure(host: str, port: int, user: str, pkey_pem: str, forwards: list[Forward]) -> None:
    """Idempotent: ensure a live reverse tunnel to `host` with `forwards`. Cheap
    no-op when already connected. Safe to call every autoscaler tick."""
    with _LOCK:
        t = _TUNNELS.get(host)
        if t is not None and _healthy(t):
            return
        if t is not None and t.client is not None:
            # Stale/half-dead tunnel — drop it before rebuilding so _connect
            # starts from a clean slate (and frees the orphaned VM listener).
            _safe_close(t)
        if t is None:
            t = _Tunnel(host=host, port=port, user=user, pkey_pem=pkey_pem, forwards=tuple(forwards))
            _TUNNELS[host] = t
        try:
            _connect(t)
        except Exception as e:
            logger.warning("vm-tunnel %s initial connect failed (monitor will retry): %s", host, e)
        if t.thread is None or not t.thread.is_alive():
            t.thread = threading.Thread(target=_monitor, args=(t,), daemon=True, name=f"vm-tunnel-{host}")
            t.thread.start()


def close(host: str) -> None:
    with _LOCK:
        t = _TUNNELS.pop(host, None)
    if t is not None:
        t.stop.set()
        _safe_close(t)
        logger.info("vm-tunnel closed: %s", host)
