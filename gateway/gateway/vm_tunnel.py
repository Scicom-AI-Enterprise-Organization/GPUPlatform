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

import atexit
import hashlib
import logging
import os
import select
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger("gateway.vm_tunnel")

SOCK_BUF = 8192
CONNECT_TIMEOUT_S = 20
MONITOR_INTERVAL_S = 15

# ---- native-OpenSSH forward tunnels (gateway → VM vLLM) --------------------
# The gateway→vLLM forward (proxy mode) runs as a real `autossh -L` subprocess
# rather than an in-process paramiko channel: OpenSSH does the byte relay in C
# (no per-token GIL pump), autossh auto-reconnects with ServerAlive keepalives,
# and it's a SEPARATE connection from the worker's reverse tunnel — so reverse-
# tunnel health churn can't tear down in-flight proxy requests. Falls back to
# plain `ssh` (the autoscaler re-spawns a dead one each tick) if autossh is absent.
_AUTOSSH = shutil.which("autossh")
_SSH = shutil.which("ssh") or "ssh"
_FWD_LOCK = threading.Lock()


@dataclass(frozen=True)
class Jump:
    """An SSH jump host (ProxyJump) in front of the VM — e.g. the TM NPU boxes
    behind ssh.tma01.gpu.tm.com.my. Key auth ONLY: the tunnels run OpenSSH in
    BatchMode, so neither hop can prompt for a password."""
    host: str
    port: int
    user: str
    pkey_pem: str


def _proxy_command_opts(jump: Optional[Jump]) -> list[str]:
    """`-o ProxyCommand=…` routing the connection through the jump host with its
    own identity file (ProxyJump can't take a per-hop -i on the CLI)."""
    if jump is None:
        return []
    jkey = _keyfile_for(f"jump:{jump.host}", jump.pkey_pem)
    proxy = (
        f"ssh -i {jkey} -W %h:%p -p {jump.port} "
        "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes "
        f"{jump.user}@{jump.host}"
    )
    return ["-o", f"ProxyCommand={proxy}"]


@dataclass
class _FwdProc:
    proc: "subprocess.Popen"
    local_port: int
    keyfile: str


_FWD_PROCS: dict = {}  # (host, port, vm_host, vm_port) -> _FwdProc
_REV_PROCS: dict = {}  # (host, port) -> _FwdProc  (the reverse `ssh -R` autossh subprocess)
# Keyed by (host, port), NOT host: two providers can share a host on different SSH
# ports (separate containers, e.g. tm :1024 + tm-2 :1023) and each needs its own
# reverse tunnel into its own container.


def _free_local_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _keyfile_for(host: str, pkey_pem: str) -> str:
    """Write the PEM to a stable 0600 temp file OpenSSH can read (`-i`). Reused
    across re-spawns; keyed by host+key so a rotated key gets a fresh file."""
    h = hashlib.sha256(f"{host}::{pkey_pem}".encode()).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), f"sgpu_vmkey_{h}")
    if not os.path.exists(path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(pkey_pem if pkey_pem.endswith("\n") else pkey_pem + "\n")
    return path


def _port_accepting(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _ssh_forward_cmd(local_port: int, vm_host: str, vm_port: int,
                     host: str, port: int, user: str, keyfile: str,
                     jump: Optional[Jump] = None) -> list[str]:
    opts = [
        "-N", "-T",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={CONNECT_TIMEOUT_S}",
        *_proxy_command_opts(jump),
        "-i", keyfile,
        "-p", str(port),
        "-L", f"127.0.0.1:{local_port}:{vm_host}:{vm_port}",
        f"{user}@{host}",
    ]
    if _AUTOSSH:
        return [_AUTOSSH, "-M", "0", *opts]  # -M 0: no monitor port; rely on ServerAlive*
    return [_SSH, *opts]


def _ssh_reverse_cmd(forwards: "tuple[Forward, ...]", host: str, port: int,
                     user: str, keyfile: str, jump: Optional[Jump] = None) -> list[str]:
    """Native-OpenSSH reverse tunnel: bind each `127.0.0.1:vm_port` on the VM and
    pump it back to the gateway's `local_host:local_port`. Same resilient flags as
    the forward path — autossh + ServerAlive keepalives + ExitOnForwardFailure, so
    a flaky bind self-heals instead of the paramiko close/reconnect port-release
    race that caused the 'TCP forwarding request denied' flapping."""
    opts = [
        "-N", "-T",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={CONNECT_TIMEOUT_S}",
        *_proxy_command_opts(jump),
        "-i", keyfile,
        "-p", str(port),
    ]
    for f in forwards:
        opts += ["-R", f"127.0.0.1:{f.vm_port}:{f.local_host}:{f.local_port}"]
    opts.append(f"{user}@{host}")
    if _AUTOSSH:
        return [_AUTOSSH, "-M", "0", *opts]  # -M 0: no monitor port; rely on ServerAlive*
    return [_SSH, *opts]


def _kill_stale_reverse(host: str, port: int, forwards: "tuple[Forward, ...]") -> None:
    """Kill a reverse autossh/ssh for `host:port` left over from a PRIOR gateway
    process. The subprocess is detached (start_new_session) so it survives a gateway
    restart and would keep the VM port bound → a fresh spawn gets 'forwarding request
    denied'. Matched by our keyfile marker + the reverse-forward of the gateway port +
    `-p {port}`, so it does NOT kill a SIBLING provider's tunnel on the same host but a
    different SSH port (e.g. tm :1024 vs tm-2 :1023)."""
    if not forwards:
        return
    marker = f"-R 127.0.0.1:{forwards[0].vm_port}:"
    port_marker = f" -p {port} "
    try:
        out = subprocess.run(["pgrep", "-af", "sgpu_vmkey"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return
    import signal
    for line in out.splitlines():
        pid_str, _, cmd = line.partition(" ")
        if host in cmd and marker in cmd and port_marker in cmd:
            try:
                os.kill(int(pid_str), signal.SIGKILL)
                logger.info("vm-tunnel reverse: killed stale autossh pid=%s for %s:%s", pid_str, host, port)
            except (ProcessLookupError, ValueError, PermissionError):
                pass


def _kill_fwd(fp: "_FwdProc") -> None:
    try:
        fp.proc.terminate()
        try:
            fp.proc.wait(timeout=3)
        except Exception:
            fp.proc.kill()
    except Exception:  # noqa: BLE001 — already-dead proc is fine; log the odd case
        logger.debug("vm-tunnel: forward-tunnel kill failed (pid=%s)",
                     getattr(fp.proc, "pid", "?"), exc_info=True)


@atexit.register
def _cleanup_forwards() -> None:
    for fp in list(_FWD_PROCS.values()):
        _kill_fwd(fp)
    for fp in list(_REV_PROCS.values()):
        _kill_fwd(fp)


@dataclass(frozen=True)
class Forward:
    """One reverse forward: bind `vm_port` on the VM's loopback and pump it to
    `local_host:local_port` (reachable from the gateway process)."""
    vm_port: int
    local_host: str
    local_port: int


@dataclass
class LocalForward:
    """One `ssh -L`-style forward: bind a local port on the GATEWAY and pump each
    accepted connection over a direct-tcpip channel to `vm_host:vm_port` on the VM.
    Lets the gateway reach a service bound to the VM's loopback (e.g. a proxy
    endpoint's vLLM on 127.0.0.1:18001). `local_port` is assigned when bound (0 →
    auto-pick a free port) and then reused across SSH reconnects."""
    vm_port: int
    vm_host: str = "127.0.0.1"
    local_port: int = 0
    _server: object = None          # bound listening socket (lives across reconnects)
    _thread: Optional[threading.Thread] = None


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
    # ssh -L forwards (gateway → VM service). Mutated under _LOCK by ensure_forward.
    local_forwards: list = field(default_factory=list)


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


def _relay(chan, sock) -> None:
    """Bidirectionally copy bytes between a paramiko channel and a socket until
    either side closes; closes both on exit. Shared by the reverse pump (which
    dials the local target) and the forward listener (which already has the
    accepted client socket)."""
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


def _pump(chan, target_host: str, target_port: int) -> None:
    """Reverse forward: a forwarded channel arrived from the VM — dial a fresh
    socket to the local target and relay. Runs in paramiko's per-connection
    handler thread."""
    try:
        sock = socket.create_connection((target_host, target_port), timeout=10)
    except Exception as e:
        logger.warning("vm-tunnel: cannot reach local target %s:%s: %s", target_host, target_port, e)
        try:
            chan.close()
        except Exception:
            pass
        return
    _relay(chan, sock)


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
    if not t.forwards:
        return True  # forward-only tunnel (ssh -L); nothing reverse-bound to verify
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


def ensure(host: str, port: int, user: str, pkey_pem: str, forwards: list[Forward],
           jump: Optional[Jump] = None) -> None:
    """Idempotent: ensure a live reverse tunnel to `host` with `forwards`. Cheap
    no-op when the autossh subprocess is still alive. Safe to call every autoscaler
    tick. Native OpenSSH (autossh `ssh -R`), NOT paramiko: OpenSSH does the relay in
    C, autossh keeps it alive + reconnects on drop, and there's no in-process
    health-check close/reconnect loop racing the VM's port release (which caused the
    'TCP forwarding request denied' flapping). Mirrors `ensure_forward`."""
    fwds = tuple(forwards)
    key = (host, int(port))
    with _LOCK:
        rp = _REV_PROCS.get(key)
        if rp is not None and rp.proc.poll() is None:
            return  # subprocess alive → reverse tunnel is up (autossh self-heals drops)
        if rp is not None:
            _kill_fwd(rp)  # ours, but dead/exited → clean up before re-spawning
        # An autossh from a PRIOR gateway process is detached and still holding the
        # VM ports → kill it so our fresh `-R` bind isn't denied.
        _kill_stale_reverse(host, int(port), fwds)
        keyfile = _keyfile_for(host, pkey_pem)
        cmd = _ssh_reverse_cmd(fwds, host, port, user, keyfile, jump=jump)
        env = {**os.environ, "AUTOSSH_GATETIME": "0"}  # keep retrying even if the 1st bind races a release
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, env=env,
        )
        _REV_PROCS[key] = _FwdProc(proc=proc, local_port=0, keyfile=keyfile)
        logger.info(
            "vm-tunnel reverse (%s) up: %s ← %s", "autossh" if _AUTOSSH else "ssh", host,
            ", ".join(f"127.0.0.1:{f.vm_port}→{f.local_host}:{f.local_port}" for f in fwds),
        )


def _forward_listener(t: _Tunnel, fwd: LocalForward) -> None:
    """Accept on the local bind socket; for each connection open a direct-tcpip
    channel to the VM target over the CURRENT SSH transport and relay. Survives
    reconnects (re-reads t.client each accept). Exits when the tunnel is stopped."""
    srv = fwd._server
    srv.settimeout(1.0)
    while not t.stop.is_set():
        try:
            client_sock, _peer = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        transport = None
        try:
            transport = t.client.get_transport() if t.client is not None else None
        except Exception:
            transport = None
        if transport is None or not transport.is_active():
            logger.warning("vm-tunnel forward %s: SSH transport down, dropping connection", t.host)
            try:
                client_sock.close()
            except Exception:
                pass
            continue
        try:
            chan = transport.open_channel(
                "direct-tcpip", (fwd.vm_host, fwd.vm_port), client_sock.getpeername(),
            )
        except Exception as e:
            logger.warning("vm-tunnel forward %s → %s:%s open_channel failed: %s",
                           t.host, fwd.vm_host, fwd.vm_port, e)
            try:
                client_sock.close()
            except Exception:
                pass
            continue
        threading.Thread(target=_relay, args=(chan, client_sock), daemon=True).start()
    try:
        srv.close()
    except Exception:
        pass


def ensure_forward(host: str, port: int, user: str, pkey_pem: str,
                   vm_port: int, vm_host: str = "127.0.0.1",
                   jump: Optional[Jump] = None) -> int:
    """Idempotent `ssh -L`: ensure a local listener on the gateway that forwards
    to `vm_host:vm_port` on the VM, via a native `autossh -L` subprocess. Returns
    the local port. Reuses a live subprocess; re-spawns a dead one (so the
    autoscaler tick heals it after a gateway restart). Safe to call every tick.

    Native OpenSSH (not paramiko): the byte relay is in C, autossh keeps the
    connection alive + reconnects on drop, and it's a dedicated connection per
    forward — decoupled from the worker's reverse tunnel, so reverse-tunnel
    health churn can't kill an in-flight proxy request."""
    key = (host, int(port), vm_host, int(vm_port))
    with _FWD_LOCK:
        fp = _FWD_PROCS.get(key)
        if fp is not None and fp.proc.poll() is None:
            return fp.local_port  # subprocess alive → forward is up
        if fp is not None:
            _kill_fwd(fp)  # dead/exited → clean up before re-spawning
        keyfile = _keyfile_for(host, pkey_pem)
        local_port = _free_local_port()
        cmd = _ssh_forward_cmd(local_port, vm_host, vm_port, host, port, user, keyfile, jump=jump)
        env = {**os.environ, "AUTOSSH_GATETIME": "0"}  # keep retrying even if the 1st connect is slow
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, env=env,
        )
        if not _port_accepting(local_port, timeout=15.0):
            logger.warning("vm-tunnel forward %s:%d → %s:%d not accepting after spawn (will retry next tick)",
                           host, port, vm_host, vm_port)
        _FWD_PROCS[key] = _FwdProc(proc=proc, local_port=local_port, keyfile=keyfile)
        logger.info("vm-tunnel forward (%s) up: 127.0.0.1:%d → %s:%d on %s",
                    "autossh" if _AUTOSSH else "ssh", local_port, vm_host, vm_port, host)
        return local_port


def close_forwards(host: str, vm_port: Optional[int] = None) -> None:
    """Kill forward subprocesses for `host` (called on teardown/terminate). With
    `vm_port`, kills only that forward — a host can serve several proxy endpoints."""
    with _FWD_LOCK:
        for key in [k for k in _FWD_PROCS if k[0] == host and (vm_port is None or k[3] == int(vm_port))]:
            _kill_fwd(_FWD_PROCS.pop(key))


def close(host: str) -> None:
    with _LOCK:
        # _REV_PROCS is keyed by (host, port) — close every tunnel for this host.
        rev = [(k, v) for k, v in _REV_PROCS.items() if k[0] == host]
        for k, _ in rev:
            _REV_PROCS.pop(k, None)
        t = _TUNNELS.pop(host, None)
    for k, rp in rev:
        _kill_fwd(rp)
        logger.info("vm-tunnel reverse closed: %s:%s", k[0], k[1])
    if t is not None:  # legacy paramiko tunnel (pre-autossh); harmless if unused
        t.stop.set()
        for fwd in t.local_forwards:
            try:
                if fwd._server is not None:
                    fwd._server.close()
            except Exception:
                pass
        _safe_close(t)
        logger.info("vm-tunnel closed: %s", host)
