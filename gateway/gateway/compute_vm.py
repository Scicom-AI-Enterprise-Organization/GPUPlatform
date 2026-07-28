"""VM-backed Compute sessions — a uv venv + JupyterLab on a registered bare-metal box.

The cloud path (`compute.py`) asks RunPod/PI for a fresh pod and hands the user
its public SSH coords + RunPod's own jupyter proxy domain. A **VM** session has
neither: the box already exists (a `kind=vm` Provider row), it's usually behind a
firewall / jump host, and nothing on it is publicly reachable. So a VM session is
deliberately much smaller:

1. SSH in, `uv venv` + `uv pip install jupyterlab` under `~/.sgpu/compute/{pod_id}`
   (idempotent — a re-provision of the same id reuses the venv).
2. Launch `jupyter lab` bound to the VM's **loopback** on a free port, with
   `--ServerApp.base_url` set to the gateway proxy path so every URL Jupyter
   generates already points back through us (no HTML/JSON rewriting anywhere).
3. Open the gateway→VM `ssh -L` forward (`vm_tunnel.ensure_forward`, the same
   autossh machinery proxy-mode endpoints use) and serve the whole thing at
   `/compute/jupyter/{pod_id}/{proxy_token}/…` (see `compute_proxy.py`).

`CUDA_VISIBLE_DEVICES` is pinned at launch from the optional `visible_devices`
field ("0,1") — the box is shared, so a session that doesn't pin gets every GPU.

⚠ Tunnels run OpenSSH in BatchMode → the provider (and its jump host, if any)
needs KEY auth. A password-only VM provider is rejected at create time with a
clear message rather than half-provisioning something unreachable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
from typing import Any, Optional

from . import vm_probe

logger = logging.getLogger("gateway.compute_vm")

# Everything a session owns lives under one per-pod dir so teardown is a single
# rm and two sessions on one box never share a venv.
REMOTE_ROOT = "~/.sgpu/compute"
# jupyterlab MUST stay first — _launch_script swaps it for `jupyterlab=={ver}`
# when the session pins a version.
BASE_PACKAGES = ("jupyterlab", "ipywidgets")

# Install can genuinely take minutes on a cold uv cache (a managed CPython
# download + the jupyterlab wheel set).
INSTALL_TIMEOUT_S = 900
SHORT_TIMEOUT_S = 30
# How long we wait for `GET {base}api/status` to answer 200 through the tunnel.
READY_TIMEOUT_S = 300
READY_POLL_S = 3.0


def validate_visible_devices(value: Optional[str]) -> Optional[str]:
    """Normalize the optional CUDA_VISIBLE_DEVICES text input ("0,1" / "" / None).

    Returns None for "all GPUs" (we then don't set the variable at all) or the
    cleaned list. Raises ValueError on anything that isn't comma-separated
    indices — this string is interpolated into a shell command line."""
    s = (value or "").strip()
    if not s:
        return None
    # Strip space AROUND each token, never inside one: a plain `.replace(" ","")`
    # turns the space-separated "0 1" into "01", which CUDA reads as device 1 —
    # a silently wrong GPU instead of an error.
    seen: list[str] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok.isdigit():
            raise ValueError(
                "visible_devices must be comma-separated GPU indices (e.g. '0' or '0,1')"
            )
        # Dedupe while preserving order — CUDA_VISIBLE_DEVICES=0,0 is a footgun.
        if tok not in seen:
            seen.append(tok)
    return ",".join(seen)


def validate_jupyter_version(value: Optional[str]) -> Optional[str]:
    """Normalize the optional JupyterLab version ("4.2.5" / "" / None).

    A bare PEP 440 version only — NOT a specifier. `>=4,<5` would need `<`/`>`
    on a remote command line, and the whole point of pinning here is to say
    exactly which JupyterLab a session gets. Blank = latest. Raises ValueError
    on anything else; the value is interpolated into `uv pip install`."""
    s = (value or "").strip().lstrip("=")
    if not s:
        return None
    # 4 / 4.2 / 4.2.5 / 4.2.5rc1 / 4.3.0.dev0 / 4.* — digits-first, then the
    # usual PEP 440 alphanumerics. Deliberately excludes shell metacharacters.
    if not re.match(r"^\d+(\.(\d+|\*))*([._-]?[A-Za-z0-9]+)*$", s):
        raise ValueError(
            "jupyter_version must be a plain version like '4.2.5' (blank = latest)"
        )
    return s


def base_path(pod_id: str, proxy_token: str) -> str:
    """The gateway path this session is served at == Jupyter's `base_url`.

    Jupyter requires a leading AND trailing slash; keeping the two identical is
    what lets us proxy without rewriting a single response body."""
    return f"/compute/jupyter/{pod_id}/{proxy_token}/"


# ---------- provider connection ----------------------------------------


async def vm_conn_for_provider(session, provider_id: str) -> dict[str, Any]:
    """Decrypted SSH kwargs (host/port/user/key/password + jump hops) for a
    `kind=vm` Provider row, in the shape every vm_probe function takes."""
    from .db import Provider
    # Lazy: providers_api imports compute at module level (circular otherwise).
    from .providers_api import _vm_conn_from_cfg

    prov = await session.get(Provider, provider_id)
    if prov is None:
        raise RuntimeError(f"provider {provider_id} no longer exists")
    if prov.kind != "vm":
        raise RuntimeError(f"provider {provider_id} is kind={prov.kind}, expected vm")
    return _vm_conn_from_cfg(prov.config or {})


def tunnel_jump(conn: dict[str, Any]):
    """`vm_tunnel.Jump` for this connection, or None. Key-only: autossh runs in
    BatchMode and can't answer a password prompt on either hop."""
    if not conn.get("jump_host"):
        return None
    from . import vm_tunnel

    if not conn.get("jump_private_key"):
        raise RuntimeError(
            "jump host has no private key — the JupyterLab tunnel needs key auth on both hops"
        )
    return vm_tunnel.Jump(
        host=conn["jump_host"],
        port=int(conn.get("jump_port") or 22),
        user=conn.get("jump_user") or "root",
        pkey_pem=conn["jump_private_key"],
    )


def require_tunnel_key(conn: dict[str, Any]) -> str:
    key = conn.get("private_key")
    if not key:
        raise RuntimeError(
            "VM provider has password-only credentials — the JupyterLab tunnel needs "
            "key auth; add a private key to the provider"
        )
    return key


def ensure_forward_sync(conn: dict[str, Any], vm_port: int) -> int:
    """Idempotent gateway→VM `ssh -L` to the session's Jupyter port. Returns the
    gateway-local port. Cheap on the hot path (live-subprocess check), so the
    proxy can call it per request — that's also what heals the forward after a
    gateway restart without any background loop."""
    from . import vm_tunnel

    return vm_tunnel.ensure_forward(
        conn["host"], int(conn["port"]), conn["user"], require_tunnel_key(conn),
        int(vm_port), "127.0.0.1", tunnel_jump(conn),
    )


def close_forward(conn: dict[str, Any], vm_port: Optional[int]) -> None:
    """Kill only THIS session's forward.

    ⚠ Never call `vm_tunnel.close_forwards(host)` without a port here: that
    reaps every forward on the host, which on a shared box (tm-2 runs
    proxy-mode serverless endpoints) would tear down live inference tunnels.
    A session with no port recorded yet simply has no forward to close."""
    if not vm_port:
        return
    from . import vm_tunnel

    try:
        vm_tunnel.close_forwards(conn["host"], int(vm_port))
    except Exception:  # noqa: BLE001 — teardown best-effort
        logger.debug("compute-vm: forward cleanup failed", exc_info=True)


# ---------- SSH helpers -------------------------------------------------


def _run(client, cmd: str, timeout: float = SHORT_TIMEOUT_S) -> str:
    _, stdout, _ = client.exec_command(cmd, timeout=timeout)
    stdout.channel.recv_exit_status()
    return stdout.read().decode(errors="replace")


def _run_full(client, cmd: str, timeout: float = SHORT_TIMEOUT_S) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    rc = stdout.channel.recv_exit_status()
    return rc, stdout.read().decode(errors="replace"), stderr.read().decode(errors="replace")


def _shellrc(venv: str, pod_id: str) -> str:
    """The rc file JupyterLab's terminal starts with.

    Terminado's default shell is `bash -l`, which sources the box's own
    profile/rc — on a GPU box that usually PREPENDS a conda to PATH and would
    shadow our venv, so a `pip install` in the notebook terminal would land in
    the machine's default python. We source the user's rc FIRST and re-assert
    the venv afterwards, so the terminal, the kernels and `uv` all agree on one
    interpreter and the box's python is never touched."""
    return (
        "# Managed by Serverless-GPU compute session — do not edit.\n"
        "[ -f ~/.bashrc ] && . ~/.bashrc\n"
        f'export VIRTUAL_ENV="{venv}"\n'
        f'export PATH="{venv}/bin:$HOME/.local/bin:$PATH"\n'
        "unset PYTHONHOME\n"
        f'export PS1="(sgpu:{pod_id}) $PS1"\n'
    )


def _launch_script(
    *, rundir: str, workdir: str, venv: str, base: str,
    jupyter_token: str, visible_devices: Optional[str], extra_packages: list[str],
    jupyter_version: Optional[str] = None,
) -> str:
    """The one shell script that bootstraps the venv and starts the server.

    **uv only** — never the box's python. `python3 -m venv` + system `pip` would
    write into whatever interpreter the box happens to ship (often a conda base
    shared by other users), which is exactly what we're avoiding: uv is
    installed on demand if missing, and a box where that fails is an error, not
    a reason to fall back. Both the venv and the install are skipped when
    already present, so a retry is cheap.

    The free port is chosen with the VENV's python for the same reason — by then
    it exists, so nothing here ever invokes the system interpreter."""
    jlab = f"jupyterlab=={jupyter_version}" if jupyter_version else "jupyterlab"
    pkgs = " ".join(
        shlex.quote(p) for p in [jlab, *BASE_PACKAGES[1:], *extra_packages]
    )
    log = f"{rundir}/jupyter.log"
    pidf = f"{rundir}/jupyter.pid"
    rcfile = f"{rundir}/shellrc"
    vpy = f"{venv}/bin/python"
    q = shlex.quote
    # Give the notebook terminal the venv too (see _shellrc).
    terminado = json.dumps({"shell_command": ["/bin/bash", "--rcfile", rcfile, "-i"]})
    port_py = (
        "import socket;s=socket.socket();s.bind(('127.0.0.1',0));"
        "print(s.getsockname()[1]);s.close()"
    )
    return (
        "set -e; "
        # uv installs to ~/.local/bin, which a non-login shell misses.
        'export PATH="$HOME/.local/bin:$PATH"; '
        f"mkdir -p {q(rundir)} {q(workdir)}; "
        "if ! command -v uv >/dev/null 2>&1; then "
        "  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true; "
        '  export PATH="$HOME/.local/bin:$PATH"; '
        "fi; "
        "command -v uv >/dev/null 2>&1 || "
        '  { echo "uv is required on the machine and could not be installed automatically" >&2; exit 3; }; '
        # --seed: a plain `uv venv` has NO pip, so `pip install` in the notebook
        # terminal would fall through PATH to the box's /usr/local/bin/pip and
        # write into the machine's default python — the exact thing we're
        # avoiding. Seeding also makes the `%pip` cell magic work.
        f"if [ ! -x {q(vpy)} ]; then uv venv --seed --python 3.11 {q(venv)}; fi; "
        f"if [ ! -x {q(venv + '/bin/jupyter')} ]; then "
        f"  uv pip install --python {q(vpy)} {pkgs}; "
        "fi; "
        f"cat > {q(rcfile)} <<'SGPU_RC'\n{_shellrc(venv, rundir.rsplit('/', 1)[-1])}SGPU_RC\n"
        f"PORT=$({q(vpy)} -c {q(port_py)}); "
        f"cd {q(workdir)}; "
        # VIRTUAL_ENV + PATH are exported into the SERVER's env so kernels and
        # terminals inherit the venv without any activation step.
        f"export VIRTUAL_ENV={q(venv)}; export PATH={q(venv + '/bin')}:\"$PATH\"; "
        # setsid → own session/pgid, so teardown kills the whole tree (jupyter
        # spawns kernels) without touching anything else on a shared box.
        f"{('CUDA_VISIBLE_DEVICES=' + q(visible_devices) + ' ') if visible_devices else ''}"
        f"setsid nohup {q(venv + '/bin/jupyter')} lab "
        '  --no-browser --ip=127.0.0.1 --port="$PORT" '
        # ⚠ Required: most GPU boxes SSH in as root, and jupyter REFUSES to run
        # as root without this — it logs "Running as root is not recommended"
        # and exits, which looks exactly like a slow start until the ready poll
        # finally times out. Harmless for non-root users.
        "  --allow-root "
        f"  --ServerApp.base_url={q(base)} "
        f"  --ServerApp.root_dir={q(workdir)} "
        f"  --ServerApp.allow_remote_access=True "
        f"  --ServerApp.open_browser=False "
        f"  --ServerApp.terminado_settings={q(terminado)} "
        f"  --IdentityProvider.token={q(jupyter_token)} "
        f"  > {q(log)} 2>&1 & "
        f"echo $! > {q(pidf)}; "
        'echo PORT="$PORT"; '
        # Fail FAST on a server that dies at startup (bad flag, port clash, a
        # broken extension in the box's global jupyter config) rather than
        # burning the whole ready-poll window on a process that's already gone.
        f"sleep 4; echo PID=$(cat {q(pidf)}); "
        'if kill -0 "$(cat ' + q(pidf) + ')" 2>/dev/null; then echo ALIVE=1; else echo ALIVE=0; fi; '
        f"echo ---LOG---; tail -n 40 {q(log)} 2>/dev/null"
    )


def provision_sync(
    conn: dict[str, Any], *, pod_id: str, workdir: Optional[str],
    visible_devices: Optional[str], base: str, jupyter_token: str,
    extra_packages: Optional[list[str]] = None,
    jupyter_version: Optional[str] = None,
) -> dict[str, Any]:
    """SSH in, bootstrap the venv, start Jupyter. Returns the launch facts
    (port/pid/paths). Runs in a thread — paramiko is sync."""
    client, jump = vm_probe._connect(**conn)
    try:
        home = _run(client, "echo $HOME").strip() or f"/home/{conn.get('user') or 'root'}"
        root = REMOTE_ROOT.replace("~", home)
        rundir = f"{root}/{pod_id}"
        wd = (workdir or "").strip() or f"{rundir}/work"
        if wd.startswith("~"):
            wd = home + wd[1:]
        venv = f"{rundir}/venv"
        script = _launch_script(
            rundir=rundir, workdir=wd, venv=venv, base=base,
            jupyter_token=jupyter_token, visible_devices=visible_devices,
            extra_packages=list(extra_packages or []),
            jupyter_version=jupyter_version,
        )
        rc, out, err = _run_full(client, f"bash -lc {shlex.quote(script)}", timeout=INSTALL_TIMEOUT_S)
        if rc != 0:
            tail = (err or out or "").strip()[-1500:]
            raise RuntimeError(f"jupyter launch failed (rc={rc}): {tail}")
        head, _, log_tail = out.partition("---LOG---")
        pid = 0
        port = 0
        alive = True
        for line in head.splitlines():
            if line.startswith("PID="):
                try:
                    pid = int(line[4:].strip())
                except ValueError:
                    pid = 0
            elif line.startswith("PORT="):
                try:
                    port = int(line[5:].strip())
                except ValueError:
                    port = 0
            elif line.startswith("ALIVE="):
                alive = line[6:].strip() == "1"
        if not alive:
            raise RuntimeError(
                "JupyterLab exited immediately after launch:\n" + log_tail.strip()[-1500:]
            )
        if not port:
            raise RuntimeError(f"could not read the chosen port from the VM: {head.strip()[-500:]}")
        return {
            "vm_port": port, "vm_pid": pid, "workdir": wd,
            "rundir": rundir, "venv": venv, "log": f"{rundir}/jupyter.log",
        }
    finally:
        vm_probe._close_quiet(client, jump)


def read_log_sync(conn: dict[str, Any], pod_id: str, tail: int = 60) -> str:
    """Tail the session's jupyter.log — the only place a bootstrap failure
    (missing uv/python, pip resolution error, port clash) is visible."""
    try:
        client, jump = vm_probe._connect(**conn)
    except Exception:  # noqa: BLE001 — box unreachable → nothing to show
        return ""
    try:
        home = _run(client, "echo $HOME").strip() or "/root"
        path = f"{REMOTE_ROOT.replace('~', home)}/{pod_id}/jupyter.log"
        return _run(client, f"tail -n {int(tail)} {shlex.quote(path)} 2>/dev/null")
    except Exception:  # noqa: BLE001
        return ""
    finally:
        vm_probe._close_quiet(client, jump)


def teardown_sync(conn: dict[str, Any], pod_id: str, purge: bool = False) -> None:
    """Kill the session's process group by its recorded pid. NEVER a pkill on
    `jupyter` — the box is shared and may run someone else's server.

    `purge` also reclaims the uv venv (a jupyterlab env is ~300 MB, and these
    accumulate one per session on a shared box) plus the log/rc scratch files.
    ⚠ It deliberately does NOT touch `{rundir}/work` — that's the DEFAULT
    workdir, so `rm -rf {rundir}` would delete the user's notebooks along with
    the environment. Purge is off for failure/abort paths so a retry can reuse
    the venv, and on for an explicit user delete (which says so in the dialog).
    """
    client, jump = vm_probe._connect(**conn)
    try:
        home = _run(client, "echo $HOME").strip() or "/root"
        rundir = f"{REMOTE_ROOT.replace('~', home)}/{pod_id}"
        pidf = f"{rundir}/jupyter.pid"
        q = shlex.quote
        extra = (
            f" rm -rf {q(rundir + '/venv')} {q(rundir + '/shellrc')} {q(rundir + '/jupyter.log')};"
            # rmdir (not rm -rf) so the per-pod dir disappears only when nothing
            # is left in it — a default workdir with notebooks in it survives,
            # an empty one doesn't linger.
            f" rmdir {q(rundir + '/work')} 2>/dev/null; rmdir {q(rundir)} 2>/dev/null;"
            if purge else ""
        )
        script = (
            f'P="$(cat {q(pidf)} 2>/dev/null)"; '
            'if [ -n "$P" ]; then '
            '  kill -TERM -- -"$P" 2>/dev/null || kill -TERM "$P" 2>/dev/null || true; '
            '  sleep 2; kill -KILL -- -"$P" 2>/dev/null || kill -KILL "$P" 2>/dev/null || true; '
            "fi; "
            # Fallback for a lost pidfile (a crashed provisioner, or someone
            # cleaning the dir by hand): match on this pod's OWN directory,
            # which carries its unique id — pod-scoped like the serverless
            # provider's `pkill -f worker-{mid}.json`, NEVER a bare
            # `pkill jupyter` on a box that may run someone else's server.
            f"pkill -9 -f {q(rundir + '/')} 2>/dev/null || true; "
            f"rm -f {q(pidf)};{extra} true"
        )
        _run(client, f"bash -lc {shlex.quote(script)}", timeout=60)
    finally:
        vm_probe._close_quiet(client, jump)


# ---------- readiness ---------------------------------------------------


async def wait_ready(local_port: int, base: str, timeout_s: float = READY_TIMEOUT_S) -> bool:
    """Poll Jupyter's `{base}api/status` through the forward tunnel. Proves the
    whole path (tunnel → jupyter → base_url) works before we call the session
    running, so the user never clicks a dead link."""
    import httpx

    url = f"http://127.0.0.1:{int(local_port)}{base}api/status"
    deadline = asyncio.get_running_loop().time() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as cli:
        while asyncio.get_running_loop().time() < deadline:
            try:
                r = await cli.get(url)
                # 200 = up; 403 would mean it's up but rejecting us (still "ready"
                # enough — the browser carries the token).
                if r.status_code in (200, 403):
                    return True
            except Exception:  # noqa: BLE001 — not up yet
                pass
            await asyncio.sleep(READY_POLL_S)
    return False
