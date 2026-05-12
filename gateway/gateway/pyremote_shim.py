"""Reconnect-per-command shim for benchmaq's pyremote dependency.

Some managed GPU clouds (e.g. TM's `ssh.tma01.gpu.tm.com.my`) sit a Go-based
SSH proxy in front of the real VM. Those proxies often enforce **one exec
channel per TCP connection** regardless of the backend sshd's `MaxSessions`
setting. pyremote's design is to open one paramiko.SSHClient and fire many
sequential `exec_command()` calls through it — perfectly fine against a
normal OpenSSH server, but the second `exec_command()` immediately raises
`SSHException: Channel closed` against these proxies.

This shim monkey-patches `pyremote.RemoteExecutor._run_command` to open a
fresh SSHClient for every call. ~1s extra per command (the auth handshake),
which adds up over a benchmark run, but it's correct and zero-API-change for
benchmaq.

The big payload exec at the bottom of `RemoteExecutor.execute()` is NOT
patched — that's the single long-lived `exec_command()` that actually runs
the benchmark, and it works fine because it's the only channel on its
connection.

Idempotent: calling `install()` twice is a no-op.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("gateway.pyremote_shim")

_INSTALLED = False


def install() -> None:
    """Patch pyremote in place. Safe to call at gateway startup."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        import paramiko
        from pyremote import RemoteExecutor  # type: ignore
    except Exception:
        logger.warning("pyremote not importable — shim not installed")
        return

    original = RemoteExecutor._run_command

    def _patched_run_command(self: Any, cmd: str, timeout: int | None = None, stream: bool = False):
        """Open a fresh SSHClient for this exec, then close it. Falls back to
        the original implementation if anything in the swap fails."""
        cfg = self.ssh_config
        kwargs = {
            "hostname": cfg.host,
            "port": cfg.port,
            "username": cfg.username,
            "timeout": cfg.timeout,
            "look_for_keys": False,
            "allow_agent": False,
        }
        if getattr(cfg, "key_filename", None):
            kwargs["key_filename"] = os.path.expanduser(cfg.key_filename)
            if getattr(cfg, "key_password", None):
                kwargs["passphrase"] = cfg.key_password
        elif getattr(cfg, "password", None):
            kwargs["password"] = cfg.password

        fresh = paramiko.SSHClient()
        fresh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            fresh.connect(**kwargs)
        except Exception as e:
            logger.warning("shim: fresh connect failed (%s) — falling back to original path", e)
            return original(self, cmd, timeout=timeout, stream=stream)

        original_client = self._client
        self._client = fresh
        try:
            return original(self, cmd, timeout=timeout, stream=stream)
        finally:
            self._client = original_client
            try:
                fresh.close()
            except Exception:
                pass

    RemoteExecutor._run_command = _patched_run_command  # type: ignore[assignment]

    # Replace _install_dependencies wholesale so we get full visibility into
    # the install output regardless of pyremote's stream behaviour. The
    # original captures stdout+stderr into separate vars but only puts
    # `stderr` into the RemoteImportError — uv writes its real error to
    # stdout, so the error message comes back empty. Here we capture both,
    # echo both to the subprocess stdout/stderr (so the gateway can stream
    # them into bench logs), and put both into the exception on failure.
    import sys as _sys
    import hashlib as _hashlib

    def _patched_install_deps(self):
        if not self.dependencies:
            return
        deps_key = ",".join(sorted(self.dependencies)) + str(self.venv) + str(self.uv)
        deps_hash = _hashlib.md5(deps_key.encode()).hexdigest()[:8]
        if deps_hash in self._deps_installed:
            return
        deps_str = " ".join(f'"{dep}"' for dep in self.dependencies)
        if self.uv:
            cmd = (
                "source $HOME/.local/bin/env 2>/dev/null || true\n"
                f"source {self.uv.activate_path}\n"
                f"uv pip install {deps_str}"
            )
        elif self.venv:
            cmd = f"{self.venv.pip_path} install {deps_str}"
        else:
            cmd = f"{self._python_path} -m pip install {deps_str}"

        print(f"[shim] installing {self.dependencies}", flush=True)
        exit_status, stdout_data, stderr_data = self._run_command(cmd, timeout=300, stream=False)
        if stdout_data:
            print(stdout_data, flush=True)
        if stderr_data:
            print(stderr_data, file=_sys.stderr, flush=True)
        print(f"[shim] install rc={exit_status}", flush=True)

        if exit_status != 0:
            from pyremote import RemoteImportError  # type: ignore
            tail = ((stderr_data or "") + "\n" + (stdout_data or "")).strip()
            tail = tail[-1500:] if tail else "(no output from uv)"
            raise RemoteImportError(
                f"Failed to install dependencies {self.dependencies} (rc={exit_status}):\n{tail}"
            )
        self._deps_installed.add(deps_hash)

    RemoteExecutor._install_dependencies = _patched_install_deps  # type: ignore[assignment]
    _INSTALLED = True
    logger.info("pyremote reconnect-per-command shim installed")
