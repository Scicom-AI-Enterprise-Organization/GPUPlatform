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
            # `uv -v` forces verbose output (resolve + per-wheel download
            # progress). Without it uv suppresses progress when stdout isn't
            # a TTY, which is exactly the case under SSH/subprocess capture.
            cmd = (
                "source $HOME/.local/bin/env 2>/dev/null || true\n"
                f"source {self.uv.activate_path}\n"
                f"uv -v pip install {deps_str}"
            )
        elif self.venv:
            cmd = f"{self.venv.pip_path} install -v {deps_str}"
        else:
            cmd = f"{self._python_path} -m pip install -v {deps_str}"

        print(f"[shim] installing {self.dependencies}", flush=True)
        # stream=True makes pyremote's _run_command print lines live; the
        # final dump below is a safety net in case the live stream path
        # doesn't bubble up under some SSH/proxy edge case.
        exit_status, stdout_data, stderr_data = self._run_command(cmd, timeout=600, stream=True)
        if stdout_data:
            print("[shim] --- uv stdout (captured) ---", flush=True)
            print(stdout_data, flush=True)
        if stderr_data:
            print("[shim] --- uv stderr (captured) ---", file=_sys.stderr, flush=True)
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

    # Patch benchmaq's `_ssh_run_stream` to use `bash -c <quoted-script>`
    # instead of `bash -s` + stdin piping + shutdown_write(). Some SSH proxies
    # (e.g. Go-based proxies fronting managed GPU VMs) don't tolerate stdin
    # streaming + shutdown_write — they close the channel immediately, which
    # surfaces as `recv_exit_status() == -1` and zero output. `bash -c` is a
    # single exec with the whole script as one argv — works against the same
    # proxies that pyremote's exec_command worked against (once we
    # reconnect-per-call).
    try:
        from benchmaq import runner as _benchmaq_runner  # type: ignore
        import re as _re
        import shlex as _shlex

        # Strip ANSI/VT escape sequences (colours, cursor moves, erase-line,
        # etc.) before printing — the bench log viewer is a plain <pre> so
        # raw escapes show up as garbage like "[2K[1A[37m".
        _ANSI_RE = _re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

        _orig_ssh_run_stream = _benchmaq_runner._ssh_run_stream

        # Replace benchmaq's `_ssh_run_stream` with a PTY-backed version
        # (same trick skypilot uses for live install output). Allocating a
        # pseudo-terminal makes uv/pip/curl call `isatty() == True`, so they
        # emit their live progress bars/spinners instead of the silent
        # non-TTY fallback. As a side benefit the PTY merges stderr into
        # stdout at the kernel level, which sidesteps the TM Go-SSH proxy's
        # habit of silently dropping the fd-2 stream — and the real exit
        # code comes back through, so we don't need a sentinel either.
        #
        # Inject `--clear` into `uv venv` for a clean rebuild — reusing a venv
        # left half-installed by an orphaned run gives "No module named
        # 'requests'" / broken .pth. The setup script kills any leftover venv
        # process first (below) so --clear isn't blocked by held files.
        def _patched_ssh_run_stream(ssh_client, cmd: str, label: str = "") -> int:
            cmd = cmd.replace("uv venv ", "uv venv --clear ")
            prefix = f"[{label}] " if label else ""
            print(f"{prefix}[shim] sending {len(cmd)}-byte script over ssh (pty)", flush=True)
            channel = ssh_client.get_transport().open_session()
            channel.get_pty(term="xterm-256color", width=120, height=40)
            channel.exec_command(f"bash -c {_shlex.quote(cmd)}")
            # Stream raw bytes through so progress bars / ANSI render live.
            # PTY mode emits \r\n at end-of-line — normalise to \n so we
            # don't double-emit blank lines. Solo \r (used by progress
            # bars to redraw in place) is still treated as a frame break,
            # so the log viewer shows each progress frame on its own line.
            buf = b""
            while True:
                chunk = channel.recv(4096)
                if not chunk:
                    break
                chunk = chunk.replace(b"\r\n", b"\n")
                buf += chunk
                while True:
                    idx = -1
                    for sep in (b"\n", b"\r"):
                        i = buf.find(sep)
                        if i != -1 and (idx == -1 or i < idx):
                            idx = i
                    if idx == -1:
                        break
                    line = buf[:idx].decode("utf-8", errors="replace")
                    buf = buf[idx + 1:]
                    line = _ANSI_RE.sub("", line).rstrip()
                    if not line:
                        continue
                    print(f"{prefix}{line}", flush=True)
            if buf:
                line = _ANSI_RE.sub("", buf.decode("utf-8", errors="replace")).rstrip()
                if line:
                    print(f"{prefix}{line}", flush=True)
            return channel.recv_exit_status()

        _benchmaq_runner._ssh_run_stream = _patched_ssh_run_stream  # type: ignore[assignment]
        logger.info("benchmaq _ssh_run_stream patched (pty form)")

        # Replace `run_remote_ssh` with a reconnect-per-step version. The
        # original opens one paramiko.SSHClient and reuses it for install,
        # SFTP config upload, and the benchmark run — but the TM proxy
        # only allows ONE session/subsystem channel per TCP connection, so
        # the SFTP step always fails with "Channel closed". We dial a
        # fresh SSH connection for each step.
        def _patched_run_remote_ssh(config, remote_cfg):
            import io
            import yaml
            import paramiko

            host = remote_cfg["host"]
            port = remote_cfg.get("port", 22)
            username = remote_cfg.get("username", "root")
            key_filename = remote_cfg.get("key_filename")
            password = remote_cfg.get("password")

            uv_cfg = remote_cfg.get("uv", {})
            venv_path = uv_cfg.get("path", "~/.bench-venv")
            python_version = uv_cfg.get("python_version", "3.11")
            benchmaq_ref = uv_cfg.get(
                "benchmaq_ref",
                "git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq.git@main",
            )
            vllm_version = uv_cfg.get("vllm_version")
            # Optional full `uv pip install` arg string for vLLM, applied (as an
            # upgrade) AFTER benchmaq[vllm] installs — for nightlies / custom CUDA
            # builds, e.g. "vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu130 ...".
            vllm_install_args = (uv_cfg.get("vllm_install_args") or "").strip()
            if vllm_install_args:
                # A leading run of NAME=VALUE tokens (e.g. `VLLM_USE_PRECOMPILED=1`,
                # which a git-fork install needs) is install-time ENV, not pip args —
                # emit it as a shell prefix so the fork installs precompiled instead
                # of uv choking on a bogus requirement. The `(?!=)` look-ahead keeps
                # real pins (`pkg==ver`) and flags out of the env split.
                import shlex as _ishlex
                import re as _ire
                _itoks = _ishlex.split(vllm_install_args)
                _ienv: dict[str, str] = {}
                _i = 0
                for _t in _itoks:
                    _m = _ire.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(?!=)(.*)$", _t)
                    if not _m:
                        break
                    _ienv[_m.group(1)] = _m.group(2)
                    _i += 1
                _iprefix = "".join(f"{_k}={_ishlex.quote(_v)} " for _k, _v in _ienv.items())
                _ipip = " ".join(_ishlex.quote(_t) for _t in _itoks[_i:])
                # sentencepiece: fork precompiled wheels skip it, but gemma/llama
                # tokenizers need it (else "Couldn't instantiate the backend tokenizer").
                _vllm_extra_install = f"{_iprefix}uv pip install -U {_ipip}\nuv pip install sentencepiece\n"
            else:
                _vllm_extra_install = ""

            # Accuracy mode: any benchmark item carrying an `accuracy:` block is
            # evaluated by our accuracy_eval.py (shipped + run after benchmaq).
            # It needs `datasets` (HF dataset loader) in the bench venv; benchmaq
            # skips these items (no `bench:` rows), so accuracy_eval owns them.
            _bench_items = config.get("benchmark") or []
            _has_accuracy = any(
                isinstance(it, dict) and it.get("accuracy") for it in _bench_items
            )

            def _is_fc(d):
                n = d if isinstance(d, str) else (d.get("name") if isinstance(d, dict) else "")
                return str(n).strip().lower() in {
                    "function-call", "function_call", "function-calling",
                    "functioncall", "taas", "scicom-intl/function-call-taas",
                }

            # The hard multi-turn function-calling benchmark is scored by the
            # vendored fc_eval.py (SyntheticGen evaluator), which needs `openai`.
            _has_function_call = any(
                isinstance(it, dict) and it.get("accuracy")
                and any(_is_fc(d) for d in (it["accuracy"].get("datasets") or []))
                for it in _bench_items
            )
            _accuracy_install = ""
            if _has_accuracy:
                _accuracy_install = "uv pip install -U datasets" + (" openai" if _has_function_call else "") + "\n"

            if key_filename:
                key_filename = os.path.expanduser(key_filename)

            # User env (incl CUDA_VISIBLE_DEVICES) exported on the VM before
            # install + benchmark, with absolute-path values mkdir'd. Set by the
            # gateway under remote.env; stripped from the uploaded config below.
            user_env = remote_cfg.get("env") or {}
            env_prefix = ""
            if user_env:
                _lines = []
                for _k, _v in user_env.items():
                    _vs = str(_v)
                    _lines.append(f"export {_k}={_shlex.quote(_vs)}")
                    if _vs.startswith("/"):
                        _lines.append(f"mkdir -p {_shlex.quote(_vs)}")
                env_prefix = "\n".join(_lines) + "\n"
                print(f"[shim] applying {len(user_env)} user env var(s) on remote", flush=True)

            def _connect():
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                kw = {"port": port, "username": username}
                if key_filename:
                    kw["key_filename"] = key_filename
                if password:
                    kw["password"] = password
                ssh.connect(host, **kw)
                return ssh

            print(
                f"Connecting to {username}@{host}:{port} "
                "(ssh backend, reconnect-per-step)..."
            )

            vllm_pin = f' "vllm=={vllm_version}"' if vllm_version else ""
            # TERM=dumb + NO_COLOR + UV_NO_PROGRESS keep uv (and pip/vllm)
            # in line-buffered mode despite the PTY — otherwise uv detects
            # a TTY and switches to a cursor-redraw progress UI that turns
            # the bench log viewer into ANSI soup.
            setup_script = (
                "set -e\n"
                "export TERM=dumb NO_COLOR=1 UV_NO_PROGRESS=1\n"
                + env_prefix +
                # Preflight: stale root-owned ~/.config from a prior run breaks
                # the uv installer's fish-completion drop. Best-effort repair.
                'sudo -n chown -R "$USER:$USER" "$HOME/.config" 2>/dev/null || true\n'
                'mkdir -p "$HOME/.config/fish/conf.d" 2>/dev/null || true\n'
                'if ! command -v uv &>/dev/null && ! [ -f "$HOME/.local/bin/uv" ]; then\n'
                "    curl -LsSf https://astral.sh/uv/install.sh | sh\n"
                "fi\n"
                'export PATH="$HOME/.local/bin:$PATH"\n'
                # Kill leftover python procs from a prior (gateway-orphaned) run
                # that still hold this venv open + occupy the GPUs — otherwise
                # `uv venv --clear` fails ("Directory not empty"). `grep -vw $$`
                # excludes THIS shell (pgrep -f also matches the script's own
                # argv, which contains the pattern). Scoped to our `.benchmark-venv`
                # so it never touches the user's own vLLM/services.
                'kill -9 $(pgrep -f "/.benchmark-venv/bin/python" 2>/dev/null | grep -vw "$$") 2>/dev/null || true\n'
                "sleep 1\n"
                f"uv venv {venv_path} --python {python_version}\n"
                f"source {venv_path}/bin/activate\n"
                f'uv pip install "benchmaq[vllm] @ {benchmaq_ref}"{vllm_pin}\n'
                # Optional: override/upgrade vLLM to a custom spec (e.g. a nightly).
                f"{_vllm_extra_install}"
                # Accuracy mode needs the HF datasets loader (GSM8K / openai/MMMLU).
                f"{_accuracy_install}"
                # huggingface_hub 1.x removed the `huggingface-cli` entrypoint
                # that benchmaq's model downloader still shells out to ("no
                # longer works. Use `hf` instead."). Forward it to the new `hf`
                # CLI — same `download <repo> --local-dir <dir>` argument shape.
                f'printf \'#!/usr/bin/env bash\\nexec hf "$@"\\n\' > {venv_path}/bin/huggingface-cli\n'
                f"chmod +x {venv_path}/bin/huggingface-cli\n"
                # A large model (e.g. an 800GB MoE like GLM-5.1-FP8) takes far longer
                # than benchmaq's default 200×5s=1000s health wait to load + compile +
                # capture CUDA graphs on its first serve, so the bench would fire
                # against a not-yet-listening server (ConnectionRefused → 0 tok/s).
                # Bump benchmaq's health-wait attempts (≈75 min ceiling; it exits as
                # soon as the server is healthy).
                f"find {venv_path}/lib -name '*.py' -path '*benchmaq*' "
                f"-exec sed -i 's/max_attempts=200/max_attempts=900/g' {{}} + || true\n"
                # …and abort the wait the moment the vLLM process dies, so a serve
                # that crashes on init (e.g. an unsupported flag / incompatible kernel)
                # fails fast instead of polling a dead port for the whole 75-min ceiling.
                f"export SGPU_BENCH_VENV={venv_path}\n"
                "python3 - <<'SGPU_PATCH_EOF'\n"
                "import glob, os\n"
                "needle = '        for attempt in range(max_attempts):\\n'\n"
                "check = '            if getattr(self, \"process\", None) is not None and self.process.poll() is not None:\\n                print(\"vLLM server process exited -- aborting health wait\"); return False\\n'\n"
                "for f in glob.glob(os.path.expanduser(os.environ['SGPU_BENCH_VENV']) + '/lib/python*/site-packages/benchmaq/**/server.py', recursive=True):\n"
                "    s = open(f).read()\n"
                "    if needle in s and 'aborting health wait' not in s:\n"
                "        open(f, 'w').write(s.replace(needle, needle + check, 1)); print('[sgpu] health-abort patched', f)\n"
                "SGPU_PATCH_EOF\n"
            )
            print()
            print("=" * 64)
            print("STEP: INSTALL")
            print("=" * 64)
            ssh = _connect()
            try:
                rc = _benchmaq_runner._ssh_run_stream(ssh, setup_script, label="install")
            finally:
                ssh.close()
            if rc != 0:
                raise RuntimeError(f"Remote setup failed (exit {rc})")

            remote_config_path = "/tmp/benchmaq_remote_config.yaml"
            print()
            print(f"Uploading config → {remote_config_path}")
            # Strip the `remote:` block before uploading — otherwise the
            # remote benchmaq reads it, sees `backend: ssh`, and recursively
            # tries to SSH back to itself through the proxy.
            remote_config = {k: v for k, v in config.items() if k != "remote"}
            config_bytes = yaml.dump(remote_config, default_flow_style=False).encode("utf-8")
            # Write via an exec channel (base64-pipe), NOT SFTP: some SSH proxies
            # fronting managed GPU VMs (PAI DSW, TM) don't support the SFTP
            # subsystem at all — `open_sftp()` fails with "EOF during
            # negotiation". exec is the only channel type these proxies allow,
            # and it's what the install step already uses. base64 keeps the
            # payload to a single safe argv (no quoting/heredoc/newline issues).
            import base64 as _base64
            b64 = _base64.b64encode(config_bytes).decode("ascii")
            write_cmd = (
                f"mkdir -p \"$(dirname {remote_config_path})\" && "
                f"printf %s {_shlex.quote(b64)} | base64 -d > {remote_config_path}"
            )
            ssh = _connect()
            try:
                chan = ssh.get_transport().open_session()
                chan.exec_command(f"bash -c {_shlex.quote(write_cmd)}")
                up_rc = chan.recv_exit_status()
            finally:
                ssh.close()
            if up_rc != 0:
                raise RuntimeError(f"Remote config upload failed (exit {up_rc})")

            # Ship accuracy_eval.py when any item runs in accuracy mode. It serves
            # each such config (benchmaq's VLLMServer) and scores GSM8K /
            # openai/MMMLU against the local endpoint, emitting @@ACCURACY lines
            # the gateway folds into result_json. Same base64-over-exec channel
            # as the config upload (no SFTP on the proxied VMs).
            remote_accuracy_path = "/tmp/sgpu_accuracy_eval.py"
            if _has_accuracy:
                def _ship(local_name, remote_path):
                    src = os.path.join(os.path.dirname(__file__), local_name)
                    with open(src, "rb") as _f:
                        b64 = _base64.b64encode(_f.read()).decode("ascii")
                    write = f"printf %s {_shlex.quote(b64)} | base64 -d > {remote_path}"
                    ssh = _connect()
                    try:
                        chan = ssh.get_transport().open_session()
                        chan.exec_command(f"bash -c {_shlex.quote(write)}")
                        rc = chan.recv_exit_status()
                    finally:
                        ssh.close()
                    if rc != 0:
                        raise RuntimeError(f"{local_name} upload failed (exit {rc})")
                    print(f"Uploaded {local_name} → {remote_path}")

                _ship("accuracy_eval.py", remote_accuracy_path)
                # The function-calling benchmark's scorer (vendored from
                # SyntheticGen); accuracy_eval.py subprocess-runs it from
                # /tmp/sgpu_fc_eval.py.
                if _has_function_call:
                    _ship("fc_eval.py", "/tmp/sgpu_fc_eval.py")

            run_script = (
                "set -e\n"
                "export TERM=dumb NO_COLOR=1\n"
                # Force plain HTTP downloads: disable the Xet CAS backend and the
                # (now-deprecated) hf_transfer accelerator. Both have caused the
                # hf download to abort on these VMs; plain HTTP is slower but
                # reliable.
                "export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0\n"
                # User env (HOME, caches, CUDA_VISIBLE_DEVICES) MUST be exported
                # before the PATH/activate lines below. With a HOME override the
                # `~`/`$HOME` in `uv venv ~/.bench-venv` (install step) expand to
                # the overridden home, so the venv lives there; if we activate
                # before re-exporting HOME here, `~`/`$HOME` fall back to the SSH
                # user's default home (/root) and miss that venv — the install and
                # benchmark steps must resolve the venv path identically.
                + env_prefix +
                'export PATH="$HOME/.local/bin:$PATH"\n'
                f"source {venv_path}/bin/activate\n"
                f"benchmaq bench {remote_config_path}\n"
                # Accuracy items have no `bench:` rows so benchmaq skips them;
                # run the quality eval against a fresh serve of each. Emits
                # @@ACCURACY lines (parsed by the gateway). Returns 0 on
                # per-dataset errors, so it never fails an otherwise-good run.
                + (f"python {remote_accuracy_path} {remote_config_path}\n" if _has_accuracy else "")
            )
            print()
            print("=" * 64)
            print("STEP: BENCHMARK")
            print("=" * 64)
            ssh = _connect()
            try:
                rc = _benchmaq_runner._ssh_run_stream(ssh, run_script, label="bench")
            finally:
                ssh.close()
            if rc != 0:
                raise RuntimeError(f"Remote benchmark failed (exit {rc})")

        _benchmaq_runner.run_remote_ssh = _patched_run_remote_ssh  # type: ignore[assignment]
        logger.info("benchmaq run_remote_ssh patched (reconnect-per-step)")
    except Exception as e:
        logger.warning("could not patch benchmaq _ssh_run_stream: %s", e)

    _INSTALLED = True
    logger.info("pyremote reconnect-per-command shim installed")
