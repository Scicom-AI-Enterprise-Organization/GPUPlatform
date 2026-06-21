"""Serverless `Provider` backed by a user-registered bare-metal VM (SSH).

Unlike RunPod/PI, a VM is a *fixed single node* — there's no cloud API to ask
"what's running". So this provider is constructed per provider-row (bound to one
host) and tracks the workers it launched in Redis (`vm_machines:{provider_id}`),
which is the authoritative liveness source the reconciler consults. That way a
transient SSH hiccup can never make the reconciler GC a live worker.

`provision()` SSHes in, drops a JSON config the worker-agent reads, bootstraps a
venv (idempotent) and `nohup`s the worker-agent. `terminate()` SSHes in and kills
that worker's process tree by the pid file it wrote. Both run paramiko in a
worker thread (mirroring `vm_probe` / `bench`), since paramiko is sync.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import shlex
import tarfile
import uuid
from pathlib import Path
from typing import Optional

from .provider import Provider, ProvisionResult, GpuAvailability
from . import vm_probe

logger = logging.getLogger("gateway.vm_provider")

SSH_TIMEOUT_S = 20
REMOTE_DIR = "~/.sgpu"


def _worker_agent_src_dir() -> str:
    """Where the worker-agent SOURCE lives so the gateway can ship it to the VM
    (no git clone of the private repo). `WORKER_AGENT_SRC_DIR` overrides it — set
    that in the gateway image where worker-agent is bundled; in a source checkout
    it's the sibling `worker-agent/` of the gateway package."""
    env = os.environ.get("WORKER_AGENT_SRC_DIR")
    if env and os.path.isdir(env):
        return env
    # gateway/gateway/vm_serverless_provider.py → <repo>/worker-agent
    cand = Path(__file__).resolve().parents[2] / "worker-agent"
    if cand.is_dir():
        return str(cand)
    raise RuntimeError(
        "worker-agent source not found — set WORKER_AGENT_SRC_DIR to its directory"
    )


def _worker_agent_tarball_b64() -> str:
    """A gzip tarball of the worker-agent source, base64-encoded for shipping over
    the SSH exec channel (proxied front-ends like PAI DSW have no SFTP). Built
    fresh per call so a code change is picked up without restarting the gateway."""
    src = _worker_agent_src_dir()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(src):
            dirs[:] = [
                d for d in dirs
                if d not in ("__pycache__", ".venv", "venv", ".git", "build", "dist")
                and not d.endswith(".egg-info")
            ]
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                full = os.path.join(root, f)
                tar.add(full, arcname=os.path.relpath(full, src))
    return base64.b64encode(buf.getvalue()).decode()


class VMProvider(Provider):
    name = "vm"

    def __init__(
        self,
        *,
        provider_id: str,
        host: str,
        port: int,
        user: str,
        private_key_pem: str,
        gpu_count: int,
        rdb,
        gateway_public_url: Optional[str] = None,
        worker_redis_url: Optional[str] = None,
        install_spec: Optional[str] = None,
        reverse_tunnel: bool = False,
    ) -> None:
        self._provider_id = provider_id
        self._host = host
        self._port = int(port or 22)
        self._user = user or "root"
        self._private_key_pem = private_key_pem
        self._gpu_count = int(gpu_count or 0)
        self._rdb = rdb
        # Where workers phone home. Same reachability caveat as RunPod: localhost
        # won't be reachable from a remote box — needs a tailnet/public URL, or
        # the reverse-tunnel mode below.
        self._gateway_url = (
            gateway_public_url
            or os.environ.get("GATEWAY_PUBLIC_URL")
            or os.environ.get("GATEWAY_URL")
            or "http://127.0.0.1:8080"
        )
        self._worker_redis_url = (
            worker_redis_url
            or os.environ.get("WORKER_REDIS_URL")
            or os.environ.get("REDIS_URL")
            or "redis://127.0.0.1:6379"
        )
        # Retained for back-compat; the worker-agent is now shipped as a source
        # tarball from the gateway (see _launch_sync), not pip-installed from git.
        self._install_spec = install_spec
        # Reverse-tunnel mode: forward the gateway + Redis back over SSH so the
        # VM reaches them at its own localhost. We compute the local targets from
        # the gateway's own bind/redis env, and tell the worker to use the VM's
        # loopback (mirroring the local ports for clarity).
        self._reverse_tunnel = reverse_tunnel
        gw_host, gw_port = ("127.0.0.1", 8080)
        if self._reverse_tunnel:
            from . import vm_tunnel
            # Local gateway endpoint reachable from the gateway process.
            gw_host, gw_port = vm_tunnel.parse_host_port(
                os.environ.get("GATEWAY_BIND", "127.0.0.1:8080"), 8080
            )
            if gw_host in ("", "0.0.0.0"):
                gw_host = "127.0.0.1"
            rd_host, rd_port = vm_tunnel.parse_host_port(
                os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"), 6379
            )
            if rd_host in ("", "0.0.0.0"):
                rd_host = "127.0.0.1"
            self._tunnel_gw = (gw_host, gw_port)
            self._tunnel_redis = (rd_host, rd_port)
            # Worker hits the VM's loopback (mirror ports); the tunnel maps it home.
            self._gateway_url = f"http://127.0.0.1:{gw_port}"
            self._worker_redis_url = f"redis://127.0.0.1:{rd_port}"

    # ---- Provider ABC -----------------------------------------------------

    async def provision(
        self,
        app_id: str,
        model: str,
        gpu: str,  # noqa: ARG002 — VM hardware is fixed; gpu is informational
        env: dict[str, str],
        gpu_count: int = 1,  # noqa: ARG002 — fixed by the VM, not requested
        cloud_type: Optional[str] = None,  # noqa: ARG002
        container_disk_gb: Optional[int] = None,  # noqa: ARG002
        volume_gb: Optional[int] = None,  # noqa: ARG002
    ) -> ProvisionResult:
        machine_id = f"m-vm-{uuid.uuid4().hex[:8]}"
        worker_env = self._build_worker_env(app_id, machine_id, model, env)
        import asyncio
        # Bring up the reverse tunnel before the worker boots so its first
        # register/redis call already resolves.
        await self._ensure_tunnel()
        await asyncio.to_thread(self._launch_sync, machine_id, worker_env)
        # Track liveness in Redis (authoritative for a host with no cloud API).
        await self._rdb.sadd(f"vm_machines:{self._provider_id}", machine_id)
        await self._rdb.set(f"vm_machine:{machine_id}:app", app_id)
        # proxy endpoint: open the gateway→VM forward tunnel to the member's vLLM
        # port and publish the upstream URL the dispatch path proxies to.
        if worker_env.get("WORKER_MODE") == "proxy":
            await self._wire_proxy_forward(app_id, worker_env)
        logger.info("vm-provision: app=%s host=%s → %s", app_id, self._host, machine_id)
        return ProvisionResult(machine_id=machine_id, cost_per_hr=None)

    async def terminate(self, machine_id: str) -> None:
        import asyncio
        try:
            await asyncio.to_thread(self._terminate_sync, machine_id)
        finally:
            await self._rdb.srem(f"vm_machines:{self._provider_id}", machine_id)
            await self._rdb.delete(f"vm_machine:{machine_id}:app")
        logger.info("vm-terminate: %s torn down on %s", machine_id, self._host)

    async def list_machines(self) -> list[str]:
        members = await self._rdb.smembers(f"vm_machines:{self._provider_id}")
        return list(members)

    async def list_machines_for_app(self, app_id: str) -> list[str]:
        out: list[str] = []
        for mid in await self.list_machines():
            if (await self._rdb.get(f"vm_machine:{mid}:app")) == app_id:
                out.append(mid)
        return out

    async def forget_machine(self, machine_id: str) -> None:
        """Drop a dead machine from this provider's redis bookkeeping (the set the
        reconciler lists from) without SSHing — the process is already gone. Lets
        the reconciler reap stale orphans so they stop being re-listed/warned."""
        await self._rdb.srem(f"vm_machines:{self._provider_id}", machine_id)
        await self._rdb.delete(f"vm_machine:{machine_id}:app")

    async def purge_app(self, app_id: str) -> int:
        """Hard cleanup: sweep EVERY on-disk worker remnant for this app (not just
        what redis tracks — crash-loop churn leaves orphan pidfiles/logs/configs),
        kill its process tree + vLLM engines, remove its files, then clear redis.
        App-scoped via each config's APP_ID so a shared box's other endpoints are
        untouched. Returns the number of machine remnants purged."""
        import asyncio
        machine_ids = await asyncio.to_thread(self._purge_sync, app_id)
        idx = set(await self._rdb.smembers(f"worker_index:{app_id}"))
        all_ids = set(machine_ids) | idx
        for mid in all_ids:
            await self._rdb.srem(f"vm_machines:{self._provider_id}", mid)
            await self._rdb.delete(
                f"vm_machine:{mid}:app", f"worker:{mid}", f"worker:{mid}:models",
                f"register_token:{mid}", f"worker:{mid}:drain",
            )
        if all_ids:
            await self._rdb.srem(f"worker_index:{app_id}", *all_ids)
        # Proxy endpoint bookkeeping (forward-tunnel upstream); the in-process
        # listener thread lingers harmlessly until the next gateway restart, but
        # the upstream key going away makes the dispatch path return 503 at once.
        await self._rdb.delete(f"proxy:{app_id}:upstream", f"proxy:{app_id}:vmport")
        await self._rdb.srem(f"vm_proxy_apps:{self._provider_id}", app_id)
        logger.info("vm-purge: app=%s purged %d remnant(s) on %s", app_id, len(machine_ids), self._host)
        return len(machine_ids)

    def _purge_sync(self, app_id: str) -> list[str]:
        client = self._connect_sync()
        try:
            # Match this app's worker configs by the exact APP_ID line. For each:
            # TERM the worker's pgroup, reap its recorded engine pids (reuse-guarded
            # cleanup module — never a box-wide vLLM pkill), pkill anything still on
            # that config, then remove all of its on-disk files.
            tag = '"APP_ID": ' + json.dumps(app_id)  # e.g. "APP_ID": "tm-fleet-2"
            script = (
                "shopt -s nullglob; "
                f'for cfg in {REMOTE_DIR}/worker-*.json; do '
                f'  if grep -qF {shlex.quote(tag)} "$cfg"; then '
                '    base="${cfg%.json}"; name="$(basename "$base")"; mid="${name#worker-}"; '
                '    pid="$(cat "$base.pid" 2>/dev/null)"; '
                '    if [ -n "$pid" ]; then kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true; fi; '
                '    pkill -9 -f "$name.json" 2>/dev/null || true; '
                # Only pay the python-startup cost to reap engines for workers that
                # actually recorded some (most crash-loop workers die at redis-ping
                # BEFORE launching vLLM → empty/absent enginepids → skip).
                f'    if [ -s "$base.enginepids" ]; then {REMOTE_DIR}/venv/bin/python -m worker_agent.multi.cleanup "$base.enginepids" 2>/dev/null || true; fi; '
                '    rm -f "$base.json" "$base.pid" "$base.log" "$base.enginepids"; '
                '    echo "PURGED $mid"; '
                "  fi; "
                "done"
            )
            out = self._run(client, f"bash -lc {shlex.quote(script)}", timeout=110)
            return [ln.split(" ", 1)[1].strip() for ln in out.splitlines() if ln.startswith("PURGED ")]
        finally:
            try:
                client.close()
            except Exception:
                pass

    async def check_availability(
        self,
        gpu: str,
        count: int,
        cloud_type: Optional[str] = None,  # noqa: ARG002
    ) -> GpuAvailability:
        # The VM's GPUs are fixed and known; treat as available if enough exist.
        return GpuAvailability(
            gpu=gpu, count=count,
            available=(count <= self._gpu_count) if self._gpu_count else None,
        )

    async def shutdown(self) -> None:
        # Don't kill VM workers on gateway restart — they keep serving and the
        # reconciler/autoscaler re-converge. Mirrors RunPodProvider.
        return

    async def ensure_connectivity(self) -> None:
        """Called each autoscaler tick. In reverse-tunnel mode, (re)establishes
        the SSH forwards — this is what heals the tunnel after a gateway restart
        (the tunnel lives in-process and dies with it). Also re-opens the
        gateway→VM forward tunnel for any proxy endpoint on this VM."""
        if self._reverse_tunnel:
            await self._ensure_tunnel()
        await self._ensure_proxy_forwards()

    async def _ensure_tunnel(self) -> None:
        if not self._reverse_tunnel:
            return
        import asyncio

        from . import vm_tunnel
        forwards = [
            vm_tunnel.Forward(self._tunnel_gw[1], self._tunnel_gw[0], self._tunnel_gw[1]),
            vm_tunnel.Forward(self._tunnel_redis[1], self._tunnel_redis[0], self._tunnel_redis[1]),
        ]
        await asyncio.to_thread(
            vm_tunnel.ensure, self._host, self._port, self._user, self._private_key_pem, forwards,
        )

    @staticmethod
    def _proxy_member_port(worker_env: dict[str, str]) -> Optional[int]:
        """The vLLM localhost port the lone proxy member binds — read from the
        MULTI_MODEL_CONFIG the autoscaler built (build_multi_model_config assigns
        it deterministically from the endpoint's GPU set)."""
        try:
            mmc = json.loads(worker_env.get("MULTI_MODEL_CONFIG") or "{}")
            members = mmc.get("models") or []
            if members:
                return int(members[0]["port"])
        except (ValueError, KeyError, TypeError):
            pass
        return None

    async def _wire_proxy_forward(self, app_id: str, worker_env: dict[str, str]) -> None:
        """Open (or reuse) a gateway→VM forward tunnel to the proxy member's vLLM
        port and publish `proxy:{app_id}:upstream` for the dispatch path. The
        member port is also stored durably so ensure_connectivity can re-open the
        forward (with a fresh local port) after a gateway restart."""
        import asyncio

        from . import vm_tunnel
        port = self._proxy_member_port(worker_env)
        if not port:
            logger.warning("vm-proxy: app=%s has no member port in MULTI_MODEL_CONFIG", app_id)
            return
        local_port = await asyncio.to_thread(
            vm_tunnel.ensure_forward,
            self._host, self._port, self._user, self._private_key_pem, port,
        )
        upstream = f"http://127.0.0.1:{local_port}"
        await self._rdb.set(f"proxy:{app_id}:upstream", upstream)
        await self._rdb.set(f"proxy:{app_id}:vmport", str(port))
        await self._rdb.sadd(f"vm_proxy_apps:{self._provider_id}", app_id)
        logger.info("vm-proxy: app=%s upstream=%s (VM vLLM :%d)", app_id, upstream, port)

    async def _ensure_proxy_forwards(self) -> None:
        """Re-establish forward tunnels for every proxy endpoint on this VM and
        refresh their published upstream (the in-process tunnel + its local port
        are lost on a gateway restart)."""
        import asyncio

        from . import vm_tunnel
        app_ids = await self._rdb.smembers(f"vm_proxy_apps:{self._provider_id}")
        for app_id in app_ids:
            vmport = await self._rdb.get(f"proxy:{app_id}:vmport")
            if not vmport:
                continue
            try:
                local_port = await asyncio.to_thread(
                    vm_tunnel.ensure_forward,
                    self._host, self._port, self._user, self._private_key_pem, int(vmport),
                )
            except Exception as e:
                logger.warning("vm-proxy: app=%s forward re-ensure failed: %s", app_id, e)
                continue
            await self._rdb.set(f"proxy:{app_id}:upstream", f"http://127.0.0.1:{local_port}")

    # ---- internals --------------------------------------------------------

    def _build_worker_env(
        self, app_id: str, machine_id: str, model: str, env: dict[str, str]
    ) -> dict[str, str]:
        worker_env: dict[str, str] = {
            "APP_ID": app_id,
            "MACHINE_ID": machine_id,
            "GATEWAY_URL": self._gateway_url,
            "WORKER_REDIS_URL": self._worker_redis_url,
            "REGISTRATION_TOKEN": env.get("REGISTRATION_TOKEN", ""),
            "WORKER_MODE": env.get("WORKER_MODE", "vllm"),
        }
        # Single-model: vLLM loads MODEL_ID. Multi: MULTI_MODEL_CONFIG carries
        # the whole fleet spec (the autoscaler injected it into `env`).
        if worker_env["WORKER_MODE"] != "multi":
            worker_env["MODEL_ID"] = model
        # Carry through everything else the autoscaler set (VLLM_EXTRA_ARGS,
        # MULTI_MODEL_CONFIG, SLEEP_LEVEL, TOTAL_GPUS, metrics env, …) without
        # clobbering the keys we already fixed above.
        for k, v in env.items():
            worker_env.setdefault(k, v)
        return worker_env

    def _connect_sync(self, attempts: int = 4):
        import time

        import paramiko

        pkey = vm_probe._load_pkey(self._private_key_pem)
        last_err: Exception | None = None
        for i in range(attempts):
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    hostname=self._host,
                    port=self._port,
                    username=self._user,
                    pkey=pkey,
                    timeout=SSH_TIMEOUT_S,
                    banner_timeout=SSH_TIMEOUT_S,
                    auth_timeout=SSH_TIMEOUT_S,
                    look_for_keys=False,
                    allow_agent=False,
                )
                return client
            except (paramiko.SSHException, EOFError, OSError) as e:
                # Proxied SSH front-ends (e.g. Alibaba PAI DSW) drop handshakes
                # under concurrent connection load with "EOF during negotiation".
                # Retry with backoff — these are transient and a clean retry
                # usually lands once the reverse-tunnel handshake isn't racing.
                last_err = e
                try:
                    client.close()
                except Exception:
                    pass
                if i < attempts - 1:
                    logger.warning("ssh connect to %s failed (%s), retry %d/%d", self._host, e, i + 1, attempts)
                    time.sleep(1.5 * (i + 1))
        raise last_err if last_err else RuntimeError("ssh connect failed")

    def _launch_sync(self, machine_id: str, worker_env: dict[str, str]) -> None:
        client = self._connect_sync()
        try:
            cfg_path = f"{REMOTE_DIR}/worker-{machine_id}.json"
            log_path = f"{REMOTE_DIR}/worker-{machine_id}.log"
            pid_path = f"{REMOTE_DIR}/worker-{machine_id}.pid"
            venv = f"{REMOTE_DIR}/venv"
            # 1. Drop the config the worker reads (keeps the big MULTI_MODEL_CONFIG
            #    JSON + the registration token off the process arg list). Written
            #    via base64 over the exec channel rather than SFTP — proxied SSH
            #    front-ends (Alibaba PAI DSW) don't expose the SFTP subsystem and
            #    reject the channel with "EOF during negotiation".
            home = self._run(client, "echo $HOME").strip() or f"/home/{self._user}"
            remote_dir = REMOTE_DIR.replace("~", home)
            remote_cfg = cfg_path.replace("~", home)
            # Tell the worker where its own stdout log lives (the nohup target
            # below) so it can ship it to the gateway as the "__worker__" source,
            # and where to persist the engine process-groups it owns so terminate
            # can reap ONLY our processes (never a box-wide pkill).
            worker_env = {
                **worker_env,
                "WORKER_SELF_LOG_PATH": f"{remote_dir}/worker-{machine_id}.log",
                "WORKER_ENGINE_PIDS_PATH": f"{remote_dir}/worker-{machine_id}.enginepids",
            }
            cfg_b64 = base64.b64encode(json.dumps(worker_env).encode()).decode()
            rc0, out0, err0 = self._run_full(
                client,
                f"mkdir -p {shlex.quote(remote_dir)} && "
                f"echo {cfg_b64} | base64 -d > {shlex.quote(remote_cfg)}",
            )
            if rc0 != 0:
                raise RuntimeError(f"failed to write worker config (rc={rc0}): {err0[:300] or out0[:300]}")
            # 2. Install the worker-agent on the VM — NEVER a git clone (the repo
            #    is private and the VM has no creds). Preferred path: ship the
            #    SOURCE tarball from the gateway (base64 over the same exec channel
            #    as the config) and `uv pip install` it, so a gateway code change
            #    propagates on the next provision. If the gateway image doesn't
            #    bundle the source (WORKER_AGENT_SRC_DIR unset / dir missing), fall
            #    back to whatever worker-agent is already installed in the VM venv
            #    rather than hard-failing the whole provision.
            wa_dir = f"{REMOTE_DIR}/worker-agent"
            try:
                wa_b64 = _worker_agent_tarball_b64()
            except Exception as e:
                wa_b64 = None
                logger.warning("worker-agent source not bundled (%s); using the VM's existing venv", e)
            if wa_b64:
                remote_wa_tgz = f"{remote_dir}/wa.tgz"
                rc_wa, out_wa, err_wa = self._run_full(
                    client, f"echo {wa_b64} | base64 -d > {shlex.quote(remote_wa_tgz)}",
                )
                if rc_wa != 0:
                    raise RuntimeError(f"failed to upload worker-agent tarball (rc={rc_wa}): {err_wa[:300] or out_wa[:300]}")
                install_block = (
                    f"rm -rf {wa_dir} && mkdir -p {wa_dir} && tar xzf {REMOTE_DIR}/wa.tgz -C {wa_dir}; "
                    f"(command -v uv >/dev/null 2>&1 && uv pip install --python {venv}/bin/python {wa_dir}) "
                    f"  || {venv}/bin/pip install {wa_dir}; "
                )
            else:
                install_block = (
                    f"if ! {venv}/bin/python -c 'import worker_agent' 2>/dev/null; then "
                    f"  echo 'worker_agent not installed on the VM and no source bundled in the gateway "
                    f"image — set WORKER_AGENT_SRC_DIR or COPY worker-agent/ into the image' >&2; exit 3; "
                    f"fi; "
                )
            script = (
                f"set -e; mkdir -p {REMOTE_DIR}; "
                f"if [ ! -x {venv}/bin/python ]; then "
                f"  (command -v uv >/dev/null 2>&1 && uv venv {venv}) "
                f"  || python3 -m venv {venv}; "
                f"fi; "
                + install_block +
                f"WORKER_CONFIG_FILE={cfg_path} nohup {venv}/bin/python -m worker_agent.main "
                f"  > {log_path} 2>&1 & echo $! > {pid_path}"
            )
            rc, out, err = self._run_full(client, f"bash -lc {shlex.quote(script)}")
            if rc != 0:
                raise RuntimeError(f"vm worker launch failed (rc={rc}): {err[:400] or out[:400]}")
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _terminate_sync(self, machine_id: str) -> None:
        client = self._connect_sync()
        try:
            pid_path = f"{REMOTE_DIR}/worker-{machine_id}.pid"
            pids_path = f"{REMOTE_DIR}/worker-{machine_id}.enginepids"
            venv = f"{REMOTE_DIR}/venv"
            # Kill the worker's process tree by its recorded pid, then a pkill
            # scoped to THIS machine's config path (so we never touch another
            # worker on the box).
            cfg_tag = shlex.quote(f"worker-{machine_id}.json")
            # The engines run in their OWN sessions (start_new_session), so they
            # survive the worker kill — orphaned, still holding GPU. We reap them
            # PRECISELY: the worker persisted the exact process-groups it owns to
            # `pids_path`, and the cleanup module kills only those (reuse-guarded
            # by start-time). NEVER a box-wide `pkill -f VLLM::` — the VM may be
            # shared with another endpoint or a user's own vLLM.
            script = (
                f"if [ -f {pid_path} ]; then "
                f"  PID=$(cat {pid_path}); "
                f"  kill -TERM -- -$PID 2>/dev/null || kill -TERM $PID 2>/dev/null || true; "
                f"  sleep 2; kill -KILL $PID 2>/dev/null || true; "
                f"fi; "
                f"pkill -9 -f {cfg_tag} 2>/dev/null || true; "
                f"{venv}/bin/python -m worker_agent.multi.cleanup {pids_path} 2>/dev/null || true; "
                f"rm -f {pid_path}; true"
            )
            self._run(client, f"bash -lc {shlex.quote(script)}")
        finally:
            try:
                client.close()
            except Exception:
                pass

    @staticmethod
    def _run(client, cmd: str, timeout: float = SSH_TIMEOUT_S) -> str:
        _, stdout, _ = client.exec_command(cmd, timeout=timeout)
        stdout.channel.recv_exit_status()
        return stdout.read().decode(errors="replace")

    @staticmethod
    def _run_full(client, cmd: str) -> tuple[int, str, str]:
        _, stdout, stderr = client.exec_command(cmd, timeout=120)
        rc = stdout.channel.recv_exit_status()
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        return rc, out, err
