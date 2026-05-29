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
import json
import logging
import os
import shlex
import uuid
from typing import Optional

from .provider import Provider, ProvisionResult, GpuAvailability
from . import vm_probe

logger = logging.getLogger("gateway.vm_provider")

SSH_TIMEOUT_S = 20
# Where the worker-agent lives so a fresh VM can `uv pip install` it. Overridable
# for forks / private mirrors. The default points at the public repo's subdir.
DEFAULT_INSTALL_SPEC = os.environ.get(
    "WORKER_AGENT_INSTALL_SPEC",
    "git+https://github.com/AIES-Infra/GPUPlatform.git#subdirectory=worker-agent",
)
REMOTE_DIR = "~/.sgpu"


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
        self._install_spec = install_spec or DEFAULT_INSTALL_SPEC
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
        (the tunnel lives in-process and dies with it)."""
        if self._reverse_tunnel:
            await self._ensure_tunnel()

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
            # 2. Bootstrap venv + worker-agent (idempotent), then nohup the worker
            #    pointed at the config file. `uv` is assumed present (same as the
            #    bench VM flow); fall back to python -m venv + pip if not.
            install = shlex.quote(self._install_spec)
            script = (
                f"set -e; mkdir -p {REMOTE_DIR}; "
                f"if [ ! -x {venv}/bin/python ]; then "
                f"  (command -v uv >/dev/null 2>&1 && uv venv {venv}) "
                f"  || python3 -m venv {venv}; "
                f"fi; "
                f"if ! {venv}/bin/python -c 'import worker_agent' 2>/dev/null; then "
                f"  (command -v uv >/dev/null 2>&1 && uv pip install --python {venv}/bin/python {install}) "
                f"  || {venv}/bin/pip install {install}; "
                f"fi; "
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
    def _run(client, cmd: str) -> str:
        _, stdout, _ = client.exec_command(cmd, timeout=SSH_TIMEOUT_S)
        stdout.channel.recv_exit_status()
        return stdout.read().decode(errors="replace")

    @staticmethod
    def _run_full(client, cmd: str) -> tuple[int, str, str]:
        _, stdout, stderr = client.exec_command(cmd, timeout=120)
        rc = stdout.channel.recv_exit_status()
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        return rc, out, err
