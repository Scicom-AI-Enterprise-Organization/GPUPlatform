"""RunPod provider — calls the RunPod REST API to launch GPU pods.

Requires (env vars):
  RUNPOD_API_KEY          - API key from runpod.io
  RUNPOD_TEMPLATE_ID      - id of a RunPod private template containing our
                            worker image (with Tailscale baked in)
  GATEWAY_PUBLIC_URL      - URL workers dial back to from RunPod pods

Optional:
  RUNPOD_API_BASE         - default: https://rest.runpod.io/v1
  RUNPOD_CLOUD_TYPE       - "COMMUNITY" (cheaper) or "SECURE" (verified hosts).
                            Default: COMMUNITY.
  RUNPOD_CONTAINER_DISK_GB - default: 50
  RUNPOD_VOLUME_GB         - default: 0 (no persistent volume)
  TS_AUTHKEY               - if set, injected into spawned pods so they join
                             the user's tailnet on boot. Required when the
                             gateway's redis is only reachable on a tailnet.
  RUNPOD_NAME_PREFIX       - default: serverlessgpu — pods we own get this
                             prefix so list_machines() filters us correctly.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from typing import Any, Optional

import httpx

from .provider import GpuAvailability, Provider, ProvisionResult

logger = logging.getLogger("gateway.runpod_provider")

_AVAILABILITY_TTL_S = 30.0


# Map app-spec GPU strings to RunPod gpuTypeIds. The right-hand strings come
# from RunPod's GraphQL `gpuTypes.id` field and are case-sensitive.
_GPU_NAME_MAP = {
    # Datacenter — current gen
    "H200": "NVIDIA H200",
    "H100": "NVIDIA H100 80GB HBM3",
    "H100-SXM": "NVIDIA H100 80GB HBM3",
    "H100-80GB": "NVIDIA H100 80GB HBM3",
    "H100-PCIe": "NVIDIA H100 PCIe",
    "H100-NVL": "NVIDIA H100 NVL",
    "B200": "NVIDIA B200",
    "MI300X": "AMD Instinct MI300X OAM",
    # Datacenter — Ampere
    "A100": "NVIDIA A100 80GB PCIe",
    "A100-80GB": "NVIDIA A100 80GB PCIe",
    "A100-SXM": "NVIDIA A100-SXM4-80GB",
    "A100-40G": "NVIDIA A100-PCIE-40GB",
    "A100-40GB": "NVIDIA A100-PCIE-40GB",
    "A40": "NVIDIA A40",
    "A10": "NVIDIA A10",
    "A10G": "NVIDIA A10",
    "A10-24GB": "NVIDIA A10",
    # Datacenter — Ada
    "L40S": "NVIDIA L40S",
    "L40S-48GB": "NVIDIA L40S",
    "L40": "NVIDIA L40",
    "L4": "NVIDIA L4",
    # Workstation
    "RTX6000-Ada": "NVIDIA RTX 6000 Ada Generation",
    "RTX-A6000": "NVIDIA RTX A6000",
    "A6000": "NVIDIA RTX A6000",
    "RTX-A5000": "NVIDIA RTX A5000",
    "A5000": "NVIDIA RTX A5000",
    "RTX-A4000": "NVIDIA RTX A4000",
    "A4000": "NVIDIA RTX A4000",
    # Consumer
    "RTX4090": "NVIDIA GeForce RTX 4090",
    "rtx4090": "NVIDIA GeForce RTX 4090",
    "RTX3090": "NVIDIA GeForce RTX 3090",
    "rtx3090": "NVIDIA GeForce RTX 3090",
    "RTX3090Ti": "NVIDIA GeForce RTX 3090 Ti",
    "rtx3090ti": "NVIDIA GeForce RTX 3090 Ti",
    "RTX5090": "NVIDIA GeForce RTX 5090",
    # Older
    "V100": "Tesla V100-SXM2-32GB",
    "V100-32GB": "Tesla V100-SXM2-32GB",
    "T4": "Tesla T4",
}


def _map_gpu(name: str) -> str:
    if name in _GPU_NAME_MAP:
        return _GPU_NAME_MAP[name]
    return name  # assume caller already used RunPod's enum


def _gen_ed25519() -> tuple[str, str]:
    """(openssh_public_key, openssh_pem_private_key) for an ephemeral tunnel key."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    k = Ed25519PrivateKey.generate()
    priv = k.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.OpenSSH,
        serialization.NoEncryption(),
    ).decode()
    pub = k.public_key().public_bytes(
        serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH
    ).decode()
    return pub, priv


# Where the worker-agent + its vLLM venv live on the pod (mirrors the VM's ~/.sgpu).
# RunPod pods run as root.
_POD_REMOTE = "/root/.sgpu"
_POD_VENV = "/root/.sgpu/venv"


class RunPodProvider(Provider):
    name = "runpod"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        template_id: Optional[str] = None,
        gateway_public_url: Optional[str] = None,
        cloud_type: Optional[str] = None,
        container_disk_in_gb: Optional[int] = None,
        volume_in_gb: Optional[int] = None,
        ts_authkey: Optional[str] = None,
        name_prefix: Optional[str] = None,
        reverse_tunnel: Optional[bool] = None,
        ssh_priv_pem: Optional[str] = None,
        ssh_pub: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise RuntimeError("RUNPOD_API_KEY env var or constructor arg required")

        self.api_base = (
            api_base or os.environ.get("RUNPOD_API_BASE", "https://rest.runpod.io/v1")
        ).rstrip("/")

        self.template_id = template_id or os.environ.get("RUNPOD_TEMPLATE_ID")
        if not self.template_id:
            raise RuntimeError(
                "RUNPOD_TEMPLATE_ID env var required — create a private template "
                "containing our worker image and set its id here"
            )

        self.gateway_public_url = (
            gateway_public_url
            or os.environ.get("GATEWAY_PUBLIC_URL")
            or os.environ.get("GATEWAY_URL")
        )
        if not self.gateway_public_url:
            raise RuntimeError(
                "GATEWAY_PUBLIC_URL env var required so RunPod workers can reach the gateway"
            )

        self.cloud_type = cloud_type or os.environ.get("RUNPOD_CLOUD_TYPE", "COMMUNITY")
        self.container_disk_in_gb = (
            container_disk_in_gb
            if container_disk_in_gb is not None
            else int(os.environ.get("RUNPOD_CONTAINER_DISK_GB", "50"))
        )
        self.volume_in_gb = (
            volume_in_gb
            if volume_in_gb is not None
            else int(os.environ.get("RUNPOD_VOLUME_GB", "0"))
        )
        # TS_AUTHKEY is optional — only required when the gateway's redis is
        # only reachable on a tailnet. Workers without it can't reach in-cluster
        # services unless redis is exposed publicly.
        self.ts_authkey = ts_authkey or os.environ.get("TS_AUTHKEY")
        self.name_prefix = name_prefix or os.environ.get("RUNPOD_NAME_PREFIX", "serverlessgpu")

        # Filter for hosts with compatible CUDA drivers. The vllm/vllm-openai:latest
        # base image needs CUDA 13+; older RunPod hosts will fail with
        # "nvidia-container-cli: requirement error: unsatisfied condition: cuda>=13.0".
        # Override via env (comma-separated, e.g. "12.4,12.8") if the worker image
        # is rebuilt against an older CUDA.
        cuda_env = os.environ.get("RUNPOD_ALLOWED_CUDA_VERSIONS", "13.0")
        self.allowed_cuda_versions = [v.strip() for v in cuda_env.split(",") if v.strip()]

        self._client = client or httpx.AsyncClient(
            base_url=self.api_base,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        self._owns_client = client is None

        # Reverse-tunnel mode: instead of the pod phoning home to a public
        # GATEWAY_URL (which from RunPod's network points at the pod itself when
        # the gateway runs on a laptop), we SSH into the pod and forward ITS
        # loopback gateway+redis ports back to the gateway process — exactly like
        # the VM provider's VM_REVERSE_TUNNEL. The pod gets our PUBLIC_KEY injected
        # (the template writes it to authorized_keys, same as benchmaq) and the
        # worker is pointed at 127.0.0.1. On by default; disable with
        # RUNPOD_REVERSE_TUNNEL=0 (then prod uses the public GATEWAY_URL).
        self.reverse_tunnel = (
            reverse_tunnel
            if reverse_tunnel is not None
            else os.environ.get("RUNPOD_REVERSE_TUNNEL", "1").strip().lower() not in ("0", "false", "no", "")
        )
        self.ssh_priv_pem = ssh_priv_pem
        self.ssh_pub = ssh_pub
        # machine_id → (public_ip, ssh_port) once the pod is RUNNING + SSH is up.
        self._ssh: dict[str, tuple[str, int]] = {}
        if self.reverse_tunnel:
            from . import vm_tunnel
            gw_host, gw_port = vm_tunnel.parse_host_port(os.environ.get("GATEWAY_BIND", "127.0.0.1:8080"), 8080)
            if gw_host in ("", "0.0.0.0"):
                gw_host = "127.0.0.1"
            rd_host, rd_port = vm_tunnel.parse_host_port(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"), 6379)
            if rd_host in ("", "0.0.0.0"):
                rd_host = "127.0.0.1"
            self._tunnel_gw = (gw_host, gw_port)
            self._tunnel_redis = (rd_host, rd_port)
            self._loop_gateway_url = f"http://127.0.0.1:{gw_port}"
            self._loop_redis_url = f"redis://127.0.0.1:{rd_port}"
            # No provider-row keypair (the gateway-default env account) → mint an
            # ephemeral Ed25519 key for this gateway process. NOTE: it doesn't
            # survive a gateway restart, so live pods can't reconnect — use a
            # RunPod *provider row* (persistent ssh_pub/ssh_priv) for prod.
            if not (self.ssh_priv_pem and self.ssh_pub):
                self.ssh_pub, self.ssh_priv_pem = _gen_ed25519()
                logger.warning(
                    "runpod reverse-tunnel: no provider-row SSH key — using an ephemeral "
                    "key (lost on restart; live pods won't reconnect). Use a provider row for prod."
                )

        # A RunPod pod is just a container/VM. In reverse-tunnel mode we boot a
        # STOCK image that runs sshd + applies PUBLIC_KEY on its own (the benchmark
        # image — CUDA 12.8 / torch 2.8), then SSH in and do exactly what the VM
        # provider does: make a uv venv, `uv pip install` vllm + the worker-agent
        # (shipped as a tarball — no git, no creds), and nohup the worker. No
        # custom image to build/push; worker-agent code changes ship per provision.
        self.image = os.environ.get(
            "RUNPOD_IMAGE", "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
        )
        # machine_id → the worker config (env dict) to write over SSH at launch.
        self._worker_env: dict[str, dict] = {}

        # machine_id (ours) → pod_id (RunPod's)
        self._pod_ids: dict[str, str] = {}

        # availability cache: (gpu, count) -> (result, expiry_epoch)
        self._avail_cache: dict[tuple[str, int], tuple[GpuAvailability, float]] = {}
        self._avail_locks: dict[tuple[str, int], asyncio.Lock] = {}

    async def provision(
        self,
        app_id: str,
        model: str,
        gpu: str,
        env: dict[str, str],
        gpu_count: int = 1,
        cloud_type: Optional[str] = None,
        container_disk_gb: Optional[int] = None,
        volume_gb: Optional[int] = None,
    ) -> "ProvisionResult":
        machine_id = f"m-rp-{uuid.uuid4().hex[:8]}"
        pod_name = f"{self.name_prefix}-{app_id}-{machine_id}"

        env_vars: dict[str, str] = {
            "APP_ID": app_id,
            "MACHINE_ID": machine_id,
            "MODEL_ID": model,
            "GATEWAY_URL": self.gateway_public_url,
            "REGISTRATION_TOKEN": env.get("REGISTRATION_TOKEN", ""),
            "WORKER_MODE": env.get("WORKER_MODE", "vllm"),
        }
        if self.ts_authkey:
            env_vars["TS_AUTHKEY"] = self.ts_authkey
        for k, v in env.items():
            if k in env_vars:
                continue
            env_vars[k] = v

        effective_cloud = (cloud_type or self.cloud_type).upper()
        effective_disk = container_disk_gb if container_disk_gb is not None else self.container_disk_in_gb
        effective_volume = volume_gb if volume_gb is not None else self.volume_in_gb
        body: dict[str, Any] = {
            "name": pod_name,
            "gpuTypeIds": [_map_gpu(gpu)],
            "cloudType": effective_cloud,
            "gpuCount": max(1, int(gpu_count)),
            "containerDiskInGb": effective_disk,
            "volumeInGb": effective_volume,
        }
        if self.reverse_tunnel:
            # Point the worker at the pod's own loopback (the tunnel forwards it to
            # the gateway+redis) and run the vLLM fleet from the venv we build on
            # the pod. The worker config is written over SSH at launch (NOT RunPod
            # env), so the only pod env needed is PUBLIC_KEY (which the stock
            # image's start.sh adds to authorized_keys so we can SSH in).
            env_vars["GATEWAY_URL"] = self._loop_gateway_url
            env_vars["WORKER_REDIS_URL"] = self._loop_redis_url
            mm = env_vars.get("MULTI_MODEL_CONFIG")
            if mm:
                try:
                    cfg = json.loads(mm)
                    cfg["venv_path"] = _POD_VENV  # where we install vllm on the pod
                    env_vars["MULTI_MODEL_CONFIG"] = json.dumps(cfg)
                except (ValueError, TypeError):
                    pass
            self._worker_env[machine_id] = dict(env_vars)
            body["imageName"] = self.image
            body["ports"] = ["22/tcp", "8000/tcp"]
            body["env"] = {"PUBLIC_KEY": self.ssh_pub} if self.ssh_pub else {}
            # Latest vLLM is a CUDA-13 build (its wheels need libcudart.so.13 /
            # driver 580+), so pin the host filter to a CUDA-13 driver — matching
            # the serverless default (RUNPOD_ALLOWED_CUDA_VERSIONS=13.0). The cu128
            # base image still runs here (forward-compatible: image-CUDA ≤ host-CUDA).
            # Override RUNPOD_TUNNEL_CUDA (e.g. "12.8") only if you also pin an older,
            # cu128-built vLLM via the member's vllm_version.
            body["allowedCudaVersions"] = [
                v.strip() for v in os.environ.get("RUNPOD_TUNNEL_CUDA", "13.0").split(",") if v.strip()
            ]
        else:
            body["templateId"] = self.template_id
            body["env"] = env_vars
            if self.allowed_cuda_versions:
                body["allowedCudaVersions"] = self.allowed_cuda_versions

        r = await self._client.post("/pods", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"RunPod provision failed: {r.status_code} {r.text}")
        data = r.json()
        pod_id = data.get("id")
        if not pod_id:
            raise RuntimeError(f"RunPod provision response missing id: {data}")

        self._pod_ids[machine_id] = pod_id

        # Bring up the reverse tunnel once the pod is RUNNING + SSH is reachable.
        # The worker retries registration, so it tolerates the tunnel arriving a
        # few seconds after it boots.
        if self.reverse_tunnel:
            asyncio.create_task(self._establish_tunnel(machine_id, pod_id))

        # RunPod's pod-create response carries the hourly rate it locked in
        # for this pod. Capture it best-effort so the UI can show a live cost
        # ticker; a missing/invalid value just leaves it None.
        cost_raw = data.get("costPerHr") or data.get("cost_per_hr")
        try:
            cost_per_hr = float(cost_raw) if cost_raw is not None else None
        except (TypeError, ValueError):
            cost_per_hr = None

        logger.info(
            "runpod-provision: app=%s gpu=%s → machine=%s pod=%s name=%s cost=%s/hr",
            app_id, gpu, machine_id, pod_id, pod_name, cost_per_hr,
        )
        return ProvisionResult(machine_id=machine_id, cost_per_hr=cost_per_hr)

    # ---- reverse-tunnel (mirrors the VM provider's VM_REVERSE_TUNNEL) --------

    async def _establish_tunnel(self, machine_id: str, pod_id: str) -> None:
        """Poll the pod until it's RUNNING + its SSH endpoint is up, then open a
        reverse tunnel forwarding the pod's loopback gateway/redis ports home."""
        from .compute import _extract_ssh  # reuse the portMappings/runtime.ports parser

        deadline = time.time() + 600
        ip = port = None
        while time.time() < deadline:
            try:
                r = await self._client.get(f"/pods/{pod_id}")
                if r.status_code == 200:
                    ip, port = _extract_ssh(r.json())
                    if ip and port:
                        break
            except Exception:  # noqa: BLE001 — transient while the pod boots
                pass
            await asyncio.sleep(5)
        if not (ip and port):
            logger.warning(
                "runpod reverse-tunnel: pod %s (machine %s) SSH endpoint not ready in 600s — "
                "worker can't phone home via the tunnel", pod_id, machine_id,
            )
            return
        self._ssh[machine_id] = (ip, int(port))
        logger.info("runpod reverse-tunnel: machine %s SSH at %s:%s", machine_id, ip, port)
        # 1) bring up the reverse tunnel (gateway+redis → pod loopback), then
        # 2) SSH in and launch the worker exactly like the VM provider (uv venv +
        #    install vllm + worker-agent + nohup). The worker phones home via the
        #    tunnel, so order matters: tunnel first.
        await self._ensure_tunnel_for(machine_id)
        try:
            await asyncio.to_thread(self._launch_worker_sync, machine_id, ip, int(port))
        except Exception:  # noqa: BLE001
            logger.exception("runpod: worker launch over SSH failed for machine %s", machine_id)

    def _launch_worker_sync(self, machine_id: str, ip: str, ssh_port: int) -> None:
        """SSH into the pod and set it up like a VM: write the worker config, ship
        the worker-agent tarball, make a uv venv, `uv pip install` vllm + the
        worker-agent, then nohup the worker. Runs in a thread (paramiko is sync)."""
        import paramiko
        import shlex

        from . import vm_probe
        from .vm_serverless_provider import _worker_agent_tarball_b64

        worker_env = dict(self._worker_env.get(machine_id) or {})
        if not worker_env:
            logger.warning("runpod: no stashed worker env for %s — skipping launch", machine_id)
            return
        cfg_path = f"{_POD_REMOTE}/worker-{machine_id}.json"
        log_path = f"{_POD_REMOTE}/worker-{machine_id}.log"
        pid_path = f"{_POD_REMOTE}/worker-{machine_id}.pid"
        worker_env = {
            **worker_env,
            "WORKER_SELF_LOG_PATH": log_path,
            "WORKER_ENGINE_PIDS_PATH": f"{_POD_REMOTE}/worker-{machine_id}.enginepids",
        }
        cfg_b64 = base64.b64encode(json.dumps(worker_env).encode()).decode()
        wa_b64 = _worker_agent_tarball_b64()  # raises if the gateway didn't bundle the source

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=ip, port=ssh_port, username="root",
            pkey=vm_probe._load_pkey(self.ssh_priv_pem),
            timeout=30, banner_timeout=30, auth_timeout=30,
            look_for_keys=False, allow_agent=False,
        )
        try:
            # vLLM version to pin in the pod venv (None → latest that resolves).
            vllm_ver = ""
            try:
                vllm_ver = (json.loads(worker_env.get("MULTI_MODEL_CONFIG") or "{}").get("vllm_version") or "")
            except (ValueError, TypeError):
                pass
            vllm_spec = f"vllm=={vllm_ver}" if vllm_ver else "vllm"
            script = (
                f"set -ex; mkdir -p {_POD_REMOTE}; "
                f"echo {cfg_b64} | base64 -d > {shlex.quote(cfg_path)}; "
                f"echo {wa_b64} | base64 -d > {_POD_REMOTE}/wa.tgz; "
                f"rm -rf {_POD_REMOTE}/worker-agent && mkdir -p {_POD_REMOTE}/worker-agent && "
                f"  tar xzf {_POD_REMOTE}/wa.tgz -C {_POD_REMOTE}/worker-agent; "
                # Log the host GPU + driver/CUDA into the -x trace (shows up if we fail).
                f"nvidia-smi || true; "
                # uv makes the big vllm install fast; refresh it (the stock image's may
                # be old) into ~/.local/bin, PATH'd first below.
                f"curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true; "
                f"export PATH=\"$HOME/.local/bin:$PATH\"; "
                f"command -v uv >/dev/null 2>&1 || pip install -q uv; "
                f"if [ ! -x {_POD_VENV}/bin/python ]; then uv venv {_POD_VENV} || python3 -m venv {_POD_VENV}; fi; "
                # vLLM + the worker-agent into the pod venv (the multi launcher runs
                # `{_POD_VENV}/bin/python -m vllm …`, set via MULTI_MODEL_CONFIG.venv_path).
                # Install vLLM with its NATIVE CUDA build (latest vLLM is cu13). The pod
                # is pinned to a CUDA-13 host, so vLLM's bundled torch + CUDA libs match
                # the driver — no torch-backend override (that mismatched vLLM's own
                # cu13 binary against cu128 torch → "libcudart.so.13 missing").
                # ninja + cmake: vLLM JIT-compiles CUDA kernels (cutlass/MoE/tilelang)
                # at model load and dies with "[Errno 2] ... 'ninja'" without them.
                f"uv pip install --python {_POD_VENV}/bin/python ninja cmake {shlex.quote(vllm_spec)} {_POD_REMOTE}/worker-agent; "
                # Fast-fail (before the slow model load) if torch can't see the GPU —
                # i.e. the installed CUDA build doesn't match the host driver. Use the
                # venv python (a `uv venv` ships no pip). nvidia-smi above shows the host
                # CUDA in the trace if this trips.
                f"{_POD_VENV}/bin/python -c \"import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)\"; "
                # venv/bin FIRST on PATH so the JIT compiler (ninja) and the cu13
                # nvcc (nvidia-cuda-nvcc, in the venv) are found by vLLM's kernel
                # build — running venv/bin/python directly does NOT add venv/bin to PATH.
                f"WORKER_CONFIG_FILE={cfg_path} PATH={_POD_VENV}/bin:$PATH "
                f"  nohup {_POD_VENV}/bin/python -m worker_agent.main "
                f"  > {log_path} 2>&1 & echo $! > {pid_path}"
            )
            stdin, stdout, stderr = client.exec_command(f"bash -lc {shlex.quote(script)}", timeout=900)
            rc = stdout.channel.recv_exit_status()
            if rc != 0:
                # `set -ex` traces every command to stderr; the real failure is at
                # the TAIL (after uv's resolve banner), so capture the end, not head.
                out = stdout.read().decode(errors="replace")
                err = stderr.read().decode(errors="replace")
                tail = (out + "\n--- stderr ---\n" + err)[-3000:]
                raise RuntimeError(f"runpod worker launch failed (rc={rc}):\n…{tail}")
            logger.info("runpod: worker launched on pod %s (machine %s)", ip, machine_id)
        finally:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

    async def _ensure_tunnel_for(self, machine_id: str) -> None:
        ep = self._ssh.get(machine_id)
        if not ep or not self.reverse_tunnel or not self.ssh_priv_pem:
            return
        ip, ssh_port = ep
        from . import vm_tunnel
        forwards = [
            vm_tunnel.Forward(self._tunnel_gw[1], self._tunnel_gw[0], self._tunnel_gw[1]),
            vm_tunnel.Forward(self._tunnel_redis[1], self._tunnel_redis[0], self._tunnel_redis[1]),
        ]
        await asyncio.to_thread(vm_tunnel.ensure, ip, ssh_port, "root", self.ssh_priv_pem, forwards)

    async def ensure_connectivity(self) -> None:
        """Each autoscaler tick: re-ensure every live pod's tunnel (heals drops /
        reconnects after a transient SSH failure). New pods get their tunnel from
        the provision-time task; this keeps the existing ones alive."""
        if not self.reverse_tunnel:
            return
        for mid in list(self._ssh.keys()):
            try:
                await self._ensure_tunnel_for(mid)
            except Exception:  # noqa: BLE001
                logger.exception("runpod reverse-tunnel: re-ensure failed for machine %s", mid)

    async def terminate(self, machine_id: str) -> None:
        self._worker_env.pop(machine_id, None)
        ep = self._ssh.pop(machine_id, None)
        if ep:
            from . import vm_tunnel
            try:
                await asyncio.to_thread(vm_tunnel.close, ep[0])
            except Exception:  # noqa: BLE001
                logger.warning("runpod reverse-tunnel: failed to close tunnel for %s", machine_id)
        pod_id = self._pod_ids.pop(machine_id, None)
        if pod_id is None:
            pod_id = await self._lookup_pod_id_by_machine_id(machine_id)
            if pod_id is None:
                logger.warning("runpod-terminate: no pod_id known for machine %s", machine_id)
                return

        r = await self._client.delete(f"/pods/{pod_id}")
        if r.status_code >= 400 and r.status_code != 404:
            raise RuntimeError(f"RunPod terminate failed: {r.status_code} {r.text}")
        logger.info("runpod-terminate: %s (pod=%s) torn down", machine_id, pod_id)

    async def list_machines(self) -> list[str]:
        out: list[str] = []
        r = await self._client.get("/pods")
        if r.status_code >= 400:
            raise RuntimeError(f"RunPod list_pods failed: {r.status_code} {r.text}")
        for pod in r.json() or []:
            name = pod.get("name", "")
            if not name.startswith(f"{self.name_prefix}-"):
                continue
            idx = name.find("m-rp-")
            if idx >= 0:
                machine_id = name[idx:]
                out.append(machine_id)
                self._pod_ids.setdefault(machine_id, pod["id"])
        return out

    async def list_machines_for_app(self, app_id: str) -> list[str]:
        out: list[str] = []
        prefix = f"{self.name_prefix}-{app_id}-"
        r = await self._client.get("/pods")
        if r.status_code >= 400:
            raise RuntimeError(f"RunPod list_pods failed: {r.status_code} {r.text}")
        for pod in r.json() or []:
            name = pod.get("name", "")
            if not name.startswith(prefix):
                continue
            idx = name.find("m-rp-")
            if idx >= 0:
                machine_id = name[idx:]
                out.append(machine_id)
                self._pod_ids.setdefault(machine_id, pod["id"])
        return out

    async def _lookup_pod_id_by_machine_id(self, machine_id: str) -> Optional[str]:
        r = await self._client.get("/pods")
        if r.status_code >= 400:
            return None
        for pod in r.json() or []:
            if machine_id in (pod.get("name", "") or ""):
                return pod["id"]
        return None

    async def check_availability(
        self,
        gpu: str,
        count: int,
        cloud_type: Optional[str] = None,
    ) -> GpuAvailability:
        """Query RunPod's GraphQL gpuTypes endpoint for live stock + price.

        RunPod's REST API (/v1/*) doesn't expose GPU listings — only GraphQL at
        api.runpod.io/graphql does. We POST a single query that returns
        stockStatus, availableGpuCounts, and lowestPrice for the requested
        (gpu, count, cloudType). Cached 30s with single-flight lock.

        `cloud_type` overrides the provider-level default per request, so the
        UI's COMMUNITY/SECURE toggle reflects in the badge.
        """
        effective_cloud = (cloud_type or self.cloud_type).upper()
        key = (gpu, count, effective_cloud)
        now = time.time()
        cached = self._avail_cache.get(key)
        if cached is not None and cached[1] > now:
            return cached[0]

        lock = self._avail_locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._avail_cache.get(key)
            if cached is not None and cached[1] > time.time():
                return cached[0]

            rp_gpu = _map_gpu(gpu)
            secure = effective_cloud == "SECURE"
            query = """
            query GpuTypes($id: String, $count: Int!, $secure: Boolean!) {
              gpuTypes(input: { id: $id }) {
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
                lowestPrice(input: { gpuCount: $count, secureCloud: $secure }) {
                  stockStatus
                  uninterruptablePrice
                  minimumBidPrice
                  availableGpuCounts
                }
              }
            }
            """
            variables = {"id": rp_gpu, "count": count, "secure": secure}
            try:
                r = await self._client.post(
                    "https://api.runpod.io/graphql",
                    json={"query": query, "variables": variables},
                )
            except Exception as e:
                logger.warning("runpod-gql request failed: %s", e)
                result = GpuAvailability(
                    gpu=gpu, count=count, available=None,
                    reason="Couldn't reach RunPod",
                )
                self._avail_cache[key] = (result, time.time() + 5.0)
                return result

            if r.status_code >= 400:
                logger.warning("runpod-gql %s: %s", r.status_code, r.text[:200])
                result = GpuAvailability(
                    gpu=gpu, count=count, available=None,
                    reason=f"RunPod GraphQL returned {r.status_code}",
                )
                self._avail_cache[key] = (result, time.time() + 5.0)
                return result

            payload = r.json() if r.content else {}
            if payload.get("errors"):
                msg = payload["errors"][0].get("message", "GraphQL error")[:120]
                logger.warning("runpod-gql errors: %s", payload["errors"])
                result = GpuAvailability(
                    gpu=gpu, count=count, available=None,
                    reason=f"RunPod: {msg}",
                )
                self._avail_cache[key] = (result, time.time() + 5.0)
                return result

            types = (payload.get("data") or {}).get("gpuTypes") or []
            if not types:
                result = GpuAvailability(
                    gpu=gpu, count=count, available=False,
                    reason=f"{gpu} not offered on RunPod",
                )
                self._avail_cache[key] = (result, time.time() + _AVAILABILITY_TTL_S)
                return result

            entry = types[0] if isinstance(types[0], dict) else {}
            # Ensure the GPU is offered on the cloud tier we use at all.
            if secure and entry.get("secureCloud") is False:
                result = GpuAvailability(
                    gpu=gpu, count=count, available=False,
                    reason=f"{gpu} not offered on RunPod SECURE",
                )
                self._avail_cache[key] = (result, time.time() + _AVAILABILITY_TTL_S)
                return result
            if (not secure) and entry.get("communityCloud") is False:
                result = GpuAvailability(
                    gpu=gpu, count=count, available=False,
                    reason=f"{gpu} not offered on RunPod COMMUNITY",
                )
                self._avail_cache[key] = (result, time.time() + _AVAILABILITY_TTL_S)
                return result

            lp = entry.get("lowestPrice") or {}
            stock = lp.get("stockStatus")
            counts = lp.get("availableGpuCounts") or []

            if stock in (None, "None") or (counts and count not in counts):
                reason = (
                    f"No host with ≥{count}× {gpu} on RunPod {effective_cloud}"
                    if counts
                    else f"No {gpu} in stock on RunPod {effective_cloud}"
                )
                result = GpuAvailability(
                    gpu=gpu, count=count, available=False, reason=reason,
                )
                self._avail_cache[key] = (result, time.time() + _AVAILABILITY_TTL_S)
                return result

            price = None
            for k in ("uninterruptablePrice", "minimumBidPrice"):
                v = lp.get(k)
                if v is not None:
                    try:
                        price = float(v)
                        break
                    except (TypeError, ValueError):
                        pass

            # RunPod GraphQL doesn't expose per-DC breakdown in this query; the
            # stockStatus value ("High"/"Medium"/"Low") goes into the regions
            # field as a coarse signal so the UI can surface it.
            regions = [f"stock:{stock}"] if stock else []

            result = GpuAvailability(
                gpu=gpu, count=count, available=True,
                cheapest_price_hr=price, regions=regions,
            )
            self._avail_cache[key] = (result, time.time() + _AVAILABILITY_TTL_S)
            return result

    async def shutdown(self) -> None:
        # Don't terminate pods on gateway shutdown — autoscaler decides scale-to-zero.
        if self._owns_client:
            await self._client.aclose()
