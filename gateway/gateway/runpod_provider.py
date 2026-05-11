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
import logging
import os
import time
import uuid
from typing import Any, Optional

import httpx

from .provider import GpuAvailability, Provider

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
    ) -> str:
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

        body: dict[str, Any] = {
            "name": pod_name,
            "templateId": self.template_id,
            "gpuTypeIds": [_map_gpu(gpu)],
            "cloudType": self.cloud_type,
            "gpuCount": max(1, int(gpu_count)),
            "containerDiskInGb": self.container_disk_in_gb,
            "volumeInGb": self.volume_in_gb,
            "env": env_vars,
        }
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
        logger.info(
            "runpod-provision: app=%s gpu=%s → machine=%s pod=%s name=%s",
            app_id, gpu, machine_id, pod_id, pod_name,
        )
        return machine_id

    async def terminate(self, machine_id: str) -> None:
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
