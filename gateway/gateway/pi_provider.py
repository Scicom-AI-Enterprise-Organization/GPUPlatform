"""Prime Intellect provider — calls the PI HTTP API to launch GPU pods.

Requires (env vars):
  PI_API_KEY              - bearer token from the PI dashboard
  PI_CUSTOM_TEMPLATE_ID   - id of a pre-baked PI template containing our
                            worker-agent (+ vllm). Templates are how PI lets
                            you bring your own image — there's no `userData`
                            field on pod-create, only env-var injection.
  GATEWAY_PUBLIC_URL      - the URL workers dial back to from PI pods (must
                            be reachable from PI's network)

Optional:
  PI_API_BASE             - default: https://api.primeintellect.ai
  PI_PROVIDER_TYPE        - default: runpod (PI re-sells from runpod, fluidstack,
                            lambdalabs, etc. — see PI dashboard for cloudIds)
  PI_CLOUD_ID             - default: "runpod" (validated working in smoke test).
                            PI rejects empty cloudId with "Field required".
  PI_GPU_SOCKET           - default: PCIe
  PI_DATA_CENTER_ID       - optional, pin a region
  PI_MAX_PRICE_HR         - optional, ceiling price per GPU-hour
  PI_NAME_PREFIX          - default: serverlessgpu — pods we own get this prefix
                            so list_machines() can filter them out from any
                            other pods on the same account.

Caveats:
  - Pods take ~30-90s to reach ACTIVE state. provision() returns the machine_id
    immediately; the worker registers asynchronously when its container starts.
  - GPU type strings: our app spec uses "H100"; PI uses "H100_80GB". A small
    mapping table here translates.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

import httpx

from .provider import GpuAvailability, Provider

logger = logging.getLogger("gateway.pi_provider")

_AVAILABILITY_TTL_S = 30.0


_GPU_NAME_MAP = {
    "H100": "H100_80GB",
    "A100": "A100_80GB",
    "A100-40G": "A100_40GB",
    "A10G": "A10",
    "A10": "A10",
    "L40S": "L40S",
    "L4": "L4",
    "RTX4090": "RTX4090_24GB",
    "RTX3090": "RTX3090_24GB",
}


def _map_gpu(name: str) -> str:
    """Normalize app-spec GPU strings to PI's gpuName enum."""
    if name in _GPU_NAME_MAP:
        return _GPU_NAME_MAP[name]
    return name  # assume caller already used PI's enum (e.g. "H100_80GB")


class PrimeIntellectProvider(Provider):
    name = "primeintellect"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        custom_template_id: Optional[str] = None,
        gateway_public_url: Optional[str] = None,
        provider_type: Optional[str] = None,
        cloud_id: Optional[str] = None,
        gpu_socket: Optional[str] = None,
        data_center_id: Optional[str] = None,
        max_price_hr: Optional[float] = None,
        name_prefix: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("PI_API_KEY")
        if not self.api_key:
            raise RuntimeError("PI_API_KEY env var or constructor arg required")

        self.api_base = (api_base or os.environ.get("PI_API_BASE", "https://api.primeintellect.ai")).rstrip("/")
        self.custom_template_id = custom_template_id or os.environ.get("PI_CUSTOM_TEMPLATE_ID")
        if not self.custom_template_id:
            raise RuntimeError(
                "PI_CUSTOM_TEMPLATE_ID env var required — bake a template "
                "containing our worker image and set its id here"
            )

        self.gateway_public_url = (
            gateway_public_url
            or os.environ.get("GATEWAY_PUBLIC_URL")
            or os.environ.get("GATEWAY_URL")
        )
        if not self.gateway_public_url:
            raise RuntimeError(
                "GATEWAY_PUBLIC_URL env var required so PI workers can reach the gateway"
            )

        self.provider_type = provider_type or os.environ.get("PI_PROVIDER_TYPE", "runpod")
        # PI rejects empty cloudId. "runpod" verified working against the live
        # API in our smoke test; override for other providers (fluidstack, etc).
        self.cloud_id = cloud_id or os.environ.get("PI_CLOUD_ID", "runpod")
        self.gpu_socket = gpu_socket or os.environ.get("PI_GPU_SOCKET", "PCIe")
        self.data_center_id = data_center_id or os.environ.get("PI_DATA_CENTER_ID")
        self.max_price_hr = max_price_hr if max_price_hr is not None else (
            float(os.environ.get("PI_MAX_PRICE_HR")) if os.environ.get("PI_MAX_PRICE_HR") else None
        )
        self.name_prefix = name_prefix or os.environ.get("PI_NAME_PREFIX", "serverlessgpu")

        self._client = client or httpx.AsyncClient(
            base_url=self.api_base,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0,
        )
        self._owns_client = client is None

        # local map machine_id (our id baked into pod name) → pod_id (PI's id).
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
        import uuid as _uuid

        machine_id = f"m-pi-{_uuid.uuid4().hex[:8]}"
        pod_name = f"{self.name_prefix}-{app_id}-{machine_id}"

        env_vars = [
            {"key": "APP_ID", "value": app_id},
            {"key": "MACHINE_ID", "value": machine_id},
            {"key": "MODEL_ID", "value": model},
            {"key": "GATEWAY_URL", "value": self.gateway_public_url},
            {"key": "REGISTRATION_TOKEN", "value": env.get("REGISTRATION_TOKEN", "")},
            {"key": "WORKER_MODE", "value": env.get("WORKER_MODE", "vllm")},
        ]
        for k, v in env.items():
            if k in {"APP_ID", "MACHINE_ID", "MODEL_ID", "GATEWAY_URL", "REGISTRATION_TOKEN", "WORKER_MODE"}:
                continue
            env_vars.append({"key": k, "value": v})

        body: dict[str, Any] = {
            "pod": {
                "name": pod_name,
                "cloudId": self.cloud_id,
                "gpuType": _map_gpu(gpu),
                "socket": self.gpu_socket,
                "gpuCount": max(1, int(gpu_count)),
                "customTemplateId": self.custom_template_id,
                "envVars": env_vars,
                "autoRestart": False,
            },
            "provider": {"type": self.provider_type},
        }
        if self.data_center_id:
            body["pod"]["dataCenterId"] = self.data_center_id
        if self.max_price_hr is not None:
            body["pod"]["maxPrice"] = self.max_price_hr

        r = await self._client.post("/api/v1/pods/", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"PI provision failed: {r.status_code} {r.text}")
        data = r.json()
        pod_id = data.get("id") or data.get("data", {}).get("id")
        if not pod_id:
            raise RuntimeError(f"PI provision response missing id: {data}")

        self._pod_ids[machine_id] = pod_id
        logger.info(
            "pi-provision: app=%s gpu=%s → machine=%s pod=%s name=%s",
            app_id, gpu, machine_id, pod_id, pod_name,
        )
        return machine_id

    async def terminate(self, machine_id: str) -> None:
        pod_id = self._pod_ids.pop(machine_id, None)
        if pod_id is None:
            # Fall back to lookup by name prefix in case we restarted and lost
            # the in-memory map. List then match.
            pod_id = await self._lookup_pod_id_by_machine_id(machine_id)
            if pod_id is None:
                logger.warning("pi-terminate: no pod_id known for machine %s", machine_id)
                return

        r = await self._client.delete(f"/api/v1/pods/{pod_id}")
        if r.status_code >= 400 and r.status_code != 404:
            raise RuntimeError(f"PI terminate failed: {r.status_code} {r.text}")
        logger.info("pi-terminate: %s (pod=%s) torn down", machine_id, pod_id)

    async def list_machines(self) -> list[str]:
        """List active pods owned by this gateway (filtered by name prefix)."""
        out: list[str] = []
        offset = 0
        while True:
            r = await self._client.get(
                "/api/v1/pods/", params={"offset": offset, "limit": 100}
            )
            if r.status_code >= 400:
                raise RuntimeError(f"PI list_pods failed: {r.status_code} {r.text}")
            page = r.json()
            data = page.get("data", [])
            for pod in data:
                name = pod.get("name", "")
                if not name.startswith(f"{self.name_prefix}-"):
                    continue
                # Pull machine_id back out of the name suffix
                # name format: {prefix}-{app_id}-{machine_id}
                parts = name.rsplit("-", 2)
                if len(parts) < 3:
                    continue
                machine_id = "-".join(parts[-2:]) if parts[-2].startswith("m") else parts[-1]
                # The simple parse: find "m-pi-XXXX" inside the name
                idx = name.find("m-pi-")
                if idx >= 0:
                    machine_id = name[idx:]
                out.append(machine_id)
                self._pod_ids.setdefault(machine_id, pod["id"])
            if len(data) < 100:
                break
            offset += 100
        return out

    async def get_pod_status(self, machine_id: str) -> Optional[dict[str, Any]]:
        """Useful for diagnostics: fetch the full pod record."""
        pod_id = self._pod_ids.get(machine_id) or await self._lookup_pod_id_by_machine_id(machine_id)
        if pod_id is None:
            return None
        r = await self._client.get(f"/api/v1/pods/{pod_id}")
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise RuntimeError(f"PI get_pod failed: {r.status_code} {r.text}")
        return r.json()

    async def _lookup_pod_id_by_machine_id(self, machine_id: str) -> Optional[str]:
        offset = 0
        while True:
            r = await self._client.get(
                "/api/v1/pods/", params={"offset": offset, "limit": 100}
            )
            if r.status_code >= 400:
                return None
            page = r.json()
            data = page.get("data", [])
            for pod in data:
                if machine_id in (pod.get("name", "") or ""):
                    return pod["id"]
            if len(data) < 100:
                break
            offset += 100
        return None

    async def check_availability(
        self,
        gpu: str,
        count: int,
        cloud_type: Optional[str] = None,  # ignored — PI has no community/secure split
    ) -> GpuAvailability:
        """Hit PI's /availability endpoint and parse the cheapest matching row.

        Cached for 30s per (gpu, count) to bound upstream RPS. Single-flighted
        with a per-key lock so concurrent UI checks coalesce into one API call.
        """
        key = (gpu, count)
        now = time.time()
        cached = self._avail_cache.get(key)
        if cached is not None and cached[1] > now:
            return cached[0]

        lock = self._avail_locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._avail_cache.get(key)
            if cached is not None and cached[1] > time.time():
                return cached[0]

            pi_gpu = _map_gpu(gpu)
            params: dict[str, Any] = {"gpu_count": count, "gpu_type": pi_gpu}
            if self.data_center_id:
                params["regions"] = self.data_center_id
            try:
                r = await self._client.get("/api/v1/availability/", params=params)
            except Exception as e:
                logger.warning("pi-availability request failed: %s", e)
                result = GpuAvailability(
                    gpu=gpu, count=count, available=None,
                    reason="Couldn't reach Prime Intellect",
                )
                self._avail_cache[key] = (result, time.time() + 5.0)
                return result

            if r.status_code >= 400:
                logger.warning("pi-availability %s: %s", r.status_code, r.text[:200])
                result = GpuAvailability(
                    gpu=gpu, count=count, available=None,
                    reason=f"PI availability check returned {r.status_code}",
                )
                self._avail_cache[key] = (result, time.time() + 5.0)
                return result

            payload = r.json()
            # PI's /availability response groups rows by gpu type. The shape
            # historically is {"<GPU_TYPE>": [row, row, ...]} or a flat list;
            # normalize to a flat list of dicts.
            rows: list[dict[str, Any]] = []
            if isinstance(payload, dict):
                if pi_gpu in payload and isinstance(payload[pi_gpu], list):
                    rows = payload[pi_gpu]
                elif "data" in payload and isinstance(payload["data"], list):
                    rows = payload["data"]
                else:
                    for v in payload.values():
                        if isinstance(v, list):
                            rows.extend(x for x in v if isinstance(x, dict))
            elif isinstance(payload, list):
                rows = [x for x in payload if isinstance(x, dict)]

            # Filter to rows that match our socket preference (PCIe vs SXM).
            socket_match = [
                row for row in rows
                if not self.gpu_socket
                or str(row.get("socket", "")).upper() == self.gpu_socket.upper()
                or row.get("socket") in (None, "")
            ]
            usable = socket_match or rows

            if not usable:
                result = GpuAvailability(
                    gpu=gpu, count=count, available=False,
                    reason=f"No {gpu}×{count} in stock on Prime Intellect right now",
                )
                self._avail_cache[key] = (result, time.time() + _AVAILABILITY_TTL_S)
                return result

            prices: list[float] = []
            for row in usable:
                for k in ("prices", "price"):
                    v = row.get(k)
                    if isinstance(v, dict):
                        for kk in ("onDemand", "on_demand", "communityPrice", "price"):
                            if kk in v:
                                try:
                                    prices.append(float(v[kk]))
                                    break
                                except (TypeError, ValueError):
                                    pass
                    elif v is not None:
                        try:
                            prices.append(float(v))
                        except (TypeError, ValueError):
                            pass
            cheapest = min(prices) if prices else None

            regions: list[str] = []
            for row in usable:
                for k in ("dataCenter", "data_center", "region", "country"):
                    v = row.get(k)
                    if isinstance(v, str) and v and v not in regions:
                        regions.append(v)
                        break
            if cheapest is None and not regions:
                logger.info("pi-availability: %d rows but no parseable price/region (sample=%s)", len(usable), usable[0] if usable else None)

            result = GpuAvailability(
                gpu=gpu, count=count, available=True,
                cheapest_price_hr=cheapest, regions=regions,
            )
            self._avail_cache[key] = (result, time.time() + _AVAILABILITY_TTL_S)
            return result

    async def shutdown(self) -> None:
        # Don't terminate pods on gateway shutdown — that would scale-to-zero
        # everything just because we restarted. Terminate only when the
        # autoscaler explicitly says so.
        if self._owns_client:
            await self._client.aclose()
