"""SSH probe for VM-type cloud providers.

Opens SSH to a bare-metal box and runs `nvidia-smi` to return the GPU
inventory. Paramiko is already a transitive dep via benchmaq, so we use it
in a worker thread rather than pulling in asyncssh.
"""
from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass, field

import paramiko

logger = logging.getLogger("gateway.vm_probe")

CONNECT_TIMEOUT_S = 10
COMMAND_TIMEOUT_S = 15


@dataclass
class VmProbeResult:
    ok: bool
    message: str
    gpus: list[str]
    gpu_count: int


@dataclass
class GpuInfo:
    index: int
    name: str
    mem_free_mib: int
    mem_total_mib: int
    util_pct: int


@dataclass
class VmAvailabilityResult:
    ok: bool
    message: str
    gpus: list[GpuInfo]
    checked_at: float


def _load_pkey(private_key: str) -> paramiko.PKey:
    """Try OpenSSH key formats until one parses. Paramiko can't auto-detect."""
    buf = private_key.strip() + "\n"
    # Build the candidate list dynamically — paramiko 4.0 dropped DSSKey, so a
    # static tuple referencing it raises AttributeError at module-load time
    # before any key gets tried.
    candidates: list[type[paramiko.PKey]] = []
    for attr in ("Ed25519Key", "ECDSAKey", "RSAKey", "DSSKey"):
        cls = getattr(paramiko, attr, None)
        if cls is not None:
            candidates.append(cls)
    last_err: Exception | None = None
    for cls in candidates:
        try:
            return cls.from_private_key(io.StringIO(buf))
        except paramiko.SSHException as e:
            last_err = e
            continue
    raise RuntimeError(f"unsupported private key format: {last_err}")


def _probe_sync(host: str, port: int, user: str, private_key: str) -> VmProbeResult:
    try:
        pkey = _load_pkey(private_key)
    except Exception as e:
        return VmProbeResult(ok=False, message=f"key parse failed: {e}", gpus=[], gpu_count=0)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host,
            port=port,
            username=user,
            pkey=pkey,
            timeout=CONNECT_TIMEOUT_S,
            banner_timeout=CONNECT_TIMEOUT_S,
            auth_timeout=CONNECT_TIMEOUT_S,
            look_for_keys=False,
            allow_agent=False,
        )
    except paramiko.AuthenticationException:
        return VmProbeResult(ok=False, message="authentication failed — check user + private key", gpus=[], gpu_count=0)
    except Exception as e:
        return VmProbeResult(ok=False, message=f"SSH connect failed: {e}", gpus=[], gpu_count=0)

    try:
        cmd = "nvidia-smi --query-gpu=name --format=csv,noheader"
        stdin, stdout, stderr = client.exec_command(cmd, timeout=COMMAND_TIMEOUT_S)
        rc = stdout.channel.recv_exit_status()
        out = stdout.read().decode(errors="replace").strip()
        err = stderr.read().decode(errors="replace").strip()
        if rc != 0:
            return VmProbeResult(
                ok=False,
                message=f"nvidia-smi exited {rc}: {err or 'no GPU detected'}",
                gpus=[],
                gpu_count=0,
            )
        gpus = [line.strip() for line in out.splitlines() if line.strip()]
        if not gpus:
            return VmProbeResult(ok=False, message="nvidia-smi returned no GPUs", gpus=[], gpu_count=0)
        return VmProbeResult(
            ok=True,
            message=f"connected · {len(gpus)} GPU{'s' if len(gpus) != 1 else ''} detected",
            gpus=gpus,
            gpu_count=len(gpus),
        )
    finally:
        try:
            client.close()
        except Exception:
            pass


async def probe_vm(host: str, port: int, user: str, private_key: str) -> VmProbeResult:
    return await asyncio.to_thread(_probe_sync, host, port, user, private_key)


def _availability_sync(host: str, port: int, user: str, private_key: str) -> VmAvailabilityResult:
    """Like _probe_sync but returns per-GPU memory + utilisation so the UI can
    show a runpod-style availability badge for VM providers."""
    import time as _time
    try:
        pkey = _load_pkey(private_key)
    except Exception as e:
        return VmAvailabilityResult(ok=False, message=f"key parse failed: {e}", gpus=[], checked_at=_time.time())

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host, port=port, username=user, pkey=pkey,
            timeout=CONNECT_TIMEOUT_S, banner_timeout=CONNECT_TIMEOUT_S,
            auth_timeout=CONNECT_TIMEOUT_S, look_for_keys=False, allow_agent=False,
        )
    except paramiko.AuthenticationException:
        return VmAvailabilityResult(ok=False, message="authentication failed", gpus=[], checked_at=_time.time())
    except Exception as e:
        return VmAvailabilityResult(ok=False, message=f"SSH connect failed: {e}", gpus=[], checked_at=_time.time())

    try:
        cmd = "nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits"
        _, stdout, stderr = client.exec_command(cmd, timeout=COMMAND_TIMEOUT_S)
        rc = stdout.channel.recv_exit_status()
        out = stdout.read().decode(errors="replace").strip()
        err = stderr.read().decode(errors="replace").strip()
        if rc != 0:
            return VmAvailabilityResult(
                ok=False,
                message=f"nvidia-smi exited {rc}: {err or 'no GPU detected'}",
                gpus=[], checked_at=_time.time(),
            )
        gpus: list[GpuInfo] = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                gpus.append(GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    mem_free_mib=int(parts[2]),
                    mem_total_mib=int(parts[3]),
                    util_pct=int(parts[4]),
                ))
            except ValueError:
                continue
        if not gpus:
            return VmAvailabilityResult(ok=False, message="nvidia-smi returned no parseable GPUs", gpus=[], checked_at=_time.time())
        return VmAvailabilityResult(
            ok=True,
            message=f"{len(gpus)} GPU{'s' if len(gpus) != 1 else ''} reachable",
            gpus=gpus,
            checked_at=_time.time(),
        )
    finally:
        try:
            client.close()
        except Exception:
            pass


async def availability_vm(host: str, port: int, user: str, private_key: str) -> VmAvailabilityResult:
    return await asyncio.to_thread(_availability_sync, host, port, user, private_key)


# --------------------------------------------------------------------------
# Live host metrics (CPU% + memory + per-GPU util/mem/temp) for the provider
# metrics page. One SSH round-trip; not persisted (the UI polls + graphs live).
# --------------------------------------------------------------------------
@dataclass
class GpuProc:
    """A process bound to a GPU (matched by CUDA_VISIBLE_DEVICES). `pid` is the
    container-namespace pid — the one `ps`/`kill` see on the box (NOT the host pid
    nvidia-smi reports, which differs under a container)."""
    pid: int
    comm: str
    cmd: str


@dataclass
class GpuMetric:
    index: int
    name: str
    util_pct: int
    mem_used_mib: int
    mem_total_mib: int
    temp_c: int
    processes: list[GpuProc] = field(default_factory=list)


@dataclass
class VmMetricsResult:
    ok: bool
    message: str
    cpu_pct: float          # overall busy %, -1 when unavailable
    mem_used_mib: int
    mem_total_mib: int
    gpus: list[GpuMetric]
    checked_at: float
    cpu_cores: list[float] = field(default_factory=list)  # per-core busy % (htop-style)


# Two /proc/stat samples (CPU%), /proc/meminfo (RAM), nvidia-smi (GPUs) — one shot.
# Two /proc/stat samples include the aggregate `cpu` line AND every per-core
# `cpuN` line, so we can report overall + per-core busy % (htop-style).
_METRICS_CMD = (
    "echo @@CPU1; grep '^cpu' /proc/stat; sleep 0.4; echo @@CPU2; grep '^cpu' /proc/stat; "
    "echo @@MEM; grep -E '^(MemTotal|MemAvailable):' /proc/meminfo; echo @@GPU; "
    "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu "
    "--format=csv,noheader,nounits 2>/dev/null; "
    # Per-GPU processes by CUDA_VISIBLE_DEVICES (container-namespace pids — the ones
    # ps/kill see; nvidia-smi's compute-apps pids are host-namespace under a
    # container and don't match). One line per GPU process: pid|cvd|comm|cmd.
    "echo @@PROC; "
    "for e in /proc/[0-9]*/environ; do "
    "p=${e%/environ}; p=${p#/proc/}; "
    "cvd=$(tr '\\0' '\\n' < \"$e\" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1); "
    "[ -z \"$cvd\" ] && continue; "
    "comm=$(cat /proc/$p/comm 2>/dev/null); "
    "cmd=$(tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-110); "
    "echo \"$p|$cvd|$comm|$cmd\"; "
    "done 2>/dev/null"
)


def _cpu_busy_total(line: str):
    """(busy, total) jiffies from a `/proc/stat` `cpu …` line; None if unparseable.
    busy excludes idle + iowait."""
    nums = [int(x) for x in line.split()[1:] if x.lstrip("-").isdigit()]
    if len(nums) < 4:
        return None
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0)  # idle + iowait
    total = sum(nums)
    return total - idle, total


def _metrics_sync(host: str, port: int, user: str, private_key: str) -> VmMetricsResult:
    import time as _time
    try:
        pkey = _load_pkey(private_key)
    except Exception as e:
        return VmMetricsResult(False, f"key parse failed: {e}", -1.0, 0, 0, [], _time.time())

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host, port=port, username=user, pkey=pkey,
            timeout=CONNECT_TIMEOUT_S, banner_timeout=CONNECT_TIMEOUT_S,
            auth_timeout=CONNECT_TIMEOUT_S, look_for_keys=False, allow_agent=False,
        )
    except paramiko.AuthenticationException:
        return VmMetricsResult(False, "authentication failed", -1.0, 0, 0, [], _time.time())
    except Exception as e:
        return VmMetricsResult(False, f"SSH connect failed: {e}", -1.0, 0, 0, [], _time.time())

    try:
        _, stdout, _ = client.exec_command(_METRICS_CMD, timeout=COMMAND_TIMEOUT_S)
        stdout.channel.recv_exit_status()
        out = stdout.read().decode(errors="replace")
        sec: dict[str, list[str]] = {}
        cur = None
        for ln in out.splitlines():
            s = ln.strip()
            if s in ("@@CPU1", "@@CPU2", "@@MEM", "@@GPU", "@@PROC"):
                cur = s
                sec[cur] = []
            elif cur:
                sec[cur].append(ln)

        # Aggregate + per-core CPU% from the two samples (delta busy / delta total).
        cpu_pct = -1.0
        cpu_cores: list[float] = []

        def _sample(lines):
            d = {}
            for ln in lines:
                name = ln.split(" ", 1)[0] if ln else ""
                if name.startswith("cpu"):
                    bt = _cpu_busy_total(ln)
                    if bt:
                        d[name] = bt
            return d

        def _pct(a, b):
            if a and b and (b[1] - a[1]) > 0:
                return round(100.0 * (b[0] - a[0]) / (b[1] - a[1]), 1)
            return None

        try:
            s1, s2 = _sample(sec.get("@@CPU1", [])), _sample(sec.get("@@CPU2", []))
            agg = _pct(s1.get("cpu"), s2.get("cpu"))
            if agg is not None:
                cpu_pct = agg
            cores = sorted((n for n in s2 if n != "cpu" and n[3:].isdigit()), key=lambda n: int(n[3:]))
            cpu_cores = [(_pct(s1.get(n), s2.get(n)) or 0.0) for n in cores]
        except Exception:  # noqa: BLE001
            pass

        mem_total_mib = mem_used_mib = 0
        try:
            kv = {}
            for ln in sec.get("@@MEM", []):
                k, _, v = ln.partition(":")
                kv[k.strip()] = int(v.strip().split()[0])  # kB
            if "MemTotal" in kv and "MemAvailable" in kv:
                mem_total_mib = kv["MemTotal"] // 1024
                mem_used_mib = (kv["MemTotal"] - kv["MemAvailable"]) // 1024
        except Exception:  # noqa: BLE001
            pass

        gpus: list[GpuMetric] = []
        for ln in sec.get("@@GPU", []):
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 6:
                continue
            try:
                gpus.append(GpuMetric(int(parts[0]), parts[1], int(parts[2]),
                                      int(parts[3]), int(parts[4]), int(parts[5])))
            except ValueError:
                continue

        # @@PROC: pid|CUDA_VISIBLE_DEVICES|comm|cmd → attach to the physical GPUs the
        # process can see (CVD value == physical index). One process can span GPUs (TP).
        proc_for: list[tuple[set[int], GpuProc]] = []
        for ln in sec.get("@@PROC", []):
            bits = ln.split("|", 3)
            if len(bits) < 4 or not bits[0].strip().isdigit():
                continue
            idxs = {int(t) for t in bits[1].split(",") if t.strip().isdigit()}
            if not idxs:
                continue
            proc_for.append((idxs, GpuProc(int(bits[0]), bits[2].strip(), bits[3].strip())))
        for g in gpus:
            g.processes = [gp for (idxs, gp) in proc_for if g.index in idxs]

        ok = mem_total_mib > 0 or bool(gpus) or cpu_pct >= 0
        return VmMetricsResult(
            ok, "ok" if ok else "no metrics parsed (is this a Linux host with nvidia-smi?)",
            cpu_pct, mem_used_mib, mem_total_mib, gpus, _time.time(), cpu_cores=cpu_cores,
        )
    finally:
        try:
            client.close()
        except Exception:
            pass


async def metrics_vm(host: str, port: int, user: str, private_key: str) -> VmMetricsResult:
    return await asyncio.to_thread(_metrics_sync, host, port, user, private_key)
