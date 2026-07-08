"""SSH probe for VM-type cloud providers.

Opens SSH to a bare-metal box and runs `nvidia-smi` (or `npu-smi` on Huawei
Ascend boxes) to return the accelerator inventory. Paramiko is already a
transitive dep via benchmaq, so we use it in a worker thread rather than
pulling in asyncssh.

Connections support key or password auth, and an optional jump host
(ProxyJump equivalent): the jump client opens a direct-tcpip channel to the
target and the target client connects through it. Needed for e.g. the TM
Huawei NPU boxes, which are only reachable via ssh.tma01.gpu.tm.com.my.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
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


def _auth_kwargs(private_key: str | None, password: str | None, hop: str) -> dict:
    """Paramiko connect() auth kwargs for one hop: key when given, else password."""
    if private_key and private_key.strip():
        try:
            return {"pkey": _load_pkey(private_key)}
        except Exception as e:
            raise RuntimeError(f"{hop} key parse failed: {e}") from e
    if password:
        return {"password": password}
    raise RuntimeError(f"{hop} needs a private key or a password")


def _close_quiet(*clients: paramiko.SSHClient | None) -> None:
    for c in clients:
        try:
            if c is not None:
                c.close()
        except Exception:
            pass


def _connect(host: str, port: int, user: str,
             private_key: str | None = None, password: str | None = None,
             jump_host: str | None = None, jump_port: int = 22,
             jump_user: str | None = None, jump_private_key: str | None = None,
             jump_password: str | None = None,
             ) -> tuple[paramiko.SSHClient, paramiko.SSHClient | None]:
    """Open SSH to the target, optionally through a jump host. Returns
    (client, jump_client) — the jump client must stay open for the target's
    channel to live; close both with _close_quiet(). Raises RuntimeError with
    a user-facing message on any failure."""
    common = dict(timeout=CONNECT_TIMEOUT_S, banner_timeout=CONNECT_TIMEOUT_S,
                  auth_timeout=CONNECT_TIMEOUT_S, look_for_keys=False, allow_agent=False)
    tkw = _auth_kwargs(private_key, password, "target")  # fail fast, pre-network

    sock = None
    jump_client: paramiko.SSHClient | None = None
    if jump_host and jump_host.strip():
        jkw = _auth_kwargs(jump_private_key, jump_password, "jump host")
        jump_client = paramiko.SSHClient()
        jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            jump_client.connect(hostname=jump_host.strip(), port=int(jump_port or 22),
                                username=(jump_user or "root").strip(), **common, **jkw)
            transport = jump_client.get_transport()
            if transport is None:
                raise RuntimeError("no transport after connect")
            sock = transport.open_channel(
                "direct-tcpip", (host, port), ("127.0.0.1", 0), timeout=CONNECT_TIMEOUT_S,
            )
        except paramiko.AuthenticationException:
            _close_quiet(jump_client)
            raise RuntimeError("jump host authentication failed — check jump user + credentials") from None
        except RuntimeError:
            _close_quiet(jump_client)
            raise
        except Exception as e:
            _close_quiet(jump_client)
            raise RuntimeError(f"jump host SSH connect failed: {e}") from e

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=host, port=port, username=user, sock=sock, **common, **tkw)
    except paramiko.AuthenticationException:
        _close_quiet(client, jump_client)
        raise RuntimeError("authentication failed — check user + credentials") from None
    except Exception as e:
        _close_quiet(client, jump_client)
        raise RuntimeError(f"SSH connect failed: {e}") from e
    return client, jump_client


def _probe_sync(host: str, port: int, user: str, private_key: str | None = None,
                password: str | None = None,
                jump_host: str | None = None, jump_port: int = 22,
                jump_user: str | None = None, jump_private_key: str | None = None,
                jump_password: str | None = None) -> VmProbeResult:
    try:
        client, jump = _connect(host, port, user, private_key, password,
                                jump_host, jump_port, jump_user, jump_private_key, jump_password)
    except RuntimeError as e:
        return VmProbeResult(ok=False, message=str(e), gpus=[], gpu_count=0)

    try:
        # nvidia-smi first; Huawei Ascend boxes have npu-smi instead — its table
        # output after the @@NPU marker is parsed by _parse_npu_info.
        cmd = ("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null; "
               "echo @@NPU; command -v npu-smi >/dev/null 2>&1 && npu-smi info 2>/dev/null")
        stdin, stdout, stderr = client.exec_command(cmd, timeout=COMMAND_TIMEOUT_S)
        stdout.channel.recv_exit_status()
        out = stdout.read().decode(errors="replace")
        head, _, npu_part = out.partition("@@NPU")
        gpus = [line.strip() for line in head.splitlines() if line.strip()]
        if gpus:
            return VmProbeResult(
                ok=True,
                message=f"connected · {len(gpus)} GPU{'s' if len(gpus) != 1 else ''} detected",
                gpus=gpus,
                gpu_count=len(gpus),
            )
        chips, _procs = _parse_npu_info(npu_part.splitlines())
        if chips:
            names = [c.name for c in chips]
            return VmProbeResult(
                ok=True,
                message=f"connected · {len(names)} NPU{'s' if len(names) != 1 else ''} detected",
                gpus=names,
                gpu_count=len(names),
            )
        return VmProbeResult(ok=False, message="no accelerators detected (nvidia-smi / npu-smi)", gpus=[], gpu_count=0)
    finally:
        _close_quiet(client, jump)


async def probe_vm(host: str, port: int, user: str, private_key: str | None = None,
                   password: str | None = None,
                   jump_host: str | None = None, jump_port: int = 22,
                   jump_user: str | None = None, jump_private_key: str | None = None,
                   jump_password: str | None = None) -> VmProbeResult:
    return await asyncio.to_thread(_probe_sync, host, port, user, private_key, password,
                                   jump_host, jump_port, jump_user, jump_private_key, jump_password)


def _availability_sync(host: str, port: int, user: str, private_key: str | None = None,
                       password: str | None = None,
                       jump_host: str | None = None, jump_port: int = 22,
                       jump_user: str | None = None, jump_private_key: str | None = None,
                       jump_password: str | None = None) -> VmAvailabilityResult:
    """Like _probe_sync but returns per-GPU memory + utilisation so the UI can
    show a runpod-style availability badge for VM providers."""
    import time as _time
    try:
        client, jump = _connect(host, port, user, private_key, password,
                                jump_host, jump_port, jump_user, jump_private_key, jump_password)
    except RuntimeError as e:
        return VmAvailabilityResult(ok=False, message=str(e), gpus=[], checked_at=_time.time())

    try:
        cmd = ("nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu "
               "--format=csv,noheader,nounits 2>/dev/null; "
               "echo @@NPU; command -v npu-smi >/dev/null 2>&1 && npu-smi info 2>/dev/null")
        _, stdout, stderr = client.exec_command(cmd, timeout=COMMAND_TIMEOUT_S)
        stdout.channel.recv_exit_status()
        out = stdout.read().decode(errors="replace")
        head, _, npu_part = out.partition("@@NPU")
        gpus: list[GpuInfo] = []
        for line in head.splitlines():
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
        kind = "GPU"
        if not gpus:
            chips, _procs = _parse_npu_info(npu_part.splitlines())
            gpus = [GpuInfo(
                index=c.index, name=c.name,
                mem_free_mib=max(0, c.mem_total_mib - c.mem_used_mib),
                mem_total_mib=c.mem_total_mib, util_pct=c.util_pct,
            ) for c in chips]
            kind = "NPU"
        if not gpus:
            return VmAvailabilityResult(ok=False, message="no parseable accelerators (nvidia-smi / npu-smi)", gpus=[], checked_at=_time.time())
        return VmAvailabilityResult(
            ok=True,
            message=f"{len(gpus)} {kind}{'s' if len(gpus) != 1 else ''} reachable",
            gpus=gpus,
            checked_at=_time.time(),
        )
    finally:
        _close_quiet(client, jump)


async def availability_vm(host: str, port: int, user: str, private_key: str | None = None,
                          password: str | None = None,
                          jump_host: str | None = None, jump_port: int = 22,
                          jump_user: str | None = None, jump_private_key: str | None = None,
                          jump_password: str | None = None) -> VmAvailabilityResult:
    return await asyncio.to_thread(_availability_sync, host, port, user, private_key, password,
                                   jump_host, jump_port, jump_user, jump_private_key, jump_password)


# --------------------------------------------------------------------------
# Live host metrics (CPU% + memory + per-GPU util/mem/temp) for the provider
# metrics page. One SSH round-trip; not persisted (the UI polls + graphs live).
# --------------------------------------------------------------------------
@dataclass
class GpuProc:
    """A process using a GPU. Sourced from `nvidia-smi` compute-apps (authoritative,
    lists EVERY owner's process with its VRAM) and enriched with /proc/<pid>/cmdline
    (world-readable, no root). `pid` is this box's pid namespace — killable on bare
    metal; host-namespace (may not match ps/kill) under a container.
    `gpu_mem_mib` is the process's VRAM on that GPU."""
    pid: int
    comm: str
    cmd: str
    gpu_mem_mib: int = 0
    gpus: str = ""   # for /proc-discovered host procs: GPU device indices it has open


@dataclass
class GpuMetric:
    index: int
    name: str
    util_pct: int
    mem_used_mib: int
    mem_total_mib: int
    temp_c: int
    # PCIe link (0 = unknown). `cur` is live — GPUs downclock the link when idle,
    # so a Gen4 card can read Gen1 ×16 at rest and Gen4 ×16 under load.
    pcie_gen_cur: int = 0
    pcie_width_cur: int = 0
    pcie_gen_max: int = 0
    pcie_width_max: int = 0
    # NVLink: active link count + aggregate per-direction bandwidth (GB/s).
    # supported=False on cards/boxes without NVLink (PCIe-only).
    nvlink_supported: bool = False
    nvlink_active: int = 0
    nvlink_gbps: float = 0.0
    # Huawei Ascend (npu-smi) fallback: kind="npu", util_pct=AICore%, mem=HBM.
    # power_w/health come from the npu-smi table (nvidia path leaves them unset).
    kind: str = "gpu"
    power_w: float = 0.0
    health: str = ""
    processes: list[GpuProc] = field(default_factory=list)


@dataclass
class DiskMetric:
    mount: str
    used_bytes: int
    total_bytes: int


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
    disks: list[DiskMetric] = field(default_factory=list)  # real filesystems (df), largest first
    # GPU processes discovered from /proc (cmdline world-readable) — catches the ones
    # NVML can't name (host / other-container pids) + frameworks whose fds we can't
    # read across the namespace. Not mapped to a specific GPU (no fd/NVML link), so
    # listed host-level alongside the per-GPU NVML VRAM.
    host_procs: list[GpuProc] = field(default_factory=list)


def _parse_npu_info(lines: list[str]) -> tuple[list[GpuMetric], dict[int, list[tuple[int, int, str]]]]:
    """Parse Huawei `npu-smi info` (Ascend) table output. Returns (chips, procs):
    chips — GpuMetric list (kind="npu"; util_pct=AICore%, mem=HBM when the card has
    it else DDR, power/temp/health from the table); procs — {npu_id: [(pid, mem_mib,
    name)]} from the trailing process table. Chip rows come in pairs:
      | 0     910B3 | OK           | 90.8   34   0 / 0        |  ← id+name | health | power temp hugepages
      | 0           | 0000:C1:00.0 | 0    0 / 0  54388/ 65536 |  ← chip | bus-id | AICore% DDR HBM
    """
    chips: list[GpuMetric] = []
    procs: dict[int, list[tuple[int, int, str]]] = {}
    pending: GpuMetric | None = None
    in_procs = False
    for ln in lines:
        s = ln.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        joined = " ".join(cells).lower()
        if "process id" in joined:
            in_procs = True
            continue
        if "health" in joined or "aicore" in joined or joined.startswith("npu-smi"):
            continue  # header / banner rows
        if in_procs:
            # | NPU Chip | pid | pname | mem(MB) | — "No running processes …" rows
            # don't have a numeric pid cell and fall through.
            if len(cells) >= 4 and cells[1].isdigit():
                m = re.match(r"(\d+)", cells[0])
                if m:
                    mem = int(cells[3]) if cells[3].isdigit() else 0
                    procs.setdefault(int(m.group(1)), []).append((int(cells[1]), mem, cells[2]))
            continue
        if len(cells) < 3:
            continue
        if pending is None:
            m = re.match(r"(\d+)\s+(\S.*)$", cells[0])
            if not m:
                continue
            g = GpuMetric(index=int(m.group(1)), name=f"Ascend {m.group(2).strip()}",
                          util_pct=0, mem_used_mib=0, mem_total_mib=0, temp_c=0,
                          kind="npu", health=cells[1])
            toks = cells[2].split()
            try:
                g.power_w = float(toks[0])
            except (ValueError, IndexError):
                pass
            try:
                g.temp_c = int(float(toks[1]))
            except (ValueError, IndexError):
                pass
            pending = g
        else:
            g, pending = pending, None
            toks = cells[2].split()
            try:
                g.util_pct = int(float(toks[0]))
            except (ValueError, IndexError):
                pass
            # "used / total" pairs: DDR first, HBM second on HBM cards — prefer
            # the last pair with a non-zero total (910x report DDR as 0/0).
            pairs = re.findall(r"(\d+)\s*/\s*(\d+)", cells[2])
            chosen = None
            for used, total in pairs:
                if int(total) > 0:
                    chosen = (int(used), int(total))
            if chosen is None and pairs:
                chosen = (int(pairs[0][0]), int(pairs[0][1]))
            if chosen:
                g.mem_used_mib, g.mem_total_mib = chosen
            chips.append(g)
    return chips, procs


# Two /proc/stat samples (CPU%), /proc/meminfo (RAM), nvidia-smi (GPUs) — one shot.
# Two /proc/stat samples include the aggregate `cpu` line AND every per-core
# `cpuN` line, so we can report overall + per-core busy % (htop-style).
_METRICS_CMD = (
    "echo @@CPU1; grep '^cpu' /proc/stat; sleep 0.4; echo @@CPU2; grep '^cpu' /proc/stat; "
    "echo @@MEM; grep -E '^(MemTotal|MemAvailable):' /proc/meminfo; echo @@GPU; "
    "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
    "pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max "
    "--format=csv,noheader,nounits 2>/dev/null; "
    # NVLink per-link status (one block per GPU; "<inactive>" links are skipped).
    "echo @@NVLINK; nvidia-smi nvlink --status 2>/dev/null; "
    # GPU processes — two complementary sources, merged in the parser:
    #   @@PROC: nvidia-smi --query-compute-apps (NVML — the SAME source nvtop uses;
    #     lists EVERY GPU process with its VRAM, including host / other-container
    #     tenants on a shared box). nvidia-smi's *human* process table is empty in a
    #     container, but the query interface still returns them. We map gpu_uuid→
    #     index (@@GPUUUID) and /proc-enrich the command where the pid is in our ns
    #     (foreign pids → process_name fallback, command unobtainable from in here).
    #   @@FDPROC: /proc/<pid>/fd open /dev/nvidiaN — THIS container's own GPU procs,
    #     to catch any the NVML list missed and supply the full command.
    "echo @@GPUUUID; nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null; "
    "echo @@PROC; "
    "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits 2>/dev/null | "
    "while IFS=, read u pid mem pname; do "
    "u=$(echo $u | tr -d ' '); pid=$(echo $pid | tr -d ' '); mem=$(echo $mem | tr -d ' '); pname=$(echo $pname | sed 's/^ *//'); "
    "[ -z \"$pid\" ] && continue; "
    "comm=$(cat /proc/$pid/comm 2>/dev/null); [ -z \"$comm\" ] && comm=\"$pname\"; "
    "cmd=$(tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-300); [ -z \"$cmd\" ] && cmd=\"$pname\"; "
    "echo \"$u|$pid|$mem|$comm|$cmd\"; done; "
    # @@FDPROC: every process in /proc that looks GPU-related — either it holds a
    # /dev/nvidiaN fd (readable for our own procs), OR its cmdline names a GPU engine
    # (vllm/sglang/…) whose fds we can't read across the container boundary. cmdline
    # is world-readable, so this surfaces e.g. a `vllm serve :8000` even when NVML
    # only knows it by an unmappable host pid. Line: pid|gpus(empty if fd unreadable)|comm|cmd.
    "echo @@FDPROC; "
    "for d in /proc/[0-9]*; do p=${d#/proc/}; "
    "cmd=$(tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-300); "
    "[ -z \"$cmd\" ] && continue; "
    "g=$(ls -l /proc/$p/fd 2>/dev/null | grep -oE '/dev/(nvidia|davinci)[0-9]+' | grep -oE '[0-9]+$' | sort -un | tr '\\n' ',' | sed 's/,$//'); "
    "gpu=0; [ -n \"$g\" ] && gpu=1; "
    "case \"$cmd\" in *vllm*|*VLLM*|*sglang*|*SGLang*|*tensorrt*|*trtllm*|*deepspeed*) gpu=1;; esac; "
    "[ \"$gpu\" = 0 ] && continue; "
    "comm=$(cat /proc/$p/comm 2>/dev/null); "
    "echo \"$p|$g|$comm|$cmd\"; done 2>/dev/null; "
    # Huawei Ascend fallback (no nvidia-smi on the box): the raw `npu-smi info`
    # table — chips + the NPU process table — parsed gateway-side by
    # _parse_npu_info. Captured once into $npu_out; @@NPUPROCCMD re-reads it to
    # resolve each pid's real command from /proc (bare metal → world-readable).
    "echo @@NPU; if command -v npu-smi >/dev/null 2>&1; then npu_out=$(npu-smi info 2>/dev/null); echo \"$npu_out\"; fi; "
    "echo @@NPUPROCCMD; if [ -n \"$npu_out\" ]; then "
    "echo \"$npu_out\" | awk -F'|' 'NF>=6 {p=$3; gsub(/ /,\"\",p); if (p ~ /^[0-9]+$/) print p}' | sort -u | "
    "while read p; do comm=$(cat /proc/$p/comm 2>/dev/null); "
    "cmd=$(tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-300); "
    "echo \"$p|$comm|$cmd\"; done; fi; "
    # Disk: real filesystems only (skip virtual mounts), bytes used + total.
    # Columns come out in --output order: <mount> <used> <size>.
    "echo @@DISK; df -B1 -x tmpfs -x devtmpfs -x overlay -x squashfs "
    "--output=target,used,size 2>/dev/null"
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


def _metrics_sync(host: str, port: int, user: str, private_key: str | None = None,
                  password: str | None = None,
                  jump_host: str | None = None, jump_port: int = 22,
                  jump_user: str | None = None, jump_private_key: str | None = None,
                  jump_password: str | None = None) -> VmMetricsResult:
    import time as _time
    try:
        client, jump = _connect(host, port, user, private_key, password,
                                jump_host, jump_port, jump_user, jump_private_key, jump_password)
    except RuntimeError as e:
        return VmMetricsResult(False, str(e), -1.0, 0, 0, [], _time.time())

    try:
        _, stdout, _ = client.exec_command(_METRICS_CMD, timeout=COMMAND_TIMEOUT_S)
        stdout.channel.recv_exit_status()
        out = stdout.read().decode(errors="replace")
        sec: dict[str, list[str]] = {}
        cur = None
        for ln in out.splitlines():
            s = ln.strip()
            if s in ("@@CPU1", "@@CPU2", "@@MEM", "@@GPU", "@@NVLINK", "@@GPUUUID", "@@PROC", "@@FDPROC", "@@NPU", "@@NPUPROCCMD", "@@DISK"):
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
                g = GpuMetric(int(parts[0]), parts[1], int(parts[2]),
                              int(parts[3]), int(parts[4]), int(parts[5]))
            except ValueError:
                continue
            # PCIe gen/width (current + max). "[N/A]" on unsupported → left at 0.
            if len(parts) >= 10:
                try:
                    g.pcie_gen_cur = int(parts[6])
                    g.pcie_width_cur = int(parts[7])
                    g.pcie_gen_max = int(parts[8])
                    g.pcie_width_max = int(parts[9])
                except ValueError:
                    pass
            gpus.append(g)

        # @@NVLINK: blocks of "GPU N: …" then "Link K: X GB/s" (or "<inactive>").
        # Aggregate active link count + total per-direction bandwidth per GPU.
        nvlink: dict[int, tuple[int, float]] = {}
        cur_idx: int | None = None
        for ln in sec.get("@@NVLINK", []):
            s = ln.strip()
            mg = re.match(r"GPU (\d+):", s)
            if mg:
                cur_idx = int(mg.group(1))
                nvlink.setdefault(cur_idx, (0, 0.0))
                continue
            if cur_idx is None:
                continue
            ml = re.search(r"Link \d+:\s*([0-9.]+)\s*GB/s", s)
            if ml:
                cnt, tot = nvlink[cur_idx]
                nvlink[cur_idx] = (cnt + 1, tot + float(ml.group(1)))
        for g in gpus:
            if g.index in nvlink:
                cnt, tot = nvlink[g.index]
                g.nvlink_supported = True
                g.nvlink_active = cnt
                g.nvlink_gbps = round(tot, 1)

        # @@GPUUUID: "index, uuid" → map so compute-apps (keyed by gpu_uuid) lands on
        # the right physical GPU.
        uuid2idx: dict[str, int] = {}
        for ln in sec.get("@@GPUUUID", []):
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                uuid2idx[parts[1]] = int(parts[0])

        # @@PROC: uuid|pid|mem_mib|comm|cmd from nvidia-smi compute-apps (NVML — every
        # GPU process + VRAM, all owners). Foreign (host/other-container) pids carry
        # VRAM but no resolvable command (process_name fallback). Authoritative.
        procs_by_gpu: dict[int, list[GpuProc]] = {}
        seen: dict[int, set[int]] = {}
        for ln in sec.get("@@PROC", []):
            bits = ln.split("|", 4)
            if len(bits) < 5 or not bits[1].strip().isdigit():
                continue
            idx = uuid2idx.get(bits[0].strip())
            if idx is None:
                continue
            mem = int(bits[2]) if bits[2].strip().isdigit() else 0
            pid, comm = int(bits[1]), (bits[3].strip() or "?")
            cmd = bits[4].strip() or comm
            # nvidia-smi can't resolve a host / other-container pid's name from inside
            # this container — make that legible instead of the raw "[Not Found]".
            if comm in ("[Not Found]", "[Insufficient Permissions]") or not comm:
                comm = "foreign pid"
                cmd = "command not visible from this container (host / other tenant)"
            procs_by_gpu.setdefault(idx, []).append(GpuProc(pid, comm, cmd, mem))
            seen.setdefault(idx, set()).add(pid)

        # @@NPU: Huawei Ascend fallback — no nvidia-smi on the box. Chips map into
        # GpuMetric (kind="npu": util=AICore%, mem=HBM, power/temp/health from the
        # table); the npu-smi process table (pid + device memory) seeds procs_by_gpu,
        # with each pid's real command resolved via @@NPUPROCCMD (pid|comm|cmd).
        if not gpus:
            npu_chips, npu_proc_rows = _parse_npu_info(sec.get("@@NPU", []))
            if npu_chips:
                gpus = npu_chips
                cmd_by_pid: dict[int, tuple[str, str]] = {}
                for ln in sec.get("@@NPUPROCCMD", []):
                    bits = ln.split("|", 2)
                    if len(bits) == 3 and bits[0].strip().isdigit():
                        cmd_by_pid[int(bits[0])] = (bits[1].strip(), bits[2].strip())
                for idx, rows in npu_proc_rows.items():
                    for pid, mem, pname in rows:
                        comm, cmdl = cmd_by_pid.get(pid, ("", ""))
                        procs_by_gpu.setdefault(idx, []).append(
                            GpuProc(pid, comm or pname, cmdl or pname, mem))
                        seen.setdefault(idx, set()).add(pid)

        # @@FDPROC: pid|gpus|comm|cmd — GPU processes discovered from /proc (full
        # command + container pid → killable). Merge into each GPU so every card lists
        # pid + command, not just the NVML host pids:
        #  - device-holders (fd readable → their /dev/nvidiaN list) → those GPUs.
        #  - framework servers whose fds we can't read across the namespace (vllm /
        #    sglang, gpus="") → the GPUs carrying a heavy unnamed foreign allocation
        #    (≥10 GiB), where a multi-GPU server is the obvious tenant. Best-effort:
        #    there's no fd/NVML pid bridge inside the container, so this correlates the
        #    NVML VRAM (host pid, no name) with the /proc command (container pid).
        # NPU mode: npu-smi's process table is complete (bare metal, no pid-namespace
        # gap), and Ascend procs hold /dev/davinci_manager — never /dev/davinciN — so
        # fds can't map them to a device; the heavy-fallback would just attach every
        # proc to every busy NPU. Skip the /proc merge entirely.
        npu_mode = bool(gpus) and gpus[0].kind == "npu"
        heavy = set() if npu_mode else {idx for idx, ps in procs_by_gpu.items()
                                        if any(p.gpu_mem_mib >= 10240 for p in ps)}
        for ln in ([] if npu_mode else sec.get("@@FDPROC", [])):
            bits = ln.split("|", 3)
            if len(bits) < 4 or not bits[0].strip().isdigit():
                continue
            pid = int(bits[0])
            dev = {int(t) for t in bits[1].split(",") if t.strip().isdigit()}
            comm = bits[2].strip() or "?"
            cmd = bits[3].strip() or comm
            for idx in (dev or heavy):
                if pid in seen.get(idx, set()):
                    continue
                procs_by_gpu.setdefault(idx, []).append(GpuProc(pid, comm, cmd, 0))
                seen.setdefault(idx, set()).add(pid)
        host_procs: list[GpuProc] = []  # merged into per-GPU lists above
        for g in gpus:
            # Heaviest VRAM first — the GPU's real tenants (NVML) on top, then the
            # /proc-named procs (0 VRAM here) beneath.
            g.processes = sorted(procs_by_gpu.get(g.index, []), key=lambda p: -p.gpu_mem_mib)

        # @@DISK: `df` rows "<mount> <used> <total>" (bytes). First line is the
        # header (non-numeric) → skipped. Drop sub-1 GiB mounts (boot/efi/…).
        disks: list[DiskMetric] = []
        for ln in sec.get("@@DISK", []):
            parts = ln.split()
            if len(parts) < 3:
                continue
            try:
                used, total = int(parts[-2]), int(parts[-1])
            except ValueError:
                continue
            mount = " ".join(parts[:-2])
            if total >= (1 << 30):
                disks.append(DiskMetric(mount=mount, used_bytes=used, total_bytes=total))
        disks.sort(key=lambda d: d.total_bytes, reverse=True)

        ok = mem_total_mib > 0 or bool(gpus) or cpu_pct >= 0
        return VmMetricsResult(
            ok, "ok" if ok else "no metrics parsed (is this a Linux host with nvidia-smi or npu-smi?)",
            cpu_pct, mem_used_mib, mem_total_mib, gpus, _time.time(),
            cpu_cores=cpu_cores, disks=disks, host_procs=host_procs,
        )
    finally:
        _close_quiet(client, jump)


async def metrics_vm(host: str, port: int, user: str, private_key: str | None = None,
                     password: str | None = None,
                     jump_host: str | None = None, jump_port: int = 22,
                     jump_user: str | None = None, jump_private_key: str | None = None,
                     jump_password: str | None = None) -> VmMetricsResult:
    return await asyncio.to_thread(_metrics_sync, host, port, user, private_key, password,
                                   jump_host, jump_port, jump_user, jump_private_key, jump_password)


# --------------------------------------------------------------------------
# On-demand bandwidth benchmark (disk read/write, sequential memory, CPU clock).
# Heavier than the live poll (writes a ~512 MiB temp file), so it's a separate
# button in the UI — NOT part of the polling loop.
# --------------------------------------------------------------------------
BANDWIDTH_TIMEOUT_S = 90

# Disk: write 512 MiB with an end fsync (so the rate includes the flush), then
# read it back with O_DIRECT when supported (falls back to a cached read). Memory:
# a /dev/zero→/dev/null sequential copy as a rough RAM/CPU throughput proxy.
_BANDWIDTH_CMD = (
    'f="$HOME/.gpf_bw_$$"; '
    "echo @@DWRITE; dd if=/dev/zero of=\"$f\" bs=1M count=512 conv=fdatasync 2>&1 | tail -1; "
    # Read back with O_DIRECT (bypasses page cache → real disk read) and keep its
    # rate on success; only fall back to a cached read if O_DIRECT isn't supported.
    "echo @@DREAD; { dd if=\"$f\" of=/dev/null bs=1M iflag=direct 2>&1 || "
    'dd if="$f" of=/dev/null bs=1M 2>&1; } | tail -1; '
    'rm -f "$f"; '
    "echo @@MEM; dd if=/dev/zero of=/dev/null bs=1M count=8192 2>&1 | tail -1; "
    "echo @@CPU; lscpu 2>/dev/null | grep -E 'Model name|CPU max MHz|CPU MHz'; "
)


@dataclass
class VmBandwidthResult:
    ok: bool
    message: str
    disk_write_mbps: float
    disk_read_mbps: float
    mem_mbps: float
    cpu_model: str
    cpu_mhz: float
    checked_at: float


def _parse_dd_rate(lines: list[str]) -> float:
    """MB/s from a `dd` summary line (`… , 2.5 s, 429 MB/s`). 0 if unparseable."""
    for ln in lines:
        m = re.search(r"([0-9.]+)\s*(GB|MB|kB)/s", ln)
        if not m:
            continue
        v = float(m.group(1))
        unit = m.group(2)
        if unit == "GB":
            return round(v * 1000, 1)
        if unit == "kB":
            return round(v / 1000, 2)
        return round(v, 1)
    return 0.0


def _bandwidth_sync(host: str, port: int, user: str, private_key: str | None = None,
                    password: str | None = None,
                    jump_host: str | None = None, jump_port: int = 22,
                    jump_user: str | None = None, jump_private_key: str | None = None,
                    jump_password: str | None = None) -> VmBandwidthResult:
    import time as _time
    try:
        client, jump = _connect(host, port, user, private_key, password,
                                jump_host, jump_port, jump_user, jump_private_key, jump_password)
    except RuntimeError as e:
        return VmBandwidthResult(False, str(e), 0, 0, 0, "", 0, _time.time())

    try:
        _, stdout, _ = client.exec_command(_BANDWIDTH_CMD, timeout=BANDWIDTH_TIMEOUT_S)
        stdout.channel.recv_exit_status()
        out = stdout.read().decode(errors="replace")
        sec: dict[str, list[str]] = {}
        cur = None
        for ln in out.splitlines():
            s = ln.strip()
            if s in ("@@DWRITE", "@@DREAD", "@@MEM", "@@CPU"):
                cur = s
                sec[cur] = []
            elif cur:
                sec[cur].append(ln)

        cpu_model = ""
        cpu_mhz = 0.0
        cpu_mhz_cur = 0.0
        for ln in sec.get("@@CPU", []):
            k, _, v = ln.partition(":")
            k, v = k.strip(), v.strip()
            if k == "Model name":
                cpu_model = v
            elif k == "CPU max MHz":
                try:
                    cpu_mhz = float(v)
                except ValueError:
                    pass
            elif k == "CPU MHz":
                try:
                    cpu_mhz_cur = float(v)
                except ValueError:
                    pass
        cpu_mhz = cpu_mhz or cpu_mhz_cur

        disk_w = _parse_dd_rate(sec.get("@@DWRITE", []))
        disk_r = _parse_dd_rate(sec.get("@@DREAD", []))
        mem = _parse_dd_rate(sec.get("@@MEM", []))
        ok = disk_w > 0 or disk_r > 0 or mem > 0
        return VmBandwidthResult(
            ok, "ok" if ok else "benchmark produced no parseable output",
            disk_w, disk_r, mem, cpu_model, round(cpu_mhz, 0), _time.time(),
        )
    finally:
        _close_quiet(client, jump)


async def bandwidth_vm(host: str, port: int, user: str, private_key: str | None = None,
                       password: str | None = None,
                       jump_host: str | None = None, jump_port: int = 22,
                       jump_user: str | None = None, jump_private_key: str | None = None,
                       jump_password: str | None = None) -> VmBandwidthResult:
    return await asyncio.to_thread(_bandwidth_sync, host, port, user, private_key, password,
                                   jump_host, jump_port, jump_user, jump_private_key, jump_password)


@dataclass
class VmKillResult:
    ok: bool
    message: str


def _kill_pid_sync(host: str, port: int, user: str, private_key: str | None = None,
                   pid: int = 0, sig: int = 9,
                   password: str | None = None,
                   jump_host: str | None = None, jump_port: int = 22,
                   jump_user: str | None = None, jump_private_key: str | None = None,
                   jump_password: str | None = None) -> VmKillResult:
    """SSH onto a VM and kill a process by pid (default SIGKILL — the metrics page
    "Terminate" button is for freeing a GPU held by a stuck/orphaned process). `pid`
    is an int (no shell interpolation of untrusted text). Reports the real outcome:
    a pid in another container's PID namespace (the orphaned-GPU case) isn't visible
    here, so `kill` reports "No such process" — surfaced so the UI says so plainly."""
    pid = int(pid)
    if pid <= 1:
        return VmKillResult(ok=False, message=f"refusing to kill pid {pid}")
    try:
        client, jump = _connect(host, port, user, private_key, password,
                                jump_host, jump_port, jump_user, jump_private_key, jump_password)
    except RuntimeError as e:
        return VmKillResult(ok=False, message=str(e))

    try:
        cmd = f"kill -{int(sig)} {pid}"
        _, stdout, stderr = client.exec_command(cmd, timeout=COMMAND_TIMEOUT_S)
        rc = stdout.channel.recv_exit_status()
        err = stderr.read().decode(errors="replace").strip()
        if rc == 0:
            return VmKillResult(ok=True, message=f"sent SIG{('KILL' if sig == 9 else sig)} to pid {pid}")
        # kill's stderr is the useful bit ("No such process" / "Operation not permitted").
        return VmKillResult(ok=False, message=err or f"kill exited {rc}")
    finally:
        _close_quiet(client, jump)


async def kill_pid_vm(host: str, port: int, user: str, private_key: str | None = None,
                      pid: int = 0, sig: int = 9,
                      password: str | None = None,
                      jump_host: str | None = None, jump_port: int = 22,
                      jump_user: str | None = None, jump_private_key: str | None = None,
                      jump_password: str | None = None) -> VmKillResult:
    return await asyncio.to_thread(_kill_pid_sync, host, port, user, private_key, pid, sig,
                                   password, jump_host, jump_port, jump_user,
                                   jump_private_key, jump_password)


def _kill_pids_sync(host: str, port: int, user: str, private_key: str | None = None,
                    pids: list[int] | None = None, sig: int = 9,
                    password: str | None = None,
                    jump_host: str | None = None, jump_port: int = 22,
                    jump_user: str | None = None, jump_private_key: str | None = None,
                    jump_password: str | None = None) -> VmKillResult:
    """SSH onto a VM and kill several pids in ONE session (the metrics page "Kill all"
    button, to free a GPU held by every process at once). Safer + far faster than N
    round-trips. pids are ints (no shell interpolation); pid<=1 is dropped. A single
    `kill -9 p1 p2 …` kills every live pid even if some are already gone — kill's
    stderr (e.g. "No such process") is surfaced but the survivors are still killed."""
    pids = [int(p) for p in (pids or []) if int(p) > 1]
    if not pids:
        return VmKillResult(ok=False, message="no killable pids")
    try:
        client, jump = _connect(host, port, user, private_key, password,
                                jump_host, jump_port, jump_user, jump_private_key, jump_password)
    except RuntimeError as e:
        return VmKillResult(ok=False, message=str(e))

    try:
        signame = "KILL" if sig == 9 else str(sig)
        cmd = "kill -{} {}".format(int(sig), " ".join(str(p) for p in pids))
        _, stdout, stderr = client.exec_command(cmd, timeout=COMMAND_TIMEOUT_S)
        rc = stdout.channel.recv_exit_status()
        err = stderr.read().decode(errors="replace").strip()
        if rc == 0:
            return VmKillResult(ok=True, message=f"sent SIG{signame} to {len(pids)} pid(s)")
        # Some pids may have been gone already; the rest were still killed.
        return VmKillResult(ok=False, message=err or f"kill exited {rc}")
    finally:
        _close_quiet(client, jump)


async def kill_pids_vm(host: str, port: int, user: str, private_key: str | None = None,
                       pids: list[int] | None = None, sig: int = 9,
                       password: str | None = None,
                       jump_host: str | None = None, jump_port: int = 22,
                       jump_user: str | None = None, jump_private_key: str | None = None,
                       jump_password: str | None = None) -> VmKillResult:
    return await asyncio.to_thread(_kill_pids_sync, host, port, user, private_key, pids, sig,
                                   password, jump_host, jump_port, jump_user,
                                   jump_private_key, jump_password)
