"""Precise, reuse-safe teardown of the vLLM process groups THIS worker spawned.

We deliberately NEVER pattern-match (`pkill -f 'VLLM::'`) or scan by port — the
VM may be shared, and another endpoint (or another user, outside the platform
entirely) can have its own vLLM running. Killing those would be unacceptable.

Instead the scheduler launches every engine in its own session
(`start_new_session=True`), so the api_server is its process-group leader and its
tp-worker / engine-core children share that pgid. The scheduler persists, per
engine, that `pgid` plus a `(pid, starttime)` sample of the group's members.

Cleanup (preflight of the next run, or the provider's terminate) reads that file
and kills ONLY processes in a recorded group — and only after proving the group
is still ours via a start-time match, so a recycled pid/pgid can never be
mistaken for one of ours. Anything we didn't record is never signalled.
"""
from __future__ import annotations

import json
import os
import signal
import time

# /proc/<pid>/stat: "pid (comm) state ppid pgrp ...". comm can contain spaces and
# parens, so we split AFTER the last ')'. In that tail (0-indexed): [0]=state,
# [1]=ppid, [2]=pgrp, ... [19]=starttime (field 22 of the full line).
_PGRP_IDX = 2
_STARTTIME_IDX = 19


def _read_stat(pid) -> tuple[int, int] | None:
    """(pgrp, starttime) for a pid, or None if it's gone/unreadable."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
    except OSError:
        return None
    try:
        tail = data.rpartition(")")[2].split()
        return int(tail[_PGRP_IDX]), int(tail[_STARTTIME_IDX])
    except (ValueError, IndexError):
        return None


def snapshot_groups(pgids) -> dict[int, list[list[int]]]:
    """One /proc pass → {pgid: [[pid, starttime], ...]} for the requested pgids."""
    want = {int(g) for g in pgids}
    out: dict[int, list[list[int]]] = {g: [] for g in want}
    if not want:
        return out
    try:
        entries = os.listdir("/proc")
    except OSError:
        return out
    for e in entries:
        if not e.isdigit():
            continue
        st = _read_stat(e)
        if st and st[0] in want:
            out[st[0]].append([int(e), st[1]])
    return out


def _alive_matches(pid: int, starttime: int) -> bool:
    st = _read_stat(pid)
    return st is not None and st[1] == starttime


def cleanup_records(records, grace_s: float = 8.0, log=None) -> list[int]:
    """Kill every current member of each recorded group that is PROVABLY still
    ours (>=1 recorded (pid, starttime) still alive & matching). Returns the pids
    signalled. Groups whose recorded members are all gone — or whose pgid was
    recycled — are skipped entirely."""
    def _log(msg: str) -> None:
        if log:
            log(msg)

    victims: list[int] = []
    for rec in records or []:
        try:
            pgid = int(rec["pgid"])
            recorded = [(int(p), int(s)) for p, s in rec.get("pids", [])]
        except (KeyError, ValueError, TypeError):
            continue
        if not any(_alive_matches(p, s) for p, s in recorded):
            continue  # group fully gone, or pgid recycled — do NOT touch it
        members = [p for p, _ in snapshot_groups([pgid])[pgid]]
        if members:
            _log(f"cleanup: model={rec.get('model')} pgid={pgid} → SIGTERM {members}")
            victims.extend(members)
            for pid in members:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
    if not victims:
        return []
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not any(_read_stat(p) for p in victims):
            return victims
        time.sleep(0.2)
    for pid in victims:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return victims


def cleanup_file(path: str | None, log=None) -> list[int]:
    """Run cleanup against records persisted at `path`. Missing/garbage file is a
    safe no-op."""
    if not path:
        return []
    try:
        with open(path) as f:
            records = json.load(f)
    except (OSError, ValueError):
        return []
    return cleanup_records(records, log=log)


def dump_records(path: str, records) -> None:
    """Atomically persist the engine-group records (write-tmp + rename)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(records, f)
    os.replace(tmp, path)


if __name__ == "__main__":
    # Invoked over SSH by the provider's terminate: kill exactly this endpoint's
    # recorded engine groups, then remove the file. Never touches anything else.
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = sys.argv[1] if len(sys.argv) > 1 else None
    if not p:
        print("usage: python -m worker_agent.multi.cleanup <pids_file>")
        sys.exit(0)
    killed = cleanup_file(p, log=print)
    print(f"cleanup: signalled {len(killed)} process(es) from {p}")
    try:
        os.remove(p)
    except OSError:
        pass
