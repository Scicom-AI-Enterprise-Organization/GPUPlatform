"""Storage cleanup — classify objects under a storage prefix into what's safe to
delete. Pure + deterministic (I/O and DB reads happen in `storage_api`, which feeds
this the object list + the live-owner id sets), so it's unit-testable and the
manual scan → dry-run → confirm-delete flow re-runs it identically at delete time.

Two definitions of "unnecessary" (both, per the storage-cleanup feature):
  * ORPHAN — an object under a known id-keyed subtree (`datasets/<id>/`,
    `training-runs/<id>/`, `benchmarks/<id>/`, `quantization-jobs/<id>/`,
    `serverless-logs/<app_id>/`) whose owning DB row no longer exists (e.g. a
    dataset deleted without purge, a run/job/app gone).
  * AGED — an object in an EPHEMERAL subtree (benchmarks / quantization-jobs /
    serverless-logs) whose group's newest object is older than `max_age_days`,
    even if the owner row still exists (old, regenerable artifacts).

Anything unrecognized — or under a live HuggingFace-mirror catalog repo's prefix —
is KEPT (never proposed for deletion). Per-blob GC inside a live versioned repo is
deliberately out of scope (content-addressed; needs a manifest cross-ref)."""
from __future__ import annotations

from typing import Iterable, Optional

# top-level folder under the storage prefix -> (owner-kind key in `live_ids`, ephemeral?)
_SUBTREES: dict[str, tuple[str, bool]] = {
    "datasets": ("dataset", False),
    "training-runs": ("training_run", False),
    "benchmarks": ("benchmark", True),
    "quantization-jobs": ("quant_job", True),
    "serverless-logs": ("app", True),
}


def _norm_base(base: Optional[str]) -> str:
    """'' or 'a/b/' — a non-empty base always ends in exactly one '/'. """
    b = (base or "").strip().strip("/")
    return f"{b}/" if b else ""


def _covered_by_repo(prefix: str, repo_prefixes: Iterable[str]) -> bool:
    """True if `prefix` sits inside (or equals) a live catalog-repo's key prefix —
    such objects belong to a registered HF-mirror repo and must be protected."""
    for rp in repo_prefixes:
        rp = rp.strip("/")
        if not rp:
            continue
        if prefix == rp or prefix.startswith(rp + "/") or (rp + "/").startswith(prefix.rstrip("/") + "/"):
            return True
    return False


def categorize(
    objects: list[dict],
    *,
    base: Optional[str],
    live_ids: dict[str, set[str]],
    repo_prefixes: Iterable[str],
    cutoff_iso: Optional[str],
) -> dict:
    """Group `objects` ([{key,size,modified}]) into per-owner groups and classify
    each. `cutoff_iso` is the age threshold (ISO-8601); a group is AGED when its
    newest object's `modified` is < cutoff. Returns:
        {groups: [{prefix, category, owner_kind, owner_id, objects, bytes,
                   newest, reason, purgeable}],
         total_objects, total_bytes, reclaimable_objects, reclaimable_bytes}
    Groups are `<base><folder>/<id>/`; objects that don't match a known subtree
    collapse into one `kept` group per top-level folder so totals still add up."""
    base = _norm_base(base)
    repo_prefixes = list(repo_prefixes or [])
    groups: dict[str, dict] = {}
    total_objects = total_bytes = 0

    for o in objects:
        key = o.get("key") or ""
        size = int(o.get("size") or 0)
        modified = o.get("modified") or ""
        total_objects += 1
        total_bytes += size
        rel = key[len(base):] if base and key.startswith(base) else key
        parts = rel.split("/")
        folder = parts[0] if parts else ""
        # group prefix: <base><folder>/<id>/ for known subtrees, else <base><folder>/
        if folder in _SUBTREES and len(parts) >= 2 and parts[1]:
            owner_id = parts[1]
            gprefix = f"{base}{folder}/{owner_id}/"
        else:
            owner_id = None
            gprefix = f"{base}{folder}/" if folder else base
        g = groups.get(gprefix)
        if g is None:
            g = groups[gprefix] = {
                "prefix": gprefix, "folder": folder, "owner_id": owner_id,
                "objects": 0, "bytes": 0, "newest": "",
            }
        g["objects"] += 1
        g["bytes"] += size
        if modified > g["newest"]:
            g["newest"] = modified

    out_groups = []
    reclaimable_objects = reclaimable_bytes = 0
    for gprefix, g in groups.items():
        folder, owner_id = g["folder"], g["owner_id"]
        category, reason, owner_kind = "kept", "", None
        if _covered_by_repo(gprefix, repo_prefixes):
            category, reason = "kept", "live HuggingFace-mirror repo"
        elif folder in _SUBTREES and owner_id is not None:
            owner_kind, ephemeral = _SUBTREES[folder]
            alive = owner_id in live_ids.get(owner_kind, set())
            aged = bool(ephemeral and cutoff_iso and g["newest"] and g["newest"] < cutoff_iso)
            if not alive:
                category = "orphan"
                reason = f"{owner_kind.replace('_', ' ')} {owner_id} no longer exists"
            elif aged:
                category = "aged"
                reason = f"{folder} artifacts older than the age cutoff"
            else:
                category = "kept"
                reason = f"{owner_kind.replace('_', ' ')} {owner_id} is live"
        else:
            reason = "unrecognized — kept for safety"
        purgeable = category in ("orphan", "aged")
        if purgeable:
            reclaimable_objects += g["objects"]
            reclaimable_bytes += g["bytes"]
        out_groups.append({
            "prefix": gprefix, "category": category, "owner_kind": owner_kind,
            "owner_id": owner_id, "objects": g["objects"], "bytes": g["bytes"],
            "newest": g["newest"], "reason": reason, "purgeable": purgeable,
        })

    out_groups.sort(key=lambda x: (not x["purgeable"], -x["bytes"]))
    return {
        "total_objects": total_objects, "total_bytes": total_bytes,
        "reclaimable_objects": reclaimable_objects, "reclaimable_bytes": reclaimable_bytes,
        "groups": out_groups,
    }
