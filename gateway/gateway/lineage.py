"""Dataset lineage — what a dataset was DERIVED from, walked to its roots.

A training corpus here is routinely four derivations deep (HF repo → transform → merge
→ merge), and until this module the chain lived in three incompatible places: a single
reverse-lookup hop on `audio_dataset_id`, a prose `description` listing merge inputs,
and the transform log. Reconstructing "what is actually in this training set" meant
reading logs, which is exactly the thing nobody does before trusting a number.

`Dataset.lineage` records one derivation ({op, sources, params}); this module walks
those edges into a tree. `resolve()` also reconstructs edges for rows written BEFORE
the column existed, so old datasets are not silently rootless.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import Dataset

MAX_DEPTH = 12          # a derivation chain deeper than this is a bug, not a corpus
MAX_NODES = 200         # a fan-out wider than this would make the response useless

# Legacy merges recorded their inputs only in prose: "Merge of 2 datasets (ds-a, ds-b)".
_LEGACY_MERGE = re.compile(r"Merge of \d+ datasets? \(([^)]*)\)")


def record(op: str, sources: list[str], **params: Any) -> dict:
    """Build a lineage value. Keep `params` small — it is stored per dataset."""
    return {"op": op,
            "sources": [s for s in sources if s],
            "params": {k: v for k, v in params.items() if v is not None}}


async def _parents(session: AsyncSession, d: Dataset) -> tuple[str, list[str], dict]:
    """(op, parent_ids, params) for one dataset, structured first then reconstructed."""
    lin = getattr(d, "lineage", None) or None
    if isinstance(lin, dict) and lin.get("sources"):
        return str(lin.get("op") or "derive"), list(lin["sources"]), dict(lin.get("params") or {})

    # ---- fallbacks for rows created before `lineage` existed -------------------
    ids = _LEGACY_MERGE.search(d.description or "")
    if ids:
        return "merge", [s.strip() for s in ids.group(1).split(",") if s.strip()], {}

    # A transform sets the SOURCE's audio_dataset_id to point at its output, so the
    # parent is found by reverse lookup rather than by a field on the child.
    src = (await session.execute(
        select(Dataset.id).where(Dataset.audio_dataset_id == d.id))).scalars().first()
    if src:
        return "transform", [src], {}
    return "", [], {}


async def resolve(session: AsyncSession, dataset_id: str,
                  _depth: int = 0, _seen: Optional[set[str]] = None) -> Optional[dict]:
    """Ancestor tree for `dataset_id`, or None when the dataset is gone.

    Cycle-safe (a dataset already on the current path becomes a `cycle` marker rather
    than recursing) and bounded by MAX_DEPTH/MAX_NODES, because this is reachable from
    an HTTP handler and the edges are user-created.
    """
    seen = set(_seen or ())
    d = await session.get(Dataset, dataset_id)
    if d is None:
        return {"id": dataset_id, "missing": True}
    node: dict[str, Any] = {
        "id": d.id, "name": d.name, "kind": d.kind, "num_rows": d.num_rows,
        "hf_repo": d.hf_repo, "hf_subsets": list(getattr(d, "hf_subsets", None) or []) or None,
    }
    if dataset_id in seen:
        node["cycle"] = True
        return node
    if _depth >= MAX_DEPTH:
        node["truncated"] = "max depth"
        return node

    op, parents, params = await _parents(session, d)
    if not parents:
        return node                      # a root: uploaded, registered, or imported
    node["op"] = op
    if params:
        node["params"] = params
    seen.add(dataset_id)
    kids = []
    for pid in parents[:MAX_NODES]:
        kids.append(await resolve(session, pid, _depth + 1, seen))
    node["sources"] = kids
    return node


def flatten(node: Optional[dict]) -> list[dict]:
    """Depth-first list of every dataset in a tree — the 'what is in here' answer."""
    if not node:
        return []
    out = [{k: node.get(k) for k in ("id", "name", "kind", "num_rows", "op", "hf_repo")}]
    for s in node.get("sources") or []:
        out.extend(flatten(s))
    return out


def roots(node: Optional[dict]) -> list[dict]:
    """Only the leaves — the original corpora everything else was derived from."""
    if not node:
        return []
    if not node.get("sources"):
        return [{k: node.get(k) for k in ("id", "name", "kind", "hf_repo", "hf_subsets")}]
    out: list[dict] = []
    for s in node["sources"]:
        out.extend(roots(s))
    return out
