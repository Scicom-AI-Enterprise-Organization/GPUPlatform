#!/usr/bin/env python3
"""Harvest the TM knowledge base out of the simulation corpus -> `kb.json`.

`kbms_search` is the one tool whose answer IS the grounding source: an evaluator
asking "did the model assert anything the tool results don't support?" is judging
against whatever this returns. So the sandbox must not invent KB text — inventing
it would mean grading a model's groundedness against fiction that drifts from the
corpus it was trained on.

Instead we harvest the real chunks the generators already produced. Every
`role: tool` message answering a `kbms_search` call in the corpus parquets carries
`{"results": [{"text", "score", "document_name", "document_tags", "source_url"}]}`;
this script pulls them all out, dedupes on (document_name, text), and writes a flat
index the server does lexical retrieval over.

⚠ Chunks are harvested from the REFERENCE, CHOSEN and REJECTED trajectories alike.
That is deliberate and safe: a rejected trajectory was rejected for what the MODEL
wrote, not for what the KB returned — the tool results in it are as legitimate as
any other. What it must never do is copy a model's *reply*, and it doesn't.

Usage
    .venv/bin/python sandbox/tm_assist/build_kb.py                 # default corpus dir
    .venv/bin/python sandbox/tm_assist/build_kb.py --corpus DIR --out kb.json
    .venv/bin/python sandbox/tm_assist/build_kb.py --max-per-doc 12
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

DEFAULT_CORPUS = Path("/Users/husein.z/Documents/ucc_ai_research/synthetic-generation/"
                      "tm-text-assist-simulation")
# Trajectory columns, in trust order. `reference` first so that when the same chunk
# text appears in several places we keep the reference copy's metadata.
TRAJECTORY_COLUMNS = ("reference", "chosen", "rejected", "messages")
HERE = Path(__file__).resolve().parent


def iter_tool_messages(msgs: list):
    """(tool_name, content) for every tool result, resolving the name via the
    tool_call_id when the message doesn't carry `name` itself."""
    pending: dict = {}
    for m in msgs:
        if not isinstance(m, dict):
            continue
        for tc in (m.get("tool_calls") or []):
            fn = (tc.get("function") or {}).get("name")
            if fn:
                pending[tc.get("id")] = fn
        if m.get("role") == "tool":
            yield (m.get("name") or pending.get(m.get("tool_call_id")) or "?",
                   m.get("content") or "")


def harvest(corpus: Path, max_per_doc: int) -> dict:
    import pandas as pd

    parquets = sorted(p for p in corpus.glob("*.parquet") if "id-clean" not in p.name)
    if not parquets:
        sys.exit(f"no parquets in {corpus}")

    chunks: dict[tuple, dict] = {}
    per_doc: collections.Counter = collections.Counter()
    entity_keys: collections.Counter = collections.Counter()
    tool_names: collections.Counter = collections.Counter()
    scanned = 0

    for pq in parquets:
        try:
            df = pd.read_parquet(pq)
        except Exception as e:  # noqa: BLE001 — a stray parquet must not kill the build
            print(f"  ! {pq.name}: {e}")
            continue
        cols = [c for c in TRAJECTORY_COLUMNS if c in df.columns]
        if not cols:
            continue
        print(f"  {pq.name}: {len(df)} rows, columns {cols}")
        if "shared_entities" in df.columns:
            for s in df["shared_entities"]:
                try:
                    o = json.loads(s) if isinstance(s, str) else s
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(o, dict):
                    entity_keys.update(o.keys())
        for col in cols:
            for cell in df[col]:
                if not (isinstance(cell, str) and cell.strip().startswith("[")):
                    continue
                try:
                    msgs = json.loads(cell)
                except Exception:  # noqa: BLE001
                    continue
                scanned += 1
                for name, content in iter_tool_messages(msgs):
                    tool_names[name] += 1
                    if name != "kbms_search":
                        continue
                    try:
                        payload = json.loads(content)
                    except Exception:  # noqa: BLE001
                        continue
                    for r in (payload.get("results") or []):
                        if not (isinstance(r, dict) and r.get("text")):
                            continue
                        doc = str(r.get("document_name") or "Untitled")
                        key = (doc, r["text"])
                        if key in chunks:
                            continue
                        if per_doc[doc] >= max_per_doc:
                            continue
                        per_doc[doc] += 1
                        chunks[key] = {
                            "text": r["text"],
                            "document_name": doc,
                            "document_tags": sorted({str(t) for t in (r.get("document_tags") or [])}),
                            "source_url": r.get("source_url") or "",
                        }

    docs = sorted({c["document_name"] for c in chunks.values()})
    tags = collections.Counter(t for c in chunks.values() for t in c["document_tags"])
    return {
        "_meta": {
            "source_corpus": str(corpus),
            "parquets": [p.name for p in parquets],
            "trajectories_scanned": scanned,
            "chunks": len(chunks),
            "documents": len(docs),
            "max_per_doc": max_per_doc,
            "tool_result_counts": dict(tool_names.most_common()),
            "shared_entity_keys": sorted(entity_keys),
        },
        "tags": [t for t, _ in tags.most_common()],
        "chunks": list(chunks.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=HERE / "kb.json")
    ap.add_argument("--max-per-doc", type=int, default=8,
                    help="cap chunks kept per document_name — the corpus has one SOP with "
                         "667 near-duplicate chunks, which would dominate retrieval and "
                         "bloat the file for no extra coverage")
    a = ap.parse_args()

    print(f"harvesting {a.corpus} …")
    kb = harvest(a.corpus, a.max_per_doc)
    a.out.write_text(json.dumps(kb, ensure_ascii=False))
    m = kb["_meta"]
    size_mb = a.out.stat().st_size / 1e6
    print(f"\nwrote {a.out}  ({size_mb:.1f} MB)")
    print(f"  {m['chunks']} chunks over {m['documents']} documents, "
          f"{len(kb['tags'])} distinct tags")
    print(f"  from {m['trajectories_scanned']} trajectories")
    top = list(m["tool_result_counts"].items())[:6]
    print(f"  tool results seen: {', '.join(f'{k}={v}' for k, v in top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
