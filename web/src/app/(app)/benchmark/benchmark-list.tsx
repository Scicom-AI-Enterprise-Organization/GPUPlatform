"use client";

import { useMemo, useState } from "react";
import yaml from "js-yaml";
import { Search, X } from "lucide-react";
import type { BenchmarkRecord } from "@/lib/types";
import { BenchmarkRow } from "./benchmark-row";

/** Pre-compute a flat searchable string per benchmark. Includes name, id,
 * status, owner, model, GPU type, and parallelism so a single query can hit
 * any of those. Done once per render via useMemo. */
function searchableText(b: BenchmarkRecord): string {
  let model = "";
  let gpu = "";
  let parallelism = "";
  try {
    const cfg = yaml.load(b.config_yaml) as
      | {
          runpod?: { pod?: { gpu_type?: string; gpu_count?: number } };
          benchmark?: Array<{
            model?: { repo_id?: string };
            serve?: { tensor_parallel_size?: number; data_parallel_size?: number };
          }>;
        }
      | null;
    gpu = cfg?.runpod?.pod?.gpu_type ?? "";
    model = cfg?.benchmark?.[0]?.model?.repo_id ?? "";
    const tp = cfg?.benchmark?.[0]?.serve?.tensor_parallel_size ?? 1;
    const dp = cfg?.benchmark?.[0]?.serve?.data_parallel_size ?? 1;
    parallelism = `tp${tp} dp${dp} tp${tp}/dp${dp}`;
  } catch {
    // ignore
  }
  return [b.name, b.id, b.status, b.created_by, model, gpu, parallelism]
    .join(" ")
    .toLowerCase();
}

export function BenchmarkList({ items }: { items: BenchmarkRecord[] }) {
  const [q, setQ] = useState("");

  const haystacks = useMemo(
    () => items.map((b) => ({ bench: b, text: searchableText(b) })),
    [items],
  );

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return items;
    // Multi-token AND search: every whitespace-separated token must match
    // somewhere in the haystack. Lets you do "qwen rtx done" as one query.
    const tokens = needle.split(/\s+/).filter(Boolean);
    return haystacks
      .filter(({ text }) => tokens.every((t) => text.includes(t)))
      .map(({ bench }) => bench);
  }, [haystacks, items, q]);

  return (
    <div>
      <div className="relative mb-4">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <input
          type="search"
          placeholder="Search by name, id, model, GPU, owner, status…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          className="h-10 w-full rounded-md border border-input bg-background pl-9 pr-9 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
        />
        {q && (
          <button
            type="button"
            onClick={() => setQ("")}
            className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
            title="Clear"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      {q && (
        <div className="mb-3 text-xs text-muted-foreground">
          {filtered.length} of {items.length} match{filtered.length === 1 ? "es" : "es"} for{" "}
          <span className="font-mono text-foreground">&quot;{q}&quot;</span>
        </div>
      )}

      {filtered.length === 0 ? (
        <div className="rounded-md border border-dashed border-border px-6 py-12 text-center text-sm text-muted-foreground">
          No benchmarks match{" "}
          <span className="font-mono text-foreground">&quot;{q}&quot;</span>. Try a different query.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 xl:grid-cols-3">
          {filtered.map((b) => (
            <BenchmarkRow key={b.id} bench={b} />
          ))}
        </div>
      )}
    </div>
  );
}
