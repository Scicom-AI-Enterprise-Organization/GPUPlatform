"use client";

// Lineage canvas — a node graph of what a dataset was DERIVED from, drawn in the same
// visual language as the proxy routing panel (shared palette vars, dot-grid canvas,
// tinted-icon node cards, curved SVG edges).
//
// The topology is a DAG that FANS IN, which is the opposite of routing's chain: many
// original corpora flow rightwards through transforms and merges into one training set.
// So it lays out left → right by derivation level, roots in the leftmost column and the
// dataset you are looking at last on the right, with every edge pointing the direction
// the data actually moved.
//
// ⚠ Nodes are DE-DUPLICATED by id here even though the API returns a diamond twice. In a
// list, collapsing a repeat would hide that a corpus was merged in twice; in a graph the
// opposite is true — one node with two outgoing edges is exactly what "included twice"
// looks like, and drawing it as two separate boxes would imply two different corpora.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Database, GitMerge, Loader2, Shuffle, Sparkles, type LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export type LineageNode = {
  id: string;
  name?: string | null;
  kind?: string | null;
  num_rows?: number | null;
  hf_repo?: string | null;
  hf_subsets?: string[] | null;
  op?: string | null;
  params?: Record<string, unknown> | null;
  sources?: LineageNode[] | null;
  missing?: boolean;
  cycle?: boolean;
  truncated?: string;
};

type LineageResponse = {
  id: string;
  tree: LineageNode | null;
  roots: LineageNode[];
  flat: LineageNode[];
  depth: number;
};

// Same theme-aware var set the routing panel uses, so the two canvases read as one system.
const PALETTE = [
  "[--live:#059669] [--fall:#b45309] [--down:#e11d48] [--wire:#cbd5e1]",
  "[--dot:rgba(15,23,42,0.07)] [--surface:#ffffff] [--canvas:#f6f8fb] [--glow:#ffffff] [--vignette:transparent]",
  "dark:[--live:#34d399] dark:[--fall:#f59e0b] dark:[--down:#fb7185] dark:[--wire:#3a4150]",
  "dark:[--dot:rgba(255,255,255,0.06)] dark:[--surface:#141a24] dark:[--canvas:#0a0d13] dark:[--glow:#141d2c] dark:[--vignette:rgba(0,0,0,0.5)]",
].join(" ");

const NODE_W = 208;
const NODE_H = 74;
const COL_GAP = 68;
const ROW_GAP = 14;
const PAD = 16;

const OP_ICON: Record<string, LucideIcon> = {
  merge: GitMerge,
  transform: Shuffle,
  pack: Shuffle,
  generate: Sparkles,
  normalize: Shuffle,
};
const OP_LABEL: Record<string, string> = {
  merge: "merge", transform: "transform", pack: "pack",
  generate: "generate", normalize: "normalize",
};

type Placed = LineageNode & { level: number; row: number; x: number; y: number };

/** Flatten the tree into unique nodes + edges, and rank each node by derivation depth.
 *  A node's level is its LONGEST path from a root, so an edge never points backwards. */
function layout(tree: LineageNode | null) {
  if (!tree) return { nodes: [] as Placed[], edges: [] as [string, string][], w: 0, h: 0 };
  const byId = new Map<string, LineageNode>();
  const edges: [string, string][] = [];      // [sourceId, derivedId] — direction of data
  const seenEdge = new Set<string>();

  (function walk(n: LineageNode) {
    if (!byId.has(n.id)) byId.set(n.id, n);
    for (const s of n.sources ?? []) {
      const key = `${s.id}->${n.id}`;
      if (!seenEdge.has(key)) {
        seenEdge.add(key);
        edges.push([s.id, n.id]);
      }
      walk(s);
    }
  })(tree);

  // Longest-path level: repeat-relax until stable (the graph is tiny and acyclic —
  // resolve() converts a cycle into a leaf marker before it reaches us).
  const level = new Map<string, number>();
  for (const id of byId.keys()) level.set(id, 0);
  for (let i = 0; i < byId.size + 1; i++) {
    let moved = false;
    for (const [src, dst] of edges) {
      const want = (level.get(src) ?? 0) + 1;
      if (want > (level.get(dst) ?? 0)) {
        level.set(dst, want);
        moved = true;
      }
    }
    if (!moved) break;
  }

  const cols = new Map<number, string[]>();
  for (const [id, lv] of level) {
    const arr = cols.get(lv) ?? [];
    arr.push(id);
    cols.set(lv, arr);
  }
  const maxLevel = Math.max(...level.values());
  const maxRows = Math.max(...[...cols.values()].map((c) => c.length));

  const nodes: Placed[] = [];
  for (const [lv, ids] of [...cols.entries()].sort((a, b) => a[0] - b[0])) {
    // Centre each column vertically so a narrow column doesn't hug the top.
    const offset = ((maxRows - ids.length) * (NODE_H + ROW_GAP)) / 2;
    ids.forEach((id, row) => {
      const n = byId.get(id)!;
      nodes.push({
        ...n, level: lv, row,
        x: PAD + lv * (NODE_W + COL_GAP),
        y: PAD + offset + row * (NODE_H + ROW_GAP),
      });
    });
  }
  return {
    nodes, edges,
    w: PAD * 2 + (maxLevel + 1) * NODE_W + maxLevel * COL_GAP,
    h: PAD * 2 + maxRows * NODE_H + Math.max(0, maxRows - 1) * ROW_GAP,
  };
}

/** Left-to-right bezier between two node edges. */
function curve(x1: number, y1: number, x2: number, y2: number) {
  const dx = Math.max(28, (x2 - x1) / 2);
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

function fmtRows(n?: number | null) {
  return n == null ? "" : n.toLocaleString();
}

function GraphNode({ n, isTarget }: { n: Placed; isTarget: boolean }) {
  const isRoot = !(n.sources ?? []).length;
  const bad = n.missing || n.cycle;
  const tone = bad ? "down" : isTarget ? "live" : isRoot ? "wire" : "live";
  const color = `var(--${tone})`;
  const Icon = bad ? Database : isRoot ? Database : OP_ICON[n.op ?? ""] ?? Shuffle;
  const ring = isTarget
    ? `inset 0 0 0 2px ${color}, 0 0 22px -8px ${color}`
    : `inset 0 0 0 1px color-mix(in srgb, ${color} 32%, transparent)`;

  return (
    <div
      className="absolute overflow-hidden rounded-xl px-3 py-2"
      style={{ left: n.x, top: n.y, width: NODE_W, height: NODE_H, background: "var(--surface)", boxShadow: ring }}
      title={n.name ?? n.id}
    >
      <div className="flex items-center gap-2">
        <span
          className="grid h-4 w-4 shrink-0 place-items-center rounded-[5px]"
          style={{ background: `color-mix(in srgb, ${color} 16%, transparent)`, color }}
        >
          <Icon className="h-2.5 w-2.5" strokeWidth={2.25} />
        </span>
        {n.missing ? (
          <span className="truncate text-[11px] font-semibold text-rose-500">{n.id} (deleted)</span>
        ) : (
          <Link
            href={`/datasets/${n.id}`}
            className="truncate font-mono text-[11px] font-semibold text-foreground hover:underline"
          >
            {n.id}
          </Link>
        )}
        {isRoot && !bad ? (
          <span className="ml-auto shrink-0 rounded bg-muted px-1 py-0.5 text-[9px] uppercase tracking-wide text-muted-foreground">
            source
          </span>
        ) : null}
      </div>

      <div className="my-1 h-px" style={{ background: "color-mix(in srgb, var(--wire) 70%, transparent)" }} />

      <div className="truncate text-[11px] text-muted-foreground">{n.hf_repo ?? n.name}</div>
      <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
        {n.kind ? <span className="rounded bg-muted px-1 py-0.5">{n.kind}</span> : null}
        {n.num_rows != null ? <span>{fmtRows(n.num_rows)} rows</span> : null}
        {n.op ? (
          <span className="ml-auto truncate" style={{ color }}>
            {OP_LABEL[n.op] ?? n.op}
          </span>
        ) : null}
      </div>
    </div>
  );
}

export function LineageCard({ datasetId }: { datasetId: string }) {
  const [data, setData] = useState<LineageResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  // Fetch in the effect body AFTER an await, never synchronously — a sync setState
  // inside an effect cascades renders (react-hooks/set-state-in-effect). `cancelled`
  // also stops a slow response from overwriting a newer one when the id changes.
  const [reloadKey, setReloadKey] = useState(0);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}/lineage`);
        if (!res.ok) throw new Error(`lineage failed (${res.status})`);
        const body = (await res.json()) as LineageResponse;
        if (!cancelled) {
          setData(body);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "failed to load lineage");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [datasetId, reloadKey]);

  const load = useCallback(() => {
    setLoading(true);
    setReloadKey((k) => k + 1);
  }, []);

  const g = useMemo(() => layout(data?.tree ?? null), [data]);
  const isRoot = !!data && !data.tree?.sources?.length;
  const pos = useMemo(() => new Map(g.nodes.map((n) => [n.id, n])), [g]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm">
          <GitMerge className="h-4 w-4" />
          Lineage
          {data && !isRoot ? (
            <span className="text-xs font-normal text-muted-foreground">
              {g.nodes.length} datasets · {data.roots.length} original{" "}
              {data.roots.length === 1 ? "corpus" : "corpora"} · {data.depth} level
              {data.depth === 1 ? "" : "s"}
            </span>
          ) : null}
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        {loading ? (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> loading…
          </div>
        ) : error ? (
          <div className="flex items-center gap-2">
            <span className="text-xs text-destructive">{error}</span>
            <Button variant="outline" size="sm" className="h-6 text-xs" onClick={() => void load()}>
              retry
            </Button>
          </div>
        ) : isRoot ? (
          <p className="text-xs text-muted-foreground">
            Not derived from another dataset — this is an original source.
          </p>
        ) : (
          <div
            ref={wrapRef}
            className={cn("relative overflow-x-auto rounded-xl ring-1 ring-black/5 dark:ring-white/10", PALETTE)}
            style={{
              backgroundColor: "var(--canvas)",
              backgroundImage:
                "radial-gradient(var(--dot) 1px, transparent 1px), radial-gradient(120% 90% at 50% -10%, var(--glow) 0%, transparent 60%)",
              backgroundSize: "16px 16px, 100% 100%",
              boxShadow: "inset 0 0 80px var(--vignette)",
            }}
          >
            <div className="relative" style={{ width: g.w, height: g.h }}>
              <svg className="absolute inset-0" width={g.w} height={g.h} aria-hidden>
                {g.edges.map(([src, dst], i) => {
                  const a = pos.get(src);
                  const b = pos.get(dst);
                  if (!a || !b) return null;
                  const d = curve(a.x + NODE_W, a.y + NODE_H / 2, b.x, b.y + NODE_H / 2);
                  return (
                    <path
                      key={i}
                      d={d}
                      fill="none"
                      strokeWidth={1.75}
                      strokeLinecap="round"
                      stroke="var(--live)"
                      strokeOpacity={0.75}
                      style={{ filter: "drop-shadow(0 0 3px color-mix(in srgb, var(--live) 40%, transparent))" }}
                    />
                  );
                })}
              </svg>
              {g.nodes.map((n) => (
                <GraphNode key={n.id} n={n} isTarget={n.id === datasetId} />
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
