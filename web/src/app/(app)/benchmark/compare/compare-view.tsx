"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import yaml from "js-yaml";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, ArrowLeft, Clock, Loader2, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { gateway } from "@/lib/gateway";
import {
  bestBy,
  fetchBenchRows,
  fmt,
  type Row,
  type StatMode,
  statPick,
} from "@/lib/bench-results";
import type { BenchmarkRecord } from "@/lib/types";
import { shortGpu } from "@/lib/gpu-format";
import { cn } from "@/lib/utils";

// Unlike the single-bench Results tab (monochrome shades for input lengths),
// Compare uses colour to encode *run identity* — a categorical, high-contrast
// palette so overlaid runs are easy to tell apart on both themes.
const COMPARE_COLORS = [
  "#2563eb", // blue-600
  "#f59e0b", // amber-500
  "#10b981", // emerald-500
  "#ec4899", // pink-500
  "#8b5cf6", // violet-500
  "#06b6d4", // cyan-500
  "#f97316", // orange-500
  "#84cc16", // lime-500
  "#ef4444", // red-500
  "#14b8a6", // teal-500
  "#a855f7", // purple-500
  "#eab308", // yellow-500
];

// One benchmark's loaded state: metadata (for labels) + parsed sweep rows.
type BenchData = {
  id: string;
  name: string;
  status: string;
  model: string | null;
  gpu: string | null;
  gpuCount: number;
  rows: Row[];
  error: string | null;
};

function shortModel(s: string | null): string {
  if (!s) return "—";
  return s.split("/").pop() ?? s;
}

function metaFromConfig(cfg_yaml: string): {
  model: string | null;
  gpu: string | null;
  gpuCount: number;
} {
  try {
    const cfg = yaml.load(cfg_yaml) as
      | {
          runpod?: { pod?: { gpu_type?: string; gpu_count?: number } };
          benchmark?: Array<{ model?: { repo_id?: string } }>;
        }
      | null;
    return {
      model: cfg?.benchmark?.[0]?.model?.repo_id ?? null,
      gpu: cfg?.runpod?.pod?.gpu_type ?? null,
      gpuCount: cfg?.runpod?.pod?.gpu_count ?? 1,
    };
  } catch {
    return { model: null, gpu: null, gpuCount: 1 };
  }
}

type Shape = { input_len: number; output_len: number };
const shapeKey = (s: Shape) => `${s.input_len}_${s.output_len}`;
const shapeLabel = (s: Shape) => `in=${s.input_len} · out=${s.output_len}`;

export function CompareView({ ids }: { ids: string[] }) {
  const [benches, setBenches] = useState<BenchData[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [statMode, setStatMode] = useState<StatMode>("median");
  const [shapeSel, setShapeSel] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const loaded = await Promise.all(
        ids.map(async (id): Promise<BenchData> => {
          try {
            const [rec, rows] = await Promise.all([
              gateway.getBenchmark(id).catch(() => null as BenchmarkRecord | null),
              fetchBenchRows(id),
            ]);
            const meta = rec ? metaFromConfig(rec.config_yaml) : { model: null, gpu: null, gpuCount: 1 };
            return {
              id,
              name: rec?.name ?? id,
              status: rec?.status ?? "unknown",
              model: meta.model,
              gpu: meta.gpu,
              gpuCount: meta.gpuCount,
              rows,
              error: rows.length === 0 ? "no result.json in S3" : null,
            };
          } catch (e) {
            return {
              id,
              name: id,
              status: "unknown",
              model: null,
              gpu: null,
              gpuCount: 1,
              rows: [],
              error: e instanceof Error ? e.message : String(e),
            };
          }
        }),
      );
      if (!cancelled) {
        setBenches(loaded);
        setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [ids]);

  // All (input,output) shapes present, ranked by how many benches cover them
  // (then by input length) so the default selection has the most overlap.
  const shapes = useMemo(() => {
    if (!benches) return [];
    const counts = new Map<string, { shape: Shape; benches: Set<string>; cells: number }>();
    for (const b of benches) {
      for (const r of b.rows) {
        const s = { input_len: r.input_len, output_len: r.output_len };
        const k = shapeKey(s);
        const e = counts.get(k) ?? { shape: s, benches: new Set<string>(), cells: 0 };
        e.benches.add(b.id);
        e.cells += 1;
        counts.set(k, e);
      }
    }
    return Array.from(counts.values()).sort(
      (a, b) => b.benches.size - a.benches.size || a.shape.input_len - b.shape.input_len || a.cells - b.cells,
    );
  }, [benches]);

  // Default to the highest-overlap shape (shapes[0]) unless the user picked one
  // that still exists. Derived (not stored) so there's no set-state-in-effect.
  const selectedKey = useMemo(() => {
    if (shapeSel && shapes.some((s) => shapeKey(s.shape) === shapeSel)) return shapeSel;
    return shapes.length > 0 ? shapeKey(shapes[0].shape) : null;
  }, [shapes, shapeSel]);

  const selectedShape = useMemo(
    () => shapes.find((s) => shapeKey(s.shape) === selectedKey)?.shape ?? null,
    [shapes, selectedKey],
  );

  const loadedBenches = useMemo(() => (benches ?? []).filter((b) => b.rows.length > 0), [benches]);

  if (loading) {
    return (
      <div className="flex items-center justify-center rounded-md border border-border px-4 py-16 text-sm text-muted-foreground">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading {ids.length} benchmark{ids.length === 1 ? "" : "s"} from S3…
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Button asChild variant="ghost" size="sm" className="-ml-2 h-7 px-2 text-muted-foreground">
              <Link href="/benchmark">
                <ArrowLeft className="h-3.5 w-3.5" /> Benchmarks
              </Link>
            </Button>
          </div>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight">Compare benchmarks</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {loadedBenches.length} of {ids.length} loaded · one line per run, overlaid across concurrency.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {selectedShape && shapes.length > 1 && (
            <select
              value={selectedKey ?? ""}
              onChange={(e) => setShapeSel(e.target.value)}
              className="h-9 rounded-md border border-input bg-background px-3 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
              title="Input/output shape to compare on"
            >
              {shapes.map(({ shape, benches: bset }) => {
                const k = shapeKey(shape);
                return (
                  <option key={k} value={k}>
                    {shapeLabel(shape)} ({bset.size}/{ids.length})
                  </option>
                );
              })}
            </select>
          )}
          <Tabs value={statMode} onValueChange={(v) => setStatMode(v as StatMode)}>
            <TabsList>
              <TabsTrigger value="median">Median</TabsTrigger>
              <TabsTrigger value="mean">Mean</TabsTrigger>
              <TabsTrigger value="p99">p99</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </div>

      {/* Per-benchmark legend: color ↔ run, with model + GPU. */}
      <div className="flex flex-wrap gap-2">
        {(benches ?? []).map((b, i) => {
          const color = COMPARE_COLORS[i % COMPARE_COLORS.length];
          return (
            <Link
              key={b.id}
              href={`/benchmark/${encodeURIComponent(b.id)}`}
              className={cn(
                "group inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-1.5 text-xs transition-colors hover:border-primary/40",
                b.error && "opacity-60",
              )}
            >
              <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color }} />
              <span className="font-medium text-foreground group-hover:underline">{b.name}</span>
              <span className="font-mono text-muted-foreground">{shortModel(b.model)}</span>
              {b.gpu && (
                <span className="font-mono text-muted-foreground">
                  · {shortGpu(b.gpu)}
                  {b.gpuCount > 1 ? `×${b.gpuCount}` : ""}
                </span>
              )}
              {b.error && <span className="text-destructive">· {b.error}</span>}
            </Link>
          );
        })}
      </div>

      {loadedBenches.length === 0 ? (
        <div className="rounded-md border border-dashed border-border px-6 py-12 text-center text-sm text-muted-foreground">
          None of the selected benchmarks have a <span className="font-mono">result.json</span> in S3 yet.
        </div>
      ) : (
        <>
          {/* Headline numbers, whole-sweep, side by side. */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Headline (whole sweep)</CardTitle>
              <CardDescription className="text-xs">
                Best of each metric across every cell in each run.
              </CardDescription>
            </CardHeader>
            <CardContent className="px-0 pb-0">
              <CompareTable benches={loadedBenches} />
            </CardContent>
          </Card>

          {selectedShape && (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <CompareChart
                title="Output throughput"
                subtitle="tokens/sec — higher is better"
                icon={<TrendingUp className="h-4 w-4" />}
                benches={loadedBenches}
                shape={selectedShape}
                y={(r) => r.output_throughput}
                yLabel="tok/s"
              />
              <CompareChart
                title="Time to first token"
                subtitle={`${statMode.toUpperCase()} TTFT (ms) — lower is better`}
                icon={<Clock className="h-4 w-4" />}
                benches={loadedBenches}
                shape={selectedShape}
                y={(r) => statPick(r, "ttft", statMode)}
                yLabel="ms"
              />
              <CompareChart
                title="Time per output token"
                subtitle={`${statMode.toUpperCase()} TPOT (ms) — lower is better`}
                icon={<Activity className="h-4 w-4" />}
                benches={loadedBenches}
                shape={selectedShape}
                y={(r) => statPick(r, "tpot", statMode)}
                yLabel="ms"
              />
              <CompareChart
                title="End-to-end latency"
                subtitle={`${statMode.toUpperCase()} E2EL (ms) — lower is better`}
                icon={<Clock className="h-4 w-4" />}
                benches={loadedBenches}
                shape={selectedShape}
                y={(r) => statPick(r, "e2el", statMode)}
                yLabel="ms"
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}

function CompareTable({ benches }: { benches: BenchData[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-muted/40">
          <tr>
            <th className="px-3 py-2 text-left text-xs uppercase tracking-wide text-muted-foreground">Run</th>
            <th className="px-3 py-2 text-left text-xs uppercase tracking-wide text-muted-foreground">Model</th>
            <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">
              Best tput (tok/s)
            </th>
            <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">
              Low median TTFT (ms)
            </th>
            <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">
              Low median TPOT (ms)
            </th>
            <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">Cells</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {benches.map((b, i) => {
            const tput = bestBy(b.rows, (r) => r.output_throughput);
            const ttft = bestBy(b.rows, (r) => r.median_ttft_ms, true);
            const tpot = bestBy(b.rows, (r) => r.median_tpot_ms, true);
            const color = COMPARE_COLORS[i % COMPARE_COLORS.length];
            return (
              <tr key={b.id}>
                <td className="px-3 py-1.5">
                  <span className="inline-flex items-center gap-2">
                    <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
                    <span className="font-medium">{b.name}</span>
                  </span>
                </td>
                <td className="px-3 py-1.5 font-mono text-xs text-muted-foreground">{shortModel(b.model)}</td>
                <td className="px-3 py-1.5 text-right tabular-nums">
                  {fmt(tput?.output_throughput ?? null, 1)}
                  {tput && (
                    <span className="ml-1 text-[10px] text-muted-foreground">c={tput.concurrency}</span>
                  )}
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums">{fmt(ttft?.median_ttft_ms ?? null, 1)}</td>
                <td className="px-3 py-1.5 text-right tabular-nums">{fmt(tpot?.median_tpot_ms ?? null, 2)}</td>
                <td className="px-3 py-1.5 text-right tabular-nums text-muted-foreground">{b.rows.length}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function CompareChart({
  title,
  subtitle,
  icon,
  benches,
  shape,
  y,
  yLabel,
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  benches: BenchData[];
  shape: Shape;
  y: (r: Row) => number | null;
  yLabel: string;
}) {
  // recharts wants one object per X point with each run as a column keyed by
  // its bench id. X = union of concurrencies present at this shape.
  const data = useMemo(() => {
    const concs = Array.from(
      new Set(
        benches.flatMap((b) =>
          b.rows
            .filter((r) => r.input_len === shape.input_len && r.output_len === shape.output_len)
            .map((r) => r.concurrency),
        ),
      ),
    ).sort((a, b) => a - b);
    return concs.map((c) => {
      const row: Record<string, number | null> = { concurrency: c };
      for (const b of benches) {
        const m = b.rows.find(
          (r) =>
            r.input_len === shape.input_len &&
            r.output_len === shape.output_len &&
            r.concurrency === c,
        );
        row[b.id] = m ? y(m) : null;
      }
      return row;
    });
  }, [benches, shape, y]);

  const nameById = useMemo(() => {
    const m: Record<string, string> = {};
    for (const b of benches) m[b.id] = b.name;
    return m;
  }, [benches]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-muted text-muted-foreground">
            {icon}
          </div>
          <div>
            <CardTitle className="text-sm">{title}</CardTitle>
            <CardDescription className="text-[11px]">{subtitle}</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 8, left: 12, bottom: 8 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis
                dataKey="concurrency"
                stroke="currentColor"
                className="text-[10px] text-muted-foreground"
                tickLine={false}
                axisLine={false}
                height={40}
                tickMargin={6}
                label={{
                  value: "concurrency",
                  position: "insideBottom",
                  offset: -16,
                  fontSize: 10,
                  fill: "currentColor",
                }}
              />
              <YAxis
                stroke="currentColor"
                className="text-[10px] text-muted-foreground"
                tickLine={false}
                axisLine={false}
                width={68}
                tickMargin={4}
                label={{
                  value: yLabel,
                  angle: -90,
                  position: "insideLeft",
                  offset: -2,
                  fontSize: 10,
                  fill: "currentColor",
                  style: { textAnchor: "middle" },
                }}
              />
              <Tooltip
                contentStyle={{
                  background: "rgb(24 24 27)",
                  border: "1px solid rgb(63 63 70)",
                  borderRadius: 6,
                  fontSize: 11,
                }}
                labelStyle={{ color: "rgb(244 244 245)" }}
                itemStyle={{ color: "rgb(228 228 231)" }}
                formatter={(value, name) => [value as number, nameById[String(name)] ?? String(name)]}
              />
              <Legend
                verticalAlign="top"
                align="center"
                iconType="plainline"
                wrapperStyle={{ fontSize: 11, paddingBottom: 8 }}
              />
              {benches.map((b, i) => (
                <Line
                  key={b.id}
                  type="monotone"
                  dataKey={b.id}
                  name={b.name}
                  stroke={COMPARE_COLORS[i % COMPARE_COLORS.length]}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                  connectNulls
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
