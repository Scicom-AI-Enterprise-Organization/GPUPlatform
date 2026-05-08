"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import {
  BarChart3,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { gateway } from "@/lib/gateway";
import type { AggregatePoint } from "@/lib/types";
import { cn } from "@/lib/utils";

const COLORS = [
  "#a78bfa", "#22d3ee", "#34d399", "#fbbf24",
  "#fb7185", "#60a5fa", "#f472b6", "#facc15",
  "#c084fc", "#2dd4bf",
];

// Recharts shape names map roughly to what shadcn aesthetic supports.
const SHAPES = ["circle", "triangle", "square", "diamond", "star", "cross", "wye"] as const;
type Shape = (typeof SHAPES)[number];

const VIEWS = [
  { id: "tput-vs-e2e",     label: "Throughput vs E2E Latency", x: "throughput",   y: "e2el",   xLabel: "Throughput / GPU (tok/s)",  yLabel: "E2E latency (ms)" },
  { id: "ttft-vs-context", label: "TTFT vs Context",            x: "context_len",  y: "ttft",   xLabel: "Context length (tokens)",   yLabel: "TTFT (ms)" },
  { id: "e2e-vs-context",  label: "E2E vs Context",             x: "context_len",  y: "e2el",   xLabel: "Context length (tokens)",   yLabel: "E2E latency (ms)" },
  { id: "tpot-vs-context", label: "TPOT vs Context",            x: "context_len",  y: "tpot",   xLabel: "Context length (tokens)",   yLabel: "TPOT (ms)" },
] as const;
type ViewId = (typeof VIEWS)[number]["id"];

function shortGpu(s: string | null | undefined): string {
  if (!s) return "—";
  return s.replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "").replace(/\s+80GB\s+(HBM3|PCIe).*$/i, " 80GB");
}

function shortModel(s: string | null | undefined): string {
  if (!s) return "—";
  return s.split("/").pop() ?? s;
}

function pickX(p: AggregatePoint, view: ViewId): number | null {
  switch (view) {
    case "tput-vs-e2e":     return p.output_throughput_per_gpu;
    case "ttft-vs-context": return p.context_len;
    case "e2e-vs-context":  return p.context_len;
    case "tpot-vs-context": return p.context_len;
  }
}

function pickY(p: AggregatePoint, view: ViewId, useP99: boolean): number | null {
  switch (view) {
    case "tput-vs-e2e":     return useP99 ? p.p99_e2el_ms ?? p.median_e2el_ms : p.median_e2el_ms ?? p.p99_e2el_ms;
    case "ttft-vs-context": return useP99 ? p.p99_ttft_ms : p.median_ttft_ms;
    case "e2e-vs-context":  return useP99 ? p.p99_e2el_ms : p.median_e2el_ms;
    case "tpot-vs-context": return useP99 ? p.p99_tpot_ms : p.median_tpot_ms;
  }
}

type SeriesKey = string; // "model · GPU · TP/DP"

export function BenchmarkExplorer() {
  const [points, setPoints] = useState<AggregatePoint[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [view, setView] = useState<ViewId>("tput-vs-e2e");
  const [gpuFilter, setGpuFilter] = useState<string>("__all__");
  const [modelFilter, setModelFilter] = useState<string>("__all__");
  const [parFilter, setParFilter] = useState<string>("__all__");
  const [logScale, setLogScale] = useState<boolean>(true);
  const [stat, setStat] = useState<"median" | "p99">("median");

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const data = await gateway.aggregateBenchmarks();
      setPoints(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    refresh();
  }, []);

  const allModels = useMemo(() => uniqueOf(points ?? [], (p) => p.model ?? "—"), [points]);
  const allGpus = useMemo(() => uniqueOf(points ?? [], (p) => p.gpu_type ?? "—"), [points]);
  const allParallelisms = useMemo(
    () => uniqueOf(points ?? [], (p) => `TP${p.tp}/DP${p.dp}`),
    [points],
  );

  const filtered = useMemo(() => {
    if (!points) return [];
    return points
      .filter((p) => modelFilter === "__all__" || (p.model ?? "—") === modelFilter)
      .filter((p) => gpuFilter === "__all__" || (p.gpu_type ?? "—") === gpuFilter)
      .filter((p) => parFilter === "__all__" || `TP${p.tp}/DP${p.dp}` === parFilter)
      .map((p) => ({
        ...p,
        _x: pickX(p, view),
        _y: pickY(p, view, stat === "p99"),
        _series: `${shortModel(p.model)} · ${shortGpu(p.gpu_type)} · TP${p.tp}/DP${p.dp}`,
      }))
      .filter((p) => p._x != null && p._y != null && (p._x as number) > 0 && (p._y as number) > 0);
  }, [points, view, gpuFilter, modelFilter, parFilter, stat]);

  // Group filtered points into series for distinct color/shape encoding.
  const series = useMemo(() => {
    const map = new Map<SeriesKey, typeof filtered>();
    for (const p of filtered) {
      const arr = map.get(p._series) ?? [];
      arr.push(p);
      map.set(p._series, arr);
    }
    // Sort each series by X so connecting lines look like sweep curves.
    const seriesArr = Array.from(map.entries()).map(([key, pts], i) => ({
      key,
      points: pts.sort((a, b) => (a._x as number) - (b._x as number)),
      color: COLORS[i % COLORS.length],
      shape: SHAPES[i % SHAPES.length] as Shape,
    }));
    return seriesArr;
  }, [filtered]);

  const activeView = VIEWS.find((v) => v.id === view)!;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-muted text-muted-foreground">
              <BarChart3 className="h-4 w-4" />
            </div>
            <div>
              <CardTitle className="text-sm">Performance explorer</CardTitle>
              <CardDescription className="text-xs">
                Every <span className="font-mono">result.json</span> across every benchmark, plotted together.
                Per-GPU throughput on log scale; color = model, shape = GPU.
              </CardDescription>
            </div>
          </div>
          <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            Refresh
          </Button>
        </div>
      </CardHeader>

      <CardContent>
        <div className="flex gap-4">
          {/* Sidebar — view switcher */}
          <nav className="hidden w-48 shrink-0 flex-col gap-1 lg:flex">
            <div className="mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
              Sections
            </div>
            {VIEWS.map((v) => (
              <button
                key={v.id}
                type="button"
                onClick={() => setView(v.id)}
                className={cn(
                  "rounded-md px-3 py-2 text-left text-sm transition-colors",
                  v.id === view
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted/40 hover:text-foreground",
                )}
              >
                {v.label}
              </button>
            ))}
          </nav>

          <div className="min-w-0 flex-1 space-y-3">
            {/* Mobile view switcher */}
            <div className="lg:hidden">
              <Select value={view} onValueChange={(v) => setView(v as ViewId)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {VIEWS.map((v) => (
                    <SelectItem key={v.id} value={v.id}>{v.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Filter row */}
            <div className="flex flex-wrap items-end gap-3">
              <FilterBox label="GPU type" value={gpuFilter} onChange={setGpuFilter} options={allGpus} format={shortGpu} />
              <FilterBox label="Model" value={modelFilter} onChange={setModelFilter} options={allModels} format={shortModel} />
              <FilterBox label="Parallelism" value={parFilter} onChange={setParFilter} options={allParallelisms} />
              <div>
                <Label className="text-[11px] text-muted-foreground">Stat</Label>
                <Select value={stat} onValueChange={(v) => setStat(v as "median" | "p99")}>
                  <SelectTrigger className="h-9 w-[100px]"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="median">Median</SelectItem>
                    <SelectItem value="p99">p99</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <button
                type="button"
                onClick={() => setLogScale((v) => !v)}
                className={cn(
                  "inline-flex h-9 items-center gap-1.5 rounded-md border px-3 text-xs font-medium transition-colors",
                  logScale
                    ? "border-violet-500/40 bg-violet-500/10 text-violet-400"
                    : "border-border bg-background text-muted-foreground hover:bg-muted/40",
                )}
              >
                <span className={cn(
                  "inline-block h-2 w-2 rounded-full",
                  logScale ? "bg-violet-400" : "bg-muted-foreground/40",
                )} />
                Log scale
              </button>
            </div>

            {error && (
              <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                {error}
              </div>
            )}

            {/* Chart */}
            <div className="rounded-lg border border-border p-3">
              <div className="mb-2">
                <h3 className="text-sm font-semibold">{activeView.label}</h3>
                <p className="text-[11px] text-muted-foreground">
                  {filtered.length} of {points?.length ?? 0} points · {series.length} series
                </p>
              </div>

              {points === null && loading ? (
                <div className="flex h-80 items-center justify-center text-sm text-muted-foreground">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Loading from S3 (cached 60 s)…
                </div>
              ) : filtered.length === 0 ? (
                <div className="flex h-80 items-center justify-center text-sm text-muted-foreground">
                  No data for the current filters. Run a benchmark sweep to populate this chart.
                </div>
              ) : (
                <div className="h-[480px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 12, right: 16, left: 8, bottom: 28 }}>
                      <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                      <XAxis
                        type="number"
                        dataKey="_x"
                        name={activeView.xLabel}
                        scale={logScale ? "log" : "linear"}
                        domain={["auto", "auto"]}
                        allowDataOverflow
                        tick={{ fontSize: 10, fill: "currentColor" }}
                        tickLine={false}
                        axisLine={false}
                        label={{
                          value: activeView.xLabel + (logScale ? " — log scale" : ""),
                          position: "insideBottom",
                          offset: -10,
                          fontSize: 11,
                          fill: "currentColor",
                        }}
                        stroke="currentColor"
                        className="text-muted-foreground"
                      />
                      <YAxis
                        type="number"
                        dataKey="_y"
                        name={activeView.yLabel}
                        scale={logScale ? "log" : "linear"}
                        domain={["auto", "auto"]}
                        allowDataOverflow
                        tick={{ fontSize: 10, fill: "currentColor" }}
                        tickLine={false}
                        axisLine={false}
                        width={70}
                        label={{
                          value: activeView.yLabel + (logScale ? " — log" : ""),
                          angle: -90,
                          position: "insideLeft",
                          fontSize: 11,
                          fill: "currentColor",
                        }}
                        stroke="currentColor"
                        className="text-muted-foreground"
                      />
                      <ZAxis range={[60, 60]} />
                      <Tooltip
                        cursor={{ strokeDasharray: "3 3" }}
                        content={<PointTooltip />}
                      />
                      {series.map((s) => (
                        <Scatter
                          key={s.key}
                          name={s.key}
                          data={s.points}
                          fill={s.color}
                          shape={s.shape}
                          line={{ stroke: s.color, strokeWidth: 1.5, strokeOpacity: 0.6 }}
                          isAnimationActive={false}
                        />
                      ))}
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Custom legend (recharts default is too crammed) */}
              {series.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {series.map((s) => (
                    <span
                      key={s.key}
                      className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/30 px-2 py-0.5 text-[10px]"
                      title={s.key}
                    >
                      <span
                        className="inline-block h-2 w-2 rounded-full"
                        style={{ background: s.color }}
                      />
                      <span className="font-mono">{s.key}</span>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function uniqueOf(points: AggregatePoint[], pick: (p: AggregatePoint) => string): string[] {
  return Array.from(new Set(points.map(pick))).sort();
}

function FilterBox({
  label,
  value,
  onChange,
  options,
  format,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
  format?: (s: string) => string;
}) {
  return (
    <div>
      <Label className="text-[11px] text-muted-foreground">{label}</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="h-9 w-[180px]">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="__all__">All</SelectItem>
          {options.map((o) => (
            <SelectItem key={o} value={o}>{format ? format(o) : o}</SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

function PointTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: AggregatePoint & { _x: number; _y: number; _series: string } }>;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs text-zinc-200 shadow-lg">
      <div className="mb-1 font-medium">{p.benchmark_name}</div>
      <div className="mb-2 text-[11px] text-zinc-500">{p._series}</div>
      <Row label="Throughput/GPU" value={fmt(p.output_throughput_per_gpu, 1, "tok/s")} />
      <Row label="Median TTFT" value={fmt(p.median_ttft_ms, 1, "ms")} />
      <Row label="Median TPOT" value={fmt(p.median_tpot_ms, 2, "ms")} />
      <Row label="Median E2EL" value={fmt(p.median_e2el_ms, 1, "ms")} />
      <Row label="Context" value={`${p.context_len} tok`} />
      <Row label="Concurrency" value={String(p.concurrency)} />
      <Row label="Output len" value={`${p.output_len} tok`} />
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 tabular-nums">
      <span className="text-zinc-500">{label}</span>
      <span className="font-mono text-zinc-100">{value}</span>
    </div>
  );
}

function fmt(v: number | null | undefined, digits: number, unit: string): string {
  if (v == null) return "—";
  if (Math.abs(v) >= 1000) return `${v.toFixed(0)} ${unit}`;
  return `${v.toFixed(digits)} ${unit}`;
}
