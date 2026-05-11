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
import { BarChart3, Loader2, RefreshCw } from "lucide-react";
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

// Monochrome series palette. Series stay distinguishable via shape (different
// model/GPU combos get different shapes); colour is only used to differentiate
// from the chart's neutral background — no decorative hues. Inside the colour
// rule (status/availability only).
const COLORS = [
  "#71717a", // zinc-500
  "#3f3f46", // zinc-700
  "#a1a1aa", // zinc-400
  "#52525b", // zinc-600
  "#27272a", // zinc-800
  "#d4d4d8", // zinc-300
  "#18181b", // zinc-900
  "#e4e4e7", // zinc-200
];

const SHAPES = ["circle", "triangle", "square", "diamond", "star", "cross", "wye"] as const;
type Shape = (typeof SHAPES)[number];

type MetricKey = "ttft" | "e2el";

const CHARTS: { key: MetricKey; title: string; yLabel: string }[] = [
  { key: "ttft", title: "TTFT vs Context",         yLabel: "TTFT (ms)" },
  { key: "e2el", title: "E2E latency vs Context",  yLabel: "E2E latency (ms)" },
];

function shortGpu(s: string | null | undefined): string {
  if (!s) return "—";
  return s.replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "").replace(/\s+80GB\s+(HBM3|PCIe).*$/i, " 80GB");
}

function shortModel(s: string | null | undefined): string {
  if (!s) return "—";
  return s.split("/").pop() ?? s;
}

function pickY(p: AggregatePoint, metric: MetricKey, useP99: boolean): number | null {
  if (metric === "ttft") return useP99 ? p.p99_ttft_ms : p.median_ttft_ms;
  return useP99 ? p.p99_e2el_ms : p.median_e2el_ms;
}

type SeriesKey = string;

export function BenchmarkExplorer({ scope = "mine" }: { scope?: "mine" | "all" }) {
  const [points, setPoints] = useState<AggregatePoint[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [gpuFilter, setGpuFilter] = useState<string>("__all__");
  const [modelFilter, setModelFilter] = useState<string>("__all__");
  const [parFilter, setParFilter] = useState<string>("__all__");
  const [logScale, setLogScale] = useState<boolean>(true);
  const [stat, setStat] = useState<"median" | "p99">("median");

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const data = await gateway.aggregateBenchmarks(scope);
      setPoints(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scope]);

  const allModels = useMemo(() => uniqueOf(points ?? [], (p) => p.model ?? "—"), [points]);
  const allGpus = useMemo(() => uniqueOf(points ?? [], (p) => p.gpu_type ?? "—"), [points]);
  const allParallelisms = useMemo(
    () => uniqueOf(points ?? [], (p) => `TP${p.tp}/DP${p.dp}`),
    [points],
  );

  // Apply filters once; both charts render off the same row set.
  const filtered = useMemo(() => {
    if (!points) return [];
    return points
      .filter((p) => modelFilter === "__all__" || (p.model ?? "—") === modelFilter)
      .filter((p) => gpuFilter === "__all__" || (p.gpu_type ?? "—") === gpuFilter)
      .filter((p) => parFilter === "__all__" || `TP${p.tp}/DP${p.dp}` === parFilter)
      .map((p) => ({
        ...p,
        _series: `${shortModel(p.model)} · ${shortGpu(p.gpu_type)} · TP${p.tp}/DP${p.dp}`,
      }))
      .filter((p) => p.context_len > 0);
  }, [points, gpuFilter, modelFilter, parFilter]);

  // Stable series → shape/colour map across both charts so the same model+GPU
  // combo looks identical in TTFT and E2E.
  const seriesStyles = useMemo(() => {
    const keys = Array.from(new Set(filtered.map((p) => p._series))).sort();
    const out = new Map<SeriesKey, { color: string; shape: Shape }>();
    keys.forEach((k, i) => {
      out.set(k, {
        color: COLORS[i % COLORS.length],
        shape: SHAPES[i % SHAPES.length] as Shape,
      });
    });
    return out;
  }, [filtered]);

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
                TTFT and end-to-end latency plotted against context length.
                Each series = model + GPU + parallelism.
              </CardDescription>
            </div>
          </div>
          <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            Refresh
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Filters apply to both charts. */}
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
                ? "border-foreground/60 bg-foreground/5 text-foreground"
                : "border-border bg-background text-muted-foreground hover:bg-muted/40",
            )}
          >
            <span className={cn(
              "inline-block h-2 w-2 rounded-full",
              logScale ? "bg-foreground" : "bg-muted-foreground/40",
            )} />
            Log scale
          </button>
        </div>

        {error && (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error}
          </div>
        )}

        {points === null && loading ? (
          <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Loading from S3 (cached 60 s)…
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
            No data for the current filters. Run a benchmark sweep to populate this chart.
          </div>
        ) : (
          <div className="grid gap-4 xl:grid-cols-2">
            {CHARTS.map((c) => (
              <ChartPanel
                key={c.key}
                metric={c.key}
                title={c.title}
                yLabel={c.yLabel}
                points={filtered}
                seriesStyles={seriesStyles}
                logScale={logScale}
                useP99={stat === "p99"}
              />
            ))}
          </div>
        )}

        {/* Shared legend — rendered once, applies to both charts. */}
        {filtered.length > 0 && (
          <div className="flex flex-wrap gap-2 pt-1">
            {Array.from(seriesStyles.entries()).map(([key, st]) => (
              <span
                key={key}
                className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/30 px-2 py-0.5 text-[10px]"
                title={key}
              >
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: st.color }} />
                <span className="font-mono">{key}</span>
              </span>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ChartPanel({
  metric,
  title,
  yLabel,
  points,
  seriesStyles,
  logScale,
  useP99,
}: {
  metric: MetricKey;
  title: string;
  yLabel: string;
  points: (AggregatePoint & { _series: string })[];
  seriesStyles: Map<SeriesKey, { color: string; shape: Shape }>;
  logScale: boolean;
  useP99: boolean;
}) {
  const decorated = useMemo(() => {
    return points
      .map((p) => ({
        ...p,
        _x: p.context_len,
        _y: pickY(p, metric, useP99),
      }))
      .filter((p) => p._x > 0 && p._y != null && (p._y as number) > 0);
  }, [points, metric, useP99]);

  const series = useMemo(() => {
    const m = new Map<SeriesKey, typeof decorated>();
    for (const p of decorated) {
      const arr = m.get(p._series) ?? [];
      arr.push(p);
      m.set(p._series, arr);
    }
    return Array.from(m.entries()).map(([key, pts]) => ({
      key,
      points: pts.sort((a, b) => a._x - b._x),
      ...(seriesStyles.get(key) ?? { color: "#71717a", shape: "circle" as Shape }),
    }));
  }, [decorated, seriesStyles]);

  return (
    <div className="rounded-lg border border-border p-3">
      <div className="mb-2">
        <h3 className="text-sm font-semibold">{title}</h3>
        <p className="text-[11px] text-muted-foreground">
          {decorated.length} points · {series.length} series
        </p>
      </div>
      <div className="h-[360px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 12, right: 16, left: 8, bottom: 28 }}>
            <CartesianGrid stroke="rgba(120,120,120,0.15)" />
            <XAxis
              type="number"
              dataKey="_x"
              name="Context length"
              scale={logScale ? "log" : "linear"}
              domain={["auto", "auto"]}
              allowDataOverflow
              tick={{ fontSize: 10, fill: "currentColor" }}
              tickLine={false}
              axisLine={false}
              label={{
                value: "Context length (tokens)" + (logScale ? " — log" : ""),
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
              name={yLabel}
              scale={logScale ? "log" : "linear"}
              domain={["auto", "auto"]}
              allowDataOverflow
              tick={{ fontSize: 10, fill: "currentColor" }}
              tickLine={false}
              axisLine={false}
              width={70}
              label={{
                value: yLabel + (logScale ? " — log" : ""),
                angle: -90,
                position: "insideLeft",
                fontSize: 11,
                fill: "currentColor",
              }}
              stroke="currentColor"
              className="text-muted-foreground"
            />
            <ZAxis range={[60, 60]} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<PointTooltip />} />
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
    </div>
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
      <Row label="Context" value={`${p.context_len} tok`} />
      <Row label="Median TTFT" value={fmt(p.median_ttft_ms, 1, "ms")} />
      <Row label="Median E2EL" value={fmt(p.median_e2el_ms, 1, "ms")} />
      <Row label="Concurrency" value={String(p.concurrency)} />
      <Row label="Output len" value={`${p.output_len} tok`} />
      <Row label="Throughput/GPU" value={fmt(p.output_throughput_per_gpu, 1, "tok/s")} />
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
