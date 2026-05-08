"use client";

import { useEffect, useMemo, useState } from "react";
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
import { Loader2, RefreshCw, TrendingUp, Zap, Clock, Activity } from "lucide-react";
import { Badge } from "@/components/ui/badge";
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
import type { BenchmarkRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

/** Shape of the rows we feed the table + charts. Sourced from vllm bench
 * serve's result.json (top-level keys) plus filename-extracted dimensions. */
type Row = {
  filename: string;
  input_len: number;
  output_len: number;
  num_prompts: number;
  concurrency: number;
  duration_s: number | null;
  output_throughput: number | null;
  request_throughput: number | null;
  total_token_throughput: number | null;
  mean_ttft_ms: number | null;
  median_ttft_ms: number | null;
  p99_ttft_ms: number | null;
  mean_tpot_ms: number | null;
  median_tpot_ms: number | null;
  p99_tpot_ms: number | null;
  mean_itl_ms: number | null;
  median_itl_ms: number | null;
  p99_itl_ms: number | null;
  mean_e2el_ms: number | null;
  median_e2el_ms: number | null;
  p99_e2el_ms: number | null;
};

type StatMode = "median" | "p99" | "mean";

// Distinct, accessible-ish line colors. Reused across all charts so the same
// input_len keeps the same color in throughput / TTFT / TPOT / E2EL.
const LINE_COLORS = [
  "#a78bfa", // violet-400
  "#22d3ee", // cyan-400
  "#34d399", // emerald-400
  "#fbbf24", // amber-400
  "#fb7185", // rose-400
  "#60a5fa", // blue-400
  "#f472b6", // pink-400
  "#facc15", // yellow-400
];

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function parseFilenameDims(name: string): {
  input_len: number;
  output_len: number;
  num_prompts: number;
  concurrency: number;
} {
  // benchmaq emits filenames like:
  //   sgpu-qwen-quick_qwen-quick_in256_out128_p50_c4_56c405.json
  const m = name.match(/_in(\d+)_out(\d+)_p(\d+)_c(\d+)/);
  return {
    input_len: m ? parseInt(m[1], 10) : 0,
    output_len: m ? parseInt(m[2], 10) : 0,
    num_prompts: m ? parseInt(m[3], 10) : 0,
    concurrency: m ? parseInt(m[4], 10) : 0,
  };
}

function rowFromJson(filename: string, json: Record<string, unknown>): Row {
  const dims = parseFilenameDims(filename);
  return {
    filename,
    input_len: dims.input_len,
    output_len: dims.output_len,
    num_prompts: (num(json.num_prompts) ?? dims.num_prompts) || 0,
    concurrency: (num(json.max_concurrency) ?? dims.concurrency) || 0,
    duration_s: num(json.duration),
    output_throughput: num(json.output_throughput),
    request_throughput: num(json.request_throughput),
    total_token_throughput: num(json.total_token_throughput),
    mean_ttft_ms: num(json.mean_ttft_ms),
    median_ttft_ms: num(json.median_ttft_ms),
    p99_ttft_ms: num(json.p99_ttft_ms),
    mean_tpot_ms: num(json.mean_tpot_ms),
    median_tpot_ms: num(json.median_tpot_ms),
    p99_tpot_ms: num(json.p99_tpot_ms),
    mean_itl_ms: num(json.mean_itl_ms),
    median_itl_ms: num(json.median_itl_ms),
    p99_itl_ms: num(json.p99_itl_ms),
    mean_e2el_ms: num(json.mean_e2el_ms),
    median_e2el_ms: num(json.median_e2el_ms),
    p99_e2el_ms: num(json.p99_e2el_ms),
  };
}

export function ResultsTab({ bench }: { bench: BenchmarkRecord }) {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [statMode, setStatMode] = useState<StatMode>("median");
  const [hiddenInputs, setHiddenInputs] = useState<Set<number>>(new Set());

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const files = await gateway.listBenchmarkFiles(bench.id);
      const jsonFiles = files.filter(
        (f) => f.name.toLowerCase().endsWith(".json") && !f.name.endsWith("_DONE"),
      );
      if (jsonFiles.length === 0) {
        setRows([]);
        return;
      }
      const parsed = await Promise.all(
        jsonFiles.map(async (f) => {
          try {
            const r = await fetch(f.download_url);
            if (!r.ok) throw new Error(`fetch ${f.name}: ${r.status}`);
            const json = (await r.json()) as Record<string, unknown>;
            return rowFromJson(f.name, json);
          } catch (e) {
            return null;
          }
        }),
      );
      setRows(parsed.filter((x): x is Row => x !== null));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setRows([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    // Auto-poll while bench is running (new result.json files trickle in
    // throughout a sweep).
    const isRunning = bench.status === "running" || bench.status === "queued";
    if (!isRunning) return;
    const t = setInterval(refresh, 12_000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bench.id, bench.status]);

  const inputLens = useMemo(() => {
    if (!rows) return [];
    return Array.from(new Set(rows.map((r) => r.input_len))).sort((a, b) => a - b);
  }, [rows]);

  const visibleRows = useMemo(() => {
    if (!rows) return [];
    return rows.filter((r) => !hiddenInputs.has(r.input_len));
  }, [rows, hiddenInputs]);

  if (rows === null && loading) {
    return (
      <div className="flex items-center justify-center rounded-md border border-border px-4 py-12 text-sm text-muted-foreground">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading results from S3…
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
        {error}
      </div>
    );
  }

  if (!rows || rows.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border px-6 py-12 text-center text-sm text-muted-foreground">
        {bench.status === "done" || bench.status === "failed"
          ? "No result.json files found in S3."
          : "Results will appear here as benchmaq writes them. The page auto-refreshes every 12 s while the run is in flight."}
      </div>
    );
  }

  // Best-of cards (always show, single result OR sweep).
  const bestThroughput = bestBy(rows, (r) => r.output_throughput);
  const bestTtft = bestBy(rows, (r) => r.median_ttft_ms, /*lower=*/ true);
  const bestTpot = bestBy(rows, (r) => r.median_tpot_ms, /*lower=*/ true);

  const showCharts = rows.length > 1; // single point: skip charts, show big numbers only

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Results</h2>
          <p className="text-xs text-muted-foreground">
            {rows.length} run{rows.length === 1 ? "" : "s"} · parsed from{" "}
            <span className="font-mono">result.json</span> in S3
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Tabs value={statMode} onValueChange={(v) => setStatMode(v as StatMode)}>
            <TabsList>
              <TabsTrigger value="median">Median</TabsTrigger>
              <TabsTrigger value="mean">Mean</TabsTrigger>
              <TabsTrigger value="p99">p99</TabsTrigger>
            </TabsList>
          </Tabs>
          <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            Refresh
          </Button>
        </div>
      </div>

      {/* Best-of KPI cards */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <KpiCard
          icon={<Zap className="h-4 w-4" />}
          label="Best throughput"
          value={bestThroughput ? `${bestThroughput.output_throughput!.toFixed(1)} tok/s` : "—"}
          sub={
            bestThroughput
              ? `c=${bestThroughput.concurrency} · in=${bestThroughput.input_len}`
              : undefined
          }
          accent="emerald"
        />
        <KpiCard
          icon={<Clock className="h-4 w-4" />}
          label="Lowest median TTFT"
          value={bestTtft ? `${bestTtft.median_ttft_ms!.toFixed(1)} ms` : "—"}
          sub={
            bestTtft
              ? `c=${bestTtft.concurrency} · in=${bestTtft.input_len}`
              : undefined
          }
          accent="violet"
        />
        <KpiCard
          icon={<Activity className="h-4 w-4" />}
          label="Lowest median TPOT"
          value={bestTpot ? `${bestTpot.median_tpot_ms!.toFixed(2)} ms` : "—"}
          sub={
            bestTpot
              ? `c=${bestTpot.concurrency} · in=${bestTpot.input_len}`
              : undefined
          }
          accent="cyan"
        />
      </div>

      {showCharts && inputLens.length > 1 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-muted-foreground">Filter input lengths:</span>
          {inputLens.map((il, i) => {
            const hidden = hiddenInputs.has(il);
            const color = LINE_COLORS[i % LINE_COLORS.length];
            return (
              <button
                key={il}
                type="button"
                onClick={() =>
                  setHiddenInputs((prev) => {
                    const n = new Set(prev);
                    if (n.has(il)) n.delete(il);
                    else n.add(il);
                    return n;
                  })
                }
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs transition-colors",
                  hidden
                    ? "border-border bg-background text-muted-foreground line-through"
                    : "border-border bg-muted/40 text-foreground",
                )}
              >
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ background: hidden ? "transparent" : color, borderColor: color, borderWidth: hidden ? 1 : 0 }}
                />
                in={il}
              </button>
            );
          })}
        </div>
      )}

      {showCharts && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <ChartCard
            title="Output throughput"
            subtitle="tokens/sec — higher is better"
            icon={<TrendingUp className="h-4 w-4" />}
          >
            <SweepChart
              rows={visibleRows}
              inputLens={inputLens}
              hidden={hiddenInputs}
              y={(r) => r.output_throughput}
              yLabel="tok/s"
            />
          </ChartCard>
          <ChartCard
            title="Time to first token"
            subtitle={`${statMode.toUpperCase()} TTFT (ms) — lower is better`}
            icon={<Clock className="h-4 w-4" />}
          >
            <SweepChart
              rows={visibleRows}
              inputLens={inputLens}
              hidden={hiddenInputs}
              y={(r) => statPick(r, "ttft", statMode)}
              yLabel="ms"
            />
          </ChartCard>
          <ChartCard
            title="Time per output token"
            subtitle={`${statMode.toUpperCase()} TPOT (ms) — lower is better`}
            icon={<Activity className="h-4 w-4" />}
          >
            <SweepChart
              rows={visibleRows}
              inputLens={inputLens}
              hidden={hiddenInputs}
              y={(r) => statPick(r, "tpot", statMode)}
              yLabel="ms"
            />
          </ChartCard>
          <ChartCard
            title="End-to-end latency"
            subtitle={`${statMode.toUpperCase()} E2EL (ms) — lower is better`}
            icon={<Clock className="h-4 w-4" />}
          >
            <SweepChart
              rows={visibleRows}
              inputLens={inputLens}
              hidden={hiddenInputs}
              y={(r) => statPick(r, "e2el", statMode)}
              yLabel="ms"
            />
          </ChartCard>
        </div>
      )}

      {/* Summary table — always shown, sortable */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Summary</CardTitle>
          <CardDescription className="text-xs">
            All {rows.length} runs. Click a column header to sort.
          </CardDescription>
        </CardHeader>
        <CardContent className="px-0 pb-0">
          <SummaryTable rows={rows} statMode={statMode} />
        </CardContent>
      </Card>
    </div>
  );
}

function bestBy(rows: Row[], pick: (r: Row) => number | null, lower = false): Row | null {
  let best: Row | null = null;
  let bestV: number | null = null;
  for (const r of rows) {
    const v = pick(r);
    if (v == null) continue;
    if (bestV == null || (lower ? v < bestV : v > bestV)) {
      best = r;
      bestV = v;
    }
  }
  return best;
}

function statPick(r: Row, metric: "ttft" | "tpot" | "itl" | "e2el", mode: StatMode): number | null {
  const k = `${mode}_${metric}_ms` as keyof Row;
  const v = r[k];
  return typeof v === "number" ? v : null;
}

function KpiCard({
  icon,
  label,
  value,
  sub,
  accent,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
  accent: "emerald" | "violet" | "cyan";
}) {
  const ring = {
    emerald: "ring-emerald-500/20 bg-emerald-500/5",
    violet: "ring-violet-500/20 bg-violet-500/5",
    cyan: "ring-cyan-500/20 bg-cyan-500/5",
  }[accent];
  const iconBg = {
    emerald: "bg-emerald-500/15 text-emerald-400",
    violet: "bg-violet-500/15 text-violet-400",
    cyan: "bg-cyan-500/15 text-cyan-400",
  }[accent];
  return (
    <div className={cn("rounded-lg border border-border p-4 ring-1", ring)}>
      <div className="flex items-center gap-2">
        <div className={cn("flex h-7 w-7 items-center justify-center rounded-md", iconBg)}>
          {icon}
        </div>
        <span className="text-xs uppercase tracking-wide text-muted-foreground">{label}</span>
      </div>
      <div className="mt-2 text-2xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-muted-foreground">{sub}</div>}
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  icon,
  children,
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
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
        <div className="h-64 w-full">{children}</div>
      </CardContent>
    </Card>
  );
}

function SweepChart({
  rows,
  inputLens,
  hidden,
  y,
  yLabel,
}: {
  rows: Row[];
  inputLens: number[];
  hidden: Set<number>;
  y: (r: Row) => number | null;
  yLabel: string;
}) {
  // recharts wants one row per X point with all series as columns. Pivot
  // (concurrency × input_len) -> { concurrency, "in128": ..., "in512": ... }
  const concurrencies = useMemo(
    () => Array.from(new Set(rows.map((r) => r.concurrency))).sort((a, b) => a - b),
    [rows],
  );
  const data = useMemo(
    () =>
      concurrencies.map((c) => {
        const row: Record<string, number | null> = { concurrency: c };
        for (const il of inputLens) {
          const match = rows.find((r) => r.concurrency === c && r.input_len === il);
          row[`in${il}`] = match ? y(match) : null;
        }
        return row;
      }),
    [concurrencies, inputLens, rows, y],
  );

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
        <XAxis
          dataKey="concurrency"
          stroke="currentColor"
          className="text-[10px] text-muted-foreground"
          tickLine={false}
          axisLine={false}
          label={{ value: "concurrency", position: "insideBottom", offset: -4, fontSize: 10, fill: "currentColor" }}
        />
        <YAxis
          stroke="currentColor"
          className="text-[10px] text-muted-foreground"
          tickLine={false}
          axisLine={false}
          width={48}
          label={{ value: yLabel, angle: -90, position: "insideLeft", fontSize: 10, fill: "currentColor" }}
        />
        <Tooltip
          contentStyle={{
            background: "rgb(24 24 27)",
            border: "1px solid rgb(63 63 70)",
            borderRadius: 6,
            fontSize: 11,
          }}
          labelStyle={{ color: "rgb(244 244 245)" }}
        />
        <Legend
          wrapperStyle={{ fontSize: 11 }}
          iconType="circle"
          iconSize={8}
        />
        {inputLens
          .filter((il) => !hidden.has(il))
          .map((il, i) => (
            <Line
              key={il}
              type="monotone"
              dataKey={`in${il}`}
              name={`in=${il}`}
              stroke={LINE_COLORS[i % LINE_COLORS.length]}
              strokeWidth={2}
              dot={{ r: 3 }}
              activeDot={{ r: 5 }}
              connectNulls
              isAnimationActive={false}
            />
          ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

type SortKey =
  | "input_len"
  | "concurrency"
  | "output_throughput"
  | "ttft"
  | "tpot"
  | "itl"
  | "e2el"
  | "duration_s";

function SummaryTable({ rows, statMode }: { rows: Row[]; statMode: StatMode }) {
  const [sortKey, setSortKey] = useState<SortKey>("input_len");
  const [asc, setAsc] = useState(true);

  const sorted = useMemo(() => {
    const get = (r: Row): number => {
      switch (sortKey) {
        case "input_len": return r.input_len;
        case "concurrency": return r.concurrency;
        case "output_throughput": return r.output_throughput ?? -Infinity;
        case "ttft": return statPick(r, "ttft", statMode) ?? Infinity;
        case "tpot": return statPick(r, "tpot", statMode) ?? Infinity;
        case "itl": return statPick(r, "itl", statMode) ?? Infinity;
        case "e2el": return statPick(r, "e2el", statMode) ?? Infinity;
        case "duration_s": return r.duration_s ?? Infinity;
      }
    };
    return [...rows].sort((a, b) => (asc ? get(a) - get(b) : get(b) - get(a)));
  }, [rows, sortKey, asc, statMode]);

  function header(label: string, key: SortKey, align: "left" | "right" = "right") {
    const active = sortKey === key;
    return (
      <th
        className={cn(
          "cursor-pointer select-none px-3 py-2 text-xs uppercase tracking-wide hover:text-foreground",
          align === "left" ? "text-left" : "text-right",
          active ? "text-foreground" : "text-muted-foreground",
        )}
        onClick={() => {
          if (sortKey === key) setAsc(!asc);
          else { setSortKey(key); setAsc(true); }
        }}
      >
        {label} {active && (asc ? "↑" : "↓")}
      </th>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-muted/40">
          <tr>
            {header("input_len", "input_len", "left")}
            {header("concurrency", "concurrency")}
            {header("throughput (tok/s)", "output_throughput")}
            {header(`TTFT (${statMode})`, "ttft")}
            {header(`TPOT (${statMode})`, "tpot")}
            {header(`ITL (${statMode})`, "itl")}
            {header(`E2EL (${statMode})`, "e2el")}
            {header("duration (s)", "duration_s")}
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {sorted.map((r) => (
            <tr key={r.filename}>
              <td className="px-3 py-1.5 font-mono text-xs">in={r.input_len} · out={r.output_len}</td>
              <td className="px-3 py-1.5 text-right font-mono text-xs">{r.concurrency}</td>
              <td className="px-3 py-1.5 text-right tabular-nums">
                {fmt(r.output_throughput, 1)}
              </td>
              <td className="px-3 py-1.5 text-right tabular-nums">
                {fmt(statPick(r, "ttft", statMode), 1)}
              </td>
              <td className="px-3 py-1.5 text-right tabular-nums">
                {fmt(statPick(r, "tpot", statMode), 2)}
              </td>
              <td className="px-3 py-1.5 text-right tabular-nums">
                {fmt(statPick(r, "itl", statMode), 2)}
              </td>
              <td className="px-3 py-1.5 text-right tabular-nums">
                {fmt(statPick(r, "e2el", statMode), 1)}
              </td>
              <td className="px-3 py-1.5 text-right tabular-nums text-muted-foreground">
                {fmt(r.duration_s, 1)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function fmt(v: number | null, digits: number): string {
  if (v == null) return "—";
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  return v.toFixed(digits);
}
