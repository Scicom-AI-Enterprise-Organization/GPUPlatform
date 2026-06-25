"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
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
import { ChevronDown, ChevronRight, Loader2, RefreshCw, TrendingUp, Zap, Clock, Activity, Target } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { BenchAccuracyResult, BenchmarkRecord } from "@/lib/types";
import {
  bestBy,
  fetchBenchRows,
  fmt,
  LINE_COLORS,
  perStreamOutputTps,
  type Row,
  type StatMode,
  statPick,
} from "@/lib/bench-results";
import { cn } from "@/lib/utils";

// Custom recharts legend content: recharts sorts its default legend
// lexicographically by dataKey (in=0, in=1024, in=128…). We re-sort the
// payload numerically by the digits in each label.
function SortedLegend(props: {
  payload?: ReadonlyArray<{ value?: unknown; color?: string; id?: string }>;
}) {
  const numOf = (v: unknown) => parseInt(String(v).replace(/\D/g, ""), 10) || 0;
  const items = (props.payload ?? []).slice().sort((a, b) => numOf(a.value) - numOf(b.value));
  return (
    <ul
      style={{
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        justifyContent: "center",
        gap: "4px 12px",
        listStyle: "none",
        margin: 0,
        padding: 0,
        paddingBottom: 8,
        fontSize: 11,
      }}
    >
      {items.map((it, i) => (
        <li key={it.id ?? String(i)} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
          <span
            style={{ display: "inline-block", width: 8, height: 8, borderRadius: 9999, background: it.color }}
          />
          <span>{String(it.value)}</span>
        </li>
      ))}
    </ul>
  );
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
      setRows(await fetchBenchRows(bench.id));
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

  // Accuracy-mode results live on the record (folded in from @@ACCURACY lines),
  // not in the S3 result.json rows the speed view fetches.
  const accuracy = useMemo<BenchAccuracyResult[]>(() => {
    const raw = (bench.result_json as Record<string, unknown> | null | undefined)?.accuracy;
    if (!Array.isArray(raw)) return [];
    return raw.filter(
      (a): a is BenchAccuracyResult =>
        !!a && typeof a === "object" && typeof (a as BenchAccuracyResult).accuracy === "number",
    );
  }, [bench.result_json]);

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
    // Accuracy runs have no benchmaq result.json rows — show the quality view.
    if (accuracy.length > 0) {
      return <AccuracyResults accuracy={accuracy} onRefresh={refresh} loading={loading} />;
    }
    return (
      <div className="rounded-md border border-dashed border-border px-6 py-12 text-center text-sm text-muted-foreground">
        {bench.status === "done" || bench.status === "failed"
          ? "No result.json files found in S3."
          : "Results will appear here as benchmaq writes them. The page auto-refreshes every 12 s while the run is in flight."}
      </div>
    );
  }

  // Best-of cards (always show, single result OR sweep).
  const bestThroughput = bestBy(rows, (r) => r.total_token_throughput);
  const bestIndivTps = bestBy(rows, (r) => perStreamOutputTps(r));
  const bestTtft = bestBy(rows, (r) => r.median_ttft_ms, /*lower=*/ true);
  const bestTpot = bestBy(rows, (r) => r.median_tpot_ms, /*lower=*/ true);

  const showCharts = rows.length > 1; // single point: skip charts, show big numbers only

  return (
    <div className="space-y-6">
      {accuracy.length > 0 && (
        <AccuracyResults accuracy={accuracy} onRefresh={refresh} loading={loading} />
      )}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
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
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <KpiCard
          icon={<Zap className="h-4 w-4" />}
          label="Best throughput (total)"
          value={
            bestThroughput?.total_token_throughput != null
              ? `${Math.round(bestThroughput.total_token_throughput).toLocaleString()} tok/s`
              : "—"
          }
          sub={
            bestThroughput
              ? `${bestThroughput.output_throughput?.toFixed(1) ?? "—"} out/s · c=${bestThroughput.concurrency} · in=${bestThroughput.input_len}`
              : undefined
          }
        />
        <KpiCard
          icon={<TrendingUp className="h-4 w-4" />}
          label="Best individual TPS"
          value={
            bestIndivTps != null
              ? `${perStreamOutputTps(bestIndivTps)!.toFixed(1)} tok/s`
              : "—"
          }
          sub={
            bestIndivTps
              ? `per stream · out/s ÷ c=${bestIndivTps.concurrency} · in=${bestIndivTps.input_len}`
              : undefined
          }
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
            All {rows.length} runs. Click a column header to sort, or a row to see
            every metric for that run.
          </CardDescription>
        </CardHeader>
        <CardContent className="px-0 pb-0">
          <SummaryTable rows={rows} statMode={statMode} />
        </CardContent>
      </Card>
    </div>
  );
}

// IQ-vs-speed view for accuracy runs. Each point is a (config × dataset):
// X = decode tok/s (the same speed axis as the throughput bench, measured over
// the eval requests), Y = accuracy %. Series are coloured by dataset so a
// multi-config run reads as "lower precision → further right; did quality hold?"
function AccuracyResults({
  accuracy,
  onRefresh,
  loading,
}: {
  accuracy: BenchAccuracyResult[];
  onRefresh: () => void;
  loading: boolean;
}) {
  const best = bestByAcc(accuracy);
  const fastest = accuracy.reduce<BenchAccuracyResult | null>(
    (acc, a) => (a.output_tok_s != null && (!acc || (a.output_tok_s ?? 0) > (acc.output_tok_s ?? 0)) ? a : acc),
    null,
  );

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Accuracy</h2>
          <p className="text-xs text-muted-foreground">
            {accuracy.length} eval{accuracy.length === 1 ? "" : "s"} · quality vs decode speed
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={onRefresh} disabled={loading}>
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <KpiCard
          icon={<Target className="h-4 w-4" />}
          label="Best accuracy"
          value={best ? `${(best.accuracy * 100).toFixed(1)}%` : "—"}
          sub={best ? `${best.config} · ${best.dataset}` : undefined}
        />
        <KpiCard
          icon={<Zap className="h-4 w-4" />}
          label="Fastest config"
          value={fastest?.output_tok_s != null ? `${fastest.output_tok_s.toFixed(0)} tok/s` : "—"}
          sub={fastest ? `${fastest.config} · ${(fastest.accuracy * 100).toFixed(1)}%` : undefined}
        />
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Accuracy detail</CardTitle>
          <CardDescription className="text-xs">
            Per config × dataset. Accuracy is exact-match (GSM8K) / single-letter (MMLU).
          </CardDescription>
        </CardHeader>
        <CardContent className="px-0 pb-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-muted/40">
                <tr>
                  <th className="px-3 py-2 text-left text-xs uppercase tracking-wide text-muted-foreground">config</th>
                  <th className="px-3 py-2 text-left text-xs uppercase tracking-wide text-muted-foreground">dataset</th>
                  <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">accuracy</th>
                  <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">correct / n</th>
                  <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">tok/s</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {accuracy.map((a, i) => (
                  <tr key={`${a.config}-${a.dataset}-${i}`}>
                    <td className="px-3 py-1.5 font-mono text-xs">{a.config}</td>
                    <td className="px-3 py-1.5 font-mono text-xs">{a.dataset}</td>
                    <td className="px-3 py-1.5 text-right tabular-nums">{(a.accuracy * 100).toFixed(1)}%</td>
                    <td className="px-3 py-1.5 text-right tabular-nums text-muted-foreground">
                      {a.correct ?? "—"} / {a.n}
                    </td>
                    <td className="px-3 py-1.5 text-right tabular-nums">{fmt(a.output_tok_s ?? null, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Full metric breakdown for evals that report more than a single accuracy
          (e.g. Function-Call-TaaS). One card per such eval, every metric shown. */}
      {accuracy
        .filter((a) => a.metrics && typeof a.metrics === "object")
        .map((a, i) => (
          <MetricsCard key={`metrics-${a.config}-${a.dataset}-${i}`} entry={a} />
        ))}
    </div>
  );
}

// All metrics for a single eval (config × dataset) that emitted a `metrics` bag
// — surfaces every field (Function-Call-TaaS reports ~14 + a counts breakdown),
// not just the headline accuracy in the table above.
function MetricsCard({ entry }: { entry: BenchAccuracyResult }) {
  const m = (entry.metrics ?? {}) as Record<string, unknown>;
  const counts =
    m._counts && typeof m._counts === "object"
      ? (m._counts as Record<string, unknown>)
      : null;
  const rows = Object.entries(m).filter(([k]) => k !== "_counts");
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">{entry.dataset} — all metrics</CardTitle>
        <CardDescription className="text-xs">
          {entry.config} · n={entry.n}
          {entry.errors ? ` · ${entry.errors} errors` : ""}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3 lg:grid-cols-4">
          {rows.map(([k, v]) => (
            <div key={k}>
              <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
                {metricLabel(k)}
              </div>
              <div className="text-sm font-semibold tabular-nums">{fmtAccVal(v)}</div>
            </div>
          ))}
        </div>
        {counts && Object.keys(counts).length > 0 && (
          <div className="mt-4 border-t border-border pt-3">
            <div className="mb-1.5 text-[11px] uppercase tracking-wide text-muted-foreground">
              counts
            </div>
            <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted-foreground">
              {Object.entries(counts).map(([k, v]) => (
                <span key={k}>
                  <span className="font-medium tabular-nums text-foreground">{String(v)}</span>{" "}
                  {metricLabel(k)}
                </span>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// "tool_call_f1" → "tool call F1"; "json_valid_rate" → "json valid rate".
function metricLabel(k: string): string {
  return k.replace(/_/g, " ").replace(/\bf1\b/gi, "F1").trim();
}

// Rates (0..1) render as a percentage; out-of-range numbers raw; null → "—".
function fmtAccVal(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "number")
    return Math.abs(v) <= 1 ? `${(v * 100).toFixed(1)}%` : v.toFixed(2);
  if (typeof v === "boolean") return v ? "yes" : "no";
  return String(v);
}

function bestByAcc(rows: BenchAccuracyResult[]): BenchAccuracyResult | null {
  return rows.reduce<BenchAccuracyResult | null>(
    (acc, r) => (!acc || r.accuracy > acc.accuracy ? r : acc),
    null,
  );
}

function KpiCard({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-muted text-muted-foreground">
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
          label={{ value: "concurrency", position: "insideBottom", offset: -16, fontSize: 10, fill: "currentColor" }}
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
          // recharts colors each item's text with the series stroke; the darkest
          // series (in=128 → zinc-900) is invisible on this dark tooltip. Force
          // light item text — the colored marker still distinguishes series.
          itemStyle={{ color: "rgb(228 228 231)" }}
          itemSorter={(item) =>
            parseInt(String(item.name).replace(/\D/g, ""), 10) || 0
          }
        />
        {/* Custom content: recharts sorts the default legend lexicographically
            by dataKey (in=0, in=1024, in=128…). We re-sort numerically. */}
        <Legend verticalAlign="top" align="center" content={<SortedLegend />} />
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
  | "total_token_throughput"
  | "output_throughput"
  | "individual_tps"
  | "ttft"
  | "tpot"
  | "itl"
  | "e2el"
  | "duration_s";

// Known result.json keys we render with friendly labels in the full-metrics
// panel. Anything numeric NOT in here (or already shown elsewhere) is listed
// under "Other" so nothing from the raw file is hidden.
const METRIC_LABELS: Record<string, string> = {
  completed: "Successful requests",
  total_input_tokens: "Total input tokens",
  total_output_tokens: "Total generated tokens",
  duration: "Benchmark duration (s)",
  request_throughput: "Request throughput (req/s)",
  output_throughput: "Output token throughput (tok/s)",
  total_token_throughput: "Total token throughput (tok/s)",
  peak_output_token_throughput: "Peak output token throughput (tok/s)",
  peak_concurrent_requests: "Peak concurrent requests",
  request_goodput: "Request goodput (req/s)",
  mean_ttft_ms: "Mean TTFT (ms)",
  median_ttft_ms: "Median TTFT (ms)",
  std_ttft_ms: "Std TTFT (ms)",
  p99_ttft_ms: "P99 TTFT (ms)",
  mean_tpot_ms: "Mean TPOT (ms)",
  median_tpot_ms: "Median TPOT (ms)",
  std_tpot_ms: "Std TPOT (ms)",
  p99_tpot_ms: "P99 TPOT (ms)",
  mean_itl_ms: "Mean ITL (ms)",
  median_itl_ms: "Median ITL (ms)",
  std_itl_ms: "Std ITL (ms)",
  p99_itl_ms: "P99 ITL (ms)",
  mean_e2el_ms: "Mean E2EL (ms)",
  median_e2el_ms: "Median E2EL (ms)",
  std_e2el_ms: "Std E2EL (ms)",
  p99_e2el_ms: "P99 E2EL (ms)",
};

// Groups define the panel layout; keys not present in a run are skipped.
const METRIC_GROUPS: { title: string; keys: string[] }[] = [
  {
    title: "Requests",
    keys: ["completed", "total_input_tokens", "total_output_tokens", "duration"],
  },
  {
    title: "Throughput",
    keys: [
      "request_throughput",
      "output_throughput",
      "total_token_throughput",
      "peak_output_token_throughput",
      "peak_concurrent_requests",
      "request_goodput",
    ],
  },
  { title: "Time to first token", keys: ["mean_ttft_ms", "median_ttft_ms", "std_ttft_ms", "p99_ttft_ms"] },
  { title: "Time per output token", keys: ["mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms"] },
  { title: "Inter-token latency", keys: ["mean_itl_ms", "median_itl_ms", "std_itl_ms", "p99_itl_ms"] },
  { title: "End-to-end latency", keys: ["mean_e2el_ms", "median_e2el_ms", "std_e2el_ms", "p99_e2el_ms"] },
];

function fmtRunMetric(key: string, v: number): string {
  // Token/request counts are integers; everything else gets 2 decimals.
  if (
    key.endsWith("_tokens") ||
    key === "completed" ||
    key === "peak_concurrent_requests"
  ) {
    return v.toLocaleString();
  }
  return v.toFixed(2);
}

function RunMetricsPanel({ row }: { row: Row }) {
  const shown = new Set<string>(["max_concurrency", "num_prompts", "input_len", "output_len"]);
  const indivTps = perStreamOutputTps(row);
  const groups = METRIC_GROUPS.map((g) => {
    const items = g.keys
      .filter((k) => {
        const v = row.raw[k];
        return typeof v === "number" && Number.isFinite(v);
      })
      .map((k) => {
        shown.add(k);
        return { key: k, label: METRIC_LABELS[k] ?? k, value: row.raw[k] };
      });
    // Derived: per-stream output rate (output tok/s ÷ concurrency) alongside the
    // aggregate throughputs — the "individual TPS" management asks for.
    if (g.title === "Throughput" && indivTps != null) {
      items.push({ key: "individual_tps", label: "Individual TPS (tok/s/req)", value: indivTps });
    }
    return { title: g.title, items };
  }).filter((g) => g.items.length > 0);

  // Anything numeric in result.json we didn't place in a group above.
  const other = Object.keys(row.raw)
    .filter((k) => !shown.has(k))
    .sort()
    .map((k) => ({ key: k, label: METRIC_LABELS[k] ?? k, value: row.raw[k] }));
  if (other.length > 0) groups.push({ title: "Other", items: other });

  return (
    <div className="grid grid-cols-1 gap-4 bg-muted/20 px-4 py-3 sm:grid-cols-2 lg:grid-cols-3">
      {groups.map((g) => (
        <div key={g.title}>
          <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            {g.title}
          </div>
          <dl className="space-y-1">
            {g.items.map((it) => (
              <div key={it.key} className="flex items-baseline justify-between gap-3 text-xs">
                <dt className="text-muted-foreground">{it.label}</dt>
                <dd className="font-mono tabular-nums">{fmtRunMetric(it.key, it.value)}</dd>
              </div>
            ))}
          </dl>
        </div>
      ))}
    </div>
  );
}

function SummaryTable({ rows, statMode }: { rows: Row[]; statMode: StatMode }) {
  const [sortKey, setSortKey] = useState<SortKey>("input_len");
  const [asc, setAsc] = useState(true);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const toggle = (fn: string) =>
    setExpanded((prev) => {
      const n = new Set(prev);
      if (n.has(fn)) n.delete(fn);
      else n.add(fn);
      return n;
    });

  const sorted = useMemo(() => {
    const get = (r: Row): number => {
      switch (sortKey) {
        case "input_len": return r.input_len;
        case "concurrency": return r.concurrency;
        case "total_token_throughput": return r.total_token_throughput ?? -Infinity;
        case "output_throughput": return r.output_throughput ?? -Infinity;
        case "individual_tps": return perStreamOutputTps(r) ?? -Infinity;
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
            <th className="w-8 px-2 py-2" />
            {header("input_len", "input_len", "left")}
            {header("concurrency", "concurrency")}
            {header("total tok/s", "total_token_throughput")}
            {header("output tok/s", "output_throughput")}
            {header("indiv tok/s", "individual_tps")}
            {header(`TTFT (${statMode})`, "ttft")}
            {header(`TPOT (${statMode})`, "tpot")}
            {header(`ITL (${statMode})`, "itl")}
            {header(`E2EL (${statMode})`, "e2el")}
            {header("duration (s)", "duration_s")}
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {sorted.map((r) => {
            const isOpen = expanded.has(r.filename);
            return (
              <Fragment key={r.filename}>
                <tr
                  className="cursor-pointer hover:bg-muted/30"
                  onClick={() => toggle(r.filename)}
                  title="Show all metrics for this run"
                >
                  <td className="px-2 py-1.5 text-muted-foreground">
                    {isOpen ? (
                      <ChevronDown className="h-3.5 w-3.5" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5" />
                    )}
                  </td>
                  <td className="px-3 py-1.5 font-mono text-xs">in={r.input_len} · out={r.output_len}</td>
                  <td className="px-3 py-1.5 text-right font-mono text-xs">{r.concurrency}</td>
                  <td className="px-3 py-1.5 text-right tabular-nums">
                    {r.total_token_throughput != null
                      ? Math.round(r.total_token_throughput).toLocaleString()
                      : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums">
                    {fmt(r.output_throughput, 1)}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums font-medium">
                    {fmt(perStreamOutputTps(r), 1)}
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
                {isOpen && (
                  <tr>
                    <td colSpan={11} className="p-0">
                      <RunMetricsPanel row={r} />
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
