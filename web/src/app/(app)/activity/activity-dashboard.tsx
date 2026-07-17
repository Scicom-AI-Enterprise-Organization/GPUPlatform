"use client";

// OpenRouter-style usage dashboard over all serverless + LLM-proxy requests:
// stat cards (requests, success rate, tokens in/out, TTFT/latency percentiles), a
// latency-percentile time-series, latency-by-upstream + latency-by-model over time,
// success/error + serverless-vs-proxy splits, requests-by-model + token-volume bars,
// and top users/models. Self-hosted → no $ spend. Time range (24h default / 7d / 30d
// / custom) + CSV export mirror /admin/analytics; granularity is 15-min / hour / day.
// A TTFT⇄E2E-latency toggle drives the three latency charts.
import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Check, ChevronDown, Copy, Download, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu, DropdownMenuCheckboxItem, DropdownMenuContent,
  DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { gateway } from "@/lib/gateway";
import type { ActivitySummary, ActivityGranularity } from "@/lib/types";

const RANGES = {
  "1h": { label: "Last 1 hour", hours: 1 },
  "6h": { label: "Last 6 hours", hours: 6 },
  "12h": { label: "Last 12 hours", hours: 12 },
  "24h": { label: "Last 24 hours", hours: 24 },
  "3d": { label: "Last 3 days", hours: 72 },
  "7d": { label: "Last 7 days", hours: 168 },
  "30d": { label: "Last 30 days", hours: 720 },
  custom: { label: "Custom range", hours: 0 },
} as const;
type RangeKey = keyof typeof RANGES;
const GRANS: { value: ActivityGranularity; label: string }[] = [
  { value: "15min", label: "Every 15 min" },
  { value: "hour", label: "Hourly" },
  { value: "day", label: "Daily" },
];
// Which timing metric the latency charts plot.
type LatMetric = "latency" | "ttft";

const localDate = (d: Date) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;

// Resolve the selected range to ISO since/until. Date.now()/new Date() are only
// "impure" during render — fine here since this runs inside the load callback.
function rangeBounds(range: RangeKey, from: string, to: string): { since: string; until?: string } {
  if (range === "custom") {
    const parse = (s: string, end: boolean) => {
      const [y, m, d] = s.split("-").map(Number);
      return new Date(y, (m ?? 1) - 1, d ?? 1, end ? 23 : 0, end ? 59 : 0, end ? 59 : 0, end ? 999 : 0);
    };
    const since = (from ? parse(from, false) : new Date(Date.now() - 6 * 86400_000)).toISOString();
    const until = (to ? parse(to, true) : new Date()).toISOString();
    return { since, until };
  }
  return { since: new Date(Date.now() - RANGES[range].hours * 3600_000).toISOString() };
}

// ---- shareable URL state ----------------------------------------------------
// Every control mirrors into the query string (via history.replaceState — no
// navigation, so the auth'd server component isn't re-run) so an admin can copy
// the address bar and hand someone the exact same filtered view. Missing param =
// its default, keeping the default view a clean `/activity`.
type UrlState = {
  metric: LatMetric; gran: ActivityGranularity; range: RangeKey;
  from: string; to: string; models: string[];
};
function parseUrlState(search: string): UrlState {
  const p = new URLSearchParams(search);
  const metric: LatMetric = p.get("metric") === "ttft" ? "ttft" : "latency";
  const g = p.get("gran");
  const gran: ActivityGranularity = g === "15min" || g === "day" ? g : "hour";
  const r = p.get("range") ?? "";
  const range: RangeKey = Object.prototype.hasOwnProperty.call(RANGES, r) ? (r as RangeKey) : "24h";
  // Tolerate both repeated (?models=a&models=b) and comma-joined (?models=a,b).
  const models = p.getAll("models").flatMap((v) => v.split(",")).map((s) => s.trim()).filter(Boolean);
  return { metric, gran, range, from: p.get("from") ?? "", to: p.get("to") ?? "", models };
}
function buildUrlQuery(s: UrlState): string {
  const p = new URLSearchParams();
  if (s.metric !== "latency") p.set("metric", s.metric);
  if (s.gran !== "hour") p.set("gran", s.gran);
  if (s.range !== "24h") p.set("range", s.range);
  if (s.range === "custom") {
    if (s.from) p.set("from", s.from);
    if (s.to) p.set("to", s.to);
  }
  for (const m of s.models) p.append("models", m);
  return p.toString();
}

function downloadCsv(filename: string, rows: (string | number)[][]) {
  const esc = (v: string | number) => {
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const blob = new Blob([rows.map((r) => r.map(esc).join(",")).join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
const PALETTE = ["#3b82f6", "#f59e0b", "#10b981", "#a855f7", "#ef4444", "#06b6d4", "#ec4899", "#84cc16", "#6366f1", "#94a3b8"];
// Fixed colors for the percentile lines (median → tail) and the categorical splits.
const PCT_COLORS = { avg: "#94a3b8", p50: "#10b981", p95: "#f59e0b", p99: "#ef4444" };
const STATUS_META: Record<string, { label: string; color: string }> = {
  ok: { label: "OK", color: "#10b981" },
  error: { label: "Error", color: "#ef4444" },
  pending: { label: "Pending", color: "#f59e0b" },
};
const SOURCE_META: Record<string, { label: string; color: string }> = {
  serverless: { label: "Serverless", color: "#3b82f6" },
  proxy: { label: "Proxy", color: "#a855f7" },
};

const fmtNum = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n);
const fmtMs = (ms: number | null | undefined) =>
  ms == null ? "—" : ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
const fmtPct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

// Bucket keys are ISO-ish: "YYYY-MM-DD" (day), "...THH:00" (hour), "...THH:MM" (minute).
const tickFmt = (b: string, g: ActivityGranularity) =>
  !b ? b
    : g === "day" ? b.slice(5)
    : g === "hour" ? `${b.slice(5, 13).replace("T", " ")}h`
    : b.slice(5, 16).replace("T", " ");
const fullLabel = (b: string) => String(b).replace("T", " ");
// Series-key ordering: alphabetical, with the "other" bucket always pinned last so
// its color/position stays stable as the tail shifts across reloads.
const orderKeys = (keys: string[]) => {
  const rest = keys.filter((k) => k !== "other").sort((a, b) => a.localeCompare(b));
  return keys.includes("other") ? [...rest, "other"] : rest;
};

export function ActivityDashboard() {
  const searchParams = useSearchParams();
  // Parse the initial view from the URL ONCE. Hydration-safe because useSearchParams is
  // populated identically on server + client, so lazily-initialized state matches the SSR
  // markup. Later URL edits are driven by us (history.replaceState below), so we deliberately
  // don't re-read on every searchParams change.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const initial = useMemo(() => parseUrlState(searchParams.toString()), []);
  const initialToday = useMemo(() => new Date(), []);
  const todayStr = localDate(initialToday);
  const [range, setRange] = useState<RangeKey>(initial.range);
  const [customFrom, setCustomFrom] = useState(initial.from || localDate(new Date(initialToday.getTime() - 6 * 86400_000)));
  const [customTo, setCustomTo] = useState(initial.to || todayStr);
  const [gran, setGran] = useState<ActivityGranularity>(initial.gran);
  const [latMetric, setLatMetric] = useState<LatMetric>(initial.metric);
  // Model filter — empty means "all models" (no server-side filter applied).
  const [selectedModels, setSelectedModels] = useState<string[]>(initial.models);
  const [summary, setSummary] = useState<ActivitySummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);  // flashes the Copy-link button

  const load = useCallback(async () => {
    setLoading(true); setErr(null);
    try {
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
      const { since, until } = rangeBounds(range, customFrom, customTo);
      setSummary(await gateway.getActivity({
        since, until, tz, granularity: gran, top: 8,
        models: selectedModels.length ? selectedModels : undefined,
      }));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [range, customFrom, customTo, gran, selectedModels]);
  // Mirror control state back into the URL (history.replaceState — no navigation, so the
  // auth'd server component isn't re-run) so the address bar is always a shareable link.
  useEffect(() => {
    const qs = buildUrlQuery({ metric: latMetric, gran, range, from: customFrom, to: customTo, models: selectedModels });
    window.history.replaceState(null, "", qs ? `${window.location.pathname}?${qs}` : window.location.pathname);
  }, [latMetric, gran, range, customFrom, customTo, selectedModels]);
  useEffect(() => {
    const timer = window.setTimeout(() => { void load(); }, 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  const copyLink = useCallback(() => {
    if (navigator.clipboard?.writeText) void navigator.clipboard.writeText(window.location.href);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }, []);

  const exportCsv = useCallback(() => {
    const bb = summary?.by_bucket ?? [];
    const slug = range === "custom" ? `${customFrom}_${customTo}` : range;
    downloadCsv(`activity-${slug}.csv`, [
      ["bucket", "requests", "prompt_tokens", "completion_tokens", "total_tokens",
       "avg_ttft_ms", "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms"],
      ...bb.map((b) => [
        b.bucket, b.requests, b.prompt_tokens, b.completion_tokens,
        b.prompt_tokens + b.completion_tokens, b.avg_ttft_ms ?? "", b.avg_latency_ms ?? "",
        b.p50_latency_ms ?? "", b.p95_latency_ms ?? "", b.p99_latency_ms ?? "",
      ]),
    ]);
  }, [summary, range, customFrom, customTo]);

  // Pivot into one row per bucket with a column per series (for the stacked bars /
  // multi-line charts). ⚠ recharts resolves a string dataKey via lodash `get()`, so a
  // series NAME containing a "." (e.g. "meta-llama/Llama-3.1-8B", a "host.com" upstream)
  // is misread as a nested path and renders nothing — so we key columns by a synthetic
  // dot-free id (`s0`, `s1`, …) and carry the display name on each `<Bar>/<Line>`'s `name`
  // prop instead. `order` optionally pins "other" last (latency charts) for stable colors.
  const pivotSeries = useCallback(
    (rows: { bucket: string }[] | undefined, key: string, val: string, order = false) => {
      let names = Array.from(new Set((rows ?? []).map((r) => (r as Record<string, string>)[key])));
      if (order) names = orderKeys(names);
      const series = names.map((name, i) => ({ id: `s${i}`, name }));
      const idOf = new Map(series.map((s) => [s.name, s.id]));
      const byBucket: Record<string, Record<string, number | null | string>> = {};
      for (const r of rows ?? []) {
        const rec = r as Record<string, string | number | null>;
        (byBucket[r.bucket] ??= { bucket: r.bucket })[idOf.get(rec[key] as string)!] = rec[val] as number;
      }
      const out = Object.values(byBucket).sort((a, b) => String(a.bucket).localeCompare(String(b.bucket)));
      return { rows: out, series };
    },
    [],
  );
  const latField = latMetric === "ttft" ? "avg_ttft_ms" : "avg_latency_ms";
  const { rows: modelBuckets, series: modelSeries } = useMemo(
    () => pivotSeries(summary?.by_model_bucket, "model", "requests"), [summary, pivotSeries]);
  const { rows: userBuckets, series: userSeries } = useMemo(
    () => pivotSeries(summary?.by_user_bucket, "user", "tokens"), [summary, pivotSeries]);
  const { rows: upstreamBuckets, series: upstreamSeries } = useMemo(
    () => pivotSeries(summary?.by_upstream_bucket, "upstream", "requests"), [summary, pivotSeries]);
  const { rows: statusBuckets, series: statusSeries } = useMemo(
    () => pivotSeries(summary?.by_status_bucket, "status", "requests"), [summary, pivotSeries]);
  const { rows: sourceBuckets, series: sourceSeries } = useMemo(
    () => pivotSeries(summary?.by_source_bucket, "source", "requests"), [summary, pivotSeries]);
  const { rows: upstreamLatRows, series: upstreamLatSeries } = useMemo(
    () => pivotSeries(summary?.by_upstream_latency_bucket, "series", latField, true), [summary, latField, pivotSeries]);
  const { rows: modelLatRows, series: modelLatSeries } = useMemo(
    () => pivotSeries(summary?.by_model_latency_bucket, "series", latField, true), [summary, latField, pivotSeries]);

  // dataKeys for the percentile chart, switched by the metric toggle.
  const M = latMetric === "ttft"
    ? { avg: "avg_ttft_ms", p50: "p50_ttft_ms", p95: "p95_ttft_ms", p99: "p99_ttft_ms" }
    : { avg: "avg_latency_ms", p50: "p50_latency_ms", p95: "p95_latency_ms", p99: "p99_latency_ms" };
  const metricLabel = latMetric === "ttft" ? "TTFT" : "E2E latency";

  const t = summary?.totals;
  const cards = [
    { label: "Requests", value: t ? fmtNum(t.requests) : "—",
      sub: t ? `${fmtNum(t.requests_ok)} ok · ${fmtNum(t.requests_error)} err` : "Serverless + proxy" },
    { label: "Success rate", value: fmtPct(t?.success_rate),
      sub: t ? `${fmtNum(t.requests_error)} errors · ${fmtNum(t.requests_pending)} pending` : "OK ÷ total" },
    { label: "Tokens in", value: t ? fmtNum(t.prompt_tokens) : "—", sub: "Prompt tokens" },
    { label: "Tokens out", value: t ? fmtNum(t.completion_tokens) : "—", sub: "Completion tokens" },
    { label: "TTFT p50", value: fmtMs(t?.p50_ttft_ms),
      sub: `p95 ${fmtMs(t?.p95_ttft_ms)} · p99 ${fmtMs(t?.p99_ttft_ms)}` },
    { label: "Latency p50", value: fmtMs(t?.p50_latency_ms),
      sub: `p95 ${fmtMs(t?.p95_latency_ms)} · p99 ${fmtMs(t?.p99_latency_ms)}` },
  ];

  const commonAxes = (numeric = false) => (
    <>
      <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
      <XAxis dataKey="bucket" tick={{ fontSize: 11 }} tickFormatter={(b) => tickFmt(b, gran)} minTickGap={28} />
      <YAxis tick={{ fontSize: 11 }} tickFormatter={numeric ? fmtNum : (v) => fmtMs(Number(v))}
        width={numeric ? 44 : 56} allowDecimals={!numeric} />
    </>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Activity</h1>
        <p className="mt-0.5 text-sm text-muted-foreground">
          Usage across serverless endpoints + LLM proxies — who, which endpoint, model, upstream, tokens, TTFT, latency.
        </p>
      </div>

      {/* Controls bar */}
      <div className="flex flex-wrap items-center gap-2 rounded-lg border bg-card px-4 py-3">
        <ModelFilter all={summary?.all_models ?? []} selected={selectedModels} onChange={setSelectedModels} />
        <Segmented
          value={latMetric}
          onChange={setLatMetric}
          options={[{ value: "latency", label: "E2E latency" }, { value: "ttft", label: "TTFT" }]}
        />
        {loading && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <Button variant="outline" size="sm" onClick={copyLink} title="Copy a shareable link to this exact view">
            {copied ? <Check className="mr-1.5 h-3.5 w-3.5" /> : <Copy className="mr-1.5 h-3.5 w-3.5" />}
            {copied ? "Copied!" : "Copy link"}
          </Button>
          <Button variant="outline" size="sm" onClick={exportCsv} disabled={loading || !summary?.by_bucket.length}>
            <Download className="mr-1.5 h-3.5 w-3.5" /> Export CSV
          </Button>
          {range === "custom" && (
            <div className="flex items-center gap-1.5">
              <input
                type="date"
                value={customFrom}
                max={customTo || todayStr}
                onChange={(e) => setCustomFrom(e.target.value)}
                className="h-8 rounded-md border bg-background px-2 text-xs text-foreground"
              />
              <span className="text-xs text-muted-foreground">to</span>
              <input
                type="date"
                value={customTo}
                min={customFrom}
                max={todayStr}
                onChange={(e) => setCustomTo(e.target.value)}
                className="h-8 rounded-md border bg-background px-2 text-xs text-foreground"
              />
            </div>
          )}
          <Select value={gran} onValueChange={(v) => setGran(v as ActivityGranularity)}>
            <SelectTrigger className="w-[130px]" size="sm"><SelectValue /></SelectTrigger>
            <SelectContent>
              {GRANS.map((g) => <SelectItem key={g.value} value={g.value}>{g.label}</SelectItem>)}
            </SelectContent>
          </Select>
          <Select value={range} onValueChange={(v) => setRange(v as RangeKey)}>
            <SelectTrigger className="w-[150px]" size="sm"><SelectValue /></SelectTrigger>
            <SelectContent>
              {Object.entries(RANGES).map(([k, v]) => <SelectItem key={k} value={k}>{v.label}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
      </div>

      {err && (
        <p className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {err}
        </p>
      )}

      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border bg-border md:grid-cols-3 lg:grid-cols-6">
        {cards.map((c) => (
          <div key={c.label} className="bg-card px-5 py-4">
            <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{c.label}</div>
            <div className="mt-1 text-2xl font-semibold tabular-nums">
              {loading && !summary ? <Loader2 className="h-5 w-5 animate-spin" /> : c.value}
            </div>
            <div className="mt-0.5 text-xs text-muted-foreground">{c.sub}</div>
          </div>
        ))}
      </div>

      {/* Latency percentiles + latency by upstream */}
      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard title={`${metricLabel} percentiles over time`}
          subtitle="p50 / p95 / p99 (avg dashed) per bucket — tail latency, not just the mean."
          empty={!summary?.by_bucket.length} loading={loading}>
          <LineChart data={summary?.by_bucket ?? []}>
            {commonAxes()}
            <Tooltip contentStyle={{ fontSize: 12 }} labelFormatter={(l) => fullLabel(String(l))}
              formatter={(v, n) => [fmtMs(Number(v)), String(n)]} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey={M.avg} name="avg" stroke={PCT_COLORS.avg} strokeWidth={1.5}
              strokeDasharray="4 3" dot={false} connectNulls />
            <Line type="monotone" dataKey={M.p50} name="p50" stroke={PCT_COLORS.p50} strokeWidth={2} dot={false} connectNulls />
            <Line type="monotone" dataKey={M.p95} name="p95" stroke={PCT_COLORS.p95} strokeWidth={2} dot={false} connectNulls />
            <Line type="monotone" dataKey={M.p99} name="p99" stroke={PCT_COLORS.p99} strokeWidth={2} dot={false} connectNulls />
          </LineChart>
        </ChartCard>

        <ChartCard title={`${metricLabel} by upstream`}
          subtitle="Avg per bucket, one line per proxy upstream — spot a slow provider."
          empty={!upstreamLatSeries.length} loading={loading}>
          <LineChart data={upstreamLatRows}>
            {commonAxes()}
            <Tooltip contentStyle={{ fontSize: 12 }} labelFormatter={(l) => fullLabel(String(l))}
              formatter={(v, n) => [fmtMs(Number(v)), String(n)]} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {upstreamLatSeries.map((s, i) => (
              <Line key={s.id} type="monotone" dataKey={s.id} name={s.name} stroke={PALETTE[i % PALETTE.length]}
                strokeWidth={2} dot={false} connectNulls />
            ))}
          </LineChart>
        </ChartCard>
      </div>

      {/* Latency by model + requests by status */}
      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard title={`${metricLabel} by model`}
          subtitle="Avg per bucket, one line per model (top 8 + other)."
          empty={!modelLatSeries.length} loading={loading}>
          <LineChart data={modelLatRows}>
            {commonAxes()}
            <Tooltip contentStyle={{ fontSize: 12 }} labelFormatter={(l) => fullLabel(String(l))}
              formatter={(v, n) => [fmtMs(Number(v)), String(n)]} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {modelLatSeries.map((s, i) => (
              <Line key={s.id} type="monotone" dataKey={s.id} name={s.name} stroke={PALETTE[i % PALETTE.length]}
                strokeWidth={2} dot={false} connectNulls />
            ))}
          </LineChart>
        </ChartCard>

        <ChartCard title="Requests by outcome"
          subtitle="OK vs error vs pending per bucket — success/error rate over time."
          empty={!statusBuckets.length} loading={loading}>
          <BarChart data={statusBuckets}>
            {commonAxes(true)}
            <Tooltip contentStyle={{ fontSize: 12 }} labelFormatter={(l) => fullLabel(String(l))}
              formatter={(v, n) => [fmtNum(Number(v)), String(n)]} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {statusSeries.map((s) => (
              <Bar key={s.id} dataKey={s.id} name={STATUS_META[s.name]?.label ?? s.name} stackId="s"
                fill={STATUS_META[s.name]?.color ?? "#94a3b8"} />
            ))}
          </BarChart>
        </ChartCard>
      </div>

      {/* Requests by model + traffic by source */}
      <div className="grid gap-6 lg:grid-cols-2">
        <ChartCard title="Requests by model" subtitle="Requests per bucket, stacked by model."
          empty={!modelBuckets.length} loading={loading}>
          <BarChart data={modelBuckets}>
            {commonAxes(true)}
            <Tooltip contentStyle={{ fontSize: 12 }} labelFormatter={(l) => fullLabel(String(l))} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {modelSeries.map((s, i) => (
              <Bar key={s.id} dataKey={s.id} name={s.name} stackId="m" fill={PALETTE[i % PALETTE.length]} />
            ))}
          </BarChart>
        </ChartCard>

        <ChartCard title="Traffic by source" subtitle="Requests per bucket — serverless queue vs API proxy."
          empty={!sourceBuckets.length} loading={loading}>
          <BarChart data={sourceBuckets}>
            {commonAxes(true)}
            <Tooltip contentStyle={{ fontSize: 12 }} labelFormatter={(l) => fullLabel(String(l))}
              formatter={(v, n) => [fmtNum(Number(v)), String(n)]} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {sourceSeries.map((s) => (
              <Bar key={s.id} dataKey={s.id} name={SOURCE_META[s.name]?.label ?? s.name} stackId="src"
                fill={SOURCE_META[s.name]?.color ?? "#94a3b8"} />
            ))}
          </BarChart>
        </ChartCard>
      </div>

      {/* Token volume (prompt vs completion) */}
      <ChartCard title="Token volume" subtitle="Prompt vs completion tokens per bucket."
        empty={!summary?.by_bucket.length} loading={loading}>
        <BarChart data={summary?.by_bucket ?? []}>
          {commonAxes(true)}
          <Tooltip
            contentStyle={{ fontSize: 12 }}
            labelFormatter={(l) => fullLabel(String(l))}
            formatter={(v, n) => [fmtNum(Number(v)), n === "prompt_tokens" ? "prompt" : "completion"]}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v) => (v === "prompt_tokens" ? "prompt" : "completion")} />
          <Bar dataKey="prompt_tokens" stackId="t" fill="#3b82f6" />
          <Bar dataKey="completion_tokens" stackId="t" fill="#a855f7" />
        </BarChart>
      </ChartCard>

      {/* Token volume by user (stacked) — own full-width row */}
      <ChartCard title="Token volume by user" subtitle="Total tokens per bucket, stacked by user."
        empty={!userBuckets.length} loading={loading}>
        <BarChart data={userBuckets}>
          {commonAxes(true)}
          <Tooltip
            contentStyle={{ fontSize: 12 }}
            labelFormatter={(l) => fullLabel(String(l))}
            formatter={(v, n) => [fmtNum(Number(v)), String(n)]}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {userSeries.map((s, i) => (
            <Bar key={s.id} dataKey={s.id} name={s.name} stackId="u" fill={PALETTE[i % PALETTE.length]} />
          ))}
        </BarChart>
      </ChartCard>

      {/* Requests by upstream (proxy only, stacked) — own full-width row */}
      <ChartCard title="Requests by upstream" subtitle="Proxy requests per bucket, stacked by upstream."
        empty={!upstreamBuckets.length} loading={loading}>
        <BarChart data={upstreamBuckets}>
          {commonAxes(true)}
          <Tooltip contentStyle={{ fontSize: 12 }} labelFormatter={(l) => fullLabel(String(l))}
            formatter={(v, n) => [fmtNum(Number(v)), String(n)]} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {upstreamSeries.map((s, i) => (
            <Bar key={s.id} dataKey={s.id} name={s.name} stackId="up" fill={PALETTE[i % PALETTE.length]} />
          ))}
        </BarChart>
      </ChartCard>

      {/* Top users + top models */}
      <div className="grid gap-6 lg:grid-cols-2">
        <RankCard title="Top users" subtitle="By total tokens in this period."
          rows={(summary?.top_users ?? []).map((u) => ({
            label: u.user, requests: u.requests, tokens: u.prompt_tokens + u.completion_tokens }))} />
        <RankCard title="Top models" subtitle="By request count in this period."
          rows={(summary?.by_model ?? []).map((m) => ({
            label: m.model, requests: m.requests, tokens: m.prompt_tokens + m.completion_tokens }))} />
      </div>

      {summary?.note && <p className="text-xs text-muted-foreground">{summary.note}</p>}
    </div>
  );
}

// Small segmented (pill) toggle — used for the TTFT⇄latency metric switch.
function Segmented<T extends string>({ value, onChange, options }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: string }[];
}) {
  return (
    <div className="inline-flex items-center rounded-md border bg-background p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
            value === o.value ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// Multi-select model picker. Empty selection = all models (no filter). Options
// come from the summary's full model list, so they stay complete while filtered.
function ModelFilter({ all, selected, onChange }: {
  all: string[]; selected: string[]; onChange: (next: string[]) => void;
}) {
  const label = selected.length === 0 ? "All models"
    : selected.length === 1 ? selected[0]
    : `${selected.length} models`;
  const toggle = (m: string) =>
    onChange(selected.includes(m) ? selected.filter((x) => x !== m) : [...selected, m]);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="w-[180px] justify-between gap-2 font-normal">
          <span className="truncate">{label}</span>
          <ChevronDown className="h-3.5 w-3.5 shrink-0 opacity-60" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="max-h-80 w-64 overflow-y-auto">
        <DropdownMenuLabel>Filter by model</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {all.length === 0 ? (
          <div className="px-2 py-1.5 text-xs text-muted-foreground">No models in range.</div>
        ) : (
          <>
            <DropdownMenuItem
              disabled={selected.length === 0}
              onSelect={(e) => { e.preventDefault(); onChange([]); }}
              className="text-xs text-muted-foreground"
            >
              Clear (show all)
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            {all.map((m) => (
              <DropdownMenuCheckboxItem
                key={m}
                checked={selected.includes(m)}
                onSelect={(e) => e.preventDefault()}
                onCheckedChange={() => toggle(m)}
                className="font-mono text-xs"
              >
                <span className="truncate">{m}</span>
              </DropdownMenuCheckboxItem>
            ))}
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function ChartCard({ title, subtitle, empty, loading, children }: {
  title: string; subtitle?: string; empty: boolean; loading: boolean; children: React.ReactElement;
}) {
  return (
    <div data-form-section={title} className="scroll-mt-6 rounded-lg border bg-card p-4">
      <h2 className="mb-1 text-sm font-semibold">{title}</h2>
      {subtitle && <p className="mb-3 text-xs text-muted-foreground">{subtitle}</p>}
      <div className="h-64">
        {empty ? (
          <Empty loading={loading} />
        ) : (
          <ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer>
        )}
      </div>
    </div>
  );
}

function Empty({ loading }: { loading: boolean }) {
  return (
    <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
      {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "No activity in the selected period."}
    </div>
  );
}

function RankCard({ title, subtitle, rows }: {
  title: string; subtitle?: string; rows: { label: string; requests: number; tokens: number }[];
}) {
  return (
    <div data-form-section={title} className="scroll-mt-6 rounded-lg border bg-card p-4">
      <h2 className="mb-1 text-sm font-semibold">{title}</h2>
      {subtitle && <p className="mb-3 text-xs text-muted-foreground">{subtitle}</p>}
      {rows.length === 0 ? (
        <p className="py-6 text-center text-sm text-muted-foreground">No data.</p>
      ) : (
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
            <span>name</span><span>requests · tokens</span>
          </div>
          {rows.map((r, i) => (
            <div key={r.label + i} className="flex items-center justify-between gap-2 text-sm">
              <span className="flex min-w-0 items-center gap-2">
                <span className="w-4 shrink-0 text-right text-[11px] text-muted-foreground">{i + 1}</span>
                <span className="truncate font-mono text-[12px]">{r.label}</span>
              </span>
              <span className="shrink-0 text-[12px] text-muted-foreground tabular-nums">
                {fmtNum(r.requests)} · {fmtNum(r.tokens)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
