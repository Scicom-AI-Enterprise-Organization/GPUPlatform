"use client";

// OpenRouter-style usage dashboard over all serverless + LLM-proxy requests:
// stat cards (requests, tokens in/out, avg TTFT, avg latency), a TTFT/latency
// time-series, requests-by-model + token-volume bars, and top users/models.
// Self-hosted → no $ spend. Granularity is selectable (minute / hour / day),
// defaulting to hourly. Styled to match /admin/analytics.
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Loader2 } from "lucide-react";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { gateway } from "@/lib/gateway";
import type { ActivitySummary, ActivityGranularity } from "@/lib/types";

const RANGES: Record<string, { label: string; days: number }> = {
  "24h": { label: "Past 24 hours", days: 1 },
  "7d": { label: "Past week", days: 7 },
  "30d": { label: "Past month", days: 30 },
};
const GRANS: { value: ActivityGranularity; label: string }[] = [
  { value: "minute", label: "Per minute" },
  { value: "hour", label: "Hourly" },
  { value: "day", label: "Daily" },
];
const PALETTE = ["#3b82f6", "#f59e0b", "#10b981", "#a855f7", "#ef4444", "#06b6d4", "#ec4899", "#84cc16", "#6366f1", "#94a3b8"];

const fmtNum = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n);
const fmtMs = (ms: number | null | undefined) =>
  ms == null ? "—" : ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;

// Bucket keys are ISO-ish: "YYYY-MM-DD" (day), "...THH:00" (hour), "...THH:MM" (minute).
const tickFmt = (b: string, g: ActivityGranularity) =>
  !b ? b
    : g === "day" ? b.slice(5)
    : g === "hour" ? `${b.slice(5, 13).replace("T", " ")}h`
    : b.slice(5, 16).replace("T", " ");
const fullLabel = (b: string) => String(b).replace("T", " ");

export function ActivityDashboard() {
  const [range, setRange] = useState<keyof typeof RANGES>("7d");
  const [gran, setGran] = useState<ActivityGranularity>("hour");
  const [summary, setSummary] = useState<ActivitySummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true); setErr(null);
    try {
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
      const since = new Date(Date.now() - RANGES[range].days * 86400_000).toISOString();
      setSummary(await gateway.getActivity({ since, tz, granularity: gran, top: 8 }));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [range, gran]);
  useEffect(() => {
    const timer = window.setTimeout(() => { void load(); }, 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  // Pivot by_model_bucket → one row per bucket with a column per model (stacked bar).
  const { modelBuckets, modelKeys } = useMemo(() => {
    const rows = summary?.by_model_bucket ?? [];
    const keys = Array.from(new Set(rows.map((d) => d.model)));
    const byBucket: Record<string, Record<string, number>> = {};
    for (const r of rows) {
      (byBucket[r.bucket] ??= { bucket: r.bucket } as Record<string, number> & { bucket?: string })[r.model] = r.requests;
    }
    const out = Object.values(byBucket).sort((a, b) => String(a.bucket).localeCompare(String(b.bucket)));
    return { modelBuckets: out, modelKeys: keys };
  }, [summary]);

  const t = summary?.totals;
  const cards = [
    { label: "Requests", value: t ? fmtNum(t.requests) : "—", sub: "Serverless + proxy" },
    { label: "Tokens in", value: t ? fmtNum(t.prompt_tokens) : "—", sub: "Prompt tokens" },
    { label: "Tokens out", value: t ? fmtNum(t.completion_tokens) : "—", sub: "Completion tokens" },
    { label: "Avg TTFT", value: fmtMs(t?.avg_ttft_ms), sub: "Time to first token" },
    { label: "Avg latency", value: fmtMs(t?.avg_latency_ms), sub: "End-to-end per request" },
  ];

  return (
    <div className="space-y-6">
      {/* Header + controls */}
      <div className="flex flex-wrap items-center gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Activity</h1>
          <p className="mt-0.5 text-sm text-muted-foreground">
            Usage across serverless endpoints + LLM proxies — who, which endpoint, model, tokens, TTFT, latency.
          </p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {loading && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
          <Select value={gran} onValueChange={(v) => setGran(v as ActivityGranularity)}>
            <SelectTrigger className="w-[130px]" size="sm"><SelectValue /></SelectTrigger>
            <SelectContent>
              {GRANS.map((g) => <SelectItem key={g.value} value={g.value}>{g.label}</SelectItem>)}
            </SelectContent>
          </Select>
          <Select value={range} onValueChange={(v) => setRange(v as keyof typeof RANGES)}>
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
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border bg-border lg:grid-cols-5">
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

      <div className="grid gap-6 lg:grid-cols-2">
        {/* TTFT + latency over time */}
        <ChartCard title="Latency over time" subtitle="Average TTFT and end-to-end latency per bucket.">
          {!summary?.by_bucket.length ? (
            <Empty loading={loading} />
          ) : (
            <LineChart data={summary.by_bucket}>
              <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
              <XAxis dataKey="bucket" tick={{ fontSize: 11 }} tickFormatter={(b) => tickFmt(b, gran)} minTickGap={28} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => fmtMs(Number(v))} width={56} />
              <Tooltip
                contentStyle={{ fontSize: 12 }}
                labelFormatter={(l) => fullLabel(String(l))}
                formatter={(v, n) => [fmtMs(Number(v)), n === "avg_ttft_ms" ? "TTFT" : "Latency"]}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v) => (v === "avg_ttft_ms" ? "TTFT" : "Latency")} />
              <Line type="monotone" dataKey="avg_ttft_ms" stroke="#3b82f6" strokeWidth={2} dot={false} connectNulls />
              <Line type="monotone" dataKey="avg_latency_ms" stroke="#f59e0b" strokeWidth={2} dot={false} connectNulls />
            </LineChart>
          )}
        </ChartCard>

        {/* Requests by model (stacked) */}
        <ChartCard title="Requests by model" subtitle="Requests per bucket, stacked by model.">
          {!modelBuckets.length ? (
            <Empty loading={loading} />
          ) : (
            <BarChart data={modelBuckets}>
              <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
              <XAxis dataKey="bucket" tick={{ fontSize: 11 }} tickFormatter={(b) => tickFmt(b, gran)} minTickGap={28} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={fmtNum} allowDecimals={false} width={44} />
              <Tooltip contentStyle={{ fontSize: 12 }} labelFormatter={(l) => fullLabel(String(l))} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {modelKeys.map((m, i) => (
                <Bar key={m} dataKey={m} stackId="m" fill={PALETTE[i % PALETTE.length]} />
              ))}
            </BarChart>
          )}
        </ChartCard>
      </div>

      {/* Token volume (prompt vs completion) */}
      <ChartCard title="Token volume" subtitle="Prompt vs completion tokens per bucket.">
        {!summary?.by_bucket.length ? (
          <Empty loading={loading} />
        ) : (
          <BarChart data={summary.by_bucket}>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
            <XAxis dataKey="bucket" tick={{ fontSize: 11 }} tickFormatter={(b) => tickFmt(b, gran)} minTickGap={28} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={fmtNum} width={44} />
            <Tooltip
              contentStyle={{ fontSize: 12 }}
              labelFormatter={(l) => fullLabel(String(l))}
              formatter={(v, n) => [fmtNum(Number(v)), n === "prompt_tokens" ? "prompt" : "completion"]}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v) => (v === "prompt_tokens" ? "prompt" : "completion")} />
            <Bar dataKey="prompt_tokens" stackId="t" fill="#3b82f6" />
            <Bar dataKey="completion_tokens" stackId="t" fill="#a855f7" />
          </BarChart>
        )}
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

function ChartCard({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactElement }) {
  return (
    <div className="rounded-lg border bg-card p-4">
      <h2 className="mb-1 text-sm font-semibold">{title}</h2>
      {subtitle && <p className="mb-3 text-xs text-muted-foreground">{subtitle}</p>}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer>
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
    <div className="rounded-lg border bg-card p-4">
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
