"use client";

// OpenRouter-style usage dashboard: cards (requests, token volume in/out, avg TTFT,
// avg latency), usage-by-model + token-volume-by-day charts, top users, and a unified
// per-request logs table — across serverless + LLM-proxy. Self-hosted → no $ spend.
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { ActivitySummary, ActivityLogRow } from "@/lib/types";

const RANGES: Record<string, { label: string; days: number }> = {
  "24h": { label: "Past 24 hours", days: 1 },
  "7d": { label: "Past week", days: 7 },
  "30d": { label: "Past month", days: 30 },
};
const PALETTE = ["#3b82f6", "#f59e0b", "#10b981", "#a855f7", "#ef4444", "#06b6d4", "#ec4899", "#84cc16", "#6366f1", "#94a3b8"];

const fmtNum = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n);
const fmtMs = (ms: number | null | undefined) =>
  ms == null ? "—" : ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;

export function ActivityDashboard() {
  const [range, setRange] = useState<keyof typeof RANGES>("7d");
  const [tab, setTab] = useState<"overview" | "logs">("overview");
  const [summary, setSummary] = useState<ActivitySummary | null>(null);
  const [logs, setLogs] = useState<ActivityLogRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const tz = useMemo(() => {
    try { return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"; } catch { return "UTC"; }
  }, []);
  const since = useMemo(
    () => new Date(Date.now() - RANGES[range].days * 86400_000).toISOString(),
    [range],
  );

  const load = useCallback(async () => {
    setLoading(true); setErr(null);
    try {
      const [s, l] = await Promise.all([
        gateway.getActivity({ since, tz, top: 8 }),
        gateway.getActivityLogs({ since, limit: 200 }),
      ]);
      setSummary(s); setLogs(l.jobs);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [since, tz]);
  useEffect(() => { load(); }, [load]);

  // Pivot by_model_day → one row per day with a column per model (for the stacked bar).
  const { modelDays, modelKeys } = useMemo(() => {
    const keys = Array.from(new Set((summary?.by_model_day ?? []).map((d) => d.model)));
    const byDay: Record<string, Record<string, number>> = {};
    for (const r of summary?.by_model_day ?? []) {
      (byDay[r.day] ??= { day: r.day } as Record<string, number> & { day?: string })[r.model] = r.requests;
    }
    const rows = Object.values(byDay).sort((a, b) => String(a.day).localeCompare(String(b.day)));
    return { modelDays: rows, modelKeys: keys };
  }, [summary]);

  const t = summary?.totals;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Activity</h1>
          <p className="mt-0.5 text-sm text-muted-foreground">Usage across serverless endpoints + LLM proxies — who, which endpoint, model, tokens, TTFT, latency.</p>
        </div>
        <div className="flex items-center gap-2">
          {loading && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
          <Select value={range} onValueChange={(v) => setRange(v as keyof typeof RANGES)}>
            <SelectTrigger className="h-8 w-40 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              {Object.entries(RANGES).map(([k, v]) => <SelectItem key={k} value={k} className="text-xs">{v.label}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
      </div>

      {err && <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">{err}</div>}

      <div className="inline-flex rounded-md border border-border p-0.5 text-sm">
        {(["overview", "logs"] as const).map((v) => (
          <button key={v} type="button" onClick={() => setTab(v)}
            className={cn("rounded px-3 py-1 capitalize transition-colors",
              tab === v ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground")}>
            {v}
          </button>
        ))}
      </div>

      {tab === "overview" ? (
        <>
          {/* stat cards */}
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <StatCard label="Requests" value={t ? fmtNum(t.requests) : "—"} />
            <StatCard label="Token volume" value={t ? fmtNum(t.total_tokens) : "—"}
              sub={t ? `${fmtNum(t.prompt_tokens)} in · ${fmtNum(t.completion_tokens)} out` : undefined} />
            <StatCard label="Avg TTFT" value={fmtMs(t?.avg_ttft_ms)} />
            <StatCard label="Avg latency" value={fmtMs(t?.avg_latency_ms)} />
          </div>

          {/* usage by model (stacked, per day) */}
          <ChartCard title="Usage by model" subtitle="requests/day, stacked by model">
            <BarChart data={modelDays}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border/50" />
              <XAxis dataKey="day" tick={{ fontSize: 11 }} tickFormatter={dayTick} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={fmtNum} width={44} />
              <Tooltip contentStyle={TOOLTIP} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {modelKeys.map((m, i) => (
                <Bar key={m} dataKey={m} stackId="m" fill={PALETTE[i % PALETTE.length]} radius={i === modelKeys.length - 1 ? [3, 3, 0, 0] : 0} />
              ))}
            </BarChart>
          </ChartCard>

          {/* token volume by day (prompt vs completion) */}
          <ChartCard title="Token volume" subtitle="prompt vs completion tokens/day">
            <BarChart data={summary?.by_day ?? []}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} className="stroke-border/50" />
              <XAxis dataKey="day" tick={{ fontSize: 11 }} tickFormatter={dayTick} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={fmtNum} width={44} />
              <Tooltip contentStyle={TOOLTIP} formatter={(v) => fmtNum(Number(v))} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="prompt_tokens" name="prompt" stackId="t" fill="#3b82f6" />
              <Bar dataKey="completion_tokens" name="completion" stackId="t" fill="#a855f7" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ChartCard>

          {/* top users + top models */}
          <div className="grid gap-3 lg:grid-cols-2">
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Top users</CardTitle></CardHeader>
              <CardContent>
                <RankTable rows={(summary?.top_users ?? []).map((u) => ({
                  label: u.user, requests: u.requests, tokens: u.prompt_tokens + u.completion_tokens }))} />
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Top models</CardTitle></CardHeader>
              <CardContent>
                <RankTable rows={(summary?.by_model ?? []).map((m) => ({
                  label: m.model, requests: m.requests, tokens: m.prompt_tokens + m.completion_tokens }))} />
              </CardContent>
            </Card>
          </div>
          {summary?.note && <p className="text-[11px] text-muted-foreground">{summary.note}</p>}
        </>
      ) : (
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm">Request log <span className="font-normal text-muted-foreground">· {logs.length} most recent</span></CardTitle></CardHeader>
          <CardContent className="px-0">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="border-b border-border text-left text-muted-foreground">
                  <tr>
                    {["Time", "User", "Source", "Endpoint", "Model", "Status", "In", "Out", "TTFT", "Latency"].map((h) => (
                      <th key={h} className="px-3 py-2 font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {logs.length === 0 ? (
                    <tr><td colSpan={10} className="px-3 py-6 text-center text-muted-foreground">No requests in this window.</td></tr>
                  ) : logs.map((r) => (
                    <tr key={`${r.kind}-${r.id}`} className="border-b border-border/50 hover:bg-muted/30">
                      <td className="whitespace-nowrap px-3 py-1.5 text-muted-foreground">{r.created_at ? new Date(r.created_at).toLocaleString() : "—"}</td>
                      <td className="px-3 py-1.5">{r.user}</td>
                      <td className="px-3 py-1.5"><span className="rounded bg-muted px-1 text-[10px] uppercase text-muted-foreground">{r.kind}</span></td>
                      <td className="max-w-[160px] truncate px-3 py-1.5 font-mono text-[11px]">{r.detail.endpoint ?? "—"}</td>
                      <td className="max-w-[200px] truncate px-3 py-1.5 font-mono text-[11px]">{r.name ?? "—"}</td>
                      <td className="px-3 py-1.5">{r.status}</td>
                      <td className="px-3 py-1.5 text-right tabular-nums">{r.detail.prompt_tokens ?? "—"}</td>
                      <td className="px-3 py-1.5 text-right tabular-nums">{r.detail.completion_tokens ?? "—"}</td>
                      <td className="px-3 py-1.5 text-right tabular-nums">{fmtMs(r.detail.ttft_ms)}</td>
                      <td className="px-3 py-1.5 text-right tabular-nums">{fmtMs(r.detail.latency_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

const TOOLTIP = { background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 } as const;
const dayTick = (d: string) => (d && d.length >= 10 ? d.slice(5) : d); // MM-DD

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <Card>
      <CardContent className="py-4">
        <div className="text-xs text-muted-foreground">{label}</div>
        <div className="mt-1 text-2xl font-semibold tracking-tight">{value}</div>
        {sub && <div className="mt-0.5 text-[11px] text-muted-foreground">{sub}</div>}
      </CardContent>
    </Card>
  );
}

function ChartCard({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactElement }) {
  return (
    <Card>
      <CardHeader className="pb-1">
        <CardTitle className="text-sm">{title}</CardTitle>
        {subtitle && <p className="text-[11px] text-muted-foreground">{subtitle}</p>}
      </CardHeader>
      <CardContent>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

function RankTable({ rows }: { rows: { label: string; requests: number; tokens: number }[] }) {
  if (rows.length === 0) return <p className="py-3 text-center text-xs text-muted-foreground">No data.</p>;
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
        <span>name</span><span>requests · tokens</span>
      </div>
      {rows.map((r, i) => (
        <div key={r.label + i} className="flex items-center justify-between gap-2 text-sm">
          <span className="flex min-w-0 items-center gap-2">
            <span className="w-4 shrink-0 text-right text-[11px] text-muted-foreground">{i + 1}</span>
            <span className="truncate font-mono text-[12px]">{r.label}</span>
          </span>
          <span className="shrink-0 text-[12px] text-muted-foreground tabular-nums">{fmtNum(r.requests)} · {fmtNum(r.tokens)}</span>
        </div>
      ))}
    </div>
  );
}
