"use client";

// Admin Usage Report — what models are being used, how many requests, the
// 4xx/5xx (approx) breakdown, and request volume over time per model. Built on
// the durable `requests` table via GET /v1/usage/report (+ /spend). Export to
// PDF (window.print over the print-only mirror) and DOCX (docx, lazy-loaded).

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer, BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, Tooltip, Legend, CartesianGrid,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu, DropdownMenuCheckboxItem, DropdownMenuContent,
  DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDown, ChevronRight, FileText, Printer, X } from "lucide-react";
import { gateway } from "@/lib/gateway";
import type { UsageReport, UsageSpend, UsageTimePoint } from "@/lib/types";

// ── Date helpers (browser-local) ────────────────────────────────────────────
function localIso(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}
const today = () => localIso(new Date());
function daysAgo(n: number) { const d = new Date(); d.setDate(d.getDate() - n); return localIso(d); }
function monthStart(off = 0) { const d = new Date(); d.setDate(1); d.setMonth(d.getMonth() + off); return localIso(d); }
function monthEnd(off = 0) { const d = new Date(); d.setDate(1); d.setMonth(d.getMonth() + off + 1); d.setDate(0); return localIso(d); }
const capToday = (s: string) => (s > today() ? today() : s);

const DATE_PRESETS = [
  { label: "Today", from: () => today(), to: () => today() },
  { label: "7d", from: () => daysAgo(6), to: () => today() },
  { label: "30d", from: () => daysAgo(29), to: () => today() },
  { label: "90d", from: () => daysAgo(89), to: () => today() },
  { label: "This month", from: () => monthStart(0), to: () => today() },
  { label: "Last month", from: () => monthStart(-1), to: () => monthEnd(-1) },
];

const STATUS_OPTIONS = ["completed", "failed", "error", "timeout", "cancelled", "pending"];
const MODEL_COLORS = ["#60a5fa", "#f97316", "#22c55e", "#a855f7", "#ec4899", "#06b6d4", "#eab308", "#ef4444", "#14b8a6", "#8b5cf6"];
const OTHER_COLOR = "#9ca3af";

const fmtInt = (n: number) => n.toLocaleString();
const fmtTokens = (n: number) => (n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));
const fmtLat = (s: number | null) => (s == null ? "—" : s < 1 ? `${Math.round(s * 1000)}ms` : s < 60 ? `${s.toFixed(1)}s` : `${(s / 60).toFixed(1)}m`);

// ── MultiSelect (ported pattern) ────────────────────────────────────────────
function MultiSelect({ label, options, selected, onChange }: {
  label: string; options: { value: string; label: string }[]; selected: string[]; onChange: (v: string[]) => void;
}) {
  const display = selected.length === 0 ? "All" : selected.length === 1 ? selected[0] : `${selected.length} selected`;
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="h-8 gap-1 font-normal max-w-56">
          <span className="text-muted-foreground text-xs mr-0.5">{label}:</span>
          <span className="truncate">{display}</span>
          <ChevronDown className="ml-auto h-3 w-3 opacity-50 shrink-0" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="max-h-72 overflow-y-auto min-w-48">
        {options.length === 0 && <div className="px-2 py-1.5 text-xs text-muted-foreground">No options</div>}
        {options.map((opt) => (
          <DropdownMenuCheckboxItem
            key={opt.value}
            checked={selected.includes(opt.value)}
            onCheckedChange={(c) => onChange(c ? [...selected, opt.value] : selected.filter((v) => v !== opt.value))}
          >
            {opt.label}
          </DropdownMenuCheckboxItem>
        ))}
        {selected.length > 0 && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem className="text-xs text-muted-foreground" onClick={() => onChange([])}>Clear selection</DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function StatCard({ label, value, sub, tone }: { label: string; value: string | number; sub?: string; tone?: "good" | "bad" | "warn" }) {
  const color = tone === "good" ? "text-emerald-500" : tone === "bad" ? "text-red-500" : tone === "warn" ? "text-amber-500" : "";
  return (
    <Card className="py-3">
      <CardContent className="px-4">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className={`mt-1 text-2xl font-semibold tabular-nums ${color}`}>{value}</p>
        {sub && <p className="mt-0.5 text-xs text-muted-foreground">{sub}</p>}
      </CardContent>
    </Card>
  );
}

// Build top-N model series + an "other" bucket from the time-series.
function buildModelSeries(ts: UsageTimePoint[], topN = 8) {
  const totals: Record<string, number> = {};
  ts.forEach((p) => Object.entries(p.by_model).forEach(([m, c]) => { totals[m] = (totals[m] || 0) + c; }));
  const top = Object.entries(totals).sort((a, b) => b[1] - a[1]).slice(0, topN).map(([m]) => m);
  const topSet = new Set(top);
  let hasOther = false;
  const rows = ts.map((p) => {
    const row: Record<string, string | number> = { label: p.label };
    let other = 0;
    Object.entries(p.by_model).forEach(([m, c]) => { if (topSet.has(m)) row[m] = c; else other += c; });
    if (other > 0) { row.__other = other; hasOther = true; }
    return row;
  });
  const series = hasOther ? [...top, "__other"] : top;
  return { rows, series };
}

// Print-only table styles (inline so the print mirror is self-contained).
const TH: React.CSSProperties = { border: "1px solid #bbb", padding: "5px 9px", textAlign: "left", fontWeight: "bold", background: "#f0f0f0", fontSize: "12px" };
const TD: React.CSSProperties = { border: "1px solid #bbb", padding: "5px 9px", fontSize: "12px", verticalAlign: "top" };
const TABLE: React.CSSProperties = { width: "100%", borderCollapse: "collapse", marginBottom: "16px" };

export function UsageReportView() {
  const tz = useMemo(() => Intl.DateTimeFormat().resolvedOptions().timeZone, []);

  const [fromDate, setFromDate] = useState(daysAgo(29));
  const [toDate, setToDate] = useState(today());
  const [appId, setAppId] = useState("__all__");
  const [ownerId, setOwnerId] = useState("__all__");
  const [models, setModels] = useState<string[]>([]);
  const [statuses, setStatuses] = useState<string[]>([]);
  const [bucket, setBucket] = useState<"auto" | "hour" | "day">("auto");

  const [data, setData] = useState<UsageReport | null>(null);
  const [spend, setSpend] = useState<UsageSpend | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [docxExporting, setDocxExporting] = useState(false);

  const fetchReport = useCallback(async () => {
    setLoading(true); setError(null);
    const params = {
      from: fromDate, to: toDate, tz, bucket,
      ...(appId !== "__all__" ? { app_id: appId } : {}),
      ...(ownerId !== "__all__" ? { owner_id: Number(ownerId) } : {}),
      ...(models.length ? { model: models.join(",") } : {}),
      ...(statuses.length ? { status: statuses.join(",") } : {}),
    };
    try {
      const [rep, spd] = await Promise.allSettled([
        gateway.getUsageReport(params),
        gateway.getUsageSpend({ from: fromDate, to: toDate, tz, ...(ownerId !== "__all__" ? { owner_id: Number(ownerId) } : {}) }),
      ]);
      if (rep.status === "fulfilled") setData(rep.value);
      else throw rep.reason;
      setSpend(spd.status === "fulfilled" ? spd.value : null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load usage report");
    } finally {
      setLoading(false);
    }
  }, [fromDate, toDate, tz, bucket, appId, ownerId, models, statuses]);

  useEffect(() => { fetchReport(); }, [fetchReport]);

  const toggleDay = (d: string) => setExpanded((p) => { const n = new Set(p); n.has(d) ? n.delete(d) : n.add(d); return n; });
  function clearFilters() {
    setAppId("__all__"); setOwnerId("__all__"); setModels([]); setStatuses([]);
    setFromDate(daysAgo(29)); setToDate(today()); setBucket("auto");
  }
  const activeFilters = [appId !== "__all__", ownerId !== "__all__", models.length > 0, statuses.length > 0].filter(Boolean).length;

  const { rows: modelRows, series: modelSeries } = useMemo(
    () => (data ? buildModelSeries(data.time_series) : { rows: [], series: [] }),
    [data],
  );
  const outcomeRows = useMemo(
    () => (data?.time_series ?? []).map((p) => ({ label: p.label, success: p.success, "~4xx": p.client_cancelled, "~5xx": p.server_error })),
    [data],
  );
  const seriesLabel = (s: string) => (s === "__other" ? "other" : s);
  const periodLabel = data ? `${data.period.from_date} → ${data.period.to_date}` : "";
  const scopeLabel = data?.scope === "owner" ? "Your usage" : "Platform-wide";

  async function onExportDocx() {
    if (!data) return;
    setDocxExporting(true);
    try {
      const { exportUsageDocx } = await import("./export-docx");
      await exportUsageDocx(data, spend, { periodLabel, scopeLabel, tz });
    } catch (e) {
      setError(e instanceof Error ? `DOCX export failed: ${e.message}` : "DOCX export failed");
    } finally {
      setDocxExporting(false);
    }
  }

  return (
    <div className="space-y-4">
      {/* ════════ SCREEN UI ════════════════════════════════════════════════ */}
      <div className="print:hidden space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Usage Reports</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Models, requests and error breakdown over time. {scopeLabel}.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" onClick={onExportDocx} disabled={loading || !data || docxExporting}>
              <FileText className="h-4 w-4 mr-2" />
              {docxExporting ? "Generating…" : "Export DOCX"}
            </Button>
            <Button size="sm" variant="outline" onClick={() => window.print()} disabled={loading || !data}>
              <Printer className="h-4 w-4 mr-2" />
              Export PDF
            </Button>
          </div>
        </div>

        {/* Filter bar */}
        <Card className="px-4 py-3">
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-muted-foreground font-medium">From</span>
              <input type="date" value={fromDate} max={toDate} onChange={(e) => setFromDate(e.target.value)}
                className="h-8 rounded-md border border-input bg-background px-2 text-sm" />
              <span className="text-xs text-muted-foreground font-medium">To</span>
              <input type="date" value={toDate} min={fromDate} max={today()} onChange={(e) => setToDate(capToday(e.target.value))}
                className="h-8 rounded-md border border-input bg-background px-2 text-sm" />
              <div className="flex flex-wrap gap-1 ml-1">
                {DATE_PRESETS.map((p) => (
                  <button key={p.label} onClick={() => { setFromDate(p.from()); setToDate(capToday(p.to())); }}
                    className="h-7 rounded px-2 text-xs border border-border hover:bg-accent hover:text-accent-foreground transition-colors">
                    {p.label}
                  </button>
                ))}
              </div>
              <div className="ml-auto flex items-center gap-1 rounded-md border border-border p-0.5">
                {(["auto", "hour", "day"] as const).map((b) => (
                  <button key={b} onClick={() => setBucket(b)}
                    className={`h-6 rounded px-2 text-xs capitalize transition-colors ${bucket === b ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`}>
                    {b}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              {data && data.apps.length > 0 && (
                <Select value={appId} onValueChange={setAppId}>
                  <SelectTrigger className="h-8 w-auto min-w-40 text-sm font-normal">
                    <span className="text-muted-foreground text-xs mr-1">Endpoint:</span>
                    <SelectValue placeholder="All" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__all__">All endpoints</SelectItem>
                    {data.apps.map((a) => <SelectItem key={a.app_id} value={a.app_id}>{a.name}</SelectItem>)}
                  </SelectContent>
                </Select>
              )}
              <MultiSelect label="Model" options={(data?.models ?? []).map((m) => ({ value: m, label: m }))} selected={models} onChange={setModels} />
              <MultiSelect label="Status" options={STATUS_OPTIONS.map((s) => ({ value: s, label: s }))} selected={statuses} onChange={setStatuses} />
              {data && data.users.length > 1 && (
                <Select value={ownerId} onValueChange={setOwnerId}>
                  <SelectTrigger className="h-8 w-auto min-w-36 text-sm font-normal">
                    <span className="text-muted-foreground text-xs mr-1">User:</span>
                    <SelectValue placeholder="All" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__all__">All users</SelectItem>
                    {data.users.map((u) => <SelectItem key={u.owner_id} value={String(u.owner_id)}>{u.username}</SelectItem>)}
                  </SelectContent>
                </Select>
              )}
              {activeFilters > 0 && (
                <Button variant="ghost" size="sm" className="h-8 text-xs text-muted-foreground gap-1" onClick={clearFilters}>
                  <X className="h-3 w-3" /> Clear <Badge variant="secondary" className="ml-1 h-4 px-1 text-[10px]">{activeFilters}</Badge>
                </Button>
              )}
            </div>
          </div>
        </Card>

        {error && <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">{error}</div>}
        {loading && !data && <div className="flex items-center justify-center py-20 text-muted-foreground text-sm">Loading usage report…</div>}

        {data && (
          <>
            {/* Stat cards */}
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
              <StatCard label="Total requests" value={fmtInt(data.summary.total_requests)} sub={`${data.summary.distinct_models} models · ${data.summary.distinct_apps} endpoints`} />
              <StatCard label="Success rate" tone={data.summary.success_rate == null ? undefined : data.summary.success_rate >= 95 ? "good" : data.summary.success_rate >= 80 ? "warn" : "bad"}
                value={data.summary.success_rate == null ? "—" : `${data.summary.success_rate}%`} sub={`${fmtInt(data.summary.completed)} completed`} />
              <StatCard label="~4xx (client)" value={fmtInt(data.summary.client_cancelled)} tone={data.summary.client_cancelled > 0 ? "warn" : undefined} sub="cancelled (approx)" />
              <StatCard label="~5xx (server)" value={fmtInt(data.summary.server_error)} tone={data.summary.server_error > 0 ? "bad" : undefined} sub="failed/error/timeout" />
              <StatCard label="Tokens" value={fmtTokens(data.summary.tokens_total)} sub={data.summary.token_coverage_pct == null ? "no token data" : `${data.summary.token_coverage_pct}% coverage`} />
              <StatCard label="Latency p95" value={fmtLat(data.summary.p95_latency_s)} sub={`avg ${fmtLat(data.summary.avg_latency_s)} · end-to-end`} />
            </div>

            {/* Requests over time, by model */}
            <Card>
              <CardHeader className="pb-2 flex-row items-center justify-between">
                <CardTitle className="text-sm font-medium">Requests over time, by model</CardTitle>
                <span className="text-xs text-muted-foreground">{data.bucket === "hour" ? "hourly" : "daily"} buckets</span>
              </CardHeader>
              <CardContent>
                {modelRows.length === 0 ? <p className="py-10 text-center text-sm text-muted-foreground">No requests in this period.</p> : (
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={modelRows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                        <XAxis dataKey="label" tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} interval="preserveStartEnd" minTickGap={20} />
                        <YAxis tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={32} allowDecimals={false} />
                        <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12 }} />
                        <Legend wrapperStyle={{ fontSize: 11 }} formatter={(v) => seriesLabel(String(v))} />
                        {modelSeries.map((s, i) => (
                          <Bar key={s} dataKey={s} name={seriesLabel(s)} stackId="m" fill={s === "__other" ? OTHER_COLOR : MODEL_COLORS[i % MODEL_COLORS.length]} />
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </CardContent>
            </Card>

            <div className="grid gap-4 lg:grid-cols-2">
              {/* By model table */}
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm font-medium">By model</CardTitle></CardHeader>
                <CardContent>
                  <ScreenTable
                    head={["Model", "Requests", "OK", "~4xx", "~5xx", "Tokens", "Avg lat"]}
                    rows={data.by_model.map((m) => [m.model, fmtInt(m.requests), fmtInt(m.completed), fmtInt(m.client_cancelled), fmtInt(m.server_error), fmtTokens(m.tokens_total), fmtLat(m.avg_latency_s)])}
                    mono={[0]} right={[1, 2, 3, 4, 5, 6]} empty="No model traffic." />
                </CardContent>
              </Card>
              {/* Outcome over time */}
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm font-medium">Outcome over time</CardTitle></CardHeader>
                <CardContent>
                  {outcomeRows.length === 0 ? <p className="py-10 text-center text-sm text-muted-foreground">No data.</p> : (
                    <div className="h-64 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={outcomeRows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                          <XAxis dataKey="label" tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} interval="preserveStartEnd" minTickGap={20} />
                          <YAxis tick={{ fontSize: 10, fill: "var(--muted-foreground)" }} axisLine={false} tickLine={false} width={32} allowDecimals={false} />
                          <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12 }} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Area type="monotone" dataKey="success" stackId="o" stroke="#22c55e" fill="#22c55e" fillOpacity={0.25} />
                          <Area type="monotone" dataKey="~4xx" stackId="o" stroke="#eab308" fill="#eab308" fillOpacity={0.25} />
                          <Area type="monotone" dataKey="~5xx" stackId="o" stroke="#ef4444" fill="#ef4444" fillOpacity={0.25} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm font-medium">By endpoint</CardTitle></CardHeader>
                <CardContent>
                  <ScreenTable head={["Endpoint", "Requests", "OK", "~5xx"]}
                    rows={data.by_endpoint.map((e) => [e.endpoint, fmtInt(e.requests), fmtInt(e.completed), fmtInt(e.server_error)])}
                    mono={[0]} right={[1, 2, 3]} empty="No endpoint traffic." />
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm font-medium">Top users</CardTitle></CardHeader>
                <CardContent>
                  <ScreenTable head={["User", "Requests", "Tokens"]}
                    rows={data.by_user.map((u) => [u.username, fmtInt(u.requests), fmtTokens(u.tokens_total)])}
                    mono={[0]} right={[1, 2]} empty="No users." />
                </CardContent>
              </Card>
            </div>

            {/* Resource spend */}
            <Card>
              <CardHeader className="pb-2 flex-row items-center justify-between">
                <CardTitle className="text-sm font-medium">Resource spend (benchmark / training / compute)</CardTitle>
                {spend && <span className="text-sm font-semibold tabular-nums">${spend.total_cost_usd.toFixed(2)}</span>}
              </CardHeader>
              <CardContent>
                {!spend ? <p className="py-4 text-sm text-muted-foreground">Spend data unavailable.</p> : (
                  <>
                    <ScreenTable head={["Type", "Count", "GPU-hours", "Cost (USD)"]}
                      rows={spend.by_type.map((r) => [r.resource_type, fmtInt(r.count), r.gpu_hours == null ? "—" : r.gpu_hours.toFixed(1), `$${r.cost_usd.toFixed(2)}`])}
                      mono={[0]} right={[1, 2, 3]} empty="No resources in period." />
                    {!spend.has_cost_data && <p className="mt-1 text-xs text-muted-foreground">No priced runs in this period (FakeProvider runs are $0).</p>}
                  </>
                )}
              </CardContent>
            </Card>

            {/* Daily breakdown */}
            <div className="space-y-2">
              <h2 className="text-lg font-semibold">Daily breakdown</h2>
              {data.daily.length === 0 && <p className="text-sm text-muted-foreground">No requests in this period.</p>}
              {data.daily.map((day) => {
                const open = expanded.has(day.date);
                return (
                  <Card key={day.date}>
                    <button className="flex w-full items-center gap-2 px-4 py-3 text-left hover:bg-accent/30 transition-colors" onClick={() => toggleDay(day.date)}>
                      {open ? <ChevronDown className="h-4 w-4 shrink-0" /> : <ChevronRight className="h-4 w-4 shrink-0" />}
                      <span className="font-medium text-sm">{day.day_label}</span>
                      <span className="ml-auto text-xs text-muted-foreground tabular-nums">
                        <span className="mr-3">{fmtInt(day.requests)} reqs</span>
                        {day.server_error > 0 && <span className="text-red-500 mr-3">{day.server_error} ~5xx</span>}
                        {day.client_cancelled > 0 && <span className="text-amber-500 mr-3">{day.client_cancelled} ~4xx</span>}
                        <span>{fmtTokens(day.tokens_total)} tok</span>
                      </span>
                    </button>
                    {open && (
                      <CardContent className="border-t pt-3">
                        <ScreenTable head={["Time", "Model", "Endpoint", "User", "Outcome", "Status", "Elapsed"]}
                          rows={day.jobs.map((j) => [j.start_time, j.model, j.endpoint, j.username, j.outcome, j.status, j.elapsed_label])}
                          mono={[0, 1, 3, 6]} right={[]} empty="No requests this day." />
                      </CardContent>
                    )}
                  </Card>
                );
              })}
            </div>

            <p className="text-xs text-muted-foreground">{data.note}</p>
          </>
        )}
      </div>

      {/* ════════ PRINT-ONLY DOCUMENT ══════════════════════════════════════ */}
      {data && (
        <div className="hidden print:block report-document" style={{ fontFamily: "Arial, sans-serif", color: "#111", lineHeight: 1.5 }}>
          <h1 style={{ fontSize: "21px", fontWeight: "bold", borderBottom: "2px solid #111", paddingBottom: "8px", marginBottom: "16px" }}>
            GPU Platform — Usage Report
          </h1>
          <table style={{ ...TABLE, width: "auto", minWidth: "320px" }}><tbody>
            <tr><td style={{ ...TH, width: "120px" }}>Period</td><td style={TD}>{periodLabel}</td></tr>
            <tr><td style={TH}>Scope</td><td style={TD}>{scopeLabel}</td></tr>
            <tr><td style={TH}>Timezone</td><td style={TD}>{tz}</td></tr>
            <tr><td style={TH}>Total requests</td><td style={TD}>{fmtInt(data.summary.total_requests)} · {data.summary.distinct_models} models · {data.summary.distinct_apps} endpoints</td></tr>
            <tr><td style={TH}>Outcomes</td><td style={TD}>{fmtInt(data.summary.completed)} ok · {fmtInt(data.summary.client_cancelled)} ~4xx · {fmtInt(data.summary.server_error)} ~5xx{data.summary.success_rate != null ? ` · ${data.summary.success_rate}% success` : ""}</td></tr>
            <tr><td style={TH}>Tokens</td><td style={TD}>{fmtInt(data.summary.tokens_total)} total ({data.summary.token_coverage_pct ?? 0}% coverage)</td></tr>
          </tbody></table>

          <PrintH n="1. By model" />
          <PrintTable head={["Model", "Requests", "OK", "~4xx", "~5xx", "Tokens", "Avg lat"]}
            rows={data.by_model.map((m) => [m.model, fmtInt(m.requests), fmtInt(m.completed), fmtInt(m.client_cancelled), fmtInt(m.server_error), fmtTokens(m.tokens_total), fmtLat(m.avg_latency_s)])} />

          <PrintH n="2. By endpoint" />
          <PrintTable head={["Endpoint", "Requests", "OK", "~5xx"]}
            rows={data.by_endpoint.map((e) => [e.endpoint, fmtInt(e.requests), fmtInt(e.completed), fmtInt(e.server_error)])} />

          <PrintH n="3. Top users" />
          <PrintTable head={["User", "Requests", "Tokens"]} rows={data.by_user.map((u) => [u.username, fmtInt(u.requests), fmtTokens(u.tokens_total)])} />

          {spend && (<>
            <PrintH n={`4. Resource spend — $${spend.total_cost_usd.toFixed(2)}`} />
            <PrintTable head={["Type", "Count", "GPU-hours", "Cost (USD)"]}
              rows={spend.by_type.map((r) => [r.resource_type, fmtInt(r.count), r.gpu_hours == null ? "—" : r.gpu_hours.toFixed(1), `$${r.cost_usd.toFixed(2)}`])} />
          </>)}

          <h2 style={{ fontSize: "16px", fontWeight: "bold", marginTop: "24px", marginBottom: "8px", borderTop: "1px solid #ccc", paddingTop: "16px" }}>Daily breakdown</h2>
          {data.daily.map((day) => (
            <div key={day.date} style={{ marginTop: "14px" }}>
              <h3 style={{ fontSize: "13px", fontWeight: "bold", marginBottom: "6px" }}>
                {day.day_label} — {fmtInt(day.requests)} reqs · {day.client_cancelled} ~4xx · {day.server_error} ~5xx · {fmtTokens(day.tokens_total)} tok
              </h3>
              {day.jobs.length > 0 && (
                <PrintTable head={["Time", "Model", "Endpoint", "User", "Outcome", "Status", "Elapsed"]}
                  rows={day.jobs.map((j) => [j.start_time, j.model, j.endpoint, j.username, j.outcome, j.status, j.elapsed_label])} />
              )}
            </div>
          ))}

          <p style={{ marginTop: "28px", fontSize: "10px", color: "#888", borderTop: "1px solid #ddd", paddingTop: "10px" }}>{data.note}</p>
        </div>
      )}
    </div>
  );
}

// ── Small screen table (no ui/table primitive in this repo) ─────────────────
function ScreenTable({ head, rows, mono = [], right = [], empty }: {
  head: string[]; rows: (string | number)[][]; mono?: number[]; right?: number[]; empty: string;
}) {
  if (rows.length === 0) return <p className="py-3 text-sm text-muted-foreground italic">{empty}</p>;
  const monoSet = new Set(mono), rightSet = new Set(right);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            {head.map((h, i) => <th key={i} className={`py-1.5 px-2 font-medium text-muted-foreground ${rightSet.has(i) ? "text-right" : "text-left"}`}>{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri} className="border-b border-border/50 last:border-0">
              {r.map((c, ci) => (
                <td key={ci} className={`py-1.5 px-2 ${rightSet.has(ci) ? "text-right tabular-nums" : ""} ${monoSet.has(ci) ? "font-mono text-xs" : ""} ${ci === 1 && !rightSet.has(ci) ? "" : ""}`}>
                  <span className={monoSet.has(ci) ? "truncate inline-block max-w-[220px] align-bottom" : ""}>{c}</span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PrintH({ n }: { n: string }) {
  return <h3 style={{ fontSize: "14px", fontWeight: "bold", marginTop: "16px", marginBottom: "6px" }}>{n}</h3>;
}
function PrintTable({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  if (rows.length === 0) return <p style={{ color: "#666", fontSize: "12px", marginBottom: "12px" }}>None.</p>;
  return (
    <table style={TABLE}>
      <thead><tr>{head.map((h, i) => <th key={i} style={TH}>{h}</th>)}</tr></thead>
      <tbody>{rows.map((r, ri) => <tr key={ri}>{r.map((c, ci) => <td key={ci} style={TD}>{c}</td>)}</tr>)}</tbody>
    </table>
  );
}
