"use client";

// Live scrape of GET /proxy/{name}/metrics — the proxy router's own Prometheus
// series (proxy_requests_total{model,upstream,status}, proxy_request_duration_seconds,
// proxy_ttft_seconds, proxy_tokens_per_second). Parsed + rendered client-side; the
// gateway holds these counters in-memory (reset on restart) and the timeline below
// accumulates in the browser only while this tab is open. Mirrors the serverless
// MetricsTab but is summarized by model / upstream / outcome instead of HTTP route.

import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, RefreshCw, ChevronRight } from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  LabelList,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import type { ProxyEndpoint } from "@/lib/types";
import {
  parseExposition,
  fmtLatency,
  fmtInt,
  fmtTps,
  histBuckets,
  SummaryCard,
  LegendDot,
  BucketChart,
  TOOLTIP_STYLE,
  COLOR_REQ,
  COLOR_ERR,
  COLOR_TTFT,
  COLOR_TPS,
  REFRESH_MS,
  type Sample,
  type Bucket,
} from "@/app/(app)/serverless/[id]/tabs/metrics";

const REQ_TOTAL = "proxy_requests_total";
const DUR = "proxy_request_duration_seconds";
const TTFT = "proxy_ttft_seconds";
const TPS = "proxy_tokens_per_second";

// A status counts as a success only when it completed; "cancelled" is neutral
// (client hung up / manual flush), everything else is an error.
const isError = (status: string) => status !== "completed" && status !== "cancelled";

type Tally = { requests: number; completed: number; cancelled: number };
const newTally = (): Tally => ({ requests: 0, completed: 0, cancelled: 0 });
const errorsOf = (t: Tally) => t.requests - t.completed - t.cancelled;

type HistoryPoint = { t: string; requests: number; errors: number };

type Summary = {
  total: number;
  errors: number;
  cancelled: number;
  byModel: { model: string; tally: Tally; latAvg: number | null; ttftAvg: number | null; tpsAvg: number | null }[];
  byUpstream: { upstream: string; tally: Tally }[];
  byStatus: { status: string; count: number }[];
  latAvgAll: number | null;
  ttftAvgAll: number | null;
  tpsAvgAll: number | null;
  ttftBuckets: Bucket[];
  tpsBuckets: Bucket[];
};

const sumOf = (samples: Sample[], name: string) =>
  samples.filter((s) => s.name === `${name}_sum`).reduce((a, s) => a + s.value, 0);
const countOf = (samples: Sample[], name: string) =>
  samples.filter((s) => s.name === `${name}_count`).reduce((a, s) => a + s.value, 0);
const avg = (sum: number, count: number) => (count > 0 ? sum / count : null);

function summarize(samples: Sample[]): Summary {
  const models = new Map<string, Tally>();
  const upstreams = new Map<string, Tally>();
  const statuses = new Map<string, number>();
  let total = 0;
  let cancelled = 0;
  const bump = (map: Map<string, Tally>, key: string, status: string, v: number) => {
    let t = map.get(key);
    if (!t) { t = newTally(); map.set(key, t); }
    t.requests += v;
    if (status === "completed") t.completed += v;
    else if (status === "cancelled") t.cancelled += v;
  };
  for (const s of samples) {
    if (s.name !== REQ_TOTAL) continue;
    const status = s.labels.status ?? "";
    const model = s.labels.model || "(default)";
    const upstream = s.labels.upstream || "(none)";
    bump(models, model, status, s.value);
    bump(upstreams, upstream, status, s.value);
    statuses.set(status, (statuses.get(status) ?? 0) + s.value);
    total += s.value;
    if (status === "cancelled") cancelled += s.value;
  }
  let errors = 0;
  for (const [status, count] of statuses) if (isError(status)) errors += count;

  // Latency / ttft / tps are labelled by model only — build a per-model lookup
  // of sum/count for the table, and the proxy-wide aggregate for the cards.
  const modelDur = new Map<string, { sum: number; count: number }>();
  const modelTtft = new Map<string, { sum: number; count: number }>();
  const modelTps = new Map<string, { sum: number; count: number }>();
  const collect = (map: Map<string, { sum: number; count: number }>, name: string) => {
    for (const s of samples) {
      if (s.name !== `${name}_sum` && s.name !== `${name}_count`) continue;
      const model = s.labels.model || "(default)";
      let e = map.get(model);
      if (!e) { e = { sum: 0, count: 0 }; map.set(model, e); }
      if (s.name === `${name}_sum`) e.sum += s.value;
      else e.count += s.value;
    }
  };
  collect(modelDur, DUR);
  collect(modelTtft, TTFT);
  collect(modelTps, TPS);

  const byModel = [...models.entries()]
    .map(([model, tally]) => ({
      model,
      tally,
      latAvg: avg(modelDur.get(model)?.sum ?? 0, modelDur.get(model)?.count ?? 0),
      ttftAvg: avg(modelTtft.get(model)?.sum ?? 0, modelTtft.get(model)?.count ?? 0),
      tpsAvg: avg(modelTps.get(model)?.sum ?? 0, modelTps.get(model)?.count ?? 0),
    }))
    .sort((a, b) => b.tally.requests - a.tally.requests);

  const byUpstream = [...upstreams.entries()]
    .map(([upstream, tally]) => ({ upstream, tally }))
    .sort((a, b) => b.tally.requests - a.tally.requests);

  const byStatus = [...statuses.entries()]
    .map(([status, count]) => ({ status, count }))
    .sort((a, b) => b.count - a.count);

  return {
    total,
    errors,
    cancelled,
    byModel,
    byUpstream,
    byStatus,
    latAvgAll: avg(sumOf(samples, DUR), countOf(samples, DUR)),
    ttftAvgAll: avg(sumOf(samples, TTFT), countOf(samples, TTFT)),
    tpsAvgAll: avg(sumOf(samples, TPS), countOf(samples, TPS)),
    ttftBuckets: histBuckets(samples, TTFT, (n) => fmtLatency(n)),
    tpsBuckets: histBuckets(samples, TPS, fmtTps),
  };
}

const STATUS_TONE: Record<string, "secondary" | "outline" | "destructive"> = {
  completed: "secondary",
  cancelled: "outline",
};

export function ProxyMetricsTab({ ep }: { ep: ProxyEndpoint }) {
  const [raw, setRaw] = useState("");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [auto, setAuto] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const inFlight = useRef(false);
  const prevTotals = useRef<{ total: number; errors: number } | null>(null);

  const scrapePath = `/proxy/${ep.name}/metrics`;

  const scrape = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`/api/proxy/proxy/${encodeURIComponent(ep.name)}/metrics`, { cache: "no-store" });
      if (!r.ok) throw new Error(`scrape failed: HTTP ${r.status}`);
      const text = await r.text();
      const sum = summarize(parseExposition(text));
      setRaw(text);
      setSummary(sum);
      const now = new Date();
      const prev = prevTotals.current;
      if (prev) {
        setHistory((h) =>
          [...h, {
            t: now.toLocaleTimeString(),
            requests: Math.max(0, sum.total - prev.total),
            errors: Math.max(0, sum.errors - prev.errors),
          }].slice(-60),
        );
      }
      prevTotals.current = { total: sum.total, errors: sum.errors };
      setUpdatedAt(now.toLocaleTimeString());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
      inFlight.current = false;
    }
  }, [ep.name]);

  useEffect(() => { scrape(); }, [scrape]);
  useEffect(() => {
    if (!auto) return;
    const id = window.setInterval(scrape, REFRESH_MS);
    return () => window.clearInterval(id);
  }, [auto, scrape]);

  const errorRate = summary && summary.total > 0 ? (summary.errors / summary.total) * 100 : 0;
  const hasData = summary != null && summary.total > 0;
  const upstreamBars = summary
    ? summary.byUpstream.map((u) => ({
        name: u.upstream.length > 24 ? `${u.upstream.slice(0, 23)}…` : u.upstream,
        success: u.tally.completed,
        errors: errorsOf(u.tally),
        requests: u.tally.requests,
      }))
    : [];
  const upstreamChartHeight = Math.min(340, Math.max(120, upstreamBars.length * 36));

  return (
    <div className="space-y-5">
      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="space-y-0.5">
          <div className="text-sm text-muted-foreground">
            Live scrape of{" "}
            <code className="rounded bg-muted px-1 py-0.5 text-xs text-foreground">GET {scrapePath}</code>{" "}
            — routing metrics for this proxy. Not persisted (resets on gateway restart).
          </div>
          {updatedAt && (
            <div className="text-xs text-muted-foreground">
              Updated {updatedAt}{auto && ` · auto-refresh ${REFRESH_MS / 1000}s`}
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <Switch checked={auto} onCheckedChange={setAuto} />
            Auto-refresh
          </label>
          <Button variant="outline" size="sm" onClick={scrape} disabled={loading}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            Refresh
          </Button>
        </div>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Summary cards */}
      {summary && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <SummaryCard label="Requests" value={fmtInt(summary.total)} />
          <SummaryCard label="Errors" value={fmtInt(summary.errors)} tone={summary.errors > 0 ? "bad" : "neutral"} />
          <SummaryCard label="Error rate" value={`${errorRate.toFixed(errorRate < 10 ? 1 : 0)}%`} tone={errorRate > 0 ? "bad" : "neutral"} />
          <SummaryCard label="Avg latency" value={fmtLatency(summary.latAvgAll)} />
          <SummaryCard label="Avg TTFT" value={fmtLatency(summary.ttftAvgAll)} />
          <SummaryCard label="Avg TPS" value={summary.tpsAvgAll != null ? `${summary.tpsAvgAll.toFixed(1)} tok/s` : "—"} />
        </div>
      )}

      {/* Timeline + requests by upstream */}
      {summary && (
        <div className="grid gap-4 lg:grid-cols-2">
          <ChartCard title="Requests & errors per interval" extra={auto ? "(live)" : "(per refresh)"}
                     legend={<><LegendDot color={COLOR_REQ} label="requests" /><LegendDot color={COLOR_ERR} label="errors" /></>}>
            <div className="h-48 w-full">
              {history.length > 1 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                    <XAxis dataKey="t" tick={{ fontSize: 10, fill: "#6b7280" }} stroke="#d4d4d8" minTickGap={28} />
                    <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: "#6b7280" }} stroke="#d4d4d8" width={36} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Line type="monotone" dataKey="requests" stroke={COLOR_REQ} strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="errors" stroke={COLOR_ERR} strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <Empty>Collecting… enable auto-refresh (or refresh again) to build the timeline.</Empty>
              )}
            </div>
          </ChartCard>

          <ChartCard title="Requests by upstream"
                     legend={<><LegendDot color={COLOR_REQ} label="completed" /><LegendDot color={COLOR_ERR} label="errors" /></>}>
            <div className="w-full" style={{ height: upstreamChartHeight }}>
              {hasData ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={upstreamBars} layout="vertical" margin={{ top: 4, right: 30, left: 4, bottom: 0 }} barCategoryGap="22%">
                    <XAxis type="number" domain={[0, "dataMax"]} allowDecimals={false} tick={{ fontSize: 10, fill: "#6b7280" }} stroke="#d4d4d8" />
                    <YAxis type="category" dataKey="name" width={140} interval={0} tickLine={false} axisLine={false} tick={{ fontSize: 10, fill: "#9ca3af" }} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "#000", opacity: 0.05 }} />
                    <Bar dataKey="success" name="completed" stackId="reqs" fill={COLOR_REQ} />
                    <Bar dataKey="errors" name="errors" stackId="reqs" fill={COLOR_ERR} radius={[0, 3, 3, 0]}>
                      <LabelList dataKey="requests" position="right" fontSize={10} fill="#9ca3af" />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <Empty>No data yet.</Empty>
              )}
            </div>
          </ChartCard>
        </div>
      )}

      {/* TTFT + TPS bucket distributions */}
      {summary && (
        <div className="grid gap-4 lg:grid-cols-2">
          <ChartCard title="Time-to-first-token distribution" extra="(streamed requests)"
                     legend={<LegendDot color={COLOR_TTFT} label="requests" />}>
            <BucketChart data={summary.ttftBuckets} color={COLOR_TTFT} />
          </ChartCard>
          <ChartCard title="Output throughput distribution" extra="(tokens/s)"
                     legend={<LegendDot color={COLOR_TPS} label="requests" />}>
            <BucketChart data={summary.tpsBuckets} color={COLOR_TPS} />
          </ChartCard>
        </div>
      )}

      {/* Status breakdown */}
      {summary && summary.total > 0 && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-muted-foreground">Outcomes:</span>
          {summary.byStatus.map((s) => (
            <Badge key={s.status} variant={STATUS_TONE[s.status] ?? "destructive"}>{s.status} {fmtInt(s.count)}</Badge>
          ))}
        </div>
      )}

      {/* Per-model table */}
      {hasData ? (
        <div className="overflow-x-auto rounded-md border border-border bg-card">
          <table className="w-full text-sm text-foreground">
            <thead>
              <tr className="border-b border-border text-left text-xs text-muted-foreground">
                <th className="px-3 py-2 font-medium">Model</th>
                <th className="px-3 py-2 text-right font-medium">Requests</th>
                <th className="px-3 py-2 text-right font-medium">Errors</th>
                <th className="px-3 py-2 text-right font-medium">Avg latency</th>
                <th className="px-3 py-2 text-right font-medium">Avg TTFT</th>
                <th className="px-3 py-2 text-right font-medium">Avg TPS</th>
              </tr>
            </thead>
            <tbody>
              {summary!.byModel.map((m) => (
                <tr key={m.model} className="border-b border-border/50 last:border-0">
                  <td className="px-3 py-2 font-mono text-xs">{m.model}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{fmtInt(m.tally.requests)}</td>
                  <td className={`px-3 py-2 text-right tabular-nums ${errorsOf(m.tally) > 0 ? "text-destructive" : "text-muted-foreground"}`}>
                    {fmtInt(errorsOf(m.tally))}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">{fmtLatency(m.latAvg)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{fmtLatency(m.ttftAvg)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{m.tpsAvg != null ? `${m.tpsAvg.toFixed(1)} tok/s` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        summary && (
          <div className="rounded-md border border-border bg-card px-4 py-6 text-center text-sm text-muted-foreground">
            No proxied requests recorded yet. Metrics appear once traffic flows through <code>/proxy/{ep.name}/v1/…</code>.
          </div>
        )
      )}

      {/* Raw exposition */}
      {raw && (
        <div className="rounded-md border border-border bg-card text-foreground">
          <button type="button" onClick={() => setShowRaw((v) => !v)}
                  className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-medium hover:bg-muted/40">
            <ChevronRight className={`h-4 w-4 transition-transform ${showRaw ? "rotate-90" : ""}`} />
            Raw exposition
          </button>
          {showRaw && (
            <pre className="max-h-[28rem] overflow-auto border-t border-border bg-card px-3 py-2 text-xs leading-relaxed text-foreground scrollbar-thin">
              {raw}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

function ChartCard({ title, extra, legend, children }: { title: string; extra?: string; legend?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="min-w-0 rounded-md border border-border bg-card p-3 text-foreground">
      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
        <span className="font-medium">{title}{extra && <span className="font-normal text-muted-foreground"> {extra}</span>}</span>
        {legend}
      </div>
      {children}
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="flex h-full items-center justify-center text-center text-xs text-muted-foreground">{children}</div>;
}
