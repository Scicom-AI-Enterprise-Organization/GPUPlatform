"use client";

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
import type { AppRecord } from "@/lib/types";

// Live scrape of GET /{app_id}/metrics (the same Prometheus endpoint Grafana
// scrapes) — parsed and rendered client-side. Nothing is persisted: the gateway
// holds these counters in-memory (reset on restart), and the timeline below is
// accumulated in the browser only while this tab is open.

type Sample = { name: string; labels: Record<string, string>; value: number };

type Row = {
  method: string;
  endpoint: string;
  requests: number;
  errors: number;
  latSum: number;
  latCount: number;
};

type HistoryPoint = { t: string; requests: number; errors: number };

const REFRESH_MS = 5000;
const COLOR_REQ = "#60a5fa"; // blue-400
const COLOR_ERR = "#f87171"; // red-400
const TOOLTIP_STYLE = {
  background: "#ffffff",
  border: "1px solid #e5e7eb",
  borderRadius: 6,
  fontSize: 12,
  color: "#111827",
} as const;

// Prometheus exposition parser. Quote-aware brace matching matters here because a
// label value like endpoint="/{app_id}/v1/chat/completions" contains a `}`.
function parseExposition(text: string): Sample[] {
  const out: Sample[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const nameMatch = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)/);
    if (!nameMatch) continue;
    const name = nameMatch[1];
    let rest = line.slice(name.length);
    const labels: Record<string, string> = {};
    if (rest.startsWith("{")) {
      let end = -1;
      let inQuote = false;
      for (let j = 1; j < rest.length; j++) {
        const c = rest[j];
        if (inQuote) {
          if (c === "\\") j++;
          else if (c === '"') inQuote = false;
        } else if (c === '"') inQuote = true;
        else if (c === "}") {
          end = j;
          break;
        }
      }
      if (end === -1) continue;
      const labelStr = rest.slice(1, end);
      rest = rest.slice(end + 1);
      const re = /([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"/g;
      let lm: RegExpExecArray | null;
      while ((lm = re.exec(labelStr))) {
        labels[lm[1]] = lm[2]
          .replace(/\\"/g, '"')
          .replace(/\\n/g, "\n")
          .replace(/\\\\/g, "\\");
      }
    }
    const valStr = rest.trim().split(/\s+/)[0];
    const value = Number(valStr);
    if (!Number.isFinite(value)) continue;
    out.push({ name, labels, value });
  }
  return out;
}

type Summary = {
  rows: Row[];
  total: number;
  errors: number;
  c2: number;
  c4: number;
  c5: number;
  latSumAll: number;
  latCountAll: number;
};

function summarize(samples: Sample[]): Summary {
  const rows = new Map<string, Row>();
  const get = (method: string, endpoint: string) => {
    const k = `${method} ${endpoint}`;
    let r = rows.get(k);
    if (!r) {
      r = { method, endpoint, requests: 0, errors: 0, latSum: 0, latCount: 0 };
      rows.set(k, r);
    }
    return r;
  };
  let total = 0;
  let errors = 0;
  let c2 = 0;
  let c4 = 0;
  let c5 = 0;
  let latSumAll = 0;
  let latCountAll = 0;
  for (const s of samples) {
    if (s.name === "http_requests_total") {
      const r = get(s.labels.method ?? "", s.labels.endpoint ?? "");
      r.requests += s.value;
      total += s.value;
      const cls = (s.labels.http_status ?? "")[0];
      if (cls === "2") c2 += s.value;
      else {
        errors += s.value;
        r.errors += s.value;
        if (cls === "4") c4 += s.value;
        else if (cls === "5") c5 += s.value;
      }
    } else if (s.name === "http_request_duration_seconds_sum") {
      const r = get(s.labels.method ?? "", s.labels.endpoint ?? "");
      r.latSum += s.value;
      latSumAll += s.value;
    } else if (s.name === "http_request_duration_seconds_count") {
      const r = get(s.labels.method ?? "", s.labels.endpoint ?? "");
      r.latCount += s.value;
      latCountAll += s.value;
    }
  }
  const list = [...rows.values()].sort((a, b) => b.requests - a.requests);
  return { rows: list, total, errors, c2, c4, c5, latSumAll, latCountAll };
}

function fmtLatency(seconds: number | null): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  return `${seconds.toFixed(2)} s`;
}

function fmtInt(n: number): string {
  return Math.round(n).toLocaleString();
}

// Trim the "/{app_id}/v1/" prefix so endpoint labels are short enough to leave
// the bars most of the card width (falls back to dropping just "/{app_id}").
function shortEndpoint(ep: string): string {
  return (
    ep.replace(/^\/\{app_id\}\/v1\//, "").replace(/^\/\{app_id\}\//, "") || ep
  );
}

export function MetricsTab({ app }: { app: AppRecord }) {
  const [raw, setRaw] = useState<string>("");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [auto, setAuto] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const inFlight = useRef(false);
  const prevTotals = useRef<{ total: number; errors: number } | null>(null);

  const scrapePath = `/${app.app_id}/metrics`;

  const scrape = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`/api/proxy/${encodeURIComponent(app.app_id)}/metrics`, {
        cache: "no-store",
      });
      if (!r.ok) throw new Error(`scrape failed: HTTP ${r.status}`);
      const text = await r.text();
      const sum = summarize(parseExposition(text));
      setRaw(text);
      setSummary(sum);
      // Build the live timeline from successive scrapes (per-interval deltas of
      // the cumulative counters; clamped ≥0 so a gateway restart reads as 0).
      const now = new Date();
      const prev = prevTotals.current;
      if (prev) {
        setHistory((h) =>
          [
            ...h,
            {
              t: now.toLocaleTimeString(),
              requests: Math.max(0, sum.total - prev.total),
              errors: Math.max(0, sum.errors - prev.errors),
            },
          ].slice(-60),
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
  }, [app.app_id]);

  useEffect(() => {
    scrape();
  }, [scrape]);

  useEffect(() => {
    if (!auto) return;
    const id = window.setInterval(scrape, REFRESH_MS);
    return () => window.clearInterval(id);
  }, [auto, scrape]);

  const errorRate =
    summary && summary.total > 0 ? (summary.errors / summary.total) * 100 : 0;
  const avgAll =
    summary && summary.latCountAll > 0 ? summary.latSumAll / summary.latCountAll : null;
  const hasHttp = summary != null && summary.rows.length > 0;
  const barData = summary
    ? summary.rows.map((r) => ({
        name: shortEndpoint(r.endpoint),
        requests: r.requests,
        errors: r.errors,
      }))
    : [];

  return (
    <div className="space-y-5">
      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="space-y-0.5">
          <div className="text-sm text-muted-foreground">
            Live scrape of{" "}
            <code className="rounded bg-muted px-1 py-0.5 text-xs text-foreground">
              GET {scrapePath}
            </code>{" "}
            — gateway API metrics for this fleet. Not persisted (resets on gateway
            restart).
          </div>
          {updatedAt && (
            <div className="text-xs text-muted-foreground">
              Updated {updatedAt}
              {auto && ` · auto-refresh ${REFRESH_MS / 1000}s`}
            </div>
          )}
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <Switch checked={auto} onCheckedChange={setAuto} />
            Auto-refresh
          </label>
          <Button variant="outline" size="sm" onClick={scrape} disabled={loading}>
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
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
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <SummaryCard label="Requests" value={fmtInt(summary.total)} />
          <SummaryCard
            label="Errors (non-2xx)"
            value={fmtInt(summary.errors)}
            tone={summary.errors > 0 ? "bad" : "neutral"}
          />
          <SummaryCard
            label="Error rate"
            value={`${errorRate.toFixed(errorRate < 10 ? 1 : 0)}%`}
            tone={errorRate > 0 ? "bad" : "neutral"}
          />
          <SummaryCard label="Avg latency" value={fmtLatency(avgAll)} />
        </div>
      )}

      {/* Charts — 2-up grid; min-w-0 lets the chart fill the grid cell */}
      {summary && (
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="min-w-0 rounded-md border border-border bg-card p-3 text-foreground">
            <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
              <span className="font-medium">
                Requests &amp; errors per interval{" "}
                <span className="font-normal text-muted-foreground">
                  {auto ? "(live)" : "(per refresh)"}
                </span>
              </span>
              <LegendDot color={COLOR_REQ} label="requests" />
              <LegendDot color={COLOR_ERR} label="errors" />
            </div>
            <div className="h-48 w-full">
              {history.length > 1 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={history}
                    margin={{ top: 4, right: 8, left: -20, bottom: 0 }}
                  >
                    <XAxis
                      dataKey="t"
                      tick={{ fontSize: 10, fill: "#6b7280" }}
                      stroke="#d4d4d8"
                      minTickGap={28}
                    />
                    <YAxis
                      allowDecimals={false}
                      tick={{ fontSize: 10, fill: "#6b7280" }}
                      stroke="#d4d4d8"
                      width={32}
                    />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Line
                      type="monotone"
                      dataKey="requests"
                      stroke={COLOR_REQ}
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="errors"
                      stroke={COLOR_ERR}
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center text-center text-xs text-muted-foreground">
                  Collecting… enable auto-refresh (or refresh again) to build the
                  timeline.
                </div>
              )}
            </div>
          </div>

          <div className="min-w-0 rounded-md border border-border bg-card p-3 text-foreground">
            <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
              <span className="font-medium">Requests by endpoint</span>
              <LegendDot color={COLOR_REQ} label="requests" />
              <LegendDot color={COLOR_ERR} label="errors" />
            </div>
            <div className="h-48 w-full">
              {hasHttp ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={barData}
                    layout="vertical"
                    margin={{ top: 4, right: 16, left: 6, bottom: 0 }}
                  >
                    <XAxis
                      type="number"
                      domain={[0, "dataMax"]}
                      allowDecimals={false}
                      tick={{ fontSize: 10, fill: "#6b7280" }}
                      stroke="#d4d4d8"
                    />
                    {/* Hidden so the bars span the full card width; the endpoint
                        name is drawn inside the requests bar instead of a left
                        column. */}
                    <YAxis type="category" dataKey="name" hide />
                    <Tooltip
                      contentStyle={TOOLTIP_STYLE}
                      cursor={{ fill: "#000", opacity: 0.05 }}
                    />
                    <Bar dataKey="requests" fill={COLOR_REQ} radius={[0, 3, 3, 0]}>
                      <LabelList
                        dataKey="name"
                        position="insideLeft"
                        fill="#ffffff"
                        fontSize={11}
                      />
                    </Bar>
                    <Bar dataKey="errors" fill={COLOR_ERR} radius={[0, 3, 3, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
                  No data yet.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Status breakdown */}
      {summary && summary.total > 0 && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-muted-foreground">Status:</span>
          <Badge variant="secondary">2xx {fmtInt(summary.c2)}</Badge>
          {summary.c4 > 0 && <Badge variant="outline">4xx {fmtInt(summary.c4)}</Badge>}
          {summary.c5 > 0 && (
            <Badge variant="destructive">5xx {fmtInt(summary.c5)}</Badge>
          )}
        </div>
      )}

      {/* Per-endpoint table */}
      {hasHttp ? (
        <div className="overflow-x-auto rounded-md border border-border bg-card">
          <table className="w-full text-sm text-foreground">
            <thead>
              <tr className="border-b border-border text-left text-xs text-muted-foreground">
                <th className="px-3 py-2 font-medium">Method</th>
                <th className="px-3 py-2 font-medium">Endpoint</th>
                <th className="px-3 py-2 text-right font-medium">Requests</th>
                <th className="px-3 py-2 text-right font-medium">Errors</th>
                <th className="px-3 py-2 text-right font-medium">Avg latency</th>
              </tr>
            </thead>
            <tbody>
              {summary!.rows.map((r) => {
                const avg = r.latCount > 0 ? r.latSum / r.latCount : null;
                return (
                  <tr
                    key={`${r.method} ${r.endpoint}`}
                    className="border-b border-border/50 last:border-0"
                  >
                    <td className="px-3 py-2 font-mono text-xs text-foreground">
                      {r.method}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-foreground">
                      {r.endpoint}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {fmtInt(r.requests)}
                    </td>
                    <td
                      className={`px-3 py-2 text-right tabular-nums ${
                        r.errors > 0 ? "text-destructive" : "text-muted-foreground"
                      }`}
                    >
                      {fmtInt(r.errors)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {fmtLatency(avg)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        summary && (
          <div className="rounded-md border border-border bg-card px-4 py-6 text-center text-sm text-muted-foreground">
            No API requests recorded yet. Metrics appear once the fleet serves
            traffic through <code>/{app.app_id}/v1/…</code>.
          </div>
        )
      )}

      {/* Raw exposition (includes the vLLM worker metrics too) */}
      {raw && (
        <div className="rounded-md border border-border bg-card text-foreground">
          <button
            type="button"
            onClick={() => setShowRaw((v) => !v)}
            className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-medium hover:bg-muted/40"
          >
            <ChevronRight
              className={`h-4 w-4 transition-transform ${showRaw ? "rotate-90" : ""}`}
            />
            Raw exposition
            <span className="text-xs font-normal text-muted-foreground">
              (gateway HTTP + vLLM worker metrics)
            </span>
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

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1 text-muted-foreground">
      <span
        className="inline-block h-2 w-2 rounded-full"
        style={{ background: color }}
      />
      {label}
    </span>
  );
}

function SummaryCard({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "bad";
}) {
  const valueColor = tone === "bad" ? "text-destructive" : "text-foreground";
  return (
    <div className="rounded-md border border-border bg-card px-3 py-2.5">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={`mt-0.5 text-xl font-semibold tabular-nums ${valueColor}`}>
        {value}
      </div>
    </div>
  );
}
