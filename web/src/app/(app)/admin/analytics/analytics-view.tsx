"use client";

/**
 * Admin Analytics — usage + spend across both platforms over a period.
 *
 * Data sources (fetched separately, deliberately NOT SlurmUI's combined
 * endpoint, so each side stays granular and independently filterable):
 *   - GPU Platform: /api/analytics/gpuplatform → gateway /v1/history/{kind}
 *     (benchmark / training / compute / inference / proxy). Cost is computed
 *     here as detail.cost_per_hr × duration; inference/proxy carry no cost —
 *     they contribute activity counts and token totals instead.
 *   - SlurmUI: /api/analytics/slurm → SlurmUI /api/reports (jobs, GPU-hours,
 *     per-day history). No $ cost — Slurm jobs contribute counts + GPU-hours.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Download, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// ── types ────────────────────────────────────────────────────────────────────

type HistoryJob = {
  kind: string;
  id: string;
  name: string | null;
  user: string;
  status: string;
  created_at: string | null;
  ended_at: string | null;
  duration_s: number | null;
  detail: Record<string, unknown>;
};

type GpuPlatformPayload = {
  kinds: Record<string, HistoryJob[]>;
  truncated: string[];
};

type SlurmDailyJob = {
  slurmJobId: number | null;
  jobName: string | null;
  unixUsername: string | null;
  state: string;
  partition: string | null;
};

type SlurmReport = {
  summary: { totalJobs: number; gpuHours: number; cpuHours: number };
  dailyJobHistory: {
    date: string;
    completed: number;
    failed: number;
    cancelled: number;
    gpuHours: number;
    jobs: SlurmDailyJob[];
  }[];
};

type SlurmPayload =
  | { configured: false }
  | { configured: true; report?: SlurmReport; error?: string };

// One normalized activity record — every chart below derives from these.
type Rec = {
  platform: "gpuplatform" | "slurmui";
  app: string; // serverless | benchmark | autotrain | compute | proxy | slurmjob
  user: string;
  date: string; // YYYY-MM-DD (local)
  status: string;
  costUsd: number; // 0 when the source has no $ figure
  gpuHours: number; // 0 when unknown
  // Where it ran — registered provider name (TM-H20, TM-VM1, …), a cloud
  // ("RunPod GPUs", "Prime Intellect GPUs") or a Slurm partition. Free-form:
  // the GPU-source filter is built dynamically from whatever shows up here.
  source: string;
};

// ── filters ──────────────────────────────────────────────────────────────────

const PERIODS = [
  { value: "7d", label: "Last 7 Days" },
  { value: "thisMonth", label: "This Month" },
  { value: "lastMonth", label: "Last Month" },
] as const;
type Period = (typeof PERIODS)[number]["value"];

const PLATFORMS = [
  { value: "gpuplatform", label: "GPU Platform" },
  { value: "slurmui", label: "SlurmUI" },
] as const;

const APPS = [
  { value: "serverless", label: "Serverless", platform: "gpuplatform" },
  { value: "benchmark", label: "Benchmark", platform: "gpuplatform" },
  { value: "autotrain", label: "Autotrain", platform: "gpuplatform" },
  { value: "compute", label: "Compute", platform: "gpuplatform" },
  { value: "proxy", label: "LLM Proxy", platform: "gpuplatform" },
  { value: "slurmjob", label: "Slurm jobs", platform: "slurmui" },
] as const;

const APP_COLORS: Record<string, string> = {
  serverless: "#60a5fa",
  benchmark: "#f59e0b",
  autotrain: "#a78bfa",
  compute: "#34d399",
  proxy: "#f472b6",
  slurmjob: "#fbbf24",
};

const KIND_TO_APP: Record<string, string> = {
  inference: "serverless",
  benchmark: "benchmark",
  training: "autotrain",
  compute: "compute",
  proxy: "proxy",
};

function periodRange(period: Period): { from: Date; to: Date } {
  const now = new Date();
  const startOfDay = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate());
  if (period === "7d") {
    const from = startOfDay(new Date(now.getTime() - 6 * 86400 * 1000));
    return { from, to: now };
  }
  if (period === "thisMonth") {
    return { from: new Date(now.getFullYear(), now.getMonth(), 1), to: now };
  }
  const from = new Date(now.getFullYear(), now.getMonth() - 1, 1);
  const to = new Date(now.getFullYear(), now.getMonth(), 0, 23, 59, 59);
  return { from, to };
}

const localDate = (d: Date) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;

function eachDay(from: Date, to: Date): string[] {
  const days: string[] = [];
  const d = new Date(from.getFullYear(), from.getMonth(), from.getDate());
  while (d <= to) {
    days.push(localDate(d));
    d.setDate(d.getDate() + 1);
  }
  return days;
}

const num = (x: unknown): number => (typeof x === "number" && isFinite(x) ? x : 0);

// ── normalization ────────────────────────────────────────────────────────────

const str = (x: unknown): string | null =>
  typeof x === "string" && x.trim() ? x.trim() : null;

// "Where did this run" label, best-effort from the history detail blob:
// a registered provider's name wins (TM-H20, TM-VM1, …); otherwise the cloud
// kind; for serverless/proxy requests fall back to the serving worker's GPU
// or the endpoint's configured GPU.
function gpuSource(detail: Record<string, unknown>): string {
  const provName = str(detail?.provider_name);
  if (provName) return provName;
  const kind = str(detail?.provider_kind) ?? str(detail?.backend);
  if (kind === "pi") return "Prime Intellect GPUs";
  if (kind === "runpod") return "RunPod GPUs";
  if (kind === "external") return "External";
  const worker = detail?.worker as Record<string, unknown> | null | undefined;
  return (
    str(worker?.gpu_name) ??
    str(detail?.requested_gpu_type) ??
    str(detail?.gpu_type) ??
    "RunPod GPUs" // platform-default cloud when no provider is recorded
  );
}

function normalizeGpuPlatform(payload: GpuPlatformPayload): Rec[] {
  const recs: Rec[] = [];
  for (const [kind, jobs] of Object.entries(payload.kinds)) {
    const app = KIND_TO_APP[kind] ?? kind;
    for (const j of jobs) {
      if (!j.created_at) continue;
      const durH = (j.duration_s ?? 0) / 3600;
      const costPerHr = num(j.detail?.cost_per_hr);
      const gpuCount = num(j.detail?.gpu_count) || 1;
      recs.push({
        source: gpuSource(j.detail ?? {}),
        platform: "gpuplatform",
        app,
        user: j.user,
        date: localDate(new Date(j.created_at)),
        status: j.status,
        costUsd: costPerHr * durH,
        // Inference/proxy requests are seconds-long API calls, not GPU
        // reservations — only the discrete job kinds count toward GPU-hours.
        gpuHours: app === "serverless" || app === "proxy" ? 0 : durH * gpuCount,
      });
    }
  }
  return recs;
}

function normalizeSlurm(report: SlurmReport): Rec[] {
  const recs: Rec[] = [];
  for (const day of report.dailyJobHistory ?? []) {
    const jobs = day.jobs ?? [];
    // GPU-hours arrive per-day, not per-job — spread evenly so user totals
    // still sum to the true daily figure.
    const perJobGpuH = jobs.length ? num(day.gpuHours) / jobs.length : 0;
    for (const j of jobs) {
      recs.push({
        source: j.partition ? `Slurm · ${j.partition}` : "Slurm",
        platform: "slurmui",
        app: "slurmjob",
        user: j.unixUsername ?? "(unknown)",
        date: day.date,
        status: (j.state ?? "").toLowerCase(),
        costUsd: 0,
        gpuHours: perJobGpuH,
      });
    }
  }
  return recs;
}

// ── CSV export ───────────────────────────────────────────────────────────────

function downloadCsv(filename: string, rows: (string | number)[][]) {
  const esc = (v: string | number) => {
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const blob = new Blob([rows.map((r) => r.map(esc).join(",")).join("\n")], {
    type: "text/csv;charset=utf-8",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── component ────────────────────────────────────────────────────────────────

export function AnalyticsView() {
  const [period, setPeriod] = useState<Period>("7d");
  const [platforms, setPlatforms] = useState<Set<string>>(
    new Set(["gpuplatform", "slurmui"]),
  );
  const [apps, setApps] = useState<Set<string>>(new Set(APPS.map((a) => a.value)));
  // GPU sources are discovered from the data, so we track the UNchecked ones —
  // newly-seen sources (a just-registered VM, a new partition) start checked.
  const [excludedSources, setExcludedSources] = useState<Set<string>>(new Set());

  const [loading, setLoading] = useState(true);
  const [gpuRecs, setGpuRecs] = useState<Rec[]>([]);
  const [slurmRecs, setSlurmRecs] = useState<Rec[]>([]);
  const [slurmState, setSlurmState] = useState<"ok" | "unconfigured" | "error">("ok");
  const [truncated, setTruncated] = useState<string[]>([]);

  const { from, to } = useMemo(() => periodRange(period), [period]);

  const load = useCallback(async () => {
    setLoading(true);
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone ?? "UTC";
    const [gpu, slurm] = await Promise.allSettled([
      fetch(
        `/api/analytics/gpuplatform?since=${from.toISOString()}&until=${new Date(to.getTime() + 1000).toISOString()}`,
        { cache: "no-store" },
      ).then((r) => (r.ok ? (r.json() as Promise<GpuPlatformPayload>) : Promise.reject(r.status))),
      fetch(
        `/api/analytics/slurm?from=${localDate(from)}&to=${localDate(to)}&tz=${encodeURIComponent(tz)}`,
        { cache: "no-store" },
      ).then((r) => r.json() as Promise<SlurmPayload>),
    ]);

    if (gpu.status === "fulfilled") {
      setGpuRecs(normalizeGpuPlatform(gpu.value));
      setTruncated(gpu.value.truncated);
    } else {
      setGpuRecs([]);
    }

    if (slurm.status === "fulfilled" && slurm.value.configured && "report" in slurm.value && slurm.value.report) {
      setSlurmRecs(normalizeSlurm(slurm.value.report));
      setSlurmState("ok");
    } else if (slurm.status === "fulfilled" && !slurm.value.configured) {
      setSlurmRecs([]);
      setSlurmState("unconfigured");
    } else {
      setSlurmRecs([]);
      setSlurmState("error");
    }
    setLoading(false);
  }, [from, to]);

  useEffect(() => {
    void load();
  }, [load]);

  // ── filtered + aggregated views ────────────────────────────────────────────

  const allSources = useMemo(
    () => [...new Set([...gpuRecs, ...slurmRecs].map((r) => r.source))].sort(),
    [gpuRecs, slurmRecs],
  );

  const recs = useMemo(
    () =>
      [...gpuRecs, ...slurmRecs].filter(
        (r) =>
          platforms.has(r.platform) &&
          apps.has(r.app) &&
          !excludedSources.has(r.source),
      ),
    [gpuRecs, slurmRecs, platforms, apps, excludedSources],
  );

  const days = useMemo(() => eachDay(from, to), [from, to]);

  const totals = useMemo(() => {
    const spend = recs.reduce((s, r) => s + r.costUsd, 0);
    const gpuHours = recs.reduce((s, r) => s + r.gpuHours, 0);
    const users = new Set(recs.map((r) => r.user));
    return {
      spend,
      gpuHours,
      dailyAvg: days.length ? spend / days.length : 0,
      activeUsers: users.size,
      days: days.length,
      activity: recs.length,
    };
  }, [recs, days]);

  const activeApps = useMemo(
    () => APPS.filter((a) => apps.has(a.value) && platforms.has(a.platform)),
    [apps, platforms],
  );

  const chartData = useMemo(() => {
    const byDay = new Map<string, Record<string, number>>();
    for (const d of days) byDay.set(d, {});
    for (const r of recs) {
      const row = byDay.get(r.date);
      if (!row) continue;
      row[r.app] = (row[r.app] ?? 0) + 1;
      row.__spend = (row.__spend ?? 0) + r.costUsd;
    }
    return days.map((d) => ({ date: d.slice(5), ...byDay.get(d) }));
  }, [recs, days]);

  // Donut: share of activity by app over the period.
  const appPie = useMemo(
    () =>
      activeApps
        .map((a) => ({
          name: a.label,
          value: recs.filter((r) => r.app === a.value).length,
          fill: APP_COLORS[a.value],
        }))
        .filter((s) => s.value > 0),
    [recs, activeApps],
  );

  // Kept (table removed) — still feeds the CSV export.
  const dailyRows = useMemo(
    () =>
      days
        .map((d) => {
          const dayRecs = recs.filter((r) => r.date === d);
          const perApp: Record<string, number> = {};
          for (const r of dayRecs) perApp[r.app] = (perApp[r.app] ?? 0) + 1;
          return {
            date: d,
            spend: dayRecs.reduce((s, r) => s + r.costUsd, 0),
            gpuHours: dayRecs.reduce((s, r) => s + r.gpuHours, 0),
            users: new Set(dayRecs.map((r) => r.user)).size,
            perApp,
            total: dayRecs.length,
          };
        })
        .reverse(),
    [recs, days],
  );

  const exportCsv = () => {
    const appCols = activeApps.map((a) => a.value);
    downloadCsv(`analytics-${localDate(from)}-${localDate(to)}.csv`, [
      ["date", "spend_usd", "gpu_hours", "active_users", ...appCols, "total_activity"],
      ...dailyRows
        .slice()
        .reverse()
        .map((d) => [
          d.date,
          d.spend.toFixed(2),
          d.gpuHours.toFixed(2),
          d.users,
          ...appCols.map((a) => d.perApp[a] ?? 0),
          d.total,
        ]),
    ]);
  };

  const toggle = (set: Set<string>, v: string, update: (s: Set<string>) => void) => {
    const next = new Set(set);
    if (next.has(v)) next.delete(v);
    else next.add(v);
    update(next);
  };

  const fmtUsd = (v: number) =>
    v.toLocaleString("en-US", { style: "currency", currency: "USD" });

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6">
      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-lg border bg-card px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Platform
          </span>
          {PLATFORMS.map((p) => (
            <label key={p.value} className="flex cursor-pointer items-center gap-1.5 text-sm">
              <Checkbox
                checked={platforms.has(p.value)}
                onCheckedChange={() => toggle(platforms, p.value, setPlatforms)}
              />
              {p.label}
            </label>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            App
          </span>
          {APPS.map((a) => (
            <label
              key={a.value}
              className={`flex cursor-pointer items-center gap-1.5 text-sm ${platforms.has(a.platform) ? "" : "opacity-40"}`}
            >
              <Checkbox
                checked={apps.has(a.value)}
                disabled={!platforms.has(a.platform)}
                onCheckedChange={() => toggle(apps, a.value, setApps)}
              />
              {a.label}
            </label>
          ))}
        </div>
        {allSources.length > 0 && (
          <div className="flex w-full flex-wrap items-center gap-3">
            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              GPU source
            </span>
            {allSources.map((s) => (
              <label key={s} className="flex cursor-pointer items-center gap-1.5 text-sm">
                <Checkbox
                  checked={!excludedSources.has(s)}
                  onCheckedChange={() => toggle(excludedSources, s, setExcludedSources)}
                />
                {s}
              </label>
            ))}
          </div>
        )}
        <div className="ml-auto flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={exportCsv} disabled={loading}>
            <Download className="mr-1.5 h-3.5 w-3.5" /> Export CSV
          </Button>
          <Select value={period} onValueChange={(v) => setPeriod(v as Period)}>
            <SelectTrigger className="w-[150px]" size="sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PERIODS.map((p) => (
                <SelectItem key={p.value} value={p.value}>
                  {p.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {slurmState !== "ok" && platforms.has("slurmui") && (
        <p className="rounded-md border border-yellow-500/30 bg-yellow-500/10 px-3 py-2 text-xs text-yellow-600 dark:text-yellow-400">
          {slurmState === "unconfigured"
            ? "SlurmUI is not configured — set SLURMUI_URL and SLURMUI_API_TOKEN on the web server to include Slurm jobs."
            : "SlurmUI could not be reached — Slurm jobs are excluded from this view."}
        </p>
      )}
      {truncated.length > 0 && (
        <p className="rounded-md border border-yellow-500/30 bg-yellow-500/10 px-3 py-2 text-xs text-yellow-600 dark:text-yellow-400">
          Partial data: {truncated.join(", ")} exceeded 5,000 records in this period; totals
          for those kinds are an undercount.
        </p>
      )}

      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border bg-border lg:grid-cols-5">
        {[
          { label: "Total spend", value: fmtUsd(totals.spend), sub: "Selected period (GPU Platform $)" },
          { label: "Daily average", value: fmtUsd(totals.dailyAvg), sub: "Mean daily spend" },
          { label: "GPU hours", value: totals.gpuHours.toFixed(1), sub: "Benchmark / autotrain / compute / Slurm" },
          { label: "Active users", value: String(totals.activeUsers), sub: "Users with activity" },
          { label: "Days tracked", value: String(totals.days), sub: "Days in this period" },
        ].map((c) => (
          <div key={c.label} className="bg-card px-5 py-4">
            <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              {c.label}
            </div>
            <div className="mt-1 text-2xl font-semibold tabular-nums">
              {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : c.value}
            </div>
            <div className="mt-0.5 text-xs text-muted-foreground">{c.sub}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
      {/* Activity by app (donut) */}
      <div className="rounded-lg border bg-card p-4">
        <h2 className="mb-1 text-sm font-semibold">Activity by app</h2>
        <p className="mb-3 text-xs text-muted-foreground">
          Share of jobs / requests per app for the selected period and filters.
        </p>
        <div className="h-64">
          {appPie.length === 0 && !loading ? (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              No analytics data for the selected period.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={appPie}
                  dataKey="value"
                  nameKey="name"
                  innerRadius="55%"
                  outerRadius="85%"
                  paddingAngle={2}
                  strokeWidth={0}
                >
                  {appPie.map((s) => (
                    <Cell key={s.name} fill={s.fill} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(v) => [num(v).toLocaleString(), "records"]}
                  contentStyle={{ fontSize: 12 }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Daily activity chart */}
      <div className="rounded-lg border bg-card p-4">
        <h2 className="mb-1 text-sm font-semibold">Daily activity</h2>
        <p className="mb-3 text-xs text-muted-foreground">
          Jobs / requests per day, by app. Hover for the per-app split and the day&apos;s spend.
        </p>
        <div className="h-64">
          {totals.activity === 0 && !loading ? (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              No analytics data for the selected period.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                <Tooltip
                  formatter={(v, name) => {
                    const n = num(v);
                    return String(name) === "__spend"
                      ? [fmtUsd(n), "Spend"]
                      : [n, APPS.find((a) => a.value === name)?.label ?? String(name)];
                  }}
                  contentStyle={{ fontSize: 12 }}
                />
                <Legend
                  formatter={(v: string) => APPS.find((a) => a.value === v)?.label ?? v}
                  wrapperStyle={{ fontSize: 12 }}
                />
                {activeApps.map((a) => (
                  <Bar key={a.value} dataKey={a.value} stackId="apps" fill={APP_COLORS[a.value]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      </div>

    </div>
  );
}
