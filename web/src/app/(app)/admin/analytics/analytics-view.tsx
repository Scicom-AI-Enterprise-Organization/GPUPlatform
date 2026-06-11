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
 *
 * Below the summary, four granular tabs (all driven by the same filtered
 * record set): Jobs explorer, GPU hours by source/model, per-node timeline,
 * and node utilization. Node/GPU columns come from the worker_meta the
 * worker-agent attaches to every result (hostname, gpu_name,
 * CUDA_VISIBLE_DEVICES, …) — records from before that enrichment show "—".
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
import { ChevronDown, Download, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

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
  // Granular fields added in SlurmUI v1.0.213 — absent from older deployments,
  // so all optional and the normalizer degrades to day-granular rows.
  clusterName?: string | null;
  nodeList?: string | null;
  gresDetail?: string | null;
  cudaVisibleDevices?: string | null;
  createdAt?: string;
  endedAt?: string | null;
  durationSec?: number;
  gpus?: number;
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
  id: string;
  name: string | null;
  user: string;
  date: string; // YYYY-MM-DD (local)
  start: Date | null; // null for Slurm (the reports API is day-granular)
  end: Date | null;
  durationS: number | null;
  status: string;
  costUsd: number; // 0 when the source has no $ figure
  gpuHours: number; // 0 when unknown
  // Where it ran — registered provider name (TM-H20, TM-VM1, …), a cloud
  // ("RunPod GPUs", "Prime Intellect GPUs") or a Slurm partition. Free-form:
  // the GPU-source filter is built dynamically from whatever shows up here.
  source: string;
  // Granular placement, from worker_meta / detail (null when not recorded).
  gpuModel: string | null;
  gpuCount: number | null;
  node: string | null; // hostname / machine id / pod id
  devices: string | null; // CUDA_VISIBLE_DEVICES on the node
  raw: Record<string, unknown>; // full record for the detail drawer
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

const APP_LABEL = (v: string) => APPS.find((a) => a.value === v)?.label ?? v;

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

const obj = (x: unknown): Record<string, unknown> =>
  x && typeof x === "object" ? (x as Record<string, unknown>) : {};

// Canonical GPU-source labels. Both platforms name the same physical machine
// differently — GPU Platform by provider name (tm-2-l40s-vm1, TM-H20), Slurm
// by cluster (tm, tm-l40s) or node hostname (scicom-gpu1-<hash>) — so fold
// every raw name onto one label via longest-prefix match (case-insensitive).
// Unmatched names pass through unchanged; extend this list as machines are
// added.
const SOURCE_ALIASES: [prefix: string, label: string][] = [
  // Slurm node hostnames (hash-suffixed, hence prefix match)
  ["scicom-gpu1", "TM-VM1"],
  ["scicom-gpu2", "TM-VM2"],
  ["scicom-ucc", "TM-UCC"],
  // GPU Platform provider names for the same VMs
  ["tm-2-l40s-vm1", "TM-VM1"],
  ["tm-2-l40s-vm2", "TM-VM2"],
  // Slurm cluster names
  ["tm-h20", "TM-H20"],
  ["tm-l40s", "TM-UCC"], // single-node cluster on the UCC machine
  ["primeintellect", "Prime Intellect GPUs"],
  ["prime-intellect", "Prime Intellect GPUs"],
];

function aliasSource(name: string | null): string | null {
  if (!name) return null;
  const n = name.toLowerCase();
  let best: string | null = null;
  let bestLen = -1;
  for (const [prefix, label] of SOURCE_ALIASES) {
    if (n.startsWith(prefix) && prefix.length > bestLen) {
      best = label;
      bestLen = prefix.length;
    }
  }
  return best;
}

// "Where did this run" label, best-effort from the history detail blob:
// a registered provider's name wins (TM-H20, TM-VM1, …); otherwise the cloud
// kind; for serverless/proxy requests fall back to the serving worker's GPU
// or the endpoint's configured GPU.
function gpuSource(detail: Record<string, unknown>): string {
  const provName = str(detail?.provider_name);
  if (provName) return aliasSource(provName) ?? provName;
  const kind = str(detail?.provider_kind) ?? str(detail?.backend);
  if (kind === "pi") return "Prime Intellect GPUs";
  if (kind === "runpod") return "RunPod GPUs";
  if (kind === "external") return "External";
  const worker = obj(detail?.worker);
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
      const start = new Date(j.created_at);
      const durS = j.duration_s ?? null;
      const end = j.ended_at
        ? new Date(j.ended_at)
        : durS != null
          ? new Date(start.getTime() + durS * 1000)
          : null;
      const durH = (durS ?? 0) / 3600;
      const detail = j.detail ?? {};
      const worker = obj(detail.worker);
      const costPerHr = num(detail.cost_per_hr);
      const gpuCount =
        num(worker.gpu_count) || num(detail.gpu_count) || num(detail.requested_gpu_count) || 0;
      recs.push({
        platform: "gpuplatform",
        app,
        id: j.id,
        name: j.name,
        user: j.user,
        date: localDate(start),
        start,
        end,
        durationS: durS,
        status: j.status,
        costUsd: costPerHr * durH,
        // Inference/proxy requests are seconds-long API calls, not GPU
        // reservations — only the discrete job kinds count toward GPU-hours.
        gpuHours:
          app === "serverless" || app === "proxy" ? 0 : durH * (gpuCount || 1),
        source: gpuSource(detail),
        gpuModel:
          str(worker.gpu_name) ?? str(detail.gpu_type) ?? str(detail.requested_gpu_type),
        gpuCount: gpuCount || null,
        node:
          str(worker.hostname) ??
          str(worker.machine_id) ??
          str(detail.machine_id) ??
          str(worker.runpod_pod_id) ??
          str(detail.runpod_pod_id) ??
          str(detail.pod_id),
        devices: str(worker.visible_devices) ?? str(detail.visible_devices),
        raw: j as unknown as Record<string, unknown>,
      });
    }
  }
  return recs;
}

function normalizeSlurm(report: SlurmReport): Rec[] {
  const recs: Rec[] = [];
  for (const day of report.dailyJobHistory ?? []) {
    const jobs = day.jobs ?? [];
    // Older SlurmUI deployments report GPU-hours per-day only — spread evenly
    // so user totals still sum to the true daily figure. Newer ones carry
    // per-job durationSec + gpus, which the same day total is computed from.
    const perJobGpuH = jobs.length ? num(day.gpuHours) / jobs.length : 0;
    for (const j of jobs) {
      const start = j.createdAt ? new Date(j.createdAt) : null;
      const durS = typeof j.durationSec === "number" ? j.durationSec : null;
      recs.push({
        platform: "slurmui",
        app: "slurmjob",
        id: j.slurmJobId != null ? String(j.slurmJobId) : "—",
        name: j.jobName,
        user: j.unixUsername ?? "(unknown)",
        date: day.date,
        start,
        end: j.endedAt
          ? new Date(j.endedAt)
          : start && durS != null
            ? new Date(start.getTime() + durS * 1000)
            : null,
        durationS: durS,
        status: (j.state ?? "").toLowerCase(),
        costUsd: 0,
        gpuHours:
          durS != null && typeof j.gpus === "number"
            ? (durS / 3600) * j.gpus
            : perJobGpuH,
        // Attribute to the machine, not the Slurm cluster: alias the first
        // node in nodeList (a Slurm "tm" cluster spans TM-VM1 + TM-VM2), then
        // the cluster name, so activity aggregates with GPU Platform jobs on
        // the same hardware.
        source:
          aliasSource(str(j.nodeList)?.split(",")[0] ?? null) ??
          aliasSource(str(j.clusterName)) ??
          str(j.clusterName) ??
          (j.partition ? `Slurm · ${j.partition}` : "Slurm"),
        gpuModel: str(j.gresDetail),
        gpuCount: typeof j.gpus === "number" && j.gpus > 0 ? j.gpus : null,
        node: str(j.nodeList),
        devices: str(j.cudaVisibleDevices),
        raw: j as unknown as Record<string, unknown>,
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

// ── multi-select dropdown filter ─────────────────────────────────────────────

function MultiFilter({
  label,
  options,
  selected, // values currently INCLUDED
  onChange,
}: {
  label: string;
  options: { value: string; label: string; disabled?: boolean }[];
  selected: Set<string>;
  onChange: (next: Set<string>) => void;
}) {
  const enabled = options.filter((o) => !o.disabled);
  const checkedCount = enabled.filter((o) => selected.has(o.value)).length;
  const summary =
    checkedCount === enabled.length
      ? "All"
      : checkedCount === 0
        ? "None"
        : checkedCount === 1
          ? enabled.find((o) => selected.has(o.value))?.label
          : `${checkedCount} selected`;
  const setAll = (on: boolean) => {
    const next = new Set(selected);
    for (const o of enabled) {
      if (on) next.add(o.value);
      else next.delete(o.value);
    }
    onChange(next);
  };
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="h-8 gap-1.5 font-normal">
          <span className="text-muted-foreground">{label}:</span>
          <span className="max-w-[10rem] truncate font-medium">{summary}</span>
          <ChevronDown className="h-3.5 w-3.5 opacity-60" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="max-h-80 w-56 overflow-y-auto">
        <div className="flex items-center justify-between px-2 py-1.5 text-xs">
          <button className="text-primary hover:underline" onClick={() => setAll(true)}>
            Select all
          </button>
          <button className="text-primary hover:underline" onClick={() => setAll(false)}>
            Clear
          </button>
        </div>
        <DropdownMenuSeparator />
        {options.map((o) => (
          <DropdownMenuCheckboxItem
            key={o.value}
            checked={selected.has(o.value)}
            disabled={o.disabled}
            // keep the menu open so several boxes can be ticked in one go
            onSelect={(e) => e.preventDefault()}
            onCheckedChange={(on) => {
              const next = new Set(selected);
              if (on) next.add(o.value);
              else next.delete(o.value);
              onChange(next);
            }}
          >
            {o.label}
          </DropdownMenuCheckboxItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

// ── helpers for granular views ───────────────────────────────────────────────

const fmtDur = (s: number | null): string => {
  if (s == null) return "—";
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
};

const fmtTime = (d: Date | null): string =>
  d
    ? `${localDate(d)} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`
    : "—";

const STATUS_COLOR = (s: string) =>
  /complet|succe|stopped/.test(s)
    ? "text-emerald-600 dark:text-emerald-400"
    : /fail|error|timeout|cancel/.test(s)
      ? "text-red-600 dark:text-red-400"
      : "text-muted-foreground";

type SortKey = "start" | "app" | "user" | "source" | "gpu" | "node" | "duration" | "status";

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

  // Jobs explorer state
  const [sortKey, setSortKey] = useState<SortKey>("start");
  const [sortAsc, setSortAsc] = useState(false);
  const [page, setPage] = useState(0);
  const [detailRec, setDetailRec] = useState<Rec | null>(null);
  const PAGE_SIZE = 50;

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

  // SlurmUI cluster names are lowercase ("tm-h20") while GPU Platform provider
  // names are uppercase ("TM-H20") for the same hardware — fold Slurm sources
  // onto the GPU Platform label case-insensitively so each machine is one
  // filter entry.
  const mergedSlurmRecs = useMemo(() => {
    const byLower = new Map(gpuRecs.map((r) => [r.source.toLowerCase(), r.source]));
    return slurmRecs.map((r) => {
      const canon = byLower.get(r.source.toLowerCase());
      return canon && canon !== r.source ? { ...r, source: canon } : r;
    });
  }, [gpuRecs, slurmRecs]);

  const allSources = useMemo(
    () => [...new Set([...gpuRecs, ...mergedSlurmRecs].map((r) => r.source))].sort(),
    [gpuRecs, mergedSlurmRecs],
  );

  const sourceSelected = useMemo(
    () => new Set(allSources.filter((s) => !excludedSources.has(s))),
    [allSources, excludedSources],
  );

  const recs = useMemo(
    () =>
      [...gpuRecs, ...mergedSlurmRecs].filter(
        (r) =>
          platforms.has(r.platform) &&
          apps.has(r.app) &&
          !excludedSources.has(r.source),
      ),
    [gpuRecs, mergedSlurmRecs, platforms, apps, excludedSources],
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

  // Feeds the CSV export.
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

  // Jobs explorer: sorted + paged
  const sortedRecs = useMemo(() => {
    const cmp = (a: Rec, b: Rec): number => {
      const dir = sortAsc ? 1 : -1;
      switch (sortKey) {
        case "start":
          return dir * ((a.start?.getTime() ?? 0) - (b.start?.getTime() ?? 0));
        case "duration":
          return dir * ((a.durationS ?? -1) - (b.durationS ?? -1));
        case "app":
          return dir * a.app.localeCompare(b.app);
        case "user":
          return dir * a.user.localeCompare(b.user);
        case "source":
          return dir * a.source.localeCompare(b.source);
        case "gpu":
          return dir * (a.gpuModel ?? "").localeCompare(b.gpuModel ?? "");
        case "node":
          return dir * (a.node ?? "").localeCompare(b.node ?? "");
        case "status":
          return dir * a.status.localeCompare(b.status);
      }
    };
    return [...recs].sort(cmp);
  }, [recs, sortKey, sortAsc]);

  const pageCount = Math.max(1, Math.ceil(sortedRecs.length / PAGE_SIZE));
  // Clamp rather than reset-in-effect: filter/sort changes can shrink the list.
  const safePage = Math.min(page, pageCount - 1);
  const pageRecs = sortedRecs.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE);

  // GPU hours grouped by source and by GPU model
  const gpuHoursBySource = useMemo(() => {
    const m = new Map<string, { gpuHours: number; jobs: number; spend: number }>();
    for (const r of recs) {
      const e = m.get(r.source) ?? { gpuHours: 0, jobs: 0, spend: 0 };
      e.gpuHours += r.gpuHours;
      e.jobs += 1;
      e.spend += r.costUsd;
      m.set(r.source, e);
    }
    return [...m.entries()]
      .map(([source, v]) => ({ source, ...v }))
      .sort((a, b) => b.gpuHours - a.gpuHours);
  }, [recs]);

  const gpuHoursByModel = useMemo(() => {
    const m = new Map<string, { gpuHours: number; jobs: number }>();
    for (const r of recs) {
      const key = r.gpuModel ?? "(not recorded)";
      const e = m.get(key) ?? { gpuHours: 0, jobs: 0 };
      e.gpuHours += r.gpuHours;
      e.jobs += 1;
      m.set(key, e);
    }
    return [...m.entries()]
      .map(([model, v]) => ({ model, ...v }))
      .sort((a, b) => b.gpuHours - a.gpuHours);
  }, [recs]);

  // Node timeline: only records with real timestamps and a known node.
  const timelineNodes = useMemo(() => {
    const byNode = new Map<string, Rec[]>();
    for (const r of recs) {
      if (!r.start || !r.node) continue;
      const list = byNode.get(r.node) ?? [];
      list.push(r);
      byNode.set(r.node, list);
    }
    return [...byNode.entries()]
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, 14);
  }, [recs]);

  // Node utilization rollup
  const nodeRows = useMemo(() => {
    const periodH = Math.max((to.getTime() - from.getTime()) / 3600_000, 1e-9);
    const m = new Map<
      string,
      { jobs: number; busyS: number; gpu: string | null; sources: Set<string>; lastSeen: Date | null }
    >();
    for (const r of recs) {
      if (!r.node) continue;
      const e =
        m.get(r.node) ?? { jobs: 0, busyS: 0, gpu: null, sources: new Set<string>(), lastSeen: null };
      e.jobs += 1;
      e.busyS += r.durationS ?? 0;
      if (r.gpuModel) e.gpu = r.gpuModel;
      e.sources.add(r.source);
      const seen = r.end ?? r.start;
      if (seen && (!e.lastSeen || seen > e.lastSeen)) e.lastSeen = seen;
      m.set(r.node, e);
    }
    return [...m.entries()]
      .map(([node, v]) => ({
        node,
        jobs: v.jobs,
        busyH: v.busyS / 3600,
        busyPct: Math.min((v.busyS / 3600 / periodH) * 100, 100),
        gpu: v.gpu,
        sources: [...v.sources].join(", "),
        lastSeen: v.lastSeen,
      }))
      .sort((a, b) => b.busyH - a.busyH);
  }, [recs, from, to]);

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

  const exportJobsCsv = () => {
    downloadCsv(`analytics-jobs-${localDate(from)}-${localDate(to)}.csv`, [
      ["start", "end", "duration_s", "platform", "app", "id", "name", "user",
        "status", "gpu_source", "gpu_model", "gpu_count", "node", "cuda_visible_devices",
        "gpu_hours", "cost_usd"],
      ...sortedRecs.map((r) => [
        fmtTime(r.start),
        fmtTime(r.end),
        r.durationS ?? "",
        r.platform,
        r.app,
        r.id,
        r.name ?? "",
        r.user,
        r.status,
        r.source,
        r.gpuModel ?? "",
        r.gpuCount ?? "",
        r.node ?? "",
        r.devices ?? "",
        r.gpuHours.toFixed(3),
        r.costUsd.toFixed(4),
      ]),
    ]);
  };

  const fmtUsd = (v: number) =>
    v.toLocaleString("en-US", { style: "currency", currency: "USD" });

  const sortHeader = (key: SortKey, label: string) => (
    <th
      className="cursor-pointer select-none whitespace-nowrap px-3 py-2 text-left font-medium hover:text-foreground"
      onClick={() => {
        if (sortKey === key) setSortAsc(!sortAsc);
        else {
          setSortKey(key);
          setSortAsc(key !== "start" && key !== "duration");
        }
      }}
    >
      {label}
      {sortKey === key ? (sortAsc ? " ↑" : " ↓") : ""}
    </th>
  );

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6">
      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-2 rounded-lg border bg-card px-4 py-3">
        <MultiFilter
          label="Platform"
          options={PLATFORMS.map((p) => ({ value: p.value, label: p.label }))}
          selected={platforms}
          onChange={setPlatforms}
        />
        <MultiFilter
          label="App"
          options={APPS.map((a) => ({
            value: a.value,
            label: a.label,
            disabled: !platforms.has(a.platform),
          }))}
          selected={apps}
          onChange={setApps}
        />
        {allSources.length > 0 && (
          <MultiFilter
            label="GPU source"
            options={allSources.map((s) => ({ value: s, label: s }))}
            selected={sourceSelected}
            onChange={(next) =>
              setExcludedSources(new Set(allSources.filter((s) => !next.has(s))))
            }
          />
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
                      : [n, APP_LABEL(String(name))];
                  }}
                  contentStyle={{ fontSize: 12 }}
                />
                <Legend formatter={(v: string) => APP_LABEL(v)} wrapperStyle={{ fontSize: 12 }} />
                {activeApps.map((a) => (
                  <Bar key={a.value} dataKey={a.value} stackId="apps" fill={APP_COLORS[a.value]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      </div>

      {/* ── Granular views ─────────────────────────────────────────────────── */}
      <Tabs defaultValue="jobs">
        <TabsList>
          <TabsTrigger value="jobs">Jobs</TabsTrigger>
          <TabsTrigger value="gpuhours">GPU hours</TabsTrigger>
          <TabsTrigger value="timeline">Node timeline</TabsTrigger>
          <TabsTrigger value="nodes">Nodes</TabsTrigger>
        </TabsList>

        {/* Jobs explorer */}
        <TabsContent value="jobs">
          <div className="rounded-lg border bg-card">
            <div className="flex items-center justify-between px-4 py-3">
              <div>
                <h2 className="text-sm font-semibold">Jobs explorer</h2>
                <p className="text-xs text-muted-foreground">
                  Every job / request in the period — which GPU it used, on which node, when and
                  for how long. Click a row for the full record. Records from before node
                  reporting shipped show &ldquo;—&rdquo;.
                </p>
              </div>
              <Button variant="outline" size="sm" onClick={exportJobsCsv} disabled={loading}>
                <Download className="mr-1.5 h-3.5 w-3.5" /> Export jobs CSV
              </Button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="border-y bg-muted/40 text-muted-foreground">
                  <tr>
                    {sortHeader("start", "Start")}
                    {sortHeader("app", "App")}
                    {sortHeader("user", "User")}
                    {sortHeader("source", "GPU source")}
                    {sortHeader("gpu", "GPU")}
                    {sortHeader("node", "Node")}
                    <th className="px-3 py-2 text-left font-medium">Devices</th>
                    {sortHeader("duration", "Duration")}
                    {sortHeader("status", "Status")}
                  </tr>
                </thead>
                <tbody>
                  {pageRecs.map((r, i) => (
                    <tr
                      key={`${r.platform}-${r.id}-${i}`}
                      className="cursor-pointer border-b last:border-0 hover:bg-muted/30"
                      onClick={() => setDetailRec(r)}
                    >
                      <td className="whitespace-nowrap px-3 py-2 font-mono tabular-nums">
                        {r.start ? fmtTime(r.start) : r.date}
                      </td>
                      <td className="whitespace-nowrap px-3 py-2">
                        <span
                          className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle"
                          style={{ background: APP_COLORS[r.app] ?? "#999" }}
                        />
                        {APP_LABEL(r.app)}
                      </td>
                      <td className="max-w-[10rem] truncate px-3 py-2">{r.user}</td>
                      <td className="whitespace-nowrap px-3 py-2">{r.source}</td>
                      <td className="whitespace-nowrap px-3 py-2">
                        {r.gpuModel ? `${r.gpuModel}${r.gpuCount ? ` ×${r.gpuCount}` : ""}` : "—"}
                      </td>
                      <td className="max-w-[12rem] truncate px-3 py-2 font-mono">{r.node ?? "—"}</td>
                      <td className="whitespace-nowrap px-3 py-2 font-mono">
                        {r.devices != null ? `GPU ${r.devices}` : "—"}
                      </td>
                      <td className="whitespace-nowrap px-3 py-2 tabular-nums">{fmtDur(r.durationS)}</td>
                      <td className={`whitespace-nowrap px-3 py-2 ${STATUS_COLOR(r.status)}`}>
                        {r.status}
                      </td>
                    </tr>
                  ))}
                  {pageRecs.length === 0 && !loading && (
                    <tr>
                      <td colSpan={9} className="px-3 py-8 text-center text-muted-foreground">
                        No records match the current filters.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="flex items-center justify-between border-t px-4 py-2 text-xs text-muted-foreground">
              <span>
                {sortedRecs.length.toLocaleString()} records
                {sortedRecs.length > PAGE_SIZE
                  ? ` — page ${safePage + 1} of ${pageCount}`
                  : ""}
              </span>
              {pageCount > 1 && (
                <span className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={safePage === 0}
                    onClick={() => setPage(safePage - 1)}
                  >
                    Prev
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={safePage >= pageCount - 1}
                    onClick={() => setPage(safePage + 1)}
                  >
                    Next
                  </Button>
                </span>
              )}
            </div>
          </div>
        </TabsContent>

        {/* GPU hours by source + model */}
        <TabsContent value="gpuhours">
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-lg border bg-card p-4">
              <h2 className="mb-1 text-sm font-semibold">GPU hours by source</h2>
              <p className="mb-3 text-xs text-muted-foreground">
                duration × GPU count, per provider / cloud / partition (job kinds only — API
                requests don&apos;t reserve GPUs).
              </p>
              <div className="h-64">
                {gpuHoursBySource.every((s) => s.gpuHours === 0) && !loading ? (
                  <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                    No GPU-hours in the selected period.
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={gpuHoursBySource.filter((s) => s.gpuHours > 0)}
                      layout="vertical"
                      margin={{ left: 8, right: 16 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.1} />
                      <XAxis type="number" tick={{ fontSize: 11 }} />
                      <YAxis
                        type="category"
                        dataKey="source"
                        width={140}
                        tick={{ fontSize: 11 }}
                      />
                      <Tooltip
                        formatter={(v) => [`${num(v).toFixed(2)} h`, "GPU hours"]}
                        contentStyle={{ fontSize: 12 }}
                      />
                      <Bar dataKey="gpuHours" fill="#60a5fa" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>
            <div className="rounded-lg border bg-card p-4">
              <h2 className="mb-1 text-sm font-semibold">By GPU model</h2>
              <p className="mb-3 text-xs text-muted-foreground">
                As reported by nvidia-smi on the serving node (or the requested type).
              </p>
              <table className="w-full text-xs">
                <thead className="border-y bg-muted/40 text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">GPU model</th>
                    <th className="px-3 py-2 text-right font-medium">Jobs / requests</th>
                    <th className="px-3 py-2 text-right font-medium">GPU hours</th>
                  </tr>
                </thead>
                <tbody>
                  {gpuHoursByModel.map((m) => (
                    <tr key={m.model} className="border-b last:border-0">
                      <td className="px-3 py-2">{m.model}</td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {m.jobs.toLocaleString()}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">
                        {m.gpuHours.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                  {gpuHoursByModel.length === 0 && !loading && (
                    <tr>
                      <td colSpan={3} className="px-3 py-8 text-center text-muted-foreground">
                        No records match the current filters.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </TabsContent>

        {/* Node timeline */}
        <TabsContent value="timeline">
          <div className="rounded-lg border bg-card p-4">
            <h2 className="mb-1 text-sm font-semibold">Node timeline</h2>
            <p className="mb-3 text-xs text-muted-foreground">
              What ran on each node across the period — one bar per job, colored by app. Hover
              for details, click for the full record. Busiest 14 nodes shown; records without
              node attribution (old jobs, older SlurmUI versions) are excluded.
            </p>
            {timelineNodes.length === 0 && !loading ? (
              <div className="flex h-32 items-center justify-center text-sm text-muted-foreground">
                No node-attributed records in the selected period.
              </div>
            ) : (
              <div className="space-y-1.5">
                {timelineNodes.map(([node, jobs]) => {
                  const t0 = from.getTime();
                  const span = Math.max(to.getTime() - t0, 1);
                  return (
                    <div key={node} className="flex items-center gap-3">
                      <div
                        className="w-44 shrink-0 truncate text-right font-mono text-[11px] text-muted-foreground"
                        title={node}
                      >
                        {node}
                      </div>
                      <div className="relative h-6 flex-1 overflow-hidden rounded bg-muted/40">
                        {jobs.map((r, i) => {
                          const s = r.start!.getTime();
                          const e = Math.min(
                            (r.end ?? r.start!).getTime(),
                            to.getTime(),
                          );
                          const left = Math.max(((s - t0) / span) * 100, 0);
                          const width = Math.max(((e - s) / span) * 100, 0.25);
                          return (
                            <div
                              key={i}
                              className="absolute top-0.5 bottom-0.5 cursor-pointer rounded-sm opacity-80 hover:opacity-100"
                              style={{
                                left: `${left}%`,
                                width: `${Math.min(width, 100 - left)}%`,
                                background: APP_COLORS[r.app] ?? "#999",
                              }}
                              title={`${APP_LABEL(r.app)} · ${r.user}\n${fmtTime(r.start)} → ${fmtTime(r.end)} (${fmtDur(r.durationS)})\nGPU: ${r.gpuModel ?? "?"}${r.devices != null ? ` (devices ${r.devices})` : ""}\nstatus: ${r.status}`}
                              onClick={() => setDetailRec(r)}
                            />
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
                <div className="flex items-center gap-3 pt-1">
                  <div className="w-44 shrink-0" />
                  <div className="flex flex-1 justify-between text-[10px] text-muted-foreground">
                    <span>{fmtTime(from)}</span>
                    <span>{fmtTime(to)}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </TabsContent>

        {/* Node utilization */}
        <TabsContent value="nodes">
          <div className="rounded-lg border bg-card">
            <div className="px-4 py-3">
              <h2 className="text-sm font-semibold">Node utilization</h2>
              <p className="text-xs text-muted-foreground">
                Per node: how many jobs it served and how busy it was over the period (sum of
                job durations ÷ period length — parallel jobs can exceed 100%, capped here).
              </p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="border-y bg-muted/40 text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">Node</th>
                    <th className="px-3 py-2 text-left font-medium">GPU</th>
                    <th className="px-3 py-2 text-left font-medium">Source</th>
                    <th className="px-3 py-2 text-right font-medium">Jobs</th>
                    <th className="px-3 py-2 text-right font-medium">Busy hours</th>
                    <th className="px-3 py-2 text-left font-medium">Busy %</th>
                    <th className="px-3 py-2 text-left font-medium">Last seen</th>
                  </tr>
                </thead>
                <tbody>
                  {nodeRows.map((n) => (
                    <tr key={n.node} className="border-b last:border-0">
                      <td className="max-w-[14rem] truncate px-3 py-2 font-mono">{n.node}</td>
                      <td className="whitespace-nowrap px-3 py-2">{n.gpu ?? "—"}</td>
                      <td className="whitespace-nowrap px-3 py-2">{n.sources}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{n.jobs}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{n.busyH.toFixed(2)}</td>
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-24 overflow-hidden rounded bg-muted">
                            <div
                              className="h-full rounded bg-blue-500"
                              style={{ width: `${n.busyPct}%` }}
                            />
                          </div>
                          <span className="tabular-nums">{n.busyPct.toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="whitespace-nowrap px-3 py-2 tabular-nums">
                        {fmtTime(n.lastSeen)}
                      </td>
                    </tr>
                  ))}
                  {nodeRows.length === 0 && !loading && (
                    <tr>
                      <td colSpan={7} className="px-3 py-8 text-center text-muted-foreground">
                        No node-attributed records in the selected period.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </TabsContent>
      </Tabs>

      {/* Record detail drawer */}
      <Dialog open={detailRec !== null} onOpenChange={(o) => !o && setDetailRec(null)}>
        <DialogContent className="max-h-[85vh] max-w-2xl overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-sm">
              {detailRec ? `${APP_LABEL(detailRec.app)} · ${detailRec.id}` : ""}
            </DialogTitle>
          </DialogHeader>
          {detailRec && (
            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-x-6 gap-y-1.5">
                {[
                  ["User", detailRec.user],
                  ["Status", detailRec.status],
                  ["Start", fmtTime(detailRec.start) === "—" ? detailRec.date : fmtTime(detailRec.start)],
                  ["End", fmtTime(detailRec.end)],
                  ["Duration", fmtDur(detailRec.durationS)],
                  ["GPU source", detailRec.source],
                  ["GPU", detailRec.gpuModel ? `${detailRec.gpuModel}${detailRec.gpuCount ? ` ×${detailRec.gpuCount}` : ""}` : "—"],
                  ["Node", detailRec.node ?? "—"],
                  ["CUDA devices", detailRec.devices ?? "—"],
                  ["GPU hours", detailRec.gpuHours.toFixed(3)],
                  ["Cost", fmtUsd(detailRec.costUsd)],
                ].map(([k, v]) => (
                  <div key={k} className="contents">
                    <span className="text-muted-foreground">{k}</span>
                    <span className="font-mono">{v}</span>
                  </div>
                ))}
              </div>
              <div>
                <div className="mb-1 font-medium text-muted-foreground">Full record</div>
                <pre className="max-h-72 overflow-auto rounded bg-muted/40 p-3 font-mono text-[11px] leading-relaxed">
                  {JSON.stringify(detailRec.raw, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
