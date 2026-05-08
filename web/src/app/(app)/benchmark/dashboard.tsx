"use client";

import { useMemo } from "react";
import yaml from "js-yaml";
import {
  Activity,
  CheckCircle2,
  Clock,
  Layers,
  TrendingUp,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { BenchmarkRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

/** Decorated row used by chart + stats. Pulls out info from result_json + config_yaml
 * once per benchmark so we don't re-parse on every render. */
type Decorated = {
  bench: BenchmarkRecord;
  output_throughput: number | null;
  median_ttft_ms: number | null;
  gpu_type: string | null;
  gpu_count: number | null;
  model: string | null;
  durationS: number | null;
};

function decorate(bench: BenchmarkRecord): Decorated {
  const r = (bench.result_json ?? {}) as Record<string, unknown>;
  let gpu_type: string | null = null;
  let gpu_count: number | null = null;
  let model: string | null = null;
  try {
    const cfg = yaml.load(bench.config_yaml) as
      | { runpod?: { pod?: { gpu_type?: string; gpu_count?: number } }; benchmark?: Array<{ model?: { repo_id?: string } }> }
      | null;
    gpu_type = cfg?.runpod?.pod?.gpu_type ?? null;
    gpu_count = cfg?.runpod?.pod?.gpu_count ?? null;
    model = cfg?.benchmark?.[0]?.model?.repo_id ?? null;
  } catch {
    // ignore — show "—" in the UI
  }
  const start = bench.started_at ? new Date(bench.started_at).getTime() : null;
  const end = bench.ended_at ? new Date(bench.ended_at).getTime() : null;
  return {
    bench,
    output_throughput:
      typeof r.output_throughput === "number" ? r.output_throughput : null,
    median_ttft_ms:
      typeof r.median_ttft_ms === "number" ? r.median_ttft_ms : null,
    gpu_type,
    gpu_count,
    model,
    durationS:
      start != null && end != null ? Math.max(0, Math.round((end - start) / 1000)) : null,
  };
}

function shortGpu(name: string | null): string {
  if (!name) return "—";
  // "NVIDIA GeForce RTX 4090" → "RTX 4090"
  return name
    .replace(/^NVIDIA\s+/i, "")
    .replace(/^GeForce\s+/i, "")
    .replace(/\s+80GB\s+(HBM3|PCIe).*$/i, " 80GB");
}

function shortModel(name: string | null): string {
  if (!name) return "—";
  return name.split("/").pop() ?? name;
}

const BAR_COLOR_BANDS = ["#a78bfa", "#22d3ee", "#34d399", "#fbbf24", "#fb7185"];

export function BenchmarkDashboard({ items }: { items: BenchmarkRecord[] }) {
  const decorated = useMemo(() => items.map(decorate), [items]);

  const total = items.length;
  const done = items.filter((b) => b.status === "done").length;
  const failed = items.filter((b) => b.status === "failed").length;
  const running = items.filter((b) => b.status === "running" || b.status === "queued").length;
  const terminal = done + failed;
  const passRate = terminal > 0 ? (done / terminal) * 100 : null;

  const totalGpuMinutes = decorated.reduce((acc, d) => {
    if (d.durationS == null) return acc;
    const gpus = d.gpu_count ?? 1;
    return acc + (d.durationS * gpus) / 60;
  }, 0);

  const bestRun = decorated
    .filter((d) => d.output_throughput != null)
    .sort((a, b) => (b.output_throughput ?? 0) - (a.output_throughput ?? 0))[0];

  // Leaderboard: top 8 done benchmarks by throughput.
  const leaderboard = useMemo(
    () =>
      decorated
        .filter((d) => d.bench.status === "done" && d.output_throughput != null)
        .sort((a, b) => (b.output_throughput ?? 0) - (a.output_throughput ?? 0))
        .slice(0, 8)
        .map((d, i) => ({
          ...d,
          label: d.bench.name,
          throughput: d.output_throughput!,
          color: BAR_COLOR_BANDS[i % BAR_COLOR_BANDS.length],
        })),
    [decorated],
  );

  if (items.length === 0) return null;

  return (
    <section className="mb-8 space-y-5">
      {/* Stats row */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard
          icon={<Layers className="h-4 w-4" />}
          label="Total runs"
          value={total.toString()}
          sub={
            running > 0
              ? `${running} in flight`
              : `${done} done · ${failed} failed`
          }
          accent="violet"
        />
        <StatCard
          icon={<CheckCircle2 className="h-4 w-4" />}
          label="Pass rate"
          value={passRate != null ? `${passRate.toFixed(0)}%` : "—"}
          sub={terminal > 0 ? `${done}/${terminal} completed` : "no completed runs"}
          accent="emerald"
        />
        <StatCard
          icon={<TrendingUp className="h-4 w-4" />}
          label="Best throughput"
          value={
            bestRun?.output_throughput
              ? `${bestRun.output_throughput.toFixed(0)} tok/s`
              : "—"
          }
          sub={
            bestRun
              ? `${shortModel(bestRun.model)} · ${shortGpu(bestRun.gpu_type)}`
              : "no runs with results yet"
          }
          accent="cyan"
        />
        <StatCard
          icon={<Clock className="h-4 w-4" />}
          label="GPU minutes"
          value={totalGpuMinutes >= 60
            ? `${(totalGpuMinutes / 60).toFixed(1)} h`
            : `${totalGpuMinutes.toFixed(0)} m`}
          sub={`across ${total} run${total === 1 ? "" : "s"}`}
          accent="amber"
        />
      </div>

      {/* Leaderboard chart */}
      {leaderboard.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="flex h-7 w-7 items-center justify-center rounded-md bg-muted text-muted-foreground">
                  <Activity className="h-4 w-4" />
                </div>
                <div>
                  <CardTitle className="text-sm">Throughput leaderboard</CardTitle>
                  <CardDescription className="text-xs">
                    Top {leaderboard.length} benchmarks by output throughput (tokens/sec).
                    Hover a bar for full context.
                  </CardDescription>
                </div>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div
              className="w-full"
              style={{ height: Math.max(180, leaderboard.length * 36 + 32) }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={leaderboard}
                  layout="vertical"
                  margin={{ top: 4, right: 16, left: 0, bottom: 4 }}
                >
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" horizontal={false} />
                  <XAxis
                    type="number"
                    stroke="currentColor"
                    className="text-[10px] text-muted-foreground"
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="label"
                    stroke="currentColor"
                    className="text-[10px] text-muted-foreground"
                    tickLine={false}
                    axisLine={false}
                    width={140}
                  />
                  <Tooltip
                    cursor={{ fill: "rgba(255,255,255,0.04)" }}
                    content={<LeaderboardTooltip />}
                  />
                  <Bar dataKey="throughput" radius={[0, 4, 4, 0]}>
                    {leaderboard.map((d) => (
                      <Cell key={d.bench.id} fill={d.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}
    </section>
  );
}

function LeaderboardTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: Decorated & { throughput: number; label: string } }>;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs shadow-lg">
      <div className="mb-1 font-medium text-zinc-100">{d.label}</div>
      <dl className="space-y-0.5 text-zinc-300">
        <Row label="Throughput" value={`${d.throughput.toFixed(1)} tok/s`} />
        {d.median_ttft_ms != null && (
          <Row label="TTFT" value={`${d.median_ttft_ms.toFixed(1)} ms`} />
        )}
        <Row label="Model" value={shortModel(d.model)} />
        <Row label="GPU" value={`${shortGpu(d.gpu_type)} × ${d.gpu_count ?? 1}`} />
        {d.durationS != null && (
          <Row label="Duration" value={`${d.durationS}s`} />
        )}
      </dl>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 tabular-nums">
      <span className="text-zinc-500">{label}</span>
      <span className="font-mono text-zinc-100">{value}</span>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  sub,
  accent,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
  accent: "violet" | "emerald" | "cyan" | "amber";
}) {
  const ring = {
    violet: "ring-violet-500/20 bg-violet-500/5",
    emerald: "ring-emerald-500/20 bg-emerald-500/5",
    cyan: "ring-cyan-500/20 bg-cyan-500/5",
    amber: "ring-amber-500/20 bg-amber-500/5",
  }[accent];
  const iconBg = {
    violet: "bg-violet-500/15 text-violet-400",
    emerald: "bg-emerald-500/15 text-emerald-400",
    cyan: "bg-cyan-500/15 text-cyan-400",
    amber: "bg-amber-500/15 text-amber-400",
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
      <div className="mt-0.5 text-xs text-muted-foreground">{sub}</div>
    </div>
  );
}
