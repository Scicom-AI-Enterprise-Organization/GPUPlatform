"use client";

import { CheckCircle2, Clock, Layers, TrendingUp } from "lucide-react";
import { shortGpu as formatGpu } from "@/lib/gpu-format";
import type { BenchStat } from "@/lib/types";

function shortGpu(name: string | null): string {
  return formatGpu(name) || "—";
}

function shortModel(name: string | null): string {
  if (!name) return "—";
  return name.split("/").pop() ?? name;
}

/** KPI row over the slim /benchmarks/_stats payload — one row per run, no
 * config YAML or result JSON to parse client-side. */
export function BenchmarkDashboard({ stats }: { stats: BenchStat[] }) {
  const total = stats.length;
  const done = stats.filter((s) => s.status === "done").length;
  const failed = stats.filter((s) => s.status === "failed").length;
  const running = stats.filter((s) => s.status === "running" || s.status === "queued").length;
  const terminal = done + failed;
  const passRate = terminal > 0 ? (done / terminal) * 100 : null;

  const totalGpuMinutes = stats.reduce((acc, s) => {
    if (s.duration_s == null) return acc;
    const gpus = s.gpu_count ?? 1;
    return acc + (s.duration_s * gpus) / 60;
  }, 0);

  const bestRun = stats
    .filter((s) => s.output_throughput != null)
    .sort((a, b) => (b.output_throughput ?? 0) - (a.output_throughput ?? 0))[0];

  if (stats.length === 0) return null;

  return (
    <section className="mb-8 space-y-5">
      {/* Stats row — all neutral. KPI cards aren't status; they're just numbers. */}
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
        />
        <StatCard
          icon={<CheckCircle2 className="h-4 w-4" />}
          label="Pass rate"
          value={passRate != null ? `${passRate.toFixed(0)}%` : "—"}
          sub={terminal > 0 ? `${done}/${terminal} completed` : "no completed runs"}
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
        />
        <StatCard
          icon={<Clock className="h-4 w-4" />}
          label="GPU minutes"
          value={totalGpuMinutes >= 60
            ? `${(totalGpuMinutes / 60).toFixed(1)} h`
            : `${totalGpuMinutes.toFixed(0)} m`}
          sub={`across ${total} run${total === 1 ? "" : "s"}`}
        />
      </div>

    </section>
  );
}

function StatCard({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
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
      <div className="mt-0.5 text-xs text-muted-foreground">{sub}</div>
    </div>
  );
}
