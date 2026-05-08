"use client";

import Link from "next/link";
import yaml from "js-yaml";
import { Clock, Cpu, Layers, TrendingUp, User } from "lucide-react";
import type { BenchmarkRecord } from "@/lib/types";
import { avatarFor } from "@/lib/avatar";
import { cn } from "@/lib/utils";

const STATUS_STYLES: Record<string, string> = {
  queued: "bg-muted text-muted-foreground",
  running: "bg-blue-500/15 text-blue-400 ring-1 ring-blue-500/30",
  done: "bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30",
  failed: "bg-destructive/15 text-destructive ring-1 ring-destructive/30",
  cancelled: "bg-muted text-muted-foreground",
};

function shortGpu(s: string | null | undefined): string {
  if (!s) return "—";
  return s
    .replace(/^NVIDIA\s+/i, "")
    .replace(/^GeForce\s+/i, "")
    .replace(/\s+80GB\s+(HBM3|PCIe).*$/i, " 80GB");
}

function shortModel(s: string | null | undefined): string {
  if (!s) return "—";
  return s.split("/").pop() ?? s;
}

function fmtTput(v: number | null | undefined): string | null {
  if (v == null || !Number.isFinite(v)) return null;
  if (v >= 1000) return `${(v / 1000).toFixed(1)}k tok/s`;
  return `${v.toFixed(0)} tok/s`;
}

function fmtDuration(secs: number | null): string | null {
  if (secs == null) return null;
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function BenchmarkRow({ bench }: { bench: BenchmarkRecord }) {
  const avatar = avatarFor(bench.name);
  const result = (bench.result_json ?? {}) as Record<string, unknown>;
  const tput = typeof result.output_throughput === "number" ? result.output_throughput : null;

  let model: string | null = null;
  let gpu: string | null = null;
  let gpuCount = 1;
  let parallelism: string | null = null;
  let benchCount = 0;
  try {
    const cfg = yaml.load(bench.config_yaml) as
      | {
          runpod?: { pod?: { gpu_type?: string; gpu_count?: number } };
          benchmark?: Array<{
            model?: { repo_id?: string };
            serve?: { tensor_parallel_size?: number; data_parallel_size?: number };
            bench?: unknown[];
          }>;
        }
      | null;
    gpu = cfg?.runpod?.pod?.gpu_type ?? null;
    gpuCount = cfg?.runpod?.pod?.gpu_count ?? 1;
    const first = cfg?.benchmark?.[0];
    model = first?.model?.repo_id ?? null;
    const tp = first?.serve?.tensor_parallel_size ?? 1;
    const dp = first?.serve?.data_parallel_size ?? 1;
    if (tp > 1 || dp > 1) parallelism = `TP${tp}/DP${dp}`;
    benchCount = Array.isArray(first?.bench) ? first.bench.length : 0;
  } catch {
    // ignore — show placeholders
  }

  const dur = (() => {
    if (!bench.started_at || !bench.ended_at) return null;
    const s = new Date(bench.started_at).getTime();
    const e = new Date(bench.ended_at).getTime();
    return Math.max(0, Math.round((e - s) / 1000));
  })();

  return (
    <Link
      href={`/benchmark/${encodeURIComponent(bench.id)}`}
      className="group block rounded-xl border border-border bg-card p-4 transition-all hover:border-primary/40 hover:bg-card/80 hover:shadow-md"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <div
            className={cn(
              "flex h-11 w-11 shrink-0 items-center justify-center rounded-lg text-base font-semibold",
              avatar.bg,
              avatar.text,
            )}
          >
            {avatar.letter}
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="truncate font-medium text-foreground">{bench.name}</span>
              <span
                className={cn(
                  "rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide",
                  STATUS_STYLES[bench.status] ?? STATUS_STYLES.queued,
                )}
              >
                {bench.status}
              </span>
            </div>
            <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="truncate font-mono" title={bench.id}>{bench.id}</span>
              <span>·</span>
              <User className="h-3 w-3" />
              <span className="truncate">{bench.created_by}</span>
            </div>
          </div>
        </div>

        {tput != null && (
          <div className="shrink-0 rounded-md bg-emerald-500/10 px-2.5 py-1 text-right ring-1 ring-emerald-500/20">
            <div className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-emerald-400/80">
              <TrendingUp className="h-3 w-3" /> Throughput
            </div>
            <div className="font-mono text-sm font-semibold tabular-nums text-emerald-400">
              {fmtTput(tput)}
            </div>
          </div>
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        {model && (
          <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs">
            <span className="font-mono">{shortModel(model)}</span>
          </span>
        )}
        {gpu && (
          <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs">
            <Cpu className="h-3 w-3 text-muted-foreground" />
            <span className="font-mono">
              {shortGpu(gpu)}
              {gpuCount > 1 ? ` × ${gpuCount}` : ""}
            </span>
          </span>
        )}
        {parallelism && (
          <span className="inline-flex items-center gap-1 rounded-md bg-violet-500/10 px-2 py-0.5 font-mono text-xs text-violet-400 ring-1 ring-violet-500/20">
            {parallelism}
          </span>
        )}
        {benchCount > 1 && (
          <span className="inline-flex items-center gap-1 rounded-md bg-fuchsia-500/10 px-2 py-0.5 text-xs text-fuchsia-400 ring-1 ring-fuchsia-500/20">
            <Layers className="h-3 w-3" />
            sweep · {benchCount} cells
          </span>
        )}
      </div>

      <div className="mt-3 flex items-center justify-between border-t border-border/60 pt-2 text-xs text-muted-foreground">
        <div className="flex items-center gap-3">
          {dur != null && (
            <span className="inline-flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {fmtDuration(dur)}
            </span>
          )}
          {bench.exit_code != null && bench.exit_code !== 0 && (
            <span className="font-mono text-destructive">exit {bench.exit_code}</span>
          )}
        </div>
        <span title={new Date(bench.created_at).toISOString()}>
          {new Date(bench.created_at).toLocaleString()}
        </span>
      </div>
    </Link>
  );
}
