"use client";

import Link from "next/link";
import yaml from "js-yaml";
import { Clock, Cpu, Globe, Layers, Lock, MoreHorizontal, Pencil, Trash2, TrendingUp, User } from "lucide-react";
import type { BenchmarkRecord } from "@/lib/types";
import { avatarFor } from "@/lib/avatar";
import { formatCostUSD, useLiveCost } from "@/lib/cost";
import { BurnFlame } from "@/components/burn-flame";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { shortGpu as formatGpu } from "@/lib/gpu-format";
import { cn } from "@/lib/utils";

// Status pill is the only place this row uses colour. Pattern matches Compute:
// soft tint + matching text + neutral border.
const STATUS_STYLES: Record<string, string> = {
  queued: "border border-border bg-muted text-muted-foreground",
  running: "border border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
  done: "border border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  failed: "border border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
  cancelled: "border border-border bg-muted text-muted-foreground",
};

function shortGpu(s: string | null | undefined): string {
  return formatGpu(s) || "—";
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

export function BenchmarkRow({
  bench,
  selectMode = false,
  selected = false,
  onToggle,
  onDelete,
  onRename,
  onTogglePublic,
}: {
  bench: BenchmarkRecord;
  selectMode?: boolean;
  selected?: boolean;
  onToggle?: (id: string) => void;
  onDelete?: (bench: BenchmarkRecord) => void;
  onRename?: (bench: BenchmarkRecord) => void;
  onTogglePublic?: (bench: BenchmarkRecord) => void;
}) {
  // is_owner is undefined on older payloads — treat as owned for back-compat.
  const owned = bench.is_owner ?? true;
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

  const inner = (
    <>
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          {selectMode && (
            <input
              type="checkbox"
              checked={selected}
              onChange={() => onToggle?.(bench.id)}
              onClick={(e) => e.stopPropagation()}
              className="h-4 w-4 shrink-0 cursor-pointer accent-primary"
              aria-label={`Select ${bench.name}`}
            />
          )}
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-border bg-muted/60 text-base font-semibold text-muted-foreground">
            {avatar.letter}
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="truncate font-medium text-foreground">{bench.name}</span>
              <span
                className={cn(
                  "rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide",
                  STATUS_STYLES[bench.status] ?? STATUS_STYLES.queued,
                )}
              >
                {bench.status}
              </span>
              {bench.is_public && (
                <span
                  className="inline-flex items-center gap-1 rounded-full border border-sky-500/40 bg-sky-500/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-sky-700 dark:text-sky-400"
                  title="Public — visible to everyone"
                >
                  <Globe className="h-3 w-3" /> Public
                </span>
              )}
            </div>
            <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="truncate font-mono" title={bench.id}>{bench.id}</span>
              <span>·</span>
              <User className="h-3 w-3" />
              <span className="truncate">
                {owned ? bench.created_by : `owned by ${bench.created_by}`}
              </span>
            </div>
          </div>
        </div>

        <div className="flex shrink-0 items-start gap-2">
          {tput != null && (
            <div className="rounded-md border border-border bg-muted/40 px-2.5 py-1 text-right">
              <div className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                <TrendingUp className="h-3 w-3" /> Throughput
              </div>
              <div className="font-mono text-sm font-semibold tabular-nums">
                {fmtTput(tput)}
              </div>
            </div>
          )}
          {!selectMode && (onDelete || onRename || onTogglePublic) && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="-mr-1 text-muted-foreground hover:text-foreground"
                  aria-label="Actions"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                  }}
                >
                  <MoreHorizontal className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
                {onTogglePublic && (
                  <DropdownMenuItem
                    onSelect={(e) => {
                      e.preventDefault();
                      onTogglePublic(bench);
                    }}
                  >
                    {bench.is_public ? (
                      <>
                        <Lock className="h-4 w-4" />
                        Make private
                      </>
                    ) : (
                      <>
                        <Globe className="h-4 w-4" />
                        Make public
                      </>
                    )}
                  </DropdownMenuItem>
                )}
                {onRename && (
                  <DropdownMenuItem
                    onSelect={(e) => {
                      e.preventDefault();
                      onRename(bench);
                    }}
                  >
                    <Pencil className="h-4 w-4" />
                    Rename
                  </DropdownMenuItem>
                )}
                {onDelete && (
                  <DropdownMenuItem
                    variant="destructive"
                    onSelect={(e) => {
                      e.preventDefault();
                      onDelete(bench);
                    }}
                  >
                    <Trash2 className="h-4 w-4" />
                    Delete benchmark
                  </DropdownMenuItem>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
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
          <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 font-mono text-xs">
            {parallelism}
          </span>
        )}
        {benchCount > 1 && (
          <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs">
            <Layers className="h-3 w-3 text-muted-foreground" />
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
          <CostCell bench={bench} />
          {bench.exit_code != null && bench.exit_code !== 0 && (
            <span className="font-mono text-destructive">exit {bench.exit_code}</span>
          )}
        </div>
        <span title={new Date(bench.created_at).toISOString()}>
          {new Date(bench.created_at).toLocaleString()}
        </span>
      </div>
    </>
  );

  const baseCard =
    "group block rounded-xl border border-border bg-card p-4 transition-all";

  if (selectMode) {
    return (
      <div
        role="button"
        tabIndex={0}
        onClick={() => onToggle?.(bench.id)}
        onKeyDown={(e) => {
          if (e.key === " " || e.key === "Enter") {
            e.preventDefault();
            onToggle?.(bench.id);
          }
        }}
        className={cn(
          baseCard,
          "cursor-pointer",
          selected
            ? "border-primary/60 bg-primary/5"
            : "hover:border-primary/40 hover:bg-card/80",
        )}
      >
        {inner}
      </div>
    );
  }

  return (
    <Link
      href={`/benchmark/${encodeURIComponent(bench.id)}`}
      className={cn(baseCard, "hover:border-primary/40 hover:bg-card/80 hover:shadow-md")}
    >
      {inner}
    </Link>
  );
}

function CostCell({ bench }: { bench: BenchmarkRecord }) {
  const live = useLiveCost(bench.started_at, bench.ended_at, bench.cost_per_hr);
  const isBurning = bench.status === "running" && bench.ended_at == null;
  if (live == null) return null;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 tabular-nums",
        isBurning && "text-amber-600 dark:text-amber-400",
      )}
    >
      {isBurning ? <BurnFlame /> : <span className="font-mono">$</span>}
      {formatCostUSD(live)}
    </span>
  );
}
