"use client";

import { Loader2, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import type { VmAvailability } from "@/lib/types";

// Live SSH-probe state for a registered VM provider. Shared by the Autotrain
// Export-to-Label tab and the Try-it compute picker so both render the same row.
export type VmAvailState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ok"; data: VmAvailability }
  | { status: "error"; message: string };

// Rich availability row: GPUs free/total + busy count, with a refresh/retry
// affordance. Mirrors the serverless deploy form's VM probe row.
export function VmAvailabilityRow({ state, onRefresh }: { state: VmAvailState; onRefresh: () => void }) {
  if (state.status === "idle") return null;
  if (state.status === "loading") {
    return (
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" /> Checking availability via SSH…
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="flex items-center justify-between gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-2.5 py-1.5 text-xs text-destructive">
        <span className="truncate" title={state.message}>{state.message}</span>
        <button type="button" onClick={onRefresh} className="inline-flex items-center gap-1 underline-offset-2 hover:underline">
          <RefreshCw className="h-3 w-3" /> Retry
        </button>
      </div>
    );
  }
  const { data } = state;
  if (!data.ok) {
    return (
      <div className="flex items-center justify-between gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-2.5 py-1.5 text-xs text-amber-700 dark:text-amber-400">
        <span className="truncate" title={data.message}>{data.message}</span>
        <button type="button" onClick={onRefresh} className="inline-flex items-center gap-1 underline-offset-2 hover:underline">
          <RefreshCw className="h-3 w-3" /> Retry
        </button>
      </div>
    );
  }
  const totalFreeMib = data.gpus.reduce((s, g) => s + g.mem_free_mib, 0);
  const totalMib = data.gpus.reduce((s, g) => s + g.mem_total_mib, 0);
  const busy = data.gpus.filter((g) => g.mem_free_mib < g.mem_total_mib * 0.2 || g.util_pct > 50).length;
  const allFree = busy === 0;
  return (
    <div className={cn("flex items-center justify-between gap-2 rounded-md border px-2.5 py-1.5 text-xs",
      allFree ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
        : "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400")}>
      <span>
        {data.gpus.length} GPU{data.gpus.length === 1 ? "" : "s"} · {fmtMib(totalFreeMib)} free / {fmtMib(totalMib)}
        {!allFree && ` · ${busy} busy`}
      </span>
      <button type="button" onClick={onRefresh} className="inline-flex items-center gap-1 underline-offset-2 hover:underline">
        <RefreshCw className="h-3 w-3" /> Refresh
      </button>
    </div>
  );
}

function fmtMib(mib: number): string {
  if (mib >= 1024) return `${(mib / 1024).toFixed(1)} GiB`;
  return `${mib} MiB`;
}
