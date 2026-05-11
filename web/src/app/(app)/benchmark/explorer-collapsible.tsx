"use client";

import { useState } from "react";
import { BarChart3, ChevronDown, ChevronRight } from "lucide-react";
import { BenchmarkExplorer } from "./explorer";

export function ExplorerCollapsible({ scope = "mine" }: { scope?: "mine" | "all" }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mb-8">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="group flex w-full items-center justify-between gap-3 rounded-lg border border-border bg-card px-4 py-3 text-left shadow-sm transition-colors hover:bg-muted/30"
      >
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-muted text-muted-foreground group-hover:text-foreground">
            <BarChart3 className="h-4 w-4" />
          </div>
          <div>
            <div className="text-sm font-semibold">Performance explorer</div>
            <div className="text-xs text-muted-foreground">
              Throughput / TTFT / TPOT / E2E across every benchmark + sub-run, with filters and log scale.
            </div>
          </div>
        </div>
        <span className="inline-flex items-center gap-1.5 rounded-md border border-border bg-background px-2 py-1 text-xs font-medium text-muted-foreground group-hover:text-foreground">
          {open ? (
            <>
              <ChevronDown className="h-3.5 w-3.5" />
              Hide
            </>
          ) : (
            <>
              <ChevronRight className="h-3.5 w-3.5" />
              Show
            </>
          )}
        </span>
      </button>
      {/* Lazy: only mounts (and fetches /benchmarks/_aggregate) when expanded. */}
      {open && (
        <div className="mt-3">
          <BenchmarkExplorer scope={scope} />
        </div>
      )}
    </div>
  );
}
