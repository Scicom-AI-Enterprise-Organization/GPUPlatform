"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import type { ComputePod, ComputeStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

export function ComputeList({ items }: { items: ComputePod[] }) {
  return (
    <ul className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {items.map((p) => (
        <li key={p.id}>
          <Link
            href={`/compute/${p.id}`}
            className="group block rounded-lg border border-border bg-card p-4 transition-colors hover:border-foreground/30"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <h3 className="truncate text-sm font-medium">{p.name}</h3>
                  <StatusPill status={p.status} />
                </div>
                <p className="mt-0.5 truncate font-mono text-xs text-muted-foreground">
                  {p.id}
                </p>
              </div>
              <ArrowRight className="h-4 w-4 shrink-0 text-muted-foreground/60 transition-transform group-hover:translate-x-0.5" />
            </div>

            {/* Neutral metadata chips — gray, never coloured. */}
            <div className="mt-3 flex flex-wrap gap-1.5">
              <Chip>{shortGpu(p.gpu_type)} × {p.gpu_count}</Chip>
              <Chip>{p.container_disk_gb} GB disk</Chip>
              {p.template_id && <Chip>{p.template_id}</Chip>}
              <Chip>{p.cloud_type.toLowerCase()}</Chip>
            </div>

            <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
              <span>{p.created_by}</span>
              <span>{relativeTime(p.created_at)}</span>
            </div>
          </Link>
        </li>
      ))}
    </ul>
  );
}

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-md border border-border bg-background px-2 py-0.5 text-[11px] text-muted-foreground">
      {children}
    </span>
  );
}

function StatusPill({ status }: { status: ComputeStatus }) {
  // Status is the ONLY place colour appears on a card. Tints are intentionally
  // soft (border + tinted bg) so they read as state-tags, not call-to-action.
  const styles: Record<ComputeStatus, string> = {
    running:
      "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
    creating:
      "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
    pending_approval:
      "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
    failed:
      "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
    rejected:
      "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
    terminated:
      "border-border bg-muted text-muted-foreground",
  };
  const label: Record<ComputeStatus, string> = {
    running: "running",
    creating: "creating",
    pending_approval: "pending",
    failed: "failed",
    rejected: "rejected",
    terminated: "terminated",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide",
        styles[status],
      )}
    >
      {label[status]}
    </span>
  );
}

function shortGpu(gpu: string): string {
  // Strip the "NVIDIA " / "GeForce " prefixes RunPod uses so cards stay readable.
  return gpu
    .replace(/^NVIDIA\s+/i, "")
    .replace(/\s+GeForce\s+/i, " ")
    .replace(/^GeForce\s+/i, "");
}

function relativeTime(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  const diff = Date.now() - d.getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const days = Math.floor(h / 24);
  return `${days}d ago`;
}
