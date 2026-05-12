"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { CheckSquare, Cpu, Inbox, LayoutGrid, List, Search, Trash2, User, X } from "lucide-react";
import type { ComputePod, ComputeStatus } from "@/lib/types";
import { avatarFor } from "@/lib/avatar";
import { formatCostUSD, useLiveCost } from "@/lib/cost";
import { BurnFlame } from "@/components/burn-flame";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { gateway } from "@/lib/gateway";
import { cn } from "@/lib/utils";

const STATUS_OPTIONS = [
  "all",
  "running",
  "creating",
  "pending_approval",
  "failed",
  "rejected",
  "terminated",
] as const;
type StatusFilter = (typeof STATUS_OPTIONS)[number];

function searchableText(p: ComputePod): string {
  return [
    p.name,
    p.id,
    p.status,
    p.created_by,
    p.gpu_type,
    p.template_id ?? "",
    p.cloud_type,
  ]
    .join(" ")
    .toLowerCase();
}

export function ComputeList({ items }: { items: ComputePod[] }) {
  const router = useRouter();
  const [q, setQ] = useState("");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [selectMode, setSelectMode] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [view, setView] = useState<"rows" | "grid">("rows");
  useEffect(() => {
    const v = window.localStorage.getItem("sgpu_compute_view");
    if (v === "rows" || v === "grid") setView(v);
  }, []);
  const setViewPersist = (v: "rows" | "grid") => {
    setView(v);
    window.localStorage.setItem("sgpu_compute_view", v);
  };

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const exitSelect = () => {
    setSelectMode(false);
    setSelected(new Set());
  };

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const tokens = needle ? needle.split(/\s+/).filter(Boolean) : [];
    return items.filter((p) => {
      if (status !== "all" && p.status !== status) return false;
      if (tokens.length === 0) return true;
      const text = searchableText(p);
      return tokens.every((t) => text.includes(t));
    });
  }, [items, q, status]);

  const hasFilter = q.trim().length > 0 || status !== "all";

  const onDeleteSelected = async () => {
    if (selected.size === 0) return;
    setDeleting(true);
    setDeleteError(null);
    const ids = Array.from(selected);
    const results = await Promise.allSettled(ids.map((id) => gateway.deleteCompute(id)));
    const failures = results.filter((r) => r.status === "rejected").length;
    setDeleting(false);
    if (failures === 0) {
      setConfirmOpen(false);
      exitSelect();
      router.refresh();
    } else {
      setDeleteError(`${failures} of ${ids.length} failed to delete`);
      router.refresh();
    }
  };

  return (
    <div>
      <div className="mb-4 flex gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="search"
            placeholder="Search by name, id, GPU, owner, status…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="h-10 w-full rounded-md border border-input bg-background pl-9 pr-9 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
          />
          {q && (
            <button
              type="button"
              onClick={() => setQ("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              title="Clear"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        <select
          value={status}
          onChange={(e) => setStatus(e.target.value as StatusFilter)}
          className="h-10 rounded-md border border-input bg-background px-3 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
          title="Filter by status"
        >
          {STATUS_OPTIONS.map((s) => (
            <option key={s} value={s}>
              {s === "all" ? "All statuses" : s.replace("_", " ")}
            </option>
          ))}
        </select>
        <div className="inline-flex h-10 items-stretch overflow-hidden rounded-md border border-input bg-background shadow-xs">
          <button
            type="button"
            onClick={() => setViewPersist("rows")}
            className={cn(
              "inline-flex items-center justify-center px-2.5 text-sm",
              view === "rows" ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/50",
            )}
            title="List view"
            aria-label="List view"
            aria-pressed={view === "rows"}
          >
            <List className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => setViewPersist("grid")}
            className={cn(
              "inline-flex items-center justify-center border-l border-input px-2.5 text-sm",
              view === "grid" ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/50",
            )}
            title="Grid view"
            aria-label="Grid view"
            aria-pressed={view === "grid"}
          >
            <LayoutGrid className="h-4 w-4" />
          </button>
        </div>
        {selectMode ? (
          <button
            type="button"
            onClick={exitSelect}
            disabled={deleting}
            className="inline-flex h-10 items-center gap-1.5 rounded-md border border-input bg-background px-3 text-sm shadow-xs hover:bg-muted disabled:opacity-50"
          >
            <X className="h-4 w-4" /> Cancel
          </button>
        ) : (
          <button
            type="button"
            onClick={() => setSelectMode(true)}
            className="inline-flex h-10 items-center gap-1.5 rounded-md border border-input bg-background px-3 text-sm shadow-xs hover:bg-muted"
          >
            <CheckSquare className="h-4 w-4" /> Select
          </button>
        )}
      </div>

      {selectMode && (
        <div className="mb-3 flex items-center justify-between rounded-md border border-border bg-muted/40 px-3 py-2 text-sm">
          <span className="text-muted-foreground">
            {selected.size} selected
            {filtered.length > 0 && (
              <>
                {" "}
                <button
                  type="button"
                  onClick={() => setSelected(new Set(filtered.map((p) => p.id)))}
                  className="ml-2 underline underline-offset-2 hover:text-foreground"
                >
                  Select all visible
                </button>
                {selected.size > 0 && (
                  <>
                    {" · "}
                    <button
                      type="button"
                      onClick={() => setSelected(new Set())}
                      className="underline underline-offset-2 hover:text-foreground"
                    >
                      Clear
                    </button>
                  </>
                )}
              </>
            )}
          </span>
          <button
            type="button"
            onClick={() => setConfirmOpen(true)}
            disabled={selected.size === 0 || deleting}
            className="inline-flex items-center gap-1.5 rounded-md bg-destructive px-3 py-1.5 text-sm font-medium text-destructive-foreground hover:bg-destructive/90 disabled:opacity-50"
          >
            <Trash2 className="h-3.5 w-3.5" />
            {deleting ? "Deleting…" : `Delete ${selected.size > 0 ? selected.size : ""}`.trim()}
          </button>
        </div>
      )}

      {hasFilter && (
        <div className="mb-3 text-xs text-muted-foreground">
          {filtered.length} of {items.length} match
          {q && (
            <>
              {" "}for <span className="font-mono text-foreground">&quot;{q}&quot;</span>
            </>
          )}
          {status !== "all" && (
            <>
              {" "}· status <span className="font-mono text-foreground">{status}</span>
            </>
          )}
        </div>
      )}

      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
          <Inbox className="h-6 w-6 text-muted-foreground/60" />
          <p className="text-sm text-muted-foreground">No pods match your filters.</p>
        </div>
      ) : (
        <ul
          className={cn(
            "gap-3",
            view === "rows"
              ? "flex flex-col"
              : "grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3",
          )}
        >
          {filtered.map((p) => (
            <PodRow
              key={p.id}
              pod={p}
              selectMode={selectMode}
              selected={selected.has(p.id)}
              onToggle={toggle}
            />
          ))}
        </ul>
      )}

      <Dialog
        open={confirmOpen}
        onOpenChange={(o) => {
          if (!deleting) {
            setConfirmOpen(o);
            if (!o) setDeleteError(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              Delete {selected.size} pod{selected.size === 1 ? "" : "s"}?
            </DialogTitle>
            <DialogDescription>
              Terminates the RunPod instances and removes the records. Billing
              stops once the pod is fully torn down.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {deleteError && (
              <p className="mr-auto text-sm text-destructive">{deleteError}</p>
            )}
            <Button variant="outline" onClick={() => setConfirmOpen(false)} disabled={deleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={onDeleteSelected} disabled={deleting}>
              {deleting ? "Deleting…" : `Delete ${selected.size}`}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function PodRow({
  pod,
  selectMode,
  selected,
  onToggle,
}: {
  pod: ComputePod;
  selectMode: boolean;
  selected: boolean;
  onToggle: (id: string) => void;
}) {
  const avatar = avatarFor(pod.name);
  const inner = (
    <>
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          {selectMode && (
            <input
              type="checkbox"
              checked={selected}
              onChange={() => onToggle(pod.id)}
              onClick={(e) => e.stopPropagation()}
              className="h-4 w-4 shrink-0 cursor-pointer accent-primary"
              aria-label={`Select ${pod.name}`}
            />
          )}
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-border bg-muted/60 text-base font-semibold text-muted-foreground">
            {avatar.letter}
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="truncate font-medium text-foreground">{pod.name}</span>
              <StatusPill status={pod.status} />
            </div>
            <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="truncate font-mono" title={pod.id}>
                {pod.id}
              </span>
              <span>·</span>
              <User className="h-3 w-3" />
              <span className="truncate">{pod.created_by}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs">
          <Cpu className="h-3 w-3 text-muted-foreground" />
          <span className="font-mono">
            {shortGpu(pod.gpu_type)}
            {pod.gpu_count > 1 ? ` × ${pod.gpu_count}` : ""}
          </span>
        </span>
        <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 font-mono text-xs">
          {pod.container_disk_gb} GB disk
        </span>
        {pod.template_id && (
          <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 font-mono text-xs">
            {pod.template_id}
          </span>
        )}
        <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 font-mono text-xs">
          {pod.cloud_type.toLowerCase()}
        </span>
      </div>

      <div className="mt-3 flex items-center justify-between border-t border-border/60 pt-2 text-xs text-muted-foreground">
        <CostCell pod={pod} />
        <span title={new Date(pod.created_at).toISOString()}>
          {new Date(pod.created_at).toLocaleString()}
        </span>
      </div>
    </>
  );

  const base = "group block rounded-xl border border-border bg-card p-4 transition-all";

  if (selectMode) {
    return (
      <li
        role="button"
        tabIndex={0}
        onClick={() => onToggle(pod.id)}
        onKeyDown={(e) => {
          if (e.key === " " || e.key === "Enter") {
            e.preventDefault();
            onToggle(pod.id);
          }
        }}
        className={cn(
          base,
          "cursor-pointer",
          selected
            ? "border-primary/60 bg-primary/5"
            : "hover:border-primary/40 hover:bg-card/80",
        )}
      >
        {inner}
      </li>
    );
  }

  return (
    <li>
      <Link
        href={`/compute/${pod.id}`}
        className={cn(base, "hover:border-primary/40 hover:bg-card/80 hover:shadow-md")}
      >
        {inner}
      </Link>
    </li>
  );
}

function CostCell({ pod }: { pod: ComputePod }) {
  // Billing starts when the pod is actually up (ready_at) and stops when it's
  // torn down (terminated_at). Anything before ready_at isn't being charged.
  const live = useLiveCost(pod.ready_at, pod.terminated_at, pod.cost_per_hr);
  const isBurning = pod.ready_at != null && pod.terminated_at == null;
  if (live == null) return null;
  return (
    <span className="inline-flex items-center gap-1 tabular-nums">
      {isBurning ? <BurnFlame /> : <span className="font-mono">$</span>}
      {formatCostUSD(live)}
    </span>
  );
}

function StatusPill({ status }: { status: ComputeStatus }) {
  const styles: Record<ComputeStatus, string> = {
    running:
      "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
    creating:
      "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
    pending_approval:
      "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
    failed: "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
    rejected: "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
    terminated: "border-border bg-muted text-muted-foreground",
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
  return gpu
    .replace(/^NVIDIA\s+/i, "")
    .replace(/\s+GeForce\s+/i, " ")
    .replace(/^GeForce\s+/i, "");
}

