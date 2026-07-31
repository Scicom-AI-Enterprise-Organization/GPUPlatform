"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { useListUrlState, readParam } from "@/lib/list-url-state";
import { Loader2, MoreHorizontal, Search, Square, Trash2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { ExperimentRecord } from "@/lib/types";
import { avatarFor } from "@/lib/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Pagination } from "@/components/ui/pagination";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SortSelect, type SortDir } from "@/components/ui/sort-select";

const STATUS_PILL: Record<string, string> = {
  queued: "border border-border bg-muted text-muted-foreground",
  running: "border border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
  completed: "border border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  failed: "border border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
  cancelled: "border border-border bg-muted text-muted-foreground",
};

const STATUS_OPTIONS = ["all", "queued", "running", "completed", "failed", "cancelled"] as const;
type StatusFilter = (typeof STATUS_OPTIONS)[number];

/** Pass rate reads as the headline number, so colour it by health. */
function rateTone(rate: number) {
  if (rate >= 0.95) return "text-emerald-600 dark:text-emerald-400";
  if (rate >= 0.8) return "text-amber-600 dark:text-amber-400";
  return "text-red-600 dark:text-red-400";
}

export function ExperimentsList({
  initialItems,
  initialTotal,
  scope,
}: {
  initialItems: ExperimentRecord[];
  initialTotal: number;
  scope: "mine" | "all";
}) {
  const sp = useSearchParams();
  const [q, setQ] = useState(() => sp.get("q") ?? "");
  const [qDebounced, setQDebounced] = useState(q);
  const [status, setStatus] = useState<StatusFilter>(() =>
    readParam(sp, "status", STATUS_OPTIONS, "all"),
  );
  const [sort, setSort] = useState<SortDir>(() =>
    readParam(sp, "sort", ["newest", "oldest"] as const, "newest"),
  );
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(12);

  const [items, setItems] = useState<ExperimentRecord[]>(initialItems);
  const [total, setTotal] = useState(initialTotal);
  const [loading, setLoading] = useState(false);
  const [pending, setPending] = useState<ExperimentRecord | null>(null);
  const [deleting, setDeleting] = useState(false);

  useListUrlState({ q, status, sort });

  useEffect(() => {
    const t = setTimeout(() => setQDebounced(q), 300);
    return () => clearTimeout(t);
  }, [q]);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const res = await gateway.listExperimentsPage({
        scope,
        q: qDebounced,
        status: status === "all" ? "" : status,
        sort,
        limit: pageSize,
        offset: (page - 1) * pageSize,
      });
      setItems(res.items);
      setTotal(res.total);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't load experiments");
    } finally {
      setLoading(false);
    }
  }, [scope, qDebounced, status, sort, page, pageSize]);

  // Skip the first run: the server component already rendered this exact page.
  const bootedRef = useRef(false);
  useEffect(() => {
    if (!bootedRef.current) {
      bootedRef.current = true;
      const ssrDefaults =
        qDebounced === "" && status === "all" && sort === "newest" && page === 1 && pageSize === 12;
      if (ssrDefaults) return;
    }
    void refresh();
  }, [refresh, qDebounced, status, sort, page, pageSize]);

  // Poll while anything is in flight so progress advances without a manual reload.
  const hasActive = items.some((i) => i.status === "running" || i.status === "queued");
  useEffect(() => {
    if (!hasActive) return;
    const t = setInterval(() => void refresh(), 4000);
    return () => clearInterval(t);
  }, [hasActive, refresh]);

  async function onCancel(row: ExperimentRecord) {
    try {
      await gateway.cancelExperiment(row.id);
      toast.success("Cancelling — results collected so far are kept.");
      void refresh();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't cancel");
    }
  }

  async function onDelete() {
    if (!pending) return;
    setDeleting(true);
    try {
      await gateway.deleteExperiment(pending.id);
      toast.success(`Deleted ${pending.name}`);
      setPending(null);
      void refresh();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't delete");
    } finally {
      setDeleting(false);
    }
  }

  return (
    <>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="relative min-w-[200px] flex-1">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <input
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              setPage(1);
            }}
            placeholder="Search experiments…"
            className="h-8 w-full rounded-md border border-input bg-background pl-8 pr-7 text-sm outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
          {q && (
            <button
              onClick={() => setQ("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        <Select
          value={status}
          onValueChange={(v) => {
            setStatus(v as StatusFilter);
            setPage(1);
          }}
        >
          <SelectTrigger className="h-8 w-[140px] text-sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {STATUS_OPTIONS.map((s) => (
              <SelectItem key={s} value={s}>
                {s === "all" ? "All statuses" : s}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <SortSelect value={sort} onValueChange={(v) => setSort(v)} className="h-8! w-[150px]" />
        {loading && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
      </div>

      <div className="overflow-hidden rounded-md border border-border">
        <table className="w-full text-sm">
          <thead className="bg-muted/40 text-xs text-muted-foreground">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Experiment</th>
              <th className="px-3 py-2 text-left font-medium">Dataset</th>
              <th className="px-3 py-2 text-left font-medium">Status</th>
              <th className="px-3 py-2 text-right font-medium">Samples</th>
              <th className="px-3 py-2 text-right font-medium">Pass rate</th>
              <th className="px-3 py-2 text-left font-medium">By</th>
              <th className="w-10 px-3 py-2" />
            </tr>
          </thead>
          <tbody>
            {items.map((row) => {
              const rate = row.summary?.totals.pass_rate;
              const progress =
                row.n_planned > 0 ? Math.round((row.n_completed / row.n_planned) * 100) : 0;
              return (
                <tr key={row.id} className="border-t border-border hover:bg-muted/30">
                  <td className="px-3 py-2">
                    <Link
                      href={`/experiments/${row.id}`}
                      className="font-medium hover:underline underline-offset-2"
                    >
                      {row.name}
                    </Link>
                    <div className="font-mono text-[11px] text-muted-foreground">{row.id}</div>
                  </td>
                  <td className="px-3 py-2">
                    <Link
                      href={`/datasets/${row.dataset_id}`}
                      className="text-muted-foreground hover:text-foreground hover:underline underline-offset-2"
                    >
                      {row.dataset_name}
                    </Link>
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className={cn(
                        "inline-flex items-center rounded-md px-1.5 py-0.5 text-[11px] font-medium",
                        STATUS_PILL[row.status] ?? STATUS_PILL.queued,
                      )}
                    >
                      {row.status === "running" && (
                        <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                      )}
                      {row.status}
                      {row.status === "running" && ` ${progress}%`}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                    {row.n_completed}
                    <span className="text-muted-foreground/60">/{row.n_planned}</span>
                    {row.n_failed > 0 && (
                      <span className="ml-1 text-red-600 dark:text-red-400">
                        ({row.n_failed} err)
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {rate === undefined ? (
                      <span className="text-muted-foreground">—</span>
                    ) : (
                      <span className={cn("font-medium", rateTone(rate))}>
                        {Math.round(rate * 100)}%
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2">
                    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                      <span
                        className={cn(
                          "inline-flex h-4 w-4 items-center justify-center rounded-full text-[9px] font-medium",
                          avatarFor(row.owner).bg,
                          avatarFor(row.owner).text,
                        )}
                      >
                        {avatarFor(row.owner).letter}
                      </span>
                      {row.owner}
                    </span>
                  </td>
                  <td className="px-3 py-2">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-7 w-7">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        {(row.status === "running" || row.status === "queued") && (
                          <DropdownMenuItem onClick={() => void onCancel(row)}>
                            <Square className="h-4 w-4" />
                            Cancel run
                          </DropdownMenuItem>
                        )}
                        <DropdownMenuItem
                          className="text-destructive"
                          disabled={row.status === "running"}
                          onClick={() => setPending(row)}
                        >
                          <Trash2 className="h-4 w-4" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <Pagination
        page={page}
        pageCount={Math.max(1, Math.ceil(total / pageSize))}
        pageSize={pageSize}
        total={total}
        itemLabel="experiments"
        onPageChange={setPage}
        onPageSizeChange={(n) => {
          setPageSize(n);
          setPage(1);
        }}
      />

      <Dialog open={!!pending} onOpenChange={(o) => !o && setPending(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete experiment?</DialogTitle>
            <DialogDescription>
              {pending?.name} and all {pending?.n_completed} of its samples will be removed. The
              dataset it ran against is not affected.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPending(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={() => void onDelete()} disabled={deleting}>
              {deleting && <Loader2 className="h-4 w-4 animate-spin" />}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
