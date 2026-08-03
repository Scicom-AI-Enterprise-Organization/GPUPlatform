"use client";

// Mirrors experiments-list.tsx (search/status/sort, `_page` pagination, 4s poll
// while a run is active). The columns differ because the headline of an
// optimization isn't a pass rate — it's the DELTA: what the prompt scored before
// and after, and how much of the metric-call budget that cost.

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { useListUrlState, readParam } from "@/lib/list-url-state";
import { ArrowRight, Loader2, MoreHorizontal, Search, Square, Trash2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { PromptOptRecord } from "@/lib/types";
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

export function pct(v: number | null | undefined) {
  if (v === null || v === undefined) return "—";
  return `${Math.round(v * 100)}%`;
}

/** The delta is the whole point, so it gets the colour: green = the search found
 * something, grey = it didn't, and "didn't" is a perfectly normal outcome. */
export function deltaTone(delta: number) {
  if (delta > 0.001) return "text-emerald-600 dark:text-emerald-400";
  if (delta < -0.001) return "text-red-600 dark:text-red-400";
  return "text-muted-foreground";
}

export function OptimizeList({
  initialItems,
  initialTotal,
  scope,
}: {
  initialItems: PromptOptRecord[];
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

  const [items, setItems] = useState<PromptOptRecord[]>(initialItems);
  const [total, setTotal] = useState(initialTotal);
  const [loading, setLoading] = useState(false);
  const [pending, setPending] = useState<PromptOptRecord | null>(null);
  const [deleting, setDeleting] = useState(false);

  useListUrlState({ q, status, sort });

  useEffect(() => {
    const t = setTimeout(() => setQDebounced(q), 300);
    return () => clearTimeout(t);
  }, [q]);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const res = await gateway.listPromptOptsPage({
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
      toast.error(e instanceof Error ? e.message : "Couldn't load optimizations");
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

  const hasActive = items.some((i) => i.status === "running" || i.status === "queued");
  useEffect(() => {
    if (!hasActive) return;
    const t = setInterval(() => void refresh(), 4000);
    return () => clearInterval(t);
  }, [hasActive, refresh]);

  async function onCancel(row: PromptOptRecord) {
    try {
      await gateway.cancelPromptOpt(row.id);
      toast.success("Cancelling — the best prompt found so far is kept.");
      void refresh();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't cancel");
    }
  }

  async function onDelete() {
    if (!pending) return;
    setDeleting(true);
    try {
      await gateway.deletePromptOpt(pending.id);
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
            placeholder="Search optimizations…"
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

      <div className="overflow-x-auto rounded-md border border-border scrollbar-thin">
        <table className="w-full text-sm">
          <thead className="bg-muted/40 text-xs text-muted-foreground">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Optimization</th>
              <th className="px-3 py-2 text-left font-medium">Dataset</th>
              <th className="px-3 py-2 text-left font-medium">Status</th>
              <th className="px-3 py-2 text-right font-medium">Score</th>
              <th className="px-3 py-2 text-right font-medium">Gain</th>
              <th className="px-3 py-2 text-right font-medium">Metric calls</th>
              <th className="px-3 py-2 text-left font-medium">By</th>
              <th className="w-10 px-3 py-2" />
            </tr>
          </thead>
          <tbody>
            {items.map((row) => {
              const delta =
                row.best_score !== null && row.seed_score !== null
                  ? row.best_score - row.seed_score
                  : null;
              const progress =
                row.budget > 0 ? Math.round((row.metric_calls / row.budget) * 100) : 0;
              return (
                <tr key={row.id} className="border-t border-border hover:bg-muted/30">
                  <td className="px-3 py-2">
                    <Link
                      href={`/experiments/optimize/${row.id}`}
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
                  <td className="px-3 py-2 text-right tabular-nums">
                    <span className="text-muted-foreground">{pct(row.seed_score)}</span>
                    <ArrowRight className="mx-1 inline h-3 w-3 text-muted-foreground/60" />
                    <span className="font-medium">{pct(row.best_score)}</span>
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {delta === null ? (
                      <span className="text-muted-foreground">—</span>
                    ) : (
                      <span className={cn("font-medium", deltaTone(delta))}>
                        {delta > 0 ? "+" : ""}
                        {Math.round(delta * 100)}pp
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                    {row.metric_calls}
                    <span className="text-muted-foreground/60">/{row.budget}</span>
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
        itemLabel="optimizations"
        onPageChange={setPage}
        onPageSizeChange={(n) => {
          setPageSize(n);
          setPage(1);
        }}
      />

      <Dialog open={!!pending} onOpenChange={(o) => !o && setPending(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete optimization?</DialogTitle>
            <DialogDescription>
              {pending?.name} and the prompts it found will be removed. Copy the winning prompt
              first if you still want it — the dataset it ran against is not affected.
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
