"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { useListUrlState, readParam } from "@/lib/list-url-state";
import {
  Cpu,
  Inbox,
  LayoutGrid,
  List,
  Loader2,
  MoreHorizontal,
  Pencil,
  Search,
  Trash2,
  User,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { TrainingRunRecord } from "@/lib/types";
import { avatarFor } from "@/lib/avatar";
import { shortGpu } from "@/lib/gpu-format";
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

// Status pill — soft tint + matching text + neutral border (matches Benchmark).
const STATUS_PILL: Record<string, string> = {
  queued: "border border-border bg-muted text-muted-foreground",
  running: "border border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
  done: "border border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  failed: "border border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
  cancelled: "border border-border bg-muted text-muted-foreground",
};

const STATUS_OPTIONS = ["all", "queued", "running", "done", "failed", "cancelled"] as const;
type StatusFilter = (typeof STATUS_OPTIONS)[number];

function taskLabel(r: TrainingRunRecord): string {
  return r.task_type === "tts" ? "TTS" : "ASR";
}

function isSweepRun(r: TrainingRunRecord): boolean {
  const sweep = (r.config_json?.sweep ?? null) as Record<string, unknown> | null;
  return !!sweep && Object.values(sweep).some((v) => Array.isArray(v) && v.length > 0);
}

// Headline metric for the card's right-hand box.
function primaryMetric(r: TrainingRunRecord): { label: string; value: string } | null {
  const best = r.result_json?.best;
  if (!best) return null;
  if (best.wer != null) return { label: "WER", value: best.wer.toFixed(2) };
  if (best.metric != null) return { label: "metric", value: best.metric.toFixed(3) };
  if (best.loss != null) return { label: "loss", value: best.loss.toFixed(3) };
  return null;
}

export function AutotrainList({
  initialItems,
  initialTotal,
  scope,
}: {
  initialItems: TrainingRunRecord[];
  initialTotal: number;
  scope: "mine" | "all";
}) {
  const sp = useSearchParams();
  // Seed search/status/sort/view from the URL (shareable); mirrored back below.
  const [q, setQ] = useState(() => sp.get("q") ?? "");
  const [qDebounced, setQDebounced] = useState(q);
  const [status, setStatus] = useState<StatusFilter>(() => readParam(sp, "status", STATUS_OPTIONS, "all"));
  const [sort, setSort] = useState<SortDir>(() => readParam(sp, "sort", ["newest", "oldest"] as const, "newest"));
  const [view, setView] = useState<"rows" | "grid">(() => readParam(sp, "view", ["rows", "grid"] as const, "grid"));
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(12);

  // Server-paginated data: SSR delivers page 1, everything after is fetched.
  const [items, setItems] = useState<TrainingRunRecord[]>(initialItems);
  const [total, setTotal] = useState(initialTotal);
  const [loading, setLoading] = useState(false);

  const [single, setSingle] = useState<TrainingRunRecord | null>(null);
  const [singleDeleting, setSingleDeleting] = useState(false);
  const [singleError, setSingleError] = useState<string | null>(null);
  const [renameTarget, setRenameTarget] = useState<TrainingRunRecord | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);

  useEffect(() => {
    if (sp.get("view")) return;   // URL view wins over the saved preference
    const v = window.localStorage.getItem("sgpu_autotrain_view");
    // Reading client-only localStorage post-mount avoids an SSR/CSR mismatch.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (v === "rows" || v === "grid") setView(v);
  }, [sp]);
  const setViewPersist = (v: "rows" | "grid") => {
    setView(v);
    window.localStorage.setItem("sgpu_autotrain_view", v);
  };
  useListUrlState({ q, status, sort, view });

  // Debounce the search box so each keystroke doesn't hit the gateway.
  useEffect(() => {
    const t = setTimeout(() => setQDebounced(q), 300);
    return () => clearTimeout(t);
  }, [q]);

  const hasFilter = q.trim().length > 0 || status !== "all";

  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  // Clamp in render so a shrinking result set never strands an empty page; the
  // search/filter handlers reset to page 1 directly.
  const currentPage = Math.min(page, pageCount);

  // Fetch the current page from the gateway (search/filter/sort run server-side).
  // seqRef discards out-of-order responses so a slow reply can't clobber a newer one.
  const seqRef = useRef(0);
  const load = useCallback(async () => {
    const seq = ++seqRef.current;
    setLoading(true);
    try {
      const res = await gateway.listTrainingRunsPage({
        scope,
        q: qDebounced,
        status: status === "all" ? "" : status,
        sort,
        limit: pageSize,
        offset: (currentPage - 1) * pageSize,
      });
      if (seq !== seqRef.current) return;
      setItems(res.items);
      setTotal(res.total);
    } catch (e) {
      // Keep the previous page on failure; surface the error once.
      if (seq === seqRef.current) toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      if (seq === seqRef.current) setLoading(false);
    }
  }, [scope, qDebounced, status, sort, pageSize, currentPage]);

  // Refetch on any query change — except the very first run while state still
  // equals the SSR defaults (the server already rendered that exact page). A
  // URL-seeded q/status/sort is non-default, so it does fetch on mount.
  const bootedRef = useRef(false);
  useEffect(() => {
    if (!bootedRef.current) {
      bootedRef.current = true;
      const ssrDefaults =
        qDebounced === "" && status === "all" && sort === "newest" && page === 1 && pageSize === 12;
      if (ssrDefaults) return;
    }
    void load();
  }, [load, qDebounced, status, sort, page, pageSize]);

  const onRename = async () => {
    if (!renameTarget) return;
    const name = renameDraft.trim();
    if (!name || name === renameTarget.name) {
      setRenameTarget(null);
      return;
    }
    setRenameError(null);
    setRenaming(true);
    try {
      await gateway.renameTrainingRun(renameTarget.id, name);
      setRenameTarget(null);
      void load();
    } catch (e) {
      setRenameError(e instanceof Error ? e.message : String(e));
    } finally {
      setRenaming(false);
    }
  };

  const onSingleDelete = async () => {
    if (!single) return;
    setSingleError(null);
    setSingleDeleting(true);
    try {
      await gateway.deleteTrainingRun(single.id);
      setSingle(null);
      void load();
    } catch (e) {
      setSingleError(e instanceof Error ? e.message : String(e));
    } finally {
      setSingleDeleting(false);
    }
  };

  const openRename = (r: TrainingRunRecord) => {
    setRenameTarget(r);
    setRenameDraft(r.name);
    setRenameError(null);
  };

  return (
    <div>
      <div className="mb-4 flex gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="search"
            placeholder="Search by name, id, model, task, dataset, GPU, owner, status…"
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              setPage(1);
            }}
            className="h-10 w-full rounded-md border border-input bg-background pl-9 pr-9 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
          />
          {q && (
            <button
              type="button"
              onClick={() => {
                setQ("");
                setPage(1);
              }}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              title="Clear"
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
          <SelectTrigger className="h-10! w-[150px]" title="Filter by status">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {STATUS_OPTIONS.map((s) => (
              <SelectItem key={s} value={s}>
                {s === "all" ? "All statuses" : s.charAt(0).toUpperCase() + s.slice(1)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <SortSelect value={sort} onValueChange={setSort} />
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
      </div>

      {hasFilter && (
        <div className="mb-3 text-xs text-muted-foreground">
          {loading && <Loader2 className="mr-1.5 inline h-3 w-3 animate-spin" />}
          {total} {total === 1 ? "match" : "matches"}
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

      {total === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
          <Inbox className="h-6 w-6 text-muted-foreground/60" />
          <p className="text-sm text-muted-foreground">No training runs match your filters.</p>
        </div>
      ) : (
        <>
          <div
            className={cn(
              "gap-3 transition-opacity",
              view === "rows" ? "flex flex-col" : "grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3",
              loading && "pointer-events-none opacity-60",
            )}
          >
            {items.map((r) => (
              <RunItem key={r.id} run={r} onRename={openRename} onDelete={setSingle} />
            ))}
          </div>
          <Pagination
            page={currentPage}
            pageCount={pageCount}
            total={total}
            pageSize={pageSize}
            onPageChange={setPage}
            onPageSizeChange={(n) => {
              setPageSize(n);
              setPage(1);
            }}
            itemLabel="runs"
          />
        </>
      )}

      <Dialog
        open={!!single}
        onOpenChange={(o) => {
          if (!singleDeleting && !o) {
            setSingle(null);
            setSingleError(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete {single?.name}?</DialogTitle>
            <DialogDescription>
              Removes the training-run record. S3 artifacts are kept. If a RunPod pod
              is still alive, terminate it from RunPod&apos;s dashboard.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {singleError && <p className="mr-auto text-sm text-destructive">{singleError}</p>}
            <Button variant="outline" onClick={() => setSingle(null)} disabled={singleDeleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={onSingleDelete} disabled={singleDeleting}>
              {singleDeleting ? "Deleting…" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={!!renameTarget}
        onOpenChange={(o) => {
          if (!renaming && !o) {
            setRenameTarget(null);
            setRenameError(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename run</DialogTitle>
            <DialogDescription>
              Updates the display name only. The run, S3 files, and config are unchanged.
            </DialogDescription>
          </DialogHeader>
          <input
            autoFocus
            value={renameDraft}
            onChange={(e) => setRenameDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !renaming && renameDraft.trim()) onRename();
            }}
            disabled={renaming}
            maxLength={200}
            placeholder="Run name"
            className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
          />
          <DialogFooter>
            {renameError && <p className="mr-auto text-sm text-destructive">{renameError}</p>}
            <Button variant="outline" onClick={() => setRenameTarget(null)} disabled={renaming}>
              Cancel
            </Button>
            <Button onClick={onRename} disabled={renaming || !renameDraft.trim()}>
              {renaming ? "Saving…" : "Save"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// A single white card, used in both list (full-width, stacked) and grid views —
// the container in AutotrainList switches the layout. Mirrors BenchmarkRow.
function RunItem({
  run,
  onRename,
  onDelete,
}: {
  run: TrainingRunRecord;
  onRename: (r: TrainingRunRecord) => void;
  onDelete: (r: TrainingRunRecord) => void;
}) {
  const avatar = avatarFor(run.name);
  const metric = primaryMetric(run);
  const cer = run.result_json?.best?.cer;
  const sweep = isSweepRun(run);
  const model = run.base_model.split("/").pop() ?? run.base_model;
  const gpu = run.gpu_type
    ? `${shortGpu(run.gpu_type)}${run.gpu_count > 1 ? ` × ${run.gpu_count}` : ""}`
    : run.visible_devices
      ? `GPUs ${run.visible_devices}`
      : null;

  const chip = "inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs";

  return (
    <Link
      href={`/autotrain/${encodeURIComponent(run.id)}`}
      className={cn(
        "group block rounded-xl border border-border bg-card p-4 transition-all",
        "hover:border-primary/40 hover:bg-card/80 hover:shadow-md",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-border bg-muted/60 text-base font-semibold text-muted-foreground">
            {avatar.letter}
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="truncate font-medium text-foreground">{run.name}</span>
              <span
                className={cn(
                  "rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide",
                  STATUS_PILL[run.status] ?? STATUS_PILL.queued,
                )}
              >
                {run.status}
              </span>
            </div>
            <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="truncate font-mono" title={run.id}>{run.id}</span>
              <span>·</span>
              <User className="h-3 w-3" />
              <span className="truncate">{run.created_by}</span>
            </div>
          </div>
        </div>

        <div className="flex shrink-0 items-start gap-2">
          {metric && (
            <div className="rounded-md border border-border bg-muted/40 px-2.5 py-1 text-right">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                Best {metric.label}
              </div>
              <div className="font-mono text-sm font-semibold tabular-nums">{metric.value}</div>
            </div>
          )}
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
              <DropdownMenuItem
                onSelect={(e) => {
                  e.preventDefault();
                  onRename(run);
                }}
              >
                <Pencil className="h-4 w-4" />
                Rename
              </DropdownMenuItem>
              <DropdownMenuItem
                variant="destructive"
                onSelect={(e) => {
                  e.preventDefault();
                  onDelete(run);
                }}
              >
                <Trash2 className="h-4 w-4" />
                Delete run
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        <span className={chip}>{taskLabel(run)}</span>
        <span
          className={cn(
            chip,
            "bg-transparent",
            sweep
              ? "border border-violet-500/40 text-violet-600 dark:text-violet-300"
              : "border border-sky-500/40 text-sky-600 dark:text-sky-300",
          )}
        >
          {sweep ? "sweep" : "single"}
        </span>
        <span className={chip}>
          <span className="font-mono">{model}</span>
        </span>
        {gpu && (
          <span className={chip}>
            <Cpu className="h-3 w-3 text-muted-foreground" />
            <span className="font-mono">{gpu}</span>
          </span>
        )}
      </div>

      <div className="mt-3 flex items-center justify-between border-t border-border/60 pt-2 text-xs text-muted-foreground">
        <div className="flex items-center gap-3">
          {cer != null && <span className="tabular-nums">CER {cer.toFixed(2)}</span>}
          {run.exit_code != null && run.exit_code !== 0 && (
            <span className="font-mono text-destructive">exit {run.exit_code}</span>
          )}
        </div>
        <span title={new Date(run.created_at).toISOString()}>
          {new Date(run.created_at).toLocaleString()}
        </span>
      </div>
    </Link>
  );
}
