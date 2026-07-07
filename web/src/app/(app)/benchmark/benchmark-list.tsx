"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useListUrlState, readParam } from "@/lib/list-url-state";
import { CheckSquare, Download, GitCompare, Globe, Inbox, LayoutGrid, List, Loader2, Lock, MoreHorizontal, Search, Trash2, X } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { BenchmarkRecord } from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
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
import { BenchmarkRow } from "./benchmark-row";

const STATUS_OPTIONS = ["all", "queued", "running", "done", "failed", "cancelled"] as const;
type StatusFilter = (typeof STATUS_OPTIONS)[number];

export function BenchmarkList({
  initialItems,
  initialTotal,
  scope,
  isAdmin = false,
}: {
  initialItems: BenchmarkRecord[];
  initialTotal: number;
  scope: "mine" | "all";
  isAdmin?: boolean;
}) {
  const router = useRouter();
  const sp = useSearchParams();
  // Initial filter/sort/view/select come from the URL (shareable), falling back to
  // defaults. useListUrlState (below) mirrors changes back into the URL.
  const [q, setQ] = useState(() => sp.get("q") ?? "");
  const [status, setStatus] = useState<StatusFilter>(() => readParam(sp, "status", STATUS_OPTIONS, "all"));
  const [sort, setSort] = useState<SortDir>(() => readParam(sp, "sort", ["newest", "oldest"] as const, "newest"));
  const [selectMode, setSelectMode] = useState(() => sp.get("select") === "1");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [deleting, setDeleting] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [single, setSingle] = useState<BenchmarkRecord | null>(null);
  const [singleDeleting, setSingleDeleting] = useState(false);
  const [singleError, setSingleError] = useState<string | null>(null);
  const [renameTarget, setRenameTarget] = useState<BenchmarkRecord | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);
  const [view, setView] = useState<"rows" | "grid">(() => readParam(sp, "view", ["rows", "grid"] as const, "grid"));
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(12);
  // Server-paginated list state. The server component renders page 1; every
  // filter/sort/page change after that fetches from the gateway.
  const [items, setItems] = useState<BenchmarkRecord[]>(initialItems);
  const [total, setTotal] = useState(initialTotal);
  const [loading, setLoading] = useState(false);
  // Debounced copy of `q` — the fetch keys off this so we don't hit the
  // gateway on every keystroke.
  const [qDebounced, setQDebounced] = useState(q);
  useEffect(() => {
    const t = setTimeout(() => setQDebounced(q), 300);
    return () => clearTimeout(t);
  }, [q]);
  useEffect(() => {
    // The URL wins over localStorage (so a shared link shows its view); only fall
    // back to the saved preference when the URL didn't specify one.
    if (sp.get("view")) return;
    const v = window.localStorage.getItem("sgpu_bench_view");
    // Reading client-only localStorage post-mount avoids an SSR/CSR mismatch.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (v === "rows" || v === "grid") setView(v);
  }, [sp]);
  const setViewPersist = (v: "rows" | "grid") => {
    setView(v);
    window.localStorage.setItem("sgpu_bench_view", v);
  };
  // Mirror the shareable state into the URL (search, status, sort, view, select).
  useListUrlState({ q, status, sort, view, select: selectMode });

  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  // Clamp in render so a shrinking result set never strands an empty page; the
  // search/filter handlers reset to page 1 directly.
  const currentPage = Math.min(page, pageCount);

  // Fetch the current page from the gateway (search/filter/sort run server-side).
  // On error keep whatever page we already have and surface a toast.
  const load = useCallback(
    async (signal?: { cancelled: boolean }) => {
      setLoading(true);
      try {
        const res = await gateway.listBenchmarksPage({
          scope,
          q: qDebounced,
          status: status === "all" ? "" : status,
          sort,
          limit: pageSize,
          offset: (currentPage - 1) * pageSize,
        });
        if (signal?.cancelled) return;
        setItems(res.items);
        setTotal(res.total);
      } catch (e) {
        if (!signal?.cancelled) {
          toast.error(e instanceof Error ? e.message : String(e), { duration: 4000 });
        }
      } finally {
        if (!signal?.cancelled) setLoading(false);
      }
    },
    [scope, qDebounced, status, sort, pageSize, currentPage],
  );

  // Refetch whenever a server-side input changes. The ref skips the very first
  // run when state matches what the server already rendered (page 1, no
  // filters); a URL-seeded q/status/sort still fetches on mount.
  const skipInitialLoad = useRef(
    qDebounced === "" && status === "all" && sort === "newest" && page === 1 && pageSize === 12,
  );
  useEffect(() => {
    if (skipInitialLoad.current) {
      skipInitialLoad.current = false;
      return;
    }
    const signal = { cancelled: false };
    load(signal);
    return () => {
      signal.cancelled = true;
    };
  }, [load]);

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

  const onCompare = () => {
    const ids = Array.from(selected);
    if (ids.length < 2) return;
    router.push(`/benchmark/compare?ids=${ids.map(encodeURIComponent).join(",")}`);
  };

  // Download an export JSON for each selected benchmark, one by one. A small
  // gap between downloads lets the browser flush each (large) file and avoids
  // its "multiple downloads" throttle. Import them elsewhere via /benchmark/import.
  const onExportSelected = async () => {
    const ids = Array.from(selected);
    if (ids.length === 0) return;
    setExporting(true);
    setExportProgress(0);
    let ok = 0;
    const failed: string[] = [];
    for (const id of ids) {
      try {
        const data = await gateway.exportBenchmark(id);
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${id}.benchmark.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        ok += 1;
      } catch (e) {
        failed.push(id);
        // keep going — one bad export shouldn't abort the batch
         
        console.error(`export ${id} failed`, e);
      }
      setExportProgress((p) => p + 1);
      await new Promise((r) => setTimeout(r, 400));
    }
    setExporting(false);
    if (failed.length === 0) {
      toast.success(`Downloaded ${ok} export${ok === 1 ? "" : "s"}`, { duration: 3000 });
    } else {
      toast.error(`Downloaded ${ok} of ${ids.length} — ${failed.length} failed`, { duration: 4000 });
    }
  };

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
      await gateway.renameBenchmark(renameTarget.id, name);
      // Patch the row in place — no need to refetch the page for a label change.
      setItems((prev) => prev.map((b) => (b.id === renameTarget.id ? { ...b, name } : b)));
      setRenameTarget(null);
    } catch (e) {
      setRenameError(e instanceof Error ? e.message : String(e));
    } finally {
      setRenaming(false);
    }
  };

  // Flip one benchmark's public flag (owner-only; the row only offers this for
  // owned runs). Optimistic toast + refresh from the server.
  const onTogglePublic = async (bench: BenchmarkRecord) => {
    const next = !bench.is_public;
    try {
      await gateway.setBenchmarkVisibility(bench.id, next);
      toast.success(next ? "Benchmark is now public" : "Benchmark is now private", {
        duration: 2500,
      });
      load();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e), { duration: 4000 });
    }
  };

  // Bulk make public/private over the selected runs the caller owns (others are
  // skipped — the gateway would 403 them anyway).
  const onBulkVisibility = async (isPublic: boolean) => {
    const targets = items.filter((b) => selected.has(b.id) && (b.is_owner ?? true));
    if (targets.length === 0) {
      toast.error("None of the selected benchmarks are yours to change", { duration: 3000 });
      return;
    }
    const results = await Promise.allSettled(
      targets.map((b) => gateway.setBenchmarkVisibility(b.id, isPublic)),
    );
    const failures = results.filter((r) => r.status === "rejected").length;
    if (failures === 0) {
      toast.success(
        `${targets.length} benchmark${targets.length === 1 ? "" : "s"} now ${isPublic ? "public" : "private"}`,
        { duration: 2500 },
      );
    } else {
      toast.error(`${failures} of ${targets.length} failed`, { duration: 4000 });
    }
    exitSelect();
    load();
  };

  const onSingleDelete = async () => {
    if (!single) return;
    setSingleError(null);
    setSingleDeleting(true);
    try {
      await gateway.deleteBenchmark(single.id);
      setSingle(null);
      load();
      // Header count + dashboard stats are server-rendered; refresh them too.
      router.refresh();
    } catch (e) {
      setSingleError(e instanceof Error ? e.message : String(e));
    } finally {
      setSingleDeleting(false);
    }
  };

  const onDeleteSelected = async () => {
    if (selected.size === 0) return;
    setDeleting(true);
    setDeleteError(null);
    const ids = Array.from(selected);
    const results = await Promise.allSettled(ids.map((id) => gateway.deleteBenchmark(id)));
    const failures = results.filter((r) => r.status === "rejected").length;
    setDeleting(false);
    if (failures === 0) {
      setConfirmOpen(false);
      exitSelect();
    } else {
      setDeleteError(`${failures} of ${ids.length} failed to delete`);
    }
    load();
    // Header count + dashboard stats are server-rendered; refresh them too.
    router.refresh();
  };

  const hasFilter = q.trim().length > 0 || status !== "all";

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-2">
        <div className="relative min-w-[180px] flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="search"
            placeholder="Search by name, id, model, GPU, owner, status…"
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
        {selectMode ? (
          <button
            type="button"
            onClick={exitSelect}
            disabled={deleting || exporting}
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
        <div className="mb-3 flex flex-col gap-2 rounded-md border border-border bg-muted/40 px-3 py-2 text-sm sm:flex-row sm:items-center sm:justify-between">
          <span className="text-muted-foreground">
            {selected.size} selected
            {items.length > 0 && (
              <>
                {" "}
                <button
                  type="button"
                  onClick={() => setSelected(new Set(items.map((b) => b.id)))}
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
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={onExportSelected}
              disabled={selected.size === 0 || exporting || deleting}
              title="Download an export JSON for each selected benchmark"
              className="inline-flex items-center gap-1.5 rounded-md border border-input bg-background px-3 py-1.5 text-sm font-medium shadow-xs hover:bg-muted disabled:opacity-50"
            >
              <Download className="h-3.5 w-3.5" />
              {exporting
                ? `Exporting ${exportProgress}/${selected.size}…`
                : `Export ${selected.size > 0 ? selected.size : ""}`.trim()}
            </button>
            <button
              type="button"
              onClick={onCompare}
              disabled={selected.size < 2 || deleting || exporting}
              title={selected.size < 2 ? "Select 2 or more to compare" : "Compare selected"}
              className="inline-flex items-center gap-1.5 rounded-md border border-input bg-background px-3 py-1.5 text-sm font-medium shadow-xs hover:bg-muted disabled:opacity-50"
            >
              <GitCompare className="h-3.5 w-3.5" />
              {`Compare ${selected.size >= 2 ? selected.size : ""}`.trim()}
            </button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  disabled={selected.size === 0 || deleting || exporting}
                  title="More actions"
                  aria-label="More actions"
                  className="inline-flex items-center justify-center rounded-md border border-input bg-background px-2 py-1.5 text-sm shadow-xs hover:bg-muted disabled:opacity-50"
                >
                  <MoreHorizontal className="h-4 w-4" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onSelect={() => onBulkVisibility(true)}>
                  <Globe className="h-4 w-4" />
                  Make public
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => onBulkVisibility(false)}>
                  <Lock className="h-4 w-4" />
                  Make private
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  variant="destructive"
                  onSelect={(e) => {
                    e.preventDefault();
                    setConfirmOpen(true);
                  }}
                >
                  <Trash2 className="h-4 w-4" />
                  Delete {selected.size > 0 ? selected.size : ""}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      )}

      {hasFilter && (
        <div className="mb-3 flex items-center gap-2 text-xs text-muted-foreground">
          <span>
            {total} {total === 1 ? "run matches" : "runs match"}
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
          </span>
          {loading && <Loader2 className="h-3 w-3 animate-spin" />}
        </div>
      )}

      {items.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
          <Inbox className="h-6 w-6 text-muted-foreground/60" />
          <p className="text-sm text-muted-foreground">No benchmarks match your filters.</p>
        </div>
      ) : (
        <>
          <div
            className={cn(
              "gap-3",
              view === "rows"
                ? "flex flex-col"
                : "grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3",
              // Dim (but keep rendering) the stale page while the next one loads.
              loading && "pointer-events-none opacity-60",
            )}
          >
            {items.map((b) => {
              const owned = b.is_owner ?? true;
              return (
                <BenchmarkRow
                  key={b.id}
                  bench={b}
                  selectMode={selectMode}
                  selected={selected.has(b.id)}
                  onToggle={toggle}
                  // Only the owner can delete / change visibility; admins may also
                  // rename anyone's run (the gateway authorizes admin on rename).
                  onDelete={owned ? (bench) => setSingle(bench) : undefined}
                  onRename={
                    owned || isAdmin
                      ? (bench) => {
                          setRenameTarget(bench);
                          setRenameDraft(bench.name);
                          setRenameError(null);
                        }
                      : undefined
                  }
                  onTogglePublic={owned ? onTogglePublic : undefined}
                />
              );
            })}
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
              Delete {selected.size} benchmark{selected.size === 1 ? "" : "s"}?
            </DialogTitle>
            <DialogDescription>
              Kills any running subprocesses and removes the benchmark records. S3
              objects are kept. If a RunPod pod is still alive, terminate it
              manually from RunPod&apos;s dashboard.
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
              Kills any running subprocess and removes the benchmark record. S3
              objects are kept. If a RunPod pod is still alive, terminate it
              manually from RunPod&apos;s dashboard.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {singleError && (
              <p className="mr-auto text-sm text-destructive">{singleError}</p>
            )}
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
            <DialogTitle>Rename benchmark</DialogTitle>
            <DialogDescription>
              Updates the display name only. The run, S3 files, and config are
              unchanged.
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
            placeholder="Benchmark name"
            className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
          />
          <DialogFooter>
            {renameError && (
              <p className="mr-auto text-sm text-destructive">{renameError}</p>
            )}
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
