"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import yaml from "js-yaml";
import { CheckSquare, Download, GitCompare, Globe, Inbox, LayoutGrid, List, Lock, MoreHorizontal, Search, Trash2, X } from "lucide-react";
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
import { SortSelect, sortByCreated, type SortDir } from "@/components/ui/sort-select";
import { BenchmarkRow } from "./benchmark-row";

/** Pre-compute a flat searchable string per benchmark. Includes name, id,
 * status, owner, model, GPU type, and parallelism so a single query can hit
 * any of those. Done once per render via useMemo. */
function searchableText(b: BenchmarkRecord): string {
  let model = "";
  let gpu = "";
  let parallelism = "";
  try {
    const cfg = yaml.load(b.config_yaml) as
      | {
          runpod?: { pod?: { gpu_type?: string; gpu_count?: number } };
          benchmark?: Array<{
            model?: { repo_id?: string };
            serve?: { tensor_parallel_size?: number; data_parallel_size?: number };
          }>;
        }
      | null;
    gpu = cfg?.runpod?.pod?.gpu_type ?? "";
    model = cfg?.benchmark?.[0]?.model?.repo_id ?? "";
    const tp = cfg?.benchmark?.[0]?.serve?.tensor_parallel_size ?? 1;
    const dp = cfg?.benchmark?.[0]?.serve?.data_parallel_size ?? 1;
    parallelism = `tp${tp} dp${dp} tp${tp}/dp${dp}`;
  } catch {
    // ignore
  }
  return [b.name, b.id, b.status, b.created_by, model, gpu, parallelism]
    .join(" ")
    .toLowerCase();
}

const STATUS_OPTIONS = ["all", "queued", "running", "done", "failed", "cancelled"] as const;
type StatusFilter = (typeof STATUS_OPTIONS)[number];

export function BenchmarkList({ items }: { items: BenchmarkRecord[] }) {
  const router = useRouter();
  const [q, setQ] = useState("");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [sort, setSort] = useState<SortDir>("newest");
  const [selectMode, setSelectMode] = useState(false);
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
  const [view, setView] = useState<"rows" | "grid">("grid");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(12);
  useEffect(() => {
    const v = window.localStorage.getItem("sgpu_bench_view");
    // Reading client-only localStorage post-mount avoids an SSR/CSR mismatch.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (v === "rows" || v === "grid") setView(v);
  }, []);
  const setViewPersist = (v: "rows" | "grid") => {
    setView(v);
    window.localStorage.setItem("sgpu_bench_view", v);
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
        // eslint-disable-next-line no-console
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
      setRenameTarget(null);
      router.refresh();
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
      router.refresh();
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
    router.refresh();
  };

  const onSingleDelete = async () => {
    if (!single) return;
    setSingleError(null);
    setSingleDeleting(true);
    try {
      await gateway.deleteBenchmark(single.id);
      setSingle(null);
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
      router.refresh();
    } else {
      setDeleteError(`${failures} of ${ids.length} failed to delete`);
      router.refresh();
    }
  };

  const haystacks = useMemo(
    () => items.map((b) => ({ bench: b, text: searchableText(b) })),
    [items],
  );

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const tokens = needle ? needle.split(/\s+/).filter(Boolean) : [];
    return haystacks
      .filter(({ bench, text }) => {
        if (status !== "all" && bench.status !== status) return false;
        if (tokens.length === 0) return true;
        return tokens.every((t) => text.includes(t));
      })
      .map(({ bench }) => bench);
  }, [haystacks, q, status]);

  const sorted = useMemo(() => sortByCreated(filtered, sort), [filtered, sort]);

  const hasFilter = q.trim().length > 0 || status !== "all";

  const pageCount = Math.max(1, Math.ceil(sorted.length / pageSize));
  // Clamp in render so a shrinking result set never strands an empty page; the
  // search/filter handlers reset to page 1 directly.
  const currentPage = Math.min(page, pageCount);
  const paged = sorted.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  return (
    <div>
      <div className="mb-4 flex gap-2">
        <div className="relative flex-1">
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
        <div className="mb-3 flex items-center justify-between rounded-md border border-border bg-muted/40 px-3 py-2 text-sm">
          <span className="text-muted-foreground">
            {selected.size} selected
            {filtered.length > 0 && (
              <>
                {" "}
                <button
                  type="button"
                  onClick={() => setSelected(new Set(paged.map((b) => b.id)))}
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
          <div className="flex items-center gap-2">
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
            )}
          >
            {paged.map((b) => {
              const owned = b.is_owner ?? true;
              return (
                <BenchmarkRow
                  key={b.id}
                  bench={b}
                  selectMode={selectMode}
                  selected={selected.has(b.id)}
                  onToggle={toggle}
                  // Only the owner can mutate a run; public runs from others are
                  // read-only (no delete/rename/visibility actions offered).
                  onDelete={owned ? (bench) => setSingle(bench) : undefined}
                  onRename={
                    owned
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
            total={filtered.length}
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
