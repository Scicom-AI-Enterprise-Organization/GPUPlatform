"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState, useTransition } from "react";
import {
  CheckSquare,
  Cpu,
  Inbox,
  LayoutGrid,
  List,
  Loader2,
  MoreHorizontal,
  Search,
  Trash2,
  User,
  X,
} from "lucide-react";
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
import { SortSelect, sortByCreated, type SortDir } from "@/components/ui/sort-select";
import type { AppRecord } from "@/lib/types";
import { avatarFor } from "@/lib/avatar";
import { cn } from "@/lib/utils";
import { deleteEndpoint } from "./actions";

function searchableText(a: AppRecord): string {
  return [a.name, a.app_id, a.model, a.gpu, a.owner ?? ""].join(" ").toLowerCase();
}

export function EndpointGrid({ apps }: { apps: AppRecord[] }) {
  const router = useRouter();
  const [single, setSingle] = useState<AppRecord | null>(null);
  const [q, setQ] = useState("");
  const [selectMode, setSelectMode] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [sort, setSort] = useState<SortDir>("newest");
  const [view, setView] = useState<"rows" | "grid">("grid");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(12);
  useEffect(() => {
    const v = window.localStorage.getItem("sgpu_serverless_view");
    // Reading client-only localStorage post-mount avoids an SSR/CSR mismatch.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (v === "rows" || v === "grid") setView(v);
  }, []);
  const setViewPersist = (v: "rows" | "grid") => {
    setView(v);
    window.localStorage.setItem("sgpu_serverless_view", v);
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
    return apps.filter((a) => {
      if (tokens.length === 0) return true;
      const text = searchableText(a);
      return tokens.every((t) => text.includes(t));
    });
  }, [apps, q]);

  const sorted = useMemo(() => sortByCreated(filtered, sort), [filtered, sort]);

  const pageCount = Math.max(1, Math.ceil(sorted.length / pageSize));
  // Clamp in render so a shrinking result set never strands an empty page; the
  // search handlers reset to page 1 directly.
  const currentPage = Math.min(page, pageCount);
  const paged = sorted.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const onDeleteSelected = async () => {
    if (selected.size === 0) return;
    setDeleting(true);
    setDeleteError(null);
    const ids = Array.from(selected);
    const results = await Promise.allSettled(ids.map((id) => deleteEndpoint(id)));
    const failures = results.filter(
      (r) => r.status === "rejected" || (r.status === "fulfilled" && !r.value.ok),
    ).length;
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

  if (apps.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
        <Inbox className="h-6 w-6 text-muted-foreground/60" />
        <p className="text-sm text-muted-foreground">
          No endpoints yet. Click <span className="font-medium text-foreground">New endpoint</span> to spin one up.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-4 flex gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="search"
            placeholder="Search by name, id, model, GPU, owner…"
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
                  onClick={() => setSelected(new Set(paged.map((a) => a.app_id)))}
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

      {q && (
        <div className="mb-3 text-xs text-muted-foreground">
          {filtered.length} of {apps.length} match for{" "}
          <span className="font-mono text-foreground">&quot;{q}&quot;</span>
        </div>
      )}

      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
          <Inbox className="h-6 w-6 text-muted-foreground/60" />
          <p className="text-sm text-muted-foreground">No endpoints match your search.</p>
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
            {paged.map((app) => (
              <EndpointCard
                key={app.app_id}
                app={app}
                selectMode={selectMode}
                selected={selected.has(app.app_id)}
                onToggle={toggle}
                onDelete={() => setSingle(app)}
              />
            ))}
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
            itemLabel="endpoints"
          />
        </>
      )}

      <SingleDeleteDialog target={single} onClose={() => setSingle(null)} />

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
              Delete {selected.size} endpoint{selected.size === 1 ? "" : "s"}?
            </DialogTitle>
            <DialogDescription>
              All workers for each endpoint will be drained and queues cleared.
              This can&apos;t be undone.
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

function EndpointCard({
  app,
  selectMode,
  selected,
  onToggle,
  onDelete,
}: {
  app: AppRecord;
  selectMode: boolean;
  selected: boolean;
  onToggle: (id: string) => void;
  onDelete: () => void;
}) {
  const avatar = avatarFor(app.name);

  const inner = (
    <>
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          {selectMode && (
            <input
              type="checkbox"
              checked={selected}
              onChange={() => onToggle(app.app_id)}
              onClick={(e) => e.stopPropagation()}
              className="h-4 w-4 shrink-0 cursor-pointer accent-primary"
              aria-label={`Select ${app.name}`}
            />
          )}
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-border bg-muted/60 text-base font-semibold text-muted-foreground">
            {avatar.letter}
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="truncate font-medium text-foreground">{app.name}</span>
              <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-emerald-700 dark:text-emerald-400">
                Ready
              </span>
            </div>
            <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="truncate font-mono" title={app.app_id}>{app.app_id}</span>
              {app.owner && (
                <>
                  <span>·</span>
                  <User className="h-3 w-3" />
                  <span className="truncate">{app.owner}</span>
                </>
              )}
            </div>
          </div>
        </div>
        {!selectMode && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon-sm"
                className="-mr-1 shrink-0 text-muted-foreground hover:text-foreground"
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
                variant="destructive"
                onSelect={(e) => {
                  e.preventDefault();
                  onDelete();
                }}
              >
                <Trash2 className="h-4 w-4" />
                Delete endpoint
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs">
          <span className="font-mono">{app.model}</span>
        </span>
        <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs">
          <Cpu className="h-3 w-3 text-muted-foreground" />
          <span className="font-mono">
            {app.gpu}
            {app.gpu_count > 1 ? ` × ${app.gpu_count}` : ""}
          </span>
        </span>
        <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 font-mono text-xs">
          max {app.autoscaler.max_containers}
        </span>
      </div>

      <div className="mt-3 flex items-center justify-between border-t border-border/60 pt-2 text-xs text-muted-foreground">
        <span />
        <span title={new Date(app.created_at).toISOString()}>
          {new Date(app.created_at).toLocaleString()}
        </span>
      </div>
    </>
  );

  const base = "group block rounded-xl border border-border bg-card p-4 transition-all";

  if (selectMode) {
    return (
      <div
        role="button"
        tabIndex={0}
        onClick={() => onToggle(app.app_id)}
        onKeyDown={(e) => {
          if (e.key === " " || e.key === "Enter") {
            e.preventDefault();
            onToggle(app.app_id);
          }
        }}
        className={cn(
          base,
          "cursor-pointer",
          selected ? "border-primary/60 bg-primary/5" : "hover:border-primary/40 hover:bg-card/80",
        )}
      >
        {inner}
      </div>
    );
  }

  return (
    <Link
      href={`/serverless/${encodeURIComponent(app.app_id)}`}
      className={cn(base, "hover:border-primary/40 hover:bg-card/80 hover:shadow-md")}
    >
      {inner}
    </Link>
  );
}

function SingleDeleteDialog({
  target,
  onClose,
}: {
  target: AppRecord | null;
  onClose: () => void;
}) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const [err, setErr] = useState<string | null>(null);

  function handleDelete() {
    if (!target) return;
    setErr(null);
    startTransition(async () => {
      const res = await deleteEndpoint(target.app_id);
      if (!res.ok) {
        setErr(res.error);
        return;
      }
      onClose();
      router.refresh();
    });
  }

  return (
    <Dialog
      open={!!target}
      onOpenChange={(open) => {
        if (!open) {
          setErr(null);
          onClose();
        }
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete {target?.name}?</DialogTitle>
          <DialogDescription>
            All workers will be drained and the queue cleared. This can&apos;t be undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          {err && <p className="mr-auto text-sm text-destructive">{err}</p>}
          <Button variant="ghost" onClick={onClose} disabled={pending}>
            Cancel
          </Button>
          <Button variant="destructive" onClick={handleDelete} disabled={pending}>
            {pending && <Loader2 className="h-4 w-4 animate-spin" />}
            Delete endpoint
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

