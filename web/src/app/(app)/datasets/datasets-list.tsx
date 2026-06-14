"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Inbox, LayoutGrid, List, Loader2, Search, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { DatasetKind, DatasetRecord } from "@/lib/types";
import { Button } from "@/components/ui/button";
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
import { DatasetCard, KIND_LABEL } from "./dataset-card";

/** Flat searchable string per dataset — name, id, kind, source ref, storage,
 * format, owner — so one query hits any of them. */
function searchableText(d: DatasetRecord): string {
  return [
    d.name,
    d.id,
    d.kind,
    KIND_LABEL[d.kind],
    d.hf_repo,
    d.s3_metadata_uri,
    d.metadata_filename,
    d.storage_name,
    d.format,
    d.created_by,
    d.description,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

const SOURCE_OPTIONS: Array<{ value: "all" | DatasetKind; label: string }> = [
  { value: "all", label: "All sources" },
  { value: "upload", label: "Upload" },
  { value: "s3", label: "S3" },
  { value: "hf", label: "HuggingFace" },
  { value: "label", label: "Labeling" },
  { value: "hosted", label: "HF repo" },
];

export function DatasetsList({ items }: { items: DatasetRecord[] }) {
  const router = useRouter();
  const [q, setQ] = useState("");
  const [source, setSource] = useState<"all" | DatasetKind>("all");
  const [sort, setSort] = useState<SortDir>("newest");
  const [view, setView] = useState<"rows" | "grid">("grid");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(12);

  // Per-item dialogs
  const [deleteTarget, setDeleteTarget] = useState<DatasetRecord | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [renameTarget, setRenameTarget] = useState<DatasetRecord | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);

  useEffect(() => {
    const v = window.localStorage.getItem("sgpu_datasets_view");
    // Reading client-only localStorage post-mount is the correct way to avoid an
    // SSR/CSR mismatch — a lazy initializer would diverge on hydrate.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (v === "rows" || v === "grid") setView(v);
  }, []);
  const setViewPersist = (v: "rows" | "grid") => {
    setView(v);
    window.localStorage.setItem("sgpu_datasets_view", v);
  };

  const haystacks = useMemo(() => items.map((d) => ({ d, text: searchableText(d) })), [items]);
  const filtered = useMemo(() => {
    const tokens = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return haystacks
      .filter(({ d, text }) => {
        if (source !== "all" && d.kind !== source) return false;
        return tokens.every((t) => text.includes(t));
      })
      .map(({ d }) => d);
  }, [haystacks, q, source]);

  const sorted = useMemo(() => sortByCreated(filtered, sort), [filtered, sort]);

  const hasFilter = q.trim().length > 0 || source !== "all";
  const pageCount = Math.max(1, Math.ceil(sorted.length / pageSize));
  // Clamp in render so a shrinking result set can't strand us on an empty page;
  // searching/filtering resets to page 1 via the change handlers below.
  const currentPage = Math.min(page, pageCount);
  const paged = sorted.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const onDelete = async () => {
    if (!deleteTarget) return;
    setDeleteError(null);
    setDeleting(true);
    try {
      // A "hosted" item is a HF-mirror catalog repo, not an Autotrain dataset.
      if (deleteTarget.kind === "hosted") {
        await gateway.deleteCatalogRepo(deleteTarget.id);
      } else {
        await gateway.deleteDataset(deleteTarget.id);
      }
      setDeleteTarget(null);
      router.refresh();
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleting(false);
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
      await gateway.updateDataset(renameTarget.id, { name });
      setRenameTarget(null);
      router.refresh();
    } catch (e) {
      setRenameError(e instanceof Error ? e.message : String(e));
    } finally {
      setRenaming(false);
    }
  };

  return (
    <div>
      <div className="mb-4 flex gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="search"
            placeholder="Search by name, id, source, storage, owner…"
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
          value={source}
          onValueChange={(v) => {
            setSource(v as "all" | DatasetKind);
            setPage(1);
          }}
        >
          <SelectTrigger className="h-10! w-[150px]" title="Filter by source">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SOURCE_OPTIONS.map((s) => (
              <SelectItem key={s.value} value={s.value}>
                {s.label}
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
          {filtered.length} of {items.length} match
          {q && (
            <>
              {" "}for <span className="font-mono text-foreground">&quot;{q}&quot;</span>
            </>
          )}
          {source !== "all" && (
            <>
              {" "}· source <span className="font-mono text-foreground">{source}</span>
            </>
          )}
        </div>
      )}

      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
          <Inbox className="h-6 w-6 text-muted-foreground/60" />
          <p className="text-sm text-muted-foreground">
            {items.length === 0 ? "No datasets yet." : "No datasets match your filters."}
          </p>
        </div>
      ) : (
        <>
          <div
            className={cn(
              "gap-3",
              view === "rows" ? "flex flex-col" : "grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3",
            )}
          >
            {paged.map((d) => (
              <DatasetCard
                key={d.id}
                dataset={d}
                onRename={(ds) => {
                  setRenameTarget(ds);
                  setRenameDraft(ds.name);
                  setRenameError(null);
                }}
                onDelete={(ds) => {
                  setDeleteTarget(ds);
                  setDeleteError(null);
                }}
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
            itemLabel="datasets"
          />
        </>
      )}

      <Dialog
        open={!!deleteTarget}
        onOpenChange={(o) => {
          if (!deleting && !o) {
            setDeleteTarget(null);
            setDeleteError(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete dataset</DialogTitle>
            <DialogDescription>
              Delete <span className="font-medium text-foreground">{deleteTarget?.name}</span>? This removes the
              dataset record. Files already written to storage are not deleted.
            </DialogDescription>
          </DialogHeader>
          {deleteError && (
            <p className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {deleteError}
            </p>
          )}
          <DialogFooter>
            <Button variant="ghost" onClick={() => setDeleteTarget(null)} disabled={deleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={onDelete} disabled={deleting}>
              {deleting ? "Deleting…" : "Delete"}
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
            <DialogTitle>Rename dataset</DialogTitle>
            <DialogDescription>Updates the display name only. The source and files are unchanged.</DialogDescription>
          </DialogHeader>
          <input
            autoFocus
            value={renameDraft}
            onChange={(e) => setRenameDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !renaming && renameDraft.trim()) onRename();
            }}
            disabled={renaming}
            maxLength={255}
            placeholder="Dataset name"
            className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
          />
          <DialogFooter>
            {renameError && <p className="mr-auto text-sm text-destructive">{renameError}</p>}
            <Button variant="outline" onClick={() => setRenameTarget(null)} disabled={renaming}>
              Cancel
            </Button>
            <Button onClick={onRename} disabled={renaming || !renameDraft.trim()}>
              {renaming ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              {renaming ? "Saving…" : "Save"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
