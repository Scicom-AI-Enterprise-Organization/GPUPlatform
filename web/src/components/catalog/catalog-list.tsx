"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Boxes, Database, Files, LayoutGrid, List, Loader2, Lock, Package, Search, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { CatalogRecord } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";
import { SortSelect, sortByCreated, type SortDir } from "@/components/ui/sort-select";

export function fmtBytes(n?: number | null): string {
  if (!n && n !== 0) return "—";
  if (n < 1024) return `${n} B`;
  const u = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(1)} ${u[i]}`;
}

type View = "rows" | "grid";

/** A single-type catalog list with search + grid/list view (mirrors Datasets).
 * `detailBase` is the route a card links to, e.g. "/models" or "/datasets/hosted". */
export function CatalogList({
  items,
  detailBase,
  noun = "repo",
}: {
  items: CatalogRecord[];
  detailBase: string;
  noun?: string;
}) {
  const router = useRouter();
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<SortDir>("newest");
  const [view, setView] = useState<View>("grid");
  const [deleteTarget, setDeleteTarget] = useState<CatalogRecord | null>(null);
  const [wipe, setWipe] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Hydrate the saved view post-mount (start from the default so SSR matches).
  useEffect(() => {
    const v = window.localStorage.getItem("sgpu_catalog_view");
    // Reading client-only localStorage post-mount avoids an SSR/CSR mismatch.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (v === "rows" || v === "grid") setView(v);
  }, []);
  const setViewPersist = (v: View) => {
    setView(v);
    window.localStorage.setItem("sgpu_catalog_view", v);
  };

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const matched = !needle
      ? items
      : items.filter((r) =>
          [r.full_id, r.id, r.storage_name, r.description, r.created_by]
            .filter(Boolean)
            .join(" ")
            .toLowerCase()
            .includes(needle),
        );
    return sortByCreated(matched, sort);
  }, [items, q, sort]);

  async function confirmDelete() {
    if (!deleteTarget) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await gateway.deleteCatalogRepo(deleteTarget.id, wipe);
      setDeleteTarget(null);
      setWipe(false);
      router.refresh();
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleting(false);
    }
  }

  const askDelete = (r: CatalogRecord) => {
    setDeleteTarget(r);
    setWipe(false);
    setDeleteError(null);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={`Search ${noun}s…`}
            className="h-10 w-full rounded-md border border-input bg-background pl-9 pr-3 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
          />
        </div>
        <SortSelect value={sort} onValueChange={setSort} />
        <div className="inline-flex h-10 items-stretch overflow-hidden rounded-md border border-input bg-background shadow-xs">
          <button
            onClick={() => setViewPersist("rows")}
            className={cn("inline-flex items-center justify-center px-2.5 transition-colors", view === "rows" ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/50")}
            title="List view"
            aria-label="List view"
            aria-pressed={view === "rows"}
          >
            <List className="h-4 w-4" />
          </button>
          <button
            onClick={() => setViewPersist("grid")}
            className={cn("inline-flex items-center justify-center border-l border-input px-2.5 transition-colors", view === "grid" ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/50")}
            title="Grid view"
            aria-label="Grid view"
            aria-pressed={view === "grid"}
          >
            <LayoutGrid className="h-4 w-4" />
          </button>
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="px-2 py-10 text-center text-sm text-muted-foreground">No {noun}s match.</div>
      ) : (
        <div className={cn(view === "grid" ? "grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3" : "flex flex-col gap-2")}>
          {filtered.map((r) => (
            <RepoTile key={r.id} r={r} view={view} detailBase={detailBase} noun={noun} onDelete={askDelete} />
          ))}
        </div>
      )}

      <Dialog open={!!deleteTarget} onOpenChange={(o) => !o && setDeleteTarget(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete {noun}</DialogTitle>
            <DialogDescription>
              Remove <span className="font-mono">{deleteTarget?.full_id}</span> from the catalog.
            </DialogDescription>
          </DialogHeader>
          <label className="flex items-start gap-2 rounded-md border border-border px-3 py-2 text-sm">
            <Checkbox checked={wipe} onCheckedChange={(v) => setWipe(v === true)} className="mt-0.5" />
            <span>
              Also delete all files from storage (
              <span className="font-mono text-xs">{deleteTarget?.prefix}</span>). Permanent.
            </span>
          </label>
          {deleteError && <p className="text-sm text-destructive">{deleteError}</p>}
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteTarget(null)} disabled={deleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmDelete} disabled={deleting}>
              {deleting && <Loader2 className="h-4 w-4 animate-spin" />}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function RepoTile({
  r,
  view,
  detailBase,
  noun,
  onDelete,
}: {
  r: CatalogRecord;
  view: View;
  detailBase: string;
  noun: string;
  onDelete: (r: CatalogRecord) => void;
}) {
  const href = `${detailBase}/${r.namespace}/${r.name}`;
  const Icon = r.repo_type === "dataset" ? Database : Package;
  const meta = (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
      <span className="inline-flex items-center gap-1">
        <Files className="h-3 w-3" /> {r.num_files ?? 0} files · {fmtBytes(r.size_bytes)}
      </span>
      {r.storage_name && (
        <span className="inline-flex items-center gap-1">
          <Boxes className="h-3 w-3" /> {r.storage_name}
        </span>
      )}
      <span>by {r.created_by}</span>
    </div>
  );
  const title = (
    <div className="flex items-center gap-2">
      <span className="truncate font-mono text-sm font-medium">{r.full_id}</span>
      {r.private && (
        <Badge variant="secondary" className="shrink-0">
          <Lock className="h-3 w-3" /> private
        </Badge>
      )}
    </div>
  );
  const del = (
    <Button
      variant="ghost"
      size="icon"
      className="shrink-0 text-muted-foreground hover:text-destructive"
      onClick={() => onDelete(r)}
      title={`Delete ${noun}`}
    >
      <Trash2 className="h-4 w-4" />
    </Button>
  );

  if (view === "grid") {
    return (
      <div className="group flex flex-col gap-2 rounded-lg border border-border bg-card p-4 transition-colors hover:border-foreground/20">
        <div className="flex items-start gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground">
            <Icon className="h-4 w-4" />
          </div>
          <Link href={href} className="min-w-0 flex-1">
            {title}
          </Link>
          {del}
        </div>
        <Link href={href}>{meta}</Link>
      </div>
    );
  }
  return (
    <div className="group flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-3 transition-colors hover:border-foreground/20">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground">
        <Icon className="h-4 w-4" />
      </div>
      <Link href={href} className="min-w-0 flex-1">
        {title}
        <div className="mt-0.5">{meta}</div>
      </Link>
      {del}
    </div>
  );
}
