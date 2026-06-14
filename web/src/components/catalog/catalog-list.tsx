"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Boxes, Database, Files, Loader2, Lock, Package, Search, Trash2 } from "lucide-react";
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

/** A single-type catalog list (models on /models, datasets on /datasets/hosted).
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
  const [deleteTarget, setDeleteTarget] = useState<CatalogRecord | null>(null);
  const [wipe, setWipe] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return items;
    return items.filter((r) =>
      [r.full_id, r.id, r.storage_name, r.description, r.created_by]
        .filter(Boolean)
        .join(" ")
        .toLowerCase()
        .includes(needle),
    );
  }, [items, q]);

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

  return (
    <div className="space-y-3">
      {items.length > 6 && (
        <div className="relative max-w-sm">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={`Search ${noun}s…`}
            className="h-9 w-full rounded-md border border-input bg-transparent pl-8 pr-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
          />
        </div>
      )}

      <ul className="space-y-2">
        {filtered.map((r) => (
          <li key={r.id}>
            <div className="group flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-3 transition-colors hover:border-foreground/20">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground">
                {r.repo_type === "dataset" ? <Database className="h-4 w-4" /> : <Package className="h-4 w-4" />}
              </div>
              <Link href={`${detailBase}/${r.id}`} className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="truncate font-mono text-sm font-medium">{r.full_id}</span>
                  {r.private && (
                    <Badge variant="secondary" className="shrink-0">
                      <Lock className="h-3 w-3" /> private
                    </Badge>
                  )}
                </div>
                <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
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
              </Link>
              <Button
                variant="ghost"
                size="icon"
                className="shrink-0 text-muted-foreground hover:text-destructive"
                onClick={() => {
                  setDeleteTarget(r);
                  setWipe(false);
                  setDeleteError(null);
                }}
                title={`Delete ${noun}`}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </li>
        ))}
        {filtered.length === 0 && (
          <li className="px-2 py-8 text-center text-sm text-muted-foreground">No {noun}s match.</li>
        )}
      </ul>

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
