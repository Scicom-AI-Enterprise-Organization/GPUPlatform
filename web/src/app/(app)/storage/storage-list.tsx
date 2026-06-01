"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { Cloud, Database, Inbox, KeyRound, LayoutGrid, List, MoreHorizontal, Power, Search, Trash2, User, X } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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
import { avatarFor } from "@/lib/avatar";
import { cn } from "@/lib/utils";
import type { StorageKind, StorageRecord } from "@/lib/types";

const KIND_LABEL: Record<StorageKind, string> = {
  s3: "s3",
  huggingface: "huggingface",
};

function searchableText(s: StorageRecord): string {
  return [s.name, s.id, s.kind, s.bucket ?? "", s.prefix ?? "", s.region ?? "", s.endpoint ?? "", s.notes ?? "", s.created_by ?? ""]
    .join(" ")
    .toLowerCase();
}

function KindIcon({ kind }: { kind: StorageKind }) {
  return kind === "huggingface" ? <Database className="h-3 w-3" /> : <Cloud className="h-3 w-3" />;
}

/** Display the bucket scope (bucket + optional prefix), or the endpoint for
 * S3-compatible providers, or — for HF — just the kind. */
function scopeOf(s: StorageRecord): string | null {
  if (s.kind === "huggingface") return null;
  if (!s.bucket) return null;
  const prefix = s.prefix ? `/${s.prefix.replace(/^\/+|\/+$/g, "")}` : "";
  return `s3://${s.bucket}${prefix}`;
}

export function StorageList({
  items,
  canWrite,
}: {
  items: StorageRecord[];
  canWrite: boolean;
}) {
  const router = useRouter();
  const [target, setTarget] = useState<StorageRecord | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [view, setView] = useState<"rows" | "grid">("grid");
  useEffect(() => {
    const v = window.localStorage.getItem("sgpu_storage_view");
    // Reading client-only localStorage post-mount avoids an SSR/CSR mismatch.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (v === "rows" || v === "grid") setView(v);
  }, []);
  const setViewPersist = (v: "rows" | "grid") => {
    setView(v);
    window.localStorage.setItem("sgpu_storage_view", v);
  };

  const filtered = useMemo(() => {
    const tokens = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    if (!tokens.length) return items;
    return items.filter((s) => {
      const text = searchableText(s);
      return tokens.every((t) => text.includes(t));
    });
  }, [items, q]);

  const onDelete = async () => {
    if (!target) return;
    setError(null);
    setDeleting(true);
    try {
      await gateway.deleteStorage(target.id);
      setTarget(null);
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleting(false);
    }
  };

  const onToggle = async (s: StorageRecord) => {
    setBusy(s.id);
    try {
      await gateway.updateStorage(s.id, { enabled: !s.enabled });
      router.refresh();
    } catch {
      // surfaced on next load; keep the row as-is
    } finally {
      setBusy(null);
    }
  };

  return (
    <div>
      <div className="mb-4 flex gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="search"
            placeholder="Search by name, id, bucket, region…"
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

      {q && (
        <div className="mb-3 text-xs text-muted-foreground">
          {filtered.length} of {items.length} match for{" "}
          <span className="font-mono text-foreground">&quot;{q}&quot;</span>
        </div>
      )}

      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
          <Inbox className="h-6 w-6 text-muted-foreground/60" />
          <p className="text-sm text-muted-foreground">No storage matches your search.</p>
        </div>
      ) : (
      <ul className={cn("gap-3", view === "rows" ? "flex flex-col" : "grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3")}>
        {filtered.map((s) => {
          const scope = scopeOf(s);
          return (
            <li
              key={s.id}
              className={
                "rounded-xl border border-border bg-card p-4 transition-all hover:border-primary/40 hover:bg-card/80 hover:shadow-md" +
                (s.enabled ? "" : " opacity-60")
              }
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-center gap-3">
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-border bg-muted/60 text-base font-semibold text-muted-foreground">
                    {avatarFor(s.name).letter}
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="truncate font-medium text-foreground">{s.name}</span>
                      <span className="inline-flex items-center gap-1 rounded-full border border-border bg-muted px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                        <KindIcon kind={s.kind} />
                        {KIND_LABEL[s.kind]}
                      </span>
                      {!s.enabled && (
                        <span className="inline-flex items-center rounded-full border border-border bg-muted px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                          disabled
                        </span>
                      )}
                    </div>
                    <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
                      <span className="truncate font-mono" title={s.id}>{s.id}</span>
                      <span>·</span>
                      <User className="h-3 w-3" />
                      <span className="truncate">{s.created_by}</span>
                    </div>
                  </div>
                </div>
                {canWrite && (
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon-sm" className="-mr-1 shrink-0 text-muted-foreground hover:text-foreground" aria-label="Actions" disabled={busy === s.id}>
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onSelect={(e) => {
                          e.preventDefault();
                          onToggle(s);
                        }}
                        disabled={busy === s.id}
                      >
                        <Power className="h-4 w-4" />
                        {s.enabled ? "Disable" : "Enable"}
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        variant="destructive"
                        onSelect={(e) => {
                          e.preventDefault();
                          setTarget(s);
                          setError(null);
                        }}
                      >
                        <Trash2 className="h-4 w-4" />
                        Delete storage
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                )}
              </div>

              {s.notes && <p className="mt-2 text-xs text-muted-foreground">{s.notes}</p>}

              <div className="mt-3 flex flex-wrap items-center gap-1.5">
                {scope && (
                  <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 font-mono text-xs">
                    {scope}
                  </span>
                )}
                {s.kind === "s3" && s.region && (
                  <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs text-muted-foreground">
                    {s.region}
                  </span>
                )}
                {s.kind === "s3" && s.endpoint && (
                  <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 font-mono text-xs text-muted-foreground" title={s.endpoint}>
                    {s.endpoint.replace(/^https?:\/\//, "")}
                  </span>
                )}
                <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs text-muted-foreground">
                  <KeyRound className="h-3 w-3" />
                  {s.has_credentials ? "stored key" : "env fallback"}
                </span>
              </div>

              <div className="mt-3 flex items-center justify-end border-t border-border/60 pt-2 text-xs text-muted-foreground">
                <span title={new Date(s.created_at).toISOString()}>
                  added {new Date(s.created_at).toLocaleString()}
                </span>
              </div>
            </li>
          );
        })}
      </ul>
      )}

      <Dialog
        open={!!target}
        onOpenChange={(o) => {
          if (!deleting && !o) {
            setTarget(null);
            setError(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete {target?.name}?</DialogTitle>
            <DialogDescription>
              Removes the storage record and its stored credentials. The remote
              bucket / repo and any data already written there are untouched.
              Features referencing it will fall back to env defaults.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {error && <p className="mr-auto text-sm text-destructive">{error}</p>}
            <Button variant="outline" onClick={() => setTarget(null)} disabled={deleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={onDelete} disabled={deleting}>
              {deleting ? "Deleting…" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
