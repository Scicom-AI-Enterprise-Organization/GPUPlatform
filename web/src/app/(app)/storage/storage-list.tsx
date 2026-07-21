"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Check, Cloud, Database, HardDrive, Inbox, KeyRound, LayoutGrid, List, Loader2, MoreHorizontal, Power, RefreshCw, Search, Sparkles, Trash2, TriangleAlert, User, X } from "lucide-react";
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
import type { PurgeJobStatus, PurgeScanResult, StorageKind, StorageRecord } from "@/lib/types";

const KIND_LABEL: Record<StorageKind, string> = {
  s3: "s3",
  huggingface: "huggingface",
  local: "local",
  sftp: "sftp",
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

function formatBytes(b?: number | null): string {
  if (b == null) return "—";
  let n = b;
  for (const u of ["B", "KB", "MB", "GB", "TB", "PB"]) {
    if (n < 1024) return `${n < 10 && u !== "B" ? n.toFixed(1) : Math.round(n)} ${u}`;
    n /= 1024;
  }
  return `${n.toFixed(1)} EB`;
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
  const [cleanup, setCleanup] = useState<StorageRecord | null>(null);
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
                      {s.purge_running && (
                        <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-700 dark:text-emerald-400">
                          <Loader2 className="h-2.5 w-2.5 animate-spin" /> cleaning…
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
                      {s.kind === "s3" && (
                        <DropdownMenuItem
                          onSelect={(e) => {
                            e.preventDefault();
                            setCleanup(s);
                          }}
                        >
                          <Sparkles className="h-4 w-4" />
                          Clean up storage
                        </DropdownMenuItem>
                      )}
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
                {s.kind === "s3" && s.total_size_bytes != null && (
                  <span
                    className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs text-muted-foreground"
                    title={`${(s.object_count ?? 0).toLocaleString()} objects${s.size_computed_at ? ` · as of ${new Date(s.size_computed_at).toLocaleString()}` : ""}`}
                  >
                    <HardDrive className="h-3 w-3" />
                    {formatBytes(s.total_size_bytes)}
                  </span>
                )}
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

      {cleanup && (
        <CleanupDialog
          storage={cleanup}
          onClose={() => setCleanup(null)}
          onChanged={() => router.refresh()}
        />
      )}
    </div>
  );
}

const CATEGORY_LABEL: Record<string, string> = {
  orphan: "Orphaned (owner deleted)",
  aged: "Aged (old, regenerable)",
};

/**
 * Manual storage cleanup: scan (dry-run) → review orphaned + aged object groups →
 * confirm delete. Auto-scans on open; nothing is deleted until the user ticks groups
 * and confirms, and the server re-validates each prefix at delete time.
 */
function CleanupDialog({
  storage,
  onClose,
  onChanged,
}: {
  storage: StorageRecord;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [scan, setScan] = useState<PurgeScanResult | null>(null);
  const [scanning, setScanning] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [ageDays, setAgeDays] = useState(30);
  const [confirm, setConfirm] = useState(false);
  const [starting, setStarting] = useState(false);
  const [job, setJob] = useState<PurgeJobStatus | null>(null); // running / done / error
  const [error, setError] = useState<string | null>(null);

  const runScan = useCallback(async (age = ageDays) => {
    setScanning(true);
    setError(null);
    setJob(null);
    try {
      const r = await gateway.storagePurgeScan(storage.id, age);
      setScan(r);
      // default: every purgeable group ticked
      setSelected(new Set(r.groups.filter((g) => g.purgeable).map((g) => g.prefix)));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setScanning(false);
    }
  }, [storage.id, ageDays]);

  useEffect(() => {
    // On open: resume an in-flight cleanup (so closing + reopening shows live
    // progress), else run a fresh dry-run scan.
    let cancelled = false;
    (async () => {
      try {
        const st = await gateway.storagePurgeStatus(storage.id);
        if (cancelled) return;
        if (st.state === "running") {
          setJob(st);
          return;
        }
      } catch {
        /* fall through to a scan */
      }
      if (!cancelled) void runScan();
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // Poll while a delete runs; refresh the list once it finishes (updates size +
    // clears the "cleaning…" badge).
    if (job?.state !== "running") return;
    const t = setInterval(async () => {
      try {
        const st = await gateway.storagePurgeStatus(storage.id);
        setJob(st);
        if (st.state !== "running") onChanged();
      } catch {
        /* transient — keep polling */
      }
    }, 1500);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.state, storage.id]);

  const purgeable = useMemo(() => (scan?.groups ?? []).filter((g) => g.purgeable), [scan]);
  const keptCount = (scan?.groups ?? []).length - purgeable.length;
  const selectedGroups = purgeable.filter((g) => selected.has(g.prefix));
  const selectedBytes = selectedGroups.reduce((a, g) => a + g.bytes, 0);
  const selectedObjects = selectedGroups.reduce((a, g) => a + g.objects, 0);

  const toggle = (prefix: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(prefix)) next.delete(prefix);
      else next.add(prefix);
      return next;
    });

  const onDelete = async () => {
    setStarting(true);
    setError(null);
    try {
      const st = await gateway.storagePurge(storage.id, [...selected], ageDays);
      setJob(st); // running — the poll effect takes over
      setConfirm(false);
      setScan(null);
      onChanged(); // reflect the "cleaning…" badge on the card
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  };

  const busy = scanning || starting;
  const jobPct =
    job && job.target_objects
      ? Math.min(100, Math.round((100 * (job.deleted_objects ?? 0)) / job.target_objects))
      : job?.state === "done"
        ? 100
        : 0;

  return (
    <Dialog open onOpenChange={(o) => !busy && !o && onClose()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="h-4 w-4" /> Clean up {storage.name}
          </DialogTitle>
          <DialogDescription>
            Finds objects safe to delete — <b>orphaned</b> (the dataset / run / job / app that
            owned them is gone) and <b>aged</b> (old artifacts in ephemeral folders). Live data
            and unrecognized files are never touched. Deletion is permanent.
          </DialogDescription>
        </DialogHeader>

        {/* progress view (a background delete is running or finished) */}
        {job ? (
          <div className="space-y-3 text-sm">
            {job.state === "running" ? (
              <p className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Deleting… {(job.deleted_objects ?? 0).toLocaleString()} /{" "}
                {(job.target_objects ?? 0).toLocaleString()} objects ·{" "}
                {formatBytes(job.freed_bytes)} / {formatBytes(job.target_bytes)} freed
              </p>
            ) : job.state === "error" ? (
              <p className="flex items-center gap-2 text-destructive">
                <TriangleAlert className="h-4 w-4" /> Cleanup failed after freeing{" "}
                {formatBytes(job.freed_bytes)}: {job.error}
              </p>
            ) : (
              <p className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400">
                <Check className="h-4 w-4" /> Freed {formatBytes(job.freed_bytes)} across{" "}
                {(job.deleted?.length ?? 0).toLocaleString()} group
                {(job.deleted?.length ?? 0) === 1 ? "" : "s"} (
                {(job.deleted_objects ?? 0).toLocaleString()} objects).
              </p>
            )}
            {/* progress bar */}
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className={cn(
                  "h-full rounded-full transition-all",
                  job.state === "error" ? "bg-destructive" : "bg-emerald-500",
                )}
                style={{ width: `${jobPct}%` }}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              {job.done_prefixes ?? 0} / {job.total_prefixes ?? 0} groups
              {job.state === "running" && (
                <> · runs in the background — you can close this dialog and it keeps deleting.</>
              )}
              {(job.skipped?.length ?? 0) > 0 && (
                <> · {job.skipped!.length} skipped (became live or already gone).</>
              )}
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
              <div
                className="flex items-center gap-2 text-muted-foreground"
                title="Ephemeral = benchmarks, quantization-jobs, serverless-logs. A group is flagged only if its NEWEST file is older than this — i.e. the whole group has sat untouched. Orphaned files (owner deleted) are always flagged regardless of age; datasets/ and training-runs/ are never aged-out."
              >
                <span>Also flag benchmark / quant-job / log files untouched for over</span>
                <select
                  value={ageDays}
                  onChange={(e) => {
                    const d = Number(e.target.value);
                    setAgeDays(d);
                    void runScan(d);
                  }}
                  disabled={busy}
                  className="h-7 rounded-md border border-input bg-background px-2 text-xs"
                >
                  {[7, 30, 90, 180, 365].map((d) => (
                    <option key={d} value={d}>{d} days</option>
                  ))}
                  <option value={0}>— off (orphans only) —</option>
                </select>
                <Button variant="outline" size="sm" className="h-7 text-xs" onClick={() => runScan()} disabled={busy}>
                  <RefreshCw className={cn("mr-1 h-3 w-3", scanning && "animate-spin")} /> Rescan
                </Button>
              </div>
              {scan && (
                <span className="text-muted-foreground">
                  total {formatBytes(scan.total_bytes)} · {scan.total_objects.toLocaleString()} objects
                </span>
              )}
            </div>

            {scanning ? (
              <div className="flex items-center justify-center gap-2 py-12 text-sm text-muted-foreground">
                <Loader2 className="h-5 w-5 animate-spin" /> scanning the bucket…
              </div>
            ) : scan && purgeable.length === 0 ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                Nothing to clean up — no orphaned or aged objects found
                {keptCount > 0 ? ` (${keptCount} live/kept groups).` : "."}
              </p>
            ) : scan ? (
              <>
                <div className="flex items-center justify-between text-xs">
                  <button
                    type="button"
                    className="text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
                    onClick={() =>
                      setSelected(
                        selected.size === purgeable.length ? new Set() : new Set(purgeable.map((g) => g.prefix)),
                      )
                    }
                  >
                    {selected.size === purgeable.length ? "deselect all" : "select all"}
                  </button>
                  <span className="text-muted-foreground">
                    reclaimable {formatBytes(scan.reclaimable_bytes)} · {keptCount} kept
                  </span>
                </div>
                <div className="max-h-72 space-y-1 overflow-auto rounded-md border border-border p-1 scrollbar-thin">
                  {purgeable.map((g) => (
                    <label
                      key={g.prefix}
                      className="flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 text-xs hover:bg-muted/50"
                    >
                      <input
                        type="checkbox"
                        checked={selected.has(g.prefix)}
                        onChange={() => toggle(g.prefix)}
                        className="h-3.5 w-3.5"
                      />
                      <span
                        className={cn(
                          "shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium",
                          g.category === "orphan"
                            ? "bg-amber-500/10 text-amber-700 dark:text-amber-400"
                            : "bg-sky-500/10 text-sky-700 dark:text-sky-400",
                        )}
                        title={CATEGORY_LABEL[g.category]}
                      >
                        {g.category}
                      </span>
                      <span className="min-w-0 flex-1 truncate font-mono" title={`${g.prefix} — ${g.reason}`}>
                        {g.prefix}
                      </span>
                      <span className="shrink-0 tabular-nums text-muted-foreground">{g.objects}</span>
                      <span className="w-16 shrink-0 text-right tabular-nums">{formatBytes(g.bytes)}</span>
                    </label>
                  ))}
                </div>
              </>
            ) : null}
          </div>
        )}

        <DialogFooter>
          {error && <p className="mr-auto max-w-[60%] truncate text-sm text-destructive" title={error}>{error}</p>}
          {job ? (
            job.state === "running" ? (
              <Button onClick={onClose}>Run in background</Button>
            ) : (
              <Button onClick={onClose}>Done</Button>
            )
          ) : confirm ? (
            <>
              <span className="mr-auto text-sm text-destructive">
                Permanently delete {selectedGroups.length} group{selectedGroups.length === 1 ? "" : "s"} ({formatBytes(selectedBytes)})?
              </span>
              <Button variant="outline" onClick={() => setConfirm(false)} disabled={starting}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={onDelete} disabled={starting}>
                {starting ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <Trash2 className="mr-1 h-4 w-4" />}
                Delete permanently
              </Button>
            </>
          ) : (
            <>
              <Button variant="outline" onClick={onClose} disabled={busy}>
                Close
              </Button>
              <Button
                variant="destructive"
                onClick={() => setConfirm(true)}
                disabled={busy || selectedGroups.length === 0}
              >
                <TriangleAlert className="mr-1 h-4 w-4" />
                Delete selected · {formatBytes(selectedBytes)} ({selectedObjects.toLocaleString()} obj)
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
