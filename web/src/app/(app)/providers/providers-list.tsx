"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { Cloud, Copy, Cpu, Inbox, KeyRound, LayoutGrid, List, MoreHorizontal, Search, Server, Trash2, User, X } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { shortGpu } from "@/lib/gpu-format";
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
import type { ProviderRecord } from "@/lib/types";

function searchableText(p: ProviderRecord): string {
  return [p.name, p.id, p.kind, p.host ?? "", p.user ?? "", p.account_email ?? "", p.created_by ?? "", ...(p.gpus ?? [])]
    .join(" ")
    .toLowerCase();
}

export function ProvidersList({ items }: { items: ProviderRecord[] }) {
  const router = useRouter();
  const [target, setTarget] = useState<ProviderRecord | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [testing, setTesting] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<Record<string, string>>({});
  const [showPub, setShowPub] = useState<Record<string, boolean>>({});
  const [q, setQ] = useState("");
  const [view, setView] = useState<"rows" | "grid">("grid");
  useEffect(() => {
    const v = window.localStorage.getItem("sgpu_providers_view");
    // Reading client-only localStorage post-mount avoids an SSR/CSR mismatch.
    if (v === "rows" || v === "grid") setView(v);
  }, []);
  const setViewPersist = (v: "rows" | "grid") => {
    setView(v);
    window.localStorage.setItem("sgpu_providers_view", v);
  };

  const filtered = useMemo(() => {
    const tokens = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    if (!tokens.length) return items;
    return items.filter((p) => {
      const text = searchableText(p);
      return tokens.every((t) => text.includes(t));
    });
  }, [items, q]);

  const onCopyPub = async (id: string, pub: string) => {
    try {
      await navigator.clipboard.writeText(pub);
      setTestResult((prev) => ({ ...prev, [id]: "OK · public key copied" }));
    } catch {
      // ignore
    }
  };

  const onDelete = async () => {
    if (!target) return;
    setError(null);
    setDeleting(true);
    try {
      await gateway.deleteProvider(target.id);
      setTarget(null);
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleting(false);
    }
  };

  const onRetest = async (p: ProviderRecord) => {
    setTesting(p.id);
    setTestResult((prev) => ({ ...prev, [p.id]: "" }));
    try {
      const r = await gateway.testProvider({ kind: p.kind, provider_id: p.id });
      setTestResult((prev) => ({
        ...prev,
        [p.id]: r.ok ? `OK · ${r.message}` : `FAIL · ${r.message}`,
      }));
      if (r.ok) router.refresh();
    } catch (e) {
      setTestResult((prev) => ({
        ...prev,
        [p.id]: `FAIL · ${e instanceof Error ? e.message : String(e)}`,
      }));
    } finally {
      setTesting(null);
    }
  };

  return (
    <div>
      <div className="mb-4 flex gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="search"
            placeholder="Search by name, id, kind, host, GPU…"
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
          <p className="text-sm text-muted-foreground">No providers match your search.</p>
        </div>
      ) : (
      <ul className={cn("gap-3", view === "rows" ? "flex flex-col" : "grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3")}>
        {filtered.map((p) => (
          <li
            key={p.id}
            className="rounded-xl border border-border bg-card p-4 transition-all hover:border-primary/40 hover:bg-card/80 hover:shadow-md"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex min-w-0 items-center gap-3">
                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-border bg-muted/60 text-base font-semibold text-muted-foreground">
                  {avatarFor(p.name).letter}
                </div>
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="truncate font-medium text-foreground">{p.name}</span>
                    <span className="inline-flex items-center gap-1 rounded-full border border-border bg-muted px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                      {p.kind === "vm" ? (
                        <Server className="h-3 w-3" />
                      ) : (
                        <Cloud className="h-3 w-3" />
                      )}
                      {p.kind === "pi" ? "prime intellect" : p.kind}
                    </span>
                  </div>
                  <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
                    <span className="truncate font-mono" title={p.id}>{p.id}</span>
                    <span>·</span>
                    <User className="h-3 w-3" />
                    <span className="truncate">{p.created_by}</span>
                  </div>
                </div>
              </div>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon-sm" className="-mr-1 shrink-0 text-muted-foreground hover:text-foreground" aria-label="Actions">
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem
                    onSelect={(e) => {
                      e.preventDefault();
                      onRetest(p);
                    }}
                    disabled={testing === p.id}
                  >
                    <Cpu className="h-4 w-4" />
                    {testing === p.id ? "Testing…" : "Re-test"}
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    variant="destructive"
                    onSelect={(e) => {
                      e.preventDefault();
                      setTarget(p);
                      setError(null);
                    }}
                  >
                    <Trash2 className="h-4 w-4" />
                    Delete provider
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              {p.kind === "vm" && p.host && (
                <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 font-mono text-xs">
                  {p.user}@{p.host}:{p.port}
                </span>
              )}
              {p.kind === "vm" && p.gpu_count != null && p.gpu_count > 0 && (
                <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs">
                  <Cpu className="h-3 w-3 text-muted-foreground" />
                  <span className="font-mono">
                    {(p.gpus ?? []).slice(0, 1).map(shortGpu).join("")}
                    {p.gpu_count > 1 ? ` × ${p.gpu_count}` : ""}
                  </span>
                </span>
              )}
              {p.kind === "vm" && (p.gpu_count == null || p.gpu_count === 0) && (
                <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs text-muted-foreground">
                  not yet probed
                </span>
              )}
              {(p.kind === "runpod" || p.kind === "pi") && p.api_key_last4 && (
                <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 font-mono text-xs">
                  <KeyRound className="h-3 w-3 text-muted-foreground" />
                  ****{p.api_key_last4}
                </span>
              )}
              {p.account_email && (
                <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs text-muted-foreground">
                  {p.account_email}
                </span>
              )}
            </div>

            {(p.kind === "runpod" || p.kind === "pi") && p.ssh_pub && (
              <div className="mt-2 text-xs">
                <button
                  type="button"
                  className="text-muted-foreground hover:text-foreground"
                  onClick={() =>
                    setShowPub((prev) => ({ ...prev, [p.id]: !prev[p.id] }))
                  }
                >
                  {showPub[p.id] ? "Hide" : "Show"} SSH pubkey
                </button>
                {showPub[p.id] && (
                  <div className="mt-1 flex items-start gap-2 rounded-md bg-muted/50 p-2 font-mono text-[11px]">
                    <span className="flex-1 break-all">{p.ssh_pub}</span>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-sm"
                      onClick={() => onCopyPub(p.id, p.ssh_pub!)}
                      aria-label="Copy public key"
                    >
                      <Copy className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                )}
              </div>
            )}

            {testResult[p.id] && (
              <div
                className={
                  "mt-3 border-t border-border/60 pt-2 text-xs " +
                  (testResult[p.id].startsWith("OK") ? "text-emerald-600 dark:text-emerald-400" : "text-destructive")
                }
              >
                {testResult[p.id]}
              </div>
            )}

            <div className="mt-3 flex items-center justify-end border-t border-border/60 pt-2 text-xs text-muted-foreground">
              <span title={new Date(p.created_at).toISOString()}>
                added {new Date(p.created_at).toLocaleString()}
              </span>
            </div>
          </li>
        ))}
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
              Removes the provider record from this account. Workloads already
              referencing it will fall back to the platform default. The remote
              VM is not touched.
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

