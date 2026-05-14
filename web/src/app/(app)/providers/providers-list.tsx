"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Cloud, Copy, Cpu, KeyRound, MoreHorizontal, Server, Trash2, User } from "lucide-react";
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
import type { ProviderRecord } from "@/lib/types";

export function ProvidersList({ items }: { items: ProviderRecord[] }) {
  const router = useRouter();
  const [target, setTarget] = useState<ProviderRecord | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [testing, setTesting] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<Record<string, string>>({});
  const [showPub, setShowPub] = useState<Record<string, boolean>>({});

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
      <ul className="flex flex-col gap-3">
        {items.map((p) => (
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

function shortGpu(gpu: string): string {
  return gpu
    .replace(/^NVIDIA\s+/i, "")
    .replace(/\s+GeForce\s+/i, " ")
    .replace(/^GeForce\s+/i, "");
}
