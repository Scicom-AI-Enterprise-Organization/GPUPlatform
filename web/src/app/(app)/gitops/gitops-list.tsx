"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { GitBranch, Loader2, RefreshCw, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { gateway } from "@/lib/gateway";
import type { GitopsRepo, GitopsSyncResult } from "@/lib/types";
import { SyncStatusPill, fmtWhen } from "./gitops-shared";

export function GitopsList({ items }: { items: GitopsRepo[] }) {
  const router = useRouter();
  const [busy, setBusy] = useState<Record<string, "sync" | "delete">>({});
  const [msg, setMsg] = useState<Record<string, string>>({});

  const setRowBusy = (id: string, v: "sync" | "delete" | null) =>
    setBusy((b) => {
      const n = { ...b };
      if (v) n[id] = v;
      else delete n[id];
      return n;
    });

  const onSync = async (id: string) => {
    setRowBusy(id, "sync");
    setMsg((m) => ({ ...m, [id]: "" }));
    try {
      const r: GitopsSyncResult = await gateway.syncGitopsRepo(id);
      const parts = [
        r.created.length ? `+${r.created.length}` : "",
        r.updated.length ? `~${r.updated.length}` : "",
        r.pruned.length ? `-${r.pruned.length}` : "",
      ].filter(Boolean);
      setMsg((m) => ({
        ...m,
        [id]: r.skipped
          ? "already up to date"
          : r.errors.length
            ? `${r.errors.length} error(s): ${r.errors[0]}`
            : `applied ${parts.join(" ") || "no changes"}`,
      }));
      router.refresh();
    } catch (e) {
      setMsg((m) => ({ ...m, [id]: e instanceof Error ? e.message : String(e) }));
    } finally {
      setRowBusy(id, null);
    }
  };

  const onDelete = async (repo: GitopsRepo, prune: boolean) => {
    const what = prune
      ? `Delete "${repo.name}" AND prune its ${repo.resource_count} managed resource(s)? This deletes the live apps/storage/etc.`
      : `Unregister "${repo.name}"? Managed resources are left running (orphaned).`;
    if (!confirm(what)) return;
    setRowBusy(repo.id, "delete");
    try {
      await gateway.deleteGitopsRepo(repo.id, prune);
      router.refresh();
    } catch (e) {
      setMsg((m) => ({ ...m, [repo.id]: e instanceof Error ? e.message : String(e) }));
      setRowBusy(repo.id, null);
    }
  };

  return (
    <ul className="divide-y divide-border rounded-lg border border-border">
      {items.map((repo) => {
        const b = busy[repo.id];
        return (
          <li key={repo.id} className="flex flex-col gap-2 px-4 py-3 sm:flex-row sm:items-center sm:gap-4">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <Link href={`/gitops/${repo.id}`} className="truncate font-medium hover:underline">
                  {repo.name}
                </Link>
                <SyncStatusPill status={repo.last_sync_status} />
                {!repo.enabled && (
                  <span className="rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] uppercase text-muted-foreground">
                    paused
                  </span>
                )}
                {repo.prune && (
                  <span className="rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[10px] uppercase text-amber-600 dark:text-amber-400">
                    prune
                  </span>
                )}
              </div>
              <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-muted-foreground">
                <span className="inline-flex items-center gap-1 font-mono">
                  <GitBranch className="h-3 w-3" />
                  {repo.url}
                  <span className="text-foreground">@{repo.branch}</span>
                  {repo.path ? <span>/{repo.path}</span> : null}
                </span>
                <span>·</span>
                <span>{repo.resource_count} resources</span>
                <span>·</span>
                <span>synced {fmtWhen(repo.last_sync_at)}</span>
                {repo.last_synced_sha && (
                  <>
                    <span>·</span>
                    <span className="font-mono">{repo.last_synced_sha.slice(0, 8)}</span>
                  </>
                )}
              </div>
              {msg[repo.id] && (
                <p className="mt-1 text-xs text-muted-foreground">{msg[repo.id]}</p>
              )}
            </div>

            <div className="flex shrink-0 items-center gap-2">
              <Button size="sm" variant="outline" disabled={!!b} onClick={() => onSync(repo.id)}>
                {b === "sync" ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                Sync now
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button size="sm" variant="ghost" disabled={!!b}>⋯</Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem asChild>
                    <Link href={`/gitops/${repo.id}`}>Open</Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => onDelete(repo, false)}>
                    Unregister (keep resources)
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    className="text-destructive focus:text-destructive"
                    onClick={() => onDelete(repo, true)}
                  >
                    <Trash2 className="h-4 w-4" /> Delete + prune resources
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
