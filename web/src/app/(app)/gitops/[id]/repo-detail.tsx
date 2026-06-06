"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { AlertCircle, GitBranch, Loader2, RefreshCw, Save, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { gateway } from "@/lib/gateway";
import { cn } from "@/lib/utils";
import type { GitopsRepo, GitopsResource, GitopsSyncResult } from "@/lib/types";
import { SyncStatusPill, fmtWhen } from "../gitops-shared";

// Where each managed resource's detail page lives, when one exists.
function hrefFor(kind: string, id: string | null): string | null {
  if (!id) return null;
  switch (kind) {
    case "app": return `/serverless/${encodeURIComponent(id)}`;
    case "dataset": return `/datasets/${encodeURIComponent(id)}`;
    case "benchmark": return `/benchmark/${encodeURIComponent(id)}`;
    case "training_run": return `/autotrain/${encodeURIComponent(id)}`;
    case "provider": return `/providers`;
    case "storage": return `/storage`;
    default: return null;
  }
}

export function RepoDetail({
  initialRepo,
  initialResources,
}: {
  initialRepo: GitopsRepo;
  initialResources: GitopsResource[];
}) {
  const router = useRouter();
  const repo = initialRepo;
  const resources = initialResources;

  const [syncing, setSyncing] = useState(false);
  const [syncMsg, setSyncMsg] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveErr, setSaveErr] = useState<string | null>(null);

  // editable settings
  const [branch, setBranch] = useState(repo.branch);
  const [path, setPath] = useState(repo.path ?? "");
  const [poll, setPoll] = useState(String(repo.poll_interval));
  const [tokenSecret, setTokenSecret] = useState(repo.token_secret ?? "");
  const [prune, setPrune] = useState(repo.prune);
  const [enabled, setEnabled] = useState(repo.enabled);

  const onSync = async () => {
    setSyncing(true);
    setSyncMsg(null);
    try {
      const r: GitopsSyncResult = await gateway.syncGitopsRepo(repo.id);
      if (r.skipped) setSyncMsg("Already up to date.");
      else {
        const parts = [
          ...r.created.map((x) => `+ ${x}`),
          ...r.updated.map((x) => `~ ${x}`),
          ...r.pruned.map((x) => `- ${x}`),
        ];
        setSyncMsg(
          (parts.length ? parts.join("\n") : "No changes.") +
            (r.errors.length ? `\n\nErrors:\n${r.errors.join("\n")}` : ""),
        );
      }
      router.refresh();
    } catch (e) {
      setSyncMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setSyncing(false);
    }
  };

  const onSave = async () => {
    setSaving(true);
    setSaveErr(null);
    try {
      await gateway.updateGitopsRepo(repo.id, {
        branch: branch.trim() || "main",
        path: path.trim() || null,
        poll_interval: Number(poll) || 300,
        token_secret: tokenSecret.trim() || null,
        prune,
        enabled,
      });
      router.refresh();
    } catch (e) {
      setSaveErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const onDelete = async (doPrune: boolean) => {
    const what = doPrune
      ? `Delete "${repo.name}" AND prune its ${repo.resource_count} managed resource(s)?`
      : `Unregister "${repo.name}"? Managed resources are left running.`;
    if (!confirm(what)) return;
    try {
      await gateway.deleteGitopsRepo(repo.id, doPrune);
      router.push("/gitops");
      router.refresh();
    } catch (e) {
      setSaveErr(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="space-y-4">
      {/* header */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight">
            {repo.name}
            <SyncStatusPill status={repo.last_sync_status} />
          </h1>
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted-foreground">
            <span className="inline-flex items-center gap-1 font-mono">
              <GitBranch className="h-3.5 w-3.5" />
              {repo.url}<span className="text-foreground">@{repo.branch}</span>{repo.path ? `/${repo.path}` : ""}
            </span>
            <span>·</span>
            <span>synced {fmtWhen(repo.last_sync_at)}</span>
            {repo.last_synced_sha && (
              <>
                <span>·</span>
                <span className="font-mono">{repo.last_synced_sha.slice(0, 12)}</span>
              </>
            )}
          </div>
        </div>
        <Button onClick={onSync} disabled={syncing}>
          {syncing ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          Sync now
        </Button>
      </div>

      {repo.last_sync_error && (
        <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <pre className="whitespace-pre-wrap break-all font-mono text-xs">{repo.last_sync_error}</pre>
        </div>
      )}

      {syncMsg && (
        <div className="rounded-md border border-border bg-muted/50 px-3 py-2">
          <pre className="whitespace-pre-wrap break-all font-mono text-xs text-muted-foreground">{syncMsg}</pre>
        </div>
      )}

      {/* managed resources */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">
            Managed resources <span className="text-[11px] font-normal text-muted-foreground">· {resources.length}</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {resources.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Nothing applied yet. Hit <span className="font-medium text-foreground">Sync now</span>.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-muted-foreground">
                    <th className="px-2 py-1.5 font-medium">Kind</th>
                    <th className="px-2 py-1.5 font-medium">Name</th>
                    <th className="px-2 py-1.5 font-medium">Resource</th>
                    <th className="px-2 py-1.5 font-medium">Status</th>
                    <th className="px-2 py-1.5 font-medium">Synced</th>
                  </tr>
                </thead>
                <tbody>
                  {resources.map((r) => {
                    const href = hrefFor(r.kind, r.resource_id);
                    return (
                      <tr key={r.id} className="border-b border-border/60 last:border-0">
                        <td className="px-2 py-1.5 font-mono text-xs">{r.kind}</td>
                        <td className="px-2 py-1.5">{r.name}{r.generation > 1 ? <span className="ml-1 text-[10px] text-muted-foreground">gen {r.generation}</span> : null}</td>
                        <td className="px-2 py-1.5 font-mono text-xs">
                          {r.resource_id ? (
                            href ? <Link href={href} className="hover:underline">{r.resource_id}</Link> : r.resource_id
                          ) : <span className="text-muted-foreground">—</span>}
                        </td>
                        <td className="px-2 py-1.5">
                          <span className={cn(
                            "rounded px-1.5 py-0.5 text-[11px] font-medium",
                            r.status === "applied"
                              ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                              : "bg-destructive/10 text-destructive",
                          )}>
                            {r.status}
                          </span>
                          {r.error && <span className="ml-2 text-xs text-destructive" title={r.error}>{r.error.slice(0, 60)}</span>}
                        </td>
                        <td className="px-2 py-1.5 text-xs text-muted-foreground">{fmtWhen(r.last_synced_at)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* settings */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div>
              <Label htmlFor="branch">Branch</Label>
              <Input id="branch" value={branch} onChange={(e) => setBranch(e.target.value)} />
            </div>
            <div>
              <Label htmlFor="path">Path</Label>
              <Input id="path" value={path} onChange={(e) => setPath(e.target.value)} placeholder="(repo root)" />
            </div>
            <div>
              <Label htmlFor="poll">Poll interval (s)</Label>
              <Input id="poll" type="number" min={30} value={poll} onChange={(e) => setPoll(e.target.value)} />
            </div>
            <div>
              <Label htmlFor="token">Token secret key</Label>
              <Input id="token" value={tokenSecret} onChange={(e) => setTokenSecret(e.target.value)} placeholder="(none)" />
            </div>
          </div>
          <div className="flex items-center justify-between gap-4 border-t border-border pt-3">
            <div>
              <Label className="text-sm">Prune</Label>
              <p className="text-[11px] text-muted-foreground">Delete resources removed from the repo.</p>
            </div>
            <Switch checked={prune} onCheckedChange={setPrune} />
          </div>
          <div className="flex items-center justify-between gap-4">
            <div>
              <Label className="text-sm">Enabled</Label>
              <p className="text-[11px] text-muted-foreground">Auto-poll on the interval.</p>
            </div>
            <Switch checked={enabled} onCheckedChange={setEnabled} />
          </div>
          {saveErr && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">{saveErr}</div>
          )}
          <div className="flex items-center justify-between gap-2 border-t border-border pt-3">
            <Button onClick={onSave} disabled={saving} variant="outline" size="sm">
              {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
              Save settings
            </Button>
            <div className="flex gap-2">
              <Button onClick={() => onDelete(false)} variant="ghost" size="sm">Unregister</Button>
              <Button onClick={() => onDelete(true)} variant="ghost" size="sm" className="text-destructive hover:text-destructive">
                <Trash2 className="h-4 w-4" /> Delete + prune
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
