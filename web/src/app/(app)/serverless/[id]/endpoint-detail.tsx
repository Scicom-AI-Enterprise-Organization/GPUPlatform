"use client";

import { useEffect, useState, useTransition } from "react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Eraser, Globe, Loader2, Lock, RotateCw, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { AppRecord } from "@/lib/types";
import { gateway, type AppStatus } from "@/lib/gateway";
import { avatarFor } from "@/lib/avatar";
import { deleteEndpoint, purgeWorkers, restartEndpoint } from "../actions";
import { OverviewTab } from "./tabs/overview";
import { RequestsTab } from "./tabs/requests";
import { StressTab } from "./tabs/stress";
import { QueueTab } from "./tabs/queue";
import { WorkersTab } from "./tabs/workers";
import { VisualTab } from "./tabs/visual";
import { MetricsTab } from "./tabs/metrics";
import { ProxyTab } from "./tabs/proxy";

const TABS = [
  { value: "overview", label: "Overview" },
  { value: "playground", label: "Playground" },
  { value: "stress", label: "Stress test" },
  { value: "queue", label: "Queue" },
  { value: "workers", label: "Workers" },
  { value: "visual", label: "Visual" },
  { value: "metrics", label: "Metrics" },
  { value: "proxy", label: "Proxy" },
] as const;

type EndpointTab = (typeof TABS)[number]["value"];

// Tabs a non-owner (read-only) viewer of a PUBLIC endpoint may see. Playground,
// Stress, and Queue are excluded — inference is owner-only server-side, and the
// queue/requests views expose the owner's traffic.
const READONLY_TABS = new Set<string>(["overview", "workers", "visual", "metrics", "proxy"]);

export function EndpointDetail({ app, readOnly = false, isAdmin = false }: { app: AppRecord; readOnly?: boolean; isAdmin?: boolean }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  // Active tab is derived from the URL (?tab=) and each trigger is a real <Link>,
  // so right-click / middle-click / ⌘-click "open in new tab" works. A normal
  // click still switches in place (useSearchParams is reactive to soft nav).
  // proxy endpoints (single-model VM, no queue) hide the Queue tab — requests are
  // forwarded straight to the model, nothing is ever enqueued.
  const modeTabs = app.mode === "proxy" ? TABS.filter((t) => t.value !== "queue") : TABS;
  const visibleTabs = readOnly ? modeTabs.filter((t) => READONLY_TABS.has(t.value)) : modeTabs;
  const visibleValues = visibleTabs.map((t) => t.value) as readonly string[];
  const tabParam = searchParams.get("tab");
  const tab: EndpointTab = tabParam && visibleValues.includes(tabParam) ? (tabParam as EndpointTab) : "overview";
  const tabHref = (v: EndpointTab) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("tab", v);
    return `${pathname}?${params.toString()}`;
  };
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [pending, startTransition] = useTransition();
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [confirmRedeploy, setConfirmRedeploy] = useState(false);
  const [redeployPending, startRedeploy] = useTransition();
  const [redeployError, setRedeployError] = useState<string | null>(null);
  const [confirmPurge, setConfirmPurge] = useState(false);
  const [purgePending, startPurge] = useTransition();
  const [purgeError, setPurgeError] = useState<string | null>(null);
  // Persistent inline result of the last worker op (redeploy/purge) — shown under
  // the header buttons. NOT a toast (those vanish); stays until the next op.
  const [opResult, setOpResult] = useState<{ ok: boolean; text: string } | null>(null);
  const [isPublic, setIsPublic] = useState(app.is_public ?? false);
  const [togglingPublic, setTogglingPublic] = useState(false);
  const avatar = avatarFor(app.name);

  async function handleTogglePublic() {
    setTogglingPublic(true);
    try {
      const updated = await gateway.setAppVisibility(app.app_id, !isPublic);
      setIsPublic(updated.is_public ?? !isPublic);
      setOpResult({
        ok: true,
        text: updated.is_public
          ? "Endpoint is now public — read-only visible to every logged-in user. They can view the overview/workers/metrics but can't edit, delete, or run inference."
          : "Endpoint is now private — only you (and admins) can see it.",
      });
      router.refresh();
    } catch (e) {
      setOpResult({ ok: false, text: `Visibility change failed: ${e instanceof Error ? e.message : String(e)}` });
    } finally {
      setTogglingPublic(false);
    }
  }

  function handleDelete() {
    setDeleteError(null);
    startTransition(async () => {
      const res = await deleteEndpoint(app.app_id);
      if (!res.ok) {
        setDeleteError(res.error);
        return;
      }
      router.push("/serverless");
    });
  }

  function handleRedeploy() {
    setRedeployError(null);
    startRedeploy(async () => {
      const res = await restartEndpoint(app.app_id);
      if (!res.ok) {
        setRedeployError(res.error);
        setOpResult({ ok: false, text: `Redeploy failed: ${res.error}` });
        return;
      }
      setOpResult({ ok: true, text: `Redeploy started — drained ${res.drained} worker(s). Models cold-start over the next minute.` });
      setConfirmRedeploy(false);
      router.refresh();
    });
  }

  function handlePurge() {
    setPurgeError(null);
    startPurge(async () => {
      const res = await purgeWorkers(app.app_id);
      if (!res.ok) {
        setPurgeError(res.error);
        setOpResult({ ok: false, text: `Purge failed: ${res.error}` });
        return;
      }
      setOpResult({ ok: true, text: `Purged ${res.purged} stale worker(s); terminated ${res.terminated} tracked. The autoscaler will bring up a fresh worker unless the fleet is paused.` });
      setConfirmPurge(false);
      router.refresh();
    });
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="border-b border-border bg-sidebar/40 px-6 pt-4 lg:px-10">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-muted/60 text-lg font-semibold text-muted-foreground">
              {avatar.letter}
            </div>
            <div>
              <h1 className="text-xl font-semibold tracking-tight">{app.name}</h1>
              <div className="mt-0.5 flex items-center gap-3 text-xs text-muted-foreground">
                <span className="font-mono">{app.app_id}</span>
                <span>·</span>
                <span className="font-mono">{app.model}</span>
                {isPublic && (
                  <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-emerald-700 dark:text-emerald-400">
                    Public
                  </span>
                )}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {readOnly ? (
              <span className="rounded-md border border-border bg-muted/40 px-2.5 py-1 text-xs text-muted-foreground">
                Read-only{app.owner ? ` · ${app.owner}` : ""}
              </span>
            ) : (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleTogglePublic}
                  disabled={togglingPublic}
                  title={isPublic ? "Make private — only you and admins can see it" : "Make public — read-only visible to every logged-in user"}
                >
                  {togglingPublic ? <Loader2 className="h-4 w-4 animate-spin" /> : isPublic ? <Globe className="h-4 w-4" /> : <Lock className="h-4 w-4" />}
                  {isPublic ? "Public" : "Make public"}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setConfirmRedeploy(true)}
                >
                  <RotateCw className="h-4 w-4" />
                  Redeploy
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setConfirmPurge(true)}
                  title="Hard-clean stale worker pidfiles + orphan processes on the box"
                >
                  <Eraser className="h-4 w-4" />
                  Purge PIDs
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setConfirmDelete(true)}
                  className="text-destructive hover:text-destructive"
                >
                  <Trash2 className="h-4 w-4" />
                  Delete
                </Button>
              </>
            )}
          </div>
        </div>

        {opResult && (
          <div
            className={`mt-3 flex items-start justify-between gap-3 rounded-md border px-3 py-2 text-sm ${
              opResult.ok
                ? "border-emerald-500/30 bg-emerald-500/5 text-emerald-700 dark:text-emerald-300"
                : "border-destructive/30 bg-destructive/5 text-destructive"
            }`}
          >
            <span className="break-words">{opResult.text}</span>
            <button
              type="button"
              onClick={() => setOpResult(null)}
              className="shrink-0 opacity-70 hover:opacity-100"
              aria-label="Dismiss"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        <KpiBar app={app} />

        <Tabs value={tab} className="mt-2">
          <TabsList variant="line" className="bg-transparent">
            {visibleTabs.map((t) => (
              <TabsTrigger key={t.value} value={t.value} asChild>
                <Link href={tabHref(t.value)} scroll={false}>{t.label}</Link>
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 scrollbar-thin">
        <Tabs value={tab}>
          <TabsContent value="overview"><OverviewTab app={app} readOnly={readOnly} /></TabsContent>
          {!readOnly && <TabsContent value="playground"><RequestsTab app={app} /></TabsContent>}
          {!readOnly && <TabsContent value="stress"><StressTab app={app} /></TabsContent>}
          {!readOnly && app.mode !== "proxy" && <TabsContent value="queue"><QueueTab app={app} /></TabsContent>}
          <TabsContent value="workers"><WorkersTab app={app} /></TabsContent>
          <TabsContent value="visual"><VisualTab app={app} /></TabsContent>
          <TabsContent value="metrics"><MetricsTab app={app} /></TabsContent>
          <TabsContent value="proxy"><ProxyTab app={app} readOnly={readOnly} isAdmin={isAdmin} /></TabsContent>
        </Tabs>
      </div>

      <Dialog
        open={confirmRedeploy}
        onOpenChange={(o) => {
          setConfirmRedeploy(o);
          if (!o) setRedeployError(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Redeploy {app.name}?</DialogTitle>
            <DialogDescription>
              Re-provisions the worker with the latest worker-agent code and current config. In-flight requests drain
              first, then the fleet reloads — there&apos;s a brief cold start while models come back up.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {redeployError && (
              <p className="mr-auto text-sm text-destructive">{redeployError}</p>
            )}
            <Button variant="ghost" onClick={() => setConfirmRedeploy(false)} disabled={redeployPending}>
              Cancel
            </Button>
            <Button onClick={handleRedeploy} disabled={redeployPending}>
              {redeployPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <RotateCw className="h-4 w-4" />}
              Redeploy
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={confirmPurge}
        onOpenChange={(o) => {
          setConfirmPurge(o);
          if (!o) setPurgeError(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Purge worker PIDs for {app.name}?</DialogTitle>
            <DialogDescription>
              Drains + terminates every worker for this endpoint, then sweeps <strong>all</strong> of its
              stale pidfiles, logs, and orphan vLLM processes off the box (the leftovers a redis-blip
              crash-loop leaves behind, which Redeploy doesn&apos;t clean), and clears Redis state. Only this
              endpoint&apos;s workers are touched. The autoscaler then brings up a fresh worker (unless the
              fleet is killed/paused).
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {purgeError && <p className="mr-auto text-sm text-destructive">{purgeError}</p>}
            <Button variant="ghost" onClick={() => setConfirmPurge(false)} disabled={purgePending}>
              Cancel
            </Button>
            <Button onClick={handlePurge} disabled={purgePending}>
              {purgePending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Eraser className="h-4 w-4" />}
              Purge PIDs
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={confirmDelete}
        onOpenChange={(o) => {
          setConfirmDelete(o);
          if (!o) setDeleteError(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete {app.name}?</DialogTitle>
            <DialogDescription>
              All workers will be drained and the queue cleared. This can&apos;t be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {deleteError && (
              <p className="mr-auto text-sm text-destructive">{deleteError}</p>
            )}
            <Button variant="ghost" onClick={() => setConfirmDelete(false)} disabled={pending}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={pending}>
              {pending && <Loader2 className="h-4 w-4 animate-spin" />}
              Delete endpoint
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function KpiBar({ app }: { app: AppRecord }) {
  // Live worker + queue counts from /apps/{id}/status (was hardcoded "0").
  const [status, setStatus] = useState<AppStatus | null>(null);
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const s = await gateway.getAppStatus(app.app_id);
        if (!cancelled) setStatus(s);
      } catch {
        // best-effort; KpiBar falls back to 0 while unreachable
      }
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, [app.app_id]);

  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
      <Kpi value={String(status?.workers ?? 0)} label="running workers" />
      <Kpi value={String(status?.queue_len ?? 0)} label="requests waiting in queue" />
      <Kpi
        value={String(Math.min(1, app.autoscaler.max_containers))}
        label="active worker recommended"
      />
    </div>
  );
}

function Kpi({ value, label }: { value: string; label: string }) {
  return (
    <span className="text-muted-foreground">
      <span className="font-mono text-foreground">{value}</span> {label}
    </span>
  );
}
