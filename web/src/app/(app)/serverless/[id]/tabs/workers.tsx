"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { AlertTriangle, Ban, ChevronDown, ChevronRight, ExternalLink, FileText, Loader2, Moon, RefreshCw, RotateCw, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { formatCostUSD, useLiveCost } from "@/lib/cost";
import { BurnFlame } from "@/components/burn-flame";
import type { AppRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

type WorkerStatus = "running" | "initializing" | "terminating" | "terminated" | "unknown";

type WorkerRow = {
  machine_id: string;
  pod_id: string;
  status: WorkerStatus;
  raw_status: string;
  region: string;
  region_code: string;
  gpu: string;
  gpu_count: number;
  vcpus: number;
  ram_gb: number;
  disk_gb: number;
  created_at: string | null;
  cost_per_hr: number | null;
};

type ApiResponse = { workers: WorkerRow[]; prefix: string; error?: string };

const STATUS_STYLES: Record<WorkerStatus, string> = {
  running:      "bg-status-active/15 text-status-active",
  initializing: "bg-status-idle/15 text-status-idle",
  terminating:  "bg-status-down/15 text-status-down",
  terminated:   "bg-muted text-muted-foreground",
  unknown:      "bg-muted text-muted-foreground",
};

const POLL_MS = 10_000;
const STORAGE_KEY = (appId: string) => `serverless-ui:workers:${appId}`;

export function WorkersTab({ app }: { app: AppRecord }) {
  // The meaningful unit for a multi-model endpoint is the model fleet (each
  // member's resident/asleep state + GPUs), from the status endpoint. A VM multi
  // endpoint has no pods, so that's all. A *cloud* (RunPod) multi endpoint also
  // runs on a real pod — show the pods table too (container link + alive status).
  const isVm = app.gpu === "vm";
  if (app.mode === "multi") {
    if (isVm) return <MultiModelFleet app={app} />;
    return (
      <div className="space-y-4">
        <MultiModelFleet app={app} />
        <RunpodWorkersTab app={app} />
      </div>
    );
  }
  return <RunpodWorkersTab app={app} />;
}

function RunpodWorkersTab({ app }: { app: AppRecord }) {
  const [live, setLive] = useState<WorkerRow[] | null>(null);
  const [remembered, setRemembered] = useState<WorkerRow[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  // Gateway-side liveness keyed by machine_id: is the worker actually registered
  // + heartbeating, vs a pod that's "running" on RunPod but never phoned home.
  const [aliveById, setAliveById] = useState<Record<string, { alive: boolean; status: string }>>({});

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY(app.app_id));
      if (raw) setRemembered(JSON.parse(raw));
    } catch {
      // ignore
    }
  }, [app.app_id]);

  const fetchLive = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const r = await fetch(`/api/runpod/pods?app=${encodeURIComponent(app.app_id)}`, {
        cache: "no-store",
      });
      const body = (await r.json()) as ApiResponse;
      if (!r.ok) throw new Error(body?.error ?? r.statusText);
      setLive(body.workers);
      // Best-effort: which of these machine_ids are actually registered +
      // heartbeating to the gateway (vs a pod RunPod calls "running" that never
      // phoned home). The pod table still renders if this fetch fails.
      try {
        const ar = await fetch(`/api/proxy/apps/${encodeURIComponent(app.app_id)}/workers`, { cache: "no-store" });
        if (ar.ok) {
          const aw = (await ar.json()) as { machine_id: string; alive: boolean; status: string }[];
          const m: Record<string, { alive: boolean; status: string }> = {};
          for (const w of aw) m[w.machine_id] = { alive: w.alive, status: w.status };
          setAliveById(m);
        }
      } catch {
        /* liveness is best-effort */
      }

      setRemembered((prev) => {
        const map = new Map(prev.map((w) => [w.machine_id, w]));
        for (const w of body.workers) map.set(w.machine_id, w);
        const merged = Array.from(map.values());
        try {
          window.localStorage.setItem(STORAGE_KEY(app.app_id), JSON.stringify(merged));
        } catch {
          // best-effort persist
        }
        return merged;
      });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [app.app_id]);

  useEffect(() => {
    fetchLive();
    const id = window.setInterval(fetchLive, POLL_MS);
    return () => window.clearInterval(id);
  }, [fetchLive]);

  const rows = useMemo(() => {
    // Until the first fetch lands, we don't actually know which cached
    // workers are still alive — show them with their last-known status
    // instead of flashing "terminated" before the real data arrives.
    if (live === null) {
      const order: WorkerStatus[] = ["running", "initializing", "terminating", "unknown", "terminated"];
      return [...remembered].sort((a, b) => {
        const oa = order.indexOf(a.status);
        const ob = order.indexOf(b.status);
        if (oa !== ob) return oa - ob;
        return (b.created_at ?? "").localeCompare(a.created_at ?? "");
      });
    }
    const liveIds = new Set(live.map((w) => w.machine_id));
    const ghosts: WorkerRow[] = remembered
      .filter((w) => !liveIds.has(w.machine_id))
      .map((w) => ({ ...w, status: "terminated" as const, raw_status: "terminated" }));
    const order: WorkerStatus[] = ["running", "initializing", "terminating", "unknown", "terminated"];
    const all = [...live, ...ghosts];
    return all.sort((a, b) => {
      const oa = order.indexOf(a.status);
      const ob = order.indexOf(b.status);
      if (oa !== ob) return oa - ob;
      return (b.created_at ?? "").localeCompare(a.created_at ?? "");
    });
  }, [live, remembered]);

  function clearHistory() {
    try {
      window.localStorage.removeItem(STORAGE_KEY(app.app_id));
    } catch {
      // ignore
    }
    setRemembered([]);
  }

  const liveCount = live === null ? "—" : live.length;
  const terminatedCount = rows.filter((r) => r.status === "terminated").length;

  return (
    <Card className="overflow-hidden">
      <div className="flex items-center justify-between gap-3 border-b border-border bg-muted/30 px-4 py-2 text-xs">
        <div className="flex items-center gap-3">
          <span className="text-muted-foreground">
            <span className="font-mono text-foreground">{liveCount}</span> live
          </span>
          <span className="text-muted-foreground">
            <span className="font-mono text-foreground">{terminatedCount}</span> remembered terminated
          </span>
          <span className="text-muted-foreground">max {app.autoscaler.max_containers}</span>
        </div>
        <div className="flex items-center gap-2">
          {terminatedCount > 0 && (
            <Button variant="ghost" size="xs" onClick={clearHistory}>
              Clear history
            </Button>
          )}
          <Button variant="outline" size="xs" onClick={fetchLive} disabled={loading}>
            {loading ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <RefreshCw className="h-3 w-3" />
            )}
            Refresh
          </Button>
        </div>
      </div>
      {err && (
        <div className="border-b border-destructive/30 bg-destructive/10 px-4 py-2 text-sm text-destructive">
          {err}
        </div>
      )}
      <CardContent className="px-0 py-0">
        <table className="w-full text-sm">
          <thead className="border-b border-border bg-muted/20 text-left text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="w-6 px-2 py-2"></th>
              <th className="px-4 py-2 font-medium">Worker ID</th>
              <th className="px-4 py-2 font-medium">Status</th>
              <th className="px-4 py-2 font-medium">Alive</th>
              <th className="px-4 py-2 font-medium">Region</th>
              <th className="px-4 py-2 font-medium">GPU</th>
              <th className="px-4 py-2 font-medium">vCPUs</th>
              <th className="px-4 py-2 font-medium">RAM</th>
              <th className="px-4 py-2 font-medium">Cost</th>
              <th className="px-4 py-2 font-medium">Created</th>
              <th className="px-4 py-2 font-medium text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((w) => (
              <WorkerRow key={w.machine_id} w={w} gw={aliveById[w.machine_id]} appId={app.app_id} onAction={fetchLive} onError={setErr} />
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={11} className="px-4 py-12 text-center text-sm text-muted-foreground">
                  {loading ? "Loading workers from RunPod…" : "No workers — fire a request to trigger the autoscaler."}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function WorkerRow({
  w, gw, appId, onAction, onError,
}: {
  w: WorkerRow;
  gw?: { alive: boolean; status: string };
  appId: string;
  onAction: () => void;
  onError: (msg: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [confirm, setConfirm] = useState(false);
  async function terminate() {
    setConfirm(false);
    setDeleting(true);
    try {
      const r = await fetch(
        `/api/proxy/apps/${encodeURIComponent(appId)}/workers/${encodeURIComponent(w.machine_id)}/terminate`,
        { method: "POST" },
      );
      if (!r.ok) {
        const b = await r.json().catch(() => ({}));
        throw new Error(b?.detail?.error ?? b?.error ?? r.statusText);
      }
      onAction();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleting(false);
    }
  }
  return (
    <>
      <tr className={cn("border-b border-border/60 last:border-b-0", w.status === "terminated" && "opacity-60")}>
        <td className="px-2 py-3 align-middle">
          <button
            onClick={() => setOpen((v) => !v)}
            className="flex items-center justify-center text-muted-foreground hover:text-foreground"
            aria-label={open ? "Hide logs" : "Show logs"}
          >
            {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
          </button>
        </td>
        <td className="px-4 py-3 font-mono text-xs">
          {w.pod_id ? (
            <a
              href={`https://console.runpod.io/pods?id=${encodeURIComponent(w.pod_id)}`}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 hover:text-foreground hover:underline"
              title="Open this pod's container in the RunPod console"
            >
              {w.machine_id}
              <ExternalLink className="h-3 w-3 opacity-60" />
            </a>
          ) : (
            w.machine_id
          )}
        </td>
        <td className="px-4 py-3">
          <span className={cn(
            "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs",
            STATUS_STYLES[w.status],
          )}>
            <span className="h-1.5 w-1.5 rounded-full bg-current" />
            {w.status}
          </span>
        </td>
        <td className="px-4 py-3">
          {w.status === "terminated" ? (
            <span className="text-xs text-muted-foreground">—</span>
          ) : gw?.alive ? (
            <span className={cn(
              "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs",
              gw.status === "ready" ? "bg-status-active/15 text-status-active" : "bg-status-idle/15 text-status-idle",
            )}>
              <span className="h-1.5 w-1.5 rounded-full bg-current" />
              {gw.status === "ready" ? "alive" : gw.status}
            </span>
          ) : (
            <span
              className="inline-flex items-center gap-1.5 rounded-full bg-status-down/15 px-2 py-0.5 text-xs text-status-down"
              title="RunPod shows this pod, but its worker hasn't registered / stopped heartbeating to the gateway"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-current" />
              not registered
            </span>
          )}
        </td>
        <td className="px-4 py-3">
          {w.region ? (
            <span className="inline-flex items-center gap-2">
              <span className="rounded bg-muted/60 px-1.5 py-0.5 font-mono text-[10px]">
                {w.region_code}
              </span>
              <span className="font-mono text-xs">{w.region}</span>
            </span>
          ) : (
            <span className="text-xs text-muted-foreground">—</span>
          )}
        </td>
        <td className="px-4 py-3 font-mono text-xs">{w.gpu}{w.gpu_count > 1 ? ` × ${w.gpu_count}` : ""}</td>
        <td className="px-4 py-3 font-mono text-xs">{w.vcpus || "—"}</td>
        <td className="px-4 py-3 font-mono text-xs">{w.ram_gb ? `${w.ram_gb} GB` : "—"}</td>
        <WorkerCostCell w={w} />
        <td className="px-4 py-3 text-xs text-muted-foreground">
          {w.created_at ? new Date(w.created_at).toLocaleString() : "—"}
        </td>
        <td className="px-4 py-3 text-right">
          {w.status !== "terminated" && (
            <Button
              variant="outline"
              size="xs"
              className="text-destructive hover:text-destructive"
              onClick={() => setConfirm(true)}
              disabled={deleting}
              title="Delete this container — the fleet re-provisions on the next request"
            >
              {deleting ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
              Delete
            </Button>
          )}
        </td>
      </tr>
      {open && (
        <tr className="border-b border-border/60 bg-muted/20">
          <td colSpan={11} className="px-4 py-3">
            <WorkerLogs machineId={w.machine_id} />
          </td>
        </tr>
      )}
      <Dialog open={confirm} onOpenChange={(o) => !deleting && setConfirm(o)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete container</DialogTitle>
            <DialogDescription>
              Delete container <code className="font-mono">{w.machine_id}</code>? The fleet drops
              to zero — the next request re-provisions it automatically. In-flight jobs on this
              container are interrupted.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setConfirm(false)} disabled={deleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={terminate} disabled={deleting}>
              {deleting && <Loader2 className="h-4 w-4 animate-spin" />}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

function WorkerCostCell({ w }: { w: WorkerRow }) {
  // Only running/initializing workers get a live ticker — RunPod doesn't
  // surface a reliable terminated_at through this proxy, so the moment a
  // worker leaves the live set we lose the ability to compute total cost
  // honestly. Show "—" instead of a misleading frozen value.
  const isLive = w.status === "running" || w.status === "initializing";
  const live = useLiveCost(isLive ? w.created_at : null, null, w.cost_per_hr);
  if (live == null) {
    return <td className="px-4 py-3 text-xs text-muted-foreground">—</td>;
  }
  return (
    <td className="px-4 py-3 font-mono text-xs tabular-nums">
      <span
        className={cn(
          "inline-flex items-center gap-1",
          isLive && "text-amber-600 dark:text-amber-400",
        )}
      >
        {isLive && <BurnFlame />}
        {formatCostUSD(live)}
      </span>
    </td>
  );
}

type LogSource = "gateway" | "container";

type GatewayEvent = { ts: number; level: "info" | "warning" | "error" | string; msg: string };

function WorkerLogs({ machineId }: { machineId: string }) {
  const [source, setSource] = useState<LogSource>("gateway");
  const [lines, setLines] = useState<string[]>([]);
  const [events, setEvents] = useState<GatewayEvent[]>([]);
  const [err, setErr] = useState<{ msg: string; hint?: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [autoTail, setAutoTail] = useState(true);

  const fetchLogs = useCallback(async () => {
    try {
      const url =
        source === "gateway"
          ? `/api/proxy/workers/${encodeURIComponent(machineId)}/events?tail=200`
          : `/api/proxy/workers/${encodeURIComponent(machineId)}/logs?tail=300`;
      const r = await fetch(url, { cache: "no-store" });
      const text = await r.text();
      if (!r.ok) {
        let msg = r.statusText;
        let hint: string | undefined;
        try {
          const body = JSON.parse(text) as { error?: string; hint?: string; detail?: string };
          msg = body.error ?? body.detail ?? msg;
          hint = body.hint;
        } catch {
          if (text) msg = text;
        }
        setErr({ msg, hint });
        return;
      }
      setErr(null);
      if (source === "gateway") {
        const body = JSON.parse(text) as { events?: GatewayEvent[] };
        setEvents(body.events ?? []);
      } else {
        const body = JSON.parse(text) as { lines?: string[] };
        setLines(body.lines ?? []);
      }
    } catch (e) {
      setErr({ msg: e instanceof Error ? e.message : String(e) });
    } finally {
      setLoading(false);
    }
  }, [machineId, source]);

  useEffect(() => {
    setLoading(true);
    setLines([]);
    setEvents([]);
    setErr(null);
  }, [source]);

  useEffect(() => {
    fetchLogs();
    if (!autoTail) return;
    const id = window.setInterval(fetchLogs, 2500);
    return () => window.clearInterval(id);
  }, [fetchLogs, autoTail]);

  const empty = source === "gateway" ? events.length === 0 : lines.length === 0;

  // Auto-tail: keep the scroll pinned to the bottom on updates so the view
  // feels live. If the user has scrolled up to read older content (more than
  // 32px above the bottom), leave them alone — yanking them away mid-read
  // is the wrong behavior.
  const scrollRef = useRef<HTMLDivElement | HTMLPreElement | null>(null);
  const wasAtBottomRef = useRef(true);
  // Capture "was the user at bottom?" BEFORE the DOM updates, then scroll
  // after — prevents the layout flash you'd get with a plain useEffect.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (wasAtBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
    wasAtBottomRef.current = distance < 32;
  }, [lines, events, source]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <div className="flex items-center gap-2">
          <div className="inline-flex rounded-md border border-border p-0.5">
            <button
              type="button"
              onClick={() => setSource("gateway")}
              className={cn(
                "rounded px-2 py-0.5 text-[10px] uppercase tracking-wide",
                source === "gateway" ? "bg-primary/15 text-primary" : "text-muted-foreground hover:text-foreground",
              )}
            >
              gateway events
            </button>
            <button
              type="button"
              onClick={() => setSource("container")}
              className={cn(
                "rounded px-2 py-0.5 text-[10px] uppercase tracking-wide",
                source === "container" ? "bg-primary/15 text-primary" : "text-muted-foreground hover:text-foreground",
              )}
            >
              container logs
            </button>
          </div>
          <span className="font-mono">machine = {machineId}</span>
          {loading && <Loader2 className="h-3 w-3 animate-spin" />}
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 text-[10px]">
            <input
              type="checkbox"
              checked={autoTail}
              onChange={(e) => setAutoTail(e.target.checked)}
              className="h-3 w-3"
            />
            tail (poll every 2.5s)
          </label>
          <Button variant="outline" size="xs" onClick={fetchLogs}>
            <RefreshCw className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {err ? (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          <div className="font-medium">{err.msg}</div>
          {err.hint && <div className="mt-1 opacity-80">{err.hint}</div>}
        </div>
      ) : empty ? (
        <div className="rounded-md border border-dashed border-border bg-background/40 px-3 py-4 text-center text-xs text-muted-foreground">
          {loading
            ? "loading…"
            : source === "gateway"
              ? "no gateway events for this worker yet"
              : "no container logs yet — vLLM may still be booting, or this worker pre-dates the log shipper"}
        </div>
      ) : source === "gateway" ? (
        <div
          ref={(el) => { scrollRef.current = el; }}
          className="terminal-block max-h-72 overflow-auto rounded-md border border-border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-200 scrollbar-thin"
        >
          {events.map((e, i) => (
            <EventRow key={i} event={e} />
          ))}
        </div>
      ) : (
        <pre
          ref={(el) => { scrollRef.current = el; }}
          className="terminal-block max-h-72 overflow-auto rounded-md border border-border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-200 scrollbar-thin"
        >
          {lines.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </pre>
      )}

      <p className="text-[10px] leading-relaxed text-muted-foreground">
        {source === "gateway" ? (
          <>
            Source: per-worker timeline pushed by the gateway as it provisions,
            registers, drains and terminates. Capped at 200 events, kept for 1h
            after the worker is gone.
          </>
        ) : (
          <>
            Source: vLLM <code className="font-mono">stdout/stderr</code> shipped from the
            worker container by the bundled log-shipper. RunPod&apos;s public API has no logs
            endpoint, so this is the only way to see container output.
          </>
        )}
      </p>
    </div>
  );
}

const LEVEL_STYLES: Record<string, string> = {
  info:    "text-muted-foreground",
  warning: "text-amber-600 dark:text-amber-400",
  error:   "text-red-600 dark:text-red-400",
};
const LEVEL_BADGES: Record<string, string> = {
  info:    "bg-muted text-muted-foreground",
  warning: "bg-amber-500/15 text-amber-600 dark:text-amber-400",
  error:   "bg-red-500/15 text-red-600 dark:text-red-400",
};

// ---- Multi-model VM fleet view -------------------------------------------

type FleetModel = {
  model: string;
  state: string;        // queued | launching | awake | asleep | dead | …
  queue_ahead?: number; // when state === "queued": models loading ahead of it
  inflight?: number;
  gpus?: number[];
  tp?: number;
  last_used_ts?: number | null;
  reason?: string | null;  // human cause when state === "dead"
  port?: number;
};
type FleetStatus = { workers: number; models: FleetModel[]; paused?: boolean };

const FLEET_STATE_STYLES: Record<string, string> = {
  awake:     "bg-status-active/15 text-status-active",
  launching: "bg-status-idle/15 text-status-idle",
  queued:    "bg-muted text-muted-foreground",
  asleep:    "bg-muted text-muted-foreground",
  dead:      "bg-status-down/15 text-status-down",
};

/** Label for a fleet member's state badge — queued models show their place in line. */
function fleetStateLabel(m: FleetModel): string {
  if (m.state === "queued") {
    const n = m.queue_ahead ?? 0;
    return n > 0 ? `queued (${n} ahead)` : "queued";
  }
  return m.state;
}

function MultiModelFleet({ app }: { app: AppRecord }) {
  const [status, setStatus] = useState<FleetStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Open log panels live in the URL as a comma-joined list (?log=modelA,modelB)
  // so the set is shareable/deep-linkable and survives the 10s status poll.
  // Multiple can be open at once; each toggles independently.
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const openLogs = useMemo(
    () => new Set((searchParams.get("log") ?? "").split(",").filter(Boolean)),
    [searchParams],
  );
  const toggleLog = useCallback(
    (model: string) => {
      const next = new Set((searchParams.get("log") ?? "").split(",").filter(Boolean));
      if (next.has(model)) next.delete(model);
      else next.add(model);
      const params = new URLSearchParams(searchParams.toString());
      if (next.size) params.set("log", Array.from(next).join(","));
      else params.delete("log");
      const qs = params.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [router, pathname, searchParams],
  );

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    try {
      const r = await fetch(`/api/proxy/apps/${encodeURIComponent(app.app_id)}/status`, { cache: "no-store" });
      if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
      setStatus((await r.json()) as FleetStatus);
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [app.app_id]);

  useEffect(() => {
    // First poll on mount (matches the RunPod workers tab's pattern in this file).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchStatus();
    const id = window.setInterval(fetchStatus, POLL_MS);
    return () => window.clearInterval(id);
  }, [fetchStatus]);

  // Fall back to the configured members (from the app record) until the worker
  // reports live state, so the table isn't empty while the fleet boots.
  const configured: FleetModel[] = (app.models ?? []).map((m) => ({
    model: m.model, state: "—", tp: m.tp,
  }));
  // Dedupe by model name: with >1 live worker the status endpoint reports the
  // same member once per worker (replicas) — collapse so each model is one row
  // (and React keys stay unique). The gateway dedupes too; this is defensive.
  const seenModels = new Set<string>();
  const rows = (status?.models?.length ? status.models : configured).filter(
    (m) => (seenModels.has(m.model) ? false : (seenModels.add(m.model), true)),
  );
  const workerUp = (status?.workers ?? 0) > 0;
  const anyAwake = rows.some((m) => m.state === "awake");

  const [sleepingAll, setSleepingAll] = useState(false);
  async function sleepAll() {
    setSleepingAll(true);
    try {
      const r = await fetch(`/api/proxy/apps/${encodeURIComponent(app.app_id)}/model-action`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "sleep_all" }),
      });
      if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
      window.setTimeout(fetchStatus, 2000);
    } catch {
      // best-effort; the table will reflect reality on the next poll
    } finally {
      setSleepingAll(false);
    }
  }

  // Kill: tear the worker down AND pause the autoscaler so it stays down (frees
  // the GPUs). Resume with Restart all. Restart: drain + reprovision (cold
  // restart), which also clears the paused flag.
  const [killing, setKilling] = useState(false);
  const [restarting, setRestarting] = useState(false);
  async function workerAction(path: string, busy: (b: boolean) => void) {
    busy(true);
    setErr(null);
    try {
      const r = await fetch(`/api/proxy/apps/${encodeURIComponent(app.app_id)}/${path}`, { method: "POST" });
      if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
      window.setTimeout(fetchStatus, 1500);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      busy(false);
    }
  }
  const killAll = () => workerAction("workers/kill", setKilling);
  const restartAll = () => workerAction("restart", setRestarting);
  const paused = status?.paused ?? false;
  const busy = sleepingAll || killing || restarting;

  return (
    <Card className="overflow-hidden">
      <div className="flex items-center justify-between gap-3 border-b border-border bg-muted/30 px-4 py-2 text-xs">
        <div className="flex items-center gap-3">
          <span className="text-muted-foreground">
            VM worker:{" "}
            <span
              className={cn(
                "font-medium",
                workerUp ? "text-status-active" : paused ? "text-status-down" : "text-muted-foreground",
              )}
            >
              {workerUp ? "up" : paused ? "killed (paused)" : status === null ? "…" : "down"}
            </span>
          </span>
          <span className="text-muted-foreground">
            <span className="font-mono text-foreground">{rows.length}</span> models
          </span>
          <span className="text-muted-foreground">sleep level {app.sleep_level ?? 1}</span>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant={openLogs.has("__worker__") ? "secondary" : "outline"}
            size="xs" onClick={() => toggleLog("__worker__")}
            title="Worker-agent scheduler log (wave-loading, sleep/wake, commands, dead reasons)"
          >
            <FileText className="h-3 w-3" />
            Worker log
          </Button>
          <Button
            variant="outline" size="xs" onClick={sleepAll}
            disabled={busy || !anyAwake}
            title="Sleep all awake models (free their VRAM; worker stays up)"
          >
            {sleepingAll ? <Loader2 className="h-3 w-3 animate-spin" /> : <Moon className="h-3 w-3" />}
            Sleep all
          </Button>
          <Button
            variant="outline" size="xs" onClick={restartAll} disabled={busy}
            title="Drain + reprovision the worker (cold restart). Also resumes a killed fleet."
          >
            {restarting ? <Loader2 className="h-3 w-3 animate-spin" /> : <RotateCw className="h-3 w-3" />}
            Restart all
          </Button>
          <Button
            variant="outline" size="xs" onClick={killAll}
            disabled={busy || !workerUp}
            title="Terminate the worker and keep it down (frees the GPUs). Resume with Restart all."
            className="text-destructive hover:text-destructive"
          >
            {killing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Ban className="h-3 w-3" />}
            Kill all workers
          </Button>
          <Button variant="outline" size="xs" onClick={fetchStatus} disabled={loading}>
            {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
            Refresh
          </Button>
        </div>
      </div>
      {err && (
        <div className="border-b border-destructive/30 bg-destructive/10 px-4 py-2 text-sm text-destructive">{err}</div>
      )}
      <CardContent className="px-0 py-0">
        {openLogs.has("__worker__") && (
          <div className="border-b border-border bg-muted/10 px-4 py-3">
            <ModelLogs
              appId={app.app_id}
              model="__worker__"
              label="worker-agent (scheduler)"
              onClose={() => toggleLog("__worker__")}
            />
          </div>
        )}
        <table className="w-full text-sm">
          <thead className="border-b border-border bg-muted/20 text-left text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="px-4 py-2 font-medium">Model</th>
              <th className="px-4 py-2 font-medium">State</th>
              <th className="px-4 py-2 font-medium">GPUs</th>
              <th className="px-4 py-2 font-medium">TP</th>
              <th className="px-4 py-2 font-medium">In-flight</th>
              <th className="px-4 py-2 font-medium text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <FleetModelRow
                key={m.model}
                appId={app.app_id}
                m={m}
                isOpen={openLogs.has(m.model)}
                onToggle={() => toggleLog(m.model)}
                onRefresh={fetchStatus}
              />
            ))}
            {rows.length === 0 && (
              <tr><td colSpan={6} className="px-4 py-12 text-center text-sm text-muted-foreground">
                {loading ? "Loading fleet…" : "No models reported yet."}
              </td></tr>
            )}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function FleetModelRow({
  appId, m, isOpen, onToggle, onRefresh,
}: {
  appId: string; m: FleetModel; isOpen: boolean; onToggle: () => void; onRefresh: () => void;
}) {
  const dead = m.state === "dead";
  const awake = m.state === "awake";
  const [busy, setBusy] = useState<"kill" | "restart" | "sleep" | null>(null);
  // Inline feedback shown in the row instead of a toast.
  const [note, setNote] = useState<{ tone: "ok" | "err"; text: string } | null>(null);

  async function doAction(action: "kill" | "restart" | "sleep") {
    setBusy(action);
    setNote(null);
    try {
      const r = await fetch(`/api/proxy/apps/${encodeURIComponent(appId)}/model-action`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: m.model, action }),
      });
      const body = await r.json().catch(() => ({} as Record<string, unknown>));
      if (!r.ok) {
        const d = (body as { detail?: unknown; error?: string }).detail;
        const msg = typeof d === "string" ? d : (d as { error?: string })?.error ?? (body as { error?: string }).error ?? r.statusText;
        throw new Error(msg);
      }
      setNote({ tone: "ok", text: `${action} queued — applies on next heartbeat` });
      window.setTimeout(() => setNote(null), 4000);
      // Give the worker a heartbeat (≤5s) to pick the command up, then refresh.
      window.setTimeout(onRefresh, 2000);
    } catch (e) {
      setNote({ tone: "err", text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <tr className={cn("border-b border-border/60 last:border-b-0", dead && "bg-status-down/[0.05]")}>
        <td className="px-4 py-3 align-top">
          <div className="font-mono text-xs">{m.model}</div>
          {m.port != null && (
            <div className="mt-0.5 font-mono text-[11px] text-muted-foreground">
              localhost:{m.port}
            </div>
          )}
          {m.reason && (
            // Surface *why* a model isn't serving (e.g. "GPU is not enough")
            // right under its name — the single most useful thing when dead.
            <div className="mt-1 flex items-start gap-1 text-[11px] leading-snug text-status-down">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
              <span className="break-words">{m.reason}</span>
            </div>
          )}
        </td>
        <td className="px-4 py-3 align-top">
          <span className={cn(
            "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs",
            FLEET_STATE_STYLES[m.state] ?? "bg-muted text-muted-foreground",
          )}>
            <span className="h-1.5 w-1.5 rounded-full bg-current" />
            {fleetStateLabel(m)}
          </span>
        </td>
        <td className="px-4 py-3 align-top font-mono text-xs">{m.gpus?.length ? m.gpus.join(",") : "—"}</td>
        <td className="px-4 py-3 align-top font-mono text-xs">{m.tp ?? "—"}</td>
        <td className="px-4 py-3 align-top font-mono text-xs">{m.inflight ?? 0}</td>
        <td className="px-4 py-3 align-top">
          <div className="flex items-center justify-end gap-0.5">
            <Button
              variant="ghost" size="icon-xs" onClick={() => doAction("sleep")}
              disabled={busy !== null || !awake}
              title="Sleep — drain + put this model to sleep, freeing its GPUs"
              aria-label="Sleep model"
            >
              {busy === "sleep" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Moon className="h-3.5 w-3.5" />}
            </Button>
            <Button
              variant="ghost" size="icon-xs" onClick={() => doAction("restart")}
              disabled={busy !== null}
              title="Restart — kill and relaunch this model's vLLM"
              aria-label="Restart model"
            >
              {busy === "restart" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCw className="h-3.5 w-3.5" />}
            </Button>
            <Button
              variant="ghost" size="icon-xs" onClick={() => doAction("kill")}
              disabled={busy !== null || dead}
              className="text-destructive hover:text-destructive"
              title="Kill — stop this model and free its GPUs (its tp workers too)"
              aria-label="Kill model"
            >
              {busy === "kill" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Ban className="h-3.5 w-3.5" />}
            </Button>
            <Button
              variant="ghost" size="icon-xs"
              onClick={onToggle} aria-expanded={isOpen}
              title={isOpen ? "Hide logs" : "Show logs"}
              aria-label={isOpen ? "Hide logs" : "Show logs"}
              className={cn(isOpen && "text-primary")}
            >
              {isOpen ? <ChevronDown className="h-3.5 w-3.5" /> : <FileText className="h-3.5 w-3.5" />}
            </Button>
          </div>
          {note && (
            <div className={cn(
              "mt-1 break-words text-right text-[10px] leading-snug",
              note.tone === "err" ? "text-status-down" : "text-muted-foreground",
            )}>
              {note.text}
            </div>
          )}
        </td>
      </tr>
      {isOpen && (
        <tr className="border-b border-border/60 bg-muted/20">
          <td colSpan={6} className="px-4 py-3">
            <ModelLogs appId={appId} model={m.model} onClose={onToggle} />
          </td>
        </tr>
      )}
    </>
  );
}

function ModelLogs({ appId, model, onClose, label }: { appId: string; model: string; onClose: () => void; label?: string }) {
  const [lines, setLines] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [autoTail, setAutoTail] = useState(true);

  const fetchLogs = useCallback(async () => {
    try {
      const url = `/api/proxy/apps/${encodeURIComponent(appId)}/models/logs?model=${encodeURIComponent(model)}&tail=400`;
      const r = await fetch(url, { cache: "no-store" });
      const text = await r.text();
      if (!r.ok) {
        let msg = r.statusText;
        try {
          const b = JSON.parse(text) as { error?: string; detail?: string };
          msg = b.error ?? b.detail ?? msg;
        } catch {
          if (text) msg = text;
        }
        setErr(msg);
        return;
      }
      setErr(null);
      const body = JSON.parse(text) as { lines?: string[] };
      setLines(body.lines ?? []);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [appId, model]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchLogs();
    if (!autoTail) return;
    const id = window.setInterval(fetchLogs, 2500);
    return () => window.clearInterval(id);
  }, [fetchLogs, autoTail]);

  // Auto-tail: keep pinned to the bottom unless the user scrolled up to read.
  const scrollRef = useRef<HTMLPreElement | null>(null);
  const wasAtBottomRef = useRef(true);
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (wasAtBottomRef.current) el.scrollTop = el.scrollHeight;
    wasAtBottomRef.current = distance < 32;
  }, [lines]);

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate font-mono">{label ?? `model = ${model}`}</span>
          {loading && <Loader2 className="h-3 w-3 shrink-0 animate-spin" />}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <label className="flex items-center gap-1 text-[10px]">
            <input
              type="checkbox"
              checked={autoTail}
              onChange={(e) => setAutoTail(e.target.checked)}
              className="h-3 w-3"
            />
            tail (poll every 2.5s)
          </label>
          <Button variant="outline" size="xs" onClick={fetchLogs}>
            <RefreshCw className="h-3 w-3" />
          </Button>
          <Button variant="outline" size="xs" onClick={onClose} aria-label="Close logs" className="text-foreground">
            <X className="h-3 w-3" />
            Close
          </Button>
        </div>
      </div>

      {err ? (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {err}
        </div>
      ) : lines.length === 0 ? (
        <div className="rounded-md border border-dashed border-border bg-background/40 px-3 py-4 text-center text-xs text-muted-foreground">
          {loading ? "loading…" : "no logs yet — vLLM may still be booting, or this model hasn't launched"}
        </div>
      ) : (
        <pre
          ref={(el) => { scrollRef.current = el; }}
          className="terminal-block max-h-80 w-full overflow-y-auto whitespace-pre-wrap break-words rounded-md border border-border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-200 scrollbar-thin"
        >
          {lines.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </pre>
      )}

      <p className="text-[10px] leading-relaxed text-muted-foreground">
        {model === "__worker__" ? (
          <>
            Source: the worker-agent&apos;s own <code className="font-mono">stdout</code> on the VM —
            wave-loading, sleep/wake, operator commands, dead-model reasons — shipped by the log-shipper.
          </>
        ) : (
          <>
            Source: this model&apos;s vLLM <code className="font-mono">stdout/stderr</code> on the VM, shipped
            per-model by the worker-agent log-shipper.
          </>
        )}{" "}
        Capped at {WORKER_LOGS_CAP_HINT} lines, kept for 1h.
      </p>
    </div>
  );
}

const WORKER_LOGS_CAP_HINT = "5000";

function EventRow({ event }: { event: GatewayEvent }) {
  const date = new Date(event.ts * 1000);
  const time = date.toLocaleTimeString(undefined, { hour12: false });
  const level = (event.level ?? "info").toLowerCase();
  const badgeCls = LEVEL_BADGES[level] ?? LEVEL_BADGES.info;
  const textCls = LEVEL_STYLES[level] ?? LEVEL_STYLES.info;
  return (
    <div className="flex items-start gap-3 border-b border-border/30 px-1.5 py-1.5 last:border-b-0">
      <span className="font-mono text-[10px] text-muted-foreground tabular-nums">{time}</span>
      <span
        className={cn(
          "rounded px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wide shrink-0",
          badgeCls,
        )}
      >
        {level}
      </span>
      <span className={cn("flex-1 break-words", textCls)}>{event.msg}</span>
    </div>
  );
}
