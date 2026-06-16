"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  AlertTriangle,
  Boxes,
  Database,
  Loader2,
  Network,
  RefreshCw,
  RotateCw,
  Server,
  Trash2,
  X,
} from "lucide-react";
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
import type { AppRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

const POLL_MS = 10_000;

// ---- shared shapes (mirror the Workers tab) ------------------------------

type FleetModel = {
  model: string;
  state: string; // queued | launching | awake | asleep | dead | …
  queue_ahead?: number;
  inflight?: number;
  gpus?: number[];
  tp?: number;
  reason?: string | null;
  port?: number;
};
type FleetStatus = { workers: number; models: FleetModel[]; paused?: boolean };
type WorkerLite = { machine_id: string; alive: boolean; status: string };

// health → dot colour. green = up, amber = transitional, grey = idle, red = bad.
type Health = "up" | "warn" | "idle" | "down";

const DOT: Record<Health, string> = {
  up: "bg-status-active",
  warn: "bg-status-idle",
  idle: "bg-muted-foreground/50",
  down: "bg-status-down",
};
const RING: Record<Health, string> = {
  up: "border-status-active/40",
  warn: "border-status-idle/40",
  idle: "border-border",
  down: "border-status-down/50",
};

function modelHealth(state: string): Health {
  switch (state) {
    case "awake":
      return "up";
    case "launching":
    case "queued":
    case "waking":
      return "warn";
    case "dead":
      return "down";
    case "asleep":
      return "idle";
    default:
      return "idle";
  }
}
function workerHealth(w: WorkerLite): Health {
  if (!w.alive) return "down";
  return w.status === "ready" ? "up" : "warn";
}

// ---- node ids for the connector graph ------------------------------------

const AIES = "__aies__";
const REDIS = "__redis__";
const wNode = (mid: string) => `w:${mid}`;
const mNode = (model: string) => `m:${model}`;

export function VisualTab({ app }: { app: AppRecord }) {
  const [status, setStatus] = useState<FleetStatus | null>(null);
  const [workers, setWorkers] = useState<WorkerLite[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [gatewayOk, setGatewayOk] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    try {
      const [sr, wr] = await Promise.all([
        fetch(`/api/proxy/apps/${encodeURIComponent(app.app_id)}/status`, { cache: "no-store" }),
        fetch(`/api/proxy/apps/${encodeURIComponent(app.app_id)}/workers`, { cache: "no-store" }),
      ]);
      if (!sr.ok) throw new Error(await sr.text().catch(() => sr.statusText));
      setStatus((await sr.json()) as FleetStatus);
      setGatewayOk(true);
      if (wr.ok) {
        setWorkers((await wr.json()) as WorkerLite[]);
      } else {
        setWorkers([]);
      }
      setErr(null);
    } catch (e) {
      setGatewayOk(false);
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [app.app_id]);

  useEffect(() => {
    fetchAll();
    const id = window.setInterval(fetchAll, POLL_MS);
    return () => window.clearInterval(id);
  }, [fetchAll]);

  // Fall back to configured members until the worker reports live state, and
  // dedupe by model name (replicas report the same member once per worker).
  const configured: FleetModel[] = (app.models ?? []).map((m) => ({
    model: m.model,
    state: "—",
    tp: m.tp,
  }));
  const seen = new Set<string>();
  const models = (status?.models?.length ? status.models : configured).filter((m) =>
    seen.has(m.model) ? false : (seen.add(m.model), true),
  );
  const liveWorkers = (workers ?? []).filter((w) => w.status !== "terminated");
  const paused = status?.paused ?? false;

  // Edges for the connector overlay: AIES→worker (dashed), worker→model (solid),
  // AIES⋯Redis (dotted dependency). With no live worker, hang models off AIES so
  // the graph still reads.
  const edges: { from: string; to: string; kind: "dash" | "solid" | "dep" }[] = [];
  edges.push({ from: AIES, to: REDIS, kind: "dep" });
  if (liveWorkers.length === 0) {
    for (const m of models) edges.push({ from: AIES, to: mNode(m.model), kind: "dash" });
  } else {
    for (const w of liveWorkers) {
      edges.push({ from: AIES, to: wNode(w.machine_id), kind: "dash" });
      for (const m of models) edges.push({ from: wNode(w.machine_id), to: mNode(m.model), kind: "solid" });
    }
  }

  const selectedWorker = liveWorkers.find((w) => wNode(w.machine_id) === selected) ?? null;

  return (
    <div className="space-y-4">
      <Card className="overflow-hidden">
        <div className="flex items-center justify-between gap-3 border-b border-border bg-muted/30 px-4 py-2 text-xs">
          <div className="flex items-center gap-3">
            <span className="inline-flex items-center gap-1.5 text-muted-foreground">
              <Network className="h-3.5 w-3.5" /> Topology
            </span>
            <span className="text-muted-foreground">
              <span className="font-mono text-foreground">{liveWorkers.length}</span> worker
              {liveWorkers.length === 1 ? "" : "s"}
            </span>
            <span className="text-muted-foreground">
              <span className="font-mono text-foreground">{models.length}</span> models
            </span>
            {paused && <span className="text-status-down">killed (paused)</span>}
            {liveWorkers.length > app.autoscaler.max_containers && (
              <span className="inline-flex items-center gap-1 text-status-down">
                <AlertTriangle className="h-3 w-3" />
                {liveWorkers.length} workers &gt; max {app.autoscaler.max_containers} (duplicate)
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <Legend />
            <Button variant="outline" size="xs" onClick={fetchAll} disabled={loading}>
              {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
              Refresh
            </Button>
          </div>
        </div>
        {err && (
          <div className="border-b border-destructive/30 bg-destructive/10 px-4 py-2 text-sm text-destructive">
            {err}
          </div>
        )}
        <CardContent className="p-0">
          <Graph
            app={app}
            gatewayOk={gatewayOk}
            liveWorkers={liveWorkers}
            models={models}
            edges={edges}
            selected={selected}
            onSelect={(id) => setSelected((cur) => (cur === id ? null : id))}
          />
        </CardContent>
      </Card>

      {selectedWorker ? (
        <WorkerDetail
          app={app}
          worker={selectedWorker}
          onClose={() => setSelected(null)}
          onChanged={fetchAll}
        />
      ) : (
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            Click a <span className="font-medium text-foreground">worker</span> node to see its
            config, logs, and restart / delete actions.
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function Legend() {
  const items: [Health, string][] = [
    ["up", "up"],
    ["warn", "loading"],
    ["idle", "asleep"],
    ["down", "down / dead"],
  ];
  return (
    <div className="hidden items-center gap-2 md:flex">
      {items.map(([h, label]) => (
        <span key={h} className="inline-flex items-center gap-1 text-[10px] text-muted-foreground">
          <span className={cn("h-2 w-2 rounded-full", DOT[h])} />
          {label}
        </span>
      ))}
    </div>
  );
}

// ---- the graph (3 columns + measured SVG connectors) ---------------------

function Graph({
  app,
  gatewayOk,
  liveWorkers,
  models,
  edges,
  selected,
  onSelect,
}: {
  app: AppRecord;
  gatewayOk: boolean;
  liveWorkers: WorkerLite[];
  models: FleetModel[];
  edges: { from: string; to: string; kind: "dash" | "solid" | "dep" }[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const nodeRefs = useRef<Map<string, HTMLElement>>(new Map());
  const setNodeRef = useCallback((id: string) => (el: HTMLElement | null) => {
    if (el) nodeRefs.current.set(id, el);
    else nodeRefs.current.delete(id);
  }, []);

  const [paths, setPaths] = useState<{ d: string; kind: string }[]>([]);
  const [size, setSize] = useState({ w: 0, h: 0 });

  const recompute = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const cRect = container.getBoundingClientRect();
    setSize({ w: cRect.width, h: cRect.height });
    const next: { d: string; kind: string }[] = [];
    for (const e of edges) {
      const a = nodeRefs.current.get(e.from);
      const b = nodeRefs.current.get(e.to);
      if (!a || !b) continue;
      const ar = a.getBoundingClientRect();
      const br = b.getBoundingClientRect();
      const x1 = ar.right - cRect.left;
      const y1 = ar.top + ar.height / 2 - cRect.top;
      const x2 = br.left - cRect.left;
      const y2 = br.top + br.height / 2 - cRect.top;
      const dx = Math.max(30, (x2 - x1) / 2);
      next.push({ d: `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`, kind: e.kind });
    }
    setPaths(next);
  }, [edges]);

  // Recompute after layout + on resize. The dependency on edges/models/workers
  // re-runs it whenever the graph's shape changes.
  useLayoutEffect(() => {
    recompute();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recompute, liveWorkers.length, models.length]);

  useEffect(() => {
    const ro = new ResizeObserver(() => recompute());
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [recompute]);

  return (
    <div ref={containerRef} className="relative min-h-[360px] w-full overflow-x-auto px-6 py-8">
      <svg
        className="pointer-events-none absolute inset-0"
        width={size.w}
        height={size.h}
        style={{ zIndex: 0 }}
      >
        {paths.map((p, i) => (
          <path
            key={i}
            d={p.d}
            fill="none"
            className={
              p.kind === "dep"
                ? "stroke-muted-foreground/30"
                : p.kind === "dash"
                  ? "stroke-muted-foreground/50"
                  : "stroke-status-active/40"
            }
            strokeWidth={1.5}
            strokeDasharray={p.kind === "solid" ? undefined : p.kind === "dep" ? "2 4" : "5 4"}
          />
        ))}
      </svg>

      <div className="relative grid grid-cols-[max-content_1fr_1fr] items-center gap-x-16 gap-y-6" style={{ zIndex: 1 }}>
        {/* Column 1: AIES gateway + Redis */}
        <div className="flex flex-col gap-6">
          <NodeBox
            ref={setNodeRef(AIES)}
            icon={<Server className="h-4 w-4" />}
            title="AIES"
            subtitle="gateway"
            health={gatewayOk ? "up" : "down"}
          />
          <NodeBox
            ref={setNodeRef(REDIS)}
            icon={<Database className="h-4 w-4" />}
            title="Redis"
            subtitle="state / registry"
            health={gatewayOk ? "up" : "warn"}
          />
        </div>

        {/* Column 2: workers */}
        <div className="flex flex-col gap-5">
          {liveWorkers.length === 0 && (
            <div className="text-xs text-muted-foreground">
              No live workers — fire a request to trigger the autoscaler.
            </div>
          )}
          {liveWorkers.map((w) => (
            <NodeBox
              key={w.machine_id}
              ref={setNodeRef(wNode(w.machine_id))}
              icon={<Boxes className="h-4 w-4" />}
              title={w.machine_id}
              subtitle={w.alive ? (w.status === "ready" ? "worker · ready" : `worker · ${w.status}`) : "worker · no heartbeat"}
              health={workerHealth(w)}
              onClick={() => onSelect(wNode(w.machine_id))}
              active={selected === wNode(w.machine_id)}
              mono
            />
          ))}
        </div>

        {/* Column 3: models */}
        <div className="flex flex-col gap-4">
          {models.map((m) => (
            <NodeBox
              key={m.model}
              ref={setNodeRef(mNode(m.model))}
              title={m.model}
              subtitle={
                m.state === "—"
                  ? `tp ${m.tp ?? "?"}`
                  : `${m.state}${m.gpus?.length ? ` · gpu ${m.gpus.join(",")}` : ""}`
              }
              health={modelHealth(m.state)}
              danger={m.reason ?? null}
              mono
            />
          ))}
          {models.length === 0 && (
            <div className="text-xs text-muted-foreground">No models reported.</div>
          )}
        </div>
      </div>
    </div>
  );
}

const NodeBox = (() => {
  function Inner(
    {
      icon,
      title,
      subtitle,
      health,
      onClick,
      active,
      mono,
      danger,
    }: {
      icon?: React.ReactNode;
      title: string;
      subtitle?: string;
      health: Health;
      onClick?: () => void;
      active?: boolean;
      mono?: boolean;
      danger?: string | null;
    },
    ref: React.Ref<HTMLDivElement>,
  ) {
    const clickable = !!onClick;
    return (
      <div
        ref={ref}
        onClick={onClick}
        role={clickable ? "button" : undefined}
        tabIndex={clickable ? 0 : undefined}
        onKeyDown={clickable ? (e) => (e.key === "Enter" || e.key === " ") && onClick!() : undefined}
        className={cn(
          "w-56 max-w-full rounded-lg border bg-background px-3 py-2.5 shadow-sm transition",
          RING[health],
          clickable && "cursor-pointer hover:border-primary/60 hover:shadow",
          active && "border-primary ring-2 ring-primary/30",
        )}
      >
        <div className="flex items-center gap-2">
          <span className={cn("h-2.5 w-2.5 shrink-0 rounded-full", DOT[health])} />
          {icon && <span className="shrink-0 text-muted-foreground">{icon}</span>}
          <span className={cn("min-w-0 flex-1 truncate text-sm font-medium", mono && "font-mono text-xs")} title={title}>
            {title}
          </span>
        </div>
        {subtitle && (
          <div className="mt-1 pl-[18px] text-[11px] text-muted-foreground">{subtitle}</div>
        )}
        {danger && (
          <div className="mt-1 flex items-start gap-1 pl-[18px] text-[11px] leading-snug text-status-down">
            <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
            <span className="break-words">{danger}</span>
          </div>
        )}
      </div>
    );
  }
  Inner.displayName = "NodeBox";
  return Inner;
})() as unknown as React.ForwardRefExoticComponent<
  {
    icon?: React.ReactNode;
    title: string;
    subtitle?: string;
    health: Health;
    onClick?: () => void;
    active?: boolean;
    mono?: boolean;
    danger?: string | null;
  } & React.RefAttributes<HTMLDivElement>
>;

// ---- worker detail panel (conf + logs + actions) -------------------------

function WorkerDetail({
  app,
  worker,
  onClose,
  onChanged,
}: {
  app: AppRecord;
  worker: WorkerLite;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [busy, setBusy] = useState<"restart" | "delete" | null>(null);
  const [note, setNote] = useState<{ tone: "ok" | "err"; text: string } | null>(null);

  async function restart() {
    setBusy("restart");
    setNote(null);
    try {
      const r = await fetch(`/api/proxy/apps/${encodeURIComponent(app.app_id)}/restart`, { method: "POST" });
      if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
      setNote({ tone: "ok", text: "Restart triggered — the worker drains and reprovisions (cold start)." });
      window.setTimeout(onChanged, 1500);
    } catch (e) {
      setNote({ tone: "err", text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(null);
    }
  }

  async function del() {
    setConfirmDelete(false);
    setBusy("delete");
    setNote(null);
    try {
      const r = await fetch(
        `/api/proxy/apps/${encodeURIComponent(app.app_id)}/workers/${encodeURIComponent(worker.machine_id)}/terminate`,
        { method: "POST" },
      );
      if (!r.ok) {
        const b = await r.json().catch(() => ({}));
        throw new Error(b?.detail?.error ?? b?.error ?? r.statusText);
      }
      setNote({ tone: "ok", text: "Worker terminated. The autoscaler re-provisions unless the fleet is paused." });
      window.setTimeout(() => {
        onChanged();
        onClose();
      }, 1200);
    } catch (e) {
      setNote({ tone: "err", text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(null);
    }
  }

  const health = workerHealth(worker);

  return (
    <Card className="overflow-hidden">
      <div className="flex items-center justify-between gap-3 border-b border-border bg-muted/30 px-4 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className={cn("h-2.5 w-2.5 shrink-0 rounded-full", DOT[health])} />
          <span className="truncate font-mono text-sm">{worker.machine_id}</span>
          <span className="shrink-0 text-xs text-muted-foreground">
            {worker.alive ? worker.status : "no heartbeat"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="xs" onClick={restart} disabled={busy !== null}>
            {busy === "restart" ? <Loader2 className="h-3 w-3 animate-spin" /> : <RotateCw className="h-3 w-3" />}
            Restart
          </Button>
          <Button
            variant="outline"
            size="xs"
            className="text-destructive hover:text-destructive"
            onClick={() => setConfirmDelete(true)}
            disabled={busy !== null}
          >
            {busy === "delete" ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
            Delete worker
          </Button>
          <Button variant="ghost" size="xs" onClick={onClose} aria-label="Close">
            <X className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {note && (
        <div
          className={cn(
            "border-b px-4 py-2 text-sm",
            note.tone === "err"
              ? "border-destructive/30 bg-destructive/10 text-destructive"
              : "border-emerald-500/30 bg-emerald-500/5 text-emerald-700 dark:text-emerald-300",
          )}
        >
          {note.text}
        </div>
      )}

      <CardContent className="grid gap-6 p-4 lg:grid-cols-2">
        <WorkerConfig app={app} worker={worker} />
        <WorkerLogPanel appId={app.app_id} machineId={worker.machine_id} />
      </CardContent>

      <Dialog open={confirmDelete} onOpenChange={(o) => busy === null && setConfirmDelete(o)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete worker</DialogTitle>
            <DialogDescription>
              Terminate <code className="font-mono">{worker.machine_id}</code>? In-flight requests on
              this worker are interrupted. The autoscaler re-provisions a fresh one on the next request
              (unless the fleet is paused).
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setConfirmDelete(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={del}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

function WorkerConfig({ app, worker }: { app: AppRecord; worker: WorkerLite }) {
  const rows: [string, React.ReactNode][] = [
    ["Machine ID", <span className="font-mono">{worker.machine_id}</span>],
    ["Gateway status", worker.alive ? worker.status : "no heartbeat (>60s)"],
    ["GPU", `${app.gpu}${app.gpu_count > 1 ? ` × ${app.gpu_count}` : ""}`],
    ["Sleep level", String(app.sleep_level ?? 1)],
    ["Heartbeat TTL", "30s (worker dropped if no heartbeat for 30s)"],
    ["Idle timeout", app.autoscaler.idle_timeout_s === 0 ? "0 (always-on)" : `${app.autoscaler.idle_timeout_s}s`],
    ["Max workers", String(app.autoscaler.max_containers)],
    ["Tasks / worker", String(app.autoscaler.tasks_per_container)],
    ["Request timeout", `${app.request_timeout_s}s`],
  ];
  return (
    <div className="space-y-3">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Config</h4>
      <dl className="divide-y divide-border/60 rounded-md border border-border">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-center justify-between gap-3 px-3 py-1.5 text-xs">
            <dt className="text-muted-foreground">{k}</dt>
            <dd className="text-right text-foreground">{v}</dd>
          </div>
        ))}
      </dl>
      {(app.models?.length ?? 0) > 0 && (
        <div>
          <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Members
          </h4>
          <div className="overflow-hidden rounded-md border border-border">
            <table className="w-full text-xs">
              <thead className="bg-muted/30 text-left text-[10px] uppercase tracking-wide text-muted-foreground">
                <tr>
                  <th className="px-3 py-1.5 font-medium">Model</th>
                  <th className="px-3 py-1.5 font-medium">TP</th>
                  <th className="px-3 py-1.5 font-medium">GPUs</th>
                </tr>
              </thead>
              <tbody>
                {app.models!.map((m) => (
                  <tr key={m.model} className="border-t border-border/60">
                    <td className="px-3 py-1.5 font-mono">{m.model}</td>
                    <td className="px-3 py-1.5 font-mono">{m.tp}</td>
                    <td className="px-3 py-1.5 font-mono">
                      {m.gpu_indices?.length ? m.gpu_indices.join(",") : "auto"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

type LogSource = "gateway" | "container";
type GatewayEvent = { ts: number; level: string; msg: string };

function WorkerLogPanel({ appId, machineId }: { appId: string; machineId: string }) {
  void appId;
  const [source, setSource] = useState<LogSource>("gateway");
  const [lines, setLines] = useState<string[]>([]);
  const [events, setEvents] = useState<GatewayEvent[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

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
      if (source === "gateway") {
        setEvents((JSON.parse(text) as { events?: GatewayEvent[] }).events ?? []);
      } else {
        setLines((JSON.parse(text) as { lines?: string[] }).lines ?? []);
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [machineId, source]);

  useEffect(() => {
    setLoading(true);
    setLines([]);
    setEvents([]);
    setErr(null);
  }, [source, machineId]);

  useEffect(() => {
    fetchLogs();
    const id = window.setInterval(fetchLogs, 2500);
    return () => window.clearInterval(id);
  }, [fetchLogs]);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const wasAtBottom = useRef(true);
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (wasAtBottom.current) el.scrollTop = el.scrollHeight;
    wasAtBottom.current = distance < 32;
  }, [lines, events, source]);

  const empty = source === "gateway" ? events.length === 0 : lines.length === 0;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Logs</h4>
        <div className="flex items-center gap-2">
          <div className="inline-flex rounded-md border border-border p-0.5">
            {(["gateway", "container"] as LogSource[]).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSource(s)}
                className={cn(
                  "rounded px-2 py-0.5 text-[10px] uppercase tracking-wide",
                  source === s ? "bg-primary/15 text-primary" : "text-muted-foreground hover:text-foreground",
                )}
              >
                {s === "gateway" ? "gateway events" : "container logs"}
              </button>
            ))}
          </div>
          {loading && <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />}
          <Button variant="outline" size="xs" onClick={fetchLogs}>
            <RefreshCw className="h-3 w-3" />
          </Button>
        </div>
      </div>
      {err ? (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {err}
        </div>
      ) : empty ? (
        <div className="rounded-md border border-dashed border-border bg-background/40 px-3 py-4 text-center text-xs text-muted-foreground">
          {loading ? "loading…" : source === "gateway" ? "no gateway events yet" : "no container logs yet"}
        </div>
      ) : (
        <div
          ref={scrollRef}
          className="terminal-block max-h-72 overflow-auto rounded-md border border-border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-200 scrollbar-thin"
        >
          {source === "gateway"
            ? events.map((e, i) => {
                const t = new Date(e.ts * 1000).toLocaleTimeString(undefined, { hour12: false });
                const lvl = (e.level ?? "info").toLowerCase();
                return (
                  <div key={i} className="flex items-start gap-2 border-b border-border/20 py-1 last:border-b-0">
                    <span className="shrink-0 text-[10px] text-zinc-500 tabular-nums">{t}</span>
                    <span
                      className={cn(
                        "shrink-0 text-[9px] uppercase",
                        lvl === "error" ? "text-red-400" : lvl === "warning" ? "text-amber-400" : "text-zinc-500",
                      )}
                    >
                      {lvl}
                    </span>
                    <span className="flex-1 break-words">{e.msg}</span>
                  </div>
                );
              })
            : lines.map((l, i) => <div key={i}>{l}</div>)}
        </div>
      )}
    </div>
  );
}
