"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { ChevronDown, ChevronRight, Copy, Loader2, RefreshCw, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { gateway } from "@/lib/gateway";
import type { AppRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

type Bucket = "in queue" | "in progress" | "completed" | "failed";

type Item = {
  request_id: string;
  bucket: Bucket;
  status: string;
  payload?: { prompt?: string; max_tokens?: number; messages?: unknown };
  endpoint?: string;
  stream?: boolean;
  timeout_s?: number;
  output?: unknown;
  has_result: boolean;
};

type QueueResponse = {
  app_id: string;
  queue_length: number;
  in_progress: number;
  completed: number;
  worker_count: number;
  items: Item[];
  error?: string;
  hint?: string;
};

const POLL_MS = 4000;

const BUCKET_ORDER: Bucket[] = ["in queue", "in progress", "completed", "failed"];

// URL slug ↔ bucket, so the nested filter is shareable via ?q= without encoded
// spaces ("in queue" → "queued"). "all" is the default and omits the param.
const BUCKET_SLUG: Record<Bucket | "all", string> = {
  all: "all",
  "in queue": "queued",
  "in progress": "running",
  completed: "completed",
  failed: "failed",
};
const SLUG_BUCKET: Record<string, Bucket | "all"> = {
  all: "all",
  queued: "in queue",
  running: "in progress",
  completed: "completed",
  failed: "failed",
};

const BUCKET_TONE: Record<Bucket, string> = {
  "in queue":    "bg-status-init/15 text-status-init",
  "in progress": "bg-status-idle/15 text-status-idle",
  completed:     "bg-status-active/15 text-status-active",
  failed:        "bg-status-down/15 text-status-down",
};

export function QueueTab({ app }: { app: AppRecord }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [data, setData] = useState<QueueResponse | null>(null);
  const [err, setErr] = useState<{ msg: string; hint?: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [flushing, setFlushing] = useState(false);
  const [confirmFlush, setConfirmFlush] = useState(false);

  // The nested filter is URL-driven (?q=queued|running|completed|failed; "all"
  // omits it) so it's shareable + survives back/forward — single source of truth.
  const filter: Bucket | "all" = SLUG_BUCKET[searchParams.get("q") ?? ""] ?? "all";
  const selectFilter = useCallback(
    (b: Bucket | "all") => {
      const params = new URLSearchParams(searchParams.toString());
      if (b === "all") params.delete("q");
      else params.set("q", BUCKET_SLUG[b]);
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    },
    [router, pathname, searchParams],
  );

  // Clicking a request id deep-links it via ?req=<id> (shareable). Clicking the
  // same id again clears it.
  const selectedReq = searchParams.get("req") ?? "";
  const selectReq = useCallback(
    (id: string) => {
      const params = new URLSearchParams(searchParams.toString());
      if (params.get("req") === id) params.delete("req");
      else params.set("req", id);
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    },
    [router, pathname, searchParams],
  );

  const fetchQueue = useCallback(async () => {
    setLoading(true);
    try {
      const r = await fetch(`/api/cluster/queue?app=${encodeURIComponent(app.app_id)}`, {
        cache: "no-store",
      });
      const body = (await r.json()) as QueueResponse;
      if (!r.ok) {
        setErr({ msg: body.error ?? r.statusText, hint: body.hint });
        return;
      }
      setErr(null);
      setData(body);
    } catch (e) {
      setErr({ msg: e instanceof Error ? e.message : String(e) });
    } finally {
      setLoading(false);
    }
  }, [app.app_id]);

  useEffect(() => {
    fetchQueue();
    const id = window.setInterval(fetchQueue, POLL_MS);
    return () => window.clearInterval(id);
  }, [fetchQueue]);

  const queued = data?.queue_length ?? 0;
  const onFlush = useCallback(async () => {
    setConfirmFlush(false);
    setFlushing(true);
    try {
      const r = await gateway.flushQueue(app.app_id);
      toast.success(
        `Cleared ${r.cancelled} queued job${r.cancelled === 1 ? "" : "s"}`,
        { duration: 3000 },
      );
      fetchQueue();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setFlushing(false);
    }
  }, [app.app_id, fetchQueue]);

  const cap = app.autoscaler.max_containers * app.autoscaler.tasks_per_container;

  const filtered = useMemo(() => {
    if (!data) return [];
    const items = filter === "all" ? data.items : data.items.filter((i) => i.bucket === filter);
    // Order: in queue (FIFO), in progress, completed, failed.
    return [...items].sort(
      (a, b) => BUCKET_ORDER.indexOf(a.bucket) - BUCKET_ORDER.indexOf(b.bucket),
    );
  }, [data, filter]);

  // Position numbers only make sense within the queued portion. Build a
  // small index so each in-queue row gets its FIFO #.
  const queuePositions = useMemo(() => {
    const m = new Map<string, number>();
    if (!data) return m;
    let pos = 1;
    for (const it of data.items) {
      if (it.bucket === "in queue") {
        m.set(it.request_id, pos++);
      }
    }
    return m;
  }, [data]);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <Stat label="In queue" value={data?.queue_length ?? 0} />
        <Stat label="In progress" value={data?.in_progress ?? 0} />
        <Stat label="Completed (live)" value={data?.completed ?? 0} />
        <Stat label="Workers" value={data?.worker_count ?? 0} />
        <Stat label="Capacity" value={cap} />
        <div className="flex-1" />
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="xs"
            onClick={() => setConfirmFlush(true)}
            disabled={flushing || queued === 0}
            title={queued === 0 ? "No queued jobs to flush" : `Drop ${queued} queued job(s)`}
            className="text-destructive hover:text-destructive"
          >
            {flushing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
            Flush queue
          </Button>
          <Button variant="outline" size="xs" onClick={fetchQueue} disabled={loading}>
            {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
            Refresh
          </Button>
        </div>
      </div>

      <div className="flex gap-1 border-b border-border">
        {(["all", ...BUCKET_ORDER] as const).map((b) => {
          const count =
            b === "all"
              ? data?.items.length ?? 0
              : data?.items.filter((i) => i.bucket === b).length ?? 0;
          return (
            <button
              key={b}
              onClick={() => selectFilter(b)}
              className={cn(
                "relative px-3 py-1.5 text-xs transition-colors",
                filter === b
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {b} <span className="text-muted-foreground">({count})</span>
              {filter === b && <span className="absolute -bottom-px left-0 right-0 h-0.5 bg-primary" />}
            </button>
          );
        })}
      </div>

      {err && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <div className="font-medium">Couldn&apos;t read the queue</div>
          <div className="mt-0.5 break-all text-xs opacity-80">{err.msg}</div>
          {err.hint && <div className="mt-1 text-xs opacity-90">{err.hint}</div>}
        </div>
      )}

      <Card className="overflow-hidden">
        <table className="w-full text-sm">
          <thead className="border-b border-border bg-muted/20 text-left text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="w-6 px-2 py-2"></th>
              <th className="px-3 py-2 font-medium">Pos</th>
              <th className="px-3 py-2 font-medium">Request ID</th>
              <th className="px-3 py-2 font-medium">Status</th>
              <th className="px-3 py-2 font-medium">Endpoint</th>
              <th className="px-3 py-2 font-medium">Input → Output</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((it) => (
              <Row
                key={it.request_id}
                item={it}
                position={queuePositions.get(it.request_id)}
                selected={it.request_id === selectedReq}
                onSelect={selectReq}
              />
            ))}
            {data && filtered.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-12 text-center text-sm text-muted-foreground">
                  {filter === "all"
                    ? "No jobs — fire a request to populate this view."
                    : `No ${filter} jobs.`}
                </td>
              </tr>
            )}
            {!data && !err && (
              <tr>
                <td colSpan={6} className="px-4 py-12 text-center text-sm text-muted-foreground">
                  Loading…
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Card>

      <p className="text-xs text-muted-foreground">
        Source: <code className="font-mono">LRANGE queue:{app.app_id}</code> + <code className="font-mono">SCAN result:req-*</code> via <code className="font-mono">kubectl exec</code> on{" "}
        <code className="font-mono">serverlessgpu-redis-0</code>. Result blobs filtered to <code className="font-mono">output.model = {app.app_id}</code>. TTL on result keys is 1 h.
      </p>

      <Dialog open={confirmFlush} onOpenChange={(o) => !flushing && setConfirmFlush(o)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Flush queue</DialogTitle>
            <DialogDescription>
              Flush {queued} queued job{queued === 1 ? "" : "s"} on{" "}
              <code className="font-mono">{app.app_id}</code>? Jobs already running on a worker
              are not affected.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setConfirmFlush(false)} disabled={flushing}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={onFlush} disabled={flushing}>
              {flushing && <Loader2 className="h-4 w-4 animate-spin" />}
              Flush queue
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function Row({
  item,
  position,
  selected,
  onSelect,
}: {
  item: Item;
  position?: number;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  // Deep-linked row (?req=<id>) starts expanded; clicking the id also opens it.
  const [open, setOpen] = useState(selected);
  const expanded = open || selected;

  const inputSummary = useMemo(() => summariseInput(item), [item]);
  const outputSummary = useMemo(() => summariseOutput(item), [item]);

  return (
    <>
      <tr className={cn("border-b border-border/60 last:border-b-0", selected && "bg-primary/5")}>
        <td className="px-2 py-2 align-top">
          <button
            onClick={() => setOpen((v) => !v)}
            className="text-muted-foreground hover:text-foreground"
            aria-label="Toggle details"
          >
            {expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
          </button>
        </td>
        <td className="px-3 py-2 font-mono text-xs text-muted-foreground">
          {position != null ? `#${position}` : "—"}
        </td>
        <td className="px-3 py-2">
          <div className="flex items-center gap-1.5">
            <button
              className={cn("font-mono text-xs hover:text-primary", selected && "font-medium text-primary")}
              onClick={() => {
                onSelect(item.request_id);
                setOpen(true);
              }}
              title="Select — adds ?req=<id> to the URL"
            >
              {item.request_id}
            </button>
            <button
              onClick={() => {
                navigator.clipboard.writeText(item.request_id);
                toast.success("Request ID copied", { duration: 3000 });
              }}
              className="text-muted-foreground hover:text-foreground"
              title="Copy request_id"
              aria-label="Copy request_id"
            >
              <Copy className="h-3 w-3" />
            </button>
          </div>
        </td>
        <td className="px-3 py-2">
          <span
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs",
              BUCKET_TONE[item.bucket],
            )}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-current" />
            {item.bucket}
          </span>
        </td>
        <td className="px-3 py-2 font-mono text-xs text-muted-foreground">
          {item.endpoint ?? "/run"}
          {item.stream && <span className="ml-1 text-status-init">(stream)</span>}
        </td>
        <td className="max-w-md truncate px-3 py-2 text-xs">
          <span className="text-muted-foreground">{inputSummary}</span>
          {outputSummary && (
            <>
              <span className="mx-1 text-muted-foreground">→</span>
              <span className="text-foreground">{outputSummary}</span>
            </>
          )}
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-border/60 bg-muted/20">
          <td colSpan={6} className="px-4 py-3">
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              <Block title="Input">
                <Pre>
                  {item.payload
                    ? JSON.stringify(item.payload, null, 2)
                    : "(no payload — gateway only stores it on the queue; this job finished before the UI proxy ever saw it queued. Future jobs will keep their inputs visible after they complete.)"}
                </Pre>
              </Block>
              <Block title="Output">
                {item.output != null ? (
                  <Pre>{JSON.stringify(item.output, null, 2)}</Pre>
                ) : item.has_result ? (
                  <Pre>{`status: ${item.status}\n(no output yet)`}</Pre>
                ) : (
                  <Pre>fetching…</Pre>
                )}
              </Block>
            </div>
            <div className="mt-2 flex items-center gap-3 text-[11px] text-muted-foreground">
              <span>timeout {item.timeout_s ?? 600}s</span>
              <span>·</span>
              <span>{item.endpoint ?? "/run"}</span>
              <Button
                variant="ghost"
                size="xs"
                className="ml-auto"
                onClick={() => {
                  navigator.clipboard.writeText(JSON.stringify(item, null, 2));
                  toast.success("Job JSON copied", { duration: 3000 });
                }}
              >
                <Copy className="h-3 w-3" />
                Copy JSON
              </Button>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <span className="text-muted-foreground">
      <span className="font-mono text-foreground">{value}</span> {label}
    </span>
  );
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">{title}</div>
      {children}
    </div>
  );
}

function Pre({ children }: { children: React.ReactNode }) {
  return (
    <pre className="max-h-72 overflow-auto rounded-md border border-border bg-background/40 p-2 font-mono text-[11px] leading-relaxed scrollbar-thin">
      {children}
    </pre>
  );
}

function summariseInput(item: Item): string {
  const p = item.payload;
  if (!p) return "(no payload)";
  if (typeof p.prompt === "string") return truncate(p.prompt, 60);
  if (Array.isArray(p.messages)) {
    const last = p.messages.at(-1) as { content?: string } | undefined;
    return truncate(last?.content ?? "(messages…)", 60);
  }
  return truncate(JSON.stringify(p), 60);
}

function summariseOutput(item: Item): string | null {
  const out = item.output as
    | {
        choices?: Array<{ text?: string; message?: { content?: string }; delta?: { content?: string } }>;
      }
    | undefined;
  if (!out || !Array.isArray(out.choices) || out.choices.length === 0) return null;
  const c = out.choices[0];
  const text = c.text ?? c.message?.content ?? c.delta?.content ?? "";
  if (!text) return null;
  return truncate(text.replace(/\s+/g, " ").trim(), 80);
}

function truncate(s: string, n: number) {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}
