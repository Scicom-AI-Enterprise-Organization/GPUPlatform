"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Copy,
  ExternalLink,
  Loader2,
  Play,
  RefreshCw,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { gateway } from "@/lib/gateway";
import { cn } from "@/lib/utils";

type RequestStatus =
  | "pending"
  | "in queue"
  | "in progress"
  | "completed"
  | "ready"
  | "failed"
  | "timeout"
  | "cancelled"
  | "expired"
  | "unknown";

type StoredRequest = {
  id: string;          // request_id
  ts: number;          // ms epoch when first seen
  prompt: string;      // truncated prompt for display
  status: RequestStatus;
  output?: unknown;    // last fetched output
  error?: string;
  app_id: string;
};

const STORAGE_KEY = (appId: string) => `serverless-ui:requests:${appId}`;
const POLL_MS = 4_000;
const MAX_HISTORY = 100;

export function RequestsTab({ appId }: { appId?: string } = {}) {
  // The endpoint detail page renders this with no props today; we lift app_id
  // out of the URL via a hook below. Accepting an optional prop keeps the
  // door open for callers that want to pass it explicitly.
  const resolvedAppId = appId ?? useAppIdFromPath();
  return resolvedAppId ? <RequestsTabInner appId={resolvedAppId} /> : null;
}

function useAppIdFromPath(): string {
  // Avoids next/navigation params plumbing — the URL is /serverless/<id>.
  const [id, setId] = useState<string>("");
  useEffect(() => {
    const seg = window.location.pathname.split("/").filter(Boolean);
    const sIdx = seg.indexOf("serverless");
    setId(seg[sIdx + 1] ?? "");
  }, []);
  return id;
}

function RequestsTabInner({ appId }: { appId: string }) {
  const [history, setHistory] = useState<StoredRequest[]>([]);
  const historyRef = useRef(history);
  historyRef.current = history;

  // Load + persist history.
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY(appId));
      if (raw) setHistory(JSON.parse(raw));
    } catch {
      // ignore
    }
  }, [appId]);

  const persist = useCallback(
    (next: StoredRequest[]) => {
      setHistory(next);
      try {
        window.localStorage.setItem(STORAGE_KEY(appId), JSON.stringify(next));
      } catch {
        // ignore
      }
    },
    [appId],
  );

  const upsert = useCallback(
    (req: StoredRequest) => {
      const cur = historyRef.current;
      const others = cur.filter((r) => r.id !== req.id);
      persist([req, ...others].slice(0, MAX_HISTORY));
    },
    [persist],
  );

  const remove = useCallback(
    (id: string) => persist(historyRef.current.filter((r) => r.id !== id)),
    [persist],
  );

  const clearAll = useCallback(() => persist([]), [persist]);

  // Poll any request that's not yet in a terminal state.
  useEffect(() => {
    const tick = async () => {
      const cur = historyRef.current;
      const live = cur.filter((r) => !isTerminal(r.status));
      if (live.length === 0) return;
      await Promise.all(
        live.map(async (r) => {
          try {
            const res = await fetch(`/api/proxy/result/${encodeURIComponent(r.id)}`, {
              cache: "no-store",
            });
            if (res.status === 404) {
              upsert({ ...r, status: "expired" });
              return;
            }
            const body = await res.json();
            const status = normalizeStatus(body?.status ?? "unknown");
            upsert({ ...r, status, output: body?.output ?? r.output });
          } catch (e) {
            upsert({ ...r, status: "unknown", error: e instanceof Error ? e.message : String(e) });
          }
        }),
      );
    };
    tick();
    const id = window.setInterval(tick, POLL_MS);
    return () => window.clearInterval(id);
  }, [upsert]);

  // ---- Send a test request ----
  const [prompt, setPrompt] = useState("Hello, world");
  const [maxTokens, setMaxTokens] = useState(16);
  const [sending, setSending] = useState(false);

  async function send() {
    if (!prompt.trim()) {
      toast.error("Prompt is required.", { duration: 5000 });
      return;
    }
    setSending(true);
    try {
      const r = await fetch(`/api/proxy/run/${encodeURIComponent(appId)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_tokens: maxTokens }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body?.detail ?? body?.error ?? r.statusText);
      const stored: StoredRequest = {
        id: body.request_id,
        ts: Date.now(),
        prompt: prompt.slice(0, 80),
        status: "pending",
        app_id: appId,
      };
      upsert(stored);
      toast.success(`Queued ${stored.id}`, { duration: 3000 });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e)), { duration: 5000 };
    } finally {
      setSending(false);
    }
  }

  // ---- Look up a known request id ----
  const [lookup, setLookup] = useState("");
  const trimmed = lookup.trim();
  const resultUrl = trimmed ? `${gateway.baseUrl}/result/${trimmed}` : "";
  const curlCmd = trimmed ? `curl -X GET '${resultUrl}'` : "";

  async function fetchAndAdd() {
    if (!trimmed) return;
    upsert({
      id: trimmed,
      ts: Date.now(),
      prompt: "(imported)",
      status: "pending",
      app_id: appId,
    });
    setLookup("");
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <div>
            <CardTitle className="text-sm font-medium">Send a test request</CardTitle>
            <p className="text-xs text-muted-foreground">
              Fires <code className="font-mono">POST /run/{appId}</code> via the gateway and tracks the result here.
            </p>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Prompt"
            rows={2}
            className="font-mono text-sm"
          />
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2 text-xs text-muted-foreground">
              max_tokens
              <Input
                type="number"
                min={1}
                max={2048}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Math.max(1, Number(e.target.value)))}
                className="h-8 w-20 font-mono"
              />
            </label>
            <div className="flex-1" />
            <Button onClick={send} disabled={sending}>
              {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              Send
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
          <div>
            <CardTitle className="text-sm font-medium">Request history</CardTitle>
            <p className="text-xs text-muted-foreground">
              Tracked per browser. {history.length} of {MAX_HISTORY} max.
            </p>
          </div>
          {history.length > 0 && (
            <Button variant="ghost" size="xs" onClick={clearAll}>
              Clear all
            </Button>
          )}
        </CardHeader>
        <div className="flex items-center gap-2 border-y border-border bg-muted/30 px-3 py-2">
          <Search className="h-3.5 w-3.5 text-muted-foreground" />
          <Input
            value={lookup}
            onChange={(e) => setLookup(e.target.value)}
            placeholder="Paste a request_id you fired via curl, then Add"
            className="h-8 border-0 bg-transparent font-mono shadow-none focus-visible:ring-0"
          />
          <Button size="xs" variant="outline" onClick={fetchAndAdd} disabled={!trimmed}>
            Add to history
          </Button>
        </div>
        <CardContent className="px-0 py-0">
          {trimmed && (
            <div className="border-b border-border px-3 py-2">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">cURL</div>
              <div className="relative mt-1">
                <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-2 font-mono text-[11px] leading-relaxed text-foreground">
                  {curlCmd}
                </pre>
                <div className="absolute right-1.5 top-1.5 flex gap-1">
                  <Button
                    variant="outline"
                    size="icon-xs"
                    onClick={() => {
                      navigator.clipboard.writeText(curlCmd);
                      toast.success("cURL copied", { duration: 3000 });
                    }}
                    aria-label="Copy cURL"
                  >
                    <Copy className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon-xs"
                    onClick={() => {
                      navigator.clipboard.writeText(resultUrl);
                      toast.success("URL copied", { duration: 3000 });
                    }}
                    aria-label="Copy URL"
                  >
                    <ExternalLink className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            </div>
          )}
          <table className="w-full text-sm">
            <thead className="border-b border-border bg-muted/20 text-left text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="w-6 px-2 py-2"></th>
                <th className="px-3 py-2 font-medium">Request ID</th>
                <th className="px-3 py-2 font-medium">Status</th>
                <th className="px-3 py-2 font-medium">Prompt</th>
                <th className="px-3 py-2 font-medium">When</th>
                <th className="px-3 py-2 font-medium" />
              </tr>
            </thead>
            <tbody>
              {history.map((r) => (
                <RequestRow key={r.id} req={r} onRemove={() => remove(r.id)} />
              ))}
              {history.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-4 py-12 text-center text-sm text-muted-foreground">
                    No requests tracked yet — send one above or paste a curl-fired ID into the lookup bar.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
}

function RequestRow({ req, onRemove }: { req: StoredRequest; onRemove: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr className="border-b border-border/60 last:border-b-0">
        <td className="px-2 py-2 align-top">
          <button
            onClick={() => setOpen((v) => !v)}
            className="text-muted-foreground hover:text-foreground"
            aria-label={open ? "Collapse" : "Expand"}
          >
            {open ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
          </button>
        </td>
        <td className="px-3 py-2 font-mono text-xs">
          <button
            onClick={() => {
              navigator.clipboard.writeText(req.id);
              toast.success("ID copied", { duration: 3000 });
            }}
            className="text-left hover:text-primary"
            title="Copy request_id"
          >
            {req.id}
          </button>
        </td>
        <td className="px-3 py-2">
          <StatusPill status={req.status} />
        </td>
        <td className="max-w-xs truncate px-3 py-2 text-xs text-muted-foreground" title={req.prompt}>
          {req.prompt}
        </td>
        <td className="px-3 py-2 text-xs text-muted-foreground">{relTime(req.ts)}</td>
        <td className="px-3 py-2 text-right">
          <Button variant="ghost" size="icon-xs" onClick={onRemove} aria-label="Remove">
            <X className="h-3 w-3" />
          </Button>
        </td>
      </tr>
      {open && (
        <tr className="border-b border-border/60 bg-muted/20">
          <td colSpan={6} className="px-4 py-3">
            {req.error ? (
              <div className="text-xs text-destructive">{req.error}</div>
            ) : req.output != null ? (
              <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-all rounded-md border border-border bg-background/40 p-2 font-mono text-[11px] leading-relaxed scrollbar-thin">
                {JSON.stringify(req.output, null, 2)}
              </pre>
            ) : (
              <div className="text-xs text-muted-foreground">no output yet</div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

function StatusPill({ status }: { status: RequestStatus }) {
  const tone =
    status === "completed" || status === "ready"
      ? "bg-status-active/15 text-status-active"
      : status === "in progress"
        ? "bg-status-idle/15 text-status-idle"
        : status === "pending" || status === "in queue"
          ? "bg-status-init/15 text-status-init"
          : status === "expired" || status === "unknown"
            ? "bg-muted text-muted-foreground"
            : "bg-status-down/15 text-status-down";
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs", tone)}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {status}
    </span>
  );
}

function isTerminal(status: RequestStatus) {
  return ["completed", "ready", "failed", "timeout", "cancelled", "expired"].includes(status);
}

function normalizeStatus(s: string): RequestStatus {
  const v = s.toLowerCase().trim();
  if (["completed", "ready", "pending", "in queue", "in progress", "failed", "timeout", "cancelled", "expired"].includes(v)) {
    return v as RequestStatus;
  }
  return "unknown";
}

function relTime(ts: number) {
  const diff = Math.max(0, (Date.now() - ts) / 1000);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}
