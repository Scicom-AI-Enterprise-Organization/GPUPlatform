"use client";

import { useCallback, useEffect, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { Copy, Eye, EyeOff, Loader2, Network, Pencil, RefreshCw, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { gateway } from "@/lib/gateway";
import { cn } from "@/lib/utils";
import type { ProxyEndpoint, ProxyRequest, ProxyUpstreamHealth } from "@/lib/types";
import { BaseUrlToggle, type UrlTarget } from "@/components/console/base-url-toggle";
import { ProxyPlayground } from "./proxy-playground";
import { ProxyStress } from "./proxy-stress";

const POLL_MS = 4000;
const TABS = [
  { value: "overview", label: "Overview" },
  { value: "playground", label: "Playground" },
  { value: "stress", label: "Stress test" },
  { value: "queue", label: "Queue" },
] as const;
const BUCKETS = ["queued", "running", "completed", "cancelled", "failed"] as const;
type ProxyTab = (typeof TABS)[number]["value"];
const TAB_VALUES = TABS.map((t) => t.value) as readonly string[];

function fmtAgo(iso: string | null | undefined): string {
  if (!iso) return "—";
  const s = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}

const maskToken = (t: string) => (t.length > 12 ? `${t.slice(0, 6)}…${t.slice(-4)}` : "•".repeat(Math.max(4, t.length)));

const STATUS_TONE: Record<string, string> = {
  queued: "bg-status-init/15 text-status-init",
  running: "bg-status-idle/15 text-status-idle",
  completed: "bg-status-active/15 text-status-active",
  failed: "bg-status-down/15 text-status-down",
  cancelled: "bg-muted text-muted-foreground",
};

export function ProxyDetail({ initial, baseUrl }: { initial: ProxyEndpoint; baseUrl: string }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const ep = initial;
  const aliases = Array.from(new Set(ep.upstreams.flatMap((u) => Object.keys(u.models))));
  const proxyBase = `${baseUrl}/proxy/${ep.name}/v1`;

  // The active tab is derived from the URL (?tab=), and each trigger is a real
  // <Link> — so right-click / middle-click / ⌘-click "open in new tab" works.
  // useSearchParams is reactive to soft navigations, so a normal click still
  // switches tabs in place without a full reload.
  const tabParam = searchParams.get("tab");
  const tab: ProxyTab = tabParam && TAB_VALUES.includes(tabParam) ? (tabParam as ProxyTab) : "overview";
  const tabHref = (v: ProxyTab) => {
    const p = new URLSearchParams(searchParams.toString());
    p.set("tab", v);
    return `${pathname}?${p.toString()}`;
  };

  const [health, setHealth] = useState<ProxyUpstreamHealth[]>([]);
  const [reqs, setReqs] = useState<ProxyRequest[]>([]);
  const [token, setToken] = useState<string | null>(null);
  const [reveal, setReveal] = useState(false);
  const [urlTarget, setUrlTarget] = useState<UrlTarget>("public");
  const [filter, setFilter] = useState<"all" | (typeof BUCKETS)[number]>("all");
  const [flushing, setFlushing] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    let abort = false;
    fetch("/api/auth/token", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((b) => { if (!abort) setToken(b?.token ?? null); })
      .catch(() => {});
    return () => { abort = true; };
  }, []);

  const poll = useCallback(async () => {
    try {
      const [h, r] = await Promise.all([gateway.getProxyHealth(ep.id), gateway.getProxyRequests(ep.id)]);
      setHealth(h);
      setReqs(r);
    } catch {
      /* transient */
    }
  }, [ep.id]);
  useEffect(() => {
    poll();
    const t = window.setInterval(poll, POLL_MS);
    return () => window.clearInterval(t);
  }, [poll]);

  const onCancel = async (rid: string) => {
    try { await gateway.cancelProxyRequest(ep.id, rid); poll(); }
    catch (e) { alert(e instanceof Error ? e.message : String(e)); }
  };
  const queuedCount = reqs.filter((r) => r.status === "queued").length;
  const onFlush = async () => {
    if (!confirm(`Flush ${queuedCount} queued request${queuedCount === 1 ? "" : "s"}? Running requests are not affected.`)) return;
    setFlushing(true);
    try {
      const r = await gateway.flushProxyQueue(ep.id);
      toast.success(`Flushed ${r.flushed} queued request${r.flushed === 1 ? "" : "s"}`, { duration: 3000 });
      poll();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setFlushing(false);
    }
  };
  const onRefresh = async () => { setRefreshing(true); await poll(); setRefreshing(false); };
  const count = (s: string) => reqs.filter((r) => r.status === s).length;
  const filtered = filter === "all" ? reqs : reqs.filter((r) => r.status === filter);
  const onDelete = async () => {
    if (!confirm(`Delete proxy endpoint "${ep.name}"?`)) return;
    try { await gateway.deleteProxy(ep.id); router.push("/proxy"); router.refresh(); }
    catch (e) { alert(e instanceof Error ? e.message : String(e)); }
  };

  const model0 = aliases[0] ?? "qwen";
  const visToken = reveal && token ? token : token ? maskToken(token) : "sgpu_…";
  const realToken = token ?? "YOUR_API_KEY";

  const internalBase = process.env.NEXT_PUBLIC_GATEWAY_INTERNAL_URL ?? "";
  // Snippet base only — the header URL + Playground keep the public base (an
  // in-cluster Service DNS isn't reachable from the browser). "internal" lets
  // an in-cluster caller copy a snippet that skips the ingress hop.
  const snippetBase =
    urlTarget === "internal" && internalBase ? `${internalBase}/proxy/${ep.name}/v1` : proxyBase;

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      {/* header zone — title + actions + tab bar (matches the serverless detail) */}
      <div className="border-b border-border bg-sidebar/40 px-6 pt-4 lg:px-10">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-muted/60 text-muted-foreground">
              <Network className="h-5 w-5" />
            </div>
            <div>
              <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight">
                {ep.name}
                {!ep.enabled && <span className="rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] uppercase text-muted-foreground">disabled</span>}
              </h1>
              <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
                <span className="font-mono">{proxyBase}</span>
                <span>·</span><span>{ep.upstreams.length} upstream{ep.upstreams.length === 1 ? "" : "s"}</span>
                <span>·</span><span>{aliases.length} model{aliases.length === 1 ? "" : "s"}</span>
                <span>·</span><span>queue {ep.max_concurrency || "∞"}</span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button asChild variant="outline" size="sm"><Link href={`/proxy/${ep.id}/edit`}><Pencil className="h-4 w-4" /> Edit</Link></Button>
            <Button variant="outline" size="sm" className="text-destructive hover:text-destructive" onClick={onDelete}><Trash2 className="h-4 w-4" /> Delete</Button>
          </div>
        </div>
        <Tabs value={tab} className="mt-3">
          <TabsList variant="line" className="bg-transparent">
            {TABS.map((t) => (
              <TabsTrigger key={t.value} value={t.value} asChild>
                <Link href={tabHref(t.value)} scroll={false}>{t.label}</Link>
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      {/* scroll zone — second Tabs holds the content (same value/handler) */}
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 scrollbar-thin">
        <Tabs value={tab}>
          {/* ---- Overview ---- */}
          <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader className="flex-row items-center justify-between gap-2">
              <div className="flex items-center gap-3">
                <CardTitle className="text-sm font-medium">Run a job</CardTitle>
                <span className="text-xs text-muted-foreground">OpenAI-compatible · priority + health-aware failover.</span>
              </div>
              <div className="flex items-center gap-2">
                {internalBase && <BaseUrlToggle value={urlTarget} onChange={setUrlTarget} />}
                {token && (
                  <Button variant="outline" size="xs" onClick={() => setReveal((v) => !v)}>
                    {reveal ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}{reveal ? "Hide" : "Reveal"} key
                  </Button>
                )}
              </div>
            </CardHeader>
            <CardContent>
              <div className="mb-3 flex flex-wrap items-center gap-1 text-xs text-muted-foreground">
                Models:
                {aliases.map((a) => <span key={a} className="rounded bg-primary/10 px-1.5 py-0.5 font-mono text-[11px] text-primary">{a}</span>)}
              </div>
              <Tabs defaultValue="curl">
                <TabsList variant="line" className="bg-transparent">
                  <TabsTrigger value="curl">cURL</TabsTrigger>
                  <TabsTrigger value="curl-stream">cURL (stream)</TabsTrigger>
                  <TabsTrigger value="embeddings">Embeddings</TabsTrigger>
                  <TabsTrigger value="openai">OpenAI client</TabsTrigger>
                </TabsList>
                <TabsContent value="curl" className="mt-3 space-y-3">
                  <p className="text-sm text-muted-foreground">
                    OpenAI <code className="font-mono">/proxy/{ep.name}/v1/chat/completions</code> — returns the full completion JSON in one call.
                  </p>
                  <CodeBlock display={curlChat(snippetBase, visToken, model0)} copy={curlChat(snippetBase, realToken, model0)} />
                </TabsContent>
                <TabsContent value="curl-stream" className="mt-3 space-y-3">
                  <p className="text-sm text-muted-foreground">
                    Same endpoint with <code className="font-mono">&quot;stream&quot;: true</code> — token-by-token Server-Sent Events.
                  </p>
                  <CodeBlock display={curlChatStream(snippetBase, visToken, model0)} copy={curlChatStream(snippetBase, realToken, model0)} />
                </TabsContent>
                <TabsContent value="embeddings" className="mt-3 space-y-3">
                  <p className="text-sm text-muted-foreground">
                    OpenAI <code className="font-mono">/proxy/{ep.name}/v1/embeddings</code> — point <code className="font-mono">model</code> at an embedding alias to get the vector list back in one call (unary, never streamed).
                  </p>
                  <CodeBlock display={curlEmbeddings(snippetBase, visToken, model0)} copy={curlEmbeddings(snippetBase, realToken, model0)} />
                </TabsContent>
                <TabsContent value="openai" className="mt-3 space-y-3">
                  <p className="text-sm text-muted-foreground">
                    Point any OpenAI client at <code className="font-mono">{snippetBase}</code> — set <code className="font-mono">model</code> to one of the aliases above.
                  </p>
                  <CodeBlock display={openaiSnippet(snippetBase, visToken, model0)} copy={openaiSnippet(snippetBase, realToken, model0)} />
                </TabsContent>
              </Tabs>
              {urlTarget === "internal" && (
                <p className="mt-3 text-[11px] text-muted-foreground">
                  In-cluster URL — reachable only from pods in the same Kubernetes cluster; bypasses the public ingress.
                </p>
              )}
              <p className="mt-3 text-[11px] text-muted-foreground">
                Shown with your session token. For scripts/CI, use a long-lived key from <Link href="/api-keys" className="underline underline-offset-2 hover:text-foreground">API tokens</Link>.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">Upstreams <span className="text-[11px] font-normal text-muted-foreground">· live health</span></CardTitle></CardHeader>
            <CardContent className="space-y-2">
              {ep.upstreams.map((u) => {
                const h = health.find((x) => x.upstream_id === u.id);
                const alive = h?.alive;
                const dot = alive === true ? "bg-emerald-500" : alive === false ? "bg-destructive" : "bg-muted-foreground/50";
                return (
                  <div key={u.id} className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md border border-border/60 px-3 py-2 text-sm">
                    <span className={cn("h-2 w-2 rounded-full", dot, h?.stale && "opacity-50")} />
                    <span className="font-medium">{u.name}</span>
                    {!u.enabled && <span className="rounded border border-border bg-muted px-1 text-[10px] uppercase text-muted-foreground">off</span>}
                    <span className="font-mono text-xs text-muted-foreground">{u.base_url}</span>
                    <span className="text-xs text-muted-foreground">pri {u.priority}</span>
                    <span className="text-xs text-muted-foreground">{Object.keys(u.models).join(", ")}</span>
                    <span className="ml-auto text-xs text-muted-foreground">
                      {alive == null ? "not probed" : alive ? `alive · ${h?.latency_ms ?? "?"}ms` : `down · ${h?.error ?? ""}`}
                    </span>
                  </div>
                );
              })}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ---- Playground ---- */}
        <TabsContent value="playground">
          <ProxyPlayground name={ep.name} aliases={aliases} baseUrl={baseUrl} />
        </TabsContent>

        {/* ---- Stress test ---- */}
        <TabsContent value="stress">
          <ProxyStress name={ep.name} aliases={aliases} />
        </TabsContent>

        {/* ---- Queue ---- */}
        <TabsContent value="queue" className="space-y-3">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
            <Stat label="In queue" value={count("queued")} />
            <Stat label="Running" value={count("running")} />
            <Stat label="Completed" value={count("completed")} />
            <Stat label="Cancelled" value={count("cancelled")} />
            <Stat label="Failed" value={count("failed")} />
            <Stat label="Capacity" value={ep.max_concurrency || "∞"} />
            <div className="flex-1" />
            <div className="flex items-center gap-2">
              <Button variant="outline" size="xs" onClick={onFlush} disabled={flushing || queuedCount === 0}
                      title={queuedCount === 0 ? "No queued requests to flush" : `Drop ${queuedCount} queued request(s)`}
                      className="text-destructive hover:text-destructive">
                {flushing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />} Flush queue
              </Button>
              <Button variant="outline" size="xs" onClick={onRefresh} disabled={refreshing}>
                {refreshing ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />} Refresh
              </Button>
            </div>
          </div>

          <div className="flex gap-1 border-b border-border">
            {(["all", ...BUCKETS] as const).map((b) => {
              const n = b === "all" ? reqs.length : count(b);
              return (
                <button key={b} onClick={() => setFilter(b)}
                        className={cn("relative px-3 py-1.5 text-xs transition-colors", filter === b ? "text-foreground" : "text-muted-foreground hover:text-foreground")}>
                  {b} <span className="text-muted-foreground">({n})</span>
                  {filter === b && <span className="absolute -bottom-px left-0 right-0 h-0.5 bg-primary" />}
                </button>
              );
            })}
          </div>

          <Card className="overflow-hidden">
            <table className="w-full text-sm">
              <thead className="border-b border-border bg-muted/20 text-left text-xs uppercase tracking-wide text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium">User</th>
                  <th className="px-3 py-2 font-medium">Model</th>
                  <th className="px-3 py-2 font-medium">Upstream</th>
                  <th className="px-3 py-2 font-medium">Code</th>
                  <th className="px-3 py-2 font-medium">Latency</th>
                  <th className="px-3 py-2 font-medium">Tokens</th>
                  <th className="px-3 py-2 font-medium">When</th>
                  <th className="px-3 py-2"></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => (
                  <tr key={r.id} className="border-b border-border/60 last:border-0">
                    <td className="px-3 py-2">
                      <span className={cn("rounded px-1.5 py-0.5 text-[11px] font-medium", STATUS_TONE[r.status] ?? "")}>{r.status}</span>
                      {r.is_stream && <span className="ml-1 text-[10px] text-status-init">stream</span>}
                    </td>
                    <td className="px-3 py-2 text-xs">{r.owner ?? "—"}</td>
                    <td className="px-3 py-2 font-mono text-xs">{r.model}</td>
                    <td className="px-3 py-2 text-xs">{r.upstream ?? "—"}</td>
                    <td className="px-3 py-2 text-xs">{r.status_code ?? "—"}</td>
                    <td className="px-3 py-2 text-xs">{r.latency_ms != null ? `${r.latency_ms}ms` : "—"}</td>
                    <td className="px-3 py-2 text-xs">{r.prompt_tokens != null ? `${r.prompt_tokens}/${r.completion_tokens ?? "?"}` : "—"}</td>
                    <td className="px-3 py-2 text-xs text-muted-foreground">{fmtAgo(r.created_at)}</td>
                    <td className="px-3 py-2">
                      {r.live && (r.status === "queued" || r.status === "running") && (
                        <Button variant="ghost" size="xs" className="text-destructive hover:text-destructive" onClick={() => onCancel(r.id)}>Cancel</Button>
                      )}
                    </td>
                  </tr>
                ))}
                {filtered.length === 0 && (
                  <tr><td colSpan={9} className="px-4 py-10 text-center text-sm text-muted-foreground">
                    {filter === "all" ? "No requests yet — fire one from the Playground." : `No ${filter} requests.`}
                  </td></tr>
                )}
              </tbody>
            </table>
          </Card>
        </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <span className="text-muted-foreground">
      <span className="font-mono text-foreground">{value}</span> {label}
    </span>
  );
}

// `base` = the proxy's OpenAI base URL, e.g. https://gw/proxy/<name>/v1
function curlChat(base: string, token: string, model: string): string {
  return `curl -X POST '${base}/chat/completions' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer ${token}' \\
  -d '{
    "model": "${model}",
    "messages": [{"role": "user", "content": "Hello, world"}],
    "max_tokens": 1024
  }'`;
}
function curlChatStream(base: string, token: string, model: string): string {
  return `curl -N -X POST '${base}/chat/completions' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer ${token}' \\
  -d '{
    "model": "${model}",
    "messages": [{"role": "user", "content": "Hello, world"}],
    "max_tokens": 1024,
    "stream": true
  }'`;
}
function curlEmbeddings(base: string, token: string, model: string): string {
  return `curl -X POST '${base}/embeddings' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer ${token}' \\
  -d '{
    "model": "${model}",
    "input": ["Hello, world"]
  }'`;
}
function openaiSnippet(base: string, token: string, model: string): string {
  return `from openai import OpenAI

client = OpenAI(
    base_url="${base}",
    api_key="${token}",
)

resp = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Hello, world"}],
    stream=True,
)

for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)`;
}

function CodeBlock({ display, copy }: { display: string; copy?: string }) {
  return (
    <div className="relative">
      <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed text-foreground scrollbar-thin">{display}</pre>
      <Button variant="outline" size="icon-sm" className="absolute right-2 top-2" aria-label="Copy"
              onClick={() => { navigator.clipboard.writeText(copy ?? display); toast.success("Copied", { duration: 3000 }); }}>
        <Copy className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}
