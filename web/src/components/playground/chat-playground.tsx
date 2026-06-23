"use client";

// Reusable chat playground. The UI (prompt, params, live answer/reasoning panels,
// tok/s stats, curl preview, per-browser history) lives here; each resource plugs
// in a `transport` describing HOW to send a request. `openAiTransport` is ready for
// any OpenAI-compatible `/v1/chat/completions` endpoint (the LLM proxy uses it);
// a resource with a different protocol (e.g. the serverless /run + /result queue)
// can supply its own transport with the same shape.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Copy, Loader2, Play, Trash2, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { NumberField } from "@/components/ui/number-field";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { DEFAULT_TOOLS_JSON } from "@/lib/playground-tools";
import { cn } from "@/lib/utils";

export type Effort = "none" | "low" | "medium" | "high";

export type ChatParams = {
  model: string;
  prompt: string;
  maxTokens: number;
  temperature: number;
  effort: Effort;
  disableThinking: boolean;
  stream: boolean;
  tools?: unknown[]; // OpenAI function schema; when set → sent with tool_choice:"auto"
};

export type SendHandlers = {
  signal: AbortSignal;
  onAnswer: (full: string) => void;
  onReasoning: (full: string) => void;
  onToolCalls: (full: string) => void;
  onToken: () => void; // call per streamed token to drive tok/s
};

export type Upstream = { url?: string; name?: string };
export type ChatTransport = {
  // Perform the request, pushing live updates through handlers; resolve with the
  // final content/reasoning/tool-calls/token-count or throw on error. `upstream`
  // (if the transport can determine it) names the backend that served the request.
  send: (params: ChatParams, h: SendHandlers) => Promise<{ content: string; reasoning: string; tokens?: number; toolCalls?: string; upstream?: Upstream }>;
  // Render an equivalent curl for the current params + bearer token.
  curl: (params: ChatParams, token: string) => string;
};

type Stats = { ttftMs: number; tokens: number; tps: number } | null;
type Stored = {
  id: string; ts: number; prompt: string; model: string;
  status: "ok" | "error"; output?: string; reasoning?: string; toolCalls?: string; tokens?: number; error?: string;
  upstream?: Upstream;
};

const MAX_HISTORY = 50;
const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

type ToolAcc = { name: string; args: string };
function formatToolCalls(calls: ToolAcc[]): string {
  return calls.filter((c) => c.name || c.args).map((c) => {
    let args = c.args;
    try { args = JSON.stringify(JSON.parse(c.args || "{}"), null, 2); } catch { /* partial mid-stream */ }
    return `${c.name || "?"}(${args})`;
  }).join("\n\n");
}
function toolCallDeltasOf(chunk: { choices?: Array<{ delta?: { tool_calls?: unknown } }> }): Array<{ index?: number; function?: { name?: unknown; arguments?: unknown } }> {
  const tc = chunk.choices?.[0]?.delta?.tool_calls;
  return Array.isArray(tc) ? tc : [];
}

// ---- OpenAI-compatible transport (proxy + any /v1/chat/completions endpoint) ----

export function openAiBody(p: ChatParams, withStream: boolean): Record<string, unknown> {
  const b: Record<string, unknown> = {
    model: p.model,
    messages: [{ role: "user", content: p.prompt }],
    max_tokens: p.maxTokens,
    temperature: p.temperature,
  };
  if (p.effort !== "none") b.reasoning_effort = p.effort;
  if (p.disableThinking) b.chat_template_kwargs = { enable_thinking: false };
  if (p.tools && p.tools.length > 0) { b.tools = p.tools; b.tool_choice = "auto"; }
  if (withStream) { b.stream = true; b.stream_options = { include_usage: true }; }
  return b;
}

export function openAiTransport(opts: { fetchPath: string; curlUrl: string }): ChatTransport {
  return {
    curl: (p, token) => {
      const body = openAiBody(p, p.stream);
      const flag = p.stream ? "-N " : "";
      return `curl ${flag}-X POST '${opts.curlUrl}' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer ${token}' \\
  -d '${JSON.stringify(body, null, 2)}'`;
    },
    send: async (p, h) => {
      const res = await fetch(opts.fetchPath, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(openAiBody(p, p.stream)),
        signal: h.signal,
      });
      if (!res.ok || !res.body) {
        const t = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status}: ${t.slice(0, 300) || res.statusText}`);
      }
      // Routing info from the proxy router (absent for streamed multi-upstream
      // requests, where the served backend can't be known before headers flush).
      const upUrl = res.headers.get("x-upstream-url") ?? undefined;
      const upName = res.headers.get("x-upstream-name") ?? undefined;
      const upstream: Upstream | undefined = upUrl || upName ? { url: upUrl, name: upName } : undefined;
      if (!p.stream) {
        const data = await res.json();
        const msg = data?.choices?.[0]?.message ?? {};
        const content = msg.content ?? data?.choices?.[0]?.text ?? "";
        const reasoning = msg.reasoning_content ?? msg.reasoning ?? "";
        const calls: ToolAcc[] = Array.isArray(msg.tool_calls)
          ? msg.tool_calls.map((tc: { function?: { name?: unknown; arguments?: unknown } }) => ({
              name: typeof tc?.function?.name === "string" ? tc.function.name : "?",
              args: typeof tc?.function?.arguments === "string" ? tc.function.arguments : JSON.stringify(tc?.function?.arguments ?? {}),
            }))
          : [];
        const toolCalls = formatToolCalls(calls);
        h.onReasoning(reasoning);
        h.onToolCalls(toolCalls);
        h.onAnswer(content);
        return { content, reasoning, toolCalls, tokens: data?.usage?.completion_tokens ?? undefined, upstream };
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      let acc = "";
      let accR = "";
      let usage: number | undefined;
      const toolAcc: ToolAcc[] = [];
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const frames = buf.split("\n\n");
        buf = frames.pop() ?? "";
        for (const frame of frames) {
          for (const lineRaw of frame.split("\n")) {
            const line = lineRaw.trimStart();
            if (!line.startsWith("data:")) continue;
            const d = line.slice(5).trim();
            if (!d || d === "[DONE]") continue;
            let c: { choices?: { delta?: { content?: string; reasoning_content?: string; tool_calls?: unknown } }[]; usage?: { completion_tokens?: number }; error?: { message?: string } };
            try { c = JSON.parse(d); } catch { continue; }
            if (c.error) throw new Error(c.error.message || JSON.stringify(c.error));
            if (c.usage?.completion_tokens != null) usage = c.usage.completion_tokens;
            const dr = c.choices?.[0]?.delta?.reasoning_content;
            if (dr) { accR += dr; h.onReasoning(accR); h.onToken(); }
            const dcp = c.choices?.[0]?.delta?.content;
            if (dcp) { acc += dcp; h.onAnswer(acc); h.onToken(); }
            const tds = toolCallDeltasOf(c);
            if (tds.length) {
              for (const t of tds) {
                const i = typeof t.index === "number" ? t.index : toolAcc.length;
                if (!toolAcc[i]) toolAcc[i] = { name: "", args: "" };
                if (typeof t.function?.name === "string") toolAcc[i].name = t.function.name;
                if (typeof t.function?.arguments === "string") toolAcc[i].args += t.function.arguments;
              }
              h.onToolCalls(formatToolCalls(toolAcc.filter(Boolean)));
              h.onToken();
            }
          }
        }
      }
      return { content: acc, reasoning: accR, toolCalls: formatToolCalls(toolAcc.filter(Boolean)), tokens: usage, upstream };
    },
  };
}

// ---- the playground UI ----

export function ChatPlayground({
  models,
  storageKey,
  transport,
  description,
}: {
  models: string[];
  storageKey: string;
  transport: ChatTransport;
  description?: React.ReactNode;
}) {
  const [model, setModel] = useState(models[0] ?? "");
  const [prompt, setPrompt] = useState("Hello, world");
  const [maxTokens, setMaxTokens] = useState(512);
  const [temperature, setTemperature] = useState(0.7);
  const [effort, setEffort] = useState<Effort>("none");
  const [disableThinking, setDisableThinking] = useState(false);
  const [stream, setStream] = useState(true);
  const [useTools, setUseTools] = useState(false);
  const [showToolsEditor, setShowToolsEditor] = useState(false);
  const [toolsText, setToolsText] = useState(DEFAULT_TOOLS_JSON);
  const parsedTools = useMemo<unknown[] | null>(() => {
    try { const p = JSON.parse(toolsText); return Array.isArray(p) ? p : null; } catch { return null; }
  }, [toolsText]);
  const toolsCount = parsedTools?.length ?? 0;

  const [sending, setSending] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [answer, setAnswer] = useState("");
  const [reasoning, setReasoning] = useState("");
  const [toolCalls, setToolCalls] = useState("");
  const [upstream, setUpstream] = useState<Upstream | null>(null);
  const [stats, setStats] = useState<Stats>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sentParams, setSentParams] = useState<ChatParams | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const [history, setHistory] = useState<Stored[]>([]);
  const [apiKeyPrefix, setApiKeyPrefix] = useState<string | null>(null);

  useEffect(() => {
    try { const raw = window.localStorage.getItem(storageKey); if (raw) setHistory(JSON.parse(raw)); } catch { /* ignore */ }
  }, [storageKey]);
  useEffect(() => {
    let abort = false;
    fetch("/api/api-keys", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : []))
      .then((keys: { prefix: string }[]) => {
        if (!abort) setApiKeyPrefix(Array.isArray(keys) && keys.length > 0 ? keys[0].prefix : null);
      })
      .catch(() => {});
    return () => { abort = true; };
  }, []);

  const persist = useCallback((next: Stored[]) => {
    setHistory(next);
    try { window.localStorage.setItem(storageKey, JSON.stringify(next)); } catch { /* ignore */ }
  }, [storageKey]);
  const clearAll = useCallback(() => persist([]), [persist]);

  const params: ChatParams = useMemo(
    () => ({ model, prompt, maxTokens, temperature, effort, disableThinking, stream, tools: useTools && parsedTools && parsedTools.length ? parsedTools : undefined }),
    [model, prompt, maxTokens, temperature, effort, disableThinking, stream, useTools, parsedTools],
  );

  const stop = () => abortRef.current?.abort();

  const onSend = async () => {
    if (!prompt.trim()) { setErr("Prompt is required."); return; }
    if (!model) { setErr("Pick a model."); return; }
    if (useTools && !parsedTools) { setErr("Tools JSON is invalid — fix it or turn off tools."); return; }
    setErr(null); setAnswer(""); setReasoning(""); setToolCalls(""); setStats(null); setUpstream(null);
    setSentParams(params);
    const id = `pg-${Date.now().toString(36)}`;
    const promptShort = prompt.slice(0, 80);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const t0 = now();
    let tFirst: number | null = null;
    let toks = 0;
    const bump = () => {
      if (tFirst === null) tFirst = now();
      toks += 1;
      const secs = (now() - tFirst) / 1000;
      setStats({ ttftMs: Math.round(tFirst - t0), tokens: toks, tps: secs > 0 ? toks / secs : 0 });
    };
    if (stream) setStreaming(true); else setSending(true);
    try {
      const r = await transport.send(params, { signal: ctrl.signal, onAnswer: setAnswer, onReasoning: setReasoning, onToolCalls: setToolCalls, onToken: bump });
      if (r.upstream) setUpstream(r.upstream);
      if (r.tokens != null) {
        const ref = tFirst ?? t0;
        const secs = (now() - ref) / 1000;
        setStats({ ttftMs: tFirst != null ? Math.round(tFirst - t0) : 0, tokens: r.tokens, tps: secs > 0 ? r.tokens / secs : 0 });
      }
      const ok: Stored = { id, ts: Date.now(), prompt: promptShort, model, status: "ok", output: r.content, reasoning: r.reasoning, toolCalls: r.toolCalls, tokens: r.tokens, upstream: r.upstream };
      persist([ok, ...history].slice(0, MAX_HISTORY));
    } catch (e) {
      const aborted = e instanceof DOMException && e.name === "AbortError";
      const m = aborted ? "stopped" : e instanceof Error ? e.message : String(e);
      if (!aborted) setErr(m);
      const failed: Stored = { id, ts: Date.now(), prompt: promptShort, model, status: "error", error: m, output: answer || undefined };
      persist([failed, ...history].slice(0, MAX_HISTORY));
    } finally {
      setSending(false); setStreaming(false); abortRef.current = null;
    }
  };

  const busy = sending || streaming;

  if (models.length === 0) {
    return (
      <Card><CardContent className="py-8 text-center text-sm text-muted-foreground">No models available.</CardContent></Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Send a test request</CardTitle>
          {description && <p className="text-xs text-muted-foreground">{description}</p>}
        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Prompt — sent as a single user message" rows={2} className="font-mono text-sm" />
          <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">model</span>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger className="h-8 w-[220px] font-mono text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>{models.map((a) => <SelectItem key={a} value={a} className="font-mono text-xs">{a}</SelectItem>)}</SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">reasoning_effort</span>
              <Select value={effort} onValueChange={(v) => setEffort(v as Effort)}>
                <SelectTrigger className="h-8 w-[150px] text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">none (omit)</SelectItem>
                  <SelectItem value="low">low</SelectItem>
                  <SelectItem value="medium">medium</SelectItem>
                  <SelectItem value="high">high</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">max_tokens</span>
              <NumberField min={1} max={32768} value={maxTokens} onChange={setMaxTokens} className="h-8 w-24 font-mono" />
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">temperature</span>
              <NumberField allowDecimal min={0} max={2} value={temperature} onChange={setTemperature} className="h-8 w-24 font-mono" />
            </div>
            <label className="flex h-8 items-center gap-2 text-xs text-muted-foreground">
              <input type="checkbox" checked={disableThinking} onChange={(e) => setDisableThinking(e.target.checked)} className="h-4 w-4 cursor-pointer accent-primary" />
              <span>disable thinking <span className="ml-1 font-mono text-[10px]">enable_thinking=false</span></span>
            </label>
            <label className="flex h-8 items-center gap-2 text-xs text-muted-foreground">
              <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} className="h-4 w-4 cursor-pointer accent-primary" />
              <span>stream</span>
            </label>
            <label className="flex h-8 items-center gap-2 text-xs text-muted-foreground">
              <input type="checkbox" checked={useTools} onChange={(e) => { setUseTools(e.target.checked); if (e.target.checked) setShowToolsEditor(true); }} className="h-4 w-4 cursor-pointer accent-primary" />
              <span>tools <span className="ml-1 font-mono text-[10px]">tool_choice=auto</span></span>
            </label>
            {useTools && (
              <Button variant="ghost" size="xs" onClick={() => setShowToolsEditor((v) => !v)}>
                {showToolsEditor ? "Hide" : "Edit"} tools ({toolsCount})
              </Button>
            )}
            <div className="flex-1" />
            {streaming ? (
              <Button variant="outline" onClick={stop}><X className="h-4 w-4" /> Stop</Button>
            ) : (
              <Button onClick={onSend} disabled={sending}>
                {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />} Send
              </Button>
            )}
          </div>

          {useTools && showToolsEditor && (
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="text-muted-foreground">tools (OpenAI function schema) — sent with <code className="font-mono">tool_choice: &quot;auto&quot;</code></span>
                <div className="flex items-center gap-2">
                  {parsedTools ? <span className="text-muted-foreground">{toolsCount} function{toolsCount === 1 ? "" : "s"}</span> : <span className="text-destructive">invalid JSON</span>}
                  <Button variant="ghost" size="xs" onClick={() => setToolsText(DEFAULT_TOOLS_JSON)}>Reset</Button>
                </div>
              </div>
              <Textarea value={toolsText} onChange={(e) => setToolsText(e.target.value)} rows={8} spellCheck={false}
                        className={cn("max-h-72 font-mono text-[11px] leading-relaxed", !parsedTools && "border-destructive focus-visible:ring-destructive/30")} />
            </div>
          )}

          {err && <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive break-words">{err}</div>}

          {(busy || answer || reasoning || toolCalls) && (
            <div className="space-y-2">
              {reasoning && (
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">{busy && !answer && <Loader2 className="h-3 w-3 animate-spin" />}<span>Reasoning</span></div>
                  <pre className="max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-md border border-dashed border-border bg-muted/20 p-3 font-mono text-[11px] italic leading-relaxed text-muted-foreground scrollbar-thin">{reasoning}</pre>
                </div>
              )}
              {toolCalls && (
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground"><span>Tool calls</span></div>
                  <pre className="max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-md border border-status-active/40 bg-status-active/5 p-3 font-mono text-[11px] leading-relaxed text-foreground scrollbar-thin">{toolCalls}</pre>
                </div>
              )}
              <div className="space-y-1">
                <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">{busy && <Loader2 className="h-3 w-3 animate-spin" />}<span>Answer</span></div>
                  {stats && <span className="font-mono tabular-nums">{stats.tps.toFixed(1)} tok/s · {stats.tokens} tok{stats.ttftMs > 0 ? ` · TTFT ${stats.ttftMs} ms` : ""}</span>}
                </div>
                <UpstreamLine upstream={upstream} />
                <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed text-foreground scrollbar-thin">{answer || (busy ? "…" : "")}</pre>
              </div>
            </div>
          )}

          {sentParams && (
            <div className="space-y-1">
              <span className="text-xs text-muted-foreground">cURL for this request</span>
              <div className="relative">
                <pre className="max-h-80 overflow-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-[11px] leading-relaxed text-foreground scrollbar-thin">
                  {transport.curl(sentParams, apiKeyPrefix ? `${apiKeyPrefix}...` : "YOUR_SGPU_API_KEY")}
                </pre>
                <Button variant="outline" size="icon-sm" className="absolute right-2 top-2" aria-label="Copy cURL"
                        onClick={() => { navigator.clipboard.writeText(transport.curl(sentParams, apiKeyPrefix ? `${apiKeyPrefix}...` : "YOUR_SGPU_API_KEY")); toast.success("cURL copied", { duration: 3000 }); }}>
                  <Copy className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-sm font-medium">Request history</CardTitle>
            <p className="text-xs text-muted-foreground">Tracked per browser. {history.length} of {MAX_HISTORY} max.</p>
          </div>
          {history.length > 0 && (
            <CardAction>
              <Button variant="outline" size="xs" className="text-muted-foreground hover:text-destructive" onClick={clearAll}>
                <Trash2 className="h-3 w-3" /> Clear all
              </Button>
            </CardAction>
          )}
        </CardHeader>
        <CardContent className="px-0 py-0">
          {history.length === 0 ? (
            <p className="px-4 py-8 text-center text-sm text-muted-foreground">No requests yet — send one above.</p>
          ) : (
            history.map((h) => <HistoryRow key={h.id} h={h} />)
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function UpstreamLine({ upstream }: { upstream?: Upstream | null }) {
  if (!upstream || (!upstream.url && !upstream.name)) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
      <span>served by</span>
      {upstream.name && <span className="rounded bg-primary/10 px-1.5 py-0.5 font-mono text-primary">{upstream.name}</span>}
      {upstream.url && <span className="break-all font-mono">{upstream.url}</span>}
    </div>
  );
}

function HistoryRow({ h }: { h: Stored }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-b border-border/60 last:border-0">
      <button onClick={() => setOpen((v) => !v)} className="flex w-full items-center gap-2 px-4 py-2 text-left text-sm hover:bg-muted/30">
        {open ? <ChevronDown className="h-3.5 w-3.5 shrink-0 text-muted-foreground" /> : <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />}
        <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", h.status === "ok" ? "bg-emerald-500" : "bg-destructive")} />
        <span className="font-mono text-xs text-muted-foreground">{h.model}</span>
        <span className="min-w-0 flex-1 truncate text-muted-foreground">{h.prompt || "(empty)"}</span>
        {h.tokens != null && <span className="shrink-0 font-mono text-[11px] text-muted-foreground">{h.tokens} tok</span>}
        <span className="shrink-0 text-[11px] text-muted-foreground">{new Date(h.ts).toLocaleTimeString()}</span>
      </button>
      {open && (
        <div className="space-y-2 px-4 pb-3 pl-9">
          <UpstreamLine upstream={h.upstream} />
          {h.reasoning && <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words rounded-md border border-dashed border-border bg-muted/20 p-2 font-mono text-[11px] italic text-muted-foreground scrollbar-thin">{h.reasoning}</pre>}
          {h.toolCalls && <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words rounded-md border border-status-active/40 bg-status-active/5 p-2 font-mono text-[11px] text-foreground scrollbar-thin">{h.toolCalls}</pre>}
          {h.error
            ? <pre className="whitespace-pre-wrap break-words rounded-md border border-destructive/40 bg-destructive/10 p-2 font-mono text-[11px] text-destructive">{h.error}</pre>
            : <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-muted/40 p-2 font-mono text-[11px] text-foreground scrollbar-thin">{h.output || "(empty)"}</pre>}
        </div>
      )}
    </div>
  );
}
