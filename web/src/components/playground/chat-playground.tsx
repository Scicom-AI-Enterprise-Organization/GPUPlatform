"use client";

// Reusable chat playground. The UI (prompt, params, live answer/reasoning panels,
// tok/s stats, curl preview, per-browser history) lives here; each resource plugs
// in a `transport` describing HOW to send a request. `openAiTransport` is ready for
// any OpenAI-compatible `/v1/chat/completions` endpoint (the LLM proxy uses it);
// a resource with a different protocol (e.g. the serverless /run + /result queue)
// can supply its own transport with the same shape.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Download, Loader2, Play, ShieldAlert, Trash2, X } from "lucide-react";
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
import { CurlBlock } from "@/components/playground/curl-block";
import {
  AttachmentBar, attachmentsBytes, chatContent, fmtBytes, partSummary, readFiles,
  useImagePaste, type Attachment, type AudioPartKind,
} from "@/components/playground/attachments";
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
  // Images / audio / rasterized PDF pages. Present → the user message becomes a content
  // PART LIST instead of a plain string (see openAiBody).
  attachments?: Attachment[];
  audioAs?: AudioPartKind;   // which audio spelling to send; see attachment-parts.ts
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
  send: (params: ChatParams, h: SendHandlers) => Promise<{ content: string; reasoning: string; tokens?: number; toolCalls?: string; upstream?: Upstream; headers?: Record<string, string> }>;
  // Render an equivalent curl for the current params + bearer token.
  curl: (params: ChatParams, token: string) => string;
  // The REAL request body, base64 payloads included. Only transports with a JSON body
  // implement it; the playground uses it to offer a request.json download, because a
  // curl with a multi-megabyte data URL inlined is not a command anyone can paste.
  requestJson?: (params: ChatParams) => string;
};

type Stats = { ttftMs: number; tokens: number; tps: number } | null;
type Stored = {
  id: string; ts: number; prompt: string; model: string;
  status: "ok" | "error"; output?: string; reasoning?: string; toolCalls?: string; tokens?: number; error?: string;
  upstream?: Upstream;
  headers?: Record<string, string>;
  // What was attached, NOT the payload — see the note in onSend.
  attachments?: { kind: string; name: string }[];
};

/** Save the real request body so `curl -d @request.json` works. Revoked immediately —
 *  the blob only has to survive the click. */
function downloadJson(text: string, name = "request.json") {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

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
  // Media parts first, then the text (vLLM's own multi-image example order); no
  // attachments → a plain string, unchanged for every text-only backend. Lives in
  // attachment-parts.ts so it can be unit-tested without this file's UI imports.
  const content = chatContent(p.prompt, p.attachments, p.audioAs);
  const b: Record<string, unknown> = {
    model: p.model,
    messages: [{ role: "user", content }],
    max_tokens: p.maxTokens,
    temperature: p.temperature,
  };
  if (p.effort !== "none") b.reasoning_effort = p.effort;
  if (p.disableThinking) b.chat_template_kwargs = { enable_thinking: false };
  if (p.tools && p.tools.length > 0) { b.tools = p.tools; b.tool_choice = "auto"; }
  if (withStream) { b.stream = true; b.stream_options = { include_usage: true }; }
  return b;
}

export function openAiTransport(opts: { fetchPath: string; curlUrl: string; extraHeaders?: Record<string, string> }): ChatTransport {
  const extra = opts.extraHeaders ?? {};
  return {
    requestJson: (p) => JSON.stringify(openAiBody(p, p.stream), null, 2),
    curl: (p, token) => {
      const flag = p.stream ? "-N " : "";
      const extraLines = Object.entries(extra).map(([k, v]) => `  -H '${k}: ${v}' \\\n`).join("");
      const head = `curl ${flag}-X POST '${opts.curlUrl}' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer ${token}' \\
${extraLines}`;
      // ⚠ With attachments the body is inlined NOWHERE. An earlier version printed the
      // parts with the base64 replaced by "<272 KB … elided>", which copies cleanly and
      // then fails at the model with a garbage image — a placeholder that looks like a
      // command is worse than no command. Point at the downloadable file instead.
      if ((p.attachments?.length ?? 0) > 0) {
        return `# body: ${partSummary(p.attachments!, p.audioAs)} + text — download request.json`
          + ` below (base64 is too large to paste) and run this next to it:\n${head}  -d @request.json`;
      }
      return `${head}  -d '${JSON.stringify(openAiBody(p, p.stream), null, 2)}'`;
    },
    send: async (p, h) => {
      const res = await fetch(opts.fetchPath, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...extra },
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
      // Capture ALL response headers so the playground can surface them (X-Request-Id,
      // upstream routing, content-type, rate-limit, etc.). Available before the body.
      const headers: Record<string, string> = {};
      res.headers.forEach((v, k) => { headers[k] = v; });
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
        return { content, reasoning, toolCalls, tokens: data?.usage?.completion_tokens ?? undefined, upstream, headers };
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      let acc = "";
      let accR = "";
      let usage: number | undefined;
      let sniffedUp: Upstream | undefined;
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
            // Leading SSE comment the proxy emits to name the failover-chosen upstream
            // (can't be a header — flushed before failover resolves). ": sgpu-upstream {json}"
            if (line.startsWith(":")) {
              const m = line.match(/sgpu-upstream\s+(\{.*\})/);
              if (m) {
                try {
                  const up = JSON.parse(m[1]) as { name?: string; url?: string };
                  sniffedUp = { name: up.name, url: up.url };
                  if (up.name) headers["x-upstream-name"] = up.name;
                  if (up.url) headers["x-upstream-url"] = up.url;
                } catch { /* ignore malformed marker */ }
              }
              continue;
            }
            if (!line.startsWith("data:")) continue;
            const d = line.slice(5).trim();
            if (!d || d === "[DONE]") continue;
            let c: { choices?: { delta?: { content?: string; reasoning_content?: string; reasoning?: string; tool_calls?: unknown } }[]; usage?: { completion_tokens?: number }; error?: { message?: string } };
            try { c = JSON.parse(d); } catch { continue; }
            if (c.error) throw new Error(c.error.message || JSON.stringify(c.error));
            if (c.usage?.completion_tokens != null) usage = c.usage.completion_tokens;
            // Reasoning models stream their chain-of-thought as `delta.reasoning_content`
            // (DeepSeek-style) OR `delta.reasoning` (GLM/vLLM) — accept both, else the
            // bubble looks empty until `content` starts (which can be many tokens later).
            const dr = c.choices?.[0]?.delta?.reasoning_content ?? c.choices?.[0]?.delta?.reasoning;
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
      return { content: acc, reasoning: accR, toolCalls: formatToolCalls(toolAcc.filter(Boolean)), tokens: usage, upstream: upstream ?? sniffedUp, headers };
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
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [audioAs, setAudioAs] = useState<AudioPartKind>("input_audio");
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
  const [respHeaders, setRespHeaders] = useState<Record<string, string> | null>(null);
  const [stats, setStats] = useState<Stats>(null);
  const [err, setErr] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Screenshot → prompt box → attachment, which is how anyone actually tests a VLM.
  const addPasted = useCallback((files: File[]) => {
    void (async () => {
      const { added, errors } = await readFiles(files);
      if (added.length) setAttachments((prev) => [...prev, ...added]);
      if (errors.length) setErr(errors.join(" · "));
    })();
  }, []);
  const onPaste = useImagePaste(addPasted);

  const [history, setHistory] = useState<Stored[]>([]);

  useEffect(() => {
    try { const raw = window.localStorage.getItem(storageKey); if (raw) setHistory(JSON.parse(raw)); } catch { /* ignore */ }
  }, [storageKey]);

  const persist = useCallback((next: Stored[]) => {
    setHistory(next);
    try { window.localStorage.setItem(storageKey, JSON.stringify(next)); } catch { /* ignore */ }
  }, [storageKey]);
  const clearAll = useCallback(() => persist([]), [persist]);

  const params: ChatParams = useMemo(
    () => ({ model, prompt, maxTokens, temperature, effort, disableThinking, stream,
             tools: useTools && parsedTools && parsedTools.length ? parsedTools : undefined,
             attachments: attachments.length ? attachments : undefined,
             audioAs }),
    [model, prompt, maxTokens, temperature, effort, disableThinking, stream, useTools, parsedTools, attachments, audioAs],
  );

  const stop = () => abortRef.current?.abort();

  const onSend = async () => {
    // An attachment alone is a valid request — "what is in this image?" is often the
    // whole point, and a vision model does not need a text turn to answer it.
    if (!prompt.trim() && attachments.length === 0) { setErr("Prompt is required (or attach a file)."); return; }
    if (!model) { setErr("Pick a model."); return; }
    if (useTools && !parsedTools) { setErr("Tools JSON is invalid — fix it or turn off tools."); return; }
    setErr(null); setAnswer(""); setReasoning(""); setToolCalls(""); setStats(null); setUpstream(null); setRespHeaders(null);
    const id = `pg-${Date.now().toString(36)}`;
    const promptShort = prompt.slice(0, 80);
    // History is localStorage — record WHAT was attached, never the payload. A couple of
    // page images would blow the ~5 MB quota and take the whole history with them.
    const attNote: Stored["attachments"] = attachments.length
      ? attachments.map((a) => ({ kind: a.kind, name: a.name }))
      : undefined;
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
      if (r.headers) setRespHeaders(r.headers);
      if (r.tokens != null) {
        const ref = tFirst ?? t0;
        const secs = (now() - ref) / 1000;
        setStats({ ttftMs: tFirst != null ? Math.round(tFirst - t0) : 0, tokens: r.tokens, tps: secs > 0 ? r.tokens / secs : 0 });
      }
      const ok: Stored = { id, ts: Date.now(), prompt: promptShort, model, status: "ok", output: r.content, reasoning: r.reasoning, toolCalls: r.toolCalls, tokens: r.tokens, upstream: r.upstream, headers: r.headers, attachments: attNote };
      persist([ok, ...history].slice(0, MAX_HISTORY));
    } catch (e) {
      const aborted = e instanceof DOMException && e.name === "AbortError";
      const m = aborted ? "stopped" : e instanceof Error ? e.message : String(e);
      if (!aborted) setErr(m);
      const failed: Stored = { id, ts: Date.now(), prompt: promptShort, model, status: "error", error: m, output: answer || undefined, attachments: attNote };
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
          <Textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} onPaste={onPaste}
                    placeholder="Prompt — sent as a single user message" rows={2} className="font-mono text-sm" />
          <AttachmentBar attachments={attachments} onChange={setAttachments} disabled={busy}
                         audioAs={audioAs} onAudioAsChange={setAudioAs} />
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

          {err && <ErrorNote>{err}</ErrorNote>}

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
                <RedTeamNote headers={respHeaders} />
                <UpstreamLine upstream={upstream} />
                <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed text-foreground scrollbar-thin">{answer || (busy ? "…" : "")}</pre>
                <HeadersPanel headers={respHeaders} />
              </div>
            </div>
          )}

          {/* Live, not gated on a send — reading the request off as curl is most useful
              BEFORE you run it, and gating it meant a freshly-opened tab showed nothing.
              Built from the current params so it always mirrors the form above. */}
          <CurlBlock build={(tok) => transport.curl(params, tok)} />
          {attachments.length > 0 && transport.requestJson && (
            <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
              <Button variant="outline" size="xs" onClick={() => downloadJson(transport.requestJson!(params))}>
                <Download className="h-3 w-3" /> Download request.json
              </Button>
              <span>
                {attachments.length} attachment part{attachments.length === 1 ? "" : "s"},{" "}
                {fmtBytes(attachmentsBytes(attachments))} of base64 — the real body, ready for{" "}
                <span className="font-mono">-d @request.json</span>.
              </span>
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

// The one error style every playground mode uses. Chat had this box while the
// embedding/rerank/audio/TTS modes printed a bare `text-sm` line, so switching mode
// changed the size of the same message.
export function ErrorNote({ children }: { children: React.ReactNode }) {
  if (!children) return null;
  return (
    <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive break-words">
      {children}
    </div>
  );
}

// The red-team guard's verdict, called out instead of left buried in the header list.
// A blocked request is a normal 200 carrying a refusal (finish_reason: content_filter,
// or the same text over SSE), so without this the answer reads as the model's own —
// when in fact no upstream was called at all.
export function RedTeamNote({ headers }: { headers?: Record<string, string> | null }) {
  if (headers?.["x-sgpu-red-team"] !== "flagged") return null;
  const type = headers["x-sgpu-red-team-type"] || "unclassified";
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md border border-amber-500/40 bg-amber-500/10 px-2 py-1.5 text-[11px] text-amber-700 dark:text-amber-300">
      <ShieldAlert className="h-3.5 w-3.5 shrink-0" />
      <span className="font-medium">Blocked by the red-team screen</span>
      <span className="opacity-80">no upstream was called — the reply below is the gateway&apos;s</span>
      <code className="ml-auto shrink-0 rounded bg-amber-500/15 px-1.5 py-0.5 font-mono text-[10px]">{type}</code>
    </div>
  );
}

export function HeadersPanel({ headers }: { headers?: Record<string, string> | null }) {
  const [open, setOpen] = useState(false);
  if (!headers || Object.keys(headers).length === 0) return null;
  const entries = Object.entries(headers).sort(([a], [b]) => a.localeCompare(b));
  const reqId = headers["x-request-id"];
  return (
    <div className="rounded-md border border-border bg-muted/20">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 px-2 py-1 text-left text-[11px] text-muted-foreground hover:text-foreground"
      >
        {open ? <ChevronDown className="h-3 w-3 shrink-0" /> : <ChevronRight className="h-3 w-3 shrink-0" />}
        <span>Response headers ({entries.length})</span>
        {reqId && !open && <span className="ml-auto truncate font-mono text-[10px]">x-request-id: {reqId}</span>}
      </button>
      {open && (
        <div className="space-y-0.5 px-2 pb-2 pl-6 font-mono text-[10px] leading-relaxed">
          {entries.map(([k, v]) => (
            <div key={k} className="break-all">
              <span className="text-muted-foreground">{k}:</span> <span className="text-foreground">{v}</span>
            </div>
          ))}
        </div>
      )}
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
        {/* Attachment payloads are never persisted, so a past request shows WHAT it
            carried — enough to tell "the model saw nothing" from "the model saw this". */}
        {h.attachments && h.attachments.length > 0 && (
          <span className="shrink-0 font-mono text-[10px] text-muted-foreground"
                title={h.attachments.map((a) => `${a.kind}: ${a.name}`).join("\n")}>
            {h.attachments.length} file{h.attachments.length === 1 ? "" : "s"}
          </span>
        )}
        {h.tokens != null && <span className="shrink-0 font-mono text-[11px] text-muted-foreground">{h.tokens} tok</span>}
        <span className="shrink-0 text-[11px] text-muted-foreground">{new Date(h.ts).toLocaleTimeString()}</span>
      </button>
      {open && (
        <div className="space-y-2 px-4 pb-3 pl-9">
          <RedTeamNote headers={h.headers} />
          <UpstreamLine upstream={h.upstream} />
          <HeadersPanel headers={h.headers} />
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
