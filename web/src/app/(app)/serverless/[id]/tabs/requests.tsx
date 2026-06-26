"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  ChevronDown,
  ChevronRight,
  Copy,
  ExternalLink,
  Eye,
  EyeOff,
  Loader2,
  Play,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { NumberField } from "@/components/ui/number-field";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { HeadersPanel } from "@/components/playground/chat-playground";
import { TranscribeTab } from "./transcribe";
import { EmbeddingTab } from "./embedding";
import { gateway } from "@/lib/gateway";
import type { AppRecord } from "@/lib/types";
import { cn } from "@/lib/utils";
import { DEFAULT_TOOLS_JSON } from "@/lib/playground-tools";

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
  tokens?: number;     // completion tokens (from usage), when known
  tps?: number;        // ≈ completion tokens / wall time (incl. queue+RTT)
};

const STORAGE_KEY = (appId: string) => `serverless-ui:requests:${appId}`;
const POLL_MS = 4_000;
const MAX_HISTORY = 100;

// Coerce a pasted tools JSON into the OpenAI `tools` shape vLLM expects:
// `[{type:"function", function:{name, description, parameters}}]`. We accept a
// few friendlier shapes people paste — a bare array of function definitions
// (`[{name, description, parameters, ...}]`, the most common), `{tools:[...]}`,
// `{functions:[...]}`, or a single function object — and wrap each entry,
// keeping only the OpenAI-recognised fields (extras like `stage`/`returns` are
// dropped; `parameters` is forwarded verbatim, so any `$ref`s must already
// resolve within that schema). Returns null only when the JSON isn't usable.
function normalizeToolsInput(parsed: unknown): unknown[] | null {
  let arr: unknown[];
  if (Array.isArray(parsed)) {
    arr = parsed;
  } else if (parsed && typeof parsed === "object") {
    const o = parsed as Record<string, unknown>;
    if (Array.isArray(o.tools)) arr = o.tools;
    else if (Array.isArray(o.functions)) arr = o.functions;
    else arr = [parsed]; // a single tool/function object
  } else {
    return null;
  }
  return arr.map((t) => {
    if (!t || typeof t !== "object") return t;
    const o = t as Record<string, unknown>;
    // Already OpenAI-shaped → leave as-is.
    if (o.type === "function" && o.function && typeof o.function === "object") return o;
    // `{function:{...}}` missing the `type` wrapper.
    if (o.function && typeof o.function === "object") return { type: "function", function: o.function };
    // Bare function def: wrap it, keeping only name/description/parameters.
    if (typeof o.name === "string") {
      const fn: Record<string, unknown> = { name: o.name };
      if (typeof o.description === "string") fn.description = o.description;
      fn.parameters =
        o.parameters && typeof o.parameters === "object" ? o.parameters : { type: "object", properties: {} };
      return { type: "function", function: fn };
    }
    return o; // unknown shape — forward and let the server decide
  });
}

export function RequestsTab({ app, appId }: { app?: AppRecord; appId?: string } = {}) {
  // Prefer the passed-in app (gives us the model list for the dropdown); fall
  // back to app_id from props or the URL. Hook is called unconditionally.
  const fromPath = useAppIdFromPath();
  const resolvedAppId = app?.app_id ?? appId ?? fromPath;
  // Chat vs audio (Whisper) vs embedding mode. Always offered — model names can be
  // anything (a custom ASR/embedding model won't match a name heuristic), so each
  // mode lists every member; you pick whichever member fits the task. The mode just
  // selects which OpenAI endpoint the request hits (a chat model errors on
  // /v1/embeddings and vice-versa).
  const [mode, setMode] = useState<"chat" | "audio" | "embedding">("chat");
  if (!resolvedAppId) return null;
  const chat = <RequestsTabInner appId={resolvedAppId} app={app} />;
  if (!app) return chat; // no app record → no member list for the audio/embedding dropdown
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">mode</span>
        <Select value={mode} onValueChange={(v) => setMode(v as "chat" | "audio" | "embedding")}>
          <SelectTrigger className="h-8 w-[240px] text-xs"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="chat" className="text-xs">Chat / text generation</SelectItem>
            <SelectItem value="embedding" className="text-xs">Embeddings (/v1/embeddings)</SelectItem>
            <SelectItem value="audio" className="text-xs">Audio transcription (Whisper)</SelectItem>
          </SelectContent>
        </Select>
      </div>
      {mode === "chat" ? chat : mode === "embedding" ? <EmbeddingTab app={app} /> : <TranscribeTab app={app} />}
    </div>
  );
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

function RequestsTabInner({ appId, app }: { appId: string; app?: AppRecord }) {
  // Models this endpoint serves: multi-model members, else the single model.
  // The `model` field both routes the job (multi-model) and tells vLLM which
  // served model to use.
  const models = useMemo(() => {
    if (app?.mode === "multi" && app.models?.length) {
      return app.models.map((m) => m.model).filter(Boolean);
    }
    return app?.model ? [app.model] : [];
  }, [app]);

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
  // The request config is mirrored into the URL so a playground setup is
  // shareable / deep-linkable. Initial values are read from the query string.
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const syncParam = useCallback(
    (key: string, value: string | null) => {
      const params = new URLSearchParams(searchParams.toString());
      if (value == null || value === "") params.delete(key);
      else params.set(key, value);
      const qs = params.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [router, pathname, searchParams],
  );

  const [prompt, setPrompt] = useState("Hello, world");
  const [maxTokens, setMaxTokens] = useState(1024);
  const [temperature, setTemperature] = useState(() => {
    const t = searchParams.get("temp");
    const n = Number(t);
    return t != null && t !== "" && Number.isFinite(n) ? n : 0.7;
  });
  const [model, setModel] = useState(searchParams.get("model") ?? "");
  const [effort, setEffort] = useState<"none" | "low" | "medium" | "high">(() => {
    const e = searchParams.get("effort");
    return e === "low" || e === "medium" || e === "high" ? e : "none";
  });
  const [disableThinking, setDisableThinking] = useState(searchParams.get("disable_thinking") === "1");
  const [stream, setStream] = useState(searchParams.get("stream") === "1");
  const [sending, setSending] = useState(false);
  // Inline error for the non-stream run path (the user does not want a toast here).
  const [sendErr, setSendErr] = useState<string | null>(null);
  // Live streaming state (the SSE path doesn't use request history).
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [streamReasoning, setStreamReasoning] = useState("");
  const [streamErr, setStreamErr] = useState<string | null>(null);
  // Live throughput while streaming: TTFT + tokens/sec (output tokens since the
  // first token). Finalised from `usage` when the model reports it.
  const [streamStats, setStreamStats] = useState<{ ttftMs: number; tokens: number; tps: number } | null>(null);
  const [streamToolCalls, setStreamToolCalls] = useState("");
  const [respHeaders, setRespHeaders] = useState<Record<string, string> | null>(null);
  const streamAbort = useRef<AbortController | null>(null);

  // Equivalent OpenAI curl for the last-sent request (shared on Run).
  const [sentBody, setSentBody] = useState<Record<string, unknown> | null>(null);
  const [revealToken, setRevealToken] = useState(false);
  // Tool calling (function calling): when on, the request carries `tools` +
  // `tool_choice` and the model can reply with tool_calls instead of prose.
  const [useTools, setUseTools] = useState(false);
  const [toolsText, setToolsText] = useState(DEFAULT_TOOLS_JSON);
  const [showToolsEditor, setShowToolsEditor] = useState(false);
  const [token, setToken] = useState<string | null>(null);
  useEffect(() => {
    let abort = false;
    fetch("/api/auth/token", { cache: "no-store" })
      .then(async (r) => {
        if (abort) return;
        const b = r.ok ? ((await r.json()) as { token?: string }) : null;
        setToken(b?.token ?? null);
      })
      .catch(() => !abort && setToken(null));
    return () => {
      abort = true;
    };
  }, []);
  const base = process.env.NEXT_PUBLIC_GATEWAY_URL ?? gateway.baseUrl;

  // Derived (not stored) so a model arriving after first render still selects.
  const selectedModel = model || models[0] || "";

  // Parse + normalize the tools JSON once per edit; null = invalid (don't send
  // it). Accepts the OpenAI `tools` shape AND a bare array of function defs
  // (`[{name, description, parameters, ...}]`) / `{tools|functions:[...]}` — all
  // coerced to what vLLM expects.
  const parsedTools = useMemo<unknown[] | null>(() => {
    try {
      return normalizeToolsInput(JSON.parse(toolsText));
    } catch {
      return null;
    }
  }, [toolsText]);
  const toolsCount = parsedTools?.length ?? 0;

  // Chat-completion payload so chat-template params apply. `endpoint` is a
  // control field the gateway pops; everything else is forwarded to vLLM.
  function buildBody(): Record<string, unknown> {
    const body: Record<string, unknown> = {
      endpoint: "/v1/chat/completions",
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
      temperature,
    };
    if (selectedModel) body.model = selectedModel;
    if (effort !== "none") body.reasoning_effort = effort;
    if (disableThinking) body.chat_template_kwargs = { enable_thinking: false };
    if (useTools && parsedTools && parsedTools.length > 0) {
      body.tools = parsedTools;
      body.tool_choice = "auto";
    }
    return body;
  }

  // The public, OpenAI-compatible body (no internal `endpoint` control field;
  // `stream` exposed as the standard flag) — what the shareable curl runs.
  function publicBody(): Record<string, unknown> {
    const b = buildBody();
    delete b.endpoint;
    if (stream) {
      b.stream = true;
      b.stream_options = { include_usage: true }; // so the final chunk carries token usage
    }
    return b;
  }

  function onSend() {
    setSendErr(null);
    setStreamErr(null);
    setStreamToolCalls("");
    if (useTools && !parsedTools) {
      setSendErr("Tools JSON is invalid — fix it or turn off tools.");
      return;
    }
    if (!prompt.trim()) {
      setSendErr("Prompt is required.");
      return;
    }
    setSentBody(publicBody()); // share the equivalent curl for this request
    if (stream) void sendStream();
    else void send();
  }

  async function send() {
    setSending(true);
    setSendErr(null);
    setStreamErr(null);
    setStreamText("");
    setStreamReasoning("");
    setStreamToolCalls("");
    setStreamStats(null);
    setRespHeaders(null);
    const t0 = perfNow();
    const ts = Date.now();
    try {
      const r = await fetch(`/api/proxy/run/${encodeURIComponent(appId)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildBody()),
      });
      const text = await r.text();
      let respBody: unknown = text;
      try { respBody = text ? JSON.parse(text) : null; } catch { /* keep raw text */ }
      if (!r.ok) {
        setSendErr(errText(respBody, r.statusText));
        return;
      }
      const id = (respBody as { request_id?: string })?.request_id as string;
      // Surface response headers; the queue request_id lives in the /run BODY (the
      // middleware's x-request-id header is a different value), so prefer the body id.
      const hdrs: Record<string, string> = {};
      r.headers.forEach((v, k) => { hdrs[k] = v; });
      if (id) hdrs["x-request-id"] = id;
      setRespHeaders(hdrs);
      const promptShort = prompt.slice(0, 80);
      upsert({ id, ts, prompt: promptShort, status: "pending", app_id: appId });
      // Await so the Send button stays busy and the reasoning/answer panels
      // fill in when the result lands (same display as the streaming path).
      await measuredPoll(id, ts, promptShort, t0);
    } catch (e) {
      setSendErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
    }
  }

  // Poll a just-fired async request quickly (250 ms) until terminal, store
  // tokens + a tokens/sec estimate, and render its reasoning + answer in the
  // shared panels. TPS is wall-time based (includes RTT + any queue wait).
  async function measuredPoll(id: string, ts: number, promptShort: string, t0: number) {
    const deadline = perfNow() + 130_000;
    while (perfNow() < deadline) {
      await new Promise((res) => setTimeout(res, 250));
      let body: { status?: string; output?: unknown } | null = null;
      try {
        const res = await fetch(`/api/proxy/result/${encodeURIComponent(id)}`, { cache: "no-store" });
        if (res.status === 404) {
          upsert({ id, ts, prompt: promptShort, status: "expired", app_id: appId });
          setSendErr("Result expired before it could be read.");
          return;
        }
        body = await res.json();
      } catch {
        continue;
      }
      const status = normalizeStatus(body?.status ?? "unknown");
      if (!isTerminal(status)) continue;
      const elapsedS = (perfNow() - t0) / 1000;
      const tokens = completionTokensOf(body?.output) ?? undefined;
      const tps = tokens != null && elapsedS > 0 ? tokens / elapsedS : undefined;
      upsert({ id, ts, prompt: promptShort, status, output: body?.output, app_id: appId, tokens, tps });
      const out = body?.output;
      const oErr = (out as { error?: unknown } | null | undefined)?.error;
      if (status === "failed" || status === "timeout" || status === "cancelled" || oErr) {
        setSendErr(typeof oErr === "string" ? oErr : `Request ${status}.`);
        return;
      }
      const { content, reasoning, toolCalls } = parseResult(out);
      setStreamReasoning(reasoning);
      setStreamText(content);
      setStreamToolCalls(toolCalls);
      setStreamStats({ ttftMs: 0, tokens: tokens ?? 0, tps: tps ?? 0 });
      return;
    }
    setSendErr("Timed out waiting for the result.");
  }

  // SSE streaming via POST /stream/{appId}. The gateway relays each vLLM chunk
  // as `data: {...}`; we accumulate choices[0].delta.content live.
  async function sendStream() {
    streamAbort.current?.abort();
    const ctrl = new AbortController();
    streamAbort.current = ctrl;
    setStreaming(true);
    setStreamErr(null);
    setStreamText("");
    setStreamReasoning("");
    setStreamToolCalls("");
    setStreamStats(null);
    setRespHeaders(null);
    // include_usage → vLLM appends a final chunk with exact token usage.
    const body = { ...buildBody(), stream_options: { include_usage: true } };
    const toolAcc: ToolCallAcc[] = [];
    const t0 = perfNow();
    let tFirst: number | null = null;
    let count = 0; // tokens seen (≈ one per delta), refined by usage at the end
    let usageTokens: number | null = null;
    const bump = () => {
      if (tFirst === null) tFirst = perfNow();
      count += 1;
      const secs = (perfNow() - tFirst) / 1000;
      setStreamStats({
        ttftMs: Math.round(tFirst - t0),
        tokens: count,
        tps: secs > 0 ? count / secs : 0,
      });
    };
    try {
      const res = await fetch(`/api/proxy/stream/${encodeURIComponent(appId)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        const txt = await res.text().catch(() => "");
        let parsed: unknown = txt;
        try { parsed = txt ? JSON.parse(txt) : ""; } catch { /* keep raw text */ }
        throw new Error(errText(parsed, res.statusText));
      }
      const hdrs: Record<string, string> = {};
      res.headers.forEach((v, k) => { hdrs[k] = v; });
      setRespHeaders(hdrs);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let acc = "";
      let accR = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const frames = buf.split("\n\n");
        buf = frames.pop() ?? "";
        for (const frame of frames) {
          for (const lineRaw of frame.split("\n")) {
            const line = lineRaw.trimStart();
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data || data === "[DONE]") continue;
            let chunk: Record<string, unknown>;
            try {
              chunk = JSON.parse(data);
            } catch {
              continue;
            }
            if (chunk.error) {
              setStreamErr(errText(chunk.error, "stream error"));
              continue;
            }
            const ct = completionTokensOf(chunk);
            if (ct != null) usageTokens = ct;
            const reason = reasoningOf(chunk);
            if (reason) {
              accR += reason;
              setStreamReasoning(accR);
              bump();
            }
            const piece = deltaOf(chunk);
            if (piece) {
              acc += piece;
              setStreamText(acc);
              bump();
            }
            const tcDeltas = toolCallDeltasOf(chunk);
            if (tcDeltas.length) {
              for (const d of tcDeltas) {
                const i = typeof d.index === "number" ? d.index : toolAcc.length;
                if (!toolAcc[i]) toolAcc[i] = { name: "", args: "" };
                if (typeof d.function?.name === "string") toolAcc[i].name = d.function.name;
                if (typeof d.function?.arguments === "string") toolAcc[i].args += d.function.arguments;
              }
              setStreamToolCalls(formatToolCalls(toolAcc.filter(Boolean)));
              bump();
            }
          }
        }
      }
      // Finalise with exact usage when the model reported it.
      if (tFirst !== null) {
        const secs = (perfNow() - tFirst) / 1000;
        const toks = usageTokens ?? count;
        setStreamStats({
          ttftMs: Math.round(tFirst - t0),
          tokens: toks,
          tps: secs > 0 ? toks / secs : 0,
        });
      }
    } catch (e) {
      if (!ctrl.signal.aborted) {
        setStreamErr(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setStreaming(false);
    }
  }

  function stopStream() {
    streamAbort.current?.abort();
    setStreaming(false);
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
              Fires a chat completion via <code className="font-mono">POST /run/{appId}</code> (vLLM{" "}
              <code className="font-mono">/v1/chat/completions</code>). Streaming relays tokens live;
              otherwise the result is tracked below.
            </p>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Prompt — sent as a single user message"
            rows={2}
            className="font-mono text-sm"
          />
          <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
            {models.length > 0 && (
              <div className="flex flex-col gap-1">
                <span className="text-xs text-muted-foreground">model</span>
                <Select
                  value={selectedModel}
                  onValueChange={(v) => {
                    setModel(v);
                    syncParam("model", v);
                  }}
                >
                  <SelectTrigger className="h-8 w-[260px] font-mono text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map((m) => (
                      <SelectItem key={m} value={m} className="font-mono text-xs">
                        {m}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">reasoning_effort</span>
              <Select
                value={effort}
                onValueChange={(v) => {
                  setEffort(v as typeof effort);
                  syncParam("effort", v === "none" ? null : v);
                }}
              >
                <SelectTrigger className="h-8 w-[150px] text-xs">
                  <SelectValue />
                </SelectTrigger>
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
              <NumberField
                min={1}
                max={32768}
                value={maxTokens}
                onChange={setMaxTokens}
                className="h-8 w-24 font-mono"
              />
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">temperature</span>
              <NumberField
                allowDecimal
                min={0}
                max={2}
                value={temperature}
                onChange={(v) => {
                  setTemperature(v);
                  syncParam("temp", String(v));
                }}
                className="h-8 w-24 font-mono"
              />
            </div>
            <label className="flex h-8 items-center gap-2 text-xs text-muted-foreground">
              <input
                type="checkbox"
                checked={disableThinking}
                onChange={(e) => {
                  setDisableThinking(e.target.checked);
                  syncParam("disable_thinking", e.target.checked ? "1" : null);
                }}
                className="h-4 w-4 cursor-pointer accent-primary"
              />
              <span>
                disable thinking
                <span className="ml-1 font-mono text-[10px]">enable_thinking=false</span>
              </span>
            </label>
            <label className="flex h-8 items-center gap-2 text-xs text-muted-foreground">
              <input
                type="checkbox"
                checked={stream}
                onChange={(e) => {
                  setStream(e.target.checked);
                  syncParam("stream", e.target.checked ? "1" : null);
                }}
                className="h-4 w-4 cursor-pointer accent-primary"
              />
              <span>stream</span>
            </label>
            <label className="flex h-8 items-center gap-2 text-xs text-muted-foreground">
              <input
                type="checkbox"
                checked={useTools}
                onChange={(e) => {
                  setUseTools(e.target.checked);
                  if (e.target.checked) setShowToolsEditor(true);
                }}
                className="h-4 w-4 cursor-pointer accent-primary"
              />
              <span>
                tools
                <span className="ml-1 font-mono text-[10px]">tool_choice=auto</span>
              </span>
            </label>
            {useTools && (
              <Button variant="ghost" size="xs" onClick={() => setShowToolsEditor((v) => !v)}>
                {showToolsEditor ? "Hide" : "Edit"} tools ({toolsCount})
              </Button>
            )}
            <div className="flex-1" />
            {streaming ? (
              <Button variant="outline" onClick={stopStream}>
                <X className="h-4 w-4" />
                Stop
              </Button>
            ) : (
              <Button onClick={onSend} disabled={sending}>
                {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                Send
              </Button>
            )}
          </div>

          {useTools && showToolsEditor && (
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="text-muted-foreground">
                  tools (OpenAI function schema) — sent with <code className="font-mono">tool_choice: &quot;auto&quot;</code>
                </span>
                <div className="flex items-center gap-2">
                  {parsedTools ? (
                    <span className="text-muted-foreground">{toolsCount} function{toolsCount === 1 ? "" : "s"}</span>
                  ) : (
                    <span className="text-destructive">invalid JSON</span>
                  )}
                  <Button variant="ghost" size="xs" onClick={() => setToolsText(DEFAULT_TOOLS_JSON)}>
                    Reset
                  </Button>
                </div>
              </div>
              <Textarea
                value={toolsText}
                onChange={(e) => setToolsText(e.target.value)}
                rows={8}
                spellCheck={false}
                className={cn(
                  "max-h-72 font-mono text-[11px] leading-relaxed",
                  !parsedTools && "border-destructive focus-visible:ring-destructive/30",
                )}
              />
            </div>
          )}

          {(sendErr || streamErr) && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {sendErr ?? streamErr}
            </div>
          )}

          {(streaming || sending || streamText || streamReasoning || streamToolCalls) && (
            <div className="space-y-2">
              {streamReasoning && (
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    {(streaming || sending) && !streamText && <Loader2 className="h-3 w-3 animate-spin" />}
                    <span>Reasoning</span>
                  </div>
                  <pre className="max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-md border border-dashed border-border bg-muted/20 p-3 font-mono text-[11px] italic leading-relaxed text-muted-foreground scrollbar-thin">
                    {streamReasoning}
                  </pre>
                </div>
              )}
              {streamToolCalls && (
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span>Tool calls</span>
                  </div>
                  <pre className="max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-md border border-status-active/40 bg-status-active/5 p-3 font-mono text-[11px] leading-relaxed text-foreground scrollbar-thin">
                    {streamToolCalls}
                  </pre>
                </div>
              )}
              {(streamText || streaming || sending) && (
                <div className="space-y-1">
                  <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                    <div className="flex items-center gap-2">
                      {(streaming || sending) && <Loader2 className="h-3 w-3 animate-spin" />}
                      <span>Answer</span>
                    </div>
                    {streamStats && (
                      <span className="font-mono tabular-nums">
                        {streamStats.tps.toFixed(1)} tok/s · {streamStats.tokens} tok
                        {streamStats.ttftMs > 0 ? ` · TTFT ${streamStats.ttftMs} ms` : ""}
                      </span>
                    )}
                  </div>
                  <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed text-foreground scrollbar-thin">
                    {streamText || ((streaming || sending) ? "…" : "")}
                  </pre>
                  <HeadersPanel headers={respHeaders} />
                </div>
              )}
            </div>
          )}

          {sentBody && (
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs text-muted-foreground">cURL for this request</span>
                {token && (
                  <Button variant="ghost" size="xs" onClick={() => setRevealToken((v) => !v)}>
                    {revealToken ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
                    {revealToken ? "Hide" : "Reveal"} key
                  </Button>
                )}
              </div>
              <div className="relative">
                <pre className="max-h-80 overflow-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-[11px] leading-relaxed text-foreground scrollbar-thin">
                  {curlFor(base, appId, revealToken && token ? token : token ? maskToken(token) : "YOUR_API_KEY", sentBody)}
                </pre>
                <Button
                  variant="outline"
                  size="icon-sm"
                  className="absolute right-2 top-2"
                  aria-label="Copy cURL"
                  onClick={() => {
                    navigator.clipboard.writeText(
                      curlFor(base, appId, token ?? "YOUR_API_KEY", sentBody),
                    );
                    toast.success("cURL copied", { duration: 3000 });
                  }}
                >
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
            <p className="text-xs text-muted-foreground">
              Tracked per browser. {history.length} of {MAX_HISTORY} max.
            </p>
          </div>
          {history.length > 0 && (
            <CardAction>
              <Button
                variant="outline"
                size="xs"
                onClick={clearAll}
                className="text-muted-foreground hover:text-destructive"
              >
                <Trash2 className="h-3 w-3" />
                Clear all
              </Button>
            </CardAction>
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
          <div className="flex flex-col items-start gap-1">
            <StatusPill status={req.status} />
            {req.tps != null && (
              <span
                className="font-mono text-[10px] tabular-nums text-muted-foreground"
                title="completion tokens / wall time (includes queue + round-trip)"
              >
                ≈{req.tps.toFixed(1)} tok/s{req.tokens != null ? ` · ${req.tokens} tok` : ""}
              </span>
            )}
          </div>
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

// Pull the text delta out of a streamed chunk — OpenAI chat shape
// (choices[0].delta.content) or the fake worker's {delta: "..."}.
/** Pull a human-readable message out of a gateway/proxy error body. The gateway
 *  returns `{detail: "..."}`, `{detail: {error: "...", ...}}`, or `{error: "..."}`;
 *  naively rendering `detail` when it's an object yields "[object Object]". When
 *  the nested object carries extra context (e.g. a fleet's member `models`), fold
 *  a short hint in so the message stays useful. */
function errText(body: unknown, fallback: string): string {
  if (body == null) return fallback;
  if (typeof body === "string") return body || fallback;
  if (typeof body === "object") {
    const o = body as Record<string, unknown>;
    const pick = (v: unknown): string | null => {
      if (typeof v === "string") return v;
      if (v && typeof v === "object") {
        const inner = v as Record<string, unknown>;
        if (typeof inner.error === "string") {
          // Some errors carry a `models` list. Only fold it in when it's a list
          // of strings (the endpoint-name hint); the warming-up error uses
          // `models: [{model, state}]`, which would stringify to "[object Object]".
          const names = Array.isArray(inner.models)
            ? inner.models.filter((x): x is string => typeof x === "string")
            : [];
          return inner.error + (names.length ? ` (${names.join(", ")})` : "");
        }
      }
      return null;
    };
    return pick(o.detail) ?? pick(o.error) ?? (typeof o.message === "string" ? o.message : null) ??
      (() => { try { return JSON.stringify(body); } catch { return fallback; } })();
  }
  return fallback;
}

function deltaOf(chunk: Record<string, unknown>): string {
  const choices = chunk.choices as Array<{ delta?: { content?: unknown } }> | undefined;
  const c = choices?.[0]?.delta?.content;
  if (typeof c === "string") return c;
  if (typeof chunk.delta === "string") return chunk.delta;
  return "";
}

function perfNow(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

// completion_tokens from an OpenAI `usage` block (a streamed usage chunk or a
// full non-streaming response). Null when absent.
function completionTokensOf(obj: unknown): number | null {
  const u = (obj as { usage?: { completion_tokens?: unknown } } | null | undefined)?.usage;
  const c = u?.completion_tokens;
  return typeof c === "number" && Number.isFinite(c) ? c : null;
}

// Reasoning (thinking) delta. vLLM emits it under `delta.reasoning`; some
// builds use `reasoning_content` — accept either.
function reasoningOf(chunk: Record<string, unknown>): string {
  const choices = chunk.choices as
    | Array<{ delta?: { reasoning?: unknown; reasoning_content?: unknown } }>
    | undefined;
  const d = choices?.[0]?.delta;
  const r = d?.reasoning ?? d?.reasoning_content;
  return typeof r === "string" ? r : "";
}

type ToolCallAcc = { name: string; args: string };

// Render accumulated tool calls as `name({pretty-args})`, one per block.
function formatToolCalls(calls: ToolCallAcc[]): string {
  return calls
    .filter((c) => c.name || c.args)
    .map((c) => {
      let args = c.args;
      try {
        args = JSON.stringify(JSON.parse(c.args || "{}"), null, 2);
      } catch {
        /* keep raw partial args (e.g. mid-stream) */
      }
      return `${c.name || "?"}(${args})`;
    })
    .join("\n\n");
}

// Streamed tool-call fragments: choices[0].delta.tool_calls = [{index, function:{name?, arguments?}}].
function toolCallDeltasOf(chunk: Record<string, unknown>): Array<{
  index?: number;
  function?: { name?: unknown; arguments?: unknown };
}> {
  const choices = chunk.choices as Array<{ delta?: { tool_calls?: unknown } }> | undefined;
  const tc = choices?.[0]?.delta?.tool_calls;
  return Array.isArray(tc) ? tc : [];
}

// Pull content + reasoning + tool calls out of a non-streaming chat completion
// response (choices[0].message), so the result renders in the same panels.
function parseResult(output: unknown): { content: string; reasoning: string; toolCalls: string } {
  const o = output as
    | {
        choices?: Array<{
          message?: {
            content?: unknown;
            reasoning?: unknown;
            reasoning_content?: unknown;
            tool_calls?: Array<{ function?: { name?: unknown; arguments?: unknown } }>;
          };
        }>;
      }
    | null
    | undefined;
  const msg = o?.choices?.[0]?.message;
  const r = msg?.reasoning ?? msg?.reasoning_content;
  const calls: ToolCallAcc[] = Array.isArray(msg?.tool_calls)
    ? msg!.tool_calls.map((tc) => ({
        name: typeof tc?.function?.name === "string" ? tc.function.name : "?",
        args:
          typeof tc?.function?.arguments === "string"
            ? tc.function.arguments
            : JSON.stringify(tc?.function?.arguments ?? {}),
      }))
    : [];
  return {
    content: typeof msg?.content === "string" ? msg.content : "",
    reasoning: typeof r === "string" ? r : "",
    toolCalls: formatToolCalls(calls),
  };
}

function maskToken(t: string) {
  if (t.length <= 8) return "•".repeat(t.length);
  return `${t.slice(0, 4)}${"•".repeat(Math.max(8, t.length - 8))}${t.slice(-4)}`;
}

// Equivalent OpenAI cURL for a request body. `-N` (unbuffered) when streaming.
function curlFor(base: string, appId: string, token: string, body: Record<string, unknown>): string {
  const flag = body.stream ? "-N " : "";
  return `curl ${flag}-X POST '${base}/${appId}/v1/chat/completions' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer ${token}' \\
  -d '${JSON.stringify(body, null, 2)}'`;
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
