"use client";

// Embedding playground: sends text(s) to the endpoint's OpenAI-compatible
// /v1/embeddings (through the Next proxy, which forwards to the gateway) and
// shows each returned vector's dimension + a short preview. Works for any
// embedding model on the fleet — pick whichever member is your embedding model.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Boxes, Loader2, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { AppRecord } from "@/lib/types";

type EmbedVec = { index: number; dim: number; preview: number[] };
type EmbedRun = {
  id: string;
  at: number;          // ms epoch
  model: string;
  n: number;           // number of inputs
  dim: number;         // vector dimension
  tokens?: number;     // usage total/prompt tokens
  inputs: string[];
  vecs: EmbedVec[];
};

const STORAGE_KEY = (appId: string) => `serverless-ui:embed:${appId}`;
const MAX_HISTORY = 50;

// Serverless wrapper — derives the member list + data-plane base path from the app.
export function EmbeddingTab({ app }: { app: AppRecord }) {
  // List every member; you pick whichever one is your embedding model (names can
  // be anything, so we never hide a model behind a heuristic).
  const models = useMemo(() => {
    if (app.mode === "multi" && app.models?.length) {
      return app.models.map((m) => m.model).filter(Boolean) as string[];
    }
    return app.model ? [app.model] : [];
  }, [app]);
  return (
    <EmbeddingPlayground
      models={models}
      basePath={`/api/proxy/${encodeURIComponent(app.app_id)}/v1`}
      storageKey={STORAGE_KEY(app.app_id)}
    />
  );
}

// Generic embedding playground — reused by serverless + the proxy. `basePath` is the
// Next-proxy prefix fronting the OpenAI data plane (…/v1); POSTs `${basePath}/embeddings`.
export function EmbeddingPlayground({ models, basePath, storageKey }: { models: string[]; basePath: string; storageKey: string }) {
  const [model, setModel] = useState(models[0] ?? "");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<EmbedRun | null>(null);

  // History — tracked per browser (localStorage), like the chat/audio playgrounds.
  const [history, setHistory] = useState<EmbedRun[]>([]);
  const historyRef = useRef(history);
  historyRef.current = history;
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (raw) setHistory(JSON.parse(raw));
    } catch { /* ignore corrupt/absent */ }
  }, [storageKey]);
  const persist = useCallback((next: EmbedRun[]) => {
    setHistory(next);
    try { window.localStorage.setItem(storageKey, JSON.stringify(next)); } catch { /* quota */ }
  }, [storageKey]);
  const removeRun = useCallback((id: string) => persist(historyRef.current.filter((r) => r.id !== id)), [persist]);
  const clearAll = useCallback(() => persist([]), [persist]);

  async function onRun() {
    // One input per non-blank line → an array (vLLM accepts a string or array).
    const inputs = text.split("\n").map((s) => s.trim()).filter(Boolean);
    if (!model || inputs.length === 0) return;
    setBusy(true); setErr(null); setResult(null);
    try {
      const r = await fetch(`${basePath}/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, input: inputs }),
      });
      const raw = await r.text();
      let body: unknown = raw;
      try { body = raw ? JSON.parse(raw) : null; } catch { /* keep raw */ }
      if (!r.ok) {
        const detail = (body as { detail?: unknown })?.detail ?? body;
        setErr(typeof detail === "string" ? detail : JSON.stringify(detail));
        return;
      }
      const data = (body as { data?: { embedding: number[]; index: number }[] }).data ?? [];
      const usage = (body as { usage?: { prompt_tokens?: number; total_tokens?: number } }).usage;
      const vecs: EmbedVec[] = data.map((d, i) => ({
        index: d.index ?? i,
        dim: Array.isArray(d.embedding) ? d.embedding.length : 0,
        preview: Array.isArray(d.embedding) ? d.embedding.slice(0, 8) : [],
      }));
      if (vecs.length === 0) {
        setErr("No embeddings returned — is this an embedding model? (a chat model returns an error here)");
        return;
      }
      const entry: EmbedRun = {
        id: globalThis.crypto?.randomUUID?.() ?? `e-${Date.now()}-${Math.round(Math.random() * 1e6)}`,
        at: Date.now(),
        model,
        n: vecs.length,
        dim: vecs[0]?.dim ?? 0,
        tokens: usage?.total_tokens ?? usage?.prompt_tokens,
        inputs,
        vecs,
      };
      setResult(entry);
      persist([entry, ...historyRef.current].slice(0, MAX_HISTORY));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const fmtVec = (v: number[]) =>
    `[${v.map((x) => (typeof x === "number" ? x.toFixed(4) : String(x))).join(", ")}${v.length ? ", …]" : "]"}`;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Embeddings</CardTitle>
          <p className="text-xs text-muted-foreground">
            Calls this endpoint&apos;s <code className="font-mono">/v1/embeddings</code> with an embedding model.
            One input per line. (A chat/generation model returns an error here — pick an embedding model.)
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">input (one per line)</span>
            <Textarea
              value={text}
              onChange={(e) => { setText(e.target.value); setResult(null); }}
              placeholder={"The quick brown fox\nhello world"}
              className="min-h-[96px] font-mono text-sm"
            />
          </div>
          <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">model</span>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger className="h-8 w-[260px] font-mono text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {models.map((m) => <SelectItem key={m} value={m} className="font-mono text-xs">{m}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <Button
              type="button"
              onClick={onRun}
              disabled={busy || !model || text.trim().length === 0}
              className="ml-auto"
            >
              {busy
                ? <><Loader2 className="h-4 w-4 animate-spin" /> Embedding…</>
                : <><Boxes className="h-4 w-4" /> Get embeddings</>}
            </Button>
          </div>
          {err && <p className="text-sm text-destructive">{err}</p>}
          {result && (
            <div className="space-y-1.5">
              <div className="text-xs text-muted-foreground">
                {result.n} embedding{result.n === 1 ? "" : "s"} · dim {result.dim}
                {result.tokens != null ? ` · ${result.tokens} tokens` : ""}
              </div>
              <div className="max-h-72 space-y-1.5 overflow-y-auto rounded-md border border-border bg-muted/30 p-3">
                {result.vecs.map((v, i) => (
                  <div key={v.index} className="text-xs">
                    <div className="truncate text-muted-foreground">[{v.index}] {result.inputs[i]}</div>
                    <code className="block font-mono text-[11px]">dim {v.dim} · {fmtVec(v.preview)}</code>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-sm font-medium">Embedding history</CardTitle>
            <p className="text-xs text-muted-foreground">Tracked per browser. {history.length} of {MAX_HISTORY} max.</p>
          </div>
          {history.length > 0 && (
            <CardAction>
              <Button variant="outline" size="xs" onClick={clearAll} className="text-muted-foreground hover:text-destructive">
                <Trash2 className="h-3 w-3" /> Clear all
              </Button>
            </CardAction>
          )}
        </CardHeader>
        <CardContent className="space-y-2">
          {history.length === 0 ? (
            <p className="py-4 text-center text-xs text-muted-foreground">No embeddings yet.</p>
          ) : (
            history.map((r) => (
              <details key={r.id} className="group rounded-md border border-border">
                <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-xs">
                  <span className="rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                    embed
                  </span>
                  <span className="font-mono text-[11px] text-muted-foreground">{r.model}</span>
                  <span className="truncate">{r.n} input{r.n === 1 ? "" : "s"} · dim {r.dim}</span>
                  <span className="ml-auto shrink-0 text-[11px] text-muted-foreground">{new Date(r.at).toLocaleString()}</span>
                  <button
                    type="button"
                    onClick={(e) => { e.preventDefault(); removeRun(r.id); }}
                    className="shrink-0 text-muted-foreground hover:text-destructive"
                    aria-label="Remove"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </summary>
                <div className="space-y-1.5 border-t border-border bg-muted/30 p-3">
                  {r.vecs.map((v, i) => (
                    <div key={v.index} className="text-xs">
                      <div className="truncate text-muted-foreground">[{v.index}] {r.inputs[i]}</div>
                      <code className="block font-mono text-[11px]">dim {v.dim} · {fmtVec(v.preview)}</code>
                    </div>
                  ))}
                </div>
              </details>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}
