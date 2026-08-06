"use client";

// Rerank playground: sends a query + candidate documents to a cross-encoder's
// /v1/rerank and shows the ranked results with their relevance scores.
//
// Unlike the embedding playground this one is deliberately opinionated about how
// the results are READ. Reranker scores are calibrated hard toward 0 or 1 — a
// genuinely relevant document lands ~0.9+, an irrelevant one ~0.0. Scores bunched
// in the middle with no separation are the signature of the scoring/chat template
// not being applied upstream (vLLM never auto-applies a reranker's chat template;
// without --chat-template the model scores a bare query+document concatenation and
// still returns HTTP 200). So we surface the top-to-next gap and warn on a flat
// spread, rather than just printing numbers.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowDownWideNarrow, Loader2, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { CurlBlock, curlJson } from "@/components/playground/curl-block";
import { ErrorNote } from "@/components/playground/chat-playground";

type RankedDoc = { index: number; score: number; text: string };
type RerankRun = {
  id: string;
  at: number;          // ms epoch
  model: string;
  query: string;
  n: number;           // documents sent
  instruction?: string;
  topN?: number;
  tokens?: number;
  results: RankedDoc[];
};

const MAX_HISTORY = 50;
// Below this top-to-next score gap the ranking isn't cleanly separated — usually a
// template problem upstream, not a borderline match. Same threshold the gateway's
// upstream "Test" probe uses.
const WEAK_GAP = 0.2;

const fmtScore = (s: number) => (s >= 0.0005 || s === 0 ? s.toFixed(4) : s.toExponential(2));

// Prefilled, not just placeholders — an empty form means Rerank is dead and there's no
// curl to copy (the chat playground starts on "Hello, world" for the same reason). The
// third document is a deliberate hard negative: a set without one scores 200-OK even
// when the upstream's chat template isn't applied, which is the failure this mode exists
// to expose.
const EXAMPLE_QUERY = "How do I check my bill?";
const EXAMPLE_DOCS = [
  "Log in to the billing portal and open the Billing tab.",
  "Restart your router by holding the reset pin.",
  "The Great Wall of China is over 20,000 km long.",
].join("\n");

// Generic rerank playground — `basePath` is the Next-proxy prefix fronting the
// OpenAI-compatible data plane (…/v1); POSTs `${basePath}/rerank`. `curlBase` is the
// same path spelled the way a caller outside the browser would (the public gateway
// URL) — only used to render the copyable curl.
export function RerankPlayground({ models, basePath, curlBase, storageKey, extraHeaders }: {
  models: string[];
  basePath: string;
  curlBase?: string;
  storageKey: string;
  extraHeaders?: Record<string, string>;
}) {
  const [model, setModel] = useState(models[0] ?? "");
  const [query, setQuery] = useState(EXAMPLE_QUERY);
  const [docsText, setDocsText] = useState(EXAMPLE_DOCS);
  const [instruction, setInstruction] = useState("");
  const [topN, setTopN] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<RerankRun | null>(null);

  // History — tracked per browser (localStorage), like the chat/embedding playgrounds.
  const [history, setHistory] = useState<RerankRun[]>([]);
  const historyRef = useRef(history);
  historyRef.current = history;
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (raw) setHistory(JSON.parse(raw));
    } catch { /* ignore corrupt/absent */ }
  }, [storageKey]);
  const persist = useCallback((next: RerankRun[]) => {
    setHistory(next);
    try { window.localStorage.setItem(storageKey, JSON.stringify(next)); } catch { /* quota */ }
  }, [storageKey]);
  const removeRun = useCallback((id: string) => persist(historyRef.current.filter((r) => r.id !== id)), [persist]);
  const clearAll = useCallback(() => persist([]), [persist]);

  const docs = useMemo(
    () => docsText.split("\n").map((s) => s.trim()).filter(Boolean),
    [docsText],
  );

  // The exact payload — shared with the curl block so what you copy is what ran.
  const topNum = Number.parseInt(topN, 10);
  const body = useMemo<Record<string, unknown>>(() => {
    const b: Record<string, unknown> = { model, query: query.trim(), documents: docs };
    if (Number.isFinite(topNum) && topNum > 0) b.top_n = topNum;
    if (instruction.trim()) b.instruction = instruction.trim();
    return b;
  }, [model, query, docs, topNum, instruction]);

  async function onRun() {
    if (!model || !query.trim() || docs.length === 0) return;
    setBusy(true); setErr(null); setResult(null);
    try {
      const n = topNum;
      const r = await fetch(`${basePath}/rerank`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(extraHeaders ?? {}) },
        body: JSON.stringify(body),
      });
      const raw = await r.text();
      let parsed: unknown = raw;
      try { parsed = raw ? JSON.parse(raw) : null; } catch { /* keep raw */ }
      if (!r.ok) {
        const detail = (parsed as { detail?: unknown; error?: unknown })?.detail
          ?? (parsed as { error?: unknown })?.error ?? parsed;
        setErr(typeof detail === "string" ? detail : JSON.stringify(detail));
        return;
      }
      // vLLM / Cohere shape: { results: [{ index, relevance_score, document: { text } }] }.
      // `score` is accepted as a fallback key; `document` may be absent (index-only servers).
      const rows = (parsed as {
        results?: { index?: number; relevance_score?: number; score?: number; document?: { text?: string } | string }[];
      }).results ?? [];
      const usage = (parsed as { usage?: { prompt_tokens?: number; total_tokens?: number } }).usage;
      const results: RankedDoc[] = rows.map((d, i) => {
        const idx = typeof d.index === "number" ? d.index : i;
        const doc = d.document;
        const text = typeof doc === "string" ? doc : doc?.text ?? docs[idx] ?? "";
        return { index: idx, score: Number(d.relevance_score ?? d.score ?? 0), text };
      }).sort((a, b) => b.score - a.score);
      if (results.length === 0) {
        setErr("No results returned — is this a reranker/cross-encoder? (an embedding or chat model errors here)");
        return;
      }
      const entry: RerankRun = {
        id: globalThis.crypto?.randomUUID?.() ?? `r-${Date.now()}-${Math.round(Math.random() * 1e6)}`,
        at: Date.now(),
        model,
        query: query.trim(),
        n: docs.length,
        instruction: instruction.trim() || undefined,
        topN: Number.isFinite(n) && n > 0 ? n : undefined,
        tokens: usage?.total_tokens ?? usage?.prompt_tokens,
        results,
      };
      setResult(entry);
      persist([entry, ...historyRef.current].slice(0, MAX_HISTORY));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Rerank</CardTitle>
          <p className="text-xs text-muted-foreground">
            Calls <code className="font-mono">/v1/rerank</code> with a cross-encoder. One candidate document per line;
            results come back sorted by relevance. Include a hard negative — a single-document test passes even when
            the model is misconfigured.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">query</span>
            <Input
              value={query}
              onChange={(e) => { setQuery(e.target.value); setResult(null); }}
              placeholder="How do I check my bill?"
              className="font-mono text-sm"
            />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">documents (one per line)</span>
            <Textarea
              value={docsText}
              onChange={(e) => { setDocsText(e.target.value); setResult(null); }}
              placeholder={"Log in to the billing portal and open the Billing tab.\nRestart your router by holding the reset pin.\nThe Great Wall of China is over 20,000 km long."}
              className="min-h-[110px] font-mono text-sm"
            />
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">instruction (optional)</span>
            <Input
              value={instruction}
              onChange={(e) => { setInstruction(e.target.value); setResult(null); }}
              placeholder="Given a customer support ticket, decide if it expresses churn intent"
              className="font-mono text-sm"
            />
            <span className="text-[11px] text-muted-foreground">
              Overrides the default web-search retrieval instruction. This genuinely changes the ranking — use it to
              score for a domain task, not as a label.
            </span>
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
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">top_n (optional)</span>
              <Input
                value={topN}
                onChange={(e) => setTopN(e.target.value.replace(/[^0-9]/g, ""))}
                placeholder="all"
                inputMode="numeric"
                className="h-8 w-[110px] font-mono text-xs"
              />
            </div>
            <Button
              type="button"
              onClick={onRun}
              disabled={busy || !model || !query.trim() || docs.length === 0}
              className="ml-auto"
            >
              {busy
                ? <><Loader2 className="h-4 w-4 animate-spin" /> Reranking…</>
                : <><ArrowDownWideNarrow className="h-4 w-4" /> Rerank</>}
            </Button>
          </div>
          {err && <ErrorNote>{err}</ErrorNote>}
          {curlBase && model && (
            <CurlBlock build={(tok) => curlJson(`${curlBase}/rerank`, tok, body, extraHeaders)} />
          )}
          {result && <Results run={result} />}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-sm font-medium">Rerank history</CardTitle>
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
            <p className="py-4 text-center text-xs text-muted-foreground">No reranks yet.</p>
          ) : (
            history.map((r) => (
              <details key={r.id} className="group rounded-md border border-border">
                <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-xs">
                  <span className="rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                    rerank
                  </span>
                  <span className="font-mono text-[11px] text-muted-foreground">{r.model}</span>
                  <span className="truncate">{r.query}</span>
                  <span className="shrink-0 text-muted-foreground">· {r.n} doc{r.n === 1 ? "" : "s"}</span>
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
                <div className="border-t border-border bg-muted/30 p-3">
                  <Results run={r} />
                </div>
              </details>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// Ranked list + the separation read. The bar is the score itself (0..1), so a healthy
// run reads as one long bar and the rest near-empty; a flat run is visually obvious.
function Results({ run }: { run: RerankRun }) {
  const top = run.results[0];
  const gap = run.results.length > 1 ? top.score - run.results[1].score : top.score;
  const weak = gap < WEAK_GAP;
  return (
    <div className="space-y-1.5">
      <div className="text-xs text-muted-foreground">
        {run.results.length} of {run.n} document{run.n === 1 ? "" : "s"}
        {run.topN ? ` · top_n ${run.topN}` : ""}
        {` · top #${top.index} @${fmtScore(top.score)} · gap ${fmtScore(gap)}`}
        {run.tokens != null ? ` · ${run.tokens} tokens` : ""}
      </div>
      {weak && (
        <p className="text-[11px] text-amber-600 dark:text-amber-400">
          Scores aren&apos;t cleanly separated. Reranker scores should sit near 0 or 1 — a flat middle spread usually
          means the scoring template isn&apos;t applied upstream (a vLLM reranker needs <code className="font-mono">--chat-template</code>),
          not a borderline match.
        </p>
      )}
      {/* Same weight as the chat playground's answer pane (font-mono text-xs on
          bg-muted/40) — the ranked list IS this mode's output, so it shouldn't read a
          size smaller than chat's just because it's a table of numbers. */}
      <div className="max-h-72 space-y-1.5 overflow-y-auto rounded-md border border-border bg-muted/40 p-3">
        {run.results.map((d, rank) => (
          <div key={`${d.index}-${rank}`} className="text-xs leading-relaxed">
            <div className="flex items-baseline gap-2">
              <span className="shrink-0 font-mono text-xs text-muted-foreground">#{rank + 1}</span>
              <span className="shrink-0 font-mono text-xs text-muted-foreground">[{d.index}]</span>
              <span className="truncate">{d.text}</span>
              <span className="ml-auto shrink-0 font-mono text-xs">{fmtScore(d.score)}</span>
            </div>
            <div className="mt-0.5 h-1 w-full overflow-hidden rounded bg-muted">
              <div
                className="h-full rounded bg-primary"
                style={{ width: `${Math.max(0, Math.min(1, d.score)) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
