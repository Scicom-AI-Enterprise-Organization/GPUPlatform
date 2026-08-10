"use client";

// End-of-utterance (turn detector) playground: sends the RAW /v1/completions request a
// LiveKit voice agent sends, and reads back the one number it actually uses.
//
// The request is unusual and the whole mode exists because of it: `max_tokens: 1` +
// `allowed_token_ids: [<|im_end|>]` + `logprobs: 1`. The model is not asked to SAY
// anything — `text` comes back empty. It is asked one question ("how likely is the turn
// to end here?") and the answer arrives only as `choices[0].logprobs.token_logprobs[0]`,
// which the client turns into a probability with exp().
//
// Like the rerank mode, this is deliberately opinionated about how the result is READ,
// because the failure here is SILENT. A LiveKit client that cannot parse the response
// falls back to p = 1.0 — "the user has finished talking" — so a broken pipeline does
// not error, it just makes the agent interrupt people. Three things are therefore
// checked and surfaced rather than left for the reader to notice:
//   1. logprobs present at all      — absent → the client would silently fall back to 1.0
//   2. allowed_token_ids enforced   — the sampled token must be the EOU token even when
//                                     it is far down the model's own preference list; if
//                                     a proxy strips the field, the argmax comes back
//                                     instead and every score is meaningless
//   3. separation across utterances — identical scores for a finished vs unfinished
//                                     sentence means nothing is being predicted
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Split, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { CurlBlock, curlJson } from "@/components/playground/curl-block";
import { ErrorNote } from "@/components/playground/chat-playground";

type EouRow = {
  text: string;
  logprob: number | null;     // null = no logprobs in the response (the silent failure)
  prob: number | null;
  token: string | null;       // token actually sampled
  argmax: string | null;      // the model's own most-likely token, from top_logprobs
  argmaxLogprob: number | null;
  blocked: boolean;           // red-team guard returned a filtered response
};
type EouRun = {
  id: string;
  at: number;
  model: string;
  tokenId: number;
  template: string;
  rows: EouRow[];
};

const MAX_HISTORY = 50;
// <|im_end|> for Qwen-family turn detectors (livekit/turn-detector). Model-specific:
// a detector built on another base uses that tokenizer's end-of-turn id.
const DEFAULT_EOU_TOKEN = 151645;
// `{text}` is substituted per line. No trailing <|im_end|> — the model predicts whether
// the turn ends HERE, so closing it first asks a different question entirely (what
// follows a completed turn is the next message's newline, and p(EOU) collapses).
const DEFAULT_TEMPLATE = "<|im_start|>user\\n{text}";
// Two utterances differing by one word: the first is mid-sentence, the second is a
// finished question. A working detector separates them by orders of magnitude. A single
// utterance can't show separation at all, which is why the box starts with a pair.
const EXAMPLE_UTTERANCES = [
  "hello how are",
  "hello how are you",
  "i want to book a flight to",
  "i want to book a flight to kuala lumpur",
].join("\n");

// Below this spread (in nats, across the run) nothing is really being discriminated.
// ~2.3 nats = a 10x probability ratio.
const WEAK_SPREAD = 2.3;

const fmtProb = (p: number) => (p >= 0.001 ? p.toFixed(4) : p.toExponential(2));
// The bar is LOG-scaled: p(EOU) legitimately ranges over many orders of magnitude
// (1e-9 … 1e-1), so a linear bar would render every row as empty and hide exactly the
// separation this mode exists to show.
const BAR_FLOOR = -20;
const barPct = (lp: number) => Math.max(0, Math.min(1, (lp - BAR_FLOOR) / -BAR_FLOOR)) * 100;

export function EouPlayground({ models, basePath, curlBase, storageKey, extraHeaders }: {
  models: string[];
  basePath: string;
  curlBase?: string;
  storageKey: string;
  extraHeaders?: Record<string, string>;
}) {
  const [model, setModel] = useState(models[0] ?? "");
  const [utterText, setUtterText] = useState(EXAMPLE_UTTERANCES);
  const [tokenId, setTokenId] = useState(String(DEFAULT_EOU_TOKEN));
  const [template, setTemplate] = useState(DEFAULT_TEMPLATE);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<EouRun | null>(null);
  // The results land BELOW the form, and with several utterances the form alone can fill
  // the viewport — so a finished run would sit off-screen and read as "nothing happened".
  // Scroll it into view once it renders instead of making the reader go find it.
  const resultRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (result) resultRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [result]);

  const [history, setHistory] = useState<EouRun[]>([]);
  const historyRef = useRef(history);
  historyRef.current = history;
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (raw) setHistory(JSON.parse(raw));
    } catch { /* ignore corrupt/absent */ }
  }, [storageKey]);
  const persist = useCallback((next: EouRun[]) => {
    setHistory(next);
    try { window.localStorage.setItem(storageKey, JSON.stringify(next)); } catch { /* quota */ }
  }, [storageKey]);
  const removeRun = useCallback((id: string) => persist(historyRef.current.filter((r) => r.id !== id)), [persist]);
  const clearAll = useCallback(() => persist([]), [persist]);

  const utterances = useMemo(
    () => utterText.split("\n").map((s) => s.trim()).filter(Boolean),
    [utterText],
  );
  const tokNum = Number.parseInt(tokenId, 10);
  const tokValid = Number.isFinite(tokNum) && tokNum >= 0;

  // `\n` is typed literally in the template box; turn it into a real newline for the
  // wire. The curl block renders the same body, so what you copy is what ran.
  const renderPrompt = useCallback(
    (text: string) => template.replace(/\\n/g, "\n").replace("{text}", text),
    [template],
  );

  const bodyFor = useCallback((text: string): Record<string, unknown> => ({
    model,
    prompt: renderPrompt(text),
    max_tokens: 1,
    logprobs: 1,
    allowed_token_ids: [tokNum],
  }), [model, renderPrompt, tokNum]);

  async function onRun() {
    if (!model || utterances.length === 0 || !tokValid) return;
    setBusy(true); setErr(null); setResult(null);
    try {
      const rows: EouRow[] = [];
      for (const text of utterances) {
        const r = await fetch(`${basePath}/completions`, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...(extraHeaders ?? {}) },
          body: JSON.stringify(bodyFor(text)),
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
        const choice = (parsed as {
          choices?: {
            finish_reason?: string;
            logprobs?: {
              token_logprobs?: (number | null)[];
              tokens?: string[];
              top_logprobs?: Record<string, number>[];
            } | null;
          }[];
        }).choices?.[0];
        const lp = choice?.logprobs;
        const logprob = lp?.token_logprobs?.[0];
        const top = lp?.top_logprobs?.[0] ?? {};
        const entries = Object.entries(top);
        const bestPair = entries.length > 0
          ? entries.reduce((a, b) => (b[1] > a[1] ? b : a))
          : null;
        rows.push({
          text,
          logprob: typeof logprob === "number" ? logprob : null,
          prob: typeof logprob === "number" ? Math.exp(logprob) : null,
          token: lp?.tokens?.[0] ?? null,
          argmax: bestPair?.[0] ?? null,
          argmaxLogprob: bestPair?.[1] ?? null,
          // The guard emits finish_reason="content_filter" with logprobs nulled.
          blocked: choice?.finish_reason === "content_filter",
        });
      }
      const entry: EouRun = {
        id: globalThis.crypto?.randomUUID?.() ?? `e-${Date.now()}-${Math.round(Math.random() * 1e6)}`,
        at: Date.now(),
        model,
        tokenId: tokNum,
        template,
        rows,
      };
      setResult(entry);
      persist([entry, ...historyRef.current].slice(0, MAX_HISTORY));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const eouToken = result?.rows.find((r) => r.token)?.token;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">End of utterance (turn detector)</CardTitle>
          <p className="text-xs text-muted-foreground">
            Sends the raw <code className="font-mono">/v1/completions</code> request a LiveKit agent sends —
            <code className="font-mono"> max_tokens:1</code>, <code className="font-mono">logprobs:1</code>,
            <code className="font-mono"> allowed_token_ids</code> — and shows
            <code className="font-mono"> p(EOU) = exp(token_logprobs[0])</code>, the one number the client reads.
            One utterance per line; include a finished and an unfinished one, because a single utterance can&apos;t
            show separation.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">utterances (one per line)</span>
            <Textarea
              value={utterText}
              onChange={(e) => { setUtterText(e.target.value); setResult(null); }}
              placeholder={"hello how are\nhello how are you"}
              className="min-h-[110px] font-mono text-sm"
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
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">EOU token id</span>
              <Input
                value={tokenId}
                onChange={(e) => { setTokenId(e.target.value.replace(/[^0-9]/g, "")); setResult(null); }}
                placeholder={String(DEFAULT_EOU_TOKEN)}
                inputMode="numeric"
                className="h-8 w-[120px] font-mono text-xs"
                title="allowed_token_ids — 151645 is <|im_end|> for Qwen-family detectors"
              />
            </div>
            <Button type="button" onClick={onRun} disabled={busy || !model || utterances.length === 0 || !tokValid} className="ml-auto">
              {busy
                ? <><Loader2 className="h-4 w-4 animate-spin" /> Predicting…</>
                : <><Split className="h-4 w-4" /> Predict ({utterances.length})</>}
            </Button>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">prompt template</span>
            <Input
              value={template}
              onChange={(e) => { setTemplate(e.target.value); setResult(null); }}
              placeholder={DEFAULT_TEMPLATE}
              className="font-mono text-sm"
            />
            <span className="text-[11px] text-muted-foreground">
              <code className="font-mono">{"{text}"}</code> is replaced per line; <code className="font-mono">\n</code> becomes
              a newline. Absolute p(EOU) only matches your agent if this matches the template the client renders —
              the SEPARATION between utterances is the reliable signal here, not the absolute value.
            </span>
          </div>
          {err && <ErrorNote>{err}</ErrorNote>}
          {curlBase && model && utterances.length > 0 && tokValid && (
            <CurlBlock
              build={(tok) => curlJson(`${curlBase}/completions`, tok, bodyFor(utterances[0]), extraHeaders)}
              label={`cURL for this request (first utterance)`}
            />
          )}
          {result && (
            <div ref={resultRef} className="scroll-mt-4">
              <Results run={result} eouToken={eouToken ?? null} />
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-sm font-medium">EOU history</CardTitle>
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
            <p className="py-4 text-center text-xs text-muted-foreground">No predictions yet.</p>
          ) : (
            history.map((r) => (
              <details key={r.id} className="group rounded-md border border-border">
                <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-xs">
                  <span className="rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                    eou
                  </span>
                  <span className="font-mono text-[11px] text-muted-foreground">{r.model}</span>
                  <span className="truncate">{r.rows.length} utterance{r.rows.length === 1 ? "" : "s"}</span>
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
                  <Results run={r} eouToken={r.rows.find((x) => x.token)?.token ?? null} />
                </div>
              </details>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function Results({ run, eouToken }: { run: EouRun; eouToken: string | null }) {
  const scored = run.rows.filter((r) => r.logprob !== null) as (EouRow & { logprob: number; prob: number })[];
  const missing = run.rows.filter((r) => r.logprob === null);
  const blocked = run.rows.filter((r) => r.blocked);
  const spread = scored.length > 1
    ? Math.max(...scored.map((r) => r.logprob)) - Math.min(...scored.map((r) => r.logprob))
    : null;
  // allowed_token_ids proves itself when the sampled token ISN'T the model's own argmax:
  // the constraint had to override the model's preference to produce it.
  const overridden = scored.filter((r) => r.argmax !== null && r.token !== null && r.argmax !== r.token);

  return (
    <div className="space-y-1.5">
      <div className="text-xs text-muted-foreground">
        {run.rows.length} utterance{run.rows.length === 1 ? "" : "s"} · token id {run.tokenId}
        {eouToken ? <> (<code className="font-mono">{eouToken}</code>)</> : null}
        {spread !== null ? ` · spread ${spread.toFixed(2)} nats (~${Math.exp(spread).toLocaleString(undefined, { maximumFractionDigits: 0 })}x)` : ""}
      </div>

      {blocked.length > 0 && (
        <p className="text-[11px] text-destructive">
          {blocked.length} response{blocked.length === 1 ? " was" : "s were"} BLOCKED by the red-teaming guard
          (<code className="font-mono">finish_reason: content_filter</code>), which returns
          <code className="font-mono"> logprobs: null</code>. A LiveKit client cannot parse that and falls back to
          p = 1.0 — the agent would treat every utterance as finished. Turn red-teaming OFF on a turn-detector endpoint.
        </p>
      )}
      {missing.length > 0 && blocked.length === 0 && (
        <p className="text-[11px] text-destructive">
          {missing.length} response{missing.length === 1 ? "" : "s"} came back with no
          <code className="font-mono"> logprobs.token_logprobs</code>. This is the silent failure: the client falls back
          to p = 1.0 and the agent interrupts constantly. Check that the upstream is a completions-capable vLLM and that
          nothing between here and it strips <code className="font-mono">logprobs</code>.
        </p>
      )}
      {scored.length > 0 && overridden.length === 0 && (
        <p className="text-[11px] text-amber-600 dark:text-amber-400">
          The sampled token was also the model&apos;s own most-likely token every time, so this run doesn&apos;t prove
          <code className="font-mono"> allowed_token_ids</code> was applied upstream. Add an unfinished utterance
          (&quot;hello how are&quot;) — the model wants to continue there, so the constraint has to override it.
        </p>
      )}
      {spread !== null && spread < WEAK_SPREAD && (
        <p className="text-[11px] text-amber-600 dark:text-amber-400">
          Scores barely differ across utterances — nothing is being discriminated. Either the prompt template
          doesn&apos;t match what the detector expects, or the response isn&apos;t a real prediction.
        </p>
      )}

      <div className="max-h-72 space-y-1.5 overflow-y-auto rounded-md border border-border bg-muted/40 p-3">
        {run.rows.map((r, i) => (
          <div key={`${i}-${r.text}`} className="text-xs leading-relaxed">
            <div className="flex items-baseline gap-2">
              <span className="shrink-0 font-mono text-xs text-muted-foreground">#{i + 1}</span>
              <span className="truncate">{r.text}</span>
              <span className="ml-auto shrink-0 font-mono text-xs">
                {r.prob !== null ? `p=${fmtProb(r.prob)}` : <span className="text-destructive">no logprobs</span>}
              </span>
            </div>
            <div className="mt-0.5 h-1 w-full overflow-hidden rounded bg-muted">
              {r.logprob !== null && (
                <div className="h-full rounded bg-primary" style={{ width: `${barPct(r.logprob)}%` }} />
              )}
            </div>
            {r.logprob !== null && (
              <div className="mt-0.5 font-mono text-[10px] text-muted-foreground">
                logprob {r.logprob.toFixed(4)}
                {r.token ? ` · sampled ${JSON.stringify(r.token)}` : ""}
                {r.argmax !== null && r.argmax !== r.token
                  ? ` · model wanted ${JSON.stringify(r.argmax)}${r.argmaxLogprob !== null ? ` (${r.argmaxLogprob.toFixed(4)})` : ""} → constraint overrode it`
                  : ""}
              </div>
            )}
          </div>
        ))}
      </div>
      <p className="text-[10px] text-muted-foreground">Bar is log-scaled over {BAR_FLOOR}…0 nats — p(EOU) spans orders of magnitude.</p>
    </div>
  );
}
