"use client";

import { useMemo, useRef, useState } from "react";
import { Activity, Clock, Loader2, Play, TrendingUp, X, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { NumberField } from "@/components/ui/number-field";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { AppRecord } from "@/lib/types";

// A vLLM-bench-serve-style load generator, run client-side against this
// endpoint's live OpenAI streaming API. Fires `num_prompts` chat completions at
// `concurrency`, measures per-request TTFT / TPOT / E2E latency + token usage,
// and reports the same metric block `vllm bench serve` (the core of llm-benchmaq)
// prints: throughput, latency percentiles, success/fail counts.
//
// Concurrency is driven from the browser through the same-origin proxy, so the
// effective parallelism is bounded by the browser's per-host connection limit
// (≈6 on HTTP/1.1, many more on HTTP/2). For very high concurrency on a remote
// box, drive `vllm bench` from a host via the API key (see the README).

type ReqResult = {
  ok: boolean;
  ttftMs: number; // time to first token
  e2eMs: number; // end to end
  tpotMs: number; // mean time per output token
  outTokens: number;
  promptTokens: number;
  error?: string;
};

type Stat = { mean: number; median: number; p99: number };
type Summary = {
  successful: number;
  failed: number;
  durationS: number;
  reqThroughput: number; // req/s
  outThroughput: number; // output tok/s
  totalThroughput: number; // (in+out) tok/s
  ttft: Stat;
  tpot: Stat;
  e2e: Stat;
};

function perfNow(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function deltaContent(chunk: Record<string, unknown>): string {
  const choices = chunk.choices as Array<{ delta?: { content?: unknown } }> | undefined;
  const c = choices?.[0]?.delta?.content;
  return typeof c === "string" ? c : "";
}
function usageTokens(chunk: Record<string, unknown>): { out: number | null; prompt: number | null } {
  const u = (chunk.usage ?? null) as { completion_tokens?: unknown; prompt_tokens?: unknown } | null;
  const out = typeof u?.completion_tokens === "number" ? u.completion_tokens : null;
  const prompt = typeof u?.prompt_tokens === "number" ? u.prompt_tokens : null;
  return { out, prompt };
}

function pct(arr: number[], p: number): number {
  if (arr.length === 0) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.max(0, Math.ceil((p / 100) * s.length) - 1));
  return s[i];
}
const mean = (arr: number[]) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
const statOf = (arr: number[]): Stat => ({ mean: mean(arr), median: pct(arr, 50), p99: pct(arr, 99) });

// ~1 token per "word ", good enough to size an input prompt for load testing.
function makePrompt(inputLen: number): string {
  const n = Math.max(1, inputLen);
  return ("word ".repeat(n)).trim();
}

export function StressTab({ app }: { app: AppRecord }) {
  const models = useMemo(() => {
    if (app.mode === "multi" && app.models?.length) return app.models.map((m) => m.model).filter(Boolean);
    return app.model ? [app.model] : [];
  }, [app]);

  const [model, setModel] = useState("");
  const selectedModel = model || models[0] || "";

  const [inputLen, setInputLen] = useState(128);
  const [outputLen, setOutputLen] = useState(128);
  const [numPrompts, setNumPrompts] = useState(50);
  const [concurrency, setConcurrency] = useState(10);

  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(0);
  const [errText, setErrText] = useState<string | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Fire one streaming chat completion, timing TTFT / E2E / output tokens.
  async function oneRequest(signal: AbortSignal): Promise<ReqResult> {
    const t0 = perfNow();
    let tFirst: number | null = null;
    let counted = 0;
    let usageOut: number | null = null;
    let promptTokens = 0;
    const body: Record<string, unknown> = {
      endpoint: "/v1/chat/completions",
      messages: [{ role: "user", content: makePrompt(inputLen) }],
      max_tokens: outputLen,
      stream_options: { include_usage: true },
    };
    if (selectedModel) body.model = selectedModel;
    try {
      const res = await fetch(`/api/proxy/stream/${encodeURIComponent(app.app_id)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      });
      if (!res.ok || !res.body) {
        const txt = await res.text().catch(() => "");
        throw new Error(txt || res.statusText);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done: rdone } = await reader.read();
        if (rdone) break;
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
            if (chunk.error) throw new Error(String(chunk.error));
            const u = usageTokens(chunk);
            if (u.out != null) usageOut = u.out;
            if (u.prompt != null) promptTokens = u.prompt;
            if (deltaContent(chunk)) {
              if (tFirst === null) tFirst = perfNow();
              counted += 1;
            }
          }
        }
      }
      const tEnd = perfNow();
      const out = usageOut ?? counted;
      const ttft = (tFirst ?? tEnd) - t0;
      const e2e = tEnd - t0;
      const gen = tEnd - (tFirst ?? tEnd);
      return {
        ok: true,
        ttftMs: ttft,
        e2eMs: e2e,
        tpotMs: out > 1 ? gen / (out - 1) : 0,
        outTokens: out,
        promptTokens,
      };
    } catch (e) {
      return {
        ok: false,
        ttftMs: 0,
        e2eMs: 0,
        tpotMs: 0,
        outTokens: 0,
        promptTokens: 0,
        error: e instanceof Error ? e.message : String(e),
      };
    }
  }

  async function run() {
    setRunning(true);
    setErrText(null);
    setSummary(null);
    setDone(0);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const total = Math.max(1, numPrompts);
    const conc = Math.max(1, Math.min(concurrency, total));
    const results: ReqResult[] = [];
    let launched = 0;
    let completed = 0;
    const batchStart = perfNow();

    const worker = async () => {
      while (launched < total && !ctrl.signal.aborted) {
        launched += 1;
        const r = await oneRequest(ctrl.signal);
        results.push(r);
        completed += 1;
        setDone(completed);
      }
    };
    try {
      await Promise.all(Array.from({ length: conc }, () => worker()));
    } finally {
      const durationS = (perfNow() - batchStart) / 1000;
      const ok = results.filter((r) => r.ok);
      const firstErr = results.find((r) => !r.ok)?.error;
      if (ok.length === 0) {
        setErrText(firstErr ? `All requests failed — ${firstErr}` : "All requests failed.");
      }
      const totalOut = ok.reduce((a, r) => a + r.outTokens, 0);
      const totalTok = ok.reduce((a, r) => a + r.outTokens + r.promptTokens, 0);
      setSummary({
        successful: ok.length,
        failed: results.length - ok.length,
        durationS,
        reqThroughput: durationS > 0 ? ok.length / durationS : 0,
        outThroughput: durationS > 0 ? totalOut / durationS : 0,
        totalThroughput: durationS > 0 ? totalTok / durationS : 0,
        ttft: statOf(ok.map((r) => r.ttftMs)),
        tpot: statOf(ok.map((r) => r.tpotMs).filter((x) => x > 0)),
        e2e: statOf(ok.map((r) => r.e2eMs)),
      });
      setRunning(false);
    }
  }

  function stop() {
    abortRef.current?.abort();
    setRunning(false);
  }

  const pctDone = running && numPrompts > 0 ? Math.round((done / numPrompts) * 100) : 0;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Stress test</CardTitle>
          <CardDescription className="text-xs">
            <code className="font-mono">vllm bench serve</code>-style load against this endpoint&apos;s streaming
            API — fires <span className="font-mono">{numPrompts}</span> requests at concurrency{" "}
            <span className="font-mono">{concurrency}</span> and reports throughput + latency percentiles.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
            {models.length > 0 && (
              <Field label="model" width="w-[240px]">
                <Select value={selectedModel} onValueChange={setModel} disabled={running}>
                  <SelectTrigger className="h-8 font-mono text-xs">
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
              </Field>
            )}
            <Field label="input len (≈tok)">
              <NumberField min={1} max={32768} value={inputLen} onChange={setInputLen} disabled={running} className="h-8 w-28 font-mono" />
            </Field>
            <Field label="output len">
              <NumberField min={1} max={8192} value={outputLen} onChange={setOutputLen} disabled={running} className="h-8 w-28 font-mono" />
            </Field>
            <Field label="num prompts">
              <NumberField min={1} max={5000} value={numPrompts} onChange={setNumPrompts} disabled={running} className="h-8 w-28 font-mono" />
            </Field>
            <Field label="concurrency">
              <NumberField min={1} max={1024} value={concurrency} onChange={setConcurrency} disabled={running} className="h-8 w-28 font-mono" />
            </Field>
            <div className="flex-1" />
            {running ? (
              <Button variant="outline" onClick={stop}>
                <X className="h-4 w-4" /> Stop
              </Button>
            ) : (
              <Button onClick={run} disabled={!selectedModel && models.length > 0}>
                <Play className="h-4 w-4" /> Run
              </Button>
            )}
          </div>

          {running && (
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" />
                {done} / {numPrompts} requests
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div className="h-full bg-primary transition-all" style={{ width: `${pctDone}%` }} />
              </div>
            </div>
          )}

          <p className="text-[11px] leading-relaxed text-muted-foreground">
            Client-driven — effective concurrency is capped by the browser&apos;s per-host connection limit. For
            very high concurrency, drive <code className="font-mono">vllm bench</code> from a host via your API key
            (see README).
          </p>
        </CardContent>
      </Card>

      {errText && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {errText}
        </div>
      )}

      {summary && (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Kpi icon={<Zap className="h-4 w-4" />} label="Output throughput" value={`${summary.outThroughput.toFixed(1)} tok/s`} />
            <Kpi icon={<TrendingUp className="h-4 w-4" />} label="Request throughput" value={`${summary.reqThroughput.toFixed(2)} req/s`} />
            <Kpi icon={<Clock className="h-4 w-4" />} label="Median TTFT" value={`${summary.ttft.median.toFixed(0)} ms`} />
            <Kpi icon={<Activity className="h-4 w-4" />} label="Median TPOT" value={`${summary.tpot.median.toFixed(1)} ms`} />
          </div>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Serving benchmark result</CardTitle>
              <CardDescription className="text-xs">
                in≈{inputLen} · out={outputLen} · {numPrompts} prompts · concurrency {concurrency}
              </CardDescription>
            </CardHeader>
            <CardContent className="px-0 pb-0">
              <table className="w-full text-sm">
                <tbody className="divide-y divide-border">
                  <Row k="Successful requests" v={String(summary.successful)} />
                  <Row k="Failed requests" v={String(summary.failed)} danger={summary.failed > 0} />
                  <Row k="Benchmark duration (s)" v={summary.durationS.toFixed(2)} />
                  <Row k="Request throughput (req/s)" v={summary.reqThroughput.toFixed(2)} />
                  <Row k="Output token throughput (tok/s)" v={summary.outThroughput.toFixed(1)} />
                  <Row k="Total token throughput (tok/s)" v={summary.totalThroughput.toFixed(1)} />
                  <StatRows label="TTFT" stat={summary.ttft} digits={0} />
                  <StatRows label="TPOT" stat={summary.tpot} digits={1} />
                  <StatRows label="E2E latency" stat={summary.e2e} digits={0} />
                </tbody>
              </table>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function Field({ label, children, width }: { label: string; children: React.ReactNode; width?: string }) {
  return (
    <div className={`flex flex-col gap-1 ${width ?? ""}`}>
      <span className="text-xs text-muted-foreground">{label}</span>
      {children}
    </div>
  );
}

function Kpi({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
        <span className="flex h-6 w-6 items-center justify-center rounded-md bg-muted text-muted-foreground">
          {icon}
        </span>
        {label}
      </div>
      <div className="mt-1.5 text-xl font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function Row({ k, v, danger }: { k: string; v: string; danger?: boolean }) {
  return (
    <tr>
      <td className="px-4 py-1.5 text-muted-foreground">{k}</td>
      <td className={`px-4 py-1.5 text-right font-mono tabular-nums ${danger ? "text-destructive" : "text-foreground"}`}>
        {v}
      </td>
    </tr>
  );
}

function StatRows({ label, stat, digits }: { label: string; stat: Stat; digits: number }) {
  return (
    <>
      <Row k={`Mean ${label} (ms)`} v={stat.mean.toFixed(digits)} />
      <Row k={`Median ${label} (ms)`} v={stat.median.toFixed(digits)} />
      <Row k={`P99 ${label} (ms)`} v={stat.p99.toFixed(digits)} />
    </>
  );
}
