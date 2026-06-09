"use client";

// Reusable stress test — the same `vllm bench serve`-style load generator the
// serverless tab uses (shared core in @/lib/stress-bench + the server runner at
// /api/stress/run). Server mode drives the gateway from the Next.js server (true
// concurrency, no browser cap); browser mode fires from this tab. Each resource
// passes how to reach its endpoint (`browserUrl` + the `/api/stress/run` payload);
// `openai` selects the OpenAI-compatible body (the LLM proxy) vs the worker queue.

import { useRef, useState } from "react";
import { Activity, Clock, Loader2, Play, TrendingUp, X, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { NumberField } from "@/components/ui/number-field";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { runStressBench, formatLine } from "@/lib/stress-bench";
import type { Stat, Summary } from "@/lib/stress-history";

type RunMode = "server" | "browser";

export function StressTest({
  models,
  browserUrl,
  serverPayload,
  openai = false,
}: {
  models: string[];
  browserUrl: string;
  serverPayload: Record<string, unknown>; // e.g. { path: "proxy/<name>/v1/chat/completions" }
  openai?: boolean;
}) {
  const [model, setModel] = useState(models[0] ?? "");
  const [inputLen, setInputLen] = useState(128);
  const [outputLen, setOutputLen] = useState(128);
  const [numPrompts, setNumPrompts] = useState(50);
  const [concurrency, setConcurrency] = useState(10);
  const [mode, setMode] = useState<RunMode>("server");

  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(0);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [errText, setErrText] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const appendLine = (line: string) =>
    setLogLines((prev) => (prev.length >= 1000 ? [...prev.slice(prev.length - 999), line] : [...prev, line]));

  const reset = () => {
    setRunning(true); setErrText(null); setSummary(null); setDone(0); setLogLines([]);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    return ctrl;
  };

  async function runBrowser() {
    const ctrl = reset();
    try {
      const { summary: sum, firstError } = await runStressBench(
        browserUrl,
        { "Content-Type": "application/json" },
        { model, inputLen, outputLen, numPrompts, concurrency, openai },
        { signal: ctrl.signal, onResult: (r, completed) => { setDone(completed); appendLine(formatLine(completed, r)); } },
      );
      setSummary(sum);
      if (sum.successful === 0) setErrText(firstError ? `All requests failed — ${firstError}` : "All requests failed.");
    } catch (e) {
      if (!ctrl.signal.aborted) setErrText(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  async function runServer() {
    const ctrl = reset();
    try {
      const res = await fetch("/api/stress/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...serverPayload, model, input_len: inputLen, output_len: outputLen, num_prompts: numPrompts, concurrency }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) throw new Error((await res.text().catch(() => "")) || res.statusText);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done: rdone } = await reader.read();
        if (rdone) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const ln of lines) {
          if (!ln.trim()) continue;
          let ev: { type?: string; done?: number; line?: string; summary?: Summary; firstError?: string; error?: string };
          try { ev = JSON.parse(ln); } catch { continue; }
          if (ev.type === "progress") {
            if (typeof ev.done === "number") setDone(ev.done);
            if (ev.line) appendLine(ev.line);
          } else if (ev.type === "summary" && ev.summary) {
            setSummary(ev.summary);
            if (ev.summary.successful === 0) setErrText(ev.firstError ? `All requests failed — ${ev.firstError}` : "All requests failed.");
          } else if (ev.type === "error" && ev.error) {
            setErrText(ev.error);
          }
        }
      }
    } catch (e) {
      if (!ctrl.signal.aborted) setErrText(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  const run = () => (mode === "server" ? runServer() : runBrowser());
  const stop = () => { abortRef.current?.abort(); setRunning(false); };

  if (models.length === 0) {
    return <Card><CardContent className="py-8 text-center text-sm text-muted-foreground">No models available.</CardContent></Card>;
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Stress test</CardTitle>
          <CardDescription className="text-xs">
            <code className="font-mono">vllm bench serve</code>-style load against this endpoint&apos;s streaming API —
            fires <span className="font-mono">{numPrompts}</span> requests at concurrency{" "}
            <span className="font-mono">{concurrency}</span> and reports throughput + latency percentiles.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
            <Field label="model" width="w-[240px]">
              <Select value={model} onValueChange={setModel} disabled={running}>
                <SelectTrigger className="h-8 w-full min-w-0 font-mono text-xs" title={model}><SelectValue /></SelectTrigger>
                <SelectContent>{models.map((m) => <SelectItem key={m} value={m} className="font-mono text-xs">{m}</SelectItem>)}</SelectContent>
              </Select>
            </Field>
            <Field label="input len (≈tok)"><NumberField min={1} max={32768} value={inputLen} onChange={setInputLen} disabled={running} className="h-8 w-28 font-mono" /></Field>
            <Field label="output len"><NumberField min={1} max={8192} value={outputLen} onChange={setOutputLen} disabled={running} className="h-8 w-28 font-mono" /></Field>
            <Field label="num prompts"><NumberField min={1} max={5000} value={numPrompts} onChange={setNumPrompts} disabled={running} className="h-8 w-28 font-mono" /></Field>
            <Field label="concurrency"><NumberField min={1} max={1024} value={concurrency} onChange={setConcurrency} disabled={running} className="h-8 w-28 font-mono" /></Field>
            <Field label="run from" width="w-[180px]">
              <Select value={mode} onValueChange={(v) => setMode(v as RunMode)} disabled={running}>
                <SelectTrigger className="h-8 w-full min-w-0 text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="server" className="text-xs">Server (full concurrency)</SelectItem>
                  <SelectItem value="browser" className="text-xs">Browser</SelectItem>
                </SelectContent>
              </Select>
            </Field>
            <div className="flex flex-col gap-1">
              <span className="text-xs text-transparent">run</span>
              {running ? (
                <Button variant="outline" onClick={stop}><X className="h-4 w-4" /> Stop</Button>
              ) : (
                <Button onClick={run}><Play className="h-4 w-4" /> Run</Button>
              )}
            </div>
          </div>

          {running && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Loader2 className="h-3 w-3 animate-spin" /> {done} / {numPrompts} requests
            </div>
          )}

          <p className="text-xs text-muted-foreground">
            {mode === "server"
              ? "Server-driven — the Next.js server runs the load and drives the gateway directly, so the requested concurrency is actually achieved (no browser connection cap). Latency is measured from the server, so it excludes your browser's hop to it."
              : "Browser-driven — effective concurrency is capped by the browser's per-host connection limit. Use Server for true concurrency."}
          </p>

          {logLines.length > 0 && (
            <pre className="max-h-48 overflow-auto rounded-md border border-border bg-muted/40 p-2 font-mono text-[11px] leading-relaxed scrollbar-thin">
              {logLines.join("\n")}
            </pre>
          )}
        </CardContent>
      </Card>

      {errText && <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">{errText}</div>}

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
              <CardDescription className="text-xs">in≈{inputLen} · out={outputLen} · {numPrompts} prompts · concurrency {concurrency}</CardDescription>
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

function Field({ label, width, children }: { label: string; width?: string; children: React.ReactNode }) {
  return (
    <div className={"flex flex-col gap-1 " + (width ?? "")}>
      <span className="text-xs text-muted-foreground">{label}</span>
      {children}
    </div>
  );
}

function Kpi({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">{icon}{label}</div>
      <div className="mt-1 font-mono text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function Row({ k, v, danger }: { k: string; v: string; danger?: boolean }) {
  return (
    <tr>
      <td className="px-4 py-2 text-muted-foreground">{k}</td>
      <td className={"px-4 py-2 text-right font-mono tabular-nums " + (danger ? "text-destructive" : "text-foreground")}>{v}</td>
    </tr>
  );
}

function StatRows({ label, stat, digits }: { label: string; stat: Stat; digits: number }) {
  return (
    <>
      <Row k={`${label} — mean (ms)`} v={stat.mean.toFixed(digits)} />
      <Row k={`${label} — median (ms)`} v={stat.median.toFixed(digits)} />
      <Row k={`${label} — p99 (ms)`} v={stat.p99.toFixed(digits)} />
    </>
  );
}
