"use client";

import { useCallback, useLayoutEffect, useMemo, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Activity, Check, Clock, Download, Link2, Loader2, Play, Trash2, TrendingUp, X, Zap } from "lucide-react";
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
import { useStressHistory, type Stat, type StressRun, type Summary } from "@/lib/stress-history";
import { runStressBench, formatLine } from "@/lib/stress-bench";

// A `vllm bench serve`-style load generator. It runs either on the Next.js
// server (default — drives the gateway directly, so the requested concurrency
// is actually achieved) or in the browser (driving the same-origin proxy, where
// the browser's ~6-connection-per-host limit on HTTP/1.1 caps real parallelism).
// Both paths share the same core (@/lib/stress-bench) and report the metric
// block `vllm bench serve` prints: throughput, latency percentiles, success/fail.
// Stat / Summary / StressRun live in @/lib/stress-history.

type RunMode = "server" | "browser";

export function StressTab({ app }: { app: AppRecord }) {
  const models = useMemo(() => {
    if (app.mode === "multi" && app.models?.length) return app.models.map((m) => m.model).filter(Boolean);
    return app.model ? [app.model] : [];
  }, [app]);

  // Test config is seeded from the URL and mirrored back to it on change, so a
  // configured run is shareable / survives reload (?model=&input_len=&output_len=
  // &num_prompts=&concurrency=).
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [model, setModel] = useState(searchParams.get("model") ?? "");
  const selectedModel = model || models[0] || "";

  const [inputLen, setInputLen] = useState(() => Number(searchParams.get("input_len")) || 128);
  const [outputLen, setOutputLen] = useState(() => Number(searchParams.get("output_len")) || 128);
  const [numPrompts, setNumPrompts] = useState(() => Number(searchParams.get("num_prompts")) || 50);
  const [concurrency, setConcurrency] = useState(() => Number(searchParams.get("concurrency")) || 10);

  const writeUrl = useCallback(
    (next: { model?: string; inputLen?: number; outputLen?: number; numPrompts?: number; concurrency?: number }) => {
      const p = new URLSearchParams();
      p.set("tab", "stress");
      const m = next.model ?? selectedModel;
      if (m) p.set("model", m);
      p.set("input_len", String(next.inputLen ?? inputLen));
      p.set("output_len", String(next.outputLen ?? outputLen));
      p.set("num_prompts", String(next.numPrompts ?? numPrompts));
      p.set("concurrency", String(next.concurrency ?? concurrency));
      router.replace(`${pathname}?${p.toString()}`, { scroll: false });
    },
    [selectedModel, inputLen, outputLen, numPrompts, concurrency, pathname, router],
  );

  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(0);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [errText, setErrText] = useState<string | null>(null);
  const [saveErr, setSaveErr] = useState<string | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [mode, setMode] = useState<RunMode>("server");
  const abortRef = useRef<AbortController | null>(null);

  // Saved runs (server-side, per-endpoint) for cross-run / cross-model compare
  // and shareable links.
  const { runs: savedRuns, error: historyError, add: saveRun, remove: removeRun, clear: clearRuns } =
    useStressHistory(app.app_id);

  // A shared comparison link pins specific run ids (?runs=id1,id2).
  const pinnedIds = useMemo(() => {
    const r = searchParams.get("runs");
    return r ? r.split(",").map((s) => s.trim()).filter(Boolean) : null;
  }, [searchParams]);

  const appendLine = (line: string) =>
    setLogLines((prev) => (prev.length >= 2000 ? [...prev.slice(prev.length - 1999), line] : [...prev, line]));

  // Auto-save a completed run (not a user-aborted one) server-side so it's there
  // to compare / share later — even an all-failed run is a useful reference.
  function persist(sum: Summary, aborted: boolean) {
    if (aborted || sum.successful + sum.failed === 0) return;
    void saveRun({
      model: selectedModel,
      input_len: inputLen,
      output_len: outputLen,
      num_prompts: numPrompts,
      concurrency,
      summary: sum,
    }).catch((e) => setSaveErr(e instanceof Error ? e.message : String(e)));
  }

  function resetForRun(): AbortController {
    setRunning(true);
    setErrText(null);
    setSaveErr(null);
    setSummary(null);
    setDone(0);
    setLogLines([]);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    return ctrl;
  }

  // Browser-driven: fires requests from this tab via the same-origin proxy.
  // Capped by the browser's per-host connection limit.
  async function runBrowser() {
    const ctrl = resetForRun();
    try {
      const { summary: sum, firstError } = await runStressBench(
        `/api/proxy/stream/${encodeURIComponent(app.app_id)}`,
        { "Content-Type": "application/json" },
        { model: selectedModel, inputLen, outputLen, numPrompts, concurrency },
        { signal: ctrl.signal, onResult: (r, completed) => { setDone(completed); appendLine(formatLine(completed, r)); } },
      );
      setSummary(sum);
      if (sum.successful === 0) setErrText(firstError ? `All requests failed — ${firstError}` : "All requests failed.");
      persist(sum, ctrl.signal.aborted);
    } catch (e) {
      if (!ctrl.signal.aborted) setErrText(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  // Server-driven: the Next.js server runs the load (no browser connection cap)
  // and streams NDJSON progress back. See app/api/stress/run/route.ts.
  async function runServer() {
    const ctrl = resetForRun();
    try {
      const res = await fetch("/api/stress/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          app_id: app.app_id,
          model: selectedModel,
          input_len: inputLen,
          output_len: outputLen,
          num_prompts: numPrompts,
          concurrency,
        }),
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) throw new Error((await res.text().catch(() => "")) || res.statusText);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let final: Summary | null = null;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const ln of lines) {
          if (!ln.trim()) continue;
          let ev: { type?: string; done?: number; line?: string; summary?: Summary; firstError?: string; error?: string };
          try {
            ev = JSON.parse(ln);
          } catch {
            continue;
          }
          if (ev.type === "progress") {
            if (typeof ev.done === "number") setDone(ev.done);
            if (ev.line) appendLine(ev.line);
          } else if (ev.type === "summary" && ev.summary) {
            final = ev.summary;
            setSummary(ev.summary);
            if (ev.summary.successful === 0) {
              setErrText(ev.firstError ? `All requests failed — ${ev.firstError}` : "All requests failed.");
            }
          } else if (ev.type === "error" && ev.error) {
            setErrText(ev.error);
          }
        }
      }
      if (final) persist(final, ctrl.signal.aborted);
    } catch (e) {
      if (!ctrl.signal.aborted) setErrText(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  const run = () => (mode === "server" ? runServer() : runBrowser());

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
                <Select value={selectedModel} onValueChange={(v) => { setModel(v); writeUrl({ model: v }); }} disabled={running}>
                  <SelectTrigger className="h-8 w-full min-w-0 font-mono text-xs" title={selectedModel}>
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
              <NumberField min={1} max={32768} value={inputLen} onChange={(v) => { setInputLen(v); writeUrl({ inputLen: v }); }} disabled={running} className="h-8 w-28 font-mono" />
            </Field>
            <Field label="output len">
              <NumberField min={1} max={8192} value={outputLen} onChange={(v) => { setOutputLen(v); writeUrl({ outputLen: v }); }} disabled={running} className="h-8 w-28 font-mono" />
            </Field>
            <Field label="num prompts">
              <NumberField min={1} max={5000} value={numPrompts} onChange={(v) => { setNumPrompts(v); writeUrl({ numPrompts: v }); }} disabled={running} className="h-8 w-28 font-mono" />
            </Field>
            <Field label="concurrency">
              <NumberField min={1} max={1024} value={concurrency} onChange={(v) => { setConcurrency(v); writeUrl({ concurrency: v }); }} disabled={running} className="h-8 w-28 font-mono" />
            </Field>
            <Field label="run from" width="w-[180px]">
              <Select value={mode} onValueChange={(v) => setMode(v as RunMode)} disabled={running}>
                <SelectTrigger className="h-8 w-full min-w-0 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="server" className="text-xs">Server (full concurrency)</SelectItem>
                  <SelectItem value="browser" className="text-xs">Browser (capped)</SelectItem>
                </SelectContent>
              </Select>
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

          {logLines.length > 0 && <StressLog lines={logLines} />}

          <p className="text-[11px] leading-relaxed text-muted-foreground">
            {mode === "server" ? (
              <>
                Server-driven — the Next.js server runs the load and drives the gateway directly, so the requested
                concurrency is actually achieved (no browser connection cap). Latency is measured from the server, so
                it excludes your browser&apos;s hop to it.
              </>
            ) : (
              <>
                Browser-driven — effective concurrency is capped by the browser&apos;s per-host connection limit
                (~6 on HTTP/1.1). Switch <span className="font-mono">run from</span> to{" "}
                <span className="font-mono">Server</span> for true concurrency. For an isolated host-side benchmark,
                use the Benchmark feature (<code className="font-mono">vllm bench serve</code>).
              </>
            )}
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

      {saveErr && (
        <div className="rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
          Run finished but couldn&apos;t be saved server-side: {saveErr}
        </div>
      )}
      {historyError && savedRuns.length === 0 && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          Couldn&apos;t load saved runs: {historyError}
        </div>
      )}
      {savedRuns.length > 0 && (
        <SavedRunsComparison
          runs={savedRuns}
          onRemove={removeRun}
          onClear={clearRuns}
          pathname={pathname}
          pinnedIds={pinnedIds}
        />
      )}
    </div>
  );
}

// ── Saved runs: a side-by-side matrix so you can compare runs / models over
// time. Metrics are highlighted per-row: best throughput / lowest latency wins.
type MetricDef = {
  label: string;
  get: (s: Summary) => number;
  fmt: (n: number) => string;
  dir: "high" | "low" | "none";
};
const COMPARE_METRICS: MetricDef[] = [
  { label: "Output throughput (tok/s)", get: (s) => s.outThroughput, fmt: (n) => n.toFixed(1), dir: "high" },
  { label: "Total throughput (tok/s)", get: (s) => s.totalThroughput, fmt: (n) => n.toFixed(1), dir: "high" },
  { label: "Request throughput (req/s)", get: (s) => s.reqThroughput, fmt: (n) => n.toFixed(2), dir: "high" },
  { label: "Median TTFT (ms)", get: (s) => s.ttft.median, fmt: (n) => n.toFixed(0), dir: "low" },
  { label: "P99 TTFT (ms)", get: (s) => s.ttft.p99, fmt: (n) => n.toFixed(0), dir: "low" },
  { label: "Median TPOT (ms)", get: (s) => s.tpot.median, fmt: (n) => n.toFixed(1), dir: "low" },
  { label: "Median E2E (ms)", get: (s) => s.e2e.median, fmt: (n) => n.toFixed(0), dir: "low" },
  { label: "P99 E2E (ms)", get: (s) => s.e2e.p99, fmt: (n) => n.toFixed(0), dir: "low" },
  { label: "Successful", get: (s) => s.successful, fmt: (n) => String(n), dir: "high" },
  { label: "Failed", get: (s) => s.failed, fmt: (n) => String(n), dir: "low" },
  { label: "Duration (s)", get: (s) => s.durationS, fmt: (n) => n.toFixed(2), dir: "none" },
];

function bestIndex(runs: StressRun[], m: MetricDef): number {
  if (m.dir === "none" || runs.length < 2) return -1;
  let bi = 0;
  for (let i = 1; i < runs.length; i++) {
    const cur = m.get(runs[i].summary);
    const best = m.get(runs[bi].summary);
    if (m.dir === "high" ? cur > best : cur < best) bi = i;
  }
  // Don't crown a winner when every column ties.
  const allEqual = runs.every((r) => m.get(r.summary) === m.get(runs[0].summary));
  return allEqual ? -1 : bi;
}

function timeAgo(at: string | number): string {
  const ms = typeof at === "number" ? at : Date.parse(at);
  if (!Number.isFinite(ms)) return "";
  const s = Math.max(0, Math.round((Date.now() - ms) / 1000));
  if (s < 60) return `${s}s ago`;
  const mins = Math.round(s / 60);
  if (mins < 60) return `${mins}m ago`;
  const h = Math.round(mins / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

function downloadRunsJson(runs: StressRun[]) {
  const appId = runs[0]?.app_id ?? "endpoint";
  const blob = new Blob([JSON.stringify({ app_id: appId, exported_at: new Date().toISOString(), runs }, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `stress-${appId}-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function SavedRunsComparison({
  runs,
  onRemove,
  onClear,
  pathname,
  pinnedIds,
}: {
  runs: StressRun[];
  onRemove: (id: string) => void;
  onClear: () => void;
  pathname: string;
  pinnedIds: string[] | null;
}) {
  // Which runs are excluded from the comparison view + share link. Seeded from a
  // shared link's ?runs= (everything not pinned starts excluded); otherwise all
  // are included. Runs added after mount default to included.
  const [excluded, setExcluded] = useState<Set<string>>(() => {
    if (!pinnedIds) return new Set();
    const pin = new Set(pinnedIds);
    return new Set(runs.filter((r) => !pin.has(r.id)).map((r) => r.id));
  });
  const [copied, setCopied] = useState(false);

  const included = runs.filter((r) => !excluded.has(r.id));
  const toggle = (id: string) =>
    setExcluded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  function copyLink() {
    const ids = included.map((r) => r.id);
    const origin = typeof window !== "undefined" ? window.location.origin : "";
    const qs = new URLSearchParams({ tab: "stress" });
    if (ids.length) qs.set("runs", ids.join(","));
    const url = `${origin}${pathname}?${qs.toString()}`;
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      void navigator.clipboard.writeText(url).then(() => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 2000);
      });
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between sm:gap-3 sm:space-y-0">
        <div className="flex min-w-0 flex-col gap-0.5">
          <CardTitle className="text-sm font-medium">Saved runs · compare</CardTitle>
          <CardDescription className="text-xs">
            Completed runs are saved server-side. Pick which to compare below, then copy a link to share the exact
            comparison with anyone who can view this endpoint. Best value per row is highlighted.
          </CardDescription>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button variant="outline" size="xs" onClick={copyLink} disabled={included.length === 0}>
            {copied ? <Check className="h-3 w-3" /> : <Link2 className="h-3 w-3" />}
            {copied ? "Copied" : "Copy link"}
          </Button>
          <Button variant="outline" size="xs" onClick={() => downloadRunsJson(included)} disabled={included.length === 0}>
            <Download className="h-3 w-3" /> JSON
          </Button>
          <Button
            variant="outline"
            size="xs"
            onClick={onClear}
            className="text-muted-foreground hover:text-destructive"
          >
            <Trash2 className="h-3 w-3" /> Clear all
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 px-0 pb-0">
        {/* run picker — toggle which saved runs are in the comparison */}
        <div className="flex flex-wrap gap-1.5 px-6">
          {runs.map((r) => {
            const on = !excluded.has(r.id);
            return (
              <button
                key={r.id}
                type="button"
                onClick={() => toggle(r.id)}
                title={`${r.model || "(default)"} · in ${r.input_len}/out ${r.output_len} · n ${r.num_prompts} · c ${r.concurrency} · ${timeAgo(r.created_at)}`}
                className={`flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-[10px] transition-colors ${
                  on
                    ? "border-primary/40 bg-primary/10 text-foreground"
                    : "border-border bg-transparent text-muted-foreground opacity-60 hover:opacity-100"
                }`}
              >
                <span className="max-w-[140px] truncate">{r.model || "(default)"}</span>
                <span className="text-muted-foreground">c{r.concurrency}</span>
              </button>
            );
          })}
        </div>

        {included.length === 0 ? (
          <p className="px-6 pb-4 text-xs text-muted-foreground">No runs selected — pick at least one above.</p>
        ) : (
          <div className="overflow-x-auto scrollbar-thin">
            <table className="w-full border-separate border-spacing-0 text-sm">
              <thead>
                <tr>
                  <th className="sticky left-0 z-10 min-w-[180px] border-y border-border bg-card px-4 py-2 text-left text-xs font-medium text-muted-foreground">
                    metric
                  </th>
                  {included.map((r) => (
                    <th
                      key={r.id}
                      className="min-w-[150px] border-y border-l border-border bg-card px-3 py-2 text-left align-top"
                    >
                      <div className="flex items-start justify-between gap-1">
                        <div className="flex flex-col gap-0.5">
                          <span className="max-w-[160px] truncate font-mono text-xs text-foreground" title={r.model}>
                            {r.model || "(default)"}
                          </span>
                          <span className="font-mono text-[10px] text-muted-foreground">
                            in {r.input_len} · out {r.output_len}
                          </span>
                          <span className="font-mono text-[10px] text-muted-foreground">
                            n {r.num_prompts} · c {r.concurrency} · {timeAgo(r.created_at)}
                          </span>
                        </div>
                        <button
                          type="button"
                          onClick={() => onRemove(r.id)}
                          aria-label="Delete run"
                          title="Delete this run permanently"
                          className="shrink-0 text-muted-foreground hover:text-destructive"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {COMPARE_METRICS.map((m) => {
                  const bi = bestIndex(included, m);
                  return (
                    <tr key={m.label}>
                      <td className="sticky left-0 z-10 border-b border-border bg-card px-4 py-1.5 text-muted-foreground">
                        {m.label}
                      </td>
                      {included.map((r, i) => {
                        const danger = m.label === "Failed" && m.get(r.summary) > 0;
                        const best = i === bi;
                        return (
                          <td
                            key={r.id}
                            className={`border-b border-l border-border px-3 py-1.5 text-right font-mono tabular-nums ${
                              best
                                ? "bg-status-active/10 font-semibold text-status-active"
                                : danger
                                  ? "text-destructive"
                                  : "text-foreground"
                            }`}
                          >
                            {m.fmt(m.get(r.summary))}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function StressLog({ lines }: { lines: string[] }) {
  const ref = useRef<HTMLPreElement | null>(null);
  const atBottom = useRef(true);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const dist = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (atBottom.current) el.scrollTop = el.scrollHeight;
    atBottom.current = dist < 40;
  }, [lines]);
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>Request log</span>
        <span className="font-mono">{lines.length} completed</span>
      </div>
      <pre
        ref={(el) => { ref.current = el; }}
        className="terminal-block max-h-72 w-full overflow-y-auto whitespace-pre-wrap break-words rounded-md border border-border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-200 scrollbar-thin"
      >
        {lines.map((l, i) => (
          <div key={i} className={l.includes("FAIL") ? "text-red-400" : undefined}>{l}</div>
        ))}
      </pre>
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
