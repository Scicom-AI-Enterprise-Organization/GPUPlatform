"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Check, Download, Loader2, Pencil, RotateCcw, Trash2, X, XCircle } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { gateway } from "@/lib/gateway";
import { cn } from "@/lib/utils";
import type { TrainingEpoch, TrainingFile, TrainingGpu, TrainingGpuSample, TrainingRunRecord, TrainingStep } from "@/lib/types";

const STATUS_STYLES: Record<string, string> = {
  queued: "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400",
  running: "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400",
  done: "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
  failed: "border-destructive/40 bg-destructive/10 text-destructive",
  cancelled: "border-border bg-muted text-muted-foreground",
};

function fmt(v: number | null | undefined, digits = 2): string {
  return v == null ? "—" : v.toFixed(digits);
}

export function TrainingDetail({ initial }: { initial: TrainingRunRecord }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [run, setRun] = useState<TrainingRunRecord>(initial);
  const [busy, setBusy] = useState(false);
  const terminal = ["done", "failed", "cancelled"].includes(run.status);

  // Tab reflected in the URL (?tab=…) so it's deep-linkable + survives refresh.
  const tab = searchParams.get("tab") || "metrics";
  const onTab = (v: string) => router.replace(`${pathname}?tab=${v}`, { scroll: false });

  // Live per-GPU utilisation (only the run's GPUs) — poll while running.
  const [gpus, setGpus] = useState<TrainingGpu[]>([]);
  useEffect(() => {
    if (terminal) {
      setGpus([]);
      return;
    }
    let cancelled = false;
    const poll = async () => {
      try {
        const r = await gateway.getTrainingGpu(run.id);
        if (!cancelled) setGpus(r.gpus || []);
      } catch {
        /* keep last */
      }
    };
    poll();
    const t = setInterval(poll, 2500);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [run.id, terminal]);

  // Poll the record while queued/running so status + metrics refresh.
  useEffect(() => {
    if (terminal) return;
    const t = setInterval(async () => {
      try {
        setRun(await gateway.getTrainingRun(run.id));
      } catch {
        /* keep last */
      }
    }, 5000);
    return () => clearInterval(t);
  }, [run.id, terminal]);

  // Single log stream for the whole page: feeds the Logs tab AND the live loss
  // curve. We parse @@STEP lines into points as they arrive; once the run
  // finalizes, result_json.steps is the authoritative set (see liveSteps use).
  const [lines, setLines] = useState<string[]>([]);
  const [liveSteps, setLiveSteps] = useState<TrainingStep[]>([]);
  useEffect(() => {
    setLines([]);
    setLiveSteps([]);
    const es = new EventSource(gateway.trainingLogsStreamUrl(run.id));
    es.onmessage = (ev) => {
      setLines((p) => [...p, ev.data]);
      // @@STEP can be prefixed by a tqdm progress bar (\r, no newline) on the
      // same captured line, so match it anywhere — not just at the start.
      const i = typeof ev.data === "string" ? ev.data.indexOf("@@STEP ") : -1;
      if (i >= 0) {
        try {
          const pt = JSON.parse(ev.data.slice(i + "@@STEP ".length)) as TrainingStep;
          if (typeof pt.step === "number") setLiveSteps((p) => [...p, pt]);
        } catch {
          /* ignore malformed */
        }
      }
    };
    es.addEventListener("end", () => es.close());
    return () => es.close();
  }, [run.id]);

  const epochs = run.result_json?.epochs ?? [];
  // Persisted steps win once the run finalizes; until then the live stream feeds the curve.
  const persistedSteps = run.result_json?.steps ?? [];
  const steps = persistedSteps.length ? persistedSteps : liveSteps;
  const best = run.result_json?.best;
  const artifact = run.result_json?.artifact;
  const trials = run.result_json?.trials ?? [];
  const isSweep = trials.length > 0;
  const metricLabel = run.task_type === "tts"
    ? "loss"
    : String((run.config_json?.eval_metric as string) || "wer").toUpperCase();
  const sortedTrials = [...trials].sort((a, b) => {
    if (a.metric == null) return 1;
    if (b.metric == null) return -1;
    return a.metric - b.metric;
  });

  async function onTerminate() {
    if (!confirm("Terminate this run? The pod (if any) is torn down.")) return;
    setBusy(true);
    try {
      setRun(await gateway.terminateTrainingRun(run.id));
      toast.success("Terminated", { duration: 3000 });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e), { duration: 5000 });
    } finally {
      setBusy(false);
    }
  }

  async function onDelete() {
    if (!confirm("Delete this run? This removes its record.")) return;
    setBusy(true);
    try {
      await gateway.deleteTrainingRun(run.id);
      toast.success("Deleted", { duration: 3000 });
      router.push("/autotrain");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e), { duration: 5000 });
      setBusy(false);
    }
  }

  const [editingName, setEditingName] = useState(false);
  const [nameDraft, setNameDraft] = useState(run.name);
  async function onRename() {
    const n = nameDraft.trim();
    if (!n || n === run.name) { setEditingName(false); return; }
    try {
      setRun(await gateway.renameTrainingRun(run.id, n));
      setEditingName(false);
      toast.success("Renamed", { duration: 2000 });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e), { duration: 5000 });
    }
  }

  async function onRestart() {
    if (!confirm("Restart? Launches a fresh run with this run's exact config.")) return;
    setBusy(true);
    try {
      const created = await gateway.restartTrainingRun(run.id);
      toast.success(`Restarted → ${created.id}`, { duration: 3000 });
      router.push(`/autotrain/${encodeURIComponent(created.id)}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e), { duration: 5000 });
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {editingName ? (
              <span className="flex items-center gap-1">
                <Input
                  value={nameDraft}
                  onChange={(e) => setNameDraft(e.target.value)}
                  autoFocus
                  className="h-8 w-72 text-lg font-semibold"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") onRename();
                    if (e.key === "Escape") setEditingName(false);
                  }}
                />
                <Button size="icon" variant="ghost" className="h-7 w-7" onClick={onRename} title="Save">
                  <Check className="h-4 w-4" />
                </Button>
                <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => setEditingName(false)} title="Cancel">
                  <X className="h-4 w-4" />
                </Button>
              </span>
            ) : (
              <>
                <h1 className="truncate text-2xl font-semibold tracking-tight">{run.name}</h1>
                <button
                  type="button"
                  onClick={() => { setNameDraft(run.name); setEditingName(true); }}
                  className="text-muted-foreground hover:text-foreground"
                  title="Rename"
                >
                  <Pencil className="h-4 w-4" />
                </button>
              </>
            )}
            <Badge variant="outline" className={STATUS_STYLES[run.status] ?? ""}>{run.status}</Badge>
          </div>
          <p className="mt-1 font-mono text-xs text-muted-foreground">
            {run.base_model} · {run.id}
            {run.cost_per_hr != null ? ` · $${run.cost_per_hr}/hr` : ""}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {!terminal && (
            <Button variant="outline" size="sm" onClick={onTerminate} disabled={busy} className="text-destructive">
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <XCircle className="h-4 w-4" />} Terminate
            </Button>
          )}
          <Button variant="outline" size="sm" onClick={onRestart} disabled={busy}>
            <RotateCcw className="h-4 w-4" /> Restart
          </Button>
          <Button variant="outline" size="sm" onClick={onDelete} disabled={busy}>
            <Trash2 className="h-4 w-4" /> Delete
          </Button>
        </div>
      </div>

      {run.error_text && run.status === "failed" && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <pre className="whitespace-pre-wrap break-words font-mono text-xs">{run.error_text}</pre>
        </div>
      )}

      {best && !isSweep && (
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm">Best checkpoint</CardTitle></CardHeader>
          <CardContent className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
            <Stat label="Best epoch" value={String(best.epoch ?? "—")} />
            <Stat label="WER" value={fmt(best.wer)} />
            <Stat label="CER" value={fmt(best.cer)} />
            <Stat label="Eval loss" value={fmt(best.eval_loss, 4)} />
            {run.result_json?.stopped_early && <Stat label="Stopped early" value="yes (patience)" />}
            {artifact?.s3_uri && <Stat label="Artifact" value={artifact.s3_uri} mono />}
            {artifact?.hf_repo && <Stat label="HF" value={artifact.hf_repo} mono />}
          </CardContent>
        </Card>
      )}

      {isSweep && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">
              Sweep · {trials.length} trial{trials.length === 1 ? "" : "s"} · best by {metricLabel} (lower is better)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto rounded-md border border-border">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 text-xs text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2 text-left">Trial</th>
                    <th className="px-3 py-2 text-left">Params</th>
                    <th className="px-3 py-2 text-right">{metricLabel}</th>
                    <th className="px-3 py-2 text-left">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedTrials.map((t, i) => {
                    const isBest = i === 0 && t.metric != null;
                    return (
                      <tr key={t.trial} className={`border-t border-border ${isBest ? "bg-emerald-500/10" : ""}`}>
                        <td className="px-3 py-2 font-mono">{t.trial}{isBest ? " ★" : ""}</td>
                        <td className="px-3 py-2 font-mono text-xs">
                          {Object.entries(t.params || {}).map(([k, v]) => `${k}=${v}`).join(", ") || "—"}
                        </td>
                        <td className="px-3 py-2 text-right font-mono">{t.metric == null ? "—" : fmt(t.metric, 3)}</td>
                        <td className="px-3 py-2">
                          <Badge variant="outline" className={`text-[10px] ${STATUS_STYLES[t.status ?? ""] ?? ""}`}>
                            {t.status ?? "—"}
                          </Badge>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              Per-trial checkpoints are under <span className="font-mono">…/trials/&lt;trial&gt;/</span> — see the Files tab.
            </p>
          </CardContent>
        </Card>
      )}

      <Tabs value={tab} onValueChange={onTab} className="!block">
        <TabsList>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="logs">Logs</TabsTrigger>
          <TabsTrigger value="files">Files</TabsTrigger>
          <TabsTrigger value="config">Config</TabsTrigger>
        </TabsList>

        <TabsContent value="metrics" className="mt-4 !flex-none space-y-4">
          <LossCurve steps={steps} epochs={epochs} live={!terminal} />
          <EvalCurve epochs={epochs} />
          <GpuCard gpus={gpus} samples={run.result_json?.gpu_samples ?? []} running={!terminal} />
          {epochs.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No per-epoch metrics yet. They appear here as each epoch finishes evaluating.
            </p>
          ) : (
            <div className="overflow-x-auto rounded-md border border-border">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 text-xs text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2 text-left">Epoch</th>
                    <th className="px-3 py-2 text-right">WER</th>
                    <th className="px-3 py-2 text-right">CER</th>
                    <th className="px-3 py-2 text-right">Eval loss</th>
                    <th className="px-3 py-2 text-right">Train loss</th>
                  </tr>
                </thead>
                <tbody>
                  {epochs.map((e, i) => {
                    const isBest = best?.epoch != null && Math.round(e.epoch) === best.epoch;
                    return (
                      <tr key={i} className={`border-t border-border ${isBest ? "bg-emerald-500/5" : ""}`}>
                        <td className="px-3 py-2 font-mono">{e.epoch}</td>
                        <td className="px-3 py-2 text-right font-mono">{fmt(e.wer)}</td>
                        <td className="px-3 py-2 text-right font-mono">{fmt(e.cer)}</td>
                        <td className="px-3 py-2 text-right font-mono">{fmt(e.eval_loss, 4)}</td>
                        <td className="px-3 py-2 text-right font-mono">{fmt(e.train_loss, 4)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </TabsContent>

        <TabsContent value="logs" className="mt-4 !flex-none">
          <LogsTab lines={lines} status={run.status} />
        </TabsContent>

        <TabsContent value="files" className="mt-4 !flex-none">
          <FilesTab run={run} />
        </TabsContent>

        <TabsContent value="config" className="mt-4 !flex-none">
          <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 px-4 py-3 font-mono text-xs leading-relaxed">
            {JSON.stringify(run.config_json, null, 2)}
          </pre>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function Stat({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className={`text-sm ${mono ? "font-mono break-all" : "font-medium"}`}>{value}</div>
    </div>
  );
}

// Live per-GPU telemetry graph (util %, memory %, temperature °C over time) for
// the run's GPUs only. Self-accumulates a rolling history from the poll prop.
const GPU_HIST_CAP = 150; // ~6 min at the 2.5s poll
type GpuSample = { i: number; util: number; mem: number; memGiB: number; temp: number };

function GpuCard({ gpus, samples, running }: { gpus: TrainingGpu[]; samples: TrainingGpuSample[]; running: boolean }) {
  const [hist, setHist] = useState<Record<number, GpuSample[]>>({});
  const tick = useRef(0);
  useEffect(() => {
    if (!running || !gpus.length) return;
    tick.current += 1;
    const i = tick.current;
    setHist((prev) => {
      const next: Record<number, GpuSample[]> = { ...prev };
      for (const g of gpus) {
        const mem = g.mem_total > 0 ? (g.mem_used / g.mem_total) * 100 : 0;
        next[g.index] = (next[g.index] ?? [])
          .concat({ i, util: g.util, mem, memGiB: g.mem_used / 1024, temp: g.temp })
          .slice(-GPU_HIST_CAP);
      }
      return next;
    });
  }, [gpus, running]);

  // Finished run → render the persisted gpu_samples (the live poll returns
  // nothing once a run ends). "current" = the last sample for the value chips.
  const fromSamples = !running && samples.length > 0;
  const series: Record<number, GpuSample[]> = {};
  let current: TrainingGpu[] = gpus;
  if (fromSamples) {
    for (const s of samples) {
      for (const g of s.gpus) {
        const mem = g.mem_total > 0 ? (g.mem_used / g.mem_total) * 100 : 0;
        (series[g.index] ??= []).push({ i: s.t, util: g.util, mem, memGiB: g.mem_used / 1024, temp: g.temp });
      }
    }
    current = samples[samples.length - 1].gpus;
  } else {
    Object.assign(series, hist);
  }

  if (!running && !current.length) return null;
  const sorted = [...current].sort((a, b) => a.index - b.index);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <span className={cn("h-1.5 w-1.5 rounded-full", running ? "animate-pulse bg-emerald-500" : "bg-muted-foreground")} />
          GPU telemetry · {running ? "live" : "final"}
          {sorted.length > 0 && (
            <span className="text-[11px] font-normal text-muted-foreground">
              {sorted.length} GPU{sorted.length === 1 ? "" : "s"}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {sorted.length === 0 && (
          <p className="text-xs text-muted-foreground">Waiting for GPU telemetry…</p>
        )}
        <div className="grid gap-3 lg:grid-cols-2">
          {sorted.map((g) => {
            const mem = g.mem_total > 0 ? (g.mem_used / g.mem_total) * 100 : 0;
            const data = series[g.index] ?? [];
            return (
              <div key={g.index} className="rounded-lg border border-border p-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate font-mono text-xs font-medium">
                    #{g.index} {g.name.replace(/^NVIDIA\s+/, "")}
                  </span>
                </div>
                <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px]">
                  <span className="text-emerald-600 dark:text-emerald-400">{g.util.toFixed(0)}% util</span>
                  <span className="text-sky-600 dark:text-sky-400">
                    {mem.toFixed(0)}% mem · {(g.mem_used / 1024).toFixed(1)}/{(g.mem_total / 1024).toFixed(1)} GiB
                  </span>
                  <span className="text-amber-600 dark:text-amber-400">{g.temp.toFixed(0)}°C</span>
                </div>
                <div className="mt-2 h-28 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border" />
                      <XAxis dataKey="i" hide type="number" domain={["dataMin", "dataMax"]} />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} stroke="currentColor"
                        className="text-muted-foreground" width={32} />
                      <RTooltip
                        contentStyle={{ fontSize: 11, borderRadius: 8 }}
                        labelFormatter={() => ""}
                        formatter={(v, n) => {
                          const num = Number(v);
                          return n === "temp" ? [`${num.toFixed(0)}°C`, "temp"]
                            : [`${num.toFixed(0)}%`, n === "util" ? "util" : "mem"];
                        }}
                      />
                      <Line type="monotone" dataKey="util" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} />
                      <Line type="monotone" dataKey="mem" stroke="#0ea5e9" strokeWidth={2} dot={false} isAnimationActive={false} />
                      <Line type="monotone" dataKey="temp" stroke="#f59e0b" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            );
          })}
        </div>
        <p className="text-[10px] text-muted-foreground">
          <span className="text-emerald-600 dark:text-emerald-400">■</span> util % ·{" "}
          <span className="text-sky-600 dark:text-sky-400">■</span> memory % ·{" "}
          <span className="text-amber-600 dark:text-amber-400">■</span> temp °C — this run&apos;s GPUs, refreshed every 2.5s.
        </p>
      </CardContent>
    </Card>
  );
}

// Live training-loss curve. Points come from the page-level @@STEP stream
// while running, or result_json.steps once finalized.
function LossCurve({ steps, epochs, live }: { steps: TrainingStep[]; epochs: TrainingEpoch[]; live: boolean }) {
  const data = steps
    .filter((s) => typeof s.loss === "number")
    .map((s) => ({ step: s.step, loss: s.loss as number, eval_loss: null as number | null, epoch: s.epoch ?? undefined }));
  // Per-epoch eval_loss is sparse; pin each to the last training step within that
  // epoch so it overlays the per-step train loss on the same step axis.
  for (const e of epochs) {
    if (typeof e.eval_loss !== "number") continue;
    let idx = -1;
    for (let i = 0; i < steps.length; i++) {
      if ((steps[i].epoch ?? 0) <= (e.epoch ?? 0) + 1e-6) idx = i;
    }
    if (idx < 0) idx = data.length - 1;
    if (data[idx]) data[idx].eval_loss = e.eval_loss;
  }
  const hasEval = data.some((d) => d.eval_loss != null);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          Loss
          {live && (
            <span className="inline-flex items-center gap-1 text-[11px] font-normal text-muted-foreground">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" /> live
            </span>
          )}
          {data.length > 0 && (
            <span className="text-[11px] font-normal text-muted-foreground">· {data.length} pts</span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {data.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">
            No loss points yet — they stream in every{" "}
            <span className="font-mono">logging_steps</span> as training runs.
          </p>
        ) : (
          <>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 8, right: 16, left: 4, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border" />
                  <XAxis
                    dataKey="step" type="number" domain={["dataMin", "dataMax"]}
                    tick={{ fontSize: 11 }} stroke="currentColor" className="text-muted-foreground"
                    label={{ value: "step", position: "insideBottomRight", offset: -4, fontSize: 11 }}
                  />
                  <YAxis
                    tick={{ fontSize: 11 }} stroke="currentColor" className="text-muted-foreground"
                    width={48} domain={["auto", "auto"]}
                    tickFormatter={(v: number) => v.toFixed(2)}
                  />
                  <RTooltip
                    contentStyle={{ fontSize: 12, borderRadius: 8 }}
                    formatter={(v, n) => [Number(v).toFixed(4), n === "eval_loss" ? "eval loss" : "train loss"]}
                    labelFormatter={(s) => `step ${s}`}
                  />
                  <Line type="monotone" dataKey="loss" name="train loss" stroke="#6366f1" strokeWidth={2}
                    dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="eval_loss" name="eval loss" stroke="#f59e0b" strokeWidth={2}
                    connectNulls dot={{ r: 3 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-1 text-[10px] text-muted-foreground">
              <span className="text-indigo-500">■</span> train loss (per step)
              {hasEval && <> · <span className="text-amber-500">■</span> eval loss (per epoch)</>}
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}

// WER / CER per epoch (lower is better). Hidden until there's eval data.
function EvalCurve({ epochs }: { epochs: TrainingEpoch[] }) {
  const data = epochs
    .filter((e) => typeof e.wer === "number" || typeof e.cer === "number")
    .map((e) => ({ epoch: e.epoch, wer: e.wer ?? null, cer: e.cer ?? null }));
  if (data.length === 0) return null;
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Eval metrics · WER / CER <span className="text-[11px] font-normal text-muted-foreground">(per epoch, lower is better)</span></CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 16, left: 4, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border" />
              <XAxis dataKey="epoch" type="number" domain={["dataMin", "dataMax"]} allowDecimals={false}
                tick={{ fontSize: 11 }} stroke="currentColor" className="text-muted-foreground"
                label={{ value: "epoch", position: "insideBottomRight", offset: -4, fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} stroke="currentColor" className="text-muted-foreground"
                width={44} domain={["auto", "auto"]} tickFormatter={(v: number) => `${v.toFixed(0)}%`} />
              <RTooltip contentStyle={{ fontSize: 12, borderRadius: 8 }}
                formatter={(v, n) => [`${Number(v).toFixed(2)}%`, String(n).toUpperCase()]}
                labelFormatter={(e) => `epoch ${e}`} />
              <Line type="monotone" dataKey="wer" name="wer" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} connectNulls isAnimationActive={false} />
              <Line type="monotone" dataKey="cer" name="cer" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} connectNulls isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-1 text-[10px] text-muted-foreground">
          <span className="text-red-500">■</span> WER · <span className="text-violet-500">■</span> CER
        </p>
      </CardContent>
    </Card>
  );
}

function LogsTab({ lines, status }: { lines: string[]; status: string }) {
  const endRef = useRef<HTMLDivElement>(null);
  const terminal = ["done", "failed", "cancelled"].includes(status);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [lines]);

  return (
    <div className="terminal-block h-[55vh] overflow-y-auto rounded-md border border-border bg-zinc-950 p-3 font-mono text-xs leading-relaxed text-zinc-200">
      {lines.length === 0 ? (
        <div className="text-zinc-500">
          {status === "queued" ? "Queued — waiting for the runner…" : "Waiting for output…"}
        </div>
      ) : (
        lines.map((l, i) => (
          <div key={i} className={
            l.startsWith("@@") ? "text-sky-300"
              : l.startsWith("[gateway]") ? "text-emerald-300"
              : "text-zinc-200"
          }>{l}</div>
        ))
      )}
      <div ref={endRef} />
      {terminal && lines.length === 0 && (
        <div className="text-zinc-500">No logs (run {status}).</div>
      )}
    </div>
  );
}

function FilesTab({ run }: { run: TrainingRunRecord }) {
  const [files, setFiles] = useState<TrainingFile[]>([]);
  const [loading, setLoading] = useState(true);
  const load = useCallback(async () => {
    setLoading(true);
    try {
      setFiles(await gateway.listTrainingFiles(run.id));
    } catch {
      setFiles([]);
    } finally {
      setLoading(false);
    }
  }, [run.id]);
  useEffect(() => { load(); }, [load]);

  if (loading) return <p className="text-sm text-muted-foreground">Loading files…</p>;
  if (files.length === 0)
    return <p className="text-sm text-muted-foreground">No files yet — artifacts upload when the run finishes.</p>;
  return (
    <ul className="divide-y divide-border rounded-md border border-border">
      {files.map((f) => (
        <li key={f.name} className="flex items-center justify-between gap-4 px-4 py-2 text-sm">
          <span className="truncate font-mono text-xs">{f.name}</span>
          <a href={f.download_url} target="_blank" rel="noreferrer"
            className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
            <Download className="h-3.5 w-3.5" /> {(f.size / 1024).toFixed(0)} KB
          </a>
        </li>
      ))}
    </ul>
  );
}
