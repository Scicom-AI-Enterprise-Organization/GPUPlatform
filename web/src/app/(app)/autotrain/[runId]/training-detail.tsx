"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { AudioLines, Check, ChevronDown, Copy, Download, Loader2, Pencil, RotateCcw, Trash2, X, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
import { JsonView } from "@/components/json-view";
import { cn } from "@/lib/utils";
import type { TrainingEpoch, TrainingFile, TrainingGpu, TrainingGpuSample, TrainingRunRecord, TrainingStep, TrainingTrial } from "@/lib/types";

// Distinct colours for per-trial sweep loss curves.
const TRIAL_COLORS = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9", "#ec4899", "#84cc16"];

function trialLabel(i: number, trials: TrainingTrial[]): string {
  const p = trials.find((x) => x.trial === i)?.params;
  if (!p) return `t${i}`;
  const parts = Object.entries(p).map(([k, v]) =>
    k === "learning_rate" ? `lr ${v}`
    : k === "precision" ? String(v)
    : k === "batch_size" ? `bs ${v}`
    : k === "grad_accum" ? `ga ${v}`
    : k === "max_epochs" ? `ep ${v}`
    : k === "weight_decay" ? `wd ${v}`
    : k === "lora_r" ? `r ${v}`
    : k === "lora_alpha" ? `α ${v}`
    : k === "freeze_encoder" ? (String(v) === "on" ? "frozen" : "full")
    : k === "augment" ? `aug ${v}`
    : `${k}=${v}`,
  );
  return `t${i}: ${parts.join(" · ")}`;
}

const STATUS_STYLES: Record<string, string> = {
  queued: "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400",
  pending: "border-border bg-muted text-muted-foreground",
  running: "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400",
  done: "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
  failed: "border-destructive/40 bg-destructive/10 text-destructive",
  cancelled: "border-border bg-muted text-muted-foreground",
};

// A run is a sweep if its config carries a non-empty sweep grid.
export function isSweepConfig(config: Record<string, unknown> | undefined | null): boolean {
  const sweep = (config?.sweep ?? null) as Record<string, unknown> | null;
  return !!sweep && Object.values(sweep).some((v) => Array.isArray(v) && v.length > 0);
}

// Single-run vs sweep badge (placed beside the status badge).
export function RunKindBadge({ sweep, trials }: { sweep: boolean; trials?: number }) {
  return (
    <Badge variant="outline" className={
      sweep ? "border-violet-500/40 bg-violet-500/10 text-violet-600 dark:text-violet-300"
            : "border-sky-500/40 bg-sky-500/10 text-sky-600 dark:text-sky-300"
    }>
      {sweep ? `sweep${trials ? ` · ${trials} trials` : ""}` : "single run"}
    </Badge>
  );
}

function fmt(v: number | null | undefined, digits = 2): string {
  return v == null ? "—" : v.toFixed(digits);
}

// Drives the shared confirmation dialog (replaces window.confirm).
type ConfirmOpts = {
  title: string;
  description: string;
  confirmLabel: string;
  busyLabel: string;
  destructive?: boolean;
  run: () => Promise<void>;
};

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
          // Sweep: the trial index comes from the "[trial N]" prefix on the line.
          const tm = (ev.data as string).match(/\[trial (\d+)\]/);
          if (tm && pt.trial == null) pt.trial = Number(tm[1]);
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
  const isSweep = isSweepConfig(run.config_json) || trials.length > 0;
  // Try-it playground: finished ASR run on a VM (inference runs on that VM).
  const canTryIt =
    run.status === "done" &&
    (run.task_type ?? "asr") === "asr" &&
    run.provider_kind === "vm" &&
    !!run.result_json?.artifact?.s3_uri;
  const metricLabel = run.task_type === "tts"
    ? "loss"
    : String((run.config_json?.eval_metric as string) || "wer").toUpperCase();
  const sortedTrials = [...trials].sort((a, b) => {
    if (a.metric == null) return 1;
    if (b.metric == null) return -1;
    return a.metric - b.metric;
  });

  // A single confirmation dialog drives terminate/delete/restart (no native
  // window.confirm). Each action sets `confirmOpts`; the dialog runs `.run()`.
  const [confirmOpts, setConfirmOpts] = useState<ConfirmOpts | null>(null);
  const [confirmError, setConfirmError] = useState<string | null>(null);
  async function runConfirm() {
    if (!confirmOpts) return;
    setBusy(true);
    setConfirmError(null);
    try {
      await confirmOpts.run();
      setConfirmOpts(null);
    } catch (e) {
      setConfirmError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function onTerminate() {
    setConfirmError(null);
    setConfirmOpts({
      title: "Terminate this run?",
      description: "Stops training now and tears down the pod (if any). Metrics collected so far are kept.",
      confirmLabel: "Terminate",
      busyLabel: "Terminating…",
      destructive: true,
      run: async () => {
        setRun(await gateway.terminateTrainingRun(run.id));
      },
    });
  }

  function onDelete() {
    setConfirmError(null);
    setConfirmOpts({
      title: `Delete ${run.name}?`,
      description: "Removes the training-run record. S3 artifacts are kept. If a pod is still alive, terminate it first.",
      confirmLabel: "Delete",
      busyLabel: "Deleting…",
      destructive: true,
      run: async () => {
        await gateway.deleteTrainingRun(run.id);
        router.push("/autotrain");
      },
    });
  }

  const [editingName, setEditingName] = useState(false);
  const [nameDraft, setNameDraft] = useState(run.name);
  const [renameError, setRenameError] = useState<string | null>(null);
  async function onRename() {
    const n = nameDraft.trim();
    if (!n || n === run.name) { setEditingName(false); setRenameError(null); return; }
    setRenameError(null);
    try {
      setRun(await gateway.renameTrainingRun(run.id, n));
      setEditingName(false);
    } catch (e) {
      setRenameError(e instanceof Error ? e.message : String(e));
    }
  }

  function onDuplicateRun() {
    setConfirmOpts({
      title: "Duplicate & run now?",
      description: "Launches a new run with this run's exact config (same dataset, model, GPUs and hyperparameters). The original is untouched.",
      confirmLabel: "Duplicate & run",
      busyLabel: "Launching…",
      run: async () => {
        const created = await gateway.restartTrainingRun(run.id);
        router.push(`/autotrain/${encodeURIComponent(created.id)}`);
      },
    });
  }

  // Open the create form pre-filled from this run's config, without launching.
  function onEditAsNew() {
    router.push(`/autotrain/new?from=${encodeURIComponent(run.id)}`);
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {editingName ? (
              <span className="flex flex-col gap-1">
                <span className="flex items-center gap-1">
                  <Input
                    value={nameDraft}
                    onChange={(e) => setNameDraft(e.target.value)}
                    autoFocus
                    className="h-8 w-72 text-lg font-semibold"
                    onKeyDown={(e) => {
                      if (e.key === "Enter") onRename();
                      if (e.key === "Escape") { setEditingName(false); setRenameError(null); }
                    }}
                  />
                  <Button size="icon" variant="ghost" className="h-7 w-7" onClick={onRename} title="Save">
                    <Check className="h-4 w-4" />
                  </Button>
                  <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => { setEditingName(false); setRenameError(null); }} title="Cancel">
                    <X className="h-4 w-4" />
                  </Button>
                </span>
                {renameError && <span className="text-xs text-destructive">{renameError}</span>}
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
            <RunKindBadge sweep={isSweep} trials={isSweep ? trials.length : undefined} />
          </div>
          <p className="mt-1 font-mono text-xs text-muted-foreground">
            {run.base_model} · {run.id}
            {run.cost_per_hr != null ? ` · $${run.cost_per_hr}/hr` : ""}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" disabled={busy}>
                <Copy className="h-4 w-4" /> Duplicate <ChevronDown className="h-3.5 w-3.5 opacity-60" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onSelect={(e) => { e.preventDefault(); onDuplicateRun(); }}>
                <RotateCcw className="h-4 w-4" /> Duplicate &amp; run now
              </DropdownMenuItem>
              <DropdownMenuItem onSelect={(e) => { e.preventDefault(); onEditAsNew(); }}>
                <Pencil className="h-4 w-4" /> Edit as new…
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          {!terminal && (
            <Button variant="outline" size="sm" onClick={onTerminate} disabled={busy} className="text-destructive">
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <XCircle className="h-4 w-4" />} Terminate
            </Button>
          )}
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
          {canTryIt && <TabsTrigger value="tryit">Try it</TabsTrigger>}
        </TabsList>

        <TabsContent value="metrics" className="mt-4 !flex-none space-y-4">
          <LossCurve steps={steps} epochs={epochs} live={!terminal} sweep={isSweep} trials={trials} />
          <EvalCurve epochs={epochs} sweep={isSweep} trials={trials} />
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
                    {isSweep && <th className="px-3 py-2 text-left">Trial</th>}
                    <th className="px-3 py-2 text-left">Epoch</th>
                    <th className="px-3 py-2 text-right">WER</th>
                    <th className="px-3 py-2 text-right">CER</th>
                    <th className="px-3 py-2 text-right">Eval loss</th>
                    <th className="px-3 py-2 text-right">Train loss</th>
                  </tr>
                </thead>
                <tbody>
                  {(isSweep
                    ? [...epochs].sort((a, b) => (a.trial ?? 0) - (b.trial ?? 0) || a.epoch - b.epoch)
                    : epochs
                  ).map((e, i) => {
                    const isBest = best?.epoch != null && Math.round(e.epoch) === best.epoch;
                    const ti = e.trial;
                    return (
                      <tr key={i} className={`border-t border-border ${isBest ? "bg-emerald-500/5" : ""}`}>
                        {isSweep && (
                          <td className="px-3 py-2 font-mono text-xs">
                            {ti == null ? "—" : (
                              <span title={trialLabel(ti, trials)}>
                                <span style={{ color: TRIAL_COLORS[ti % TRIAL_COLORS.length] }}>■</span> {trialLabel(ti, trials)}
                              </span>
                            )}
                          </td>
                        )}
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

        <TabsContent value="config" className="mt-4 !flex-none space-y-4">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">Compute</CardTitle></CardHeader>
            <CardContent className="flex flex-wrap gap-x-8 gap-y-3 text-sm">
              <Stat
                label={run.provider_kind === "vm" ? "VM" : "Provider"}
                value={
                  run.provider_name
                    ? `${run.provider_name}${run.provider_kind ? ` (${run.provider_kind})` : ""}`
                    : run.provider_id || "—"
                }
              />
              <Stat
                label="GPU"
                value={run.gpu_type ? `${run.gpu_type}${run.gpu_count > 1 ? ` × ${run.gpu_count}` : ""}` : "—"}
              />
              {run.visible_devices && <Stat label="GPU ids" value={run.visible_devices} mono />}
              <Stat label="Storage" value={run.storage_name || run.storage_id || "—"} />
              <Stat label="Dataset" value={run.dataset_id} mono />
              {run.test_dataset_id && <Stat label="Test dataset" value={run.test_dataset_id} mono />}
              <Stat label="Base model" value={run.base_model} mono />
            </CardContent>
          </Card>
          <JsonView value={run.config_json} />
        </TabsContent>

        {canTryIt && (
          <TabsContent value="tryit" className="mt-4 !flex-none">
            <PlaygroundTab runId={run.id} visibleDevices={run.visible_devices ?? null} />
          </TabsContent>
        )}
      </Tabs>

      <Dialog
        open={!!confirmOpts}
        onOpenChange={(o) => {
          if (!busy && !o) {
            setConfirmOpts(null);
            setConfirmError(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{confirmOpts?.title}</DialogTitle>
            <DialogDescription>{confirmOpts?.description}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {confirmError && <p className="mr-auto text-sm text-destructive">{confirmError}</p>}
            <Button variant="outline" onClick={() => setConfirmOpts(null)} disabled={busy}>
              Cancel
            </Button>
            <Button
              variant={confirmOpts?.destructive ? "destructive" : "default"}
              onClick={runConfirm}
              disabled={busy}
            >
              {busy ? confirmOpts?.busyLabel : confirmOpts?.confirmLabel}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
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
function LossCurve({ steps, epochs, live, sweep, trials }: { steps: TrainingStep[]; epochs: TrainingEpoch[]; live: boolean; sweep: boolean; trials: TrainingTrial[] }) {
  // Sweep: one train-loss line per trial (steps carry a `trial` index), legended
  // by each trial's hyperparameters.
  if (sweep && steps.some((s) => s.trial != null)) {
    const idxs = [...new Set(steps.map((s) => s.trial).filter((t): t is number => t != null))].sort((a, b) => a - b);
    const byStep = new Map<number, Record<string, number>>();
    for (const s of steps) {
      if (typeof s.loss !== "number" || s.trial == null) continue;
      const row = byStep.get(s.step) ?? { step: s.step };
      row[`t${s.trial}`] = s.loss;
      byStep.set(s.step, row);
    }
    const data = [...byStep.values()].sort((a, b) => a.step - b.step);
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-sm">
            Training loss · per trial
            {live && (
              <span className="inline-flex items-center gap-1 text-[11px] font-normal text-muted-foreground">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" /> live
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {data.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">No loss points yet — trials stream in as they run.</p>
          ) : (
            <>
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data} margin={{ top: 8, right: 16, left: 4, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border" />
                    <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} tick={{ fontSize: 11 }}
                      stroke="currentColor" className="text-muted-foreground"
                      label={{ value: "step", position: "insideBottomRight", offset: -4, fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} stroke="currentColor" className="text-muted-foreground"
                      width={48} domain={["auto", "auto"]} tickFormatter={(v: number) => v.toFixed(2)} />
                    <RTooltip contentStyle={{ fontSize: 12, borderRadius: 8 }}
                      formatter={(v, n) => [Number(v).toFixed(4), trialLabel(Number(String(n).slice(1)), trials)]}
                      labelFormatter={(s) => `step ${s}`} />
                    {idxs.map((i) => (
                      <Line key={i} type="monotone" dataKey={`t${i}`} name={`t${i}`}
                        stroke={TRIAL_COLORS[i % TRIAL_COLORS.length]} strokeWidth={2} dot={false}
                        connectNulls isAnimationActive={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
                {idxs.map((i) => (
                  <span key={i}>
                    <span style={{ color: TRIAL_COLORS[i % TRIAL_COLORS.length] }}>■</span> {trialLabel(i, trials)}
                  </span>
                ))}
              </div>
            </>
          )}
        </CardContent>
      </Card>
    );
  }

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

// One per-trial metric chart (WER or CER): a line per trial, legended by params.
function PerTrialEvalChart({ epochs, metric, idxs, trials }: {
  epochs: TrainingEpoch[]; metric: "wer" | "cer"; idxs: number[]; trials: TrainingTrial[];
}) {
  const byEpoch = new Map<number, Record<string, number>>();
  for (const e of epochs) {
    const v = e[metric];
    if (typeof v !== "number" || e.trial == null) continue;
    const row = byEpoch.get(e.epoch) ?? { epoch: e.epoch };
    row[`t${e.trial}`] = v;
    byEpoch.set(e.epoch, row);
  }
  const data = [...byEpoch.values()].sort((a, b) => a.epoch - b.epoch);
  if (data.length === 0) return null;
  return (
    <div>
      <div className="mb-1 text-xs font-medium text-muted-foreground">{metric.toUpperCase()}</div>
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
              formatter={(v, n) => [`${Number(v).toFixed(2)}%`, trialLabel(Number(String(n).slice(1)), trials)]}
              labelFormatter={(e) => `epoch ${e}`} />
            {idxs.map((i) => (
              <Line key={i} type="monotone" dataKey={`t${i}`} name={`t${i}`}
                stroke={TRIAL_COLORS[i % TRIAL_COLORS.length]} strokeWidth={2} dot={{ r: 3 }}
                connectNulls isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// WER / CER per epoch (lower is better). Hidden until there's eval data. Sweeps
// split each metric into one line per trial, legended like the loss curve.
function EvalCurve({ epochs, sweep, trials }: { epochs: TrainingEpoch[]; sweep: boolean; trials: TrainingTrial[] }) {
  const hasData = epochs.some((e) => typeof e.wer === "number" || typeof e.cer === "number");
  if (!hasData) return null;

  if (sweep && epochs.some((e) => e.trial != null)) {
    const idxs = [...new Set(epochs.map((e) => e.trial).filter((t): t is number => t != null))].sort((a, b) => a - b);
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">
            Eval metrics · per trial{" "}
            <span className="text-[11px] font-normal text-muted-foreground">(per epoch, lower is better)</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <PerTrialEvalChart epochs={epochs} metric="wer" idxs={idxs} trials={trials} />
          <PerTrialEvalChart epochs={epochs} metric="cer" idxs={idxs} trials={trials} />
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
            {idxs.map((i) => (
              <span key={i}>
                <span style={{ color: TRIAL_COLORS[i % TRIAL_COLORS.length] }}>■</span> {trialLabel(i, trials)}
              </span>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  const data = epochs
    .filter((e) => typeof e.wer === "number" || typeof e.cer === "number")
    .map((e) => ({ epoch: e.epoch, wer: e.wer ?? null, cer: e.cer ?? null }));
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

// Try-it playground — upload a clip, pick a GPU the run used (or CPU), and
// transcribe it with the finetuned model on the run's VM (over SSH).
function PlaygroundTab({ runId, visibleDevices }: { runId: string; visibleDevices: string | null }) {
  const gpuIds = (visibleDevices ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  const [file, setFile] = useState<File | null>(null);
  const [gpu, setGpu] = useState<string>(gpuIds[0] ?? "auto");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ text: string; device?: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function onTranscribe() {
    if (!file) return;
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      setResult(await gateway.transcribeTrainingRun(runId, file, gpu));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader className="pb-2"><CardTitle className="text-sm">Try it — transcribe a clip</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <p className="text-xs text-muted-foreground">
          Runs the finetuned model on this run&apos;s VM. Pick a GPU the run used (or CPU), upload a short
          audio clip (≤ 25 MB), and transcribe. The first request downloads the model onto the VM, so it
          can take a little longer.
        </p>
        <div className="flex flex-wrap items-end gap-3">
          <div className="space-y-1.5">
            <label className="block text-xs font-medium">Audio clip</label>
            <input
              type="file"
              accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm"
              onChange={(e) => { setFile(e.target.files?.[0] ?? null); setResult(null); }}
              className="block text-xs file:mr-3 file:rounded-md file:border-0 file:bg-muted file:px-2.5 file:py-1.5 file:text-foreground hover:file:bg-muted/70"
            />
          </div>
          <div className="space-y-1.5">
            <label className="block text-xs font-medium">Run on</label>
            <select
              value={gpu}
              onChange={(e) => setGpu(e.target.value)}
              className="h-9 rounded-md border border-border bg-background px-2 text-sm"
            >
              {gpuIds.map((g) => <option key={g} value={g}>GPU {g}</option>)}
              <option value="auto">Auto (most-free GPU)</option>
              <option value="cpu">CPU</option>
            </select>
          </div>
          <Button type="button" onClick={onTranscribe} disabled={busy || !file}>
            {busy
              ? <><Loader2 className="h-4 w-4 animate-spin" /> Transcribing…</>
              : <><AudioLines className="h-4 w-4" /> Transcribe</>}
          </Button>
        </div>
        {err && <p className="text-sm text-destructive">{err}</p>}
        {result && (
          <div className="space-y-1.5">
            <div className="text-xs text-muted-foreground">
              Transcription{result.device ? ` · ran on ${result.device}` : ""}
            </div>
            <div className="whitespace-pre-wrap rounded-md border border-border bg-muted/30 p-3 text-sm">
              {result.text || <span className="text-muted-foreground">(empty — the model returned nothing)</span>}
            </div>
          </div>
        )}
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
