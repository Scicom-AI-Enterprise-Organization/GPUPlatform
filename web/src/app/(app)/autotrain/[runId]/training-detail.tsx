"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { AudioLines, Check, ChevronDown, Copy, Download, ExternalLink, Loader2, PackageOpen, Pencil, RotateCcw, Trash2, Upload, X, XCircle } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
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
import { formatCostUSD, formatRateUSD, useLiveCost } from "@/lib/cost";
import { BurnFlame } from "@/components/burn-flame";
import { JsonView } from "@/components/json-view";
import { cn } from "@/lib/utils";
import type { DatasetRecord, StorageRecord, TrainingEpoch, TrainingFile, TrainingGpu, TrainingGpuSample, TrainingRunRecord, TrainingStep, TrainingTrial } from "@/lib/types";
import { LabelExportTab } from "./label-export-tab";
import { TryItCompute, defaultCompute, type ComputeChoice } from "./tryit-compute";
import { gpuTypeToChoice } from "@/lib/gpu-catalog";

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

// Friendly label for the current post-training phase (from result_json.progress.step,
// set by the trainer's [AUTOTRAIN_PROGRESS] markers). Shown beside the "running"
// status so a long, otherwise-silent step (e.g. multi-GB S3 upload) isn't mistaken
// for a stuck run. Returns null when not running or no phase is known.
const PHASE_LABELS: Record<string, string> = {
  evaluating: "evaluating",
  tts_eval_gen: "evaluating",
  uploading: "uploading to S3",
  pushing_hf: "pushing to Hugging Face",
};
function runningPhase(run: TrainingRunRecord): string | null {
  if (run.status !== "running") return null;
  const p = run.result_json?.progress;
  const step = p?.step ? String(p.step) : "";
  if (!step) return null;
  const label = PHASE_LABELS[step] ?? step.replace(/_/g, " ");
  const pct = typeof p?.percent === "number" ? p.percent : null;
  return pct != null && (step === "uploading" || step === "pushing_hf")
    ? `${label} · ${Math.round(pct)}%`
    : label;
}

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
  // A post-train Label export OR a Hugging Face push is running in the background —
  // the run itself is already "done", so keep polling so the status/link refresh.
  const exportingLabel =
    run.result_json?.hf_export?.status === "running" ? "exporting to HF"
    : run.result_json?.label_export?.status === "running" ? "exporting to Label"
    : null;
  const exporting = exportingLabel != null;

  // Tab reflected in the URL (?tab=…) so it's deep-linkable + survives refresh.
  const tab = searchParams.get("tab") || "metrics";
  // Each tab trigger is a real <Link> (so right/middle/⌘-click opens it in a new
  // tab); a plain click still switches in place because `tab` derives from the URL.
  const tabHref = (v: string) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("tab", v);
    return `${pathname}?${params.toString()}`;
  };

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

  // Poll the record while queued/running — or while a post-train Label export is
  // in flight — so the status + the label_project card refresh.
  useEffect(() => {
    if (terminal && !exporting) return;
    const t = setInterval(async () => {
      try {
        setRun(await gateway.getTrainingRun(run.id));
      } catch {
        /* keep last */
      }
    }, 5000);
    return () => clearInterval(t);
  }, [run.id, terminal, exporting]);

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
  // A pack-only run (NeuCodec encode + multipack, no training) has no loss curve
  // or per-epoch eval — hide those empty panels for it.
  const packOnly = run.config_json?.pack_only === true;
  // Try-it playground — ASR transcribes an uploaded clip; TTS synthesizes speech.
  // The compute target is chosen at load time (a fresh RunPod pod or a registered
  // VM — see TryItCompute), decoupled from where the run trained, so ASR + TTS can
  // try-it regardless of provider. LLM (vLLM) try-it stays VM-only for now.
  const isVmRun = run.provider_kind === "vm";
  const canTryIt =
    run.status === "done" &&
    !!run.result_json?.artifact?.s3_uri &&
    ((run.task_type ?? "asr") === "asr" || run.task_type === "tts" || (run.task_type === "llm" && isVmRun));
  const metricLabel = run.task_type === "tts"
    ? "loss"
    : String((run.config_json?.eval_metric as string) || "wer").toUpperCase();
  // TTS has no WER/CER (ASR-only metrics). Its eval signal is the held-out loss
  // on the test split; CER / MOS / speaker-similarity only exist when those eval
  // methods were selected (shown separately in the TTS evaluation card).
  const isTts = run.task_type === "tts";
  const evalLosses = epochs
    .map((e) => e.eval_loss)
    .filter((n): n is number => typeof n === "number");
  const bestEvalLoss = evalLosses.length ? Math.min(...evalLosses) : (best?.eval_loss ?? null);
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

  // Label-platform export now lives in its own tab (LabelExportTab) — with a
  // serverless-style "Run on" picker (VM or a fresh RunPod pod). `lcfg` is still
  // read below by the HF export.
  const lcfg = (run.config_json ?? {}) as Record<string, unknown>;
  const canLabelExport = (run.task_type === "tts" || run.task_type === "llm") && run.status === "done";

  // On-demand "Export to Hugging Face" — pushes the run's best/final model to a HF
  // repo, with the token taken from a selected kind=huggingface storage.
  const [hfOpen, setHfOpen] = useState(false);
  const [hfStorages, setHfStorages] = useState<StorageRecord[]>([]);
  const [hfStorageId, setHfStorageId] = useState("");
  const [hfRepo, setHfRepo] = useState(typeof lcfg.hf_push_repo === "string" ? lcfg.hf_push_repo : "");
  const [hfPrivate, setHfPrivate] = useState(true);  // default private
  const [hfBusy, setHfBusy] = useState(false);
  const [hfErr, setHfErr] = useState<string | null>(null);
  const [hfDone, setHfDone] = useState(false);

  useEffect(() => {
    if (!hfOpen || hfStorages.length) return;
    gateway
      .listStorage()
      .then((rows) => setHfStorages(rows.filter((s) => s.kind === "huggingface")))
      .catch(() => {});
  }, [hfOpen, hfStorages.length]);

  async function submitHfExport() {
    setHfBusy(true);
    setHfErr(null);
    try {
      await gateway.exportToHuggingFace(run.id, {
        repo: hfRepo.trim(),
        storage_id: hfStorageId || null,
        private: hfPrivate,
      });
      setHfDone(true);
    } catch (e) {
      setHfErr(e instanceof Error ? e.message : String(e));
    } finally {
      setHfBusy(false);
    }
  }

  // Stop a stuck/running HF export (also kills the VM-side process; clears a status
  // left stuck on "running" by a gateway restart).
  const [hfStopping, setHfStopping] = useState(false);
  async function stopHfExport() {
    setHfStopping(true);
    try {
      await gateway.cancelHuggingFaceExport(run.id);
    } catch {
      // best-effort; the next poll reflects the real state
    } finally {
      setHfStopping(false);
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

  function onStopEarly() {
    setConfirmError(null);
    setConfirmOpts({
      title: "Stop training early?",
      description: "The trainer finishes the current step, then saves + uploads the partial model and finalizes the run (and runs any Label/HF export). Unlike Terminate, the model trained so far is kept.",
      confirmLabel: "Stop & save",
      busyLabel: "Signalling…",
      run: async () => {
        setRun(await gateway.stopTrainingEarly(run.id));
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

  // Portable export: a self-contained JSON (config + metrics/loss + small S3 files)
  // to import into another deployment's dashboard via /autotrain/import.
  const [portExporting, setPortExporting] = useState(false);
  async function onExport() {
    setPortExporting(true);
    try {
      const data = await gateway.exportTrainingRun(run.id);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${run.id}.autotrain.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      const omitted = Array.isArray((data as { files_omitted?: unknown[] }).files_omitted)
        ? (data as { files_omitted: unknown[] }).files_omitted.length
        : 0;
      toast.success(
        omitted > 0
          ? `Exported (${omitted} file${omitted === 1 ? "" : "s"} omitted — over size cap)`
          : "Run exported",
        { duration: 3000 },
      );
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setPortExporting(false);
    }
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
  // Pass ?task= so the form mounts with the right task (and thus the right model
  // list + dataset filter) on the FIRST render — otherwise it defaults to asr and
  // an effect flips asr→tts, which leaves the model/dataset/test Selects applying
  // their (tts) values against the wrong (asr) option set, so they don't inherit.
  function onEditAsNew() {
    const task = run.task_type === "tts" ? "tts" : "asr";
    router.push(`/autotrain/new?from=${encodeURIComponent(run.id)}&task=${task}`);
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="border-b border-border bg-sidebar/40 px-6 pt-4 lg:px-10">
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
            {exporting ? (
              <Badge variant="outline" className={STATUS_STYLES.running}>
                <Loader2 className="mr-1 h-3 w-3 animate-spin" /> {exportingLabel}
              </Badge>
            ) : (
              <Badge variant="outline" className={STATUS_STYLES[run.status] ?? ""}>{run.status}</Badge>
            )}
            {!exporting && runningPhase(run) && (
              <span className="flex items-center gap-1.5 text-xs font-medium text-amber-600 dark:text-amber-400">
                <Loader2 className="h-3 w-3 animate-spin" />
                {runningPhase(run)}
              </span>
            )}
            <RunKindBadge sweep={isSweep} trials={isSweep ? trials.length : undefined} />
          </div>
          <p className="mt-1 font-mono text-xs text-muted-foreground">
            {run.base_model} · {run.id}
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
          {run.status === "running" && (
            <Button variant="outline" size="sm" onClick={onStopEarly}
              disabled={busy || !!run.result_json?.stopping_early}
              title="Finish the current step, save + upload the partial model, then finalize">
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
              {run.result_json?.stopping_early ? "Stopping…" : "Stop & save"}
            </Button>
          )}
          {!terminal && (
            <Button variant="outline" size="sm" onClick={onTerminate} disabled={busy} className="text-destructive">
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <XCircle className="h-4 w-4" />} Terminate
            </Button>
          )}
          {canTryIt && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => { setHfErr(null); setHfDone(false); setHfOpen(true); }}
              title="Push the best/final model to a Hugging Face repo"
            >
              <Upload className="h-4 w-4" /> Export to HF
            </Button>
          )}
          <Button variant="outline" size="sm" onClick={onExport} disabled={portExporting}
            title="Download a portable JSON (config + metrics/loss + logs) to import into another deployment">
            {portExporting ? <Loader2 className="h-4 w-4 animate-spin" /> : <PackageOpen className="h-4 w-4" />} Export
          </Button>
          <Button variant="outline" size="sm" onClick={onDelete} disabled={busy}>
            <Trash2 className="h-4 w-4" /> Delete
          </Button>
        </div>
      </div>

        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-5">
          <Kpi label="Status" value={exporting ? exportingLabel! : run.status} />
          <Kpi
            label="GPU"
            value={run.gpu_type ? `${run.gpu_type}${run.gpu_count > 1 ? ` ×${run.gpu_count}` : ""}` : "—"}
          />
          <CostKpi run={run} />
          <Kpi
            label={isTts ? "Best eval loss" : `Best ${metricLabel}`}
            value={isTts ? fmt(bestEvalLoss, 4) : fmt(best?.wer ?? null)}
          />
          {isSweep && <Kpi label="Trials" value={String(trials.length)} />}
        </div>

        <Tabs value={tab} className="mt-4">
          <TabsList variant="line" className="bg-transparent">
            <TabsTrigger value="metrics" asChild><Link href={tabHref("metrics")} scroll={false}>Metrics</Link></TabsTrigger>
            <TabsTrigger value="logs" asChild><Link href={tabHref("logs")} scroll={false}>Logs</Link></TabsTrigger>
            <TabsTrigger value="files" asChild><Link href={tabHref("files")} scroll={false}>Files</Link></TabsTrigger>
            <TabsTrigger value="config" asChild><Link href={tabHref("config")} scroll={false}>Config</Link></TabsTrigger>
            {canTryIt && <TabsTrigger value="tryit" asChild><Link href={tabHref("tryit")} scroll={false}>Try it</Link></TabsTrigger>}
            {canLabelExport && <TabsTrigger value="label" asChild><Link href={tabHref("label")} scroll={false}>Export to Label</Link></TabsTrigger>}
          </TabsList>
        </Tabs>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
      {run.error_text && run.status === "failed" && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <pre className="whitespace-pre-wrap break-words font-mono text-xs">{run.error_text}</pre>
        </div>
      )}

      {tab === "metrics" && (<div className="space-y-4 mb-4">
      {best && !isSweep && (
        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm">Best checkpoint</CardTitle></CardHeader>
          <CardContent className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
            <Stat label="Best epoch" value={String(best.epoch ?? "—")} />
            {!isTts && <Stat label="WER" value={fmt(best.wer)} />}
            {!isTts && <Stat label="CER" value={fmt(best.cer)} />}
            <Stat label="Eval loss" value={fmt(best.eval_loss, 4)} />
            {run.result_json?.stopped_early && <Stat label="Stopped early" value="yes (patience)" />}
            {artifact?.s3_uri && <Stat label="Artifact" value={artifact.s3_uri} mono />}
            {artifact?.hf_repo && <Stat label="HF" value={artifact.hf_repo} mono />}
          </CardContent>
        </Card>
      )}

      {run.result_json?.tts_eval && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">
              TTS evaluation{run.result_json.tts_eval.samples != null ? ` · ${run.result_json.tts_eval.samples} samples` : ""}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
            {run.result_json.tts_eval.cer != null && <Stat label="CER ↓" value={fmt(run.result_json.tts_eval.cer, 4)} />}
            {run.result_json.tts_eval.mos != null && <Stat label="MOS ↑ (UTMOSv2)" value={fmt(run.result_json.tts_eval.mos, 3)} />}
            {run.result_json.tts_eval.similarity != null && <Stat label="Speaker sim ↑ (TitaNet)" value={fmt(run.result_json.tts_eval.similarity, 4)} />}
          </CardContent>
        </Card>
      )}

      {/* One card per Label project (multiple when split per speaker; falls back to
          the single label_project for runs created before that). */}
      {(run.result_json?.label_projects
        ?? (run.result_json?.label_project ? [run.result_json.label_project] : [])
      ).map((lp) => (
        <Card key={lp.id}>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">
              Label project created
              {lp.speaker ? ` · ${lp.speaker}` : ""}
              {lp.count != null
                ? ` · ${lp.count} ${lp.project_type === "human_mos" ? "conversations" : "clips"}`
                : ""}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap items-center gap-x-8 gap-y-2 text-sm">
            <a
              href={lp.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 font-medium text-primary hover:underline"
            >
              Open in Label
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
            {lp.dataset_id && (
              <a
                href={`/datasets/${lp.dataset_id}`}
                className="inline-flex items-center gap-1.5 text-muted-foreground hover:text-foreground hover:underline"
              >
                Linked dataset <span className="font-mono">{lp.dataset_id}</span>
              </a>
            )}
          </CardContent>
        </Card>
      ))}

      {run.result_json?.hf_export && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Hugging Face export</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap items-center gap-x-8 gap-y-2 text-sm">
            {run.result_json.hf_export.status === "running" && (
              <>
                <span className="flex items-center gap-1.5 text-amber-600 dark:text-amber-400">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" /> pushing {run.result_json.hf_export.repo} …
                </span>
                <Button variant="outline" size="sm" onClick={stopHfExport} disabled={hfStopping}>
                  {hfStopping ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <X className="h-3.5 w-3.5" />}
                  Stop
                </Button>
              </>
            )}
            {run.result_json.hf_export.status === "cancelled" && (
              <span className="flex items-center gap-x-3 text-muted-foreground">
                push stopped{run.result_json.hf_export.error ? ` — ${run.result_json.hf_export.error}` : ""}
              </span>
            )}
            {run.result_json.hf_export.status === "done" && run.result_json.hf_export.url && (
              <a
                href={run.result_json.hf_export.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 font-medium text-primary hover:underline"
              >
                Open on Hugging Face — {run.result_json.hf_export.repo}
                <ExternalLink className="h-3.5 w-3.5" />
              </a>
            )}
            {run.result_json.hf_export.status === "failed" && (
              <span className="text-destructive">push failed: {run.result_json.hf_export.error}</span>
            )}
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
      </div>)}

      <Tabs value={tab} className="!block">
        <TabsContent value="metrics" className="!flex-none space-y-4">
          {!packOnly && <LossCurve steps={steps} epochs={epochs} live={!terminal} sweep={isSweep} trials={trials} />}
          <EvalCurve epochs={epochs} sweep={isSweep} trials={trials} />
          <GpuCard gpus={gpus} samples={run.result_json?.gpu_samples ?? []} running={!terminal} />
          {epochs.length === 0 ? (
            packOnly ? null : (
            <p className="text-sm text-muted-foreground">
              No per-epoch metrics yet. They appear here as each epoch finishes evaluating.
            </p>
            )
          ) : (
            <div className="overflow-x-auto rounded-md border border-border">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 text-xs text-muted-foreground">
                  <tr>
                    {isSweep && <th className="px-3 py-2 text-left">Trial</th>}
                    <th className="px-3 py-2 text-left">Epoch</th>
                    {!isTts && <th className="px-3 py-2 text-right">WER</th>}
                    {!isTts && <th className="px-3 py-2 text-right">CER</th>}
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
                        {!isTts && <td className="px-3 py-2 text-right font-mono">{fmt(e.wer)}</td>}
                        {!isTts && <td className="px-3 py-2 text-right font-mono">{fmt(e.cer)}</td>}
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

        <TabsContent value="logs" className="!flex-none">
          <LogsTab lines={lines} status={run.status} />
        </TabsContent>

        <TabsContent value="files" className="!flex-none">
          <FilesTab run={run} />
        </TabsContent>

        <TabsContent value="config" className="!flex-none space-y-4">
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
                href={run.provider_id ? "/providers" : undefined}
              />
              <Stat
                label="GPU"
                value={run.gpu_type ? `${run.gpu_type}${run.gpu_count > 1 ? ` × ${run.gpu_count}` : ""}` : "—"}
              />
              {run.visible_devices && <Stat label="GPU ids" value={run.visible_devices} mono />}
              <Stat label="Storage" value={run.storage_name || run.storage_id || "—"}
                href={run.storage_id ? "/storage" : undefined} />
              <Stat label="Dataset" value={run.dataset_id} mono
                href={run.dataset_id ? `/datasets/${run.dataset_id}` : undefined} />
              {run.test_dataset_id && <Stat label="Test dataset" value={run.test_dataset_id} mono
                href={`/datasets/${run.test_dataset_id}`} />}
              <Stat label="Base model" value={run.base_model} mono />
            </CardContent>
          </Card>
          <StepEstimate run={run} />
          <JsonView value={run.config_json} />
        </TabsContent>

        {canTryIt && (
          <TabsContent value="tryit" className="!flex-none">
            {(run.task_type ?? "asr") === "tts"
              ? <TtsPlaygroundTab runId={run.id} visibleDevices={run.visible_devices ?? null}
                  runProviderId={run.provider_id ?? null} trainedOnVm={isVmRun}
                  gpuType={run.gpu_type ?? null} gpuCount={run.gpu_count ?? null} />
              : run.task_type === "llm"
                ? <LlmPlaygroundTab runId={run.id} visibleDevices={run.visible_devices ?? null}
                    runProviderId={run.provider_id ?? null} />
                : <PlaygroundTab runId={run.id} visibleDevices={run.visible_devices ?? null}
                    runProviderId={run.provider_id ?? null} trainedOnVm={isVmRun}
                    gpuType={run.gpu_type ?? null} gpuCount={run.gpu_count ?? null} />}
          </TabsContent>
        )}
        {canLabelExport && (
          <TabsContent value="label" className="!flex-none">
            <LabelExportTab run={run} onStarted={() => router.refresh()} />
          </TabsContent>
        )}
      </Tabs>
      </div>

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

      <Dialog open={hfOpen} onOpenChange={(o) => { if (!hfBusy) setHfOpen(o); }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Export to Hugging Face</DialogTitle>
            <DialogDescription>
              Push this run&apos;s <span className="font-medium">best (final) checkpoint</span> to a Hugging Face
              model repo. Runs in the background — watch the Logs tab; the link appears on this page when it finishes.
            </DialogDescription>
          </DialogHeader>
          {hfDone ? (
            <p className="flex items-center gap-2 py-2 text-sm text-emerald-600 dark:text-emerald-400">
              <Check className="h-4 w-4" /> Push started — the model downloads from S3 then uploads to HF; the
              status shows "pushing to Hugging Face" and an "Open on HF" link appears on this page when done.
            </p>
          ) : (
            <div className="space-y-3">
              <div className="space-y-1.5">
                <label className="text-xs uppercase tracking-wide text-muted-foreground">Repo name</label>
                <Input className="font-mono" value={hfRepo} placeholder="org/model-name"
                  onChange={(e) => setHfRepo(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs uppercase tracking-wide text-muted-foreground">Hugging Face storage</label>
                <Select value={hfStorageId} onValueChange={setHfStorageId}>
                  <SelectTrigger>
                    <SelectValue placeholder={hfStorages.length ? "Choose a HuggingFace storage" : "None configured"} />
                  </SelectTrigger>
                  <SelectContent>
                    {hfStorages.map((s) => (
                      <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-[11px] text-muted-foreground">
                  Its token is used to push.{hfStorages.length === 0 ? " No HuggingFace storage configured — the platform HF_TOKEN secret is used instead, if set." : ""}
                </p>
              </div>
              <label className="flex cursor-pointer items-center gap-2 text-sm">
                <input type="checkbox" checked={hfPrivate} onChange={(e) => setHfPrivate(e.target.checked)}
                  className="h-4 w-4 accent-primary" />
                <span>Private repo</span>
              </label>
            </div>
          )}
          <DialogFooter>
            {hfErr && <p className="mr-auto text-sm text-destructive">{hfErr}</p>}
            <Button variant="outline" onClick={() => setHfOpen(false)} disabled={hfBusy}>
              {hfDone ? "Close" : "Cancel"}
            </Button>
            {!hfDone && (
              <Button onClick={submitHfExport} disabled={hfBusy || !hfRepo.trim()}>
                {hfBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                Push to HF
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// Estimated optimizer-step count for this run's config. steps/epoch ≈
// ceil(train_rows / (batch × grad_accum × world_size)). world_size = #GPUs whenever
// DDP runs: TTS ALWAYS torchruns (nproc = #GPUs), ASR only on multi-GPU with DDP on.
// train_rows = the tts_packed train split, else the dataset's rows (minus the
// auto-split eval fraction for ASR). Fetches the dataset for its row count.
function StepEstimate({ run }: { run: TrainingRunRecord }) {
  const [ds, setDs] = useState<DatasetRecord | null>(null);
  useEffect(() => {
    let cancelled = false;
    if (!run.dataset_id) return;
    gateway.getDataset(run.dataset_id).then((d) => { if (!cancelled) setDs(d); }).catch(() => {});
    return () => { cancelled = true; };
  }, [run.dataset_id]);

  const c = (run.config_json ?? {}) as Record<string, unknown>;
  const numOf = (v: unknown, d: number) =>
    typeof v === "number" && Number.isFinite(v) ? v
    : typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v)) ? Number(v) : d;
  const isTts = (run.task_type ?? c.task_type) === "tts";
  const batch = numOf(c.batch_size, 0);
  const gradAccum = Math.max(1, numOf(c.grad_accum, 1));
  const epochs = Math.max(1, numOf(c.max_epochs, 0));
  const maxSteps = numOf(c.max_steps, 0);
  const useDdp = c.use_ddp !== false;

  // train rows
  let trainRows: number | null = null;
  const sp = (ds?.split_fields as Record<string, unknown> | null | undefined)?.["_tts_pack"];
  const splits = (sp as Record<string, unknown> | null | undefined)?.["splits"] as Record<string, unknown> | undefined;
  if (splits && typeof splits.train === "number") trainRows = splits.train as number;
  else if (ds && typeof ds.num_rows === "number") {
    const evalPct = numOf(c.eval_split_pct, 0);
    trainRows = !isTts && !run.test_dataset_id && evalPct > 0
      ? Math.round(ds.num_rows * (1 - evalPct / 100))
      : ds.num_rows;
  }

  const nGpus = run.visible_devices && run.visible_devices.trim()
    ? run.visible_devices.split(",").filter((x) => x.trim()).length
    : Math.max(1, run.gpu_count || 1);
  const worldSize = isTts ? Math.max(1, nGpus) : (useDdp && nGpus > 1 ? nGpus : 1);
  const effBatch = Math.max(1, batch) * gradAccum * worldSize;
  const perEpoch = trainRows && trainRows > 0 ? Math.ceil(trainRows / effBatch) : null;
  if (batch <= 0 || perEpoch == null) return null;
  const total = perEpoch * epochs;

  return (
    <Card>
      <CardHeader className="pb-2"><CardTitle className="text-sm">Estimated steps</CardTitle></CardHeader>
      <CardContent className="space-y-2">
        <div className="flex flex-wrap gap-x-8 gap-y-3 text-sm">
          <Stat label="Steps / epoch" value={`≈ ${perEpoch.toLocaleString()}`} />
          <Stat
            label={maxSteps > 0 ? "Total (step-capped)" : "Total steps"}
            value={maxSteps > 0 ? `≈ ${Math.min(maxSteps, total).toLocaleString()} (cap ${maxSteps.toLocaleString()})` : `≈ ${total.toLocaleString()}`}
          />
          <Stat label="Effective batch" value={`${effBatch.toLocaleString()}`} />
          <Stat label="World size" value={worldSize > 1 ? `${worldSize} GPUs (DDP)` : "1"} />
        </div>
        <p className="text-[11px] text-muted-foreground">
          {trainRows!.toLocaleString()} train rows ÷ (batch {batch} × grad-accum {gradAccum}
          {worldSize > 1 ? ` × ${worldSize} GPUs` : ""}) × {epochs} epoch{epochs === 1 ? "" : "s"}.
        </p>
      </CardContent>
    </Card>
  );
}

function Stat({ label, value, mono, href }: { label: string; value: string; mono?: boolean; href?: string }) {
  const cls = `text-sm ${mono ? "font-mono break-all" : "font-medium"}`;
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      {href ? (
        <a href={href} target="_blank" rel="noreferrer" className={`${cls} text-primary hover:underline`}>{value}</a>
      ) : (
        <div className={cls}>{value}</div>
      )}
    </div>
  );
}

// Header-band KPI cell (matches the benchmark / serverless detail headers).
function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-0.5 truncate text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}

// Live-ticking spend while running; final total once the run ends.
function CostKpi({ run }: { run: TrainingRunRecord }) {
  const live = useLiveCost(run.started_at, run.ended_at, run.cost_per_hr);
  const isBurning = run.status === "running" && run.cost_per_hr != null && run.ended_at == null;
  return (
    <div>
      <div className="text-xs text-muted-foreground">Cost {isBurning ? "(live)" : ""}</div>
      <div
        className={cn(
          "mt-0.5 flex items-center gap-1.5 text-lg font-semibold tabular-nums",
          isBurning && "text-amber-600 dark:text-amber-400",
        )}
      >
        {isBurning && <BurnFlame size="h-4 w-4" />}
        {formatCostUSD(live)}
      </div>
      <div className="text-[10px] text-muted-foreground">
        {run.cost_per_hr != null ? `at ${formatRateUSD(run.cost_per_hr)}` : "—"}
      </div>
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

// Explicit, de-duplicated x-axis ticks. recharts' auto-tick generator (esp. with
// allowDecimals=false) can emit duplicate ticks when points sit at ~the same value
// (a degenerate domain — common with step-based eval, where all points share a near
// epoch), tripping React's "two children with the same key" warning. Cap at ~8.
function uniqTicks(values: number[]): number[] {
  const xs = [...new Set(values)].sort((a, b) => a - b);
  if (xs.length <= 8) return xs;
  return [...new Set(Array.from({ length: 8 }, (_, i) => xs[Math.round((i * (xs.length - 1)) / 7)]))];
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
              ticks={uniqTicks(data.map((d) => d.epoch))}
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
                ticks={uniqTicks(data.map((d) => d.epoch))}
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

// Try-it playground (ASR) — pick where to run (a fresh RunPod pod or a registered
// VM, via TryItCompute), load the model, then upload a clip and transcribe it.
function PlaygroundTab({ runId, visibleDevices, runProviderId, trainedOnVm, gpuType, gpuCount }: {
  runId: string; visibleDevices: string | null;
  runProviderId: string | null; trainedOnVm: boolean;
  gpuType: string | null; gpuCount: number | null;
}) {
  const [compute, setCompute] = useState<ComputeChoice>(() => defaultCompute({
    trainedOnVm, runProviderId, gpuChoice: gpuTypeToChoice(gpuType), gpuCount,
    pins: (visibleDevices ?? "").split(",").map((s) => s.trim()).filter(Boolean),
  }));
  const [loaded, setLoaded] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ text: string; raw?: string | null; device?: string; logs?: string[] } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function onTranscribe() {
    if (!file) return;
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      // gpu is the device index for the VM target; "auto" for a cloud pod.
      setResult(await gateway.transcribeTrainingRun(runId, file, compute.gpu));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <TryItCompute value={compute} onChange={setCompute} disabled={loaded}
        runProviderId={runProviderId} visibleDevices={visibleDevices} />
      <Card>
      <CardHeader className="pb-2"><CardTitle className="text-sm">Try it — transcribe a clip</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <p className="text-xs text-muted-foreground">
          Load the model, then upload a short clip (≤ 25 MB) and transcribe. A cloud pod spins up on
          demand (first load ~10 min) and auto-stops when idle; a VM keeps the model resident. The
          first request downloads the model onto the box.
        </p>
        <PersistentControls runId={runId} compute={compute} onRunningChange={setLoaded} />
        <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">audio clip</span>
            <input
              type="file"
              accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm"
              onChange={(e) => { setFile(e.target.files?.[0] ?? null); setResult(null); }}
              className="flex h-8 items-center rounded-md border border-input bg-transparent text-xs shadow-xs file:mr-3 file:h-8 file:border-0 file:border-r file:border-input file:bg-muted file:px-2.5 file:text-foreground hover:file:bg-muted/70"
            />
          </div>
          <Button type="button" onClick={onTranscribe} disabled={busy || !file} className="ml-auto">
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
        {result?.raw && (
          <div className="space-y-1.5">
            <div className="text-xs text-muted-foreground">
              Raw output (special tokens kept · first 30s) — verify the Whisper prompt +{" "}
              <span className="font-mono">{"<|endoftext|>"}</span> EOS
            </div>
            <div className="whitespace-pre-wrap break-all rounded-md border border-border bg-muted/30 p-3 font-mono text-xs">
              {result.raw}
            </div>
          </div>
        )}
        {result && <TryItLogs lines={result.logs ?? []} />}
      </CardContent>
      </Card>
    </div>
  );
}

// Collapsible VM-side log block for the playground (model download / inference).
function TryItLogs({ lines }: { lines: string[] }) {
  if (!lines || lines.length === 0) return null;
  return (
    <details className="rounded-md border border-border">
      <summary className="cursor-pointer select-none px-3 py-1.5 text-xs text-muted-foreground">
        Logs ({lines.length} lines)
      </summary>
      <div className="terminal-block max-h-64 overflow-y-auto border-t border-border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-200">
        {lines.map((l, i) => (
          <div key={i} className={
            l.includes("[tryit]") ? "text-emerald-300"
              : l.startsWith("@@") ? "text-sky-300"
              : "text-zinc-300"
          }>{l}</div>
        ))}
      </div>
    </details>
  );
}

// Persistent worker controls — load the model once on the chosen compute (a fresh
// RunPod pod or a registered VM, per `compute`) so try-it requests skip the per-call
// model load, with Load / Restart / Unload. Reports running state up so the parent
// can lock the compute picker while something is loaded.
function PersistentControls({ runId, compute, onRunningChange }: {
  runId: string; compute: ComputeChoice; onRunningChange?: (running: boolean) => void;
}) {
  const [st, setSt] = useState<{ running: boolean; ready: boolean; device?: string; logs?: string[] } | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const poll = useCallback(async () => {
    try { setSt(await gateway.playgroundStatus(runId)); } catch { /* transient */ }
  }, [runId]);
  useEffect(() => { poll(); }, [poll]);
  // While loading, poll until ready (the model load takes ~10-15s on a VM, ~10 min
  // on a fresh cloud pod).
  useEffect(() => {
    if (!st?.running || st.ready) return;
    const t = setInterval(poll, 3000);
    return () => clearInterval(t);
  }, [st, poll]);
  // Lock the compute picker whenever a worker/pod is loading or loaded.
  useEffect(() => { onRunningChange?.(!!st?.running); }, [st?.running, onRunningChange]);

  async function act(fn: () => Promise<unknown>) {
    setBusy(true); setErr(null);
    try { await fn(); await poll(); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  const label = !st?.running ? "not loaded" : st.ready ? `ready${st.device ? ` · ${st.device}` : ""}` : "loading…";
  const dot = !st?.running ? "bg-muted-foreground/50" : st.ready ? "bg-emerald-500" : "bg-amber-500 animate-pulse";

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-muted/20 px-3 py-2 text-xs">
      <span className="font-medium">Persistent model</span>
      <span className={cn("inline-block h-2 w-2 rounded-full", dot)} />
      <span className="text-muted-foreground">{label}</span>
      <span className="hidden text-[11px] text-muted-foreground sm:inline">— keep it resident so requests skip the load</span>
      <div className="ml-auto flex items-center gap-2">
        {busy && <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />}
        {!st?.running ? (
          <Button type="button" variant="outline" className="h-7 text-xs" disabled={busy}
            onClick={() => act(() => gateway.playgroundStart(runId, compute))}>Load model</Button>
        ) : (
          <>
            <Button type="button" variant="outline" className="h-7 text-xs" disabled={busy}
              onClick={() => act(async () => { await gateway.playgroundStop(runId); await gateway.playgroundStart(runId, compute); })}>Restart</Button>
            <Button type="button" variant="outline" className="h-7 text-xs" disabled={busy}
              onClick={() => act(() => gateway.playgroundStop(runId))}>Unload</Button>
          </>
        )}
      </div>
      {err && <span className="w-full text-destructive">{err}</span>}
      {st?.running && (st.logs?.length ?? 0) > 0 && (
        <div className="terminal-block max-h-32 w-full overflow-y-auto rounded-md border border-border bg-zinc-950 p-2 font-mono text-[10px] leading-snug text-zinc-300">
          {st.logs!.map((l, i) => (
            <div key={i} className={l.includes("[server]") ? "text-emerald-300" : "text-zinc-400"}>{l}</div>
          ))}
        </div>
      )}
    </div>
  );
}

type LlmToolCall = { index: number; id: string | null; name: string; argsBuf: string };

// Merge streamed OpenAI tool_call deltas (fragmented across chunks) into per-index slots.
function mergeToolCallDeltas(prev: LlmToolCall[], deltas: Array<Record<string, unknown>>): LlmToolCall[] {
  const next = prev.slice();
  for (const d of deltas) {
    const idx = typeof d.index === "number" ? d.index : 0;
    let slot = next.find((s) => s.index === idx);
    if (!slot) { slot = { index: idx, id: null, name: "", argsBuf: "" }; next.push(slot); }
    if (typeof d.id === "string") slot.id = d.id;
    const fn = d.function as { name?: string; arguments?: string } | undefined;
    if (fn?.name) slot.name = fn.name;
    if (typeof fn?.arguments === "string") slot.argsBuf += fn.arguments;
  }
  return next.sort((a, b) => a.index - b.index);
}

// "Sample" button — a couple of OpenAI-shape function specs to exercise tool calling.
const LLM_SAMPLE_TOOLS = JSON.stringify([
  { type: "function", function: { name: "get_weather", description: "Get the current weather in a location",
    parameters: { type: "object", properties: { location: { type: "string", description: "City, e.g. Kuala Lumpur, MY" },
      unit: { type: "string", enum: ["celsius", "fahrenheit"] } }, required: ["location"] } } },
  { type: "function", function: { name: "get_stock_price", description: "Latest stock price for a ticker symbol",
    parameters: { type: "object", properties: { ticker: { type: "string", description: "e.g. AAPL, MAYBANK" } }, required: ["ticker"] } } },
], null, 2);

// Try-it playground (LLM, gemma-4) — load the finetuned model via vLLM (eager) on
// the run's VM (download LoRA → merge → save → serve), then stream chat completions.
function LlmPlaygroundTab({ runId, visibleDevices, runProviderId }: {
  runId: string; visibleDevices: string | null; runProviderId: string | null;
}) {
  type Role = "user" | "assistant" | "tool";
  type Msg = { role: Role; content: string; toolCalls?: LlmToolCall[] };
  const [st, setSt] = useState<{ running: boolean; ready: boolean; device?: string; logs?: string[] } | null>(null);
  const [busy, setBusy] = useState(false);
  const [ctlErr, setCtlErr] = useState<string | null>(null);
  const [system, setSystem] = useState("");
  const [messages, setMessages] = useState<Msg[]>([{ role: "user", content: "" }]);
  const [streaming, setStreaming] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  // Compute target (VM-only for LLM): provider + GPU list + vLLM args, picked in the
  // Run-on / Pod cards (TryItCompute) above and passed verbatim to playgroundStart.
  const [compute, setCompute] = useState<ComputeChoice>(() => defaultCompute({
    llm: true, trainedOnVm: true, runProviderId,
    pins: (visibleDevices ?? "").split(",").map((s) => s.trim()).filter(Boolean),
  }));
  const [toolsJson, setToolsJson] = useState("");
  const [toolChoice, setToolChoice] = useState("auto");
  const [chatErr, setChatErr] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const poll = useCallback(async () => {
    try { setSt(await gateway.playgroundStatus(runId)); } catch { /* transient */ }
  }, [runId]);
  useEffect(() => { poll(); }, [poll]);
  // The LLM load is long (download + merge + first-time vLLM venv build + serve) —
  // keep polling the step log while it's running-but-not-ready.
  useEffect(() => {
    if (!st?.running || st.ready) return;
    const t = setInterval(poll, 5000);
    return () => clearInterval(t);
  }, [st, poll]);
  useEffect(() => { scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight }); }, [messages]);

  async function ctl(fn: () => Promise<unknown>) {
    setBusy(true); setCtlErr(null);
    try { await fn(); await poll(); }
    catch (e) { setCtlErr(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }

  const addMsg = () => setMessages((m) => [...m, { role: "user", content: "" }]);
  const updateMsg = (i: number, patch: Partial<Msg>) =>
    setMessages((m) => m.map((x, idx) => (idx === i ? { ...x, ...patch } : x)));
  const removeMsg = (i: number) => setMessages((m) => m.filter((_, idx) => idx !== i));

  // Send the current conversation (system + every non-empty message, with its chosen
  // role) and stream the model's reply into a freshly-appended assistant message.
  async function run() {
    if (streaming) return;
    const hist: { role: string; content: string }[] = [
      ...(system.trim() ? [{ role: "system", content: system.trim() }] : []),
      ...messages.filter((m) => m.content.trim()).map((m) => ({ role: m.role, content: m.content })),
    ];
    if (hist.length === 0) { setChatErr("add at least one message"); return; }
    // Optional tool catalog (OpenAI-shape JSON array).
    let tools: unknown[] | undefined;
    if (toolsJson.trim()) {
      try {
        const parsed = JSON.parse(toolsJson);
        if (!Array.isArray(parsed)) throw new Error("tools must be a JSON array");
        tools = parsed;
      } catch (e) { setChatErr(`tools: ${(e as Error).message}`); return; }
    }
    setStreaming(true); setChatErr(null);
    setMessages((m) => [...m, { role: "assistant", content: "" }]);
    const ctrl = new AbortController(); abortRef.current = ctrl;
    const updateLast = (fn: (m: Msg) => Msg) => setMessages((arr) => {
      const copy = arr.slice(); const last = copy[copy.length - 1];
      if (last && last.role === "assistant") copy[copy.length - 1] = fn(last);
      return copy;
    });
    try {
      const res = await gateway.playgroundChatStream(runId,
        { messages: hist, temperature, max_tokens: maxTokens,
          ...(tools ? { tools, tool_choice: toolChoice } : {}) }, ctrl.signal);
      if (!res.ok || !res.body) {
        const t = await res.text().catch(() => "");
        throw new Error(`chat failed (${res.status})${t ? `: ${t.slice(0, 300)}` : ""}`);
      }
      const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
      while (true) {
        const { value, done } = await reader.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx: number;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
          for (const line of frame.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data || data === "[DONE]") continue;
            try {
              const j = JSON.parse(data);
              if (j.error) { setChatErr(String(j.error)); continue; }
              const delta = j?.choices?.[0]?.delta;
              if (typeof delta?.content === "string" && delta.content)
                updateLast((m) => ({ ...m, content: m.content + delta.content }));
              if (Array.isArray(delta?.tool_calls))
                updateLast((m) => ({ ...m, toolCalls: mergeToolCallDeltas(m.toolCalls ?? [], delta.tool_calls) }));
            } catch { /* ignore keepalives / non-JSON */ }
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") setChatErr(e instanceof Error ? e.message : String(e));
    } finally { setStreaming(false); abortRef.current = null; }
  }

  function stop() { abortRef.current?.abort(); setStreaming(false); }

  const dot = !st?.running ? "bg-muted-foreground/50" : st.ready ? "bg-emerald-500" : "bg-amber-500 animate-pulse";
  const label = !st?.running ? "not loaded" : st.ready ? `ready${st.device ? ` · GPU ${st.device}` : ""}` : "loading…";

  return (
    <div className="space-y-5">
      <TryItCompute value={compute} onChange={setCompute} disabled={!!st?.running} llm
        runProviderId={runProviderId} visibleDevices={visibleDevices} />
      <Card>
      <CardHeader className="pb-2"><CardTitle className="text-sm">Try it — chat (vLLM, eager)</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-muted/20 px-3 py-2 text-xs">
          <span className="font-medium">vLLM server</span>
          <span className={cn("inline-block h-2 w-2 rounded-full", dot)} />
          <span className="text-muted-foreground">{label}</span>
          <span className="hidden text-[11px] text-muted-foreground md:inline">— download LoRA → merge → save → serve (first load builds the vLLM venv; ~15–20 min)</span>
          <div className="ml-auto flex items-center gap-2">
            {busy && <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />}
            {!st?.running ? (
              <Button type="button" variant="outline" className="h-7 text-xs" disabled={busy || !String(compute.gpu ?? "").trim()}
                onClick={() => ctl(() => gateway.playgroundStart(runId, compute))}>Load model</Button>
            ) : (
              <Button type="button" variant="outline" className="h-7 text-xs" disabled={busy}
                onClick={() => ctl(() => gateway.playgroundStop(runId))}>Unload</Button>
            )}
          </div>
          {ctlErr && <span className="w-full text-destructive">{ctlErr}</span>}
          {st?.running && (st.logs?.length ?? 0) > 0 && (
            <div className="terminal-block max-h-48 w-full overflow-y-auto rounded-md border border-border bg-zinc-950 p-2 font-mono text-[10px] leading-snug text-zinc-300">
              {st.logs!.map((l, i) => (
                <div key={i} className={
                  l.includes("✅") ? "text-emerald-300"
                    : (l.includes("ERROR") || l.includes("❌")) ? "text-red-300"
                    : l.includes("[playground]") ? "text-sky-300" : "text-zinc-400"
                }>{l}</div>
              ))}
            </div>
          )}
        </div>

        <div className="grid gap-2 sm:grid-cols-[1fr_auto_auto] sm:items-end">
          <div>
            <label className="pr-3 text-[11px] text-muted-foreground">System prompt (optional)</label>
            <Input value={system} onChange={(e) => setSystem(e.target.value)} placeholder="You are a helpful assistant." className="h-8 text-sm" />
          </div>
          <div>
            <label className="pr-3 text-[11px] text-muted-foreground">Temp</label>
            <Input type="number" min={0} max={2} step={0.1} value={temperature}
              onChange={(e) => setTemperature(Math.max(0, Math.min(2, Number(e.target.value) || 0)))}
              className="h-8 w-20 font-mono text-sm" />
          </div>
          <div>
            <label className="pr-3 text-[11px] text-muted-foreground">Max tokens</label>
            <Input type="number" min={1} max={8192} step={64} value={maxTokens}
              onChange={(e) => setMaxTokens(Math.max(1, Number(e.target.value) || 1))}
              className="h-8 w-24 font-mono text-sm" />
          </div>
        </div>

        {/* Tool calling (function specs) — like the SyntheticGen playground. The model
            emits tool_calls (rendered under the assistant message); add a `tool` message
            with the result to continue. Needs the server loaded with a tool-call parser. */}
        <details className="rounded-md border border-border">
          <summary className="cursor-pointer select-none px-3 py-1.5 text-xs text-muted-foreground">
            Tools (function calling){toolsJson.trim() ? " · configured" : " · none"}
          </summary>
          <div className="space-y-2 border-t border-border p-3">
            <div className="flex flex-wrap items-center gap-2">
              <Button type="button" variant="outline" className="h-7 text-xs" onClick={() => setToolsJson(LLM_SAMPLE_TOOLS)}>Sample</Button>
              <Button type="button" variant="ghost" className="h-7 text-xs" disabled={!toolsJson.trim()} onClick={() => setToolsJson("")}>Clear</Button>
              <label className="pl-2 text-[11px] text-muted-foreground">tool_choice</label>
              <Select value={toolChoice} onValueChange={setToolChoice}>
                <SelectTrigger className="h-7 w-28 text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">auto</SelectItem>
                  <SelectItem value="required">required</SelectItem>
                  <SelectItem value="none">none</SelectItem>
                </SelectContent>
              </Select>
              <span className="text-[10px] text-muted-foreground">
                OpenAI tool specs (JSON array). Load with <span className="font-mono">--enable-auto-tool-choice --tool-call-parser …</span>
              </span>
            </div>
            <Textarea value={toolsJson} onChange={(e) => setToolsJson(e.target.value)} rows={6}
              placeholder={'[{"type":"function","function":{"name":"get_weather","parameters":{...}}}]'}
              className="bg-transparent font-mono text-[11px] dark:bg-transparent" />
          </div>
        </details>

        {/* Editable conversation — each message has a role (user / assistant / tool).
            Run streams the model's reply into a new assistant message you can then keep
            editing for multi-turn / few-shot / tool-result replay. */}
        <div ref={scrollRef} className="max-h-[460px] space-y-2 overflow-y-auto rounded-md border border-border p-3">
          {messages.length === 0 && (
            <div className="text-xs text-muted-foreground">No messages — add one to start.</div>
          )}
          {messages.map((m, i) => {
            const streamingLast = streaming && i === messages.length - 1 && m.role === "assistant";
            return (
              <div key={i} className="space-y-1">
                <div className="flex items-start gap-2">
                  <Select value={m.role} onValueChange={(v) => updateMsg(i, { role: v as Role })} disabled={streaming}>
                    <SelectTrigger className="h-8 w-28 shrink-0 text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="user">user</SelectItem>
                      <SelectItem value="assistant">assistant</SelectItem>
                      <SelectItem value="tool">tool</SelectItem>
                    </SelectContent>
                  </Select>
                  <Textarea value={m.content} onChange={(e) => updateMsg(i, { content: e.target.value })}
                    rows={2} disabled={streaming}
                    placeholder={m.role === "tool" ? "tool result…" : m.role === "assistant" ? "assistant message…" : "user message…"}
                    onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); run(); } }}
                    className={cn("flex-1 bg-transparent text-sm dark:bg-transparent", streamingLast && "animate-pulse")} />
                  <Button type="button" variant="ghost" className="h-8 shrink-0 px-2" disabled={streaming}
                    onClick={() => removeMsg(i)} aria-label="remove message"><X className="h-4 w-4" /></Button>
                </div>
                {m.toolCalls && m.toolCalls.length > 0 && (
                  <div className="ml-[120px] space-y-1">
                    {m.toolCalls.map((tc) => (
                      <div key={tc.index} className="rounded border border-purple-500/30 bg-purple-500/5 p-1.5">
                        <div className="font-mono text-[10px] font-semibold">
                          {tc.name || "(no name yet)"}
                          {tc.id && <span className="ml-2 font-normal text-muted-foreground">id: {tc.id}</span>}
                        </div>
                        <pre className="overflow-x-auto whitespace-pre-wrap break-all font-mono text-[10px]">{tc.argsBuf || "(no args yet)"}</pre>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
        {chatErr && <div className="text-xs text-destructive">{chatErr}</div>}

        <div className="flex flex-wrap items-center gap-2">
          <Button type="button" variant="outline" className="h-8 text-xs" disabled={streaming} onClick={addMsg}>
            + Add message
          </Button>
          <span className="text-[11px] text-muted-foreground">⌘/Ctrl+Enter to run</span>
          <div className="ml-auto flex items-center gap-2">
            {!streaming && messages.some((m) => m.content.trim()) && (
              <Button type="button" variant="ghost" className="h-8"
                onClick={() => setMessages([{ role: "user", content: "" }])}>Clear</Button>
            )}
            {streaming ? (
              <Button type="button" variant="outline" onClick={stop}>Stop</Button>
            ) : (
              <Button type="button" onClick={run} disabled={!st?.ready || !messages.some((m) => m.content.trim())}>Run</Button>
            )}
          </div>
        </div>
      </CardContent>
      </Card>
    </div>
  );
}

// Try-it playground (TTS) — type text, pick a GPU the run used (or CPU), and
// synthesize speech with the finetuned model on the run's VM (over SSH), then play it.
function TtsPlaygroundTab({ runId, visibleDevices, runProviderId, trainedOnVm, gpuType, gpuCount }: {
  runId: string; visibleDevices: string | null;
  runProviderId: string | null; trainedOnVm: boolean;
  gpuType: string | null; gpuCount: number | null;
}) {
  const [compute, setCompute] = useState<ComputeChoice>(() => defaultCompute({
    trainedOnVm, runProviderId, gpuChoice: gpuTypeToChoice(gpuType), gpuCount,
    pins: (visibleDevices ?? "").split(",").map((s) => s.trim()).filter(Boolean),
  }));
  const [loaded, setLoaded] = useState(false);
  const [text, setText] = useState("");
  const [speaker, setSpeaker] = useState("");
  const [busy, setBusy] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [device, setDevice] = useState<string | undefined>();
  const [logs, setLogs] = useState<string[]>([]);
  const [prompt, setPrompt] = useState<string | null>(null);
  const [genText, setGenText] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function onSynthesize() {
    if (!text.trim()) return;
    setBusy(true);
    setErr(null);
    setLogs([]);
    setPrompt(null);
    setGenText(null);
    if (audioUrl) { URL.revokeObjectURL(audioUrl); setAudioUrl(null); }
    try {
      const r = await gateway.synthesizeTrainingRun(runId, text.trim(), {
        speaker: speaker.trim() || undefined, gpu: compute.gpu,
      });
      setAudioUrl(r.url);
      setDevice(r.device);
      setLogs(r.logs ?? []);
      setPrompt(r.prompt ?? null);
      setGenText(r.genText ?? null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <TryItCompute value={compute} onChange={setCompute} disabled={loaded}
        runProviderId={runProviderId} visibleDevices={visibleDevices} />
      <Card>
      <CardHeader className="pb-2"><CardTitle className="text-sm">Try it — synthesize speech</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <p className="text-xs text-muted-foreground">
          Load the model, then type text (optionally a speaker name, as the data was packed) and
          synthesize. A cloud pod spins up on demand (first load ~10 min) and auto-stops when idle; a VM
          keeps the model resident. The first request downloads the model onto the box.
        </p>
        <PersistentControls runId={runId} compute={compute} onRunningChange={setLoaded} />
        <div className="flex flex-col gap-1">
          <span className="text-xs text-muted-foreground">text</span>
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={3}
            placeholder="Type something to speak…"
            className="text-sm"
          />
        </div>
        <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">speaker (optional)</span>
            <Input
              value={speaker}
              onChange={(e) => setSpeaker(e.target.value)}
              placeholder="(default)"
              className="h-8 w-[200px] text-sm"
            />
          </div>
          <Button type="button" onClick={onSynthesize} disabled={busy || !text.trim()} className="ml-auto">
            {busy
              ? <><Loader2 className="h-4 w-4 animate-spin" /> Synthesizing…</>
              : <><AudioLines className="h-4 w-4" /> Synthesize</>}
          </Button>
        </div>
        {err && <p className="text-sm text-destructive">{err}</p>}
        {audioUrl && (
          <div className="space-y-1.5">
            <div className="text-xs text-muted-foreground">Generated audio{device ? ` · ran on ${device}` : ""}</div>
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <audio controls src={audioUrl} className="w-full" />
          </div>
        )}
        {prompt && (
          <div className="space-y-1">
            <div className="text-xs font-medium text-muted-foreground">Prompt (fed to the model)</div>
            <pre className="max-h-24 overflow-y-auto whitespace-pre-wrap break-all rounded-md border border-border bg-muted/30 p-2 font-mono text-[11px]">{prompt}</pre>
          </div>
        )}
        {genText && (
          <details className="rounded-md border border-border">
            <summary className="cursor-pointer select-none px-3 py-1.5 text-xs text-muted-foreground">
              Generated tokens (before NeuCodec)
            </summary>
            <pre className="max-h-48 overflow-y-auto whitespace-pre-wrap break-all border-t border-border bg-muted/30 p-2 font-mono text-[11px]">{genText}</pre>
          </details>
        )}
        <TryItLogs lines={logs} />
      </CardContent>
      </Card>
    </div>
  );
}

function LogsTab({ lines, status }: { lines: string[]; status: string }) {
  const endRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const terminal = ["done", "failed", "cancelled"].includes(status);

  // Only stick to the bottom when auto-scroll is on — so you can scroll up to
  // read without being yanked back down as new lines stream in.
  useEffect(() => {
    if (autoScroll) endRef.current?.scrollIntoView({ block: "end" });
  }, [lines, autoScroll]);

  return (
    <div className="space-y-2">
      <label className="flex w-fit cursor-pointer select-none items-center gap-2 text-xs text-muted-foreground">
        <input
          type="checkbox"
          checked={autoScroll}
          onChange={(e) => setAutoScroll(e.target.checked)}
          className="h-3.5 w-3.5 accent-primary"
        />
        Auto-scroll to latest{!autoScroll && lines.length > 0 ? " (paused)" : ""}
      </label>
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
    </div>
  );
}

// Human-readable byte size (B/KB/MB/GB/TB) — files range from <1 KB configs to
// multi-GB safetensors, so a fixed KB unit is unreadable.
function fmtBytes(n?: number | null): string {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  const u = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${u[i]}`;
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
            <Download className="h-3.5 w-3.5" /> {fmtBytes(f.size)}
          </a>
        </li>
      ))}
    </ul>
  );
}
