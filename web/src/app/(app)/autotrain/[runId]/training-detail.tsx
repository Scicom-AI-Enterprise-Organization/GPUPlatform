"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { AudioLines, Check, ChevronDown, Copy, Download, ExternalLink, Loader2, Pencil, RotateCcw, Trash2, Upload, X, XCircle } from "lucide-react";
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
import type { DatasetRecord, GlobalEnvRecord, TrainingEpoch, TrainingFile, TrainingGpu, TrainingGpuSample, TrainingRunRecord, TrainingStep, TrainingTrial } from "@/lib/types";

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
  // A post-train Label export is running in the background — the run itself is
  // already "done", so show "exporting to Label" instead and keep polling.
  const exporting = run.result_json?.label_export?.status === "running";

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
  // Try-it playground: a finished run on a VM (inference runs on that VM) — ASR
  // transcribes an uploaded clip; TTS synthesizes speech from text.
  const canTryIt =
    run.status === "done" &&
    run.provider_kind === "vm" &&
    !!run.result_json?.artifact?.s3_uri;
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

  // On-demand Label-platform export (retry) for finished TTS runs — the run may
  // have finished without label creds, or you may want to re-export. Prefilled
  // from the run's config; the token is never stored back unencrypted.
  const lcfg = (run.config_json ?? {}) as Record<string, unknown>;
  const [labelOpen, setLabelOpen] = useState(false);
  // URL + token can each be typed in or referenced from the Secrets page (GlobalEnv),
  // mirroring the autotrain create form. Prefilled from the run's config.
  const [labelUrlMode, setLabelUrlMode] = useState<"paste" | "secret">(
    typeof lcfg.label_base_url_secret === "string" && lcfg.label_base_url_secret ? "secret" : "paste",
  );
  const [labelUrl, setLabelUrl] = useState(
    typeof lcfg.label_base_url === "string" && lcfg.label_base_url ? lcfg.label_base_url : "http://localhost:3002",
  );
  const [labelUrlSecret, setLabelUrlSecret] = useState(
    typeof lcfg.label_base_url_secret === "string" ? lcfg.label_base_url_secret : "",
  );
  const [labelTokenMode, setLabelTokenMode] = useState<"paste" | "secret">(
    typeof lcfg.label_token_secret === "string" && lcfg.label_token_secret ? "secret" : "paste",
  );
  const [labelToken, setLabelToken] = useState("");
  const [labelTokenSecret, setLabelTokenSecret] = useState(
    typeof lcfg.label_token_secret === "string" ? lcfg.label_token_secret : "",
  );
  const [labelSecrets, setLabelSecrets] = useState<GlobalEnvRecord[]>([]);
  const [labelProject, setLabelProject] = useState(
    typeof lcfg.label_project_name === "string" ? lcfg.label_project_name : "",
  );
  const [labelSamples, setLabelSamples] = useState(
    typeof lcfg.label_samples === "number" ? lcfg.label_samples : 32,
  );
  const [labelAxes, setLabelAxes] = useState(
    Array.isArray(lcfg.label_mos_axes) ? (lcfg.label_mos_axes as unknown[]).map(String).join(", ") : "Naturalness, Intelligibility, Noise",
  );
  const [labelBusy, setLabelBusy] = useState(false);
  const [labelErr, setLabelErr] = useState<string | null>(null);
  const [labelDone, setLabelDone] = useState(false);

  // Load Secrets-page keys the first time the export dialog opens.
  useEffect(() => {
    if (!labelOpen || labelSecrets.length) return;
    gateway.listGlobalEnv().then(setLabelSecrets).catch(() => {});
  }, [labelOpen, labelSecrets.length]);

  const labelUrlOk = labelUrlMode === "paste" ? !!labelUrl.trim() : !!labelUrlSecret;
  const labelTokenOk = labelTokenMode === "paste" ? !!labelToken.trim() : !!labelTokenSecret;

  async function submitLabelExport() {
    setLabelBusy(true);
    setLabelErr(null);
    try {
      await gateway.retryLabelExport(run.id, {
        base_url: labelUrlMode === "paste" ? (labelUrl.trim() || undefined) : undefined,
        base_url_secret: labelUrlMode === "secret" ? (labelUrlSecret || null) : null,
        token: labelTokenMode === "paste" ? (labelToken.trim() || undefined) : undefined,
        token_secret: labelTokenMode === "secret" ? (labelTokenSecret || null) : null,
        project_name: labelProject.trim() || null,
        samples: labelSamples,
        mos_axes: labelAxes.split(",").map((s) => s.trim()).filter(Boolean),
      });
      setLabelDone(true);
    } catch (e) {
      setLabelErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLabelBusy(false);
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
                <Loader2 className="mr-1 h-3 w-3 animate-spin" /> exporting to Label
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
          {!terminal && (
            <Button variant="outline" size="sm" onClick={onTerminate} disabled={busy} className="text-destructive">
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <XCircle className="h-4 w-4" />} Terminate
            </Button>
          )}
          {run.task_type === "tts" && run.status === "done" && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => { setLabelErr(null); setLabelDone(false); setLabelOpen(true); }}
              title="Synthesize sample clips and create a Label-platform recording+MOS project"
            >
              <Upload className="h-4 w-4" /> Export to Label
            </Button>
          )}
          <Button variant="outline" size="sm" onClick={onDelete} disabled={busy}>
            <Trash2 className="h-4 w-4" /> Delete
          </Button>
        </div>
      </div>

        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-5">
          <Kpi label="Status" value={exporting ? "exporting to Label" : run.status} />
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

        <Tabs value={tab} onValueChange={onTab} className="mt-4">
          <TabsList variant="line" className="bg-transparent">
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="logs">Logs</TabsTrigger>
            <TabsTrigger value="files">Files</TabsTrigger>
            <TabsTrigger value="config">Config</TabsTrigger>
            {canTryIt && <TabsTrigger value="tryit">Try it</TabsTrigger>}
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

      {run.result_json?.label_project && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">
              Label project created
              {run.result_json.label_project.count != null ? ` · ${run.result_json.label_project.count} clips` : ""}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap items-center gap-x-8 gap-y-2 text-sm">
            <a
              href={run.result_json.label_project.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 font-medium text-primary hover:underline"
            >
              Open in Label
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
            {run.result_json.label_project.dataset_id && (
              <a
                href={`/datasets/${run.result_json.label_project.dataset_id}`}
                className="inline-flex items-center gap-1.5 text-muted-foreground hover:text-foreground hover:underline"
              >
                Linked dataset <span className="font-mono">{run.result_json.label_project.dataset_id}</span>
              </a>
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

      <Tabs value={tab} onValueChange={onTab} className="!block">
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
              ? <TtsPlaygroundTab runId={run.id} visibleDevices={run.visible_devices ?? null} />
              : <PlaygroundTab runId={run.id} visibleDevices={run.visible_devices ?? null} />}
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

      <Dialog open={labelOpen} onOpenChange={(o) => { if (!labelBusy) setLabelOpen(o); }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Export to Label platform</DialogTitle>
            <DialogDescription>
              Synthesize {labelSamples} clip{labelSamples === 1 ? "" : "s"} from this run&apos;s trained model and
              create a Label-platform recording project with MOS rating, seeded with them. Runs in the background;
              watch the Logs tab for progress.
            </DialogDescription>
          </DialogHeader>
          {labelDone ? (
            <p className="flex items-center gap-2 py-2 text-sm text-emerald-600 dark:text-emerald-400">
              <Check className="h-4 w-4" /> Export started — the status shows “exporting to Label” and the
              synthesis streams to the Logs tab; an “Open in Label” link appears on this page when it finishes.
            </p>
          ) : (
            <div className="space-y-3">
              <div className="space-y-1.5">
                <div className="flex items-center gap-3">
                  <label className="text-xs uppercase tracking-wide text-muted-foreground">Label platform URL</label>
                  <div className="inline-flex overflow-hidden rounded-md border border-border text-xs">
                    {(["paste", "secret"] as const).map((m) => (
                      <button key={m} type="button" onClick={() => setLabelUrlMode(m)}
                        className={cn("px-2.5 py-1 transition-colors",
                          labelUrlMode === m ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground")}>
                        {m === "paste" ? "Paste" : "From secret"}
                      </button>
                    ))}
                  </div>
                </div>
                {labelUrlMode === "paste" ? (
                  <Input className="font-mono" value={labelUrl} placeholder="http://localhost:3002"
                    onChange={(e) => setLabelUrl(e.target.value)} />
                ) : (
                  <Select value={labelUrlSecret} onValueChange={setLabelUrlSecret}>
                    <SelectTrigger><SelectValue placeholder={labelSecrets.length ? "Choose a secret" : "No secrets configured"} /></SelectTrigger>
                    <SelectContent>
                      {labelSecrets.map((s) => (
                        <SelectItem key={s.key} value={s.key}>{s.key}{s.value_preview ? ` — ${s.value_preview}` : ""}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </div>
              <div className="space-y-1.5">
                <div className="flex items-center gap-3">
                  <label className="text-xs uppercase tracking-wide text-muted-foreground">API token</label>
                  <div className="inline-flex overflow-hidden rounded-md border border-border text-xs">
                    {(["paste", "secret"] as const).map((m) => (
                      <button key={m} type="button" onClick={() => setLabelTokenMode(m)}
                        className={cn("px-2.5 py-1 transition-colors",
                          labelTokenMode === m ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground")}>
                        {m === "paste" ? "Paste" : "From secret"}
                      </button>
                    ))}
                  </div>
                </div>
                {labelTokenMode === "paste" ? (
                  <>
                    <Input type="password" className="font-mono" value={labelToken} placeholder="lpat_…"
                      onChange={(e) => setLabelToken(e.target.value)} />
                    <p className="text-[11px] text-muted-foreground">Admin personal access token. Stored encrypted on the run.</p>
                  </>
                ) : (
                  <Select value={labelTokenSecret} onValueChange={setLabelTokenSecret}>
                    <SelectTrigger><SelectValue placeholder={labelSecrets.some((s) => s.is_secret) ? "Choose a secret" : "No secrets configured"} /></SelectTrigger>
                    <SelectContent>
                      {labelSecrets.filter((s) => s.is_secret).map((s) => (
                        <SelectItem key={s.key} value={s.key}>{s.key}{s.value_preview ? ` — ${s.value_preview}` : ""}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label className="text-xs uppercase tracking-wide text-muted-foreground">Project name</label>
                  <Input value={labelProject} placeholder={`${run.name}-eval`}
                    onChange={(e) => setLabelProject(e.target.value)} />
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs uppercase tracking-wide text-muted-foreground">Samples</label>
                  <Input type="number" min={1} value={labelSamples}
                    onChange={(e) => setLabelSamples(Math.max(1, Number.parseInt(e.target.value, 10) || 1))} />
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-xs uppercase tracking-wide text-muted-foreground">MOS axes</label>
                <Input value={labelAxes} placeholder="Naturalness, Intelligibility, Noise"
                  onChange={(e) => setLabelAxes(e.target.value)} />
              </div>
            </div>
          )}
          <DialogFooter>
            {labelErr && <p className="mr-auto text-sm text-destructive">{labelErr}</p>}
            <Button variant="outline" onClick={() => setLabelOpen(false)} disabled={labelBusy}>
              {labelDone ? "Close" : "Cancel"}
            </Button>
            {!labelDone && (
              <Button onClick={submitLabelExport} disabled={labelBusy || !labelUrlOk || !labelTokenOk}>
                {labelBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                Start export
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

// Try-it playground — upload a clip, pick a GPU the run used (or CPU), and
// transcribe it with the finetuned model on the run's VM (over SSH).
function PlaygroundTab({ runId, visibleDevices }: { runId: string; visibleDevices: string | null }) {
  const gpuIds = (visibleDevices ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  const [file, setFile] = useState<File | null>(null);
  const [gpu, setGpu] = useState<string>(gpuIds[0] ?? "auto");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ text: string; device?: string; logs?: string[] } | null>(null);
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
        <PersistentControls runId={runId} gpu={gpu} />
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
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">run on</span>
            <Select value={gpu} onValueChange={setGpu}>
              <SelectTrigger className="h-8 w-[180px] text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {gpuIds.map((g) => <SelectItem key={g} value={g} className="text-xs">GPU {g}</SelectItem>)}
                <SelectItem value="auto" className="text-xs">Auto (most-free GPU)</SelectItem>
                <SelectItem value="cpu" className="text-xs">CPU</SelectItem>
              </SelectContent>
            </Select>
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
        {result && <TryItLogs lines={result.logs ?? []} />}
      </CardContent>
    </Card>
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

// Persistent worker controls — load the model once on the VM (resident on the GPU)
// so try-it requests skip the per-call model load, with Load / Restart / Unload.
function PersistentControls({ runId, gpu }: { runId: string; gpu: string }) {
  const [st, setSt] = useState<{ running: boolean; ready: boolean; device?: string; logs?: string[] } | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const poll = useCallback(async () => {
    try { setSt(await gateway.playgroundStatus(runId)); } catch { /* transient */ }
  }, [runId]);
  useEffect(() => { poll(); }, [poll]);
  // While loading, poll until ready (the model load takes ~10-15s).
  useEffect(() => {
    if (!st?.running || st.ready) return;
    const t = setInterval(poll, 3000);
    return () => clearInterval(t);
  }, [st, poll]);

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
            onClick={() => act(() => gateway.playgroundStart(runId, gpu))}>Load model</Button>
        ) : (
          <>
            <Button type="button" variant="outline" className="h-7 text-xs" disabled={busy}
              onClick={() => act(async () => { await gateway.playgroundStop(runId); await gateway.playgroundStart(runId, gpu); })}>Restart</Button>
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

// Try-it playground (TTS) — type text, pick a GPU the run used (or CPU), and
// synthesize speech with the finetuned model on the run's VM (over SSH), then play it.
function TtsPlaygroundTab({ runId, visibleDevices }: { runId: string; visibleDevices: string | null }) {
  const gpuIds = (visibleDevices ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  const [text, setText] = useState("");
  const [speaker, setSpeaker] = useState("");
  const [gpu, setGpu] = useState<string>(gpuIds[0] ?? "auto");
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
        speaker: speaker.trim() || undefined, gpu,
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
    <Card>
      <CardHeader className="pb-2"><CardTitle className="text-sm">Try it — synthesize speech</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <p className="text-xs text-muted-foreground">
          Runs the finetuned TTS model on this run&apos;s VM. Type text, optionally a speaker name (as the
          data was packed), pick a GPU the run used (or CPU), and synthesize. The first request downloads the
          model onto the VM, so it can take a little longer.
        </p>
        <PersistentControls runId={runId} gpu={gpu} />
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
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">run on</span>
            <Select value={gpu} onValueChange={setGpu}>
              <SelectTrigger className="h-8 w-[180px] text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {gpuIds.map((g) => <SelectItem key={g} value={g} className="text-xs">GPU {g}</SelectItem>)}
                <SelectItem value="auto" className="text-xs">Auto (most-free GPU)</SelectItem>
                <SelectItem value="cpu" className="text-xs">CPU</SelectItem>
              </SelectContent>
            </Select>
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
