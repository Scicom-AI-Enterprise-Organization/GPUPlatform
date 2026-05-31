"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  Check,
  Cpu,
  Database,
  FlaskConical,
  Loader2,
  RefreshCw,
  Server,
  Sparkles,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { NumberField } from "@/components/ui/number-field";
import { SearchableSelect } from "@/components/ui/searchable-select";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AvailabilityBadge } from "@/components/availability-badge";
import { useGpuAvailability } from "@/lib/use-gpu-availability";
import { gateway } from "@/lib/gateway";
import { cn } from "@/lib/utils";
import type {
  CreateTrainingRunRequest,
  DatasetRecord,
  GpuTypeOption,
  ProviderRecord,
  StorageRecord,
  VmAvailability,
} from "@/lib/types";

const WHISPER_MODELS = [
  "openai/whisper-tiny",
  "openai/whisper-base",
  "openai/whisper-small",
  "openai/whisper-medium",
  "openai/whisper-large-v3",
  "openai/whisper-large-v3-turbo",
];
const DEFAULT_WHISPER = "openai/whisper-large-v3-turbo";
const TTS_BASE_MODELS = [
  "Scicom-intl/Multilingual-Expressive-TTS-1.7B",
  "Scicom-intl/Multilingual-Expressive-TTS-0.6B",
  "Scicom-intl/Multilingual-TTS-1.7B-Base",
  "Scicom-intl/Multilingual-TTS-0.6B-Base",
];
const DEFAULT_TTS_TOKENIZER = "Scicom-intl/Multilingual-Expressive-TTS-1.7B";
const CUSTOM = "__custom__";
const AUTO_SPLIT = "__auto__";

const GPU_COUNT_CHOICES = [1, 2, 4, 8] as const;

// Parse a pasted env block into a dict. Accepts `KEY=value` and
// `export KEY=value`; skips blanks, comments, and non-KEY=value lines (mkdir …).
function parseEnvVars(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const raw of text.split("\n")) {
    let line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    if (line.startsWith("export ")) line = line.slice("export ".length).trim();
    const eq = line.indexOf("=");
    if (eq <= 0) continue;
    const key = line.slice(0, eq).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) continue;
    let val = line.slice(eq + 1).trim();
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1);
    }
    out[key] = val;
  }
  return out;
}

function parseCsvNums(s: string, asInt: boolean): number[] {
  return s
    .split(/[,\s]+/)
    .map((x) => x.trim())
    .filter(Boolean)
    .map((x) => (asInt ? parseInt(x, 10) : parseFloat(x)))
    .filter((n) => Number.isFinite(n));
}

// Rough capacity estimate (mirrors the benchmark/serverless forms).
function capacityHint(vramGb: number, count: number): string {
  const total = vramGb * count;
  const weights = total * 0.55;
  const fp16 = weights / 2;
  const q4 = weights / 0.6;
  const r = (b: number) => (b >= 100 ? `${Math.round(b / 10) * 10}B` : `${Math.round(b)}B`);
  const totalStr = total >= 100 ? `${Math.round(total)} GB` : `${total} GB`;
  return `${totalStr} VRAM${count > 1 ? ` · TP=${count} sharding` : ""} · fits ~${r(fp16)} FP16 / ~${r(q4)} 4-bit (KV-cache budgeted)`;
}

// Fallback until the live catalog (/compute/runpod/gpu-types) lands.
const RUNPOD_GPU_FALLBACK: GpuTypeOption[] = [
  { id: "NVIDIA RTX A4000", label: "RTX A4000", vram_gb: 16, hint: "16 GB · cheap baseline" },
  { id: "NVIDIA RTX A5000", label: "RTX A5000", vram_gb: 24, hint: "24 GB" },
  { id: "NVIDIA RTX A6000", label: "RTX A6000", vram_gb: 48, hint: "48 GB" },
  { id: "NVIDIA GeForce RTX 4090", label: "RTX 4090", vram_gb: 24, hint: "24 GB · consumer" },
  { id: "NVIDIA L40", label: "L40", vram_gb: 48, hint: "48 GB" },
  { id: "NVIDIA L40S", label: "L40S", vram_gb: 48, hint: "48 GB · faster L40" },
  { id: "NVIDIA A100 80GB PCIe", label: "A100 80GB", vram_gb: 80, hint: "datacenter" },
  { id: "NVIDIA H100 80GB HBM3", label: "H100 80GB", vram_gb: 80, hint: "fastest" },
];

type VmAvailState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ok"; data: VmAvailability }
  | { status: "error"; message: string };

export function TrainingForm() {
  const router = useRouter();
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [storages, setStorages] = useState<StorageRecord[]>([]);
  const [providers, setProviders] = useState<ProviderRecord[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // task + model + data
  const [taskType, setTaskType] = useState<"asr" | "tts">("asr");
  const [name, setName] = useState("whisper-finetune");
  const [modelChoice, setModelChoice] = useState(DEFAULT_WHISPER);
  const [customModel, setCustomModel] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [testDatasetId, setTestDatasetId] = useState(AUTO_SPLIT);
  const [evalSplitPct, setEvalSplitPct] = useState(10);
  // TTS-only (Qwen3 + NeuCodec)
  const [ttsTokenizer, setTtsTokenizer] = useState(DEFAULT_TTS_TOKENIZER);
  const [blockSize, setBlockSize] = useState(10240);
  const [packSeq, setPackSeq] = useState(4096);
  const [defaultSpeaker, setDefaultSpeaker] = useState("speaker");
  const [gradAccum, setGradAccum] = useState(4);
  // training
  const [evalMetric, setEvalMetric] = useState<"wer" | "cer">("wer");
  const [maxEpochs, setMaxEpochs] = useState(3);
  const [patience, setPatience] = useState(1);
  const [batchSize, setBatchSize] = useState(8);
  const [loggingSteps, setLoggingSteps] = useState(10);
  const [learningRate, setLearningRate] = useState("1e-5");
  const [precision, setPrecision] = useState<"fp16" | "bf16" | "fp32">("bf16");
  const [language, setLanguage] = useState("");
  // hyperparameter sweep
  const [sweepOn, setSweepOn] = useState(false);
  const [gpusPerTrial, setGpusPerTrial] = useState(1);
  const [sweepLr, setSweepLr] = useState("");
  const [sweepBatch, setSweepBatch] = useState("");
  const [sweepGradAccum, setSweepGradAccum] = useState("");
  const [sweepEpochs, setSweepEpochs] = useState("");
  const [sweepBlock, setSweepBlock] = useState("");
  // run on (pod card — mirrors benchmark/new)
  const [target, setTarget] = useState<"cloud" | "vm">("cloud");
  const [providerId, setProviderId] = useState(""); // vm provider
  const [runpodProviderId, setRunpodProviderId] = useState(""); // runpod account
  const [gpuType, setGpuType] = useState("NVIDIA L40S");
  const [gpuCount, setGpuCount] = useState(1);
  const [secureCloud, setSecureCloud] = useState(true);
  const [diskGb, setDiskGb] = useState(60);
  const [volumeGb, setVolumeGb] = useState(80);
  const [visibleDevices, setVisibleDevices] = useState("");
  const [envText, setEnvText] = useState("");
  const [gpuOptions, setGpuOptions] = useState<GpuTypeOption[]>(RUNPOD_GPU_FALLBACK);
  // artifacts
  const [storageId, setStorageId] = useState("");
  const [hfPushRepo, setHfPushRepo] = useState("");
  // experiment tracking (creds come from the global Secrets page)
  const [wandbOn, setWandbOn] = useState(false);
  const [wandbProject, setWandbProject] = useState("");
  const [wandbEntity, setWandbEntity] = useState("");
  const [mlflowOn, setMlflowOn] = useState(false);
  const [mlflowUri, setMlflowUri] = useState("");
  const [mlflowExperiment, setMlflowExperiment] = useState("");

  const availability = useGpuAvailability(
    gpuType, gpuCount, target === "cloud", secureCloud ? "SECURE" : "COMMUNITY",
  );

  const [vmAvail, setVmAvail] = useState<VmAvailState>({ status: "idle" });
  const refreshVmAvail = useCallback(async (id: string) => {
    if (!id) {
      setVmAvail({ status: "idle" });
      return;
    }
    setVmAvail({ status: "loading" });
    try {
      setVmAvail({ status: "ok", data: await gateway.getVmAvailability(id) });
    } catch (e) {
      setVmAvail({ status: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }, []);

  useEffect(() => {
    gateway.listDatasets().then(setDatasets).catch(() => {});
    gateway.listStorage().then(setStorages).catch(() => {});
    gateway.listProviders().then(setProviders).catch(() => {});
    gateway
      .listRunpodGpuTypes()
      .then((rows) => {
        if (rows.length === 0) return;
        setGpuOptions(rows);
        setGpuType((cur) => (rows.some((g) => g.id === cur) ? cur : rows[0].id));
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (target === "vm" && providerId) refreshVmAvail(providerId);
    else setVmAvail({ status: "idle" });
  }, [target, providerId, refreshVmAvail]);

  const s3Storages = useMemo(
    () => storages.filter((s) => s.kind === "s3" && s.enabled),
    [storages],
  );
  const vmProviders = useMemo(() => providers.filter((p) => p.kind === "vm"), [providers]);
  const runpodProviders = useMemo(() => providers.filter((p) => p.kind === "runpod"), [providers]);
  const baseModel = modelChoice === CUSTOM ? customModel.trim() : modelChoice;
  const hasStorage = s3Storages.length > 0;
  const isTts = taskType === "tts";
  const MODELS = isTts ? TTS_BASE_MODELS : WHISPER_MODELS;

  function pickTask(t: "asr" | "tts") {
    setTaskType(t);
    setModelChoice(t === "tts" ? TTS_BASE_MODELS[0] : DEFAULT_WHISPER);
    if (t === "tts") {
      setPrecision("bf16");
      setName((n) => (n === "whisper-finetune" ? "tts-finetune" : n));
    } else {
      setName((n) => (n === "tts-finetune" ? "whisper-finetune" : n));
    }
  }

  function buildSweep(): Record<string, number[]> {
    const s: Record<string, number[]> = {};
    const lr = parseCsvNums(sweepLr, false);
    if (lr.length) s.learning_rate = lr;
    const b = parseCsvNums(sweepBatch, true);
    if (b.length) s.batch_size = b;
    const ga = parseCsvNums(sweepGradAccum, true);
    if (ga.length) s.grad_accum = ga;
    const ep = parseCsvNums(sweepEpochs, true);
    if (ep.length) s.max_epochs = ep;
    if (isTts) {
      const bs = parseCsvNums(sweepBlock, true);
      if (bs.length) s.block_size = bs;
    }
    return s;
  }
  const sweepGrid = sweepOn ? buildSweep() : {};
  const trialCount = Object.values(sweepGrid).reduce((acc, vs) => acc * (vs.length || 1), 1);
  const envVars = parseEnvVars(envText);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!name.trim()) return setError("Name is required.");
    if (!baseModel) return setError("Pick or enter a base model.");
    if (!datasetId) return setError("Pick a training dataset.");
    if (!storageId) return setError("Pick an S3 storage for artifacts + logs.");
    if (target === "vm" && !providerId) return setError("Pick a VM provider, or switch to Default cloud.");

    const body: CreateTrainingRunRequest = {
      name: name.trim(),
      dataset_id: datasetId,
      base_model: baseModel,
      task_type: taskType,
      test_dataset_id: isTts ? null : (testDatasetId === AUTO_SPLIT ? null : testDatasetId),
      eval_metric: evalMetric,
      max_epochs: maxEpochs,
      patience: isTts ? 0 : patience,
      eval_split_pct: evalSplitPct,
      batch_size: batchSize,
      grad_accum: gradAccum,
      learning_rate: Number(learningRate) || (isTts ? 2e-5 : 1e-5),
      logging_steps: loggingSteps,
      precision,
      language: isTts ? null : (language.trim() || null),
      ...(isTts ? {
        tokenizer: ttsTokenizer.trim() || DEFAULT_TTS_TOKENIZER,
        block_size: blockSize,
        pack_sequence_length: packSeq,
        default_speaker: defaultSpeaker.trim() || "speaker",
      } : {}),
      ...(sweepOn && Object.keys(sweepGrid).length
        ? { sweep: sweepGrid, gpus_per_trial: gpusPerTrial }
        : {}),
      provider_id: target === "vm" ? providerId : runpodProviderId || null,
      gpu_type: gpuType,
      gpu_count: gpuCount,
      secure_cloud: secureCloud,
      disk_gb: diskGb,
      volume_gb: volumeGb,
      visible_devices: visibleDevices.trim() || null,
      ...(Object.keys(envVars).length ? { env_vars: envVars } : {}),
      storage_id: storageId,
      hf_push_repo: hfPushRepo.trim() || null,
      report_to: [
        ...(wandbOn ? (["wandb"] as const) : []),
        ...(mlflowOn ? (["mlflow"] as const) : []),
      ],
      wandb_project: wandbOn ? wandbProject.trim() || null : null,
      wandb_entity: wandbOn ? wandbEntity.trim() || null : null,
      mlflow_tracking_uri: mlflowOn ? mlflowUri.trim() || null : null,
      mlflow_experiment: mlflowOn ? mlflowExperiment.trim() || null : null,
    };

    setSubmitting(true);
    try {
      const created = await gateway.createTrainingRun(body);
      toast.success(`Created ${created.id}`, { duration: 4000 });
      router.push(`/autotrain/${encodeURIComponent(created.id)}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">New training run</h1>
        <p className="mt-1 max-w-xl text-sm text-muted-foreground">
          {isTts
            ? "Finetune a Qwen3 + NeuCodec TTS model on a dataset. Audio is tokenized + packed, then trained as a causal LM (loss-only; metrics to W&B/MLflow)."
            : "Finetune a Whisper model on a dataset. WER + CER are evaluated each epoch; training stops at the max-epoch cap or early on patience."}
        </p>
      </div>

      {/* Training type */}
      <Section icon={<Sparkles className="h-4 w-4" />} title="Training type"
        description="What kind of model to finetune.">
        <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
          <button type="button" onClick={() => pickTask("asr")}
            className={cn("flex items-start gap-3 rounded-md border px-3 py-2.5 text-left text-sm transition-colors",
              !isTts ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40")}>
            <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <div className="font-medium">ASR — Whisper</div>
              <div className="text-xs text-muted-foreground">Speech→text. Per-epoch WER/CER + early stop.</div>
            </div>
          </button>
          <button type="button" onClick={() => pickTask("tts")}
            className={cn("flex items-start gap-3 rounded-md border px-3 py-2.5 text-left text-sm transition-colors",
              isTts ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40")}>
            <Activity className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <div className="font-medium">TTS — Qwen3 + NeuCodec</div>
              <div className="text-xs text-muted-foreground">Text→speech. Tokenize → pack → finetune (loss-only).</div>
            </div>
          </button>
        </div>
      </Section>

      {/* Model + data */}
      <Section icon={<Sparkles className="h-4 w-4" />} title="Model & data"
        description={isTts
          ? "The base Qwen3 model + the {audio, transcription} dataset to finetune on."
          : "The base Whisper checkpoint and the dataset to finetune on."}>
        <Grid>
          <FieldWrap label={isTts ? "Base TTS model" : "Base Whisper model"}>
            <Select value={modelChoice} onValueChange={setModelChoice}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {MODELS.map((m) => <SelectItem key={m} value={m}>{m}</SelectItem>)}
                <SelectItem value={CUSTOM}>Custom…</SelectItem>
              </SelectContent>
            </Select>
            {modelChoice === CUSTOM && (
              <Input className="mt-2 font-mono" placeholder={isTts ? "Scicom-intl/Multilingual-…-TTS" : "org/whisper-variant"}
                value={customModel} onChange={(e) => setCustomModel(e.target.value)} />
            )}
          </FieldWrap>
          <FieldWrap label="Run name">
            <Input className="font-mono" value={name} onChange={(e) => setName(e.target.value)} />
          </FieldWrap>
          <FieldWrap label="Training dataset" hint="From the Datasets page.">
            <Select value={datasetId} onValueChange={setDatasetId}>
              <SelectTrigger><SelectValue placeholder={datasets.length ? "Pick a dataset…" : "No datasets yet"} /></SelectTrigger>
              <SelectContent>
                {datasets.map((d) => (
                  <SelectItem key={d.id} value={d.id}>
                    {d.name}{d.num_rows != null ? ` · ${d.num_rows} rows` : ""} · {d.kind}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </FieldWrap>
          {!isTts && (
            <FieldWrap label="Test dataset" hint="Held out for per-epoch WER/CER. Auto-split if none.">
              <Select value={testDatasetId} onValueChange={setTestDatasetId}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value={AUTO_SPLIT}>— Auto-split from training set —</SelectItem>
                  {datasets.filter((d) => d.id !== datasetId).map((d) => (
                    <SelectItem key={d.id} value={d.id}>{d.name} · {d.kind}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {testDatasetId === AUTO_SPLIT && (
                <div className="mt-2 flex items-center gap-2">
                  <Label className="text-xs text-muted-foreground">Hold-out %</Label>
                  <Input type="number" min={1} max={50} className="w-24"
                    value={evalSplitPct} onChange={(e) => setEvalSplitPct(Number(e.target.value))} />
                  <span className="text-[11px] text-muted-foreground">uses a `split` column if present</span>
                </div>
              )}
            </FieldWrap>
          )}
        </Grid>
      </Section>

      {/* Training — single run vs. hyperparameter sweep (tab) */}
      <Section icon={<Cpu className="h-4 w-4" />} title="Training"
        description={sweepOn
          ? "Sweep: comma-separate the values to try — the cross-product is the trial grid, run in parallel across your GPUs."
          : (isTts
            ? "Qwen3 + NeuCodec finetune hyperparameters (loss-only; no per-epoch WER/CER)."
            : "Epochs, early stopping, and core hyperparameters.")}>
        <div className="mb-5 flex items-center gap-3">
          <div className="inline-flex rounded-md border border-border p-0.5 text-sm">
            {([["single", "Single run"], ["sweep", "Sweep"]] as const).map(([v, label]) => {
              const active = (v === "sweep") === sweepOn;
              return (
                <button key={v} type="button" onClick={() => setSweepOn(v === "sweep")}
                  className={cn("rounded px-3 py-1 transition-colors",
                    active ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground")}>
                  {label}
                </button>
              );
            })}
          </div>
          {sweepOn && (
            <span className="rounded-md border border-border bg-muted/60 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
              {trialCount} trial{trialCount === 1 ? "" : "s"}
            </span>
          )}
        </div>

        <Grid>
          {/* always-single knobs */}
          {!isTts && (
            <FieldWrap label="Eval metric" hint={sweepOn ? "Ranks the trials (lower is better)." : "Drives early stopping + best-model selection."}>
              <Select value={evalMetric} onValueChange={(v) => setEvalMetric(v as "wer" | "cer")}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="wer">WER</SelectItem>
                  <SelectItem value="cer">CER</SelectItem>
                </SelectContent>
              </Select>
            </FieldWrap>
          )}
          <FieldWrap label="Precision">
            <Select value={precision} onValueChange={(v) => setPrecision(v as "fp16" | "bf16" | "fp32")}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="bf16">bf16</SelectItem>
                <SelectItem value="fp16">fp16</SelectItem>
                {isTts && <SelectItem value="fp32">fp32</SelectItem>}
              </SelectContent>
            </Select>
          </FieldWrap>
          {!isTts && (
            <FieldWrap label="Early-stop patience" hint="Epochs without eval improvement before stopping. 0 = off.">
              <NumberField min={0} value={patience} onChange={setPatience} />
            </FieldWrap>
          )}
          {!isTts && (
            <FieldWrap label="Language" hint="ISO code (e.g. en, ms). Empty = multilingual / model default.">
              <Input className="font-mono" placeholder="en" value={language} onChange={(e) => setLanguage(e.target.value)} />
            </FieldWrap>
          )}

          {/* sweepable knobs — single value, or comma-separated list in sweep mode */}
          {sweepOn ? (
            <FieldWrap label="Max epochs" hint="e.g. 3, 5">
              <Input className="font-mono" placeholder="3, 5" value={sweepEpochs} onChange={(e) => setSweepEpochs(e.target.value)} />
            </FieldWrap>
          ) : (
            <FieldWrap label="Max epochs"><NumberField min={1} value={maxEpochs} onChange={setMaxEpochs} /></FieldWrap>
          )}
          {sweepOn ? (
            <FieldWrap label="Batch sizes" hint="e.g. 8, 16">
              <Input className="font-mono" placeholder="8, 16" value={sweepBatch} onChange={(e) => setSweepBatch(e.target.value)} />
            </FieldWrap>
          ) : (
            <FieldWrap label="Batch size (per device)"><NumberField min={1} value={batchSize} onChange={setBatchSize} /></FieldWrap>
          )}
          {sweepOn ? (
            <FieldWrap label="Learning rates" hint="e.g. 1e-5, 2e-5">
              <Input className="font-mono" placeholder="1e-5, 2e-5" value={sweepLr} onChange={(e) => setSweepLr(e.target.value)} />
            </FieldWrap>
          ) : (
            <FieldWrap label="Learning rate"><Input className="font-mono" value={learningRate} onChange={(e) => setLearningRate(e.target.value)} /></FieldWrap>
          )}
          {sweepOn ? (
            <FieldWrap label="Grad-accum steps" hint="e.g. 1, 4">
              <Input className="font-mono" placeholder="1, 4" value={sweepGradAccum} onChange={(e) => setSweepGradAccum(e.target.value)} />
            </FieldWrap>
          ) : isTts ? (
            <FieldWrap label="Grad accumulation"><NumberField min={1} value={gradAccum} onChange={setGradAccum} /></FieldWrap>
          ) : null}

          <FieldWrap label="Log loss every N steps" hint="Streams a training-loss point every N steps (@@STEP) for the live loss curve. Smaller = smoother, more log lines.">
            <NumberField min={1} value={loggingSteps} onChange={setLoggingSteps} />
          </FieldWrap>

          {/* TTS knobs */}
          {isTts && (sweepOn ? (
            <FieldWrap label="Block sizes" hint="e.g. 8192, 10240">
              <Input className="font-mono" placeholder="8192, 10240" value={sweepBlock} onChange={(e) => setSweepBlock(e.target.value)} />
            </FieldWrap>
          ) : (
            <FieldWrap label="Block size (training ctx)" hint="qwen3_tts_flash --block_size">
              <NumberField min={512} value={blockSize} onChange={setBlockSize} />
            </FieldWrap>
          ))}
          {isTts && (
            <>
              <FieldWrap label="Pack sequence length" hint="Per-utterance pack length (pack_stage1).">
                <NumberField min={256} value={packSeq} onChange={setPackSeq} />
              </FieldWrap>
              <FieldWrap label="Pack tokenizer" hint="Tokenizer carrying the NeuCodec speech tokens.">
                <Input className="font-mono" value={ttsTokenizer} onChange={(e) => setTtsTokenizer(e.target.value)} />
              </FieldWrap>
              <FieldWrap label="Default speaker" hint="Used when a row has no speaker column.">
                <Input className="font-mono" value={defaultSpeaker} onChange={(e) => setDefaultSpeaker(e.target.value)} />
              </FieldWrap>
            </>
          )}

          {sweepOn && (
            <FieldWrap label="GPUs per trial" hint="Trials run concurrently = #GPUs / this.">
              <NumberField min={1} value={gpusPerTrial} onChange={setGpusPerTrial} />
            </FieldWrap>
          )}
        </Grid>

        {sweepOn && (
          <p className="mt-4 text-[11px] leading-snug text-muted-foreground">
            Trials are pinned via <span className="font-mono">CUDA_VISIBLE_DEVICES</span> across the GPUs from{" "}
            <span className="font-medium">Run on</span> (the pin on a VM, or the GPU count on RunPod), {gpusPerTrial} each —
            e.g. GPUs <span className="font-mono">6,7</span> with 1/trial → 2 at a time. Best model chosen by{" "}
            {isTts ? "lowest final loss" : "lowest WER/CER"}; each trial&apos;s checkpoint lands under{" "}
            <span className="font-mono">…/trials/&lt;i&gt;/</span>.
          </p>
        )}
      </Section>

      {/* Run on — pod card (mirrors benchmark/new) */}
      <Section icon={<Server className="h-4 w-4" />} title="Run on"
        description="Default cloud spawns a fresh RunPod pod. Bare metal uses a VM you've registered under GPU Providers.">
        <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
          <button type="button" onClick={() => setTarget("cloud")}
            className={cn("flex items-start gap-3 rounded-md border px-3 py-2.5 text-left text-sm transition-colors",
              target === "cloud" ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40")}>
            <Cpu className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <div className="font-medium">Default cloud (RunPod)</div>
              <div className="text-xs text-muted-foreground">Provision a fresh pod on demand. Pay-per-second.</div>
            </div>
          </button>
          <button type="button" onClick={() => setTarget("vm")}
            className={cn("flex items-start gap-3 rounded-md border px-3 py-2.5 text-left text-sm transition-colors",
              target === "vm" ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40")}>
            <Server className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <div className="font-medium">Bare metal (VM)</div>
              <div className="text-xs text-muted-foreground">SSH onto a registered VM. No spin-up cost.</div>
            </div>
          </button>
        </div>
      </Section>

      <Section icon={<Server className="h-4 w-4" />} title="Pod"
        description={target === "cloud"
          ? "GPU, count, and cloud tier for the RunPod instance the trainer spawns."
          : "Which registered VM to SSH into. Hardware is fixed by the VM."}>
        {target === "vm" && (
          <div className="space-y-3">
            <div className="space-y-1.5">
              <Label htmlFor="train-provider" className="text-xs">VM provider</Label>
              {vmProviders.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  No VM providers registered. Add one at{" "}
                  <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">GPU Providers → New provider</a>.
                </p>
              ) : (
                <Select value={providerId} onValueChange={setProviderId}>
                  <SelectTrigger id="train-provider"><SelectValue placeholder="Pick a VM…" /></SelectTrigger>
                  <SelectContent>
                    {vmProviders.map((p) => (
                      <SelectItem key={p.id} value={p.id}>
                        {p.name}
                        {p.gpu_count != null && p.gpu_count > 0 ? ` · ${p.gpu_count} GPU` : ""}
                        {p.host ? ` · ${p.host}` : ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
              <p className="text-xs text-muted-foreground">
                The trainer runs directly on the VM via SSH. The VM&apos;s GPUs, disk, and Python are used as-is.
              </p>
              {providerId && <VmAvailabilityRow state={vmAvail} onRefresh={() => refreshVmAvail(providerId)} />}
            </div>
          </div>
        )}
        {target === "cloud" && (
          <div className="space-y-5">
            <FieldWrap label="RunPod account" hint="Which RunPod provider to bill against. Default = gateway env key.">
              <Select value={runpodProviderId || "__default__"}
                onValueChange={(v) => setRunpodProviderId(v === "__default__" ? "" : v)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__default__">Gateway default (RunPod)</SelectItem>
                  {runpodProviders.map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      {p.name}{p.api_key_last4 ? ` · ****${p.api_key_last4}` : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {runpodProviders.length === 0 && (
                <p className="text-xs text-muted-foreground">
                  None registered. <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">Add a RunPod account →</a>
                </p>
              )}
            </FieldWrap>

            <FieldWrap label="Cloud tier" hint="Community is cheaper with variable hosts; Secure uses vetted hosts with more capacity.">
              <div className="grid grid-cols-2 gap-2">
                {([["secure", "Secure", "vetted hosts, more capacity"], ["community", "Community", "cheaper, variable hosts"]] as const).map(
                  ([val, title, sub]) => {
                    const selected = (val === "secure") === secureCloud;
                    return (
                      <button key={val} type="button" onClick={() => setSecureCloud(val === "secure")}
                        className={cn("rounded-md border p-3 text-left transition-colors",
                          selected ? "border-foreground/60 ring-1 ring-foreground/20" : "border-border hover:border-foreground/40")}>
                        <div className="text-sm font-medium">{title}</div>
                        <div className="mt-0.5 text-xs text-muted-foreground">{sub}</div>
                      </button>
                    );
                  },
                )}
              </div>
            </FieldWrap>

            <FieldWrap label="GPU"
              hint={(() => {
                const g = gpuOptions.find((o) => o.id === gpuType);
                return g ? capacityHint(g.vram_gb, gpuCount) : undefined;
              })()}
              extra={<AvailabilityBadge state={availability} count={gpuCount} />}>
              <div className="flex gap-2">
                <SearchableSelect
                  className="flex-1"
                  value={gpuType}
                  onChange={setGpuType}
                  options={gpuOptions.map((g) => ({ value: g.id, label: g.label, hint: capacityHint(g.vram_gb, 1) }))}
                  placeholder="Choose a GPU"
                  searchPlaceholder="Search GPUs (e.g. h100, 24gb, ada)…"
                />
                <Select value={String(gpuCount)} onValueChange={(v) => setGpuCount(Number.parseInt(v, 10))}>
                  <SelectTrigger className="w-24 shrink-0"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {GPU_COUNT_CHOICES.map((n) => <SelectItem key={n} value={String(n)}>×{n}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
            </FieldWrap>

            <div className="grid grid-cols-2 gap-3">
              <FieldWrap label="Container disk (GB)" hint="Ephemeral workspace. Resets when the pod stops.">
                <NumberField min={20} value={diskGb} onChange={setDiskGb} />
              </FieldWrap>
              <FieldWrap label="Volume (GB)" hint="Persistent volume mounted at /workspace (model cache).">
                <NumberField min={0} value={volumeGb} onChange={setVolumeGb} />
              </FieldWrap>
            </div>

            <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>Pick a GPU with enough VRAM for the model + batch. Whisper-large needs ~24 GB+ at modest batch sizes.</span>
            </div>
          </div>
        )}

        <div className="mt-5 space-y-1.5 border-t border-border pt-4">
          <Label htmlFor="train-cuda" className="text-xs">CUDA_VISIBLE_DEVICES</Label>
          <Input id="train-cuda" className="font-mono text-xs" placeholder="e.g. 0,1 (empty = all GPUs)"
            value={visibleDevices} onChange={(e) => setVisibleDevices(e.target.value)} />
          <p className="text-xs text-muted-foreground">Pins which GPUs the trainer uses. Empty = all visible GPUs.</p>
        </div>

        <div className="mt-4 space-y-1.5">
          <Label htmlFor="train-env" className="text-xs">Environment variables</Label>
          <Textarea
            id="train-env"
            rows={8}
            spellCheck={false}
            className="font-mono text-xs"
            placeholder={'export HOME="/share/home"\nexport HF_HOME="/share/huggingface"\nexport XDG_CACHE_HOME="/share/.cache"\nexport TRITON_CACHE_DIR="/share/triton_cache"'}
            value={envText}
            onChange={(e) => setEnvText(e.target.value)}
          />
          <p className="text-xs text-muted-foreground">
            One <span className="font-mono">KEY=value</span> per line (<span className="font-mono">export</span> prefix ok).
            Exported before the run; absolute-path values are <span className="font-mono">mkdir -p</span>&apos;d. Useful to
            redirect HOME + caches (HF/Triton/torchinductor/vLLM/…) to a shared disk.
            {Object.keys(parseEnvVars(envText)).length > 0 && (
              <>
                {" "}· parsed: <span className="font-mono">{Object.keys(parseEnvVars(envText)).join(", ")}</span>
              </>
            )}
          </p>
        </div>
      </Section>

      {/* Experiment tracking */}
      <Section icon={<Activity className="h-4 w-4" />} title="Experiment tracking"
        description="Push per-epoch metrics to W&B and/or MLflow via HF Trainer. Credentials are read from the global Secrets page at run time — set them under Secrets, not here.">
        <div className="space-y-4">
          <label className="flex cursor-pointer items-center gap-2.5 text-sm">
            <Switch checked={wandbOn} onCheckedChange={setWandbOn} />
            <span className="font-medium">Weights &amp; Biases</span>
            <span className="text-xs text-muted-foreground">uses <span className="font-mono">WANDB_API_KEY</span> from Secrets</span>
          </label>
          {wandbOn && (
            <Grid>
              <FieldWrap label="W&B project"><Input className="font-mono" placeholder="whisper-finetune" value={wandbProject} onChange={(e) => setWandbProject(e.target.value)} /></FieldWrap>
              <FieldWrap label="W&B entity (optional)"><Input className="font-mono" placeholder="my-team" value={wandbEntity} onChange={(e) => setWandbEntity(e.target.value)} /></FieldWrap>
            </Grid>
          )}

          <label className="flex cursor-pointer items-center gap-2.5 border-t border-border pt-4 text-sm">
            <Switch checked={mlflowOn} onCheckedChange={setMlflowOn} />
            <span className="font-medium">MLflow</span>
            <span className="text-xs text-muted-foreground">uses <span className="font-mono">MLFLOW_TRACKING_URI/USERNAME/PASSWORD</span> from Secrets</span>
          </label>
          {mlflowOn && (
            <Grid>
              <FieldWrap label="Tracking URI (optional)" hint="Overrides MLFLOW_TRACKING_URI from Secrets for this run.">
                <Input className="font-mono" placeholder="https://mlflow.aies.scicom.dev" value={mlflowUri} onChange={(e) => setMlflowUri(e.target.value)} />
              </FieldWrap>
              <FieldWrap label="Experiment" hint="MLFLOW_EXPERIMENT_NAME, e.g. test-classification.">
                <Input className="font-mono" placeholder="whisper-finetune" value={mlflowExperiment} onChange={(e) => setMlflowExperiment(e.target.value)} />
              </FieldWrap>
            </Grid>
          )}
        </div>
      </Section>

      {/* Artifacts */}
      <Section icon={<Database className="h-4 w-4" />} title="Artifacts"
        description="Where the best model, per-epoch metrics, and logs are written.">
        <Grid>
          <FieldWrap label="S3 storage" hint="Enabled s3 backend. Required.">
            {!hasStorage ? (
              <p className="text-xs text-muted-foreground">
                No S3 storage. Add one under <a href="/storage/new" className="underline">Storage → New</a>.
              </p>
            ) : (
              <Select value={storageId} onValueChange={setStorageId}>
                <SelectTrigger><SelectValue placeholder="Pick a storage…" /></SelectTrigger>
                <SelectContent>
                  {s3Storages.map((s) => (
                    <SelectItem key={s.id} value={s.id}>
                      {s.name}{s.bucket ? ` · s3://${s.bucket}` : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </FieldWrap>
          <FieldWrap label="Push best model to HF (optional)" hint="HuggingFace repo, e.g. you/whisper-ms. Uses HF_TOKEN.">
            <Input className="font-mono" placeholder="org/model-finetuned" value={hfPushRepo} onChange={(e) => setHfPushRepo(e.target.value)} />
          </FieldWrap>
        </Grid>
      </Section>

      <div className="flex items-center justify-end gap-3 border-t border-border pt-4">
        {error && <p className="text-sm text-destructive">{error}</p>}
        <Button type="button" variant="outline" onClick={() => router.push("/autotrain")}>Cancel</Button>
        <Button type="submit" disabled={submitting || !hasStorage} className="min-w-36">
          {submitting ? (<><Loader2 className="h-4 w-4 animate-spin" /> Creating…</>)
            : (<><FlaskConical className="h-4 w-4" /> Start training</>)}
        </Button>
      </div>
    </form>
  );
}

function Section({ icon, title, description, children }: {
  icon: React.ReactNode; title: string; description?: string; children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="pb-4">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-muted text-muted-foreground">{icon}</div>
          <CardTitle className="text-base">{title}</CardTitle>
        </div>
        {description && <CardDescription className="text-xs">{description}</CardDescription>}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function Grid({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-1 gap-x-4 gap-y-5 sm:grid-cols-2">{children}</div>;
}

function FieldWrap({ label, hint, extra, children }: {
  label: string; hint?: string; extra?: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <Label className="text-xs font-medium">{label}</Label>
        {extra}
      </div>
      {children}
      {hint && <p className="text-[11px] leading-snug text-muted-foreground">{hint}</p>}
    </div>
  );
}

// Inline availability row under the VM provider dropdown (mirrors benchmark/new).
function VmAvailabilityRow({ state, onRefresh }: { state: VmAvailState; onRefresh: () => void }) {
  if (state.status === "idle") return null;
  if (state.status === "loading") {
    return (
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" /> Checking availability via SSH…
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="flex items-center justify-between gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-2.5 py-1.5 text-xs text-destructive">
        <span className="inline-flex items-center gap-1.5 truncate">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" />
          <span className="truncate" title={state.message}>{state.message}</span>
        </span>
        <button type="button" onClick={onRefresh} className="inline-flex items-center gap-1 underline-offset-2 hover:underline">
          <RefreshCw className="h-3 w-3" /> Retry
        </button>
      </div>
    );
  }
  const { data } = state;
  if (!data.ok) {
    return (
      <div className="flex items-center justify-between gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-2.5 py-1.5 text-xs text-amber-700 dark:text-amber-400">
        <span className="inline-flex items-center gap-1.5 truncate">
          <X className="h-3.5 w-3.5 shrink-0" />
          <span className="truncate" title={data.message}>{data.message}</span>
        </span>
        <button type="button" onClick={onRefresh} className="inline-flex items-center gap-1 underline-offset-2 hover:underline">
          <RefreshCw className="h-3 w-3" /> Retry
        </button>
      </div>
    );
  }
  const totalFreeMib = data.gpus.reduce((s, g) => s + g.mem_free_mib, 0);
  const totalMib = data.gpus.reduce((s, g) => s + g.mem_total_mib, 0);
  const busy = data.gpus.filter((g) => g.mem_free_mib < g.mem_total_mib * 0.2 || g.util_pct > 50).length;
  const allFree = busy === 0;
  return (
    <div className={cn("space-y-1 rounded-md border px-2.5 py-1.5 text-xs",
      allFree ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
        : "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400")}>
      <div className="flex items-center justify-between gap-2">
        <span className="inline-flex items-center gap-1.5">
          {allFree ? <Check className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />}
          {data.gpus.length} GPU{data.gpus.length === 1 ? "" : "s"} · {fmtMib(totalFreeMib)} free / {fmtMib(totalMib)}
          {!allFree && ` · ${busy} busy`}
        </span>
        <button type="button" onClick={onRefresh} className="inline-flex items-center gap-1 underline-offset-2 hover:underline">
          <RefreshCw className="h-3 w-3" /> Refresh
        </button>
      </div>
      <div className="flex flex-col gap-0.5 font-mono text-[10px] text-muted-foreground">
        {data.gpus.map((g) => (
          <span key={g.index}>
            #{g.index} {g.name.replace(/^NVIDIA\s+/, "")} · {fmtMib(g.mem_free_mib)}/{fmtMib(g.mem_total_mib)} free · {g.util_pct}% util
          </span>
        ))}
      </div>
    </div>
  );
}

function fmtMib(mib: number): string {
  if (mib >= 1024) return `${(mib / 1024).toFixed(1)} GiB`;
  return `${mib} MiB`;
}
