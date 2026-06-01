"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  Check,
  ChevronDown,
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
import { NumberField } from "@/components/ui/number-field";
import { SearchableSelect } from "@/components/ui/searchable-select";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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
  TrackingCredentialRecord,
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

// Training-audio augmentation techniques (mirror whisper_finetune._AUG_FUNCS).
// One enabled technique is applied at random per augmented sample.
const AUG_OPTIONS: { id: string; label: string; desc: string }[] = [
  { id: "telephone", label: "Telephone", desc: "Phone-line degradation: band-pass + downsample + clip + dropout" },
  { id: "noise", label: "Noise", desc: "Additive Gaussian noise at random SNR (10–40 dB)" },
  { id: "dropout", label: "Dropout", desc: "Zero random ~25 ms chunks (packet loss)" },
  { id: "gain", label: "Gain", desc: "Random volume change (−20 … +6 dB)" },
  { id: "pitch", label: "Pitch shift", desc: "±3 semitones (duration preserved)" },
  { id: "speed", label: "Speed", desc: "Time-stretch 0.9–1.1× (speaking rate)" },
  { id: "reverb", label: "Reverb", desc: "Light room reverb" },
  { id: "bandpass", label: "Band-pass", desc: "Telephone 300–3400 Hz band only" },
];

// precision = "<weight load dtype>-<mixed-precision (AMP) train dtype>".
const PRECISIONS: { value: string; label: string }[] = [
  { value: "fp32-bf16", label: "fp32 load · bf16 mixed (recommended)" },
  { value: "bf16-bf16", label: "bf16 load · bf16 mixed" },
  { value: "fp32-fp16", label: "fp32 load · fp16 mixed" },
  { value: "fp16-fp16", label: "fp16 load · fp16 mixed" },
];

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

// Comma/space-separated tokens that AREN'T a positive number (kind="int" also
// requires an integer). Used to reject bad sweep cells instead of dropping them.
function invalidNumTokens(s: string, kind: "num" | "int" | "nonneg"): string[] {
  return s
    .split(/[,\s]+/)
    .map((t) => t.trim())
    .filter(Boolean)
    .filter((t) => {
      const n = Number(t);
      if (!Number.isFinite(n)) return true;
      if (kind === "nonneg") return n < 0;       // weight decay: 0 is valid
      if (n <= 0) return true;
      return kind === "int" && !Number.isInteger(n);
    });
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
  const searchParams = useSearchParams();
  // ?from=<runId> → prefill the form with that run's config ("Edit as new").
  const fromId = searchParams.get("from");
  const [prefilling, setPrefilling] = useState(!!fromId);
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
  const [normalizeText, setNormalizeText] = useState(true);
  const [maxEpochs, setMaxEpochs] = useState(3);
  const [patience, setPatience] = useState(1);
  const [batchSize, setBatchSize] = useState(8);
  const [loggingSteps, setLoggingSteps] = useState(10);
  const [learningRate, setLearningRate] = useState("1e-5");
  const [precision, setPrecision] = useState<string>("fp32-bf16");
  const [language, setLanguage] = useState("");
  const [weightDecay, setWeightDecay] = useState(0.0);
  // LoRA / PEFT (merged into base at save → drop-in checkpoint)
  const [useLora, setUseLora] = useState(false);
  const [loraR, setLoraR] = useState(16);
  const [loraAlphaRatio, setLoraAlphaRatio] = useState(2);
  const [loraDropout, setLoraDropout] = useState(0.05);
  const [freezeEncoder, setFreezeEncoder] = useState(false);
  // Multi-GPU single run: DDP (torchrun) vs DataParallel.
  const [useDdp, setUseDdp] = useState(true);
  // hyperparameter sweep
  const [sweepOn, setSweepOn] = useState(false);
  const [gpusPerTrial, setGpusPerTrial] = useState(1);
  const [sweepLr, setSweepLr] = useState("");
  const [sweepBatch, setSweepBatch] = useState("");
  const [sweepGradAccum, setSweepGradAccum] = useState("");
  const [sweepEpochs, setSweepEpochs] = useState("");
  const [sweepBlock, setSweepBlock] = useState("");
  const [sweepWeightDecay, setSweepWeightDecay] = useState("");
  const [sweepLoraR, setSweepLoraR] = useState("");
  const [sweepPrecisions, setSweepPrecisions] = useState<string[]>([]);
  // compare augmentation on/off as a sweep dimension (the "on" arm uses the
  // selected techniques + probability below; the "off" arm trains clean audio).
  const [sweepAugment, setSweepAugment] = useState(false);
  // compare freeze-encoder on/off as a sweep dimension.
  const [sweepFreeze, setSweepFreeze] = useState(false);
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
  const [workDir, setWorkDir] = useState("/share");
  const [cleanupCheckpoints, setCleanupCheckpoints] = useState(true);
  const [augmentTechniques, setAugmentTechniques] = useState<string[]>([]);
  const [augmentProb, setAugmentProb] = useState(0.5);
  // experiment tracking — named credentials from the Secrets page (picked per run)
  const [trackingCreds, setTrackingCreds] = useState<TrackingCredentialRecord[]>([]);
  const [wandbCredId, setWandbCredId] = useState("");
  const [mlflowCredId, setMlflowCredId] = useState("");
  const [wandbProject, setWandbProject] = useState("");
  const [wandbEntity, setWandbEntity] = useState("");
  const [mlflowUri, setMlflowUri] = useState("");
  const [mlflowExperiment, setMlflowExperiment] = useState("");
  const wandbOn = !!wandbCredId;
  const mlflowOn = !!mlflowCredId;

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
    gateway.listTrackingCredentials().then(setTrackingCreds).catch(() => {});
    gateway
      .listRunpodGpuTypes()
      .then((rows) => {
        if (rows.length === 0) return;
        setGpuOptions(rows);
        setGpuType((cur) => (rows.some((g) => g.id === cur) ? cur : rows[0].id));
      })
      .catch(() => {});
  }, []);

  // "Edit as new": fetch the source run and replay its config_json + record into
  // form state. All sets happen inside the async callback (post-mount), so they
  // override the useState defaults without fighting the initial render.
  useEffect(() => {
    if (!fromId) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await gateway.getTrainingRun(fromId);
        if (cancelled) return;
        const c = (r.config_json ?? {}) as Record<string, unknown>;
        const num = (v: unknown, d: number) => (typeof v === "number" && Number.isFinite(v) ? v : d);
        const str = (v: unknown) => (v == null ? "" : String(v));
        const arr = (v: unknown) => (Array.isArray(v) ? (v as unknown[]) : []);
        const csv = (v: unknown) => arr(v).join(", ");

        const tt: "asr" | "tts" = (r.task_type ?? c.task_type) === "tts" ? "tts" : "asr";
        setTaskType(tt);
        const models = tt === "tts" ? TTS_BASE_MODELS : WHISPER_MODELS;
        if (r.base_model && models.includes(r.base_model)) setModelChoice(r.base_model);
        else if (r.base_model) { setModelChoice(CUSTOM); setCustomModel(r.base_model); }
        setName(r.name || (tt === "tts" ? "tts-finetune" : "whisper-finetune"));
        setDatasetId(r.dataset_id || "");
        setTestDatasetId(r.test_dataset_id || AUTO_SPLIT);
        if (c.eval_split_pct != null) setEvalSplitPct(num(c.eval_split_pct, 10));
        // TTS knobs
        if (c.tokenizer) setTtsTokenizer(str(c.tokenizer));
        if (c.block_size != null) setBlockSize(num(c.block_size, 10240));
        if (c.pack_sequence_length != null) setPackSeq(num(c.pack_sequence_length, 4096));
        if (c.default_speaker) setDefaultSpeaker(str(c.default_speaker));
        // training
        if (c.grad_accum != null) setGradAccum(num(c.grad_accum, 4));
        if (c.eval_metric === "wer" || c.eval_metric === "cer") setEvalMetric(c.eval_metric);
        if (c.normalize_text != null) setNormalizeText(!!c.normalize_text);
        if (c.max_epochs != null) setMaxEpochs(num(c.max_epochs, 3));
        if (c.patience != null) setPatience(num(c.patience, 1));
        if (c.batch_size != null) setBatchSize(num(c.batch_size, 8));
        if (c.logging_steps != null) setLoggingSteps(num(c.logging_steps, 10));
        if (c.learning_rate != null) setLearningRate(str(c.learning_rate));
        if (c.weight_decay != null) setWeightDecay(num(c.weight_decay, 0));
        if (c.use_lora != null) setUseLora(!!c.use_lora);
        if (c.lora_r != null) setLoraR(num(c.lora_r, 16));
        if (c.lora_alpha_ratio != null) setLoraAlphaRatio(num(c.lora_alpha_ratio, 2));
        if (c.lora_dropout != null) setLoraDropout(num(c.lora_dropout, 0.05));
        if (c.freeze_encoder != null) setFreezeEncoder(!!c.freeze_encoder);
        if (c.use_ddp != null) setUseDdp(!!c.use_ddp);
        if (c.precision) setPrecision(str(c.precision));
        if (c.language != null) setLanguage(str(c.language));
        // sweep
        const sweep = (c.sweep ?? {}) as Record<string, unknown>;
        if (Object.values(sweep).some((v) => Array.isArray(v) && v.length)) {
          setSweepOn(true);
          if (c.gpus_per_trial != null) setGpusPerTrial(num(c.gpus_per_trial, 1));
          setSweepLr(csv(sweep.learning_rate));
          setSweepBatch(csv(sweep.batch_size));
          setSweepGradAccum(csv(sweep.grad_accum));
          setSweepEpochs(csv(sweep.max_epochs));
          setSweepBlock(csv(sweep.block_size));
          setSweepWeightDecay(csv(sweep.weight_decay));
          setSweepLoraR(csv(sweep.lora_r));
          setSweepPrecisions(arr(sweep.precision).map(String));
          setSweepAugment(arr(sweep.augment).length > 0);
          setSweepFreeze(arr(sweep.freeze_encoder).length > 0);
        }
        // augmentation
        if (Array.isArray(c.augment_techniques)) setAugmentTechniques(arr(c.augment_techniques).map(String));
        if (c.augment_prob != null) setAugmentProb(num(c.augment_prob, 0.5));
        // run on
        if (r.provider_kind === "vm") { setTarget("vm"); setProviderId(r.provider_id || ""); }
        else if (r.provider_id) { setTarget("cloud"); setRunpodProviderId(r.provider_id); }
        if (r.gpu_type) setGpuType(r.gpu_type);
        if (r.gpu_count != null) setGpuCount(r.gpu_count);
        if (c.secure_cloud != null) setSecureCloud(!!c.secure_cloud);
        if (c.disk_gb != null) setDiskGb(num(c.disk_gb, 60));
        if (c.volume_gb != null) setVolumeGb(num(c.volume_gb, 80));
        setVisibleDevices(r.visible_devices || "");
        // env vars dict → KEY=value lines
        const ev = (c.env_vars ?? {}) as Record<string, unknown>;
        const evText = Object.entries(ev).map(([k, v]) => `${k}=${v}`).join("\n");
        if (evText) setEnvText(evText);
        // artifacts
        setStorageId(r.storage_id || "");
        if (c.hf_push_repo) setHfPushRepo(str(c.hf_push_repo));
        if (c.work_dir) setWorkDir(str(c.work_dir));
        if (c.cleanup_checkpoints != null) setCleanupCheckpoints(!!c.cleanup_checkpoints);
        // experiment tracking
        if (c.wandb_credential_id) setWandbCredId(str(c.wandb_credential_id));
        if (c.mlflow_credential_id) setMlflowCredId(str(c.mlflow_credential_id));
        if (c.wandb_project) setWandbProject(str(c.wandb_project));
        if (c.wandb_entity) setWandbEntity(str(c.wandb_entity));
        if (c.mlflow_tracking_uri) setMlflowUri(str(c.mlflow_tracking_uri));
        if (c.mlflow_experiment) setMlflowExperiment(str(c.mlflow_experiment));
      } catch (e) {
        toast.error(`Couldn't load ${fromId}: ${e instanceof Error ? e.message : String(e)}`, { duration: 6000 });
      } finally {
        if (!cancelled) setPrefilling(false);
      }
    })();
    return () => { cancelled = true; };
  }, [fromId]);

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
  const wandbCreds = useMemo(() => trackingCreds.filter((c) => c.kind === "wandb"), [trackingCreds]);
  const mlflowCreds = useMemo(() => trackingCreds.filter((c) => c.kind === "mlflow"), [trackingCreds]);
  const baseModel = modelChoice === CUSTOM ? customModel.trim() : modelChoice;
  // GPUs available on the chosen target — a VM's registered GPU count, or the
  // RunPod pod's chosen count. 0 = unknown (skip the upper-bound check).
  const gpuBound = useMemo(
    () => (target === "vm" ? (vmProviders.find((p) => p.id === providerId)?.gpu_count ?? 0) : gpuCount),
    [target, vmProviders, providerId, gpuCount],
  );
  // Live validation of the GPU pin (shown inline under the field as you type).
  const vdError = useMemo(() => {
    const vd = visibleDevices.trim();
    if (!vd) return null;
    const toks = vd.split(",").map((t) => t.trim()).filter(Boolean);
    for (const t of toks) if (!/^\d+$/.test(t)) return `"${t}" isn't a valid GPU index — use non-negative integers like 0,1.`;
    const ids = toks.map(Number);
    if (new Set(ids).size !== ids.length) return "Duplicate GPU indices.";
    if (gpuBound > 0) {
      const oob = [...new Set(ids.filter((i) => i >= gpuBound))].sort((a, b) => a - b);
      if (oob.length) return `GPU ${oob.join(", ")} out of range — valid indices are 0–${gpuBound - 1}.`;
    }
    return null;
  }, [visibleDevices, gpuBound]);
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

  function buildSweep(): Record<string, (number | string)[]> {
    const s: Record<string, (number | string)[]> = {};
    const lr = parseCsvNums(sweepLr, false);
    if (lr.length) s.learning_rate = lr;
    const b = parseCsvNums(sweepBatch, true);
    if (b.length) s.batch_size = b;
    const ga = parseCsvNums(sweepGradAccum, true);
    if (ga.length) s.grad_accum = ga;
    const ep = parseCsvNums(sweepEpochs, true);
    if (ep.length) s.max_epochs = ep;
    if (sweepPrecisions.length) s.precision = sweepPrecisions;
    const wd = parseCsvNums(sweepWeightDecay, false);
    if (wd.length) s.weight_decay = wd;
    const lr_ = parseCsvNums(sweepLoraR, true);
    if (lr_.length) s.lora_r = lr_;
    // Augment vs. no-augment. The "on" arm reuses augmentTechniques/augmentProb;
    // only meaningful when at least one technique is selected.
    if (sweepAugment && augmentTechniques.length) s.augment = ["on", "off"];
    // Freeze-encoder vs full as a sweep dimension (ASR only).
    if (sweepFreeze && !isTts) s.freeze_encoder = ["on", "off"];
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

    // --- GPU pin: non-negative integers, no dupes, within the target's count ---
    const vd = visibleDevices.trim();
    const pinned: number[] = [];
    if (vd) {
      const toks = vd.split(",").map((t) => t.trim()).filter(Boolean);
      for (const t of toks) {
        if (!/^\d+$/.test(t)) {
          return setError(`CUDA_VISIBLE_DEVICES: "${t}" is not a valid GPU index — use non-negative integers like 0,1.`);
        }
        pinned.push(Number(t));
      }
      if (new Set(pinned).size !== pinned.length) {
        return setError("CUDA_VISIBLE_DEVICES has duplicate GPU indices.");
      }
      if (gpuBound > 0) {
        const oob = [...new Set(pinned.filter((i) => i >= gpuBound))].sort((a, b) => a - b);
        if (oob.length) {
          return setError(
            `GPU index ${oob.join(", ")} out of range — ${target === "vm" ? "this VM" : "the pod"} has ` +
            `${gpuBound} GPU${gpuBound === 1 ? "" : "s"} (valid indices 0–${gpuBound - 1}).`,
          );
        }
      }
    }

    // --- learning rate must be a positive number like 1e-4 ---
    if (!sweepOn) {
      const lr = Number(learningRate);
      if (!learningRate.trim() || !Number.isFinite(lr) || lr <= 0) {
        return setError(`Learning rate must be a positive number like 1e-4 (got "${learningRate}").`);
      }
    } else {
      const badLr = invalidNumTokens(sweepLr, "num");
      if (sweepLr.trim() && badLr.length) {
        return setError(`Learning rates (sweep): ${badLr.map((b) => `"${b}"`).join(", ")} — use positive numbers like 1e-4.`);
      }
      const numFields: { label: string; val: string; kind: "int" | "nonneg" }[] = [
        { label: "Batch sizes", val: sweepBatch, kind: "int" },
        { label: "Grad-accum steps", val: sweepGradAccum, kind: "int" },
        { label: "Max epochs", val: sweepEpochs, kind: "int" },
        { label: "LoRA r", val: sweepLoraR, kind: "int" },
        { label: "Weight decay", val: sweepWeightDecay, kind: "nonneg" },
        ...(isTts ? [{ label: "Block sizes", val: sweepBlock, kind: "int" as const }] : []),
      ];
      for (const f of numFields) {
        const bad = invalidNumTokens(f.val, f.kind);
        if (f.val.trim() && bad.length) {
          const want = f.kind === "nonneg" ? "non-negative numbers" : "positive integers";
          return setError(`${f.label} (sweep): ${bad.map((b) => `"${b}"`).join(", ")} — use ${want}.`);
        }
      }
      // GPUs per trial can't exceed the GPUs available to the sweep
      const slots = pinned.length || gpuBound;
      if (slots > 0 && gpusPerTrial > slots) {
        return setError(
          `GPUs per trial (${gpusPerTrial}) exceeds the ${slots} GPU${slots === 1 ? "" : "s"} ` +
          `available to the sweep${pinned.length ? " (pinned)" : ` on ${target === "vm" ? "this VM" : "the pod"}`}.`,
        );
      }
    }

    const body: CreateTrainingRunRequest = {
      name: name.trim(),
      dataset_id: datasetId,
      base_model: baseModel,
      task_type: taskType,
      test_dataset_id: isTts ? null : (testDatasetId === AUTO_SPLIT ? null : testDatasetId),
      eval_metric: evalMetric,
      normalize_text: normalizeText,
      max_epochs: maxEpochs,
      patience: isTts ? 0 : patience,
      eval_split_pct: evalSplitPct,
      batch_size: batchSize,
      grad_accum: gradAccum,
      learning_rate: Number(learningRate) || (isTts ? 2e-5 : 1e-5),
      weight_decay: weightDecay,
      use_lora: useLora || (sweepOn && sweepLoraR.trim() !== ""),
      lora_r: loraR,
      lora_alpha_ratio: loraAlphaRatio,
      lora_dropout: loraDropout,
      freeze_encoder: freezeEncoder,
      use_ddp: useDdp,
      logging_steps: loggingSteps,
      precision: precision as CreateTrainingRunRequest["precision"],
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
      work_dir: workDir.trim() || "/share",
      cleanup_checkpoints: cleanupCheckpoints,
      augment_techniques: augmentTechniques,
      augment_prob: augmentProb,
      report_to: [
        ...(wandbOn ? (["wandb"] as const) : []),
        ...(mlflowOn ? (["mlflow"] as const) : []),
      ],
      wandb_credential_id: wandbCredId || null,
      mlflow_credential_id: mlflowCredId || null,
      wandb_project: wandbOn ? wandbProject.trim() || null : null,
      wandb_entity: wandbOn ? wandbEntity.trim() || null : null,
      mlflow_tracking_uri: mlflowOn ? mlflowUri.trim() || null : null,
      mlflow_experiment: mlflowOn ? mlflowExperiment.trim() || null : null,
    };

    setSubmitting(true);
    try {
      const created = await gateway.createTrainingRun(body);
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
        <p className="mt-1 text-sm text-muted-foreground">
          {isTts
            ? "Finetune a Qwen3 + NeuCodec TTS model on a dataset. Audio is tokenized + packed, then trained as a causal LM (loss-only; metrics to W&B/MLflow)."
            : "Finetune a Whisper model on a dataset. WER + CER are evaluated each epoch; training stops at the max-epoch cap or early on patience."}
        </p>
        {fromId && (
          <p className="mt-2 inline-flex items-center gap-1.5 rounded-md border border-border bg-muted/40 px-2.5 py-1 text-xs text-muted-foreground">
            {prefilling ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
            {prefilling
              ? `Loading config from ${fromId}…`
              : <>Pre-filled from <span className="font-mono">{fromId}</span> — review and tweak before creating.</>}
          </p>
        )}
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
                  {datasets.map((d) => (
                    <SelectItem key={d.id} value={d.id}>
                      {d.name} · {d.kind}
                      {d.id === datasetId ? " — its own test split" : ""}
                    </SelectItem>
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
              {testDatasetId === datasetId && datasetId !== "" && (
                <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
                  Same as training — evaluation uses this dataset&apos;s{" "}
                  <span className="font-mono">test</span>/<span className="font-mono">validation</span>{" "}
                  rows (its <span className="font-mono">split</span> column). Falls back to a seeded
                  hold-out if it has none.
                </p>
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
        <div className="mb-5 grid max-w-xl grid-cols-1 gap-x-4 gap-y-5 sm:grid-cols-2">
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
          <FieldWrap label="Log loss every N steps" hint="Streams a training-loss point every N steps (@@STEP) for the live loss curve. Smaller = smoother, more log lines.">
            <NumberField min={1} value={loggingSteps} onChange={setLoggingSteps} />
          </FieldWrap>
          {!isTts && (
            <FieldWrap label="WER / CER text" hint="Normalize (Whisper-style: lowercase, strip punctuation, spell out numbers) before scoring, or score raw text.">
              <label className="flex cursor-pointer items-center gap-2 text-sm">
                <input type="checkbox" checked={normalizeText} onChange={(e) => setNormalizeText(e.target.checked)}
                  className="h-4 w-4 accent-primary" />
                <span>Normalize before WER/CER</span>
              </label>
            </FieldWrap>
          )}
        </div>

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
          {/* precision — single combo, or a multi-select to sweep over combos.
              "<load dtype>-<AMP train dtype>". */}
          {sweepOn ? (
            <FieldWrap label="Precisions (sweep)" hint="Load · AMP combos to try — one trial each.">
              <PrecisionMultiSelect selected={sweepPrecisions} onChange={setSweepPrecisions} />
            </FieldWrap>
          ) : (
            <FieldWrap label="Precision" hint="Weight load dtype · mixed-precision (AMP) train dtype.">
              <Select value={precision} onValueChange={setPrecision}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {PRECISIONS.map((p) => <SelectItem key={p.value} value={p.value}>{p.label}</SelectItem>)}
                </SelectContent>
              </Select>
            </FieldWrap>
          )}
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
          {sweepOn ? (
            <FieldWrap label="Weight decay" hint="AdamW L2 — e.g. 0, 0.01, 0.1">
              <Input className="font-mono" placeholder="0, 0.01, 0.1" value={sweepWeightDecay} onChange={(e) => setSweepWeightDecay(e.target.value)} />
            </FieldWrap>
          ) : (
            <FieldWrap label="Weight decay (AdamW)" hint="L2 regularization. 0 = off.">
              <Input className="font-mono" type="number" min={0} step={0.01} value={weightDecay}
                onChange={(e) => setWeightDecay(Math.max(0, Number(e.target.value) || 0))} />
            </FieldWrap>
          )}

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

        {/* LoRA + freeze-encoder (ASR / Whisper) */}
        {!isTts && (
          <div className="mt-4 space-y-3 border-t border-border pt-4">
            <div className="flex flex-wrap items-center gap-x-8 gap-y-2">
              <label className="flex cursor-pointer items-center gap-2 text-sm">
                <input type="checkbox" checked={useLora} onChange={(e) => setUseLora(e.target.checked)}
                  className="h-4 w-4 accent-primary" />
                <span className="font-medium">Use LoRA</span>
                <span className="text-xs text-muted-foreground">adapters on q/v attention, merged into the base at save</span>
              </label>
              {sweepOn ? (
                <label className="flex cursor-pointer items-center gap-2 text-sm">
                  <input type="checkbox" checked={sweepFreeze} onChange={(e) => setSweepFreeze(e.target.checked)}
                    className="h-4 w-4 accent-primary" />
                  <span className="font-medium">Sweep freeze-encoder</span>
                  <span className="text-xs text-muted-foreground">compare frozen vs full (×2 trials)</span>
                </label>
              ) : (
                <label className="flex cursor-pointer items-center gap-2 text-sm">
                  <input type="checkbox" checked={freezeEncoder} onChange={(e) => setFreezeEncoder(e.target.checked)}
                    className="h-4 w-4 accent-primary" />
                  <span className="font-medium">Freeze encoder</span>
                  <span className="text-xs text-muted-foreground">train the decoder only</span>
                </label>
              )}
            </div>
            {useLora && (
              <div className="grid grid-cols-1 gap-x-4 gap-y-4 sm:grid-cols-3">
                {sweepOn ? (
                  <FieldWrap label="LoRA r" hint="e.g. 8, 16, 32">
                    <Input className="font-mono" placeholder="8, 16, 32" value={sweepLoraR} onChange={(e) => setSweepLoraR(e.target.value)} />
                  </FieldWrap>
                ) : (
                  <FieldWrap label="LoRA r" hint="Adapter rank."><NumberField min={1} value={loraR} onChange={setLoraR} /></FieldWrap>
                )}
                <FieldWrap
                  label="LoRA alpha ratio"
                  hint={sweepOn
                    ? `alpha = round(r × ${loraAlphaRatio}) per trial (e.g. r 32 → ${Math.round(32 * loraAlphaRatio)})`
                    : `alpha = ${Math.round(loraR * loraAlphaRatio)} (r ${loraR} × ${loraAlphaRatio}). 2× is typical.`}>
                  <Input className="font-mono" type="number" min={0} step={0.5} value={loraAlphaRatio}
                    onChange={(e) => setLoraAlphaRatio(Math.max(0, Number(e.target.value) || 0))} />
                </FieldWrap>
                <FieldWrap label="LoRA dropout" hint="0–1, on the adapters.">
                  <Input className="font-mono" type="number" min={0} max={1} step={0.01} value={loraDropout}
                    onChange={(e) => setLoraDropout(Math.max(0, Math.min(1, Number(e.target.value) || 0)))} />
                </FieldWrap>
              </div>
            )}
          </div>
        )}

        {sweepOn && (
          <p className="mt-4 text-[11px] leading-snug text-muted-foreground">
            Trials are pinned via <span className="font-mono">CUDA_VISIBLE_DEVICES</span> across the GPUs from{" "}
            <span className="font-medium">Run on</span> (the pin on a VM, or the GPU count on RunPod), {gpusPerTrial} each —
            e.g. GPUs <span className="font-mono">6,7</span> with 1/trial → 2 at a time. Best model chosen by{" "}
            {isTts ? "lowest final loss" : "lowest WER/CER"}; each trial&apos;s checkpoint lands under{" "}
            <span className="font-mono">…/trials/&lt;i&gt;/</span>.
          </p>
        )}
        <div className="mt-5 space-y-1.5 border-t border-border pt-4">
          <Label className="text-xs">Audio augmentation (training only)</Label>
          <p className="text-xs text-muted-foreground">
            Select techniques to harden the model against noisy / phone audio. One enabled
            technique is applied at random to each augmented training clip; eval is never augmented.
          </p>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {AUG_OPTIONS.map((o) => {
              const on = augmentTechniques.includes(o.id);
              return (
                <button
                  key={o.id}
                  type="button"
                  title={o.desc}
                  onClick={() =>
                    setAugmentTechniques((prev) =>
                      prev.includes(o.id) ? prev.filter((x) => x !== o.id) : [...prev, o.id],
                    )
                  }
                  className={cn(
                    "rounded-md border px-2.5 py-1.5 text-left text-xs transition-colors",
                    on ? "border-primary/60 bg-primary/10 text-foreground"
                       : "border-border text-muted-foreground hover:border-primary/40 hover:bg-muted/40",
                  )}
                >
                  <span className="block font-medium">{o.label}</span>
                  <span className="block truncate text-[10px] opacity-70">{o.desc}</span>
                </button>
              );
            })}
          </div>
          {augmentTechniques.length > 0 && (
            <div className="flex items-center gap-2 pt-1">
              <Label htmlFor="aug-prob" className="text-xs">Augment probability</Label>
              <Input id="aug-prob" type="number" min={0} max={1} step={0.05}
                className="h-8 w-24 font-mono text-xs"
                value={augmentProb}
                onChange={(e) => setAugmentProb(Math.max(0, Math.min(1, Number(e.target.value) || 0)))} />
              <span className="text-[11px] text-muted-foreground">
                fraction of training clips augmented ({augmentTechniques.length} technique{augmentTechniques.length === 1 ? "" : "s"})
              </span>
            </div>
          )}
          {sweepOn && !isTts && (
            <label
              className={cn(
                "mt-2 flex items-start gap-2 rounded-md border px-3 py-2 text-xs",
                augmentTechniques.length
                  ? "cursor-pointer border-border hover:bg-muted/40"
                  : "cursor-not-allowed border-dashed border-border opacity-60",
              )}
            >
              <input
                type="checkbox"
                className="mt-0.5"
                checked={sweepAugment && augmentTechniques.length > 0}
                disabled={augmentTechniques.length === 0}
                onChange={(e) => setSweepAugment(e.target.checked)}
              />
              <span>
                <span className="block font-medium text-foreground">
                  Sweep augmentation — compare augment vs. no-augment
                </span>
                <span className="block text-[11px] text-muted-foreground">
                  {augmentTechniques.length
                    ? "Adds an extra ×2 to the trial grid: one arm with the selected techniques, one without."
                    : "Select at least one technique above to enable this."}
                </span>
              </span>
            </label>
          )}
        </div>
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
          <Input id="train-cuda"
            className={cn("font-mono text-xs", vdError && "border-destructive focus-visible:ring-destructive")}
            placeholder="e.g. 0,1 (empty = all GPUs)"
            value={visibleDevices} onChange={(e) => setVisibleDevices(e.target.value)} />
          {vdError ? (
            <p className="text-xs text-destructive">{vdError}</p>
          ) : (
            <p className="text-xs text-muted-foreground">
              Pins which GPUs the trainer uses. Empty = all visible GPUs.
              {gpuBound > 0 && (
                <> {target === "vm" ? "This VM" : "The pod"} has {gpuBound} GPU{gpuBound === 1 ? "" : "s"} — valid indices <span className="font-mono">0–{gpuBound - 1}</span>.</>
              )}
            </p>
          )}
        </div>

        {/* DDP — only meaningful for a multi-GPU single run (sweeps pin 1/trial) */}
        {!sweepOn && !isTts && (() => {
          const n = visibleDevices.trim()
            ? visibleDevices.split(",").filter((x) => x.trim()).length
            : gpuBound;
          if (n <= 1) return null;
          return (
            <label className="mt-4 flex cursor-pointer items-start gap-2.5 rounded-md border border-border bg-muted/30 px-3 py-2.5 text-sm hover:bg-muted/50">
              <input type="checkbox" checked={useDdp} onChange={(e) => setUseDdp(e.target.checked)}
                className="mt-0.5 h-4 w-4 shrink-0 cursor-pointer accent-primary" />
              <span className="min-w-0">
                <span className="font-medium">Distributed training (DDP)</span>
                <span className="block text-xs text-muted-foreground">
                  One process per GPU via <span className="font-mono">torchrun</span> ({n} GPUs) — faster + balanced than
                  single-process DataParallel. Uncheck to use DataParallel.
                </span>
              </span>
            </label>
          );
        })()}

        <div className="mt-4 space-y-1.5">
          <Label htmlFor="train-workdir" className="text-xs">Checkpoint / temp directory</Label>
          <Input id="train-workdir" className="font-mono text-xs" placeholder="/share"
            value={workDir} onChange={(e) => setWorkDir(e.target.value)} />
          <p className="text-xs text-muted-foreground">
            Roomy dir on the VM for checkpoints + temp (<span className="font-mono">TMPDIR</span>). Default{" "}
            <span className="font-mono">/share</span> — avoid <span className="font-mono">/tmp</span> (small disk).
            The best model is uploaded to S3 regardless.
          </p>
        </div>

        <label className="mt-4 flex cursor-pointer items-start gap-2.5 rounded-md border border-border bg-muted/30 px-3 py-2.5 text-sm hover:bg-muted/50">
          <input type="checkbox" checked={cleanupCheckpoints}
            onChange={(e) => setCleanupCheckpoints(e.target.checked)}
            className="mt-0.5 h-4 w-4 shrink-0 cursor-pointer accent-primary" />
          <span className="min-w-0">
            <span className="font-medium">Clean checkpoints after run</span>
            <span className="block text-xs text-muted-foreground">
              Delete the checkpoint/work dir on the VM when the run ends (the best model is already on S3).
              Keeps the disk from filling across runs.
            </span>
          </span>
        </label>

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
        description="Push per-epoch metrics to W&B and/or MLflow via HF Trainer. Pick a named credential — manage them on the Secrets page (Tracking credentials).">
        <div className="space-y-4">
          <Grid>
            <FieldWrap label="W&B credential" hint={wandbCreds.length ? "Select to enable W&B." : "None registered — add one under Secrets."}>
              <Select value={wandbCredId || "__off__"} onValueChange={(v) => setWandbCredId(v === "__off__" ? "" : v)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__off__">— Off —</SelectItem>
                  {wandbCreds.map((c) => (
                    <SelectItem key={c.id} value={c.id}>{c.name} · {c.preview}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </FieldWrap>
            {wandbOn && (
              <>
                <FieldWrap label="W&B project"><Input className="font-mono" placeholder="whisper-finetune" value={wandbProject} onChange={(e) => setWandbProject(e.target.value)} /></FieldWrap>
                <FieldWrap label="W&B entity (optional)"><Input className="font-mono" placeholder="my-team" value={wandbEntity} onChange={(e) => setWandbEntity(e.target.value)} /></FieldWrap>
              </>
            )}
          </Grid>

          <div className="border-t border-border pt-4">
            <Grid>
              <FieldWrap label="MLflow credential" hint={mlflowCreds.length ? "Select to enable MLflow (uri + user/pass)." : "None registered — add one under Secrets."}>
                <Select value={mlflowCredId || "__off__"} onValueChange={(v) => setMlflowCredId(v === "__off__" ? "" : v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__off__">— Off —</SelectItem>
                    {mlflowCreds.map((c) => (
                      <SelectItem key={c.id} value={c.id}>{c.name} · {c.preview}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FieldWrap>
              {mlflowOn && (
                <>
                  <FieldWrap label="Experiment" hint="MLFLOW_EXPERIMENT_NAME, e.g. test-classification.">
                    <Input className="font-mono" placeholder="whisper-finetune" value={mlflowExperiment} onChange={(e) => setMlflowExperiment(e.target.value)} />
                  </FieldWrap>
                  <FieldWrap label="Tracking URI override (optional)" hint="Overrides the credential's URI for this run.">
                    <Input className="font-mono" placeholder="https://mlflow.aies.scicom.dev" value={mlflowUri} onChange={(e) => setMlflowUri(e.target.value)} />
                  </FieldWrap>
                </>
              )}
            </Grid>
          </div>
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
        <Button type="submit" disabled={submitting || !hasStorage || !!vdError} className="min-w-36">
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

// Multi-select dropdown over the precision combos (sweep mode). Stays open on
// toggle so several can be picked.
function PrecisionMultiSelect({ selected, onChange }: {
  selected: string[]; onChange: (v: string[]) => void;
}) {
  const toggle = (v: string) =>
    onChange(selected.includes(v) ? selected.filter((x) => x !== v) : [...selected, v]);
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button type="button" variant="outline" className="w-full justify-between font-normal">
          <span className="truncate">
            {selected.length ? `${selected.length} selected` : "Pick precisions…"}
          </span>
          <ChevronDown className="h-4 w-4 opacity-50" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-64">
        {PRECISIONS.map((p) => (
          <DropdownMenuCheckboxItem
            key={p.value}
            checked={selected.includes(p.value)}
            onCheckedChange={() => toggle(p.value)}
            onSelect={(e) => e.preventDefault()}
          >
            {p.label}
          </DropdownMenuCheckboxItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
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
