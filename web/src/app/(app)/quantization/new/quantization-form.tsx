"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { AlertTriangle, Cpu, Database, KeyRound, Library, Loader2, Server, Shrink } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type {
  CreateQuantizationJobRequest,
  DatasetRecord,
  GlobalEnvRecord,
  GpuTypeOption,
  ProviderRecord,
  StorageRecord,
} from "@/lib/types";
import { FormFooter, FormShell } from "@/components/form-shell";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NumberField } from "@/components/ui/number-field";
import { Switch } from "@/components/ui/switch";
import { SearchableSelect } from "@/components/ui/searchable-select";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RegionSelect } from "@/components/region-select";

const GPU_COUNT_CHOICES = [1, 2, 4, 8] as const;
const RUNPOD_GPU_FALLBACK: GpuTypeOption[] = [
  { id: "NVIDIA L40S", label: "L40S", vram_gb: 48, hint: "48 GB" },
  { id: "NVIDIA A100 80GB PCIe", label: "A100 80GB", vram_gb: 80, hint: "datacenter" },
  { id: "NVIDIA H100 80GB HBM3", label: "H100 80GB", vram_gb: 80, hint: "fastest" },
  { id: "NVIDIA H200", label: "H200", vram_gb: 141, hint: "141 GB" },
];
// Only text-bearing dataset kinds can calibrate an LLM (mirror the gateway's
// _CALIB_DATASET_KINDS). An audio / packed dataset has nothing to calibrate on.
const CALIB_KINDS = new Set(["hf", "llm", "upload", "s3"]);

type SchemeInfo = { label: string; needs_calibration: boolean };

export function QuantizationForm() {
  const router = useRouter();

  const [schemes, setSchemes] = useState<Record<string, SchemeInfo>>({});
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [providers, setProviders] = useState<ProviderRecord[]>([]);
  const [storages, setStorages] = useState<StorageRecord[]>([]);
  const [secrets, setSecrets] = useState<GlobalEnvRecord[]>([]);
  const [gpuOptions, setGpuOptions] = useState<GpuTypeOption[]>(RUNPOD_GPU_FALLBACK);

  // Core
  const [name, setName] = useState("");
  const [sourceModel, setSourceModel] = useState("");
  const [scheme, setScheme] = useState("fp8-dynamic");

  // Calibration
  const [datasetId, setDatasetId] = useState("");
  const [numSamples, setNumSamples] = useState(512);
  const [maxSeqLen, setMaxSeqLen] = useState(2048);
  const [textField, setTextField] = useState("");
  const [messagesField, setMessagesField] = useState("");

  // Recipe knobs
  const [ignoreLayers, setIgnoreLayers] = useState("lm_head");
  const [quantizeVision, setQuantizeVision] = useState(false);
  const [smoothing, setSmoothing] = useState(0.8);
  const [dampening, setDampening] = useState(0.01);

  // Compute
  const [runOn, setRunOn] = useState<"cloud" | "vm">("cloud");
  const [runpodProviderId, setRunpodProviderId] = useState("");
  const [vmProviderId, setVmProviderId] = useState("");
  const [gpuType, setGpuType] = useState("NVIDIA H100 80GB HBM3");
  const [gpuCount, setGpuCount] = useState(1);
  const [secureCloud, setSecureCloud] = useState(true);
  const [dataCenterId, setDataCenterId] = useState("");
  const [diskGb, setDiskGb] = useState(60);
  const [volumeGb, setVolumeGb] = useState(120);
  const [visibleDevices, setVisibleDevices] = useState("");

  // HF token for the job (gated/private source model + push to Hub) — pasted, or a
  // global-secret reference (default; rotates without editing this job).
  const [hfTokenMode, setHfTokenMode] = useState<"paste" | "secret">("secret");
  const [hfToken, setHfToken] = useState("");
  const [hfTokenSecret, setHfTokenSecret] = useState("");

  // Output
  const [storageId, setStorageId] = useState("");
  const [hfPushRepo, setHfPushRepo] = useState("");
  const [hfPrivate, setHfPrivate] = useState(true);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    gateway.listQuantizationSchemes().then((r) => setSchemes(r.schemes)).catch(() => {});
    gateway.listDatasets().then(setDatasets).catch(() => {});
    gateway.listStorage().then(setStorages).catch(() => {});
    gateway.listGlobalEnv().then(setSecrets).catch(() => {});
    gateway
      .listProviders()
      .then((ps) => {
        setProviders(ps);
        const firstRunpod = ps.find((p) => p.kind === "runpod");
        if (firstRunpod) setRunpodProviderId((cur) => cur || firstRunpod.id);
      })
      .catch(() => {});
    gateway
      .listRunpodGpuTypes()
      .then((rows) => {
        if (rows.length) setGpuOptions(rows);
      })
      .catch(() => {});
  }, []);

  const needsCalib = schemes[scheme]?.needs_calibration ?? false;
  const calibDatasets = useMemo(() => datasets.filter((d) => CALIB_KINDS.has(d.kind)), [datasets]);
  const selectedDataset = calibDatasets.find((d) => d.id === datasetId);
  const isChat = selectedDataset?.kind === "llm";
  const vmProviders = useMemo(() => providers.filter((p) => p.kind === "vm"), [providers]);
  const runpodProviders = useMemo(() => providers.filter((p) => p.kind === "runpod"), [providers]);
  const s3Storages = useMemo(() => storages.filter((s) => s.kind === "s3"), [storages]);
  // Global-secret keys the HF token can reference (keys only; values stay server-side).
  const hfSecretKeys = useMemo(() => secrets.filter((s) => s.is_secret).map((s) => s.key), [secrets]);

  const gpuBound =
    runOn === "vm"
      ? vmProviders.find((p) => p.id === vmProviderId)?.gpu_count ?? 0
      : gpuCount;
  const vdError = useMemo(() => visibleDevicesError(visibleDevices, gpuBound), [visibleDevices, gpuBound]);

  const canSubmit =
    name.trim().length > 0 &&
    sourceModel.trim().length > 0 &&
    (!needsCalib || datasetId) &&
    (runOn === "cloud" ? !!runpodProviderId : !!vmProviderId) &&
    !vdError &&
    !submitting;

  const submitHint = !name.trim()
    ? "Name your job."
    : !sourceModel.trim()
      ? "Enter the source model repo (owner/name)."
      : needsCalib && !datasetId
        ? `Scheme "${scheme}" needs a calibration dataset.`
        : runOn === "cloud" && !runpodProviderId
          ? "Pick a RunPod account (or add one under GPU Providers)."
          : runOn === "vm" && !vmProviderId
            ? "Pick a VM provider."
            : vdError
              ? `Fix the GPU pin: ${vdError}`
              : `${scheme} · ${sourceModel.trim()}`;

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const providerId = runOn === "cloud" ? runpodProviderId : vmProviderId;
      const body: CreateQuantizationJobRequest = {
        name: name.trim(),
        source_model: sourceModel.trim(),
        scheme,
        calibration_dataset_id: needsCalib ? datasetId || null : null,
        num_calibration_samples: numSamples,
        max_seq_length: maxSeqLen,
        calib_text_field: textField.trim() || null,
        calib_messages_field: messagesField.trim() || null,
        ignore_layers: ignoreLayers.split(",").map((s) => s.trim()).filter(Boolean),
        quantize_vision: quantizeVision,
        smoothing_strength: smoothing,
        dampening_frac: dampening,
        hf_push_repo: hfPushRepo.trim() || null,
        hf_push_private: hfPrivate,
        hf_token: hfTokenMode === "paste" ? (hfToken.trim() || null) : null,
        hf_token_secret: hfTokenMode === "secret" ? (hfTokenSecret || null) : null,
        provider_id: providerId || null,
        storage_id: storageId || null,
        gpu_type: gpuType,
        gpu_count: gpuCount,
        visible_devices: visibleDevices.trim() || null,
        secure_cloud: secureCloud,
        data_center_id: dataCenterId || null,
        disk_gb: diskGb,
        volume_gb: volumeGb,
      };
      const job = await gateway.createQuantizationJob(body);
      toast.success("Quantization job queued");
      router.push(`/quantization/${encodeURIComponent(job.id)}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  };

  return (
    <FormShell>
    <form onSubmit={onSubmit} className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">New quantization job</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Compress an LLM with llm-compressor — pull the model from Hugging Face, quantize it
          (data-free or calibrated on a text dataset), and store the compressed-tensors model
          (loadable by vLLM) on S3, optionally pushing it back to the Hub.
        </p>
      </div>

      {/* Model */}
      <Section icon={<Shrink className="h-4 w-4" />} title="Model"
        description="The Hugging Face model to pull and quantize.">
        <Grid>
          <FieldWrap label="Source model" hint="Any causal LLM on the Hub, e.g. Qwen/Qwen3-8B. Private / gated repos: set an HF token below.">
            <Input
              className="font-mono"
              placeholder="owner/model"
              value={sourceModel}
              onChange={(e) => setSourceModel(e.target.value)}
            />
          </FieldWrap>
          <FieldWrap label="Job name">
            <Input className="font-mono" placeholder="e.g. qwen3-8b-fp8" value={name} onChange={(e) => setName(e.target.value)} />
          </FieldWrap>
        </Grid>
      </Section>

      {/* HuggingFace token — pulls a gated/private SOURCE model and authenticates the
          optional push to the Hub. A global secret (recommended — rotates without
          editing this job) or a pasted token. Mirrors autotrain/new's HF-token card. */}
      <Section icon={<KeyRound className="h-4 w-4" />} title="HuggingFace token"
        description="For a gated / private source model and pushing the quantized model to the Hub. Use a global secret or paste a token. Empty falls back to the org HF_TOKEN secret.">
        <div className="space-y-2">
          <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
            {(["secret", "paste"] as const).map((src) => (
              <button
                key={src}
                type="button"
                onClick={() => setHfTokenMode(src)}
                className={cn(
                  "rounded px-2.5 py-1 transition-colors",
                  hfTokenMode === src ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
                )}
              >
                {src === "secret" ? "Global secret" : "Paste a token"}
              </button>
            ))}
          </div>

          {hfTokenMode === "secret" ? (
            hfSecretKeys.length > 0 ? (
              <div className="space-y-1.5">
                <Label htmlFor="q-hf-secret" className="text-xs uppercase tracking-wide text-muted-foreground">Global secret</Label>
                <Select value={hfTokenSecret} onValueChange={setHfTokenSecret}>
                  <SelectTrigger id="q-hf-secret"><SelectValue placeholder="Select a secret (e.g. HF_TOKEN)" /></SelectTrigger>
                  <SelectContent>
                    {hfSecretKeys.map((k) => (
                      <SelectItem key={k} value={k} className="font-mono text-xs">{k}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Resolved from{" "}
                  <Link href="/admin/secrets" className="underline underline-offset-2 hover:text-foreground">Secrets</Link>{" "}
                  at run time and injected as <span className="font-mono">HF_TOKEN</span>.
                </p>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">
                No global secrets yet. Add one under{" "}
                <Link href="/admin/secrets" className="underline underline-offset-2 hover:text-foreground">Secrets</Link>{" "}
                (e.g. <span className="font-mono">HF_TOKEN</span>), then pick it here — or switch to{" "}
                <span className="font-medium">Paste a token</span>.
              </p>
            )
          ) : (
            <div className="space-y-1.5">
              <Label htmlFor="q-hf-token" className="text-xs uppercase tracking-wide text-muted-foreground">Token</Label>
              <Input
                id="q-hf-token"
                type="password"
                autoComplete="off"
                className="font-mono text-xs"
                placeholder="hf_…"
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Sent with this job, stored Fernet-encrypted (never returned). Prefer a global secret for shared / rotating tokens.
              </p>
            </div>
          )}
        </div>
      </Section>

      {/* Scheme */}
      <Section icon={<Shrink className="h-4 w-4" />} title="Quantization scheme"
        description="How weights/activations are compressed. Calibrated schemes fit their scales on a small text dataset; data-free schemes need nothing.">
        <Grid>
          <FieldWrap
            label="Scheme"
            hint={needsCalib
              ? "Calibrated — pick a calibration dataset below."
              : "Data-free — no calibration dataset required."}
          >
            <Select value={scheme} onValueChange={setScheme}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {Object.entries(schemes).map(([id, info]) => (
                  <SelectItem key={id} value={id}>
                    {info.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </FieldWrap>
          <FieldWrap label="Ignore layers" hint="Comma-separated modules kept in full precision (lm_head is typical).">
            <Input
              className="font-mono"
              placeholder="lm_head"
              value={ignoreLayers}
              onChange={(e) => setIgnoreLayers(e.target.value)}
            />
          </FieldWrap>
          <FieldWrap label="Quantize vision tower" hint="Vision-language models only. Off (default) keeps the vision tower / multimodal projector in full precision — a quantized vision tower is not vLLM-servable. Turn on only if you know your runtime supports it.">
            <div className="flex h-9 items-center">
              <Switch checked={quantizeVision} onCheckedChange={setQuantizeVision} />
              <span className="ml-2 text-xs text-muted-foreground">{quantizeVision ? "Quantize vision (advanced)" : "Keep vision full-precision"}</span>
            </div>
          </FieldWrap>
          {scheme === "w8a8-int8" && (
            <FieldWrap label="SmoothQuant strength" hint="Migrates activation outliers into weights (0–1). 0.8 is a good default.">
              <NumberField min={0} max={1} allowDecimal value={smoothing} onChange={setSmoothing} />
            </FieldWrap>
          )}
          {(scheme === "w4a16" || scheme === "w8a8-int8") && (
            <FieldWrap label="GPTQ dampening fraction" hint="Hessian dampening for GPTQ. 0.01 is standard.">
              <NumberField min={0} max={1} allowDecimal value={dampening} onChange={setDampening} />
            </FieldWrap>
          )}
        </Grid>
      </Section>

      {/* Calibration dataset */}
      {needsCalib && (
        <Section icon={<Library className="h-4 w-4" />} title="Calibration dataset"
          description="A few hundred text samples from your Datasets are enough to fit the quantization scales.">
          <Grid>
            <FieldWrap label="Dataset" hint="Text datasets only (HuggingFace, LLM chat, or an uploaded/S3 text file).">
              <SearchableSelect
                value={datasetId}
                onChange={setDatasetId}
                options={calibDatasets.map((d) => ({
                  value: d.id,
                  label: d.name,
                  hint: `${d.num_rows != null ? `${d.num_rows} rows · ` : ""}${d.kind}`,
                  group: d.kind,
                }))}
                placeholder={calibDatasets.length ? "Pick a calibration dataset…" : "No text datasets yet"}
                searchPlaceholder="Search datasets by name…"
              />
              {calibDatasets.length === 0 && (
                <p className="mt-1.5 text-[11px] text-muted-foreground">
                  <Link href="/datasets/new" className="underline underline-offset-2 hover:text-foreground">
                    Add one
                  </Link>{" "}
                  under Datasets.
                </p>
              )}
            </FieldWrap>
            {isChat ? (
              <FieldWrap label="Messages column (optional)" hint="Chat column of [{role,content}]. Blank → auto-detect 'messages'.">
                <Input
                  className="font-mono"
                  placeholder="messages"
                  value={messagesField}
                  onChange={(e) => setMessagesField(e.target.value)}
                />
              </FieldWrap>
            ) : (
              <FieldWrap label="Text column (optional)" hint="Column holding the raw text. Blank → auto-detect (text/content/prompt…).">
                <Input
                  className="font-mono"
                  placeholder="text"
                  value={textField}
                  onChange={(e) => setTextField(e.target.value)}
                />
              </FieldWrap>
            )}
            <FieldWrap label="Calibration samples" hint="More = better scales, slower. 512 is typical.">
              <NumberField min={1} value={numSamples} onChange={setNumSamples} />
            </FieldWrap>
            <FieldWrap label="Max sequence length">
              <NumberField min={128} value={maxSeqLen} onChange={setMaxSeqLen} />
            </FieldWrap>
          </Grid>
        </Section>
      )}

      {/* Run on — pod card (mirrors autotrain/new) */}
      <Section icon={<Server className="h-4 w-4" />} title="Run on"
        description="Default cloud spawns a fresh RunPod pod, then tears it down. Bare metal runs on a VM you've registered under GPU Providers.">
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <button type="button" onClick={() => setRunOn("cloud")}
            className={cn("flex items-start gap-3 rounded-md border px-3 py-2.5 text-left text-sm transition-colors",
              runOn === "cloud" ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40")}>
            <Cpu className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <div className="font-medium">Default cloud (RunPod)</div>
              <div className="text-xs text-muted-foreground">Provision a fresh pod on demand, run, tear down. Pay-per-second.</div>
            </div>
          </button>
          <button type="button" onClick={() => setRunOn("vm")}
            className={cn("flex items-start gap-3 rounded-md border px-3 py-2.5 text-left text-sm transition-colors",
              runOn === "vm" ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40")}>
            <Server className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <div className="font-medium">Bare metal (VM)</div>
              <div className="text-xs text-muted-foreground">SSH onto a registered VM. No spin-up cost.</div>
            </div>
          </button>
        </div>
      </Section>

      {/* Pod — provider + hardware */}
      <Section icon={<Server className="h-4 w-4" />} title="Pod"
        description={runOn === "cloud"
          ? "GPU, count, and cloud tier for the pod."
          : "Which registered VM the job runs on. Hardware is fixed by the VM."}>
        <div className="space-y-5">
          {runOn === "cloud" ? (
            <>
              <FieldWrap label="RunPod account" hint="Which RunPod provider to bill against.">
                {runpodProviders.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    None registered.{" "}
                    <Link href="/providers/new" className="underline underline-offset-2 hover:text-foreground">
                      Add a RunPod account →
                    </Link>
                  </p>
                ) : (
                  <Select value={runpodProviderId} onValueChange={setRunpodProviderId}>
                    <SelectTrigger><SelectValue placeholder="Choose a RunPod account…" /></SelectTrigger>
                    <SelectContent>
                      {runpodProviders.map((p) => (
                        <SelectItem key={p.id} value={p.id}>
                          {p.name}{p.api_key_last4 ? ` · ****${p.api_key_last4}` : ""}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </FieldWrap>
              <FieldWrap label="Cloud tier" hint="Community is cheaper with variable hosts; Secure uses vetted hosts with more capacity.">
                <div className="grid grid-cols-2 gap-2">
                  {([true, false] as const).map((secure) => (
                    <button
                      key={String(secure)}
                      type="button"
                      onClick={() => setSecureCloud(secure)}
                      className={cn(
                        "rounded-md border p-3 text-left transition-colors",
                        secureCloud === secure ? "border-foreground/60 ring-1 ring-foreground/20" : "border-border hover:border-foreground/40",
                      )}
                    >
                      <div className="text-sm font-medium">{secure ? "Secure" : "Community"}</div>
                      <div className="mt-0.5 text-xs text-muted-foreground">
                        {secure ? "vetted hosts, more capacity" : "cheaper, variable hosts"}
                      </div>
                    </button>
                  ))}
                </div>
              </FieldWrap>
              <FieldWrap label="Region" hint="Pin the pod to a RunPod data center, or Auto to let RunPod pick any region with capacity.">
                <RegionSelect value={dataCenterId} onChange={setDataCenterId} className="text-sm" />
              </FieldWrap>
              <FieldWrap label="GPU" hint={vramNote(gpuOptions, gpuType, gpuCount)}>
                <div className="flex gap-2">
                  <SearchableSelect
                    className="flex-1"
                    value={gpuType}
                    onChange={setGpuType}
                    options={gpuOptions.map((g) => ({ value: g.id, label: g.label, hint: `${g.vram_gb} GB` }))}
                    placeholder="Choose a GPU"
                    searchPlaceholder="Search GPUs (e.g. h100, 80gb)…"
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
                <FieldWrap label="Volume (GB)" hint="Persistent volume for the model cache. 0 = no persistent storage.">
                  <NumberField min={0} value={volumeGb} onChange={setVolumeGb} />
                </FieldWrap>
              </div>
              <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>
                  Pick a GPU with enough VRAM to hold the full-precision model plus overhead —
                  quantization loads the unquantized weights first.
                </span>
              </div>
            </>
          ) : (
            <FieldWrap label="VM provider" hint="The registered VM the job SSHes onto. Hardware is fixed by the VM.">
              {vmProviders.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No VM providers registered. Add one at{" "}
                  <Link href="/providers/new" className="underline underline-offset-2 hover:text-foreground">
                    GPU Providers → New provider
                  </Link>.
                </p>
              ) : (
                <Select value={vmProviderId} onValueChange={setVmProviderId}>
                  <SelectTrigger><SelectValue placeholder="Pick a VM…" /></SelectTrigger>
                  <SelectContent>
                    {vmProviders.map((p) => (
                      <SelectItem key={p.id} value={p.id}>
                        {p.name}{p.gpu_count ? ` · ${p.gpu_count} GPU` : ""}{p.host ? ` · ${p.host}` : ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </FieldWrap>
          )}

          <FieldWrap
            label="CUDA_VISIBLE_DEVICES (optional)"
            hint={!vdError && gpuBound > 0
              ? `${runOn === "vm" ? "This VM" : "The pod"} has ${gpuBound} GPU${gpuBound === 1 ? "" : "s"} — valid indices 0–${gpuBound - 1}.`
              : undefined}
          >
            <Input
              className={cn("font-mono", vdError && "border-destructive focus-visible:ring-destructive")}
              placeholder="e.g. 0 (empty = all GPUs)"
              value={visibleDevices}
              onChange={(e) => setVisibleDevices(e.target.value)}
            />
            {vdError && <p className="text-[11px] text-destructive">{vdError}</p>}
          </FieldWrap>
        </div>
      </Section>

      {/* Output */}
      <Section icon={<Database className="h-4 w-4" />} title="Artifacts"
        description="Where logs + the compressed model are uploaded, and an optional Hugging Face push when the job finishes.">
        <Grid>
          <FieldWrap label="Storage (S3)" hint="Blank → the platform default bucket.">
            <Select value={storageId || "__default__"} onValueChange={(v) => setStorageId(v === "__default__" ? "" : v)}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="__default__">Platform default</SelectItem>
                {s3Storages.map((s) => (
                  <SelectItem key={s.id} value={s.id}>{s.name}{s.bucket ? ` · ${s.bucket}` : ""}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </FieldWrap>
          <FieldWrap label="Push to HF (optional)" hint="HuggingFace repo, e.g. you/model-fp8. Pushed when the job finishes, using the HF token above.">
            <Input
              className="font-mono"
              placeholder="org/model-fp8"
              value={hfPushRepo}
              onChange={(e) => setHfPushRepo(e.target.value)}
            />
          </FieldWrap>
          {hfPushRepo.trim() && (
            <FieldWrap label="Private repo">
              <div className="flex h-9 items-center">
                <Switch checked={hfPrivate} onCheckedChange={setHfPrivate} />
                <span className="ml-2 text-xs text-muted-foreground">{hfPrivate ? "Private" : "Public"}</span>
              </div>
            </FieldWrap>
          )}
        </Grid>
      </Section>

      <FormFooter error={error} hint={submitHint}>
        <Button type="button" variant="outline" onClick={() => router.push("/quantization")}>Cancel</Button>
        <Button type="submit" disabled={!canSubmit} className="min-w-36">
          {submitting ? (<><Loader2 className="h-4 w-4 animate-spin" /> Creating…</>)
            : (<><Shrink className="h-4 w-4" /> Start quantization</>)}
        </Button>
      </FormFooter>
    </form>
    </FormShell>
  );
}

function vramNote(opts: GpuTypeOption[], id: string, count: number): string | undefined {
  const g = opts.find((o) => o.id === id);
  if (!g) return undefined;
  const total = g.vram_gb * count;
  return `${total} GB VRAM${count > 1 ? ` · ×${count}` : ""}`;
}

function visibleDevicesError(visibleDevices: string, gpuBound: number): string | null {
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
}

function Section({ icon, title, description, children }: {
  icon: React.ReactNode; title: string; description?: string; children: React.ReactNode;
}) {
  return (
    // data-form-section feeds the FormShell scrollspy rail; scroll-mt keeps the
    // heading visible after a rail jump. Mirrors autotrain/new's Section.
    <Card data-form-section={title} className="scroll-mt-6">
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
        <Label className="text-xs uppercase tracking-wide text-muted-foreground">{label}</Label>
        {extra}
      </div>
      {children}
      {hint && <p className="text-[11px] leading-snug text-muted-foreground">{hint}</p>}
    </div>
  );
}
