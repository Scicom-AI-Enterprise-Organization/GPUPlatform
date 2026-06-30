"use client";

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { Check, Cpu, Loader2, RefreshCw, Server, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
import { AvailabilityBadge } from "@/components/availability-badge";
import { useGpuAvailability } from "@/lib/use-gpu-availability";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type {
  GlobalEnvRecord,
  GpuTypeOption,
  ProviderRecord,
  TrainingRunRecord,
  VmAvailability,
} from "@/lib/types";

const GPU_COUNT_CHOICES = [1, 2, 4, 8] as const;

// Fallback until the live catalog (/compute/runpod/gpu-types) lands.
const RUNPOD_GPU_FALLBACK: GpuTypeOption[] = [
  { id: "NVIDIA RTX A5000", label: "RTX A5000", vram_gb: 24, hint: "24 GB" },
  { id: "NVIDIA RTX A6000", label: "RTX A6000", vram_gb: 48, hint: "48 GB" },
  { id: "NVIDIA L40S", label: "L40S", vram_gb: 48, hint: "48 GB" },
  { id: "NVIDIA A100 80GB PCIe", label: "A100 80GB", vram_gb: 80, hint: "datacenter" },
  { id: "NVIDIA H100 80GB HBM3", label: "H100 80GB", vram_gb: 80, hint: "fastest" },
];

function gpuHint(vramGb: number, count: number): string {
  const total = vramGb * count;
  return `${total >= 100 ? Math.round(total) : total} GB VRAM${count > 1 ? ` · ×${count}` : ""}`;
}

type VmAvailState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ok"; data: VmAvailability }
  | { status: "error"; message: string };

// Export-to-Label as a tab: synthesize N clips from the finished TTS model and seed
// a Label-platform recording + MOS project. "Run on" mirrors serverless/new — a
// fresh RunPod pod (pick GPU type/count/tier) or a registered bare-metal VM.
export function LabelExportTab({
  run,
  onStarted,
}: {
  run: TrainingRunRecord;
  onStarted?: () => void;
}) {
  const lcfg = (run.config_json ?? {}) as Record<string, unknown>;
  const str = (k: string, d = ""): string => (typeof lcfg[k] === "string" ? (lcfg[k] as string) : d);
  const num = (k: string, d: number): number => (typeof lcfg[k] === "number" ? (lcfg[k] as number) : d);
  const arr = (k: string): string => (Array.isArray(lcfg[k]) ? (lcfg[k] as unknown[]).map(String).join(", ") : "");

  // ---- Label-platform creds + project knobs (prefilled from the run's config) ----
  const [urlMode, setUrlMode] = useState<"paste" | "secret">(str("label_base_url_secret") ? "secret" : "paste");
  const [url, setUrl] = useState(str("label_base_url") || "http://localhost:3002");
  const [urlSecret, setUrlSecret] = useState(str("label_base_url_secret"));
  const [tokenMode, setTokenMode] = useState<"paste" | "secret">(str("label_token_secret") ? "secret" : "paste");
  const [token, setToken] = useState("");
  const [tokenSecret, setTokenSecret] = useState(str("label_token_secret"));
  const [secrets, setSecrets] = useState<GlobalEnvRecord[]>([]);
  const [project, setProject] = useState(str("label_project_name"));
  const [samples, setSamples] = useState(num("label_samples", 32));
  const [axes, setAxes] = useState(arr("label_mos_axes") || "Naturalness, Intelligibility, Noise");
  const [speakers, setSpeakers] = useState(arr("label_speakers"));
  const [speakerPrefix, setSpeakerPrefix] = useState(!!lcfg.label_speaker_prefix);
  const [rejectKeywords, setRejectKeywords] = useState(arr("label_reject_keywords"));
  const [perSpeaker, setPerSpeaker] = useState(!!lcfg.label_per_speaker);

  // ---- Run-on (pod card) ----
  const [providers, setProviders] = useState<ProviderRecord[]>([]);
  const [target, setTarget] = useState<"cloud" | "vm">(str("label_run_on") === "cloud" ? "cloud" : "vm");
  const [vmProviderId, setVmProviderId] = useState(
    str("label_run_on") === "vm" ? str("label_provider_id") || run.provider_id || "" : run.provider_id || "",
  );
  const [runpodProviderId, setRunpodProviderId] = useState(
    str("label_run_on") === "cloud" ? str("label_provider_id") : "",
  );
  const [gpuType, setGpuType] = useState(str("label_gpu_type") || run.gpu_type || "NVIDIA L40S");
  const [gpuCount, setGpuCount] = useState(num("label_gpu_count", 1));
  const [secureCloud, setSecureCloud] = useState(typeof lcfg.label_secure_cloud === "boolean" ? (lcfg.label_secure_cloud as boolean) : true);
  const [diskGb, setDiskGb] = useState(num("label_disk_gb", 60));
  const [volumeGb, setVolumeGb] = useState(num("label_volume_gb", 80));
  const [visibleDevices, setVisibleDevices] = useState(str("label_visible_devices"));
  const [venvPath, setVenvPath] = useState(str("venv_path") || "/share/autotrain-tts");
  const [gpuOptions, setGpuOptions] = useState<GpuTypeOption[]>(RUNPOD_GPU_FALLBACK);

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  const [vmAvail, setVmAvail] = useState<VmAvailState>({ status: "idle" });
  const refreshVmAvail = useCallback(async (id: string) => {
    if (!id) return setVmAvail({ status: "idle" });
    setVmAvail({ status: "loading" });
    try {
      setVmAvail({ status: "ok", data: await gateway.getVmAvailability(id) });
    } catch (e) {
      setVmAvail({ status: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }, []);

  const vmProviders = useMemo(() => providers.filter((p) => p.kind === "vm"), [providers]);
  const runpodProviders = useMemo(() => providers.filter((p) => p.kind === "runpod"), [providers]);
  const gpuBound = useMemo(
    () => (target === "vm" ? vmProviders.find((p) => p.id === vmProviderId)?.gpu_count ?? 0 : gpuCount),
    [target, vmProviders, vmProviderId, gpuCount],
  );
  const availability = useGpuAvailability(gpuType, gpuCount, target === "cloud", secureCloud ? "SECURE" : "COMMUNITY");

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

  useEffect(() => {
    gateway.listGlobalEnv().then(setSecrets).catch(() => {});
    gateway
      .listProviders()
      .then((ps) => {
        setProviders(ps);
        // Auto-select the first registered RunPod account — no gateway-default fallback.
        const firstRunpod = ps.find((p) => p.kind === "runpod");
        if (firstRunpod) setRunpodProviderId((cur) => cur || firstRunpod.id);
      })
      .catch(() => {});
    gateway
      .listRunpodGpuTypes()
      .then((rows) => {
        if (!rows.length) return;
        setGpuOptions(rows);
        setGpuType((cur) => (rows.some((g) => g.id === cur) ? cur : rows[0].id));
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (target === "vm" && vmProviderId) refreshVmAvail(vmProviderId);
    else setVmAvail({ status: "idle" });
  }, [target, vmProviderId, refreshVmAvail]);

  const urlOk = urlMode === "paste" ? !!url.trim() : !!urlSecret;
  const tokenOk = tokenMode === "paste" ? !!token.trim() : !!tokenSecret;
  const running = run.result_json?.label_export?.status === "running";

  async function submit() {
    setErr(null);
    if (target === "vm" && !vmProviderId) return setErr("Pick a VM provider, or switch to cloud.");
    if (target === "cloud" && !runpodProviderId) return setErr("Select a RunPod provider — add one under GPU Providers.");
    if (vdError) return setErr(vdError);
    setBusy(true);
    try {
      await gateway.retryLabelExport(run.id, {
        base_url: urlMode === "paste" ? (url.trim() || undefined) : undefined,
        base_url_secret: urlMode === "secret" ? (urlSecret || null) : null,
        token: tokenMode === "paste" ? (token.trim() || undefined) : undefined,
        token_secret: tokenMode === "secret" ? (tokenSecret || null) : null,
        project_name: project.trim() || null,
        samples,
        mos_axes: axes.split(",").map((s) => s.trim()).filter(Boolean),
        speakers: speakers.split(",").map((s) => s.trim()).filter(Boolean),
        speaker_prefix: speakerPrefix,
        reject_keywords: rejectKeywords.split(/[,\n]/).map((s) => s.trim()).filter(Boolean),
        per_speaker: perSpeaker,
        run_on: target,
        provider_id: target === "vm" ? vmProviderId : runpodProviderId,
        gpu_type: gpuType,
        gpu_count: gpuCount,
        secure_cloud: secureCloud,
        disk_gb: diskGb,
        volume_gb: volumeGb,
        visible_devices: visibleDevices.trim() || null,
        venv_path: venvPath.trim() || null,
      });
      setDone(true);
      onStarted?.();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  if (done || running) {
    return (
      <p className="flex items-center gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2.5 text-sm text-emerald-700 dark:text-emerald-400">
        <Check className="h-4 w-4 shrink-0" />
        Export {running ? "is running" : "started"} — the run status shows “exporting to Label” and synthesis streams to
        the Logs tab; an “Open in Label” link appears on the Metrics tab when it finishes.
      </p>
    );
  }

  return (
    <div className="space-y-5">
      <p className="text-sm text-muted-foreground">
        Synthesize {samples} clip{samples === 1 ? "" : "s"} from this run&apos;s trained model and create a
        Label-platform recording project (MOS rating), seeded with them. Runs in the background; watch the Logs tab.
      </p>

      {/* Run on — same card as serverless/new */}
      <Section
        title="Run on"
        description="Default cloud spawns a fresh RunPod pod for the synthesis, then tears it down. Bare metal runs on a VM you've registered under GPU Providers."
      >
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <button type="button" onClick={() => setTarget("cloud")}
            className={cn("flex items-start gap-3 rounded-md border px-3 py-2.5 text-left text-sm transition-colors",
              target === "cloud" ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40")}>
            <Cpu className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <div className="font-medium">Default cloud (RunPod)</div>
              <div className="text-xs text-muted-foreground">Provision a fresh pod on demand, synthesize, tear down. Pay-per-second.</div>
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

      {/* Pod — provider + hardware (same card as serverless/new) */}
      <Section
        title="Pod"
        description={target === "cloud"
          ? "GPU, count, and cloud tier for the synthesis pod."
          : "Which registered VM the synthesis runs on. Hardware is fixed by the VM."}
      >
        <div className="space-y-3">
      {target === "vm" ? (
        <div className="space-y-1.5">
          <Label className="text-xs">VM provider</Label>
          {vmProviders.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No VM providers registered. Add one at{" "}
              <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">GPU Providers → New provider</a>.
            </p>
          ) : (
            <Select value={vmProviderId} onValueChange={setVmProviderId}>
              <SelectTrigger className="text-xs"><SelectValue placeholder="Pick a VM…" /></SelectTrigger>
              <SelectContent>
                {vmProviders.map((p) => (
                  <SelectItem key={p.id} value={p.id}>
                    {p.name}{p.gpu_count ? ` · ${p.gpu_count} GPU` : ""}{p.host ? ` · ${p.host}` : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          {vmProviderId && <VmAvailabilityRow state={vmAvail} onRefresh={() => refreshVmAvail(vmProviderId)} />}
        </div>
      ) : (
        <div className="space-y-3">
          <div className="space-y-1.5">
            <Label className="text-xs">RunPod account</Label>
            <Select value={runpodProviderId} onValueChange={setRunpodProviderId}>
              <SelectTrigger className="text-xs"><SelectValue placeholder="Choose a RunPod account…" /></SelectTrigger>
              <SelectContent>
                {runpodProviders.map((p) => (
                  <SelectItem key={p.id} value={p.id}>
                    {p.name}{p.api_key_last4 ? ` · ****${p.api_key_last4}` : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {runpodProviders.length === 0 && (
              <p className="text-[11px] text-muted-foreground">
                None registered. <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">Add a RunPod account →</a>
              </p>
            )}
          </div>
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <Label className="text-xs">GPU</Label>
              <AvailabilityBadge state={availability} count={gpuCount} />
            </div>
            <div className="flex gap-2">
              <SearchableSelect
                className="flex-1"
                value={gpuType}
                onChange={setGpuType}
                options={gpuOptions.map((g) => ({ value: g.id, label: g.label, hint: gpuHint(g.vram_gb, 1) }))}
                placeholder="Choose a GPU"
                searchPlaceholder="Search GPUs (e.g. h100, 24gb)…"
              />
              <Select value={String(gpuCount)} onValueChange={(v) => setGpuCount(Number.parseInt(v, 10))}>
                <SelectTrigger className="w-20 shrink-0"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {GPU_COUNT_CHOICES.map((n) => <SelectItem key={n} value={String(n)}>×{n}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs">Cloud tier</Label>
              <Select value={secureCloud ? "secure" : "community"} onValueChange={(v) => setSecureCloud(v === "secure")}>
                <SelectTrigger className="text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="secure">Secure</SelectItem>
                  <SelectItem value="community">Community</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Disk (GB)</Label>
              <NumberField min={20} value={diskGb} onChange={setDiskGb} />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Volume (GB)</Label>
              <NumberField min={0} value={volumeGb} onChange={setVolumeGb} />
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label className="text-xs">uv venv path (TTS)</Label>
          <Input className="font-mono text-xs" placeholder="/share/autotrain-tts"
            value={venvPath} onChange={(e) => setVenvPath(e.target.value)} />
        </div>
        <div className="space-y-1.5">
          <Label className="text-xs">CUDA_VISIBLE_DEVICES (optional)</Label>
          <Input className={cn("font-mono text-xs", vdError && "border-destructive focus-visible:ring-destructive")}
            placeholder="e.g. 0 (empty = all GPUs)"
            value={visibleDevices} onChange={(e) => setVisibleDevices(e.target.value)} />
          {vdError ? (
            <p className="text-[11px] text-destructive">{vdError}</p>
          ) : (
            gpuBound > 0 && (
              <p className="text-[11px] text-muted-foreground">
                {target === "vm" ? "This VM" : "The pod"} has {gpuBound} GPU{gpuBound === 1 ? "" : "s"} — valid indices{" "}
                <span className="font-mono">0–{gpuBound - 1}</span>.
              </p>
            )
          )}
        </div>
      </div>
        </div>
      </Section>

      {/* Label project — destination + project knobs */}
      <Section
        title="Label project"
        description="Where the synthesized clips land — the Label platform URL, an admin token, and the recording project's settings."
      >
        <div className="space-y-3">
        {/* Label platform URL */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-3">
            <label className="text-xs uppercase tracking-wide text-muted-foreground">Label platform URL</label>
            <div className="inline-flex overflow-hidden rounded-md border border-border text-xs">
              {(["paste", "secret"] as const).map((m) => (
                <button key={m} type="button" onClick={() => setUrlMode(m)}
                  className={cn("px-2.5 py-1 transition-colors",
                    urlMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                  {m === "paste" ? "Paste" : "From secret"}
                </button>
              ))}
            </div>
          </div>
          {urlMode === "paste" ? (
            <Input className="font-mono" value={url} placeholder="http://localhost:3002" onChange={(e) => setUrl(e.target.value)} />
          ) : (
            <Select value={urlSecret} onValueChange={setUrlSecret}>
              <SelectTrigger><SelectValue placeholder={secrets.length ? "Choose a secret" : "No secrets configured"} /></SelectTrigger>
              <SelectContent>
                {secrets.map((s) => (
                  <SelectItem key={s.key} value={s.key}>{s.key}{s.value_preview ? ` — ${s.value_preview}` : ""}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
        {/* API token */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-3">
            <label className="text-xs uppercase tracking-wide text-muted-foreground">API token</label>
            <div className="inline-flex overflow-hidden rounded-md border border-border text-xs">
              {(["paste", "secret"] as const).map((m) => (
                <button key={m} type="button" onClick={() => setTokenMode(m)}
                  className={cn("px-2.5 py-1 transition-colors",
                    tokenMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")}>
                  {m === "paste" ? "Paste" : "From secret"}
                </button>
              ))}
            </div>
          </div>
          {tokenMode === "paste" ? (
            <>
              <Input type="password" className="font-mono" value={token} placeholder="lpat_…" onChange={(e) => setToken(e.target.value)} />
              <p className="text-[11px] text-muted-foreground">Admin personal access token. Stored encrypted on the run.</p>
            </>
          ) : (
            <Select value={tokenSecret} onValueChange={setTokenSecret}>
              <SelectTrigger><SelectValue placeholder={secrets.some((s) => s.is_secret) ? "Choose a secret" : "No secrets configured"} /></SelectTrigger>
              <SelectContent>
                {secrets.filter((s) => s.is_secret).map((s) => (
                  <SelectItem key={s.key} value={s.key}>{s.key}{s.value_preview ? ` — ${s.value_preview}` : ""}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <label className="text-xs uppercase tracking-wide text-muted-foreground">Project name</label>
            <Input value={project} placeholder={`${run.name}-eval`} onChange={(e) => setProject(e.target.value)} />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs uppercase tracking-wide text-muted-foreground">Samples</label>
            <Input type="number" min={1} value={samples}
              onChange={(e) => setSamples(Math.max(1, Number.parseInt(e.target.value, 10) || 1))} />
          </div>
        </div>
        <div className="space-y-1.5">
          <label className="text-xs uppercase tracking-wide text-muted-foreground">MOS axes</label>
          <Input value={axes} placeholder="Naturalness, Intelligibility, Noise" onChange={(e) => setAxes(e.target.value)} />
        </div>
        <div className="space-y-1.5">
          <label className="text-xs uppercase tracking-wide text-muted-foreground">Reject keywords (optional)</label>
          <Input value={rejectKeywords} placeholder="EMGS, E M G S, Husein" onChange={(e) => setRejectKeywords(e.target.value)} />
          <p className="text-xs text-muted-foreground">
            Comma- or newline-separated. Text samples containing any phrase are dropped (case-insensitive, spacing-agnostic).
          </p>
        </div>
        <div className="space-y-1.5">
          <label className="text-xs uppercase tracking-wide text-muted-foreground">Speaker names (optional)</label>
          <Input value={speakers} placeholder="speakerA, speakerB" onChange={(e) => setSpeakers(e.target.value)} />
          <p className="text-xs text-muted-foreground">
            {perSpeaker
              ? `Comma-separated. One project per speaker, each from that speaker's own clips (${samples} per speaker). Names must match the dataset's speaker labels.`
              : `Comma-separated. Balances the clips evenly across these voices (e.g. 2 speakers + ${samples} samples → ${Math.floor(samples / 2)} each). Blank → the dataset's original voices.`}
          </p>
        </div>
        <label className="flex cursor-pointer items-center gap-2 text-sm">
          <input type="checkbox" checked={perSpeaker} onChange={(e) => setPerSpeaker(e.target.checked)} className="h-4 w-4 accent-primary" />
          <span>Separate project per speaker <span className="text-muted-foreground">(each from that speaker&apos;s own clips)</span></span>
        </label>
        <label className="flex cursor-pointer items-center gap-2 text-sm">
          <input type="checkbox" checked={speakerPrefix} onChange={(e) => setSpeakerPrefix(e.target.checked)} className="h-4 w-4 accent-primary" />
          <span>Prefix transcription with speaker name <span className="text-muted-foreground">(e.g. “TM_Mandarin: …”)</span></span>
        </label>
        </div>
      </Section>

      <div className="flex items-center justify-end gap-3">
        {err && <p className="mr-auto text-sm text-destructive">{err}</p>}
        <Button onClick={submit} disabled={busy || !urlOk || !tokenOk}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
          Start export
        </Button>
      </div>
    </div>
  );
}

// Card section matching serverless/new's "Run on" / "Pod" cards.
function Section({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-lg border border-border bg-card p-5">
      <div className="mb-4">
        <h2 className="text-sm font-semibold">{title}</h2>
        {description && <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>}
      </div>
      {children}
    </section>
  );
}

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
        <span className="truncate" title={state.message}>{state.message}</span>
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
        <span className="truncate" title={data.message}>{data.message}</span>
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
    <div className={cn("flex items-center justify-between gap-2 rounded-md border px-2.5 py-1.5 text-xs",
      allFree ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
        : "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400")}>
      <span>
        {data.gpus.length} GPU{data.gpus.length === 1 ? "" : "s"} · {fmtMib(totalFreeMib)} free / {fmtMib(totalMib)}
        {!allFree && ` · ${busy} busy`}
      </span>
      <button type="button" onClick={onRefresh} className="inline-flex items-center gap-1 underline-offset-2 hover:underline">
        <RefreshCw className="h-3 w-3" /> Refresh
      </button>
    </div>
  );
}

function fmtMib(mib: number): string {
  if (mib >= 1024) return `${(mib / 1024).toFixed(1)} GiB`;
  return `${mib} MiB`;
}
