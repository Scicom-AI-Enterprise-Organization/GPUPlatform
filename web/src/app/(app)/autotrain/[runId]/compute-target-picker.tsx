"use client";

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { AlertTriangle, Cpu, Server } from "lucide-react";
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
import { VmAvailabilityRow, type VmAvailState } from "@/components/vm-availability-row";
import { RegionSelect } from "@/components/region-select";
import { useGpuAvailability } from "@/lib/use-gpu-availability";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { GpuTypeOption, ProviderRecord, TrainingRunRecord } from "@/lib/types";

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

// The compute-target value: where a background job runs (a fresh RunPod pod, or a
// registered bare-metal VM) and the pod hardware spec. `runOn` toggles cloud vs vm.
export type ComputeTarget = {
  runOn: "vm" | "cloud";
  vmProviderId: string;
  runpodProviderId: string;
  gpuType: string;
  gpuCount: number;
  secureCloud: boolean;
  dataCenterId: string;
  diskGb: number;
  volumeGb: number;
  visibleDevices: string;
  venvPath: string;
};

// Sensible defaults for a fresh compute target, prefilled from the run where it makes
// sense (its provider / gpu type). `str`/`num` read config_json overrides (label_* keys)
// so a retried export inherits the previous run-on choice.
export function defaultComputeTarget(run: TrainingRunRecord): ComputeTarget {
  const lcfg = (run.config_json ?? {}) as Record<string, unknown>;
  const str = (k: string, d = ""): string => (typeof lcfg[k] === "string" ? (lcfg[k] as string) : d);
  const num = (k: string, d: number): number => (typeof lcfg[k] === "number" ? (lcfg[k] as number) : d);
  const runOn: "vm" | "cloud" = str("label_run_on") === "cloud" ? "cloud" : "vm";
  return {
    runOn,
    vmProviderId: runOn === "vm" ? str("label_provider_id") || run.provider_id || "" : run.provider_id || "",
    runpodProviderId: runOn === "cloud" ? str("label_provider_id") : "",
    gpuType: str("label_gpu_type") || run.gpu_type || "NVIDIA L40S",
    gpuCount: num("label_gpu_count", 1),
    secureCloud: typeof lcfg.label_secure_cloud === "boolean" ? (lcfg.label_secure_cloud as boolean) : true,
    dataCenterId: str("label_data_center_id"),
    diskGb: num("label_disk_gb", 60),
    volumeGb: num("label_volume_gb", 80),
    visibleDevices: str("label_visible_devices"),
    venvPath: str("venv_path"),
  };
}

// True when the CUDA_VISIBLE_DEVICES string is malformed given the GPU count bound.
// Returns the error message, or null when valid/empty. Exported so callers can gate
// submit on it (mirrors the inline check the picker previously carried).
export function computeVisibleDevicesError(visibleDevices: string, gpuBound: number): string | null {
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

// Shared "Run on" + "Pod" cards: pick a fresh RunPod pod (GPU type/count/tier/region,
// disk/volume) or a registered bare-metal VM, plus a uv-venv path and an optional
// CUDA_VISIBLE_DEVICES override. Controlled — the compute fields live in the parent.
// Task-specific bits (e.g. TTS NeuCodec decoder) stay in the consuming tab.
export function ComputeTargetPicker({
  run,
  value,
  onChange,
  venvPlaceholder,
  venvLabel,
  showVenv = true,
  vramHint,
}: {
  run: TrainingRunRecord;
  value: ComputeTarget;
  onChange: (next: ComputeTarget) => void;
  /** Placeholder for the uv venv path input (e.g. "/share/autotrain-tts"). */
  venvPlaceholder?: string;
  /** Label for the uv venv path field. Defaults to "uv venv path". */
  venvLabel?: string;
  /** Hide the uv venv path field entirely. */
  showVenv?: boolean;
  /** Amber note under the GPU picker (VRAM guidance for this task). */
  vramHint?: string;
}) {
  const set = <K extends keyof ComputeTarget>(k: K, v: ComputeTarget[K]) => onChange({ ...value, [k]: v });

  const [providers, setProviders] = useState<ProviderRecord[]>([]);
  const [gpuOptions, setGpuOptions] = useState<GpuTypeOption[]>(RUNPOD_GPU_FALLBACK);

  const target = value.runOn;
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
  // GPU bound for CUDA_VISIBLE_DEVICES validation: on cloud it's the picked count; on
  // VM the selected provider's count, falling back to the run's own GPU count (e.g.
  // when the picked VM is the run's own and its provider row hasn't loaded yet).
  const gpuBound = useMemo(
    () =>
      target === "vm"
        ? vmProviders.find((p) => p.id === value.vmProviderId)?.gpu_count
            ?? (value.vmProviderId && value.vmProviderId === run.provider_id ? run.gpu_count ?? 0 : 0)
        : value.gpuCount,
    [target, vmProviders, value.vmProviderId, value.gpuCount, run.provider_id, run.gpu_count],
  );
  const availability = useGpuAvailability(value.gpuType, value.gpuCount, target === "cloud", value.secureCloud ? "SECURE" : "COMMUNITY");

  const vdError = useMemo(
    () => computeVisibleDevicesError(value.visibleDevices, gpuBound),
    [value.visibleDevices, gpuBound],
  );

  useEffect(() => {
    gateway
      .listProviders()
      .then((ps) => {
        setProviders(ps);
        // Auto-select the first registered RunPod account — no gateway-default fallback.
        const firstRunpod = ps.find((p) => p.kind === "runpod");
        if (firstRunpod && !value.runpodProviderId) onChange({ ...value, runpodProviderId: firstRunpod.id });
      })
      .catch(() => {});
    gateway
      .listRunpodGpuTypes()
      .then((rows) => {
        if (!rows.length) return;
        setGpuOptions(rows);
        if (!rows.some((g) => g.id === value.gpuType)) onChange({ ...value, gpuType: rows[0].id });
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (target === "vm" && value.vmProviderId) refreshVmAvail(value.vmProviderId);
    else setVmAvail({ status: "idle" });
  }, [target, value.vmProviderId, refreshVmAvail]);

  return (
    <>
      {/* Run on — same card as serverless/new */}
      <Section
        title="Run on"
        description="Default cloud spawns a fresh RunPod pod, then tears it down. Bare metal runs on a VM you've registered under GPU Providers."
      >
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <button type="button" onClick={() => set("runOn", "cloud")}
            className={cn("flex items-start gap-3 rounded-md border px-3 py-2.5 text-left text-sm transition-colors",
              target === "cloud" ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40")}>
            <Cpu className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <div className="font-medium">Default cloud (RunPod)</div>
              <div className="text-xs text-muted-foreground">Provision a fresh pod on demand, run, tear down. Pay-per-second.</div>
            </div>
          </button>
          <button type="button" onClick={() => set("runOn", "vm")}
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
          ? "GPU, count, and cloud tier for the pod."
          : "Which registered VM the job runs on. Hardware is fixed by the VM."}
      >
        <div className="space-y-5">
          {target === "cloud" ? (
            <Field label="RunPod account" hint="Which registered RunPod provider to run on.">
              {runpodProviders.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No RunPod providers registered.{" "}
                  <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">Add one</a>{" "}
                  under GPU Providers.
                </p>
              ) : (
                <Select value={value.runpodProviderId} onValueChange={(v) => set("runpodProviderId", v)}>
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
            </Field>
          ) : (
            <Field label="VM provider" hint="The registered VM the job SSHes onto. Hardware is fixed by the VM.">
              {vmProviders.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No VM providers registered. Add one at{" "}
                  <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">GPU Providers → New provider</a>.
                </p>
              ) : (
                <Select value={value.vmProviderId} onValueChange={(v) => set("vmProviderId", v)}>
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
              {value.vmProviderId && (
                <div className="mt-1.5">
                  <VmAvailabilityRow state={vmAvail} onRefresh={() => refreshVmAvail(value.vmProviderId)} />
                </div>
              )}
            </Field>
          )}

          {target === "cloud" && (
            <>
              <Field label="Cloud tier" hint="Community is cheaper with variable hosts; Secure uses vetted hosts with more capacity.">
                <div className="grid grid-cols-2 gap-2">
                  {([true, false] as const).map((secure) => (
                    <button
                      key={String(secure)}
                      type="button"
                      onClick={() => set("secureCloud", secure)}
                      className={cn(
                        "rounded-md border p-3 text-left transition-colors",
                        value.secureCloud === secure
                          ? "border-foreground/60 ring-1 ring-foreground/20"
                          : "border-border hover:border-foreground/40",
                      )}
                    >
                      <div className="text-sm font-medium">{secure ? "Secure" : "Community"}</div>
                      <div className="mt-0.5 text-xs text-muted-foreground">
                        {secure ? "vetted hosts, more capacity" : "cheaper, variable hosts"}
                      </div>
                    </button>
                  ))}
                </div>
              </Field>

              <Field label="Region" hint="Pin the pod to a RunPod data center, or Auto to let RunPod pick any region with capacity.">
                <RegionSelect value={value.dataCenterId} onChange={(v) => set("dataCenterId", v)} className="text-sm" />
              </Field>

              <Field
                label="GPU"
                hint={(() => {
                  const g = gpuOptions.find((o) => o.id === value.gpuType);
                  return g ? gpuHint(g.vram_gb, value.gpuCount) : undefined;
                })()}
                extra={<AvailabilityBadge state={availability} count={value.gpuCount} />}
              >
                <div className="flex gap-2">
                  <SearchableSelect
                    className="flex-1"
                    value={value.gpuType}
                    onChange={(v) => set("gpuType", v)}
                    options={gpuOptions.map((g) => ({ value: g.id, label: g.label, hint: gpuHint(g.vram_gb, 1) }))}
                    placeholder="Choose a GPU"
                    searchPlaceholder="Search GPUs (e.g. h100, 24gb)…"
                  />
                  <Select value={String(value.gpuCount)} onValueChange={(v) => set("gpuCount", Number.parseInt(v, 10))}>
                    <SelectTrigger className="w-24 shrink-0"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {GPU_COUNT_CHOICES.map((n) => <SelectItem key={n} value={String(n)}>×{n}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
              </Field>

              <div className="grid grid-cols-2 gap-3">
                <Field label="Container disk (GB)" hint="Ephemeral workspace. Resets when the pod stops.">
                  <NumberField min={20} value={value.diskGb} onChange={(v) => set("diskGb", v)} />
                </Field>
                <Field label="Volume (GB)" hint="Persistent volume for the model cache. 0 = no persistent storage.">
                  <NumberField min={0} value={value.volumeGb} onChange={(v) => set("volumeGb", v)} />
                </Field>
              </div>

              {vramHint && (
                <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>{vramHint}</span>
                </div>
              )}
            </>
          )}

          <div className={cn("grid grid-cols-1 gap-3", showVenv && "sm:grid-cols-2")}>
            {showVenv && (
              <Field label={venvLabel ?? "uv venv path"}>
                <Input className="font-mono text-xs" placeholder={venvPlaceholder}
                  value={value.venvPath} onChange={(e) => set("venvPath", e.target.value)} />
              </Field>
            )}
            <Field
              label="CUDA_VISIBLE_DEVICES (optional)"
              hint={!vdError && gpuBound > 0
                ? `${target === "vm" ? "This VM" : "The pod"} has ${gpuBound} GPU${gpuBound === 1 ? "" : "s"} — valid indices 0–${gpuBound - 1}.`
                : undefined}
            >
              <Input className={cn("font-mono text-xs", vdError && "border-destructive focus-visible:ring-destructive")}
                placeholder="e.g. 0 (empty = all GPUs)"
                value={value.visibleDevices} onChange={(e) => set("visibleDevices", e.target.value)} />
              {vdError && <p className="text-[11px] text-destructive">{vdError}</p>}
            </Field>
          </div>
        </div>
      </Section>
    </>
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

// Labelled field matching serverless/new's Field (uppercase label + hint + optional
// right-aligned `extra`).
function Field({
  label,
  hint,
  children,
  extra,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
  extra?: ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <Label className="text-xs uppercase tracking-wide text-muted-foreground">{label}</Label>
        {extra}
      </div>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}
