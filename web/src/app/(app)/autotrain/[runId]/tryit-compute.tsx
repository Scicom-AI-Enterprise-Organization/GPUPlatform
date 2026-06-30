"use client";

import { useEffect, useState } from "react";
import { Cpu, Server, Wallet } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SearchableSelect } from "@/components/ui/searchable-select";
import { AvailabilityBadge } from "@/components/availability-badge";
import { VmAvailabilityRow, type VmAvailState } from "@/components/vm-availability-row";
import { useGpuAvailability } from "@/lib/use-gpu-availability";
import { gateway } from "@/lib/gateway";
import { cn } from "@/lib/utils";
import { GPU_CHOICES, GPU_COUNT_CHOICES, capacityHint } from "@/lib/gpu-catalog";
import type { ProviderRecord, ProviderBalance, TryItTarget } from "@/lib/types";

// The full "Run on" choice: a target spec (cloud pod / VM provider) plus, for the
// VM target, which device to run on (a GPU index, "auto", or "cpu"). `gpu` is
// meaningless for the cloud target (the fresh pod has its own single device). In
// LLM mode `gpu` holds a comma-separated GPU list (tensor-parallel, e.g. "6,7") and
// `vllmArgs` is appended to `vllm serve` verbatim.
export type ComputeChoice = TryItTarget & { gpu: string; vllmArgs?: string };

// Radix <Select> forbids an empty value, so use sentinels.
const AUTO = "auto";

/** Initial compute choice for a run — defaults to where it trained: a cloud-trained
 * run → a fresh pod with the same GPU; a VM-trained run → that VM. LLM is VM-only and
 * serves via vLLM, so its `gpu` defaults to the run's full GPU list (tensor-parallel). */
export function defaultCompute(opts: {
  trainedOnVm: boolean;
  runProviderId?: string | null;
  gpuChoice?: string | null; // run's training GPU mapped to a catalog value
  gpuCount?: number | null;
  pins: string[]; // run's GPU pins (visible_devices)
  llm?: boolean;
}): ComputeChoice {
  if (opts.llm) {
    // VM-only vLLM serve: `gpu` is a comma-separated tensor-parallel list (the run's
    // training GPUs); empty → the backend falls back to the run's visible_devices.
    return {
      target: "vm",
      provider_id: opts.runProviderId ?? null,
      gpu: opts.pins.join(","),
      vllmArgs: "",
    };
  }
  if (opts.trainedOnVm) {
    return {
      target: "vm",
      provider_id: opts.runProviderId ?? null,
      gpu: opts.pins[0] ?? AUTO,
    };
  }
  return {
    target: "cloud",
    provider_id: opts.runProviderId ?? null,
    gpu_type: opts.gpuChoice || "L40S",
    gpu_count: opts.gpuCount && opts.gpuCount > 0 ? opts.gpuCount : 1,
    cloud_type: "SECURE",
    gpu: AUTO,
  };
}

function RunOnCard({
  active,
  onClick,
  disabled,
  icon,
  title,
  subtitle,
}: {
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  icon: React.ReactNode;
  title: string;
  subtitle: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "flex items-start gap-3 rounded-md border px-3 py-2.5 text-left text-sm transition-colors disabled:opacity-50",
        active ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40",
      )}
    >
      <span className="mt-0.5 shrink-0 text-muted-foreground">{icon}</span>
      <span className="min-w-0">
        <span className="block font-medium">{title}</span>
        <span className="block text-xs text-muted-foreground">{subtitle}</span>
      </span>
    </button>
  );
}

/**
 * "Run on" + "Pod" cards for the Try-it playground — mirrors the Export-to-Label
 * tab (`autotrain/[runId]/label-export-tab.tsx`), which in turn mirrors the
 * serverless deploy form. Lets you run inference on a fresh RunPod pod (chosen GPU
 * / count / tier / account) or any registered VM provider, decoupled from where the
 * run trained. Controlled via `value` / `onChange`. (No disk/volume/venv knobs —
 * the try-it pod auto-sizes from the run's config; those aren't user-tunable here.)
 */
export function TryItCompute({
  value,
  onChange,
  disabled,
  runProviderId,
  visibleDevices,
  llm,
}: {
  value: ComputeChoice;
  onChange: (c: ComputeChoice) => void;
  disabled?: boolean;
  runProviderId?: string | null;
  visibleDevices?: string | null;
  // LLM (gemma-4) serves via vLLM on a VM only: cloud is disabled, and the VM Pod
  // card swaps the single-device picker for a tensor-parallel GPU list + vLLM args.
  llm?: boolean;
}) {
  const [providers, setProviders] = useState<ProviderRecord[]>([]);
  const [balance, setBalance] = useState<ProviderBalance | null>(null);
  const [balanceLoading, setBalanceLoading] = useState(false);
  const [vmAvail, setVmAvail] = useState<VmAvailState>({ status: "idle" });

  useEffect(() => {
    gateway.listProviders().then(setProviders).catch(() => {});
  }, []);

  const runpods = providers.filter((p) => p.kind === "runpod");
  const vms = providers.filter((p) => p.kind === "vm");
  const isCloud = value.target === "cloud";

  // RunPod credit for the chosen account (cloud only).
  useEffect(() => {
    if (!isCloud || !value.provider_id) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setBalance(null);
      return;
    }
    let cancelled = false;
    setBalanceLoading(true);
    setBalance(null);
    gateway
      .getProviderBalance(value.provider_id)
      .then((b) => { if (!cancelled) setBalance(b); })
      .catch(() => { if (!cancelled) setBalance(null); })
      .finally(() => { if (!cancelled) setBalanceLoading(false); });
    return () => { cancelled = true; };
  }, [isCloud, value.provider_id]);

  async function refreshVm(id: string) {
    if (!id) { setVmAvail({ status: "idle" }); return; }
    setVmAvail({ status: "loading" });
    try {
      setVmAvail({ status: "ok", data: await gateway.getVmAvailability(id) });
    } catch (e) {
      setVmAvail({ status: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }
  // Probe the VM on selection / switch to the VM target.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (value.target === "vm" && value.provider_id) refreshVm(value.provider_id);
    else setVmAvail({ status: "idle" });
  }, [value.target, value.provider_id]);

  const availability = useGpuAvailability(
    value.gpu_type || "", value.gpu_count || 1, isCloud && !!value.provider_id, value.cloud_type,
  );

  // GPU device options for the VM target. We only know the physical indices when
  // the chosen VM is the run's own box (the run's pins); otherwise just auto/cpu.
  const pins = (visibleDevices ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  const showPins = value.provider_id && value.provider_id === runProviderId ? pins : [];

  function pickCloud() {
    onChange({
      target: "cloud",
      provider_id: runpods.some((p) => p.id === value.provider_id) ? value.provider_id : (runpods[0]?.id ?? null),
      gpu_type: value.gpu_type || "L40S",
      gpu_count: value.gpu_count || 1,
      cloud_type: value.cloud_type || "SECURE",
      gpu: AUTO,
    });
  }
  function pickVm() {
    const provider_id = vms.some((p) => p.id === value.provider_id)
      ? value.provider_id
      : (vms.find((p) => p.id === runProviderId)?.id ?? vms[0]?.id ?? null);
    // LLM stays VM-only — preserve the GPU list + vLLM args; non-LLM resets the device.
    if (llm) { onChange({ target: "vm", provider_id, gpu: value.gpu, vllmArgs: value.vllmArgs }); return; }
    onChange({ target: "vm", provider_id, gpu: AUTO });
  }

  return (
    <div className="space-y-5">
      {/* Run on — same card as the Export-to-Label tab / serverless/new */}
      <Section
        title="Run on"
        description={llm
          ? "LLM serves via vLLM on a registered VM. Cloud pods aren't supported for chat try-it."
          : "Default cloud spawns a fresh RunPod pod for inference, then auto-stops when idle. Bare metal runs on a VM you've registered under GPU Providers."}
      >
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <RunOnCard
            active={isCloud}
            onClick={pickCloud}
            disabled={disabled || llm}
            icon={<Cpu className="h-4 w-4" />}
            title="Default cloud (RunPod)"
            subtitle={llm
              ? "Not available for LLM — vLLM runs on a VM."
              : "Provision a fresh pod on demand, then auto-stop when idle. Pay-per-second."}
          />
          <RunOnCard
            active={!isCloud}
            onClick={pickVm}
            disabled={disabled}
            icon={<Server className="h-4 w-4" />}
            title="Bare metal (VM)"
            subtitle="Run on a registered VM. No spin-up cost; the model stays resident."
          />
        </div>
      </Section>

      {/* Pod — provider + hardware (same card as the Export-to-Label tab) */}
      <Section
        title="Pod"
        description={isCloud
          ? "GPU, count, and cloud tier for the try-it pod."
          : "Which registered VM the model loads on. Hardware is fixed by the VM."}
      >
        <div className="space-y-3">
          {isCloud ? (
            runpods.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                No RunPod providers registered.{" "}
                <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">Add one</a>{" "}
                under GPU Providers.
              </p>
            ) : (
              <div className="space-y-3">
                <div className="space-y-1.5">
                  <Label className="text-xs">RunPod account</Label>
                  <Select
                    value={value.provider_id ?? ""}
                    onValueChange={(v) => onChange({ ...value, provider_id: v })}
                    disabled={disabled}
                  >
                    <SelectTrigger className="text-xs"><SelectValue placeholder="Choose a RunPod account…" /></SelectTrigger>
                    <SelectContent>
                      {runpods.map((p) => (
                        <SelectItem key={p.id} value={p.id} className="text-xs">
                          {p.name}{p.api_key_last4 ? ` · ****${p.api_key_last4}` : ""}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {value.provider_id && (
                    <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                      <Wallet className="h-3 w-3" />
                      {balanceLoading
                        ? "Checking credit…"
                        : balance?.ok && typeof balance.balance === "number"
                          ? <>Credit: <span className="font-medium text-emerald-600">${balance.balance.toFixed(2)}</span></>
                          : "Credit unavailable"}
                    </span>
                  )}
                </div>

                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs">GPU</Label>
                    <AvailabilityBadge state={availability} count={value.gpu_count || 1} />
                  </div>
                  <div className="flex gap-2">
                    {/* SearchableSelect has no `disabled` prop — gate it with a wrapper. */}
                    <div className={cn("flex-1", disabled && "pointer-events-none opacity-50")}>
                      <SearchableSelect
                        value={value.gpu_type || ""}
                        onChange={(v) => onChange({ ...value, gpu_type: v })}
                        options={GPU_CHOICES.map((g) => ({ value: g.value, label: g.label, group: g.group, hint: capacityHint(g.vramGb, 1) }))}
                        placeholder="Choose a GPU"
                        searchPlaceholder="Search GPUs (e.g. h100, 24gb, ada)…"
                      />
                    </div>
                    <Select
                      value={String(value.gpu_count || 1)}
                      onValueChange={(v) => onChange({ ...value, gpu_count: Number.parseInt(v, 10) })}
                      disabled={disabled}
                    >
                      <SelectTrigger className="w-20 shrink-0"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {GPU_COUNT_CHOICES.map((n) => <SelectItem key={n} value={String(n)}>×{n}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  {(() => {
                    const g = GPU_CHOICES.find((c) => c.value === value.gpu_type);
                    return g ? <p className="text-[11px] text-muted-foreground">{capacityHint(g.vramGb, value.gpu_count || 1)}</p> : null;
                  })()}
                </div>

                <div className="space-y-1.5">
                  <Label className="text-xs">Cloud tier</Label>
                  <Select
                    value={value.cloud_type || "SECURE"}
                    onValueChange={(v) => onChange({ ...value, cloud_type: v as "SECURE" | "COMMUNITY" })}
                    disabled={disabled}
                  >
                    <SelectTrigger className="text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="SECURE">Secure</SelectItem>
                      <SelectItem value="COMMUNITY">Community</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            )
          ) : (
            vms.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                No VM providers registered. Add one at{" "}
                <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">GPU Providers → New provider</a>.
              </p>
            ) : (
              <div className="space-y-3">
                <div className="space-y-1.5">
                  <Label className="text-xs">VM provider</Label>
                  <Select
                    value={value.provider_id ?? ""}
                    onValueChange={(v) => onChange(llm ? { ...value, provider_id: v } : { ...value, provider_id: v, gpu: AUTO })}
                    disabled={disabled}
                  >
                    <SelectTrigger className="text-xs"><SelectValue placeholder="Pick a VM…" /></SelectTrigger>
                    <SelectContent>
                      {vms.map((p) => (
                        <SelectItem key={p.id} value={p.id} className="text-xs">
                          {p.name}
                          {p.gpu_count != null && p.gpu_count > 0 ? ` · ${p.gpu_count} GPU` : ""}
                          {p.host ? ` · ${p.host}` : ""}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {value.provider_id && <VmAvailabilityRow state={vmAvail} onRefresh={() => refreshVm(value.provider_id!)} />}
                </div>

                {llm ? (
                  <>
                    <div className="space-y-1.5">
                      <Label className="text-xs">GPUs</Label>
                      <Input
                        value={value.gpu === AUTO ? "" : value.gpu}
                        onChange={(e) => onChange({ ...value, gpu: e.target.value })}
                        disabled={disabled}
                        placeholder="6,7"
                        className="w-32 font-mono text-xs"
                      />
                      <p className="text-[11px] text-muted-foreground">
                        Comma-separated GPU indices the vLLM server runs on (tensor-parallel = count). Defaults to the run&apos;s training GPUs.
                      </p>
                    </div>
                    <div className="space-y-1.5">
                      <Label className="text-xs">Custom vLLM args (optional)</Label>
                      <Input
                        value={value.vllmArgs ?? ""}
                        onChange={(e) => onChange({ ...value, vllmArgs: e.target.value })}
                        disabled={disabled}
                        placeholder="--enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768"
                        className="font-mono text-[11px]"
                      />
                      <p className="text-[11px] text-muted-foreground">
                        Appended to <span className="font-mono">vllm serve</span> verbatim. Platform-set flags (model / port / served-name / tensor-parallel) are rejected.
                      </p>
                    </div>
                  </>
                ) : (
                  <div className="space-y-1.5">
                    <Label className="text-xs">Run on device</Label>
                    <Select value={value.gpu} onValueChange={(v) => onChange({ ...value, gpu: v })} disabled={disabled}>
                      <SelectTrigger className="w-[200px] text-xs"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {showPins.map((g) => <SelectItem key={g} value={g} className="text-xs">GPU {g}</SelectItem>)}
                        <SelectItem value={AUTO} className="text-xs">Auto (most-free GPU)</SelectItem>
                        <SelectItem value="cpu" className="text-xs">CPU</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}
              </div>
            )
          )}
        </div>
      </Section>
    </div>
  );
}

// Card section matching the Export-to-Label tab's "Run on" / "Pod" cards.
function Section({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
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
