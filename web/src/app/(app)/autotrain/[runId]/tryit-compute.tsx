"use client";

import { useEffect, useState } from "react";
import { Cpu, Loader2, RefreshCw, Server, Wallet } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SearchableSelect } from "@/components/ui/searchable-select";
import { AvailabilityBadge } from "@/components/availability-badge";
import { useGpuAvailability } from "@/lib/use-gpu-availability";
import { gateway } from "@/lib/gateway";
import { cn } from "@/lib/utils";
import { GPU_CHOICES, GPU_COUNT_CHOICES, capacityHint } from "@/lib/gpu-catalog";
import type { ProviderRecord, ProviderBalance, TryItTarget, VmAvailability } from "@/lib/types";

// The full "Run on" choice: a target spec (cloud pod / VM provider) plus, for the
// VM target, which device to run on (a GPU index, "auto", or "cpu"). `gpu` is
// meaningless for the cloud target (the fresh pod has its own single device).
export type ComputeChoice = TryItTarget & { gpu: string };

// Radix <Select> forbids an empty value, so use sentinels.
const AUTO = "auto";

/** Initial compute choice for a run — defaults to where it trained: a cloud-trained
 * run → a fresh pod with the same GPU; a VM-trained run → that VM. */
export function defaultCompute(opts: {
  trainedOnVm: boolean;
  runProviderId?: string | null;
  gpuChoice?: string | null; // run's training GPU mapped to a catalog value
  gpuCount?: number | null;
  pins: string[]; // run's GPU pins (visible_devices)
}): ComputeChoice {
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
 * "Run on" + "Pod" cards for the Try-it playground — mirrors the serverless deploy
 * form (`serverless/new/inference-form.tsx`). Lets you run inference on a fresh
 * RunPod pod (chosen GPU / count / tier / account) or any registered VM provider,
 * decoupled from where the run trained. Controlled via `value` / `onChange`.
 */
export function TryItCompute({
  value,
  onChange,
  disabled,
  runProviderId,
  visibleDevices,
}: {
  value: ComputeChoice;
  onChange: (c: ComputeChoice) => void;
  disabled?: boolean;
  runProviderId?: string | null;
  visibleDevices?: string | null;
}) {
  const [providers, setProviders] = useState<ProviderRecord[]>([]);
  const [balance, setBalance] = useState<ProviderBalance | null>(null);
  const [balanceLoading, setBalanceLoading] = useState(false);
  const [vmAvail, setVmAvail] = useState<
    { status: "idle" | "loading" } | { status: "ok"; data: VmAvailability } | { status: "error"; message: string }
  >({ status: "idle" });

  useEffect(() => {
    gateway.listProviders().then(setProviders).catch(() => {});
  }, []);

  const runpods = providers.filter((p) => p.kind === "runpod");
  const vms = providers.filter((p) => p.kind === "vm");
  const isCloud = value.target === "cloud";

  // RunPod credit for the chosen account (cloud only).
  useEffect(() => {
    if (!isCloud || !value.provider_id) {
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
    onChange({ target: "vm", provider_id, gpu: AUTO });
  }

  return (
    <div className="space-y-3 rounded-md border border-border bg-muted/10 p-3">
      <div className="text-xs font-medium text-muted-foreground">Run on</div>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        <RunOnCard
          active={isCloud}
          onClick={pickCloud}
          disabled={disabled}
          icon={<Cpu className="h-4 w-4" />}
          title="Cloud (RunPod)"
          subtitle="Spin up a temporary pod. Auto-stops when idle."
        />
        <RunOnCard
          active={!isCloud}
          onClick={pickVm}
          disabled={disabled}
          icon={<Server className="h-4 w-4" />}
          title="Bare metal (VM)"
          subtitle="Run on a registered VM. No spin-up cost."
        />
      </div>

      {isCloud ? (
        runpods.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No RunPod providers registered.{" "}
            <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">Add one</a>{" "}
            under GPU Providers.
          </p>
        ) : (
          <div className="space-y-2">
            <div className="flex flex-col gap-1">
              <span className="text-[11px] uppercase tracking-wide text-muted-foreground">RunPod account</span>
              <Select
                value={value.provider_id ?? ""}
                onValueChange={(v) => onChange({ ...value, provider_id: v })}
                disabled={disabled}
              >
                <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="Choose a RunPod account…" /></SelectTrigger>
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

            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[11px] uppercase tracking-wide text-muted-foreground">GPU</span>
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
                  <SelectTrigger className="h-8 w-20 shrink-0 text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {GPU_COUNT_CHOICES.map((n) => <SelectItem key={n} value={String(n)} className="text-xs">×{n}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              {(() => {
                const g = GPU_CHOICES.find((c) => c.value === value.gpu_type);
                return g ? <span className="text-[11px] text-muted-foreground">{capacityHint(g.vramGb, value.gpu_count || 1)}</span> : null;
              })()}
            </div>

            <div className="flex flex-col gap-1">
              <span className="text-[11px] uppercase tracking-wide text-muted-foreground">Cloud tier</span>
              <div className="grid grid-cols-2 gap-2">
                {(["SECURE", "COMMUNITY"] as const).map((tier) => (
                  <button
                    key={tier}
                    type="button"
                    disabled={disabled}
                    onClick={() => onChange({ ...value, cloud_type: tier })}
                    className={cn(
                      "rounded-md border px-2.5 py-1.5 text-left text-xs transition-colors disabled:opacity-50",
                      (value.cloud_type || "SECURE") === tier ? "border-foreground/60 ring-1 ring-foreground/20" : "border-border hover:border-foreground/40",
                    )}
                  >
                    <div className="font-medium">{tier === "COMMUNITY" ? "Community" : "Secure"}</div>
                    <div className="text-[11px] text-muted-foreground">{tier === "COMMUNITY" ? "cheaper, variable" : "vetted, more capacity"}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )
      ) : (
        vms.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No VM providers registered.{" "}
            <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">Add one</a>{" "}
            under GPU Providers.
          </p>
        ) : (
          <div className="space-y-2">
            <div className="flex flex-col gap-1">
              <span className="text-[11px] uppercase tracking-wide text-muted-foreground">VM provider</span>
              <Select
                value={value.provider_id ?? ""}
                onValueChange={(v) => onChange({ ...value, provider_id: v, gpu: AUTO })}
                disabled={disabled}
              >
                <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="Pick a VM…" /></SelectTrigger>
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
              {value.provider_id && (
                <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                  {vmAvail.status === "loading" && <><Loader2 className="h-3 w-3 animate-spin" /> probing VM…</>}
                  {vmAvail.status === "ok" && (
                    <>
                      {vmAvail.data.ok ? "✓" : "✗"} {vmAvail.data.message || `${vmAvail.data.gpus.length} GPU`}
                      <button type="button" onClick={() => refreshVm(value.provider_id!)} className="ml-1 hover:text-foreground" title="Re-probe">
                        <RefreshCw className="h-3 w-3" />
                      </button>
                    </>
                  )}
                  {vmAvail.status === "error" && <span className="text-destructive">{vmAvail.message}</span>}
                </span>
              )}
            </div>

            <div className="flex flex-col gap-1">
              <span className="text-[11px] uppercase tracking-wide text-muted-foreground">Run on device</span>
              <Select value={value.gpu} onValueChange={(v) => onChange({ ...value, gpu: v })} disabled={disabled}>
                <SelectTrigger className="h-8 w-[180px] text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {showPins.map((g) => <SelectItem key={g} value={g} className="text-xs">GPU {g}</SelectItem>)}
                  <SelectItem value={AUTO} className="text-xs">Auto (most-free GPU)</SelectItem>
                  <SelectItem value="cpu" className="text-xs">CPU</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        )
      )}
    </div>
  );
}
