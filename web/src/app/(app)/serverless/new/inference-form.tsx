"use client";

import { useCallback, useEffect, useId, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { AlertCircle, AlertTriangle, Check, ChevronDown, ChevronRight, Cpu, Loader2, RefreshCw, Server, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SearchableSelect } from "@/components/ui/searchable-select";
import { cn } from "@/lib/utils";
import { parseGpuIds, suggestPacking } from "@/lib/gpu-pin";
import { cleanVllmArgs } from "@/lib/vllm-args";
import { deployEndpoint } from "../actions";
import { AvailabilityBadge } from "@/components/availability-badge";
import { useGpuAvailability } from "@/lib/use-gpu-availability";
import { gateway } from "@/lib/gateway";
import type { ProviderRecord, VmAvailability } from "@/lib/types";

// vLLM is what the live RunPod template runs. SGLang is a placeholder for a
// future template — keep it disabled so the option is visible but inert.
const FRAMEWORKS = [
  { value: "vllm", label: "vLLM", available: true },
  { value: "sglang", label: "SGLang (coming soon)", available: false },
] as const;

// Curated RunPod GPU catalog — values match runpod_provider._GPU_NAME_MAP.
// vramGb feeds the dynamic capacity hint (recomputed with gpuCount).
// Catalog last reviewed: 2026-05.
type GpuChoice = { value: string; label: string; group: string; vramGb: number };
const GPU_CHOICES: GpuChoice[] = [
  // Datacenter — current gen
  { value: "B200", label: "B200 (180 GB)", group: "Datacenter — current", vramGb: 180 },
  { value: "H200", label: "H200 (141 GB)", group: "Datacenter — current", vramGb: 141 },
  { value: "H100", label: "H100 SXM (80 GB)", group: "Datacenter — current", vramGb: 80 },
  { value: "H100-PCIe", label: "H100 PCIe (80 GB)", group: "Datacenter — current", vramGb: 80 },
  { value: "H100-NVL", label: "H100 NVL (94 GB)", group: "Datacenter — current", vramGb: 94 },
  { value: "MI300X", label: "MI300X (192 GB)", group: "Datacenter — current", vramGb: 192 },
  // Datacenter — Ampere
  { value: "A100", label: "A100 PCIe (80 GB)", group: "Datacenter — Ampere", vramGb: 80 },
  { value: "A100-SXM", label: "A100 SXM (80 GB)", group: "Datacenter — Ampere", vramGb: 80 },
  { value: "A100-40G", label: "A100 (40 GB)", group: "Datacenter — Ampere", vramGb: 40 },
  { value: "A40", label: "A40 (48 GB)", group: "Datacenter — Ampere", vramGb: 48 },
  { value: "A10", label: "A10 (24 GB)", group: "Datacenter — Ampere", vramGb: 24 },
  // Datacenter — Ada
  { value: "L40S", label: "L40S (48 GB)", group: "Datacenter — Ada", vramGb: 48 },
  { value: "L40", label: "L40 (48 GB)", group: "Datacenter — Ada", vramGb: 48 },
  { value: "L4", label: "L4 (24 GB)", group: "Datacenter — Ada", vramGb: 24 },
  // Workstation
  { value: "RTX6000-Ada", label: "RTX 6000 Ada (48 GB)", group: "Workstation", vramGb: 48 },
  { value: "A6000", label: "RTX A6000 (48 GB)", group: "Workstation", vramGb: 48 },
  { value: "A5000", label: "RTX A5000 (24 GB)", group: "Workstation", vramGb: 24 },
  { value: "A4000", label: "RTX A4000 (16 GB)", group: "Workstation", vramGb: 16 },
  // Consumer
  { value: "RTX5090", label: "RTX 5090 (32 GB)", group: "Consumer", vramGb: 32 },
  { value: "RTX4090", label: "RTX 4090 (24 GB)", group: "Consumer", vramGb: 24 },
  { value: "RTX3090Ti", label: "RTX 3090 Ti (24 GB)", group: "Consumer", vramGb: 24 },
  { value: "RTX3090", label: "RTX 3090 (24 GB)", group: "Consumer", vramGb: 24 },
];

const GPU_COUNT_CHOICES = [1, 2, 4, 8] as const;

// Estimate the largest model that fits given VRAM budget. Rough formula
// calibrated against vLLM defaults at moderate context (~4-8k) and small
// batches; assumes ~45% of total VRAM is consumed by KV cache, activations,
// and framework overhead, leaving ~55% for weights.
//
//   weights_budget = total_vram * 0.55
//   FP16: 2 bytes/param  → max_B = budget / 2
//   4-bit (AWQ/GPTQ ~0.6 effective): max_B = budget / 0.6
//
// Real-world capacity drops fast at long context — these are upper bounds.
function capacityHint(vramPerGpu: number, count: number): string {
  const total = vramPerGpu * count;
  const weightsBudget = total * 0.55;
  const fp16B = weightsBudget / 2;
  const q4B = weightsBudget / 0.6;
  const fp16Str = fp16B >= 100 ? `${Math.round(fp16B / 10) * 10}B` : `${Math.round(fp16B)}B`;
  const q4Str = q4B >= 100 ? `${Math.round(q4B / 10) * 10}B` : `${Math.round(q4B)}B`;
  const totalStr = total >= 100 ? `${Math.round(total)} GB` : `${total} GB`;
  const tpHint =
    count === 1
      ? ""
      : ` · TP=${count} sharding`;
  return `${totalStr} VRAM${tpHint} · fits ~${fp16Str} FP16 / ~${q4Str} 4-bit (KV-cache budgeted)`;
}

const MAX_WORKERS = 1;

// Common vLLM engine args. Defaults are conservative — users can override.
// Reference: https://docs.vllm.ai/en/stable/configuration/engine_args/
const DEFAULT_VLLM_ARGS = {
  max_model_len: "",
  gpu_memory_utilization: "0.9",
  dtype: "auto",
  max_num_seqs: "",
  tensor_parallel_size: "1",
  extra: "",
};

const DTYPE_CHOICES = ["auto", "float16", "bfloat16", "float32"] as const;

function buildVllmArgs(v: typeof DEFAULT_VLLM_ARGS): string {
  const parts: string[] = [];
  if (v.max_model_len.trim()) parts.push(`--max-model-len ${v.max_model_len.trim()}`);
  if (v.gpu_memory_utilization.trim() && v.gpu_memory_utilization.trim() !== "0.9") {
    parts.push(`--gpu-memory-utilization ${v.gpu_memory_utilization.trim()}`);
  }
  if (v.dtype && v.dtype !== "auto") parts.push(`--dtype ${v.dtype}`);
  if (v.max_num_seqs.trim()) parts.push(`--max-num-seqs ${v.max_num_seqs.trim()}`);
  if (v.tensor_parallel_size.trim() && v.tensor_parallel_size.trim() !== "1") {
    parts.push(`--tensor-parallel-size ${v.tensor_parallel_size.trim()}`);
  }
  if (v.extra.trim()) parts.push(v.extra.trim());
  return parts.join(" ");
}

// Flags the platform sets on the `vllm serve` command itself — passing them in
// user args makes vLLM see a duplicate and refuse to start. Mirrors the
// gateway's create-time validation so the form catches it instantly.
const VLLM_RESERVED_SINGLE = ["--model", "--served-model-name", "--port"];
const VLLM_RESERVED_MULTI = [...VLLM_RESERVED_SINGLE, "--tensor-parallel-size", "-tp", "--pipeline-parallel-size", "-pp", "--enable-sleep-mode"];

/** Returns an error string for obviously-broken vLLM args (stray line-continuation
 * backslash, unbalanced quotes, platform-reserved flags), else null. */
function vllmArgsError(args: string, reserved: string[], label: string): string | null {
  const s = (args ?? "").trim();
  if (!s) return null;
  if (/(^|\s)\\(\s|$)/.test(s)) {
    return `${label}: stray "\\" — looks like a pasted shell line-continuation. Put all args on one line.`;
  }
  if (((s.match(/"/g) ?? []).length % 2 !== 0) || ((s.match(/'/g) ?? []).length % 2 !== 0)) {
    return `${label}: unbalanced quotes in vLLM args.`;
  }
  for (const tok of s.split(/\s+/)) {
    const flag = tok.split("=")[0];
    if (reserved.includes(flag)) return `${label}: remove "${flag}" — the platform sets it automatically.`;
  }
  return null;
}

export function InferenceForm() {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const [framework, setFramework] = useState("vllm");
  // Derive the default name from useId() (stable across SSR + client) rather
  // than Math.random() in the initializer, which runs twice with different
  // values → hydration mismatch.
  const reactId = useId();
  const [name, setName] = useState(() => suggestName(reactId));
  const [model, setModel] = useState("");
  const [gpu, setGpu] = useState("RTX3090");
  const [gpuCount, setGpuCount] = useState<number>(1);
  const [cloudType, setCloudType] = useState<"COMMUNITY" | "SECURE">("COMMUNITY");
  const [containerDisk, setContainerDisk] = useState<string>("50");
  const [volumeGb, setVolumeGb] = useState<string>("0");
  const [idleInput, setIdleInput] = useState("120"); // 2 min scale-to-zero (single-model only; multi VM fleets are always-on)
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [enableMetrics, setEnableMetrics] = useState(true);
  // Run-on target (mirrors the benchmark form): "cloud" spawns a fresh RunPod
  // pod, "vm" SSHes onto a registered VM. Each keeps its own provider pick.
  const [target, setTarget] = useState<"cloud" | "vm">("cloud");
  const [vmProviderId, setVmProviderId] = useState<string>("");
  const [runpodProviderId, setRunpodProviderId] = useState<string>("");
  const [providers, setProviders] = useState<ProviderRecord[]>([]);
  const [vllm, setVllm] = useState({ ...DEFAULT_VLLM_ARGS });
  const [members, setMembers] = useState<{ model: string; tp: number; pp: number; extra_args: string; gpus: string; audio: boolean }[]>([
    { model: "", tp: 1, pp: 1, extra_args: "", gpus: "", audio: false },
  ]);
  const [sleepLevel, setSleepLevel] = useState<1 | 2>(1);
  // VM-only: pin to specific physical GPU ids, e.g. "0,1,2,3". Empty = all GPUs.
  const [visibleDevices, setVisibleDevices] = useState("");
  // VM-only: uv venv the worker runs `vllm serve` from + the vLLM version to pin.
  // Default to the standard shared venv on the VM — leaving this empty makes the
  // worker fall back to bare `python3` (no vLLM), which silently never launches.
  const [venvPath, setVenvPath] = useState("/share/vllm-venv");
  const [vllmVersion, setVllmVersion] = useState("");
  // Endpoint-level env applied to every vLLM process (cache/home dirs, etc.).
  // Pasted as `KEY=value` / `export KEY=value` lines; `mkdir` lines are ignored
  // (the worker auto-creates absolute-path values).
  const [envText, setEnvText] = useState("");

  useEffect(() => {
    gateway.listProviders().then(setProviders).catch(() => {});
  }, []);

  // Live SSH probe of the selected VM — re-fires on provider change + refresh.
  type VmAvailState =
    | { status: "idle" }
    | { status: "loading" }
    | { status: "ok"; data: VmAvailability }
    | { status: "error"; message: string };
  const [vmAvail, setVmAvail] = useState<VmAvailState>({ status: "idle" });
  const refreshVmAvail = useCallback(async (id: string) => {
    if (!id) {
      setVmAvail({ status: "idle" });
      return;
    }
    setVmAvail({ status: "loading" });
    try {
      const data = await gateway.getVmAvailability(id);
      setVmAvail({ status: "ok", data });
    } catch (e) {
      setVmAvail({ status: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }, []);

  // Effective provider id sent to the gateway: the VM when target=vm, else the
  // chosen RunPod account ("" = gateway default). isVm now keys off the target.
  const providerId = target === "vm" ? vmProviderId : runpodProviderId;
  const isVm = target === "vm";
  // Serving mode follows the target: VM → multi-model fleet (single-model serving
  // + sleep/wake eviction); cloud (RunPod/PI) → single-model scale-to-zero. There
  // is no single-model mode on a VM.
  const mode: "single" | "multi" = isVm ? "multi" : "single";
  const selectedProvider = providers.find((p) => p.id === vmProviderId) || null;
  const vmGpuCount = selectedProvider?.gpu_count ?? 0;
  // Optional GPU pin. vdIds = chosen physical ids; vdInvalid flags bad input;
  // effectiveVmGpuCount (pin length, or all VM GPUs) drives TP choices + packing.
  const vdRaw = visibleDevices.trim();
  const vdIds = vdRaw
    ? vdRaw.split(",").map((s) => s.trim()).filter(Boolean).map(Number)
    : [];
  const vdInvalid =
    isVm &&
    vdRaw !== "" &&
    (vdIds.some((n) => !Number.isInteger(n) || n < 0 || (vmGpuCount > 0 && n >= vmGpuCount)) ||
      new Set(vdIds).size !== vdIds.length);
  const effectiveVmGpuCount = isVm && vdIds.length > 0 && !vdInvalid ? vdIds.length : vmGpuCount;
  // tp choices for multi: any (power-of-two) size up to the usable GPU count. We
  // do NOT require tp to divide the count — the packer + optional per-model GPU
  // pin handle non-dividing layouts, e.g. tp=4 on 7 GPUs → [0,1,2,3].
  const tpChoices = effectiveVmGpuCount > 0
    ? GPU_COUNT_CHOICES.filter((n) => n <= effectiveVmGpuCount)
    : [1, 2, 4, 8];
  // Pipeline-parallel choices: any integer 1..usable (PP need not be a power of
  // two — e.g. TP=2 × PP=3 = 6 GPUs). A member uses tp × pp GPUs total.
  const ppChoices = Array.from(
    { length: Math.max(1, effectiveVmGpuCount || 8) },
    (_, i) => i + 1,
  );
  // Physical GPU universe used to suggest per-model pins: an explicit
  // visible_devices pin, else 0..vmGpuCount-1. Suggestions mirror the gateway's
  // auto-packer so "what you see is what deploys".
  const physForSuggest =
    isVm && vdIds.length > 0 && !vdInvalid
      ? vdIds
      : isVm
        ? Array.from({ length: vmGpuCount }, (_, i) => i)
        : [];
  // Each member occupies tp × pp consecutive GPUs.
  const memberSuggestions = suggestPacking(members.map((m) => m.tp * (m.pp || 1)), physForSuggest);
  const [unavailableModal, setUnavailableModal] = useState<
    | { gpu: string; gpu_count: number; reason: string }
    | null
  >(null);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const gpuMemInvalid = (() => {
    const s = vllm.gpu_memory_utilization.trim();
    if (!s) return false;
    const n = Number.parseFloat(s);
    return !Number.isFinite(n) || n <= 0 || n > 1;
  })();
  const intFieldInvalid = (s: string) => {
    if (!s.trim()) return false;
    return !/^\d+$/.test(s.trim()) || Number.parseInt(s.trim(), 10) < 1;
  };
  const advancedInvalid =
    gpuMemInvalid ||
    intFieldInvalid(vllm.max_model_len) ||
    intFieldInvalid(vllm.max_num_seqs) ||
    intFieldInvalid(vllm.tensor_parallel_size);

  const parsedDisk = Number.parseInt(containerDisk, 10);
  const diskInvalid =
    !Number.isFinite(parsedDisk) || parsedDisk < 1 || parsedDisk > 2000;
  const parsedVolume = Number.parseInt(volumeGb, 10);
  const volumeInvalid =
    !Number.isFinite(parsedVolume) || parsedVolume < 0 || parsedVolume > 4000;

  // VM hardware is fixed/known — don't hit the RunPod availability API for it.
  const availability = useGpuAvailability(gpu, gpuCount, !isVm, cloudType);
  const explicitlyUnavailable =
    !isVm && availability.status === "ok" && availability.data.available === false;

  const parsedIdle = Number.parseInt(idleInput, 10);
  const idleInvalid =
    !Number.isFinite(parsedIdle) || parsedIdle < 0 || parsedIdle > 86400;

  const cleanedMembers = members
    .map((m) => ({ model: m.model.trim(), tp: m.tp, pp: m.pp || 1, extra_args: m.extra_args.trim() }))
    .filter((m) => m.model);
  // Total GPUs the fleet wants = Σ (tp × pp) across members.
  const gpusUsed = cleanedMembers.reduce((acc, m) => acc + m.tp * m.pp, 0);
  const oversubscribed = mode === "multi" && effectiveVmGpuCount > 0 && gpusUsed > effectiveVmGpuCount;
  const envVars = parseEnvVars(envText);
  const hasEnvVars = Object.keys(envVars).length > 0;

  type DeployArg = Parameters<typeof deployEndpoint>[0];
  function applyResult(res: Awaited<ReturnType<typeof deployEndpoint>>) {
    if (!res.ok) {
      if (res.unavailable) setUnavailableModal(res.unavailable);
      else setSubmitError(res.error);
      return;
    }
    toast.success(`Endpoint ${res.app_id} created`, { duration: 4000 });
    router.push(`/serverless/${encodeURIComponent(res.app_id)}`);
  }

  function submit() {
    setSubmitError(null);
    if (!name.trim()) {
      setSubmitError("Endpoint name is required.");
      return;
    }

    if (mode === "multi") {
      if (!isVm) {
        setSubmitError("Multi-model mode requires a VM provider.");
        return;
      }
      if (cleanedMembers.length === 0) {
        setSubmitError("Add at least one model.");
        return;
      }
      const names = cleanedMembers.map((m) => m.model);
      if (new Set(names).size !== names.length) {
        setSubmitError("Duplicate model names — each member must be unique.");
        return;
      }
      if (vdInvalid) {
        setSubmitError(`GPU IDs must be unique indices in 0..${(vmGpuCount || 1) - 1}.`);
        return;
      }
      const allowedGpus =
        vdIds.length > 0 && !vdInvalid
          ? new Set(vdIds)
          : new Set(Array.from({ length: vmGpuCount }, (_, k) => k));
      const modelsPayload: { model: string; tp: number; pp: number; extra_args: string; gpu_indices?: number[]; task?: "transcription" }[] = [];
      for (let i = 0; i < members.length; i++) {
        const raw = members[i];
        const mdl = raw.model.trim();
        if (!mdl) continue;
        const pp = raw.pp || 1;
        const width = raw.tp * pp; // GPUs this member occupies (tensor × pipeline)
        if (effectiveVmGpuCount > 0 && width > effectiveVmGpuCount) {
          setSubmitError(`${mdl}: tp×pp=${width} (tp=${raw.tp}, pp=${pp}) exceeds the ${effectiveVmGpuCount} selected GPUs.`);
          return;
        }
        const argErr = vllmArgsError(raw.extra_args.trim(), VLLM_RESERVED_MULTI, `model ${mdl}`);
        if (argErr) {
          setSubmitError(argErr);
          return;
        }
        // The field shows the suggested pin by default; submit what's shown so
        // "what you see is what deploys". Blank only happens when no GPUs are known.
        let gpu_indices: number[] | null;
        try {
          gpu_indices = parseGpuIds(raw.gpus || memberSuggestions[i] || "", width, `model ${mdl}`);
        } catch (e) {
          setSubmitError(e instanceof Error ? e.message : String(e));
          return;
        }
        if (gpu_indices && effectiveVmGpuCount > 0) {
          const bad = gpu_indices.filter((g) => !allowedGpus.has(g));
          if (bad.length) {
            const pool = [...allowedGpus].sort((a, b) => a - b).join(",");
            setSubmitError(`model ${mdl}: GPU id(s) ${bad.join(",")} aren't in the selected GPUs (${pool}).`);
            return;
          }
        }
        modelsPayload.push({
          model: mdl,
          tp: raw.tp,
          pp,
          extra_args: raw.extra_args.trim(),
          ...(gpu_indices ? { gpu_indices } : {}),
          ...(raw.audio ? { task: "transcription" } : {}),
        });
      }
      const body: DeployArg = {
        name: slugify(name),
        gpu: "vm",
        gpu_count: effectiveVmGpuCount,
        provider_id: providerId || null,
        mode: "multi",
        models: modelsPayload,
        sleep_level: sleepLevel,
        autoscaler: { max_containers: 1, tasks_per_container: 64, idle_timeout_s: 0 },
        enable_metrics: enableMetrics,
        ...(hasEnvVars ? { env_vars: envVars } : {}),
        ...(vdRaw ? { visible_devices: vdRaw } : {}),
        ...(venvPath.trim() ? { venv_path: venvPath.trim() } : {}),
        ...(vllmVersion.trim() ? { vllm_version: vllmVersion.trim() } : {}),
      };
      startTransition(async () => applyResult(await deployEndpoint(body)));
      return;
    }

    // single mode
    if (!model.trim()) {
      setSubmitError("Model name is required.");
      return;
    }
    if (idleInvalid) {
      setSubmitError("Enter a non-negative idle timeout in seconds (0 keeps the worker on forever).");
      return;
    }
    if (advancedInvalid) {
      setSubmitError("Fix the invalid values in Advanced options.");
      return;
    }
    if (!isVm && diskInvalid) {
      setSubmitError("Container disk must be between 1 and 2000 GB.");
      return;
    }
    if (!isVm && volumeInvalid) {
      setSubmitError("Volume must be between 0 and 4000 GB.");
      return;
    }
    if (explicitlyUnavailable) {
      const reason =
        (availability.status === "ok" && availability.data.reason) ||
        `${gpu}×${gpuCount} isn't available on the active provider right now.`;
      setSubmitError(reason);
      return;
    }
    if (isVm && vdInvalid) {
      setSubmitError(`GPU IDs must be unique indices in 0..${(vmGpuCount || 1) - 1}.`);
      return;
    }
    const vllmArgs = buildVllmArgs(vllm);
    const vllmArgErr = vllmArgsError(vllmArgs, VLLM_RESERVED_SINGLE, "vLLM args");
    if (vllmArgErr) {
      setSubmitError(vllmArgErr);
      return;
    }
    const body: DeployArg = {
      name: slugify(name),
      model: model.trim(),
      gpu: isVm ? "vm" : gpu,
      gpu_count: isVm ? effectiveVmGpuCount : gpuCount,
      autoscaler: { max_containers: MAX_WORKERS, idle_timeout_s: parsedIdle },
      vllm_args: vllmArgs,
      enable_metrics: enableMetrics,
      provider_id: providerId || null,
      ...(hasEnvVars ? { env_vars: envVars } : {}),
      ...(isVm
        ? {
            ...(vdRaw ? { visible_devices: vdRaw } : {}),
            ...(venvPath.trim() ? { venv_path: venvPath.trim() } : {}),
            ...(vllmVersion.trim() ? { vllm_version: vllmVersion.trim() } : {}),
          }
        : { cloud_type: cloudType, container_disk_gb: parsedDisk, volume_gb: parsedVolume }),
    };
    startTransition(async () => applyResult(await deployEndpoint(body)));
  }

  return (
    <div className="">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">Create inference endpoint</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Pick a framework and a model. The endpoint scales to zero when idle.
        </p>
      </div>

      <div className="space-y-5">
        <Section
          title="Run on"
          description="Default cloud spawns a fresh RunPod pod per worker. Bare metal runs on a VM you've registered under GPU Providers. Multi-model serving requires a VM."
        >
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <button
              type="button"
              onClick={() => {
                setTarget("cloud");
                setVmAvail({ status: "idle" });
              }}
              className={cn(
                "flex items-start gap-3 rounded-md border px-3 py-2.5 text-left text-sm transition-colors",
                target === "cloud"
                  ? "border-primary/60 bg-primary/5"
                  : "border-border hover:border-primary/40 hover:bg-muted/40",
              )}
            >
              <Cpu className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
              <div className="min-w-0">
                <div className="font-medium">Default cloud (RunPod)</div>
                <div className="text-xs text-muted-foreground">
                  Provision a fresh pod on demand. Pay-per-second.
                </div>
              </div>
            </button>
            <button
              type="button"
              onClick={() => {
                setTarget("vm");
                if (vmProviderId) refreshVmAvail(vmProviderId);
              }}
              className={cn(
                "flex items-start gap-3 rounded-md border px-3 py-2.5 text-left text-sm transition-colors",
                target === "vm"
                  ? "border-primary/60 bg-primary/5"
                  : "border-border hover:border-primary/40 hover:bg-muted/40",
              )}
            >
              <Server className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
              <div className="min-w-0">
                <div className="font-medium">Bare metal (VM)</div>
                <div className="text-xs text-muted-foreground">
                  SSH onto a registered VM. No spin-up cost.
                </div>
              </div>
            </button>
          </div>
        </Section>

        <Section
          title="Pod"
          description={
            target === "cloud"
              ? "GPU, count, and cloud tier for the RunPod workers."
              : "Which registered VM workers SSH into. Hardware is fixed by the VM."
          }
        >
          <div className="space-y-5">
            {target === "cloud" ? (
              <Field
                label="RunPod account"
                hint="Which RunPod provider to bill against. Default = gateway env key."
              >
                <Select
                  value={runpodProviderId || "__default__"}
                  onValueChange={(v) => setRunpodProviderId(v === "__default__" ? "" : v)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__default__">Gateway default (RunPod)</SelectItem>
                    {providers
                      .filter((p) => p.kind === "runpod")
                      .map((p) => (
                        <SelectItem key={p.id} value={p.id}>
                          {p.name}
                          {p.api_key_last4 ? ` · ****${p.api_key_last4}` : ""}
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>
              </Field>
            ) : (
              <Field
                label="VM provider"
                hint="The registered VM workers SSH into. Hardware is fixed by the VM."
              >
                {providers.filter((p) => p.kind === "vm").length === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    No VM providers registered. Add one at{" "}
                    <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">
                      GPU Providers → New provider
                    </a>
                    .
                  </p>
                ) : (
                  <Select
                    value={vmProviderId}
                    onValueChange={(id) => {
                      setVmProviderId(id);
                      refreshVmAvail(id);
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Pick a VM…" />
                    </SelectTrigger>
                    <SelectContent>
                      {providers
                        .filter((p) => p.kind === "vm")
                        .map((p) => (
                          <SelectItem key={p.id} value={p.id}>
                            {p.name}
                            {p.gpu_count != null && p.gpu_count > 0 ? ` · ${p.gpu_count} GPU` : ""}
                            {p.host ? ` · ${p.host}` : ""}
                          </SelectItem>
                        ))}
                    </SelectContent>
                  </Select>
                )}
                {vmProviderId && (
                  <div className="mt-1.5">
                    <VmAvailabilityRow state={vmAvail} onRefresh={() => refreshVmAvail(vmProviderId)} />
                  </div>
                )}
              </Field>
            )}

            {isVm ? (
              <>
              <Field
                label="GPUs (fixed)"
                hint="Detected on this VM. Multi-model packs models across these GPUs and time-shares them via sleep/wake."
              >
                <div className="rounded-md border border-border bg-muted/40 px-3 py-2 text-sm">
                  {vmGpuCount > 0
                    ? `${vmGpuCount} × ${selectedProvider?.gpus?.[0] ?? "GPU"}`
                    : "Not probed yet — run Test on the provider first."}
                </div>
              </Field>
              <Field
                label="GPU IDs (optional)"
                hint="Pin to specific GPU indices on the VM, e.g. 0,1,2,3 or 1,2,3,4. Empty = all the VM's GPUs. Sets CUDA_VISIBLE_DEVICES (single model) / restricts the multi-model packer to these GPUs."
              >
                <Input
                  value={visibleDevices}
                  onChange={(e) => setVisibleDevices(e.target.value)}
                  placeholder={
                    vmGpuCount
                      ? `e.g. ${Array.from({ length: Math.min(vmGpuCount, 4) }, (_, i) => i).join(",")}`
                      : "e.g. 0,1,2,3"
                  }
                  aria-invalid={vdInvalid}
                />
                {vdInvalid ? (
                  <p className="text-xs text-destructive">
                    Use unique indices in 0..{(vmGpuCount || 1) - 1}, comma-separated.
                  </p>
                ) : vdIds.length > 0 ? (
                  <p className="text-xs text-muted-foreground">
                    Using {vdIds.length} GPU{vdIds.length === 1 ? "" : "s"}: {vdIds.join(", ")}.
                  </p>
                ) : null}
              </Field>
              <Field
                label="vLLM venv path (optional)"
                hint="A uv venv on the VM that has vLLM, e.g. /share/vllm-venv. The worker runs {venv}/bin/python -m vllm. Empty = bare python3 on the VM's PATH."
              >
                <Input
                  value={venvPath}
                  onChange={(e) => setVenvPath(e.target.value)}
                  placeholder="/share/vllm-venv"
                  className="font-mono text-xs"
                />
              </Field>
              </>
            ) : null}

            {!isVm && (
            <>
            <Field
              label="Cloud tier"
              hint="Community is cheaper with variable hosts; Secure uses vetted hosts with more capacity."
            >
              <div className="grid grid-cols-2 gap-2">
                {(["COMMUNITY", "SECURE"] as const).map((tier) => (
                  <button
                    key={tier}
                    type="button"
                    onClick={() => setCloudType(tier)}
                    className={cn(
                      "rounded-md border p-3 text-left transition-colors",
                      cloudType === tier
                        ? "border-foreground/60 ring-1 ring-foreground/20"
                        : "border-border hover:border-foreground/40",
                    )}
                  >
                    <div className="text-sm font-medium">
                      {tier === "COMMUNITY" ? "Community" : "Secure"}
                    </div>
                    <div className="mt-0.5 text-xs text-muted-foreground">
                      {tier === "COMMUNITY"
                        ? "cheaper, variable hosts"
                        : "vetted hosts, more capacity"}
                    </div>
                  </button>
                ))}
              </div>
            </Field>

            <Field
              label="GPU"
              hint={(() => {
                const g = GPU_CHOICES.find((c) => c.value === gpu);
                return g ? capacityHint(g.vramGb, gpuCount) : undefined;
              })()}
              extra={<AvailabilityBadge state={availability} count={gpuCount} />}
            >
              <div className="flex gap-2">
                <SearchableSelect
                  className="flex-1"
                  value={gpu}
                  onChange={setGpu}
                  options={GPU_CHOICES.map((g) => ({
                    value: g.value,
                    label: g.label,
                    group: g.group,
                    hint: capacityHint(g.vramGb, 1),
                  }))}
                  placeholder="Choose a GPU"
                  searchPlaceholder="Search GPUs (e.g. h100, 24gb, ada)…"
                />
                <Select
                  value={String(gpuCount)}
                  onValueChange={(v) => setGpuCount(Number.parseInt(v, 10))}
                >
                  <SelectTrigger className="w-24 shrink-0">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {GPU_COUNT_CHOICES.map((n) => (
                      <SelectItem key={n} value={String(n)}>
                        ×{n}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </Field>

            <div className="grid grid-cols-2 gap-3">
              <Field
                label="Container disk (GB)"
                hint="Ephemeral workspace. Resets when the worker stops."
              >
                <Input
                  type="text"
                  inputMode="numeric"
                  value={containerDisk}
                  onChange={(e) => setContainerDisk(e.target.value)}
                  placeholder="50"
                  aria-invalid={diskInvalid}
                />
              </Field>
              <Field
                label="Volume (GB)"
                hint="Persistent volume. 0 = no persistent storage."
              >
                <Input
                  type="text"
                  inputMode="numeric"
                  value={volumeGb}
                  onChange={(e) => setVolumeGb(e.target.value)}
                  placeholder="0"
                  aria-invalid={volumeInvalid}
                />
              </Field>
            </div>

            <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>
                Pick a GPU with enough VRAM for your model. vLLM will fail to load if the
                weights plus KV cache exceed GPU memory.
              </span>
            </div>
            </>
            )}
          </div>
        </Section>

        <Section title="Endpoint" description="What you'll call this endpoint and the model it serves.">
          <div className="space-y-5">
            <Field
              label="Inference framework"
              hint="Choose the inference server. Only vLLM is enabled today."
            >
              <Select value={framework} onValueChange={setFramework}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {FRAMEWORKS.map((f) => (
                    <SelectItem key={f.value} value={f.value} disabled={!f.available}>
                      {f.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            {isVm && (
              <Field
                label="vLLM version (optional)"
                hint="Pin vLLM to this version in the VM's vLLM venv (set under Pod) — the worker uv pip installs it if missing. Empty = use whatever's installed."
              >
                <Input
                  value={vllmVersion}
                  onChange={(e) => setVllmVersion(e.target.value)}
                  placeholder="0.19.1"
                  className="font-mono text-xs"
                />
              </Field>
            )}

            <div className="rounded-md border border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
              {isVm ? (
                <>
                  <span className="font-medium text-foreground">Multi-model fleet</span> — models share this VM&apos;s
                  GPUs and swap in via sleep/wake. Add one model for a single-model endpoint (you still get the
                  sleep/wake benefits).
                </>
              ) : (
                <>
                  <span className="font-medium text-foreground">Single-model endpoint</span> — one model per RunPod /
                  PI pod, scale-to-zero. Use a VM provider for a multi-model fleet.
                </>
              )}
            </div>

            <Field label="Endpoint name" required>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="my-endpoint"
                className="bg-muted"
              />
            </Field>

            {mode === "single" ? (
              <Field label="Model" hint="Hugging Face repo (e.g. Qwen/Qwen2.5-7B-Instruct)" required>
                <Input
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder="Qwen/Qwen2.5-7B-Instruct"
                  className="bg-muted/50"
                />
              </Field>
            ) : (
              <div className="space-y-4">
                <Field
                  label="Models"
                  hint={`Each model has its own TP × PP (set by the dropdowns — don't add --tensor-parallel-size / --pipeline-parallel-size; it uses tp×pp GPUs, e.g. TP=2 × PP=3 = 6), its own GPU ids (pre-filled with a suggestion — edit to pin, e.g. 0,1,2,3 or 3,4,5,6), and its own vLLM args (e.g. --reasoning-parser / --tool-call-parser). Models on disjoint GPUs stay resident together; overlapping ones swap in via sleep/wake. Whisper/ASR models (e.g. openai/whisper-large-v3-turbo) work too — use TP=1; they're served via /v1/audio/transcriptions and get a Transcribe tab.`}
                  required
                >
                  <div className="space-y-2">
                    {members.map((m, i) => (
                      <div key={i} className="space-y-2 rounded-md border border-border p-2">
                        <div className="flex items-start gap-2">
                          <Input
                            className="flex-1 bg-muted/50"
                            value={m.model}
                            onChange={(e) =>
                              setMembers((arr) => arr.map((x, j) => (j === i ? { ...x, model: e.target.value } : x)))
                            }
                            placeholder="Qwen/Qwen3.6-35B-A3B"
                          />
                          <Select
                            value={String(m.tp)}
                            onValueChange={(v) =>
                              setMembers((arr) =>
                                arr.map((x, j) => (j === i ? { ...x, tp: Number.parseInt(v, 10) } : x)),
                              )
                            }
                          >
                            <SelectTrigger className="w-24 shrink-0">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {tpChoices.map((n) => (
                                <SelectItem key={n} value={String(n)}>
                                  TP={n}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <Select
                            value={String(m.pp || 1)}
                            onValueChange={(v) =>
                              setMembers((arr) =>
                                arr.map((x, j) => (j === i ? { ...x, pp: Number.parseInt(v, 10) } : x)),
                              )
                            }
                          >
                            <SelectTrigger className="w-24 shrink-0">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {ppChoices.map((n) => (
                                <SelectItem key={n} value={String(n)}>
                                  PP={n}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="shrink-0"
                            disabled={members.length <= 1}
                            onClick={() => setMembers((arr) => arr.filter((_, j) => j !== i))}
                          >
                            Remove
                          </Button>
                        </div>
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="shrink-0 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                            GPU ids
                          </span>
                          <Input
                            className="w-44 bg-muted/50 font-mono text-xs"
                            value={m.gpus || memberSuggestions[i]}
                            onChange={(e) =>
                              setMembers((arr) => arr.map((x, j) => (j === i ? { ...x, gpus: e.target.value } : x)))
                            }
                            placeholder="auto"
                            aria-label="GPU ids"
                          />
                          {memberSuggestions[i] && (
                            <span className="text-[11px] text-muted-foreground">
                              {(m.gpus || memberSuggestions[i]) === memberSuggestions[i]
                                ? "suggested — edit to pin different GPUs"
                                : `suggested: ${memberSuggestions[i]}`}
                            </span>
                          )}
                          <label className="ml-auto flex cursor-pointer select-none items-center gap-1.5 text-[11px] text-muted-foreground" title="Marks this as an audio/ASR (Whisper) model so the worker installs audio-decode deps. Set it for ASR finetunes whose name doesn't say 'whisper'.">
                            <input
                              type="checkbox"
                              checked={m.audio}
                              onChange={(e) =>
                                setMembers((arr) => arr.map((x, j) => (j === i ? { ...x, audio: e.target.checked } : x)))
                              }
                              className="h-3.5 w-3.5 accent-primary"
                            />
                            Audio / ASR (Whisper)
                          </label>
                        </div>
                        <Input
                          className="bg-muted/50 font-mono text-xs"
                          value={m.extra_args}
                          onChange={(e) =>
                            setMembers((arr) =>
                              arr.map((x, j) => (j === i ? { ...x, extra_args: cleanVllmArgs(e.target.value) } : x)),
                            )
                          }
                          placeholder="vLLM args, e.g. --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice --max-model-len 262144"
                        />
                      </div>
                    ))}
                  </div>
                  <button
                    type="button"
                    onClick={() => setMembers((arr) => [...arr, { model: "", tp: 1, pp: 1, extra_args: "", gpus: "", audio: false }])}
                    className="mt-2 text-xs text-primary hover:underline"
                  >
                    + Add model
                  </button>
                </Field>

                {oversubscribed && (
                  <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
                    <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                    <span>
                      Models need {gpusUsed} GPUs but the VM has {vmGpuCount} — they won&apos;t all stay
                      resident. Extra models are swapped in on demand via vLLM sleep/wake (first
                      request to a sleeping model waits for the swap).
                    </span>
                  </div>
                )}

                <Field
                  label="Sleep level"
                  hint="How an evicted model frees VRAM. L1 offloads weights to CPU RAM (fast wake, needs RAM); L2 discards them and reloads from disk (minimal RAM, slower wake)."
                >
                  <Select value={String(sleepLevel)} onValueChange={(v) => setSleepLevel(Number.parseInt(v, 10) as 1 | 2)}>
                    <SelectTrigger className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">Level 1 — offload to CPU RAM (fast)</SelectItem>
                      <SelectItem value="2">Level 2 — discard + reload from disk</SelectItem>
                    </SelectContent>
                  </Select>
                </Field>
              </div>
            )}
          </div>
        </Section>

        <Section title="Engine" description="Scaling behaviour, vLLM args, and metrics.">
          <div className="space-y-5">
            {mode === "multi" && (
              <p className="text-xs text-muted-foreground">
                Multi-model endpoints are always-on (no scale-to-zero); per-model vLLM args
                are set per model above. Models are evicted via sleep/wake, not torn down.
              </p>
            )}
            {mode === "single" && (
            <>
            <Field
              label="Idle timeout (s)"
              hint="Worker is torn down after this many seconds with no traffic. 0 keeps the worker on forever."
            >
              <Input
                type="text"
                inputMode="numeric"
                value={idleInput}
                onChange={(e) => setIdleInput(e.target.value)}
                placeholder="e.g. 300 (0 = always-on)"
                aria-invalid={idleInvalid}
              />
            </Field>

            <p className="text-xs text-muted-foreground">
              Max workers is fixed at <span className="font-medium text-foreground">1</span> for now.
            </p>

            <div className="border-t border-border pt-4">
              <button
                type="button"
                onClick={() => setAdvancedOpen((v) => !v)}
                className="flex w-full items-center gap-1.5 text-left text-xs font-medium uppercase tracking-wide text-muted-foreground hover:text-foreground"
              >
            {advancedOpen ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
            Advanced options (vLLM engine args)
          </button>
          {advancedOpen && (
            <div className="mt-4 space-y-4">
              <p className="text-xs text-muted-foreground">
                Defaults are sensible for most models. Override only when you know you need to.
                See{" "}
                <a
                  href="https://docs.vllm.ai/en/stable/configuration/engine_args/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-foreground"
                >
                  vLLM engine args
                </a>
                .
              </p>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <Field
                  label="max-model-len"
                  hint="Context window in tokens. Empty = model's default."
                >
                  <Input
                    type="text"
                    inputMode="numeric"
                    value={vllm.max_model_len}
                    onChange={(e) =>
                      setVllm((v) => ({ ...v, max_model_len: e.target.value }))
                    }
                    placeholder="e.g. 4096"
                    aria-invalid={intFieldInvalid(vllm.max_model_len)}
                  />
                </Field>
                <Field
                  label="gpu-memory-utilization"
                  hint="Fraction of VRAM vLLM may use (0–1). Default 0.9."
                >
                  <Input
                    type="text"
                    inputMode="decimal"
                    value={vllm.gpu_memory_utilization}
                    onChange={(e) =>
                      setVllm((v) => ({ ...v, gpu_memory_utilization: e.target.value }))
                    }
                    placeholder="0.9"
                    aria-invalid={gpuMemInvalid}
                  />
                </Field>
                <Field label="dtype" hint="Weight precision.">
                  <Select
                    value={vllm.dtype}
                    onValueChange={(val) => setVllm((v) => ({ ...v, dtype: val }))}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {DTYPE_CHOICES.map((d) => (
                        <SelectItem key={d} value={d}>
                          {d}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </Field>
                <Field
                  label="max-num-seqs"
                  hint="Max concurrent sequences. Empty = vLLM default."
                >
                  <Input
                    type="text"
                    inputMode="numeric"
                    value={vllm.max_num_seqs}
                    onChange={(e) =>
                      setVllm((v) => ({ ...v, max_num_seqs: e.target.value }))
                    }
                    placeholder="e.g. 256"
                    aria-invalid={intFieldInvalid(vllm.max_num_seqs)}
                  />
                </Field>
                <Field
                  label="tensor-parallel-size"
                  hint="Number of GPUs for tensor parallelism. Default 1."
                >
                  <Input
                    type="text"
                    inputMode="numeric"
                    value={vllm.tensor_parallel_size}
                    onChange={(e) =>
                      setVllm((v) => ({ ...v, tensor_parallel_size: e.target.value }))
                    }
                    placeholder="1"
                    aria-invalid={intFieldInvalid(vllm.tensor_parallel_size)}
                  />
                </Field>
              </div>
              <Field
                label="Extra args (raw)"
                hint="Appended verbatim to the vllm serve command. e.g. --enforce-eager --quantization awq"
              >
                <textarea
                  value={vllm.extra}
                  onChange={(e) => setVllm((v) => ({ ...v, extra: e.target.value }))}
                  placeholder="--enforce-eager"
                  rows={2}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
                />
              </Field>
              {buildVllmArgs(vllm) && (
                <div className="rounded-md bg-muted/50 px-3 py-2 text-xs">
                  <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                    Final command
                  </div>
                  <code className="break-words font-mono text-foreground">
                    vllm serve {`<model>`} {buildVllmArgs(vllm)}
                  </code>
                </div>
              )}
            </div>
          )}
        </div>
            </>
            )}

            {isVm && (
              <div className="border-t border-border pt-4">
                <Field
                  label="Environment variables"
                  hint="Applied to every vLLM process on the VM. One KEY=value per line (export / mkdir lines are fine — absolute-path values are auto-created). CUDA_VISIBLE_DEVICES is set per model automatically."
                >
                  <textarea
                    value={envText}
                    onChange={(e) => setEnvText(e.target.value)}
                    rows={6}
                    placeholder={"export HF_HOME=/share/huggingface\nexport TRITON_CACHE_DIR=/share/triton_cache\nexport VLLM_CACHE_ROOT=/share/vllm_cache\nexport TORCHINDUCTOR_CACHE_DIR=/share/torchinductor_cache"}
                    className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
                  />
                  {hasEnvVars && (
                    <p className="mt-1 text-xs text-muted-foreground">
                      {Object.keys(envVars).length} variable(s) parsed:{" "}
                      <span className="font-mono">{Object.keys(envVars).join(", ")}</span>
                    </p>
                  )}
                </Field>
              </div>
            )}

            <div className="border-t border-border pt-4">
              <label className="flex items-start gap-3 cursor-pointer">
                <Checkbox
                  checked={enableMetrics}
                  onCheckedChange={(v) => setEnableMetrics(v === true)}
                  className="mt-0.5"
                />
                <div className="flex-1">
                  <div className="text-sm font-medium">Enable metrics</div>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    Each worker self-installs Alloy + DCGM, node, and vLLM exporters
                    on boot via ansible-pull. Metrics push to VictoriaMetrics with
                    an{" "}
                    <code className="font-mono text-[11px]">endpoint=&lt;name&gt;</code>{" "}
                    label so you can filter the Grafana dashboard per endpoint.
                    Adds ~20 s to cold-start; runs in the background after vLLM is
                    ready so requests aren&apos;t delayed.
                  </p>
                </div>
              </label>
            </div>
          </div>
        </Section>
      </div>

      <div className="mt-5 flex items-center justify-end gap-3">
        {submitError && (
          <p className="mr-auto text-sm text-destructive">{submitError}</p>
        )}
        <Button variant="ghost" onClick={() => router.push("/serverless")} disabled={pending}>
          Cancel
        </Button>
        <Button
          onClick={submit}
          disabled={
            pending ||
            (isVm && vdInvalid) ||
            (mode === "single" &&
              (idleInvalid ||
                advancedInvalid ||
                (!isVm && (diskInvalid || volumeInvalid)) ||
                explicitlyUnavailable))
          }
        >
          {pending && <Loader2 className="h-4 w-4 animate-spin" />}
          Create endpoint
        </Button>
      </div>

      <Dialog
        open={unavailableModal !== null}
        onOpenChange={(open) => {
          if (!open) setUnavailableModal(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-600" />
              GPU not available right now
            </DialogTitle>
            <DialogDescription>
              The provider rejected the worker for{" "}
              <span className="font-mono text-foreground">
                {unavailableModal?.gpu}
              </span>{" "}
              ×{unavailableModal?.gpu_count}. The endpoint wasn&apos;t created so you
              can pick a different combo and retry.
            </DialogDescription>
          </DialogHeader>
          <div className="rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed text-foreground break-words">
            {unavailableModal?.reason}
          </div>
          <p className="text-xs text-muted-foreground">
            Suggestions: try a smaller count (e.g. ×1), pick a different GPU
            (RTX 3090, RTX 4090, A100 are usually well-stocked), or switch to
            on-demand idle so the worker only spawns when you actually fire a
            request.
          </p>
          <DialogFooter>
            <Button onClick={() => setUnavailableModal(null)}>Got it</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

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
        {description && (
          <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>
        )}
      </div>
      {children}
    </section>
  );
}

function Field({
  label,
  hint,
  required,
  children,
  extra,
}: {
  label: string;
  hint?: string;
  required?: boolean;
  children: React.ReactNode;
  extra?: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <Label className="text-xs uppercase tracking-wide text-muted-foreground">
          {label}
          {required && <span className="ml-1 text-destructive">*</span>}
        </Label>
        {extra}
      </div>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

// Parse a pasted env block into a dict. Accepts `KEY=value` and
// `export KEY=value`; skips blanks, comments, and `mkdir`/other shell lines.
function parseEnvVars(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const raw of text.split("\n")) {
    let line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    if (line.startsWith("export ")) line = line.slice("export ".length).trim();
    const eq = line.indexOf("=");
    if (eq <= 0) continue; // not a KEY=value line (e.g. mkdir -p …)
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

function slugify(s: string) {
  return s.toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "");
}

function suggestName(seed: string) {
  // Stable across SSR + client (seed is React's useId). Keep the last few
  // alphanumerics so the default looks like "endpoint-ab12c".
  const suffix = seed.replace(/[^a-z0-9]/gi, "").slice(-5).toLowerCase() || "1";
  return `endpoint-${suffix}`;
}

// Live VM availability row — mirrors the benchmark form's SSH probe summary.
function VmAvailabilityRow({
  state,
  onRefresh,
}: {
  state:
    | { status: "idle" }
    | { status: "loading" }
    | { status: "ok"; data: VmAvailability }
    | { status: "error"; message: string };
  onRefresh: () => void;
}) {
  if (state.status === "idle") return null;
  if (state.status === "loading") {
    return (
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        Checking availability via SSH…
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
  // Treat a GPU as "busy" if <20% memory free OR utilisation > 50%.
  const busy = data.gpus.filter((g) => g.mem_free_mib < g.mem_total_mib * 0.2 || g.util_pct > 50).length;
  const allFree = busy === 0;
  return (
    <div
      className={cn(
        "space-y-1 rounded-md border px-2.5 py-1.5 text-xs",
        allFree
          ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
          : "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
      )}
    >
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
