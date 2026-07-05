"use client";

import { useCallback, useEffect, useId, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { AlertCircle, AlertTriangle, Check, ChevronDown, ChevronRight, Cpu, Loader2, RefreshCw, Server, Wallet, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
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
import { FormFooter, FormShell } from "@/components/form-shell";
import { RegionSelect } from "@/components/region-select";
import { useGpuAvailability } from "@/lib/use-gpu-availability";
import { gateway } from "@/lib/gateway";
import { GPU_CHOICES, GPU_COUNT_CHOICES, capacityHint } from "@/lib/gpu-catalog";
import type { ProviderRecord, ProviderBalance, VmAvailability } from "@/lib/types";

// vLLM is what the live RunPod template runs. SGLang is a placeholder for a
// future template — keep it disabled so the option is visible but inert.
const FRAMEWORKS = [
  { value: "vllm", label: "vLLM", available: true },
  { value: "sglang", label: "SGLang (coming soon)", available: false },
] as const;

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

// Common pre-launch setup for high-tier (large-MoE) models — builds DeepGEMM.
const DEEPGEMM_SCRIPT =
  "bash <(curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh)";
// A nightly vLLM install (cu130) — handy starting point for the install-args field.
const VLLM_NIGHTLY_ARGS =
  "-U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu130 --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match";
// Default vLLM version installed into the VM/pod venv when none is pinned.
const DEFAULT_VLLM_VERSION = "0.23.0";
// Custom vLLM fork (git) — the Gemma-4 FA4 "CUTE" fork + what it needs to run.
const GEMMA4_FA4_FORK_URL = "https://github.com/Scicom-AI-Enterprise-Organization/vllm-gemma4-fa4-cute";
const GEMMA4_FA4_REF = "main";
const GEMMA4_FA4_BACKEND = "--attention-backend FLASH_ATTN_CUTE"; // activates the FA4 CUTE backend
// The FA4 CUTE backend REQUIRES the flash-attention-512 "cute" fork (it provides
// `flash_attn.cute`, which the vLLM fork imports). Installed in the pre-launch
// step below (the vLLM fork's own deps don't include it).
const GEMMA4_FA4_FLASH_ATTN_URL = "https://github.com/Scicom-AI-Enterprise-Organization/flash-attention-512";
// The fork must live in its OWN venv — NOT the shared /share/vllm-venv (hand-built,
// no .sgpu_vllm_spec marker → the worker refuses to touch it, so the fork would never install).
const GEMMA4_FA4_VENV = "/share/vllm-gemma4-fa4-venv";
// vLLM 0.23.0 ships a prometheus-fastapi-instrumentator that 500s every route
// (incl. /health) → the worker never goes healthy. Pin the fixed release.
const VLLM_PROM_FIX = 'uv pip install -U "prometheus-fastapi-instrumentator>=7"';
// Pre-launch script for the Gemma-4 FA4 fork, run by the worker once in the fork
// venv (GEMMA4_FA4_VENV) after vLLM is installed and before serving:
//   1. install the flash-attention-512 cute fork (provides flash_attn.cute), then
//   2. PIN nvidia-cutlass-dsl==4.4.2 + quack-kernels==0.3.10 — the fork's CLAUDE.md
//      says cutlass-dsl 4.5.x breaks FA4 ("nvvm has no attribute vote_ballot_sync"),
//      so force the known-good combo AFTER any install that pulled a newer one, then
//   3. the vLLM 0.23.0 prometheus fix. (Adjust the cutlass [cuXX] extra if the box
//      isn't CUDA 13.)
const GEMMA4_FA4_PRESCRIPT = [
  `uv pip install "git+${GEMMA4_FA4_FLASH_ATTN_URL}.git@main#subdirectory=flash_attn/cute"`,
  'uv pip install "nvidia-cutlass-dsl==4.4.2" "quack-kernels==0.3.10"',
  VLLM_PROM_FIX,
].join("\n");

// Compose a verbatim `uv pip install` arg string for a git-fork vLLM. A leading
// `VLLM_USE_PRECOMPILED=1` (the worker reads leading NAME=VALUE tokens as install
// env, not pip args) reuses precompiled vLLM binaries — fast, no CUDA toolchain.
// Uncheck precompiled to build CUDA from source. extraDeps (e.g. cutlass-dsl) are
// installed in the same command; --torch-backend=auto picks the driver-matched torch.
function composeForkArgs(url: string, ref: string, precompiled: boolean, extraDeps: string[] = []): string {
  const u = url.trim();
  if (!u) return "";
  const spec = `git+${u}${ref.trim() ? "@" + ref.trim() : ""}`;
  return [
    ...(precompiled ? ["VLLM_USE_PRECOMPILED=1"] : []),
    spec,
    ...extraDeps,
    "--torch-backend=auto",
  ].join(" ");
}

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
  const [cloudType, setCloudType] = useState<"COMMUNITY" | "SECURE">("SECURE");
  // RunPod data center to pin ("" = Auto → RunPod picks). Cloud-only.
  const [dataCenterId, setDataCenterId] = useState("");
  const [containerDisk, setContainerDisk] = useState<string>("50");
  const [volumeGb, setVolumeGb] = useState<string>("50");
  const [idleInput, setIdleInput] = useState("120"); // 2 min scale-to-zero (single-model only; multi VM fleets are always-on)
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [enableMetrics, setEnableMetrics] = useState(true);
  // Visibility at create time. false (default) = private (only you + admins).
  // true = public: read-only visible to every logged-in user (they can view the
  // overview/workers/metrics but can't edit, delete, or run inference). Owner can
  // flip it later from the endpoint detail page.
  const [isPublic, setIsPublic] = useState(false);
  // Run-on target (mirrors the benchmark form): "cloud" spawns a fresh RunPod
  // pod, "vm" SSHes onto a registered VM. Each keeps its own provider pick.
  const [target, setTarget] = useState<"cloud" | "vm">("cloud");
  const [vmProviderId, setVmProviderId] = useState<string>("");
  const [runpodProviderId, setRunpodProviderId] = useState<string>("");
  const [providers, setProviders] = useState<ProviderRecord[]>([]);
  const [vllm, setVllm] = useState({ ...DEFAULT_VLLM_ARGS });
  const [members, setMembers] = useState<{ model: string; tp: number; pp: number; extra_args: string; gpus: string }[]>([
    { model: "", tp: 1, pp: 1, extra_args: "", gpus: "" },
  ]);
  const [sleepLevel, setSleepLevel] = useState<1 | 2>(1);
  // Cloud (RunPod) can also run a multi-model fleet — same GPU time-sharing as a
  // VM, but billed hourly so it honours the idle timeout (deletes the pod after N
  // idle seconds, re-provisioned on demand). VM fleets are always multi + always-on.
  const [cloudMulti, setCloudMulti] = useState(false);
  // VM deployment shape: a multi-model fleet (queue + GPU time-share via sleep/wake)
  // or a single-model "proxy" endpoint — one always-on model the gateway proxies to
  // directly over a forward tunnel (no queue, no sleep). Only meaningful when isVm.
  const [vmKind, setVmKind] = useState<"fleet" | "proxy">("fleet");
  const [runpodBalance, setRunpodBalance] = useState<ProviderBalance | null>(null);
  const [runpodBalanceLoading, setRunpodBalanceLoading] = useState(false);
  // VM-only: pin to specific physical GPU ids, e.g. "0,1,2,3". Empty = all GPUs.
  const [visibleDevices, setVisibleDevices] = useState("");
  // VM-only: uv venv the worker runs `vllm serve` from + the vLLM version to pin.
  // Default to the standard shared venv on the VM — leaving this empty makes the
  // worker fall back to bare `python3` (no vLLM), which silently never launches.
  const [venvPath, setVenvPath] = useState("/share/vllm-venv");
  const [vllmVersion, setVllmVersion] = useState(DEFAULT_VLLM_VERSION);
  // Advanced: a full `uv pip install` arg string for vLLM, used verbatim instead of
  // the version — e.g. a nightly with extra index URLs, or a git fork. Overrides the version.
  const [vllmInstallArgs, setVllmInstallArgs] = useState("");
  // Custom vLLM fork (git): repo URL + ref, installed via `git+…@ref`. Precompiled
  // (default) overlays stock binaries onto the fork's Python (fast); uncheck to build
  // CUDA from source. Composing fills vllmInstallArgs (which overrides the version).
  const [forkUrl, setForkUrl] = useState("");
  const [forkRef, setForkRef] = useState("main");
  const [forkPrecompiled, setForkPrecompiled] = useState(true);
  // Append a serve flag (e.g. --attention-backend FLASH_ATTN_CUTE) to every member
  // that doesn't already carry it. VM (proxy/fleet) + cloud-multi serve via members.
  const addServeFlagToMembers = (flag: string) =>
    setMembers((arr) =>
      arr.map((m) =>
        m.extra_args.includes(flag.split(" ")[1] ?? flag)
          ? m
          : { ...m, extra_args: (m.extra_args.trim() ? m.extra_args.trim() + " " : "") + flag },
      ),
    );
  // Fill vllmInstallArgs from a git fork (URL+ref+precompiled). Install args override
  // the version field, so blank the version to avoid a confusing dual spec.
  const applyFork = (extraDeps: string[] = []) => {
    const args = composeForkArgs(forkUrl, forkRef, forkPrecompiled, extraDeps);
    if (!args) return;
    setVllmInstallArgs(args);
    setVllmVersion("");
  };
  // One-click: the Gemma-4 FA4 CUTE fork — git install (precompiled) of the vLLM
  // fork, the FA4 serve flag on every member, and a pre-launch script that installs
  // the REQUIRED flash-attention-512 cute fork + pins cutlass-dsl/quack to the
  // FA4-compatible combo (+ the vLLM 0.23.0 prometheus fix).
  const useGemma4Fa4Preset = () => {
    setForkUrl(GEMMA4_FA4_FORK_URL);
    setForkRef(GEMMA4_FA4_REF);
    setForkPrecompiled(true);
    // No extra deps in the vLLM install itself — the flash-attn fork + the cutlass/
    // quack pins are applied in the pre-launch script (which runs last, so its pins win).
    setVllmInstallArgs(composeForkArgs(GEMMA4_FA4_FORK_URL, GEMMA4_FA4_REF, true));
    setVllmVersion("");
    // Install the fork into its OWN venv, never the shared hand-built /share/vllm-venv
    // (no marker → the worker won't touch it, so the fork would silently never install).
    setVenvPath(GEMMA4_FA4_VENV);
    addServeFlagToMembers(GEMMA4_FA4_BACKEND);
    setPreScript((s) =>
      s.includes("flash-attention-512")
        ? s
        : (s.trim() ? s.trimEnd() + "\n" : "") + GEMMA4_FA4_PRESCRIPT,
    );
  };
  // Optional setup script the worker runs once after the venv is ready and before
  // launching models — e.g. building DeepGEMM. Empty = none.
  const [preScript, setPreScript] = useState("");
  // HuggingFace cache dir, exported as HF_HOME to every vLLM process. On a
  // mounted volume → downloaded weights persist across (re-)provisions.
  const [hfHome, setHfHome] = useState("/share/huggingface");
  // HF auth token for gated / private models → the HF_TOKEN env var. Pick a
  // global secret (referenced, resolved server-side at run-time) or paste one.
  const [hfTokenSource, setHfTokenSource] = useState<"secret" | "paste">("secret");
  const [hfToken, setHfToken] = useState("");
  const [hfTokenSecret, setHfTokenSecret] = useState("");
  const [secretKeys, setSecretKeys] = useState<string[]>([]);
  // Endpoint-level env applied to every vLLM process (cache/home dirs, etc.).
  // Pasted as `KEY=value` / `export KEY=value` lines; `mkdir` lines are ignored
  // (the worker auto-creates absolute-path values).
  const [envText, setEnvText] = useState("");

  useEffect(() => {
    gateway.listProviders().then(setProviders).catch(() => {});
  }, []);

  // Global secrets (admin Secrets) the HF token can reference — keys only.
  useEffect(() => {
    let cancel = false;
    fetch("/api/proxy/v1/global-env", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : []))
      .then((rows) => {
        if (!cancel && Array.isArray(rows)) setSecretKeys(rows.map((r: { key: string }) => r.key));
      })
      .catch(() => {});
    return () => {
      cancel = true;
    };
  }, []);

  // Fetch the selected RunPod account's credit (named accounts only — the gateway
  // default has no provider row). Re-runs whenever the account dropdown changes.
  useEffect(() => {
    if (target !== "cloud" || !runpodProviderId) {
      setRunpodBalance(null);
      return;
    }
    let cancelled = false;
    setRunpodBalanceLoading(true);
    setRunpodBalance(null);
    gateway
      .getProviderBalance(runpodProviderId)
      .then((b) => { if (!cancelled) setRunpodBalance(b); })
      .catch(() => { if (!cancelled) setRunpodBalance(null); })
      .finally(() => { if (!cancelled) setRunpodBalanceLoading(false); });
    return () => { cancelled = true; };
  }, [target, runpodProviderId]);

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
  // Serving mode: a VM is always a multi-model fleet (single-model serving +
  // sleep/wake eviction). Cloud (RunPod) is single-model by default, or a
  // multi-model fleet when the user opts in (cloudMulti) — same time-sharing as
  // VM, plus idle-timeout auto-delete.
  const mode: "single" | "multi" | "proxy" =
    isVm ? (vmKind === "proxy" ? "proxy" : "multi") : cloudMulti ? "multi" : "single";
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
  // GPU universe the fleet packs onto: a VM uses its (optionally pinned) probed
  // GPUs; a cloud multi-model fleet uses the requested RunPod pod gpuCount.
  const fleetGpus = isVm ? effectiveVmGpuCount : gpuCount;
  // tp choices for multi: any (power-of-two) size up to the usable GPU count. We
  // do NOT require tp to divide the count — the packer + optional per-model GPU
  // pin handle non-dividing layouts, e.g. tp=4 on 7 GPUs → [0,1,2,3].
  const tpChoices = fleetGpus > 0
    ? GPU_COUNT_CHOICES.filter((n) => n <= fleetGpus)
    : [1, 2, 4, 8];
  // Pipeline-parallel choices: any integer 1..usable (PP need not be a power of
  // two — e.g. TP=2 × PP=3 = 6 GPUs). A member uses tp × pp GPUs total.
  const ppChoices = Array.from(
    { length: Math.max(1, fleetGpus || 8) },
    (_, i) => i + 1,
  );
  // Physical GPU universe used to suggest per-model pins. VM: an explicit
  // visible_devices pin, else 0..vmGpuCount-1. Cloud multi: 0..gpuCount-1.
  // Suggestions mirror the gateway's auto-packer so "what you see is what deploys".
  const physForSuggest =
    mode !== "multi"
      ? []
      : isVm && vdIds.length > 0 && !vdInvalid
        ? vdIds
        : Array.from({ length: fleetGpus }, (_, i) => i);
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
  const oversubscribed = mode === "multi" && fleetGpus > 0 && gpusUsed > fleetGpus;
  // HF_HOME + HF_TOKEN (their own fields) are exported as env vars; an explicit
  // value in the env-vars box wins. Only where the venv/env fields show (VM/fleet).
  // HF_TOKEN is either a pasted value or a `secret://KEY` ref the gateway resolves.
  const hfTokenEnv: Record<string, string> =
    !(isVm || cloudMulti)
      ? {}
      : hfTokenSource === "paste" && hfToken.trim()
        ? { HF_TOKEN: hfToken.trim() }
        : hfTokenSource === "secret" && hfTokenSecret
          ? { HF_TOKEN: `secret://${hfTokenSecret}` }
          : {};
  const envVars = {
    ...((isVm || cloudMulti) && hfHome.trim() ? { HF_HOME: hfHome.trim() } : {}),
    ...hfTokenEnv,
    ...parseEnvVars(envText),
  };
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
    if (target === "cloud" && !runpodProviderId) {
      setSubmitError("Select a RunPod provider — add one under GPU Providers.");
      return;
    }

    if (mode === "multi" || mode === "proxy") {
      if (cleanedMembers.length === 0) {
        setSubmitError("Add at least one model.");
        return;
      }
      if (mode === "proxy" && cleanedMembers.length !== 1) {
        setSubmitError("A proxy endpoint serves exactly one model.");
        return;
      }
      const names = cleanedMembers.map((m) => m.model);
      if (new Set(names).size !== names.length) {
        setSubmitError("Duplicate model names — each member must be unique.");
        return;
      }
      if (isVm && vdInvalid) {
        setSubmitError(`GPU IDs must be unique indices in 0..${(vmGpuCount || 1) - 1}.`);
        return;
      }
      if (!isVm && (idleInvalid)) {
        setSubmitError("Idle timeout must be 0–86400 seconds.");
        return;
      }
      // The GPU pool models may pin onto: a VM's pinned/probed ids, else 0..N-1
      // of the fleet's GPU count (cloud uses the requested gpuCount).
      const allowedGpus =
        isVm && vdIds.length > 0 && !vdInvalid
          ? new Set(vdIds)
          : new Set(Array.from({ length: fleetGpus }, (_, k) => k));
      const modelsPayload: { model: string; tp: number; pp: number; extra_args: string; gpu_indices?: number[]; task?: "transcription" }[] = [];
      for (let i = 0; i < members.length; i++) {
        const raw = members[i];
        const mdl = raw.model.trim();
        if (!mdl) continue;
        const pp = raw.pp || 1;
        const width = raw.tp * pp; // GPUs this member occupies (tensor × pipeline)
        if (fleetGpus > 0 && width > fleetGpus) {
          setSubmitError(`${mdl}: tp×pp=${width} (tp=${raw.tp}, pp=${pp}) exceeds the ${fleetGpus} GPUs.`);
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
        if (gpu_indices && fleetGpus > 0) {
          const bad = gpu_indices.filter((g) => !allowedGpus.has(g));
          if (bad.length) {
            const pool = [...allowedGpus].sort((a, b) => a - b).join(",");
            setSubmitError(`model ${mdl}: GPU id(s) ${bad.join(",")} aren't in the GPU pool (${pool}).`);
            return;
          }
        }
        modelsPayload.push({
          model: mdl,
          tp: raw.tp,
          pp,
          extra_args: raw.extra_args.trim(),
          ...(gpu_indices ? { gpu_indices } : {}),
        });
      }
      // VM multi-model fleet: always-on (idle 0), the VM's own GPUs, ships a
      // worker-agent tarball over SSH (venv/visible_devices/vllm_version apply).
      // Cloud multi-model fleet (RunPod): the requested GPU type + count, honours
      // the idle timeout (deletes the pod after N idle seconds), pod-baked image.
      const body: DeployArg = isVm
        ? {
            name: slugify(name),
            gpu: "vm",
            gpu_count: fleetGpus,
            provider_id: providerId || null,
            // "proxy" = single-model VM endpoint (no queue, no sleep); "multi" =
            // fleet. Both ship the same per-member spec; the gateway branches on mode.
            mode,
            models: modelsPayload,
            sleep_level: sleepLevel,
            autoscaler: { max_containers: 1, tasks_per_container: 64, idle_timeout_s: 0 },
            enable_metrics: enableMetrics,
            is_public: isPublic,
            ...(hasEnvVars ? { env_vars: envVars } : {}),
            ...(vdRaw ? { visible_devices: vdRaw } : {}),
            ...(venvPath.trim() ? { venv_path: venvPath.trim() } : {}),
            ...(vllmVersion.trim() ? { vllm_version: vllmVersion.trim() } : {}),
            ...(vllmInstallArgs.trim() ? { vllm_install_args: vllmInstallArgs.trim() } : {}),
            ...(preScript.trim() ? { pre_script: preScript } : {}),
          }
        : {
            name: slugify(name),
            gpu,
            gpu_count: gpuCount,
            cloud_type: cloudType,
            data_center_id: dataCenterId || undefined,
            container_disk_gb: parsedDisk,
            volume_gb: parsedVolume,
            provider_id: providerId || null,
            mode: "multi",
            models: modelsPayload,
            sleep_level: sleepLevel,
            autoscaler: { max_containers: 1, tasks_per_container: 64, idle_timeout_s: parsedIdle },
            enable_metrics: enableMetrics,
            is_public: isPublic,
            ...(hasEnvVars ? { env_vars: envVars } : {}),
            // The fleet installs vLLM into its venv (a volume path persists across
            // re-provisions); pin the version (default 0.19.1) for reproducibility.
            ...(venvPath.trim() ? { venv_path: venvPath.trim() } : {}),
            // A custom install-args string owns the whole spec → don't also pin a
            // default version (let the worker run the args verbatim).
            ...(vllmInstallArgs.trim()
              ? { vllm_install_args: vllmInstallArgs.trim() }
              : { vllm_version: vllmVersion.trim() || DEFAULT_VLLM_VERSION }),
            ...(preScript.trim() ? { pre_script: preScript } : {}),
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
      is_public: isPublic,
      provider_id: providerId || null,
      ...(hasEnvVars ? { env_vars: envVars } : {}),
      ...(isVm
        ? {
            ...(vdRaw ? { visible_devices: vdRaw } : {}),
            ...(venvPath.trim() ? { venv_path: venvPath.trim() } : {}),
            ...(vllmVersion.trim() ? { vllm_version: vllmVersion.trim() } : {}),
            ...(vllmInstallArgs.trim() ? { vllm_install_args: vllmInstallArgs.trim() } : {}),
            ...(preScript.trim() ? { pre_script: preScript } : {}),
          }
        : { cloud_type: cloudType, data_center_id: dataCenterId || undefined, container_disk_gb: parsedDisk, volume_gb: parsedVolume }),
    };
    startTransition(async () => applyResult(await deployEndpoint(body)));
  }

  return (
    <FormShell>
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
                hint="Which registered RunPod provider to run on."
              >
                {providers.filter((p) => p.kind === "runpod").length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    No RunPod providers registered.{" "}
                    <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">
                      Add one
                    </a>{" "}
                    under GPU Providers.
                  </p>
                ) : (
                  <Select value={runpodProviderId} onValueChange={setRunpodProviderId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Choose a RunPod account…" />
                    </SelectTrigger>
                    <SelectContent>
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
                )}
                {runpodProviderId && (
                  <p className="mt-1 flex items-center gap-1.5 text-xs text-muted-foreground">
                    <Wallet className="h-3.5 w-3.5" />
                    {runpodBalanceLoading
                      ? "Checking credit…"
                      : runpodBalance?.ok && typeof runpodBalance.balance === "number"
                        ? <>Credit: <span className="font-medium text-emerald-600">${runpodBalance.balance.toFixed(2)}</span></>
                        : "Credit unavailable"}
                  </p>
                )}
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
              </>
            ) : null}

            {!isVm && (
            <>
            <Field
              label="Cloud tier"
              hint="Community is cheaper with variable hosts; Secure uses vetted hosts with more capacity."
            >
              <div className="grid grid-cols-2 gap-2">
                {(["SECURE", "COMMUNITY"] as const).map((tier) => (
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
              label="Region"
              hint="Pin the pod to a RunPod data center, or Auto to let RunPod pick any region with capacity."
            >
              <RegionSelect value={dataCenterId} onChange={setDataCenterId} className="text-sm" />
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
                hint="Persistent volume mounted at /workspace (model cache). 0 = no persistent storage."
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

            {target === "cloud" && (
              <Field
                label="Serving mode"
                hint="Single = one model per pod, scale-to-zero. Multi-model fleet = several models time-sharing the pod's GPUs (sleep/wake), deleted after the idle timeout and re-provisioned on demand."
              >
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setCloudMulti(false)}
                    className={cn(
                      "rounded-md border px-3 py-2 text-left text-sm transition-colors",
                      !cloudMulti ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40",
                    )}
                  >
                    <div className="font-medium">Single model</div>
                    <div className="text-xs text-muted-foreground">One model, scale-to-zero.</div>
                  </button>
                  <button
                    type="button"
                    onClick={() => setCloudMulti(true)}
                    className={cn(
                      "rounded-md border px-3 py-2 text-left text-sm transition-colors",
                      cloudMulti ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40",
                    )}
                  >
                    <div className="font-medium">Multi-model fleet</div>
                    <div className="text-xs text-muted-foreground">Many models, idle auto-delete.</div>
                  </button>
                </div>
              </Field>
            )}

            {isVm && (
              <Field
                label="Deployment"
                hint="Multi-model fleet = several models time-share the VM's GPUs (queue + sleep/wake). Single model (proxy) = one always-on model the gateway proxies to directly over a forward tunnel — no queue, no sleep."
              >
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setVmKind("proxy");
                      setMembers((arr) => arr.slice(0, 1));
                    }}
                    className={cn(
                      "rounded-md border px-3 py-2 text-left text-sm transition-colors",
                      vmKind === "proxy" ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40",
                    )}
                  >
                    <div className="font-medium">Single model (proxy)</div>
                    <div className="text-xs text-muted-foreground">One model, no queue/sleep.</div>
                  </button>
                  <button
                    type="button"
                    onClick={() => setVmKind("fleet")}
                    className={cn(
                      "rounded-md border px-3 py-2 text-left text-sm transition-colors",
                      vmKind === "fleet" ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40",
                    )}
                  >
                    <div className="font-medium">Multi-model fleet</div>
                    <div className="text-xs text-muted-foreground">Many models, sleep/wake.</div>
                  </button>
                </div>
              </Field>
            )}

            <div className="rounded-md border border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
              {isVm && vmKind === "proxy" ? (
                <>
                  <span className="font-medium text-foreground">Single model · direct proxy</span> — one always-on
                  model on this VM. The gateway forwards each request straight to it over a tunnel: no queue, no
                  sleep/wake, no scale-to-zero.
                </>
              ) : isVm ? (
                <>
                  <span className="font-medium text-foreground">Multi-model fleet</span> — models share this VM&apos;s
                  GPUs and swap in via sleep/wake. Add one model for a single-model endpoint (you still get the
                  sleep/wake benefits).
                </>
              ) : cloudMulti ? (
                <>
                  <span className="font-medium text-foreground">Multi-model fleet on RunPod</span> — several models
                  time-share the pod&apos;s GPUs via sleep/wake. The whole pod is deleted after the idle timeout and
                  re-provisioned on the next request (set Idle timeout under Engine).
                </>
              ) : (
                <>
                  <span className="font-medium text-foreground">Single-model endpoint</span> — one model per RunPod /
                  PI pod, scale-to-zero.
                </>
              )}
            </div>

            {(isVm || cloudMulti) && (
              <Field
                label="vLLM version (optional)"
                hint={
                  isVm
                    ? `Pin vLLM to this version in the VM's vLLM venv (below) — the worker uv pip installs it if missing. Default ${DEFAULT_VLLM_VERSION}; empty = use whatever's installed.`
                    : `Pin vLLM to this version — installed into the pod's venv on each provision. Empty = ${DEFAULT_VLLM_VERSION}.`
                }
              >
                <Input
                  value={vllmVersion}
                  onChange={(e) => setVllmVersion(e.target.value)}
                  placeholder={DEFAULT_VLLM_VERSION}
                  disabled={!!vllmInstallArgs.trim()}
                  className="font-mono text-xs"
                />
              </Field>
            )}

            {(isVm || cloudMulti) && (
              <Field
                label="Custom vLLM fork (git)"
                hint="Install a forked vLLM from a git repo (e.g. the Gemma-4 FA4 CUTE fork). Precompiled reuses stock binaries over the fork's Python — fast, no CUDA toolchain; uncheck to build CUDA from source. Applying fills the install args below (which override the version)."
              >
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={useGemma4Fa4Preset}
                    className="rounded border border-border px-2 py-1 text-[11px] text-muted-foreground hover:bg-muted"
                  >
                    Use Gemma-4 FA4 fork
                  </button>
                  <span className="text-[11px] text-muted-foreground">
                    sets a dedicated venv + {GEMMA4_FA4_BACKEND} + the 0.23.0 prometheus fix
                  </span>
                </div>
                <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                  <Input
                    value={forkUrl}
                    onChange={(e) => setForkUrl(e.target.value)}
                    placeholder="https://github.com/org/vllm-fork"
                    className="font-mono text-xs sm:flex-1"
                  />
                  <Input
                    value={forkRef}
                    onChange={(e) => setForkRef(e.target.value)}
                    placeholder="main"
                    className="font-mono text-xs sm:w-32"
                  />
                </div>
                <div className="mt-2 flex flex-wrap items-center gap-3">
                  <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={forkPrecompiled}
                      onChange={(e) => setForkPrecompiled(e.target.checked)}
                    />
                    Use precompiled binaries (uncheck = build CUDA from source)
                  </label>
                  <button
                    type="button"
                    disabled={!forkUrl.trim()}
                    onClick={() => applyFork()}
                    className="rounded border border-border px-2 py-1 text-[11px] text-muted-foreground hover:bg-muted disabled:opacity-40"
                  >
                    Apply to install args ↓
                  </button>
                </div>
              </Field>
            )}

            {(isVm || cloudMulti) && (
              <Field
                label="vLLM install args (advanced)"
                hint="Full `uv pip install` args for vLLM, used verbatim (overrides the version above) — for nightly / custom CUDA / git-fork builds. Leading NAME=VALUE tokens become install env (e.g. VLLM_USE_PRECOMPILED=1). The worker runs `uv pip install --python {venv}/bin/python <these args>`."
              >
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setVllmInstallArgs(VLLM_NIGHTLY_ARGS)}
                    className="rounded border border-border px-2 py-1 text-[11px] text-muted-foreground hover:bg-muted"
                  >
                    Insert nightly (cu130)
                  </button>
                  {vllmInstallArgs.trim() && (
                    <button
                      type="button"
                      onClick={() => setVllmInstallArgs("")}
                      className="text-[11px] text-muted-foreground underline hover:text-foreground"
                    >
                      clear
                    </button>
                  )}
                </div>
                <Textarea
                  value={vllmInstallArgs}
                  onChange={(e) => setVllmInstallArgs(e.target.value)}
                  placeholder={VLLM_NIGHTLY_ARGS}
                  rows={3}
                  className="mt-2 font-mono text-xs"
                />
              </Field>
            )}

            {(isVm || cloudMulti) && (
              <Field
                label="vLLM venv path"
                hint={
                  isVm
                    ? "A uv venv on the VM that has vLLM, e.g. /share/vllm-venv. The worker runs {venv}/bin/python -m vllm. Empty = bare python3 on PATH."
                    : "Where the pod builds its vLLM venv. Put it on a mounted volume (e.g. /share/vllm-venv) so vLLM isn't reinstalled on each re-provision."
                }
              >
                <Input
                  value={venvPath}
                  onChange={(e) => setVenvPath(e.target.value)}
                  placeholder="/share/vllm-venv"
                  className="font-mono text-xs"
                />
              </Field>
            )}

            {(isVm || cloudMulti) && (
              <Field
                label="Pre-launch script (optional)"
                hint="Shell run once per worker boot, after the venv is ready and before models launch — with the venv on PATH. For setup that isn't a pip install, e.g. building DeepGEMM. Runs under bash (process substitution works)."
              >
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() =>
                      setPreScript((s) =>
                        s.includes(DEEPGEMM_SCRIPT)
                          ? s
                          : (s.trim() ? s.trimEnd() + "\n" : "") + DEEPGEMM_SCRIPT,
                      )
                    }
                    className="rounded border border-border px-2 py-1 text-[11px] text-muted-foreground hover:bg-muted"
                  >
                    + Install DeepGEMM
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setPreScript((s) =>
                        s.includes("prometheus-fastapi-instrumentator")
                          ? s
                          : (s.trim() ? s.trimEnd() + "\n" : "") + VLLM_PROM_FIX,
                      )
                    }
                    className="rounded border border-border px-2 py-1 text-[11px] text-muted-foreground hover:bg-muted"
                  >
                    + Fix vLLM 0.23.0 prometheus
                  </button>
                  <span className="text-[11px] text-muted-foreground">common for high-tier (large-MoE) models</span>
                </div>
                <Textarea
                  value={preScript}
                  onChange={(e) => setPreScript(e.target.value)}
                  placeholder={DEEPGEMM_SCRIPT}
                  rows={3}
                  className="mt-2 font-mono text-xs"
                />
              </Field>
            )}

            {(isVm || cloudMulti) && (
              <Field
                label="HF_HOME (model cache)"
                hint="HuggingFace cache dir, exported to every vLLM process. Put it on a mounted volume so downloaded weights persist across re-provisions. Empty = image / OS default."
              >
                <Input
                  value={hfHome}
                  onChange={(e) => setHfHome(e.target.value)}
                  placeholder="/share/huggingface"
                  className="font-mono text-xs"
                />
              </Field>
            )}

            {(isVm || cloudMulti) && (
              <Field
                label="HF token (optional)"
                hint="For gated / private models — exported as HF_TOKEN. Pick a global secret (referenced, resolved at run-time — rotate it in Secrets) or paste a token."
              >
                <div className="space-y-2">
                  <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
                    {(["secret", "paste"] as const).map((src) => (
                      <button
                        key={src}
                        type="button"
                        onClick={() => setHfTokenSource(src)}
                        className={cn(
                          "rounded px-2.5 py-1 transition-colors",
                          hfTokenSource === src
                            ? "bg-primary text-primary-foreground"
                            : "text-muted-foreground hover:text-foreground",
                        )}
                      >
                        {src === "secret" ? "Global secret" : "Paste a token"}
                      </button>
                    ))}
                  </div>
                  {hfTokenSource === "secret" ? (
                    secretKeys.length > 0 ? (
                      <Select value={hfTokenSecret} onValueChange={setHfTokenSecret}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select a secret (e.g. HF_TOKEN)" />
                        </SelectTrigger>
                        <SelectContent>
                          {secretKeys.map((k) => (
                            <SelectItem key={k} value={k}>
                              {k}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    ) : (
                      <p className="text-xs text-muted-foreground">
                        No global secrets yet. Add one under{" "}
                        <a href="/admin/secrets" className="underline underline-offset-2 hover:text-foreground">
                          Secrets
                        </a>{" "}
                        (e.g. <span className="font-mono">HF_TOKEN</span>), or switch to{" "}
                        <span className="font-medium">Paste a token</span>.
                      </p>
                    )
                  ) : (
                    <Input
                      type="password"
                      value={hfToken}
                      onChange={(e) => setHfToken(e.target.value)}
                      placeholder="hf_…"
                      autoComplete="off"
                      className="font-mono text-xs"
                    />
                  )}
                </div>
              </Field>
            )}

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
                  label={mode === "proxy" ? "Model" : "Models"}
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
                  {mode !== "proxy" && (
                    <button
                      type="button"
                      onClick={() => setMembers((arr) => [...arr, { model: "", tp: 1, pp: 1, extra_args: "", gpus: "" }])}
                      className="mt-2 text-xs text-primary hover:underline"
                    >
                      + Add model
                    </button>
                  )}
                </Field>

                {oversubscribed && (
                  <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
                    <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                    <span>
                      Models need {gpusUsed} GPUs but the fleet has {fleetGpus} — they won&apos;t all stay
                      resident. Extra models are swapped in on demand via vLLM sleep/wake (first
                      request to a sleeping model waits for the swap).
                    </span>
                  </div>
                )}

                {mode !== "proxy" && (
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
                )}
              </div>
            )}
          </div>
        </Section>

        <Section
          title="Visibility"
          description="Who can see this endpoint. You can change it later from the endpoint page or the list menu."
        >
          <label className="flex cursor-pointer items-start gap-3">
            <Checkbox
              checked={isPublic}
              onCheckedChange={(v) => setIsPublic(v === true)}
              className="mt-0.5"
            />
            <div className="flex-1">
              <div className="text-sm font-medium">Make public (read-only)</div>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Off (default) keeps the endpoint private — only you and admins can
                see it. On shares it read-only with every logged-in user: they can
                view the overview, workers, and metrics but can&apos;t edit, delete,
                or run inference.
              </p>
            </div>
          </label>
        </Section>

        <Section title="Engine" description="Scaling behaviour, vLLM args, and metrics.">
          <div className="space-y-5">
            {(mode === "multi" || mode === "proxy") && isVm && (
              <p className="text-xs text-muted-foreground">
                {mode === "proxy"
                  ? "VM proxy endpoints are always-on; the gateway forwards each request straight to the model over a tunnel — no queue, no sleep/wake, no scale-to-zero."
                  : "VM multi-model fleets are always-on (no scale-to-zero); per-model vLLM args are set per model above. Models are evicted via sleep/wake, not torn down."}
              </p>
            )}
            {!isVm && (
            <>
            <Field
              label="Idle timeout (s)"
              hint={cloudMulti
                ? "After this many idle seconds the whole fleet pod is deleted (re-provisioned on the next request). 0 = always-on."
                : "Worker is torn down after this many seconds with no traffic. 0 keeps the worker on forever."}
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

            {/* Single-model global vLLM args. A multi-model fleet sets vLLM args
                per model in the Models section instead. */}
            {mode === "single" && (
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
            )}
            </>
            )}

            {(isVm || cloudMulti) && (
              <div className="border-t border-border pt-4">
                <Field
                  label="Environment variables"
                  hint={
                    isVm
                      ? "Applied to every vLLM process on the VM. One KEY=value per line (export / mkdir lines are fine — absolute-path values are auto-created). CUDA_VISIBLE_DEVICES is set per model automatically."
                      : "Applied to every vLLM process in the pod fleet. One KEY=value per line. The pod is ephemeral — point caches at a mounted volume if you need them to survive a re-provision. CUDA_VISIBLE_DEVICES is set per model automatically."
                  }
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

      <FormFooter
        error={submitError}
        hint={
          isVm && vdInvalid ? "Fix the GPU IDs pin to enable deploy."
          : !isVm && explicitlyUnavailable ? "The selected GPU has no capacity — pick another type or region."
          : !isVm && (idleInvalid || diskInvalid || volumeInvalid) ? "Fix the pod sizing / idle timeout values."
          : mode === "single" && advancedInvalid ? "Fix the vLLM engine args to enable deploy."
          : `${isVm ? "Bare metal" : "Cloud"} · ${mode === "single" ? "single model" : mode}`
        }
      >
        <Button variant="ghost" onClick={() => router.push("/serverless")} disabled={pending}>
          Cancel
        </Button>
        <Button
          onClick={submit}
          disabled={
            pending ||
            (isVm && vdInvalid) ||
            // Cloud (single OR multi-fleet): pod sizing + idle + availability.
            (!isVm && (idleInvalid || diskInvalid || volumeInvalid || explicitlyUnavailable)) ||
            // Single-model only has global vLLM engine args to validate.
            (mode === "single" && advancedInvalid)
          }
        >
          {pending && <Loader2 className="h-4 w-4 animate-spin" />}
          Create endpoint
        </Button>
      </FormFooter>

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
    </FormShell>
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
    // data-form-section feeds the FormShell scrollspy rail; scroll-mt keeps the
    // heading visible after a rail jump.
    <section data-form-section={title} className="scroll-mt-6 rounded-lg border border-border bg-card p-5">
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
