"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { AlertTriangle, ChevronDown, ChevronRight, Loader2 } from "lucide-react";
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
import { deployEndpoint } from "../actions";
import { AvailabilityBadge } from "@/components/availability-badge";
import { useGpuAvailability } from "@/lib/use-gpu-availability";

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

export function InferenceForm() {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const [framework, setFramework] = useState("vllm");
  const [name, setName] = useState(suggestName());
  const [model, setModel] = useState("");
  const [gpu, setGpu] = useState("RTX3090");
  const [gpuCount, setGpuCount] = useState<number>(1);
  const [idleInput, setIdleInput] = useState("");
  const [alwaysOn, setAlwaysOn] = useState(true);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [enableMetrics, setEnableMetrics] = useState(true);
  const [vllm, setVllm] = useState({ ...DEFAULT_VLLM_ARGS });
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

  const availability = useGpuAvailability(gpu, gpuCount, true);
  const explicitlyUnavailable =
    availability.status === "ok" && availability.data.available === false;

  const parsedIdle = Number.parseInt(idleInput, 10);
  const idleInvalid =
    !alwaysOn && (!Number.isFinite(parsedIdle) || parsedIdle < 1 || parsedIdle > 86400);

  function submit() {
    setSubmitError(null);
    if (!name.trim() || !model.trim()) {
      setSubmitError("Endpoint name and model name are required.");
      return;
    }
    if (idleInvalid) {
      setSubmitError(
        "Enter a positive idle timeout in seconds, or tick 'No idle timeout'.",
      );
      return;
    }
    if (advancedInvalid) {
      setSubmitError("Fix the invalid values in Advanced options.");
      return;
    }
    if (explicitlyUnavailable) {
      const reason =
        (availability.status === "ok" && availability.data.reason) ||
        `${gpu}×${gpuCount} isn't available on the active provider right now.`;
      setSubmitError(reason);
      return;
    }
    const vllmArgs = buildVllmArgs(vllm);
    startTransition(async () => {
      const res = await deployEndpoint({
        name: slugify(name),
        model: model.trim(),
        gpu,
        gpu_count: gpuCount,
        autoscaler: {
          max_containers: MAX_WORKERS,
          idle_timeout_s: alwaysOn ? 0 : parsedIdle,
        },
        vllm_args: vllmArgs,
        enable_metrics: enableMetrics,
      });
      if (!res.ok) {
        if (res.unavailable) {
          setUnavailableModal(res.unavailable);
        } else {
          setSubmitError(res.error);
        }
        return;
      }
      toast.success(`Endpoint ${res.app_id} created`, { duration: 4000 });
      router.push(`/serverless/${encodeURIComponent(res.app_id)}`);
    });
  }

  return (
    <div className="">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">Create inference endpoint</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Pick a framework and a model. The endpoint scales to zero when idle.
        </p>
      </div>

      <div className="space-y-5 rounded-xl border border-border bg-card p-6">
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

        <Field label="Endpoint name" required>
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="my-endpoint"
            className="bg-muted"
          />
        </Field>

        <Field label="Model" hint="Hugging Face repo (e.g. Qwen/Qwen2.5-7B-Instruct)" required>
          <Input
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="Qwen/Qwen2.5-7B-Instruct"
            className="bg-muted/50"
          />
        </Field>

        <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>
            Pick a GPU with enough VRAM for your model. vLLM will fail to load if the
            weights plus KV cache exceed GPU memory.
          </span>
        </div>

        <Field
          label="Idle timeout (s)"
          hint="Worker is torn down after this many seconds with no traffic."
        >
          <div className="flex flex-col gap-2">
            <Input
              type="text"
              inputMode="numeric"
              value={idleInput}
              onChange={(e) => {
                setIdleInput(e.target.value);
                if (e.target.value.trim() !== "") setAlwaysOn(false);
              }}
              placeholder={alwaysOn ? "Always-on (no timeout)" : "e.g. 300"}
              aria-invalid={idleInvalid}
            />
            <label className="flex items-center gap-2 text-sm text-muted-foreground">
              <Checkbox
                checked={alwaysOn}
                onCheckedChange={(v) => {
                  const next = v === true;
                  setAlwaysOn(next);
                  if (next) setIdleInput("");
                }}
              />
              <span>No idle timeout (keep worker up forever)</span>
            </label>
          </div>
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
                ready so requests aren't delayed.
              </p>
            </div>
          </label>
        </div>
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
          disabled={pending || idleInvalid || advancedInvalid || explicitlyUnavailable}
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
              ×{unavailableModal?.gpu_count}. The endpoint wasn't created so you
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

function slugify(s: string) {
  return s.toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "");
}

function suggestName() {
  const suffix = Math.floor(Math.random() * 9000 + 1000);
  return `endpoint-${suffix}`;
}
