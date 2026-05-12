"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import yaml from "js-yaml";
import {
  AlertTriangle,
  Bookmark,
  Box,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Cpu,
  FileCode2,
  FlaskConical,
  Gauge,
  Info,
  Loader2,
  Package,
  Server,
  ShieldAlert,
  Sparkles,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
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
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { AvailabilityBadge } from "@/components/availability-badge";
import { useGpuAvailability } from "@/lib/use-gpu-availability";
import { gateway } from "@/lib/gateway";
import type { BenchmarkTemplate } from "@/lib/types";
import { cn } from "@/lib/utils";

const GPU_OPTIONS = [
  { id: "NVIDIA RTX A4000", label: "RTX A4000", hint: "16 GB · cheap baseline" },
  { id: "NVIDIA RTX A5000", label: "RTX A5000", hint: "24 GB" },
  { id: "NVIDIA RTX A6000", label: "RTX A6000", hint: "48 GB" },
  { id: "NVIDIA GeForce RTX 4090", label: "RTX 4090", hint: "24 GB · consumer" },
  { id: "NVIDIA L40", label: "L40", hint: "48 GB" },
  { id: "NVIDIA L40S", label: "L40S", hint: "48 GB · faster L40" },
  { id: "NVIDIA A100 80GB PCIe", label: "A100 80GB", hint: "datacenter" },
  { id: "NVIDIA H100 80GB HBM3", label: "H100 80GB", hint: "fastest" },
];

// Container image presets for the RunPod pod. CUDA version matters because
// flashinfer's Hopper kernels (used by Qwen3-Next + GDN linear attention)
// need PTX intrinsics that only exist in CUDA 12.6+ — CUDA 12.4 will fail
// to JIT-compile gdn_prefill_sm90 mid-inference.
const DEFAULT_CONTAINER_IMAGE =
  "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04";
const CONTAINER_IMAGE_OPTIONS = [
  {
    id: DEFAULT_CONTAINER_IMAGE,
    label: "CUDA 12.4 · pytorch 2.4",
    hint: "default",
  },
  {
    id: "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    label: "CUDA 12.8 · torch 2.8",
    hint: "Qwen3-Next / flashinfer GDN · official RunPod template",
  },
];
const CUSTOM_IMAGE_SENTINEL = "__custom__";

// CUDA toolkit version → minimum NVIDIA driver version (Linux)
const CUDA_MIN_DRIVER: Record<string, string> = {
  "11.0": "450.80", "11.1": "455.23", "11.2": "460.27", "11.3": "465.19",
  "11.4": "470.57", "11.5": "495.29", "11.6": "510.39", "11.7": "515.43",
  "11.8": "520.61",
  "12.0": "525.60", "12.1": "530.30", "12.2": "535.54", "12.3": "545.23",
  "12.4": "550.54", "12.5": "555.42", "12.6": "560.28", "12.7": "565.57",
  "12.8": "570.00",
};

// Extract CUDA major.minor from a container image tag.
// Handles: cuda12.4.1, cuda12.8, cu1281 (= 12.8.1), cu124 (= 12.4), etc.
function parseCudaFromImage(image: string): string | null {
  const m1 = image.match(/cuda[-_]?(\d+)[._](\d+)/i);
  if (m1) return `${m1[1]}.${m1[2]}`;
  const m2 = image.match(/\bcu(\d{4})\b/i);
  if (m2) return `${m2[1].slice(0, 2)}.${m2[1][2]}`;
  const m3 = image.match(/\bcu(\d{3})\b/i);
  if (m3) return `${m3[1].slice(0, 2)}.${m3[1][2]}`;
  return null;
}

// Pull the container image out of raw YAML text without a full parse.
function extractImageFromYaml(src: string): string | null {
  const m = src.match(/^\s+image:\s*["']?([^\s"'\n#]+)["']?\s*$/m);
  return m ? m[1] : null;
}

type FormState = {
  benchName: string;
  gpu_type: string;
  gpu_count: number;
  secure_cloud: boolean;
  disk_size: number;
  container_image: string;
  model_repo_id: string;
  // All vLLM engine args are strings so empty = "use vLLM default" — same
  // ergonomics as the serverless endpoint create form.
  tensor_parallel_size: string;
  data_parallel_size: string;
  max_model_len: string;
  gpu_memory_utilization: string;
  max_num_seqs: string;
  dtype: "auto" | "bfloat16" | "float16" | "float32";
  vllm_version: string;
  // Cmdline-style flags appended to vLLM. Parsed into snake_case serve: keys
  // at render time. e.g. "--enforce-eager --quantization awq"
  extra_args_raw: string;
  // Workload
  request_rate: string;
  // Workload — single-value when sweep_mode is off, CSV-derived arrays when on.
  sweep_mode: boolean;
  input_len: number;
  output_len: number;
  num_prompts: number;
  max_concurrency: number;
  // Comma-separated lists used in sweep mode. Free-form strings preserve typing.
  input_lens_csv: string;
  concurrencies_csv: string;
  hf_home: string;
};

const DEFAULTS: FormState = {
  benchName: "qwen-quick",
  gpu_type: "NVIDIA RTX A4000",
  gpu_count: 1,
  secure_cloud: false,
  disk_size: 80,
  container_image: DEFAULT_CONTAINER_IMAGE,
  model_repo_id: "Qwen/Qwen2.5-0.5B-Instruct",
  tensor_parallel_size: "",
  data_parallel_size: "",
  max_model_len: "",
  gpu_memory_utilization: "",
  max_num_seqs: "",
  dtype: "auto",
  vllm_version: "0.15.0",
  // Benchmark-default extras: prefix caching off (so cache hits don't skew
  // numbers). --disable-log-requests was removed in vLLM > 0.15 and now
  // causes the server to refuse to start, so it's no longer in the default.
  extra_args_raw: "--no-enable-prefix-caching",
  request_rate: "inf",
  sweep_mode: false,
  input_len: 256,
  output_len: 128,
  num_prompts: 50,
  max_concurrency: 4,
  input_lens_csv: "128, 512, 1024, 2048",
  concurrencies_csv: "10, 25, 50, 200",
  hf_home: "/workspace/hf_home",
};

function parseCsvInts(s: string): number[] {
  return s
    .split(/[,\s]+/)
    .map((x) => x.trim())
    .filter(Boolean)
    .map((x) => parseInt(x, 10))
    .filter((n) => Number.isFinite(n) && n > 0);
}

function modelToLocalDir(repo: string): string {
  const tail = repo.split("/").pop() || "model";
  const slug = tail.toLowerCase().replace(/\./g, "p").replace(/[^a-z0-9-]/g, "-");
  return `/workspace/models/${slug}`;
}

function renderBenchEntries(s: FormState): string {
  const inputs = s.sweep_mode ? parseCsvInts(s.input_lens_csv) : [s.input_len];
  const concs = s.sweep_mode ? parseCsvInts(s.concurrencies_csv) : [s.max_concurrency];
  const safeInputs = inputs.length ? inputs : [s.input_len];
  const safeConcs = concs.length ? concs : [s.max_concurrency];

  const rate = (s.request_rate || "inf").trim() || "inf";
  const lines: string[] = [];
  for (const inLen of safeInputs) {
    for (const c of safeConcs) {
      const extra = s.sweep_mode
        ? `, percentile_metrics: "ttft,tpot,itl,e2el"`
        : "";
      lines.push(
        `      - { endpoint: /v1/completions, dataset_name: random, ` +
          `random_input_len: ${inLen}, random_output_len: ${s.output_len}, ` +
          `num_prompts: ${s.num_prompts}, max_concurrency: ${c}, ` +
          `request_rate: ${rate}, ignore_eos: true${extra} }`,
      );
    }
  }
  return lines.join("\n");
}

type ServeKV = string | number | boolean;

/** Parse vllm-style cmdline flags into a serve-keys map. Translates kebab to
 * snake-case so `--enforce-eager --quantization awq --max-num-batched-tokens 8192`
 * becomes { enforce_eager: true, quantization: "awq", max_num_batched_tokens: 8192 }. */
function parseExtraArgs(raw: string): Record<string, ServeKV> {
  const out: Record<string, ServeKV> = {};
  const tokens = raw.trim().split(/\s+/).filter(Boolean);
  let i = 0;
  while (i < tokens.length) {
    const t = tokens[i];
    if (!t.startsWith("--")) { i++; continue; }
    const key = t.slice(2).replace(/-/g, "_");
    const next = tokens[i + 1];
    if (next !== undefined && !next.startsWith("--")) {
      const n = Number(next);
      out[key] = Number.isFinite(n) && next.trim() !== "" ? n : next;
      i += 2;
    } else {
      out[key] = true;
      i++;
    }
  }
  return out;
}

function renderServeBlock(s: FormState): string {
  // Build a single serve dict, structured fields first (so extras can override
  // explicitly if the user wants — last-write-wins matches cmdline semantics).
  const merged: Record<string, ServeKV> = {};

  const setIfNum = (k: string, v: string) => {
    const t = v.trim();
    if (!t) return;
    const n = Number(t);
    if (Number.isFinite(n)) merged[k] = n;
  };

  setIfNum("tensor_parallel_size", s.tensor_parallel_size);
  setIfNum("data_parallel_size", s.data_parallel_size);
  setIfNum("max_model_len", s.max_model_len);
  setIfNum("gpu_memory_utilization", s.gpu_memory_utilization);
  setIfNum("max_num_seqs", s.max_num_seqs);
  if (s.dtype !== "auto") merged["dtype"] = s.dtype;

  Object.assign(merged, parseExtraArgs(s.extra_args_raw));

  if (Object.keys(merged).length === 0) return "      {}";
  return Object.entries(merged)
    .map(([k, v]) => `      ${k}: ${typeof v === "string" ? v : v}`)
    .join("\n");
}

function totalRuns(s: FormState): number {
  if (!s.sweep_mode) return 1;
  const inputs = parseCsvInts(s.input_lens_csv);
  const concs = parseCsvInts(s.concurrencies_csv);
  return Math.max(1, inputs.length) * Math.max(1, concs.length);
}

function renderYaml(s: FormState): string {
  return `runpod:
  ssh_private_key: ""
  runpod_api_key: ""
  pod:
    name: "sgpu-${s.benchName.replace(/[^a-z0-9-]/gi, "-").toLowerCase()}"
    gpu_type: "${s.gpu_type}"
    gpu_count: ${s.gpu_count}
    instance_type: on_demand
    secure_cloud: ${s.secure_cloud}
  container:
    image: "${s.container_image || DEFAULT_CONTAINER_IMAGE}"
    disk_size: ${s.disk_size}
  storage:
    volume_size: ${s.disk_size}
    mount_path: "/workspace"
  ports:
    http: [8000]
    tcp: [22]
  env:
    HF_HOME: "${s.hf_home}"

remote:
  key_filename: ""
  uv:
    path: ~/.venv
    python_version: "3.11"
  dependencies:
    - vllm==${s.vllm_version || "0.15.0"}
    - huggingface_hub
    - hf_transfer

benchmark:
  - name: ${s.benchName}
    engine: vllm
    model:
      repo_id: "${s.model_repo_id}"
      local_dir: "${modelToLocalDir(s.model_repo_id)}"
    serve:
${renderServeBlock(s)}
    bench:
${renderBenchEntries(s)}
    results:
      save_result: true
      save_detailed: true
`;
}

// Serve keys that map 1:1 to a Form field. Everything else under
// benchmark[0].serve is flattened back into Extra args (raw cmdline form).
const FORM_SERVE_KEYS = new Set([
  "tensor_parallel_size",
  "data_parallel_size",
  "max_model_len",
  "gpu_memory_utilization",
  "max_num_seqs",
  "dtype",
]);
const FORM_DTYPES = new Set<FormState["dtype"]>([
  "auto",
  "bfloat16",
  "float16",
  "float32",
]);

type ParseYamlResult = {
  state: FormState;
  unknownKeys: string[];
  parseError: string | null;
};

/** Parse a benchmaq YAML config back into FormState. Anything the form
 * doesn't represent (extra env vars, multiple bench items, custom engine,
 * etc.) is collected into `unknownKeys` so we can warn the user that
 * round-tripping through Form mode will drop those keys. */
function parseYamlToForm(src: string, fallback: FormState): ParseYamlResult {
  let doc: unknown;
  try {
    doc = yaml.load(src);
  } catch (e) {
    return {
      state: fallback,
      unknownKeys: [],
      parseError: e instanceof Error ? e.message : String(e),
    };
  }
  if (!doc || typeof doc !== "object") {
    return { state: fallback, unknownKeys: [], parseError: "empty config" };
  }
  const d = doc as Record<string, unknown>;
  const next = { ...fallback };
  const unknown: string[] = [];

  // ---- runpod.pod
  const pod = ((d.runpod as Record<string, unknown> | undefined)?.pod ??
    {}) as Record<string, unknown>;
  if (typeof pod.gpu_type === "string") next.gpu_type = pod.gpu_type;
  if (typeof pod.gpu_count === "number") next.gpu_count = pod.gpu_count;
  if (typeof pod.secure_cloud === "boolean")
    next.secure_cloud = pod.secure_cloud;

  // ---- runpod.container
  const container = ((d.runpod as Record<string, unknown> | undefined)
    ?.container ?? {}) as Record<string, unknown>;
  if (typeof container.image === "string") next.container_image = container.image;
  if (typeof container.disk_size === "number") next.disk_size = container.disk_size;

  // ---- runpod.env
  const env = ((d.runpod as Record<string, unknown> | undefined)?.env ?? {}) as
    Record<string, unknown>;
  if (typeof env.HF_HOME === "string") next.hf_home = env.HF_HOME;
  for (const k of Object.keys(env)) {
    if (k !== "HF_HOME") unknown.push(`runpod.env.${k}`);
  }

  // ---- remote.dependencies — pick the vllm pin if present.
  const deps = ((d.remote as Record<string, unknown> | undefined)
    ?.dependencies ?? []) as unknown[];
  if (Array.isArray(deps)) {
    for (const dep of deps) {
      if (typeof dep === "string") {
        const m = dep.match(/^vllm==(.+)$/);
        if (m) {
          next.vllm_version = m[1];
          break;
        }
      }
    }
  }

  // ---- benchmark[]
  const benches = Array.isArray(d.benchmark) ? (d.benchmark as unknown[]) : [];
  if (benches.length > 1) {
    unknown.push(
      `benchmark[1..${benches.length - 1}] (Form mode only edits benchmark[0])`,
    );
  }
  const first = (benches[0] ?? {}) as Record<string, unknown>;

  if (typeof first.name === "string") next.benchName = first.name;
  if (typeof first.engine === "string" && first.engine !== "vllm") {
    unknown.push(`benchmark[0].engine = ${first.engine} (Form mode assumes vllm)`);
  }
  const model = (first.model ?? {}) as Record<string, unknown>;
  if (typeof model.repo_id === "string") next.model_repo_id = model.repo_id;

  // ---- benchmark[0].serve — split into form-mapped keys + Extra args.
  const serve = (first.serve ?? {}) as Record<string, unknown>;
  const extras: string[] = [];
  for (const [k, v] of Object.entries(serve)) {
    if (!FORM_SERVE_KEYS.has(k)) {
      // Re-render as a cmdline flag.
      const flag = `--${k.replace(/_/g, "-")}`;
      if (v === true) extras.push(flag);
      else if (v === false) {
        // false-valued booleans don't have a clean cmdline form; surface
        // it as an unknown key rather than silently dropping it.
        unknown.push(`benchmark[0].serve.${k} = false`);
      } else if (typeof v === "number" || typeof v === "string") {
        extras.push(`${flag} ${v}`);
      } else {
        unknown.push(`benchmark[0].serve.${k}`);
      }
      continue;
    }
    if (k === "dtype" && typeof v === "string" && FORM_DTYPES.has(v as FormState["dtype"])) {
      next.dtype = v as FormState["dtype"];
    } else if (typeof v === "number") {
      // The form stores numeric serve args as strings so empty = "use vLLM default".
      (next as unknown as Record<string, string>)[k] = String(v);
    }
  }
  next.extra_args_raw = extras.join(" ");

  // ---- benchmark[0].bench[] — sweep detection + workload fields.
  const benchRows = Array.isArray(first.bench)
    ? (first.bench as Record<string, unknown>[])
    : [];
  if (benchRows.length > 0) {
    const inputLens = new Set<number>();
    const concs = new Set<number>();
    let outLen: number | undefined;
    let nPrompts: number | undefined;
    let rate: string | undefined;
    for (const row of benchRows) {
      if (typeof row.random_input_len === "number") inputLens.add(row.random_input_len);
      if (typeof row.max_concurrency === "number") concs.add(row.max_concurrency);
      if (typeof row.random_output_len === "number" && outLen === undefined)
        outLen = row.random_output_len;
      if (typeof row.num_prompts === "number" && nPrompts === undefined)
        nPrompts = row.num_prompts;
      if (row.request_rate !== undefined && rate === undefined)
        rate = String(row.request_rate);
    }
    if (outLen !== undefined) next.output_len = outLen;
    if (nPrompts !== undefined) next.num_prompts = nPrompts;
    if (rate !== undefined) next.request_rate = rate;

    const isSweep =
      benchRows.length > 1 || inputLens.size > 1 || concs.size > 1;
    next.sweep_mode = isSweep;
    if (isSweep) {
      next.input_lens_csv = [...inputLens]
        .sort((a, b) => a - b)
        .join(", ");
      next.concurrencies_csv = [...concs]
        .sort((a, b) => a - b)
        .join(", ");
    } else {
      const inLen = [...inputLens][0];
      const c = [...concs][0];
      if (typeof inLen === "number") next.input_len = inLen;
      if (typeof c === "number") next.max_concurrency = c;
    }
  }

  return { state: next, unknownKeys: unknown, parseError: null };
}

export function BenchmarkForm({
  initialName,
  initialYaml,
}: {
  initialName?: string;
  initialYaml?: string;
} = {}) {
  const router = useRouter();
  // Duplicate flow: start in YAML mode with the source config pre-filled so
  // the round-trip is exact. Switching back to Form mode parses the YAML
  // and back-fills the form (lossy for keys the form doesn't represent).
  const [mode, setMode] = useState<"form" | "yaml">(initialYaml ? "yaml" : "form");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const [form, setForm] = useState<FormState>(DEFAULTS);
  const [name, setName] = useState(initialName ?? DEFAULTS.benchName);
  const availability = useGpuAvailability(
    form.gpu_type,
    form.gpu_count,
    mode === "form",
    form.secure_cloud ? "SECURE" : "COMMUNITY",
  );
  const [yamlBuf, setYamlBuf] = useState<string>(initialYaml ?? renderYaml(DEFAULTS));
  const formYaml = useMemo(
    () => renderYaml({ ...form, benchName: name || "untitled" }),
    [form, name],
  );

  const [templates, setTemplates] = useState<BenchmarkTemplate[]>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>("");
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveName, setSaveName] = useState("");

  useEffect(() => {
    gateway.listBenchmarkTemplates().then(setTemplates).catch(() => {});
  }, []);

  useEffect(() => {
    if (mode === "form") setYamlBuf(formYaml);
  }, [mode, formYaml]);

  function field<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function loadTemplate(id: string) {
    setSelectedTemplateId(id);
    if (!id) return;
    const t = templates.find((x) => x.id === id);
    if (!t) return;
    setYamlBuf(t.config_yaml);
    setMode("yaml");
    toast.success(`Loaded template: ${t.name}`, { duration: 3000 });
  }

  async function deleteTemplate(id: string) {
    try {
      await gateway.deleteBenchmarkTemplate(id);
      setTemplates((prev) => prev.filter((t) => t.id !== id));
      if (selectedTemplateId === id) setSelectedTemplateId("");
      toast.success("Template deleted", { duration: 3000 });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e)), { duration: 5000 };
    }
  }

  async function handleSaveTemplate() {
    if (!saveName.trim()) {
      toast.error("Template needs a name", { duration: 5000 });
      return;
    }
    const yamlToSave = mode === "form" ? formYaml : yamlBuf;
    try {
      const t = await gateway.createBenchmarkTemplate(saveName.trim(), yamlToSave);
      setTemplates((prev) => [t, ...prev]);
      setSaveOpen(false);
      setSaveName("");
      toast.success(`Saved template: ${t.name}`, { duration: 3000 });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e)), { duration: 5000 };
    }
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);
    if (!name.trim()) {
      setSubmitError("Name is required.");
      return;
    }
    const config_yaml = mode === "form" ? formYaml : yamlBuf;
    setSubmitting(true);
    try {
      const created = await gateway.createBenchmark({
        name: name.trim(),
        config_yaml,
      });
      toast.success(`Created ${created.id}`, { duration: 4000 });
      router.push(`/benchmark/${encodeURIComponent(created.id)}`);
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="space-y-6">
      {/* Header — plain, no gradient. */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Create benchmark</h1>
        <p className="mt-1 max-w-xl text-sm text-muted-foreground">
          Spin up a RunPod GPU, run <span className="font-mono text-xs">benchmaq</span>{" "}
          against vLLM, and stream the logs back here. Save as a template if
          you&apos;ll re-run it.
        </p>
      </div>

      {/* Templates */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileCode2 className="h-4 w-4 text-muted-foreground" />
              <CardTitle className="text-sm">Templates</CardTitle>
              <Badge variant="secondary" className="ml-1 text-[10px]">
                {templates.length}
              </Badge>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setSaveOpen(true)}
            >
              <Bookmark className="h-4 w-4" />
              Save current
            </Button>
          </div>
          <CardDescription className="text-xs">
            Reuse a saved configuration instead of filling everything from scratch.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Select
              value={selectedTemplateId || "__none__"}
              onValueChange={(v) =>
                loadTemplate(v === "__none__" ? "" : v)
              }
            >
              <SelectTrigger className="flex-1">
                <SelectValue
                  placeholder={
                    templates.length ? "Pick a template…" : "No templates yet"
                  }
                />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">
                  — None (use form below) —
                </SelectItem>
                {templates.map((t) => (
                  <SelectItem key={t.id} value={t.id}>
                    {t.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedTemplateId && (
              <Button
                type="button"
                variant="outline"
                size="icon"
                onClick={() => deleteTemplate(selectedTemplateId)}
                className="text-destructive"
                title="Delete this template"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Name */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-2">
            <Label htmlFor="benchName" className="text-sm font-medium">
              Benchmark name
            </Label>
            <Input
              id="benchName"
              placeholder="qwen-quick"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                field("benchName", e.target.value);
              }}
              autoFocus
              className="font-mono"
            />
            <p className="text-xs text-muted-foreground">
              Shows up in the list view and as the pod name on RunPod.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Form / YAML toggle */}
      <Tabs
        value={mode}
        onValueChange={(v) => {
          const next = v as "form" | "yaml";
          if (next === "form" && mode === "yaml") {
            // Parse the YAML buffer back into the form so edits made in YAML
            // mode aren't lost when flipping back.
            const parsed = parseYamlToForm(yamlBuf, form);
            if (parsed.parseError) {
              toast.error(`Can't parse YAML: ${parsed.parseError}`, { duration: 5000 });
              return;
            }
            setForm(parsed.state);
            setName(parsed.state.benchName);
            if (parsed.unknownKeys.length > 0) {
              toast.warning(
                `Form mode can't represent: ${parsed.unknownKeys.join(", ")}. ` +
                  `These will be dropped if you submit from Form.`,
                { duration: 8000 },
              );
            }
          }
          setMode(next);
        }}
      >
        <div className="flex items-center justify-between">
          <TabsList>
            <TabsTrigger value="form">
              <Sparkles className="h-3.5 w-3.5" />
              Form
            </TabsTrigger>
            <TabsTrigger value="yaml">
              <FileCode2 className="h-3.5 w-3.5" />
              YAML
            </TabsTrigger>
          </TabsList>
          <span className="text-xs text-muted-foreground">
            {mode === "form"
              ? "Most common knobs + sweeps. YAML for multi-engine configs or per-row overrides."
              : "Edit raw config. Switching back to Form re-parses your edits (keys the form can't represent will be dropped)."}
          </span>
        </div>

        <TabsContent value="form" className="mt-4 space-y-6">
          {/* Pod */}
          <SectionCard
            icon={<Server className="h-4 w-4" />}
            title="Pod"
            description="GPU and disk for the RunPod instance benchmaq spawns."
            action={
              <AvailabilityBadge
                state={availability}
                count={form.gpu_count}
              />
            }
          >
            <Grid>
              <FieldWrap
                label="GPU type"
                hint="Pick what fits your model in VRAM."
                wide
              >
                <Select
                  value={form.gpu_type}
                  onValueChange={(v) => field("gpu_type", v)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {GPU_OPTIONS.map((g) => (
                      <SelectItem key={g.id} value={g.id}>
                        <div className="flex w-full items-center justify-between gap-3">
                          <span>{g.label}</span>
                          <span className="text-xs text-muted-foreground">
                            {g.hint}
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FieldWrap>
              <FieldWrap label="GPU count" hint="Set > 1 for tensor parallelism.">
                <Input
                  type="number"
                  min={1}
                  max={8}
                  value={form.gpu_count}
                  onChange={(e) =>
                    field("gpu_count", parseInt(e.target.value || "1", 10))
                  }
                />
              </FieldWrap>
              <FieldWrap label="Disk (GB)" hint="Container + volume. Big models need more.">
                <Input
                  type="number"
                  min={20}
                  value={form.disk_size}
                  onChange={(e) =>
                    field("disk_size", parseInt(e.target.value || "80", 10))
                  }
                />
              </FieldWrap>
              <FieldWrap
                label="Cloud type"
                hint="Community = cheaper, sometimes flaky."
                wide
              >
                <Select
                  value={form.secure_cloud ? "secure" : "community"}
                  onValueChange={(v) =>
                    field("secure_cloud", v === "secure")
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="community">Community</SelectItem>
                    <SelectItem value="secure">Secure</SelectItem>
                  </SelectContent>
                </Select>
              </FieldWrap>
            </Grid>
          </SectionCard>

          {/* Container image — picks the CUDA / pytorch baseline on the pod. */}
          <SectionCard
            icon={<Box className="h-4 w-4" />}
            title="Container"
            description="Base image the RunPod pod boots from. CUDA version must match what your model needs."
          >
            <ContainerImagePicker
              value={form.container_image}
              onChange={(v) => field("container_image", v)}
            />
          </SectionCard>

          {/* Engine runtime — what gets installed on the pod */}
          <SectionCard
            icon={<Package className="h-4 w-4" />}
            title="Engine"
            description="Pinned vLLM version installed on the pod via uv pip."
          >
            <Grid>
              <FieldWrap
                label="vLLM version"
                hint="Newer versions may drop CLI flags (e.g. --disable-log-requests was removed after 0.15)."
                wide
              >
                <Input
                  className="font-mono"
                  value={form.vllm_version}
                  onChange={(e) => field("vllm_version", e.target.value)}
                  placeholder="0.15.0"
                />
              </FieldWrap>
            </Grid>
          </SectionCard>

          {/* Model + Serve */}
          <SectionCard
            icon={<Cpu className="h-4 w-4" />}
            title="Model"
            description="The model to serve. Engine knobs in the advanced section below."
          >
            <FieldWrap
              label="HuggingFace repo"
              hint="Anything you can pip-load. Gated models use HF_TOKEN from gateway env."
            >
              <Input
                className="font-mono"
                value={form.model_repo_id}
                onChange={(e) => field("model_repo_id", e.target.value)}
                placeholder="Qwen/Qwen2.5-0.5B-Instruct"
              />
            </FieldWrap>

            <AdvancedVllmArgs form={form} setField={field} />
          </SectionCard>

          {/* Bench */}
          <SectionCard
            icon={<Gauge className="h-4 w-4" />}
            title="Workload"
            description="What benchmaq fires at the engine. Use the Sweep pill →  to cross-product input length × concurrency."
            action={
              <SweepToggle
                on={form.sweep_mode}
                onChange={(v) => field("sweep_mode", v)}
                runs={totalRuns(form)}
              />
            }
          >
            {form.sweep_mode ? (
              <Grid>
                <FieldWrap
                  label="Input lengths"
                  hint="Comma-separated list of token counts. Each value × each concurrency = one bench run."
                  wide
                >
                  <Input
                    className="font-mono"
                    placeholder="128, 512, 1024, 2048"
                    value={form.input_lens_csv}
                    onChange={(e) => field("input_lens_csv", e.target.value)}
                  />
                  <SweepChips values={parseCsvInts(form.input_lens_csv)} suffix="tok" />
                </FieldWrap>
                <FieldWrap
                  label="Concurrencies"
                  hint="In-flight requests per run. Sweep this to find the throughput knee."
                  wide
                >
                  <Input
                    className="font-mono"
                    placeholder="10, 25, 50, 200"
                    value={form.concurrencies_csv}
                    onChange={(e) => field("concurrencies_csv", e.target.value)}
                  />
                  <SweepChips values={parseCsvInts(form.concurrencies_csv)} />
                </FieldWrap>
                <FieldWrap label="Output length" hint="Same for every run.">
                  <Input
                    type="number"
                    min={1}
                    value={form.output_len}
                    onChange={(e) =>
                      field("output_len", parseInt(e.target.value || "128", 10))
                    }
                  />
                </FieldWrap>
                <FieldWrap label="Num prompts" hint="Total requests per run.">
                  <Input
                    type="number"
                    min={1}
                    value={form.num_prompts}
                    onChange={(e) =>
                      field("num_prompts", parseInt(e.target.value || "50", 10))
                    }
                  />
                </FieldWrap>
                <FieldWrap
                  label="Request rate"
                  hint='"inf" = blast at max — what you usually want. Or set a number to simulate a fixed QPS.'
                >
                  <Input
                    className="font-mono"
                    placeholder="inf"
                    value={form.request_rate}
                    onChange={(e) => field("request_rate", e.target.value)}
                  />
                </FieldWrap>
              </Grid>
            ) : (
              <Grid>
                <FieldWrap label="Input length" hint="Random tokens per prompt.">
                  <Input
                    type="number"
                    min={1}
                    value={form.input_len}
                    onChange={(e) =>
                      field("input_len", parseInt(e.target.value || "256", 10))
                    }
                  />
                </FieldWrap>
                <FieldWrap label="Output length" hint="Tokens to generate per request.">
                  <Input
                    type="number"
                    min={1}
                    value={form.output_len}
                    onChange={(e) =>
                      field("output_len", parseInt(e.target.value || "128", 10))
                    }
                  />
                </FieldWrap>
                <FieldWrap label="Num prompts" hint="Total requests in this run.">
                  <Input
                    type="number"
                    min={1}
                    value={form.num_prompts}
                    onChange={(e) =>
                      field("num_prompts", parseInt(e.target.value || "50", 10))
                    }
                  />
                </FieldWrap>
                <FieldWrap
                  label="Max concurrency"
                  hint="In-flight requests. Tune for throughput."
                >
                  <Input
                    type="number"
                    min={1}
                    value={form.max_concurrency}
                    onChange={(e) =>
                      field("max_concurrency", parseInt(e.target.value || "4", 10))
                    }
                  />
                </FieldWrap>
                <FieldWrap
                  label="Request rate"
                  hint='"inf" = no rate limit. Set a number for fixed QPS.'
                  wide
                >
                  <Input
                    className="font-mono"
                    placeholder="inf"
                    value={form.request_rate}
                    onChange={(e) => field("request_rate", e.target.value)}
                  />
                </FieldWrap>
              </Grid>
            )}
          </SectionCard>

          {/* YAML preview — plain code block, not a terminal. Capped height so
              long configs don't bleed under the sticky action bar. */}
          <details className="group rounded-lg border border-border">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-4 py-3 text-sm font-medium hover:bg-muted/40 [&::-webkit-details-marker]:hidden">
              <div className="flex items-center gap-2">
                <ChevronRight className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-90" />
                <FileCode2 className="h-4 w-4 text-muted-foreground" />
                YAML preview
                <Badge variant="secondary" className="text-[10px]">
                  read-only
                </Badge>
              </div>
              <Info className="h-3.5 w-3.5 text-muted-foreground" />
            </summary>
            <pre className="max-h-96 overflow-auto rounded-b-lg border-t border-border bg-muted/40 px-4 py-3 font-mono text-xs leading-relaxed text-foreground">
              {formYaml}
            </pre>
          </details>
        </TabsContent>

        <TabsContent value="yaml" className="mt-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Raw YAML</CardTitle>
              <CardDescription className="text-xs">
                Full benchmaq runpod-mode config. Sweeps via{" "}
                <span className="font-mono">benchmark[]</span> array; multiple
                bench items run on the same pod.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                rows={28}
                spellCheck={false}
                value={yamlBuf}
                onChange={(e) => setYamlBuf(e.target.value)}
                className="rounded-md border border-border bg-muted/40 font-mono text-xs leading-relaxed text-foreground focus-visible:ring-foreground/30"
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* CUDA pre-flight check — shown in both form and YAML mode. */}
      <CudaPreflightPanel
        image={
          mode === "form"
            ? form.container_image
            : (extractImageFromYaml(yamlBuf) ?? form.container_image)
        }
      />

      {/* Action bar — plain, sits at the bottom of the form (not floating). */}
      <div className="mt-6 flex items-center justify-between gap-3 border-t border-border pt-4">
        <div className="text-xs text-muted-foreground">
          A new RunPod pod will be created and torn down automatically.
        </div>
        <div className="flex items-center gap-3">
          {submitError && (
            <p className="text-sm text-destructive">{submitError}</p>
          )}
          <Button
            type="button"
            variant="outline"
            onClick={() => router.push("/benchmark")}
          >
            Cancel
          </Button>
          <Button type="submit" disabled={submitting} className="min-w-36">
            {submitting ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Creating…
              </>
            ) : (
              <>
                <FlaskConical className="h-4 w-4" />
                Create benchmark
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Save-as-template */}
      <Dialog open={saveOpen} onOpenChange={setSaveOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Save as template</DialogTitle>
            <DialogDescription>
              Saves the current{" "}
              {mode === "form" ? "form values" : "YAML"} for re-use. Templates
              are scoped to your account.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Label htmlFor="tplName">Template name</Label>
            <Input
              id="tplName"
              autoFocus
              placeholder="qwen-l40s baseline"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleSaveTemplate();
                }
              }}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setSaveOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleSaveTemplate}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </form>
  );
}

function CudaPreflightPanel({ image }: { image: string }) {
  const cuda = parseCudaFromImage(image);
  if (!cuda) return null;

  const minDriver = CUDA_MIN_DRIVER[cuda];
  const [major, minor] = cuda.split(".").map(Number);

  type Status = "ok" | "warn" | "risk";
  let status: Status;
  let msg: string;

  if (major > 12 || (major === 12 && minor >= 7)) {
    status = "risk";
    msg =
      "RunPod community nodes rarely have this driver. You may get assigned a node that rejects the container — switch to Secure cloud or use a CUDA 12.4 image.";
  } else if (major === 12 && minor >= 5) {
    status = "warn";
    msg =
      "CUDA 12.5–12.6 nodes are less common on RunPod community cloud. If you hit a mismatch, switch to Secure cloud or a CUDA 12.4 image.";
  } else {
    status = "ok";
    msg = "Driver requirement is widely available on RunPod community and secure nodes.";
  }

  const rowCls = cn(
    "rounded-lg border px-4 py-3",
    status === "ok" && "border-border bg-muted/20",
    status === "warn" && "border-yellow-500/30 bg-yellow-500/5",
    status === "risk" && "border-destructive/30 bg-destructive/5",
  );

  return (
    <div className={rowCls}>
      <div className="flex items-start gap-3">
        {status === "ok" && (
          <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-green-500" />
        )}
        {status === "warn" && (
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-yellow-500" />
        )}
        {status === "risk" && (
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
        )}
        <div className="min-w-0 space-y-1">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-0.5 text-sm font-medium">
            <span>
              Container CUDA:{" "}
              <span className="font-mono">{cuda}</span>
            </span>
            {minDriver && (
              <span>
                Requires driver:{" "}
                <span className="font-mono">≥ {minDriver}</span>
              </span>
            )}
          </div>
          <p className="text-xs text-muted-foreground">{msg}</p>
          {status !== "ok" && (
            <p className="text-xs text-muted-foreground">
              Unlike RunPod&apos;s own UI (which filters by compatible hosts),{" "}
              <span className="font-mono">benchmaq</span> uses{" "}
              <span className="font-mono">runpodctl</span> which does not send{" "}
              <span className="font-mono">allowedCudaVersions</span> — any
              available node may be assigned regardless of its driver.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function SectionCard({
  icon,
  title,
  description,
  action,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="pb-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-muted text-muted-foreground">
              {icon}
            </div>
            <CardTitle className="text-base">{title}</CardTitle>
          </div>
          {action}
        </div>
        {description && (
          <CardDescription className="text-xs">{description}</CardDescription>
        )}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function SweepToggle({
  on,
  onChange,
  runs,
}: {
  on: boolean;
  onChange: (v: boolean) => void;
  runs: number;
}) {
  return (
    <div className="flex items-center gap-2">
      {on && (
        <Badge variant="secondary" className="font-mono text-[10px]">
          {runs} run{runs === 1 ? "" : "s"}
        </Badge>
      )}
      <Label
        htmlFor="sweep-switch"
        className="cursor-pointer text-xs font-medium text-muted-foreground"
      >
        Sweep
      </Label>
      <Switch
        id="sweep-switch"
        checked={on}
        onCheckedChange={onChange}
        size="sm"
      />
    </div>
  );
}

function SweepChips({
  values,
  suffix,
}: {
  values: number[];
  suffix?: string;
}) {
  if (values.length === 0) {
    return (
      <p className="text-[11px] text-destructive">
        No values parsed — type comma-separated positive integers.
      </p>
    );
  }
  return (
    <div className="flex flex-wrap gap-1">
      {values.map((v, i) => (
        <span
          key={`${v}-${i}`}
          className="rounded-md bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
        >
          {v}
          {suffix ? ` ${suffix}` : ""}
        </span>
      ))}
    </div>
  );
}

function Grid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-1 gap-x-4 gap-y-5 sm:grid-cols-2 lg:grid-cols-4">
      {children}
    </div>
  );
}

function AdvancedVllmArgs({
  form,
  setField,
}: {
  form: FormState;
  setField: <K extends keyof FormState>(k: K, v: FormState[K]) => void;
}) {
  const [open, setOpen] = useState(false);
  const finalServe = renderServeBlock(form);
  return (
    <div className="mt-6 border-t border-border pt-4">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 text-left text-xs font-medium uppercase tracking-wide text-muted-foreground hover:text-foreground"
      >
        {open ? (
          <ChevronDown className="h-3.5 w-3.5" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" />
        )}
        Advanced options (vLLM engine args)
      </button>
      {open && (
        <div className="mt-4 space-y-4">
          <p className="text-xs text-muted-foreground">
            Defaults are sensible for most models. Override only when you know
            you need to. See{" "}
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
            <KebabField
              label="max-model-len"
              hint="Context window in tokens. Empty = model's default."
            >
              <Input
                type="text"
                inputMode="numeric"
                value={form.max_model_len}
                onChange={(e) => setField("max_model_len", e.target.value)}
                placeholder="e.g. 4096"
              />
            </KebabField>
            <KebabField
              label="gpu-memory-utilization"
              hint="Fraction of VRAM vLLM may use (0–1). Default 0.9."
            >
              <Input
                type="text"
                inputMode="decimal"
                value={form.gpu_memory_utilization}
                onChange={(e) =>
                  setField("gpu_memory_utilization", e.target.value)
                }
                placeholder="0.9"
              />
            </KebabField>
            <KebabField label="dtype" hint="Weight precision.">
              <Select
                value={form.dtype}
                onValueChange={(v) =>
                  setField("dtype", v as FormState["dtype"])
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">auto</SelectItem>
                  <SelectItem value="bfloat16">bfloat16</SelectItem>
                  <SelectItem value="float16">float16</SelectItem>
                  <SelectItem value="float32">float32</SelectItem>
                </SelectContent>
              </Select>
            </KebabField>
            <KebabField
              label="max-num-seqs"
              hint="Max concurrent sequences. Empty = vLLM default."
            >
              <Input
                type="text"
                inputMode="numeric"
                value={form.max_num_seqs}
                onChange={(e) => setField("max_num_seqs", e.target.value)}
                placeholder="e.g. 256"
              />
            </KebabField>
            <KebabField
              label="tensor-parallel-size"
              hint="Number of GPUs for tensor parallelism. Default 1."
            >
              <Input
                type="text"
                inputMode="numeric"
                value={form.tensor_parallel_size}
                onChange={(e) =>
                  setField("tensor_parallel_size", e.target.value)
                }
                placeholder="1"
              />
            </KebabField>
            <KebabField
              label="data-parallel-size"
              hint="Replicates the model. TP × DP must = GPU count. Default 1."
            >
              <Input
                type="text"
                inputMode="numeric"
                value={form.data_parallel_size}
                onChange={(e) =>
                  setField("data_parallel_size", e.target.value)
                }
                placeholder="1"
              />
            </KebabField>
          </div>
          <KebabField
            label="Extra args (raw)"
            hint="Cmdline-style flags appended to vLLM. Translated to serve config keys (--no-enable-prefix-caching → no_enable_prefix_caching: true). e.g. --enforce-eager --quantization awq"
          >
            <textarea
              value={form.extra_args_raw}
              onChange={(e) => setField("extra_args_raw", e.target.value)}
              placeholder="--enforce-eager"
              rows={2}
              spellCheck={false}
              className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
            />
          </KebabField>
          {finalServe.trim() && finalServe.trim() !== "{}" && (
            <div className="rounded-md bg-muted/50 px-3 py-2">
              <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                Final serve config
              </div>
              <pre className="overflow-x-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-foreground">
                {finalServe.replace(/^ {6}/gm, "")}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function KebabField({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <Label className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </Label>
      {children}
      {hint && (
        <p className="text-[11px] leading-snug text-muted-foreground">{hint}</p>
      )}
    </div>
  );
}

function ToggleRow({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={cn(
        "flex w-full items-start gap-3 rounded-md border p-3 text-left transition-colors",
        checked
          ? "border-foreground/60 bg-foreground/5"
          : "border-border bg-background hover:bg-muted/30",
      )}
    >
      <span
        className={cn(
          "mt-0.5 inline-flex h-4 w-7 shrink-0 items-center rounded-full transition-colors",
          checked ? "bg-foreground" : "bg-muted-foreground/30",
        )}
      >
        <span
          className={cn(
            "inline-block h-3 w-3 transform rounded-full bg-white transition-transform",
            checked ? "translate-x-3.5" : "translate-x-0.5",
          )}
        />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-xs font-medium">{label}</span>
        {hint && <span className="mt-0.5 block text-[11px] leading-snug text-muted-foreground">{hint}</span>}
      </span>
    </button>
  );
}

function FieldWrap({
  label,
  hint,
  wide,
  children,
}: {
  label: string;
  hint?: string;
  wide?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className={cn("space-y-1.5", wide ? "sm:col-span-2 lg:col-span-2" : "")}>
      <Label className="text-xs font-medium">{label}</Label>
      {children}
      {hint && <p className="text-[11px] leading-snug text-muted-foreground">{hint}</p>}
    </div>
  );
}

function ContainerImagePicker({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const isPreset = CONTAINER_IMAGE_OPTIONS.some((p) => p.id === value);
  return (
    <div className="space-y-3">
      <FieldWrap
        label="Image"
        hint="Qwen3-Next + vLLM ≥ 0.17 needs CUDA 12.6+ for the flashinfer GDN kernel. Stick with CUDA 12.4 for everything else."
        wide
      >
        <Select
          value={isPreset ? value : CUSTOM_IMAGE_SENTINEL}
          onValueChange={(v) => {
            if (v === CUSTOM_IMAGE_SENTINEL) {
              if (isPreset) onChange("");
            } else {
              onChange(v);
            }
          }}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {CONTAINER_IMAGE_OPTIONS.map((o) => (
              <SelectItem key={o.id} value={o.id}>
                <div className="flex w-full items-center justify-between gap-3">
                  <span>{o.label}</span>
                  <span className="text-xs text-muted-foreground">{o.hint}</span>
                </div>
              </SelectItem>
            ))}
            <SelectItem value={CUSTOM_IMAGE_SENTINEL}>Custom…</SelectItem>
          </SelectContent>
        </Select>
      </FieldWrap>
      {!isPreset && (
        <FieldWrap
          label="Custom image"
          hint="Full Docker reference, e.g. runpod/pytorch:2.8.0-py3.11-cuda12.8.1-devel-ubuntu22.04"
          wide
        >
          <Input
            className="font-mono"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder={DEFAULT_CONTAINER_IMAGE}
          />
        </FieldWrap>
      )}
    </div>
  );
}
