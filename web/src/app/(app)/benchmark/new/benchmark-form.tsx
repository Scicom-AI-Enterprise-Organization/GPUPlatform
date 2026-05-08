"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Bookmark,
  ChevronDown,
  ChevronRight,
  Cpu,
  FileCode2,
  FlaskConical,
  Gauge,
  Info,
  Loader2,
  Server,
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

type FormState = {
  benchName: string;
  gpu_type: string;
  gpu_count: number;
  secure_cloud: boolean;
  disk_size: number;
  model_repo_id: string;
  // All vLLM engine args are strings so empty = "use vLLM default" — same
  // ergonomics as the serverless endpoint create form.
  tensor_parallel_size: string;
  data_parallel_size: string;
  max_model_len: string;
  gpu_memory_utilization: string;
  max_num_seqs: string;
  dtype: "auto" | "bfloat16" | "float16" | "float32";
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
  model_repo_id: "Qwen/Qwen2.5-0.5B-Instruct",
  tensor_parallel_size: "",
  data_parallel_size: "",
  max_model_len: "",
  gpu_memory_utilization: "",
  max_num_seqs: "",
  dtype: "auto",
  // Benchmark-default extras: prefix caching off (so cache hits don't skew
  // numbers) + log requests off (so stdout spam doesn't tank throughput).
  // User can edit or clear this freely.
  extra_args_raw: "--no-enable-prefix-caching --disable-log-requests",
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
    image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
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
    - vllm==0.15.0
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

export function BenchmarkForm() {
  const router = useRouter();
  const [mode, setMode] = useState<"form" | "yaml">("form");
  const [submitting, setSubmitting] = useState(false);

  const [form, setForm] = useState<FormState>(DEFAULTS);
  const [name, setName] = useState(DEFAULTS.benchName);
  const availability = useGpuAvailability(form.gpu_type, form.gpu_count, mode === "form");
  const [yamlBuf, setYamlBuf] = useState<string>(renderYaml(DEFAULTS));
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
    toast.success(`Loaded template: ${t.name}`);
  }

  async function deleteTemplate(id: string) {
    try {
      await gateway.deleteBenchmarkTemplate(id);
      setTemplates((prev) => prev.filter((t) => t.id !== id));
      if (selectedTemplateId === id) setSelectedTemplateId("");
      toast.success("Template deleted");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  }

  async function handleSaveTemplate() {
    if (!saveName.trim()) {
      toast.error("Template needs a name");
      return;
    }
    const yamlToSave = mode === "form" ? formYaml : yamlBuf;
    try {
      const t = await gateway.createBenchmarkTemplate(saveName.trim(), yamlToSave);
      setTemplates((prev) => [t, ...prev]);
      setSaveOpen(false);
      setSaveName("");
      toast.success(`Saved template: ${t.name}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) {
      toast.error("Name is required");
      return;
    }
    const config_yaml = mode === "form" ? formYaml : yamlBuf;
    setSubmitting(true);
    try {
      const created = await gateway.createBenchmark({
        name: name.trim(),
        config_yaml,
      });
      toast.success(`Created ${created.id}`);
      router.push(`/benchmark/${encodeURIComponent(created.id)}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="mx-auto max-w-5xl space-y-6 pb-24">
      {/* Hero */}
      <div className="flex items-start gap-4 rounded-xl border border-border bg-gradient-to-br from-violet-500/10 via-background to-fuchsia-500/5 p-6">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white shadow-lg shadow-fuchsia-500/30">
          <FlaskConical className="h-6 w-6" />
        </div>
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Create benchmark</h1>
          <p className="mt-1 max-w-xl text-sm text-muted-foreground">
            Spin up a RunPod GPU, run <span className="font-mono text-xs">benchmaq</span>{" "}
            against vLLM, and stream the logs back here. Save as a template if
            you&apos;ll re-run it.
          </p>
        </div>
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
      <Tabs value={mode} onValueChange={(v) => setMode(v as "form" | "yaml")}>
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
              : "Edit raw config. Switching back to Form discards YAML edits."}
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

          {/* YAML preview */}
          <details className="group rounded-lg border border-border">
            <summary className="flex cursor-pointer items-center justify-between gap-2 px-4 py-3 text-sm font-medium hover:bg-muted/40">
              <div className="flex items-center gap-2">
                <FileCode2 className="h-4 w-4 text-muted-foreground" />
                YAML preview
                <Badge variant="secondary" className="text-[10px]">
                  read-only
                </Badge>
              </div>
              <Info className="h-3.5 w-3.5 text-muted-foreground" />
            </summary>
            <pre className="overflow-x-auto rounded-b-lg border-t border-border bg-zinc-950 px-4 py-3 font-mono text-xs leading-relaxed text-zinc-200">
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
                className="rounded-md border border-border bg-zinc-950 font-mono text-xs leading-relaxed text-zinc-200 focus-visible:ring-violet-500/40"
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Sticky-feeling action bar */}
      <div
        className={cn(
          "sticky bottom-0 -mx-6 mt-6 flex items-center justify-between gap-3",
          "border-t border-border bg-background/85 px-6 py-4 backdrop-blur",
          "lg:-mx-10 lg:px-10",
        )}
      >
        <div className="text-xs text-muted-foreground">
          A new RunPod pod will be created and torn down automatically.
        </div>
        <div className="flex items-center gap-2">
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
      <button
        type="button"
        onClick={() => onChange(!on)}
        className={cn(
          "inline-flex h-7 items-center gap-1.5 rounded-full border px-3 text-xs font-medium transition-colors",
          on
            ? "border-violet-500/40 bg-violet-500/10 text-violet-400"
            : "border-border bg-background text-muted-foreground hover:bg-muted/40",
        )}
      >
        <span
          className={cn(
            "inline-block h-2 w-2 rounded-full",
            on ? "bg-violet-400" : "bg-muted-foreground/40",
          )}
        />
        Sweep
      </button>
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
          ? "border-violet-500/40 bg-violet-500/5"
          : "border-border bg-background hover:bg-muted/30",
      )}
    >
      <span
        className={cn(
          "mt-0.5 inline-flex h-4 w-7 shrink-0 items-center rounded-full transition-colors",
          checked ? "bg-violet-500" : "bg-muted-foreground/30",
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
