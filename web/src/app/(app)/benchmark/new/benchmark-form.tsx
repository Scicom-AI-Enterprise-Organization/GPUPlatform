"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import yaml from "js-yaml";
import {
  AlertCircle,
  AlertTriangle,
  Bookmark,
  Box,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Cpu,
  Database,
  FileCode2,
  FlaskConical,
  Gauge,
  Globe,
  Info,
  KeyRound,
  Loader2,
  Package,
  RefreshCw,
  Server,
  ShieldAlert,
  Sparkles,
  Trash2,
  X,
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
import { NumberField } from "@/components/ui/number-field";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SearchableSelect } from "@/components/ui/searchable-select";
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
import { FormFooter, FormShell } from "@/components/form-shell";
import { useGpuAvailability } from "@/lib/use-gpu-availability";
import { GPU_TYPE_SUGGESTIONS } from "@/lib/bench-gpu-suggestions";
import { gateway } from "@/lib/gateway";
import type { BenchmarkTemplate, GpuTypeOption, ProviderRecord, StorageRecord, VmAvailability } from "@/lib/types";
import { cn } from "@/lib/utils";

// Fallback only — the live list comes from the gateway (/compute/runpod/gpu-types).
// Used until that fetch lands, or if it fails (offline / no key).
const GPU_COUNT_CHOICES = [1, 2, 4, 8] as const;

// Rough capacity estimate (mirrors the serverless form): ~55% of total VRAM for
// weights, the rest for KV cache / activations / overhead.
function capacityHint(vramGb: number, count: number): string {
  const total = vramGb * count;
  const weights = total * 0.55;
  const fp16 = weights / 2;
  const q4 = weights / 0.6;
  const r = (b: number) => (b >= 100 ? `${Math.round(b / 10) * 10}B` : `${Math.round(b)}B`);
  const totalStr = total >= 100 ? `${Math.round(total)} GB` : `${total} GB`;
  return `${totalStr} VRAM${count > 1 ? ` · TP=${count} sharding` : ""} · fits ~${r(fp16)} FP16 / ~${r(q4)} 4-bit (KV-cache budgeted)`;
}

const RUNPOD_GPU_FALLBACK: GpuTypeOption[] = [
  { id: "NVIDIA RTX A4000", label: "RTX A4000", vram_gb: 16, hint: "16 GB · cheap baseline" },
  { id: "NVIDIA RTX A5000", label: "RTX A5000", vram_gb: 24, hint: "24 GB" },
  { id: "NVIDIA RTX A6000", label: "RTX A6000", vram_gb: 48, hint: "48 GB" },
  { id: "NVIDIA GeForce RTX 4090", label: "RTX 4090", vram_gb: 24, hint: "24 GB · consumer" },
  { id: "NVIDIA L40", label: "L40", vram_gb: 48, hint: "48 GB" },
  { id: "NVIDIA L40S", label: "L40S", vram_gb: 48, hint: "48 GB · faster L40" },
  { id: "NVIDIA A100 80GB PCIe", label: "A100 80GB", vram_gb: 80, hint: "datacenter" },
  { id: "NVIDIA H100 80GB HBM3", label: "H100 80GB", vram_gb: 80, hint: "fastest" },
];

// Container image presets for the RunPod pod. We default to CUDA 12.8 for two
// reasons:
//   1. benchmaq derives `allowedCudaVersions` from the image tag and sends it
//      to RunPod as an exact-match host filter. A CUDA 12.4 image yields
//      ['12.4'], which excludes modern datacenter hosts (H100 SXM Secure report
//      12.8 / driver 570+) → "no instances available" even when stock is High.
//      A 12.8 image matches those hosts.
//   2. flashinfer's Hopper kernels (Qwen3-Next + GDN linear attention) need PTX
//      intrinsics that only exist in CUDA 12.6+ — CUDA 12.4 fails to JIT-compile
//      gdn_prefill_sm90 mid-inference.
// The CUDA 12.4 image stays available as a lighter/older baseline.
const DEFAULT_CONTAINER_IMAGE =
  "runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404";
const CONTAINER_IMAGE_OPTIONS = [
  {
    id: DEFAULT_CONTAINER_IMAGE,
    label: "CUDA 13.0 · torch 2.9",
    hint: "default · vLLM ≥ 0.23 (CUDA-13 torch) · ≥580-driver hosts",
  },
  {
    id: "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    label: "CUDA 12.8 · torch 2.8",
    hint: "older vLLM (≤ 0.22) · 12.8-driver hosts",
  },
  {
    id: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    label: "CUDA 12.4 · pytorch 2.4",
    hint: "older / lighter baseline",
  },
];
const CUSTOM_IMAGE_SENTINEL = "__custom__";

// Custom vLLM fork (git) — the Gemma-4 FA4 "CUTE" fork (mirrors serverless/new).
const GEMMA4_FA4_FORK_URL = "https://github.com/Scicom-AI-Enterprise-Organization/vllm-gemma4-fa4-cute";
const GEMMA4_FA4_REF = "main";
const GEMMA4_FA4_BACKEND = "--attention-backend FLASH_ATTN_CUTE"; // FA4 CUTE serve flag

// Compose a verbatim `uv pip install` arg string for a git-fork vLLM. A leading
// `VLLM_USE_PRECOMPILED=1` (the gateway reads leading NAME=VALUE tokens as install
// env, not pip args) reuses precompiled vLLM binaries — fast, no CUDA build. The
// gateway forwards this to the VM (pyremote `uv pip install -U …`) or translates
// it into the RunPod pod's dependencies + env.
function composeForkArgs(url: string, ref: string, precompiled: boolean): string {
  const u = url.trim();
  if (!u) return "";
  const spec = `git+${u}${ref.trim() ? "@" + ref.trim() : ""}`;
  return [...(precompiled ? ["VLLM_USE_PRECOMPILED=1"] : []), spec, "--torch-backend=auto"].join(" ");
}

// CUDA toolkit version → minimum NVIDIA driver version (Linux)
const CUDA_MIN_DRIVER: Record<string, string> = {
  "11.0": "450.80", "11.1": "455.23", "11.2": "460.27", "11.3": "465.19",
  "11.4": "470.57", "11.5": "495.29", "11.6": "510.39", "11.7": "515.43",
  "11.8": "520.61",
  "12.0": "525.60", "12.1": "530.30", "12.2": "535.54", "12.3": "545.23",
  "12.4": "550.54", "12.5": "555.42", "12.6": "560.28", "12.7": "565.57",
  "12.8": "570.00", "12.9": "575.51",
  "13.0": "580.65",
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
  volume_size: number;
  container_image: string;
  model_repo_id: string;
  // Ingress target only: the already-served, ingressed vLLM endpoint to bench
  // (e.g. https://my-model.example.com). Empty for cloud/vm targets. When set,
  // the run skips all provisioning and hits this URL directly. model_repo_id
  // doubles as the served model name for the API request.
  ingress_base_url: string;
  // Ingress target only: the GPU behind the external endpoint. Nothing is
  // spawned so the platform can't detect it — the author states it here so the
  // run groups by GPU in stats/comparisons/the API (visible in Parameters →
  // Hardware without a manual post-run edit). A label like "NVIDIA H20" /
  // "Ascend 910B3"; empty = unknown. Independent of cloud's runpod gpu_type.
  ingress_gpu_type: string;
  ingress_gpu_count: number;
  // All vLLM engine args are strings so empty = "use vLLM default" — same
  // ergonomics as the serverless endpoint create form.
  tensor_parallel_size: string;
  data_parallel_size: string;
  max_model_len: string;
  gpu_memory_utilization: string;
  max_num_seqs: string;
  // vLLM HTTP port. Empty = default (8000).
  port: string;
  dtype: "auto" | "bfloat16" | "float16" | "float32";
  vllm_version: string;
  // A full `uv pip install` arg string for vLLM (a git fork or nightly), used
  // verbatim instead of the version pin. Overrides vllm_version when non-empty.
  vllm_install_args: string;
  // Cmdline-style flags appended to vLLM. Parsed into snake_case serve: keys
  // at render time. e.g. "--enforce-eager --quantization awq"
  extra_args_raw: string;
  // What this run measures: throughput/latency ("speed", the classic bench
  // rows) or model quality ("accuracy", dataset evals). Accuracy runs ALSO
  // report a decode tok/s, so a multi-config accuracy run draws the
  // IQ-vs-speed plot on its own.
  bench_type: "speed" | "accuracy";
  // Accuracy mode
  acc_gsm8k: boolean;
  acc_mmmlu: boolean;
  acc_funccall: boolean; // hard multi-turn function-calling (Scicom-intl/Function-Call-TaaS)
  acc_limit: number; // total samples per dataset (split across MMMLU langs)
  acc_concurrency: number;
  acc_mmmlu_langs: string; // CSV of openai/MMMLU language configs
  // Workload
  request_rate: string;
  // Per-run warm-up: when on, each measured bench row carries vLLM's native
  // `num_warmups` (= the row's own concurrency, one full wave). vLLM fires those
  // requests at the run's exact shape + concurrency before measuring and
  // excludes them from the reported metrics — kills first-request cold start
  // (cuBLAS/cublasLt autotune, allocator growth, NCCL all-reduce at TP>1,
  // torch.compile guards) without polluting results.
  warmup: boolean;
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
  // Bare-metal only: base directory on the VM where model weights + results
  // get written. RunPod mounts /workspace; on a bare-metal VM the default is
  // `~` (the SSH user's home).
  vm_base_dir: string;
};

export const DEFAULTS: FormState = {
  benchName: "qwen-quick",
  gpu_type: "NVIDIA RTX A4000",
  gpu_count: 1,
  secure_cloud: true,
  disk_size: 80,
  volume_size: 80,
  container_image: DEFAULT_CONTAINER_IMAGE,
  model_repo_id: "Qwen/Qwen2.5-0.5B-Instruct",
  ingress_base_url: "",
  ingress_gpu_type: "",
  ingress_gpu_count: 1,
  // Default to 1 (vLLM's own default) so the knobs are visible in the template
  // YAML for runpod/VM instead of hidden until first edited. Empty still means
  // "omit / use vLLM default"; 1 renders as an explicit tensor_parallel_size: 1.
  tensor_parallel_size: "1",
  data_parallel_size: "1",
  max_model_len: "",
  gpu_memory_utilization: "",
  max_num_seqs: "",
  port: "",
  dtype: "auto",
  vllm_version: "0.23.0",
  vllm_install_args: "",
  // Benchmark-default extras: prefix caching off (so cache hits don't skew
  // numbers). --disable-log-requests was removed in vLLM > 0.15 and now
  // causes the server to refuse to start, so it's no longer in the default.
  extra_args_raw: "--no-enable-prefix-caching",
  bench_type: "speed",
  acc_gsm8k: true,
  acc_mmmlu: true,
  acc_funccall: true,
  acc_limit: 200,
  acc_concurrency: 32,
  acc_mmmlu_langs: "FR_FR, DE_DE, ES_LA, ZH_CN, JA_JP",
  request_rate: "inf",
  warmup: true,
  sweep_mode: false,
  input_len: 256,
  output_len: 128,
  num_prompts: 100,
  max_concurrency: 4,
  input_lens_csv: "128, 512, 2048, 4096, 8192, 16384",
  concurrencies_csv: "10, 50, 100",
  hf_home: "/workspace/hf_home",
  vm_base_dir: "~",
};

function parseCsvInts(s: string): number[] {
  return s
    .split(/[,\s]+/)
    .map((x) => x.trim())
    .filter(Boolean)
    .map((x) => parseInt(x, 10))
    .filter((n) => Number.isFinite(n) && n > 0);
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
    if (eq <= 0) continue;
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

function modelSlug(repo: string): string {
  const tail = repo.split("/").pop() || "model";
  return tail.toLowerCase().replace(/\./g, "p").replace(/[^a-z0-9-]/g, "-");
}

function modelToLocalDir(repo: string, baseDir = "/workspace"): string {
  const base = (baseDir || "/workspace").replace(/\/+$/, "");
  return `${base}/models/${modelSlug(repo)}`;
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
      // Per-run warm-up via vLLM's native --num-warmups: fire one full wave (=
      // this row's concurrency) at the run's exact shape before measuring.
      // vLLM excludes warm-up requests from the reported metrics, so each row —
      // and every sweep cell — self-warms with no separate row or filtering.
      const warm = s.warmup ? `, num_warmups: ${c}` : "";
      lines.push(
        `      - { endpoint: /v1/completions, dataset_name: random, ` +
          `random_input_len: ${inLen}, random_output_len: ${s.output_len}, ` +
          `num_prompts: ${s.num_prompts}, max_concurrency: ${c}, ` +
          `request_rate: ${rate}, ignore_eos: true${warm}${extra} }`,
      );
    }
  }
  return lines.join("\n");
}

/** The `accuracy:` block for an accuracy-mode run. benchmaq skips items with
 * no `bench:` rows, so only our accuracy_eval.py picks these up. */
function renderAccuracyBlock(s: FormState): string {
  const datasets: string[] = [];
  if (s.acc_gsm8k) datasets.push("gsm8k");
  if (s.acc_mmmlu) datasets.push("openai/MMMLU");
  if (s.acc_funccall) datasets.push("function-call");
  if (datasets.length === 0) datasets.push("gsm8k");
  const langs = s.acc_mmmlu_langs
    .split(/[,\s]+/)
    .map((x) => x.trim())
    .filter(Boolean);
  const lines = [
    `      datasets: [${datasets.join(", ")}]`,
    `      limit: ${s.acc_limit}`,
    `      concurrency: ${s.acc_concurrency}`,
  ];
  if (s.acc_mmmlu && langs.length) lines.push(`      languages: [${langs.join(", ")}]`);
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
  setIfNum("port", s.port);
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

export function renderYaml(
  s: FormState,
  target: "cloud" | "vm" | "ingress" = "cloud",
  storageName?: string,
  vmExtras?: {
    providerName?: string;
    cleanupModel?: boolean;
    visibleDevices?: string;
    envText?: string;
  },
): string {
  // RunPod / pod / container blocks only apply when we're provisioning a
  // fresh pod. On a registered VM the hardware is fixed and the gateway
  // injects host/port/username/key_filename into `remote:` at runtime — so
  // we render an empty placeholder block here purely for the preview.
  const runpodBlock = target === "cloud"
    ? `runpod:
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
    volume_size: ${s.volume_size}
    mount_path: "/workspace"
  ports:
    http: [8000]
    tcp: [22]
  env:
    HF_HOME: "${s.hf_home}"

`
    : "";

  // VM-only remote extras — provider / workdir / cleanup / env mirror the Pod
  // and Runtime-environment form sections so the VM YAML round-trips with the
  // form (parsed back by parseYamlToForm). Cloud keeps its existing shape.
  const vmEnv: Record<string, string> = { ...parseEnvVars(vmExtras?.envText ?? "") };
  if ((vmExtras?.visibleDevices ?? "").trim()) {
    vmEnv.CUDA_VISIBLE_DEVICES = (vmExtras!.visibleDevices as string).trim();
  }
  const vmEnvBlock =
    Object.keys(vmEnv).length > 0
      ? "  env:\n" +
        Object.entries(vmEnv)
          .map(([k, v]) => `    ${k}: ${JSON.stringify(v)}\n`)
          .join("")
      : "";
  // When no VM is picked yet, ship a fillable placeholder + guidance comment so
  // YAML-mode users know this is where the GPU provider name goes (an empty
  // value parses back to "no provider", same as omitting it).
  const vmProviderLine = vmExtras?.providerName
    ? `  provider: ${JSON.stringify(vmExtras.providerName)}\n`
    : `  provider: ""  # STATE THE NAME OF GPU PROVIDER\n`;
  const vmWorkdirLine = `  workdir: ${JSON.stringify(s.vm_base_dir || "~")}\n`;
  const vmCleanupLine =
    vmExtras?.cleanupModel !== undefined
      ? `  cleanup_model: ${vmExtras.cleanupModel}\n`
      : "";

  // Custom-fork / nightly vLLM: a full install-args string overrides the version.
  // It rides `remote.uv.vllm_install_args` — the gateway feeds it to the VM
  // (pyremote `uv pip install -U …`) or translates it into the RunPod pod's deps
  // + env. When set, drop the `vllm==` pin so the two specs don't fight.
  const forkArgs = s.vllm_install_args.trim();
  const uvInstallLine = forkArgs ? `    vllm_install_args: ${JSON.stringify(forkArgs)}\n` : "";
  const vllmDep = forkArgs ? "" : `    - vllm==${s.vllm_version || "0.23.0"}\n`;

  const remoteBlock = target === "cloud"
    ? `remote:
  key_filename: ""
  uv:
    path: ~/.venv
    python_version: "3.11"
${uvInstallLine}  dependencies:
${vllmDep}    - huggingface_hub
    - hf_transfer
`
    : `# host/port/username/key_filename are injected by the gateway from
# the selected VM provider at run time.
remote:
${vmProviderLine}${vmWorkdirLine}${vmCleanupLine}${vmEnvBlock}  uv:
    path: ~/.benchmark-venv
    python_version: "3.11"
${uvInstallLine}  dependencies:
${vllmDep}    - huggingface_hub
    - hf_transfer
`;

  // Storage backend (S3 logs/results target) for this run, referenced by name
  // on the benchmark item. The runner (benchmaq) ignores this key; the gateway
  // resolves it to a real storage backend at submit, and parseYamlToForm reads
  // it back so the Form's Storage dropdown survives a YAML round-trip. The
  // caller defaults this to the first enabled S3 backend (cloud target) so the
  // template ships pre-filled with a name that actually resolves.
  const benchStorageLine = storageName
    ? `    storage: ${JSON.stringify(storageName)}\n`
    : "";

  // Accuracy runs measure quality (dataset evals) instead of throughput; they
  // carry an `accuracy:` block and NO `bench:` rows (benchmaq skips them, our
  // accuracy_eval.py serves + evaluates them). Speed runs are the classic
  // throughput/latency bench rows.
  const workloadBlock = s.bench_type === "accuracy"
    ? `    accuracy:
${renderAccuracyBlock(s)}
`
    : `    bench:
${renderBenchEntries(s)}
    results:
      save_result: true
      save_detailed: true
`;

  // Ingress: bench an already-served, ingressed vLLM. No pod, no remote, no
  // serve block — just point base_url at the endpoint. The gateway sees the
  // base_url (and no machine provider) and runs its in-gateway httpx client.
  // model.repo_id is the served model name for the API request. Storage still
  // applies — results + logs land in the selected S3 backend.
  if (target === "ingress") {
    // gpu_type / gpu_count are stated by the author (nothing is spawned, so the
    // platform can't detect the GPU). The gateway reads them off the item and
    // stores them on the run, so Parameters → Hardware shows the GPU with no
    // manual edit. Kept on the benchmark item, same as base_url / storage.
    //
    // TP/DP are descriptive for ingress — nothing is launched, so they only
    // record how the external endpoint is served (tensor-parallel ×
    // data-parallel). They show in Parameters → Model and let the run group by
    // parallelism in stats/comparisons. Only these two serve keys (NOT the full
    // engine-arg block, which would falsely imply the platform set them).
    const ingressServe: string[] = [];
    const itp = s.tensor_parallel_size.trim();
    const idp = s.data_parallel_size.trim();
    if (itp && Number.isFinite(Number(itp)))
      ingressServe.push(`      tensor_parallel_size: ${Number(itp)}`);
    if (idp && Number.isFinite(Number(idp)))
      ingressServe.push(`      data_parallel_size: ${Number(idp)}`);
    const ingressServeBlock = ingressServe.length
      ? `    serve:\n${ingressServe.join("\n")}\n`
      : "";
    return `benchmark:
  - name: ${s.benchName}
${benchStorageLine}    engine: vllm
    base_url: "${s.ingress_base_url}"
    model:
      repo_id: "${s.model_repo_id}"
${ingressServeBlock}    gpu_type: "${s.ingress_gpu_type}"  # GPU behind the endpoint — shows in Parameters → Hardware
    gpu_count: ${s.ingress_gpu_count}
${workloadBlock}`;
  }

  return `${runpodBlock}${remoteBlock}
benchmark:
  - name: ${s.benchName}
${benchStorageLine}    engine: vllm
    model:
      repo_id: "${s.model_repo_id}"
      local_dir: "${modelToLocalDir(s.model_repo_id, target === "vm" ? s.vm_base_dir : "/workspace")}"
    serve:
${renderServeBlock(s)}
${workloadBlock}`;
}

// Serve keys that map 1:1 to a Form field. Everything else under
// benchmark[0].serve is flattened back into Extra args (raw cmdline form).
const FORM_SERVE_KEYS = new Set([
  "tensor_parallel_size",
  "data_parallel_size",
  "max_model_len",
  "gpu_memory_utilization",
  "max_num_seqs",
  "port",
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
  /** Raw top-level `storage:` value from the YAML (a backend name), if present.
   * Held separately from FormState because storage selection is its own state;
   * the caller resolves it to a real storage id against the loaded list. */
  storageRef: string | null;
  /** VM-only `remote.*` controls held outside FormState (their own React
   * state). The caller resolves providerRef → a provider id and applies the
   * rest to their setters. `null` = the YAML didn't specify it. */
  providerRef: string | null;
  cleanupModel: boolean | null;
  visibleDevices: string | null;
  envText: string | null;
  /** Ingress `base_url` (top-level or on benchmark[0]) if present. The caller
   * uses its presence to switch the Run-on target to "ingress". */
  baseUrl: string | null;
};

/** Parse a benchmaq YAML config back into FormState. Anything the form
 * doesn't represent (extra env vars, multiple bench items, custom engine,
 * etc.) is collected into `unknownKeys` so we can warn the user that
 * round-tripping through Form mode will drop those keys. */
export function parseYamlToForm(src: string, fallback: FormState): ParseYamlResult {
  let doc: unknown;
  try {
    doc = yaml.load(src);
  } catch (e) {
    return {
      state: fallback,
      unknownKeys: [],
      parseError: e instanceof Error ? e.message : String(e),
      storageRef: null,
      providerRef: null,
      cleanupModel: null,
      visibleDevices: null,
      envText: null,
      baseUrl: null,
    };
  }
  if (!doc || typeof doc !== "object") {
    return {
      state: fallback,
      unknownKeys: [],
      parseError: "empty config",
      storageRef: null,
      providerRef: null,
      cleanupModel: null,
      visibleDevices: null,
      envText: null,
      baseUrl: null,
    };
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

  // ---- runpod.storage — pod volume. renderYaml writes volume_size here, so we
  // must read it back; otherwise a YAML→Form→YAML round-trip silently resets the
  // volume to the form default (the "disk reverts to default" bug).
  const podStorage = ((d.runpod as Record<string, unknown> | undefined)
    ?.storage ?? {}) as Record<string, unknown>;
  if (typeof podStorage.volume_size === "number")
    next.volume_size = podStorage.volume_size;

  // ---- runpod.env
  const env = ((d.runpod as Record<string, unknown> | undefined)?.env ?? {}) as
    Record<string, unknown>;
  if (typeof env.HF_HOME === "string") next.hf_home = env.HF_HOME;
  for (const k of Object.keys(env)) {
    if (k !== "HF_HOME") unknown.push(`runpod.env.${k}`);
  }

  // ---- remote.uv.vllm_install_args — a fork / custom install string (overrides
  // the version). Round-trips so a fork survives YAML ↔ Form.
  const uvBlock = (d.remote as Record<string, unknown> | undefined)?.uv as
    | Record<string, unknown>
    | undefined;
  if (uvBlock && typeof uvBlock.vllm_install_args === "string") {
    next.vllm_install_args = uvBlock.vllm_install_args.trim();
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
  // Ingress configs carry the served name as a bare `model: "name"` string.
  if (typeof first.model === "string") next.model_repo_id = first.model;

  // ---- base_url (ingress) — top-level or on the item. Its presence is what the
  // caller keys on to flip the Run-on target to "ingress" after a YAML edit.
  const itemBaseUrl = typeof first.base_url === "string" ? first.base_url.trim() : "";
  const topBaseUrl = typeof d.base_url === "string" ? (d.base_url as string).trim() : "";
  const baseUrl = itemBaseUrl || topBaseUrl || null;
  if (baseUrl) next.ingress_base_url = baseUrl;

  // ---- ingress gpu_type / gpu_count — the author-stated GPU behind the
  // endpoint (item-level, top-level fallback). Same keys the gateway parses;
  // round-trips so flipping Form<->YAML doesn't drop the hardware label.
  const itemGpuType = typeof first.gpu_type === "string" ? first.gpu_type.trim() : "";
  const topGpuType = typeof d.gpu_type === "string" ? (d.gpu_type as string).trim() : "";
  const gpuType = itemGpuType || topGpuType;
  if (gpuType) next.ingress_gpu_type = gpuType;
  const gpuCount =
    typeof first.gpu_count === "number"
      ? first.gpu_count
      : typeof d.gpu_count === "number"
        ? (d.gpu_count as number)
        : null;
  if (gpuCount != null && gpuCount > 0) next.ingress_gpu_count = gpuCount;

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
  // Per-run warm-up is native now (vLLM `num_warmups` on each row); any row
  // carrying num_warmups > 0 flips the warm-up toggle back on. The key is
  // otherwise ignored by sweep/workload parsing.
  const benchRows = Array.isArray(first.bench)
    ? (first.bench as Record<string, unknown>[])
    : [];
  next.warmup = benchRows.some(
    (r) => typeof r.num_warmups === "number" && r.num_warmups > 0,
  );
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

  // ---- benchmark[0].accuracy — quality-eval mode. Its presence flips the
  // run type to "accuracy"; benchmaq ignores the block and accuracy_eval.py
  // serves + evaluates it.
  const acc = (first.accuracy ?? null) as Record<string, unknown> | null;
  if (acc && typeof acc === "object") {
    next.bench_type = "accuracy";
    const ds = Array.isArray(acc.datasets) ? acc.datasets.map((x) => String(x).toLowerCase()) : [];
    if (ds.length) {
      next.acc_gsm8k = ds.some((x) => x.includes("gsm8k"));
      next.acc_mmmlu = ds.some((x) => x.includes("mmmlu"));
      next.acc_funccall = ds.some(
        (x) => x.includes("function-call") || x.includes("function_call") || x.includes("taas"),
      );
    }
    if (typeof acc.limit === "number") next.acc_limit = acc.limit;
    if (typeof acc.concurrency === "number") next.acc_concurrency = acc.concurrency;
    if (Array.isArray(acc.languages)) next.acc_mmmlu_langs = acc.languages.map(String).join(", ");
  } else {
    next.bench_type = "speed";
  }

  // ---- benchmark[].storage: backend name on a bench item (resolved to an id
  // by the caller). Use the first item that names one.
  let storageRef: string | null = null;
  for (const b of benches) {
    if (b && typeof b === "object") {
      const sv = (b as Record<string, unknown>).storage;
      if (typeof sv === "string" && sv.trim()) {
        storageRef = sv.trim();
        break;
      }
    }
  }

  // ---- remote.* — VM-only controls that live outside FormState. provider →
  // resolved to an id by the caller; workdir → vm_base_dir (in FormState);
  // cleanup_model / env (CUDA_VISIBLE_DEVICES + extra vars) → caller setters.
  const remote = (d.remote ?? {}) as Record<string, unknown>;
  const providerRef =
    typeof remote.provider === "string" && remote.provider.trim()
      ? remote.provider.trim()
      : null;
  if (typeof remote.workdir === "string" && remote.workdir.trim()) {
    next.vm_base_dir = remote.workdir.trim();
  }
  const cleanupModel =
    typeof remote.cleanup_model === "boolean" ? remote.cleanup_model : null;
  let visibleDevices: string | null = null;
  let envText: string | null = null;
  const remoteEnv = remote.env;
  if (remoteEnv && typeof remoteEnv === "object") {
    const envLines: string[] = [];
    for (const [k, v] of Object.entries(remoteEnv as Record<string, unknown>)) {
      if (k === "CUDA_VISIBLE_DEVICES") {
        visibleDevices = String(v);
        continue;
      }
      envLines.push(`export ${k}=${String(v)}`);
    }
    if (envLines.length > 0) envText = envLines.join("\n");
  }

  return {
    state: next,
    unknownKeys: unknown,
    parseError: null,
    storageRef,
    providerRef,
    cleanupModel,
    visibleDevices,
    envText,
    baseUrl,
  };
}

export function BenchmarkForm({
  initialName,
  initialYaml,
  initialProviderId,
}: {
  initialName?: string;
  initialYaml?: string;
  /** When duplicating a benchmark, carry over the source's provider choice
   * so the "Run on" tile + VM provider dropdown reflect the original. */
  initialProviderId?: string | null;
} = {}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  // Duplicate flow: start in YAML mode with the source config pre-filled so
  // the round-trip is exact. Switching back to Form mode parses the YAML
  // and back-fills the form (lossy for keys the form doesn't represent).
  // ?tab= in the URL wins so the active tab is shareable / survives refresh;
  // otherwise the duplicate flow's YAML preference decides the default.
  const initialMode: "form" | "yaml" = (() => {
    const t = searchParams.get("tab");
    if (t === "form" || t === "yaml") return t;
    return initialYaml ? "yaml" : "form";
  })();
  const [mode, setModeState] = useState<"form" | "yaml">(initialMode);
  // Reflect the active tab in the URL (no history spam, no scroll jump).
  const setMode = (next: "form" | "yaml") => {
    setModeState(next);
    const params = new URLSearchParams(searchParams.toString());
    params.set("tab", next);
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
  };
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

  // Provider selection — "cloud" means platform default (RunPod), "vm" routes
  // execution to a user-registered VM via the GPU Providers page. Declared
  // before formYaml so renderYaml picks up the choice in the preview.
  const [providers, setProviders] = useState<ProviderRecord[]>([]);
  // Source-of-truth: if duplicating from a bench that ran on a VM, start in
  // "vm" mode with that provider preselected; otherwise default to cloud.
  const [target, setTarget] = useState<"cloud" | "vm" | "ingress">(
    initialProviderId
      ? "vm"
      : initialYaml && /\bbase_url\s*:/.test(initialYaml)
        ? "ingress"
        : "cloud",
  );
  const [providerId, setProviderId] = useState<string>(initialProviderId ?? "");
  // RunPod-account selection for cloud target. Empty = gateway-default key.
  const [runpodProviderId, setRunpodProviderId] = useState<string>("");
  // Storage backend for this run's logs + result files. Required. Only s3-kind,
  // enabled storages are eligible.
  const [storages, setStorages] = useState<StorageRecord[]>([]);
  const [storageId, setStorageId] = useState<string>("");
  // RunPod GPU catalog — fetched live from the gateway, fallback until it lands.
  const [gpuOptions, setGpuOptions] = useState<GpuTypeOption[]>(RUNPOD_GPU_FALLBACK);
  const [cleanupModel, setCleanupModel] = useState(true);
  // Visibility at create time. false (default) = private (only you + admins).
  // true = public: read-only visible to every logged-in user. Owner can flip it
  // later from the benchmark row menu / detail page. Platform flag — NOT part of
  // the benchmaq config YAML, so it's a form-only control.
  const [isPublic, setIsPublic] = useState(false);
  // CUDA_VISIBLE_DEVICES pin + extra env exported for the run (cache/home dirs).
  const [visibleDevices, setVisibleDevices] = useState("");
  const [envText, setEnvText] = useState("");
  // HuggingFace token for gated models — from a global secret (default) or a
  // pasted token. `secretKeys` is the list of admin Secrets keys to pick from.
  const [secretKeys, setSecretKeys] = useState<string[]>([]);
  const [hfSource, setHfSource] = useState<"secret" | "paste">("secret");
  const [hfToken, setHfToken] = useState("");
  const [hfTokenSecret, setHfTokenSecret] = useState("");
  // Ingress endpoint API key (optional) — for a served vLLM behind auth. From a
  // global secret (aliased to OPENAI_API_KEY at launch) or a pasted key. The
  // ingress client sends it as `Authorization: Bearer <key>`.
  const [apiKeySource, setApiKeySource] = useState<"secret" | "paste">("paste");
  const [apiKey, setApiKey] = useState("");
  const [apiKeySecret, setApiKeySecret] = useState("");

  // Live SSH probe of the selected VM. Re-fires when the user changes the
  // provider, and when they hit the refresh button.
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
      setVmAvail({
        status: "error",
        message: e instanceof Error ? e.message : String(e),
      });
    }
  }, []);
  useEffect(() => {
    if (target === "vm" && providerId) refreshVmAvail(providerId);
    else setVmAvail({ status: "idle" });
  }, [target, providerId, refreshVmAvail]);

  const selectedStorageName = useMemo(
    () => storages.find((s) => s.id === storageId)?.name,
    [storages, storageId],
  );
  // For both the RunPod (cloud) and VM templates, pre-fill `storage:` with the
  // first enabled S3 backend so it ships with a name that actually resolves at
  // submit (a literal like "s3" would fail unless a backend happened to be named
  // that). An explicit dropdown pick always wins.
  const defaultStorageName = useMemo(() => {
    if (selectedStorageName) return selectedStorageName;
    return storages.find((s) => s.kind === "s3" && s.enabled)?.name;
  }, [selectedStorageName, storages]);
  const selectedProviderName = useMemo(
    () => providers.find((p) => p.id === providerId)?.name,
    [providers, providerId],
  );
  const formYaml = useMemo(
    () =>
      renderYaml(
        { ...form, benchName: name || "untitled" },
        target,
        defaultStorageName,
        target === "vm"
          ? { providerName: selectedProviderName, cleanupModel, visibleDevices, envText }
          : undefined,
      ),
    [
      form,
      name,
      target,
      defaultStorageName,
      selectedProviderName,
      cleanupModel,
      visibleDevices,
      envText,
    ],
  );

  const [templates, setTemplates] = useState<BenchmarkTemplate[]>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>("");
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveName, setSaveName] = useState("");
  // Controlled disclosure for the YAML preview (see the note at its render).
  const [yamlPreviewOpen, setYamlPreviewOpen] = useState(false);

  useEffect(() => {
    gateway.listBenchmarkTemplates().then(setTemplates).catch(() => {});
    gateway
      .listProviders()
      .then((ps) => {
        setProviders(ps);
        // Auto-select the first registered RunPod account — no gateway-default fallback.
        const firstRunpod = ps.find((p) => p.kind === "runpod");
        if (firstRunpod) setRunpodProviderId((cur) => cur || firstRunpod.id);
      })
      .catch(() => {});
    gateway.listStorage().then(setStorages).catch(() => {});
    // Global-secret keys the HF token can reference (keys only; values stay server-side).
    fetch("/api/proxy/v1/global-env", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : []))
      .then((rows) => {
        if (Array.isArray(rows)) setSecretKeys(rows.map((r: { key: string }) => r.key));
      })
      .catch(() => {});
    gateway
      .listRunpodGpuTypes()
      .then((rows) => {
        if (rows.length === 0) return;
        setGpuOptions(rows);
        // If the current pick isn't in the live catalog, fall to the first.
        setForm((f) => (rows.some((g) => g.id === f.gpu_type) ? f : { ...f, gpu_type: rows[0].id }));
      })
      .catch(() => {});
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

  // Only enabled S3 storages can hold a run's logs + result files. A benchmark
  // can't be created without one, so this gates the submit button.
  const eligibleStorages = storages.filter((s) => s.kind === "s3" && s.enabled);
  const hasStorage = eligibleStorages.length > 0;

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);
    if (!name.trim()) {
      setSubmitError("Name is required.");
      return;
    }
    const config_yaml = mode === "form" ? formYaml : yamlBuf;
    // In YAML mode the VM/runtime config lives in the YAML — derive the submit
    // fields from it (falling back to form state) so a YAML-only edit drives
    // the run without re-picking everything in the form.
    const parsedYaml = mode === "yaml" ? parseYamlToForm(yamlBuf, form) : null;
    let effectiveProviderId = providerId;
    if (target === "vm" && !effectiveProviderId && parsedYaml?.providerRef) {
      const pref = parsedYaml.providerRef.toLowerCase();
      const pmatch = providers.find(
        (p) =>
          p.kind === "vm" &&
          (p.name.toLowerCase() === pref || p.id.toLowerCase() === pref),
      );
      if (pmatch) effectiveProviderId = pmatch.id;
    }
    if (target === "vm" && !effectiveProviderId) {
      setSubmitError(
        "Pick a VM provider, name one in the YAML (remote.provider), or switch back to cloud.",
      );
      return;
    }
    if (target === "cloud" && !runpodProviderId) {
      setSubmitError("Select a RunPod provider — add one under GPU Providers.");
      return;
    }
    if (target === "ingress") {
      const hasBaseUrl =
        mode === "yaml"
          ? /\bbase_url\s*:/.test(yamlBuf)
          : !!form.ingress_base_url.trim();
      if (!hasBaseUrl) {
        setSubmitError("Enter the endpoint URL (base_url) of the served vLLM to benchmark.");
        return;
      }
    }
    if (!storageId) {
      // In YAML mode the backend can be named inside the config (`storage:`),
      // which the gateway resolves at submit — so don't force a dropdown pick
      // when the YAML already names one.
      const yamlNamesStorage = parsedYaml?.storageRef != null;
      if (!yamlNamesStorage) {
        setSubmitError("Pick a storage for the run's logs and metrics.");
        return;
      }
    }
    // GPU pin / env / cleanup: prefer the YAML when in YAML mode, else the form.
    const effectiveVisibleDevices =
      parsedYaml?.visibleDevices != null ? parsedYaml.visibleDevices : visibleDevices;
    const effectiveCleanup =
      parsedYaml?.cleanupModel != null ? parsedYaml.cleanupModel : cleanupModel;
    const effectiveEnvText =
      parsedYaml?.envText != null ? parsedYaml.envText : envText;
    // Ingress GPU identity: the YAML in YAML mode, else the form. config_yaml
    // already carries gpu_type on the item (the gateway parses it regardless),
    // but sending it also sets the run's row — matching the manual Hardware
    // editor — so Parameters → Hardware shows the GPU with no post-run edit.
    const effIngressGpuType =
      (mode === "yaml" ? parsedYaml?.state.ingress_gpu_type : form.ingress_gpu_type) || "";
    const effIngressGpuCount =
      (mode === "yaml" ? parsedYaml?.state.ingress_gpu_count : form.ingress_gpu_count) || 1;
    setSubmitting(true);
    try {
      const envVars = parseEnvVars(effectiveEnvText);
      // A pasted HF token rides along in env_vars (highest precedence at launch);
      // a chosen global secret is sent as a key ref the gateway aliases to HF_TOKEN.
      if (hfSource === "paste" && hfToken.trim()) {
        envVars.HF_TOKEN = hfToken.trim();
      }
      // Ingress endpoint API key: a pasted key rides in env_vars as OPENAI_API_KEY
      // (the ingress client sends it as Authorization: Bearer); a chosen global
      // secret is sent as a key ref the gateway aliases to OPENAI_API_KEY at launch.
      if (target === "ingress" && apiKeySource === "paste" && apiKey.trim()) {
        envVars.OPENAI_API_KEY = apiKey.trim();
      }
      const created = await gateway.createBenchmark({
        name: name.trim(),
        config_yaml,
        provider_id:
          target === "vm"
            ? effectiveProviderId
            : target === "ingress"
              ? null // ingress provisions nothing — gateway detects base_url
              : runpodProviderId,
        storage_id: storageId || null,
        is_public: isPublic,
        cleanup_model: target === "vm" ? effectiveCleanup : undefined,
        ...(Object.keys(envVars).length ? { env_vars: envVars } : {}),
        ...(effectiveVisibleDevices.trim()
          ? { visible_devices: effectiveVisibleDevices.trim() }
          : {}),
        ...(hfSource === "secret" && hfTokenSecret ? { hf_token_secret: hfTokenSecret } : {}),
        ...(target === "ingress" && apiKeySource === "secret" && apiKeySecret ? { api_key_secret: apiKeySecret } : {}),
        ...(target === "ingress" && effIngressGpuType.trim()
          ? { gpu_type: effIngressGpuType.trim(), gpu_count: effIngressGpuCount }
          : {}),
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
    <FormShell>
    <form onSubmit={onSubmit} className="space-y-6">
      {/* Header — plain, no gradient. */}
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Create benchmark</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Spin up a RunPod GPU, run <span className="font-mono text-xs">benchmaq</span>{" "}
          against vLLM, and stream the logs back here. Save as a template if
          you&apos;ll re-run it.
        </p>
      </div>

      {/* Templates */}
      <Card data-form-section="Templates" className="scroll-mt-6">
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
      <Card data-form-section="Name" className="scroll-mt-6">
        <CardContent className="pt-6">
          <div className="space-y-2">
            <Label htmlFor="benchName" className="text-xs uppercase tracking-wide text-muted-foreground">
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
        className="!block"
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
            // A base_url in the YAML means ingress — flip the Run-on target so
            // the form shows the Endpoint card (and submits with no provider).
            if (parsed.baseUrl) setTarget("ingress");
            // Resolve the YAML's `storage:` name to a real backend id and
            // select it, so the dropdown — and the storage_id we submit —
            // matches what the YAML asked for. Match by name (case-insensitive),
            // then fall back to id in case the user wrote the raw id. Unmatched
            // or absent → leave the current selection untouched.
            if (parsed.storageRef) {
              const ref = parsed.storageRef.toLowerCase();
              const match = eligibleStorages.find(
                (s) => s.name.toLowerCase() === ref || s.id.toLowerCase() === ref,
              );
              if (match) {
                setStorageId(match.id);
              } else {
                toast.warning(
                  `No enabled S3 storage named "${parsed.storageRef}". ` +
                    `Pick one in the Storage section.`,
                  { duration: 6000 },
                );
              }
            }
            // VM-only remote controls — apply to their own state so the Pod +
            // Runtime-environment form sections reflect what the YAML asked for.
            if (parsed.cleanupModel !== null) setCleanupModel(parsed.cleanupModel);
            if (parsed.visibleDevices !== null) setVisibleDevices(parsed.visibleDevices);
            if (parsed.envText !== null) setEnvText(parsed.envText);
            if (parsed.providerRef) {
              const pref = parsed.providerRef.toLowerCase();
              const pmatch = providers.find(
                (p) =>
                  p.kind === "vm" &&
                  (p.name.toLowerCase() === pref || p.id.toLowerCase() === pref),
              );
              if (pmatch) {
                setProviderId(pmatch.id);
              } else {
                toast.warning(
                  `No VM provider named "${parsed.providerRef}". Pick one in the Pod section.`,
                  { duration: 6000 },
                );
              }
            }
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

        <TabsContent value="form" className="mt-4 space-y-6 !flex-none">
          {/* Where to run — platform cloud (RunPod) or one of the user's
              registered VMs (GPU Providers page). Sits at the top of the
              Form tab so users pick target before configuring everything else. */}
          <SectionCard
            icon={<Server className="h-4 w-4" />}
            title="Run on"
            description="Default cloud spawns a fresh RunPod pod per run. Bare metal uses a registered VM. Ingress benchmarks a vLLM you've already served + exposed — no provisioning."
          >
            <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
              <button
                type="button"
                onClick={() => setTarget("cloud")}
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
                onClick={() => setTarget("vm")}
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
              <button
                type="button"
                onClick={() => setTarget("ingress")}
                className={cn(
                  "flex items-start gap-3 rounded-md border px-3 py-2.5 text-left text-sm transition-colors",
                  target === "ingress"
                    ? "border-primary/60 bg-primary/5"
                    : "border-border hover:border-primary/40 hover:bg-muted/40",
                )}
              >
                <Globe className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
                <div className="min-w-0">
                  <div className="font-medium">Ingress (existing endpoint)</div>
                  <div className="text-xs text-muted-foreground">
                    Bench a vLLM you already serve. No pod, no SSH.
                  </div>
                </div>
              </button>
            </div>
          </SectionCard>

          {/* Ingress — bench an already-served, ingressed vLLM. No pod / no SSH;
              the gateway hits this URL directly with its in-gateway client. */}
          {target === "ingress" && (
            <SectionCard
              icon={<Globe className="h-4 w-4" />}
              title="Endpoint"
              description="The already-served vLLM to benchmark. No GPU is provisioned — the gateway sends the workload straight to this URL."
            >
              <div className="space-y-4">
                <FieldWrap
                  label="Endpoint URL (base_url)"
                  hint="The OpenAI-compatible base. The path (e.g. /v1/completions) is appended per bench row — don't include it. If it's behind auth, set an API key below."
                  wide
                >
                  <Input
                    className="font-mono"
                    value={form.ingress_base_url}
                    onChange={(e) => field("ingress_base_url", e.target.value)}
                    placeholder="https://my-model.example.com"
                  />
                </FieldWrap>
                <FieldWrap
                  label="Served model name"
                  hint="Must match the model id the endpoint serves (its /v1/models id) — it's sent verbatim in each request. No weights are downloaded."
                  wide
                >
                  <Input
                    className="font-mono"
                    value={form.model_repo_id}
                    onChange={(e) => field("model_repo_id", e.target.value)}
                    placeholder="google/gemma-4-31b-it"
                  />
                </FieldWrap>
                <FieldWrap
                  label="GPU type × count"
                  hint="Nothing is provisioned, so state the GPU behind this endpoint. It's written into the YAML and shown in Parameters → Hardware — the run groups by GPU in stats, comparisons, and the API with no manual edit. Pick a known GPU or type any (e.g. Ascend 910B3). Leave blank if unknown."
                  wide
                >
                  <div className="flex items-center gap-2">
                    <Input
                      className="font-mono"
                      list="ingress-gpu-suggestions"
                      value={form.ingress_gpu_type}
                      onChange={(e) => field("ingress_gpu_type", e.target.value)}
                      placeholder="NVIDIA H20"
                    />
                    <span className="shrink-0 text-muted-foreground">×</span>
                    <Input
                      type="number"
                      min={1}
                      className="w-20 font-mono"
                      value={form.ingress_gpu_count}
                      onChange={(e) =>
                        field("ingress_gpu_count", Math.max(1, parseInt(e.target.value, 10) || 1))
                      }
                    />
                    <datalist id="ingress-gpu-suggestions">
                      {GPU_TYPE_SUGGESTIONS.map((g) => (
                        <option key={g} value={g} />
                      ))}
                    </datalist>
                  </div>
                </FieldWrap>
                <FieldWrap
                  label="Parallelism — TP × DP (optional)"
                  hint="Descriptive only — nothing is launched. Records how the endpoint is served: tensor-parallel (left) × data-parallel (right). Written to the YAML, shown in Parameters → Model, and groups runs by parallelism. Leave 1 × 1 if unknown."
                  wide
                >
                  <div className="flex items-end gap-2">
                    <div className="flex flex-col gap-1">
                      <span className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                        TP
                      </span>
                      <Input
                        type="text"
                        inputMode="numeric"
                        className="w-24 font-mono"
                        value={form.tensor_parallel_size}
                        onChange={(e) => field("tensor_parallel_size", e.target.value)}
                        placeholder="1"
                        aria-label="tensor-parallel-size"
                      />
                    </div>
                    <span className="shrink-0 pb-2 text-muted-foreground">×</span>
                    <div className="flex flex-col gap-1">
                      <span className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                        DP
                      </span>
                      <Input
                        type="text"
                        inputMode="numeric"
                        className="w-24 font-mono"
                        value={form.data_parallel_size}
                        onChange={(e) => field("data_parallel_size", e.target.value)}
                        placeholder="1"
                        aria-label="data-parallel-size"
                      />
                    </div>
                  </div>
                </FieldWrap>
                <FieldWrap
                  label="API key (optional)"
                  hint="For an endpoint behind auth — sent as Authorization: Bearer on each request. Use a global secret (resolved to OPENAI_API_KEY at launch, rotates without editing this benchmark) or paste a key. Leave empty for an open endpoint."
                  wide
                >
                  <div className="space-y-2">
                    <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
                      {(["secret", "paste"] as const).map((src) => (
                        <button
                          key={src}
                          type="button"
                          onClick={() => setApiKeySource(src)}
                          className={
                            "rounded px-2.5 py-1 transition-colors " +
                            (apiKeySource === src ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")
                          }
                        >
                          {src === "secret" ? "Global secret" : "Paste a key"}
                        </button>
                      ))}
                    </div>
                    {apiKeySource === "secret" ? (
                      secretKeys.length > 0 ? (
                        <Select value={apiKeySecret} onValueChange={setApiKeySecret}>
                          <SelectTrigger>
                            <SelectValue placeholder="Select a secret (e.g. OPENAI_API_KEY)" />
                          </SelectTrigger>
                          <SelectContent>
                            {secretKeys.map((k) => (
                              <SelectItem key={k} value={k} className="font-mono text-xs">{k}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      ) : (
                        <p className="text-xs text-muted-foreground">
                          No global secrets yet. Add one under{" "}
                          <a href="/admin/secrets" className="underline underline-offset-2 hover:text-foreground">Secrets</a>{" "}
                          then pick it here — or switch to <span className="font-medium">Paste a key</span>.
                        </p>
                      )
                    ) : (
                      <Input
                        type="password"
                        autoComplete="off"
                        className="font-mono"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        placeholder="sk-… (leave empty if no key is needed)"
                      />
                    )}
                  </div>
                </FieldWrap>
              </div>
            </SectionCard>
          )}

          {/* Pod — when cloud, shows GPU/disk knobs for the fresh RunPod pod.
              When bare-metal, swaps to a VM picker + cleanup toggle. Same
              section so the layout stays consistent regardless of target.
              Hidden for ingress (nothing is provisioned). */}
          {target !== "ingress" && (
          <SectionCard
            icon={<Server className="h-4 w-4" />}
            title="Pod"
            description={
              target === "cloud"
                ? "GPU, count, and cloud tier for the RunPod instance benchmaq spawns."
                : "Which registered VM benchmaq should SSH into. Hardware is fixed by the VM."
            }
          >
          {target === "vm" && (
            <div className="space-y-3">
              <div className="space-y-1.5">
                <Label htmlFor="bench-provider" className="text-xs uppercase tracking-wide text-muted-foreground">VM provider</Label>
                {providers.length === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    No VM providers registered. Add one at{" "}
                    <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">
                      GPU Providers → New provider
                    </a>
                    .
                  </p>
                ) : (
                  <Select value={providerId} onValueChange={setProviderId}>
                    <SelectTrigger id="bench-provider">
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
                <p className="text-xs text-muted-foreground">
                  benchmaq runs directly on the VM via SSH. The VM&apos;s GPUs,
                  disk, and Python environment are used as-is.
                </p>
                {providerId && <VmAvailabilityRow state={vmAvail} onRefresh={() => refreshVmAvail(providerId)} />}
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="bench-vm-base" className="text-xs uppercase tracking-wide text-muted-foreground">Working directory on VM</Label>
                <Input
                  id="bench-vm-base"
                  value={form.vm_base_dir}
                  onChange={(e) => field("vm_base_dir", e.target.value)}
                  placeholder="~"
                  className="font-mono text-xs"
                />
                <p className="text-xs text-muted-foreground">
                  Base path where the model + run artifacts get written. Models
                  land under <span className="font-mono">{(form.vm_base_dir || "~").replace(/\/+$/, "")}/models/&lt;name&gt;</span>.
                  Default <span className="font-mono">~</span> (the SSH user&apos;s home). Use an
                  absolute path like <span className="font-mono">/mnt/scratch</span> if the home
                  partition is small.
                </p>
              </div>

              <label className="flex cursor-pointer items-start gap-2.5 rounded-md border border-border bg-muted/30 px-3 py-2.5 text-sm hover:bg-muted/50">
                <input
                  type="checkbox"
                  checked={cleanupModel}
                  onChange={(e) => setCleanupModel(e.target.checked)}
                  className="mt-0.5 h-4 w-4 shrink-0 cursor-pointer accent-primary"
                />
                <div className="min-w-0">
                  <div className="font-medium">Clean up model after run</div>
                  <div className="text-xs text-muted-foreground">
                    SSH back in when benchmaq exits (success or fail) and
                    <span className="font-mono"> rm -rf </span>
                    the model&apos;s <span className="font-mono">local_dir</span>{" "}
                    + HF hub cache. Keeps the VM&apos;s disk from filling up
                    across runs.
                  </div>
                </div>
              </label>
            </div>
          )}
          {target === "cloud" && (
            <div className="space-y-5">
              <FieldWrap
                label="RunPod account"
                hint="Which RunPod provider to bill against."
              >
                <Select
                  value={runpodProviderId}
                  onValueChange={setRunpodProviderId}
                >
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
                {providers.filter((p) => p.kind === "runpod").length === 0 && (
                  <p className="text-xs text-muted-foreground">
                    None registered. <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">Add a RunPod account →</a>
                  </p>
                )}
              </FieldWrap>

              <FieldWrap
                label="Cloud tier"
                hint="Community is cheaper with variable hosts; Secure uses vetted hosts with more capacity."
              >
                <div className="grid grid-cols-2 gap-2">
                  {([["secure", "Secure", "vetted hosts, more capacity"], ["community", "Community", "cheaper, variable hosts"]] as const).map(
                    ([val, title, sub]) => {
                      const selected = (val === "secure") === form.secure_cloud;
                      return (
                        <button
                          key={val}
                          type="button"
                          onClick={() => field("secure_cloud", val === "secure")}
                          className={cn(
                            "rounded-md border p-3 text-left transition-colors",
                            selected
                              ? "border-foreground/60 ring-1 ring-foreground/20"
                              : "border-border hover:border-foreground/40",
                          )}
                        >
                          <div className="text-sm font-medium">{title}</div>
                          <div className="mt-0.5 text-xs text-muted-foreground">{sub}</div>
                        </button>
                      );
                    },
                  )}
                </div>
              </FieldWrap>

              <FieldWrap
                label="GPU"
                hint={(() => {
                  const g = gpuOptions.find((o) => o.id === form.gpu_type);
                  return g ? capacityHint(g.vram_gb, form.gpu_count) : undefined;
                })()}
                extra={<AvailabilityBadge state={availability} count={form.gpu_count} />}
              >
                <div className="flex gap-2">
                  <SearchableSelect
                    className="flex-1"
                    value={form.gpu_type}
                    onChange={(v) => field("gpu_type", v)}
                    options={gpuOptions.map((g) => ({
                      value: g.id,
                      label: g.label,
                      hint: capacityHint(g.vram_gb, 1),
                    }))}
                    placeholder="Choose a GPU"
                    searchPlaceholder="Search GPUs (e.g. h100, 24gb, ada)…"
                  />
                  <Select
                    value={String(form.gpu_count)}
                    onValueChange={(v) => field("gpu_count", Number.parseInt(v, 10))}
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
              </FieldWrap>

              <div className="grid grid-cols-2 gap-3">
                <FieldWrap label="Container disk (GB)" hint="Ephemeral workspace. Resets when the pod stops.">
                  <NumberField min={20} value={form.disk_size} onChange={(v) => field("disk_size", v)} />
                </FieldWrap>
                <FieldWrap label="Volume (GB)" hint="Persistent volume mounted at /workspace (model cache).">
                  <NumberField min={0} value={form.volume_size} onChange={(v) => field("volume_size", v)} />
                </FieldWrap>
              </div>

              <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>
                  Pick a GPU with enough VRAM for your model. vLLM will fail to load if the
                  weights plus KV cache exceed GPU memory.
                </span>
              </div>
            </div>
          )}
          </SectionCard>
          )}

          {/* Runtime environment — GPU pinning + extra env exported for the run.
              Hidden for ingress: nothing is provisioned, so GPU pinning / run env
              don't apply. */}
          {target !== "ingress" && (
          <SectionCard
            icon={<Cpu className="h-4 w-4" />}
            title="Runtime environment"
            description="GPU pinning and environment variables for the run."
          >
            <div className="space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="bench-cuda" className="text-xs uppercase tracking-wide text-muted-foreground">CUDA_VISIBLE_DEVICES</Label>
                <Input
                  id="bench-cuda"
                  value={visibleDevices}
                  onChange={(e) => setVisibleDevices(e.target.value)}
                  placeholder="e.g. 0,1,2,3 (empty = all GPUs)"
                  className="font-mono text-xs"
                />
                <p className="text-xs text-muted-foreground">
                  Pins which GPUs the run uses. Empty = all visible GPUs.
                </p>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="bench-env" className="text-xs uppercase tracking-wide text-muted-foreground">Environment variables</Label>
                <Textarea
                  id="bench-env"
                  value={envText}
                  onChange={(e) => setEnvText(e.target.value)}
                  rows={6}
                  placeholder={"export HF_HOME=/share/huggingface\nexport TRITON_CACHE_DIR=/share/triton_cache\nexport VLLM_CACHE_ROOT=/share/vllm_cache"}
                  className="font-mono text-xs"
                />
                <p className="text-xs text-muted-foreground">
                  One <span className="font-mono">KEY=value</span> per line (<span className="font-mono">export</span> /{" "}
                  <span className="font-mono">mkdir</span> lines are fine). On a VM these are exported before the run and
                  absolute-path values are auto-created; on RunPod they&apos;re passed to the pod.
                  {Object.keys(parseEnvVars(envText)).length > 0 && (
                    <>
                      {" "}· parsed:{" "}
                      <span className="font-mono">{Object.keys(parseEnvVars(envText)).join(", ")}</span>
                    </>
                  )}
                </p>
              </div>
            </div>
          </SectionCard>
          )}

          {/* HuggingFace token for gated models — a global secret (recommended,
              rotates without editing this run) or a pasted token for this run.
              Hidden for ingress: no weights are downloaded, so it doesn't apply. */}
          {target !== "ingress" && (
          <SectionCard
            icon={<KeyRound className="h-4 w-4" />}
            title="HuggingFace token"
            description="For gated models. Use a global secret or paste a token. Leave on a secret to rotate without editing this benchmark."
          >
            <div className="space-y-2">
              <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
                {(["secret", "paste"] as const).map((src) => (
                  <button
                    key={src}
                    type="button"
                    onClick={() => setHfSource(src)}
                    className={
                      "rounded px-2.5 py-1 transition-colors " +
                      (hfSource === src ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")
                    }
                  >
                    {src === "secret" ? "Global secret" : "Paste a token"}
                  </button>
                ))}
              </div>

              {hfSource === "secret" ? (
                secretKeys.length > 0 ? (
                  <div className="space-y-1.5">
                    <Label htmlFor="bench-hf-secret" className="text-xs uppercase tracking-wide text-muted-foreground">Global secret</Label>
                    <Select value={hfTokenSecret} onValueChange={setHfTokenSecret}>
                      <SelectTrigger id="bench-hf-secret">
                        <SelectValue placeholder="Select a secret (e.g. HF_TOKEN)" />
                      </SelectTrigger>
                      <SelectContent>
                        {secretKeys.map((k) => (
                          <SelectItem key={k} value={k} className="font-mono text-xs">{k}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      Resolved from{" "}
                      <a href="/admin/secrets" className="underline underline-offset-2 hover:text-foreground">Secrets</a>{" "}
                      at launch and injected as <span className="font-mono">HF_TOKEN</span>.
                    </p>
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    No global secrets yet. Add one under{" "}
                    <a href="/admin/secrets" className="underline underline-offset-2 hover:text-foreground">Secrets</a>{" "}
                    (e.g. <span className="font-mono">HF_TOKEN</span>), then pick it here — or switch to{" "}
                    <span className="font-medium">Paste a token</span>.
                  </p>
                )
              ) : (
                <div className="space-y-1.5">
                  <Label htmlFor="bench-hf-token" className="text-xs uppercase tracking-wide text-muted-foreground">Token</Label>
                  <Input
                    id="bench-hf-token"
                    type="password"
                    autoComplete="off"
                    value={hfToken}
                    onChange={(e) => setHfToken(e.target.value)}
                    placeholder="hf_..."
                    className="font-mono text-xs"
                  />
                  <p className="text-xs text-muted-foreground">
                    Sent with this run as <span className="font-mono">HF_TOKEN</span> (stored with the benchmark&apos;s env).
                    Prefer a global secret for shared / rotating tokens.
                  </p>
                </div>
              )}
            </div>
          </SectionCard>
          )}

          {/* Storage — where this run's logs.txt + result files land. Required;
              only enabled S3 storages are eligible (HF storages can't hold the
              raw log/result objects). */}
          <SectionCard
            icon={<Database className="h-4 w-4" />}
            title="Storage"
            description="S3 bucket the run's logs and metrics are written to."
          >
            <div className="space-y-1.5">
              <Label htmlFor="bench-storage" className="text-xs uppercase tracking-wide text-muted-foreground">Storage</Label>
              {!hasStorage ? (
                <p className="text-xs text-muted-foreground">
                  No S3 storage configured. Add one at{" "}
                  <a href="/storage/new" className="underline underline-offset-2 hover:text-foreground">
                    Storage → New storage
                  </a>
                  .
                </p>
              ) : (
                <Select value={storageId} onValueChange={setStorageId}>
                  <SelectTrigger id="bench-storage">
                    <SelectValue placeholder="Pick a storage…" />
                  </SelectTrigger>
                  <SelectContent>
                    {eligibleStorages.map((s) => (
                      <SelectItem key={s.id} value={s.id}>
                        {s.name}
                        {s.bucket ? ` · s3://${s.bucket}${s.prefix ? `/${s.prefix.replace(/^\/+|\/+$/g, "")}` : ""}` : ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
              <p className="text-xs text-muted-foreground">
                logs.txt + result files are written under{" "}
                <span className="font-mono">&lt;bucket&gt;/&lt;prefix&gt;/benchmarks/&lt;id&gt;/</span>.
                Manage backends under <a href="/storage" className="underline underline-offset-2 hover:text-foreground">Storage</a>.
              </p>
            </div>
          </SectionCard>

          {/* Visibility — platform-level public/private flag (NOT part of the
              benchmaq config YAML, so it's form-only). Mirrors the post-creation
              "Make public" toggle: public = read-only to every logged-in user. */}
          <SectionCard
            icon={<Globe className="h-4 w-4" />}
            title="Visibility"
            description="Who can see this run. You can change it later from the benchmark menu."
          >
            <label className="flex cursor-pointer items-start gap-2.5 rounded-md border border-border bg-muted/30 px-3 py-2.5 text-sm hover:bg-muted/50">
              <input
                type="checkbox"
                checked={isPublic}
                onChange={(e) => setIsPublic(e.target.checked)}
                className="mt-0.5 h-4 w-4 shrink-0 cursor-pointer accent-primary"
              />
              <div className="min-w-0">
                <div className="font-medium">Make public (read-only)</div>
                <div className="text-xs text-muted-foreground">
                  Off (default) keeps the run private — only you and admins can
                  see it. On shares it read-only with every logged-in user: they
                  can view the results but can&apos;t edit, delete, or re-run it.
                </div>
              </div>
            </label>
          </SectionCard>

          {/* Container image — picks the CUDA / pytorch baseline on the pod.
              VMs come with their own preinstalled environment, so we hide this
              when running on bare metal. */}
          {target === "cloud" && (
          <SectionCard
            icon={<Box className="h-4 w-4" />}
            title="Container"
            description="Base image the RunPod pod boots from. CUDA version must match what your model needs."
          >
            <ContainerImagePicker
              value={form.container_image}
              onChange={(v) => field("container_image", v)}
            />
            {/* CUDA pre-flight — sits below the image picker since it's a
                compatibility check on the selected image. Cloud-only; VMs use
                their host driver directly. */}
            <div className="mt-4">
              <CudaPreflightPanel
                image={
                  mode === "form"
                    ? form.container_image
                    : (extractImageFromYaml(yamlBuf) ?? form.container_image)
                }
              />
            </div>
          </SectionCard>
          )}

          {/* Engine runtime — what gets installed on the pod. Ingress hits an
              already-running server, so nothing is installed → hidden. */}
          {target !== "ingress" && (
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
                  placeholder="0.23.0"
                  disabled={!!form.vllm_install_args.trim()}
                />
              </FieldWrap>
              <FieldWrap
                label="Custom fork / install args"
                hint="A full `uv pip install` arg string — overrides the version. For a git fork or nightly; a leading VLLM_USE_PRECOMPILED=1 installs precompiled binaries. Works on both VM and RunPod targets."
                wide
              >
                <Input
                  className="font-mono"
                  value={form.vllm_install_args}
                  onChange={(e) => field("vllm_install_args", e.target.value)}
                  placeholder="VLLM_USE_PRECOMPILED=1 git+https://github.com/owner/vllm-fork@ref --torch-backend=auto"
                />
              </FieldWrap>
              <FieldWrap
                label="Presets"
                hint="One-click forks — fills the install args above and adds the matching serve flag to Advanced args."
                wide
              >
                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      field("vllm_install_args", composeForkArgs(GEMMA4_FA4_FORK_URL, GEMMA4_FA4_REF, true));
                      field("vllm_version", "");
                      if (!form.extra_args_raw.includes("FLASH_ATTN_CUTE")) {
                        field(
                          "extra_args_raw",
                          (form.extra_args_raw.trim() ? form.extra_args_raw.trim() + " " : "") + GEMMA4_FA4_BACKEND,
                        );
                      }
                    }}
                  >
                    Gemma-4 FA4 fork
                  </Button>
                  {form.vllm_install_args.trim() ? (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => field("vllm_install_args", "")}
                    >
                      Clear fork
                    </Button>
                  ) : null}
                </div>
              </FieldWrap>
            </Grid>
          </SectionCard>
          )}

          {/* Model + Serve — for ingress the model name + URL live in the
              Endpoint card above, and there's no serve block to configure. */}
          {target !== "ingress" && (
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
          )}

          {/* Bench */}
          <SectionCard
            icon={<Gauge className="h-4 w-4" />}
            title={form.bench_type === "accuracy" ? "Accuracy eval" : "Workload"}
            description={
              form.bench_type === "accuracy"
                ? "Serves the model, sends a dataset's questions, scores the answers. Reports accuracy AND a decode tok/s — run several configs to plot IQ vs speed."
                : "What benchmaq fires at the engine. Use the Sweep pill →  to cross-product input length × concurrency."
            }
            action={
              form.bench_type === "speed" ? (
                <SweepToggle
                  on={form.sweep_mode}
                  onChange={(v) => field("sweep_mode", v)}
                  runs={totalRuns(form)}
                />
              ) : null
            }
          >
            <div className="mb-5">
              <FieldWrap
                label="Benchmark type"
                hint="Speed measures throughput & latency. Accuracy scores the model on GSM8K / multilingual MMLU (and still reports tok/s, so multiple configs plot IQ vs speed)."
                wide
              >
                <div className="inline-flex rounded-lg border border-border p-0.5">
                  {(["speed", "accuracy"] as const).map((t) => (
                    <button
                      key={t}
                      type="button"
                      onClick={() => field("bench_type", t)}
                      className={cn(
                        "rounded-md px-4 py-1.5 text-sm font-medium capitalize transition-colors",
                        form.bench_type === t
                          ? "bg-primary text-primary-foreground"
                          : "text-muted-foreground hover:text-foreground",
                      )}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </FieldWrap>
            </div>
            {form.bench_type === "accuracy" ? (
              <div className="space-y-5">
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <ToggleRow
                    label="GSM8K"
                    hint="Grade-school math reasoning. Flexible last-number match."
                    checked={form.acc_gsm8k}
                    onChange={(v) => field("acc_gsm8k", v)}
                  />
                  <ToggleRow
                    label="Multilingual MMLU"
                    hint="openai/MMMLU — translated MMLU across languages. Single-letter multiple choice."
                    checked={form.acc_mmmlu}
                    onChange={(v) => field("acc_mmmlu", v)}
                  />
                  <ToggleRow
                    label="Function calling (TaaS) — hard"
                    hint="Scicom-intl/Function-Call-TaaS: 100 multi-turn (14–17 turn) tool-calling conversations, Manglish/Tamil/Chinese. Scored by tool-call F1. Heavy — lower Samples to cap conversations."
                    checked={form.acc_funccall}
                    onChange={(v) => field("acc_funccall", v)}
                  />
                </div>
                <Grid>
                  <FieldWrap
                    label="Samples"
                    hint="Questions per dataset (MMLU splits this across its languages). Lower = faster."
                  >
                    <NumberField
                      min={1}
                      value={form.acc_limit}
                      onChange={(v) => field("acc_limit", v)}
                    />
                  </FieldWrap>
                  <FieldWrap label="Concurrency" hint="In-flight requests while scoring.">
                    <NumberField
                      min={1}
                      value={form.acc_concurrency}
                      onChange={(v) => field("acc_concurrency", v)}
                    />
                  </FieldWrap>
                  {form.acc_mmmlu && (
                    <FieldWrap
                      label="MMLU languages"
                      hint="openai/MMMLU config codes, comma-separated."
                      wide
                    >
                      <Input
                        className="font-mono"
                        value={form.acc_mmmlu_langs}
                        onChange={(e) => field("acc_mmmlu_langs", e.target.value)}
                        placeholder="FR_FR, DE_DE, ZH_CN"
                      />
                    </FieldWrap>
                  )}
                </Grid>
              </div>
            ) : (
              <>
            <div className="mb-5">
              <ToggleRow
                label="Warm up each run"
                hint="Adds vLLM --num-warmups (= each run's own concurrency) so every run fires one full warm-up wave at its exact shape before measuring. vLLM excludes these from the metrics — kills first-request cold start (cuBLAS/NCCL/torch.compile) without a separate row."
                checked={form.warmup}
                onChange={(v) => field("warmup", v)}
              />
            </div>
            {form.sweep_mode ? (
              <Grid>
                <FieldWrap
                  label="Input lengths"
                  hint="Comma-separated list of token counts. Each value × each concurrency = one bench run."
                  wide
                >
                  <Input
                    className="font-mono"
                    placeholder="128, 512, 2048, 4096, 8192, 16384"
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
                    placeholder="10, 50, 100"
                    value={form.concurrencies_csv}
                    onChange={(e) => field("concurrencies_csv", e.target.value)}
                  />
                  <SweepChips values={parseCsvInts(form.concurrencies_csv)} />
                </FieldWrap>
                <FieldWrap label="Output length" hint="Same for every run.">
                  <NumberField
                    min={1}
                    value={form.output_len}
                    onChange={(v) => field("output_len", v)}
                  />
                </FieldWrap>
                <FieldWrap label="Num prompts" hint="Total requests per run.">
                  <NumberField
                    min={1}
                    value={form.num_prompts}
                    onChange={(v) => field("num_prompts", v)}
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
                  <NumberField
                    min={1}
                    value={form.input_len}
                    onChange={(v) => field("input_len", v)}
                  />
                </FieldWrap>
                <FieldWrap label="Output length" hint="Tokens to generate per request.">
                  <NumberField
                    min={1}
                    value={form.output_len}
                    onChange={(v) => field("output_len", v)}
                  />
                </FieldWrap>
                <FieldWrap label="Num prompts" hint="Total requests in this run.">
                  <NumberField
                    min={1}
                    value={form.num_prompts}
                    onChange={(v) => field("num_prompts", v)}
                  />
                </FieldWrap>
                <FieldWrap
                  label="Max concurrency"
                  hint="In-flight requests. Tune for throughput."
                >
                  <NumberField
                    min={1}
                    value={form.max_concurrency}
                    onChange={(v) => field("max_concurrency", v)}
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
              </>
            )}
          </SectionCard>

          {/* YAML preview — plain code block, not a terminal. Controlled
              disclosure rather than a native <details>: when a <details> is
              collapsed, recent Chrome builds still lay out its content and
              reserve the capped-height <pre> at its full *intrinsic* height
              (the whole rendered config, ignoring max-h/overflow). That leaked
              ~a screenful of dead scroll space below the action bar on the Form
              tab. Rendering the <pre> only when open removes the hidden-but-
              sized element entirely. */}
          <div className="rounded-lg border border-border">
            <button
              type="button"
              onClick={() => setYamlPreviewOpen((v) => !v)}
              aria-expanded={yamlPreviewOpen}
              className="flex w-full cursor-pointer items-center justify-between gap-2 px-4 py-3 text-sm font-medium hover:bg-muted/40"
            >
              <div className="flex items-center gap-2">
                <ChevronRight
                  className={cn(
                    "h-4 w-4 text-muted-foreground transition-transform",
                    yamlPreviewOpen && "rotate-90",
                  )}
                />
                <FileCode2 className="h-4 w-4 text-muted-foreground" />
                YAML preview
                <Badge variant="secondary" className="text-[10px]">
                  read-only
                </Badge>
              </div>
              <Info className="h-3.5 w-3.5 text-muted-foreground" />
            </button>
            {yamlPreviewOpen && (
              <pre className="max-h-80 overflow-auto rounded-b-lg border-t border-border bg-muted/40 px-4 py-3 font-mono text-xs leading-relaxed text-foreground">
                {formYaml}
              </pre>
            )}
          </div>
        </TabsContent>

        <TabsContent value="yaml" className="mt-4 !flex-none">
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

      {/* Action bar — sticky (FormFooter), so submit + errors stay visible. */}
      <FormFooter
        error={submitError}
        hint={
          !hasStorage ? (
            <>
              Add an S3 storage at{" "}
              <a href="/storage/new" className="underline underline-offset-2 hover:text-foreground">
                Storage → New storage
              </a>{" "}
              to create a benchmark.
            </>
          ) : target === "vm"
            ? "Runs on your registered VM via SSH. No cloud pod is created."
            : "A new RunPod pod will be created and torn down automatically."
        }
      >
        <Button
          type="button"
          variant="outline"
          onClick={() => router.push("/benchmark")}
        >
          Cancel
        </Button>
        <Button
          type="submit"
          disabled={submitting || !hasStorage}
          className="min-w-36"
          title={!hasStorage ? "Add an S3 storage backend first" : undefined}
        >
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
      </FormFooter>

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
            <Label htmlFor="tplName" className="text-xs uppercase tracking-wide text-muted-foreground">Template name</Label>
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
    </FormShell>
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
    // data-form-section feeds the FormShell scrollspy rail; scroll-mt keeps the
    // heading visible after a rail jump.
    <Card data-form-section={title} className="scroll-mt-6">
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
            <KebabField
              label="port"
              hint="HTTP port vLLM serves on. Default 8000."
            >
              <Input
                type="text"
                inputMode="numeric"
                value={form.port}
                onChange={(e) => setField("port", e.target.value)}
                placeholder="8000"
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
      <Label className="text-xs uppercase tracking-wide text-muted-foreground">
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
  extra,
  children,
}: {
  label: string;
  hint?: string;
  wide?: boolean;
  extra?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className={cn("space-y-1.5", wide ? "sm:col-span-2 lg:col-span-2" : "")}>
      <div className="flex items-center justify-between gap-2">
        <Label className="text-xs uppercase tracking-wide text-muted-foreground">{label}</Label>
        {extra}
      </div>
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

// Inline availability row shown under the VM provider dropdown. SSHes the VM
// and reports per-GPU memory free / total — runpod-equivalent for bare metal.
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
