"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { Check, Copy, KeyRound, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { gateway } from "@/lib/gateway";

/**
 * API reference — three-column-ish layout modelled on the scicom-aura docs:
 * a searchable endpoint nav on the left, and per-endpoint sections in the
 * main column that each split into docs (left) + request/response samples
 * (right). Samples track the real gateway route shapes; if you change a
 * route's wire shape, update its entry in ENDPOINTS below.
 *
 * The base URL is the gateway (NEXT_PUBLIC_GATEWAY_URL), not the web origin —
 * every call hits the FastAPI gateway directly with a Bearer API key.
 */

function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      className="absolute right-1.5 top-1.5 opacity-50 hover:opacity-100"
      onClick={() => {
        navigator.clipboard?.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      title={copied ? "Copied" : "Copy"}
    >
      {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
    </Button>
  );
}

function CodeBlock({ children, label }: { children: string; label?: string }) {
  return (
    <div className="relative rounded-md border border-border bg-muted p-3">
      {label && (
        <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          {label}
        </p>
      )}
      <pre className="overflow-x-auto pr-8 font-mono text-xs leading-relaxed text-foreground/90">
        {children}
      </pre>
      <CopyBtn text={children} />
    </div>
  );
}

type Method = "GET" | "POST" | "PATCH" | "DELETE";

function MethodBadge({ method, size = "sm" }: { method: Method; size?: "sm" | "xs" }) {
  const colour =
    method === "GET" ? "bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-200"
    : method === "POST" ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200"
    : method === "PATCH" ? "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-200"
    : "bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-200";
  const sizing = size === "xs" ? "h-4 px-1 text-[9px]" : "h-5 px-1.5 text-[10px]";
  return (
    <span className={"inline-flex items-center rounded font-mono font-semibold tracking-wider " + sizing + " " + colour}>
      {method}
    </span>
  );
}

function StatusBadge({ code, label }: { code: number; label: string }) {
  const ok = code >= 200 && code < 300;
  const colour = ok
    ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200"
    : "bg-rose-100 text-rose-800 dark:bg-rose-900/40 dark:text-rose-200";
  return (
    <span className={"inline-flex h-5 items-center rounded px-1.5 font-mono text-[10px] font-semibold " + colour}>
      {code} {label}
    </span>
  );
}

interface Endpoint {
  id: string;
  group: string;
  method: Method;
  path: string;
  title: string;
  description: React.ReactNode;
  parameters?: Array<{ name: string; in: "query" | "body" | "path"; type: string; required?: boolean; doc: React.ReactNode }>;
  request: { sample: string };
  responses: Array<{ code: number; codeLabel: string; doc?: React.ReactNode; sample: string }>;
}

interface Group {
  id: string;
  title: string;
  blurb?: React.ReactNode;
}

const GROUPS: Group[] = [
  { id: "apikeys", title: "API keys", blurb: <>Mint and revoke the long-lived <code>sgpu_…</code> bearer tokens these endpoints authenticate with.</> },
  { id: "serverless", title: "Serverless endpoints", blurb: <>Create and manage autoscaling vLLM endpoints. Each endpoint scales to zero when idle.</> },
  { id: "inference", title: "Inference", blurb: <>Send requests to a deployed endpoint — OpenAI-compatible, or the native sync / streaming routes.</> },
  { id: "benchmarks", title: "Benchmarks", blurb: <>Run llm-benchmaq throughput/latency sweeps on a RunPod pod or a registered VM. Logs + results land in a storage backend.</> },
  { id: "datasets", title: "Datasets", blurb: <>Audio + transcription datasets — backed by an uploaded metadata file, an S3 metadata file, or a HuggingFace repo. Browse rows, set the audio/transcription columns (per-split for HF repos with differing schemas), and transform a zip-of-audio repo into one with a real <code>audio</code> column (pushed to HF or materialised to S3).</> },
  { id: "autotrain", title: "Autotrain", blurb: <>Fine-tune Whisper models on your datasets — SSH-orchestrated on a RunPod pod or a registered VM. Logs stream live; checkpoints + metrics land in a storage backend.</> },
  { id: "compute", title: "Compute pods", blurb: <>Raw RunPod pods with SSH + JupyterLab. Creation may require admin approval.</> },
  { id: "storage", title: "Storage", blurb: <>S3 / HuggingFace backends the platform writes to (dataset files, benchmark logs, inference logs). Writes are admin-only.</> },
  { id: "providers", title: "GPU providers", blurb: <>Registered VMs / RunPod / Prime Intellect accounts that endpoints, benchmarks, and compute can target. Writes are admin-only.</> },
];

const ENDPOINTS: Endpoint[] = [
  // ───── API keys ─────
  {
    id: "list-api-keys",
    group: "apikeys",
    method: "GET",
    path: "/api-keys",
    title: "List your API keys",
    description: <>Returns all of your non-revoked keys (newest first). The secret is never returned — only the <code>prefix</code> for display.</>,
    request: { sample: `curl -s "$SGPU/api-keys" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [
      {
        code: 200,
        codeLabel: "OK",
        sample: `[
  {
    "id": "ak-9f1d83c49a2e",
    "name": "ci-bot",
    "prefix": "sgpu_AbCd12",
    "created_at": "2026-05-29T03:21:08+00:00",
    "last_used_at": "2026-05-29T04:10:55+00:00"
  }
]`,
      },
    ],
  },
  {
    id: "create-api-key",
    group: "apikeys",
    method: "POST",
    path: "/api-keys",
    title: "Create an API key",
    description: <>Mints a new key. The full plaintext <code>key</code> is returned <b>once</b> in this response — store it now. The key inherits your role + section access. You can hold as many keys as you like.</>,
    parameters: [{ name: "name", in: "body", type: "string", required: true, doc: "A label for the key, e.g. ci-bot, laptop." }],
    request: {
      sample: `curl -s -X POST "$SGPU/api-keys" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{"name": "ci-bot"}'`,
    },
    responses: [
      {
        code: 200,
        codeLabel: "OK",
        sample: `{
  "id": "ak-9f1d83c49a2e",
  "name": "ci-bot",
  "prefix": "sgpu_AbCd12",
  "created_at": "2026-05-29T03:21:08+00:00",
  "last_used_at": null,
  "key": "sgpu_AbCd12EfGh34IjKl56MnOp78"
}`,
      },
    ],
  },
  {
    id: "revoke-api-key",
    group: "apikeys",
    method: "DELETE",
    path: "/api-keys/:id",
    title: "Revoke an API key",
    description: <>Revokes a key by id. Takes effect immediately — in-flight requests using it start failing with 401.</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Key id (the ak-… value, not the secret)." }],
    request: { sample: `curl -s -X DELETE "$SGPU/api-keys/ak-9f1d83c49a2e" \\
  -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK", sample: `{ "ok": true, "id": "ak-9f1d83c49a2e" }` }],
  },

  // ───── Serverless endpoints ─────
  {
    id: "create-app",
    group: "serverless",
    method: "POST",
    path: "/apps",
    title: "Create a serverless endpoint",
    description: (
      <>
        <p>Deploys an autoscaling vLLM endpoint. The endpoint name becomes the route segment for inference (<code>/run/&lt;name&gt;</code>) and the OpenAI <code>model</code> value for single-model endpoints.</p>
        <p className="mt-2 text-xs text-muted-foreground">Set <code>provider_id</code> to a VM provider to run on bare metal; omit it (or pass a RunPod provider) for cloud. <code>visible_devices</code> pins GPU indices on a VM.</p>
      </>
    ),
    parameters: [
      { name: "name", in: "body", type: "string", required: true, doc: "Unique endpoint name (slug)." },
      { name: "model", in: "body", type: "string", doc: "HuggingFace repo id. Required for single-model." },
      { name: "gpu", in: "body", type: "string", required: true, doc: 'GPU type (e.g. "H100") or "vm" for a VM provider.' },
      { name: "gpu_count", in: "body", type: "number", doc: "GPUs per worker. Default 1." },
      { name: "autoscaler", in: "body", type: "object", doc: "{ max_containers, idle_timeout_s }. idle_timeout_s=0 keeps the worker always-on." },
      { name: "provider_id", in: "body", type: "string", doc: "Provider row id (vm / runpod / pi). Omit for the gateway default." },
      { name: "mode", in: "body", type: '"single" | "multi"', doc: "multi = a vLLM fleet on one VM; requires a vm provider + models[]." },
      { name: "models", in: "body", type: "Array<{model,tp,extra_args}>", doc: "Multi-model members." },
      { name: "visible_devices", in: "body", type: "string", doc: 'VM-only GPU pin, e.g. "0,1,2,3".' },
      { name: "vllm_args", in: "body", type: "string", doc: "Extra vLLM CLI flags appended verbatim." },
      { name: "env_vars", in: "body", type: "object", doc: "Env applied to every vLLM process (HF_HOME, cache dirs, …)." },
    ],
    request: {
      sample: `curl -s -X POST "$SGPU/apps" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{
    "name": "my-endpoint",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "gpu": "H100", "gpu_count": 1,
    "autoscaler": {"max_containers": 1, "idle_timeout_s": 0}
  }'`,
    },
    responses: [
      { code: 200, codeLabel: "OK", sample: `{ "app_id": "my-endpoint", "url": "/run/my-endpoint" }` },
      { code: 503, codeLabel: "Unavailable", doc: "The provider rejected the spec at the create-time provision (out of stock / GPU not on this tier). Body carries gpu, gpu_count, reason.", sample: `{ "detail": { "error": "GPU not available", "gpu": "H100", "gpu_count": 1, "reason": "no instances available" } }` },
    ],
  },
  {
    id: "list-apps",
    group: "serverless",
    method: "GET",
    path: "/apps",
    title: "List endpoints",
    description: <>Lists your endpoints. <code>scope=all</code> (admin) returns every endpoint.</>,
    parameters: [{ name: "scope", in: "query", type: '"mine" | "all"', doc: "Default mine." }],
    request: { sample: `curl -s "$SGPU/apps?scope=mine" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [
      {
        code: 200,
        codeLabel: "OK",
        sample: `[
  {
    "app_id": "my-endpoint",
    "name": "my-endpoint",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "gpu": "H100", "gpu_count": 1,
    "mode": "single",
    "autoscaler": {"max_containers": 1, "tasks_per_container": 30, "idle_timeout_s": 0},
    "provider_id": null,
    "created_at": "2026-05-29T03:21:08+00:00",
    "owner": "admin"
  }
]`,
      },
    ],
  },
  {
    id: "app-status",
    group: "serverless",
    method: "GET",
    path: "/apps/:id/status",
    title: "Endpoint status",
    description: <>Live queue depth, worker count, and the last provisioning error (handy while a worker cold-starts).</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Endpoint name." }],
    request: { sample: `curl -s "$SGPU/apps/my-endpoint/status" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [
      {
        code: 200,
        codeLabel: "OK",
        sample: `{
  "app_id": "my-endpoint",
  "queue_len": 0,
  "workers": 1,
  "last_provision_error": null,
  "last_provision_error_at": null,
  "provision_cooldown_remaining_s": 0
}`,
      },
    ],
  },
  {
    id: "update-autoscaler",
    group: "serverless",
    method: "PATCH",
    path: "/apps/:id/autoscaler",
    title: "Update autoscaler / vLLM args",
    description: <>Patch scaling knobs or <code>vllm_args</code> in place. Takes effect on the next worker spawn.</>,
    parameters: [
      { name: "id", in: "path", type: "string", required: true, doc: "Endpoint name." },
      { name: "max_containers", in: "body", type: "number", doc: "" },
      { name: "idle_timeout_s", in: "body", type: "number", doc: "0 = always-on." },
      { name: "vllm_args", in: "body", type: "string", doc: "" },
    ],
    request: {
      sample: `curl -s -X PATCH "$SGPU/apps/my-endpoint/autoscaler" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{"idle_timeout_s": 300}'`,
    },
    responses: [{ code: 200, codeLabel: "OK", sample: `{ "app_id": "my-endpoint", "...": "full AppRecord" }` }],
  },
  {
    id: "delete-app",
    group: "serverless",
    method: "DELETE",
    path: "/apps/:id",
    title: "Delete an endpoint",
    description: <>Drains and tears down all workers, then removes the endpoint.</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Endpoint name." }],
    request: { sample: `curl -s -X DELETE "$SGPU/apps/my-endpoint" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK", sample: `{ "ok": true, "app_id": "my-endpoint", "drained_workers": 1 }` }],
  },

  // ───── Inference ─────
  {
    id: "chat-completions",
    group: "inference",
    method: "POST",
    path: "/v1/chat/completions",
    title: "Chat completions (OpenAI-compatible)",
    description: (
      <>
        <p>Drop-in OpenAI Chat Completions. Set <code>model</code> to the endpoint name (single-model) or to a member model name (multi-model). Body is forwarded to vLLM verbatim; <code>stream: true</code> returns SSE.</p>
        <p className="mt-2 text-xs text-muted-foreground">Point any OpenAI SDK at <code>{`<base>/v1`}</code> with your key.</p>
      </>
    ),
    parameters: [
      { name: "model", in: "body", type: "string", required: true, doc: "Endpoint name (single) or member model (multi)." },
      { name: "messages", in: "body", type: "object[]", required: true, doc: "OpenAI chat messages." },
      { name: "stream", in: "body", type: "boolean", doc: "true → text/event-stream." },
    ],
    request: {
      sample: `curl -s -X POST "$SGPU/v1/chat/completions" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{"model": "my-endpoint", "messages": [{"role":"user","content":"hi"}]}'`,
    },
    responses: [
      {
        code: 200,
        codeLabel: "OK",
        sample: `{
  "id": "chatcmpl-…",
  "object": "chat.completion",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
}`,
      },
      { code: 504, codeLabel: "Timeout", doc: "No completion within 60s — the worker is probably cold-starting. Retry or use stream:true.", sample: `{ "detail": { "error": "no completion in 60s — worker probably cold-starting", "request_id": "req-…" } }` },
    ],
  },
  {
    id: "completions",
    group: "inference",
    method: "POST",
    path: "/v1/completions",
    title: "Completions (OpenAI-compatible)",
    description: (
      <>
        <p>Legacy OpenAI text completions (prompt-in, text-out). Body is forwarded to vLLM verbatim; <code>stream: true</code> returns SSE. Prefer <code>/v1/chat/completions</code> for instruct / chat models.</p>
        <p className="mt-2 text-xs text-muted-foreground">Scope to one endpoint without the <code>model</code> field via <code>{`<base>/<endpoint>/v1/completions`}</code>.</p>
      </>
    ),
    parameters: [
      { name: "model", in: "body", type: "string", required: true, doc: "Endpoint name (single) or member model (multi)." },
      { name: "prompt", in: "body", type: "string | string[]", required: true, doc: "Prompt(s) to complete." },
      { name: "max_tokens", in: "body", type: "integer", doc: "Upper bound on generated tokens." },
      { name: "stream", in: "body", type: "boolean", doc: "true → text/event-stream." },
    ],
    request: {
      sample: `curl -s -X POST "$SGPU/v1/completions" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{"model": "my-endpoint", "prompt": "The capital of France is", "max_tokens": 16}'`,
    },
    responses: [
      {
        code: 200,
        codeLabel: "OK",
        sample: `{
  "id": "cmpl-…",
  "object": "text_completion",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "choices": [{"index": 0, "text": " Paris.", "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
}`,
      },
    ],
  },
  {
    id: "list-models",
    group: "inference",
    method: "GET",
    path: "/v1/models",
    title: "List models",
    description: (
      <>
        <p>OpenAI-compatible model list. A single-model endpoint returns its one served model; a multi-model fleet lists every member you can target in the <code>model</code> field.</p>
        <p className="mt-2 text-xs text-muted-foreground">The scoped form <code>{`<base>/<endpoint>/v1/models`}</code> lists just that endpoint&apos;s models.</p>
      </>
    ),
    request: {
      sample: `curl -s "$SGPU/my-endpoint/v1/models" \\
  -H "Authorization: Bearer $SGPU_API_KEY"`,
    },
    responses: [
      {
        code: 200,
        codeLabel: "OK",
        sample: `{
  "object": "list",
  "data": [
    {"id": "Qwen/Qwen2.5-7B-Instruct", "object": "model", "owned_by": "sgpu"}
  ]
}`,
      },
    ],
  },
  {
    id: "run-sync",
    group: "inference",
    method: "POST",
    path: "/run/:app_id",
    title: "Run (native, async-enqueue)",
    description: <>Enqueues a raw request for the endpoint and returns a <code>request_id</code> + <code>poll_url</code> to fetch the result. Use <code>/v1/chat/completions</code> for the synchronous OpenAI path instead.</>,
    parameters: [
      { name: "app_id", in: "path", type: "string", required: true, doc: "Endpoint name." },
      { name: "(body)", in: "body", type: "object", doc: "Forwarded to the worker as the job payload." },
    ],
    request: {
      sample: `curl -s -X POST "$SGPU/run/my-endpoint" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{"prompt": "hello"}'`,
    },
    responses: [{ code: 200, codeLabel: "OK", sample: `{ "request_id": "req-1a2b3c", "poll_url": "/results/req-1a2b3c" }` }],
  },
  {
    id: "stream",
    group: "inference",
    method: "POST",
    path: "/stream/:app_id",
    title: "Stream (native SSE)",
    description: <>Server-sent-events stream of worker output chunks; the final event is <code>{`{"done": true}`}</code>.</>,
    parameters: [{ name: "app_id", in: "path", type: "string", required: true, doc: "Endpoint name." }],
    request: {
      sample: `curl -N -X POST "$SGPU/stream/my-endpoint" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{"prompt": "hello"}'`,
    },
    responses: [{ code: 200, codeLabel: "OK · text/event-stream", sample: `data: {"index": 0, "delta": "Hel"}
data: {"index": 1, "delta": "lo"}
data: {"done": true}` }],
  },
  {
    id: "metrics",
    group: "inference",
    method: "GET",
    path: "/metrics",
    title: "Metrics (Prometheus)",
    description: (
      <>
        <p>Prometheus exposition for the gateway itself. For per-worker vLLM metrics — queue depth, throughput, GPU KV-cache usage — across the whole fleet, scrape <code>/metrics/workers</code>: it relabels each live worker&apos;s <code>/metrics</code> with its <code>app_id</code> + model and concatenates them.</p>
      </>
    ),
    request: {
      sample: `curl -s "$SGPU/metrics/workers" \\
  -H "Authorization: Bearer $SGPU_API_KEY"`,
    },
    responses: [
      {
        code: 200,
        codeLabel: "OK · text/plain",
        sample: `vllm:num_requests_running{app_id="my-endpoint",model="…"} 2
vllm:num_requests_waiting{app_id="my-endpoint",model="…"} 0
vllm:gpu_cache_usage_perc{app_id="my-endpoint",model="…"} 0.41`,
      },
    ],
  },
  {
    id: "health",
    group: "inference",
    method: "GET",
    path: "/health",
    title: "Health check",
    description: (
      <>
        <p>Liveness probe for the gateway — <code>200</code> when healthy. Each vLLM worker also exposes its own <code>/health</code> (sometimes <code>/healthz</code>) once it finishes loading; a cold-starting worker is unhealthy until the model is resident.</p>
      </>
    ),
    request: {
      sample: `curl -s -o /dev/null -w "%{http_code}\\n" "$SGPU/health"`,
    },
    responses: [
      { code: 200, codeLabel: "OK", doc: "Gateway is up and serving.", sample: `200` },
    ],
  },

  // ───── Benchmarks ─────
  {
    id: "create-benchmark",
    group: "benchmarks",
    method: "POST",
    path: "/benchmarks",
    title: "Create a benchmark",
    description: <>Runs <code>benchmaq</code> against the YAML config on the chosen provider. Logs + result files are written to the selected storage (<code>storage_id</code>).</>,
    parameters: [
      { name: "name", in: "body", type: "string", required: true, doc: "" },
      { name: "config_yaml", in: "body", type: "string", required: true, doc: "benchmaq config." },
      { name: "storage_id", in: "body", type: "string", doc: "S3 storage for logs + results (required by the web form)." },
      { name: "provider_id", in: "body", type: "string", doc: "VM or RunPod provider. Omit for default cloud." },
      { name: "visible_devices", in: "body", type: "string", doc: 'CUDA_VISIBLE_DEVICES pin, e.g. "4,5".' },
      { name: "env_vars", in: "body", type: "object", doc: "Extra env exported for the run." },
      { name: "cleanup_model", in: "body", type: "boolean", doc: "VM runs: rm the model after the run. Default true." },
    ],
    request: {
      sample: `curl -s -X POST "$SGPU/benchmarks" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{
    "name": "qwen-quick",
    "config_yaml": "benchmark:\\n  - name: qwen\\n    engine: vllm\\n    model:\\n      repo_id: Qwen/Qwen2.5-7B-Instruct",
    "provider_id": "<vm-or-runpod-provider-id>",
    "storage_id": "<s3-storage-id>",
    "visible_devices": "4,5"
  }'`,
    },
    responses: [
      {
        code: 200,
        codeLabel: "OK",
        sample: `{
  "id": "bench-1a2b3c4d",
  "name": "qwen-quick",
  "status": "queued",
  "s3_prefix": "benchmarks/bench-1a2b3c4d/",
  "provider_id": "prov-…",
  "storage_id": "store-…",
  "created_by": "admin",
  "created_at": "2026-05-29T03:21:08+00:00"
}`,
      },
    ],
  },
  {
    id: "list-benchmarks",
    group: "benchmarks",
    method: "GET",
    path: "/benchmarks",
    title: "List benchmarks",
    description: <>Your benchmark runs. <code>scope=all</code> (admin) returns everyone&apos;s.</>,
    parameters: [{ name: "scope", in: "query", type: '"mine" | "all"', doc: "Default mine." }],
    request: { sample: `curl -s "$SGPU/benchmarks?scope=mine" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK", sample: `[ { "id": "bench-1a2b3c4d", "name": "qwen-quick", "status": "done", "exit_code": 0, "...": "BenchmarkRecord" } ]` }],
  },
  {
    id: "benchmark-files",
    group: "benchmarks",
    method: "GET",
    path: "/benchmarks/:id/files",
    title: "List result files",
    description: <>Lists every file under the run&apos;s storage prefix with a presigned download URL.</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Benchmark id." }],
    request: { sample: `curl -s "$SGPU/benchmarks/bench-1a2b3c4d/files" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK", sample: `[
  { "name": "logs.txt", "size": 48213, "modified": "2026-05-29T03:48:31+00:00", "download_url": "https://…" },
  { "name": "result.json", "size": 1822, "modified": "2026-05-29T03:48:31+00:00", "download_url": "https://…" }
]` }],
  },
  {
    id: "terminate-benchmark",
    group: "benchmarks",
    method: "POST",
    path: "/benchmarks/:id/terminate",
    title: "Terminate a running benchmark",
    description: <>Stops the run, SSH-kills the remote process, and tears down the RunPod pod (cloud runs).</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Benchmark id." }],
    request: { sample: `curl -s -X POST "$SGPU/benchmarks/bench-1a2b3c4d/terminate" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK", sample: `{ "ok": true, "id": "bench-1a2b3c4d", "status": "cancelled" }` }],
  },

  // ───── Datasets ─────
  {
    id: "list-datasets",
    group: "datasets",
    method: "GET",
    path: "/v1/datasets",
    title: "List datasets",
    description: <>Your datasets, newest first. <code>scope=all</code> (admin) returns everyone&apos;s.</>,
    parameters: [{ name: "scope", in: "query", type: '"mine" | "all"', doc: "Default mine." }],
    request: { sample: `curl -s "$SGPU/v1/datasets?scope=mine" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [
      {
        code: 200,
        codeLabel: "OK",
        sample: `[
  {
    "id": "ds-1a2b3c4d",
    "name": "emgs-recording",
    "kind": "hf",
    "storage_id": "store-1a2b3c4d", "storage_name": "hf-token",
    "hf_repo": "Scicom-intl/emgs-recording-2025-10-13",
    "audio_field": "audio_filename", "transcription_field": "text",
    "split_fields": {"train": "text", "test": "after"},
    "audio_dataset_id": "ds-07b26489",
    "num_rows": 3058,
    "transform_status": "done",
    "created_by": "admin", "created_at": "2026-05-29T03:21:08+00:00"
  }
]`,
      },
    ],
  },
  {
    id: "create-dataset",
    group: "datasets",
    method: "POST",
    path: "/v1/datasets",
    title: "Create a dataset",
    description: (
      <>
        <p>Registers a dataset pointer. <code>kind</code> selects the source: <code>upload</code> (then POST a metadata file to <code>/upload</code>), <code>s3</code> (point at an existing <code>s3_metadata_uri</code>), or <code>hf</code> (an existing HuggingFace repo).</p>
        <p className="mt-2 text-xs text-muted-foreground"><code>storage_id</code> references a Storage row — <code>kind=s3</code> for upload/s3 datasets, <code>kind=huggingface</code> for hf (used to resolve the HF token).</p>
      </>
    ),
    parameters: [
      { name: "name", in: "body", type: "string", required: true, doc: "Dataset name." },
      { name: "kind", in: "body", type: '"upload" | "s3" | "hf"', doc: "Source type. Default upload." },
      { name: "storage_id", in: "body", type: "string", doc: "Storage row id (s3 backend, or hf token holder)." },
      { name: "s3_metadata_uri", in: "body", type: "string", doc: "kind=s3: s3://bucket/key of the metadata file." },
      { name: "hf_repo", in: "body", type: "string", doc: "kind=hf: owner/name of the source repo." },
      { name: "audio_prefix", in: "body", type: "string", doc: "Optional key prefix audio paths resolve against." },
      { name: "description", in: "body", type: "string", doc: "Optional." },
    ],
    request: {
      sample: `curl -s -X POST "$SGPU/v1/datasets" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{"name": "emgs-recording", "kind": "hf",
       "hf_repo": "Scicom-intl/emgs-recording-2025-10-13",
       "storage_id": "store-1a2b3c4d"}'`,
    },
    responses: [
      { code: 200, codeLabel: "OK", sample: `{ "id": "ds-1a2b3c4d", "name": "emgs-recording", "kind": "hf", "...": "DatasetRecord" }` },
      { code: 400, codeLabel: "Bad Request", doc: "Missing/invalid field — e.g. hf_repo required for kind=hf, or storage_id (an S3 storage) required for upload/s3.", sample: `{ "detail": "hf_repo (owner/name) is required for kind=hf" }` },
    ],
  },
  {
    id: "get-dataset",
    group: "datasets",
    method: "GET",
    path: "/v1/datasets/:id",
    title: "Get a dataset",
    description: <>Full record — source, column mapping, row count, and transform status.</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Dataset id (ds-…)." }],
    request: { sample: `curl -s "$SGPU/v1/datasets/ds-1a2b3c4d" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [
      { code: 200, codeLabel: "OK", sample: `{ "id": "ds-1a2b3c4d", "kind": "hf", "audio_field": "audio_filename", "transcription_field": "text", "...": "DatasetRecord" }` },
      { code: 403, codeLabel: "Forbidden", doc: "Not yours (and you're not admin).", sample: `{ "detail": "forbidden" }` },
    ],
  },
  {
    id: "update-dataset",
    group: "datasets",
    method: "PATCH",
    path: "/v1/datasets/:id",
    title: "Update columns / metadata",
    description: (
      <>
        <p>Patch the audio / transcription column mapping (and name, description, audio_prefix). For an HF source whose splits have different schemas, set <code>split_fields</code> to map each split&apos;s transcription column — pass <code>{`{}`}</code> to clear.</p>
      </>
    ),
    parameters: [
      { name: "id", in: "path", type: "string", required: true, doc: "Dataset id." },
      { name: "audio_field", in: "body", type: "string", doc: "Column holding the audio path/ref." },
      { name: "transcription_field", in: "body", type: "string", doc: "Default/output transcription column." },
      { name: "split_fields", in: "body", type: "object", doc: 'Per-split transcription columns, e.g. {"train":"text","test":"after"}.' },
      { name: "name / description / audio_prefix", in: "body", type: "string", doc: "Optional metadata edits." },
    ],
    request: {
      sample: `curl -s -X PATCH "$SGPU/v1/datasets/ds-1a2b3c4d" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{"audio_field": "audio_filename", "transcription_field": "text",
       "split_fields": {"train": "text", "test": "after"}}'`,
    },
    responses: [{ code: 200, codeLabel: "OK", sample: `{ "id": "ds-1a2b3c4d", "split_fields": {"train": "text", "test": "after"}, "...": "DatasetRecord" }` }],
  },
  {
    id: "dataset-preview",
    group: "datasets",
    method: "GET",
    path: "/v1/datasets/:id/preview",
    title: "Browse rows (paginated)",
    description: (
      <>
        <p>A page of rows with each audio reference resolved to a playable URL and the transcription resolved per the (per-split) column mapping. Returns the full <code>total</code> for pagination.</p>
        <p className="mt-2 text-xs text-muted-foreground">For HF sources, <code>split</code> selects which split to read; <code>splits</code> lists the available ones. Audio URLs point at the gateway proxy (<code>/v1/datasets/:id/audio</code>) so they stream same-origin.</p>
      </>
    ),
    parameters: [
      { name: "id", in: "path", type: "string", required: true, doc: "Dataset id." },
      { name: "offset", in: "query", type: "integer", doc: "Row offset. Default 0." },
      { name: "limit", in: "query", type: "integer", doc: "Rows per page (1–200). Default 20." },
      { name: "split", in: "query", type: "string", doc: "HF only: which split to read. Default the first." },
    ],
    request: { sample: `curl -s "$SGPU/v1/datasets/ds-1a2b3c4d/preview?offset=0&limit=20&split=train" \\
  -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [
      {
        code: 200,
        codeLabel: "OK",
        sample: `{
  "audio_field": "audio_filename",
  "transcription_field": "text",
  "offset": 0, "limit": 20, "total": 3000,
  "split": "train", "splits": ["test", "train"],
  "rows": [
    {
      "audio_url": "/v1/datasets/ds-1a2b3c4d/audio?src=https%3A%2F%2F…",
      "transcription": "welcome to the contact centre…",
      "audio_filename": "clip-0001.mp3", "text": "welcome to the contact centre…"
    }
  ]
}`,
      },
    ],
  },
  {
    id: "dataset-splits",
    group: "datasets",
    method: "GET",
    path: "/v1/datasets/:id/splits",
    title: "List splits + columns (HF)",
    description: <>Per-split column names for an HF source (read from each split&apos;s parquet footer), so you can map a transcription column per split. Empty for non-HF datasets.</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Dataset id." }],
    request: { sample: `curl -s "$SGPU/v1/datasets/ds-1a2b3c4d/splits" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [
      {
        code: 200,
        codeLabel: "OK",
        sample: `{
  "splits": [
    {"split": "test",  "columns": ["after", "audio_filename", "before", "metadata"], "num_rows": 58},
    {"split": "train", "columns": ["audio_filename", "metadata", "text"], "num_rows": 3000}
  ]
}`,
      },
    ],
  },
  {
    id: "transform-dataset",
    group: "datasets",
    method: "POST",
    path: "/v1/datasets/:id/transform",
    title: "Transform → audio-column dataset",
    description: (
      <>
        <p>For an HF source that stores audio inside zip/tar archives (no playable <code>audio</code> column): unzips it, joins each metadata row&apos;s audio to its file (honouring the per-split mapping), and builds a <b>new</b> dataset with a real <code>audio</code> column — pushed to HuggingFace (<code>target=hf</code>) or materialised to S3 (<code>target=s3</code>). Non-destructive; runs as a background job (poll <code>GET /:id</code> for <code>transform_status</code> / <code>transform_log</code>).</p>
      </>
    ),
    parameters: [
      { name: "id", in: "path", type: "string", required: true, doc: "Source dataset id (must be HF)." },
      { name: "target", in: "body", type: '"hf" | "s3"', required: true, doc: "Output destination." },
      { name: "hf_repo", in: "body", type: "string", doc: "target=hf: owner/name to push to." },
      { name: "storage_id", in: "body", type: "string", doc: "target=s3: a kind=s3 storage." },
      { name: "s3_folder", in: "body", type: "string", doc: "target=s3: folder within the storage prefix. Default datasets/{id}/transformed." },
    ],
    request: {
      sample: `curl -s -X POST "$SGPU/v1/datasets/ds-1a2b3c4d/transform" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{"target": "s3", "storage_id": "store-1a2b3c4d", "s3_folder": "datasets/emgs-audio"}'`,
    },
    responses: [
      { code: 200, codeLabel: "OK", doc: "Job queued; the returned record's transform_status flips to running.", sample: `{ "id": "ds-1a2b3c4d", "transform_status": "running", "transform_log": "[03:04:25] transform queued (target=s3)", "...": "DatasetRecord" }` },
      { code: 400, codeLabel: "Bad Request", doc: "Source has no HF repo, target invalid, or storage_id isn't a kind=s3 storage.", sample: `{ "detail": "transform needs a source HuggingFace repo (owner/name) on the dataset" }` },
      { code: 409, codeLabel: "Conflict", doc: "A transform is already running for this dataset.", sample: `{ "detail": "a transform is already running for this dataset" }` },
    ],
  },
  {
    id: "delete-dataset",
    group: "datasets",
    method: "DELETE",
    path: "/v1/datasets/:id",
    title: "Delete a dataset",
    description: <>Removes the dataset record. Files already written to storage (uploaded metadata, materialised audio) are not deleted.</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Dataset id." }],
    request: { sample: `curl -s -X DELETE "$SGPU/v1/datasets/ds-1a2b3c4d" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK", sample: `{ "ok": true, "id": "ds-1a2b3c4d" }` }],
  },

  // ───── Autotrain ─────
  {
    id: "create-training-run",
    group: "autotrain",
    method: "POST",
    path: "/v1/training-runs",
    title: "Create a training run",
    description: (
      <>
        <p>Queues a Whisper finetune against a dataset. The gateway SSHes to the target (a RunPod pod it spawns, or a registered VM via <code>provider_id</code>), runs the trainer, streams logs, and writes checkpoints + metrics under the run&apos;s storage prefix.</p>
        <p className="mt-2 text-xs text-muted-foreground">All hyperparameters are optional — the trainer has sensible defaults. Experiment-tracking creds (W&amp;B / MLflow) come from the global Secrets page, not the body.</p>
      </>
    ),
    parameters: [
      { name: "name", in: "body", type: "string", required: true, doc: "Run label (also the W&B/MLflow run name)." },
      { name: "dataset_id", in: "body", type: "string", required: true, doc: "Dataset to train on (must be yours, or admin)." },
      { name: "base_model", in: "body", type: "string", required: true, doc: 'HF Whisper repo, e.g. "openai/whisper-small".' },
      { name: "test_dataset_id", in: "body", type: "string", doc: "Held-out eval dataset. Omit to split from train (eval_split_pct)." },
      { name: "eval_metric", in: "body", type: '"wer" | "cer"', doc: "Default wer." },
      { name: "max_epochs / patience", in: "body", type: "number", doc: "Epoch cap; patience=0 disables early stop." },
      { name: "batch_size / grad_accum / learning_rate / warmup_steps / weight_decay", in: "body", type: "number", doc: "Optimizer knobs. Defaults: 8 / 1 / 1e-5 / 0 / 0." },
      { name: "precision", in: "body", type: '"fp16" | "bf16"', doc: "Default fp16." },
      { name: "language / task", in: "body", type: "string", doc: 'e.g. "ms" / "transcribe".' },
      { name: "provider_id", in: "body", type: "string", doc: "vm provider → bare metal; omit (or a runpod provider) → cloud pod." },
      { name: "gpu_type / gpu_count / secure_cloud / disk_gb / volume_gb", in: "body", type: "mixed", doc: "Cloud-pod hardware. Defaults: L40S / 1 / true / 60 / 80." },
      { name: "visible_devices", in: "body", type: "string", doc: 'VM-only GPU pin, e.g. "0,1".' },
      { name: "storage_id", in: "body", type: "string", doc: "Enabled S3 backend for logs + artifacts. Omit for the gateway default." },
      { name: "hf_push_repo", in: "body", type: "string", doc: "Push the finished model to this HF repo (uses storage / env HF_TOKEN)." },
      { name: "report_to", in: "body", type: '("mlflow" | "wandb")[]', doc: "Experiment trackers to log to. Default none." },
    ],
    request: {
      sample: `curl -s -X POST "$SGPU/v1/training-runs" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{
    "name": "whisper-ms-v1",
    "dataset_id": "ds-1a2b3c4d",
    "base_model": "openai/whisper-small",
    "language": "ms", "max_epochs": 3, "precision": "bf16",
    "gpu_type": "NVIDIA L40S", "gpu_count": 1,
    "storage_id": "store-1a2b3c4d"
  }'`,
    },
    responses: [
      {
        code: 200,
        codeLabel: "OK",
        sample: `{
  "id": "train-1a2b3c4d",
  "name": "whisper-ms-v1",
  "status": "queued",
  "dataset_id": "ds-1a2b3c4d",
  "test_dataset_id": null,
  "base_model": "openai/whisper-small",
  "s3_prefix": "training-runs/train-1a2b3c4d/",
  "config_json": { "eval_metric": "wer", "max_epochs": 3, "precision": "bf16", "...": "…" },
  "exit_code": null, "error_text": null, "result_json": null,
  "created_by": "admin",
  "created_at": "2026-05-29T03:21:08+00:00",
  "started_at": null, "ended_at": null,
  "provider_id": null, "storage_id": "store-1a2b3c4d",
  "gpu_type": "NVIDIA L40S", "gpu_count": 1, "visible_devices": null
}`,
      },
      { code: 400, codeLabel: "Bad Request", doc: "Unknown dataset_id / test_dataset_id / provider_id, or storage_id isn't an enabled S3 backend.", sample: `{ "detail": "unknown dataset_id" }` },
    ],
  },
  {
    id: "list-training-runs",
    group: "autotrain",
    method: "GET",
    path: "/v1/training-runs",
    title: "List training runs",
    description: <>Your runs, newest first. <code>scope=all</code> (admin) returns everyone&apos;s.</>,
    parameters: [{ name: "scope", in: "query", type: '"mine" | "all"', doc: "Default mine." }],
    request: { sample: `curl -s "$SGPU/v1/training-runs?scope=mine" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK", sample: `[ { "id": "train-1a2b3c4d", "name": "whisper-ms-v1", "status": "done", "exit_code": 0, "result_json": { "wer": 0.142 }, "...": "TrainingRunRecord" } ]` }],
  },
  {
    id: "get-training-run",
    group: "autotrain",
    method: "GET",
    path: "/v1/training-runs/:id",
    title: "Get a training run",
    description: <>Full record for one run — status, hyperparameters (<code>config_json</code>), and final metrics (<code>result_json</code>) once it finishes.</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Run id (train-…)." }],
    request: { sample: `curl -s "$SGPU/v1/training-runs/train-1a2b3c4d" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [
      { code: 200, codeLabel: "OK", sample: `{ "id": "train-1a2b3c4d", "status": "running", "...": "TrainingRunRecord" }` },
      { code: 403, codeLabel: "Forbidden", doc: "The run isn't yours (and you're not admin).", sample: `{ "detail": "not yours" }` },
    ],
  },
  {
    id: "training-logs",
    group: "autotrain",
    method: "GET",
    path: "/v1/training-runs/:id/logs",
    title: "Fetch logs (tail)",
    description: <>Last <code>tail</code> log lines plus the live <code>status</code> and any <code>error_text</code>. Falls back to the on-disk log if Redis has rotated.</>,
    parameters: [
      { name: "id", in: "path", type: "string", required: true, doc: "Run id." },
      { name: "tail", in: "query", type: "number", doc: "Lines from the end. Default 400." },
    ],
    request: { sample: `curl -s "$SGPU/v1/training-runs/train-1a2b3c4d/logs?tail=200" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK", sample: `{
  "status": "running",
  "error_text": null,
  "lines": [
    "[gateway] starting autotrain run train-1a2b3c4d",
    "epoch 1/3 | step 50 | loss 0.81",
    "eval | wer 0.182"
  ]
}` }],
  },
  {
    id: "training-logs-stream",
    group: "autotrain",
    method: "GET",
    path: "/v1/training-runs/:id/logs/stream",
    title: "Stream logs (SSE)",
    description: <>Server-sent-events tail of the run&apos;s log. Emits each new line as <code>data:</code> and closes with an <code>end</code> event when the run reaches a terminal state.</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Run id." }],
    request: { sample: `curl -N "$SGPU/v1/training-runs/train-1a2b3c4d/logs/stream" \\
  -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK · text/event-stream", sample: `data: epoch 1/3 | step 50 | loss 0.81
data: eval | wer 0.182
event: end
data: end` }],
  },
  {
    id: "training-files",
    group: "autotrain",
    method: "GET",
    path: "/v1/training-runs/:id/files",
    title: "List artifacts",
    description: <>Every file under the run&apos;s storage prefix (checkpoints, metrics, the merged model) with a presigned download URL.</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Run id." }],
    request: { sample: `curl -s "$SGPU/v1/training-runs/train-1a2b3c4d/files" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK", sample: `[
  { "name": "metrics.json", "size": 412, "modified": "2026-05-29T04:48:31+00:00", "download_url": "https://…" },
  { "name": "model/model.safetensors", "size": 967482112, "modified": "2026-05-29T04:50:02+00:00", "download_url": "https://…" }
]` }],
  },
  {
    id: "terminate-training-run",
    group: "autotrain",
    method: "POST",
    path: "/v1/training-runs/:id/terminate",
    title: "Terminate a running run",
    description: <>Cancels the run, kills the trainer, and tears down the RunPod pod (cloud runs). No-op-safe only while active.</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Run id." }],
    request: { sample: `curl -s -X POST "$SGPU/v1/training-runs/train-1a2b3c4d/terminate" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [
      { code: 200, codeLabel: "OK", sample: `{ "id": "train-1a2b3c4d", "status": "cancelled", "...": "TrainingRunRecord" }` },
      { code: 409, codeLabel: "Conflict", doc: "The run already finished (done / failed / cancelled).", sample: `{ "detail": "already done" }` },
    ],
  },
  {
    id: "delete-training-run",
    group: "autotrain",
    method: "DELETE",
    path: "/v1/training-runs/:id",
    title: "Delete a training run",
    description: <>Cancels it if still running (tearing down the pod), then removes the record. Artifacts already written to storage are not deleted.</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Run id." }],
    request: { sample: `curl -s -X DELETE "$SGPU/v1/training-runs/train-1a2b3c4d" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK", sample: `{ "ok": true, "id": "train-1a2b3c4d" }` }],
  },

  // ───── Compute pods ─────
  {
    id: "create-compute",
    group: "compute",
    method: "POST",
    path: "/compute",
    title: "Create a compute pod",
    description: <>Spawns a raw RunPod pod with SSH + JupyterLab. Depending on policy it may land in <code>pending_approval</code> until an admin approves it.</>,
    parameters: [
      { name: "name", in: "body", type: "string", required: true, doc: "" },
      { name: "gpu_type", in: "body", type: "string", required: true, doc: 'RunPod GPU type, e.g. "NVIDIA H100 80GB HBM3".' },
      { name: "gpu_count", in: "body", type: "number", doc: "Default 1." },
      { name: "template_id", in: "body", type: "string", required: true, doc: "Curated template id or a RunPod template id." },
      { name: "image", in: "body", type: "string", doc: "Resolved image when template_id isn't a curated favourite." },
      { name: "cloud_type", in: "body", type: '"COMMUNITY" | "SECURE"', doc: "" },
      { name: "provider_id", in: "body", type: "string", doc: "RunPod account. Omit for the gateway key." },
    ],
    request: {
      sample: `curl -s -X POST "$SGPU/compute" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{"name": "dev-box", "gpu_type": "NVIDIA H100 80GB HBM3", "gpu_count": 1, "template_id": "pytorch-2.4-cuda12.4"}'`,
    },
    responses: [{ code: 200, codeLabel: "OK", sample: `{ "id": "pod-…", "name": "dev-box", "status": "creating", "gpu_type": "NVIDIA H100 80GB HBM3", "gpu_count": 1, "...": "ComputePod" }` }],
  },
  {
    id: "compute-ssh",
    group: "compute",
    method: "GET",
    path: "/compute/:id/ssh",
    title: "Get SSH access",
    description: <>Returns the SSH command + the private key for a running pod.</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Pod id." }],
    request: { sample: `curl -s "$SGPU/compute/pod-…/ssh" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK", sample: `{
  "ssh_command": "ssh root@1.2.3.4 -p 22000 -i key.pem",
  "ssh_user": "root", "ssh_host": "1.2.3.4", "ssh_port": 22000,
  "private_key": "-----BEGIN OPENSSH PRIVATE KEY-----\\n…"
}` }],
  },
  {
    id: "delete-compute",
    group: "compute",
    method: "DELETE",
    path: "/compute/:id",
    title: "Terminate a compute pod",
    description: <>Tears down the pod on the provider and marks it terminated.</>,
    parameters: [{ name: "id", in: "path", type: "string", required: true, doc: "Pod id." }],
    request: { sample: `curl -s -X DELETE "$SGPU/compute/pod-…" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK", sample: `{ "ok": true, "id": "pod-…" }` }],
  },

  // ───── Storage ─────
  {
    id: "list-storage",
    group: "storage",
    method: "GET",
    path: "/v1/storage",
    title: "List storage backends",
    description: <>Org-wide list of configured storage backends. Credentials are never returned — only <code>has_credentials</code>.</>,
    request: { sample: `curl -s "$SGPU/v1/storage" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK", sample: `[
  {
    "id": "store-1a2b3c4d",
    "name": "results", "kind": "s3",
    "bucket": "gpuplatform", "prefix": "datasets", "region": "ap-southeast-5",
    "endpoint": null, "has_credentials": true, "enabled": true,
    "created_by": "admin", "created_at": "2026-05-29T03:21:08+00:00"
  }
]` }],
  },
  {
    id: "create-storage",
    group: "storage",
    method: "POST",
    path: "/v1/storage",
    title: "Create a storage backend (admin)",
    description: <>Registers an S3 (or S3-compatible) bucket or a HuggingFace token holder. Credentials are encrypted at rest; leave them blank to fall back to the gateway&apos;s <code>AWS_*</code> / <code>HF_TOKEN</code> env. Admin only.</>,
    parameters: [
      { name: "name", in: "body", type: "string", required: true, doc: "Unique name." },
      { name: "kind", in: "body", type: '"s3" | "huggingface"', required: true, doc: "" },
      { name: "bucket", in: "body", type: "string", doc: "s3: required." },
      { name: "prefix / region / endpoint", in: "body", type: "string", doc: "s3: optional (endpoint for R2 / MinIO)." },
      { name: "access_key_id / secret_access_key", in: "body", type: "string", doc: "s3 creds. Both blank → env fallback." },
      { name: "hf_token", in: "body", type: "string", doc: "huggingface creds. Blank → HF_TOKEN env." },
    ],
    request: {
      sample: `curl -s -X POST "$SGPU/v1/storage" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{"name": "results", "kind": "s3", "bucket": "my-bucket", "region": "us-east-1",
       "access_key_id": "…", "secret_access_key": "…"}'`,
    },
    responses: [{ code: 200, codeLabel: "OK", sample: `{ "id": "store-…", "name": "results", "kind": "s3", "has_credentials": true, "enabled": true, "...": "StorageRecord" }` }],
  },

  // ───── Providers ─────
  {
    id: "list-providers",
    group: "providers",
    method: "GET",
    path: "/v1/providers",
    title: "List GPU providers",
    description: <>Org-wide list of registered providers. Secrets (private key / API key) are never returned — only summary fields like <code>api_key_last4</code>.</>,
    request: { sample: `curl -s "$SGPU/v1/providers" -H "Authorization: Bearer $SGPU_API_KEY"` },
    responses: [{ code: 200, codeLabel: "OK", sample: `[
  {
    "id": "prov-1a2b3c4d", "name": "lab-rig-01", "kind": "vm",
    "host": "10.0.0.5", "port": 22, "user": "root",
    "gpus": ["NVIDIA RTX 3090"], "gpu_count": 3,
    "created_by": "admin", "created_at": "2026-05-29T03:21:08+00:00"
  }
]` }],
  },
  {
    id: "create-provider",
    group: "providers",
    method: "POST",
    path: "/v1/providers",
    title: "Register a provider (admin)",
    description: <>Registers a VM (SSH), RunPod, or Prime Intellect account. The gateway validates the credential before saving. Admin only.</>,
    parameters: [
      { name: "name", in: "body", type: "string", required: true, doc: "" },
      { name: "kind", in: "body", type: '"vm" | "runpod" | "pi"', required: true, doc: "" },
      { name: "vm", in: "body", type: "{host,port,user,private_key}", doc: "Required for kind=vm." },
      { name: "api", in: "body", type: "{api_key}", doc: "Required for kind=runpod / pi." },
    ],
    request: {
      sample: `curl -s -X POST "$SGPU/v1/providers" \\
  -H "Authorization: Bearer $SGPU_API_KEY" -H "Content-Type: application/json" \\
  -d '{"name": "runpod-main", "kind": "runpod", "api": {"api_key": "rpa_…"}}'`,
    },
    responses: [{ code: 200, codeLabel: "OK", sample: `{ "id": "prov-…", "name": "runpod-main", "kind": "runpod", "api_key_last4": "0KV9", "...": "ProviderRecord" }` }],
  },
];

const ERROR_TABLE: Array<{ code: string; meaning: string }> = [
  { code: "401 Unauthorized", meaning: "Missing / revoked / malformed Authorization header." },
  { code: "403 Forbidden", meaning: "Your role/section access doesn't permit this (e.g. non-admin hitting an admin write, or reading someone else's resource)." },
  { code: "404 Not Found", meaning: "Unknown id (endpoint, benchmark, pod, storage, provider, or key)." },
  { code: "400 Bad Request", meaning: "Invalid body — missing required field, bad enum value, or a validation rule failed." },
  { code: "409 Conflict", meaning: "Name already taken (e.g. an endpoint or storage with that name exists)." },
  { code: "503 Unavailable", meaning: "A serverless create couldn't provision a worker right now (GPU out of stock / wrong tier). Body carries gpu, gpu_count, reason." },
  { code: "504 Gateway Timeout", meaning: "Synchronous inference didn't complete in time — usually a cold-starting worker. Retry or stream." },
];

// Resizable nav bounds. Under 200 px the endpoint paths truncate unreadably;
// over 560 px the samples column gets squeezed on a laptop. Width is persisted
// per-browser so it survives reloads.
const SIDEBAR_MIN_PX = 200;
const SIDEBAR_MAX_PX = 560;
const SIDEBAR_DEFAULT_PX = 240;
const SIDEBAR_LS_KEY = "sgpu.apidocs.sidebarWidth";

export function ApiDocs() {
  const base = gateway.baseUrl;
  const [query, setQuery] = useState("");
  const [sidebarWidth, setSidebarWidth] = useState<number>(SIDEBAR_DEFAULT_PX);
  // Ref (not state) so the mousemove handler isn't recreated each pixel and
  // sees the latest start values without a stale closure.
  const dragRef = useRef<{ startX: number; startWidth: number } | null>(null);

  // Hydrate the saved width on mount only — SSR can't read localStorage, and
  // starting from the default keeps the first client render matching the server.
  useEffect(() => {
    const saved = window.localStorage.getItem(SIDEBAR_LS_KEY);
    if (!saved) return;
    const n = parseInt(saved, 10);
    if (Number.isFinite(n) && n >= SIDEBAR_MIN_PX && n <= SIDEBAR_MAX_PX) {
      // Reading client-only localStorage post-mount is the correct way to avoid
      // an SSR/CSR width mismatch — a lazy initializer would diverge on hydrate.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSidebarWidth(n);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem(SIDEBAR_LS_KEY, String(sidebarWidth));
  }, [sidebarWidth]);

  const onDragStart = (e: React.MouseEvent<HTMLDivElement>) => {
    dragRef.current = { startX: e.clientX, startWidth: sidebarWidth };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    const move = (mv: MouseEvent) => {
      if (!dragRef.current) return;
      const delta = mv.clientX - dragRef.current.startX;
      setSidebarWidth(
        Math.max(SIDEBAR_MIN_PX, Math.min(SIDEBAR_MAX_PX, dragRef.current.startWidth + delta)),
      );
    };
    const up = () => {
      dragRef.current = null;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    e.preventDefault();
  };

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return ENDPOINTS;
    return ENDPOINTS.filter(
      (e) =>
        e.title.toLowerCase().includes(q) ||
        e.path.toLowerCase().includes(q) ||
        e.method.toLowerCase().includes(q),
    );
  }, [query]);

  const grouped = useMemo(() => {
    const out: Array<{ group: Group; items: Endpoint[] }> = [];
    for (const g of GROUPS) {
      const items = filtered.filter((e) => e.group === g.id);
      if (items.length > 0) out.push({ group: g, items });
    }
    return out;
  }, [filtered]);

  return (
    <div
      className="grid grid-cols-1 lg:grid-cols-[var(--sgpu-docs-w)_1px_minmax(0,1fr)]"
      style={{ "--sgpu-docs-w": `${sidebarWidth}px` } as React.CSSProperties}
    >
      {/* Endpoint nav */}
      <aside className="hidden lg:block">
        <div className="sticky top-0 max-h-[calc(100vh-3.5rem)] overflow-y-auto px-3 py-4 scrollbar-thin">
          <div className="relative flex h-9 items-center">
            <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search endpoints…"
              className="h-9 pl-8 text-xs"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <nav className="mt-3 space-y-2.5 text-sm">
            <a href="#auth" className="block px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-foreground/80 hover:text-foreground">
              Authentication
            </a>
            {grouped.map(({ group, items }) => (
              <div key={group.id} className="space-y-px">
                <a href={`#${group.id}`} className="block px-2 pb-0.5 text-[10px] font-semibold uppercase tracking-wider text-foreground/80">
                  {group.title}
                </a>
                <ul>
                  {items.map((e) => (
                    <li key={e.id}>
                      <a href={`#${e.id}`} className="flex items-center gap-1.5 rounded px-2 py-0.5 hover:bg-muted">
                        <MethodBadge method={e.method} size="xs" />
                        <span className="truncate font-mono text-[11px] text-muted-foreground">{e.path}</span>
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
            <a href="#errors" className="block px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-foreground/80 hover:text-foreground">
              Errors
            </a>
          </nav>
        </div>
      </aside>

      {/* Drag handle — 1 px visible bar, wider hit-target via the inset child.
          Double-click resets to the default width. */}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize endpoint list"
        onMouseDown={onDragStart}
        onDoubleClick={() => setSidebarWidth(SIDEBAR_DEFAULT_PX)}
        className="relative hidden cursor-col-resize select-none bg-border transition-colors hover:bg-primary/40 lg:block"
        title="Drag to resize · double-click to reset"
      >
        <div className="absolute inset-y-0 -left-1 -right-1" />
      </div>

      {/* Main content */}
      <div className="min-w-0 px-6 pt-6 pb-10 lg:px-10">
        <header className="space-y-4 pb-6">
          <div className="flex items-center justify-between gap-4">
            <h1 className="text-2xl font-semibold tracking-tight">HTTP API</h1>
            <Button asChild size="sm">
              <Link href="/api-keys">
                <KeyRound className="h-4 w-4" /> Manage tokens
              </Link>
            </Button>
          </div>
          <p className="max-w-2xl text-sm text-muted-foreground">
            Every action in the console is an HTTP call against the gateway. Authenticate with an API
            key as a <code>Bearer</code> token; a key acts as you and can only do what your role +
            section access allows.
          </p>

          <div className="grid gap-3 md:grid-cols-3">
            <InfoCard label="Base URL" body={base} />
            <InfoCard label="Auth header" body="Authorization: Bearer sgpu_xxxxxxxxxxxx" />
            <InfoCard
              label="Set your shell"
              body={`export SGPU="${base}"
export SGPU_API_KEY="sgpu_…"`}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Create a token at{" "}
            <Link href="/api-keys" className="underline underline-offset-2">API tokens</Link>. You can hold multiple tokens.
          </p>
        </header>

        <section id="auth" className="space-y-2 scroll-mt-4 border-t border-border pt-5">
          <h2 className="text-lg font-semibold tracking-tight">Authentication</h2>
          <p className="text-sm text-muted-foreground">
            Send <code>Authorization: Bearer &lt;key&gt;</code> on every request to{" "}
            <code>{base}</code>. Browser sessions (the console&apos;s login cookie) are also accepted,
            so the same routes back the UI. Keys are shown once at creation and stored hashed — rotate
            by creating a new key and revoking the old one.
          </p>
        </section>

        {grouped.map(({ group, items }) => (
          <div key={group.id}>
            <section id={group.id} className="scroll-mt-4 border-t border-border pt-5">
              <h2 className="text-lg font-semibold tracking-tight">{group.title}</h2>
              {group.blurb && <p className="mt-0.5 text-sm text-muted-foreground">{group.blurb}</p>}
            </section>
            {items.map((e) => (
              <EndpointSection key={e.id} endpoint={e} />
            ))}
          </div>
        ))}

        {filtered.length === 0 && (
          <div className="border-t border-border py-12 text-center text-sm text-muted-foreground">
            No endpoints match <code>&quot;{query}&quot;</code>.
          </div>
        )}

        <section id="errors" className="mt-8 space-y-3 scroll-mt-4 border-t border-border pt-5">
          <h2 className="text-xl font-semibold tracking-tight">Errors</h2>
          <div className="overflow-hidden rounded-md border border-border">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="px-3 py-2 text-left font-medium">Code</th>
                  <th className="px-3 py-2 text-left font-medium">Meaning</th>
                </tr>
              </thead>
              <tbody>
                {ERROR_TABLE.map((row) => (
                  <tr key={row.code} className="border-t border-border">
                    <td className="px-3 py-2 font-mono text-xs">{row.code}</td>
                    <td className="px-3 py-2 text-xs">{row.meaning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}

function InfoCard({ label, body }: { label: string; body: string }) {
  return (
    <div className="relative rounded-md border border-border bg-muted/30 p-3">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{label}</p>
      <pre className="mt-1 overflow-x-auto whitespace-pre-wrap break-all font-mono text-[11px] leading-snug text-foreground/90">
        {body}
      </pre>
      <CopyBtn text={body} />
    </div>
  );
}

function EndpointSection({ endpoint: e }: { endpoint: Endpoint }) {
  return (
    <section id={e.id} className="grid scroll-mt-4 gap-5 border-t border-border py-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,440px)]">
      {/* docs */}
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <MethodBadge method={e.method} />
          <code className="font-mono text-sm">{e.path}</code>
        </div>
        <h3 className="text-base font-semibold tracking-tight">{e.title}</h3>
        <div className="max-w-none text-sm">{e.description}</div>

        {e.parameters && e.parameters.length > 0 && (
          <div>
            <h4 className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Parameters</h4>
            <div className="overflow-hidden rounded-md border border-border">
              <table className="w-full text-sm">
                <thead className="bg-muted/40">
                  <tr>
                    <th className="px-2.5 py-1.5 text-left text-xs font-medium">Name</th>
                    <th className="px-2.5 py-1.5 text-left text-xs font-medium">In</th>
                    <th className="px-2.5 py-1.5 text-left text-xs font-medium">Type</th>
                    <th className="px-2.5 py-1.5 text-left text-xs font-medium">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {e.parameters.map((p) => (
                    <tr key={`${p.in}:${p.name}`} className="border-t border-border align-top">
                      <td className="px-2.5 py-1.5 font-mono text-xs">
                        {p.name}
                        {p.required && <span className="ml-1 text-rose-600">*</span>}
                      </td>
                      <td className="px-2.5 py-1.5 text-xs text-muted-foreground">{p.in}</td>
                      <td className="px-2.5 py-1.5 font-mono text-xs text-muted-foreground">{p.type}</td>
                      <td className="px-2.5 py-1.5 text-xs">{p.doc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* samples */}
      <div className="space-y-3">
        <CodeBlock label="Request">{e.request.sample}</CodeBlock>
        <div className="space-y-2.5">
          {e.responses.map((r, i) => (
            <div key={i} className="space-y-1">
              <StatusBadge code={r.code} label={r.codeLabel} />
              {r.doc && <p className="text-xs text-muted-foreground">{r.doc}</p>}
              <CodeBlock label="Response">{r.sample}</CodeBlock>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
