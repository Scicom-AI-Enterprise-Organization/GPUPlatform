// Thin client for the serverless-gpu FastAPI gateway.
//
// On the server: read the session cookie and attach it to gateway requests.
// In the browser: route through /api/proxy/* — the proxy does the cookie →
// Bearer-token translation server-side, so the token never hits the bundle.

import type {
  AppProxyLink,
  AdminUserRecord,
  AggregatePoint,
  ApiKeyRecord,
  AppRecord,
  CreateApiKeyResponse,
  AuditLogRecord,
  BenchmarkFile,
  BenchmarkRecord,
  BenchmarkTemplate,
  ComputePod,
  ComputeSshInfo,
  ComputeTemplate,
  GpuTypeOption,
  RegionOption,
  PiImageOption,
  RunpodTemplateSearchResult,
  CreateAppRequest,
  CreateAppResponse,
  ChatCompletionRequest,
  ChatCompletionResponse,
  ModelsListResponse,
  CreateBenchmarkRequest,
  CreateComputeRequest,
  CreateProviderRequest,
  CreateStorageRequest,
  CatalogRecord,
  CatalogRef,
  CatalogDataPreview,
  CreateCatalogRequest,
  UpdateCatalogRequest,
  CatalogRepoType,
  CreateDatasetRequest,
  UpdateDatasetRequest,
  TransformDatasetRequest,
  TtsPackRequest,
  OmnivoicePackRequest,
  LlmPackRequest,
  DatasetMergeRequest,
  DatasetRecord,
  DatasetPreview,
  DatasetFile,
  SyncDatasetRequest,
  GlobalEnvRecord,
  PolicyRole,
  ProviderRecord,
  SectionKey,
  StorageRecord,
  TestStorageRequest,
  TestStorageResponse,
  UpdateStorageRequest,
  TestProviderRequest,
  TestProviderResponse,
  VmAvailability,
  ProviderMetrics,
  ProviderBandwidth,
  ProviderBalance,
  TrainingRunRecord,
  TrainingGpuResponse,
  TrainingMetrics,
  CreateTrainingRunRequest,
  TrainingFile,
  TryItTarget,
  TrackingCredentialRecord,
  CreateTrackingCredentialRequest,
  GitopsRepo,
  GitopsResource,
  GitopsSyncResult,
  CreateGitopsRepoBody,
  UpdateGitopsRepoBody,
  TestGitopsRepoBody,
  TestGitopsRepoResult,
  ProxyEndpoint,
  ProxyUpstreamHealth,
  ProxyRequest,
  CreateProxyBody,
  UpdateProxyBody,
  TestProxyUpstreamBody,
  TestProxyUpstreamResult,
  ActivitySummary,
  ActivityLogsResponse,
  LogFilesResponse,
  LogFileContent,
} from "./types";

export type GpuAvailability = {
  gpu: string;
  count: number;
  available: boolean | null;
  cheapest_price_hr: number | null;
  regions: string[];
  reason: string | null;
  checked_at: number;
  provider: string;
};

export type GatewayRequestRecord = {
  request_id: string;
  app_id: string;
  endpoint: string;
  payload: unknown;
  status: string;
  output: unknown | null;
  is_stream: boolean;
  created_at: string;
  completed_at: string | null;
  requested_by: string | null;
};

const PUBLIC_BASE = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";
const isServer = typeof window === "undefined";

async function authHeaders(): Promise<Record<string, string>> {
  if (!isServer) return {};
  // Lazy import keeps `next/headers` out of the client bundle.
  const { cookies } = await import("next/headers");
  const jar = await cookies();
  const token = jar.get("sgpu_token")?.value;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export class GatewayError extends Error {
  status: number;
  body: string;
  parsed: unknown;
  constructor(status: number, body: string) {
    super(GatewayError.humanMessage(status, body));
    this.status = status;
    this.body = body;
    try {
      this.parsed = body ? JSON.parse(body) : null;
    } catch {
      this.parsed = null;
    }
  }

  /** A clean, user-facing message: prefer FastAPI's `detail` (string, or an
   * object with an `error` field) over the raw `gateway <status>: {json}` dump. */
  private static humanMessage(status: number, body: string): string {
    if (body) {
      try {
        const p = JSON.parse(body) as { detail?: unknown; error?: unknown };
        const d = p?.detail ?? p?.error;
        if (typeof d === "string" && d.trim()) return d;
        if (d && typeof d === "object" && typeof (d as { error?: unknown }).error === "string") {
          return (d as { error: string }).error;
        }
      } catch {
        /* not JSON — fall through to the raw body */
      }
      return body;
    }
    return `gateway error ${status}`;
  }
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 30_000): Promise<T> {
  const url = isServer ? `${PUBLIC_BASE}${path}` : `/api/proxy${path}`;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(await authHeaders()),
    ...((init?.headers as Record<string, string>) ?? {}),
  };
  // Always bound the request: without a timeout a slow/hung gateway call (e.g. a
  // worker op that SSHes to a flaky box) spins forever with no error. On timeout
  // or a network failure we throw a clear GatewayError so callers can surface it.
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(url, { ...init, headers, cache: "no-store", signal: ctrl.signal });
  } catch (e) {
    if (ctrl.signal.aborted) {
      throw new GatewayError(
        504,
        `request timed out after ${Math.round(timeoutMs / 1000)}s — the gateway took too long (the worker box may be slow or unreachable)`,
      );
    }
    throw new GatewayError(0, e instanceof Error ? e.message : String(e));
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new GatewayError(res.status, body);
  }
  const text = await res.text();
  return (text ? JSON.parse(text) : null) as T;
}

export const gateway = {
  baseUrl: PUBLIC_BASE,
  listApps: (scope: "mine" | "all" = "mine") =>
    request<AppRecord[]>(`/apps?scope=${scope}`),
  getApp: (id: string) => request<AppRecord>(`/apps/${encodeURIComponent(id)}`),
  createApp: (body: CreateAppRequest) =>
    request<CreateAppResponse>("/apps", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  updateAutoscaler: (id: string, body: Partial<{ max_containers: number; tasks_per_container: number; idle_timeout_s: number; vllm_args: string; request_timeout_s: number }>) =>
    request<AppRecord>(`/apps/${encodeURIComponent(id)}/autoscaler`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  // Edit a multi-model fleet in place (add/remove members, change tp, change
  // per-model vLLM args) and re-provision. Pass the FULL new member list.
  updateModels: (
    id: string,
    body: { models: { model: string; tp?: number; extra_args?: string }[]; sleep_level?: number; visible_devices?: string },
  ) =>
    request<AppRecord>(`/apps/${encodeURIComponent(id)}/models`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  deleteApp: (id: string) =>
    request<{ ok: boolean; app_id: string; drained_workers: number }>(
      `/apps/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),
  restartApp: (id: string) =>
    request<{ ok: boolean; app_id: string; drained_workers: number }>(
      `/apps/${encodeURIComponent(id)}/restart`,
      { method: "POST" },
      120_000, // drains + terminates over SSH — allow up to 2 min before timing out
    ),
  purgeApp: (id: string) =>
    request<{ ok: boolean; app_id: string; terminated: number; purged: number }>(
      `/apps/${encodeURIComponent(id)}/workers/purge`,
      { method: "POST" },
      120_000, // SSHes to the box, kills processes + sweeps many pidfiles — can be slow
    ),
  // Drop every job still waiting in the queue (running requests are untouched).
  flushQueue: (id: string) =>
    request<{ ok: boolean; app_id: string; flushed: number; cancelled: number }>(
      `/apps/${encodeURIComponent(id)}/queue/flush`,
      { method: "POST" },
    ),
  listAppRequests: (
    id: string,
    opts: { limit?: number; owner?: string; model?: string; status?: string; sort?: string; order?: string; requestId?: string } = {},
  ) => {
    const q = new URLSearchParams({ limit: String(opts.limit ?? 100) });
    if (opts.owner) q.set("owner", opts.owner);
    if (opts.model) q.set("model", opts.model);
    if (opts.status) q.set("status_filter", opts.status);
    if (opts.sort) q.set("sort", opts.sort);
    if (opts.order) q.set("order", opts.order);
    if (opts.requestId) q.set("request_id", opts.requestId);
    return request<GatewayRequestRecord[]>(`/apps/${encodeURIComponent(id)}/requests?${q.toString()}`);
  },
  getAppRequestFacets: (id: string) =>
    request<{ users: string[]; models: string[] }>(`/apps/${encodeURIComponent(id)}/request-facets`),
  checkAvailability: (gpu: string, count = 1, cloudType?: "COMMUNITY" | "SECURE") => {
    const params = new URLSearchParams({ gpu, count: String(count) });
    if (cloudType) params.set("cloud_type", cloudType);
    return request<GpuAvailability>(`/v1/availability?${params.toString()}`);
  },
  getAppStatus: (id: string) =>
    request<AppStatus>(`/apps/${encodeURIComponent(id)}/status`),
  /** Owner/admin only: make an endpoint public (read-only visible to all logged-in
   * users) or private again. */
  setAppVisibility: (id: string, isPublic: boolean) =>
    request<AppRecord>(`/apps/${encodeURIComponent(id)}/visibility`, {
      method: "POST",
      body: JSON.stringify({ is_public: isPublic }),
    }),
  /** LLM API proxies that front this endpoint (secret-stripped: name + serving
   * path + model aliases). Non-admins see only public proxies. */
  listAppProxies: (id: string) =>
    request<AppProxyLink[]>(`/apps/${encodeURIComponent(id)}/proxies`),

  // ---- inference (OpenAI-compatible) ----
  /** Send a chat-completion to a specific model. For a multi-model endpoint,
   * `body.model` is the member model the gateway routes/wakes (e.g.
   * "Qwen/Qwen3.6-27B"). Returns the completion synchronously (gateway polls
   * internally up to 60s); a dead/warming member surfaces as a GatewayError. */
  chatCompletion: (body: ChatCompletionRequest) =>
    request<ChatCompletionResponse>("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  /** Public model-discovery list (every model id usable in the `model` field). */
  listModels: () => request<ModelsListResponse>("/v1/models"),

  // ---- API keys ----
  listApiKeys: () => request<ApiKeyRecord[]>("/api-keys"),
  createApiKey: (name: string) =>
    request<CreateApiKeyResponse>("/api-keys", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  revokeApiKey: (id: string) =>
    request<{ ok: boolean; id: string }>(`/api-keys/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),

  // ---- Benchmarks ----
  listBenchmarks: (scope: "mine" | "all" = "mine") =>
    request<BenchmarkRecord[]>(`/benchmarks?scope=${scope}`),
  getBenchmark: (id: string) =>
    request<BenchmarkRecord>(`/benchmarks/${encodeURIComponent(id)}`),
  renameBenchmark: (id: string, name: string) =>
    request<BenchmarkRecord>(`/benchmarks/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ name }),
    }),
  setBenchmarkVisibility: (id: string, isPublic: boolean) =>
    request<BenchmarkRecord>(`/benchmarks/${encodeURIComponent(id)}/visibility`, {
      method: "POST",
      body: JSON.stringify({ is_public: isPublic }),
    }),
  createBenchmark: (body: CreateBenchmarkRequest) =>
    request<BenchmarkRecord>("/benchmarks", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteBenchmark: (id: string) =>
    request<{ ok: boolean; id: string }>(
      `/benchmarks/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),
  terminateBenchmark: (id: string) =>
    request<{ ok: boolean; id: string; status: string }>(
      `/benchmarks/${encodeURIComponent(id)}/terminate`,
      { method: "POST" },
    ),
  listBenchmarkFiles: (id: string) =>
    request<BenchmarkFile[]>(`/benchmarks/${encodeURIComponent(id)}/files`),
  /** Self-contained export (results + config + embedded S3 files as base64) for
   * moving a finished benchmark to another deployment. Returns the JSON object;
   * the caller downloads it. Generous timeout — files are inlined. */
  exportBenchmark: (id: string) =>
    request<Record<string, unknown>>(
      `/benchmarks/${encodeURIComponent(id)}/export`,
      undefined,
      120_000,
    ),
  /** Re-create a benchmark from an exported JSON (see exportBenchmark). Mints a
   * new id, writes embedded files into this deployment's bucket. */
  importBenchmark: (body: unknown) =>
    request<BenchmarkRecord>(
      "/benchmarks/import",
      { method: "POST", body: JSON.stringify(body) },
      180_000,
    ),
  /** Mint a public, no-auth comparison share link for a set of benchmark ids.
   * Returns the share token; the public page lives at /share/compare/{token}. */
  createBenchmarkShare: (ids: string[], notes = "", pairing: Record<string, string> = {}) =>
    request<{ token: string }>("/benchmarks/share", {
      method: "POST",
      body: JSON.stringify({ ids, notes, pairing }),
    }),
  /** Browser EventSource URL for SSE log stream — proxied through Next so the
   * session cookie is translated to a Bearer token server-side. */
  benchmarkLogsStreamUrl: (id: string) =>
    `/api/proxy/benchmarks/${encodeURIComponent(id)}/logs/stream`,
  /** Same-origin URL for a result file's bytes, served through the gateway
   * (cookie→Bearer via the Next proxy). Avoids browser→S3 CORS on the
   * presigned download_url. */
  benchmarkFileContentUrl: (id: string, name: string) =>
    `/api/proxy/benchmarks/${encodeURIComponent(id)}/files/content?path=${encodeURIComponent(name)}`,

  // ---- Autotrain runs ----
  listTrainingRuns: (scope: "mine" | "all" = "mine") =>
    request<TrainingRunRecord[]>(`/v1/training-runs?scope=${scope}`),
  getTrainingRun: (id: string) =>
    request<TrainingRunRecord>(`/v1/training-runs/${encodeURIComponent(id)}`),
  /** All persisted metrics in one call: loss steps, per-epoch eval, GPU samples. */
  getTrainingMetrics: (id: string) =>
    request<TrainingMetrics>(`/v1/training-runs/${encodeURIComponent(id)}/metrics`),
  renameTrainingRun: (id: string, name: string) =>
    request<TrainingRunRecord>(`/v1/training-runs/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ name }),
    }),
  createTrainingRun: (body: CreateTrainingRunRequest) =>
    request<TrainingRunRecord>("/v1/training-runs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteTrainingRun: (id: string) =>
    request<{ ok: boolean; id: string }>(
      `/v1/training-runs/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),
  terminateTrainingRun: (id: string) =>
    request<TrainingRunRecord>(
      `/v1/training-runs/${encodeURIComponent(id)}/terminate`,
      { method: "POST" },
    ),
  /** Gracefully stop a running run: the trainer saves + uploads the partial model
   * and finalizes (vs terminate, which hard-kills + discards). */
  stopTrainingEarly: (id: string) =>
    request<TrainingRunRecord>(
      `/v1/training-runs/${encodeURIComponent(id)}/stop-early`,
      { method: "POST" },
    ),
  /** (Re)run the Label-platform export for a finished TTS run (synthesize N clips +
   * create a recording+MOS project). Runs in the background; progress streams to logs. */
  retryLabelExport: (
    id: string,
    body: {
      base_url?: string;
      base_url_secret?: string | null;
      token?: string;
      token_secret?: string | null;
      project_name?: string | null;
      samples?: number;
      mos_axes?: string[];
      speakers?: string[];
      speaker_prefix?: boolean;
      reject_keywords?: string[];
      per_speaker?: boolean;
      tts_codec?: string;
      run_on?: "vm" | "cloud";
      provider_id?: string | null;
      gpu_type?: string;
      gpu_count?: number;
      secure_cloud?: boolean;
      data_center_id?: string | null;
      disk_gb?: number;
      volume_gb?: number;
      visible_devices?: string | null;
      venv_path?: string | null;
      // LLM-only override fields
      llm_eval_dataset_id?: string | null;
      llm_samples?: number;
      llm_mos_axes?: string[];
      llm_max_new_tokens?: number;
      vllm_version?: string | null;
    },
  ) =>
    request<{ status: string }>(
      `/v1/training-runs/${encodeURIComponent(id)}/label-export`,
      { method: "POST", body: JSON.stringify(body) },
    ),
  /** Stop a running Label export: cancels the gateway task, kills the box-side synth,
   * and tears down any cloud pod it spawned. Use when it looks stuck (or was orphaned
   * by a gateway restart, which leaves the status pinned at "running"). */
  cancelLabelExport: (id: string) =>
    request<{ status: string }>(
      `/v1/training-runs/${encodeURIComponent(id)}/label-export/cancel`,
      { method: "POST" },
    ),
  /** Push a finished run's best/final model to a Hugging Face repo. Token comes
   * from the selected kind=huggingface storage (or the platform HF_TOKEN). Runs in
   * the background; status + link land in result_json.hf_export. */
  exportToHuggingFace: (
    id: string,
    body: { repo: string; storage_id?: string | null; private?: boolean },
  ) =>
    request<{ status: string }>(
      `/v1/training-runs/${encodeURIComponent(id)}/hf-export`,
      { method: "POST", body: JSON.stringify(body) },
    ),
  /** Stop a stuck/running HF export: cancels the gateway task and kills the VM-side
   * download/upload process. Use when an export looks stuck (or was orphaned by a
   * gateway restart, which leaves the status pinned at "running"). */
  cancelHuggingFaceExport: (id: string) =>
    request<{ status: string; vm_process_killed?: boolean }>(
      `/v1/training-runs/${encodeURIComponent(id)}/hf-export/cancel`,
      { method: "POST" },
    ),
  /** Clone a run's config into a fresh queued run and launch it. */
  restartTrainingRun: (id: string) =>
    request<TrainingRunRecord>(
      `/v1/training-runs/${encodeURIComponent(id)}/restart`,
      { method: "POST" },
    ),
  listTrainingFiles: (id: string) =>
    request<TrainingFile[]>(`/v1/training-runs/${encodeURIComponent(id)}/files`),
  /** Live per-GPU utilisation for the run's GPUs only (poll while running). */
  getTrainingGpu: (id: string) =>
    request<TrainingGpuResponse>(`/v1/training-runs/${encodeURIComponent(id)}/gpu`),
  trainingLogsStreamUrl: (id: string) =>
    `/api/proxy/v1/training-runs/${encodeURIComponent(id)}/logs/stream`,
  /** Try-it playground: transcribe a clip with the run's finetuned model (runs
   * on the run's VM over SSH). `gpu` is a GPU index, "cpu", or "auto". */
  transcribeTrainingRun: async (id: string, file: File, gpu?: string) => {
    const buf = await file.arrayBuffer();
    const q = new URLSearchParams({ filename: file.name || "audio.wav" });
    if (gpu) q.set("gpu", gpu);
    return request<{ text: string; raw?: string | null; device?: string; logs?: string[] }>(
      `/v1/training-runs/${encodeURIComponent(id)}/transcribe?${q.toString()}`,
      { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: buf },
    );
  },
  /** Try-it playground (TTS): synthesize speech for `text` with the run's finetuned
   * model (runs on the run's VM over SSH). Returns a playable object-URL for the WAV. */
  synthesizeTrainingRun: async (id: string, text: string, opts?: { speaker?: string; gpu?: string }) => {
    const q = new URLSearchParams({ text });
    if (opts?.speaker) q.set("speaker", opts.speaker);
    if (opts?.gpu) q.set("gpu", opts.gpu);
    const res = await request<{ audio_b64: string; sample_rate: number; device?: string; logs?: string[]; prompt?: string; gen_text?: string }>(
      `/v1/training-runs/${encodeURIComponent(id)}/synthesize?${q.toString()}`,
      { method: "POST" },
    );
    const bin = atob(res.audio_b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
    return { url, sampleRate: res.sample_rate, device: res.device, logs: res.logs ?? [],
             prompt: res.prompt, genText: res.gen_text };
  },
  /** Persistent try-it worker: load the model once and keep it resident
   * (subsequent transcribe/synthesize skip the per-request model load). The
   * compute target (a fresh RunPod pod or a registered VM) is chosen per-call —
   * see TryItTarget; omitting it preserves the legacy run's-own-box behaviour. */
  playgroundStart: (id: string, opts?: Partial<TryItTarget> & { gpu?: string; vllmArgs?: string; vllmVersion?: string }) => {
    const qs = new URLSearchParams();
    if (opts?.target) qs.set("target", opts.target);
    if (opts?.target === "cloud") {
      if (opts.gpu_type) qs.set("gpu_type", opts.gpu_type);
      if (opts.gpu_count) qs.set("gpu_count", String(opts.gpu_count));
      if (opts.cloud_type) qs.set("cloud_type", opts.cloud_type);
    }
    // provider_id = the RunPod account (cloud) or the chosen VM provider (vm).
    if (opts?.provider_id) qs.set("provider_id", opts.provider_id);
    // GPU device index (vm target) + vLLM args/version (llm) — ignored by the cloud path.
    if (opts?.gpu) qs.set("gpu", opts.gpu);
    if (opts?.vllmArgs && opts.vllmArgs.trim()) qs.set("vllm_args", opts.vllmArgs.trim());
    if (opts?.vllmVersion && opts.vllmVersion.trim()) qs.set("vllm_version", opts.vllmVersion.trim());
    const q = qs.toString();
    return request<{ running: boolean; ready: boolean; device?: string; kind?: string; logs?: string[] }>(
      `/v1/training-runs/${encodeURIComponent(id)}/playground/start${q ? `?${q}` : ""}`,
      { method: "POST" },
    );
  },
  playgroundStatus: (id: string) =>
    request<{ running: boolean; ready: boolean; device?: string; kind?: string; logs?: string[] }>(
      `/v1/training-runs/${encodeURIComponent(id)}/playground/status`,
    ),
  playgroundStop: (id: string) =>
    request<{ running: boolean; ready: boolean; device?: string; kind?: string; logs?: string[] }>(
      `/v1/training-runs/${encodeURIComponent(id)}/playground/stop`,
      { method: "POST" },
    ),
  /** LLM try-it: stream a chat completion from the run's vLLM server. Returns the
   * raw streaming Response (OpenAI SSE) — the caller reads response.body. Client-only
   * (goes through /api/proxy, which pipes text/event-stream through unbuffered). */
  playgroundChatStream: (
    id: string,
    body: {
      messages: { role: string; content: string }[];
      temperature?: number;
      top_p?: number;
      max_tokens?: number;
      tools?: unknown[];
      tool_choice?: unknown;
    },
    signal?: AbortSignal,
  ): Promise<Response> =>
    fetch(`/api/proxy/v1/training-runs/${encodeURIComponent(id)}/playground/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    }),

  // ---- Experiment-tracker credentials (Secrets page card) ----
  listTrackingCredentials: (kind?: "wandb" | "mlflow") =>
    request<TrackingCredentialRecord[]>(
      `/v1/tracking-credentials${kind ? `?kind=${kind}` : ""}`,
    ),
  createTrackingCredential: (body: CreateTrackingCredentialRequest) =>
    request<TrackingCredentialRecord>("/v1/tracking-credentials", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteTrackingCredential: (id: string) =>
    request<{ ok: boolean; id: string }>(
      `/v1/tracking-credentials/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),

  // ---- Cross-benchmark aggregate (one point per result.json across all benches) ----
  aggregateBenchmarks: (scope: "mine" | "all" = "mine") =>
    request<AggregatePoint[]>(`/benchmarks/_aggregate?scope=${scope}`),

  // ---- Benchmark templates ----
  listBenchmarkTemplates: () =>
    request<BenchmarkTemplate[]>("/benchmarks/templates"),
  createBenchmarkTemplate: (name: string, config_yaml: string) =>
    request<BenchmarkTemplate>("/benchmarks/templates", {
      method: "POST",
      body: JSON.stringify({ name, config_yaml }),
    }),
  deleteBenchmarkTemplate: (id: string) =>
    request<{ ok: boolean; id: string }>(
      `/benchmarks/templates/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),

  // ---- Compute ----
  listCompute: (scope: "mine" | "all" = "mine") =>
    request<ComputePod[]>(`/compute?scope=${scope}`),
  getCompute: (id: string) =>
    request<ComputePod>(`/compute/${encodeURIComponent(id)}`),
  createCompute: (body: CreateComputeRequest) =>
    request<ComputePod>("/compute", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteCompute: (id: string) =>
    request<{ ok: boolean; id: string }>(
      `/compute/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),
  getComputeSsh: (id: string) =>
    request<ComputeSshInfo>(`/compute/${encodeURIComponent(id)}/ssh`),
  listComputeTemplates: () =>
    request<ComputeTemplate[]>("/compute/templates"),
  searchRunpodTemplates: (params: { q?: string; limit?: number; provider_id?: string | null }) => {
    const qs = new URLSearchParams();
    if (params.q) qs.set("q", params.q);
    if (params.limit) qs.set("limit", String(params.limit));
    if (params.provider_id) qs.set("provider_id", params.provider_id);
    const q = qs.toString();
    return request<RunpodTemplateSearchResult[]>(
      `/compute/runpod/templates${q ? `?${q}` : ""}`,
    );
  },
  listRunpodGpuTypes: () =>
    request<GpuTypeOption[]>("/compute/runpod/gpu-types"),
  listRunpodRegions: () =>
    request<RegionOption[]>("/compute/runpod/regions"),
  listPiGpuTypes: () => request<GpuTypeOption[]>("/compute/pi/gpu-types"),
  listPiImages: () => request<PiImageOption[]>("/compute/pi/images"),
  listPiCompatibleImages: (params: {
    gpu: string;
    count: number;
    cloud_type: "COMMUNITY" | "SECURE";
    provider_id?: string | null;
  }) => {
    const qs = new URLSearchParams({
      gpu: params.gpu,
      count: String(params.count),
      cloud_type: params.cloud_type,
    });
    if (params.provider_id) qs.set("provider_id", params.provider_id);
    return request<PiImageOption[]>(`/compute/pi/images/compatible?${qs.toString()}`);
  },
  listComputeApprovals: () => request<ComputePod[]>("/compute/approvals"),
  approveCompute: (id: string) =>
    request<ComputePod>(`/compute/${encodeURIComponent(id)}/approve`, {
      method: "POST",
    }),
  rejectCompute: (id: string, reason?: string) =>
    request<ComputePod>(`/compute/${encodeURIComponent(id)}/reject`, {
      method: "POST",
      body: JSON.stringify({ reason: reason ?? null }),
    }),

  // ---- Cloud providers ----
  listProviders: () => request<ProviderRecord[]>("/v1/providers"),
  createProvider: (body: CreateProviderRequest) =>
    request<ProviderRecord>("/v1/providers", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteProvider: (id: string) =>
    request<{ ok: boolean; id: string }>(
      `/v1/providers/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),
  testProvider: (body: TestProviderRequest) =>
    request<TestProviderResponse>("/v1/providers/test", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getVmAvailability: (id: string) =>
    request<VmAvailability>(`/v1/providers/${encodeURIComponent(id)}/availability`),
  // Live VM host metrics (CPU / memory / GPU) — polled by the metrics page.
  getProviderMetrics: (id: string) =>
    request<ProviderMetrics>(`/v1/providers/${encodeURIComponent(id)}/metrics`),
  // Kill a process by pid on a VM provider (metrics page "Terminate" button) to
  // free a GPU held by a stuck/orphaned process. Owner/admin only.
  killProviderPid: (id: string, pid: number) =>
    request<{ ok: boolean; message: string }>(
      `/v1/providers/${encodeURIComponent(id)}/kill-pid`,
      { method: "POST", body: JSON.stringify({ pid }) },
    ),
  // On-demand disk/memory/CPU bandwidth benchmark (button-triggered, not polled).
  getProviderBandwidth: (id: string) =>
    request<ProviderBandwidth>(`/v1/providers/${encodeURIComponent(id)}/bandwidth`),
  getProviderBalance: (id: string) =>
    request<ProviderBalance>(`/v1/providers/${encodeURIComponent(id)}/balance`),

  // ---- Storage backends ----
  listStorage: () => request<StorageRecord[]>("/v1/storage"),
  createStorage: (body: CreateStorageRequest) =>
    request<StorageRecord>("/v1/storage", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  updateStorage: (id: string, body: UpdateStorageRequest) =>
    request<StorageRecord>(`/v1/storage/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  testStorage: (body: TestStorageRequest) =>
    request<TestStorageResponse>("/v1/storage/test", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteStorage: (id: string) =>
    request<{ ok: boolean; id: string }>(
      `/v1/storage/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),

  // ---- GitOps ----
  listGitopsRepos: () => request<GitopsRepo[]>("/v1/gitops"),
  getGitopsRepo: (id: string) =>
    request<GitopsRepo>(`/v1/gitops/${encodeURIComponent(id)}`),
  createGitopsRepo: (body: CreateGitopsRepoBody) =>
    request<GitopsRepo>("/v1/gitops", { method: "POST", body: JSON.stringify(body) }),
  testGitopsRepo: (body: TestGitopsRepoBody) =>
    request<TestGitopsRepoResult>("/v1/gitops/test", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  updateGitopsRepo: (id: string, body: UpdateGitopsRepoBody) =>
    request<GitopsRepo>(`/v1/gitops/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  deleteGitopsRepo: (id: string, prune = false) =>
    request<{ ok: boolean; id: string; pruned: boolean }>(
      `/v1/gitops/${encodeURIComponent(id)}?prune=${prune ? "true" : "false"}`,
      { method: "DELETE" },
    ),
  listGitopsResources: (id: string) =>
    request<GitopsResource[]>(`/v1/gitops/${encodeURIComponent(id)}/resources`),
  // A manual reconcile can take a while (git fetch + create/spawn) — give it room.
  syncGitopsRepo: (id: string) =>
    request<GitopsSyncResult>(`/v1/gitops/${encodeURIComponent(id)}/sync`, { method: "POST" }, 120_000),

  // ---- LLM API proxy ----
  listProxies: () => request<ProxyEndpoint[]>("/v1/proxy"),
  getProxy: (id: string) => request<ProxyEndpoint>(`/v1/proxy/${encodeURIComponent(id)}`),
  /** Read-only public proxies — visible to any logged-in user (non-admins). */
  listPublicProxies: () => request<ProxyEndpoint[]>("/v1/proxy/public"),
  getPublicProxy: (id: string) =>
    request<ProxyEndpoint>(`/v1/proxy/${encodeURIComponent(id)}/public`),
  createProxy: (body: CreateProxyBody) =>
    request<ProxyEndpoint>("/v1/proxy", { method: "POST", body: JSON.stringify(body) }),
  updateProxy: (id: string, body: UpdateProxyBody) =>
    request<ProxyEndpoint>(`/v1/proxy/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteProxy: (id: string) =>
    request<{ ok: boolean; id: string }>(`/v1/proxy/${encodeURIComponent(id)}`, { method: "DELETE" }),
  getProxyHealth: (id: string) =>
    request<ProxyUpstreamHealth[]>(`/v1/proxy/${encodeURIComponent(id)}/health`),
  // ---- usage activity analytics (admin) ----
  getActivity: (params: { since?: string; until?: string; tz?: string; granularity?: string; top?: number; models?: string[] } = {}) => {
    const q = new URLSearchParams();
    if (params.since) q.set("since", params.since);
    if (params.until) q.set("until", params.until);
    if (params.tz) q.set("tz", params.tz);
    if (params.granularity) q.set("granularity", params.granularity);
    if (params.top != null) q.set("top", String(params.top));
    for (const m of params.models ?? []) q.append("models", m);
    const s = q.toString();
    return request<ActivitySummary>(`/v1/history/activity${s ? `?${s}` : ""}`);
  },
  // ---- serverless endpoint logs (Logs tab) ----
  // Set (storageId) or clear (null) the endpoint's s3 log-archive storage.
  setLogStorage: (appId: string, storageId: string | null) =>
    request<AppRecord>(`/apps/${encodeURIComponent(appId)}/log-storage`, {
      method: "PATCH",
      body: JSON.stringify({ storage_id: storageId }),
    }),
  listAppLogFiles: (
    appId: string,
    params: { source?: string; q?: string; sort?: string; limit?: number; offset?: number } = {},
  ) => {
    const p = new URLSearchParams();
    if (params.source) p.set("source", params.source);
    if (params.q) p.set("q", params.q);
    if (params.sort) p.set("sort", params.sort);
    if (params.limit != null) p.set("limit", String(params.limit));
    if (params.offset != null) p.set("offset", String(params.offset));
    const s = p.toString();
    return request<LogFilesResponse>(`/apps/${encodeURIComponent(appId)}/logs/files${s ? `?${s}` : ""}`);
  },
  getAppLogFile: (appId: string, fileId: string, tail?: number) =>
    request<LogFileContent>(
      `/apps/${encodeURIComponent(appId)}/logs/files/${encodeURIComponent(fileId)}${tail ? `?tail=${tail}` : ""}`,
    ),
  // Browser download URL (anchor href) — goes through the cookie-auth proxy.
  appLogFileDownloadUrl: (appId: string, fileId: string) =>
    `/api/proxy/apps/${encodeURIComponent(appId)}/logs/files/${encodeURIComponent(fileId)}/download`,
  getActivityLogs: (params: { since?: string; until?: string; user?: string; source?: string; limit?: number; offset?: number } = {}) => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v != null && v !== "") q.set(k, String(v));
    const s = q.toString();
    return request<ActivityLogsResponse>(`/v1/history/activity/logs${s ? `?${s}` : ""}`);
  },
  getProxyRequests: (
    id: string,
    opts: { limit?: number; owner?: string; upstream?: string; status?: string; sort?: string; order?: string; requestId?: string } = {},
  ) => {
    const q = new URLSearchParams({ limit: String(opts.limit ?? 50) });
    if (opts.owner) q.set("owner", opts.owner);
    if (opts.upstream) q.set("upstream", opts.upstream);
    if (opts.status) q.set("status", opts.status);
    if (opts.sort) q.set("sort", opts.sort);
    if (opts.order) q.set("order", opts.order);
    if (opts.requestId) q.set("request_id", opts.requestId);
    return request<ProxyRequest[]>(`/v1/proxy/${encodeURIComponent(id)}/requests?${q.toString()}`);
  },
  getProxyRequestFacets: (id: string) =>
    request<{ users: string[]; upstreams: string[] }>(
      `/v1/proxy/${encodeURIComponent(id)}/request-facets`,
    ),
  cancelProxyRequest: (id: string, reqId: string) =>
    request<{ ok: boolean; id: string }>(
      `/v1/proxy/${encodeURIComponent(id)}/requests/${encodeURIComponent(reqId)}/cancel`,
      { method: "POST" },
    ),
  flushProxyQueue: (id: string) =>
    request<{ ok: boolean; flushed: number }>(
      `/v1/proxy/${encodeURIComponent(id)}/flush`,
      { method: "POST" },
    ),
  testProxyUpstream: (body: TestProxyUpstreamBody) =>
    request<TestProxyUpstreamResult>("/v1/proxy/test", { method: "POST", body: JSON.stringify(body) }),

  // ---- Datasets (Autotrain) ----
  listDatasets: (scope: "mine" | "all" = "mine") =>
    request<DatasetRecord[]>(`/v1/datasets?scope=${scope}`),
  getDataset: (id: string) =>
    request<DatasetRecord>(`/v1/datasets/${encodeURIComponent(id)}`),
  createDataset: (body: CreateDatasetRequest) =>
    request<DatasetRecord>("/v1/datasets", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  updateDataset: (id: string, body: UpdateDatasetRequest) =>
    request<DatasetRecord>(`/v1/datasets/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  // Include/exclude rows from training (manual curation in the row browser).
  setRowInclusion: (
    id: string,
    body: { indices?: number[]; included?: boolean; clear?: boolean },
  ) =>
    request<{ excluded_count: number }>(
      `/v1/datasets/${encodeURIComponent(id)}/row-inclusion`,
      { method: "POST", body: JSON.stringify(body) },
    ),
  deleteDataset: (id: string, purge = false) =>
    request<{ ok: boolean; id: string; purged_objects?: number }>(
      `/v1/datasets/${encodeURIComponent(id)}${purge ? "?purge=true" : ""}`,
      { method: "DELETE" },
    ),
  /** Publish an S3-backed dataset to the HF mirror as a hosted dataset repo. */
  publishDataset: (id: string) =>
    request<{ repo_id: string; full_id: string; repo_type: string; num_files: number; size_bytes: number }>(
      `/v1/datasets/${encodeURIComponent(id)}/publish`,
      { method: "POST" },
    ),

  // ---- Model/Dataset catalog (self-hosted HuggingFace mirror) ----
  listCatalog: (scope: "mine" | "all" = "mine", repoType?: CatalogRepoType) => {
    const q = new URLSearchParams({ scope });
    if (repoType) q.set("repo_type", repoType);
    return request<CatalogRecord[]>(`/v1/catalog?${q.toString()}`);
  },
  getCatalogRepo: (id: string, revision?: string) => {
    const q = revision ? `?revision=${encodeURIComponent(revision)}` : "";
    return request<CatalogRecord>(`/v1/catalog/${encodeURIComponent(id)}${q}`);
  },
  /** Branches of a versioned repo (head `main` + each named revision). */
  listCatalogRefs: (id: string) =>
    request<{ branches: CatalogRef[] }>(`/v1/catalog/${encodeURIComponent(id)}/refs`),
  /** Parquet row preview for a hosted dataset repo (config/subset + split). */
  getCatalogData: (
    id: string,
    params: { config?: string; split?: string; offset?: number; limit?: number } = {},
  ) => {
    const q = new URLSearchParams();
    if (params.config) q.set("config", params.config);
    if (params.split) q.set("split", params.split);
    if (params.offset != null) q.set("offset", String(params.offset));
    if (params.limit != null) q.set("limit", String(params.limit));
    return request<CatalogDataPreview>(`/v1/catalog/${encodeURIComponent(id)}/data?${q.toString()}`);
  },
  /** Resolve a repo by its HF id (repo_type + namespace/name) for name-based URLs. */
  lookupCatalogRepo: (repoType: CatalogRepoType, namespace: string, name: string, revision?: string) => {
    const q = new URLSearchParams({ repo_type: repoType, namespace, name });
    if (revision) q.set("revision", revision);
    return request<CatalogRecord>(`/v1/catalog/lookup?${q.toString()}`);
  },
  createCatalogRepo: (body: CreateCatalogRequest) =>
    request<CatalogRecord>("/v1/catalog", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  updateCatalogRepo: (id: string, body: UpdateCatalogRequest) =>
    request<CatalogRecord>(`/v1/catalog/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  reindexCatalogRepo: (id: string) =>
    request<CatalogRecord>(`/v1/catalog/${encodeURIComponent(id)}/reindex`, {
      method: "POST",
    }),
  deleteCatalogRepo: (id: string, wipe = false) =>
    request<{ ok: boolean; id: string }>(
      `/v1/catalog/${encodeURIComponent(id)}${wipe ? "?wipe=true" : ""}`,
      { method: "DELETE" },
    ),
  /** S3 objects backing a dataset (Files tab). `split` narrows a split-aware
   * tts_packed dataset to that split's subdir. */
  listDatasetFiles: (id: string, split?: string | null) =>
    request<DatasetFile[]>(
      `/v1/datasets/${encodeURIComponent(id)}/files${split ? `?split=${encodeURIComponent(split)}` : ""}`,
    ),
  getDatasetPreview: (
    id: string,
    limit = 20,
    offset = 0,
    split?: string | null,
    speaker?: string | null,
  ) => {
    const q = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (split) q.set("split", split);
    if (speaker) q.set("speaker", speaker);
    return request<DatasetPreview>(
      `/v1/datasets/${encodeURIComponent(id)}/preview?${q.toString()}`,
    );
  },
  syncDataset: (id: string, body: SyncDatasetRequest) =>
    request<DatasetRecord>(`/v1/datasets/${encodeURIComponent(id)}/sync`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  // Extract a real audio column (hf archive / label export → audio) → HF or S3.
  // Background job; poll getDataset(id).transform_status / transform_log.
  transformDataset: (id: string, body: TransformDatasetRequest) =>
    request<DatasetRecord>(`/v1/datasets/${encodeURIComponent(id)}/transform`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  // NeuCodec-encode + multipack for TTS → a new packed dataset. Background job;
  // poll getDataset(id).transform_status / transform_log.
  packTtsDataset: (id: string, body: TtsPackRequest) =>
    request<DatasetRecord>(`/v1/datasets/${encodeURIComponent(id)}/pack-tts`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  // Higgs-codec tokenize an {audio,text} dataset into OmniVoice WebDataset shards
  // (kind=omnivoice_packed) on a GPU box. Poll getDataset(id).transform_status.
  packOmnivoiceDataset: (id: string, body: OmnivoicePackRequest) =>
    request<DatasetRecord>(`/v1/datasets/${encodeURIComponent(id)}/pack-omnivoice`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  // Tokenize + multipack a chat (kind=llm) dataset's messages column into a
  // ChiniDataset (kind=llm_packed). In-process background job; poll
  // getDataset(id).transform_status / transform_log.
  packLlmDataset: (id: string, body: LlmPackRequest) =>
    request<DatasetRecord>(`/v1/datasets/${encodeURIComponent(id)}/pack-llm`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  // Concatenate >=2 kind=label datasets into one combined audio dataset.
  // Returns the NEW dataset (transform_status=running) — poll getDataset(id).
  mergeDatasets: (body: DatasetMergeRequest) =>
    request<DatasetRecord>("/v1/datasets/merge", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  // Cancel a running transform (audio extraction or TTS pack) for a dataset.
  cancelDatasetTransform: (id: string) =>
    request<DatasetRecord>(`/v1/datasets/${encodeURIComponent(id)}/cancel-transform`, {
      method: "POST",
    }),

  // ---- Global secrets (admin-managed; values masked) ----
  listGlobalEnv: () => request<GlobalEnvRecord[]>("/v1/global-env"),

  // ---- Admin: users, policy roles, audit ----
  adminListUsers: () => request<AdminUserRecord[]>("/admin/users"),
  adminGetUser: (id: number) =>
    request<AdminUserRecord>(`/admin/users/${id}`),
  adminSetUserRole: (id: number, role: "user" | "developer" | "admin") =>
    request<AdminUserRecord>(`/admin/users/${id}/role`, {
      method: "PATCH",
      body: JSON.stringify({ role }),
    }),
  adminSetUserPolicyRole: (id: number, policy_role_id: string | null) =>
    request<AdminUserRecord>(`/admin/users/${id}/policy-role`, {
      method: "PATCH",
      body: JSON.stringify({ policy_role_id }),
    }),
  adminDeleteUser: (id: number) =>
    request<{ ok: boolean; username: string }>(`/admin/users/${id}`, {
      method: "DELETE",
    }),
  adminListPolicyRoles: () => request<PolicyRole[]>("/admin/policy-roles"),
  adminCreatePolicyRole: (
    id: string,
    name: string,
    sections: Record<SectionKey, boolean>,
  ) =>
    request<PolicyRole>("/admin/policy-roles", {
      method: "POST",
      body: JSON.stringify({ id, name, sections }),
    }),
  adminUpdatePolicyRole: (
    id: string,
    body: { name?: string; sections?: Record<SectionKey, boolean> },
  ) =>
    request<PolicyRole>(`/admin/policy-roles/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  adminDeletePolicyRole: (id: string) =>
    request<{ ok: boolean; id: string }>(
      `/admin/policy-roles/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),
  adminListAuditLogs: (
    params: {
      limit?: number;
      actor?: string;
      resource_type?: string;
      action?: string;
    } = {},
  ) => {
    const q = new URLSearchParams();
    if (params.limit) q.set("limit", String(params.limit));
    if (params.actor) q.set("actor", params.actor);
    if (params.resource_type) q.set("resource_type", params.resource_type);
    if (params.action) q.set("action", params.action);
    const qs = q.toString();
    return request<AuditLogRecord[]>(
      `/admin/audit-logs${qs ? `?${qs}` : ""}`,
    );
  },
};

// Per-model state reported by a multi-model worker's heartbeat.
export type ModelState = {
  model: string;
  state: "awake" | "asleep" | "loading" | "waking" | "draining" | "error";
  inflight?: number;
  slot?: number | null;
  last_used_ts?: number | null;
  queue_len?: number;
};

export type AppStatus = {
  app_id: string;
  queue_len: number;
  workers: number;
  last_provision_error: string | null;
  last_provision_error_at: number | null;
  provision_cooldown_remaining_s: number;
  mode?: "single" | "multi";
  models?: ModelState[];
  sleep_level?: number;
};
