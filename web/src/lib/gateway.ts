// Thin client for the serverless-gpu FastAPI gateway.
//
// On the server: read the session cookie and attach it to gateway requests.
// In the browser: route through /api/proxy/* — the proxy does the cookie →
// Bearer-token translation server-side, so the token never hits the bundle.

import type {
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
  CreateDatasetRequest,
  UpdateDatasetRequest,
  TransformDatasetRequest,
  TtsPackRequest,
  DatasetRecord,
  DatasetPreview,
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
  TrainingRunRecord,
  TrainingGpuResponse,
  TrainingMetrics,
  CreateTrainingRunRequest,
  TrainingFile,
  TrackingCredentialRecord,
  CreateTrackingCredentialRequest,
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
    super(`gateway ${status}: ${body || "<empty>"}`);
    this.status = status;
    this.body = body;
    try {
      this.parsed = body ? JSON.parse(body) : null;
    } catch {
      this.parsed = null;
    }
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = isServer ? `${PUBLIC_BASE}${path}` : `/api/proxy${path}`;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(await authHeaders()),
    ...((init?.headers as Record<string, string>) ?? {}),
  };
  const res = await fetch(url, { ...init, headers, cache: "no-store" });
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
  updateAutoscaler: (id: string, body: Partial<{ max_containers: number; tasks_per_container: number; idle_timeout_s: number; vllm_args: string }>) =>
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
    ),
  listAppRequests: (id: string, limit = 100) =>
    request<GatewayRequestRecord[]>(
      `/apps/${encodeURIComponent(id)}/requests?limit=${limit}`,
    ),
  checkAvailability: (gpu: string, count = 1, cloudType?: "COMMUNITY" | "SECURE") => {
    const params = new URLSearchParams({ gpu, count: String(count) });
    if (cloudType) params.set("cloud_type", cloudType);
    return request<GpuAvailability>(`/v1/availability?${params.toString()}`);
  },
  getAppStatus: (id: string) =>
    request<AppStatus>(`/apps/${encodeURIComponent(id)}/status`),

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
    return request<{ text: string; device?: string }>(
      `/v1/training-runs/${encodeURIComponent(id)}/transcribe?${q.toString()}`,
      { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: buf },
    );
  },

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
  deleteDataset: (id: string) =>
    request<{ ok: boolean; id: string }>(
      `/v1/datasets/${encodeURIComponent(id)}`,
      { method: "DELETE" },
    ),
  getDatasetPreview: (id: string, limit = 20, offset = 0, split?: string | null) => {
    const q = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (split) q.set("split", split);
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
