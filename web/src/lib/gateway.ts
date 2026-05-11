// Thin client for the serverless-gpu FastAPI gateway.
//
// On the server: read the session cookie and attach it to gateway requests.
// In the browser: route through /api/proxy/* — the proxy does the cookie →
// Bearer-token translation server-side, so the token never hits the bundle.

import type {
  AdminUserRecord,
  AggregatePoint,
  AppRecord,
  AuditLogRecord,
  BenchmarkFile,
  BenchmarkRecord,
  BenchmarkTemplate,
  ComputePod,
  ComputeSshInfo,
  ComputeTemplate,
  CreateAppRequest,
  CreateAppResponse,
  CreateBenchmarkRequest,
  CreateComputeRequest,
  PolicyRole,
  SectionKey,
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

  // ---- Benchmarks ----
  listBenchmarks: (scope: "mine" | "all" = "mine") =>
    request<BenchmarkRecord[]>(`/benchmarks?scope=${scope}`),
  getBenchmark: (id: string) =>
    request<BenchmarkRecord>(`/benchmarks/${encodeURIComponent(id)}`),
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
  listBenchmarkFiles: (id: string) =>
    request<BenchmarkFile[]>(`/benchmarks/${encodeURIComponent(id)}/files`),
  /** Browser EventSource URL for SSE log stream — proxied through Next so the
   * session cookie is translated to a Bearer token server-side. */
  benchmarkLogsStreamUrl: (id: string) =>
    `/api/proxy/benchmarks/${encodeURIComponent(id)}/logs/stream`,

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

export type AppStatus = {
  app_id: string;
  queue_len: number;
  workers: number;
  last_provision_error: string | null;
  last_provision_error_at: number | null;
  provision_cooldown_remaining_s: number;
};
