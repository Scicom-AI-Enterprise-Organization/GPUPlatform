// Mirrors gateway/main.py Pydantic models. Keep field names + defaults in sync.

export type AutoscalerSpec = {
  max_containers: number;
  tasks_per_container: number;
  idle_timeout_s: number;
};

export type AppRecord = {
  app_id: string;
  name: string;
  model: string;
  gpu: string;
  gpu_count: number;
  autoscaler: AutoscalerSpec;
  cpu: number;
  memory: string;
  request_timeout_s: number;
  vllm_args: string;
  enable_metrics: boolean;
  cloud_type?: "COMMUNITY" | "SECURE" | null;
  container_disk_gb?: number | null;
  volume_gb?: number | null;
  created_at: string;
  owner: string;
};

export type CreateAppRequest = {
  name: string;
  model: string;
  gpu: string;
  gpu_count?: number;
  autoscaler?: Partial<AutoscalerSpec>;
  cpu?: number;
  memory?: string;
  request_timeout_s?: number;
  vllm_args?: string;
  enable_metrics?: boolean;
  cloud_type?: "COMMUNITY" | "SECURE";
  container_disk_gb?: number;
  volume_gb?: number;
  provider_id?: string | null;
};

export type CreateAppResponse = {
  app_id: string;
  url: string;
};

// Worker types are not exposed by the gateway directly today — mock data
// shapes matching what the dashboard renders.
export type WorkerStatus = "idle" | "running" | "initializing" | "throttled" | "down";

export type WorkerRow = {
  id: string;
  status: WorkerStatus;
  region_code: string;   // "IN", "US", "DE"
  region: string;        // "AP-IN-1"
  gpu: string;           // "H100 SXM"
  vcpus: number;
  ram: string;           // "251 GB"
  release: string;       // "Latest"
  count: number;
};

export type RequestRow = {
  id: string;
  status: "in queue" | "in progress" | "completed" | "failed";
  duration_ms: number;
  delay_ms?: number;
  cost_usd?: number;
};

export type Me = {
  user_id: number;
  username: string;
};

// ---- Benchmarks ----

export type BenchmarkRecord = {
  id: string;
  name: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  s3_prefix: string;
  config_yaml: string;
  exit_code?: number | null;
  error_text?: string | null;
  result_json?: Record<string, unknown> | null;
  created_by: string;
  created_at: string;
  started_at?: string | null;
  ended_at?: string | null;
  cost_per_hr?: number | null;
  provider_id?: string | null;
};

export type CreateBenchmarkRequest = {
  name: string;
  config_yaml: string;
  provider_id?: string | null;
  // VM runs only: rm -rf the model's local_dir + HF hub cache on the VM after
  // the run exits. Default true on the UI side.
  cleanup_model?: boolean;
};

export type BenchmarkFile = {
  name: string;
  size: number;
  modified: string;
  download_url: string;
};

export type BenchmarkTemplate = {
  id: string;
  name: string;
  config_yaml: string;
  created_at: string;
};

export type AggregatePoint = {
  benchmark_id: string;
  benchmark_name: string;
  model: string | null;
  gpu_type: string | null;
  gpu_count: number;
  engine: string;
  tp: number;
  dp: number;
  context_len: number;
  output_len: number;
  concurrency: number;
  num_prompts: number;
  duration_s: number | null;
  output_throughput: number | null;
  output_throughput_per_gpu: number | null;
  request_throughput: number | null;
  median_ttft_ms: number | null;
  p99_ttft_ms: number | null;
  median_tpot_ms: number | null;
  p99_tpot_ms: number | null;
  median_itl_ms: number | null;
  median_e2el_ms: number | null;
  p99_e2el_ms: number | null;
};

// ---- Compute (raw RunPod pods with SSH + JupyterLab) ----

export type ComputeStatus =
  | "pending_approval"
  | "creating"
  | "running"
  | "failed"
  | "terminated"
  | "rejected";

export type ComputePod = {
  id: string;
  name: string;
  gpu_type: string;
  gpu_count: number;
  container_disk_gb: number;
  volume_gb: number;
  image: string;
  template_id: string | null;
  cloud_type: "COMMUNITY" | "SECURE";
  status: ComputeStatus;
  runpod_pod_id: string | null;
  public_ip: string | null;
  ssh_port: number | null;
  ssh_user: string;
  jupyter_url: string | null;
  jupyter_password: string | null;
  cost_per_hr: number | null;
  error_text: string | null;
  reject_reason: string | null;
  provider_id: string | null;
  created_by: string;
  created_at: string;
  ready_at: string | null;
  terminated_at: string | null;
};

export type CreateComputeRequest = {
  name: string;
  gpu_type: string;
  gpu_count?: number;
  container_disk_gb?: number;
  volume_gb?: number;
  template_id: string;
  cloud_type?: "COMMUNITY" | "SECURE";
  provider_id?: string | null;
};

export type ComputeTemplate = {
  id: string;
  name: string;
  image: string;
  description: string;
};

export type ComputeSshInfo = {
  ssh_command: string;
  ssh_user: string;
  ssh_host: string;
  ssh_port: number;
  private_key: string;
};

// ---- Cloud providers (user-registered VMs / RunPod / PI accounts) ----

export type ProviderKind = "vm" | "runpod" | "pi";

export type ProviderRecord = {
  id: string;
  name: string;
  kind: ProviderKind;
  created_at: string;
  created_by: string;
  host?: string | null;
  port?: number | null;
  user?: string | null;
  gpus?: string[] | null;
  gpu_count?: number | null;
  api_key_last4?: string | null;
  ssh_pub?: string | null;
  validated_at?: string | null;
  account_email?: string | null;
};

export type VmConfigInput = {
  host: string;
  port: number;
  user: string;
  private_key?: string;
};

export type ApiKeyConfigInput = {
  api_key?: string;
};

export type CreateProviderRequest = {
  name: string;
  kind: ProviderKind;
  vm?: VmConfigInput;
  api?: ApiKeyConfigInput;
};

export type TestProviderRequest = {
  kind: ProviderKind;
  vm?: VmConfigInput;
  api?: ApiKeyConfigInput;
  provider_id?: string;
};

export type TestProviderResponse = {
  ok: boolean;
  message: string;
  gpus: string[];
  gpu_count: number;
};

export type GpuLiveInfo = {
  index: number;
  name: string;
  mem_free_mib: number;
  mem_total_mib: number;
  util_pct: number;
};

export type VmAvailability = {
  ok: boolean;
  message: string;
  gpus: GpuLiveInfo[];
  checked_at: number;
};

// ---- Admin: roles + audit ----

export type SectionKey = "inference" | "benchmark" | "compute";

export type AdminUserRecord = {
  id: number;
  username: string;
  email: string | null;
  role: "user" | "developer" | "admin";
  is_admin: boolean;
  policy_role_id: string | null;
  policy_role_name: string | null;
  section_permissions: Record<SectionKey, boolean>;
  created_at: string;
  auth_provider: "password" | "github";
  github_id: string | null;
};

export type PolicyRole = {
  id: string;
  name: string;
  sections: Record<SectionKey, boolean>;
  is_system: boolean;
  created_at: string;
};

export type AuditLogRecord = {
  id: number;
  actor_id: number | null;
  actor_username: string;
  action: string;
  resource_type: string;
  resource_id: string | null;
  resource_name: string | null;
  details: Record<string, unknown> | null;
  created_at: string;
};
