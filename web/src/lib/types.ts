// Mirrors gateway/main.py Pydantic models. Keep field names + defaults in sync.

export type AutoscalerSpec = {
  max_containers: number;
  tasks_per_container: number;
  idle_timeout_s: number;
};

// A model served by a multi-model endpoint. `model` is the HuggingFace id and
// doubles as the served-model-name clients send in the OpenAI `model` field.
export type MultiModelMember = {
  model: string;
  tp: number;          // tensor-parallel size
  pp?: number;         // pipeline-parallel size; GPUs this model needs = tp * pp
  extra_args: string;  // per-model vLLM CLI args
  // Optional explicit GPU pin (physical ids within visible_devices, len == tp*pp).
  // null/omitted = auto-pack into the next free (tp*pp)-wide slot.
  gpu_indices?: number[] | null;
};

export type ServingMode = "single" | "multi";

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
  provider_id?: string | null;
  mode?: ServingMode;
  models?: MultiModelMember[] | null;
  sleep_level?: number;
  env_vars?: Record<string, string> | null;
  visible_devices?: string | null;
  venv_path?: string | null;
  vllm_version?: string | null;
  created_at: string;
  owner: string;
};

export type CreateAppRequest = {
  name: string;
  model?: string;
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
  mode?: ServingMode;
  models?: MultiModelMember[];
  sleep_level?: number;
  env_vars?: Record<string, string>;
  // VM-only GPU pin, e.g. "0,1,2,3". Empty/omitted = all the VM's GPUs.
  visible_devices?: string | null;
  // VM-only: uv venv the worker runs `vllm serve` from, e.g. "/share/vllm-venv".
  venv_path?: string | null;
  // VM-only: pin vLLM to this version in venv_path, e.g. "0.19.1".
  vllm_version?: string | null;
};

export type CreateAppResponse = {
  app_id: string;
  url: string;
};

// ---- OpenAI-compatible inference (request a specific model) ----

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export type ChatCompletionRequest = {
  // For a multi-model endpoint this is the member model name the gateway routes
  // by (e.g. "Qwen/Qwen3.6-27B"); for a single endpoint it's the endpoint name.
  model: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
  reasoning_effort?: "low" | "medium" | "high";
  chat_template_kwargs?: Record<string, unknown>;
  stream?: boolean;
};

export type ChatCompletionResponse = {
  id: string;
  object: string;
  model: string;
  choices: {
    index: number;
    message: { role: string; content: string | null; reasoning_content?: string | null };
    finish_reason: string | null;
  }[];
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
};

export type ModelsListResponse = {
  object: "list";
  data: { id: string; object: "model"; created: number; owned_by: string }[];
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

// ---- API keys (long-lived, revocable bearer tokens) ----
export type ApiKeyRecord = {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at?: string | null;
};

// The create response additionally carries the full plaintext key — shown once.
export type CreateApiKeyResponse = ApiKeyRecord & { key: string };

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
  storage_id?: string | null;
  env_vars?: Record<string, string> | null;
  visible_devices?: string | null;
};

export type CreateBenchmarkRequest = {
  name: string;
  config_yaml: string;
  provider_id?: string | null;
  // Storage backend (Storage row, kind=s3) for logs + result files. Required
  // by the form.
  storage_id?: string | null;
  // VM runs only: rm -rf the model's local_dir + HF hub cache on the VM after
  // the run exits. Default true on the UI side.
  cleanup_model?: boolean;
  // Extra env exported for the run (cache/home dirs, etc.). Absolute-path
  // values are mkdir -p'd on the VM; RunPod runs pass these to the pod.
  env_vars?: Record<string, string>;
  // CUDA_VISIBLE_DEVICES pin, e.g. "0,1,2,3". Empty = all GPUs.
  visible_devices?: string;
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

// ---- Autotrain (Whisper finetuning) ----

export type TrainingEpoch = {
  epoch: number;
  wer?: number | null;
  cer?: number | null;
  eval_loss?: number | null;
  train_loss?: number | null;
  // Sweep runs: which trial this eval belongs to (for per-trial eval curves/table).
  trial?: number | null;
};

export type TrainingTrial = {
  trial: number;
  params: Record<string, number | string>;
  metric?: number | null;
  status?: string;
};

export type TrainingStep = {
  step: number;
  loss?: number | null;
  lr?: number | null;
  epoch?: number | null;
  // Sweep runs: which trial this step belongs to (for per-trial loss curves).
  trial?: number | null;
};

export type TrainingGpuSample = { t: number; gpus: TrainingGpu[] };

export type TrainingResult = {
  epochs?: TrainingEpoch[];
  // Per-N-step training loss (@@STEP) for the live loss curve.
  steps?: TrainingStep[];
  // Per-poll GPU util/mem/temp samples, persisted so finished runs show the graph.
  gpu_samples?: TrainingGpuSample[];
  best?: {
    epoch?: number; wer?: number | null; cer?: number | null; eval_loss?: number | null;
    loss?: number | null;
    // sweep winner
    trial?: number; params?: Record<string, number | string>; metric?: number | null;
  } | null;
  artifact?: { s3_uri?: string | null; hf_repo?: string | null } | null;
  stopped_early?: boolean;
  trials?: TrainingTrial[];
  // TTS audio eval (post-training): CER / MOS (UTMOSv2) / speaker similarity (TitaNet).
  tts_eval?: {
    samples?: number;
    cer?: number | null;
    mos?: number | null;
    similarity?: number | null;
  } | null;
  progress?: { step?: string; percent?: number } | null;
  error?: string;
};

// Response of GET /v1/training-runs/:id/metrics — all persisted metrics in one
// call (works for finished runs too).
export type TrainingMetrics = {
  id: string;
  status: string;
  steps: TrainingStep[];
  epochs: TrainingEpoch[];
  gpu_samples: TrainingGpuSample[];
  trials?: TrainingTrial[] | null;
  best: TrainingResult["best"];
  artifact: TrainingResult["artifact"];
  stopped_early: boolean;
  error?: string | null;
};

export type TrainingRunRecord = {
  id: string;
  name: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  dataset_id: string;
  test_dataset_id?: string | null;
  base_model: string;
  task_type?: "asr" | "tts";
  s3_prefix: string;
  config_json: Record<string, unknown>;
  exit_code?: number | null;
  error_text?: string | null;
  result_json?: TrainingResult | null;
  created_by: string;
  created_at: string;
  started_at?: string | null;
  ended_at?: string | null;
  cost_per_hr?: number | null;
  provider_id?: string | null;
  provider_name?: string | null;
  provider_kind?: string | null;
  storage_id?: string | null;
  storage_name?: string | null;
  gpu_type?: string | null;
  gpu_count: number;
  visible_devices?: string | null;
};

export type CreateTrainingRunRequest = {
  name: string;
  dataset_id: string;
  base_model: string;
  task_type?: "asr" | "tts";
  test_dataset_id?: string | null;
  // TTS-only (Qwen3 + NeuCodec)
  tokenizer?: string | null;
  block_size?: number;
  pack_sequence_length?: number;
  default_speaker?: string | null;
  speaker_field?: string | null;
  // hyperparameter sweep: {param: [values]} → cross-product = trials.
  // values are numbers for most knobs; `precision` is a string list.
  sweep?: Record<string, (number | string)[]>;
  gpus_per_trial?: number;
  eval_metric?: "wer" | "cer";
  max_epochs?: number;
  patience?: number;
  // Normalize text (case/punctuation) before WER/CER (Whisper-style). Off = raw.
  normalize_text?: boolean;
  eval_split_pct?: number;
  split_seed?: number;
  batch_size?: number;
  grad_accum?: number;
  learning_rate?: number;
  warmup_steps?: number;
  weight_decay?: number;
  // LoRA / PEFT — adapters merged into the base at save time (drop-in checkpoint).
  use_lora?: boolean;
  lora_r?: number;
  // alpha as a ratio of r (alpha = round(r × ratio)); avoids permuting alpha.
  lora_alpha_ratio?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  // Freeze the encoder; train the decoder only.
  freeze_encoder?: boolean;
  // Multi-GPU single run: DDP via torchrun (default) vs DataParallel.
  use_ddp?: boolean;
  // "<load>-<amp>": weight load dtype + mixed-precision (AMP) train dtype.
  precision?: "fp32-bf16" | "bf16-bf16" | "fp32-fp16" | "fp16-fp16";
  language?: string | null;
  task?: string;
  provider_id?: string | null;
  gpu_type?: string;
  gpu_count?: number;
  secure_cloud?: boolean;
  disk_gb?: number;
  volume_gb?: number;
  visible_devices?: string | null;
  storage_id?: string | null;
  hf_push_repo?: string | null;
  // experiment tracking — non-secret per-run knobs only; creds come from the
  // global Secrets page.
  report_to?: ("mlflow" | "wandb")[];
  // named tracking credentials (Secrets page); selecting one enables that tracker
  wandb_credential_id?: string | null;
  mlflow_credential_id?: string | null;
  wandb_project?: string | null;
  wandb_entity?: string | null;
  mlflow_tracking_uri?: string | null;
  mlflow_experiment?: string | null;
  // Emit a training-loss point every N steps (@@STEP) for the live loss curve.
  logging_steps?: number;
  // OS env vars exported on the remote before the trainer (HOME, cache dirs, …).
  env_vars?: Record<string, string>;
  // Isolated uv venv for the trainer deps (like serverless's vLLM venv_path);
  // default /share/autotrain-whisper (asr) or /share/autotrain-tts (tts).
  venv_path?: string | null;
  // Roomy dir on the remote for checkpoints + temp (default /share; /tmp is small).
  work_dir?: string;
  // rm the checkpoint/work dir after the run (best model is on S3). Default true.
  cleanup_checkpoints?: boolean;
  // Training-audio augmentation: multi-select technique names. Empty = off.
  augment_techniques?: string[];
  augment_prob?: number;
  // TTS-only: audio eval methods to run on the test set (cer | mos | similarity).
  eval_methods?: string[];
};

export type TrainingFile = {
  name: string;
  size: number;
  modified: string;
  download_url: string;
};

export type TrainingGpu = {
  index: number;
  util: number;        // % GPU utilisation
  mem_used: number;    // MiB
  mem_total: number;   // MiB
  temp: number;        // °C
  name: string;
};

export type TrainingGpuResponse = {
  status?: string;
  gpus: TrainingGpu[];
  error?: string;
};

// Named experiment-tracker credentials (Secrets page → Tracking credentials).
export type TrackingCredentialRecord = {
  id: string;
  name: string;
  kind: "wandb" | "mlflow";
  preview: string;       // masked hint, never the secret
  created_by: string;
  created_at: string;
};

export type CreateTrackingCredentialRequest = {
  name: string;
  kind: "wandb" | "mlflow";
  api_key?: string;      // wandb
  uri?: string;          // mlflow
  username?: string;     // mlflow
  password?: string;     // mlflow
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
  // Required when template_id isn't one of the curated favourites — the
  // resolved imageName from the RunPod templates search.
  image?: string | null;
  cloud_type?: "COMMUNITY" | "SECURE";
  provider_id?: string | null;
};

export type RunpodTemplateSearchResult = {
  id: string;
  name: string;
  image: string;
  category?: string | null;
  is_public: boolean;
  is_runpod: boolean;
};

export type PiImageOption = {
  id: string;
  name: string;
  description: string;
};

export type GpuTypeOption = {
  id: string;
  label: string;
  vram_gb: number;
  hint: string;
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

// ---- Storage backends (S3 / HuggingFace destinations the platform writes to) ----
export type StorageKind = "s3" | "huggingface";

export type StorageRecord = {
  id: string;
  name: string;
  kind: StorageKind;
  bucket?: string | null;
  prefix?: string | null;
  region?: string | null;
  endpoint?: string | null;
  has_credentials: boolean;
  hf_token_secret?: string | null;
  enabled: boolean;
  notes?: string | null;
  created_at: string;
  created_by: string;
};

// ---- Datasets (Autotrain) ----

export type DatasetKind = "upload" | "s3" | "hf" | "label" | "tts_packed";

export type DatasetRecord = {
  id: string;
  name: string;
  description?: string | null;
  kind: DatasetKind;
  storage_id?: string | null;
  storage_name?: string | null;
  audio_prefix?: string | null;
  s3_metadata_uri?: string | null;
  size_bytes?: number | null;
  metadata_filename?: string | null;
  format?: string | null;
  num_rows?: number | null;
  audio_field: string;
  transcription_field: string;
  speaker_field?: string | null; // TTS-only speaker column (null → one voice)
  split_fields?: Record<string, string> | null; // per-split transcription overrides
  audio_dataset_id?: string | null; // materialised S3 audio dataset (source → output link)
  // Lineage for a transformed dataset: the dataset it was derived from + that
  // source's original HF repo (computed server-side).
  source_dataset_id?: string | null;
  source_name?: string | null;
  source_hf_repo?: string | null;
  hf_repo?: string | null;
  hf_revision?: string | null;
  hf_synced_at?: string | null;
  label_base_url?: string | null; // kind=label source (token never returned)
  label_project_id?: string | null;
  label_status?: string | null; // approved | rejected | not_reviewed | all
  label_token_secret?: string | null; // global-secret key (if used instead of a stored token)
  transform_status?: string | null; // "" | running | done | failed
  transform_log?: string | null;
  created_at: string;
  updated_at: string;
  created_by: string;
};

export type CreateDatasetRequest = {
  name: string;
  kind: DatasetKind;
  storage_id?: string | null;
  description?: string | null;
  audio_prefix?: string | null;
  s3_metadata_uri?: string | null;
  hf_repo?: string | null;
  // kind=label — import from a labeling-platform project
  label_base_url?: string | null;
  label_project_id?: string | null;
  label_token?: string | null;
  label_token_secret?: string | null; // OR: a global-secret key holding the token
  label_status?: string | null; // approved | rejected | not_reviewed | all
};

export type UpdateDatasetRequest = {
  name?: string;
  description?: string | null;
  audio_prefix?: string | null;
  audio_field?: string;
  transcription_field?: string;
};

export type DatasetUploadResult = {
  filename: string;
  format: string;
  num_rows: number;
  columns: string[];
  audio_field: string;
  transcription_field: string;
  preview: Record<string, unknown>[];
};

export type DatasetPreviewRow = {
  audio_url?: string | null;
  transcription?: unknown;
  row_index?: number; // stable metadata-file index (for include/exclude)
  included?: boolean; // false → manually excluded from training
  [k: string]: unknown;
};

export type DatasetPreview = {
  audio_field: string;
  transcription_field: string;
  rows: DatasetPreviewRow[];
  offset?: number;
  limit?: number;
  total?: number | null;
  split?: string | null; // which HF split these rows came from
  splits?: string[] | null; // available HF splits (for a picker)
  speakers?: string[] | null; // distinct speaker values (for a filter dropdown)
  speaker?: string | null; // the selected speaker filter, echoed back
  excluded_count?: number; // rows manually un-ticked (excluded from training)
  error?: string | null;
};

export type SyncDatasetRequest = { hf_repo: string; private: boolean };

// Turn a source dataset (hf archive / label-platform export) into one with a
// real audio column — pushed to HF or materialised to S3. Mirrors the gateway
// TransformRequest. Runs as a gateway background job (poll transform_status).
export type TransformDatasetRequest = {
  target: "hf" | "s3";
  hf_repo?: string | null; // required for target=hf (owner/name)
  storage_id?: string | null; // required for target=s3 (a kind=s3 storage)
  s3_folder?: string | null; // target=s3 dest folder; blank → datasets/{id}/transformed
};

// NeuCodec-encode + multipack a {audio, transcription} dataset into a packed
// (tts_packed) dataset on a GPU. Mirrors the gateway TtsPackRequest. provider_id
// = a VM (SSH) or RunPod account; null → spawn a pod with the gpu_type/tier below.
export type TtsPackRequest = {
  provider_id?: string | null;
  storage_id: string;
  tokenizer?: string | null;
  sequence_length?: number;
  gpu_count?: number;
  visible_devices?: string | null;
  venv_path?: string | null; // isolated uv venv for the NeuCodec/TTS deps
  // RunPod pod knobs (ignored for a VM provider)
  gpu_type?: string;
  secure_cloud?: boolean;
  disk_gb?: number;
  volume_gb?: number;
};

// Org-wide env var / secret (admin-managed). Mirrors gateway GlobalEnvRecord.
// `value` is plaintext for non-secrets, null for secrets; `value_preview` is a
// masked hint for secrets.
export type GlobalEnvRecord = {
  key: string;
  is_secret: boolean;
  value: string | null;
  value_preview: string | null;
  description: string | null;
  updated_by: string;
  updated_at: string;
};

export type CreateStorageRequest = {
  name: string;
  kind: StorageKind;
  bucket?: string | null;
  prefix?: string | null;
  region?: string | null;
  endpoint?: string | null;
  access_key_id?: string | null;
  secret_access_key?: string | null;
  hf_token?: string | null;
  // Reference a global secret (admin Secrets) by key instead of a pasted token.
  hf_token_secret?: string | null;
  notes?: string | null;
  enabled?: boolean;
};

export type UpdateStorageRequest = Partial<Omit<CreateStorageRequest, "kind">>;

export type TestStorageRequest = {
  kind: StorageKind;
  bucket?: string | null;
  region?: string | null;
  endpoint?: string | null;
  access_key_id?: string | null;
  secret_access_key?: string | null;
  hf_token?: string | null;
  hf_token_secret?: string | null;
};

export type TestStorageResponse = {
  ok: boolean;
  message: string;
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

export type SectionKey = "inference" | "benchmark" | "compute" | "datasets";

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
  actor_email?: string | null;
  action: string;
  resource_type: string;
  resource_id: string | null;
  resource_name: string | null;
  details: Record<string, unknown> | null;
  created_at: string;
};
