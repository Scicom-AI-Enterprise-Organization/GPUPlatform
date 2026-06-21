// Mirrors gateway/main.py Pydantic models. Keep field names + defaults in sync.

export type AutoscalerSpec = {
  max_containers: number;
  tasks_per_container: number;
  idle_timeout_s: number;
  // Per-app heartbeat liveness TTL (seconds). Optional for back-compat with
  // apps created before the setting existed; gateway defaults to 3600.
  worker_ttl_s?: number;
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
  // "transcription" → an audio/ASR (Whisper-family) model. Drives the audio-dep
  // install on the worker + the audio playground; set it for ASR models whose
  // name doesn't obviously say "whisper". Omitted/"generate" = a text model.
  task?: "generate" | "transcription" | null;
};

export type ServingMode = "single" | "multi" | "proxy";

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
  vllm_install_args?: string | null;
  pre_script?: string | null;
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
  // Full `uv pip install` arg string for vLLM, used verbatim instead of the version
  // (e.g. a nightly with extra index URLs). Overrides vllm_version when set.
  vllm_install_args?: string | null;
  // Optional setup script run once per worker boot before launching models, e.g.
  // `bash <(curl -fsSL …/install_deepgemm.sh)`. VM venv / RunPod multi-model only.
  pre_script?: string | null;
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

// One accuracy-eval result (a config × dataset), emitted by accuracy_eval.py
// and folded into BenchmarkRecord.result_json.accuracy. `accuracy` is a
// fraction 0..1; `output_tok_s` is the decode speed measured over the same
// requests — the two axes of the IQ-vs-speed plot.
export type BenchAccuracyResult = {
  config: string;
  dataset: string;
  accuracy: number;
  n: number;
  correct?: number;
  errors?: number;
  output_tokens?: number;
  elapsed_s?: number;
  output_tok_s?: number;
  // Rich per-eval metric bag for datasets that report more than a single
  // accuracy (e.g. Function-Call-TaaS: tool-call precision/recall/F1, name
  // accuracy, json_valid_rate, hallucination_rate, type_accuracy, and a nested
  // `_counts` breakdown). Absent for plain exact-match evals (GSM8K/MMLU).
  metrics?: Record<string, unknown> | null;
};

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
  // Public runs show up (read-only) in every user's list. is_owner is true when
  // the requesting user owns the run (controls whether edit/delete are offered).
  is_public?: boolean;
  is_owner?: boolean | null;
  created_at: string;
  started_at?: string | null;
  ended_at?: string | null;
  cost_per_hr?: number | null;
  provider_id?: string | null;
  storage_id?: string | null;
  env_vars?: Record<string, string> | null;
  visible_devices?: string | null;
  hf_token_secret?: string | null;
  cleanup_model?: boolean | null;
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
  // A global-secret key whose value is injected as HF_TOKEN at launch (gated
  // models). A pasted token is sent via env_vars["HF_TOKEN"] instead.
  hf_token_secret?: string | null;
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
  // Set when the user clicked "Stop & save" — the trainer is finishing the current
  // step + saving the partial model. Cleared implicitly when the run finalizes.
  stopping_early?: boolean;
  trials?: TrainingTrial[];
  // TTS audio eval (post-training): CER / MOS (UTMOSv2) / speaker similarity (TitaNet).
  tts_eval?: {
    samples?: number;
    cer?: number | null;
    mos?: number | null;
    similarity?: number | null;
  } | null;
  progress?: { step?: string; percent?: number } | null;
  // TTS only: the auto-created Label-platform recording+MOS project (post-train).
  label_project?: {
    id: string;
    url: string;
    count: number;
    dataset_id?: string | null;
    project_name?: string | null;
  } | null;
  // TTS only: status of an in-flight / finished post-train Label export. While
  // "running" the UI shows "exporting to Label" instead of the run's "done".
  label_export?: {
    status: "running" | "done" | "failed";
    error?: string | null;
  } | null;
  // On-demand "Export to Hugging Face" (pushes the best/final model). status +
  // resulting repo/url; surfaced on the run page.
  hf_export?: {
    status: "running" | "done" | "failed" | "cancelled";
    repo?: string | null;
    url?: string | null;
    error?: string | null;
  } | null;
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
  task_type?: "asr" | "tts" | "llm";
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
  task_type?: "asr" | "tts" | "llm";
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
  max_steps?: number;
  eval_strategy?: "epoch" | "steps";
  eval_steps?: number;
  save_strategy?: "epoch" | "steps";
  save_steps?: number;
  patience?: number;
  // "No test set" — train on everything with no eval / WER-CER / best-checkpoint /
  // early stop. test_dataset_id should be null when set.
  no_eval?: boolean;
  // Normalize text (case/punctuation) before WER/CER (Whisper-style). Off = raw.
  normalize_text?: boolean;
  eval_split_pct?: number;
  split_seed?: number;
  batch_size?: number;
  grad_accum?: number;
  learning_rate?: number;
  warmup_steps?: number;
  lr_scheduler_type?: "linear" | "cosine" | "constant_with_warmup" | "constant";
  weight_decay?: number;
  // LoRA / PEFT — adapters merged into the base at save time (drop-in checkpoint).
  use_lora?: boolean;
  lora_r?: number;
  // alpha as a ratio of r (alpha = round(r × ratio)); avoids permuting alpha.
  lora_alpha_ratio?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  // LLM-only (gemma4): which linear projections get LoRA. Default q/k/v/o; can add
  // MLP/dense layers (gate_proj, up_proj, down_proj). LLM finetune is always LoRA.
  lora_target_modules?: string[];
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
  eval_max_samples?: number;
  // TTS-only: after a successful run, synthesize N clips from the trained model and
  // auto-create a Label-platform recording+MOS project seeded with them. The token
  // is Fernet-encrypted into the run config server-side — never stored raw.
  label_export?: boolean;
  label_base_url?: string;
  label_base_url_secret?: string | null; // GlobalEnv key holding the URL (wins over label_base_url)
  label_token?: string | null;
  label_token_secret?: string | null;    // GlobalEnv key holding the lpat (wins over label_token)
  label_project_name?: string | null;
  label_samples?: number;
  label_mos_axes?: string[];
  label_speakers?: string[]; // balance synthesized eval clips across these speaker names
  label_speaker_prefix?: boolean; // prefix each task transcription with the speaker name
};

export type TrainingFile = {
  name: string;
  size: number;
  modified: string;
  download_url: string;
};

// An S3 object backing a dataset (Files tab). `name` is relative to the listed
// prefix; `key` is the full S3 key.
export type DatasetFile = {
  name: string;
  key: string;
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
  | "auto_terminated"
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
  // Auto-terminate after this many idle seconds (no GPU compute & no GPU memory
  // in use). 0 = disabled.
  idle_terminate_after_s: number;
  last_active_at: string | null;
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
  // Auto-terminate after this many idle seconds. 0 = off (default).
  idle_terminate_after_s?: number;
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

// RunPod account credit (USD) — GET /v1/providers/{id}/balance.
export type ProviderBalance = {
  ok: boolean;
  balance?: number | null;
  currency: string;
  message: string;
};

// ---- Storage backends (S3 / HuggingFace destinations the platform writes to) ----
export type StorageKind = "s3" | "huggingface" | "local" | "sftp";

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
  // huggingface: global-secret key the custom HF_ENDPOINT resolves from (if any).
  endpoint_secret?: string | null;
  // s3: global-secret keys the credentials resolve from (if any).
  access_key_id_secret?: string | null;
  secret_access_key_secret?: string | null;
  // local
  path?: string | null;
  // sftp (non-secret fields)
  host?: string | null;
  port?: number | null;
  username?: string | null;
  base_path?: string | null;
  enabled: boolean;
  notes?: string | null;
  created_at: string;
  created_by: string;
};

// ---- Datasets (Autotrain) ----

// "hosted" = a HuggingFace-mirror dataset repo pushed directly (hf upload /
// push_to_hub), surfaced in the Datasets list alongside Autotrain datasets.
// "llm_packed" = a chat dataset (kind=llm) tokenized + bin-packed into a
// ChiniDataset for LLM finetuning (the chat-text analogue of "tts_packed").
export type DatasetKind =
  | "upload"
  | "s3"
  | "hf"
  | "label"
  | "tts_packed"
  | "hosted"
  | "llm"
  | "llm_packed";

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
  messages_field?: string | null; // kind=llm: column holding the messages array
  label_base_url?: string | null; // kind=label source (token never returned)
  label_project_id?: string | null;
  label_status?: string | null; // approved | rejected | not_reviewed | all
  label_token_secret?: string | null; // global-secret key (if used instead of a stored token)
  transform_status?: string | null; // "" | running | done | failed
  transform_log?: string | null;
  catalog_repo_id?: string | null; // hosted HF-mirror dataset repo (if published)
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
  // kind=tts_packed / llm_packed — register existing ChiniDataset shards already in S3
  tokenizer?: string | null;
  sequence_length?: number | null;
  subset?: string | null; // kind=llm_packed — source subset that was packed (metadata)
  hf_repo?: string | null;
  hf_revision?: string | null; // kind=hf/llm — commit/branch/tag to pin
  messages_field?: string | null; // kind=llm / llm_packed — column holding the messages array
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

// Chat → multipack a kind=llm dataset's messages column into a ChiniDataset
// (kind=llm_packed) for LLM finetuning. Mirrors the gateway LlmPackRequest. Runs
// IN-PROCESS in the gateway (CPU tokenization — no GPU box).
export type LlmPackRequest = {
  storage_id: string; // kind=s3 storage for the packed shards
  tokenizer: string; // HF tokenizer (chat template), e.g. google/gemma-4-31B-it
  subset?: string | null; // which subset/split to pack (null → first)
  sequence_length?: number; // multipack bin length (tokens); longer convs dropped
  tools_field?: string | null; // source tool/function column (blank → no tools)
  all_reasoning?: boolean; // gemma-4: render every assistant turn's reasoning
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
  // s3 credentials by reference: a global-secret key resolved at use-time.
  access_key_id_secret?: string | null;
  secret_access_key_secret?: string | null;
  hf_token?: string | null;
  // Reference a global secret (admin Secrets) by key instead of a pasted token.
  hf_token_secret?: string | null;
  // huggingface: a global-secret key holding a custom HF_ENDPOINT.
  endpoint_secret?: string | null;
  // local
  path?: string | null;
  // sftp
  host?: string | null;
  port?: number | null;
  username?: string | null;
  password?: string | null;
  private_key?: string | null;
  base_path?: string | null;
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
  access_key_id_secret?: string | null;
  secret_access_key_secret?: string | null;
  hf_token?: string | null;
  hf_token_secret?: string | null;
  endpoint_secret?: string | null;
  path?: string | null;
  host?: string | null;
  port?: number | null;
  username?: string | null;
  password?: string | null;
  private_key?: string | null;
  base_path?: string | null;
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

// Live VM host metrics (CPU / memory / per-GPU), polled + graphed (not persisted).
export type ProviderGpuProc = {
  pid: number; // container-namespace pid (what ps/kill see on the box)
  comm: string;
  cmd: string;
};

export type ProviderGpuMetric = {
  index: number;
  name: string;
  util_pct: number;
  mem_used_mib: number;
  mem_total_mib: number;
  temp_c: number;
  // PCIe link — `cur` is live (GPUs downclock the link at idle), 0 = unknown.
  pcie_gen_cur?: number;
  pcie_width_cur?: number;
  pcie_gen_max?: number;
  pcie_width_max?: number;
  // NVLink — active links + aggregate per-direction GB/s. supported=false = PCIe-only.
  nvlink_supported?: boolean;
  nvlink_active?: number;
  nvlink_gbps?: number;
  processes?: ProviderGpuProc[];
};

export type ProviderDisk = {
  mount: string;
  used_bytes: number;
  total_bytes: number;
};

export type ProviderMetrics = {
  ok: boolean;
  message: string;
  cpu_pct: number; // overall CPU busy %, -1 if unavailable
  cpu_cores: number[]; // per-core busy % (htop-style)
  mem_used_mib: number;
  mem_total_mib: number;
  gpus: ProviderGpuMetric[];
  disks: ProviderDisk[]; // real filesystems (df), largest first
  checked_at: number;
};

// On-demand disk/memory/CPU bandwidth benchmark (button-triggered, not polled).
export type ProviderBandwidth = {
  ok: boolean;
  message: string;
  disk_write_mbps: number;
  disk_read_mbps: number;
  mem_mbps: number;
  cpu_model: string;
  cpu_mhz: number;
  checked_at: number;
};

// ---- Admin: roles + audit ----

export type SectionKey = "inference" | "benchmark" | "compute" | "datasets" | "catalog";

// ---- Model/Dataset catalog (self-hosted HuggingFace mirror) ----

export type CatalogRepoType = "model" | "dataset";

export type CatalogFile = {
  path: string;
  size?: number | null;
  lfs: boolean;
  oid?: string | null;
};

export type CatalogRecord = {
  id: string;
  repo_type: CatalogRepoType;
  namespace: string;
  name: string;
  full_id: string;
  storage_id?: string | null;
  storage_name?: string | null;
  prefix: string;
  sha?: string | null;
  private: boolean;
  description?: string | null;
  size_bytes?: number | null;
  num_files?: number | null;
  created_at: string;
  updated_at: string;
  created_by: string;
  files?: CatalogFile[] | null;
  // Versioned repos (mirror-native pushes) have named overwriteable branches;
  // flat repos (registered/published) are single-`main`. `revision` echoes which
  // branch the files/sha reflect.
  versioned?: boolean;
  default_branch?: string;
  revision?: string | null;
};

export type CatalogRef = {
  name: string;
  sha?: string | null;
  num_files?: number | null;
  size_bytes?: number | null;
};

export type CreateCatalogRequest = {
  repo_type: CatalogRepoType;
  namespace: string;
  name: string;
  storage_id: string;
  prefix?: string | null;
  private?: boolean;
  description?: string | null;
};

export type UpdateCatalogRequest = {
  private?: boolean;
  description?: string | null;
  storage_id?: string | null;
  prefix?: string | null;
};

export type CatalogDataPreview = {
  configs: string[];
  config: string;
  splits: string[];
  split: string;
  columns: string[];
  rows: Record<string, unknown>[];
  num_rows: number; // rows in the previewed shard
  shards: number;
  error?: string | null;
};

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

// ---- GitOps ----

export type GitopsSyncStatus = "never" | "syncing" | "ok" | "error";

export type GitopsRepo = {
  id: string;
  name: string;
  url: string;
  branch: string;
  path: string | null;
  token_secret: string | null;
  has_webhook_secret: boolean;
  prune: boolean;
  poll_interval: number;
  enabled: boolean;
  last_synced_sha: string | null;
  last_sync_at: string | null;
  last_sync_status: GitopsSyncStatus;
  last_sync_error: string | null;
  resource_count: number;
  created_at: string;
  created_by: string;
};

export type GitopsResource = {
  id: string;
  kind: string;
  name: string;
  resource_id: string | null;
  generation: number;
  status: "applied" | "error";
  error: string | null;
  last_synced_at: string;
};

export type CreateGitopsRepoBody = {
  name: string;
  url: string;
  branch?: string;
  path?: string | null;
  token_secret?: string | null;
  webhook_secret?: string | null;
  prune?: boolean;
  poll_interval?: number;
  enabled?: boolean;
};

export type UpdateGitopsRepoBody = Partial<CreateGitopsRepoBody>;

export type TestGitopsRepoBody = {
  url: string;
  branch?: string;
  token_secret?: string | null;
};

export type TestGitopsRepoResult = {
  ok: boolean;
  message: string;
  sha?: string | null;
};

export type GitopsSyncResult = {
  ok: boolean;
  skipped: boolean;
  sha: string | null;
  created: string[];
  updated: string[];
  pruned: string[];
  unchanged: number;
  errors: string[];
};

// ---- LLM API proxy ----

export type ProxyUpstream = {
  id: string;
  name: string;
  base_url: string;
  api_key_secret?: string | null;
  has_inline_key: boolean;
  models: Record<string, string>; // alias -> real upstream model
  priority: number;
  enabled: boolean;
};

export type ProxyEndpoint = {
  id: string;
  name: string;
  enabled: boolean;
  max_concurrency: number;
  timeout_s: number;
  upstreams: ProxyUpstream[];
  inflight: number;
  queued: number;
  created_at: string;
  created_by: string;
};

// Upstream spec sent on create/update. `api_key` is write-only (paste); blank on
// edit preserves the stored key. `api_key_secret` references a Secrets key.
export type ProxyUpstreamSpec = {
  id?: string;
  name: string;
  base_url: string;
  api_key_secret?: string | null;
  api_key?: string | null;
  models: Record<string, string>;
  priority: number;
  enabled: boolean;
};

export type CreateProxyBody = {
  name: string;
  max_concurrency?: number;
  timeout_s?: number;
  enabled?: boolean;
  upstreams: ProxyUpstreamSpec[];
};

export type UpdateProxyBody = Partial<CreateProxyBody>;

export type ProxyUpstreamHealth = {
  upstream_id: string;
  name: string;
  alive: boolean | null; // null = not probed yet
  latency_ms?: number | null;
  checked_at?: number | null;
  error?: string | null;
  stale: boolean;
};

export type ProxyRequest = {
  id: string;
  endpoint_id: string;
  owner?: string | null;
  model?: string | null;
  upstream?: string | null;
  status: "queued" | "running" | "completed" | "cancelled" | "failed";
  is_stream: boolean;
  status_code?: number | null;
  latency_ms?: number | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  error_text?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  live: boolean;
};

export type TestProxyUpstreamBody = {
  base_url: string;
  api_key_secret?: string | null;
  api_key?: string | null;
  model?: string | null; // real upstream model to chat-test; omitted = probe /models
};

export type TestProxyUpstreamResult = {
  ok: boolean;
  message: string;
  latency_ms?: number | null;
  models: string[];
};

