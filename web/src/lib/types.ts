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
  provider_name?: string | null;
  storage_id?: string | null;   // log-archive Storage (kind=s3); null = Redis-only
  storage_name?: string | null; // resolved on the single-app GET
  mode?: ServingMode;
  models?: MultiModelMember[] | null;
  sleep_level?: number;
  env_vars?: Record<string, string> | null;
  visible_devices?: string | null;
  venv_path?: string | null;
  vllm_version?: string | null;
  vllm_install_args?: string | null;
  pre_script?: string | null;
  // Multi-model fleet auto-retry (crash recovery) tuning; null = worker default.
  retry_max?: number | null;
  retry_forever?: boolean | null;
  retry_backoff_base_s?: number | null;
  retry_backoff_cap_s?: number | null;
  retry_require_free_gpu?: boolean | null;
  retry_gpu_free_pct?: number | null;
  health_fail_limit?: number | null;
  is_public?: boolean;
  created_at: string;
  owner: string;
};

// An LLM API proxy that fronts a serverless endpoint (matched by upstream URL or
// model). Secret-stripped: only name + stable serving path + model aliases.
export type AppProxyLink = {
  id: string;
  name: string;
  public: boolean;
  serving_path: string;
  models: string[];
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
  // Create already public (read-only visible to every logged-in user). Omitted/
  // false = private. Same flag the /apps/{id}/visibility toggle flips.
  is_public?: boolean;
  cloud_type?: "COMMUNITY" | "SECURE";
  // RunPod data center to pin (empty/omitted → auto). See RegionOption.
  data_center_id?: string;
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
  // ---- Multi-model fleet auto-retry (crash recovery); omitted = worker default ----
  // Max relaunch attempts before a crashed member is left DEAD.
  retry_max?: number | null;
  // Never give up — relaunch indefinitely, ignoring retry_max (waits for free GPU
  // memory forever when retry_require_free_gpu is on).
  retry_forever?: boolean | null;
  // Backoff between relaunches (seconds): initial delay, doubled per attempt up to
  // the cap (the "patience" ceiling — longest wait before a retry).
  retry_backoff_base_s?: number | null;
  retry_backoff_cap_s?: number | null;
  // Hold a relaunch until the member's GPUs have free VRAM (don't OOM-loop against a
  // foreign job); polls without spending the retry budget.
  retry_require_free_gpu?: boolean | null;
  // Min free GPU memory (% of total) required to relaunch when the above is on.
  retry_gpu_free_pct?: number | null;
  // Consecutive failed /health probes before a settled engine is declared dead.
  health_fail_limit?: number | null;
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

/** One page of a server-paginated list (benchmarks / training runs / datasets).
 * `total` counts ALL matching rows so the pager can show page numbers. */
export type PageResponse<T> = { total: number; items: T[] };

/** Slim per-run numbers for the /benchmark dashboard KPI row — fetched from
 * /benchmarks/_stats so the dashboard doesn't need every full record. */
export type BenchStat = {
  id: string;
  status: string;
  model: string | null;
  gpu_type: string | null;
  gpu_count: number | null;
  output_throughput: number | null;
  duration_s: number | null;
};

/** Shared query params for the server-paginated list endpoints. */
export type PageQuery = {
  scope?: "mine" | "all";
  q?: string;
  status?: string; // benchmarks + training runs (datasets use `kind`)
  kind?: string; // datasets source filter
  sort?: "newest" | "oldest";
  limit?: number;
  offset?: number;
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
  // Effective GPU identity, resolved server-side: the manually-set value when
  // present (the only way ingress/Slurm runs get one — set at create or via
  // updateBenchmark), else the config's runpod.pod / top-level gpu_type.
  gpu_type?: string | null;
  gpu_count?: number | null;
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
  // Ingress only: a global-secret key whose value is injected as OPENAI_API_KEY
  // at launch (the ingress client sends it as Authorization: Bearer). A pasted
  // key is sent via env_vars["OPENAI_API_KEY"] instead.
  api_key_secret?: string | null;
  // Create already public (read-only visible to every logged-in user). Omitted/
  // false = private. Same flag the /benchmarks/{id}/visibility toggle flips.
  is_public?: boolean;
  // GPU identity for runs the platform doesn't provision (ingress/Slurm): the
  // hardware behind base_url, e.g. "NVIDIA H20". Also settable as a top-level
  // gpu_type/gpu_count key in the YAML, or later on the detail page.
  gpu_type?: string | null;
  gpu_count?: number | null;
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

// One auto-created Label-platform recording+MOS project (post-train TTS). `speaker`
// is set only when projects were split per speaker.
export type LabelProjectCard = {
  id: string;
  url: string;
  count: number;
  dataset_id?: string | null;
  project_name?: string | null;
  speaker?: string | null;
  // human_mos (LLM label export) projects carry a type discriminator.
  project_type?: string | null;
};

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
  // label_project is the first project (back-compat); label_projects holds all of
  // them — one per speaker when "separate project per speaker" was used.
  label_project?: LabelProjectCard | null;
  label_projects?: LabelProjectCard[] | null;
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

// ---- Quantization (llm-compressor) ----

export type QuantizationSchemesResponse = {
  // scheme id → {label, whether it needs a calibration dataset}
  schemes: Record<string, { label: string; needs_calibration: boolean }>;
  calib_dataset_kinds: string[];
};

export type QuantizationResult = {
  artifact?: string | null; // s3:// uri of the compressed model
  hf_repo?: string | null;
  sizes?: { source_gb?: number; quantized_gb?: number } | null;
  progress?: { stage?: string; percent?: number } | null;
  hf_export?: { status?: string; repo?: string; url?: string; error?: string } | null;
  error?: string;
};

export type QuantizationJobRecord = {
  id: string;
  name: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  source_model: string;
  scheme: string;
  calibration_dataset_id?: string | null;
  s3_prefix: string;
  config_json: Record<string, unknown>;
  exit_code?: number | null;
  error_text?: string | null;
  result_json?: QuantizationResult | null;
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

export type CreateQuantizationJobRequest = {
  name: string;
  source_model: string;
  scheme: string;
  calibration_dataset_id?: string | null;
  num_calibration_samples?: number;
  max_seq_length?: number;
  calib_text_field?: string | null;
  calib_messages_field?: string | null;
  ignore_layers?: string[];
  quantize_vision?: boolean;
  smoothing_strength?: number;
  dampening_frac?: number;
  hf_push_repo?: string | null;
  hf_push_private?: boolean;
  hf_token?: string | null;
  hf_token_secret?: string | null;
  provider_id?: string | null;
  storage_id?: string | null;
  gpu_type?: string;
  gpu_count?: number;
  visible_devices?: string | null;
  secure_cloud?: boolean;
  data_center_id?: string | null;
  disk_gb?: number;
  volume_gb?: number;
  image?: string | null;
  work_dir?: string | null;
  venv_path?: string | null;
  env_vars?: Record<string, string>;
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
  cpu_offload?: boolean | null;  // LLM FSDP CPU offload; null = per-arch default
  // LLM context parallelism (gemma4 zigzag ring / qwen3.5-3.6 GatedDeltaNet state relay + full-attn
  // ring) — shards one long packed sequence across all GPUs so context longer than one GPU's VRAM
  // can train. Needs >=2 GPUs.
  context_parallel?: boolean | null;
  // CP group size (GPUs per group that shard one sequence). null/0 → all run GPUs (dp=1); when set
  // it must divide the run's GPU count (dp_size = world / cp_size). Only used when context_parallel.
  cp_size?: number | null;
  // LLM training objective: "sft" (default, kind=llm_packed dataset) or "dpo"
  // (kind=llm_dpo_packed preference pairs; qwen base models only; no context parallel).
  training_type?: "sft" | "dpo";
  dpo_beta?: number; // DPO temperature β (training_type=dpo)
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
  // LLM-only (gemma4): also full-train embed_tokens + lm_head (tied → one ~1.4B weight)
  // on top of LoRA — helps the model reliably emit special tokens (e.g. tool calls).
  train_embeddings?: boolean;
  // LLM-only: use DoRA (weight-decomposed LoRA) instead of plain LoRA for every adapted module
  // (attention + the fused MoE experts on minimax/mistral/qwen-MoE/gemma-MoE). Incompatible with DPO.
  use_dora?: boolean;
  // LLM MoE-only: skip the fused routed-expert adapter (adapt attention only). Experts are 3D
  // tensors, not nn.Linear, so they're not in lora_target_modules — they're adapted by default;
  // set this to opt out. No effect on dense models / nemotron (experts always frozen there).
  no_moe_lora?: boolean;
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
  // RunPod data center to pin (empty/omitted → Auto). Cloud-only.
  data_center_id?: string;
  disk_gb?: number;
  volume_gb?: number;
  visible_devices?: string | null;
  storage_id?: string | null;
  hf_push_repo?: string | null;
  // HF token for the run (gated/private datasets + push to Hub). hf_token_secret = a
  // Secrets-page key reference (wins); hf_token = a pasted token (stored encrypted).
  hf_token?: string | null;
  hf_token_secret?: string | null;
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
  label_reject_keywords?: string[]; // drop text samples containing any of these phrases
  label_per_speaker?: boolean; // one project per speaker, each from that speaker's own clips
};

export type TrainingFile = {
  name: string;
  size: number;
  modified: string;
  download_url: string;
};

// Where a finished run's "Try it" playground runs inference — chosen at load time,
// decoupled from where the run trained. "cloud" spins up a fresh RunPod pod with the
// given GPU; "vm" reuses a registered VM provider. `provider_id` is the RunPod account
// (cloud) or the VM provider (vm). Mirrors the serverless deploy form's Run-on/Pod cards.
export type TryItTarget = {
  target: "cloud" | "vm";
  provider_id?: string | null;
  // cloud only — the pod to provision (gpu_type is a GPU_CHOICES catalog value):
  gpu_type?: string;
  gpu_count?: number;
  cloud_type?: "SECURE" | "COMMUNITY";
};

// Live state of the persistent try-it worker / on-demand try-it pod.
export type PlaygroundStatus = {
  running: boolean;
  ready: boolean;
  device?: string;
  kind?: string;
  logs?: string[];
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
  // RunPod data center to pin (empty/omitted → Auto). RunPod-only.
  data_center_id?: string;
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

// A curated RunPod data center for region pinning (GET /compute/runpod/regions).
// `id` is the RunPod `dataCenterIds` value; empty selection ("Auto") pins nothing.
export type RegionOption = {
  id: string;
  label: string;
  country?: string;
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
  jump_host?: string | null; // set when the VM is reached via ProxyJump
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
// "llm_dpo_packed" = chosen/rejected preference PAIRS packed for DPO training
// (whole pairs per bin, chosen-first layout — Pack for LLM with objective=dpo).
export type DatasetKind =
  | "upload"
  | "s3"
  | "hf"
  | "label"
  | "tts_packed"
  | "omnivoice_packed" // {audio,text} Higgs-codec → OmniVoice WebDataset shards (Pack for OmniVoice)
  | "hosted"
  | "llm"
  | "llm_packed"
  | "llm_dpo_packed";

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
  messages_field?: string | null; // kind=llm: column holding the messages array (= chosen in DPO mode)
  rejected_field?: string | null; // kind=llm DPO (preference) mode: rejected-response column (null → chat mode)
  label_base_url?: string | null; // kind=label source (token never returned)
  label_project_id?: string | null;
  label_status?: string | null; // approved | rejected | not_reviewed | all
  label_updated_until?: string | null; // ISO-8601 point-in-time import cutoff (null → no upper bound)
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
  messages_field?: string | null; // kind=llm / llm_packed — column holding the messages array (= chosen in DPO mode)
  rejected_field?: string | null; // kind=llm DPO mode — rejected-response column
  // kind=label — import from a labeling-platform project
  label_base_url?: string | null;
  label_project_id?: string | null;
  label_token?: string | null;
  label_token_secret?: string | null; // OR: a global-secret key holding the token
  label_status?: string | null; // approved | rejected | not_reviewed | all
  label_updated_until?: string | null; // ISO-8601 cutoff — import only tasks last updated at/before it
};

export type UpdateDatasetRequest = {
  name?: string;
  description?: string | null;
  audio_prefix?: string | null;
  audio_field?: string;
  transcription_field?: string;
  speaker_field?: string;
  // kind=llm column mapping. messages_field = the messages/chosen column; rejected_field
  // set → DPO (preference) mode, "" → chat mode. null → leave unchanged.
  messages_field?: string | null;
  rejected_field?: string | null;
  // kind=label import filters (null → unchanged; pass "" for label_updated_until to clear the cutoff)
  label_status?: string | null;
  label_updated_until?: string | null;
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
  test_split_pct?: number | null; // hold out this % of rows as a `test` split (0–100)
  test_split_count?: number | null; // hold out this many rows as a `test` split (overrides pct)
  test_min_chars?: number | null; // min transcription length (chars) to be eligible for the test split; short/junk transcripts stay in train
  test_exclude_regex?: string | null; // regex; transcripts matching it (re.search) are excluded from the test split (kept in train)
  test_split_ref_dataset_id?: string | null; // reuse this dataset's exact test set (matched by audio); mutually exclusive with pct/count
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
  data_center_id?: string; // RunPod region pin ("" / omitted → Auto)
  disk_gb?: number;
  volume_gb?: number;
};

// Higgs-codec tokenize an {audio,text} dataset into OmniVoice WebDataset shards
// (kind=omnivoice_packed) on a GPU. Mirrors the gateway OmnivoicePackRequest.
export type OmnivoicePackRequest = {
  provider_id?: string | null;
  storage_id: string;
  tokenizer?: string | null;          // Higgs codec
  default_language?: string | null;   // language_id when the dataset has no language column
  language_field?: string | null;     // dataset column holding per-row language_id
  eval_test_per_speaker?: number;
  gpu_count?: number;
  visible_devices?: string | null;
  venv_path?: string | null;          // null → /share/autotrain-omnivoice
  // RunPod pod knobs (OmniVoice needs CUDA 12.8)
  gpu_type?: string;
  image?: string | null;
  secure_cloud?: boolean;
  data_center_id?: string; // RunPod region pin ("" / omitted → Auto)
  disk_gb?: number;
  volume_gb?: number;
};

// Chat → multipack a kind=llm dataset's messages column into a ChiniDataset
// (kind=llm_packed) for LLM finetuning. Mirrors the gateway LlmPackRequest. Runs
// IN-PROCESS in the gateway (CPU tokenization — no GPU box).
export type LlmPackRequest = {
  storage_id: string; // kind=s3 storage for the packed shards
  tokenizer: string; // HF tokenizer (chat template), e.g. google/gemma-4-31B-it
  subset?: string | null; // single subset/split to pack (null → first); legacy/fallback
  subsets?: string[] | null; // multiple subset/split labels packed together (rows concatenated); takes precedence over subset
  sequence_length?: number; // multipack bin length (tokens); longer convs dropped
  tools_field?: string | null; // source tool/function column (blank → no tools)
  all_reasoning?: boolean; // render every assistant turn's reasoning (gemma-4/MiniMax-M2 templates; no-op otherwise)
  objective?: "sft" | "dpo"; // dpo = pack chosen/rejected preference pairs → kind=llm_dpo_packed
  chosen_field?: string; // objective=dpo: preferred-response column (messages list or string)
  rejected_field?: string; // objective=dpo: dispreferred-response column
  prompt_field?: string | null; // objective=dpo: shared prompt column (only when chosen/rejected are strings)
};

// Concatenate >=2 kind=label datasets into one combined audio dataset (HF or
// S3). Mirrors the gateway DatasetMergeRequest. Background job: the response is
// the new dataset (transform_status=running) — poll getDataset(id).
export type DatasetMergeRequest = {
  source_ids: string[]; // >= 2 kind=label dataset ids
  target: "hf" | "s3";
  hf_repo?: string | null; // owner/name (target=hf)
  storage_id?: string | null; // kind=s3 storage (target=s3)
  s3_folder?: string | null; // blank → datasets/{new_id}/transformed
  name?: string | null; // output dataset name (blank → auto)
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
  // Key OR password (key preferred when both).
  private_key?: string;
  password?: string;
  // Optional jump host (ProxyJump) for boxes not directly reachable.
  jump_host?: string;
  jump_port?: number;
  jump_user?: string;
  jump_private_key?: string;
  jump_password?: string;
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
  gpu_mem_mib?: number; // this process's VRAM on the GPU (from nvidia-smi)
  gpus?: string; // host procs: GPU device indices it has open (if fd readable)
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
  // Huawei Ascend (npu-smi): kind="npu", util_pct=AICore%, mem=HBM.
  kind?: "gpu" | "npu";
  power_w?: number;
  health?: string;
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
  host_gpu_procs?: ProviderGpuProc[]; // GPU procs found via /proc (cmd + container pid), not GPU-mapped
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

export type SectionKey = "inference" | "benchmark" | "compute" | "datasets" | "catalog" | "quantization";

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
  public?: boolean;
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
  public?: boolean;
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
  model?: string | null; // real upstream model to end-to-end test; omitted = probe /models
  mode?: "chat" | "embedding"; // which endpoint to test (default chat)
};

export type TestProxyUpstreamResult = {
  ok: boolean;
  message: string;
  latency_ms?: number | null;
  models: string[];
};


// ---- Usage activity analytics (the "Activity" dashboard) ----
export type ActivityGranularity = "15min" | "hour" | "day";

export type ActivitySummary = {
  window: { since: string | null; until: string | null; tz: string; granularity: ActivityGranularity };
  totals: {
    requests: number;
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    avg_ttft_ms: number | null;
    avg_latency_ms: number | null;
    p50_ttft_ms: number | null;
    p95_ttft_ms: number | null;
    p99_ttft_ms: number | null;
    p50_latency_ms: number | null;
    p95_latency_ms: number | null;
    p99_latency_ms: number | null;
    requests_ok: number;
    requests_error: number;
    requests_pending: number;
    success_rate: number | null;
  };
  by_bucket: {
    bucket: string;
    requests: number;
    prompt_tokens: number;
    completion_tokens: number;
    avg_ttft_ms: number | null;
    avg_latency_ms: number | null;
    p50_ttft_ms: number | null;
    p95_ttft_ms: number | null;
    p99_ttft_ms: number | null;
    p50_latency_ms: number | null;
    p95_latency_ms: number | null;
    p99_latency_ms: number | null;
  }[];
  by_model: { model: string; requests: number; prompt_tokens: number; completion_tokens: number }[];
  top_users: { user: string; owner_id: number | null; requests: number; prompt_tokens: number; completion_tokens: number }[];
  by_model_bucket: { bucket: string; model: string; requests: number; tokens: number }[];
  by_user_bucket: { bucket: string; user: string; requests: number; tokens: number }[];
  by_upstream_bucket: { bucket: string; upstream: string; requests: number; tokens: number }[];
  // Avg latency/TTFT per bucket, split by series (top-N + "other"). `series` = the
  // model / upstream name. avg_* are null in buckets where that series had no timing.
  by_model_latency_bucket: { bucket: string; series: string; avg_latency_ms: number | null; avg_ttft_ms: number | null; requests: number }[];
  by_upstream_latency_bucket: { bucket: string; series: string; avg_latency_ms: number | null; avg_ttft_ms: number | null; requests: number }[];
  by_status_bucket: { bucket: string; status: "ok" | "error" | "pending"; requests: number }[];
  by_source_bucket: { bucket: string; source: "serverless" | "proxy"; requests: number; tokens: number }[];
  all_models: string[];
  note: string;
};

// ---- serverless endpoint log files (Logs tab) ----
export type LogFile = {
  id: string;          // "{slug}:{session}"
  source: string;      // model name, or "__worker__" for the worker-agent log
  slug: string;
  session: string;     // "YYYYMMDD-HHMMSS"
  started_at: string;  // ISO
  bytes: number | null;
  lines: number;
  crash: string | null;
  archived: boolean;   // persisted to s3 (vs Redis-only)
  live: boolean;       // session updated within the last minute
};
export type LogFilesResponse = { files: LogFile[]; total: number; archived: boolean };
export type LogFileContent = {
  lines: string[];
  count: number;
  archived: boolean;
  session: string;
  source: string;
  crash: string | null;
};

export type ActivityLogRow = {
  kind: string;
  id: string;
  name: string | null;
  user: string;
  status: string;
  created_at: string | null;
  ended_at: string | null;
  duration_s: number | null;
  detail: {
    endpoint?: string;
    model?: string | null;
    upstream?: string | null;
    is_stream?: boolean;
    status_code?: number | null;
    prompt_tokens?: number | null;
    completion_tokens?: number | null;
    ttft_ms?: number | null;
    latency_ms?: number | null;
  };
};

export type ActivityLogsResponse = {
  kind: string;
  count: number;
  has_more: boolean;
  jobs: ActivityLogRow[];
};
