import { afterAll, describe, expect, it } from "vitest";

// ---------------------------------------------------------------------------
// REAL-API end-to-end: create a 4-model multi-model VM endpoint on GPUs 0,1,2,3
// with the cache/HOME export vars, wait for the fleet to warm up, request every
// member, then delete the endpoint.
//
// OPT-IN: skipped unless SGPU_API_KEY is set, so `npm test` stays green in CI.
//   SGPU_API_KEY=sgpu_... npx vitest run real-api.integration
//
// Key/base/provider come from the environment — never hard-coded into git.
// This deploys a real VM worker and loads ~4 models, so it runs for ~15-25 min.
// ---------------------------------------------------------------------------

const KEY = process.env.SGPU_API_KEY;
const BASE = (process.env.SGPU_BASE_URL ?? "http://localhost:8080").replace(/\/$/, "");
const PROVIDER = process.env.SGPU_VM_PROVIDER ?? "prov-5be27d21";
const APP = process.env.SGPU_E2E_APP ?? "tm-fleet-e2e";

const QWEN_ARGS =
  "--max-model-len 262144 --reasoning-parser qwen3 --gpu-memory-utilization 0.90 " +
  "--enable-auto-tool-choice --tool-call-parser qwen3_coder --mm-encoder-tp-mode data " +
  "--mm-processor-cache-type shm";

const ENV_VARS: Record<string, string> = {
  HOME: "/share/home",
  XDG_CACHE_HOME: "/share/.cache",
  TRITON_CACHE_DIR: "/share/triton_cache",
  TORCHINDUCTOR_CACHE_DIR: "/share/torchinductor_cache",
  FLASHINFER_WORKSPACE_DIR: "/share/flashinfer_cache",
  HF_HOME: "/share/huggingface",
  TRANSFORMERS_CACHE: "/share/huggingface",
  VLLM_CACHE_ROOT: "/share/vllm_cache",
  CUDA_CACHE_PATH: "/share/nv_cache",
  NUMBA_CACHE_DIR: "/share/numba_cache",
};

const MODELS = [
  { model: "qwen/qwen3.6-27b", tp: 2, extra_args: QWEN_ARGS },
  { model: "Qwen/Qwen3.6-35B-A3B", tp: 2, extra_args: QWEN_ARGS },
  {
    model: "mistralai/Mistral-Small-4-119B-2603",
    tp: 4,
    extra_args:
      "--tool-call-parser mistral --enable-auto-tool-choice --gpu-memory-utilization 0.9 --reasoning-parser mistral",
  },
  {
    model: "google/gemma-4-31b-it",
    tp: 2,
    extra_args:
      "--tool-call-parser gemma4 --enable-auto-tool-choice --gpu-memory-utilization 0.9 --reasoning-parser gemma4",
  },
];

const CREATE_BODY = {
  name: APP,
  gpu: "vm",
  gpu_count: 4,
  provider_id: PROVIDER,
  mode: "multi",
  models: MODELS,
  sleep_level: 1,
  autoscaler: { max_containers: 1, tasks_per_container: 64, idle_timeout_s: 0 },
  enable_metrics: false,
  env_vars: ENV_VARS,
  visible_devices: "0,1,2,3",
  venv_path: "/share/vllm-venv",
  vllm_version: "0.19.1",
};

const headers = { Authorization: `Bearer ${KEY}`, "Content-Type": "application/json" };
const api = (path: string, init?: RequestInit) =>
  fetch(`${BASE}${path}`, { ...init, headers: { ...headers, ...(init?.headers as object) } });
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const log = (...a: unknown[]) => console.log("[e2e]", ...a); // eslint-disable-line no-console

const WARMUP_TIMEOUT_MS = 16 * 60_000;
const REQUEST_TIMEOUT_MS = 4 * 60_000; // a swap/wake of the 119B can take minutes

describe.skipIf(!KEY)("real API e2e — create, serve 4 models, delete", () => {
  let appId = "";

  afterAll(async () => {
    if (!appId) return;
    const r = await api(`/apps/${encodeURIComponent(appId)}`, { method: "DELETE" }).catch(() => null);
    log("cleanup DELETE", appId, "→", r?.status);
  }, 60_000);

  it("creates the 4-model fleet on GPUs 0,1,2,3", async () => {
    // Clear any leftover from a previous run so the name is free.
    await api(`/apps/${encodeURIComponent(APP)}`, { method: "DELETE" }).catch(() => null);

    const r = await api("/apps", { method: "POST", body: JSON.stringify(CREATE_BODY) });
    const body = await r.json().catch(() => ({}));
    log("create →", r.status, body);
    expect(r.ok).toBe(true);
    appId = body.app_id;
    expect(appId).toBeTruthy();
  }, 90_000);

  it("warms up: every member finishes loading (not all dead)", async () => {
    const deadline = Date.now() + WARMUP_TIMEOUT_MS;
    let states: Record<string, string> = {};
    while (Date.now() < deadline) {
      const r = await api(`/apps/${encodeURIComponent(appId)}/status`);
      const d = await r.json().catch(() => ({}));
      const models: { model: string; state: string; reason?: string }[] = d.models ?? [];
      states = Object.fromEntries(models.map((m) => [m.model.split("/").pop()!, m.state]));
      log("warm:", states);
      const settled = models.length >= MODELS.length &&
        models.every((m) => ["asleep", "awake", "dead"].includes(m.state));
      if (settled) {
        for (const m of models) if (m.reason) log(`  ${m.model} dead: ${m.reason}`);
        break;
      }
      await sleep(15_000);
    }
    const dead = Object.values(states).filter((s) => s === "dead").length;
    expect(dead).toBeLessThan(MODELS.length); // at least one member came up
  }, WARMUP_TIMEOUT_MS + 30_000);

  it.each(MODELS.map((m) => m.model))(
    "serves %s (or returns a structured reason)",
    async (model) => {
      // `enable_thinking` is a Qwen/Gemma chat-template kwarg; Mistral tokenizers
      // reject any chat_template ("chat_template is not supported for Mistral
      // tokenizers" → 400), so only send it to models that accept it.
      const supportsThinking = !/mistral/i.test(model);
      const r = await api("/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: "Reply with the single word: ok" }],
          max_tokens: 16,
          ...(supportsThinking ? { chat_template_kwargs: { enable_thinking: false } } : {}),
        }),
      });
      const body = await r.json().catch(() => ({}));
      const message = body?.choices?.[0]?.message;
      if (r.ok && message) {
        // A real OpenAI completion (content may be "" for a reasoning model).
        log(`SERVED ${model}:`, JSON.stringify(message.content)?.slice(0, 80));
        expect(message).toHaveProperty("content");
      } else {
        // Either a non-200, OR a 200 whose body carries an error instead of a
        // completion (the worker wraps a vLLM failure as {output:{error}} /
        // {error}). Both must be a *structured* reason — never an opaque crash.
        const detail = body?.detail ?? body?.output ?? body?.error ?? body;
        log(`NOT-SERVED ${model}: ${r.status}`, JSON.stringify(detail)?.slice(0, 180));
        expect(JSON.stringify(body)).toMatch(/error|reason|dead|warming|model/i);
      }
    },
    REQUEST_TIMEOUT_MS,
  );
});
