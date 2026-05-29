import { afterEach, describe, expect, it, vi } from "vitest";
import type { CreateAppRequest, MultiModelMember } from "@/lib/types";

// gateway.createApp runs server-side (typeof window === "undefined"), where it
// pulls the bearer token from the `sgpu_token` cookie via next/headers. Mock it.
// `mockToken` is mutable so a test can simulate the no-session case; vitest
// allows factory references to vars prefixed with `mock`.
let mockToken: string | undefined = "test-token";
vi.mock("next/headers", () => ({
  cookies: async () => ({
    get: (name: string) =>
      name === "sgpu_token" && mockToken ? { value: mockToken } : undefined,
  }),
}));

// Import AFTER the mock is registered.
const { gateway, GatewayError } = await import("@/lib/gateway");

// ---- the battle config: the exact multi-model SSH fleet the user deployed ----

const QWEN_ARGS =
  "--max-model-len 262144 --reasoning-parser qwen3 --gpu-memory-utilization 0.90 " +
  "--enable-auto-tool-choice --tool-call-parser qwen3_coder --mm-encoder-tp-mode data " +
  "--mm-processor-cache-type shm";

const BATTLE_ENV: Record<string, string> = {
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

const BATTLE_MODELS: MultiModelMember[] = [
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

const BATTLE_REQUEST: CreateAppRequest = {
  name: "tm-fleet",
  gpu: "VM",
  gpu_count: 4,
  mode: "multi",
  provider_id: "prov-5be27d21",
  visible_devices: "0,1,2,3",
  venv_path: "/share/vllm-venv",
  vllm_version: "0.19.1",
  sleep_level: 1,
  env_vars: BATTLE_ENV,
  models: BATTLE_MODELS,
};

function jsonResponse(obj: unknown, status = 200): Response {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

type FetchArgs = [input?: string, init?: RequestInit];

/** Stub fetch and return the captured (url, init) of the single call. */
function stubFetch(impl: (...args: FetchArgs) => Promise<Response>) {
  const fetchMock = vi.fn(impl);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function sentBody(fetchMock: ReturnType<typeof stubFetch>): CreateAppRequest {
  const init = fetchMock.mock.calls[0][1] as RequestInit;
  return JSON.parse(init.body as string) as CreateAppRequest;
}

afterEach(() => {
  vi.unstubAllGlobals();
  mockToken = "test-token";
});

describe("gateway.createApp — multi-model SSH serverless via the JS API", () => {
  it("POSTs /apps with method, JSON content-type and bearer auth", async () => {
    const fetchMock = stubFetch(async () =>
      jsonResponse({ app_id: "tm-fleet", url: "/run/tm-fleet" }),
    );

    const res = await gateway.createApp(BATTLE_REQUEST);
    expect(res).toEqual({ app_id: "tm-fleet", url: "/run/tm-fleet" });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:8080/apps");
    expect(init.method).toBe("POST");
    const headers = init.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
    expect(headers.Authorization).toBe("Bearer test-token");
  });

  it("serializes the VM multi-model envelope (mode, pin, venv, version, sleep)", async () => {
    const fetchMock = stubFetch(async () =>
      jsonResponse({ app_id: "tm-fleet", url: "/run/tm-fleet" }),
    );
    await gateway.createApp(BATTLE_REQUEST);

    const body = sentBody(fetchMock);
    expect(body.mode).toBe("multi");
    expect(body.gpu_count).toBe(4);
    expect(body.visible_devices).toBe("0,1,2,3");
    expect(body.venv_path).toBe("/share/vllm-venv");
    expect(body.vllm_version).toBe("0.19.1");
    expect(body.sleep_level).toBe(1);
    expect(body.provider_id).toBe("prov-5be27d21");
    expect(body.models).toHaveLength(4);
  });

  it("preserves each member's tp + per-model vLLM args verbatim", async () => {
    const fetchMock = stubFetch(async () =>
      jsonResponse({ app_id: "tm-fleet", url: "/run/tm-fleet" }),
    );
    await gateway.createApp(BATTLE_REQUEST);

    const body = sentBody(fetchMock);
    const byName = Object.fromEntries((body.models ?? []).map((m) => [m.model, m]));

    expect(byName["qwen/qwen3.6-27b"].tp).toBe(2);
    expect(byName["qwen/qwen3.6-27b"].extra_args).toContain("--tool-call-parser qwen3_coder");
    expect(byName["qwen/qwen3.6-27b"].extra_args).toContain("--mm-encoder-tp-mode data");
    expect(byName["qwen/qwen3.6-27b"].extra_args).toContain("--mm-processor-cache-type shm");

    expect(byName["Qwen/Qwen3.6-35B-A3B"].tp).toBe(2);

    expect(byName["mistralai/Mistral-Small-4-119B-2603"].tp).toBe(4);
    expect(byName["mistralai/Mistral-Small-4-119B-2603"].extra_args).toContain("--reasoning-parser mistral");
    expect(byName["mistralai/Mistral-Small-4-119B-2603"].extra_args).toContain("--tool-call-parser mistral");

    expect(byName["google/gemma-4-31b-it"].tp).toBe(2);
    expect(byName["google/gemma-4-31b-it"].extra_args).toContain("--tool-call-parser gemma4");
    expect(byName["google/gemma-4-31b-it"].extra_args).toContain("--reasoning-parser gemma4");
  });

  it("sends all 10 cache/HOME env vars unchanged", async () => {
    const fetchMock = stubFetch(async () =>
      jsonResponse({ app_id: "tm-fleet", url: "/run/tm-fleet" }),
    );
    await gateway.createApp(BATTLE_REQUEST);

    const body = sentBody(fetchMock);
    expect(body.env_vars).toEqual(BATTLE_ENV);
    expect(Object.keys(body.env_vars ?? {})).toHaveLength(10);
    expect(body.env_vars?.HOME).toBe("/share/home");
    expect(body.env_vars?.HF_HOME).toBe("/share/huggingface");
    expect(body.env_vars?.TRANSFORMERS_CACHE).toBe("/share/huggingface");
    expect(body.env_vars?.CUDA_CACHE_PATH).toBe("/share/nv_cache");
  });

  it("round-trips the request unchanged (no field dropped or mutated)", async () => {
    const fetchMock = stubFetch(async () =>
      jsonResponse({ app_id: "tm-fleet", url: "/run/tm-fleet" }),
    );
    await gateway.createApp(BATTLE_REQUEST);
    expect(sentBody(fetchMock)).toEqual(BATTLE_REQUEST);
  });

  it("omits the Authorization header when there is no session cookie", async () => {
    mockToken = undefined;
    const fetchMock = stubFetch(async () =>
      jsonResponse({ app_id: "tm-fleet", url: "/run/tm-fleet" }),
    );
    await gateway.createApp(BATTLE_REQUEST);
    const headers = (fetchMock.mock.calls[0][1] as RequestInit).headers as Record<string, string>;
    expect(headers.Authorization).toBeUndefined();
  });

  it("throws GatewayError carrying status + parsed body on a 5xx", async () => {
    stubFetch(async () =>
      jsonResponse({ detail: { error: "PROVIDER_SECRET_KEY not set" } }, 500),
    );
    await expect(gateway.createApp(BATTLE_REQUEST)).rejects.toBeInstanceOf(GatewayError);

    stubFetch(async () =>
      jsonResponse({ detail: { error: "PROVIDER_SECRET_KEY not set" } }, 500),
    );
    try {
      await gateway.createApp(BATTLE_REQUEST);
      throw new Error("expected createApp to reject");
    } catch (e) {
      expect(e).toBeInstanceOf(GatewayError);
      expect((e as InstanceType<typeof GatewayError>).status).toBe(500);
      expect((e as InstanceType<typeof GatewayError>).parsed).toMatchObject({
        detail: { error: "PROVIDER_SECRET_KEY not set" },
      });
    }
  });
});
