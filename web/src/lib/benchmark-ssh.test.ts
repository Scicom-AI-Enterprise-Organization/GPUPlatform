// Battle-test the JS API layer for SSH/VM-based benchmarks: that the gateway
// client turns a bench-db16ea55-style config (pinned to GPUs 6,7) into the
// exact POST the gateway expects, and that the surrounding benchmark API
// methods (get/list/rename/terminate/delete/files) hit the right routes.
//
// The network is mocked — deterministic unit tests, no live gateway. next/headers
// is stubbed so the client's server-side cookie→Bearer path runs without a Next
// request context. Mirrors src/lib/__tests__/create-inference.test.ts.
import { afterEach, describe, expect, it, vi } from "vitest";
import yaml from "js-yaml";
import type { CreateBenchmarkRequest } from "@/lib/types";

let mockToken: string | undefined = "test-token";
vi.mock("next/headers", () => ({
  cookies: async () => ({
    get: (name: string) =>
      name === "sgpu_token" && mockToken ? { value: mockToken } : undefined,
  }),
}));

// Import AFTER the mock is registered.
const { gateway, GatewayError } = await import("@/lib/gateway");

// ---- fixture: replicate bench-db16ea55 (vLLM sweep) for the SSH/VM path ----

const SHARE_ENV: Record<string, string> = {
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

const INPUT_LENS = [128, 512, 1024, 2048];
const CONCURRENCIES = [10, 25, 50, 200];

type BenchCell = {
  endpoint: string;
  random_input_len: number;
  random_output_len: number;
  num_prompts: number;
  max_concurrency: number;
};
type ParsedBenchConfig = {
  remote: { uv: { python_version: string; path: string }; dependencies: string[] };
  benchmark: Array<{
    name: string;
    engine: string;
    model: { repo_id: string; local_dir: string };
    serve: { tensor_parallel_size: number; no_enable_prefix_caching: boolean; port: number };
    bench: BenchCell[];
    results: { save_result: boolean; save_detailed: boolean };
  }>;
};

/** The 16-cell sweep bench-db16ea55 runs (in × concurrency, out=128, 50 prompts). */
function sweepCells() {
  return INPUT_LENS.flatMap((inp) =>
    CONCURRENCIES.map((c) => ({
      endpoint: "/v1/completions",
      dataset_name: "random",
      random_input_len: inp,
      random_output_len: 128,
      num_prompts: 50,
      max_concurrency: c,
      request_rate: "inf",
      ignore_eos: true,
      percentile_metrics: "ttft,tpot,itl,e2el",
    })),
  );
}

/** CreateBenchmarkRequest for an SSH/VM run mirroring bench-db16ea55, pinned to
 * the given GPUs (default 6,7 → TP=2). */
function sshBenchmarkRequest(opts: {
  providerId: string;
  storageId: string;
  gpus?: string;
  model?: string;
  port?: number;
}): CreateBenchmarkRequest {
  const gpus = opts.gpus ?? "6,7";
  const tp = gpus.split(",").filter(Boolean).length; // TP = #GPUs pinned
  const model = opts.model ?? "Qwen/Qwen3.6-35B-A3B";
  const cfg = {
    remote: {
      uv: { path: "~/.benchmark-venv", python_version: "3.11" },
      dependencies: ["vllm==0.19.1", "huggingface_hub", "hf_transfer"],
    },
    benchmark: [
      {
        name: "qwen-quick",
        engine: "vllm",
        model: { repo_id: model, local_dir: `~/models/${model.split("/").pop()!.toLowerCase()}` },
        serve: {
          tensor_parallel_size: tp,
          no_enable_prefix_caching: true,
          port: opts.port ?? 18017,
        },
        bench: sweepCells(),
        results: { save_result: true, save_detailed: true },
      },
    ],
  };
  return {
    name: `db16ea55-ssh-gpu${gpus.replace(/,/g, "")}`,
    config_yaml: yaml.dump(cfg, { sortKeys: false }),
    provider_id: opts.providerId,
    storage_id: opts.storageId,
    visible_devices: gpus,
    env_vars: SHARE_ENV,
    cleanup_model: false,
  };
}

// ---- fetch mock plumbing (matches create-inference.test.ts) ----

function jsonResponse(obj: unknown, status = 200): Response {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function stubFetch(impl: () => Promise<Response>) {
  const fetchMock = vi.fn(impl);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function call(fetchMock: ReturnType<typeof stubFetch>): { url: string; init: RequestInit } {
  expect(fetchMock).toHaveBeenCalledTimes(1);
  const [input, init] = fetchMock.mock.calls[0] as unknown as [RequestInfo | URL, RequestInit?];
  return { url: String(input), init: init ?? {} };
}

function sentBody(fetchMock: ReturnType<typeof stubFetch>): CreateBenchmarkRequest {
  return JSON.parse(String(call(fetchMock).init.body)) as CreateBenchmarkRequest;
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
  mockToken = "test-token";
});

// ---- the fixture itself is a valid, faithful replica ----

describe("sshBenchmarkRequest fixture (bench-db16ea55 replica)", () => {
  it("produces a 16-cell vLLM sweep pinned to GPUs 6,7 (TP=2)", () => {
    const req = sshBenchmarkRequest({ providerId: "prov-x", storageId: "store-y" });
    expect(req.visible_devices).toBe("6,7");
    expect(req.cleanup_model).toBe(false);
    expect(req.env_vars).toMatchObject({ HF_HOME: "/share/huggingface" });
    expect(Object.keys(req.env_vars ?? {})).toHaveLength(9);

    const cfg = yaml.load(req.config_yaml) as ParsedBenchConfig;
    expect(cfg.remote.uv.python_version).toBe("3.11");
    expect(cfg.remote.dependencies).toContain("vllm==0.19.1");
    const b = cfg.benchmark[0];
    expect(b.engine).toBe("vllm");
    expect(b.model.repo_id).toBe("Qwen/Qwen3.6-35B-A3B");
    expect(b.serve.tensor_parallel_size).toBe(2);
    expect(b.serve.no_enable_prefix_caching).toBe(true);
    expect(b.bench).toHaveLength(16);
    expect(new Set(b.bench.map((x) => x.random_input_len))).toEqual(new Set(INPUT_LENS));
    expect(new Set(b.bench.map((x) => x.max_concurrency))).toEqual(new Set(CONCURRENCIES));
    expect(b.bench.every((x) => x.random_output_len === 128 && x.num_prompts === 50)).toBe(true);
    expect(b.results.save_result).toBe(true);
  });

  it("derives TP from the GPU pin (4 GPUs → TP=4)", () => {
    const req = sshBenchmarkRequest({ providerId: "p", storageId: "s", gpus: "0,1,2,3" });
    expect(req.visible_devices).toBe("0,1,2,3");
    const cfg = yaml.load(req.config_yaml) as ParsedBenchConfig;
    expect(cfg.benchmark[0].serve.tensor_parallel_size).toBe(4);
  });
});

// ---- createBenchmark sends the exact request the gateway expects ----

describe("gateway.createBenchmark — SSH/VM run on GPUs 6,7", () => {
  it("POSTs /benchmarks with method, JSON content-type and bearer auth", async () => {
    const fetchMock = stubFetch(async () => jsonResponse({ id: "bench-abc123", status: "queued" }));
    const created = await gateway.createBenchmark(
      sshBenchmarkRequest({ providerId: "prov-5be27d21", storageId: "store-23b84331" }),
    );
    expect(created).toMatchObject({ id: "bench-abc123", status: "queued" });

    const { url, init } = call(fetchMock);
    expect(url).toBe("http://localhost:8080/benchmarks");
    expect(init.method).toBe("POST");
    const headers = init.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
    expect(headers.Authorization).toBe("Bearer test-token");
  });

  it("serializes the vm provider, storage, GPU pin + cache env", async () => {
    const fetchMock = stubFetch(async () => jsonResponse({ id: "bench-abc123", status: "queued" }));
    await gateway.createBenchmark(
      sshBenchmarkRequest({ providerId: "prov-5be27d21", storageId: "store-23b84331" }),
    );
    const body = sentBody(fetchMock);
    expect(body.provider_id).toBe("prov-5be27d21"); // routes to the VM provider
    expect(body.storage_id).toBe("store-23b84331");
    expect(body.visible_devices).toBe("6,7"); // CUDA_VISIBLE_DEVICES on the VM
    expect(body.cleanup_model).toBe(false); // keep the cached model
    expect(body.env_vars).toEqual(SHARE_ENV);
  });

  it("round-trips the sweep config intact (16 cells, port, TP)", async () => {
    const fetchMock = stubFetch(async () => jsonResponse({ id: "bench-abc123", status: "queued" }));
    await gateway.createBenchmark(
      sshBenchmarkRequest({ providerId: "prov-5be27d21", storageId: "store-23b84331" }),
    );
    const cfg = yaml.load(sentBody(fetchMock).config_yaml) as ParsedBenchConfig;
    expect(cfg.benchmark[0].bench).toHaveLength(16);
    expect(cfg.benchmark[0].serve.port).toBe(18017);
    expect(cfg.benchmark[0].serve.tensor_parallel_size).toBe(2);
  });

  it("omits Authorization when there is no session cookie", async () => {
    mockToken = undefined;
    const fetchMock = stubFetch(async () => jsonResponse({ id: "bench-abc123", status: "queued" }));
    await gateway.createBenchmark(sshBenchmarkRequest({ providerId: "p", storageId: "s" }));
    const headers = call(fetchMock).init.headers as Record<string, string>;
    expect(headers.Authorization).toBeUndefined();
  });

  it("throws GatewayError carrying status + parsed body on a 4xx", async () => {
    stubFetch(async () => jsonResponse({ error: "bad config" }, 400));
    await expect(
      gateway.createBenchmark(sshBenchmarkRequest({ providerId: "p", storageId: "s" })),
    ).rejects.toBeInstanceOf(GatewayError);

    vi.unstubAllGlobals();
    stubFetch(async () => jsonResponse({ error: "bad config" }, 400));
    const err = await gateway
      .createBenchmark(sshBenchmarkRequest({ providerId: "p", storageId: "s" }))
      .catch((e) => e);
    expect(err).toBeInstanceOf(GatewayError);
    expect((err as InstanceType<typeof GatewayError>).status).toBe(400);
    expect((err as InstanceType<typeof GatewayError>).parsed).toMatchObject({ error: "bad config" });
  });
});

// ---- the rest of the benchmark API surface used to drive a run ----

describe("benchmark API routes", () => {
  it("getBenchmark → GET /benchmarks/{id}", async () => {
    const fetchMock = stubFetch(async () => jsonResponse({ id: "bench-1", status: "done" }));
    await gateway.getBenchmark("bench-1");
    const { url, init } = call(fetchMock);
    expect(url).toBe("http://localhost:8080/benchmarks/bench-1");
    expect(init.method ?? "GET").toBe("GET");
  });

  it("listBenchmarks → GET /benchmarks?scope=all", async () => {
    const fetchMock = stubFetch(async () => jsonResponse([]));
    await gateway.listBenchmarks("all");
    expect(call(fetchMock).url).toBe("http://localhost:8080/benchmarks?scope=all");
  });

  it("renameBenchmark → PATCH /benchmarks/{id} {name}", async () => {
    const fetchMock = stubFetch(async () => jsonResponse({ id: "bench-1", name: "renamed" }));
    await gateway.renameBenchmark("bench-1", "renamed");
    const { url, init } = call(fetchMock);
    expect(url).toBe("http://localhost:8080/benchmarks/bench-1");
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(String(init.body))).toEqual({ name: "renamed" });
  });

  it("terminateBenchmark → POST /benchmarks/{id}/terminate", async () => {
    const fetchMock = stubFetch(async () =>
      jsonResponse({ ok: true, id: "bench-1", status: "cancelled" }),
    );
    await gateway.terminateBenchmark("bench-1");
    const { url, init } = call(fetchMock);
    expect(url).toBe("http://localhost:8080/benchmarks/bench-1/terminate");
    expect(init.method).toBe("POST");
  });

  it("deleteBenchmark → DELETE /benchmarks/{id}", async () => {
    const fetchMock = stubFetch(async () => jsonResponse({ ok: true, id: "bench-1" }));
    await gateway.deleteBenchmark("bench-1");
    const { url, init } = call(fetchMock);
    expect(url).toBe("http://localhost:8080/benchmarks/bench-1");
    expect(init.method).toBe("DELETE");
  });

  it("listBenchmarkFiles → GET /benchmarks/{id}/files", async () => {
    const fetchMock = stubFetch(async () => jsonResponse([]));
    await gateway.listBenchmarkFiles("bench-1");
    expect(call(fetchMock).url).toBe("http://localhost:8080/benchmarks/bench-1/files");
  });
});

// ---- result.json fast-path URL (the Results/Compare views fetch this) ----

describe("benchmarkFileContentUrl", () => {
  it("builds the same-origin proxy URL for result.json", () => {
    expect(gateway.benchmarkFileContentUrl("bench-1", "result.json")).toBe(
      "/api/proxy/benchmarks/bench-1/files/content?path=result.json",
    );
  });

  it("encodes ids and nested paths", () => {
    expect(gateway.benchmarkFileContentUrl("bench/x y", "dir/sub file.json")).toBe(
      "/api/proxy/benchmarks/bench%2Fx%20y/files/content?path=dir%2Fsub%20file.json",
    );
  });
});
