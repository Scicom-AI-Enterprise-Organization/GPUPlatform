// Unit test for deploying a RunPod (cloud) benchmark via the JS API. Asserts
// that gateway.createBenchmark(...) turns a /benchmark/new "Pod" config into the
// exact POST the gateway expects: the `runpod:` block (GPU type/count, cloud
// tier, container image, disk + volume) + the benchmaq sweep. Network + the
// sgpu_token cookie → Bearer auth are mocked. Sibling of benchmark-ssh.test.ts.
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

const { gateway, GatewayError } = await import("@/lib/gateway");

// ---- fixture: a RunPod cloud benchmark (what the Pod card produces) ----

type ParsedRunpodConfig = {
  runpod: {
    pod: { gpu_type: string; gpu_count: number; secure_cloud: boolean; instance_type: string };
    container: { image: string; disk_size: number };
    storage: { volume_size: number; mount_path: string };
  };
  benchmark: Array<{
    engine: string;
    model: { repo_id: string };
    serve: { tensor_parallel_size: number };
    bench: unknown[];
  }>;
};

const CU128 = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404";

function runpodBenchmarkRequest(opts: {
  gpuType?: string;
  gpuCount?: number;
  secure?: boolean;
  image?: string;
  diskGb?: number;
  volumeGb?: number;
  providerId?: string | null;
  storageId: string;
}): CreateBenchmarkRequest {
  const gpuCount = opts.gpuCount ?? 1;
  const cfg = {
    runpod: {
      ssh_private_key: "",
      runpod_api_key: "",
      pod: {
        name: "sgpu-rp",
        gpu_type: opts.gpuType ?? "NVIDIA H100 80GB HBM3",
        gpu_count: gpuCount,
        instance_type: "on_demand",
        secure_cloud: opts.secure ?? true,
      },
      container: { image: opts.image ?? CU128, disk_size: opts.diskGb ?? 80 },
      storage: { volume_size: opts.volumeGb ?? 80, mount_path: "/workspace" },
      ports: { http: [8000], tcp: [22] },
      env: { HF_HOME: "/workspace/hf_home" },
    },
    remote: {
      key_filename: "",
      uv: { path: "~/.venv", python_version: "3.11" },
      dependencies: ["vllm==0.19.1", "huggingface_hub", "hf_transfer"],
    },
    benchmark: [
      {
        name: "rp-quick",
        engine: "vllm",
        model: { repo_id: "Qwen/Qwen2.5-0.5B-Instruct", local_dir: "/workspace/models/qwen2.5-0.5b" },
        serve: { tensor_parallel_size: gpuCount, no_enable_prefix_caching: true },
        bench: [
          {
            endpoint: "/v1/completions",
            dataset_name: "random",
            random_input_len: 128,
            random_output_len: 32,
            num_prompts: 10,
            max_concurrency: 4,
          },
        ],
        results: { save_result: true },
      },
    ],
  };
  return {
    name: "rp-bench",
    config_yaml: yaml.dump(cfg, { sortKeys: false }),
    provider_id: opts.providerId ?? null,
    storage_id: opts.storageId,
  };
}

// ---- fetch mock plumbing (matches benchmark-ssh.test.ts) ----

function jsonResponse(obj: unknown, status = 200): Response {
  return new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });
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

// ---- the RunPod request is built correctly ----

describe("runpodBenchmarkRequest fixture", () => {
  it("is a SECURE H100×1 RunPod pod on the CUDA 12.8 image", () => {
    const req = runpodBenchmarkRequest({ storageId: "store-1" });
    expect(req.provider_id).toBeNull(); // null = gateway-default RunPod key
    expect(req.storage_id).toBe("store-1");
    const cfg = yaml.load(req.config_yaml) as ParsedRunpodConfig;
    expect(cfg.runpod.pod.gpu_type).toBe("NVIDIA H100 80GB HBM3");
    expect(cfg.runpod.pod.gpu_count).toBe(1);
    expect(cfg.runpod.pod.secure_cloud).toBe(true);
    expect(cfg.runpod.container.image).toBe(CU128); // 12.8 — matches current hosts
    expect(cfg.runpod.container.disk_size).toBe(80);
    expect(cfg.runpod.storage.volume_size).toBe(80);
    expect(cfg.benchmark[0].engine).toBe("vllm");
    expect(cfg.benchmark[0].bench).toHaveLength(1);
  });

  it("tensor-parallel size tracks the GPU count", () => {
    const cfg = yaml.load(
      runpodBenchmarkRequest({ gpuCount: 2, storageId: "s" }).config_yaml,
    ) as ParsedRunpodConfig;
    expect(cfg.runpod.pod.gpu_count).toBe(2);
    expect(cfg.benchmark[0].serve.tensor_parallel_size).toBe(2);
  });
});

// ---- createBenchmark deploys it via the API ----

describe("gateway.createBenchmark — RunPod (cloud) deploy", () => {
  it("POSTs /benchmarks with bearer auth and the RunPod config intact", async () => {
    const fetchMock = stubFetch(async () => jsonResponse({ id: "bench-rp1", status: "queued" }));
    const res = await gateway.createBenchmark(
      runpodBenchmarkRequest({ providerId: "prov-runpod1", storageId: "store-1" }),
    );
    expect(res).toMatchObject({ id: "bench-rp1", status: "queued" });

    const { url, init } = call(fetchMock);
    expect(url).toBe("http://localhost:8080/benchmarks");
    expect(init.method).toBe("POST");
    const headers = init.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
    expect(headers.Authorization).toBe("Bearer test-token");

    const body = sentBody(fetchMock);
    expect(body.provider_id).toBe("prov-runpod1"); // the chosen RunPod account
    const cfg = yaml.load(body.config_yaml) as ParsedRunpodConfig;
    expect(cfg.runpod.pod.secure_cloud).toBe(true);
    expect(cfg.runpod.container.image).toBe(CU128);
    expect(cfg.runpod.storage.volume_size).toBe(80);
  });

  it("serializes a COMMUNITY tier + custom GPU/disk/volume selection", async () => {
    const fetchMock = stubFetch(async () => jsonResponse({ id: "bench-rp2", status: "queued" }));
    await gateway.createBenchmark(
      runpodBenchmarkRequest({
        gpuType: "NVIDIA GeForce RTX 4090",
        gpuCount: 1,
        secure: false,
        diskGb: 50,
        volumeGb: 0,
        storageId: "store-1",
      }),
    );
    const cfg = yaml.load(sentBody(fetchMock).config_yaml) as ParsedRunpodConfig;
    expect(cfg.runpod.pod.gpu_type).toBe("NVIDIA GeForce RTX 4090");
    expect(cfg.runpod.pod.secure_cloud).toBe(false);
    expect(cfg.runpod.container.disk_size).toBe(50);
    expect(cfg.runpod.storage.volume_size).toBe(0);
  });

  it("propagates a gateway error (e.g. bad config) as GatewayError", async () => {
    stubFetch(async () => jsonResponse({ detail: "no instances currently available" }, 500));
    const err = await gateway
      .createBenchmark(runpodBenchmarkRequest({ storageId: "store-1" }))
      .catch((e) => e);
    expect(err).toBeInstanceOf(GatewayError);
    expect((err as InstanceType<typeof GatewayError>).status).toBe(500);
  });
});
