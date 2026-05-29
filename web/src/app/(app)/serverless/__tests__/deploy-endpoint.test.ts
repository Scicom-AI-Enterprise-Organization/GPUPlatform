import { afterEach, describe, expect, it, vi } from "vitest";
import type { CreateAppRequest } from "@/lib/types";

// The action pulls auth from a cookie (next/headers) and revalidates the route
// cache (next/cache) — both stubbed so we can run it outside a Next request.
let mockToken: string | undefined = "test-token";
vi.mock("next/headers", () => ({
  cookies: async () => ({
    get: (name: string) =>
      name === "sgpu_token" && mockToken ? { value: mockToken } : undefined,
  }),
}));
const revalidatePath = vi.fn();
vi.mock("next/cache", () => ({ revalidatePath: (p: string) => revalidatePath(p) }));

const { deployEndpoint } = await import("@/app/(app)/serverless/actions");

const REQUEST: CreateAppRequest = {
  name: "tm-fleet",
  gpu: "VM",
  gpu_count: 4,
  mode: "multi",
  provider_id: "prov-5be27d21",
  visible_devices: "0,1,2,3",
  venv_path: "/share/vllm-venv",
  vllm_version: "0.19.1",
  sleep_level: 1,
  models: [
    { model: "qwen/qwen3.6-27b", tp: 2, extra_args: "--reasoning-parser qwen3" },
    { model: "mistralai/Mistral-Small-4-119B-2603", tp: 4, extra_args: "--reasoning-parser mistral" },
  ],
};

function jsonResponse(obj: unknown, status = 200): Response {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
type FetchArgs = [input?: string, init?: RequestInit];
function stubFetch(impl: (...args: FetchArgs) => Promise<Response>) {
  const m = vi.fn(impl);
  vi.stubGlobal("fetch", m);
  return m;
}

afterEach(() => {
  vi.unstubAllGlobals();
  revalidatePath.mockClear();
});

describe("deployEndpoint — create multi-model inference action", () => {
  it("returns { ok, app_id } and revalidates on success", async () => {
    const fetchMock = stubFetch(async () =>
      jsonResponse({ app_id: "tm-fleet", url: "/run/tm-fleet" }),
    );

    const res = await deployEndpoint(REQUEST);

    expect(res).toEqual({ ok: true, app_id: "tm-fleet" });
    expect(revalidatePath).toHaveBeenCalledWith("/serverless");
    // It forwarded the multi-model body to POST /apps.
    const body = JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string);
    expect(body.mode).toBe("multi");
    expect(body.models).toHaveLength(2);
  });

  it("shapes a 503 GPU-unavailable error into { unavailable }", async () => {
    stubFetch(async () =>
      jsonResponse(
        { detail: { error: "GPU not available", gpu: "H100", gpu_count: 4, reason: "out of stock in region" } },
        503,
      ),
    );

    const res = await deployEndpoint(REQUEST);
    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.error).toBe("GPU not available");
      expect(res.unavailable).toEqual({ gpu: "H100", gpu_count: 4, reason: "out of stock in region" });
    }
    expect(revalidatePath).not.toHaveBeenCalled();
  });

  it("returns a plain error (no `unavailable`) for other gateway failures", async () => {
    stubFetch(async () =>
      jsonResponse({ detail: { error: "PROVIDER_SECRET_KEY not set" } }, 500),
    );

    const res = await deployEndpoint(REQUEST);
    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.unavailable).toBeUndefined();
      expect(res.error).toContain("500");
    }
  });

  it("does not mistake a 503 *without* gpu detail for an unavailability error", async () => {
    // e.g. the multi-model "fleet still warming up" 503 — generic, not GPU stock.
    stubFetch(async () =>
      jsonResponse({ detail: { error: "the model fleet is still warming up", state: "warming_up" } }, 503),
    );

    const res = await deployEndpoint(REQUEST);
    expect(res.ok).toBe(false);
    if (!res.ok) expect(res.unavailable).toBeUndefined();
  });
});
