import { afterEach, describe, expect, it, vi } from "vitest";
import type { ChatCompletionResponse } from "@/lib/types";

let mockToken: string | undefined = "test-token";
vi.mock("next/headers", () => ({
  cookies: async () => ({
    get: (name: string) =>
      name === "sgpu_token" && mockToken ? { value: mockToken } : undefined,
  }),
}));

const { gateway, GatewayError } = await import("@/lib/gateway");

// Every member of the deployed multi-model fleet — the `model` field the
// gateway routes (and wakes) by.
const FLEET = [
  "qwen/qwen3.6-27b",
  "Qwen/Qwen3.6-35B-A3B",
  "mistralai/Mistral-Small-4-119B-2603",
  "google/gemma-4-31b-it",
] as const;

type FetchArgs = [input?: string, init?: RequestInit];
function stubFetch(impl: (...args: FetchArgs) => Promise<Response>) {
  const m = vi.fn(impl);
  vi.stubGlobal("fetch", m);
  return m;
}
function jsonResponse(obj: unknown, status = 200): Response {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
function completion(model: string, content: string): ChatCompletionResponse {
  return {
    id: "chatcmpl-test",
    object: "chat.completion",
    model,
    choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
    usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  mockToken = "test-token";
});

describe("gateway.chatCompletion — request each fleet model", () => {
  it.each(FLEET)("routes a request to %s and parses the completion", async (model) => {
    const fetchMock = stubFetch(async () => jsonResponse(completion(model, "hello there")));

    const res = await gateway.chatCompletion({
      model,
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 16,
    });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8080/v1/chat/completions");
    expect((init as RequestInit).method).toBe("POST");
    const headers = (init as RequestInit).headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer test-token");

    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.model).toBe(model); // <-- routed to THIS member
    expect(body.messages).toEqual([{ role: "user", content: "hi" }]);
    expect(body.max_tokens).toBe(16);

    expect(res.model).toBe(model);
    expect(res.choices[0].message.content).toBe("hello there");
    expect(res.choices[0].finish_reason).toBe("stop");
  });

  it("forwards reasoning_effort + chat_template_kwargs (disable thinking) per request", async () => {
    const fetchMock = stubFetch(async () => jsonResponse(completion("qwen/qwen3.6-27b", "ok")));
    await gateway.chatCompletion({
      model: "qwen/qwen3.6-27b",
      messages: [{ role: "user", content: "hi" }],
      reasoning_effort: "low",
      chat_template_kwargs: { enable_thinking: false },
    });
    const body = JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string);
    expect(body.reasoning_effort).toBe("low");
    expect(body.chat_template_kwargs).toEqual({ enable_thinking: false });
  });

  it.each(FLEET)("surfaces a dead %s as a 503 GatewayError with the reason", async (model) => {
    stubFetch(async () =>
      jsonResponse(
        { detail: { error: `model '${model}' is not running (dead) — restart it from the Workers tab.`, state: "dead", model, reason: "Not enough free GPU memory" } },
        503,
      ),
    );

    try {
      await gateway.chatCompletion({ model, messages: [{ role: "user", content: "hi" }] });
      throw new Error("expected a GatewayError");
    } catch (e) {
      expect(e).toBeInstanceOf(GatewayError);
      expect((e as InstanceType<typeof GatewayError>).status).toBe(503);
      const parsed = (e as InstanceType<typeof GatewayError>).parsed as { detail?: { state?: string; model?: string } };
      expect(parsed.detail?.state).toBe("dead");
      expect(parsed.detail?.model).toBe(model);
    }
  });

  it("surfaces a warming-up fleet as a 503 (retryable)", async () => {
    stubFetch(async () =>
      jsonResponse({ detail: { error: "the model fleet is still warming up", state: "warming_up" } }, 503),
    );
    await expect(
      gateway.chatCompletion({ model: "Qwen/Qwen3.6-35B-A3B", messages: [{ role: "user", content: "hi" }] }),
    ).rejects.toMatchObject({ status: 503 });
  });

  it("times out as a 504 when the worker is cold-starting", async () => {
    stubFetch(async () =>
      jsonResponse({ error: "no completion in 60s — worker probably cold-starting", request_id: "req-abc" }, 504),
    );
    await expect(
      gateway.chatCompletion({ model: "mistralai/Mistral-Small-4-119B-2603", messages: [{ role: "user", content: "hi" }] }),
    ).rejects.toMatchObject({ status: 504 });
  });
});

describe("gateway.listModels — model discovery", () => {
  it("lists every fleet member id", async () => {
    stubFetch(async () =>
      jsonResponse({
        object: "list",
        data: FLEET.map((id) => ({ id, object: "model", created: 0, owned_by: "tm-fleet" })),
      }),
    );
    const res = await gateway.listModels();
    expect(res.object).toBe("list");
    expect(res.data.map((m) => m.id)).toEqual([...FLEET]);
  });
});
