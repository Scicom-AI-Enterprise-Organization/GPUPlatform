"use client";

// Thin wrapper over the shared <StressTest> — same vLLM-bench-style load as the
// serverless tab, targeting the proxy's OpenAI data plane (also exercises its
// concurrency queue). Server mode drives the gateway directly for true concurrency.
import { StressTest } from "@/components/playground/stress-test";

export function ProxyStress({ name, aliases }: { name: string; aliases: string[] }) {
  const chatPath = `proxy/${name}/v1/chat/completions`;
  return (
    <StressTest
      models={aliases}
      browserUrl={`/api/proxy/${chatPath}`}
      serverPayload={{ path: chatPath }}
      openai
    />
  );
}
