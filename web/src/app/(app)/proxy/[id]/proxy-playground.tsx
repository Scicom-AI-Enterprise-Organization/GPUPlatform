"use client";

// Thin wrapper over the shared <ChatPlayground> — the proxy is an OpenAI-compatible
// endpoint, so it just plugs in the OpenAI transport pointed at its data plane.
import { ChatPlayground, openAiTransport } from "@/components/playground/chat-playground";

export function ProxyPlayground({ name, aliases, baseUrl }: { name: string; aliases: string[]; baseUrl: string }) {
  return (
    <ChatPlayground
      models={aliases}
      storageKey={`serverless-ui:proxy-playground:${name}`}
      description={<>Routes through <code className="font-mono">POST /proxy/{name}/v1/chat/completions</code> to a live backend (priority + failover). Hit Stop mid-stream to trigger the proxy&apos;s auto-cancel.</>}
      transport={openAiTransport({
        fetchPath: `/api/proxy/proxy/${encodeURIComponent(name)}/v1/chat/completions`,
        curlUrl: `${baseUrl}/proxy/${name}/v1/chat/completions`,
      })}
    />
  );
}
