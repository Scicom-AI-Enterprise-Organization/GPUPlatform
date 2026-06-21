"use client";

// Proxy playground with a mode DROPDOWN (chat / embeddings / audio transcription),
// exactly like the serverless endpoint playground. The proxy is OpenAI-compatible,
// so each mode just points the shared playground component at the matching
// data-plane path; embeddings + audio reuse the same generic components the
// serverless playground uses.
import { useState } from "react";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { ChatPlayground, openAiTransport } from "@/components/playground/chat-playground";
import { EmbeddingPlayground } from "@/app/(app)/serverless/[id]/tabs/embedding";
import { TranscribePlayground } from "@/app/(app)/serverless/[id]/tabs/transcribe";

export function ProxyPlayground({ name, aliases, baseUrl }: { name: string; aliases: string[]; baseUrl: string }) {
  // The data-plane base behind the Next proxy: /api/proxy → gateway, then
  // /proxy/{name}/v1 → the proxy router. Each mode appends its OpenAI sub-path.
  const apiBase = `/api/proxy/proxy/${encodeURIComponent(name)}/v1`;
  const [mode, setMode] = useState<"chat" | "embedding" | "audio">("chat");

  const chat = (
    <ChatPlayground
      models={aliases}
      storageKey={`serverless-ui:proxy-playground:${name}`}
      description={<>Routes through <code className="font-mono">POST /proxy/{name}/v1/chat/completions</code> to a live backend (priority + failover). Hit Stop mid-stream to trigger the proxy&apos;s auto-cancel.</>}
      transport={openAiTransport({
        fetchPath: `${apiBase}/chat/completions`,
        curlUrl: `${baseUrl}/proxy/${name}/v1/chat/completions`,
      })}
    />
  );

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">mode</span>
        <Select value={mode} onValueChange={(v) => setMode(v as "chat" | "embedding" | "audio")}>
          <SelectTrigger className="h-8 w-[280px] text-xs"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="chat" className="text-xs">Chat / text generation</SelectItem>
            <SelectItem value="embedding" className="text-xs">Embeddings (/v1/embeddings)</SelectItem>
            <SelectItem value="audio" className="text-xs">Audio transcription (Whisper)</SelectItem>
          </SelectContent>
        </Select>
        <span className="hidden text-[11px] text-muted-foreground sm:inline">
          routes by the <code className="font-mono">model</code> you pick (priority + failover, like chat)
        </span>
      </div>
      {mode === "chat" ? chat
        : mode === "embedding"
          ? <EmbeddingPlayground models={aliases} basePath={apiBase} storageKey={`serverless-ui:embed:proxy:${name}`} />
          : <TranscribePlayground models={aliases} basePath={apiBase} storageKey={`serverless-ui:transcribe:proxy:${name}`} />}
    </div>
  );
}
