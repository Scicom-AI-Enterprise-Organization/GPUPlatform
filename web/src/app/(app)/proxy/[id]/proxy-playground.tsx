"use client";

// Proxy playground with a mode DROPDOWN (chat / embeddings / audio transcription),
// exactly like the serverless endpoint playground. The proxy is OpenAI-compatible,
// so each mode just points the shared playground component at the matching
// data-plane path; embeddings + audio reuse the same generic components the
// serverless playground uses.
import { useMemo, useState } from "react";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { ChatPlayground, openAiTransport } from "@/components/playground/chat-playground";
import { EmbeddingPlayground } from "@/app/(app)/serverless/[id]/tabs/embedding";
import { TranscribePlayground } from "@/app/(app)/serverless/[id]/tabs/transcribe";
import { SpeechPlayground } from "@/app/(app)/serverless/[id]/tabs/speech";

// The upstream identity the force dropdown offers. Only enabled upstreams are
// forceable (the backend 404s a forced disabled/absent upstream).
export type PlaygroundUpstream = { id: string; name: string; enabled: boolean };

const AUTO = "__auto";

export function ProxyPlayground(
  { name, aliases, baseUrl, upstreams = [] }:
  { name: string; aliases: string[]; baseUrl: string; upstreams?: PlaygroundUpstream[] },
) {
  // The data-plane base behind the Next proxy: /api/proxy → gateway, then
  // /proxy/{name}/v1 → the proxy router. Each mode appends its OpenAI sub-path.
  const apiBase = `/api/proxy/proxy/${encodeURIComponent(name)}/v1`;
  const [mode, setMode] = useState<"chat" | "embedding" | "audio" | "tts">("chat");

  // Force-provider: send X-SGPU-Upstream to pin routing to ONE upstream (no
  // failover). "" / auto → normal priority+health routing across all upstreams.
  const forceable = upstreams.filter((u) => u.enabled);
  const [forced, setForced] = useState<string>(AUTO);
  // Memoized so the header object identity is stable per selection — the speech
  // playground keys a fetch effect on it.
  const extraHeaders = useMemo<Record<string, string> | undefined>(
    () => (forced && forced !== AUTO ? { "X-SGPU-Upstream": forced } : undefined),
    [forced],
  );

  const chat = (
    <ChatPlayground
      models={aliases}
      storageKey={`serverless-ui:proxy-playground:${name}`}
      description={<>Routes through <code className="font-mono">POST /proxy/{name}/v1/chat/completions</code> to a live backend (priority + failover). Hit Stop mid-stream to trigger the proxy&apos;s auto-cancel.</>}
      transport={openAiTransport({
        fetchPath: `${apiBase}/chat/completions`,
        curlUrl: `${baseUrl}/proxy/${name}/v1/chat/completions`,
        extraHeaders,
      })}
    />
  );

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted-foreground">mode</span>
        <Select value={mode} onValueChange={(v) => setMode(v as "chat" | "embedding" | "audio" | "tts")}>
          <SelectTrigger className="h-8 w-[280px] text-xs"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="chat" className="text-xs">Chat / text generation</SelectItem>
            <SelectItem value="embedding" className="text-xs">Embeddings (/v1/embeddings)</SelectItem>
            <SelectItem value="audio" className="text-xs">Audio transcription (Whisper)</SelectItem>
            <SelectItem value="tts" className="text-xs">Text to speech (/v1/audio/speech)</SelectItem>
          </SelectContent>
        </Select>
        {forceable.length > 1 && (
          <>
            <span className="text-xs text-muted-foreground">provider</span>
            <Select value={forced} onValueChange={setForced}>
              <SelectTrigger className="h-8 w-[220px] text-xs" title="Force which upstream serves this request (X-SGPU-Upstream)"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value={AUTO} className="text-xs">Auto (priority + failover)</SelectItem>
                {forceable.map((u) => (
                  <SelectItem key={u.id} value={u.name} className="text-xs">Force: {u.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </>
        )}
        <span className="hidden text-[11px] text-muted-foreground sm:inline">
          {extraHeaders
            ? <>pinned to <code className="font-mono">{forced}</code> via <code className="font-mono">X-SGPU-Upstream</code> (no failover)</>
            : <>routes by the <code className="font-mono">model</code> you pick (priority + failover)</>}
        </span>
      </div>
      {mode === "chat" ? chat
        : mode === "embedding"
          ? <EmbeddingPlayground models={aliases} basePath={apiBase} storageKey={`serverless-ui:embed:proxy:${name}`} extraHeaders={extraHeaders} />
          : mode === "audio"
            ? <TranscribePlayground models={aliases} basePath={apiBase} storageKey={`serverless-ui:transcribe:proxy:${name}`} extraHeaders={extraHeaders} />
            : <SpeechPlayground models={aliases} basePath={apiBase} storageKey={`serverless-ui:speech:proxy:${name}`} extraHeaders={extraHeaders} />}
    </div>
  );
}
