"use client";

// Proxy playground with a mode DROPDOWN (chat / EOU / embeddings / rerank / audio
// transcription), exactly like the serverless endpoint playground. The proxy is
// OpenAI-compatible, so each mode just points the shared playground component at the
// matching data-plane path; embeddings + audio reuse the same generic components the
// serverless playground uses. Rerank is proxy-only — the serverless data plane routes
// through the worker queue, which has no rerank job type. EOU (turn detector) is the
// only mode on the RAW /v1/completions path: it's the one caller that needs
// allowed_token_ids + logprobs, which no other mode sends.
import { useMemo, useState } from "react";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { ChatPlayground, openAiTransport } from "@/components/playground/chat-playground";
import { RerankPlayground } from "@/components/playground/rerank-playground";
import { EouPlayground } from "@/components/playground/eou-playground";
import { EmbeddingPlayground } from "@/app/(app)/serverless/[id]/tabs/embedding";
import { TranscribePlayground } from "@/app/(app)/serverless/[id]/tabs/transcribe";
import { SpeechPlayground } from "@/app/(app)/serverless/[id]/tabs/speech";

// The upstream identity the force dropdown offers. Only enabled upstreams are
// forceable (the backend 404s a forced disabled/absent upstream).
export type PlaygroundUpstream = { id: string; name: string; enabled: boolean };

type PlaygroundMode = "chat" | "completions-eou" | "embedding" | "rerank" | "audio" | "tts";

// Short labels for the segmented selector; the full endpoint lives in the tooltip so the
// row stays readable at narrow widths.
const MODES: { id: PlaygroundMode; label: string; title: string }[] = [
  { id: "chat", label: "Chat", title: "Chat / text generation (/v1/chat/completions)" },
  { id: "completions-eou", label: "Turn detector", title: "End of utterance (/v1/completions)" },
  { id: "embedding", label: "Embeddings", title: "Embeddings (/v1/embeddings)" },
  { id: "rerank", label: "Rerank", title: "Rerank (/v1/rerank)" },
  { id: "audio", label: "Transcribe", title: "Audio transcription (Whisper)" },
  { id: "tts", label: "TTS", title: "Text to speech (/v1/audio/speech)" },
];

const AUTO = "__auto";

export function ProxyPlayground(
  { name, aliases, baseUrl, upstreams = [] }:
  { name: string; aliases: string[]; baseUrl: string; upstreams?: PlaygroundUpstream[] },
) {
  // The data-plane base behind the Next proxy: /api/proxy → gateway, then
  // /proxy/{name}/v1 → the proxy router. Each mode appends its OpenAI sub-path.
  const apiBase = `/api/proxy/proxy/${encodeURIComponent(name)}/v1`;
  // Same route as a caller outside the browser spells it — every mode renders its
  // copyable curl against this, never against the Next-proxy path above (which only
  // resolves from inside the console).
  const curlBase = `${baseUrl}/proxy/${name}/v1`;
  const [mode, setMode] = useState<PlaygroundMode>("chat");

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
        curlUrl: `${curlBase}/chat/completions`,
        extraHeaders,
      })}
    />
  );

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted-foreground">mode</span>
        {/* Segmented row rather than a dropdown: every mode stays visible, so which one
            you're in — and what else exists — reads at a glance and is one click away.
            Same control the upstream editor's Test row uses. */}
        <div className="inline-flex flex-wrap rounded-md border border-border p-0.5 text-xs">
          {MODES.map(({ id, label, title }) => (
            <button
              key={id}
              type="button"
              title={title}
              onClick={() => setMode(id)}
              className={"rounded px-2 py-1 " + (mode === id
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:text-foreground")}
            >
              {label}
            </button>
          ))}
        </div>
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
        : mode === "completions-eou" ? <EouPlayground models={aliases} basePath={apiBase} curlBase={curlBase} storageKey={`serverless-ui:eou:proxy:${name}`} extraHeaders={extraHeaders} />
        : mode === "embedding" ? <EmbeddingPlayground models={aliases} basePath={apiBase} curlBase={curlBase} storageKey={`serverless-ui:embed:proxy:${name}`} extraHeaders={extraHeaders} />
        : mode === "rerank" ? <RerankPlayground models={aliases} basePath={apiBase} curlBase={curlBase} storageKey={`serverless-ui:rerank:proxy:${name}`} extraHeaders={extraHeaders} />
        : mode === "audio" ? <TranscribePlayground models={aliases} basePath={apiBase} curlBase={curlBase} storageKey={`serverless-ui:transcribe:proxy:${name}`} extraHeaders={extraHeaders} />
        : <SpeechPlayground models={aliases} basePath={apiBase} curlBase={curlBase} storageKey={`serverless-ui:speech:proxy:${name}`} extraHeaders={extraHeaders} />}
    </div>
  );
}
