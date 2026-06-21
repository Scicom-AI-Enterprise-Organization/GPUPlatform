"use client";

import { useState } from "react";
import { Copy, Loader2, Play } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { ChatPlayground, openAiTransport } from "@/components/playground/chat-playground";

// ---- Embeddings playground ----

function EmbeddingsPlayground({ aliases, baseUrl, proxyName }: { aliases: string[]; baseUrl: string; proxyName: string }) {
  const [model, setModel] = useState(aliases[0] ?? "");
  const [input, setInput] = useState("Hello, world");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ dims: number; preview: number[]; raw: object } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    if (!model) { setErr("Pick a model."); return; }
    if (!input.trim()) { setErr("Enter some text."); return; }
    setErr(null); setResult(null); setLoading(true);
    try {
      const res = await fetch(`/api/proxy/proxy/${encodeURIComponent(proxyName)}/v1/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, input: input.trim() }),
      });
      const json = await res.json();
      if (!res.ok) { setErr(json?.detail ?? json?.error ?? `HTTP ${res.status}`); return; }
      const emb: number[] = json?.data?.[0]?.embedding ?? [];
      setResult({ dims: emb.length, preview: emb.slice(0, 8), raw: json });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader><CardTitle className="text-sm font-medium">Embeddings</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">model</span>
          <Select value={model} onValueChange={setModel}>
            <SelectTrigger className="h-7 w-auto min-w-40 text-xs font-mono">
              <SelectValue placeholder="select model" />
            </SelectTrigger>
            <SelectContent>
              {aliases.map((a) => <SelectItem key={a} value={a} className="font-mono text-xs">{a}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Text to embed…"
          className="min-h-24 font-mono text-xs"
        />
        <Button size="sm" onClick={run} disabled={loading}>
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
          {loading ? "Running…" : "Embed"}
        </Button>
        {err && <p className="rounded border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">{err}</p>}
        {result && (
          <div className="space-y-2 rounded-md border border-border bg-muted/30 p-3 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">dims: <span className="font-mono text-foreground">{result.dims}</span></span>
              <Button variant="outline" size="xs" onClick={() => { navigator.clipboard.writeText(JSON.stringify(result.raw, null, 2)); toast.success("Copied", { duration: 2000 }); }}>
                <Copy className="h-3 w-3" /> Copy JSON
              </Button>
            </div>
            <div className="font-mono text-muted-foreground">
              [{result.preview.map((v) => v.toFixed(6)).join(", ")}{result.dims > 8 ? ", …" : ""}]
            </div>
          </div>
        )}
        <p className="text-[11px] text-muted-foreground">
          Routes to <code className="font-mono">{baseUrl}/proxy/{proxyName}/v1/embeddings</code>
        </p>
      </CardContent>
    </Card>
  );
}

// ---- Proxy playground with mode toggle ----

type Mode = "chat" | "embeddings";

export function ProxyPlayground({ name, aliases, baseUrl }: { name: string; aliases: string[]; baseUrl: string }) {
  const [mode, setMode] = useState<Mode>("chat");

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1 rounded-md border border-border bg-muted/40 p-1 w-fit">
        <button
          onClick={() => setMode("chat")}
          className={`rounded px-3 py-1 text-xs font-medium transition-colors ${mode === "chat" ? "bg-background shadow text-foreground" : "text-muted-foreground hover:text-foreground"}`}
        >
          Chat
        </button>
        <button
          onClick={() => setMode("embeddings")}
          className={`rounded px-3 py-1 text-xs font-medium transition-colors ${mode === "embeddings" ? "bg-background shadow text-foreground" : "text-muted-foreground hover:text-foreground"}`}
        >
          Embeddings
        </button>
      </div>

      {mode === "chat" ? (
        <ChatPlayground
          models={aliases}
          storageKey={`serverless-ui:proxy-playground:${name}`}
          description={<>Routes through <code className="font-mono">POST /proxy/{name}/v1/chat/completions</code> to a live backend (priority + failover). Hit Stop mid-stream to trigger the proxy&apos;s auto-cancel.</>}
          transport={openAiTransport({
            fetchPath: `/api/proxy/proxy/${encodeURIComponent(name)}/v1/chat/completions`,
            curlUrl: `${baseUrl}/proxy/${name}/v1/chat/completions`,
          })}
        />
      ) : (
        <EmbeddingsPlayground aliases={aliases} baseUrl={baseUrl} proxyName={name} />
      )}
    </div>
  );
}
