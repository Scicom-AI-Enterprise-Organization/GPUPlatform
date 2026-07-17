"use client";

// TTS playground for endpoints / proxies that serve a text-to-speech model. Types
// text, picks a speaker, and calls OpenAI-compatible /v1/audio/speech through the
// Next proxy — which forwards to the backend and (for a proxy with an STT callback
// configured) transcribes the result to record CER/WER in Prometheus, asynchronously.
import { useCallback, useEffect, useRef, useState } from "react";
import { AudioLines, Loader2, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";

type SpeechRun = {
  id: string;
  at: number;
  model: string;
  voice: string;
  text: string;
  url: string;   // object URL (session-lived)
};

const MAX_HISTORY = 20;

// This TTS engine streams WAV with bogus 0x7FFFFFFF RIFF/data sizes (unknown length);
// some browsers won't decode that. We know the real byte length here, so patch the
// RIFF chunk size and the data-chunk size to real values before playback. No-op if the
// blob isn't a standard RIFF/WAVE.
function fixWavSizes(buf: ArrayBuffer): ArrayBuffer {
  const b = new Uint8Array(buf);
  if (b.length < 44 || b[0] !== 0x52 || b[1] !== 0x49 || b[2] !== 0x46 || b[3] !== 0x46) return buf; // "RIFF"
  const dv = new DataView(buf);
  // locate the "data" chunk (0x64 0x61 0x74 0x61), scanning past fmt/other chunks
  let off = 12;
  while (off + 8 <= b.length) {
    const id = String.fromCharCode(b[off], b[off + 1], b[off + 2], b[off + 3]);
    const size = dv.getUint32(off + 4, true);
    if (id === "data") {
      dv.setUint32(4, b.length - 8, true);          // RIFF chunk size
      dv.setUint32(off + 4, b.length - (off + 8), true); // data chunk size
      return buf;
    }
    off += 8 + size + (size % 2);
    if (size === 0xffffffff || size < 0) break; // bogus size on a non-data chunk → give up
  }
  return buf;
}

// Generic TTS playground — reused by the proxy playground (and any serverless TTS
// endpoint). POSTs to `${basePath}/audio/speech`, plays the returned audio, and
// pulls the speaker list from `${basePath}/audio/speaker`.
export function SpeechPlayground({ models, basePath, storageKey }: { models: string[]; basePath: string; storageKey: string }) {
  const [model, setModel] = useState(models[0] ?? "");
  const [text, setText] = useState("Hello, this is a test of the text to speech system.");
  const [voice, setVoice] = useState("");
  const [voices, setVoices] = useState<string[]>([]);
  const [voicesErr, setVoicesErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [runs, setRuns] = useState<SpeechRun[]>([]);
  const runsRef = useRef(runs);
  runsRef.current = runs;

  // Speaker dropdown — from the endpoint's /v1/audio/speaker (array of names, or
  // {voices:[…]} / {speakers:[…]}). Best-effort; a failure just leaves it free-text-less.
  useEffect(() => {
    let cancelled = false;
    fetch(`${basePath}/audio/speaker`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((list) => {
        if (cancelled) return;
        const arr: string[] = Array.isArray(list)
          ? list.map(String)
          : Array.isArray(list?.voices) ? list.voices.map(String)
          : Array.isArray(list?.speakers) ? list.speakers.map(String)
          : Array.isArray(list?.data) ? list.data.map((v: unknown) => (typeof v === "string" ? v : (v as { id?: string })?.id ?? String(v)))
          : [];
        setVoices(arr);
        setVoice((v) => v || arr[0] || "");
      })
      .catch((e) => { if (!cancelled) setVoicesErr(e instanceof Error ? e.message : String(e)); });
    return () => { cancelled = true; };
  }, [basePath]);

  // Revoke object URLs on unmount to avoid leaking blobs.
  useEffect(() => () => { runsRef.current.forEach((r) => URL.revokeObjectURL(r.url)); }, []);

  const removeRun = useCallback((id: string) => {
    setRuns((rs) => {
      const gone = rs.find((r) => r.id === id);
      if (gone) URL.revokeObjectURL(gone.url);
      return rs.filter((r) => r.id !== id);
    });
  }, []);

  async function onRun() {
    if (!text.trim() || !model) return;
    setBusy(true); setErr(null);
    try {
      // Ask for wav so the browser can play it (the engine's default is raw PCM).
      const r = await fetch(`${basePath}/audio/speech`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, input: text, voice: voice || undefined, response_format: "wav" }),
      });
      if (!r.ok) {
        const raw = await r.text();
        let body: unknown = raw;
        try { body = raw ? JSON.parse(raw) : null; } catch { /* keep raw */ }
        const detail = (body as { error?: { message?: string } | string })?.error ?? (body as { detail?: unknown })?.detail ?? body;
        setErr(typeof detail === "string" ? detail : JSON.stringify(detail));
        return;
      }
      const buf = fixWavSizes(await r.arrayBuffer());
      const url = URL.createObjectURL(new Blob([buf], { type: r.headers.get("content-type") || "audio/wav" }));
      const entry: SpeechRun = {
        id: globalThis.crypto?.randomUUID?.() ?? `r-${Date.now()}`,
        at: Date.now(), model, voice, text: text.trim(), url,
      };
      setRuns((rs) => {
        const next = [entry, ...rs];
        next.slice(MAX_HISTORY).forEach((r) => URL.revokeObjectURL(r.url));
        return next.slice(0, MAX_HISTORY);
      });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Text to speech</CardTitle>
          <p className="text-xs text-muted-foreground">
            Calls <code className="font-mono">/v1/audio/speech</code> and plays the result. If this proxy has an
            STT callback configured, <span className="font-mono">CER</span>/<span className="font-mono">WER</span> are
            scored against this text and recorded to Prometheus <span className="font-medium">asynchronously</span> — not returned here.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">text</span>
            <Textarea value={text} onChange={(e) => setText(e.target.value)} rows={3}
                      placeholder="Type something to synthesize…" className="text-sm" />
          </div>
          <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">model</span>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger className="h-8 w-[220px] font-mono text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {models.map((m) => <SelectItem key={m} value={m} className="font-mono text-xs">{m}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">speaker</span>
              {voices.length > 0 ? (
                <Select value={voice} onValueChange={setVoice}>
                  <SelectTrigger className="h-8 w-[200px] text-xs"><SelectValue placeholder="(default)" /></SelectTrigger>
                  <SelectContent>
                    {voices.map((v) => <SelectItem key={v} value={v} className="text-xs">{v}</SelectItem>)}
                  </SelectContent>
                </Select>
              ) : (
                <span className="flex h-8 items-center text-[11px] text-muted-foreground">
                  {voicesErr ? `no speaker list (${voicesErr})` : "loading speakers…"}
                </span>
              )}
            </div>
            <Button type="button" onClick={onRun} disabled={busy || !text.trim() || !model} className="ml-auto">
              {busy ? <><Loader2 className="h-4 w-4 animate-spin" /> Synthesizing…</> : <><AudioLines className="h-4 w-4" /> Generate</>}
            </Button>
          </div>
          {err && <p className="text-sm text-destructive">{err}</p>}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-sm font-medium">Generated audio</CardTitle>
            <p className="text-xs text-muted-foreground">Session only (in-memory). {runs.length} of {MAX_HISTORY} max.</p>
          </div>
          {runs.length > 0 && (
            <CardAction>
              <Button variant="outline" size="xs" onClick={() => { runsRef.current.forEach((r) => URL.revokeObjectURL(r.url)); setRuns([]); }}
                      className="text-muted-foreground hover:text-destructive">
                <Trash2 className="h-3 w-3" /> Clear all
              </Button>
            </CardAction>
          )}
        </CardHeader>
        <CardContent className="space-y-2">
          {runs.length === 0 ? (
            <p className="py-4 text-center text-xs text-muted-foreground">No audio generated yet.</p>
          ) : (
            runs.map((r) => (
              <div key={r.id} className="flex items-center gap-3 rounded-md border border-border px-3 py-2">
                <span className="rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">{r.voice || "default"}</span>
                <audio controls src={r.url} className="h-8 max-w-[280px]" />
                <span className="truncate text-xs text-muted-foreground">{r.text}</span>
                <span className="ml-auto shrink-0 text-[11px] text-muted-foreground">{new Date(r.at).toLocaleTimeString()}</span>
                <button type="button" onClick={() => removeRun(r.id)} className="shrink-0 text-muted-foreground hover:text-destructive" aria-label="Remove">
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}
