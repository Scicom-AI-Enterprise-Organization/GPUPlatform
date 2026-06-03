"use client";

// Audio playground for endpoints that serve a Whisper / ASR model. Uploads a clip
// and calls the endpoint's OpenAI-compatible /v1/audio/transcriptions (or
// /translations) through the Next proxy — which forwards multipart verbatim.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AudioLines, Loader2, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import type { AppRecord } from "@/lib/types";

export const AUDIO_MODEL_RE = /whisper|asr|transcrib|audio|speech-to-text|stt/i;

/** Models this endpoint serves that look like ASR/Whisper models. */
export function audioModelsOf(app: AppRecord): string[] {
  if (app.mode === "multi" && app.models?.length) {
    return app.models.map((m) => m.model).filter((m): m is string => !!m && AUDIO_MODEL_RE.test(m));
  }
  return app.model && AUDIO_MODEL_RE.test(app.model) ? [app.model] : [];
}

type AudioRun = {
  id: string;
  at: number;        // ms epoch
  model: string;
  task: "transcriptions" | "translations";
  filename: string;
  language?: string;
  text: string;
};

const STORAGE_KEY = (appId: string) => `serverless-ui:transcribe:${appId}`;
const MAX_HISTORY = 50;

export function TranscribeTab({ app }: { app: AppRecord }) {
  // List EVERY member (ASR-looking ones first for convenience) — a custom ASR
  // finetune may have any name, so we never hide a model behind a heuristic; you
  // pick whichever member is your Whisper/ASR model.
  const models = useMemo(() => {
    if (app.mode === "multi" && app.models?.length) {
      const all = app.models.map((m) => m.model).filter(Boolean) as string[];
      return [...all.filter((m) => AUDIO_MODEL_RE.test(m)), ...all.filter((m) => !AUDIO_MODEL_RE.test(m))];
    }
    return app.model ? [app.model] : [];
  }, [app]);

  const [model, setModel] = useState(models[0] ?? "");
  const [task, setTask] = useState<"transcriptions" | "translations">("transcriptions");
  const [language, setLanguage] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [text, setText] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // History — tracked per browser (localStorage), like the chat playground.
  const [history, setHistory] = useState<AudioRun[]>([]);
  const historyRef = useRef(history);
  historyRef.current = history;
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY(app.app_id));
      if (raw) setHistory(JSON.parse(raw));
    } catch { /* ignore corrupt/absent */ }
  }, [app.app_id]);
  const persist = useCallback((next: AudioRun[]) => {
    setHistory(next);
    try { window.localStorage.setItem(STORAGE_KEY(app.app_id), JSON.stringify(next)); } catch { /* quota */ }
  }, [app.app_id]);
  const removeRun = useCallback((id: string) => persist(historyRef.current.filter((r) => r.id !== id)), [persist]);
  const clearAll = useCallback(() => persist([]), [persist]);

  async function onRun() {
    if (!file || !model) return;
    setBusy(true); setErr(null); setText(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("model", model);
      fd.append("response_format", "json");
      if (task === "transcriptions" && language.trim()) fd.append("language", language.trim());
      // No explicit Content-Type — the browser sets multipart/form-data + boundary.
      const r = await fetch(`/api/proxy/${encodeURIComponent(app.app_id)}/v1/audio/${task}`, {
        method: "POST",
        body: fd,
      });
      const raw = await r.text();
      let body: unknown = raw;
      try { body = raw ? JSON.parse(raw) : null; } catch { /* keep raw */ }
      if (!r.ok) {
        const detail = (body as { detail?: unknown })?.detail ?? body;
        setErr(typeof detail === "string" ? detail : JSON.stringify(detail));
        return;
      }
      const out = (body as { text?: string })?.text;
      const resultText = typeof out === "string" ? out : JSON.stringify(body, null, 2);
      setText(resultText);
      const entry: AudioRun = {
        id: (globalThis.crypto?.randomUUID?.() ?? `r-${Date.now()}-${Math.round(Math.random() * 1e6)}`),
        at: Date.now(),
        model,
        task,
        filename: file.name || "audio",
        ...(task === "transcriptions" && language.trim() ? { language: language.trim() } : {}),
        text: resultText,
      };
      persist([entry, ...historyRef.current].slice(0, MAX_HISTORY));
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
          <CardTitle className="text-sm font-medium">Transcribe audio</CardTitle>
          <p className="text-xs text-muted-foreground">
            Calls this endpoint&apos;s <code className="font-mono">/v1/audio/{task}</code> with a Whisper model.
            Upload a clip (wav/flac/ogg/mp3 supported; m4a/webm too). Translate outputs English. Max 25 MB.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">audio clip</span>
            <input
              type="file"
              accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm"
              onChange={(e) => { setFile(e.target.files?.[0] ?? null); setText(null); }}
              className="flex h-8 items-center rounded-md border border-input bg-transparent text-xs shadow-xs file:mr-3 file:h-8 file:border-0 file:border-r file:border-input file:bg-muted file:px-2.5 file:text-foreground hover:file:bg-muted/70"
            />
          </div>
          <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">model</span>
              <Select value={model} onValueChange={setModel}>
                <SelectTrigger className="h-8 w-[260px] font-mono text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {models.map((m) => <SelectItem key={m} value={m} className="font-mono text-xs">{m}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-xs text-muted-foreground">task</span>
              <Select value={task} onValueChange={(v) => setTask(v as typeof task)}>
                <SelectTrigger className="h-8 w-[170px] text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="transcriptions" className="text-xs">Transcribe</SelectItem>
                  <SelectItem value="translations" className="text-xs">Translate → English</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {task === "transcriptions" && (
              <div className="flex flex-col gap-1">
                <span className="text-xs text-muted-foreground">language (optional)</span>
                <Input
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  placeholder="auto (e.g. en, ms)"
                  className="h-8 w-[160px] text-sm"
                />
              </div>
            )}
            <Button type="button" onClick={onRun} disabled={busy || !file || !model} className="ml-auto">
              {busy
                ? <><Loader2 className="h-4 w-4 animate-spin" /> {task === "translations" ? "Translating…" : "Transcribing…"}</>
                : <><AudioLines className="h-4 w-4" /> {task === "translations" ? "Translate" : "Transcribe"}</>}
            </Button>
          </div>
          {err && <p className="text-sm text-destructive">{err}</p>}
          {text != null && (
            <div className="space-y-1.5">
              <div className="text-xs text-muted-foreground">Result</div>
              <pre className="max-h-72 overflow-y-auto whitespace-pre-wrap rounded-md border border-border bg-muted/30 p-3 text-sm">{text}</pre>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
          <div>
            <CardTitle className="text-sm font-medium">Transcription history</CardTitle>
            <p className="text-xs text-muted-foreground">Tracked per browser. {history.length} of {MAX_HISTORY} max.</p>
          </div>
          {history.length > 0 && (
            <Button variant="outline" size="xs" onClick={clearAll} className="text-muted-foreground hover:text-destructive">
              <Trash2 className="h-3 w-3" /> Clear all
            </Button>
          )}
        </CardHeader>
        <CardContent className="space-y-2">
          {history.length === 0 ? (
            <p className="py-4 text-center text-xs text-muted-foreground">No transcriptions yet.</p>
          ) : (
            history.map((r) => (
              <details key={r.id} className="group rounded-md border border-border">
                <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-xs">
                  <span className="rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                    {r.task === "translations" ? "translate" : "transcribe"}
                  </span>
                  <span className="font-mono text-[11px] text-muted-foreground">{r.model}</span>
                  <span className="truncate">{r.filename}</span>
                  <span className="ml-auto shrink-0 text-[11px] text-muted-foreground">{new Date(r.at).toLocaleString()}</span>
                  <button
                    type="button"
                    onClick={(e) => { e.preventDefault(); removeRun(r.id); }}
                    className="shrink-0 text-muted-foreground hover:text-destructive"
                    aria-label="Remove"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </summary>
                <pre className="max-h-60 overflow-y-auto whitespace-pre-wrap border-t border-border bg-muted/30 p-3 text-sm">{r.text}</pre>
              </details>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}
