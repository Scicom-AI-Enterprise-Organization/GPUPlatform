"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowRight, Loader2, Sparkles, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ProgressEta } from "@/components/progress-eta";
import { gateway, GatewayError } from "@/lib/gateway";

function errText(body: unknown, fallback: string): string {
  if (typeof body === "string") return body || fallback;
  if (body && typeof body === "object") {
    const d = (body as Record<string, unknown>).detail;
    if (typeof d === "string") return d;
    if (d && typeof d === "object" && typeof (d as Record<string, unknown>).error === "string") {
      return (d as Record<string, string>).error;
    }
  }
  return fallback;
}

const DEFAULT_BASE_URL = "https://serverlessgpu.aies.scicom.dev/proxy/for-agentic/v1";
const DEFAULT_MODEL = "google/gemma-4-31b-it";

/**
 * LLM transcription-normalization card (kind=s3 audio datasets only). Runs a
 * constrained-respelling pass over the transcription column (particle/filler
 * spellings, Malay affix spacing, zh spacing) via any OpenAI-compatible chat
 * endpoint, and registers a NEW kind=s3 dataset over the SAME audio — metadata
 * only, nothing is re-uploaded. Mirrors TransformCard's poll/cancel/log UX.
 */
export function NormalizeCard({
  datasetId,
  initialStatus,
  initialLog,
}: {
  datasetId: string;
  initialStatus: string | null;
  initialLog: string | null;
}) {
  const router = useRouter();
  const [baseUrl, setBaseUrl] = useState(DEFAULT_BASE_URL);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [apiKey, setApiKey] = useState("");
  const [workers, setWorkers] = useState("8");
  const [judge, setJudge] = useState(false);
  const [limit, setLimit] = useState("");
  const [status, setStatus] = useState<string | null>(initialStatus);
  const [log, setLog] = useState<string | null>(initialLog);
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const poll = useRef<ReturnType<typeof setInterval> | null>(null);
  const logRef = useRef<HTMLPreElement | null>(null);

  const running = status === "running";
  const newDatasetId =
    status === "done" ? log?.match(/created dataset (ds-[0-9a-f]+)/i)?.[1] : undefined;

  // Poll the dataset while a job is running; refresh the page when it ends.
  useEffect(() => {
    if (!running) {
      if (poll.current) {
        clearInterval(poll.current);
        poll.current = null;
      }
      return;
    }
    const id = setInterval(async () => {
      try {
        const d = await gateway.getDataset(datasetId);
        setStatus(d.transform_status ?? null);
        setLog(d.transform_log ?? null);
        if (d.transform_status !== "running") router.refresh();
      } catch {
        /* transient; keep polling */
      }
    }, 3000);
    poll.current = id;
    return () => clearInterval(id);
  }, [running, datasetId, router]);

  // Auto-scroll the live log to the newest line while running.
  useEffect(() => {
    if (running && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log, running]);

  async function run() {
    setErr(null);
    if (!baseUrl.trim()) {
      setErr("Enter the LLM base URL (an OpenAI-compatible endpoint).");
      return;
    }
    if (!model.trim()) {
      setErr("Enter the model id to use.");
      return;
    }
    const w = Number(workers);
    if (!Number.isFinite(w) || w < 1) {
      setErr("Enter a valid worker count (≥ 1).");
      return;
    }
    let lim: number | null = null;
    if (limit.trim()) {
      const l = Number(limit);
      if (!Number.isFinite(l) || l < 0) {
        setErr("Enter a valid row limit (≥ 0, or blank for all).");
        return;
      }
      lim = Math.round(l) || null;
    }
    setStarting(true);
    try {
      const d = await gateway.normalizeTranscription(datasetId, {
        base_url: baseUrl.trim(),
        model: model.trim(),
        api_key: apiKey.trim() || null,
        workers: Math.min(32, Math.max(1, Math.round(w))),
        judge,
        limit: lim,
      });
      setStatus(d.transform_status ?? "running");
      setLog(d.transform_log ?? null);
    } catch (e) {
      setErr(
        e instanceof GatewayError
          ? errText(e.parsed, e.message)
          : e instanceof Error ? e.message : String(e),
      );
    } finally {
      setStarting(false);
    }
  }

  async function cancel() {
    setErr(null);
    setCancelling(true);
    try {
      const d = await gateway.cancelDatasetTransform(datasetId);
      setStatus(d.transform_status ?? null);
      setLog(d.transform_log ?? null);
      router.refresh();
    } catch (e) {
      setErr(
        e instanceof GatewayError
          ? errText(e.parsed, e.message)
          : e instanceof Error ? e.message : String(e),
      );
    } finally {
      setCancelling(false);
    }
  }

  const desc = (
    <p className="text-xs text-muted-foreground">
      Rewrite the <span className="font-mono">transcription</span> column with an LLM
      constrained-respelling pass — canonicalises particle/filler spellings (la/lah, ya/ye),
      Malay affix spacing, and Chinese spacing <em>without</em> changing what was said. Creates a
      new <span className="font-mono">s3</span> dataset over the <strong>same</strong> audio
      (metadata only — nothing is re-uploaded). Every normalization is guarded (deterministic
      structural check{judge ? " + LLM judge" : ""}); a rejected row keeps its original text. Runs
      on the gateway; watch progress below.
    </p>
  );

  const body = (
    <>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-1 sm:col-span-2">
          <Label htmlFor="nm-base" className="text-xs">LLM base URL (OpenAI-compatible)</Label>
          <Input
            id="nm-base"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder={DEFAULT_BASE_URL}
            disabled={running}
            className="font-mono text-xs"
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="nm-model" className="text-xs">Model</Label>
          <Input
            id="nm-model"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder={DEFAULT_MODEL}
            disabled={running}
            className="font-mono text-xs"
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="nm-key" className="text-xs">API key (optional)</Label>
          <Input
            id="nm-key"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-… / sgpu_…"
            disabled={running}
            className="font-mono text-xs"
          />
        </div>
        <div className="flex items-end gap-4">
          <div className="space-y-1">
            <Label htmlFor="nm-workers" className="text-xs">Concurrent workers</Label>
            <Input
              id="nm-workers"
              type="number"
              min={1}
              max={32}
              value={workers}
              onChange={(e) => setWorkers(e.target.value)}
              disabled={running}
              className="h-9 w-24 text-xs"
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="nm-limit" className="text-xs">Limit rows (blank = all)</Label>
            <Input
              id="nm-limit"
              type="number"
              min={0}
              placeholder="all"
              value={limit}
              onChange={(e) => setLimit(e.target.value)}
              disabled={running}
              className="h-9 w-24 text-xs"
            />
          </div>
        </div>
        <div className="flex items-end">
          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              className="h-4 w-4 accent-primary"
              checked={judge}
              onChange={(e) => setJudge(e.target.checked)}
              disabled={running}
            />
            Extra LLM judge pass (2× calls; noisy — off by default)
          </label>
        </div>
      </div>

      {err && <p className="text-sm text-destructive">{err}</p>}

      {log && (
        <div className="space-y-1">
          <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            {running && <Loader2 className="h-3 w-3 animate-spin" />}
            <span>{running ? "Live log" : "Log"}</span>
            <ProgressEta log={log} running={running} />
          </div>
          <pre ref={logRef} className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-200 scrollbar-thin">
            {log}
          </pre>
        </div>
      )}
    </>
  );

  const actions = (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex items-center gap-3">
        {status && status !== "running" && (
          <span className={status === "done" ? "text-sm text-emerald-600 dark:text-emerald-400" : "text-sm text-destructive"}>
            {status === "done" ? "✓ done" : `✕ ${status}`}
          </span>
        )}
        {newDatasetId && (
          <Link
            href={`/datasets/${newDatasetId}`}
            className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
          >
            Open new dataset <span className="font-mono text-xs">{newDatasetId}</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        )}
      </div>
      <div className="flex items-center gap-3">
        {running && (
          <Button variant="outline" onClick={cancel} disabled={cancelling} className="text-destructive">
            {cancelling ? <Loader2 className="h-4 w-4 animate-spin" /> : <X className="h-4 w-4" />}
            {cancelling ? "Cancelling…" : "Cancel"}
          </Button>
        )}
        <Button onClick={run} disabled={running || starting}>
          {running || starting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
          {running ? "Normalizing…" : "Normalize transcriptions"}
        </Button>
      </div>
    </div>
  );

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="flex flex-col gap-0.5">
          <CardTitle className="text-base">Normalize transcription — LLM respelling</CardTitle>
          {desc}
        </CardHeader>
        <CardContent className="space-y-3">{body}</CardContent>
      </Card>
      {actions}
    </div>
  );
}
