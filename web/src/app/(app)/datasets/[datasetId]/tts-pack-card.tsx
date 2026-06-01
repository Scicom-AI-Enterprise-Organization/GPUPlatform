"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowRight, Boxes, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NumberField } from "@/components/ui/number-field";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { gateway } from "@/lib/gateway";
import type { DatasetRecord, ProviderRecord, StorageRecord } from "@/lib/types";

function errText(body: unknown, fallback: string): string {
  if (typeof body === "string") return body || fallback;
  if (body && typeof body === "object") {
    const d = (body as Record<string, unknown>).detail;
    if (typeof d === "string") return d;
  }
  return fallback;
}

const DEFAULT_TOKENIZER = "Scicom-intl/Multilingual-Expressive-TTS-1.7B";

// NeuCodec-encode + multipack a {audio, transcription} dataset into a ChiniDataset
// (Parquet streaming) on a GPU provider over SSH → a new packed dataset that TTS
// training streams directly (skips convert+pack per run).
export function TtsPackCard({
  datasetId,
  s3Storages,
  initialStatus,
  initialLog,
  bare = false,
}: {
  datasetId: string;
  s3Storages: StorageRecord[];
  initialStatus: string | null;
  initialLog: string | null;
  bare?: boolean;
}) {
  const router = useRouter();
  const [providers, setProviders] = useState<ProviderRecord[]>([]);
  const [providerId, setProviderId] = useState("");
  const [storageId, setStorageId] = useState(s3Storages[0]?.id ?? "");
  const [tokenizer, setTokenizer] = useState(DEFAULT_TOKENIZER);
  const [seqLen, setSeqLen] = useState(4096);
  const [visibleDevices, setVisibleDevices] = useState("");
  const [status, setStatus] = useState<string | null>(initialStatus);
  const [log, setLog] = useState<string | null>(initialLog);
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const poll = useRef<ReturnType<typeof setInterval> | null>(null);

  const running = status === "running";
  const newDatasetId =
    status === "done" ? log?.match(/created dataset (ds-[0-9a-f]+)/i)?.[1] : undefined;

  useEffect(() => {
    gateway
      .listProviders()
      .then((p) => {
        const gpu = p.filter((x) => x.kind === "vm" || x.kind === "runpod");
        setProviders(gpu);
        setProviderId((cur) => cur || gpu[0]?.id || "");
      })
      .catch(() => {});
  }, []);

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
        const r = await fetch(`/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}`, { cache: "no-store" });
        if (!r.ok) return;
        const d = (await r.json()) as DatasetRecord;
        setStatus(d.transform_status ?? null);
        setLog(d.transform_log ?? null);
        if (d.transform_status !== "running") router.refresh();
      } catch {
        /* transient; keep polling */
      }
    }, 4000);
    poll.current = id;
    return () => clearInterval(id);
  }, [running, datasetId, router]);

  async function run() {
    setErr(null);
    if (!providerId) return setErr("Pick a GPU provider (NeuCodec needs a GPU).");
    if (!storageId) return setErr("Pick an S3 storage for the packed shards.");
    setStarting(true);
    try {
      const r = await fetch(`/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}/pack-tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider_id: providerId,
          storage_id: storageId,
          tokenizer: tokenizer.trim() || null,
          sequence_length: seqLen,
          visible_devices: visibleDevices.trim() || null,
        }),
      });
      const text = await r.text();
      let parsed: unknown = text;
      try {
        parsed = text ? JSON.parse(text) : null;
      } catch {
        /* keep raw */
      }
      if (!r.ok) {
        setErr(errText(parsed, r.statusText));
        return;
      }
      const d = parsed as DatasetRecord;
      setStatus(d.transform_status ?? "running");
      setLog(d.transform_log ?? null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  }

  const desc = (
    <span className="text-xs text-muted-foreground">
      Encode the audio to <span className="font-mono">NeuCodec</span> speech tokens and multipack into a{" "}
      <span className="font-mono">ChiniDataset</span> (sequence length {seqLen}), then upload the Parquet shards to S3.
      TTS training streams the packed dataset directly, skipping convert+pack per run. Runs on a GPU provider over SSH.
    </span>
  );
  const body = (
    <div className="space-y-3">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="space-y-1">
            <Label className="text-xs">GPU provider</Label>
            <Select value={providerId} onValueChange={setProviderId} disabled={running}>
              <SelectTrigger className="text-xs">
                <SelectValue placeholder={providers.length ? "Pick a provider" : "No GPU providers"} />
              </SelectTrigger>
              <SelectContent>
                {providers.map((p) => (
                  <SelectItem key={p.id} value={p.id}>
                    {p.name} · {p.kind}{p.gpu_count ? ` · ${p.gpu_count} GPU` : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">S3 storage (packed shards)</Label>
            <Select value={storageId} onValueChange={setStorageId} disabled={running}>
              <SelectTrigger className="text-xs">
                <SelectValue placeholder={s3Storages.length ? "Pick a storage" : "No S3 storage"} />
              </SelectTrigger>
              <SelectContent>
                {s3Storages.map((s) => (
                  <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Sequence length (multipack)</Label>
            <NumberField min={256} value={seqLen} onChange={setSeqLen} />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">CUDA_VISIBLE_DEVICES (optional)</Label>
            <Input className="font-mono text-xs" placeholder="e.g. 6,7 (empty = all)"
              value={visibleDevices} onChange={(e) => setVisibleDevices(e.target.value)} disabled={running} />
          </div>
        </div>
        <div className="space-y-1">
          <Label className="text-xs">Speech-token tokenizer</Label>
          <Input className="font-mono text-xs" value={tokenizer}
            onChange={(e) => setTokenizer(e.target.value)} disabled={running} />
        </div>

        {err && <p className="text-sm text-destructive">{err}</p>}

        <div className="flex items-center gap-3">
          <Button onClick={run} disabled={running || starting}>
            {running || starting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Boxes className="h-4 w-4" />}
            {running ? "Packing…" : "Pack for TTS"}
          </Button>
          {status && status !== "running" && (
            <span className={status === "done" ? "text-sm text-emerald-600 dark:text-emerald-400" : "text-sm text-destructive"}>
              {status === "done" ? "✓ done" : `✕ ${status}`}
            </span>
          )}
          {newDatasetId && (
            <Link href={`/datasets/${newDatasetId}`} className="inline-flex items-center gap-1 text-sm text-primary hover:underline">
              Open packed dataset <span className="font-mono text-xs">{newDatasetId}</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          )}
        </div>

        {log && (
          <pre className="max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-200 scrollbar-thin">
            {log}
          </pre>
        )}
    </div>
  );

  if (bare) return <div className="space-y-3">{desc}{body}</div>;
  return (
    <Card>
      <CardHeader className="flex flex-col gap-0.5">
        <CardTitle className="text-base">Pack for TTS — NeuCodec + multipack</CardTitle>
        {desc}
      </CardHeader>
      <CardContent>{body}</CardContent>
    </Card>
  );
}
