"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowRight, Loader2, Wand2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { DatasetRecord, StorageRecord } from "@/lib/types";

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

export function TransformCard({
  datasetId,
  hfRepo,
  s3Storages,
  initialStatus,
  initialLog,
}: {
  datasetId: string;
  hfRepo: string | null;
  s3Storages: StorageRecord[];
  initialStatus: string | null;
  initialLog: string | null;
}) {
  const router = useRouter();
  const [target, setTarget] = useState<"hf" | "s3">("hf");
  const [outRepo, setOutRepo] = useState(hfRepo ? `${hfRepo}-audio` : "");
  const [storageId, setStorageId] = useState(s3Storages[0]?.id ?? "");
  const [s3Folder, setS3Folder] = useState(`datasets/${datasetId}/transformed`);
  const [status, setStatus] = useState<string | null>(initialStatus);
  const [log, setLog] = useState<string | null>(initialLog);
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const poll = useRef<ReturnType<typeof setInterval> | null>(null);

  const running = status === "running";
  // The job log ends with "… created dataset ds-xxxxxxxx (N rows)" — surface a
  // link to the freshly-created audio dataset when the transform succeeds.
  const newDatasetId =
    status === "done" ? log?.match(/created dataset (ds-[0-9a-f]+)/i)?.[1] : undefined;

  // Poll the dataset while a transform is running; refresh the page when it ends.
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
    }, 3000);
    poll.current = id;
    return () => clearInterval(id);
  }, [running, datasetId, router]);

  async function run() {
    setErr(null);
    if (target === "hf" && (!outRepo.trim() || !outRepo.includes("/"))) {
      setErr("Enter the target HF repo as owner/name.");
      return;
    }
    if (target === "s3" && !storageId) {
      setErr("Pick an S3 storage.");
      return;
    }
    setStarting(true);
    try {
      const r = await fetch(`/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}/transform`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          target === "hf"
            ? { target: "hf", hf_repo: outRepo.trim() }
            : { target: "s3", storage_id: storageId, s3_folder: s3Folder.trim() || null },
        ),
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

  return (
    <Card>
      <CardHeader className="flex flex-col gap-0.5">
        <CardTitle className="text-base">Transform — extract audio column</CardTitle>
        <span className="text-xs text-muted-foreground">
          This repo stores audio in archives, so there&apos;s no audio column to preview. Unzip it and rebuild a
          dataset with a real audio column (joined on the <span className="font-mono">audio</span> column you set
          above). Runs on the gateway; watch progress below.
        </span>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
          {(["hf", "s3"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTarget(t)}
              disabled={running}
              className={
                "rounded px-2.5 py-1 transition-colors " +
                (target === t ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")
              }
            >
              {t === "hf" ? "Push to HuggingFace" : "Materialize to S3"}
            </button>
          ))}
        </div>

        {target === "hf" ? (
          <div className="space-y-1">
            <Label htmlFor="tf-repo" className="text-xs">Target HF repo</Label>
            <Input
              id="tf-repo"
              value={outRepo}
              onChange={(e) => setOutRepo(e.target.value)}
              placeholder="owner/dataset-with-audio"
              disabled={running}
              className="font-mono text-xs"
            />
          </div>
        ) : (
          <div className="space-y-3">
            <div className="space-y-1">
              <Label className="text-xs">S3 storage</Label>
              <Select value={storageId} onValueChange={setStorageId} disabled={running}>
                <SelectTrigger className="text-xs">
                  <SelectValue placeholder={s3Storages.length ? "Choose an S3 storage" : "No S3 storage configured"} />
                </SelectTrigger>
                <SelectContent>
                  {s3Storages.map((s) => (
                    <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label htmlFor="tf-folder" className="text-xs">Folder in S3</Label>
              <Input
                id="tf-folder"
                value={s3Folder}
                onChange={(e) => setS3Folder(e.target.value)}
                placeholder="datasets/my-audio"
                disabled={running}
                className="font-mono text-xs"
              />
              <p className="text-xs text-muted-foreground">
                Written under the storage&apos;s prefix. Audio → <span className="font-mono">{(s3Folder.trim() || "…").replace(/^\/+|\/+$/g, "")}/audio/</span>, metadata →{" "}
                <span className="font-mono">{(s3Folder.trim() || "…").replace(/^\/+|\/+$/g, "")}/metadata.csv</span>.
              </p>
            </div>
          </div>
        )}

        {err && <p className="text-sm text-destructive">{err}</p>}

        <div className="flex items-center gap-3">
          <Button onClick={run} disabled={running || starting}>
            {running || starting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Wand2 className="h-4 w-4" />}
            {running ? "Transforming…" : "Run transform"}
          </Button>
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

        {log && (
          <pre className="max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-200 scrollbar-thin">
            {log}
          </pre>
        )}
      </CardContent>
    </Card>
  );
}
