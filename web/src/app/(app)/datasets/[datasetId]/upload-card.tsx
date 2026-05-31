"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { CloudUpload, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DatasetUploadResult } from "@/lib/types";

export function UploadCard({ datasetId, hasFile }: { datasetId: string; hasFile: boolean }) {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DatasetUploadResult | null>(null);

  const onUpload = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/upload`, {
        method: "POST",
        body: fd,
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(typeof body?.detail === "string" ? body.detail : body?.error || `upload failed (${res.status})`);
      }
      setResult(body as DatasetUploadResult);
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          {hasFile ? "Replace metadata file" : "Upload metadata file"}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground">
          A CSV / JSON / JSONL with an audio column (audio / audio_path / audio_url / url)
          and a transcription column (transcription / text / sentence / transcript).
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.json,.jsonl,.ndjson"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="block text-sm file:mr-3 file:rounded-md file:border file:border-border file:bg-muted/40 file:px-3 file:py-1.5 file:text-sm file:text-foreground hover:file:bg-muted"
          />
          <Button onClick={onUpload} disabled={!file || busy} size="sm">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <CloudUpload className="h-4 w-4" />}
            Upload
          </Button>
        </div>

        {error && (
          <p className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error}
          </p>
        )}

        {result && (
          <div className="rounded-md border border-border p-3 text-sm">
            <div className="mb-2 text-xs text-muted-foreground">
              Parsed {result.num_rows} rows · {result.format.toUpperCase()} · columns:{" "}
              {result.columns.join(", ")} · audio=<span className="font-mono">{result.audio_field}</span>,
              transcription=<span className="font-mono">{result.transcription_field}</span>
            </div>
            <pre className="max-h-48 overflow-auto rounded bg-muted/40 p-2 text-xs">
              {JSON.stringify(result.preview, null, 2)}
            </pre>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
