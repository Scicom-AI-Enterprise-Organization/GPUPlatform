"use client";

import { useCallback, useEffect, useState } from "react";
import { Download, Loader2, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { gateway } from "@/lib/gateway";
import type { DatasetFile } from "@/lib/types";

function fmtBytes(n?: number | null): string {
  if (!n && n !== 0) return "—";
  if (n < 1024) return `${n} B`;
  const u = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(1)} ${u[i]}`;
}

/** Files tab — the S3 objects backing the dataset, with presigned download links.
 * Split-aware: for a tts_packed dataset, passing `split` lists only that subdir. */
export function DatasetFilesCard({ datasetId, split }: { datasetId: string; split?: string | null }) {
  const [files, setFiles] = useState<DatasetFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      setFiles(await gateway.listDatasetFiles(datasetId, split || undefined));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setFiles([]);
    } finally {
      setLoading(false);
    }
  }, [datasetId, split]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  if (loading)
    return (
      <p className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading files…
      </p>
    );
  if (err) return <p className="text-sm text-destructive">{err}</p>;
  if (files.length === 0)
    return (
      <p className="text-sm text-muted-foreground">
        No files{split ? ` for split “${split}”` : ""}.
      </p>
    );

  const total = files.reduce((a, f) => a + (f.size || 0), 0);
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs text-muted-foreground">
          <span className="font-mono text-foreground">{files.length}</span> object{files.length === 1 ? "" : "s"} ·{" "}
          {fmtBytes(total)}
          {split ? <> · split <span className="font-mono text-foreground">{split}</span></> : null}
          {files.length >= 1000 ? " · showing first 1000" : ""}
        </p>
        <Button variant="outline" size="xs" onClick={load}>
          <RotateCw className="h-3.5 w-3.5" /> Refresh
        </Button>
      </div>
      <ul className="divide-y divide-border rounded-md border border-border">
        {files.map((f) => (
          <li key={f.key} className="flex items-center justify-between gap-4 px-4 py-2 text-sm">
            <span className="truncate font-mono text-xs" title={f.key}>
              {f.name}
            </span>
            <span className="flex shrink-0 items-center gap-3 text-xs text-muted-foreground">
              {f.modified && <span className="tabular-nums">{new Date(f.modified).toLocaleString()}</span>}
              <a
                href={f.download_url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 hover:text-foreground"
              >
                <Download className="h-3.5 w-3.5" /> {fmtBytes(f.size)}
              </a>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
