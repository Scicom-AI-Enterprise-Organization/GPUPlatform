"use client";

import { useEffect, useState } from "react";
import { Download, Loader2, PackageOpen, RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { gateway } from "@/lib/gateway";
import type { BenchmarkFile, BenchmarkRecord } from "@/lib/types";

export function FilesTab({ bench }: { bench: BenchmarkRecord }) {
  const [files, setFiles] = useState<BenchmarkFile[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);

  // Download a self-contained export (results + config + S3 files) for importing
  // into another deployment's dashboard via /benchmark/import.
  async function onExport() {
    setExporting(true);
    try {
      const data = await gateway.exportBenchmark(bench.id);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${bench.id}.benchmark.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      const omitted = Array.isArray((data as { files_omitted?: unknown[] }).files_omitted)
        ? (data as { files_omitted: unknown[] }).files_omitted.length
        : 0;
      toast.success(
        omitted > 0
          ? `Exported (${omitted} file${omitted === 1 ? "" : "s"} omitted — over size cap)`
          : "Benchmark exported",
        { duration: 3000 },
      );
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setExporting(false);
    }
  }

  async function refresh() {
    try {
      const list = await gateway.listBenchmarkFiles(bench.id);
      setFiles(list);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  useEffect(() => {
    refresh();
    // Auto-poll while not terminal — files appear as benchmaq drops them.
    const terminal =
      bench.status === "done" ||
      bench.status === "failed" ||
      bench.status === "cancelled";
    if (terminal) return;
    const t = setInterval(refresh, 8000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bench.id, bench.status]);

  return (
    <div>
      <div className="mb-4 flex items-end justify-between">
        <div>
          <h2 className="text-lg font-semibold">Files</h2>
          <p className="text-xs text-muted-foreground">
            S3 prefix: <span className="font-mono">{bench.s3_prefix}</span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={onExport} disabled={exporting}>
            {exporting ? <Loader2 className="h-4 w-4 animate-spin" /> : <PackageOpen className="h-4 w-4" />}
            Export
          </Button>
          <Button variant="outline" size="sm" onClick={refresh}>
            <RefreshCw className="h-4 w-4" /> Refresh
          </Button>
        </div>
      </div>

      {error && (
        <div className="mb-3 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      {files === null ? (
        <div className="rounded-md border border-border px-4 py-8 text-center text-sm text-muted-foreground">
          Loading…
        </div>
      ) : files.length === 0 ? (
        <div className="rounded-md border border-dashed border-border px-6 py-12 text-center text-sm text-muted-foreground">
          No files yet — benchmaq uploads here when the run finishes.
        </div>
      ) : (
        <div className="overflow-hidden rounded-md border border-border">
          <table className="w-full text-sm">
            <thead className="bg-muted/40 text-xs uppercase text-muted-foreground">
              <tr>
                <th className="px-3 py-2 text-left">Name</th>
                <th className="px-3 py-2 text-right">Size</th>
                <th className="px-3 py-2 text-left">Modified</th>
                <th className="px-3 py-2"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {files.map((f) => (
                <tr key={f.name}>
                  <td className="px-3 py-2 font-mono text-xs">{f.name}</td>
                  <td className="px-3 py-2 text-right font-mono text-xs">{formatBytes(f.size)}</td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">
                    {new Date(f.modified).toLocaleString()}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <a
                      href={f.download_url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
                    >
                      <Download className="h-3 w-3" /> Download
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}
