"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { CheckCircle2, Loader2, PackageOpen, Upload, X, XCircle } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { gateway } from "@/lib/gateway";
import { cn } from "@/lib/utils";

const EXPORT_KIND = "gpuplatform.autotrain.export";

type ParsedExport = {
  kind?: string;
  source_run_id?: string;
  run?: { name?: string; status?: string; task_type?: string; base_model?: string; result_json?: unknown };
  files?: unknown[];
  files_omitted?: unknown[];
};

type Entry = {
  key: string;
  fileName: string;
  parsed: ParsedExport | null;
  name: string;
  parseError?: string;
  status: "ready" | "importing" | "done" | "failed";
  resultId?: string;
  importError?: string;
};

export function ImportTrainingForm() {
  const router = useRouter();
  const [entries, setEntries] = useState<Entry[]>([]);
  const [dragging, setDragging] = useState(false);
  const [importing, setImporting] = useState(false);

  function addFiles(files: FileList | File[]) {
    const list = Array.from(files).filter((f) => /\.json$/i.test(f.name) || f.type === "application/json");
    if (list.length === 0) {
      toast.error("Drop .json autotrain export files");
      return;
    }
    list.forEach((f, i) => {
      const key = `${f.name}-${f.size}-${Date.now()}-${i}`;
      const reader = new FileReader();
      reader.onload = () => {
        let parsed: ParsedExport | null = null;
        let parseError: string | undefined;
        let name = "";
        try {
          const obj = JSON.parse(String(reader.result)) as ParsedExport;
          if (obj?.kind !== EXPORT_KIND) {
            parseError = "Not an autotrain export (wrong `kind`).";
          } else {
            parsed = obj;
            name = obj.run?.name ?? "";
          }
        } catch (e) {
          parseError = "Invalid JSON: " + (e instanceof Error ? e.message : String(e));
        }
        setEntries((prev) => [
          ...prev,
          { key, fileName: f.name, parsed, name, parseError, status: "ready" },
        ]);
      };
      reader.readAsText(f);
    });
  }

  function removeEntry(key: string) {
    setEntries((prev) => prev.filter((e) => e.key !== key));
  }
  function setEntryName(key: string, name: string) {
    setEntries((prev) => prev.map((e) => (e.key === key ? { ...e, name } : e)));
  }

  const importable = entries.filter((e) => e.parsed && e.status !== "done");

  async function onImportAll() {
    if (importable.length === 0) return;
    setImporting(true);
    let ok = 0;
    let lastId = "";
    for (const target of importable) {
      setEntries((prev) =>
        prev.map((e) => (e.key === target.key ? { ...e, status: "importing", importError: undefined } : e)),
      );
      try {
        const body = {
          ...target.parsed,
          run: {
            ...target.parsed!.run,
            name: target.name.trim() || target.parsed!.run?.name || "imported",
          },
        };
        const rec = await gateway.importTrainingRun(body);
        ok += 1;
        lastId = rec.id;
        setEntries((prev) =>
          prev.map((e) => (e.key === target.key ? { ...e, status: "done", resultId: rec.id } : e)),
        );
      } catch (e) {
        setEntries((prev) =>
          prev.map((en) =>
            en.key === target.key
              ? { ...en, status: "failed", importError: e instanceof Error ? e.message : String(e) }
              : en,
          ),
        );
      }
    }
    setImporting(false);
    if (ok > 0) {
      toast.success(`Imported ${ok} of ${importable.length} run${importable.length === 1 ? "" : "s"}`, {
        duration: 3500,
      });
      // All succeeded: jump to the single run, or the list for a batch. Any failures
      // → stay so the per-file errors remain visible.
      if (ok === importable.length) {
        if (ok === 1 && lastId) router.push(`/autotrain/${lastId}`);
        else router.push("/autotrain");
      }
    }
  }

  return (
    <div className="space-y-5">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setDragging(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
        }}
        onClick={() => document.getElementById("autotrain-import-file")?.click()}
        role="button"
        tabIndex={0}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed px-6 py-10 text-center transition-colors",
          dragging ? "border-primary bg-primary/5" : "border-border bg-card hover:border-foreground/40",
        )}
      >
        <Upload className="h-6 w-6 text-muted-foreground" />
        <span className="text-sm font-medium">
          Drag &amp; drop <span className="font-mono text-xs">.autotrain.json</span> files here, or click to choose
        </span>
        <span className="text-xs text-muted-foreground">
          Multiple files supported — exported from a run&apos;s Export button.
        </span>
        <input
          id="autotrain-import-file"
          type="file"
          accept="application/json,.json"
          multiple
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.length) addFiles(e.target.files);
            e.target.value = ""; // allow re-selecting the same file
          }}
        />
      </div>

      {entries.length > 0 && (
        <div className="space-y-2">
          {entries.map((e) => (
            <div
              key={e.key}
              className={cn(
                "rounded-lg border bg-card p-3",
                e.status === "failed" || e.parseError ? "border-destructive/40" : "border-border",
              )}
            >
              <div className="flex items-center gap-3">
                <StatusIcon status={e.parseError ? "failed" : e.status} />
                <div className="min-w-0 flex-1">
                  <div className="truncate font-mono text-xs text-muted-foreground">{e.fileName}</div>
                  {e.parsed ? (
                    <Input
                      value={e.name}
                      onChange={(ev) => setEntryName(e.key, ev.target.value)}
                      placeholder="run name"
                      maxLength={128}
                      disabled={importing || e.status === "done"}
                      className="mt-1 h-8"
                    />
                  ) : (
                    <div className="mt-0.5 text-xs text-destructive">{e.parseError}</div>
                  )}
                  {e.importError && <div className="mt-1 text-xs text-destructive">{e.importError}</div>}
                  {e.status === "done" && (
                    <div className="mt-1 text-xs text-emerald-600 dark:text-emerald-400">
                      Imported → {e.resultId}
                    </div>
                  )}
                  {e.parsed && (
                    <div className="mt-1 text-[11px] text-muted-foreground">
                      {e.parsed.run?.task_type ? `${e.parsed.run.task_type} · ` : ""}
                      {e.parsed.run?.base_model ? `${e.parsed.run.base_model} · ` : ""}
                      {Array.isArray(e.parsed.files) ? e.parsed.files.length : 0} files
                      {e.parsed.run?.result_json ? " · has results" : " · no results"}
                      {e.parsed.source_run_id ? ` · from ${e.parsed.source_run_id}` : ""}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => removeEntry(e.key)}
                  disabled={importing}
                  className="text-muted-foreground hover:text-foreground disabled:opacity-40"
                  title="Remove"
                  aria-label="Remove"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center justify-end gap-3">
        {entries.length > 0 && (
          <Button variant="ghost" onClick={() => setEntries([])} disabled={importing}>
            Clear
          </Button>
        )}
        <Button onClick={onImportAll} disabled={importing || importable.length === 0}>
          {importing ? <Loader2 className="h-4 w-4 animate-spin" /> : <PackageOpen className="h-4 w-4" />}
          {importing
            ? "Importing…"
            : importable.length > 1
              ? `Import ${importable.length} runs`
              : "Import run"}
        </Button>
      </div>
    </div>
  );
}

function StatusIcon({ status }: { status: Entry["status"] }) {
  if (status === "importing") return <Loader2 className="h-4 w-4 shrink-0 animate-spin text-muted-foreground" />;
  if (status === "done") return <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-500" />;
  if (status === "failed") return <XCircle className="h-4 w-4 shrink-0 text-destructive" />;
  return <PackageOpen className="h-4 w-4 shrink-0 text-muted-foreground" />;
}
