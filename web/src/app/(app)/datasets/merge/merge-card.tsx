"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowRight, ChevronDown, GitMerge, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ProgressEta } from "@/components/progress-eta";
import { gateway, GatewayError } from "@/lib/gateway";
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

// Turn a dataset name into a safe S3 path segment: lowercase, non-alphanumerics
// (spaces, punctuation, non-ASCII) collapse to a single dash, trimmed. Empty if
// nothing survives (e.g. an all-CJK name) → caller falls back to the id default.
function slugify(s: string): string {
  return s
    .normalize("NFKD")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80)
    .replace(/-+$/g, "");
}

export function MergeCard({
  labelDatasets,
  s3Storages,
}: {
  labelDatasets: DatasetRecord[];
  s3Storages: StorageRecord[];
}) {
  const router = useRouter();
  const [selected, setSelected] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [target, setTarget] = useState<"hf" | "s3">("s3");
  const [outRepo, setOutRepo] = useState("");
  const [storageId, setStorageId] = useState(s3Storages[0]?.id ?? "");
  // S3 folder is derived from the (effective) dataset name; null until the user
  // manually overrides it in the field.
  const [folderOverride, setFolderOverride] = useState<string | null>(null);
  const [newId, setNewId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [log, setLog] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const poll = useRef<ReturnType<typeof setInterval> | null>(null);
  const logRef = useRef<HTMLPreElement | null>(null);

  const running = status === "running";

  // Poll the new dataset while the merge runs; refresh once it ends.
  useEffect(() => {
    if (!running || !newId) {
      if (poll.current) {
        clearInterval(poll.current);
        poll.current = null;
      }
      return;
    }
    const id = setInterval(async () => {
      try {
        const d = await gateway.getDataset(newId);
        setStatus(d.transform_status ?? null);
        setLog(d.transform_log ?? null);
        if (d.transform_status !== "running") router.refresh();
      } catch {
        /* transient; keep polling */
      }
    }, 3000);
    poll.current = id;
    return () => clearInterval(id);
  }, [running, newId, router]);

  useEffect(() => {
    if (running && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log, running]);

  const byId = new Map(labelDatasets.map((d) => [d.id, d]));
  const triggerLabel =
    selected.length === 0
      ? "Pick label datasets"
      : selected.length === 1
        ? byId.get(selected[0])?.name ?? selected[0]
        : `${selected.length} datasets`;

  // The name the merge will use: the typed name, else the auto "Merged: A + B"
  // (mirrors the gateway). The S3 folder is derived from it (slugified) unless
  // the user has overridden the field — blank slug → the gateway's id default.
  const autoName = selected.length
    ? `Merged: ${selected.map((id) => byId.get(id)?.name ?? id).join(" + ")}`
    : "";
  const effectiveName = name.trim() || autoName;
  const nameSlug = slugify(effectiveName);
  const derivedFolder = nameSlug ? `datasets/${nameSlug}/transformed` : "";
  const s3Folder = folderOverride ?? derivedFolder;

  async function run() {
    setErr(null);
    if (selected.length < 2) {
      setErr("Select at least 2 label datasets to merge.");
      return;
    }
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
      const d = await gateway.mergeDatasets({
        source_ids: selected,
        target,
        name: name.trim() || null,
        ...(target === "hf"
          ? { hf_repo: outRepo.trim() }
          : { storage_id: storageId, s3_folder: s3Folder.trim() || null }),
      });
      setNewId(d.id);
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

  const done = status === "done";
  const failed = !!status && status !== "running" && status !== "done";

  return (
    <div className="space-y-4">
      <Card>
      <CardHeader className="flex flex-col gap-0.5">
        <CardTitle className="text-base">Merge → one combined audio dataset</CardTitle>
        <span className="text-xs text-muted-foreground">
          Each selected label dataset&apos;s clips are downloaded and paired with their transcription,
          then concatenated and written as a single dataset. Runs on the gateway; watch progress below.
        </span>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {labelDatasets.length < 2 && (
            <p className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-400">
              You need at least 2 <span className="font-mono">label</span> datasets to merge. Import each
              labeling-platform project on{" "}
              <Link href="/datasets/new" className="font-medium underline underline-offset-2">
                New dataset
              </Link>{" "}
              (kind = Labeling platform) first.
            </p>
          )}

          <div className="space-y-1">
            <Label className="text-xs">Label datasets to merge</Label>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="outline"
                  disabled={running || labelDatasets.length === 0}
                  className="h-9 w-full justify-between px-3 text-xs font-normal sm:w-96"
                >
                  <span className="truncate">{triggerLabel}</span>
                  <ChevronDown className="ml-1 h-3.5 w-3.5 shrink-0 opacity-60" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="max-h-72 w-80 overflow-y-auto">
                {labelDatasets.map((d) => (
                  <DropdownMenuCheckboxItem
                    key={d.id}
                    checked={selected.includes(d.id)}
                    onSelect={(e) => e.preventDefault()}
                    onCheckedChange={(c) => {
                      setSelected((prev) => {
                        const next = c ? [...prev, d.id] : prev.filter((x) => x !== d.id);
                        // keep selection in list order (stable concat order)
                        return labelDatasets.map((x) => x.id).filter((x) => next.includes(x));
                      });
                    }}
                    className="text-xs"
                  >
                    <span className="truncate">{d.name}</span>
                    {typeof d.num_rows === "number" ? (
                      <span className="ml-1 text-muted-foreground">· {d.num_rows} rows</span>
                    ) : null}
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            {selected.length > 1 && (
              <p className="text-[11px] text-muted-foreground">
                {selected.length} datasets → concatenated into one dataset.
              </p>
            )}
          </div>

          <div className="space-y-1 sm:w-96">
            <Label htmlFor="mg-name" className="text-xs">Merged dataset name</Label>
            <Input
              id="mg-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="(auto: Merged: A + B)"
              disabled={running}
              className="text-xs"
            />
          </div>

          <div className="space-y-2">
            <Label className="text-xs">Write the combined dataset to</Label>
            <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
              {(["s3", "hf"] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setTarget(t)}
                  disabled={running}
                  className={
                    "rounded px-2.5 py-1 transition-colors " +
                    (target === t
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground")
                  }
                >
                  {t === "s3" ? "Materialize to S3" : "Push to HuggingFace"}
                </button>
              ))}
            </div>

            {target === "hf" ? (
              <div className="space-y-1 sm:w-96">
                <Label htmlFor="mg-repo" className="text-xs">Target HF repo</Label>
                <Input
                  id="mg-repo"
                  value={outRepo}
                  onChange={(e) => setOutRepo(e.target.value)}
                  placeholder="owner/combined-dataset"
                  disabled={running}
                  className="font-mono text-xs"
                />
              </div>
            ) : (
              <div className="space-y-3 sm:w-96">
                <div className="space-y-1">
                  <Label className="text-xs">S3 storage</Label>
                  <Select value={storageId} onValueChange={setStorageId} disabled={running}>
                    <SelectTrigger className="text-xs">
                      <SelectValue
                        placeholder={s3Storages.length ? "Choose an S3 storage" : "No S3 storage configured"}
                      />
                    </SelectTrigger>
                    <SelectContent>
                      {s3Storages.map((s) => (
                        <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="mg-folder" className="text-xs">Folder in S3</Label>
                  <Input
                    id="mg-folder"
                    value={s3Folder}
                    onChange={(e) => setFolderOverride(e.target.value)}
                    placeholder="datasets/<new-id>/transformed"
                    disabled={running}
                    className="font-mono text-xs"
                  />
                  <p className="text-xs text-muted-foreground">
                    Auto-derived from the dataset name (slugified); edit to override. Blank →{" "}
                    <span className="font-mono">datasets/&lt;new-id&gt;/transformed</span> under the storage
                    prefix. Audio → <span className="font-mono">…/audio/</span>, metadata →{" "}
                    <span className="font-mono">…/metadata.csv</span>.
                  </p>
                </div>
              </div>
            )}
          </div>

          {log && (
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                {running && <Loader2 className="h-3 w-3 animate-spin" />}
                <span>{running ? "Live log" : "Log"}</span>
                <ProgressEta log={log} running={running} />
              </div>
              <pre
                ref={logRef}
                className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-200 scrollbar-thin"
              >
                {log}
              </pre>
            </div>
          )}
          {/* failed state already shown via the status pill below */}
          {failed && !log && <p className="text-sm text-destructive">Merge {status}.</p>}
        </div>
      </CardContent>
      </Card>

      {/* Primary action lives outside the card, aligned right, at the bottom. */}
      <div className="flex items-center justify-end gap-3">
        {err && <span className="mr-auto text-sm text-destructive">{err}</span>}
        {status && status !== "running" && (
          <span className={done ? "text-sm text-emerald-600 dark:text-emerald-400" : "text-sm text-destructive"}>
            {done ? "✓ done" : `✕ ${status}`}
          </span>
        )}
        {newId && (
          <Link
            href={`/datasets/${newId}`}
            className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
          >
            Open merged dataset <span className="font-mono text-xs">{newId}</span>
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        )}
        <Button onClick={run} disabled={running || starting || labelDatasets.length < 2}>
          {running || starting ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitMerge className="h-4 w-4" />}
          {running ? "Merging…" : "Merge datasets"}
        </Button>
      </div>
    </div>
  );
}
