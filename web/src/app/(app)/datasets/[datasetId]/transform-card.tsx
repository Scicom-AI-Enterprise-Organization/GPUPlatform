"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowRight, Loader2, Wand2, X } from "lucide-react";
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
import { ProgressEta } from "@/components/progress-eta";
import { SearchableSelect, type SearchableOption } from "@/components/ui/searchable-select";
import { gateway, GatewayError } from "@/lib/gateway";
import type { DatasetKind, DatasetRecord, StorageRecord, TransformDatasetRequest } from "@/lib/types";

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
  kind,
  hfRepo,
  s3Storages,
  initialStatus,
  initialLog,
  bare = false,
}: {
  datasetId: string;
  kind: DatasetKind;
  hfRepo: string | null;
  s3Storages: StorageRecord[];
  initialStatus: string | null;
  initialLog: string | null;
  bare?: boolean;
}) {
  const isLabel = kind === "label";
  const router = useRouter();
  const [target, setTarget] = useState<"hf" | "s3">("hf");
  const [outRepo, setOutRepo] = useState(hfRepo ? `${hfRepo}-audio` : "");
  const [storageId, setStorageId] = useState(s3Storages[0]?.id ?? "");
  const [s3Folder, setS3Folder] = useState(`datasets/${datasetId}/transformed`);
  const [testSplitOn, setTestSplitOn] = useState(false);
  const [testSplitMode, setTestSplitMode] = useState<"pct" | "count" | "ref">("pct");
  const [testSplitValue, setTestSplitValue] = useState("10");
  // "From dataset" mode: reuse another dataset's exact test set (no overlap when
  // this dataset is a superset of it). The candidate list is fetched client-side.
  const [refDatasetId, setRefDatasetId] = useState("");
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  useEffect(() => {
    gateway.listDatasets().then(setDatasets).catch(() => {});
  }, []);
  // Only datasets that can yield a test split: S3/exported (own `split` column) or
  // label (resolved server-side to its exported S3 twin). Never this dataset.
  const refOptions = useMemo<SearchableOption[]>(
    () =>
      datasets
        .filter((d) => d.id !== datasetId && (d.kind === "s3" || d.kind === "upload" || d.kind === "label"))
        .map((d) => ({ value: d.id, label: d.name, hint: d.kind, group: d.kind })),
    [datasets, datasetId],
  );
  const refName = datasets.find((d) => d.id === refDatasetId)?.name;
  // Min transcription length (chars) for a row to be eligible for the test split.
  // Blank/0 → no minimum. Keeps junk transcripts ("[silent]") out of eval.
  const [testMinChars, setTestMinChars] = useState("");
  // Regex; transcripts matching it are excluded from the test split (kept in train).
  // e.g. ^\s*\[.*\]\s*$ drops bracketed placeholder tags like [silent].
  const [testExcludeRegex, setTestExcludeRegex] = useState("");
  const regexError = useMemo(() => {
    if (!testExcludeRegex.trim()) return null;
    try {
      new RegExp(testExcludeRegex);
      return null;
    } catch (e) {
      return e instanceof Error ? e.message : String(e);
    }
  }, [testExcludeRegex]);
  const [status, setStatus] = useState<string | null>(initialStatus);
  const [log, setLog] = useState<string | null>(initialLog);
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const poll = useRef<ReturnType<typeof setInterval> | null>(null);
  const logRef = useRef<HTMLPreElement | null>(null);

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
    let testSplit: Pick<
      TransformDatasetRequest,
      "test_split_pct" | "test_split_count" | "test_min_chars" | "test_exclude_regex" | "test_split_ref_dataset_id"
    > = {};
    if (testSplitOn && testSplitMode === "ref") {
      if (!refDatasetId) {
        setErr("Pick a dataset to reuse the test split from.");
        return;
      }
      testSplit = { test_split_ref_dataset_id: refDatasetId };
    } else if (testSplitOn) {
      const v = Number(testSplitValue);
      if (!Number.isFinite(v) || v < 0) {
        setErr("Enter a valid test-split size.");
        return;
      }
      if (testSplitMode === "pct") {
        if (v >= 100) {
          setErr("Test split percentage must be below 100.");
          return;
        }
        testSplit = { test_split_pct: v };
      } else {
        testSplit = { test_split_count: Math.round(v) };
      }
      const mc = Number(testMinChars);
      if (testMinChars.trim()) {
        if (!Number.isFinite(mc) || mc < 0) {
          setErr("Enter a valid minimum transcription length.");
          return;
        }
        if (mc > 0) testSplit.test_min_chars = Math.round(mc);
      }
      if (testExcludeRegex.trim()) {
        if (regexError) {
          setErr(`Fix the test-exclude regex: ${regexError}`);
          return;
        }
        testSplit.test_exclude_regex = testExcludeRegex.trim();
      }
    }
    setStarting(true);
    try {
      const d = await gateway.transformDataset(
        datasetId,
        target === "hf"
          ? { target: "hf", hf_repo: outRepo.trim(), ...testSplit }
          : { target: "s3", storage_id: storageId, s3_folder: s3Folder.trim() || null, ...testSplit },
      );
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

  // Auto-scroll the live log to the newest line while running.
  useEffect(() => {
    if (running && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log, running]);

  const desc = (
    <span className="text-xs text-muted-foreground">
      {isLabel ? (
        <>
          Export this labeling-platform project&apos;s reviewed tasks (per the dataset&apos;s status filter),
          download each clip, and build a dataset with a real <span className="font-mono">audio</span> column +
          its <span className="font-mono">transcription</span>. Runs on the gateway; watch progress below.
        </>
      ) : (
        <>
          This repo stores audio in archives, so there&apos;s no audio column to preview. Unzip it and rebuild a
          dataset with a real audio column (joined on the <span className="font-mono">audio</span> column you set
          above). Runs on the gateway; watch progress below.
        </>
      )}
    </span>
  );
  const body = (
    <div className="space-y-3">
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

        <div className="space-y-2 rounded-md border border-border p-3">
          <label className="flex items-center gap-2 text-xs font-medium">
            <input
              type="checkbox"
              className="h-4 w-4 accent-primary"
              checked={testSplitOn}
              onChange={(e) => setTestSplitOn(e.target.checked)}
              disabled={running}
            />
            Add a held-out test split
          </label>
          {testSplitOn && (
            <div className="space-y-2 pl-6">
              <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
                {(["pct", "count", "ref"] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setTestSplitMode(m)}
                    disabled={running}
                    className={
                      "rounded px-2.5 py-1 transition-colors " +
                      (testSplitMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")
                    }
                  >
                    {m === "pct" ? "% of rows" : m === "count" ? "# rows" : "From dataset"}
                  </button>
                ))}
              </div>

              {testSplitMode === "ref" ? (
                <div className="space-y-2">
                  <SearchableSelect
                    value={refDatasetId}
                    onChange={setRefDatasetId}
                    options={refOptions}
                    placeholder="Choose a dataset with a test split"
                    searchPlaceholder="Search datasets…"
                    className="text-xs"
                  />
                  <p className="text-xs text-muted-foreground">
                    Reuses{" "}
                    {refName ? (
                      <span className="font-medium text-foreground">{refName}</span>
                    ) : (
                      "the selected dataset"
                    )}
                    &apos;s exact <span className="font-mono">test</span> split as this dataset&apos;s test set
                    (matched by audio filename); every other row becomes{" "}
                    <span className="font-mono">train</span>. Guarantees no train/test overlap — ideal when this
                    dataset just adds more rows on top of that one. A <span className="font-mono">label</span>{" "}
                    dataset resolves to its exported S3 version.
                  </p>
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-2">
                    <Input
                      type="number"
                      min={0}
                      value={testSplitValue}
                      onChange={(e) => setTestSplitValue(e.target.value)}
                      disabled={running}
                      className="h-8 w-24 text-xs"
                    />
                    <span className="text-xs text-muted-foreground">
                      {testSplitMode === "pct" ? "% of rows" : "rows"}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Randomly holds out{" "}
                    {testSplitMode === "pct" ? `${testSplitValue || "0"}%` : `${testSplitValue || "0"} row(s)`} as a{" "}
                    <span className="font-mono">test</span> split; the rest become{" "}
                    <span className="font-mono">train</span>. Any source splits are collapsed.
                  </p>
                  <div className="flex items-center gap-2 pt-1">
                    <label className="text-xs text-muted-foreground">Min transcription length</label>
                    <Input
                      type="number"
                      min={0}
                      placeholder="0"
                      value={testMinChars}
                      onChange={(e) => setTestMinChars(e.target.value)}
                      disabled={running}
                      className="h-8 w-20 text-xs"
                    />
                    <span className="text-xs text-muted-foreground">chars</span>
                  </div>
                  {testMinChars.trim() && Number(testMinChars) > 0 && (
                    <p className="text-xs text-muted-foreground">
                      Only rows whose transcription is ≥ {Math.round(Number(testMinChars))} characters are
                      eligible for <span className="font-mono">test</span> — shorter transcripts stay in{" "}
                      <span className="font-mono">train</span>.
                    </p>
                  )}
                  <div className="space-y-1 pt-1">
                    <div className="flex items-center gap-2">
                      <label className="text-xs text-muted-foreground">Exclude from test (regex)</label>
                      <button
                        type="button"
                        disabled={running}
                        onClick={() => setTestExcludeRegex("^\\s*\\[.*\\]\\s*$")}
                        className="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted-foreground hover:text-foreground"
                        title="Match transcripts that are entirely a [bracketed] tag"
                      >
                        [bracket] tags
                      </button>
                    </div>
                    <Input
                      value={testExcludeRegex}
                      onChange={(e) => setTestExcludeRegex(e.target.value)}
                      disabled={running}
                      placeholder="^\s*\[.*\]\s*$"
                      className="h-8 font-mono text-xs"
                    />
                    {regexError ? (
                      <p className="text-xs text-destructive">Invalid regex: {regexError}</p>
                    ) : (
                      testExcludeRegex.trim() && (
                        <p className="text-xs text-muted-foreground">
                          Transcripts matching <span className="font-mono">/{testExcludeRegex}/</span> stay in{" "}
                          <span className="font-mono">train</span> — e.g. <span className="font-mono">[silent]</span>,{" "}
                          <span className="font-mono">[unintelligible]</span> are kept out of{" "}
                          <span className="font-mono">test</span>.
                        </p>
                      )
                    )}
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {err && <p className="text-sm text-destructive">{err}</p>}

        <div className="flex items-center gap-3">
          <Button onClick={run} disabled={running || starting}>
            {running || starting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Wand2 className="h-4 w-4" />}
            {running ? "Transforming…" : "Run transform"}
          </Button>
          {running && (
            <Button variant="outline" onClick={cancel} disabled={cancelling} className="text-destructive">
              {cancelling ? <Loader2 className="h-4 w-4 animate-spin" /> : <X className="h-4 w-4" />}
              {cancelling ? "Cancelling…" : "Cancel"}
            </Button>
          )}
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
    </div>
  );

  if (bare) return <div className="space-y-3">{desc}{body}</div>;
  return (
    <Card>
      <CardHeader className="flex flex-col gap-0.5">
        <CardTitle className="text-base">
          {isLabel ? "Transform — export labels to a dataset" : "Transform — extract audio column"}
        </CardTitle>
        {desc}
      </CardHeader>
      <CardContent>{body}</CardContent>
    </Card>
  );
}
