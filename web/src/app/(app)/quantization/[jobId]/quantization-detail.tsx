"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Check, Download, ExternalLink, Loader2, Pencil, Trash2, UploadCloud, X, XCircle } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { gateway } from "@/lib/gateway";
import { formatCostUSD, formatRateUSD, useLiveCost } from "@/lib/cost";
import { BurnFlame } from "@/components/burn-flame";
import { JsonView } from "@/components/json-view";
import { cn } from "@/lib/utils";
import type { QuantizationJobRecord, StorageRecord, TrainingFile } from "@/lib/types";

// Mirror training-detail's status badge styles.
const STATUS_STYLES: Record<string, string> = {
  queued: "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400",
  running: "border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400",
  done: "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
  failed: "border-destructive/40 bg-destructive/10 text-destructive",
  cancelled: "border-border bg-muted text-muted-foreground",
};

type ConfirmOpts = {
  title: string;
  description: string;
  confirmLabel: string;
  busyLabel: string;
  destructive?: boolean;
  run: () => Promise<void>;
};

export function QuantizationDetail({ initial }: { initial: QuantizationJobRecord }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [job, setJob] = useState<QuantizationJobRecord>(initial);
  const [busy, setBusy] = useState(false);
  const [confirmOpts, setConfirmOpts] = useState<ConfirmOpts | null>(null);
  const [confirmError, setConfirmError] = useState<string | null>(null);

  // Inline rename (mirrors training-detail).
  const [editingName, setEditingName] = useState(false);
  const [nameDraft, setNameDraft] = useState(job.name);
  const [renameError, setRenameError] = useState<string | null>(null);

  const terminal = job.status === "done" || job.status === "failed" || job.status === "cancelled";
  const exporting = job.result_json?.hf_export?.status === "running";
  const result = job.result_json;
  const canExport = job.status === "done" && !!result?.artifact;

  const tab = searchParams.get("tab") || "overview";
  // Each tab trigger is a real <Link> (right/middle/⌘-click opens in a new tab);
  // a plain click still switches in place because `tab` derives from the URL.
  const tabHref = (v: string) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("tab", v);
    return `${pathname}?${params.toString()}`;
  };

  // Poll the job while non-terminal (or while an HF export runs) so status +
  // result_json (artifact, sizes, progress, hf_export) stay live.
  useEffect(() => {
    if (terminal && !exporting) return;
    const t = setInterval(async () => {
      try {
        setJob(await gateway.getQuantizationJob(job.id));
      } catch {
        /* keep last */
      }
    }, 5000);
    return () => clearInterval(t);
  }, [job.id, terminal, exporting]);

  // Live log stream (SSE) — mirrors training-detail.
  const [lines, setLines] = useState<string[]>([]);
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLines([]);
    const es = new EventSource(gateway.quantizationLogsStreamUrl(job.id));
    // The server replays the whole Redis log list on every (re)connect. onopen
    // fires on the initial connect AND every transparent reconnect, so clear
    // here — otherwise a network blip doubles every line already shown.
    es.onopen = () => setLines([]);
    es.onmessage = (ev) => setLines((p) => [...p, ev.data as string]);
    es.addEventListener("end", () => es.close());
    return () => es.close();
  }, [job.id]);

  const onRename = async () => {
    const name = nameDraft.trim();
    if (!name || name === job.name) {
      setEditingName(false);
      return;
    }
    setRenameError(null);
    try {
      setJob(await gateway.renameQuantizationJob(job.id, name));
      setEditingName(false);
    } catch (e) {
      setRenameError(e instanceof Error ? e.message : String(e));
    }
  };

  const onTerminate = () =>
    setConfirmOpts({
      title: `Terminate ${job.name}?`,
      description: "Stops the quantization job. Nothing is saved.",
      confirmLabel: "Terminate",
      busyLabel: "Terminating…",
      destructive: true,
      run: async () => {
        setJob(await gateway.terminateQuantizationJob(job.id));
        toast.success("Job terminated");
      },
    });

  const onDelete = () =>
    setConfirmOpts({
      title: `Delete ${job.name}?`,
      description:
        "Removes the job record. S3 artifacts are kept. If a RunPod pod is still alive it is terminated.",
      confirmLabel: "Delete",
      busyLabel: "Deleting…",
      destructive: true,
      run: async () => {
        await gateway.deleteQuantizationJob(job.id);
        toast.success("Job deleted");
        router.push("/quantization");
      },
    });

  const runConfirm = async () => {
    if (!confirmOpts) return;
    setConfirmError(null);
    setBusy(true);
    try {
      await confirmOpts.run();
      setConfirmOpts(null);
    } catch (e) {
      setConfirmError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const progress = result?.progress;
  const quantGb = result?.sizes?.quantized_gb;

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="border-b border-border bg-sidebar/40 px-6 pt-4 lg:px-10">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              {editingName ? (
                <span className="flex flex-col gap-1">
                  <span className="flex items-center gap-1">
                    <Input
                      value={nameDraft}
                      onChange={(e) => setNameDraft(e.target.value)}
                      autoFocus
                      className="h-8 w-72 text-lg font-semibold"
                      onKeyDown={(e) => {
                        if (e.key === "Enter") onRename();
                        if (e.key === "Escape") { setEditingName(false); setRenameError(null); }
                      }}
                    />
                    <Button size="icon" variant="ghost" className="h-7 w-7" onClick={onRename} title="Save">
                      <Check className="h-4 w-4" />
                    </Button>
                    <Button size="icon" variant="ghost" className="h-7 w-7" onClick={() => { setEditingName(false); setRenameError(null); }} title="Cancel">
                      <X className="h-4 w-4" />
                    </Button>
                  </span>
                  {renameError && <span className="text-xs text-destructive">{renameError}</span>}
                </span>
              ) : (
                <>
                  <h1 className="truncate text-2xl font-semibold tracking-tight">{job.name}</h1>
                  <button
                    type="button"
                    onClick={() => { setNameDraft(job.name); setEditingName(true); }}
                    className="text-muted-foreground hover:text-foreground"
                    title="Rename"
                  >
                    <Pencil className="h-4 w-4" />
                  </button>
                </>
              )}
              {exporting ? (
                <Badge variant="outline" className={STATUS_STYLES.running}>
                  <Loader2 className="mr-1 h-3 w-3 animate-spin" /> pushing to HF
                </Badge>
              ) : (
                <Badge variant="outline" className={STATUS_STYLES[job.status] ?? ""}>{job.status}</Badge>
              )}
              {!exporting && job.status === "running" && progress?.stage && (
                <span className="flex items-center gap-1.5 text-xs font-medium text-amber-600 dark:text-amber-400">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  {progress.stage}{typeof progress.percent === "number" ? ` · ${Math.round(progress.percent)}%` : ""}
                </span>
              )}
              <Badge variant="outline" className="border-violet-500/40 bg-violet-500/10 text-violet-600 dark:text-violet-300">
                {job.scheme}
              </Badge>
            </div>
            <p className="mt-1 font-mono text-xs text-muted-foreground">
              {job.source_model} · {job.id}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {!terminal && (
              <Button variant="outline" size="sm" onClick={onTerminate} disabled={busy} className="text-destructive">
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <XCircle className="h-4 w-4" />} Terminate
              </Button>
            )}
            <Button variant="outline" size="sm" onClick={onDelete} disabled={busy}>
              <Trash2 className="h-4 w-4" /> Delete
            </Button>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-5">
          <Kpi label="Status" value={exporting ? "pushing to HF" : job.status} />
          <Kpi
            label="GPU"
            value={job.gpu_type ? `${job.gpu_type}${job.gpu_count > 1 ? ` ×${job.gpu_count}` : ""}` : "—"}
          />
          <CostKpi job={job} />
          <Kpi label="Quantized size" value={quantGb != null ? `${quantGb} GB` : "—"} />
          <Kpi label="Scheme" value={job.scheme} />
        </div>

        <Tabs value={tab} className="mt-4">
          <TabsList variant="line" className="bg-transparent">
            <TabsTrigger value="overview" asChild><Link href={tabHref("overview")} scroll={false}>Overview</Link></TabsTrigger>
            <TabsTrigger value="logs" asChild><Link href={tabHref("logs")} scroll={false}>Logs</Link></TabsTrigger>
            <TabsTrigger value="files" asChild><Link href={tabHref("files")} scroll={false}>Files</Link></TabsTrigger>
            <TabsTrigger value="config" asChild><Link href={tabHref("config")} scroll={false}>Config</Link></TabsTrigger>
            {canExport && <TabsTrigger value="hf" asChild><Link href={tabHref("hf")} scroll={false}>Export to HF</Link></TabsTrigger>}
          </TabsList>
        </Tabs>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        {job.error_text && job.status === "failed" && (
          <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            <pre className="whitespace-pre-wrap break-words font-mono text-xs">{job.error_text}</pre>
          </div>
        )}

        <Tabs value={tab} className="!block">
          <TabsContent value="overview" className="!flex-none space-y-4">
            {!terminal && (
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Progress</CardTitle></CardHeader>
                <CardContent>
                  <div className="mb-1.5 flex items-center justify-between text-xs">
                    <span className="font-medium text-foreground">{progress?.stage ?? "starting"}</span>
                    <span className="tabular-nums text-muted-foreground">{Math.round(progress?.percent ?? 0)}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full bg-primary transition-all"
                      style={{ width: `${Math.min(100, Math.max(0, progress?.percent ?? 0))}%` }}
                    />
                  </div>
                </CardContent>
              </Card>
            )}

            {job.status === "done" && (
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Quantized model</CardTitle></CardHeader>
                <CardContent className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
                  <Stat label="Scheme" value={job.scheme} />
                  {quantGb != null && <Stat label="Quantized size" value={`${quantGb} GB`} />}
                  {result?.artifact && <Stat label="Artifact" value={result.artifact} mono />}
                  {result?.hf_repo && (
                    <Stat label="HF" value={result.hf_repo} mono href={`https://huggingface.co/${result.hf_repo}`} />
                  )}
                </CardContent>
              </Card>
            )}

            {result?.hf_export && (
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Hugging Face export</CardTitle></CardHeader>
                <CardContent className="text-sm">
                  {result.hf_export.status === "running" && (
                    <span className="flex items-center gap-1.5 text-amber-600 dark:text-amber-400">
                      <Loader2 className="h-3.5 w-3.5 animate-spin" /> Pushing to {result.hf_export.repo}…
                    </span>
                  )}
                  {result.hf_export.status === "done" && (
                    <a
                      href={result.hf_export.url ?? `https://huggingface.co/${result.hf_export.repo}`}
                      target="_blank" rel="noreferrer"
                      className="inline-flex items-center gap-1.5 font-medium text-primary hover:underline"
                    >
                      {result.hf_export.repo} <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                  )}
                  {result.hf_export.status === "failed" && (
                    <span className="text-destructive">Export failed: {result.hf_export.error}</span>
                  )}
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="logs" className="!flex-none">
            <LogsPanel lines={lines} status={job.status} />
          </TabsContent>

          <TabsContent value="files" className="!flex-none">
            <FilesTab job={job} />
          </TabsContent>

          <TabsContent value="config" className="!flex-none space-y-4">
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Compute</CardTitle></CardHeader>
              <CardContent className="flex flex-wrap gap-x-8 gap-y-3 text-sm">
                <Stat
                  label={job.provider_kind === "vm" ? "VM" : "Provider"}
                  value={
                    job.provider_name
                      ? `${job.provider_name}${job.provider_kind ? ` (${job.provider_kind})` : ""}`
                      : job.provider_id || "—"
                  }
                  href={job.provider_id ? "/providers" : undefined}
                />
                <Stat
                  label="GPU"
                  value={job.gpu_type ? `${job.gpu_type}${job.gpu_count > 1 ? ` × ${job.gpu_count}` : ""}` : "—"}
                />
                {job.visible_devices && <Stat label="GPU ids" value={job.visible_devices} mono />}
                <Stat label="Storage" value={job.storage_name || job.storage_id || "—"}
                  href={job.storage_id ? "/storage" : undefined} />
                {job.calibration_dataset_id && (
                  <Stat label="Calibration dataset" value={job.calibration_dataset_id} mono
                    href={`/datasets/${job.calibration_dataset_id}`} />
                )}
                <Stat label="Source model" value={job.source_model} mono />
              </CardContent>
            </Card>
            <ConfigJson config={job.config_json} />
          </TabsContent>

          {canExport && (
            <TabsContent value="hf" className="!flex-none">
              <HfExportPanel job={job} onStarted={(j) => setJob(j)} />
            </TabsContent>
          )}
        </Tabs>
      </div>

      <Dialog
        open={!!confirmOpts}
        onOpenChange={(o) => {
          if (!busy && !o) {
            setConfirmOpts(null);
            setConfirmError(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{confirmOpts?.title}</DialogTitle>
            <DialogDescription>{confirmOpts?.description}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {confirmError && <p className="mr-auto text-sm text-destructive">{confirmError}</p>}
            <Button variant="outline" onClick={() => setConfirmOpts(null)} disabled={busy}>
              Cancel
            </Button>
            <Button variant={confirmOpts?.destructive ? "destructive" : "default"} onClick={runConfirm} disabled={busy}>
              {busy ? confirmOpts?.busyLabel : confirmOpts?.confirmLabel}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// ---- HF export tab (mirrors autotrain's hf-export-tab layout) ---------------

function HfExportPanel({
  job,
  onStarted,
}: {
  job: QuantizationJobRecord;
  onStarted: (j: QuantizationJobRecord) => void;
}) {
  const cfg = (job.config_json ?? {}) as Record<string, unknown>;
  const [storages, setStorages] = useState<StorageRecord[]>([]);
  const [storageId, setStorageId] = useState("");
  const [repo, setRepo] = useState(
    job.result_json?.hf_repo
      ?? (typeof cfg.hf_push_repo === "string" ? (cfg.hf_push_repo as string) : ""),
  );
  const [isPrivate, setIsPrivate] = useState(true); // default private
  // Where the push runs: "gateway" (default) fetches from S3 and pushes here — no
  // dependency on the job's box; "vm" pushes from the VM this job quantized on
  // (needs that box + the quant venv). Only offered for VM jobs.
  const isVmJob = job.provider_kind === "vm";
  const [runOn, setRunOn] = useState<"gateway" | "vm">("gateway");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);
  const state = job.result_json?.hf_export;
  const running = state?.status === "running";

  useEffect(() => {
    gateway
      .listStorage()
      .then((rows) => setStorages(rows.filter((s) => s.kind === "huggingface")))
      .catch(() => {});
  }, []);

  const submit = async () => {
    setErr(null);
    if (!repo.trim()) return setErr("Enter a repo name (org/model-name).");
    setBusy(true);
    try {
      const updated = await gateway.exportQuantizationToHuggingface(job.id, {
        repo: repo.trim(),
        storageId: storageId || null,
        private: isPrivate,
        runOn,
      });
      onStarted(updated);
      toast.success("Export started");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setStopping(true);
    try {
      onStarted(await gateway.cancelQuantizationHfExport(job.id));
    } catch {
      // best-effort; the next poll reflects the real state
    } finally {
      setStopping(false);
    }
  };

  return (
    <div className="space-y-5">
      <p className="text-sm text-muted-foreground">
        Push the <span className="font-medium">compressed model</span> to a Hugging Face model repo.
        Runs on the gateway (no GPU needed) — the compressed-tensors format loads directly in vLLM.
      </p>

      <ExportSection title="Destination" description="The Hugging Face repo and the token used to push to it.">
        <div className="space-y-3">
          <div className="space-y-1.5">
            <Label className="text-xs uppercase tracking-wide text-muted-foreground">Repo name</Label>
            <Input className="font-mono" value={repo} placeholder="org/model-name"
              onChange={(e) => setRepo(e.target.value)} />
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs uppercase tracking-wide text-muted-foreground">Hugging Face storage</Label>
            <Select value={storageId} onValueChange={setStorageId}>
              <SelectTrigger>
                <SelectValue placeholder={storages.length ? "Choose a HuggingFace storage" : "None configured — platform HF_TOKEN used"} />
              </SelectTrigger>
              <SelectContent>
                {storages.map((s) => (
                  <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-[11px] text-muted-foreground">
              Provides the push token.{storages.length === 0 ? " None configured — the platform HF_TOKEN secret is used, if set." : ""}
            </p>
          </div>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input type="checkbox" checked={isPrivate} onChange={(e) => setIsPrivate(e.target.checked)}
              className="h-4 w-4 accent-primary" />
            <span>Private repo</span>
          </label>
        </div>
      </ExportSection>

      {/* Run on — VM jobs only. No GPU needed (the artifact is a complete compressed
          model), so the choice is just gateway (box-independent) vs the job's VM. */}
      {isVmJob && (
        <ExportSection title="Run on" description="Where the push runs. The artifact is already a complete model — no GPU required.">
          <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
            {([
              ["gateway", "Gateway (recommended)"],
              ["vm", "Job's VM"],
            ] as const).map(([val, label]) => (
              <button
                key={val}
                type="button"
                onClick={() => setRunOn(val)}
                className={`rounded px-2.5 py-1 transition-colors ${
                  runOn === val ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <p className="mt-2 text-[11px] text-muted-foreground">
            {runOn === "gateway"
              ? "Pushes from the gateway — the model is fetched from S3 and uploaded here. Works even if the job's box (and its uv venv) is gone."
              : "Pushes from the box this job quantized on. Requires that VM — and the quant uv venv — to still exist."}
          </p>
        </ExportSection>
      )}

      <div className="flex items-center justify-end gap-3">
        {err && <p className="mr-auto text-sm text-destructive">{err}</p>}
        {running && (
          <Button variant="destructive" onClick={stop} disabled={stopping}>
            {stopping ? <Loader2 className="h-4 w-4 animate-spin" /> : <X className="h-4 w-4" />}
            Cancel export
          </Button>
        )}
        <Button onClick={submit} disabled={busy || running || !repo.trim()}>
          {(busy || running) ? <Loader2 className="h-4 w-4 animate-spin" /> : <UploadCloud className="h-4 w-4" />}
          {running ? "Exporting…" : "Push to HF"}
        </Button>
      </div>

      {/* Export status — pushing (while running), a link when done, or a failed/cancelled note. */}
      {state && (
        <section className="rounded-lg border border-border bg-card p-5">
          <h2 className="mb-3 text-sm font-semibold">Hugging Face export</h2>
          <div className="flex flex-wrap items-center gap-x-8 gap-y-2 text-sm">
            {state.status === "running" && (
              <span className="flex items-center gap-1.5 text-amber-600 dark:text-amber-400">
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> pushing {state.repo} …
              </span>
            )}
            {state.status === "cancelled" && (
              <span className="flex items-center gap-x-3 text-muted-foreground">
                push stopped{state.error ? ` — ${state.error}` : ""}
              </span>
            )}
            {state.status === "done" && (
              <a
                href={state.url ?? `https://huggingface.co/${state.repo}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 font-medium text-primary hover:underline"
              >
                Open on Hugging Face — {state.repo}
                <ExternalLink className="h-3.5 w-3.5" />
              </a>
            )}
            {state.status === "failed" && (
              <span className="text-destructive">push failed: {state.error}</span>
            )}
          </div>
        </section>
      )}
    </div>
  );
}

// Card section matching autotrain's hf-export-tab Section.
function ExportSection({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-border bg-card p-5">
      <div className="mb-4">
        <h2 className="text-sm font-semibold">{title}</h2>
        {description && <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>}
      </div>
      {children}
    </section>
  );
}

// ---- Files (mirrors training-detail's FilesTab) -----------------------------

function fmtBytes(n?: number | null): string {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  const u = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${u[i]}`;
}

function FilesTab({ job }: { job: QuantizationJobRecord }) {
  const [files, setFiles] = useState<TrainingFile[]>([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    gateway
      .listQuantizationFiles(job.id)
      .then((f) => { if (!cancelled) setFiles(f); })
      .catch(() => { if (!cancelled) setFiles([]); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [job.id]);

  if (loading) return <p className="text-sm text-muted-foreground">Loading files…</p>;
  if (files.length === 0)
    return <p className="text-sm text-muted-foreground">No files yet — the compressed model uploads when the job finishes.</p>;
  return (
    <ul className="divide-y divide-border rounded-md border border-border">
      {files.map((f) => (
        <li key={f.name} className="flex items-center justify-between gap-4 px-4 py-2 text-sm">
          <span className="truncate font-mono text-xs">{f.name}</span>
          <a href={f.download_url} target="_blank" rel="noreferrer"
            className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
            <Download className="h-3.5 w-3.5" /> {fmtBytes(f.size)}
          </a>
        </li>
      ))}
    </ul>
  );
}

// ---- Logs (mirrors training-detail's LogsTab, minus the trim button) -------

function LogsPanel({ lines, status }: { lines: string[]; status: string }) {
  const endRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const terminal = ["done", "failed", "cancelled"].includes(status);

  useEffect(() => {
    if (autoScroll) endRef.current?.scrollIntoView({ block: "end" });
  }, [lines, autoScroll]);

  return (
    <div className="space-y-2">
      <label className="flex w-fit cursor-pointer select-none items-center gap-2 text-xs text-muted-foreground">
        <input
          type="checkbox"
          checked={autoScroll}
          onChange={(e) => setAutoScroll(e.target.checked)}
          className="h-3.5 w-3.5 accent-primary"
        />
        Auto-scroll to latest{!autoScroll && lines.length > 0 ? " (paused)" : ""}
      </label>
      <div className="terminal-block h-[55vh] overflow-y-auto rounded-md border border-border bg-zinc-950 p-3 font-mono text-xs leading-relaxed text-zinc-200">
        {lines.length === 0 ? (
          <div className="text-zinc-500">
            {terminal
              ? `No logs (job ${status}).`
              : status === "queued" ? "Queued — waiting for the runner…" : "Waiting for output…"}
          </div>
        ) : (
          lines.map((l, i) => (
            <div key={i} className={
              l.startsWith("@@") ? "text-sky-300"
                : l.startsWith("[gateway]") ? "text-emerald-300"
                : "text-zinc-200"
            }>{l}</div>
          ))
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}

// ---- Config JSON (mirrors training-detail's ConfigJson) ---------------------

function pruneEmpty(v: unknown): unknown {
  if (v === null || v === undefined || v === "") return undefined;
  if (Array.isArray(v)) return v.length ? v : undefined;
  if (typeof v === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
      const p = pruneEmpty(val);
      if (p !== undefined) out[k] = p;
    }
    return Object.keys(out).length ? out : undefined;
  }
  return v;
}

function ConfigJson({ config }: { config: unknown }) {
  const [raw, setRaw] = useState(false);
  const pruned = useMemo(() => pruneEmpty(config) ?? {}, [config]);
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm">Configuration</CardTitle>
        <button
          type="button"
          onClick={() => setRaw((r) => !r)}
          className="text-[11px] text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
        >
          {raw ? "Show set values only" : "Show all fields (raw)"}
        </button>
      </CardHeader>
      <CardContent>
        <JsonView value={raw ? config : pruned} />
      </CardContent>
    </Card>
  );
}

// ---- Small helpers (mirror training-detail) ---------------------------------

function Stat({ label, value, mono, href }: { label: string; value: string; mono?: boolean; href?: string }) {
  const cls = `text-sm ${mono ? "font-mono break-all" : "font-medium"}`;
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      {href ? (
        <a href={href} target="_blank" rel="noreferrer" className={`${cls} text-primary hover:underline`}>{value}</a>
      ) : (
        <div className={cls}>{value}</div>
      )}
    </div>
  );
}

// Header-band KPI cell (matches the autotrain / benchmark detail headers).
function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-0.5 truncate text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}

// Live-ticking spend while running; final total once the job ends. VM jobs have
// no hourly rate → shows "—".
function CostKpi({ job }: { job: QuantizationJobRecord }) {
  const live = useLiveCost(job.started_at, job.ended_at, job.cost_per_hr);
  const isBurning = job.status === "running" && job.cost_per_hr != null && job.ended_at == null;
  return (
    <div>
      <div className="text-xs text-muted-foreground">Cost {isBurning ? "(live)" : ""}</div>
      <div
        className={cn(
          "mt-0.5 flex items-center gap-1.5 text-lg font-semibold tabular-nums",
          isBurning && "text-amber-600 dark:text-amber-400",
        )}
      >
        {isBurning && <BurnFlame size="h-4 w-4" />}
        {formatCostUSD(live)}
      </div>
      <div className="text-[10px] text-muted-foreground">
        {job.cost_per_hr != null ? `at ${formatRateUSD(job.cost_per_hr)}` : "—"}
      </div>
    </div>
  );
}
