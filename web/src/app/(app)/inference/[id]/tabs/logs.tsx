"use client";

// Logs tab — a searchable, paginated browser over an endpoint's log files (one
// per launch session per source: each model's vLLM stdout + the worker-agent
// scheduler log). Built to scale to hundreds of files where the old dropdown
// couldn't. When the endpoint has an s3 log-archive storage set (chosen here),
// logs are persisted UNCAPPED and downloadable; with no storage they stay in
// Redis (5000-line cap, ~1h TTL) and this still lists the recent sessions.
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  AlertTriangle, ArrowDown, ArrowUp, ChevronDown, ChevronRight, Download,
  HardDrive, Loader2, RefreshCw, Search,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { gateway } from "@/lib/gateway";
import type { AppRecord, LogFile, StorageRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

const PAGE = 10;
const NONE = "__none__"; // Select sentinel: clear archival (Redis-only)
const ALL = "__all__"; // Select sentinel: no source filter

const fmtBytes = (n: number | null) =>
  n == null ? "—"
    : n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB`
    : n >= 1e6 ? `${(n / 1e6).toFixed(2)} MB`
    : n >= 1e3 ? `${(n / 1e3).toFixed(1)} KB`
    : `${n} B`;

const isWorker = (source: string) => source === "__worker__";
const typeLabel = (source: string) => (isWorker(source) ? "worker" : source);

export function LogsTab({ app }: { app: AppRecord }) {
  // --- archive storage config (lives on the app; editable here) ---
  const [storageId, setStorageId] = useState<string | null>(app.storage_id ?? null);
  const [storageName, setStorageName] = useState<string | null>(app.storage_name ?? null);
  const [storages, setStorages] = useState<StorageRecord[] | null>(null);
  const [savingStorage, setSavingStorage] = useState(false);
  const [storageErr, setStorageErr] = useState<string | null>(null);

  // --- file browser state ---
  const [files, setFiles] = useState<LogFile[]>([]);
  const [total, setTotal] = useState(0);
  const [archived, setArchived] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [source, setSource] = useState<string>(ALL);
  const [sort, setSort] = useState<"started_desc" | "started_asc">("started_desc");

  // Source options come from the configured members + the worker-agent log.
  const sourceOptions = useMemo(() => {
    const opts = [{ value: ALL, label: "All sources" }, { value: "__worker__", label: "worker-agent" }];
    for (const m of app.models ?? []) if (m.model) opts.push({ value: m.model, label: m.model });
    return opts;
  }, [app.models]);

  useEffect(() => {
    let cancelled = false;
    gateway.listStorage()
      .then((all) => { if (!cancelled) setStorages(all.filter((s) => s.kind === "s3")); })
      .catch(() => { if (!cancelled) setStorages([]); });
    return () => { cancelled = true; };
  }, []);

  const load = useCallback(async (offset: number, append: boolean) => {
    setLoading(true);
    setErr(null);
    try {
      const res = await gateway.listAppLogFiles(app.app_id, {
        source: source === ALL ? undefined : source,
        q: q.trim() || undefined,
        sort,
        limit: PAGE,
        offset,
      });
      setArchived(res.archived);
      setTotal(res.total);
      setFiles((prev) => (append ? [...prev, ...res.files] : res.files));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [app.app_id, source, q, sort]);

  // Refetch from the top whenever a filter changes (search is debounced).
  useEffect(() => {
    const t = window.setTimeout(() => { void load(0, false); }, 250);
    return () => window.clearTimeout(t);
  }, [load]);

  async function changeStorage(value: string) {
    const next = value === NONE ? null : value;
    setSavingStorage(true);
    setStorageErr(null);
    try {
      const rec = await gateway.setLogStorage(app.app_id, next);
      setStorageId(rec.storage_id ?? null);
      setStorageName(rec.storage_name ?? (next ? storages?.find((s) => s.id === next)?.name ?? null : null));
      // Newly enabled/disabled archival changes how files are sourced — reload.
      void load(0, false);
    } catch (e) {
      setStorageErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSavingStorage(false);
    }
  }

  const noStorages = storages !== null && storages.length === 0;

  return (
    <div className="space-y-4">
      {/* Archive storage config */}
      <Card>
        <CardContent className="flex flex-wrap items-center gap-x-4 gap-y-3 px-4 py-3 text-sm">
          <div className="flex items-center gap-2">
            <HardDrive className="h-4 w-4 text-muted-foreground" />
            <span className="font-medium">Log archive</span>
          </div>
          <Select
            value={storageId ?? NONE}
            onValueChange={changeStorage}
            disabled={savingStorage || storages === null}
          >
            <SelectTrigger size="sm" className="w-[260px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NONE}>None — keep in Redis (capped)</SelectItem>
              {(storages ?? []).map((s) => (
                <SelectItem key={s.id} value={s.id}>
                  {s.name}
                  {s.bucket ? <span className="text-muted-foreground"> · {s.bucket}</span> : null}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {savingStorage && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
          <p className="min-w-0 flex-1 text-xs text-muted-foreground">
            {storageId ? (
              <>Archiving every worker &amp; model log line to <span className="font-mono text-foreground">{storageName ?? storageId}</span> (uncapped, downloadable).</>
            ) : (
              <>Not archiving — logs stay in Redis with a 5000-line cap and ~1h retention. Pick an S3 storage to keep them permanently.</>
            )}
          </p>
          {noStorages && !storageId && (
            <Link href="/storage" className="text-xs font-medium text-primary hover:underline">
              + Add S3 storage
            </Link>
          )}
          {storageErr && <p className="w-full text-xs text-destructive">{storageErr}</p>}
        </CardContent>
      </Card>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search by model…"
            className="h-8 w-56 rounded-md border bg-background pl-8 pr-2 text-xs text-foreground"
          />
        </div>
        <Select value={source} onValueChange={setSource}>
          <SelectTrigger size="sm" className="w-[200px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {sourceOptions.map((o) => (
              <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div className="ml-auto flex items-center gap-2">
          <span className="text-xs text-muted-foreground">{total} file{total === 1 ? "" : "s"}</span>
          <Button variant="outline" size="xs" onClick={() => load(0, false)} disabled={loading}>
            {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
            Refresh
          </Button>
        </div>
      </div>

      {err && (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-4 py-2 text-sm text-destructive">{err}</div>
      )}

      {/* File table */}
      <Card className="overflow-hidden">
        <CardContent className="px-0 py-0">
          <table className="w-full text-sm">
            <thead className="border-b border-border bg-muted/20 text-left text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="w-6 px-2 py-2"></th>
                <th className="px-4 py-2 font-medium">Type</th>
                <th className="px-4 py-2 font-medium">
                  <button
                    type="button"
                    onClick={() => setSort((s) => (s === "started_desc" ? "started_asc" : "started_desc"))}
                    className="inline-flex items-center gap-1 hover:text-foreground"
                    title="Sort by start time"
                  >
                    Started
                    {sort === "started_desc" ? <ArrowDown className="h-3 w-3" /> : <ArrowUp className="h-3 w-3" />}
                  </button>
                </th>
                <th className="px-4 py-2 font-medium">Lines</th>
                <th className="px-4 py-2 font-medium">Size</th>
                <th className="px-4 py-2 font-medium">Crash</th>
                <th className="px-4 py-2 font-medium text-right">Download</th>
              </tr>
            </thead>
            <tbody>
              {files.map((f) => (
                <LogFileRow key={f.id} appId={app.app_id} f={f} />
              ))}
              {files.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-sm text-muted-foreground">
                    {loading ? "Loading log files…" : "No log files yet — they appear once a model or the worker has produced output."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {files.length < total && (
        <div className="flex justify-center">
          <Button variant="outline" size="sm" onClick={() => load(files.length, true)} disabled={loading}>
            {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ChevronDown className="h-3.5 w-3.5" />}
            Load more ({total - files.length} more)
          </Button>
        </div>
      )}

      <p className="text-[10px] leading-relaxed text-muted-foreground">
        {archived
          ? "Files are persisted to the selected S3 storage (uncapped). Each file is one launch session of a model — or the worker-agent scheduler log."
          : "Showing recent sessions from Redis (capped at 5000 lines, ~1h retention). Select an S3 storage above to keep full logs permanently."}
      </p>
    </div>
  );
}

function LogFileRow({ appId, f }: { appId: string; f: LogFile }) {
  const [open, setOpen] = useState(false);
  const startedLabel = f.started_at ? new Date(f.started_at).toLocaleString() : f.session;
  return (
    <>
      <tr className="border-b border-border/60 last:border-b-0">
        <td className="px-2 py-3 align-middle">
          <button
            onClick={() => setOpen((v) => !v)}
            className="flex items-center justify-center text-muted-foreground hover:text-foreground"
            aria-label={open ? "Hide log" : "View log"}
          >
            {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
          </button>
        </td>
        <td className="px-4 py-3">
          <span className="inline-flex items-center gap-2">
            <span className={cn(
              "rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide",
              isWorker(f.source) ? "bg-primary/15 text-primary" : "bg-muted text-muted-foreground",
            )}>
              {isWorker(f.source) ? "worker" : "model"}
            </span>
            <span className="truncate font-mono text-xs" title={typeLabel(f.source)}>{typeLabel(f.source)}</span>
            {f.live && (
              <span className="inline-flex items-center gap-1 text-[10px] text-status-active">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" /> live
              </span>
            )}
          </span>
        </td>
        <td className="px-4 py-3 text-xs text-muted-foreground" title={f.session}>{startedLabel}</td>
        <td className="px-4 py-3 font-mono text-xs tabular-nums">{f.lines.toLocaleString()}</td>
        <td className="px-4 py-3 font-mono text-xs tabular-nums">{fmtBytes(f.bytes)}</td>
        <td className="px-4 py-3">
          {f.crash ? (
            <span className="inline-flex max-w-[280px] items-start gap-1 text-xs text-status-down" title={f.crash}>
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
              <span className="truncate">{f.crash}</span>
            </span>
          ) : (
            <span className="text-xs text-muted-foreground">—</span>
          )}
        </td>
        <td className="px-4 py-3 text-right">
          <Button asChild variant="outline" size="xs">
            <a href={gateway.appLogFileDownloadUrl(appId, f.id)} download>
              <Download className="h-3 w-3" /> Download
            </a>
          </Button>
        </td>
      </tr>
      {open && (
        <tr className="border-b border-border/60 bg-muted/20">
          <td colSpan={7} className="px-4 py-3">
            <LogViewer appId={appId} file={f} />
          </td>
        </tr>
      )}
    </>
  );
}

const VIEW_TAIL = 2000;

function LogViewer({ appId, file }: { appId: string; file: LogFile }) {
  const [lines, setLines] = useState<string[]>([]);
  const [count, setCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const fetchContent = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const res = await gateway.getAppLogFile(appId, file.id, VIEW_TAIL);
      setLines(res.lines);
      setCount(res.count);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [appId, file.id]);

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { void fetchContent(); }, [fetchContent]);

  // Pin to the bottom on first load (newest output is what you usually want).
  const scrollRef = useRef<HTMLPreElement | null>(null);
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  const truncated = count >= VIEW_TAIL;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span className="font-mono">{typeLabel(file.source)} · {file.session}</span>
        <div className="flex items-center gap-2">
          {loading && <Loader2 className="h-3 w-3 animate-spin" />}
          <Button variant="outline" size="xs" onClick={fetchContent}>
            <RefreshCw className="h-3 w-3" />
          </Button>
          <Button asChild variant="outline" size="xs">
            <a href={gateway.appLogFileDownloadUrl(appId, file.id)} download>
              <Download className="h-3 w-3" /> Download full
            </a>
          </Button>
        </div>
      </div>
      {err ? (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">{err}</div>
      ) : lines.length === 0 ? (
        <div className="rounded-md border border-dashed border-border bg-background/40 px-3 py-4 text-center text-xs text-muted-foreground">
          {loading ? "loading…" : "this log file is empty"}
        </div>
      ) : (
        <pre
          ref={(el) => { scrollRef.current = el; }}
          className="terminal-block max-h-80 w-full overflow-y-auto whitespace-pre-wrap break-words rounded-md border border-border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-200 scrollbar-thin"
        >
          {lines.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </pre>
      )}
      {truncated && (
        <p className="text-[10px] text-muted-foreground">
          Showing the last {VIEW_TAIL.toLocaleString()} lines — download the file for the full log.
        </p>
      )}
    </div>
  );
}
