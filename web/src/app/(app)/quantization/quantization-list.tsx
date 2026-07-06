"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { useListUrlState, readParam } from "@/lib/list-url-state";
import { Cpu, Inbox, Loader2, MoreHorizontal, Search, Trash2, User, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { QuantizationJobRecord } from "@/lib/types";
import { avatarFor } from "@/lib/avatar";
import { shortGpu } from "@/lib/gpu-format";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Pagination } from "@/components/ui/pagination";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SortSelect, type SortDir } from "@/components/ui/sort-select";

const STATUS_PILL: Record<string, string> = {
  queued: "border border-border bg-muted text-muted-foreground",
  running: "border border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
  done: "border border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  failed: "border border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
  cancelled: "border border-border bg-muted text-muted-foreground",
};

const STATUS_OPTIONS = ["all", "queued", "running", "done", "failed", "cancelled"] as const;
type StatusFilter = (typeof STATUS_OPTIONS)[number];

export function QuantizationList({
  initialItems,
  initialTotal,
  scope,
}: {
  initialItems: QuantizationJobRecord[];
  initialTotal: number;
  scope: "mine" | "all";
}) {
  const sp = useSearchParams();
  const [q, setQ] = useState(() => sp.get("q") ?? "");
  const [qDebounced, setQDebounced] = useState(q);
  const [status, setStatus] = useState<StatusFilter>(() => readParam(sp, "status", STATUS_OPTIONS, "all"));
  const [sort, setSort] = useState<SortDir>(() => readParam(sp, "sort", ["newest", "oldest"] as const, "newest"));
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(12);

  const [items, setItems] = useState<QuantizationJobRecord[]>(initialItems);
  const [total, setTotal] = useState(initialTotal);
  const [loading, setLoading] = useState(false);

  const [single, setSingle] = useState<QuantizationJobRecord | null>(null);
  const [singleDeleting, setSingleDeleting] = useState(false);
  const [singleError, setSingleError] = useState<string | null>(null);

  useListUrlState({ q, status, sort });

  useEffect(() => {
    const t = setTimeout(() => setQDebounced(q), 300);
    return () => clearTimeout(t);
  }, [q]);

  const hasFilter = q.trim().length > 0 || status !== "all";
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const currentPage = Math.min(page, pageCount);

  const seqRef = useRef(0);
  const load = useCallback(async () => {
    const seq = ++seqRef.current;
    setLoading(true);
    try {
      const res = await gateway.listQuantizationJobsPage({
        scope,
        q: qDebounced,
        status: status === "all" ? "" : status,
        sort,
        limit: pageSize,
        offset: (currentPage - 1) * pageSize,
      });
      if (seq !== seqRef.current) return;
      setItems(res.items);
      setTotal(res.total);
    } catch (e) {
      if (seq === seqRef.current) toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      if (seq === seqRef.current) setLoading(false);
    }
  }, [scope, qDebounced, status, sort, pageSize, currentPage]);

  const bootedRef = useRef(false);
  useEffect(() => {
    if (!bootedRef.current) {
      bootedRef.current = true;
      const ssrDefaults =
        qDebounced === "" && status === "all" && sort === "newest" && page === 1 && pageSize === 12;
      if (ssrDefaults) return;
    }
    void load();
  }, [load, qDebounced, status, sort, page, pageSize]);

  const onSingleDelete = async () => {
    if (!single) return;
    setSingleError(null);
    setSingleDeleting(true);
    try {
      await gateway.deleteQuantizationJob(single.id);
      setSingle(null);
      void load();
    } catch (e) {
      setSingleError(e instanceof Error ? e.message : String(e));
    } finally {
      setSingleDeleting(false);
    }
  };

  return (
    <div>
      <div className="mb-4 flex gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="search"
            placeholder="Search by name, id, model, scheme, owner, status…"
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              setPage(1);
            }}
            className="h-10 w-full rounded-md border border-input bg-background pl-9 pr-9 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
          />
          {q && (
            <button
              type="button"
              onClick={() => {
                setQ("");
                setPage(1);
              }}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              title="Clear"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        <Select
          value={status}
          onValueChange={(v) => {
            setStatus(v as StatusFilter);
            setPage(1);
          }}
        >
          <SelectTrigger className="h-10! w-[150px]" title="Filter by status">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {STATUS_OPTIONS.map((s) => (
              <SelectItem key={s} value={s}>
                {s === "all" ? "All statuses" : s.charAt(0).toUpperCase() + s.slice(1)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <SortSelect value={sort} onValueChange={setSort} />
      </div>

      {hasFilter && (
        <div className="mb-3 text-xs text-muted-foreground">
          {loading && <Loader2 className="mr-1.5 inline h-3 w-3 animate-spin" />}
          {total} {total === 1 ? "match" : "matches"}
          {q && (
            <>
              {" "}for <span className="font-mono text-foreground">&quot;{q}&quot;</span>
            </>
          )}
          {status !== "all" && (
            <>
              {" "}· status <span className="font-mono text-foreground">{status}</span>
            </>
          )}
        </div>
      )}

      {total === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
          <Inbox className="h-6 w-6 text-muted-foreground/60" />
          <p className="text-sm text-muted-foreground">No quantization jobs match your filters.</p>
        </div>
      ) : (
        <>
          <div
            className={cn(
              "grid grid-cols-1 gap-3 transition-opacity md:grid-cols-2 xl:grid-cols-3",
              loading && "pointer-events-none opacity-60",
            )}
          >
            {items.map((r) => (
              <JobItem key={r.id} job={r} onDelete={setSingle} />
            ))}
          </div>
          <Pagination
            page={currentPage}
            pageCount={pageCount}
            total={total}
            pageSize={pageSize}
            onPageChange={setPage}
            onPageSizeChange={(n) => {
              setPageSize(n);
              setPage(1);
            }}
            itemLabel="jobs"
          />
        </>
      )}

      <Dialog
        open={!!single}
        onOpenChange={(o) => {
          if (!singleDeleting && !o) {
            setSingle(null);
            setSingleError(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete {single?.name}?</DialogTitle>
            <DialogDescription>
              Removes the quantization-job record. S3 artifacts are kept. If a RunPod pod
              is still alive, terminate it from RunPod&apos;s dashboard.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {singleError && <p className="mr-auto text-sm text-destructive">{singleError}</p>}
            <Button variant="outline" onClick={() => setSingle(null)} disabled={singleDeleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={onSingleDelete} disabled={singleDeleting}>
              {singleDeleting ? "Deleting…" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function JobItem({
  job,
  onDelete,
}: {
  job: QuantizationJobRecord;
  onDelete: (r: QuantizationJobRecord) => void;
}) {
  const avatar = avatarFor(job.name);
  const model = job.source_model.split("/").pop() ?? job.source_model;
  const gpu = job.gpu_type
    ? `${shortGpu(job.gpu_type)}${job.gpu_count > 1 ? ` × ${job.gpu_count}` : ""}`
    : job.visible_devices
      ? `GPUs ${job.visible_devices}`
      : null;
  const quantGb = job.result_json?.sizes?.quantized_gb;
  const chip = "inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs";

  return (
    <Link
      href={`/quantization/${encodeURIComponent(job.id)}`}
      className={cn(
        "group block rounded-xl border border-border bg-card p-4 transition-all",
        "hover:border-primary/40 hover:bg-card/80 hover:shadow-md",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-border bg-muted/60 text-base font-semibold text-muted-foreground">
            {avatar.letter}
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="truncate font-medium text-foreground">{job.name}</span>
              <span
                className={cn(
                  "rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide",
                  STATUS_PILL[job.status] ?? STATUS_PILL.queued,
                )}
              >
                {job.status}
              </span>
            </div>
            <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="truncate font-mono" title={job.id}>{job.id}</span>
              <span>·</span>
              <User className="h-3 w-3" />
              <span className="truncate">{job.created_by}</span>
            </div>
          </div>
        </div>

        <div className="flex shrink-0 items-start gap-2">
          {quantGb != null && (
            <div className="rounded-md border border-border bg-muted/40 px-2.5 py-1 text-right">
              <div className="text-[10px] uppercase tracking-wide text-muted-foreground">Size</div>
              <div className="font-mono text-sm font-semibold tabular-nums">{quantGb} GB</div>
            </div>
          )}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon-sm"
                className="-mr-1 text-muted-foreground hover:text-foreground"
                aria-label="Actions"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                }}
              >
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
              <DropdownMenuItem
                variant="destructive"
                onSelect={(e) => {
                  e.preventDefault();
                  onDelete(job);
                }}
              >
                <Trash2 className="h-4 w-4" />
                Delete job
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        <span
          className={cn(chip, "bg-transparent border border-violet-500/40 text-violet-600 dark:text-violet-300")}
        >
          {job.scheme}
        </span>
        <span className={chip}>
          <span className="font-mono">{model}</span>
        </span>
        {gpu && (
          <span className={chip}>
            <Cpu className="h-3 w-3 text-muted-foreground" />
            <span className="font-mono">{gpu}</span>
          </span>
        )}
      </div>

      <div className="mt-3 flex items-center justify-between border-t border-border/60 pt-2 text-xs text-muted-foreground">
        <div className="flex items-center gap-3">
          {job.result_json?.hf_repo && (
            <span className="truncate font-mono" title={job.result_json.hf_repo}>
              → {job.result_json.hf_repo}
            </span>
          )}
          {job.exit_code != null && job.exit_code !== 0 && (
            <span className="font-mono text-destructive">exit {job.exit_code}</span>
          )}
        </div>
        <span title={new Date(job.created_at).toISOString()}>
          {new Date(job.created_at).toLocaleString()}
        </span>
      </div>
    </Link>
  );
}
