"use client";

import { useMemo, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Box, Boxes, Cpu, Inbox, Loader2, RefreshCw, Trash2, User } from "lucide-react";
import { avatarFor } from "@/lib/avatar";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { gateway } from "@/lib/gateway";
import type { AppRecord, ComputePod } from "@/lib/types";
import { cn } from "@/lib/utils";

type Row =
  | {
      kind: "compute";
      id: string;
      name: string;
      owner: string;
      gpu: string;
      gpu_count: number;
      status: string;
      created_at: string;
      ready_at: string | null;
      detail_href: string;
      raw: ComputePod;
    }
  | {
      kind: "inference";
      id: string;
      name: string;
      owner: string;
      gpu: string;
      gpu_count: number;
      status: string;
      created_at: string;
      ready_at: string | null;
      detail_href: string;
      raw: AppRecord;
    };

// Compute states that are *active* (live or in-flight). Pending/rejected/failed
// are operationally interesting elsewhere but they don't represent something
// running — keep this view focused on "what's billing right now".
const LIVE_COMPUTE_STATES = new Set(["running", "creating"]);

export function ProvisionedList({
  initialComputes,
  initialApps,
}: {
  initialComputes: ComputePod[];
  initialApps: AppRecord[];
}) {
  const router = useRouter();
  const [computes, setComputes] = useState(initialComputes);
  const [apps, setApps] = useState(initialApps);
  const [confirm, setConfirm] = useState<Row | null>(null);
  const [pending, startTransition] = useTransition();
  const [refreshing, setRefreshing] = useState(false);
  const [terminateError, setTerminateError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);

  const rows: Row[] = useMemo(() => {
    const computeRows: Row[] = computes
      .filter((c) => LIVE_COMPUTE_STATES.has(c.status))
      .map((c) => ({
        kind: "compute",
        id: c.id,
        name: c.name,
        owner: c.created_by,
        gpu: c.gpu_type,
        gpu_count: c.gpu_count,
        status: c.status,
        created_at: c.created_at,
        ready_at: c.ready_at,
        detail_href: `/compute/${c.id}`,
        raw: c,
      }));
    const appRows: Row[] = apps.map((a) => ({
      kind: "inference",
      id: a.app_id,
      name: a.name,
      owner: a.owner,
      gpu: a.gpu,
      gpu_count: a.gpu_count,
      status: "running",
      created_at: a.created_at,
      ready_at: a.created_at,
      detail_href: `/serverless/${a.app_id}`,
      raw: a,
    }));
    return [...computeRows, ...appRows].sort((a, b) =>
      b.created_at.localeCompare(a.created_at),
    );
  }, [computes, apps]);

  async function refresh() {
    setRefreshing(true);
    setRefreshError(null);
    try {
      const [c, a] = await Promise.all([
        gateway.listCompute(),
        gateway.listApps(),
      ]);
      setComputes(c);
      setApps(a);
    } catch (e) {
      setRefreshError(e instanceof Error ? e.message : String(e));
    } finally {
      setRefreshing(false);
    }
  }

  function terminate() {
    if (!confirm) return;
    const target = confirm;
    setTerminateError(null);
    startTransition(async () => {
      try {
        if (target.kind === "compute") {
          await gateway.deleteCompute(target.id);
          setComputes((cur) => cur.filter((c) => c.id !== target.id));
        } else {
          await gateway.deleteApp(target.id);
          setApps((cur) => cur.filter((a) => a.app_id !== target.id));
        }
        setConfirm(null);
        router.refresh();
      } catch (e) {
        setTerminateError(e instanceof Error ? e.message : String(e));
      }
    });
  }

  return (
    <>
      <div className="mb-3 flex items-center justify-between gap-3">
        <span className="text-xs text-muted-foreground">
          {rows.length} live · {computes.filter((c) => LIVE_COMPUTE_STATES.has(c.status)).length} compute · {apps.length} inference
        </span>
        <div className="flex items-center gap-3">
          {refreshError && (
            <span className="text-xs text-destructive">{refreshError}</span>
          )}
          <Button variant="outline" size="sm" onClick={refresh} disabled={refreshing}>
            {refreshing ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
            Refresh
          </Button>
        </div>
      </div>

      {rows.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
          <Inbox className="h-6 w-6 text-muted-foreground/60" />
          <p className="text-sm text-muted-foreground">Nothing provisioned right now.</p>
        </div>
      ) : (
        <ul className="flex flex-col gap-3">
          {rows.map((r) => (
            <ProvisionedCard
              key={`${r.kind}-${r.id}`}
              row={r}
              onTerminate={() => setConfirm(r)}
              terminating={pending}
            />
          ))}
        </ul>
      )}

      <Dialog
        open={!!confirm}
        onOpenChange={(o) => {
          if (!o && !pending) setConfirm(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Terminate {confirm?.name}?</DialogTitle>
            <DialogDescription>
              {confirm?.kind === "compute"
                ? "Stops billing immediately and deletes the pod from RunPod. Anything not saved to a persistent volume is lost."
                : "All workers will be drained and the queue cleared."}{" "}
              This can&apos;t be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {terminateError && (
              <p className="mr-auto text-sm text-destructive">{terminateError}</p>
            )}
            <Button
              variant="outline"
              onClick={() => setConfirm(null)}
              disabled={pending}
            >
              Cancel
            </Button>
            <Button variant="destructive" onClick={terminate} disabled={pending}>
              {pending && <Loader2 className="h-4 w-4 animate-spin" />}
              Terminate
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

function ProvisionedCard({
  row,
  onTerminate,
  terminating,
}: {
  row: Row;
  onTerminate: () => void;
  terminating: boolean;
}) {
  const avatar = avatarFor(row.name);
  const TypeIcon = row.kind === "compute" ? Box : Boxes;
  const typeLabel = row.kind === "compute" ? "Compute" : "Inference";
  return (
    <li className="group block rounded-xl border border-border bg-card p-4 transition-all hover:border-primary/40 hover:bg-card/80 hover:shadow-md">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-border bg-muted/60 text-base font-semibold text-muted-foreground">
            {avatar.letter}
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Link
                href={row.detail_href}
                className="truncate font-medium text-foreground underline-offset-2 hover:underline"
              >
                {row.name}
              </Link>
              <StatusPill status={row.status} />
            </div>
            <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="truncate font-mono" title={row.id}>{row.id}</span>
              <span>·</span>
              <User className="h-3 w-3" />
              <span className="truncate">{row.owner || "—"}</span>
            </div>
          </div>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={onTerminate}
          disabled={terminating}
          className="shrink-0 border-destructive/40 text-destructive hover:border-destructive/60 hover:bg-destructive/10 hover:text-destructive"
        >
          <Trash2 className="h-4 w-4" />
          Terminate
        </Button>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs">
          <TypeIcon className="h-3 w-3 text-muted-foreground" />
          <span>{typeLabel}</span>
        </span>
        <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs">
          <Cpu className="h-3 w-3 text-muted-foreground" />
          <span className="font-mono">
            {shortGpu(row.gpu)}
            {row.gpu_count > 1 ? ` × ${row.gpu_count}` : ""}
          </span>
        </span>
      </div>

      <div className="mt-3 flex items-center justify-end border-t border-border/60 pt-2 text-xs text-muted-foreground">
        <span title={new Date(row.created_at).toISOString()}>
          {new Date(row.created_at).toLocaleString()}
        </span>
      </div>
    </li>
  );
}

function StatusPill({ status }: { status: string }) {
  const styles =
    status === "running"
      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
      : status === "creating"
        ? "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400"
        : "border-border bg-muted text-muted-foreground";
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide",
        styles,
      )}
    >
      {status}
    </span>
  );
}

function shortGpu(gpu: string): string {
  return gpu
    .replace(/^NVIDIA\s+/i, "")
    .replace(/\s+GeForce\s+/i, " ")
    .replace(/^GeForce\s+/i, "");
}

