"use client";

import { useMemo, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Box, Boxes, Loader2, RefreshCw, Trash2 } from "lucide-react";
import { toast } from "sonner";
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
    try {
      const [c, a] = await Promise.all([
        gateway.listCompute(),
        gateway.listApps(),
      ]);
      setComputes(c);
      setApps(a);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setRefreshing(false);
    }
  }

  function terminate() {
    if (!confirm) return;
    const target = confirm;
    startTransition(async () => {
      try {
        if (target.kind === "compute") {
          await gateway.deleteCompute(target.id);
          setComputes((cur) => cur.filter((c) => c.id !== target.id));
        } else {
          await gateway.deleteApp(target.id);
          setApps((cur) => cur.filter((a) => a.app_id !== target.id));
        }
        toast.success(`Terminated ${target.name}`);
        setConfirm(null);
        router.refresh();
      } catch (e) {
        toast.error(e instanceof Error ? e.message : String(e));
      }
    });
  }

  return (
    <>
      <div className="mb-3 flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {rows.length} live · {computes.filter((c) => LIVE_COMPUTE_STATES.has(c.status)).length} compute · {apps.length} inference
        </span>
        <Button variant="outline" size="sm" onClick={refresh} disabled={refreshing}>
          {refreshing ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <RefreshCw className="h-4 w-4" />
          )}
          Refresh
        </Button>
      </div>

      {rows.length === 0 ? (
        <div className="rounded-md border border-dashed border-border px-6 py-12 text-center text-sm text-muted-foreground">
          Nothing provisioned right now.
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead className="bg-muted/40 text-left text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-3 py-2">Type</th>
                <th className="px-3 py-2">Name</th>
                <th className="px-3 py-2">Owner</th>
                <th className="px-3 py-2">GPU</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2">Provisioned</th>
                <th className="px-3 py-2 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.kind}-${r.id}`} className="border-t border-border">
                  <td className="px-3 py-2">
                    <TypeBadge kind={r.kind} />
                  </td>
                  <td className="px-3 py-2">
                    <Link
                      href={r.detail_href}
                      className="font-medium text-foreground underline-offset-2 hover:underline"
                    >
                      {r.name}
                    </Link>
                    <div className="font-mono text-[11px] text-muted-foreground">
                      {r.id}
                    </div>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">{r.owner || "—"}</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    {shortGpu(r.gpu)} × {r.gpu_count}
                  </td>
                  <td className="px-3 py-2">
                    <StatusPill status={r.status} />
                  </td>
                  <td className="px-3 py-2 text-muted-foreground" title={r.created_at}>
                    {relativeTime(r.created_at)}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setConfirm(r)}
                      disabled={pending}
                      className="text-destructive hover:text-destructive"
                    >
                      <Trash2 className="h-4 w-4" />
                      Terminate
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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

function TypeBadge({ kind }: { kind: "compute" | "inference" }) {
  const Icon = kind === "compute" ? Box : Boxes;
  const label = kind === "compute" ? "Compute" : "Inference";
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-border bg-background px-2 py-0.5 text-[11px] text-muted-foreground">
      <Icon className="h-3.5 w-3.5" />
      {label}
    </span>
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

function relativeTime(iso: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  const diff = Date.now() - d.getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}
