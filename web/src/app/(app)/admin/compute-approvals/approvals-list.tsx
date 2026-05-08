"use client";

import { useState, useTransition } from "react";
import { Check, Loader2, X } from "lucide-react";
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
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { gateway } from "@/lib/gateway";
import type { ComputePod } from "@/lib/types";

export function ApprovalsList({ initial }: { initial: ComputePod[] }) {
  const [items, setItems] = useState<ComputePod[]>(initial);
  const [rejecting, setRejecting] = useState<ComputePod | null>(null);
  const [reason, setReason] = useState("");
  const [pending, startTransition] = useTransition();
  const [busyId, setBusyId] = useState<string | null>(null);

  function approve(pod: ComputePod) {
    setBusyId(pod.id);
    startTransition(async () => {
      try {
        await gateway.approveCompute(pod.id);
        setItems((cur) => cur.filter((p) => p.id !== pod.id));
        toast.success(`Approved ${pod.name} — provisioning now`);
      } catch (e) {
        toast.error(e instanceof Error ? e.message : String(e));
      } finally {
        setBusyId(null);
      }
    });
  }

  function reject() {
    if (!rejecting) return;
    const target = rejecting;
    setBusyId(target.id);
    startTransition(async () => {
      try {
        await gateway.rejectCompute(target.id, reason.trim() || undefined);
        setItems((cur) => cur.filter((p) => p.id !== target.id));
        toast.success(`Rejected ${target.name}`);
        setRejecting(null);
        setReason("");
      } catch (e) {
        toast.error(e instanceof Error ? e.message : String(e));
      } finally {
        setBusyId(null);
      }
    });
  }

  if (items.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border px-6 py-12 text-center text-sm text-muted-foreground">
        No pending requests.
      </div>
    );
  }

  return (
    <>
      <ul className="grid gap-3">
        {items.map((p) => (
          <li
            key={p.id}
            className="rounded-lg border border-border bg-card p-4"
          >
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <h3 className="truncate text-sm font-medium">{p.name}</h3>
                  <span className="inline-flex items-center rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-700 dark:text-amber-400">
                    pending
                  </span>
                </div>
                <p className="mt-0.5 truncate font-mono text-xs text-muted-foreground">
                  {p.id}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setRejecting(p)}
                  disabled={pending && busyId === p.id}
                  className="text-destructive hover:text-destructive"
                >
                  <X className="h-4 w-4" />
                  Reject
                </Button>
                <Button
                  size="sm"
                  onClick={() => approve(p)}
                  disabled={pending && busyId === p.id}
                >
                  {pending && busyId === p.id ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Check className="h-4 w-4" />
                  )}
                  Approve
                </Button>
              </div>
            </div>

            <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 text-xs sm:grid-cols-4">
              <Field label="Requested by" value={p.created_by} />
              <Field
                label="GPU"
                value={`${shortGpu(p.gpu_type)} × ${p.gpu_count}`}
              />
              <Field label="Disk" value={`${p.container_disk_gb} GB`} />
              <Field label="Cloud" value={p.cloud_type.toLowerCase()} />
              <Field label="Template" value={p.template_id ?? "—"} />
              <Field
                label="Volume"
                value={p.volume_gb > 0 ? `${p.volume_gb} GB` : "—"}
              />
              <Field
                label="Submitted"
                value={relativeTime(p.created_at)}
                className="col-span-2"
              />
            </dl>
          </li>
        ))}
      </ul>

      <Dialog
        open={!!rejecting}
        onOpenChange={(o) => {
          if (!o && !pending) {
            setRejecting(null);
            setReason("");
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reject {rejecting?.name}?</DialogTitle>
            <DialogDescription>
              The requester will see this on their pod page. Optional but
              encouraged so they know what to change.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-1.5">
            <Label htmlFor="reject-reason">Reason (optional)</Label>
            <Textarea
              id="reject-reason"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="e.g. Pick a smaller GPU; H100 is reserved for prod."
              rows={3}
              maxLength={1024}
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setRejecting(null);
                setReason("");
              }}
              disabled={pending}
            >
              Cancel
            </Button>
            <Button variant="destructive" onClick={reject} disabled={pending}>
              {pending && <Loader2 className="h-4 w-4 animate-spin" />}
              Reject request
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

function Field({
  label,
  value,
  className,
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div className={className}>
      <dt className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </dt>
      <dd className="mt-0.5 break-words text-foreground">{value}</dd>
    </div>
  );
}

function shortGpu(gpu: string): string {
  return gpu
    .replace(/^NVIDIA\s+/i, "")
    .replace(/\s+GeForce\s+/i, " ")
    .replace(/^GeForce\s+/i, "");
}

function relativeTime(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  const diff = Date.now() - d.getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}
