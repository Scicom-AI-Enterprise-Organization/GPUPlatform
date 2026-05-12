"use client";

import { useState, useTransition } from "react";
import { Check, Cpu, Inbox, Loader2, User, X } from "lucide-react";
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
  const [rowError, setRowError] = useState<{ id: string; msg: string } | null>(null);
  const [rejectError, setRejectError] = useState<string | null>(null);

  function approve(pod: ComputePod) {
    setBusyId(pod.id);
    setRowError(null);
    startTransition(async () => {
      try {
        await gateway.approveCompute(pod.id);
        setItems((cur) => cur.filter((p) => p.id !== pod.id));
      } catch (e) {
        setRowError({ id: pod.id, msg: e instanceof Error ? e.message : String(e) });
      } finally {
        setBusyId(null);
      }
    });
  }

  function reject() {
    if (!rejecting) return;
    const target = rejecting;
    setBusyId(target.id);
    setRejectError(null);
    startTransition(async () => {
      try {
        await gateway.rejectCompute(target.id, reason.trim() || undefined);
        setItems((cur) => cur.filter((p) => p.id !== target.id));
        setRejecting(null);
        setReason("");
      } catch (e) {
        setRejectError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusyId(null);
      }
    });
  }

  if (items.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
        <Inbox className="h-6 w-6 text-muted-foreground/60" />
        <p className="text-sm text-muted-foreground">No pending requests.</p>
      </div>
    );
  }

  return (
    <>
      <ul className="flex flex-col gap-3">
        {items.map((p) => {
          const avatar = avatarFor(p.name);
          const busy = pending && busyId === p.id;
          return (
            <li
              key={p.id}
              className="group block rounded-xl border border-border bg-card p-4 transition-all"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-center gap-3">
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-border bg-muted/60 text-base font-semibold text-muted-foreground">
                    {avatar.letter}
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="truncate font-medium text-foreground">{p.name}</span>
                      <span className="inline-flex items-center rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-700 dark:text-amber-400">
                        pending
                      </span>
                    </div>
                    <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
                      <span className="truncate font-mono" title={p.id}>{p.id}</span>
                      <span>·</span>
                      <User className="h-3 w-3" />
                      <span className="truncate">{p.created_by}</span>
                    </div>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setRejecting(p)}
                    disabled={busy}
                    className="border-destructive/40 text-destructive hover:border-destructive/60 hover:bg-destructive/10 hover:text-destructive"
                  >
                    <X className="h-4 w-4" />
                    Reject
                  </Button>
                  <Button size="sm" onClick={() => approve(p)} disabled={busy}>
                    {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
                    Approve
                  </Button>
                </div>
              </div>

              <div className="mt-3 flex flex-wrap items-center gap-1.5">
                <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 text-xs">
                  <Cpu className="h-3 w-3 text-muted-foreground" />
                  <span className="font-mono">
                    {shortGpu(p.gpu_type)}
                    {p.gpu_count > 1 ? ` × ${p.gpu_count}` : ""}
                  </span>
                </span>
                <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 font-mono text-xs">
                  {p.container_disk_gb} GB disk
                </span>
                {p.volume_gb > 0 && (
                  <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 font-mono text-xs">
                    {p.volume_gb} GB vol
                  </span>
                )}
                {p.template_id && (
                  <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 font-mono text-xs">
                    {p.template_id}
                  </span>
                )}
                <span className="inline-flex items-center gap-1 rounded-md bg-muted/50 px-2 py-0.5 font-mono text-xs">
                  {p.cloud_type.toLowerCase()}
                </span>
              </div>

              <div className="mt-3 flex items-center justify-between gap-3 border-t border-border/60 pt-2 text-xs text-muted-foreground">
                {rowError?.id === p.id ? (
                  <span className="text-destructive">{rowError.msg}</span>
                ) : (
                  <span />
                )}
                <span title={new Date(p.created_at).toISOString()}>
                  {new Date(p.created_at).toLocaleString()}
                </span>
              </div>
            </li>
          );
        })}
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
            {rejectError && (
              <p className="mr-auto text-sm text-destructive">{rejectError}</p>
            )}
            <Button
              variant="outline"
              onClick={() => {
                setRejecting(null);
                setReason("");
                setRejectError(null);
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

function shortGpu(gpu: string): string {
  return gpu
    .replace(/^NVIDIA\s+/i, "")
    .replace(/\s+GeForce\s+/i, " ")
    .replace(/^GeForce\s+/i, "");
}
