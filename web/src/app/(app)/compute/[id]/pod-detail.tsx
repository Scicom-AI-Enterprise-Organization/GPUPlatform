"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Copy, Download, ExternalLink, Loader2, Trash2 } from "lucide-react";
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
import { formatCostUSD, formatRateUSD, useLiveCost } from "@/lib/cost";
import { BurnFlame } from "@/components/burn-flame";
import { shortGpu } from "@/lib/gpu-format";
import type { ComputePod, ComputeStatus, ProviderRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

const POLL_MS = 4000;

export function PodDetail({ initial }: { initial: ComputePod }) {
  const router = useRouter();
  const [pod, setPod] = useState<ComputePod>(initial);
  const [terminating, setTerminating] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [terminateError, setTerminateError] = useState<string | null>(null);
  const [provider, setProvider] = useState<ProviderRecord | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    if (!pod.provider_id) {
      setProvider(null);
      return;
    }
    gateway
      .listProviders()
      .then((rows) => setProvider(rows.find((r) => r.id === pod.provider_id) ?? null))
      .catch(() => {});
  }, [pod.provider_id]);

  const providerLabel =
    provider?.kind === "pi"
      ? "Prime Intellect"
      : provider?.kind === "runpod"
        ? "RunPod"
        : pod.provider_id
          ? "Custom account"
          : "RunPod (gateway default)";
  const keyHint = pod.provider_id ? `sgpu-${pod.provider_id}` : "sgpu-runpod";

  // Poll while we're waiting for the pod to move out of an in-flight state.
  // 'pending_approval' polls so the requester sees the status flip the moment
  // an admin approves; 'creating' polls so SSH info appears as soon as RunPod
  // is ready. Terminal states (running/failed/terminated/rejected) stop polling.
  useEffect(() => {
    const inFlight = pod.status === "creating" || pod.status === "pending_approval";
    if (!inFlight) {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }
    if (pollRef.current) return;
    pollRef.current = window.setInterval(async () => {
      try {
        const next = await gateway.getCompute(pod.id);
        setPod(next);
      } catch {
        // Ignore — leave the previous state on screen until the next tick.
      }
    }, POLL_MS);
    return () => {
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [pod.id, pod.status]);

  async function copy(value: string, what: string) {
    try {
      await navigator.clipboard.writeText(value);
      toast.success(`${what} copied`, { duration: 3000 });
    } catch {
      toast.error("Couldn't access clipboard", { duration: 4000 });
    }
  }

  async function downloadKey() {
    try {
      const info = await gateway.getComputeSsh(pod.id);
      const blob = new Blob([info.private_key], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = keyHint;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      toast.success("SSH key downloaded — chmod 600 before use", { duration: 4000 });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e), { duration: 5000 });
    }
  }

  async function terminate() {
    setTerminating(true);
    setTerminateError(null);
    try {
      await gateway.deleteCompute(pod.id);
      router.push("/compute");
    } catch (e) {
      setTerminateError(e instanceof Error ? e.message : String(e));
      setTerminating(false);
    }
  }

  const sshCmd =
    pod.status === "running" && pod.public_ip && pod.ssh_port
      ? `ssh -i ~/.ssh/${keyHint} -p ${pod.ssh_port} ${pod.ssh_user}@${pod.public_ip}`
      : null;

  return (
    <div className="space-y-6">
      {/* Header — name + status pill + actions. No coloured icon tile. */}
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border pb-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="truncate text-2xl font-semibold tracking-tight">{pod.name}</h1>
            <StatusPill status={pod.status} />
          </div>
          <p className="mt-1 font-mono text-xs text-muted-foreground">{pod.id}</p>
        </div>
        {pod.status !== "terminated" && pod.status !== "rejected" && (
          <Button
            variant="outline"
            onClick={() => setConfirmOpen(true)}
            disabled={terminating}
            className="border-destructive/40 text-destructive hover:border-destructive/60 hover:bg-destructive/10 hover:text-destructive"
          >
            {terminating ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Trash2 className="h-4 w-4" />
            )}
            {pod.status === "pending_approval" ? "Withdraw" : "Terminate"}
          </Button>
        )}
      </div>

      <Dialog
        open={confirmOpen}
        onOpenChange={(o) => {
          if (!terminating) {
            setConfirmOpen(o);
            if (!o) setTerminateError(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {pod.status === "pending_approval"
                ? `Withdraw request for ${pod.name}?`
                : `Terminate ${pod.name}?`}
            </DialogTitle>
            <DialogDescription>
              {pod.status === "pending_approval"
                ? "The approval request will be cancelled. You can submit a new one any time."
                : `Stops billing immediately and deletes the pod from ${providerLabel}. Anything not saved to a persistent volume is lost. This can't be undone.`}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {terminateError && (
              <p className="mr-auto text-sm text-destructive">{terminateError}</p>
            )}
            <Button
              variant="outline"
              onClick={() => setConfirmOpen(false)}
              disabled={terminating}
            >
              Cancel
            </Button>
            <Button variant="destructive" onClick={terminate} disabled={terminating}>
              {terminating && <Loader2 className="h-4 w-4 animate-spin" />}
              {terminating
                ? pod.status === "pending_approval"
                  ? "Withdrawing…"
                  : "Terminating…"
                : pod.status === "pending_approval"
                  ? "Withdraw"
                  : "Terminate"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Spec — neutral metadata, no colour. */}
      <Card title="Specification">
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3">
          <Field label="GPU" value={`${shortGpu(pod.gpu_type)} × ${pod.gpu_count}`} />
          <Field label="Container disk" value={`${pod.container_disk_gb} GB`} />
          <Field label="Volume" value={pod.volume_gb > 0 ? `${pod.volume_gb} GB` : "—"} />
          <Field label="Cloud" value={pod.cloud_type.toLowerCase()} />
          <Field label="Provider" value={providerLabel} />
          <Field label="Template" value={pod.template_id ?? "—"} />
          <Field label="Rate" value={formatRateUSD(pod.cost_per_hr)} />
          <Field
            label="Auto-terminate"
            value={
              pod.idle_terminate_after_s > 0
                ? `idle ${formatIdle(pod.idle_terminate_after_s)}`
                : "off"
            }
          />
          <LiveCostField pod={pod} />
          <Field label="Image" value={pod.image} className="col-span-full font-mono text-xs" />
        </dl>
      </Card>

      {/* Pending / rejected — explain what's happening before showing connect UI. */}
      {pod.status === "pending_approval" && (
        <Card title="Awaiting approval">
          <p className="text-sm text-muted-foreground">
            Your request has been submitted to an admin for approval. Once approved
            the pod will start provisioning automatically — this page will refresh
            on its own.
          </p>
        </Card>
      )}

      {pod.status === "rejected" && (
        <Card title="Request rejected">
          <p className="text-sm text-muted-foreground">
            An admin rejected this request. You can submit a new one with different
            specs from the Compute page.
          </p>
          {pod.reject_reason && (
            <pre className="mt-3 whitespace-pre-wrap break-words rounded-md border border-border bg-muted/40 p-3 font-mono text-xs text-muted-foreground">
              {pod.reject_reason}
            </pre>
          )}
        </Card>
      )}

      {/* Connect — only meaningful when running. Show a clear "wait" UI in
          other states rather than empty/dead controls. */}
      {pod.status === "creating" && (
        <Card title="Connect">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Provisioning — SSH and JupyterLab will appear here once the pod is ready.
          </div>
        </Card>
      )}

      {pod.status === "failed" && (
        <Card title="Failure">
          <pre className="whitespace-pre-wrap break-words rounded-md border border-border bg-muted/40 p-3 font-mono text-xs text-muted-foreground">
            {pod.error_text ?? "Pod failed to provision."}
          </pre>
        </Card>
      )}

      {pod.status === "terminated" && (
        <Card title="Terminated">
          <p className="text-sm text-muted-foreground">
            This pod has been terminated. Billing stopped at{" "}
            {pod.terminated_at ? new Date(pod.terminated_at).toLocaleString() : "—"}.
          </p>
        </Card>
      )}

      {pod.status === "running" && (
        <>
          {pod.jupyter_url && (
          <Card
            title="JupyterLab"
            subtitle="One-time token baked into the URL — click Open, no password prompt."
          >
            <div className="space-y-3">
              <Row label="URL">
                <div className="flex flex-1 items-center gap-2">
                  <a
                    href={pod.jupyter_url ?? "#"}
                    target="_blank"
                    rel="noreferrer"
                    className="flex-1 truncate font-mono text-xs text-foreground underline-offset-2 hover:underline"
                  >
                    {pod.jupyter_url}
                  </a>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => pod.jupyter_url && copy(pod.jupyter_url, "URL")}
                    title="Copy URL"
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                  <Button asChild type="button" variant="outline" size="sm" title="Open">
                    <a href={pod.jupyter_url ?? "#"} target="_blank" rel="noreferrer">
                      <ExternalLink className="h-4 w-4" />
                      Open
                    </a>
                  </Button>
                </div>
              </Row>
            </div>
          </Card>
          )}

          <Card title="SSH" subtitle="For terminal access. Download the key once and chmod 0600.">
            <div className="space-y-3">
              <Row label="Command">
                <div className="flex flex-1 items-center gap-2">
                  <code className="terminal-block flex-1 truncate rounded-md border border-border bg-muted/40 px-2 py-1 font-mono text-xs">
                    {sshCmd}
                  </code>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => sshCmd && copy(sshCmd, "SSH command")}
                    title="Copy command"
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </Row>
              <Row label="Private key">
                <Button type="button" variant="outline" size="sm" onClick={downloadKey}>
                  <Download className="h-4 w-4" />
                  Download key
                </Button>
              </Row>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}

function Card({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-border bg-card p-5">
      <div className="mb-4">
        <h2 className="text-sm font-semibold">{title}</h2>
        {subtitle && (
          <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>
        )}
      </div>
      {children}
    </section>
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
      <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </dt>
      <dd className="mt-0.5 break-words text-sm">{value}</dd>
    </div>
  );
}

function LiveCostField({ pod }: { pod: ComputePod }) {
  // Billing starts when ready_at is set (pod actually came up) and freezes
  // at terminated_at. Anything before ready_at isn't charged.
  const live = useLiveCost(pod.ready_at, pod.terminated_at, pod.cost_per_hr);
  const isLive = pod.ready_at != null && pod.terminated_at == null;
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        Cost {isLive ? "(live)" : ""}
      </dt>
      <dd
        className={cn(
          "mt-0.5 flex items-center gap-1.5 break-words text-sm tabular-nums",
          isLive && "text-amber-600 dark:text-amber-400",
        )}
      >
        {isLive && <BurnFlame />}
        {formatCostUSD(live)}
      </dd>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <span className="w-24 shrink-0 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      {children}
    </div>
  );
}

// Render an idle-timeout (seconds) as a compact human string: "30m", "2h", "1h30m".
function formatIdle(seconds: number): string {
  const mins = Math.round(seconds / 60);
  if (mins < 60) return `${mins}m`;
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return m === 0 ? `${h}h` : `${h}h${m}m`;
}

function StatusPill({ status }: { status: ComputeStatus }) {
  const styles: Record<ComputeStatus, string> = {
    running:
      "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
    creating:
      "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
    pending_approval:
      "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
    failed:
      "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
    rejected:
      "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
    terminated: "border-border bg-muted text-muted-foreground",
  };
  const label: Record<ComputeStatus, string> = {
    running: "running",
    creating: "creating",
    pending_approval: "pending",
    failed: "failed",
    rejected: "rejected",
    terminated: "terminated",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide",
        styles[status],
      )}
    >
      {label[status]}
    </span>
  );
}

