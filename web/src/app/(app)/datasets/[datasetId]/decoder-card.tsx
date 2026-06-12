"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Check, Cpu, Loader2, Play, RefreshCw, Server, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { SearchableSelect } from "@/components/ui/searchable-select";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AvailabilityBadge } from "@/components/availability-badge";
import { useGpuAvailability } from "@/lib/use-gpu-availability";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { GpuTypeOption, ProviderRecord, VmAvailability } from "@/lib/types";

/** State the card lifts up so the row browser knows where to send decode calls. */
export type DecoderState = { providerId: string; ready: boolean };

type Status = { running: boolean; ready: boolean; device?: string | null; logs?: string[] };

const GPU_COUNT_CHOICES = [1, 2, 4, 8] as const;
const RUNPOD_GPU_FALLBACK: GpuTypeOption[] = [
  { id: "NVIDIA L40S", label: "L40S", vram_gb: 48, hint: "" },
  { id: "NVIDIA A100 80GB PCIe", label: "A100 80GB", vram_gb: 80, hint: "datacenter" },
  { id: "NVIDIA H100 80GB HBM3", label: "H100 80GB", vram_gb: 80, hint: "fastest" },
];
const gpuHint = (vramGb: number, count: number) =>
  `${vramGb * count >= 100 ? Math.round(vramGb * count) : vramGb * count} GB VRAM${count > 1 ? ` · ×${count}` : ""}`;
const fmtMib = (mib: number) => (mib >= 1024 ? `${(mib / 1024).toFixed(1)} GiB` : `${mib} MiB`);

/**
 * Loads NeuCodec persistently on a chosen box and keeps it resident (idle auto-
 * unloads server-side) so each "play utt N" on a packed dataset is instant. The
 * "Run on" picker mirrors the Transform card — a fresh RunPod pod, or a registered
 * VM. (RunPod cold-installs the codec on first load; a VM with the TTS venv is
 * instant.)
 */
export function DecoderCard({
  datasetId,
  onState,
}: {
  datasetId: string;
  onState: (s: DecoderState | null) => void;
}) {
  const [providers, setProviders] = useState<ProviderRecord[]>([]);
  const [target, setTarget] = useState<"cloud" | "vm">("vm");
  const [vmProviderId, setVmProviderId] = useState("");
  const [runpodProviderId, setRunpodProviderId] = useState("");
  const [gpuType, setGpuType] = useState("NVIDIA L40S");
  const [gpuCount, setGpuCount] = useState(1);
  const [secureCloud, setSecureCloud] = useState(true);
  const [vmGpu, setVmGpu] = useState("auto"); // which VM GPU id to load NeuCodec on ("auto" = most free)
  const [gpuOptions, setGpuOptions] = useState<GpuTypeOption[]>(RUNPOD_GPU_FALLBACK);
  const [vmAvail, setVmAvail] = useState<{ status: "idle" | "loading" | "ok" | "error"; data?: VmAvailability; message?: string }>({ status: "idle" });

  const [status, setStatus] = useState<Status | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const poll = useRef<ReturnType<typeof setInterval> | null>(null);

  const vmProviders = useMemo(() => providers.filter((p) => p.kind === "vm"), [providers]);
  const runpodProviders = useMemo(() => providers.filter((p) => p.kind === "runpod"), [providers]);
  const availability = useGpuAvailability(gpuType, gpuCount, target === "cloud", secureCloud ? "SECURE" : "COMMUNITY");

  // The provider id the decode calls target: the VM, or the chosen/default RunPod account.
  const activeProviderId = target === "vm" ? vmProviderId : (runpodProviderId || "__runpod_default__");

  useEffect(() => {
    gateway.listProviders().then(setProviders).catch(() => {});
    gateway.listRunpodGpuTypes().then((t) => { if (t?.length) setGpuOptions(t); }).catch(() => {});
  }, []);

  const refreshVmAvail = useCallback(async (id: string) => {
    if (!id) return setVmAvail({ status: "idle" });
    setVmAvail({ status: "loading" });
    try {
      setVmAvail({ status: "ok", data: await gateway.getVmAvailability(id) });
    } catch (e) {
      setVmAvail({ status: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }, []);
  useEffect(() => {
    if (target === "vm" && vmProviderId) void refreshVmAvail(vmProviderId);
    else setVmAvail({ status: "idle" });
  }, [target, vmProviderId, refreshVmAvail]);

  const fetchStatus = useCallback(async () => {
    if (!activeProviderId) return;
    try {
      const r = await fetch(
        `/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}/decoder/status?provider_id=${encodeURIComponent(activeProviderId)}`,
        { cache: "no-store" },
      );
      const j = await r.json();
      if (r.ok) setStatus(j as Status);
    } catch {
      /* transient; keep polling */
    }
  }, [datasetId, activeProviderId]);

  useEffect(() => {
    if (poll.current) clearInterval(poll.current);
    if (!activeProviderId || (target === "vm" && !vmProviderId)) {
      setStatus(null);
      return;
    }
    void fetchStatus();
    poll.current = setInterval(fetchStatus, 4000);
    return () => {
      if (poll.current) clearInterval(poll.current);
    };
  }, [activeProviderId, vmProviderId, target, fetchStatus]);

  const ready = !!status?.ready;
  const loading = !!status?.running && !ready;

  useEffect(() => {
    onState(ready && activeProviderId ? { providerId: activeProviderId, ready: true } : null);
  }, [ready, activeProviderId, onState]);

  function loadBody() {
    return {
      target,
      provider_id: target === "vm" ? vmProviderId : runpodProviderId || null,
      gpu: target === "vm" ? vmGpu : "auto",
      gpu_type: gpuType,
      gpu_count: gpuCount,
      secure_cloud: secureCloud,
      idle_timeout_s: 600,
    };
  }

  async function load() {
    if (target === "vm" && !vmProviderId) return setErr("Pick a VM provider, or switch to Default cloud.");
    setBusy(true);
    setErr(null);
    try {
      const r = await fetch(`/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}/decoder/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(loadBody()),
      });
      const j = await r.json();
      if (!r.ok) throw new Error((j && (j.detail || j.error)) || `load failed (${r.status})`);
      setStatus(j as Status);
      void fetchStatus();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function unload() {
    setBusy(true);
    setErr(null);
    try {
      await fetch(`/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}/decoder/unload`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider_id: activeProviderId }),
      });
      setStatus({ running: false, ready: false });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const lockPicker = busy || ready || loading;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">
          Audio decoder{" "}
          <span className="text-[11px] font-normal text-muted-foreground">
            NeuCodec, kept resident · idle auto-unloads after 10 min
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-xs text-muted-foreground">
          Packed rows are speech tokens, not audio. Load NeuCodec on a GPU below, then press ▶ on any
          utterance to decode it back to audio. The model stays loaded so each play is instant.
        </p>

        {/* Run on — RunPod pod vs registered VM (mirrors the Transform card) */}
        <div>
          <Label className="text-xs font-medium">Run on</Label>
          <div className="mt-1.5 grid grid-cols-1 gap-2 sm:grid-cols-2">
            <button
              type="button"
              onClick={() => setTarget("cloud")}
              disabled={lockPicker}
              className={cn(
                "flex items-start gap-2.5 rounded-md border px-3 py-2 text-left text-sm transition-colors disabled:opacity-60",
                target === "cloud" ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40",
              )}
            >
              <Cpu className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
              <div className="min-w-0">
                <div className="font-medium">Default cloud (RunPod)</div>
                <div className="text-xs text-muted-foreground">Spawn a pod. Cold-installs the codec on first load.</div>
              </div>
            </button>
            <button
              type="button"
              onClick={() => setTarget("vm")}
              disabled={lockPicker}
              className={cn(
                "flex items-start gap-2.5 rounded-md border px-3 py-2 text-left text-sm transition-colors disabled:opacity-60",
                target === "vm" ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40",
              )}
            >
              <Server className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
              <div className="min-w-0">
                <div className="font-medium">Bare metal (VM)</div>
                <div className="text-xs text-muted-foreground">SSH onto a registered VM. Instant — TTS venv present.</div>
              </div>
            </button>
          </div>
        </div>

        {target === "vm" ? (
          <div className="space-y-1.5">
            <Label className="text-xs">VM provider</Label>
            {vmProviders.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                No VM providers registered. Add one at{" "}
                <a href="/providers/new" className="underline underline-offset-2 hover:text-foreground">GPU Providers → New provider</a>.
              </p>
            ) : (
              <Select value={vmProviderId} onValueChange={setVmProviderId} disabled={lockPicker}>
                <SelectTrigger className="text-xs"><SelectValue placeholder="Pick a VM…" /></SelectTrigger>
                <SelectContent>
                  {vmProviders.map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      {p.name}{p.gpu_count ? ` · ${p.gpu_count} GPU` : ""}{p.host ? ` · ${p.host}` : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            {vmProviderId && vmAvail.status === "loading" && (
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" /> checking GPUs via SSH…
              </div>
            )}
            {vmProviderId && vmAvail.status === "error" && (
              <div className="flex items-center justify-between gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-2.5 py-1.5 text-xs text-destructive">
                <span className="truncate" title={vmAvail.message}>{vmAvail.message}</span>
                <button type="button" onClick={() => refreshVmAvail(vmProviderId)} className="inline-flex shrink-0 items-center gap-1 underline-offset-2 hover:underline">
                  <RefreshCw className="h-3 w-3" /> retry
                </button>
              </div>
            )}
            {vmProviderId && vmAvail.status === "ok" && vmAvail.data && !vmAvail.data.ok && (
              <p className="text-xs text-amber-600 dark:text-amber-400">{vmAvail.data.message}</p>
            )}
            {vmProviderId && vmAvail.status === "ok" && vmAvail.data?.ok && (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label className="text-xs">GPU on this VM</Label>
                  <button type="button" onClick={() => refreshVmAvail(vmProviderId)} className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground">
                    <RefreshCw className="h-3 w-3" /> refresh utilization
                  </button>
                </div>
                <div className="space-y-1">
                  <button
                    type="button"
                    onClick={() => setVmGpu("auto")}
                    disabled={lockPicker}
                    className={cn(
                      "flex w-full items-center justify-between rounded-md border px-2.5 py-1.5 text-left text-xs transition-colors disabled:opacity-60",
                      vmGpu === "auto" ? "border-primary/60 bg-primary/5" : "border-border hover:bg-muted/40",
                    )}
                  >
                    <span>Auto — pick the most-free GPU at load</span>
                    {vmGpu === "auto" && <Check className="h-3.5 w-3.5 text-primary" />}
                  </button>
                  {vmAvail.data.gpus.map((g) => {
                    const busy = g.mem_free_mib < g.mem_total_mib * 0.2 || g.util_pct > 50;
                    return (
                      <button
                        key={g.index}
                        type="button"
                        onClick={() => setVmGpu(String(g.index))}
                        disabled={lockPicker}
                        className={cn(
                          "flex w-full items-center justify-between gap-2 rounded-md border px-2.5 py-1.5 text-left transition-colors disabled:opacity-60",
                          vmGpu === String(g.index) ? "border-primary/60 bg-primary/5" : "border-border hover:bg-muted/40",
                        )}
                      >
                        <span className="font-mono text-xs">
                          #{g.index} {g.name.replace(/^NVIDIA\s+/, "")}
                        </span>
                        <span className="flex items-center gap-2">
                          <span className={cn("font-mono text-[10px]", busy ? "text-amber-600 dark:text-amber-400" : "text-muted-foreground")}>
                            {fmtMib(g.mem_free_mib)}/{fmtMib(g.mem_total_mib)} free · {g.util_pct}% util
                            {busy ? <AlertTriangle className="ml-1 inline h-3 w-3" /> : null}
                          </span>
                          {vmGpu === String(g.index) && <Check className="h-3.5 w-3.5 shrink-0 text-primary" />}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            <div className="space-y-1.5">
              <Label className="text-xs">RunPod account</Label>
              <Select
                value={runpodProviderId || "__default__"}
                onValueChange={(v) => setRunpodProviderId(v === "__default__" ? "" : v)}
                disabled={lockPicker}
              >
                <SelectTrigger className="text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__default__">Gateway default (RunPod)</SelectItem>
                  {runpodProviders.map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      {p.name}{p.api_key_last4 ? ` · ****${p.api_key_last4}` : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label className="text-xs">GPU</Label>
                <AvailabilityBadge state={availability} count={gpuCount} />
              </div>
              <div className="flex gap-2">
                <SearchableSelect
                  className="flex-1"
                  value={gpuType}
                  onChange={setGpuType}
                  options={gpuOptions.map((g) => ({ value: g.id, label: g.label, hint: gpuHint(g.vram_gb, 1) }))}
                  placeholder="Choose a GPU"
                  searchPlaceholder="Search GPUs (e.g. l40s, 24gb)…"
                />
                <Select value={String(gpuCount)} onValueChange={(v) => setGpuCount(Number.parseInt(v, 10))} disabled={lockPicker}>
                  <SelectTrigger className="w-20 shrink-0"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {GPU_COUNT_CHOICES.map((n) => <SelectItem key={n} value={String(n)}>×{n}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="space-y-1.5 sm:max-w-[12rem]">
              <Label className="text-xs">Cloud tier</Label>
              <Select value={secureCloud ? "secure" : "community"} onValueChange={(v) => setSecureCloud(v === "secure")} disabled={lockPicker}>
                <SelectTrigger className="text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="secure">Secure</SelectItem>
                  <SelectItem value="community">Community</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        )}

        <div className="flex flex-wrap items-center gap-3">
          {!ready ? (
            <Button size="sm" onClick={load} disabled={busy || loading || (target === "vm" && !vmProviderId)}>
              {busy || loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              {loading ? "loading…" : "Load decoder"}
            </Button>
          ) : (
            <Button size="sm" variant="outline" onClick={unload} disabled={busy}>
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Square className="h-3.5 w-3.5" />} Unload
            </Button>
          )}
          {ready && (
            <span className="inline-flex items-center gap-1.5 text-xs text-status-active">
              <span className="h-1.5 w-1.5 rounded-full bg-current" /> ready{status?.device ? ` · ${status.device}` : ""}
            </span>
          )}
          {loading && <span className="text-xs text-muted-foreground">loading NeuCodec…</span>}
        </div>
        {err && <p className="text-xs text-destructive">{err}</p>}
        {loading && status?.logs?.length ? (
          <pre className="max-h-32 overflow-auto rounded bg-muted/40 p-2 font-mono text-[10px] leading-relaxed scrollbar-thin">
            {status.logs.join("\n")}
          </pre>
        ) : null}
      </CardContent>
    </Card>
  );
}
