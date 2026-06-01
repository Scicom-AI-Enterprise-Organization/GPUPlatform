"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  Boxes,
  Check,
  Cpu,
  Loader2,
  RefreshCw,
  Server,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NumberField } from "@/components/ui/number-field";
import { SearchableSelect } from "@/components/ui/searchable-select";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AvailabilityBadge } from "@/components/availability-badge";
import { ProgressEta } from "@/components/progress-eta";
import { useGpuAvailability } from "@/lib/use-gpu-availability";
import { cn } from "@/lib/utils";
import { gateway, GatewayError } from "@/lib/gateway";
import type {
  GpuTypeOption,
  ProviderRecord,
  StorageRecord,
  VmAvailability,
} from "@/lib/types";

const GPU_COUNT_CHOICES = [1, 2, 4, 8] as const;

// Fallback until the live catalog (/compute/runpod/gpu-types) lands.
const RUNPOD_GPU_FALLBACK: GpuTypeOption[] = [
  { id: "NVIDIA RTX A5000", label: "RTX A5000", vram_gb: 24, hint: "24 GB" },
  { id: "NVIDIA RTX A6000", label: "RTX A6000", vram_gb: 48, hint: "48 GB" },
  { id: "NVIDIA L40S", label: "L40S", vram_gb: 48, hint: "48 GB" },
  { id: "NVIDIA A100 80GB PCIe", label: "A100 80GB", vram_gb: 80, hint: "datacenter" },
  { id: "NVIDIA H100 80GB HBM3", label: "H100 80GB", vram_gb: 80, hint: "fastest" },
];

function gpuHint(vramGb: number, count: number): string {
  const total = vramGb * count;
  return `${total >= 100 ? Math.round(total) : total} GB VRAM${count > 1 ? ` · ×${count}` : ""}`;
}

function errText(body: unknown, fallback: string): string {
  if (typeof body === "string") return body || fallback;
  if (body && typeof body === "object") {
    const d = (body as Record<string, unknown>).detail;
    if (typeof d === "string") return d;
  }
  return fallback;
}

type VmAvailState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ok"; data: VmAvailability }
  | { status: "error"; message: string };

// NeuCodec-encode + multipack a {audio, transcription} dataset into a ChiniDataset
// on a GPU. "Run on" mirrors Autotrain/Benchmark: a fresh RunPod pod (pick GPU
// type/count/tier) or a registered bare-metal VM. Output is a new packed dataset
// TTS training streams directly (skips convert+pack per run).
export function TtsPackCard({
  datasetId,
  s3Storages,
  initialStatus,
  initialLog,
  bare = false,
}: {
  datasetId: string;
  s3Storages: StorageRecord[];
  initialStatus: string | null;
  initialLog: string | null;
  bare?: boolean;
}) {
  const router = useRouter();
  const [providers, setProviders] = useState<ProviderRecord[]>([]);
  const [storageId, setStorageId] = useState(s3Storages[0]?.id ?? "");
  const [seqLen, setSeqLen] = useState(4096);
  // Isolated uv venv for the NeuCodec/TTS deps (mirrors Autotrain). Reused +
  // cached across packs. Dedicated to NeuCodec by default.
  const [venvPath, setVenvPath] = useState("/share/neucodec-tts");

  // Run-on (pod card)
  const [target, setTarget] = useState<"cloud" | "vm">("cloud");
  const [vmProviderId, setVmProviderId] = useState("");
  const [runpodProviderId, setRunpodProviderId] = useState("");
  const [gpuType, setGpuType] = useState("NVIDIA L40S");
  const [gpuCount, setGpuCount] = useState(1);
  const [secureCloud, setSecureCloud] = useState(true);
  const [diskGb, setDiskGb] = useState(60);
  const [volumeGb, setVolumeGb] = useState(80);
  const [visibleDevices, setVisibleDevices] = useState("");
  const [gpuOptions, setGpuOptions] = useState<GpuTypeOption[]>(RUNPOD_GPU_FALLBACK);

  const [status, setStatus] = useState<string | null>(initialStatus);
  const [log, setLog] = useState<string | null>(initialLog);
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const poll = useRef<ReturnType<typeof setInterval> | null>(null);
  const logRef = useRef<HTMLPreElement | null>(null);

  const [vmAvail, setVmAvail] = useState<VmAvailState>({ status: "idle" });
  const refreshVmAvail = useCallback(async (id: string) => {
    if (!id) return setVmAvail({ status: "idle" });
    setVmAvail({ status: "loading" });
    try {
      setVmAvail({ status: "ok", data: await gateway.getVmAvailability(id) });
    } catch (e) {
      setVmAvail({ status: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }, []);

  const running = status === "running";
  const newDatasetId =
    status === "done" ? log?.match(/created dataset (ds-[0-9a-f]+)/i)?.[1] : undefined;

  const vmProviders = useMemo(() => providers.filter((p) => p.kind === "vm"), [providers]);
  const runpodProviders = useMemo(() => providers.filter((p) => p.kind === "runpod"), [providers]);
  const gpuBound = useMemo(
    () => (target === "vm" ? vmProviders.find((p) => p.id === vmProviderId)?.gpu_count ?? 0 : gpuCount),
    [target, vmProviders, vmProviderId, gpuCount],
  );
  const availability = useGpuAvailability(
    gpuType, gpuCount, target === "cloud", secureCloud ? "SECURE" : "COMMUNITY",
  );
  // Live validation of the GPU pin (shown inline under the field as you type),
  // mirroring the Autotrain form: non-negative integers, no dupes, within range.
  const vdError = useMemo(() => {
    const vd = visibleDevices.trim();
    if (!vd) return null;
    const toks = vd.split(",").map((t) => t.trim()).filter(Boolean);
    for (const t of toks) if (!/^\d+$/.test(t)) return `"${t}" isn't a valid GPU index — use non-negative integers like 0,1.`;
    const ids = toks.map(Number);
    if (new Set(ids).size !== ids.length) return "Duplicate GPU indices.";
    if (gpuBound > 0) {
      const oob = [...new Set(ids.filter((i) => i >= gpuBound))].sort((a, b) => a - b);
      if (oob.length) return `GPU ${oob.join(", ")} out of range — valid indices are 0–${gpuBound - 1}.`;
    }
    return null;
  }, [visibleDevices, gpuBound]);

  useEffect(() => {
    gateway.listProviders().then(setProviders).catch(() => {});
    gateway
      .listRunpodGpuTypes()
      .then((rows) => {
        if (!rows.length) return;
        setGpuOptions(rows);
        setGpuType((cur) => (rows.some((g) => g.id === cur) ? cur : rows[0].id));
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    // refresh (or reset) the VM availability check; both setState synchronously
    // off-render, which is intended here (mirrors the Autotrain form).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (target === "vm" && vmProviderId) refreshVmAvail(vmProviderId);
    else setVmAvail({ status: "idle" });
  }, [target, vmProviderId, refreshVmAvail]);

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
    }, 4000);
    poll.current = id;
    return () => clearInterval(id);
  }, [running, datasetId, router]);

  async function run() {
    setErr(null);
    if (!storageId) return setErr("Pick an S3 storage for the packed shards.");
    if (target === "vm" && !vmProviderId) return setErr("Pick a VM provider, or switch to Default cloud.");
    if (vdError) return setErr(vdError);
    const vd = visibleDevices.trim();
    setStarting(true);
    try {
      const d = await gateway.packTtsDataset(datasetId, {
        provider_id: target === "vm" ? vmProviderId : runpodProviderId || null,
        storage_id: storageId,
        sequence_length: seqLen,
        venv_path: venvPath.trim() || null,
        gpu_count: gpuCount,
        gpu_type: gpuType,
        secure_cloud: secureCloud,
        disk_gb: diskGb,
        volume_gb: volumeGb,
        visible_devices: vd || null,
      });
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
      Encode the audio to <span className="font-mono">NeuCodec</span> speech tokens and multipack into a{" "}
      <span className="font-mono">ChiniDataset</span> (sequence length {seqLen}), then upload the shards to S3.
      TTS training streams the packed dataset directly, skipping convert+pack per run.
    </span>
  );

  const body = (
    <div className="space-y-4">
      {/* Run on — RunPod pod vs registered VM (mirrors Autotrain / Benchmark) */}
      <div>
        <Label className="text-xs font-medium">Run on</Label>
        <div className="mt-1.5 grid grid-cols-1 gap-2 sm:grid-cols-2">
          <button type="button" onClick={() => setTarget("cloud")} disabled={running}
            className={cn("flex items-start gap-2.5 rounded-md border px-3 py-2 text-left text-sm transition-colors",
              target === "cloud" ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40")}>
            <Cpu className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <div className="font-medium">Default cloud (RunPod)</div>
              <div className="text-xs text-muted-foreground">Spawn a fresh pod. Pay-per-second.</div>
            </div>
          </button>
          <button type="button" onClick={() => setTarget("vm")} disabled={running}
            className={cn("flex items-start gap-2.5 rounded-md border px-3 py-2 text-left text-sm transition-colors",
              target === "vm" ? "border-primary/60 bg-primary/5" : "border-border hover:border-primary/40 hover:bg-muted/40")}>
            <Server className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <div className="font-medium">Bare metal (VM)</div>
              <div className="text-xs text-muted-foreground">SSH onto a registered VM.</div>
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
            <Select value={vmProviderId} onValueChange={setVmProviderId} disabled={running}>
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
          {vmProviderId && <VmAvailabilityRow state={vmAvail} onRefresh={() => refreshVmAvail(vmProviderId)} />}
        </div>
      ) : (
        <div className="space-y-3">
          <div className="space-y-1.5">
            <Label className="text-xs">RunPod account</Label>
            <Select value={runpodProviderId || "__default__"}
              onValueChange={(v) => setRunpodProviderId(v === "__default__" ? "" : v)} disabled={running}>
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
                searchPlaceholder="Search GPUs (e.g. h100, 24gb)…"
              />
              <Select value={String(gpuCount)} onValueChange={(v) => setGpuCount(Number.parseInt(v, 10))}>
                <SelectTrigger className="w-20 shrink-0"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {GPU_COUNT_CHOICES.map((n) => <SelectItem key={n} value={String(n)}>×{n}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs">Cloud tier</Label>
              <Select value={secureCloud ? "secure" : "community"}
                onValueChange={(v) => setSecureCloud(v === "secure")} disabled={running}>
                <SelectTrigger className="text-xs"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="secure">Secure</SelectItem>
                  <SelectItem value="community">Community</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Disk (GB)</Label>
              <NumberField min={20} value={diskGb} onChange={setDiskGb} />
            </div>
            <div className="space-y-1.5">
              <Label className="text-xs">Volume (GB)</Label>
              <NumberField min={0} value={volumeGb} onChange={setVolumeGb} />
            </div>
          </div>
        </div>
      )}

      {/* Output + packing knobs */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label className="text-xs">S3 storage (packed shards)</Label>
          <Select value={storageId} onValueChange={setStorageId} disabled={running}>
            <SelectTrigger className="text-xs">
              <SelectValue placeholder={s3Storages.length ? "Pick a storage" : "No S3 storage"} />
            </SelectTrigger>
            <SelectContent>
              {s3Storages.map((s) => <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label className="text-xs">Sequence length (multipack)</Label>
          <NumberField min={256} value={seqLen} onChange={setSeqLen} />
        </div>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label className="text-xs">uv venv path</Label>
          <Input className="font-mono text-xs" placeholder="/share/neucodec-tts"
            value={venvPath} onChange={(e) => setVenvPath(e.target.value)} disabled={running} />
          <p className="text-[11px] text-muted-foreground">
            Isolated <span className="font-mono">uv</span> venv for the NeuCodec deps — reused + cached across
            packs. Put it on a big disk like <span className="font-mono">/share</span>.
          </p>
        </div>
        <div className="space-y-1.5">
          <Label className="text-xs">CUDA_VISIBLE_DEVICES (optional)</Label>
          <Input className={cn("font-mono text-xs", vdError && "border-destructive focus-visible:ring-destructive")}
            placeholder="e.g. 6,7 (empty = all GPUs)"
            value={visibleDevices} onChange={(e) => setVisibleDevices(e.target.value)} disabled={running} />
          {vdError ? (
            <p className="text-[11px] text-destructive">{vdError}</p>
          ) : (
            <p className="text-[11px] text-muted-foreground">
              Pins which GPUs the pack uses. Empty = all visible GPUs.
              {gpuBound > 0 && (
                <> {target === "vm" ? "This VM" : "The pod"} has {gpuBound} GPU{gpuBound === 1 ? "" : "s"} — valid indices <span className="font-mono">0–{gpuBound - 1}</span>.</>
              )}
            </p>
          )}
        </div>
      </div>
      <p className="text-[11px] text-muted-foreground">
        Speech tokenizer is fixed — all Scicom TTS models share the NeuCodec speech-token vocab.
      </p>
    </div>
  );

  const logBlock = log ? (
    <div className="space-y-1">
      <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
        {running && <Loader2 className="h-3 w-3 animate-spin" />}
        <span>{running ? "Live log (NeuCodec encode + multipack on the GPU box)" : "Log"}</span>
        <ProgressEta log={log} running={running} />
      </div>
      <pre ref={logRef} className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-200 scrollbar-thin">
        {log}
      </pre>
    </div>
  ) : null;

  // The primary action lives in its own footer row (outside the form card when
  // standalone): status / result link on the left, Pack + Cancel on the right.
  const actions = (
    <div className="space-y-3">
      {err && <p className="text-sm text-destructive">{err}</p>}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          {status && status !== "running" && (
            <span className={status === "done" ? "text-sm text-emerald-600 dark:text-emerald-400" : "text-sm text-destructive"}>
              {status === "done" ? "✓ done" : `✕ ${status}`}
            </span>
          )}
          {newDatasetId && (
            <Link href={`/datasets/${newDatasetId}`} className="inline-flex items-center gap-1 text-sm text-primary hover:underline">
              Open packed dataset <span className="font-mono text-xs">{newDatasetId}</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          )}
        </div>
        <div className="flex items-center gap-3">
          {running && (
            <Button variant="outline" onClick={cancel} disabled={cancelling} className="text-destructive">
              {cancelling ? <Loader2 className="h-4 w-4 animate-spin" /> : <X className="h-4 w-4" />}
              {cancelling ? "Cancelling…" : "Cancel"}
            </Button>
          )}
          <Button onClick={run} disabled={running || starting}>
            {running || starting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Boxes className="h-4 w-4" />}
            {running ? "Packing…" : "Pack for TTS"}
          </Button>
        </div>
      </div>
    </div>
  );

  // Embedded in the hf/label transform tabs: no card, action row inline.
  if (bare)
    return (
      <div className="space-y-3">
        {desc}
        {body}
        {actions}
        {logBlock}
      </div>
    );
  // s3 / upload: settings + log inside the card, with the Pack button below it.
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="flex flex-col gap-0.5">
          <CardTitle className="text-base">Pack for TTS — NeuCodec + multipack</CardTitle>
          {desc}
        </CardHeader>
        <CardContent className="space-y-4">
          {body}
          {logBlock}
        </CardContent>
      </Card>
      {actions}
    </div>
  );
}

function VmAvailabilityRow({ state, onRefresh }: { state: VmAvailState; onRefresh: () => void }) {
  if (state.status === "idle") return null;
  if (state.status === "loading") {
    return (
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" /> Checking availability via SSH…
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="flex items-center justify-between gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-2.5 py-1.5 text-xs text-destructive">
        <span className="inline-flex items-center gap-1.5 truncate">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" />
          <span className="truncate" title={state.message}>{state.message}</span>
        </span>
        <button type="button" onClick={onRefresh} className="inline-flex items-center gap-1 underline-offset-2 hover:underline">
          <RefreshCw className="h-3 w-3" /> Retry
        </button>
      </div>
    );
  }
  const { data } = state;
  if (!data.ok) {
    return (
      <div className="flex items-center justify-between gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-2.5 py-1.5 text-xs text-amber-700 dark:text-amber-400">
        <span className="inline-flex items-center gap-1.5 truncate">
          <X className="h-3.5 w-3.5 shrink-0" />
          <span className="truncate" title={data.message}>{data.message}</span>
        </span>
        <button type="button" onClick={onRefresh} className="inline-flex items-center gap-1 underline-offset-2 hover:underline">
          <RefreshCw className="h-3 w-3" /> Retry
        </button>
      </div>
    );
  }
  const totalFreeMib = data.gpus.reduce((s, g) => s + g.mem_free_mib, 0);
  const totalMib = data.gpus.reduce((s, g) => s + g.mem_total_mib, 0);
  const busy = data.gpus.filter((g) => g.mem_free_mib < g.mem_total_mib * 0.2 || g.util_pct > 50).length;
  const allFree = busy === 0;
  return (
    <div className={cn("space-y-1 rounded-md border px-2.5 py-1.5 text-xs",
      allFree ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
        : "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400")}>
      <div className="flex items-center justify-between gap-2">
        <span className="inline-flex items-center gap-1.5">
          {allFree ? <Check className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />}
          {data.gpus.length} GPU{data.gpus.length === 1 ? "" : "s"} · {fmtMib(totalFreeMib)} free / {fmtMib(totalMib)}
          {!allFree && ` · ${busy} busy`}
        </span>
        <button type="button" onClick={onRefresh} className="inline-flex items-center gap-1 underline-offset-2 hover:underline">
          <RefreshCw className="h-3 w-3" /> Refresh
        </button>
      </div>
      <div className="flex flex-col gap-0.5 font-mono text-[10px] text-muted-foreground">
        {data.gpus.map((g) => (
          <span key={g.index}>
            #{g.index} {g.name.replace(/^NVIDIA\s+/, "")} · {fmtMib(g.mem_free_mib)}/{fmtMib(g.mem_total_mib)} free · {g.util_pct}% util
          </span>
        ))}
      </div>
    </div>
  );
}

function fmtMib(mib: number): string {
  if (mib >= 1024) return `${(mib / 1024).toFixed(1)} GiB`;
  return `${mib} MiB`;
}
