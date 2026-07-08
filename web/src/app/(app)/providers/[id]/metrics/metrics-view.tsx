"use client";

import { useEffect, useRef, useState } from "react";
import { Activity, Cpu, MemoryStick, AlertCircle, Gauge, HardDrive, Loader2 } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
import { cn } from "@/lib/utils";
import type { ProviderBandwidth, ProviderMetrics, ProviderRecord } from "@/lib/types";

const POLL_CHOICES = [5, 10, 15, 30] as const; // seconds
const DEFAULT_POLL_S = 10;
const MAX_POINTS = 120; // rolling window (count) — in-memory only, not persisted

type HostPoint = { i: number; cpu: number; mem: number };
type GpuPoint = { i: number; util: number; mem: number; temp: number };

const gib = (mib: number) => (mib / 1024).toFixed(1);
// MB/s → "1.2 GB/s" / "430 MB/s" (0 = "—").
const mbps = (v: number) =>
  v <= 0 ? "—" : v >= 1000 ? `${(v / 1000).toFixed(2)} GB/s` : `${v.toFixed(0)} MB/s`;
// bytes → "1.42 TiB" / "729 GiB" / "12 MiB".
const fmtBytes = (b: number) =>
  b >= 1_099_511_627_776 ? `${(b / 1_099_511_627_776).toFixed(2)} TiB`
  : b >= 1_073_741_824 ? `${(b / 1_073_741_824).toFixed(0)} GiB`
  : b >= 1_048_576 ? `${(b / 1_048_576).toFixed(0)} MiB`
  : `${b} B`;
// htop-style core load colour: green (idle) → amber (busy) → red (saturated).
const coreColor = (p: number) => (p >= 85 ? "#ef4444" : p >= 50 ? "#f59e0b" : "#10b981");

function MiniChart({
  data,
  keys,
}: {
  data: Record<string, number>[];
  keys: { k: string; color: string; label: string }[];
}) {
  return (
    <div className="mt-2 h-28 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-border" />
          <XAxis dataKey="i" hide type="number" domain={["dataMin", "dataMax"]} />
          <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} stroke="currentColor" className="text-muted-foreground" width={32} />
          <RTooltip
            contentStyle={{ fontSize: 11, borderRadius: 8 }}
            labelFormatter={() => ""}
            formatter={(v, n) => {
              const lbl = keys.find((x) => x.k === n)?.label ?? String(n);
              return [`${Number(v).toFixed(0)}%`, lbl];
            }}
          />
          {keys.map((x) => (
            <Line key={x.k} type="monotone" dataKey={x.k} stroke={x.color} strokeWidth={2} dot={false} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ProviderMetricsView({ id, provider }: { id: string; provider: ProviderRecord | null }) {
  const [m, setM] = useState<ProviderMetrics | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [host, setHost] = useState<HostPoint[]>([]);
  const [gpuSeries, setGpuSeries] = useState<Record<number, GpuPoint[]>>({});
  const [pollSec, setPollSec] = useState<number>(DEFAULT_POLL_S);
  const [bw, setBw] = useState<ProviderBandwidth | null>(null);
  const [bwLoading, setBwLoading] = useState(false);
  const [bwErr, setBwErr] = useState<string | null>(null);
  const [killing, setKilling] = useState<Set<number>>(new Set());
  const [killNote, setKillNote] = useState<string | null>(null);
  // A pending kill is one-or-more pids plus a human label ("pid 123" / "all 11 processes on GPU #7").
  const [pendingKill, setPendingKill] = useState<{ pids: number[]; label: string } | null>(null);
  const tick = useRef(0);

  // Killing GPU processes is destructive (SIGKILL on the VM over SSH), so confirm via the themed
  // dialog: a Kill button opens it (requestKill / requestKillGpu), the dialog's confirm runs it.
  function requestKill(pid: number) {
    setKillNote(null);
    setPendingKill({ pids: [pid], label: `pid ${pid}` });
  }

  // "Kill all" for one GPU — every pid bound to it, in a single SSH session.
  function requestKillGpu(index: number, pids: number[]) {
    if (pids.length === 0) return;
    setKillNote(null);
    setPendingKill({ pids, label: `all ${pids.length} process${pids.length === 1 ? "" : "es"} on GPU #${index}` });
  }

  // Terminate the pending pid(s). On success, refresh metrics so the freed GPU shows immediately.
  async function confirmKill() {
    const pend = pendingKill;
    if (pend == null) return;
    const { pids } = pend;
    setPendingKill(null);
    setKilling((s) => { const n = new Set(s); pids.forEach((p) => n.add(p)); return n; });
    try {
      const r = pids.length === 1
        ? await gateway.killProviderPid(id, pids[0])
        : await gateway.killProviderPids(id, pids);
      setKillNote(`${pend.label}: ${r.message}`);
      try { setM(await gateway.getProviderMetrics(id)); } catch { /* next poll refreshes */ }
    } catch (e) {
      setKillNote(`${pend.label}: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setKilling((s) => { const n = new Set(s); pids.forEach((p) => n.delete(p)); return n; });
    }
  }

  // The poll loop reads the interval from a ref so changing it takes effect on
  // the next tick without restarting the loop or clearing the graphed history.
  const pollMsRef = useRef(DEFAULT_POLL_S * 1000);

  // Live poll → graph. Recursive setTimeout (not setInterval) so SSH round-trips
  // never overlap. Series live in component state only — nothing is persisted.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const r = await gateway.getProviderMetrics(id);
        if (cancelled) return;
        setM(r);
        setErr(r.ok ? null : r.message);
        const i = tick.current++;
        const memPct = r.mem_total_mib > 0 ? (r.mem_used_mib / r.mem_total_mib) * 100 : 0;
        setHost((p) => [...p, { i, cpu: Math.max(0, r.cpu_pct), mem: memPct }].slice(-MAX_POINTS));
        setGpuSeries((prev) => {
          const next: Record<number, GpuPoint[]> = { ...prev };
          for (const g of r.gpus) {
            const mp = g.mem_total_mib > 0 ? (g.mem_used_mib / g.mem_total_mib) * 100 : 0;
            next[g.index] = [...(next[g.index] ?? []), { i, util: g.util_pct, mem: mp, temp: g.temp_c }].slice(-MAX_POINTS);
          }
          return next;
        });
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) timer = setTimeout(poll, pollMsRef.current);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [id]);

  async function runBandwidth() {
    setBwLoading(true);
    setBwErr(null);
    try {
      const r = await gateway.getProviderBandwidth(id);
      setBw(r);
      if (!r.ok) setBwErr(r.message);
    } catch (e) {
      setBwErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBwLoading(false);
    }
  }

  const memPct = m && m.mem_total_mib > 0 ? (m.mem_used_mib / m.mem_total_mib) * 100 : 0;
  const gpus = m?.gpus ?? [];
  // Huawei Ascend boxes report through npu-smi: util is AICore%, memory is HBM,
  // and the cards show power + health instead of PCIe/NVLink.
  const isNpu = gpus.length > 0 && gpus[0].kind === "npu";
  const accel = isNpu ? "NPU" : "GPU";

  return (
    <div className="space-y-4">
      <div>
        <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight">
          <Activity className="h-5 w-5 text-muted-foreground" /> {provider?.name ?? id}
          <span className="rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
            VM
          </span>
        </h1>
        <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
          <span className="font-mono">{provider?.user ?? "root"}@{provider?.host ?? "—"}{provider?.port ? `:${provider.port}` : ""}</span>
          {provider?.jump_host && (
            <span className="font-mono text-xs" title="Reached via ProxyJump">via {provider.jump_host}</span>
          )}
          <span>·</span>
          <span className="inline-flex items-center gap-1">
            <span className={cn("h-1.5 w-1.5 rounded-full", err ? "bg-destructive" : "animate-pulse bg-emerald-500")} />
            {err ? "error" : "live"} · not stored
          </span>
          <span>·</span>
          <span className="inline-flex items-center gap-1.5">
            refresh
            <Select
              value={String(pollSec)}
              onValueChange={(v) => {
                const n = Number(v);
                setPollSec(n);
                pollMsRef.current = n * 1000; // next tick uses the new interval
              }}
            >
              <SelectTrigger className="h-7 w-[84px] text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                {POLL_CHOICES.map((s) => (
                  <SelectItem key={s} value={String(s)} className="text-xs">every {s}s</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </span>
        </div>
      </div>

      {err && (
        <div className="flex items-center gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <AlertCircle className="h-4 w-4 shrink-0" /> {err}
        </div>
      )}

      {!m && !err && <p className="text-sm text-muted-foreground">Connecting to the VM over SSH…</p>}

      {m && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Cpu className="h-4 w-4 text-emerald-600 dark:text-emerald-400" /> CPU
                <span className="ml-auto font-mono text-base font-semibold tabular-nums">
                  {m.cpu_pct >= 0 ? `${m.cpu_pct.toFixed(0)}%` : "—"}
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <MiniChart data={host} keys={[{ k: "cpu", color: "#10b981", label: "CPU" }]} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <MemoryStick className="h-4 w-4 text-sky-600 dark:text-sky-400" /> Memory
                <span className="ml-auto font-mono text-sm tabular-nums text-muted-foreground">
                  {gib(m.mem_used_mib)} / {gib(m.mem_total_mib)} GiB · <span className="text-foreground">{memPct.toFixed(0)}%</span>
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <MiniChart data={host} keys={[{ k: "mem", color: "#0ea5e9", label: "RAM" }]} />
            </CardContent>
          </Card>
        </div>
      )}

      {m && m.disks && m.disks.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <HardDrive className="h-4 w-4 text-amber-600 dark:text-amber-400" /> Disk
              <span className="ml-auto text-xs font-normal text-muted-foreground">
                {m.disks.length} filesystem{m.disks.length === 1 ? "" : "s"}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {m.disks.map((d) => {
              const pct = d.total_bytes > 0 ? (d.used_bytes / d.total_bytes) * 100 : 0;
              const full = pct >= 90;
              return (
                <div key={d.mount}>
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-mono">{d.mount}</span>
                    <span className="font-mono tabular-nums text-muted-foreground">
                      {fmtBytes(d.used_bytes)} / {fmtBytes(d.total_bytes)} ·{" "}
                      <span className={cn(full ? "text-destructive" : "text-foreground")}>{pct.toFixed(0)}%</span>
                    </span>
                  </div>
                  <div className="mt-1 h-2 overflow-hidden rounded bg-muted">
                    <div
                      className={cn("h-full rounded", full ? "bg-destructive" : "bg-amber-500")}
                      style={{ width: `${Math.min(100, pct)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>
      )}

      {m && m.cpu_cores.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">
              Per-core CPU <span className="text-[11px] font-normal text-muted-foreground">· {m.cpu_cores.length} cores</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className="grid gap-1"
              style={{ gridTemplateColumns: "repeat(auto-fill, minmax(56px, 1fr))" }}
            >
              {m.cpu_cores.map((p, i) => (
                <div key={i} title={`core ${i}: ${p.toFixed(0)}%`} className="rounded border border-border/60 px-1 py-0.5">
                  <div className="flex justify-between font-mono text-[9px] leading-none text-muted-foreground">
                    <span>{i}</span>
                    <span>{p.toFixed(0)}</span>
                  </div>
                  <div className="mt-0.5 h-1 overflow-hidden rounded bg-muted">
                    <div className="h-full rounded" style={{ width: `${Math.min(100, Math.max(0, p))}%`, backgroundColor: coreColor(p) }} />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {m && m.host_gpu_procs && m.host_gpu_procs.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              GPU processes
              <span className="text-[11px] font-normal text-muted-foreground">· via /proc · {m.host_gpu_procs.length}</span>
            </CardTitle>
            <p className="mt-1 text-[11px] text-muted-foreground">
              Commands of GPU processes seen on this host (including other tenants). On a
              shared / containerized box these can&apos;t be pinned to a specific GPU — match
              them to the per-GPU VRAM below by command.
            </p>
          </CardHeader>
          <CardContent className="space-y-1">
            {m.host_gpu_procs.map((p) => {
              const busy = killing.has(p.pid);
              return (
                <div key={p.pid} className="flex items-start justify-between gap-2 text-xs" title={p.cmd}>
                  <span className="shrink-0 text-foreground">pid {p.pid}</span>
                  {p.gpus ? (
                    <span className="shrink-0 font-mono text-[10px] text-muted-foreground" title="GPU device nodes held">gpu {p.gpus}</span>
                  ) : null}
                  <span className="min-w-0 flex-1 truncate text-left font-mono text-[11px] text-muted-foreground">
                    {p.cmd || p.comm || "?"}
                  </span>
                  <button
                    type="button"
                    onClick={() => requestKill(p.pid)}
                    disabled={busy}
                    title={`SIGKILL pid ${p.pid}`}
                    className="shrink-0 rounded border border-destructive/40 px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-destructive transition-colors hover:bg-destructive/10 disabled:opacity-50"
                  >
                    {busy ? "…" : "Kill"}
                  </button>
                </div>
              );
            })}
          </CardContent>
        </Card>
      )}

      {gpus.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">
              {accel}s <span className="text-[11px] font-normal text-muted-foreground">· {gpus.length} × {gpus[0].name.replace(/^NVIDIA\s+/, "")}</span>
            </CardTitle>
            {killNote && (
              <p className="mt-1 font-mono text-[11px] text-muted-foreground" title="kill result">{killNote}</p>
            )}
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 lg:grid-cols-2">
              {gpus.map((g) => {
                const mp = g.mem_total_mib > 0 ? (g.mem_used_mib / g.mem_total_mib) * 100 : 0;
                const procCount = g.processes?.length ?? 0;
                const gpuBusy = (g.processes ?? []).some((p) => killing.has(p.pid));
                return (
                  <div key={g.index} className="rounded-lg border border-border p-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate font-mono text-xs font-medium">
                        #{g.index} {g.name.replace(/^NVIDIA\s+/, "")}
                      </span>
                      {procCount > 0 && (
                        <button
                          type="button"
                          onClick={() => requestKillGpu(g.index, g.processes!.map((p) => p.pid))}
                          disabled={gpuBusy}
                          title={`SIGKILL all ${procCount} process${procCount === 1 ? "" : "es"} on GPU #${g.index}`}
                          className="shrink-0 rounded border border-destructive/40 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wide text-destructive transition-colors hover:bg-destructive/10 disabled:opacity-50"
                        >
                          {gpuBusy ? "…" : `Kill all (${procCount})`}
                        </button>
                      )}
                    </div>
                    <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px]">
                      <span className="text-emerald-600 dark:text-emerald-400">
                        {g.util_pct}% {isNpu ? "AICore" : "util"}
                      </span>
                      <span className="text-sky-600 dark:text-sky-400">
                        {mp.toFixed(0)}% {isNpu ? "HBM" : "mem"} · {gib(g.mem_used_mib)}/{gib(g.mem_total_mib)} GiB
                      </span>
                      <span className="text-amber-600 dark:text-amber-400">{g.temp_c}°C</span>
                    </div>
                    {isNpu ? (
                      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-muted-foreground">
                        <span title="Board power draw (npu-smi)">
                          Power <span className="text-foreground">{g.power_w ? `${g.power_w.toFixed(0)} W` : "—"}</span>
                        </span>
                        <span title="npu-smi health status">
                          Health{" "}
                          <span className={g.health && g.health !== "OK" ? "text-destructive" : "text-foreground"}>
                            {g.health || "—"}
                          </span>
                        </span>
                      </div>
                    ) : (
                      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-muted-foreground">
                        {g.pcie_gen_cur ? (
                          <span title="PCIe link — current (live; downclocks at idle) vs max">
                            PCIe{" "}
                            <span className="text-foreground">Gen{g.pcie_gen_cur} ×{g.pcie_width_cur}</span>
                            {g.pcie_gen_max ? <> / max Gen{g.pcie_gen_max} ×{g.pcie_width_max}</> : null}
                          </span>
                        ) : (
                          <span title="PCIe link unavailable">PCIe —</span>
                        )}
                        {g.nvlink_supported ? (
                          <span title="NVLink — active links · aggregate per-direction bandwidth">
                            NVLink{" "}
                            <span className="text-foreground">{g.nvlink_active}× · {g.nvlink_gbps} GB/s</span>
                          </span>
                        ) : (
                          <span title="No NVLink on this GPU (PCIe-only)">NVLink —</span>
                        )}
                      </div>
                    )}
                    <MiniChart
                      data={gpuSeries[g.index] ?? []}
                      keys={[
                        { k: "util", color: "#10b981", label: "util" },
                        { k: "mem", color: "#0ea5e9", label: "mem" },
                      ]}
                    />
                    <div className="mt-2 border-t border-border pt-1.5 font-mono text-[10px]">
                      {(g.processes?.length ?? 0) === 0 ? (
                        <span className="text-muted-foreground">no processes</span>
                      ) : (
                        <div className="space-y-0.5">
                          {g.processes!.map((p) => {
                            const model = p.cmd.match(/--model[=\s]+(\S+)/)?.[1];
                            const busy = killing.has(p.pid);
                            // Show the full command line (what actually identifies the
                            // process, e.g. `… vllm serve …`), not just comm ("python3").
                            // Lead with the model when it's a vLLM serve; tooltip has the rest.
                            const desc = p.cmd?.trim() || p.comm || "?";
                            return (
                              <div key={p.pid} className="flex items-start justify-between gap-2" title={p.cmd || p.comm}>
                                <span className="shrink-0 text-foreground">pid {p.pid}</span>
                                {p.gpu_mem_mib ? (
                                  <span className="shrink-0 font-mono text-[11px] text-foreground" title="VRAM held on this GPU">
                                    {p.gpu_mem_mib >= 1024 ? `${(p.gpu_mem_mib / 1024).toFixed(1)} GiB` : `${p.gpu_mem_mib} MiB`}
                                  </span>
                                ) : null}
                                <span className="min-w-0 flex-1 truncate text-left font-mono text-[11px] text-muted-foreground">
                                  {model ? <span className="text-foreground">{model}</span> : null}
                                  {model ? "  " : null}
                                  {desc}
                                </span>
                                <button
                                  type="button"
                                  onClick={() => requestKill(p.pid)}
                                  disabled={busy}
                                  title={`SIGKILL pid ${p.pid} to free this GPU`}
                                  className="shrink-0 rounded border border-destructive/40 px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-destructive transition-colors hover:bg-destructive/10 disabled:opacity-50"
                                >
                                  {busy ? "…" : "Kill"}
                                </button>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="mt-2 text-[10px] text-muted-foreground">
              <span className="text-emerald-600 dark:text-emerald-400">■</span> {isNpu ? "AICore %" : "util %"} ·{" "}
              <span className="text-sky-600 dark:text-sky-400">■</span> {isNpu ? "HBM %" : "memory %"} — temperature shown live above (°C).
            </p>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-sm">
            <Gauge className="h-4 w-4 text-violet-600 dark:text-violet-400" /> Bandwidth
            <span className="text-[11px] font-normal text-muted-foreground">· disk · memory · CPU</span>
            <Button
              size="sm"
              variant="outline"
              className="ml-auto h-7 text-xs"
              onClick={runBandwidth}
              disabled={bwLoading}
            >
              {bwLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Gauge className="h-3.5 w-3.5" />}
              {bwLoading ? "Running…" : bw ? "Re-run test" : "Run test"}
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {bwErr && (
            <div className="mb-3 flex items-center gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertCircle className="h-4 w-4 shrink-0" /> {bwErr}
            </div>
          )}
          {!bw && !bwLoading && !bwErr ? (
            <p className="text-xs text-muted-foreground">
              One-shot benchmark — writes a ~512&nbsp;MiB temp file (disk read/write),
              a sequential memory copy, and reads the CPU clock. Takes a few seconds; runs
              only when you click <span className="font-medium text-foreground">Run test</span>.
            </p>
          ) : (
            <div className="grid gap-3 sm:grid-cols-3">
              <BwStat
                icon={<HardDrive className="h-3.5 w-3.5 text-amber-600 dark:text-amber-400" />}
                label="Disk"
                rows={[
                  ["write", mbps(bw?.disk_write_mbps ?? 0)],
                  ["read", mbps(bw?.disk_read_mbps ?? 0)],
                ]}
              />
              <BwStat
                icon={<MemoryStick className="h-3.5 w-3.5 text-sky-600 dark:text-sky-400" />}
                label="Memory (sequential)"
                rows={[["copy", mbps(bw?.mem_mbps ?? 0)]]}
              />
              <BwStat
                icon={<Cpu className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />}
                label="CPU"
                rows={[
                  ["clock", bw?.cpu_mhz ? `${(bw.cpu_mhz / 1000).toFixed(2)} GHz` : "—"],
                  ["model", bw?.cpu_model || "—"],
                ]}
              />
            </div>
          )}
          {bw && (
            <p className="mt-2 text-[10px] text-muted-foreground">
              Disk read uses O_DIRECT when supported (else cached). Memory figure is a
              <span className="font-mono"> dd</span> sequential copy — a rough proxy, not STREAM.
            </p>
          )}
        </CardContent>
      </Card>

      <Dialog open={pendingKill !== null} onOpenChange={(o) => { if (!o) setPendingKill(null); }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{(pendingKill?.pids.length ?? 0) > 1 ? "Terminate processes?" : "Terminate process?"}</DialogTitle>
            <DialogDescription>
              SIGKILL <span className="font-mono text-foreground">{pendingKill?.label}</span> on this VM to
              free the GPU{(pendingKill?.pids.length ?? 0) > 1 ? "" : " it holds"}. They are killed immediately and cannot be resumed.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPendingKill(null)}>Cancel</Button>
            <Button variant="destructive" onClick={confirmKill}>Kill{(pendingKill?.pids.length ?? 0) > 1 ? " all" : ""}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function BwStat({
  icon,
  label,
  rows,
}: {
  icon: React.ReactNode;
  label: string;
  rows: [string, string][];
}) {
  return (
    <div className="rounded-lg border border-border p-3">
      <div className="flex items-center gap-1.5 text-xs font-medium">
        {icon} {label}
      </div>
      <div className="mt-1.5 space-y-0.5 font-mono text-[11px]">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-baseline justify-between gap-2">
            <span className="text-muted-foreground">{k}</span>
            <span className="truncate text-foreground" title={v}>{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
