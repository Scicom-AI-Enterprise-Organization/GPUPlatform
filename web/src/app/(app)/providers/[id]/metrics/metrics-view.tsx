"use client";

import { useEffect, useRef, useState } from "react";
import { Activity, Cpu, MemoryStick, AlertCircle } from "lucide-react";
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
import { gateway } from "@/lib/gateway";
import { cn } from "@/lib/utils";
import type { ProviderMetrics, ProviderRecord } from "@/lib/types";

const POLL_CHOICES = [5, 10, 15, 30] as const; // seconds
const DEFAULT_POLL_S = 10;
const MAX_POINTS = 120; // rolling window (count) — in-memory only, not persisted

type HostPoint = { i: number; cpu: number; mem: number };
type GpuPoint = { i: number; util: number; mem: number; temp: number };

const gib = (mib: number) => (mib / 1024).toFixed(1);
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
  const tick = useRef(0);
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

  const memPct = m && m.mem_total_mib > 0 ? (m.mem_used_mib / m.mem_total_mib) * 100 : 0;
  const gpus = m?.gpus ?? [];

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

      {gpus.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">
              GPUs <span className="text-[11px] font-normal text-muted-foreground">· {gpus.length} × {gpus[0].name.replace(/^NVIDIA\s+/, "")}</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 lg:grid-cols-2">
              {gpus.map((g) => {
                const mp = g.mem_total_mib > 0 ? (g.mem_used_mib / g.mem_total_mib) * 100 : 0;
                return (
                  <div key={g.index} className="rounded-lg border border-border p-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate font-mono text-xs font-medium">
                        #{g.index} {g.name.replace(/^NVIDIA\s+/, "")}
                      </span>
                    </div>
                    <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px]">
                      <span className="text-emerald-600 dark:text-emerald-400">{g.util_pct}% util</span>
                      <span className="text-sky-600 dark:text-sky-400">
                        {mp.toFixed(0)}% mem · {gib(g.mem_used_mib)}/{gib(g.mem_total_mib)} GiB
                      </span>
                      <span className="text-amber-600 dark:text-amber-400">{g.temp_c}°C</span>
                    </div>
                    <MiniChart
                      data={gpuSeries[g.index] ?? []}
                      keys={[
                        { k: "util", color: "#10b981", label: "util" },
                        { k: "mem", color: "#0ea5e9", label: "mem" },
                      ]}
                    />
                  </div>
                );
              })}
            </div>
            <p className="mt-2 text-[10px] text-muted-foreground">
              <span className="text-emerald-600 dark:text-emerald-400">■</span> util % ·{" "}
              <span className="text-sky-600 dark:text-sky-400">■</span> memory % — temperature shown live above (°C).
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
