"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, ArrowLeft, Download, Gauge, Loader2, Target, TrendingDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { gateway } from "@/lib/gateway";
import type { TrainingEpoch, TrainingResult, TrainingRunRecord, TrainingStep } from "@/lib/types";
import { cn } from "@/lib/utils";

// Colour encodes RUN IDENTITY (categorical, high-contrast on both themes) — same
// palette as the benchmark compare so overlaid runs are easy to tell apart.
const COMPARE_COLORS = [
  "#2563eb", // blue-600
  "#f59e0b", // amber-500
  "#10b981", // emerald-500
  "#ec4899", // pink-500
  "#8b5cf6", // violet-500
  "#06b6d4", // cyan-500
  "#f97316", // orange-500
  "#84cc16", // lime-500
  "#ef4444", // red-500
  "#14b8a6", // teal-500
  "#a855f7", // purple-500
  "#eab308", // yellow-500
];

// One run's loaded state: metadata (for labels) + the metric series needed to
// overlay loss + accuracy across runs.
type RunData = {
  id: string;
  name: string;
  status: string;
  task: string;
  model: string | null;
  steps: TrainingStep[];
  epochs: TrainingEpoch[];
  best: TrainingResult["best"];
  error: string | null;
};

function shortModel(s: string | null): string {
  if (!s) return "—";
  return s.split("/").pop() ?? s;
}

function taskLabel(t: string): string {
  return t === "tts" ? "TTS" : t === "llm" ? "LLM" : t === "asr" ? "ASR" : "—";
}

function fmt(v: number | null | undefined, digits = 2): string {
  return v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits);
}

// Collapse duplicate optimizer-step numbers into ONE point by averaging. With
// gradient accumulation the trainer emits a @@STEP per microbatch (all sharing
// one opt-step number); plotting them raw draws a vertical zig-zag. Mirrors the
// single-run LossCurve.
function stepLossPoints(steps: TrainingStep[]): { x: number; y: number }[] {
  const byStep = new Map<number, { sum: number; n: number }>();
  for (const s of steps) {
    if (typeof s.loss !== "number") continue;
    const cur = byStep.get(s.step);
    if (cur) {
      cur.sum += s.loss;
      cur.n += 1;
    } else {
      byStep.set(s.step, { sum: s.loss, n: 1 });
    }
  }
  return [...byStep.entries()].sort((a, b) => a[0] - b[0]).map(([x, v]) => ({ x, y: v.sum / v.n }));
}

// Merge each run's (x, y) series into recharts rows keyed by x, one column per
// run id (null where a run has no value at that x → the line skips the gap).
function mergeSeries(
  runs: RunData[],
  pts: (r: RunData) => { x: number; y: number | null }[],
): { rows: Record<string, number | null>[]; hasData: boolean } {
  const xs = new Set<number>();
  const perRun = runs.map((r) => {
    const m = new Map<number, number>();
    for (const p of pts(r)) {
      if (p.y != null && Number.isFinite(p.y)) {
        m.set(p.x, p.y);
        xs.add(p.x);
      }
    }
    return m;
  });
  const sorted = [...xs].sort((a, b) => a - b);
  const rows = sorted.map((x) => {
    const row: Record<string, number | null> = { x };
    runs.forEach((r, i) => {
      row[r.id] = perRun[i].has(x) ? perRun[i].get(x)! : null;
    });
    return row;
  });
  return { rows, hasData: sorted.length > 0 };
}

function bestEvalLoss(r: RunData): number | null {
  const fromEpochs = r.epochs.map((e) => e.eval_loss).filter((n): n is number => typeof n === "number");
  if (fromEpochs.length) return Math.min(...fromEpochs);
  return r.best?.eval_loss ?? null;
}

function finalTrainLoss(r: RunData): number | null {
  const pts = stepLossPoints(r.steps);
  if (pts.length) return pts[pts.length - 1].y;
  return r.best?.loss ?? null;
}

export function CompareView({ ids }: { ids: string[] }) {
  const [runs, setRuns] = useState<RunData[] | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const loaded = await Promise.all(
        ids.map(async (id): Promise<RunData> => {
          try {
            const rec: TrainingRunRecord = await gateway.getTrainingRun(id);
            const rj = rec.result_json ?? {};
            const steps = rj.steps ?? [];
            const epochs = rj.epochs ?? [];
            return {
              id,
              name: rec.name ?? id,
              status: rec.status ?? "unknown",
              task: rec.task_type ?? "asr",
              model: rec.base_model ?? null,
              steps,
              epochs,
              best: rj.best ?? null,
              error: steps.length === 0 && epochs.length === 0 ? "no metrics yet" : null,
            };
          } catch (e) {
            return {
              id,
              name: id,
              status: "unknown",
              task: "other",
              model: null,
              steps: [],
              epochs: [],
              best: null,
              error: e instanceof Error ? e.message : String(e),
            };
          }
        }),
      );
      if (!cancelled) {
        setRuns(loaded);
        setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [ids]);

  const trainLoss = useMemo(
    () => mergeSeries(runs ?? [], (r) => stepLossPoints(r.steps).map((p) => ({ x: p.x, y: p.y }))),
    [runs],
  );
  const evalLoss = useMemo(
    () =>
      mergeSeries(runs ?? [], (r) =>
        r.epochs.map((e) => ({ x: e.epoch, y: e.eval_loss ?? null })),
      ),
    [runs],
  );
  const werSeries = useMemo(
    () => mergeSeries(runs ?? [], (r) => r.epochs.map((e) => ({ x: e.epoch, y: e.wer ?? null }))),
    [runs],
  );
  const cerSeries = useMemo(
    () => mergeSeries(runs ?? [], (r) => r.epochs.map((e) => ({ x: e.epoch, y: e.cer ?? null }))),
    [runs],
  );

  if (loading || !runs) {
    return (
      <div className="flex items-center justify-center rounded-md border border-border px-4 py-16 text-sm text-muted-foreground">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading {ids.length} run{ids.length === 1 ? "" : "s"}…
      </div>
    );
  }

  const loadedCount = runs.filter((r) => r.steps.length > 0 || r.epochs.length > 0).length;
  const noData = !trainLoss.hasData && !evalLoss.hasData && !werSeries.hasData && !cerSeries.hasData;

  // Table columns only appear when at least one run carries that metric — a set
  // of all-LLM runs shouldn't show empty WER/CER columns, etc.
  const anyEval = runs.some((r) => bestEvalLoss(r) != null);
  const anyWer = runs.some((r) => r.best?.wer != null);
  const anyCer = runs.some((r) => r.best?.cer != null);
  const anyMetric = runs.some((r) => r.best?.metric != null);
  const anyTrain = runs.some((r) => finalTrainLoss(r) != null);

  return (
    <div className="space-y-6 bg-background">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Button asChild variant="ghost" size="sm" className="-ml-2 h-7 px-2 text-muted-foreground">
            <Link href="/autotrain">
              <ArrowLeft className="h-3.5 w-3.5" /> Autotrain
            </Link>
          </Button>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight">Compare training runs</h1>
          <p className="mt-1 text-sm text-muted-foreground">{loadedCount} of {ids.length} loaded</p>
        </div>
        <div className="flex items-center gap-2" data-html2canvas-ignore="true">
          <Button
            variant="outline"
            size="sm"
            onClick={() => window.print()}
            title="Opens the print dialog — choose “Save as PDF”"
          >
            <Download className="h-4 w-4" />
            PDF
          </Button>
        </div>
      </div>

      {/* Per-run legend: colour ↔ run + model + task. */}
      <div className="flex flex-wrap gap-2">
        {runs.map((r, i) => {
          const color = COMPARE_COLORS[i % COMPARE_COLORS.length];
          return (
            <Link
              key={r.id}
              href={`/autotrain/${encodeURIComponent(r.id)}`}
              className={cn(
                "inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-1.5 text-xs transition-colors hover:border-primary/40",
                r.error && "opacity-60",
              )}
            >
              <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color }} />
              <span className="font-medium text-foreground">{r.name}</span>
              <span className="font-mono text-muted-foreground">{shortModel(r.model)}</span>
              <span className="text-[10px] uppercase text-muted-foreground">{taskLabel(r.task)}</span>
              {r.error && <span className="text-destructive">· {r.error}</span>}
            </Link>
          );
        })}
      </div>

      {noData ? (
        <div className="rounded-md border border-dashed border-border px-6 py-12 text-center text-sm text-muted-foreground">
          None of the selected runs have metrics yet.
        </div>
      ) : (
        <>
          {/* Headline numbers side by side. */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Headline</CardTitle>
              <CardDescription className="text-xs">Best checkpoint per run (lower is better).</CardDescription>
            </CardHeader>
            <CardContent className="px-0 pb-0">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-muted/40">
                    <tr>
                      <th className="px-3 py-2 text-left text-xs uppercase tracking-wide text-muted-foreground">Run</th>
                      <th className="px-3 py-2 text-left text-xs uppercase tracking-wide text-muted-foreground">Model</th>
                      <th className="px-3 py-2 text-left text-xs uppercase tracking-wide text-muted-foreground">Task</th>
                      {anyEval && <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">Best eval loss</th>}
                      {anyWer && <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">Best WER</th>}
                      {anyCer && <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">Best CER</th>}
                      {anyMetric && <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">Best metric</th>}
                      {anyTrain && <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">Final train loss</th>}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {runs.map((r, i) => {
                      const color = COMPARE_COLORS[i % COMPARE_COLORS.length];
                      return (
                        <tr key={r.id}>
                          <td className="px-3 py-1.5">
                            <span className="inline-flex items-center gap-2">
                              <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
                              <span className="font-medium">{r.name}</span>
                            </span>
                          </td>
                          <td className="px-3 py-1.5 font-mono text-xs text-muted-foreground">{shortModel(r.model)}</td>
                          <td className="px-3 py-1.5 text-xs text-muted-foreground">{taskLabel(r.task)}</td>
                          {anyEval && <td className="px-3 py-1.5 text-right tabular-nums">{fmt(bestEvalLoss(r), 4)}</td>}
                          {anyWer && <td className="px-3 py-1.5 text-right tabular-nums">{fmt(r.best?.wer)}</td>}
                          {anyCer && <td className="px-3 py-1.5 text-right tabular-nums">{fmt(r.best?.cer)}</td>}
                          {anyMetric && <td className="px-3 py-1.5 text-right tabular-nums">{fmt(r.best?.metric, 3)}</td>}
                          {anyTrain && <td className="px-3 py-1.5 text-right tabular-nums">{fmt(finalTrainLoss(r), 4)}</td>}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {trainLoss.hasData && (
              <ChartCard
                title="Training loss"
                subtitle="per optimizer step — lower is better"
                icon={<Activity className="h-4 w-4" />}
              >
                <OverlayLineChart
                  runs={runs}
                  data={trainLoss.rows}
                  xLabel="step"
                  yLabel="loss"
                  yTickFmt={(v) => v.toFixed(2)}
                  tipFmt={(v) => v.toFixed(4)}
                />
              </ChartCard>
            )}

            {evalLoss.hasData && (
              <ChartCard
                title="Eval loss"
                subtitle="per epoch — lower is better"
                icon={<TrendingDown className="h-4 w-4" />}
              >
                <OverlayLineChart
                  runs={runs}
                  data={evalLoss.rows}
                  xLabel="epoch"
                  yLabel="loss"
                  yTickFmt={(v) => v.toFixed(2)}
                  tipFmt={(v) => v.toFixed(4)}
                />
              </ChartCard>
            )}

            {werSeries.hasData && (
              <ChartCard
                title="WER"
                subtitle="word error rate per epoch — lower is better"
                icon={<Target className="h-4 w-4" />}
              >
                <OverlayLineChart
                  runs={runs}
                  data={werSeries.rows}
                  xLabel="epoch"
                  yLabel="%"
                  yTickFmt={(v) => `${v.toFixed(0)}%`}
                  tipFmt={(v) => `${v.toFixed(2)}%`}
                />
              </ChartCard>
            )}

            {cerSeries.hasData && (
              <ChartCard
                title="CER"
                subtitle="character error rate per epoch — lower is better"
                icon={<Gauge className="h-4 w-4" />}
              >
                <OverlayLineChart
                  runs={runs}
                  data={cerSeries.rows}
                  xLabel="epoch"
                  yLabel="%"
                  yTickFmt={(v) => `${v.toFixed(0)}%`}
                  tipFmt={(v) => `${v.toFixed(2)}%`}
                />
              </ChartCard>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  icon,
  children,
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-muted text-muted-foreground">
            {icon}
          </div>
          <div>
            <CardTitle className="text-sm">{title}</CardTitle>
            <CardDescription className="text-[11px]">{subtitle}</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function OverlayLineChart({
  runs,
  data,
  xLabel,
  yLabel,
  yTickFmt,
  tipFmt,
}: {
  runs: RunData[];
  data: Record<string, number | null>[];
  xLabel: string;
  yLabel: string;
  yTickFmt: (v: number) => string;
  tipFmt: (v: number) => string;
}) {
  const nameById = useMemo(() => {
    const m: Record<string, string> = {};
    for (const r of runs) m[r.id] = r.name;
    return m;
  }, [runs]);

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, left: 12, bottom: 28 }}>
          <CartesianGrid stroke="currentColor" strokeOpacity={0.12} vertical={false} />
          <XAxis
            dataKey="x"
            type="number"
            domain={["dataMin", "dataMax"]}
            stroke="currentColor"
            className="text-[10px] text-muted-foreground"
            tickLine={false}
            axisLine={false}
            height={44}
            tickMargin={6}
            label={{ value: xLabel, position: "insideBottom", offset: -8, fontSize: 11, fill: "currentColor" }}
          />
          <YAxis
            stroke="currentColor"
            className="text-[10px] text-muted-foreground"
            tickLine={false}
            axisLine={false}
            width={56}
            tickMargin={4}
            domain={["auto", "auto"]}
            tickFormatter={(v: number) => yTickFmt(Number(v))}
            label={{
              value: yLabel,
              angle: -90,
              position: "insideLeft",
              offset: 2,
              fontSize: 10,
              fill: "currentColor",
              style: { textAnchor: "middle" },
            }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--popover)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              fontSize: 11,
              color: "var(--popover-foreground)",
            }}
            labelStyle={{ color: "var(--popover-foreground)" }}
            itemStyle={{ color: "var(--popover-foreground)" }}
            formatter={(value, name) => [tipFmt(Number(value)), nameById[String(name)] ?? String(name)]}
            labelFormatter={(l) => `${xLabel} ${l}`}
          />
          <Legend verticalAlign="top" align="center" iconType="plainline" wrapperStyle={{ fontSize: 11, paddingBottom: 8 }} />
          {runs.map((r, i) => (
            <Line
              key={r.id}
              type="monotone"
              dataKey={r.id}
              name={r.name}
              stroke={COMPARE_COLORS[i % COMPARE_COLORS.length]}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
