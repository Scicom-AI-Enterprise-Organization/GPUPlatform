"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { ArrowLeft, CheckCircle2, ChevronRight, Loader2, Search, Square, X, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type {
  ExperimentCell,
  ExperimentRecord,
  ExperimentSampleRecord,
} from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { JsonView } from "@/components/json-view";
import { Progress } from "@/components/ui/progress";
import { TradeoffPlot, SERIES_CLASS } from "../tradeoff-plot";

const STATUS_PILL: Record<string, string> = {
  queued: "border border-border bg-muted text-muted-foreground",
  running: "border border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
  completed: "border border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  failed: "border border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
  cancelled: "border border-border bg-muted text-muted-foreground",
};

function rateTone(rate: number) {
  if (rate >= 0.95) return "text-emerald-600 dark:text-emerald-400";
  if (rate >= 0.8) return "text-amber-600 dark:text-amber-400";
  return "text-red-600 dark:text-red-400";
}

function fmtMs(v: number | null | undefined) {
  if (v === null || v === undefined) return "—";
  return v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${Math.round(v)}ms`;
}

const TAB_VALUES = ["overview", "tradeoff", "samples", "config"] as const;
type ExperimentTab = (typeof TAB_VALUES)[number];

export function ExperimentDetail({ initial }: { initial: ExperimentRecord }) {
  // The active tab lives in the URL (?tab=), and each trigger is a real <Link> —
  // so a tab is linkable/shareable and ⌘-click opens it in a new tab. Same
  // convention as the proxy detail page. useSearchParams is reactive to soft
  // navigations, so a normal click still switches in place with no reload.
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const tabParam = searchParams.get("tab");
  const tab: ExperimentTab = (TAB_VALUES as readonly string[]).includes(tabParam ?? "")
    ? (tabParam as ExperimentTab)
    : "overview";
  const tabHref = (v: ExperimentTab) => {
    const p = new URLSearchParams(searchParams.toString());
    p.set("tab", v);
    return `${pathname}?${p.toString()}`;
  };

  const [exp, setExp] = useState(initial);
  const active = exp.status === "running" || exp.status === "queued";

  const refresh = useCallback(async () => {
    try {
      setExp(await gateway.getExperiment(initial.id));
    } catch {
      /* transient — the poll will retry */
    }
  }, [initial.id]);

  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => void refresh(), 3000);
    return () => clearInterval(t);
  }, [active, refresh]);

  async function onCancel() {
    try {
      await gateway.cancelExperiment(exp.id);
      toast.success("Cancelling — results so far are kept.");
      void refresh();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't cancel");
    }
  }

  const summary = exp.summary;
  const progress = exp.n_planned > 0 ? (exp.n_completed / exp.n_planned) * 100 : 0;

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin">
      {/* header band */}
      <div className="border-b border-border bg-sidebar/40 px-6 py-5 lg:px-10">
        <Link
          href="/experiments"
          className="mb-3 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Experiments
        </Link>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-semibold tracking-tight">{exp.name}</h1>
              <span
                className={cn(
                  "inline-flex items-center rounded-md px-1.5 py-0.5 text-[11px] font-medium",
                  STATUS_PILL[exp.status] ?? STATUS_PILL.queued,
                )}
              >
                {active && <Loader2 className="mr-1 h-3 w-3 animate-spin" />}
                {exp.status}
              </span>
            </div>
            <p className="mt-1 text-sm text-muted-foreground">
              <Link
                href={`/datasets/${exp.dataset_id}`}
                className="hover:text-foreground hover:underline underline-offset-2"
              >
                {exp.dataset_name}
              </Link>{" "}
              · {exp.owner} · <span className="font-mono text-[11px]">{exp.id}</span>
            </p>
          </div>
          <div className="flex items-center gap-6">
            <Kpi label="Samples" value={`${exp.n_completed}/${exp.n_planned}`} />
            <Kpi
              label="Pass rate"
              value={summary ? `${Math.round(summary.totals.pass_rate * 100)}%` : "—"}
              tone={summary ? rateTone(summary.totals.pass_rate) : undefined}
            />
            <Kpi
              label="Errors"
              value={String(exp.n_failed)}
              tone={exp.n_failed > 0 ? "text-red-600 dark:text-red-400" : undefined}
            />
            {active && (
              <Button variant="outline" size="sm" onClick={() => void onCancel()}>
                <Square className="h-4 w-4" />
                Cancel
              </Button>
            )}
          </div>
        </div>
        {active && <Progress value={progress} className="mt-4 h-1.5" />}
        {exp.error_text && (
          <div className="mt-3 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {exp.error_text}
          </div>
        )}
      </div>

      <div className="px-6 py-6 lg:px-10">
        <Tabs value={tab}>
          <TabsList>
            {TAB_VALUES.map((v) => (
              <TabsTrigger key={v} value={v} asChild>
                <Link href={tabHref(v)} scroll={false} className="capitalize">
                  {v}
                </Link>
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value="overview" className="mt-4">
            {summary ? (
              <>
                <CellsTable summary={summary} />
                <BenchmarkMetrics summary={summary} />
              </>
            ) : (
              <p className="py-12 text-center text-sm text-muted-foreground">
                {active ? "Running — results appear when the run finishes." : "No results."}
              </p>
            )}
          </TabsContent>

          <TabsContent value="tradeoff" className="mt-4">
            {summary ? (
              <>
                <div className="rounded-lg border border-border bg-card p-4">
                  <h2 className="text-sm font-medium">Configuration tradeoff</h2>
                  <p className="mb-2 text-xs text-muted-foreground">
                    One line per target × variant across every measure. A line that crosses
                    another between two axes is a real tradeoff — better on one, worse on the next.
                  </p>
                  <TradeoffPlot summary={summary} />
                </div>
                {/* The table is the accessible companion to the plot, not a duplicate:
                    it carries the exact numbers the axes only position. */}
                <div className="mt-4">
                  <CellsTable summary={summary} />
                </div>
              </>
            ) : (
              <p className="py-12 text-center text-sm text-muted-foreground">
                Nothing to plot yet.
              </p>
            )}
          </TabsContent>

          <TabsContent value="samples" className="mt-4">
            <SamplesTab experimentId={exp.id} summary={summary} />
          </TabsContent>

          <TabsContent value="config" className="mt-4">
            <div className="rounded-lg border border-border bg-card p-4">
              <JsonView value={exp.config} />
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

function Kpi({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="text-right">
      <p className={cn("text-xl font-semibold tabular-nums", tone)}>{value}</p>
      <p className="text-xs text-muted-foreground">{label}</p>
    </div>
  );
}

function CellsTable({ summary }: { summary: NonNullable<ExperimentRecord["summary"]> }) {
  const evalIds = (summary.evaluator_ids ?? []).filter((id) =>
    summary.cells.some((c) => c.evals[id]),
  );
  return (
    <div className="overflow-x-auto rounded-md border border-border scrollbar-thin">
      <table className="w-full text-sm">
        <thead className="bg-muted/40 text-xs text-muted-foreground">
          <tr>
            <th className="px-3 py-2 text-left font-medium">Target</th>
            <th className="px-3 py-2 text-left font-medium">Variant</th>
            <th className="px-3 py-2 text-right font-medium">n</th>
            <th className="px-3 py-2 text-right font-medium">Pass</th>
            <th className="px-3 py-2 text-right font-medium">Errors</th>
            <th className="px-3 py-2 text-right font-medium">p50</th>
            <th className="px-3 py-2 text-right font-medium">p95</th>
            <th className="px-3 py-2 text-right font-medium">TTFT p95</th>
            {evalIds.map((id) => (
              <th key={id} className="whitespace-nowrap px-3 py-2 text-right font-medium">
                {id.replace(/_/g, " ")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {summary.cells.map((c: ExperimentCell, i) => (
            <tr key={`${c.target}/${c.variant}`} className="border-t border-border hover:bg-muted/30">
              <td className="whitespace-nowrap px-3 py-2">
                <span className="inline-flex items-center gap-1.5">
                  <span
                    className={cn(
                      "inline-block h-2.5 w-2.5 rounded-sm bg-current",
                      SERIES_CLASS[i % SERIES_CLASS.length],
                    )}
                  />
                  {c.target}
                </span>
              </td>
              <td className="whitespace-nowrap px-3 py-2 text-muted-foreground">{c.variant}</td>
              <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{c.n}</td>
              <td className={cn("px-3 py-2 text-right font-medium tabular-nums", rateTone(c.pass_rate))}>
                {Math.round(c.pass_rate * 100)}%
              </td>
              <td className="px-3 py-2 text-right tabular-nums">
                {c.n_error > 0 ? (
                  <span className="text-red-600 dark:text-red-400">{c.n_error}</span>
                ) : (
                  <span className="text-muted-foreground">0</span>
                )}
              </td>
              <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                {fmtMs(c.latency_ms.p50)}
              </td>
              <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                {fmtMs(c.latency_ms.p95)}
              </td>
              <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                {fmtMs(c.ttft_ms.p95)}
              </td>
              {evalIds.map((id) => {
                const st = c.evals[id];
                if (!st) return <td key={id} className="px-3 py-2 text-right text-muted-foreground">—</td>;
                return (
                  <td key={id} className="px-3 py-2 text-right tabular-nums">
                    {st.n_failed === 0 ? (
                      <span className="text-muted-foreground">0</span>
                    ) : (
                      <span className="font-medium text-red-600 dark:text-red-400">
                        {st.n_failed}
                        <span className="ml-1 text-[11px] font-normal text-muted-foreground">
                          {Math.round(st.fail_rate * 100)}%
                        </span>
                      </span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Corpus-level metrics from the benchmark-derived evaluators. These are pooled
 * across every sample in a cell (an F1 can't be averaged), so they get their own
 * block rather than a column in the per-evaluator failure table. */
function BenchmarkMetrics({ summary }: { summary: NonNullable<ExperimentRecord["summary"]> }) {
  const blocks = summary.cells.flatMap((cell, i) =>
    Object.entries(cell.evals)
      .filter(([, stat]) => stat.metrics && Object.keys(stat.metrics).length > 0)
      .map(([id, stat]) => ({ cell, i, id, stat })),
  );
  if (!blocks.length) return null;

  return (
    <div className="mt-4 space-y-3">
      {blocks.map(({ cell, i, id, stat }) => {
        const metrics = stat.metrics ?? {};
        const headline = new Set(stat.headline ?? []);
        const keys = [
          ...(stat.headline ?? []).filter((k) => k in metrics),
          ...Object.keys(metrics).filter((k) => !headline.has(k)),
        ];
        return (
          <div
            key={`${cell.target}/${cell.variant}/${id}`}
            className="rounded-lg border border-border bg-card p-4"
          >
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <span
                className={cn(
                  "inline-block h-2.5 w-2.5 rounded-sm bg-current",
                  SERIES_CLASS[i % SERIES_CLASS.length],
                )}
              />
              <h3 className="text-sm font-medium">{id.replace(/_/g, " ")}</h3>
              <span className="text-xs text-muted-foreground">
                {cell.target} / {cell.variant}
              </span>
            </div>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4 lg:grid-cols-5">
              {keys.map((k) => {
                const v = metrics[k];
                return (
                  <div key={k}>
                    <dt className="truncate text-[11px] uppercase tracking-wide text-muted-foreground">
                      {k.replace(/_/g, " ")}
                    </dt>
                    <dd
                      className={cn(
                        "tabular-nums",
                        headline.has(k) ? "text-lg font-semibold" : "text-sm",
                      )}
                    >
                      {typeof v === "number" ? v.toFixed(v < 1 && v > 0 ? 3 : 2) : (v ?? "—")}
                    </dd>
                  </div>
                );
              })}
            </dl>
          </div>
        );
      })}
    </div>
  );
}

function SamplesTab({
  experimentId,
  summary,
}: {
  experimentId: string;
  summary: ExperimentRecord["summary"];
}) {
  const [items, setItems] = useState<ExperimentSampleRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [onlyFailed, setOnlyFailed] = useState(true);
  const [target, setTarget] = useState("");
  const [variant, setVariant] = useState("");
  const [q, setQ] = useState("");
  const [qDebounced, setQDebounced] = useState("");
  const [page, setPage] = useState(0);

  const targets = [...new Set(summary?.cells.map((c) => c.target) ?? [])];
  const variants = [...new Set(summary?.cells.map((c) => c.variant) ?? [])];

  useEffect(() => {
    const t = setTimeout(() => setQDebounced(q), 300);
    return () => clearTimeout(t);
  }, [q]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await gateway.listExperimentSamples(experimentId, {
        only_failed: onlyFailed,
        target: target || undefined,
        variant: variant || undefined,
        q: qDebounced || undefined,
        limit: 25,
        offset: page * 25,
      });
      setItems(res.items);
      setTotal(res.total);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't load samples");
    } finally {
      setLoading(false);
    }
  }, [experimentId, onlyFailed, target, variant, qDebounced, page]);

  useEffect(() => {
    void load();
  }, [load]);

  const ALL = "__all";

  return (
    <div className="space-y-3">
      {/* Filters, one row above the list — the interaction convention used across
          the app's tables (see the proxy queue tab). */}
      <div className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-card px-3 py-2">
        {/* Pass/fail scope as a segmented control: two mutually exclusive views,
            not a setting — a checkbox reads as "and also…" which it isn't. */}
        <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
          {([true, false] as const).map((v) => (
            <button
              key={String(v)}
              type="button"
              onClick={() => {
                setOnlyFailed(v);
                setPage(0);
              }}
              className={cn(
                "rounded px-2 py-1 transition-colors",
                onlyFailed === v
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {v ? "Failures" : "All"}
            </button>
          ))}
        </div>

        {targets.length > 1 && (
          <Select
            value={target || ALL}
            onValueChange={(v) => {
              setTarget(v === ALL ? "" : v);
              setPage(0);
            }}
          >
            <SelectTrigger className="h-8 w-[150px] text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL} className="text-xs">All targets</SelectItem>
              {targets.map((t) => (
                <SelectItem key={t} value={t} className="text-xs">{t}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {variants.length > 1 && (
          <Select
            value={variant || ALL}
            onValueChange={(v) => {
              setVariant(v === ALL ? "" : v);
              setPage(0);
            }}
          >
            <SelectTrigger className="h-8 w-[150px] text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL} className="text-xs">All variants</SelectItem>
              {variants.map((v) => (
                <SelectItem key={v} value={v} className="text-xs">{v}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        <div className="relative min-w-[200px] flex-1">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <input
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              setPage(0);
            }}
            placeholder="Search reply text…"
            className="h-8 w-full rounded-md border border-input bg-background pl-8 pr-7 text-sm outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
          {q && (
            <button
              type="button"
              onClick={() => setQ("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              aria-label="Clear search"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>

        <span className="ml-auto flex items-center gap-2 text-xs tabular-nums text-muted-foreground">
          {loading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          {total} {onlyFailed ? "failing" : "total"}
        </span>
      </div>

      {items.length === 0 && !loading ? (
        <div className="rounded-md border border-dashed border-border py-12 text-center">
          {onlyFailed ? (
            <>
              <CheckCircle2 className="mx-auto mb-2 h-6 w-6 text-emerald-600 dark:text-emerald-400" />
              <p className="text-sm font-medium">No failures</p>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Every reply passed every evaluator. Switch to <span className="font-medium">All</span> to read them.
              </p>
            </>
          ) : (
            <p className="text-sm text-muted-foreground">No samples match these filters.</p>
          )}
        </div>
      ) : (
        <ul className="space-y-1.5">
          {items.map((s) => (
            <SampleCard key={s.id} s={s} />
          ))}
        </ul>
      )}

      {total > 25 && (
        <div className="flex items-center justify-center gap-2 pt-1">
          <Button
            variant="outline"
            size="sm"
            disabled={page === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
          >
            Previous
          </Button>
          <span className="text-xs tabular-nums text-muted-foreground">
            {page * 25 + 1}–{Math.min((page + 1) * 25, total)} of {total}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={(page + 1) * 25 >= total}
            onClick={() => setPage((p) => p + 1)}
          >
            Next
          </Button>
        </div>
      )}
    </div>
  );
}

/** A reference chip carried on an evaluator's flags — for a red-team row, which
 * attack category it is. The corpus's whole point is that a row is an attack or a
 * benign control, so showing only the reply hides what the verdict means. */
function referenceChip(s: ExperimentSampleRecord): { label: string; attack: boolean } | null {
  for (const v of Object.values(s.evals)) {
    const f = v.flags as { attack?: boolean; attack_type?: string } | undefined;
    if (!f || typeof f.attack !== "boolean") continue;
    return { label: f.attack ? f.attack_type || "attack" : "benign", attack: f.attack };
  }
  return null;
}

function SampleCard({ s }: { s: ExperimentSampleRecord }) {
  const [open, setOpen] = useState(false);
  const evals = Object.entries(s.evals);
  const failures = evals.filter(([, v]) => !v.passed);
  const ref = referenceChip(s);

  return (
    <li
      className={cn(
        "overflow-hidden rounded-md border transition-colors",
        s.passed ? "border-border hover:border-foreground/25" : "border-red-500/40",
      )}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-start gap-3 px-3 py-2.5 text-left hover:bg-muted/40"
      >
        {/* Verdict is an icon + colour, never colour alone. */}
        <span className="mt-0.5 shrink-0">
          {s.passed ? (
            <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          ) : (
            <XCircle className="h-4 w-4 text-red-600 dark:text-red-400" />
          )}
        </span>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <span className="truncate font-mono text-xs font-medium">{s.case_name}</span>
            {ref && (
              <span
                className={cn(
                  "rounded px-1.5 py-0.5 font-mono text-[10px]",
                  ref.attack
                    ? "bg-amber-500/15 text-amber-700 dark:text-amber-400"
                    : "bg-muted text-muted-foreground",
                )}
                title={ref.attack ? "attack row — a refusal is the correct answer"
                                  : "benign control — the model should answer normally"}
              >
                {ref.label}
              </span>
            )}
            <span className="text-[11px] text-muted-foreground">
              {s.target} · {s.variant}
              {s.repeat > 1 ? ` · run ${s.repeat}` : ""}
            </span>
          </div>

          <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
            {s.error_text ? (
              <span className="text-red-600 dark:text-red-400">{s.error_text}</span>
            ) : (
              s.content.slice(0, 240) || <span className="italic">(empty reply)</span>
            )}
          </p>

          {failures.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {failures.map(([id, v]) => (
                <span
                  key={id}
                  title={v.reason ?? undefined}
                  className="rounded border border-red-500/40 bg-red-500/10 px-1.5 py-0.5 text-[10px] font-medium text-red-700 dark:text-red-400"
                >
                  {id.replace(/_/g, " ")}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Per-sample metrics, right-aligned and tabular so they form a column
            down the list instead of a run-on of dot-separated values. */}
        <div className="hidden shrink-0 items-center gap-3 pt-0.5 text-[11px] tabular-nums text-muted-foreground sm:flex">
          {s.latency_ms !== null && <span title="latency">{fmtMs(s.latency_ms)}</span>}
          {s.ttft_ms !== null && <span title="time to first token">ttft {fmtMs(s.ttft_ms)}</span>}
          {s.completion_tokens !== null && <span title="output tokens">{s.completion_tokens} tok</span>}
          <ChevronRight className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-90")} />
        </div>
      </button>

      {open && (
        <div className="space-y-3 border-t border-border bg-muted/20 px-3 py-3">
          {evals.length > 0 && (
            <div>
              <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                Evaluators
              </p>
              <ul className="space-y-1">
                {evals.map(([id, v]) => (
                  <li key={id} className="flex items-start gap-2 text-xs">
                    {v.passed ? (
                      <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-600 dark:text-emerald-400" />
                    ) : (
                      <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-600 dark:text-red-400" />
                    )}
                    <span className="font-medium">{id.replace(/_/g, " ")}</span>
                    {v.reason && <span className="min-w-0 text-muted-foreground">— {v.reason}</span>}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div>
            <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Reply
            </p>
            <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-background p-2.5 font-mono text-[11px] leading-relaxed scrollbar-thin">
              {s.content || "(empty)"}
            </pre>
          </div>

          {s.reasoning && (
            <div>
              <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                Reasoning
              </p>
              <pre className="max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-background p-2.5 font-mono text-[11px] leading-relaxed text-muted-foreground scrollbar-thin">
                {s.reasoning}
              </pre>
            </div>
          )}

          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] tabular-nums text-muted-foreground">
            {s.prompt_tokens !== null && <span>prompt {s.prompt_tokens} tok</span>}
            {s.completion_tokens !== null && <span>output {s.completion_tokens} tok</span>}
            {s.latency_ms !== null && <span>latency {fmtMs(s.latency_ms)}</span>}
            {s.ttft_ms !== null && <span>ttft {fmtMs(s.ttft_ms)}</span>}
            {s.finish_reason && <span>finish {s.finish_reason}</span>}
            {s.status_code !== null && <span>HTTP {s.status_code}</span>}
          </div>
        </div>
      )}
    </li>
  );
}
