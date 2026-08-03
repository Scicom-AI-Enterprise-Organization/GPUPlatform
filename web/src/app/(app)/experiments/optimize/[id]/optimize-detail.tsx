"use client";

// Detail view for one GEPA run. Mirrors experiment-detail.tsx (header band with
// KPIs + status pill, 3s poll while active, Tabs). The tabs differ because the
// artefact is a PROMPT, not a matrix:
//
//   Prompt      the winning text, next to the one it replaced
//   Candidates  the lineage — what was kept, from which parent
//   Reflections the turn-by-turn log, including the failures each rewrite read
//   Config      the run as submitted

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Copy,
  Loader2,
  Square,
  Wand2,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { PromptOptIteration, PromptOptRecord } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { JsonView } from "@/components/json-view";
import { Progress } from "@/components/ui/progress";

const STATUS_PILL: Record<string, string> = {
  queued: "border border-border bg-muted text-muted-foreground",
  running: "border border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
  completed: "border border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  failed: "border border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
  cancelled: "border border-border bg-muted text-muted-foreground",
};

const STOPPED_REASON: Record<string, string> = {
  budget: "the metric-call budget ran out",
  iterations: "no budget-consuming work was left to do",
  cancelled: "you cancelled it",
};

function pct(v: number | null | undefined) {
  if (v === null || v === undefined) return "—";
  return `${Math.round(v * 100)}%`;
}

export function OptimizeDetail({ initial }: { initial: PromptOptRecord }) {
  const [opt, setOpt] = useState(initial);
  const active = opt.status === "running" || opt.status === "queued";

  const refresh = useCallback(async () => {
    try {
      setOpt(await gateway.getPromptOpt(initial.id));
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
      await gateway.cancelPromptOpt(opt.id);
      toast.success("Cancelling — the best prompt found so far is kept.");
      void refresh();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't cancel");
    }
  }

  const result = opt.result;
  const bestPrompt = result?.best?.texts?.system_prompt ?? "";
  const seedText = result?.seed?.texts?.system_prompt ?? "";
  const delta =
    opt.best_score !== null && opt.seed_score !== null ? opt.best_score - opt.seed_score : null;
  const progress = opt.budget > 0 ? (opt.metric_calls / opt.budget) * 100 : 0;
  const iterations = result?.iterations ?? [];
  const accepted = iterations.filter((i) => i.accepted).length;

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin">
      <div className="border-b border-border bg-sidebar/40 px-6 py-5 lg:px-10">
        <Link
          href="/experiments/optimize"
          className="mb-3 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Prompt optimization
        </Link>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-semibold tracking-tight">{opt.name}</h1>
              <span
                className={cn(
                  "inline-flex items-center rounded-md px-1.5 py-0.5 text-[11px] font-medium",
                  STATUS_PILL[opt.status] ?? STATUS_PILL.queued,
                )}
              >
                {active && <Loader2 className="mr-1 h-3 w-3 animate-spin" />}
                {opt.status}
              </span>
            </div>
            <p className="mt-1 text-sm text-muted-foreground">
              <Link
                href={`/datasets/${opt.dataset_id}`}
                className="hover:text-foreground hover:underline underline-offset-2"
              >
                {opt.dataset_name}
              </Link>{" "}
              · {opt.owner} · <span className="font-mono text-[11px]">{opt.id}</span>
            </p>
          </div>
          <div className="flex items-center gap-6">
            <Kpi label="Baseline" value={pct(opt.seed_score)} />
            <Kpi label="Optimized" value={pct(opt.best_score)} />
            <Kpi
              label="Gain"
              value={delta === null ? "—" : `${delta > 0 ? "+" : ""}${Math.round(delta * 100)}pp`}
              tone={
                delta === null || Math.abs(delta) < 0.001
                  ? undefined
                  : delta > 0
                    ? "text-emerald-600 dark:text-emerald-400"
                    : "text-red-600 dark:text-red-400"
              }
            />
            <Kpi label="Metric calls" value={`${opt.metric_calls}/${opt.budget}`} />
            {active && (
              <Button variant="outline" size="sm" onClick={() => void onCancel()}>
                <Square className="h-4 w-4" />
                Cancel
              </Button>
            )}
          </div>
        </div>
        {active && <Progress value={progress} className="mt-4 h-1.5" />}
        {opt.error_text && (
          <div className="mt-3 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {opt.error_text}
          </div>
        )}
      </div>

      <div className="px-6 py-6 lg:px-10">
        {/* Two ways a green number can be meaningless. Both are load-bearing
            warnings, not decoration — see the CLAUDE.md notes on `scored`. */}
        {result && result.unscored_rollouts > 0 && (
          <Warning tone={result.unscored_rollouts >= result.rollouts ? "error" : "warn"}>
            {result.unscored_rollouts >= result.rollouts ? (
              <>
                <strong>No evaluator scored anything.</strong> All {result.rollouts} replies were
                graded by nothing — every detector abstained (usually: the rows carry no reference
                data). The score was flat, so nothing could improve and this result means nothing.
              </>
            ) : (
              <>
                {result.unscored_rollouts} of {result.rollouts} replies were graded by no evaluator
                — those rows carry no reference data and contributed 0 to every candidate alike.
              </>
            )}
          </Warning>
        )}
        {result?.in_sample && (
          <Warning tone="warn">
            Too few rows to split, so the reflection minibatches reused the validation rows. The
            gain above is measured on the rows the prompt was tuned against — confirm it on held-out
            data before trusting it.
          </Warning>
        )}

        <Tabs defaultValue="prompt">
          <TabsList>
            <TabsTrigger value="prompt">Prompt</TabsTrigger>
            <TabsTrigger value="candidates">
              Candidates
              {result ? ` (${result.candidates.length})` : ""}
            </TabsTrigger>
            <TabsTrigger value="reflections">
              Reflections
              {iterations.length ? ` (${iterations.length})` : ""}
            </TabsTrigger>
            <TabsTrigger value="config">Config</TabsTrigger>
          </TabsList>

          {/* ---------------- Prompt ---------------- */}
          <TabsContent value="prompt" className="mt-4 space-y-4">
            {!result ? (
              <Empty>
                {active ? "Scoring the starting prompt…" : "This run produced no result."}
              </Empty>
            ) : (
              <>
                <div className="flex flex-wrap items-center gap-2">
                  <Button asChild size="sm">
                    <Link href={`/experiments/new?prompt=${opt.id}`}>
                      <Wand2 className="h-4 w-4" />
                      Run an experiment with it
                    </Link>
                  </Button>
                  <CopyButton text={bestPrompt} />
                  <span className="text-xs text-muted-foreground">
                    {accepted} of {iterations.length} rewrites kept ·{" "}
                    {result.reflection_calls} reflection calls · stopped because{" "}
                    {STOPPED_REASON[result.stopped] ?? result.stopped}
                  </span>
                </div>
                <div className="grid gap-4 lg:grid-cols-2">
                  <PromptPane
                    title="Optimized"
                    score={result.best.score}
                    text={bestPrompt}
                    highlight
                  />
                  <PromptPane title="Baseline" score={result.seed.score} text={seedText} />
                </div>
                {result.components.includes("user_suffix") && (
                  <div className="grid gap-4 lg:grid-cols-2">
                    <PromptPane
                      title="Optimized user-turn reminder"
                      text={result.best.texts.user_suffix ?? ""}
                      highlight
                    />
                    <PromptPane
                      title="Baseline user-turn reminder"
                      text={result.seed.texts.user_suffix ?? ""}
                    />
                  </div>
                )}
                <p className="text-xs text-muted-foreground">
                  Scored over {result.val_rows} validation rows ({result.train_rows} train) from{" "}
                  {result.n_rows} pulled. Both prompts were measured the same way, by the same
                  evaluators — which is why the experiment you run next reproduces this number.
                </p>
              </>
            )}
          </TabsContent>

          {/* ---------------- Candidates ---------------- */}
          <TabsContent value="candidates" className="mt-4">
            {!result?.candidates.length ? (
              <Empty>No candidates yet.</Empty>
            ) : (
              <div className="overflow-x-auto rounded-md border border-border scrollbar-thin">
                <table className="w-full text-sm">
                  <thead className="bg-muted/40 text-xs text-muted-foreground">
                    <tr>
                      <th className="px-3 py-2 text-left font-medium">#</th>
                      <th className="px-3 py-2 text-left font-medium">Origin</th>
                      <th className="px-3 py-2 text-left font-medium">Parent</th>
                      <th className="px-3 py-2 text-right font-medium">Turn</th>
                      <th className="px-3 py-2 text-right font-medium">Validation</th>
                      <th className="px-3 py-2 text-left font-medium">Prompt</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...result.candidates]
                      .sort((a, b) => b.score - a.score)
                      .map((c) => (
                        <tr
                          key={c.index}
                          className={cn(
                            "border-t border-border align-top hover:bg-muted/30",
                            c.index === result.best.index && "bg-emerald-500/5",
                          )}
                        >
                          <td className="px-3 py-2 tabular-nums">
                            {c.index}
                            {c.index === result.best.index && (
                              <Badge variant="secondary" className="ml-1.5 text-[10px]">
                                best
                              </Badge>
                            )}
                          </td>
                          <td className="px-3 py-2 text-muted-foreground">
                            {c.origin === "seed" ? "seed" : `rewrote ${c.component}`}
                          </td>
                          <td className="px-3 py-2 tabular-nums text-muted-foreground">
                            {c.parent === null ? "—" : c.parent}
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                            {c.iteration || "—"}
                          </td>
                          <td className="px-3 py-2 text-right font-medium tabular-nums">
                            {pct(c.score)}
                          </td>
                          <td className="max-w-md px-3 py-2">
                            <p className="line-clamp-3 whitespace-pre-wrap break-words font-mono text-[11px] text-muted-foreground">
                              {c.texts.system_prompt || "(empty)"}
                            </p>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            )}
          </TabsContent>

          {/* ---------------- Reflections ---------------- */}
          <TabsContent value="reflections" className="mt-4 space-y-3">
            {!iterations.length ? (
              <Empty>{active ? "Waiting for the first rewrite…" : "No rewrites were tried."}</Empty>
            ) : (
              iterations
                .slice()
                .reverse()
                .map((it) => <IterationCard key={it.i} it={it} />)
            )}
          </TabsContent>

          <TabsContent value="config" className="mt-4">
            <JsonView value={opt.config} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ pieces */

function IterationCard({ it }: { it: PromptOptIteration }) {
  const [open, setOpen] = useState(false);
  return (
    <div
      className={cn(
        "rounded-md border",
        it.accepted ? "border-emerald-500/40 bg-emerald-500/5" : "border-border",
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-start gap-3 px-3 py-2.5 text-left"
      >
        <span className="mt-0.5">
          {it.accepted ? (
            <Check className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          ) : (
            <X className="h-4 w-4 text-muted-foreground" />
          )}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-2 text-sm">
            <span className="font-medium">Turn {it.i}</span>
            <span className="text-muted-foreground">
              rewrote <span className="font-mono text-xs">{it.component}</span> of candidate{" "}
              {it.parent}
            </span>
            <span className="tabular-nums text-muted-foreground">
              minibatch {pct(it.parent_score)}
              <ArrowRight className="mx-1 inline h-3 w-3" />
              {pct(it.child_score)}
            </span>
            {it.accepted && it.val_score !== null && (
              <Badge variant="secondary" className="text-[10px]">
                validation {pct(it.val_score)}
              </Badge>
            )}
          </span>
          <span className="mt-0.5 block text-xs text-muted-foreground">{it.note}</span>
        </span>
        <span className="shrink-0 text-xs text-muted-foreground">{open ? "Hide" : "Details"}</span>
      </button>
      {open && (
        <div className="space-y-4 border-t border-border px-3 py-3">
          {it.proposal && (
            <div>
              <p className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">
                Proposed prompt
              </p>
              <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded-md border border-border bg-muted/30 p-3 font-mono text-[11px] scrollbar-thin">
                {it.proposal}
              </pre>
            </div>
          )}
          {it.examples.length > 0 && (
            <div>
              <p className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">
                What the reflection model was shown
              </p>
              <div className="space-y-2">
                {it.examples.map((ex, idx) => (
                  <div key={`${ex.row_id}-${idx}`} className="rounded-md border border-border p-2.5">
                    <p className="mb-1 text-xs text-muted-foreground">
                      {ex.row_name || ex.row_id} · scored {pct(ex.score)}
                    </p>
                    <pre className="max-h-32 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-muted-foreground scrollbar-thin">
                      {ex.output || "(empty response)"}
                    </pre>
                    <pre className="mt-1.5 max-h-32 overflow-auto whitespace-pre-wrap border-t border-border pt-1.5 font-mono text-[11px] scrollbar-thin">
                      {ex.feedback}
                    </pre>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function PromptPane({
  title,
  score,
  text,
  highlight,
}: {
  title: string;
  score?: number;
  text: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-md border",
        highlight ? "border-emerald-500/40" : "border-border",
      )}
    >
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {title}
        </span>
        {score !== undefined && (
          <span className="text-sm font-semibold tabular-nums">{pct(score)}</span>
        )}
      </div>
      <pre className="max-h-[28rem] overflow-auto whitespace-pre-wrap p-3 font-mono text-xs scrollbar-thin">
        {text || "(empty)"}
      </pre>
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      onClick={() => {
        void navigator.clipboard.writeText(text);
        setDone(true);
        setTimeout(() => setDone(false), 1500);
      }}
    >
      {done ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
      {done ? "Copied" : "Copy prompt"}
    </Button>
  );
}

function Warning({ tone, children }: { tone: "warn" | "error"; children: React.ReactNode }) {
  return (
    <div
      className={cn(
        "mb-4 rounded-md border px-3 py-2 text-sm",
        tone === "error"
          ? "border-destructive/40 bg-destructive/10 text-destructive"
          : "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
      )}
    >
      {children}
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-border px-6 py-12 text-center text-sm text-muted-foreground">
      {children}
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
