"use client";

// The Evaluators tab: your reusable library on top, the platform's built-ins
// below for reference. Built-ins are configured per-experiment (their options
// depend on the run), so they're read-only here.

import { useMemo, useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { BarChart3, Globe, Inbox, Lock, Pencil, Plus, Search, Sliders, Terminal, Trash2, Wand2, X } from "lucide-react";
import { gateway } from "@/lib/gateway";
import type { CustomEvaluatorRecord, EvaluatorRegistry } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const MODE_ICON: Record<string, React.ElementType> = {
  expression: Wand2,
  api: Globe,
  python: Terminal,
};

export function EvaluatorsManager({ registry }: { registry: EvaluatorRegistry }) {
  const [customs, setCustoms] = useState<CustomEvaluatorRecord[]>(registry.custom ?? []);
  const [pending, setPending] = useState<CustomEvaluatorRecord | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [builtinQ, setBuiltinQ] = useState("");

  // Filter is over label/id/description — NEVER a hardcoded id list. The registry
  // is server-driven, so adding a detector in evaluators.py must need no change here.
  const builtins = useMemo(() => {
    const needle = builtinQ.trim().toLowerCase();
    const all = registry.evaluators;
    if (!needle) return all;
    return all.filter((sp) =>
      `${sp.label} ${sp.id} ${sp.description}`.toLowerCase().includes(needle),
    );
  }, [registry.evaluators, builtinQ]);

  async function onDelete() {
    if (!pending) return;
    setDeleting(true);
    try {
      await gateway.deleteCustomEvaluator(pending.id);
      setCustoms((xs) => xs.filter((x) => x.id !== pending.id));
      toast.success(`Deleted ${pending.name}`);
      setPending(null);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't delete");
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="space-y-8">
      {/* ---------------- your library ---------------- */}
      <section>
        <div className="mb-3 flex items-start justify-between gap-4 border-b border-border pb-2">
          <div>
            <h2 className="flex items-center gap-2 text-base font-medium">
              Your evaluators
              {customs.length > 0 && (
                <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] font-normal tabular-nums text-muted-foreground">
                  {customs.length}
                </span>
              )}
            </h2>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Reusable across every experiment. A run <span className="font-medium">snapshots</span>{" "}
              the definition, so editing one here never changes what a finished run measured.
            </p>
          </div>
          <Button asChild size="sm">
            <Link href="/experiments/evaluators/new">
              <Plus className="h-4 w-4" />
              New evaluator
            </Link>
          </Button>
        </div>


        {customs.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 rounded-md border border-dashed border-border bg-card px-6 py-14 text-center">
            <Inbox className="h-6 w-6 text-muted-foreground/60" />
            <p className="max-w-md text-sm text-muted-foreground">
              None yet. Write a one-line expression, point at an API you already run, or (if
              enabled) a Python function — then reuse it across experiments.
            </p>
          </div>
        ) : (
          <ul className="space-y-2">
            {customs.map((c) => {
              const Icon = MODE_ICON[c.mode] ?? Wand2;
              const summary =
                c.mode === "api"
                  ? String(c.config?.url ?? "")
                  : c.code.split("\n")[0] + (c.code.includes("\n") ? " …" : "");
              return (
                <li
                  key={c.id}
                  className="flex items-start gap-3 rounded-md border border-border bg-card px-3 py-2.5 transition-colors hover:border-foreground/25"
                >
                  <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground">
                    <Icon className="h-3.5 w-3.5" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span className="text-sm font-medium">{c.name}</span>
                      <Badge variant="secondary" className="text-[10px] font-normal">
                        {c.mode}
                      </Badge>
                      {c.fail_when_true && (
                        <Badge variant="secondary" className="text-[10px] font-normal">
                          true = fail
                        </Badge>
                      )}
                    </div>
                    {c.description && (
                      <p className="mt-0.5 text-xs leading-snug text-muted-foreground">
                        {c.description}
                      </p>
                    )}
                    <pre className="mt-1 truncate font-mono text-[11px] text-muted-foreground">
                      {summary}
                    </pre>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <Link
                      href={`/experiments/evaluators/new?id=${c.id}`}
                      title="Edit"
                      className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </Link>
                    <button
                      onClick={() => setPending(c)}
                      title="Delete"
                      className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {/* ---------------- built-ins ---------------- */}
      <section>
        <div className="mb-3 flex flex-wrap items-end justify-between gap-3 border-b border-border pb-2">
          <div>
          <h2 className="flex items-center gap-2 text-base font-medium">
            Built-in
            <span className="rounded-full bg-muted px-2 py-0.5 text-[11px] font-normal tabular-nums text-muted-foreground">
              {registry.evaluators.length}
            </span>
          </h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Ship with the platform. Their options depend on the run, so you configure them when
            you{" "}
            <Link
              href="/experiments/new"
              className="font-medium text-foreground underline underline-offset-2"
            >
              create an experiment
            </Link>
            .
          </p>
          </div>
          {/* The list grows with every detector added in evaluators.py, so it needs
              a way in that isn't "read all of them". */}
          <div className="relative w-full max-w-[220px]">
            <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <input
              value={builtinQ}
              onChange={(e) => setBuiltinQ(e.target.value)}
              placeholder="Filter built-ins…"
              className="h-8 w-full rounded-md border border-input bg-background pl-8 pr-7 text-sm outline-none focus-visible:ring-1 focus-visible:ring-ring"
            />
            {builtinQ && (
              <button
                type="button"
                onClick={() => setBuiltinQ("")}
                aria-label="Clear filter"
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>
        {builtins.length === 0 ? (
          <p className="rounded-md border border-dashed border-border bg-card px-3 py-10 text-center text-sm text-muted-foreground">
            No built-in matches “{builtinQ}”.
          </p>
        ) : (
        <ul className="grid gap-2 sm:grid-cols-2">
          {builtins.map((spec) => {
            const alwaysOn = registry.always_on.includes(spec.id);
            return (
              <li
                key={spec.id}
                className="rounded-md border border-border bg-card px-3 py-2.5 transition-colors hover:border-foreground/25"
              >
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span className="text-sm font-medium">{spec.label}</span>
                  <code className="font-mono text-[10px] text-muted-foreground">{spec.id}</code>
                  {alwaysOn && (
                    <Badge variant="secondary" className="text-[10px] font-normal">
                      <Lock className="mr-1 h-2.5 w-2.5" />
                      always on
                    </Badge>
                  )}
                  {spec.deferred && (
                    <Badge variant="secondary" className="text-[10px] font-normal">
                      after the replay
                    </Badge>
                  )}
                </div>
                <p className="mt-0.5 text-xs leading-snug text-muted-foreground">
                  {spec.description}
                </p>
                {/* Both facts are server-driven: `headline` marks a detector that
                    reports CORPUS-level metrics (an F1 or per-class accuracy that
                    can't be averaged from per-sample rates), `options` how much
                    there is to configure at run time. */}
                {(spec.options.length > 0 || (spec.headline?.length ?? 0) > 0) && (
                  <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[10px] text-muted-foreground">
                    {(spec.headline?.length ?? 0) > 0 && (
                      <span
                        className="inline-flex items-center gap-1"
                        title={`Corpus-level metrics: ${spec.headline?.join(", ")}`}
                      >
                        <BarChart3 className="h-3 w-3" />
                        corpus metrics
                      </span>
                    )}
                    {spec.options.length > 0 && (
                      <span className="inline-flex items-center gap-1">
                        <Sliders className="h-3 w-3" />
                        {spec.options.length} option{spec.options.length === 1 ? "" : "s"}
                      </span>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
        )}
      </section>

      <Dialog open={!!pending} onOpenChange={(o) => !o && setPending(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete evaluator?</DialogTitle>
            <DialogDescription>
              {pending?.name} will be removed from your library. Experiments that already used it
              keep their results — each run stores its own snapshot of the definition.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPending(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={() => void onDelete()} disabled={deleting}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
