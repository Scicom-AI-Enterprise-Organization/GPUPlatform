"use client";

import { formatEta, parseAutotrainProgress, prettyStep, useEta } from "@/lib/eta";

/** A compact "step · 42% · ETA ~3m" badge for a running transformation, driven
 * by the `[AUTOTRAIN_PROGRESS]` markers in its live log. Renders nothing when
 * not running or before any progress is reported. */
export function ProgressEta({
  log,
  running,
}: {
  log: string | null | undefined;
  running: boolean;
}) {
  // Only feed the hook real markers while running, so it resets when stopped.
  const marker = parseAutotrainProgress(running ? log : null);
  const eta = useEta(marker, running);

  if (!running) return null;
  const pct = marker?.percent ?? null;
  const etaStr = formatEta(eta);
  const step = prettyStep(marker?.step);
  if (pct === null && etaStr === null && !step) return null;

  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 text-[10px] font-medium tabular-nums text-foreground/80">
      {step && <span className="text-muted-foreground">{step}</span>}
      {pct !== null && <span>{Math.round(pct)}%</span>}
      <span className="text-muted-foreground">{etaStr ? `ETA ${etaStr}` : "estimating…"}</span>
    </span>
  );
}
