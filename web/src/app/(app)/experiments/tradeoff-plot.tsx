"use client";

/**
 * Parallel-coordinates "tradeoff" plot for an experiment's summary.
 *
 * One polyline per cell (target × variant); one vertical axis per field. Reading
 * it: lines that stay high across the output axes are the configurations that
 * win, and a line that crosses others between two axes is a genuine tradeoff
 * (better here, worse there) — which is the whole point of the view. A grid of
 * numbers can't show a crossing.
 *
 * Colour is chosen, not inherited: the two palettes below are the app's existing
 * categorical hues re-ordered so every ADJACENT pair separates under simulated
 * colour-vision deficiency, and the dark set is stepped down to the dark
 * lightness band rather than being an automatic flip of the light one. Both were
 * checked with the palette validator rather than eyeballed.
 */

import { useMemo, useState } from "react";
import type { ExperimentCell, ExperimentSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

// Series colour = fixed order, never cycled by rank; index N is always the same
// cell across re-renders and filters. Light/dark are separate validated sets
// applied via Tailwind variants so there's no theme-detection hydration flash.
//
// ⚠ Both palettes carry a validator WARN that is discharged by SECONDARY
// ENCODING — the left-hand direct labels and the table underneath. Don't remove
// either and leave the colours doing the work alone.
const SERIES_CLASS = [
  "text-[#2563eb] dark:text-[#3b82f6]", // blue
  "text-[#f59e0b] dark:text-[#d97706]", // amber
  "text-[#10b981] dark:text-[#059669]", // emerald
  "text-[#8b5cf6] dark:text-[#8b5cf6]", // violet
  "text-[#06b6d4] dark:text-[#0891b2]", // cyan
  "text-[#ec4899] dark:text-[#ec4899]", // pink
];

type Axis = {
  key: string;
  label: string;
  group: "input" | "output";
  /** Higher is better — drives nothing but the axis hint arrow. */
  higherBetter?: boolean;
  values: number[];
  min: number;
  max: number;
  format: (v: number) => string;
  /** Categorical axes show tick labels instead of a numeric range. */
  categories?: string[];
};

const PAD = { top: 42, right: 28, bottom: 34, left: 150 };
const HEIGHT = 340;

function pct(v: number) {
  return `${Math.round(v * 100)}%`;
}

function ms(v: number) {
  return v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${Math.round(v)}ms`;
}

function usd(v: number) {
  if (v === 0) return "$0";
  return v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(3)}`;
}

function buildAxes(cells: ExperimentCell[], evaluatorIds: string[]): Axis[] {
  const axes: Axis[] = [];

  const targets = [...new Set(cells.map((c) => c.target))];
  const variants = [...new Set(cells.map((c) => c.variant))];

  const categorical = (
    key: string,
    label: string,
    cats: string[],
    pick: (c: ExperimentCell) => string,
  ): Axis => ({
    key,
    label,
    group: "input",
    categories: cats,
    values: cells.map((c) => cats.indexOf(pick(c))),
    min: 0,
    // A single category would give a zero-height axis and divide by zero.
    max: Math.max(1, cats.length - 1),
    format: (v) => cats[Math.round(v)] ?? "",
  });

  if (targets.length > 1 || variants.length <= 1) {
    axes.push(categorical("target", "target", targets, (c) => c.target));
  }
  if (variants.length > 1) {
    axes.push(categorical("variant", "variant", variants, (c) => c.variant));
  }

  const numeric = (
    key: string,
    label: string,
    pick: (c: ExperimentCell) => number | null | undefined,
    format: (v: number) => string,
    higherBetter: boolean,
  ): Axis | null => {
    const vals = cells.map((c) => {
      const v = pick(c);
      return v === null || v === undefined ? NaN : v;
    });
    if (vals.every((v) => Number.isNaN(v))) return null;
    const real = vals.filter((v) => !Number.isNaN(v));
    let min = Math.min(...real);
    let max = Math.max(...real);
    if (min === max) {
      // A flat axis still deserves to render — centre the line on it.
      min = min - 1;
      max = max + 1;
    }
    return { key, label, group: "output", higherBetter, values: vals, min, max, format };
  };

  const push = (a: Axis | null) => {
    if (a) axes.push(a);
  };

  push(numeric("pass_rate", "pass rate", (c) => c.pass_rate, pct, true));
  for (const eid of evaluatorIds) {
    if (eid === "request_error") continue; // shown as error rate below
    push(
      numeric(
        `eval:${eid}`,
        eid.replace(/_/g, " "),
        (c) => c.evals[eid]?.pass_rate ?? null,
        pct,
        true,
      ),
    );
  }
  push(numeric("error_rate", "error rate", (c) => c.error_rate, pct, false));
  push(numeric("p95", "p95 latency", (c) => c.latency_ms.p95, ms, false));
  push(numeric("ttft", "p95 ttft", (c) => c.ttft_ms.p95, ms, false));
  push(
    numeric("tokens", "out tokens", (c) => c.completion_tokens_mean, (v) => String(Math.round(v)), false),
  );
  push(numeric("cost", "cost / req", (c) => c.cost_usd_mean, usd, false));

  return axes;
}

export function TradeoffPlot({
  summary,
  className,
}: {
  summary: ExperimentSummary;
  className?: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const [pinned, setPinned] = useState<number | null>(null);
  const active = pinned ?? hover;

  const cells = summary.cells;
  const axes = useMemo(
    () => buildAxes(cells, summary.evaluator_ids ?? []),
    [cells, summary.evaluator_ids],
  );

  if (!cells.length) {
    return (
      <p className="px-4 py-10 text-center text-sm text-muted-foreground">
        No results to plot yet.
      </p>
    );
  }
  if (axes.length < 2) {
    return (
      <p className="px-4 py-10 text-center text-sm text-muted-foreground">
        Not enough distinct measures to draw a tradeoff plot — see the table below.
      </p>
    );
  }

  // Fixed viewBox + preserveAspectRatio="none" would distort text, so the SVG
  // scales by width only and keeps a constant pixel height.
  const width = Math.max(560, PAD.left + PAD.right + (axes.length - 1) * 132);
  const plotW = width - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const xOf = (i: number) => PAD.left + (plotW * i) / (axes.length - 1);
  const yOf = (axis: Axis, v: number) => {
    if (Number.isNaN(v)) return null;
    const t = (v - axis.min) / (axis.max - axis.min);
    return PAD.top + plotH - t * plotH;
  };

  const inputCount = axes.filter((a) => a.group === "input").length;

  return (
    <div className={cn("w-full", className)}>
      <div className="overflow-x-auto scrollbar-thin">
        <svg
          width={width}
          height={HEIGHT}
          viewBox={`0 0 ${width} ${HEIGHT}`}
          role="img"
          aria-label={`Tradeoff plot comparing ${cells.length} configurations across ${axes.length} measures`}
          className="min-w-full"
          onMouseLeave={() => setHover(null)}
        >
          {/* Input / output band headers, mirroring the axis grouping. */}
          {inputCount > 0 && (
            <text
              x={PAD.left}
              y={16}
              className="fill-muted-foreground text-[10px] font-medium uppercase tracking-wide"
            >
              {inputCount} input{inputCount === 1 ? "" : "s"}
            </text>
          )}
          {axes.length - inputCount > 0 && (
            <text
              x={xOf(inputCount)}
              y={16}
              className="fill-muted-foreground text-[10px] font-medium uppercase tracking-wide"
            >
              {axes.length - inputCount} output{axes.length - inputCount === 1 ? "" : "s"}
            </text>
          )}

          {/* Axes: recessive verticals with min/max end labels. */}
          {axes.map((axis, i) => {
            const x = xOf(i);
            return (
              <g key={axis.key}>
                <line
                  x1={x}
                  y1={PAD.top}
                  x2={x}
                  y2={PAD.top + plotH}
                  className="stroke-border"
                  strokeWidth={1}
                />
                <text
                  x={x}
                  y={PAD.top - 10}
                  textAnchor={i === axes.length - 1 ? "end" : "middle"}
                  className="fill-foreground text-[10px] font-medium"
                >
                  {axis.label}
                </text>
                {axis.categories ? (
                  axis.categories.map((cat, ci) => {
                    const y = yOf(axis, ci);
                    return y === null ? null : (
                      <text
                        key={cat}
                        x={x + 5}
                        y={y + 3}
                        className="fill-muted-foreground text-[9px]"
                      >
                        {cat.length > 16 ? `${cat.slice(0, 15)}…` : cat}
                      </text>
                    );
                  })
                ) : (
                  <>
                    <text
                      x={x + 5}
                      y={PAD.top + 4}
                      className="fill-muted-foreground text-[9px] tabular-nums"
                    >
                      {axis.format(axis.max)}
                    </text>
                    <text
                      x={x + 5}
                      y={PAD.top + plotH + 2}
                      className="fill-muted-foreground text-[9px] tabular-nums"
                    >
                      {axis.format(axis.min)}
                    </text>
                  </>
                )}
              </g>
            );
          })}

          {/* Series. Dimmed when another line is active, so a crossing reads. */}
          {cells.map((cell, ci) => {
            const colour = SERIES_CLASS[ci % SERIES_CLASS.length];
            const pts: Array<[number, number]> = [];
            axes.forEach((axis, ai) => {
              const y = yOf(axis, axis.values[ci]);
              if (y !== null) pts.push([xOf(ai), y]);
            });
            if (pts.length < 2) return null;
            const d = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x},${y}`).join(" ");
            const isActive = active === ci;
            const dim = active !== null && !isActive;
            return (
              <g
                key={`${cell.target}/${cell.variant}`}
                className={cn(colour, "cursor-pointer transition-opacity")}
                opacity={dim ? 0.15 : 1}
                onMouseEnter={() => setHover(ci)}
                onClick={() => setPinned(pinned === ci ? null : ci)}
              >
                {/* Fat invisible hit line — the visible stroke is too thin to aim at. */}
                <path d={d} stroke="transparent" strokeWidth={14} fill="none" />
                <path
                  d={d}
                  stroke="currentColor"
                  strokeWidth={isActive ? 3 : 2}
                  fill="none"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
                {pts.map(([x, y], i) => (
                  <circle
                    key={i}
                    cx={x}
                    cy={y}
                    r={isActive ? 4.5 : 3.5}
                    fill="currentColor"
                    // 2px surface ring keeps overlapping markers separable.
                    className="stroke-background"
                    strokeWidth={2}
                  />
                ))}
                {isActive &&
                  pts.map(([x, y], i) => (
                    <text
                      key={`v${i}`}
                      x={x}
                      y={y - 10}
                      textAnchor="middle"
                      className="fill-foreground text-[10px] font-medium tabular-nums"
                    >
                      {axes[i].format(axes[i].values[ci])}
                    </text>
                  ))}
              </g>
            );
          })}

          {/* Direct labels — identity is never colour-alone. */}
          {cells.map((cell, ci) => {
            const y = yOf(axes[0], axes[0].values[ci]);
            if (y === null) return null;
            const isActive = active === ci;
            const dim = active !== null && !isActive;
            // Stack labels that would collide on a shared categorical value.
            const sameSlot = cells
              .slice(0, ci)
              .filter((_, k) => yOf(axes[0], axes[0].values[k]) === y).length;
            return (
              <g
                key={`lbl-${cell.target}/${cell.variant}`}
                opacity={dim ? 0.3 : 1}
                className="cursor-pointer"
                onMouseEnter={() => setHover(ci)}
                onClick={() => setPinned(pinned === ci ? null : ci)}
              >
                <rect
                  x={8}
                  y={y - 7 + sameSlot * 15}
                  width={7}
                  height={7}
                  rx={1.5}
                  fill="currentColor"
                  className={SERIES_CLASS[ci % SERIES_CLASS.length]}
                />
                <text
                  x={20}
                  y={y + sameSlot * 15}
                  className={cn(
                    "text-[10px] tabular-nums",
                    isActive ? "fill-foreground font-medium" : "fill-muted-foreground",
                  )}
                >
                  {`${cell.target} / ${cell.variant}`.length > 22
                    ? `${`${cell.target} / ${cell.variant}`.slice(0, 21)}…`
                    : `${cell.target} / ${cell.variant}`}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <p className="mt-1 px-1 text-[11px] text-muted-foreground">
        Hover a line to read its values; click to pin.{" "}
        {pinned !== null && (
          <button
            className="underline underline-offset-2 hover:text-foreground"
            onClick={() => setPinned(null)}
          >
            Unpin
          </button>
        )}
      </p>
    </div>
  );
}

export { SERIES_CLASS };
