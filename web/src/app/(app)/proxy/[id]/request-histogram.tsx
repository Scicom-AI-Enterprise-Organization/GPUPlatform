"use client";

/**
 * Request volume over time — the Queue tab's Kibana-Discover-style header.
 *
 * Three controls, and the reason each exists:
 *   * **time range** — the row list is one page of a possibly enormous history, so
 *     "which slice am I looking at" has to be explicit rather than implied by
 *     whatever the page size happens to be.
 *   * **interval** — `Auto` asks the SERVER to pick (it knows the real span and targets
 *     ~50 bars); the explicit rungs pin it. The resolved value is always displayed,
 *     because an unlabelled x-axis on an auto-scaled chart is unreadable.
 *   * **status legend** — doubles as a filter. `blocked` is the series people open this
 *     tab for, so it gets the amber that the row badges already use.
 *
 * Bars are stacked by status and drawn as plain divs: a dependency-free chart is worth
 * more here than a charting library, since the only interaction is hover + click.
 */

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import type { ProxyRequestCounts } from "@/lib/types";

/**
 * Default window when `?range=` is absent. A Queue tab opened on an endpoint with months
 * of history should not aggregate all of it just to show the last few requests.
 *
 * ⚠ Callers must write the param on EVERY change, including `0` ("All time"). Omitting it
 * for the falsy value would delete `?range=` and snap straight back to this default, so
 * "All time" would be unselectable.
 */
export const DEFAULT_RANGE_H = 24 * 7;

/** Read `?range=` with the default applied: absent → 7d, explicit `0` → all time. */
export function rangeFromParam(v: string | null): number {
  return v === null ? DEFAULT_RANGE_H : (Number(v) || 0);
}

/** Hours back, or 0 for "all time". Mirrors the gateway's `since_hours`. */
export const RANGES: { label: string; hours: number }[] = [
  { label: "Last 15 minutes", hours: 0.25 },
  { label: "Last hour", hours: 1 },
  { label: "Last 6 hours", hours: 6 },
  { label: "Last 24 hours", hours: 24 },
  { label: "Last 7 days", hours: 24 * 7 },
  { label: "Last 30 days", hours: 24 * 30 },
  { label: "All time", hours: 0 },
];

/** Must stay a subset of the gateway's `_BUCKETS` keys, or the server 400s. */
export const INTERVALS = ["auto", "1m", "5m", "10m", "30m", "1h", "3h", "6h", "12h", "1d"];

// Same status colours the row badges use, so the chart and the table agree at a glance.
export type HistogramSeries = { key: string; label: string; cls: string };

const SERIES: HistogramSeries[] = [
  { key: "completed", label: "completed", cls: "bg-emerald-500" },
  { key: "blocked", label: "blocked", cls: "bg-amber-500" },
  { key: "failed", label: "failed", cls: "bg-rose-500" },
  { key: "cancelled", label: "cancelled", cls: "bg-slate-400" },
  { key: "running", label: "running", cls: "bg-sky-500" },
  { key: "queued", label: "queued", cls: "bg-indigo-400" },
];

// The inference queue's status vocabulary differs (no `blocked`; adds `pending`,
// `timeout`, `ready`) — same chart, different legend.
export const APP_SERIES: HistogramSeries[] = [
  { key: "completed", label: "completed", cls: "bg-emerald-500" },
  { key: "ready", label: "ready", cls: "bg-emerald-400" },
  { key: "failed", label: "failed", cls: "bg-rose-500" },
  { key: "timeout", label: "timeout", cls: "bg-amber-500" },
  { key: "cancelled", label: "cancelled", cls: "bg-slate-400" },
  { key: "running", label: "running", cls: "bg-sky-500" },
  { key: "queued", label: "queued", cls: "bg-indigo-400" },
  { key: "pending", label: "pending", cls: "bg-indigo-300" },
];

/**
 * The time-range picker, exported separately from the chart.
 *
 * It scopes the WHOLE tab — chart, status tabs, table and pager — so it belongs in the
 * filter row next to "All users" / "All upstreams", not inside the chart card where it
 * read as a chart-only control. The interval dropdown stays on the card because it
 * genuinely only affects the chart (this is the split Kibana uses too).
 */
export function RangeSelect({ hours, onHours, className }: {
  hours: number;
  onHours: (h: number) => void;
  className?: string;
}) {
  return (
    <Select value={String(hours)} onValueChange={(v) => onHours(Number(v))}>
      <SelectTrigger className={className ?? "h-8 w-[150px] text-xs"}><SelectValue /></SelectTrigger>
      <SelectContent>
        {RANGES.map((r) => (
          <SelectItem key={r.label} value={String(r.hours)} className="text-xs">
            {r.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function fmtTick(iso: string, seconds: number): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  // Sub-day buckets want a clock; day-or-coarser wants a date. Showing both everywhere
  // is what makes these axes unreadable at one end of the range or the other.
  return seconds < 86400
    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString([], { month: "short", day: "numeric" });
}

export function RequestHistogram({
  counts, interval, onInterval, statusFilter = "", onStatusFilter,
  series = SERIES, onBrush, loading,
}: {
  counts: ProxyRequestCounts | null;
  interval: string;
  onInterval: (i: string) => void;
  /** Which status series to draw + legend, in stacking order. Defaults to the proxy's. */
  series?: HistogramSeries[];
  /**
   * Drag-to-select finished: absolute ISO bounds for the swept span. Omit to disable
   * brushing. Timestamps are derived from the server's `axis_from`/`axis_to`, never from
   * the client's idea of the window — an open-ended request resolves against the data,
   * which only the server knows.
   */
  onBrush?: (fromISO: string, toISO: string) => void;
  statusFilter?: string;
  /**
   * Omit to make the legend display-only. The inference queue does: its tab filter works
   * on coarse display buckets ("failed" covers failed + timeout + cancelled), so a click
   * on a fine-grained status would filter to something wider than the label promises.
   */
  onStatusFilter?: (s: string) => void;
  loading?: boolean;
}) {
  // `counts?.buckets ?? []` is a fresh array identity on every render when buckets are
  // absent, which would make both memos below recompute every time. Memoise the fallback.
  const buckets = useMemo(() => counts?.buckets ?? [], [counts]);
  const secs = counts?.bucket_seconds ?? 3600;
  const peak = useMemo(
    () => Math.max(1, ...buckets.map((b) => b.total)),
    [buckets],
  );
  // Only render a series that actually occurs — an always-on legend of six statuses
  // implies traffic that isn't there.
  const present = useMemo(
    () => series.filter((s) => (counts?.by_status?.[s.key] ?? 0) > 0),
    [counts, series],
  );

  const resolved = counts?.bucket ?? null;
  const isAuto = (counts?.bucket_requested ?? interval) === "auto";

  // Hovered bar index. Local state, so the tab's 4s poll re-render doesn't disturb it
  // (a native tooltip's timer, by contrast, was being restarted by those re-renders).
  const [hover, setHover] = useState<number | null>(null);
  const hovered = hover !== null ? buckets[hover] : undefined;
  // Centre of the hovered bar, 0..1 across the chart. Bars are equal-width flex children,
  // so the geometry is exact without measuring the DOM.
  const hoverPct = hover !== null && buckets.length
    ? (hover + 0.5) / buckets.length
    : 0;

  // Drag-to-select, as fractions 0..1 of the plot width. Kept in state (not painted
  // imperatively) because the plot element's identity is stable across React renders, so
  // the pointer capture survives the tab's 4s poll.
  const [brush, setBrush] = useState<{ a: number; b: number } | null>(null);
  const canBrush = Boolean(onBrush && counts?.axis_from && counts?.axis_to);

  const pctAt = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return Math.min(1, Math.max(0, (e.clientX - r.left) / (r.width || 1)));
  };

  const endBrush = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!brush || !canBrush) { setBrush(null); return; }
    const lo = Math.min(brush.a, brush.b);
    const hi = Math.max(brush.a, brush.b);
    setBrush(null);
    e.currentTarget.releasePointerCapture?.(e.pointerId);
    // A click is a zero-width drag. Zooming to an instant would empty the table and read
    // as a bug, so require a real sweep before acting.
    if (hi - lo < 0.01) return;
    const from = new Date(counts!.axis_from!).getTime();
    const to = new Date(counts!.axis_to!).getTime();
    const at = (p: number) => new Date(from + (to - from) * p).toISOString();
    onBrush!(at(lo), at(hi));
  };

  return (
    <div className="space-y-2 rounded-md border bg-card p-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="mr-auto flex items-baseline gap-2">
          <span className="text-lg font-semibold tabular-nums">
            {counts ? counts.total.toLocaleString() : "—"}
          </span>
          <span className="text-xs text-muted-foreground">
            request{counts?.total === 1 ? "" : "s"}
            {/* An approximate total must never be shown as though it were exact. */}
            {counts && counts.exact === false ? " (at least — approximate)" : ""}
          </span>
        </div>

        <Select value={interval} onValueChange={onInterval}>
          <SelectTrigger className="h-7 w-[124px] text-xs"><SelectValue /></SelectTrigger>
          <SelectContent>
            {INTERVALS.map((i) => (
              <SelectItem key={i} value={i} className="text-xs">
                {i === "auto" ? "Auto interval" : i}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* ⚠ Do NOT put the breakdown back on a `title` attribute. That renders an OS
          tooltip, which browsers delay ~1-2s before showing and offer no way to speed up
          — it reads as the chart being slow. This one appears on mouseenter. */}
      <div className="flex gap-1">
        {/* Y axis. Three ticks (peak / half / 0) is enough to read magnitude off the
            chart; more would crowd a 104px plot. `tabular-nums` stops the labels
            shifting width as the peak changes between polls. */}
        <div className="flex h-[104px] w-9 shrink-0 flex-col justify-between pr-1
                        text-right text-[10px] tabular-nums text-muted-foreground">
          <span>{buckets.length ? peak.toLocaleString() : ""}</span>
          <span>{buckets.length ? Math.round(peak / 2).toLocaleString() : ""}</span>
          <span>{buckets.length ? "0" : ""}</span>
        </div>

      {/* `border-b` matters more than it looks: the server now zero-fills empty periods,
          so most columns can be 0px tall. Without a baseline the chart reads as a few
          floating bars rather than a continuous axis with gaps in it. */}
      <div className={`relative flex h-[104px] flex-1 items-end gap-px border-b border-border/60
                       ${canBrush ? "cursor-crosshair select-none" : ""}`}
           onMouseLeave={() => setHover(null)}
           onPointerDown={canBrush ? (e) => {
             if (e.button) return;
             e.currentTarget.setPointerCapture?.(e.pointerId);
             const p = pctAt(e);
             setBrush({ a: p, b: p });
           } : undefined}
           onPointerMove={canBrush ? (e) => {
             if (brush) setBrush({ a: brush.a, b: pctAt(e) });
           } : undefined}
           onPointerUp={canBrush ? endBrush : undefined}
           onPointerCancel={canBrush ? endBrush : undefined}>
        {/* Dotted gridlines at the peak and half-peak levels, aligned with the y-axis
            ticks beside them — that alignment is the whole point, since it is what lets a
            bar's height be read as a number. The 0 line is the container's own border-b.
            Behind the bars and pointer-transparent so it never eats a brush drag. */}
        {buckets.length > 0 && (
          <div className="pointer-events-none absolute inset-0">
            <div className="absolute inset-x-0 top-0 border-t border-dashed border-border/70" />
            <div className="absolute inset-x-0 top-1/2 border-t border-dashed border-border/70" />
          </div>
        )}
        {buckets.length === 0 ? (
          <div className="flex h-full w-full items-center justify-center text-xs text-muted-foreground">
            {loading ? "loading…"
              : counts?.source === "trace"
                ? "the trace store cannot produce a histogram — switch PROXY_HISTORY_SOURCE to db for the chart"
                : "no requests in this range"}
          </div>
        ) : (
          buckets.map((b, i) => (
            <div key={b.ts}
                 className={`relative flex h-full flex-1 flex-col justify-end ${
                   hover === i ? "bg-foreground/5" : ""}`}
                 onMouseEnter={() => setHover(i)}>
              {/* Stack from the bottom; order follows `series` so colours stay stable.
                  `min-h-[2px]`: one request against a peak of 132 is 0.8px, which is
                  indistinguishable from the empty bucket beside it — and "was there any
                  traffic here" is the one distinction this chart exists to make. Only
                  non-zero segments are rendered at all, so a genuine nothing stays
                  nothing. */}
              {series.filter((s) => b.by_status[s.key]).map((s) => (
                <div key={s.key} className={`${s.cls} min-h-[2px] w-full`}
                     style={{ height: `${((b.by_status[s.key] ?? 0) / peak) * 100}%` }} />
              ))}
            </div>
          ))
        )}

        {hovered && (
          // Follows the hovered BAR rather than sticking to a container edge. `left` is
          // the bar's centre as a percentage; the translate then centres the panel on it,
          // easing to 0%/-100% within the outer 15% so it cannot overflow the card at
          // either end. Percentages have to be an inline style — the value is only known
          // at runtime, so there is no Tailwind class for it.
          <div style={{
                 left: `${hoverPct * 100}%`,
                 transform: `translateX(${hoverPct < 0.15 ? 0 : hoverPct > 0.85 ? -100 : -50}%)`,
               }}
               className="pointer-events-none absolute bottom-full z-10 mb-1 w-max rounded-md
                          border bg-popover px-2 py-1.5 text-[11px] shadow-md">
            <div className="font-medium">{new Date(hovered.ts).toLocaleString()}</div>
            <div className="text-muted-foreground">
              {/* Zero-filled periods are hoverable too — "0 requests" is the answer to
                  "was it quiet or is the chart broken", so it must be sayable. */}
              {hovered.total.toLocaleString()} request{hovered.total === 1 ? "" : "s"}
            </div>
            {series.filter((s) => hovered.by_status[s.key]).map((s) => (
              <div key={s.key} className="flex items-center gap-1.5">
                <span className={`inline-block h-2 w-2 rounded-sm ${s.cls}`} />
                <span>{s.label}</span>
                <span className="ml-auto pl-2 tabular-nums">{hovered.by_status[s.key]}</span>
              </div>
            ))}
          </div>
        )}

        {brush && Math.abs(brush.b - brush.a) > 0.002 && (
          // Drawn over the bars, ignoring pointer events so the capture stays on the plot.
          <div className="pointer-events-none absolute inset-y-0 border-x border-primary
                          bg-primary/20"
               style={{
                 left: `${Math.min(brush.a, brush.b) * 100}%`,
                 width: `${Math.abs(brush.b - brush.a) * 100}%`,
               }} />
        )}
      </div>
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 pl-10 text-[11px] text-muted-foreground">
        {buckets.length > 0 && (
          <span className="tabular-nums">
            {fmtTick(buckets[0].ts, secs)} → {fmtTick(buckets[buckets.length - 1].ts, secs)}
          </span>
        )}
        {resolved && (
          <span>interval: {isAuto ? `Auto — ${resolved}` : resolved}</span>
        )}
        {canBrush && <span>· drag to narrow</span>}
        <span className="ml-auto flex flex-wrap items-center gap-2">
          {present.map((s) => {
            const n = (counts?.by_status?.[s.key] ?? 0).toLocaleString();
            // No handler → a plain swatch. Rendering a Button that does nothing invites
            // the click and then ignores it, which reads as a broken filter.
            return onStatusFilter ? (
              <Button key={s.key} variant="ghost" size="xs"
                      className={`h-5 gap-1 px-1 text-[11px] ${statusFilter === s.key ? "bg-accent" : ""}`}
                      title={`Show only ${s.label} requests`}
                      onClick={() => onStatusFilter(statusFilter === s.key ? "" : s.key)}>
                <span className={`inline-block h-2 w-2 rounded-sm ${s.cls}`} />
                {s.label}
                <span className="tabular-nums">{n}</span>
              </Button>
            ) : (
              <span key={s.key} className="flex items-center gap-1">
                <span className={`inline-block h-2 w-2 rounded-sm ${s.cls}`} />
                {s.label}
                <span className="tabular-nums">{n}</span>
              </span>
            );
          })}
        </span>
      </div>

      {counts?.note && (
        <p className="text-[11px] text-amber-600 dark:text-amber-400">{counts.note}</p>
      )}
    </div>
  );
}
