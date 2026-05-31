"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Loader2, Volume2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { WaveformPlayer } from "@/components/waveform-player";
import { cn } from "@/lib/utils";
import type { DatasetPreview, DatasetPreviewRow } from "@/lib/types";

const PAGE_SIZES = [10, 20, 50];

function audioOf(r: DatasetPreviewRow): string | null {
  const u = r.audio_url;
  if (typeof u !== "string" || !u) return null;
  // Gateway-relative audio (the same-origin proxy) → reach it via /api/proxy;
  // absolute URLs (e.g. HF) are used as-is.
  return u.startsWith("/") ? `/api/proxy${u}` : u;
}

function textOf(r: DatasetPreviewRow): string {
  const t = r.transcription;
  if (t == null) return "";
  return typeof t === "string" ? t : JSON.stringify(t);
}

/**
 * One collapsible row. The waveform player only mounts when expanded, so audio
 * + server-side peaks are fetched lazily (per click) instead of for every row
 * on the page — decoding N clips up front is expensive. Keyed by row index, so
 * it remounts collapsed on page change.
 */
function RowItem({ index, row }: { index: number; row: DatasetPreviewRow }) {
  const [open, setOpen] = useState(false);
  const audio = audioOf(row);
  const text = textOf(row);
  return (
    <div className="overflow-hidden rounded-md border border-border">
      <button
        type="button"
        onClick={() => audio && setOpen((o) => !o)}
        disabled={!audio}
        className={cn(
          "flex w-full items-start gap-2 p-3 text-left transition-colors",
          audio ? "hover:bg-muted/40" : "cursor-default",
        )}
      >
        <ChevronRight
          className={cn(
            "mt-0.5 h-4 w-4 shrink-0 text-muted-foreground transition-transform",
            open && "rotate-90",
            !audio && "opacity-0",
          )}
        />
        <span className="mt-0.5 w-9 shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">
          #{index + 1}
        </span>
        <span className={cn("flex-1 whitespace-pre-wrap break-words text-sm", !open && "line-clamp-2")}>
          {text || <span className="text-muted-foreground">(empty)</span>}
        </span>
        {audio ? (
          <Volume2 className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        ) : (
          <span className="mt-0.5 shrink-0 text-xs text-muted-foreground">no audio</span>
        )}
      </button>
      {open && audio && (
        <div className="border-t border-border p-3">
          <WaveformPlayer src={audio} />
        </div>
      )}
    </div>
  );
}

/**
 * Paginated browser over *all* rows of a dataset — inspect one by one with a
 * waveform player. Seeds from the server-rendered first page, then fetches each
 * page from the gateway on navigation.
 */
export function RowBrowser({
  datasetId,
  initial,
}: {
  datasetId: string;
  initial: DatasetPreview;
}) {
  const [limit, setLimit] = useState(initial.limit && initial.limit > 0 ? initial.limit : 20);
  const [offset, setOffset] = useState(initial.offset ?? 0);
  const [split, setSplit] = useState<string | null>(initial.split ?? null);
  const [splits] = useState<string[]>(initial.splits ?? []);
  const [rows, setRows] = useState<DatasetPreviewRow[]>(initial.rows ?? []);
  const [total, setTotal] = useState<number | null>(initial.total ?? null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(initial.error ?? null);
  // Skip the very first fetch — we already have the server-rendered page.
  const seeded = useRef(true);

  const fetchPage = useCallback(
    async (off: number, lim: number, spl: string | null) => {
      setLoading(true);
      setError(null);
      try {
        const q = new URLSearchParams({ offset: String(off), limit: String(lim) });
        if (spl) q.set("split", spl);
        const r = await fetch(
          `/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}/preview?${q.toString()}`,
          { cache: "no-store" },
        );
        const data = (await r.json()) as DatasetPreview;
        if (!r.ok) {
          setError(data?.error || `Failed to load rows (${r.status})`);
          return;
        }
        setRows(data.rows ?? []);
        if (typeof data.total === "number") setTotal(data.total);
        setError(data.error ?? null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [datasetId],
  );

  useEffect(() => {
    // First render already matches the server-rendered (URL-driven) page.
    if (seeded.current) {
      seeded.current = false;
      return;
    }
    // Reflect pagination in the URL (shareable + survives refresh) without a
    // server round-trip, then fetch the page client-side.
    if (typeof window !== "undefined") {
      const q = new URLSearchParams(window.location.search);
      q.set("offset", String(offset));
      q.set("limit", String(limit));
      if (split) q.set("split", split);
      else q.delete("split");
      window.history.replaceState(null, "", `${window.location.pathname}?${q.toString()}`);
    }
    void fetchPage(offset, limit, split);
  }, [offset, limit, split, fetchPage]);

  const from = total === 0 ? 0 : offset + 1;
  const to = offset + rows.length;
  const hasPrev = offset > 0;
  const hasNext = total != null ? offset + limit < total : rows.length === limit;
  const lastOffset = total != null ? Math.max(0, Math.floor((total - 1) / limit) * limit) : offset;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-3 space-y-0">
        <CardTitle className="text-base">
          Rows{total != null ? ` · ${total.toLocaleString()}` : ""}
        </CardTitle>
        <div className="flex items-center gap-2">
          {splits.length > 1 && (
            <Select
              value={split ?? splits[0]}
              onValueChange={(v) => {
                setOffset(0);
                setSplit(v);
              }}
            >
              <SelectTrigger className="h-8 w-[130px] text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {splits.map((s) => (
                  <SelectItem key={s} value={s} className="text-xs">
                    split: {s}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          <Select
            value={String(limit)}
            onValueChange={(v) => {
              setOffset(0);
              setLimit(Number(v));
            }}
          >
            <SelectTrigger className="h-8 w-[112px] text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PAGE_SIZES.map((n) => (
                <SelectItem key={n} value={String(n)} className="text-xs">
                  {n} / page
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {error ? (
          <p className="text-sm text-destructive">{error}</p>
        ) : rows.length === 0 && !loading ? (
          <p className="text-sm text-muted-foreground">No rows.</p>
        ) : (
          <div className="relative space-y-3">
            {loading && (
              <div className="absolute inset-0 z-10 flex items-center justify-center rounded-md bg-background/60 backdrop-blur-[1px]">
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              </div>
            )}
            {rows.map((r, i) => (
              <RowItem key={offset + i} index={offset + i} row={r} />
            ))}
          </div>
        )}

        {!error && (rows.length > 0 || offset > 0) && (
          <div className="flex items-center justify-between gap-3 pt-1">
            <span className="text-xs text-muted-foreground tabular-nums">
              {from.toLocaleString()}–{to.toLocaleString()}
              {total != null ? ` of ${total.toLocaleString()}` : ""}
            </span>
            <div className="flex items-center gap-1">
              <Button variant="outline" size="icon-sm" disabled={!hasPrev || loading} onClick={() => setOffset(0)} aria-label="First page">
                <ChevronsLeft className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="icon-sm" disabled={!hasPrev || loading} onClick={() => setOffset(Math.max(0, offset - limit))} aria-label="Previous page">
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="icon-sm" disabled={!hasNext || loading} onClick={() => setOffset(offset + limit)} aria-label="Next page">
                <ChevronRight className="h-4 w-4" />
              </Button>
              {total != null && (
                <Button variant="outline" size="icon-sm" disabled={!hasNext || loading} onClick={() => setOffset(lastOffset)} aria-label="Last page">
                  <ChevronsRight className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
