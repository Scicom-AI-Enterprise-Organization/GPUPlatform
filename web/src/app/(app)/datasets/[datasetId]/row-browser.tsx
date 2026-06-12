"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Loader2, Mic, Play, Volume2 } from "lucide-react";
import type { DecoderState } from "./decoder-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { WaveformPlayer } from "@/components/waveform-player";
import { gateway } from "@/lib/gateway";
import { cn } from "@/lib/utils";
import type { DatasetPreview, DatasetPreviewRow } from "@/lib/types";

const PAGE_SIZES = [10, 20, 50];
// Radix <Select> forbids an empty value, so "all speakers" uses a sentinel.
const ALL_SPEAKERS = "__all__";

function audioOf(r: DatasetPreviewRow): string | null {
  const u = r.audio_url;
  if (typeof u !== "string" || !u) return null;
  // `/api/…` is already a same-origin Next route (binary-safe — e.g. the label
  // platform's `label-audio` proxy), so use it directly. Other gateway-relative
  // paths (`/v1/…`) reach the gateway via the generic proxy. Absolute URLs
  // (e.g. HF) are used as-is.
  if (u.startsWith("/api/")) return u;
  return u.startsWith("/") ? `/api/proxy${u}` : u;
}

function textOf(r: DatasetPreviewRow): string {
  const t = r.transcription;
  if (t == null) return "";
  return typeof t === "string" ? t : JSON.stringify(t);
}

/** The row's speaker value (from the dataset's speaker column, default "speaker"). */
function speakerOf(r: DatasetPreviewRow, speakerField?: string | null): string | null {
  const v = r[speakerField || "speaker"];
  if (v == null || typeof v === "object") return null;
  const s = String(v).trim();
  return s || null;
}

/**
 * One collapsible row. The waveform player only mounts when expanded, so audio
 * + server-side peaks are fetched lazily (per click) instead of for every row
 * on the page — decoding N clips up front is expensive. Keyed by row index, so
 * it remounts collapsed on page change.
 */
function RowItem({
  index,
  row,
  onToggle,
  speakerField,
}: {
  index: number;
  row: DatasetPreviewRow;
  onToggle?: (rowIndex: number, included: boolean) => void;
  speakerField?: string | null;
}) {
  const [open, setOpen] = useState(false);
  const audio = audioOf(row);
  const text = textOf(row);
  const speaker = speakerOf(row, speakerField);
  const rowIndex = typeof row.row_index === "number" ? row.row_index : null;
  const included = row.included !== false; // default: included
  return (
    <div
      className={cn(
        "overflow-hidden rounded-md border border-border",
        !included && "border-dashed opacity-55",
      )}
    >
      <div className="flex items-stretch">
        {rowIndex !== null && onToggle && (
          <label
            className="flex shrink-0 cursor-pointer items-center border-r border-border px-2.5 hover:bg-muted/40"
            title={included ? "Included in training — untick to exclude" : "Excluded from training"}
          >
            <Checkbox
              checked={included}
              onCheckedChange={(v) => onToggle(rowIndex, v === true)}
              aria-label="include in training"
            />
          </label>
        )}
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
          {speaker && (
            <span
              className="mt-0.5 inline-flex shrink-0 items-center gap-1 rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
              title={`speaker: ${speaker}`}
            >
              <Mic className="h-3 w-3" />
              {speaker}
            </span>
          )}
          <span className={cn("flex-1 whitespace-pre-wrap break-words text-sm", !open && "line-clamp-2")}>
            {text || <span className="text-muted-foreground">(empty)</span>}
          </span>
          {audio ? (
            <Volume2 className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
          ) : (
            <span className="mt-0.5 shrink-0 text-xs text-muted-foreground">no audio</span>
          )}
        </button>
      </div>
      {open && audio && (
        <div className="border-t border-border p-3">
          <WaveformPlayer src={audio} />
        </div>
      )}
    </div>
  );
}

type PackedDecode = {
  tokenizer: string;
  num_tokens: number;
  num_utterances: number;
  utterances: { tokens: number; text: string }[];
  full_text: string;
};

/**
 * One multipacked block (tts_packed). The header shows token / utterance counts;
 * opening the collapse decodes the block to text via the run's Qwen3 tokenizer
 * (fetched lazily, server-side) so you can inspect what got packed together.
 */
function PackedRowItem({
  datasetId,
  index,
  row,
  split,
  decoder,
}: {
  datasetId: string;
  index: number;
  row: DatasetPreviewRow;
  split?: string | null;
  decoder?: DecoderState | null;
}) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<PackedDecode | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const tokens = typeof row.tokens === "number" ? row.tokens : undefined;
  const utts = typeof row.utterances === "number" ? row.utterances : undefined;
  // Per-utterance audio decode (NeuCodec on the resident decoder). `decoding` is
  // the utterance index currently being decoded; `audioUrls` holds each decoded
  // clip as a data: URL so it gets a full WaveformPlayer (like an audio dataset).
  const [decoding, setDecoding] = useState<number | null>(null);
  const [audioUrls, setAudioUrls] = useState<Record<number, string>>({});
  const [playErr, setPlayErr] = useState<string | null>(null);

  async function decodeUtt(j: number) {
    if (!decoder?.ready) return;
    setDecoding(j);
    setPlayErr(null);
    try {
      const r = await fetch(`/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}/decoder/decode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider_id: decoder.providerId, index, utt: j, split: split ?? null }),
      });
      const audioJson = await r.json();
      if (!r.ok || !audioJson?.wav_b64) {
        throw new Error((audioJson && (audioJson.detail || audioJson.error)) || `decode failed (${r.status})`);
      }
      setAudioUrls((m) => ({ ...m, [j]: `data:audio/wav;base64,${audioJson.wav_b64}` }));
    } catch (e) {
      setPlayErr(`utt ${j + 1}: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setDecoding(null);
    }
  }

  async function toggle() {
    const next = !open;
    setOpen(next);
    if (next && !data && !loading) {
      setLoading(true);
      setErr(null);
      try {
        const r = await fetch(
          `/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}/packed-row?index=${index}` +
            (split ? `&split=${encodeURIComponent(split)}` : ""),
          { cache: "no-store" },
        );
        const j = await r.json();
        if (!r.ok) setErr((j && (j.detail || j.error)) || `decode failed (${r.status})`);
        else setData(j as PackedDecode);
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    }
  }

  return (
    <div className="overflow-hidden rounded-md border border-border">
      <button type="button" onClick={toggle} className="flex w-full items-center gap-2 p-3 text-left transition-colors hover:bg-muted/40">
        <ChevronRight className={cn("h-4 w-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-90")} />
        <span className="w-9 shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">#{index + 1}</span>
        <span className="flex-1 text-sm">
          Packed block · <span className="font-mono">{tokens ?? "?"}</span> tokens ·{" "}
          <span className="font-mono">{utts ?? "?"}</span> utterance{utts === 1 ? "" : "s"}
        </span>
        <span className="shrink-0 text-xs text-muted-foreground">{open ? "hide" : "decode"}</span>
      </button>
      {open && (
        <div className="space-y-2 border-t border-border p-3 text-xs">
          {loading && (
            <span className="inline-flex items-center gap-1.5 text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> decoding with the TTS tokenizer…
            </span>
          )}
          {err && <span className="text-destructive">{err}</span>}
          {data && (
            <>
              <div className="text-[11px] text-muted-foreground">
                {data.num_utterances} utterance{data.num_utterances === 1 ? "" : "s"} multipacked into{" "}
                {data.num_tokens} tokens · decoded with <span className="font-mono">{data.tokenizer}</span>
              </div>
              <ol className="space-y-1.5">
                {data.utterances.map((u, j) => (
                  <li key={j} className="rounded border border-border/60 bg-muted/30 p-2">
                    <div className="mb-0.5 flex items-center gap-2 font-mono text-[10px] text-muted-foreground">
                      {decoder?.ready && (
                        <button
                          type="button"
                          onClick={() => decodeUtt(j)}
                          disabled={decoding !== null}
                          title={audioUrls[j] ? "Re-decode this utterance" : "Decode this utterance to audio"}
                          className="inline-flex h-5 w-5 items-center justify-center rounded text-primary hover:bg-primary/10 disabled:opacity-50"
                        >
                          {decoding === j ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />}
                        </button>
                      )}
                      <span>utt {j + 1} · {u.tokens} tokens</span>
                    </div>
                    <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed scrollbar-thin">{u.text}</pre>
                    {audioUrls[j] && (
                      <div className="mt-2">
                        <WaveformPlayer key={audioUrls[j]} src={audioUrls[j]} />
                      </div>
                    )}
                  </li>
                ))}
              </ol>
              {playErr && <p className="text-destructive">{playErr}</p>}
            </>
          )}
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
  speakerField,
  decoder,
}: {
  datasetId: string;
  initial: DatasetPreview;
  speakerField?: string | null;
  decoder?: DecoderState | null;
}) {
  const [limit, setLimit] = useState(initial.limit && initial.limit > 0 ? initial.limit : 20);
  const [offset, setOffset] = useState(initial.offset ?? 0);
  const [split, setSplit] = useState<string | null>(initial.split ?? null);
  const [splits] = useState<string[]>(initial.splits ?? []);
  // Speaker filter (S3/upload datasets with a speaker column). The list is
  // per-split, so it's refreshed from each fetch.
  const [speaker, setSpeaker] = useState<string | null>(initial.speaker ?? null);
  const [speakers, setSpeakers] = useState<string[]>(initial.speakers ?? []);
  const [rows, setRows] = useState<DatasetPreviewRow[]>(initial.rows ?? []);
  const [total, setTotal] = useState<number | null>(initial.total ?? null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(initial.error ?? null);
  // Manual training-inclusion curation: count of rows un-ticked (excluded).
  const [excludedCount, setExcludedCount] = useState(initial.excluded_count ?? 0);
  const [toggleErr, setToggleErr] = useState<string | null>(null);
  // Skip the very first fetch — we already have the server-rendered page.
  const seeded = useRef(true);

  // Tick/un-tick a row → include/exclude it from training. Optimistic; reverts
  // on failure. The server is the source of truth for the excluded count.
  const setIncluded = useCallback(
    async (rowIndex: number, included: boolean) => {
      setToggleErr(null);
      setRows((prev) => prev.map((r) => (r.row_index === rowIndex ? { ...r, included } : r)));
      try {
        const res = await gateway.setRowInclusion(datasetId, { indices: [rowIndex], included });
        setExcludedCount(res.excluded_count);
      } catch (e) {
        setRows((prev) =>
          prev.map((r) => (r.row_index === rowIndex ? { ...r, included: !included } : r)),
        );
        setToggleErr(e instanceof Error ? e.message : String(e));
      }
    },
    [datasetId],
  );

  const includeAll = useCallback(async () => {
    setToggleErr(null);
    try {
      const res = await gateway.setRowInclusion(datasetId, { clear: true });
      setExcludedCount(res.excluded_count);
      setRows((prev) => prev.map((r) => ({ ...r, included: true })));
    } catch (e) {
      setToggleErr(e instanceof Error ? e.message : String(e));
    }
  }, [datasetId]);

  const fetchPage = useCallback(
    async (off: number, lim: number, spl: string | null, spk: string | null) => {
      setLoading(true);
      setError(null);
      try {
        const q = new URLSearchParams({ offset: String(off), limit: String(lim) });
        if (spl) q.set("split", spl);
        if (spk) q.set("speaker", spk);
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
        if (Array.isArray(data.speakers)) setSpeakers(data.speakers);
        if (typeof data.excluded_count === "number") setExcludedCount(data.excluded_count);
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
      if (speaker) q.set("speaker", speaker);
      else q.delete("speaker");
      window.history.replaceState(null, "", `${window.location.pathname}?${q.toString()}`);
    }
    void fetchPage(offset, limit, split, speaker);
  }, [offset, limit, split, speaker, fetchPage]);

  const from = total === 0 ? 0 : offset + 1;
  const to = offset + rows.length;
  const hasPrev = offset > 0;
  const hasNext = total != null ? offset + limit < total : rows.length === limit;
  const lastOffset = total != null ? Math.max(0, Math.floor((total - 1) / limit) * limit) : offset;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-3 space-y-0">
        <div className="flex flex-col gap-0.5">
          <CardTitle className="text-base">
            Rows{total != null ? ` · ${total.toLocaleString()}` : ""}
          </CardTitle>
          {excludedCount > 0 ? (
            <span className="text-xs text-muted-foreground">
              {excludedCount.toLocaleString()} excluded from training ·{" "}
              <button type="button" onClick={includeAll} className="underline underline-offset-2 hover:text-foreground">
                include all
              </button>
            </span>
          ) : (
            <span className="text-xs text-muted-foreground">Untick a row to exclude it from training.</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {splits.length > 1 && (
            <Select
              value={split ?? splits[0]}
              onValueChange={(v) => {
                setOffset(0);
                setSpeaker(null); // speaker lists are per-split; reset on split change
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
          {speakers.length > 1 && (
            <Select
              value={speaker ?? ALL_SPEAKERS}
              onValueChange={(v) => {
                setOffset(0);
                setSpeaker(v === ALL_SPEAKERS ? null : v);
              }}
            >
              <SelectTrigger className="h-8 w-[150px] text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL_SPEAKERS} className="text-xs">
                  all speakers
                </SelectItem>
                {speakers.map((sp) => (
                  <SelectItem key={sp} value={sp} className="font-mono text-xs">
                    🎤 {sp}
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
            {rows.map((r, i) =>
              r.packed === true ? (
                <PackedRowItem key={offset + i} datasetId={datasetId} index={offset + i} row={r} split={split} decoder={decoder} />
              ) : (
                <RowItem key={offset + i} index={offset + i} row={r} onToggle={setIncluded} speakerField={speakerField} />
              ),
            )}
          </div>
        )}

        {toggleErr && <p className="text-xs text-destructive">Couldn’t save selection: {toggleErr}</p>}

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
