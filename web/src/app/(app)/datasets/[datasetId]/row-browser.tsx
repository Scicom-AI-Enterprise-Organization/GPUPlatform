"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Bot, ChevronDown, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Loader2, Mic, Play, Terminal, User, Volume2, Wrench } from "lucide-react";
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
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
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

// ── LLM / chat row ──────────────────────────────────────────────────────────

type ToolCall = {
  id: string;
  type: string;
  function: { name: string; arguments: string };
};

type ChatMessage = {
  role: string;
  content: unknown;
  reasoning?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string; // tool response: the function name
};

function isChatMessages(v: unknown): v is ChatMessage[] {
  return (
    Array.isArray(v) && v.length > 0 &&
    typeof (v[0] as Record<string, unknown>)?.role === "string"
  );
}

function contentStr(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((p) => {
        if (typeof p === "string") return p;
        if (p && typeof p === "object") {
          const part = p as Record<string, unknown>;
          if (part.type === "text" && typeof part.text === "string") return part.text;
          if (typeof part.content === "string") return part.content;
        }
        return JSON.stringify(p);
      })
      .join("\n");
  }
  return JSON.stringify(content);
}

function tryParseJson(s: string): unknown {
  try { return JSON.parse(s); } catch { return s; }
}

/** Collapsible reasoning trace block shown inside a bubble.
 *  `invert` = true when the bubble has a dark/primary background (user messages)
 *  — use opacity-based colours so the text remains legible against any tint. */
function ReasoningBlock({ text, invert = false }: { text: string; invert?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={cn("mb-2 rounded border text-[11px]",
      invert ? "border-white/20" : "border-border/50",
    )}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={cn(
          "flex w-full items-center gap-1 px-2 py-1 text-left",
          invert ? "text-white/60 hover:text-white/90" : "text-muted-foreground hover:text-foreground",
        )}
      >
        <ChevronDown className={cn("h-3 w-3 shrink-0 transition-transform", !open && "-rotate-90")} />
        <span className="font-medium">reasoning</span>
      </button>
      {open && (
        <p className={cn(
          "whitespace-pre-wrap break-words border-t px-2 py-1.5 italic",
          invert ? "border-white/20 text-white/70" : "border-border/50 text-muted-foreground",
        )}>
          {text}
        </p>
      )}
    </div>
  );
}

/** JSON value with syntax colouring — strings green, numbers amber, booleans/null blue. */
function JsonValue({ value, depth = 0 }: { value: unknown; depth?: number }) {
  const [collapsed, setCollapsed] = useState(depth > 0);
  const indent = "  ".repeat(depth);
  const innerIndent = "  ".repeat(depth + 1);

  if (value === null) return <span className="text-blue-400">null</span>;
  if (typeof value === "boolean") return <span className="text-blue-400">{String(value)}</span>;
  if (typeof value === "number") return <span className="text-amber-400">{String(value)}</span>;
  if (typeof value === "string") return <span className="text-emerald-400">&quot;{value}&quot;</span>;

  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="text-muted-foreground">[]</span>;
    return (
      <span>
        <button type="button" onClick={() => setCollapsed((c) => !c)} className="text-muted-foreground hover:text-foreground">
          {collapsed ? `[… ${value.length}]` : "["}
        </button>
        {!collapsed && (
          <>
            {value.map((item, i) => (
              <div key={i} style={{ paddingLeft: "1.2em" }}>
                <JsonValue value={item} depth={depth + 1} />
                {i < value.length - 1 && <span className="text-muted-foreground">,</span>}
              </div>
            ))}
            <span className="text-muted-foreground">{indent}]</span>
          </>
        )}
      </span>
    );
  }

  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) return <span className="text-muted-foreground">{"{}"}</span>;
    return (
      <span>
        <button type="button" onClick={() => setCollapsed((c) => !c)} className="text-muted-foreground hover:text-foreground">
          {collapsed ? `{… ${entries.length}}` : "{"}
        </button>
        {!collapsed && (
          <>
            {entries.map(([k, v], i) => (
              <div key={k} style={{ paddingLeft: "1.2em" }}>
                <span className="text-sky-300">&quot;{k}&quot;</span>
                <span className="text-muted-foreground">: </span>
                <JsonValue value={v} depth={depth + 1} />
                {i < entries.length - 1 && <span className="text-muted-foreground">,</span>}
              </div>
            ))}
            <span className="text-muted-foreground">{indent}{"}"}</span>
          </>
        )}
      </span>
    );
  }

  return <span>{String(value)}</span>;
}

/** Renders tool response content: JSON gets a collapsible syntax-coloured tree;
 *  plain text falls back to a pre block. */
function ToolContent({ content }: { content: unknown }) {
  const str = contentStr(content);
  const [expanded, setExpanded] = useState(false);

  let parsed: unknown = null;
  let isJson = false;
  try {
    parsed = JSON.parse(str);
    isJson = parsed !== null && typeof parsed === "object";
  } catch { /* not JSON */ }

  if (!isJson) {
    return <p className="whitespace-pre-wrap break-words text-sm">{str}</p>;
  }

  return (
    <div className="rounded border border-emerald-500/20 bg-emerald-500/5">
      <button
        type="button"
        onClick={() => setExpanded((e) => !e)}
        className="flex w-full items-center gap-1 px-2 py-1.5 text-left text-xs text-emerald-400/80 hover:text-emerald-300"
      >
        <ChevronDown className={cn("h-3 w-3 shrink-0 transition-transform", !expanded && "-rotate-90")} />
        <span className="font-mono font-medium">{"{}"} JSON response</span>
        {!expanded && (
          <span className="ml-auto text-[10px] text-muted-foreground">
            {Object.keys(parsed as object).join(", ").slice(0, 60)}
          </span>
        )}
      </button>
      {expanded && (
        <pre className="overflow-x-auto border-t border-emerald-500/20 px-3 py-2 text-[11px] leading-relaxed">
          <JsonValue value={parsed} depth={0} />
        </pre>
      )}
    </div>
  );
}

/** One tool call inside an assistant bubble — collapsible JSON args, amber theme. */
function ToolCallArgs({ tc }: { tc: ToolCall }) {
  const [expanded, setExpanded] = useState(false);
  const args = typeof tc.function.arguments === "string"
    ? tryParseJson(tc.function.arguments)
    : tc.function.arguments;
  const isObj = args !== null && typeof args === "object";

  return (
    <div className="rounded border border-amber-500/30 bg-amber-500/5 text-xs">
      <button
        type="button"
        onClick={() => setExpanded((e) => !e)}
        className="flex w-full items-center gap-1 px-2 py-1.5 text-left font-mono text-amber-400 hover:text-amber-300"
      >
        <ChevronDown className={cn("h-3 w-3 shrink-0 transition-transform", !expanded && "-rotate-90")} />
        <span className="font-semibold">{tc.function.name}</span>
        {!expanded && isObj && (
          <span className="ml-1 text-[10px] text-amber-400/60">
            {Object.keys(args as object).join(", ").slice(0, 60)}
          </span>
        )}
      </button>
      {expanded && (
        <pre className="overflow-x-auto border-t border-amber-500/20 px-3 py-2 leading-relaxed">
          <JsonValue value={args} depth={0} />
        </pre>
      )}
    </div>
  );
}

/** Chat-bubble rendering for a message list — shared by the SFT row view and the
 * DPO chosen/rejected view. Per-role colours, reasoning traces, tool calls. */
function ChatBubbles({ msgs }: { msgs: ChatMessage[] }) {
  return (
    <div className="space-y-2">
      {msgs.map((m, i) => {
        const isUser = m.role === "user";
        const isSystem = m.role === "system";
        const isTool = m.role === "tool";
        const hasToolCalls = (m.tool_calls?.length ?? 0) > 0;

        // ── avatar ──
        const avatarCls = cn(
          "flex h-6 w-6 shrink-0 items-center justify-center rounded-full",
          isUser        ? "bg-primary text-primary-foreground"
          : isTool      ? "bg-emerald-600/20 text-emerald-500"
          : hasToolCalls? "bg-amber-500/20 text-amber-500"
          : isSystem    ? "bg-muted text-muted-foreground"
          :               "bg-violet-500/20 text-violet-400",
        );
        const AvatarIcon = isUser ? User : isTool ? Terminal : hasToolCalls ? Wrench : Bot;

        // ── bubble ──
        const bubbleCls = cn(
          "max-w-[85%] rounded-lg px-3 py-2 text-sm",
          isUser        ? "bg-primary text-primary-foreground"
          : isTool      ? "border border-emerald-500/30 bg-emerald-500/10 text-foreground"
          : hasToolCalls? "border border-amber-500/30 bg-amber-500/10 text-foreground"
          : isSystem    ? "bg-muted/50 text-muted-foreground italic"
          :               "border border-violet-500/20 bg-violet-500/10 text-foreground",
        );

        const roleLabel =
          isUser        ? null
          : isTool      ? `tool · ${m.name ?? ""}`
          : hasToolCalls? "assistant · tool call"
          : isSystem    ? "system"
          :               "assistant";

        return (
          <div key={i} className={cn("flex items-start gap-2", isUser ? "flex-row-reverse" : "flex-row")}>
            {/* avatar */}
            <div className={avatarCls} title={m.role}>
              <AvatarIcon className="h-3 w-3" />
            </div>

            {/* bubble */}
            <div className={bubbleCls}>
              {roleLabel && (
                <span className="mb-1 block text-[10px] font-semibold uppercase tracking-wide opacity-60">
                  {roleLabel}
                </span>
              )}

              {/* reasoning trace (collapsible) */}
              {m.reasoning && <ReasoningBlock text={m.reasoning} invert={isUser} />}

              {/* main content — tool responses get a collapsible JSON tree */}
              {isTool ? (
                <ToolContent content={m.content} />
              ) : contentStr(m.content) ? (
                <p className="whitespace-pre-wrap break-words">{contentStr(m.content)}</p>
              ) : null}

              {/* tool calls list */}
              {hasToolCalls && (
                <div className="mt-2 space-y-1.5">
                  {m.tool_calls!.map((tc) => (
                    <ToolCallArgs key={tc.id} tc={tc} />
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** One LLM / multimodal row — chat bubbles with per-role colours, reasoning traces, tool calls. */
function LlmRowItem({
  index,
  row,
  messagesField,
}: {
  index: number;
  row: DatasetPreviewRow;
  messagesField: string;
}) {
  const [open, setOpen] = useState(false);
  const raw = row[messagesField] ?? row.messages;
  const msgs = isChatMessages(raw) ? raw : null;
  const audio = audioOf(row);
  const preview = msgs ? msgs.slice(0, 1) : null;

  return (
    <div className="overflow-hidden rounded-md border border-border">
      {/* ── collapsed header ── */}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-start gap-2 p-3 text-left transition-colors hover:bg-muted/40"
      >
        <ChevronRight className={cn("mt-0.5 h-4 w-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-90")} />
        <span className="mt-0.5 w-9 shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">
          #{index + 1}
        </span>
        <span className="flex-1 space-y-0.5">
          {msgs == null ? (
            <span className="text-sm text-muted-foreground">(no messages)</span>
          ) : open ? (
            <span className="text-xs text-muted-foreground">{msgs.length} message{msgs.length !== 1 ? "s" : ""}</span>
          ) : (
            preview?.map((m, i) => (
              <span key={i} className="block truncate text-sm">
                <span className="mr-1.5 font-medium capitalize text-muted-foreground">{m.role}:</span>
                {contentStr(m.content).slice(0, 120)}
              </span>
            ))
          )}
        </span>
        <div className="mt-0.5 flex shrink-0 items-center gap-1.5 text-xs text-muted-foreground">
          {audio && <Volume2 className="h-3.5 w-3.5" />}
          {msgs != null && !open && <span>{msgs.length} msg</span>}
        </div>
      </button>

      {/* ── expanded body ── */}
      {open && (
        <div className="space-y-3 border-t border-border p-3">
          {audio && <WaveformPlayer src={audio} />}
          {msgs && <ChatBubbles msgs={msgs} />}
        </div>
      )}
    </div>
  );
}

/**
 * One DPO preference row — the shared prompt shown once, then the chosen (✓) and
 * rejected (✕) responses side by side. `chosenField`/`rejectedField` name the two
 * message-list columns (chosen defaults to the dataset's messages column).
 */
function DpoRowItem({
  index,
  row,
  chosenField,
  rejectedField,
}: {
  index: number;
  row: DatasetPreviewRow;
  chosenField: string;
  rejectedField: string;
}) {
  const [open, setOpen] = useState(false);
  const chosen = isChatMessages(row[chosenField]) ? (row[chosenField] as ChatMessage[]) : null;
  const rejected = isChatMessages(row[rejectedField]) ? (row[rejectedField] as ChatMessage[]) : null;

  // Chosen & rejected agree on the prompt turns; find how far, so the shared prompt
  // is shown once and only the divergent tails go under chosen/rejected.
  let shared = 0;
  if (chosen && rejected) {
    const n = Math.min(chosen.length, rejected.length);
    while (shared < n && JSON.stringify(chosen[shared]) === JSON.stringify(rejected[shared])) shared++;
  }
  const prompt = chosen ? chosen.slice(0, shared) : [];
  const chosenTail = chosen ? chosen.slice(shared) : [];
  const rejectedTail = rejected ? rejected.slice(shared) : [];
  const firstUser = prompt.find((m) => m.role === "user") ?? prompt[0] ?? chosen?.[0];

  return (
    <div className="overflow-hidden rounded-md border border-border">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-start gap-2 p-3 text-left transition-colors hover:bg-muted/40"
      >
        <ChevronRight className={cn("mt-0.5 h-4 w-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-90")} />
        <span className="mt-0.5 w-9 shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">#{index + 1}</span>
        <span className="flex-1 space-y-0.5">
          {chosen == null || rejected == null ? (
            <span className="text-sm text-muted-foreground">
              (row is missing a valid <span className="font-mono">{chosenField}</span> /{" "}
              <span className="font-mono">{rejectedField}</span> message list)
            </span>
          ) : (
            <span className="block truncate text-sm">
              <span className="mr-1.5 font-medium text-muted-foreground">preference pair:</span>
              {contentStr(firstUser?.content).slice(0, 120)}
            </span>
          )}
        </span>
        <span className="mt-0.5 shrink-0 text-xs text-muted-foreground">{open ? "hide" : "compare"}</span>
      </button>

      {open && chosen && rejected && (
        <div className="space-y-3 border-t border-border p-3">
          {prompt.length > 0 && (
            <div className="space-y-1">
              <div className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">prompt</div>
              <ChatBubbles msgs={prompt} />
            </div>
          )}
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div className="rounded-md border border-emerald-600/40 bg-emerald-500/5 p-2">
              <div className="mb-1 font-mono text-[10px] font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-400">
                ✓ chosen
              </div>
              <ChatBubbles msgs={chosenTail.length ? chosenTail : chosen} />
            </div>
            <div className="rounded-md border border-destructive/40 bg-destructive/5 p-2">
              <div className="mb-1 font-mono text-[10px] font-semibold uppercase tracking-wide text-destructive">
                ✕ rejected
              </div>
              <ChatBubbles msgs={rejectedTail.length ? rejectedTail : rejected} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── TTS packed row ───────────────────────────────────────────────────────────

type PackedUtt = { tokens: number; text: string };
type PackedDecode = {
  tokenizer: string;
  num_tokens: number;
  num_utterances: number;
  utterances: PackedUtt[];
  full_text: string;
  // DPO packs (kind=llm_dpo_packed) also return preference pairs so the block can
  // render chosen ↔ rejected side by side instead of a flat 2K-utterance list.
  objective?: string;
  num_pairs?: number;
  pairs?: { index: number; chosen: PackedUtt; rejected: PackedUtt }[];
};

/**
 * One multipacked block. TTS/LLM packs show token + utterance counts; a DPO pack
 * (kind=llm_dpo_packed) shows the PREFERENCE PAIR count and, on expand, each pair's
 * chosen vs rejected response. Opening the collapse decodes the block to text via
 * the pack tokenizer (fetched lazily, server-side).
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
  // DPO packs report a preference-pair count (chosen+rejected = one pair).
  const isDpo = row.objective === "dpo";
  const pairs = typeof row.pairs === "number" ? row.pairs : undefined;
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
          {isDpo ? (
            <>
              Packed DPO block · <span className="font-mono">{tokens ?? "?"}</span> tokens ·{" "}
              <span className="font-mono">{pairs ?? (utts != null ? utts / 2 : "?")}</span>{" "}
              preference pair{pairs === 1 ? "" : "s"}
            </>
          ) : (
            <>
              Packed block · <span className="font-mono">{tokens ?? "?"}</span> tokens ·{" "}
              <span className="font-mono">{utts ?? "?"}</span> utterance{utts === 1 ? "" : "s"}
            </>
          )}
        </span>
        <span className="shrink-0 text-xs text-muted-foreground">{open ? "hide" : "decode"}</span>
      </button>
      {open && (
        <div className="space-y-2 border-t border-border p-3 text-xs">
          {loading && (
            <span className="inline-flex items-center gap-1.5 text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> decoding with the tokenizer…
            </span>
          )}
          {err && <span className="text-destructive">{err}</span>}
          {data && data.objective === "dpo" && data.pairs ? (
            <>
              <div className="text-[11px] text-muted-foreground">
                {data.num_pairs ?? data.pairs.length} preference pair
                {(data.num_pairs ?? data.pairs.length) === 1 ? "" : "s"} multipacked into{" "}
                {data.num_tokens} tokens · decoded with <span className="font-mono">{data.tokenizer}</span>
              </div>
              <ol className="space-y-2">
                {data.pairs.map((p, j) => (
                  <li key={j} className="rounded border border-border/60 bg-muted/30 p-2">
                    <div className="mb-1 font-mono text-[10px] text-muted-foreground">pair {j + 1}</div>
                    <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                      <div className="rounded border border-emerald-600/40 bg-emerald-500/5 p-2">
                        <div className="mb-0.5 font-mono text-[10px] text-emerald-600 dark:text-emerald-400">
                          ✓ chosen · {p.chosen.tokens} tokens
                        </div>
                        <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed scrollbar-thin">{p.chosen.text}</pre>
                      </div>
                      <div className="rounded border border-destructive/40 bg-destructive/5 p-2">
                        <div className="mb-0.5 font-mono text-[10px] text-destructive">
                          ✕ rejected · {p.rejected.tokens} tokens
                        </div>
                        <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed scrollbar-thin">{p.rejected.text}</pre>
                      </div>
                    </div>
                  </li>
                ))}
              </ol>
            </>
          ) : data ? (
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
          ) : null}
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
  kind,
  speakerField,
  messagesField,
  rejectedField,
  decoder,
}: {
  datasetId: string;
  initial: DatasetPreview;
  kind?: string | null;
  speakerField?: string | null;
  messagesField?: string | null;
  rejectedField?: string | null;
  decoder?: DecoderState | null;
}) {
  // A rejected column set → DPO (preference) dataset: render chosen ↔ rejected pairs.
  // Else a messages column → chat-bubble view. chosen = the messages column.
  const isDpo = !!(rejectedField ?? "").trim();
  const chosenField = (messagesField ?? "").trim() || "chosen";
  const rejField = (rejectedField ?? "").trim() || "rejected";
  // Use chat-bubble view whenever a messages column is configured, regardless of kind.
  const isLlm = !isDpo && !!(messagesField ?? "").trim();
  const [limit, setLimit] = useState(initial.limit && initial.limit > 0 ? initial.limit : 20);
  const [offset, setOffset] = useState(initial.offset ?? 0);
  // Subsets (HF splits) are a MULTISELECT: pick several and the rows are merged
  // into one paged list (combined total) server-side. initial.split may be a
  // comma-joined list (shareable URL). Empty = the dataset's first split (default).
  const [selected, setSelected] = useState<string[]>(
    (initial.split ?? "").split(",").map((s) => s.trim()).filter(Boolean),
  );
  const [splits] = useState<string[]>(initial.splits ?? []);
  const splitKey = selected.join(",");
  const multiSubset = selected.length > 1;
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
      if (splitKey) q.set("split", splitKey);
      else q.delete("split");
      if (speaker) q.set("speaker", speaker);
      else q.delete("speaker");
      window.history.replaceState(null, "", `${window.location.pathname}?${q.toString()}`);
    }
    void fetchPage(offset, limit, splitKey, speaker);
  }, [offset, limit, splitKey, speaker, fetchPage]);

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
          {!isLlm && (
            excludedCount > 0 ? (
              <span className="text-xs text-muted-foreground">
                {excludedCount.toLocaleString()} excluded from training ·{" "}
                <button type="button" onClick={includeAll} className="underline underline-offset-2 hover:text-foreground">
                  include all
                </button>
              </span>
            ) : (
              <span className="text-xs text-muted-foreground">Untick a row to exclude it from training.</span>
            )
          )}
        </div>
        <div className="flex items-center gap-2">
          {splits.length > 1 && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" className="h-8 text-xs font-normal">
                  {selected.length === 0
                    ? `subset: ${splits[0]}`
                    : selected.length === 1
                      ? `subset: ${selected[0]}`
                      : `${selected.length} subsets`}
                  <ChevronDown className="ml-1 h-3.5 w-3.5 opacity-60" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="max-h-72 overflow-y-auto">
                {splits.map((s) => (
                  <DropdownMenuCheckboxItem
                    key={s}
                    checked={selected.includes(s)}
                    // keep the menu open so several subsets can be ticked in one go
                    onSelect={(e) => e.preventDefault()}
                    onCheckedChange={(c) => {
                      setOffset(0);
                      setSpeaker(null); // speaker lists are per-split; reset on change
                      setSelected((prev) => {
                        const next = c ? [...prev, s] : prev.filter((x) => x !== s);
                        // keep selection in the dataset's split order (stable merge order)
                        return splits.filter((x) => next.includes(x));
                      });
                    }}
                    className="text-xs"
                  >
                    {s}
                  </DropdownMenuCheckboxItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
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
            {rows.map((r, i) => {
              // In merged multi-subset view, tag each row with which subset it came from.
              const rowSplit = typeof r.__split === "string" ? r.__split : null;
              const item = r.packed === true ? (
                <PackedRowItem datasetId={datasetId} index={offset + i} row={r} split={rowSplit ?? selected[0] ?? null} decoder={decoder} />
              ) : isDpo ? (
                <DpoRowItem index={offset + i} row={r} chosenField={chosenField} rejectedField={rejField} />
              ) : isLlm ? (
                <LlmRowItem index={offset + i} row={r} messagesField={messagesField ?? "messages"} />
              ) : (
                <RowItem index={offset + i} row={r} onToggle={setIncluded} speakerField={speakerField} />
              );
              return (
                <div key={offset + i} className="space-y-1">
                  {multiSubset && rowSplit && (
                    <span className="inline-block rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                      subset: {rowSplit}
                    </span>
                  )}
                  {item}
                </div>
              );
            })}
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
