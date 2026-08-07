"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Check, ChevronDown, Loader2, Pencil } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

type SplitInfo = {
  split: string;
  columns: string[];
  num_rows?: number | null;
  // Only set by `?all=1`: whether the subset is inside the dataset's stored scope.
  in_scope?: boolean | null;
};

/** Pull a readable message out of the gateway's {detail} / {detail:{error}} shape. */
function errText(body: unknown, fallback: string): string {
  if (typeof body === "string") return body || fallback;
  if (body && typeof body === "object") {
    const d = (body as Record<string, unknown>).detail;
    if (typeof d === "string") return d;
    if (d && typeof d === "object" && typeof (d as Record<string, unknown>).error === "string") {
      return (d as Record<string, string>).error;
    }
  }
  return fallback;
}

// Radix <Select> forbids an empty-string item value, so represent "no speaker
// column" with a sentinel that maps back to "" on save.
const NO_SPEAKER = "__none__";

const sameScope = (a: string[], b: string[]) =>
  a.length === b.length && a.every((x, i) => x === b[i]);

export function ColumnsCard({
  datasetId,
  kind,
  audioField,
  transcriptionField,
  speakerField,
  splitFields,
  messagesField,
  rejectedField,
  promptField,
  hfSubsets,
}: {
  datasetId: string;
  kind: string;
  audioField: string;
  transcriptionField: string;
  speakerField?: string | null;
  splitFields?: Record<string, string> | null;
  messagesField?: string | null;
  rejectedField?: string | null;
  promptField?: string | null;
  hfSubsets?: string[] | null;
}) {
  const router = useRouter();
  const isHfLike = kind === "hf" || kind === "llm"; // column list comes from HF splits API
  const isLlm = kind === "llm";
  const isHf = kind === "hf";
  // Any dataset with messages_field set is treated as a chat dataset in the viewer.
  const hasMessages = !!(messagesField ?? "").trim();
  // A pure-chat dataset has ONLY a messages column (no audio): the legacy kind=llm,
  // or any non-hf dataset with a messages column mapped (e.g. an uploaded chat file
  // — kind=upload + messages_field). A kind=hf dataset can carry BOTH audio (TTS)
  // and a messages column, so it keeps the mode toggle below instead.
  const isChatOnly = isLlm || (!isHf && hasMessages);
  // A kind=hf dataset can carry BOTH audio (TTS) and a messages column (LLM/chat) —
  // e.g. a multimodal function-call set. A mode toggle declutters the mapping:
  //   LLM → messages + audio only;  TTS → everything except messages.
  // Pure-chat kinds are always LLM; audio-only kinds (s3/upload) are always TTS.
  // Saves are scoped to the visible fields so switching modes never clobbers the other side.
  const [mode, setMode] = useState<"llm" | "tts">(hasMessages ? "llm" : "tts");
  const effMode: "llm" | "tts" = isChatOnly ? "llm" : isHf ? mode : "tts";
  const showMessages = effMode === "llm";
  const showAudio = !isChatOnly;                     // every kind but pure-chat has audio
  const showTts = effMode === "tts" && !isChatOnly;  // transcription (per-split) + speaker
  const [editing, setEditing] = useState(false);
  const [audio, setAudio] = useState(audioField);
  const [transcription, setTranscription] = useState(transcriptionField);
  // kind=llm primary column (= the chosen column in DPO mode)
  const [messages, setMessages] = useState(messagesField ?? "messages");
  // kind=llm DPO (preference) mode: whether to map a rejected column, and its name.
  // A rejected column on the dataset means it's already a preference dataset.
  const [dpoMode, setDpoMode] = useState(!!(rejectedField ?? "").trim());
  const [rejected, setRejected] = useState(rejectedField ?? "");
  // DPO continuation shape: the shared-prompt column. "" → chosen/rejected are full
  // conversations and the prompt is their common prefix.
  const [prompt, setPrompt] = useState(promptField ?? "");
  // TTS-only speaker column (one global column, like audio). "" → one voice.
  const [speaker, setSpeaker] = useState(speakerField ?? "");
  // kind=hf source scope — which of the repo's configs/splits this dataset covers.
  // Editable here (not only at create) because it's the setting you most want to
  // fix after seeing the rows: it drives the row browser, the split pickers below
  // AND which files a transform downloads. Held as canonical labels, ticked off the
  // repo's real subset list rather than typed — a typo here empties the dataset.
  const [subsets, setSubsets] = useState<string[]>([]);
  // What the scope resolved to when the splits loaded. The stored scope may be a
  // bare config name ("synthetic") that expands to several labels, so "did the user
  // change it" compares against this, not against the raw stored value — otherwise
  // merely opening the editor would rewrite the scope.
  const [scopeBaseline, setScopeBaseline] = useState<string[]>([]);
  // Per-split transcription column choices (only when the HF source exposes splits).
  const [perSplit, setPerSplit] = useState<Record<string, string>>({});
  const [splits, setSplits] = useState<SplitInfo[] | null>(null);
  // Every subset the repo exposes (from ?all=1) — the pool the picker offers, so
  // the scope can be widened again and not just narrowed.
  const [allSubsets, setAllSubsets] = useState<SplitInfo[]>([]);
  const [loadingSplits, setLoadingSplits] = useState(false);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Only string-valued entries are real per-split transcription columns. A
  // packed (tts_packed) dataset stashes a nested `_tts_pack` metadata object in
  // split_fields — exclude it so we never try to render an object as a child.
  const stringSplitFields: Record<string, string> = Object.fromEntries(
    Object.entries(splitFields ?? {}).filter(([, v]) => typeof v === "string"),
  ) as Record<string, string>;
  const hasSplitOverrides = Object.keys(stringSplitFields).length > 0;

  // Seed per-split picks from current overrides, falling back to the global
  // transcription column when a split has it, else its first non-audio column.
  const seedPerSplit = useCallback(
    (info: SplitInfo[]) => {
      const next: Record<string, string> = {};
      for (const s of info) {
        const cols = s.columns;
        const pick =
          splitFields?.[s.split] ||
          (cols.includes(transcriptionField) ? transcriptionField : "") ||
          cols.find((c) => c !== audioField) ||
          cols[0] ||
          "";
        next[s.split] = pick;
      }
      setPerSplit(next);
    },
    [splitFields, transcriptionField, audioField],
  );

  const loadSplits = useCallback(async () => {
    if (!isHfLike) {
      setSplits([]);
      return;
    }
    setLoadingSplits(true);
    try {
      // ?all=1 in ONE call: everything the repo has, each flagged in_scope — the
      // pool for the scope picker, and (filtered) the splits this dataset reads.
      const r = await fetch(
        `/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}/splits?all=1`,
        { cache: "no-store" },
      );
      const data = (await r.json()) as { splits?: SplitInfo[] };
      const every = data.splits ?? [];
      const inScope = every.filter((s) => s.in_scope !== false);
      setAllSubsets(every);
      // A scope resolving to nothing leaves the mapping section on the full list
      // rather than on an empty one; the picker is right there to fix it.
      const info = inScope.length ? inScope : every;
      setSplits(info);
      const scoped = inScope.map((s) => s.split);
      setSubsets(scoped);
      setScopeBaseline(scoped);
      seedPerSplit(info);
    } catch {
      setSplits([]);
    } finally {
      setLoadingSplits(false);
    }
  }, [datasetId, isHfLike, seedPerSplit]);

  // Eagerly load column names for hf/llm so dropdowns are ready when editing.
  useEffect(() => {
    if (isHfLike) void loadSplits();
  }, [isHfLike, loadSplits]);

  function startEdit() {
    setAudio(audioField);
    setTranscription(transcriptionField);
    setMessages(messagesField ?? "messages");
    setDpoMode(!!(rejectedField ?? "").trim());
    setRejected(rejectedField ?? "");
    setPrompt(promptField ?? "");
    setSpeaker(speakerField ?? "");
    setSubsets(scopeBaseline);
    setErr(null);
    setEditing(true);
    if (splits) seedPerSplit(splits);
    else void loadSplits();
  }

  // Live view of the same check `save()` makes, so the warning about the per-split
  // mapping shows while the picker is open.
  const scopeChangedNow = isHf && !sameScope(subsets, scopeBaseline);
  const multiSplit = showTts && (splits?.length ?? 0) > 1;
  // Union of all known columns from the HF splits API, used for all dropdowns.
  const allColumns = Array.from(new Set((splits ?? []).flatMap((s) => s.columns)));
  // Options for the speaker dropdown: all columns except the current audio pick.
  const columnOptions = allColumns.filter((c) => c !== audio);

  async function save() {
    setErr(null);
    if (isChatOnly) {
      if (!messages.trim()) {
        setErr(dpoMode ? "Chosen column is required." : "Messages column is required.");
        return;
      }
      if (dpoMode && !rejected.trim()) {
        setErr("Rejected column is required in DPO mode.");
        return;
      }
      if (dpoMode && rejected.trim() === messages.trim()) {
        setErr("Chosen and rejected must be different columns.");
        return;
      }
      if (dpoMode && prompt.trim() && (prompt.trim() === messages.trim() || prompt.trim() === rejected.trim())) {
        setErr("Prompt must be a different column from chosen and rejected.");
        return;
      }
      // rejected_field: a name → DPO (preference) mode; "" → chat mode.
      // prompt_field: a name → continuation shape; "" → infer from the common prefix.
      const body = {
        messages_field: messages.trim(),
        rejected_field: dpoMode ? rejected.trim() : "",
        prompt_field: dpoMode ? prompt.trim() : "",
      };
      setSaving(true);
      try {
        const r = await fetch(`/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const text = await r.text();
        let parsed: unknown = text;
        try { parsed = text ? JSON.parse(text) : null; } catch { /* keep raw */ }
        if (!r.ok) { setErr(errText(parsed, r.statusText)); return; }
        setEditing(false);
        router.refresh();
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      } finally {
        setSaving(false);
      }
      return;
    }
    if (!audio.trim()) {
      setErr("Audio column is required.");
      return;
    }
    // Scope the PATCH to the visible modality so switching LLM↔TTS never clobbers
    // the other side's mapping (a kind=hf dataset can carry both).
    let body: Record<string, unknown>;
    if (scopeChangedNow) {
      // Changing the scope changes which splits EXIST, so the per-split picks
      // below were derived from a list that's about to be replaced — saving them
      // would write a mapping keyed by labels this dataset no longer has. Save the
      // scope, clear the overrides, and let the reloaded splits be re-mapped.
      // Ticking every subset means "the whole repo" → store no scope at all, so a
      // config added upstream later is picked up instead of being silently excluded.
      const everything = subsets.length === allSubsets.length;
      body = {
        hf_subsets: everything ? [] : subsets,
        split_fields: {},
        audio_field: audio.trim(),
      };
    } else if (showMessages && !showTts) {
      // hf · LLM mode: messages + audio only (leave the TTS mapping untouched).
      if (!messages.trim()) {
        setErr("Messages column is required.");
        return;
      }
      body = { messages_field: messages.trim(), audio_field: audio.trim() };
    } else if (multiSplit) {
      const missing = (splits ?? []).filter((s) => !perSplit[s.split]?.trim());
      if (missing.length) {
        setErr(`Pick a transcription column for: ${missing.map((s) => s.split).join(", ")}`);
        return;
      }
      // Output/default column name = the first split's pick (e.g. train→text).
      const primary = perSplit[(splits ?? [])[0].split];
      body = {
        audio_field: audio.trim(),
        transcription_field: primary,
        speaker_field: speaker.trim(),
        split_fields: perSplit,
      };
    } else {
      if (!transcription.trim()) {
        setErr("Transcription column is required.");
        return;
      }
      body = {
        audio_field: audio.trim(),
        transcription_field: transcription.trim(),
        speaker_field: speaker.trim(),
        split_fields: {}, // clear any stale per-split overrides
      };
    }
    setSaving(true);
    try {
      const r = await fetch(`/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const text = await r.text();
      let parsed: unknown = text;
      try {
        parsed = text ? JSON.parse(text) : null;
      } catch {
        /* keep raw */
      }
      if (!r.ok) {
        setErr(errText(parsed, r.statusText));
        return;
      }
      setEditing(false);
      router.refresh(); // re-fetch the dataset with the new columns
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
        <div className="flex flex-col gap-0.5">
          <CardTitle className="text-base">Column mapping</CardTitle>
          <span className="text-xs text-muted-foreground">
            Which columns hold the audio and the transcription. Splits with different schemas (e.g.{" "}
            <span className="font-mono">train</span> uses <span className="font-mono">text</span>,{" "}
            <span className="font-mono">test</span> uses <span className="font-mono">after</span>) can map their
            transcription independently.
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {isHf && (
            <div className="flex items-center gap-0.5 rounded-md border border-border p-0.5" title="Which modality to map: LLM (messages + audio) or TTS (audio / transcription / speaker)">
              {(["llm", "tts"] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMode(m)}
                  className={cn(
                    "rounded px-2 py-1 text-xs font-medium uppercase tracking-wide transition-colors",
                    mode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {m}
                </button>
              ))}
            </div>
          )}
          {!editing && (
            <Button variant="outline" size="xs" onClick={startEdit}>
              <Pencil className="h-3 w-3" /> Edit
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {!editing ? (
          <div className="divide-y divide-border/60">
            {isHf && (
              <div className="flex items-baseline justify-between gap-4 py-1.5">
                <span className="text-xs text-muted-foreground">Subsets</span>
                <span className="text-right font-mono text-xs">
                  {(hfSubsets ?? []).length ? (
                    (hfSubsets ?? []).join(", ")
                  ) : (
                    <span className="text-muted-foreground/50">whole repo</span>
                  )}
                </span>
              </div>
            )}
            {showAudio && (
              <div className="flex items-baseline justify-between gap-4 py-1.5">
                <span className="text-xs text-muted-foreground">Audio column</span>
                <span className="font-mono text-xs">{audioField}</span>
              </div>
            )}
            {showTts && (
              <>
                {hasSplitOverrides ? (
                  Object.entries(stringSplitFields).map(([split, col]) => (
                    <div key={split} className="flex items-baseline justify-between gap-4 py-1.5">
                      <span className="text-xs text-muted-foreground">
                        Transcription · <span className="font-mono">{split}</span>
                      </span>
                      <span className="font-mono text-xs">{col}</span>
                    </div>
                  ))
                ) : (
                  <div className="flex items-baseline justify-between gap-4 py-1.5">
                    <span className="text-xs text-muted-foreground">Transcription column</span>
                    <span className="font-mono text-xs">{transcriptionField}</span>
                  </div>
                )}
                {speakerField && (
                  <div className="flex items-baseline justify-between gap-4 py-1.5">
                    <span className="text-xs text-muted-foreground">Speaker column</span>
                    <span className="font-mono text-xs">{speakerField}</span>
                  </div>
                )}
              </>
            )}
            {showMessages && isChatOnly && (
              <div className="flex items-baseline justify-between gap-4 py-1.5">
                <span className="text-xs text-muted-foreground">Mode</span>
                <span className="font-mono text-xs">
                  {(rejectedField ?? "").trim() ? "DPO — preference pairs" : "Chat — SFT"}
                </span>
              </div>
            )}
            {showMessages && (
              <div className="flex items-baseline justify-between gap-4 py-1.5">
                <span className="text-xs text-muted-foreground">
                  {(rejectedField ?? "").trim() ? "Chosen column" : "Messages column"}{" "}
                  <span className="text-[10px]">(chat / LLM)</span>
                </span>
                <span className="font-mono text-xs">{messagesField || <span className="text-muted-foreground/50">not set</span>}</span>
              </div>
            )}
            {showMessages && (rejectedField ?? "").trim() && (
              <div className="flex items-baseline justify-between gap-4 py-1.5">
                <span className="text-xs text-muted-foreground">
                  Rejected column <span className="text-[10px]">(DPO)</span>
                </span>
                <span className="font-mono text-xs">{rejectedField}</span>
              </div>
            )}
            {showMessages && (rejectedField ?? "").trim() && (promptField ?? "").trim() && (
              <div className="flex items-baseline justify-between gap-4 py-1.5">
                <span className="text-xs text-muted-foreground">
                  Prompt column <span className="text-[10px]">(DPO)</span>
                </span>
                <span className="font-mono text-xs">{promptField}</span>
              </div>
            )}
          </div>
        ) : isChatOnly ? (
          // Pure-chat dataset (kind=llm, or an uploaded chat file). Chat (SFT) maps a
          // single messages column; DPO (preference) maps chosen + rejected columns.
          <div className="space-y-3">
            <div className="space-y-1">
              <Label className="text-xs">Mode</Label>
              <div className="flex w-fit items-center gap-0.5 rounded-md border border-border p-0.5">
                {([["sft", "Chat (SFT)"], ["dpo", "Preference (DPO)"]] as const).map(([m, lbl]) => (
                  <button
                    key={m}
                    type="button"
                    disabled={saving}
                    onClick={() => setDpoMode(m === "dpo")}
                    className={cn(
                      "rounded px-2.5 py-1 text-xs font-medium transition-colors",
                      (m === "dpo") === dpoMode
                        ? "bg-primary text-primary-foreground"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {lbl}
                  </button>
                ))}
              </div>
              <p className="text-[11px] text-muted-foreground">
                {dpoMode
                  ? "Preference pairs: pick the chosen and rejected columns. Pack with objective=DPO to train a DPO run."
                  : "Supervised: one messages column, packed for a standard SFT run."}
              </p>
            </div>

            <div className="space-y-1 sm:max-w-xs">
              <Label htmlFor="ds-messages" className="text-xs">{dpoMode ? "Chosen column" : "Messages column"}</Label>
              {loadingSplits ? (
                <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin" /> reading columns…
                </p>
              ) : allColumns.length > 0 ? (
                <Select value={messages} onValueChange={setMessages} disabled={saving}>
                  <SelectTrigger className="font-mono text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {allColumns.map((c) => (
                      <SelectItem key={c} value={c} className="font-mono text-xs">{c}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : (
                <Input
                  id="ds-messages"
                  value={messages}
                  onChange={(e) => setMessages(e.target.value)}
                  placeholder={dpoMode ? "chosen" : "messages"}
                  disabled={saving}
                  className="font-mono text-xs"
                />
              )}
            </div>

            {dpoMode && (
              <div className="space-y-1 sm:max-w-xs">
                <Label htmlFor="ds-rejected" className="text-xs">Rejected column</Label>
                {!loadingSplits && allColumns.length > 0 ? (
                  <Select value={rejected || "__none__"} onValueChange={(v) => setRejected(v === "__none__" ? "" : v)} disabled={saving}>
                    <SelectTrigger className="font-mono text-xs"><SelectValue placeholder="Choose a column" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__none__" className="text-xs text-muted-foreground">— choose —</SelectItem>
                      {allColumns.map((c) => (
                        <SelectItem key={c} value={c} className="font-mono text-xs">{c}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    id="ds-rejected"
                    value={rejected}
                    onChange={(e) => setRejected(e.target.value)}
                    placeholder="rejected"
                    disabled={saving}
                    className="font-mono text-xs"
                  />
                )}
                <p className="text-[11px] text-muted-foreground">
                  The dispreferred response. Chosen &amp; rejected should share the prompt turns.
                </p>
              </div>
            )}

            {dpoMode && (
              <div className="space-y-1 sm:max-w-xs">
                <Label htmlFor="ds-prompt" className="text-xs">Prompt column (optional)</Label>
                {!loadingSplits && allColumns.length > 0 ? (
                  <Select value={prompt || "__none__"} onValueChange={(v) => setPrompt(v === "__none__" ? "" : v)} disabled={saving}>
                    <SelectTrigger className="font-mono text-xs"><SelectValue placeholder="Choose a column" /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__none__" className="text-xs text-muted-foreground">— none (shared prefix) —</SelectItem>
                      {allColumns.map((c) => (
                        <SelectItem key={c} value={c} className="font-mono text-xs">{c}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    id="ds-prompt"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="prompt"
                    disabled={saving}
                    className="font-mono text-xs"
                  />
                )}
                <p className="text-[11px] text-muted-foreground">
                  Only for the <span className="font-medium">continuation</span> shape: chosen &amp; rejected hold
                  just the turns AFTER the prompt (an agentic pair that diverges from turn 0). Leave empty when
                  they are full conversations — the prompt is then their common prefix.
                </p>
              </div>
            )}
            {err && <p className="text-sm text-destructive">{err}</p>}
            <div className="flex items-center gap-2">
              <Button size="sm" onClick={save} disabled={saving}>
                {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
                Save
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setEditing(false)} disabled={saving}>Cancel</Button>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {isHf && (
              <div className="space-y-1 sm:max-w-md">
                <Label className="text-xs">Subsets</Label>
                {loadingSplits ? (
                  <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <Loader2 className="h-3 w-3 animate-spin" /> reading subsets…
                  </p>
                ) : (
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={saving || allSubsets.length === 0}
                        className="h-8 w-56 justify-between text-xs font-normal"
                      >
                        <span className="truncate">
                          {subsets.length === 0 || subsets.length === allSubsets.length
                            ? "whole repo"
                            : subsets.length === 1
                              ? subsets[0]
                              : `${subsets.length} subsets`}
                        </span>
                        <ChevronDown className="ml-1 h-3.5 w-3.5 shrink-0 opacity-60" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start" className="max-h-72 overflow-y-auto">
                      {allSubsets.map((s) => (
                        <DropdownMenuCheckboxItem
                          key={s.split}
                          checked={subsets.includes(s.split)}
                          // keep the menu open so several can be ticked in one go
                          onSelect={(e) => e.preventDefault()}
                          onCheckedChange={(c) =>
                            setSubsets((prev) => {
                              const next = c ? [...prev, s.split] : prev.filter((x) => x !== s.split);
                              // keep the repo's own order, so the saved scope is stable
                              return allSubsets.map((x) => x.split).filter((x) => next.includes(x));
                            })
                          }
                          className="text-xs"
                        >
                          <span className="font-mono">{s.split}</span>
                          {typeof s.num_rows === "number" && (
                            <span className="ml-1.5 text-[10px] opacity-60">({s.num_rows.toLocaleString()})</span>
                          )}
                        </DropdownMenuCheckboxItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
                )}
                <p className="text-[11px] text-muted-foreground">
                  Which of the repo&apos;s configs/splits this dataset covers. Bounds the row browser, the
                  split pickers below, and which files a transform downloads. Tick everything for the whole
                  repo.
                  {scopeChangedNow && (
                    <span className="block pt-1 text-foreground">
                      Changing this replaces the split list, so saving clears the per-split transcription
                      mapping — re-pick it once the new subsets load.
                    </span>
                  )}
                </p>
              </div>
            )}
            <div className="space-y-1 sm:max-w-xs">
              <Label htmlFor="ds-audio" className="text-xs">Audio column</Label>
              {isHfLike && allColumns.length > 0 ? (
                <Select value={audio} onValueChange={setAudio} disabled={saving}>
                  <SelectTrigger className="font-mono text-xs"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {allColumns.map((c) => (
                      <SelectItem key={c} value={c} className="font-mono text-xs">{c}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : (
                <Input
                  id="ds-audio"
                  value={audio}
                  onChange={(e) => setAudio(e.target.value)}
                  placeholder="audio"
                  disabled={saving}
                  className="font-mono text-xs"
                />
              )}
            </div>

            {showTts && (loadingSplits ? (
              <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" /> reading splits…
              </p>
            ) : multiSplit ? (
              <div className="space-y-2">
                <Label className="text-xs">Transcription column per split</Label>
                <div className="space-y-2">
                  {(splits ?? []).map((s) => (
                    <div key={s.split} className="flex items-center gap-3">
                      {/* A subset label is `config/split` and can be long
                          (synthetic_podcast/train). The old fixed w-24 didn't clip,
                          so the text ran straight over the dropdown next to it —
                          hence the width bump + truncate + a title for the full name. */}
                      <span
                        className="w-44 shrink-0 truncate font-mono text-xs text-muted-foreground"
                        title={
                          typeof s.num_rows === "number"
                            ? `${s.split} — ${s.num_rows.toLocaleString()} rows`
                            : s.split
                        }
                      >
                        {s.split}
                        {typeof s.num_rows === "number" && (
                          <span className="ml-1 text-[10px] opacity-60">({s.num_rows})</span>
                        )}
                      </span>
                      <Select
                        value={perSplit[s.split] ?? ""}
                        onValueChange={(v) => setPerSplit((p) => ({ ...p, [s.split]: v }))}
                        disabled={saving}
                      >
                        <SelectTrigger className="w-56 shrink-0 text-xs">
                          <SelectValue placeholder="Choose a column" />
                        </SelectTrigger>
                        <SelectContent>
                          {s.columns
                            .filter((c) => c !== audio)
                            .map((c) => (
                              <SelectItem key={c} value={c} className="font-mono text-xs">
                                {c}
                              </SelectItem>
                            ))}
                        </SelectContent>
                      </Select>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground">
                  The output dataset uses one transcription column named{" "}
                  <span className="font-mono">{perSplit[(splits ?? [])[0]?.split] || transcriptionField}</span> (the{" "}
                  <span className="font-mono">{(splits ?? [])[0]?.split}</span> pick), filled per split.
                </p>
              </div>
            ) : (
              <div className="space-y-1 sm:max-w-xs">
                <Label htmlFor="ds-transcription" className="text-xs">Transcription column</Label>
                {isHfLike && allColumns.length > 0 ? (
                  <Select value={transcription} onValueChange={setTranscription} disabled={saving}>
                    <SelectTrigger className="font-mono text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {allColumns.map((c) => (
                        <SelectItem key={c} value={c} className="font-mono text-xs">{c}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    id="ds-transcription"
                    value={transcription}
                    onChange={(e) => setTranscription(e.target.value)}
                    placeholder="transcription"
                    disabled={saving}
                    className="font-mono text-xs"
                  />
                )}
              </div>
            ))}

            {showTts && !loadingSplits && (
              <div className="space-y-1 sm:max-w-xs">
                <Label htmlFor="ds-speaker" className="text-xs">
                  Speaker column <span className="text-muted-foreground">(optional · TTS)</span>
                </Label>
                {columnOptions.length > 0 ? (
                  <Select
                    value={speaker || NO_SPEAKER}
                    onValueChange={(v) => setSpeaker(v === NO_SPEAKER ? "" : v)}
                    disabled={saving}
                  >
                    <SelectTrigger className="text-xs">
                      <SelectValue placeholder="No speaker column" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={NO_SPEAKER} className="text-xs">
                        — none (one voice) —
                      </SelectItem>
                      {Array.from(new Set([...(speaker ? [speaker] : []), ...columnOptions])).map(
                        (c) => (
                          <SelectItem key={c} value={c} className="font-mono text-xs">
                            {c}
                          </SelectItem>
                        ),
                      )}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    id="ds-speaker"
                    value={speaker}
                    onChange={(e) => setSpeaker(e.target.value)}
                    placeholder="speaker"
                    disabled={saving}
                    className="font-mono text-xs"
                  />
                )}
                <p className="text-xs text-muted-foreground">
                  Prepended to each line when packing for TTS. Leave empty to train a single voice.
                </p>
              </div>
            )}

            {showMessages && (
              <div className="space-y-1 sm:max-w-xs">
                <Label htmlFor="ds-messages" className="text-xs">
                  Messages column <span className="text-muted-foreground">(optional · LLM / chat)</span>
                </Label>
                {allColumns.length > 0 ? (
                  <Select
                    value={messages || "__none__"}
                    onValueChange={(v) => setMessages(v === "__none__" ? "" : v)}
                    disabled={saving}
                  >
                    <SelectTrigger className="text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__none__" className="text-xs text-muted-foreground">— none —</SelectItem>
                      {allColumns.map((c) => (
                        <SelectItem key={c} value={c} className="font-mono text-xs">{c}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    id="ds-messages"
                    value={messages}
                    onChange={(e) => setMessages(e.target.value)}
                    placeholder="messages"
                    disabled={saving}
                    className="font-mono text-xs"
                  />
                )}
                <p className="text-xs text-muted-foreground">
                  Set this to switch the row viewer to chat-bubble mode for LLM datasets.
                </p>
              </div>
            )}

            {err && <p className="text-sm text-destructive">{err}</p>}
            <div className="flex items-center gap-2">
              <Button size="sm" onClick={save} disabled={saving}>
                {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
                Save
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setEditing(false)} disabled={saving}>
                Cancel
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
