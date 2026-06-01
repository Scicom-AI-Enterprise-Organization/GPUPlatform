"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { Check, Loader2, Pencil } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type SplitInfo = { split: string; columns: string[]; num_rows?: number | null };

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

export function ColumnsCard({
  datasetId,
  kind,
  audioField,
  transcriptionField,
  splitFields,
}: {
  datasetId: string;
  kind: string;
  audioField: string;
  transcriptionField: string;
  splitFields?: Record<string, string> | null;
}) {
  const router = useRouter();
  const [editing, setEditing] = useState(false);
  const [audio, setAudio] = useState(audioField);
  const [transcription, setTranscription] = useState(transcriptionField);
  // Per-split transcription column choices (only when the HF source exposes splits).
  const [perSplit, setPerSplit] = useState<Record<string, string>>({});
  const [splits, setSplits] = useState<SplitInfo[] | null>(null);
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
    if (kind !== "hf") {
      setSplits([]);
      return;
    }
    setLoadingSplits(true);
    try {
      const r = await fetch(`/api/proxy/v1/datasets/${encodeURIComponent(datasetId)}/splits`, { cache: "no-store" });
      const data = (await r.json()) as { splits?: SplitInfo[] };
      const info = data.splits ?? [];
      setSplits(info);
      seedPerSplit(info);
    } catch {
      setSplits([]);
    } finally {
      setLoadingSplits(false);
    }
  }, [datasetId, kind, seedPerSplit]);

  function startEdit() {
    setAudio(audioField);
    setTranscription(transcriptionField);
    setErr(null);
    setEditing(true);
    // Lazily fetch the split columns for the per-split pickers (HF only); the
    // read-only display already renders the saved map from `splitFields`.
    if (splits) seedPerSplit(splits);
    else void loadSplits();
  }

  const multiSplit = (splits?.length ?? 0) > 1;

  async function save() {
    setErr(null);
    if (!audio.trim()) {
      setErr("Audio column is required.");
      return;
    }
    let body: Record<string, unknown>;
    if (multiSplit) {
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
        {!editing && (
          <Button variant="outline" size="xs" onClick={startEdit}>
            <Pencil className="h-3 w-3" /> Edit
          </Button>
        )}
      </CardHeader>
      <CardContent>
        {!editing ? (
          <div className="divide-y divide-border/60">
            <div className="flex items-baseline justify-between gap-4 py-1.5">
              <span className="text-xs text-muted-foreground">Audio column</span>
              <span className="font-mono text-xs">{audioField}</span>
            </div>
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
          </div>
        ) : (
          <div className="space-y-3">
            <div className="space-y-1 sm:max-w-xs">
              <Label htmlFor="ds-audio" className="text-xs">Audio column</Label>
              <Input
                id="ds-audio"
                value={audio}
                onChange={(e) => setAudio(e.target.value)}
                placeholder="audio"
                disabled={saving}
                className="font-mono text-xs"
              />
            </div>

            {loadingSplits ? (
              <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" /> reading splits…
              </p>
            ) : multiSplit ? (
              <div className="space-y-2">
                <Label className="text-xs">Transcription column per split</Label>
                <div className="space-y-2">
                  {(splits ?? []).map((s) => (
                    <div key={s.split} className="flex items-center gap-3">
                      <span className="w-24 shrink-0 font-mono text-xs text-muted-foreground">
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
                        <SelectTrigger className="text-xs">
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
                <Input
                  id="ds-transcription"
                  value={transcription}
                  onChange={(e) => setTranscription(e.target.value)}
                  placeholder="transcription"
                  disabled={saving}
                  className="font-mono text-xs"
                />
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
