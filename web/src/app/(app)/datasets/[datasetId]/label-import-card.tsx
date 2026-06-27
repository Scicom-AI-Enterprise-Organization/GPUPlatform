"use client";

import { useState } from "react";
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

const STATUS_LABELS: Record<string, string> = {
  approved: "Approved only (review-passed)",
  all: "All tasks",
  not_reviewed: "Not reviewed",
  rejected: "Rejected",
};

/** A stored UTC ISO instant → the "YYYY-MM-DDTHH:mm" local-wall-clock value a
 *  <input type="datetime-local"> expects (the inverse of `new Date(value).toISOString()`). */
function isoToLocalInput(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** Edit a kind=label dataset's import filter — which review status to pull and an
 *  optional point-in-time cutoff (only tasks last updated at/before it). Saving
 *  re-counts the dataset's rows on the gateway. */
export function LabelImportCard({
  datasetId,
  labelStatus,
  labelUpdatedUntil,
}: {
  datasetId: string;
  labelStatus?: string | null;
  labelUpdatedUntil?: string | null;
}) {
  const router = useRouter();
  const curStatus = labelStatus || "approved";
  const [editing, setEditing] = useState(false);
  const [status, setStatus] = useState(curStatus);
  const [until, setUntil] = useState(isoToLocalInput(labelUpdatedUntil));
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  function startEdit() {
    setStatus(curStatus);
    setUntil(isoToLocalInput(labelUpdatedUntil));
    setErr(null);
    setEditing(true);
  }

  async function save() {
    setErr(null);
    // Always send both knobs the card owns. "" cutoff clears it on the gateway;
    // a value is converted from local wall-clock to a UTC ISO instant.
    const body = {
      label_status: status,
      label_updated_until: until ? new Date(until).toISOString() : "",
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
      router.refresh(); // re-fetch the dataset with the new filter + row count
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
          <CardTitle className="text-base">Import filter</CardTitle>
          <span className="text-xs text-muted-foreground">
            Which tasks this dataset pulls from the labeling project — by review status and an optional
            point-in-time cutoff. Changing either re-counts the rows and changes what an export materialises.
          </span>
        </div>
        {!editing && (
          <Button variant="outline" size="xs" onClick={startEdit} className="shrink-0">
            <Pencil className="h-3 w-3" /> Edit
          </Button>
        )}
      </CardHeader>
      <CardContent>
        {!editing ? (
          <div className="divide-y divide-border/60">
            <div className="flex items-baseline justify-between gap-4 py-1.5">
              <span className="text-xs text-muted-foreground">Review status</span>
              <span className="text-xs">{STATUS_LABELS[curStatus] ?? curStatus}</span>
            </div>
            <div className="flex items-baseline justify-between gap-4 py-1.5">
              <span className="text-xs text-muted-foreground">Up to (cutoff)</span>
              <span className="font-mono text-xs">
                {labelUpdatedUntil ? (
                  new Date(labelUpdatedUntil).toLocaleString()
                ) : (
                  <span className="text-muted-foreground/50">no cutoff (all tasks)</span>
                )}
              </span>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="space-y-1 sm:max-w-xs">
              <Label className="text-xs">Import which tasks</Label>
              <Select value={status} onValueChange={setStatus} disabled={saving}>
                <SelectTrigger className="text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="approved">{STATUS_LABELS.approved}</SelectItem>
                  <SelectItem value="all">{STATUS_LABELS.all}</SelectItem>
                  <SelectItem value="not_reviewed">{STATUS_LABELS.not_reviewed}</SelectItem>
                  <SelectItem value="rejected">{STATUS_LABELS.rejected}</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1 sm:max-w-xs">
              <Label htmlFor="ds-label-cutoff" className="text-xs">
                Up to (timestamp cutoff) <span className="text-muted-foreground">(optional)</span>
              </Label>
              <Input
                id="ds-label-cutoff"
                type="datetime-local"
                value={until}
                onChange={(e) => setUntil(e.target.value)}
                disabled={saving}
                className="text-xs"
              />
              <p className="text-xs text-muted-foreground">
                Only tasks last updated at or before this moment. Read in your local timezone
                {until ? (
                  <> (= <span className="font-mono">{new Date(until).toISOString()}</span> UTC)</>
                ) : null}
                . Clear to import every task.
              </p>
            </div>
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
