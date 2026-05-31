"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Check, Loader2, Pencil, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

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

/** Dataset title with inline rename (PATCH name → refresh). */
export function DatasetTitle({ id, name }: { id: string; name: string }) {
  const router = useRouter();
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(name);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function save() {
    const n = value.trim();
    if (!n) {
      setErr("Name can't be blank.");
      return;
    }
    if (n === name) {
      setEditing(false);
      return;
    }
    setSaving(true);
    setErr(null);
    try {
      const r = await fetch(`/api/proxy/v1/datasets/${encodeURIComponent(id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: n }),
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
      router.refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  if (editing) {
    return (
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <Input
            value={value}
            onChange={(e) => setValue(e.target.value)}
            disabled={saving}
            autoFocus
            className="h-9 max-w-sm text-lg font-semibold"
            onKeyDown={(e) => {
              if (e.key === "Enter") void save();
              if (e.key === "Escape") {
                setEditing(false);
                setValue(name);
                setErr(null);
              }
            }}
          />
          <Button size="icon-sm" onClick={save} disabled={saving} aria-label="Save name">
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => {
              setEditing(false);
              setValue(name);
              setErr(null);
            }}
            disabled={saving}
            aria-label="Cancel rename"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        {err && <p className="text-sm text-destructive">{err}</p>}
      </div>
    );
  }

  return (
    <div className="group flex items-center gap-2">
      <h1 className="text-2xl font-semibold tracking-tight">{name}</h1>
      <Button
        variant="ghost"
        size="icon-sm"
        className="opacity-0 transition-opacity group-hover:opacity-100"
        onClick={() => {
          setValue(name);
          setErr(null);
          setEditing(true);
        }}
        title="Rename dataset"
        aria-label="Rename dataset"
      >
        <Pencil className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}
