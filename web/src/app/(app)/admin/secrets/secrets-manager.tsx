"use client";

import { useState } from "react";
import { Inbox, KeyRound, Loader2, Lock, Pencil, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import type { GlobalEnvRecord } from "@/lib/types";

const BASE = "/api/proxy/v1/global-env";
const KEY_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

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

type Form = { key: string; value: string; is_secret: boolean; description: string };
const EMPTY: Form = { key: "", value: "", is_secret: true, description: "" };

export function SecretsManager({ initial }: { initial: GlobalEnvRecord[] }) {
  const [rows, setRows] = useState<GlobalEnvRecord[]>(initial);
  const [open, setOpen] = useState(false);
  const [editingKey, setEditingKey] = useState<string | null>(null); // null = adding
  const [form, setForm] = useState<Form>(EMPTY);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  async function reload() {
    try {
      const r = await fetch(BASE, { cache: "no-store" });
      if (r.ok) setRows((await r.json()) as GlobalEnvRecord[]);
    } catch {
      /* keep the optimistic view; next load reconciles */
    }
  }

  function openAdd() {
    setEditingKey(null);
    setForm(EMPTY);
    setErr(null);
    setOpen(true);
  }
  function openEdit(row: GlobalEnvRecord) {
    setEditingKey(row.key);
    setForm({ key: row.key, value: row.value ?? "", is_secret: row.is_secret, description: row.description ?? "" });
    setErr(null);
    setOpen(true);
  }

  async function save() {
    const key = form.key.trim();
    setErr(null);
    if (!KEY_RE.test(key)) {
      setErr("Key must be a valid env var name (letters, digits, underscore; not starting with a digit).");
      return;
    }
    if (!form.value) {
      setErr(editingKey && form.is_secret ? "Re-enter the value to save a secret (it can't be read back)." : "Value is required.");
      return;
    }
    setSaving(true);
    try {
      const r = await fetch(`${BASE}/${encodeURIComponent(key)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          value: form.value,
          is_secret: form.is_secret,
          description: form.description.trim() || null,
        }),
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
      setOpen(false);
      await reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  async function remove(key: string) {
    setDeleting(true);
    setRows((prev) => prev.filter((r) => r.key !== key)); // optimistic
    try {
      await fetch(`${BASE}/${encodeURIComponent(key)}`, { method: "DELETE" });
    } catch {
      void reload();
    } finally {
      setDeleting(false);
      setConfirmDelete(null);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {rows.length} variable{rows.length === 1 ? "" : "s"}
        </span>
        <Button size="sm" onClick={openAdd}>
          <Plus className="h-4 w-4" /> Add variable
        </Button>
      </div>

      {rows.length === 0 ? (
        <div className="flex flex-col items-center gap-2 rounded-md border border-dashed border-border bg-muted/20 px-4 py-12 text-center text-sm text-muted-foreground">
          <Inbox className="h-5 w-5" />
          No global variables yet. Add <span className="font-mono">HF_TOKEN</span> here and every benchmark + worker
          picks it up.
        </div>
      ) : (
        <div className="overflow-hidden rounded-md border border-border">
          <table className="w-full text-sm">
            <thead className="border-b border-border bg-muted/30 text-left text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-4 py-2 font-medium">Key</th>
                <th className="px-4 py-2 font-medium">Value</th>
                <th className="px-4 py-2 font-medium">Description</th>
                <th className="px-4 py-2 font-medium">Updated</th>
                <th className="px-4 py-2" />
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map((r) => (
                <tr key={r.key} className="hover:bg-muted/20">
                  <td className="px-4 py-2">
                    <span className="inline-flex items-center gap-1.5 font-mono text-xs text-foreground">
                      {r.is_secret ? (
                        <Lock className="h-3 w-3 text-muted-foreground" />
                      ) : (
                        <KeyRound className="h-3 w-3 text-muted-foreground" />
                      )}
                      {r.key}
                    </span>
                  </td>
                  <td className="px-4 py-2 font-mono text-xs text-muted-foreground">
                    {r.is_secret ? (r.value_preview ?? "••••••") : (r.value ?? "")}
                  </td>
                  <td className="px-4 py-2 text-xs text-muted-foreground">{r.description || "—"}</td>
                  <td className="px-4 py-2 text-xs text-muted-foreground" suppressHydrationWarning>
                    {r.updated_by} · {relTime(r.updated_at)}
                  </td>
                  <td className="px-4 py-2">
                    <div className="flex items-center justify-end gap-1">
                      <Button variant="ghost" size="icon-sm" onClick={() => openEdit(r)} aria-label="Edit">
                        <Pencil className="h-3.5 w-3.5" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        onClick={() => setConfirmDelete(r.key)}
                        aria-label="Delete"
                        className="text-muted-foreground hover:text-destructive"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* add / edit */}
      <Dialog open={open} onOpenChange={(o) => { setOpen(o); if (!o) setErr(null); }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{editingKey ? `Edit ${editingKey}` : "Add global variable"}</DialogTitle>
            <DialogDescription>
              Injected into benchmark runs and serverless workers. A per-resource variable of the same name overrides
              this global one.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <div className="space-y-1">
              <Label className="text-xs">Key</Label>
              <Input
                value={form.key}
                onChange={(e) => setForm((f) => ({ ...f, key: e.target.value }))}
                placeholder="HF_TOKEN"
                disabled={!!editingKey || saving}
                className="font-mono text-sm"
              />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">Value</Label>
              <Input
                value={form.value}
                onChange={(e) => setForm((f) => ({ ...f, value: e.target.value }))}
                placeholder={editingKey && form.is_secret ? "enter a new value (secrets can't be read back)" : "hf_…"}
                disabled={saving}
                type={form.is_secret ? "password" : "text"}
                className="font-mono text-sm"
              />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">Description (optional)</Label>
              <Input
                value={form.description}
                onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
                placeholder="what this is for"
                disabled={saving}
                className="text-sm"
              />
            </div>
            <label className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={form.is_secret}
                onCheckedChange={(c) => setForm((f) => ({ ...f, is_secret: c === true }))}
                disabled={saving}
              />
              <span>Secret — encrypt and never show the value again (uncheck for non-sensitive vars like a region)</span>
            </label>
          </div>
          <DialogFooter>
            {err && <p className="mr-auto text-sm text-destructive">{err}</p>}
            <Button variant="ghost" onClick={() => setOpen(false)} disabled={saving}>
              Cancel
            </Button>
            <Button onClick={save} disabled={saving}>
              {saving && <Loader2 className="h-4 w-4 animate-spin" />}
              {editingKey ? "Save" : "Add"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* delete confirm */}
      <Dialog open={confirmDelete !== null} onOpenChange={(o) => !deleting && setConfirmDelete(o ? confirmDelete : null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete {confirmDelete}?</DialogTitle>
            <DialogDescription>
              New benchmark runs and workers will no longer receive this variable. Already-running workers keep it
              until they restart. This can&apos;t be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setConfirmDelete(null)} disabled={deleting}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => confirmDelete && remove(confirmDelete)}
              disabled={deleting}
            >
              {deleting && <Loader2 className="h-4 w-4 animate-spin" />}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function relTime(iso: string): string {
  const ms = Date.parse(iso);
  if (!Number.isFinite(ms)) return "";
  const s = Math.max(0, Math.round((Date.now() - ms) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}
