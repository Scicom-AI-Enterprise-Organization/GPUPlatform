"use client";

import { useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Activity, Inbox, Loader2, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { gateway } from "@/lib/gateway";
import type { TrackingCredentialRecord } from "@/lib/types";

type Form = {
  name: string;
  kind: "wandb" | "mlflow";
  api_key: string;
  uri: string;
  username: string;
  password: string;
};
const EMPTY: Form = { name: "", kind: "wandb", api_key: "", uri: "", username: "", password: "" };
// URL flag so an open "Add tracking credential" dialog shows in the address bar
// (shareable / survives reload). Shares the ?add key with the global-env manager
// above; each only reads/clears its own value.
const ADD_PARAM = "tracking-credential";

export function TrackingCredentialsManager({ initial }: { initial: TrackingCredentialRecord[] }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [rows, setRows] = useState<TrackingCredentialRecord[]>(initial);
  const [open, setOpen] = useState(searchParams.get("add") === ADD_PARAM);
  const [form, setForm] = useState<Form>(EMPTY);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  async function reload() {
    try {
      setRows(await gateway.listTrackingCredentials());
    } catch {
      /* keep optimistic view */
    }
  }

  // Add/remove ?add=tracking-credential without disturbing other query params.
  function syncAddParam(on: boolean) {
    const params = new URLSearchParams(searchParams.toString());
    if (on) params.set("add", ADD_PARAM);
    else if (params.get("add") === ADD_PARAM) params.delete("add");
    const qs = params.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  }
  function openAdd() {
    setForm(EMPTY);
    setErr(null);
    setOpen(true);
    syncAddParam(true);
  }
  function closeDialog() {
    setOpen(false);
    setErr(null);
    syncAddParam(false);
  }

  async function onSave() {
    setErr(null);
    if (!form.name.trim()) return setErr("Name is required.");
    if (form.kind === "wandb" && !form.api_key.trim()) return setErr("API key is required for W&B.");
    if (form.kind === "mlflow" && !form.uri.trim()) return setErr("Tracking URI is required for MLflow.");
    setSaving(true);
    try {
      await gateway.createTrackingCredential({
        name: form.name.trim(),
        kind: form.kind,
        ...(form.kind === "wandb"
          ? { api_key: form.api_key.trim() }
          : { uri: form.uri.trim(), username: form.username.trim(), password: form.password.trim() }),
      });
      closeDialog();
      setForm(EMPTY);
      await reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  async function onDelete(id: string) {
    if (!confirm("Delete this tracking credential? Runs referencing it lose tracking.")) return;
    setDeleting(id);
    try {
      await gateway.deleteTrackingCredential(id);
      setRows((p) => p.filter((r) => r.id !== id));
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleting(null);
    }
  }

  return (
    <section className="mt-10">
      <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-muted-foreground" />
          <h2 className="text-base font-medium">Tracking credentials</h2>
          <Badge variant="secondary" className="text-[10px]">{rows.length}</Badge>
        </div>
        <Button size="sm" onClick={openAdd}>
          <Plus className="h-4 w-4" /> Add credential
        </Button>
      </div>
      <p className="mb-3 text-xs text-muted-foreground">
        Named Weights &amp; Biases / MLflow credentials. Autotrain runs pick one per run; the
        runner injects <span className="font-mono">WANDB_API_KEY</span> or{" "}
        <span className="font-mono">MLFLOW_TRACKING_URI/USERNAME/PASSWORD</span>. Encrypted at rest, never shown again.
      </p>

      {rows.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center">
          <Inbox className="h-6 w-6 text-muted-foreground/60" />
          <p className="text-sm text-muted-foreground">No tracking credentials yet.</p>
        </div>
      ) : (
        <ul className="divide-y divide-border rounded-md border border-border">
          {rows.map((r) => (
            <li key={r.id} className="flex items-center justify-between gap-4 px-4 py-2.5 text-sm">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="truncate font-medium">{r.name}</span>
                  <Badge variant="outline" className="text-[10px] uppercase">{r.kind}</Badge>
                </div>
                <div className="truncate font-mono text-xs text-muted-foreground">{r.preview}</div>
              </div>
              <Button variant="outline" size="icon" className="text-destructive shrink-0"
                onClick={() => onDelete(r.id)} disabled={deleting === r.id} title="Delete">
                {deleting === r.id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
              </Button>
            </li>
          ))}
        </ul>
      )}

      <Dialog open={open} onOpenChange={(o) => (o ? setOpen(true) : closeDialog())}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add tracking credential</DialogTitle>
            <DialogDescription>
              Stored encrypted; the secret is never shown again. Referenced by name on the Autotrain form.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label className="text-xs">Tracker</Label>
                <Select value={form.kind} onValueChange={(v) => setForm((f) => ({ ...f, kind: v as "wandb" | "mlflow" }))}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="wandb">Weights &amp; Biases</SelectItem>
                    <SelectItem value="mlflow">MLflow</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs">Name</Label>
                <Input placeholder="e.g. personal / scicom-team" value={form.name}
                  onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))} />
              </div>
            </div>
            {form.kind === "wandb" ? (
              <div className="space-y-1.5">
                <Label className="text-xs">API key</Label>
                <Input type="password" className="font-mono" placeholder="wandb_…" value={form.api_key}
                  onChange={(e) => setForm((f) => ({ ...f, api_key: e.target.value }))} />
              </div>
            ) : (
              <>
                <div className="space-y-1.5">
                  <Label className="text-xs">Tracking URI</Label>
                  <Input className="font-mono" placeholder="https://mlflow.aies.scicom.dev" value={form.uri}
                    onChange={(e) => setForm((f) => ({ ...f, uri: e.target.value }))} />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <Label className="text-xs">Username</Label>
                    <Input className="font-mono" placeholder="you@org" value={form.username}
                      onChange={(e) => setForm((f) => ({ ...f, username: e.target.value }))} />
                  </div>
                  <div className="space-y-1.5">
                    <Label className="text-xs">Password</Label>
                    <Input type="password" className="font-mono" value={form.password}
                      onChange={(e) => setForm((f) => ({ ...f, password: e.target.value }))} />
                  </div>
                </div>
              </>
            )}
            {err && <p className="text-sm text-destructive">{err}</p>}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={closeDialog}>Cancel</Button>
            <Button onClick={onSave} disabled={saving}>
              {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : null} Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  );
}
