"use client";

import { useState, useTransition } from "react";
import { Inbox, Loader2, Lock, Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";
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
import { gateway } from "@/lib/gateway";
import type { PolicyRole, SectionKey } from "@/lib/types";
import { cn } from "@/lib/utils";

const SECTIONS: SectionKey[] = ["inference", "benchmark", "compute", "datasets"];
const SECTION_LABEL: Record<SectionKey, string> = {
  inference: "Serverless Inference",
  benchmark: "Benchmark",
  compute: "Compute",
  datasets: "Datasets",
};

function slugify(s: string): string {
  return s
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);
}

export function RolesManager({ initial }: { initial: PolicyRole[] }) {
  const [roles, setRoles] = useState<PolicyRole[]>(initial);
  const [createOpen, setCreateOpen] = useState(false);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between border-b border-border pb-2">
        <span className="text-sm text-muted-foreground">
          {roles.length} role{roles.length === 1 ? "" : "s"}
        </span>
        <Button size="sm" onClick={() => setCreateOpen(true)}>
          <Plus className="h-4 w-4" />
          New role
        </Button>
      </div>

      <ul className="grid gap-3">
        {roles.map((r) => (
          <RoleCard
            key={r.id}
            role={r}
            onChange={(updated) =>
              setRoles((prev) => prev.map((p) => (p.id === updated.id ? updated : p)))
            }
            onDelete={(id) => setRoles((prev) => prev.filter((p) => p.id !== id))}
          />
        ))}
        {roles.length === 0 && (
          <li className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
            <Inbox className="h-6 w-6 text-muted-foreground/60" />
            <p className="text-sm text-muted-foreground">
              No roles yet. Click <span className="font-medium text-foreground">New role</span> to create one.
            </p>
          </li>
        )}
      </ul>

      <CreateRoleDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        existingIds={new Set(roles.map((r) => r.id))}
        onCreated={(r) => {
          setRoles((prev) => [...prev, r]);
        }}
      />
    </div>
  );
}

function RoleCard({
  role,
  onChange,
  onDelete,
}: {
  role: PolicyRole;
  onChange: (next: PolicyRole) => void;
  onDelete: (id: string) => void;
}) {
  const [pending, startTransition] = useTransition();
  const [deleteOpen, setDeleteOpen] = useState(false);

  const toggleSection = (s: SectionKey) => {
    const nextSections = { ...role.sections, [s]: !role.sections[s] };
    startTransition(async () => {
      try {
        const updated = await gateway.adminUpdatePolicyRole(role.id, {
          sections: nextSections,
        });
        onChange(updated);
        toast.success(`${role.name} updated`, { duration: 3000 });
      } catch (e) {
        toast.error(e instanceof Error ? e.message : String(e)), { duration: 5000 };
      }
    });
  };

  const renameRole = (next: string) => {
    const trimmed = next.trim();
    if (!trimmed || trimmed === role.name) return;
    startTransition(async () => {
      try {
        const updated = await gateway.adminUpdatePolicyRole(role.id, { name: trimmed });
        onChange(updated);
      } catch (e) {
        toast.error(e instanceof Error ? e.message : String(e)), { duration: 5000 };
      }
    });
  };

  const doDelete = () => {
    setDeleteOpen(false);
    startTransition(async () => {
      try {
        await gateway.adminDeletePolicyRole(role.id);
        onDelete(role.id);
        toast.success(`Deleted ${role.name}`, { duration: 3000 });
      } catch (e) {
        toast.error(e instanceof Error ? e.message : String(e)), { duration: 5000 };
      }
    });
  };

  return (
    <li className="rounded-lg border border-border bg-card p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <input
              defaultValue={role.name}
              disabled={pending}
              onBlur={(e) => renameRole(e.currentTarget.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") (e.target as HTMLInputElement).blur();
              }}
              className="bg-transparent text-base font-medium outline-none focus:ring-1 focus:ring-foreground/20 rounded px-1 -mx-1"
            />
            {role.is_system && (
              <span
                className="inline-flex items-center gap-1 rounded-full border border-border bg-muted px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground"
                title="Built-in role — can't be deleted"
              >
                <Lock className="h-3 w-3" />
                system
              </span>
            )}
          </div>
          <p className="mt-0.5 font-mono text-xs text-muted-foreground">{role.id}</p>
        </div>
        <div className="flex items-center gap-2">
          {pending && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
          {!role.is_system && (
            <Button
              variant="ghost"
              size="icon"
              disabled={pending}
              onClick={() => setDeleteOpen(true)}
              className="text-muted-foreground hover:text-destructive"
              title="Delete role"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {SECTIONS.map((s) => {
          const on = role.sections[s];
          return (
            <button
              key={s}
              type="button"
              onClick={() => toggleSection(s)}
              disabled={pending}
              className={cn(
                "inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors",
                on
                  ? "border-foreground/60 bg-foreground/5 text-foreground"
                  : "border-border bg-background text-muted-foreground hover:bg-muted/40",
              )}
            >
              <span
                className={cn(
                  "inline-block h-2 w-2 rounded-full",
                  on ? "bg-foreground" : "bg-muted-foreground/40",
                )}
              />
              {SECTION_LABEL[s]}
            </button>
          );
        })}
      </div>

      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete {role.name}?</DialogTitle>
            <DialogDescription>
              Users currently attached to this role will lose all section access
              until you re-attach them to another role. This can&apos;t be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteOpen(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={doDelete} disabled={pending}>
              <Trash2 className="h-4 w-4" />
              Delete role
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </li>
  );
}

function CreateRoleDialog({
  open,
  onOpenChange,
  existingIds,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  existingIds: Set<string>;
  onCreated: (r: PolicyRole) => void;
}) {
  const [name, setName] = useState("");
  const [id, setId] = useState("");
  const [idTouched, setIdTouched] = useState(false);
  const [sections, setSections] = useState<Record<SectionKey, boolean>>({
    inference: false,
    benchmark: false,
    compute: false,
    datasets: false,
  });
  const [pending, setPending] = useState(false);

  const effectiveId = idTouched ? id : slugify(name);
  const idIsValid = /^[a-z0-9-]{2,64}$/.test(effectiveId);
  const idIsDupe = existingIds.has(effectiveId);

  const reset = () => {
    setName("");
    setId("");
    setIdTouched(false);
    setSections({ inference: false, benchmark: false, compute: false, datasets: false });
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !idIsValid || idIsDupe) return;
    setPending(true);
    try {
      const r = await gateway.adminCreatePolicyRole(effectiveId, name.trim(), sections);
      onCreated(r);
      toast.success(`Created ${r.name}`, { duration: 3000 });
      reset();
      onOpenChange(false);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e)), { duration: 5000 };
    } finally {
      setPending(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!pending) onOpenChange(o);
        if (!o) reset();
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Create role</DialogTitle>
          <DialogDescription>
            Pick which sections this role unlocks. The id is the slug used in
            the URL — auto-derived from the name if you don&apos;t edit it.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={submit} className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="role-name">Name</Label>
            <Input
              id="role-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Researcher"
              disabled={pending}
              required
              maxLength={128}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="role-id">ID (slug)</Label>
            <Input
              id="role-id"
              value={effectiveId}
              onChange={(e) => {
                setIdTouched(true);
                setId(e.target.value);
              }}
              placeholder="researcher"
              disabled={pending}
              maxLength={64}
              className="font-mono text-sm"
            />
            {!idIsValid && effectiveId.length > 0 && (
              <p className="text-xs text-destructive">
                Lowercase letters, digits, dashes; 2-64 chars.
              </p>
            )}
            {idIsValid && idIsDupe && (
              <p className="text-xs text-destructive">A role with that id already exists.</p>
            )}
          </div>
          <div className="space-y-1.5">
            <Label>Sections</Label>
            <div className="flex flex-wrap gap-2">
              {SECTIONS.map((s) => {
                const on = sections[s];
                return (
                  <button
                    key={s}
                    type="button"
                    onClick={() =>
                      setSections((prev) => ({ ...prev, [s]: !prev[s] }))
                    }
                    disabled={pending}
                    className={cn(
                      "inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors",
                      on
                        ? "border-foreground/60 bg-foreground/5 text-foreground"
                        : "border-border bg-background text-muted-foreground hover:bg-muted/40",
                    )}
                  >
                    <span
                      className={cn(
                        "inline-block h-2 w-2 rounded-full",
                        on ? "bg-foreground" : "bg-muted-foreground/40",
                      )}
                    />
                    {SECTION_LABEL[s]}
                  </button>
                );
              })}
            </div>
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={pending}>
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={pending || !name.trim() || !idIsValid || idIsDupe}
            >
              {pending && <Loader2 className="h-4 w-4 animate-spin" />}
              Create role
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
