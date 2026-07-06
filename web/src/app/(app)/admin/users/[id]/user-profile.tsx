"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";
import { Check, Copy, Loader2, Trash2 } from "lucide-react";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { AdminUserRecord, PolicyRole, SectionKey } from "@/lib/types";
import { cn } from "@/lib/utils";

const SECTION_LABEL: Record<SectionKey, string> = {
  inference: "Inference",
  benchmark: "Benchmark",
  compute: "Compute",
  datasets: "Datasets",
  quantization: "Quantization",
  catalog: "Models",
};

const NO_ROLE = "__none__";

export function UserProfile({
  user,
  policyRoles,
  isSelf,
}: {
  user: AdminUserRecord;
  policyRoles: PolicyRole[];
  isSelf: boolean;
}) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const [deleteOpen, setDeleteOpen] = useState(false);

  function setTier(next: AdminUserRecord["role"]) {
    if (next === user.role) return;
    startTransition(async () => {
      const r = await fetch(`/api/proxy/admin/users/${user.id}/role`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role: next }),
      });
      if (!r.ok) {
        toast.error(`Failed: ${await r.text()}`, { duration: 5000 });
        return;
      }
      toast.success(`${user.username} → ${next}`, { duration: 3000 });
      router.refresh();
    });
  }

  function setPolicy(next: string) {
    const value = next === NO_ROLE ? null : next;
    if (value === user.policy_role_id) return;
    startTransition(async () => {
      const r = await fetch(`/api/proxy/admin/users/${user.id}/policy-role`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ policy_role_id: value }),
      });
      if (!r.ok) {
        toast.error(`Failed: ${await r.text()}`, { duration: 5000 });
        return;
      }
      toast.success(value ? `${user.username} → ${value}` : `${user.username} detached from role`, { duration: 3000 });
      router.refresh();
    });
  }

  function deleteUser() {
    setDeleteOpen(false);
    startTransition(async () => {
      const r = await fetch(`/api/proxy/admin/users/${user.id}`, { method: "DELETE" });
      if (!r.ok) {
        toast.error(`Failed: ${await r.text()}`, { duration: 5000 });
        return;
      }
      toast.success(`Deleted ${user.username}`, { duration: 3000 });
      router.replace("/organization");
    });
  }

  // Policy role only matters when the user is on a tier that gates on it.
  const policyDisabled = pending || user.role === "user";

  return (
    <div className="space-y-6">
      <header className="flex items-start gap-4 border-b border-border pb-5">
        <div className="flex h-14 w-14 items-center justify-center rounded-full border border-border bg-muted text-xl font-semibold text-muted-foreground">
          {(user.username[0] ?? "?").toUpperCase()}
        </div>
        <div className="min-w-0 flex-1">
          <h1 className="text-2xl font-semibold tracking-tight">{user.username}</h1>
          <p className="mt-0.5 font-mono text-xs text-muted-foreground">id={user.id}</p>
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <Badge>{user.role}</Badge>
            <AuthBadge provider={user.auth_provider} />
            {user.policy_role_name && <Badge>{user.policy_role_name}</Badge>}
          </div>
        </div>
      </header>

      <Section title="Account">
        <Field label="Email" value={user.email ?? "—"} copyable={!!user.email} />
        <Field
          label="Sign-in method"
          value={
            user.auth_provider === "github"
              ? `GitHub SSO${user.github_id ? ` (${user.github_id})` : ""}`
              : "Email + password"
          }
        />
        <Field label="Created" value={fmtDate(user.created_at)} />
      </Section>

      <Section title="Access">
        <div>
          <Label>Tier</Label>
          <Select
            value={user.role}
            disabled={pending || isSelf}
            onValueChange={(v) => setTier(v as AdminUserRecord["role"])}
          >
            <SelectTrigger
              className="mt-1 w-full capitalize"
              title={isSelf ? "You can't change your own tier" : ""}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="user" className="capitalize">user</SelectItem>
              <SelectItem value="developer" className="capitalize">developer</SelectItem>
              <SelectItem value="admin" className="capitalize">admin</SelectItem>
            </SelectContent>
          </Select>
          {isSelf && (
            <p className="mt-1 text-[11px] text-muted-foreground">
              You can&apos;t change your own tier.
            </p>
          )}
        </div>

        <div>
          <Label>Policy role</Label>
          <Select
            value={user.policy_role_id ?? NO_ROLE}
            disabled={policyDisabled}
            onValueChange={setPolicy}
          >
            <SelectTrigger
              className="mt-1 w-full"
              title={user.role === "user" ? "Promote to developer first" : ""}
            >
              <SelectValue placeholder="No role" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={NO_ROLE}>No role (no access)</SelectItem>
              {policyRoles.map((r) => (
                <SelectItem key={r.id} value={r.id}>
                  {r.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {user.role === "user" && (
            <p className="mt-1 text-[11px] text-muted-foreground">
              Promote tier to <span className="font-medium">developer</span> first to attach a policy role.
            </p>
          )}
        </div>

        <div className="sm:col-span-2">
          <Label>Sections</Label>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {(Object.entries(user.section_permissions) as [SectionKey, boolean][]).map(
              ([key, allowed]) => (
                <span
                  key={key}
                  className={cn(
                    "inline-flex items-center rounded-md border border-border bg-muted/40 px-2 py-0.5 text-[11px]",
                    !allowed && "text-muted-foreground line-through opacity-70",
                  )}
                >
                  {SECTION_LABEL[key]}
                </span>
              ),
            )}
          </div>
        </div>
      </Section>

      <section data-form-section="Danger zone" className="scroll-mt-6 rounded-lg border border-destructive/30 bg-destructive/5 p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-sm font-semibold text-destructive">Danger zone</h2>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Removes the user and any endpoints they own. Cannot be undone.
            </p>
          </div>
          <Button
            variant="outline"
            disabled={pending || isSelf}
            title={isSelf ? "You can't delete yourself" : "Delete user"}
            onClick={() => setDeleteOpen(true)}
            className="text-destructive hover:text-destructive"
          >
            {pending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Trash2 className="h-4 w-4" />
            )}
            Delete user
          </Button>
        </div>
      </section>

      <Dialog open={deleteOpen} onOpenChange={(o) => !pending && setDeleteOpen(o)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete {user.username}?</DialogTitle>
            <DialogDescription>
              This permanently removes the user and any endpoints they own.
              This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteOpen(false)} disabled={pending}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={deleteUser} disabled={pending}>
              {pending && <Loader2 className="h-4 w-4 animate-spin" />}
              Delete user
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section data-form-section={title} className="scroll-mt-6 rounded-lg border border-border bg-card p-5">
      <h2 className="mb-4 text-sm font-semibold">{title}</h2>
      <dl className="grid gap-4 sm:grid-cols-2">{children}</dl>
    </section>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
      {children}
    </span>
  );
}

function Field({
  label,
  value,
  copyable = false,
}: {
  label: string;
  value: string;
  copyable?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      toast.success(`${label} copied`, { duration: 3000 });
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      toast.error("Couldn't access clipboard", { duration: 5000 });
    }
  }
  return (
    <div className="min-w-0">
      <Label>{label}</Label>
      <dd className="mt-1 flex min-w-0 items-center gap-1.5 break-all text-sm">
        <span className="min-w-0 flex-1">{value}</span>
        {copyable && (
          <button
            type="button"
            onClick={copy}
            aria-label={copied ? "Copied" : `Copy ${label}`}
            title={copied ? "Copied" : `Copy ${label}`}
            className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md border border-transparent text-muted-foreground hover:border-border hover:bg-muted/60 hover:text-foreground"
          >
            {copied ? (
              <Check className="h-3.5 w-3.5" />
            ) : (
              <Copy className="h-3.5 w-3.5" />
            )}
          </button>
        )}
      </dd>
    </div>
  );
}

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-md border border-border bg-muted/60 px-2 py-0.5 text-[11px] capitalize text-muted-foreground">
      {children}
    </span>
  );
}

function AuthBadge({ provider }: { provider: AdminUserRecord["auth_provider"] }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-border bg-background px-2 py-0.5 text-[11px] text-muted-foreground">
      {provider === "github" ? <GithubMark /> : <span aria-hidden>✉</span>}
      {provider === "github" ? "GitHub SSO" : "Password"}
    </span>
  );
}

function GithubMark() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="11"
      height="11"
      aria-hidden="true"
      fill="currentColor"
    >
      <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56 0-.27-.01-1.16-.02-2.1-3.2.69-3.87-1.36-3.87-1.36-.52-1.34-1.27-1.7-1.27-1.7-1.04-.71.08-.69.08-.69 1.15.08 1.76 1.18 1.76 1.18 1.02 1.75 2.68 1.24 3.34.95.1-.74.4-1.24.73-1.52-2.55-.29-5.24-1.28-5.24-5.7 0-1.26.45-2.29 1.18-3.1-.12-.29-.51-1.46.11-3.04 0 0 .97-.31 3.18 1.18.92-.26 1.91-.39 2.89-.39.98 0 1.97.13 2.89.39 2.21-1.49 3.18-1.18 3.18-1.18.62 1.58.23 2.75.11 3.04.74.81 1.18 1.84 1.18 3.1 0 4.43-2.7 5.41-5.27 5.69.41.36.78 1.06.78 2.14 0 1.55-.01 2.79-.01 3.17 0 .31.21.67.8.56C20.21 21.39 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5Z" />
    </svg>
  );
}

function fmtDate(iso: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString();
}
