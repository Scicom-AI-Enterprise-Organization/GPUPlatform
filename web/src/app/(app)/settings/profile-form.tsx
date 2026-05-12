"use client";

import { useState, useTransition } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function ProfileForm({
  username,
  email,
  role,
}: {
  username: string;
  email: string;
  role: string;
}) {
  const [pending, startTransition] = useTransition();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [err, setErr] = useState<string | null>(null);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    if (next.length < 8) {
      setErr("New password must be at least 8 characters.");
      return;
    }
    if (next !== confirm) {
      setErr("New passwords don't match.");
      return;
    }
    startTransition(async () => {
      const r = await fetch("/api/auth/change-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ current_password: current, new_password: next }),
      });
      const body = await r.json().catch(() => null);
      if (!r.ok) {
        let msg: string | undefined;
        if (Array.isArray(body?.detail)) {
          msg = body.detail
            .map((d: { loc?: string[]; msg?: string }) =>
              `${(d.loc ?? []).slice(-1)[0] ?? "field"}: ${d.msg ?? "invalid"}`,
            )
            .join("; ");
        } else {
          msg = body?.detail?.error ?? body?.error;
        }
        setErr(msg ?? `Failed (${r.status})`);
        return;
      }
      toast.success("Password changed — sign in again to continue", { duration: 4000 });
      setTimeout(() => {
        window.location.href = "/login";
      }, 800);
    });
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base font-semibold">Profile</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Field label="Username" value={username} />
          <Field label="Email" value={email || "—"} />
          <Field label="Role" value={role} className="capitalize" />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base font-semibold">Change password</CardTitle>
          <p className="text-xs text-muted-foreground">
            You&apos;ll be signed out after a successful change and have to log in again with the new password.
          </p>
        </CardHeader>
        <CardContent>
          <form onSubmit={submit} className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="current">Current password</Label>
              <Input
                id="current"
                type="password"
                required
                autoComplete="current-password"
                value={current}
                onChange={(e) => setCurrent(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="next">New password</Label>
              <Input
                id="next"
                type="password"
                required
                autoComplete="new-password"
                value={next}
                onChange={(e) => setNext(e.target.value)}
                placeholder="At least 8 characters"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="confirm">Confirm new password</Label>
              <Input
                id="confirm"
                type="password"
                required
                autoComplete="new-password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
              />
            </div>
            {err && (
              <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                {err}
              </div>
            )}
            <Button type="submit" disabled={pending}>
              {pending && <Loader2 className="h-4 w-4 animate-spin" />}
              Change password
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

function Field({ label, value, className = "" }: { label: string; value: string; className?: string }) {
  return (
    <div className="space-y-1.5">
      <Label>{label}</Label>
      <Input value={value} readOnly disabled className={`bg-muted/30 ${className}`} />
    </div>
  );
}
