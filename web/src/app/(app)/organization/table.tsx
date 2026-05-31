"use client";

import Link from "next/link";
import { ChevronRight } from "lucide-react";
import type { AdminUserRecord, SectionKey } from "@/lib/types";
import { cn } from "@/lib/utils";

const SECTION_LABEL: Record<SectionKey, string> = {
  inference: "Inference",
  benchmark: "Benchmark",
  compute: "Compute",
  datasets: "Datasets",
};

function fmtDate(iso: string) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString();
  } catch {
    return iso;
  }
}

// Read-only directory view. Editing tier / policy role / deleting users
// happens on the profile page (/admin/users/{id}) — clicking a row takes
// you there. Keeping the table itself a clean overview avoids the cramped
// dropdown grid we had before.
export function OrganizationTable({ users }: { users: AdminUserRecord[] }) {
  if (users.length === 0) {
    return (
      <div className="px-5 py-12 text-center text-sm text-muted-foreground">
        No users.
      </div>
    );
  }
  return (
    <ul className="divide-y divide-border">
      {users.map((u) => (
        <li key={u.id}>
          <Link
            href={`/admin/users/${u.id}`}
            className="group flex items-center gap-4 px-5 py-3 transition-colors hover:bg-muted/40"
          >
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-border bg-muted text-sm font-semibold text-muted-foreground">
              {(u.username[0] ?? "?").toUpperCase()}
            </div>

            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="truncate font-medium text-foreground">
                  {u.username}
                </span>
                <TierBadge role={u.role} />
                <span className="rounded-md border border-border bg-background px-1.5 py-0.5 text-[10px] text-muted-foreground">
                  {u.policy_role_name ?? "no role"}
                </span>
              </div>
              <div className="mt-0.5 flex items-center gap-2 text-xs text-muted-foreground">
                <span className="font-mono">id={u.id}</span>
                <span>·</span>
                <span className="truncate font-mono" title={u.email ?? undefined}>
                  {u.email ?? "no email"}
                </span>
                <span>·</span>
                <span className="shrink-0">created {fmtDate(u.created_at)}</span>
              </div>
            </div>

            <div className="hidden shrink-0 gap-1 sm:flex">
              {(["inference", "benchmark", "compute"] as SectionKey[]).map((s) => {
                const allowed = u.section_permissions[s];
                return (
                  <span
                    key={s}
                    className={cn(
                      "inline-flex items-center rounded-md border px-1.5 py-0.5 text-[10px]",
                      allowed
                        ? "border-border bg-muted text-foreground"
                        : "border-border bg-background text-muted-foreground/60 line-through",
                    )}
                    title={allowed ? `Has ${s} access` : `No ${s} access`}
                  >
                    {SECTION_LABEL[s]}
                  </span>
                );
              })}
            </div>

            <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground/60 transition-transform group-hover:translate-x-0.5" />
          </Link>
        </li>
      ))}
    </ul>
  );
}

function TierBadge({ role }: { role: AdminUserRecord["role"] }) {
  return (
    <span className="rounded-md border border-border bg-muted/60 px-1.5 py-0.5 text-[10px] capitalize text-muted-foreground">
      {role}
    </span>
  );
}
