import { cn } from "@/lib/utils";
import type { GitopsSyncStatus } from "@/lib/types";

export function SyncStatusPill({ status, className }: { status: GitopsSyncStatus; className?: string }) {
  const map: Record<GitopsSyncStatus, { label: string; cls: string; dot: string }> = {
    ok: { label: "synced", cls: "border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400", dot: "bg-emerald-500" },
    syncing: { label: "syncing", cls: "border-sky-500/40 bg-sky-500/10 text-sky-600 dark:text-sky-400", dot: "bg-sky-500 animate-pulse" },
    error: { label: "error", cls: "border-destructive/40 bg-destructive/10 text-destructive", dot: "bg-destructive" },
    never: { label: "never synced", cls: "border-border bg-muted text-muted-foreground", dot: "bg-muted-foreground/50" },
  };
  const s = map[status] ?? map.never;
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-medium", s.cls, className)}>
      <span className={cn("h-1.5 w-1.5 rounded-full", s.dot)} />
      {s.label}
    </span>
  );
}

export function fmtWhen(iso: string | null | undefined): string {
  if (!iso) return "—";
  const t = new Date(iso).getTime();
  if (!Number.isFinite(t)) return "—";
  const diff = Date.now() - t;
  const s = Math.round(diff / 1000);
  if (s < 0) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.round(h / 24);
  return `${d}d ago`;
}
