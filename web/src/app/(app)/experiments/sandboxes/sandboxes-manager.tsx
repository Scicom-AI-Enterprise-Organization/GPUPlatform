"use client";

// The Sandboxes tab: your reusable library on top, the modes the gateway
// actually implements below. Mirrors evaluators-manager.tsx — diff against it
// before changing structure here.

import { useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { Database, Globe, Inbox, Pencil, Plus, Sparkles, Terminal, Trash2 } from "lucide-react";
import { gateway } from "@/lib/gateway";
import type { CustomSandboxRecord, SandboxRegistry } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const MODE_ICON: Record<string, React.ElementType> = {
  replay: Database,
  api: Globe,
  llm: Sparkles,
  python: Terminal,
};

/** One-line summary of where this sandbox's answers come from. */
function summarize(s: CustomSandboxRecord): string {
  const cfg = s.config ?? {};
  if (s.mode === "api") {
    return String((cfg.api as Record<string, unknown> | undefined)?.url ?? "");
  }
  const replay = (cfg.replay as Record<string, unknown> | undefined) ?? {};
  const field = String(replay.seed_field ?? "tool_seed");
  const match = String(replay.match ?? "name");
  return `expected.${field} · matched by ${match}`;
}

export function SandboxesManager({ registry }: { registry: SandboxRegistry }) {
  const [rows, setRows] = useState<CustomSandboxRecord[]>(registry.sandboxes ?? []);
  const [pending, setPending] = useState<CustomSandboxRecord | null>(null);
  const [deleting, setDeleting] = useState(false);

  async function onDelete() {
    if (!pending) return;
    setDeleting(true);
    try {
      await gateway.deleteCustomSandbox(pending.id);
      setRows((xs) => xs.filter((x) => x.id !== pending.id));
      toast.success(`Deleted ${pending.name}`);
      setPending(null);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Couldn't delete");
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="space-y-8">
      {/* ---------------- your library ---------------- */}
      <section>
        {/* Same list header as /experiments and /experiments/optimize — heading
            plus a plain count. Keep the three in step. */}
        <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
          <div className="flex items-baseline gap-3">
            <h2 className="text-base font-medium">Sandboxes</h2>
            <span className="text-xs text-muted-foreground">
              {rows.length} {rows.length === 1 ? "sandbox" : "sandboxes"}
            </span>
          </div>
          <Button asChild size="sm">
            <Link href="/experiments/sandboxes/new">
              <Plus className="h-4 w-4" />
              New sandbox
            </Link>
          </Button>
        </div>

        {rows.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 rounded-md border border-dashed border-border bg-card px-6 py-14 text-center">
            <Inbox className="h-6 w-6 text-muted-foreground/60" />
            <p className="max-w-md text-sm text-muted-foreground">
              None yet. Replay tool results straight off the dataset row (free, offline,
              reproducible), or point at a service you already run.
            </p>
          </div>
        ) : (
          <ul className="space-y-2">
            {rows.map((s) => {
              const Icon = MODE_ICON[s.mode] ?? Database;
              const loop = (s.config?.loop as Record<string, unknown> | undefined) ?? {};
              return (
                <li
                  key={s.id}
                  className="flex items-start gap-3 rounded-md border border-border bg-card px-3 py-2.5 transition-colors hover:border-foreground/25"
                >
                  <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground">
                    <Icon className="h-3.5 w-3.5" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span className="text-sm font-medium">{s.name}</span>
                      <Badge variant="secondary" className="text-[10px] font-normal">
                        {s.mode}
                      </Badge>
                      {loop.max_tool_rounds !== undefined && (
                        <Badge variant="secondary" className="text-[10px] font-normal">
                          ≤{String(loop.max_tool_rounds)} rounds
                        </Badge>
                      )}
                    </div>
                    {s.description && (
                      <p className="mt-0.5 text-xs leading-snug text-muted-foreground">
                        {s.description}
                      </p>
                    )}
                    <pre className="mt-1 truncate font-mono text-[11px] text-muted-foreground">
                      {summarize(s)}
                    </pre>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <Link
                      href={`/experiments/sandboxes/new?id=${s.id}`}
                      title="Edit"
                      className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </Link>
                    <button
                      onClick={() => setPending(s)}
                      title="Delete"
                      className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <Dialog open={!!pending} onOpenChange={(o) => !o && setPending(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete sandbox?</DialogTitle>
            <DialogDescription>
              {pending?.name} will be removed from your library. Experiments that already used it
              keep their results — each run stores its own snapshot of the definition.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPending(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={() => void onDelete()} disabled={deleting}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
