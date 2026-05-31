import Link from "next/link";
import { Inbox, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { gateway } from "@/lib/gateway";
import type { TrainingRunRecord } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { ScopeToggle } from "@/components/scope-toggle";
import { AutotrainList } from "./autotrain-list";

async function loadRuns(
  scope: "mine" | "all",
): Promise<{ items: TrainingRunRecord[]; error: string | null }> {
  try {
    return { items: await gateway.listTrainingRuns(scope), error: null };
  } catch (e) {
    return { items: [], error: e instanceof Error ? e.message : String(e) };
  }
}

export default async function AutotrainPage({
  searchParams,
}: {
  searchParams: Promise<{ scope?: string }>;
}) {
  const me = await getMe();
  const sections = me?.sections as Record<string, boolean> | undefined;
  const noAccess = me ? !(me.is_admin || sections?.autotrain) : false;
  const sp = await searchParams;
  const scope: "mine" | "all" = me?.is_admin && sp.scope === "all" ? "all" : "mine";

  const [{ items, error }, username] = await Promise.all([
    noAccess ? Promise.resolve({ items: [], error: null }) : loadRuns(scope),
    currentUsername(),
  ]);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Autotrain" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Autotrain</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Finetune a Whisper model on a dataset. WER + CER are evaluated every
              epoch; training stops at the max-epoch cap or early on patience.
            </p>
          </div>
          {!noAccess && me?.is_admin && <ScopeToggle scope={scope} />}
        </div>

        {noAccess && <NoAccessAlert />}

        {error && !noAccess && (
          <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            Couldn&apos;t reach the gateway: {error}
          </div>
        )}

        {!noAccess && (
          <section>
            <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
              <div className="flex items-baseline gap-3">
                <h2 className="text-base font-medium">Training runs</h2>
                <span className="text-xs text-muted-foreground">
                  {items.length} {items.length === 1 ? "run" : "runs"}
                  {me?.is_admin && scope === "all" && " · all users"}
                </span>
              </div>
              <Button asChild size="sm">
                <Link href="/autotrain/new">
                  <Plus className="h-4 w-4" />
                  New run
                </Link>
              </Button>
            </div>

            {items.length === 0 ? (
              <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
                <Inbox className="h-6 w-6 text-muted-foreground/60" />
                <p className="text-sm text-muted-foreground">
                  No training runs yet. Click{" "}
                  <span className="font-medium text-foreground">New run</span> to start one.
                </p>
              </div>
            ) : (
              <AutotrainList items={items} />
            )}
          </section>
        )}
      </div>
    </div>
  );
}
