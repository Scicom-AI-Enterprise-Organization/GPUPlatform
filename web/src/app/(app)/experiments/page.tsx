import Link from "next/link";
import { Inbox, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { gateway } from "@/lib/gateway";
import type { ExperimentRecord, PageResponse } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { ExperimentsList } from "./experiments-list";

async function loadExperiments(
): Promise<{ page: PageResponse<ExperimentRecord>; error: string | null }> {
  try {
    return {
      page: await gateway.listExperimentsPage({ limit: 12, offset: 0 }),
      error: null,
    };
  } catch (e) {
    return { page: { total: 0, items: [] }, error: e instanceof Error ? e.message : String(e) };
  }
}

export default async function ExperimentsPage() {
  const me = await getMe();
  const sections = me?.sections as Record<string, boolean> | undefined;
  const noAccess = me ? !(me.is_admin || sections?.experiments) : false;

  const [{ page, error }, username] = await Promise.all([
    noAccess ? Promise.resolve({ page: { total: 0, items: [] }, error: null }) : loadExperiments(),
    currentUsername(),
  ]);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Experiments" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <h1 className="text-2xl font-semibold tracking-tight">Experiments</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Replay captured requests against your endpoints and score every reply. Point one at
              any chat dataset from the Datasets section — or capture one from a Langfuse trace or
              your own served traffic — then sweep it across endpoints and prompt variants to find
              control-token leaks, empty replies, degeneration, broken JSON, and latency or cost
              regressions.
            </p>
          </div>
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
                <h2 className="text-base font-medium">Runs</h2>
                <span className="text-xs text-muted-foreground">
                  {page.total} {page.total === 1 ? "run" : "runs"}
                </span>
              </div>
              <Button asChild size="sm">
                <Link href="/experiments/new">
                  <Plus className="h-4 w-4" />
                  New experiment
                </Link>
              </Button>
            </div>

            {page.total === 0 ? (
              <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
                <Inbox className="h-6 w-6 text-muted-foreground/60" />
                <p className="text-sm text-muted-foreground">
                  No experiments yet. Point one at a{" "}
                  <Link href="/datasets" className="font-medium text-foreground underline underline-offset-2">
                    dataset
                  </Link>{" "}
                  with a messages column — or capture requests into a new one from the form.
                </p>
              </div>
            ) : (
              <ExperimentsList
                initialItems={page.items}
                initialTotal={page.total}
              />
            )}
          </section>
        )}
      </div>
    </div>
  );
}
