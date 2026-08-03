import Link from "next/link";
import { Inbox, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { gateway } from "@/lib/gateway";
import type { PromptOptRecord, PageResponse } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { ScopeToggle } from "@/components/scope-toggle";
import { SectionTabs } from "../section-tabs";
import { OptimizeList } from "./optimize-list";

async function loadOptimizations(
  scope: "mine" | "all",
): Promise<{ page: PageResponse<PromptOptRecord>; error: string | null }> {
  try {
    return {
      page: await gateway.listPromptOptsPage({ scope, limit: 12, offset: 0 }),
      error: null,
    };
  } catch (e) {
    return { page: { total: 0, items: [] }, error: e instanceof Error ? e.message : String(e) };
  }
}

export default async function OptimizePage({
  searchParams,
}: {
  searchParams: Promise<{ scope?: string }>;
}) {
  const me = await getMe();
  const sections = me?.sections as Record<string, boolean> | undefined;
  const noAccess = me ? !(me.is_admin || sections?.experiments) : false;
  const sp = await searchParams;
  const scope: "mine" | "all" = me?.is_admin && sp.scope === "all" ? "all" : "mine";

  const [{ page, error }, username] = await Promise.all([
    noAccess
      ? Promise.resolve({ page: { total: 0, items: [] }, error: null })
      : loadOptimizations(scope),
    currentUsername(),
  ]);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Experiments", href: "/experiments" }, { label: "Optimize" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <h1 className="text-2xl font-semibold tracking-tight">Prompt optimization</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              GEPA searches for a better system prompt by reading its own failures: it replays your
              dataset, scores every reply with the evaluators you already use, and has a reflection
              model rewrite the prompt from what went wrong — keeping only what measurably improves.
              The winner drops straight into an experiment as a variant.
            </p>
          </div>
          {!noAccess && me?.is_admin && <ScopeToggle scope={scope} />}
        </div>

        <SectionTabs />

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
                  {me?.is_admin && scope === "all" && " · all users"}
                </span>
              </div>
              <Button asChild size="sm">
                <Link href="/experiments/optimize/new">
                  <Plus className="h-4 w-4" />
                  New optimization
                </Link>
              </Button>
            </div>

            {page.total === 0 ? (
              <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
                <Inbox className="h-6 w-6 text-muted-foreground/60" />
                <p className="max-w-lg text-sm text-muted-foreground">
                  No optimizations yet. Point one at a chat{" "}
                  <Link
                    href="/datasets"
                    className="font-medium text-foreground underline underline-offset-2"
                  >
                    dataset
                  </Link>{" "}
                  and the{" "}
                  <Link
                    href="/experiments/evaluators"
                    className="font-medium text-foreground underline underline-offset-2"
                  >
                    evaluators
                  </Link>{" "}
                  that define a good reply — those evaluators are the score it climbs.
                </p>
              </div>
            ) : (
              <OptimizeList
                key={scope}
                initialItems={page.items}
                initialTotal={page.total}
                scope={scope}
              />
            )}
          </section>
        )}
      </div>
    </div>
  );
}
