import Link from "next/link";
import { Inbox, PackageOpen, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { gateway } from "@/lib/gateway";
import type { BenchmarkRecord } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { BenchmarkList } from "./benchmark-list";
import { BenchmarkDashboard } from "./dashboard";
import { ExplorerCollapsible } from "./explorer-collapsible";
import { ScopeToggle } from "@/components/scope-toggle";

async function loadBenchmarks(
  scope: "mine" | "all",
): Promise<{ items: BenchmarkRecord[]; error: string | null }> {
  try {
    const items = await gateway.listBenchmarks(scope);
    return { items, error: null };
  } catch (e) {
    return { items: [], error: e instanceof Error ? e.message : String(e) };
  }
}

export default async function BenchmarkPage({
  searchParams,
}: {
  searchParams: Promise<{ scope?: string }>;
}) {
  const me = await getMe();
  const noAccess = !me?.sections?.benchmark;
  const sp = await searchParams;
  const scope: "mine" | "all" =
    me?.is_admin && sp.scope === "all" ? "all" : "mine";

  const [{ items, error }, username] = await Promise.all([
    noAccess ? Promise.resolve({ items: [], error: null }) : loadBenchmarks(scope),
    currentUsername(),
  ]);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Benchmark" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Benchmark</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Run <span className="font-mono text-xs">llm-benchmaq</span> sweeps on real GPUs.
              Results land in S3; metrics and files surface in the detail view.
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

        {!noAccess && items.length > 0 && <BenchmarkDashboard items={items} />}
        {!noAccess && items.length > 0 && <ExplorerCollapsible scope={scope} />}

        {!noAccess && (
          <section>
            <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
              <div className="flex items-baseline gap-3">
                <h2 className="text-base font-medium">Benchmarks</h2>
                <span className="text-xs text-muted-foreground">
                  {items.length} {items.length === 1 ? "run" : "runs"}
                  {me?.is_admin && scope === "all" && " · all users"}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Button asChild size="sm" variant="outline">
                  <Link href="/benchmark/import">
                    <PackageOpen className="h-4 w-4" />
                    Import
                  </Link>
                </Button>
                <Button asChild size="sm">
                  <Link href="/benchmark/new">
                    <Plus className="h-4 w-4" />
                    New benchmark
                  </Link>
                </Button>
              </div>
            </div>
            {items.length === 0 ? (
              <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
                <Inbox className="h-6 w-6 text-muted-foreground/60" />
                <p className="text-sm text-muted-foreground">
                  No benchmarks yet. Click <span className="font-medium text-foreground">New benchmark</span> to start one.
                </p>
              </div>
            ) : (
              <BenchmarkList items={items} />
            )}
          </section>
        )}
      </div>
    </div>
  );
}
