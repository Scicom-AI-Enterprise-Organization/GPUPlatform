import Link from "next/link";
import { FlaskConical, Plus } from "lucide-react";
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

async function loadBenchmarks(): Promise<{ items: BenchmarkRecord[]; error: string | null }> {
  try {
    const items = await gateway.listBenchmarks();
    return { items, error: null };
  } catch (e) {
    return { items: [], error: e instanceof Error ? e.message : String(e) };
  }
}

export default async function BenchmarkPage() {
  const me = await getMe();
  const noAccess = me?.role === "user";
  const [{ items, error }, username] = await Promise.all([
    noAccess ? Promise.resolve({ items: [], error: null }) : loadBenchmarks(),
    currentUsername(),
  ]);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Benchmark" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6 flex items-start gap-4 rounded-xl border border-border bg-gradient-to-br from-violet-500/10 via-background to-fuchsia-500/5 p-6">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 text-white shadow-lg shadow-fuchsia-500/30">
            <FlaskConical className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Benchmark</h1>
            <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
              Run <span className="font-mono text-xs">llm-benchmaq</span> sweeps on real GPUs.
              Results land in S3; metrics and files surface in the detail view.
            </p>
          </div>
        </div>

        {noAccess && <NoAccessAlert />}

        {error && !noAccess && (
          <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            Couldn&apos;t reach the gateway: {error}
          </div>
        )}

        {!noAccess && items.length > 0 && <BenchmarkDashboard items={items} />}
        {!noAccess && items.length > 0 && <ExplorerCollapsible />}

        {!noAccess && (
          <section>
            <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
              <div className="flex items-baseline gap-3">
                <h2 className="text-base font-medium">Benchmarks</h2>
                <span className="text-xs text-muted-foreground">
                  {items.length} {items.length === 1 ? "run" : "runs"}
                </span>
              </div>
              <Button asChild size="sm">
                <Link href="/benchmark/new">
                  <Plus className="h-4 w-4" />
                  New benchmark
                </Link>
              </Button>
            </div>
            {items.length === 0 ? (
              <div className="rounded-md border border-dashed border-border px-6 py-12 text-center text-sm text-muted-foreground">
                No benchmarks yet. Click <span className="font-medium">New benchmark</span> to start one.
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
