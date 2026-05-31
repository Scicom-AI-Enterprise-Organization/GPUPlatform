import Link from "next/link";
import { Inbox, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { ScopeToggle } from "@/components/scope-toggle";
import { gateway } from "@/lib/gateway";
import type { ComputePod } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { ComputeList } from "./compute-list";

async function loadCompute(
  scope: "mine" | "all",
): Promise<{ items: ComputePod[]; error: string | null }> {
  try {
    const items = await gateway.listCompute(scope);
    return { items, error: null };
  } catch (e) {
    return { items: [], error: e instanceof Error ? e.message : String(e) };
  }
}

export default async function ComputePage({
  searchParams,
}: {
  searchParams: Promise<{ scope?: string }>;
}) {
  const me = await getMe();
  const noAccess = !me?.sections?.compute;
  const sp = await searchParams;
  const scope: "mine" | "all" =
    me?.is_admin && sp.scope === "all" ? "all" : "mine";

  const [{ items, error }, username] = await Promise.all([
    noAccess ? Promise.resolve({ items: [], error: null }) : loadCompute(scope),
    currentUsername(),
  ]);

  const active = items.filter((p) => p.status === "running" || p.status === "creating").length;

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Compute" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        {/* Plain header — no gradient, no coloured icon tile. Colour is reserved
            for state. */}
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Compute</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Provision a raw GPU pod with SSH and JupyterLab. Pods bill per-second
              until you terminate them — pick a template, click create, you get back
              an SSH command and a JupyterLab URL.
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
                <h2 className="text-base font-medium">Pods</h2>
                <span className="text-xs text-muted-foreground">
                  {items.length} total · {active} active
                  {me?.is_admin && scope === "all" && " · all users"}
                </span>
              </div>
              <Button asChild size="sm">
                <Link href="/compute/new">
                  <Plus className="h-4 w-4" />
                  New pod
                </Link>
              </Button>
            </div>
            {items.length === 0 ? (
              <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
                <Inbox className="h-6 w-6 text-muted-foreground/60" />
                <p className="text-sm text-muted-foreground">
                  No pods yet. Click <span className="font-medium text-foreground">New pod</span> to spin one up.
                </p>
              </div>
            ) : (
              <ComputeList items={items} />
            )}
          </section>
        )}
      </div>
    </div>
  );
}
