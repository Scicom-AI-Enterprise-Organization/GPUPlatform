import Link from "next/link";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { ScopeToggle } from "@/components/scope-toggle";
import { gateway } from "@/lib/gateway";
import type { AppRecord } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { EndpointGrid } from "./endpoint-grid";

async function loadEndpoints(
  scope: "mine" | "all",
): Promise<{ apps: AppRecord[]; error: string | null }> {
  try {
    const apps = await gateway.listApps(scope);
    return { apps, error: null };
  } catch (e) {
    return { apps: [], error: e instanceof Error ? e.message : String(e) };
  }
}

export default async function ServerlessPage({
  searchParams,
}: {
  searchParams: Promise<{ scope?: string }>;
}) {
  const me = await getMe();
  const noAccess = !me?.sections?.inference;
  const sp = await searchParams;
  const scope: "mine" | "all" =
    me?.is_admin && sp.scope === "all" ? "all" : "mine";

  const [{ apps, error }, username] = await Promise.all([
    noAccess ? Promise.resolve({ apps: [], error: null }) : loadEndpoints(scope),
    currentUsername(),
  ]);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Serverless Inference" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Serverless Inference</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Deploy and scale GPU-powered inference endpoints.
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
                <h2 className="text-base font-medium">Endpoints</h2>
                <span className="text-xs text-muted-foreground">
                  {apps.length} {apps.length === 1 ? "endpoint" : "endpoints"}
                  {me?.is_admin && scope === "all" && " · all users"}
                </span>
              </div>
              <Button asChild size="sm">
                <Link href="/serverless/new">
                  <Plus className="h-4 w-4" />
                  New endpoint
                </Link>
              </Button>
            </div>
            <EndpointGrid apps={apps} viewerUsername={username} isAdmin={me?.is_admin ?? false} />
          </section>
        )}
      </div>
    </div>
  );
}
