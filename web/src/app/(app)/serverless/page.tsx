import Link from "next/link";
import { Boxes, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { gateway } from "@/lib/gateway";
import type { AppRecord } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { EndpointGrid } from "./endpoint-grid";

async function loadEndpoints(): Promise<{ apps: AppRecord[]; error: string | null }> {
  try {
    const apps = await gateway.listApps();
    return { apps, error: null };
  } catch (e) {
    return { apps: [], error: e instanceof Error ? e.message : String(e) };
  }
}

export default async function ServerlessPage() {
  const me = await getMe();
  const noAccess = me?.role === "user";
  const [{ apps, error }, username] = await Promise.all([
    noAccess ? Promise.resolve({ apps: [], error: null }) : loadEndpoints(),
    currentUsername(),
  ]);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Serverless Inference" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6 flex items-start gap-4 rounded-xl border border-border bg-gradient-to-br from-sky-500/10 via-background to-blue-500/5 p-6">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-sky-500 to-blue-500 text-white shadow-lg shadow-blue-500/30">
            <Boxes className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Serverless Inference</h1>
            <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
              Deploy and scale GPU-powered inference endpoints. Pay per second of compute.
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
                <h2 className="text-base font-medium">Endpoints</h2>
                <span className="text-xs text-muted-foreground">
                  {apps.length} {apps.length === 1 ? "endpoint" : "endpoints"}
                </span>
              </div>
              <Button asChild size="sm">
                <Link href="/serverless/new">
                  <Plus className="h-4 w-4" />
                  New endpoint
                </Link>
              </Button>
            </div>
            <EndpointGrid apps={apps} />
          </section>
        )}
      </div>
    </div>
  );
}
