import Link from "next/link";
import { redirect } from "next/navigation";
import { Inbox, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { gateway } from "@/lib/gateway";
import type { ProxyEndpoint } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { ProxyList } from "./proxy-list";

async function load(admin: boolean): Promise<{ items: ProxyEndpoint[]; error: string | null }> {
  try {
    return { items: admin ? await gateway.listProxies() : await gateway.listPublicProxies(), error: null };
  } catch (e) {
    return { items: [], error: e instanceof Error ? e.message : String(e) };
  }
}

export default async function ProxyPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  // Admins manage all proxies; everyone else gets a read-only view of the PUBLIC
  // ones (proxies are admin-managed, so non-admins can view + use, never edit).
  const isAdmin = me.role === "admin";

  const [{ items, error }, username] = await Promise.all([load(isAdmin), currentUsername()]);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "LLM API Proxy" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">LLM API Proxy</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            One stable OpenAI-compatible endpoint + model name that routes to multiple backends
            behind the scenes — priority + health-aware failover, a per-endpoint queue, and
            auto-cancel on client disconnect. Your team points their client at{" "}
            <span className="font-mono">/proxy/&lt;name&gt;/v1</span> and never changes anything.
            {!isAdmin && " Showing public endpoints — view and use them; only admins can edit."}
          </p>
        </div>

        {error && (
          <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            Couldn&apos;t reach the gateway: {error}
          </div>
        )}

        <section>
          <div className="mb-3 flex items-center justify-between border-b border-border pb-2">
            <div className="flex items-baseline gap-3">
              <h2 className="text-base font-medium">Endpoints</h2>
              <span className="text-xs text-muted-foreground">{items.length} total</span>
            </div>
            {isAdmin && (
              <Button asChild size="sm">
                <Link href="/proxy/new"><Plus className="h-4 w-4" /> New endpoint</Link>
              </Button>
            )}
          </div>

          {items.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
              <Inbox className="h-6 w-6 text-muted-foreground/60" />
              <p className="text-sm text-muted-foreground">
                {isAdmin ? (
                  <>No proxy endpoints yet. Click <span className="font-medium text-foreground">New endpoint</span> to create one.</>
                ) : (
                  "No public proxy endpoints yet."
                )}
              </p>
            </div>
          ) : (
            <ProxyList items={items} readOnly={!isAdmin} />
          )}
        </section>
      </div>
    </div>
  );
}
