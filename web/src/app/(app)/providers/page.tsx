import Link from "next/link";
import { redirect } from "next/navigation";
import { Inbox, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { gateway } from "@/lib/gateway";
import type { ProviderRecord } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { ProvidersList } from "./providers-list";

async function loadProviders(): Promise<{ items: ProviderRecord[]; error: string | null }> {
  try {
    const items = await gateway.listProviders();
    return { items, error: null };
  } catch (e) {
    return { items: [], error: e instanceof Error ? e.message : String(e) };
  }
}

export default async function ProvidersPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");

  const [{ items, error }, username] = await Promise.all([
    loadProviders(),
    currentUsername(),
  ]);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "GPU Providers" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">GPU Providers</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Register VMs and cloud accounts that benchmarks and serverless
            endpoints can target. For VMs we SSH in to detect GPUs via
            <span className="font-mono"> nvidia-smi</span>.
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
              <h2 className="text-base font-medium">Providers</h2>
              <span className="text-xs text-muted-foreground">{items.length} total</span>
            </div>
            <Button asChild size="sm">
              <Link href="/providers/new">
                <Plus className="h-4 w-4" />
                New provider
              </Link>
            </Button>
          </div>

          {items.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
              <Inbox className="h-6 w-6 text-muted-foreground/60" />
              <p className="text-sm text-muted-foreground">
                No providers yet. Click <span className="font-medium text-foreground">New provider</span> to register one.
              </p>
            </div>
          ) : (
            <ProvidersList items={items} />
          )}
        </section>
      </div>
    </div>
  );
}
