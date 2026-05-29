import Link from "next/link";
import { redirect } from "next/navigation";
import { Inbox, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { gateway } from "@/lib/gateway";
import type { StorageRecord } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { StorageList } from "./storage-list";

async function loadStorage(): Promise<{ items: StorageRecord[]; error: string | null }> {
  try {
    const items = await gateway.listStorage();
    return { items, error: null };
  } catch (e) {
    return { items: [], error: e instanceof Error ? e.message : String(e) };
  }
}

export default async function StoragePage() {
  const me = await getMe();
  if (!me) redirect("/login");
  const canWrite = me.role === "admin";

  const [{ items, error }, username] = await Promise.all([
    loadStorage(),
    currentUsername(),
  ]);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Storage" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Storage</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Storage backends the platform writes to — S3 (or S3-compatible: R2,
            MinIO) buckets and HuggingFace token holders. Datasets, benchmark
            logs, and serverless inference logs are persisted here. Credentials
            are encrypted at rest; leave them blank to fall back to the
            gateway&apos;s <span className="font-mono">AWS_*</span> /{" "}
            <span className="font-mono">HF_TOKEN</span> env.
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
              <h2 className="text-base font-medium">Configured storage</h2>
              <span className="text-xs text-muted-foreground">{items.length} total</span>
            </div>
            {canWrite && (
              <Button asChild size="sm">
                <Link href="/storage/new">
                  <Plus className="h-4 w-4" />
                  New storage
                </Link>
              </Button>
            )}
          </div>

          {items.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
              <Inbox className="h-6 w-6 text-muted-foreground/60" />
              <p className="text-sm text-muted-foreground">
                {canWrite ? (
                  <>
                    No storage yet. Click <span className="font-medium text-foreground">New storage</span> to add an S3 bucket or HuggingFace token.
                  </>
                ) : (
                  <>No storage configured yet. Ask an admin to add one.</>
                )}
              </p>
            </div>
          ) : (
            <StorageList items={items} canWrite={canWrite} />
          )}
        </section>
      </div>
    </div>
  );
}
