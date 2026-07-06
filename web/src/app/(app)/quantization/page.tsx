import Link from "next/link";
import { Inbox, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { gateway } from "@/lib/gateway";
import type { PageResponse, QuantizationJobRecord } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { ScopeToggle } from "@/components/scope-toggle";
import { QuantizationList } from "./quantization-list";

async function loadJobs(
  scope: "mine" | "all",
): Promise<{ page: PageResponse<QuantizationJobRecord>; error: string | null }> {
  try {
    return {
      page: await gateway.listQuantizationJobsPage({ scope, limit: 12, offset: 0 }),
      error: null,
    };
  } catch (e) {
    return { page: { total: 0, items: [] }, error: e instanceof Error ? e.message : String(e) };
  }
}

export default async function QuantizationPage({
  searchParams,
}: {
  searchParams: Promise<{ scope?: string }>;
}) {
  const me = await getMe();
  const sections = me?.sections as Record<string, boolean> | undefined;
  const noAccess = me ? !(me.is_admin || sections?.quantization) : false;
  const sp = await searchParams;
  const scope: "mine" | "all" = me?.is_admin && sp.scope === "all" ? "all" : "mine";

  const [{ page, error }, username] = await Promise.all([
    noAccess ? Promise.resolve({ page: { total: 0, items: [] }, error: null }) : loadJobs(scope),
    currentUsername(),
  ]);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Quantization" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Quantization</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Compress an LLM with llm-compressor — pull a model from Hugging Face, quantize
              (data-free or calibrated on a dataset), and push the compressed model back.
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
                <h2 className="text-base font-medium">Quantization jobs</h2>
                <span className="text-xs text-muted-foreground">
                  {page.total} {page.total === 1 ? "job" : "jobs"}
                  {me?.is_admin && scope === "all" && " · all users"}
                </span>
              </div>
              <Button asChild size="sm">
                <Link href="/quantization/new">
                  <Plus className="h-4 w-4" />
                  New job
                </Link>
              </Button>
            </div>

            {page.total === 0 ? (
              <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
                <Inbox className="h-6 w-6 text-muted-foreground/60" />
                <p className="text-sm text-muted-foreground">
                  No quantization jobs yet. Click{" "}
                  <span className="font-medium text-foreground">New job</span> to start one.
                </p>
              </div>
            ) : (
              <QuantizationList
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
