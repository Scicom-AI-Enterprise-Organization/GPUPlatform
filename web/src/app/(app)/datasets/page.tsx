import Link from "next/link";
import { Inbox, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { ScopeToggle } from "@/components/scope-toggle";
import { gateway } from "@/lib/gateway";
import type { CatalogRecord, DatasetRecord } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { DatasetsList } from "./datasets-list";

async function loadDatasets(
  scope: "mine" | "all",
): Promise<{ items: DatasetRecord[]; error: string | null }> {
  try {
    const items = await gateway.listDatasets(scope);
    return { items, error: null };
  } catch (e) {
    return { items: [], error: e instanceof Error ? e.message : String(e) };
  }
}

async function loadHosted(scope: "mine" | "all"): Promise<CatalogRecord[]> {
  try {
    return await gateway.listCatalog(scope, "dataset");
  } catch {
    return [];
  }
}

/** Adapt a HF-mirror dataset repo into a DatasetRecord so it lives in the one
 * Datasets list (kind="hosted" → card links to /datasets/hosted/<ns>/<name>). */
function hostedToDataset(r: CatalogRecord): DatasetRecord {
  return {
    id: r.id,
    name: r.full_id, // "ns/name" — the card splits this for the hosted detail URL
    kind: "hosted",
    storage_id: r.storage_id ?? null,
    storage_name: r.storage_name ?? null,
    size_bytes: r.size_bytes ?? null,
    num_rows: null, // it's files, not rows — don't show a misleading rows count
    audio_field: "",
    transcription_field: "",
    description: r.description ?? null,
    catalog_repo_id: r.id,
    created_at: r.created_at,
    updated_at: r.updated_at,
    created_by: r.created_by,
  };
}

export default async function DatasetsPage({
  searchParams,
}: {
  searchParams: Promise<{ scope?: string }>;
}) {
  const me = await getMe();
  const noAccess = !me?.sections?.datasets;
  const sp = await searchParams;
  const scope: "mine" | "all" =
    me?.is_admin && sp.scope === "all" ? "all" : "mine";

  const hasCatalog = !!me?.sections?.catalog;
  const [{ items: autotrain, error }, hostedAll, username] = await Promise.all([
    noAccess ? Promise.resolve({ items: [], error: null }) : loadDatasets(scope),
    !noAccess && hasCatalog ? loadHosted(scope) : Promise.resolve<CatalogRecord[]>([]),
    currentUsername(),
  ]);
  // Merge HF-mirror dataset repos into the one list, EXCLUDING those already
  // shown as a published Autotrain dataset (dedup on catalog_repo_id).
  const linkedRepoIds = new Set(autotrain.map((d) => d.catalog_repo_id).filter(Boolean));
  const standaloneHosted = hostedAll.filter((h) => !linkedRepoIds.has(h.id)).map(hostedToDataset);
  const items = [...autotrain, ...standaloneHosted];

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Datasets" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Datasets</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Register <span className="font-mono text-xs">{`{audio, transcription}`}</span>{" "}
              datasets for Autotrain — upload metadata to storage, preview rows with inline audio, sync to HuggingFace.
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
                <h2 className="text-base font-medium">Datasets</h2>
                <span className="text-xs text-muted-foreground">
                  {items.length} {items.length === 1 ? "dataset" : "datasets"}
                  {me?.is_admin && scope === "all" && " · all users"}
                </span>
              </div>
              <Button asChild size="sm">
                <Link href="/datasets/new">
                  <Plus className="h-4 w-4" />
                  Register dataset
                </Link>
              </Button>
            </div>

            {items.length === 0 ? (
              <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
                <Inbox className="h-6 w-6 text-muted-foreground/60" />
                <p className="text-sm text-muted-foreground">
                  No datasets yet. Click{" "}
                  <span className="font-medium text-foreground">Register dataset</span>{" "}
                  to add one.
                </p>
              </div>
            ) : (
              <DatasetsList items={items} />
            )}
          </section>
        )}
      </div>
    </div>
  );
}
