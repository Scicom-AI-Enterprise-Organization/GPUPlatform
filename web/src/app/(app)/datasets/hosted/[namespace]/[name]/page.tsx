import { notFound, redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { gateway } from "@/lib/gateway";
import type { CatalogRecord } from "@/lib/types";
import { CatalogDetail } from "@/components/catalog/catalog-detail";

export default async function HostedDatasetDetailPage({
  params,
  searchParams,
}: {
  params: Promise<{ namespace: string; name: string }>;
  searchParams: Promise<{ view?: string }>;
}) {
  const me = await getMe();
  if (!me) redirect("/login");
  if (!me.sections?.catalog) redirect("/datasets");

  const { namespace, name } = await params;
  const sp = await searchParams;
  let repo: CatalogRecord;
  try {
    repo = await gateway.lookupCatalogRepo("dataset", namespace, name);
  } catch {
    notFound();
  }

  // If this repo was published from an Autotrain dataset, they're "one" — send
  // the user to the dataset's page (which carries the HF card) instead.
  // (redirect() must be OUTSIDE the try — it throws NEXT_REDIRECT.)
  let linkedDatasetId: string | null = null;
  try {
    const datasets = await gateway.listDatasets(me.is_admin ? "all" : "mine");
    linkedDatasetId = datasets.find((d) => d.catalog_repo_id === repo.id)?.id ?? null;
  } catch {
    /* if the lookup fails, just show the hosted repo page */
  }
  if (linkedDatasetId) redirect(`/datasets/${linkedDatasetId}`);

  const username = await currentUsername();
  const gatewayUrl = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Datasets", href: "/datasets" }, { label: repo.full_id }]}
        username={username}
      />
      <CatalogDetail repo={repo} gatewayUrl={gatewayUrl} backHref="/datasets" initialView={sp.view} />
    </div>
  );
}
