import { notFound, redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { gateway } from "@/lib/gateway";
import type { CatalogRecord } from "@/lib/types";
import { CatalogDetail } from "@/components/catalog/catalog-detail";

export default async function ModelDetailPage({
  params,
  searchParams,
}: {
  params: Promise<{ namespace: string; name: string }>;
  searchParams: Promise<{ view?: string }>;
}) {
  const me = await getMe();
  if (!me) redirect("/login");
  if (!me.sections?.catalog) redirect("/models");

  const { namespace, name } = await params;
  const sp = await searchParams;
  let repo: CatalogRecord;
  try {
    repo = await gateway.lookupCatalogRepo("model", namespace, name);
  } catch {
    notFound();
  }
  const username = await currentUsername();
  const gatewayUrl = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Models", href: "/models" }, { label: repo.full_id }]}
        username={username}
      />
      <CatalogDetail repo={repo} gatewayUrl={gatewayUrl} backHref="/models" initialView={sp.view} />
    </div>
  );
}
