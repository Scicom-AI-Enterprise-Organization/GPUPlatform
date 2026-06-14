import { notFound, redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { gateway } from "@/lib/gateway";
import type { CatalogRecord } from "@/lib/types";
import { CatalogDetail } from "@/components/catalog/catalog-detail";

export default async function ModelDetailPage({
  params,
}: {
  params: Promise<{ repoId: string }>;
}) {
  const me = await getMe();
  if (!me) redirect("/login");
  if (!me.sections?.catalog) redirect("/models");

  const { repoId } = await params;
  let repo: CatalogRecord;
  try {
    repo = await gateway.getCatalogRepo(repoId);
  } catch {
    notFound();
  }
  // /models is models-only — dataset repos live under /datasets/hosted.
  if (repo.repo_type === "dataset") redirect(`/datasets/hosted/${repo.id}`);
  const username = await currentUsername();
  const gatewayUrl = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Models", href: "/models" }, { label: repo.full_id }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <CatalogDetail repo={repo} gatewayUrl={gatewayUrl} backHref="/models" />
      </div>
    </div>
  );
}
