import { notFound, redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { gateway, GatewayError } from "@/lib/gateway";
import type { GitopsRepo, GitopsResource } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { RepoDetail } from "./repo-detail";

export default async function GitopsRepoPage({ params }: { params: Promise<{ id: string }> }) {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");

  const { id } = await params;
  let repo: GitopsRepo;
  let resources: GitopsResource[] = [];
  try {
    repo = await gateway.getGitopsRepo(id);
    resources = await gateway.listGitopsResources(id);
  } catch (e) {
    if (e instanceof GatewayError && e.status === 404) notFound();
    throw e;
  }
  const username = await currentUsername();

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "GitOps", href: "/gitops" }, { label: repo.name }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <RepoDetail initialRepo={repo} initialResources={resources} />
      </div>
    </div>
  );
}
