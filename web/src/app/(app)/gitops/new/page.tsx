import { redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { RepoForm } from "./repo-form";

export default async function NewGitopsRepoPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");
  const username = await currentUsername();
  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "GitOps", href: "/gitops" }, { label: "Add repository" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Add repository</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Connect a git repo of YAML manifests. After saving, hit{" "}
            <span className="font-medium text-foreground">Sync now</span> to reconcile.
          </p>
        </div>
        <RepoForm />
      </div>
    </div>
  );
}
