import { redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { gateway } from "@/lib/gateway";
import type { StorageRecord } from "@/lib/types";
import { CatalogForm } from "@/components/catalog/catalog-form";

export default async function NewHostedDatasetPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (!me.sections?.catalog) redirect("/datasets");

  let storages: StorageRecord[] = [];
  try {
    storages = await gateway.listStorage();
  } catch {
    storages = [];
  }
  const hostable = storages.filter(
    (s) => s.enabled && (s.kind === "s3" || s.kind === "local" || s.kind === "sftp"),
  );
  const username = await currentUsername();

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Datasets", href: "/datasets" }, { label: "New hosted dataset" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">New hosted dataset</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Register a HuggingFace-compatible dataset repo on one of your storage backends.
            Push files with <span className="font-mono text-xs">hf upload … --repo-type dataset</span>{" "}
            or register a prefix that already holds HuggingFace-layout files.
          </p>
        </div>
        <CatalogForm storages={hostable} defaultNamespace={username} repoType="dataset" />
      </div>
    </div>
  );
}
