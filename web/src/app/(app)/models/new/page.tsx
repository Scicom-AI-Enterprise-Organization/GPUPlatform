import { redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { gateway } from "@/lib/gateway";
import type { StorageRecord } from "@/lib/types";
import { CatalogForm } from "@/components/catalog/catalog-form";

export default async function NewModelPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (!me.sections?.catalog) redirect("/models");

  let storages: StorageRecord[] = [];
  try {
    storages = await gateway.listStorage();
  } catch {
    storages = [];
  }
  // Only s3 / local / sftp storages can host repos.
  const hostable = storages.filter(
    (s) => s.enabled && (s.kind === "s3" || s.kind === "local" || s.kind === "sftp"),
  );
  const username = await currentUsername();

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Models", href: "/models" }, { label: "New model" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">New model</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Register a model repo on one of your storage backends. Push files with{" "}
            <span className="font-mono text-xs">hf upload</span> /{" "}
            <span className="font-mono text-xs">push_to_hub</span>, or register a prefix
            that already holds HuggingFace-layout files.
          </p>
        </div>
        <CatalogForm storages={hostable} defaultNamespace={username} repoType="model" />
      </div>
    </div>
  );
}
