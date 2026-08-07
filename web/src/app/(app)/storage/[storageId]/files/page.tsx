import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { Button } from "@/components/ui/button";
import { ConsoleTopbar } from "@/components/console/topbar";
import { gateway } from "@/lib/gateway";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { FileBrowser } from "./file-browser";

// Kinds with something to browse — mirrors storage_api.BROWSABLE_KINDS.
// huggingface/sftp have no viewer (a HF repo browses on the Hub; sftp would need
// its own connection per listing).
const BROWSABLE = ["s3", "local"];

export default async function StorageFilesPage({
  params,
  searchParams,
}: {
  params: Promise<{ storageId: string }>;
  searchParams: Promise<{ path?: string; file?: string }>;
}) {
  const me = await getMe();
  if (!me) redirect("/login");
  const { storageId } = await params;
  const { path, file } = await searchParams;
  const username = await currentUsername();

  // No single-storage GET route — the list is one small query and already
  // carries everything the header needs.
  const items = await gateway.listStorage().catch(() => []);
  const storage = items.find((s) => s.id === storageId);
  if (!storage) notFound();

  const crumbs = [
    { label: "Storage", href: "/storage" },
    { label: storage.name, href: "/storage" },
    { label: "Files" },
  ];
  const location =
    storage.kind === "local"
      ? storage.path
      : `s3://${storage.bucket}${storage.prefix ? `/${storage.prefix.replace(/^\/+|\/+$/g, "")}` : ""}`;

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={crumbs} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">{storage.name}</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Read-only viewer over <span className="font-mono">{location}</span> ·{" "}
            <span className="font-mono">{storage.id}</span>
            {storage.kind === "local" && " · on the gateway host"}
          </p>
        </div>

        {me.role !== "admin" ? (
          <div className="rounded-md border border-border px-4 py-10 text-center text-sm text-muted-foreground">
            Browsing a storage backend&apos;s raw contents is admin-only.
          </div>
        ) : !BROWSABLE.includes(storage.kind) ? (
          <div className="rounded-md border border-border px-4 py-10 text-center text-sm text-muted-foreground">
            <p>
              The file viewer supports <span className="font-mono">s3</span> and{" "}
              <span className="font-mono">local</span> storage — this one is{" "}
              <span className="font-mono">{storage.kind}</span>.
            </p>
            <Button asChild variant="outline" size="sm" className="mt-4">
              <Link href="/storage">Back to storage</Link>
            </Button>
          </div>
        ) : (
          <FileBrowser storage={storage} initialPath={path ?? ""} initialFile={file ?? ""} />
        )}
      </div>
    </div>
  );
}
