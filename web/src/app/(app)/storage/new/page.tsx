import { redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { StorageForm } from "./storage-form";

export default async function NewStoragePage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/storage");
  const username = await currentUsername();
  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Storage", href: "/storage" }, { label: "New storage" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">New storage</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Add a storage backend. Pick an S3 (or S3-compatible) bucket or a
            HuggingFace token holder. Credentials are encrypted at rest; leave
            them blank to fall back to the gateway&apos;s env.
          </p>
        </div>
        <StorageForm />
      </div>
    </div>
  );
}
