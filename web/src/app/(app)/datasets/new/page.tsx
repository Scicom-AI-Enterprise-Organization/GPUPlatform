import { redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { gateway } from "@/lib/gateway";
import type { StorageRecord } from "@/lib/types";
import { DatasetForm } from "./dataset-form";

export default async function NewDatasetPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (!me.sections?.datasets) redirect("/datasets");

  let storages: StorageRecord[] = [];
  try {
    storages = await gateway.listStorage();
  } catch {
    storages = [];
  }
  const username = await currentUsername();

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Datasets", href: "/datasets" }, { label: "Register dataset" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Register dataset</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Point at a metadata source of <span className="font-mono text-xs">{`{audio, transcription}`}</span>{" "}
            rows. Upload a file to an S3 storage, reference one already in S3, or
            link an existing HuggingFace dataset.
          </p>
        </div>
        <DatasetForm storages={storages} />
      </div>
    </div>
  );
}
