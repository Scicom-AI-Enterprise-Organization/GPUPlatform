import { redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { gateway } from "@/lib/gateway";
import type { DatasetRecord, StorageRecord } from "@/lib/types";
import { MergeCard } from "./merge-card";

// Combine two or more kind=label/s3 datasets (each yielding {audio, transcription}
// rows) into ONE new audio dataset — the "import / materialize several, merge
// later" flow. Preselect rows by passing ?ids=a,b,c (from the datasets list's
// multi-select Merge action).
export default async function MergeDatasetsPage({
  searchParams,
}: {
  searchParams: Promise<{ ids?: string }>;
}) {
  const me = await getMe();
  if (!me) redirect("/login");
  if (!me.sections?.datasets) redirect("/datasets");

  let datasets: DatasetRecord[] = [];
  let storages: StorageRecord[] = [];
  try {
    [datasets, storages] = await Promise.all([gateway.listDatasets("mine"), gateway.listStorage()]);
  } catch {
    /* render with whatever loaded; the form surfaces gateway errors on submit */
  }
  // label + s3 are the mergeable kinds (both yield {audio, transcription} rows).
  const mergeableDatasets = datasets.filter((d) => d.kind === "label" || d.kind === "s3");
  const s3Storages = storages.filter((s) => s.kind === "s3");
  const sp = await searchParams;
  const initialSelected = (sp.ids ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter((id) => mergeableDatasets.some((d) => d.id === id));
  const username = await currentUsername();

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Datasets", href: "/datasets" }, { label: "Merge datasets" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Merge datasets</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Concatenate two or more <span className="font-mono text-xs">label</span> or{" "}
            <span className="font-mono text-xs">s3</span> datasets into one combined{" "}
            <span className="font-mono text-xs">{`{audio, transcription}`}</span> dataset. Each
            source&apos;s clips are downloaded, paired with their transcription, and written to
            HuggingFace or S3 as a single dataset.
          </p>
        </div>
        <MergeCard
          mergeableDatasets={mergeableDatasets}
          s3Storages={s3Storages}
          initialSelected={initialSelected}
        />
      </div>
    </div>
  );
}
