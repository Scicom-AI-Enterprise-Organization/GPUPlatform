import { redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { gateway } from "@/lib/gateway";
import type { DatasetRecord, StorageRecord } from "@/lib/types";
import { MergeCard } from "./merge-card";

// Combine two or more kind=label datasets (each pulling {audio, transcription}
// from a labeling-platform project) into ONE new audio dataset — the "import 2
// projects, merge later" flow. Import each project on /datasets/new first.
export default async function MergeDatasetsPage() {
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
  const labelDatasets = datasets.filter((d) => d.kind === "label");
  const s3Storages = storages.filter((s) => s.kind === "s3");
  const username = await currentUsername();

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Datasets", href: "/datasets" }, { label: "Merge label datasets" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Merge label datasets</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Concatenate two or more <span className="font-mono text-xs">label</span> datasets into one
            combined <span className="font-mono text-xs">{`{audio, transcription}`}</span> dataset.
            Each project&apos;s clips are downloaded, paired with their transcription, and written to
            HuggingFace or S3 as a single dataset. Don&apos;t see a project? Import it first on{" "}
            <span className="font-medium text-foreground">New dataset → Labeling platform</span>.
          </p>
        </div>
        <MergeCard labelDatasets={labelDatasets} s3Storages={s3Storages} />
      </div>
    </div>
  );
}
