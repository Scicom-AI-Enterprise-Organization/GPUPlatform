import { ConsoleTopbar } from "@/components/console/topbar";
import { gateway } from "@/lib/gateway";
import type { DatasetPreview, DatasetRecord, StorageRecord } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { DatasetDetail } from "./dataset-detail";

export default async function DatasetDetailPage({
  params,
  searchParams,
}: {
  params: Promise<{ datasetId: string }>;
  searchParams: Promise<{ offset?: string; limit?: string; split?: string }>;
}) {
  const { datasetId } = await params;
  const sp = await searchParams;
  const username = await currentUsername();

  // Pagination state lives in the URL so pages are shareable + survive refresh.
  const offset = Math.max(0, Number.parseInt(sp.offset ?? "", 10) || 0);
  const limit = Math.min(200, Math.max(1, Number.parseInt(sp.limit ?? "", 10) || 20));
  const split = sp.split || undefined;

  let dataset: DatasetRecord | null = null;
  let loadError: string | null = null;
  try {
    dataset = await gateway.getDataset(datasetId);
  } catch (e) {
    loadError = e instanceof Error ? e.message : String(e);
  }

  const hasMetadata =
    !!dataset &&
    (dataset.kind === "hf"
      ? !!dataset.hf_repo
      : dataset.kind === "label"
        ? !!dataset.label_project_id
        : !!dataset.metadata_filename || !!dataset.s3_metadata_uri);

  let preview: DatasetPreview | null = null;
  if (dataset && hasMetadata) {
    try {
      preview = await gateway.getDatasetPreview(dataset.id, limit, offset, split);
    } catch (e) {
      preview = { audio_field: dataset.audio_field, transcription_field: dataset.transcription_field, rows: [], error: e instanceof Error ? e.message : String(e) };
    }
  }

  // S3 storages to offer as a transform target (HF audio-zip / label platform →
  // audio column).
  const canTransform = dataset?.kind === "hf" || dataset?.kind === "label";
  // Pack {audio, transcription} → NeuCodec + multipack ChiniDataset (TTS). Not
  // for label (transform to audio first) or an already-packed dataset.
  const canPack =
    dataset?.kind === "s3" || dataset?.kind === "upload" ||
    dataset?.kind === "hf" || dataset?.kind === "label";
  let s3Storages: StorageRecord[] = [];
  if (canTransform || canPack) {
    try {
      s3Storages = (await gateway.listStorage()).filter((s) => s.kind === "s3" && s.enabled);
    } catch {
      s3Storages = [];
    }
  }

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[
          { label: "Datasets", href: "/datasets" },
          { label: dataset?.name ?? datasetId },
        ]}
        username={username}
      />
      {loadError || !dataset ? (
        <div className="flex-1 px-6 py-8 lg:px-10">
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            Couldn&apos;t load dataset: {loadError ?? "not found"}
          </div>
        </div>
      ) : (
        <DatasetDetail
          dataset={dataset}
          preview={preview}
          s3Storages={s3Storages}
          hasMetadata={hasMetadata}
          canTransform={canTransform}
          canPack={canPack}
        />
      )}
    </div>
  );
}
