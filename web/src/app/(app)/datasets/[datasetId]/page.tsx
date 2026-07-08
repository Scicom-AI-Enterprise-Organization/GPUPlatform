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
  searchParams: Promise<{ offset?: string; limit?: string; split?: string; speaker?: string; view?: string }>;
}) {
  const { datasetId } = await params;
  const sp = await searchParams;
  const username = await currentUsername();

  // Pagination state lives in the URL so pages are shareable + survive refresh.
  const offset = Math.max(0, Number.parseInt(sp.offset ?? "", 10) || 0);
  const limit = Math.min(200, Math.max(1, Number.parseInt(sp.limit ?? "", 10) || 20));
  const split = sp.split || undefined;
  const speaker = sp.speaker || undefined;

  let dataset: DatasetRecord | null = null;
  let loadError: string | null = null;
  try {
    dataset = await gateway.getDataset(datasetId);
  } catch (e) {
    loadError = e instanceof Error ? e.message : String(e);
  }

  const hasMetadata =
    !!dataset &&
    (dataset.kind === "hf" || dataset.kind === "llm"
      ? !!dataset.hf_repo
      : dataset.kind === "label"
        ? !!dataset.label_project_id
        : !!dataset.metadata_filename || !!dataset.s3_metadata_uri);

  // Don't block the page render on the row preview. For a big S3/parquet upload the
  // gateway has to download the whole metadata file (tens of MB → 10s+), which used
  // to hang the entire route and blow the 30s SSR timeout. Instead we hand the
  // RowBrowser a lightweight seed (the URL-driven page window) and it fetches page 1
  // client-side behind its own loading spinner — off the render critical path, and
  // the client fetch has no 30s abort so it just waits instead of erroring.
  const preview: DatasetPreview | null =
    dataset && hasMetadata
      ? {
          audio_field: dataset.audio_field,
          transcription_field: dataset.transcription_field,
          rows: [],
          offset,
          limit,
          split: split ?? null,
          speaker: speaker ?? null,
        }
      : null;

  // S3 storages to offer as a transform target (HF audio-zip / label platform →
  // audio column).
  const canTransform = dataset?.kind === "hf" || dataset?.kind === "label";
  // Pack {audio, transcription} → NeuCodec + multipack ChiniDataset (TTS). Only
  // for datasets that already have a real audio column (s3 / upload). hf / label
  // sources must extract an audio column first, then pack the resulting dataset.
  // A chat upload (kind=upload with a messages column) has no audio → not TTS-packable.
  const canPack =
    (dataset?.kind === "s3" || dataset?.kind === "upload") && !dataset?.messages_field;
  // Chat → multipack: a chat dataset's messages column → a ChiniDataset
  // (kind=llm_packed) for LLM finetuning. In-process (CPU tokenization, no GPU).
  // A kind=llm dataset, OR a kind=hf dataset with a messages column mapped (a chat
  // dataset registered as plain hf) — the latter is what surfaces a chat preview.
  // A preference (DPO) dataset is identified by its mapped `rejected_field` (Columns
  // card → Preference/DPO mode). We no longer sniff the first preview row for
  // chosen/rejected columns, since the preview isn't fetched server-side anymore.
  const canPackLlm =
    dataset?.kind === "llm" ||
    ((dataset?.kind === "hf" || dataset?.kind === "upload") &&
      (!!dataset?.messages_field || !!dataset?.rejected_field));
  let s3Storages: StorageRecord[] = [];
  if (canTransform || canPack || canPackLlm) {
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
          canPackLlm={canPackLlm}
          initialView={sp.view}
          initialSplit={split ?? null}
        />
      )}
    </div>
  );
}
