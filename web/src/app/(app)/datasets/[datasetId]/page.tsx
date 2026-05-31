import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ConsoleTopbar } from "@/components/console/topbar";
import { gateway } from "@/lib/gateway";
import type { DatasetPreview, DatasetRecord, StorageRecord } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { UploadCard } from "./upload-card";
import { SyncCard } from "./sync-card";
import { DeleteButton } from "./delete-button";
import { DatasetTitle } from "./dataset-title";
import { ColumnsCard } from "./columns-card";
import { TransformCard } from "./transform-card";
import { RowBrowser } from "./row-browser";

function fmtBytes(n?: number | null): string {
  if (!n && n !== 0) return "—";
  if (n < 1024) return `${n} B`;
  const u = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(1)} ${u[i]}`;
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-sm">{value}</span>
    </div>
  );
}

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
  let s3Storages: StorageRecord[] = [];
  if (canTransform) {
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
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <Link href="/datasets" className="mb-4 inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground">
          <ArrowLeft className="h-4 w-4" />
          Back to datasets
        </Link>

        {loadError && (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            Couldn&apos;t load dataset: {loadError}
          </div>
        )}

        {dataset && (
          <div className="space-y-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <DatasetTitle id={dataset.id} name={dataset.name} />
                {dataset.description && (
                  <p className="mt-1 text-sm text-muted-foreground">{dataset.description}</p>
                )}
              </div>
              <DeleteButton id={dataset.id} name={dataset.name} />
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Metadata</CardTitle>
              </CardHeader>
              <CardContent className="divide-y divide-border/60">
                <Row label="Source" value={dataset.kind} />
                <Row label="Storage" value={dataset.storage_name ?? "—"} />
                {dataset.kind === "s3" && <Row label="S3 metadata URI" value={<span className="font-mono text-xs">{dataset.s3_metadata_uri ?? "—"}</span>} />}
                {dataset.kind === "label" && (
                  <Row
                    label="Labeling project"
                    value={
                      <span className="font-mono text-xs">
                        {dataset.label_base_url}/dashboard/projects/{dataset.label_project_id}
                      </span>
                    }
                  />
                )}
                <Row label="Format" value={dataset.format ? dataset.format.toUpperCase() : "—"} />
                <Row label="Rows" value={dataset.num_rows ?? "—"} />
                {dataset.audio_prefix && <Row label="Audio prefix" value={<span className="font-mono text-xs">{dataset.audio_prefix}</span>} />}
                <Row label="Size" value={fmtBytes(dataset.size_bytes)} />
                <Row
                  label="HuggingFace"
                  value={
                    dataset.hf_synced_at
                      ? `synced → ${dataset.hf_repo} (${new Date(dataset.hf_synced_at).toLocaleString()})`
                      : dataset.hf_repo
                        ? dataset.hf_repo
                        : "not synced"
                  }
                />
              </CardContent>
            </Card>

            <ColumnsCard
              datasetId={dataset.id}
              kind={dataset.kind}
              audioField={dataset.audio_field}
              transcriptionField={dataset.transcription_field}
              splitFields={dataset.split_fields}
            />

            {canTransform && (
              <TransformCard
                datasetId={dataset.id}
                kind={dataset.kind}
                hfRepo={dataset.hf_repo ?? null}
                s3Storages={s3Storages}
                initialStatus={dataset.transform_status ?? null}
                initialLog={dataset.transform_log ?? null}
              />
            )}

            {dataset.kind === "upload" && <UploadCard datasetId={dataset.id} hasFile={!!dataset.metadata_filename} />}

            {hasMetadata && preview && (
              <RowBrowser datasetId={dataset.id} initial={preview} />
            )}

            {dataset.kind === "upload" && (
              <SyncCard
                datasetId={dataset.id}
                canSync={!!dataset.metadata_filename}
                currentRepo={dataset.hf_repo}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
