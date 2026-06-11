"use client";

import { useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { DatasetPreview, DatasetRecord, StorageRecord } from "@/lib/types";
import { DatasetTitle } from "./dataset-title";
import { DeleteButton } from "./delete-button";
import { ColumnsCard } from "./columns-card";
import { TransformationCard } from "./transformation-card";
import { RowBrowser } from "./row-browser";
import { UploadCard } from "./upload-card";
import { SyncCard } from "./sync-card";
import { DatasetFilesCard } from "./files-card";

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

function Kpi({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-0.5 truncate text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-sm">{value}</span>
    </div>
  );
}

/**
 * Dataset detail shell — matches the benchmark / serverless layout: a tinted
 * header band (title + KPIs + line tabs) over a scrolling content area that
 * swaps per tab. The data tab (Rows) leads; field mapping, transformation and
 * the raw metadata live behind their own tabs instead of one long card stack.
 */
export function DatasetDetail({
  dataset,
  preview,
  s3Storages,
  hasMetadata,
  canTransform,
  canPack,
}: {
  dataset: DatasetRecord;
  preview: DatasetPreview | null;
  s3Storages: StorageRecord[];
  hasMetadata: boolean;
  canTransform: boolean;
  canPack: boolean;
}) {
  const showRows = hasMetadata && !!preview;
  const showTransform = canTransform || canPack;
  const isUpload = dataset.kind === "upload";
  // S3-backed datasets get a Files tab listing their objects (presigned downloads).
  // hf / label datasets have no S3 backing → no tab.
  const showFiles = !!dataset.storage_name && ["s3", "tts_packed", "upload"].includes(dataset.kind);

  const tabs = [
    showRows && { value: "rows", label: "Rows" },
    { value: "columns", label: "Columns" },
    showFiles && { value: "files", label: "Files" },
    showTransform && { value: "transform", label: "Transform" },
    { value: "details", label: "Details" },
  ].filter(Boolean) as { value: string; label: string }[];
  const valid = tabs.map((t) => t.value);
  const defaultTab = showRows ? "rows" : isUpload && !hasMetadata ? "details" : valid[0];

  // Top-level tab lives in `?view=` (the Transform card owns `?tab=` for its own
  // audio/pack sub-tabs — keep them separate). Update via history.replaceState so
  // switching tabs doesn't re-run the page's server fetch (dataset + preview).
  const searchParams = useSearchParams();
  const [tab, setTabState] = useState(() => {
    const v = searchParams.get("view");
    return v && valid.includes(v) ? v : defaultTab;
  });
  const setTab = (v: string) => {
    setTabState(v);
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    params.set("view", v);
    window.history.replaceState(null, "", `${window.location.pathname}?${params.toString()}`);
  };

  // The HF repo this dataset lives at, or — for a transformed dataset — the
  // original HF dataset it was derived from.
  const hfRepo = dataset.hf_repo || dataset.source_hf_repo || null;
  const hfUrl = hfRepo ? `https://huggingface.co/datasets/${hfRepo}` : null;
  const hfValue = hfRepo ? (
    <a href={hfUrl!} target="_blank" rel="noreferrer" title={hfRepo} className="text-primary hover:underline">
      {hfRepo}
    </a>
  ) : "—";

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="border-b border-border bg-sidebar/40 px-6 pt-4 lg:px-10">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <DatasetTitle id={dataset.id} name={dataset.name} />
              <span className="rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                {dataset.kind}
              </span>
            </div>
            {dataset.description && (
              <p className="mt-1 text-sm text-muted-foreground">{dataset.description}</p>
            )}
            <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
              <span className="font-mono">{dataset.id}</span>
              {dataset.storage_name && (
                <>
                  <span>·</span>
                  <a href="/storage" target="_blank" rel="noreferrer" className="text-primary hover:underline">
                    {dataset.storage_name}
                  </a>
                </>
              )}
            </div>
          </div>
          <DeleteButton id={dataset.id} name={dataset.name} />
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-5">
          <Kpi label="Source" value={dataset.kind} />
          <Kpi label="Format" value={dataset.format ? dataset.format.toUpperCase() : "—"} />
          <Kpi label="Rows" value={dataset.num_rows != null ? dataset.num_rows.toLocaleString() : "—"} />
          <Kpi label="Size" value={fmtBytes(dataset.size_bytes)} />
          <Kpi label="HuggingFace" value={hfValue} />
        </div>

        <Tabs value={tab} onValueChange={setTab} className="mt-4">
          <TabsList variant="line" className="bg-transparent">
            {tabs.map((t) => (
              <TabsTrigger key={t.value} value={t.value}>
                {t.label}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <Tabs value={tab} onValueChange={setTab} className="!block">
          {showRows && preview && (
            <TabsContent value="rows" className="!flex-none">
              <RowBrowser datasetId={dataset.id} initial={preview} speakerField={dataset.speaker_field} />
            </TabsContent>
          )}

          <TabsContent value="columns" className="!flex-none">
            <ColumnsCard
              datasetId={dataset.id}
              kind={dataset.kind}
              audioField={dataset.audio_field}
              transcriptionField={dataset.transcription_field}
              speakerField={dataset.speaker_field}
              splitFields={dataset.split_fields}
            />
          </TabsContent>

          {showFiles && (
            <TabsContent value="files" className="!flex-none">
              <DatasetFilesCard datasetId={dataset.id} split={searchParams.get("split")} />
            </TabsContent>
          )}

          {showTransform && (
            <TabsContent value="transform" className="!flex-none">
              <TransformationCard
                datasetId={dataset.id}
                kind={dataset.kind}
                hfRepo={dataset.hf_repo ?? null}
                s3Storages={s3Storages}
                initialStatus={dataset.transform_status ?? null}
                initialLog={dataset.transform_log ?? null}
              />
            </TabsContent>
          )}

          <TabsContent value="details" className="!flex-none space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Metadata</CardTitle>
              </CardHeader>
              <CardContent className="divide-y divide-border/60">
                <Row label="Source" value={dataset.kind} />
                <Row
                  label="Storage"
                  value={
                    dataset.storage_name ? (
                      <a href="/storage" target="_blank" rel="noreferrer" className="text-sm text-primary hover:underline">
                        {dataset.storage_name}
                      </a>
                    ) : (
                      "—"
                    )
                  }
                />
                {dataset.kind === "s3" && (
                  <Row label="S3 metadata URI" value={<span className="font-mono text-xs">{dataset.s3_metadata_uri ?? "—"}</span>} />
                )}
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
                {dataset.audio_prefix && (
                  <Row label="Audio prefix" value={<span className="font-mono text-xs">{dataset.audio_prefix}</span>} />
                )}
                <Row label="Size" value={fmtBytes(dataset.size_bytes)} />
                {dataset.source_dataset_id && (
                  <Row
                    label="Transformed from"
                    value={
                      <Link href={`/datasets/${dataset.source_dataset_id}`} className="text-sm text-primary hover:underline">
                        {dataset.source_name ?? dataset.source_dataset_id}
                        {dataset.source_hf_repo && (
                          <span className="ml-1 font-mono text-xs text-muted-foreground">· {dataset.source_hf_repo}</span>
                        )}
                      </Link>
                    }
                  />
                )}
                <Row
                  label="HuggingFace"
                  value={
                    hfRepo ? (
                      <a href={hfUrl!} target="_blank" rel="noreferrer" className="text-sm text-primary hover:underline">
                        {dataset.hf_synced_at && dataset.hf_repo
                          ? `synced → ${dataset.hf_repo} (${new Date(dataset.hf_synced_at).toLocaleString()})`
                          : dataset.hf_repo
                            ? dataset.hf_repo
                            : `from ${dataset.source_hf_repo}`}
                      </a>
                    ) : (
                      "not synced"
                    )
                  }
                />
              </CardContent>
            </Card>

            {isUpload && <UploadCard datasetId={dataset.id} hasFile={!!dataset.metadata_filename} />}
            {isUpload && (
              <SyncCard datasetId={dataset.id} canSync={!!dataset.metadata_filename} currentRepo={dataset.hf_repo} />
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
