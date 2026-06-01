"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TransformCard } from "./transform-card";
import { TtsPackCard } from "./tts-pack-card";
import type { DatasetKind, StorageRecord } from "@/lib/types";

// One "Transformation" card with tabs: extract a real audio column (HF archive /
// label export → audio) and/or pack for TTS (NeuCodec encode + multipack). The
// audio-column tab only applies to hf/label sources; s3/upload datasets already
// have audio, so they just get the pack tab.
export function TransformationCard({
  datasetId,
  kind,
  hfRepo,
  s3Storages,
  initialStatus,
  initialLog,
}: {
  datasetId: string;
  kind: DatasetKind;
  hfRepo: string | null;
  s3Storages: StorageRecord[];
  initialStatus: string | null;
  initialLog: string | null;
}) {
  const canTransform = kind === "hf" || kind === "label";
  const audioLabel = kind === "label" ? "Export labels → audio" : "Extract audio column";
  const [tab, setTab] = useState(canTransform ? "audio" : "pack");

  const packTab = (
    <TtsPackCard
      datasetId={datasetId}
      s3Storages={s3Storages}
      initialStatus={initialStatus}
      initialLog={initialLog}
      bare
    />
  );

  return (
    <Card>
      <CardHeader className="flex flex-col gap-0.5">
        <CardTitle className="text-base">Transformation</CardTitle>
        <span className="text-xs text-muted-foreground">
          Convert this dataset for training — extract a real audio column, or NeuCodec-encode + multipack for TTS.
        </span>
      </CardHeader>
      <CardContent>
        {canTransform ? (
          <Tabs value={tab} onValueChange={setTab}>
            <TabsList className="mb-3">
              <TabsTrigger value="audio">{audioLabel}</TabsTrigger>
              <TabsTrigger value="pack">Pack for TTS (NeuCodec)</TabsTrigger>
            </TabsList>
            <TabsContent value="audio">
              <TransformCard
                datasetId={datasetId}
                kind={kind}
                hfRepo={hfRepo}
                s3Storages={s3Storages}
                initialStatus={initialStatus}
                initialLog={initialLog}
                bare
              />
            </TabsContent>
            <TabsContent value="pack">{packTab}</TabsContent>
          </Tabs>
        ) : (
          packTab
        )}
      </CardContent>
    </Card>
  );
}
