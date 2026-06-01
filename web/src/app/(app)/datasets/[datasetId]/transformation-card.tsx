"use client";

import { TransformCard } from "./transform-card";
import { TtsPackCard } from "./tts-pack-card";
import type { DatasetKind, StorageRecord } from "@/lib/types";

// One transformation per source kind:
//  • hf / label → extract a real audio column (the archive / label export has no
//    playable audio column yet). Pack-for-TTS isn't offered here — extract first,
//    then pack the resulting audio dataset.
//  • s3 / upload → already have audio, so NeuCodec-encode + multipack for TTS.
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

  if (canTransform) {
    return (
      <TransformCard
        datasetId={datasetId}
        kind={kind}
        hfRepo={hfRepo}
        s3Storages={s3Storages}
        initialStatus={initialStatus}
        initialLog={initialLog}
      />
    );
  }

  return (
    <TtsPackCard
      datasetId={datasetId}
      s3Storages={s3Storages}
      initialStatus={initialStatus}
      initialLog={initialLog}
    />
  );
}
