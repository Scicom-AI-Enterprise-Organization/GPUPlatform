"use client";

import { LlmPackCard } from "./llm-pack-card";
import { TransformCard } from "./transform-card";
import { TtsPackCard } from "./tts-pack-card";
import type { DatasetKind, StorageRecord } from "@/lib/types";

// One transformation per source kind:
//  • hf / label → extract a real audio column (the archive / label export has no
//    playable audio column yet). Pack-for-TTS isn't offered here — extract first,
//    then pack the resulting audio dataset.
//  • llm → tokenize the messages column + bin-pack into a ChiniDataset (llm_packed)
//    for LLM finetuning. Runs in-process (CPU tokenization, no GPU).
//  • s3 / upload → already have audio, so NeuCodec-encode + multipack for TTS.
export function TransformationCard({
  datasetId,
  kind,
  hfRepo,
  messagesField,
  s3Storages,
  initialStatus,
  initialLog,
}: {
  datasetId: string;
  kind: DatasetKind;
  hfRepo: string | null;
  messagesField?: string | null;
  s3Storages: StorageRecord[];
  initialStatus: string | null;
  initialLog: string | null;
}) {
  // A chat dataset = kind=llm, OR a kind=hf dataset with a messages column mapped
  // (a chat dataset registered as plain hf — that's what shows a chat preview).
  // Prefer the LLM pack over audio extraction in the ambiguous hf case.
  const isChat = kind === "llm" || (kind === "hf" && !!messagesField);
  if (isChat) {
    return (
      <LlmPackCard
        datasetId={datasetId}
        messagesField={messagesField ?? "messages"}
        s3Storages={s3Storages}
        initialStatus={initialStatus}
        initialLog={initialLog}
      />
    );
  }

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
