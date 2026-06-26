"use client";

import { useState } from "react";
import { LlmPackCard } from "./llm-pack-card";
import { TransformCard } from "./transform-card";
import { TtsPackCard } from "./tts-pack-card";
import type { DatasetKind, StorageRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

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
  initialSplit,
}: {
  datasetId: string;
  kind: DatasetKind;
  hfRepo: string | null;
  messagesField?: string | null;
  s3Storages: StorageRecord[];
  initialStatus: string | null;
  initialLog: string | null;
  initialSplit?: string | null;
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
        initialSplit={initialSplit}
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
    <AudioPackPicker
      datasetId={datasetId}
      s3Storages={s3Storages}
      initialStatus={initialStatus}
      initialLog={initialLog}
    />
  );
}

// s3 / upload audio dataset: pick the codec/target — Qwen3+NeuCodec (tts_packed)
// or OmniVoice/Higgs (omnivoice_packed). Both reuse TtsPackCard (variant prop).
function AudioPackPicker({
  datasetId,
  s3Storages,
  initialStatus,
  initialLog,
}: {
  datasetId: string;
  s3Storages: StorageRecord[];
  initialStatus: string | null;
  initialLog: string | null;
}) {
  const [variant, setVariant] = useState<"tts" | "omnivoice">("tts");
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Pack for</span>
        <div className="flex items-center gap-0.5 rounded-md border border-border p-0.5">
          {(["tts", "omnivoice"] as const).map((v) => (
            <button
              key={v}
              type="button"
              onClick={() => setVariant(v)}
              className={cn(
                "rounded px-2.5 py-1 text-xs font-medium transition-colors",
                variant === v ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              {v === "tts" ? "TTS (NeuCodec)" : "OmniVoice (Higgs)"}
            </button>
          ))}
        </div>
      </div>
      {/* key forces a fresh card (state/poll) when switching codecs */}
      <TtsPackCard
        key={variant}
        datasetId={datasetId}
        s3Storages={s3Storages}
        initialStatus={initialStatus}
        initialLog={initialLog}
        variant={variant}
      />
    </div>
  );
}
