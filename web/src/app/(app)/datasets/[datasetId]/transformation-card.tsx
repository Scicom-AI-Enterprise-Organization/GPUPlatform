"use client";

import { useState } from "react";
import { LlmPackCard } from "./llm-pack-card";
import { NormalizeCard } from "./normalize-card";
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
  rejectedField,
  speakerField,
  s3Storages,
  initialStatus,
  initialLog,
  initialSplit,
}: {
  datasetId: string;
  kind: DatasetKind;
  hfRepo: string | null;
  messagesField?: string | null;
  rejectedField?: string | null;
  speakerField?: string | null;
  s3Storages: StorageRecord[];
  initialStatus: string | null;
  initialLog: string | null;
  initialSplit?: string | null;
}) {
  // A chat dataset = kind=llm, OR a kind=hf / kind=upload dataset with a messages
  // column mapped (a chat dataset registered as hf, or an uploaded chat file). A
  // rejected column additionally makes it a DPO (preference) dataset.
  // Prefer the LLM pack over audio extraction in the ambiguous hf case.
  const isChat =
    kind === "llm" || ((kind === "hf" || kind === "upload") && (!!messagesField || !!rejectedField));
  if (isChat) {
    return (
      <LlmPackCard
        datasetId={datasetId}
        messagesField={messagesField ?? "messages"}
        rejectedField={rejectedField ?? null}
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
        speakerField={speakerField}
        s3Storages={s3Storages}
        initialStatus={initialStatus}
        initialLog={initialLog}
      />
    );
  }

  return (
    <AudioPackPicker
      datasetId={datasetId}
      kind={kind}
      s3Storages={s3Storages}
      initialStatus={initialStatus}
      initialLog={initialLog}
    />
  );
}

// s3 / upload audio dataset. Top-level mode:
//  • normalize — LLM-respell the transcription column → a new s3 dataset over the
//    SAME audio (kind=s3 only; metadata-only, no re-upload).
//  • pack — NeuCodec (tts_packed) or OmniVoice/Higgs (omnivoice_packed) on a GPU.
//    Both reuse TtsPackCard (variant prop).
function AudioPackPicker({
  datasetId,
  kind,
  s3Storages,
  initialStatus,
  initialLog,
}: {
  datasetId: string;
  kind: DatasetKind;
  s3Storages: StorageRecord[];
  initialStatus: string | null;
  initialLog: string | null;
}) {
  // Normalize is an S3-metadata rewrite — offered only for kind=s3 datasets.
  const canNormalize = kind === "s3";
  const [mode, setMode] = useState<"pack" | "normalize">("pack");
  const [variant, setVariant] = useState<"tts" | "omnivoice">("tts");
  return (
    <div className="space-y-4">
      {canNormalize && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Transform</span>
          <div className="flex items-center gap-0.5 rounded-md border border-border p-0.5">
            {(["normalize", "pack"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={cn(
                  "rounded px-2.5 py-1 text-xs font-medium transition-colors",
                  mode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
                )}
              >
                {m === "normalize" ? "Normalize transcription" : "Pack for training"}
              </button>
            ))}
          </div>
        </div>
      )}

      {canNormalize && mode === "normalize" ? (
        <NormalizeCard datasetId={datasetId} initialStatus={initialStatus} initialLog={initialLog} />
      ) : (
        <>
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
        </>
      )}
    </div>
  );
}
