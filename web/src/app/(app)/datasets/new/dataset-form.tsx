"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { DatasetKind, GlobalEnvRecord, StorageRecord } from "@/lib/types";

// `upload_chat` is a UI-only pseudo-kind: it maps to kind=upload with a messages
// column set (an uploaded chat dataset). Everything else is a real DatasetKind.
type FormKind = DatasetKind | "upload_chat";

const KINDS: { value: FormKind; label: string; description: string }[] = [
  { value: "upload", label: "Upload metadata", description: "Upload a CSV / JSON / JSONL with {audio, transcription} rows to an S3 storage." },
  { value: "upload_chat", label: "Chat dataset (upload)", description: "Upload a JSON / JSONL / Parquet file whose rows carry a messages column ([{role, content}] — OpenAI chat format) to an S3 storage." },
  { value: "s3", label: "Existing S3 metadata", description: "Reference a metadata file that already lives in S3 (s3://bucket/key)." },
  { value: "hf", label: "HuggingFace dataset", description: "Reference an existing HuggingFace dataset repo (owner/name). Set a messages column for a chat dataset; leave it empty for audio." },
  { value: "label", label: "Labeling platform", description: "Import {audio, transcription} from a labeling-platform project using its API token." },
  { value: "tts_packed", label: "TTS packed (existing S3 shards)", description: "Register ChiniDataset parquet shards (NeuCodec multipack) already in S3 by their prefix." },
  { value: "llm_packed", label: "LLM packed (existing S3 shards)", description: "Register chat-multipack ChiniDataset parquet shards already in S3 by their prefix." },
];

// The selected source card lives in the URL (?source=…) so it's shareable +
// survives refresh. Anything unknown falls back to the first card.
const SOURCE_VALUES = KINDS.map((k) => k.value) as string[];
function normSource(s: string | undefined): FormKind {
  return SOURCE_VALUES.includes(s ?? "") ? (s as FormKind) : "upload";
}

// Pull the base URL + project id out of a pasted project URL like
// http://localhost:3002/dashboard/projects/<uuid>.
function parseLabelProjectUrl(raw: string): { base: string; id: string } | null {
  try {
    const u = new URL(raw.trim());
    const m = u.pathname.match(/\/projects\/([^/?#]+)/);
    if (!m) return null;
    return { base: u.origin, id: m[1] };
  } catch {
    return null;
  }
}

export function DatasetForm({
  storages,
  initialSource,
}: {
  storages: StorageRecord[];
  initialSource?: string;
}) {
  const router = useRouter();

  const [kind, setKind] = useState<FormKind>(() => normSource(initialSource));
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [storageId, setStorageId] = useState("");
  const [audioPrefix, setAudioPrefix] = useState("");
  const [s3MetadataUri, setS3MetadataUri] = useState("");
  const [hfRepo, setHfRepo] = useState("");
  const [hfRevision, setHfRevision] = useState("");
  // tts_packed: the tokenizer + multipack sequence length the shards were packed with
  const [packTokenizer, setPackTokenizer] = useState("Scicom-intl/Multilingual-Expressive-TTS-1.7B");
  const [packSeqLen, setPackSeqLen] = useState(4096);
  // llm_packed: tokenizer + seq len + source subset the chat shards were packed with
  const [llmPackTokenizer, setLlmPackTokenizer] = useState("");
  const [llmPackSeqLen, setLlmPackSeqLen] = useState(32768);
  const [llmPackSubset, setLlmPackSubset] = useState("");
  // hf / upload_chat / llm_packed: which column holds the messages array. Default
  // empty — on an hf dataset an empty value means "audio dataset, no chat".
  const [messagesField, setMessagesField] = useState("");
  // upload_chat: the chat file to upload in-form (json / jsonl / parquet).
  const [chatFile, setChatFile] = useState<File | null>(null);
  // label: paste the project URL + a token (typed, or from a global secret)
  const [labelProjectUrl, setLabelProjectUrl] = useState("");
  const [labelToken, setLabelToken] = useState("");
  const [labelStatus, setLabelStatus] = useState("approved");
  // label: optional point-in-time cutoff — only import tasks last updated at/before
  // this instant. Held as a `datetime-local` value (browser-local wall clock); sent
  // as a UTC ISO-8601 string. Empty → no upper bound (import every task).
  const [labelUpdatedUntil, setLabelUpdatedUntil] = useState("");
  const [tokenMode, setTokenMode] = useState<"paste" | "secret">("paste");
  const [labelTokenSecret, setLabelTokenSecret] = useState("");
  const [secrets, setSecrets] = useState<GlobalEnvRecord[]>([]);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Global secrets (admin-managed) for the "from secret" token option. Falls
  // back silently to paste-only if the list is forbidden/empty.
  useEffect(() => {
    gateway.listGlobalEnv().then((rows) => setSecrets(rows.filter((r) => r.is_secret))).catch(() => {});
  }, []);

  // s3 storages back upload / upload_chat / s3; huggingface storages (optional,
  // for the token) back the hf kind.
  const storageOptions = useMemo(
    () =>
      storages.filter((s) =>
        kind === "hf" ? s.kind === "huggingface" : s.kind === "s3" && s.enabled,
      ),
    [storages, kind],
  );

  const validate = (): string | null => {
    if (!name.trim()) return "Name is required.";
    if (kind === "upload" || kind === "upload_chat" || kind === "s3" || kind === "tts_packed" || kind === "llm_packed") {
      if (!storageId) return "Pick an S3 storage backend.";
      if (kind === "s3" && !s3MetadataUri.trim()) return "S3 metadata URI is required.";
      if ((kind === "tts_packed" || kind === "llm_packed") && !s3MetadataUri.trim()) return "S3 shards prefix is required.";
      if (kind === "upload_chat" && !chatFile) return "Choose a JSON / JSONL / Parquet file to upload.";
    }
    if (kind === "hf" && !hfRepo.trim()) return "HuggingFace repo (owner/name) is required.";
    if (kind === "label") {
      if (!parseLabelProjectUrl(labelProjectUrl)) return "Enter a valid project URL (…/projects/<id>).";
      if (tokenMode === "paste" && !labelToken.trim()) return "API token (lpat_…) is required.";
      if (tokenMode === "secret" && !labelTokenSecret) return "Pick a secret holding the token.";
    }
    return null;
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const err = validate();
    if (err) {
      setError(err);
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      const labelParsed = kind === "label" ? parseLabelProjectUrl(labelProjectUrl) : null;
      // upload_chat is a UI-only kind → a kind=upload dataset with a messages column.
      const isChatUpload = kind === "upload_chat";
      const realKind: DatasetKind = isChatUpload ? "upload" : (kind as DatasetKind);
      const created = await gateway.createDataset({
        name: name.trim(),
        kind: realKind,
        storage_id: storageId || null,
        description: description.trim() || null,
        audio_prefix: audioPrefix.trim() || null,
        s3_metadata_uri: kind === "s3" || kind === "tts_packed" || kind === "llm_packed" ? s3MetadataUri.trim() : null,
        tokenizer:
          kind === "tts_packed" ? packTokenizer.trim() || null
          : kind === "llm_packed" ? llmPackTokenizer.trim() || null
          : null,
        sequence_length:
          kind === "tts_packed" ? packSeqLen
          : kind === "llm_packed" ? llmPackSeqLen
          : null,
        subset: kind === "llm_packed" ? llmPackSubset.trim() || null : null,
        hf_repo: kind === "hf" ? hfRepo.trim() : null,
        hf_revision: kind === "hf" ? hfRevision.trim() || null : null,
        messages_field:
          isChatUpload || kind === "llm_packed"
            ? messagesField.trim() || "messages"
            : kind === "hf"
              ? messagesField.trim() || null
              : null,
        label_base_url: labelParsed?.base ?? null,
        label_project_id: labelParsed?.id ?? null,
        label_token: kind === "label" && tokenMode === "paste" ? labelToken.trim() : null,
        label_token_secret: kind === "label" && tokenMode === "secret" ? labelTokenSecret : null,
        label_status: kind === "label" ? labelStatus : null,
        label_updated_until:
          kind === "label" && labelUpdatedUntil ? new Date(labelUpdatedUntil).toISOString() : null,
      });
      // In-form file upload: push the chat file straight to the new dataset's
      // /upload endpoint (same multipart call the detail-page UploadCard makes —
      // the route expects a `file` form field, not a raw body).
      if (isChatUpload && chatFile) {
        const fd = new FormData();
        fd.append("file", chatFile);
        const res = await fetch(`/api/datasets/${encodeURIComponent(created.id)}/upload`, {
          method: "POST",
          body: fd,
        });
        if (!res.ok) {
          // The dataset was created but the file didn't land — send the user to
          // its detail page, where the Upload card surfaces the error and lets
          // them retry without re-creating the dataset.
          router.push(`/datasets/${encodeURIComponent(created.id)}?view=details`);
          return;
        }
      }
      router.push(`/datasets/${encodeURIComponent(created.id)}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={onSubmit} className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Dataset</CardTitle>
          <CardDescription>A name and an optional description.</CardDescription>
        </CardHeader>
        <CardContent className="grid items-start gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="ds-name" className="text-xs uppercase tracking-wide text-muted-foreground">Name</Label>
            <Input id="ds-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="libritts-clean" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="ds-desc" className="text-xs uppercase tracking-wide text-muted-foreground">Description (optional)</Label>
            <Textarea
              id="ds-desc"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
              placeholder="LibriTTS clean subset, 24kHz"
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Source</CardTitle>
          <CardDescription>Where the {`{audio, transcription}`} rows come from.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {KINDS.map((k) => {
              const selected = kind === k.value;
              return (
                <button
                  key={k.value}
                  type="button"
                  onClick={() => {
                    setKind(k.value);
                    setStorageId("");
                    // Reflect the selected source in the URL (?source=…) without a
                    // navigation / server re-fetch — mirrors the detail page's ?view=.
                    if (typeof window !== "undefined") {
                      const params = new URLSearchParams(window.location.search);
                      params.set("source", k.value);
                      window.history.replaceState(null, "", `${window.location.pathname}?${params.toString()}`);
                    }
                  }}
                  className={cn(
                    "rounded-md border p-3 text-left transition-colors",
                    selected
                      ? "border-foreground/60 ring-1 ring-foreground/20"
                      : "border-border hover:border-foreground/40",
                  )}
                >
                  <div className="text-sm font-medium">{k.label}</div>
                  <div className="mt-0.5 text-xs text-muted-foreground">{k.description}</div>
                </button>
              );
            })}
          </div>

          {(kind === "upload" || kind === "s3") && (
            <div className="grid items-start gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label className="text-xs uppercase tracking-wide text-muted-foreground">S3 storage</Label>
                <Select value={storageId} onValueChange={setStorageId}>
                  <SelectTrigger>
                    <SelectValue placeholder={storageOptions.length ? "Choose a storage" : "No S3 storage configured"} />
                  </SelectTrigger>
                  <SelectContent>
                    {storageOptions.map((s) => (
                      <SelectItem key={s.id} value={s.id}>
                        {s.name}
                        {s.bucket ? ` — s3://${s.bucket}${s.prefix ? "/" + s.prefix.replace(/^\/+|\/+$/g, "") : ""}` : ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {storageOptions.length === 0 && (
                  <p className="text-xs text-muted-foreground">
                    Add an S3 storage under{" "}
                    <a href="/storage/new" className="underline">Storage</a> first.
                  </p>
                )}
              </div>
              {kind === "s3" && (
                <div className="space-y-2">
                  <Label htmlFor="ds-s3uri" className="text-xs uppercase tracking-wide text-muted-foreground">S3 metadata URI</Label>
                  <Input
                    id="ds-s3uri"
                    value={s3MetadataUri}
                    onChange={(e) => setS3MetadataUri(e.target.value)}
                    placeholder="s3://my-bucket/path/metadata.csv"
                  />
                </div>
              )}
              <div className="space-y-2">
                <Label htmlFor="ds-audioprefix" className="text-xs uppercase tracking-wide text-muted-foreground">Audio prefix (optional)</Label>
                <Input
                  id="ds-audioprefix"
                  value={audioPrefix}
                  onChange={(e) => setAudioPrefix(e.target.value)}
                  placeholder="datasets/libritts/audio"
                />
                <p className="text-xs text-muted-foreground">
                  Relative audio paths in the metadata resolve under the storage prefix + this.
                </p>
              </div>
            </div>
          )}

          {kind === "tts_packed" && (
            <div className="grid items-start gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label className="text-xs uppercase tracking-wide text-muted-foreground">S3 storage</Label>
                <Select value={storageId} onValueChange={setStorageId}>
                  <SelectTrigger>
                    <SelectValue placeholder={storageOptions.length ? "Choose a storage" : "No S3 storage configured"} />
                  </SelectTrigger>
                  <SelectContent>
                    {storageOptions.map((s) => (
                      <SelectItem key={s.id} value={s.id}>
                        {s.name}
                        {s.bucket ? ` — s3://${s.bucket}${s.prefix ? "/" + s.prefix.replace(/^\/+|\/+$/g, "") : ""}` : ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {storageOptions.length === 0 && (
                  <p className="text-xs text-muted-foreground">
                    Add an S3 storage under <a href="/storage/new" className="underline">Storage</a> first.
                  </p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="ds-packprefix" className="text-xs uppercase tracking-wide text-muted-foreground">S3 shards prefix</Label>
                <Input
                  id="ds-packprefix"
                  value={s3MetadataUri}
                  onChange={(e) => setS3MetadataUri(e.target.value)}
                  placeholder="s3://my-bucket/path/packed/"
                />
                <p className="text-xs text-muted-foreground">
                  Folder holding the ChiniDataset parquet shards (with <code>train/</code> + <code>test/</code> subdirs).
                  Splits, row counts and size are read from it.
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="ds-tok" className="text-xs uppercase tracking-wide text-muted-foreground">Tokenizer</Label>
                <Input
                  id="ds-tok"
                  value={packTokenizer}
                  onChange={(e) => setPackTokenizer(e.target.value)}
                  placeholder="owner/model"
                />
                <p className="text-xs text-muted-foreground">The speech-token tokenizer the shards were packed with (decodes rows to text).</p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="ds-seqlen" className="text-xs uppercase tracking-wide text-muted-foreground">Sequence length</Label>
                <Input
                  id="ds-seqlen"
                  type="number"
                  value={packSeqLen}
                  onChange={(e) => setPackSeqLen(Number.parseInt(e.target.value, 10) || 4096)}
                />
              </div>
            </div>
          )}

          {kind === "llm_packed" && (
            <div className="grid items-start gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label className="text-xs uppercase tracking-wide text-muted-foreground">S3 storage</Label>
                <Select value={storageId} onValueChange={setStorageId}>
                  <SelectTrigger>
                    <SelectValue placeholder={storageOptions.length ? "Choose a storage" : "No S3 storage configured"} />
                  </SelectTrigger>
                  <SelectContent>
                    {storageOptions.map((s) => (
                      <SelectItem key={s.id} value={s.id}>
                        {s.name}
                        {s.bucket ? ` — s3://${s.bucket}${s.prefix ? "/" + s.prefix.replace(/^\/+|\/+$/g, "") : ""}` : ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {storageOptions.length === 0 && (
                  <p className="text-xs text-muted-foreground">
                    Add an S3 storage under <a href="/storage/new" className="underline">Storage</a> first.
                  </p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="ds-llmpackprefix" className="text-xs uppercase tracking-wide text-muted-foreground">S3 shards prefix</Label>
                <Input
                  id="ds-llmpackprefix"
                  value={s3MetadataUri}
                  onChange={(e) => setS3MetadataUri(e.target.value)}
                  placeholder="s3://my-bucket/datasets/ds-xxxx/packed"
                />
                <p className="text-xs text-muted-foreground">
                  Folder holding the chat-multipack ChiniDataset parquet shards. Row counts and size are read from it.
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="ds-llmtok" className="text-xs uppercase tracking-wide text-muted-foreground">Tokenizer <span className="normal-case text-muted-foreground">(optional)</span></Label>
                <Input
                  id="ds-llmtok"
                  value={llmPackTokenizer}
                  onChange={(e) => setLlmPackTokenizer(e.target.value)}
                  placeholder="google/gemma-4-31B-it"
                />
                <p className="text-xs text-muted-foreground">The tokenizer the shards were packed with (used to decode rows to text).</p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="ds-llmseqlen" className="text-xs uppercase tracking-wide text-muted-foreground">Sequence length</Label>
                <Input
                  id="ds-llmseqlen"
                  type="number"
                  value={llmPackSeqLen}
                  onChange={(e) => setLlmPackSeqLen(Number.parseInt(e.target.value, 10) || 32768)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="ds-llmmsgs" className="text-xs uppercase tracking-wide text-muted-foreground">Messages column</Label>
                <Input
                  id="ds-llmmsgs"
                  value={messagesField}
                  onChange={(e) => setMessagesField(e.target.value)}
                  placeholder="messages"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="ds-llmsubset" className="text-xs uppercase tracking-wide text-muted-foreground">Source subset <span className="normal-case text-muted-foreground">(optional)</span></Label>
                <Input
                  id="ds-llmsubset"
                  value={llmPackSubset}
                  onChange={(e) => setLlmPackSubset(e.target.value)}
                  placeholder="glm5.1-fp8-test"
                />
                <p className="text-xs text-muted-foreground">The source config/subset that was packed (descriptive metadata).</p>
              </div>
            </div>
          )}

          {kind === "hf" && (
            <div className="grid items-start gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="ds-hfrepo" className="text-xs uppercase tracking-wide text-muted-foreground">HuggingFace repo</Label>
                <Input
                  id="ds-hfrepo"
                  value={hfRepo}
                  onChange={(e) => setHfRepo(e.target.value)}
                  placeholder="owner/dataset-name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="ds-hf-messages" className="text-xs uppercase tracking-wide text-muted-foreground">Messages column <span className="normal-case text-muted-foreground">(optional)</span></Label>
                <Input
                  id="ds-hf-messages"
                  value={messagesField}
                  onChange={(e) => setMessagesField(e.target.value)}
                  placeholder="messages"
                />
                <p className="text-xs text-muted-foreground">
                  Set for a chat / LLM dataset (OpenAI-format array, usually <span className="font-mono">messages</span>). Leave empty for an audio dataset.
                </p>
              </div>
              <div className="space-y-2">
                <Label className="text-xs uppercase tracking-wide text-muted-foreground">HuggingFace storage (optional, for private repos)</Label>
                <Select value={storageId} onValueChange={setStorageId}>
                  <SelectTrigger>
                    <SelectValue placeholder={storageOptions.length ? "Choose a HuggingFace storage (optional)" : "No HuggingFace storage configured"} />
                  </SelectTrigger>
                  <SelectContent>
                    {storageOptions.map((s) => (
                      <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="ds-hfrev" className="text-xs uppercase tracking-wide text-muted-foreground">Revision (optional)</Label>
                <Input
                  id="ds-hfrev"
                  value={hfRevision}
                  onChange={(e) => setHfRevision(e.target.value)}
                  placeholder="main, v1.0.0, or a commit SHA"
                />
                <p className="text-xs text-muted-foreground">
                  Git branch, tag, or commit hash to pin. Blank → the repo&apos;s default branch.
                </p>
              </div>
            </div>
          )}

          {kind === "upload_chat" && (
            <div className="grid items-start gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label className="text-xs uppercase tracking-wide text-muted-foreground">S3 storage</Label>
                <Select value={storageId} onValueChange={setStorageId}>
                  <SelectTrigger>
                    <SelectValue placeholder={storageOptions.length ? "Choose a storage" : "No S3 storage configured"} />
                  </SelectTrigger>
                  <SelectContent>
                    {storageOptions.map((s) => (
                      <SelectItem key={s.id} value={s.id}>
                        {s.name}
                        {s.bucket ? ` — s3://${s.bucket}${s.prefix ? "/" + s.prefix.replace(/^\/+|\/+$/g, "") : ""}` : ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {storageOptions.length === 0 && (
                  <p className="text-xs text-muted-foreground">
                    Add an S3 storage under <a href="/storage/new" className="underline">Storage</a> first.
                  </p>
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="ds-chat-messages" className="text-xs uppercase tracking-wide text-muted-foreground">Messages column</Label>
                <Input
                  id="ds-chat-messages"
                  value={messagesField}
                  onChange={(e) => setMessagesField(e.target.value)}
                  placeholder="messages"
                />
                <p className="text-xs text-muted-foreground">Column in the file holding the OpenAI-format chat array. Usually <span className="font-mono">messages</span>.</p>
              </div>
              <div className="space-y-2 sm:col-span-2">
                <Label htmlFor="ds-chat-file" className="text-xs uppercase tracking-wide text-muted-foreground">Chat file</Label>
                <Input
                  id="ds-chat-file"
                  type="file"
                  accept=".json,.jsonl,.ndjson,.parquet"
                  onChange={(e) => setChatFile(e.target.files?.[0] ?? null)}
                />
                <p className="text-xs text-muted-foreground">
                  JSON / JSONL / Parquet. Each row carries a <span className="font-mono">{messagesField.trim() || "messages"}</span> column
                  ({`[{role, content}]`}). Uploaded to the selected storage on submit.
                </p>
              </div>
            </div>
          )}

          {kind === "label" && (
            <div className="space-y-4">
              <div className="grid items-start gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="ds-labelurl" className="text-xs uppercase tracking-wide text-muted-foreground">Project URL</Label>
                  <Input
                    id="ds-labelurl"
                    value={labelProjectUrl}
                    onChange={(e) => setLabelProjectUrl(e.target.value)}
                    placeholder="http://localhost:3002/dashboard/projects/<id>"
                  />
                  <p className="text-xs text-muted-foreground">
                    Paste a labeling-platform project URL. {`{audio, transcription}`} rows are imported live.
                  </p>
                </div>
                <div className="space-y-2">
                  <Label className="text-xs uppercase tracking-wide text-muted-foreground">Import which tasks</Label>
                  <Select value={labelStatus} onValueChange={setLabelStatus}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="approved">Approved only (review-passed)</SelectItem>
                      <SelectItem value="all">All tasks</SelectItem>
                      <SelectItem value="not_reviewed">Not reviewed</SelectItem>
                      <SelectItem value="rejected">Rejected</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="ds-labelcutoff" className="text-xs uppercase tracking-wide text-muted-foreground">
                  Up to (timestamp cutoff) <span className="text-muted-foreground/60 normal-case">— optional</span>
                </Label>
                <Input
                  id="ds-labelcutoff"
                  type="datetime-local"
                  value={labelUpdatedUntil}
                  onChange={(e) => setLabelUpdatedUntil(e.target.value)}
                  className="sm:max-w-xs"
                />
                <p className="text-xs text-muted-foreground">
                  Only import tasks last updated at or before this moment — a point-in-time snapshot. Read in your
                  local timezone
                  {labelUpdatedUntil ? (
                    <> (= <span className="font-mono">{new Date(labelUpdatedUntil).toISOString()}</span> UTC)</>
                  ) : null}
                  . Leave blank to import every task.
                </p>
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-3">
                  <Label className="text-xs uppercase tracking-wide text-muted-foreground">API token</Label>
                  <div className="inline-flex overflow-hidden rounded-md border border-border text-xs">
                    {(["paste", "secret"] as const).map((m) => (
                      <button
                        key={m}
                        type="button"
                        onClick={() => setTokenMode(m)}
                        className={cn(
                          "px-2.5 py-1 transition-colors",
                          tokenMode === m ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
                        )}
                      >
                        {m === "paste" ? "Paste" : "From secret"}
                      </button>
                    ))}
                  </div>
                </div>
                {tokenMode === "paste" ? (
                  <>
                    <Input
                      id="ds-labeltoken"
                      type="password"
                      className="font-mono"
                      value={labelToken}
                      onChange={(e) => setLabelToken(e.target.value)}
                      placeholder="lpat_…"
                    />
                    <p className="text-xs text-muted-foreground">
                      Personal access token (<span className="font-mono">lpat_…</span>). Stored encrypted; never shown again.
                    </p>
                  </>
                ) : (
                  <>
                    <Select value={labelTokenSecret} onValueChange={setLabelTokenSecret}>
                      <SelectTrigger>
                        <SelectValue placeholder={secrets.length ? "Choose a secret" : "No secrets configured"} />
                      </SelectTrigger>
                      <SelectContent>
                        {secrets.map((s) => (
                          <SelectItem key={s.key} value={s.key}>
                            {s.key}{s.value_preview ? ` — ${s.value_preview}` : ""}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      Resolved from{" "}
                      <a href="/admin/secrets" className="underline">global secrets</a>{" "}
                      at import time — nothing token-related is stored on the dataset.
                    </p>
                  </>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {error && (
        <p className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </p>
      )}

      <div className="flex items-center justify-end gap-3">
        <Button type="button" variant="ghost" onClick={() => router.push("/datasets")}>
          Cancel
        </Button>
        <Button type="submit" disabled={submitting}>
          {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
          Register dataset
        </Button>
      </div>
    </form>
  );
}
