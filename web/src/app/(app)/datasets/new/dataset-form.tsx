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
import { FormFooter, FormShell } from "@/components/form-shell";
import { gateway } from "@/lib/gateway";
import type {
  DatasetKind,
  GenerateDatasetOptions,
  GenerateDatasetRequest,
  GenerateDatasetRow,
  GlobalEnvRecord,
  StorageRecord,
} from "@/lib/types";

// `upload_chat` and `generate` are UI-only pseudo-kinds. Both produce a
// kind=upload chat dataset; `generate` posts to /v1/datasets/generate instead of
// the plain create route, because its rows are WRITTEN BY A MODEL in the
// background rather than uploaded. Everything else is a real DatasetKind.
type FormKind = DatasetKind | "upload_chat" | "generate";

const KINDS: { value: FormKind; label: string; description: string }[] = [
  { value: "upload", label: "Upload metadata", description: "Upload a CSV / JSON / JSONL with {audio, transcription} rows to an S3 storage." },
  { value: "upload_chat", label: "Chat dataset (upload)", description: "Upload a JSON / JSONL / Parquet file whose rows carry a messages column ([{role, content}] — OpenAI chat format) to an S3 storage." },
  { value: "s3", label: "Existing S3 metadata", description: "Reference a metadata file that already lives in S3 (s3://bucket/key)." },
  { value: "hf", label: "HuggingFace dataset", description: "Reference an existing HuggingFace dataset repo (owner/name). Set a messages column for a chat dataset; leave it empty for audio." },
  { value: "label", label: "Labeling platform", description: "Import {audio, transcription} from a labeling-platform project using its API token." },
  { value: "tts_packed", label: "TTS packed (existing S3 shards)", description: "Register ChiniDataset parquet shards (NeuCodec multipack) already in S3 by their prefix." },
  { value: "llm_packed", label: "LLM packed (existing S3 shards)", description: "Register chat-multipack ChiniDataset parquet shards already in S3 by their prefix." },
  { value: "generate", label: "Generate (synthetic)", description: "Have an OpenAI-compatible model write the rows — e.g. a red-team corpus of attack prompts plus benign controls. The dataset is created immediately and its rows fill in the background." },
];

// The selected source card lives in the URL (?source=…) so it's shareable +
// survives refresh. Anything unknown falls back to the first card.
const SOURCE_VALUES = KINDS.map((k) => k.value) as string[];
function normSource(s: string | undefined): FormKind {
  return SOURCE_VALUES.includes(s ?? "") ? (s as FormKind) : "upload";
}

// "synthetic/train, synthetic_podcast/train" (or one per line) → the array the
// gateway stores as the dataset's subset scope. Blank → null (the whole repo).
function parseSubsets(raw: string): string[] | null {
  const out = raw
    .split(/[\n,]/)
    .map((s) => s.trim())
    .filter(Boolean);
  return out.length ? Array.from(new Set(out)) : null;
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
  // generate (synthetic): the generator endpoint + what to write. Options
  // (taxonomy + ceilings) are server-driven — adding a category in synthetic.py
  // needs no change here.
  const [genOpts, setGenOpts] = useState<GenerateDatasetOptions | null>(null);
  const [genBase, setGenBase] = useState("");
  const [genModel, setGenModel] = useState("");
  const [genKeyMode, setGenKeyMode] = useState<"secret" | "paste">("secret");
  const [genKey, setGenKey] = useState("");
  const [genKeySecret, setGenKeySecret] = useState("");
  const [genMode, setGenMode] = useState<"attack" | "benign" | "mixed">("mixed");
  const [genRows, setGenRows] = useState(30);
  const [genCats, setGenCats] = useState<Set<string>>(new Set());
  const [genDomain, setGenDomain] = useState("");
  const [genLanguages, setGenLanguages] = useState("");
  const [genSystemPrompt, setGenSystemPrompt] = useState("");
  const [genExtra, setGenExtra] = useState("");
  const [genPreview, setGenPreview] = useState<GenerateDatasetRow[] | null>(null);
  const [genPreviewBusy, setGenPreviewBusy] = useState(false);
  const [hfRepo, setHfRepo] = useState("");
  const [hfRevision, setHfRevision] = useState("");
  // hf: scope the dataset to some of the repo's declared configs/splits. Comma-
  // or newline-separated; blank = the whole repo.
  const [hfSubsets, setHfSubsets] = useState("");
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
  // label: per-clip audio-download retry cap for the transform. Empty → retry
  // until success (the default — the platform ingress flakes under load and a
  // single attempt silently drops rows). A positive number caps the attempts.
  const [labelDownloadRetries, setLabelDownloadRetries] = useState("");
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

  useEffect(() => {
    if (kind !== "generate" || genOpts) return;
    gateway
      .generateDatasetOptions()
      .then((o) => {
        setGenOpts(o);
        setGenRows(o.default_rows);
        setGenCats(new Set(o.categories.map((c) => c.id)));   // all on by default
      })
      .catch(() => {});
  }, [kind, genOpts]);

  const genBody = (): GenerateDatasetRequest => ({
    name: name.trim(),
    storage_id: storageId,
    description: description.trim() || null,
    base_url: genBase.trim(),
    model: genModel.trim(),
    api_key: genKeyMode === "paste" ? genKey.trim() || null : null,
    api_key_secret: genKeyMode === "secret" ? genKeySecret || null : null,
    mode: genMode,
    n_rows: genRows,
    categories: [...genCats],
    languages: genLanguages.split(/[,\n]+/).map((x) => x.trim()).filter(Boolean),
    domain: genDomain.trim(),
    system_prompt: genSystemPrompt.trim(),
    extra_instructions: genExtra.trim(),
  });

  const onPreviewGenerated = async () => {
    setGenPreviewBusy(true);
    setError(null);
    try {
      const res = await gateway.previewGeneratedRows(genBody());
      setGenPreview(res.rows);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setGenPreviewBusy(false);
    }
  };

  const validate = (): string | null => {
    if (!name.trim()) return "Name is required.";
    if (kind === "generate") {
      if (!storageId) return "Pick an S3 storage backend.";
      if (!genBase.trim()) return "Generator base URL is required.";
      if (!genModel.trim()) return "Generator model is required.";
      if (genCats.size === 0) return "Pick at least one attack category.";
    }
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
      // Generation returns as soon as the EMPTY row exists — the rows arrive
      // afterwards, so we navigate straight to the detail page where the
      // transform card shows them growing.
      if (kind === "generate") {
        const created = await gateway.generateDataset(genBody());
        router.push(`/datasets/${encodeURIComponent(created.id)}`);
        return;
      }
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
        hf_subsets: kind === "hf" ? parseSubsets(hfSubsets) : null,
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
        // Empty → null → gateway retries until success (the default).
        label_download_retries:
          kind === "label" && labelDownloadRetries.trim()
            ? Number.parseInt(labelDownloadRetries, 10)
            : null,
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
    <FormShell>
    <form onSubmit={onSubmit} className="space-y-6">
      <Card data-form-section="Dataset" className="scroll-mt-6">
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

      <Card data-form-section="Source" className="scroll-mt-6">
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
              <div className="space-y-2 sm:col-span-2">
                <Label htmlFor="ds-hf-subsets" className="text-xs uppercase tracking-wide text-muted-foreground">Subsets <span className="normal-case text-muted-foreground">(optional)</span></Label>
                <Input
                  id="ds-hf-subsets"
                  value={hfSubsets}
                  onChange={(e) => setHfSubsets(e.target.value)}
                  placeholder="synthetic/train, synthetic_podcast/train"
                />
                <p className="text-xs text-muted-foreground">
                  Scope this dataset to some of the repo&apos;s declared configs/splits — comma-separated,
                  named as the row browser shows them (<span className="font-mono">config/split</span>), or a bare
                  config name to take all of its splits. Blank → the whole repo. Transforms only download the
                  selected ones, so an unused multi-GB config costs nothing.
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

          {kind === "generate" && (
            <div className="space-y-4">
              <div className="grid items-start gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label className="text-xs uppercase tracking-wide text-muted-foreground">S3 storage</Label>
                  <Select value={storageId} onValueChange={setStorageId}>
                    <SelectTrigger>
                      <SelectValue placeholder={storageOptions.length ? "Choose a storage" : "No S3 storage configured"} />
                    </SelectTrigger>
                    <SelectContent>
                      {storageOptions.map((st) => (
                        <SelectItem key={st.id} value={st.id}>
                          {st.name}
                          {st.bucket ? ` — s3://${st.bucket}` : ""}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    The generated rows are written here as <span className="font-mono">cases.jsonl</span>, republished after every batch.
                  </p>
                </div>
                <div className="space-y-2">
                  <Label className="text-xs uppercase tracking-wide text-muted-foreground">Corpus</Label>
                  <Select value={genMode} onValueChange={(v) => setGenMode(v as typeof genMode)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="mixed">Attacks + benign controls</SelectItem>
                      <SelectItem value="attack">Attacks only</SelectItem>
                      <SelectItem value="benign">Benign controls only</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Benign controls are harmless look-alikes the model <em>should</em> answer — without them, a model that refuses everything scores a perfect refusal rate.
                  </p>
                </div>
              </div>

              <div className="grid items-start gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="gen-base" className="text-xs uppercase tracking-wide text-muted-foreground">Generator base URL</Label>
                  <Input id="gen-base" value={genBase} onChange={(e) => setGenBase(e.target.value)}
                         placeholder="https://…/proxy/gemma/v1" className="font-mono text-xs" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="gen-model" className="text-xs uppercase tracking-wide text-muted-foreground">Generator model</Label>
                  <Input id="gen-model" value={genModel} onChange={(e) => setGenModel(e.target.value)}
                         placeholder="gemma" className="font-mono text-xs" />
                </div>
              </div>

              <div className="space-y-2">
                <Label className="text-xs uppercase tracking-wide text-muted-foreground">Generator API key</Label>
                <div className="flex flex-wrap items-center gap-2">
                  <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
                    {(["secret", "paste"] as const).map((m) => (
                      <button key={m} type="button" onClick={() => setGenKeyMode(m)}
                              className={cn("rounded px-2 py-1",
                                genKeyMode === m ? "bg-primary text-primary-foreground"
                                                 : "text-muted-foreground hover:text-foreground")}>
                        {m === "secret" ? "Secret ref" : "Paste"}
                      </button>
                    ))}
                  </div>
                  {genKeyMode === "secret" ? (
                    <Select value={genKeySecret} onValueChange={setGenKeySecret}>
                      <SelectTrigger className="h-8 max-w-xs text-xs"><SelectValue placeholder="Pick a secret" /></SelectTrigger>
                      <SelectContent>
                        {secrets.map((x) => (
                          <SelectItem key={x.key} value={x.key} className="text-xs">{x.key}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  ) : (
                    <Input type="password" autoComplete="off" value={genKey}
                           onChange={(e) => setGenKey(e.target.value)}
                           placeholder="sgpu_… (never stored)" className="h-8 max-w-xs font-mono text-xs" />
                  )}
                  <span className="text-xs text-muted-foreground">optional — omit for a keyless endpoint</span>
                </div>
              </div>

              <div className="grid items-start gap-4 sm:grid-cols-3">
                <div className="space-y-2">
                  <Label htmlFor="gen-rows" className="text-xs uppercase tracking-wide text-muted-foreground">Rows</Label>
                  <Input id="gen-rows" type="number" min={1} max={genOpts?.max_rows ?? 200} value={genRows}
                         onChange={(e) => setGenRows(Number(e.target.value) || 1)} />
                  <p className="text-xs text-muted-foreground">max {genOpts?.max_rows ?? 200}</p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="gen-langs" className="text-xs uppercase tracking-wide text-muted-foreground">Languages</Label>
                  <Input id="gen-langs" value={genLanguages} onChange={(e) => setGenLanguages(e.target.value)}
                         placeholder="english, malay" />
                  <p className="text-xs text-muted-foreground">blank = English</p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="gen-domain" className="text-xs uppercase tracking-wide text-muted-foreground">Target the attacks at</Label>
                  <Input id="gen-domain" value={genDomain} onChange={(e) => setGenDomain(e.target.value)}
                         placeholder="a telco customer-service agent" />
                </div>
              </div>

              <div className="space-y-2">
                <Label className="text-xs uppercase tracking-wide text-muted-foreground">Attack categories</Label>
                <div className="flex flex-wrap gap-1.5">
                  {(genOpts?.categories ?? []).map((c) => {
                    const on = genCats.has(c.id);
                    return (
                      <button key={c.id} type="button" title={c.brief}
                              onClick={() => setGenCats((prev) => {
                                const next = new Set(prev);
                                if (next.has(c.id)) next.delete(c.id);
                                else next.add(c.id);
                                return next;
                              })}
                              className={cn("rounded-md border px-2 py-1 font-mono text-[11px] transition-colors",
                                on ? "border-primary bg-primary/10 text-foreground"
                                   : "border-border text-muted-foreground hover:text-foreground")}>
                        {c.id}
                      </button>
                    );
                  })}
                </div>
                <p className="text-xs text-muted-foreground">
                  Rows are split evenly across the selected categories — the same taxonomy the proxy&apos;s red-team guard reports.
                </p>
              </div>

              <div className="grid items-start gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="gen-sys" className="text-xs uppercase tracking-wide text-muted-foreground">System prompt for each row</Label>
                  <Input id="gen-sys" value={genSystemPrompt} onChange={(e) => setGenSystemPrompt(e.target.value)}
                         placeholder="optional — the prompt under test" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="gen-extra" className="text-xs uppercase tracking-wide text-muted-foreground">Extra instructions to the generator</Label>
                  <Textarea id="gen-extra" rows={2} value={genExtra} onChange={(e) => setGenExtra(e.target.value)}
                            placeholder="optional — e.g. keep prompts under 40 words" className="text-xs" />
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2 border-t border-border/60 pt-3">
                <Button type="button" variant="outline" size="sm" onClick={() => void onPreviewGenerated()}
                        disabled={genPreviewBusy || !genBase.trim() || !genModel.trim() || genCats.size === 0}>
                  {genPreviewBusy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                  Preview {genOpts?.preview_rows ?? 6} rows
                </Button>
                <span className="text-xs text-muted-foreground">
                  costs one call and creates nothing — check the endpoint before generating {genRows} rows
                </span>
              </div>

              {genPreview && (
                <ul className="max-h-60 space-y-1.5 overflow-y-auto rounded-md border border-border p-2">
                  {genPreview.map((r, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs">
                      <span className={cn("mt-0.5 shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px]",
                        r.expected?.attack ? "bg-amber-500/15 text-amber-700 dark:text-amber-400"
                                           : "bg-muted text-muted-foreground")}>
                        {r.expected?.attack ? r.expected?.attack_type || "attack" : "benign"}
                      </span>
                      <span className="min-w-0 flex-1">{r.messages[r.messages.length - 1]?.content}</span>
                    </li>
                  ))}
                </ul>
              )}
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
                <Label htmlFor="ds-labelretries" className="text-xs uppercase tracking-wide text-muted-foreground">
                  Download retries <span className="text-muted-foreground/60 normal-case">— optional</span>
                </Label>
                <Input
                  id="ds-labelretries"
                  type="number"
                  min={0}
                  value={labelDownloadRetries}
                  onChange={(e) => setLabelDownloadRetries(e.target.value)}
                  placeholder="retry until success"
                  className="sm:max-w-xs"
                />
                <p className="text-xs text-muted-foreground">
                  How many times the transform retries a clip whose audio fails to download before giving up. The
                  platform ingress flakes under the load of a big project, so{" "}
                  <span className="font-medium">leave blank to retry until success</span> (recommended) — no rows are
                  silently dropped. Set a number to cap the attempts instead.
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

      <FormFooter error={error}>
        <Button type="button" variant="ghost" onClick={() => router.push("/datasets")}>
          Cancel
        </Button>
        <Button type="submit" disabled={submitting}>
          {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
          Register dataset
        </Button>
      </FormFooter>
    </form>
    </FormShell>
  );
}
