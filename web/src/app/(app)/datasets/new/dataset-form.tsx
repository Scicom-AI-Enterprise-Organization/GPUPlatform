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

const KINDS: { value: DatasetKind; label: string; description: string }[] = [
  { value: "upload", label: "Upload metadata", description: "Upload a CSV / JSON / JSONL with {audio, transcription} rows to an S3 storage." },
  { value: "s3", label: "Existing S3 metadata", description: "Reference a metadata file that already lives in S3 (s3://bucket/key)." },
  { value: "hf", label: "HuggingFace dataset", description: "Reference an existing HuggingFace dataset repo by id (owner/name)." },
  { value: "label", label: "Labeling platform", description: "Import {audio, transcription} from a labeling-platform project using its API token." },
];

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

export function DatasetForm({ storages }: { storages: StorageRecord[] }) {
  const router = useRouter();

  const [kind, setKind] = useState<DatasetKind>("upload");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [storageId, setStorageId] = useState("");
  const [audioPrefix, setAudioPrefix] = useState("");
  const [s3MetadataUri, setS3MetadataUri] = useState("");
  const [hfRepo, setHfRepo] = useState("");
  // label: paste the project URL + a token (typed, or from a global secret)
  const [labelProjectUrl, setLabelProjectUrl] = useState("");
  const [labelToken, setLabelToken] = useState("");
  const [labelStatus, setLabelStatus] = useState("approved");
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

  // s3 storages back upload/s3; huggingface storages (optional, for the token)
  // back the hf kind.
  const storageOptions = useMemo(
    () =>
      storages.filter((s) =>
        kind === "hf" ? s.kind === "huggingface" : s.kind === "s3" && s.enabled,
      ),
    [storages, kind],
  );

  const validate = (): string | null => {
    if (!name.trim()) return "Name is required.";
    if (kind === "upload" || kind === "s3") {
      if (!storageId) return "Pick an S3 storage backend.";
      if (kind === "s3" && !s3MetadataUri.trim()) return "S3 metadata URI is required.";
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
      const created = await gateway.createDataset({
        name: name.trim(),
        kind,
        storage_id: storageId || null,
        description: description.trim() || null,
        audio_prefix: audioPrefix.trim() || null,
        s3_metadata_uri: kind === "s3" ? s3MetadataUri.trim() : null,
        hf_repo: kind === "hf" ? hfRepo.trim() : null,
        label_base_url: labelParsed?.base ?? null,
        label_project_id: labelParsed?.id ?? null,
        label_token: kind === "label" && tokenMode === "paste" ? labelToken.trim() : null,
        label_token_secret: kind === "label" && tokenMode === "secret" ? labelTokenSecret : null,
        label_status: kind === "label" ? labelStatus : null,
      });
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
                          tokenMode === m ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground",
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
