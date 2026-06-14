"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { gateway } from "@/lib/gateway";
import type { CatalogRepoType, StorageRecord } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const KIND_LABEL: Record<StorageRecord["kind"], string> = {
  s3: "S3",
  local: "Local",
  sftp: "SFTP",
  huggingface: "HuggingFace",
};

/** Register a model or dataset repo. `repoType` is fixed by the caller
 * (/models → model, /datasets/hosted → dataset). On success redirects to the
 * repo's detail page under the matching section. */
export function CatalogForm({
  storages,
  defaultNamespace,
  repoType,
}: {
  storages: StorageRecord[];
  defaultNamespace: string;
  repoType: CatalogRepoType;
}) {
  const router = useRouter();
  const detailBase = repoType === "dataset" ? "/datasets/hosted" : "/models";
  const listBase = repoType === "dataset" ? "/datasets" : "/models";

  const [namespace, setNamespace] = useState(defaultNamespace || "");
  const [name, setName] = useState("");
  const [storageId, setStorageId] = useState(storages[0]?.id ?? "");
  const [prefix, setPrefix] = useState("");
  const [isPrivate, setIsPrivate] = useState(true);
  const [description, setDescription] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const defaultPrefix = namespace && name ? `catalog/${namespace}/${name}` : "catalog/<ns>/<name>";
  const canSubmit = namespace.trim() && name.trim() && storageId && !submitting;

  async function submit() {
    setSubmitting(true);
    setError(null);
    try {
      const repo = await gateway.createCatalogRepo({
        repo_type: repoType,
        namespace: namespace.trim(),
        name: name.trim(),
        storage_id: storageId,
        prefix: prefix.trim() || null,
        private: isPrivate,
        description: description.trim() || null,
      });
      router.push(`${detailBase}/${repo.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSubmitting(false);
    }
  }

  if (storages.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-6 text-sm">
        <p className="text-muted-foreground">
          No S3, local, or SFTP storage is configured. Repos need a storage backend to
          hold their files.
        </p>
        <Button asChild className="mt-4" size="sm">
          <Link href="/storage">Add a storage</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-border bg-card p-6 space-y-5">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="ns">Namespace</Label>
            <Input id="ns" value={namespace} onChange={(e) => setNamespace(e.target.value)} placeholder="my-org" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="name">Name</Label>
            <Input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={repoType === "dataset" ? "my-dataset" : "my-model"}
            />
          </div>
        </div>
        <p className="-mt-2 text-xs text-muted-foreground">
          Repo id: <span className="font-mono">{namespace || "<ns>"}/{name || "<name>"}</span>
        </p>

        <div className="space-y-2">
          <Label>Storage</Label>
          <Select value={storageId} onValueChange={setStorageId}>
            <SelectTrigger>
              <SelectValue placeholder="Choose a storage" />
            </SelectTrigger>
            <SelectContent>
              {storages.map((s) => (
                <SelectItem key={s.id} value={s.id}>
                  {s.name} · {KIND_LABEL[s.kind] ?? s.kind}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="prefix">
            Prefix <span className="text-muted-foreground">(optional)</span>
          </Label>
          <Input
            id="prefix"
            value={prefix}
            onChange={(e) => setPrefix(e.target.value)}
            placeholder={defaultPrefix}
            className="font-mono text-sm"
          />
          <p className="text-xs text-muted-foreground">
            Key prefix within the storage where files live. Defaults to{" "}
            <span className="font-mono">{defaultPrefix}</span>.
          </p>
        </div>

        <div className="flex items-center justify-between rounded-md border border-border px-3 py-2.5">
          <div>
            <Label className="cursor-pointer">Private</Label>
            <p className="text-xs text-muted-foreground">
              Only you (and admins) can pull this repo with your API key.
            </p>
          </div>
          <Switch checked={isPrivate} onCheckedChange={setIsPrivate} />
        </div>

        <div className="space-y-2">
          <Label htmlFor="desc">
            Description <span className="text-muted-foreground">(optional)</span>
          </Label>
          <Textarea id="desc" value={description} onChange={(e) => setDescription(e.target.value)} rows={2} />
        </div>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="flex items-center justify-end gap-2">
        <Button variant="outline" asChild>
          <Link href={listBase}>Cancel</Link>
        </Button>
        <Button onClick={submit} disabled={!canSubmit}>
          {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
          Create {repoType === "dataset" ? "dataset" : "repo"}
        </Button>
      </div>
    </div>
  );
}
