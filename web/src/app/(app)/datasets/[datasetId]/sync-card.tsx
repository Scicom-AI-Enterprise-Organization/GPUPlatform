"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, UploadCloud } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { gateway } from "@/lib/gateway";

export function SyncCard({
  datasetId,
  canSync,
  currentRepo,
}: {
  datasetId: string;
  canSync: boolean;
  currentRepo?: string | null;
}) {
  const router = useRouter();
  const [repo, setRepo] = useState(currentRepo ?? "");
  const [priv, setPriv] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSync = async () => {
    if (!repo.trim()) {
      setError("Enter a repo (owner/name).");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await gateway.syncDataset(datasetId, { hf_repo: repo.trim(), private: priv });
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Sync to HuggingFace</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground">
          Push the uploaded metadata file to a HuggingFace dataset repo. Uses a
          HuggingFace storage token if configured, otherwise the gateway&apos;s{" "}
          <span className="font-mono text-xs">HF_TOKEN</span>.
        </p>
        <div className="space-y-2">
          <Label htmlFor="sync-repo">Repo</Label>
          <Input
            id="sync-repo"
            value={repo}
            onChange={(e) => setRepo(e.target.value)}
            placeholder="owner/dataset-name"
            disabled={!canSync}
          />
        </div>
        <div className="flex items-center gap-2">
          <Switch id="sync-private" checked={priv} onCheckedChange={setPriv} disabled={!canSync} />
          <Label htmlFor="sync-private" className="text-sm font-normal">Private repo</Label>
        </div>

        {!canSync && (
          <p className="text-xs text-muted-foreground">Upload a metadata file first.</p>
        )}
        {error && (
          <p className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error}
          </p>
        )}

        <Button onClick={onSync} disabled={!canSync || busy} size="sm">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <UploadCloud className="h-4 w-4" />}
          Sync to HuggingFace
        </Button>
      </CardContent>
    </Card>
  );
}
