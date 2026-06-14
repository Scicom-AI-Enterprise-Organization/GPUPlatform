"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  Check, Copy, Database, FileText, Lock, Package, RefreshCw, Trash2,
} from "lucide-react";
import { gateway } from "@/lib/gateway";
import type { CatalogRecord } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";
import { fmtBytes } from "./catalog-list";

function CopyBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="relative">
      <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 pr-10 text-xs leading-relaxed scrollbar-thin">
        <code className="font-mono">{code}</code>
      </pre>
      <button
        onClick={async () => {
          await navigator.clipboard.writeText(code);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        }}
        className="absolute right-2 top-2 rounded-md bg-background/80 p-1.5 text-muted-foreground hover:text-foreground"
        title="Copy"
      >
        {copied ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-card px-4 py-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-0.5 truncate text-sm font-medium" title={typeof value === "string" ? value : undefined}>
        {value}
      </div>
    </div>
  );
}

export function CatalogDetail({
  repo,
  gatewayUrl,
  backHref,
}: {
  repo: CatalogRecord;
  gatewayUrl: string;
  backHref: string;
}) {
  const router = useRouter();
  const [reindexing, setReindexing] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [wipe, setWipe] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const endpoint = `${gatewayUrl.replace(/\/$/, "")}/hf`;
  const isDataset = repo.repo_type === "dataset";
  const files = repo.files ?? [];

  const envSnippet = `export HF_ENDPOINT=${endpoint}\nexport HF_TOKEN=sgpu_…   # your platform API key\n# or:  hf auth login --token sgpu_…`;

  const pullSnippet = isDataset
    ? `# Python\nfrom huggingface_hub import snapshot_download\nsnapshot_download("${repo.full_id}", repo_type="dataset")\n\n# CLI\nhf download ${repo.full_id} --repo-type dataset`
    : `# Python\nfrom transformers import AutoModel\nAutoModel.from_pretrained("${repo.full_id}")\n\n# CLI\nhf download ${repo.full_id}`;

  const pushSnippet = isDataset
    ? `# CLI\nhf upload ${repo.full_id} ./my-dataset --repo-type dataset\n\n# Python\nfrom huggingface_hub import HfApi\nHfApi().upload_folder(folder_path="./my-dataset", repo_id="${repo.full_id}", repo_type="dataset")`
    : `# CLI\nhf upload ${repo.full_id} ./my-model\n\n# Python\nmodel.push_to_hub("${repo.full_id}")`;

  async function reindex() {
    setReindexing(true);
    setErr(null);
    try {
      await gateway.reindexCatalogRepo(repo.id);
      router.refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setReindexing(false);
    }
  }

  async function doDelete() {
    setBusy(true);
    setErr(null);
    try {
      await gateway.deleteCatalogRepo(repo.id, wipe);
      router.push(backHref);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-md bg-muted text-muted-foreground">
            {isDataset ? <Database className="h-5 w-5" /> : <Package className="h-5 w-5" />}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-mono text-xl font-semibold tracking-tight">{repo.full_id}</h1>
              <Badge variant="outline">{repo.repo_type}</Badge>
              {repo.private && (
                <Badge variant="secondary">
                  <Lock className="h-3 w-3" /> private
                </Badge>
              )}
            </div>
            {repo.description && <p className="mt-1 text-sm text-muted-foreground">{repo.description}</p>}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={reindex} disabled={reindexing}>
            <RefreshCw className={`h-4 w-4 ${reindexing ? "animate-spin" : ""}`} />
            Reindex
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="text-destructive hover:text-destructive"
            onClick={() => {
              setDeleteOpen(true);
              setWipe(false);
              setErr(null);
            }}
          >
            <Trash2 className="h-4 w-4" />
            Delete
          </Button>
        </div>
      </div>

      {err && (
        <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {err}
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Kpi label="Files" value={String(repo.num_files ?? files.length)} />
        <Kpi label="Size" value={fmtBytes(repo.size_bytes)} />
        <Kpi label="Storage" value={repo.storage_name ?? repo.storage_id ?? "—"} />
        <Kpi label="Revision" value={<span className="font-mono">{(repo.sha ?? "").slice(0, 12) || "—"}</span>} />
      </div>

      <section className="space-y-3">
        <h2 className="text-base font-medium">Use this repo</h2>
        <div className="rounded-lg border border-border bg-card p-4 space-y-4">
          <div>
            <p className="mb-1.5 text-xs font-medium text-muted-foreground">1 · Point HF tooling at this gateway</p>
            <CopyBlock code={envSnippet} />
          </div>
          <div>
            <p className="mb-1.5 text-xs font-medium text-muted-foreground">Pull (download)</p>
            <CopyBlock code={pullSnippet} />
          </div>
          <div>
            <p className="mb-1.5 text-xs font-medium text-muted-foreground">Push (upload)</p>
            <CopyBlock code={pushSnippet} />
          </div>
        </div>
      </section>

      <section className="space-y-3">
        <div className="flex items-baseline gap-3 border-b border-border pb-2">
          <h2 className="text-base font-medium">Files</h2>
          <span className="text-xs text-muted-foreground">
            {files.length} {files.length === 1 ? "file" : "files"} on <span className="font-mono">{repo.prefix}</span>
          </span>
        </div>
        {files.length === 0 ? (
          <p className="px-2 py-8 text-center text-sm text-muted-foreground">
            No files yet — <span className="font-mono">hf upload</span> to this repo, then it&apos;ll appear here.
          </p>
        ) : (
          <ul className="divide-y divide-border rounded-lg border border-border">
            {files.map((f) => (
              <li key={f.path} className="flex items-center gap-3 px-4 py-2.5 text-sm">
                <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                <span className="min-w-0 flex-1 truncate font-mono text-xs">{f.path}</span>
                {f.lfs && <Badge variant="outline" className="shrink-0">LFS</Badge>}
                <span className="shrink-0 tabular-nums text-xs text-muted-foreground">{fmtBytes(f.size)}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete repo</DialogTitle>
            <DialogDescription>
              Remove <span className="font-mono">{repo.full_id}</span> from the catalog.
            </DialogDescription>
          </DialogHeader>
          <label className="flex items-start gap-2 rounded-md border border-border px-3 py-2 text-sm">
            <Checkbox checked={wipe} onCheckedChange={(v) => setWipe(v === true)} className="mt-0.5" />
            <span>
              Also delete all files from storage (
              <span className="font-mono text-xs">{repo.prefix}</span>). Permanent.
            </span>
          </label>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteOpen(false)} disabled={busy}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={doDelete} disabled={busy}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
