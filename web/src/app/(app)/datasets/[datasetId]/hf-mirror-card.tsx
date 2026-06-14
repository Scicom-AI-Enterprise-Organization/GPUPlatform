"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Check, Copy, Loader2, Package } from "lucide-react";
import { gateway } from "@/lib/gateway";
import type { DatasetRecord } from "@/lib/types";
import { Button } from "@/components/ui/button";

const PUBLISHABLE = new Set(["s3", "upload", "tts_packed"]);
const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

function Copyable({ code }: { code: string }) {
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

/** "Use with HuggingFace" — publish an S3-backed dataset to the platform's HF
 * mirror, then show pull snippets. Reads NEXT_PUBLIC_GATEWAY_URL for the endpoint. */
export function HfMirrorCard({ dataset }: { dataset: DatasetRecord }) {
  const router = useRouter();
  const endpoint = `${GATEWAY.replace(/\/$/, "")}/hf`;
  const [repoId, setRepoId] = useState<string | null>(dataset.catalog_repo_id ?? null);
  const [fullId, setFullId] = useState<string | null>(null);
  const [publishing, setPublishing] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!repoId) return;
    let cancel = false;
    gateway
      .getCatalogRepo(repoId)
      .then((r) => !cancel && setFullId(r.full_id))
      .catch(() => !cancel && setFullId(null));
    return () => {
      cancel = true;
    };
  }, [repoId]);

  async function publish() {
    setPublishing(true);
    setErr(null);
    try {
      const r = await gateway.publishDataset(dataset.id);
      setRepoId(r.repo_id);
      setFullId(r.full_id);
      router.refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setPublishing(false);
    }
  }

  // S3-backed directly, or an hf/label dataset that's been materialised to S3
  // (publish then delegates to that twin).
  const publishable = PUBLISHABLE.has(dataset.kind) || !!dataset.audio_dataset_id;
  // An HF-source dataset is already usable from huggingface.co directly.
  const hfRepo = dataset.hf_repo ?? dataset.source_hf_repo ?? null;
  const hfRepoSnippet = `# Already on huggingface.co — use it directly:
from datasets import load_dataset
load_dataset("${hfRepo}")

# or
hf download ${hfRepo} --repo-type dataset`;
  const id = fullId ?? "<ns>/<name>";
  const snippet = `export HF_ENDPOINT=${endpoint}
export HF_TOKEN=sgpu_…   # your platform API key

# CLI
hf download ${id} --repo-type dataset

# Python
from huggingface_hub import snapshot_download
snapshot_download("${id}", repo_type="dataset")`;

  return (
    <section className="rounded-lg border border-border bg-card p-5">
      <div className="mb-1 flex items-center gap-2 text-sm font-medium">
        <Package className="h-4 w-4 text-muted-foreground" />
        Use with HuggingFace
      </div>

      {repoId ? (
        <>
          <p className="mb-3 text-xs text-muted-foreground">
            Published to the HF mirror as{" "}
            <span className="font-mono text-foreground">{fullId ?? "…"}</span>. Pull it with the
            standard <span className="font-mono">hf</span> CLI / library:
          </p>
          <Copyable code={snippet} />
        </>
      ) : publishable ? (
        <>
          <p className="mb-3 text-xs text-muted-foreground">
            Make this dataset HuggingFace-compatible — publish it as a hosted dataset repo on
            the mirror, then pull it anywhere with <span className="font-mono">hf download … --repo-type dataset</span>.
          </p>
          {err && <p className="mb-2 text-sm text-destructive">{err}</p>}
          <Button onClick={publish} disabled={publishing} size="sm">
            {publishing && <Loader2 className="h-4 w-4 animate-spin" />}
            Publish to HF mirror
          </Button>
        </>
      ) : hfRepo ? (
        <>
          <p className="mb-3 text-xs text-muted-foreground">
            This dataset lives on huggingface.co (<span className="font-mono">{hfRepo}</span>) — it&apos;s
            already HuggingFace-usable from there directly (no mirror needed). To self-host it on the
            mirror instead, materialise it to S3 (Transform → S3), then publish.
          </p>
          <Copyable code={hfRepoSnippet} />
        </>
      ) : (
        <p className="text-xs text-muted-foreground">
          This is a <span className="font-mono">{dataset.kind}</span> dataset with no files in your
          storage yet. Materialise it to S3 (Transform → S3) first, then publish to the HF mirror.
        </p>
      )}
    </section>
  );
}
