"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { CloudUpload, Database, FileAudio, Tags, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { gateway } from "@/lib/gateway";
import type { DatasetKind, DatasetRecord } from "@/lib/types";

const KIND_LABEL: Record<DatasetKind, string> = {
  upload: "upload",
  s3: "s3",
  hf: "huggingface",
  label: "labeling",
};

function KindIcon({ kind }: { kind: DatasetKind }) {
  if (kind === "hf") return <Database className="h-3 w-3" />;
  if (kind === "s3") return <CloudUpload className="h-3 w-3" />;
  if (kind === "label") return <Tags className="h-3 w-3" />;
  return <FileAudio className="h-3 w-3" />;
}

function metaOf(d: DatasetRecord): string {
  if (d.kind === "hf") return d.hf_repo || "—";
  if (!d.metadata_filename && !d.s3_metadata_uri) return "no metadata";
  const fmt = (d.format || "").toUpperCase();
  const rows = typeof d.num_rows === "number" ? `${d.num_rows} rows` : "";
  return [fmt, rows].filter(Boolean).join(" · ") || (d.metadata_filename ?? "—");
}

export function DatasetsList({ items }: { items: DatasetRecord[] }) {
  const router = useRouter();
  const [target, setTarget] = useState<DatasetRecord | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDelete = async () => {
    if (!target) return;
    setError(null);
    setDeleting(true);
    try {
      await gateway.deleteDataset(target.id);
      setTarget(null);
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleting(false);
    }
  };

  return (
    <>
      <div className="overflow-x-auto rounded-md border border-border">
        <table className="w-full text-sm">
          <thead className="border-b border-border bg-muted/30 text-left text-xs text-muted-foreground">
            <tr>
              <th className="px-4 py-2.5 font-medium">Name</th>
              <th className="px-4 py-2.5 font-medium">Source</th>
              <th className="px-4 py-2.5 font-medium">Storage</th>
              <th className="px-4 py-2.5 font-medium">Metadata</th>
              <th className="px-4 py-2.5 font-medium">Owner</th>
              <th className="px-4 py-2.5" />
            </tr>
          </thead>
          <tbody>
            {items.map((d) => (
              <tr key={d.id} className="border-b border-border last:border-0 hover:bg-muted/20">
                <td className="px-4 py-2.5">
                  <Link
                    href={`/datasets/${encodeURIComponent(d.id)}`}
                    className="font-medium hover:underline"
                  >
                    {d.name}
                  </Link>
                  {d.description && (
                    <div className="mt-0.5 max-w-xs truncate text-xs text-muted-foreground">
                      {d.description}
                    </div>
                  )}
                </td>
                <td className="px-4 py-2.5">
                  <span className="inline-flex items-center gap-1.5 rounded-md border border-border bg-muted/40 px-1.5 py-0.5 text-xs text-muted-foreground">
                    <KindIcon kind={d.kind} />
                    {KIND_LABEL[d.kind]}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-muted-foreground">{d.storage_name ?? "—"}</td>
                <td className="px-4 py-2.5 text-muted-foreground">{metaOf(d)}</td>
                <td className="px-4 py-2.5 text-muted-foreground">{d.created_by}</td>
                <td className="px-4 py-2.5 text-right">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setTarget(d)}
                    aria-label={`Delete ${d.name}`}
                  >
                    <Trash2 className="h-4 w-4 text-muted-foreground" />
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Dialog open={!!target} onOpenChange={(o) => !o && setTarget(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete dataset</DialogTitle>
            <DialogDescription>
              Delete <span className="font-medium text-foreground">{target?.name}</span>? This
              removes the dataset record. Files already written to storage are not deleted.
            </DialogDescription>
          </DialogHeader>
          {error && (
            <p className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </p>
          )}
          <DialogFooter>
            <Button variant="ghost" onClick={() => setTarget(null)} disabled={deleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={onDelete} disabled={deleting}>
              {deleting ? "Deleting…" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
