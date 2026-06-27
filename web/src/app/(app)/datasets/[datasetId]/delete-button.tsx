"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { gateway } from "@/lib/gateway";
import type { DatasetKind } from "@/lib/types";

// Kinds whose files live in OUR S3 storage (so "purge" can delete them). hf is
// pushed to a HuggingFace repo and label lives on the labeling platform — for
// those there's nothing in storage to purge.
const PURGEABLE_KINDS: DatasetKind[] = ["s3", "tts_packed", "llm_packed", "upload"];

export function DeleteButton({ id, name, kind }: { id: string; name: string; kind: DatasetKind }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [purge, setPurge] = useState(false);
  const [confirmName, setConfirmName] = useState("");

  const canPurge = PURGEABLE_KINDS.includes(kind);
  // Purging is destructive + irreversible → gate it behind typing the exact name.
  const nameConfirmed = confirmName.trim() === name;
  const blocked = busy || (purge && !nameConfirmed);

  const reset = () => {
    setOpen(false);
    setError(null);
    setPurge(false);
    setConfirmName("");
  };

  const onDelete = async () => {
    setBusy(true);
    setError(null);
    try {
      await gateway.deleteDataset(id, purge);
      router.push("/datasets");
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  return (
    <>
      <Button variant="outline" size="sm" onClick={() => setOpen(true)}>
        <Trash2 className="h-4 w-4" />
        Delete
      </Button>
      <Dialog open={open} onOpenChange={(o) => !o && reset()}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete dataset</DialogTitle>
            <DialogDescription>
              Delete <span className="font-medium text-foreground">{name}</span>? This removes the
              dataset record.{" "}
              {purge ? (
                <span className="text-destructive">
                  Its files in S3 storage will also be <span className="font-medium">permanently deleted</span>.
                </span>
              ) : (
                <>Files already written to storage are not deleted.</>
              )}
            </DialogDescription>
          </DialogHeader>

          {canPurge && (
            <div className="space-y-3">
              <label className="flex items-start gap-2 text-sm">
                <Checkbox
                  checked={purge}
                  onCheckedChange={(v) => {
                    setPurge(v === true);
                    setConfirmName("");
                  }}
                  disabled={busy}
                  className="mt-0.5"
                />
                <span>
                  <span className="font-medium">Also delete the files in storage (S3)</span> — removes this
                  dataset&apos;s objects under its storage prefix. This cannot be undone.
                </span>
              </label>

              {purge && (
                <div className="space-y-1.5">
                  {/* Plain text (not a <label>) so the name highlights + copies normally. */}
                  <p className="text-xs text-muted-foreground">
                    Type <span className="font-mono text-foreground">{name}</span> to confirm
                  </p>
                  <Input
                    id="del-confirm"
                    value={confirmName}
                    onChange={(e) => setConfirmName(e.target.value)}
                    placeholder={name}
                    autoComplete="off"
                    aria-label="Type the dataset name to confirm deletion"
                    disabled={busy}
                    className="text-sm"
                  />
                </div>
              )}
            </div>
          )}

          {error && (
            <p className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </p>
          )}
          <DialogFooter>
            <Button variant="ghost" onClick={reset} disabled={busy}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={onDelete} disabled={blocked}>
              {busy ? "Deleting…" : purge ? "Delete + purge files" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
