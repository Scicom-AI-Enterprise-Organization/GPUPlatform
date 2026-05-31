"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Trash2 } from "lucide-react";
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

export function DeleteButton({ id, name }: { id: string; name: string }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDelete = async () => {
    setBusy(true);
    setError(null);
    try {
      await gateway.deleteDataset(id);
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
      <Dialog open={open} onOpenChange={(o) => !o && setOpen(false)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete dataset</DialogTitle>
            <DialogDescription>
              Delete <span className="font-medium text-foreground">{name}</span>? This removes the
              dataset record. Files already written to storage are not deleted.
            </DialogDescription>
          </DialogHeader>
          {error && (
            <p className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </p>
          )}
          <DialogFooter>
            <Button variant="ghost" onClick={() => setOpen(false)} disabled={busy}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={onDelete} disabled={busy}>
              {busy ? "Deleting…" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
