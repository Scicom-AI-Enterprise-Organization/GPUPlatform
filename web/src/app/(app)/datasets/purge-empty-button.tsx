"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Loader2, Trash2 } from "lucide-react";
import { gateway } from "@/lib/gateway";
import type { DatasetRecord } from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

// The _page endpoint caps limit at 100, so walk pages to cover ALL datasets.
const PAGE = 100;

/** Header action beside "New dataset": find every dataset whose row count is
 * exactly 0 (empty/broken registrations — NOT null, which just means the count
 * is unknown) and delete their records. S3 files are left in place (purge=false)
 * since a 0-row dataset has nothing meaningful stored. */
export function PurgeEmptyButton({ scope }: { scope: "mine" | "all" }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [empty, setEmpty] = useState<DatasetRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  const scan = useCallback(async () => {
    setScanning(true);
    setError(null);
    setEmpty([]);
    try {
      const found: DatasetRecord[] = [];
      let offset = 0;
      for (;;) {
        const res = await gateway.listDatasetsPage({ scope, limit: PAGE, offset });
        found.push(...res.items.filter((d) => d.num_rows === 0));
        offset += res.items.length;
        if (res.items.length < PAGE || offset >= res.total) break;
      }
      setEmpty(found);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setScanning(false);
    }
  }, [scope]);

  const openDialog = () => {
    setOpen(true);
    void scan();
  };

  const onPurge = async () => {
    setDeleting(true);
    setError(null);
    try {
      const results = await Promise.allSettled(
        empty.map((d) => gateway.deleteDataset(d.id)),
      );
      const failed = results.filter((r) => r.status === "rejected").length;
      const ok = empty.length - failed;
      if (ok > 0) toast.success(`Deleted ${ok} empty dataset${ok === 1 ? "" : "s"}`);
      if (failed > 0) {
        setError(`${failed} dataset${failed === 1 ? "" : "s"} could not be deleted.`);
        setDeleting(false);
        void scan(); // refresh the list so the user sees what's left
        return;
      }
      setOpen(false);
      // router.refresh() alone won't drop the stale cards held in DatasetsList's
      // client state, so reload to get a fully consistent list + header count.
      router.refresh();
      window.location.reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setDeleting(false);
    }
  };

  return (
    <>
      <Button size="sm" variant="outline" onClick={openDialog}>
        <Trash2 className="h-4 w-4" />
        Purge empty
      </Button>

      <Dialog
        open={open}
        onOpenChange={(o) => {
          if (!deleting && !o) {
            setOpen(false);
            setError(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Purge empty datasets</DialogTitle>
            <DialogDescription>
              Deletes the records of datasets with <span className="font-medium text-foreground">0 rows</span>
              {scope === "all" ? " across all users" : ""}. Files in storage are left in place.
            </DialogDescription>
          </DialogHeader>

          {scanning ? (
            <div className="flex items-center gap-2 py-4 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" /> Scanning for empty datasets…
            </div>
          ) : empty.length === 0 ? (
            <p className="py-2 text-sm text-muted-foreground">No empty datasets to purge.</p>
          ) : (
            <div className="max-h-56 space-y-1 overflow-y-auto rounded-md border border-border bg-muted/30 p-2 text-sm">
              {empty.map((d) => (
                <div key={d.id} className="flex items-center justify-between gap-3">
                  <span className="truncate font-medium">{d.name}</span>
                  <span className="shrink-0 font-mono text-xs text-muted-foreground">{d.kind} · {d.id}</span>
                </div>
              ))}
            </div>
          )}

          {error && (
            <p className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </p>
          )}

          <DialogFooter>
            <Button variant="ghost" onClick={() => setOpen(false)} disabled={deleting}>
              {empty.length === 0 && !scanning ? "Close" : "Cancel"}
            </Button>
            {empty.length > 0 && (
              <Button variant="destructive" onClick={onPurge} disabled={scanning || deleting}>
                {deleting ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                {deleting ? "Deleting…" : `Delete ${empty.length} empty dataset${empty.length === 1 ? "" : "s"}`}
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
