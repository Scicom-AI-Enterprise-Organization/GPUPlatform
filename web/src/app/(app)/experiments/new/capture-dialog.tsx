"use client";

// Capture requests INTO a platform dataset.
//
// Experiments owns no dataset store, so a capture writes a real `kind=upload`
// chat dataset into /datasets — browsable, publishable, packable, and reusable
// by anything else on the platform. This dialog is just the front end for that.

import { useState } from "react";
import { toast } from "sonner";
import { Cloud, Database, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { LangfuseGeneration, StorageRecord } from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export function CaptureDialog({
  open,
  onOpenChange,
  storages,
  onCaptured,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  storages: StorageRecord[];
  onCaptured: (datasetId: string, name: string, nRows: number) => void;
}) {
  const [name, setName] = useState("");
  const [storageId, setStorageId] = useState(storages[0]?.id ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const common = { name: name.trim(), storage_id: storageId };
  const ready = !!name.trim() && !!storageId;

  function done(res: { dataset_id: string; name: string; n_rows: number }) {
    toast.success(`Captured ${res.n_rows} request${res.n_rows === 1 ? "" : "s"} into ${res.name}`);
    onCaptured(res.dataset_id, res.name, res.n_rows);
    onOpenChange(false);
    setName("");
    setError(null);
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Capture requests into a dataset</DialogTitle>
          <DialogDescription>
            Creates a chat dataset in{" "}
            <span className="font-medium text-foreground">Datasets</span> and selects it for this
            experiment. It behaves like any other dataset afterwards.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="cap-name">Dataset name</Label>
            <Input
              id="cap-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="billing-livechat-captures"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="cap-storage">S3 storage</Label>
            {storages.length === 0 ? (
              <p className="pt-2 text-xs text-muted-foreground">
                No S3 storage yet — add one at{" "}
                <a href="/storage/new" className="underline underline-offset-2">
                  Storage → New
                </a>
                .
              </p>
            ) : (
              <Select value={storageId} onValueChange={setStorageId}>
                <SelectTrigger id="cap-storage">
                  <SelectValue placeholder="Pick a storage" />
                </SelectTrigger>
                <SelectContent>
                  {storages.map((s) => (
                    <SelectItem key={s.id} value={s.id}>
                      {s.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
        </div>

        <Tabs defaultValue="platform" className="mt-2">
          <TabsList>
            <TabsTrigger value="platform">
              <Database className="h-3.5 w-3.5" />
              Served traffic
            </TabsTrigger>
            <TabsTrigger value="langfuse">
              <Cloud className="h-3.5 w-3.5" />
              Langfuse trace
            </TabsTrigger>
          </TabsList>

          <TabsContent value="platform" className="mt-3">
            <PlatformCapture
              common={common}
              ready={ready}
              busy={busy}
              setBusy={setBusy}
              setError={setError}
              onDone={done}
            />
          </TabsContent>
          <TabsContent value="langfuse" className="mt-3">
            <LangfuseCapture
              common={common}
              ready={ready}
              busy={busy}
              setBusy={setBusy}
              setError={setError}
              onDone={done}
            />
          </TabsContent>
        </Tabs>

        {error && (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error}
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

type Shared = {
  common: { name: string; storage_id: string };
  ready: boolean;
  busy: boolean;
  setBusy: (b: boolean) => void;
  setError: (e: string | null) => void;
  onDone: (r: { dataset_id: string; name: string; n_rows: number }) => void;
};

function PlatformCapture({ common, ready, busy, setBusy, setError, onDone }: Shared) {
  const [appId, setAppId] = useState("");
  const [search, setSearch] = useState("");
  const [limit, setLimit] = useState(20);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      onDone(
        await gateway.capturePlatformRequests({
          ...common,
          app_id: appId || undefined,
          search: search || undefined,
          limit,
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        Requests this platform already served. Serverless endpoints store the full body, so a real
        production request becomes a replayable row directly. Proxy-mode traffic records only model and
        usage (deliberately slim, for throughput), so those rows carry no body.
      </p>
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="space-y-1.5">
          <Label>Endpoint id</Label>
          <Input value={appId} onChange={(e) => setAppId(e.target.value)} placeholder="any" />
        </div>
        <div className="space-y-1.5">
          <Label>Payload contains</Label>
          <Input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="any" />
        </div>
        <div className="space-y-1.5">
          <Label>Max requests</Label>
          <Input
            type="number"
            min={1}
            max={500}
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value) || 1)}
          />
        </div>
      </div>
      <Button onClick={() => void run()} disabled={busy || !ready}>
        {busy && <Loader2 className="h-4 w-4 animate-spin" />}
        Capture most recent
      </Button>
    </div>
  );
}

function LangfuseCapture({ common, ready, busy, setBusy, setError, onDone }: Shared) {
  const [url, setUrl] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [publicKey, setPublicKey] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [gens, setGens] = useState<LangfuseGeneration[] | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());

  const creds = {
    url,
    base_url: baseUrl || undefined,
    public_key: publicKey || undefined,
    secret_key: secretKey || undefined,
  };

  async function preview() {
    setBusy(true);
    setError(null);
    try {
      const res = await gateway.previewLangfuseTrace(creds);
      setGens(res.generations);
      const pre = new Set<string>();
      if (res.suggested_observation) pre.add(res.suggested_observation);
      else res.generations.filter((g) => g.replayable).slice(0, 1).forEach((g) => pre.add(g.id));
      setPicked(pre);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setGens(null);
    } finally {
      setBusy(false);
    }
  }

  async function capture() {
    setBusy(true);
    setError(null);
    try {
      onDone(
        await gateway.captureLangfuseTrace({
          ...common,
          ...creds,
          observation_ids: [...picked],
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        Paste a Langfuse URL — a list view with <code className="font-mono">?peek=</code> or{" "}
        <code className="font-mono">?traceId=</code>, a trace permalink, or a bare trace id. The
        gateway pulls the full observation payload via the public API, which the UI&apos;s
        download button strips. Keys are project-scoped.
      </p>
      <div className="space-y-1.5">
        <Label>Trace URL</Label>
        <Input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://langfuse.example.com/project/…/traces?peek=…"
          className="font-mono text-xs"
        />
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="space-y-1.5">
          <Label>Base URL</Label>
          <Input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)}
                 placeholder="https://cloud.langfuse.com" />
        </div>
        <div className="space-y-1.5">
          <Label>Public key</Label>
          <Input value={publicKey} onChange={(e) => setPublicKey(e.target.value)} placeholder="pk-lf-…" />
        </div>
        <div className="space-y-1.5">
          <Label>Secret key</Label>
          <Input type="password" value={secretKey} onChange={(e) => setSecretKey(e.target.value)}
                 placeholder="sk-lf-…" />
        </div>
      </div>

      <Button variant="outline" onClick={() => void preview()} disabled={busy || !url.trim()}>
        {busy && <Loader2 className="h-4 w-4 animate-spin" />}
        Fetch trace
      </Button>

      {gens && (
        <div className="space-y-2">
          <p className="text-sm font-medium">
            {gens.length} generation{gens.length === 1 ? "" : "s"}
          </p>
          <ul className="max-h-56 space-y-1 overflow-y-auto scrollbar-thin">
            {gens.map((g) => (
              <li
                key={g.id}
                className={cn(
                  "flex items-center gap-2 rounded-md border px-2 py-1.5 text-sm",
                  g.replayable ? "border-border" : "border-border/50 opacity-60",
                )}
              >
                <input
                  type="checkbox"
                  disabled={!g.replayable}
                  checked={picked.has(g.id)}
                  onChange={(e) => {
                    const next = new Set(picked);
                    if (e.target.checked) next.add(g.id);
                    else next.delete(g.id);
                    setPicked(next);
                  }}
                  className="h-3.5 w-3.5"
                />
                <span className="min-w-0 flex-1 truncate">{g.name ?? g.id}</span>
                <span className="shrink-0 text-xs text-muted-foreground">
                  {g.model ?? "—"} · {g.n_messages} msg
                  {!g.replayable && " · not replayable"}
                </span>
              </li>
            ))}
          </ul>
          <Button onClick={() => void capture()} disabled={busy || !ready || picked.size === 0}>
            {busy && <Loader2 className="h-4 w-4 animate-spin" />}
            Capture {picked.size} request{picked.size === 1 ? "" : "s"}
          </Button>
        </div>
      )}
    </div>
  );
}
