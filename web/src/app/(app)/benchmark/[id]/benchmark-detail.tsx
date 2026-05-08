"use client";

import { useEffect, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { gateway } from "@/lib/gateway";
import type { BenchmarkRecord } from "@/lib/types";
import { LogsTab } from "./tabs/logs";
import { FilesTab } from "./tabs/files";
import { ResultsTab } from "./tabs/results";
import { ParametersTab } from "./tabs/parameters";

const TABS = [
  { value: "logs", label: "Logs" },
  { value: "results", label: "Results" },
  { value: "parameters", label: "Parameters" },
  { value: "files", label: "Files" },
] as const;

const STATUS_STYLES: Record<string, string> = {
  queued: "bg-muted text-muted-foreground",
  running: "bg-blue-500/15 text-blue-400",
  done: "bg-emerald-500/15 text-emerald-400",
  failed: "bg-destructive/15 text-destructive",
  cancelled: "bg-muted text-muted-foreground",
};

export function BenchmarkDetail({ bench: initial }: { bench: BenchmarkRecord }) {
  const router = useRouter();
  const [bench, setBench] = useState(initial);
  const [tab, setTab] = useState<(typeof TABS)[number]["value"]>("logs");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [pending, startTransition] = useTransition();

  // Auto-refresh while not terminal so KPIs (status, exit_code, etc.) stay live.
  useEffect(() => {
    const inFlight = bench.status === "queued" || bench.status === "running";
    if (!inFlight) return;
    const t = setInterval(async () => {
      try {
        const next = await gateway.getBenchmark(bench.id);
        setBench(next);
      } catch {
        // ignore — next tick will retry
      }
    }, 5000);
    return () => clearInterval(t);
  }, [bench.id, bench.status]);

  function handleDelete() {
    startTransition(async () => {
      try {
        await gateway.deleteBenchmark(bench.id);
        toast.success(`Deleted ${bench.id}`);
        router.push("/benchmark");
      } catch (e) {
        toast.error(e instanceof Error ? e.message : String(e));
      }
    });
  }

  const dur = (() => {
    if (!bench.started_at) return null;
    const start = new Date(bench.started_at).getTime();
    const end = bench.ended_at ? new Date(bench.ended_at).getTime() : Date.now();
    return Math.max(0, Math.round((end - start) / 1000));
  })();

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="border-b border-border bg-sidebar/40 px-6 pt-4 lg:px-10">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-semibold tracking-tight">{bench.name}</h1>
              <span
                className={`rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${
                  STATUS_STYLES[bench.status] ?? STATUS_STYLES.queued
                }`}
              >
                {bench.status}
              </span>
            </div>
            <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
              <span className="font-mono">{bench.id}</span>
              <span>·</span>
              <span>by {bench.created_by}</span>
              <span>·</span>
              <span>{new Date(bench.created_at).toLocaleString()}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setConfirmDelete(true)}
              className="text-destructive hover:text-destructive"
            >
              <Trash2 className="h-4 w-4" />
              Delete
            </Button>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Kpi label="Status" value={bench.status} />
          <Kpi label="Duration" value={dur != null ? `${dur}s` : "—"} />
          <Kpi label="Exit code" value={bench.exit_code != null ? String(bench.exit_code) : "—"} />
          <Kpi
            label="Result"
            value={bench.result_json ? "Yes" : "—"}
          />
        </div>

        <Tabs value={tab} onValueChange={(v) => setTab(v as typeof tab)} className="mt-4">
          <TabsList variant="line" className="bg-transparent">
            {TABS.map((t) => (
              <TabsTrigger key={t.value} value={t.value}>
                {t.label}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <Tabs value={tab} onValueChange={(v) => setTab(v as typeof tab)}>
          <TabsContent value="logs"><LogsTab bench={bench} /></TabsContent>
          <TabsContent value="results"><ResultsTab bench={bench} /></TabsContent>
          <TabsContent value="parameters"><ParametersTab bench={bench} /></TabsContent>
          <TabsContent value="files"><FilesTab bench={bench} /></TabsContent>
        </Tabs>
      </div>

      <Dialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete benchmark?</DialogTitle>
            <DialogDescription>
              Kills any running subprocess and removes the benchmark record. S3
              objects are kept. If a RunPod pod is still alive (rare — benchmaq
              terminates on exit), terminate it manually from RunPod&apos;s dashboard.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmDelete(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={pending}>
              {pending ? "Deleting…" : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-0.5 text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}
