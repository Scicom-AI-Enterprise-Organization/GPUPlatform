"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { Loader2, MoreHorizontal, Trash2 } from "lucide-react";
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
import type { AppRecord } from "@/lib/types";
import { avatarFor } from "@/lib/avatar";
import { cn } from "@/lib/utils";
import { deleteEndpoint } from "../actions";
import { OverviewTab } from "./tabs/overview";
import { RequestsTab } from "./tabs/requests";
import { QueueTab } from "./tabs/queue";
import { WorkersTab } from "./tabs/workers";

const TABS = [
  { value: "overview", label: "Overview" },
  { value: "playground", label: "Playground" },
  { value: "queue", label: "Queue" },
  { value: "workers", label: "Workers" },
] as const;

export function EndpointDetail({ app }: { app: AppRecord }) {
  const router = useRouter();
  const [tab, setTab] = useState<(typeof TABS)[number]["value"]>("overview");
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [pending, startTransition] = useTransition();
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const avatar = avatarFor(app.name);

  function handleDelete() {
    setDeleteError(null);
    startTransition(async () => {
      const res = await deleteEndpoint(app.app_id);
      if (!res.ok) {
        setDeleteError(res.error);
        return;
      }
      router.push("/serverless");
    });
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="border-b border-border bg-sidebar/40 px-6 pt-4 lg:px-10">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-muted/60 text-lg font-semibold text-muted-foreground">
              {avatar.letter}
            </div>
            <div>
              <h1 className="text-xl font-semibold tracking-tight">{app.name}</h1>
              <div className="mt-0.5 flex items-center gap-3 text-xs text-muted-foreground">
                <span className="font-mono">{app.app_id}</span>
                <span>·</span>
                <span className="font-mono">{app.model}</span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
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

        <KpiBar app={app} />

        <Tabs value={tab} onValueChange={(v) => setTab(v as typeof tab)} className="mt-2">
          <TabsList variant="line" className="bg-transparent">
            {TABS.map((t) => (
              <TabsTrigger key={t.value} value={t.value}>
                {t.label}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 scrollbar-thin">
        <Tabs value={tab} onValueChange={(v) => setTab(v as typeof tab)}>
          <TabsContent value="overview"><OverviewTab app={app} /></TabsContent>
          <TabsContent value="playground"><RequestsTab /></TabsContent>
          <TabsContent value="queue"><QueueTab app={app} /></TabsContent>
          <TabsContent value="workers"><WorkersTab app={app} /></TabsContent>
        </Tabs>
      </div>

      <Dialog
        open={confirmDelete}
        onOpenChange={(o) => {
          setConfirmDelete(o);
          if (!o) setDeleteError(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete {app.name}?</DialogTitle>
            <DialogDescription>
              All workers will be drained and the queue cleared. This can&apos;t be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {deleteError && (
              <p className="mr-auto text-sm text-destructive">{deleteError}</p>
            )}
            <Button variant="ghost" onClick={() => setConfirmDelete(false)} disabled={pending}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDelete} disabled={pending}>
              {pending && <Loader2 className="h-4 w-4 animate-spin" />}
              Delete endpoint
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function KpiBar({ app }: { app: AppRecord }) {
  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
      <Kpi value="0" label="running workers" />
      <Kpi value="0" label="jobs in progress" />
      <Kpi value="0" label="jobs waiting in queue" />
      <Kpi
        value={String(Math.min(1, app.autoscaler.max_containers))}
        label="active worker recommended"
      />
    </div>
  );
}

function Kpi({ value, label }: { value: string; label: string }) {
  return (
    <span className="text-muted-foreground">
      <span className="font-mono text-foreground">{value}</span> {label}
    </span>
  );
}
