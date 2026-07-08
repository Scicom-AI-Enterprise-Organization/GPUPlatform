"use client";

import { useEffect, useState, useTransition } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Check, Copy, Download, Globe, Lock, Octagon, Pencil, RotateCw, Trash2, X } from "lucide-react";
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
import { formatCostUSD, formatRateUSD, useLiveCost } from "@/lib/cost";
import { BurnFlame } from "@/components/burn-flame";
import { cn } from "@/lib/utils";
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
  queued: "border border-border bg-muted text-muted-foreground",
  running: "border border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400",
  done: "border border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  failed: "border border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
  cancelled: "border border-border bg-muted text-muted-foreground",
};

type BenchTab = (typeof TABS)[number]["value"];
const BENCH_TAB_VALUES = TABS.map((t) => t.value) as readonly string[];

export function BenchmarkDetail({ bench: initial, isAdmin = false }: { bench: BenchmarkRecord; isAdmin?: boolean }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const initialTab: BenchTab = (() => {
    const t = searchParams.get("tab");
    return t && BENCH_TAB_VALUES.includes(t) ? (t as BenchTab) : "logs";
  })();
  const [bench, setBench] = useState(initial);
  // is_owner is undefined on older payloads — treat as owned for back-compat.
  const owned = bench.is_owner ?? true;
  // Admins may rename anyone's benchmark (the gateway authorizes it); delete /
  // visibility stay owner-only.
  const canRename = owned || isAdmin;
  const [tab, setTabState] = useState<BenchTab>(initialTab);
  const setTab = (v: BenchTab) => {
    setTabState(v);
    const params = new URLSearchParams(searchParams.toString());
    params.set("tab", v);
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
  };
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [confirmTerminate, setConfirmTerminate] = useState(false);
  const [pending, startTransition] = useTransition();
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [terminateError, setTerminateError] = useState<string | null>(null);
  const [rerunError, setRerunError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [editingName, setEditingName] = useState(false);
  const [nameDraft, setNameDraft] = useState(bench.name);
  const [renaming, setRenaming] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);
  const inFlight = bench.status === "queued" || bench.status === "running";

  function startRename() {
    setNameDraft(bench.name);
    setRenameError(null);
    setEditingName(true);
  }

  function cancelRename() {
    setEditingName(false);
    setNameDraft(bench.name);
    setRenameError(null);
  }

  async function handleRename() {
    const name = nameDraft.trim();
    if (!name || name === bench.name) {
      cancelRename();
      return;
    }
    setRenameError(null);
    setRenaming(true);
    try {
      const next = await gateway.renameBenchmark(bench.id, name);
      setBench(next);
      setEditingName(false);
    } catch (e) {
      setRenameError(e instanceof Error ? e.message : String(e));
    } finally {
      setRenaming(false);
    }
  }

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
    setDeleteError(null);
    startTransition(async () => {
      try {
        await gateway.deleteBenchmark(bench.id);
        router.push("/benchmark");
      } catch (e) {
        setDeleteError(e instanceof Error ? e.message : String(e));
      }
    });
  }

  function handleTerminate() {
    setTerminateError(null);
    startTransition(async () => {
      try {
        await gateway.terminateBenchmark(bench.id);
        setConfirmTerminate(false);
        // Refresh so the status pill and KPIs flip to cancelled immediately;
        // cleanup log lines stream in via the existing log tab.
        try {
          const next = await gateway.getBenchmark(bench.id);
          setBench(next);
        } catch {
          // ignore — next auto-refresh tick will pick it up
        }
      } catch (e) {
        setTerminateError(e instanceof Error ? e.message : String(e));
      }
    });
  }

  // Self-contained export (results + config + GPU/serve metadata + embedded S3
  // files). Mirrors the Files tab action but lives in the header so it's
  // reachable from any tab.
  async function handleExport() {
    setExporting(true);
    try {
      const data = await gateway.exportBenchmark(bench.id);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${bench.id}.benchmark.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      toast.success("Benchmark exported", { duration: 3000 });
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setExporting(false);
    }
  }

  function handleTogglePublic() {
    const next = !bench.is_public;
    startTransition(async () => {
      try {
        const updated = await gateway.setBenchmarkVisibility(bench.id, next);
        setBench(updated);
      } catch {
        // best-effort — the button reflects server state on next load
      }
    });
  }

  // Re-run: recreate this benchmark from its saved config (same provider /
  // storage / env / cleanup flag) and jump to the new run. Recovers a run that
  // failed or was orphaned by a gateway restart in one click.
  function handleRerun() {
    setRerunError(null);
    startTransition(async () => {
      try {
        const created = await gateway.createBenchmark({
          name: bench.name,
          config_yaml: bench.config_yaml,
          provider_id: bench.provider_id ?? null,
          storage_id: bench.storage_id ?? null,
          ...(bench.cleanup_model != null ? { cleanup_model: bench.cleanup_model } : {}),
          ...(bench.env_vars ? { env_vars: bench.env_vars } : {}),
          ...(bench.visible_devices ? { visible_devices: bench.visible_devices } : {}),
          ...(bench.hf_token_secret ? { hf_token_secret: bench.hf_token_secret } : {}),
        });
        router.push(`/benchmark/${encodeURIComponent(created.id)}`);
      } catch (e) {
        setRerunError(e instanceof Error ? e.message : String(e));
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
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between sm:gap-4">
          <div>
            <div className="flex items-center gap-2">
              {editingName ? (
                <>
                  <input
                    autoFocus
                    value={nameDraft}
                    onChange={(e) => setNameDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleRename();
                      else if (e.key === "Escape") cancelRename();
                    }}
                    disabled={renaming}
                    maxLength={200}
                    className="h-9 w-72 max-w-full rounded-md border border-input bg-background px-2 text-xl font-semibold tracking-tight outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
                    aria-label="Benchmark name"
                  />
                  <Button
                    size="icon-sm"
                    variant="ghost"
                    onClick={handleRename}
                    disabled={renaming}
                    aria-label="Save name"
                  >
                    <Check className="h-4 w-4" />
                  </Button>
                  <Button
                    size="icon-sm"
                    variant="ghost"
                    onClick={cancelRename}
                    disabled={renaming}
                    aria-label="Cancel rename"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </>
              ) : (
                <>
                  <h1 className="text-xl font-semibold tracking-tight">{bench.name}</h1>
                  {canRename && (
                    <button
                      type="button"
                      onClick={startRename}
                      className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
                      title="Rename benchmark"
                      aria-label="Rename benchmark"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                  )}
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${
                      STATUS_STYLES[bench.status] ?? STATUS_STYLES.queued
                    }`}
                  >
                    {bench.status}
                  </span>
                  {bench.is_public && (
                    <span
                      className="inline-flex items-center gap-1 rounded border border-sky-500/40 bg-sky-500/10 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-sky-700 dark:text-sky-400"
                      title="Public — visible to everyone"
                    >
                      <Globe className="h-3 w-3" /> Public
                    </span>
                  )}
                </>
              )}
            </div>
            {renameError && <p className="mt-1 text-xs text-destructive">{renameError}</p>}
            <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
              <span className="font-mono">{bench.id}</span>
              <span>·</span>
              <span>by {bench.created_by}</span>
              <span>·</span>
              <span>{new Date(bench.created_at).toLocaleString()}</span>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 sm:shrink-0 sm:justify-end">
            {owned && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleTogglePublic}
                disabled={pending}
                title={
                  bench.is_public
                    ? "Make private — hide from other users"
                    : "Make public — anyone can view (read-only)"
                }
              >
                {bench.is_public ? (
                  <>
                    <Lock className="h-4 w-4" />
                    Make private
                  </>
                ) : (
                  <>
                    <Globe className="h-4 w-4" />
                    Make public
                  </>
                )}
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                router.push(`/benchmark/new?from=${encodeURIComponent(bench.id)}`)
              }
            >
              <Copy className="h-4 w-4" />
              Duplicate
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleExport}
              disabled={exporting}
              title="Download a self-contained JSON (results + config + GPU/serve metadata + files)"
            >
              <Download className="h-4 w-4" />
              {exporting ? "Exporting…" : "Export"}
            </Button>
            {!inFlight && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleRerun}
                disabled={pending}
                title={rerunError ?? "Re-run this benchmark with the same config"}
                className={rerunError ? "text-destructive hover:text-destructive" : undefined}
              >
                <RotateCw className="h-4 w-4" />
                {pending ? "Re-running…" : "Re-run"}
              </Button>
            )}
            {inFlight && owned && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setConfirmTerminate(true)}
                className="text-destructive hover:text-destructive"
              >
                <Octagon className="h-4 w-4" />
                Terminate
              </Button>
            )}
            {owned && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setConfirmDelete(true)}
                className="text-destructive hover:text-destructive"
              >
                <Trash2 className="h-4 w-4" />
                Delete
              </Button>
            )}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-5">
          <Kpi label="Status" value={bench.status} />
          <Kpi label="Duration" value={dur != null ? `${dur}s` : "—"} />
          <Kpi label="Exit code" value={bench.exit_code != null ? String(bench.exit_code) : "—"} />
          <Kpi
            label="Result"
            value={bench.result_json ? "Yes" : "—"}
          />
          <CostKpi bench={bench} />
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
          <TabsContent value="parameters"><ParametersTab bench={bench} canEdit={canRename} onBenchChange={setBench} /></TabsContent>
          <TabsContent value="files"><FilesTab bench={bench} /></TabsContent>
        </Tabs>
      </div>

      <Dialog
        open={confirmTerminate}
        onOpenChange={(o) => {
          setConfirmTerminate(o);
          if (!o) setTerminateError(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Terminate benchmark?</DialogTitle>
            <DialogDescription>
              Stops the run, kills any remote bench process over SSH, removes
              the downloaded model from the VM, and tears down the RunPod pod
              if one was provisioned. The row stays so logs remain viewable.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {terminateError && (
              <p className="mr-auto text-sm text-destructive">{terminateError}</p>
            )}
            <Button variant="outline" onClick={() => setConfirmTerminate(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleTerminate} disabled={pending}>
              {pending ? "Terminating…" : "Terminate"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={confirmDelete}
        onOpenChange={(o) => {
          setConfirmDelete(o);
          if (!o) setDeleteError(null);
        }}
      >
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
            {deleteError && (
              <p className="mr-auto text-sm text-destructive">{deleteError}</p>
            )}
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

function CostKpi({ bench }: { bench: BenchmarkRecord }) {
  const live = useLiveCost(bench.started_at, bench.ended_at, bench.cost_per_hr);
  const isBurning =
    bench.status === "running" && bench.cost_per_hr != null && bench.ended_at == null;
  return (
    <div>
      <div className="text-xs text-muted-foreground">
        Cost {isBurning ? "(live)" : ""}
      </div>
      <div
        className={cn(
          "mt-0.5 flex items-center gap-1.5 text-lg font-semibold tabular-nums",
          isBurning && "text-amber-600 dark:text-amber-400",
        )}
      >
        {isBurning && <BurnFlame size="h-4 w-4" />}
        {formatCostUSD(live)}
      </div>
      <div className="text-[10px] text-muted-foreground">
        {bench.cost_per_hr != null ? `at ${formatRateUSD(bench.cost_per_hr)}` : "—"}
      </div>
    </div>
  );
}
