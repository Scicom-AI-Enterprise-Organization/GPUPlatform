"use client";

// Generation status for a SYNTHETIC dataset (one whose rows were written by a
// generator model — `gen_spec` is set), shown above the row browser.
//
// It lives on the Rows tab on purpose: the rows are what you came to look at, and
// while a generation is running they are still arriving. The log is collapsed by
// default — a finished corpus shouldn't spend a screen on progress lines — and
// expands to the full tail the gateway keeps.

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { ChevronRight, Loader2, RefreshCw, Sparkles, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type { DatasetRecord } from "@/lib/types";

const POLL_MS = 2000;

/** Pull the newest `[AUTOTRAIN_PROGRESS] … percent=NN` out of the log tail. */
function progressOf(log: string | null | undefined): number | null {
  if (!log) return null;
  const matches = [...log.matchAll(/percent=([\d.]+)/g)];
  const last = matches.at(-1);
  return last ? Math.min(100, Number(last[1])) : null;
}

const STATUS_TONE: Record<string, string> = {
  running: "bg-status-idle/15 text-status-idle",
  done: "bg-status-active/15 text-status-active",
  failed: "bg-status-down/15 text-status-down",
  cancelled: "bg-muted text-muted-foreground",
};

export function GenerationCard({ dataset }: { dataset: DatasetRecord }) {
  const router = useRouter();
  const [d, setD] = useState(dataset);
  const [showLog, setShowLog] = useState(false);
  const [busy, setBusy] = useState(false);
  const [askKey, setAskKey] = useState(false);
  const [key, setKey] = useState("");
  const [rows, setRows] = useState(String(dataset.gen_spec?.n_rows ?? 30));
  const [error, setError] = useState<string | null>(null);
  const wasRunning = useRef(dataset.transform_status === "running");
  const logRef = useRef<HTMLPreElement | null>(null);

  const spec = d.gen_spec ?? {};
  const running = d.transform_status === "running";

  // Follow the tail while a generation is running (mirrors the transform card) —
  // otherwise the newest batch scrolls out of view exactly when it's interesting.
  useEffect(() => {
    if (running && showLog && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [running, showLog, d.transform_log]);

  const refresh = useCallback(async () => {
    try {
      const next = await gateway.getDataset(dataset.id);
      setD(next);
      // The row browser holds server-fetched rows, so when the job finishes the
      // page has to re-read them — otherwise a completed corpus still shows the
      // count it had when the tab was opened.
      if (wasRunning.current && next.transform_status !== "running") {
        wasRunning.current = false;
        router.refresh();
      }
      if (next.transform_status === "running") wasRunning.current = true;
    } catch {
      /* transient — the next tick retries */
    }
  }, [dataset.id, router]);

  useEffect(() => {
    if (!running) return;
    const t = window.setInterval(refresh, POLL_MS);
    return () => window.clearInterval(t);
  }, [running, refresh]);

  const needsKey = !spec.keyless && !spec.api_key_secret;

  async function regenerate(withKey?: string) {
    setBusy(true);
    setError(null);
    try {
      const next = await gateway.regenerateDataset(dataset.id, {
        api_key: withKey?.trim() || null,
        n_rows: Number(rows) || undefined,
      });
      setD(next);
      wasRunning.current = true;
      setAskKey(false);
      setKey("");
      toast.success("Regenerating — rows will refill as batches complete");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function cancel() {
    setBusy(true);
    try {
      await gateway.cancelDatasetTransform(dataset.id);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const pct = running ? progressOf(d.transform_log) : null;
  const target = spec.n_rows ?? 0;

  return (
    <div className="rounded-md border border-border bg-card">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-4 py-3">
        <Sparkles className="h-4 w-4 shrink-0 text-muted-foreground" />
        <span className="text-sm font-medium">Synthetic corpus</span>
        <span className={cn("rounded px-1.5 py-0.5 text-[10px]",
          STATUS_TONE[d.transform_status ?? ""] ?? "bg-muted text-muted-foreground")}>
          {d.transform_status || "—"}
        </span>
        <span className="text-xs text-muted-foreground">
          {d.num_rows ?? 0}
          {running && target ? ` / ${target}` : ""} rows
          {spec.mode ? ` · ${spec.mode}` : ""}
          {spec.model ? ` · ${spec.model}` : ""}
        </span>

        <div className="ml-auto flex items-center gap-2">
          {running ? (
            <Button variant="outline" size="xs" onClick={() => void cancel()} disabled={busy}>
              {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <XCircle className="h-3 w-3" />}
              Cancel
            </Button>
          ) : (
            <>
              <Input
                type="number"
                min={1}
                value={rows}
                onChange={(e) => setRows(e.target.value)}
                className="h-7 w-20 text-xs"
                aria-label="Rows to generate"
              />
              <Button
                variant="outline"
                size="xs"
                disabled={busy}
                onClick={() => (needsKey ? setAskKey((v) => !v) : void regenerate())}
              >
                {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
                Regenerate
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Replacing the corpus is not obviously destructive from a button labelled
          "Regenerate", so say it where the click happens. */}
      {!running && (
        <p className="px-4 pb-2 text-[11px] text-muted-foreground">
          Regenerating <span className="font-medium text-foreground">replaces</span> these rows —
          anything already scored against them was measuring a different corpus.
        </p>
      )}

      {askKey && !running && (
        <div className="flex flex-wrap items-end gap-2 border-t border-border px-4 py-3">
          <div className="space-y-1">
            <Label htmlFor="regen-key" className="text-xs text-muted-foreground">
              Generator API key
            </Label>
            <Input
              id="regen-key"
              type="password"
              autoComplete="off"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder="sgpu_… (never stored)"
              className="h-8 w-72 font-mono text-xs"
            />
          </div>
          <Button size="xs" disabled={busy || !key.trim()} onClick={() => void regenerate(key)}>
            {busy && <Loader2 className="h-3 w-3 animate-spin" />}
            Start
          </Button>
          <span className="text-[11px] text-muted-foreground">
            the original run used a pasted key, which is never stored
          </span>
        </div>
      )}

      {running && pct != null && (
        <div className="px-4 pb-3">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div className="h-full rounded-full bg-status-idle transition-all" style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}

      {error && (
        <div className="mx-4 mb-3 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}

      {d.transform_log && (
        <div className="border-t border-border">
          <button
            type="button"
            onClick={() => setShowLog((v) => !v)}
            className="flex w-full items-center gap-2 px-4 py-2 text-left text-xs font-medium text-muted-foreground hover:bg-muted/40 hover:text-foreground"
          >
            <ChevronRight className={cn("h-3.5 w-3.5 transition-transform", showLog && "rotate-90")} />
            {running ? (
              <span className="flex items-center gap-1.5">
                <Loader2 className="h-3 w-3 animate-spin" />
                Live log
              </span>
            ) : (
              "Generation log"
            )}
          </button>
          {showLog && (
            // Same terminal treatment as the transform card's log — one look for
            // "this is machine output", not two.
            <div className="px-4 pb-3">
              <pre
                ref={logRef}
                className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-zinc-950 p-3 font-mono text-[11px] leading-relaxed text-zinc-200 scrollbar-thin"
              >
                {d.transform_log}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
