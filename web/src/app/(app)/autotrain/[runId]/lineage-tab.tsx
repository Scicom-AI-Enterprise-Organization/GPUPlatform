"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { GitBranch, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineageCard, type LineageNode } from "@/app/(app)/datasets/[datasetId]/lineage-card";

/**
 * What a run actually trained on. "Which dataset id" is not an answer when the corpus
 * is four derivations deep AND `train_splits` then selects only part of it — the id
 * names a container, not the contents. So this leads with the split filter and the
 * original corpora, then offers the full tree per dataset role.
 */

type RunLineage = {
  run_id: string;
  name?: string | null;
  status?: string | null;
  train_splits?: string[] | null;
  datasets: Record<string, { id?: string; tree?: LineageNode | null; roots?: LineageNode[]; same_as?: string }>;
  all_datasets: LineageNode[];
};

export function LineageTab({ runId }: { runId: string }) {
  const [data, setData] = useState<RunLineage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Fetch in the effect body AFTER an await, never synchronously — a sync setState
  // inside an effect cascades renders (react-hooks/set-state-in-effect). `cancelled`
  // also stops a slow response from overwriting a newer one when the id changes.
  const [reloadKey, setReloadKey] = useState(0);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`/api/proxy/v1/training-runs/${encodeURIComponent(runId)}/lineage`);
        if (!res.ok) throw new Error(`lineage failed (${res.status})`);
        const body = (await res.json()) as RunLineage;
        if (!cancelled) {
          setData(body);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "failed to load lineage");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runId, reloadKey]);

  const load = useCallback(() => {
    setLoading(true);
    setReloadKey((k) => k + 1);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin" /> loading lineage…
      </div>
    );
  }
  if (error) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-xs text-destructive">{error}</span>
        <Button variant="outline" size="sm" className="h-6 text-xs" onClick={() => void load()}>
          retry
        </Button>
      </div>
    );
  }
  if (!data) return null;

  const roles = Object.entries(data.datasets);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <GitBranch className="h-4 w-4" /> What this run trained on
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 pt-0 text-xs">
          <div>
            <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Split filter
            </div>
            {data.train_splits?.length ? (
              <div className="flex flex-wrap gap-1">
                {data.train_splits.map((s) => (
                  <Badge key={s} variant="secondary" className="h-5 font-mono text-[10px]">
                    {s}
                  </Badge>
                ))}
              </div>
            ) : (
              // No filter means every non-eval row, which is a materially different
              // corpus from the same dataset with train_splits set — say so explicitly.
              <p className="text-muted-foreground">
                none — trained on every non-eval row of the dataset
              </p>
            )}
          </div>

          <div>
            <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Datasets ({data.all_datasets.length} in the chain)
            </div>
            <ul className="space-y-0.5">
              {roles.map(([role, v]) => (
                <li key={role} className="flex flex-wrap items-center gap-x-2">
                  <Badge variant="outline" className="h-4 px-1 text-[10px]">{role}</Badge>
                  {v.same_as ? (
                    <span className="text-muted-foreground">same as {v.same_as}</span>
                  ) : (
                    <Link href={`/datasets/${v.id}`} className="font-mono text-primary hover:underline">
                      {v.id}
                    </Link>
                  )}
                  {v.roots?.length ? (
                    <span className="text-[11px] text-muted-foreground">
                      from {v.roots.length} original corpus/corpora
                    </span>
                  ) : null}
                </li>
              ))}
            </ul>
          </div>
        </CardContent>
      </Card>

      {roles
        .filter(([, v]) => v.id && !v.same_as)
        .map(([role, v]) => (
          <div key={role} className="space-y-1">
            <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              {role} dataset
            </div>
            <LineageCard datasetId={v.id as string} />
          </div>
        ))}
    </div>
  );
}
