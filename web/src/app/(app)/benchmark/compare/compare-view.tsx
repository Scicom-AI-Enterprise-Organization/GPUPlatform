"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import yaml from "js-yaml";
import {
  CartesianGrid,
  LabelList,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { Activity, ArrowLeft, Check, Clock, Download, Link2, Loader2, Target, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { gateway } from "@/lib/gateway";
import {
  bestBy,
  fetchBenchRows,
  fmt,
  rowFromJson,
  type Row,
  type StatMode,
  statPick,
} from "@/lib/bench-results";
import type { BenchAccuracyResult, BenchmarkRecord } from "@/lib/types";
import { shortGpu } from "@/lib/gpu-format";
import { cn } from "@/lib/utils";
import { Markdown } from "@/components/markdown";

// Unlike the single-bench Results tab (monochrome shades for input lengths),
// Compare uses colour to encode *run identity* — a categorical, high-contrast
// palette so overlaid runs are easy to tell apart on both themes.
const COMPARE_COLORS = [
  "#2563eb", // blue-600
  "#f59e0b", // amber-500
  "#10b981", // emerald-500
  "#ec4899", // pink-500
  "#8b5cf6", // violet-500
  "#06b6d4", // cyan-500
  "#f97316", // orange-500
  "#84cc16", // lime-500
  "#ef4444", // red-500
  "#14b8a6", // teal-500
  "#a855f7", // purple-500
  "#eab308", // yellow-500
];

// One benchmark's loaded state: metadata (for labels) + parsed sweep rows +
// accuracy evals. A run can be a speed run (rows), an accuracy run (accuracy),
// or both.
export type BenchData = {
  id: string;
  name: string;
  status: string;
  model: string | null;
  gpu: string | null;
  gpuCount: number;
  rows: Row[];
  accuracy: BenchAccuracyResult[];
  error: string | null;
};

// Build BenchData from the public-compare payload (no auth / no S3 fetch):
// the gateway inlines each run's record + the aggregate result.json's rows.
export function benchDataFromPublic(item: {
  id: string;
  name?: string;
  status?: string;
  config_yaml?: string;
  result_json?: Record<string, unknown> | null;
  result_rows?: Array<Record<string, unknown>>;
}): BenchData {
  const meta = metaFromConfig(item.config_yaml ?? "");
  const rows = (item.result_rows ?? []).map((e) => rowFromJson(String(e.file ?? ""), e));
  const accRaw = (item.result_json as Record<string, unknown> | null | undefined)?.accuracy;
  const accuracy = Array.isArray(accRaw)
    ? (accRaw.filter(
        (a) => a && typeof a === "object" && typeof (a as BenchAccuracyResult).accuracy === "number",
      ) as BenchAccuracyResult[])
    : [];
  return {
    id: item.id,
    name: item.name ?? item.id,
    status: item.status ?? "unknown",
    model: meta.model,
    gpu: meta.gpu,
    gpuCount: meta.gpuCount,
    rows,
    accuracy,
    error: rows.length === 0 && accuracy.length === 0 ? "no results" : null,
  };
}

function shortModel(s: string | null): string {
  if (!s) return "—";
  return s.split("/").pop() ?? s;
}

function metaFromConfig(cfg_yaml: string): {
  model: string | null;
  gpu: string | null;
  gpuCount: number;
} {
  try {
    const cfg = yaml.load(cfg_yaml) as
      | {
          runpod?: { pod?: { gpu_type?: string; gpu_count?: number } };
          benchmark?: Array<{ model?: { repo_id?: string } }>;
        }
      | null;
    return {
      model: cfg?.benchmark?.[0]?.model?.repo_id ?? null,
      gpu: cfg?.runpod?.pod?.gpu_type ?? null,
      gpuCount: cfg?.runpod?.pod?.gpu_count ?? 1,
    };
  } catch {
    return { model: null, gpu: null, gpuCount: 1 };
  }
}

type Shape = { input_len: number; output_len: number };
const shapeKey = (s: Shape) => `${s.input_len}_${s.output_len}`;
const shapeLabel = (s: Shape) => `in=${s.input_len} · out=${s.output_len}`;

// IQ-vs-speed pairing: auto-match each accuracy run to a speed run by name-token
// overlap (so "glm51-bench-fp8-kvfp8" pairs with the fp8-kvfp8 speed run). NvN —
// every accuracy run becomes one point; the user can re-pick its speed partner.
function nameTokens(s: string): Set<string> {
  return new Set(s.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean));
}
function autoPairSpeed(acc: BenchData, speeds: BenchData[]): string {
  if (speeds.length === 0) return "";
  const at = nameTokens(acc.name);
  let best = speeds[0].id;
  let bestScore = -1;
  for (const s of speeds) {
    let n = 0;
    for (const t of nameTokens(s.name)) if (at.has(t)) n++;
    if (n > bestScore) {
      bestScore = n;
      best = s.id;
    }
  }
  return best;
}

// Pairing overrides are persisted in the URL (`?pair=accId:speedId,…`) so a
// comparison's chosen pairing is shareable + survives reload. Bench ids contain
// no ':' or ',' so those are safe separators.
function parsePairs(s: string | null): Record<string, string> {
  const out: Record<string, string> = {};
  for (const tok of (s ?? "").split(",")) {
    const [a, b] = tok.split(":");
    if (a && b) out[a] = b;
  }
  return out;
}
function serializePairs(m: Record<string, string>): string {
  return Object.entries(m)
    .map(([a, b]) => `${a}:${b}`)
    .join(",");
}

export function CompareView({
  ids,
  initialBenches,
  initialNotes = "",
  initialPairing,
  publicMode = false,
}: {
  ids: string[];
  initialBenches?: BenchData[];
  initialNotes?: string;
  // Frozen accuracy→speed pairing captured into a public share link.
  initialPairing?: Record<string, string>;
  publicMode?: boolean;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [benches, setBenches] = useState<BenchData[] | null>(initialBenches ?? null);
  const [loading, setLoading] = useState(!initialBenches);
  const [notes, setNotes] = useState(initialNotes);
  const [editingNotes, setEditingNotes] = useState(false);
  const [statMode, setStatMode] = useState<StatMode>("median");
  const [shapeSel, setShapeSel] = useState<string | null>(null);

  useEffect(() => {
    if (initialBenches) return; // public page passes pre-loaded data; no fetch
    let cancelled = false;
    (async () => {
      const loaded = await Promise.all(
        ids.map(async (id): Promise<BenchData> => {
          try {
            const [rec, rows] = await Promise.all([
              gateway.getBenchmark(id).catch(() => null as BenchmarkRecord | null),
              fetchBenchRows(id),
            ]);
            const meta = rec ? metaFromConfig(rec.config_yaml) : { model: null, gpu: null, gpuCount: 1 };
            const accRaw = (rec?.result_json as Record<string, unknown> | null | undefined)?.accuracy;
            const accuracy = Array.isArray(accRaw)
              ? (accRaw.filter(
                  (a) => a && typeof a === "object" && typeof (a as BenchAccuracyResult).accuracy === "number",
                ) as BenchAccuracyResult[])
              : [];
            return {
              id,
              name: rec?.name ?? id,
              status: rec?.status ?? "unknown",
              model: meta.model,
              gpu: meta.gpu,
              gpuCount: meta.gpuCount,
              rows,
              accuracy,
              error: rows.length === 0 && accuracy.length === 0 ? "no results in S3" : null,
            };
          } catch (e) {
            return {
              id,
              name: id,
              status: "unknown",
              model: null,
              gpu: null,
              gpuCount: 1,
              rows: [],
              accuracy: [],
              error: e instanceof Error ? e.message : String(e),
            };
          }
        }),
      );
      if (!cancelled) {
        setBenches(loaded);
        setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [ids, initialBenches]);

  // All (input,output) shapes present, ranked by how many benches cover them
  // (then by input length) so the default selection has the most overlap.
  const shapes = useMemo(() => {
    if (!benches) return [];
    const counts = new Map<string, { shape: Shape; benches: Set<string>; cells: number }>();
    for (const b of benches) {
      for (const r of b.rows) {
        const s = { input_len: r.input_len, output_len: r.output_len };
        const k = shapeKey(s);
        const e = counts.get(k) ?? { shape: s, benches: new Set<string>(), cells: 0 };
        e.benches.add(b.id);
        e.cells += 1;
        counts.set(k, e);
      }
    }
    return Array.from(counts.values()).sort(
      (a, b) => b.benches.size - a.benches.size || a.shape.input_len - b.shape.input_len || a.cells - b.cells,
    );
  }, [benches]);

  // Default to the highest-overlap shape (shapes[0]) unless the user picked one
  // that still exists. Derived (not stored) so there's no set-state-in-effect.
  const selectedKey = useMemo(() => {
    if (shapeSel && shapes.some((s) => shapeKey(s.shape) === shapeSel)) return shapeSel;
    return shapes.length > 0 ? shapeKey(shapes[0].shape) : null;
  }, [shapes, shapeSel]);

  const selectedShape = useMemo(
    () => shapes.find((s) => shapeKey(s.shape) === selectedKey)?.shape ?? null,
    [shapes, selectedKey],
  );

  const loadedBenches = useMemo(() => (benches ?? []).filter((b) => b.rows.length > 0), [benches]);

  // ---- IQ-vs-speed (auto-detected when the set has both speed + accuracy runs) ----
  const speedRuns = loadedBenches; // rows > 0
  const accRuns = useMemo(() => (benches ?? []).filter((b) => b.accuracy.length > 0), [benches]);
  const showIq = speedRuns.length > 0 && accRuns.length > 0;

  const accDatasets = useMemo(() => {
    const s = new Set<string>();
    for (const b of accRuns) for (const a of b.accuracy) s.add(a.dataset);
    return Array.from(s).sort();
  }, [accRuns]);

  // Each accuracy run → a paired speed run (its x-coordinate source). Auto-matched
  // by name; user can override per run (NvN rearrange).
  // Public links freeze the captured pairing; in the app it's URL-driven (`?pair=`)
  // so a chosen pairing is shareable + survives reload — single source of truth.
  const pairOverride = useMemo<Record<string, string>>(
    () => (publicMode ? (initialPairing ?? {}) : parsePairs(searchParams.get("pair"))),
    [publicMode, initialPairing, searchParams],
  );
  const setPair = useCallback(
    (accId: string, speedId: string) => {
      const params = new URLSearchParams(searchParams.toString());
      params.set("pair", serializePairs({ ...pairOverride, [accId]: speedId }));
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    },
    [pairOverride, searchParams, router, pathname],
  );
  const pairing = useMemo(() => {
    const p: Record<string, string> = {};
    for (const a of accRuns) p[a.id] = pairOverride[a.id] ?? autoPairSpeed(a, speedRuns);
    return p;
  }, [accRuns, speedRuns, pairOverride]);

  // One set of points PER dataset (GSM8K / MMLU / Function-Call-TaaS / …) so the
  // IQ-vs-speed section shows a chart for each, no dataset dropdown. X = the paired
  // speed run's best throughput; Y = that dataset's accuracy %.
  const iqPointsByDataset = useMemo(() => {
    const byDs: Record<string, IqPoint[]> = {};
    for (const ds of accDatasets) {
      byDs[ds] = accRuns
        .map((a) => {
          const sp = speedRuns.find((s) => s.id === pairing[a.id]);
          const tput = sp ? bestBy(sp.rows, (r) => r.total_token_throughput)?.total_token_throughput ?? null : null;
          const acc = a.accuracy.find((x) => x.dataset === ds)?.accuracy ?? null;
          return {
            id: a.id,
            name: a.name,
            speed: tput,
            acc: acc != null ? +(acc * 100).toFixed(2) : null,
            paired: sp?.name ?? null,
          };
        })
        .filter((d): d is IqPoint => d.speed != null && d.acc != null);
    }
    return byDs;
  }, [accDatasets, accRuns, speedRuns, pairing]);

  // ---- Export PDF via the browser's own print engine (renders the real page →
  // the PDF matches it exactly; "Save as PDF" in the dialog). The app chrome +
  // interactive controls are dropped via @media print in globals.css. ----
  const downloadPdf = useCallback(() => {
    window.print();
  }, []);

  // ---- Create public (no-auth) share link ----
  const [shareUrl, setShareUrl] = useState<string | null>(null);
  const [shareBusy, setShareBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const createShare = useCallback(async () => {
    setShareBusy(true);
    try {
      const { token } = await gateway.createBenchmarkShare(ids, notes, pairing);
      const url = `${window.location.origin}/share/compare/${token}`;
      setShareUrl(url);
      try {
        await navigator.clipboard.writeText(url);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch {
        /* clipboard blocked — link is still shown for manual copy */
      }
    } finally {
      setShareBusy(false);
    }
  }, [ids, notes, pairing]);

  if (loading) {
    return (
      <div className="flex items-center justify-center rounded-md border border-border px-4 py-16 text-sm text-muted-foreground">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading {ids.length} benchmark{ids.length === 1 ? "" : "s"} from S3…
      </div>
    );
  }

  const loadedCount = (benches ?? []).filter((b) => b.rows.length > 0 || b.accuracy.length > 0).length;
  const noData = loadedBenches.length === 0 && accRuns.length === 0;

  return (
    <div className="space-y-6 bg-background">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          {!publicMode && (
            <Button asChild variant="ghost" size="sm" className="-ml-2 h-7 px-2 text-muted-foreground">
              <Link href="/benchmark">
                <ArrowLeft className="h-3.5 w-3.5" /> Benchmarks
              </Link>
            </Button>
          )}
          <h1 className="mt-1 text-2xl font-semibold tracking-tight">Compare benchmarks</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {loadedCount} of {ids.length} loaded{showIq ? " · IQ-vs-speed detected" : ""}
          </p>
        </div>
        <div className="flex items-center gap-2" data-html2canvas-ignore="true">
          {selectedShape && shapes.length > 1 && (
            <Select value={selectedKey ?? ""} onValueChange={setShapeSel}>
              <SelectTrigger className="h-9 w-[170px] text-sm" title="Input/output shape to compare on">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {shapes.map(({ shape, benches: bset }) => {
                  const k = shapeKey(shape);
                  return (
                    <SelectItem key={k} value={k}>
                      {shapeLabel(shape)} ({bset.size}/{ids.length})
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          )}
          {loadedBenches.length > 0 && (
            <Tabs value={statMode} onValueChange={(v) => setStatMode(v as StatMode)}>
              <TabsList>
                <TabsTrigger value="median">Median</TabsTrigger>
                <TabsTrigger value="mean">Mean</TabsTrigger>
                <TabsTrigger value="p99">p99</TabsTrigger>
              </TabsList>
            </Tabs>
          )}
          <Button variant="outline" size="sm" onClick={downloadPdf} title="Opens the print dialog — choose “Save as PDF”">
            <Download className="h-4 w-4" />
            PDF
          </Button>
          {!publicMode && (
            <Button variant="outline" size="sm" onClick={createShare} disabled={shareBusy}>
              {shareBusy ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : copied ? (
                <Check className="h-4 w-4" />
              ) : (
                <Link2 className="h-4 w-4" />
              )}
              {copied ? "Copied!" : "Public link"}
            </Button>
          )}
        </div>
      </div>

      {shareUrl && !publicMode && (
        <div
          className="flex items-center gap-2 rounded-md border border-border bg-muted/40 px-3 py-2 text-xs"
          data-html2canvas-ignore="true"
        >
          <Link2 className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <span className="text-muted-foreground">Public link (no login needed):</span>
          <a href={shareUrl} target="_blank" rel="noopener noreferrer" className="truncate font-mono text-primary underline">
            {shareUrl}
          </a>
        </div>
      )}

      {/* Per-benchmark legend: color ↔ run + role (speed/accuracy). */}
      <div className="flex flex-wrap gap-2">
        {(benches ?? []).map((b, i) => {
          const color = COMPARE_COLORS[i % COMPARE_COLORS.length];
          const inner = (
            <>
              <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color }} />
              <span className="font-medium text-foreground">{b.name}</span>
              <span className="font-mono text-muted-foreground">{shortModel(b.model)}</span>
              {b.gpu && (
                <span className="font-mono text-muted-foreground">
                  · {shortGpu(b.gpu)}
                  {b.gpuCount > 1 ? `×${b.gpuCount}` : ""}
                </span>
              )}
              {b.rows.length > 0 && <span className="text-[10px] uppercase text-blue-400">speed</span>}
              {b.accuracy.length > 0 && <span className="text-[10px] uppercase text-emerald-500">acc</span>}
              {b.error && <span className="text-destructive">· {b.error}</span>}
            </>
          );
          const cls = cn(
            "inline-flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-1.5 text-xs",
            b.error && "opacity-60",
          );
          return publicMode ? (
            <span key={b.id} className={cls}>
              {inner}
            </span>
          ) : (
            <Link key={b.id} href={`/benchmark/${encodeURIComponent(b.id)}`} className={cn(cls, "transition-colors hover:border-primary/40")}>
              {inner}
            </Link>
          );
        })}
      </div>

      {/* Summary / notes (markdown) — report summary + extra text. */}
      {!publicMode ? (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between gap-2">
              <CardTitle className="text-sm">Summary</CardTitle>
              <Button
                variant="ghost"
                size="sm"
                className="h-7"
                data-html2canvas-ignore="true"
                onClick={() => setEditingNotes((v) => !v)}
              >
                {editingNotes ? "Preview" : "Edit"}
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {editingNotes ? (
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={12}
                spellCheck={false}
                placeholder="Paste the report summary + extra notes — markdown (headings, **bold**, tables, lists). Saved into the public link + included in the PDF."
                className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs leading-relaxed shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
              />
            ) : notes.trim() ? (
              <Markdown>{notes}</Markdown>
            ) : (
              <p className="text-sm text-muted-foreground">
                No summary yet — click <span className="font-medium">Edit</span> to add one (markdown; included in the PDF + public link).
              </p>
            )}
          </CardContent>
        </Card>
      ) : (
        notes.trim() && (
          <Card>
            <CardContent className="pt-6">
              <Markdown>{notes}</Markdown>
            </CardContent>
          </Card>
        )
      )}

      {noData ? (
        <div className="rounded-md border border-dashed border-border px-6 py-12 text-center text-sm text-muted-foreground">
          None of the selected benchmarks have results yet.
        </div>
      ) : (
        <>
          {showIq && (
            <IqVsSpeedCard
              pointsByDataset={iqPointsByDataset}
              datasets={accDatasets}
              accRuns={accRuns}
              speedRuns={speedRuns}
              pairing={pairing}
              setPair={setPair}
              publicMode={publicMode}
            />
          )}

          {loadedBenches.length > 0 && (
            <>
              {/* Headline numbers, whole-sweep, side by side. */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Headline (whole sweep)</CardTitle>
                  <CardDescription className="text-xs">
                    Best of each metric across every cell in each run.
                  </CardDescription>
                </CardHeader>
                <CardContent className="px-0 pb-0">
                  <CompareTable benches={loadedBenches} />
                </CardContent>
              </Card>

              {selectedShape && (
                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                  <CompareChart
                    title="Output throughput"
                    subtitle="tokens/sec — higher is better"
                    icon={<TrendingUp className="h-4 w-4" />}
                    benches={loadedBenches}
                    shape={selectedShape}
                    y={(r) => r.output_throughput}
                    yLabel="tok/s"
                  />
                  <CompareChart
                    title="Time to first token"
                    subtitle={`${statMode.toUpperCase()} TTFT (ms) — lower is better`}
                    icon={<Clock className="h-4 w-4" />}
                    benches={loadedBenches}
                    shape={selectedShape}
                    y={(r) => statPick(r, "ttft", statMode)}
                    yLabel="ms"
                  />
                  <CompareChart
                    title="Time per output token"
                    subtitle={`${statMode.toUpperCase()} TPOT (ms) — lower is better`}
                    icon={<Activity className="h-4 w-4" />}
                    benches={loadedBenches}
                    shape={selectedShape}
                    y={(r) => statPick(r, "tpot", statMode)}
                    yLabel="ms"
                  />
                  <CompareChart
                    title="End-to-end latency"
                    subtitle={`${statMode.toUpperCase()} E2EL (ms) — lower is better`}
                    icon={<Clock className="h-4 w-4" />}
                    benches={loadedBenches}
                    shape={selectedShape}
                    y={(r) => statPick(r, "e2el", statMode)}
                    yLabel="ms"
                  />
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}

type IqPoint = { id: string; name: string; speed: number; acc: number; paired: string | null };

function IqVsSpeedCard({
  pointsByDataset,
  datasets,
  accRuns,
  speedRuns,
  pairing,
  setPair,
  publicMode,
}: {
  pointsByDataset: Record<string, IqPoint[]>;
  datasets: string[];
  accRuns: BenchData[];
  speedRuns: BenchData[];
  pairing: Record<string, string>;
  setPair: (accId: string, speedId: string) => void;
  publicMode: boolean;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-muted text-muted-foreground">
            <Target className="h-4 w-4" />
          </div>
          <div>
            <CardTitle className="text-sm">IQ vs speed</CardTitle>
            <CardDescription className="text-[11px]">
              accuracy (%) vs throughput (tok/s) — one chart per dataset; up-and-right is better
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {datasets.map((ds) => (
            <IqScatter key={ds} title={ds} points={pointsByDataset[ds] ?? []} />
          ))}
        </div>

        {/* Pairing (NvN rearrange) — each accuracy run's x comes from a chosen speed
            run; applies to every dataset, so it's shown once. */}
        {!publicMode && speedRuns.length > 1 && (
          <div className="space-y-2 border-t border-border pt-3" data-html2canvas-ignore="true">
            <div className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
              Pairing — accuracy run → speed run (x-axis)
            </div>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {accRuns.map((a) => (
                <div key={a.id} className="flex items-center gap-2 text-xs">
                  <span className="min-w-0 flex-1 truncate font-medium">{a.name}</span>
                  <span className="text-muted-foreground">→</span>
                  <Select
                    value={pairing[a.id] ?? ""}
                    onValueChange={(v) => setPair(a.id, v)}
                  >
                    <SelectTrigger className="h-8 w-[180px] text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {speedRuns.map((s) => (
                        <SelectItem key={s.id} value={s.id}>
                          {s.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// One IQ-vs-speed scatter for a single dataset (titled by the dataset name).
function IqScatter({ title, points }: { title: string; points: IqPoint[] }) {
  return (
    <div className="space-y-1">
      <div className="truncate text-[11px] font-medium text-muted-foreground" title={title}>{title}</div>
      {points.length === 0 ? (
        <div className="flex h-60 w-full items-center justify-center rounded-md border border-dashed border-border text-center text-xs text-muted-foreground">
          No paired results for this dataset.
        </div>
      ) : (
        <div className="h-60 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 16, right: 56, left: 12, bottom: 8 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" />
              <XAxis
                type="number"
                dataKey="speed"
                name="tok/s"
                stroke="currentColor"
                className="text-[10px] text-muted-foreground"
                tickLine={false}
                axisLine={false}
                height={40}
                tickMargin={6}
                domain={["dataMin", "dataMax"]}
                // Pad the plot area so a point's centered label doesn't overlap the
                // Y-axis (left) or get clipped at the right edge.
                padding={{ left: 36, right: 36 }}
                tickFormatter={(v: number) => Number(v).toFixed(2)}
                label={{ value: "throughput (tok/s)", position: "insideBottom", offset: -16, fontSize: 10, fill: "currentColor" }}
              />
              <YAxis
                type="number"
                dataKey="acc"
                name="accuracy"
                unit="%"
                stroke="currentColor"
                className="text-[10px] text-muted-foreground"
                tickLine={false}
                axisLine={false}
                width={56}
                tickMargin={4}
                domain={[0, 100]}
                label={{ value: "accuracy %", angle: -90, position: "insideLeft", offset: 4, fontSize: 10, fill: "currentColor", style: { textAnchor: "middle" } }}
              />
              <ZAxis range={[100, 100]} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<IqTooltip />} />
              {points.map((p, i) => (
                <Scatter key={p.id} name={p.name} data={[p]} fill={COMPARE_COLORS[i % COMPARE_COLORS.length]} isAnimationActive={false}>
                  <LabelList dataKey="name" position="top" style={{ fontSize: 9, fill: "currentColor" }} />
                </Scatter>
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function IqTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: ReadonlyArray<{ payload?: IqPoint }>;
}) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0]?.payload;
  if (!p) return null;
  return (
    <div className="rounded-md border border-zinc-700 bg-zinc-900 px-2.5 py-1.5 text-[11px] text-zinc-200">
      <div className="font-medium">{p.name}</div>
      <div>accuracy: {p.acc.toFixed(1)}%</div>
      <div>speed: {Math.round(p.speed).toLocaleString()} tok/s{p.paired ? ` (${p.paired})` : ""}</div>
    </div>
  );
}

function CompareTable({ benches }: { benches: BenchData[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-muted/40">
          <tr>
            <th className="px-3 py-2 text-left text-xs uppercase tracking-wide text-muted-foreground">Run</th>
            <th className="px-3 py-2 text-left text-xs uppercase tracking-wide text-muted-foreground">Model</th>
            <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">
              Best tput (tok/s)
            </th>
            <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">
              Low median TTFT (ms)
            </th>
            <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">
              Low median TPOT (ms)
            </th>
            <th className="px-3 py-2 text-right text-xs uppercase tracking-wide text-muted-foreground">Cells</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {benches.map((b, i) => {
            const tput = bestBy(b.rows, (r) => r.output_throughput);
            const ttft = bestBy(b.rows, (r) => r.median_ttft_ms, true);
            const tpot = bestBy(b.rows, (r) => r.median_tpot_ms, true);
            const color = COMPARE_COLORS[i % COMPARE_COLORS.length];
            return (
              <tr key={b.id}>
                <td className="px-3 py-1.5">
                  <span className="inline-flex items-center gap-2">
                    <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
                    <span className="font-medium">{b.name}</span>
                  </span>
                </td>
                <td className="px-3 py-1.5 font-mono text-xs text-muted-foreground">{shortModel(b.model)}</td>
                <td className="px-3 py-1.5 text-right tabular-nums">
                  {fmt(tput?.output_throughput ?? null, 1)}
                  {tput && (
                    <span className="ml-1 text-[10px] text-muted-foreground">c={tput.concurrency}</span>
                  )}
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums">{fmt(ttft?.median_ttft_ms ?? null, 1)}</td>
                <td className="px-3 py-1.5 text-right tabular-nums">{fmt(tpot?.median_tpot_ms ?? null, 2)}</td>
                <td className="px-3 py-1.5 text-right tabular-nums text-muted-foreground">{b.rows.length}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function CompareChart({
  title,
  subtitle,
  icon,
  benches,
  shape,
  y,
  yLabel,
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  benches: BenchData[];
  shape: Shape;
  y: (r: Row) => number | null;
  yLabel: string;
}) {
  // recharts wants one object per X point with each run as a column keyed by
  // its bench id. X = union of concurrencies present at this shape.
  const data = useMemo(() => {
    const concs = Array.from(
      new Set(
        benches.flatMap((b) =>
          b.rows
            .filter((r) => r.input_len === shape.input_len && r.output_len === shape.output_len)
            .map((r) => r.concurrency),
        ),
      ),
    ).sort((a, b) => a - b);
    return concs.map((c) => {
      const row: Record<string, number | null> = { concurrency: c };
      for (const b of benches) {
        const m = b.rows.find(
          (r) =>
            r.input_len === shape.input_len &&
            r.output_len === shape.output_len &&
            r.concurrency === c,
        );
        const v = m ? y(m) : null;
        // ≤0 isn't a real measurement (a crashed/unmeasured config reports 0, and
        // some metrics like e2el are simply absent → null). Drop it so the line
        // skips the gap instead of plotting a misleading 0.
        row[b.id] = typeof v === "number" && v > 0 ? v : null;
      }
      return row;
    });
  }, [benches, shape, y]);

  // Whether ANY run has a real value for this metric at this shape. If not (e.g.
  // the runs' result rows carry no `e2el_ms` fields), there's nothing to plot —
  // show a "no data" note instead of an empty/broken chart.
  const hasData = useMemo(
    () => data.some((d) => benches.some((b) => typeof d[b.id] === "number")),
    [data, benches],
  );

  const nameById = useMemo(() => {
    const m: Record<string, string> = {};
    for (const b of benches) m[b.id] = b.name;
    return m;
  }, [benches]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-muted text-muted-foreground">
            {icon}
          </div>
          <div>
            <CardTitle className="text-sm">{title}</CardTitle>
            <CardDescription className="text-[11px]">{subtitle}</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {!hasData ? (
          <div className="flex h-64 w-full items-center justify-center rounded-md border border-dashed border-border text-center text-xs text-muted-foreground">
            No {yLabel === "ms" ? title.toLowerCase() : "data"} recorded for these runs.
          </div>
        ) : (
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 8, left: 12, bottom: 8 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis
                dataKey="concurrency"
                stroke="currentColor"
                className="text-[10px] text-muted-foreground"
                tickLine={false}
                axisLine={false}
                height={40}
                tickMargin={6}
                label={{
                  value: "concurrency",
                  position: "insideBottom",
                  offset: -16,
                  fontSize: 10,
                  fill: "currentColor",
                }}
              />
              <YAxis
                stroke="currentColor"
                className="text-[10px] text-muted-foreground"
                tickLine={false}
                axisLine={false}
                width={68}
                tickMargin={4}
                label={{
                  value: yLabel,
                  angle: -90,
                  position: "insideLeft",
                  offset: -2,
                  fontSize: 10,
                  fill: "currentColor",
                  style: { textAnchor: "middle" },
                }}
              />
              <Tooltip
                contentStyle={{
                  background: "rgb(24 24 27)",
                  border: "1px solid rgb(63 63 70)",
                  borderRadius: 6,
                  fontSize: 11,
                }}
                labelStyle={{ color: "rgb(244 244 245)" }}
                itemStyle={{ color: "rgb(228 228 231)" }}
                formatter={(value, name) => [value as number, nameById[String(name)] ?? String(name)]}
              />
              <Legend
                verticalAlign="top"
                align="center"
                iconType="plainline"
                wrapperStyle={{ fontSize: 11, paddingBottom: 8 }}
              />
              {benches.map((b, i) => (
                <Line
                  key={b.id}
                  type="monotone"
                  dataKey={b.id}
                  name={b.name}
                  stroke={COMPARE_COLORS[i % COMPARE_COLORS.length]}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                  connectNulls
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
        )}
      </CardContent>
    </Card>
  );
}
