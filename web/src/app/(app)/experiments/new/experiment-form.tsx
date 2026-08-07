"use client";

// Create-experiment form. Built on the same shell as /benchmark/new and
// /autotrain/new: FormShell (readable column + scrollspy rail discovered from
// `data-form-section`), icon SectionCards, a 4-column Grid of FieldWraps, and a
// sticky FormFooter carrying the submit error and the primary action. When those
// forms' patterns evolve, this one should follow — diff against benchmark-form.tsx.

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  ChevronDown,
  ChevronRight,
  Database,
  FlaskConical,
  Download,
  Gauge,
  Loader2,
  Plus,
  Server,
  Shuffle,
  ShieldAlert,
  Sparkles,
  Trash2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type {
  CustomEvaluatorRecord,
  GlobalEnvRecord,
  ExperimentDatasetOption,
  ExperimentLimits,
  StorageRecord,
  EvaluatorRegistry,
  EvaluatorSpec,
  ExperimentRecord,
  ExperimentTargetSpec,
  ExperimentTargetsResponse,
  ExperimentVariantSpec,
} from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { FormFooter, FormShell } from "@/components/form-shell";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  CaptureDialog,
  captureStateFromParam,
  type CaptureSource,
} from "./capture-dialog";

// `_keyMode` is client-only: which side of the API-key toggle this target is on.
type TargetDraft = ExperimentTargetSpec & { _id: number; _keyMode: "secret" | "paste" };
type VariantDraft = ExperimentVariantSpec & { _id: number };

let seq = 0;
const nextId = () => ++seq;

const emptyTarget = (init: Partial<ExperimentTargetSpec> = {}): TargetDraft => ({
  _id: nextId(),
  _keyMode: "secret",
  label: "",
  base_url: "",
  model: "",
  api_key_secret: "",
  api_key: "",
  ...init,
});

const emptyVariant = (label: string, init: Partial<ExperimentVariantSpec> = {}): VariantDraft => ({
  _id: nextId(),
  label,
  params: {},
  system_override: "",
  system_prefix: "",
  system_suffix: "",
  user_suffix: "",
  assistant_prefill: "",
  response_format: null,
  strip_tools: false,
  ...init,
});

/** Detectors preselected on a fresh form — the ones that catch the failures
 * showing up on almost every endpoint, with no configuration needed. */
const DEFAULT_EVALUATORS = ["control_token_leak", "empty_response", "degeneration", "finish_length"];

export function ExperimentForm({
  datasets: initialDatasets,
  storages,
  registry,
  suggestions,
  limits,
  initialDatasetId,
  clone,
  optimized,
}: {
  datasets: ExperimentDatasetOption[];
  storages: StorageRecord[];
  registry: EvaluatorRegistry;
  suggestions: ExperimentTargetsResponse;
  limits: ExperimentLimits;
  initialDatasetId: string;
  clone?: ExperimentRecord;
  /** Arrived from a GEPA run (`?prompt=opt-…`) — see the two-variant seed below. */
  optimized?: { id: string; name: string; prompt: string; user_suffix: string; dataset_id: string };
}) {
  const router = useRouter();
  const cfg = (clone?.config ?? {}) as Record<string, unknown>;
  const cloneNum = (k: string, d: number) => (typeof cfg[k] === "number" ? (cfg[k] as number) : d);

  /** The two uses of this feature want opposite settings, and the dataset size
   * says which one you're in: a captured trace (a handful of rows) is replayed
   * MANY times to catch intermittent failures, while a corpus is swept ONCE over
   * a sample. Guessing wrong is how a 19k-row dataset becomes 380k requests. */
  function defaultsFor(d: ExperimentDatasetOption | undefined) {
    const rows = d?.num_rows ?? 0;
    if (rows > 0 && rows <= limits.sweep_row_threshold) {
      return { repeats: 20, maxRows: 0 };          // flakiness hunt
    }
    return {
      repeats: 1,
      maxRows: Math.min(rows || limits.sweep_sample_rows, limits.sweep_sample_rows),
    };
  }

  const firstDatasetId = initialDatasetId || clone?.dataset_id || initialDatasets[0]?.id || "";
  const initialDefaults = defaultsFor(initialDatasets.find((d) => d.id === firstDatasetId));

  // Datasets come from the platform's /datasets section — Experiments has no
  // corpus of its own. A capture appends a freshly created one to this list.
  const [datasets, setDatasets] = useState(initialDatasets);
  // The capture dialog and its source tab live in the URL (?capture=platform |
  // langfuse), same router.replace convention as the benchmark form's ?tab= —
  // shareable, survives a refresh, and adds no history entries.
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const { open: captureOpen, source: captureSource } = captureStateFromParam(
    searchParams.get("capture"),
  );

  const setCapture = (next: CaptureSource | null) => {
    const params = new URLSearchParams(searchParams.toString());
    if (next) params.set("capture", next);
    else params.delete("capture");
    const qs = params.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  };
  const [name, setName] = useState(
    clone ? `${clone.name}-copy` : optimized ? `${optimized.name}-confirm` : "",
  );
  const [datasetId, setDatasetId] = useState(firstDatasetId);
  const [targets, setTargets] = useState<TargetDraft[]>(() => {
    const src = (cfg.targets as ExperimentTargetSpec[] | undefined) ?? [];
    // Keys are never returned by the API, so a clone always re-asks for them.
    if (!src.length) return [emptyTarget()];
    return src.map((t) => ({
      ...emptyTarget({ ...t, api_key: "", api_key_secret: t.api_key_secret ?? "" }),
      // Keys are never returned by the API, so a clone re-asks; land on the tab
      // the original used so the prompt makes sense.
      _keyMode: (t.api_key_secret ? "secret" : "paste") as "secret" | "paste",
    }));
  });
  const [variants, setVariants] = useState<VariantDraft[]>(() => {
    const src = (cfg.variants as ExperimentVariantSpec[] | undefined) ?? [];
    if (src.length) return src.map((v) => emptyVariant(v.label, v));
    // A GEPA result arrives as a COMPARISON, not a single variant: the optimizer
    // scored it on a validation slice, and the run that confirms it has to put
    // the new prompt next to the one it replaced on the same rows.
    if (optimized) {
      return [
        emptyVariant("baseline"),
        emptyVariant("optimized", {
          system_override: optimized.prompt,
          user_suffix: optimized.user_suffix,
        }),
      ];
    }
    return [emptyVariant("baseline")];
  });
  const [selected, setSelected] = useState<Record<string, Record<string, unknown>>>(() => {
    const src = cfg.evaluators as Array<{ id: string; options?: Record<string, unknown> }> | undefined;
    if (src?.length) return Object.fromEntries(src.map((s) => [s.id, s.options ?? {}]));
    return Object.fromEntries(DEFAULT_EVALUATORS.map((id) => [id, {}]));
  });
  // Initial repeats/max-cases must match what pickDataset() would choose for the
  // initially-selected dataset — otherwise a first page load on a 19k-row corpus
  // opens at 19k units while choosing that same dataset by hand opens at 200.
  const [repeats, setRepeats] = useState(cloneNum("repeats", initialDefaults.repeats));
  const [concurrency, setConcurrency] = useState(
    cloneNum("concurrency", limits.default_concurrency),
  );
  const [retries, setRetries] = useState(cloneNum("retries", 1));
  const [timeoutS, setTimeoutS] = useState(cloneNum("timeout_s", 300));
  const [stream, setStream] = useState(cfg.stream === undefined ? true : !!cfg.stream);
  const [maxRows, setMaxRows] = useState(cloneNum("max_rows", initialDefaults.maxRows));
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  // Global secrets a target's API key can reference — keys only, resolved
  // server-side at call time so the value never reaches the browser.
  const [secrets, setSecrets] = useState<GlobalEnvRecord[]>([]);
  useEffect(() => {
    gateway
      .listGlobalEnv()
      .then((rows) => setSecrets(rows.filter((r) => r.is_secret)))
      .catch(() => {});
  }, []);

  // Custom evaluators are AUTHORED on the Evaluators tab (they're reusable across
  // experiments); this form only picks from the library. Selecting one stores
  // "custom:<id>" and the gateway snapshots its definition into the run, so a
  // later edit can't change what a finished experiment measured.
  const customs = registry.custom ?? [];

  const dataset = datasets.find((d) => d.id === datasetId);
  // A dataset reports its row count; the exact case count is only known once the
  // rows are read, so the footer labels the unit total as an estimate.
  const rowCount = dataset?.num_rows ?? 0;
  const nRows = maxRows > 0 ? Math.min(rowCount, maxRows) : rowCount;
  const units = nRows * targets.length * variants.length * Math.max(1, repeats);

  const evaluatorList = useMemo(
    () => registry.evaluators.filter((e) => !registry.always_on.includes(e.id)),
    [registry],
  );

  function pickDataset(id: string) {
    setDatasetId(id);
    const d = datasets.find((x) => x.id === id);
    const next = defaultsFor(d);
    setRepeats(next.repeats);
    setMaxRows(next.maxRows);
  }

  function patchTarget(id: number, patch: Partial<TargetDraft>) {
    setTargets((xs) => xs.map((t) => (t._id === id ? { ...t, ...patch } : t)));
  }
  function patchVariant(id: number, patch: Partial<VariantDraft>) {
    setVariants((xs) => xs.map((v) => (v._id === id ? { ...v, ...patch } : v)));
  }

  const blocked = !datasets.length
    ? "no-datasets"
    : !datasetId
      ? "empty-dataset"
      : targets.some((t) => !t.base_url.trim() || !t.model.trim())
        ? "incomplete-target"
        : !name.trim()
          ? "no-name"
          : units > limits.max_units
            ? "over-cap"
            : null;

  /** Trim the matrix to the cap: drop repeats to 1 first (a sweep rarely needs
   * them), then sample rows. Leaves targets and variants alone — those are the
   * comparison you came for. */
  function fitToCap() {
    const perRow = targets.length * variants.length;
    if (repeats > 1 && rowCount * perRow <= limits.max_units) {
      setRepeats(1);
      setMaxRows(0);
      return;
    }
    setRepeats(1);
    setMaxRows(Math.max(1, Math.floor(limits.max_units / Math.max(1, perRow))));
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (blocked) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const row = await gateway.createExperiment({
        name: name.trim(),
        dataset_id: datasetId,
        targets: targets.map((t) => ({
          label: t.label.trim() || t.model || t.base_url,
          base_url: t.base_url.trim(),
          model: t.model.trim(),
          api_key_secret: t._keyMode === "secret" ? t.api_key_secret || undefined : undefined,
          api_key: t._keyMode === "paste" ? t.api_key || undefined : undefined,
          extra_body: t.extra_body,
          path: t.path,
        })),
        variants: variants.map((v) => ({
          label: v.label,
          params: v.params,
          system_override: v.system_override,
          system_prefix: v.system_prefix,
          system_suffix: v.system_suffix,
          user_suffix: v.user_suffix,
          assistant_prefill: v.assistant_prefill,
          response_format: v.response_format,
          strip_tools: v.strip_tools,
          extra_body: v.extra_body,
        })),
        evaluators: Object.entries(selected).map(([id, options]) => ({ id, options })),
        repeats,
        concurrency,
        retries,
        timeout_s: timeoutS,
        stream,
        max_rows: maxRows,
      });
      router.push(`/experiments/${row.id}`);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }

  return (
    <FormShell>
      <form onSubmit={onSubmit} className="space-y-6">
        {/* Header — plain, no gradient. */}
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Create experiment</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Replay every captured request in a dataset against each target and variant, then score
            the replies. Use it to catch control-token leaks, empty responses, degeneration, broken
            JSON, and latency or cost regressions before your users do.
          </p>
        </div>

        {optimized && (
          <div className="rounded-md border border-emerald-500/40 bg-emerald-500/5 px-3 py-2.5 text-sm">
            Seeded from{" "}
            <Link
              href={`/experiments/optimize/${optimized.id}`}
              className="font-medium underline underline-offset-2"
            >
              {optimized.name}
            </Link>
            : two variants, <span className="font-medium">baseline</span> versus{" "}
            <span className="font-medium">optimized</span>, on the dataset it searched. The
            optimizer scored the new prompt on a validation slice — this run confirms it on the
            whole corpus with your full evaluator stack.
          </div>
        )}

        {/* ---------------- Dataset ---------------- */}
        <SectionCard
          icon={<Database className="h-4 w-4" />}
          title="Dataset"
          description="Any dataset from the Datasets section with a messages column. Rows become replayable requests, keeping whatever sampling parameters they were captured with."
          action={
            <div className="flex items-center gap-2">
              <Button type="button" variant="outline" size="sm" onClick={() => setCapture("platform")}>
                <Download className="h-4 w-4" />
                Capture requests
              </Button>
              {/* Red teaming has nothing to capture, so the corpus is GENERATED —
                  which is a Datasets concern, not an Experiments one. */}
              <Button type="button" variant="outline" size="sm" asChild>
                <Link href="/datasets/new?source=generate">
                  <ShieldAlert className="h-4 w-4" />
                  Generate corpus
                </Link>
              </Button>
              <Button type="button" variant="outline" size="sm" asChild>
                <Link href="/datasets">Datasets</Link>
              </Button>
            </div>
          }
        >
          {datasets.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No chat datasets yet. Either{" "}
              <button
                type="button"
                onClick={() => setCapture("platform")}
                className="font-medium text-foreground underline underline-offset-2"
              >
                capture requests
              </button>{" "}
              from a Langfuse trace or your served traffic,{" "}
              <Link
                href="/datasets/new?source=generate"
                className="font-medium text-foreground underline underline-offset-2"
              >
                generate a red-team corpus
              </Link>
              , or add one in{" "}
              <Link
                href="/datasets"
                className="font-medium text-foreground underline underline-offset-2"
              >
                Datasets
              </Link>{" "}
              with its messages column mapped.
            </p>
          ) : (
            <Grid>
              <FieldWrap label="Name" hint="Shown in the run list." wide>
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="channel-leak-sweep"
                />
              </FieldWrap>
              <FieldWrap
                label="Dataset"
                hint={
                  dataset
                    ? `${dataset.kind} · ${dataset.messages_field} column${
                        dataset.num_rows ? ` · ${dataset.num_rows} rows` : ""
                      }`
                    : undefined
                }
                wide
              >
                <Select value={datasetId} onValueChange={pickDataset}>
                  <SelectTrigger id="exp-ds">
                    <SelectValue placeholder="Pick a dataset" />
                  </SelectTrigger>
                  <SelectContent>
                    {datasets.map((d) => (
                      <SelectItem key={d.id} value={d.id}>
                        {d.name}
                        {d.num_rows ? ` (${d.num_rows} rows)` : ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FieldWrap>
            </Grid>
          )}
        </SectionCard>


        {/* ---------------- Targets ---------------- */}
        <SectionCard
          icon={<Server className="h-4 w-4" />}
          title="Targets"
          description="Any OpenAI-compatible endpoint. Picking one of this platform's only prefills the URL and model — a third-party endpoint works the same way."
          action={
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="text-[10px]">
                {targets.length}
              </Badge>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setTargets((xs) => [...xs, emptyTarget()])}
              >
                <Plus className="h-4 w-4" />
                Add
              </Button>
            </div>
          }
        >
          {suggestions.targets.length > 0 && (
            <div className="mb-4">
              <p className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground">
                From this platform
              </p>
              <div className="flex flex-wrap gap-1.5">
                {suggestions.targets.slice(0, 12).map((s) => (
                  <button
                    key={`${s.kind}-${s.id}`}
                    type="button"
                    onClick={() =>
                      setTargets((xs) => {
                        const kept = xs.filter((t) => t.base_url || t.model);
                        return [
                          ...kept,
                          emptyTarget({ label: s.label, base_url: s.base_url, model: s.model }),
                        ];
                      })
                    }
                    className="inline-flex items-center gap-1.5 rounded-md border border-border bg-muted/40 px-2 py-1 text-xs transition-colors hover:border-foreground/30 hover:bg-muted"
                  >
                    <Server className="h-3 w-3 text-muted-foreground" />
                    {s.label}
                    <span className="text-muted-foreground">· {s.kind}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="space-y-4">
            {targets.map((t, i) => (
              <div
                key={t._id}
                className="rounded-md border border-border bg-muted/20 p-3"
              >
                <div className="mb-3 flex items-center justify-between">
                  <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Target {i + 1}
                  </span>
                  {targets.length > 1 && (
                    <button
                      type="button"
                      onClick={() => setTargets((xs) => xs.filter((x) => x._id !== t._id))}
                      className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                      title="Remove target"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
                <Grid>
                  <FieldWrap label="Label" hint="Names this column in the results.">
                    <Input
                      value={t.label}
                      onChange={(e) => patchTarget(t._id, { label: e.target.value })}
                      placeholder="for-agentic"
                    />
                  </FieldWrap>
                  <FieldWrap label="Model">
                    <Input
                      value={t.model}
                      onChange={(e) => patchTarget(t._id, { model: e.target.value })}
                      placeholder="google/gemma-4-31b-it"
                    />
                  </FieldWrap>
                  <FieldWrap
                    label="Base URL"
                    hint="/v1/chat/completions is appended."
                    wide
                  >
                    <Input
                      value={t.base_url}
                      onChange={(e) => patchTarget(t._id, { base_url: e.target.value })}
                      placeholder="https://example.com"
                    />
                  </FieldWrap>
                  <FieldWrap
                    label="API key (optional)"
                    hint={
                      t._keyMode === "secret"
                        ? "Referenced, resolved at run time — rotate it in Secrets."
                        : "Encrypted at rest; never returned by the API."
                    }
                    wide
                  >
                    <div className="space-y-2">
                      {/* Same segmented toggle as the HF-token field on /serverless/new. */}
                      <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
                        {(["secret", "paste"] as const).map((src) => (
                          <button
                            key={src}
                            type="button"
                            onClick={() => patchTarget(t._id, { _keyMode: src })}
                            className={cn(
                              "rounded px-2.5 py-1 transition-colors",
                              t._keyMode === src
                                ? "bg-primary text-primary-foreground"
                                : "text-muted-foreground hover:text-foreground",
                            )}
                          >
                            {src === "secret" ? "Global secret" : "Paste a key"}
                          </button>
                        ))}
                      </div>
                      {t._keyMode === "secret" ? (
                        secrets.length > 0 ? (
                          <Select
                            value={t.api_key_secret ?? ""}
                            onValueChange={(v) => patchTarget(t._id, { api_key_secret: v })}
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select a secret (e.g. OPENAI_API_KEY)" />
                            </SelectTrigger>
                            <SelectContent>
                              {secrets.map((sec) => (
                                <SelectItem key={sec.key} value={sec.key}>
                                  {sec.key}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        ) : (
                          <p className="text-xs text-muted-foreground">
                            No global secrets yet. Add one under{" "}
                            <Link
                              href="/admin/secrets"
                              className="underline underline-offset-2 hover:text-foreground"
                            >
                              Secrets
                            </Link>
                            , or switch to <span className="font-medium">Paste a key</span>.
                          </p>
                        )
                      ) : (
                        <Input
                          type="password"
                          value={t.api_key ?? ""}
                          onChange={(e) => patchTarget(t._id, { api_key: e.target.value })}
                          placeholder="sgpu_…"
                          autoComplete="off"
                          className="font-mono text-xs"
                        />
                      )}
                    </div>
                  </FieldWrap>
                </Grid>
              </div>
            ))}
          </div>
        </SectionCard>

        {/* ---------------- Variants ---------------- */}
        <SectionCard
          icon={<Shuffle className="h-4 w-4" />}
          title="Variants"
          description="Each variant mutates the captured request before sending — the axis you're testing. Keep a single 'baseline' to replay requests exactly as captured."
          action={
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="text-[10px]">
                {variants.length}
              </Badge>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() =>
                  setVariants((xs) => [...xs, emptyVariant(`variant-${xs.length + 1}`)])
                }
              >
                <Plus className="h-4 w-4" />
                Add
              </Button>
            </div>
          }
        >
          <div className="space-y-3">
            {variants.map((v, i) => (
              <VariantEditor
                key={v._id}
                variant={v}
                index={i}
                canRemove={variants.length > 1}
                onPatch={(p) => patchVariant(v._id, p)}
                onRemove={() => setVariants((xs) => xs.filter((x) => x._id !== v._id))}
              />
            ))}
          </div>
        </SectionCard>

        {/* ---------------- Evaluators ---------------- */}
        <SectionCard
          icon={<FlaskConical className="h-4 w-4" />}
          title="Evaluators"
          description={`Scored on every reply. ${registry.always_on.join(", ").replace(/_/g, " ")} always runs, so a failed call is never counted as clean.`}
          action={
            <Badge variant="secondary" className="text-[10px]">
              {Object.keys(selected).length} on
            </Badge>
          }
        >
          <div className="space-y-2">
            {evaluatorList.map((spec) => (
              <EvaluatorRow
                key={spec.id}
                spec={spec}
                checked={spec.id in selected}
                options={selected[spec.id] ?? {}}
                onToggle={(on) =>
                  setSelected((s) => {
                    const next = { ...s };
                    if (on) next[spec.id] = {};
                    else delete next[spec.id];
                    return next;
                  })
                }
                onOptions={(opts) => setSelected((s) => ({ ...s, [spec.id]: opts }))}
              />
            ))}
          </div>

          {/* ---- custom evaluators (authored on the Evaluators tab) ---- */}
          <div className="mt-6 border-t border-border pt-4">
            <div className="mb-2 flex items-center justify-between gap-3">
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Your evaluators
                </p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  Reusable checks from your library — an expression, an API endpoint you already
                  run, or a Python function.
                </p>
              </div>
              <Button type="button" variant="outline" size="sm" asChild>
                <Link href="/experiments/evaluators">
                  {customs.length ? "Manage" : "Create one"}
                </Link>
              </Button>
            </div>

            {customs.length === 0 ? (
              <p className="py-3 text-xs text-muted-foreground">
                None yet — add one on the{" "}
                <Link
                  href="/experiments/evaluators"
                  className="font-medium text-foreground underline underline-offset-2"
                >
                  Evaluators tab
                </Link>{" "}
                and it&apos;ll appear here for every experiment.
              </p>
            ) : (
              <div className="space-y-2">
                {customs.map((c) => (
                  <CustomEvaluatorRow
                    key={c.id}
                    row={c}
                    checked={`custom:${c.id}` in selected}
                    onToggle={(on) =>
                      setSelected((s) => {
                        const next = { ...s };
                        if (on) next[`custom:${c.id}`] = {};
                        else delete next[`custom:${c.id}`];
                        return next;
                      })
                    }
                  />
                ))}
              </div>
            )}
          </div>
        </SectionCard>

        {/* ---------------- Run ---------------- */}
        <SectionCard
          icon={<Gauge className="h-4 w-4" />}
          title="Run"
          description="How hard to push, and how many samples to draw per cell."
        >
          <Grid>
            <FieldWrap
              label="Repeats"
              hint="Per row × target × variant. Raise it to catch intermittent failures on a small captured set; leave at 1 to sweep a corpus."
            >
              <NumInput value={repeats} onChange={setRepeats} min={1} />
            </FieldWrap>
            <FieldWrap label="Concurrency" hint="In-flight requests, max 64.">
              <NumInput value={concurrency} onChange={setConcurrency} min={1} max={64} />
            </FieldWrap>
            <FieldWrap label="Retries" hint="1 = none, so failures are measured, not masked.">
              <NumInput value={retries} onChange={setRetries} min={1} max={5} />
            </FieldWrap>
            <FieldWrap label="Timeout (s)" hint="Per request.">
              <NumInput value={timeoutS} onChange={setTimeoutS} min={1} />
            </FieldWrap>
            <FieldWrap label="Max rows" hint="0 = every row in the dataset.">
              <NumInput value={maxRows} onChange={setMaxRows} min={0} />
            </FieldWrap>
            <FieldWrap label="Streaming" hint="Matches how most clients call the model; needed for TTFT.">
              <div className="flex h-9 items-center gap-2">
                <Switch id="exp-stream" checked={stream} onCheckedChange={setStream} />
                <Label htmlFor="exp-stream" className="cursor-pointer text-sm font-normal">
                  {stream ? "Stream responses" : "Non-streaming"}
                </Label>
              </div>
            </FieldWrap>
          </Grid>

          <div className="mt-6 border-t border-border pt-4">
            <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1 text-sm">
              <span className="text-2xl font-semibold tabular-nums">
                {units.toLocaleString()}
              </span>
              <span className="text-muted-foreground">
                request{units === 1 ? "" : "s"} will be sent —{" "}
                <span className="font-mono text-xs">
                  {nRows.toLocaleString()} dataset row{nRows === 1 ? "" : "s"} × {targets.length} target
                  {targets.length === 1 ? "" : "s"} × {variants.length} variant
                  {variants.length === 1 ? "" : "s"} × {repeats} repeat
                  {repeats === 1 ? "" : "s"}
                </span>
              </span>
            </div>
            {units > limits.max_units ? (
              <p className="mt-1.5 text-[11px] leading-snug text-red-600 dark:text-red-400">
                Over the {limits.max_units.toLocaleString()}-request cap.{" "}
                <button
                  type="button"
                  onClick={fitToCap}
                  className="underline underline-offset-2 hover:text-foreground"
                >
                  Fit to cap
                </button>{" "}
                — sets repeats to 1 and samples the dataset, keeping every target and variant.
              </p>
            ) : units > 2000 ? (
              <p className="mt-1.5 text-[11px] leading-snug text-amber-600 dark:text-amber-400">
                That&apos;s a lot of billed inference against a real endpoint.
              </p>
            ) : null}
            {rowCount > limits.sweep_row_threshold && maxRows > 0 && maxRows < rowCount && (
              <p className="mt-1 text-[11px] leading-snug text-muted-foreground">
                Sampling the first {maxRows.toLocaleString()} of {rowCount.toLocaleString()}{" "}
                rows. Set <span className="font-medium">Max rows</span> to 0 to use them all.
              </p>
            )}
          </div>
        </SectionCard>

        <FormFooter
          error={submitError}
          hint={
            blocked === "no-datasets" ? (
              <>
                Capture requests, or add a chat dataset in{" "}
                <Link href="/datasets" className="underline underline-offset-2 hover:text-foreground">
                  Datasets
                </Link>
                , to run an experiment.
              </>
            ) : blocked === "empty-dataset" ? (
              "Pick a dataset to replay."
            ) : blocked === "incomplete-target" ? (
              "Every target needs a base URL and a model."
            ) : blocked === "no-name" ? (
              "Give the experiment a name."
            ) : blocked === "over-cap" ? (
              <>
                {units.toLocaleString()} requests is over the{" "}
                {limits.max_units.toLocaleString()} cap —{" "}
                <button
                  type="button"
                  onClick={fitToCap}
                  className="underline underline-offset-2 hover:text-foreground"
                >
                  fit it
                </button>
                .
              </>
            ) : (
              "Runs on the gateway — no GPU is provisioned. You can cancel mid-run and keep the results collected so far."
            )
          }
        >
          <Button type="button" variant="outline" onClick={() => router.push("/experiments")}>
            Cancel
          </Button>
          <Button type="submit" disabled={submitting || !!blocked} className="min-w-36">
            {submitting ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Starting…
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" />
                Run experiment
              </>
            )}
          </Button>
        </FormFooter>
      </form>

      <CaptureDialog
        open={captureOpen}
        source={captureSource}
        onOpenChange={(o) => setCapture(o ? captureSource : null)}
        onSourceChange={(next) => setCapture(next)}
        storages={storages}
        onCaptured={(id, dsName, nRows) => {
          setDatasets((xs) => [
            {
              id, name: dsName, kind: "upload", messages_field: "messages",
              num_rows: nRows, owner: "you", usable: true,
            },
            ...xs,
          ]);
          setDatasetId(id);
          if (!name.trim()) setName(`${dsName}-run`);
          setCapture(null);
        }}
      />
    </FormShell>
  );
}

/* ------------------------------------------------------------------ shared */

function SectionCard({
  icon,
  title,
  description,
  action,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    // data-form-section feeds the FormShell scrollspy rail; scroll-mt keeps the
    // heading visible after a rail jump.
    <Card data-form-section={title} className="scroll-mt-6">
      <CardHeader className="pb-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-muted text-muted-foreground">
              {icon}
            </div>
            <CardTitle className="text-base">{title}</CardTitle>
          </div>
          {action}
        </div>
        {description && <CardDescription className="text-xs">{description}</CardDescription>}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function Grid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-1 gap-x-4 gap-y-5 sm:grid-cols-2 lg:grid-cols-4">
      {children}
    </div>
  );
}

function FieldWrap({
  label,
  hint,
  wide,
  extra,
  children,
}: {
  label: string;
  hint?: string;
  wide?: boolean;
  extra?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className={cn("space-y-1.5", wide ? "sm:col-span-2 lg:col-span-2" : "")}>
      <div className="flex items-center justify-between gap-2">
        <Label className="text-xs uppercase tracking-wide text-muted-foreground">{label}</Label>
        {extra}
      </div>
      {children}
      {hint && <p className="text-[11px] leading-snug text-muted-foreground">{hint}</p>}
    </div>
  );
}

function NumInput({
  value,
  onChange,
  min,
  max,
}: {
  value: number;
  onChange: (n: number) => void;
  min?: number;
  max?: number;
}) {
  return (
    <Input
      type="number"
      min={min}
      max={max}
      value={value}
      onChange={(e) => {
        const n = Number(e.target.value);
        onChange(Number.isFinite(n) ? n : (min ?? 0));
      }}
    />
  );
}

/* ---------------------------------------------------------------- variants */

function VariantEditor({
  variant,
  index,
  canRemove,
  onPatch,
  onRemove,
}: {
  variant: VariantDraft;
  index: number;
  canRemove: boolean;
  onPatch: (p: Partial<VariantDraft>) => void;
  onRemove: () => void;
}) {
  // First variant collapsed (it's usually the untouched baseline); the rest open,
  // since adding one means you're about to configure it.
  const [open, setOpen] = useState(index > 0);
  const params = (variant.params ?? {}) as Record<string, unknown>;

  function setParam(key: string, raw: string) {
    const next = { ...params };
    if (raw === "") delete next[key];
    else next[key] = key === "enable_thinking" ? raw === "true" : Number(raw);
    onPatch({ params: next });
  }

  const mutations = [
    Object.keys(params).length ? `${Object.keys(params).length} param` : "",
    variant.system_override ? "system replaced" : "",
    variant.system_suffix ? "system" : "",
    variant.user_suffix ? "user" : "",
    variant.assistant_prefill ? "prefill" : "",
    variant.response_format ? String(variant.response_format) : "",
    variant.strip_tools ? "no tools" : "",
  ].filter(Boolean);

  return (
    <div className="rounded-md border border-border bg-muted/20">
      <div className="flex items-center gap-2 px-3 py-2">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="text-muted-foreground transition-colors hover:text-foreground"
          aria-label={open ? "Collapse variant" : "Expand variant"}
        >
          {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        </button>
        <Input
          value={variant.label}
          onChange={(e) => onPatch({ label: e.target.value })}
          className="h-8 max-w-[220px]"
          placeholder="variant label"
        />
        {!open && mutations.length > 0 && (
          <span className="truncate text-xs text-muted-foreground">{mutations.join(" · ")}</span>
        )}
        {!open && mutations.length === 0 && (
          <span className="text-xs text-muted-foreground">as captured</span>
        )}
        <span className="flex-1" />
        {canRemove && (
          <button
            type="button"
            onClick={onRemove}
            className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
            title="Remove variant"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      {open && (
        <div className="space-y-5 border-t border-border px-3 py-4">
          <Grid>
            {(["temperature", "top_p", "top_k", "max_tokens"] as const).map((k) => (
              <FieldWrap key={k} label={k.replace(/_/g, " ")}>
                <Input
                  value={params[k] === undefined ? "" : String(params[k])}
                  onChange={(e) => setParam(k, e.target.value)}
                  placeholder="as captured"
                />
              </FieldWrap>
            ))}
            <FieldWrap label="enable_thinking">
              <Select
                value={
                  params.enable_thinking === undefined ? "inherit" : String(params.enable_thinking)
                }
                onValueChange={(v) => setParam("enable_thinking", v === "inherit" ? "" : v)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="inherit">as captured</SelectItem>
                  <SelectItem value="true">true</SelectItem>
                  <SelectItem value="false">false</SelectItem>
                </SelectContent>
              </Select>
            </FieldWrap>
            <FieldWrap label="response_format" hint="Forces structured decoding.">
              <Select
                value={(variant.response_format as string) || "none"}
                onValueChange={(v) => onPatch({ response_format: v === "none" ? null : v })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">none</SelectItem>
                  <SelectItem value="json_object">json_object</SelectItem>
                </SelectContent>
              </Select>
            </FieldWrap>
            <FieldWrap label="Tool declarations">
              <div className="flex h-9 items-center gap-2">
                <Switch
                  id={`strip-${variant._id}`}
                  checked={!!variant.strip_tools}
                  onCheckedChange={(v) => onPatch({ strip_tools: v })}
                />
                <Label
                  htmlFor={`strip-${variant._id}`}
                  className="cursor-pointer text-sm font-normal"
                >
                  {variant.strip_tools ? "Stripped" : "Kept"}
                </Label>
              </div>
            </FieldWrap>
          </Grid>

          <FieldWrap
            label="Replace system prompt"
            hint="Swaps the row's system message entirely (the append below still applies on top). This is where a GEPA-optimized prompt lands."
          >
            <Textarea
              rows={variant.system_override ? 8 : 3}
              className="font-mono text-xs"
              value={variant.system_override ?? ""}
              onChange={(e) => onPatch({ system_override: e.target.value })}
              placeholder="Leave blank to keep each row's own system message."
            />
          </FieldWrap>

          <div className="grid grid-cols-1 gap-x-4 gap-y-5 sm:grid-cols-2">
            <FieldWrap
              label="Appended to system prompt"
              hint="Inserted as a system turn if the request has none."
            >
              <Textarea
                rows={3}
                className="text-xs"
                value={variant.system_suffix ?? ""}
                onChange={(e) => onPatch({ system_suffix: e.target.value })}
                placeholder="Output bare JSON with no markdown fence."
              />
            </FieldWrap>
            <div className="space-y-5">
              <FieldWrap label="Appended to last user turn">
                <Input
                  value={variant.user_suffix ?? ""}
                  onChange={(e) => onPatch({ user_suffix: e.target.value })}
                  placeholder="Answer in one sentence."
                />
              </FieldWrap>
              <FieldWrap
                label="Assistant prefill"
                hint="Seeds the reply — the model continues from it."
              >
                <Input
                  value={variant.assistant_prefill ?? ""}
                  onChange={(e) => onPatch({ assistant_prefill: e.target.value })}
                  placeholder={'{"intent":'}
                />
              </FieldWrap>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------- evaluators */

function EvaluatorRow({
  spec,
  checked,
  options,
  onToggle,
  onOptions,
}: {
  spec: EvaluatorSpec;
  checked: boolean;
  options: Record<string, unknown>;
  onToggle: (on: boolean) => void;
  onOptions: (o: Record<string, unknown>) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div
      className={cn(
        "rounded-md border transition-colors",
        checked ? "border-border bg-muted/20" : "border-border/60",
      )}
    >
      <div className="flex items-start gap-3 px-3 py-2.5">
        <Switch
          id={`ev-${spec.id}`}
          checked={checked}
          onCheckedChange={onToggle}
          className="mt-0.5"
        />
        <div className="min-w-0 flex-1">
          <Label htmlFor={`ev-${spec.id}`} className="cursor-pointer text-sm font-medium">
            {spec.label}
            {spec.deferred && (
              <Badge variant="secondary" className="ml-2 text-[10px] font-normal">
                runs after the replay
              </Badge>
            )}
          </Label>
          <p className="mt-0.5 text-xs leading-snug text-muted-foreground">{spec.description}</p>
        </div>
        {checked && spec.options.length > 0 && (
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            className="shrink-0 whitespace-nowrap text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            {open ? "Hide options" : `${spec.options.length} options`}
          </button>
        )}
      </div>
      {checked && open && (
        <div className="border-t border-border px-3 py-4">
          <Grid>
            {spec.options.map((opt) => (
              <FieldWrap
                key={opt.name}
                label={opt.label}
                hint={opt.type === "list" ? undefined : opt.help}
                wide={opt.type === "list"}
              >
                {opt.type === "boolean" ? (
                  <div className="flex h-9 items-center">
                    <Switch
                      checked={
                        options[opt.name] === undefined ? !!opt.default : !!options[opt.name]
                      }
                      onCheckedChange={(v) => onOptions({ ...options, [opt.name]: v })}
                    />
                  </div>
                ) : opt.type === "list" ? (
                  <Textarea
                    rows={2}
                    className="text-xs"
                    value={(options[opt.name] as string) ?? ""}
                    onChange={(e) => onOptions({ ...options, [opt.name]: e.target.value })}
                    placeholder={opt.help}
                  />
                ) : (
                  <Input
                    type={opt.type === "number" ? "number" : "text"}
                    value={options[opt.name] === undefined ? "" : String(options[opt.name])}
                    onChange={(e) => {
                      const raw = e.target.value;
                      onOptions({
                        ...options,
                        [opt.name]:
                          raw === ""
                            ? undefined
                            : opt.type === "number"
                              ? Number(raw)
                              : raw,
                      });
                    }}
                    placeholder={
                      opt.default === null || opt.default === undefined
                        ? "—"
                        : String(opt.default)
                    }
                  />
                )}
              </FieldWrap>
            ))}
          </Grid>
        </div>
      )}
    </div>
  );
}

function CustomEvaluatorRow({
  row,
  checked,
  onToggle,
}: {
  row: CustomEvaluatorRecord;
  checked: boolean;
  onToggle: (on: boolean) => void;
}) {
  const summary =
    row.mode === "api"
      ? String(row.config?.url ?? "")
      : row.code.split("\n")[0] + (row.code.includes("\n") ? " …" : "");
  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-md border px-3 py-2.5 transition-colors",
        checked ? "border-border bg-muted/20" : "border-border/60",
      )}
    >
      <Switch
        id={`ce-${row.id}`}
        checked={checked}
        onCheckedChange={onToggle}
        className="mt-0.5"
      />
      <div className="min-w-0 flex-1">
        <Label htmlFor={`ce-${row.id}`} className="cursor-pointer text-sm font-medium">
          {row.name}
          <Badge variant="secondary" className="ml-2 text-[10px] font-normal">
            {row.mode}
          </Badge>
          {row.fail_when_true && (
            <Badge variant="secondary" className="ml-1 text-[10px] font-normal">
              true = fail
            </Badge>
          )}
        </Label>
        {row.description && (
          <p className="mt-0.5 text-xs leading-snug text-muted-foreground">{row.description}</p>
        )}
        <pre className="mt-1 truncate font-mono text-[11px] text-muted-foreground">{summary}</pre>
      </div>
    </div>
  );
}
