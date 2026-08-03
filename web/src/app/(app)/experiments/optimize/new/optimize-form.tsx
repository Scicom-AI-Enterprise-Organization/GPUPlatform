"use client";

// Create-optimization form. Same shell as /experiments/new (which follows
// /benchmark/new): FormShell + scrollspy discovered from `data-form-section`,
// icon SectionCards, a 4-column Grid of FieldWraps, a sticky FormFooter with the
// `blocked` discriminant. Diff against experiment-form.tsx before changing
// anything structural here.
//
// The one thing this form does that the others don't: it PRICES the run. GEPA's
// budget is denominated in real billed requests, so the footer shows that number
// before you can submit, and the ceilings come from the server
// (GET /v1/prompt-optimizations/limits) rather than being hardcoded.

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Database,
  FlaskConical,
  Gauge,
  Loader2,
  Server,
  Sparkles,
  Wand2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type {
  CustomEvaluatorRecord,
  EvaluatorRegistry,
  EvaluatorSpec,
  ExperimentDatasetOption,
  ExperimentTargetSpec,
  ExperimentTargetsResponse,
  GlobalEnvRecord,
  PromptOptLimits,
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

type KeyMode = "secret" | "paste";
type TargetDraft = ExperimentTargetSpec & { _keyMode: KeyMode };

const emptyTarget = (init: Partial<ExperimentTargetSpec> = {}): TargetDraft => ({
  _keyMode: "secret",
  label: "",
  base_url: "",
  model: "",
  api_key_secret: "",
  api_key: "",
  ...init,
});

/** Detectors that make sense as an OBJECTIVE rather than a smoke alarm: each one
 * is something a prompt can actually be rewritten to satisfy. Latency and cost
 * are deliberately absent — a prompt can't make the GPU faster, and optimizing
 * against them just discovers "say less". */
const DEFAULT_EVALUATORS = ["json_output"];

export function OptimizeForm({
  datasets,
  registry,
  suggestions,
  limits,
  initialDatasetId,
}: {
  datasets: ExperimentDatasetOption[];
  registry: EvaluatorRegistry;
  suggestions: ExperimentTargetsResponse;
  limits: PromptOptLimits;
  initialDatasetId: string;
}) {
  const router = useRouter();

  const [name, setName] = useState("");
  const [datasetId, setDatasetId] = useState(initialDatasetId || datasets[0]?.id || "");
  const [target, setTarget] = useState<TargetDraft>(() => emptyTarget());
  const [sameReflection, setSameReflection] = useState(true);
  const [reflection, setReflection] = useState<TargetDraft>(() => emptyTarget());
  const [guidance, setGuidance] = useState("");
  const [seedPrompt, setSeedPrompt] = useState("");
  const [seedNote, setSeedNote] = useState<string | null>(null);
  const [seedLoading, setSeedLoading] = useState(false);
  const [components, setComponents] = useState<string[]>(["system_prompt"]);
  const [seedUserSuffix, setSeedUserSuffix] = useState("");
  const [selected, setSelected] = useState<Record<string, Record<string, unknown>>>(() =>
    Object.fromEntries(
      DEFAULT_EVALUATORS.filter((id) => registry.evaluators.some((e) => e.id === id)).map((id) => [
        id,
        {},
      ]),
    ),
  );
  const [maxRows, setMaxRows] = useState(limits.default_rows);
  const [valRows, setValRows] = useState(0);
  const [budget, setBudget] = useState("light");
  const [customCalls, setCustomCalls] = useState(limits.default_rows * 6);
  const [minibatch, setMinibatch] = useState(limits.default_minibatch);
  const [concurrency, setConcurrency] = useState(limits.default_concurrency);
  const [timeoutS, setTimeoutS] = useState(300);
  const [includeExpected, setIncludeExpected] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [secrets, setSecrets] = useState<GlobalEnvRecord[]>([]);

  useEffect(() => {
    gateway
      .listGlobalEnv()
      .then((rows) => setSecrets(rows.filter((r) => r.is_secret)))
      .catch(() => {});
  }, []);

  const dataset = datasets.find((d) => d.id === datasetId);
  const rowCount = dataset?.num_rows ?? 0;
  const nRows = Math.min(rowCount || maxRows, maxRows, limits.max_rows);
  // Mirrors the server's auto split (40% validation) so the priced budget below
  // matches what the runner actually resolves.
  const nVal = valRows > 0 ? Math.min(valRows, nRows) : Math.max(1, Math.round(nRows * 0.4));
  const nTrain = Math.max(0, nRows - nVal) || nVal;

  /** Same arithmetic as `resolve_budget()` in prompt_opt_api.py — the number the
   * user is agreeing to spend must be the number the runner enforces. */
  const resolvedBudget = useMemo(() => {
    const floor = 2 * Math.max(1, nVal) + 2 * Math.max(1, minibatch);
    const want =
      budget === "custom"
        ? customCalls
        : (limits.auto_budgets[budget] ?? limits.auto_budgets.light ?? 6) * Math.max(1, nVal);
    return Math.max(floor, Math.min(limits.max_metric_calls, want || floor));
  }, [budget, customCalls, limits, minibatch, nVal]);

  // The dataset's own system prompt is the honest starting point: optimizing a
  // prompt the rows were never captured under measures something else.
  const loadSeed = useCallback(async (id: string) => {
    if (!id) return;
    setSeedLoading(true);
    try {
      const res = await gateway.promptOptSeed(id);
      setSeedPrompt(res.seed_prompt);
      setSeedNote(
        res.source === "dataset"
          ? `Prefilled from the dataset — ${res.n_with_system} of ${res.n_rows} rows carry a system message${
              res.distinct_system > 1 ? `, ${res.distinct_system} distinct` : ""
            }.`
          : "No row in this dataset has a system message — write the prompt you want to improve.",
      );
    } catch {
      setSeedNote(null);
    } finally {
      setSeedLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSeed(datasetId);
  }, [datasetId, loadSeed]);

  const evaluatorList = useMemo(
    () => registry.evaluators.filter((e) => !registry.always_on.includes(e.id)),
    [registry],
  );
  const customs = registry.custom ?? [];

  const blocked = !datasets.length
    ? "no-datasets"
    : !datasetId
      ? "empty-dataset"
      : !name.trim()
        ? "no-name"
        : !target.base_url.trim() || !target.model.trim()
          ? "incomplete-target"
          : !sameReflection && (!reflection.base_url.trim() || !reflection.model.trim())
            ? "incomplete-reflection"
            : Object.keys(selected).length === 0
              ? "no-evaluators"
              : !components.length
                ? "no-components"
                : null;

  function toggleComponent(id: string, on: boolean) {
    setComponents((xs) => (on ? [...new Set([...xs, id])] : xs.filter((x) => x !== id)));
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (blocked) return;
    setSubmitting(true);
    setSubmitError(null);
    const asSpec = (t: TargetDraft): ExperimentTargetSpec => ({
      label: t.label.trim() || t.model || t.base_url,
      base_url: t.base_url.trim(),
      model: t.model.trim(),
      api_key_secret: t._keyMode === "secret" ? t.api_key_secret || undefined : undefined,
      api_key: t._keyMode === "paste" ? t.api_key || undefined : undefined,
    });
    try {
      const row = await gateway.createPromptOpt({
        name: name.trim(),
        dataset_id: datasetId,
        target: asSpec(target),
        reflection: sameReflection ? null : asSpec(reflection),
        reflection_guidance: guidance,
        evaluators: Object.entries(selected).map(([id, options]) => ({ id, options })),
        components,
        seed_prompt: seedPrompt,
        seed_user_suffix: seedUserSuffix,
        max_rows: maxRows,
        val_rows: valRows,
        budget,
        max_metric_calls: budget === "custom" ? customCalls : 0,
        minibatch_size: minibatch,
        concurrency,
        timeout_s: timeoutS,
        include_expected: includeExpected,
      });
      router.push(`/experiments/optimize/${row.id}`);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }

  return (
    <FormShell>
      <form onSubmit={onSubmit} className="space-y-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">New prompt optimization</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            GEPA replays your dataset under a candidate prompt, scores every reply with the
            evaluators you pick, shows the failures to a reflection model, and keeps the rewrite
            only if it measurably wins. What you get back is a system prompt you can drop into an
            experiment.
          </p>
        </div>

        {/* ---------------- Dataset ---------------- */}
        <SectionCard
          icon={<Database className="h-4 w-4" />}
          title="Dataset"
          description="The rows the prompt is tuned against. They're split into a validation set (which scores every candidate) and a train set (which feeds the reflection minibatches)."
          action={
            <Button type="button" variant="outline" size="sm" asChild>
              <Link href="/datasets">Datasets</Link>
            </Button>
          }
        >
          {datasets.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No chat datasets yet. Add one in{" "}
              <Link
                href="/datasets"
                className="font-medium text-foreground underline underline-offset-2"
              >
                Datasets
              </Link>{" "}
              with its messages column mapped, or capture requests from the{" "}
              <Link
                href="/experiments/new"
                className="font-medium text-foreground underline underline-offset-2"
              >
                experiment form
              </Link>
              .
            </p>
          ) : (
            <Grid>
              <FieldWrap label="Name" hint="Shown in the run list." wide>
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="support-prompt-v3"
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
                <Select value={datasetId} onValueChange={setDatasetId}>
                  <SelectTrigger>
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
              <FieldWrap
                label="Rows to use"
                hint={`Capped at ${limits.max_rows}. Cost scales with rows × candidates.`}
              >
                <NumInput value={maxRows} onChange={setMaxRows} min={1} max={limits.max_rows} />
              </FieldWrap>
              <FieldWrap
                label="Validation rows"
                hint={`0 = auto (40%). Now: ${nVal} validation / ${nTrain} train.`}
              >
                <NumInput value={valRows} onChange={setValRows} min={0} max={nRows || undefined} />
              </FieldWrap>
              {nRows > 0 && nVal >= nRows && (
                <div className="sm:col-span-2 lg:col-span-4">
                  <p className="text-[11px] leading-snug text-amber-600 dark:text-amber-400">
                    Every row is in the validation set, so the reflection minibatches reuse it —
                    the reported gain will be measured on the rows it tuned against.
                  </p>
                </div>
              )}
            </Grid>
          )}
        </SectionCard>

        {/* ---------------- Starting prompt ---------------- */}
        <SectionCard
          icon={<Wand2 className="h-4 w-4" />}
          title="Starting prompt"
          description="The instruction GEPA rewrites. Prefilled with the dataset's own system message so the baseline is the prompt those rows were actually captured under."
        >
          <div className="space-y-4">
            <div className="flex flex-wrap gap-3">
              {limits.components.map((c) => (
                <label
                  key={c.id}
                  className={cn(
                    "flex flex-1 min-w-[220px] cursor-pointer items-start gap-3 rounded-md border px-3 py-2.5 transition-colors",
                    components.includes(c.id) ? "border-border bg-muted/20" : "border-border/60",
                  )}
                >
                  <Switch
                    checked={components.includes(c.id)}
                    onCheckedChange={(on) => toggleComponent(c.id, on)}
                    className="mt-0.5"
                  />
                  <span className="min-w-0">
                    <span className="block text-sm font-medium">{c.label}</span>
                    <span className="mt-0.5 block text-xs leading-snug text-muted-foreground">
                      {c.description}
                    </span>
                  </span>
                </label>
              ))}
            </div>

            {components.includes("system_prompt") && (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <Label className="text-xs uppercase tracking-wide text-muted-foreground">
                    System prompt
                  </Label>
                  {seedLoading && (
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                  )}
                </div>
                <Textarea
                  rows={7}
                  value={seedPrompt}
                  onChange={(e) => setSeedPrompt(e.target.value)}
                  placeholder="You are a support agent for…"
                  className="font-mono text-xs"
                />
                {seedNote && (
                  <p className="text-[11px] leading-snug text-muted-foreground">{seedNote}</p>
                )}
              </div>
            )}

            {components.includes("user_suffix") && (
              <div className="space-y-1.5">
                <Label className="text-xs uppercase tracking-wide text-muted-foreground">
                  User-turn reminder
                </Label>
                <Textarea
                  rows={3}
                  value={seedUserSuffix}
                  onChange={(e) => setSeedUserSuffix(e.target.value)}
                  placeholder="(appended to the last user message — leave blank to start from nothing)"
                  className="font-mono text-xs"
                />
              </div>
            )}
          </div>
        </SectionCard>

        {/* ---------------- Target ---------------- */}
        <SectionCard
          icon={<Server className="h-4 w-4" />}
          title="Target"
          description="The model whose prompt is being optimized. Any OpenAI-compatible endpoint; picking one of this platform's just prefills the URL and model."
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
                      setTarget((t) => ({
                        ...t,
                        label: s.label,
                        base_url: s.base_url,
                        model: s.model,
                      }))
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
          <TargetFields
            draft={target}
            onPatch={(p) => setTarget((t) => ({ ...t, ...p }))}
            secrets={secrets}
          />
        </SectionCard>

        {/* ---------------- Reflection ---------------- */}
        <SectionCard
          icon={<Sparkles className="h-4 w-4" />}
          title="Reflection model"
          description="The model that WRITES the new prompts. GEPA's published result comes from a small student and a stronger reflector — using the target for both works, but is the weaker setup."
          action={
            <div className="flex items-center gap-2">
              <Label htmlFor="same-refl" className="cursor-pointer text-xs text-muted-foreground">
                Same as target
              </Label>
              <Switch id="same-refl" checked={sameReflection} onCheckedChange={setSameReflection} />
            </div>
          }
        >
          <div className="space-y-4">
            {!sameReflection && (
              <TargetFields
                draft={reflection}
                onPatch={(p) => setReflection((t) => ({ ...t, ...p }))}
                secrets={secrets}
              />
            )}
            <Grid>
              <FieldWrap
                label="Extra guidance (optional)"
                hint="Appended to the meta-prompt — house style, length limits, a language it must answer in."
                wide
              >
                <Textarea
                  rows={2}
                  value={guidance}
                  onChange={(e) => setGuidance(e.target.value)}
                  placeholder="Keep it under 200 words. Never mention internal tool names."
                  className="text-xs"
                />
              </FieldWrap>
              <FieldWrap
                label="Show reference answers"
                hint="Lets the reflector bake the row's `expected` data into the prompt. Powerful, and the main way a run overfits."
                wide
              >
                <div className="flex h-9 items-center">
                  <Switch checked={includeExpected} onCheckedChange={setIncludeExpected} />
                </div>
              </FieldWrap>
            </Grid>
          </div>
        </SectionCard>

        {/* ---------------- Evaluators ---------------- */}
        <SectionCard
          icon={<FlaskConical className="h-4 w-4" />}
          title="Evaluators"
          description="These ARE the score. Each detector's verdict becomes a number GEPA climbs and a written reason the reflection model reads — pick the ones that describe a good reply."
          action={
            <Badge variant="secondary" className="text-[10px]">
              {Object.keys(selected).length} selected
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
                onOptions={(o) => setSelected((s) => ({ ...s, [spec.id]: o }))}
              />
            ))}
            {customs.length > 0 && (
              <>
                <p className="pt-2 text-xs uppercase tracking-wide text-muted-foreground">
                  Your evaluators
                </p>
                {customs.map((row) => (
                  <CustomEvaluatorRow
                    key={row.id}
                    row={row}
                    checked={`custom:${row.id}` in selected}
                    onToggle={(on) =>
                      setSelected((s) => {
                        const next = { ...s };
                        if (on) next[`custom:${row.id}`] = {};
                        else delete next[`custom:${row.id}`];
                        return next;
                      })
                    }
                  />
                ))}
              </>
            )}
            <p className="pt-1 text-[11px] leading-snug text-muted-foreground">
              A detector that abstains on a row (no reference data) scores nothing rather than
              passing it — if every evaluator abstains, the search has nothing to climb. Author new
              ones on the{" "}
              <Link
                href="/experiments/evaluators"
                className="underline underline-offset-2 hover:text-foreground"
              >
                Evaluators
              </Link>{" "}
              tab.
            </p>
          </div>
        </SectionCard>

        {/* ---------------- Budget ---------------- */}
        <SectionCard
          icon={<Gauge className="h-4 w-4" />}
          title="Budget"
          description="How many real, billed requests the search may spend on the target. Reflection calls are counted separately and hit the reflection endpoint."
        >
          <Grid>
            <FieldWrap label="Preset" hint="Multiples of the validation set.">
              <Select value={budget} onValueChange={setBudget}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(limits.auto_budgets).map(([k, mult]) => (
                    <SelectItem key={k} value={k}>
                      {k} (×{mult})
                    </SelectItem>
                  ))}
                  <SelectItem value="custom">custom</SelectItem>
                </SelectContent>
              </Select>
            </FieldWrap>
            {budget === "custom" && (
              <FieldWrap label="Metric calls" hint={`Ceiling ${limits.max_metric_calls}.`}>
                <NumInput
                  value={customCalls}
                  onChange={setCustomCalls}
                  min={1}
                  max={limits.max_metric_calls}
                />
              </FieldWrap>
            )}
            <FieldWrap label="Minibatch" hint="Rows shown to the reflection model per turn.">
              <NumInput value={minibatch} onChange={setMinibatch} min={1} max={20} />
            </FieldWrap>
            <FieldWrap label="Concurrency" hint={`Max ${limits.max_concurrency}.`}>
              <NumInput
                value={concurrency}
                onChange={setConcurrency}
                min={1}
                max={limits.max_concurrency}
              />
            </FieldWrap>
            <FieldWrap label="Timeout (s)" hint="Per request.">
              <NumInput value={timeoutS} onChange={setTimeoutS} min={1} />
            </FieldWrap>
          </Grid>
          <div className="mt-4 rounded-md border border-border bg-muted/20 px-3 py-2.5">
            <p className="text-sm">
              Up to{" "}
              <span className="font-semibold tabular-nums">
                {resolvedBudget.toLocaleString()}
              </span>{" "}
              billed requests to the target
            </p>
            <p className="mt-0.5 text-[11px] leading-snug text-muted-foreground">
              {nVal} validation rows score every surviving candidate; each turn spends{" "}
              {2 * minibatch} more on the minibatch. The run stops before it can overshoot, so this
              is a ceiling, not an estimate.
            </p>
            {resolvedBudget >= limits.max_metric_calls && (
              <p className="mt-1.5 text-[11px] leading-snug text-amber-600 dark:text-amber-400">
                Clamped to the {limits.max_metric_calls.toLocaleString()} hard ceiling.
              </p>
            )}
          </div>
        </SectionCard>

        <FormFooter
          error={submitError}
          hint={
            blocked === "no-datasets" ? (
              <>
                Add a chat dataset in{" "}
                <Link href="/datasets" className="underline underline-offset-2 hover:text-foreground">
                  Datasets
                </Link>{" "}
                to optimize against.
              </>
            ) : blocked === "empty-dataset" ? (
              "Pick a dataset."
            ) : blocked === "no-name" ? (
              "Give the optimization a name."
            ) : blocked === "incomplete-target" ? (
              "The target needs a base URL and a model."
            ) : blocked === "incomplete-reflection" ? (
              "The reflection endpoint needs a base URL and a model — or switch it back to the target."
            ) : blocked === "no-evaluators" ? (
              "Pick at least one evaluator — it is the score GEPA optimizes against."
            ) : blocked === "no-components" ? (
              "Pick something for GEPA to rewrite."
            ) : (
              "Runs on the gateway — no GPU is provisioned. Cancel any time; the best prompt found so far is kept."
            )
          }
        >
          <Button
            type="button"
            variant="outline"
            onClick={() => router.push("/experiments/optimize")}
          >
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
                <Wand2 className="h-4 w-4" />
                Optimize
              </>
            )}
          </Button>
        </FormFooter>
      </form>
    </FormShell>
  );
}

/* ------------------------------------------------------------------ shared */

function TargetFields({
  draft,
  onPatch,
  secrets,
}: {
  draft: TargetDraft;
  onPatch: (p: Partial<TargetDraft>) => void;
  secrets: GlobalEnvRecord[];
}) {
  return (
    <Grid>
      <FieldWrap label="Label" hint="Names it in the record.">
        <Input
          value={draft.label}
          onChange={(e) => onPatch({ label: e.target.value })}
          placeholder="prod-gemma"
        />
      </FieldWrap>
      <FieldWrap label="Model">
        <Input
          value={draft.model}
          onChange={(e) => onPatch({ model: e.target.value })}
          placeholder="google/gemma-4-31b-it"
        />
      </FieldWrap>
      <FieldWrap label="Base URL" hint="/v1/chat/completions is appended." wide>
        <Input
          value={draft.base_url}
          onChange={(e) => onPatch({ base_url: e.target.value })}
          placeholder="https://example.com"
        />
      </FieldWrap>
      <FieldWrap
        label="API key (optional)"
        hint={
          draft._keyMode === "secret"
            ? "Referenced, resolved at run time — rotate it in Secrets."
            : "Encrypted at rest; never returned by the API."
        }
        wide
      >
        <div className="space-y-2">
          <div className="inline-flex rounded-md border border-border p-0.5 text-xs">
            {(["secret", "paste"] as const).map((src) => (
              <button
                key={src}
                type="button"
                onClick={() => onPatch({ _keyMode: src })}
                className={cn(
                  "rounded px-2.5 py-1 transition-colors",
                  draft._keyMode === src
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {src === "secret" ? "Global secret" : "Paste a key"}
              </button>
            ))}
          </div>
          {draft._keyMode === "secret" ? (
            secrets.length > 0 ? (
              <Select
                value={draft.api_key_secret ?? ""}
                onValueChange={(v) => onPatch({ api_key_secret: v })}
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
              value={draft.api_key ?? ""}
              onChange={(e) => onPatch({ api_key: e.target.value })}
              placeholder="sgpu_…"
              autoComplete="off"
              className="font-mono text-xs"
            />
          )}
        </div>
      </FieldWrap>
    </Grid>
  );
}

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
          id={`opt-ev-${spec.id}`}
          checked={checked}
          onCheckedChange={onToggle}
          className="mt-0.5"
        />
        <div className="min-w-0 flex-1">
          <Label htmlFor={`opt-ev-${spec.id}`} className="cursor-pointer text-sm font-medium">
            {spec.label}
            {spec.deferred && (
              <Badge variant="secondary" className="ml-2 text-[10px] font-normal">
                one extra call per rollout
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
                      checked={options[opt.name] === undefined ? !!opt.default : !!options[opt.name]}
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
                          raw === "" ? undefined : opt.type === "number" ? Number(raw) : raw,
                      });
                    }}
                    placeholder={
                      opt.default === null || opt.default === undefined ? "—" : String(opt.default)
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
      <Switch id={`opt-ce-${row.id}`} checked={checked} onCheckedChange={onToggle} className="mt-0.5" />
      <div className="min-w-0 flex-1">
        <Label htmlFor={`opt-ce-${row.id}`} className="cursor-pointer text-sm font-medium">
          {row.name}
          <Badge variant="secondary" className="ml-2 text-[10px] font-normal">
            {row.mode}
          </Badge>
        </Label>
        {row.description && (
          <p className="mt-0.5 text-xs leading-snug text-muted-foreground">{row.description}</p>
        )}
        <pre className="mt-1 truncate font-mono text-[11px] text-muted-foreground">{summary}</pre>
      </div>
    </div>
  );
}

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
  // data-form-section feeds the FormShell scrollspy rail; scroll-mt keeps the
  // heading visible after a rail jump.
  return (
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
    <div className="grid grid-cols-1 gap-x-4 gap-y-5 sm:grid-cols-2 lg:grid-cols-4">{children}</div>
  );
}

function FieldWrap({
  label,
  hint,
  wide,
  children,
}: {
  label: string;
  hint?: string;
  wide?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className={cn("space-y-1.5", wide ? "sm:col-span-2 lg:col-span-2" : "")}>
      <Label className="text-xs uppercase tracking-wide text-muted-foreground">{label}</Label>
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
      onChange={(e) => onChange(Number(e.target.value) || 0)}
    />
  );
}
