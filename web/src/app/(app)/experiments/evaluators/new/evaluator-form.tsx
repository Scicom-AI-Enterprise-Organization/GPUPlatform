"use client";

// Author a custom evaluator on its own page. Built on the same shell as
// /experiments/optimize/new (itself mirroring /benchmark/new): FormShell +
// scrollspy discovered from `data-form-section`, icon SectionCards, a 4-column
// Grid of FieldWraps, and a sticky FormFooter carrying the submit error and a
// hint explaining *why* submit is disabled. Diff against optimize-form.tsx
// before changing structure here.
//
// The Test section is the point of this form, not a nicety: `fail_when_true`
// silently inverts every result if you pick it wrong, and an expression that
// throws at runtime is reported as a non-failure — both are invisible until a
// whole run has been spent. Testing against one reply costs nothing.

import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { AlertTriangle, Check, FlaskConical, Loader2, Play, Save, Settings2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type {
  CustomEvaluatorContext,
  CustomEvaluatorRecord,
  CustomEvaluatorTestResponse,
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
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const SAMPLE_DEFAULT = '```json\n{"intent": "billing", "confidence": 0.9}\n```';

const API_FIELDS: Array<{
  key: string;
  label: string;
  hint?: string;
  placeholder?: string;
  wide?: boolean;
}> = [
  { key: "url", label: "Endpoint URL", wide: true,
    placeholder: "https://scorer.internal/score",
    hint: "The completion is POSTed as JSON. Internal hosts are fine; link-local and cloud-metadata addresses are refused." },
  { key: "passed_field", label: "Passed field", placeholder: "passed",
    hint: "Dotted path, e.g. result.verdict. Blank = the whole response." },
  { key: "score_field", label: "Score field", placeholder: "score" },
  { key: "reason_field", label: "Reason field", placeholder: "reason" },
  { key: "flags_field", label: "Flags field", placeholder: "flags" },
  { key: "api_key_secret", label: "API key secret", placeholder: "MY_SCORER_KEY",
    hint: "Name of a global secret; sent as the auth header." },
  { key: "auth_header", label: "Auth header", placeholder: "Authorization" },
  { key: "auth_prefix", label: "Auth prefix", placeholder: "Bearer " },
  { key: "timeout_s", label: "Timeout (s)", placeholder: "30" },
  { key: "concurrency", label: "Concurrency", placeholder: "4",
    hint: "In-flight calls to your endpoint." },
];

export function EvaluatorForm({
  context,
  editing,
}: {
  context?: CustomEvaluatorContext;
  /** Present when the page was opened as `?id=ce-…`. */
  editing: CustomEvaluatorRecord | null;
}) {
  const router = useRouter();
  const [name, setName] = useState(editing?.name ?? "");
  const [description, setDescription] = useState(editing?.description ?? "");
  const [mode, setMode] = useState(editing?.mode ?? "expression");
  const [code, setCode] = useState(editing?.code ?? "");
  const [failWhenTrue, setFailWhenTrue] = useState(editing?.fail_when_true ?? false);
  const [apiCfg, setApiCfg] = useState<Record<string, unknown>>(editing?.config ?? {});

  const [sample, setSample] = useState(SAMPLE_DEFAULT);
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<CustomEvaluatorTestResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [showRef, setShowRef] = useState(false);

  const pythonAllowed = context?.python_allowed ?? false;
  const pythonBlocked = mode === "python" && !pythonAllowed;
  const isApi = mode === "api";
  const codeReady = isApi ? !!String(apiCfg.url ?? "").trim() : !!code.trim();

  const blocked = !name.trim()
    ? "no-name"
    : pythonBlocked
      ? "python-blocked"
      : !codeReady
        ? isApi ? "no-url" : "no-code"
        : null;

  function setApiField(key: string, value: string) {
    setApiCfg((c) => ({
      ...c,
      // "" is meaningful for passed_field (whole response) and auth_prefix (no
      // prefix), so keep it rather than deleting the key.
      [key]: key === "timeout_s" || key === "concurrency" ? Number(value) || value : value,
    }));
    setResult(null);
  }

  async function onTest() {
    setTesting(true);
    setResult(null);
    try {
      setResult(
        await gateway.testCustomEvaluator({
          mode, code, config: apiCfg, fail_when_true: failWhenTrue,
          name: name.trim() || "preview", content: sample,
        }),
      );
    } catch (e) {
      setResult({ ok: false, error: e instanceof Error ? e.message : String(e) });
    } finally {
      setTesting(false);
    }
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (blocked) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const body = {
        name: name.trim(),
        description: description.trim(),
        mode,
        code,
        config: apiCfg,
        fail_when_true: failWhenTrue,
      };
      const row = editing
        ? await gateway.updateCustomEvaluator(editing.id, body)
        : await gateway.createCustomEvaluator(body);
      toast.success(editing ? `Updated ${row.name}` : `Saved ${row.name}`);
      router.push("/experiments/evaluators");
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }

  function loadExample(exName: string) {
    const ex = context?.examples.find((x) => x.name === exName);
    if (!ex) return;
    setMode(ex.mode);
    setCode(ex.code);
    setFailWhenTrue(ex.fail_when_true);
    if (!name.trim()) setName(ex.name);
    setResult(null);
  }

  return (
    <FormShell>
      <form onSubmit={onSubmit} className="space-y-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            {editing ? `Edit ${editing.name}` : "New evaluator"}
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            A reusable check that scores every reply in an experiment. Saved to your library and
            available to any run — which <span className="font-medium">snapshots</span> the
            definition, so editing it later never changes what a finished run measured.
          </p>
        </div>

        {/* ---------------- Identity ---------------- */}
        <SectionCard
          icon={<FlaskConical className="h-4 w-4" />}
          title="Evaluator"
          description="What it's called and which way its result reads."
        >
          <Grid>
            <FieldWrap label="Name" hint="Becomes its column in the results.">
              <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="no-fence" />
            </FieldWrap>
            <FieldWrap label="Mode">
              <Select value={mode} onValueChange={(v) => { setMode(v); setResult(null); }}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="expression">Expression</SelectItem>
                  <SelectItem value="api">API endpoint</SelectItem>
                  <SelectItem value="python" disabled={!pythonAllowed}>
                    Python function{pythonAllowed ? "" : " (disabled)"}
                  </SelectItem>
                </SelectContent>
              </Select>
            </FieldWrap>
            <FieldWrap label="When true" hint="Which way the result reads.">
              <Select
                value={failWhenTrue ? "fail" : "pass"}
                onValueChange={(v) => { setFailWhenTrue(v === "fail"); setResult(null); }}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="pass">counts as PASS</SelectItem>
                  <SelectItem value="fail">counts as FAIL</SelectItem>
                </SelectContent>
              </Select>
            </FieldWrap>
            <FieldWrap label="Start from">
              <Select value="" onValueChange={loadExample}>
                <SelectTrigger>
                  <SelectValue placeholder="an example…" />
                </SelectTrigger>
                <SelectContent>
                  {(context?.examples ?? [])
                    .filter((ex) => ex.mode !== "python" || pythonAllowed)
                    .map((ex) => (
                      <SelectItem key={ex.name} value={ex.name}>
                        {ex.name}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </FieldWrap>
            <FieldWrap label="Description" wide>
              <Input
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Reply must not be wrapped in a markdown fence."
              />
            </FieldWrap>
          </Grid>

          {pythonBlocked && (
            <div className="mt-4 flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-400">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>
                Python evaluators execute arbitrary code on the gateway host, so they require admin
                role — and are unavailable entirely when{" "}
                <code className="font-mono">{context?.python_env_var}=0</code> is set in the gateway
                environment. Expression mode needs neither.
              </span>
            </div>
          )}
        </SectionCard>

        {/* ---------------- Definition ---------------- */}
        <SectionCard
          icon={<Settings2 className="h-4 w-4" />}
          title="Definition"
          description={
            isApi
              ? "Your endpoint scores the reply; nothing runs on the gateway."
              : mode === "python"
                ? "def check(c) — returns a bool, a number, or a dict."
                : "One Python expression over the reply."
          }
          action={
            <button
              type="button"
              onClick={() => setShowRef((v) => !v)}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              {showRef ? "Hide reference" : isApi ? "What gets sent?" : "What can I use?"}
            </button>
          }
        >
          {isApi ? (
            <Grid>
              {API_FIELDS.map((f) => (
                <FieldWrap key={f.key} label={f.label} hint={f.hint} wide={f.wide}>
                  <Input
                    value={
                      apiCfg[f.key] === undefined || apiCfg[f.key] === null
                        ? ""
                        : String(apiCfg[f.key])
                    }
                    onChange={(e) => setApiField(f.key, e.target.value)}
                    placeholder={f.placeholder}
                    className={f.key === "url" ? "font-mono text-xs" : undefined}
                  />
                </FieldWrap>
              ))}
            </Grid>
          ) : (
            <Textarea
              value={code}
              onChange={(e) => { setCode(e.target.value); setResult(null); }}
              rows={mode === "python" ? 12 : 4}
              spellCheck={false}
              className="font-mono text-xs"
              placeholder={
                mode === "python"
                  ? 'def check(c):\n    return {"passed": len(c["content"]) > 0}'
                  : 're_search("```", content)'
              }
            />
          )}

          {showRef && context && <Reference context={context} mode={mode} />}
        </SectionCard>

        {/* ---------------- Test ---------------- */}
        <SectionCard
          icon={<Play className="h-4 w-4" />}
          title="Test"
          description="Run it against one reply before spending a whole experiment on it."
          action={
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => void onTest()}
              disabled={testing || !codeReady || pythonBlocked}
            >
              {testing ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Play className="h-3.5 w-3.5" />
              )}
              Test
            </Button>
          }
        >
          <Textarea
            value={sample}
            onChange={(e) => setSample(e.target.value)}
            rows={4}
            spellCheck={false}
            className="font-mono text-xs"
            placeholder="Paste a model reply to check against…"
          />
          {result && (
            <div
              className={cn(
                "mt-3 rounded-md border px-3 py-2 text-xs",
                !result.ok
                  ? "border-destructive/40 bg-destructive/10 text-destructive"
                  : result.passed
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
                    : "border-red-500/40 bg-red-500/10 text-red-700 dark:text-red-400",
              )}
            >
              {!result.ok ? (
                <span className="font-mono">{result.error}</span>
              ) : (
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <span className="inline-flex items-center gap-1 font-medium">
                    {result.passed ? <Check className="h-3.5 w-3.5" /> : <X className="h-3.5 w-3.5" />}
                    {result.passed ? "PASS" : "FAIL"}
                  </span>
                  {result.score !== null && result.score !== undefined && (
                    <span className="tabular-nums opacity-80">score {result.score}</span>
                  )}
                  {result.reason && <span className="opacity-80">{result.reason}</span>}
                  {result.flags && Object.keys(result.flags).length > 0 && (
                    <span className="font-mono opacity-70">{JSON.stringify(result.flags)}</span>
                  )}
                </div>
              )}
            </div>
          )}
        </SectionCard>

        <FormFooter
          error={submitError}
          hint={
            blocked === "no-name" ? (
              "Give the evaluator a name — it becomes its column in the results."
            ) : blocked === "python-blocked" ? (
              "Python mode needs admin role on this gateway."
            ) : blocked === "no-url" ? (
              "An API evaluator needs an endpoint URL."
            ) : blocked === "no-code" ? (
              mode === "python" ? "Write a check(c) function." : "Write an expression."
            ) : (
              <>
                Saved to your library and selectable on any experiment. Runs already using it keep
                their own snapshot — see{" "}
                <Link
                  href="/experiments/evaluators"
                  className="underline underline-offset-2 hover:text-foreground"
                >
                  Evaluators
                </Link>
                .
              </>
            )
          }
        >
          <Button
            type="button"
            variant="outline"
            onClick={() => router.push("/experiments/evaluators")}
          >
            Cancel
          </Button>
          <Button type="submit" disabled={submitting || !!blocked} className="min-w-36">
            {submitting ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Saving…
              </>
            ) : (
              <>
                <Save className="h-4 w-4" />
                {editing ? "Save changes" : "Save evaluator"}
              </>
            )}
          </Button>
        </FormFooter>
      </form>
    </FormShell>
  );
}

function Reference({ context, mode }: { context: CustomEvaluatorContext; mode: string }) {
  return (
    <div className="mt-4 space-y-3 rounded-md border border-border bg-muted/30 p-3 text-xs">
      <div>
        <p className="mb-1 font-medium">
          {mode === "python" ? "c is a dict with these keys" : "Variables"}
        </p>
        <div className="flex flex-wrap gap-1">
          {context.variables.map((v) => (
            <Badge key={v.name} variant="secondary" className="font-mono text-[10px]">
              {v.name}
            </Badge>
          ))}
        </div>
      </div>
      {mode === "api" ? (
        <>
          <p className="text-muted-foreground">
            Your endpoint receives a JSON POST whose body is exactly the fields above (plus{" "}
            <code className="font-mono">evaluator</code>, the name). Answer with JSON and point the
            field paths at it — dotted paths like{" "}
            <code className="font-mono">result.verdict</code> walk objects and arrays.
          </p>
          <pre className="overflow-x-auto rounded border border-border bg-background p-2 font-mono text-[11px] scrollbar-thin">
{`{"result": {"verdict": "PASS", "confidence": 0.91,
            "detail": {"why": "greeting present"}}}

passed_field: result.verdict      score_field: result.confidence
reason_field: result.detail.why`}
          </pre>
          <p className="text-muted-foreground">
            A verdict may be a bool, a number, or a word —{" "}
            <code className="font-mono">PASS/FAIL</code>, <code className="font-mono">yes/no</code>,{" "}
            <code className="font-mono">true/false</code> all read correctly. Nothing runs on the
            gateway, so this mode needs no sandbox.
          </p>
        </>
      ) : mode === "expression" ? (
        <>
          <div>
            <p className="mb-1 font-medium">Helpers</p>
            <div className="flex flex-wrap gap-1">
              {context.helpers.map((h) => (
                <Badge key={h} variant="secondary" className="font-mono text-[10px]">
                  {h}
                </Badge>
              ))}
            </div>
          </div>
          <p className="text-muted-foreground">
            One expression, no statements. Loops, comprehensions, imports and{" "}
            <code className="font-mono">**</code> are rejected — string/dict methods like{" "}
            <code className="font-mono">.lower()</code> and <code className="font-mono">.get()</code>{" "}
            work.
          </p>
        </>
      ) : (
        <p className="text-muted-foreground">
          Define <code className="font-mono">check(c)</code>. Return a bool, a number, or{" "}
          <code className="font-mono">{'{"passed":…, "score":…, "reason":…, "flags":{…}}'}</code>.
          Runs in a separate process with the gateway&apos;s credentials stripped and CPU, memory
          and wall-clock limits. A dict states its own verdict, so{" "}
          <span className="font-medium">When true</span> does not flip it.
        </p>
      )}
    </div>
  );
}

// ---- form shell primitives (copied from optimize-form.tsx — the existing
// convention is to keep these in sync per-route rather than import across) ----

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
    <div className={cn("space-y-1.5", wide ? "sm:col-span-2 lg:col-span-4" : "")}>
      <Label className="text-xs uppercase tracking-wide text-muted-foreground">{label}</Label>
      {children}
      {hint && <p className="text-[11px] leading-snug text-muted-foreground">{hint}</p>}
    </div>
  );
}
