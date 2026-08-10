"use client";

// Author a sandbox on its own page. Same shell as /experiments/optimize/new —
// FormShell + scrollspy from `data-form-section`, icon SectionCards, a 4-column
// Grid of FieldWraps, sticky FormFooter. Diff against optimize-form.tsx before
// changing structure here.
//
// Every field below is server-driven (`registry.modes[].options` +
// `loop_options`), so a new mode or option in sandbox.py needs no change here.
//
// The Test section is load-bearing: a sandbox that answers NOTHING still
// produces trajectories, and the detectors score them happily — so a run can
// come back green having measured a broken environment. Resolving one call
// against one real row is what catches a missing seed column, a URL that 404s,
// or a result path that isn't in the response.

import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { AlertTriangle, Boxes, Check, Loader2, Play, Repeat, Save } from "lucide-react";
import { cn } from "@/lib/utils";
import { gateway } from "@/lib/gateway";
import type {
  CustomSandboxRecord,
  CustomSandboxTestResponse,
  EvaluatorOption,
  ExperimentDatasetOption,
  SandboxRegistry,
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type Cfg = Record<string, Record<string, unknown>>;

export function SandboxForm({
  registry,
  datasets,
  editing,
}: {
  registry: SandboxRegistry;
  datasets: ExperimentDatasetOption[];
  /** Present when the page was opened as `?id=sb-…`. */
  editing: CustomSandboxRecord | null;
}) {
  const router = useRouter();
  const [name, setName] = useState(editing?.name ?? "");
  const [description, setDescription] = useState(editing?.description ?? "");
  const [mode, setMode] = useState(editing?.mode ?? "replay");
  const [cfg, setCfg] = useState<Cfg>((editing?.config as Cfg) ?? {});

  const [datasetId, setDatasetId] = useState("");
  const [rowIndex, setRowIndex] = useState("0");
  const [toolName, setToolName] = useState("");
  const [toolArgs, setToolArgs] = useState("{}");
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<CustomSandboxTestResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const spec = registry.modes.find((m) => m.id === mode);
  const notImplemented = !!spec && !spec.implemented;
  const needsUrl = mode === "api" && !String(cfg.api?.url ?? "").trim();

  const blocked = !name.trim()
    ? "no-name"
    : notImplemented
      ? "not-implemented"
      : needsUrl
        ? "no-url"
        : null;

  function setField(group: string, key: string, value: unknown) {
    setCfg((c) => ({ ...c, [group]: { ...(c[group] ?? {}), [key]: value } }));
    setResult(null);
  }

  async function onTest() {
    setTesting(true);
    setResult(null);
    try {
      const call = toolName.trim()
        ? {
            id: "test",
            type: "function",
            function: { name: toolName.trim(), arguments: toolArgs || "{}" },
          }
        : {};
      setResult(
        await gateway.testCustomSandbox({
          mode,
          config: cfg,
          tool_call: call,
          dataset_id: datasetId || undefined,
          row_index: Number(rowIndex) || 0,
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
        code: "",
        config: cfg,
      };
      const row = editing
        ? await gateway.updateCustomSandbox(editing.id, body)
        : await gateway.createCustomSandbox(body);
      toast.success(editing ? `Updated ${row.name}` : `Saved ${row.name}`);
      router.push("/experiments/sandboxes");
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }

  return (
    <FormShell>
      <form onSubmit={onSubmit} className="space-y-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            {editing ? `Edit ${editing.name}` : "New sandbox"}
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            A sandbox answers the model&apos;s tool calls during a replay, so a dataset row becomes
            a whole conversation instead of a single request. A run{" "}
            <span className="font-medium">snapshots</span> the definition — it&apos;s the
            environment the model was measured in, so editing it later would make old numbers
            incomparable rather than merely different.
          </p>
        </div>

        {/* ---------------- Identity ---------------- */}
        <SectionCard
          icon={<Boxes className="h-4 w-4" />}
          title="Sandbox"
          description="What it's called and where its answers come from."
        >
          <Grid>
            <FieldWrap label="Name">
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="care-simulator"
              />
            </FieldWrap>
            <FieldWrap label="Mode">
              <Select value={mode} onValueChange={(v) => { setMode(v); setResult(null); }}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {registry.modes.map((m) => (
                    <SelectItem key={m.id} value={m.id} disabled={!m.implemented}>
                      {m.label}
                      {m.implemented ? "" : " (not yet)"}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </FieldWrap>
            <FieldWrap label="Description" wide>
              <Input
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Replays the reference tool results carried on each row."
              />
            </FieldWrap>
          </Grid>

          {spec && (
            <p className="mt-4 text-xs leading-snug text-muted-foreground">{spec.description}</p>
          )}
          {notImplemented && (
            <div className="mt-3 flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-400">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>This mode isn&apos;t implemented in the gateway yet, so it can&apos;t be saved.</span>
            </div>
          )}
        </SectionCard>

        {/* ---------------- Source (server-driven per mode) ---------------- */}
        {spec && spec.options.length > 0 && (
          <SectionCard
            icon={<Boxes className="h-4 w-4" />}
            title={mode === "api" ? "Endpoint" : "Source"}
            description="Where a tool result comes from. Never a fabricated success: a call the sandbox can't answer gets a deterministic error the model sees."
          >
            <OptionGrid
              options={spec.options}
              values={cfg[mode] ?? {}}
              onChange={(k, v) => setField(mode, k, v)}
            />
          </SectionCard>
        )}

        {/* ---------------- Loop ---------------- */}
        <SectionCard
          icon={<Repeat className="h-4 w-4" />}
          title="Loop"
          description="How far the conversation runs. Each round is another billed model call for every row × target × variant × repeat."
        >
          <OptionGrid
            options={registry.loop_options}
            values={cfg.loop ?? {}}
            onChange={(k, v) => setField("loop", k, v)}
          />
        </SectionCard>

        {/* ---------------- Test ---------------- */}
        <SectionCard
          icon={<Play className="h-4 w-4" />}
          title="Test"
          description="Resolve one call against one real row before spending a whole matrix on it."
          action={
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => void onTest()}
              disabled={testing || notImplemented || needsUrl}
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
          <Grid>
            <FieldWrap label="Dataset" hint="Its rows carry the reference results.">
              <Select value={datasetId} onValueChange={(v) => { setDatasetId(v); setResult(null); }}>
                <SelectTrigger>
                  <SelectValue placeholder="pick a dataset…" />
                </SelectTrigger>
                <SelectContent>
                  {datasets.map((d) => (
                    <SelectItem key={d.id} value={d.id}>
                      {d.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </FieldWrap>
            <FieldWrap label="Row">
              <Input value={rowIndex} onChange={(e) => setRowIndex(e.target.value)} placeholder="0" />
            </FieldWrap>
            <FieldWrap
              label="Tool"
              hint={mode === "replay" ? "Blank lists what the row's seed offers." : "Any name your service answers."}
            >
              <Input
                value={toolName}
                onChange={(e) => { setToolName(e.target.value); setResult(null); }}
                placeholder="get_balance"
                className="font-mono text-xs"
              />
            </FieldWrap>
            {toolName.trim() && (
              <FieldWrap label="Arguments (JSON)">
                <Input
                  value={toolArgs}
                  onChange={(e) => setToolArgs(e.target.value)}
                  className="font-mono text-xs"
                />
              </FieldWrap>
            )}
          </Grid>

          {result && <TestResult result={result} />}
        </SectionCard>

        <FormFooter
          error={submitError}
          hint={
            blocked === "no-name" ? (
              "Give the sandbox a name."
            ) : blocked === "not-implemented" ? (
              "Pick a mode the gateway implements."
            ) : blocked === "no-url" ? (
              "An API sandbox needs an endpoint URL."
            ) : (
              <>
                Saved to your library and selectable on any experiment — see{" "}
                <Link
                  href="/experiments/sandboxes"
                  className="underline underline-offset-2 hover:text-foreground"
                >
                  Sandboxes
                </Link>
                .
              </>
            )
          }
        >
          <Button
            type="button"
            variant="outline"
            onClick={() => router.push("/experiments/sandboxes")}
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
                {editing ? "Save changes" : "Save sandbox"}
              </>
            )}
          </Button>
        </FormFooter>
      </form>
    </FormShell>
  );
}

function TestResult({ result }: { result: CustomSandboxTestResponse }) {
  // The seed probe (replay, no tool name) and a resolved call answer different
  // questions: "is my seed column wired up?" vs "what would the model receive?".
  const isProbe = result.seed_entries !== undefined;
  return (
    <div
      className={cn(
        "mt-4 rounded-md border px-3 py-2 text-xs",
        !result.ok
          ? "border-destructive/40 bg-destructive/10 text-destructive"
          : "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
      )}
    >
      {!result.ok ? (
        <span className="font-mono">{result.error}</span>
      ) : isProbe ? (
        <div className="space-y-1">
          <span className="inline-flex items-center gap-1 font-medium">
            <Check className="h-3.5 w-3.5" />
            {result.seed_entries} reference result{result.seed_entries === 1 ? "" : "s"} on{" "}
            {result.row ?? "this row"}
          </span>
          <div className="flex flex-wrap gap-1">
            {(result.seed_names ?? []).map((n) => (
              <Badge key={n} variant="secondary" className="font-mono text-[10px]">
                {n}
              </Badge>
            ))}
          </div>
        </div>
      ) : (
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="inline-flex items-center gap-1 font-medium">
              <Check className="h-3.5 w-3.5" />
              answered
            </span>
            <span className="opacity-80">via {result.provenance}</span>
            {result.matched && <span className="opacity-80">matched by {result.matched}</span>}
          </div>
          <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all rounded border border-current/20 bg-background/50 p-2 font-mono text-[11px] text-foreground scrollbar-thin">
            {result.content}
          </pre>
        </div>
      )}
    </div>
  );
}

/** Renders a server-supplied option schema — the same five field types the
 * evaluator form understands, so the two stay interchangeable. */
function OptionGrid({
  options,
  values,
  onChange,
}: {
  options: EvaluatorOption[];
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}) {
  return (
    <Grid>
      {options.map((opt) => {
        const raw = values[opt.name];
        const val = raw === undefined ? opt.default : raw;
        const wide = opt.name === "url" || opt.name === "send_expected";
        return (
          <FieldWrap key={opt.name} label={opt.label} hint={opt.help} wide={wide}>
            {opt.type === "boolean" ? (
              <div className="flex h-9 items-center">
                <Switch checked={!!val} onCheckedChange={(v) => onChange(opt.name, v)} />
              </div>
            ) : opt.type === "select" ? (
              <Select value={String(val ?? "")} onValueChange={(v) => onChange(opt.name, v)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {(opt.options ?? []).map((o) => (
                    <SelectItem key={String(o)} value={String(o)}>
                      {String(o)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <Input
                value={val === null || val === undefined ? "" : String(val)}
                onChange={(e) =>
                  onChange(
                    opt.name,
                    // "" stays "" — it's a MEANINGFUL value for the result path
                    // (whole response) and the auth prefix (no `Bearer `).
                    opt.type === "number"
                      ? Number(e.target.value) || e.target.value
                      : e.target.value,
                  )
                }
                className={opt.name === "url" ? "font-mono text-xs" : undefined}
              />
            )}
          </FieldWrap>
        );
      })}
    </Grid>
  );
}

// ---- form shell primitives (kept in sync per-route, the existing convention) ----

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
