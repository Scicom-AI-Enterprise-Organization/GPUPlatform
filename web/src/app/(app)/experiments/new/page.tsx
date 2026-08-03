import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { currentUsername } from "@/lib/current-user";
import { gateway } from "@/lib/gateway";
import { getMe } from "@/lib/me";
import type { ExperimentLimits } from "@/lib/types";
import { ExperimentForm } from "./experiment-form";

export default async function NewExperimentPage({
  searchParams,
}: {
  searchParams: Promise<{ dataset?: string; from?: string; prompt?: string }>;
}) {
  const me = await getMe();
  const sections = me?.sections as Record<string, boolean> | undefined;
  const noAccess = me ? !(me.is_admin || sections?.experiments) : false;
  const [username, sp] = await Promise.all([currentUsername(), searchParams]);

  // Fail soft: an unreachable gateway shouldn't blank the form — the client
  // surfaces the error and the user can still retry.
  // Only used when the gateway is unreachable — the real ceilings come from
  // GET /v1/experiments/limits so the form can't drift from what the runner enforces.
  const FALLBACK_LIMITS: ExperimentLimits = {
    max_units: 20000, max_rows: 2000, max_concurrency: 64, default_concurrency: 8,
    sweep_row_threshold: 20, sweep_sample_rows: 200,
  };
  const [datasets, storages, registry, targets, limits] = noAccess
    ? [[], [], { evaluators: [], always_on: [] }, { targets: [], gateway_url: "" }, FALLBACK_LIMITS]
    : await Promise.all([
        // Datasets come from the platform's own section — Experiments keeps no
        // parallel store; this is /datasets filtered to chat-shaped kinds.
        gateway.listExperimentDatasets().catch(() => []),
        gateway.listStorage().catch(() => []),
        gateway.listEvaluators().catch(() => ({ evaluators: [], always_on: [] })),
        gateway.listExperimentTargets().catch(() => ({ targets: [], gateway_url: "" })),
        gateway.experimentLimits().catch(() => FALLBACK_LIMITS),
      ]);

  // "Run it again" — clone an earlier experiment's whole matrix.
  let clone;
  if (sp.from && !noAccess) {
    try {
      clone = await gateway.getExperiment(sp.from);
    } catch {
      // ignore — fall back to the default empty form
    }
  }

  // "?prompt=opt-…" — confirm a GEPA result. The form opens with TWO variants,
  // baseline and optimized, on the dataset the search used: the optimizer's own
  // number is measured on a validation slice, and this is the run that checks it
  // on the whole corpus with the full evaluator stack.
  let optimized;
  if (sp.prompt && !noAccess) {
    try {
      const run = await gateway.getPromptOpt(sp.prompt);
      const text = run.result?.best?.texts?.system_prompt ?? "";
      if (text) {
        optimized = {
          id: run.id,
          name: run.name,
          prompt: text,
          user_suffix: run.result?.best?.texts?.user_suffix ?? "",
          dataset_id: run.dataset_id,
        };
      }
    } catch {
      // ignore — fall back to the default empty form
    }
  }

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Experiments", href: "/experiments" }, { label: "New experiment" }]}
        username={username}
      />
      {/* Inner scroller with symmetric vertical padding (mirrors /benchmark/new)
          so the sticky action bar keeps bottom breathing room. `relative` makes
          this the containing block for Radix <Switch>'s absolutely-positioned
          form-bubble input, which would otherwise anchor to <html> and stretch
          the document past the action bar. */}
      <div className="relative flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        {noAccess ? (
          <NoAccessAlert />
        ) : (
          <ExperimentForm
            datasets={datasets}
            storages={storages.filter((s) => s.kind === "s3")}
            registry={registry}
            suggestions={targets}
            limits={limits}
            initialDatasetId={sp.dataset ?? optimized?.dataset_id ?? ""}
            clone={clone}
            optimized={optimized}
          />
        )}
      </div>
    </div>
  );
}
