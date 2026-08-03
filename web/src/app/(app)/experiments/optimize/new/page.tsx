import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { currentUsername } from "@/lib/current-user";
import { gateway } from "@/lib/gateway";
import { getMe } from "@/lib/me";
import type { PromptOptLimits } from "@/lib/types";
import { OptimizeForm } from "./optimize-form";

export default async function NewOptimizationPage({
  searchParams,
}: {
  searchParams: Promise<{ dataset?: string }>;
}) {
  const me = await getMe();
  const sections = me?.sections as Record<string, boolean> | undefined;
  const noAccess = me ? !(me.is_admin || sections?.experiments) : false;
  const [username, sp] = await Promise.all([currentUsername(), searchParams]);

  // Only used when the gateway is unreachable — the real ceilings come from
  // GET /v1/prompt-optimizations/limits so the form can't drift from what the
  // runner enforces.
  const FALLBACK_LIMITS: PromptOptLimits = {
    max_metric_calls: 5000,
    max_rows: 200,
    default_rows: 50,
    max_concurrency: 64,
    default_concurrency: 8,
    auto_budgets: { light: 6, medium: 15, heavy: 40 },
    default_minibatch: 3,
    components: [],
  };

  const [datasets, registry, targets, limits] = noAccess
    ? [[], { evaluators: [], always_on: [] }, { targets: [], gateway_url: "" }, FALLBACK_LIMITS]
    : await Promise.all([
        gateway.listExperimentDatasets().catch(() => []),
        gateway.listEvaluators().catch(() => ({ evaluators: [], always_on: [] })),
        gateway.listExperimentTargets().catch(() => ({ targets: [], gateway_url: "" })),
        gateway.promptOptLimits().catch(() => FALLBACK_LIMITS),
      ]);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[
          { label: "Experiments", href: "/experiments" },
          { label: "Optimize", href: "/experiments/optimize" },
          { label: "New" },
        ]}
        username={username}
      />
      {/* `relative` makes this the containing block for Radix <Switch>'s
          absolutely-positioned form-bubble input, which would otherwise anchor to
          <html> and stretch the document past the sticky action bar. */}
      <div className="relative flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        {noAccess ? (
          <NoAccessAlert />
        ) : (
          <OptimizeForm
            datasets={datasets}
            registry={registry}
            suggestions={targets}
            limits={limits}
            initialDatasetId={sp.dataset ?? ""}
          />
        )}
      </div>
    </div>
  );
}
