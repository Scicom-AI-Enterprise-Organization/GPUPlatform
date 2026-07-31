import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { gateway } from "@/lib/gateway";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { SectionTabs } from "../section-tabs";
import { EvaluatorsManager } from "./evaluators-manager";

export default async function EvaluatorsPage() {
  const me = await getMe();
  const sections = me?.sections as Record<string, boolean> | undefined;
  const noAccess = me ? !(me.is_admin || sections?.experiments) : false;
  const username = await currentUsername();

  const [registry, error] = noAccess
    ? [null, null]
    : await gateway
        .listEvaluators()
        .then((r) => [r, null] as const)
        .catch((e) => [null, e instanceof Error ? e.message : String(e)] as const);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Experiments", href: "/experiments" }, { label: "Evaluators" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Evaluators</h1>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
            The checks that score every reply. Built-ins ship with the platform; the ones you
            write here are saved to your library and can be reused across any experiment.
          </p>
        </div>

        <SectionTabs />

        {noAccess && <NoAccessAlert />}

        {error && !noAccess && (
          <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            Couldn&apos;t reach the gateway: {error}
          </div>
        )}

        {!noAccess && registry && <EvaluatorsManager registry={registry} />}
      </div>
    </div>
  );
}
