import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { gateway } from "@/lib/gateway";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { SectionTabs } from "../section-tabs";
import { SandboxesManager } from "./sandboxes-manager";

export default async function SandboxesPage() {
  const me = await getMe();
  const sections = me?.sections as Record<string, boolean> | undefined;
  const noAccess = me ? !(me.is_admin || sections?.experiments) : false;
  const username = await currentUsername();

  const [registry, error] = noAccess
    ? [null, null]
    : await gateway
        .listSandboxes()
        .then((r) => [r, null] as const)
        .catch((e) => [null, e instanceof Error ? e.message : String(e)] as const);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Experiments", href: "/experiments" }, { label: "Sandboxes" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Sandboxes</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            A sandbox answers the model&apos;s tool calls during a replay, so a row becomes a
            whole conversation instead of a single request. Pick one on an experiment to score
            what the model does across several turns, not just its first.
          </p>
        </div>

        <SectionTabs />

        {noAccess && <NoAccessAlert />}

        {error && !noAccess && (
          <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            Couldn&apos;t reach the gateway: {error}
          </div>
        )}

        {!noAccess && registry && <SandboxesManager registry={registry} />}
      </div>
    </div>
  );
}
