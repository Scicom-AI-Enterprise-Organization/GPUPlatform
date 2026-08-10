import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { currentUsername } from "@/lib/current-user";
import { gateway } from "@/lib/gateway";
import { getMe } from "@/lib/me";
import type { SandboxRegistry } from "@/lib/types";
import { SandboxForm } from "./sandbox-form";

const EMPTY: SandboxRegistry = {
  modes: [], loop_options: [], implemented_modes: [], max_tool_rounds_cap: 20,
  sandboxes: [], python_allowed: false,
};

export default async function NewSandboxPage({
  searchParams,
}: {
  searchParams: Promise<{ id?: string }>;
}) {
  const me = await getMe();
  const sections = me?.sections as Record<string, boolean> | undefined;
  const noAccess = me ? !(me.is_admin || sections?.experiments) : false;
  const [username, sp] = await Promise.all([currentUsername(), searchParams]);

  const [registry, datasets] = noAccess
    ? [EMPTY, []]
    : await Promise.all([
        gateway.listSandboxes().catch(() => EMPTY),
        gateway.listExperimentDatasets().catch(() => []),
      ]);

  // `?id=` edits an existing entry, resolved from the (unscoped) library.
  const editing = sp.id ? registry.sandboxes.find((s) => s.id === sp.id) ?? null : null;

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[
          { label: "Experiments", href: "/experiments" },
          { label: "Sandboxes", href: "/experiments/sandboxes" },
          { label: editing ? editing.name : "New" },
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
          <SandboxForm
            registry={registry}
            datasets={datasets.filter((d) => d.usable)}
            editing={editing}
          />
        )}
      </div>
    </div>
  );
}
