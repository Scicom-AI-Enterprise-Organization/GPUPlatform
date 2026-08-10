import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { currentUsername } from "@/lib/current-user";
import { gateway } from "@/lib/gateway";
import { getMe } from "@/lib/me";
import { EvaluatorForm } from "./evaluator-form";

export default async function NewEvaluatorPage({
  searchParams,
}: {
  searchParams: Promise<{ id?: string }>;
}) {
  const me = await getMe();
  const sections = me?.sections as Record<string, boolean> | undefined;
  const noAccess = me ? !(me.is_admin || sections?.experiments) : false;
  const [username, sp] = await Promise.all([currentUsername(), searchParams]);

  const registry = noAccess
    ? { evaluators: [], always_on: [] }
    : await gateway.listEvaluators().catch(() => ({ evaluators: [], always_on: [] }));

  // `?id=` edits an existing entry. There's no GET-one route for custom
  // evaluators, and the list is unscoped now, so resolve it from the library.
  const editing = sp.id ? (registry.custom ?? []).find((c) => c.id === sp.id) ?? null : null;

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[
          { label: "Experiments", href: "/experiments" },
          { label: "Evaluators", href: "/experiments/evaluators" },
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
          <EvaluatorForm context={registry.custom_context} editing={editing} />
        )}
      </div>
    </div>
  );
}
