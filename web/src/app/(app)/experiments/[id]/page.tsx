import { notFound } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { gateway } from "@/lib/gateway";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { ExperimentDetail } from "./experiment-detail";

export default async function ExperimentPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  // Only `exp-…` ids are experiments. Without this, a stale or mistyped path like
  // /experiments/datasets resolves against the sibling API route of the same name
  // (`GET /v1/experiments/datasets`, the dataset picker) and renders an empty
  // detail page instead of a 404.
  if (!id.startsWith("exp-")) notFound();
  const me = await getMe();
  const sections = me?.sections as Record<string, boolean> | undefined;
  const noAccess = me ? !(me.is_admin || sections?.experiments) : false;
  const username = await currentUsername();

  if (noAccess) {
    return (
      <div className="flex h-full flex-col">
        <ConsoleTopbar crumbs={[{ label: "Experiments", href: "/experiments" }]} username={username} />
        <div className="flex-1 px-6 py-6 lg:px-10 lg:py-8">
          <NoAccessAlert />
        </div>
      </div>
    );
  }

  let experiment;
  try {
    experiment = await gateway.getExperiment(id);
  } catch {
    notFound();
  }

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Experiments", href: "/experiments" }, { label: experiment.name }]}
        username={username}
      />
      <ExperimentDetail initial={experiment} />
    </div>
  );
}
