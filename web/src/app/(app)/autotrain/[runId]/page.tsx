import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { gateway } from "@/lib/gateway";
import type { TrainingRunRecord } from "@/lib/types";
import { TrainingDetail } from "./training-detail";

export default async function TrainingRunPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;
  const username = await currentUsername();
  let run: TrainingRunRecord | null = null;
  let error: string | null = null;
  try {
    run = await gateway.getTrainingRun(runId);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[
          { label: "Autotrain", href: "/autotrain" },
          { label: run?.name ?? runId },
        ]}
        username={username}
      />
      {error || !run ? (
        <div className="flex-1 px-6 py-8 lg:px-10">
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error ?? "Run not found."}
          </div>
        </div>
      ) : (
        <TrainingDetail initial={run} />
      )}
    </div>
  );
}
