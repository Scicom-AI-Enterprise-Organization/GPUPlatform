import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { gateway } from "@/lib/gateway";
import type { QuantizationJobRecord } from "@/lib/types";
import { QuantizationDetail } from "./quantization-detail";

export default async function QuantizationJobPage({
  params,
}: {
  params: Promise<{ jobId: string }>;
}) {
  const { jobId } = await params;
  const username = await currentUsername();
  let job: QuantizationJobRecord | null = null;
  let error: string | null = null;
  try {
    job = await gateway.getQuantizationJob(jobId);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Quantization", href: "/quantization" }, { label: job?.name ?? jobId }]}
        username={username}
      />
      {error || !job ? (
        <div className="flex-1 px-6 py-8 lg:px-10">
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error ?? "Job not found."}
          </div>
        </div>
      ) : (
        <QuantizationDetail initial={job} />
      )}
    </div>
  );
}
