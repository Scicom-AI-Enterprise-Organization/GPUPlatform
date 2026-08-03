import { notFound } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { gateway } from "@/lib/gateway";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { OptimizeDetail } from "./optimize-detail";

export default async function OptimizationPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  // Only `opt-…` ids are runs — same guard as /experiments/[id], so a stale path
  // like /experiments/optimize/new can't resolve here.
  if (!id.startsWith("opt-")) notFound();
  const me = await getMe();
  const sections = me?.sections as Record<string, boolean> | undefined;
  const noAccess = me ? !(me.is_admin || sections?.experiments) : false;
  const username = await currentUsername();

  if (noAccess) {
    return (
      <div className="flex h-full flex-col">
        <ConsoleTopbar
          crumbs={[{ label: "Experiments", href: "/experiments" }, { label: "Optimize" }]}
          username={username}
        />
        <div className="flex-1 px-6 py-6 lg:px-10 lg:py-8">
          <NoAccessAlert />
        </div>
      </div>
    );
  }

  let optimization;
  try {
    optimization = await gateway.getPromptOpt(id);
  } catch {
    notFound();
  }

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[
          { label: "Experiments", href: "/experiments" },
          { label: "Optimize", href: "/experiments/optimize" },
          { label: optimization.name },
        ]}
        username={username}
      />
      <OptimizeDetail initial={optimization} />
    </div>
  );
}
