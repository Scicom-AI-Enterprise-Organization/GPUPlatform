import { redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { gateway } from "@/lib/gateway";
import type { ProviderRecord } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { ProviderMetricsView } from "./metrics-view";

export default async function ProviderMetricsPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");

  const { id } = await params;
  const username = await currentUsername();

  let provider: ProviderRecord | null = null;
  try {
    provider = (await gateway.listProviders()).find((p) => p.id === id) ?? null;
  } catch {
    /* the client view surfaces fetch errors from the live poll */
  }

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[
          { label: "GPU Providers", href: "/providers" },
          { label: provider?.name ?? id },
        ]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <ProviderMetricsView id={id} provider={provider} />
      </div>
    </div>
  );
}
