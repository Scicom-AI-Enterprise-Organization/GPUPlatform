import { redirect } from "next/navigation";
import { getMe } from "@/lib/me";
import { ConsoleTopbar } from "@/components/console/topbar";
import { AnalyticsView } from "./analytics-view";

export default async function AnalyticsPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Analytics" }]} username={me.username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <header className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Analytics</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Platform-wide GPU usage and spend across GPU Platform (serverless, endpoints,
            benchmark, autotrain, compute) and SlurmUI jobs.
          </p>
        </header>
        <AnalyticsView />
      </div>
    </div>
  );
}
