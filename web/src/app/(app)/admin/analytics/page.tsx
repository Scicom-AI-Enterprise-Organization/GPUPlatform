import { redirect } from "next/navigation";
import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";
import { getMe } from "@/lib/me";
import { ConsoleTopbar } from "@/components/console/topbar";
import { FormShell } from "@/components/form-shell";

const AnalyticsView = dynamic(
  () => import("./analytics-view").then((mod) => mod.AnalyticsView),
  {
    loading: () => (
      <div className="flex h-40 items-center justify-center rounded-lg border bg-card text-sm text-muted-foreground">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading analytics...
      </div>
    ),
  },
);

export default async function AnalyticsPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Analytics" }]} username={me.username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <FormShell>
        <div>
        <header className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Analytics</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Platform-wide GPU usage and spend across GPU Platform (serverless, endpoints,
            benchmark, autotrain, compute) and SlurmUI jobs.
          </p>
        </header>
        <AnalyticsView />
        </div>
        </FormShell>
      </div>
    </div>
  );
}
