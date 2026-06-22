import { redirect } from "next/navigation";
import { getMe } from "@/lib/me";
import { ConsoleTopbar } from "@/components/console/topbar";
import { ActivityDashboard } from "./activity-dashboard";

// Usage activity — OpenRouter-style dashboard over all serverless + LLM-proxy
// requests (who / endpoint / model / time / tokens / TTFT / latency). Admin-only,
// like the rest of /v1/history.
export default async function ActivityPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");
  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Activity" }]} username={me.username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <ActivityDashboard />
      </div>
    </div>
  );
}
