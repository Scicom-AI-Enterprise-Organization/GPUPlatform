import { redirect } from "next/navigation";
import { getMe } from "@/lib/me";
import { ConsoleTopbar } from "@/components/console/topbar";
import { UsageReportView } from "./usage-report";

export default async function UsageReportsPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Usage Reports" }]} username={me.username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <UsageReportView />
      </div>
    </div>
  );
}
