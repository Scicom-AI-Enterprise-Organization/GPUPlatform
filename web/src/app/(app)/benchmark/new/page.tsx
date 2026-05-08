import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { BenchmarkForm } from "./benchmark-form";

export default async function NewBenchmarkPage() {
  const me = await getMe();
  const noAccess = me?.role === "user";
  const username = await currentUsername();

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[
          { label: "Benchmark", href: "/benchmark" },
          { label: "New benchmark" },
        ]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        {noAccess ? <NoAccessAlert /> : <BenchmarkForm />}
      </div>
    </div>
  );
}
