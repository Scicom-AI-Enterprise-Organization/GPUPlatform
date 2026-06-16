import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { ImportBenchmarkForm } from "./import-form";

export default async function ImportBenchmarkPage() {
  const me = await getMe();
  const noAccess = !me?.sections?.benchmark;
  const username = await currentUsername();

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Benchmark" }, { label: "Import" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Import benchmark</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Upload a <span className="font-mono text-xs">.benchmark.json</span> exported from
            another deployment (a benchmark&apos;s <span className="font-medium">Files → Export</span>).
            It re-creates the run here with its results, config, and files.
          </p>
        </div>
        {noAccess ? <NoAccessAlert /> : <ImportBenchmarkForm />}
      </div>
    </div>
  );
}
