import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { ImportTrainingForm } from "./import-form";

export default async function ImportTrainingPage() {
  const me = await getMe();
  const sections = me?.sections as Record<string, boolean> | undefined;
  const noAccess = me ? !(me.is_admin || sections?.autotrain) : false;
  const username = await currentUsername();

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Autotrain" }, { label: "Import" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Import training run</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Upload an <span className="font-mono text-xs">.autotrain.json</span> exported from
            another deployment (a run&apos;s <span className="font-medium">Export</span> button).
            It re-creates the run here with its config, metrics/loss, and logs — so you can inspect
            results a teammate produced elsewhere. Large checkpoints aren&apos;t embedded, so the
            import is view-only (it can&apos;t be resumed or served).
          </p>
        </div>
        {noAccess ? <NoAccessAlert /> : <ImportTrainingForm />}
      </div>
    </div>
  );
}
