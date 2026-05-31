import { ConsoleTopbar } from "@/components/console/topbar";
import { NoAccessAlert } from "@/components/no-access-alert";
import { gateway } from "@/lib/gateway";
import type { ComputeTemplate } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { NewPodForm } from "./new-pod-form";

async function loadTemplates(): Promise<{ templates: ComputeTemplate[]; error: string | null }> {
  try {
    const templates = await gateway.listComputeTemplates();
    return { templates, error: null };
  } catch (e) {
    return { templates: [], error: e instanceof Error ? e.message : String(e) };
  }
}

export default async function NewComputePage() {
  const me = await getMe();
  const noAccess = me?.role === "user";
  const [{ templates, error }, username] = await Promise.all([
    noAccess ? Promise.resolve({ templates: [], error: null }) : loadTemplates(),
    currentUsername(),
  ]);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Compute", href: "/compute" }, { label: "New pod" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">New pod</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Pick a GPU + template; we&apos;ll provision it on RunPod and surface
            SSH and JupyterLab when it&apos;s ready.
          </p>
        </div>

        {noAccess && <NoAccessAlert />}

        {error && !noAccess && (
          <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            Couldn&apos;t load templates: {error}
          </div>
        )}

        {!noAccess && <NewPodForm templates={templates} />}
      </div>
    </div>
  );
}
