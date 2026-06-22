import { notFound } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { gateway } from "@/lib/gateway";
import { getMe } from "@/lib/me";
import { EndpointDetail } from "./endpoint-detail";

export default async function EndpointPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const me = await getMe();
  const username = me?.username ?? "";
  let app;
  try {
    app = await gateway.getApp(id);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes("404")) notFound();
    return (
      <div className="flex h-full flex-col">
        <ConsoleTopbar
          crumbs={[{ label: "Serverless Inference", href: "/serverless" }, { label: id }]}
          username={username}
        />
        <div className="flex-1 px-6 py-8">
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            Couldn&apos;t reach the gateway: {msg}
          </div>
        </div>
      </div>
    );
  }

  // A non-owner viewing a public endpoint gets the read-only UI. The gateway is
  // the real enforcer (redacted record + owner-gated writes); this just hides the
  // edit controls. Admins and the owner get the full UI.
  const readOnly = !(me?.is_admin || (me?.username && me.username === app.owner));

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Serverless Inference", href: "/serverless" }, { label: app.name }]}
        username={username}
      />
      <EndpointDetail app={app} readOnly={readOnly} isAdmin={me?.is_admin ?? false} />
    </div>
  );
}
