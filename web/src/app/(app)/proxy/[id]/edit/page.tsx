import { notFound, redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { gateway, GatewayError } from "@/lib/gateway";
import type { ProxyEndpoint } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { ProxyForm } from "../../proxy-form";

export default async function EditProxyPage({ params }: { params: Promise<{ id: string }> }) {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");
  const { id } = await params;
  let ep: ProxyEndpoint;
  try {
    ep = await gateway.getProxy(id);
  } catch (e) {
    if (e instanceof GatewayError && e.status === 404) notFound();
    throw e;
  }
  const username = await currentUsername();
  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "LLM API Proxy", href: "/proxy" }, { label: ep.name, href: `/proxy/${ep.id}` }, { label: "Edit" }]}
        username={username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <h1 className="mb-6 text-2xl font-semibold tracking-tight">Edit {ep.name}</h1>
        <ProxyForm initial={ep} />
      </div>
    </div>
  );
}
