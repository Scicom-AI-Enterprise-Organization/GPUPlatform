import { notFound, redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { gateway, GatewayError } from "@/lib/gateway";
import type { ProxyEndpoint } from "@/lib/types";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { ProxyDetail } from "./proxy-detail";

export default async function ProxyDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const me = await getMe();
  if (!me) redirect("/login");
  const isAdmin = me.role === "admin";
  const { id } = await params;
  let ep: ProxyEndpoint;
  try {
    // Admins see the full config; everyone else sees the secret-stripped public
    // view (404s a private proxy, so non-admins can't probe non-public ones).
    ep = isAdmin ? await gateway.getProxy(id) : await gateway.getPublicProxy(id);
  } catch (e) {
    if (e instanceof GatewayError && e.status === 404) notFound();
    throw e;
  }
  const username = await currentUsername();
  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "API Proxy", href: "/proxy" }, { label: ep.name }]} username={username} />
      <ProxyDetail initial={ep} baseUrl={gateway.baseUrl} readOnly={!isAdmin} />
    </div>
  );
}
