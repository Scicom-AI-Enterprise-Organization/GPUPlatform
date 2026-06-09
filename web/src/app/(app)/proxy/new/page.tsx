import { redirect } from "next/navigation";
import { ConsoleTopbar } from "@/components/console/topbar";
import { currentUsername } from "@/lib/current-user";
import { getMe } from "@/lib/me";
import { ProxyForm } from "../proxy-form";

export default async function NewProxyPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");
  const username = await currentUsername();
  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "LLM API Proxy", href: "/proxy" }, { label: "New endpoint" }]} username={username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">New proxy endpoint</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Add OpenAI-compatible upstreams and map a stable alias (e.g. <span className="font-mono">qwen</span>) to each
            backend&apos;s real model name.
          </p>
        </div>
        <ProxyForm />
      </div>
    </div>
  );
}
