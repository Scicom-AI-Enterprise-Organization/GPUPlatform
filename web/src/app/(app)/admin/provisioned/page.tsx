import { redirect } from "next/navigation";
import { cookies } from "next/headers";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";
import { getMe } from "@/lib/me";
import { ConsoleTopbar } from "@/components/console/topbar";
import type { AppRecord, ComputePod } from "@/lib/types";
import { ProvisionedList } from "./provisioned-list";

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

async function load(token: string): Promise<{
  computes: ComputePod[];
  apps: AppRecord[];
}> {
  const headers = { Authorization: `Bearer ${token}` };
  const [c, a] = await Promise.allSettled([
    fetch(`${GATEWAY}/compute`, { headers, cache: "no-store" }).then((r) =>
      r.ok ? (r.json() as Promise<ComputePod[]>) : [],
    ),
    fetch(`${GATEWAY}/apps`, { headers, cache: "no-store" }).then((r) =>
      r.ok ? (r.json() as Promise<AppRecord[]>) : [],
    ),
  ]);
  return {
    computes: c.status === "fulfilled" ? c.value : [],
    apps: a.status === "fulfilled" ? a.value : [],
  };
}

export default async function ProvisionedPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");

  const jar = await cookies();
  const token = jar.get(TOKEN_COOKIE)?.value ?? "";
  const { computes, apps } = await load(token);

  return (
    <div className="flex min-h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Provisioned" }]} username={me.username} />
      <div className="mx-auto w-full max-w-6xl px-6 py-10">
        <header className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Provisioned</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Everything currently running across the platform — compute pods and
            inference endpoints, who provisioned them, and when. Terminate any of
            them from here.
          </p>
        </header>

        <ProvisionedList initialComputes={computes} initialApps={apps} />
      </div>
    </div>
  );
}
