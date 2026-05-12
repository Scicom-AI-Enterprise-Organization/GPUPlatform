import { redirect } from "next/navigation";
import Link from "next/link";
import { cookies } from "next/headers";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";
import { getMe } from "@/lib/me";
import { ConsoleTopbar } from "@/components/console/topbar";
import type { AdminUserRecord } from "@/lib/types";
import { OrganizationTable } from "./table";

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

async function loadJson<T>(token: string, path: string): Promise<T | null> {
  try {
    const r = await fetch(`${GATEWAY}${path}`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null;
  }
}

export default async function OrganizationPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");

  const jar = await cookies();
  const token = jar.get(TOKEN_COOKIE)?.value ?? "";
  const users = await loadJson<AdminUserRecord[]>(token, "/admin/users");

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Organization" }]} username={me.username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="">
        <header className="mb-8">
          <h1 className="text-2xl font-semibold tracking-tight">Organization</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Members + role assignments. The <span className="font-medium">Tier</span>{" "}
            decides admin vs developer vs no-access; the{" "}
            <span className="font-medium">Policy role</span> decides which
            sections a developer can see (manage these in{" "}
            <Link href="/admin/roles" className="underline underline-offset-2">
              Roles
            </Link>
            ).
          </p>
        </header>

        <section className="rounded-xl border border-border bg-card">
          <div className="flex items-center justify-between border-b border-border px-5 py-4">
            <div>
              <h2 className="text-base font-semibold">Members ({users?.length ?? 0})</h2>
              <p className="text-xs text-muted-foreground">All users with access to this gateway.</p>
            </div>
          </div>
          <OrganizationTable users={users ?? []} />
        </section>
        </div>
      </div>
    </div>
  );
}
