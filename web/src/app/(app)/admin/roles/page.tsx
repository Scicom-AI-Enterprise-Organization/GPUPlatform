import { redirect } from "next/navigation";
import { cookies } from "next/headers";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";
import { getMe } from "@/lib/me";
import { ConsoleTopbar } from "@/components/console/topbar";
import type { PolicyRole } from "@/lib/types";
import { RolesManager } from "./roles-manager";

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

async function loadRoles(token: string): Promise<PolicyRole[]> {
  try {
    const r = await fetch(`${GATEWAY}/admin/policy-roles`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!r.ok) return [];
    return (await r.json()) as PolicyRole[];
  } catch {
    return [];
  }
}

export default async function RolesPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");

  const jar = await cookies();
  const token = jar.get(TOKEN_COOKIE)?.value ?? "";
  const roles = await loadRoles(token);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Roles" }]} username={me.username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="">
        <header className="mb-8">
          <h1 className="text-2xl font-semibold tracking-tight">Roles</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Each role bundles a set of section permissions. Attach roles to
            users in <span className="font-medium">Organization</span>. System
            roles can be edited but not deleted.
          </p>
        </header>

        <RolesManager initial={roles} />
        </div>
      </div>
    </div>
  );
}
