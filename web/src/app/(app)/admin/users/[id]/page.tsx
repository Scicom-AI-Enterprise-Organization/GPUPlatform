import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { cookies } from "next/headers";
import { ChevronLeft } from "lucide-react";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";
import { getMe } from "@/lib/me";
import { ConsoleTopbar } from "@/components/console/topbar";
import type { AdminUserRecord, PolicyRole } from "@/lib/types";
import { UserProfile } from "./user-profile";

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

async function loadUser(id: string, token: string): Promise<AdminUserRecord | null> {
  const r = await fetch(`${GATEWAY}/admin/users/${encodeURIComponent(id)}`, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
  }).catch(() => null);
  if (!r || !r.ok) return null;
  return (await r.json()) as AdminUserRecord;
}

async function loadPolicyRoles(token: string): Promise<PolicyRole[]> {
  const r = await fetch(`${GATEWAY}/admin/policy-roles`, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
  }).catch(() => null);
  if (!r || !r.ok) return [];
  return (await r.json()) as PolicyRole[];
}

export default async function UserProfilePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");

  const { id } = await params;
  const jar = await cookies();
  const token = jar.get(TOKEN_COOKIE)?.value ?? "";
  const [user, policyRoles] = await Promise.all([
    loadUser(id, token),
    loadPolicyRoles(token),
  ]);
  if (!user) notFound();

  return (
    <div className="flex min-h-full flex-col">
      <ConsoleTopbar
        crumbs={[
          { label: "Organization", href: "/organization" },
          { label: user.username },
        ]}
        username={me.username}
      />
      <div className="w-full px-6 py-10">
        <Link
          href="/organization"
          className="mb-4 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          <ChevronLeft className="h-3.5 w-3.5" />
          Back to organization
        </Link>
        <UserProfile
          user={user}
          policyRoles={policyRoles}
          isSelf={user.id === me.user_id}
        />
      </div>
    </div>
  );
}
