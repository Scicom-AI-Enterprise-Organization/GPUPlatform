import { redirect } from "next/navigation";
import { cookies } from "next/headers";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";
import { getMe } from "@/lib/me";
import { ConsoleTopbar } from "@/components/console/topbar";
import type { AuditLogRecord } from "@/lib/types";
import { AuditList } from "./audit-list";

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

async function loadAudit(token: string): Promise<AuditLogRecord[]> {
  try {
    const r = await fetch(`${GATEWAY}/admin/audit-logs?limit=1000`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!r.ok) return [];
    return (await r.json()) as AuditLogRecord[];
  } catch {
    return [];
  }
}

export default async function AuditPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/serverless");

  const jar = await cookies();
  const token = jar.get(TOKEN_COOKIE)?.value ?? "";
  const events = await loadAudit(token);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar crumbs={[{ label: "Audit log" }]} username={me.username} />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="">
        <header className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Audit log</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Every state-changing action across the platform — who, what, when.
            Most recent first; up to 1000 events shown.
          </p>
        </header>

        <AuditList initial={events} />
        </div>
      </div>
    </div>
  );
}
