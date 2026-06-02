import { cookies } from "next/headers";
import { ConsoleSidebar } from "@/components/console/sidebar";
import { SidebarStateProvider } from "@/components/console/sidebar-state";
import { TerminalThemeInit } from "@/components/terminal-theme-init";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";
import { getMe } from "@/lib/me";
import { disabledSections } from "@/lib/sections";

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

// Tally of things admins might want to see at a glance in the sidebar.
// Fetched only when isAdmin so non-admin sessions don't pay the round-trip.
async function loadAdminCounts(token: string): Promise<{
  pendingApprovals: number;
  provisioned: number;
}> {
  const headers = { Authorization: `Bearer ${token}` };
  const [approvals, computes, apps] = await Promise.allSettled([
    fetch(`${GATEWAY}/compute/approvals`, { headers, cache: "no-store" }).then(
      (r) => (r.ok ? (r.json() as Promise<unknown[]>) : []),
    ),
    fetch(`${GATEWAY}/compute`, { headers, cache: "no-store" }).then((r) =>
      r.ok ? (r.json() as Promise<{ status: string }[]>) : [],
    ),
    fetch(`${GATEWAY}/apps`, { headers, cache: "no-store" }).then((r) =>
      r.ok ? (r.json() as Promise<unknown[]>) : [],
    ),
  ]);
  const pendingApprovals =
    approvals.status === "fulfilled" ? approvals.value.length : 0;
  const liveComputes =
    computes.status === "fulfilled"
      ? computes.value.filter(
          (c) => c.status === "running" || c.status === "creating",
        ).length
      : 0;
  const liveApps = apps.status === "fulfilled" ? apps.value.length : 0;
  return { pendingApprovals, provisioned: liveComputes + liveApps };
}

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const me = await getMe();
  const isAdmin = me?.role === "admin";
  // Surfaces turned off platform-wide via DISABLED_SECTIONS — hidden from the
  // nav entirely (vs per-user denial, where the item shows + the page alerts).
  const disabled = [...disabledSections()];

  let counts = { pendingApprovals: 0, provisioned: 0 };
  if (isAdmin) {
    const jar = await cookies();
    const token = jar.get(TOKEN_COOKIE)?.value ?? "";
    if (token) counts = await loadAdminCounts(token);
  }

  return (
    <SidebarStateProvider>
      <TerminalThemeInit />
      <div className="flex h-screen flex-col overflow-hidden bg-background text-foreground">
        <div className="flex min-h-0 flex-1">
          <ConsoleSidebar
            isAdmin={isAdmin}
            disabled={disabled}
            counts={counts}
          />
          <main className="min-w-0 flex-1 overflow-y-auto scrollbar-thin">
            {children}
          </main>
        </div>
      </div>
    </SidebarStateProvider>
  );
}
