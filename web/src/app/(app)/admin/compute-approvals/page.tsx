import { redirect } from "next/navigation";
import { cookies } from "next/headers";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";
import { getMe } from "@/lib/me";
import { ConsoleTopbar } from "@/components/console/topbar";
import type { ComputePod } from "@/lib/types";
import { ApprovalsList } from "./approvals-list";

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

async function loadApprovals(token: string): Promise<ComputePod[]> {
  try {
    const r = await fetch(`${GATEWAY}/compute/approvals`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!r.ok) return [];
    return (await r.json()) as ComputePod[];
  } catch {
    return [];
  }
}

export default async function ComputeApprovalsPage() {
  const me = await getMe();
  if (!me) redirect("/login");
  if (me.role !== "admin") redirect("/compute");

  const jar = await cookies();
  const token = jar.get(TOKEN_COOKIE)?.value ?? "";
  const initial = await loadApprovals(token);

  return (
    <div className="flex h-full flex-col">
      <ConsoleTopbar
        crumbs={[{ label: "Compute approvals" }]}
        username={me.username}
      />
      <div className="flex-1 overflow-y-auto px-6 py-6 lg:px-10 lg:py-8 scrollbar-thin">
        <div className="">
          <header className="mb-6">
            <h1 className="text-2xl font-semibold tracking-tight">Compute approvals</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Pending pod requests from non-admin users. Approving kicks off
              provisioning immediately and starts billing on RunPod.
            </p>
          </header>

          <ApprovalsList initial={initial} />
        </div>
      </div>
    </div>
  );
}
