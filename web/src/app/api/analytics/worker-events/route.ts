import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";

/**
 * Server-side proxy to the gateway's admin /admin/worker-events feed for the
 * Analytics → GPU Timeline tab. Forwards the caller's own session token so the
 * gateway's admin check still applies (non-admins get a 403). Returns the raw
 * durable worker lifecycle events across every inference endpoint; the client
 * pairs them into on/off spans.
 *
 * GET /api/analytics/worker-events?since=ISO&until=ISO
 * → { count: number, events: WorkerEvent[] }
 */

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

export async function GET(req: NextRequest) {
  const jar = await cookies();
  const token = jar.get(TOKEN_COOKIE)?.value ?? "";
  if (!token) return NextResponse.json({ error: "not authenticated" }, { status: 401 });

  const since = req.nextUrl.searchParams.get("since") ?? "";
  const until = req.nextUrl.searchParams.get("until") ?? "";
  const qs = new URLSearchParams();
  if (since) qs.set("since", since);
  if (until) qs.set("until", until);
  qs.set("limit", "20000");

  const r = await fetch(`${GATEWAY}/admin/worker-events?${qs}`, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
  });
  if (!r.ok) {
    return NextResponse.json({ count: 0, events: [] }, { status: r.status });
  }
  return NextResponse.json(await r.json());
}
