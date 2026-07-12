import { NextRequest, NextResponse } from "next/server";
import { requireSession } from "@/lib/require-session";

/**
 * Server-side proxy to SlurmUI's /api/reports for the Analytics page. The
 * SlurmUI credentials are a server secret (an `aura_…` API token of an ADMIN
 * user), so the browser never sees them; we only forward to logged-in callers.
 *
 * GET /api/analytics/slurm?from=YYYY-MM-DD&to=YYYY-MM-DD&tz=Area/City
 * → SlurmUI report JSON, or { configured: false } when env is unset.
 *
 * Env (web server side):
 *   SLURMUI_URL       e.g. https://slurm.aies.scicom.dev
 *   SLURMUI_API_TOKEN aura_… token (ADMIN role → platform-wide report)
 */

export async function GET(req: NextRequest) {
  // SlurmUI is queried with a server-side ADMIN token, so a forgeable
  // cookie-presence check isn't enough — validate the session against the
  // gateway before forwarding.
  const session = await requireSession(req);
  if (!session.ok) {
    return NextResponse.json({ error: "not authenticated" }, { status: 401 });
  }

  const base = (process.env.SLURMUI_URL ?? "").replace(/\/$/, "");
  const token = process.env.SLURMUI_API_TOKEN ?? "";
  if (!base || !token) return NextResponse.json({ configured: false });

  const sp = req.nextUrl.searchParams;
  const qs = new URLSearchParams();
  for (const k of ["from", "to", "tz"]) {
    const v = sp.get(k);
    if (v) qs.set(k, v);
  }

  try {
    const r = await fetch(`${base}/api/reports?${qs}`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!r.ok) {
      return NextResponse.json(
        { configured: true, error: `SlurmUI responded ${r.status}` },
        { status: 502 },
      );
    }
    const body = await r.json();
    // baseUrl lets the client build deep links to Aura job pages
    // (/clusters/{clusterId}/jobs/{id}) — it's not a secret, only the token is.
    return NextResponse.json({ configured: true, baseUrl: base, report: body });
  } catch {
    return NextResponse.json(
      { configured: true, error: "SlurmUI unreachable" },
      { status: 502 },
    );
  }
}
