import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";

/**
 * GPU-source alias map for the Analytics page, persisted in the gateway's
 * admin global-env store under ANALYTICS_SOURCE_ALIASES (non-secret JSON:
 * [{ prefix, label }, ...]). Auth is the caller's own session token, so the
 * gateway's admin check applies to both read and write.
 */

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";
const KEY = "ANALYTICS_SOURCE_ALIASES";

export type SourceAlias = { prefix: string; label: string };

async function token(): Promise<string> {
  const jar = await cookies();
  return jar.get(TOKEN_COOKIE)?.value ?? "";
}

export async function GET() {
  const t = await token();
  if (!t) return NextResponse.json({ error: "not authenticated" }, { status: 401 });
  const r = await fetch(`${GATEWAY}/v1/global-env`, {
    headers: { Authorization: `Bearer ${t}` },
    cache: "no-store",
  });
  if (!r.ok) return NextResponse.json({ aliases: null }, { status: r.status === 403 ? 403 : 200 });
  const rows = (await r.json()) as { key: string; value: string | null }[];
  const row = rows.find((x) => x.key === KEY);
  if (!row?.value) return NextResponse.json({ aliases: null });
  try {
    const parsed = JSON.parse(row.value) as SourceAlias[];
    const aliases = parsed.filter(
      (a) => typeof a?.prefix === "string" && a.prefix && typeof a?.label === "string" && a.label,
    );
    return NextResponse.json({ aliases });
  } catch {
    return NextResponse.json({ aliases: null });
  }
}

export async function PUT(req: NextRequest) {
  const t = await token();
  if (!t) return NextResponse.json({ error: "not authenticated" }, { status: 401 });
  const body = (await req.json()) as { aliases?: SourceAlias[] };
  const aliases = (body.aliases ?? []).filter(
    (a) => typeof a?.prefix === "string" && a.prefix.trim() && typeof a?.label === "string" && a.label.trim(),
  ).map((a) => ({ prefix: a.prefix.trim(), label: a.label.trim() }));
  const value = JSON.stringify(aliases);
  if (value.length > 7900) {
    return NextResponse.json({ error: "alias map too large" }, { status: 400 });
  }
  const r = await fetch(`${GATEWAY}/v1/global-env/${KEY}`, {
    method: "PUT",
    headers: { Authorization: `Bearer ${t}`, "Content-Type": "application/json" },
    body: JSON.stringify({ value, is_secret: false }),
  });
  if (!r.ok) {
    return NextResponse.json({ error: `gateway returned ${r.status}` }, { status: r.status });
  }
  return NextResponse.json({ aliases });
}
