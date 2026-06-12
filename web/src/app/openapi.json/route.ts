import { NextResponse } from "next/server";

const BASE = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

type Json = Record<string, unknown>;

// Routes that need no token — kept off the global security requirement.
const PUBLIC_PATHS = new Set([
  "/",
  "/health",
  "/ready",
  "/version",
  "/openapi.json",
  "/v1/models",
  "/auth/login",
  "/auth/register",
  "/auth/github/upsert",
]);

// Public OpenAPI schema, served on the web origin. The gateway *generates* it
// (source of truth) but isn't exposed in prod, so the web — the public edge —
// serves it. We don't blindly proxy it: we augment the spec to reflect how the
// API is actually used from outside —
//   • every call goes through the web at `/api/proxy/…` (the gateway is internal),
//   • authenticated with `Authorization: Bearer <sgpu_ API key>`.
// No session cookie is attached here; reading the schema needs no auth.
export async function GET() {
  let schema: Json;
  try {
    const res = await fetch(`${BASE}/openapi.json`, { cache: "no-store" });
    if (!res.ok) {
      return NextResponse.json(
        { error: `gateway returned ${res.status} for /openapi.json` },
        { status: 502 },
      );
    }
    schema = (await res.json()) as Json;
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : String(e) },
      { status: 502 },
    );
  }

  // Bearer (API key) security scheme + a global requirement — so the schema
  // documents the header regardless of what the gateway's own spec carries.
  const components = (schema.components ??= {}) as Json;
  const schemes = (components.securitySchemes ??= {}) as Json;
  schemes.bearerAuth = {
    type: "http",
    scheme: "bearer",
    description:
      "API key (prefix `sgpu_`) from the API keys page. Send it as `Authorization: Bearer <key>`.",
  };
  schema.security = [{ bearerAuth: [] }];

  // The genuinely-public routes shouldn't read as gated.
  const paths = (schema.paths ?? {}) as Record<string, Json>;
  for (const [path, ops] of Object.entries(paths)) {
    if (!PUBLIC_PATHS.has(path)) continue;
    for (const op of Object.values(ops)) {
      if (op && typeof op === "object") (op as Json).security = [];
    }
  }

  // In prod the gateway is internal; clients reach it through the web proxy.
  // A relative server URL resolves against whatever origin served this schema
  // (localhost:3000 in dev, the public web origin in prod).
  schema.servers = [
    { url: "/api/proxy", description: "Public API via the web edge — Authorization: Bearer sgpu_…" },
  ];

  return NextResponse.json(schema, { headers: { "Cache-Control": "public, max-age=300" } });
}
