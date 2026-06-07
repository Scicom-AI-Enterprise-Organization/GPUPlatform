// Server-side proxy to the gateway. The browser hits /api/proxy/...; we
// pull the user's session token out of the httpOnly cookie and forward it
// as `Authorization: Bearer <token>`. Body, query, and method are passed
// through verbatim.

import { NextRequest, NextResponse } from "next/server";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";

const BASE = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

export async function GET(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, ctx);
}
export async function POST(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, ctx);
}
export async function PUT(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, ctx);
}
export async function DELETE(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, ctx);
}
export async function PATCH(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  return forward(req, ctx);
}

async function forward(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const url = new URL(req.url);
  const target = `${BASE}/${path.join("/")}${url.search}`;

  const headers: Record<string, string> = {
    "Content-Type": req.headers.get("content-type") ?? "application/json",
  };
  // Prefer an explicit Authorization header — this is how external API-key
  // clients (an `sgpu_` token) reach the gateway in prod, where the gateway
  // itself isn't exposed. Fall back to the httpOnly session cookie for in-app
  // (browser) calls, which don't send an Authorization header.
  const authHeader = req.headers.get("authorization");
  const token = req.cookies.get(TOKEN_COOKIE)?.value;
  if (authHeader) headers["Authorization"] = authHeader;
  else if (token) headers["Authorization"] = `Bearer ${token}`;
  // Forward Range so media elements can request byte ranges (audio seeking).
  const range = req.headers.get("range");
  if (range) headers["Range"] = range;

  const init: RequestInit = { method: req.method, headers };
  if (req.method !== "GET" && req.method !== "HEAD") {
    // Text bodies (JSON/form/etc.) forward as text; binary bodies (e.g. an
    // uploaded audio clip) MUST forward as raw bytes — `req.text()` UTF-8-decodes
    // and corrupts them.
    const reqCt = (req.headers.get("content-type") ?? "").toLowerCase();
    const reqIsText =
      reqCt === "" ||
      reqCt.startsWith("text/") ||
      reqCt.includes("json") ||
      reqCt.includes("xml") ||
      reqCt.includes("x-www-form-urlencoded");
    init.body = reqIsText ? await req.text() : await req.arrayBuffer();
  }

  try {
    const res = await fetch(target, init);
    const ct = res.headers.get("content-type") ?? "application/json";
    // SSE / chunked: pipe the body through instead of buffering — buffering
    // would break long-running streams (e.g. benchmark log tails).
    if (ct.includes("text/event-stream") || ct.includes("application/x-ndjson")) {
      return new NextResponse(res.body, {
        status: res.status,
        headers: {
          "Content-Type": ct,
          "Cache-Control": "no-cache",
          "X-Accel-Buffering": "no",
        },
      });
    }
    // Binary (audio/image/video/octet-stream/etc.): pipe the body through as-is.
    // `res.text()` would UTF-8-decode and corrupt binary payloads — which is why
    // proxied audio failed to play. Forward media-relevant headers too.
    const isText =
      ct.startsWith("text/") ||
      ct.includes("application/json") ||
      ct.includes("application/javascript") ||
      ct.includes("+json") ||
      ct.includes("application/xml");
    if (!isText) {
      const out = new Headers({ "Content-Type": ct });
      for (const h of ["content-length", "accept-ranges", "content-range", "cache-control", "content-disposition"]) {
        const v = res.headers.get(h);
        if (v) out.set(h, v);
      }
      return new NextResponse(res.body, { status: res.status, headers: out });
    }
    const body = await res.text();
    return new NextResponse(body, {
      status: res.status,
      headers: { "Content-Type": ct },
    });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : String(e) },
      { status: 502 },
    );
  }
}
