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
  // Forward SGPU control headers (e.g. X-SGPU-Upstream, which pins a proxy request
  // to one upstream) + an inbound trace id. Everything else is intentionally
  // dropped so the browser can't smuggle arbitrary headers to the gateway.
  req.headers.forEach((v, k) => {
    if (k.startsWith("x-sgpu-") || k === "x-request-id") headers[k] = v;
  });

  const init: RequestInit & { duplex?: "half" } = { method: req.method, headers };
  if (req.method !== "GET" && req.method !== "HEAD") {
    // Stream the request body straight through instead of buffering it with
    // req.text()/arrayBuffer(). Buffering truncated large uploads (a ~17MB
    // benchmark import was cut to ~8MB → "Unterminated string" at the gateway)
    // and UTF-8-decoding corrupted binary bodies. Forwarding the raw stream is
    // byte-exact for both text and binary; undici requires duplex:"half" for a
    // streamed request body.
    init.body = req.body;
    init.duplex = "half";
  }

  try {
    const res = await fetch(target, init);
    const ct = res.headers.get("content-type") ?? "application/json";
    // Pass through the proxy router's routing-info headers so the browser
    // (e.g. the proxy Playground) can show which upstream served the request.
    const upstreamHeaders: Record<string, string> = {};
    for (const k of ["x-upstream-url", "x-upstream-name", "x-request-id"]) {
      const v = res.headers.get(k);
      if (v) upstreamHeaders[k] = v;
    }
    // SSE / chunked: pipe the body through instead of buffering — buffering
    // would break long-running streams (e.g. benchmark log tails).
    if (ct.includes("text/event-stream") || ct.includes("application/x-ndjson")) {
      return new NextResponse(res.body, {
        status: res.status,
        headers: {
          "Content-Type": ct,
          "Cache-Control": "no-cache",
          "X-Accel-Buffering": "no",
          ...upstreamHeaders,
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
      headers: { "Content-Type": ct, ...upstreamHeaders },
    });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : String(e) },
      { status: 502 },
    );
  }
}
