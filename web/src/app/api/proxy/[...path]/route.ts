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
  const token = req.cookies.get(TOKEN_COOKIE)?.value;
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const init: RequestInit = { method: req.method, headers };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = await req.text();
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
