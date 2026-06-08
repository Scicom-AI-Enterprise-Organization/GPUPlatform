// Prometheus scrape target on the web host: proxies the gateway's resource
// exporter (GET {gateway}/api/metrics — serverless apps, benchmarks, storage,
// datasets, GPU providers, GitOps and the autotrain run series/counter).
//
// Unauthenticated by design (scrapers send no cookie); gate via network/ingress
// like any Prometheus endpoint. The gateway endpoint is likewise auth-exempt.

import { NextResponse } from "next/server";

const BASE = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

// A scrape must never be cached.
export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  try {
    const r = await fetch(`${BASE}/metrics/resources`, { cache: "no-store" });
    const body = await r.text();
    return new NextResponse(body, {
      status: r.status,
      headers: {
        "content-type":
          r.headers.get("content-type") ?? "text/plain; version=0.0.4; charset=utf-8",
        "cache-control": "no-store",
      },
    });
  } catch (e) {
    // Surface as a 502 so Prometheus marks the target down rather than parsing junk.
    return new NextResponse(
      `# gateway unreachable: ${e instanceof Error ? e.message : String(e)}\n`,
      { status: 502, headers: { "content-type": "text/plain; charset=utf-8" } },
    );
  }
}
