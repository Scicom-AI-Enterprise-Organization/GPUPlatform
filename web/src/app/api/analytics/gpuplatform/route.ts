import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";

/**
 * Server-side fan-out to the gateway's admin /v1/history/{kind} endpoints for
 * the Analytics page. Runs on the server so the per-kind pagination happens
 * close to the gateway; auth is the caller's own session token, so the
 * gateway's admin check still applies (non-admins get 403s and an empty
 * result, same as hitting the API directly).
 *
 * GET /api/analytics/gpuplatform?since=ISO&until=ISO
 * → { kinds: { benchmark: JobRecord[], training: [...], compute: [...],
 *              inference: [...], proxy: [...] }, truncated: string[] }
 */

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

const KINDS: { kind: string; path: string }[] = [
  { kind: "benchmark", path: "benchmarks" },
  { kind: "training", path: "training" },
  { kind: "compute", path: "compute" },
  { kind: "inference", path: "inference" },
  { kind: "proxy", path: "proxy" },
  { kind: "endpoint", path: "endpoints" },
];

const PAGE = 1000; // gateway max
const MAX_PAGES = 5; // bound the high-volume kinds; report truncation instead of hanging

type JobRecord = Record<string, unknown>;

async function fetchKind(
  token: string,
  path: string,
  since: string,
  until: string,
): Promise<{ jobs: JobRecord[]; truncated: boolean }> {
  const jobs: JobRecord[] = [];
  for (let page = 0; page < MAX_PAGES; page++) {
    // Newest-first: when a high-volume kind (inference) hits MAX_PAGES, the
    // dropped records are the oldest in the window, not the latest traffic.
    const qs = new URLSearchParams({
      since,
      until,
      order: "desc",
      limit: String(PAGE),
      offset: String(page * PAGE),
    });
    const r = await fetch(`${GATEWAY}/v1/history/${path}?${qs}`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!r.ok) return { jobs, truncated: false };
    const body = (await r.json()) as { jobs: JobRecord[]; has_more: boolean };
    jobs.push(...body.jobs);
    if (!body.has_more) return { jobs, truncated: false };
  }
  return { jobs, truncated: true };
}

export async function GET(req: NextRequest) {
  const jar = await cookies();
  const token = jar.get(TOKEN_COOKIE)?.value ?? "";
  if (!token) return NextResponse.json({ error: "not authenticated" }, { status: 401 });

  const since = req.nextUrl.searchParams.get("since") ?? "";
  const until = req.nextUrl.searchParams.get("until") ?? "";
  if (!since || !until) {
    return NextResponse.json({ error: "since and until are required" }, { status: 400 });
  }

  // Exact creation counts for the high-volume inference kind — charts use
  // these; the raw (capped) inference page only feeds the Jobs explorer.
  const tz = req.nextUrl.searchParams.get("tz") ?? "UTC";
  const summaryReq = fetch(
    `${GATEWAY}/v1/history/summary?${new URLSearchParams({ since, until, tz })}`,
    { headers: { Authorization: `Bearer ${token}` }, cache: "no-store" },
  )
    .then(async (r) => (r.ok ? ((await r.json()) as { rows: unknown[] }).rows : null))
    .catch(() => null);

  const [results, inferenceSummary] = await Promise.all([
    Promise.all(
      KINDS.map(async ({ kind, path }) => ({ kind, ...(await fetchKind(token, path, since, until)) })),
    ),
    summaryReq,
  ]);
  const kinds: Record<string, JobRecord[]> = {};
  const truncated: string[] = [];
  for (const r of results) {
    kinds[r.kind] = r.jobs;
    if (r.truncated) truncated.push(r.kind);
  }
  return NextResponse.json({ kinds, truncated, inference_summary: inferenceSummary });
}
