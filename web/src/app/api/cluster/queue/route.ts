// Reads per-app request history from the gateway's /apps/{id}/requests
// endpoint, which is backed by Postgres — payloads survive worker pickup
// and Redis TTL. Replaces the old `kubectl exec redis-cli` dev hack.

import { NextRequest, NextResponse } from "next/server";
import { gateway, type GatewayRequestRecord } from "@/lib/gateway";

const APP_ID_RX = /^[a-z0-9][a-z0-9._-]{0,127}$/i;
const LIMIT = 200;

type Bucket = "in queue" | "in progress" | "completed" | "failed";

type Item = {
  request_id: string;
  payload?: unknown;
  endpoint?: string;
  stream?: boolean;
  bucket: Bucket;
  status: string;
  output?: unknown;
  has_result: boolean;
  created_at?: string;
  completed_at?: string | null;
};

function bucketFor(status: string): Bucket {
  const s = status.toLowerCase();
  if (s === "completed" || s === "ready") return "completed";
  // Any terminal failure → "failed". `failed` was missing here, so worker errors
  // (_error_result writes status="failed") wrongly showed as "in progress" forever.
  if (s === "timeout" || s === "cancelled" || s === "error" || s === "failed") return "failed";
  if (s === "pending" || s === "queued") return "in queue";
  return "in progress";
}

export async function GET(req: NextRequest) {
  const appId = req.nextUrl.searchParams.get("app") ?? "";
  if (!APP_ID_RX.test(appId)) {
    return NextResponse.json({ error: "invalid or missing app" }, { status: 400 });
  }

  try {
    const rows: GatewayRequestRecord[] = await gateway.listAppRequests(appId, LIMIT);

    const items: Item[] = rows.map((r) => ({
      request_id: r.request_id,
      payload: r.payload,
      endpoint: r.endpoint,
      stream: r.is_stream,
      bucket: bucketFor(r.status),
      status: r.status,
      output: r.output ?? undefined,
      has_result: r.output !== null,
      created_at: r.created_at,
      completed_at: r.completed_at,
    }));

    return NextResponse.json({
      app_id: appId,
      queue_length: items.filter((i) => i.bucket === "in queue").length,
      in_progress: items.filter((i) => i.bucket === "in progress").length,
      completed: items.filter((i) => i.bucket === "completed").length,
      worker_count: 0,
      items,
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    const isAuth = /401|unauthorized|invalid.*session/i.test(msg);
    return NextResponse.json(
      {
        error: msg,
        hint: isAuth ? "session expired — log in again" : undefined,
      },
      { status: isAuth ? 401 : 502 },
    );
  }
}
