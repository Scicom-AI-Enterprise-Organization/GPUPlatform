import { NextRequest, NextResponse } from "next/server";
import {
  fetchKinds,
  getGatewayToken,
  GPUPLATFORM_ALL_KINDS,
} from "../_gpuplatform";

/**
 * Feature-scoped analytics payload for record-heavy surfaces:
 * Running now, Jobs explorer, Node timeline, and Node utilization.
 *
 * GET /api/analytics/gpuplatform-records?since=ISO&until=ISO
 * → { kinds: { benchmark, training, compute, inference, endpoint }, truncated }
 */

export async function GET(req: NextRequest) {
  const token = await getGatewayToken();
  if (!token) return NextResponse.json({ error: "not authenticated" }, { status: 401 });

  const since = req.nextUrl.searchParams.get("since") ?? "";
  const until = req.nextUrl.searchParams.get("until") ?? "";
  if (!since || !until) {
    return NextResponse.json({ error: "since and until are required" }, { status: 400 });
  }

  const { kinds, truncated } = await fetchKinds(token, GPUPLATFORM_ALL_KINDS, since, until);
  return NextResponse.json({ kinds, truncated });
}
