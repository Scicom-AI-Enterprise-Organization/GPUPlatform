import { NextRequest, NextResponse } from "next/server";
import {
  fetchInferenceSummary,
  fetchKinds,
  getGatewayToken,
  GPUPLATFORM_OVERVIEW_KINDS,
} from "../_gpuplatform";

/**
 * Feature-scoped analytics payload for the summary cards and charts.
 *
 * It excludes raw inference history because those records are high-volume and
 * the overview uses the gateway's exact summary endpoint instead.
 *
 * GET /api/analytics/gpuplatform-overview?since=ISO&until=ISO&tz=Area/City
 * → { kinds: { benchmark, training, compute, endpoint }, truncated,
 *     inference_summary }
 */

export async function GET(req: NextRequest) {
  const token = await getGatewayToken();
  if (!token) return NextResponse.json({ error: "not authenticated" }, { status: 401 });

  const since = req.nextUrl.searchParams.get("since") ?? "";
  const until = req.nextUrl.searchParams.get("until") ?? "";
  if (!since || !until) {
    return NextResponse.json({ error: "since and until are required" }, { status: 400 });
  }

  const tz = req.nextUrl.searchParams.get("tz") ?? "UTC";
  const [{ kinds, truncated }, inferenceSummary] = await Promise.all([
    fetchKinds(token, GPUPLATFORM_OVERVIEW_KINDS, since, until),
    fetchInferenceSummary(token, since, until, tz),
  ]);
  return NextResponse.json({ kinds, truncated, inference_summary: inferenceSummary });
}
