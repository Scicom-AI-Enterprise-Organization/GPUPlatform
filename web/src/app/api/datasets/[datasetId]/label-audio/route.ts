// Streams a labeling-platform task's audio from the gateway. The generic
// /api/proxy buffers via res.text() (corrupts binary), so audio gets its own
// route that pipes the body through. Cookie → Bearer, like the other routes.

import { NextRequest, NextResponse } from "next/server";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";

const BASE = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

export async function GET(
  req: NextRequest,
  ctx: { params: Promise<{ datasetId: string }> },
) {
  const { datasetId } = await ctx.params;
  const taskId = req.nextUrl.searchParams.get("task_id") ?? "";
  const token = req.cookies.get(TOKEN_COOKIE)?.value;

  const target =
    `${BASE}/v1/datasets/${encodeURIComponent(datasetId)}/label-audio` +
    `?task_id=${encodeURIComponent(taskId)}`;
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  // Forward Range so <audio> can request byte ranges (seek); the gateway slices
  // the buffered clip and answers 206 + Content-Range.
  const range = req.headers.get("range");
  if (range) headers["Range"] = range;

  try {
    const res = await fetch(target, { headers, cache: "no-store" });
    const outHeaders: Record<string, string> = {
      "Content-Type": res.headers.get("content-type") ?? "audio/wav",
      "Cache-Control": "no-store",
      "Accept-Ranges": res.headers.get("accept-ranges") ?? "bytes",
    };
    const cl = res.headers.get("content-length");
    if (cl) outHeaders["Content-Length"] = cl;
    const cr = res.headers.get("content-range");
    if (cr) outHeaders["Content-Range"] = cr;
    return new NextResponse(res.body, { status: res.status, headers: outHeaders });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 502 });
  }
}
