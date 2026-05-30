// Server-side stress runner. The browser POSTs the test config here; this runs
// the load generator on the Next.js (Node) server — driving the gateway with
// the user's bearer token — and streams NDJSON progress back. Running here (vs.
// the browser) escapes the browser's ~6-connection-per-host limit, so the
// requested concurrency is actually achieved. Reuses the same bench core as the
// in-browser path, and the same gateway /stream/{app_id} endpoint.

import { NextRequest } from "next/server";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";
import { runStressBench, formatLine, type StressConfig } from "@/lib/stress-bench";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BASE = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

function clamp(v: unknown, lo: number, hi: number, dflt: number): number {
  const n = Math.floor(Number(v));
  if (!Number.isFinite(n)) return dflt;
  return Math.min(hi, Math.max(lo, n));
}

export async function POST(req: NextRequest) {
  const token = req.cookies.get(TOKEN_COOKIE)?.value;
  const body = (await req.json().catch(() => null)) as Record<string, unknown> | null;
  const appId = typeof body?.app_id === "string" ? body.app_id : "";
  if (!appId) {
    return new Response(JSON.stringify({ error: "app_id required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const cfg: StressConfig = {
    model: typeof body?.model === "string" ? body.model : "",
    inputLen: clamp(body?.input_len, 1, 32768, 128),
    outputLen: clamp(body?.output_len, 1, 8192, 128),
    numPrompts: clamp(body?.num_prompts, 1, 5000, 50),
    concurrency: clamp(body?.concurrency, 1, 1024, 10),
  };

  const streamUrl = `${BASE}/stream/${encodeURIComponent(appId)}`;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  // Abort the in-flight worker requests when the client disconnects / hits Stop.
  const ctrl = new AbortController();
  req.signal.addEventListener("abort", () => ctrl.abort());

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (obj: unknown) => {
        try {
          controller.enqueue(encoder.encode(JSON.stringify(obj) + "\n"));
        } catch {
          /* stream already closed */
        }
      };
      try {
        const { summary, firstError } = await runStressBench(streamUrl, headers, cfg, {
          signal: ctrl.signal,
          onResult: (r, done) => send({ type: "progress", done, line: formatLine(done, r) }),
        });
        send({ type: "summary", summary, firstError });
      } catch (e) {
        send({ type: "error", error: e instanceof Error ? e.message : String(e) });
      } finally {
        try {
          controller.close();
        } catch {
          /* already closed */
        }
      }
    },
    cancel() {
      ctrl.abort();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
    },
  });
}
