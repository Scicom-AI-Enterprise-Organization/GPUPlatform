// Binary-safe upload bridge for dataset metadata files.
//
// The generic /api/proxy forwards bodies via `req.text()`, which UTF-8-decodes
// and would corrupt a binary multipart body. Here we read the multipart form
// with `req.formData()` (Next handles it natively), then stream the file's raw
// bytes to the gateway as application/octet-stream with the cookie→Bearer
// header. The gateway parses + stores it (see datasets_api.upload_metadata).

import { NextRequest, NextResponse } from "next/server";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";

const BASE = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

export async function POST(
  req: NextRequest,
  ctx: { params: Promise<{ datasetId: string }> },
) {
  const { datasetId } = await ctx.params;

  let file: File | null = null;
  try {
    const form = await req.formData();
    const f = form.get("file");
    if (f instanceof File) file = f;
  } catch {
    return NextResponse.json({ error: "expected multipart/form-data with a 'file' field" }, { status: 400 });
  }
  if (!file) {
    return NextResponse.json({ error: "no file provided" }, { status: 400 });
  }

  const token = req.cookies.get(TOKEN_COOKIE)?.value;
  const target =
    `${BASE}/v1/datasets/${encodeURIComponent(datasetId)}/upload` +
    `?filename=${encodeURIComponent(file.name)}`;

  const headers: Record<string, string> = { "Content-Type": "application/octet-stream" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  try {
    const buf = await file.arrayBuffer();
    const res = await fetch(target, { method: "POST", headers, body: Buffer.from(buf) });
    const body = await res.text();
    return new NextResponse(body, {
      status: res.status,
      headers: { "Content-Type": res.headers.get("content-type") ?? "application/json" },
    });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 502 });
  }
}
