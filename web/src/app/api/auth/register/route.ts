import { NextRequest, NextResponse } from "next/server";
import { setAuthCookies } from "@/lib/auth-cookie";

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

export async function POST(req: NextRequest) {
  if (process.env.REGISTRATION_DISABLED === "1") {
    return NextResponse.json({ error: "registration is disabled" }, { status: 403 });
  }
  const body = await req.json().catch(() => ({}));
  const { username, password, email } = body as {
    username?: string;
    password?: string;
    email?: string;
  };
  if (!username || !password || !email) {
    return NextResponse.json({ error: "username, email and password required" }, { status: 400 });
  }

  const r = await fetch(`${GATEWAY}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, email }),
    cache: "no-store",
  });
  const text = await r.text();
  if (!r.ok) {
    return new NextResponse(text || JSON.stringify({ error: r.statusText }), {
      status: r.status,
      headers: { "Content-Type": "application/json" },
    });
  }

  const data = JSON.parse(text) as { token: string; username: string };
  const res = NextResponse.json({ username: data.username });
  setAuthCookies(res.cookies, data.token, data.username);
  return res;
}
