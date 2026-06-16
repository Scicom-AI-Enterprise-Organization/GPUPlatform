// Next.js 16 proxy/middleware.
//
// Auth model: optional. If the user has a session cookie we pass it through;
// if they don't, we still let them browse so the UI works against gateways
// that haven't enabled multitenant auth yet. Pages and APIs handle 401s
// gracefully (a "Sign in" call-to-action shows in the topbar).
//
// We do still bounce already-signed-in users away from /login and /register.

import { NextRequest, NextResponse } from "next/server";
import { TOKEN_COOKIE } from "@/lib/auth-cookie";

const AUTH_PATHS = ["/login", "/register"];

export default function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  if (
    pathname.startsWith("/_next") ||
    pathname.startsWith("/api/") ||
    pathname === "/favicon.ico" ||
    pathname.startsWith("/images")
  ) {
    return NextResponse.next();
  }

  const hasSession = !!req.cookies.get(TOKEN_COOKIE)?.value;
  const isAuthPage = AUTH_PATHS.some(
    (p) => pathname === p || pathname.startsWith(`${p}/`),
  );

  // Already signed in but visiting /login or /register → bounce to home.
  if (hasSession && isAuthPage) {
    const url = req.nextUrl.clone();
    url.pathname = "/serverless";
    url.searchParams.delete("next");
    return NextResponse.redirect(url);
  }

  return NextResponse.next();
}

export const config = {
  // Exclude `api` so API requests never traverse the middleware/Edge layer —
  // that layer caps the request body (~8 MB), which truncated large benchmark
  // imports through /api/proxy. Middleware only guards page routes (bouncing
  // signed-in users off /login,/register), so skipping /api is safe.
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};
