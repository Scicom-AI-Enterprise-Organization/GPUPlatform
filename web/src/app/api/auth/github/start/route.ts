// GET /api/auth/github/start — kicks off the GitHub OAuth dance.
//
// Generates a one-shot CSRF state, stashes it in an httpOnly cookie, and
// redirects to GitHub's authorize URL. The matching state is checked in the
// callback below. The `next` query (where to land after sign-in) is also
// folded into the cookie so we don't need to round-trip it via GitHub.

import { NextRequest, NextResponse } from "next/server";
import crypto from "node:crypto";

const GH_CLIENT_ID = process.env.GITHUB_CLIENT_ID ?? "";
const GH_REDIRECT_URI = process.env.GITHUB_OAUTH_REDIRECT_URI ?? "";
const STATE_COOKIE = "sgpu_gh_oauth";
const STATE_TTL_S = 600; // 10 min — plenty for the GitHub roundtrip

export async function GET(req: NextRequest) {
  if (!GH_CLIENT_ID || !GH_REDIRECT_URI) {
    return NextResponse.json(
      { error: "GitHub SSO not configured (GITHUB_CLIENT_ID / GITHUB_OAUTH_REDIRECT_URI)" },
      { status: 503 },
    );
  }

  const next = sanitizeNext(req.nextUrl.searchParams.get("next"));
  const state = crypto.randomBytes(24).toString("hex");

  // We pack state + next into one cookie value so both survive the
  // GitHub roundtrip without leaking either through the URL.
  const cookieValue = JSON.stringify({ state, next });

  const params = new URLSearchParams({
    client_id: GH_CLIENT_ID,
    redirect_uri: GH_REDIRECT_URI,
    scope: "read:user user:email",
    state,
    allow_signup: "true",
  });

  const res = NextResponse.redirect(
    `https://github.com/login/oauth/authorize?${params.toString()}`,
  );
  res.cookies.set(STATE_COOKIE, cookieValue, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: STATE_TTL_S,
  });
  return res;
}

function sanitizeNext(raw: string | null): string {
  // Only allow same-origin paths; otherwise default to /inference. Stops
  // an attacker turning the SSO flow into an open redirect.
  if (!raw) return "/inference";
  if (!raw.startsWith("/") || raw.startsWith("//")) return "/inference";
  return raw;
}
