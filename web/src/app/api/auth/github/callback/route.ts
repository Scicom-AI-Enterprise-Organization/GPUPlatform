// GET /api/auth/github/callback — completes GitHub OAuth.
//
// 1. Verify state matches the one we stashed in start/route.ts.
// 2. Exchange the auth code for a GitHub access token.
// 3. Fetch the user profile + primary email from GitHub.
// 4. Hand off (github_id, login, email, name) to the gateway's
//    /auth/github/upsert endpoint to mint a session token.
// 5. Set our normal sgpu_token / sgpu_user cookies and redirect to `next`.

import { NextRequest, NextResponse } from "next/server";
import { setAuthCookies } from "@/lib/auth-cookie";

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";
const GH_CLIENT_ID = process.env.GITHUB_CLIENT_ID ?? "";
const GH_CLIENT_SECRET = process.env.GITHUB_CLIENT_SECRET ?? "";
const GH_REDIRECT_URI = process.env.GITHUB_OAUTH_REDIRECT_URI ?? "";
const INTERNAL_TOKEN = process.env.INTERNAL_AUTH_TOKEN ?? "";
const STATE_COOKIE = "sgpu_gh_oauth";

type GhUser = { id: number; login: string; name: string | null; email: string | null };
type GhEmail = { email: string; primary: boolean; verified: boolean };

export async function GET(req: NextRequest) {
  if (!GH_CLIENT_ID || !GH_CLIENT_SECRET || !GH_REDIRECT_URI || !INTERNAL_TOKEN) {
    return failRedirect(
      req,
      "GitHub SSO not configured (missing GITHUB_* / INTERNAL_AUTH_TOKEN)",
    );
  }

  const code = req.nextUrl.searchParams.get("code");
  const state = req.nextUrl.searchParams.get("state");
  const cookie = req.cookies.get(STATE_COOKIE)?.value;
  if (!code || !state || !cookie) {
    return failRedirect(req, "Missing OAuth params");
  }

  // Compare against the stashed state. Don't bother with constant-time
  // here — state is fresh per attempt and unguessable.
  let stored: { state: string; next: string };
  try {
    stored = JSON.parse(cookie) as { state: string; next: string };
  } catch {
    return failRedirect(req, "Bad state cookie");
  }
  if (stored.state !== state) {
    return failRedirect(req, "OAuth state mismatch");
  }
  const next = stored.next || "/serverless";

  // Exchange code for access token.
  const tokenRes = await fetch("https://github.com/login/oauth/access_token", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      client_id: GH_CLIENT_ID,
      client_secret: GH_CLIENT_SECRET,
      code,
      redirect_uri: GH_REDIRECT_URI,
    }),
  });
  if (!tokenRes.ok) {
    return failRedirect(req, `GitHub token exchange failed (${tokenRes.status})`);
  }
  const tokenJson = (await tokenRes.json()) as {
    access_token?: string;
    error?: string;
  };
  const ghToken = tokenJson.access_token;
  if (!ghToken) {
    return failRedirect(req, tokenJson.error ?? "No GitHub access token");
  }

  // Fetch user profile + primary email in parallel.
  const ghHeaders = {
    Authorization: `Bearer ${ghToken}`,
    Accept: "application/vnd.github+json",
    "User-Agent": "serverlessgpu-web",
  };
  const [userRes, emailsRes] = await Promise.all([
    fetch("https://api.github.com/user", { headers: ghHeaders, cache: "no-store" }),
    fetch("https://api.github.com/user/emails", {
      headers: ghHeaders,
      cache: "no-store",
    }),
  ]);
  if (!userRes.ok) {
    return failRedirect(req, `GitHub /user fetch failed (${userRes.status})`);
  }
  const ghUser = (await userRes.json()) as GhUser;
  const emails = emailsRes.ok ? ((await emailsRes.json()) as GhEmail[]) : [];
  const primary = emails.find((e) => e.primary && e.verified)?.email;
  const email = primary ?? ghUser.email ?? null;

  // Mint a platform session via the gateway.
  const upsertRes = await fetch(`${GATEWAY}/auth/github/upsert`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Internal-Token": INTERNAL_TOKEN,
    },
    body: JSON.stringify({
      github_id: String(ghUser.id),
      login: ghUser.login,
      email,
      name: ghUser.name,
    }),
    cache: "no-store",
  });
  if (!upsertRes.ok) {
    const body = await upsertRes.text().catch(() => "");
    return failRedirect(req, `Gateway upsert failed: ${upsertRes.status} ${body}`);
  }
  const sess = (await upsertRes.json()) as { token: string; username: string };

  const res = NextResponse.redirect(absoluteUrl(req, next));
  setAuthCookies(res.cookies, sess.token, sess.username);
  res.cookies.delete(STATE_COOKIE);
  return res;
}

function failRedirect(req: NextRequest, message: string) {
  const url = absoluteUrl(req, "/login");
  url.searchParams.set("sso_error", message);
  const res = NextResponse.redirect(url);
  res.cookies.delete(STATE_COOKIE);
  return res;
}

// Build a public-facing absolute URL for redirects. `req.url` ends up using
// the container's internal bind (0.0.0.0:3000) once we're behind an ingress
// that doesn't propagate Host into Next's URL parsing — so prefer:
//
//   1. The origin of the configured GITHUB_OAUTH_REDIRECT_URI (already the
//      public URL, since GitHub round-tripped through it).
//   2. The X-Forwarded-Host + X-Forwarded-Proto pair if set.
//   3. Fall back to req.url (good enough for local dev).
function absoluteUrl(req: NextRequest, path: string): URL {
  const fromCallback = (() => {
    if (!GH_REDIRECT_URI) return null;
    try {
      return new URL(GH_REDIRECT_URI).origin;
    } catch {
      return null;
    }
  })();
  if (fromCallback) return new URL(path, fromCallback);

  const xfHost = req.headers.get("x-forwarded-host");
  const xfProto = req.headers.get("x-forwarded-proto") ?? "https";
  if (xfHost) return new URL(path, `${xfProto}://${xfHost}`);

  return new URL(path, req.url);
}
