// Shared server-route session guard.
//
// `src/proxy.ts` excludes `/api/*` from middleware, so every Next API route
// authenticates itself. Routes that forward the session token to the gateway
// get validation for free (the gateway rejects a bad token). Routes that act
// with a server-side secret (RUNPOD_API_KEY, the SlurmUI ADMIN token, …) never
// forward the caller's token, so they MUST verify the session here first —
// otherwise the presence of any (forgeable) cookie would be enough.
//
// This validates the httpOnly session cookie against the gateway's cheap
// authenticated `/auth/me` endpoint (the same call `me.ts` / `/api/auth/me`
// use) and returns the user, or a sentinel the route turns into a 401.

import type { NextRequest } from "next/server";
import { TOKEN_COOKIE } from "./auth-cookie";

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

export type SessionUser = {
  user_id: number;
  username: string;
  is_admin: boolean;
  role: "user" | "developer" | "admin";
};

export type SessionResult =
  | { ok: true; user: SessionUser }
  | { ok: false };

/** Validate the caller's session cookie against the gateway. On success returns
 * `{ ok: true, user }`; on a missing/invalid token (or an unreachable gateway)
 * returns `{ ok: false }` — the route should respond 401. */
export async function requireSession(req: NextRequest): Promise<SessionResult> {
  const token = req.cookies.get(TOKEN_COOKIE)?.value;
  if (!token) return { ok: false };
  try {
    const r = await fetch(`${GATEWAY}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!r.ok) return { ok: false };
    const user = (await r.json()) as SessionUser;
    return { ok: true, user };
  } catch {
    return { ok: false };
  }
}
