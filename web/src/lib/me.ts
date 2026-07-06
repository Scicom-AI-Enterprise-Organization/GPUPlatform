// Server-side helper: ask the gateway who the caller is, including role.
// Used by the (app) layout to decide whether to render admin-only nav.

import { cookies } from "next/headers";
import { TOKEN_COOKIE } from "./auth-cookie";
import { disabledSections } from "./sections";

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";

export type Section = "inference" | "benchmark" | "compute" | "datasets" | "catalog" | "quantization";

export type Me = {
  user_id: number;
  username: string;
  email?: string | null;
  is_admin: boolean;
  role: "user" | "developer" | "admin";
  policy_role_id: string | null;
  sections: Record<Section, boolean>;
};

export async function getMe(): Promise<Me | null> {
  const jar = await cookies();
  const token = jar.get(TOKEN_COOKIE)?.value;
  if (!token) return null;
  try {
    const r = await fetch(`${GATEWAY}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!r.ok) return null;
    const me = (await r.json()) as Me;
    // Honor the web's own DISABLED_SECTIONS — zero out disabled surfaces so the
    // sidebar nav AND every page that gates on `me.sections.*` treat them as off,
    // even if the gateway hasn't disabled them.
    const disabled = disabledSections();
    if (disabled.size && me.sections) {
      for (const s of disabled) {
        if (s in me.sections) (me.sections as Record<string, boolean>)[s] = false;
      }
    }
    return me;
  } catch {
    return null;
  }
}
