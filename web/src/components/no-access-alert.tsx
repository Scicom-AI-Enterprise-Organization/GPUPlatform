"use client";

import { useEffect, useState } from "react";
import { ShieldAlert } from "lucide-react";

/**
 * Shown when the current user lacks the role/section for a page. Permissions are
 * computed server-side from a live `/auth/me`, but a page render can be reused
 * from the client Router Cache, so a role an admin JUST granted (or a just-fixed
 * session) may not show until a full reload — which used to need a logout/relogin.
 *
 * Self-heal without a manual button, and WITHOUT a redirect loop:
 *   - One-shot `relogin` marker: if we land here already carrying it, we've
 *     revalidated once and are still denied → genuine no-access (a re-render
 *     won't change it), so stop and show the explanation.
 *   - Signed in? A single HARD reload re-runs the server with the same session
 *     and picks up freshly-granted access. (Bouncing an authenticated user to the
 *     login form can't grant a missing section — it just ping-pongs/flickers.)
 *   - Signed out? Send to /login (re-auth is the only thing that helps), with
 *     `next` bringing them back here.
 */
export function NoAccessAlert({
  title = "You don't have permission to view this",
  message,
}: {
  title?: string;
  message?: string;
}) {
  const [denied, setDenied] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("relogin") === "1") {
      // Already revalidated once and still denied → genuinely no access.
      setDenied(true);
      return;
    }
    const here = `${window.location.pathname}?relogin=1`;
    // sgpu_user is the non-httpOnly companion cookie — its presence means there's
    // a session, so a reload (not a login bounce) is the right way to refresh.
    const signedIn = document.cookie
      .split("; ")
      .some((c) => c.startsWith("sgpu_user="));
    // replace() (not assign) so Back doesn't bounce straight back into this.
    window.location.replace(
      signedIn ? here : `/login?next=${encodeURIComponent(here)}`,
    );
  }, []);

  // While revalidating (reload or login redirect), render nothing — no flash.
  if (!denied) return null;

  return (
    <div
      role="alert"
      className="mb-6 flex items-start gap-3 rounded-md border border-l-4 border-destructive/40 border-l-destructive bg-destructive/5 px-4 py-3 text-sm"
    >
      <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
      <div className="flex-1 space-y-1">
        <div className="font-semibold text-destructive">{title}</div>
        <p className="text-foreground/80">
          {message ??
            "Your account doesn't have the required role for this page. Ask an admin to grant you developer access."}
        </p>
      </div>
    </div>
  );
}
