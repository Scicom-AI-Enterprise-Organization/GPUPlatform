"use client";

import { useRouter, useSearchParams, usePathname } from "next/navigation";
import { Users, User as UserIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export function ScopeToggle({ scope }: { scope: "mine" | "all" }) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();

  function setScope(next: "mine" | "all") {
    if (next === scope) return;
    // Read the LIVE URL rather than useSearchParams: the list components mirror their
    // search/status/sort/view/select state into the URL via history.replaceState,
    // which the Next router (and useSearchParams) doesn't observe. Using
    // window.location.search preserves those filters so toggling scope keeps the
    // full shareable URL intact (scope + filters together).
    const live = typeof window !== "undefined" ? window.location.search : `?${params.toString()}`;
    const sp = new URLSearchParams(live);
    if (next === "mine") sp.delete("scope");
    else sp.set("scope", "all");
    const qs = sp.toString();
    router.push(qs ? `${pathname}?${qs}` : pathname);
  }

  return (
    <div className="inline-flex shrink-0 items-center rounded-md border border-border bg-background p-0.5 text-xs">
      <button
        type="button"
        onClick={() => setScope("mine")}
        className={cn(
          "inline-flex items-center gap-1.5 whitespace-nowrap rounded px-2.5 py-1 font-medium transition-colors",
          scope === "mine"
            ? "bg-foreground text-background"
            : "text-muted-foreground hover:text-foreground",
        )}
      >
        <UserIcon className="h-3.5 w-3.5" />
        Mine
      </button>
      <button
        type="button"
        onClick={() => setScope("all")}
        className={cn(
          "inline-flex items-center gap-1.5 whitespace-nowrap rounded px-2.5 py-1 font-medium transition-colors",
          scope === "all"
            ? "bg-foreground text-background"
            : "text-muted-foreground hover:text-foreground",
        )}
      >
        <Users className="h-3.5 w-3.5" />
        All users
      </button>
    </div>
  );
}
