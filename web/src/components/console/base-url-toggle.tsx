"use client";

import { cn } from "@/lib/utils";

export type UrlTarget = "public" | "internal";

/**
 * Public ⇄ Internal switch for the "Run a job" snippet cards. "Internal" swaps
 * the snippet base URL to the gateway's in-cluster Service DNS so callers
 * running inside the same Kubernetes cluster skip the public ingress hop.
 *
 * Render this only when an internal URL is actually configured
 * (`gateway.internalBaseUrl` non-empty) — otherwise there's nothing to switch to.
 */
export function BaseUrlToggle({
  value,
  onChange,
  className,
}: {
  value: UrlTarget;
  onChange: (v: UrlTarget) => void;
  className?: string;
}) {
  return (
    <div className={cn("inline-flex items-center gap-1.5", className)}>
      <span className="text-[11px] text-muted-foreground">URL</span>
      <div className="inline-flex rounded-md border border-border p-0.5">
        {(["public", "internal"] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => onChange(t)}
            aria-pressed={value === t}
            className={cn(
              "rounded px-2 py-0.5 text-xs capitalize transition-colors",
              value === t
                ? "bg-muted text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}
