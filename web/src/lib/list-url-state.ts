"use client";

// Mirror a list page's filter/sort/view/select state into the URL query string so
// the exact view is shareable (and survives a refresh). Used by the benchmark,
// serverless, datasets, models, and autotrain lists.
//
// Writes happen via `history.replaceState` — NOT router navigation — so updating a
// filter doesn't refetch the server component or push history entries; it's purely a
// cosmetic, shareable URL. Initial state is read from the URL in each component's
// useState initializers (URL wins over localStorage for `view`). Only non-default
// values are written, so a pristine list stays at a clean `/benchmark`.

import { useEffect } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import type { ReadonlyURLSearchParams } from "next/navigation";

/** Read an enum-ish param, falling back when absent/invalid. */
export function readParam<T extends string>(
  sp: ReadonlyURLSearchParams,
  key: string,
  allowed: readonly T[],
  fallback: T,
): T {
  const v = sp.get(key);
  return v != null && (allowed as readonly string[]).includes(v) ? (v as T) : fallback;
}

export type ListFilters = {
  q?: string;          // free-text search ("" = omit)
  status?: string;     // status filter ("all" = omit)
  sort?: string;       // "newest" (default) | "oldest"
  view?: string;       // "grid" (default) | "rows" | …
  select?: boolean;    // multi-select mode on
  // Page-specific filters keyed by URL param → its value + the value to treat as
  // "default" (omitted). e.g. datasets' source filter: { source: { value, def: "all" } }.
  extra?: Record<string, { value: string; def: string }>;
};

// Values that mean "default" and so are dropped from the URL to keep it clean.
const DEFAULTS: Record<string, string> = { status: "all", sort: "newest", view: "grid" };

/**
 * Keep the URL query in sync with the given list state. Pass only the keys a page
 * actually has; other existing params (e.g. `?scope`) are preserved.
 */
export function useListUrlState(f: ListFilters): void {
  const pathname = usePathname();
  const sp = useSearchParams();
  // Re-run only when a managed value changes (not on every render).
  const signature = JSON.stringify([
    "q" in f ? f.q ?? "" : null,
    "status" in f ? f.status ?? "" : null,
    "sort" in f ? f.sort ?? "" : null,
    "view" in f ? f.view ?? "" : null,
    "select" in f ? !!f.select : null,
    f.extra ? Object.entries(f.extra).map(([k, v]) => [k, v.value]) : null,
  ]);
  useEffect(() => {
    const params = new URLSearchParams(Array.from(sp.entries()));
    const put = (k: string, v: string | undefined) => {
      const def = DEFAULTS[k];
      if (!v || (def !== undefined && v === def)) params.delete(k);
      else params.set(k, v);
    };
    if ("q" in f) put("q", f.q?.trim());
    if ("status" in f) put("status", f.status);
    if ("sort" in f) put("sort", f.sort);
    if ("view" in f) put("view", f.view);
    if ("select" in f) {
      if (f.select) params.set("select", "1");
      else params.delete("select");
    }
    for (const [k, { value, def }] of Object.entries(f.extra ?? {})) {
      if (!value || value === def) params.delete(k);
      else params.set(k, value);
    }
    const qs = params.toString();
    window.history.replaceState(null, "", qs ? `${pathname}?${qs}` : pathname);
    // sp is stable between navigations (we only replaceState); re-running on its
    // identity isn't needed and would be a no-op. Keyed on `signature` + pathname.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature, pathname]);
}
