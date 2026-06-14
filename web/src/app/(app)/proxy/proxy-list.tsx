"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Inbox, LayoutGrid, List, Network, Search, Trash2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { SortSelect, sortByCreated, type SortDir } from "@/components/ui/sort-select";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { gateway } from "@/lib/gateway";
import type { ProxyEndpoint } from "@/lib/types";

/** Flat searchable string per endpoint — name, path, owner, and every model
 * alias across its upstreams — so one query hits any of them. */
function searchableText(ep: ProxyEndpoint): string {
  const aliases = ep.upstreams.flatMap((u) => Object.keys(u.models));
  return [ep.name, `/proxy/${ep.name}/v1`, ep.created_by, ...aliases].join(" ").toLowerCase();
}

export function ProxyList({ items }: { items: ProxyEndpoint[] }) {
  const router = useRouter();
  const [busy, setBusy] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<SortDir>("newest");
  const [view, setView] = useState<"rows" | "grid">("grid");

  useEffect(() => {
    const v = window.localStorage.getItem("sgpu_proxy_view");
    // Reading client-only localStorage post-mount avoids an SSR/CSR mismatch.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (v === "rows" || v === "grid") setView(v);
  }, []);
  const setViewPersist = (v: "rows" | "grid") => {
    setView(v);
    window.localStorage.setItem("sgpu_proxy_view", v);
  };

  const haystacks = useMemo(() => items.map((ep) => ({ ep, text: searchableText(ep) })), [items]);
  const filtered = useMemo(() => {
    const tokens = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const matched = haystacks
      .filter(({ text }) => tokens.every((t) => text.includes(t)))
      .map(({ ep }) => ep);
    return sortByCreated(matched, sort);
  }, [haystacks, q, sort]);

  const onDelete = async (ep: ProxyEndpoint) => {
    if (!confirm(`Delete proxy endpoint "${ep.name}"? Clients pointing at /proxy/${ep.name}/v1 will stop working.`)) return;
    setBusy(ep.id);
    try {
      await gateway.deleteProxy(ep.id);
      router.refresh();
    } catch (e) {
      alert(e instanceof Error ? e.message : String(e));
      setBusy(null);
    }
  };

  return (
    <div>
      <div className="mb-4 flex gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="search"
            placeholder="Search by name, path, model, owner…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="h-10 w-full rounded-md border border-input bg-background pl-9 pr-9 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
          />
          {q && (
            <button
              type="button"
              onClick={() => setQ("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              title="Clear"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        <SortSelect value={sort} onValueChange={setSort} />
        <div className="inline-flex h-10 items-stretch overflow-hidden rounded-md border border-input bg-background shadow-xs">
          <button
            type="button"
            onClick={() => setViewPersist("rows")}
            className={cn(
              "inline-flex items-center justify-center px-2.5 text-sm",
              view === "rows" ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/50",
            )}
            title="List view"
            aria-label="List view"
            aria-pressed={view === "rows"}
          >
            <List className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => setViewPersist("grid")}
            className={cn(
              "inline-flex items-center justify-center border-l border-input px-2.5 text-sm",
              view === "grid" ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/50",
            )}
            title="Grid view"
            aria-label="Grid view"
            aria-pressed={view === "grid"}
          >
            <LayoutGrid className="h-4 w-4" />
          </button>
        </div>
      </div>

      {q && (
        <div className="mb-3 text-xs text-muted-foreground">
          {filtered.length} of {items.length} match for{" "}
          <span className="font-mono text-foreground">&quot;{q}&quot;</span>
        </div>
      )}

      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
          <Inbox className="h-6 w-6 text-muted-foreground/60" />
          <p className="text-sm text-muted-foreground">No proxy endpoints match your search.</p>
        </div>
      ) : (
        <ul className={cn("gap-3", view === "rows" ? "flex flex-col" : "grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3")}>
          {filtered.map((ep) => {
            const aliases = new Set<string>();
            ep.upstreams.forEach((u) => Object.keys(u.models).forEach((a) => aliases.add(a)));
            return (
              <li
                key={ep.id}
                onClick={() => router.push(`/proxy/${ep.id}`)}
                className="flex cursor-pointer flex-col rounded-xl border border-border bg-card p-4 transition-all hover:border-primary/40 hover:shadow-md"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-3">
                    <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg border border-border bg-muted/60 text-muted-foreground">
                      <Network className="h-5 w-5" />
                    </div>
                    <div className="min-w-0">
                      <Link href={`/proxy/${ep.id}`} onClick={(e) => e.stopPropagation()} className="truncate font-medium text-foreground hover:underline">{ep.name}</Link>
                      <div className="mt-0.5 truncate font-mono text-xs text-muted-foreground">/proxy/{ep.name}/v1</div>
                    </div>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon-sm" onClick={(e) => e.stopPropagation()} className="-mr-1 shrink-0 text-muted-foreground hover:text-foreground" disabled={busy === ep.id}>⋯</Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem asChild><Link href={`/proxy/${ep.id}`}>Open</Link></DropdownMenuItem>
                      <DropdownMenuItem asChild><Link href={`/proxy/${ep.id}/edit`}>Edit</Link></DropdownMenuItem>
                      <DropdownMenuItem variant="destructive" onSelect={(e) => { e.preventDefault(); onDelete(ep); }}>
                        <Trash2 className="h-4 w-4" /> Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-1.5 text-xs">
                  {!ep.enabled && <span className="rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] uppercase text-muted-foreground">disabled</span>}
                  <span className="rounded-md bg-muted/50 px-2 py-0.5">{ep.upstreams.length} upstream{ep.upstreams.length === 1 ? "" : "s"}</span>
                  <span className="rounded-md bg-muted/50 px-2 py-0.5">{aliases.size} model{aliases.size === 1 ? "" : "s"}</span>
                  <span className="rounded-md bg-muted/50 px-2 py-0.5">conc {ep.max_concurrency || "∞"}</span>
                </div>
                <div className="mt-2 mb-3 flex flex-wrap gap-1">
                  {[...aliases].slice(0, 6).map((a) => (
                    <span key={a} className="rounded bg-primary/10 px-1.5 py-0.5 font-mono text-[11px] text-primary">{a}</span>
                  ))}
                </div>
                <div className="mt-auto flex items-center justify-between border-t border-border/60 pt-2 text-xs text-muted-foreground">
                  <span>{ep.queued > 0 ? `${ep.queued} queued · ` : ""}{ep.inflight} in-flight</span>
                  <span title={new Date(ep.created_at).toISOString()}>{ep.created_by}</span>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
