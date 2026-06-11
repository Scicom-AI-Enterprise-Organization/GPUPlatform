"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";
import { useState } from "react";
import { Network, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { gateway } from "@/lib/gateway";
import type { ProxyEndpoint } from "@/lib/types";

export function ProxyList({ items }: { items: ProxyEndpoint[] }) {
  const router = useRouter();
  const [busy, setBusy] = useState<string | null>(null);

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
    <ul className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
      {items.map((ep) => {
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
  );
}
