"use client";

import { useMemo, useState } from "react";
import { DollarSign, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Pagination } from "@/components/ui/pagination";
import { formatCostUSD } from "@/lib/cost";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { AuditLogRecord } from "@/lib/types";

const ANY = "__any__";

function fmtTs(iso: string): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function relativeTime(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  const diff = Date.now() - d.getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const days = Math.floor(h / 24);
  return `${days}d ago`;
}

export function AuditList({ initial }: { initial: AuditLogRecord[] }) {
  const [query, setQuery] = useState("");
  const [resourceType, setResourceType] = useState<string>(ANY);
  const [actor, setActor] = useState<string>(ANY);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(24);

  const resourceTypes = useMemo(() => {
    const s = new Set<string>();
    initial.forEach((e) => s.add(e.resource_type));
    return Array.from(s).sort();
  }, [initial]);

  const actors = useMemo(() => {
    const s = new Set<string>();
    initial.forEach((e) => s.add(e.actor_username));
    return Array.from(s).sort();
  }, [initial]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return initial.filter((e) => {
      if (resourceType !== ANY && e.resource_type !== resourceType) return false;
      if (actor !== ANY && e.actor_username !== actor) return false;
      if (!q) return true;
      const hay = [
        e.actor_username,
        e.action,
        e.resource_type,
        e.resource_id ?? "",
        e.resource_name ?? "",
        JSON.stringify(e.details ?? {}),
      ]
        .join(" ")
        .toLowerCase();
      return hay.includes(q);
    });
  }, [initial, query, resourceType, actor]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  // Clamp in render so a shrinking result set never strands an empty page;
  // changing any filter resets to page 1 via the handlers below.
  const currentPage = Math.min(page, pageCount);
  const paged = filtered.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[240px]">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setPage(1);
            }}
            placeholder="Search action, resource, actor…"
            className="pl-9"
          />
        </div>
        <Select value={actor} onValueChange={(v) => { setActor(v); setPage(1); }}>
          <SelectTrigger className="w-44">
            <SelectValue placeholder="Any actor" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ANY}>Any actor</SelectItem>
            {actors.map((a) => (
              <SelectItem key={a} value={a}>{a}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={resourceType} onValueChange={(v) => { setResourceType(v); setPage(1); }}>
          <SelectTrigger className="w-44">
            <SelectValue placeholder="Any resource" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ANY}>Any resource</SelectItem>
            {resourceTypes.map((t) => (
              <SelectItem key={t} value={t}>{t}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <span className="text-xs text-muted-foreground">
          {filtered.length} of {initial.length}
        </span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-border bg-card">
        <table className="w-full text-sm">
          <thead className="bg-muted/40 text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="px-4 py-2 text-left font-medium">When</th>
              <th className="px-4 py-2 text-left font-medium">Actor</th>
              <th className="px-4 py-2 text-left font-medium">Action</th>
              <th className="px-4 py-2 text-left font-medium">Resource</th>
              <th className="px-4 py-2 text-left font-medium">Details</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {paged.map((e) => (
              <tr key={e.id} className="hover:bg-muted/20">
                <td className="px-4 py-2 align-top">
                  <div className="text-xs">{fmtTs(e.created_at)}</div>
                  <div className="text-[11px] text-muted-foreground">
                    {relativeTime(e.created_at)}
                  </div>
                </td>
                <td className="px-4 py-2 align-top">
                  <div className="font-medium">{e.actor_username}</div>
                  {e.actor_email && (
                    <div className="text-[11px] text-muted-foreground">{e.actor_email}</div>
                  )}
                </td>
                <td className="px-4 py-2 align-top">
                  <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">
                    {e.action}
                  </code>
                </td>
                <td className="px-4 py-2 align-top">
                  <div className="text-xs">
                    <span className="text-muted-foreground">{e.resource_type}</span>
                    {e.resource_name && <span> · {e.resource_name}</span>}
                  </div>
                  {e.resource_id && (
                    <div className="font-mono text-[11px] text-muted-foreground">
                      {e.resource_id}
                    </div>
                  )}
                </td>
                <td className="px-4 py-2 align-top">
                  {e.details ? (
                    <div className="space-y-1">
                      <CostPill details={e.details} />
                      <details className="max-w-md">
                        <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
                          view
                        </summary>
                        <pre className="mt-1 max-h-48 overflow-auto rounded-md border border-border bg-muted/40 p-2 font-mono text-[11px]">
                          {JSON.stringify(e.details, null, 2)}
                        </pre>
                      </details>
                    </div>
                  ) : (
                    <span className="text-xs text-muted-foreground">—</span>
                  )}
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={5} className="px-4 py-12 text-center text-sm text-muted-foreground">
                  No matching events.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {filtered.length > 0 && (
        <Pagination
          page={currentPage}
          pageCount={pageCount}
          total={filtered.length}
          pageSize={pageSize}
          onPageChange={setPage}
          onPageSizeChange={(n) => {
            setPageSize(n);
            setPage(1);
          }}
          itemLabel="events"
        />
      )}
    </div>
  );
}

/** Surfaces a "spent $X.XX" pill when the audit row's details include a
 * cost breakdown (final_cost_usd from compute.delete / benchmark.delete).
 * Renders nothing for non-cost events so the column stays tidy. */
function CostPill({ details }: { details: Record<string, unknown> }) {
  const cost = details.final_cost_usd;
  if (typeof cost !== "number" || !Number.isFinite(cost)) return null;
  const dur = typeof details.duration_s === "number" ? details.duration_s : null;
  return (
    <div className="inline-flex items-center gap-1 rounded-md border border-amber-500/30 bg-amber-500/10 px-1.5 py-0.5 text-[11px] tabular-nums">
      <DollarSign className="h-3 w-3 text-amber-700 dark:text-amber-400" />
      <span className="font-semibold text-foreground">{formatCostUSD(cost)}</span>
      {dur != null && (
        <span className="text-muted-foreground">· {fmtDurationShort(dur)}</span>
      )}
    </div>
  );
}

function fmtDurationShort(secs: number): string {
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}
