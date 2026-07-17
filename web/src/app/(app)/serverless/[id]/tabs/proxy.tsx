"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowUpRight, Copy, Loader2, Network, Plus } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { gateway } from "@/lib/gateway";
import type { AppProxyLink, AppRecord } from "@/lib/types";

// "Proxy" tab — surfaces the LLM API proxies that front this endpoint (matched
// server-side by upstream URL or model). Read-only and secret-stripped: we only
// ever get the proxy name, its stable serving path, and the model aliases — never
// upstream base_urls or keys. Non-owners of a public endpoint see only PUBLIC
// proxies; admins can also deep-link to create a proxy pre-pointed at this endpoint.
export function ProxyTab({
  app,
  readOnly = false,
  isAdmin = false,
}: {
  app: AppRecord;
  readOnly?: boolean;
  isAdmin?: boolean;
}) {
  const [links, setLinks] = useState<AppProxyLink[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    gateway
      .listAppProxies(app.app_id)
      .then((r) => { if (!cancelled) setLinks(r); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)); });
    return () => { cancelled = true; };
  }, [app.app_id]);

  // The OpenAI serving URL of THIS endpoint — what a proxy upstream points at.
  const servingUrl = `${gateway.baseUrl}/${app.app_id}/v1`;
  const seedModel = app.model || app.models?.[0]?.model || "";
  const newProxyHref =
    `/proxy/new?name=${encodeURIComponent(app.name)}` +
    `&base=${encodeURIComponent(servingUrl)}` +
    `&model=${encodeURIComponent(seedModel)}`;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">API Proxy</CardTitle>
          <span className="text-xs text-muted-foreground">
            A proxy gives this endpoint a stable OpenAI-compatible URL + model alias with
            priority/health-aware failover across backends. Clients point at{" "}
            <code className="font-mono">/proxy/&lt;name&gt;/v1</code> and never change anything.
          </span>
        </CardHeader>
        <CardContent className="space-y-3">
          {error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </div>
          )}

          {links === null && !error && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" /> Looking for proxies that route here…
            </div>
          )}

          {links !== null && links.length === 0 && !error && (
            <p className="text-sm text-muted-foreground">
              No {readOnly ? "public " : ""}API proxy currently routes to this endpoint.
              {isAdmin && " Create one below to give it a stable alias + failover."}
            </p>
          )}

          {links?.map((p) => {
            const serveUrl = `${gateway.baseUrl}${p.serving_path}`;
            const alias = p.models[0] ?? seedModel;
            return (
              <div key={p.id} className="rounded-md border border-border p-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <Network className="h-4 w-4 shrink-0 text-muted-foreground" />
                    <Link href={`/proxy/${p.id}`} className="truncate font-medium hover:underline">{p.name}</Link>
                    {p.public && (
                      <span className="rounded border border-emerald-500/40 bg-emerald-500/10 px-1.5 py-0.5 text-[10px] uppercase text-emerald-700 dark:text-emerald-400">
                        public
                      </span>
                    )}
                  </div>
                  <Button asChild variant="outline" size="xs">
                    <Link href={`/proxy/${p.id}`}>Open <ArrowUpRight className="h-3 w-3" /></Link>
                  </Button>
                </div>
                <div className="mt-2 truncate font-mono text-xs text-muted-foreground">{serveUrl}</div>
                {p.models.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {p.models.map((m) => (
                      <span key={m} className="rounded bg-primary/10 px-1.5 py-0.5 font-mono text-[11px] text-primary">{m}</span>
                    ))}
                  </div>
                )}
                <CurlSnippet base={serveUrl} model={alias} />
              </div>
            );
          })}

          {isAdmin && (
            <Button asChild variant="outline" size="sm">
              <Link href={newProxyHref}><Plus className="h-4 w-4" /> Create a proxy for this endpoint</Link>
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function CurlSnippet({ base, model }: { base: string; model: string }) {
  const snippet = `curl -X POST '${base}/chat/completions' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer sgpu_…' \\
  -d '{"model": "${model}", "messages": [{"role":"user","content":"Hello"}]}'`;
  return (
    <div className="relative mt-2">
      <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-2.5 font-mono text-[11px] leading-relaxed text-foreground scrollbar-thin">{snippet}</pre>
      <Button
        variant="outline"
        size="icon-sm"
        className="absolute right-2 top-2"
        aria-label="Copy"
        onClick={() => { navigator.clipboard.writeText(snippet); toast.success("Copied", { duration: 2000 }); }}
      >
        <Copy className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}
