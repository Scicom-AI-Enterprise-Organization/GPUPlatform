"use client";

// "cURL for this request" — the copyable equivalent of what a playground sends.
// Shared by every mode (chat, embeddings, rerank, audio, TTS) so the snippet reads
// the same everywhere instead of only existing on the chat tab.
//
// The bearer is DISPLAYED as the account's first key prefix (so you can see which
// key the browser session is using) but COPIED as the placeholder — the real secret
// is never in the page, and pasting a truncated `sgpu_abc…` would just 401.
import { useEffect, useState } from "react";
import { Copy } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";

export const KEY_PLACEHOLDER = "YOUR_SGPU_API_KEY";

/** Prefix of the account's first API key, or null (no keys / not signed in). */
export function useApiKeyPrefix(): string | null {
  const [prefix, setPrefix] = useState<string | null>(null);
  useEffect(() => {
    let abort = false;
    fetch("/api/api-keys", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : []))
      .then((keys: { prefix: string }[]) => {
        if (!abort) setPrefix(Array.isArray(keys) && keys.length > 0 ? keys[0].prefix : null);
      })
      .catch(() => {});
    return () => { abort = true; };
  }, []);
  return prefix;
}

// `build` is called twice — once with the token to show, once with the placeholder
// that actually lands on the clipboard.
export function CurlBlock({ build, label = "cURL for this request" }: {
  build: (token: string) => string;
  label?: string;
}) {
  const prefix = useApiKeyPrefix();
  const copy = build(KEY_PLACEHOLDER);
  return (
    <div className="space-y-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <div className="relative">
        <pre className="max-h-80 overflow-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-[11px] leading-relaxed text-foreground scrollbar-thin">
          {prefix ? build(`${prefix}...`) : copy}
        </pre>
        <Button variant="outline" size="icon-sm" className="absolute right-2 top-2" aria-label="Copy cURL"
                onClick={() => { navigator.clipboard.writeText(copy); toast.success("cURL copied", { duration: 3000 }); }}>
          <Copy className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  );
}

/** `-H` continuation lines for the extra headers a playground sends (e.g. X-SGPU-Upstream). */
export function headerLines(extra?: Record<string, string>): string {
  return Object.entries(extra ?? {}).map(([k, v]) => `  -H '${k}: ${shq(v)}' \\\n`).join("");
}

// Body for a single-quoted shell argument. These snippets carry whatever you typed
// into the playground, and one apostrophe ("customer's bill") would otherwise end the
// quote and leave you pasting a command that dies in the shell.
export function shq(s: string): string {
  return s.replace(/'/g, `'\\''`);
}

/** A JSON-body POST: `curl -X POST '<url>' -H … -d '<pretty json>'`. */
export function curlJson(
  url: string, token: string, body: unknown, extra?: Record<string, string>,
): string {
  return `curl -X POST '${url}' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer ${token}' \\
${headerLines(extra)}  -d '${shq(JSON.stringify(body, null, 2))}'`;
}
