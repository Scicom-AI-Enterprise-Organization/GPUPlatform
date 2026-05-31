"use client";

import { useMemo, useState, type ReactNode } from "react";
import { Check, Copy } from "lucide-react";
import { cn } from "@/lib/utils";

// Single pass over pretty-printed JSON. Groups:
//   1 = string literal (no trailing colon)   2 = the "<ws>:" that marks it a key
//   3 = true|false   4 = null   5 = number
// Everything between matches (braces, commas, indentation) is emitted verbatim.
const TOKEN =
  /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false)\b|\b(null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;

function highlight(json: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;
  TOKEN.lastIndex = 0;
  while ((m = TOKEN.exec(json)) !== null) {
    if (m.index > last) out.push(json.slice(last, m.index));
    if (m[1] !== undefined) {
      if (m[2] !== undefined) {
        // object key
        out.push(
          <span key={key++} className="text-sky-700 dark:text-sky-300">
            {m[1]}
          </span>,
        );
        out.push(m[2]); // the whitespace + colon, uncolored
      } else {
        // string value
        out.push(
          <span key={key++} className="text-emerald-700 dark:text-emerald-400">
            {m[1]}
          </span>,
        );
      }
    } else if (m[3] !== undefined) {
      out.push(
        <span key={key++} className="text-amber-600 dark:text-amber-400">
          {m[3]}
        </span>,
      );
    } else if (m[4] !== undefined) {
      out.push(
        <span key={key++} className="text-rose-500 dark:text-rose-400">
          {m[4]}
        </span>,
      );
    } else if (m[5] !== undefined) {
      out.push(
        <span key={key++} className="text-violet-600 dark:text-violet-400">
          {m[5]}
        </span>,
      );
    }
    last = m.index + m[0].length;
  }
  if (last < json.length) out.push(json.slice(last));
  return out;
}

/** Pretty, syntax-highlighted, copyable JSON block. */
export function JsonView({
  value,
  className,
}: {
  value: unknown;
  className?: string;
}) {
  const json = useMemo(() => {
    try {
      return JSON.stringify(value ?? null, null, 2);
    } catch {
      return String(value);
    }
  }, [value]);
  const nodes = useMemo(() => highlight(json), [json]);
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(json);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked — no-op */
    }
  };

  return (
    <div
      className={cn(
        "group relative overflow-hidden rounded-lg border border-border bg-muted/30",
        className,
      )}
    >
      <button
        type="button"
        onClick={onCopy}
        className="absolute right-2 top-2 z-10 inline-flex items-center gap-1 rounded-md border border-border bg-background/80 px-2 py-1 text-[11px] text-muted-foreground opacity-0 backdrop-blur transition-opacity hover:text-foreground group-hover:opacity-100 focus-visible:opacity-100"
        title="Copy JSON"
        aria-label="Copy JSON"
      >
        {copied ? (
          <>
            <Check className="h-3 w-3 text-emerald-600 dark:text-emerald-400" /> Copied
          </>
        ) : (
          <>
            <Copy className="h-3 w-3" /> Copy
          </>
        )}
      </button>
      <pre className="overflow-x-auto px-4 py-3 font-mono text-xs leading-relaxed scrollbar-thin">
        <code>{nodes}</code>
      </pre>
    </div>
  );
}
