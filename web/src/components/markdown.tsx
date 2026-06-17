"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Compact GitHub-flavoured markdown renderer (tables, code, lists) styled to
 * match the console theme — no typography plugin needed. Used for benchmark
 * comparison summaries / notes. */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="text-sm leading-relaxed text-foreground">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: (p) => <h1 className="mb-2 mt-4 text-xl font-semibold tracking-tight" {...p} />,
          h2: (p) => <h2 className="mb-2 mt-4 text-lg font-semibold tracking-tight" {...p} />,
          h3: (p) => <h3 className="mb-1.5 mt-3 text-base font-semibold" {...p} />,
          p: (p) => <p className="my-2" {...p} />,
          ul: (p) => <ul className="my-2 ml-5 list-disc space-y-1" {...p} />,
          ol: (p) => <ol className="my-2 ml-5 list-decimal space-y-1" {...p} />,
          a: (p) => (
            <a className="text-primary underline underline-offset-2" target="_blank" rel="noopener noreferrer" {...p} />
          ),
          code: (p) => <code className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]" {...p} />,
          pre: (p) => <pre className="my-2 overflow-x-auto rounded-md bg-muted/60 p-3 font-mono text-xs" {...p} />,
          table: (p) => (
            <div className="my-3 overflow-x-auto">
              <table className="w-full border-collapse text-xs" {...p} />
            </div>
          ),
          th: (p) => <th className="border border-border bg-muted/40 px-2 py-1 text-left font-medium" {...p} />,
          td: (p) => <td className="border border-border px-2 py-1 tabular-nums" {...p} />,
          blockquote: (p) => (
            <blockquote className="my-2 border-l-2 border-border pl-3 text-muted-foreground" {...p} />
          ),
          hr: () => <hr className="my-3 border-border" />,
          strong: (p) => <strong className="font-semibold text-foreground" {...p} />,
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
