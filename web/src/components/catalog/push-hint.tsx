"use client";

import { useState } from "react";
import { Check, Copy, Terminal } from "lucide-react";

/** Collapsible "how to push" hint shown on the catalog list pages. Mirrors the
 * snippets on the detail page, but parameterized for a brand-new repo so users
 * can push before a repo even exists (the mirror creates it on first upload). */
export function PushHint({
  gatewayUrl,
  repoType,
}: {
  gatewayUrl: string;
  repoType: "model" | "dataset";
}) {
  const [copied, setCopied] = useState(false);
  const endpoint = `${gatewayUrl.replace(/\/$/, "")}/hf`;
  const isDataset = repoType === "dataset";
  const example = isDataset ? "my-namespace/my-dataset" : "my-namespace/my-model";
  const folder = isDataset ? "./my-dataset" : "./my-model";

  const lines = [
    `# 1 · Point HF tooling at this gateway`,
    `export HF_ENDPOINT=${endpoint}`,
    `export HF_TOKEN=sgpu_…   # your platform API key`,
    ``,
    `# 2 · Push with the hf CLI (creates the repo on first upload)`,
    isDataset
      ? `hf upload ${example} ${folder} --repo-type dataset`
      : `hf upload ${example} ${folder}`,
  ];
  const code = lines.join("\n");

  return (
    <details className="mb-6 rounded-lg border border-border bg-card">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-3 text-sm font-medium">
        <Terminal className="h-4 w-4 text-muted-foreground" />
        Push a {repoType} with the <span className="font-mono text-xs">hf</span> CLI
      </summary>
      <div className="border-t border-border px-4 py-3">
        <div className="relative">
          <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 pr-10 text-xs leading-relaxed scrollbar-thin">
            <code className="font-mono">{code}</code>
          </pre>
          <button
            onClick={async () => {
              await navigator.clipboard.writeText(code);
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            }}
            className="absolute right-2 top-2 rounded-md bg-background/80 p-1.5 text-muted-foreground hover:text-foreground"
            title="Copy"
          >
            {copied ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
          </button>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          Already have <span className="font-mono">huggingface_hub</span>?{" "}
          <span className="font-mono">model.push_to_hub(&quot;{example}&quot;)</span> works too. Open a{" "}
          {repoType} below for its exact pull/push commands.
        </p>
      </div>
    </details>
  );
}
