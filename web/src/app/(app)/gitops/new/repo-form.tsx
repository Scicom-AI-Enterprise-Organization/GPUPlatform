"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { gateway } from "@/lib/gateway";
import { FormFooter, FormShell } from "@/components/form-shell";

export function RepoForm() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [branch, setBranch] = useState("main");
  const [path, setPath] = useState("");
  const [tokenSecret, setTokenSecret] = useState("");
  const [webhookSecret, setWebhookSecret] = useState("");
  const [prune, setPrune] = useState(true);
  const [enabled, setEnabled] = useState(true);
  const [pollInterval, setPollInterval] = useState("300");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [testState, setTestState] = useState<"idle" | "testing" | "passed" | "failed">("idle");
  const [testMsg, setTestMsg] = useState<string | null>(null);

  // A passing connection test gates "Add repository". If the repo identity
  // (url/branch/token) changes after a pass, invalidate it so the user must
  // re-test — Add disables again.
  useEffect(() => {
    setTestState("idle");
    setTestMsg(null);
  }, [url, branch, tokenSecret]);

  const onTest = async () => {
    if (!url.trim()) {
      setError("Repository URL is required.");
      return;
    }
    setError(null);
    setTestState("testing");
    setTestMsg(null);
    try {
      const res = await gateway.testGitopsRepo({
        url: url.trim(),
        branch: branch.trim() || "main",
        token_secret: tokenSecret.trim() || null,
      });
      setTestState(res.ok ? "passed" : "failed");
      setTestMsg(res.message);
    } catch (e) {
      setTestState("failed");
      setTestMsg(e instanceof Error ? e.message : String(e));
    }
  };

  const validate = (): string | null => {
    if (!name.trim()) return "Name is required.";
    if (!url.trim()) return "Repository URL is required.";
    const p = Number(pollInterval);
    if (!Number.isFinite(p) || p < 30) return "Poll interval must be ≥ 30 seconds.";
    return null;
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const err = validate();
    if (err) {
      setError(err);
      return;
    }
    setSubmitting(true);
    try {
      const repo = await gateway.createGitopsRepo({
        name: name.trim(),
        url: url.trim(),
        branch: branch.trim() || "main",
        path: path.trim() || null,
        token_secret: tokenSecret.trim() || null,
        webhook_secret: webhookSecret.trim() || null,
        prune,
        poll_interval: Number(pollInterval),
        enabled,
      });
      router.push(`/gitops/${repo.id}`);
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSubmitting(false);
    }
  };

  return (
    <FormShell>
    <form onSubmit={onSubmit} className="flex flex-col gap-5">
      <section data-form-section="Repository" className="scroll-mt-6 rounded-lg border border-border bg-card p-5">
        <div className="mb-4">
          <h2 className="text-base font-medium">Repository</h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            HTTPS URL. For a private repo, store a git token as a{" "}
            <a href="/admin/secrets" className="underline-offset-2 hover:underline">Secret</a>{" "}
            and reference its key below.
          </p>
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground" htmlFor="name">Name</Label>
            <Input id="name" value={name} onChange={(e) => setName(e.target.value)} placeholder="prod-infra" />
          </div>
          <div>
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground" htmlFor="branch">Branch</Label>
            <Input id="branch" value={branch} onChange={(e) => setBranch(e.target.value)} placeholder="main" />
          </div>
          <div className="md:col-span-2">
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground" htmlFor="url">Repository URL</Label>
            <Input id="url" value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://github.com/org/platform-gitops.git" />
          </div>
          <div>
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground" htmlFor="path">Path <span className="text-muted-foreground">(optional)</span></Label>
            <Input id="path" value={path} onChange={(e) => setPath(e.target.value)} placeholder="manifests/" />
          </div>
          <div>
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground" htmlFor="poll">Poll interval (seconds)</Label>
            <Input id="poll" type="number" min={30} value={pollInterval} onChange={(e) => setPollInterval(e.target.value)} />
          </div>
        </div>
      </section>

      <section data-form-section="Auth & webhook" className="scroll-mt-6 rounded-lg border border-border bg-card p-5">
        <div className="mb-4">
          <h2 className="text-base font-medium">Auth & webhook <span className="text-xs font-normal text-muted-foreground">(optional)</span></h2>
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground" htmlFor="token">Token secret key</Label>
            <Input id="token" value={tokenSecret} onChange={(e) => setTokenSecret(e.target.value)} placeholder="GITHUB_DEPLOY_TOKEN" />
            <p className="mt-1 text-[11px] text-muted-foreground">A Secrets key holding a git access token (private repos).</p>
          </div>
          <div>
            <Label className="mb-1.5 text-xs uppercase tracking-wide text-muted-foreground" htmlFor="wh">Webhook HMAC secret</Label>
            <Input id="wh" type="password" value={webhookSecret} onChange={(e) => setWebhookSecret(e.target.value)} placeholder="(for push webhook)" />
            <p className="mt-1 text-[11px] text-muted-foreground">Point the push webhook at <span className="font-mono">/v1/gitops/webhook</span>.</p>
          </div>
        </div>
      </section>

      <section data-form-section="Options" className="scroll-mt-6 rounded-lg border border-border bg-card p-5">
        <div className="flex items-center justify-between gap-4 py-1">
          <div>
            <Label className="text-sm">Prune</Label>
            <p className="text-[11px] text-muted-foreground">Delete resources removed from the repo (full GitOps).</p>
          </div>
          <Switch checked={prune} onCheckedChange={setPrune} />
        </div>
        <div className="mt-3 flex items-center justify-between gap-4 border-t border-border pt-3">
          <div>
            <Label className="text-sm">Enabled</Label>
            <p className="text-[11px] text-muted-foreground">Auto-poll this repo on the interval above.</p>
          </div>
          <Switch checked={enabled} onCheckedChange={setEnabled} />
        </div>
      </section>

      <FormFooter
        error={error}
        hint={
          testState === "idle" ? "Run Test connection — Add is enabled once it passes." : (
            <span
              className={
                testState === "passed" ? "text-emerald-600"
                : testState === "failed" ? "text-destructive"
                : undefined
              }
            >
              {testState === "testing" ? "Testing connection…" : testMsg}
            </span>
          )
        }
      >
        <Button
          type="button"
          variant="outline"
          onClick={onTest}
          disabled={testState === "testing" || !url.trim()}
        >
          {testState === "testing" && <Loader2 className="h-4 w-4 animate-spin" />}
          Test connection
        </Button>
        <Button type="button" variant="ghost" onClick={() => router.push("/gitops")}>Cancel</Button>
        <Button type="submit" disabled={submitting || testState !== "passed"}>
          {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
          Add repository
        </Button>
      </FormFooter>
    </form>
    </FormShell>
  );
}
