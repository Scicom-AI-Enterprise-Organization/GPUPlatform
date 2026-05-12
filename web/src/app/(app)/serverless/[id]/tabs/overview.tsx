"use client";

import { useEffect, useState, useTransition } from "react";
import { AlertTriangle, ArrowUpRight, Copy, Eye, EyeOff, Loader2, Pencil, RotateCw } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { AppRecord } from "@/lib/types";
import { gateway, type AppStatus } from "@/lib/gateway";
import { restartEndpoint, updateAutoscaler } from "../../actions";

export function OverviewTab({ app }: { app: AppRecord }) {
  return (
    <div className="space-y-4">
      <ProvisionErrorBanner appId={app.app_id} />
      <RequestPanel app={app} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <DetailCard app={app} />
        <ScaleStrategyCard app={app} />
      </div>

      <EngineArgsCard app={app} />
    </div>
  );
}

function ProvisionErrorBanner({ appId }: { appId: string }) {
  const [status, setStatus] = useState<AppStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      try {
        const res = await fetch(
          `/api/proxy/apps/${encodeURIComponent(appId)}/status`,
          { cache: "no-store" },
        );
        if (!res.ok) return;
        const data = (await res.json()) as AppStatus;
        if (!cancelled) setStatus(data);
      } catch {
        // best-effort; banner stays hidden on failure
      }
    }
    poll();
    const id = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [appId]);

  if (!status?.last_provision_error) return null;

  const cooldown = status.provision_cooldown_remaining_s;
  const at = status.last_provision_error_at
    ? new Date(status.last_provision_error_at * 1000)
    : null;
  const ago = at ? formatAgo(at) : null;

  return (
    <div className="flex items-start gap-3 rounded-md border border-red-500/30 bg-red-500/5 px-4 py-3 text-sm text-red-700 dark:text-red-300">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="flex-1 space-y-1">
        <div className="font-medium">
          Couldn't start a worker
          {ago ? <span className="text-xs font-normal opacity-75"> · {ago}</span> : null}
        </div>
        <div className="font-mono text-xs leading-relaxed opacity-90 break-words">
          {status.last_provision_error}
        </div>
        {cooldown > 0 ? (
          <div className="text-xs opacity-75">
            Auto-retry in {cooldown}s. Pick a different GPU / count if this combo isn't in stock.
          </div>
        ) : (
          <div className="text-xs opacity-75">
            The autoscaler will retry on the next request — or change GPU / count above.
          </div>
        )}
      </div>
    </div>
  );
}

function formatAgo(d: Date): string {
  const s = Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

function RequestPanel({ app }: { app: AppRecord }) {
  const [reveal, setReveal] = useState(false);
  const { token, loading: tokenLoading } = useApiToken();
  const endpoint = `${gateway.baseUrl}/run/${app.app_id}`;

  // The visible / copyable forms of every snippet. Visible may be masked;
  // copy always pastes the real key so the user gets a working command.
  const visibleToken = reveal && token ? token : token ? maskToken(token) : "YOUR_API_KEY";
  const realToken = token ?? "YOUR_API_KEY";

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <CardTitle className="text-sm font-medium">Run a job</CardTitle>
          <span className="text-xs text-muted-foreground">
            Autoscales to meet demand.
          </span>
        </div>
        {token ? (
          <Button variant="outline" size="xs" onClick={() => setReveal((v) => !v)}>
            {reveal ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
            {reveal ? "Hide" : "Reveal"} key
          </Button>
        ) : !tokenLoading ? (
          <Link
            href="/login?next=/serverless"
            className="text-xs text-primary hover:underline"
          >
            Sign in to use your key
          </Link>
        ) : null}
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="curl">
          <TabsList variant="line" className="bg-transparent">
            <TabsTrigger value="curl">cURL</TabsTrigger>
            <TabsTrigger value="curl-poll">cURL (poll)</TabsTrigger>
            <TabsTrigger value="openai">OpenAI client</TabsTrigger>
          </TabsList>

          <TabsContent value="curl" className="mt-3 space-y-3">
            <p className="text-sm text-muted-foreground">
              Async — returns a <code className="font-mono">request_id</code> immediately, then poll{" "}
              <code className="font-mono">/result/&#123;id&#125;</code> for the output.
            </p>
            <CodeBlock
              displayCode={curlSnippet(endpoint, visibleToken)}
              copyCode={curlSnippet(endpoint, realToken)}
            />
            <DocsLink />
          </TabsContent>

          <TabsContent value="curl-poll" className="mt-3 space-y-3">
            <p className="text-sm text-muted-foreground">
              Sync — gateway polls internally and returns the completion in one call (60s ceiling). No request_id juggling.
            </p>
            <CodeBlock
              displayCode={curlPollSnippet(app.app_id, visibleToken)}
              copyCode={curlPollSnippet(app.app_id, realToken)}
            />
            <DocsLink />
          </TabsContent>

          <TabsContent value="openai" className="mt-3 space-y-3">
            <p className="text-sm text-muted-foreground">
              vLLM exposes an OpenAI-compatible API. Point any OpenAI client at the gateway and use the endpoint name as the <code className="font-mono">model</code>.
            </p>
            <CodeBlock
              displayCode={openaiSnippet(app.app_id, visibleToken)}
              copyCode={openaiSnippet(app.app_id, realToken)}
            />
            <DocsLink />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

function useApiToken() {
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let abort = false;
    fetch("/api/auth/token", { cache: "no-store" })
      .then(async (r) => {
        if (abort) return;
        if (!r.ok) {
          setToken(null);
          return;
        }
        const body = (await r.json()) as { token?: string };
        setToken(body.token ?? null);
      })
      .catch(() => !abort && setToken(null))
      .finally(() => !abort && setLoading(false));
    return () => {
      abort = true;
    };
  }, []);
  return { token, loading };
}

function maskToken(t: string) {
  if (t.length <= 8) return "•".repeat(t.length);
  return `${t.slice(0, 4)}${"•".repeat(Math.max(8, t.length - 8))}${t.slice(-4)}`;
}

function CopyButton({ text }: { text: string }) {
  return (
    <Button
      variant="outline"
      size="icon-sm"
      onClick={() => {
        navigator.clipboard.writeText(text);
        toast.success("Copied", { duration: 3000 });
      }}
    >
      <Copy className="h-3.5 w-3.5" />
    </Button>
  );
}

function CodeBlock({
  displayCode,
  copyCode,
}: {
  displayCode: string;
  copyCode?: string;
}) {
  return (
    <div className="relative">
      <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed text-foreground scrollbar-thin">
        {displayCode}
      </pre>
      <div className="absolute right-2 top-2">
        <CopyButton text={copyCode ?? displayCode} />
      </div>
    </div>
  );
}

function DocsLink() {
  return (
    <a
      className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
      href="#"
    >
      Job operations documentation
      <ArrowUpRight className="h-3 w-3" />
    </a>
  );
}

function curlSnippet(endpoint: string, token: string) {
  return `curl -X POST '${endpoint}' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer ${token}' \\
  -d '{
    "prompt": "Hello, world",
    "max_tokens": 64
  }'`;
}

function curlPollSnippet(appId: string, token: string) {
  // Hits the OpenAI-compatible chat-completions endpoint, which polls
  // internally for up to 60s before returning the actual completion JSON.
  const base = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080";
  return `curl -X POST '${base}/v1/chat/completions' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer ${token}' \\
  -d '{
    "model": "${appId}",
    "messages": [{"role": "user", "content": "Hello, world"}]
  }'`;
}

function openaiSnippet(appId: string, token: string) {
  return `from openai import OpenAI

client = OpenAI(
    base_url="${process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8080"}/v1",
    api_key="${token}",
)

resp = client.chat.completions.create(
    model="${appId}",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)

for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)`;
}

function DetailCard({ app }: { app: AppRecord }) {
  return (
    <Card>
      <CardContent className="space-y-3 px-6 py-4 text-sm">
        <Row label="Endpoint ID" value={<code className="font-mono">{app.app_id}</code>} />
        <Row
          label="Created"
          value={new Date(app.created_at).toLocaleDateString("en-GB", {
            day: "2-digit",
            month: "short",
            year: "numeric",
          })}
        />
        <Row
          label="Framework"
          value={
            <span className="inline-flex items-center gap-1.5">
              <span className="flex h-5 w-5 items-center justify-center rounded border border-border bg-muted text-[10px] font-semibold text-muted-foreground">
                v
              </span>
              vLLM
            </span>
          }
        />
        <Row label="GPU count" value={`×${app.gpu_count ?? 1}`} />
        <Row label="GPU types" value={<span className="font-mono">{app.gpu}</span>} />
      </CardContent>
    </Card>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-border/40 pb-2 last:border-b-0 last:pb-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-foreground">{value}</span>
    </div>
  );
}

function ScaleStrategyCard({ app }: { app: AppRecord }) {
  const router = useRouter();
  const [editing, setEditing] = useState(false);
  const [maxInput, setMaxInput] = useState(String(app.autoscaler.max_containers));
  const [idleInput, setIdleInput] = useState(String(app.autoscaler.idle_timeout_s));
  const [pending, startTransition] = useTransition();

  useEffect(() => {
    setMaxInput(String(app.autoscaler.max_containers));
    setIdleInput(String(app.autoscaler.idle_timeout_s));
  }, [app.autoscaler.max_containers, app.autoscaler.idle_timeout_s]);

  const parsedMax = Number.parseInt(maxInput, 10);
  const parsedIdle = Number.parseInt(idleInput, 10);
  const maxInvalid =
    !/^\d+$/.test(maxInput.trim()) || !Number.isFinite(parsedMax) || parsedMax < 1 || parsedMax > 20;
  const idleInvalid =
    !/^\d+$/.test(idleInput.trim()) || !Number.isFinite(parsedIdle) || parsedIdle < 0 || parsedIdle > 86400;

  function save() {
    if (maxInvalid) {
      toast.error("Max workers must be an integer between 1 and 20.", { duration: 5000 });
      return;
    }
    if (idleInvalid) {
      toast.error("Idle timeout must be an integer 0–86400 seconds (0 = always-on, { duration: 5000 }).");
      return;
    }
    startTransition(async () => {
      const res = await updateAutoscaler(app.app_id, {
        max_containers: parsedMax,
        idle_timeout_s: parsedIdle,
      });
      if (!res.ok) {
        toast.error(res.error, { duration: 5000 });
        return;
      }
      toast.success("Scale strategy updated", { duration: 3000 });
      setEditing(false);
      router.refresh();
    });
  }

  function cancel() {
    setMaxInput(String(app.autoscaler.max_containers));
    setIdleInput(String(app.autoscaler.idle_timeout_s));
    setEditing(false);
  }

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-2">
        <CardTitle className="text-sm font-medium">Scale strategy</CardTitle>
        {!editing ? (
          <Button variant="outline" size="xs" onClick={() => setEditing(true)}>
            <Pencil className="h-3 w-3" />
            Edit
          </Button>
        ) : (
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="xs" onClick={cancel} disabled={pending}>
              Cancel
            </Button>
            <Button
              size="xs"
              onClick={save}
              disabled={pending || maxInvalid || idleInvalid}
            >
              {pending && <Loader2 className="h-3 w-3 animate-spin" />}
              Save
            </Button>
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <Row label="Active workers" value={<code className="font-mono">0</code>} />
        {editing ? (
          <>
            <EditRow label="Max workers">
              <Input
                type="text"
                inputMode="numeric"
                value={maxInput}
                onChange={(e) => setMaxInput(e.target.value)}
                placeholder="1–20"
                aria-invalid={maxInvalid}
                className="h-8 w-24 text-right font-mono"
                disabled={pending}
              />
            </EditRow>
            <EditRow label="Idle timeout (s)">
              <Input
                type="text"
                inputMode="numeric"
                value={idleInput}
                onChange={(e) => setIdleInput(e.target.value)}
                placeholder="0 = always-on"
                aria-invalid={idleInvalid}
                className="h-8 w-24 text-right font-mono"
                disabled={pending}
              />
            </EditRow>
          </>
        ) : (
          <>
            <Row
              label="Max workers"
              value={<code className="font-mono">{app.autoscaler.max_containers}</code>}
            />
            <Row
              label="Idle timeout"
              value={<code className="font-mono">{app.autoscaler.idle_timeout_s} s</code>}
            />
          </>
        )}
        <Row label="Auto scaling method" value="Queue delay" />
        <p className="rounded-md border border-border bg-muted/40 px-3 py-2 text-xs leading-relaxed text-muted-foreground">
          Scale up after <strong className="text-foreground">4</strong> seconds of queue delay.
          With zero workers initially, the first request adds one worker. Subsequent requests
          add workers only after waiting in the queue for 4 seconds.
        </p>
        <p className="text-xs text-muted-foreground">
          Assuming <strong className="text-foreground">1</strong> req/sec with{" "}
          <strong className="text-foreground">0.5</strong> s processing time.
        </p>
      </CardContent>
    </Card>
  );
}

function EngineArgsCard({ app }: { app: AppRecord }) {
  const router = useRouter();
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(app.vllm_args ?? "");
  const [pending, startTransition] = useTransition();
  const [restarting, startRestart] = useTransition();
  const [confirmRestart, setConfirmRestart] = useState(false);

  useEffect(() => {
    setValue(app.vllm_args ?? "");
  }, [app.vllm_args]);

  const tooLong = value.length > 2048;

  function save() {
    if (tooLong) {
      toast.error("Engine args too long (max 2048 chars, { duration: 5000 }).");
      return;
    }
    startTransition(async () => {
      const res = await updateAutoscaler(app.app_id, { vllm_args: value.trim() });
      if (!res.ok) {
        toast.error(res.error, { duration: 5000 });
        return;
      }
      toast.success("Engine args saved. Click Restart to apply now.", { duration: 3000 });
      setEditing(false);
      router.refresh();
    });
  }

  function restart() {
    startRestart(async () => {
      const res = await restartEndpoint(app.app_id);
      if (!res.ok) {
        toast.error(res.error, { duration: 5000 });
        return;
      }
      if (res.drained === 0) {
        toast.success("No live workers to restart — next cold start will use the latest config.", { duration: 3000 });
      } else {
        toast.success(`Draining ${res.drained} worker${res.drained === 1 ? "" : "s"} — autoscaler will respawn.`, { duration: 3000 });
      }
      setConfirmRestart(false);
      router.refresh();
    });
  }

  function cancel() {
    setValue(app.vllm_args ?? "");
    setEditing(false);
  }

  const display = (app.vllm_args ?? "").trim();
  return (
    <>
    <Dialog open={confirmRestart} onOpenChange={(open) => !restarting && setConfirmRestart(open)}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Restart workers?</DialogTitle>
          <DialogDescription>
            All running workers for this endpoint will be drained. In-flight requests
            finish; new ones spawn with the latest config.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="ghost" onClick={() => setConfirmRestart(false)} disabled={restarting}>
            Cancel
          </Button>
          <Button onClick={restart} disabled={restarting}>
            {restarting && <Loader2 className="h-4 w-4 animate-spin" />}
            Restart workers
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-2">
        <div className="flex flex-col gap-0.5">
          <CardTitle className="text-sm font-medium">vLLM engine args</CardTitle>
          <span className="text-xs text-muted-foreground">
            Appended to the <code className="font-mono">vllm serve</code> command on each worker
            boot. Changes apply on the next cold start.
          </span>
        </div>
        {!editing ? (
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="xs"
              onClick={() => setConfirmRestart(true)}
              disabled={restarting}
              title="Drain workers so the next cold start picks up the latest config"
            >
              {restarting ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <RotateCw className="h-3 w-3" />
              )}
              Restart workers
            </Button>
            <Button variant="outline" size="xs" onClick={() => setEditing(true)}>
              <Pencil className="h-3 w-3" />
              Edit
            </Button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="xs" onClick={cancel} disabled={pending}>
              Cancel
            </Button>
            <Button size="xs" onClick={save} disabled={pending || tooLong}>
              {pending && <Loader2 className="h-3 w-3 animate-spin" />}
              Save
            </Button>
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        {editing ? (
          <>
            <textarea
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder="--max-model-len 4096 --gpu-memory-utilization 0.9"
              rows={3}
              aria-invalid={tooLong}
              className="w-full rounded-md border border-input bg-background px-3 py-2 font-mono text-xs shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/30 aria-invalid:border-destructive"
              disabled={pending}
            />
            <p className="text-xs text-muted-foreground">
              See{" "}
              <a
                href="https://docs.vllm.ai/en/stable/configuration/engine_args/"
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-foreground"
              >
                vLLM engine args
              </a>
              . {value.length}/2048 chars.
            </p>
          </>
        ) : display ? (
          <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed text-foreground scrollbar-thin">
            {display}
          </pre>
        ) : (
          <p className="rounded-md border border-dashed border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
            No custom args — vLLM uses its built-in defaults.
          </p>
        )}
      </CardContent>
    </Card>
    </>
  );
}

function EditRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-border/40 pb-2 last:border-b-0 last:pb-0">
      <Label className="text-muted-foreground">{label}</Label>
      {children}
    </div>
  );
}

