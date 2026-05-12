"use client";

import { useEffect, useState } from "react";
import { Copy, Eye, EyeOff, Loader2, RotateCw } from "lucide-react";
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
import { gateway } from "@/lib/gateway";
import { cn } from "@/lib/utils";

type TokenResponse = {
  token: string;
  username: string | null;
};

export function ApiKeyPanel() {
  const [data, setData] = useState<TokenResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [reveal, setReveal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [rotating, setRotating] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);

  useEffect(() => {
    let abort = false;
    fetch("/api/auth/token", { cache: "no-store" })
      .then(async (r) => {
        const body = await r.json();
        if (abort) return;
        if (!r.ok) {
          setErr(body?.error ?? r.statusText);
          return;
        }
        setData(body);
      })
      .catch((e) => !abort && setErr(e instanceof Error ? e.message : String(e)))
      .finally(() => !abort && setLoading(false));
    return () => {
      abort = true;
    };
  }, []);

  function copy(text: string, label: string) {
    navigator.clipboard.writeText(text);
    toast.success(`${label} copied`, { duration: 3000 });
  }

  async function rotate() {
    if (!data) return;
    setConfirmOpen(false);
    setRotating(true);
    try {
      // "Rotate" = invalidate the current session and create a new one. The
      // simplest path is logout then re-login; the user will need to re-enter
      // their password. We don't store passwords, so the practical UX here
      // is: bounce them to /login with a "key rotated" flash.
      await fetch("/api/auth/logout", { method: "POST" });
      window.location.href = "/login?next=/api-keys";
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e), { duration: 5000 });
    } finally {
      setRotating(false);
    }
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center px-6 py-12">
          <Loader2 className="h-4 w-4 animate-spin" />
        </CardContent>
      </Card>
    );
  }
  if (err) {
    return (
      <Card>
        <CardContent className="px-6 py-6 text-sm text-destructive">
          {err}
        </CardContent>
      </Card>
    );
  }
  if (!data) return null;

  const masked = data.token.replace(/.(?=.{4})/g, "•");
  const display = reveal ? data.token : masked;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <div>
            <CardTitle className="text-sm font-medium">Default key</CardTitle>
            <p className="text-xs text-muted-foreground">
              Tied to your current session
              {data.username && (
                <>
                  {" "}
                  (<span className="font-mono">{data.username}</span>)
                </>
              )}
              . Logging out invalidates it.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setReveal((v) => !v)}
            >
              {reveal ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              {reveal ? "Hide" : "Reveal"}
            </Button>
            <Button variant="outline" size="sm" onClick={() => copy(data.token, "Token")}>
              <Copy className="h-4 w-4" />
              Copy
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setConfirmOpen(true)}
              disabled={rotating}
            >
              {rotating ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RotateCw className="h-4 w-4" />
              )}
              Rotate
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <pre
            className={cn(
              "overflow-x-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-sm scrollbar-thin",
              !reveal && "select-none",
            )}
          >
            {display}
          </pre>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Use it</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <Snippet
            label="cURL"
            code={`curl -X POST '${gateway.baseUrl}/v1/chat/completions' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer YOUR_API_KEY' \\
  -d '{
    "model": "<your-endpoint-name>",
    "messages": [{"role": "user", "content": "Hello"}]
  }'`}
          />
          <Snippet
            label="Python (OpenAI SDK)"
            code={`from openai import OpenAI

client = OpenAI(
    base_url="${gateway.baseUrl}/v1",
    api_key="YOUR_API_KEY",
)

resp = client.chat.completions.create(
    model="<your-endpoint-name>",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)`}
          />
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground">
        Multi-key support (named keys, separate revoke, last-used timestamps) needs a small gateway
        change. Today, your bearer token <em>is</em> your session — rotate by signing out and back in.
      </p>

      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Rotate API key?</DialogTitle>
            <DialogDescription>
              The current key stops working immediately. Anything still using it
              (CI jobs, scripts, SDK clients) will start receiving 401 until you
              update them with the new token.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmOpen(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={rotate}>
              <RotateCw className="h-4 w-4" />
              Rotate now
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function Snippet({ label, code }: { label: string; code: string }) {
  return (
    <div>
      <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="relative">
        <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-xs leading-relaxed scrollbar-thin">
          {code}
        </pre>
        <Button
          variant="outline"
          size="icon-xs"
          className="absolute right-2 top-2"
          onClick={() => {
            navigator.clipboard.writeText(code);
            toast.success("Snippet copied", { duration: 3000 });
          }}
        >
          <Copy className="h-3 w-3" />
        </Button>
      </div>
    </div>
  );
}
