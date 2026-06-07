"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { gateway } from "@/lib/gateway";
import type { ProviderKind } from "@/lib/types";

type TestState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "ok"; message: string; gpus: string[]; gpu_count: number }
  | { status: "fail"; message: string };

export function ProviderForm() {
  const router = useRouter();
  const fileRef = useRef<HTMLInputElement>(null);

  const [kind, setKind] = useState<ProviderKind>("vm");
  const [name, setName] = useState("");

  // VM-only fields
  const [host, setHost] = useState("");
  const [port, setPort] = useState("22");
  const [user, setUser] = useState("root");
  const [privateKey, setPrivateKey] = useState("");

  // API-key kinds (runpod / pi)
  const [apiKey, setApiKey] = useState("");

  const [test, setTest] = useState<TestState>({ status: "idle" });
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const isApiKind = kind === "runpod" || kind === "pi";

  // Any edit to a tested field invalidates a prior pass, so the user must
  // re-test before the Create button re-enables. No-op when already idle to
  // avoid re-render churn on every keystroke.
  const invalidateTest = () =>
    setTest((t) => (t.status === "idle" ? t : { status: "idle" }));

  const validate = (): string | null => {
    if (!name.trim()) return "Name is required.";
    if (kind === "vm") {
      if (!host.trim()) return "Host is required.";
      const p = Number(port);
      if (!Number.isFinite(p) || p < 1 || p > 65535) return "Port must be 1..65535.";
      if (!user.trim()) return "SSH user is required.";
      if (!privateKey.trim()) return "Private key is required.";
    } else {
      if (!apiKey.trim()) return "API key is required.";
    }
    return null;
  };

  const onPickFile = () => fileRef.current?.click();

  const onFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const text = await f.text();
    setPrivateKey(text);
    invalidateTest();
    e.target.value = "";
  };

  const onTest = async () => {
    setSubmitError(null);
    const err = validate();
    if (err) {
      setTest({ status: "fail", message: err });
      return;
    }
    setTest({ status: "running" });
    try {
      const r = await gateway.testProvider(
        kind === "vm"
          ? {
              kind: "vm",
              vm: {
                host: host.trim(),
                port: Number(port),
                user: user.trim(),
                private_key: privateKey,
              },
            }
          : {
              kind,
              api: { api_key: apiKey.trim() },
            },
      );
      if (r.ok) {
        setTest({ status: "ok", message: r.message, gpus: r.gpus, gpu_count: r.gpu_count });
      } else {
        setTest({ status: "fail", message: r.message });
      }
    } catch (e) {
      setTest({ status: "fail", message: e instanceof Error ? e.message : String(e) });
    }
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitError(null);
    const err = validate();
    if (err) {
      setSubmitError(err);
      return;
    }
    setSubmitting(true);
    try {
      await gateway.createProvider(
        kind === "vm"
          ? {
              name: name.trim(),
              kind: "vm",
              vm: {
                host: host.trim(),
                port: Number(port),
                user: user.trim(),
                private_key: privateKey,
              },
            }
          : {
              name: name.trim(),
              kind,
              api: { api_key: apiKey.trim() },
            },
      );
      router.push("/providers");
      router.refresh();
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={onSubmit} className="flex flex-col gap-5">
      <section className="rounded-lg border border-border bg-card p-5">
        <div className="mb-4">
          <h2 className="text-base font-medium">Provider</h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            VM is bare-metal SSH. RunPod and Prime Intellect connect with an
            API key from your account dashboard.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <Label htmlFor="provider-kind" className="text-xs uppercase tracking-wide text-muted-foreground">Type</Label>
            <Select
              value={kind}
              onValueChange={(v) => {
                setKind(v as ProviderKind);
                setTest({ status: "idle" });
              }}
            >
              <SelectTrigger id="provider-kind" className="mt-1.5">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="vm">SSH</SelectItem>
                <SelectItem value="runpod">RunPod (API key)</SelectItem>
                <SelectItem value="pi">Prime Intellect (API key)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label htmlFor="provider-name" className="text-xs uppercase tracking-wide text-muted-foreground">Name</Label>
            <Input
              id="provider-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={
                kind === "vm" ? "e.g. lab-rig-01" : "e.g. my-runpod-account"
              }
              className="mt-1.5"
            />
          </div>
        </div>
      </section>

      {kind === "vm" && (
        <section className="rounded-lg border border-border bg-card p-5">
          <div className="mb-4">
            <h2 className="text-base font-medium">SSH access</h2>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Paste the <span className="font-medium text-foreground">private</span> key
              that authenticates as <span className="font-mono">{user || "root"}</span> on this VM.
              We&apos;ll use it to SSH in and run <span className="font-mono">nvidia-smi</span>.
              Stored encrypted at rest with Fernet.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr_120px_1fr]">
            <div>
              <Label htmlFor="vm-host" className="text-xs uppercase tracking-wide text-muted-foreground">Host</Label>
              <Input
                id="vm-host"
                value={host}
                onChange={(e) => {
                  setHost(e.target.value);
                  invalidateTest();
                }}
                placeholder="10.0.0.5 or vm.example.com"
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="vm-port" className="text-xs uppercase tracking-wide text-muted-foreground">Port</Label>
              <Input
                id="vm-port"
                value={port}
                onChange={(e) => {
                  setPort(e.target.value);
                  invalidateTest();
                }}
                inputMode="numeric"
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="vm-user" className="text-xs uppercase tracking-wide text-muted-foreground">User</Label>
              <Input
                id="vm-user"
                value={user}
                onChange={(e) => {
                  setUser(e.target.value);
                  invalidateTest();
                }}
                placeholder="root"
                className="mt-1.5"
              />
            </div>
          </div>

          <div className="mt-4">
            <div className="flex items-end justify-between">
              <Label htmlFor="vm-key" className="text-xs uppercase tracking-wide text-muted-foreground">Private key (PEM / OpenSSH)</Label>
              <Button type="button" variant="outline" size="sm" onClick={onPickFile}>
                <Upload className="h-3.5 w-3.5" />
                Upload key file
              </Button>
              <input
                ref={fileRef}
                type="file"
                accept=".pem,.key,.txt,*"
                className="hidden"
                onChange={onFileChange}
              />
            </div>
            <Textarea
              id="vm-key"
              value={privateKey}
              onChange={(e) => {
                setPrivateKey(e.target.value);
                invalidateTest();
              }}
              placeholder={"-----BEGIN OPENSSH PRIVATE KEY-----\n..."}
              rows={8}
              className="mt-1.5 font-mono text-xs"
            />
            <p className="mt-1 text-xs text-muted-foreground">
              Make sure the matching public half is already in <span className="font-mono">~/.ssh/authorized_keys</span> on the VM — we never need or store the public key here.
            </p>
          </div>
        </section>
      )}

      {isApiKind && (
        <section className="rounded-lg border border-border bg-card p-5">
          <div className="mb-4">
            <h2 className="text-base font-medium">API key</h2>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {kind === "runpod"
                ? "Paste a key from runpod.io → Settings → API Keys. Stored encrypted at rest with Fernet. Test validates it by listing 1 pod."
                : "Paste a bearer token from app.primeintellect.ai. Stored encrypted at rest with Fernet. Test validates it by listing 1 pod."}
            </p>
          </div>

          <div className="grid grid-cols-1 gap-4">
            <div>
              <Label htmlFor="api-key" className="text-xs uppercase tracking-wide text-muted-foreground">API key</Label>
              <Input
                id="api-key"
                type="password"
                value={apiKey}
                onChange={(e) => {
                  setApiKey(e.target.value);
                  invalidateTest();
                }}
                placeholder={kind === "runpod" ? "rpa_..." : "pi_..."}
                className="mt-1.5 font-mono text-xs"
                autoComplete="off"
              />
            </div>
            <p className="text-xs text-muted-foreground">
              Community vs. Secure tier is picked per workload — this key
              works on both. On save we generate an ed25519 SSH keypair for
              this provider and inject the public half into spawned pods
              automatically — no upload step. The private key never leaves
              the gateway.
            </p>
          </div>
        </section>
      )}

      <div className="flex items-center gap-3">
        {submitError && (
          <span className="text-sm text-destructive">{submitError}</span>
        )}
        <div className="ml-auto flex items-center gap-3">
          {test.status === "ok" && (
            <span className="text-sm text-emerald-600 dark:text-emerald-400">
              ✓ {test.message}
              {test.gpus.length > 0 && (
                <span className="ml-1 font-mono text-xs text-muted-foreground">
                  ({test.gpus.slice(0, 2).join(", ")}{test.gpus.length > 2 ? `, …+${test.gpus.length - 2}` : ""})
                </span>
              )}
            </span>
          )}
          {test.status === "fail" && (
            <span className="text-sm text-destructive">✕ {test.message}</span>
          )}
          <Button
            type="button"
            variant="outline"
            onClick={onTest}
            disabled={test.status === "running" || submitting}
          >
            {test.status === "running" && <Loader2 className="h-4 w-4 animate-spin" />}
            {test.status === "running" ? "Testing…" : "Test"}
          </Button>
          <Button
            type="submit"
            disabled={submitting || test.status !== "ok"}
            title={test.status !== "ok" ? "Pass the connection test first" : undefined}
          >
            {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
            {submitting ? "Creating…" : "Create provider"}
          </Button>
        </div>
      </div>
    </form>
  );
}
