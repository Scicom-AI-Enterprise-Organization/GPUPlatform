"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
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
import type { StorageKind } from "@/lib/types";

type TestState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "ok"; message: string }
  | { status: "fail"; message: string };

export function StorageForm() {
  const router = useRouter();

  const [kind, setKind] = useState<StorageKind>("s3");
  const [name, setName] = useState("");
  const [notes, setNotes] = useState("");

  // s3
  const [bucket, setBucket] = useState("");
  const [prefix, setPrefix] = useState("");
  const [region, setRegion] = useState("");
  const [endpoint, setEndpoint] = useState("");
  const [accessKeyId, setAccessKeyId] = useState("");
  const [secretAccessKey, setSecretAccessKey] = useState("");

  // huggingface — token comes from a global secret (default) or a pasted token.
  const [hfSource, setHfSource] = useState<"secret" | "paste">("secret");
  const [hfToken, setHfToken] = useState("");
  const [hfTokenSecret, setHfTokenSecret] = useState("");
  const [secretKeys, setSecretKeys] = useState<string[]>([]);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [test, setTest] = useState<TestState>({ status: "idle" });

  // Global secrets (admin Secrets) the HF token can reference — keys only.
  useEffect(() => {
    let cancel = false;
    fetch("/api/proxy/v1/global-env", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : []))
      .then((rows: { key: string }[]) => {
        if (!cancel && Array.isArray(rows)) setSecretKeys(rows.map((r) => r.key));
      })
      .catch(() => {
        /* admins only; non-admins just won't see the picker */
      });
    return () => {
      cancel = true;
    };
  }, []);

  // Editing a tested field invalidates a prior pass, so Create re-disables
  // until the user re-tests. No-op when already idle to avoid render churn.
  const invalidateTest = () =>
    setTest((t) => (t.status === "idle" ? t : { status: "idle" }));

  const validate = (): string | null => {
    if (!name.trim()) return "Name is required.";
    if (kind === "s3") {
      if (!bucket.trim()) return "Bucket is required.";
      const hasOne = accessKeyId.trim() || secretAccessKey.trim();
      const hasBoth = accessKeyId.trim() && secretAccessKey.trim();
      if (hasOne && !hasBoth) {
        return "Provide both Access key ID and Secret access key, or leave both blank.";
      }
    }
    return null;
  };

  const onTest = async () => {
    setError(null);
    const err = validate();
    if (err) {
      setTest({ status: "fail", message: err });
      return;
    }
    setTest({ status: "running" });
    try {
      const r = await gateway.testStorage(
        kind === "s3"
          ? {
              kind,
              bucket: bucket.trim(),
              region: region.trim() || null,
              endpoint: endpoint.trim() || null,
              access_key_id: accessKeyId.trim() || null,
              secret_access_key: secretAccessKey.trim() || null,
            }
          : hfSource === "secret"
            ? { kind, hf_token_secret: hfTokenSecret || null }
            : { kind, hf_token: hfToken.trim() || null },
      );
      setTest(r.ok ? { status: "ok", message: r.message } : { status: "fail", message: r.message });
    } catch (e) {
      setTest({ status: "fail", message: e instanceof Error ? e.message : String(e) });
    }
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
      await gateway.createStorage({
        name: name.trim(),
        kind,
        notes: notes.trim() || null,
        ...(kind === "s3"
          ? {
              bucket: bucket.trim(),
              prefix: prefix.trim() || null,
              region: region.trim() || null,
              endpoint: endpoint.trim() || null,
              access_key_id: accessKeyId.trim() || null,
              secret_access_key: secretAccessKey.trim() || null,
            }
          : hfSource === "secret"
            ? { hf_token_secret: hfTokenSecret || null }
            : { hf_token: hfToken.trim() || null }),
      });
      router.push("/storage");
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={onSubmit} className="flex flex-col gap-5">
      <section className="rounded-lg border border-border bg-card p-5">
        <div className="mb-4">
          <h2 className="text-base font-medium">Storage backend</h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {kind === "s3"
              ? "An S3 (or S3-compatible: R2, MinIO) bucket the platform writes to."
              : "A HuggingFace token holder for pushing datasets / models to the Hub."}
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <Label htmlFor="storage-kind">Kind</Label>
            <Select
              value={kind}
              onValueChange={(v) => {
                setKind(v as StorageKind);
                invalidateTest();
              }}
            >
              <SelectTrigger id="storage-kind" className="mt-1.5">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="s3">S3 (or S3-compatible)</SelectItem>
                <SelectItem value="huggingface">HuggingFace</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label htmlFor="storage-name">Name</Label>
            <Input
              id="storage-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={kind === "s3" ? "e.g. prod-s3" : "e.g. hf-scicom"}
              className="mt-1.5"
            />
          </div>
        </div>
      </section>

      {kind === "s3" && (
        <section className="rounded-lg border border-border bg-card p-5">
          <h2 className="mb-4 text-base font-medium">Bucket</h2>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div>
              <Label htmlFor="s3-bucket">Bucket</Label>
              <Input
                id="s3-bucket"
                value={bucket}
                onChange={(e) => {
                  setBucket(e.target.value);
                  invalidateTest();
                }}
                placeholder="gpuplatform"
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="s3-prefix">Prefix (optional)</Label>
              <Input
                id="s3-prefix"
                value={prefix}
                onChange={(e) => setPrefix(e.target.value)}
                placeholder="datasets"
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="s3-region">Region (optional)</Label>
              <Input
                id="s3-region"
                value={region}
                onChange={(e) => {
                  setRegion(e.target.value);
                  invalidateTest();
                }}
                placeholder="ap-southeast-5"
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="s3-endpoint">Endpoint (optional)</Label>
              <Input
                id="s3-endpoint"
                value={endpoint}
                onChange={(e) => {
                  setEndpoint(e.target.value);
                  invalidateTest();
                }}
                placeholder="https://<account>.r2.cloudflarestorage.com"
                className="mt-1.5 font-mono text-xs"
              />
              <p className="mt-1 text-xs text-muted-foreground">
                Set for R2, MinIO, or any S3-compatible API.
              </p>
            </div>
          </div>

          <div className="mt-4 rounded-md border border-border p-3">
            <div className="mb-1 text-sm font-medium">Credentials</div>
            <p className="mb-3 text-xs text-muted-foreground">
              Optional — leave both blank to fall back to{" "}
              <span className="font-mono">AWS_ACCESS_KEY_ID</span> /{" "}
              <span className="font-mono">AWS_SECRET_ACCESS_KEY</span> on the gateway.
              Stored encrypted at rest with Fernet.
            </p>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div>
                <Label htmlFor="s3-akid">Access key ID</Label>
                <Input
                  id="s3-akid"
                  type="password"
                  autoComplete="off"
                  value={accessKeyId}
                  onChange={(e) => {
                    setAccessKeyId(e.target.value);
                    invalidateTest();
                  }}
                  className="mt-1.5 font-mono text-xs"
                />
              </div>
              <div>
                <Label htmlFor="s3-sak">Secret access key</Label>
                <Input
                  id="s3-sak"
                  type="password"
                  autoComplete="off"
                  value={secretAccessKey}
                  onChange={(e) => {
                    setSecretAccessKey(e.target.value);
                    invalidateTest();
                  }}
                  className="mt-1.5 font-mono text-xs"
                />
              </div>
            </div>
          </div>
        </section>
      )}

      {kind === "huggingface" && (
        <section className="rounded-lg border border-border bg-card p-5">
          <div className="mb-1 text-sm font-medium">HuggingFace API token</div>
          <p className="mb-3 text-xs text-muted-foreground">
            Use a global secret (managed once under{" "}
            <a href="/admin/secrets" className="underline">Secrets</a>, shared, rotate without touching this storage)
            or paste a token (stored encrypted here). Leave blank to fall back to{" "}
            <span className="font-mono">HF_TOKEN</span> on the gateway.
          </p>

          <div className="mb-3 inline-flex rounded-md border border-border p-0.5 text-xs">
            {(["secret", "paste"] as const).map((src) => (
              <button
                key={src}
                type="button"
                onClick={() => {
                  setHfSource(src);
                  invalidateTest();
                }}
                className={
                  "rounded px-2.5 py-1 transition-colors " +
                  (hfSource === src ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")
                }
              >
                {src === "secret" ? "Global secret" : "Paste a token"}
              </button>
            ))}
          </div>

          {hfSource === "secret" ? (
            secretKeys.length > 0 ? (
              <>
                <Label htmlFor="hf-secret">Global secret</Label>
                <Select
                  value={hfTokenSecret}
                  onValueChange={(v) => {
                    setHfTokenSecret(v);
                    invalidateTest();
                  }}
                >
                  <SelectTrigger id="hf-secret" className="mt-1.5">
                    <SelectValue placeholder="Select a secret (e.g. HF_TOKEN)" />
                  </SelectTrigger>
                  <SelectContent>
                    {secretKeys.map((k) => (
                      <SelectItem key={k} value={k} className="font-mono text-xs">
                        {k}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="mt-1.5 text-xs text-muted-foreground">
                  Resolved from Secrets at use-time — rotate it there and this storage picks it up automatically.
                </p>
              </>
            ) : (
              <p className="text-xs text-muted-foreground">
                No global secrets yet. Add one under{" "}
                <a href="/admin/secrets" className="underline">Secrets</a> (e.g. <span className="font-mono">HF_TOKEN</span>),
                then pick it here — or switch to <span className="font-medium">Paste a token</span>.
              </p>
            )
          ) : (
            <>
              <Label htmlFor="hf-token">Token</Label>
              <Input
                id="hf-token"
                type="password"
                autoComplete="off"
                value={hfToken}
                onChange={(e) => {
                  setHfToken(e.target.value);
                  invalidateTest();
                }}
                placeholder="hf_..."
                className="mt-1.5 font-mono text-xs"
              />
              <p className="mt-1.5 text-xs text-muted-foreground">
                Stored encrypted at rest with Fernet. Generate one at{" "}
                <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" className="underline">
                  huggingface.co/settings/tokens
                </a>
                .
              </p>
            </>
          )}
        </section>
      )}

      <section className="rounded-lg border border-border bg-card p-5">
        <Label htmlFor="storage-notes">Notes (optional)</Label>
        <Textarea
          id="storage-notes"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={2}
          className="mt-1.5"
        />
      </section>

      <div className="flex items-center gap-3">
        {error && <span className="text-sm text-destructive">{error}</span>}
        <div className="ml-auto flex items-center gap-3">
          {test.status === "ok" && (
            <span className="text-sm text-emerald-600 dark:text-emerald-400">✓ {test.message}</span>
          )}
          {test.status === "fail" && (
            <span className="text-right text-sm text-destructive">✕ {test.message}</span>
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
            {submitting ? "Creating…" : "Create storage"}
          </Button>
        </div>
      </div>
    </form>
  );
}
